"""Run matched SO-101 A* and SmolVLA checkpoint comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inference_protocol import SocketActionChunkPolicy
from .so101 import (
    SO101AStarController,
    SO101LearnedExecutor,
    SO101Simulation,
    resolve_so101_task,
    save_comparison_chart,
)


PROMPTS = (
    "Move the apple to the target",
    "Carry the Coke can to the destination",
    "Transfer the pen into the goal",
    "Place the apple to the right of the Coke can",
    "Put the pen in front of the apple",
)
SCENE_SEEDS = (10009, 10019, 10020, 10039, 10049)


def run_comparison(
    episodes: int,
    checkpoint: str,
    host: str,
    port: int,
    chart_dir: Path,
    max_chunks: int = 20,
) -> dict:
    if not 1 <= episodes <= len(SCENE_SEEDS):
        raise ValueError(f"episodes must be between 1 and {len(SCENE_SEEDS)}")
    policy = SocketActionChunkPolicy(host, port, timeout=180.0)
    results = []
    try:
        for index in range(episodes):
            seed = SCENE_SEEDS[index]
            prompt = PROMPTS[index % len(PROMPTS)]
            with SO101Simulation(training_seed=seed) as simulation:
                object_name, destination = resolve_so101_task(simulation, prompt)
                astar = SO101AStarController(simulation)
                try:
                    metrics = astar.move_object(object_name, destination)
                    astar_result = {
                        "controller": "astar", "task_success": True,
                        "failure_reason": None,
                        "trajectory": astar.samples,
                        "joint_trajectory": astar.joint_samples,
                        **metrics,
                    }
                except ValueError as error:
                    astar_result = {
                        "controller": "astar", "task_success": False,
                        "failure_reason": str(error),
                        "collision_count": float(simulation.collision_count),
                    }
                save_comparison_chart(
                    chart_dir / f"episode_{index:03d}_astar.png",
                    astar.samples, astar.joint_samples, simulation.obstacle_bounds,
                    f"Episode {index}: SO-101 A*",
                )
            with SO101Simulation(training_seed=seed) as simulation:
                object_name, destination = resolve_so101_task(simulation, prompt)
                learned = SO101LearnedExecutor(simulation, policy, checkpoint)
                learned_result = learned.execute(
                    prompt, object_name, destination, max_chunks
                )
                save_comparison_chart(
                    chart_dir / f"episode_{index:03d}_smolvla.png",
                    learned.samples, learned.joint_samples, simulation.obstacle_bounds,
                    f"Episode {index}: SmolVLA",
                )
            results.append({
                "episode": index, "seed": seed, "prompt": prompt,
                "object": object_name, "destination": list(destination),
                "astar": astar_result, "smolvla": learned_result,
            })
    finally:
        policy.close()
    summary = summarize(results)
    save_summary_chart(chart_dir / "summary.png", summary)
    return {"checkpoint": checkpoint, "episodes": episodes, "summary": summary, "results": results}


def summarize(results: list[dict]) -> dict:
    output = {}
    for name in ("astar", "smolvla"):
        rows = [row[name] for row in results]
        successful = [row for row in rows if row.get("task_success")]
        endpoint_errors = [
            float(row["endpoint_error_m"])
            for row in rows
            if row.get("endpoint_error_m") is not None
        ]
        output[name] = {
            "task_success_rate": sum(bool(row.get("task_success")) for row in rows) / len(rows),
            "collision_free_rate": sum(float(row.get("collision_count", 0)) == 0 for row in rows) / len(rows),
            "mean_endpoint_error_m": float(np.mean(endpoint_errors)) if endpoint_errors else None,
            "mean_path_length_m": float(np.mean([row.get("path_length_m", 0) for row in successful])) if successful else None,
            "mean_max_jerk_m_s3": float(np.mean([row.get("max_jerk_m_s3", 0) for row in successful])) if successful else None,
            "mean_latency_s": float(np.mean([
                row.get("inference_latency_s", row.get("planning_time_s", 0)) for row in rows
            ])),
        }
    return output


def save_summary_chart(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 730), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "SO-101 A* vs SmolVLA", fill="black")
    metrics = (
        ("task_success_rate", "Task success", 1.0),
        ("collision_free_rate", "Collision-free", 1.0),
        ("mean_endpoint_error_m", "Endpoint error (m)", None),
        ("mean_path_length_m", "Path length (m)", None),
        ("mean_max_jerk_m_s3", "Maximum jerk (m/s^3)", None),
        ("mean_latency_s", "Planning / inference time (s)", None),
    )
    colors = {"astar": (45, 110, 200), "smolvla": (155, 70, 180)}
    for row, (key, label, fixed_max) in enumerate(metrics):
        top = 70 + row * 105
        values = [float(summary[name].get(key) or 0) for name in ("astar", "smolvla")]
        maximum = fixed_max or max(values + [1e-9])
        draw.text((30, top), label, fill="black")
        for column, name in enumerate(("astar", "smolvla")):
            x = 250 + column * 310
            width = 240 * values[column] / maximum
            draw.rectangle((x, top, x + width, top + 28), fill=colors[name])
            draw.text((x, top + 34), f"{name}: {values[column]:.4g}", fill=colors[name])
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5478)
    parser.add_argument("--charts", type=Path, default=Path("artifacts/so101-comparison"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=20)
    args = parser.parse_args()
    result = run_comparison(
        args.episodes, args.checkpoint, args.inference_host, args.inference_port,
        args.charts, args.max_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

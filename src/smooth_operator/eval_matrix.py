"""Scope 2: Batch Evaluation Matrix Pipeline over Golden Data SceneSpecs & Prompt Lists.

Given:
- A Golden Evaluation Dataset (or prompt list + seed bank with embedded PyBullet SceneSpecs)
- A Model (`astar_oracle`, `smolvla`, `fast_vla`)
- A list of InverseSpectrumModule schemas (`direct`, `fast_idct`, `band_idct:0.35`, `ifft:0.35`,
  `idwt_haar`, `idwt_db4`, `custom_hybrid`)

Executes deterministic PyBullet rollouts across all episodes and schemas, and exports:
- `eval_metadata.json` (complete structured evaluation matrix + per-episode results)
- `eval_summary.csv` (tabular matrix ready for LaTeX / Mid-Term presentation)
- Per-episode 4-panel spectral & trajectory diagnostic charts
- `summary_matrix.png` (aggregate comparison chart across all schemas)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .inference_protocol import SocketActionChunkPolicy
from .metrics import save_spectral_diagnostic_chart
from .pipeline import (
    ModularExecutionPipeline,
    PrecomputedChunkPolicy,
    SceneSpec,
    apply_scene_spec,
    generate_oracle_action_sequence,
    resolve_scene_from_prompt,
)
from .planning import WORKSPACE as SCARA_WORKSPACE
from .simulation import ManipulatorSimulation
from .so101 import SO101Simulation
from .spectrum import create_spectrum_module


DEFAULT_EVAL_PROMPTS = (
    "Move the apple to the target",
    "Carry the Coke can to the destination",
    "Transfer the pen into the goal",
    "Place the apple to the right of the Coke can",
    "Put the pen in front of the apple",
)
DEFAULT_EVAL_SEEDS = (9, 19, 20, 29, 39, 49, 59, 69, 79, 89)


def load_eval_scene_specs(
    golden_data: Optional[Path],
    split: str = "evaluation",
    episodes: int = 5,
    robot: str = "so101",
    prompts: Sequence[str] = DEFAULT_EVAL_PROMPTS,
    seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
) -> list[tuple[str, SceneSpec, Optional[np.ndarray]]]:
    """Load reproducible SceneSpecs from a Golden Dataset directory or synthesize from seeds/prompts."""
    loaded: list[tuple[str, SceneSpec, Optional[np.ndarray]]] = []
    if golden_data is not None and golden_data.exists():
        ep_dirs = sorted(golden_data.glob("episode_*"))
        for ep_dir in ep_dirs:
            meta_file = ep_dir / "metadata.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text())
            if split and meta.get("split") != split:
                continue
            spec = SceneSpec.from_episode_metadata(meta)
            saved_actions: Optional[np.ndarray] = None
            frames_file = ep_dir / "frames.npz"
            if frames_file.exists():
                with np.load(frames_file) as archive:
                    if "actions" in archive.files:
                        saved_actions = archive["actions"].astype(float)
            loaded.append((ep_dir.name, spec, saved_actions))
            if len(loaded) >= episodes:
                break
        if loaded:
            return loaded

    # Fallback: build deterministic Golden SceneSpecs from the seed & prompt bank
    sim_cls = SO101Simulation if robot == "so101" else ManipulatorSimulation
    seed_idx = 0
    while len(loaded) < episodes and seed_idx < max(episodes * 12, len(seeds)):
        seed = seeds[seed_idx] if seed_idx < len(seeds) else (9 + seed_idx * 10)
        prompt = prompts[len(loaded) % len(prompts)]
        seed_idx += 1
        try:
            with sim_cls(gui=False, training_seed=seed) as sim:
                spec = resolve_scene_from_prompt(sim, prompt, seed)
            oracle_actions, oracle_samples, oracle_len = generate_oracle_action_sequence(spec)
            if len(oracle_actions) == 0:
                continue
            # Verify the oracle rollout itself is collision-free before adding to Golden Eval bank
            with sim_cls(gui=False, training_seed=seed) as sim:
                apply_scene_spec(sim, spec)
                check_res = ModularExecutionPipeline(
                    simulation=sim,
                    policy=PrecomputedChunkPolicy(oracle_actions, chunk_size=10),
                    spectrum_module=create_spectrum_module("direct"),
                    model_name="astar_oracle",
                ).execute_task(spec)
                if not check_res["task_success"]:
                    continue
            spec.oracle_path_length_m = oracle_len
            spec.oracle_trajectory = [list(map(float, row)) for row in oracle_samples]
            loaded.append((f"episode_{len(loaded):06d}", spec, oracle_actions))
        except ValueError:
            continue
    if not loaded:
        raise RuntimeError("Could not load or generate any valid evaluation SceneSpecs")
    return loaded


def aggregate_schema_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute Mean, Std, and Rate statistics across all episodes for a single schema configuration."""
    if not rows:
        return {}
    n = len(rows)
    numeric_keys = (
        "endpoint_error_m",
        "collision_count",
        "path_length_m",
        "path_efficiency_ratio",
        "joint_rms_jerk_rad_s3",
        "joint_max_jerk_rad_s3",
        "ee_rms_jerk_m_s3",
        "object_max_jerk_m_s3",
        "object_rms_jerk_m_s3",
        "chunk_boundary_jump_rad_s",
        "max_chunk_boundary_jump_rad_s",
        "dimensionless_jerk_ldlj",
        "sparc_smoothness",
        "high_freq_energy_ratio_hfer",
        "spectral_centroid_hz",
        "token_compression_ratio",
        "mean_model_inference_ms",
        "mean_spectrum_decode_ms",
        "total_control_latency_ms",
        "episode_wall_time_s",
    )
    summary: dict[str, Any] = {
        "episodes": n,
        "task_success_rate": sum(bool(r.get("task_success")) for r in rows) / n,
        "collision_free_rate": sum(float(r.get("collision_count", 0)) == 0.0 for r in rows) / n,
        "grasp_drop_rate": sum(bool(r.get("grasp_slip_or_drop")) for r in rows) / n,
    }
    for key in numeric_keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))
        else:
            summary[f"mean_{key}"] = None
            summary[f"std_{key}"] = None
    return summary


def save_summary_csv(path: Path, summary_by_schema: dict[str, dict[str, Any]]) -> None:
    """Write a clean tabular CSV summary of all evaluated configurations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "config_name",
        "episodes",
        "task_success_rate",
        "collision_free_rate",
        "mean_endpoint_error_m",
        "mean_joint_rms_jerk_rad_s3",
        "mean_joint_max_jerk_rad_s3",
        "mean_object_max_jerk_m_s3",
        "mean_chunk_boundary_jump_rad_s",
        "mean_sparc_smoothness",
        "mean_high_freq_energy_ratio_hfer",
        "mean_spectral_centroid_hz",
        "mean_token_compression_ratio",
        "mean_model_inference_ms",
        "mean_spectrum_decode_ms",
        "mean_total_control_latency_ms",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for config_name, stats in summary_by_schema.items():
            row = {"config_name": config_name}
            for col in columns[1:]:
                val = stats.get(col)
                row[col] = f"{val:.6g}" if isinstance(val, float) else val
            writer.writerow(row)


def save_summary_matrix_chart(
    path: Path, summary_by_schema: dict[str, dict[str, Any]]
) -> None:
    """Render a multi-row comparison bar chart across all evaluated spectrum schemas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = list(summary_by_schema.keys())
    if not configs:
        return
    metrics = (
        ("task_success_rate", "Task Success Rate (higher = better)", 1.0),
        ("mean_endpoint_error_m", "Mean Endpoint Error [m] (lower = better)", None),
        ("mean_joint_rms_jerk_rad_s3", "Joint RMS Jerk [rad/s^3] (lower = smoother)", None),
        ("mean_chunk_boundary_jump_rad_s", "Chunk Boundary Jump [rad/s] (lower = smoother)", None),
        ("mean_high_freq_energy_ratio_hfer", "High-Freq Energy Ratio >5Hz (lower = smoother)", None),
        ("mean_total_control_latency_ms", "Control Latency [ms/chunk] (lower = faster)", None),
    )
    palette = (
        (45, 105, 195),
        (210, 80, 45),
        (35, 150, 85),
        (145, 60, 175),
        (215, 145, 25),
        (40, 155, 165),
        (185, 50, 110),
    )
    width = max(1050, 260 + 150 * len(configs))
    height = 100 + 115 * len(metrics)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), "Smooth Operator — Spectral & VLA Evaluation Matrix", fill="black")

    for row_idx, (key, label, fixed_max) in enumerate(metrics):
        top = 65 + row_idx * 115
        vals = [abs(float(summary_by_schema[c].get(key) or 0.0)) for c in configs]
        maximum = fixed_max or max(vals + [1e-9])
        draw.text((30, top), label, fill="black")
        for col_idx, config_name in enumerate(configs):
            x = 30 + col_idx * 165
            bar_y = top + 24
            raw_val = float(summary_by_schema[config_name].get(key) or 0.0)
            bar_w = int(140 * min(1.0, abs(raw_val) / maximum))
            color = palette[col_idx % len(palette)]
            draw.rectangle((x, bar_y, x + max(bar_w, 2), bar_y + 24), fill=color)
            draw.text((x, bar_y + 28), f"{config_name[:18]}", fill=(40, 40, 40))
            draw.text((x, bar_y + 42), f"{raw_val:.4g}", fill=color)

    image.save(path)


def run_evaluation_matrix(
    schemas: Sequence[str],
    model: str = "astar_oracle",
    checkpoint: str = "lerobot/smolvla_base",
    golden_data: Optional[Path] = None,
    split: str = "evaluation",
    episodes: int = 5,
    robot: str = "so101",
    inference_host: str = "127.0.0.1",
    inference_port: int = 5478,
    quant_scale: float = 20.0,
    max_chunks: int = 25,
    output_dir: Path = Path("artifacts/eval_matrix"),
) -> dict[str, Any]:
    """Run the full Scope 2 evaluation matrix across episodes and spectrum schemas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    scene_records = load_eval_scene_specs(
        golden_data=golden_data,
        split=split,
        episodes=episodes,
        robot=robot,
    )
    sim_cls = SO101Simulation if robot == "so101" else ManipulatorSimulation
    workspace = SO101Simulation.WORKSPACE if robot == "so101" else SCARA_WORKSPACE

    socket_policy: Optional[SocketActionChunkPolicy] = None
    if model != "astar_oracle":
        socket_policy = SocketActionChunkPolicy(inference_host, inference_port, timeout=180.0)

    results_by_schema: dict[str, list[dict[str, Any]]] = {s: [] for s in schemas}
    episode_reports: list[dict[str, Any]] = []
    config_descriptors: list[dict[str, Any]] = []

    for schema_spec in schemas:
        mod = create_spectrum_module(schema_spec, quant_scale=quant_scale)
        config_descriptors.append(
            {
                "config_name": schema_spec,
                "model": model,
                "checkpoint": checkpoint if model != "astar_oracle" else None,
                "fast_tokenization": mod.fast_tokenization,
                "spectrum_schema": mod.schema_name,
                "freq_band": list(mod.freq_band),
            }
        )

    try:
        for ep_id, spec, saved_actions in scene_records:
            oracle_actions = saved_actions
            oracle_xy: list[Sequence[float]] = [row[1:3] for row in spec.oracle_trajectory]
            if oracle_actions is None:
                oracle_actions, oracle_samples, oracle_len = generate_oracle_action_sequence(spec)
                spec.oracle_path_length_m = oracle_len
                oracle_xy = [row[1:3] for row in oracle_samples]

            ep_entry: dict[str, Any] = {
                "episode_id": ep_id,
                "seed": spec.seed,
                "robot": spec.robot,
                "prompt": spec.prompt,
                "task_type": spec.task_type,
                "target_object": spec.target_object,
                "destination_xy": list(spec.destination_xy),
                "results_by_config": {},
            }

            for schema_spec in schemas:
                spectrum_mod = create_spectrum_module(schema_spec, quant_scale=quant_scale)
                active_policy: Any = (
                    PrecomputedChunkPolicy(oracle_actions, chunk_size=10)
                    if model == "astar_oracle"
                    else socket_policy
                )
                with sim_cls(gui=False, training_seed=spec.seed) as sim:
                    apply_scene_spec(sim, spec)
                    pipeline = ModularExecutionPipeline(
                        simulation=sim,
                        policy=active_policy,
                        spectrum_module=spectrum_mod,
                        model_name=model,
                        checkpoint=checkpoint,
                    )
                    res = pipeline.execute_task(spec, max_chunks=max_chunks)
                    safe_name = schema_spec.replace(":", "_")
                    chart_path = charts_dir / f"{ep_id}_{safe_name}.png"
                    save_spectral_diagnostic_chart(
                        path=chart_path,
                        object_samples=pipeline.object_samples,
                        joint_samples=pipeline.joint_samples,
                        obstacles=sim.obstacle_bounds,
                        metrics=res,
                        title=f"{ep_id} | {model} | {schema_spec} ({spec.prompt})",
                        workspace=workspace,
                        oracle_xy=oracle_xy,
                    )
                    # Strip large arrays from per-episode JSON summary to keep metadata compact
                    compact_res = {
                        k: v
                        for k, v in res.items()
                        if k not in ("trajectory", "joint_trajectory", "psd_freqs_hz", "psd_power")
                    }
                    compact_res["chart_path"] = str(chart_path)
                    results_by_schema[schema_spec].append(compact_res)
                    ep_entry["results_by_config"][schema_spec] = compact_res

            episode_reports.append(ep_entry)
    finally:
        if socket_policy is not None:
            socket_policy.close()

    aggregate_summary = {
        schema_spec: aggregate_schema_metrics(rows)
        for schema_spec, rows in results_by_schema.items()
    }

    metadata_payload = {
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{robot}_{model}",
        "robot": robot,
        "model": model,
        "checkpoint": checkpoint,
        "golden_dataset_path": str(golden_data) if golden_data else None,
        "split": split,
        "num_episodes": len(episode_reports),
        "configurations_compared": config_descriptors,
        "aggregate_summary": aggregate_summary,
        "episodes": episode_reports,
    }

    meta_path = output_dir / "eval_metadata.json"
    csv_path = output_dir / "eval_summary.csv"
    summary_chart_path = output_dir / "summary_matrix.png"
    meta_path.write_text(json.dumps(metadata_payload, indent=2) + "\n")
    save_summary_csv(csv_path, aggregate_summary)
    save_summary_matrix_chart(summary_chart_path, aggregate_summary)
    return metadata_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-data", type=Path, help="Path to Golden Dataset root")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--robot", choices=("so101", "scara"), default="so101")
    parser.add_argument("--model", choices=("astar_oracle", "smolvla", "fast_vla"), default="astar_oracle")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5478)
    parser.add_argument(
        "--schemas",
        default="direct,fast_idct,band_idct:0.35,ifft:0.35,idwt_haar,custom_hybrid",
        help="Comma-separated InverseSpectrumModule schemas to evaluate",
    )
    parser.add_argument("--quant-scale", type=float, default=20.0)
    parser.add_argument("--max-chunks", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval_matrix"))
    args = parser.parse_args()

    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    report = run_evaluation_matrix(
        schemas=schemas,
        model=args.model,
        checkpoint=args.checkpoint,
        golden_data=args.golden_data,
        split=args.split,
        episodes=args.episodes,
        robot=args.robot,
        inference_host=args.inference_host,
        inference_port=args.inference_port,
        quant_scale=args.quant_scale,
        max_chunks=args.max_chunks,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["aggregate_summary"], indent=2))


if __name__ == "__main__":
    main()

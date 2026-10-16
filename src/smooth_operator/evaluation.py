"""Evaluate deterministic or SmolVLA control on held-out prompt phrasings."""

import argparse
import json
from pathlib import Path

import pybullet as p

from .motion import BasicPickPlaceController
from .inference_protocol import SocketActionChunkPolicy
from .simulation import ManipulatorSimulation
from .vla import PromptResolver, SafeVLAExecutor


HELD_OUT_PROMPTS = (
    "Set the apple in the target",
    "Carry the Coke can to the green destination",
    "Transfer the pen into the goal",
    "Place the apple to the right of the Coke can",
    "Put the pen in front of the apple",
)


def evaluate(
    controller_name: str,
    checkpoint: str,
    episodes: int,
    host: str = "127.0.0.1",
    port: int = 5478,
) -> dict:
    policy = SocketActionChunkPolicy(host, port) if controller_name == "smolvla" else None
    results = []
    for index in range(episodes):
        prompt = HELD_OUT_PROMPTS[index % len(HELD_OUT_PROMPTS)]
        scene_seed = 10009 + index * 10
        with ManipulatorSimulation(training_seed=scene_seed) as simulation:
            positions = {
                name: p.getBasePositionAndOrientation(
                    body, physicsClientId=simulation.client_id
                )[0][:2]
                for name, body in simulation.object_ids.items()
            }
            task = PromptResolver().resolve(prompt, positions)
            if any(word in prompt.lower() for word in ("target", "destination", "goal")):
                simulation.set_goal_marker(task.destination)
            if policy is not None:
                result = SafeVLAExecutor(simulation, policy, checkpoint).execute(prompt)
            else:
                try:
                    metrics = BasicPickPlaceController(simulation).move_object(
                        task.object_name, task.destination
                    )
                    result = {
                        "controller": "astar",
                        "fallback_reason": None,
                        "prompt": prompt,
                        "checkpoint": None,
                        "seed": scene_seed,
                        "task_success": True,
                        **metrics,
                    }
                except ValueError as error:
                    result = {
                        "controller": "astar",
                        "fallback_reason": str(error),
                        "prompt": prompt,
                        "checkpoint": None,
                        "seed": scene_seed,
                        "task_success": False,
                        "collision_count": float(simulation.collision_count),
                    }
            result["seed"] = scene_seed
            results.append(result)
    if policy is not None:
        policy.close()
    successes = sum(
        result.get("task_success", False) and result.get("collision_count", 0) == 0
        for result in results
    )
    fallbacks = sum(result.get("controller") == "astar_fallback" for result in results)
    return {
        "episodes": episodes,
        "collision_free_success_rate": successes / episodes,
        "fallback_rate": fallbacks / episodes,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("astar", "smolvla"), default="astar")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5478)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.controller,
        args.checkpoint,
        args.episodes,
        args.inference_host,
        args.inference_port,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

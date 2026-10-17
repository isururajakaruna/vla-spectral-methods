"""Generate portable expert episodes with the Python 3.9 simulator."""

import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from .motion import BasicPickPlaceController
from .planning import CartesianAStarPlanner, WORKSPACE
from .simulation import ManipulatorSimulation
from .so101 import SO101AStarController, SO101Simulation


OBJECT_LABELS = {"apple": "apple", "coke_can": "Coke can", "pen": "pen"}
TARGET_PROMPTS = (
    "Move the {object} to the target",
    "Place the {object} in the destination",
    "Transfer the {object} to the goal cell",
    "Carry the {object} to the marked target",
)
RELATIONS = {
    "left of": (-0.20, 0.0),
    "to the right of": (0.20, 0.0),
    "in front of": (0.0, 0.20),
    "behind": (0.0, -0.20),
}
RELATION_PROMPTS = (
    "Place the {object} {relation} the {anchor}",
    "Move the {object} {relation} the {anchor}",
    "Put the {object} {relation} the {anchor}",
)


def split_for_seed(seed: int) -> str:
    remainder = seed % 10
    if remainder < 8:
        return "train"
    return "validation" if remainder == 8 else "evaluation"


def _object_radius(simulation: ManipulatorSimulation, object_name: str) -> float:
    bounds = simulation.body_bounds(simulation.object_ids[object_name])
    return float(
        np.linalg.norm(((bounds[2] - bounds[0]) / 2.0, (bounds[3] - bounds[1]) / 2.0))
    )


def _planner(simulation: ManipulatorSimulation, object_name: str):
    radius = _object_radius(simulation, object_name)
    if isinstance(simulation, SO101Simulation):
        return SO101AStarController(simulation)._planner(radius)
    return CartesianAStarPlanner(radius, simulation.obstacle_bounds)


def _valid_destination(simulation: ManipulatorSimulation, object_name: str, seed: int):
    rng = random.Random(seed * 7919 + 17)
    object_id = simulation.object_ids[object_name]
    planner = _planner(simulation, object_name)
    source = simulation.object_position(object_id)[:2]
    occupied = [
        simulation.object_position(body)[:2]
        for name, body in simulation.object_ids.items()
        if name != object_name
    ]
    workspace = simulation.WORKSPACE if isinstance(simulation, SO101Simulation) else WORKSPACE
    xmin, ymin, xmax, ymax = workspace
    spacing = 0.06 if isinstance(simulation, SO101Simulation) else 0.10
    margin = 0.04 if isinstance(simulation, SO101Simulation) else 0.08
    minimum_distance = 0.10 if isinstance(simulation, SO101Simulation) else 0.25
    candidates = [
        (round(x, 2), round(y, 2))
        for x in np.arange(xmin + margin, xmax - margin + 1e-6, spacing)
        for y in np.arange(ymin + margin, ymax - margin + 1e-6, spacing)
    ]
    rng.shuffle(candidates)
    for destination in candidates:
        if np.linalg.norm(np.asarray(destination) - source) < minimum_distance:
            continue
        if any(
            np.linalg.norm(np.asarray(destination) - position) < 0.07
            for position in occupied
        ):
            continue
        try:
            planner.plan(source, destination)
        except ValueError:
            continue
        return destination
    raise ValueError("Scene has no valid destination")


def _relation_task(
    simulation: ManipulatorSimulation, object_name: str, rng: random.Random
) -> tuple[tuple[float, float], str]:
    anchors = [name for name in simulation.object_ids if name != object_name]
    anchor = rng.choice(anchors)
    relation, offset = rng.choice(tuple(RELATIONS.items()))
    if isinstance(simulation, SO101Simulation):
        offset = tuple(value * 0.3 for value in offset)
    anchor_xy = simulation.object_position(simulation.object_ids[anchor])[:2]
    destination = tuple(map(float, anchor_xy + np.asarray(offset)))
    object_id = simulation.object_ids[object_name]
    planner = _planner(simulation, object_name)
    planner.plan(simulation.object_position(object_id)[:2], destination)
    prompt = rng.choice(RELATION_PROMPTS).format(
        object=OBJECT_LABELS[object_name],
        relation=relation,
        anchor=OBJECT_LABELS[anchor],
    )
    return destination, prompt


def _save_episode(root: Path, index: int, frames: list[dict], metadata: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="episode-", dir=root))
    try:
        states = np.stack([frame["observation.state"] for frame in frames]).astype(np.float32)
        actions = np.stack([frame["action"] for frame in frames]).astype(np.float32)
        timestamps = np.arange(len(frames), dtype=np.float64) / float(metadata["fps"])
        arrays = {"states": states, "actions": actions, "timestamps": timestamps}
        if metadata.get("robot") == "so101":
            for name in ("camera1", "camera2", "camera3"):
                key = f"observation.images.{name}"
                arrays[name] = np.stack([frame[key] for frame in frames])
        else:
            arrays["images"] = np.stack(
                [frame["observation.images.workspace"] for frame in frames]
            )
        np.savez_compressed(temporary / "frames.npz", **arrays)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.rename(root / f"episode_{index:06d}")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def generate_raw_episodes(
    episodes: int, root: Path, start_seed: int = 0, robot: str = "scara"
) -> dict:
    if episodes < 1:
        raise ValueError("Episode count must be positive")
    completed = 0
    seed = start_seed
    attempts = 0
    counts = {"train": 0, "validation": 0, "evaluation": 0}
    if robot not in {"scara", "so101"}:
        raise ValueError("robot must be 'scara' or 'so101'")
    simulation_type = SO101Simulation if robot == "so101" else ManipulatorSimulation
    while completed < episodes:
        attempts += 1
        rng = random.Random(seed)
        try:
            with simulation_type(training_seed=seed) as simulation:
                object_name = rng.choice(tuple(simulation.object_ids))
                source = simulation.object_position(simulation.object_ids[object_name])[:2]
                if (seed // 10 + seed) % 2 == 0:
                    destination = _valid_destination(simulation, object_name, seed)
                    simulation.set_goal_marker(destination)
                    prompt = rng.choice(TARGET_PROMPTS).format(object=OBJECT_LABELS[object_name])
                    task_type = "target"
                else:
                    destination, prompt = _relation_task(simulation, object_name, rng)
                    task_type = "spatial_relation"
                controller = (
                    SO101AStarController(simulation, record_demonstration=True)
                    if robot == "so101"
                    else BasicPickPlaceController(simulation)
                )
                metrics = controller.move_object(object_name, destination)
                if not controller.demonstration or metrics.get("collision_count", 1) != 0:
                    raise ValueError("Expert episode was incomplete or collided")
                split = split_for_seed(seed)
                metadata = {
                    "schema_version": 2 if robot == "so101" else 1,
                    "robot": robot,
                    "fps": 30 if robot == "so101" else 10,
                    "seed": seed,
                    "split": split,
                    "prompt": prompt,
                    "task_type": task_type,
                    "object": object_name,
                    "source": list(map(float, source)),
                    "destination": list(map(float, destination)),
                    "obstacles": [list(bounds) for bounds in simulation.obstacle_bounds],
                    "metrics": metrics,
                }
                _save_episode(root, completed, controller.demonstration, metadata)
                counts[split] += 1
                completed += 1
        except ValueError:
            pass
        seed += 1
    manifest = {
        "schema_version": 2 if robot == "so101" else 1,
        "episodes": completed,
        "attempts": attempts,
        "start_seed": start_seed,
        "next_seed": seed,
        "robot": robot,
        "splits": counts,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--robot", choices=("scara", "so101"), default="scara")
    args = parser.parse_args()
    print(
        json.dumps(
            generate_raw_episodes(args.episodes, args.output, args.start_seed, args.robot),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

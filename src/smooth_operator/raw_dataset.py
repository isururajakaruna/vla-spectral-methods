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


def _valid_destination(simulation: ManipulatorSimulation, object_name: str, seed: int):
    rng = random.Random(seed * 7919 + 17)
    object_id = simulation.object_ids[object_name]
    bounds = simulation.body_bounds(object_id)
    radius = float(np.linalg.norm(((bounds[2] - bounds[0]) / 2.0, (bounds[3] - bounds[1]) / 2.0)))
    planner = CartesianAStarPlanner(radius, simulation.obstacle_bounds)
    source = simulation.object_position(object_id)[:2]
    xmin, ymin, xmax, ymax = WORKSPACE
    candidates = [
        (round(x, 2), round(y, 2))
        for x in np.arange(xmin + 0.08, xmax - 0.07, 0.10)
        for y in np.arange(ymin + 0.08, ymax - 0.07, 0.10)
    ]
    rng.shuffle(candidates)
    for destination in candidates:
        if np.linalg.norm(np.asarray(destination) - source) < 0.25:
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
    anchor_xy = simulation.object_position(simulation.object_ids[anchor])[:2]
    destination = tuple(map(float, anchor_xy + np.asarray(offset)))
    object_id = simulation.object_ids[object_name]
    bounds = simulation.body_bounds(object_id)
    radius = float(np.linalg.norm(((bounds[2] - bounds[0]) / 2.0, (bounds[3] - bounds[1]) / 2.0)))
    planner = CartesianAStarPlanner(radius, simulation.obstacle_bounds)
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
        images = np.stack([frame["observation.images.workspace"] for frame in frames])
        states = np.stack([frame["observation.state"] for frame in frames]).astype(np.float32)
        actions = np.stack([frame["action"] for frame in frames]).astype(np.float32)
        timestamps = np.arange(len(frames), dtype=np.float64) / 10.0
        np.savez_compressed(
            temporary / "frames.npz",
            images=images,
            states=states,
            actions=actions,
            timestamps=timestamps,
        )
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.rename(root / f"episode_{index:06d}")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def generate_raw_episodes(episodes: int, root: Path, start_seed: int = 0) -> dict:
    if episodes < 1:
        raise ValueError("Episode count must be positive")
    completed = 0
    seed = start_seed
    attempts = 0
    counts = {"train": 0, "validation": 0, "evaluation": 0}
    while completed < episodes:
        attempts += 1
        rng = random.Random(seed)
        try:
            with ManipulatorSimulation(training_seed=seed) as simulation:
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
                controller = BasicPickPlaceController(simulation)
                metrics = controller.move_object(object_name, destination)
                if not controller.demonstration or metrics.get("collision_count", 1) != 0:
                    raise ValueError("Expert episode was incomplete or collided")
                split = split_for_seed(seed)
                metadata = {
                    "schema_version": 1,
                    "fps": 10,
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
        "schema_version": 1,
        "episodes": completed,
        "attempts": attempts,
        "start_seed": start_seed,
        "next_seed": seed,
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
    args = parser.parse_args()
    print(json.dumps(generate_raw_episodes(args.episodes, args.output, args.start_seed), indent=2))


if __name__ == "__main__":
    main()

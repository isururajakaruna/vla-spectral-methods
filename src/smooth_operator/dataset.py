"""Validate portable episodes and convert them to LeRobot datasets."""

import argparse
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "evaluation")
SCHEMAS = {
    "scara": {
        "images": {"images": ("observation.images.workspace", (120, 160, 3))},
        "vector_shape": (4,),
        "names": ["shoulder", "elbow", "wrist", "gripper"],
    },
    "so101": {
        "images": {
            name: (f"observation.images.{name}", (256, 256, 3))
            for name in ("camera1", "camera2", "camera3")
        },
        "vector_shape": (6,),
        "names": [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ],
    },
}


def validate_raw_episode(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((path / "metadata.json").read_text())
    with np.load(path / "frames.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    robot = metadata.get("robot", "scara")
    if robot not in SCHEMAS:
        raise ValueError(f"{path.name}: unsupported robot {robot!r}")
    schema = SCHEMAS[robot]
    required = {*schema["images"], "states", "actions", "timestamps"}
    if set(arrays) != required:
        raise ValueError(f"{path.name}: raw episode fields do not match the schema")
    frame_count = len(arrays["timestamps"])
    if frame_count < 2:
        raise ValueError(f"{path.name}: episode has too few frames")
    for archive_key, (_, shape) in schema["images"].items():
        if arrays[archive_key].shape != (frame_count, *shape):
            raise ValueError(f"{path.name}: invalid {archive_key} shape")
        if arrays[archive_key].dtype != np.uint8:
            raise ValueError(f"{path.name}: {archive_key} must be uint8")
    vector_shape = schema["vector_shape"]
    if arrays["states"].shape != (frame_count, *vector_shape):
        raise ValueError(f"{path.name}: invalid state shape")
    if arrays["actions"].shape != (frame_count, *vector_shape):
        raise ValueError(f"{path.name}: invalid action shape")
    if not all(np.all(np.isfinite(arrays[key])) for key in ("states", "actions", "timestamps")):
        raise ValueError(f"{path.name}: episode contains non-finite values")
    expected = np.arange(frame_count) / float(metadata["fps"])
    if not np.allclose(arrays["timestamps"], expected, atol=1e-6):
        raise ValueError(f"{path.name}: timestamps are not uniformly sampled")
    if metadata.get("split") not in SPLITS or not metadata.get("prompt"):
        raise ValueError(f"{path.name}: invalid metadata")
    if metadata.get("metrics", {}).get("collision_count") != 0:
        raise ValueError(f"{path.name}: colliding episode cannot be converted")
    return metadata, arrays


def _create_dataset(root: Path, repo_id: str, robot: str):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("Conversion requires the Python 3.12 LeRobot environment") from error
    schema = SCHEMAS[robot]
    features = {
        feature_key: {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
        for feature_key, shape in schema["images"].values()
    }
    features.update({
        "observation.state": {
            "dtype": "float32",
            "shape": schema["vector_shape"],
            "names": schema["names"],
        },
        "action": {
            "dtype": "float32",
            "shape": schema["vector_shape"],
            "names": schema["names"],
        },
    })
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=10,
        robot_type=f"pybullet_{robot}",
        use_videos=False,
        features=features,
    )


def convert_raw_dataset(source: Path, output: Path, repo_id: str) -> dict[str, int]:
    episode_paths = sorted(source.glob("episode_*"))
    if not episode_paths:
        raise ValueError("No raw episodes were found")
    validated = [(path, *validate_raw_episode(path)) for path in episode_paths]
    robots = {metadata.get("robot", "scara") for _, metadata, _ in validated}
    if len(robots) != 1:
        raise ValueError("A raw dataset cannot mix robot embodiments")
    robot = robots.pop()
    present_splits = {metadata["split"] for _, metadata, _ in validated}
    datasets = {
        split: _create_dataset(output / split, f"{repo_id}_{split}", robot)
        for split in SPLITS
        if split in present_splits
    }
    counts = {split: 0 for split in SPLITS}
    split_seeds = {split: set() for split in SPLITS}
    for path, metadata, arrays in validated:
        split = metadata["split"]
        seed = int(metadata["seed"])
        if any(seed in seeds for name, seeds in split_seeds.items() if name != split):
            raise ValueError(f"Seed {seed} occurs in more than one split")
        split_seeds[split].add(seed)
        dataset = datasets[split]
        schema = SCHEMAS[robot]
        for frame_index, (state, action) in enumerate(
            zip(arrays["states"], arrays["actions"])
        ):
            frame = {
                feature_key: arrays[archive_key][frame_index]
                for archive_key, (feature_key, _) in schema["images"].items()
            }
            frame.update({
                "observation.state": state,
                "action": action,
                "task": metadata["prompt"],
            })
            dataset.add_frame(frame)
        dataset.save_episode()
        counts[split] += 1
    for dataset in datasets.values():
        dataset.finalize()
    output.mkdir(parents=True, exist_ok=True)
    conversion = {"robot": robot, "splits": counts}
    (output / "conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/smooth_operator")
    args = parser.parse_args()
    print(json.dumps(convert_raw_dataset(args.input, args.output, args.repo_id), indent=2))


if __name__ == "__main__":
    main()

"""Validate portable episodes and convert them to LeRobot datasets."""

import argparse
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "evaluation")
IMAGE_SHAPE = (120, 160, 3)
VECTOR_SHAPE = (4,)


def validate_raw_episode(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((path / "metadata.json").read_text())
    with np.load(path / "frames.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"images", "states", "actions", "timestamps"}
    if set(arrays) != required:
        raise ValueError(f"{path.name}: raw episode fields do not match the schema")
    frame_count = len(arrays["timestamps"])
    if frame_count < 2:
        raise ValueError(f"{path.name}: episode has too few frames")
    if arrays["images"].shape != (frame_count, *IMAGE_SHAPE):
        raise ValueError(f"{path.name}: invalid image shape")
    if arrays["states"].shape != (frame_count, *VECTOR_SHAPE):
        raise ValueError(f"{path.name}: invalid state shape")
    if arrays["actions"].shape != (frame_count, *VECTOR_SHAPE):
        raise ValueError(f"{path.name}: invalid action shape")
    if arrays["images"].dtype != np.uint8:
        raise ValueError(f"{path.name}: images must be uint8")
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


def _create_dataset(root: Path, repo_id: str):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("Conversion requires the Python 3.12 LeRobot environment") from error
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=10,
        robot_type="pybullet_scara",
        use_videos=False,
        features={
            "observation.images.workspace": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": VECTOR_SHAPE,
                "names": ["shoulder", "elbow", "wrist", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": VECTOR_SHAPE,
                "names": ["shoulder", "elbow", "wrist", "gripper"],
            },
        },
    )


def convert_raw_dataset(source: Path, output: Path, repo_id: str) -> dict[str, int]:
    episode_paths = sorted(source.glob("episode_*"))
    if not episode_paths:
        raise ValueError("No raw episodes were found")
    validated = [(path, *validate_raw_episode(path)) for path in episode_paths]
    present_splits = {metadata["split"] for _, metadata, _ in validated}
    datasets = {
        split: _create_dataset(output / split, f"{repo_id}_{split}")
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
        for image, state, action in zip(
            arrays["images"], arrays["states"], arrays["actions"]
        ):
            dataset.add_frame(
                {
                    "observation.images.workspace": image,
                    "observation.state": state,
                    "action": action,
                    "task": metadata["prompt"],
                }
            )
        dataset.save_episode()
        counts[split] += 1
    for dataset in datasets.values():
        dataset.finalize()
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversion.json").write_text(json.dumps(counts, indent=2) + "\n")
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

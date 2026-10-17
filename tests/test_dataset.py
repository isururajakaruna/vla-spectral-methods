import unittest
from unittest.mock import patch

import json
from pathlib import Path
import tempfile

import numpy as np

from smooth_operator.dataset import validate_raw_episode
from smooth_operator.raw_dataset import split_for_seed
from smooth_operator.training import dataset_robot, training_command


class DatasetTests(unittest.TestCase):
    def test_seed_splits_do_not_overlap(self):
        splits = {split_for_seed(seed) for seed in range(20)}
        self.assertEqual(splits, {"train", "validation", "evaluation"})
        self.assertEqual(split_for_seed(10), "train")
        self.assertEqual(split_for_seed(18), "validation")
        self.assertEqual(split_for_seed(19), "evaluation")

    def test_raw_episode_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {
                "fps": 10,
                "split": "train",
                "prompt": "Move the apple to the target",
                "metrics": {"collision_count": 0},
            }
            (path / "metadata.json").write_text(json.dumps(metadata))
            np.savez_compressed(
                path / "frames.npz",
                images=np.zeros((2, 120, 160, 3), dtype=np.uint8),
                states=np.zeros((2, 4), dtype=np.float32),
                actions=np.zeros((2, 4), dtype=np.float32),
                timestamps=np.asarray((0.0, 0.1)),
            )
            loaded, arrays = validate_raw_episode(path)
            self.assertEqual(loaded["split"], "train")
            self.assertEqual(arrays["actions"].shape, (2, 4))

    def test_so101_raw_episode_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {
                "robot": "so101", "fps": 30, "split": "train",
                "prompt": "Move the apple to the target",
                "metrics": {"collision_count": 0},
            }
            (path / "metadata.json").write_text(json.dumps(metadata))
            images = np.zeros((2, 256, 256, 3), dtype=np.uint8)
            np.savez_compressed(
                path / "frames.npz", camera1=images, camera2=images,
                camera3=images, states=np.zeros((2, 6), dtype=np.float32),
                actions=np.zeros((2, 6), dtype=np.float32),
                timestamps=np.asarray((0.0, 1 / 30)),
            )
            loaded, arrays = validate_raw_episode(path)
            self.assertEqual(loaded["robot"], "so101")
            self.assertEqual(arrays["camera3"].shape, (2, 256, 256, 3))
            self.assertEqual(arrays["actions"].shape, (2, 6))

    @patch("smooth_operator.training.shutil.which", return_value="/bin/lerobot-train")
    def test_training_configuration(self, _):
        command = training_command(
            "user/data", Path("out"), 20, True, config_path=Path("checkpoint/config.json")
        )
        self.assertIn("--policy.device=mps", command)
        self.assertIn("--batch_size=1", command)
        self.assertIn("--policy.chunk_size=10", command)
        self.assertIn("--policy.push_to_hub=false", command)
        self.assertIn(
            '--rename_map={"observation.images.workspace":"observation.images.camera1"}',
            command,
        )
        self.assertIn("--resume=true", command)

    @patch("smooth_operator.training.shutil.which", return_value="/bin/lerobot-train")
    def test_so101_training_keeps_three_camera_keys(self, _):
        command = training_command(
            "local/smooth_operator_so101_train", Path("out"), 20, False,
            device="cuda", robot="so101",
        )
        self.assertIn("--policy.device=cuda", command)
        self.assertFalse(any(value.startswith("--rename_map=") for value in command))
        self.assertEqual(
            dataset_robot(None, "auto", "local/smooth_operator_so101_train"),
            "so101",
        )


if __name__ == "__main__":
    unittest.main()

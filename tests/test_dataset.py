import unittest
from unittest.mock import patch

import json
from pathlib import Path
import tempfile

import numpy as np

from smooth_operator.dataset import validate_raw_episode
from smooth_operator.raw_dataset import split_for_seed
from smooth_operator.training import training_command


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


if __name__ == "__main__":
    unittest.main()

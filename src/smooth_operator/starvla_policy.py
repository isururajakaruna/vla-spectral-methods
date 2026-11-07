"""StarVLA checkpoints (github.com/starVLA/starVLA) driving LIBERO from raw observations.

StarVLA is not a LeRobot policy: it ships a single `.pt` state dict plus its own
framework code, so this adapter loads it through StarVLA's `baseframework`, feeds it
the observation exactly as StarVLA's LIBERO evaluation does, and exposes the same
`(actions, spectrum)` interface as `LeRobotVLAPolicy` for the rollout loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

DEFAULT_STARVLA_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "starVLA"


def _starvla_root() -> Path:
    root = Path(os.environ.get("STARVLA_ROOT", DEFAULT_STARVLA_ROOT))
    if not (root / "starVLA" / "model" / "framework").is_dir():
        raise RuntimeError(
            f"StarVLA code not found at {root}. Clone https://github.com/starVLA/starVLA there "
            "or set STARVLA_ROOT."
        )
    return root


class StarVLAPolicy:
    """A StarVLA `QwenFast` checkpoint: FAST tokens decoded to 7-D LIBERO end-effector deltas."""

    raw_observation = True

    def __init__(self, spec: Mapping[str, Any], device: str = "auto") -> None:
        import torch
        from huggingface_hub import snapshot_download

        self.spec = dict(spec)
        self.embodiment = self.spec.get("embodiment", "libero")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("StarVLA checkpoints need a CUDA GPU")
        self.device = torch.device(device)

        root = _starvla_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import starVLA.model.framework.VLM4A.QwenFast as qwen_fast
        import starVLA.model.modules.action_model.fast_ActionHeader as fast_header
        from starVLA.model.framework.base_framework import baseframework

        fast_name = self.spec.get("fast_tokenizer", "physical-intelligence/fast")

        def fast_action_head(config=None):
            # StarVLA's default head loads a local `playground/` copy of physical-intelligence/fast.
            head = fast_header.Fast_Action_Tokenizer.__new__(fast_header.Fast_Action_Tokenizer)
            fast_header.nn.Module.__init__(head)
            head.fast_tokenizer = fast_header._load_fast_processor(fast_name)
            return head

        qwen_fast.get_action_model = fast_action_head

        repo = Path(snapshot_download(self.spec["checkpoint"]))
        overrides = [
            f"framework.qwenvl.base_vlm={self.spec['base_vlm']}",
            "framework.qwenvl.attn_implementation=sdpa",
        ]
        self.model = baseframework.from_pretrained(repo / self.spec["checkpoint_file"], config_overrides=overrides)
        self.model = self.model.to(self.device).eval()

        self.n_action_steps = int(self.model.action_horizon)
        self.image_size = tuple(self.spec.get("image_size", (224, 224)))
        # LeRobot LIBERO pixel keys: "image" is the agent view, "image2" the wrist camera.
        self.cameras = tuple(self.spec.get("cameras", ("image",)))
        stats = self.model.norm_stats[self.spec["unnorm_key"]]["action"]
        lo, hi = np.asarray(stats["min"], dtype=np.float64), np.asarray(stats["max"], dtype=np.float64)
        normalized = np.asarray(stats["mask"], dtype=bool)
        # min_max normalization on masked dims; the others (the 0..1 gripper) are raw.
        self._scale = np.where(normalized, (hi - lo) / 2.0, 1.0)
        self._offset = np.where(normalized, (hi + lo) / 2.0, 0.0)

        self.emits_dct = self.spec.get("native_output") == "dct_tokens"
        self._captured_coefficients: list[np.ndarray] = []
        self._install_dct_capture()
        self.reset()

    def _install_dct_capture(self) -> None:
        processor = self.model.action_model.fast_tokenizer
        # The FAST processor module may be loaded from file without a sys.modules entry.
        namespace = type(processor).decode.__globals__
        original_idct = namespace["idct"]
        captured = self._captured_coefficients

        def capturing_idct(coefficients, *args, **kwargs):
            captured.append(np.array(coefficients, dtype=np.float64))
            return original_idct(coefficients, *args, **kwargs)

        namespace["idct"] = capturing_idct

    def reset(self) -> None:
        self._captured_coefficients.clear()
        self.undecodable_chunks = 0
        self._gripper = 1.0

    def _hold_chunk(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Zero end-effector motion with the gripper unchanged, for a chunk the model failed to write."""
        self.undecodable_chunks += 1
        hold = np.zeros(7)
        hold[6] = self._gripper
        spectrum = {
            "basis": "dct-ii-ortho",
            "dct_coeffs": np.zeros((self.n_action_steps, 7)).tolist(),
            "offset": hold.tolist(),
            "non_zero_ratio": 0.0,
        }
        return np.tile(hold, (self.n_action_steps, 1)), spectrum

    def predict_raw(self, obs: Mapping[str, Any], instruction: str) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """One action chunk from a LeRobot LIBERO observation, prepared like StarVLA's LIBERO client."""
        from PIL import Image

        height, width = self.image_size
        images = []
        for key in self.cameras:
            # MuJoCo renders bottom-up; StarVLA's LIBERO eval rotates each view by 180° before use.
            view = np.ascontiguousarray(np.asarray(obs["pixels"][key][0])[::-1, ::-1])
            if view.shape[:2] != (height, width):
                view = np.asarray(Image.fromarray(view).resize((width, height), Image.BILINEAR))
            images.append(view)
        self._captured_coefficients.clear()
        out = self.model.predict_action([{"image": images, "lang": instruction}])
        normalized = np.asarray(out["normalized_actions"], dtype=np.float64)[0]
        if not self._captured_coefficients or not np.any(self._captured_coefficients[-1]):
            # The FAST processor substitutes zeros when the generated tokens do not form a chunk.
            return self._hold_chunk()
        actions = normalized * self._scale + self._offset
        self._gripper = float(actions[-1, 6])
        coefficients = self._captured_coefficients[-1] * self._scale
        spectrum = {
            "basis": "dct-ii-ortho",
            "dct_coeffs": coefficients.tolist(),
            "offset": self._offset.tolist(),
            "non_zero_ratio": float(np.count_nonzero(self._captured_coefficients[-1]))
            / float(self._captured_coefficients[-1].size),
        }
        return actions, spectrum

    @staticmethod
    def to_env_action(row: np.ndarray) -> np.ndarray:
        """StarVLA predicts gripper openness in [0, 1]; LIBERO wants -1 (open) or +1 (close)."""
        action = np.asarray(row, dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action

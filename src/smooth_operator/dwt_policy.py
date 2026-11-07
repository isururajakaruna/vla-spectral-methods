"""Adaptive/fixed Haar-DWT action-token policies trained in experiment/adaptive_dwt_vla.

The checkpoints hold only LoRA + action-row adapters on top of `lerobot/pi0fast-base`, so they are
loaded through the experiment's `lerobot_policy_adaptive_dwt` plugin (installed editable in the VLA
environment) and exposed with the `LeRobotVLAPolicy` interface the LIBERO rollout uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "adaptive_dwt_vla"


class AdaptiveDWTVLAPolicy:
    def __init__(self, spec: Mapping[str, Any], device: str = "auto") -> None:
        import torch
        from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import load_checkpoint
        from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import make_processors

        if device not in ("auto", "cuda") or not torch.cuda.is_available():
            raise RuntimeError("DWT-token checkpoints need a CUDA GPU")
        self.torch = torch
        self.spec = dict(spec)
        self.embodiment = "libero"
        path = EXPERIMENT_ROOT / self.spec["checkpoint"]
        if not (path / "adapters.safetensors").exists():
            raise RuntimeError(f"Checkpoint not found: {path}")
        self.model, meta = load_checkpoint(path)
        self.config = self.model.config
        stats = meta["norm_stats"]
        ds_stats = {
            "action": {k: torch.tensor(stats["action"][k], dtype=torch.float32) for k in ("mean", "std")},
            "observation.state": {k: torch.tensor(stats["state"][k], dtype=torch.float32) for k in ("mean", "std")},
        }
        self.preprocess, self.postprocess = make_processors(self.config, ds_stats)
        self.emits_dct = False
        self.undecodable_chunks = 0

    def predict_frame(self, frame: Mapping[str, Any]) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        with self.torch.inference_mode():
            chunk = self.model.predict_action_chunk(self.preprocess(dict(frame)))
            actions = np.asarray(self.postprocess(chunk).float().cpu(), dtype=np.float64)[0]
        info = self.model.decode_log[-1] if self.model.decode_log else {}
        if info.get("valid") is False:
            self.undecodable_chunks += 1
        return actions, None

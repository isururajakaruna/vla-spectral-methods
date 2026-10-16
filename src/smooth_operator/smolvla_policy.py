"""PyBullet-independent SmolVLA model adapter."""

import numpy as np


class SmolVLAPolicy:
    def __init__(self, checkpoint: str, device: str = "mps") -> None:
        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
        except ImportError as error:
            raise RuntimeError("SmolVLA requires the Python 3.12 LeRobot environment") from error
        self.torch = torch
        self.device = torch.device(device)
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self.model = Policy.from_pretrained(checkpoint).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def predict_action_chunk(
        self, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        torch = self.torch
        frame = {
            "observation.images.camera1": torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device),
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(self.device),
            "task": [instruction],
        }
        with torch.inference_mode():
            action = self.model.predict_action_chunk(self.preprocess(frame))
            action = self.postprocess(action)
        return np.asarray(action.detach().cpu()).reshape(-1, 4)[:10]

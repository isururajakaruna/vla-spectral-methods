"""PyBullet-independent SmolVLA model adapter."""

import numpy as np


class SmolVLAPolicy:
    SO100_ACTION_MEAN = np.asarray(
        (1.595914, 119.944237, 109.769775, 56.705513, -27.422693, 12.002786),
        dtype=np.float32,
    )
    SO100_ACTION_STD = np.asarray(
        (26.391779, 52.411404, 49.853806, 36.998302, 59.359951, 19.039955),
        dtype=np.float32,
    )

    def __init__(self, checkpoint: str, device: str = "auto") -> None:
        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
        except ImportError as error:
            raise RuntimeError("SmolVLA requires the Python 3.12 LeRobot environment") from error
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self.model = Policy.from_pretrained(checkpoint).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.image_keys = tuple(
            key for key in self.model.config.input_features if "images" in key
        )
        self.state_size = self.model.config.input_features["observation.state"].shape[0]
        self.action_size = self.model.config.output_features["action"].shape[0]
        self.is_so100_base = self.action_size == 6 and len(self.image_keys) == 3

    def predict_action_chunk(
        self, image, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        torch = self.torch
        if isinstance(image, dict):
            missing = set(self.image_keys) - set(image)
            if missing:
                raise ValueError(f"Missing SmolVLA cameras: {sorted(missing)}")
            images = {key: image[key] for key in self.image_keys}
        else:
            if len(self.image_keys) != 1:
                raise ValueError(f"SmolVLA checkpoint requires {len(self.image_keys)} cameras")
            images = {self.image_keys[0]: image}
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_size,):
            raise ValueError(f"SmolVLA checkpoint requires state shape ({self.state_size},)")
        frame = {
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(self.device),
            "task": [instruction],
        }
        for key, value in images.items():
            frame[key] = torch.from_numpy(np.asarray(value).copy()) \
                .permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            action = self.model.predict_action_chunk(self.preprocess(frame))
            action = self.postprocess(action)
        actions = np.asarray(action.detach().cpu()).reshape(-1, self.action_size)
        # The released base checkpoint contains `so100`, `so100-red`, and
        # `so100-blue` statistics. LeRobot 0.6.1 cannot associate its generic
        # `action` output with one of those names, so its saved postprocessor
        # leaves the chunk normalized. Select the canonical single-arm stats.
        if self.is_so100_base and float(np.median(np.abs(actions[:, 1:4]))) < 10.0:
            actions = actions * self.SO100_ACTION_STD + self.SO100_ACTION_MEAN
        return actions

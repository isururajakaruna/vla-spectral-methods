"""PyBullet-independent adapter that runs any LeRobot VLA checkpoint on SO-101 observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .vla_models import LEGACY_OFFSET_DEGREES, LEGACY_SIGN, SIM_JOINT_NAMES

COMMUNITY_STATS_PATH = Path(__file__).with_name("assets") / "so100_community_stats.json"
SIM_JOINTS = 6


def gated_repo_message(error: BaseException) -> Optional[str]:
    """Explain a Hugging Face gated-repo failure anywhere in the exception chain, if there is one."""
    seen = error
    while seen is not None:
        text = str(seen)
        if "gated repo" in text.lower():
            repo = next((w for w in text.split() if w.startswith("https://huggingface.co/")), "")
            repo = repo.split("/resolve/")[0].rstrip(".")
            return (
                f"A required file is in a gated Hugging Face repo ({repo or 'see server log'}). "
                "Accept its license on that page with the account whose token is configured "
                "(`hf auth login`), then select the model again."
            )
        seen = seen.__cause__ or seen.__context__
    return None


def _pad(values: np.ndarray, size: int, fill: float) -> np.ndarray:
    out = np.full(size, fill, dtype=np.float32)
    out[: len(values)] = values[:size]
    return out


class LeRobotVLAPolicy:
    """Loads a LeRobot policy plus its processors and predicts legacy-degree joint chunks."""

    def __init__(self, spec: Mapping[str, Any], device: str = "auto") -> None:
        try:
            import torch
            from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.factory import get_policy_class
        except ImportError as error:
            raise RuntimeError("VLA inference requires the Python 3.12 LeRobot environment") from error
        self.torch = torch
        self.spec = dict(spec)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)

        checkpoint = self.spec["checkpoint"]
        config = PreTrainedConfig.from_pretrained(checkpoint)
        for key, value in self.spec.get("config_overrides", {}).items():
            setattr(config, key, value)
        config.device = str(self.device)
        policy_cls = get_policy_class(config.type)
        self.model = policy_cls.from_pretrained(checkpoint, config=config).to(self.device).eval()
        self.config = self.model.config

        self.action_dim = int(self.config.output_features["action"].shape[0])
        self.embodiment = self.spec.get("embodiment", "so101")
        if self.embodiment == "libero":
            # LIBERO observations arrive already keyed and processed by LeRobot's env processors.
            self.camera_map: dict[str, str] = {}
            self.out_dims = self.action_dim
        else:
            self.camera_map = dict(self.spec["camera_map"])
            unknown = set(self.camera_map) - set(self.config.image_features)
            if unknown:
                raise ValueError(f"{checkpoint} has no camera inputs named {sorted(unknown)}")
            if self.action_dim < SIM_JOINTS:
                raise ValueError(f"{checkpoint} predicts {self.action_dim}-D actions; SO-101 needs 6")
            self.out_dims = SIM_JOINTS

        pre_overrides: dict[str, Any] = {"device_processor": {"device": str(self.device)}}
        post_overrides: dict[str, Any] = {}
        self.action_affine: Optional[tuple[np.ndarray, np.ndarray]] = None
        if self.spec.get("stats") == "so100_community":
            stats, features = self._community_stats(FeatureType, PolicyFeature)
            norm_map = dict(self.config.normalization_mapping)
            for overrides, name in ((pre_overrides, "normalizer_processor"), (post_overrides, "unnormalizer_processor")):
                overrides[name] = {"stats": stats, "features": features, "norm_map": norm_map}
            self.action_affine = self._affine(stats["action"], norm_map.get("ACTION"), NormalizationMode)
        pre_overrides.update(self.spec.get("preprocessor_overrides", {}))
        post_overrides.update(self.spec.get("postprocessor_overrides", {}))
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.config,
            checkpoint,
            preprocessor_overrides=pre_overrides,
            postprocessor_overrides=post_overrides,
        )

        self._setup_joint_frame()
        self.emits_dct = self.spec.get("native_output") == "dct_tokens"
        self._captured_coefficients: list[np.ndarray] = []
        if self.emits_dct:
            if self.action_affine is None:
                self.action_affine = self._action_affine_from_processor(NormalizationMode)
            self._install_dct_capture()

    def _community_stats(self, FeatureType, PolicyFeature):
        raw = json.loads(COMMUNITY_STATS_PATH.read_text(encoding="utf-8"))
        torch = self.torch
        stats: dict[str, dict[str, Any]] = {}
        # Padded action dimensions normalize to themselves (zero mean, unit scale).
        fills = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}
        for key, size in (("observation.state", SIM_JOINTS), ("action", self.action_dim)):
            stats[key] = {
                name: torch.tensor(_pad(np.asarray(raw[key][name], dtype=np.float32), size, fill))
                for name, fill in fills.items()
            }
        features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(SIM_JOINTS,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }
        return stats, features

    @staticmethod
    def _affine(stats, mode, NormalizationMode) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint (scale, offset) so that action = normalized * scale + offset."""
        as_np = {k: np.asarray(v, dtype=np.float64) for k, v in stats.items()}
        if mode == NormalizationMode.MEAN_STD:
            return as_np["std"], as_np["mean"]
        if mode == NormalizationMode.QUANTILES:
            return (as_np["q99"] - as_np["q01"]) / 2.0, (as_np["q99"] + as_np["q01"]) / 2.0
        if mode == NormalizationMode.MIN_MAX:
            return (as_np["max"] - as_np["min"]) / 2.0, (as_np["max"] + as_np["min"]) / 2.0
        raise ValueError(f"Unsupported action normalization {mode}")

    def _install_dct_capture(self) -> None:
        import importlib

        module = importlib.import_module(type(self.model).__module__)
        original_idct = module.idct
        captured = self._captured_coefficients

        def capturing_idct(coefficients, *args, **kwargs):
            captured.append(np.array(coefficients, dtype=np.float64))
            return original_idct(coefficients, *args, **kwargs)

        module.idct = capturing_idct

    def _spectrum(self, model_actions: np.ndarray) -> dict[str, Any]:
        if not self._captured_coefficients:
            raise RuntimeError("The model did not decode FAST tokens for this chunk")
        dims = self.out_dims
        normalized = self._captured_coefficients[-1][:, :dims]
        if not np.any(normalized):
            # LeRobot substitutes all-zero coefficients when the generated tokens do not decode.
            raise RuntimeError("The model's FAST tokens did not decode to a valid action chunk")
        scale, offset = (v[:dims] for v in self.action_affine)
        coefficients = normalized * scale
        # The model's own decoded chunk must equal IDCT(coefficients) + offset; otherwise a
        # processor step (e.g. relative actions) changed it and the spectrum would be wrong.
        from scipy.fftpack import idct

        rebuilt = idct(coefficients, axis=0, norm="ortho") + offset
        if rebuilt.shape != model_actions.shape or not np.allclose(rebuilt, model_actions, atol=1e-3):
            raise RuntimeError("Captured DCT coefficients do not reproduce the model's actions")
        # The joint-frame map is affine per joint, so it carries over to the coefficients exactly.
        return {
            "basis": "dct-ii-ortho",
            "dct_coeffs": (coefficients[:, self._perm] * self._frame_scale).tolist(),
            "offset": (offset[self._perm] * self._frame_scale + self._frame_shift).tolist(),
            "non_zero_ratio": float(np.count_nonzero(normalized)) / float(normalized.size),
        }

    def _setup_joint_frame(self) -> None:
        """Simulator joint j = scale[j] * model joint perm[j] + shift[j] (legacy degrees by default)."""
        frame = self.spec.get("joint_frame")
        if frame is None:
            self._perm = np.arange(self.out_dims)
            self._frame_scale = np.ones(self.out_dims)
            self._frame_shift = np.zeros(self.out_dims)
            return
        if frame["units"] != "radians":
            raise ValueError(f"Unsupported joint frame units {frame['units']!r}")
        order = list(frame["order"])
        if sorted(order) != sorted(SIM_JOINT_NAMES):
            raise ValueError(f"Joint frame order must be a permutation of {SIM_JOINT_NAMES}")
        self._perm = np.asarray([order.index(name) for name in SIM_JOINT_NAMES])
        self._frame_scale = np.asarray(LEGACY_SIGN) * (180.0 / np.pi)
        self._frame_shift = np.asarray(LEGACY_OFFSET_DEGREES)

    def _model_state(self, state: np.ndarray) -> np.ndarray:
        in_model_units = (state - self._frame_shift) / self._frame_scale
        out = np.empty(SIM_JOINTS)
        out[self._perm] = in_model_units
        return out.astype(np.float32)

    def _action_affine_from_processor(self, NormalizationMode) -> tuple[np.ndarray, np.ndarray]:
        for step in getattr(self.postprocess, "steps", []):
            stats = getattr(step, "stats", None)
            if stats and "action" in stats:
                action_stats = {k: np.asarray(v.cpu() if hasattr(v, "cpu") else v) for k, v in stats["action"].items()}
                return self._affine(action_stats, self.config.normalization_mapping.get("ACTION"), NormalizationMode)
        raise ValueError("DCT coefficient export needs the checkpoint's action statistics")

    def predict(
        self, images: Mapping[str, np.ndarray], state: np.ndarray, instruction: str
    ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        torch = self.torch
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (SIM_JOINTS,):
            raise ValueError(f"Expected a {SIM_JOINTS}-D SO-101 joint state")
        missing = set(self.camera_map.values()) - set(images)
        if missing:
            raise ValueError(f"Missing simulator cameras: {sorted(missing)}")
        frame: dict[str, Any] = {
            "observation.state": torch.from_numpy(self._model_state(state)).unsqueeze(0),
            "task": [instruction],
        }
        for model_key, sim_key in self.camera_map.items():
            frame[model_key] = (
                torch.from_numpy(np.asarray(images[sim_key]).copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            )
        return self.predict_frame(frame)

    def predict_frame(self, frame: Mapping[str, Any]) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Run one chunk on a LeRobot-format observation batch; actions are in the simulator frame."""
        torch = self.torch
        self._captured_coefficients.clear()
        with torch.inference_mode():
            batch = self.preprocess(dict(frame))
            action = self.model.predict_action_chunk(batch)
            action = self.postprocess(action)
        model_actions = np.asarray(action.detach().float().cpu(), dtype=np.float64)
        model_actions = model_actions.reshape(-1, self.action_dim)[:, : self.out_dims]
        actions = model_actions[:, self._perm] * self._frame_scale + self._frame_shift
        spectrum = self._spectrum(model_actions) if self.emits_dct else None
        return actions, spectrum

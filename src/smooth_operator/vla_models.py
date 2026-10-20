"""VLA checkpoints the studio can run, and how each one is wired to the SO-101 simulator.

Shared by the web server (catalog, validation) and the Python 3.12 inference
server (loading). Joint values exchanged with every model are legacy LeRobot
`so100` degrees, the frame of the SO-100 community data (see `SO101Simulation`).
"""

from __future__ import annotations

from typing import Any, Optional

SIM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Legacy `so100` degrees = sign * URDF degrees + offset (LeRobot's backward-compatibility transform).
LEGACY_SIGN = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
LEGACY_OFFSET_DEGREES = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)

TIME_DOMAIN_DECODERS = ("direct",)
# Decoders that invert DCT coefficients, i.e. valid for FAST/DCT action tokens.
DCT_DECODERS = ("fast_idct", "band_idct", "custom_hybrid")

SIM_TOP = "observation.images.camera1"
SIM_WRIST = "observation.images.camera2"
SIM_SIDE = "observation.images.camera3"

PI_CAMERAS = {
    "observation.images.base_0_rgb": SIM_TOP,
    "observation.images.left_wrist_0_rgb": SIM_WRIST,
}

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "id": "smolvla_base",
        "name": "SmolVLA base",
        "family": "SmolVLA (Hugging Face)",
        "policy_type": "smolvla",
        "checkpoint": "lerobot/smolvla_base",
        "parameters": "450M",
        "badge": "VLA · 450M · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 50-step continuous chunks",
        "camera_map": {SIM_TOP: SIM_TOP, SIM_WRIST: SIM_WRIST, SIM_SIDE: SIM_SIDE},
        "stats": "so100_community",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "SO-100 community datasets (same arm family)",
        "description": (
            "Small VLA pretrained on ~480 LeRobot community SO-100 datasets. Its checkpoint ships no "
            "state statistics, so state and action use SO-100 community statistics."
        ),
    },
    {
        "id": "molmoact2_so100",
        "name": "MolmoAct2 SO-100/101",
        "family": "MolmoAct2 (Ai2)",
        "policy_type": "molmoact2",
        "checkpoint": "lerobot/MolmoAct2-SO100_101-LeRobot",
        "parameters": "≈5B",
        "badge": "VLA · SO-101 tuned · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 30-step continuous chunks",
        "camera_map": {"observation.images.cam0": SIM_TOP, "observation.images.cam1": SIM_WRIST},
        "stats": "checkpoint",
        # The simulator already sends legacy degrees, the frame this checkpoint was trained in.
        "preprocessor_overrides": {
            "molmoact2_state_frame_transform": {"joint_signs": None, "joint_offsets": None},
        },
        "postprocessor_overrides": {
            "molmoact2_action_frame_transform": {"joint_signs": None, "joint_offsets": None},
        },
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "Generalist MolmoAct2, fine-tuned on SO-100/101 community data",
        "description": (
            "Ai2's generalist MolmoAct2 fine-tuned for single SO-100/101 arms, with its own SO-100/101 "
            "normalization. The closest of these models to zero-shot deployment on this arm."
        ),
    },
    {
        "id": "pi05_base",
        "name": "π0.5 base",
        "family": "π0.5 (Physical Intelligence)",
        "policy_type": "pi05",
        "checkpoint": "lerobot/pi05_base",
        "parameters": "≈3B",
        "badge": "Generalist VLA · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 50-step continuous chunks",
        "camera_map": PI_CAMERAS,
        "stats": "so100_community",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "Many robot embodiments + web data; no SO-100",
        "description": (
            "Physical Intelligence's most general open model (cross-embodiment robot data plus web "
            "data). SO-100 was not in its pretraining, so it uses SO-100 community statistics and "
            "is expected to need fine-tuning."
        ),
    },
    {
        "id": "pi0_base",
        "name": "π0 base",
        "family": "π0 (Physical Intelligence)",
        "policy_type": "pi0",
        "checkpoint": "lerobot/pi0_base",
        "parameters": "≈3B",
        "badge": "Generalist VLA · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 50-step continuous chunks",
        "camera_map": PI_CAMERAS,
        "stats": "so100_community",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "Many robot embodiments; no SO-100",
        "description": (
            "Physical Intelligence's original generalist flow-matching VLA. SO-100 was not in its "
            "pretraining, so it uses SO-100 community statistics and is expected to need fine-tuning."
        ),
    },
    {
        "id": "pi0fast_so101_pick_cube",
        "name": "π0-FAST SO-101 pick & place",
        "family": "π0-FAST (Physical Intelligence), community fine-tune",
        "policy_type": "pi0_fast",
        "checkpoint": "gpudad/pi0fast-so101-pick-cube",
        "parameters": "≈3B",
        "badge": "VLA · SO-101 tuned · FAST DCT tokens",
        "native_output": "dct_tokens",
        "action_head": "Autoregressive FAST tokens: BPE-compressed, quantized DCT-II coefficients, 10-step chunks",
        "camera_map": {
            "observation.images.overhead": SIM_TOP,
            "observation.images.wrist": SIM_WRIST,
            "observation.images.front": SIM_SIDE,
        },
        "stats": "checkpoint",
        # Trained in MuJoCo on so101_new_calib (the simulator's URDF), joints in alphabetical order.
        "joint_frame": {
            "units": "radians",
            "order": ("elbow_flex", "gripper", "shoulder_lift", "shoulder_pan", "wrist_flex", "wrist_roll"),
        },
        "config_overrides": {"action_tokenizer_name": "lerobot/fast-action-tokenizer"},
        "preprocessor_overrides": {
            "action_tokenizer_processor": {"action_tokenizer_name": "lerobot/fast-action-tokenizer"},
        },
        "decoders": DCT_DECODERS,
        "pretraining": "π0-FAST base, fine-tuned 10k steps on 11k simulated SO-101 pick-and-place episodes",
        "prompt_hint": "pick up the object and place it in the target location",
        "unavailable_reason": (
            "Undertrained (a 10k-step validation run, final loss 2.35): it generates text such as "
            "'Subtask: …' instead of an 'Action: …|' FAST sequence, so no chunk can be decoded."
        ),
        "description": (
            "Community fine-tune of π0-FAST on gpudad/so101_pick_cube (MuJoCo, same SO-101 model as this "
            "simulator, 3 cameras). It predicts each 10-step chunk as FAST tokens; the decoder you pick "
            "inverts the model's own DCT coefficients, and 'FAST DCT tokens' is its native inverse DCT."
        ),
    },
    {
        "id": "groot_n15",
        "name": "GR00T N1.5",
        "family": "GR00T (NVIDIA)",
        "policy_type": "groot",
        "checkpoint": "nvidia/GR00T-N1.5-3B",
        "parameters": "3B",
        "badge": "Generalist VLA · diffusion",
        "native_output": "continuous",
        "action_head": "Diffusion-transformer action head per embodiment",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "Humanoid GR-1, DROID Franka, AgiBot",
        "unavailable_reason": (
            "Pretrained action heads exist only for GR-1, DROID and AgiBot; SO-101 needs a fine-tuned "
            "embodiment head."
        ),
        "description": "NVIDIA's cross-embodiment foundation model with per-embodiment action heads.",
    },
    {
        "id": "xvla_base",
        "name": "X-VLA base",
        "family": "X-VLA",
        "policy_type": "xvla",
        "checkpoint": "lerobot/xvla-base",
        "parameters": "0.9B",
        "badge": "Generalist VLA · flow",
        "native_output": "continuous",
        "action_head": "Flow matching with soft prompts per embodiment",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "Cross-embodiment, end-effector action space",
        "unavailable_reason": "Acts in end-effector pose space (20-D), not SO-101 joint positions.",
        "description": "Cross-embodiment VLA using learned soft prompts per robot.",
    },
    {
        "id": "wall_oss_flow",
        "name": "WALL-OSS (Qwen2.5-VL)",
        "family": "WALL-OSS (X Square Robot)",
        "policy_type": "wall_x",
        "checkpoint": "lerobot/wall-oss-0.5",
        "parameters": "4B",
        "badge": "Qwen VLA · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching head on Qwen2.5-VL",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": "X Square robots (26-D action layout)",
        "unavailable_reason": "Trained on a 26-D action layout of other robots; SO-101 needs fine-tuning.",
        "description": "Qwen2.5-VL based embodied foundation model with a flow action head.",
    },
    {
        "id": "wall_oss_fast",
        "name": "WALL-OSS-FAST (Qwen2.5-VL)",
        "family": "WALL-OSS (X Square Robot)",
        "policy_type": "wall_x",
        "checkpoint": "x-square-robot/wall-oss-fast",
        "parameters": "4B",
        "badge": "Qwen VLA · FAST DCT tokens",
        "native_output": "dct_tokens",
        "action_head": "Autoregressive FAST DCT tokens on Qwen2.5-VL",
        "decoders": DCT_DECODERS,
        "pretraining": "X Square robots (26-D action layout)",
        "unavailable_reason": (
            "Released only in the original WALL-X format (not a LeRobot checkpoint) and for a "
            "26-D action layout; needs conversion and SO-101 fine-tuning."
        ),
        "description": "Qwen2.5-VL based model that predicts FAST (DCT) action tokens.",
    },
]

LIBERO_PRETRAINING = "Fine-tuned on the LIBERO benchmark (Franka Panda, all four suites)"
LIBERO_INPUTS = "Agent-view and wrist cameras (256×256) + 8-D end-effector state"

LIBERO_SPECS: list[dict[str, Any]] = [
    {
        "id": "smolvla_libero",
        "name": "SmolVLA LIBERO",
        "family": "SmolVLA (Hugging Face)",
        "policy_type": "smolvla",
        "checkpoint": "lerobot/smolvla_libero",
        "parameters": "450M",
        "badge": "VLA · LIBERO tuned · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 50-step chunks of 7-D end-effector deltas",
        "stats": "checkpoint",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": LIBERO_PRETRAINING,
        "description": "SmolVLA fine-tuned by LeRobot on LIBERO.",
    },
    {
        "id": "molmoact2_libero",
        "name": "MolmoAct2 LIBERO",
        "family": "MolmoAct2 (Ai2)",
        "policy_type": "molmoact2",
        "checkpoint": "lerobot/MolmoAct2-LIBERO-LeRobot",
        "parameters": "≈5B",
        "badge": "VLA · LIBERO tuned · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 10-step chunks of 7-D end-effector deltas",
        "stats": "checkpoint",
        "preprocessor_overrides": {
            "rename_observations_processor": {
                "rename_map": {"observation.images.image2": "observation.images.wrist_image"},
            },
        },
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": LIBERO_PRETRAINING,
        "description": "Ai2's MolmoAct2 fine-tuned on LIBERO.",
    },
    {
        "id": "pi05_libero",
        "name": "π0.5 LIBERO",
        "family": "π0.5 (Physical Intelligence)",
        "policy_type": "pi05",
        "checkpoint": "lerobot/pi05_libero_finetuned_v044",
        # Saved with torch.compile enabled, which needs a C compiler at load time.
        "config_overrides": {"compile_model": False},
        "parameters": "≈3B",
        "badge": "VLA · LIBERO tuned · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 50-step chunks of 7-D end-effector deltas",
        "stats": "checkpoint",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": LIBERO_PRETRAINING,
        "description": "π0.5 fine-tuned by LeRobot on LIBERO.",
    },
    {
        "id": "pi0_libero",
        "name": "π0 LIBERO",
        "family": "π0 (Physical Intelligence)",
        "policy_type": "pi0",
        "checkpoint": "lerobot/pi0_libero_finetuned_v044",
        # Saved with torch.compile enabled, which needs a C compiler at load time.
        "config_overrides": {"compile_model": False},
        "parameters": "≈3B",
        "badge": "VLA · LIBERO tuned · flow",
        "native_output": "continuous",
        "action_head": "Flow-matching action expert, 50-step chunks of 7-D end-effector deltas",
        "stats": "checkpoint",
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": LIBERO_PRETRAINING,
        "description": "π0 fine-tuned by LeRobot on LIBERO.",
    },
    {
        "id": "pi0fast_libero",
        "name": "π0-FAST LIBERO",
        "family": "π0-FAST (Physical Intelligence)",
        "policy_type": "pi0_fast",
        "checkpoint": "lerobot/pi0fast-libero",
        "parameters": "≈3B",
        "badge": "VLA · LIBERO tuned · FAST DCT tokens",
        "native_output": "dct_tokens",
        "action_head": "Autoregressive FAST tokens: BPE-compressed, quantized DCT-II coefficients, 10-step chunks",
        "stats": "checkpoint",
        "decoders": DCT_DECODERS,
        "pretraining": LIBERO_PRETRAINING,
        "description": (
            "π0-FAST fine-tuned by LeRobot on LIBERO. It predicts each 10-step chunk as FAST tokens; "
            "the decoder you pick inverts the model's own DCT coefficients."
        ),
    },
    {
        "id": "xvla_libero",
        "name": "X-VLA LIBERO",
        "family": "X-VLA",
        "policy_type": "xvla",
        "checkpoint": "lerobot/xvla-libero",
        "parameters": "0.9B",
        "badge": "VLA · LIBERO tuned · flow",
        "native_output": "continuous",
        "action_head": "Flow matching, 30-step chunks of 20-D end-effector pose actions",
        "stats": "checkpoint",
        # Its LIBERO postprocessor outputs absolute end-effector targets, not deltas.
        "libero_env": {"control_mode": "absolute"},
        "decoders": TIME_DOMAIN_DECODERS,
        "pretraining": LIBERO_PRETRAINING,
        "description": (
            "X-VLA fine-tuned on LIBERO. Its 20-D pose actions (position, 6-D rotation, gripper) are "
            "converted to LIBERO's 7-D controls by X-VLA's own LIBERO processor."
        ),
    },
]
for _spec in MODEL_SPECS:
    _spec["embodiment"] = "so101"
for _spec in LIBERO_SPECS:
    _spec.update(embodiment="libero", inputs=LIBERO_INPUTS)
MODEL_SPECS += LIBERO_SPECS

# A family is one model listed in the studio, with a checkpoint per embodiment it can drive.
MODEL_FAMILIES: list[dict[str, Any]] = [
    {"id": "smolvla", "name": "SmolVLA", "variants": {"libero": "smolvla_libero", "so101": "smolvla_base"}},
    {"id": "molmoact2", "name": "MolmoAct2", "variants": {"libero": "molmoact2_libero", "so101": "molmoact2_so100"}},
    {"id": "pi05", "name": "π0.5", "variants": {"libero": "pi05_libero", "so101": "pi05_base"}},
    {"id": "pi0", "name": "π0", "variants": {"libero": "pi0_libero", "so101": "pi0_base"}},
    {"id": "pi0fast", "name": "π0-FAST", "variants": {"libero": "pi0fast_libero", "so101": "pi0fast_so101_pick_cube"}},
    {"id": "xvla", "name": "X-VLA", "variants": {"libero": "xvla_libero", "so101": "xvla_base"}},
    {"id": "groot", "name": "GR00T N1.5", "variants": {"so101": "groot_n15"}},
    {"id": "wall_oss_flow", "name": "WALL-OSS", "variants": {"so101": "wall_oss_flow"}},
    {"id": "wall_oss_fast", "name": "WALL-OSS-FAST", "variants": {"so101": "wall_oss_fast"}},
]

# Environment each embodiment opens with when a model is selected.
DEFAULT_ARRANGEMENT = {"libero": "libero_object", "so101": "so101_seed_9", "scara": "scara_seed_9"}


def get_model_spec(model_id: str) -> dict[str, Any]:
    for spec in MODEL_SPECS:
        if spec["id"] == model_id:
            return spec
    raise ValueError(f"Unknown VLA model {model_id!r}")


def domain_of(spec: dict[str, Any]) -> str:
    return "frequency_domain" if spec["native_output"] == "dct_tokens" else "time_domain"


def finetuned_spec(base_id: str, model_id: str, name: str, checkpoint: str) -> dict[str, Any]:
    """Spec for a local fine-tune: same wiring as its base, statistics from the checkpoint."""
    base = get_model_spec(base_id)
    return {
        **base,
        "id": model_id,
        "name": name,
        "badge": base["badge"].split(" · ")[0] + " · fine-tuned",
        "checkpoint": checkpoint,
        "stats": "checkpoint",
        "description": f"Local fine-tune of {base['name']}, loaded from {checkpoint}.",
    }


def resolve_spec(model_id: str, checkpoint: Optional[str] = None) -> dict[str, Any]:
    spec = dict(get_model_spec(model_id))
    if checkpoint:
        spec["checkpoint"] = checkpoint
    return spec

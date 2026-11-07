"""Config for the adaptive-DWT policy: π₀-FAST backbone settings + DWT codec selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig


@PreTrainedConfig.register_subclass("adaptive_dwt")
@dataclass
class AdaptiveDWTConfig(PI0FastConfig):
    # "adaptive": DP-oracle targets, model predicts LEN/RES; "fixed": one fixed (L, k) schedule
    dwt_mode: str = "adaptive"
    fixed_plan: list[list[int]] = field(default_factory=list)
    oracle_gamma: float = 0.005
    oracle_beta: float = 0.5
    # trainable adaptation
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

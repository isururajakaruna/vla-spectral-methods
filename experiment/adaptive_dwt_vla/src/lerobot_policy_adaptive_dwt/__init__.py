"""Adaptive-DWT action tokenization policy plugin for LeRobot (π₀-FAST backbone)."""

from .codec import AdaptiveDWTCodec, CodecSpec, enumerate_plans
from .oracle_dp import OracleWeights, solve
from .token_grammar import GrammarError, GrammarFSM, Vocab

__all__ = [
    "AdaptiveDWTCodec",
    "CodecSpec",
    "GrammarError",
    "GrammarFSM",
    "OracleWeights",
    "Vocab",
    "enumerate_plans",
    "solve",
]

try:  # policy classes need torch + lerobot; the codec itself does not
    from .configuration_adaptive_dwt import AdaptiveDWTConfig  # noqa: F401
    from .modeling_adaptive_dwt import AdaptiveDWTPolicy  # noqa: F401

    __all__ += ["AdaptiveDWTConfig", "AdaptiveDWTPolicy"]
except ImportError:  # pragma: no cover
    pass

"""Processor pipeline: π₀-FAST pre/post-processing with the FAST action tokenizer swapped for adaptive DWT.

Action-token id mapping (verified against google/paligemma-3b-pt-224, vocab 257152):
local DWT token t -> PaliGemma id ``257152 - 1 - 128 - t`` (same formula π₀-FAST uses for FAST
tokens). The 302 local tokens land on ``<loc0722>`` .. ``<loc1023>``; these never occur in the
text prompt and differ from BOS(2)/EOS(1)/PAD(0) and the "|" FAST terminator.

Stream layout (teacher forcing and generation are identical):
    [BOS] "Action: " <LEN> <RES> coef... <G0|G1> <GT_i>... <G_END> ... <A_END>   (+ padding, mask=0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from lerobot.processor import ProcessorStepRegistry
from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep

from .codec import AdaptiveDWTCodec
from .oracle_dp import OracleWeights, solve

PALI_VOCAB = 257152
FAST_SKIP = 128
ACTION_PREFIX_TEXT = "Action: "


def local_to_pali(local: np.ndarray | list[int] | int):
    return PALI_VOCAB - 1 - FAST_SKIP - np.asarray(local)


def pali_to_local(pali: np.ndarray | list[int] | int):
    return PALI_VOCAB - 1 - FAST_SKIP - np.asarray(pali)


@dataclass
@ProcessorStepRegistry.register(name="adaptive_dwt_action_tokenizer")
class DWTActionTokenizerStep(ActionTokenizerProcessorStep):
    """Encodes normalized (B, H, 7) actions into DWT token streams (oracle or fixed plan)."""

    codec_state: dict | None = None
    dwt_mode: str = "adaptive"
    fixed_plan: list | None = None
    oracle_gamma: float = 0.005
    oracle_beta: float = 0.5
    _codec: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.codec_state is None:
            raise ValueError("codec_state is required")
        self._codec = AdaptiveDWTCodec.from_dict(self.codec_state)
        self._prefix = self._paligemma_tokenizer.encode(ACTION_PREFIX_TEXT, add_special_tokens=False)
        self.last_plans: list = []

    @property
    def codec(self) -> AdaptiveDWTCodec:
        return self._codec

    def plan_for(self, window: np.ndarray):
        if self.dwt_mode == "fixed":
            return [tuple(p) for p in self.fixed_plan]
        plan, _ = solve(self._codec, window, OracleWeights(gamma=self.oracle_gamma, beta=self.oracle_beta))
        return plan

    def encode_window(self, window: np.ndarray) -> tuple[list[int], list]:
        plan = self.plan_for(window)
        local, _, _ = self._codec.encode_plan(window, plan)
        ids = [self._paligemma_tokenizer.bos_token_id, *self._prefix, *local_to_pali(local).tolist()]
        return ids, plan

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = action.dim() == 2
        a = action.unsqueeze(0) if single else action
        arr = a.detach().to(torch.float64).cpu().numpy()
        toks, masks, self.last_plans = [], [], []
        for w in arr:
            ids, plan = self.encode_window(w)
            self.last_plans.append(plan)
            if len(ids) > self.max_action_tokens:
                raise ValueError(f"DWT stream of {len(ids)} tokens exceeds max_action_tokens={self.max_action_tokens}")
            pad = self.max_action_tokens - len(ids)
            toks.append(ids + [0] * pad)
            masks.append([True] * len(ids) + [False] * pad)
        t = torch.tensor(toks, dtype=torch.long, device=action.device)
        m = torch.tensor(masks, dtype=torch.bool, device=action.device)
        return (t[0], m[0]) if single else (t, m)


def make_processors(config, dataset_stats: dict, action_step: ActionTokenizerProcessorStep | None = None):
    """π₀-FAST processors; optionally replace the action tokenizer step (DWT variants)."""
    from lerobot.policies.pi0_fast.processor_pi0_fast import make_pi0_fast_pre_post_processors

    pre, post = make_pi0_fast_pre_post_processors(config, dataset_stats)
    if action_step is not None:
        idx = [i for i, s in enumerate(pre.steps) if isinstance(s, ActionTokenizerProcessorStep)]
        assert len(idx) == 1, "expected exactly one action tokenizer step"
        pre.steps[idx[0]] = action_step
    return pre, post

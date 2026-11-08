"""Compact stream for a fixed segmentation plan: no per-segment structure tokens.

With a fixed plan the LEN / RES tokens carry no information and the gripper needs only one block per
chunk, so the stream is

    coef... (all segments, plan order) <G0|G1> <GT_i>... <A_END>

with ``GT_i`` a gripper flip at chunk offset i (1..H-1, increasing). Same local ids as
:class:`~.token_grammar.Vocab`; coefficients are those of the wrapped segment codec.
"""

from __future__ import annotations

import numpy as np

from .codec import AdaptiveDWTCodec, Plan
from .token_grammar import GrammarError, Segment, n_coef_tokens


class CompactDWTCodec:
    kind = "compact"

    def __init__(self, seg: AdaptiveDWTCodec, plan: Plan):
        plan = [tuple(int(v) for v in p) for p in plan]
        if sum(length for length, _ in plan) != seg.spec.horizon:
            raise ValueError(f"plan {plan} does not cover H={seg.spec.horizon}")
        self.seg, self.plan = seg, plan
        self.spec, self.vocab, self.quantizer = seg.spec, seg.vocab, seg.quantizer
        self.n_coef_tokens = sum(n_coef_tokens(k, self.spec.n_cont) for _, k in plan)

    def to_dict(self) -> dict:
        return {**self.seg.to_dict(), "kind": self.kind, "plan": [list(p) for p in self.plan],
                "max_stream_tokens": self.max_stream_tokens}

    @classmethod
    def from_dict(cls, d: dict) -> "CompactDWTCodec":
        return cls(AdaptiveDWTCodec.from_dict(d), d["plan"])

    @property
    def max_stream_tokens(self) -> int:
        return self.n_coef_tokens + 1 + (self.spec.horizon - 1) + 1

    def full_plan(self) -> Plan:
        return list(self.plan)

    def _check_plan(self, plan: Plan) -> None:
        if [tuple(p) for p in plan] != self.plan:
            raise ValueError(f"compact codec is fixed to plan {self.plan}, got {plan}")

    def encode_plan(self, window: np.ndarray, plan: Plan) -> tuple[list[int], np.ndarray, dict]:
        self._check_plan(plan)
        v = self.vocab
        toks, recon, info = self.seg.encode_plan(window, self.plan)
        coefs = [b for s in self.seg.segments(toks) for b in s.coef_bins]
        bits = self.seg.gripper_bits(window)
        flips = [i for i in range(1, self.spec.horizon) if bits[i] != bits[i - 1]]
        return [*coefs, v.g1 if bits[0] else v.g0, *(v.gt_token(i) for i in flips), v.a_end], recon, info

    def solve(self, window: np.ndarray, w) -> tuple[Plan, float]:
        return list(self.plan), 0.0

    def segments(self, tokens: list[int]) -> list[Segment]:
        fsm = self.fsm()
        for t in tokens:
            fsm.step(t)
        if not fsm.done:
            raise GrammarError("truncated stream: missing <A_END>")
        g = np.full(self.spec.horizon, fsm.gripper_init, dtype=np.int64)
        for f in fsm.flips:
            g[f:] = 1 - g[f:]
        out, s, c = [], 0, 0
        for length, k in self.plan:
            n = n_coef_tokens(k, self.spec.n_cont)
            seg_g = g[s : s + length]
            out.append(Segment(length, k, fsm.coefs[c : c + n], int(seg_g[0]),
                               [i for i in range(1, length) if seg_g[i] != seg_g[i - 1]]))
            s, c = s + length, c + n
        return out

    def decode(self, tokens: list[int]) -> np.ndarray:
        return self.seg.decode_segments(self.segments(list(tokens)))

    def fsm(self) -> "CompactGrammarFSM":
        return CompactGrammarFSM(self)


class CompactGrammarFSM:
    COEF, GRIP_INIT, GRIP_EVENTS, DONE = range(4)

    def __init__(self, codec: CompactDWTCodec):
        self.vocab, self.horizon = codec.vocab, codec.spec.horizon
        self.state, self.coef_left = self.COEF, codec.n_coef_tokens
        self.coefs: list[int] = []
        self.gripper_init, self.flips, self.last_flip = 0, [], 0
        self._bins = list(range(self.vocab.n_bins))

    @property
    def done(self) -> bool:
        return self.state == self.DONE

    def allowed(self) -> list[int]:
        v = self.vocab
        if self.state == self.COEF:
            return self._bins
        if self.state == self.GRIP_INIT:
            return [v.g0, v.g1]
        if self.state == self.GRIP_EVENTS:
            return [v.gt_token(i) for i in range(self.last_flip + 1, self.horizon)] + [v.a_end]
        return []

    def step(self, tok: int) -> None:
        v, tok = self.vocab, int(tok)
        if self.state == self.DONE:
            raise GrammarError("token after <A_END>")
        if tok not in self.allowed():
            raise GrammarError(f"{v.describe(tok)} not allowed in compact-grammar state {self.state}")
        if self.state == self.COEF:
            self.coefs.append(tok)
            self.coef_left -= 1
            if self.coef_left == 0:
                self.state = self.GRIP_INIT
        elif self.state == self.GRIP_INIT:
            self.gripper_init = int(tok == v.g1)
            self.state = self.GRIP_EVENTS
        elif tok == v.a_end:
            self.state = self.DONE
        else:
            self.last_flip = tok - v.gt_offset + 1
            self.flips.append(self.last_flip)

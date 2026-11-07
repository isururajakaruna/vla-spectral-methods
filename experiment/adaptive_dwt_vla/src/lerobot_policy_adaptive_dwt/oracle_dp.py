"""Dynamic-programming oracle over segmentations/resolutions (training targets only).

objective(plan) = sum over segments of
    alpha * SSE(recon, x)                      (normalized continuous dims)
  + beta  * SSE(diff(recon), diff(x))          (within-segment velocity error)
  + gamma * serialized tokens                  (LEN + RES + coefs + gripper + G_END; A_END is constant)
  + penalty * max(0, max_abs_err - tau)        (soft per-axis max-error constraint)
Gripper events are reproduced exactly by construction. Every term is additive
over segments, so the DP returns the exact optimum over all legal plans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .codec import AdaptiveDWTCodec, Plan
from .token_grammar import segment_token_count

# parts[(s, L, k)] = (sse, vel_sse, tokens, max_abs_err)
Parts = dict[tuple[int, int, int], tuple[float, float, int, float]]


@dataclass(frozen=True)
class OracleWeights:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.02
    tau: float | None = None
    penalty: float = 100.0

    def to_dict(self) -> dict:
        return asdict(self)


def candidate_parts(codec: AdaptiveDWTCodec, window: np.ndarray) -> Parts:
    spec = codec.spec
    window = np.asarray(window, dtype=np.float64)
    cont = window[:, spec.cont_dims]
    bits = codec.gripper_bits(window)
    parts: Parts = {}
    for s in range(spec.horizon):
        for length in spec.lengths:
            if s + length > spec.horizon:
                continue
            seg = cont[s : s + length]
            cand = codec.segment_candidates(seg)
            g = bits[s : s + length]
            flips = int((g[1:] != g[:-1]).sum())
            dx = np.diff(seg, axis=0)
            for k, rec in cand["recons"].items():
                err = rec - seg
                parts[(s, length, k)] = (
                    float((err**2).sum()),
                    float(((np.diff(rec, axis=0) - dx) ** 2).sum()),
                    segment_token_count(k, flips, spec.n_cont),
                    float(np.abs(err).max()),
                )
    return parts


def _cost(p: tuple[float, float, int, float], w: OracleWeights) -> float:
    sse, vel, tokens, max_err = p
    c = w.alpha * sse + w.beta * vel + w.gamma * tokens
    if w.tau is not None:
        c += w.penalty * max(0.0, max_err - w.tau)
    return c


def solve_parts(parts: Parts, horizon: int, lengths: tuple[int, ...], w: OracleWeights) -> tuple[Plan, float]:
    best = np.full(horizon + 1, np.inf)
    back: list[tuple[int, int, int] | None] = [None] * (horizon + 1)
    best[0] = 0.0
    for end in range(1, horizon + 1):
        for length in lengths:
            s = end - length
            if s < 0 or not np.isfinite(best[s]):
                continue
            for k in range(length.bit_length()):
                c = best[s] + _cost(parts[(s, length, k)], w)
                if c < best[end]:
                    best[end] = c
                    back[end] = (s, length, k)
    plan: Plan = []
    end = horizon
    while end > 0:
        s, length, k = back[end]
        plan.append((length, k))
        end = s
    return plan[::-1], float(best[horizon])


def solve(codec: AdaptiveDWTCodec, window: np.ndarray, w: OracleWeights) -> tuple[Plan, float]:
    return solve_parts(candidate_parts(codec, window), codec.spec.horizon, tuple(codec.spec.lengths), w)


def plan_objective(codec: AdaptiveDWTCodec, window: np.ndarray, plan: Plan, w: OracleWeights) -> float:
    parts = candidate_parts(codec, window)
    total, s = 0.0, 0
    for length, k in plan:
        total += _cost(parts[(s, length, k)], w)
        s += length
    return total

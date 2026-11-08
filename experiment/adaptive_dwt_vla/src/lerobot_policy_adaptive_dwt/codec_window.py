"""Whole-window DWT codec with an arbitrary wavelet per level (e.g. ``sym4`` at the finest level, ``haar`` above).

The whole H-step window is transformed at once (no segmentation), so the coefficient count per
resolution depends on the wavelet and the boundary mode rather than being ``2**k``. Stream:

    <RES_k> coef... <G0|G1> <GT_i>... <G_END> <A_END>

using the same local ids as :class:`~.token_grammar.Vocab` (no LEN token). ``RES_k`` keeps the
wavelet groups ``cA_J, cD_J, ..., cD_(J-k+1)``; ``k = J`` is full resolution. A plan is
``[(H, k)]`` so the processor / BPE code paths are shared with the segment codec.

``waves=["dct"]`` uses the orthonormal DCT-II instead (FAST's transform, no BPE): coefficients are grouped
into octave bands (DC, f1, f2–3, f4–7, ...; the last band takes the remainder), which play the role of
wavelet levels for the quantizer and for ``RES_k``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from functools import cached_property

import numpy as np
import pywt

from .codec import CodecSpec, Plan
from .token_grammar import GrammarError, Segment, Vocab


def _forward(x, waves, mode, levels):
    a, details, lens = np.asarray(x, dtype=np.float64), [], []
    for j in range(levels):
        lens.append(len(a))
        a, d = pywt.dwt(a, waves[min(j, len(waves) - 1)], mode=mode)
        details.append(d)
    return [a] + details[::-1], lens


def _inverse(cs, waves, mode, lens):
    levels = len(lens)
    a = cs[0]
    for j in reversed(range(levels)):
        a = pywt.idwt(a, cs[levels - j], waves[min(j, len(waves) - 1)], mode=mode)[: lens[j]]
    return a


def dct_bands(horizon: int) -> list[int]:
    sizes, n = [1, 1], 2
    while n + 2 * sizes[-1] <= horizon:
        sizes.append(2 * sizes[-1])
        n += sizes[-1]
    sizes[-1] += horizon - n
    return sizes


def dct_matrices(horizon: int):
    k, n = np.arange(horizon)[:, None], np.arange(horizon)[None, :]
    A = np.sqrt(2 / horizon) * np.cos(np.pi * (2 * n + 1) * k / (2 * horizon))
    A[0] /= np.sqrt(2)
    return A, A.T.copy(), dct_bands(horizon)


def transform_matrices(waves, mode, horizon, levels):
    """Analysis A (n_coef, H), synthesis S (H, n_coef), coefficient count per group (cA_J, cD_J, ..., cD_1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols, shapes, lens = [], None, None
        for e in np.eye(horizon):
            cs, lens = _forward(e, waves, mode, levels)
            shapes = [len(c) for c in cs]
            cols.append(np.concatenate(cs))
        A = np.stack(cols, axis=1)
        n = A.shape[0]
        S = np.zeros((horizon, n))
        offs = np.cumsum([0] + shapes)
        for i in range(n):
            u = np.zeros(n)
            u[i] = 1
            S[:, i] = _inverse([u[offs[g]:offs[g + 1]] for g in range(len(shapes))], waves, mode, lens)
    return A, S, shapes


class WindowDWTCodec:
    kind = "window"

    def __init__(self, spec: CodecSpec, waves: list[str], mode: str, levels: int,
                 center: np.ndarray | None = None, step: np.ndarray | None = None):
        self.spec, self.waves, self.mode, self.levels = spec, list(waves), mode, levels
        self.vocab = Vocab(n_bins=spec.n_bins, max_len=32)
        if levels > 5 or spec.horizon > self.vocab.max_len:
            raise ValueError("levels must be <= 5 and horizon <= 32")
        if self.waves == ["dct"]:
            self.A, self.S, self.group_sizes = dct_matrices(spec.horizon)
            if len(self.group_sizes) - 1 != levels:
                raise ValueError(f"DCT at H={spec.horizon} has {len(self.group_sizes) - 1} levels, got {levels}")
        else:
            self.A, self.S, self.group_sizes = transform_matrices(self.waves, mode, spec.horizon, levels)
        if np.abs(self.S @ self.A - np.eye(spec.horizon)).max() > 1e-8:
            raise ValueError(f"{waves}/{mode} is not perfectly reconstructing at H={spec.horizon}")
        self.n_coef = self.A.shape[0]
        self.groups = np.concatenate([np.full(s, g) for g, s in enumerate(self.group_sizes)])
        self.center, self.step = center, step  # (n_coef, D); None => unfitted
        self.mid = spec.n_bins // 2 - 1

    @property
    def name(self) -> str:
        return "mix:" + "→".join(self.waves) if len(self.waves) > 1 else self.waves[0]

    def coefs_at(self, k: int) -> int:
        return int(sum(self.group_sizes[: k + 1]))

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, spec: CodecSpec, train_windows: np.ndarray, waves: list[str], mode: str = "symmetric",
            levels: int = 3) -> "WindowDWTCodec":
        c = cls(spec, waves, mode, levels)
        x = np.asarray(train_windows, dtype=np.float64)[..., spec.cont_dims]
        coefs = np.einsum("ch,nhd->ncd", c.A, x)
        n_g, D = len(c.group_sizes), x.shape[-1]
        center, step = np.zeros((n_g, D)), np.zeros((n_g, D))
        for g in range(n_g):
            v = coefs[:, c.groups == g].reshape(-1, D)
            if g == 0:
                lo, hi = v.min(0), v.max(0)
                center[g], half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            else:
                half = np.abs(v).max(0)
            step[g] = np.maximum(half, 1e-6) / c.mid
        c.center, c.step = center[c.groups], step[c.groups]
        return c

    def to_dict(self) -> dict:
        return {"kind": self.kind, "spec": asdict(self.spec), "waves": self.waves, "mode": self.mode,
                "levels": self.levels, "center": self.center.tolist(), "step": self.step.tolist(),
                "vocab_size": self.vocab.size, "max_stream_tokens": self.max_stream_tokens}

    @classmethod
    def from_dict(cls, d: dict) -> "WindowDWTCodec":
        spec = dict(d["spec"])
        spec["lengths"] = tuple(spec["lengths"])
        return cls(CodecSpec(**spec), d["waves"], d["mode"], d["levels"], np.array(d["center"]), np.array(d["step"]))

    @property
    def max_stream_tokens(self) -> int:
        return 1 + self.n_coef * self.spec.n_cont + 1 + (self.spec.horizon - 1) + 1 + 1

    def full_plan(self) -> Plan:
        return [(self.spec.horizon, self.levels)]

    # ----------------------------------------------------------------- gripper
    def gripper_bits(self, window: np.ndarray) -> np.ndarray:
        return (np.asarray(window)[:, self.spec.gripper_dim] > self.spec.gripper_threshold).astype(np.int64)

    def gripper_values(self, bits: np.ndarray) -> np.ndarray:
        return np.where(bits > 0, self.spec.gripper_high, self.spec.gripper_low)

    # ------------------------------------------------------------------- codec
    def quantize(self, coefs: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((coefs - self.center) / self.step) + self.mid, 0, self.spec.n_bins - 1).astype(np.int64)

    def dequantize(self, bins: np.ndarray, n: int) -> np.ndarray:
        return self.center[:n] + (np.asarray(bins, dtype=np.float64) - self.mid) * self.step[:n]

    def candidates(self, window: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """k -> (bins (n_k, D), reconstruction (H, D) of the continuous dims)."""
        cont = np.asarray(window, dtype=np.float64)[:, self.spec.cont_dims]
        bins = self.quantize(self.A @ cont)
        out = {}
        for k in range(self.levels + 1):
            n = self.coefs_at(k)
            out[k] = (bins[:n], self.S[:, :n] @ self.dequantize(bins[:n], n))
        return out

    def token_count(self, k: int, n_flips: int) -> int:
        return 1 + self.coefs_at(k) * self.spec.n_cont + 1 + n_flips + 1 + 1

    def encode_plan(self, window: np.ndarray, plan: Plan) -> tuple[list[int], np.ndarray, dict]:
        spec, v = self.spec, self.vocab
        window = np.asarray(window, dtype=np.float64)
        (length, k), = plan
        if length != spec.horizon or not 0 <= k <= self.levels:
            raise ValueError(f"illegal window plan {plan}")
        bins, rec = self.candidates(window)[k]
        bits = self.gripper_bits(window)
        flips = [int(i) for i in range(1, spec.horizon) if bits[i] != bits[i - 1]]
        toks = [v.res_token(k), *bins.reshape(-1).tolist(), v.g1 if bits[0] else v.g0,
                *(v.gt_token(i) for i in flips), v.g_end, v.a_end]
        recon = np.zeros_like(window)
        recon[:, spec.cont_dims] = rec
        recon[:, spec.gripper_dim] = self.gripper_values(bits)
        return toks, recon, {"clipped": 0, "coefs": int(bins.size)}

    def solve(self, window: np.ndarray, w) -> tuple[Plan, float]:
        """Per-window depth with the segment oracle's objective (alpha SSE + beta velocity SSE + gamma tokens)."""
        window = np.asarray(window, dtype=np.float64)
        x = window[:, self.spec.cont_dims]
        bits = self.gripper_bits(window)
        n_flips = int((bits[1:] != bits[:-1]).sum())
        best = (None, np.inf)
        for k, (_, rec) in self.candidates(window).items():
            c = (w.alpha * ((rec - x) ** 2).sum() + w.beta * ((np.diff(rec, axis=0) - np.diff(x, axis=0)) ** 2).sum()
                 + w.gamma * self.token_count(k, n_flips))
            if w.tau is not None:
                c += w.penalty * max(0.0, np.abs(rec - x).max() - w.tau)
            if c < best[1]:
                best = ([(self.spec.horizon, k)], float(c))
        return best

    def segments(self, tokens: list[int]) -> list[Segment]:
        fsm = self.fsm()
        for t in tokens:
            fsm.step(t)
        if not fsm.done:
            raise GrammarError("truncated stream: missing <A_END>")
        return [fsm.segment]

    def decode(self, tokens: list[int]) -> np.ndarray:
        seg = self.segments(list(tokens))[0]
        spec = self.spec
        n = self.coefs_at(seg.res)
        bins = np.asarray(seg.coef_bins, dtype=np.int64).reshape(n, spec.n_cont)
        out = np.zeros((spec.horizon, spec.action_dim))
        out[:, spec.cont_dims] = self.S[:, :n] @ self.dequantize(bins, n)
        g = np.full(spec.horizon, seg.gripper_init, dtype=np.int64)
        for f in seg.flips:
            g[f:] = 1 - g[f:]
        out[:, spec.gripper_dim] = self.gripper_values(g)
        return out

    def fsm(self) -> "WindowGrammarFSM":
        return WindowGrammarFSM(self)


class WindowGrammarFSM:
    RES, COEF, GRIP_INIT, GRIP_EVENTS, END, DONE = range(6)

    def __init__(self, codec: WindowDWTCodec):
        self.codec, self.vocab, self.horizon = codec, codec.vocab, codec.spec.horizon
        self.state, self.coef_left, self.last_flip = self.RES, 0, 0
        self.segment: Segment | None = None

    @property
    def done(self) -> bool:
        return self.state == self.DONE

    @cached_property
    def _bins(self) -> list[int]:
        return list(range(self.vocab.n_bins))

    def allowed(self) -> list[int]:
        v = self.vocab
        if self.state == self.RES:
            return [v.res_token(k) for k in range(self.codec.levels + 1)]
        if self.state == self.COEF:
            return self._bins
        if self.state == self.GRIP_INIT:
            return [v.g0, v.g1]
        if self.state == self.GRIP_EVENTS:
            return [v.gt_token(i) for i in range(self.last_flip + 1, self.horizon)] + [v.g_end]
        if self.state == self.END:
            return [v.a_end]
        return []

    def step(self, tok: int) -> None:
        v, tok = self.vocab, int(tok)
        if self.state == self.DONE:
            raise GrammarError("token after <A_END>")
        if tok not in self.allowed():
            raise GrammarError(f"{v.describe(tok)} not allowed in window-grammar state {self.state}")
        if self.state == self.RES:
            k = tok - v.res_offset
            self.segment = Segment(self.horizon, k, [], 0)
            self.coef_left = self.codec.coefs_at(k) * self.codec.spec.n_cont
            self.state = self.COEF
        elif self.state == self.COEF:
            self.segment.coef_bins.append(tok)
            self.coef_left -= 1
            if self.coef_left == 0:
                self.state = self.GRIP_INIT
        elif self.state == self.GRIP_INIT:
            self.segment.gripper_init = int(tok == v.g1)
            self.state = self.GRIP_EVENTS
        elif self.state == self.GRIP_EVENTS:
            if tok == v.g_end:
                self.state = self.END
            else:
                self.last_flip = tok - v.gt_offset + 1
                self.segment.flips.append(self.last_flip)
        else:
            self.state = self.DONE

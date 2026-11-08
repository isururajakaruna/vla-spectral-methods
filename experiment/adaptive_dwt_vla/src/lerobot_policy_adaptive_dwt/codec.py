"""Adaptive Haar-DWT action codec operating on *normalized* action windows.

A window is (H, 7): 6 continuous end-effector dims + 1 binary gripper dim.
A plan is a list of ``(L, k)`` segments covering H exactly; segment length L
is a power of two, k in 0..log2 L keeps the 2**k coarsest Haar coefficients
per continuous dimension. The gripper is encoded exactly as an initial state
plus in-segment flip offsets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .codec_basis import MODE, WAVELET, haar_matrix
from .quantizer import CoefficientQuantizer
from .token_grammar import (
    GrammarFSM,
    Segment,
    Vocab,
    coverable,
    max_stream_tokens,
    parse,
    segment_token_count,
    serialize,
)

Plan = list[tuple[int, int]]


@dataclass(frozen=True)
class CodecSpec:
    horizon: int = 10
    lengths: tuple[int, ...] = (2, 4, 8)
    action_dim: int = 7
    gripper_dim: int = 6
    n_bins: int = 256
    quantile: float = 1.0
    # normalized gripper values for binary state 0 / 1 and the decision threshold between them
    gripper_low: float = -1.0
    gripper_high: float = 1.0

    @property
    def n_cont(self) -> int:
        return self.action_dim - 1

    @property
    def cont_dims(self) -> list[int]:
        return [d for d in range(self.action_dim) if d != self.gripper_dim]

    @property
    def gripper_threshold(self) -> float:
        return 0.5 * (self.gripper_low + self.gripper_high)


class AdaptiveDWTCodec:
    def __init__(self, spec: CodecSpec, quantizer: CoefficientQuantizer | None = None):
        if spec.horizon not in coverable(spec.horizon, tuple(spec.lengths)):
            raise ValueError(f"H={spec.horizon} is not exactly coverable by {spec.lengths}")
        self.spec = spec
        self.quantizer = quantizer  # None => lossless (float coefficients) test mode
        self.vocab = Vocab(n_bins=spec.n_bins, max_len=32)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, spec: CodecSpec, train_windows: np.ndarray, **quantizer_kw) -> "AdaptiveDWTCodec":
        x = np.asarray(train_windows, dtype=np.float64)[..., spec.cont_dims]
        q = CoefficientQuantizer(spec.n_bins, spec.quantile, **quantizer_kw).fit(x, spec.lengths)
        return cls(spec, q)

    def to_dict(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "wavelet": WAVELET,
            "mode": MODE,
            "quantizer": None if self.quantizer is None else self.quantizer.to_dict(),
            "vocab_size": self.vocab.size,
            "max_stream_tokens": self.max_stream_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveDWTCodec":
        spec = dict(d["spec"])
        spec["lengths"] = tuple(spec["lengths"])
        q = None if d["quantizer"] is None else CoefficientQuantizer.from_dict(d["quantizer"])
        return cls(CodecSpec(**spec), q)

    @property
    def max_stream_tokens(self) -> int:
        return max_stream_tokens(self.spec.horizon, tuple(self.spec.lengths), self.spec.n_cont)

    def fsm(self) -> GrammarFSM:
        return GrammarFSM(self.vocab, self.spec.horizon, tuple(self.spec.lengths), self.spec.n_cont)

    # ----------------------------------------------------------------- gripper
    def gripper_bits(self, window: np.ndarray) -> np.ndarray:
        return (np.asarray(window)[:, self.spec.gripper_dim] > self.spec.gripper_threshold).astype(np.int64)

    def gripper_values(self, bits: np.ndarray) -> np.ndarray:
        return np.where(bits > 0, self.spec.gripper_high, self.spec.gripper_low)

    # ------------------------------------------------------------- segment I/O
    def segment_candidates(self, cont: np.ndarray) -> dict:
        """For a (L, D) continuous segment return bins/clips and reconstructions for every k."""
        length = cont.shape[0]
        basis = haar_matrix(length)
        coefs = basis @ cont  # (L, D)
        if self.quantizer is None:
            bins, clipped, deq = None, np.zeros_like(coefs, dtype=bool), coefs
        else:
            bins, clipped = self.quantizer.quantize(coefs, length)
            deq = self.quantizer.dequantize(bins, length)
        recons = {}
        for k in range(length.bit_length()):
            n = 2**k
            recons[k] = basis[:n].T @ deq[:n]
        return {"coefs": coefs, "bins": bins, "clipped": clipped, "recons": recons}

    def encode_plan(self, window: np.ndarray, plan: Plan) -> tuple[list[int], np.ndarray, dict]:
        """Serialize ``window`` (H, 7) with a given segmentation/resolution plan."""
        spec = self.spec
        window = np.asarray(window, dtype=np.float64)
        if window.shape != (spec.horizon, spec.action_dim):
            raise ValueError(f"window shape {window.shape} != ({spec.horizon}, {spec.action_dim})")
        if sum(length for length, _ in plan) != spec.horizon:
            raise ValueError(f"plan {plan} does not cover H={spec.horizon}")
        if self.quantizer is None:
            raise RuntimeError("token serialization requires a fitted quantizer")
        cont = window[:, spec.cont_dims]
        bits = self.gripper_bits(window)
        recon = np.zeros_like(window)
        segments: list[Segment] = []
        n_clipped = n_coefs = 0
        s = 0
        for length, k in plan:
            if length not in spec.lengths or not 0 <= k <= length.bit_length() - 1:
                raise ValueError(f"illegal segment (L={length}, k={k})")
            cand = self.segment_candidates(cont[s : s + length])
            n = 2**k
            seg_bins = cand["bins"][:n]  # (n, D)
            n_clipped += int(cand["clipped"][:n].sum())
            n_coefs += seg_bins.size
            g = bits[s : s + length]
            flips = [int(i) for i in range(1, length) if g[i] != g[i - 1]]
            segments.append(Segment(length, k, seg_bins.reshape(-1).tolist(), int(g[0]), flips))
            recon[s : s + length, spec.cont_dims] = cand["recons"][k]
            recon[s : s + length, spec.gripper_dim] = self.gripper_values(g)
            s += length
        tokens = serialize(segments, self.vocab)
        return tokens, recon, {"clipped": n_clipped, "coefs": n_coefs}

    def segments(self, tokens: list[int]) -> list[Segment]:
        return parse(list(tokens), self.vocab, self.spec.horizon, tuple(self.spec.lengths), self.spec.n_cont)

    def decode(self, tokens: list[int]) -> np.ndarray:
        """Strictly parse and reconstruct a (H, 7) normalized action window."""
        return self.decode_segments(self.segments(tokens))

    def decode_segments(self, segments: list[Segment]) -> np.ndarray:
        spec = self.spec
        out = np.zeros((spec.horizon, spec.action_dim))
        s = 0
        for seg in segments:
            n = 2**seg.res
            bins = np.asarray(seg.coef_bins, dtype=np.int64).reshape(n, spec.n_cont)
            deq = self.quantizer.dequantize(bins, seg.length, n)
            basis = haar_matrix(seg.length)
            out[s : s + seg.length, spec.cont_dims] = basis[:n].T @ deq
            g = np.full(seg.length, seg.gripper_init, dtype=np.int64)
            for f in seg.flips:
                g[f:] = 1 - g[f:]
            out[s : s + seg.length, spec.gripper_dim] = self.gripper_values(g)
            s += seg.length
        return out

    def reconstruct_lossless(self, window: np.ndarray, plan: Plan) -> np.ndarray:
        """Float (unquantized) reconstruction, used for the no-quantization test mode."""
        spec = self.spec
        window = np.asarray(window, dtype=np.float64)
        cont = window[:, spec.cont_dims]
        out = np.zeros_like(window)
        s = 0
        for length, k in plan:
            basis = haar_matrix(length)
            n = 2**k
            out[s : s + length, spec.cont_dims] = basis[:n].T @ (basis[:n] @ cont[s : s + length])
            s += length
        out[:, spec.gripper_dim] = self.gripper_values(self.gripper_bits(window))
        return out

    def plan_token_count(self, window: np.ndarray, plan: Plan) -> int:
        bits = self.gripper_bits(window)
        total, s = 1, 0  # A_END
        for length, k in plan:
            g = bits[s : s + length]
            total += segment_token_count(k, int((g[1:] != g[:-1]).sum()), self.spec.n_cont)
            s += length
        return total


def enumerate_plans(horizon: int, lengths: tuple[int, ...], fixed_k: int | None = None) -> list[Plan]:
    """All ordered exact covers of ``horizon`` with every legal resolution (tests / fixed search)."""
    out: list[Plan] = []

    def rec(rem: int, acc: Plan) -> None:
        if rem == 0:
            out.append(list(acc))
            return
        for length in lengths:
            if length <= rem:
                ks = range(length.bit_length()) if fixed_k is None else [min(fixed_k, length.bit_length() - 1)]
                for k in ks:
                    acc.append((length, k))
                    rec(rem - length, acc)
                    acc.pop()

    rec(horizon, [])
    return out

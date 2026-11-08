"""Per-(segment length, wavelet group, dimension) scalar coefficient quantizer.

Each coefficient type has a fixed ``center`` and ``step``; bins are
``i = clip(round((x - center) / step) + mid, 0, n_bins - 1)`` and
``x_hat = center + (i - mid) * step`` with ``mid = n_bins // 2 - 1`` so that
``x == center`` is exactly representable (detail groups use ``center = 0``,
so zero detail is reproduced exactly). Calibrated on training windows only.

Optional coarse detail quantization (defaults reproduce the plain quantizer): detail group g of J gets
its step multiplied by ``detail_scale`` ("uniform") or ``detail_scale ** (g / J)`` ("fine": the finest
details are the coarsest), and detail values with ``|x| / step < 0.5 + deadzone`` round to zero.
"""

from __future__ import annotations

import numpy as np

from .codec_basis import group_of, haar_matrix


class CoefficientQuantizer:
    def __init__(self, n_bins: int = 256, quantile: float = 0.999, detail_scale: float = 1.0,
                 detail_scheme: str = "uniform", deadzone: float = 0.0):
        if detail_scheme not in ("uniform", "fine"):
            raise ValueError(f"unknown detail_scheme {detail_scheme!r}")
        self.n_bins = n_bins
        self.mid = n_bins // 2 - 1
        self.quantile = quantile
        self.detail_scale, self.detail_scheme, self.deadzone = detail_scale, detail_scheme, deadzone
        # params[L] = (center[n_groups, D], step[n_groups, D])
        self.params: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, continuous_windows: np.ndarray, lengths: tuple[int, ...]) -> "CoefficientQuantizer":
        """``continuous_windows``: (N, H, D) normalized continuous actions from the training split."""
        x = np.asarray(continuous_windows, dtype=np.float64)
        n, horizon, _ = x.shape
        q = self.quantile
        for length in lengths:
            basis = haar_matrix(length)
            segs = [x[:, s : s + length] for s in range(0, horizon - length + 1)]
            coefs = np.einsum("cl,nld->ncd", basis, np.concatenate(segs, axis=0))  # (M, L, D)
            groups = group_of(length)
            n_groups = int(groups.max()) + 1
            center = np.zeros((n_groups, coefs.shape[-1]))
            step = np.zeros_like(center)
            for g in range(n_groups):
                vals = coefs[:, groups == g, :].reshape(-1, coefs.shape[-1])
                if g == 0:
                    lo, hi = np.quantile(vals, 1 - q, axis=0), np.quantile(vals, q, axis=0)
                    center[g] = 0.5 * (lo + hi)
                    half = 0.5 * (hi - lo)
                else:
                    half = np.quantile(np.abs(vals), q, axis=0)
                step[g] = np.maximum(half, 1e-6) / self.mid
                if g > 0:
                    step[g] *= self.detail_scale ** (g / (n_groups - 1) if self.detail_scheme == "fine" else 1.0)
            self.params[length] = (center, step)
        return self

    def _expand(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        center, step = self.params[length]
        groups = group_of(length)
        return center[groups], step[groups]  # (L, D)

    def quantize(self, coefs: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
        """coefs (L, D) -> (bins int (L, D), clipped bool (L, D))."""
        center, step = self._expand(length)
        u = (coefs - center) / step
        r = np.rint(u)
        if self.deadzone:
            detail = np.broadcast_to((group_of(length) > 0)[:, None], u.shape)
            r = np.where(detail & (np.abs(u) < 0.5 + self.deadzone), 0.0, r)
        raw = r + self.mid
        clipped = (raw < 0) | (raw > self.n_bins - 1)
        return np.clip(raw, 0, self.n_bins - 1).astype(np.int64), clipped

    def dequantize(self, bins: np.ndarray, length: int, n_coefs: int | None = None) -> np.ndarray:
        """bins (n, D) for the first n coefficients of an L-segment."""
        center, step = self._expand(length)
        n = bins.shape[0] if n_coefs is None else n_coefs
        return center[:n] + (np.asarray(bins, dtype=np.float64) - self.mid) * step[:n]

    def to_dict(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "quantile": self.quantile,
            "detail_scale": self.detail_scale,
            "detail_scheme": self.detail_scheme,
            "deadzone": self.deadzone,
            "params": {
                str(length): {"center": c.tolist(), "step": s.tolist()} for length, (c, s) in self.params.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoefficientQuantizer":
        q = cls(n_bins=d["n_bins"], quantile=d["quantile"], detail_scale=d.get("detail_scale", 1.0),
                detail_scheme=d.get("detail_scheme", "uniform"), deadzone=d.get("deadzone", 0.0))
        q.params = {
            int(length): (np.array(p["center"], dtype=np.float64), np.array(p["step"], dtype=np.float64))
            for length, p in d["params"].items()
        }
        return q

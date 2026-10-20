"""Pluggable frequency-to-time reconstruction (Inverse Spectrum Module).

Supports direct time-domain pass-through, standard FAST orthogonal DCT-II/IDCT-III
with quantization, band-limited IDCT (rectangular, Hann, Butterworth windows),
Inverse Real FFT (IFFT), Discrete Wavelet Transform (Haar / Daubechies-4 lifting/filtering),
and our proposed Channel-Decoupled + C1 Boundary-Constrained Custom Hybrid Spectrum Schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Optional, Protocol, Sequence

import numpy as np


WindowType = Literal["rectangular", "hann", "butterworth"]
SchemaName = Literal[
    "direct",
    "fast_idct",
    "band_idct",
    "ifft",
    "idwt_haar",
    "idwt_db4",
    "custom_hybrid",
]


def dct_ii_matrix(n: int) -> np.ndarray:
    """Construct an orthonormal Type-II DCT transform matrix of size (n, n)."""
    if n < 1:
        raise ValueError("DCT size must be positive")
    k = np.arange(n, dtype=float)[:, None]
    i = np.arange(n, dtype=float)[None, :]
    matrix = np.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    matrix[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        matrix[1:, :] *= math.sqrt(2.0 / n)
    return matrix


def dct_ii(x: np.ndarray) -> np.ndarray:
    """Apply orthonormal Type-II DCT along axis 0 of a 2D array (H, D)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (H, D) for DCT-II")
    basis = dct_ii_matrix(arr.shape[0])
    return basis @ arr


def idct_iii(coeffs: np.ndarray) -> np.ndarray:
    """Apply orthonormal Type-III Inverse DCT along axis 0 of a 2D array (H, D)."""
    arr = np.asarray(coeffs, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (H, D) for IDCT-III")
    basis = dct_ii_matrix(arr.shape[0])
    return basis.T @ arr


def build_spectral_window(
    num_bins: int,
    freq_band: tuple[float, float] = (0.0, 1.0),
    window: WindowType = "rectangular",
    butterworth_order: int = 3,
) -> np.ndarray:
    """Create a 1D spectral gain mask over normalized frequencies in [0, 1]."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    f_low, f_high = float(freq_band[0]), float(freq_band[1])
    if not (0.0 <= f_low <= f_high <= 1.0) or f_high <= 0.0:
        raise ValueError(f"Invalid normalized freq_band {freq_band}; expected 0 <= low <= high <= 1")
    if num_bins == 1:
        return np.ones(1, dtype=float)

    norm_freqs = np.linspace(0.0, 1.0, num_bins)
    mask = np.zeros(num_bins, dtype=float)
    in_band = (norm_freqs >= f_low - 1e-12) & (norm_freqs <= f_high + 1e-12)

    if window == "rectangular":
        mask[in_band] = 1.0
    elif window == "hann":
        for idx, f in enumerate(norm_freqs):
            if f < f_low or f > f_high:
                mask[idx] = 0.0
            elif f_high > f_low:
                rel = (f - f_low) / (f_high - f_low)
                # Cosine roll-off from 1 at f_low to 0 at f_high
                mask[idx] = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, rel))))
            else:
                mask[idx] = 1.0
    elif window == "butterworth":
        cutoff = max(f_high, 1e-6)
        gain = 1.0 / np.sqrt(1.0 + (norm_freqs / cutoff) ** (2 * butterworth_order))
        gain[norm_freqs < f_low - 1e-12] = 0.0
        mask = gain
    else:
        raise ValueError(f"Unsupported spectral window {window!r}")

    # Always preserve the DC component when f_low == 0.0
    if f_low == 0.0:
        mask[0] = 1.0
    return mask


# Daubechies-4 (db4) orthogonal low-pass scaling filter h and high-pass wavelet filter g
_DB4_H = np.asarray(
    [
        (1.0 + math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
        (3.0 + math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
        (3.0 - math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
        (1.0 - math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
    ],
    dtype=float,
)
_DB4_G = np.asarray([_DB4_H[3], -_DB4_H[2], _DB4_H[1], -_DB4_H[0]], dtype=float)

_HAAR_H = np.asarray([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=float)
_HAAR_G = np.asarray([1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)], dtype=float)


def _dwt_single_level(signal: np.ndarray, h: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic 1-level 1D DWT along axis 0 for an even-length 2D array (N, D)."""
    n, d = signal.shape
    half = n // 2
    approx = np.zeros((half, d), dtype=float)
    detail = np.zeros((half, d), dtype=float)
    flen = len(h)
    for k in range(half):
        for m in range(flen):
            idx = (2 * k + m) % n
            approx[k] += h[m] * signal[idx]
            detail[k] += g[m] * signal[idx]
    return approx, detail


def _idwt_single_level(
    approx: np.ndarray, detail: np.ndarray, h: np.ndarray, g: np.ndarray
) -> np.ndarray:
    """Exact inverse of `_dwt_single_level` using transpose periodic convolution."""
    half, d = approx.shape
    n = half * 2
    out = np.zeros((n, d), dtype=float)
    flen = len(h)
    for k in range(half):
        for m in range(flen):
            idx = (2 * k + m) % n
            out[idx] += h[m] * approx[k] + g[m] * detail[k]
    return out


class InverseSpectrumModule(Protocol):
    """Protocol for pluggable frequency-to-time reconstruction modules."""

    schema_name: str
    fast_tokenization: bool
    freq_band: tuple[float, float]

    def encode(self, time_chunk: np.ndarray) -> dict[str, Any]:
        """Encode a time-domain chunk (H, D) into a spectral payload."""
        ...

    def decode(
        self,
        spectral_payload: dict[str, Any] | np.ndarray,
        prev_chunk_tail: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct a time-domain chunk (H, D) from a spectral payload."""
        ...


@dataclass
class DirectTimeDomainModule:
    """Identity pass-through when `fast_tokenization=False`."""

    schema_name: str = "direct"
    fast_tokenization: bool = False
    freq_band: tuple[float, float] = (0.0, 1.0)

    def encode(self, time_chunk: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(time_chunk, dtype=float)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("Expected finite 2D action chunk (H, D)")
        return {
            "schema": self.schema_name,
            "shape": list(arr.shape),
            "actions": arr.copy(),
            "non_zero_ratio": 1.0,
        }

    def decode(
        self,
        spectral_payload: dict[str, Any] | np.ndarray,
        prev_chunk_tail: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(spectral_payload, np.ndarray):
            arr = np.asarray(spectral_payload, dtype=float)
        else:
            arr = np.asarray(spectral_payload["actions"], dtype=float)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("Invalid time-domain action chunk")
        return arr.copy()


@dataclass
class FASTIDCTModule:
    """Standard FAST tokenization codec (Orthogonal DCT-II + quantization + IDCT-III)."""

    quant_scale: float = 20.0
    freq_band: tuple[float, float] = (0.0, 1.0)
    window: WindowType = "rectangular"
    schema_name: str = "fast_idct"
    fast_tokenization: bool = True

    def encode(self, time_chunk: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(time_chunk, dtype=float)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("Expected finite 2D action chunk (H, D)")
        # Center around initial frame of the chunk to avoid large DC quantization error
        offset = arr[0:1, :].copy()
        centered = arr - offset
        coeffs = dct_ii(centered)
        mask = build_spectral_window(arr.shape[0], self.freq_band, self.window)[:, None]
        filtered = coeffs * mask
        if self.quant_scale > 0:
            tokens = np.rint(filtered * self.quant_scale).astype(int)
        else:
            tokens = filtered
        non_zero = float(np.count_nonzero(tokens)) / float(tokens.size) if tokens.size else 0.0
        return {
            "schema": self.schema_name,
            "shape": list(arr.shape),
            "offset": offset,
            "dct_tokens": tokens,
            "dct_coeffs": filtered,
            "quant_scale": self.quant_scale,
            "freq_band": list(self.freq_band),
            "non_zero_ratio": non_zero,
        }

    def decode(
        self,
        spectral_payload: dict[str, Any] | np.ndarray,
        prev_chunk_tail: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(spectral_payload, np.ndarray):
            coeffs = np.asarray(spectral_payload, dtype=float)
            mask = build_spectral_window(coeffs.shape[0], self.freq_band, self.window)[:, None]
            return idct_iii(coeffs * mask)

        scale = float(spectral_payload.get("quant_scale", self.quant_scale))
        if "dct_tokens" in spectral_payload and scale > 0:
            coeffs = np.asarray(spectral_payload["dct_tokens"], dtype=float) / scale
        else:
            coeffs = np.asarray(spectral_payload["dct_coeffs"], dtype=float)
        mask = build_spectral_window(coeffs.shape[0], self.freq_band, self.window)[:, None]
        reconstructed = idct_iii(coeffs * mask)
        if "offset" in spectral_payload:
            reconstructed = reconstructed + np.asarray(spectral_payload["offset"], dtype=float)
        return reconstructed


@dataclass
class BandLimitedIDCTModule(FASTIDCTModule):
    """FAST DCT codec with explicit frequency band truncation or Butterworth/Hann smoothing."""

    quant_scale: float = 50.0
    freq_band: tuple[float, float] = (0.0, 0.4)
    window: WindowType = "butterworth"
    schema_name: str = "band_idct"
    fast_tokenization: bool = True


@dataclass
class IFFTInverseSpectrumModule:
    """Fourier-domain (RFFT / IRFFT) action chunk encoder and band-limited decoder."""

    freq_band: tuple[float, float] = (0.0, 0.45)
    window: WindowType = "butterworth"
    quant_scale: float = 50.0
    schema_name: str = "ifft"
    fast_tokenization: bool = True

    def encode(self, time_chunk: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(time_chunk, dtype=float)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("Expected finite 2D action chunk (H, D)")
        h, d = arr.shape
        # Remove linear trend between start and end of chunk to eliminate periodic wrap-around jump
        start = arr[0:1, :]
        end = arr[-1:, :]
        t = np.linspace(0.0, 1.0, h)[:, None]
        trend = start + t * (end - start) if h > 1 else start.copy()
        residual = arr - trend

        spectrum = np.fft.rfft(residual, axis=0)
        mask = build_spectral_window(spectrum.shape[0], self.freq_band, self.window)[:, None]
        filtered = spectrum * mask
        if self.quant_scale > 0:
            real_q = np.rint(filtered.real * self.quant_scale) / self.quant_scale
            imag_q = np.rint(filtered.imag * self.quant_scale) / self.quant_scale
            filtered = real_q + 1j * imag_q
        active = np.abs(filtered) > 1e-6
        non_zero = float(np.count_nonzero(active)) / float(filtered.size) if filtered.size else 0.0
        return {
            "schema": self.schema_name,
            "shape": [h, d],
            "start": start,
            "end": end,
            "fft_real": filtered.real,
            "fft_imag": filtered.imag,
            "freq_band": list(self.freq_band),
            "non_zero_ratio": non_zero,
        }

    def decode(
        self,
        spectral_payload: dict[str, Any] | np.ndarray,
        prev_chunk_tail: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(spectral_payload, np.ndarray):
            h = spectral_payload.shape[0]
            spec = np.fft.rfft(spectral_payload, axis=0)
            mask = build_spectral_window(spec.shape[0], self.freq_band, self.window)[:, None]
            return np.fft.irfft(spec * mask, n=h, axis=0)

        h, _ = spectral_payload["shape"]
        real = np.asarray(spectral_payload["fft_real"], dtype=float)
        imag = np.asarray(spectral_payload["fft_imag"], dtype=float)
        spectrum = real + 1j * imag
        mask = build_spectral_window(spectrum.shape[0], self.freq_band, self.window)[:, None]
        residual = np.fft.irfft(spectrum * mask, n=h, axis=0)
        start = np.asarray(spectral_payload["start"], dtype=float)
        end = np.asarray(spectral_payload["end"], dtype=float)
        t = np.linspace(0.0, 1.0, h)[:, None]
        trend = start + t * (end - start) if h > 1 else start.copy()
        return trend + residual


@dataclass
class WaveletInverseSpectrumModule:
    """Multi-resolution Discrete Wavelet Transform (Haar or Daubechies-4) codec.

    Shrinks high-frequency detail coefficients according to `freq_band` while
    preserving localized step changes better than global cosine/Fourier bases.
    """

    wavelet: Literal["haar", "db4"] = "haar"
    freq_band: tuple[float, float] = (0.0, 0.5)
    detail_threshold: float = 0.02
    schema_name: str = "idwt_haar"
    fast_tokenization: bool = True

    def __post_init__(self) -> None:
        self.schema_name = f"idwt_{self.wavelet}"

    def _filters(self) -> tuple[np.ndarray, np.ndarray]:
        return (_DB4_H, _DB4_G) if self.wavelet == "db4" else (_HAAR_H, _HAAR_G)

    def encode(self, time_chunk: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(time_chunk, dtype=float)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("Expected finite 2D action chunk (H, D)")
        h_orig, d = arr.shape
        # Remove linear endpoint trend so periodic DWT boundaries do not wrap jump
        start = arr[0:1, :]
        end = arr[-1:, :]
        t = np.linspace(0.0, 1.0, h_orig)[:, None]
        trend = start + t * (end - start) if h_orig > 1 else start.copy()
        residual = arr - trend

        # Pad residual to next power-of-two >= 4 with zeros (endpoints are already zero)
        padded_len = max(4, 1 << (h_orig - 1).bit_length())
        pad_rows = padded_len - h_orig
        if pad_rows > 0:
            padded = np.vstack([residual, np.zeros((pad_rows, d), dtype=float)])
        else:
            padded = residual.copy()

        low_f, high_f = self._filters()
        approx1, detail1 = _dwt_single_level(padded, low_f, high_f)
        approx2, detail2 = _dwt_single_level(approx1, low_f, high_f)

        # `freq_band[1]` scales the retention of finest detail level `detail1`
        # and second detail level `detail2`, plus soft-thresholding for denoising
        cutoff = float(self.freq_band[1])
        gain1 = float(np.clip((cutoff - 0.25) / 0.75, 0.0, 1.0))
        gain2 = float(np.clip(cutoff / 0.5, 0.0, 1.0))

        def soft_thresh(x: np.ndarray, thresh: float, gain: float) -> np.ndarray:
            shrunk = np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)
            return shrunk * gain

        d1_filt = soft_thresh(detail1, self.detail_threshold, gain1)
        d2_filt = soft_thresh(detail2, self.detail_threshold * 0.5, gain2)

        total_coeffs = approx2.size + d2_filt.size + d1_filt.size
        active_coeffs = (
            np.count_nonzero(np.abs(approx2) > 1e-6)
            + np.count_nonzero(np.abs(d2_filt) > 1e-6)
            + np.count_nonzero(np.abs(d1_filt) > 1e-6)
        )
        return {
            "schema": self.schema_name,
            "wavelet": self.wavelet,
            "shape": [h_orig, d],
            "padded_len": padded_len,
            "start": start,
            "end": end,
            "approx2": approx2,
            "detail2": d2_filt,
            "detail1": d1_filt,
            "freq_band": list(self.freq_band),
            "non_zero_ratio": float(active_coeffs) / float(max(total_coeffs, 1)),
        }

    def decode(
        self,
        spectral_payload: dict[str, Any] | np.ndarray,
        prev_chunk_tail: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(spectral_payload, np.ndarray):
            return self.decode(self.encode(spectral_payload), prev_chunk_tail=prev_chunk_tail)
        low_f, high_f = self._filters()
        approx2 = np.asarray(spectral_payload["approx2"], dtype=float)
        detail2 = np.asarray(spectral_payload["detail2"], dtype=float)
        detail1 = np.asarray(spectral_payload["detail1"], dtype=float)
        approx1 = _idwt_single_level(approx2, detail2, low_f, high_f)
        reconstructed = _idwt_single_level(approx1, detail1, low_f, high_f)
        h_orig = int(spectral_payload["shape"][0])
        residual = reconstructed[:h_orig, :]
        if "start" in spectral_payload and "end" in spectral_payload:
            start = np.asarray(spectral_payload["start"], dtype=float)
            end = np.asarray(spectral_payload["end"], dtype=float)
            t = np.linspace(0.0, 1.0, h_orig)[:, None]
            trend = start + t * (end - start) if h_orig > 1 else start.copy()
            return trend + residual
        return residual


@dataclass
class CustomHybridSpectrumModule:
    """Proposed Channel-Decoupled + C1 Boundary-Constrained Spectrum Schema.

    Key Innovations:
    1. Channel-Decoupled Spectral Representation:
       - Arm transport joints (all joints except the final gripper channel) are
         transformed with trend-removed Orthogonal DCT-II and smoothed with a
         Butterworth spectral window (`freq_band`) to eliminate high-frequency
         A* corner oscillations and VLM token noise.
       - The gripper channel (last column) bypasses global cosine smoothing and
         uses time-localized step preservation so grasp/release transitions
         never suffer from Gibbs ringing or premature opening.
    2. C1 Inter-Chunk Boundary Blending:
       - When `prev_chunk_tail` (last 1 or 2 executed joint states from the
         previous chunk) is supplied at `decode()` time, an exponentially
         decaying Hermite/velocity correction aligns the initial position and
         velocity of the new chunk with the tail of the previous chunk,
         suppressing the 10-step chunk-boundary jerk spike.
    """

    freq_band: tuple[float, float] = (0.0, 0.35)
    quant_scale: float = 50.0
    boundary_blend_steps: int = 4
    gripper_channel: int = -1
    schema_name: str = "custom_hybrid"
    fast_tokenization: bool = True

    def encode(self, time_chunk: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(time_chunk, dtype=float)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("Expected finite 2D action chunk (H, D)")
        h, d = arr.shape
        grip_idx = self.gripper_channel if self.gripper_channel >= 0 else d + self.gripper_channel
        arm_indices = [i for i in range(d) if i != grip_idx]

        arm_chunk = arr[:, arm_indices]
        grip_chunk = arr[:, [grip_idx]]

        # Center arm joints around initial frame of the chunk (consistent with DCT-II natural
        # zero-slope boundary reflection) so low-pass filtering does not pin endpoints
        offset = arm_chunk[0:1, :].copy()
        centered = arm_chunk - offset

        dct_coeffs = dct_ii(centered)
        mask = build_spectral_window(h, self.freq_band, window="butterworth", butterworth_order=3)[:, None]
        filtered = dct_coeffs * mask
        if self.quant_scale > 0:
            tokens = np.rint(filtered * self.quant_scale).astype(int)
        else:
            tokens = filtered

        # Run-length / step quantization for gripper channel
        grip_tokens = np.rint(grip_chunk * 100.0).astype(int)
        active = np.count_nonzero(tokens) + np.count_nonzero(np.diff(grip_tokens, axis=0)) + 1
        non_zero = float(active) / float(arr.size) if arr.size else 0.0
        return {
            "schema": self.schema_name,
            "shape": [h, d],
            "grip_idx": grip_idx,
            "arm_indices": arm_indices,
            "arm_offset": offset,
            "arm_dct_tokens": tokens,
            "quant_scale": self.quant_scale,
            "gripper_trace": grip_chunk.copy(),
            "freq_band": list(self.freq_band),
            "non_zero_ratio": min(1.0, non_zero),
        }

    def decode(
        self,
        spectral_payload: dict[str, Any] | np.ndarray,
        prev_chunk_tail: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(spectral_payload, np.ndarray):
            return self.decode(self.encode(spectral_payload), prev_chunk_tail=prev_chunk_tail)

        h, d = spectral_payload["shape"]
        grip_idx = int(spectral_payload["grip_idx"])
        arm_indices = list(spectral_payload["arm_indices"])
        scale = float(spectral_payload.get("quant_scale", self.quant_scale))
        tokens = np.asarray(spectral_payload["arm_dct_tokens"], dtype=float)
        coeffs = tokens / scale if scale > 0 else tokens

        mask = build_spectral_window(h, self.freq_band, window="butterworth", butterworth_order=3)[:, None]
        offset = np.asarray(spectral_payload["arm_offset"], dtype=float)
        arm_reconstructed = offset + idct_iii(coeffs * mask)

        # C1 boundary continuity correction if previous chunk tail is available
        if prev_chunk_tail is not None and h >= 2:
            tail = np.asarray(prev_chunk_tail, dtype=float)
            if tail.ndim == 1:
                tail = tail[None, :]
            prev_arm_last = tail[-1, arm_indices]
            prev_arm_vel = (
                tail[-1, arm_indices] - tail[-2, arm_indices]
                if len(tail) >= 2
                else arm_reconstructed[1] - arm_reconstructed[0]
            )
            # Only blend when there is an unnatural high-frequency jump against extrapolated position,
            # leaving smooth natural inter-chunk step transitions untouched
            expected_first = prev_arm_last + prev_arm_vel
            pos_mismatch = expected_first - arm_reconstructed[0]
            blend_len = min(self.boundary_blend_steps, h)
            tau = np.linspace(0.0, 0.5 * math.pi, blend_len)[:, None]
            decay = 0.5 * (np.cos(tau) ** 2)
            arm_reconstructed[:blend_len] += decay * pos_mismatch

        out = np.zeros((h, d), dtype=float)
        out[:, arm_indices] = arm_reconstructed
        out[:, [grip_idx]] = np.asarray(spectral_payload["gripper_trace"], dtype=float)
        return out


def decode_model_spectrum(
    module: InverseSpectrumModule,
    spectrum: dict[str, Any],
    prev_chunk_tail: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode DCT coefficients predicted natively by a model (e.g. π0-FAST tokens).

    `spectrum` holds `dct_coeffs` (H, D) on the orthonormal DCT-II basis and a per-joint
    `offset` (D,), such that `idct_iii(dct_coeffs) + offset` is the model's own action chunk.
    """
    coeffs = np.asarray(spectrum["dct_coeffs"], dtype=float)
    offset = np.asarray(spectrum["offset"], dtype=float).reshape(1, -1)
    if coeffs.ndim != 2 or offset.shape[1] != coeffs.shape[1] or not np.all(np.isfinite(coeffs)):
        raise ValueError("Malformed model spectrum")
    if isinstance(module, FASTIDCTModule):
        return module.decode({"dct_coeffs": coeffs, "offset": offset, "quant_scale": 0.0})
    if isinstance(module, CustomHybridSpectrumModule):
        h, d = coeffs.shape
        grip_idx = module.gripper_channel if module.gripper_channel >= 0 else d + module.gripper_channel
        arm_indices = [i for i in range(d) if i != grip_idx]
        payload = {
            "shape": [h, d],
            "grip_idx": grip_idx,
            "arm_indices": arm_indices,
            "arm_dct_tokens": coeffs[:, arm_indices],
            "quant_scale": 0.0,
            "arm_offset": offset[:, arm_indices],
            "gripper_trace": idct_iii(coeffs[:, [grip_idx]]) + offset[:, [grip_idx]],
        }
        return module.decode(payload, prev_chunk_tail=prev_chunk_tail)
    raise ValueError(
        f"Decoder {module.schema_name!r} cannot invert DCT coefficients; use a DCT-family decoder"
    )


def create_spectrum_module(
    schema: str = "direct",
    freq_band: Sequence[float] = (0.0, 1.0),
    quant_scale: float = 20.0,
    window: WindowType = "butterworth",
) -> InverseSpectrumModule:
    """Factory to instantiate any supported InverseSpectrumModule by name.

    Supports compact specification strings such as `"band_idct:0.35"` or
    `"ifft:0.4"`, where the suffix sets the upper cutoff of `freq_band`.
    """
    band = (float(freq_band[0]), float(freq_band[1]))
    name = schema.strip().lower()
    if ":" in name:
        name, cutoff_str = name.split(":", 1)
        band = (0.0, float(cutoff_str))

    if name in ("direct", "none", "time"):
        return DirectTimeDomainModule(freq_band=band)
    if name in ("fast_idct", "fast", "fast_dct", "dct"):
        return FASTIDCTModule(quant_scale=quant_scale, freq_band=band, window="rectangular")
    if name == "band_idct":
        default_band = band if band != (0.0, 1.0) else (0.0, 0.35)
        return BandLimitedIDCTModule(quant_scale=quant_scale, freq_band=default_band, window=window)
    if name == "ifft":
        default_band = band if band != (0.0, 1.0) else (0.0, 0.40)
        return IFFTInverseSpectrumModule(freq_band=default_band, window=window, quant_scale=quant_scale)
    if name == "idwt_haar":
        default_band = band if band != (0.0, 1.0) else (0.0, 0.50)
        return WaveletInverseSpectrumModule(wavelet="haar", freq_band=default_band)
    if name == "idwt_db4":
        default_band = band if band != (0.0, 1.0) else (0.0, 0.50)
        return WaveletInverseSpectrumModule(wavelet="db4", freq_band=default_band)
    if name in ("custom_hybrid", "hybrid", "custom"):
        default_band = band if band != (0.0, 1.0) else (0.0, 0.35)
        return CustomHybridSpectrumModule(freq_band=default_band, quant_scale=quant_scale)
    raise ValueError(f"Unknown InverseSpectrumModule schema {schema!r}")


def parse_freq_band(spec: str | Sequence[float]) -> tuple[float, float]:
    """Parse `'0.0,0.4'` or `(0.0, 0.4)` into a normalized frequency band tuple."""
    if isinstance(spec, str):
        parts = [float(item.strip()) for item in spec.split(",") if item.strip()]
        if len(parts) == 1:
            return (0.0, parts[0])
        if len(parts) == 2:
            return (parts[0], parts[1])
        raise ValueError(f"Expected 'low,high' or 'high' for freq_band, got {spec!r}")
    values = tuple(float(v) for v in spec)
    if len(values) != 2:
        raise ValueError("freq_band sequence must have 2 elements")
    return values[0], values[1]

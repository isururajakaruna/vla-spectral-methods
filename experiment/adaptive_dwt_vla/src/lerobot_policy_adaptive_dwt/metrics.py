"""Reconstruction, gripper-event and success metrics."""

from __future__ import annotations

import math

import numpy as np


def reconstruction_metrics(true: np.ndarray, recon: np.ndarray, cont_dims: list[int]) -> dict:
    """true/recon: (N, H, D) normalized windows. Continuous dims only."""
    t = np.asarray(true, dtype=np.float64)[..., cont_dims]
    r = np.asarray(recon, dtype=np.float64)[..., cont_dims]
    err = r - t
    vel = np.diff(r, axis=1) - np.diff(t, axis=1)
    acc = np.diff(r, 2, axis=1) - np.diff(t, 2, axis=1)
    jerk = np.diff(r, 3, axis=1) - np.diff(t, 3, axis=1)
    return {
        "mse": float((err**2).mean()),
        "mae": float(np.abs(err).mean()),
        "max_abs": float(np.abs(err).max()),
        "p99_abs": float(np.quantile(np.abs(err), 0.99)),
        "vel_mse": float((vel**2).mean()),
        "acc_mse": float((acc**2).mean()) if acc.size else float("nan"),
        "jerk_mse": float((jerk**2).mean()) if jerk.size else float("nan"),
    }


def transition_positions(bits: np.ndarray) -> set[int]:
    b = np.asarray(bits).astype(int)
    return {int(i) for i in np.nonzero(b[1:] != b[:-1])[0] + 1}


def gripper_metrics(true_bits: np.ndarray, recon_bits: np.ndarray) -> dict:
    """true/recon bits: (N, H). Transition precision/recall + exact-window rate."""
    tp = fp = fn = 0
    exact = 0
    for t, r in zip(true_bits, recon_bits, strict=True):
        tt, rr = transition_positions(t), transition_positions(r)
        tp += len(tt & rr)
        fp += len(rr - tt)
        fn += len(tt - rr)
        exact += int(np.array_equal(np.asarray(t).astype(int), np.asarray(r).astype(int)))
    return {
        "grip_transitions": tp + fn,
        "grip_precision": tp / (tp + fp) if tp + fp else 1.0,
        "grip_recall": tp / (tp + fn) if tp + fn else 1.0,
        "grip_exact_rate": exact / max(1, len(true_bits)),
    }


def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

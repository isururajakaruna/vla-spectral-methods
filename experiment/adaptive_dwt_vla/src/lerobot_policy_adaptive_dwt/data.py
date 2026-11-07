"""Window extraction and training-split normalization for the offline/indexed LIBERO data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
CACHE = EXP_ROOT / "data_cache"


def load_index() -> tuple[dict, dict]:
    idx = dict(np.load(CACHE / "libero_index.npz"))
    splits = json.loads((CACHE / "splits.json").read_text())
    return idx, splits


def split_episodes(splits: dict, task_ids: list[int] | None, split: str) -> list[int]:
    tasks = splits["tasks"]
    keys = [str(t) for t in task_ids] if task_ids is not None else list(tasks)
    return sorted(e for k in keys for e in tasks[k][split])


def norm_stats(idx: dict, episodes: list[int]) -> dict:
    """MEAN_STD stats for action/state from the given (training) episodes only."""
    m = np.isin(idx["episode"], episodes)
    out = {}
    for key in ("action", "state"):
        x = idx[key][m].astype(np.float64)
        out[key] = {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), 1e-8).tolist()}
    return out


def normalize_actions(a: np.ndarray, stats: dict) -> np.ndarray:
    return (a - np.asarray(stats["action"]["mean"])) / np.asarray(stats["action"]["std"])


def window_starts(idx: dict, episodes: list[int], horizon: int, stride: int = 1) -> np.ndarray:
    """Global frame rows r such that rows r..r+H-1 lie inside one episode (no padding)."""
    ep = idx["episode"]
    starts = []
    for e in episodes:
        rows = np.nonzero(ep == e)[0]
        n = len(rows)
        for i in range(0, n - horizon + 1, stride):
            starts.append(rows[i])
    return np.asarray(starts, dtype=np.int64)


def windows(idx: dict, starts: np.ndarray, horizon: int, stats: dict) -> np.ndarray:
    a = idx["action"]
    w = np.stack([a[s : s + horizon] for s in starts]).astype(np.float64)
    return normalize_actions(w, stats)

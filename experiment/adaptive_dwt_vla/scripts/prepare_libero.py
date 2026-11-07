"""Index lerobot/libero actions and build episode-level train/val splits.

Writes data_cache/libero_index.npz (all frames' actions/states + episode/task ids)
and data_cache/splits.json (per-task train/val episode lists, seed 0, 90/10).
Normalization statistics are computed later from the *training* episodes only.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

REPO = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=["meta/*", "data/*"]))
    info = json.loads((root / "meta/info.json").read_text())
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task_names = {int(i): str(name) for name, i in zip(tasks.index, tasks["task_index"], strict=True)}

    frames = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet")))])
    frames = frames.sort_values("index").reset_index(drop=True)
    actions = np.stack(frames["action"].to_numpy()).astype(np.float32)
    states = np.stack(frames["observation.state"].to_numpy()).astype(np.float32)
    ep = frames["episode_index"].to_numpy().astype(np.int64)
    fi = frames["frame_index"].to_numpy().astype(np.int64)
    ti = frames["task_index"].to_numpy().astype(np.int64)

    out = ROOT / "data_cache"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "libero_index.npz", action=actions, state=states, episode=ep, frame=fi, task=ti,
                        index=frames["index"].to_numpy().astype(np.int64))

    rng = np.random.default_rng(args.seed)
    splits = {"repo": REPO, "revision": REVISION, "fps": info["fps"], "seed": args.seed, "val_frac": args.val_frac, "tasks": {}}
    for t in sorted(task_names):
        eps = np.unique(ep[ti == t])
        perm = rng.permutation(eps)
        n_val = max(1, int(round(len(eps) * args.val_frac)))
        lengths = {int(e): int((ep == e).sum()) for e in eps}
        splits["tasks"][str(t)] = {
            "instruction": task_names[t],
            "train": sorted(int(e) for e in perm[n_val:]),
            "val": sorted(int(e) for e in perm[:n_val]),
            "episode_lengths": lengths,
        }
    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    n_ep = len(np.unique(ep))
    print(f"frames={len(frames)} episodes={n_ep} tasks={len(task_names)} -> {out}")
    print("gripper values:", np.unique(actions[:, 6]))


if __name__ == "__main__":
    main()

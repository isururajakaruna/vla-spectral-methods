"""Offline check of the training codecs with and without BPE on the training task (libero_object #0 = task 24).

Haar (segment codec, fixed plan / oracle plans) vs whole-window sym4→haar (symmetric, full depth / per-chunk
depth), each with BPE fitted on the training windows exactly as scripts/train.py does. Normalization uses the
task's training-split stats (as in training), so errors are not comparable with codec_choice_H10.json.

    python scripts/bpe_offline_check.py      # ~15 s, writes results/bpe_offline_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_adaptive_dwt.bpe import BPEDWTCodec  # noqa: E402
from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec  # noqa: E402
from lerobot_policy_adaptive_dwt.codec_window import WindowDWTCodec  # noqa: E402
from lerobot_policy_adaptive_dwt.data import load_index, norm_stats, split_episodes, window_starts, windows  # noqa: E402
from lerobot_policy_adaptive_dwt.oracle_dp import OracleWeights, solve  # noqa: E402

TASK, H = 24, 10


def jerk(x):
    return float((np.diff(x, 3, axis=1) ** 2).mean())


def main():
    idx, splits = load_index()
    tr, va = split_episodes(splits, [TASK], "train"), split_episodes(splits, [TASK], "val")
    st = norm_stats(idx, tr)
    m, s = np.asarray(st["action"]["mean"]), np.asarray(st["action"]["std"])
    spec = CodecSpec(horizon=H, lengths=(2, 4, 8), gripper_low=(-1 - m[6]) / s[6], gripper_high=(1 - m[6]) / s[6])
    cal = windows(idx, window_starts(idx, tr, H, stride=1), H, st)
    val = windows(idx, window_starts(idx, va, H, stride=1), H, st)
    choice = json.loads((ROOT / "results" / f"codec_choice_H{H}.json").read_text())
    w = OracleWeights(gamma=choice["adaptive"]["weights"]["gamma"])
    haar = AdaptiveDWTCodec.fit(spec, cal)
    s4h = WindowDWTCodec.fit(spec, cal, ["sym4", "haar"], "symmetric", 3)
    hfix = [tuple(p) for p in choice["fixed"]["plan"]]
    rows = []
    for cname, c in (("haar", haar), ("mix:sym4→haar", s4h)):
        for mode in ("fixed", "adaptive"):
            if mode == "fixed":
                fixed = hfix if c is haar else s4h.full_plan()
                plan_of = lambda x, p=fixed: p  # noqa: E731
            else:
                plan_of = lambda x, c=c: solve(c, x, w)[0]  # noqa: E731
            ptr, pva = [plan_of(x) for x in cal], [plan_of(x) for x in val]
            enc = [c.encode_plan(x, p) for x, p in zip(val, pva)]
            rec = np.stack([e[1] for e in enc])[..., :6]
            err = ((rec - val[..., :6]) ** 2).mean(axis=(1, 2))
            bpe = BPEDWTCodec.fit(c, cal, ptr)
            rows.append({
                "codec": cname, "mode": mode, "val_windows": len(val),
                "tokens": float(np.mean([len(e[0]) for e in enc])),
                "tokens_bpe": float(np.mean([len(bpe.compress(e[0])) for e in enc])),
                "merged_tokens": len(bpe.expansion),
                "val_mse": float(err.mean()), "val_mse_median_window": float(np.median(err)),
                "jerk_ratio": jerk(rec) / jerk(val[..., :6]),
            })
            r = rows[-1]
            print(f"{cname:14s} {mode:8s} tokens {r['tokens']:6.1f} -> BPE {r['tokens_bpe']:5.1f}  "
                  f"val MSE {r['val_mse']:.2e} (median window {r['val_mse_median_window']:.2e})  jerk {r['jerk_ratio']:.2f}")
    out = ROOT / "results" / "bpe_offline_check.json"
    out.write_text(json.dumps({"task": TASK, "horizon": H, "gamma": w.gamma, "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

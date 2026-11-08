"""Offline sweep: coarser quantization of the Haar detail coefficients + BPE, against FAST's error and token count.

Streams: "segment" (LEN/RES/gripper block per segment, as in the learning curve) and "compact" (codec_compact.py:
coefficients, one gripper block, A_END).

Same data protocol as Stage A (scripts/evaluate_codec.py): all 40 tasks, training-split normalization, quantizer and
BPE fitted on training windows (stride 2, subsampled), metrics on disjoint validation windows (stride H). The FAST
reference is read from results/codec_choice_H10.json (same validation windows).

Detail step = base step × scale_g, base step = 256-bin full-range step. Schemes:
  uniform  every detail group × s
  fine     group g of J × s**(g/J): the finest details are the coarsest, the coarsest details × s**(1/J)
Optional dead zone dz: |x|/step < 0.5 + dz rounds to 0 (dz = 0 is plain rounding).

    python scripts/quant_sweep.py      # writes results/quant_sweep.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_adaptive_dwt.bpe import BPEDWTCodec  # noqa: E402
from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec  # noqa: E402
from lerobot_policy_adaptive_dwt.codec_compact import CompactDWTCodec  # noqa: E402
from lerobot_policy_adaptive_dwt.data import load_index, norm_stats, split_episodes, window_starts, windows  # noqa: E402

H = 10
SCALES = [1, 2, 4, 6, 8, 12, 16, 24, 32]
SCHEMES = ["uniform", "fine"]
DEADZONES = [0.0, 0.3]
STREAMS = ["segment", "compact"]
N_FIT = 40000


def main():
    t0 = time.time()
    idx, splits = load_index()
    tr, va = split_episodes(splits, None, "train"), split_episodes(splits, None, "val")
    st = norm_stats(idx, tr)
    m, s = np.asarray(st["action"]["mean"]), np.asarray(st["action"]["std"])
    spec = CodecSpec(horizon=H, lengths=(2, 4, 8), gripper_low=(-1 - m[6]) / s[6], gripper_high=(1 - m[6]) / s[6])
    cal = windows(idx, window_starts(idx, tr, H, stride=2), H, st)
    val = windows(idx, window_starts(idx, va, H, stride=H), H, st)
    rng = np.random.default_rng(0)
    fit = cal[rng.choice(len(cal), size=min(N_FIT, len(cal)), replace=False)]
    choice = json.loads((ROOT / "results" / f"codec_choice_H{H}.json").read_text())
    fast = choice["fast_libero_val"]
    plan = [tuple(p) for p in choice["fixed"]["plan"]]
    print(f"{len(cal)} train windows ({len(fit)} for BPE), {len(val)} val windows; FAST: {fast['tokens_mean']:.1f} tokens, MSE {fast['mse']:.2e}", flush=True)
    rows = []
    for stream, scheme, dz, sc in [(st_, sch, dz_, sc_) for st_ in STREAMS for sch in SCHEMES for dz_ in DEADZONES for sc_ in SCALES]:
        if sc == 1 and (scheme != "uniform" or dz):
            continue
        c = AdaptiveDWTCodec.fit(spec, cal, detail_scale=sc, detail_scheme=scheme, deadzone=dz)
        if stream == "compact":
            c = CompactDWTCodec(c, plan)
        enc = [c.encode_plan(x, plan) for x in val]
        rec = np.stack([e[1] for e in enc])
        mse = float(((rec - val)[..., spec.cont_dims] ** 2).mean())
        bpe = BPEDWTCodec.fit(c, fit, [plan] * len(fit))
        tb = [len(bpe.compress(e[0])) for e in enc]
        bins = np.concatenate([[t for t in e[0] if t < 256] for e in enc])
        row = {"stream": stream, "scheme": scheme, "scale": sc, "deadzone": dz, "mse": mse, "tokens": float(np.mean([len(e[0]) for e in enc])),
               "tokens_bpe": float(np.mean(tb)), "tokens_bpe_p95": float(np.quantile(tb, 0.95)),
               "zero_frac": float(np.mean(bins == c.quantizer.mid)), "merged": len(bpe.expansion),
               "mse_vs_fast": mse / fast["mse"], "tokens_vs_fast": float(np.mean(tb)) / fast["tokens_mean"]}
        rows.append(row)
        print(f"{stream:7s} {scheme:7s} s={sc:<3} dz={dz:.1f}  BPE tokens {row['tokens_bpe']:5.1f} ({row['tokens_vs_fast']:.2f}× FAST)  "
              f"MSE {mse:.2e} ({row['mse_vs_fast']:.2f}× FAST)  zero bins {row['zero_frac']:.0%}", flush=True)
    out = ROOT / "results" / "quant_sweep.json"
    out.write_text(json.dumps({"horizon": H, "plan": plan, "fast": {"tokens": fast["tokens_mean"], "mse": fast["mse"]}, "rows": rows}, indent=1))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

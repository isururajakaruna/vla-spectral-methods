"""Stage A: offline codec comparison (adaptive oracle DWT, fixed DWT, FAST) on identical normalized windows.

Calibration (normalization stats, quantizer, oracle/fixed hyper-parameters) uses training-split
episodes only; reported metrics are computed on disjoint (stride = H) validation-episode windows.

Outputs (results/):
  codec_metrics.csv, codec_choice_H{H}.json, pareto_H{H}.png, example_stream_H{H}.txt,
  example_trajectory_H{H}.png, offline_codec_report.md (written by --report)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec, enumerate_plans
from lerobot_policy_adaptive_dwt.codec_basis import group_of
from lerobot_policy_adaptive_dwt.data import load_index, norm_stats, split_episodes, window_starts, windows
from lerobot_policy_adaptive_dwt.metrics import gripper_metrics, reconstruction_metrics
from lerobot_policy_adaptive_dwt.oracle_dp import OracleWeights, candidate_parts, solve_parts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LENGTHS = {10: (2, 4, 8), 32: (4, 8, 16, 32)}
GAMMAS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
FAST_TOKENIZERS = {"fast_libero": "jadechoghari/tokenizer-lib-mean", "fast_universal": "lerobot/fast-action-tokenizer"}


def entropy_bits(counter: Counter) -> float:
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values()) if n else 0.0


def summarize(name, family, cfg, true, recon, tokens_per, all_tokens, vocab_size, codec, extra=None):
    cont = codec.spec.cont_dims
    m = reconstruction_metrics(true, recon, cont)
    g = gripper_metrics(np.stack([codec.gripper_bits(w) for w in true]), np.stack([codec.gripper_bits(r) for r in recon]))
    t = np.asarray(tokens_per)
    h = entropy_bits(Counter(all_tokens))
    row = {
        "method": name, "family": family, "config": cfg, "n_windows": len(true),
        "tokens_mean": float(t.mean()), "tokens_p95": float(np.quantile(t, 0.95)), "tokens_max": int(t.max()),
        "vocab_size": vocab_size, "capacity_bits_per_chunk": float(t.mean() * math.log2(vocab_size)),
        "unigram_entropy_bits_per_chunk": float(t.mean() * h),
        "bits_per_dim_step": float(t.mean() * h / (codec.spec.horizon * codec.spec.action_dim)),
        **m, **g,
    }
    # error breakdown: steps within 2 of a gripper transition ("contact") vs the rest
    bits = np.stack([codec.gripper_bits(w) for w in true])
    near = np.zeros(bits.shape, dtype=bool)
    for i, b in enumerate(bits):
        for p in np.nonzero(b[1:] != b[:-1])[0] + 1:
            near[i, max(0, p - 2) : p + 3] = True
    se = ((np.asarray(recon)[..., cont] - np.asarray(true)[..., cont]) ** 2).mean(-1)
    row["mse_near_gripper_event"] = float(se[near].mean()) if near.any() else float("nan")
    row["mse_elsewhere"] = float(se[~near].mean())
    if extra:
        row.update(extra)
    return row


def run_dwt(codec, wins, plans, name, family, cfg, extra=None):
    recon, toks_per, all_toks = [], [], []
    clipped = coefs = 0
    boundary_se, interior_se = [], []
    group_bins: dict[int, list[int]] = {}
    for w, plan in zip(wins, plans, strict=True):
        toks, r, info = codec.encode_plan(w, plan)
        assert np.allclose(codec.decode(toks), r)
        recon.append(r)
        toks_per.append(len(toks))
        all_toks.extend(toks)
        clipped += info["clipped"]
        coefs += info["coefs"]
        se = ((r - w)[:, codec.spec.cont_dims] ** 2).mean(-1)
        s = 0
        for length, k in plan:
            boundary_se += [se[s], se[s + length - 1]]
            interior_se += list(se[s + 1 : s + length - 1])
            n = 2**k
            seg_groups = group_of(length)[:n]
            for g in np.unique(seg_groups):
                group_bins.setdefault(int(g), [])
            s += length
    ex = {"clip_rate": clipped / max(1, coefs), "mse_segment_boundary": float(np.mean(boundary_se)),
          "mse_segment_interior": float(np.mean(interior_se)) if interior_se else float("nan")}
    ex.update(extra or {})
    return summarize(name, family, cfg, wins, np.stack(recon), toks_per, all_toks, codec.vocab.size, codec, ex), np.stack(recon)


def run_fast(tok, wins, codec, name):
    toks = tok(wins.astype(np.float32))
    recon = np.asarray(tok.decode(toks, time_horizon=wins.shape[1], action_dim=wins.shape[2]), dtype=np.float64)
    per = [len(t) + 1 for t in toks]  # + the "|" end-of-action token used by pi0-FAST
    flat = [x for t in toks for x in t]
    return summarize(name, "fast", f"scale={tok.scale},bpe_vocab={tok.vocab_size}", wins, recon, per, flat,
                     tok.vocab_size, codec, {"clip_rate": float("nan")}), recon


def coef_distribution(codec, wins) -> dict:
    """Quantized-bin spread and clip rate per (L, wavelet group) at full resolution."""
    out = {}
    cont = wins[..., codec.spec.cont_dims]
    for length in codec.spec.lengths:
        segs = np.concatenate([cont[:, s : s + length] for s in range(0, codec.spec.horizon - length + 1, length)])
        from lerobot_policy_adaptive_dwt.codec_basis import haar_matrix

        coefs = np.einsum("cl,nld->ncd", haar_matrix(length), segs)
        groups = group_of(length)
        for g in np.unique(groups):
            sel = coefs[:, groups == g]
            bins, clip = zip(*[codec.quantizer.quantize(c, length) for c in coefs], strict=True)
            b = np.stack(bins)[:, groups == g]
            c = np.stack(clip)[:, groups == g]
            out[f"L{length}_g{g}"] = {"coef_std": float(sel.std()), "bin_std": float(b.std()),
                                       "bins_used": int(len(np.unique(b))), "clip_rate": float(c.mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 32])
    ap.add_argument("--n-select", type=int, default=3000, help="training windows used to choose hyper-parameters")
    args = ap.parse_args()
    from transformers import AutoProcessor

    RESULTS.mkdir(exist_ok=True)
    idx, splits = load_index()
    train_eps = split_episodes(splits, None, "train")
    val_eps = split_episodes(splits, None, "val")
    stats = norm_stats(idx, train_eps)
    rows = []
    fast = {k: AutoProcessor.from_pretrained(v, trust_remote_code=True) for k, v in FAST_TOKENIZERS.items()}
    rng = np.random.default_rng(0)
    for horizon in args.horizons:
        t0 = time.time()
        a_mean, a_std = stats["action"]["mean"][6], stats["action"]["std"][6]
        spec = CodecSpec(horizon=horizon, lengths=LENGTHS[horizon],
                         gripper_low=(-1 - a_mean) / a_std, gripper_high=(1 - a_mean) / a_std)
        cal = windows(idx, window_starts(idx, train_eps, horizon, stride=2), horizon, stats)
        codec = AdaptiveDWTCodec.fit(spec, cal)
        sel = cal[rng.choice(len(cal), size=min(args.n_select, len(cal)), replace=False)]
        val = windows(idx, window_starts(idx, val_eps, horizon, stride=horizon), horizon, stats)
        print(f"H={horizon}: calib windows={len(cal)} select={len(sel)} val(disjoint)={len(val)}", flush=True)

        # FAST on the same windows (train subset for matching, val for reporting)
        fast_rows = {}
        for name, tok in fast.items():
            r_sel, _ = run_fast(tok, sel, codec, name)
            row, _ = run_fast(tok, val, codec, name)
            row["horizon"] = horizon
            rows.append(row)
            fast_rows[name] = (r_sel, row)

        parts_sel = [candidate_parts(codec, w) for w in sel]
        parts_val = [candidate_parts(codec, w) for w in val]
        sweep = []
        for gamma in GAMMAS:
            w = OracleWeights(gamma=gamma)
            plans_sel = [solve_parts(p, horizon, spec.lengths, w)[0] for p in parts_sel]
            rs, _ = run_dwt(codec, sel, plans_sel, "dwt_adaptive_oracle", "adaptive", f"gamma={gamma}")
            plans_val = [solve_parts(p, horizon, spec.lengths, w)[0] for p in parts_val]
            usage = Counter((L, k) for p in plans_val for L, k in p)
            rv, _ = run_dwt(codec, val, plans_val, "dwt_adaptive_oracle", "adaptive", f"gamma={gamma}",
                            {"segments_per_chunk": float(np.mean([len(p) for p in plans_val])),
                             "len_res_usage": json.dumps({f"L{L}k{k}": c for (L, k), c in sorted(usage.items())})})
            rv["horizon"] = horizon
            rows.append(rv)
            sweep.append((gamma, rs, rv))

        fixed_cfgs = []
        for pattern in {tuple(sorted((L for L, _ in p), reverse=True)) for p in enumerate_plans(horizon, spec.lengths, fixed_k=0)}:
            for k in range(max(pattern).bit_length()):
                plan = [(L, min(k, L.bit_length() - 1)) for L in pattern]
                if plan not in [c for c, _, _ in fixed_cfgs]:
                    rs, _ = run_dwt(codec, sel, [plan] * len(sel), "dwt_fixed", "fixed", str(plan))
                    rv, _ = run_dwt(codec, val, [plan] * len(val), "dwt_fixed", "fixed", str(plan))
                    rv["horizon"] = horizon
                    rows.append(rv)
                    fixed_cfgs.append((plan, rs, rv))

        # operating point: adaptive gamma with train MSE closest to (<=) FAST-libero train MSE; fixed matched to it
        fast_mse = fast_rows["fast_libero"][0]["mse"]
        ok = [s for s in sweep if s[1]["mse"] <= fast_mse]
        gamma, ad_sel, ad_val = max(ok, key=lambda s: s[0]) if ok else min(sweep, key=lambda s: s[1]["mse"])
        fx_ok = [f for f in fixed_cfgs if f[1]["mse"] <= ad_sel["mse"]]
        fixed_plan, fx_sel, fx_val = min(fx_ok, key=lambda f: f[1]["tokens_mean"]) if fx_ok else min(fixed_cfgs, key=lambda f: f[1]["mse"])
        choice = {
            "horizon": horizon, "lengths": list(spec.lengths), "norm_stats_source": "train split, all 40 tasks",
            "criterion": "largest gamma whose train-window MSE <= FAST(libero) train MSE; fixed = fewest tokens with train MSE <= adaptive",
            "fast_libero_train_mse": fast_mse,
            "adaptive": {"weights": OracleWeights(gamma=gamma).to_dict(), "train": ad_sel, "val": ad_val},
            "fixed": {"plan": fixed_plan, "train": fx_sel, "val": fx_val},
            "fast_libero_val": fast_rows["fast_libero"][1], "fast_universal_val": fast_rows["fast_universal"][1],
            "coef_distribution_val": coef_distribution(codec, val),
            "codec": codec.to_dict(),
        }
        (RESULTS / f"codec_choice_H{horizon}.json").write_text(json.dumps(choice, indent=1))
        plot_pareto(horizon, sweep, fixed_cfgs, fast_rows, gamma, fixed_plan)
        example(horizon, codec, val, OracleWeights(gamma=gamma))
        print(f"H={horizon} done in {time.time() - t0:.0f}s: gamma*={gamma} fixed*={fixed_plan}", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("horizon", "method", "family", "config"), k))
    with open(RESULTS / "codec_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    (RESULTS / "norm_stats_all_tasks_train.json").write_text(json.dumps(stats, indent=1))


def plot_pareto(horizon, sweep, fixed_cfgs, fast_rows, gamma_star, fixed_star):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=140)
    ax.plot([s[2]["tokens_mean"] for s in sweep], [s[2]["mse"] for s in sweep], "o-", color="#0f766e", label="adaptive DWT oracle (γ sweep)")
    for g, _, rv in sweep:
        if g == gamma_star:
            ax.plot(rv["tokens_mean"], rv["mse"], "*", ms=16, color="#0f766e", mec="k", label=f"adaptive operating point (γ={g})")
    ax.scatter([f[2]["tokens_mean"] for f in fixed_cfgs], [f[2]["mse"] for f in fixed_cfgs], marker="s", color="#94a3b8", label="fixed DWT configs")
    for plan, _, rv in fixed_cfgs:
        if plan == fixed_star:
            ax.plot(rv["tokens_mean"], rv["mse"], "s", ms=11, color="#475569", mec="k", label="fixed operating point")
    for name, (_, rv), c in zip(fast_rows, fast_rows.values(), ["#d97706", "#b91c1c"], strict=True):
        ax.plot(rv["tokens_mean"], rv["mse"], "D", ms=10, color=c, mec="k", label=f"{name} (BPE)")
    ax.set_yscale("log")
    ax.set_xlabel("serialized action tokens per chunk (incl. end token)")
    ax.set_ylabel("normalized continuous-action MSE (val)")
    ax.set_title(f"LIBERO offline codec Pareto, H={horizon} (measured, val episodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / f"pareto_H{horizon}.png")
    plt.close(fig)


def example(horizon, codec, val, w):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick a validation window that contains a gripper transition
    i = next((i for i, x in enumerate(val) if len(set(codec.gripper_bits(x))) > 1), 0)
    win = val[i]
    plan, _ = solve_parts(candidate_parts(codec, win), horizon, codec.spec.lengths, w)
    toks, recon, _ = codec.encode_plan(win, plan)
    fsm = codec.fsm()
    lines = [f"plan={plan}  tokens={len(toks)}  (prefix 'Action: ' shared with FAST is not counted)", ""]
    for t in toks:
        names = {0: "SEG_START", 1: "RES", 2: "COEF", 3: "GRIP_INIT", 4: "GRIP_EVENTS", 5: "DONE"}
        lines.append(f"{names[fsm.state]:<12} n_allowed={len(fsm.allowed()):<4} emit {codec.vocab.describe(t)}")
        fsm.step(t)
    (RESULTS / f"example_stream_H{horizon}.txt").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(7, 1, figsize=(7, 9), dpi=120, sharex=True)
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    for d, ax in enumerate(axes):
        ax.plot(win[:, d], "k.-", lw=1, label="demo (normalized)")
        ax.plot(recon[:, d], "-", color="#0f766e", lw=1.6, label="adaptive DWT decode")
        s = 0
        for length, k in plan:
            ax.axvline(s - 0.5, color="#cbd5e1", lw=0.8)
            if d == 0:
                ax.text(s + length / 2 - 0.5, ax.get_ylim()[1], f"L{length} R{k}", ha="center", va="bottom", fontsize=7)
            s += length
        ax.set_ylabel(names[d], fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(RESULTS / f"example_trajectory_H{horizon}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

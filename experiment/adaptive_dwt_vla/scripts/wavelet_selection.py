"""Offline wavelet selection test: which wavelet represents LIBERO action chunks most compactly?

For every candidate transform (20 standard wavelets + mixed transforms that use a different wavelet at each
decomposition level), in two boundary modes, on normalized action windows:

  forward multilevel DWT -> 256-bin quantizer (calibrated on training windows, same rule as the codec)
  -> keep the coarsest k groups [cA_J, cD_J, ..., cD_(J-k+1)] -> inverse DWT -> error vs the original.

Two keep rules:
  fixed     the same k for every chunk (tokens = 6 * kept coefficients)
  adaptive  k chosen per chunk by min  MSE + gamma * tokens  (tokens = 6 * kept + 1 depth token), gamma swept

Only the 6 continuous dims are coded; the gripper is encoded exactly and identically for every transform, so
its tokens are left out. Whole-window transforms (no segmentation DP), so absolute token counts differ from
the deployed Haar codec; the ranking compares transforms under identical rules.

Calibration: training-split episodes (all 40 tasks). Metrics: disjoint (stride = H) validation-episode windows.
Outputs: experiment/reports/wavelet_selection.html, results/wavelet_selection.{json,csv}. No dataset content is written.

Run with the VLA env:  python scripts/wavelet_selection.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORTS = ROOT.parent / "reports"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot_policy_adaptive_dwt.data import load_index, norm_stats, split_episodes, window_starts, windows  # noqa: E402

N_BINS, MID = 256, 127
CONT = [0, 1, 2, 3, 4, 5]
LEVELS = {10: 3, 32: 4}  # coarsest approximation has 2 samples per dim in periodization mode
MODES = ("periodization", "symmetric")
GAMMAS = np.concatenate([[0.0], np.logspace(-8, 0, 97)])

SINGLE = ["haar", "db2", "db3", "db4", "db6", "sym4", "sym5", "sym6", "coif1", "coif2", "coif3",
          "bior1.3", "bior2.2", "bior2.4", "bior3.3", "bior4.4", "bior6.8", "rbio2.2", "rbio3.3", "dmey"]
# per-level wavelets, finest level first; the last entry is reused for any deeper level
MIXED = {
    "mix:haar→db2→db4": ["haar", "db2", "db4"],
    "mix:haar→sym4": ["haar", "sym4"],
    "mix:haar→bior2.2": ["haar", "bior2.2"],
    "mix:haar,haar→bior4.4": ["haar", "haar", "bior4.4"],
    "mix:db2→coif1": ["db2", "coif1"],
    "mix:sym4→haar": ["sym4", "haar"],
}
POPULAR = {"haar", "db2", "db4", "sym4", "coif1", "bior2.2", "bior4.4", "dmey"}


def candidates():
    for w in SINGLE:
        yield w, [w], "single"
    for name, ws in MIXED.items():
        yield name, ws, "mixed"


# ----------------------------------------------------------------------------------------------- transforms


def forward(x, waves, mode, levels):
    a, details, lens = np.asarray(x, dtype=np.float64), [], []
    for j in range(levels):
        lens.append(len(a))
        a, d = pywt.dwt(a, waves[min(j, len(waves) - 1)], mode=mode)
        details.append(d)
    return [a] + details[::-1], lens


def inverse(cs, waves, mode, lens):
    levels = len(lens)
    a = cs[0]
    for j in reversed(range(levels)):
        a = pywt.idwt(a, cs[levels - j], waves[min(j, len(waves) - 1)], mode=mode)[: lens[j]]
    return a


def transform_matrices(waves, mode, horizon, levels):
    """Analysis A (n_coef, H), synthesis S (H, n_coef), group id per coefficient (0 = cA_J ... J = cD_1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols, shapes, lens = [], None, None
        for e in np.eye(horizon):
            cs, lens = forward(e, waves, mode, levels)
            shapes = [len(c) for c in cs]
            cols.append(np.concatenate(cs))
        A = np.stack(cols, axis=1)
        n = A.shape[0]
        S = np.zeros((horizon, n))
        offs = np.cumsum([0] + shapes)
        for i in range(n):
            u = np.zeros(n)
            u[i] = 1
            S[:, i] = inverse([u[offs[g]:offs[g + 1]] for g in range(len(shapes))], waves, mode, lens)
    groups = np.concatenate([np.full(s, g) for g, s in enumerate(shapes)])
    return A, S, groups, shapes


# ----------------------------------------------------------------------------------------------- evaluation


def fit_quantizer(A, groups, train_chunks):
    """Per (group, dim): approximation group centered on its range, detail groups centered on 0 (quantile 1.0)."""
    n_g, D = int(groups.max()) + 1, len(CONT)
    lo, hi, amax = np.full((n_g, D), np.inf), np.full((n_g, D), -np.inf), np.zeros((n_g, D))
    for X in train_chunks:
        C = np.einsum("ch,nhd->ncd", A, X)
        for g in range(n_g):
            v = C[:, groups == g].reshape(-1, D)
            lo[g], hi[g] = np.minimum(lo[g], v.min(0)), np.maximum(hi[g], v.max(0))
            amax[g] = np.maximum(amax[g], np.abs(v).max(0))
    center, step = np.zeros((n_g, D)), np.zeros((n_g, D))
    center[0] = 0.5 * (lo[0] + hi[0])
    step[0] = np.maximum(0.5 * (hi[0] - lo[0]), 1e-6) / MID
    step[1:] = np.maximum(amax[1:], 1e-6) / MID
    return center[groups], step[groups]  # (n_coef, D)


def interp_tokens(points, target):
    """Fewest tokens reaching `target` MSE on the lower convex hull of the curve. Between two hull points the
    interpolation is linear in both tokens and MSE, which a codec achieves by mixing the two settings across chunks."""
    hull = []
    for p in sorted(points):
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) <= (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    best = float("inf")
    for i, (t, m) in enumerate(hull):
        if m <= target:
            best = min(best, float(t))
            if i:
                t0, m0 = hull[i - 1]
                if m0 > target:
                    best = min(best, float(t0 + (m0 - target) / (m0 - m) * (t - t0)))
    return best


def mse_at_budget(points, budget):
    """MSE reachable with `budget` mean tokens on the lower convex hull (nan below the cheapest setting)."""
    hull = []
    for p in sorted(points):
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) <= (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    if budget < hull[0][0]:
        return float("nan")
    for (t0, m0), (t1, m1) in zip(hull, hull[1:]):
        if t0 <= budget <= t1:
            return float(m0 + (budget - t0) / max(t1 - t0, 1e-12) * (m1 - m0))
    return float(min(m for _, m in hull))


def evaluate(name, waves, kind, mode, horizon, train_chunks, val, targets):
    levels = LEVELS[horizon]
    A, S, groups, shapes = transform_matrices(waves, mode, horizon, levels)
    pr_err = float(np.abs(S @ A - np.eye(horizon)).max())
    center, step = fit_quantizer(A, groups, train_chunks)
    C = np.einsum("ch,nhd->ncd", A, val)
    raw = np.rint((C - center) / step) + MID
    clip_rate = float(((raw < 0) | (raw > N_BINS - 1)).mean())
    Cq = center + (np.clip(raw, 0, N_BINS - 1) - MID) * step

    K = levels + 1
    n_keep = np.array([int((groups <= k).sum()) for k in range(K)])
    N = len(val)
    mse, maxabs, bse, jerk = (np.zeros((N, K)) for _ in range(4))
    jerk_ref = np.abs(np.diff(val, 3, axis=1)).mean()
    for k in range(K):
        R = np.einsum("hc,ncd->nhd", S[:, : n_keep[k]], Cq[:, : n_keep[k]])
        e = R - val
        mse[:, k] = (e**2).mean((1, 2))
        maxabs[:, k] = np.abs(e).max((1, 2))
        bse[:, k] = 0.5 * ((e[:, 0] ** 2).mean(-1) + (e[:, -1] ** 2).mean(-1))
        jerk[:, k] = np.abs(np.diff(R, 3, axis=1)).mean((1, 2))

    tok_fixed = 6 * n_keep
    tok_adapt = 6 * n_keep + 1
    fixed_curve = [(int(t), float(m)) for t, m in zip(tok_fixed, mse.mean(0))]
    adapt_curve, choices = [], []
    for g in GAMMAS:
        ch = np.argmin(mse + g * tok_adapt[None, :], axis=1)
        choices.append(ch)
        adapt_curve.append((float(tok_adapt[ch].mean()), float(mse[np.arange(N), ch].mean())))

    row = {"name": name, "kind": kind, "mode": mode, "horizon": horizon, "levels": levels, "waves": waves,
           "filter_lengths": [pywt.Wavelet(w).dec_len for w in waves], "coef_per_dim": int(len(groups)),
           "group_sizes": shapes, "pr_error": pr_err, "clip_rate": clip_rate,
           "mse_full_resolution": float(mse[:, -1].mean()), "fixed_curve": fixed_curve, "adaptive_curve": adapt_curve}
    for label, tgt in targets.items():
        row[f"tokens_adaptive@{label}"] = interp_tokens(adapt_curve, tgt)
        row[f"tokens_fixed@{label}"] = interp_tokens(fixed_curve, tgt)
    # operating point: largest gamma whose mean MSE <= FAST MSE
    tgt = targets["fast"]
    ok = [i for i, (_, m) in enumerate(adapt_curve) if m <= tgt]
    if ok:
        i = max(ok, key=lambda j: GAMMAS[j])
        ch = choices[i]
        idx = np.arange(N)
        row.update({"op_gamma": float(GAMMAS[i]), "op_tokens": adapt_curve[i][0], "op_mse": adapt_curve[i][1],
                    "op_maxabs_p99": float(np.quantile(maxabs[idx, ch], 0.99)), "op_maxabs_max": float(maxabs[idx, ch].max()),
                    "op_boundary_mse": float(bse[idx, ch].mean()), "op_jerk_ratio": float(jerk[idx, ch].mean() / jerk_ref),
                    "op_depth_usage": np.bincount(ch, minlength=K).tolist()})
    return row, S, groups


# ----------------------------------------------------------------------------------------------- ranking


def add_scores(rows):
    """score = tokens at FAST accuracy / Haar's (periodization) at the same horizon; inf if not perfectly reconstructing.
    Also the MSE each transform reaches with a tighter budget of 2/3 of Haar's tokens at FAST accuracy."""
    for h in sorted({r["horizon"] for r in rows}):
        ref = next(r for r in rows if r["name"] == "haar" and r["mode"] == "periodization" and r["horizon"] == h)["tokens_adaptive@fast"]
        for r in rows:
            if r["horizon"] == h:
                r["score"] = r["tokens_adaptive@fast"] / ref if r["pr_error"] < 1e-6 else float("inf")
                r["budget_tokens"] = 2 / 3 * ref
                r["mse@budget"] = mse_at_budget(r["adaptive_curve"], r["budget_tokens"])


# ----------------------------------------------------------------------------------------------- report


def build_html(rows, meta, atoms):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from write_training_report import CSS, esc, fig_html, table

    hs = sorted({r["horizon"] for r in rows})
    names = list(dict.fromkeys(r["name"] for r in rows))
    best = {}  # (name, H) -> best-mode row
    for r in rows:
        k = (r["name"], r["horizon"])
        if k not in best or r["score"] < best[k]["score"]:
            best[k] = r
    overall = sorted(names, key=lambda n: (np.mean([best[(n, h)]["score"] for h in hs]), best[(n, max(hs))].get("tokens_adaptive@fast", np.inf)))
    haar = {h: next(r for r in rows if r["name"] == "haar" and r["mode"] == "periodization" and r["horizon"] == h) for h in hs}

    def tok(x):
        return "not reached" if not np.isfinite(x) else f"{x:.1f}"

    def rel(x, h):
        b = haar[h]["tokens_adaptive@fast"]
        return "–" if not (np.isfinite(x) and np.isfinite(b)) else f"{100 * (x / b - 1):+.0f}%"

    h_ = []
    add = h_.append
    add(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Wavelet selection test</title><style>{CSS}"
        ".kpi b{font-size:1.15rem;overflow-wrap:anywhere} .wide{overflow-x:auto} .wide table{font-size:.8rem}</style></head><body>")
    add("<h1>Wavelet selection test for DWT action tokens</h1>")
    add(f"<p class='sub'>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/wavelet_selection.py</code> in {meta['runtime_s']:.0f} s on CPU. "
        "Offline only: no model is trained; every number is measured on held-out LIBERO demonstration windows. No dataset content is included. "
        "Trained models and their results: <a href='training_report.html'>training_report.html</a>.</p>")

    winner = overall[0]
    add("<div class='kpis'>")
    for h in hs:
        b = best[(winner, h)]
        add(f"<div class='kpi'><span>overall #1, H = {h}</span><b>{esc(winner)}</b><span>{esc(b['mode'])}: {tok(b['tokens_adaptive@fast'])} tokens at FAST accuracy "
            f"({rel(b['tokens_adaptive@fast'], h)} vs Haar)</span></div>")
    for h in hs:
        top = min((best[(n, h)] for n in names), key=lambda r: r.get("tokens_adaptive@fast", np.inf))
        add(f"<div class='kpi'><span>fewest tokens, H = {h}</span><b>{esc(top['name'])}</b><span>{esc(top['mode'])}: {tok(top['tokens_adaptive@fast'])} "
            f"({rel(top['tokens_adaptive@fast'], h)} vs Haar {tok(haar[h]['tokens_adaptive@fast'])})</span></div>")
    add("</div>")

    add("<h2>1. What the test does</h2><ol>")
    for s in [
        f"Cut demonstration actions into windows of H = {', '.join(map(str, hs))} steps; normalize with training-split mean/std. "
        f"Calibration: {', '.join(f'{meta[f'n_train_{h}']} (H={h})' for h in hs)} training windows (stride 2); evaluation: {', '.join(f'{meta[f'n_val_{h}']} (H={h})' for h in hs)} disjoint validation windows, all 40 tasks.",
        "Forward multilevel DWT per action dimension (6 continuous dims; the gripper is coded exactly by every variant and left out). "
        f"Levels: {', '.join(f'{LEVELS[h]} at H={h}' for h in hs)}, giving coefficient groups cA (trend) and cD per level (detail, coarse → fine).",
        "Quantize every coefficient to 256 bins with a per-(level, dimension) step calibrated on training windows (same rule as the deployed codec).",
        "Keep the coarsest k groups and drop the finer detail. <b>Fixed</b>: the same k for every chunk. <b>Adaptive</b>: k chosen per chunk to minimize MSE + γ·tokens (one extra token states k), γ swept.",
        "Inverse DWT and compare with the original: mean squared error, worst-case error, error at the chunk's first/last step (where chunks join) and added roughness (jerk ratio).",
    ]:
        add(f"<li>{s}</li>")
    add("</ol>")
    add(f"<p><b>Ranking metric:</b> tokens per chunk needed to match FAST's reconstruction accuracy (FAST-libero validation MSE {', '.join(f'{meta[f'fast_mse_{h}']:.2e} at H={h}' for h in hs)}), "
        "read off the adaptive rate–distortion curve (lower convex hull, linear interpolation = mixing two settings across chunks), divided by Haar's. "
        "Each transform uses its better boundary mode per horizon; the overall ranking averages the two horizons. "
        "Worst-case error, chunk-edge error and jerk at the FAST-matched operating point are reported alongside but not ranked: a transform can buy low worst-case error simply by spending more tokens.</p>")
    add("<p><b>Mixed transforms</b> use a different wavelet at each decomposition level (finest level first). <code>haar→…</code> variants keep Haar's sharp steps at the finest level "
        "for abrupt events (contact, sudden stops) and use a smooth wavelet for the slow coarse motion; <code>sym4→haar</code> is the reverse ordering.</p>")
    add("<h2>Findings (computed from the table below)</h2><ul>")
    for h in hs:
        ranked = sorted(names, key=lambda n: best[(n, h)]["score"])
        better = [n for n in ranked if best[(n, h)]["score"] < 0.98]
        within = [n for n in ranked if 0.98 <= best[(n, h)]["score"] <= 1.02 and n != "haar"]
        top = best[(ranked[0], h)]
        add(f"<li>H = {h}: fewest tokens <code>{esc(ranked[0])}</code> ({esc(top['mode'])}) at {top['tokens_adaptive@fast']:.1f} vs Haar {haar[h]['tokens_adaptive@fast']:.1f} "
            f"({100 * (top['score'] - 1):+.1f}%). {len(better)} of {len(names) - 1} alternatives need ≥2% fewer tokens than Haar"
            + (f" ({', '.join(f'<code>{esc(n)}</code>' for n in better)})" if better else "")
            + f"; {len(within)} are within ±2%.</li>")
    for h in hs:
        rp = [r["score"] for r in rows if r["horizon"] == h and r["mode"] == "periodization" and np.isfinite(r["score"]) and r["name"] != "haar"]
        rs = [r["score"] for r in rows if r["horizon"] == h and r["mode"] == "symmetric" and np.isfinite(r["score"]) and r["name"] != "haar"]
        add(f"<li>H = {h}: median token cost vs Haar is {100 * (np.median(rp) - 1):+.0f}% in periodization mode and {100 * (np.median(rs) - 1):+.0f}% in symmetric mode "
            "(symmetric extension adds edge coefficients that have to be sent).</li>")
    for h in hs:
        ok = [r for r in rows if r["horizon"] == h and r["pr_error"] < 1e-6 and np.isfinite(r["mse@budget"])]
        top = min(ok, key=lambda r: r["mse@budget"])
        hm = haar[h]["mse@budget"]
        add(f"<li>H = {h}, tighter budget of {haar[h]['budget_tokens']:.0f} tokens (⅔ of Haar's tokens at FAST accuracy): lowest error <code>{esc(top['name'])}</code> ({esc(top['mode'])}) "
            f"with MSE {top['mse@budget']:.2e} vs Haar {hm:.2e} ({hm / top['mse@budget']:.1f}× lower).</li>")
    jr = {h: haar[h].get("op_jerk_ratio", np.nan) for h in hs}
    add(f"<li>Haar reconstructions are rougher than the demonstrations (jerk ratio {', '.join(f'{jr[h]:.2f} at H={h}' for h in hs)}); smooth wavelets sit near 1.0. "
        "Roughness is not part of the ranking but matters for executing the actions on a robot.</li>")
    excl = sorted({r["name"] for r in rows if not np.isfinite(r["score"]) and r["pr_error"] >= 1e-6})
    if excl:
        add(f"<li>Excluded (inverse does not reconstruct the input to 1e-6): {', '.join(f'<code>{esc(n)}</code>' for n in excl)}; "
            "the discrete Meyer filter is a truncated FIR approximation.</li>")
    add("</ul>")
    add("<div class='note'>Whole-window transforms with a fixed depth, no segmentation search: absolute token counts are not those of the deployed Haar codec "
        "(which splits a chunk into 8+2 or similar segments), and gripper/structure tokens are excluded. Compare rows with each other, not with the training report.</div>")

    add("<h2>2. Overall ranking</h2>")
    body = []
    for i, n in enumerate(overall, 1):
        cells = [f"<b>{i}</b>", f"<code>{esc(n)}</code>" + (" <span class='tag'>popular</span>" if n in POPULAR else ""),
                 esc(best[(n, hs[0])]["kind"]), ", ".join(str(L) for L in best[(n, hs[0])]["filter_lengths"])]
        for h in hs:
            b = best[(n, h)]
            cells += [esc(b["mode"][:5] + "."), tok(b.get("tokens_adaptive@fast", np.inf)), rel(b.get("tokens_adaptive@fast", np.inf), h),
                      "–" if "op_maxabs_p99" not in b else f"{b['op_maxabs_p99']:.3f}", "–" if "op_boundary_mse" not in b else f"{b['op_boundary_mse']:.1e}"]
        sc = np.mean([best[(n, h)]["score"] for h in hs])
        cells.append("excluded" if not np.isfinite(sc) else f"<b>{100 * (sc - 1):+.1f}%</b>")
        body.append(cells)
    hdr = ["#", "transform", "type", "filter taps"]
    for h in hs:
        hdr += [f"H{h} mode", f"H{h} tokens @FAST", f"H{h} vs Haar", f"H{h} p99 max err", f"H{h} edge MSE"]
    add("<div class='wide'>" + table(hdr + ["mean tokens vs Haar"], body) + "</div>")

    # --- plots
    fig, axes = plt.subplots(1, len(hs), figsize=(7 * len(hs), 4.6))
    for ax, h in zip(np.atleast_1d(axes), hs):
        top = [n for n in overall[:6]]
        for n in names:
            b = best[(n, h)]
            c = sorted(b["adaptive_curve"])
            if n in top or n == "haar":
                continue
            ax.plot([p[0] for p in c], [p[1] for p in c], color="#c5ccd8", lw=0.8)
        for n, col in zip(top + (["haar"] if "haar" not in top else []), plt.cm.tab10.colors):
            b = best[(n, h)]
            c = sorted(b["adaptive_curve"])
            ax.plot([p[0] for p in c], [p[1] for p in c], color=col, lw=2 if n != "haar" else 2.4, ls="-" if n != "haar" else "--", label=f"{n} ({b['mode'][:5]}.)")
        ax.axhline(meta[f"fast_mse_{h}"], color="#d9534f", lw=1, ls=":", label="FAST-libero MSE")
        ax.set_yscale("log")
        ax.set_xlabel("coefficient tokens per chunk (adaptive depth)")
        ax.set_ylabel("validation MSE (normalized actions)")
        ax.set_title(f"H = {h}: rate–distortion, top 6 + Haar (grey: others)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    add("<h2>3. Rate–distortion curves</h2>" + fig_html(fig, "rd curves"))

    fig, axes = plt.subplots(1, len(hs), figsize=(7 * len(hs), 6.5))
    for ax, h in zip(np.atleast_1d(axes), hs):
        ns = sorted((n for n in names if best[(n, h)]["pr_error"] < 1e-6), key=lambda n: best[(n, h)].get("tokens_adaptive@fast", np.inf))
        vals = [best[(n, h)].get("tokens_adaptive@fast", np.inf) for n in ns]
        b = haar[h]["tokens_adaptive@fast"]
        v = [100 * (x / b - 1) if np.isfinite(x) else np.nan for x in vals]
        cols = ["#3c9d5d" if x < 0 else "#d9534f" for x in np.nan_to_num(v)]
        ax.barh(range(len(ns)), np.nan_to_num(v, nan=0), color=cols)
        for i, x in enumerate(v):
            if not np.isfinite(x):
                ax.text(0, i, " not reached", va="center", fontsize=7)
        ax.set_yticks(range(len(ns)), [f"{n} ({best[(n, h)]['mode'][:5]}.)" for n in ns], fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("tokens at FAST accuracy, % vs Haar (negative = fewer tokens)")
        ax.set_title(f"H = {h}")
        ax.grid(alpha=0.3, axis="x")
    add("<h2>4. Tokens needed at FAST accuracy, relative to Haar</h2>" + fig_html(fig, "token bars"))

    # basis atoms: one synthesis vector per level at H=32 periodization
    n_atoms = len(atoms)
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(n_atoms / cols)), cols, figsize=(16, 2.1 * np.ceil(n_atoms / cols)))
    for ax, (n, (S, groups)) in zip(axes.ravel(), atoms.items()):
        for g in range(int(groups.max()) + 1):
            idx = np.nonzero(groups == g)[0]
            v = S[:, idx[len(idx) // 2]]
            ax.plot(v / (np.abs(v).max() + 1e-12) + 2.2 * g, lw=1.1)
        ax.set_title(n, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_atoms:]:
        ax.axis("off")
    add("<h2>5. What each transform looks like</h2><p>One synthesis atom per level on a 32-step chunk (bottom: coarsest trend, top: finest detail), periodization mode. "
        "Mixed transforms visibly change shape from level to level.</p>" + fig_html(fig, "atoms"))

    # full table
    add("<h2>6. Full results (every transform × boundary mode × horizon)</h2>")
    for h in hs:
        sub = sorted([r for r in rows if r["horizon"] == h], key=lambda r: r["score"])
        body = [[f"<code>{esc(r['name'])}</code>", esc(r["mode"]), r["coef_per_dim"], f"{r['pr_error']:.0e}", f"{100 * r['clip_rate']:.2f}%",
                 f"{r['mse_full_resolution']:.2e}", tok(r["tokens_fixed@fast"]), tok(r["tokens_adaptive@fast"]), tok(r["tokens_adaptive@fast/2"]),
                 "–" if "op_maxabs_p99" not in r else f"{r['op_maxabs_p99']:.3f}", "–" if "op_boundary_mse" not in r else f"{r['op_boundary_mse']:.1e}",
                 "–" if "op_jerk_ratio" not in r else f"{r['op_jerk_ratio']:.2f}",
                 "–" if not np.isfinite(r["mse@budget"]) else f"{r['mse@budget']:.2e}", "excluded" if r["pr_error"] >= 1e-6 else ("not reached" if not np.isfinite(r["score"]) else f"{100 * (r['score'] - 1):+.1f}%")] for r in sub]
        add(f"<h3>H = {h}</h3>")
        add("<div class='wide'>" + table(["transform", "mode", "coefs / dim", "reconstruction check", "clip rate", "MSE all coefs kept", "tokens @FAST (fixed)",
                   "tokens @FAST (adaptive)", "tokens @½FAST (adaptive)", "p99 max err", "edge MSE", "jerk ratio", f"MSE @ {sub[0]['budget_tokens']:.0f} tokens", "tokens vs Haar"], body) + "</div>")
    add("<p><i>coefs / dim</i>: coefficients per action dimension before dropping (symmetric mode is redundant, so it can exceed H). "
        "<i>Reconstruction check</i>: max |S·A − I|, must be ≈0. <i>MSE all coefs kept</i>: quantization-only error. "
        "<i>Jerk ratio</i>: mean |third difference| of the reconstruction over that of the original; above 1 means the codec adds roughness.</p>")

    add("<h2>7. Caveats</h2><ul>" + "".join(f"<li>{esc(c)}</li>" for c in [
        "This measures how well a transform can represent the motion, not how well a model can learn to predict its tokens. The winner still has to be trained and evaluated in simulation.",
        "Depth is chosen per whole chunk; the deployed codec also searches over segmentations, which helps Haar most because Haar segments do not overlap.",
        "Symmetric mode adds coefficients at the edges (redundant expansion); those extra coefficients are counted as tokens.",
        "Biorthogonal transforms are not energy-preserving, so the same quantizer step gives different signal-domain noise; that effect is included in the measurements.",
        "Ties and near-ties within a few percent are within the variation you would see from a different validation split.",
    ]) + "</ul>")
    add("<h2>8. Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla\n~/miniconda3/envs/smooth-operator-vla/bin/python scripts/wavelet_selection.py</pre></body></html>")
    return "\n".join(h_)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 32])
    ap.add_argument("--train-stride", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    idx, splits = load_index()
    train_eps, val_eps = split_episodes(splits, None, "train"), split_episodes(splits, None, "val")
    stats = norm_stats(idx, train_eps)
    meta, rows, atoms = {}, [], {}
    for h in args.horizons:
        fast = json.loads((RES / f"codec_choice_H{h}.json").read_text())["fast_libero_val"]
        targets = {"fast": fast["mse"], "fast/2": fast["mse"] / 2}
        meta[f"fast_mse_{h}"], meta[f"fast_tokens_{h}"] = fast["mse"], fast["tokens_mean"]
        starts = window_starts(idx, train_eps, h, stride=args.train_stride)
        train_chunks = [windows(idx, starts[i:i + 20000], h, stats)[..., CONT] for i in range(0, len(starts), 20000)]
        val = windows(idx, window_starts(idx, val_eps, h, stride=h), h, stats)[..., CONT]
        meta[f"n_train_{h}"] = len(starts)
        meta[f"n_val_{h}"] = len(val)
        print(f"H={h}: {len(starts)} train windows, {len(val)} val windows", flush=True)
        for name, waves, kind in candidates():
            for mode in MODES:
                row, S, groups = evaluate(name, waves, kind, mode, h, train_chunks, val, targets)
                rows.append(row)
                if h == 32 and mode == "periodization":
                    atoms[name] = (S, groups)
                print(f"  {name:<26} {mode:<13} tokens@FAST adaptive {row['tokens_adaptive@fast']:7.1f}  fixed {row['tokens_fixed@fast']:7.1f}  "
                      f"PR {row['pr_error']:.0e}", flush=True)
    add_scores(rows)
    meta["runtime_s"] = time.time() - t0
    RES.mkdir(exist_ok=True)
    (RES / "wavelet_selection.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1, default=float))
    flat = [k for k in rows[0] if not isinstance(rows[0][k], (list, dict))]
    keys = flat + [k for r in rows for k in r if k.startswith("op_") and k not in flat and k != "op_depth_usage"]
    keys = list(dict.fromkeys(keys)) + ["score"]
    with open(RES / "wavelet_selection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "wavelet_selection.html").write_text(build_html(rows, meta, atoms))
    print(f"wrote {REPORTS / 'wavelet_selection.html'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

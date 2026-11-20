"""How much do a trajectory's strongest coefficients move when the trajectory is edited?

Companion to transform_shift_analysis.py (same shapes, edits, transforms and equal-accuracy quantizer). For every shape
take its top 3% coefficients by |quantized bin| (k = ceil(0.03 N), at least 1), apply each of the 9 edits, quantize again
and measure, per transform:
  bin shift       mean |bin change| of the original top-k coefficients (same coefficient positions before and after)
  relative shift  the same divided by the original |bin| of those coefficients
  set change      share of the original top-k positions that are no longer in the edited shape's top-k
  rank shift      the top-k positions sorted by strength before and after, paired by rank; mean |position difference|
                  (in coefficient-index units; for wavelets index distance mixes scale and time, so read it with care)
Outputs experiment/reports/transform_topk_shift_analysis.html and results/transform_topk_shift.json.
Run with the VLA env:  python scripts/transform_topk_shift_analysis.py
"""

from __future__ import annotations

import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from transform_shift_analysis import (  # noqa: E402
    COLORS, CSS, EDITS, MAIN_STEP, REPORTS, RES, esc, family, fig_html, grid, normalized, random_shape, table, transforms)

SIZES = (32, 64, 128)
MAIN_N = 64
N_SHAPES = 400
TOP_FRAC = 0.03


def topk_metrics(T, X0, X1, step, k):
    q = step / np.linalg.norm(np.linalg.inv(T), axis=0)
    b0, b1 = np.round(X0 @ T.T / q), np.round(X1 @ T.T / q)
    rows = np.arange(len(b0))[:, None]
    i0 = np.argsort(-np.abs(b0), axis=1, kind="stable")[:, :k]
    i1 = np.argsort(-np.abs(b1), axis=1, kind="stable")[:, :k]
    a0 = np.abs(b0[rows, i0])
    d = np.abs(b1[rows, i0] - b0[rows, i0])
    left = np.array([len(set(x) - set(y)) for x, y in zip(i0, i1)]) / k
    return {"bin_shift": d.mean(axis=1), "rel_shift": (d / np.maximum(a0, 1)).mean(axis=1), "set_change": left,
            "rank_shift": np.abs(i0 - i1).mean(axis=1), "top_bin": a0.mean(axis=1)}


def run():
    out = {}
    for n in SIZES:
        rng = np.random.default_rng(0)
        T = transforms(n, rng)
        t = grid(n)
        shapes = [normalized(random_shape(rng)) for _ in range(N_SHAPES)]
        us = rng.random(N_SHAPES)
        X0 = np.array([f(t) for f in shapes])
        k = max(1, math.ceil(TOP_FRAC * n))
        res = {}
        for name, M in T.items():
            per = {}
            for ed, fn in EDITS.items():
                X1 = np.array([fn(f, t, u) for f, u in zip(shapes, us)])
                per[ed] = {m: float(v.mean()) for m, v in topk_metrics(M, X0, X1, MAIN_STEP, k).items()}
            res[name] = {"family": family(name), "edits": per,
                         **{m: float(np.mean([p[m] for p in per.values()])) for m in next(iter(per.values()))}}
        out[str(n)] = {"k": k, "results": res}
        print(f"N={n} k={k}: " + ", ".join(f"{a} {res[a]['bin_shift']:.1f}" for a in
                                          sorted(res, key=lambda a: -res[a]["bin_shift"])), flush=True)
    return out


def bar(res, key, label, fmt):
    names = sorted(res, key=lambda a: -res[a][key])
    v = [res[a][key] for a in names]
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(names) + 1))
    ax.barh(range(len(names)), v, color=[COLORS[res[a]["family"]] for a in names])
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    for i, x in enumerate(v):
        ax.text(x, i, " " + fmt(x), va="center", fontsize=8)
    ax.set_xlabel(label, fontsize=9)
    ax.set_xlim(0, max(v) * 1.15)
    ax.legend(handles=[Patch(color=c, label=f) for f, c in COLORS.items()], fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return fig_html(fig, label)


def write_report(out):
    main = out[str(MAIN_N)]
    res, k = main["results"], main["k"]
    rank = sorted(res, key=lambda a: -res[a]["bin_shift"])
    pos = {a: i + 1 for i, a in enumerate(rank)}
    d, hr = res["DCT-II"], res["Haar"]
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Shift of the strongest coefficients</title>"
         f"<style>{CSS}</style></head><body>",
         "<h1>How much do the strongest coefficients move when a trajectory is edited?</h1>",
         f"<p class='sub'>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/transform_topk_shift_analysis.py</code>. "
         "Synthetic shapes only; companion to <a href='transform_shift_analysis.html'>the token-shift analysis</a> and "
         "<a href='transform_fingerprint_analysis.html'>the fingerprint analysis</a>.</p>",
         f"<p><b>Result (N = {MAIN_N}, top {k} coefficients = 3%).</b> After an edit, the original top-3% coefficients move on "
         f"average {d['bin_shift']:.1f} bins with DCT-II (rank {pos['DCT-II']} of {len(rank)}) and {hr['bin_shift']:.1f} bins with "
         f"Haar (rank {pos['Haar']}); largest: {esc(rank[0])} ({res[rank[0]]['bin_shift']:.1f}). Relative to their own size that is "
         f"{100 * d['rel_shift']:.0f}% for DCT and {100 * hr['rel_shift']:.0f}% for Haar. {100 * d['set_change']:.0f}% (DCT) and "
         f"{100 * hr['set_change']:.0f}% (Haar) of the top-3% positions are replaced by other coefficients.</p>",
         "<h2>Method</h2>"
         f"<p>{N_SHAPES} random smooth shapes (cosine + cubic + Gaussian bump + tanh), the 9 visible edits and the 18 transforms of "
         f"the token-shift analysis, equal-accuracy quantizer step {MAIN_STEP} on a range of 2 (so one bin means the same "
         "reconstruction error in every transform). For each shape the top 3% coefficients by |bin| are taken "
         f"(k = ceil(0.03 N): {', '.join(f'N = {n}: {out[str(n)]['k']}' for n in SIZES)}), the edit is applied, and the "
         "coefficients are quantized again.</p><ul>"
         "<li><b>Bin shift</b>: mean |bin change| of the original top-k coefficients (same positions before and after).</li>"
         "<li><b>Relative shift</b>: bin shift divided by those coefficients' original |bin|.</li>"
         "<li><b>Set change</b>: share of the original top-k positions that drop out of the edited shape's top-k.</li>"
         "<li><b>Rank shift</b>: top-k positions sorted by strength before and after, paired by rank, mean |index difference|. "
         "For wavelets an index distance mixes scale and time, so this one is only comparable within a family.</li></ul>",
         f"<h2>Average bin shift of the top 3%, N = {MAIN_N}</h2>",
         bar(res, "bin_shift", f"mean |bin change| of the top {k} coefficients (bins)", lambda x: f"{x:.1f}")]
    rows = [[f"{pos[a]}", f"<b>{esc(a)}</b>" if a in ("DCT-II", "Haar") else esc(a), esc(res[a]["family"]),
             f"{res[a]['top_bin']:.0f}", f"{res[a]['bin_shift']:.1f}", f"{100 * res[a]['rel_shift']:.0f}%",
             f"{100 * res[a]['set_change']:.0f}%", f"{res[a]['rank_shift']:.2f}"] for a in rank]
    h.append(table(["rank", "transform", "family", "original |bin| of top-k", "bin shift", "relative shift", "set change",
                    "rank shift (index)"], rows))
    h.append("<h2>Bin shift at every window size</h2>")
    h.append(table(["transform", *[f"N = {n} (k = {out[str(n)]['k']})" for n in SIZES], *[f"relative, N = {n}" for n in SIZES],
                    *[f"set change, N = {n}" for n in SIZES]],
                   [[esc(a), *[f"{out[str(n)]['results'][a]['bin_shift']:.1f}" for n in SIZES],
                     *[f"{100 * out[str(n)]['results'][a]['rel_shift']:.0f}%" for n in SIZES],
                     *[f"{100 * out[str(n)]['results'][a]['set_change']:.0f}%" for n in SIZES]] for a in rank]))
    eds = list(EDITS)
    names = ["orthogonal polynomials", "DCT-II", "PCA (learned)", "bior4.4", "db4", "Walsh–Hadamard", "Haar", "raw samples"]
    h.append(f"<h2>Bin shift per edit, N = {MAIN_N}</h2>")
    h.append(table(["transform", *eds], [[esc(a), *[f"{res[a]['edits'][e]['bin_shift']:.1f}" for e in eds]] for a in names]))
    h.append(f"<h2>Set change per edit, N = {MAIN_N}</h2>")
    h.append(table(["transform", *eds], [[esc(a), *[f"{100 * res[a]['edits'][e]['set_change']:.0f}%" for e in eds]]
                                         for a in names]))
    h.append("<h2>Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla/scripts\n"
             "~/miniconda3/envs/smooth-operator-vla/bin/python transform_topk_shift_analysis.py</pre></body></html>")
    path = REPORTS / "transform_topk_shift_analysis.html"
    path.write_text("\n".join(h))
    print("wrote", path)


def main():
    out = run()
    RES.mkdir(exist_ok=True)
    (RES / "transform_topk_shift.json").write_text(json.dumps({"sizes": SIZES, "step": MAIN_STEP, "top_frac": TOP_FRAC,
                                                               "edits": list(EDITS), "by_n": out}, indent=1))
    write_report(out)


if __name__ == "__main__":
    main()

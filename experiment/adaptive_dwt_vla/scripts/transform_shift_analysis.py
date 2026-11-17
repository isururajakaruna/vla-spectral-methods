"""Which transform changes the fewest coefficients when a smooth shape is edited in a visible way?

Synthetic study, no dataset content. Random smooth shapes (cosine + cubic polynomial + Gaussian bump + tanh step,
each part present at random) are edited in recognizable ways (slope up/down, offset, growing or decaying amplitude,
local bump, time shift, frequency drift, added curvature). Every transform is written as an N x N analysis matrix T
(coefficients c = T x) with synthesis S = T^-1, so all are compared under identical rules:

  quantizer   uniform, step_i = s / ||S[:, i]||: every coefficient's rounding adds the same reconstruction error,
              so equal steps mean equal accuracy across transforms (for orthonormal ones step_i = s)
  shifted     a coefficient whose quantized bin differs between the original and the edited shape (= a token that
              changes); reported as a fraction of all N coefficients
  N90         fewest coefficients holding 90% of the change, measured as reconstruction energy (|dc_i| ||S[:, i]||)^2

Ranked by the mean shifted fraction over all edits and shapes. Outputs experiment/reports/transform_shift_analysis.html
and results/transform_shift.json.
Run with the VLA env:  python scripts/transform_shift_analysis.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pywt  # noqa: E402
from scipy.fft import dct, dst  # noqa: E402
from scipy.linalg import hadamard  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORTS = ROOT.parent / "reports"
SIZES = (16, 32, 64)
MAIN_N = 32
N_SHAPES = 400
STEPS = (0.01, 0.02, 0.05)  # quantizer steps s, in units of a shape whose peak-to-peak is 2
MAIN_STEP = 0.02
WAVELETS = ["haar", "db2", "db4", "db6", "sym4", "sym6", "coif1", "coif2", "bior2.2", "bior4.4", "dmey"]
FAMILY = {"raw samples": "baseline", "DCT-II": "Fourier family", "DST-II": "Fourier family", "real DFT": "Fourier family",
          "Walsh–Hadamard": "other", "orthogonal polynomials": "polynomial", "PCA (learned)": "data-driven"}
COLORS = {"wavelet": "#2a7ab9", "Fourier family": "#e08e0b", "polynomial": "#3c9d5d", "data-driven": "#8e44ad",
          "baseline": "#777777", "other": "#c0392b"}


# ----------------------------------------------------------------------------------------------- transforms


def wavelet_matrix(n, w):
    levels = int(np.log2(n)) - 1  # coarsest band holds 2 coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols = [np.concatenate(pywt.wavedec(e, w, mode="periodization", level=levels)) for e in np.eye(n)]
    return np.array(cols).T


def real_dft_matrix(n):
    t = np.arange(n)
    rows = [np.ones(n) / np.sqrt(n)]
    for k in range(1, (n + 1) // 2):
        rows += [np.sqrt(2 / n) * np.cos(2 * np.pi * k * t / n), np.sqrt(2 / n) * np.sin(2 * np.pi * k * t / n)]
    if n % 2 == 0:
        rows.append(np.cos(np.pi * t) / np.sqrt(n))
    return np.array(rows)


def walsh_matrix(n):
    h = hadamard(n) / np.sqrt(n)
    return h[np.argsort((np.diff(h, axis=1) != 0).sum(axis=1))]  # sequency order


def poly_matrix(n):
    t = np.linspace(-1, 1, n)
    q, _ = np.linalg.qr(np.polynomial.legendre.legvander(t, n - 1))
    q *= np.sign(q[-1])  # deterministic signs
    return q.T


def pca_matrix(n, rng):
    x = np.array([render(random_shape(rng), grid(n)) for _ in range(4000)])
    x -= x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt


def transforms(n, rng):
    out = {"raw samples": np.eye(n), "DCT-II": dct(np.eye(n), norm="ortho", axis=0),
           "DST-II": dst(np.eye(n), norm="ortho", axis=0), "real DFT": real_dft_matrix(n),
           "Walsh–Hadamard": walsh_matrix(n), "orthogonal polynomials": poly_matrix(n), "PCA (learned)": pca_matrix(n, rng)}
    for w in WAVELETS:
        out[w if w != "haar" else "Haar"] = wavelet_matrix(n, w)
    return out


def family(name):
    return FAMILY.get(name, "wavelet")


# ----------------------------------------------------------------------------------------------- shapes and edits


def grid(n):
    return (np.arange(n) + 0.5) / n


def random_shape(rng):
    """Smooth shape as parameters; every part is present with probability 0.7 (at least one part)."""
    on = rng.random(4) < 0.7
    if not on.any():
        on[rng.integers(4)] = True
    return {"cos": (rng.uniform(0.3, 1.0), rng.uniform(0.3, 2.5), rng.uniform(0, 2 * np.pi)) if on[0] else None,
            "poly": tuple(rng.normal(0, [0.3, 0.6, 0.6, 0.6])) if on[1] else None,
            "bump": (rng.uniform(-1, 1), rng.uniform(0.15, 0.85), rng.uniform(0.08, 0.25)) if on[2] else None,
            "tanh": (rng.uniform(-1, 1), rng.uniform(0.2, 0.8), rng.uniform(0.08, 0.3)) if on[3] else None}


def render(p, t):
    x = np.zeros_like(t)
    if p["cos"]:
        a, f, ph = p["cos"]
        x += a * np.cos(2 * np.pi * f * t + ph)
    if p["poly"]:
        x += np.polynomial.polynomial.polyval(t - 0.5, p["poly"])
    if p["bump"]:
        a, m, w = p["bump"]
        x += a * np.exp(-((t - m) ** 2) / (2 * w**2))
    if p["tanh"]:
        a, m, w = p["tanh"]
        x += a * np.tanh((t - m) / w)
    return x


def normalized(p):
    """Shape function scaled to peak-to-peak 2 and centred, measured on a fine grid."""
    tf = np.linspace(0, 1, 2001)
    xf = render(p, tf)
    lo, hi = xf.min(), xf.max()
    scale = 2 / max(hi - lo, 1e-6)
    return lambda t: (render(p, t) - (lo + hi) / 2) * scale


# Each edit maps (shape function f, t, rng-drawn parameter) to the edited samples; sizes are about 15-30% of the
# peak-to-peak range, clearly visible in a plot.
EDITS = {
    "slope up": lambda f, t, u: f(t) + 0.6 * (t - 0.5),
    "slope down": lambda f, t, u: f(t) - 0.6 * (t - 0.5),
    "offset": lambda f, t, u: f(t) + 0.3,
    "growing amplitude": lambda f, t, u: f(t) * (1 + 0.6 * t),
    "decaying amplitude": lambda f, t, u: f(t) * (1 - 0.5 * t),
    "added curvature": lambda f, t, u: f(t) + 1.2 * ((t - 0.5) ** 2 - 1 / 12),
    "local bump": lambda f, t, u: f(t) + 0.4 * np.exp(-((t - (0.2 + 0.6 * u)) ** 2) / (2 * 0.06**2)),
    "time shift (delay 6%)": lambda f, t, u: f(t - 0.06),
    "frequency drift (time warp)": lambda f, t, u: f(t**1.25),
}
EDIT_KIND = {"slope up": "global, smooth", "slope down": "global, smooth", "offset": "global, smooth",
             "growing amplitude": "global, evolving", "decaying amplitude": "global, evolving",
             "added curvature": "global, smooth", "local bump": "local", "time shift (delay 6%)": "global, evolving",
             "frequency drift (time warp)": "global, evolving"}


# ----------------------------------------------------------------------------------------------- metrics


def measure(T, X0, X1, step):
    """Shifted-token fraction, N90 of the change, and nonzero tokens of the original, per shape."""
    S = np.linalg.inv(T)
    snorm = np.linalg.norm(S, axis=0)
    q = step / snorm
    c0, c1 = X0 @ T.T, X1 @ T.T
    b0, b1 = np.round(c0 / q), np.round(c1 / q)
    shifted = (b0 != b1).mean(axis=1)
    e = ((c1 - c0) * snorm) ** 2
    e = -np.sort(-e, axis=1)
    cum = np.cumsum(e, axis=1) / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
    n90 = (cum < 0.9).sum(axis=1) + 1
    nonzero = (b0 != 0).mean(axis=1)
    return shifted, n90, nonzero


def run(n, rng):
    T = transforms(n, rng)
    shapes = [normalized(random_shape(rng)) for _ in range(N_SHAPES)]
    us = rng.random(N_SHAPES)
    t = grid(n)
    X0 = np.array([f(t) for f in shapes])
    res = {}
    for name, M in T.items():
        res[name] = {"family": family(name), "edits": {}}
        for ed, fn in EDITS.items():
            X1 = np.array([fn(f, t, u) for f, u in zip(shapes, us)])
            row = {}
            for s in STEPS:
                sh, n90, nz = measure(M, X0, X1, s)
                row[f"shifted@{s}"] = float(sh.mean())
                if s == MAIN_STEP:
                    row["shifted_sd"] = float(sh.std())
                    row["n90"] = float(n90.mean())
                    row["nonzero"] = float(nz.mean())
            res[name]["edits"][ed] = row
        for key in [f"shifted@{s}" for s in STEPS] + ["n90", "nonzero"]:
            res[name][key] = float(np.mean([r[key] for r in res[name]["edits"].values()]))
    return res, T, shapes, us


# ----------------------------------------------------------------------------------------------- report


def esc(x):
    return html.escape(str(x))


def fig_html(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table(header, rows):
    out = ["<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return "\n".join(out) + "</tbody></table>"


CSS = """body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;
color:#1d2433;line-height:1.5}h1{font-size:1.6rem}h2{margin-top:2.2rem;border-bottom:1px solid #dde3ec;padding-bottom:.3rem}
table{border-collapse:collapse;margin:1rem 0;font-size:.88rem}th,td{border:1px solid #dde3ec;padding:.3rem .55rem;text-align:right}
th{background:#f3f6fa}td:first-child,th:first-child{text-align:left}img{max-width:100%}.sub{color:#5b6475}
.hl td{background:#fff7e0}code{background:#f3f6fa;padding:0 .25rem;border-radius:3px}"""


def example_index(shapes):
    """First shape with two turning points: a visibly wavy example."""
    tf = np.linspace(0, 1, 400)
    for i, f in enumerate(shapes):
        if (np.diff(np.sign(np.diff(f(tf)))) != 0).sum() == 2:
            return i
    return 0


def plot_examples(T, shapes, us, n):
    t, tf = grid(n), np.linspace(0, 1, 400)
    ix = example_index(shapes)
    f = shapes[ix]
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, (ed, fn) in zip(axes.ravel(), EDITS.items()):
        ax.plot(tf, f(tf), color="#777", lw=1.6, label="original")
        ax.plot(tf, fn(f, tf, us[ix]), color="#d9534f", lw=1.6, label="edited")
        ax.plot(t, f(t), ".", color="#777", ms=4)
        ax.plot(t, fn(f, t, us[ix]), ".", color="#d9534f", ms=4)
        ax.set_title(ed, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"One random smooth shape and its nine edits (dots: the N = {n} samples every transform sees)", fontsize=11)
    return fig_html(fig, "example edits")


def plot_coef_changes(T, shapes, us, n):
    t = grid(n)
    ix = example_index(shapes)
    f = shapes[ix]
    names = ["raw samples", "DCT-II", "orthogonal polynomials", "Haar", "db4", "PCA (learned)"]
    eds = ["slope up", "local bump", "time shift (delay 6%)"]
    fig, axes = plt.subplots(len(eds), len(names), figsize=(15, 7), sharex=True)
    for i, ed in enumerate(eds):
        x0, x1 = f(t), EDITS[ed](f, t, us[ix])
        for j, nm in enumerate(names):
            M = T[nm]
            snorm = np.linalg.norm(np.linalg.inv(M), axis=0)
            q = MAIN_STEP / snorm
            dc = (M @ x1 - M @ x0) * snorm
            moved = np.round(M @ x1 / q) != np.round(M @ x0 / q)
            ax = axes[i, j]
            ax.bar(np.arange(n), dc, color=np.where(moved, "#d9534f", "#c8ced8"), width=0.8)
            ax.axhline(0, color="#333", lw=0.6)
            if i == 0:
                ax.set_title(nm, fontsize=10)
            if j == 0:
                ax.set_ylabel(ed, fontsize=9)
            ax.text(0.98, 0.92, f"{moved.sum()}/{n} shifted", transform=ax.transAxes, ha="right", fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle("Change of each coefficient (scaled by its reconstruction weight); red = its quantized token changes", fontsize=11)
    return fig_html(fig, "coefficient changes")


def plot_ranking(res, n):
    names = sorted(res, key=lambda k: res[k][f"shifted@{MAIN_STEP}"])
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(names) + 1))
    v = [100 * res[k][f"shifted@{MAIN_STEP}"] for k in names]
    ax.barh(range(len(names)), v, color=[COLORS[res[k]["family"]] for k in names])
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    for i, x in enumerate(v):
        ax.text(x + 0.6, i, f"{x:.0f}%", va="center", fontsize=8)
    ax.set_xlabel(f"tokens shifted by an edit (% of the {n} coefficients, mean over edits and shapes)")
    ax.set_xlim(0, 105)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=fam) for fam, c in COLORS.items()], fontsize=8, loc="center right")
    ax.grid(axis="x", alpha=0.3)
    return fig_html(fig, "ranking")


def plot_heatmap(res, n):
    names = sorted(res, key=lambda k: res[k][f"shifted@{MAIN_STEP}"])
    eds = list(EDITS)
    m = np.array([[100 * res[k]["edits"][e][f"shifted@{MAIN_STEP}"] for e in eds] for k in names])
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(names) + 1.8))
    im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(eds)), eds, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(eds)):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="black" if m[i, j] < 60 else "white")
    fig.colorbar(im, ax=ax, label="% tokens shifted")
    ax.set_title(f"Tokens shifted per edit (N = {n}, step {MAIN_STEP})", fontsize=11)
    return fig_html(fig, "heatmap")


def write_report(allres, T, shapes, us):
    res = allres[MAIN_N]
    n = MAIN_N
    rank = sorted(res, key=lambda k: res[k][f"shifted@{MAIN_STEP}"])
    pos = {k: i + 1 for i, k in enumerate(rank)}
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Which transform shifts the fewest tokens?</title>"
         f"<style>{CSS}</style></head><body>",
         "<h1>Which transform shifts the fewest tokens when a smooth motion is edited?</h1>",
         f"<p class='sub'>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/transform_shift_analysis.py</code>. "
         "Synthetic shapes only; no dataset content.</p>"]
    most, least = rank[-1], rank[0]
    h.append(f"<p><b>Result.</b> With every transform held to the same accuracy, a visible edit of a smooth shape changes on average "
             f"{100 * res[least][f'shifted@{MAIN_STEP}']:.0f}% of the tokens for the best transform ({esc(least)}) and "
             f"{100 * res[most][f'shifted@{MAIN_STEP}']:.0f}% for the worst ({esc(most)}). DCT-II ranks {pos['DCT-II']} and Haar "
             f"{pos['Haar']} of {len(rank)}: DCT shifts {100 * res['DCT-II'][f'shifted@{MAIN_STEP}']:.0f}% of its tokens, Haar "
             f"{100 * res['Haar'][f'shifted@{MAIN_STEP}']:.0f}%. Smooth global edits (slopes, curvature, amplitude changes) are compact "
             "in the smooth global bases (polynomials, DCT, learned PCA) and spread over every level of a wavelet. The local bump is the "
             "only edit where the polynomial basis falls to the bottom; there raw samples do best and the wavelets and DCT are close "
             f"(Haar {100 * res['Haar']['edits']['local bump'][f'shifted@{MAIN_STEP}']:.0f}%, DCT "
             f"{100 * res['DCT-II']['edits']['local bump'][f'shifted@{MAIN_STEP}']:.0f}%).</p>")
    h.append("<h2>1. Method</h2>")
    h.append(f"<p><b>Shapes.</b> {N_SHAPES} random smooth shapes per signal length: a cosine (frequency 0.3–2.5 cycles per window), a cubic "
             "polynomial, a Gaussian bump (width 0.08–0.25) and a tanh step (width 0.08–0.3), each part present with probability 0.7, "
             "then scaled to peak-to-peak 2. No sharp edges. Sampled at N points, the length a chunk tokenizer sees.</p>")
    h.append("<p><b>Edits</b> (sizes about 15–30% of the range, clearly visible):</p>")
    h.append(table(["edit", "kind", "definition (t in [0, 1])"], [
        ["slope up / down", "global, smooth", "x(t) ± 0.6 (t − 0.5)"], ["offset", "global, smooth", "x(t) + 0.3"],
        ["growing / decaying amplitude", "global, evolving", "x(t)(1 + 0.6t) / x(t)(1 − 0.5t)"],
        ["added curvature", "global, smooth", "x(t) + 1.2((t − 0.5)² − 1/12)"],
        ["local bump", "local", "x(t) + 0.4 exp(−(t − μ)² / 2·0.06²), μ uniform in [0.2, 0.8]"],
        ["time shift", "global, evolving", "x(t − 0.06): the motion starts 6% later"],
        ["frequency drift", "global, evolving", "x(t^1.25): the motion speeds up through the window"]]))
    h.append(f"<p><b>Transforms</b> ({len(rank)}), each as an N × N matrix: raw samples (no transform); DCT-II (the transform inside FAST), DST-II, a real "
             "DFT; Walsh–Hadamard (sequency order); orthogonal polynomials (Legendre, orthonormalized on the sample grid); PCA learned from "
             f"4,000 other random shapes of the same family (the best possible linear basis for these shapes); and {len(WAVELETS)} wavelets "
             "in periodization mode, decomposed until the coarsest band holds 2 coefficients: " + ", ".join(WAVELETS) +
             ". The biorthogonal ones (bior2.2, bior4.4) are not orthonormal; the step rule below keeps them comparable.</p>")
    h.append("<p><b>Tokens and the metric.</b> Every coefficient c<sub>i</sub> is quantized with step s / ‖S<sub>:,i</sub>‖, where S is the "
             "inverse transform; each token's rounding then adds the same reconstruction error, so equal s means equal accuracy for every "
             f"transform (orthonormal transforms simply use step s). Main setting s = {MAIN_STEP} on a range of 2 (about 1%), also "
             f"checked at {', '.join(str(s) for s in STEPS if s != MAIN_STEP)}. A token is <b>shifted</b> when its quantized bin differs "
             "between the original and the edited shape. Lower is better if we want a behaviour change to touch few tokens. "
             "N90 counts the fewest coefficients that carry 90% of the change, measured as reconstruction energy.</p>")
    h.append("<h2>2. Example</h2>")
    h.append(plot_examples(T, shapes, us, n))
    h.append(plot_coef_changes(T, shapes, us, n))
    h.append("<p>The slope edit is a single degree-1 polynomial term, a few low DCT terms, but a non-zero detail coefficient at every "
             "position of every Haar level, because a ramp has a constant non-zero difference everywhere. The local bump is the "
             "opposite case: Haar's changes stay near the bump, while the polynomial basis needs many terms to build a narrow bump. "
             "Edits that change the whole motion over time (time shift, frequency drift, amplitude) move a large share of the tokens "
             "in every basis.</p>")
    h.append(f"<h2>3. Ranking (N = {n})</h2>")
    h.append(plot_ranking(res, n))
    body = []
    for k in rank:
        r = res[k]
        body.append([f"{pos[k]}", f"<b>{esc(k)}</b>" if k in ("DCT-II", "Haar") else esc(k), esc(r["family"]),
                     f"{100 * r[f'shifted@{MAIN_STEP}']:.1f}%", *[f"{100 * r[f'shifted@{s}']:.1f}%" for s in STEPS if s != MAIN_STEP],
                     f"{r['n90']:.1f}", f"{100 * r['nonzero']:.0f}%"])
    h.append(table(["rank", "transform", "family", f"shifted (s = {MAIN_STEP})", *[f"shifted (s = {s})" for s in STEPS if s != MAIN_STEP],
                    f"N90 of the change (of {n})", "non-zero tokens of the original"], body))
    h.append("<p>The non-zero column shows how compactly each transform writes the shape itself (fewer non-zero tokens = sparser). "
             "The ranking by shifted tokens is nearly the same at all three quantizer steps.</p>")
    h.append(plot_heatmap(res, n))
    h.append("<h2>4. Signal length</h2>")
    body = []
    for k in rank:
        cells = []
        for nn in SIZES:
            rr = allres[nn]
            rk = sorted(rr, key=lambda x: rr[x][f"shifted@{MAIN_STEP}"])
            cells += [f"{100 * rr[k][f'shifted@{MAIN_STEP}']:.0f}%", f"{rk.index(k) + 1}"]
        body.append([esc(k), *cells])
    h.append(table(["transform", *[c for nn in SIZES for c in (f"N = {nn} shifted", f"N = {nn} rank")]], body))
    h.append("<p>Longer windows lower the shifted fraction for every transform except raw samples, but the order barely changes: "
             "orthogonal polynomials, learned PCA and DCT lead at every length, the smooth wavelets (db4, sym, coif, bior4.4) come next, "
             "and Haar stays second to last, ahead only of raw samples.</p>")
    h.append("<h2>5. What this does and does not say</h2><ul>")
    for c in [
        "It measures how many tokens a visible, smooth change of the motion touches. Fewer shifted tokens means a behaviour variant "
        "differs from the original in fewer places of the token sequence, which plausibly makes related motions easier to learn and "
        "share across tasks; this is a hypothesis, not tested here.",
        "It agrees with the full-LIBERO result where DCT without BPE learned much faster than Haar DWT fixed (77% vs 58.5% after one "
        "epoch, both plain cross-entropy), and with the single-task result going the other way: on one task the policy only has to "
        "reproduce one motion, where variation across motions matters little.",
        "The shapes are smooth by construction. Real action chunks also contain contact events and gripper switches, which favour local "
        "bases; the wavelet-selection report measures compactness on real LIBERO chunks instead.",
        "Our deployed Haar codec splits a 10-step chunk into 8 + 2 segments and drops fine levels adaptively; this study uses whole-window "
        "transforms with every coefficient kept, so it ranks bases, not the deployed codecs.",
        "PCA is learned from the same shape family (not from the edits), so it is a data-adapted reference rather than a fair "
        "competitor; it compresses the shapes best but is not the best for the edits.",
        "Boundary handling matters. DCT-II implicitly mirrors the window, so a slope stays smooth at the edges; the real DFT and the "
        "longer periodized wavelets treat the window as periodic, so a slope becomes a jump at the edge (the DFT's poor slope scores). "
        "Haar's 2-tap filters never cross the window edge, so its slope cost comes from the ramp itself, not the boundary.",
    ]:
        h.append(f"<li>{esc(c)}</li>")
    h.append("</ul><h2>6. Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla\n"
             "~/miniconda3/envs/smooth-operator-vla/bin/python scripts/transform_shift_analysis.py   # about a minute on CPU</pre>")
    h.append("</body></html>")
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "transform_shift_analysis.html"
    out.write_text("\n".join(h))
    print("wrote", out)


def main():
    allres, keep = {}, None
    for n in SIZES:
        rng = np.random.default_rng(0)
        res, T, shapes, us = run(n, rng)
        allres[n] = res
        if n == MAIN_N:
            keep = (T, shapes, us)
        rk = sorted(res, key=lambda k: res[k][f"shifted@{MAIN_STEP}"])
        print(f"N={n}: " + ", ".join(f"{k} {100 * res[k][f'shifted@{MAIN_STEP}']:.0f}%" for k in rk), flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "transform_shift.json").write_text(json.dumps({"sizes": SIZES, "steps": STEPS, "main_step": MAIN_STEP, "n_shapes": N_SHAPES,
                                                          "edits": list(EDITS), "results": {str(k): v for k, v in allres.items()}}, indent=1))
    write_report(allres, *keep)


if __name__ == "__main__":
    main()

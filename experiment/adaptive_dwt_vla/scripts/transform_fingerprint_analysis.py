"""Which transform gives each trajectory change the most distinctive token fingerprint?

Companion to transform_shift_analysis.py (same shapes, edits, transforms and equal-accuracy quantizer). There we counted
how many tokens an edit moves. Here we ask whether different edits leave different, recognizable token patterns, and
whether a small edit stays recognizable when the trajectory also carries small irrelevant noise (demo-to-demo jitter).

For an orthonormal transform the Euclidean distance between two trajectories is the same in every basis, so before
quantization every orthonormal transform separates trajectories equally well. What differs is how the difference is
split over tokens, which is what a token model sees: a categorical decoder compares tokens as symbols.

Metrics per transform (N = 32, step 0.02 on a range of 2), at full edit size and at a quarter of it (subtle edits):
  peak jump      largest |bin change| of any token, mean over shapes and edits
  top-3 share    share of the total |bin change| carried by the 3 most-changed tokens
  overlap        Jaccard overlap of the sets of token positions moved by two different edits of the same shape
                 (low = different edits touch different tokens = more diverse fingerprints)
  identified     the original shape and its 9 edits give 10 clean token sequences; each one is perturbed by nuisance
                 noise and matched back to the nearest clean sequence, by token mismatches (Hamming, the categorical view)
                 and by summed |bin difference| (the ordinal view); accuracy, chance = 10%
Outputs experiment/reports/transform_fingerprint_analysis.html and results/transform_fingerprint.json.
Run with the VLA env:  python scripts/transform_fingerprint_analysis.py
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from transform_shift_analysis import (  # noqa: E402
    COLORS, CSS, EDITS, MAIN_STEP, REPORTS, RES, esc, family, fig_html, grid, normalized, random_shape, table, transforms)

N = 32
N_SHAPES = 400
SCALES = (1.0, 0.25)
REPEATS = 4
NOISES = {
    "smooth jitter": "a random smooth shape of the same family scaled to peak-to-peak 0.06 (3% of the range) plus white noise "
                     "of standard deviation 0.003",
    "white jitter": "independent noise on every sample, standard deviation 0.01 (0.5% of the range)",
}


def nuisance(kind, rng, n_rows, t):
    if kind == "white jitter":
        return rng.normal(0, 0.01, (n_rows, len(t)))
    return np.array([0.03 * normalized(random_shape(rng))(t) for _ in range(n_rows)]) + rng.normal(0, 0.003, (n_rows, len(t)))


def bins(T, X, step):
    q = step / np.linalg.norm(np.linalg.inv(T), axis=0)
    return np.round(X @ T.T / q)


def analyse(T, X, noisy, step):
    """X: (shapes, 10, N) clean original + 9 edits; noisy: kind -> (repeats, shapes, 10, N)."""
    B = bins(T, X, step)
    D = B[:, 1:] - B[:, :1]
    absd = np.abs(D)
    peak = absd.max(axis=2)
    top3 = -np.sort(-absd, axis=2)[:, :, :3].sum(axis=2) / np.maximum(absd.sum(axis=2), 1)
    moved = D != 0
    ne = moved.shape[1]
    jac = []
    for i in range(ne):
        for j in range(i + 1, ne):
            inter = (moved[:, i] & moved[:, j]).sum(axis=1)
            union = (moved[:, i] | moved[:, j]).sum(axis=1)
            jac.append(np.where(union > 0, inter / np.maximum(union, 1), np.nan))
    out = {"peak": float(peak.mean()), "top3": float(top3[absd.sum(axis=2) > 0].mean()), "overlap": float(np.nanmean(jac)),
           "peak_by_edit": peak.mean(axis=0).tolist(), "shifted": float(moved.mean())}
    truth = np.arange(X.shape[1])
    for kind, Xn in noisy.items():
        Bn = bins(T, Xn, step)
        ham = (Bn[:, :, :, None, :] != B[None, :, None, :, :]).sum(axis=-1)
        l1 = np.abs(Bn[:, :, :, None, :] - B[None, :, None, :, :]).sum(axis=-1)
        tie = np.random.default_rng(1).random(ham.shape) * 1e-3
        out[f"id_hamming|{kind}"] = float((np.argmin(ham + tie, axis=-1) == truth).mean())
        out[f"id_l1|{kind}"] = float((np.argmin(l1 + tie, axis=-1) == truth).mean())
        out[f"noise_moved|{kind}"] = float((Bn[:, :, 0] != B[None, :, 0]).mean())
    return out


def run():
    rng = np.random.default_rng(0)
    T = transforms(N, rng)
    t = grid(N)
    shapes = [normalized(random_shape(rng)) for _ in range(N_SHAPES)]
    us = rng.random(N_SHAPES)
    res = {}
    for scale in SCALES:
        X = np.array([[f(t)] + [f(t) + scale * (fn(f, t, u) - f(t)) for fn in EDITS.values()] for f, u in zip(shapes, us)])
        nrng = np.random.default_rng(7)
        noisy = {k: X[None] + np.stack([nuisance(k, nrng, X.shape[0] * X.shape[1], t).reshape(X.shape) for _ in range(REPEATS)])
                 for k in NOISES}
        res[str(scale)] = {name: {"family": family(name), **analyse(M, X, noisy, MAIN_STEP)} for name, M in T.items()}
        print(f"scale {scale}: " + ", ".join(f"{k} {100 * v['id_hamming|smooth jitter']:.0f}%" for k, v in
                                            sorted(res[str(scale)].items(), key=lambda kv: -kv[1]["id_hamming|smooth jitter"])),
              flush=True)
    return res


# ----------------------------------------------------------------------------------------------- report


def bar(res, key, label, best_high=True, pct=True):
    names = sorted(res, key=lambda k: -res[k][key] if best_high else res[k][key])
    v = [res[k][key] * (100 if pct else 1) for k in names]
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(names) + 1))
    ax.barh(range(len(names)), v, color=[COLORS[res[k]["family"]] for k in names])
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    for i, x in enumerate(v):
        ax.text(x, i, f" {x:.0f}%" if pct else f" {x:.1f}", va="center", fontsize=8)
    ax.set_xlabel(label, fontsize=9)
    ax.set_xlim(0, max(v) * 1.15)
    ax.legend(handles=[Patch(color=c, label=f) for f, c in COLORS.items()], fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return fig_html(fig, label)


def write_report(res):
    full, sub = res["1.0"], res["0.25"]
    key = "id_hamming|smooth jitter"
    rank = sorted(sub, key=lambda k: -sub[k][key])
    pos = {k: i + 1 for i, k in enumerate(rank)}
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Token fingerprints of trajectory changes</title>"
         f"<style>{CSS}</style></head><body>",
         "<h1>Which transform gives each trajectory change the most distinctive token fingerprint?</h1>",
         f"<p class='sub'>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/transform_fingerprint_analysis.py</code>. "
         "Synthetic shapes only; companion to <a href='transform_shift_analysis.html'>the token-shift analysis</a>.</p>"]
    d, hr = sub["DCT-II"], sub["Haar"]
    h.append(f"<p><b>Result.</b> With subtle edits (a quarter of the visible size) and small smooth jitter on top, the edit is recovered "
             f"from the token sequence {100 * d[key]:.0f}% of the time with DCT-II (rank {pos['DCT-II']} of {len(rank)}) and "
             f"{100 * hr[key]:.0f}% with Haar (rank {pos['Haar']}); chance is 10%. Best: {esc(rank[0])} "
             f"({100 * sub[rank[0]][key]:.0f}%). Different edits of the same shape move overlapping token sets "
             f"{100 * full['DCT-II']['overlap']:.0f}% of the time with DCT and {100 * full['Haar']['overlap']:.0f}% with Haar, DCT's "
             f"largest token jump per edit is {full['DCT-II']['peak']:.0f} bins against {full['Haar']['peak']:.0f} for Haar, and its "
             f"3 most-changed tokens carry {100 * full['DCT-II']['top3']:.0f}% of the change against {100 * full['Haar']['top3']:.0f}%. "
             "So DCT writes each edit as a few strong, edit-specific tokens, while Haar spreads it over many tokens; the spreading is "
             "why token-mismatch matching recovers subtle edits more often with Haar.</p>")
    h.append("<h2>1. Why count tokens differently from distances</h2>"
             "<p>Most transforms here are orthonormal, so they preserve distances: before quantization a trajectory change has exactly "
             "the same size in DCT, Haar or any orthonormal basis, and every one of them separates trajectories equally well. What "
             "differs is how the change is split over tokens. A token model predicts each token as a symbol from a vocabulary, so a "
             "change written as a few large jumps in specific tokens is a clear, edit-specific fingerprint, while the same change spread "
             "as ±1 nudges over most tokens looks like quantization noise and resembles every other edit. The metrics below measure "
             "that split on the quantized tokens.</p>")
    h.append("<h2>2. Method</h2>"
             f"<p>{N_SHAPES} random smooth shapes, the 9 edits and the 18 transforms of the token-shift analysis, N = {N}, equal-accuracy "
             f"quantizer step {MAIN_STEP} on a range of 2. Every metric is computed at the visible edit size and at a quarter of it "
             "(subtle edits: a slope of ±0.15 over the window, a 1.5% delay).</p><ul>"
             "<li><b>Peak jump</b>: the largest |bin change| of any token, averaged over shapes and edits.</li>"
             "<li><b>Top-3 share</b>: share of the total |bin change| carried by the 3 most-changed tokens.</li>"
             "<li><b>Overlap</b>: for two different edits of the same shape, the Jaccard overlap of the token positions each one moves "
             "(lower = edits touch different tokens).</li>"
             "<li><b>Identified</b>: the original and its 9 edits give 10 clean token sequences. Each is perturbed by nuisance noise "
             f"({REPEATS} draws) and matched back to the nearest clean sequence, either by the number of mismatched tokens (Hamming, "
             "the symbol view of a categorical decoder) or by summed |bin difference|. Accuracy; chance is 10%. Noise models: "
             + "; ".join(f"<i>{esc(k)}</i>: {esc(v)}" for k, v in NOISES.items()) + ".</li></ul>")
    h.append("<h2>3. Recognizing subtle edits under jitter</h2>")
    h.append(bar(sub, key, "subtle edit identified from tokens, smooth jitter, Hamming match (%)"))
    rows = []
    for k in rank:
        cells = [f"{pos[k]}", f"<b>{esc(k)}</b>" if k in ("DCT-II", "Haar") else esc(k), esc(sub[k]["family"])]
        for r in (sub, full):
            cells += [f"{100 * r[k][f'id_hamming|{nz}']:.0f}%" for nz in NOISES] + [f"{100 * r[k][f'id_l1|{nz}']:.0f}%" for nz in NOISES]
        rows.append(cells)
    hdr = ["rank", "transform", "family"]
    for lab in ("subtle", "visible"):
        hdr += [f"{lab}, Hamming, {nz}" for nz in NOISES] + [f"{lab}, |bin| sum, {nz}" for nz in NOISES]
    h.append(table(hdr, rows))
    h.append("<h2>4. How the change is written into tokens</h2>")
    h.append(bar(full, "overlap", "token positions shared by two different edits of a shape (Jaccard, %; lower = more diverse)",
                 best_high=False))
    rows = []
    for k in sorted(full, key=lambda x: full[x]["overlap"]):
        rows.append([f"<b>{esc(k)}</b>" if k in ("DCT-II", "Haar") else esc(k), f"{100 * full[k]['overlap']:.0f}%",
                     f"{full[k]['peak']:.1f}", f"{100 * full[k]['top3']:.0f}%", f"{100 * full[k]['shifted']:.0f}%",
                     f"{100 * sub[k]['overlap']:.0f}%", f"{sub[k]['peak']:.1f}", f"{100 * sub[k]['top3']:.0f}%",
                     f"{100 * full[k]['noise_moved|smooth jitter']:.0f}%"])
    h.append(table(["transform", "overlap (visible)", "peak jump, bins (visible)", "top-3 share (visible)", "tokens moved (visible)",
                    "overlap (subtle)", "peak jump (subtle)", "top-3 share (subtle)", "tokens moved by smooth jitter alone"], rows))
    eds = list(EDITS)
    names = ["orthogonal polynomials", "DCT-II", "bior4.4", "db4", "Walsh–Hadamard", "Haar", "raw samples"]
    h.append("<p>Peak jump per edit, visible size (bins):</p>")
    h.append(table(["transform", *eds], [[esc(k), *[f"{full[k]['peak_by_edit'][i]:.0f}" for i in range(len(eds))]] for k in names]))
    h.append("<h2>5. Reading it</h2><ul>")
    for c in [
        "Two different notions of a distinctive fingerprint pull apart. DCT (with orthogonal polynomials and Walsh–Hadamard) writes "
        "each edit as a few large jumps in its own tokens: the highest peak jump, the highest top-3 share and among the lowest "
        "overlap between edits. Haar spreads every edit over many tokens with large overlap between edits.",
        "But counting mismatched tokens rewards spreading: an edit that moves 70% of the tokens gives a nearest-match search many "
        "independent votes, so Haar recovers subtle edits under jitter more often than DCT. Concentration alone does not make "
        "edits easier to recognize by symbol matching.",
        "Matched by summed |bin difference| every transform recovers the edit 89–100% of the time: the information is equally "
        "present in all of them (distances are preserved), and the differences come only from the symbol view.",
        "So the case for DCT is about what a model has to learn, not raw separability: a family of similar motions shares most DCT "
        "tokens and differs in a few predictable ones, while in Haar the same small change rewrites most of the sequence. "
        "This study cannot tell which helps a policy more; the full-LIBERO results (DCT ahead of Haar) suggest the former.",
        "The shapes are smooth by construction; real chunks with contacts and gripper switches are less favourable to global bases.",
        "Identification uses 10 candidates per shape (the original and its 9 edits), all from the same shape; it measures separating "
        "variants of one motion, not separating unrelated motions.",
    ]:
        h.append(f"<li>{esc(c)}</li>")
    h.append("</ul><h2>6. Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla/scripts\n"
             "~/miniconda3/envs/smooth-operator-vla/bin/python transform_fingerprint_analysis.py</pre></body></html>")
    out = REPORTS / "transform_fingerprint_analysis.html"
    out.write_text("\n".join(h))
    print("wrote", out)


def main():
    res = run()
    RES.mkdir(exist_ok=True)
    (RES / "transform_fingerprint.json").write_text(json.dumps({"n": N, "step": MAIN_STEP, "scales": SCALES, "noises": NOISES,
                                                                 "edits": list(EDITS), "results": res}, indent=1))
    write_report(res)


if __name__ == "__main__":
    main()

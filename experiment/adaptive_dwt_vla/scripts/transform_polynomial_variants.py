"""Polynomial and spline bases compared on the token-shift, fingerprint and top-3% shift metrics.

Companion to transform_shift_analysis.py, transform_fingerprint_analysis.py and transform_topk_shift_analysis.py (same
shapes, edits, nuisance noise and equal-accuracy quantizer step_i = s / ||S[:, i]||, s = 0.02 on a range of 2).

Every complete polynomial basis spans the same space, and QR-orthonormalizing any of them in degree order gives the same
discrete orthogonal (Gram) polynomials, which is what the poly_fixed codec uses. The raw bases below are not orthogonal
on the sample points, so they are different transforms: the synthesis matrix S holds the basis functions sampled on the
grid and the analysis matrix is A = S^-1. Bases with condition number above 1e8 at a window size are skipped there.
Chebyshev polynomials sampled at Chebyshev nodes are exactly the DCT-II, so that variant is the DCT row.
Splines: clamped uniform B-splines with H basis functions (local support), raw and symmetrically orthonormalized
(S = B (B^T B)^-1/2, which keeps them local).

Window sizes 10 (the training horizon), 16 and 32. Outputs experiment/reports/transform_polynomial_variants.html and
results/transform_polynomial_variants.json.
Run with the VLA env:  python scripts/transform_polynomial_variants.py
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev, hermite, legendre, polynomial
from scipy.interpolate import BSpline
from scipy.linalg import sqrtm
from scipy.special import comb

from transform_fingerprint_analysis import NOISES, REPEATS, nuisance
from transform_shift_analysis import (CSS, EDITS, MAIN_STEP, REPORTS, RES, esc, grid, normalized, random_shape, table,
                                      wavelet_matrix)

sys.path.insert(0, str(RES.parent / "src"))
from lerobot_policy_adaptive_dwt.codec_window import dct_matrices, poly_matrices, transform_matrices  # noqa: E402

SIZES = (10, 16, 32)
N_SHAPES = 400
SUBTLE = 0.25
TOP_FRAC = 0.03
MAX_COND = 1e8
WINDOW_LEVELS = {10: 3, 16: 3, 32: 4}
GROUP = {"DCT-II": "reference", "Haar": "reference", "sym4→Haar (training codec)": "reference"}
DESC = {
    "Gram polynomials (orthonormal, poly_fixed)": "Legendre QR-orthonormalized on the samples; the trained poly codec",
    "Legendre (raw)": "P_k(u), u uniform in [-1, 1]",
    "Chebyshev (raw)": "T_k(u) on the uniform samples (on Chebyshev nodes it would be the DCT-II)",
    "Hermite (raw)": "H_k(2u), physicists' Hermite",
    "monomials t^k (raw)": "u^k",
    "Bernstein / Bézier (raw)": "C(H-1,k) x^k (1-x)^(H-1-k), x uniform in [0, 1]",
    "cubic B-spline (raw)": "clamped uniform knots, H basis functions, local support of 4 knot spans",
    "cubic B-spline (orthonormalized)": "S = B (B^T B)^-1/2, still nearly local",
    "linear B-spline (hat functions)": "piecewise linear; with H hats on H samples it is the identity, a sanity row",
}


def bspline(n, k):
    knots = np.r_[[0.0] * k, np.linspace(0, 1, n - k + 1), [1.0] * k]
    return BSpline.design_matrix(np.linspace(0, 1, n), knots, k).toarray()


def bases(n):
    """name -> synthesis matrix S (H, n_coef); columns are sampled basis functions."""
    u, x = np.linspace(-1, 1, n), np.linspace(0, 1, n)
    b3 = bspline(n, 3)
    out = {
        "Gram polynomials (orthonormal, poly_fixed)": poly_matrices(n)[1],
        "Legendre (raw)": legendre.legvander(u, n - 1),
        "Chebyshev (raw)": chebyshev.chebvander(u, n - 1),
        "Hermite (raw)": hermite.hermvander(2 * u, n - 1),
        "monomials t^k (raw)": polynomial.polyvander(u, n - 1),
        "Bernstein / Bézier (raw)": np.array([[comb(n - 1, k) * v**k * (1 - v) ** (n - 1 - k) for k in range(n)] for v in x]),
        "cubic B-spline (raw)": b3,
        "cubic B-spline (orthonormalized)": np.real(b3 @ np.linalg.inv(sqrtm(b3.T @ b3))),
        "linear B-spline (hat functions)": bspline(n, 1),
        "DCT-II": dct_matrices(n)[1],
    }
    if n & (n - 1) == 0:
        out["Haar"] = np.linalg.inv(wavelet_matrix(n, "haar"))
    return out


def transforms(n):
    """name -> (A, S, condition number); square bases are inverted, the training wavelet codec is overcomplete."""
    out = {}
    for name, S in bases(n).items():
        cond = float(np.linalg.cond(S / np.linalg.norm(S, axis=0)))
        if cond <= MAX_COND:
            out[name] = (np.linalg.inv(S), S, cond)
    A, S, _ = transform_matrices(["sym4", "haar"], "symmetric", n, WINDOW_LEVELS[n])
    out["sym4→Haar (training codec)"] = (A, S, float(np.linalg.cond(S / np.linalg.norm(S, axis=0))))
    return out


def bins(A, S, X):
    return np.round(X @ A.T / (MAIN_STEP / np.linalg.norm(S, axis=0)))


def metrics(A, S, X, noisy, k):
    """X: (shapes, 10, H) original + 9 edits at visible size; noisy: kind -> (repeats, shapes, 10, H) at subtle size."""
    B = bins(A, S, X)
    D = B[:, 1:] - B[:, :1]
    absd, moved = np.abs(D), D != 0
    jac = []
    for i in range(moved.shape[1]):
        for j in range(i + 1, moved.shape[1]):
            union = (moved[:, i] | moved[:, j]).sum(axis=1)
            jac.append(np.where(union > 0, (moved[:, i] & moved[:, j]).sum(axis=1) / np.maximum(union, 1), np.nan))
    tot = absd.sum(axis=2)
    top3 = -np.sort(-absd, axis=2)[:, :, :3].sum(axis=2) / np.maximum(tot, 1)
    b0 = B[:, 0]
    i0 = np.argsort(-np.abs(b0), axis=1, kind="stable")[:, :k]
    rows = np.arange(len(b0))[:, None]
    a0 = np.abs(b0[rows, i0])
    top_shift, top_rel, set_change = [], [], []
    for e in range(D.shape[1]):
        be = B[:, e + 1]
        i1 = np.argsort(-np.abs(be), axis=1, kind="stable")[:, :k]
        d = np.abs(be[rows, i0] - b0[rows, i0])
        top_shift.append(d.mean())
        top_rel.append((d / np.maximum(a0, 1)).mean())
        set_change.append(np.mean([len(set(p) - set(q)) / k for p, q in zip(i0, i1)]))
    out = {"moved": float(moved.mean()), "moved_by_edit": moved.mean(axis=(0, 2)).tolist(), "overlap": float(np.nanmean(jac)),
           "peak": float(absd.max(axis=2).mean()), "top3": float(top3[tot > 0].mean()), "nonzero": float((b0 != 0).mean()),
           "top_shift": float(np.mean(top_shift)), "top_rel": float(np.mean(top_rel)), "set_change": float(np.mean(set_change)),
           "top_shift_by_edit": [float(v) for v in top_shift]}
    return out


def identification(A, S, Xs, noisy):
    B = bins(A, S, Xs)
    truth = np.arange(Xs.shape[1])
    out = {}
    for kind, Xn in noisy.items():
        Bn = bins(A, S, Xn)
        ham = (Bn[:, :, :, None, :] != B[None, :, None, :, :]).sum(axis=-1)
        l1 = np.abs(Bn[:, :, :, None, :] - B[None, :, None, :, :]).sum(axis=-1)
        tie = np.random.default_rng(1).random(ham.shape) * 1e-3
        out[f"id_hamming|{kind}"] = float((np.argmin(ham + tie, axis=-1) == truth).mean())
        out[f"id_l1|{kind}"] = float((np.argmin(l1 + tie, axis=-1) == truth).mean())
    return out


def run():
    out = {}
    for n in SIZES:
        rng = np.random.default_rng(0)
        t = grid(n)
        shapes = [normalized(random_shape(rng)) for _ in range(N_SHAPES)]
        us = rng.random(N_SHAPES)
        X = np.array([[f(t)] + [fn(f, t, u) for fn in EDITS.values()] for f, u in zip(shapes, us)])
        Xs = X[:, :1] + SUBTLE * (X - X[:, :1])
        nrng = np.random.default_rng(7)
        noisy = {kd: Xs[None] + np.stack([nuisance(kd, nrng, Xs.shape[0] * Xs.shape[1], t).reshape(Xs.shape)
                                          for _ in range(REPEATS)]) for kd in NOISES}
        k = max(1, math.ceil(TOP_FRAC * n))
        res = {}
        for name, (A, S, cond) in transforms(n).items():
            res[name] = {"group": GROUP.get(name, "spline" if "spline" in name else "polynomial"), "cond": cond,
                         "n_coef": int(A.shape[0]), **metrics(A, S, X, noisy, k), **identification(A, S, Xs, noisy)}
        skipped = sorted(set(bases(n)) - set(res))
        out[str(n)] = {"k": k, "skipped": skipped, "results": res}
        print(f"N={n}: " + ", ".join(f"{a} moved {100 * r['moved']:.0f}% id {100 * r['id_hamming|smooth jitter']:.0f}%"
                                      for a, r in sorted(res.items(), key=lambda kv: kv[1]["moved"])), flush=True)
        if skipped:
            print(f"  skipped (cond > {MAX_COND:.0e}): {', '.join(skipped)}", flush=True)
    return out


def write_report(out):
    key = "id_hamming|smooth jitter"
    r10 = out["10"]["results"]
    gram, dct = r10["Gram polynomials (orthonormal, poly_fixed)"], r10["DCT-II"]
    best = min(r10, key=lambda a: r10[a]["moved"])
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Polynomial and spline bases</title>"
         f"<style>{CSS}</style></head><body>",
         "<h1>Polynomial and spline bases: which one moves the fewest tokens?</h1>",
         f"<p class='sub'>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/transform_polynomial_variants.py</code>. "
         "Synthetic shapes only; companion to <a href='transform_shift_analysis.html'>token shift</a>, "
         "<a href='transform_fingerprint_analysis.html'>fingerprints</a> and <a href='transform_topk_shift_analysis.html'>top-3% "
         "shift</a>.</p>",
         f"<p><b>Result at the training horizon (10 samples).</b> Fewest tokens moved by an edit: {esc(best)} "
         f"({100 * r10[best]['moved']:.0f}%). The trained Gram polynomials move {100 * gram['moved']:.0f}% and DCT-II "
         f"{100 * dct['moved']:.0f}%. Subtle edits under smooth jitter are identified {100 * gram[key]:.0f}% (Gram) and "
         f"{100 * dct[key]:.0f}% (DCT) of the time.</p>",
         "<h2>Bases</h2>",
         table(["basis", "definition"], [[esc(a), esc(d)] for a, d in DESC.items()] +
               [["DCT-II, Haar, sym4→Haar", "references; sym4→Haar is the wavelet codec the Haar training runs use"]]),
         "<p>Why the raw polynomial bases differ from each other: they span the same space, so a trajectory has the same "
         "reconstruction in each, but the coefficients are different linear mixtures and the quantizer rounds each coefficient "
         "on its own. Equal-accuracy steps keep the rounding error per coefficient identical, so the comparison is fair. "
         f"Bases with condition number above {MAX_COND:.0e} are not usable for tokenization at that size and are skipped.</p>"]
    cols = ["basis", "group", "condition number", "tokens moved", "nonzero tokens", "overlap between edits", "peak jump (bins)",
            "top-3 share", "top-3% shift (bins)", "top-3% relative", "top-3% set change", "subtle id, Hamming, smooth jitter",
            "subtle id, Hamming, white jitter", "subtle id, |bin| sum, smooth jitter"]
    for n in SIZES:
        res, sk = out[str(n)]["results"], out[str(n)]["skipped"]
        h.append(f"<h2>{n} samples (top 3% = {out[str(n)]['k']} coefficient)</h2>")
        if sk:
            h.append(f"<p class='sub'>Skipped, too ill-conditioned: {esc(', '.join(sk))}.</p>")
        rows = []
        for a in sorted(res, key=lambda a: res[a]["moved"]):
            r = res[a]
            rows.append([f"<b>{esc(a)}</b>" if r["group"] == "reference" else esc(a), esc(r["group"]), f"{r['cond']:.0f}",
                         f"{100 * r['moved']:.0f}%", f"{100 * r['nonzero']:.0f}%", f"{100 * r['overlap']:.0f}%", f"{r['peak']:.0f}",
                         f"{100 * r['top3']:.0f}%", f"{r['top_shift']:.1f}", f"{100 * r['top_rel']:.0f}%",
                         f"{100 * r['set_change']:.0f}%", f"{100 * r[key]:.0f}%", f"{100 * r['id_hamming|white jitter']:.0f}%",
                         f"{100 * r['id_l1|smooth jitter']:.0f}%"])
        h.append(table(cols, rows))
    eds = list(EDITS)
    h.append("<h2>Tokens moved per edit, 10 samples</h2>")
    h.append(table(["basis", *eds], [[esc(a), *[f"{100 * v:.0f}%" for v in r10[a]["moved_by_edit"]]]
                                     for a in sorted(r10, key=lambda a: r10[a]["moved"])]))
    h.append("<h2>Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla/scripts\n"
             "~/miniconda3/envs/smooth-operator-vla/bin/python transform_polynomial_variants.py</pre></body></html>")
    path = REPORTS / "transform_polynomial_variants.html"
    path.write_text("\n".join(h))
    print("wrote", path)


def main():
    out = run()
    RES.mkdir(exist_ok=True)
    (RES / "transform_polynomial_variants.json").write_text(json.dumps(
        {"sizes": SIZES, "step": MAIN_STEP, "subtle": SUBTLE, "max_cond": MAX_COND, "edits": list(EDITS), "by_n": out}, indent=1))
    write_report(out)


if __name__ == "__main__":
    main()

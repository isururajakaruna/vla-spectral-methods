"""Self-contained HTML training report (experiment/reports/training_report.html) built only from measured files:
runs/*/{config,status,metrics}, results/eval*, results/codec_choice_H10.json and weights/*.

No dataset content (frames, actions, episode ids, rollout videos) is embedded.
Run with the VLA env:  python scripts/write_training_report.py
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
WEIGHTS = ROOT / "weights"
REPORTS = ROOT.parent / "reports"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_reports import TRAIN_TASK, load_run, mcnemar_exact  # noqa: E402
from lerobot_policy_adaptive_dwt.metrics import wilson_ci  # noqa: E402

VARIANTS = ("fast_matched", "dwt_fixed", "dwt_adaptive", "dct_fixed")
LABEL = {"fast_matched": "FAST (DCT + BPE)", "dwt_fixed": "DWT fixed plan", "dwt_adaptive": "DWT adaptive", "dct_fixed": "DCT, no BPE (control)"}
FOLLOW_UPS = (("dwt_fixed_bpe", "Haar DWT fixed + BPE"), ("dwt_fixed_s4h", "sym4→haar DWT fixed"),
              ("dwt_fixed_lw", "Haar DWT fixed, scale-weighted loss"), ("dwt_fixed_slw", "Haar DWT fixed, soft labels + scale-weighted loss"))
COLOR = {"fast_matched": "#d9534f", "dwt_fixed": "#2a7ab9", "dwt_adaptive": "#3c9d5d", "dct_fixed": "#e08e0b"}
SEEDS = (0, 1, 2)
DASH = {0: "-", 1: "--", 2: ":"}
COEF_ACC = {"fast_matched": "val_acc_fast", "dwt_fixed": "val_acc_coef", "dwt_adaptive": "val_acc_coef", "dct_fixed": "val_acc_coef"}


def esc(x) -> str:
    return html.escape(str(x))


def num(x, nd=2) -> str:
    if x is None:
        return "–"
    x = float(x)
    if x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def hms(sec: float) -> str:
    m = int(round(sec / 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def fig_html(fig, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def png_html(path: Path, alt: str) -> str:
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def table(header, rows, cls="") -> str:
    out = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return "\n".join(out) + "</tbody></table>"


def ema(y, a=0.9):
    out, s = [], None
    for v in y:
        s = v if s is None else a * s + (1 - a) * v
        out.append(s)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- data


def load_eval():
    rows = [r for r in json.loads((RES / "eval.json").read_text())["rows"] if r["task"] == TRAIN_TASK]

    def episodes_of(r):
        suite, task = r["task"].split("#")
        p = RES / "eval" / f"{r['model'].replace(':', '_').replace('/', '_')}__{suite}_t{task}.json"
        return json.loads(p.read_text())["episodes"] if p.exists() else []

    return rows, episodes_of


FULL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FULL_REF = "full40_ref_pi0fast_libero"
FULL_EPOCH = {14500: 1, 29000: 2, 43500: 3, 58000: 4}
FULL_LOSS = {"dwt_fixed": "plain cross-entropy", "dwt_fixed_lw": "scale-weighted", "dwt_fixed_slw": "soft labels + scale-weighted",
             "dct_fixed": "plain cross-entropy", "poly_fixed": "plain cross-entropy"}
BW5 = "5-level stepped weights (0.3–1.5)"


def load_full():
    """Full-LIBERO evaluations: [(label, name, tokens, epoch, loss, {(suite, task, init): success})], display order."""
    eps = {}
    for p in (RES / "eval").glob("full40_*__*.json"):
        d = json.loads(p.read_text())
        m = eps.setdefault(d["model"], {})
        for e in d["episodes"]:
            m[(d["suite"], d["task_id"], e["init_state"])] = bool(e["success"])
    out = []
    for label, m in eps.items():
        if label == FULL_REF:
            out.append((label, "π₀-FAST LIBERO (LeRobot, full fine-tune)", "ref", None, "LeRobot's recipe", m))
            continue
        g = re.match(r"^full40_(\w+?)@(full_s0|full_e4|full_e3)_\w+:step_(\d+)$", label)
        if not g or int(g.group(3)) not in FULL_EPOCH or g.group(1) not in FULL_LOSS:
            continue
        v, run, ep = g.group(1), g.group(2), FULL_EPOCH[int(g.group(3))]
        sqrtw = label.endswith(f"_sqrtw:step_{g.group(3)}")
        bw5 = label.endswith(f"_bw5:step_{g.group(3)}")
        tok = ("poly" if v == "poly_fixed" else ("dctw" if sqrtw else "dctb" if bw5 else "dct") if v == "dct_fixed"
               else "haarp" if (run, v) == ("full_s0", "dwt_fixed") else "haar")
        name = {"dct": f"DCT, no BPE · epoch {ep}", "dctw": f"DCT, no BPE · epoch {ep}" + (" (from epoch 2)" if run == "full_e3" else " (from scratch)"),
                "dctb": f"DCT, no BPE · epoch {ep} (from epoch 3)",
                "poly": f"Orthogonal polynomials, no BPE · epoch {ep} (from scratch)",
                "haarp": f"Haar DWT fixed, plain from scratch · epoch {ep}",
                "haar": f"Haar DWT fixed · epoch {ep}" + (" (from epoch 3)" if run == "full_e4" else "")}[tok]
        out.append((label, name, tok, ep, "√step-weighted" if sqrtw else BW5 if bw5 else FULL_LOSS[v], m))
    order = {"haar": 0, "haarp": 1, "dct": 2, "dctw": 3, "dctb": 4, "poly": 5, "ref": 6}
    return sorted(out, key=lambda r: (order[r[2]], r[3] or 0, r[4]))


def lc_results(rows, episodes_of):
    pat = re.compile(r"^(\w+)@lc_s(\d+)_\w+:step_(\d+)$")
    res, rowmap = {}, {}
    for r in rows:
        m = pat.match(r["model"])
        if m:
            v, seed, step = m.group(1), int(m.group(2)), int(m.group(3))
            res.setdefault((v, step), {})[seed] = {e["init_state"]: bool(e["success"]) for e in episodes_of(r)}
            rowmap[(v, seed, step)] = r
    return res, rowmap


# ----------------------------------------------------------------------------------------------- plots


def plot_train_loss(lc):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=False)
    axes = axes.ravel()
    for ax, v in zip(axes, VARIANTS):
        for s in SEEDS:
            r = lc.get((v, s))
            if not r:
                continue
            st = [t["step"] for t in r["train"]]
            ax.plot(st, ema([t["loss"] for t in r["train"]]), DASH[s], color=COLOR[v], label=f"seed {s}")
        ax.set_title(f"{LABEL[v]}: training loss (EMA 0.9)")
        ax.set_xlabel("optimizer step")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig_html(fig, "training loss")


def plot_val(lc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for v in VARIANTS:
        for s in SEEDS:
            r = lc.get((v, s))
            if not r:
                continue
            st = [x["step"] for x in r["val"]]
            kw = dict(color=COLOR[v], ls=DASH[s], marker="o", ms=3, label=f"{LABEL[v]} s{s}")
            axes[0].plot(st, [x["val_loss"] for x in r["val"]], **kw)
            axes[1].plot(st, [x.get(COEF_ACC[v], np.nan) for x in r["val"]], **kw)
    axes[0].set_title("held-out token loss (64 fixed windows, 4 val episodes)")
    axes[1].set_title("held-out next-token accuracy on value tokens\n(FAST BPE tokens vs DWT coefficient tokens)")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=3)
    return fig_html(fig, "validation curves")


def plot_opt(lc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for v in VARIANTS:
        r = lc.get((v, 0))
        if not r:
            continue
        st = [t["step"] for t in r["train"]]
        axes[0].plot(st, ema([t["grad_norm"] for t in r["train"]]), color=COLOR[v], label=LABEL[v])
        axes[1].plot(st, [t["lr"] for t in r["train"]], color=COLOR[v], label=LABEL[v])
    axes[0].set_title("gradient norm before clipping (seed 0, EMA 0.9)")
    axes[0].set_yscale("log")
    axes[1].set_title("learning rate (warmup 100, cosine to 5000, stopped at 2000)")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig_html(fig, "optimizer")


def plot_success(res):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for v in VARIANTS:
        steps = sorted(s for (vv, s) in res if vv == v)
        if not steps:
            continue
        k = [sum(sum(d.values()) for d in res[(v, s)].values()) for s in steps]
        n = [sum(len(d) for d in res[(v, s)].values()) for s in steps]
        p = np.array(k) / np.array(n)
        ci = np.array([wilson_ci(a, b) for a, b in zip(k, n)])
        ax.plot(steps, p, "o-", color=COLOR[v], lw=2.2, label=f"{LABEL[v]} (pooled, 3 seeds)")
        ax.fill_between(steps, ci[:, 0], ci[:, 1], color=COLOR[v], alpha=0.13)
        for seed in SEEDS:
            ys = [sum(res[(v, s)][seed].values()) / len(res[(v, s)][seed]) if seed in res[(v, s)] else np.nan for s in steps]
            ax.plot(steps, ys, DASH[seed], color=COLOR[v], alpha=0.35, lw=1)
    ax.set_xlabel("optimizer step (effective batch 16)")
    ax.set_ylabel("closed-loop success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("LIBERO libero_object #0, 20 fixed initial states per checkpoint\nthin lines: individual seeds; band: 95% Wilson CI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return fig_html(fig, "success vs step")


def plot_full(full):
    """Success vs epoch on full LIBERO, overall and LIBERO-10; only models with all 400 episodes."""
    done = [r for r in full if len(r[5]) == 400]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    marker = {"plain cross-entropy": "o", "scale-weighted": "s", "soft labels + scale-weighted": "D", "√step-weighted": "^", BW5: "v"}
    for ax, suites, title in ((axes[0], FULL_SUITES, "all four suites (400 episodes)"), (axes[1], ("libero_10",), "LIBERO-10 (100 episodes)")):
        def rate(m):
            ks = [k for k in m if k[0] in suites]
            return sum(m[k] for k in ks) / len(ks)
        for tok, color, lab in (("haar", COLOR["dwt_fixed"], "Haar DWT fixed (epochs 1–3 soft + weighted)"),
                                ("haarp", "#7b4fc9", "Haar DWT fixed, plain from scratch"), ("dct", COLOR["dct_fixed"], "DCT, no BPE"),
                                ("dctw", COLOR["dct_fixed"], None), ("dctb", COLOR["dct_fixed"], None),
                                ("poly", "#3c9d5d", "Orthogonal polynomials, √step-weighted")):
            main = sorted((r[3], rate(r[5])) for r in done if r[2] == tok and (tok != "haar" or r[3] < 4))
            if main and tok not in ("dctw", "dctb"):
                ax.plot(*zip(*main), "-", color=color, lw=2.2, label=lab if ax is axes[0] else None)
            for r in done:
                if r[2] == tok:
                    ax.plot(r[3], rate(r[5]), marker[r[4]], color=color, ms=7, mfc="white" if r[4] != "plain cross-entropy" else color)
            e4 = [r for r in done if r[2] == "haar" and r[3] == 4]
            if tok == "haar" and main and e4:
                for r in e4:
                    ax.plot([main[-1][0], 4], [main[-1][1], rate(r[5])], ":", color=color, lw=1)
        for r in done:
            if r[2] == "ref":
                ax.axhline(rate(r[5]), color="#555", ls="--", lw=1.4, label="π₀-FAST LIBERO (LeRobot)" if ax is axes[0] else None)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("epoch (14,500 optimizer steps each)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("closed-loop success rate")
    for lab, mk in marker.items():
        axes[1].plot([], [], mk, color="#777", mfc="white" if lab != "plain cross-entropy" else "#777", label=f"loss: {lab}")
    axes[1].legend(fontsize=7.5, loc="lower right")
    axes[0].legend(fontsize=7.5, loc="lower right")
    return fig_html(fig, "full LIBERO success vs epoch")


def plot_pilot(pilot):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for v, r in pilot.items():
        st = [t["step"] for t in r["train"]]
        axes[0].plot(st, ema([t["loss"] for t in r["train"]]), color=COLOR[v], label=LABEL[v])
        axes[1].plot([x["step"] for x in r["val"]], [x["val_loss"] for x in r["val"]], "o-", ms=3, color=COLOR[v], label=LABEL[v])
    axes[0].set_title("pilot (seed 0, 5000 steps): training loss (EMA 0.9)")
    axes[0].set_yscale("log")
    axes[1].set_title("pilot: held-out token loss")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig_html(fig, "pilot curves")


# ----------------------------------------------------------------------------------------------- report


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1.2rem;color:#1d2330;line-height:1.5}
h1{font-size:1.9rem;margin-bottom:.2rem} h2{border-bottom:2px solid #e3e7ee;padding-bottom:.3rem;margin-top:2.4rem}
h3{margin-top:1.6rem} .sub{color:#5b6475;margin-top:0}
table{border-collapse:collapse;margin:.8rem 0 1.2rem;font-size:.86rem;width:100%}
th,td{border:1px solid #dde2ea;padding:.35rem .55rem;text-align:left;vertical-align:top}
th{background:#f3f5f9} tr:nth-child(even) td{background:#fafbfd}
code{background:#f1f3f7;padding:.05rem .3rem;border-radius:3px;font-size:.85em}
pre{background:#f1f3f7;padding:.8rem;border-radius:6px;overflow-x:auto;font-size:.82rem}
img{max-width:100%;display:block;margin:.6rem 0 1rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem;margin:1.2rem 0}
.kpi{border:1px solid #dde2ea;border-radius:8px;padding:.7rem .9rem;background:#fafbfd}
.kpi b{display:block;font-size:1.45rem} .kpi span{color:#5b6475;font-size:.85rem}
.note{background:#fff8e6;border-left:4px solid #e0a800;padding:.6rem .9rem;margin:1rem 0}
.tag{display:inline-block;padding:0 .4rem;border-radius:4px;font-size:.75rem;background:#e6f0fa;color:#2a7ab9}
td.num{text-align:right;font-variant-numeric:tabular-nums}
"""


def main() -> None:
    lc = {(v, s): load_run(f"lc_s{s}_{v}") for v in VARIANTS for s in SEEDS}
    lc = {k: r for k, r in lc.items() if r}
    pilot = {v: r for v in VARIANTS if (r := load_run(f"pilot_{v}"))}
    overfit = load_run("overfit_dwt_adaptive")
    rows, episodes_of = load_eval()
    res, rowmap = lc_results(rows, episodes_of)
    codec = json.loads((RES / "codec_choice_H10.json").read_text())
    any_run = next(iter(lc.values()))
    cfg, args = any_run["config"], any_run["config"]["args"]
    counts = {v: next(r for (vv, _), r in lc.items() if vv == v)["config"]["counts"] for v in VARIANTS}

    def pooled(v, step):
        d = res.get((v, step), {})
        k, n = sum(sum(x.values()) for x in d.values()), sum(len(x) for x in d.values())
        return k, n

    final = {v: pooled(v, 2000) for v in VARIANTS}
    h = []
    add = h.append
    add(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Adaptive DWT vs FAST: LoRA training report</title><style>{CSS}</style></head><body>")
    add("<h1>Adaptive DWT action tokens vs FAST on π₀-FAST: LoRA training report</h1>")
    add(f"<p class='sub'>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/write_training_report.py</code> from measured run logs and evaluation files. "
        "Every number below is measured on the listed hardware unless explicitly marked otherwise. No dataset content is included.</p>")

    # ---- headline
    add("<div class='kpis'>")
    for v in VARIANTS:
        k, n = final[v]
        lo, hi = wilson_ci(k, n)
        add(f"<div class='kpi'><span>{esc(LABEL[v])} @ 2000 steps</span><b style='color:{COLOR[v]}'>{k}/{n} = {k / n:.2f}</b><span>95% CI [{lo:.2f}, {hi:.2f}], 3 seeds × 20 episodes</span></div>")
    gh = np.mean([r["status"]["gpu_hours"] for r in lc.values()])
    add(f"<div class='kpi'><span>training time per 2000-step run</span><b>{gh:.2f} GPU-h</b><span>{esc(cfg['gpu'])}, ≈{np.mean([r['status']['sec_per_step'] for r in lc.values()]):.2f} s/step</span></div>")
    add("</div>")
    add("<p>Three ways of turning a 10-step robot action chunk into tokens were fine-tuned with an identical LoRA recipe on one LIBERO task "
        "and compared by closed-loop success in simulation. With the same steps and the same GPU time, the fixed-plan DWT beat the FAST baseline "
        "at every checkpoint from 1000 steps on, and adaptive DWT overtook FAST from 1500 steps on. "
        "The FAST baseline peaked at 1000 steps and then declined, while almost all of its generated streams were malformed (they did not decode to exactly 70 DCT coefficients).</p>")
    if all((v, s) in lc for v in ("dct_fixed",) for s in SEEDS):
        kd, kf, kh = final["dct_fixed"][0], final["fast_matched"][0], final["dwt_fixed"][0]
        add(f"<p>A fourth variant, added afterwards as a control, keeps FAST's transform (DCT) but drops BPE and uses the DWT runs' quantizer, vocabulary and "
            f"grammar-constrained decoding. It reached {kd}/60 at 2000 steps, between FAST ({kf}/60) and fixed-plan Haar DWT ({kh}/60), with no malformed streams. "
            "Most of FAST's gap is therefore the token format (BPE and free decoding); the transform accounts for a smaller but still measurable part "
            "(paired tests in section 4: Haar beats DCT at 1000 and 2000 steps, p &lt; 0.05 uncorrected).</p>")
    fb, fs = rowmap.get(("dwt_fixed_bpe", 0, 2000)), rowmap.get(("dwt_fixed_s4h", 0, 2000))
    if fb and fs:
        add(f"<p>Single-seed follow-ups (section 6): adding BPE to Haar DWT cut success to {fb['successes']}/20, and the mixed sym4→haar wavelet "
            f"reached {fs['successes']}/20 with {fs['tokens_mean']:.0f} tokens per chunk, so neither beats plain Haar at 10-step chunks. "
            "A zero-shot probe on nine new objects (section 7) finds that none of the single-task policies generalizes (0–13% success).</p>")
    ss = [s for s in SEEDS if ("dwt_fixed_slw", s, 2000) in rowmap]
    if ss:
        k_s = sum(rowmap[("dwt_fixed_slw", s, 2000)]["successes"] for s in ss)
        k_p = sum(rowmap[("dwt_fixed", s, 2000)]["successes"] for s in ss)
        add(f"<p>Changing only the loss on Haar fixed (section 6b): soft labels plus weighting by coefficient scale reached {k_s}/{20 * len(ss)} "
            f"at 2000 steps over {len(ss)} seed(s) against {k_p}/{20 * len(ss)} with plain cross-entropy, same token count: no significant "
            "difference, and slower learning in mid-training.</p>")
    full_h = [r for r in load_full() if len(r[5]) == 400]
    hbest = max((r for r in full_h if r[2] in ("haar", "haarp")), key=lambda r: sum(r[5].values()), default=None)
    href = next((r for r in full_h if r[2] == "ref"), None)
    best = max((r for r in full_h if r[2] != "ref"), key=lambda r: sum(r[5].values()), default=None)
    if hbest and href and best:
        add(f"<p>On all 40 LIBERO tasks (section 7b, same 400 episodes for every model) the best model so far is {esc(best[1])} ({esc(best[4])}) "
            f"at {sum(best[5].values()) / 4:.1f}%, against {sum(href[5].values()) / 4:.1f}% for LeRobot's fully fine-tuned π₀-FAST LIBERO. "
            f"The best Haar DWT fixed model reached {sum(hbest[5].values()) / 4:.1f}% ({esc(hbest[1])}, {esc(hbest[4])}). With the same plain "
            "cross-entropy DCT without BPE beats Haar at every epoch, the reverse of the single-task result, and soft labels + full step "
            "weighting slowed Haar down. Most of the remaining gap is on the long LIBERO-10 tasks.</p>")

    # ---- setup
    add("<h2>1. What was trained</h2>")
    add(table(["item", "value"], [
        ["base model", f"<code>{esc(cfg['base_checkpoint'])}</code> (PaliGemma 3B backbone, SigLIP vision encoder), frozen"],
        ["fine-tuning method", f"LoRA rank {args['lora_rank']}, α {args['lora_alpha']:g} on q/k/v/o and gate/up/down projections of every Gemma LM layer "
         f"({counts['dwt_fixed']['lora_modules']} wrapped linears) + residual <i>action-row adapters</i> on the input embedding and LM head for the action-token rows only"],
        ["trainable / total parameters", "<br>".join(f"{esc(LABEL[v])}: {counts[v]['trainable_params'] / 1e6:.2f} M / {counts[v]['total_params'] / 1e9:.2f} B "
                                                     f"({100 * counts[v]['trainable_params'] / counts[v]['total_params']:.2f}%), {counts[v]['action_rows']} action rows" for v in VARIANTS)],
        ["task", f"LIBERO <code>libero_object</code> #0: “{esc(list(cfg['instructions'].values())[0])}” (single task, single instruction)"],
        ["training data", f"{len(cfg['train_episodes'])} demonstration episodes → {cfg['n_train_windows']} training windows (stride 1); "
         f"validation: {len(cfg['val_episodes'])} held-out episodes, {cfg['n_val_windows']} fixed windows (same for every run and seed)"],
        ["action chunk", f"H = {args['horizon']} steps × 7 dims (6 end-effector deltas + gripper), max {args['max_action_tokens']} action tokens (padded to the same length for every variant)"],
        ["optimizer", f"AdamW (β = 0.9, 0.95, no weight decay), grad-norm clip 1.0, lr {args['lr']:g}, linear warmup {args['warmup']} steps, cosine schedule over {args['steps']} steps, "
         f"micro-batch {args['micro_batch']} × grad-accumulation {args['grad_accum']} = effective batch {cfg['effective_batch']}"],
        ["length", "learning-curve runs paused at 2000 steps (<code>--stop-at 2000</code>, schedule unchanged); pilot runs trained to 5000"],
        ["seeds", "0, 1, 2: changes adapter initialization and batch order only; base weights, data split, validation windows and evaluation initial states are identical"],
        ["software", ", ".join(f"{k} {esc(v)}" for k, v in cfg["versions"].items())],
        ["hardware", f"1 × {esc(cfg['gpu'])} (48 GB), bf16"],
    ]))

    add("<h3>The four action tokenizers</h3>")
    ad, fx = codec["adaptive"], codec["fixed"]
    add(table(["variant", "encoding", "tokens / chunk (train windows)", "vocabulary", "decoding at inference"], [
        [esc(LABEL["fast_matched"]), "DCT per dimension → quantize → BPE (stock π₀-FAST tokenizer)", "≈31 (measured in rollouts)", "FAST BPE vocab (pretrained on other robots)",
         "free generation; relaxed decode pads/truncates to 70 coefficients"],
        [esc(LABEL["dwt_fixed"]), f"Haar (db1) wavelet, periodization; one fixed coefficient plan <code>{esc(fx['plan'])}</code>", f"{num(fx['train']['tokens_mean'], 1)}",
         f"{fx['train']['vocab_size']} reserved tokens", "grammar-constrained (FSM-masked) decoding → inverse DWT"],
        [esc(LABEL["dwt_adaptive"]), f"Haar (db1), periodization; plan chosen per chunk by a rate–distortion DP oracle (γ = {ad['weights']['gamma']})",
         f"{num(ad['train']['tokens_mean'], 1)} (p95 {num(ad['train']['tokens_p95'], 0)})", f"{ad['train']['vocab_size']} reserved tokens", "grammar-constrained decoding → inverse DWT"],
        [esc(LABEL["dct_fixed"]), "orthonormal DCT-II over the whole chunk per dimension, octave bands as levels; same quantizer and stream grammar as DWT fixed, no BPE",
         "≈64 (measured in rollouts)", "same reserved-token scheme as DWT", "grammar-constrained decoding → inverse DCT"],
    ]))
    add("<p>Single-seed follow-up tokenizers (section 6), same recipe:</p>")
    add(table(["variant", "encoding", "tokens / chunk", "vocabulary", "decoding at inference"], [
        ["Haar DWT fixed + BPE", "fixed-plan Haar coefficients, BPE merges of coefficient runs (structure and gripper tokens never merged)",
         "≈54 (measured in rollouts)", "1024 reserved tokens", "grammar-constrained decoding → inverse DWT"],
        ["sym4→haar DWT fixed", "whole-chunk DWT, sym4 at the finest level and Haar at the coarser levels, symmetric boundary mode, full-depth fixed plan, no BPE",
         "≈100 (measured in rollouts)", "same reserved-token scheme as DWT", "grammar-constrained decoding → inverse DWT"],
    ]))
    if (RES / "pareto_H10.png").exists():
        add("<p>Offline codec comparison (reconstruction error vs tokens per chunk, before any training):</p>" + png_html(RES / "pareto_H10.png", "codec pareto"))

    # ---- compute
    add("<h2>2. Training time and compute</h2>")
    body = []
    for v in VARIANTS:
        for s in SEEDS:
            r = lc.get((v, s))
            if not r:
                continue
            st, t = r["status"], r["train"][-1]
            body.append([f"<code>lc_s{s}_{v}</code>", esc(LABEL[v]), s, st["step"], f"{st['gpu_hours']:.2f}", hms(st["wall_s"]),
                         f"{st['sec_per_step']:.2f}", f"{t['vram_reserved_gb']:.1f}", f"{t['chunks_seen']:,}", f"{t['action_tokens_seen']:,}"])
    for v, lab in FOLLOW_UPS:
        if (ROOT / "runs" / f"lc_s0_{v}" / "config.json").exists():
            r = load_run(f"lc_s0_{v}")
            st, t = r["status"], r["train"][-1]
            body.append([f"<code>lc_s0_{v}</code>", esc(lab), 0, st["step"], f"{st['gpu_hours']:.2f}", hms(st["wall_s"]),
                         f"{st['sec_per_step']:.2f}", f"{t['vram_reserved_gb']:.1f}", f"{t['chunks_seen']:,}", f"{t['action_tokens_seen']:,}"])
    for v, r in pilot.items():
        st, t = r["status"], r["train"][-1]
        body.append([f"<code>pilot_{v}</code>", esc(LABEL[v]), 0, st["step"], f"{st['gpu_hours']:.2f}", "2000 + resumed to 5000",
                     f"{st['sec_per_step']:.2f}", f"{t['vram_reserved_gb']:.1f}", f"{t['chunks_seen']:,}", f"{t['action_tokens_seen']:,}"])
    if overfit:
        st, t = overfit["status"], overfit["train"][-1]
        body.append(["<code>overfit_dwt_adaptive</code>", "DWT adaptive, 64 windows (sanity check)", 0, st["step"], f"{st['gpu_hours']:.2f}",
                     hms(st["wall_s"]), f"{st['sec_per_step']:.2f}", f"{t['vram_reserved_gb']:.1f}", f"{t['chunks_seen']:,}", f"{t['action_tokens_seen']:,}"])
    add(table(["run", "variant", "seed", "steps", "GPU-h (training loop)", "wall-clock", "s / step", "VRAM reserved GB", "chunks seen", "action tokens seen"], body))
    add("<p>Wall-clock includes model loading, validation every 250 steps and checkpoint writes. All variants cost the same per step because the "
        "action sequence is padded to the same length, so the step axis below is also an equal-compute axis. DWT streams carry ≈2× more real action tokens per chunk than FAST.</p>")

    # ---- curves
    add("<h2>3. Training curves (learning-curve runs, seeds 0–2)</h2>")
    add(plot_train_loss(lc))
    epoch = cfg["n_train_windows"] / cfg["effective_batch"]
    add(f"<p>The step-like drops in training loss sit at multiples of {epoch:.0f} steps, one pass over the {cfg['n_train_windows']} training windows "
        f"at batch {cfg['effective_batch']} ({', '.join(str(round(epoch * k)) for k in range(1, int(2000 // epoch) + 1))}): each new epoch revisits windows already seen, "
        "a sign of memorization. Seeds are almost indistinguishable in training loss.</p>")
    add(plot_val(lc))
    add("<div class='note'>Held-out token loss rises after roughly 1000 steps for every variant: with 40 demonstrations the adapters start to memorize them. "
        "It does not predict closed-loop success (next section), and losses are not comparable across variants because the vocabularies and stream lengths differ.</div>")
    add(plot_opt(lc))
    last_val = []
    for v in VARIANTS:
        for s in SEEDS:
            r = lc.get((v, s))
            if not r:
                continue
            x = r["val"][-1]
            per = ", ".join(f"{k[8:]} {x[k]:.3f}" for k in x if k.startswith("val_acc_") and k != "val_acc_all")
            last_val.append([esc(LABEL[v]), s, x["step"], f"{x['val_loss']:.3f}", f"{x['val_acc_all']:.3f}", per])
    add("<h3>Held-out token metrics at the final step</h3>" + table(["variant", "seed", "step", "val loss", "val token acc", "accuracy per token class"], last_val))

    # ---- closed loop
    add("<h2>4. Closed-loop success in LIBERO</h2>")
    add("<p>Each checkpoint runs 20 episodes from LIBERO initial states 0–19 (identical for every model), 256×256 renders, 10 actions executed per chunk, at most 280 steps.</p>")
    add(plot_success(res))
    body = []
    for v in VARIANTS:
        for step in sorted(s for (vv, s) in res if vv == v):
            d = res[(v, step)]
            k, n = pooled(v, step)
            lo, hi = wilson_ci(k, n)
            mal = [rowmap[(v, s, step)] for s in SEEDS if (v, s, step) in rowmap]
            mc, ch = sum(r["invalid_streams"] or 0 for r in mal), sum(r["chunks"] for r in mal)
            body.append([esc(LABEL[v]), step, *[f"{sum(d[s].values())}/{len(d[s])}" if s in d else "–" for s in SEEDS],
                         f"<b>{k}/{n} = {k / n:.2f}</b>", f"[{lo:.2f}, {hi:.2f}]", f"{100 * mc / max(ch, 1):.0f}%"])
    add(table(["variant", "step", "seed 0", "seed 1", "seed 2", "pooled", "95% Wilson CI", "malformed chunks"], body))
    def paired(v, ref):
        body = []
        for step in sorted(s for (vv, s) in res if vv == v):
            b = c = 0
            for s in SEEDS:
                a, f_ = res.get((v, step), {}).get(s, {}), res.get((ref, step), {}).get(s, {})
                for i in a.keys() & f_.keys():
                    b += a[i] and not f_[i]
                    c += f_[i] and not a[i]
            body.append([esc(LABEL.get(v, dict(FOLLOW_UPS).get(v, v))), step, b, c, f"{mcnemar_exact(b, c):.3f}"])
        return body

    add("<h3>Paired test against FAST (same seed, step and initial state; exact McNemar)</h3>")
    add(table(["variant", "step", "only this variant succeeds", "only FAST succeeds", "p"],
              [r for v in ("dwt_fixed", "dwt_adaptive", "dct_fixed") for r in paired(v, "fast_matched")]))
    add("<h3>Paired test: Haar DWT fixed vs DCT without BPE (transform only)</h3>")
    add(table(["variant", "step", "only Haar succeeds", "only DCT succeeds", "p"], paired("dwt_fixed", "dct_fixed")))

    pil = [r for r in rows if "@pilot_" in r["model"] or r["model"].startswith("ref_")]
    body = [[f"<code>{esc(r['model'])}</code>", f"{r['successes']}/{r['episodes']}", f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]", num(r["tokens_mean"], 1),
             num(r["latency_ms_mean"], 0), num(r["latency_ms_p95"], 0), "n/a" if r["invalid_streams"] is None else f"{r['invalid_streams']}/{r['chunks']}"] for r in pil]
    add("<h3>Pilot runs (seed 0, 2000 and 5000 steps) and reference checkpoints</h3>")
    add(plot_pilot(pilot))
    add(table(["model", "successes", "95% CI", "tokens / chunk", "latency ms (mean)", "latency ms (p95)", "malformed chunks"], body))
    add("<p><code>ref_*</code> are the unmodified LeRobot π₀-FAST and π₀ checkpoints fully fine-tuned on all 40 LIBERO tasks with far more compute; they bound what is reachable and are not a matched comparison.</p>")

    # ---- inference
    add("<h2>5. Inference cost</h2>")
    body = []
    for v in VARIANTS:
        rs = [rowmap[(v, s, 2000)] for s in SEEDS if (v, s, 2000) in rowmap]
        if rs:
            body.append([esc(LABEL[v]), num(np.mean([r["tokens_mean"] for r in rs]), 1), num(np.mean([r["latency_ms_mean"] for r in rs]), 0),
                         num(np.mean([r["latency_ms_p95"] for r in rs]), 0)])
    for v, lab in FOLLOW_UPS:
        r = rowmap.get((v, 0, 2000))
        if r:
            body.append([esc(lab + " (seed 0 only)"), num(r["tokens_mean"], 1), num(r["latency_ms_mean"], 0), num(r["latency_ms_p95"], 0)])
    add(table(["variant (step 2000, mean over seeds)", "tokens / chunk", "latency ms / chunk", "p95 ms"], body))
    add("<p>Latency is autoregressive generation of one chunk with a KV cache on the A6000. DWT streams are about twice as long as FAST streams, so latency roughly doubles; "
        "at 10 executed actions per chunk this is still within LIBERO's simulated control loop but would need speculative or parallel decoding for real-time control at higher rates.</p>")

    # ---- follow-up controls
    add("<h2>6. Follow-up: which part of the token format matters? BPE, transform, wavelet (seed 0)</h2>")
    add("<p>The FAST baseline always used BPE (it is part of FAST), while the DWT runs above did not, so the FAST-vs-DWT gap mixes the transform "
        "(DCT vs wavelet) with BPE. Single-seed runs with the identical recipe separate them: Haar DWT <i>with</i> BPE, and FAST's DCT "
        "<i>without</i> BPE (same quantizer, vocabulary and grammar-constrained decoding as DWT). A third run swaps Haar for the mixed "
        "sym4→haar wavelet that won the offline wavelet test at 32-step chunks (section 9), fixed plan, no BPE. "
        "Same base model, task, seed 0 and initial states as above.</p>")
    fu = [("fast_matched", "DCT (FAST)", "yes (FAST)", "#d9534f"), ("dwt_fixed", "Haar DWT", "no", "#2a7ab9"),
          ("dwt_fixed_bpe", "Haar DWT", "yes", "#8e44ad"), ("dct_fixed", "DCT, whole chunk", "no", "#e08e0b"),
          ("dwt_fixed_s4h", "sym4→haar DWT", "no", "#17a2b8")]
    steps_fu = [250, 500, 1000, 1500, 2000]
    body, fig, ax = [], *plt.subplots(figsize=(8.5, 4.2))
    for v, tr, bpe, col in fu:
        rs = {st: rowmap.get((v, 0, st)) for st in steps_fu}
        run = load_run(f"lc_s0_{v}") if (ROOT / "runs" / f"lc_s0_{v}" / "config.json").exists() else None
        if not run:
            body.append([f"<code>{v}</code>", tr, bpe, "–", *["–"] * len(steps_fu), "not run yet", "–", "–"])
            continue
        b = run["config"].get("bpe")
        ttok = f"{b['train_tokens_mean_after']:.1f}" if b else "–"
        cells = [f"{rs[st]['successes']}/{rs[st]['episodes']}" if rs[st] else "–" for st in steps_fu]
        last = rs[2000]
        state = run["status"].get("state", "")
        body.append([f"<code>{v}</code>", tr, bpe, ttok, *cells,
                     num(last["tokens_mean"], 1) if last else (f"training ({state})" if state != "done" else "evaluating"),
                     num(last["latency_ms_mean"], 0) if last else "–",
                     ("n/a" if last["invalid_streams"] is None else f"{100 * last['invalid_streams'] / max(last['chunks'], 1):.0f}%") if last else "–"])
        xs = [st for st in steps_fu if rs[st]]
        if xs:
            ax.plot(xs, [rs[st]["successes"] / rs[st]["episodes"] for st in xs], "o-", color=col, lw=2, label=f"{tr}, BPE {bpe}")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("success rate (20 episodes)"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    ax.set_title("Seed 0: transform and BPE separated"); ax.legend(fontsize=8, loc="upper left")
    add(fig_html(fig, "follow-up success vs step"))
    add(table(["run (seed 0)", "transform", "BPE", "train tokens / chunk after BPE", *[f"success @{st}" for st in steps_fu],
               "tokens / chunk @2000 (inference)", "latency ms @2000", "malformed chunks @2000"], body))
    rb, rf = rowmap.get(("dwt_fixed_bpe", 0, 2000)), rowmap.get(("dwt_fixed", 0, 2000))
    rd = rowmap.get(("dct_fixed", 0, 2000))
    if rb and rf:
        add(f"<p><b>BPE hurts DWT.</b> With BPE the Haar DWT policy reached {rb['successes']}/20 at 2000 steps against {rf['successes']}/20 without it "
            f"(and {rowmap[('dwt_fixed_bpe', 0, 1000)]['successes']}/20 vs {rowmap[('dwt_fixed', 0, 1000)]['successes']}/20 at 1000 steps), although streams were shorter "
            f"({rb['tokens_mean']:.0f} vs {rf['tokens_mean']:.0f} tokens) and generation faster ({rb['latency_ms_mean']:.0f} vs {rf['latency_ms_mean']:.0f} ms). "
            "Held-out loss per chunk was about 20% higher with BPE: each merged token fixes several coefficient values at once and is drawn from a 4× larger "
            "vocabulary. Early checkpoints also spelled coefficients out instead of using merges (67 tokens per chunk at 250 steps). One seed.</p>")
    if rd and rf:
        add(f"<p><b>DCT without BPE</b> reached {rd['successes']}/20 at 2000 steps, against {rf['successes']}/20 for Haar without BPE and "
            f"{rowmap.get(('fast_matched', 0, 2000), {}).get('successes', '–')}/20 for FAST (DCT with BPE), same seed. "
            "It was then repeated with seeds 1 and 2 and appears as a fourth variant in sections 3–5.</p>")
    else:
        add("<p><b>DCT without BPE</b> is training now; this section fills in when its evaluation finishes.</p>")
    rs4 = rowmap.get(("dwt_fixed_s4h", 0, 2000))
    if rs4 and rf:
        add(f"<p><b>sym4→haar does not beat Haar at 10-step chunks.</b> It reached {rs4['successes']}/20 at 2000 steps against {rf['successes']}/20 for Haar, "
            f"but learned more slowly and less steadily ({', '.join(str(rowmap[('dwt_fixed_s4h', 0, st)]['successes']) for st in steps_fu)} of 20 at "
            f"{', '.join(map(str, steps_fu))} steps). At this length sym4 needs symmetric padding, so its streams are {rs4['tokens_mean']:.0f} tokens "
            f"instead of {rf['tokens_mean']:.0f} and generation is slower ({rs4['latency_ms_mean']:.0f} vs {rf['latency_ms_mean']:.0f} ms), "
            "as the offline check predicted. Its offline advantage is at 32-step chunks, which no policy has been trained on yet. One seed.</p>")
    add("<div class='note'>An offline sweep (<code>scripts/quant_sweep.py</code>) found that coarser rounding of the detail coefficients plus a compact stream "
        "would bring DWT + BPE to about 29 tokens per chunk at FAST's error (FAST: 25). That run was dropped in favour of the DCT control above; "
        "see the <a href='experiment_trail.html'>experiment trail</a>.</div>")

    lv = [("dwt_fixed", "plain cross-entropy", "#2a7ab9"), ("dwt_fixed_lw", "scale-weighted", "#2ca02c"),
          ("dwt_fixed_slw", "soft labels + scale-weighted", "#c0392b")]
    if all(rowmap.get((v, 0, 2000)) for v, _, _ in lv):
        add("<h3>6b. Follow-up: the loss on Haar DWT fixed</h3>")
        add("<p>Every run above trains with one-hot cross-entropy, which treats all action tokens alike and every wrong bin alike. Haar is "
            "orthonormal, so a coefficient error adds its square to the trajectory error (Parseval), and a one-bin miss costs one quantizer step. "
            "Those steps differ about 30× across levels (0.041–0.119 for the coarsest coefficient, 0.0037–0.0142 for the finest details, which are "
            "half of the 60 coefficient tokens). Two losses use this, with the same tokens, recipe, seed and decoding as Haar fixed: "
            "<b>scale-weighted</b> multiplies each coefficient token's cross-entropy by its step, normalized to mean 1 (≈1.8–5.2 for the "
            "coarsest coefficients, 0.16–0.62 for the finest); <b>soft labels + scale-weighted</b> also replaces the one-hot target by "
            "q(b) ∝ exp(−((b − b*)·step)² / 2σ²) with σ = 0.007, so a neighbour bin gets real credit at the finest level and almost none at "
            "the coarsest. Structure and gripper tokens are unchanged, and the logged losses stay plain cross-entropy. "
            "Soft labels + weighting was run with seeds 0–2, weighting alone with seed 0 only.</p>")
        seeds_of = lambda v: [s for s in SEEDS if (v, s, 2000) in rowmap]  # noqa: E731
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        body = []
        for v, lab, col in lv:
            ss = seeds_of(v)
            ks = {st: (sum(rowmap[(v, s, st)]["successes"] for s in ss if (v, s, st) in rowmap),
                       sum(rowmap[(v, s, st)]["episodes"] for s in ss if (v, s, st) in rowmap)) for st in steps_fu}
            ax.plot(steps_fu, [k / n for k, n in ks.values()], "o-", color=col, lw=2, label=f"{lab} ({len(ss)} seed{'s' if len(ss) > 1 else ''})")
            jerk = np.mean([np.mean([e["action_jerk_mean"] for e in episodes_of(rowmap[(v, s, 2000)])]) for s in ss])
            body.append([f"<code>{v}</code>", esc(lab), ", ".join(map(str, ss)), *[f"{k}/{n}" for k, n in ks.values()],
                         num(np.mean([rowmap[(v, s, 2000)]["tokens_mean"] for s in ss]), 1),
                         num(np.mean([rowmap[(v, s, 2000)]["latency_ms_mean"] for s in ss]), 0), f"{jerk:.4f}",
                         f"{sum(rowmap[(v, s, 2000)]['invalid_streams'] for s in ss)}/{sum(rowmap[(v, s, 2000)]['chunks'] for s in ss)}"])
        ax.set_xlabel("optimizer step"); ax.set_ylabel("success rate (pooled over seeds)"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.set_title("Haar DWT fixed: loss variants"); ax.legend(fontsize=8, loc="lower right")
        add(fig_html(fig, "loss variants success vs step"))
        add(table(["run", "loss", "seeds", *[f"success @{st}" for st in steps_fu], "tokens / chunk @2000", "latency ms @2000",
                   "action jerk @2000", "malformed chunks @2000"], body))
        per_seed = [[f"<code>{v}</code>", s, *[f"{rowmap[(v, s, st)]['successes']}/20" if (v, s, st) in rowmap else "–" for st in steps_fu]]
                    for v in ("dwt_fixed", "dwt_fixed_slw") for s in seeds_of(v)]
        add("<p>Per seed:</p>" + table(["run", "seed", *[f"@{st}" for st in steps_fu]], per_seed))
        add("<p>Paired against plain cross-entropy (same seed, step and initial state, pooled over the seeds each variant has; exact McNemar):</p>")
        pr = {v: paired(v, "dwt_fixed") for v in ("dwt_fixed_lw", "dwt_fixed_slw")}
        add(table(["variant", "step", "only this variant succeeds", "only plain succeeds", "p"],
                  [r for v in pr for r in pr[v] if r[1] != 250]))
        ss = seeds_of("dwt_fixed_slw")
        k_s = sum(rowmap[("dwt_fixed_slw", s, 2000)]["successes"] for s in ss)
        k_p = sum(rowmap[("dwt_fixed", s, 2000)]["successes"] for s in ss)
        p2k = next(r for r in pr["dwt_fixed_slw"] if r[1] == 2000)
        add(f"<p><b>Neither loss beats plain cross-entropy.</b> Soft labels + weighting reached {k_s}/{20 * len(ss)} at 2000 steps over "
            f"{len(ss)} seeds against {k_p}/{20 * len(ss)} for plain (paired: {p2k[2]} episodes won only by soft labels, {p2k[3]} only by plain, "
            f"p = {p2k[4]}). Its seed-0 score of 20/20 was the best single run here but did not repeat "
            f"({', '.join(str(rowmap[('dwt_fixed_slw', s, 2000)]['successes']) for s in ss)} of 20 for seeds {', '.join(map(str, ss))}). "
            "It also learns more slowly in mid-training (behind plain at 1000 and 1500 steps, p ≈ 0.09 and 0.12), as expected from targets "
            "that no longer reward the exact bin; its held-out exact-bin accuracy stays about 2 points below plain. Weighting alone (one seed) "
            "is within noise of plain at every step. Token count and latency are unchanged and motion is about as smooth. "
            "Plain cross-entropy stays the default; soft labels + weighting was nevertheless chosen for the full-LIBERO run, where its "
            "slower start matters less over 14,500 steps.</p>")

    # ---- zero-shot generalization
    gen_rows = [r for r in json.loads((RES / "eval.json").read_text())["rows"]
                if r["task"] != TRAIN_TASK and re.match(r"^\w+@lc_s0_\w+:step_02000$", r["model"])]
    if gen_rows:
        add("<h2>7. Zero-shot generalization to new objects (seed 0)</h2>")
        add("<p>Every model above was trained on one instruction only. As a first generalization probe, the seed-0 step-2000 checkpoints were run "
            "on the other nine <code>libero_object</code> tasks: the same scene type and basket, a different object to pick up, and the new instruction "
            "in the prompt. Five initial states per task were drawn at random (seed 0) from states 0–19, a different draw per task and the same "
            "states for every model; 45 episodes per model. Nothing was trained on these tasks.</p>")
        gv = [v for v in ("dwt_fixed", "dwt_fixed_lw", "dwt_fixed_slw", "dwt_fixed_s4h", "dct_fixed", "dwt_adaptive")
              if any(r["model"].startswith(v + "@") for r in gen_rows)]
        glabel = {"dwt_fixed": "Haar DWT fixed", "dwt_fixed_lw": "Haar fixed, weighted loss", "dwt_fixed_slw": "Haar fixed, soft + weighted loss",
                  "dwt_fixed_s4h": "sym4→haar DWT fixed", "dct_fixed": "DCT, no BPE", "dwt_adaptive": "Haar DWT adaptive"}
        by = {(r["model"].split("@")[0], r["task"]): r for r in gen_rows}
        tasks = sorted({r["task"] for r in gen_rows}, key=lambda t: int(t.split("#")[1]))
        obj = {r["task"]: re.sub(r"^pick up the (.*) and place it in the basket$", r"\1", r["instruction"]) for r in gen_rows}
        body = []
        for t in tasks:
            cells = []
            for v in gv:
                r = by.get((v, t))
                cells.append(f"{r['successes']}/{r['episodes']}" if r else "–")
            body.append([f"#{t.split('#')[1]} {esc(obj[t])}", esc(", ".join(map(str, by[(gv[0], t)]["init_states"]))), *cells])
        tot = {v: (sum(by[(v, t)]["successes"] for t in tasks if (v, t) in by), sum(by[(v, t)]["episodes"] for t in tasks if (v, t) in by)) for v in gv}
        body.append(["<b>total</b>", "", *[f"<b>{k}/{n}</b>" for k, n in (tot[v] for v in gv)]])
        tr0 = {v: rowmap.get((v, 0, 2000)) for v in gv}
        body.append(["training task (#0, 20 states)", "0–19", *[f"{tr0[v]['successes']}/20" if tr0[v] else "–" for v in gv]])
        add(table(["task (new object)", "initial states", *[glabel[v] for v in gv]], body))

        def succ(v):
            out = {}
            for t in tasks:
                r = by.get((v, t))
                for e in (episodes_of(r) if r else []):
                    out[(t, e["init_state"])] = bool(e["success"])
            return out

        ref = succ("dwt_fixed")
        pbody, sig = [], []
        for v in gv[1:]:
            o = succ(v)
            b = sum(ref[k] and not o[k] for k in ref.keys() & o.keys())
            c = sum(o[k] and not ref[k] for k in ref.keys() & o.keys())
            pbody.append([glabel[v], b, c, f"{mcnemar_exact(b, c):.2f}"])
            pv = mcnemar_exact(b, c)
            if pv < 0.05:
                sig.append(f"{glabel[v]} (p = {pv:.2f})")
        add("<h3>Paired test against Haar DWT fixed (same task and initial state; exact McNemar)</h3>")
        add(table(["variant", "only Haar fixed succeeds", "only this variant succeeds", "p"], pbody))
        hit = sorted({t for (v, t), r in by.items() if r["successes"]}, key=lambda t: int(t.split("#")[1]))
        add(f"<p>All models fail almost everywhere: every success of every model is on {' or '.join(esc(obj[t]) for t in hit)}, and the "
            "successful episodes are about as long as a successful alphabet-soup episode (≈145 steps). The likeliest reading is that the policies "
            "replay the motion learned on the training task and succeed where the named object happens to be reachable that way, rather than "
            "following the new instruction; no videos were recorded, so this is not confirmed. Roughly as on the training task, the Haar "
            "fixed policies lead and adaptive is lowest, but the loss changes do not carry over: soft labels + weighting, best on the training "
            "task, scores the same as plain cross-entropy here, and the three Haar fixed losses are within one episode of each other. "
            + (f"Haar fixed beats {', '.join(sig)} in the paired test (uncorrected, one seed); the other differences are not significant. "
               if sig else "No difference is significant with one seed and 45 episodes. ")
            + "A real generalization test needs multi-task training.</p>")

    # ---- full LIBERO
    full = load_full()
    if full:
        add("<h2>7b. Full LIBERO: all 40 tasks, accuracy per epoch</h2>")
        add("<p>Every result above is on one task. To compare with published π₀-FAST numbers, Haar DWT fixed was trained once on all of "
            "<code>lerobot/libero</code> (40 tasks in four suites, 1,525 training demos, 232,282 windows, codec and normalization refitted) with "
            "the same LoRA recipe (rank 16, 20.85 M trainable parameters, effective batch 16, peak lr 2·10⁻⁴, warmup 300; one epoch = 14,500 steps). "
            "The schedule was extended as the run went: epoch 1 under a 29,000-step cosine, epochs 2–3 under 43,500 steps (resumed, lr stepped up "
            "from 1.1·10⁻⁴ to 1.6·10⁻⁴), and epoch 4 under 58,000 steps, branched three times from the epoch-3 checkpoint with a different loss "
            "each (lr 4.6·10⁻⁵ decaying to 2·10⁻⁵). Epochs 1–3 used soft labels + scale weighting. The DCT control (no BPE, plain cross-entropy) "
            "repeats exactly this learning-rate history from scratch, and so does a plain-cross-entropy Haar run (2 epochs) that separates the "
            "transform from the loss. Later runs test a gentler √step weighting on DCT (one epoch from the plain DCT epoch-2 checkpoint, one "
            "epoch from scratch) and an orthogonal-polynomial tokenizer (the best basis in the synthetic "
            "<a href='transform_shift_analysis.html'>token-shift</a> and <a href='transform_fingerprint_analysis.html'>fingerprint</a> "
            "studies) with the same weighting from scratch. Every model is evaluated on the same 400 episodes: all four suites, every "
            "task, initial states 0–9, suite-default step limits (Spatial/Object 280, Goal 300, LIBERO-10 520). LeRobot's "
            "<code>lerobot/pi0fast-libero</code> (full fine-tune on all of LIBERO) was evaluated by us, unmodified, on the same episodes.</p>")
        add(plot_full(full))
        body = []
        for label, name, tok, ep, loss, m in full:
            row = [esc(name), esc(loss)]
            if len(m) < 400:
                body.append(row + ["–"] * 4 + [f"evaluating ({len(m)}/400 episodes)"])
                continue
            for su in FULL_SUITES:
                ks = [k for k in m if k[0] == su]
                row.append((f"{100 * sum(m[k] for k in ks) / len(ks):.0f}%" + ("" if len(ks) == 100 else f" ({len(ks)} eps)")) if ks else "–")
            k, n = sum(m.values()), len(m)
            row.append(f"<b>{100 * k / n:.1f}%</b> ({k}/{n})")
            body.append(row)
        add(table(["model", "loss", "Spatial", "Object", "Goal", "LIBERO-10", "overall"], body))

        def pair(a, b):
            ks = a.keys() & b.keys()
            x, y = sum(a[k] and not b[k] for k in ks), sum(b[k] and not a[k] for k in ks)
            return f"{x} / {y}, p = {mcnemar_exact(x, y):.2g}"

        complete = [r for r in full if len(r[5]) == 400]
        ref = next((r[5] for r in complete if r[2] == "ref"), None)
        haar = {r[3]: r[5] for r in complete if r[2] == "haar" and (r[3] < 4 or r[4] == "plain cross-entropy")}
        haarp = {r[3]: r[5] for r in complete if r[2] == "haarp"}
        dctp = {r[3]: r[5] for r in complete if r[2] == "dct"}
        pbody = []
        for label, name, tok, ep, loss, m in complete:
            if tok == "ref":
                continue
            c1 = c2 = c3 = "–"
            if tok == "haar" and ep == 4 and 3 in haar:
                c1 = pair(m, haar[3]) + " (vs epoch 3)"
            elif tok in ("dct", "dctw", "poly", "haarp") and ep in haar:
                c1 = pair(m, haar[ep])
            if tok in ("dct", "dctw", "poly") and ep in haarp:
                c2 = pair(m, haarp[ep])
            if tok in ("dctw", "dctb", "poly") and ep in dctp:
                c3 = pair(m, dctp[ep])
            pbody.append([esc(name), esc(loss), pair(m, ref) if ref else "–", c1, c2, c3])
        add("<h3>Paired tests on the same episodes (exact McNemar; episodes won only by the row model / only by the other)</h3>")
        add(table(["model", "loss", "vs π₀-FAST LIBERO", "vs Haar soft + weighted, same epoch (epoch 4: plain branch)",
                   "vs Haar plain from scratch, same epoch", "vs DCT plain, same epoch"], pbody))

        def pct(m):
            return f"{100 * sum(m.values()) / len(m):.1f}%"
        done = [r for r in full if len(r[5]) == 400]
        hs = ", ".join(f"epoch {r[3]} {pct(r[5])}" for r in done if r[2] == "haar" and r[3] < 4)
        e4 = ", ".join(f"{r[4]} {pct(r[5])}" for r in done if r[2] == "haar" and r[3] == 4)
        ds = ", ".join(f"epoch {r[3]} {pct(r[5])}" for r in done if r[2] == "dct")
        ps = ", ".join(f"epoch {r[3]} {pct(r[5])}" for r in done if r[2] == "haarp")
        dw = ", ".join(f"epoch {r[3]} {'from epoch 2' if 'full_e3' in r[0] else 'from scratch'} {pct(r[5])}"
                       for r in done if r[2] == "dctw")
        po = ", ".join(f"epoch {r[3]} {pct(r[5])}" for r in done if r[2] == "poly")

        def vs(a, b):
            x, y = sum(a[k] and not b[k] for k in a), sum(b[k] and not a[k] for k in a)
            return f"{pct(a)} vs {pct(b)}, {x} vs {y} episodes won alone, p = {mcnemar_exact(x, y):.2g}"

        dctw = {("e3" if "full_e3" in r[0] else "s0", r[3]): r[5] for r in done if r[2] == "dctw"}
        polyd = {r[3]: r[5] for r in done if r[2] == "poly"}
        finds = []
        tr = [e for e in sorted(dctp) if e in haarp]
        if tr:
            finds.append("<b>Transform, same plain cross-entropy:</b> " + "; ".join(f"epoch {e}: DCT vs Haar {vs(dctp[e], haarp[e])}" for e in tr)
                         + ". DCT learns faster and stays ahead, the reverse of the single-task result.")
        lo = [e for e in sorted(haarp) if e in haar]
        if lo:
            finds.append("<b>Loss on Haar:</b> " + "; ".join(f"epoch {e}: plain vs soft + weighted {vs(haarp[e], haar[e])}" for e in lo)
                         + ". Soft labels + full step weighting slowed Haar down; plain Haar after two epochs is already level with "
                         "every soft or weighted Haar model after three or four.")
        if ("e3", 3) in dctw and 2 in dctp:
            a = dctw[("e3", 3)]
            l10 = {k: v for k, v in a.items() if k[0] == "libero_10"}
            finds.append(f"<b>√step weighting on DCT, one epoch from the plain epoch-2 checkpoint:</b> {vs(a, dctp[2])} against its starting "
                         f"point, LIBERO-10 {vs(l10, {k: dctp[2][k] for k in l10})}"
                         + (f"; against plain DCT epoch 3 (same start, schedule and epoch, plain loss): {vs(a, dctp[3])}" if 3 in dctp else
                            "; plain DCT epoch 3, the matched comparison (same start and schedule, plain loss), is still training")
                         + (f"; against π₀-FAST LIBERO: {vs(a, ref)}" if ref else "") + ".")
        dctb = {r[3]: r[5] for r in done if r[2] == "dctb"}
        if 4 in dctb:
            a = dctb[4]
            finds.append("<b>5-level stepped weights on DCT, one epoch from the plain epoch-3 checkpoint</b> (the codec's step range cut "
                         "into 5 equal log-scale segments weighing 0.3, 0.6, 0.9, 1.2, 1.5, smallest step first; not normalized, no soft "
                         f"labels): {vs(a, dctp[3])} against its starting point"
                         + (f"; against plain DCT epoch 4 (same start, schedule and epoch, plain loss): {vs(a, dctp[4])}" if 4 in dctp else
                            "; plain DCT epoch 4, the matched comparison, is queued")
                         + (f"; against π₀-FAST LIBERO: {vs(a, ref)}" if ref else "") + ".")
        s0e = [e for e in sorted(e for t, e in dctw if t == "s0") if e in dctp]
        if s0e:
            finds.append("<b>√step weighting on DCT from scratch:</b> " + "; ".join(
                f"epoch {e}: {vs(dctw[('s0', e)], dctp[e])} against plain DCT epoch {e}" for e in s0e) + ".")
        if 1 in polyd:
            finds.append(f"<b>Orthogonal polynomials (√step-weighted, from scratch), epoch 1:</b> {vs(polyd[1], dctp[1])} against plain DCT"
                         + (f", {vs(polyd[1], dctw[('s0', 1)])} against √step-weighted DCT" if ("s0", 1) in dctw else "") + ".")
        dct_note = "".join(f"<p>{f}</p>" for f in finds)
        add(f"<p>Haar DWT fixed: {hs}; epoch 4 from the epoch-3 checkpoint: {e4}. "
            + (f"π₀-FAST LIBERO: {pct(ref)}. " if ref else "")
            + (f"DCT without BPE (control): {ds}. " if ds else "The DCT control is still training or evaluating; its rows appear as epochs finish. ")
            + (f"DCT with √step weighting (coefficient-token loss × step<sup>0.5</sup>, normalized to mean 1; DCT weights 0.56–2.8 "
               f"instead of 0.25–6.4 for full step weighting): {dw}. " if dw else "")
            + (f"Haar with plain cross-entropy from scratch: {ps}. " if ps else "")
            + (f"Orthogonal polynomials (Legendre basis orthonormalized on the 10 samples, same quantizer, grammar and 64 tokens as DCT, "
               f"√step-weighted, from scratch): {po}. " if po else "")
            + "</p>" + dct_note
            + "<p>The soft + weighted Haar run rose steeply over the first epochs, then its fourth epoch changed nothing for any of the three "
            "losses (paired p ≥ 0.3 against each other and against epoch 3). Its validation token loss bottomed out around the end of epoch 1 "
            "(≈2.02) and rose to 2.12 by the end of epoch 3 while the training loss kept falling, so the adapter increasingly fits the training "
            "demos; closed-loop success still improved until epoch 3. Most of every model's gap to the fully fine-tuned π₀-FAST LIBERO lies in "
            "LIBERO-10 (long, two-step tasks), where errors accumulate over up to 520 steps. One seed and 10 episodes per task (95% interval "
            "about ±4 points on the overall rate); the standard LIBERO protocol uses 50.</p>")

    # ---- weights
    add("<h2>8. Released LoRA weights</h2>")
    add("<p>Adapter-only checkpoints (LoRA + action-row adapters, fp32 safetensors) plus <code>meta.json</code> (codec parameters, normalization statistics, base-model revision). "
        "The frozen base model is downloaded from Hugging Face on load. Stored in git with Git LFS under <code>experiment/adaptive_dwt_vla/weights/</code>.</p>")
    body = []
    pilot_rows = {r["model"]: r for r in rows}
    full_ok = {r[0]: r for r in (full or [])}
    for d in sorted(WEIGHTS.glob("*/adapters.safetensors")):
        name = d.parent.name
        fw = re.match(r"^(full_(?:s0|e3|e4)_\w+?)_step_(\d+)$", name)
        if fw:
            v = json.loads((d.parent / "meta.json").read_text())["variant"]
            r = full_ok.get(f"full40_{v}@{fw.group(1)}:step_{int(fw.group(2)):05d}")
            succ = f"{sum(r[5].values())}/{len(r[5])} (full LIBERO)" if r else "–"
            lab = ({"dct_fixed": "DCT, no BPE", "poly_fixed": "Orthogonal polynomials, no BPE"}.get(v, "Haar DWT fixed")
                   + f", all 40 tasks, {r[4] if r else FULL_LOSS.get(v, v)}")
            body.append([f"<code>weights/{esc(name)}</code>", esc(lab), int(fw.group(2)), f"{d.stat().st_size / 1e6:.1f} MB", succ,
                         f"<code>{sha256(d)[:16]}…</code>"])
            continue
        m = re.match(r"^(lc_s(\d+)|pilot)_(\w+?)_step_(\d+)$", name)
        if not m:
            continue
        v, step = m.group(3), int(m.group(4))
        if m.group(2) is not None:
            r = rowmap.get((v, int(m.group(2)), step))
        else:
            r = pilot_rows.get(f"{v}@pilot_{v}:step_{step:05d}")
        succ = f"{r['successes']}/{r['episodes']}" if r else "–"
        body.append([f"<code>weights/{esc(name)}</code>", esc(LABEL.get(v, dict(FOLLOW_UPS).get(v, v))), step, f"{d.stat().st_size / 1e6:.1f} MB", succ, f"<code>{sha256(d)[:16]}…</code>"])
    add(table(["path", "variant", "step", "adapter size", "success (20 eps on the training task, or 400 on full LIBERO)", "sha256"], body))
    add("<p>Load one for inference:</p><pre>from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import load_checkpoint\n"
        "policy, meta = load_checkpoint(\"experiment/adaptive_dwt_vla/weights/lc_s2_dwt_fixed_step_02000\")</pre>")
    add("<p>The web studio exposes <code>lc_s2_dwt_fixed_step_02000</code> and <code>lc_s2_dwt_adaptive_step_02000</code> as “π₀-FAST + DWT (fixed / adaptive)”.</p>")

    # ---- wavelet choice
    ws = RES / "wavelet_selection.json"
    if ws.exists():
        wrows = json.loads(ws.read_text())["rows"]
        hzs = sorted({r["horizon"] for r in wrows})
        best = {}
        for r in wrows:
            k = (r["name"], r["horizon"])
            if k not in best or r["score"] < best[k]["score"]:
                best[k] = r
        names = sorted({r["name"] for r in wrows}, key=lambda n: np.mean([best[(n, hz)]["score"] for hz in hzs]))
        show = names[:6] + [n for n in ("haar", "db4", "sym4", "bior4.4") if n not in names[:6]]
        body = []
        for n in show:
            cells = [f"{names.index(n) + 1}", f"<code>{esc(n)}</code>"]
            for hz in hzs:
                b = best[(n, hz)]
                cells += [esc(b["mode"]), num(b["tokens_adaptive@fast"], 1),
                          "–" if not np.isfinite(b["score"]) else f"{100 * (b['score'] - 1):+.1f}%",
                          "–" if "op_jerk_ratio" not in b else f"{b['op_jerk_ratio']:.2f}"]
            body.append(cells)
        hdr = ["rank", "transform"] + [c for hz in hzs for c in (f"H{hz} mode", f"H{hz} tokens @FAST", f"H{hz} vs Haar", f"H{hz} jerk ratio")]
        add("<h2>9. Wavelet choice (offline selection test)</h2>")
        add(f"<p>The main models use Haar (db1). An offline test compared {len(names)} transforms (20 standard wavelets and mixed transforms that use a different wavelet "
            "at each level, each in periodization and symmetric mode) by the coefficient tokens needed to match FAST's reconstruction accuracy on held-out windows. "
            "Top 6 plus common wavelets for reference; full results, rate–distortion curves and basis-shape plots in "
            "<a href='wavelet_selection.html'>wavelet_selection.html</a>.</p>")
        add(table(hdr, body))
        add("<p>Token counts here exclude gripper and structure tokens and use whole-window transforms, so they are not the deployed codec's counts. "
            "This measures how compactly each transform represents the motion, not how well a policy learns its tokens. Of the alternatives, only "
            "sym4→haar has been trained, at 10-step chunks where it has no offline advantage (section 6).</p>")

    # ---- caveats
    add("<h2>10. Caveats</h2><ul>")
    for c in [
        "Sections 1–7 use one task and one instruction, and the zero-shot probe on nine new objects (section 7) mostly fails for every model. "
        "The multi-task results (section 7b) are one seed with 10 episodes per task; nothing here covers other robots.",
        "20 episodes per checkpoint and 3 seeds: confidence intervals are wide, and the paired tests are not corrected for multiple comparisons.",
        "The FAST baseline learns LIBERO FAST tokens through the same small LoRA budget. Its low value-token accuracy and malformed streams suggest part of the gap is "
        "how hard BPE tokens are to adapt under this recipe; the fully fine-tuned reference shows FAST itself can solve the task. This was tested directly: "
        "adding BPE to DWT made it much worse (one seed, section 6), and removing BPE from DCT made it much better (three seeds, sections 4–5).",
        "Training is not bit-reproducible: rerunning seed 0 did not exactly reproduce the pilot because of nondeterministic GPU kernels.",
        "Models are LoRA fine-tunes, not full fine-tunes; the base model stays frozen.",
    ]:
        add(f"<li>{esc(c)}</li>")
    add("</ul>")

    add("<h2>11. Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla\n"
        "PY=~/miniconda3/envs/smooth-operator-vla/bin/python\n"
        "$PY scripts/prepare_libero.py                 # index + train/val split (downloads lerobot/libero)\n"
        "bash scripts/run_learning_curve.sh 0 1 2      # 9 runs × 2000 steps + checkpoint evals (≈24 h on one A6000)\n"
        "bash scripts/run_variant.sh dct_fixed 0        # follow-up runs: dwt_fixed_bpe, dct_fixed (seeds 0-2), dwt_fixed_s4h, dwt_fixed_lw, dwt_fixed_slw\n"
        "bash scripts/run_generalization.sh 0 dwt_fixed dwt_fixed_s4h dct_fixed dwt_adaptive dwt_fixed_lw dwt_fixed_slw   # section 7\n"
        "bash scripts/run_full_libero.sh dwt_fixed_slw  # section 7b, epoch 1; then STEPS=43500 STOP=29000 and STOP=43500 for epochs 2-3\n"
        "bash scripts/run_full_e4_lw.sh dwt_fixed       # epoch 4 from the epoch-3 checkpoint (also dwt_fixed_lw, dwt_fixed_slw)\n"
        "bash scripts/run_full_dct_control.sh           # DCT without BPE, 4 epochs with the same lr history, eval after each\n"
        "$PY scripts/write_reports.py codec summary\n"
        "$PY scripts/write_training_report.py          # this file</pre>")
    add("</body></html>")
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "training_report.html"
    out.write_text("\n".join(h))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

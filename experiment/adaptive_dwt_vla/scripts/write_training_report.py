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

from write_reports import load_run, mcnemar_exact  # noqa: E402
from lerobot_policy_adaptive_dwt.metrics import wilson_ci  # noqa: E402

VARIANTS = ("fast_matched", "dwt_fixed", "dwt_adaptive")
LABEL = {"fast_matched": "FAST (DCT + BPE)", "dwt_fixed": "DWT fixed plan", "dwt_adaptive": "DWT adaptive"}
COLOR = {"fast_matched": "#d9534f", "dwt_fixed": "#2a7ab9", "dwt_adaptive": "#3c9d5d"}
SEEDS = (0, 1, 2)
DASH = {0: "-", 1: "--", 2: ":"}
COEF_ACC = {"fast_matched": "val_acc_fast", "dwt_fixed": "val_acc_coef", "dwt_adaptive": "val_acc_coef"}


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
    rows = json.loads((RES / "eval.json").read_text())["rows"]

    def episodes_of(r):
        suite, task = r["task"].split("#")
        p = RES / "eval" / f"{r['model'].replace(':', '_').replace('/', '_')}__{suite}_t{task}.json"
        return json.loads(p.read_text())["episodes"] if p.exists() else []

    return rows, episodes_of


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
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=False)
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

    add("<h3>The three action tokenizers</h3>")
    ad, fx = codec["adaptive"], codec["fixed"]
    add(table(["variant", "encoding", "tokens / chunk (train windows)", "vocabulary", "decoding at inference"], [
        [esc(LABEL["fast_matched"]), "DCT per dimension → quantize → BPE (stock π₀-FAST tokenizer)", "≈31 (measured in rollouts)", "FAST BPE vocab (pretrained on other robots)",
         "free generation; relaxed decode pads/truncates to 70 coefficients"],
        [esc(LABEL["dwt_fixed"]), f"Haar (db1) wavelet, periodization; one fixed coefficient plan <code>{esc(fx['plan'])}</code>", f"{num(fx['train']['tokens_mean'], 1)}",
         f"{fx['train']['vocab_size']} reserved tokens", "grammar-constrained (FSM-masked) decoding → inverse DWT"],
        [esc(LABEL["dwt_adaptive"]), f"Haar (db1), periodization; plan chosen per chunk by a rate–distortion DP oracle (γ = {ad['weights']['gamma']})",
         f"{num(ad['train']['tokens_mean'], 1)} (p95 {num(ad['train']['tokens_p95'], 0)})", f"{ad['train']['vocab_size']} reserved tokens", "grammar-constrained decoding → inverse DWT"],
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
    body = []
    for v in ("dwt_fixed", "dwt_adaptive"):
        for step in sorted(s for (vv, s) in res if vv == v):
            b = c = 0
            for s in SEEDS:
                a, f_ = res.get((v, step), {}).get(s, {}), res.get(("fast_matched", step), {}).get(s, {})
                for i in a.keys() & f_.keys():
                    b += a[i] and not f_[i]
                    c += f_[i] and not a[i]
            body.append([esc(LABEL[v]), step, b, c, f"{mcnemar_exact(b, c):.3f}"])
    add("<h3>Paired test against FAST (same seed, step and initial state; exact McNemar)</h3>")
    add(table(["variant", "step", "only DWT succeeds", "only FAST succeeds", "p"], body))

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
    add(table(["variant (step 2000, mean over seeds)", "tokens / chunk", "latency ms / chunk", "p95 ms"], body))
    add("<p>Latency is autoregressive generation of one chunk with a KV cache on the A6000. DWT streams are about twice as long as FAST streams, so latency roughly doubles; "
        "at 10 executed actions per chunk this is still within LIBERO's simulated control loop but would need speculative or parallel decoding for real-time control at higher rates.</p>")

    # ---- weights
    add("<h2>6. Released LoRA weights</h2>")
    add("<p>Adapter-only checkpoints (LoRA + action-row adapters, fp32 safetensors) plus <code>meta.json</code> (codec parameters, normalization statistics, base-model revision). "
        "The frozen base model is downloaded from Hugging Face on load. Stored in git with Git LFS under <code>experiment/adaptive_dwt_vla/weights/</code>.</p>")
    body = []
    pilot_rows = {r["model"]: r for r in rows}
    for d in sorted(WEIGHTS.glob("*/adapters.safetensors")):
        name = d.parent.name
        m = re.match(r"^(lc_s(\d+)|pilot)_(\w+?)_step_(\d+)$", name)
        if not m:
            continue
        v, step = m.group(3), int(m.group(4))
        if m.group(2) is not None:
            r = rowmap.get((v, int(m.group(2)), step))
        else:
            r = pilot_rows.get(f"{v}@pilot_{v}:step_{step:05d}")
        succ = f"{r['successes']}/{r['episodes']}" if r else "–"
        body.append([f"<code>weights/{esc(name)}</code>", esc(LABEL.get(v, v)), step, f"{d.stat().st_size / 1e6:.1f} MB", succ, f"<code>{sha256(d)[:16]}…</code>"])
    add(table(["path", "variant", "step", "adapter size", "success (20 eps)", "sha256"], body))
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
        add("<h2>7. Wavelet choice (offline selection test)</h2>")
        add(f"<p>All trained models use Haar (db1). An offline test compared {len(names)} transforms (20 standard wavelets and mixed transforms that use a different wavelet "
            "at each level, each in periodization and symmetric mode) by the coefficient tokens needed to match FAST's reconstruction accuracy on held-out windows. "
            "Top 6 plus common wavelets for reference; full results, rate–distortion curves and basis-shape plots in "
            "<a href='wavelet_selection.html'>wavelet_selection.html</a>.</p>")
        add(table(hdr, body))
        add("<p>Token counts here exclude gripper and structure tokens and use whole-window transforms, so they are not the deployed codec's counts. "
            "This measures how compactly each transform represents the motion, not how well a policy learns its tokens; none of the alternatives has been trained.</p>")

    # ---- caveats
    add("<h2>8. Caveats</h2><ul>")
    for c in [
        "One task and one instruction. Nothing here says how the tokenizers behave in multi-task training or on other robots.",
        "20 episodes per checkpoint and 3 seeds: confidence intervals are wide, and the paired tests are not corrected for multiple comparisons.",
        "The FAST baseline learns LIBERO FAST tokens through the same small LoRA budget. Its low value-token accuracy and malformed streams suggest part of the gap is "
        "how hard BPE tokens are to adapt under this recipe; the fully fine-tuned reference shows FAST itself can solve the task.",
        "Training is not bit-reproducible: rerunning seed 0 did not exactly reproduce the pilot because of nondeterministic GPU kernels.",
        "Models are LoRA fine-tunes, not full fine-tunes; the base model stays frozen.",
    ]:
        add(f"<li>{esc(c)}</li>")
    add("</ul>")

    add("<h2>9. Reproduce</h2><pre>cd smooth-operator/experiment/adaptive_dwt_vla\n"
        "PY=~/miniconda3/envs/smooth-operator-vla/bin/python\n"
        "$PY scripts/prepare_libero.py                 # index + train/val split (downloads lerobot/libero)\n"
        "bash scripts/run_learning_curve.sh 0 1 2      # 9 runs × 2000 steps + checkpoint evals (≈24 h on one A6000)\n"
        "$PY scripts/write_reports.py codec summary\n"
        "$PY scripts/write_training_report.py          # this file</pre>")
    add("</body></html>")
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "training_report.html"
    out.write_text("\n".join(h))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

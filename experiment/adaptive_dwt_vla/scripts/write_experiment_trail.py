"""Experiment trail: a mind-map of every experiment branch, its status and its measured outcome.

Statuses and numbers are read from runs/ and results/ each time, so re-running this script after a
run finishes updates the map. Writes experiment/reports/experiment_trail.html (self-contained).
Run with the VLA env:  python scripts/write_experiment_trail.py
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUNS = ROOT / "runs"
REPORTS = ROOT.parent / "reports"
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_adaptive_dwt.metrics import wilson_ci  # noqa: E402


def esc(x) -> str:
    return html.escape(str(x))


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def eval_rows():
    d = jload(RES / "eval.json")
    return d["rows"] if d else []


def episodes_of(r):
    suite, task = r["task"].split("#")
    p = RES / "eval" / f"{r['model'].replace(':', '_').replace('/', '_')}__{suite}_t{task}.json"
    return jload(p)["episodes"] if p.exists() else []


def lc_pooled(rows, variant, step):
    pat = re.compile(rf"^{variant}@lc_s(\d+)_{variant}:step_{step:05d}$")
    hits = [r for r in rows if pat.match(r["model"])]
    k, n = sum(r["successes"] for r in hits), sum(r["episodes"] for r in hits)
    return k, n, len(hits)


def run_status(name):
    st = jload(RUNS / name / "status.json")
    if not st:
        return "planned", None
    if st.get("state") in ("running", "starting"):
        return "running", st
    if st.get("state") == "failed":
        return "failed", st
    return "done", st


# ----------------------------------------------------------------------------------------------- nodes


# Facts shared by many nodes (from runs/*/config.json and the model cards).
BASE = ("π₀-FAST-base (lerobot/pi0fast-base @308eb844), pretrained by Physical Intelligence on its own robot data with FAST "
        "tokens. It has never been trained on LIBERO. Its weights stay frozen.")
NO_MODEL = "None: no neural network is trained or run. This is an offline test of the tokenizer on recorded actions."
OUR_DATA = ("LIBERO, ONE task only: libero_object #0, “pick up the alphabet soup and place it in the basket” "
            "(task 24 of lerobot/libero @a1aaacb), 40 training demos, 4 validation demos.")
EVAL = "LIBERO simulator, the same task, initial states 0–19 (20 episodes per checkpoint)."
REFS_TXT = ("LeRobot's public checkpoints lerobot/pi0fast-libero (π₀-FAST) and lerobot/pi0_libero_finetuned_v044 (π₀), "
            "fine-tuned by LeRobot on all of LIBERO (four suites, every task). We only downloaded and evaluated them; they show what a fully trained policy reaches.")
FAST_TOK = "jadechoghari/tokenizer-lib-mean (a FAST DCT + BPE tokenizer for LIBERO actions)"


def lora(steps, extra=""):
    return (f"LoRA rank 16 on every Gemma attention + MLP layer, plus an action-token row adapter (≈21–24M trained of 3.5B parameters). "
            f"{steps}, effective batch 16, lr 2e-4.{(' ' + extra) if extra else ''}")


def build_nodes():
    rows = eval_rows()
    by_model = {r["model"]: r for r in rows}
    refs = ", ".join(f"{r['model'].removeprefix('ref_')} {r['successes']}/{r['episodes']}" for r in rows if r["model"].startswith("ref_"))
    N = {}

    def node(nid, parent, title, status, why, question, setup, result, notes="", link=None, short=None):
        N[nid] = {"id": nid, "parent": parent, "title": title, "status": status, "why": why, "question": question,
                  "setup": setup, "result": result, "short": short or result, "notes": notes, "link": link, "children": []}

    node("root", None, "Wavelet action tokens for π₀-FAST", "done",
         "π₀-FAST writes each action chunk as FAST tokens (DCT + BPE). A wavelet (DWT) tokenizer can spend tokens only where the motion "
         "is detailed, and its fixed token layout lets a grammar rule out malformed outputs. Does that make a better VLA action format?",
         "Can a multiresolution (DWT) action tokenizer replace FAST's DCT + BPE tokens in a VLA?",
         [("Base model (all our trained runs)", BASE),
          ("Our training data", OUR_DATA),
          ("Our training", lora("2,000–5,000 steps", "Tokenizers are compared under an identical recipe; only the action tokens differ.")),
          ("Evaluation", EVAL),
          ("Reference models (not ours)", REFS_TXT + (f" Measured: {refs}." if refs else ""))],
         "See the branches. Headline: with the same LoRA budget on one task, DWT tokens train into a far better policy than FAST tokens.",
         short="LoRA on π₀-FAST-base, one LIBERO task")

    c10, c32 = jload(RES / "codec_choice_H10.json"), jload(RES / "codec_choice_H32.json")
    fast_tok = c10["fast_libero_val"]["tokens_mean"] if c10 else 25
    fix_tok = c10["fixed"]["val"]["tokens_mean"] if c10 else 69
    if c10:
        a, f, x = c10["adaptive"]["val"], c10["fast_libero_val"], c10["fixed"]["val"]
        res = (f"10-step chunks: adaptive DWT {a['tokens_mean']:.1f} tokens (MSE {a['mse']:.1e}), fixed DWT {x['tokens_mean']:.1f} (MSE {x['mse']:.1e}), "
               f"FAST {f['tokens_mean']:.1f} (MSE {f['mse']:.1e}).")
        if c32:
            res += f" 32-step chunks: adaptive {c32['adaptive']['val']['tokens_mean']:.1f}, FAST {c32['fast_libero_val']['tokens_mean']:.1f}."
        node("A", "root", "A · Offline codec", "done",
             "First experiment. Before spending GPU time, check that Haar wavelet tokens can represent LIBERO actions accurately and at what token cost.",
             "Can Haar DWT tokens encode LIBERO actions accurately, and at what token cost compared with FAST?",
             [("Model", NO_MODEL),
              ("Data", "LIBERO actions (lerobot/libero), all 40 tasks: codec fitted on the training split, scored on held-out episodes."),
              ("Compared with", f"FAST tokenizer {FAST_TOK} on the same chunks.")],
             res, "DWT reaches lower error but needs about 2–3× more tokens per chunk: BPE merges are FAST's advantage. "
             "The fixed plan and the adaptive γ used in all later training were chosen here.",
             short="DWT more accurate, ≈2–3× tokens", link="training_report.html")

    dc = jload(RUNS / "overfit_dwt_adaptive" / "overfit_decode_check.json")
    node("B", "A", "B · Overfit sanity check", "done" if dc else "planned",
         "Stage A showed the tokenizer is accurate. Next, make sure training and grammar-constrained decoding work end to end before multi-hour runs.",
         "Can the model memorize a few chunks and then decode them exactly under the grammar?",
         [("Base model", BASE),
          ("Training data", "64 chunks from the training task (libero_object #0), memorized on purpose."),
          ("Training", "LoRA + row adapter (20.8M trained parameters), 300 steps, lr 5e-4, adaptive DWT tokens.")],
         f"Constrained decoding reproduced {dc['exact_stream_reproduction']}/{dc['windows']} memorized token streams exactly." if dc else "–",
         "Gate check only; not a policy.", short=f"{dc['exact_stream_reproduction']}/{dc['windows']} exact" if dc else None)

    p = {v: by_model.get(f"{v}@pilot_{v}:step_02000") for v in ("fast_matched", "dwt_fixed", "dwt_adaptive")}
    p5 = {v: by_model.get(f"{v}@pilot_{v}:step_05000") for v in ("fast_matched", "dwt_fixed", "dwt_adaptive")}
    if all(p.values()):
        s2 = ", ".join(f"{v} {p[v]['successes']}/{p[v]['episodes']}" for v in p)
        s5 = ", ".join(f"{v} {p5[v]['successes']}/{p5[v]['episodes']}" for v in p5 if p5[v])
        fm = p["fast_matched"]
        node("CD", "B", "C+D · Pilot training + LIBERO eval", "done",
             "Stage B showed the pipeline works. This is the first real comparison of the tokenizers under an identical training budget.",
             "Which action tokenizer trains into a better policy?",
             [("Base model", BASE + " Same base for all three runs."),
              ("Training data", OUR_DATA),
              ("Training", lora("5,000-step cosine schedule, seed 0, checkpoints at 2,000 and 5,000 steps")),
              ("Runs (only the tokenizer differs)", f"fast_matched: FAST tokens via {FAST_TOK} · dwt_fixed: Haar DWT, fixed 8+2 plan · "
               "dwt_adaptive: Haar DWT, per-chunk plan"),
              ("Evaluation", EVAL),
              ("Reference models (not ours)", REFS_TXT)],
             f"2,000 steps: {s2}. 5,000 steps: {s5}. References: {refs}.",
             f"The FAST baseline's output was malformed in {100 * fm['invalid_streams'] / fm['chunks']:.0f}% of chunks (see the child node).",
             short=f"DWT > FAST at 2k: {p['dwt_adaptive']['successes']}, {p['dwt_fixed']['successes']} vs {fm['successes']}/20", link="training_report.html")
        node("CDf", "CD", "Finding: FAST baseline streams malformed", "done",
             "In C+D the FAST baseline scored far below DWT although the base model was pretrained on FAST tokens. Check whether the "
             "policy or the token format is failing.",
             "Why does the FAST baseline do poorly?",
             [("Model", "The C+D fast_matched run (no new training); its decoded token streams were analysed."),
              ("Data", "The C+D evaluation episodes.")],
             f"{fm['invalid_streams']}/{fm['chunks']} FAST chunks did not decode to exactly 70 coefficients (10 steps × 7 dims); they were executed after a relaxed decode.",
             "BPE tokens adapted with a small LoRA budget are hard to emit exactly; the DWT grammar makes malformed streams impossible.",
             short=f"{100 * fm['invalid_streams'] / fm['chunks']:.0f}% malformed chunks")

    lc = {v: lc_pooled(rows, v, 2000) for v in ("fast_matched", "dwt_fixed", "dwt_adaptive")}
    if all(n for _, n, _ in lc.values()):
        txt = ", ".join(f"{v} {k}/{n} = {k / n:.2f}" for v, (k, n, _) in lc.items())
        k1 = {v: lc_pooled(rows, v, 1000) for v in lc}
        node("LC", "CD", "Learning curve · 3 seeds", "done",
             "C+D used one seed and 20 episodes, so a gap that large could be luck, and two checkpoints say nothing about learning speed.",
             "Does the DWT advantage hold across seeds, and which tokenizer learns faster?",
             [("Base model", BASE),
              ("Training data", OUR_DATA),
              ("Training", lora("Same 5,000-step cosine schedule stopped at 2,000 steps; seeds 0, 1, 2 × three tokenizers = 9 runs",
                                "Checkpoints at 250, 500, 1,000, 1,500 and 2,000 steps.")),
              ("Evaluation", EVAL + " Every checkpoint of every run (45 × 20 episodes).")],
             f"Pooled success at 2,000 steps: {txt}. At 1,000 steps: " + ", ".join(f"{v} {k / n:.2f}" for v, (k, n, _) in k1.items() if n) + ".",
             "Fixed-plan DWT is best from 1,000 steps on; adaptive must also predict its own plan and learns more slowly.",
             short=f"fixed {lc['dwt_fixed'][0] / lc['dwt_fixed'][1]:.2f} · adaptive {lc['dwt_adaptive'][0] / lc['dwt_adaptive'][1]:.2f} · FAST {lc['fast_matched'][0] / lc['fast_matched'][1]:.2f}",
             link="training_report.html")
        node("W", "LC", "Released weights + training report", "done",
             "The learning curve confirmed the result, so the trained adapters were published to make it reproducible.",
             "What was released?",
             [("What is in git", "LoRA + row-adapter weights only (Git LFS), 12 final checkpoints: 9 learning-curve runs + 3 pilots. "
               "The base model is not included; it is downloaded from Hugging Face."),
              ("Trained on", OUR_DATA)],
             "12 adapter checkpoints with SHA-256 sums and an HTML training report.", link="training_report.html", short="12 checkpoints on git (LFS)")

    ws = jload(RES / "wavelet_selection.json")
    if ws:
        best = {}
        for r in ws["rows"]:
            k = (r["name"], r["horizon"])
            if k not in best or r["score"] < best[k]["score"]:
                best[k] = r
        t32 = min((best[k] for k in best if k[1] == 32), key=lambda r: r["score"])
        t10 = min((best[k] for k in best if k[1] == 10), key=lambda r: r["score"])
        node("WS", "A", "Wavelet selection test · 26 transforms", "done",
             "Stage A used only Haar, whose step-shaped basis gives blocky reconstructions. Check whether a smoother or per-level mixed wavelet is more compact.",
             "Would another wavelet represent actions more compactly than Haar?",
             [("Model", NO_MODEL),
              ("Data", "LIBERO actions, all 40 tasks: quantizers fitted on the training split, scored on held-out episodes; 10- and 32-step chunks."),
              ("Candidates", "20 standard wavelets + 6 per-level mixed transforms, periodization and symmetric boundary modes.")],
             f"No standard wavelet beats Haar. Best at 32 steps: {t32['name']} ({t32['mode']}) {100 * (t32['score'] - 1):+.1f}% tokens; at 10 steps: {t10['name']} {100 * (t10['score'] - 1):+.1f}%.",
             "Whole-chunk transforms without the segmentation search that the training codec uses.",
             link="wavelet_selection.html", short=f"Haar ≈ best; {t32['name']} {100 * (t32['score'] - 1):+.1f}% at H=32")

    def bpe_run(variant, ref_variant):
        name = f"lc_s0_{variant}"
        st, s = run_status(name)
        cfg = jload(RUNS / name / "config.json")
        res = "–"
        if cfg and cfg.get("bpe"):
            b = cfg["bpe"]
            res = f"BPE shortens training streams from {b['train_tokens_mean_before']:.1f} to {b['train_tokens_mean_after']:.1f} tokens ({b['local_tokens']}-token vocabulary)."
        ev = [(int(m.group(1)), r) for r in rows if (m := re.match(rf"^{variant}@{name}:step_(\d+)$", r["model"]))]
        if ev:
            res += " Success: " + ", ".join(f"{step}: {r['successes']}/{r['episodes']}" for step, r in sorted(ev)) + "."
            ref = by_model.get(f"{ref_variant}@lc_s0_{ref_variant}:step_02000")
            if ref:
                res += f" (Without BPE, same seed, 2,000 steps: {ref['successes']}/{ref['episodes']}.)"
        if st == "running" and s:
            res += f" Training: step {s.get('step', 0)} / {s.get('stop_at', 2000)}."
        if st == "done" and not ev:
            st = "running"
            res += " Training done; evaluation pending."
        return st, res

    bpe_setup = lambda tok, ref: [  # noqa: E731
        ("Base model", BASE),
        ("Training data", OUR_DATA),
        ("Training", lora("Seed 0, same recipe as the learning curve (5,000-step schedule stopped at 2,000 steps)")),
        ("Tokenizer", tok),
        ("Compared with", ref),
        ("Evaluation", EVAL + " Every checkpoint (250–2,000 steps).")]
    st, res = bpe_run("dwt_fixed_bpe", "dwt_fixed")
    node("BPE", "LC", "BPE over DWT tokens (fixed Haar) · smoke test", st,
         f"The learning curve showed DWT wins on success but uses ≈{fix_tok / fast_tok:.1f}× FAST's tokens per chunk ({fix_tok:.0f} vs {fast_tok:.0f}), "
         "which means slower inference. FAST gets its short streams from BPE, so try BPE on the DWT coefficients.",
         "Can FAST-style BPE close DWT's token gap without losing its learning advantage?",
         bpe_setup("Fixed-plan Haar DWT + BPE merges of coefficient runs (structure and gripper tokens never merged, so the grammar still applies); 1,024 tokens.",
                   "lc_s0_dwt_fixed: same seed and recipe, no BPE."),
         res, "Smoke test: one seed only.", short=res.split(". ")[0] if res != "–" else "1 seed, 2000 steps")

    oc = jload(RES / "bpe_offline_check.json")
    if oc:
        r = {(x["codec"], x["mode"]): x for x in oc["rows"]}
        fmt = lambda x: f"{x['tokens']:.1f} → {x['tokens_bpe']:.1f} tokens with BPE, val MSE {x['val_mse']:.1e}, jerk {x['jerk_ratio']:.2f}"  # noqa: E731
        node("S4H", "BPE", "Finding: sym4→haar loses at 10-step chunks", "done",
             "The wavelet test ranked mix:sym4→haar best at 32-step chunks, and the next step was to train it with BPE. "
             "Our policy uses 10-step chunks, so it was checked offline at that length first.",
             "Is the wavelet-test winner worth training with BPE on 10-step chunks?",
             [("Model", NO_MODEL),
              ("Data", "Training task only (libero_object #0): codecs and BPE fitted on its 40 training demos, scored on its 4 validation demos, "
               "with the same normalization as training."),
              ("Compared", "Haar vs sym4→haar, fixed vs adaptive depth, with and without BPE.")],
             "; ".join(f"{c} {m}: {fmt(x)}" for (c, m), x in r.items()) + ".",
             "Haar's 8+2 segmentation suits 10 steps, while sym4→haar needs symmetric padding (16 coefficients per dimension instead of 10). "
             "Its only advantage is smoother output. Not trained; revisit with 32-step chunks.",
             short=f"Haar wins: {r[('haar', 'adaptive')]['tokens_bpe']:.1f} vs {r[('mix:sym4→haar', 'adaptive')]['tokens_bpe']:.1f} tokens (adaptive + BPE)")
    st, res = bpe_run("dwt_adaptive_bpe", "dwt_adaptive")
    if res == "–":
        st, res = "planned", "Queued to start after the fixed-Haar BPE smoke test."
    node("ABPE", "S4H" if oc else "BPE", "BPE over adaptive Haar tokens", st,
         "The offline check (parent node) found Haar with per-chunk plans + BPE gives the fewest tokens (≈50 per chunk) at the lowest error "
         "of the four codec/BPE combinations, so it replaced the sym4→haar run.",
         "Does BPE on adaptive plans give the fewest tokens without hurting success?",
         bpe_setup("Haar DWT with per-chunk plans chosen by the oracle + BPE merges of coefficient runs; 1,024 tokens.",
                   "lc_s0_dwt_adaptive: same seed and recipe, no BPE."),
         res, "One seed only.", short=res.split(". ")[0] if st != "planned" else "queued, 1 seed")
    node("E", "LC", "Stage E · 32-step chunks", "planned",
         "Offline, at 32-step chunks adaptive DWT saves 17% tokens over fixed and sym4→haar beats Haar by about 5%; longer chunks are where "
         "those advantages should show up in a policy.",
         "Does adaptive DWT (and sym4→haar) pay off with longer chunks?",
         [("Base model", BASE), ("Training data", OUR_DATA + " (planned)"), ("Training", lora("Planned, 32-step chunks"))],
         "Not run.", short="not run")
    for n in N.values():
        if n["parent"]:
            N[n["parent"]]["children"].append(n["id"])
    extra_edges = [("WS", "S4H"), ("WS", "E")]
    return N, extra_edges


# ----------------------------------------------------------------------------------------------- layout + html


COLW, ROWH, BOXW, BOXH = 270, 112, 232, 92
COLOR = {"done": "#3c9d5d", "running": "#e0a800", "planned": "#8a94a6", "failed": "#d9534f"}


def layout(N):
    pos, y = {}, [0]

    def place(nid, depth):
        kids = N[nid]["children"]
        if not kids:
            pos[nid] = (depth, y[0])
            y[0] += 1
        else:
            for k in kids:
                place(k, depth + 1)
            pos[nid] = (depth, sum(pos[k][1] for k in kids) / len(kids))

    place("root", 0)
    return pos


def render(N, extra):
    pos = layout(N)
    W = (max(d for d, _ in pos.values()) + 1) * COLW + 40
    H = (max(r for _, r in pos.values()) + 1) * ROWH + 40

    def xy(nid):
        d, r = pos[nid]
        return 20 + d * COLW, 20 + r * ROWH

    edges = []
    for n in N.values():
        if n["parent"]:
            (x0, y0), (x1, y1) = xy(n["parent"]), xy(n["id"])
            edges.append((x0 + BOXW, y0 + BOXH / 2, x1, y1 + BOXH / 2, False))
    for a, b in extra:
        if a in N and b in N:
            (x0, y0), (x1, y1) = xy(a), xy(b)
            edges.append((x0 + BOXW, y0 + BOXH / 2, x1, y1 + BOXH / 2, True))
    svg = [f"<svg width='{W}' height='{H}' style='position:absolute;left:0;top:0'>"]
    for x0, y0, x1, y1, dashed in edges:
        mx = (x0 + x1) / 2
        svg.append(f"<path d='M{x0},{y0} C{mx},{y0} {mx},{y1} {x1},{y1}' fill='none' stroke='{'#b8c0cc' if dashed else '#7d8798'}' "
                   f"stroke-width='{1.5 if dashed else 2}' {'stroke-dasharray=\"6 5\"' if dashed else ''}/>")
    svg.append("</svg>")
    boxes = []
    for n in N.values():
        x, y = xy(n["id"])
        c = COLOR[n["status"]]
        boxes.append(f"<div class='node' data-id='{n['id']}' style='left:{x}px;top:{y}px;border-color:{c}'>"
                     f"<span class='st' style='background:{c}'>{n['status']}</span><b>{esc(n['title'])}</b><small>{esc(n['short'])}</small></div>")
    detail = {k: {kk: n[kk] for kk in ("title", "status", "why", "question", "setup", "result", "notes", "link")} for k, n in N.items()}
    counts = {s: sum(1 for n in N.values() if n["status"] == s) for s in COLOR}
    legend = " ".join(f"<span class='lg'><i style='background:{c}'></i>{s} ({counts[s]})</span>" for s, c in COLOR.items() if counts[s])
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Experiment trail</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1d2330;background:#f6f7fa}}
header{{padding:1.2rem 1.6rem .4rem}} h1{{margin:0;font-size:1.6rem}} .sub{{color:#5b6475;margin:.3rem 0}}
.lg{{margin-right:1rem;font-size:.85rem}} .lg i{{display:inline-block;width:.8rem;height:.8rem;border-radius:3px;margin-right:.3rem;vertical-align:-1px}}
.wrap{{display:flex;gap:1rem;padding:0 1.6rem 1.6rem}} .map{{position:relative;overflow:auto;background:#fff;border:1px solid #dde2ea;border-radius:10px;flex:1;height:calc(100vh - 150px)}}
.canvas{{position:relative;width:{W}px;height:{H}px}}
.node{{position:absolute;width:{BOXW - 22}px;height:{BOXH - 18}px;padding:8px 10px;background:#fff;border:2px solid;border-radius:10px;cursor:pointer;box-shadow:0 1px 3px #0001;overflow:hidden}}
.node:hover,.node.sel{{box-shadow:0 0 0 3px #2a7ab933}} .node b{{display:block;font-size:.86rem;line-height:1.2;margin-top:2px}} .node small{{display:block;color:#5b6475;font-size:.76rem;margin-top:4px;line-height:1.25}}
.st{{float:right;color:#fff;font-size:.65rem;padding:1px 6px;border-radius:8px;margin-left:4px}}
aside{{width:420px;background:#fff;border:1px solid #dde2ea;border-radius:10px;padding:1rem 1.1rem;height:calc(100vh - 150px);overflow:auto;box-sizing:border-box}}
aside h2{{font-size:1.1rem;margin:.2rem 0 .6rem}} aside h3{{font-size:.8rem;text-transform:uppercase;color:#5b6475;margin:1rem 0 .2rem;letter-spacing:.04em}} aside p{{margin:.2rem 0;line-height:1.45;font-size:.9rem}}
a{{color:#2a7ab9}}
aside dl{{margin:.2rem 0;font-size:.86rem}} aside dt{{font-weight:600;margin-top:.5rem}} aside dd{{margin:.1rem 0 0;line-height:1.4;color:#2d3443}}
.why{{background:#f1f6fc;border-left:3px solid #2a7ab9;padding:.1rem .7rem .4rem;margin:.6rem 0;border-radius:4px}} .why h3{{margin-top:.5rem}}
.models{{display:flex;gap:.8rem;flex-wrap:wrap;margin:.5rem 0}} .models div{{background:#fff;border:1px solid #dde2ea;border-radius:8px;padding:.45rem .7rem;font-size:.82rem;max-width:30rem;line-height:1.35}}
.zoom{{margin-left:1rem}} .zoom button{{font-size:.8rem;margin-right:.3rem;padding:2px 8px;border:1px solid #c5ccd8;background:#fff;border-radius:5px;cursor:pointer}}
@media (max-width:1300px){{.wrap{{flex-direction:column}} aside{{width:auto;height:auto}} .map{{height:auto;min-height:300px}}}}
</style></head><body>
<header><h1>Experiment trail</h1><p class='sub'>Adaptive DWT vs FAST action tokens on π₀-FAST. Each box is one experiment; solid lines show what led to what, dashed lines show results that feed a later branch.
Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/write_experiment_trail.py</code> from runs/ and results/. Click a box for details.</p>
<div class='models'><div><b>Base model of every run we trained:</b> π₀-FAST-base, pretrained by Physical Intelligence, <b>never trained on LIBERO</b>, frozen.</div><div><b>Our runs:</b> LoRA + row adapter on that base, trained on <b>one LIBERO task</b> (40 demos), identical recipe for FAST and DWT tokens.</div><div><b>References (not ours):</b> LeRobot's π₀-FAST / π₀ checkpoints fine-tuned on <b>all of LIBERO</b>; only evaluated.</div></div><p>{legend}<span class='zoom'><button id='fit'>Fit</button><button id='full'>100%</button></span></p></header>
<div class='wrap'><div class='map' id='map'><div id='sizer'><div class='canvas' id='canvas'>{''.join(svg)}{''.join(boxes)}</div></div></div>
<aside id='panel'><h2>Select an experiment</h2><p>Click any box in the map.</p></aside></div>
<script>
const D={json.dumps(detail)};
const esc=s=>String(s).replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
function show(id){{const n=D[id];document.querySelectorAll('.node').forEach(e=>e.classList.toggle('sel',e.dataset.id===id));
const setup=n.setup.map(([k,v])=>`<dt>${{esc(k)}}</dt><dd>${{esc(v)}}</dd>`).join('');
document.getElementById('panel').innerHTML=`<h2>${{esc(n.title)}}</h2><p><b>Status:</b> ${{n.status}}</p>
<div class='why'><h3>Why we ran this</h3><p>${{esc(n.why)}}</p></div><h3>Question</h3><p>${{esc(n.question)}}</p>
<h3>Setup</h3><dl>${{setup}}</dl><h3>Result</h3><p>${{esc(n.result)}}</p>${{n.notes?`<h3>Notes</h3><p>${{esc(n.notes)}}</p>`:''}}${{n.link?`<h3>Report</h3><p><a href="${{n.link}}">${{n.link}}</a></p>`:''}}`;}}
document.querySelectorAll('.node').forEach(e=>e.onclick=()=>show(e.dataset.id));
show('root');
const cv=document.getElementById('canvas'),sz=document.getElementById('sizer'),mp=document.getElementById('map');
function zoom(s){{cv.style.transform=`scale(${{s}})`;cv.style.transformOrigin='0 0';sz.style.width={W}*s+'px';sz.style.height={H}*s+'px';}}
function fit(){{zoom(Math.min(1,(mp.clientWidth-8)/{W}));}}
document.getElementById('fit').onclick=fit;document.getElementById('full').onclick=()=>zoom(1);
fit();addEventListener('resize',fit);
</script></body></html>"""


def main() -> None:
    N, extra = build_nodes()
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "experiment_trail.html"
    out.write_text(render(N, extra))
    print("wrote", out, f"({len(N)} experiments)")


if __name__ == "__main__":
    main()

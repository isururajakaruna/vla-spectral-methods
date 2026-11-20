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
    return [r for r in d["rows"] if r["task"] == "libero_object#0"] if d else []


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
        n_w = len(list((ROOT / "weights").glob("*/adapters.safetensors")))
        node("W", "LC", "Released weights + training report", "done",
             "The learning curve confirmed the result, so the trained adapters were published to make it reproducible.",
             "What was released?",
             [("What is in git", f"LoRA + row-adapter weights only (Git LFS), {n_w} final checkpoints: 9 learning-curve runs, 3 pilots, then the "
               "follow-up step-2000 checkpoints (DCT without BPE × 3 seeds, sym4→haar, Haar fixed with the weighted and soft + weighted losses). "
               "The base model is not included; it is downloaded from Hugging Face."),
              ("Trained on", OUR_DATA)],
             f"{n_w} adapter checkpoints with SHA-256 sums and an HTML training report.", link="training_report.html", short=f"{n_w} checkpoints on git (LFS)")

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
             "Its only advantage is smoother output. Not trained with BPE; the fixed plan without BPE was trained later (child node).",
             short=f"Haar wins: {r[('haar', 'adaptive')]['tokens_bpe']:.1f} vs {r[('mix:sym4→haar', 'adaptive')]['tokens_bpe']:.1f} tokens (adaptive + BPE)")
    qs = jload(RES / "quant_sweep.json")
    if qs:
        fq = qs["fast"]
        pick = lambda st_, sch, sc, dz: next((x for x in qs["rows"] if (x["stream"], x["scheme"], x["scale"], x["deadzone"]) == (st_, sch, sc, dz)), None)  # noqa: E731
        pts = [("current quantizer, current stream", pick("segment", "uniform", 1, 0.0)),
               ("detail steps ×4, current stream", pick("segment", "uniform", 4, 0.0)),
               ("current quantizer, compact stream", pick("compact", "uniform", 1, 0.0)),
               ("finest details ×6 + zero band, compact stream (chosen)", pick("compact", "fine", 6, 0.3)),
               ("detail steps ×8, compact stream", pick("compact", "uniform", 8, 0.0))]
        node("QS", "BPE", "Finding: coarse detail quantization + compact stream", "done",
             "BPE alone only shortened DWT chunks from 69 to about 51 tokens. FAST is short mainly because it rounds high-frequency DCT "
             "coefficients coarsely (most become 0) and BPE packs the zeros; our DWT quantizer is nearly lossless, and about 9 tokens per "
             "chunk are structure tokens that a fixed plan does not need.",
             "How close to FAST's token count can DWT + BPE get at FAST's error?",
             [("Model", NO_MODEL),
              ("Data", "Same protocol as Stage A: LIBERO actions from all 40 tasks, fitted on the training split, scored on 2,674 held-out chunks."),
              ("Swept", "Detail-coefficient step ×1–32 (uniform, or finest details coarsest), optional zero band, current vs compact stream "
               "(coefficients, one gripper block, end: no LEN/RES tokens)."),
              ("Target", f"FAST: {fq['tokens']:.1f} tokens per chunk at MSE {fq['mse']:.1e}.")],
             " ".join(f"{name}: {x['tokens_bpe']:.1f} tokens, {x['mse_vs_fast']:.2f}× FAST's error." for name, x in pts if x),
             "At FAST's error DWT needs about 29 tokens (1.17× FAST). Matching FAST's token count exactly would cost about 2.7× its error. "
             "Not trained: after the BPE smoke test lost badly, a DCT control without BPE was run instead (scripts/quant_sweep.py).",
             short="≈29 tokens at FAST's error (FAST 25)")
    st, _ = bpe_run("dct_fixed", "dwt_fixed")
    steps = (250, 500, 1000, 1500, 2000)
    pooled = {v: {step: lc_pooled(rows, v, step) for step in steps} for v in ("dct_fixed", "dwt_fixed", "fast_matched")}
    n_seeds = max(p[2] for p in pooled["dct_fixed"].values())
    if n_seeds == 0:
        res, short = "Starting.", "queued"
    else:
        line = lambda v: ", ".join(f"{step}: {k}/{n}" for step, (k, n, _) in pooled[v].items() if n)  # noqa: E731
        res = (f"Success pooled over {n_seeds} seed(s), 20 episodes each. DCT without BPE: {line('dct_fixed')}. "
               f"Haar without BPE: {line('dwt_fixed')}. FAST (DCT + BPE): {line('fast_matched')}. "
               "DCT without BPE produced no malformed streams (FAST: 80–100% malformed). ")
        k2, n2, _ = pooled["dct_fixed"][2000]
        short = f"{k2}/{n2} @2k ({n_seeds} seeds)" if n2 else f"{n_seeds} seeds, evaluating"
        if n_seeds == 3 and n2 == 60:
            res += ("Removing BPE from the DCT closes most of the gap to Haar, so most of FAST's deficit is the token format (BPE, free decoding). "
                    "Haar still beats DCT in paired tests at 1,000 and 2,000 steps (p ≈ 0.002 and 0.04, uncorrected), so the transform matters too, less.")
        st = "done" if n_seeds == 3 and n2 == 60 else "running"
    node("DCT", "BPE", "DCT control without BPE", st,
         "The FAST baseline always used BPE (it is part of FAST), while every DWT run before the BPE smoke test did not, and adding BPE "
         "to DWT made it much worse. So the FAST-vs-DWT gap may come from BPE, not from DCT vs wavelets. This control replaced the planned "
         "coarse-quantization DWT run, because FAST's other ingredients (BPE, coarse rounding) are removed here too.",
         "With BPE removed from both, does the wavelet transform still beat the DCT?",
         bpe_setup("Whole-chunk orthonormal DCT-II (FAST's transform), 10 coefficients per dimension at full resolution, the same 256-level "
                   "quantizer, token vocabulary and grammar-constrained decoding as DWT; no BPE and no FAST-style coarse rounding. 64 tokens per chunk.",
                   "lc_s{0,1,2}_dwt_fixed (Haar, 69 tokens, no BPE) and lc_s{0,1,2}_fast_matched (FAST DCT + BPE), same seeds and recipe."),
         res, "Offline at equal quantization DCT and Haar are about equally accurate (MSE 1.6e-4 vs 1.45e-4 on all tasks). "
         "Seed 0 ran first; seeds 1 and 2 were added so the control matches the 3-seed learning curve.",
         short=short)
    s4 = {step: by_model.get(f"dwt_fixed_s4h@lc_s0_dwt_fixed_s4h:step_{step:05d}") for step in steps}
    st4, _ = run_status("lc_s0_dwt_fixed_s4h")
    if st4 != "planned" and "S4H" in N:
        hf = {step: by_model.get(f"dwt_fixed@lc_s0_dwt_fixed:step_{step:05d}") for step in steps}
        dc = {step: by_model.get(f"dct_fixed@lc_s0_dct_fixed:step_{step:05d}") for step in steps}
        seq = lambda d: ", ".join(f"{step}: {r['successes']}/20" for step, r in d.items() if r)  # noqa: E731
        if s4[2000]:
            r4, rh = s4[2000], hf[2000]
            res = (f"Success, seed 0: sym4→haar {seq(s4)}. Haar fixed {seq(hf)}. DCT without BPE {seq(dc)}. "
                   f"Tokens per chunk {r4['tokens_mean']:.0f} vs Haar {rh['tokens_mean']:.0f}; latency {r4['latency_ms_mean']:.0f} vs {rh['latency_ms_mean']:.0f} ms; no malformed chunks.")
            short4 = f"{r4['successes']}/20 @2k vs Haar {rh['successes']}/20, {r4['tokens_mean']:.0f} tokens"
            st4 = "done"
        else:
            res, short4, st4 = "Training or evaluating.", "running", "running"
        node("S4HT", "S4H", "sym4→haar DWT trained (fixed, no BPE)", st4,
             "The offline check said sym4→haar loses at 10-step chunks, but offline token counts do not always predict how well a policy "
             "learns. Train it once, exactly like Haar fixed, to measure it instead of assuming.",
             "Does the mixed sym4→haar wavelet train into a better policy than Haar at 10-step chunks?",
             bpe_setup("Whole-chunk sym4→haar DWT (sym4 at the finest level, Haar at the coarser levels), symmetric boundary mode, full-depth fixed plan, same "
                       "quantizer, vocabulary and grammar-constrained decoding as Haar; no BPE. 100 tokens per chunk (token cap 117 instead of 96).",
                       "lc_s0_dwt_fixed (Haar, 69 tokens) and lc_s0_dct_fixed (DCT, 64 tokens), same seed and recipe."),
             res, "Close to Haar at 2,000 steps but slower and less steady to learn, with 45% more tokens per chunk and slower generation. "
             "Its offline advantage is at 32-step chunks (Stage E). One seed.", short=short4, link="training_report.html")

    gen = [r for r in (jload(RES / "eval.json") or {"rows": []})["rows"]
           if r["task"] != "libero_object#0" and re.match(r"^\w+@lc_s0_\w+:step_02000$", r["model"])]
    if gen:
        gl = {"dwt_fixed": "Haar fixed", "dwt_fixed_s4h": "sym4→haar fixed", "dct_fixed": "DCT no BPE", "dwt_adaptive": "Haar adaptive",
              "dwt_fixed_lw": "Haar fixed, weighted loss", "dwt_fixed_slw": "Haar fixed, soft + weighted loss"}
        tot = {}
        for r in gen:
            v = r["model"].split("@")[0]
            k, n = tot.get(v, (0, 0))
            tot[v] = (k + r["successes"], n + r["episodes"])
        hit = sorted({re.sub(r"^pick up the (.*) and place it in the basket$", r"\1", r["instruction"]) for r in gen if r["successes"]})
        res = ("Success on the 9 new objects (45 episodes each): " + ", ".join(f"{gl.get(v, v)} {k}/{n}" for v, (k, n) in tot.items()) + ". "
               f"Every success is on {' or '.join(hit)}.")
        best = max(tot, key=lambda v: tot[v][0])
        node("GEN", "LC", "Zero-shot generalization · 9 new objects", "done",
             "Every policy was trained on one instruction only, so its training-task score says nothing about whether the token format "
             "helps it handle new instructions. Probe that cheaply with the existing checkpoints before considering multi-task training.",
             "Do the single-task policies transfer to new objects in the same scene, and does the token format matter there?",
             [("Models", "The seed-0, step-2000 checkpoints of Haar fixed, Haar adaptive, DCT without BPE and sym4→haar fixed, "
              "later also Haar fixed with the weighted and soft + weighted losses. No new training."),
              ("Trained on", OUR_DATA),
              ("Evaluation", "The other nine libero_object tasks (#1–#9: same scene and basket, a different object and instruction), never seen in "
               "training. 5 initial states per task drawn at random (seed 0) from 0–19, the same states for every model.")],
             res, "The successes take about as long as an alphabet-soup success: the policies most likely replay the learned motion and succeed "
             "where the new object happens to be reachable that way, rather than following the instruction (no videos to confirm). "
             "The Haar fixed policies lead and adaptive is lowest, as on the training task, but the loss variants do not carry over: soft labels + "
             "weighting (20/20 on the training task) ties plain cross-entropy here and weighting alone is one episode ahead (both p = 1.0). "
             "Only Haar fixed vs Haar adaptive is significant in a paired test (6 vs 0 episodes won alone, p = 0.03 uncorrected); Haar vs "
             "DCT (p = 0.22) and Haar vs sym4→haar (p = 0.73) are not. Seeds 1–2 were not run. "
             "Real generalization needs multi-task training.",
             short=f"≈0%: best {gl.get(best, best)} {tot[best][0]}/{tot[best][1]}", link="training_report.html")
    def loss_run(variant):
        st, s = run_status(f"lc_s0_{variant}")
        if st == "planned":
            return st, "Queued.", "queued"
        seeds = [k for k in range(3) if by_model.get(f"{variant}@lc_s{k}_{variant}:step_02000")]
        if not seeds:
            res = f"Training: step {s.get('step', 0)} / {s.get('stop_at', 2000)}." if st == "running" and s else "Training done; evaluation pending."
            return ("failed" if st == "failed" else "running"), res, "running"
        pool = lambda v: {step: tuple(map(sum, zip(*[(r["successes"], r["episodes"]) for k in seeds  # noqa: E731
                                                     if (r := by_model.get(f"{v}@lc_s{k}_{v}:step_{step:05d}"))]))) for step in steps}
        line = lambda d: ", ".join(f"{step}: {k}/{n}" for step, (k, n) in d.items())  # noqa: E731
        ev, hf = pool(variant), pool("dwt_fixed")
        per = ", ".join(f"seed {k} {by_model[f'{variant}@lc_s{k}_{variant}:step_02000']['successes']}/20" for k in seeds)
        res = (f"Success pooled over seed(s) {', '.join(map(str, seeds))}: {line(ev)}. Plain-loss Haar fixed, same seeds: {line(hf)}. "
               f"At 2,000 steps per seed: {per}. No malformed chunks.")
        k, n = ev[2000]
        return "done", res, f"{k}/{n} @2k vs plain {hf[2000][0]}/{hf[2000][1]} ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})"

    loss_setup = lambda loss: [  # noqa: E731
        ("Base model", BASE),
        ("Training data", OUR_DATA),
        ("Training", lora("Seed 0, same recipe as the learning curve (5,000-step schedule stopped at 2,000 steps)")),
        ("Tokens", "Fixed-plan Haar DWT (8+2 segments, 69 tokens per chunk), exactly as lc_s0_dwt_fixed; inference and decoding unchanged."),
        ("Loss", loss),
        ("Compared with", "lc_s0_dwt_fixed: same tokens, seed and recipe, plain one-hot cross-entropy."),
        ("Evaluation", EVAL + " Every checkpoint (250–2,000 steps).")]
    st, res, short = loss_run("dwt_fixed_lw")
    node("LW", "LC", "Loss weighted by coefficient scale (Haar fixed)", st,
         "Every run so far trained with plain cross-entropy, which scores each action token as right or wrong and treats all tokens and all "
         "wrong bins alike. For wavelet tokens that ignores what the token means. The Haar basis is orthonormal, so by Parseval a coefficient "
         "error adds exactly its square to the squared trajectory error, and a one-bin miss costs one quantizer step. Those steps differ by "
         "about 30× across levels: 0.041–0.119 for the coarsest Level-8 coefficient (the chunk's average motion), down to 0.0037–0.0142 for "
         "the finest details, which are half of the 60 coefficient tokens. So the loss spends half its gradient on tokens whose mistakes "
         "barely move the arm, and as much on a trajectory-ruining coarse miss as on an invisible fine one. Our failures fit this: BPE and "
         "FAST lost mostly through sloppy or malformed tokens, and the jerk measurements show output detail matters. Haar fixed is the best "
         "format, so it is the one to improve. Related work: FreqPolicy (arXiv 2506.08822) and FreDF (ICLR 2025) weight losses by frequency, "
         "but on continuous outputs; FreqPolicy-AR (2506.01583), MINT and Wavelet Policy (2507.04331) tokenize or predict frequency/wavelet "
         "bands with plain losses. We found no work weighting the token loss of discrete wavelet-coefficient tokens in a VLA by the error each "
         "token causes.",
         "Does weighting each coefficient token's loss by the trajectory error a miss would cause make Haar fixed learn faster or better?",
         loss_setup("Coefficient-token cross-entropy × w, with w = quantizer step of that coefficient (its segment, level and dimension) divided "
                    "by the mean step over the batch's coefficient tokens. Coarsest coefficients get w ≈ 1.8–5.2, finest details 0.16–0.62. "
                    "Structure and gripper tokens keep weight 1. The logged train/val loss is still the plain cross-entropy, so curves stay comparable."),
         res, "Weighting changes how much each token counts but still rewards only the exact bin; the soft-label child adds credit for near "
         "misses. One seed, a probe: a clear gap either way would justify more seeds.", short=short)
    st, res, short = loss_run("dwt_fixed_slw")
    node("SLW", "LW", "Soft labels + scale weighting (Haar fixed)", st,
         "Weighting fixes how much each token counts, but one-hot cross-entropy still treats a prediction one bin off exactly like one 200 "
         "bins off, although the bins are ordered values and a neighbour bin is a nearly correct answer. Soft labels give partial credit "
         "to bins close in value. Using the quantizer step to measure closeness makes them follow the same Parseval logic as the weights: "
         "a fine-level neighbour is only ≈0.004–0.014 off and should get real credit, while a coarse-level neighbour is 0.04–0.12 off and "
         "should get almost none. Gaussian soft labels over discretized actions are used by ActionMap (arXiv 2606.06904) in a VLA action "
         "head, and distance-aware soft targets for ordered tokens by DIST² (arXiv 2503.02379), HL-Gauss (Farebrother et al. 2024) and "
         "ordinal-regression losses (DORN, squared EMD). None scale the label width by wavelet level. Run on top of weighting so the "
         "difference from the LW node isolates the soft labels.",
         "Do step-aware soft labels add anything on top of scale weighting?",
         loss_setup("As the weighted node, plus a soft target on each coefficient token: q(b) ∝ exp(−((b − b*)·step)² / 2σ²) over its 256 bins, "
                    "σ = 0.007 in normalized action units (about the median finest-level step). A neighbour bin gets 13–87% of the target's "
                    "probability at the finest level, up to 65% one level coarser, at most 20% at the next, and ≈0 at the coarsest. Loss = "
                    "−Σ q log p over the bins; structure and gripper tokens stay one-hot."),
         res, "Seed 0 reached 20/20, the best single run so far, which prompted seeds 1 and 2; over three seeds it ties plain "
         "cross-entropy at 2,000 steps (paired p = 1.0) and learns more slowly in mid-training (p ≈ 0.09 at 1,000 steps). Not a gain. "
         "σ was set from the step table, not tuned. It was still chosen for the full-LIBERO run, where 14,500 steps make the slow start "
         "matter less; plain Haar on full LIBERO is planned after it.", short=short)
    fst, fs = run_status("full_s0_dwt_fixed_slw")
    ftot = {}
    for r in (jload(RES / "eval.json") or {"rows": []})["rows"]:
        if r["model"].startswith("full40_"):
            k, n = ftot.get(r["model"], (0, 0))
            ftot[r["model"]] = (k + r["successes"], n + r["episodes"])
    fpct = lambda m: 100 * ftot[m][0] / 400 if ftot.get(m, (0, 0))[1] == 400 else None  # noqa: E731
    f_haar = "full40_dwt_fixed_slw@full_s0_dwt_fixed_slw:step_{:05d}"
    f_e4 = "full40_{0}@full_e4_{0}:step_58000"
    f_dct = "full40_dct_fixed@full_s0_dct_fixed:step_{:05d}"
    f_ref = fpct("full40_ref_pi0fast_libero")
    ep_steps = ((1, 14500), (2, 29000), (3, 43500), (4, 58000))
    haar_ep = [(e, fpct(f_haar.format(s))) for e, s in ep_steps[:3]]
    e4 = [(lab, fpct(f_e4.format(v))) for v, lab in (("dwt_fixed", "plain cross-entropy"), ("dwt_fixed_lw", "weighted"),
                                                      ("dwt_fixed_slw", "soft + weighted"))]
    if fst != "planned" or ftot:
        got = [p for _, p in haar_ep + e4 if p is not None]
        res = ("Success on the same 400 episodes. Haar DWT fixed: " + ", ".join(f"epoch {e} {p:.1f}%" for e, p in haar_ep if p is not None)
               + ("; epoch 4 from the epoch-3 checkpoint: " + ", ".join(f"{lab} {p:.1f}%" for lab, p in e4 if p is not None)
                  if any(p is not None for _, p in e4) else "")
               + (f". LeRobot π₀-FAST LIBERO: {f_ref:.1f}%." if f_ref is not None else "."))
        short_f = (f"best {max(got):.0f}% vs π₀-FAST {f_ref:.0f}%" if got and f_ref is not None else "evaluating")
        node("FULL", "SLW", "Full LIBERO · 40 tasks, epochs 1–4", "done" if haar_ep[2][1] is not None else ("failed" if fst == "failed" else "running"),
             "Every result so far is on one task, and the single-task policies do not generalize. The benchmark that published π₀-FAST "
             "numbers use is all 40 LIBERO tasks, so train once on the full dataset and compare head-to-head with LeRobot's own π₀-FAST "
             "LIBERO checkpoint on the same episodes.",
             "With the same base model and a much smaller budget, does a DWT-token policy match or beat LeRobot's π₀-FAST on all of LIBERO?",
             [("Base model", BASE),
              ("Training data", "All of lerobot/libero @a1aaacb: 40 tasks in four suites (Spatial, Object, Goal, LIBERO-10), 1,525 training "
               "and 168 validation demos, 232,282 windows (one epoch = 14,500 steps). Codec and normalization refitted on these."),
              ("Training", lora("Haar fixed, soft labels + scale weighting, seed 0, warmup 300. Epoch 1 under a 29,000-step cosine; epochs 2–3 "
                                "resumed under 43,500 steps (lr stepped up from 1.1e-4 to 1.6e-4); epoch 4 branched three times from the "
                                "epoch-3 checkpoint under 58,000 steps (lr 4.6e-5 → 2e-5) with plain cross-entropy, weighting only, and "
                                "soft labels + weighting")),
              ("Compared with", "lerobot/pi0fast-libero (fully fine-tuned by LeRobot on all of LIBERO), evaluated by us unmodified."),
              ("Evaluation", "All four suites, every task, initial states 0–9: 400 episodes per model, the same for every model.")],
             res, "Success rose steeply to epoch 3, then epoch 4 changed nothing for any loss (paired p ≥ 0.3), so this run plateaus at about "
             "three quarters; validation token loss bottomed out around epoch 1 while training loss kept falling. The later plain-Haar run "
             "showed the soft + weighted loss of epochs 1–3 had slowed it down. The ≈13-point gap to π₀-FAST LIBERO is mostly LIBERO-10 "
             "(≈57–61% vs 87%). One seed; 10 episodes per task gives about ±4 points on the overall rate (the standard protocol uses 50).",
             short=short_f, link="training_report.html")
    dst, ds = run_status("full_s0_dct_fixed")
    dct_ep = [(e, fpct(f_dct.format(s))) for e, s in ep_steps]
    hmatch = dict(haar_ep + [(4, e4[0][1])])
    doneep = [(e, p) for e, p in dct_ep if p is not None]
    res = ("Same 400 episodes. " + ", ".join(f"epoch {e}: DCT {p:.1f}% vs Haar {hmatch[e]:.1f}%" if hmatch.get(e) is not None
                                              else f"epoch {e}: DCT {p:.1f}%" for e, p in doneep) + "."
           if doneep else "")
    if dst == "running" and ds:
        res += f" Training: step {ds.get('step', 0)} / {ds.get('stop_at', 0)}."
    node("FDCT", "FULL", "Full LIBERO control · DCT without BPE, 4 epochs", "done" if dct_ep[3][1] is not None else
         (dst if dst in ("failed", "planned") else "running"),
         "The Haar policy plateaus 13 points below π₀-FAST LIBERO. On one task, DCT without BPE (FAST's transform with our quantizer, "
         "vocabulary and grammar) sat between FAST and Haar; training it the same way on all 40 tasks tells whether the wavelet "
         "transform itself helps at scale, separately from dropping BPE.",
         "On all of LIBERO, with the same tokens apart from the transform, does Haar beat DCT at every epoch?",
         [("Base model", BASE), ("Training data", "As the Haar full-LIBERO run."),
          ("Training", lora("DCT without BPE (dct_fixed), plain cross-entropy, seed 0, warmup 300, exactly the Haar run's learning-rate "
                            "history (29,000-step cosine for epoch 1, 43,500 for epochs 2–3, 58,000 for epoch 4)")),
          ("Compared with", "The Haar run at the same epoch (epochs 1–3 soft labels + weighting, epoch 4 plain cross-entropy), and "
           "π₀-FAST LIBERO."),
          ("Evaluation", "400 episodes after every epoch, the same as the Haar run.")],
         res or "Queued after the Haar epoch-2 evaluation.", "DCT leads at every epoch. Epochs 1–3 of the Haar run used soft labels + "
         "weighting while DCT uses plain cross-entropy, so part of that lead was the loss; the plain-Haar control isolates the transform "
         "(DCT still ahead at epochs 1 and 2).",
         short=(f"DCT {doneep[-1][1]:.0f}% @ epoch {doneep[-1][0]}" if doneep else "queued" if dst == "planned" else "running"), link="training_report.html")
    f_plain = "full40_dwt_fixed@full_s0_dwt_fixed:step_{:05d}"
    pst, ps_ = run_status("full_s0_dwt_fixed")
    plain_ep = [(e, fpct(f_plain.format(s))) for e, s in ep_steps[:2]]
    pdone = [(e, p) for e, p in plain_ep if p is not None]
    pres = ("Same 400 episodes. " + ", ".join(
        f"epoch {e}: plain Haar {p:.1f}% vs soft + weighted Haar {dict(haar_ep)[e]:.1f}%"
        + (f" and DCT {dict(dct_ep)[e]:.1f}%" if dict(dct_ep).get(e) is not None else "") for e, p in pdone) + ".") if pdone else ""
    if pst == "running" and ps_:
        pres += f" Training: step {ps_.get('step', 0)} / {ps_.get('stop_at', 0)}."
    node("FPLAIN", "FULL", "Full LIBERO loss control · plain Haar from scratch, 2 epochs",
         "done" if plain_ep[1][1] is not None else (pst if pst in ("failed", "planned") else "running"),
         "After one epoch DCT without BPE (plain cross-entropy) led the Haar run (soft labels + weighting) by 24 points, the reverse of the "
         "single-task result. The Haar run differs in both transform and loss, and soft labels + weighting learned more slowly in "
         "mid-training on one task, so train Haar with plain cross-entropy from scratch to separate the two.",
         "Is DCT's early lead the transform, or did the soft + weighted loss slow Haar down?",
         [("Base model", BASE), ("Training data", "As the Haar full-LIBERO run."),
          ("Training", lora("Haar fixed (dwt_fixed), plain cross-entropy, seed 0, warmup 300, the same learning-rate history for "
                            "two epochs (29,000-step cosine for epoch 1, resumed under 43,500 steps for epoch 2)")),
          ("Compared with", "The soft + weighted Haar run and the DCT control at the same epoch."),
          ("Evaluation", "400 episodes after each epoch, the same as the other full runs.")],
         pres or "Queued after DCT epoch 2.", "Both effects are real. Plain Haar beats soft + weighted Haar (69.0% → 79.2% at epoch 2, "
         "paired p ≈ 1e-4), so the modified loss slowed Haar down; DCT still beats plain Haar at the same epoch (77.0% vs 58.5% at epoch 1, "
         "85.5% vs 79.2% at epoch 2, p ≈ 0.006), so the transform matters too, and more."
         if plain_ep[1][1] is not None else "If plain Haar matches DCT, the loss slowed the first run down; if it stays near the "
         "soft + weighted run, DCT's transform is genuinely better at this scale.",
         short=(f"plain Haar {pdone[-1][1]:.0f}% @ epoch {pdone[-1][0]}" if pdone else "queued" if pst == "planned" else "running"),
         link="training_report.html")

    def one(model, run):
        p = fpct(model)
        st_, s_ = run_status(run)
        status = "done" if p is not None else (st_ if st_ in ("failed", "planned") else "running")
        extra = f" Training: step {s_.get('step', 0)} / {s_.get('stop_at', 0)}." if st_ == "running" and s_ else ""
        return p, status, extra

    sqrtw_setup = ("Coefficient-token loss × quantizer step<sup>0.5</sup>, normalized to mean 1 per batch (DCT weights 0.56–2.8, against "
                   "0.25–6.4 for full step weighting); structure and gripper tokens weight 1; no soft labels")
    pw, stw, xw = one("full40_dct_fixed@full_e3_dct_fixed_sqrtw:step_43500", "full_e3_dct_fixed_sqrtw")
    d2, d3 = fpct(f_dct.format(29000)), fpct(f_dct.format(43500))
    node("FDCTW", "FDCT", "DCT + √step weighting · 1 epoch from DCT epoch 2", stw,
         "Full step weighting did not help Haar, but on DCT the step sizes put most of the weight on the few low-frequency tokens that "
         "carry each motion's fingerprint (transform fingerprint study). A gentler √step weighting keeps the fine tokens at more than half "
         "weight. Branching from the plain epoch-2 checkpoint gives a matched comparison with plain DCT epoch 3.",
         "Does a gentle step weighting improve DCT beyond what one more plain epoch gives?",
         [("Base model", BASE), ("Training data", "As the Haar full-LIBERO run."),
          ("Training", lora("DCT without BPE, " + sqrtw_setup + ". Adapters + optimizer from plain DCT step 29,000, trained to step 43,500 "
                            "under the 43,500-step cosine, exactly like plain DCT epoch 3")),
          ("Compared with", "Plain DCT epoch 2 (its start) and epoch 3 (same start and schedule, plain loss); π₀-FAST LIBERO."),
          ("Evaluation", "400 episodes, the same as every full run.")],
         (f"Same 400 episodes: {pw:.1f}%, against plain DCT epoch 2 {d2:.1f}%" + (f" and epoch 3 {d3:.1f}%" if d3 is not None else "")
          + (f"; π₀-FAST LIBERO {f_ref:.1f}%" if f_ref is not None else "") + "." if pw is not None else "Queued." + xw),
         ("The gain over epoch 2 was the extra epoch, not the weighting: plain DCT epoch 3 gained the same, LIBERO-10 included "
          "(66% → 79% plain vs 80% weighted), and the two tie (paired p = 0.68). Both are level with π₀-FAST LIBERO."
          if d3 is not None else
          "The gain over epoch 2 is almost all LIBERO-10 (66% → 80%, paired p = 0.02); overall it is level with π₀-FAST LIBERO (p = 0.53). "
          "Plain DCT epoch 3 decides whether that is the weighting or the extra epoch.") if pw is not None else
         "Compared against plain DCT epoch 3 once both are in.",
         short=(f"{pw:.0f}% @ epoch 3" if pw is not None else stw), link="training_report.html")
    pb, stb, xb = one("full40_dct_fixed@full_e4_dct_fixed_bw5:step_58000", "full_e4_dct_fixed_bw5")
    d4 = fpct(f_dct.format(58000))
    node("FDCTB", "FDCTW", "DCT + 5-level stepped weights · 1 epoch from DCT epoch 3", stb,
         "√step weighting gave nothing beyond the extra epoch. Plain DCT epoch 3 matches π₀-FAST LIBERO overall but still trails on "
         "LIBERO-10. A coarser, fixed importance scale tests whether emphasizing the low-frequency DCT tokens in a few clear tiers, "
         "with no averaging to 1, helps the final epoch.",
         "Do fixed importance tiers on DCT tokens beat one more plain epoch from epoch 3?",
         [("Base model", BASE), ("Training data", "As the Haar full-LIBERO run."),
          ("Training", lora("DCT without BPE. The codec's quantizer-step range (0.007–0.179) is cut into 5 equal log-scale segments; "
                            "coefficient tokens in them weigh 0.3, 0.6, 0.9, 1.2, 1.5 from the finest to the coarsest (24, 18, 8, 7 "
                            "and 3 of the 60 coefficient tokens per chunk; DC of the rotation dims gets 1.5, the 6 finest bands of "
                            "the position dims 0.3). Not normalized, structure and gripper tokens 1, no soft labels. Adapters + "
                            "optimizer from plain DCT step 43,500, trained to 58,000 under the 58,000-step cosine, like plain DCT epoch 4")),
          ("Compared with", "Plain DCT epoch 3 (its start) and epoch 4 (same start and schedule, plain loss); π₀-FAST LIBERO."),
          ("Evaluation", "400 episodes.")],
         (f"Same 400 episodes: {pb:.1f}%, against plain DCT epoch 3 {d3:.1f}%" + (f" and epoch 4 {d4:.1f}%" if d4 is not None else "")
          + (f"; π₀-FAST LIBERO {f_ref:.1f}%" if f_ref is not None else "") + "." if pb is not None else
          "Queued after √step-weighted DCT from scratch." + xb),
         ("Ties its starting point (paired p = 1.0) and π₀-FAST LIBERO (p = 1.0). LIBERO-10 reached 83%, the best of any of our models "
          "(plain DCT epoch 3: 79%, π₀-FAST LIBERO: 87%), but the +4 points are not significant (p = 0.56). "
          + ("Plain DCT epoch 4 (same start and schedule, plain loss) separates the weighting from the extra epoch."
             if d4 is None else "See the training report for the paired test against plain DCT epoch 4."))
         if pb is not None else "Pending.",
         short=(f"{pb:.0f}% @ epoch 4" if pb is not None else stb), link="training_report.html")
    ps0, sts0, xs0 = one("full40_dct_fixed@full_s0_dct_fixed_sqrtw:step_14500", "full_s0_dct_fixed_sqrtw")
    d1 = fpct(f_dct.format(14500))
    ps2 = fpct("full40_dct_fixed@full_s0_dct_fixed_sqrtw:step_29000")
    s2st = run_status("full_s0_dct_fixed_sqrtw")[1] or {}
    e2_running = ps0 is not None and ps2 is None and s2st.get("step", 0) > 14500
    node("FDCTWS", "FDCTW", "DCT + √step weighting · from scratch, 2 epochs",
         "done" if ps2 is not None else "running" if ps0 is None or e2_running else "planned",
         "The branch from epoch 2 only shows the weighting as a late fine-tune. On Haar the modified loss hurt most early in training, so "
         "train the same weighting from scratch and compare with plain DCT epoch by epoch. Epoch 1 came out slightly ahead of plain DCT "
         "(not significant, LIBERO-10 +10 points), so a second epoch was added.",
         "Does √step weighting help or hurt DCT when used from the start?",
         [("Base model", BASE), ("Training data", "As the Haar full-LIBERO run."),
          ("Training", lora("DCT without BPE, " + sqrtw_setup + ". From scratch with plain DCT's learning-rate history: epoch 1 under "
                            "the 29,000-step cosine, epoch 2 resumed under 43,500 steps")),
          ("Compared with", "Plain DCT at the same epoch."), ("Evaluation", "400 episodes after each epoch.")],
         (f"Same 400 episodes: epoch 1 {ps0:.1f}% against plain DCT {d1:.1f}%"
          + (f"; epoch 2 {ps2:.1f}% against plain DCT {fpct(f_dct.format(29000)):.1f}%" if ps2 is not None else
             f"; epoch 2 training, step {s2st.get('step', 0)} / 29000" if e2_running else "; epoch 2 queued after the 5-level branch")
          + "." if ps0 is not None else "Training after plain DCT epoch 3." + xs0),
         "See the training report for the paired tests." if ps0 is not None else "Pending.",
         short=(f"{ps2:.0f}% @ epoch 2" if ps2 is not None else f"{ps0:.0f}% @ epoch 1" if ps0 is not None else sts0),
         link="training_report.html")
    pp, stp, xp = one("full40_poly_fixed@full_s0_poly_fixed_sqrtw:step_14500", "full_s0_poly_fixed_sqrtw")
    pp2 = fpct("full40_poly_fixed@full_s0_poly_fixed_sqrtw:step_29000")
    p2st = run_status("full_s0_poly_fixed_sqrtw")[1] or {}
    p2_running = pp is not None and pp2 is None and p2st.get("step", 0) > 14500
    ps2w = fpct("full40_dct_fixed@full_s0_dct_fixed_sqrtw:step_29000")
    node("FPOLY", "FDCTWS", "Orthogonal polynomials + √step weighting · from scratch, 2 epochs",
         "done" if pp2 is not None else "running" if p2_running else stp if pp is None else "planned",
         "In the synthetic studies orthogonal polynomials ranked first of 18 transforms: a smooth edit of a motion moves the fewest tokens "
         "(22% vs 30% for DCT and 69% for Haar), and different edits land on different tokens. Offline on LIBERO chunks the polynomial "
         "codec is as accurate as DCT at the same 64 tokens (MSE 1.64e-4 vs 1.65e-4).",
         "Does the best basis in the synthetic studies beat DCT as a VLA action tokenizer?",
         [("Base model", BASE), ("Training data", "As the Haar full-LIBERO run."),
          ("Training", lora("poly_fixed: Legendre polynomials orthonormalized on the 10 sample points, grouped by degree (0, 1, 2–3, 4–9) "
                            "like the DCT bands, same quantizer, vocabulary and grammar, no BPE; " + sqrtw_setup + ". From scratch with "
                            "DCT's learning-rate history: epoch 1 under the 29,000-step cosine, epoch 2 resumed under 43,500 steps")),
          ("Compared with", "√step-weighted DCT from scratch (same loss) and plain DCT at the same epoch."),
          ("Evaluation", "400 episodes after each epoch.")],
         (f"Same 400 episodes: epoch 1 {pp:.1f}%" + (f" against √step-weighted DCT {ps0:.1f}%" if ps0 is not None else "")
          + f" and plain DCT {d1:.1f}%"
          + (f"; epoch 2 {pp2:.1f}%" + (f" against √step-weighted DCT {ps2w:.1f}%" if ps2w is not None else "")
             + f" and plain DCT {fpct(f_dct.format(29000)):.1f}%" if pp2 is not None else
             f"; epoch 2 training, step {p2st.get('step', 0)} / 29000" if p2_running else "; epoch 2 queued")
          + "." if pp is not None else "Queued after epoch 2 of √step-weighted DCT." + xp),
         "See the training report for the paired tests." if pp is not None else "Pending.",
         short=(f"{pp2:.0f}% @ epoch 2" if pp2 is not None else f"{pp:.0f}% @ epoch 1" if pp is not None else stp),
         link="training_report.html")
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

"""Read-only training dashboard for experiment/adaptive_dwt_vla (progress, loss curves, results)."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from .devtools import PROJECT_ROOT

EXPERIMENT_ROOT = PROJECT_ROOT / "experiment" / "adaptive_dwt_vla"
RUNS = EXPERIMENT_ROOT / "runs"
RESULTS = EXPERIMENT_ROOT / "results"
RESULT_TYPES = {".png": "image/png", ".json": "application/json", ".csv": "text/csv; charset=utf-8",
                ".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                ".mp4": "video/mp4", ".gif": "image/gif"}
STALE_AFTER_S = 180.0


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def list_runs() -> dict:
    runs = []
    if RUNS.is_dir():
        for d in sorted(RUNS.iterdir()):
            if not d.is_dir() or not (d / "status.json").exists():
                continue
            st = _read_json(d / "status.json")
            cfg = _read_json(d / "config.json")
            if st.get("state") == "running" and time.time() - float(st.get("updated", 0)) > STALE_AFTER_S:
                st["state"] = "stalled"
            ck = d / "ckpt"
            ckpts = [{"name": c.name, "mb": round(_dir_size(c) / 1e6, 1)} for c in sorted(ck.iterdir())
                     if c.is_dir() and not c.name.endswith(".tmp")] if ck.is_dir() else []
            args = cfg.get("args", {})
            runs.append({
                "name": d.name, **st,
                "tasks": args.get("tasks"), "instructions": cfg.get("instructions"),
                "effective_batch": cfg.get("effective_batch"), "micro_batch": args.get("micro_batch"),
                "lr": args.get("lr"), "lora_rank": args.get("lora_rank"), "horizon": args.get("horizon"),
                "overfit": args.get("overfit"), "n_train_windows": cfg.get("n_train_windows"),
                "trainable_params": (cfg.get("counts") or {}).get("trainable_params"),
                "checkpoints": ckpts, "run_mb": round(_dir_size(d) / 1e6, 1),
            })
    usage = shutil.disk_usage(EXPERIMENT_ROOT if EXPERIMENT_ROOT.exists() else PROJECT_ROOT)
    return {"runs": runs, "disk_free_gb": round(usage.free / 1e9, 1), "disk_total_gb": round(usage.total / 1e9, 1),
            "now": time.time()}


def run_metrics(name: str, max_points: int = 1500) -> Optional[dict]:
    d = RUNS / name
    if "/" in name or ".." in name or not (d / "metrics.jsonl").exists():
        return None
    train, val = [], []
    with open(d / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            (val if rec.get("kind") == "val" else train).append(rec)
    if len(train) > max_points:
        stride = len(train) / max_points
        train = [train[int(i * stride)] for i in range(max_points)] + [train[-1]]
    return {"name": name, "train": train, "val": val}


def results_summary() -> dict:
    out: dict[str, Any] = {"codec": [], "files": [], "eval": None, "summary_md": None}
    if not RESULTS.is_dir():
        return out
    for p in sorted(RESULTS.glob("codec_choice_H*.json")):
        c = _read_json(p)
        if not c:
            continue
        pick = lambda r: {k: r.get(k) for k in ("tokens_mean", "tokens_p95", "mse", "max_abs", "vel_mse",  # noqa: E731
                                               "grip_exact_rate", "unigram_entropy_bits_per_chunk", "n_windows")}
        out["codec"].append({
            "horizon": c.get("horizon"),
            "rows": [
                {"method": "FAST (LIBERO BPE)", **pick(c.get("fast_libero_val", {}))},
                {"method": "FAST (universal BPE)", **pick(c.get("fast_universal_val", {}))},
                {"method": f"adaptive DWT oracle (γ={c['adaptive']['weights']['gamma']})", **pick(c["adaptive"]["val"])},
                {"method": f"fixed DWT {c['fixed']['plan']}", **pick(c["fixed"]["val"])},
            ],
        })
    out["files"] = [p.name for p in sorted(RESULTS.iterdir()) if p.suffix in RESULT_TYPES]
    if (RESULTS / "eval.json").exists():
        out["eval"] = _read_json(RESULTS / "eval.json")
    if (RESULTS / "summary.md").exists():
        out["summary_md"] = (RESULTS / "summary.md").read_text(encoding="utf-8")
    return out


def result_file(name: str) -> Optional[tuple[Path, str]]:
    path = (RESULTS / name).resolve()
    if path.parent != RESULTS.resolve() or path.suffix not in RESULT_TYPES or not path.is_file():
        return None
    return path, RESULT_TYPES[path.suffix]


EXPERIMENTS_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Experiments · Smooth Operator</title>
  <style>
    __STYLE__
    main { max-width: 1500px; margin: 0 auto; padding: 18px 20px 40px; }
    h2 { font-size: 15px; font-weight: 500; margin: 22px 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
    .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 12px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
    .run { cursor: pointer; transition: box-shadow .15s, border-color .15s; }
    .run:hover { box-shadow: 0 2px 10px rgba(0,0,0,.07); }
    .run.selected { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
    .run-top { display: flex; align-items: center; gap: 8px; }
    .swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; }
    .run-name { font-weight: 500; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .badge { margin-left: auto; font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500; }
    .b-running { background: #e6f4ea; color: #137333; } .b-done { background: var(--accent-soft); color: var(--accent); }
    .b-failed, .b-stalled { background: #fce8e6; color: var(--danger); } .b-starting, .b-paused { background: #fef7e0; color: #b06000; }
    .bar { height: 6px; background: #eef1f4; border-radius: 3px; margin: 10px 0 8px; overflow: hidden; }
    .bar > div { height: 100%; background: var(--accent); border-radius: 3px; transition: width .4s; }
    .kv { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px 10px; font-size: 12px; }
    .kv b { display: block; font-weight: 500; font-size: 13px; color: var(--text); }
    .kv span { color: var(--muted); }
    .err { color: var(--danger); font-size: 12px; margin-top: 6px; word-break: break-word; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 1000px) { .charts { grid-template-columns: 1fr; } }
    .chart-head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
    .chart-head h3 { font-size: 14px; font-weight: 500; margin: 0; }
    .chart-head label { margin-left: auto; font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 4px; }
    svg.chart { width: 100%; height: 280px; display: block; }
    svg.chart text { font-size: 11px; fill: #5f6368; font-family: inherit; }
    .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--muted); margin-top: 4px; }
    .legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
    .legend i.dash { background: repeating-linear-gradient(90deg, currentColor 0 4px, transparent 4px 7px) !important; height: 2px; }
    table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
    th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid #eef1f4; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 500; }
    .imgs { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
    .imgs img { width: 100%; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    .tip { position: fixed; pointer-events: none; background: #202124; color: #fff; font-size: 11.5px; padding: 5px 8px;
           border-radius: 6px; white-space: pre; z-index: 9; display: none; }
    .topline { display: flex; align-items: baseline; gap: 14px; font-size: 13px; }
    pre.md { white-space: pre-wrap; font-size: 12.5px; margin: 0; }
  </style>
</head>
<body>
  <header>
    <h1>Experiments · adaptive-DWT action tokens</h1>
    <span class="muted" id="updated" style="font-size:12.5px"></span>
    __NAV__
  </header>
  <main>
    <div class="topline"><span id="disk" class="muted"></span><span class="muted">Auto-refresh every 10 s · click a run to inspect its per-token-class curves</span></div>
    <h2>Runs</h2>
    <div class="grid" id="runs"><div class="muted">No runs yet.</div></div>
    <h2>Loss curves</h2>
    <div class="charts">
      <div class="card"><div class="chart-head"><h3>Training loss (all runs)</h3><label><input type="checkbox" id="logy" checked /> log y</label></div>
        <svg class="chart" id="c-train"></svg><div class="legend" id="l-train"></div></div>
      <div class="card"><div class="chart-head"><h3>Validation loss (all runs)</h3></div>
        <svg class="chart" id="c-val"></svg><div class="legend" id="l-val"></div></div>
      <div class="card"><div class="chart-head"><h3 id="t-cls-acc">Token accuracy by class</h3></div>
        <svg class="chart" id="c-acc"></svg><div class="legend" id="l-acc"></div></div>
      <div class="card"><div class="chart-head"><h3 id="t-cls-loss">Loss by token class</h3></div>
        <svg class="chart" id="c-cls"></svg><div class="legend" id="l-cls"></div></div>
      <div class="card"><div class="chart-head"><h3>Seconds per optimizer step</h3></div>
        <svg class="chart" id="c-speed"></svg><div class="legend" id="l-speed"></div></div>
      <div class="card"><div class="chart-head"><h3>Peak GPU memory reserved (GB)</h3></div>
        <svg class="chart" id="c-vram"></svg><div class="legend" id="l-vram"></div></div>
    </div>
    <h2>Closed-loop evaluation</h2>
    <div class="card" id="eval"><span class="muted">No rollout results yet (results/eval.json).</span></div>
    <h2>Offline codec (Stage A, validation episodes)</h2>
    <div id="codec"></div>
    <div class="imgs" id="imgs" style="margin-top:12px"></div>
    <h2>Summary</h2>
    <div class="card" id="summary"><span class="muted">results/summary.md not written yet.</span></div>
  </main>
  <div class="tip" id="tip"></div>
  <script>
    const PALETTE = ["#1a73e8", "#e8710a", "#188038", "#d93025", "#9334e6", "#12b5cb", "#f9ab00", "#5f6368"];
    const CLASS_COLORS = { all: "#202124", len: "#1a73e8", res: "#9334e6", coef: "#188038", grip: "#e8710a", eos: "#d93025", fast: "#12b5cb", prefix: "#9aa0a6" };
    let runs = [], metrics = {}, selected = null, colorOf = {};
    const $ = (id) => document.getElementById(id);
    const fmt = (v, d = 3) => v == null || Number.isNaN(v) ? "–" : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(d);
    const dur = (s) => s == null ? "–" : s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(0)} min` : `${(s / 3600).toFixed(1)} h`;
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

    async function getJSON(url) { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error(r.status); return r.json(); }

    function niceTicks(lo, hi, n = 5) {
      if (!(hi > lo)) { hi = lo + 1; }
      const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
      const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
      return out;
    }

    function lineChart(svg, series, opts = {}) {
      const W = svg.clientWidth || 600, H = svg.clientHeight || 280, m = { l: 52, r: 14, t: 10, b: 30 };
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const pts = series.flatMap((s) => s.points).filter((p) => p[1] != null && Number.isFinite(p[1]) && (!opts.logY || p[1] > 0));
      if (!pts.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">no data yet</text>`; return; }
      let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
      if (x1 === x0) x1 = x0 + 1;
      const tf = opts.logY ? (v) => Math.log10(v) : (v) => v;
      let y0 = Math.min(...pts.map((p) => tf(p[1]))), y1 = Math.max(...pts.map((p) => tf(p[1])));
      if (opts.yMin != null) y0 = Math.min(y0, opts.yMin); if (opts.yMax != null) y1 = Math.max(y1, opts.yMax);
      if (y1 === y0) { y1 += 0.5; y0 -= 0.5; }
      const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
      const X = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), Y = (y) => H - m.b - (tf(y) - y0) / (y1 - y0) * (H - m.t - m.b);
      let g = "";
      let yt;
      if (opts.logY) {
        const mult = y1 - y0 < 2.5 ? [1, 2, 5] : [1];
        yt = [];
        for (let e = Math.floor(y0); e <= Math.ceil(y1); e++) for (const k of mult) yt.push(k * Math.pow(10, e));
      } else yt = niceTicks(y0, y1, 5);
      const logLabel = (v) => v >= 1 && v < 1e4 ? String(+v.toPrecision(3)) : v.toExponential(0);
      for (const v of yt) { const y = Y(v); if (y < m.t - 1 || y > H - m.b + 1) continue;
        g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="#eef1f4"/><text x="${m.l - 6}" y="${y + 4}" text-anchor="end">${opts.logY ? logLabel(v) : fmt(v, 2)}</text>`; }
      for (const v of niceTicks(x0, x1, 6)) { const x = X(v); if (x < m.l - 1 || x > W - m.r + 1) continue;
        g += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H - m.b}" stroke="#f3f4f6"/><text x="${x}" y="${H - m.b + 16}" text-anchor="middle">${v >= 1000 ? (v / 1000) + "k" : v}</text>`; }
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="#bdc1c6"/>`;
      for (const s of series) {
        const p = s.points.filter((q) => q[1] != null && Number.isFinite(q[1]) && (!opts.logY || q[1] > 0));
        if (!p.length) continue;
        const d = p.map((q, i) => `${i ? "L" : "M"}${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join("");
        g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.8}" ${s.dash ? 'stroke-dasharray="5 3"' : ""} stroke-linejoin="round"/>`;
        if (s.dots || p.length < 3) g += p.map((q) => `<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="2.6" fill="${s.color}"/>`).join("");
      }
      g += `<line id="hx" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="#9aa0a6" stroke-dasharray="3 3" visibility="hidden"/>`;
      g += `<text x="${W - m.r}" y="${H - 4}" text-anchor="end">step</text>`;
      svg.innerHTML = g;
      svg.onmousemove = (ev) => {
        const r = svg.getBoundingClientRect(), xs = (ev.clientX - r.left) * W / r.width;
        if (xs < m.l || xs > W - m.r) { hide(); return; }
        const xv = x0 + (xs - m.l) / (W - m.l - m.r) * (x1 - x0);
        const lines = [];
        let sx = null;
        for (const s of series) {
          let best = null; for (const q of s.points) if (q[1] != null && (!best || Math.abs(q[0] - xv) < Math.abs(best[0] - xv))) best = q;
          if (best) { lines.push(`${s.name}: ${fmt(best[1], 4)}  @${best[0]}`); sx = sx ?? best[0]; }
        }
        const hx = svg.querySelector("#hx"); hx.setAttribute("x1", X(sx)); hx.setAttribute("x2", X(sx)); hx.setAttribute("visibility", "visible");
        const tip = $("tip"); tip.textContent = lines.join("\n"); tip.style.display = "block";
        tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY + 10) + "px";
      };
      const hide = () => { $("tip").style.display = "none"; const hx = svg.querySelector("#hx"); if (hx) hx.setAttribute("visibility", "hidden"); };
      svg.onmouseleave = hide;
    }

    function legend(el, series) {
      el.innerHTML = series.map((s) => `<span><i class="${s.dash ? "dash" : ""}" style="background:${s.color};color:${s.color}"></i>${esc(s.name)}</span>`).join("");
    }

    function renderRuns() {
      const el = $("runs");
      if (!runs.length) { el.innerHTML = '<div class="muted">No runs yet.</div>'; return; }
      el.innerHTML = runs.map((r) => {
        const target = r.stop_at || r.total_steps;
        const pct = target ? Math.min(100, 100 * (r.step || 0) / target) : 0;
        const ck = (r.checkpoints || []).map((c) => `${c.name} ${c.mb} MB`).join(" · ") || "none";
        return `<div class="card run ${r.name === selected ? "selected" : ""}" data-name="${esc(r.name)}">
          <div class="run-top"><span class="swatch" style="background:${colorOf[r.name]}"></span><span class="run-name" title="${esc(r.name)}">${esc(r.name)}</span>
            <span class="badge b-${esc(r.state || "starting")}">${esc(r.state || "?")}</span></div>
          <div class="muted" style="font-size:12px;margin-top:4px">${esc(r.variant || "")} · H=${r.horizon ?? "?"} · tasks ${esc(JSON.stringify(r.tasks || []))}${r.overfit ? ` · overfit ${r.overfit} windows` : ""}</div>
          <div class="bar"><div style="width:${pct}%"></div></div>
          <div class="kv">
            <div><span>step</span><b>${r.step ?? 0} / ${target ?? "?"}${r.stop_at && r.stop_at !== r.total_steps ? ` <span class="muted">of ${r.total_steps}</span>` : ""}</b></div>
            <div><span>ETA</span><b>${r.state === "running" ? dur(r.eta_s) : r.state === "done" ? "done" : r.state === "paused" ? "paused" : "–"}</b></div>
            <div><span>s / step</span><b>${fmt(r.sec_per_step, 2)}</b></div>
            <div><span>train loss</span><b>${fmt(r.loss)}</b></div>
            <div><span>val loss</span><b>${fmt(r.val_loss)}</b></div>
            <div><span>VRAM</span><b>${fmt(r.vram_reserved_gb, 1)} GB</b></div>
            <div><span>eff. batch</span><b>${r.effective_batch ?? "–"}</b></div>
            <div><span>GPU-hours</span><b>${fmt(r.gpu_hours, 2)}</b></div>
            <div><span>trainable</span><b>${r.trainable_params ? (r.trainable_params / 1e6).toFixed(1) + "M" : "–"}</b></div>
          </div>
          <div class="muted" style="font-size:11.5px;margin-top:8px">checkpoints: ${esc(ck)} · run dir ${r.run_mb} MB</div>
          ${r.error ? `<div class="err">${esc(r.error)}</div>` : ""}
        </div>`;
      }).join("");
      el.querySelectorAll(".run").forEach((c) => c.onclick = () => { selected = c.dataset.name; renderRuns(); renderCharts(); });
    }

    function renderCharts() {
      const logY = $("logy").checked;
      const names = runs.map((r) => r.name).filter((n) => metrics[n]);
      const trainS = names.map((n) => ({ name: n, color: colorOf[n], points: metrics[n].train.map((r) => [r.step, r.loss]) }));
      lineChart($("c-train"), trainS, { logY }); legend($("l-train"), trainS);
      const valS = names.map((n) => ({ name: n, color: colorOf[n], dots: true, points: metrics[n].val.map((r) => [r.step, r.val_loss]) }));
      lineChart($("c-val"), valS, { logY }); legend($("l-val"), valS);
      const sp = names.map((n) => ({ name: n, color: colorOf[n], points: metrics[n].train.map((r) => [r.step, r.sec_per_step]) }));
      lineChart($("c-speed"), sp, {}); legend($("l-speed"), sp);
      const vr = names.map((n) => ({ name: n, color: colorOf[n], points: metrics[n].train.map((r) => [r.step, r.vram_reserved_gb]) }));
      lineChart($("c-vram"), vr, { yMin: 0 }); legend($("l-vram"), vr);
      const sel = selected && metrics[selected] ? selected : names[names.length - 1];
      if (!sel) return;
      $("t-cls-acc").textContent = `Token accuracy by class · ${sel}`;
      $("t-cls-loss").textContent = `Loss by token class · ${sel}`;
      const m = metrics[sel];
      const classes = Object.keys(CLASS_COLORS).filter((c) => m.train.some((r) => r["acc_" + c] != null) || m.val.some((r) => r["val_acc_" + c] != null));
      const acc = classes.flatMap((c) => [
        { name: `${c} (train)`, color: CLASS_COLORS[c], points: m.train.map((r) => [r.step, r["acc_" + c]]) },
        { name: `${c} (val)`, color: CLASS_COLORS[c], dash: true, dots: true, points: m.val.map((r) => [r.step, r["val_acc_" + c]]) },
      ]).filter((s) => s.points.some((p) => p[1] != null));
      lineChart($("c-acc"), acc, { yMin: 0, yMax: 1 }); legend($("l-acc"), acc);
      const cl = classes.flatMap((c) => [
        { name: `${c} (train)`, color: CLASS_COLORS[c], points: m.train.map((r) => [r.step, r["loss_" + c]]) },
        { name: `${c} (val)`, color: CLASS_COLORS[c], dash: true, dots: true, points: m.val.map((r) => [r.step, r["val_loss_" + c]]) },
      ]).filter((s) => s.points.some((p) => p[1] != null));
      lineChart($("c-cls"), cl, { logY }); legend($("l-cls"), cl);
    }

    function renderResults(res) {
      $("codec").innerHTML = (res.codec || []).map((c) => `<div class="card" style="margin-bottom:12px">
        <div style="font-weight:500;margin-bottom:6px">H = ${c.horizon} · ${c.rows[0].n_windows ?? "?"} disjoint validation windows (measured)</div>
        <table><tr><th>method</th><th>tokens / chunk</th><th>p95</th><th>MSE (norm.)</th><th>max |err|</th><th>velocity MSE</th><th>gripper exact</th><th>entropy bits / chunk</th></tr>
        ${c.rows.map((r) => `<tr><td>${esc(r.method)}</td><td>${fmt(r.tokens_mean, 1)}</td><td>${fmt(r.tokens_p95, 0)}</td><td>${fmt(r.mse)}</td><td>${fmt(r.max_abs)}</td><td>${fmt(r.vel_mse)}</td><td>${fmt(r.grip_exact_rate, 3)}</td><td>${fmt(r.unigram_entropy_bits_per_chunk, 0)}</td></tr>`).join("")}
        </table></div>`).join("") || '<div class="card muted">Stage A has not been run yet.</div>';
      const imgs = (res.files || []).filter((f) => f.endsWith(".png"));
      const cur = [...$("imgs").querySelectorAll("img")].map((i) => i.dataset.name).join("|");
      if (cur !== imgs.join("|")) $("imgs").innerHTML = imgs.map((f) => `<a href="/api/experiments/results/${encodeURIComponent(f)}" target="_blank"><img data-name="${esc(f)}" src="/api/experiments/results/${encodeURIComponent(f)}" alt="${esc(f)}" /></a>`).join("");
      if (res.eval && res.eval.rows) {
        $("eval").innerHTML = `<table><tr><th>model</th><th>task</th><th>success</th><th>95% Wilson CI</th><th>tokens / chunk</th><th>latency mean / p95 (ms)</th><th>invalid streams</th></tr>
          ${res.eval.rows.map((r) => `<tr><td>${esc(r.model)}</td><td>${esc(r.task)}</td><td>${r.successes}/${r.episodes}</td><td>${fmt(r.ci_low, 2)}–${fmt(r.ci_high, 2)}</td><td>${fmt(r.tokens_mean, 1)}</td><td>${fmt(r.latency_ms_mean, 0)} / ${fmt(r.latency_ms_p95, 0)}</td><td>${r.invalid_streams ?? "–"}</td></tr>`).join("")}</table>`;
      }
      if (res.summary_md) $("summary").innerHTML = `<pre class="md">${esc(res.summary_md)}</pre>`;
    }

    async function refresh() {
      try {
        const data = await getJSON("/api/experiments/runs");
        runs = data.runs;
        runs.forEach((r, i) => colorOf[r.name] = PALETTE[i % PALETTE.length]);
        $("disk").textContent = `Disk free: ${data.disk_free_gb} / ${data.disk_total_gb} GB`;
        await Promise.all(runs.map(async (r) => { try { metrics[r.name] = await getJSON(`/api/experiments/metrics?run=${encodeURIComponent(r.name)}`); } catch (e) {} }));
        renderRuns(); renderCharts();
        renderResults(await getJSON("/api/experiments/results"));
        $("updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
      } catch (e) { $("updated").textContent = `refresh failed: ${e}`; }
    }
    $("logy").onchange = renderCharts;
    window.addEventListener("resize", renderCharts);
    refresh(); setInterval(refresh, 10000);
  </script>
</body>
</html>
"""

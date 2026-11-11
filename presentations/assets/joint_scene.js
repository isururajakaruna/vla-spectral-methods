// Arm + per-joint angle charts. `chunks: false` → arm moves alone, then callouts and live
// charts appear; `chunks: true` → charts open already segmented into action chunks;
// `spectrum: true` → each chunk band shows that chunk's DCT spectrum instead of the trace.
const REVEAL_AFTER = 2.2;   // scene seconds of arm-only motion before the callouts appear
const CHUNK_STEPS = 10;     // samples per chunk fed to the DCT (H = 10, as in the LIBERO runs)
const GROW = 0.35;          // seconds for a freshly predicted spectrum to grow in

function dct2(x) {
  const n = x.length;
  return x.map((_, k) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * Math.cos(Math.PI / n * (i + 0.5) * k);
    return s * Math.sqrt((k ? 2 : 1) / n);
  });
}

function startJointScene({ index, total, prev, next, chunks, spectrum = false }) {
  chunks = chunks || spectrum;
  const slide = document.querySelector('.slide');
  slide.dataset.step = chunks ? 1 : 0;
  const t2 = spectrum
    ? ['The same chunks in the frequency domain',
       'Each chunk’s 10 angle samples → DCT (as in FAST). Frequency goes up, amplitude goes across.']
    : ['Cut the traces into action chunks',
       'A VLA predicts a whole chunk of future actions at once, executes it, then looks again.'];
  slide.querySelector('svg').innerHTML = `
    <g class="on0"><text x="80" y="78" class="h1">Meet the arm: three joints and a gripper</text>
      <text x="80" y="114" class="h2">The policy's job is to decide how every joint moves, at every moment.</text></g>
    <g class="on1"><text x="80" y="78" class="h1">Each joint traces an angle over time</text>
      <text x="80" y="114" class="h2">θ₁, θ₂, θ₃ are what the robot actually executes, step after step.</text></g>
    <g class="on2"><text x="80" y="78" class="h1">${t2[0]}</text>
      <text x="80" y="114" class="h2">${t2[1]}</text></g>
    <g id="arrows" class="b1"></g>
    <g id="arm"></g>
    <g id="labels" class="on0"></g>
    <g id="rings" class="b1"></g>
    <g id="charts" class="b1"></g>
    <g id="chip" class="b2"></g>
    <text x="80" y="695" class="foot">Smooth Operator</text>`;

  const S = { x: 330, y: 520, s: 1.6 };
  const arm = makeArm(document.getElementById('arm'), { ...S, table: [-170, 330] });
  const BOX = { x: 790, w: 430, h: 148, y: [150, 312, 474] };
  const BOX_OF_JOINT = [2, 1, 0];          // base joint at the bottom, wrist on top
  const JNAME = ['shoulder', 'elbow', 'wrist'];
  const N_SAMPLES = 450;

  const arrowsG = document.getElementById('arrows');
  const ringsG = document.getElementById('rings');
  const chartsG = document.getElementById('charts');
  const labelsG = document.getElementById('labels');

  const arrows = [0, 1, 2].map(j => el('path', {
    fill: 'none', stroke: JCOL[j], 'stroke-width': 3.5, class: 'dots', 'marker-end': `url(#ah-${j})`,
  }, arrowsG));

  const rings = [0, 1, 2].map(j => {
    const c = el('circle', { r: 22, fill: 'none', stroke: JCOL[j], 'stroke-width': 2 }, ringsG);
    el('animate', { attributeName: 'r', values: '22;34;22', dur: '1.8s', repeatCount: 'indefinite' }, c);
    el('animate', { attributeName: 'stroke-opacity', values: '.9;0;.9', dur: '1.8s', repeatCount: 'indefinite' }, c);
    return c;
  });

  const labelTxt = ['shoulder', 'elbow', 'wrist', 'gripper'].map(name => {
    const e = el('text', { style: 'font-size:15px;font-weight:500;fill:#5F6368' }, labelsG);
    e.textContent = name;
    return e;
  });

  const samples = [0, 1, 2].map(j => {
    const a = [];
    for (let i = 0; i <= N_SAMPLES; i++) a.push(Q[j](i / N_SAMPLES * T));
    return a;
  });

  const charts = [0, 1, 2].map(j => {
    const by = BOX.y[BOX_OF_JOINT[j]], bx = BOX.x;
    const px0 = bx + 20, px1 = bx + BOX.w - 20, py0 = by + 66, py1 = by + BOX.h - 14;
    const bw = (px1 - px0) / NCH;
    let lo = Math.min(...samples[j]), hi = Math.max(...samples[j]);
    const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    const X = t => px0 + t / T * (px1 - px0);
    const Y = v => py1 - (v - lo) / (hi - lo) * (py1 - py0);
    const pt = (t, v) => `${X(t).toFixed(1)},${Y(v).toFixed(1)}`;
    const P = samples[j].map((v, i) => pt(i / N_SAMPLES * T, v));

    const g = el('g', {}, chartsG);
    el('rect', { x: bx, y: by, width: BOX.w, height: BOX.h, rx: 14, fill: '#fff', filter: 'url(#sh)' }, g);
    el('circle', { cx: bx + 26, cy: by + 27, r: 7, fill: JCOL[j] }, g);
    const title = el('text', { x: bx + 42, y: by + 33, style: 'font-size:17px;font-weight:500' }, g);
    title.textContent = `Joint ${j + 1} · ${JNAME[j]}`;
    const val = el('text', { x: px1, y: by + 33, 'text-anchor': 'end', class: 'mono', style: `font-size:16px;font-weight:500;fill:${JCOL[j]}` }, g);

    const bandG = el('g', {}, g);
    const bands = [], bandLabels = [];
    for (let k = 0; k < NCH; k++) {
      const bg = el('g', { class: 'b2', style: `transition-delay:${0.15 * k}s` }, bandG);
      bands.push(el('rect', { x: px0 + k * bw, y: by + 44, width: bw, height: py1 + 6 - (by + 44), fill: k % 2 ? '#FAFBFC' : '#F1F3F4' }, bg));
      if (k) el('line', { x1: px0 + k * bw, y1: by + 44, x2: px0 + k * bw, y2: py1 + 6, stroke: '#BDC1C6', 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, bg);
      const lt = el('text', { x: px0 + k * bw + 6, y: by + 58, style: 'font-size:11px;fill:#5F6368' }, bg);
      lt.textContent = `chunk ${k + 1}`;
      bandLabels.push(lt);
    }
    if (!spectrum) for (let r = 0; r < 3; r++) {
      const yy = py0 + r * (py1 - py0) / 2;
      el('line', { x1: px0, y1: yy, x2: px1, y2: yy, stroke: '#E8EAED', 'stroke-width': 1 }, g);
    }
    const tl = el('text', { x: px1, y: by + BOX.h - 3, 'text-anchor': 'end', style: 'font-size:10px;fill:#9AA0A6' }, g);
    tl.textContent = 'time →';

    if (!spectrum) el('polyline', { points: P.join(' '), fill: 'none', stroke: JCOL[j], 'stroke-opacity': 0.22, 'stroke-width': 2, class: 'b2' }, g);
    const planned = el('polyline', { fill: 'none', stroke: JCOL[j], 'stroke-width': 2.5, 'stroke-dasharray': '4 5', 'stroke-linecap': 'round', class: 'b2' }, g);
    const live = el('polyline', { fill: 'none', stroke: JCOL[j], 'stroke-width': 3, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, g);

    let bars = null;
    if (spectrum) {
      const top = by + 62, bot = py1 + 4, bh = (bot - top) / CHUNK_STEPS, maxW = bw - 12;
      const fl = el('text', { transform: `translate(${bx + 13},${(top + bot) / 2}) rotate(-90)`, 'text-anchor': 'middle', style: 'font-size:10px;fill:#9AA0A6' }, g);
      fl.textContent = 'freq →';
      const amps = [...Array(NCH)].map((_, c) =>
        dct2([...Array(CHUNK_STEPS)].map((_, i) => Q[j]((c + (i + 0.5) / CHUNK_STEPS) * H))).map(Math.abs));
      const acMax = Math.max(...amps.flatMap(a => a.slice(1)));
      bars = amps.map((a, c) => {
        const x0 = px0 + c * bw + 5;
        const col = a.map((amp, f) => ({
          w: f ? amp / acMax * maxW : maxW,
          r: el('rect', { x: x0, y: bot - (f + 1) * bh + 1, height: bh - 2, width: 0, rx: 1.5,
                          fill: JCOL[j], 'fill-opacity': f ? 1 : 0.35 }, g),
        }));
        // k = 0 is the chunk mean and dwarfs the rest, so its bar is clipped and marked with a break
        const bx0 = x0 + maxW * 0.62, by0 = bot - bh + 0.5;
        col.brk = el('path', { d: `M${bx0},${by0 + bh} l3,${-bh - 1} M${bx0 + 3},${by0 + bh} l3,${-bh - 1}`,
                                stroke: '#fff', 'stroke-width': 1.6, opacity: 0 }, g);
        return col;
      });
    }

    const head = el('line', { y1: py0 - 6, y2: py1 + 4, stroke: '#9AA0A6', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, g);
    const dot = el('circle', { r: spectrum ? 0 : 5, fill: '#fff', stroke: JCOL[j], 'stroke-width': 3 }, g);

    return { by, px0, bw, X, Y, pt, P, val, bands, bandLabels, planned, live, head, dot, bars };
  });

  {
    const c = charts[0], y0 = BOX.y[2] + BOX.h + 16;
    const g = el('g', { class: 'b2' }, chartsG);
    el('path', { d: `M${c.px0},${y0 - 7} V${y0} H${c.px0 + c.bw} V${y0 - 7}`, fill: 'none', stroke: '#5F6368', 'stroke-width': 2 }, g);
    const t = el('text', { x: c.px0 + c.bw + 12, y: y0 + 5, style: 'font-size:14px;fill:#5F6368' }, g);
    t.textContent = spectrum
      ? 'one chunk’s spectrum · faded bar = freq 0 (mean), clipped'
      : 'one action chunk = H future actions, predicted in one shot';
  }

  const chipG = el('g', { transform: 'translate(80,160)' }, document.getElementById('chip'));
  const chipRect = el('rect', { class: 'chip', width: 300, height: 46, rx: 23, fill: '#fff', stroke: '#DADCE0', 'stroke-width': 1.5 }, chipG);
  const chipVLA = el('text', { x: 22, y: 30, style: 'font-size:19px;font-weight:700' }, chipG);
  chipVLA.innerHTML = '<tspan fill="#4285F4">V</tspan><tspan fill="#EA4335">L</tspan><tspan fill="#34A853">A</tspan>';
  const chipTxt = el('text', { x: 72, y: 29, style: 'font-size:16px;fill:#3C4043' }, chipG);

  let recFrom = chunks ? 0 : null, lastT = null, lastK = -1;
  // let the browser paint the un-chunked charts first so the bands fade in one by one
  if (chunks) setTimeout(() => { slide.dataset.step = 2; }, 450);

  function frame(t) {
    const st = sceneState(t);
    arm.update(st);
    if (!chunks && recFrom === null && t >= REVEAL_AFTER) { recFrom = t; slide.dataset.step = 1; }
    if (lastT !== null && t < lastT && recFrom !== null) recFrom = 0;
    lastT = t;
    const step = +slide.dataset.step;

    const J = [st.fk.p0, st.fk.p1, st.fk.p2].map(arm.toSlide);
    const tip = arm.toSlide(st.fk.tip);
    const k = Math.min(NCH - 1, Math.floor(t / H));

    if (step === 0) {
      const pos = [[J[0][0] - 50, J[0][1] + 6, 'end'], [J[1][0] - 30, J[1][1] + 6, 'end'],
                   [J[2][0] + 26, J[2][1] - 18, 'start'], [tip[0] + 36, tip[1] + 6, 'start']];
      labelTxt.forEach((e, i) => { e.setAttribute('x', pos[i][0]); e.setAttribute('y', pos[i][1]); e.setAttribute('text-anchor', pos[i][2]); });
    }

    for (let j = 0; j < 3; j++) {
      const [jx, jy] = J[j];
      rings[j].setAttribute('cx', jx); rings[j].setAttribute('cy', jy);
      const c = charts[j];
      const bx = BOX.x - 6, by = c.by + BOX.h / 2, sx = jx + 22, dx = bx - sx;
      arrows[j].setAttribute('d', `M${sx},${jy} C${sx + dx * 0.5},${jy} ${bx - dx * 0.5},${by} ${bx},${by}`);

      const v = st.q[j];
      c.val.textContent = `θ${SUB[j]} = ${fmtDeg(v)}`;
      if (spectrum) {
        c.bars.forEach((col, ci) => {
          const g = ci < k ? 1 : ci > k ? 0 : Math.min(1, (t - ci * H) / GROW), e = 1 - (1 - g) ** 3;
          col.forEach(b => b.r.setAttribute('width', b.w * e));
          col.brk.setAttribute('opacity', g >= 1 ? 1 : 0);
        });
      } else if (recFrom !== null) {
        const i0 = Math.ceil(recFrom / T * N_SAMPLES), i1 = Math.floor(t / T * N_SAMPLES);
        const pts = c.P.slice(i0, i1 + 1); pts.push(c.pt(t, v));
        c.live.setAttribute('points', pts.join(' '));
      } else c.live.setAttribute('points', '');
      const hx = c.X(t);
      c.head.setAttribute('x1', hx); c.head.setAttribute('x2', hx);
      c.dot.setAttribute('cx', hx); c.dot.setAttribute('cy', c.Y(v));

      if (step >= 2 && !spectrum) {
        const end = (k + 1) * H;
        const pts = [c.pt(t, v), ...c.P.slice(Math.ceil(t / T * N_SAMPLES), Math.floor(end / T * N_SAMPLES) + 1)];
        c.planned.setAttribute('points', pts.join(' '));
      }
      c.bands.forEach((b, i) => b.setAttribute('fill', step >= 2 && i === k ? '#E8F0FE' : (i % 2 ? '#FAFBFC' : '#F1F3F4')));
      c.bandLabels.forEach((b, i) => b.setAttribute('style',
        step >= 2 && i === k ? 'font-size:11px;font-weight:700;fill:#1967D2' : 'font-size:11px;fill:#5F6368'));
    }

    if (k !== lastK) {
      lastK = k;
      chipTxt.textContent = spectrum
        ? `→ predicts chunk ${k + 1} as ${CHUNK_STEPS} DCT coefficients`
        : `→ predicts chunk ${k + 1} of ${NCH}`;
      chipRect.setAttribute('width', chipTxt.getComputedTextLength() + 94);
      chipG.classList.remove('flash'); void chipG.getBBox(); chipG.classList.add('flash');
    }
  }

  startDeck({ index, total, prev, next, frame });
}

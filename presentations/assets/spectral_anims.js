// Looping SVG animations for the spectral-token slides. Each builder draws in its own
// native coordinates (laid out for the three-column slide) inside a group with the given
// transform, and returns update(t) for t in [0, T).
const YEL = '#FBBC05', GRAY = '#9AA0A6';
const ease = u => u <= 0 ? 0 : u >= 1 ? 1 : (1 - Math.cos(Math.PI * u)) / 2;
const phase = (t, a, b) => ease((t - a) / (b - a));
const loopFade = t => t < 0.3 ? t / 0.3 : t > 8.5 ? (9 - t) / 0.5 : 1;
const txt = (s, attrs, p) => { const e = el('text', attrs, p); e.textContent = s; return e; };

const SIG = [2.0, 3.2, 5.1, 6.0, 5.8, 4.6, 3.1, 2.3];
const haarStep = v => {
  const a = [], d = [];
  for (let i = 0; i < v.length; i += 2) { a.push((v[i] + v[i + 1]) / Math.SQRT2); d.push((v[i] - v[i + 1]) / Math.SQRT2); }
  return [a, d];
};

// native box: x 96..430, y 220..480
function buildWavelet(svg, transform = '') {
  const root = el('g', { transform }, svg);
  const [A1, D1] = haarStep(SIG), [A2, D2] = haarStep(A1), [A3, D3] = haarStep(A2);
  const seg = (vals, color) => vals.map(v => ({ v, color }));
  const ROWS = [
    { name: 'signal',  bars: seg(SIG, GRAY), cuts: [] },
    { name: 'level 1', bars: [...seg(A1, BLUE), ...seg(D1, RED)], cuts: [4] },
    { name: 'level 2', bars: [...seg(A2, BLUE), ...seg(D2, YEL), ...seg(D1, RED)], cuts: [2, 4] },
    { name: 'level 3', bars: [...seg(A3, BLUE), ...seg(D3, GREEN), ...seg(D2, YEL), ...seg(D1, RED)], cuts: [1, 2, 4] },
  ];
  const C = { x0: 160, slot: 33, bw: 23, base: [250, 318, 386, 454], amp: 26 };
  const rows = ROWS.map((row, r) => {
    const g = el('g', {}, root), y = C.base[r];
    txt(row.name, { x: 100, y: y + 4, style: 'font-size:12px;fill:#5F6368' }, g);
    el('line', { x1: C.x0 - 6, x2: C.x0 + 8 * C.slot, y1: y, y2: y, stroke: '#E8EAED' }, g);
    for (const c of row.cuts) {
      const x = C.x0 + c * C.slot - (C.slot - C.bw) / 2;
      el('line', { x1: x, x2: x, y1: y - C.amp - 4, y2: y + C.amp + 4, stroke: '#BDC1C6', 'stroke-dasharray': '3 3' }, g);
    }
    const scale = C.amp / Math.max(...row.bars.map(b => Math.abs(b.v)));
    const rects = row.bars.map((b, i) => ({
      h: b.v * scale,
      r: el('rect', { x: C.x0 + i * C.slot, width: C.bw, rx: 3, fill: b.color }, g),
    }));
    if (r) txt('↓', { x: 110, y: y - 30, style: 'font-size:13px;fill:#BDC1C6' }, g);
    return { g, y, rects };
  });
  return t => {
    root.setAttribute('opacity', loopFade(t));
    rows.forEach((row, r) => {
      const p = phase(t, 0.3 + 1.5 * r, 1.4 + 1.5 * r);
      row.g.setAttribute('opacity', Math.min(1, p * 1.5));
      row.rects.forEach(({ h, r: rect }) => {
        const hh = h * p;
        rect.setAttribute('y', hh >= 0 ? row.y - hh : row.y);
        rect.setAttribute('height', Math.abs(hh));
      });
    });
  };
}

// native box: x 486..800, y 212..470
function buildWeighting(svg, transform = '', { formula = true } = {}) {
  const root = el('g', { transform }, svg);
  if (formula) txt('loss weight = quantizer step (mean 1)', { x: 640, y: 226, 'text-anchor': 'middle', class: 'mono', style: 'font-size:11.5px;fill:#5F6368' }, root);
  // a one-bin miss in A₃ shifts every step (flat Haar basis); in D₁ it nudges one pair of steps
  const MISS = [
    { label: 'one-bin miss in the coarse trend', top: 280, delta: SIG.map(() => -1.4), w: '×3.2', big: true },
    { label: 'one-bin miss in the finest detail', top: 394, delta: SIG.map((_, i) => i === 4 ? 0.3 : i === 5 ? -0.3 : 0), w: '×0.4', big: false },
  ];
  const panels = MISS.map(m => {
    txt(m.label, { x: 490, y: m.top - 14, style: 'font-size:12px;fill:#3C4043' }, root);
    const px = i => 496 + i * 31, py = v => m.top + 6 + (6.5 - v) / 5 * 54;
    const err = el('path', { fill: RED, 'fill-opacity': 0.16 }, root);
    const orig = el('polyline', { points: SIG.map((v, i) => `${px(i)},${py(v)}`).join(' '), fill: 'none', stroke: '#3C4043', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, root);
    const dots = SIG.map((v, i) => el('circle', { cx: px(i), cy: py(v), r: 3.5, fill: '#3C4043' }, root));
    const pert = el('polyline', { fill: 'none', stroke: RED, 'stroke-width': 2.5, 'stroke-dasharray': '5 4', 'stroke-linejoin': 'round' }, root);
    const cx = 762, cy = m.top + 32, cw = m.big ? 64 : 44, ch = m.big ? 34 : 24;
    const chip = el('g', {}, root);
    el('rect', { x: -cw / 2, y: -ch / 2, width: cw, height: ch, rx: ch / 2, fill: m.big ? RED : '#FCE8E6' }, chip);
    txt(m.w, { y: m.big ? 7 : 4.5, 'text-anchor': 'middle', class: 'mono', style: `font-size:${m.big ? 19 : 13}px;font-weight:500;fill:${m.big ? '#fff' : '#C5221F'}` }, chip);
    const len = orig.getTotalLength();
    orig.setAttribute('stroke-dasharray', len);
    return { m, px, py, err, orig, dots, pert, chip, cx, cy, len };
  });
  return t => {
    root.setAttribute('opacity', loopFade(t));
    const draw = phase(t, 0.3, 1.5), miss = phase(t, 2.0, 3.4), pop = phase(t, 4.0, 4.8);
    panels.forEach(p => {
      p.orig.setAttribute('stroke-dashoffset', p.len * (1 - draw));
      p.dots.forEach((d, i) => d.setAttribute('opacity', draw * SIG.length > i ? 1 : 0));
      const P = SIG.map((v, i) => [p.px(i), p.py(v + p.m.delta[i] * miss)]);
      p.pert.setAttribute('points', P.map(q => q.join(',')).join(' '));
      p.pert.setAttribute('opacity', Math.min(1, miss * 4));
      const O = SIG.map((v, i) => [p.px(i), p.py(v)]);
      p.err.setAttribute('d', 'M' + [...O, ...P.reverse()].map(q => q.join(',')).join(' L') + ' Z');
      const s = pop * (1 + 0.25 * Math.sin(Math.PI * pop));
      p.chip.setAttribute('transform', `translate(${p.cx},${p.cy}) scale(${s})`);
    });
  };
}

// native box: x 862..1180, y 212..480
function buildSoftLabels(svg, transform = '', { formula = true } = {}) {
  const root = el('g', { transform }, svg);
  const SIGMA = 0.007, NB = 11, MID = 5;
  const soft = step => {
    const q = [...Array(NB)].map((_, b) => Math.exp(-(((b - MID) * step) ** 2) / (2 * SIGMA ** 2)));
    const s = q.reduce((a, b) => a + b, 0);
    return q.map(v => v / s);
  };
  if (formula) txt('q(b) ∝ exp(−((b − b*)·step)² / 2σ²)', { x: 1020, y: 226, 'text-anchor': 'middle', class: 'mono', style: 'font-size:11.5px;fill:#5F6368' }, root);
  const modeA = txt('one-hot target', { x: 1020, y: 250, 'text-anchor': 'middle', style: 'font-size:13px;font-weight:500;fill:#5F6368' }, root);
  const modeB = txt('soft target', { x: 1020, y: 250, 'text-anchor': 'middle', style: 'font-size:13px;font-weight:500;fill:#137333' }, root);
  const PLOTS = [
    { label: 'coarsest coefficient · step 0.08', step: 0.08, base: 352, col: BLUE },
    { label: 'finest detail · step 0.007', step: 0.007, base: 466, col: RED },
  ];
  const C = { x0: 866, slot: 28, bw: 18, hmax: 76 };
  const plots = PLOTS.map(p => {
    txt(p.label, { x: C.x0, y: p.base - C.hmax - 10, style: 'font-size:12px;fill:#3C4043' }, root);
    el('line', { x1: C.x0 - 4, x2: C.x0 + NB * C.slot - 6, y1: p.base, y2: p.base, stroke: '#BDC1C6' }, root);
    txt('b*', { x: C.x0 + MID * C.slot + C.bw / 2, y: p.base + 14, 'text-anchor': 'middle', style: 'font-size:11px;font-style:italic;fill:#5F6368' }, root);
    const bars = soft(p.step).map((v, b) => ({
      one: b === MID ? 1 : 0, soft: v,
      r: el('rect', { x: C.x0 + b * C.slot, width: C.bw, rx: 2, fill: p.col, 'fill-opacity': b === MID ? 1 : 0.7 }, root),
    }));
    return { p, bars };
  });
  return t => {
    root.setAttribute('opacity', loopFade(t));
    const s = phase(t, 3.2, 5.0);
    modeA.setAttribute('opacity', 1 - s);
    modeB.setAttribute('opacity', s);
    plots.forEach(({ p, bars }) => bars.forEach(b => {
      const h = C.hmax * (b.one + (b.soft - b.one) * s);
      b.r.setAttribute('y', p.base - h); b.r.setAttribute('height', h);
    }));
  };
}

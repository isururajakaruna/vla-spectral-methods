// Small animated SVG charts for the experiment slides (needs el() from arm.js).
const C = { fast: '#EA4335', dct: '#F9AB00', haar: '#4285F4', adapt: '#34A853', ref: '#9AA0A6', ink: '#202124' };

function wilson(k, n, z = 1.96) {
  const p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n);
  const m = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [Math.max(0, (c - m) / d), Math.min(1, (c + m) / d)];
}

function label(parent, x, y, s, attrs = {}) {
  const t = el('text', { x, y, ...attrs }, parent);
  t.textContent = s;
  return t;
}

const delay = (node, s) => (node.style.animationDelay = s.toFixed(2) + 's', node);

function drawIn(path, s, dur) {
  path.style.setProperty('--len', path.getTotalLength() + 1);
  if (dur) path.style.animationDuration = dur + 's';
  path.classList.add('draw');
  return delay(path, s);
}

function grid(g, { x, y, w, h, yMax, yTicks, yFmt, yLabel }) {
  const Y = v => y + h - v / yMax * h;
  for (const v of yTicks) {
    el('line', { x1: x, x2: x + w, y1: Y(v), y2: Y(v), stroke: v ? '#ECEDEF' : '#BDC1C6', 'stroke-width': v ? 1 : 1.4 }, g);
    label(g, x - 10, Y(v) + 4, yFmt(v), { class: 'ax', 'text-anchor': 'end' });
  }
  if (yLabel) label(g, 0, 0, yLabel, {
    class: 'ax-t', 'text-anchor': 'middle', transform: `translate(${x - 54},${y + h / 2}) rotate(-90)`,
  });
  return Y;
}

const pct = v => Math.round(v * 100) + '%';
const kfmt = v => v >= 1000 ? v / 1000 + 'k' : '' + v;

// series: [{ name, color, k: [...], n, dash, width, dy }]
function lineChart(svg, o) {
  const { x, y, w, h, xs, xMin = 0, xMax = xs[xs.length - 1], yMax = 1, yTicks = [0, .25, .5, .75, 1],
    yFmt = pct, xLabel, yLabel, series, t0 = 0.3, ci = true, dur = 1.6 } = o;
  const g = el('g', {}, svg);
  const X = v => x + (v - xMin) / (xMax - xMin) * w;
  const Y = grid(g, { x, y, w, h, yMax, yTicks, yFmt, yLabel });
  for (const v of xs) {
    el('line', { x1: X(v), x2: X(v), y1: y + h, y2: y + h + 5, stroke: '#BDC1C6' }, g);
    label(g, X(v), y + h + 21, kfmt(v), { class: 'ax', 'text-anchor': 'middle' });
  }
  if (xLabel) label(g, x + w / 2, y + h + 46, xLabel, { class: 'ax-t', 'text-anchor': 'middle' });

  const layer = { bands: el('g', {}, g), lines: el('g', {}, g), dots: el('g', {}, g), labels: el('g', {}, g) };
  series.forEach((s, si) => {
    const st = t0 + si * 0.25;
    const ps = s.k.map(k => k / s.n);
    if (ci && s.ci !== false) {
      const iv = s.k.map(k => wilson(k, s.n));
      const pts = xs.map((v, i) => `${X(v)},${Y(iv[i][1])}`)
        .concat(xs.map((v, i) => `${X(v)},${Y(iv[i][0])}`).reverse());
      delay(el('polygon', { points: pts.join(' '), fill: s.color, 'fill-opacity': 0.1, class: 'pop' }, layer.bands), st + dur);
    }
    const d = xs.map((v, i) => `${i ? 'L' : 'M'}${X(v)},${Y(ps[i])}`).join('');
    const line = el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 3.2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, layer.lines);
    drawIn(line, st, dur);
    xs.forEach((v, i) => delay(el('circle', { cx: X(v), cy: Y(ps[i]), r: 4.5, fill: '#fff',
      stroke: s.color, 'stroke-width': 2.6, class: 'pop' }, layer.dots), st + dur * (X(v) - x) / w));
    const ly = Y(ps[ps.length - 1]) + 5 + (s.dy || 0);
    const lt = label(layer.labels, X(xs[xs.length - 1]) + 14, ly, '', { class: 'pop', 'font-size': 15 });
    const nm = el('tspan', { fill: s.color, 'font-weight': 500 }, lt); nm.textContent = s.name;
    const vl = el('tspan', { fill: '#5F6368', dx: 7, 'font-size': 13 }, lt);
    vl.textContent = `${s.k[s.k.length - 1]}/${s.n}`;
    delay(lt, st + dur);
  });
  return { X, Y, g };
}

// bars: [{ label, sub, v, color, text, faded }]; groups: [{ from, to, label }]
function barChart(svg, o) {
  const { x, y, w, h, yMax = 1, yTicks = [0, .25, .5, .75, 1], yFmt = pct, yLabel, bars, groups = [],
    t0 = 0.3, barFrac = 0.56, stagger = 0.12 } = o;
  const g = el('g', {}, svg);
  const Y = grid(g, { x, y, w, h, yMax, yTicks, yFmt, yLabel });
  const slot = w / bars.length, bw = slot * barFrac;
  const cx = i => x + slot * (i + 0.5);
  const bg = el('g', {}, g);
  bars.forEach((b, i) => {
    const st = t0 + i * stagger, top = Y(b.v), bh = Math.max(y + h - top, 2.5);
    const r = el('rect', { x: cx(i) - bw / 2, y: y + h - bh, width: bw, height: bh, rx: 5,
      fill: b.color, class: 'grow' }, bg);
    if (b.faded) {
      r.setAttribute('fill', b.color); r.setAttribute('fill-opacity', 0.35);
      r.setAttribute('stroke', b.color); r.setAttribute('stroke-width', 2); r.setAttribute('stroke-dasharray', '5 4');
    }
    delay(r, st);
    delay(label(g, cx(i), y + h - bh - 10, b.text ?? yFmt(b.v), {
      class: 'pop', 'text-anchor': 'middle', 'font-size': 16, 'font-weight': 500, fill: b.textColor || C.ink,
    }), st + 0.6);
    label(g, cx(i), y + h + 22, b.label, { 'text-anchor': 'middle', 'font-size': 14, fill: C.ink });
    if (b.sub) label(g, cx(i), y + h + 39, b.sub, { 'text-anchor': 'middle', 'font-size': 11.5, fill: '#80868B' });
  });
  el('line', { x1: x, x2: x + w, y1: y + h, y2: y + h, stroke: '#BDC1C6', 'stroke-width': 1.4 }, g);
  for (const gr of groups) {
    const a = cx(gr.from) - bw / 2 - 6, b = cx(gr.to) + bw / 2 + 6, gy = y + h + 54;
    el('path', { d: `M${a},${gy - 5}V${gy}H${b}V${gy - 5}`, fill: 'none', stroke: '#BDC1C6', 'stroke-width': 1.4 }, g);
    label(g, (a + b) / 2, gy + 18, gr.label, { 'text-anchor': 'middle', class: 'ax-t' });
  }
  return { Y, cx, g };
}

// Vertical significance bracket between two y positions at x.
function bracket(svg, x, y1, y2, text, t, color = '#5F6368') {
  const g = delay(el('g', { class: 'pop' }, svg), t);
  el('path', { d: `M${x - 6},${y1}H${x}V${y2}H${x - 6}`, fill: 'none', stroke: color, 'stroke-width': 1.6 }, g);
  label(g, x + 8, (y1 + y2) / 2 + 5, text, { 'font-size': 13.5, fill: color, 'font-weight': 500 });
  return g;
}

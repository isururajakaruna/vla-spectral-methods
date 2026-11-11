// Shared 3-link arm: kinematics, pick-and-place trajectory and the SVG cartoon.
const NS = 'http://www.w3.org/2000/svg';
const BLUE = '#4285F4', RED = '#EA4335', GREEN = '#34A853';
const JCOL = [BLUE, RED, GREEN];
const SUB = ['₁', '₂', '₃'];

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const deg = r => r * 180 / Math.PI, rad = d => d * Math.PI / 180;
const fmtDeg = v => (v < 0 ? '−' : '') + Math.abs(Math.round(v)) + '°';

// arm-local coords: y up, shoulder at origin
const L1 = 140, L2 = 120, L3 = 56;
const T = 9, H = 1.5, NCH = Math.round(T / H), SPEED = 0.85;
const CAN_A = [210, -13], CAN_REST_Y = -10;
const T_ATTACH = 2.9, T_RELEASE = 6.45;

function ik(x, y, phiDeg) {
  const phi = rad(phiDeg);
  const wx = x - L3 * Math.cos(phi), wy = y - L3 * Math.sin(phi);
  const D = Math.max(-1, Math.min(1, (wx * wx + wy * wy - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
  const q2 = -Math.acos(D);
  const q1 = Math.atan2(wy, wx) - Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
  return [deg(q1), deg(q2), phiDeg - deg(q1) - deg(q2)];
}

function fk(q) {
  const a1 = rad(q[0]), a2 = a1 + rad(q[1]), a3 = a2 + rad(q[2]);
  const p1 = [L1 * Math.cos(a1), L1 * Math.sin(a1)];
  const p2 = [p1[0] + L2 * Math.cos(a2), p1[1] + L2 * Math.sin(a2)];
  const tip = [p2[0] + L3 * Math.cos(a3), p2[1] + L3 * Math.sin(a3)];
  return { p0: [0, 0], p1, p2, tip, a3: deg(a3) };
}

function pchip(ts, ys) {
  const n = ts.length, h = [], d = [], m = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) { h[i] = ts[i + 1] - ts[i]; d[i] = (ys[i + 1] - ys[i]) / h[i]; }
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] > 0) {
      const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1];
      m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
    }
  }
  return t => {
    let i = 0;
    while (i < n - 2 && t > ts[i + 1]) i++;
    const hh = h[i], s = Math.min(1, Math.max(0, (t - ts[i]) / hh)), s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * ys[i] + (s3 - 2 * s2 + s) * hh * m[i]
         + (-2 * s3 + 3 * s2) * ys[i + 1] + (s3 - s2) * hh * m[i + 1];
  };
}

// t, tip x, tip y, gripper angle, gripper open (1) / closed (0)
const KF = [
  [0.0, 150, 120, -60, 1],
  [1.4, 210,  60, -90, 1],
  [2.4, 210, -13, -90, 1],
  [3.0, 210, -13, -90, 0],
  [4.0, 210,  60, -90, 0],
  [5.4,  90,  70, -90, 0],
  [6.3,  90,  20, -90, 0],
  [6.8,  90,  20, -90, 1],
  [7.6,  90, 100, -90, 1],
  [9.0, 150, 120, -60, 1],
];
const kts = KF.map(k => k[0]);
const kq = KF.map(k => ik(k[1], k[2], k[3]));
const Q = [0, 1, 2].map(j => pchip(kts, kq.map(q => q[j])));
const G = pchip(kts, KF.map(k => k[4]));
const releasePose = fk(Q.map(f => f(T_RELEASE))).tip;

function sceneState(t) {
  const q = Q.map(f => f(t)), g = Math.max(0, Math.min(1, G(t))), k = fk(q);
  let can = { x: CAN_A[0], y: CAN_A[1], rot: 0, op: t < 0.5 ? t / 0.5 : 1 };
  if (t >= T_ATTACH && t < T_RELEASE) {
    can = { x: k.tip[0], y: k.tip[1], rot: k.a3 + 90, op: 1 };
  } else if (t >= T_RELEASE) {
    const u = Math.min(1, (t - T_RELEASE) / 0.28);
    can = { x: releasePose[0], y: releasePose[1] + (CAN_REST_Y - releasePose[1]) * u * u, rot: 0,
            op: t > T - 0.6 ? (T - t) / 0.6 : 1 };
  }
  return { q, g, fk: k, can };
}

function makeArm(parent, { x, y, s, table = [-170, 330], mono = false }) {
  const nss = { 'vector-effect': 'non-scaling-stroke' };
  const root = el('g', { transform: `translate(${x},${y}) scale(${s},${-s})` }, parent);
  el('rect', { x: table[0], y: -79, width: table[1] - table[0], height: 46, fill: '#F8F9FA' }, root);
  el('line', { x1: table[0], y1: -33, x2: table[1], y2: -33, stroke: '#BDC1C6', 'stroke-width': 2, ...nss }, root);
  el('rect', { x: 52, y: -33, width: 76, height: 34, fill: '#E8EAED' }, root);

  const can = el('g', {}, root);
  el('rect', { x: -15, y: -20, width: 30, height: 40, rx: 5, fill: mono ? '#DADCE0' : '#FBBC05',
    stroke: mono ? '#80868B' : '#E37400', 'stroke-width': 1.5, ...nss }, can);
  el('rect', { x: -15, y: -7, width: 30, height: 13, fill: '#fff', 'fill-opacity': 0.9 }, can);
  el('rect', { x: -15, y: 15, width: 30, height: 5, rx: 2, fill: mono ? '#9AA0A6' : '#F29900' }, can);

  el('path', { d: 'M50,-33 L130,-33 L135,-2 L45,-2 Z', fill: '#fff', stroke: '#9AA0A6', 'stroke-width': 2, 'stroke-linejoin': 'round', ...nss }, root);
  for (const bx of [68, 84, 100, 116])
    el('line', { x1: bx, y1: -31, x2: bx + (bx - 90) * 0.06, y2: -4, stroke: '#DADCE0', 'stroke-width': 2, ...nss }, root);
  el('rect', { x: 42, y: -4, width: 96, height: 7, rx: 3.5, fill: '#9AA0A6' }, root);
  el('path', { d: 'M-48,-33 L48,-33 L30,-6 L-30,-6 Z', fill: '#E8EAED', stroke: '#9AA0A6', 'stroke-width': 2, 'stroke-linejoin': 'round', ...nss }, root);

  const link = { fill: '#fff', stroke: '#3C4043', 'stroke-width': 2.5, ...nss };
  const g1 = el('g', {}, root);
  el('rect', { x: -15, y: -15, width: L1 + 30, height: 30, rx: 15, ...link }, g1);
  const g2 = el('g', {}, g1);
  el('rect', { x: -13, y: -13, width: L2 + 26, height: 26, rx: 13, ...link }, g2);
  const g3 = el('g', {}, g2);
  el('rect', { x: -9, y: -9, width: L3 - 15, height: 18, rx: 9, ...link }, g3);
  el('rect', { x: L3 - 30, y: -33, width: 9, height: 66, rx: 4, fill: '#5F6368' }, g3);
  const fU = el('g', {}, g3), fL = el('g', {}, g3);
  el('rect', { x: L3 - 23, y: 0, width: 37, height: 8, rx: 3, fill: '#5F6368' }, fU);
  el('rect', { x: L3 - 23, y: -8, width: 37, height: 8, rx: 3, fill: '#5F6368' }, fL);
  const joint = (p, c) => {
    el('circle', { r: 15, fill: c, stroke: '#fff', 'stroke-width': 3, ...nss }, p);
    el('circle', { r: 5, fill: '#fff' }, p);
  };
  const jc = mono ? ['#3C4043', '#3C4043', '#3C4043'] : JCOL;
  joint(g3, jc[2]); joint(g2, jc[1]); joint(g1, jc[0]);

  return {
    update(st) {
      const [q1, q2, q3] = st.q;
      g1.setAttribute('transform', `rotate(${q1})`);
      g2.setAttribute('transform', `translate(${L1},0) rotate(${q2})`);
      g3.setAttribute('transform', `translate(${L2},0) rotate(${q3})`);
      const gap = 15.5 + st.g * 8.5;
      fU.setAttribute('transform', `translate(0,${gap})`);
      fL.setAttribute('transform', `translate(0,${-gap})`);
      can.setAttribute('transform', `translate(${st.can.x},${st.can.y}) rotate(${st.can.rot})`);
      can.setAttribute('opacity', st.can.op);
    },
    toSlide: p => [x + s * p[0], y - s * p[1]],
  };
}

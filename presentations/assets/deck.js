// Per-page slide controller. Each slide is its own HTML file; stepping past the
// last build follows `next`, stepping back from build 0 opens `prev` at its last build.
const SVG_DEFS = `
<filter id="sh" x="-10%" y="-15%" width="120%" height="140%">
  <feDropShadow dx="0" dy="1" stdDeviation="1" flood-color="#3C4043" flood-opacity=".22"/>
  <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#3C4043" flood-opacity=".10"/>
</filter>
<marker id="ah-gray" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto">
  <path d="M0,0 L10,5 L0,10 z" fill="#5F6368"/></marker>
${['#4285F4', '#EA4335', '#34A853'].map((c, i) => `
<marker id="ah-${i}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto">
  <path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`).join('')}`;

function startDeck({ index, total, prev = null, next = null, steps = 0, frame, enter, step: onStep }) {
  document.body.insertAdjacentHTML('afterbegin',
    `<svg width="0" height="0" style="position:absolute"><defs>${SVG_DEFS}</defs></svg>`);
  const deck = document.getElementById('deck');
  const slide = deck.querySelector('.slide');

  const link = (href, ch, label) => href ? `<a href="${href}" title="${label}">${ch}</a>` : `<span class="off">${ch}</span>`;
  deck.insertAdjacentHTML('beforeend',
    `<div id="pager">${link(prev, '‹', 'Previous slide')}<span>${index} / ${total}</span>${link(next, '›', 'Next slide')}</div>`);
  if (index === 1) deck.insertAdjacentHTML('beforeend',
    '<div id="hint">→ / space / click to advance · ← to go back · F for fullscreen</div>');

  let cur = 0, t0 = performance.now();
  const sceneT = () => ((performance.now() - t0) / 1000 * SPEED) % T;

  function setStep(k, initial) {
    cur = Math.max(0, Math.min(steps, k));
    if (steps) slide.dataset.step = cur;
    initial ? enter?.(cur) : onStep?.(cur, sceneT());
    history.replaceState(null, '', cur ? `#${cur}` : location.pathname);
    const hint = document.getElementById('hint');
    if (hint && cur > 0) hint.style.opacity = 0;
  }
  const fwd = () => cur < steps ? setStep(cur + 1) : next && (location.href = next);
  const back = () => cur > 0 ? setStep(cur - 1) : prev && (location.href = prev + '#last');

  addEventListener('keydown', e => {
    if (['ArrowRight', ' ', 'PageDown', 'Enter'].includes(e.key)) { e.preventDefault(); fwd(); }
    else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) { e.preventDefault(); back(); }
    else if (e.key === 'f' || e.key === 'F') {
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    }
  });
  addEventListener('click', e => {
    if (e.target.closest('a')) return;
    e.clientX < innerWidth / 3 ? back() : fwd();
  });

  const fit = () => {
    const s = Math.min(innerWidth / 1280, innerHeight / 720) * 0.94;
    deck.style.transform = `translate(-50%,-50%) scale(${s})`;
  };
  addEventListener('resize', fit); fit();

  const fromHash = () => {
    const h = location.hash.slice(1);
    setStep(h === 'last' ? steps : (+h || 0), true);
  };
  addEventListener('hashchange', fromHash);
  fromHash();

  (function tick() {
    frame?.(sceneT(), cur);
    requestAnimationFrame(tick);
  })();
}

'use strict';
// Toy SCPO run: state, drawing, and controls for the geometry viewer.
const run = {w: SCPOGeometry.start.slice(), previous: null, path: [SCPOGeometry.start.slice()], iteration: 0};
const rateControl = document.querySelector('#learning-rate'), boundControl = document.querySelector('#bound-scale');
const screen = w => [240 + 155 * w[0], 239 - 155 * w[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const point = (id, w) => { const [x, y] = screen(w), e = document.getElementById(id); e.setAttribute('cx', x); e.setAttribute('cy', y); };
const line = (id, a, b) => { const [x1, y1] = screen(a), [x2, y2] = screen(b); const e = document.getElementById(id); Object.entries({x1, y1, x2, y2}).forEach(([k, v]) => e.setAttribute(k, v)); };
const label = (id, w, text, dx = 9, dy = -9) => { const [x, y] = screen(w), e = document.getElementById(id); e.setAttribute('x', x + dx); e.setAttribute('y', y + dy); if (text !== undefined) e.textContent = text; };
const show = (ids, visible) => ids.forEach(id => { document.getElementById(id).style.display = visible ? '' : 'none'; });
const number = x => (Math.abs(x) < 0.0005 ? 0 : x).toFixed(3).replace('-', '−');
const svgEl = (name, attrs) => { const e = document.createElementNS('http://www.w3.org/2000/svg', name); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); return e; };

// Static scenery: loss contours and the loss minimum.
(() => {
  const contours = document.querySelector('#loss-contours');
  for (const level of [0.08, 0.3, 0.7, 1.3, 2.2]) {
    const c = SCPOGeometry.contour(level), [cx, cy] = screen(c.center);
    contours.append(svgEl('ellipse', {cx, cy, rx: 155 * c.rx, ry: 155 * c.ry, transform: `rotate(${-c.angle} ${cx} ${cy})`}));
  }
  point('target-point', SCPOGeometry.target); label('target-label', SCPOGeometry.target, undefined, 0, 19);
  document.querySelector('#trust-circle').setAttribute('r', 155 * SCPOGeometry.trustRadius);
})();

function currentStep() {
  const eta = Number(rateControl.value) / 100, ell = Number(boundControl.value) / 100;
  const bank = SCPOGeometry.setup(run.w, run.previous, eta, ell), projected = SCPOGeometry.project(bank);
  const alpha = SCPOGeometry.backtrack(run.w, projected), accepted = projected.map(x => x * alpha);
  return {bank, projected, alpha, accepted};
}
function drawRun() {
  const {bank, projected, alpha, accepted} = currentStep(), w = run.w;
  const coords = SCPOGeometry.boundary(bank).map(v => screen(add(w, v)));
  const path = coords.map(([x, y], i) => (i ? 'L' : 'M') + x.toFixed(3) + ' ' + y.toFixed(3)).join(' ') + (bank.count === 2 ? ' Z' : '');
  const region = document.querySelector('#bound-region');
  region.setAttribute('d', path); region.setAttribute('fill', bank.count === 2 ? '#bfd7bb' : 'none'); region.setAttribute('stroke-width', bank.count === 2 ? '1.6' : '7');
  const rawEnd = add(w, bank.raw), safeEnd = add(w, accepted);
  point('current-point', w); point('trust-circle', w); label('current-label', w, 'θ' + subscript(run.iteration), -30, 16);
  point('raw-point', rawEnd); line('raw-line', w, rawEnd); label('raw-label', rawEnd, bank.count === 2 ? 'Raw / d₂' : 'Raw / d₁');
  point('safe-point', safeEnd); line('safe-line', w, safeEnd); line('correction-line', rawEnd, add(w, projected));
  show(['extra-line', 'extra-point', 'extra-label'], bank.count === 2);
  if (bank.count === 2) { const prevEnd = add(w, bank.D[0]); line('extra-line', w, prevEnd); point('extra-point', prevEnd); label('extra-label', prevEnd, 'Prev / d₁', 9, 16); }
  const stepLength = Math.hypot(...accepted), converged = stepLength < 1e-4;
  show(['safe-line'], !converged);
  document.querySelector('#trajectory').setAttribute('points', run.path.map(screen).map(p => p.join(',')).join(' '));
  document.querySelector('#trajectory-points').replaceChildren(...run.path.map(p => { const [cx, cy] = screen(p); return svgEl('circle', {cx, cy, r: 2.5}); }));
  document.querySelector('#bank-dimension').textContent = `Iteration ${run.iteration} · bank rank ${bank.count}`;
  document.querySelector('#rate-value').textContent = (Number(rateControl.value) / 100).toFixed(2);
  document.querySelector('#bound-value').textContent = bank.ell.toFixed(2) + '×';
  document.querySelector('#current-loss').textContent = number(SCPOGeometry.loss(w));
  document.querySelector('#current-safety').textContent = 'g = ' + number(bank.g0);
  const distance = Math.hypot(projected[0] - bank.raw[0], projected[1] - bank.raw[1]);
  document.querySelector('#projection-distance').textContent = number(distance);
  document.querySelector('#step-length').textContent = number(stepLength);
  document.querySelector('#step-note').textContent = alpha === 1 ? 'Full projected step accepted' : alpha === 0 ? 'No descent step available' : `Backtracked to α = ${alpha}`;
  const status = document.querySelector('#projection-status');
  status.textContent = converged
    ? 'The certified step has shrunk to zero: the iterate sits on the constraint boundary and the bound admits no further move. Reset to run again.'
    : distance < 1e-6 ? 'The raw step lies inside the certified region and is accepted unchanged.'
    : SCPOGeometry.g(rawEnd) <= 0 ? 'The raw step is truly feasible, but the conservative bound still shortens it.'
    : 'The raw step would leave the feasible set. The projection returns the closest certified update.';
  document.querySelector('#take-step').disabled = converged;
  const xi = SCPOGeometry.coefficients(projected, bank);
  document.querySelector('#candidate-rows').replaceChildren(...bank.D.map((d, i) => {
    const row = document.createElement('tr');
    const title = i === bank.D.length - 1 ? 'Raw step' : 'Previous update';
    [title, number(bank.G[i]) + (bank.G[i] <= 0 ? ' · feasible' : ' · infeasible'), number(xi[i])].forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); });
    return row;
  }));
}
function subscript(n) { return String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]); }
function takeStep() {
  const {accepted} = currentStep();
  if (Math.hypot(...accepted) < 1e-4) return;
  run.w = add(run.w, accepted); run.previous = accepted; run.path.push(run.w.slice()); run.iteration += 1;
  drawRun();
}
function resetRun() {
  run.w = SCPOGeometry.start.slice(); run.previous = null; run.path = [run.w.slice()]; run.iteration = 0;
  drawRun();
}
[rateControl, boundControl].forEach(control => control.addEventListener('input', drawRun));
document.querySelector('#take-step').addEventListener('click', takeStep);
document.querySelector('#reset-run').addEventListener('click', resetRun);
drawRun();

const demonstrations = {
  unsafe: {
    file: 'unsafe-teacher',
    heading: 'Unsafe teacher: a diverging proportional controller',
    label: 'Training with an unsafe teacher',
    description: 'The demonstrator pushes the state away from the origin. Over training, the learned policy moves toward the demonstrator’s behavior, yet every displayed rollout still converges to the origin.'
  },
  safe: {
    file: 'safe-teacher',
    heading: 'Safe teacher: a stabilizing LQR controller',
    label: 'Training with a safe teacher',
    description: 'With a stabilizing LQR demonstrator, the learned policy moves much closer to the teacher. Its rollouts track the demonstrator while still converging to the origin.'
  }
};
const talkVideo = document.querySelector('#talk-video');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduceMotion) { talkVideo.removeAttribute('autoplay'); talkVideo.pause(); }
document.querySelectorAll('[data-demonstration]').forEach(button => {
  button.addEventListener('click', () => {
    if (button.getAttribute('aria-pressed') === 'true') return;
    const demo = demonstrations[button.dataset.demonstration];
    const base = 'assets/animations/' + demo.file;
    talkVideo.pause();
    talkVideo.poster = base + '-poster.png';
    talkVideo.querySelector('source').src = base + '.mp4';
    talkVideo.querySelector('a').href = base + '.gif';
    talkVideo.setAttribute('aria-label', demo.label);
    talkVideo.load();
    if (!reduceMotion) talkVideo.play().catch(() => {});
    document.querySelector('#animation-heading').textContent = demo.heading;
    document.querySelector('#animation-description').textContent = demo.description;
    document.querySelectorAll('[data-demonstration]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
  });
});
// Pause when the page is hidden; resume when it returns.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) talkVideo.pause();
  else if (!reduceMotion) talkVideo.play().catch(() => {});
});

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const code = document.querySelector('#bibtex');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange(); range.selectNodeContents(code);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

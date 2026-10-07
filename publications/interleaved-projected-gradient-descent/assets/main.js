'use strict';
// Affine toy: interleaved training against a fused penalty in policy-output space.
const DEMO = [1.9, 0.5], START = [-0.6, -0.5];
// The recipe: the imitation learning rate decays geometrically from 0.2 to 0.002 over the run, both methods
// share it, and each interleaved safety step uses 0.9 of its stability limit 1/‖H‖.
// The run lasts T = 80 (1 + h)/(1 − h) iterations: parameter sharing h makes H = [[1, h], [h, 1]] worse
// conditioned (eigenvalues 1 ± h), and imitation along its weak direction needs proportionally longer.
const ETA0 = 0.2, ETA_END = 0.002, SAFETY_FRACTION = 0.9, BASE_HORIZON = 80;
const learningRate = t => ETA0 * (ETA_END / ETA0) ** (t / run.T);
// Safe action set: the ellipse u₁²/a₁² + u₂²/a₂² ≤ 1.
const AXES = [1.2, 0.9];
// Euclidean projection onto the ellipse: q = a² ∘ p / (a² + μ), with μ ≥ 0 found by bisection
// so that q lies on the boundary (the KKT conditions of min ‖q − p‖² subject to the ellipse).
function project(p) {
  if ((p[0] / AXES[0]) ** 2 + (p[1] / AXES[1]) ** 2 <= 1) return p.slice();
  const gap = mu => (AXES[0] * p[0] / (AXES[0] ** 2 + mu)) ** 2 + (AXES[1] * p[1] / (AXES[1] ** 2 + mu)) ** 2 - 1;
  let lo = 0, hi = 1;
  while (gap(hi) > 0) hi *= 2;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (gap(mid) > 0) lo = mid; else hi = mid; }
  return p.map((x, i) => AXES[i] ** 2 * x / (AXES[i] ** 2 + hi));
}
const OPTIMUM = project(DEMO), DEMO_GAP = Math.hypot(DEMO[0] - OPTIMUM[0], DEMO[1] - OPTIMUM[1]);
const lambdaControl = document.querySelector('#lambda'), couplingControl = document.querySelector('#coupling');
const playButton = document.querySelector('#play-toggle');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const screen = v => [186 + 115 * v[0], 220 - 115 * v[1]];
// Magnified inset centred on v*, where the safety steps act.
const ZOOM = 4, INSET_CENTER = [394, 352];
const zoomed = v => [INSET_CENTER[0] + 115 * ZOOM * (v[0] - OPTIMUM[0]), INSET_CENTER[1] - 115 * ZOOM * (v[1] - OPTIMUM[1])];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const norm = v => Math.hypot(v[0], v[1]);
const number = x => (Math.abs(x) < 0.0005 ? 0 : x).toFixed(3);
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const $ = id => document.getElementById(id);
const subscript = n => String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]);

const settings = () => {
  const lambda = 10 ** Number(lambdaControl.value), h = Number(couplingControl.value) / 100;
  // Safety budget of Algorithm 1 with c = 1: K_t = ⌈ln(t + 2)⌉ safety steps at outer iteration t.
  const K = t => Math.max(1, Math.ceil(Math.log(t + 2)));
  // Fused step: the shared learning rate, or that rate divided by 1 + λ for stability. The fused update is
  // stable while η_f (1 + λ)‖H‖ < 2 (Section IV), with ‖H‖ = 1 + h.
  const normalized = document.querySelector('[data-fused-step="normalized"]').getAttribute('aria-pressed') === 'true';
  const etaFused = t => learningRate(t) / (normalized ? 1 + lambda : 1);
  return {lambda, h, K, beta: SAFETY_FRACTION / (1 + h), normalized, etaFused, stability: t => etaFused(t) * (1 + lambda) * (1 + h)};
};
// H v for the NTK Gram matrix H = [[1, h], [h, 1]].
const applyH = (v, h) => [v[0] + h * v[1], h * v[0] + v[1]];
const scaled = (v, k) => [k * v[0], k * v[1]];

const DIVERGED = 5;  // distance from the origin beyond which fused training counts as diverged
const horizon = h => Math.round(BASE_HORIZON * (1 + h) / (1 - h));
const fresh = () => ({T: horizon(settings().h), fusedDiverged: 0, iteration: 0, corrected: 0, inter: START.slice(), fused: START.slice(), interPath: [START], fusedPath: [START], interSteps: 0, fusedSteps: 0});
const run = fresh();
// anim: the iteration being animated, or null. playing: whether iterations keep following each other.
let anim = null, playing = false, frame = null;

// One outer iteration, split into the stages the animation shows: the imitation step, then each safety step.
function plan() {
  const s = settings(), stages = [];
  const K = s.K(run.iteration);
  let v = sub(run.inter, applyH(scaled(sub(run.inter, DEMO), learningRate(run.iteration)), s.h));
  stages.push({to: v, kind: 'imitation'});
  for (let k = 0; k < K; k++) {
    const target = project(v), active = norm(sub(v, target)) > 1e-9;  // refreshed projection target
    v = sub(v, applyH(scaled(sub(v, target), s.beta), s.h));
    stages.push({to: v, kind: 'safety', target, active, index: k + 1, of: K});
  }
  const w = run.fused, grad = sub(w, DEMO), pen = sub(w, project(w));
  let fusedTo = run.fusedDiverged ? w : sub(w, applyH(scaled([grad[0] + s.lambda * pen[0], grad[1] + s.lambda * pen[1]], s.etaFused(run.iteration)), s.h));
  // A fused step that leaves the plot by far has diverged: stop it at the plot edge, along its direction.
  const diverges = !run.fusedDiverged && !(norm(fusedTo) < DIVERGED);
  if (diverges) fusedTo = Number.isFinite(norm(fusedTo)) ? scaled(fusedTo, DIVERGED / norm(fusedTo)) : w;
  return {stages, fusedFrom: w, fusedTo, diverges, corrects: stages.some(x => x.active)};
}
function commitStage(p, i) { run.inter = p.stages[i].to; run.interPath.push(run.inter); run.interSteps += 1; }
function commitIteration(p) {
  run.corrected += p.corrects ? 1 : 0;
  if (!run.fusedDiverged) { run.fused = p.fusedTo; run.fusedPath.push(run.fused); run.fusedSteps += 1; }
  if (p.diverges) run.fusedDiverged = run.iteration + 1;
  run.iteration += 1;
}
function iterate() { const p = plan(); p.stages.forEach((_, i) => commitStage(p, i)); commitIteration(p); }

function place(id, v, attrs = ['cx', 'cy'], map = screen) { const [x, y] = map(v); $(id).setAttribute(attrs[0], x); $(id).setAttribute(attrs[1], y); }
function segment(id, a, b, map = screen) { const [x1, y1] = map(a), [x2, y2] = map(b); Object.entries({x1, y1, x2, y2}).forEach(([k, v]) => $(id).setAttribute(k, v)); }
function polyline(id, path, map = screen) { $(id).setAttribute('points', path.map(map).map(p => p.map(x => x.toFixed(2)).join(',')).join(' ')); }
// Draw the moving parts in the main plot and the inset.
function scene(interPath, inter, fusedPath, fused, target) {
  polyline('inter-path', interPath); polyline('fused-path', fusedPath);
  place('inter-point', inter); place('fused-point', fused);
  polyline('inset-inter-path', interPath.slice(-60), zoomed); polyline('inset-fused-path', fusedPath.slice(-60), zoomed);
  place('inset-inter', inter, undefined, zoomed); place('inset-fused', fused, undefined, zoomed);
  show(['target-point', 'target-line', 'inset-target', 'inset-target-line'], Boolean(target));
  if (target) {
    place('target-point', target); segment('target-line', inter, target);
    place('inset-target', target, undefined, zoomed); segment('inset-target-line', inter, target, zoomed);
  }
}
const show = (ids, visible) => ids.forEach(id => { $(id).style.display = visible ? '' : 'none'; });

// Static scenery.
(() => {
  const [cx, cy] = screen([0, 0]);
  Object.entries({cx, cy, rx: 115 * AXES[0], ry: 115 * AXES[1]}).forEach(([k, v]) => $('safe-set').setAttribute(k, v));
  place('safe-label', [-0.8, 0.5], ['x', 'y']);
  place('demo-point', DEMO); place('optimum-point', OPTIMUM); place('start-point', START);
  segment('demo-link', DEMO, OPTIMUM);
  const [icx, icy] = zoomed([0, 0]);
  Object.entries({cx: icx, cy: icy, rx: 115 * ZOOM * AXES[0], ry: 115 * ZOOM * AXES[1]}).forEach(([k, v]) => $('inset-set').setAttribute(k, v));
  place('inset-optimum', OPTIMUM, undefined, zoomed);
  const [dx, dy] = screen(DEMO), [ox, oy] = screen(OPTIMUM);
  $('demo-label').setAttribute('x', dx); $('demo-label').setAttribute('y', dy - 14);
  $('optimum-label').setAttribute('x', ox - 10); $('optimum-label').setAttribute('y', oy - 10);
  const [sx, sy] = screen(START); $('start-label').setAttribute('x', sx + 8); $('start-label').setAttribute('y', sy + 16);
})();

function draw() {
  const s = settings();
  scene(run.interPath, run.inter, run.fusedPath, run.fused, null);
  $('phase-label').textContent = run.iteration ? `Iteration ${run.iteration}` : 'Iteration 0';
  $('iteration').textContent = run.iteration;
  $('lambda-value').textContent = String(Number(s.lambda.toPrecision(2)));
  $('coupling-value').textContent = s.h.toFixed(2);
  const steps = k => `${k} safety step${k === 1 ? '' : 's'}`;
  $('iteration-label').textContent = `Outer iteration t of ${run.T}`;
  $('iteration-note').textContent = run.iteration >= run.T ? `Last block: ${steps(s.K(run.T - 1))}` : `Next block: K${subscript(run.iteration)} = ${steps(s.K(run.iteration))}`;
  // Rates for the iteration about to run (the last one once the run is over).
  const t = Math.min(run.iteration, run.T - 1), stability = s.stability(t);
  $('learning-rate').textContent = number(learningRate(t));
  $('fused-eta-note').textContent = s.normalized
    ? `Fused uses η/(1 + λ) = ${s.etaFused(t).toExponential(1)}`
    : `Fused: η(1 + λ)‖H‖ = ${stability.toFixed(2)}${stability < 2 ? ' < 2, stable' : ' > 2, overshoots'}`;
  // The fused fixed point solves (v − ū) + λ(v − Π(v)) = 0 for any full-rank H. Then Π(v) = v* and
  // v = v* + (ū − v*)/(1 + λ): it sits ‖ū − v*‖/(1 + λ) outside the safe set.
  $('fused-bias').textContent = number(DEMO_GAP / (1 + s.lambda));
  $('kappa-s').textContent = number(s.h / Math.hypot(1, s.h));
  for (const [key, v] of Object.entries({inter: run.inter, fused: run.fused})) {
    const diverged = key === 'fused' && run.fusedDiverged;
    $(key + '-dc').textContent = diverged ? 'diverged' : number(norm(sub(v, project(v))));
    $(key + '-dv').textContent = diverged ? 'diverged' : number(norm(sub(v, OPTIMUM)));
  }
  show(['fused-point', 'inset-fused'], !run.fusedDiverged);
  $('inter-steps').textContent = run.interSteps; $('fused-steps').textContent = run.fusedSteps;
  if (!anim) $('toy-status').textContent = statusText();
  const done = run.iteration >= run.T;
  playButton.disabled = done; $('step-once').disabled = done || playing; $('run-end').disabled = done;
  playButton.textContent = playing ? 'Pause' : run.iteration && !done ? 'Resume' : 'Play animation';
  // Narration changes several times a second while playing; announce only the settled summary.
  $('toy-status').setAttribute('aria-live', playing || anim ? 'off' : 'polite');
  playButton.setAttribute('aria-pressed', String(playing));
}

function statusText() {
  if (run.iteration === 0) return 'Press Play animation to watch both policies train from the same start, one step at a time.';
  const s = settings(), lam = Number(s.lambda.toPrecision(2));
  const fusedOut = norm(sub(run.fused, project(run.fused))), fusedOpt = norm(sub(run.fused, OPTIMUM)), interOpt = norm(sub(run.inter, OPTIMUM));
  const interOut = norm(sub(run.inter, project(run.inter)));
  if (run.iteration < 15) return 'Both policies first move toward the demonstrations, which lie outside the safe set.';
  const parts = [];
  const stability = s.stability(Math.min(run.iteration, run.T - 1));
  if (run.fusedDiverged) parts.push(`Fused training diverged at iteration ${run.fusedDiverged}: with the shared step, λ = ${lam} and parameter sharing h = ${s.h.toFixed(2)} give η(1 + λ)‖H‖ = ${s.stability(run.fusedDiverged - 1).toFixed(1)}, far above the stability limit of 2, and each penalty step overshoots across the whole safe set.`);
  else if (!s.normalized && stability > 2 && fusedOpt > 0.02) parts.push(`With the shared step, λ = ${lam} makes each penalty step overshoot (η(1 + λ)‖H‖ = ${stability.toFixed(1)} > 2): every correction throws the actions deep into the safe set, so fused training chatters at the boundary, ${number(fusedOpt)} from v*.`);
  else if (fusedOut > 0.002) parts.push(`Fused training settles ${number(fusedOut)} outside the safe set: at a finite λ, the constrained optimum is not a stationary point.`);
  else if (s.normalized && fusedOpt > 0.02) parts.push(`Normalizing the step to η/(1 + λ) keeps fused training stable, but its imitation step is ${Number((1 + s.lambda).toPrecision(2))}× smaller, so after ${run.iteration} iterations it is still ${number(fusedOpt)} from v*.`);
  else if (fusedOpt > 0.02) parts.push(`Fused training is still ${number(fusedOpt)} from v*.`);
  else parts.push('Fused training is close to v* here.');
  if (interOpt < 0.005 && interOut < 0.002) {
    parts.push('Interleaving reaches v* and stays safe, without a weight to tune.');
    const cond = (1 + s.h) / (1 - s.h);
    if (cond >= 2.5) parts.push(`Parameter sharing h = ${s.h.toFixed(2)} makes the problem ${cond.toFixed(1)}× worse conditioned, so the run lasts ${run.T} iterations instead of ${BASE_HORIZON}.`);
  }
  else parts.push('Interleaving is still closing in on v*.');
  return parts.join(' ');
}

// Animation: each stage is tweened, and the fused point moves across the whole iteration.
// Safety steps that change nothing (actions already safe) are skipped. Iterations before the first
// correction pass quickly; the first corrected ones play slowly so the zigzag at the boundary is
// visible; later ones speed up until the run fast-forwards to the end.
function iterationDuration(p) {
  if (!p.corrects) return 260;
  const seen = run.corrected;
  return seen < 3 ? 1400 : 1400 * 0.8 ** (seen - 2);
}
const visible = stage => stage.kind === 'imitation' || stage.active;
function beginIteration(duration) {
  const p = plan(), shown = p.stages.filter(visible).length;
  anim = {p, stage: 0, start: performance.now(), duration: (duration ?? iterationDuration(p)) / shown, from: run.inter};
  narrate();
}
function narrate() {
  const stage = anim.p.stages[anim.stage], n = run.iteration + 1;
  $('toy-status').textContent = stage.kind === 'imitation'
    ? `Iteration ${n}, imitation step: both policies move toward the demonstrations. Fused training spreads its one combined step over the whole iteration.`
    : `Iteration ${n}, safety step ${stage.index} of ${stage.of}: the projection of the current actions is recomputed (hollow green marker), and the actions move toward it.`;
}
function tick(now) {
  frame = null;
  if (!anim) return;
  const {p} = anim, stage = p.stages[anim.stage], n = p.stages.length;
  const t = Math.min(1, (now - anim.start) / anim.duration), e = ease(t);
  const point = lerp(anim.from, stage.to, e);
  const fusedPoint = lerp(p.fusedFrom, p.fusedTo, (anim.stage + e) / n);
  const safety = stage.kind === 'safety';
  scene([...run.interPath, point], point, [...run.fusedPath, fusedPoint], fusedPoint, safety ? stage.target : null);
  $('phase-label').textContent = `Iteration ${run.iteration + 1} · ` + (safety ? `safety step ${stage.index}/${stage.of}` : 'imitation step');
  if (t < 1) { frame = requestAnimationFrame(tick); return; }
  commitStage(p, anim.stage);
  // Commit the safety steps that change nothing without animating them.
  while (anim.stage + 1 < n && !visible(p.stages[anim.stage + 1])) commitStage(p, ++anim.stage);
  if (anim.stage + 1 < n) {
    Object.assign(anim, {stage: anim.stage + 1, start: now, from: run.inter});
    narrate();
    frame = requestAnimationFrame(tick);
    return;
  }
  commitIteration(p);
  anim = null;
  if (playing && run.iteration < run.T) {
    const next = plan();
    if (next.corrects && iterationDuration(next) < 80) frame = requestAnimationFrame(fastForward);
    else { beginIteration(); frame = requestAnimationFrame(tick); }
  } else playing = false;
  draw();
}
// Once corrections are routine, show whole iterations: two per frame until the end.
function fastForward() {
  frame = null;
  // Long runs (strong parameter sharing) fast-forward proportionally faster, so every run ends in a few seconds.
  const perFrame = Math.max(2, Math.ceil(run.T / 100));
  for (let i = 0; i < perFrame && run.iteration < run.T; i++) iterate();
  if (run.iteration >= run.T) playing = false;
  else if (playing) frame = requestAnimationFrame(fastForward);
  draw();
}
// Finish the iteration in flight instantly, so pausing never leaves a half-applied update.
function settle() {
  if (frame) cancelAnimationFrame(frame);
  frame = null;
  if (anim) { for (let i = anim.stage; i < anim.p.stages.length; i++) commitStage(anim.p, i); commitIteration(anim.p); anim = null; }
}
function play() {
  if (run.iteration >= run.T) return;
  if (reduceMotion.matches) { runToEnd(); return; }
  playing = true;
  if (!anim) beginIteration();
  draw();
  if (!frame) frame = requestAnimationFrame(tick);
}
function pause() { playing = false; settle(); draw(); }
function stepOnce() {
  if (playing || anim || run.iteration >= run.T) return;
  if (reduceMotion.matches) { iterate(); draw(); return; }
  beginIteration(1000);
  draw();
  frame = requestAnimationFrame(tick);
}
function runToEnd() { playing = false; settle(); while (run.iteration < run.T) iterate(); draw(); }
function resetRun() { playing = false; if (frame) cancelAnimationFrame(frame); frame = null; anim = null; Object.assign(run, fresh()); draw(); }

playButton.addEventListener('click', () => (playing ? pause() : play()));
$('step-once').addEventListener('click', stepOnce);
$('run-end').addEventListener('click', runToEnd);
$('reset-run').addEventListener('click', resetRun);
document.querySelectorAll('[data-fused-step]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-fused-step]').forEach(other => {
    other.setAttribute('aria-pressed', String(other === button));
    other.classList.toggle('active', other === button);
  });
  resetRun();
}));
// Changing a setting restarts the run so both paths reflect one configuration.
[lambdaControl, couplingControl].forEach(control => control.addEventListener('input', resetRun));
draw();

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

// Race-track toy: fused vs interleaved training of a policy's first action on a fixed set of
// states. Data from web/scripts/make_track_toy.py (window.TRACK_TOY). Everything below runs in
// scaled action coordinates o = u / u_max, where each state's cost metric M0 is well conditioned.
(() => {
  'use strict';
  const D = window.TRACK_TOY;
  const UM = D.u_max;
  const NS = 'http://www.w3.org/2000/svg';
  const C = { fused: '#c47f00', inter: '#0b63a8', label: '#cc2f2b', vstar: '#16202a', safe: '#2f8a4e', band: '#9aa6b1' };
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ data in scaled coordinates
  const scaleM = (M) => [[M[0][0] * UM[0] * UM[0], M[0][1] * UM[0] * UM[1]], [M[1][0] * UM[1] * UM[0], M[1][1] * UM[1] * UM[1]]];
  const states = D.states.map((s, i) => ({
    i, s: s.s, ey: s.ey, epsi: s.epsi, v: s.v, xy: s.xy, x: s.input,
    M: scaleM(s.M0),
    ubar: [s.u_bar[0] / UM[0], s.u_bar[1] / UM[1]],
    vstar: [s.v_star[0] / UM[0], s.v_star[1] / UM[1]],
    poly: s.poly.map((p) => [p[0] / UM[0], p[1] / UM[1]]),
    A: s.A.map((a) => [a[0] * UM[0], a[1] * UM[1]]), b: s.b,
    pathP: s.path_p, pathK: s.path_K, violation: s.label_violation,
  }));
  const n = states.length;
  const mScale = states.reduce((acc, s) => acc + (s.M[0][0] + s.M[1][1]) / 2, 0) / n;
  states.forEach((s) => { s.M = s.M.map((r) => r.map((v) => v / mScale)); });

  const quad = (M, d) => d[0] * (M[0][0] * d[0] + M[0][1] * d[1]) + d[1] * (M[1][0] * d[0] + M[1][1] * d[1]);
  const mvec = (M, d) => [M[0][0] * d[0] + M[0][1] * d[1], M[1][0] * d[0] + M[1][1] * d[1]];

  function inside(s, o) {
    for (let k = 0; k < s.A.length; k++) if (s.A[k][0] * o[0] + s.A[k][1] * o[1] > s.b[k] + 1e-12) return false;
    return true;
  }
  // M-metric projection onto the convex polygon: the closest point of the boundary if o is outside.
  function project(s, o) {
    if (inside(s, o)) return o.slice();
    let best = null, bd = Infinity;
    const P = s.poly;
    for (let k = 0; k < P.length; k++) {
      const a = P[k], c = P[(k + 1) % P.length], d = [c[0] - a[0], c[1] - a[1]];
      const w = [o[0] - a[0], o[1] - a[1]];
      const Md = mvec(s.M, d);
      let t = (Md[0] * w[0] + Md[1] * w[1]) / (Md[0] * d[0] + Md[1] * d[1]);
      t = Math.max(0, Math.min(1, t));
      const p = [a[0] + t * d[0], a[1] + t * d[1]];
      const dd = quad(s.M, [p[0] - o[0], p[1] - o[1]]);
      if (dd < bd) { bd = dd; best = p; }
    }
    return best;
  }
  // predicted lateral offsets e_y(k), k = 0..N, when the first action is o and the LTV controller takes over
  function lateral(s, o) {
    const u = [o[0] * UM[0], o[1] * UM[1]];
    return s.pathP.map((p, k) => p[1] + s.pathK[k][1][0] * u[0] + s.pathK[k][1][1] * u[1]);
  }

  // how far (m) the predicted path of first action o leaves the band
  const bandViolation = (s, o) => Math.max(...lateral(s, o).slice(1).map((v) => Math.abs(v))) - D.geometry.band;

  // ------------------------------------------------------------------ policy network (6 -> 64 -> 64 -> 2, ReLU)
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  }
  const H = 64;
  function initNet(seed) {
    const g = rng(seed), mat = (r, c, sd) => Array.from({ length: r }, () => Array.from({ length: c }, () => g() * sd));
    return { W1: mat(6, H, Math.SQRT2 / Math.sqrt(6)), b1: new Array(H).fill(0), W2: mat(H, H, Math.SQRT2 / Math.sqrt(H)),
             b2: new Array(H).fill(0), W3: mat(H, 2, 0.3 / Math.sqrt(H)), b3: [0, 0] };
  }
  const cloneNet = (w) => JSON.parse(JSON.stringify(w));
  function forward(w) {
    const out = [], cache = [];
    for (const s of states) {
      const h1 = w.b1.slice(), h2 = w.b2.slice();
      for (let j = 0; j < H; j++) { for (let i = 0; i < 6; i++) h1[j] += s.x[i] * w.W1[i][j]; if (h1[j] < 0) h1[j] = 0; }
      for (let j = 0; j < H; j++) { for (let i = 0; i < H; i++) h2[j] += h1[i] * w.W2[i][j]; if (h2[j] < 0) h2[j] = 0; }
      const o = w.b3.slice();
      for (let i = 0; i < H; i++) { o[0] += h2[i] * w.W3[i][0]; o[1] += h2[i] * w.W3[i][1]; }
      out.push(o); cache.push([h1, h2]);
    }
    return { out, cache };
  }
  // one gradient step given dL/do for every state
  // gradient buffers, allocated once and reused by every step (avoids garbage-collection pauses)
  const GB = { W1: Array.from({ length: 6 }, () => new Float64Array(H)), b1: new Float64Array(H),
               W2: Array.from({ length: H }, () => new Float64Array(H)), b2: new Float64Array(H),
               W3: Array.from({ length: H }, () => new Float64Array(2)), b3: new Float64Array(2),
               gh1: new Float64Array(H), gh2: new Float64Array(H) };
  function step(w, cache, gout, lr) {
    const { W1: gW1, b1: gb1, W2: gW2, b2: gb2, W3: gW3, b3: gb3, gh1, gh2 } = GB;
    for (const G of [gW1, gW2, gW3]) for (const r of G) r.fill(0);
    gb1.fill(0); gb2.fill(0); gb3.fill(0);
    states.forEach((s, n_) => {
      const [h1, h2] = cache[n_], go = gout[n_];
      gb3[0] += go[0]; gb3[1] += go[1];
      for (let i = 0; i < H; i++) {
        gW3[i][0] += h2[i] * go[0]; gW3[i][1] += h2[i] * go[1];
        gh2[i] = h2[i] > 0 ? w.W3[i][0] * go[0] + w.W3[i][1] * go[1] : 0;
      }
      gh1.fill(0);
      for (let j = 0; j < H; j++) {
        if (gh2[j] === 0) continue;
        gb2[j] += gh2[j];
        for (let i = 0; i < H; i++) { gW2[i][j] += h1[i] * gh2[j]; gh1[i] += w.W2[i][j] * gh2[j]; }
      }
      for (let j = 0; j < H; j++) {
        if (h1[j] <= 0) continue;
        gb1[j] += gh1[j];
        for (let i = 0; i < 6; i++) gW1[i][j] += s.x[i] * gh1[j];
      }
    });
    // clip the global gradient norm to 1, as in the paper's runs
    let sq = gb3[0] ** 2 + gb3[1] ** 2;
    for (const G of [gW1, gW2, gW3]) for (const r of G) for (let j = 0; j < r.length; j++) sq += r[j] * r[j];
    for (let j = 0; j < H; j++) sq += gb1[j] ** 2 + gb2[j] ** 2;
    lr *= Math.min(1, 1 / Math.sqrt(sq + 1e-30));
    const upd = (W, G) => { for (let i = 0; i < W.length; i++) { const r = W[i], g = G[i]; for (let j = 0; j < r.length; j++) r[j] -= lr * g[j]; } };
    upd(w.W1, gW1); upd(w.W2, gW2); upd(w.W3, gW3);
    for (let j = 0; j < H; j++) { w.b1[j] -= lr * gb1[j]; w.b2[j] -= lr * gb2[j]; }
    w.b3[0] -= lr * gb3[0]; w.b3[1] -= lr * gb3[1];
  }

  // gradients of J = mean 1/2 ||o - ubar||_M^2 and of the regression onto targets
  const gradJ = (out) => out.map((o, i) => mvec(states[i].M, [o[0] - states[i].ubar[0], o[1] - states[i].ubar[1]]).map((v) => v / n));
  const gradR = (out, tgt) => out.map((o, i) => mvec(states[i].M, [o[0] - tgt[i][0], o[1] - tgt[i][1]]).map((v) => v / n));
  const targetsOf = (out) => out.map((o, i) => project(states[i], o));

  // ------------------------------------------------------------------ the two training runs
  const DEFAULTS = { lam: 10, normalized: true, eta: 0.2, beta: 0.4, c: 2, refresh: true, speed: 0 };   // speed: log2 of the playback multiplier
  const show = { fused: true, inter: true };   // which methods are drawn (both always train)
  const params = { ...DEFAULTS };
  const TRAIL = 50, TRAIL_EVERY = 3;   // trail: one point every 3 iterations
  let runs, t, playing = false, timer = null;

  function newRun(kind, w0) {
    const w = cloneNet(w0), f = forward(w);
    return { kind, w, out: f.out, cache: f.cache, diverged: false, trails: f.out.map((o) => [o.slice()]), hist: [], ring: [f.out.map((o) => o.slice())] };
  }
  function reset() {
    const w0 = initNet(7);
    runs = { fused: newRun('fused', w0), inter: newRun('inter', w0) };
    t = 0; anim = null; acc = 0; stopAfterMove = false; trailsStarted = false; shadow = null; record(); if (fig) { fig.items.forEach((it) => { it.view = null; }); draw(true); }
  }
  // one training iteration; `onSub(kind, label, out)` (optional) sees the outputs after every sub-step
  function iterateRun(r, onSub, tt = t) {
    if (r.diverged) return;
    const K = Math.ceil(params.c * Math.log(tt + 2));
    if (r.kind === 'fused') {
      const tg = targetsOf(r.out), gj = gradJ(r.out), gr = gradR(r.out, tg);
      const lr = params.normalized ? params.eta / (1 + params.lam) : params.eta;
      step(r.w, r.cache, gj.map((g, i) => [g[0] + params.lam * gr[i][0], g[1] + params.lam * gr[i][1]]), lr);
      if (onSub) onSub('fused', 'fused step', forward(r.w).out);
    } else {
      step(r.w, r.cache, gradJ(r.out), params.eta);                     // performance step
      let f = forward(r.w), tg = targetsOf(f.out);
      if (onSub) onSub('perf', 'performance step', f.out);
      for (let k = 0; k < K; k++) {                                      // safety block
        if (k > 0) f = forward(r.w);
        if (params.refresh && k > 0) tg = targetsOf(f.out);
        step(r.w, f.cache, gradR(f.out, tg), params.beta);
        if (onSub) onSub('safety', `safety step ${k + 1} of ${K}`, forward(r.w).out);
      }
    }
    const f = forward(r.w);
    r.out = f.out; r.cache = f.cache;
    if (r.out.some((o) => !isFinite(o[0]) || !isFinite(o[1]) || Math.abs(o[0]) > 50 || Math.abs(o[1]) > 50)) r.diverged = true;
  }
  const pushTrail = (r, out) => out.forEach((o, i) => { r.trails[i].push(o.slice()); if (r.trails[i].length > TRAIL) r.trails[i].shift(); });
  function metrics(out) {
    let dv = 0, dc = 0, unsafe = 0;
    out.forEach((o, i) => {
      const s = states[i], p = project(s, o);
      dv += quad(s.M, [o[0] - s.vstar[0], o[1] - s.vstar[1]]);
      dc += quad(s.M, [o[0] - p[0], o[1] - p[1]]);
      if (bandViolation(s, o) > 0.005) unsafe += 1;            // leaves the band by more than 5 mm
    });
    return { dv: Math.sqrt(dv / n), dc: Math.sqrt(dc / n), unsafe };
  }
  const HIST_MAX = 3000;     // beyond this, every other stored point is dropped (the plot draws at most ~400)
  function record() {
    for (const r of Object.values(runs)) {
      r.hist.push({ t, m: r.diverged ? null : metrics(r.out) });
      if (r.hist.length > HIST_MAX) r.hist = r.hist.filter((_, j) => j % 2 === 0 || j === r.hist.length - 1);
    }
  }
  // a plain iteration of both methods (after the detailed opening)
  // positions over the last TREND iterations, for the trend arrows
  function noteTrend() {
    for (const r of Object.values(runs)) { r.ring.push(copy(r.out)); if (r.ring.length > TREND + 1) r.ring.shift(); }
  }
  // mean of ring entries [a, b): removes the step-to-step oscillation of the fused method
  const ringMean = (r, a, b) => { const sl = r.ring.slice(a, b); return sl[0].map((_, i) => [sl.reduce((m, o) => m + o[i][0], 0) / sl.length, sl.reduce((m, o) => m + o[i][1], 0) / sl.length]); };
  function iteration() {
    iterateRun(runs.fused); iterateRun(runs.inter); t += 1; record(); noteTrend();
    if (t % TRAIL_EVERY === 0) for (const r of Object.values(runs)) if (!r.diverged) pushTrail(r, ringMean(r, -SMOOTH));
  }

  // ---- playback tempo
  // Iterations per frame rise smoothly (log-linearly, with a smooth start and finish) from R0 to R1
  // between iterations RAMP_T0 and RAMP_T1, times the speed multiplier 2^params.speed. The duration of each iteration follows from
  // the tempo; how much of it is shown depends on how much time there is:
  //   level 1: performance step, pause, each of the K_t safety steps, pause
  //   level 2: performance step, pause, the result of the K_t safety steps, pause
  //   level 3: the result of the whole iteration
  //   faster than one iteration per frame: several iterations per frame, shown as a smoothed trend
  const R0 = 1 / 200, R1 = 2, RAMP_T0 = 1, RAMP_T1 = 35, TREND = 30, SMOOTH = 3;
  const LEVEL2_BELOW = 160, LEVEL3_BELOW = 55;          // frames per iteration at which the display coarsens
  const DESC = {
    perf: 'One gradient step on the imitation loss. The nine actions share one network, so they move together rather than each straight to its label ✕.',
    safety: 'One step toward the projections of the current actions (open circles). All nine actions move, including those that are already safe.',
    block: 'The K_t safety steps in one go, along their zig-zag path. As training speeds up, the path fades into a straight arrow to their combined result.',
    whole: 'One whole iteration: the performance step, then the safety block. As training speeds up, the bent path straightens into one arrow per iteration.',
  };
  let anim = null, acc = 0, stopAfterMove = false;
  const copy = (out) => out.map((o) => o.slice());
  function tempo(tt) {
    const u = Math.min(1, Math.max(0, (tt - RAMP_T0) / (RAMP_T1 - RAMP_T0))), g = u * u * (3 - 2 * u);
    return Math.pow(2, params.speed) * Math.exp(Math.log(R0) + (Math.log(R1) - Math.log(R0)) * g);
  }
  const levelOf = (framesPerIter) => (framesPerIter >= LEVEL2_BELOW ? 1 : framesPerIter >= LEVEL3_BELOW ? 2 : 3);
  const trendMode = () => !anim && tempo(t) >= 1;
  // one iteration of both methods, recorded sub-step by sub-step, turned into timed stages
  // ---- look-ahead for the windows' zoom: a shadow copy of both runs stays a few iterations ahead of the
  // display and records, per future iteration, method and state, the box its actions pass through. It
  // advances a little every frame, so no single frame pays for the whole look-ahead.
  let shadow = null;
  const lookLen = () => Math.min(60, Math.max(6, Math.ceil(tempo(t) * 60)));
  const realT = () => t + (anim ? 1 : 0);              // the iteration the real runs are about to take
  function cloneRun(r) { const w = cloneNet(r.w), f = forward(w); return { kind: r.kind, w, out: f.out, cache: f.cache, diverged: r.diverged }; }
  function resetShadow() { shadow = { runs: { fused: cloneRun(runs.fused), inter: cloneRun(runs.inter) }, t: realT(), boxes: [] }; }
  function advanceShadow(budget) {
    if (!shadow || shadow.t < realT()) resetShadow();
    const target = realT() + lookLen();
    for (let j = 0; j < budget && shadow.t < target; j++) {
      const box = {};
      for (const k of ['fused', 'inter']) {
        const b = states.map(() => [Infinity, -Infinity, Infinity, -Infinity]), sr = shadow.runs[k];
        iterateRun(sr, (kind, label, out) => out.forEach((o, i) => {
          if (Math.abs(o[0]) > 1.5 || Math.abs(o[1]) > 1.5) return;
          b[i][0] = Math.min(b[i][0], o[0]); b[i][1] = Math.max(b[i][1], o[0]); b[i][2] = Math.min(b[i][2], o[1]); b[i][3] = Math.max(b[i][3], o[1]);
        }), shadow.t);
        box[k] = b;
      }
      shadow.boxes.push({ t: shadow.t, box }); shadow.t += 1;
    }
    while (shadow.boxes.length && shadow.boxes[0].t < t) shadow.boxes.shift();
  }
  let trailsStarted = false;
  function buildIteration() {
    advanceShadow(3);
    const D = 1 / tempo(t), level = levelOf(D), K = Math.ceil(params.c * Math.log(t + 2));
    if (level === 3 && !trailsStarted) { trailsStarted = true; for (const r of Object.values(runs)) r.trails = r.out.map((o) => [o.slice()]); }
    const f0 = copy(runs.fused.out), i0 = copy(runs.inter.out), fs = [], is = [];
    iterateRun(runs.fused, (kind, label, out) => fs.push(copy(out)));
    iterateRun(runs.inter, (kind, label, out) => is.push({ kind, label, out: copy(out) }));
    const fEnd = fs.length ? fs[fs.length - 1] : f0, iPerf = is.length ? is[0].out : i0, iEnd = is.length ? is[is.length - 1].out : i0;
    // a move follows `path` (output snapshots); alpha in [0, 1] fades its wiggly path into the straight line
    const still = { path: [fEnd, fEnd] }, logf = (x) => Math.log(Math.max(x, 1));
    const a2 = Math.min(1, Math.max(0, logf(LEVEL2_BELOW / D) / logf(LEVEL2_BELOW / LEVEL3_BELOW)));
    const a3 = Math.min(1, Math.max(0, logf(LEVEL3_BELOW / D) / logf(LEVEL3_BELOW)));
    const plan = [];                                      // [weight, stage]
    const move = (label, desc, inter, fused, alpha) => ({ move: true, label, desc, inter, fused, alpha: alpha || 0 });
    const safetyPath = [iPerf, ...is.slice(1).map((x) => x.out)];
    if (level === 3) plan.push([1, move('One iteration', DESC.whole, { path: [i0, iPerf, iEnd] }, { path: [f0, fEnd] }, a3)]);
    else {
      plan.push([2, move('Performance step', DESC.perf + ' Fused takes its single combined step meanwhile.', { path: [i0, iPerf] }, { path: [f0, fEnd] })]);
      plan.push([1, null]);
      if (level === 1) {
        let prev = iPerf;
        is.slice(1).forEach((sub) => { plan.push([1, move(`Safety step ${sub.label.split(' ').slice(2).join(' ')}`, DESC.safety, { path: [prev, sub.out] }, still)]); prev = sub.out; });
      } else plan.push([2, move(`Safety block: ${K} steps`, DESC.block, { path: safetyPath }, still, a2)]);
      plan.push([1, null]);
    }
    const W = plan.reduce((a, p) => a + p[0], 0);
    let last = null;
    const stages = plan.map(([w, st]) => {
      const frames = Math.max(1, Math.round(D * w / W));
      if (st) { last = st; return { ...st, frames }; }
      return { ...last, move: false, frames, pause: true };
    });
    return { stages, idx: 0, f: 0, level };
  }
  const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
  // what is drawn: animated positions during a move, the end of the last move during a pause
  // position at fraction e of a polyline (by length)
  function along(pts, e) {
    const L = []; let tot = 0;
    for (let j = 1; j < pts.length; j++) { tot += Math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1]); L.push(tot); }
    if (tot < 1e-12) return { p: pts[pts.length - 1].slice(), upto: pts.slice() };
    const target = e * tot; let j = 0;
    while (j < L.length - 1 && L[j] < target) j++;
    const L0 = j ? L[j - 1] : 0, u = (target - L0) / Math.max(L[j] - L0, 1e-12);
    const p = [pts[j][0] + (pts[j + 1][0] - pts[j][0]) * u, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * u];
    return { p, upto: [...pts.slice(0, j + 1), p] };
  }
  // what is drawn for method k: per state the dot, the wiggly path so far, and the straight line so far
  function shown(k) {
    if (!anim) return { out: runs[k].out, moves: null };
    const st = anim.stages[anim.idx], path = st[k].path, e = st.pause ? 1 : ease(Math.min(1, anim.f / st.frames));
    const moving = path[0] !== path[path.length - 1], alpha = st.alpha;
    const moves = [], out = [];
    for (let i = 0; i < n; i++) {
      const pts = path.map((P) => P[i]), w = along(pts, e), a = pts[0], z = pts[pts.length - 1];
      const straight = [a[0] + (z[0] - a[0]) * e, a[1] + (z[1] - a[1]) * e];
      out.push([w.p[0] * (1 - alpha) + straight[0] * alpha, w.p[1] * (1 - alpha) + straight[1] * alpha]);
      moves.push({ wiggle: w.upto, from: a, straight, alpha, moving });
    }
    return { out, moves };
  }
  function advanceAnim() {
    const st = anim.stages[anim.idx];
    anim.f += 1;
    if (anim.f >= st.frames) {
      const wasMove = st.move;
      anim.idx += 1; anim.f = 0;
      if (anim.idx === anim.stages.length) {
        const lvl = anim.level; anim = null; t += 1; record(); noteTrend();
        if (lvl === 3 && t % TRAIL_EVERY === 0) for (const r of Object.values(runs)) if (!r.diverged) pushTrail(r, ringMean(r, -SMOOTH));
      }
      if (stopAfterMove && wasMove) { stopAfterMove = false; pause(); }
    }
  }
  let msPerIter = 1;
  const FRAME_BUDGET_MS = 10, MAX_PER_FRAME = 8;
  const maxPerFrame = () => Math.max(1, Math.min(MAX_PER_FRAME, Math.floor(FRAME_BUDGET_MS / Math.max(msPerIter, 1e-3))));
  function frame() {
    if (anim) advanceAnim();
    else if (tempo(t) < 1) { anim = buildIteration(); advanceAnim(); }
    else {
      // several iterations per frame, capped so that training takes at most ~10 ms of each frame
      shadow = null; acc = Math.min(acc + tempo(t), maxPerFrame());
      const t0 = performance.now(); let done = 0;
      while (acc >= 1) { iteration(); acc -= 1; done += 1; }
      if (done) msPerIter = 0.8 * msPerIter + 0.2 * (performance.now() - t0) / done;
    }
    if (anim) advanceShadow(3);                          // the simulated look-ahead is only used while animating
    draw();
  }
  function play() { if (playing) return; playing = true; $('tt-play').textContent = 'Pause'; const loop = () => { if (!playing) return; frame(); timer = requestAnimationFrame(loop); }; loop(); }
  function pause() { playing = false; $('tt-play').textContent = 'Play'; if (timer) cancelAnimationFrame(timer); }
  // Step: play until the next move has finished (one sub-step, block, or iteration)
  function stepOnce() {
    if (anim || tempo(t) < 1) { if (!anim) anim = buildIteration(); stopAfterMove = true; if (!playing) play(); return; }
    pause(); iteration(); draw();
  }
  function fastForward(m) { anim = null; trailsStarted = true; shadow = null; for (let k = 0; k < m; k++) iteration(); draw(true); }

  // ------------------------------------------------------------------ drawing helpers
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const set = (e, attrs) => { for (const k in attrs) e.setAttribute(k, attrs[k]); };
  const ptsStr = (arr) => arr.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  // a polyline drawn as a few chunks whose opacity ramps from `a0` (start) to `a1` (end)
  function fadingLine(parent, attrs, chunks) {
    return Array.from({ length: chunks }, () => el('polyline', { fill: 'none', ...attrs }, parent));
  }
  function setFading(segs, xy, a0, a1) {
    const m = segs.length, L = xy.length;
    segs.forEach((seg, j) => {
      const i0 = Math.floor(j * (L - 1) / m), i1 = Math.floor((j + 1) * (L - 1) / m);
      seg.setAttribute('points', ptsStr(xy.slice(i0, i1 + 1)));
      const f = (j + 0.5) / m;
      seg.setAttribute('stroke-opacity', (a1 + (a0 - a1) * (1 - f) * (1 - f)).toFixed(3));
    });
  }

  // ------------------------------------------------------------------ track geometry (Frenet -> world)
  const G = D.geometry, NC = G.center.length;
  function world(s, ey) {
    const x = ((s % G.L) + G.L) % G.L / G.L * (NC - 1), i = Math.min(Math.floor(x), NC - 2), f = x - i;
    const c = [G.center[i][0] * (1 - f) + G.center[i + 1][0] * f, G.center[i][1] * (1 - f) + G.center[i + 1][1] * f];
    const q = [G.inner[i][0] * (1 - f) + G.inner[i + 1][0] * f, G.inner[i][1] * (1 - f) + G.inner[i + 1][1] * f];
    return [c[0] + (q[0] - c[0]) / G.half_width * ey, c[1] + (q[1] - c[1]) / G.half_width * ey];
  }
  // world path (k = 0..N) of the continuation after first action o (scaled)
  const worldPath = (s, o) => { const u = [o[0] * UM[0], o[1] * UM[1]]; return s.pathP.map((p, k) => world(p[0] + s.pathK[k][0][0] * u[0] + s.pathK[k][0][1] * u[1], p[1] + s.pathK[k][1][0] * u[0] + s.pathK[k][1][1] * u[1])); };

  // ------------------------------------------------------------------ the figure: track + callouts
  const FIG = { W: 1160, H: 900, scale: 86 };
  let fig;
  // a zoomed first-action window for state s (the figure's callouts and the single window on small screens)
  const CW = 176, CH = 150;
  function makeWindow(s, parent, x0, y0, clipId, interactive) {
    const i = s.i;
    const g = el('g', interactive ? { transform: `translate(${x0},${y0})`, class: 'tt-callout', tabindex: 0, role: 'button',
                                      'aria-label': `State ${i + 1}: first-action window. Activate to enlarge.`, 'aria-pressed': 'false' }
                                  : { transform: `translate(${x0},${y0})` }, parent);
    el('rect', { x: 0, y: 0, width: CW, height: CH, rx: 3, fill: 'white', stroke: '#d4dbe1' }, g);
    const cap = el('text', { x: 8, y: 14, 'font-size': 10, fill: '#55616c' }, g);
    cap.innerHTML = `<tspan font-weight="700" fill="#16202a">${i + 1}</tspan>  ${s.v.toFixed(1)} m/s`;
    const P = { x: 30, y: 20, w: CW - 38, h: CH - 50 };
    el('rect', { x: P.x, y: P.y, width: P.w, height: P.h }, el('clipPath', { id: clipId }, el('defs', {}, g)));
    el('rect', { x: P.x, y: P.y, width: P.w, height: P.h, fill: '#fafbfc', stroke: '#dbe1e6' }, g);
    // ticks in physical units, re-spaced every frame as the window zooms
    const gridG = el('g', { 'clip-path': `url(#${clipId})` }, g), tickG = el('g', {}, g);
    const mk = (axis) => Array.from({ length: 7 }, () => ({ grid: el('line', { stroke: '#e3e8ec', 'stroke-width': 0.6 }, gridG),
      tick: el('line', { stroke: '#8a96a2', 'stroke-width': 0.8 }, tickG),
      text: el('text', { 'font-size': 7, fill: '#55616c', 'text-anchor': axis === 'x' ? 'middle' : 'end' }, tickG) }));
    const ticks = { x: mk('x'), y: mk('y') };
    const ux = el('text', { x: 3, y: P.y + P.h + 10, 'font-size': 6.5, fill: '#55616c' }, g); ux.textContent = 'm/s²';
    const uy = el('text', { x: 0, y: 0, 'font-size': 6.5, fill: '#55616c', 'text-anchor': 'middle',
                            transform: `translate(7,${P.y + P.h / 2}) rotate(-90)` }, g); uy.textContent = 'steer (rad)';
    const plane = el('g', { 'clip-path': `url(#${clipId})` }, g);
    const poly = el('polygon', { fill: C.safe, 'fill-opacity': 0.14, stroke: C.safe, 'stroke-width': 1.2 }, plane);
    const ell = el('polyline', { fill: 'none', stroke: C.label, 'stroke-opacity': 0.4, 'stroke-dasharray': '2 2' }, plane);
    const ubarG = el('g', { stroke: C.label, 'stroke-width': 1.6 }, plane), ub1 = el('line', {}, ubarG), ub2 = el('line', {}, ubarG);
    const vsC = el('circle', { r: 4.5, fill: 'none', stroke: C.vstar, 'stroke-width': 1.5 }, plane);
    const dyn = {};
    for (const k of ['fused', 'inter']) {
      dyn[k] = { arrow: el('line', { stroke: C[k], 'stroke-width': 1.6, 'marker-end': `url(#tt-arrow-${k})`, visibility: 'hidden' }, plane),
                 wiggle: el('polyline', { fill: 'none', stroke: C[k], 'stroke-width': 1.4, 'stroke-linejoin': 'round', 'marker-end': `url(#tt-arrow-${k})`, visibility: 'hidden' }, plane),
                 trail: fadingLine(plane, { stroke: C[k], 'stroke-width': 1.1 }, 5),
                 proj: el('circle', { r: 2.6, fill: 'white', stroke: C[k], 'stroke-width': 1 }, plane),
                 dot: el('circle', { r: 3.6, fill: C[k] }, plane) };
    }
    const readout = el('text', { x: 8, y: CH - 7, 'font-size': 9, fill: '#55616c' }, g);
    // static iso-cost ellipse through v* (in scaled coordinates)
    const r2 = quad(s.M, [s.vstar[0] - s.ubar[0], s.vstar[1] - s.ubar[1]]);
    const A_ = s.M[0][0], B_ = s.M[0][1], D_ = s.M[1][1], tr = (A_ + D_) / 2, det = Math.sqrt(((A_ - D_) / 2) ** 2 + B_ * B_);
    const l1 = tr + det, l2 = tr - det, th0 = 0.5 * Math.atan2(2 * B_, A_ - D_);
    const ellPts = Array.from({ length: 73 }, (_, k) => { const th = 2 * Math.PI * k / 72, e1 = Math.sqrt(r2 / l1) * Math.cos(th), e2 = Math.sqrt(r2 / l2) * Math.sin(th);
      return [s.ubar[0] + e1 * Math.cos(th0) - e2 * Math.sin(th0), s.ubar[1] + e1 * Math.sin(th0) + e2 * Math.cos(th0)]; });
    return { s, g, P, poly, ell, ellPts, ub1, ub2, vsC, dyn, ticks, readout };
  }
  function buildFigure() {
    const all = G.inner.concat(G.outer), xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const tx = (p) => [FIG.W / 2 + (p[0] - cx) * FIG.scale, FIG.H / 2 - (p[1] - cy) * FIG.scale];
    const svg = el('svg', { viewBox: `0 0 ${FIG.W} ${FIG.H}`, role: 'img', 'aria-label': 'Race track with the training states, their predicted continuations, and a zoomed first-action window per state' });
    const toS = (arr) => ptsStr(arr.map(tx));
    const band = (e) => G.center.map((_, k) => world(k / (NC - 1) * G.L, e));
    el('polygon', { points: toS(G.outer), fill: '#e9eef2', stroke: C.vstar, 'stroke-width': 1.2 }, svg);
    el('polygon', { points: toS(G.inner), fill: '#f4f6f7', stroke: C.vstar, 'stroke-width': 1.2 }, svg);
    for (const e of [G.band, -G.band]) el('polyline', { points: toS(band(e)), fill: 'none', stroke: C.band, 'stroke-dasharray': '5 4', 'stroke-width': 0.8 }, svg);
    el('polyline', { points: toS(G.raceline.map((p) => world(p[0], p[1]))), fill: 'none', stroke: '#8a96a2', 'stroke-width': 1.4, 'stroke-opacity': 0.7 }, svg);
    const defs = el('defs', {}, svg);
    for (const k of ['fused', 'inter']) {
      const mk = el('marker', { id: `tt-arrow-${k}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse', markerUnits: 'userSpaceOnUse' }, defs);
      el('path', { d: 'M0,1 L10,5 L0,9 z', fill: C[k] }, mk);
    }
    const pathLayer = el('g', {}, svg), leaderLayer = el('g', { class: 'tt-leaders' }, svg), markerLayer = el('g', {}, svg), calloutLayer = el('g', { class: 'tt-callouts' }, svg);

    // callout placement: on a ring around the track, along the ray from the track centre, spread by angle
    const ring = { x0: FIG.W / 2 - (Math.max(...xs) - cx) * FIG.scale - 30, x1: FIG.W / 2 + (Math.max(...xs) - cx) * FIG.scale + 30,
                                        y0: FIG.H / 2 - (Math.max(...ys) - cy) * FIG.scale - 30, y1: FIG.H / 2 + (Math.max(...ys) - cy) * FIG.scale + 30 };
    const anchors = states.map((s) => tx(s.xy));
    let ang = anchors.map((a) => Math.atan2(a[1] - FIG.H / 2, a[0] - FIG.W / 2));
    const order = ang.map((a, i) => i).sort((i, j) => ang[i] - ang[j]);
    const minGap = 2 * Math.PI / (states.length + 1.5);
    for (let it = 0; it < 200; it++) {                      // spread neighbours apart on the circle
      for (let k = 0; k < order.length; k++) {
        const i = order[k], j = order[(k + 1) % order.length];
        let d = ang[j] - ang[i]; if (k === order.length - 1) d += 2 * Math.PI;
        if (d < minGap) { const push = (minGap - d) / 2; ang[i] -= push; ang[j] += push; }
      }
    }
    const placeOnRing = (a) => {                           // where the ray at angle a leaves the ring rectangle, pushed out by half a callout
      const dx = Math.cos(a), dy = Math.sin(a);
      const tX = dx > 0 ? (ring.x1 - FIG.W / 2) / dx : (ring.x0 - FIG.W / 2) / dx, tY = dy > 0 ? (ring.y1 - FIG.H / 2) / dy : (ring.y0 - FIG.H / 2) / dy;
      const t = Math.min(Math.abs(tX), Math.abs(tY));
      let x = FIG.W / 2 + dx * t + Math.sign(dx) * (Math.abs(tX) <= Math.abs(tY) ? CW / 2 : 0);
      let y = FIG.H / 2 + dy * t + Math.sign(dy) * (Math.abs(tY) < Math.abs(tX) ? CH / 2 : 0);
      x = Math.max(CW / 2 + 4, Math.min(FIG.W - CW / 2 - 4, x)); y = Math.max(CH / 2 + 4, Math.min(FIG.H - CH / 2 - 4, y));
      return [x, y];
    };

    const items = states.map((s, i) => {
      const [ccx, ccy] = placeOnRing(ang[i]), x0 = ccx - CW / 2, y0 = ccy - CH / 2, a = anchors[i];
      // leader line from the state to the nearest point of the callout box
      const lx = Math.max(x0, Math.min(x0 + CW, a[0])), ly = Math.max(y0, Math.min(y0 + CH, a[1]));
      el('line', { x1: a[0], y1: a[1], x2: lx, y2: ly, stroke: '#9aa6b1', 'stroke-width': 0.8 }, leaderLayer);
      // continuations on the track: label (static), v* (static), fused and interleaved (live), fading along the horizon
      const pg = el('g', { class: 'tt-paths', visibility: 'hidden' }, pathLayer);   // shown on hover / when enlarged
      const lab = fadingLine(pg, { stroke: C.label, 'stroke-width': 1.1, 'stroke-dasharray': '3 2' }, 8);
      setFading(lab, worldPath(s, s.ubar).map(tx), 0.85, 0.0);
      const vs = fadingLine(pg, { stroke: C.vstar, 'stroke-width': 1.1, 'stroke-dasharray': '1 2' }, 8);
      setFading(vs, worldPath(s, s.vstar).map(tx), 0.85, 0.0);
      const live = { fused: fadingLine(pg, { stroke: C.fused, 'stroke-width': 1.8 }, 8), inter: fadingLine(pg, { stroke: C.inter, 'stroke-width': 1.8 }, 8) };
      const dot = el('g', { class: 'tt-marker' }, markerLayer);
      el('circle', { cx: a[0], cy: a[1], r: 24, fill: 'transparent', class: 'tt-hit' }, dot);      // touch target
      el('circle', { cx: a[0], cy: a[1], r: 6.5, fill: C.vstar, class: 'tt-dot' }, dot);
      const num = el('text', { x: a[0], y: a[1] + 3.3, 'text-anchor': 'middle', 'font-size': 9, 'font-weight': 600, fill: 'white' }, dot); num.textContent = i + 1;

      // the callout: a zoomed first-action window
      const win = makeWindow(s, calloutLayer, x0, y0, `tt-cc-${i}`, true);
      return { ...win, live, view: null, dot, pg, home: [x0, y0] };
    });
    const banner = el('g', { 'pointer-events': 'none', class: 'tt-banner' }, svg);
    // crop the canvas to the drawing: the track and every callout
    const boxes = all.map(tx).concat(items.flatMap((it) => { const m = /translate\(([-\d.]+),([-\d.]+)\)/.exec(it.g.getAttribute('transform')); const x = +m[1], y = +m[2]; return [[x, y], [x + CW, y + CH]]; }));
    const bx0 = Math.min(...boxes.map((b) => b[0])) - 8, bx1 = Math.max(...boxes.map((b) => b[0])) + 8;
    const by0 = Math.min(...boxes.map((b) => b[1])) - 8, by1 = Math.max(...boxes.map((b) => b[1])) + 8;
    const fullBox = `${bx0.toFixed(0)} ${by0.toFixed(0)} ${(bx1 - bx0).toFixed(0)} ${(by1 - by0).toFixed(0)}`;
    const tp = all.map(tx), tX = tp.map((p) => p[0]), tY = tp.map((p) => p[1]);
    const trackBox = `${(Math.min(...tX) - 12).toFixed(0)} ${(Math.min(...tY) - 12).toFixed(0)} ${(Math.max(...tX) - Math.min(...tX) + 24).toFixed(0)} ${(Math.max(...tY) - Math.min(...tY) + 24).toFixed(0)}`;
    svg.setAttribute('viewBox', fullBox);
    // caption of what is being played, in the empty top-right corner of the figure
    const BW = 250, bxL = bx1 - BW - 10, byT = by0 + 26;
    const bKicker = el('text', { x: bxL, y: byT, 'font-size': 10, 'letter-spacing': 1.2, 'font-weight': 700, fill: '#55616c' }, banner);
    const bTitle = el('text', { x: bxL, y: byT + 18, 'font-size': 14, 'font-weight': 600, fill: C.vstar }, banner);
    const bDesc = [0, 1, 2, 3, 4].map((j) => el('text', { x: bxL, y: byT + 38 + 14 * j, 'font-size': 10.5, fill: '#55616c' }, banner));
    // backdrop shown behind an enlarged window
    const backdrop = el('rect', { x: bx0, y: by0, width: bx1 - bx0, height: by1 - by0, fill: '#f4f6f7', 'fill-opacity': 0.72, visibility: 'hidden' }, calloutLayer);
    const zoom = { k: Math.min(0.72 * (bx1 - bx0) / CW, 0.8 * (by1 - by0) / CH), cx: (bx0 + bx1) / 2, cy: (by0 + by1) / 2 };
    $('tt-figure').appendChild(svg);
    fig = { svg, tx, items, backdrop, zoom, calloutLayer, CW, CH, open: -1, bKicker, bTitle, bDesc, fullBox, trackBox };
    items.forEach((it, i) => {
      for (const node of [it.g, it.dot]) {
        node.addEventListener('mouseenter', () => highlight(i)); node.addEventListener('mouseleave', () => highlight(-1));
        node.addEventListener('click', (e) => { e.stopPropagation(); if (compact) select(i); else toggleZoom(i); });
      }
      it.g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleZoom(i); } });
    });
    backdrop.addEventListener('click', () => toggleZoom(fig.open));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && fig.open >= 0) toggleZoom(fig.open); });
  }
  // only the hovered (or enlarged) state's continuations are drawn on the track
  const showPaths = (i) => fig.items.forEach((it, j) => it.pg.setAttribute('visibility', j === i ? 'visible' : 'hidden'));
  function highlight(i) {
    if (fig.open >= 0 || compact) return;
    fig.items.forEach((it, j) => { it.g.style.opacity = i < 0 || i === j ? 1 : 0.35; });
    showPaths(i);
  }
  // ---- small screens: the track alone, and one full-width window for the selected state
  let compact = false, sel = 0, single = null;
  const COMPACT_QUERY = window.matchMedia ? window.matchMedia('(max-width: 720px)') : { matches: false, addEventListener() {} };
  function select(i) {
    sel = (i + n) % n;
    const svgS = $('tt-single-svg');
    while (svgS.firstChild) svgS.removeChild(svgS.firstChild);
    const it = fig.items[sel], w = makeWindow(it.s, svgS, 0, 0, 'tt-single-clip', false);
    single = { ...w, live: it.live, pg: it.pg, view: null };
    $('tt-single-label').textContent = `State ${sel + 1} of ${n} · ${it.s.v.toFixed(1)} m/s`;
    showPaths(sel);
    fig.items.forEach((o, j) => o.dot.classList.toggle('active', j === sel));
    if (runs) drawCallout(single, true);
  }
  function applyMode() {
    compact = COMPACT_QUERY.matches;
    if (compact && fig.open >= 0) toggleZoom(fig.open);
    fig.svg.setAttribute('viewBox', compact ? fig.trackBox : fig.fullBox);
    $('tt-figure').classList.toggle('compact', compact);
    // bigger numbered markers on small screens, where the figure is drawn at about half size
    fig.items.forEach((o) => {
      const c = o.dot.querySelector('.tt-dot'), tx = o.dot.querySelector('text'), y = +c.getAttribute('cy');
      c.setAttribute('r', compact ? 12 : 6.5); tx.setAttribute('font-size', compact ? 15 : 9); tx.setAttribute('y', y + (compact ? 5.3 : 3.3));
    });
    if (compact) select(sel); else { showPaths(-1); fig.items.forEach((o) => o.dot.classList.remove('active')); }
    if (runs) draw(true);
  }

  // click a window (or its marker) to enlarge it over the track; click again, the backdrop, or Escape to go back
  function animateTo(it, x, y, k, done) {
    const m = /translate\(([-\d.]+),([-\d.]+)\)(?: scale\(([-\d.]+)\))?/.exec(it.g.getAttribute('transform'));
    const from = [+m[1], +m[2], m[3] ? +m[3] : 1], t0 = performance.now(), dur = 260;
    const ease = (u) => 1 - Math.pow(1 - u, 3);
    const frame = (now) => {
      const u = Math.min(1, (now - t0) / dur), e = ease(u);
      const cur = [from[0] + (x - from[0]) * e, from[1] + (y - from[1]) * e, from[2] + (k - from[2]) * e];
      it.g.setAttribute('transform', `translate(${cur[0].toFixed(1)},${cur[1].toFixed(1)}) scale(${cur[2].toFixed(3)})`);
      if (u < 1) requestAnimationFrame(frame); else if (done) done();
    };
    requestAnimationFrame(frame);
  }
  function toggleZoom(i) {
    if (i < 0) return;
    const it = fig.items[i], z = fig.zoom;
    if (fig.open === i) {                                  // zoom back out
      fig.open = -1;
      it.g.setAttribute('aria-pressed', 'false');
      fig.backdrop.setAttribute('visibility', 'hidden');
      fig.items.forEach((o) => { o.g.style.opacity = 1; });
      showPaths(-1);
      animateTo(it, it.home[0], it.home[1], 1);
      return;
    }
    if (fig.open >= 0) { const prev = fig.items[fig.open]; prev.g.setAttribute('aria-pressed', 'false'); animateTo(prev, prev.home[0], prev.home[1], 1); }
    fig.open = i;
    it.g.setAttribute('aria-pressed', 'true');
    fig.items.forEach((o) => { o.g.style.opacity = 1; });
    showPaths(i);
    fig.calloutLayer.appendChild(fig.backdrop); fig.calloutLayer.appendChild(it.g);   // backdrop, then this window, on top
    fig.backdrop.setAttribute('visibility', 'visible');
    animateTo(it, z.cx - fig.CW * z.k / 2, z.cy - fig.CH * z.k / 2, z.k);
  }

  // zoom of a window: the box around the label, v*, both outputs and their projections, smoothed over frames
  function targetView(it) {
    const s = it.s, P = [s.ubar, s.vstar];
    if (trendMode()) for (const k of ['fused', 'inter']) {
      const r = runs[k];
      if (!show[k] || r.diverged || r.ring.length <= 2 * SMOOTH) continue;
      for (const o of r.ring) P.push(o[s.i]);
      const a = ringMean(r, 0, SMOOTH)[s.i], z = ringMean(r, -SMOOTH)[s.i];
      P.push([2 * z[0] - a[0], 2 * z[1] - a[1]]);        // where the trend points TREND iterations ahead
    }
    if (shadow) for (const e of shadow.boxes) for (const k of ['fused', 'inter']) {
      if (!show[k]) continue;
      const b = e.box[k][s.i]; if (b[0] <= b[1]) P.push([b[0], b[2]], [b[1], b[3]]);
    }
    for (const k of ['fused', 'inter']) if (!runs[k].diverged && show[k]) {
      const sh = shown(k), o = sh.out[s.i]; P.push(o, project(s, o)); if (sh.moves) P.push(...sh.moves[s.i].wiggle);
    }
    let x0 = Math.min(...P.map((p) => p[0])), x1 = Math.max(...P.map((p) => p[0])), y0 = Math.min(...P.map((p) => p[1])), y1 = Math.max(...P.map((p) => p[1]));
    const half = Math.max((x1 - x0) / 2 * 1.35, (y1 - y0) / 2 * 1.35 * it.P.w / it.P.h, 0.06);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return { cx, cy, half: Math.min(half, 1.3) };
  }
  function drawCallout(it, snap) {
    const tv = targetView(it);
    if (!it.view || snap) it.view = tv;
    else {
      const v0 = it.view, ah = tv.half > v0.half ? 0.12 : 0.012, ac = 0.05;
      // keep the needed box inside the view while panning: grow first, then move the centre
      it.view = { cx: v0.cx + ac * (tv.cx - v0.cx), cy: v0.cy + ac * (tv.cy - v0.cy), half: v0.half + ah * (tv.half - v0.half) };
    }
    const v = it.view, sc = it.P.w / 2 / v.half;      // px per scaled unit, equal on both axes
    const X = (p) => [it.P.x + it.P.w / 2 + (p[0] - v.cx) * sc, it.P.y + it.P.h / 2 - (p[1] - v.cy) * sc];
    set(it.poly, { points: ptsStr(it.s.poly.map(X)) });
    set(it.ell, { points: ptsStr(it.ellPts.map(X)) });
    const ub = X(it.s.ubar); set(it.ub1, { x1: ub[0] - 4, y1: ub[1] - 4, x2: ub[0] + 4, y2: ub[1] + 4 }); set(it.ub2, { x1: ub[0] - 4, y1: ub[1] + 4, x2: ub[0] + 4, y2: ub[1] - 4 });
    const vs = X(it.s.vstar); set(it.vsC, { cx: vs[0], cy: vs[1] });
    const parts = [];
    for (const k of ['fused', 'inter']) {
      const r = runs[k], d = it.dyn[k];
      if (!show[k]) {
        for (const e of [d.dot, d.proj, d.arrow, d.wiggle]) e.setAttribute('visibility', 'hidden');
        setFading(d.trail, [[0, 0], [0, 0]], 0, 0); setFading(it.live[k], [[0, 0], [0, 0]], 0, 0); continue;
      }
      if (r.diverged) { for (const e of [d.dot, d.proj]) e.setAttribute('visibility', 'hidden'); setFading(d.trail, [[0, 0], [0, 0]], 0, 0); setFading(it.live[k], [[0, 0], [0, 0]], 0, 0); parts.push(`${k === 'fused' ? 'fused' : 'inter'} —`); continue; }
      const sh = shown(k), o = sh.out[it.s.i], p = project(it.s, o), Xo = X(o), Xp = X(p);
      for (const e of [d.dot, d.proj]) e.setAttribute('visibility', 'visible');
      set(d.dot, { cx: Xo[0], cy: Xo[1] }); set(d.proj, { cx: Xp[0], cy: Xp[1] });
      // arrow for the sub-step being played: from where this action was to where it is now
      const mv = sh.moves && sh.moves[it.s.i];
      if (mv && mv.moving) {
        const Xs = X(mv.straight), Xa = X(mv.from), long = Math.hypot(Xs[0] - Xa[0], Xs[1] - Xa[1]) > 3;
        set(d.arrow, { x1: Xa[0], y1: Xa[1], x2: Xs[0], y2: Xs[1], visibility: long && mv.alpha > 0.01 ? 'visible' : 'hidden', 'stroke-opacity': mv.alpha, 'stroke-dasharray': '' });
        set(d.wiggle, { points: ptsStr(mv.wiggle.map(X)), visibility: mv.alpha < 0.99 && mv.wiggle.length > 1 ? 'visible' : 'hidden', 'stroke-opacity': 1 - mv.alpha });
      } else if (trendMode() && r.ring.length > 2 * SMOOTH) {
        d.wiggle.setAttribute('visibility', 'hidden');
        const Xf = X(ringMean(r, 0, SMOOTH)[it.s.i]), Xto = X(ringMean(r, -SMOOTH)[it.s.i]);
        if (Math.hypot(Xf[0] - Xto[0], Xf[1] - Xto[1]) > 3) set(d.arrow, { x1: Xf[0], y1: Xf[1], x2: Xto[0], y2: Xto[1], visibility: 'visible', 'stroke-opacity': 0.7, 'stroke-dasharray': '4 2' });
        else d.arrow.setAttribute('visibility', 'hidden');
      } else { d.arrow.setAttribute('visibility', 'hidden'); d.wiggle.setAttribute('visibility', 'hidden'); }
      if (trailsStarted) setFading(d.trail, r.trails[it.s.i].map(X), 0.05, 0.75); else setFading(d.trail, [[0, 0], [0, 0]], 0, 0);
      setFading(it.live[k], worldPath(it.s, o).map(fig.tx), 0.95, 0.0);
      const vcm = 100 * bandViolation(it.s, o);
      parts.push(`<tspan fill="${C[k]}">${k === 'fused' ? 'fused' : 'inter'} ${vcm > 0.05 ? '+' + vcm.toFixed(1) + ' cm' : 'in band'}</tspan>`);
    }
    it.readout.innerHTML = parts.join('   ');
    // ticks: about four per axis at round values, in m/s^2 (x) and rad (y)
    const halfY = v.half * it.P.h / it.P.w;
    const axes = [['x', v.cx - v.half, v.cx + v.half, UM[0]], ['y', v.cy - halfY, v.cy + halfY, UM[1]]];
    for (const [ax, lo, hi, um] of axes) {
      const plo = lo * um, phi = hi * um, raw = (phi - plo) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => m >= raw);
      const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9) + (step / mag === 2.5 ? 1 : 0));
      const first = Math.ceil(plo / step) * step;
      it.ticks[ax].forEach((tk, j) => {
        const val = first + j * step;
        if (val > phi + 1e-12) { for (const e of [tk.grid, tk.tick, tk.text]) e.setAttribute('visibility', 'hidden'); return; }
        for (const e of [tk.grid, tk.tick, tk.text]) e.setAttribute('visibility', 'visible');
        tk.text.textContent = (Math.abs(val) < step / 1e6 ? 0 : val).toFixed(dec);
        if (ax === 'x') {
          const px = X([val / um, v.cy])[0], yb = it.P.y + it.P.h;
          set(tk.grid, { x1: px, x2: px, y1: it.P.y, y2: yb }); set(tk.tick, { x1: px, x2: px, y1: yb, y2: yb + 3 });
          set(tk.text, { x: px, y: yb + 10 });
        } else {
          const py = X([v.cx, val / um])[1], xl = it.P.x;
          set(tk.grid, { x1: xl, x2: xl + it.P.w, y1: py, y2: py }); set(tk.tick, { x1: xl - 3, x2: xl, y1: py, y2: py });
          set(tk.text, { x: xl - 4, y: py + 2.5 });
        }
      });
    }
  }

  // ---- convergence plot
  let plot;
  function buildPlot() {
    const W = 520, Hh = 160, l = 44, r = 10, top = 10, bot = 26;
    const svg = el('svg', { viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': 'Distance to the constrained optimum over iterations' });
    const ly = (v) => top + (Math.log10(1) - Math.log10(Math.max(v, 1e-3))) / 3 * (Hh - top - bot);
    for (const v of [1, 0.1, 0.01, 0.001]) {
      el('line', { x1: l, x2: W - r, y1: ly(v), y2: ly(v), stroke: '#e0e6eb' }, svg);
      const tx = el('text', { x: l - 6, y: ly(v) + 3, 'text-anchor': 'end', 'font-size': 9, fill: '#55616c' }, svg); tx.textContent = v;
    }
    const xl = el('text', { x: (l + W - r) / 2, y: Hh - 6, 'text-anchor': 'middle', 'font-size': 9, fill: '#55616c' }, svg); xl.textContent = 'iteration';
    const lines = {};
    for (const k of ['fused', 'inter']) {
      lines[k] = { dv: el('polyline', { fill: 'none', stroke: C[k], 'stroke-width': 1.8 }, svg),
                   dc: el('polyline', { fill: 'none', stroke: C[k], 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, svg) };
    }
    const xmax = el('text', { x: W - r, y: Hh - 6, 'text-anchor': 'end', 'font-size': 9, fill: '#55616c' }, svg);
    $('tt-plot').appendChild(svg);
    plot = { W, l, r, ly, lines, xmax };
  }

  // ------------------------------------------------------------------ draw
  function wrap(text, width) {
    const words = text.split(' '), lines = [''];
    for (const w of words) { if ((lines[lines.length - 1] + ' ' + w).trim().length > width) lines.push(w); else lines[lines.length - 1] = (lines[lines.length - 1] + ' ' + w).trim(); }
    return lines;
  }
  function drawBanner() {
    let kicker = '', title = '', desc = '';
    if (anim) { const st = anim.stages[anim.idx]; kicker = `ITERATION ${t + 1}`; title = st.label; desc = st.desc; }
    else if (t === 0) { title = 'Press Play'; desc = 'The first iterations are shown one sub-step at a time; the pace then picks up.'; }
    else if (trendMode()) { kicker = `ITERATION ${t}`; title = 'Trend'; desc = `Dashed arrows: where each action has been heading over the last ${TREND} iterations (averaged to hide step-to-step jitter).`; }
    fig.bKicker.textContent = kicker; fig.bTitle.textContent = title;
    $('tt-cap-kicker').textContent = kicker; $('tt-cap-title').textContent = title; $('tt-cap-desc').textContent = desc;
    const lines = wrap(desc, 42);
    fig.bDesc.forEach((e, j) => { e.textContent = lines[j] || ''; });
  }
  function draw(snap) {
    if (!shadow && !trendMode()) advanceShadow(3);
    if (compact) { if (single) drawCallout(single, snap); } else fig.items.forEach((it) => drawCallout(it, snap));
    drawBanner();
    const T = Math.max(t, 50), px = (i) => plot.l + i / T * (plot.W - plot.l - plot.r);
    for (const k of ['fused', 'inter']) {
      const h = runs[k].hist, stride = Math.max(1, Math.floor(h.length / 400)), a = [], b = [];
      for (let i = 0; i < h.length; i += stride) if (h[i].m) { const x = px(h[i].t).toFixed(1); a.push(`${x},${plot.ly(h[i].m.dv).toFixed(1)}`); b.push(`${x},${plot.ly(h[i].m.dc).toFixed(1)}`); }
      plot.lines[k].dv.setAttribute('points', show[k] ? a.join(' ') : ''); plot.lines[k].dc.setAttribute('points', show[k] ? b.join(' ') : '');
    }
    plot.xmax.textContent = T;
    const fmt = (k) => { const r = runs[k]; if (r.diverged) return 'diverged'; const m = r.hist[r.hist.length - 1].m; return `${m.dv.toFixed(3)} from v*, ${m.unsafe} of ${n} paths leave the band`; };
    $('tt-status').textContent = `Iteration ${t}. Fused: ${fmt('fused')}. Interleaved: ${fmt('inter')}.`;
  }

  // ------------------------------------------------------------------ controls
  function bind() {
    $('tt-play').addEventListener('click', () => (playing ? pause() : play()));
    $('tt-step').addEventListener('click', stepOnce);
    $('tt-reset').addEventListener('click', () => { pause(); reset(); });
    const LAMS = [0, 1, 10, 100];
    const lam = $('tt-lam'); lam.addEventListener('input', () => { params.lam = LAMS[+lam.value]; $('tt-lam-v').textContent = params.lam; });
    $('tt-norm').addEventListener('change', (e) => { params.normalized = e.target.checked; });
    const c = $('tt-c'); c.addEventListener('input', () => { params.c = +c.value; $('tt-c-v').textContent = params.c; });
    const beta = $('tt-beta'); beta.addEventListener('input', () => { params.beta = +beta.value; $('tt-beta-v').textContent = params.beta.toFixed(2); });
    $('tt-refresh').addEventListener('change', (e) => { params.refresh = e.target.checked; });
    for (const k of ['fused', 'inter']) $(`tt-show-${k}`).addEventListener('change', (e) => { show[k] = e.target.checked; draw(); });
    for (const id of ['tt-lam', 'tt-norm', 'tt-c', 'tt-beta', 'tt-refresh', 'tt-speed']) $(id).addEventListener('input', () => { shadow = null; });
    for (const id of ['tt-norm', 'tt-refresh']) $(id).addEventListener('change', () => { shadow = null; });
    const sp = $('tt-speed');
    sp.addEventListener('input', () => { params.speed = +sp.value; const m = Math.pow(2, params.speed); $('tt-speed-v').textContent = `${parseFloat(m.toPrecision(2))}×`; });
    $('tt-settings').addEventListener('click', () => {
      const open = !$('tt-sidebar').classList.contains('open');
      $('tt-sidebar').classList.toggle('open', open); $('tt-settings').setAttribute('aria-expanded', String(open));
    });
    // on small screens the control bar is fixed to the bottom of the screen: show it only while the demo is visible
    const section = document.getElementById('track-demo');
    if (section && 'IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => {
        $('tt-sidebar').classList.toggle('offscreen', !e.isIntersecting);
        if (!e.isIntersecting) { $('tt-sidebar').classList.remove('open'); $('tt-settings').setAttribute('aria-expanded', 'false'); }
      }, { rootMargin: '0px 0px -30% 0px' }).observe(section);
    }
    $('tt-prev').addEventListener('click', () => select(sel - 1));
    $('tt-next').addEventListener('click', () => select(sel + 1));
    let touchX = null;
    $('tt-single').addEventListener('touchstart', (e) => { touchX = e.touches[0].clientX; }, { passive: true });
    $('tt-single').addEventListener('touchend', (e) => {
      if (touchX === null) return;
      const dx = e.changedTouches[0].clientX - touchX; touchX = null;
      if (Math.abs(dx) > 40) select(sel + (dx < 0 ? 1 : -1));
    });
    // click the convergence plot to enlarge it; click again, the backdrop, or Escape to put it back
    const pb = $('tt-plotbox'), bd = $('tt-plot-backdrop');
    // the sidebar is sticky (its own stacking context), so the enlarged plot is moved to <body> and back
    const slot = document.createComment('plot slot'); pb.parentNode.insertBefore(slot, pb);
    const setPlot = (open) => {
      if (open) document.body.appendChild(pb); else slot.parentNode.insertBefore(pb, slot.nextSibling);
      pb.classList.toggle('enlarged', open); pb.setAttribute('aria-expanded', String(open)); bd.hidden = !open;
      pb.querySelector('.hint').textContent = open ? 'click to close' : 'click to enlarge';
      if (open) pb.focus();
    };
    pb.addEventListener('click', () => setPlot(!pb.classList.contains('enlarged')));
    pb.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPlot(!pb.classList.contains('enlarged')); } });
    bd.addEventListener('click', () => setPlot(false));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && pb.classList.contains('enlarged')) setPlot(false); });
    if (new URLSearchParams(window.location.search).get('plot') === 'open') setPlot(true);
    if (new URLSearchParams(window.location.search).get('settings') === 'open') $('tt-settings').click();
    // restore the default settings; the training run itself continues from where it is
    $('tt-defaults').addEventListener('click', () => {
      const fire = (id, v, ev) => { const e = $(id); if (typeof v === 'boolean') e.checked = v; else e.value = v; e.dispatchEvent(new Event(ev)); };
      fire('tt-lam', LAMS.indexOf(DEFAULTS.lam), 'input'); fire('tt-norm', DEFAULTS.normalized, 'change');
      fire('tt-c', DEFAULTS.c, 'input'); fire('tt-beta', DEFAULTS.beta, 'input'); fire('tt-refresh', DEFAULTS.refresh, 'change');
      fire('tt-speed', DEFAULTS.speed, 'input');
    });
  }

  buildFigure(); buildPlot(); bind(); reset(); applyMode();
  COMPACT_QUERY.addEventListener('change', applyMode);
  // ?iters=N runs N iterations on load (for screenshots and links to a trained state)
  const pre = +(new URLSearchParams(window.location.search).get('iters') || 0);
  if (pre > 0) fastForward(Math.min(pre, 20000));
  // ?frames=N simulates N animation frames from the start (for screenshots of the opening)
  const fr = +(new URLSearchParams(window.location.search).get('frames') || 0);
  for (let k = 0; k < fr; k++) frame();
  // ?bench=N runs N frames synchronously and records the script time per frame (training + DOM updates,
  // without the browser's own layout and paint) in <body data-bench>
  const bn = +(new URLSearchParams(window.location.search).get('bench') || 0);
  if (bn > 0) {
    const ms = [];
    for (let k = 0; k < bn; k++) { const t0 = performance.now(); frame(); ms.push(performance.now() - t0); }
    const q = (a, f) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
    const part = (a, b) => ms.slice(a, b);
    document.body.dataset.bench = JSON.stringify({ frames: bn, t, opening: { median: q(part(0, 1500), 0.5), p95: q(part(0, 1500), 0.95) },
      trend: { median: q(part(2000, bn), 0.5), p95: q(part(2000, bn), 0.95), max: Math.max(...part(2000, bn)) } });
  }
  // ?zoom=N opens window N on load
  const z0 = +(new URLSearchParams(window.location.search).get('zoom') || 0);
  if (z0 >= 1 && z0 <= n) toggleZoom(z0 - 1);
})();

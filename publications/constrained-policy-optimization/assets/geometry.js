'use strict';
// Equation (29) on a quadratic toy constraint with a known, valid curvature bound,
// iterated as SCPO would run it: each accepted update becomes the next candidate.
// All optimization is in update space, so the coefficient metric is D^T D.
const SCPOGeometry = (() => {
  const center = [-0.25, 0.15], trueRadius = 0.9;
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const scale = (a, t) => [a[0] * t, a[1] * t];
  const norm = a => Math.hypot(...a);
  const g = w => 0.5 * ((w[0] - center[0]) ** 2 + (w[1] - center[1]) ** 2 - trueRadius ** 2);

  // Toy task loss: an anisotropic quadratic whose minimum lies outside the feasible disk,
  // so the raw gradient step always points toward infeasible weights. The anisotropy makes
  // the gradient direction rotate along the path, which is what gives the bank its rank.
  const target = [1.0, -0.75];
  const lossMatrix = (() => {
    // Valley (eigenvalue l1) runs at -20 degrees; the stiff direction (l2) is perpendicular.
    const phi = -20 * Math.PI / 180, c = Math.cos(phi), s = Math.sin(phi), l1 = 1, l2 = 8;
    return [[l1 * c * c + l2 * s * s, (l1 - l2) * c * s], [(l1 - l2) * c * s, l1 * s * s + l2 * c * c]];
  })();
  const gradient = w => { const e = add(w, scale(target, -1)); return [dot(lossMatrix[0], e), dot(lossMatrix[1], e)]; };
  const loss = w => { const e = add(w, scale(target, -1)); return 0.5 * dot(e, gradient(w)); };
  const start = [0, 0];

  // Raw update: a gradient step with learning rate eta, clipped to the trust radius. The
  // smoothness bound is only used locally, and because the certified region is convex and
  // contains the origin, projecting a point inside the trust ball keeps the result inside it.
  const trustRadius = 0.5;
  function rawStep(w, eta) {
    const step = scale(gradient(w), -eta), length = norm(step);
    return length > trustRadius ? scale(step, trustRadius / length) : step;
  }

  // Armijo backtracking along the projected update. The certified region is convex and
  // contains the origin, so every shortened step alpha * p stays certified.
  function backtrack(w, p, c1 = 1e-4, maxHalvings = 8) {
    const slope = dot(gradient(w), p), base = loss(w);
    if (!(slope < 0)) return 0;
    let alpha = 1;
    for (let i = 0; i <= maxHalvings; i++, alpha /= 2) {
      if (loss(add(w, scale(p, alpha))) <= base + c1 * alpha * slope) return alpha;
    }
    return 0;
  }

  // Bank at the current weights w: the previous accepted update (if it is independent of the
  // raw step) and the raw step. The raw step is always the last column.
  function setup(w, previous, eta, ell) {
    const raw = rawStep(w, eta);
    let D = [raw];
    if (previous) {
      const cross = previous[0] * raw[1] - previous[1] * raw[0];
      if (Math.abs(cross) > 0.03 * norm(previous) * norm(raw)) D = [previous, raw];
    }
    const G = D.map(d => g(add(w, d))), q = D.map(d => dot(d, d));
    let inverse = null;
    if (D.length === 2) {
      const [a, b] = D, det = a[0] * b[1] - a[1] * b[0];
      inverse = [[b[1] / det, -b[0] / det], [-a[1] / det, a[0] / det]];
    }
    return {w, D, G, q, raw, ell, inverse, count: D.length, g0: g(w)};
  }
  function coefficients(u, bank) {
    return bank.inverse ? bank.inverse.map(row => dot(row, u)) : [dot(u, bank.D[0]) / bank.q[0]];
  }
  function upper(u, bank) {
    const xi = coefficients(u, bank);
    return bank.g0 + xi.reduce((sum, x, i) => sum + x * (bank.G[i] - bank.g0), 0)
      + 0.5 * bank.ell * (dot(u, u) + xi.reduce((sum, x, i) => sum + Math.abs(x) * bank.q[i], 0));
  }
  function extent(v, bank) {
    const xi = coefficients(v, bank);
    const h = xi.reduce((sum, x, i) => sum + x * (bank.G[i] - bank.g0) + 0.5 * bank.ell * Math.abs(x) * bank.q[i], 0);
    return Math.max(0, (-h + Math.sqrt(Math.max(0, h * h - 2 * bank.ell * bank.g0))) / bank.ell);
  }
  function project(bank) {
    const candidates = [[0, 0]];
    // The boundaries between sign cones are rays through the candidate vectors.
    for (const d of bank.D) for (const sign of [-1, 1]) {
      const v = scale(d, sign / norm(d));
      candidates.push(scale(v, Math.max(0, Math.min(dot(bank.raw, v), extent(v, bank)))));
    }
    if (bank.count === 2) {
      // For each coefficient sign pattern, the bound is a disk intersected with a cone.
      // Its nearest point is either the disk projection or one of the boundary-ray projections.
      for (const s0 of [-1, 1]) for (const s1 of [-1, 1]) {
        const signs = [s0, s1];
        const b = bank.G.map((value, i) => value - bank.g0 + 0.5 * bank.ell * bank.q[i] * signs[i]);
        const a = [bank.inverse[0][0] * b[0] + bank.inverse[1][0] * b[1], bank.inverse[0][1] * b[0] + bank.inverse[1][1] * b[1]];
        const c = scale(a, -1 / bank.ell);
        const r2 = dot(a, a) / bank.ell ** 2 - 2 * bank.g0 / bank.ell;
        if (r2 < 0) continue;
        const r = Math.sqrt(r2);
        const offset = add(bank.raw, scale(c, -1)), length = norm(offset);
        const p = length <= r ? bank.raw : add(c, scale(offset, r / length));
        const xi = coefficients(p, bank);
        if (xi.every((x, i) => signs[i] * x >= -1e-10)) candidates.push(p);
      }
    }
    return candidates.reduce((best, p) => norm(add(p, scale(bank.raw, -1))) < norm(add(best, scale(bank.raw, -1))) ? p : best);
  }
  // Certified region intersected with the trust ball, as radial extents.
  function boundary(bank, resolution = 720) {
    const reach = v => Math.min(extent(v, bank), trustRadius);
    if (bank.count === 1) {
      const v = scale(bank.raw, 1 / norm(bank.raw));
      return [scale(v, -reach(scale(v, -1))), scale(v, reach(v))];
    }
    const angles = Array.from({length: resolution}, (_, i) => i * 2 * Math.PI / resolution);
    for (const d of bank.D) for (const sign of [-1, 1]) angles.push((Math.atan2(sign * d[1], sign * d[0]) + 2 * Math.PI) % (2 * Math.PI));
    return angles.sort((a, b) => a - b).map(a => {
      const v = [Math.cos(a), Math.sin(a)];
      return scale(v, reach(v));
    });
  }
  // Loss contour L(w) = level as an ellipse: centre, semi-axes, rotation (degrees).
  function contour(level) {
    const [[a, b], [, d]] = lossMatrix;
    const tr = a + d, det = a * d - b * b, disc = Math.sqrt(tr * tr / 4 - det);
    const l1 = tr / 2 - disc, l2 = tr / 2 + disc;
    const angle = Math.atan2(l1 - a, b) * 180 / Math.PI;
    return {center: target, rx: Math.sqrt(2 * level / l1), ry: Math.sqrt(2 * level / l2), angle};
  }
  return {setup, project, boundary, backtrack, coefficients, upper, g, loss, gradient, rawStep, contour, center, trueRadius, trustRadius, target, start};
})();
if (typeof module !== 'undefined') module.exports = SCPOGeometry;

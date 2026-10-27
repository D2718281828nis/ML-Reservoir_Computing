// Part 8: a reservoir read as a state-space model. A plant x(t) = A x(t−1) + B u(t), y(t) = C x(t) + D u(t) (SISO, 2 states)
// makes the data; a reservoir is trained on it; the trained reservoir, linearized at x = 0, is again an (A, B, C, D) model:
//   A = (1−α) I + α W,  B = α W_in,  C = W_out (state part),  D = W_out (input part),  plus the constant W_out (bias part).
// Both use the same timing as the ESN (the state is updated with u(t) before y(t) is read), so both have the impulse response
//   h(0) = C B + D,  h(k) = C Aᵏ B  (k ≥ 1).

import { rng } from "./util.js";
import { Reservoir, nrmse } from "./models.js";
import { gram, solveRidge, applyReadout } from "./nn.js";

export const CTRL = { n: 1600, washout: 100, trainEnd: 900, valEnd: 1100, N: 50, degree: 8, K: 60 };
const LAMBDAS = [1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4];

// A second-order plant from its pole pair r·e^{±iθ}, in controllable canonical form, scaled to unit DC gain (before D).
export function plant({ r = 0.9, theta = 0.3, d = 0 } = {}) {
  const a1 = -2 * r * Math.cos(theta);
  const a2 = r * r;
  const A = [[0, 1], [-a2, -a1]];
  const B = [0, 1];
  // C (I − A)⁻¹ B = 1 with C = [c, c]:  (I − A)⁻¹ B = [1, 1] / (1 + a1 + a2)
  const c = (1 + a1 + a2) / 2;
  return { A, B, C: [c, c], D: d, n: 2, poles: [{ re: r * Math.cos(theta), im: r * Math.sin(theta) }, { re: r * Math.cos(theta), im: -r * Math.sin(theta) }] };
}

// y for an input sequence u, from x = 0. sys: dense { A (n × n rows), B, C, D, n }.
export function simulate(sys, u) {
  const { A, B, C, D, n } = sys;
  let x = new Float64Array(n);
  let nx = new Float64Array(n);
  const y = new Float64Array(u.length);
  for (let t = 0; t < u.length; t++) {
    for (let i = 0; i < n; i++) {
      let a = B[i] * u[t];
      for (let j = 0; j < n; j++) a += A[i][j] * x[j];
      nx[i] = a;
    }
    [x, nx] = [nx, x];
    let o = D * u[t];
    for (let i = 0; i < n; i++) o += C[i] * x[i];
    y[t] = o;
  }
  return y;
}

// h(0) = C B + D, h(k) = C Aᵏ B
export function impulse(sys, K) {
  const { A, B, C, D, n } = sys;
  let v = Float64Array.from(B);
  let nv = new Float64Array(n);
  const h = new Float64Array(K + 1);
  for (let k = 0; k <= K; k++) {
    let o = k === 0 ? D : 0;
    for (let i = 0; i < n; i++) o += C[i] * v[i];
    h[k] = o;
    for (let i = 0; i < n; i++) {
      let a = 0;
      for (let j = 0; j < n; j++) a += A[i][j] * v[j];
      nv[i] = a;
    }
    [v, nv] = [nv, v];
  }
  return h;
}

// The excitation: a random staircase (levels in [−1, 1], held 3–40 steps), the usual test input for system identification.
export function staircase(n, seed = 5) {
  const r = rng(seed);
  const u = new Float64Array(n);
  let t = 0;
  while (t < n) {
    const v = 2 * r.uniform() - 1;
    const len = 3 + Math.floor(r.uniform() * 38);
    for (let k = 0; k < len && t < n; k++) u[t++] = v;
  }
  return u;
}

// Train a reservoir on (u, y) of the plant and read it back as (A, B, C, D).
// λ is chosen on the validation part (trainEnd … valEnd); the test part (valEnd … n) is scored.
export function identify(sys, { rho = 0.99, leak = 0.5, inScale = 0.01, seed = 1 } = {}, u = staircase(CTRL.n)) {
  const { n, washout, trainEnd, valEnd, N, degree } = CTRL;
  const y = simulate(sys, u);
  const res = new Reservoir({ N, degree, seed }).configure({ rho, leak, inScale, bias: 0 });
  const X = res.run(u, n);
  const { F, p } = res.features(u, X, n);
  const G = gram(F, y, p, 1, washout, trainEnd);
  let best = null;
  for (const lambda of LAMBDAS) {
    const B = solveRidge(G, lambda);
    const v = nrmse(y, applyReadout(F, B, p, 1, trainEnd, valEnd), valEnd - trainEnd, 1, 0, 0, trainEnd);
    if (!best || v < best.v) best = { v, B, lambda };
  }
  const B_ = best.B;
  const P = applyReadout(F, B_, p, 1, 0, n);
  const err = nrmse(y, P, n - valEnd, 1, 0, valEnd, valEnd);
  // the readout split by what it reads: the constant, the input (D), the states (C)
  const model = linearize(res, B_);
  return { u, y, P, err, lambda: best.lambda, X, res, readout: B_, model };
}

// The reservoir at x = 0 with b = 0, where tanh′ = 1: its (A, B, C, D) and the constant term of the readout.
export function linearize(res, readout) {
  const { N, deg, idx, w, win, leak } = res;
  const A = Array.from({ length: N }, (_, i) => {
    const row = new Float64Array(N);
    row[i] = 1 - leak;
    for (let k = i * deg; k < (i + 1) * deg; k++) row[idx[k]] += leak * w[k];
    return row;
  });
  const B = Float64Array.from(win, (v) => leak * v);
  const C = readout.slice(2, 2 + N);
  return { A, B, C, D: readout[1], c0: readout[0], n: N };
}

// Singular values of the Hankel matrix [h(i + j + 1)], i, j < m: how many states the input–output map really needs.
// Computed as √eig(HᵀH); H is symmetric, so these are |eig(H)|, sorted.
export function hankelSV(h, m, eig) {
  const H = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => h[i + j + 1]));
  return eig(H).map((e) => Math.hypot(e.re, e.im)).sort((a, b) => b - a);
}

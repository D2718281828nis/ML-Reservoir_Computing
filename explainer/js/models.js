// The models compared on the page, all in plain typed arrays:
//  * Reservoir — an echo state network (ESN): fixed random recurrent network, trained linear readout;
//  * RandomFeatures — an extreme learning machine (ELM): a window through a fixed random tanh layer, trained readout;
//  * NG-RC — a window through fixed polynomial features, trained readout;
//  * FFN — a multilayer perceptron on a window, every weight trained with backprop and Adam;
//  * linear features of a window (an autoregressive model) as the simplest readout-only baseline.

import { rng } from "./util.js";
import { Dense, Net, mseGrad, gram, solveRidge, applyReadout } from "./nn.js";

// ───────────── echo state network ─────────────
// Largest |eigenvalue| of a sparse matrix (row i: columns idx[i·deg + k], values w[i·deg + k]) by power iteration.
// The leading eigenvalues of a random matrix are often a complex pair, so the norm of Wᵏv oscillates instead of
// settling; its average growth rate per step still converges to the spectral radius.
export function spectralRadius(idx, w, N, deg, iters = 1000, burn = 200) {
  const r = rng(12345);
  let v = Float64Array.from({ length: N }, () => r.normal());
  let u = new Float64Array(N);
  let acc = 0;
  for (let t = 0; t < iters; t++) {
    let nrm = 0;
    for (let i = 0; i < N; i++) {
      let a = 0;
      for (let k = i * deg, e = k + deg; k < e; k++) a += w[k] * v[idx[k]];
      u[i] = a;
      nrm += a * a;
    }
    nrm = Math.sqrt(nrm) || 1e-300;
    if (t >= burn) acc += Math.log(nrm);
    for (let i = 0; i < N; i++) u[i] /= nrm;
    [v, u] = [u, v];
  }
  return Math.exp(acc / (iters - burn));
}

export class Reservoir {
  // Every neuron listens to `degree` random others. The random draws are made once; configure() only rescales them,
  // so moving a slider changes ρ or the input gain of the *same* reservoir.
  constructor({ N = 300, degree = 10, nIn = 1, seed = 1, ...rest } = {}) {
    const r = rng(seed);
    const deg = Math.min(degree, N);
    this.N = N;
    this.nIn = nIn;
    this.deg = deg;
    this.idx = new Int32Array(N * deg);
    this.w0 = new Float64Array(N * deg);
    const used = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const picked = [];
      while (picked.length < deg) {
        const c = Math.floor(r.uniform() * N);
        if (used[c]) continue;
        used[c] = 1;
        picked.push(c);
      }
      picked.forEach((c, k) => {
        used[c] = 0;
        this.idx[i * deg + k] = c;
        this.w0[i * deg + k] = 2 * r.uniform() - 1;
      });
    }
    this.win0 = Float64Array.from({ length: N * nIn }, () => 2 * r.uniform() - 1);
    this.b0 = Float64Array.from({ length: N }, () => 2 * r.uniform() - 1);
    this.rho0 = spectralRadius(this.idx, this.w0, N, deg);
    this.configure({ rho: 0.9, leak: 0.5, inScale: 0.5, bias: 0.2, ...rest });
  }

  configure({ rho = this.rho, leak = this.leak, inScale = this.inScale, bias = this.bias }) {
    Object.assign(this, { rho, leak, inScale, bias });
    const f = rho / this.rho0;
    this.w = this.w0.map((v) => v * f);
    this.win = this.win0.map((v) => v * inScale);
    this.b = this.b0.map((v) => v * bias);
    return this;
  }

  get nTrained() {
    return 1 + this.nIn + this.N; // one readout weight per feature [1, u, x], per output
  }

  // x ← (1 − α) x + α tanh(W_in u + W x + b), in place. u: array-like of length nIn.
  step(x, u, tmp = new Float64Array(this.N)) {
    const { N, nIn, deg, idx, w, win, b, leak } = this;
    for (let i = 0; i < N; i++) {
      let a = b[i];
      for (let c = 0; c < nIn; c++) a += win[i * nIn + c] * u[c];
      for (let k = i * deg, e = k + deg; k < e; k++) a += w[k] * x[idx[k]];
      tmp[i] = Math.tanh(a);
    }
    for (let i = 0; i < N; i++) x[i] = (1 - leak) * x[i] + leak * tmp[i];
    return x;
  }

  // States for inputs U (T × nIn, row-major): row t is the state right after reading u(t). Starts from x0 (zeros).
  run(U, T, x0 = null) {
    const { N, nIn } = this;
    const X = new Float64Array(T * N);
    const x = x0 ? Float64Array.from(x0) : new Float64Array(N);
    const tmp = new Float64Array(N);
    const u = new Float64Array(nIn);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < nIn; c++) u[c] = U[t * nIn + c];
      this.step(x, u, tmp);
      X.set(x, t * N);
    }
    this.last = x;
    return X;
  }

  // Design matrix of the readout: row t = [1, u(t), x(t)].
  features(U, X, T) {
    const { N, nIn } = this;
    const p = 1 + nIn + N;
    const F = new Float64Array(T * p);
    for (let t = 0; t < T; t++) {
      const o = t * p;
      F[o] = 1;
      for (let c = 0; c < nIn; c++) F[o + 1 + c] = U[t * nIn + c];
      F.set(X.subarray(t * N, (t + 1) * N), o + 1 + nIn);
    }
    return { F, p };
  }
}

// ───────────── readout-only models on a window ─────────────
// Linear features of a window: [1, z(t), …, z(t−L+1)]. A ridge readout on them is a linear autoregressive model.
export function linearFeatures(Xw, n, L) {
  const p = 1 + L;
  const F = new Float64Array(n * p);
  for (let r = 0; r < n; r++) {
    F[r * p] = 1;
    for (let j = 0; j < L; j++) F[r * p + 1 + j] = Xw[r * L + j];
  }
  return { F, p };
}

// Extreme learning machine: a window through M fixed random tanh units, features [1, window, tanh(A w + c)].
export class RandomFeatures {
  constructor(L, M = 300, { seed = 1, scale = 1.5 } = {}) {
    const r = rng(seed);
    this.L = L;
    this.M = M;
    const g = scale / Math.sqrt(L);
    this.A = Float64Array.from({ length: M * L }, () => (2 * r.uniform() - 1) * g);
    this.c = Float64Array.from({ length: M }, () => r.uniform() - 0.5);
  }
  get nTrained() {
    return 1 + this.L + this.M;
  }
  features(Xw, n) {
    const { L, M, A, c } = this;
    const p = 1 + L + M;
    const F = new Float64Array(n * p);
    for (let r = 0; r < n; r++) {
      const o = r * p;
      const wo = r * L;
      F[o] = 1;
      for (let j = 0; j < L; j++) F[o + 1 + j] = Xw[wo + j];
      for (let m = 0; m < M; m++) {
        let a = c[m];
        for (let j = 0; j < L; j++) a += A[m * L + j] * Xw[wo + j];
        F[o + 1 + L + m] = Math.tanh(a);
      }
    }
    return { F, p };
  }
}

// ───────────── NG-RC (nonlinear vector autoregression) ─────────────
// All multisets of size `deg` from m variables, as index tuples (i ≤ j ≤ …): the unique monomials of that degree.
export function monomials(m, deg) {
  const out = [];
  const rec = (start, cur) => {
    if (cur.length === deg) return out.push(cur.slice());
    for (let i = start; i < m; i++) {
      cur.push(i);
      rec(i, cur);
      cur.pop();
    }
  };
  rec(0, []);
  return out;
}
export const binom = (n, k) => {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
};

export class NGRC {
  // d input channels, k delay taps spaced s steps apart, monomials of the listed degrees, optional constant.
  constructor({ d = 3, k = 2, s = 1, degrees = [2], constant = true } = {}) {
    Object.assign(this, { d, k, s, degrees, constant });
    this.nLin = d * k;
    this.combos = degrees.flatMap((deg) => monomials(this.nLin, deg));
    this.p = (constant ? 1 : 0) + this.nLin + this.combos.length;
    this.warmup = (k - 1) * s; // past points needed before the first feature vector exists
  }
  get nTrained() {
    return this.p * this.d;
  }
  // Feature vector from the linear part lin = [v(t), v(t−s), …] (length d·k), written into out at offset o.
  row(lin, out, o = 0) {
    let j = o;
    if (this.constant) out[j++] = 1;
    for (let i = 0; i < this.nLin; i++) out[j++] = lin[i];
    for (const c of this.combos) {
      let v = 1;
      for (const i of c) v *= lin[i];
      out[j++] = v;
    }
    return out;
  }
  // Rows for t in [t0, t1) of the series V (T × d); requires t0 ≥ warmup.
  features(V, t0, t1) {
    const { d, k, s, p } = this;
    const F = new Float64Array((t1 - t0) * p);
    const lin = new Float64Array(d * k);
    for (let t = t0; t < t1; t++) {
      for (let j = 0; j < k; j++) for (let c = 0; c < d; c++) lin[j * d + c] = V[(t - j * s) * d + c];
      this.row(lin, F, (t - t0) * p);
    }
    return F;
  }
}

// ───────────── feed-forward network ─────────────
export class FFN {
  // window L → hidden tanh layers → H outputs (one per horizon), every weight trained
  constructor(L, hidden = [32, 32], H = 1, seed = 1) {
    const sizes = [L, ...hidden, H];
    this.L = L;
    this.H = H;
    this.net = new Net(sizes.slice(1).map((o, i) => new Dense(sizes[i], o, i < hidden.length ? "tanh" : "linear", seed * 100 + i)));
    this.epochs = 0;
  }
  get nTrained() {
    return this.net.nParams;
  }
  // One pass over the n training windows in random mini-batches; returns the mean training loss.
  epoch(X, Y, n, { lr = 2e-3, batch = 32, r }) {
    const { L, H, net } = this;
    const order = Uint32Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(r.uniform() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let total = 0;
    for (let b0 = 0; b0 < n; b0 += batch) {
      const m = Math.min(batch, n - b0);
      const xb = new Float32Array(m * L);
      const yb = new Float32Array(m * H);
      for (let q = 0; q < m; q++) {
        const row = order[b0 + q];
        for (let j = 0; j < L; j++) xb[q * L + j] = X[row * L + j];
        for (let j = 0; j < H; j++) yb[q * H + j] = Y[row * H + j];
      }
      const { loss, grad } = mseGrad(net.forward(xb, m), yb);
      net.zeroGrad();
      net.backward(grad);
      net.step(lr);
      total += loss * m;
    }
    this.epochs++;
    return total / n;
  }
  predict(X, n) {
    return Float64Array.from(this.net.forward(X, n));
  }
}

// ───────────── fitting a readout ─────────────
// Fit a ridge readout on training rows [t0, t1) and pick λ from `lambdas` by validation error on rows [v0, v1).
export function fitReadout(F, p, Y, q, [t0, t1], [v0, v1], lambdas) {
  const g = gram(F, Y, p, q, t0, t1);
  let best = null;
  for (const lambda of lambdas) {
    const B = solveRidge(g, lambda);
    const P = applyReadout(F, B, p, q, v0, v1);
    let se = 0;
    for (let i = 0; i < P.length; i++) se += (P[i] - Y[v0 * q + i]) ** 2;
    const mse = se / P.length;
    if (!best || mse < best.mse) best = { B, lambda, mse };
  }
  return best;
}

// ───────────── metrics ─────────────
// Root-mean-square error over rows divided by the standard deviation of the truth, for column `col` of q.
export function nrmse(Y, P, n, q = 1, col = 0, pOff = 0, yOff = 0) {
  let m = 0;
  for (let r = 0; r < n; r++) m += Y[(yOff + r) * q + col];
  m /= n;
  let se = 0;
  let vv = 0;
  for (let r = 0; r < n; r++) {
    const y = Y[(yOff + r) * q + col];
    se += (P[(pOff + r) * q + col] - y) ** 2;
    vv += (y - m) ** 2;
  }
  return Math.sqrt(se / (vv || 1e-300));
}

export function nrmseByHorizon(Y, P, n, H, pOff = 0, yOff = 0) {
  return Float64Array.from({ length: H }, (_, h) => nrmse(Y, P, n, H, h, pOff, yOff));
}

// First step at which the error exceeds `thr` (in units of the truth's standard deviation), or the length if never.
export function validSteps(truth, pred, n, d = 1, thr = 0.4) {
  const mu = new Float64Array(d);
  for (let t = 0; t < n; t++) for (let c = 0; c < d; c++) mu[c] += truth[t * d + c] / n;
  let v = 0;
  for (let t = 0; t < n; t++) for (let c = 0; c < d; c++) v += (truth[t * d + c] - mu[c]) ** 2 / n;
  for (let t = 0; t < n; t++) {
    let e = 0;
    for (let c = 0; c < d; c++) e += (truth[t * d + c] - pred[t * d + c]) ** 2;
    if (!(Math.sqrt(e / v) <= thr)) return t;
  }
  return n;
}

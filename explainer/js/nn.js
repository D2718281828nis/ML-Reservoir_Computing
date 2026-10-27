// Small dense-network math in plain typed arrays.
//  * a trainable Dense/Net with backprop and Adam: the feed-forward network of parts 2, 4 and 5;
//  * ridge least squares through Gram matrices and Cholesky: the readout of every reservoir model.

import { rng } from "./util.js";

// ───────────── trainable layers ─────────────
const ACT = {
  linear: { f: (z) => z, df: () => 1 },
  tanh: { f: Math.tanh, df: (a) => 1 - a * a }, // derivative written in terms of the output a
};

export class Dense {
  constructor(nIn, nOut, act = "linear", seed = 1) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.act = ACT[act];
    const r = rng(seed);
    const bound = 1 / Math.sqrt(nIn); // PyTorch's default nn.Linear init
    this.W = Float32Array.from({ length: nIn * nOut }, () => (2 * r.uniform() - 1) * bound);
    this.b = Float32Array.from({ length: nOut }, () => (2 * r.uniform() - 1) * bound);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(nOut);
    this.mW = new Float32Array(this.W.length);
    this.vW = new Float32Array(this.W.length);
    this.mb = new Float32Array(nOut);
    this.vb = new Float32Array(nOut);
  }

  // X: Float32Array(n * nIn) → A: Float32Array(n * nOut)
  forward(X, n) {
    const { nIn, nOut, W, b } = this;
    const A = new Float32Array(n * nOut);
    for (let r = 0; r < n; r++) {
      const xo = r * nIn;
      for (let o = 0; o < nOut; o++) {
        let acc = b[o];
        const wo = o * nIn;
        for (let i = 0; i < nIn; i++) acc += W[wo + i] * X[xo + i];
        A[r * nOut + o] = this.act.f(acc);
      }
    }
    this.X = X;
    this.A = A;
    this.n = n;
    return A;
  }

  // dA: gradient w.r.t. this layer's output. Accumulates gW, gb; returns gradient w.r.t. input.
  backward(dA) {
    const { nIn, nOut, W, X, A, n } = this;
    const dX = new Float32Array(n * nIn);
    for (let r = 0; r < n; r++) {
      for (let o = 0; o < nOut; o++) {
        const k = r * nOut + o;
        const dz = dA[k] * this.act.df(A[k]);
        if (dz === 0) continue;
        this.gb[o] += dz;
        const wo = o * nIn;
        const xo = r * nIn;
        for (let i = 0; i < nIn; i++) {
          this.gW[wo + i] += dz * X[xo + i];
          dX[xo + i] += dz * W[wo + i];
        }
      }
    }
    return dX;
  }

  zeroGrad() {
    this.gW.fill(0);
    this.gb.fill(0);
  }

  adam(lr, t, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    const c1 = 1 - b1 ** t;
    const c2 = 1 - b2 ** t;
    const upd = (p, g, m, v) => {
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    };
    upd(this.W, this.gW, this.mW, this.vW);
    upd(this.b, this.gb, this.mb, this.vb);
  }
}

export class Net {
  constructor(layers) {
    this.layers = layers;
    this.t = 0;
  }
  forward(X, n) {
    let a = X;
    for (const l of this.layers) a = l.forward(a, n);
    return a;
  }
  backward(dOut) {
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) d = this.layers[i].backward(d);
    return d;
  }
  zeroGrad() {
    for (const l of this.layers) l.zeroGrad();
  }
  step(lr) {
    this.t++;
    for (const l of this.layers) l.adam(lr, this.t);
  }
  get nParams() {
    return this.layers.reduce((a, l) => a + l.W.length + l.b.length, 0);
  }
  // copies of every weight, for early stopping on the validation set
  snapshot() {
    return this.layers.map((l) => [l.W.slice(), l.b.slice()]);
  }
  restore(snap) {
    snap.forEach(([W, b], i) => {
      this.layers[i].W.set(W);
      this.layers[i].b.set(b);
    });
  }
}

// mean squared error and its gradient for pred vs target (same length)
export function mseGrad(pred, target) {
  const g = new Float32Array(pred.length);
  let loss = 0;
  const scale = 2 / pred.length;
  for (let i = 0; i < pred.length; i++) {
    const d = pred[i] - target[i];
    loss += d * d;
    g[i] = scale * d;
  }
  return { loss: loss / pred.length, grad: g };
}

// ───────────── least squares ─────────────
// Gram matrices of a row-major design F (rows × p) and targets Y (rows × q), summed over rows [r0, r1).
// Computed once, they let solveRidge try many λ without another pass over the data.
export function gram(F, Y, p, q, r0, r1) {
  const G = new Float64Array(p * p);
  const R = new Float64Array(p * q);
  for (let r = r0; r < r1; r++) {
    const fo = r * p;
    const yo = r * q;
    for (let i = 0; i < p; i++) {
      const fi = F[fo + i];
      if (fi === 0) continue;
      const gi = i * p;
      for (let j = i; j < p; j++) G[gi + j] += fi * F[fo + j];
      const ri = i * q;
      for (let j = 0; j < q; j++) R[ri + j] += fi * Y[yo + j];
    }
  }
  for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) G[i * p + j] = G[j * p + i];
  return { G, R, n: r1 - r0, p, q };
}

// Solve (G + λ n I) B = R for B (p × q), i.e. the readout that minimises (1/n) Σ‖Bᵀf − y‖² + λ‖B‖².
export function solveRidge({ G, R, n, p, q }, lambda) {
  const C = new Float64Array(p * p); // Cholesky factor (lower) of G + λ n I
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = G[i * p + j] + (i === j ? lambda * n : 0);
      for (let k = 0; k < j; k++) sum -= C[i * p + k] * C[j * p + k];
      C[i * p + j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / C[j * p + j];
    }
  }
  const B = new Float64Array(p * q);
  const tmp = new Float64Array(p);
  for (let col = 0; col < q; col++) {
    for (let i = 0; i < p; i++) {
      let sum = R[i * q + col];
      for (let k = 0; k < i; k++) sum -= C[i * p + k] * tmp[k];
      tmp[i] = sum / C[i * p + i];
    }
    for (let i = p - 1; i >= 0; i--) {
      let sum = tmp[i];
      for (let k = i + 1; k < p; k++) sum -= C[k * p + i] * B[k * q + col];
      B[i * q + col] = sum / C[i * p + i];
    }
  }
  return B;
}

export const ridge = (F, Y, p, q, r0, r1, lambda) => solveRidge(gram(F, Y, p, q, r0, r1), lambda);

// Y = F B for rows [r0, r1); returns Float64Array((r1 − r0) × q).
export function applyReadout(F, B, p, q, r0, r1) {
  const out = new Float64Array((r1 - r0) * q);
  for (let r = r0; r < r1; r++) {
    const fo = r * p;
    const oo = (r - r0) * q;
    for (let i = 0; i < p; i++) {
      const fi = F[fo + i];
      if (fi === 0) continue;
      const bi = i * q;
      for (let j = 0; j < q; j++) out[oo + j] += fi * B[bi + j];
    }
  }
  return out;
}

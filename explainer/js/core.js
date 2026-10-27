// Every experiment on the page, without any DOM: the sections draw what these functions return.
// Node can import this file directly (tools/check.mjs, calibration scripts).

import { rng } from "./util.js";
import { standardize, windowMatrix, targetMatrix, mackeyGlass, lorenz, LORENZ_LYAPUNOV } from "./data.js";
import { Reservoir, RandomFeatures, NGRC, FFN, linearFeatures, fitReadout, nrmse, nrmseByHorizon, validSteps } from "./models.js";
import { applyReadout, gram, solveRidge } from "./nn.js";

export const LAMBDAS = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
export const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

// ───────────── a supervised forecasting task ─────────────
// Forecast origins t with targets Y (rows × q); chronological train / validation / test ranges of origins.
// Every model sees the same origins: from t0 = 50 on (the reservoir's washout), so windows of up to 51 values fit.
function task(z, q, targetsOf, { trainEnd, valEnd, end, lead }) {
  const t0 = 50;
  return {
    z, q, Y: targetsOf(end - lead), end, lead,
    train: [t0, trainEnd - lead], val: [trainEnd, valEnd - lead], test: [valEnd, end - lead],
  };
}
const rowsOf = ([a, b]) => b - a;

// Window features for every origin; rows before the first full window stay zero (they are never used).
function windows(z, L, n) {
  const X = windowMatrix(z, L, 0, n);
  for (let t = 0; t < Math.min(L - 1, n); t++) X.fill(0, t * L, (t + 1) * L);
  return X;
}

// ───────────── parts 2 and 4: Mackey–Glass, one value H steps ahead ─────────────
export const MG = { n: 4000, trainEnd: 2500, valEnd: 3000, H: 20 };
let mgCache = null;
export function mackey() {
  if (!mgCache) {
    const x = mackeyGlass(MG.n);
    mgCache = { x, ...standardize(x, 0, MG.trainEnd) };
  }
  return mgCache;
}
export function mgTask(H = MG.H) {
  const { z } = mackey();
  return task(z, 1, (n) => Float64Array.from({ length: n }, (_, t) => z[t + H]), { ...MG, end: MG.n, lead: H });
}

// Readout-only models: fit on train, choose λ on validation, score on test.
function scoreReadout(T, F, p, t1) {
  const fit = fitReadout(F, p, T.Y, T.q, T.train, T.val, LAMBDAS);
  const P = applyReadout(F, fit.B, p, T.q, ...T.test);
  const err = T.q === 1 ? nrmse(T.Y, P, rowsOf(T.test), 1, 0, 0, T.test[0]) : nrmseByHorizon(T.Y, P, rowsOf(T.test), T.q, 0, T.test[0]);
  return { P, err, lambda: fit.lambda, time: now() - t1, nTrained: p * T.q };
}
export function fitLinear(T, L) {
  const t1 = now();
  const n = T.end - T.lead;
  const { F, p } = linearFeatures(windows(T.z, L, n), n, L);
  return scoreReadout(T, F, p, t1);
}
export function fitELM(T, L, { M = 300, seed = 1 } = {}) {
  const t1 = now();
  const n = T.end - T.lead;
  const elm = new RandomFeatures(L, M, { seed });
  const { F, p } = elm.features(windows(T.z, L, n), n);
  return scoreReadout(T, F, p, t1);
}
export const ESN_MG = { N: 300, rho: 1.0, leak: 0.5, inScale: 0.5, bias: 0.2 };
export function fitESN(T, cfg = ESN_MG, seed = 1) {
  const t1 = now();
  const n = T.end - T.lead;
  const res = new Reservoir({ N: cfg.N, seed }).configure(cfg);
  const X = res.run(T.z, n);
  const { F, p } = res.features(T.z, X, n);
  return { ...scoreReadout(T, F, p, t1), res };
}

// Feed-forward network on a window, trained with Adam in budgeted slices (so the page can redraw between them).
// Keeps the weights of the epoch with the lowest validation error; `history` has the error curves.
export class Trainer {
  constructor(T, L, { hidden = [32, 32], seed = 1, lr = 2e-3, epochs = 150 } = {}) {
    this.T = T;
    this.L = L;
    this.lr = lr;
    this.maxEpochs = epochs;
    this.ffn = new FFN(L, hidden, T.q, seed);
    this.r = rng(seed + 99);
    const X = windows(T.z, L, T.end - T.lead);
    const part = ([a, b]) => ({ X: X.subarray(a * L, b * L), Y: T.Y.subarray(a * T.q, b * T.q), n: b - a });
    this.tr = part(T.train);
    this.va = part(T.val);
    this.te = part(T.test);
    this.best = { mse: Infinity, epoch: 0, snap: null };
    this.history = [];
    this.time = 0;
  }
  get epochs() {
    return this.ffn.epochs;
  }
  get done() {
    return this.ffn.epochs >= this.maxEpochs;
  }
  get nTrained() {
    return this.ffn.nTrained;
  }
  score(part) {
    const P = this.ffn.predict(part.X, part.n);
    let se = 0;
    for (let i = 0; i < P.length; i++) se += (P[i] - part.Y[i]) ** 2;
    return { P, mse: se / P.length };
  }
  run(budget = 30, track = true) {
    const t1 = now();
    while (!this.done && now() - t1 < budget) {
      const s = now();
      const loss = this.ffn.epoch(this.tr.X, this.tr.Y, this.tr.n, { lr: this.lr, r: this.r });
      this.time += now() - s;
      const v = this.score(this.va);
      if (v.mse < this.best.mse) this.best = { mse: v.mse, epoch: this.ffn.epochs, snap: this.ffn.net.snapshot() };
      if (track) this.history.push({ epoch: this.ffn.epochs, time: this.time, train: loss, val: v.mse, test: this.testError(this.score(this.te).P) });
    }
  }
  testError(P) {
    const { T } = this;
    return T.q === 1 ? nrmse(this.te.Y, P, this.te.n) : nrmseByHorizon(this.te.Y, P, this.te.n, T.q);
  }
  // test predictions and error of the best-validation weights
  result() {
    const live = this.ffn.net.snapshot();
    if (this.best.snap) this.ffn.net.restore(this.best.snap);
    const { P } = this.score(this.te);
    this.ffn.net.restore(live);
    return { P, err: this.testError(P), time: this.time, nTrained: this.nTrained, bestEpoch: this.best.epoch };
  }
  current() {
    return this.score(this.te).P;
  }
}

// ───────────── part 5.2: the forecasting lab, H horizons at once ─────────────
// One preset per dataset; splits are chronological.
export function presets(series) {
  const out = {
    mackey: {
      x: mackey().x, trainEnd: MG.trainEnd, valEnd: MG.valEnd, H: 50, period: 0, L: 20, hidden: [32], epochs: 120, lr: 2e-3,
      esn: ESN_MG, step: 1,
    },
  };
  const elec = series?.electricity;
  if (elec) {
    const start = Date.parse(`${elec.start}T00:00:00Z`);
    const dayIndex = (iso) => Math.round((Date.parse(`${iso}T00:00:00Z`) - start) / 864e5);
    out.electricity = {
      x: Float64Array.from(elec.values), trainEnd: dayIndex("2017-01-01"), valEnd: dayIndex("2018-01-01"), H: 7, period: 7, L: 28,
      hidden: [32], epochs: 300, lr: 1e-3, esn: { N: 200, rho: 0.9, leak: 1.0, inScale: 0.5, bias: 0.2 }, unit: elec.unit, start,
    };
  }
  const ecg = series?.ecg;
  if (ecg) {
    const n = ecg.values.length;
    out.ecg = {
      x: Float64Array.from(ecg.values), trainEnd: Math.round(n * 0.6), valEnd: Math.round(n * 0.8), H: 25, period: 0, L: 50,
      hidden: [32], epochs: 100, lr: 1e-3, esn: { N: 300, rho: 0.9, leak: 0.3, inScale: 0.5, bias: 0.2 }, unit: ecg.unit, fs: ecg.fs,
    };
  }
  return out;
}

export function labTask(P) {
  const { z, mu, sd } = standardize(P.x, 0, P.trainEnd);
  const end = P.x.length;
  const T = task(z, P.H, (n) => targetMatrix(z, P.H, 0, n), { trainEnd: P.trainEnd, valEnd: P.valEnd, end, lead: P.H });
  return { ...T, P, mu: mu[0], sd: sd[0] };
}

// persistence ŷ(t+h) = z(t); seasonal naive ŷ(t+h) = z(t+h−period) (the same day last week, for the daily load)
export function baselines(T) {
  const { z, q: H, test } = T;
  const period = T.P.period;
  const n = rowsOf(test);
  const pers = new Float64Array(n * H);
  const seas = period ? new Float64Array(n * H) : null;
  for (let r = 0; r < n; r++) {
    const t = test[0] + r;
    for (let h = 1; h <= H; h++) {
      pers[r * H + h - 1] = z[t];
      if (seas) seas[r * H + h - 1] = z[t + h - period * Math.ceil(h / period)];
    }
  }
  const err = (P) => nrmseByHorizon(T.Y, P, n, H, 0, test[0]);
  return { persistence: { P: pers, err: err(pers) }, seasonal: seas ? { P: seas, err: err(seas) } : null };
}

// ───────────── part 5.3: closed loop on Mackey–Glass ─────────────
// One-step models that run on their own output. "Valid" = |error| ≤ 0.2 standard deviations of the true series.
// The ESN's λ is chosen by the length of its valid free run on the validation part, not by its one-step error.
export const FREE = { steps: 1000, thr: 0.2, L: 20 };

export function esnFreeRun({ seed = 1, ...cfg } = {}) {
  const t1 = now();
  const { z } = mackey();
  const { trainEnd, valEnd } = MG;
  const { steps, thr } = FREE;
  const c = { ...ESN_MG, ...cfg };
  const N = c.N;
  const res = new Reservoir({ N, seed }).configure(c);
  const X = res.run(z, valEnd);
  const { F, p } = res.features(z, X, valEnd);
  const Y = Float64Array.from({ length: valEnd }, (_, t) => z[t + 1]);
  const g = gram(F, Y, p, 1, 50, trainEnd - 1);
  const free = (B, from, n) => {
    // start from the state after reading the true z(from) and feed back each prediction
    const x = Float64Array.from(X.subarray(from * N, (from + 1) * N));
    const out = new Float64Array(n);
    const tmp = new Float64Array(N);
    const u = new Float64Array(1);
    let y = 0;
    for (let i = 0; i < p; i++) y += F[from * p + i] * B[i];
    for (let i = 0; i < n; i++) {
      out[i] = y;
      u[0] = y;
      res.step(x, u, tmp);
      y = B[0] + B[1] * y;
      for (let j = 0; j < N; j++) y += B[2 + j] * x[j];
    }
    return out;
  };
  let best = null;
  for (const lambda of LAMBDAS) {
    const B = solveRidge(g, lambda);
    const valid = validSteps(z.subarray(trainEnd, valEnd), free(B, trainEnd - 1, valEnd - trainEnd), valEnd - trainEnd, 1, thr);
    if (!best || valid > best.valid) best = { B, lambda, valid };
  }
  const pred = free(best.B, valEnd - 1, steps);
  return { pred, lambda: best.lambda, time: now() - t1, valid: validSteps(z.subarray(valEnd, valEnd + steps), pred, steps, 1, thr) };
}

export function freeTrainer(seed = 1) {
  const T = mgTask(1);
  return new Trainer(T, FREE.L, { hidden: [32, 32], seed, epochs: 150 });
}
export function ffnFreeRun(trainer) {
  const { z } = mackey();
  const { valEnd } = MG;
  const { steps, thr, L } = FREE;
  const ffn = trainer.ffn;
  const live = ffn.net.snapshot();
  if (trainer.best.snap) ffn.net.restore(trainer.best.snap);
  const win = new Float64Array(L);
  for (let j = 0; j < L; j++) win[j] = z[valEnd - 1 - j];
  const pred = new Float64Array(steps);
  for (let i = 0; i < steps; i++) {
    const y = ffn.predict(win, 1)[0];
    pred[i] = y;
    win.copyWithin(1, 0, L - 1);
    win[0] = y;
  }
  ffn.net.restore(live);
  return { pred, valid: validSteps(z.subarray(valEnd, valEnd + steps), pred, steps, 1, thr) };
}

// ───────────── part 6: Lorenz-63, NG-RC against an ESN ─────────────
// Train on [warm, warm + ntr), then run free from 5 starting points after the training data. One chaotic forecast
// depends a lot on where it starts, so the valid time is averaged over the 5; the first run is the one drawn.
export const LZ = { dt: 0.025, warm: 200, maxTrain: 4000, steps: 2400, starts: 5, gap: 250, testSteps: 600, thr: 0.4 };
export const LZ_STEPS_PER_LT = 1 / LORENZ_LYAPUNOV / LZ.dt; // ≈ 44 steps per Lyapunov time
let lzCache = null;
export function lorenzData() {
  lzCache ||= lorenz(LZ.warm + LZ.maxTrain + LZ.gap * (LZ.starts - 1) + LZ.steps + 2);
  return lzCache;
}
const lzStarts = (ts) => Array.from({ length: LZ.starts }, (_, j) => ts + j * LZ.gap);

export function lorenzNGRC({ ntr = 400, k = 2, quad = true, lambda = 2.5e-7 } = {}) {
  const t1 = now();
  const V = lorenzData();
  const { warm } = LZ;
  const ts = warm + ntr;
  const { z: Z, mu, sd } = standardize(V, 0, ts, 3);
  const ng = new NGRC({ d: 3, k, s: 1, degrees: quad ? [2] : [] });
  // rows t in [warm, ts − 1): the last target is v(ts − 1), as for the ESN; the free run starts at ts
  const n = ntr - 1;
  const F = ng.features(Z, warm, warm + n);
  const Y = new Float64Array(n * 3); // the model predicts the step v(t+1) − v(t)
  for (let r = 0; r < n; r++) for (let c = 0; c < 3; c++) Y[r * 3 + c] = Z[(warm + r + 1) * 3 + c] - Z[(warm + r) * 3 + c];
  const B = solveRidge(gram(F, Y, ng.p, 3, 0, n), lambda);
  const fitTime = now() - t1;
  // free run from t0: the last k true points seed the delay line
  const run = (t0, steps) => {
    const lin = new Float64Array(3 * k);
    const f = new Float64Array(ng.p);
    const hist = new Float64Array((steps + k) * 3);
    hist.set(Z.subarray((t0 - k) * 3, t0 * 3));
    const pred = new Float64Array(steps * 3);
    for (let i = 0; i < steps; i++) {
      const cur = k + i - 1; // index in hist of v(t)
      for (let j = 0; j < k; j++) for (let c = 0; c < 3; c++) lin[j * 3 + c] = hist[(cur - j) * 3 + c];
      ng.row(lin, f);
      for (let c = 0; c < 3; c++) {
        let a = hist[cur * 3 + c];
        for (let q = 0; q < ng.p; q++) a += f[q] * B[q * 3 + c];
        hist[(cur + 1) * 3 + c] = a;
        pred[i * 3 + c] = a * sd[c] + mu[c];
      }
    }
    return pred;
  };
  return finishLorenz(run, ts, { fitTime, p: ng.p, nTrained: ng.nTrained, warmup: k - 1 });
}

export const ESN_LZ = { N: 300, rho: 0.9, leak: 0.5, inScale: 0.5, bias: 0.5 };
export function lorenzESN({ ntr = 400, seed = 1, lambda = 1e-9 } = {}) {
  const t1 = now();
  const V = lorenzData();
  const { warm } = LZ;
  const ts = warm + ntr;
  const last = lzStarts(ts).at(-1);
  const { z: Z, mu, sd } = standardize(V, 0, ts, 3);
  const res = new Reservoir({ N: ESN_LZ.N, nIn: 3, seed }).configure(ESN_LZ);
  const N = res.N;
  const X = res.run(Z, last); // teacher-forced states up to the last starting point (only [warm, ts) is used to fit)
  const { F, p } = res.features(Z, X, last);
  const Y = Z.slice(3, (last + 1) * 3); // row t: z(t+1)
  const B = solveRidge(gram(F, Y, p, 3, warm, ts - 1), lambda);
  const fitTime = now() - t1;
  const run = (t0, steps) => {
    const x = Float64Array.from(X.subarray((t0 - 1) * N, t0 * N));
    const tmp = new Float64Array(N);
    const u = new Float64Array(3);
    const f = Float64Array.from(F.subarray((t0 - 1) * p, t0 * p));
    const pred = new Float64Array(steps * 3);
    for (let i = 0; i < steps; i++) {
      for (let c = 0; c < 3; c++) {
        let a = 0;
        for (let q = 0; q < p; q++) a += f[q] * B[q * 3 + c];
        u[c] = Number.isFinite(a) ? Math.max(-50, Math.min(50, a)) : 0;
        pred[i * 3 + c] = Number.isFinite(a) ? a * sd[c] + mu[c] : NaN;
      }
      res.step(x, u, tmp);
      f.set(u, 1);
      f.set(x, 4);
    }
    return pred;
  };
  return finishLorenz(run, ts, { fitTime, p, nTrained: p * 3, warmup: warm });
}

function finishLorenz(run, ts, info) {
  const V = lorenzData();
  const truthAt = (t0, n) => V.subarray(t0 * 3, (t0 + n) * 3);
  const pred = run(ts, LZ.steps);
  const valids = lzStarts(ts).map((t0, j) => {
    const n = LZ.testSteps;
    const p = j === 0 ? pred.subarray(0, n * 3) : run(t0, n);
    return validSteps(truthAt(t0, n), p, n, 3, LZ.thr) / LZ_STEPS_PER_LT;
  });
  const validLT = valids.reduce((a, b) => a + b, 0) / valids.length;
  return { pred, truth: truthAt(ts, LZ.steps), ts, valids, validLT, ...info };
}

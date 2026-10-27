// Run the page's JavaScript math in Node and compare it with the numpy results that
// build_assets.py stored in assets/checks.json.
//
//     node explainer/tools/check.mjs

import { readFileSync } from "node:fs";
import { mackeyGlass, lorenz } from "../js/data.js";
import { Reservoir, NGRC, spectralRadius, FFN } from "../js/models.js";
import { ridge, applyReadout, mseGrad } from "../js/nn.js";

const checks = JSON.parse(readFileSync(new URL("../assets/checks.json", import.meta.url), "utf8"));
let worst = 0;
let failed = 0;

function compare(label, got, want, tol = 1e-6, relative = false) {
  const g = Array.from(got).flat(Infinity);
  const w = [want].flat(Infinity);
  if (g.length !== w.length) throw new Error(`${label}: length ${g.length} vs ${w.length}`);
  const diff = Math.max(...g.map((v, i) => Math.abs(v - w[i]) / (relative ? Math.abs(w[i]) : 1)));
  if (!relative) worst = Math.max(worst, diff);
  const ok = diff <= tol;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label.padEnd(44)} max ${relative ? "rel. " : ""}|Δ| = ${diff.toExponential(2)}`);
}

// integrators
{
  const c = checks.mackeyGlass;
  const mg = mackeyGlass(c.n);
  compare("Mackey–Glass (RK4 with delay)", c.at.map((t) => mg[t]), c.values, 1e-10);
  const l = checks.lorenz;
  const lz = lorenz(l.n);
  compare("Lorenz-63 (RK4)", l.at.map((t) => Array.from(lz.subarray(3 * t, 3 * t + 3))), l.rows, 1e-9);
}

// echo state network with numpy's matrices
{
  const c = checks.esn;
  const res = new Reservoir({ N: c.N, degree: c.deg });
  res.idx = Int32Array.from(c.idx);
  res.w = Float64Array.from(c.w);
  res.win = Float64Array.from(c.win);
  res.b = Float64Array.from(c.b);
  res.leak = c.leak;
  const T = c.u.length;
  const U = Float64Array.from(c.u);
  const X = res.run(U, T);
  compare("ESN states x(t)", c.statesAt.map((t) => Array.from(X.subarray(t * c.N, (t + 1) * c.N))), c.states, 1e-12);
  const { F, p } = res.features(U, X, T);
  const Y = Float64Array.from({ length: T }, (_, t) => c.u[(t + 1) % T]);
  const B = ridge(F, Y, p, 1, 50, 250, 1e-6);
  compare("ESN ridge readout", B, c.readout, 1e-6);
  compare("ESN predictions", applyReadout(F, B, p, 1, 250, 299), c.pred, 1e-8);
}

// NG-RC features and readout on standardized Lorenz data
{
  const c = checks.ngrc;
  const lz = lorenz(checks.lorenz.n);
  const mu = [0, 0, 0];
  const sd = [0, 0, 0];
  for (let t = 0; t < 300; t++) for (let j = 0; j < 3; j++) mu[j] += lz[3 * t + j] / 300;
  for (let t = 0; t < 300; t++) for (let j = 0; j < 3; j++) sd[j] += (lz[3 * t + j] - mu[j]) ** 2 / 300;
  const Z = lz.map((v, i) => (v - mu[i % 3]) / Math.sqrt(sd[i % 3]));
  const ng = new NGRC({ d: 3, k: c.k, s: c.s, degrees: [2] });
  const F = ng.features(Z, 1, 300);
  if (ng.p !== c.p) throw new Error(`NG-RC feature count ${ng.p} vs ${c.p}`);
  compare("NG-RC feature vectors", c.featuresAt.map((r) => Array.from(F.subarray(r * ng.p, (r + 1) * ng.p))), c.features, 1e-12);
  const Y = new Float64Array(299 * 3);
  for (let r = 0; r < 299; r++) for (let j = 0; j < 3; j++) Y[3 * r + j] = Z[3 * (r + 2) + j] - Z[3 * (r + 1) + j];
  compare("NG-RC ridge readout", ridge(F, Y, ng.p, 3, 0, 299, c.lambda), c.readout, 1e-5);
}

// ridge with several outputs
{
  const c = checks.ridge;
  compare("ridge, 3 outputs (Cholesky vs numpy)", ridge(Float64Array.from(c.F), Float64Array.from(c.Y), 9, 3, 0, 120, c.lambda), c.B, 1e-10);
}

// spectral radius: power-iteration estimate vs numpy's eigenvalues
{
  const c = checks.spectral;
  const est = spectralRadius(Int32Array.from(c.idx), Float64Array.from(c.w), c.N, c.deg);
  compare("spectral radius estimate (300 × 300)", [est], [c.rho], 0.01, true);
}

// backprop of the feed-forward network, against central finite differences
{
  const ffn = new FFN(4, [5, 3], 2, 1);
  const net = ffn.net;
  const X = Float32Array.from({ length: 6 * 4 }, (_, i) => Math.sin(i * 1.7));
  const Yt = Float32Array.from({ length: 6 * 2 }, (_, i) => Math.cos(i * 0.9));
  const loss = () => mseGrad(net.forward(X, 6), Yt).loss;
  net.zeroGrad();
  net.backward(mseGrad(net.forward(X, 6), Yt).grad);
  const analytic = [];
  const numeric = [];
  for (const layer of net.layers) {
    for (const [p, g] of [[layer.W, layer.gW], [layer.b, layer.gb]]) {
      for (let i = 0; i < p.length; i++) {
        const keep = p[i];
        p[i] = keep + 1e-3;
        const up = loss();
        p[i] = keep - 1e-3;
        const down = loss();
        p[i] = keep;
        analytic.push(g[i]);
        numeric.push((up - down) / 2e-3);
      }
    }
  }
  compare("FFN backprop vs finite differences", analytic, numeric, 2e-3);
}

console.log(`\n${failed ? `${failed} check(s) FAILED` : "all checks passed"} · largest absolute difference ${worst.toExponential(2)}`);
process.exit(failed ? 1 : 0);

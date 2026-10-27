// 3.2 — inside a reservoir: 100 neurons driven by Mackey–Glass, two runs from different starting states.
// 3.3 — memory: how well a linear readout of the state recalls the input k steps back, against a window of L values.

import { h, fmt, ctl, segToggle, slider, widthOf, rng, showTip, hideTip, perFrame, divLegend } from "./util.js";
import { lineChart, heatmap, legendRow, log10, powFmt } from "./charts.js";
import { mackey, MG } from "./core.js";
import { whiteNoise } from "./data.js";
import { Reservoir } from "./models.js";
import { gram, solveRidge, applyReadout } from "./nn.js";
import { T } from "./i18n.js";

const fmt2 = (v) => fmt(v, 2);

// ───────────── 3.2 ─────────────
const N_IN = 100;
const T_IN = 300;
const TRACES = [0, 1, 2];

export function initInside(root) {
  const st = { rho: 0.9, leak: 0.5, inScale: 0.5, seed: 1, start: 1 };
  const u = mackey().z.subarray(MG.valEnd, MG.valEnd + T_IN);
  let res = new Reservoir({ N: N_IN, seed: st.seed });
  let out = null;

  const recompute = perFrame(() => {
    compute();
    draw();
  });
  const newRes = h("button", { class: "btn", type: "button" }, T.newReservoir);
  const newStart = h("button", { class: "btn", type: "button" }, T.newStarts);
  newRes.addEventListener("click", () => {
    st.seed++;
    res = new Reservoir({ N: N_IN, seed: st.seed });
    recompute();
  });
  newStart.addEventListener("click", () => {
    st.start++;
    recompute();
  });
  const inBox = h("div");
  const heatBox = h("div");
  const traceBox = h("div", { class: "minis-3" });
  const distBox = h("div");
  const verdict = h("div", { class: "callout" });

  root.replaceChildren(
    h("div", { class: "controls" },
      slider(T.rhoLabel, { min: 0, max: 2, step: 0.05, value: st.rho, format: fmt2, onInput: (v) => ((st.rho = v), recompute()) }),
      slider(T.leakLabel, { min: 0.05, max: 1, step: 0.05, value: st.leak, format: fmt2, onInput: (v) => ((st.leak = v), recompute()) }),
      slider(T.inputGain, { min: 0.05, max: 2, step: 0.05, value: st.inScale, format: fmt2, onInput: (v) => ((st.inScale = v), recompute()) }),
      newRes, newStart),
    h("p", { class: "panel-title" }, T.inputSignal),
    inBox,
    h("p", { class: "panel-title" }, T.statesTitle(N_IN)),
    heatBox,
    h("div", { class: "viz-row" },
      h("div", { class: "grow" }, h("p", { class: "panel-title" }, T.tracesTitle), legendRow([["s1", T.runA], ["s2", T.runB, "dash"]]), traceBox),
      h("div", { class: "grow" }, h("p", { class: "panel-title" }, T.distanceTitle), distBox)),
    verdict,
  );

  function compute() {
    res.configure({ rho: st.rho, leak: st.leak, inScale: st.inScale, bias: 0.2 });
    const r = rng(1000 + st.start);
    const xa = Float64Array.from({ length: N_IN }, () => 2 * r.uniform() - 1);
    const xb = Float64Array.from({ length: N_IN }, () => 2 * r.uniform() - 1);
    const XA = res.run(u, T_IN, xa);
    const XB = res.run(u, T_IN, xb);
    const dist = new Float64Array(T_IN);
    for (let t = 0; t < T_IN; t++) {
      let d = 0;
      for (let i = 0; i < N_IN; i++) d += (XA[t * N_IN + i] - XB[t * N_IN + i]) ** 2;
      dist[t] = Math.sqrt(d / N_IN);
    }
    let agree = -1;
    for (let t = 0; t < T_IN; t++) if (dist[t] < 1e-6) {
      agree = t + 1;
      break;
    }
    out = { XA, XB, dist, agree };
  }

  function draw() {
    const W = widthOf(root);
    const xs = Array.from({ length: T_IN }, (_, t) => t + 1);
    inBox.replaceChildren(lineChart({ width: W, height: 90, series: [{ xs, ys: Array.from(u), cls: "ink", label: "u(t)" }], yFmt: (v) => fmt(v, 0), yTicks: [-2, 0, 2], tipFmt: fmt2, margin: { l: 42, r: 12, t: 6, b: 20 } }));

    const M = new Float64Array(N_IN * T_IN);
    for (let i = 0; i < N_IN; i++) for (let t = 0; t < T_IN; t++) M[i * T_IN + t] = out.XA[t * N_IN + i];
    heatBox.replaceChildren(
      h("div", { class: "heat-wrap" },
        h("div", { class: "heat-y" }, h("span", null, "1"), h("span", null, String(N_IN))),
        heatmap(M, N_IN, T_IN, 1, {
          height: 170,
          onHover: (i, t, ev) => showTip(`${T.neuron} ${i + 1}, t = ${t + 1}<br><span class="mono">x = ${fmt(M[i * T_IN + t], 3)}</span>`, ev),
          onLeave: hideTip,
        }),
        h("div", { class: "heat-x" }, h("span", null, "t = 1"), h("span", null, `t = ${T_IN}`))),
      h("div", { class: "sub" }, divLegend(1, 0), " ", T.stateLegend),
    );

    const wide = W > 700;
    const half = wide ? Math.floor((W - 20) / 2) : W;
    traceBox.replaceChildren(...TRACES.map((i) => lineChart({
      width: half, height: 70, y0: -1, y1: 1, yTicks: [-1, 0, 1], yFmt: (v) => fmt(v, 0), hover: false,
      series: [
        { xs, ys: xs.map((t) => out.XA[(t - 1) * N_IN + i]), cls: "s1", thin: true },
        { xs, ys: xs.map((t) => out.XB[(t - 1) * N_IN + i]), cls: "s2", thin: true, dash: true },
      ],
      margin: { l: 30, r: 8, t: 4, b: 18 },
    })));

    distBox.replaceChildren(lineChart({
      width: half, height: 230,
      series: [{ xs, ys: Array.from(out.dist, log10), cls: "ink", label: T.rmsDifference }],
      hlines: [{ y: -6, label: "10⁻⁶" }],
      y0: -16, y1: 1, yTicks: [-15, -12, -9, -6, -3, 0], yFmt: powFmt, xLabel: T.timeStep,
      tipFmt: (v) => (10 ** v).toExponential(1), tipTitle: (x) => `t = ${x}`,
    }));

    verdict.replaceChildren(h("p", null, out.agree > 0 ? T.echoYes(out.agree) : T.echoNo(out.dist[T_IN - 1])));
  }

  compute();
  draw();
  return { redraw: draw };
}

// ───────────── 3.3 ─────────────
const T_MEM = 2200;
const K = 60;
const MEM_TRAIN = [100, 1600];
const MEM_TEST = [1600, T_MEM];

export function initMemory(root) {
  const st = { N: 100, rho: 0.9, leak: 1, inScale: 0.1, L: 20 };
  const u = whiteNoise(T_MEM, rng(5));
  const cache = new Map();
  const reservoir = (N) => {
    if (!cache.has(N)) cache.set(N, new Reservoir({ N, seed: 3 }));
    return cache.get(N);
  };
  // target row t: [u(t−1), …, u(t−K)] (zero before the start; those rows are outside both ranges)
  const Y = new Float64Array(T_MEM * K);
  for (let t = 0; t < T_MEM; t++) for (let k = 1; k <= K; k++) Y[t * K + k - 1] = t - k >= 0 ? u[t - k] : 0;
  let r2 = null;
  const recompute = perFrame(() => {
    compute();
    draw();
  });

  const chartBox = h("div");
  const stats = h("div", { class: "stats" });
  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.neuronsN, segToggle([50, 100, 200, 300].map((v) => ({ value: v, label: String(v) })), st.N, (v) => ((st.N = v), recompute()), T.neuronsN)),
      slider(T.rhoLabel, { min: 0, max: 1.5, step: 0.05, value: st.rho, format: fmt2, onInput: (v) => ((st.rho = v), recompute()) }),
      slider(T.leakLabel, { min: 0.05, max: 1, step: 0.05, value: st.leak, format: fmt2, onInput: (v) => ((st.leak = v), recompute()) }),
      slider(T.inputGain, { min: 0.05, max: 2, step: 0.05, value: st.inScale, format: fmt2, onInput: (v) => ((st.inScale = v), recompute()) }),
      slider(T.windowL, { min: 1, max: K, step: 1, value: st.L, onInput: (v) => ((st.L = v), draw()) })),
    legendRow([["s1", T.reservoirReadout], ["s2", T.windowOf]]),
    chartBox,
    stats,
  );

  function compute() {
    const res = reservoir(st.N).configure({ rho: st.rho, leak: st.leak, inScale: st.inScale, bias: 0 });
    const X = res.run(u, T_MEM);
    const { F, p } = res.features(u, X, T_MEM);
    const B = solveRidge(gram(F, Y, p, K, ...MEM_TRAIN), 1e-7);
    const P = applyReadout(F, B, p, K, ...MEM_TEST);
    const n = MEM_TEST[1] - MEM_TEST[0];
    r2 = Float64Array.from({ length: K }, (_, k) => {
      let sy = 0, sp = 0, syy = 0, spp = 0, syp = 0;
      for (let r = 0; r < n; r++) {
        const y = Y[(MEM_TEST[0] + r) * K + k];
        const q = P[r * K + k];
        sy += y; sp += q; syy += y * y; spp += q * q; syp += y * q;
      }
      const cov = syp / n - (sy / n) * (sp / n);
      const vy = syy / n - (sy / n) ** 2;
      const vp = spp / n - (sp / n) ** 2;
      return vp > 0 ? (cov * cov) / (vy * vp) : 0;
    });
  }

  function draw() {
    const W = widthOf(root);
    const ks = Array.from({ length: K }, (_, i) => i + 1);
    const win = ks.map((k) => (k <= st.L - 1 ? 1 : 0));
    chartBox.replaceChildren(lineChart({
      width: W, height: 240, y0: 0, y1: 1.05, yTicks: [0, 0.25, 0.5, 0.75, 1], yFmt: fmt2, x0: 1, x1: K,
      series: [
        { xs: ks, ys: Array.from(r2), cls: "s1", label: T.reservoirReadout },
        { xs: ks, ys: win, cls: "s2", label: T.windowOf, dash: true },
      ],
      xLabel: T.delayK, tipFmt: (v) => fmt(v, 3), tipTitle: (k) => `${T.recall} u(t−${k})`,
    }));
    const mc = r2.reduce((a, b) => a + b, 0);
    let half = 0;
    while (half < K && r2[half] >= 0.5) half++;
    stats.replaceChildren(
      h("span", null, h("b", null, fmt(mc, 1)), T.memoryCapacity(st.N)),
      h("span", null, h("b", null, String(half)), T.halfRecall),
      h("span", null, h("b", null, String(Math.max(0, st.L - 1))), T.windowMemory(st.L)),
    );
  }

  compute();
  draw();
  return { redraw: draw };
}

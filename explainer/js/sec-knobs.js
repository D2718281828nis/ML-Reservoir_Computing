// 1.3 — the two numbers of the ESN equation, one at a time.
//  α: the leak, isolated. A given drive d(t) stands in for tanh(W_in u + W x + b), and x(t) = (1−α) x(t−1) + α d(t).
//  ρ: the eigenvalues of one random W (100 neurons), computed once and rescaled with the slider, and how a disturbance
//     carried by the links alone, v ← W v, grows or fades.

import { h, s, fmt, slider, widthOf, rng, showTip, hideTip, perFrame } from "./util.js";
import { lineChart, legendRow, log10, powFmt } from "./charts.js";
import { Reservoir, eigenvalues, denseW } from "./models.js";
import { T } from "./i18n.js";

const fmt2 = (v) => fmt(v, 2);
const T_A = 150;
const K = 60;

export function initKnobs(root) {
  const st = { leak: 0.3, rho: 0.9 };

  // the drive: a level that jumps every 30 steps, plus noise
  const r = rng(7);
  const levels = [-0.6, 0.7, 0.1, -0.5, 0.8];
  const drive = Float64Array.from({ length: T_A }, (_, t) => levels[Math.floor(t / 30)] + 0.25 * r.normal());

  // one random W, its eigenvalues before scaling, and a fixed random starting disturbance
  const res = new Reservoir({ N: 100, degree: 10, seed: 1 });
  const { N, deg, idx, w0, rho0 } = res;
  const ev0 = eigenvalues(denseW(idx, w0, N, deg));
  const v0 = Float64Array.from({ length: N }, () => r.normal());

  const redraw = perFrame(() => draw());
  const aBox = h("div");
  const aNote = h("p", { class: "sub" });
  const planeBox = h("div", { class: "eig-plane" });
  const decayBox = h("div", { class: "grow" });
  const rNote = h("p", { class: "sub" });
  root.replaceChildren(
    h("div", { class: "viz-row" },
      h("div", { class: "grow" },
        h("p", { class: "panel-title", html: T.knobLeakTitle }),
        h("div", { class: "controls" }, slider(T.leakLabel, { min: 0.02, max: 1, step: 0.02, value: st.leak, format: fmt2, onInput: (v) => ((st.leak = v), redraw()) })),
        legendRow([["muted", T.knobDrive], ["s1", T.knobState]]),
        aBox, aNote),
      h("div", { class: "grow" },
        h("p", { class: "panel-title", html: T.knobRhoTitle }),
        h("div", { class: "controls" }, slider(T.rhoLabel, { min: 0.1, max: 1.5, step: 0.05, value: st.rho, format: fmt2, onInput: (v) => ((st.rho = v), redraw()) })),
        h("div", { class: "eig-row" }, planeBox, decayBox),
        rNote)),
  );

  function draw() {
    drawLeak();
    drawRho();
  }

  function drawLeak() {
    const a = st.leak;
    const x = new Float64Array(T_A);
    let prev = 0;
    for (let t = 0; t < T_A; t++) prev = x[t] = (1 - a) * prev + a * drive[t];
    const ts = Array.from({ length: T_A }, (_, t) => t);
    const W = Math.max(280, widthOf(aBox.parentElement, 420));
    aBox.replaceChildren(lineChart({
      width: W, height: 220, x0: 0, x1: T_A - 1, y0: -1.4, y1: 1.4, yTicks: [-1, 0, 1], xLabel: T.knobStep,
      series: [
        { xs: ts, ys: Array.from(drive), cls: "muted", thin: true, label: T.knobDrive },
        { xs: ts, ys: Array.from(x), cls: "s1", label: T.knobState },
      ],
      tipTitle: (t) => `t = ${t}`,
    }));
    const tau = a >= 1 ? 0 : -1 / Math.log(1 - a);
    aNote.innerHTML = T.knobLeakNote(fmt2(a), fmt2(1 - a), a >= 1 ? "0" : fmt(tau, tau < 10 ? 1 : 0), fmt((1 - a) ** 10, 3));
  }

  function drawRho() {
    const f = st.rho / rho0;
    const ev = ev0.map((e) => ({ re: e.re * f, im: e.im * f }));
    const rho = Math.max(...ev.map((e) => Math.hypot(e.re, e.im)));

    // the complex plane: the unit circle, the circle of radius ρ, and the 100 eigenvalues
    const S = 240;
    const lim = Math.max(1.25, rho * 1.12);
    const P = (v) => S / 2 + (v / lim) * (S / 2 - 10);
    const Q = (v) => S / 2 - (v / lim) * (S / 2 - 10);
    const svg = s("svg", { class: "eigviz", viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": T.knobPlaneAria });
    svg.append(
      s("line", { class: "axis", x1: 4, x2: S - 4, y1: S / 2, y2: S / 2 }),
      s("line", { class: "axis", x1: S / 2, x2: S / 2, y1: 4, y2: S - 4 }),
      s("circle", { class: "unit", cx: S / 2, cy: S / 2, r: P(1) - S / 2 }),
      s("circle", { class: "rho", cx: S / 2, cy: S / 2, r: P(rho) - S / 2 }),
      s("text", { class: "lbl", x: P(1) + 3, y: S / 2 - 4 }, "1"),
      s("text", { class: "lbl", x: S - 6, y: S / 2 + 14, "text-anchor": "end" }, "Re λ"),
      s("text", { class: "lbl", x: S / 2 + 5, y: 14 }, "Im λ"),
    );
    for (const e of ev) {
      const c = s("circle", { class: "ev", cx: P(e.re), cy: Q(e.im), r: 3.2 });
      c.addEventListener("pointermove", (ev_) => showTip(`<span class="mono">λ = ${fmt(e.re, 3)} ${e.im < 0 ? "−" : "+"} ${fmt(Math.abs(e.im), 3)}i<br>|λ| = ${fmt(Math.hypot(e.re, e.im), 3)}</span>`, ev_));
      c.addEventListener("pointerleave", hideTip);
      svg.append(c);
    }
    planeBox.replaceChildren(svg, h("p", { class: "sub" }, T.knobPlaneCap));

    // a disturbance carried by the links alone: ‖Wᵏv‖ / ‖v‖ against ρᵏ
    const w = w0.map((v) => v * f);
    let v = Float64Array.from(v0);
    let u = new Float64Array(N);
    const n0 = Math.hypot(...v);
    const norms = [0];
    for (let k = 1; k <= K; k++) {
      for (let i = 0; i < N; i++) {
        let acc = 0;
        for (let q = i * deg, e = q + deg; q < e; q++) acc += w[q] * v[idx[q]];
        u[i] = acc;
      }
      [v, u] = [u, v];
      norms.push(log10(Math.hypot(...v) / n0));
    }
    const ks = Array.from({ length: K + 1 }, (_, k) => k);
    const ref = ks.map((k) => k * Math.log10(rho));
    const all = [...norms, ...ref];
    const y0 = Math.max(-12, Math.floor(Math.min(...all)));
    const y1 = Math.max(1, Math.ceil(Math.max(...all)));
    const ticks = [];
    const stepT = Math.max(1, Math.ceil((y1 - y0) / 5));
    for (let e = y1; e >= y0; e -= stepT) ticks.push(e);
    const W = Math.max(220, widthOf(decayBox, 260));
    decayBox.replaceChildren(
      legendRow([["s1", T.knobNorm], ["ink", T.knobRhoK, "dash"]]),
      lineChart({
        width: W, height: 210, x0: 0, x1: K, y0, y1, yTicks: ticks, yFmt: powFmt, xLabel: T.knobSteps,
        series: [
          { xs: ks, ys: norms, cls: "s1", label: T.knobNorm },
          { xs: ks, ys: ref, cls: "ink", dash: true, thin: true, label: T.knobRhoK },
        ],
        tipFmt: (v) => (10 ** v).toExponential(1), tipTitle: (k) => `k = ${k}`,
      }));
    const g = rho ** 20;
    rNote.innerHTML = T.knobRhoNote(fmt2(rho0), fmt2(f), fmt2(rho), g < 0.01 || g >= 1000 ? g.toExponential(1) : fmt(g, 3));
  }

  draw();
  return { redraw: draw };
}

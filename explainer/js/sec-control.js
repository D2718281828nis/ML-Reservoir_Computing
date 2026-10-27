// 8.1 — a plant in state-space form: its (A, B, C, D) drawn as a signal-flow graph, its poles, and the data it makes.
// 8.2 — a reservoir trained on that data (one ridge solve), scored on a held-out test part.
// 8.3 — the trained reservoir read back as (A, B, C, D): the same kind of graph, its eigenvalues against the plant's poles,
//        the two impulse responses, and the Hankel singular values that reveal how many states the map really needs.
// One plant and one reservoir are shared by all three steps through a small publish/subscribe api.

import { h, s, fmt, ctl, segToggle, slider, widthOf, visibility, diverging, rgbCss, showTip, hideTip, perFrame } from "./util.js";
import { lineChart, legendRow, log10, powFmt } from "./charts.js";
import { graphLayout } from "./core.js";
import { eigenvalues } from "./models.js";
import { CTRL, plant, impulse, staircase, identify, hankelSV, realizePoles, residues, impulseStates } from "./control.js";
import { T } from "./i18n.js";

const fmt2 = (v) => fmt(v, 2);
const SHOW = 400; // steps of input and output shown in 8.1
const KG = 40; // steps of the impulse played through the graph

export function initControl(plantRoot, identRoot, abcdRoot) {
  const st = { r: 0.9, theta: 0.3, d: 0, rho: 0.99, leak: 0.5, inScale: 0.01, seed: 1 };
  const u = staircase(CTRL.n);
  let sys = null;
  let id = null;
  let fitMs = 0;

  const refit = perFrame(() => {
    compute();
    drawAll();
  });

  // ── 8.1 the plant ──
  const plantGraph = h("div", { class: "ctl-graph" });
  const plantPlane = h("div", { class: "eig-plane" });
  const plantMats = h("div");
  const plantSeries = h("div");
  plantRoot.replaceChildren(
    h("div", { class: "controls" },
      slider(T.ctlR, { min: 0.5, max: 0.97, step: 0.01, value: st.r, format: fmt2, onInput: (v) => ((st.r = v), refit()) }),
      slider(T.ctlTheta, { min: 0.05, max: 1.2, step: 0.05, value: st.theta, format: fmt2, onInput: (v) => ((st.theta = v), refit()) }),
      slider("D", { min: 0, max: 0.5, step: 0.05, value: st.d, format: fmt2, onInput: (v) => ((st.d = v), refit()) })),
    plantMats,
    h("div", { class: "viz-row" },
      h("div", { class: "grow" }, h("p", { class: "panel-title" }, T.ctlPlantGraph), plantGraph, h("p", { class: "sub", html: T.ctlPlantGraphNote })),
      h("div", { class: "ctl-side" }, h("p", { class: "panel-title" }, T.ctlPoles), plantPlane)),
    h("p", { class: "panel-title" }, T.ctlSeriesTitle),
    legendRow([["muted", "u(t)"], ["ink", "y(t)"]]),
    plantSeries,
  );

  // ── 8.2 the reservoir, trained ──
  const newRes = h("button", { class: "btn", type: "button" }, T.newReservoir);
  newRes.addEventListener("click", () => ((st.seed += 1), refit()));
  const fitChart = h("div");
  const fitNote = h("p", { class: "sub" });
  identRoot.replaceChildren(
    h("div", { class: "controls" },
      slider(T.rhoLabel, { min: 0.5, max: 1.2, step: 0.01, value: st.rho, format: fmt2, onInput: (v) => ((st.rho = v), refit()) }),
      slider(T.leakLabel, { min: 0.1, max: 1, step: 0.05, value: st.leak, format: fmt2, onInput: (v) => ((st.leak = v), refit()) }),
      slider(T.inputGain, { min: 0.01, max: 1, step: 0.01, value: st.inScale, format: fmt2, onInput: (v) => ((st.inScale = v), refit()) }),
      newRes),
    legendRow([["ink", T.ctlPlantOut], ["s1", T.ctlResOut, "dash"]]),
    fitChart,
    fitNote,
  );

  // ── 8.3 the reservoir as (A, B, C, D) ──
  // g: what the graph shows. "weights": every edge colored by its entry of A, B, C, D. "impulse": a unit impulse u(0) = 1
  // played through the graph, x(k) = Aᵏ B, every edge colored by what it carries at step k. focus: hovered or clicked neuron.
  const g = { mode: "weights", k: 0, hover: null, sel: null, playing: false };
  const isVisible = visibility(abcdRoot);
  let graph = null;
  const resGraph = h("div", { class: "ctl-graph" });
  const graphInfo = h("div", { class: "ctl-info" });
  const resPlane = h("div", { class: "eig-plane" });
  const eraNote = h("p", { class: "sub" });
  const resMats = h("div");
  const impBox = h("div", { class: "grow" });
  const hankBox = h("div", { class: "grow" });
  const abcdNote = h("p", { class: "sub" });
  const kCtl = slider(T.ctlKLabel, { min: 0, max: KG, step: 1, value: g.k, onInput: (v) => ((g.k = v), paint(), drawImpulse()), wide: true });
  const playBtn = h("button", { class: "btn", type: "button" }, T.play);
  playBtn.addEventListener("click", () => (g.playing ? stopPlay() : startPlay()));
  const impulseCtl = h("div", { class: "controls", style: { display: "none" } }, kCtl, playBtn);
  abcdRoot.replaceChildren(
    resMats,
    h("div", { class: "controls" }, ctl(T.ctlShow, segToggle([["weights", T.ctlModeWeights], ["impulse", T.ctlModeImpulse]].map(([value, label]) => ({ value, label })), g.mode, (v) => {
      g.mode = v;
      impulseCtl.style.display = v === "impulse" ? "" : "none";
      if (v !== "impulse") stopPlay();
      paint();
      drawImpulse();
    }, T.ctlShow))),
    impulseCtl,
    h("div", { class: "viz-row" },
      h("div", { class: "grow" }, h("p", { class: "panel-title" }, T.ctlResGraph(CTRL.N)), resGraph, graphInfo, h("p", { class: "sub", html: T.ctlResGraphNote })),
      h("div", { class: "ctl-side" }, h("p", { class: "panel-title" }, T.ctlEigs), resPlane, eraNote)),
    h("div", { class: "viz-row" },
      h("div", { class: "grow" }, h("p", { class: "panel-title" }, T.ctlImpulse), legendRow([["ink", T.ctlPlantShort], ["s1", T.ctlResShort, "dash"]]), impBox),
      h("div", { class: "grow" }, h("p", { class: "panel-title" }, T.ctlHankel), legendRow([["ink", T.ctlPlantShort], ["s1", T.ctlResShort, "dash"]]), hankBox)),
    abcdNote,
  );

  function compute() {
    sys = plant({ r: st.r, theta: st.theta, d: st.d });
    const t0 = performance.now();
    id = identify(sys, { rho: st.rho, leak: st.leak, inScale: st.inScale, seed: st.seed }, u);
    fitMs = performance.now() - t0;
    id.layout = graphLayout(id.res);
    id.eigs = eigenvalues(id.model.A.map((row) => Array.from(row)));
    id.hP = impulse(sys, CTRL.K);
    id.hR = impulse(id.model, CTRL.K);
    id.svP = hankelSV(impulse(sys, 2 * 30), 30, eigenvalues);
    id.svR = hankelSV(impulse(id.model, 2 * 30), 30, eigenvalues);
    // poles of the order-2 model realized from each impulse response (Ho–Kalman / ERA), and the reservoir's modal residues
    id.eraR = realizePoles(impulse(id.model, 70), 2, 30, eigenvalues);
    id.rho = residues(id.model, id.eigs);
    id.X = impulseStates(id.model, KG);
  }

  function drawAll() {
    drawPlant();
    drawFit();
    drawABCD();
  }

  // ── drawing ──
  function drawPlant() {
    const { A, B, C, D } = sys;
    tex(plantMats, `A = ${mat(A)},\\quad B = ${col(B)},\\quad C = ${row(C)},\\quad D = ${num(D)}`,
      `A = [[${A.map((r) => r.map((v) => fmt(v, 3)).join(", ")).join("], [")}]], B = [${B.join(", ")}], C = [${C.map((v) => fmt(v, 3)).join(", ")}], D = ${fmt(D, 2)}`);
    // signal-flow graph: an edge j → i carries A_ij · x_j(t−1) into x_i(t)
    const nodes = [
      { id: "u", x: 40, y: 150, label: "u(t)", cls: "io-in" },
      { id: "x2", x: 175, y: 150, label: "x₂", cls: "state" },
      { id: "x1", x: 310, y: 150, label: "x₁", cls: "state" },
      { id: "y", x: 440, y: 150, label: "y(t)", cls: "io-out" },
    ];
    const lab = (k, v) => `${k} = ${fmt(v, 3)}`;
    const edges = [
      { from: "u", to: "x2", w: B[1], label: lab("B₂", B[1]) },
      { from: "x2", to: "x1", w: A[0][1], curve: -34, label: lab("A₁₂", A[0][1]) },
      { from: "x1", to: "x2", w: A[1][0], curve: -34, label: lab("A₂₁", A[1][0]) },
      { from: "x2", to: "x2", w: A[1][1], label: lab("A₂₂", A[1][1]) },
      { from: "x1", to: "y", w: C[0], label: lab("C₁", C[0]) },
      { from: "x2", to: "y", w: C[1], curve: 70, label: lab("C₂", C[1]), labelAt: 0.75 },
      { from: "u", to: "y", w: D, curve: -150, label: lab("D", D), labelAt: 0.62, dashed: D === 0 },
    ];
    plantGraph.replaceChildren(flowGraph({ width: 480, height: 260, nodes, edges, scale: 1.5, aria: T.ctlPlantGraph }));
    plantPlane.replaceChildren(zPlane({ crosses: sys.poles, dots: [], aria: T.ctlPoles }));
    const W = widthOf(plantRoot);
    const ts = Array.from({ length: SHOW }, (_, t) => t);
    plantSeries.replaceChildren(lineChart({
      width: W, height: 190, x0: 0, x1: SHOW - 1, xLabel: T.timeStep,
      series: [
        { xs: ts, ys: Array.from(u.subarray(0, SHOW)), cls: "muted", thin: true, label: "u(t)" },
        { xs: ts, ys: Array.from(id.y.subarray(0, SHOW)), cls: "ink", label: "y(t)" },
      ],
      tipFmt: (v) => fmt(v, 3), tipTitle: (t) => `t = ${t}`,
    }));
  }

  function drawFit() {
    const a = CTRL.valEnd;
    const b = Math.min(CTRL.n, a + 400);
    const ts = Array.from({ length: b - a }, (_, i) => a + i);
    fitChart.replaceChildren(lineChart({
      width: widthOf(identRoot), height: 210, x0: a, x1: b - 1, xLabel: T.timeStep,
      series: [
        { xs: ts, ys: ts.map((t) => id.y[t]), cls: "ink", label: T.ctlPlantOut },
        { xs: ts, ys: ts.map((t) => id.P[t]), cls: "s1", dash: true, label: T.ctlResOut },
      ],
      tipFmt: (v) => fmt(v, 4), tipTitle: (t) => `t = ${t}`,
    }));
    fitNote.innerHTML = T.ctlFitNote(CTRL, id.err < 1e-3 ? id.err.toExponential(1) : fmt(id.err, 4), id.lambda.toExponential(0), fitMs.toFixed(0));
  }

  function drawABCD() {
    const m = id.model;
    const N = m.n;
    const nnz = m.A.reduce((acc, r) => acc + r.reduce((c, v) => c + (v !== 0), 0), 0);
    tex(resMats, `A = (1-\\alpha) I + \\alpha W \\in \\mathbb{R}^{${N}\\times${N}},\\quad B = \\alpha\\, W_{\\text{in}},\\quad C = W_{\\text{out}}^{(x)},\\quad D = W_{\\text{out}}^{(u)} = ${num(m.D)}`,
      `A = (1−α)I + αW (${N}×${N}), B = α·W_in, C = W_out(x), D = W_out(u) = ${fmt(m.D, 3)}`);

    graph = buildResGraph();
    paint();

    // the z-plane: plant poles, reservoir eigenvalues sized by their residue, poles of the order-2 model realized from the reservoir
    const rmax = Math.max(1e-300, ...id.rho.map((r) => Math.hypot(r.re, r.im)));
    const dots = id.eigs.map((e, i) => ({ ...e, r: 2 + 3.5 * Math.sqrt(Math.hypot(id.rho[i].re, id.rho[i].im) / rmax), extra: `ρ = ${cfmt(id.rho[i], 3)}` }));
    resPlane.replaceChildren(zPlane({ crosses: sys.poles, dots, rings: id.eraR, aria: T.ctlEigs }),
      legendRow([["ink", T.ctlPlantPoles], ["s1", T.ctlResEigs], ["accent", T.ctlEraPoles]]));
    const sumRho = id.rho.reduce((acc, r) => acc + Math.hypot(r.re, r.im), 0);
    const hmax = Math.max(...Array.from(id.hR, Math.abs));
    const pp = sys.poles.find((p) => p.im >= 0) ?? sys.poles[0];
    const pe = id.eraR.find((p) => p.im >= 0) ?? id.eraR[0];
    eraNote.innerHTML = T.ctlEraNote(cfmt(pp, 4), cfmt(pe, 4), fmt(sumRho, sumRho < 10 ? 2 : 0), fmt(hmax, 3));

    drawImpulse();
    const M = 10;
    const is = Array.from({ length: M }, (_, i) => i + 1);
    const lp = id.svP.slice(0, M).map(log10);
    const lr = id.svR.slice(0, M).map(log10);
    const y0 = Math.max(-17, Math.floor(Math.min(...lp, ...lr)));
    const yt = [];
    for (let e = Math.ceil(Math.max(...lp, ...lr)); e >= y0; e -= Math.max(1, Math.ceil((Math.ceil(Math.max(...lp, ...lr)) - y0) / 5))) yt.push(e);
    hankBox.replaceChildren(lineChart({
      width: Math.max(260, widthOf(hankBox, 360)), height: 200, x0: 0.5, x1: M + 0.5, y0, y1: Math.max(1, ...yt), yTicks: yt, yFmt: powFmt,
      xTicks: is, xLabel: T.ctlSvIndex, hover: false,
      series: [{ xs: is, ys: lp, cls: "ink", thin: true }, { xs: is, ys: lr, cls: "s1", thin: true, dash: true }],
      points: [...is.map((i, j) => ({ x: i, y: lp[j], cls: "ink", r: 4, tip: `σ${sub(i)} = ${id.svP[j].toExponential(2)}` })),
        ...is.map((i, j) => ({ x: i, y: lr[j], cls: "s1", r: 4, tip: `σ${sub(i)} = ${id.svR[j].toExponential(2)}` }))],
    }));
    const rel = relErr(id.hR, id.hP);
    abcdNote.innerHTML = T.ctlAbcdNote(N, nnz, fmt(rel, 3), id.svR[1] ? (id.svR[2] / id.svR[1]).toExponential(0) : "–", fmt(m.D, 2), fmt(id.hR[0], 3), fmt(id.hP[0], 3));
  }

  function drawImpulse() {
    const ks = Array.from({ length: CTRL.K + 1 }, (_, k) => k);
    impBox.replaceChildren(lineChart({
      width: Math.max(260, widthOf(impBox, 360)), height: 200, x0: 0, x1: CTRL.K, xLabel: T.ctlK,
      series: [
        { xs: ks, ys: Array.from(id.hP), cls: "ink", label: T.ctlPlantShort },
        { xs: ks, ys: Array.from(id.hR), cls: "s1", dash: true, label: T.ctlResShort },
      ],
      marks: g.mode === "impulse" ? [{ x: g.k, cls: "now", label: "k" }] : [],
      points: g.mode === "impulse" ? [{ x: g.k, y: id.hR[g.k], cls: "s1", r: 5 }] : [],
      tipFmt: (v) => fmt(v, 4), tipTitle: (k) => `k = ${k}`,
    }));
  }

  function startPlay() {
    g.playing = true;
    playBtn.textContent = T.pause;
    if (g.k >= KG) g.k = 0;
    let last = 0;
    const tick = (now) => {
      if (!g.playing) return;
      if (now - last > 300 && isVisible()) {
        last = now;
        if (g.k >= KG) return stopPlay();
        g.k++;
        kCtl.set(g.k);
        paint();
        drawImpulse();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  function stopPlay() {
    g.playing = false;
    playBtn.textContent = T.play;
  }

  // The reservoir graph, built once per trained model; paint() recolors it for the mode, the step k and the focused neuron.
  // Same reading as the plant's graph: an edge j → i is the entry A_ij; fans are B and C; the arc on top is D; rings are A_ii = 1 − α.
  function buildResGraph() {
    const m = id.model;
    const N = m.n;
    const P = id.layout;
    const GW = 480;
    const GH = 380;
    const U = { x: 30, y: GH / 2 };
    const Y = { x: GW - 30, y: GH / 2 };
    const pos = Array.from({ length: N }, (_, i) => ({ x: 90 + P[2 * i] * (GW - 180), y: 34 + P[2 * i + 1] * (GH - 60) }));
    const svg = s("svg", { class: "ctlviz", viewBox: `0 0 ${GW} ${GH}`, role: "img", "aria-label": T.ctlResGraph(N) });
    const mid = `ah-${Math.random().toString(36).slice(2, 8)}`;
    svg.append(s("defs", null, s("marker", { id: mid, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 8, markerHeight: 8, markerUnits: "userSpaceOnUse", orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-head-ink" }))));
    const edges = [];
    const line = (a, b, ra, rb) => {
      const [dx, dy] = unit(b.x - a.x, b.y - a.y);
      return { x1: a.x + dx * ra, y1: a.y + dy * ra, x2: b.x - dx * rb, y2: b.y - dy * rb };
    };
    const layer = s("g");
    const add = (el, e) => {
      layer.append(el);
      edges.push({ el, ...e });
    };
    const { deg, idx } = id.res;
    for (let i = 0; i < N; i++) add(s("line", { class: "ctl-edge thin", ...line(U, pos[i], 21, 8) }), { kind: "B", i });
    for (let i = 0; i < N; i++) for (let q = i * deg; q < (i + 1) * deg; q++) {
      const j = idx[q];
      if (j !== i) add(s("line", { class: "ctl-edge thin", ...line(pos[j], pos[i], 7, 8) }), { kind: "A", i, j });
    }
    for (let i = 0; i < N; i++) add(s("line", { class: "ctl-edge thin", ...line(pos[i], Y, 7, 23) }), { kind: "C", i });
    add(s("path", { class: "ctl-edge", d: `M${U.x},${U.y - 21} Q${GW / 2},${-GH / 2 + 40} ${Y.x},${Y.y - 23}` }), { kind: "D" });
    svg.append(layer);
    const dLabel = s("text", { class: "ctl-elabel", x: GW / 2, y: 18, "text-anchor": "middle" });
    const nodes = pos.map((p, i) => {
      const ring = s("circle", { class: "ring", cx: p.x, cy: p.y, r: 9, "stroke-opacity": Math.min(1, 0.2 + (1 - st.leak)) });
      const c = s("circle", { cx: p.x, cy: p.y, r: 6.5 });
      const gEl = s("g", { class: "ctl-node state small hot" }, ring, c);
      gEl.addEventListener("pointerenter", () => ((g.hover = i), paint()));
      gEl.addEventListener("pointerleave", () => ((g.hover = null), paint()));
      gEl.addEventListener("click", () => ((g.sel = g.sel === i ? null : i), paint()));
      return { g: gEl, c };
    });
    const uVal = s("text", { class: "g-io-val", x: 4, y: U.y + 38, "text-anchor": "start" });
    const yVal = s("text", { class: "g-io-val", x: GW - 4, y: Y.y + 40, "text-anchor": "end" });
    svg.append(
      dLabel,
      ...nodes.map((n) => n.g),
      s("g", { class: "ctl-node io-in" }, s("circle", { cx: U.x, cy: U.y, r: 20 }), s("text", { x: U.x, y: U.y + 4.5, "text-anchor": "middle" }, "u(t)")), uVal,
      s("g", { class: "ctl-node io-out" }, s("circle", { cx: Y.x, cy: Y.y, r: 20 }), s("text", { x: Y.x, y: Y.y + 4.5, "text-anchor": "middle" }, "y(t)")), yVal,
    );
    svg.addEventListener("click", (ev) => {
      if (ev.target === svg) ((g.sel = null), paint());
    });
    resGraph.replaceChildren(svg);
    return { svg, edges, nodes, dLabel, uVal, yVal, mid };
  }

  // the value each edge carries: in "weights" mode the matrix entry itself, in "impulse" mode the signal at step k
  function edgeValue(e) {
    const m = id.model;
    if (g.mode === "weights") return e.kind === "A" ? m.A[e.i][e.j] : e.kind === "B" ? m.B[e.i] : e.kind === "C" ? m.C[e.i] : m.D;
    const k = g.k;
    if (e.kind === "A") return k > 0 ? m.A[e.i][e.j] * id.X[k - 1][e.j] : 0;
    if (e.kind === "B") return k === 0 ? m.B[e.i] : 0;
    if (e.kind === "C") return m.C[e.i] * id.X[k][e.i];
    return k === 0 ? m.D : 0;
  }

  function paint() {
    if (!graph) return;
    const m = id.model;
    const focus = g.hover ?? g.sel;
    const lim = {};
    const vals = graph.edges.map((e) => {
      const v = edgeValue(e);
      lim[e.kind] = Math.max(lim[e.kind] ?? 1e-300, Math.abs(v));
      return v;
    });
    graph.edges.forEach((e, n) => {
      const v = vals[n];
      const t = Math.min(1, Math.abs(v) / lim[e.kind]);
      const touches = focus != null && (e.i === focus || e.j === focus);
      e.el.setAttribute("stroke", rgbCss(diverging(v, lim[e.kind])));
      e.el.setAttribute("stroke-width", (e.kind === "D" ? 1.4 : touches ? 1.6 : 0.5) + (e.kind === "D" ? 2.6 : touches ? 2.2 : 1.6) * t);
      e.el.classList.toggle("hl", touches);
      e.el.classList.toggle("zero", v === 0);
      if (touches || e.kind === "D") e.el.setAttribute("marker-end", `url(#${graph.mid})`);
      else e.el.removeAttribute("marker-end");
    });
    graph.svg.classList.toggle("focus", focus != null);
    const imp = g.mode === "impulse";
    const xk = imp ? id.X[g.k] : null;
    const xl = imp ? Math.max(1e-300, ...Array.from(xk, Math.abs)) : 1;
    graph.nodes.forEach((n, i) => {
      n.c.style.fill = imp ? rgbCss(diverging(xk[i], xl)) : "";
      n.g.classList.toggle("on", i === focus);
      n.g.classList.toggle("sel", i === g.sel);
    });
    graph.dLabel.textContent = imp ? `D·u(${g.k}) = ${fmt(edgeValue({ kind: "D" }), 3)}` : `D = ${fmt(m.D, 3)}`;
    graph.uVal.textContent = imp ? `u(${g.k}) = ${g.k === 0 ? 1 : 0}` : "";
    graph.yVal.textContent = imp ? `h(${g.k}) = ${fmt(id.hR[g.k], 4)}` : "";
    drawInfo(focus);
  }

  // the focused neuron's own row of the state equation, with numbers
  function drawInfo(i) {
    if (i == null) {
      graphInfo.innerHTML = `<p class="sub">${g.mode === "impulse" ? T.ctlImpulseHint(g.k, fmt(id.hR[g.k], 4)) : T.ctlInfoHint}</p>`;
      return;
    }
    const m = id.model;
    const { deg, idx } = id.res;
    const n = i + 1;
    const ins = [];
    for (let q = i * deg; q < (i + 1) * deg; q++) if (idx[q] !== i) ins.push(idx[q]);
    const terms = ins.map((j) => `${sgn(m.A[i][j])} ${fmt(Math.abs(m.A[i][j]), 3)}·x${sub(j + 1)}`).join(" ");
    let html = `<p class="panel-title">${T.ctlState} x${sub(n)}${g.sel === i ? ` · ${T.ctlPinned}` : ""}</p>
      <p class="mono ctl-eq">x${sub(n)}(t) = ${fmt(m.A[i][i], 3)}·x${sub(n)}(t−1) ${terms} ${sgn(m.B[i])} ${fmt(Math.abs(m.B[i]), 4)}·u(t)</p>
      <p class="sub">${T.ctlRowNote(n, fmt(m.A[i][i], 3), ins.length, fmt(m.B[i], 4), fmt(m.C[i], 2))}</p>`;
    if (g.mode === "impulse") html += `<p class="sub">${T.ctlRowImpulse(n, g.k, fmt(id.X[g.k][i], 4), fmt(m.C[i] * id.X[g.k][i], 4), fmt(id.hR[g.k], 4))}</p>`;
    graphInfo.innerHTML = html;
  }

  compute();
  drawAll();
  return { redraw: drawAll };
}

// ───────────── helpers ─────────────
const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const sub = (n) => String(n).split("").map((c) => SUBS[c]).join("");
const num = (v) => fmt(v, 3).replace("−", "-");
const sgn = (v) => (v < 0 ? "−" : "+");
const cfmt = (z, d) => `${fmt(z.re, d)} ${z.im < 0 ? "−" : "+"} ${fmt(Math.abs(z.im), d)}i`;
const mat = (A) => `\\begin{bmatrix} ${A.map((r) => r.map(num).join(" & ")).join(" \\\\ ")} \\end{bmatrix}`;
const col = (v) => `\\begin{bmatrix} ${Array.from(v).map(num).join(" \\\\ ")} \\end{bmatrix}`;
const row = (v) => `\\begin{bmatrix} ${Array.from(v).map(num).join(" & ")} \\end{bmatrix}`;

function relErr(a, b) {
  let se = 0;
  let v = 0;
  for (let i = 0; i < a.length; i++) {
    se += (a[i] - b[i]) ** 2;
    v += b[i] ** 2;
  }
  return Math.sqrt(se / v);
}

// a display formula, typeset by KaTeX once it has loaded (it loads async), plain text until then
function tex(box, src, plain) {
  const el = h("div", { class: "formula" });
  const render = () => window.katex.render(src, el, { displayMode: true, throwOnError: false });
  if (window.katex) render();
  else {
    el.textContent = plain;
    document.getElementById("katex-js")?.addEventListener("load", render, { once: true });
  }
  box.replaceChildren(el);
}

// A signal-flow graph. nodes: [{ id, x, y, label, cls, tip }], edges: [{ from, to, w, curve, label, labelAt, dashed, thin, group }].
// Edge color and width follow the weight (diverging palette), scaled per group when perGroupScale is set.
function flowGraph({ width, height, nodes, edges, scale = null, perGroupScale = false, ring = null, aria, tip }) {
  const svg = s("svg", { class: "ctlviz", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": aria });
  const mid = `ah-${Math.random().toString(36).slice(2, 8)}`;
  svg.append(s("defs", null, s("marker", { id: mid, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 9, markerHeight: 9, markerUnits: "userSpaceOnUse", orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-head-ink" }))));
  const at = new Map(nodes.map((n) => [n.id, n]));
  const rOf = (n) => (n.cls.includes("small") ? 6 : n.cls.startsWith("io") ? 20 : 18);
  const lims = new Map();
  for (const e of edges) {
    const g = perGroupScale ? e.group ?? "A" : "all";
    lims.set(g, Math.max(lims.get(g) ?? 1e-12, Math.abs(e.w)));
  }
  const edgeLayer = s("g");
  const labelLayer = s("g");
  for (const e of edges) {
    const a = at.get(e.from);
    const b = at.get(e.to);
    const lim = scale ?? lims.get(perGroupScale ? e.group ?? "A" : "all");
    const stroke = rgbCss(diverging(e.w, lim));
    const sw = (e.thin ? 0.5 : 1.4) + (e.thin ? 1.6 : 2.6) * Math.min(1, Math.abs(e.w) / lim);
    let d;
    let lx;
    let ly;
    if (e.from === e.to) {
      const r = rOf(a);
      d = `M${a.x - 8},${a.y - r + 2} C${a.x - 30},${a.y - r - 44} ${a.x + 30},${a.y - r - 44} ${a.x + 8},${a.y - r + 2}`;
      lx = a.x - 34;
      ly = a.y - r - 24;
    } else {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const L = Math.hypot(dx, dy) || 1;
      const nx = -dy / L;
      const ny = dx / L;
      const c = e.curve ?? 0;
      const cx = (a.x + b.x) / 2 + nx * c;
      const cy = (a.y + b.y) / 2 + ny * c;
      // start and end on the node rims, along the tangent of the curve
      const s0 = unit(cx - a.x, cy - a.y);
      const s1 = unit(b.x - cx, b.y - cy);
      const x0 = a.x + s0[0] * (rOf(a) + 1);
      const y0 = a.y + s0[1] * (rOf(a) + 1);
      const x1 = b.x - s1[0] * (rOf(b) + 2);
      const y1 = b.y - s1[1] * (rOf(b) + 2);
      d = `M${x0.toFixed(1)},${y0.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
      const tt = e.labelAt ?? 0.5;
      lx = (1 - tt) ** 2 * x0 + 2 * (1 - tt) * tt * cx + tt * tt * x1;
      ly = (1 - tt) ** 2 * y0 + 2 * (1 - tt) * tt * cy + tt * tt * y1;
    }
    edgeLayer.append(s("path", { class: `ctl-edge${e.thin ? " thin" : ""}`, d, stroke, "stroke-width": sw.toFixed(2), "marker-end": e.thin ? null : `url(#${mid})`, "stroke-dasharray": e.dashed ? "4 4" : null }));
    if (e.label) labelLayer.append(s("text", { class: "ctl-elabel", x: lx, y: ly - 4, "text-anchor": "middle" }, e.label));
  }
  svg.append(edgeLayer);
  for (const n of nodes) {
    const g = s("g", { class: `ctl-node ${n.cls}` });
    const r = rOf(n);
    g.append(s("circle", { cx: n.x, cy: n.y, r }));
    if (ring != null && n.cls.includes("small")) g.append(s("circle", { class: "ring", cx: n.x, cy: n.y, r: r + 2.5, "stroke-opacity": Math.min(1, 0.2 + ring) }));
    if (n.label) g.append(s("text", { x: n.x, y: n.y + 4.5, "text-anchor": "middle" }, n.label));
    if (tip && n.tip != null) {
      g.addEventListener("pointermove", (ev) => showTip(tip(n.tip), ev));
      g.addEventListener("pointerleave", hideTip);
    }
    svg.append(g);
  }
  svg.append(labelLayer);
  return svg;
}

const unit = (x, y) => {
  const L = Math.hypot(x, y) || 1;
  return [x / L, y / L];
};

// The z-plane: unit circle, plant poles as crosses, reservoir eigenvalues as dots.
function zPlane({ crosses, dots, rings = [], aria }) {
  const S = 240;
  const lim = 1.15;
  const P = (v) => S / 2 + (v / lim) * (S / 2 - 10);
  const Q = (v) => S / 2 - (v / lim) * (S / 2 - 10);
  const svg = s("svg", { class: "eigviz", viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": aria });
  svg.append(
    s("line", { class: "axis", x1: 4, x2: S - 4, y1: S / 2, y2: S / 2 }),
    s("line", { class: "axis", x1: S / 2, x2: S / 2, y1: 4, y2: S - 4 }),
    s("circle", { class: "unit", cx: S / 2, cy: S / 2, r: P(1) - S / 2 }),
    s("text", { class: "lbl", x: P(1) + 3, y: S / 2 - 4 }, "1"),
    s("text", { class: "lbl", x: S - 6, y: S / 2 + 14, "text-anchor": "end" }, "Re z"),
    s("text", { class: "lbl", x: S / 2 + 5, y: 14 }, "Im z"),
  );
  const tipOf = (e) => `<span class="mono">z = ${fmt(e.re, 3)} ${e.im < 0 ? "−" : "+"} ${fmt(Math.abs(e.im), 3)}i<br>|z| = ${fmt(Math.hypot(e.re, e.im), 3)}</span>`;
  for (const e of dots) {
    const c = s("circle", { class: "ev", cx: P(e.re), cy: Q(e.im), r: e.r ?? 3 });
    c.addEventListener("pointermove", (ev) => showTip(tipOf(e) + (e.extra ? `<br><span class="mono">${e.extra}</span>` : ""), ev));
    c.addEventListener("pointerleave", hideTip);
    svg.append(c);
  }
  for (const e of rings) {
    const c = s("circle", { class: "era", cx: P(e.re), cy: Q(e.im), r: 9 });
    c.addEventListener("pointermove", (ev) => showTip(tipOf(e), ev));
    c.addEventListener("pointerleave", hideTip);
    svg.append(c);
  }
  for (const e of crosses) {
    const x = P(e.re);
    const y = Q(e.im);
    const g = s("g", { class: "pole" }, s("line", { x1: x - 6, y1: y - 6, x2: x + 6, y2: y + 6 }), s("line", { x1: x - 6, y1: y + 6, x2: x + 6, y2: y - 6 }));
    g.addEventListener("pointermove", (ev) => showTip(tipOf(e), ev));
    g.addEventListener("pointerleave", hideTip);
    svg.append(g);
  }
  return svg;
}

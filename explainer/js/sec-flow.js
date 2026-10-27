// 2 — one forecast, followed through both networks. The design follows Transformer Explainer (Cho et al., CHI 2026):
//  * one continuous data flow; every vector is a 1-D heatmap, 3 px per number, so a band is as wide as the vector it carries
//    and a matrix that changes a vector's length shows up as a funnel;
//  * click a component for an expanded view of its math with this step's numbers; hover a cell for its exact value;
//  * step or play through time, pick an example, change ρ and α live; a guided tour highlights one component at a time.

import { h, s, fmt, ctl, segToggle, slider, widthOf, visibility, perFrame, diverging, rgbCss, showTip, hideTip } from "./util.js";
import { lineChart, legendRow } from "./charts.js";
import { flowSetup, finishFlowFFN, refitFlowESN, explainFFN, explainESN, FLOW, MG } from "./core.js";
import { T } from "./i18n.js";

const VW = 1000;
const VH = 600;
const C = 3; // px per number
const COLW = 26;
const X = { series: 20, seriesEnd: 222, in: 300, mid: 540, out: 780 };
const FFN_Y = 92; // lane centers
const ESN_TOP = 214; // top of the 100-number state columns
const U_Y = 176; // the ESN's one-number input
const SHOW = 60; // series points drawn

const colH = (n) => Math.max(10, n * C);

export function initFlow(root) {
  const isVisible = visibility(root);
  const st = { t: MG.valEnd + 112, rho: FLOW.esn.rho, leak: FLOW.esn.leak, lambda: null, open: "neuron", unit: null, neuron: null, tour: -1, playing: false, ready: false };
  const flow = flowSetup();
  let ffn = null;
  let esn = null;

  // ── controls ──
  const back = h("button", { class: "btn", type: "button", "aria-label": T.stepBack }, "◀");
  const step = h("button", { class: "btn primary", type: "button" }, T.stepForward);
  const play = h("button", { class: "btn", type: "button" }, T.play);
  const tourBtn = h("button", { class: "btn", type: "button" }, T.guidedTour);
  const t0 = flow.T.test[0];
  const t1 = flow.T.test[1] - 1;
  const tCtl = slider(T.timeT, { min: t0, max: t1, step: 1, value: st.t, onInput: (v) => go(v, false), wide: true });
  back.addEventListener("click", () => go(st.t - 1, true));
  step.addEventListener("click", () => go(st.t + 1, true));
  play.addEventListener("click", () => (st.playing ? stopPlay() : startPlay()));
  tourBtn.addEventListener("click", () => (st.tour >= 0 ? endTour() : tour(0)));
  const examples = findExamples(flow.z, t0, t1);
  const exCtl = ctl(T.examples, segToggle(examples.map((e) => ({ value: e.t, label: T[e.key] })), null, (v) => go(v, true), T.examples));
  const refit = perFrame(() => {
    refitFlowESN(flow, { ...FLOW.esn, rho: st.rho, leak: st.leak }, st.lambda);
    draw();
  });
  const rhoCtl = slider(T.rhoLabel, { min: 0.1, max: 1.5, step: 0.05, value: st.rho, format: (v) => fmt(v, 2), onInput: (v) => ((st.rho = v), refit()) });
  const leakCtl = slider(T.leakLabel, { min: 0.05, max: 1, step: 0.05, value: st.leak, format: (v) => fmt(v, 2), onInput: (v) => ((st.leak = v), refit()) });
  // the ridge penalty: small λ buys accuracy with huge, cancelling readout terms; large λ gives readable weights and a worse fit
  const lamCtl = ctl("λ", segToggle([[null, T.lamValidation], [1e-8, "10⁻⁸"], [1e-6, "10⁻⁶"], [1e-4, "10⁻⁴"]].map(([value, label]) => ({ value, label })), null,
    (v) => ((st.lambda = v), refit()), "λ"));

  const tourCard = h("div", { class: "tour-card", hidden: true });
  const svgBox = h("div", { class: "flow-scroll" });
  const status = h("p", { class: "sub" });
  const panel = h("div", { class: "flow-panel" });
  const chartBox = h("div");
  root.replaceChildren(
    h("div", { class: "controls" }, back, step, play, tCtl, tourBtn),
    h("div", { class: "controls" }, exCtl, h("span", { class: "ctl" }, h("span", { class: "lbl" }, T.esnShort)), rhoCtl, leakCtl, lamCtl),
    tourCard,
    svgBox,
    h("p", { class: "sub narrow-only" }, T.scrollHint),
    status,
    panel,
    h("p", { class: "panel-title" }, T.flowChartTitle(MG.H)),
    legendRow([["ink", T.truth], ["s2", T.ffn], ["s1", T.esn]]),
    chartBox,
  );

  // train the small FFN once, in slices, so the page stays responsive
  const trainStep = () => {
    if (isVisible() || !st.ready) flow.trainer.run(25, false);
    if (!flow.trainer.done) {
      status.textContent = T.flowTraining(flow.trainer.epochs, FLOW.epochs);
      return requestAnimationFrame(trainStep);
    }
    finishFlowFFN(flow);
    st.ready = true;
    draw();
  };

  function go(t, animate) {
    st.t = Math.max(t0, Math.min(t1, t));
    tCtl.set(st.t);
    draw(animate);
  }
  function startPlay() {
    st.playing = true;
    play.textContent = T.pause;
    let last = 0;
    const tick = (now) => {
      if (!st.playing) return;
      if (now - last > 320 && isVisible()) {
        last = now;
        if (st.t >= t1) return stopPlay();
        go(st.t + 1, true);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  function stopPlay() {
    st.playing = false;
    play.textContent = T.play;
  }

  // ── guided tour ──
  const TOUR = [
    { comps: ["series"], key: 0 },
    { comps: ["window", "w1"], key: 1 },
    { comps: ["hidden", "w2", "ffnout"], key: 2 },
    { comps: ["u", "win"], key: 3 },
    { comps: ["prev", "wrec", "state", "loop"], key: 4 },
    { comps: ["wout", "esnout"], key: 5 },
    { comps: ["ffnout", "esnout"], key: 6 },
  ];
  function tour(i) {
    st.tour = i;
    tourBtn.textContent = T.endTour;
    const prev = h("button", { class: "btn", type: "button", disabled: i === 0 }, T.prev);
    const next = h("button", { class: "btn primary", type: "button" }, i === TOUR.length - 1 ? T.done : T.next);
    prev.addEventListener("click", () => tour(i - 1));
    next.addEventListener("click", () => (i === TOUR.length - 1 ? endTour() : tour(i + 1)));
    tourCard.hidden = false;
    tourCard.replaceChildren(h("span", { class: "tour-num" }, `${i + 1} / ${TOUR.length}`), h("p", { html: T.tourText[TOUR[i].key] }), h("div", { class: "tour-nav" }, prev, next));
    highlight(TOUR[i].comps);
  }
  function endTour() {
    st.tour = -1;
    tourBtn.textContent = T.guidedTour;
    tourCard.hidden = true;
    highlight(null);
  }
  function highlight(comps) {
    const svg = svgBox.querySelector("svg");
    if (!svg) return;
    svg.classList.toggle("touring", !!comps);
    for (const g of svg.querySelectorAll("[data-comp]")) g.classList.toggle("hl", !!comps && comps.includes(g.dataset.comp));
  }

  // ── drawing ──
  function draw(animate = false) {
    if (!st.ready) {
      svgBox.replaceChildren(h("div", { class: "placeholder" }, h("span", { class: "spinner" }), T.flowTraining(flow.trainer.epochs, FLOW.epochs)));
      return;
    }
    ffn = explainFFN(flow, st.t);
    esn = explainESN(flow, st.t);
    if (st.unit == null) st.unit = argmaxAbs(ffn.outTerms);
    if (st.neuron == null) st.neuron = argmaxAbs(esn.outTerms.subarray(2)) ;
    const svg = buildSVG(animate);
    svgBox.replaceChildren(svg);
    if (st.tour >= 0) highlight(TOUR[st.tour].comps);
    const truth = flow.T.Y[st.t];
    status.innerHTML = T.flowStatus(st.t, MG.H, ffn.y, esn.y, truth, flow.ffnErr, flow.esnErr);
    drawPanel();
    drawChart();
  }

  function buildSVG(animate) {
    const svg = s("svg", { class: `flowviz${animate ? " flowing" : ""}`, viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": T.flowAria });
    const defs = s("defs");
    svg.append(defs);
    grad(defs, "g-ffn-in", "muted", "s2");
    grad(defs, "g-ffn", "s2", "s2");
    grad(defs, "g-esn-in", "muted", "s1");
    grad(defs, "g-esn", "s1", "s1");
    const comp = (key, ...kids) => {
      const g = s("g", { "data-comp": key }, ...kids);
      svg.append(g);
      return g;
    };
    const winH = colH(FLOW.L);
    const hidH = colH(FLOW.hidden);
    const resH = colH(FLOW.N);
    const ffnTop = FFN_Y - winH / 2;

    // lane labels
    svg.append(
      s("text", { class: "lane", x: X.in - 6, y: ffnTop - 26 }, T.laneFFN),
      s("text", { class: "lane", x: X.in - 6, y: U_Y - 26 }, T.laneESN),
    );

    // ── shared input: the series up to t ──
    const sg = comp("series");
    const sx0 = X.series;
    const sx1 = X.seriesEnd;
    const sy0 = 40;
    const sy1 = 196;
    sg.append(s("rect", { class: "panel-bg", x: sx0 - 6, y: sy0 - 22, width: sx1 - sx0 + 12, height: sy1 - sy0 + 34, rx: 8 }));
    sg.append(s("text", { class: "clabel", x: sx0, y: sy0 - 8 }, T.seriesUpTo));
    const z = flow.z;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = st.t - SHOW + 1; i <= st.t; i++) (lo = Math.min(lo, z[i])), (hi = Math.max(hi, z[i]));
    const px = (i) => sx0 + ((i - (st.t - SHOW + 1)) / (SHOW - 1)) * (sx1 - sx0);
    const py = (v) => sy1 - ((v - lo) / (hi - lo || 1)) * (sy1 - sy0 - 10) - 5;
    sg.append(s("rect", { class: "win-shade", x: px(st.t - FLOW.L + 1) - 2, y: sy0, width: px(st.t) - px(st.t - FLOW.L + 1) + 4, height: sy1 - sy0 }));
    let d = "";
    for (let i = st.t - SHOW + 1; i <= st.t; i++) d += `${d ? "L" : "M"}${px(i).toFixed(1)},${py(z[i]).toFixed(1)}`;
    sg.append(s("path", { class: "series-line", d }));
    sg.append(s("circle", { class: "now-dot", cx: px(st.t), cy: py(z[st.t]), r: 4 }));
    const dotUp = py(z[st.t]) > (sy0 + sy1) / 2;
    sg.append(s("text", { class: "csub", x: px(st.t) - 7, y: py(z[st.t]) + (dotUp ? -8 : 14), "text-anchor": "end" }, "z(t)"));
    sg.append(s("text", { class: "csub", x: (px(st.t - FLOW.L + 1) + px(st.t)) / 2, y: sy1 + 9, "text-anchor": "middle" }, T.lastL(FLOW.L)));

    // ── FFN lane ──
    band(comp("w1in"), sx1 + 6, ffnTop, ffnTop + winH, X.in, ffnTop, ffnTop + winH, "g-ffn-in", false, null);
    const wcol = column(comp("window"), X.in, ffnTop, ffn.w, 2, (k) => `z(t${k ? `−${k}` : ""}) = ${fmt(ffn.w[k], 3)}`, null);
    label(wcol, X.in, ffnTop, T.colWindow(FLOW.L), null);
    const hidTop = FFN_Y - hidH / 2;
    band(comp("w1"), X.in + COLW, ffnTop, ffnTop + winH, X.mid, hidTop, hidTop + hidH, "g-ffn", true, `W₁ · ${FLOW.L} → ${FLOW.hidden} · ${T.trained}`);
    const hcol = column(comp("hidden"), X.mid, hidTop, ffn.h, 1, (j) => `h${sub(j + 1)} = tanh(${fmt(ffn.pre[j], 3)}) = ${fmt(ffn.h[j], 3)}`, (j) => open("hidden", { unit: j }), st.open === "hidden" ? st.unit : null);
    label(hcol, X.mid, hidTop, T.colHidden(FLOW.hidden), () => open("hidden"));
    const outTop = FFN_Y - 5;
    band(comp("w2"), X.mid + COLW, hidTop, hidTop + hidH, X.out, outTop, outTop + 10, "g-ffn", true, `W₂ · ${FLOW.hidden} → 1 · ${T.trained}`);
    const fo = outNode(comp("ffnout"), outTop, ffn.y, "s2", () => open("ffnout"));
    label(fo, X.out, outTop, "ŷ(t+20)", () => open("ffnout"));
    hcol.append(s("text", { class: "csub", x: X.mid + COLW / 2, y: hidTop + hidH + 16, "text-anchor": "middle" }, T.noCarry));

    // ── ESN lane ──
    band(comp("uin"), sx1 + 6, U_Y - 4, U_Y + 4, X.in, U_Y - 4, U_Y + 4, "g-esn-in", false, null);
    const ucol = column(comp("u"), X.in, U_Y - 5, [esn.u], 2, () => `u(t) = z(t) = ${fmt(esn.u, 3)}`, null);
    ucol.querySelector("rect").setAttribute("height", 10);
    label(ucol, X.in, U_Y - 5, "u(t) · 1", null);
    const pcol = column(comp("prev"), X.in, ESN_TOP, esn.prev, 1, (i) => `x${sub(i + 1)}(t−1) = ${fmt(esn.prev[i], 3)}`, null, null, neighbours());
    label(pcol, X.in, ESN_TOP, T.colPrev(FLOW.N), null);
    band(comp("win"), X.in + COLW, U_Y - 5, U_Y + 5, X.mid, ESN_TOP, ESN_TOP + resH, "g-esn-in", false, `W_in · 1 → ${FLOW.N} · ${T.fixed}`, 0, { x: X.in + COLW + 34, y: U_Y - 9, anchor: "start" });
    band(comp("wrec"), X.in + COLW, ESN_TOP, ESN_TOP + resH, X.mid, ESN_TOP, ESN_TOP + resH, "g-esn", false, `W · ${FLOW.N} → ${FLOW.N} · ${T.fixed}`, 40);
    const scol = column(comp("state"), X.mid, ESN_TOP, esn.x, 1, (i) => `x${sub(i + 1)}(t) = ${fmt(esn.x[i], 3)}`, (i) => open("neuron", { neuron: i }), st.open === "neuron" ? st.neuron : null);
    label(scol, X.mid, ESN_TOP, T.colState(FLOW.N), () => open("neuron"));
    const eoTop = ESN_TOP + resH / 2 - 5;
    band(comp("wout"), X.mid + COLW, ESN_TOP, ESN_TOP + resH, X.out, eoTop, eoTop + 10, "g-esn", true, `W_out · ${FLOW.N + 2} → 1 · ${T.trained}`);
    const eo = outNode(comp("esnout"), eoTop, esn.y, "s1", () => open("readout"));
    label(eo, X.out, eoTop, "ŷ(t+20)", () => open("readout"));
    // the recurrence: x(t) becomes x(t−1) one step later
    const lg = comp("loop");
    const ly = ESN_TOP + resH + 30;
    lg.append(s("path", { class: "loop-path", d: `M${X.mid + COLW / 2},${ESN_TOP + resH + 2} C${X.mid + COLW / 2},${ly + 10} ${X.in + COLW / 2},${ly + 10} ${X.in + COLW / 2},${ESN_TOP + resH + 2}`, "marker-end": "url(#arrow)" }));
    lg.append(s("text", { class: "csub", x: (X.in + X.mid + COLW) / 2, y: ly + 22, "text-anchor": "middle" }, T.loopLabel));
    defs.append(s("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-head" })));

    // magnifier hints at the expandable parts
    lens(svg, X.mid + 48, hidTop - 12, () => open("hidden"));
    lens(svg, X.out + 54, outTop - 12, () => open("ffnout"));
    lens(svg, X.mid + 62, ESN_TOP - 12, () => open("neuron"));
    lens(svg, X.out + 54, eoTop - 12, () => open("readout"));
    return svg;
  }

  function neighbours() {
    if (st.open !== "neuron" || st.neuron == null) return null;
    const { idx, deg } = esn;
    return new Set(Array.from(idx.subarray(st.neuron * deg, (st.neuron + 1) * deg)));
  }

  // a 1-D heatmap: one stripe per number
  function column(g, x, y, vals, scale, tip, onClick, selected = null, marked = null) {
    g.append(s("rect", { class: "col-frame", x: x - 1, y: y - 1, width: COLW + 2, height: vals.length * C + 2, rx: 2 }));
    vals.forEach((v, i) => {
      g.append(s("rect", { class: "cell", x, y: y + i * C, width: COLW, height: C, fill: rgbCss(diverging(v, scale)) }));
      // the inputs of the selected neuron: a tick beside each one (the stripes are too thin to outline)
      if (marked?.has(i)) g.append(s("rect", { class: "mark", x: x + COLW + 3, y: y + i * C - 0.5, width: 7, height: C + 1, rx: 1 }));
    });
    // hit targets bigger than the stripes: one per number, full width, at least 6 px tall
    const hit = s("rect", { class: "hit", x: x - 8, y, width: COLW + 16, height: vals.length * C });
    hit.style.cursor = onClick ? "pointer" : "default";
    const at = (ev) => {
      const r = hit.getBoundingClientRect();
      return Math.max(0, Math.min(vals.length - 1, Math.floor(((ev.clientY - r.top) / r.height) * vals.length)));
    };
    hit.addEventListener("pointermove", (ev) => showTip(`<span class="mono">${tip(at(ev))}</span>${onClick ? `<br>${T.clickToExpand}` : ""}`, ev));
    hit.addEventListener("pointerleave", hideTip);
    if (onClick) hit.addEventListener("click", (ev) => onClick(at(ev)));
    g.append(hit);
    if (selected != null) g.append(s("rect", { class: "sel-ring", x: x - 3, y: y + selected * C - 2, width: COLW + 6, height: C + 4, rx: 2 }));
    return g;
  }

  function outNode(g, y, v, cls, onClick) {
    const r = s("rect", { class: "out-node", x: X.out, y, width: COLW, height: 10, rx: 2, fill: rgbCss(diverging(v, 2)) });
    r.style.stroke = `var(--${cls})`;
    g.append(r, s("text", { class: "out-val", x: X.out + COLW + 26, y: y + 9 }, fmt(v, 3)));
    const hit = s("rect", { class: "hit", x: X.out - 6, y: y - 8, width: COLW + 90, height: 26 });
    hit.style.cursor = "pointer";
    hit.addEventListener("click", onClick);
    g.append(hit);
    return g;
  }

  function label(g, x, y, text, onClick, below = false) {
    const t = s("text", { class: `clabel${onClick ? " link" : ""}`, x: x + COLW / 2, y: below ? y : y - 8, "text-anchor": "middle" }, text);
    if (onClick) t.addEventListener("click", onClick);
    g.append(t);
  }

  // at: an explicit label position { x, y, anchor } where the band's middle is crowded
  function band(g, xa, ya0, ya1, xb, yb0, yb1, gradId, trained, text, labelDy = 0, at = null) {
    const m = (xa + xb) / 2;
    const d = `M${xa},${ya0} C${m},${ya0} ${m},${yb0} ${xb},${yb0} L${xb},${yb1} C${m},${yb1} ${m},${ya1} ${xa},${ya1} Z`;
    g.append(s("path", { class: `band${trained ? " trained" : " fixed"}`, d, fill: `url(#${gradId})` }));
    const cy0 = (ya0 + ya1) / 2;
    const cy1 = (yb0 + yb1) / 2;
    g.append(s("path", { class: "band-flow", d: `M${xa},${cy0} C${m},${cy0} ${m},${cy1} ${xb},${cy1}` }));
    if (text) g.append(s("text", { class: "blabel", x: at?.x ?? m, y: at?.y ?? (cy0 + cy1) / 2 + labelDy - 4, "text-anchor": at?.anchor ?? "middle" }, text));
  }

  function lens(svg, x, y, onClick) {
    const g = s("g", { class: "lens", transform: `translate(${x},${y})`, role: "button", tabindex: 0, "aria-label": T.expand },
      s("circle", { cx: 0, cy: 0, r: 8 }), s("circle", { class: "glass", cx: -1, cy: -1, r: 3.4 }), s("line", { x1: 1.6, y1: 1.6, x2: 4.4, y2: 4.4 }));
    g.addEventListener("click", onClick);
    g.addEventListener("keydown", (ev) => (ev.key === "Enter" || ev.key === " ") && (ev.preventDefault(), onClick()));
    svg.append(g);
  }

  function open(key, extra = {}) {
    st.open = key;
    Object.assign(st, extra);
    draw();
    panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // ── expanded views ──
  function drawPanel() {
    const tabs = segToggle(
      [["hidden", T.xHidden], ["ffnout", T.xFfnOut], ["neuron", T.xNeuron], ["readout", T.xReadout]].map(([value, label]) => ({ value, label })),
      st.open, (v) => open(v), T.expandedViews);
    let body;
    if (st.open === "hidden") body = viewHidden();
    else if (st.open === "ffnout") body = viewFfnOut();
    else if (st.open === "readout") body = viewReadout();
    else body = viewNeuron();
    panel.replaceChildren(h("div", { class: "controls" }, h("span", { class: "ctl" }, h("span", { class: "lbl" }, T.expanded)), tabs), body);
    renderTex(panel);
  }

  function viewHidden() {
    const j = st.unit;
    const rows = Array.from({ length: ffn.L }, (_, k) => ({ label: `W₁[${j + 1},${k}]·z(t${k ? `−${k}` : ""})`, value: ffn.terms[j * ffn.L + k] }));
    rows.push({ label: `b${sub(j + 1)}`, value: ffn.b1[j], bias: true });
    return h("div", null,
      h("div", { class: "controls" }, ctl(T.unit, segToggle(Array.from({ length: ffn.M }, (_, i) => ({ value: i, label: String(i + 1) })), j, (v) => open("hidden", { unit: v }), T.unit))),
      h("div", { class: "viz-row" },
        stepCard("①", T.stepWeightedSum(ffn.L), barList(rows, ffn.pre[j])),
        stepCard("②", T.stepTanh, tanhPlot(ffn.pre[j], ffn.h[j]))),
      eq(`h_{${j + 1}} = \\tanh\\Big(\\textstyle\\sum_{k=0}^{${ffn.L - 1}} W_1[${j + 1},k]\\,z(t-k) + b_{${j + 1}}\\Big) = \\tanh(${tx(ffn.pre[j])}) = ${tx(ffn.h[j])}`,
        `h${sub(j + 1)} = tanh(Σ W₁·z + b) = tanh(${fmt(ffn.pre[j], 3)}) = ${fmt(ffn.h[j], 3)}`),
      h("p", { class: "sub" }, T.hiddenNote(ffn.L + 1)));
  }

  function viewFfnOut() {
    const rows = Array.from(ffn.outTerms, (v, j) => ({ label: `W₂[${j + 1}]·h${sub(j + 1)}`, value: v, onClick: () => open("hidden", { unit: j }) }));
    rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    rows.push({ label: "b₂", value: ffn.b2, bias: true });
    return h("div", null,
      stepCard("①", T.stepOutputSum(ffn.M), barList(rows, ffn.y)),
      eq(`\\hat y(t+20) = \\textstyle\\sum_{j=1}^{${ffn.M}} W_2[j]\\,h_j + b_2 = ${tx(ffn.y)} \\qquad y(t+20) = ${tx(flow.T.Y[st.t])}`,
        `ŷ(t+20) = Σ W₂·h + b₂ = ${fmt(ffn.y, 3)}; truth ${fmt(flow.T.Y[st.t], 3)}`),
      h("p", { class: "sub" }, T.ffnOutNote(flow.trainer.nTrained)));
  }

  function viewNeuron() {
    const i = st.neuron;
    const { deg, idx } = esn;
    const rows = [{ label: `W_in[${i + 1}]·u(t)`, value: esn.inTerm[i] }];
    for (let k = 0; k < deg; k++) {
      const jx = idx[i * deg + k];
      rows.push({ label: `W[${i + 1},${jx + 1}]·x${sub(jx + 1)}(t−1)`, value: esn.recTerms[i * deg + k] });
    }
    rows.push({ label: `b${sub(i + 1)}`, value: esn.bias[i], bias: true });
    const a = esn.leak;
    const mixRows = [
      { label: `(1−α)·x${sub(i + 1)}(t−1)`, value: (1 - a) * esn.prev[i] },
      { label: `α·tanh(a${sub(i + 1)})`, value: a * esn.act[i] },
    ];
    return h("div", null,
      h("div", { class: "controls" }, ctl(T.neuron, h("span", { class: "mono" }, `${i + 1} / ${esn.N}`)), h("span", { class: "sub" }, T.neuronHint)),
      h("div", { class: "viz-row" },
        stepCard("①", T.stepPre(deg), barList(rows, esn.pre[i])),
        stepCard("②", T.stepTanh, tanhPlot(esn.pre[i], esn.act[i])),
        stepCard("③", T.stepLeak(fmt(a, 2)), barList(mixRows, esn.x[i]))),
      eq(`x_{${i + 1}}(t) = (1-\\alpha)\\,x_{${i + 1}}(t-1) + \\alpha \\tanh\\big(W_{\\text{in}}u(t) + \\textstyle\\sum_j W_{${i + 1}j}\\,x_j(t-1) + b_{${i + 1}}\\big) = ${tx((1 - a) * esn.prev[i])} + ${tx(a * esn.act[i])} = ${tx(esn.x[i])}`,
        `x(t) = (1−α)·x(t−1) + α·tanh(a) = ${fmt(esn.x[i], 3)}`),
      h("p", { class: "sub" }, T.neuronNote(deg + 2)));
  }

  function viewReadout() {
    const names = (k) => (k === 0 ? "W_out[0]·1" : k === 1 ? "W_out[u]·u(t)" : `W_out[${k - 1}]·x${sub(k - 1)}(t)`);
    const all = Array.from(esn.outTerms, (v, k) => ({ label: names(k), value: v, k }));
    all.sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
    const top = all.slice(0, 12).map((r) => ({ ...r, onClick: r.k >= 2 ? () => open("neuron", { neuron: r.k - 2 }) : null }));
    const rest = all.slice(12).reduce((acc, r) => acc + r.value, 0);
    top.push({ label: T.otherTerms(all.length - 12), value: rest, bias: true });
    return h("div", null,
      stepCard("①", T.stepReadout(esn.p), barList(top, esn.y)),
      eq(`\\hat y(t+20) = W_{\\text{out}} \\begin{bmatrix} 1 \\\\ u(t) \\\\ x(t) \\end{bmatrix} = \\textstyle\\sum_{k=1}^{${esn.p}} W_{\\text{out},k}\\, f_k(t) = ${tx(esn.y)} \\qquad y(t+20) = ${tx(flow.T.Y[st.t])}`,
        `ŷ(t+20) = W_out·[1; u; x] = ${fmt(esn.y, 3)}; truth ${fmt(flow.T.Y[st.t], 3)}`),
      h("p", { class: "sub" }, T.readoutNote(esn.p, flow.lambda, Array.from(esn.outTerms).reduce((a, v) => a + Math.abs(v), 0), esn.y, st.lambda == null)));
  }

  function stepCard(num, title, body) {
    return h("div", { class: "step-card grow" }, h("p", { class: "panel-title" }, h("span", { class: "snum" }, num), " ", title), body);
  }

  // signed contributions as bars around zero, with their sum; rows with onClick jump to that unit's own view
  function barList(rows, total) {
    const lim = Math.max(1e-9, ...rows.map((r) => Math.abs(r.value)), Math.abs(total));
    const wrap = h("div", { class: "contrib" });
    for (const r of rows) {
      const w = (Math.abs(r.value) / lim) * 50;
      const bar = h("span", { class: "cbar", style: { left: r.value < 0 ? `${50 - w}%` : "50%", width: `${w}%`, background: rgbCss(diverging(r.value, lim)) } });
      const name = h("span", { class: `cname${r.onClick ? " link" : ""}${r.bias ? " muted" : ""}` }, r.label);
      if (r.onClick) name.addEventListener("click", r.onClick);
      wrap.append(name, h("span", { class: "ctrack" }, h("span", { class: "czero" }), bar), h("span", { class: "cval" }, fmt(r.value, 3)));
    }
    wrap.append(h("span", { class: "cname total" }, "Σ"), h("span", { class: "ctrack" }, h("span", { class: "czero" }),
      h("span", { class: "cbar total", style: { left: total < 0 ? `${50 - (Math.abs(total) / lim) * 50}%` : "50%", width: `${(Math.abs(total) / lim) * 50}%` } })),
    h("span", { class: "cval total" }, fmt(total, 3)));
    return wrap;
  }

  function tanhPlot(a, y) {
    const W = 220;
    const H = 130;
    const xr = Math.max(3, Math.ceil(Math.abs(a) + 0.5));
    const X_ = (v) => 20 + ((v + xr) / (2 * xr)) * (W - 30);
    const Y_ = (v) => H / 2 - v * (H / 2 - 12);
    let d = "";
    for (let i = 0; i <= 80; i++) {
      const v = -xr + (2 * xr * i) / 80;
      d += `${i ? "L" : "M"}${X_(v).toFixed(1)},${Y_(Math.tanh(v)).toFixed(1)}`;
    }
    return s("svg", { class: "chart tanh", viewBox: `0 0 ${W} ${H}`, width: W, height: H },
      s("line", { class: "axis", x1: 20, x2: W - 10, y1: H / 2, y2: H / 2 }), s("line", { class: "axis", x1: X_(0), x2: X_(0), y1: 6, y2: H - 6 }),
      s("path", { class: "line ink thin", d }),
      s("line", { class: "hair", x1: X_(a), x2: X_(a), y1: H / 2, y2: Y_(y), "stroke-dasharray": "3 3" }),
      s("circle", { class: "dot", cx: X_(a), cy: Y_(y), r: 5, fill: "var(--accent)" }),
      s("text", { x: a > 0 ? X_(a) - 8 : X_(a) + 8, y: Y_(y) + (y > 0 ? 16 : -8), "text-anchor": a > 0 ? "end" : "start" }, `tanh(${fmt(a, 2)}) = ${fmt(y, 3)}`),
      s("text", { x: W - 10, y: H / 2 - 4, "text-anchor": "end" }, "a"), s("text", { x: X_(0) + 4, y: 14 }, "tanh a"));
  }

  function eq(texSrc, plain) {
    return h("div", { class: "formula" }, h("span", { class: "tex", dataset: { tex: texSrc } }, plain));
  }

  function drawChart() {
    const W = widthOf(root);
    const from = Math.max(flow.T.test[0], st.t - 120);
    const origins = Array.from({ length: st.t - from + 1 }, (_, i) => from + i);
    const xsF = origins.map((o) => o + MG.H);
    const ff = [];
    const ee = [];
    for (const o of origins) {
      ff.push(explainFFN(flow, o).y);
      ee.push(flowESNPred(o));
    }
    const truthXs = Array.from({ length: st.t + MG.H - (from + MG.H) + 1 + MG.H }, (_, i) => from + i);
    chartBox.replaceChildren(lineChart({
      width: W, height: 200,
      series: [
        { xs: truthXs, ys: truthXs.map((x) => flow.z[x]), cls: "ink", label: T.truth },
        { xs: xsF, ys: ff, cls: "s2", label: T.ffn },
        { xs: xsF, ys: ee, cls: "s1", label: T.esn },
      ],
      marks: [{ x: st.t, cls: "now", label: "t" }, { x: st.t + MG.H, cls: "now", label: "t+20" }],
      points: [{ x: st.t + MG.H, y: ffn.y, cls: "s2" }, { x: st.t + MG.H, y: esn.y, cls: "s1" }],
      xLabel: T.timeStep, yFmt: (v) => fmt(v, 1), tipFmt: (v) => fmt(v, 3), tipTitle: (x) => `t = ${x}`,
    }));
  }
  function flowESNPred(o) {
    let y = 0;
    for (let k = 0; k < flow.p; k++) y += flow.F[o * flow.p + k] * flow.B[k];
    return y;
  }

  draw();
  requestAnimationFrame(trainStep);
  return { redraw: () => draw() };
}

// ── helpers ──
function grad(defs, id, a, b) {
  defs.append(s("linearGradient", { id, x1: 0, x2: 1, y1: 0, y2: 0 },
    s("stop", { offset: "0%", style: `stop-color: var(--${a})` }), s("stop", { offset: "100%", style: `stop-color: var(--${b})` })));
}
const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const sub = (n) => String(n).split("").map((c) => SUBS[c]).join("");
const tx = (v) => fmt(v, 3).replace("−", "-");
function argmaxAbs(a) {
  let k = 0;
  for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[k])) k = i;
  return k;
}
function renderTex(el) {
  if (!window.katex) return;
  for (const e of el.querySelectorAll(".tex[data-tex]")) {
    try {
      window.katex.render(e.dataset.tex, e, { displayMode: true, throwOnError: false, strict: "ignore" });
    } catch {
      /* keep the fallback text */
    }
  }
}
// three starting points worth looking at: a steep rise, a peak, a trough (in the first 400 test steps)
function findExamples(z, t0, t1) {
  const end = Math.min(t1, t0 + 400);
  let rise = t0 + 1;
  let peak = t0 + 1;
  let trough = t0 + 1;
  for (let t = t0 + 1; t < end; t++) {
    if (z[t] - z[t - 1] > z[rise] - z[rise - 1]) rise = t;
    if (z[t] > z[peak]) peak = t;
    if (z[t] < z[trough]) trough = t;
  }
  return [{ key: "exRise", t: rise }, { key: "exPeak", t: peak }, { key: "exTrough", t: trough }];
}

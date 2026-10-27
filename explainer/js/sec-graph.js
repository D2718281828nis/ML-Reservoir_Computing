// 2.2 — the loop, unrolled: the same 100 neurons at four consecutive steps. The slot labels stay put while the columns slide,
//        so on Step the column that was x(t) moves under the label x(t−1): that is all "x(t) becomes x(t−1)" means.
// 2.3 — the reservoir as a graph: 100 neurons and their 1,000 links. Hover to see whom a neuron listens to and who listens to it,
//        click to follow it in the other views, and watch one change (a nudge to one neuron, or one changed input) echo and fade.
// Both follow the flow view of 2.1 through its api: same time t, same selected neuron, same ρ, α and λ.

import { h, s, fmt, ctl, segToggle, slider, visibility, diverging, rgbCss, showTip, hideTip, robustScale } from "./util.js";
import { lineChart, legendRow, log10, powFmt } from "./charts.js";
import { explainESN, flowEcho, graphLayout } from "./core.js";
import { T } from "./i18n.js";

const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const sub = (n) => String(n).split("").map((c) => SUBS[c]).join("");

// ───────────── 2.2 the loop, unrolled ─────────────
const LC = 3; // px per number, as in 2.1
const LW = 26;
const SLOT = 220;
const SLOTS = [150, 370, 590, 810];
const TOP = 96;

export function initLoop(root, api) {
  const flow = api.flow;
  const N = flow.res.N;
  const svgBox = h("div", { class: "flow-scroll" });
  const ledger = h("div", { class: "scroll-x" });
  const controls = h("div", { class: "controls" });
  root.replaceChildren(controls, svgBox, h("p", { class: "sub narrow-only" }, T.scrollHint), h("p", { class: "panel-title" }, T.ledgerTitle), ledger);
  let shownT = null;

  const back = h("button", { class: "btn", type: "button", "aria-label": T.stepBack }, "◀");
  const fwd = h("button", { class: "btn primary", type: "button" }, T.stepForward);
  back.addEventListener("click", () => api.go(api.t - 1));
  fwd.addEventListener("click", () => api.go(api.t + 1));
  const who = h("span", { class: "sub" });
  controls.replaceChildren(back, fwd, who);

  function render(dt) {
    const t = api.t;
    const i = api.neuron;
    const { res, X, z } = flow;
    const { deg, idx, w, leak } = res;
    const times = [t - 3, t - 2, t - 1, t];
    who.innerHTML = T.loopWho(i + 1);
    const resH = N * LC;
    const svg = s("svg", { class: "flowviz loopviz", viewBox: `0 0 1000 ${TOP + resH + 86}`, role: "img", "aria-label": T.loopAria });
    const defs = s("defs");
    defs.append(
      s("clipPath", { id: "loop-clip" }, s("rect", { x: 60, y: 0, width: 900, height: TOP + resH + 86 })),
      s("marker", { id: "arrow2", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-head" })),
    );
    svg.append(defs);

    // fixed layer: the slot names. They do not move; the columns slide under them.
    const names = ["x(t−3)", "x(t−2)", "x(t−1)", "x(t)"];
    SLOTS.forEach((x, c) => svg.append(s("text", { class: `clabel${c === 3 ? " now" : ""}`, x: x + LW / 2, y: TOP + resH + 18, "text-anchor": "middle" }, names[c])));
    svg.append(
      s("text", { class: "csub", x: SLOTS[2] + LW / 2, y: TOP + resH + 34, "text-anchor": "middle" }, T.readNow),
      s("text", { class: "csub", x: SLOTS[3] + LW / 2, y: TOP + resH + 34, "text-anchor": "middle" }, T.writtenNow),
    );
    // the handover arrow, from the x(t) slot back to the x(t−1) slot
    const ay = TOP + resH + 50;
    svg.append(
      s("path", { class: "loop-path", d: `M${SLOTS[3] + LW / 2},${ay - 8} C${SLOTS[3]},${ay + 12} ${SLOTS[2] + LW},${ay + 12} ${SLOTS[2] + LW / 2},${ay - 8}`, "marker-end": "url(#arrow2)" }),
      s("text", { class: "csub", x: (SLOTS[2] + SLOTS[3] + LW) / 2, y: ay + 22, "text-anchor": "middle" }, T.handover),
    );

    // sliding layer: the columns and what passes between them
    const slide = s("g", { class: "slide" });
    const clip = s("g", { "clip-path": "url(#loop-clip)" }, slide);
    svg.append(clip);
    const yOf = (j) => TOP + j * LC + LC / 2;
    times.forEach((tt, c) => {
      const x = SLOTS[c];
      const st = X.subarray(tt * N, (tt + 1) * N);
      // the input of this step enters from above
      const u = z[tt];
      slide.append(
        s("text", { class: "csub", x: x + LW / 2, y: 26, "text-anchor": "middle" }, `u(${tlabel(tt - t)}) = ${fmt(u, 2)}`),
        s("rect", { class: "out-node", x, y: 34, width: LW, height: 10, rx: 2, fill: rgbCss(diverging(u, 2)), style: "stroke: var(--line-strong)" }),
        s("line", { class: "in-line", x1: x + LW / 2, y1: 46, x2: x + LW / 2, y2: TOP - 4, "marker-end": "url(#arrow2)" }),
      );
      slide.append(s("rect", { class: "col-frame", x: x - 1, y: TOP - 1, width: LW + 2, height: resH + 2, rx: 2, style: "stroke-dasharray: 3 2" }));
      for (let j = 0; j < N; j++) slide.append(s("rect", { class: "cell", x, y: TOP + j * LC, width: LW, height: LC, fill: rgbCss(diverging(st[j], 1)) }));
      slide.append(s("rect", { class: "sel-ring", x: x - 3, y: TOP + i * LC - 2, width: LW + 6, height: LC + 4, rx: 2 }));
      const hit = s("rect", { class: "hit", x: x - 6, y: TOP, width: LW + 12, height: resH });
      hit.addEventListener("pointermove", (ev) => {
        const r = hit.getBoundingClientRect();
        const j = Math.max(0, Math.min(N - 1, Math.floor(((ev.clientY - r.top) / r.height) * N)));
        showTip(`<span class="mono">x${sub(j + 1)}(${tlabel(tt - t)}) = ${fmt(st[j], 3)}</span><br>${T.clickToFollow}`, ev);
      });
      hit.addEventListener("pointerleave", hideTip);
      hit.addEventListener("click", (ev) => {
        const r = hit.getBoundingClientRect();
        api.select(Math.max(0, Math.min(N - 1, Math.floor(((ev.clientY - r.top) / r.height) * N))));
      });
      if (c === 0) return;
      // between column c−1 and c: the leak keeps each neuron's own value (straight band) ...
      const x0 = SLOTS[c - 1] + LW;
      slide.append(s("rect", { class: "keep-band", x: x0, y: TOP, width: x - x0, height: resH }));
      slide.append(s("line", { class: "keep-line", x1: x0, y1: yOf(i), x2: x, y2: yOf(i) }));
      // ... and W mixes in the 10 neurons that neuron i listens to
      const prev = X.subarray((tt - 1) * N, tt * N);
      const contrib = Array.from({ length: deg }, (_, k) => w[i * deg + k] * prev[idx[i * deg + k]]);
      const lim = Math.max(1e-9, ...contrib.map(Math.abs));
      for (let k = 0; k < deg; k++) {
        const j = idx[i * deg + k];
        const cv = contrib[k];
        slide.append(s("path", {
          class: "mix-line", d: `M${x0},${yOf(j)} C${x0 + 60},${yOf(j)} ${x - 60},${yOf(i)} ${x},${yOf(i)}`,
          stroke: rgbCss(diverging(cv, lim)), "stroke-width": 0.8 + (2.6 * Math.abs(cv)) / lim,
        }));
        slide.append(s("rect", { class: "mark", x: x0 + 2, y: TOP + j * LC - 0.5, width: 6, height: LC + 1, rx: 1 }));
      }
      if (c === 1) {
        slide.append(s("text", { class: "blabel", x: (x0 + x) / 2, y: TOP - 10, "text-anchor": "middle" }, T.keepLabel(fmt(1 - leak, 2))));
      }
      slide.append(s("text", { class: "csub", x: (x0 + x) / 2, y: TOP + resH + 18, "text-anchor": "middle" }, T.updateAt(tlabel(tt - t))));
    });
    svgBox.replaceChildren(svg);
    // slide in: the new columns start one slot to the right (or left) of where they belong, then move into place
    if (dt === 1 || dt === -1) {
      slide.style.transform = `translateX(${dt * SLOT}px)`;
      slide.getBoundingClientRect();
      requestAnimationFrame(() => {
        slide.style.transition = "transform .55s cubic-bezier(.3,.7,.2,1)";
        slide.style.transform = "translateX(0)";
      });
    }
    renderLedger(times, i);
  }

  function renderLedger(times, i) {
    const t = api.t;
    const steps = times.slice(1).map((tt) => explainESN(flow, tt));
    const a = flow.res.leak;
    const row = (label, f, cls = "") => h("tr", { class: cls }, h("td", null, label), ...steps.map((e, c) => f(e, c)));
    const num = (v, extra = null) => h("td", extra, fmt(v, 3));
    const recSum = (e) => {
      let r = 0;
      for (let k = 0; k < e.deg; k++) r += e.recTerms[i * e.deg + k];
      return r;
    };
    ledger.replaceChildren(h("table", { class: "tbl ledger" },
      h("thead", null, h("tr", null, h("th", null, ""), ...steps.map((e) => h("th", null, T.updateAt(tlabel(e.t - t)))))),
      h("tbody", null,
        row(T.ledReads(i + 1), (e, c) => num(e.prev[i], { class: c > 0 ? "hand" : null, title: c > 0 ? T.sameNumber : null })),
        row(T.ledInput, (e) => num(e.inTerm[i])),
        row(T.ledRecurrent, (e) => num(recSum(e))),
        row(T.ledBias, (e) => num(e.bias[i])),
        row(T.ledTanh, (e) => h("td", null, `tanh(${fmt(e.pre[i], 2)}) = ${fmt(e.act[i], 3)}`)),
        row(T.ledWrites(i + 1, fmt(a, 2)), (e, c) => num(e.x[i], { class: c < steps.length - 1 ? "hand" : "hand-now", title: c < steps.length - 1 ? T.sameNumber : null }), "writes"),
      )),
    h("p", { class: "sub" }, T.ledgerNote));
  }

  api.subscribe(({ dt }) => {
    const animate = shownT != null && api.t - shownT === dt ? dt : 0;
    shownT = api.t;
    render(animate);
  });
  shownT = api.t;
  render(0);
  return { redraw: () => render(0) };
}

const tlabel = (d) => (d === 0 ? "t" : `t${d < 0 ? "−" : "+"}${Math.abs(d)}`);

// ───────────── 2.3 the reservoir as a graph ─────────────
const GS = 600; // graph viewBox
const GM = 28;

export function initGraph(root, api) {
  const flow = api.flow;
  const res = flow.res;
  const { N, deg, idx } = res;
  const isVisible = visibility(root);
  const P = graphLayout(res);
  const gx = (i) => GM + P[2 * i] * (GS - 2 * GM);
  const gy = (i) => GM + P[2 * i + 1] * (GS - 2 * GM);
  const outs = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) for (let k = 0; k < deg; k++) outs[idx[i * deg + k]].push(i);
  const st = { mode: "state", source: "neuron", k: 3, hover: null, echo: null, playing: false };

  // ── the graph ──
  const svg = s("svg", { class: "graphviz", viewBox: `0 0 ${GS} ${GS}`, role: "img", "aria-label": T.graphAria });
  svg.append(s("defs", null,
    s("marker", { id: "arrow3", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, markerUnits: "userSpaceOnUse", orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-head-ink" }))));
  const edgeLayer = s("g", { class: "g-edges" });
  for (let i = 0; i < N; i++) for (let k = 0; k < deg; k++) {
    const j = idx[i * deg + k];
    edgeLayer.append(s("line", { x1: gx(j), y1: gy(j), x2: gx(i), y2: gy(i) }));
  }
  const overlay = s("g", { class: "g-overlay" });
  const nodeLayer = s("g", { class: "g-nodes" });
  const nodes = Array.from({ length: N }, (_, i) => {
    const c = s("circle", { cx: gx(i), cy: gy(i), r: 7, class: "g-node" });
    c.addEventListener("pointerenter", (ev) => ((st.hover = i), drawOverlay(), tip(i, ev)));
    c.addEventListener("pointermove", (ev) => tip(i, ev));
    c.addEventListener("pointerleave", () => ((st.hover = null), drawOverlay(), hideTip()));
    c.addEventListener("click", () => api.select(i));
    nodeLayer.append(c);
    return c;
  });
  const selRing = s("circle", { class: "g-sel", r: 11 });
  const inputBadge = s("g", { class: "g-input" }, s("circle", { cx: 22, cy: 22, r: 14 }), s("text", { x: 22, y: 26, "text-anchor": "middle" }, "u"));
  svg.append(edgeLayer, overlay, nodeLayer, selRing, inputBadge);

  // ── controls and side panel ──
  const modeCtl = ctl(T.colorBy, segToggle([["state", T.modeState], ["readout", T.modeReadout], ["echo", T.modeEcho]].map(([value, label]) => ({ value, label })), st.mode, (v) => {
    st.mode = v;
    echoBox.hidden = v !== "echo";
    refresh();
  }, T.colorBy));
  const sourceCtl = ctl(T.echoSource, segToggle([["neuron", T.echoNudge], ["input", T.echoInput]].map(([value, label]) => ({ value, label })), st.source, (v) => {
    st.source = v;
    computeEcho();
    refresh();
  }, T.echoSource));
  const kCtl = slider(T.stepsAfter, { min: 0, max: 39, step: 1, value: st.k, onInput: (v) => ((st.k = v), refresh()), wide: true });
  const playBtn = h("button", { class: "btn", type: "button" }, T.play);
  playBtn.addEventListener("click", () => (st.playing ? stopPlay() : startPlay()));
  const echoCharts = h("div");
  const echoText = h("p", { class: "sub" });
  const echoBox = h("div", { class: "echo-box", hidden: true }, h("div", { class: "controls" }, sourceCtl), h("div", { class: "controls" }, kCtl, playBtn), echoText, echoCharts);
  const facts = h("p", { class: "sub" });
  const legend = h("div");
  root.replaceChildren(
    h("div", { class: "controls" }, modeCtl),
    h("div", { class: "viz-row" }, h("div", { class: "graph-wrap" }, svg), h("div", { class: "side graph-side" }, legend, facts, echoBox)),
  );

  function startPlay() {
    st.playing = true;
    playBtn.textContent = T.pause;
    if (st.k >= 39) st.k = 0;
    let last = 0;
    const tick = (now) => {
      if (!st.playing) return;
      if (now - last > 260 && isVisible()) {
        last = now;
        if (st.k >= 39) return stopPlay();
        st.k++;
        kCtl.set(st.k);
        refresh();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  function stopPlay() {
    st.playing = false;
    playBtn.textContent = T.play;
  }

  function computeEcho() {
    st.echo = flowEcho(flow, api.t, { neuron: st.source === "neuron" ? api.neuron : null, delta: 0.5, K: 40 });
  }

  // node values and edge contributions for the current mode
  function values() {
    const t = api.t;
    if (st.mode === "echo") {
      const e = st.echo;
      const k = st.k;
      const v = e.dX.subarray(k * N, (k + 1) * N);
      const prev = k > 0 ? e.dX.subarray((k - 1) * N, k * N) : null;
      return { v, scale: Math.max(1e-12, ...Array.from(v, Math.abs)), prev, prevScale: prev ? Math.max(1e-12, ...Array.from(prev, Math.abs)) : 1 };
    }
    const x = flow.X.subarray(t * N, (t + 1) * N);
    const prev = flow.X.subarray((t - 1) * N, t * N);
    if (st.mode === "readout") {
      const v = Float64Array.from(x, (xi, i) => flow.B[2 + i] * xi);
      return { v, scale: robustScale(v, 0.95), prev, prevScale: 1 };
    }
    return { v: x, scale: 1, prev, prevScale: 1 };
  }

  function refresh() {
    const { v, scale } = values();
    nodes.forEach((c, i) => c.setAttribute("fill", rgbCss(diverging(v[i], scale))));
    const sel = api.neuron;
    selRing.setAttribute("cx", gx(sel));
    selRing.setAttribute("cy", gy(sel));
    inputBadge.classList.toggle("lit", st.mode === "echo" && st.source === "input" && st.k === 0);
    drawOverlay();
    drawSide();
  }

  // highlighted links: the hovered (or selected) neuron's incoming links, colored by what they pass on at this step;
  // in echo mode without hover, the 80 links that pass on the most of the difference
  function drawOverlay() {
    overlay.replaceChildren();
    const { prev } = values();
    const w = res.w;
    const focus = st.hover ?? (st.mode === "echo" ? null : api.neuron);
    const items = [];
    if (focus != null && prev) {
      for (let k = 0; k < deg; k++) items.push({ j: idx[focus * deg + k], i: focus, c: w[focus * deg + k] * prev[idx[focus * deg + k]] });
    } else if (st.mode === "echo" && prev) {
      for (let i = 0; i < N; i++) for (let k = 0; k < deg; k++) items.push({ j: idx[i * deg + k], i, c: w[i * deg + k] * prev[idx[i * deg + k]] });
      items.sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
      items.length = Math.min(items.length, 80);
    }
    const lim = Math.max(1e-12, ...items.map((e) => Math.abs(e.c)));
    for (const e of items) {
      const dx = gx(e.i) - gx(e.j);
      const dy = gy(e.i) - gy(e.j);
      const d = Math.hypot(dx, dy) || 1;
      // stop short of the target circle so the arrowhead stays visible
      overlay.append(s("line", {
        class: "g-hl", x1: gx(e.j), y1: gy(e.j), x2: gx(e.i) - (dx / d) * 9, y2: gy(e.i) - (dy / d) * 9,
        stroke: rgbCss(diverging(e.c, lim)), "stroke-width": 0.9 + (3 * Math.abs(e.c)) / lim, "marker-end": "url(#arrow3)",
      }));
    }
    if (focus != null) {
      for (const tgt of outs[focus]) overlay.append(s("line", { class: "g-out", x1: gx(focus), y1: gy(focus), x2: gx(tgt), y2: gy(tgt) }));
    }
  }

  function tip(i, ev) {
    const t = api.t;
    const x = flow.X[t * N + i];
    const parts = [`<b>${T.neuron} ${i + 1}</b>`, `<span class="mono">x${sub(i + 1)}(t) = ${fmt(x, 3)}</span>`,
      `<span class="mono">W_out·x = ${fmt(flow.B[2 + i] * x, 2)}</span>`, T.listensTo(deg, outs[i].length)];
    if (st.mode === "echo") parts.splice(2, 0, `<span class="mono">Δx = ${st.echo.dX[st.k * N + i].toExponential(1)}</span>`);
    showTip(parts.join("<br>"), ev);
  }

  function drawSide() {
    const m = st.mode;
    legend.replaceChildren(
      h("p", { class: "panel-title" }, T.graphLegendTitle),
      h("p", { class: "sub" }, m === "state" ? T.legendState : m === "readout" ? T.legendReadout : T.legendEcho),
    );
    const outDeg = outs.map((o) => o.length);
    facts.innerHTML = T.graphFacts(N, N * deg, deg, Math.min(...outDeg), Math.max(...outDeg), fmt(res.rho, 2), api.neuron + 1);
    if (m !== "echo") return;
    const e = st.echo;
    const k = st.k;
    let reach = 0;
    for (let i = 0; i < N; i++) if (Math.abs(e.dX[k * N + i]) > 1e-3) reach++;
    const num = (v) => (v >= 0.01 ? fmt(v, v >= 10 ? 0 : 2) : v.toExponential(1));
    echoText.innerHTML = st.source === "neuron" ? T.echoTextNudge(api.neuron + 1, k, reach, num(e.rms[k]), num(Math.abs(e.dY[k]))) : T.echoTextInput(k, reach, num(e.rms[k]), num(Math.abs(e.dY[k])));
    const ks = Array.from({ length: e.K }, (_, q) => q);
    const W = Math.max(260, (echoCharts.parentElement?.clientWidth || 320) - 4);
    echoCharts.replaceChildren(
      legendRow([["ink", T.echoRms], ["s1", T.echoForecast]]),
      lineChart({
        width: W, height: 170, x0: 0, x1: e.K - 1, y0: -5, y1: 2, yTicks: [-4, -2, 0, 2], yFmt: powFmt,
        series: [
          { xs: ks, ys: Array.from(e.rms, log10), cls: "ink", label: T.echoRms },
          { xs: ks, ys: Array.from(e.dY, (v) => log10(Math.abs(v))), cls: "s1", label: T.echoForecast },
        ],
        marks: [{ x: k, cls: "now" }], xLabel: T.stepsAfterShort, tipFmt: (v) => (10 ** v).toExponential(1), tipTitle: (x) => `k = ${x}`,
        margin: { l: 40, r: 8, t: 6, b: 34 },
      }));
  }

  api.subscribe(() => {
    computeEcho();
    refresh();
  });
  computeEcho();
  refresh();
  return { redraw: refresh };
}


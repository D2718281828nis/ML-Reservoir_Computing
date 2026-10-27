// 6.1 — counting NG-RC features: constant, delay taps and their unique monomials.
// 6.2 — Lorenz-63: NG-RC against an ESN, trained on the same points, both run free afterwards.

import { h, fmt, ctl, segToggle, slider, widthOf, fitCanvas, colors, perFrame, ms, int } from "./util.js";
import { lineChart, legendRow } from "./charts.js";
import { monomials, binom } from "./models.js";
import { lorenzNGRC, lorenzESN, LZ, LZ_STEPS_PER_LT } from "./core.js";
import { T } from "./i18n.js";

// ───────────── 6.1 ─────────────
const SUB = "₀₁₂₃₄₅₆₇₈₉";
const SUP = { 2: "²", 3: "³" };
const PRESETS = [
  { key: "presetLorenz", d: 3, k: 2, s: 1, deg: [false, true, true, false], c: true },
  { key: "presetScroll", d: 3, k: 2, s: 1, deg: [false, true, false, true], c: false },
  { key: "presetOscillator", d: 2, k: 2, s: 1, deg: [false, true, false, true], c: false },
];

export function initCount(root) {
  const st = { d: 3, k: 2, s: 1, c: true, deg: [false, true, true, false] }; // deg[1..3]: linear, quadratic, cubic
  const out = h("div");
  const ctlD = slider(T.channelsD, { min: 1, max: 6, step: 1, value: st.d, onInput: (v) => ((st.d = v), draw()) });
  const ctlK = slider(T.tapsK, { min: 1, max: 5, step: 1, value: st.k, onInput: (v) => ((st.k = v), draw()) });
  const ctlS = slider(T.spacingS, { min: 1, max: 10, step: 1, value: st.s, onInput: (v) => ((st.s = v), draw()) });
  const box = (label, get, set) => {
    const input = h("input", { type: "checkbox" });
    input.checked = get();
    input.addEventListener("change", () => (set(input.checked), draw()));
    const el = h("label", { class: "toggle" }, input, label);
    el.sync = () => (input.checked = get());
    return el;
  };
  const toggles = [
    box(T.constantTerm, () => st.c, (v) => (st.c = v)),
    box(T.linearTerms, () => st.deg[1], (v) => (st.deg[1] = v)),
    box(T.quadraticTerms, () => st.deg[2], (v) => (st.deg[2] = v)),
    box(T.cubicTerms, () => st.deg[3], (v) => (st.deg[3] = v)),
  ];
  const presetBtns = PRESETS.map((p) => {
    const b = h("button", { class: "btn", type: "button" }, T[p.key]);
    b.addEventListener("click", () => {
      Object.assign(st, { d: p.d, k: p.k, s: p.s, c: p.c, deg: [...p.deg] });
      ctlD.set(p.d);
      ctlK.set(p.k);
      ctlS.set(p.s);
      toggles.forEach((t) => t.sync());
      draw();
    });
    return b;
  });
  root.replaceChildren(
    h("div", { class: "controls" }, ctlD, ctlK, ctlS),
    h("div", { class: "controls" }, ...toggles),
    h("div", { class: "controls" }, h("span", { class: "ctl" }, h("span", { class: "lbl" }, T.examples)), ...presetBtns),
    out,
  );

  function name(i) {
    const { d, s } = st;
    const c = i % d;
    const j = Math.floor(i / d);
    const v = d <= 3 ? "xyz"[c] : `u${String(c + 1).split("").map((q) => SUB[q]).join("")}`;
    return `${v}(t${j ? `−${j * s}` : ""})`;
  }
  function draw() {
    const m = st.d * st.k;
    const parts = [
      { label: T.constantTerm, n: st.c ? 1 : 0, op: 0.3, cls: "c0" },
      { label: T.linearTerms, n: st.deg[1] ? m : 0, op: 0.55, cls: "c1" },
      { label: T.quadraticTerms, n: st.deg[2] ? binom(m + 1, 2) : 0, op: 0.8, cls: "c2" },
      { label: T.cubicTerms, n: st.deg[3] ? binom(m + 2, 3) : 0, op: 1, cls: "c3" },
    ];
    const total = parts.reduce((a, p) => a + p.n, 0);
    const chips = [];
    if (total <= 140) {
      if (st.c) chips.push(h("span", { class: "c0" }, "1"));
      for (let deg = 1; deg <= 3; deg++) {
        if (!st.deg[deg]) continue;
        for (const combo of monomials(m, deg)) {
          const counts = new Map();
          for (const i of combo) counts.set(i, (counts.get(i) || 0) + 1);
          chips.push(h("span", { class: `c${deg}` }, [...counts].map(([i, n]) => name(i) + (SUP[n] || "")).join("·")));
        }
      }
    }
    out.replaceChildren(
      h("div", { class: "countbar", role: "img", "aria-label": T.featureMix },
        parts.filter((p) => p.n).map((p) => h("i", { style: { flex: `${p.n} 0 0`, background: "var(--s3)", opacity: p.op }, title: `${p.label}: ${p.n}` }))),
      h("div", { class: "legend-row" }, parts.map((p) => h("span", null, h("i", { class: "band bg-s3", style: { opacity: p.op } }), `${p.label} ${p.n}`))),
      h("div", { class: "stats" },
        h("span", null, h("b", null, int(total)), T.featuresTotal),
        h("span", null, h("b", null, int(total * st.d)), T.trainedWeightsD(st.d)),
        h("span", null, h("b", null, String((st.k - 1) * st.s)), T.lookBack((st.k - 1) * st.s))),
      chips.length ? h("div", { class: "chips" }, chips) : h("p", { class: "sub" }, T.tooManyToList),
    );
  }
  draw();
  return { redraw: draw };
}

// ───────────── 6.2 ─────────────
const SHOW_LT = 10;
const RANGES = [[-25, 25], [-30, 30], [0, 55]];

export function initLorenz(root) {
  const st = { ntr: 400, k: 2, quad: true };
  const esnCache = new Map();
  let ng = null;
  let esn = null;
  const recompute = perFrame(() => {
    compute();
    draw();
  });
  const quadBox = h("input", { type: "checkbox" });
  quadBox.checked = st.quad;
  quadBox.addEventListener("change", () => ((st.quad = quadBox.checked), recompute()));
  const charts = h("div");
  const attract = h("div", { class: "viz-row" });
  const stats = h("div", { class: "stats" });
  const note = h("p", { class: "sub" });
  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.trainPoints, segToggle([100, 200, 400, 1000, 4000].map((v) => ({ value: v, label: String(v) })), st.ntr, (v) => ((st.ntr = v), recompute()), T.trainPoints)),
      ctl(T.tapsK, segToggle([1, 2, 3, 4].map((v) => ({ value: v, label: String(v) })), st.k, (v) => ((st.k = v), recompute()), T.tapsK)),
      h("label", { class: "toggle" }, quadBox, T.quadraticTerms)),
    legendRow([["ink", T.truth], ["s3", T.ngrc], ["s1", T.esn]]),
    charts,
    attract,
    stats,
    note,
  );

  function compute() {
    ng = lorenzNGRC({ ntr: st.ntr, k: st.k, quad: st.quad });
    if (!esnCache.has(st.ntr)) esnCache.set(st.ntr, lorenzESN({ ntr: st.ntr }));
    esn = esnCache.get(st.ntr);
  }

  function draw() {
    const W = widthOf(root);
    const n = Math.round(SHOW_LT * LZ_STEPS_PER_LT);
    const xs = Array.from({ length: n }, (_, i) => i / LZ_STEPS_PER_LT);
    const clip = (v, [a, b]) => (Number.isFinite(v) ? Math.max(a - 15, Math.min(b + 15, v)) : NaN);
    const marks = [
      { x: ng.valids[0], cls: "now", label: T.ngrc },
      { x: esn.valids[0], cls: "now", label: T.esn },
    ];
    charts.replaceChildren(...["x", "y", "z"].map((name, c) => h("div", null,
      lineChart({
        width: W, height: 110, x0: 0, x1: SHOW_LT, y0: RANGES[c][0], y1: RANGES[c][1],
        series: [
          { xs, ys: xs.map((_, i) => ng.truth[i * 3 + c]), cls: "ink", label: T.truth },
          { xs, ys: xs.map((_, i) => clip(ng.pred[i * 3 + c], RANGES[c])), cls: "s3", label: T.ngrc },
          { xs, ys: xs.map((_, i) => clip(esn.pred[i * 3 + c], RANGES[c])), cls: "s1", label: T.esn },
        ],
        marks: c === 0 ? marks : marks.map((m) => ({ ...m, label: null })),
        yTicks: c === 2 ? [10, 30, 50] : [-20, 0, 20], yFmt: (v) => `${name} ${fmt(v, 0)}`,
        xTicks: [0, 2, 4, 6, 8, 10], xFmt: (v) => fmt(v, 0), xLabel: c === 2 ? T.lyapunovTimes : "", tipFmt: (v) => fmt(v, 2), tipTitle: (x) => `${fmt(x, 2)} ${T.lyapunovShort}`,
        margin: { l: 52, r: 12, t: 6, b: c === 2 ? 34 : 18 },
      }))));

    const wide = W > 640;
    const pw = wide ? Math.floor((W - 20) / 2) : W;
    const ph = Math.round(Math.min(300, pw * 0.75));
    attract.replaceChildren(
      panel(T.attractorOf(T.ngrc), ng, "s3", pw, ph),
      panel(T.attractorOf(T.esn), esn, "s1", pw, ph),
    );

    const span = (r) => `${fmt(Math.min(...r.valids), 1)}–${fmt(Math.max(...r.valids), 1)}`;
    stats.replaceChildren(
      h("span", null, h("b", null, fmt(ng.validLT, 1)), T.validLTof(T.ngrc, span(ng))),
      h("span", null, h("b", null, fmt(esn.validLT, 1)), T.validLTof(T.esn, span(esn))),
      h("span", null, h("b", null, `${int(ng.nTrained)} / ${int(esn.nTrained)}`), T.trainedNgEsn),
      h("span", null, h("b", null, `${ng.warmup} / ${esn.warmup}`), T.warmupNgEsn),
      h("span", null, h("b", null, `${ms(ng.fitTime)} / ${ms(esn.fitTime)}`), T.fitNgEsn),
    );
    note.textContent = T.lorenzNote(LZ.starts, LZ.gap, LZ.thr);
  }

  function panel(title, r, cls, w, hgt) {
    const canvas = h("canvas", { class: "plot" });
    const wrap = h("div", { class: "grow" }, h("p", { class: "panel-title" }, title), canvas);
    requestAnimationFrame(() => {
      const ctx = fitCanvas(canvas, w, hgt);
      const c = colors();
      const X = (v) => ((v - RANGES[0][0]) / (RANGES[0][1] - RANGES[0][0])) * w;
      const Y = (v) => hgt - ((v - RANGES[2][0]) / (RANGES[2][1] - RANGES[2][0])) * hgt;
      const line = (arr, stroke, width, alpha) => {
        ctx.strokeStyle = stroke;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = width;
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < LZ.steps; i++) {
          const x = arr[i * 3];
          const z = arr[i * 3 + 2];
          if (!Number.isFinite(x) || !Number.isFinite(z) || Math.abs(x) > 200 || Math.abs(z) > 400) {
            pen = false;
            continue;
          }
          if (pen) ctx.lineTo(X(x), Y(z));
          else ctx.moveTo(X(x), Y(z));
          pen = true;
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      };
      ctx.clearRect(0, 0, w, hgt);
      line(r.truth, c.muted, 1, 0.55);
      line(r.pred, c[cls], 1.2, 0.9);
      ctx.fillStyle = c.muted;
      ctx.font = "10.5px ui-monospace, monospace";
      ctx.fillText("x →", w - 30, hgt - 6);
      ctx.fillText("z ↑", 6, 14);
    });
    return wrap;
  }

  compute();
  draw();
  return { redraw: draw };
}

// 5.2 — the forecasting lab: three series, two baselines, a linear model, an ESN and an FFN, H horizons at once.
// 5.3 — closed loop: one-step models on Mackey–Glass that run on their own output.

import { h, fmt, ctl, segToggle, slider, widthOf, visibility, perFrame, ms, int, LANG_LOCALE } from "./util.js";
import { lineChart, legendRow, log10, powFmt, logTicks, logRange } from "./charts.js";
import { presets, labTask, baselines, fitLinear, fitESN, Trainer, esnFreeRun, freeTrainer, ffnFreeRun, mackey, MG, FREE } from "./core.js";
import { T } from "./i18n.js";

const fmt2 = (v) => fmt(v, 2);

// ───────────── 5.2 ─────────────
export function initLab(root, series) {
  const P = presets(series);
  const keys = ["electricity", "ecg", "mackey"].filter((k) => P[k]);
  const st = { ds: keys[0], pos: 0.42, esn: Object.fromEntries(keys.map((k) => [k, { ...P[k].esn }])), ffn: {}, training: null };
  const isVisible = visibility(root);
  let D = null;

  const trainBtn = h("button", { class: "btn primary", type: "button" }, T.trainFFN);
  const bar = h("i");
  const progress = h("span", { class: "progress", "aria-hidden": "true", hidden: true }, bar);
  trainBtn.addEventListener("click", () => trainFFN());
  const rhoCtl = slider(T.rhoLabel, { min: 0.1, max: 1.5, step: 0.05, value: st.esn[st.ds].rho, format: fmt2, onInput: (v) => ((st.esn[st.ds].rho = v), refit()) });
  const leakCtl = slider(T.leakLabel, { min: 0.05, max: 1, step: 0.05, value: st.esn[st.ds].leak, format: fmt2, onInput: (v) => ((st.esn[st.ds].leak = v), refit()) });
  const posCtl = slider(T.forecastFrom, { min: 0, max: 1, step: 0.002, value: st.pos, format: () => "", onInput: (v) => ((st.pos = v), drawTrajectory()), wide: true });
  const about = h("p", { class: "sub" });
  const trajBox = h("div");
  const errBox = h("div");
  const tableBox = h("div", { class: "scroll-x" });
  const legendBox = h("div");

  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.dataset, segToggle(keys.map((k) => ({ value: k, label: T.dsName[k] })), st.ds, (v) => pick(v), T.dataset)),
      trainBtn, progress),
    about,
    h("div", { class: "controls" }, h("span", { class: "ctl" }, h("span", { class: "lbl" }, T.esnShort)), rhoCtl, leakCtl),
    h("p", { class: "panel-title" }, T.oneForecast),
    legendBox,
    trajBox,
    h("div", { class: "controls" }, posCtl),
    h("p", { class: "panel-title" }, T.errorByHorizon),
    errBox,
    h("p", { class: "panel-title" }, T.testScores),
    tableBox,
  );

  function pick(ds) {
    st.ds = ds;
    rhoCtl.set(st.esn[ds].rho);
    leakCtl.set(st.esn[ds].leak);
    compute();
    draw();
  }

  function compute() {
    const p = P[st.ds];
    D = { p, task: labTask(p) };
    D.base = baselines(D.task);
    D.linear = fitLinear(D.task, p.L);
    D.esn = fitESN(D.task, { ...p.esn, ...st.esn[st.ds] });
  }
  const refit = perFrame(() => {
    D.esn = fitESN(D.task, { ...D.p.esn, ...st.esn[st.ds] });
    draw();
  });

  function trainFFN() {
    if (st.training) return;
    const ds = st.ds;
    const p = P[ds];
    const tr = new Trainer(D.task, p.L, { hidden: p.hidden, epochs: p.epochs, lr: p.lr, seed: 1 });
    st.training = { ds, tr };
    trainBtn.disabled = true;
    progress.hidden = false;
    const step = () => {
      if (isVisible()) tr.run(30, false);
      bar.style.width = `${(100 * tr.epochs) / p.epochs}%`;
      trainBtn.textContent = T.trainingEpoch(tr.epochs, p.epochs);
      if (!tr.done) return requestAnimationFrame(step);
      st.ffn[ds] = tr.result();
      st.training = null;
      progress.hidden = true;
      trainBtn.disabled = false;
      trainBtn.textContent = T.trainFFNAgain;
      if (st.ds === ds) draw();
    };
    requestAnimationFrame(step);
  }

  // ── formatting by dataset ──
  const xLabel = (t) => {
    const p = D.p;
    if (p.start != null) return new Date(p.start + t * 864e5).toLocaleDateString(LANG_LOCALE, { month: "short", day: "numeric", timeZone: "UTC" });
    if (p.fs) return `${fmt(t / p.fs, 2)} s`;
    return String(t);
  };
  const xTip = (t) => {
    const p = D.p;
    if (p.start != null) return new Date(p.start + t * 864e5).toLocaleDateString(LANG_LOCALE, { weekday: "short", year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
    if (p.fs) return `${fmt(t / p.fs, 3)} s`;
    return `t = ${t}`;
  };
  const hLabel = (hh) => {
    const p = D.p;
    if (p.start != null) return T.days(hh);
    if (p.fs) return `${fmt((hh * 1000) / p.fs, 0)} ms`;
    return String(hh);
  };

  function models() {
    const ffn = st.ffn[st.ds];
    return [
      { key: "esn", cls: "s1", r: D.esn },
      { key: "ffn", cls: "s2", r: ffn },
      { key: "linear", cls: "muted", r: D.linear, thin: true },
      { key: "persistence", cls: "muted", r: D.base.persistence, dash: true },
      { key: "seasonal", cls: "muted", r: D.base.seasonal, dot: true },
    ].filter((m) => m.r);
  }

  function drawTrajectory() {
    const { task } = D;
    const H = task.q;
    const W = widthOf(root);
    const nTest = task.test[1] - task.test[0];
    const r = Math.min(nTest - 1, Math.round(st.pos * (nTest - 1)));
    const t = task.test[0] + r;
    const back = D.p.start != null ? 35 : 150;
    const toX = (v) => D.task.mu + D.task.sd * v;
    const past = Array.from({ length: back + 1 }, (_, i) => t - back + i);
    const fut = Array.from({ length: H + 1 }, (_, i) => t + i);
    const series = [
      { xs: past, ys: past.map((i) => toX(task.z[i])), cls: "past", label: T.past, noTip: true },
      { xs: fut, ys: fut.map((i) => toX(task.z[i])), cls: "ink", label: T.truth },
    ];
    for (const m of models()) {
      const ys = [toX(task.z[t]), ...Array.from({ length: H }, (_, hh) => toX(m.r.P[r * H + hh]))];
      series.push({ xs: fut, ys, cls: m.cls, label: T[m.key], thin: m.thin, dash: m.dash, dot: m.dot });
    }
    trajBox.replaceChildren(lineChart({
      width: W, height: 250, series, marks: [{ x: t, cls: "now" }],
      xFmt: (v) => xLabel(Math.round(v)), yFmt: (v) => fmt(v, D.p.start != null ? 0 : 1), tipFmt: (v) => fmt(v, 2), tipTitle: (x) => xTip(x),
      xLabel: D.p.unit ? `${T.forecastOrigin} ${xTip(t)} · ${T.unitName(D.p.unit)}` : `${T.forecastOrigin} ${xTip(t)}`,
    }));
  }

  function draw() {
    const { task } = D;
    const H = task.q;
    const W = widthOf(root);
    about.innerHTML = T.dsAbout[st.ds](D.p, task);
    legendBox.replaceChildren(legendRow([["ink", T.truth], ["s1", T.esn], ["s2", T.ffn], ["muted", T.linear], ["muted", T.persistence, "dash"],
      ...(D.base.seasonal ? [["muted", T.seasonal, "dot"]] : [])]));
    trainBtn.textContent = st.training ? trainBtn.textContent : st.ffn[st.ds] ? T.trainFFNAgain : T.trainFFN;
    progress.hidden = !st.training;
    drawTrajectory();

    const hs = Array.from({ length: H }, (_, i) => i + 1);
    const ms_ = models();
    let lo = Infinity;
    let hi = -Infinity;
    for (const m of ms_) for (const e of m.r.err) (lo = Math.min(lo, log10(e))), (hi = Math.max(hi, log10(e)));
    const [y0, y1] = logRange(lo, hi);
    errBox.replaceChildren(lineChart({
      width: W, height: 200, x0: 1, x1: H, y0, y1, yTicks: logTicks(y0, y1), yFmt: powFmt, xTicks: horizonTicks(H, W),
      series: ms_.map((m) => ({ xs: hs, ys: Array.from(m.r.err, log10), cls: m.cls, label: T[m.key], thin: m.thin, dash: m.dash, dot: m.dot })),
      xFmt: (v) => hLabel(Math.round(v)), xLabel: T.horizon, tipFmt: (v) => fmt(10 ** v, 3), tipTitle: (x) => `${T.horizon}: ${hLabel(x)}`,
    }));

    const unit = D.p.unit;
    const rmse1 = (m) => {
      let se = 0;
      const n = task.test[1] - task.test[0];
      for (let r = 0; r < n; r++) se += (m.r.P[r * H] - task.Y[(task.test[0] + r) * H]) ** 2;
      return Math.sqrt(se / n) * task.sd;
    };
    const all = [
      { key: "esn", cls: "s1", r: D.esn },
      { key: "ffn", cls: "s2", r: st.ffn[st.ds] },
      { key: "linear", cls: "muted", r: D.linear },
      { key: "persistence", cls: "muted", r: D.base.persistence },
      ...(D.base.seasonal ? [{ key: "seasonal", cls: "muted", r: D.base.seasonal }] : []),
    ];
    tableBox.replaceChildren(h("table", { class: "tbl" },
      h("thead", null, h("tr", null, h("th", null, T.model), h("th", null, T.trainedWeights), h("th", null, T.fitTime), h("th", null, `h = ${hLabel(1)}`), h("th", null, `h = ${hLabel(H)}`),
        unit ? h("th", null, T.rmseIn(hLabel(1), T.unitName(unit))) : null)),
      h("tbody", null, all.map((m) => h("tr", { class: m.key === "persistence" || m.key === "seasonal" ? "ref" : null },
        h("td", null, h("span", { class: `key bg-${m.cls}`, style: { marginRight: "7px" } }), T[m.key]),
        h("td", null, m.r?.nTrained ? int(m.r.nTrained) : m.r ? "0" : "–"),
        h("td", null, m.r?.time != null ? ms(m.r.time) : "–"),
        h("td", null, m.r ? fmt(m.r.err[0], 3) : T.notTrained),
        h("td", null, m.r ? fmt(m.r.err[H - 1], 3) : "–"),
        unit ? h("td", null, m.r ? fmt(rmse1(m), D.p.start != null ? 2 : 1) : "–") : null)))),
      h("p", { class: "sub" }, T.labTableNote(D.esn.lambda, D.linear.lambda, st.ffn[st.ds]?.bestEpoch)));
  }

  compute();
  draw();
  return { redraw: draw };
}

// horizons are whole steps: integer ticks, about one per 90 px
function horizonTicks(H, W) {
  const step = [1, 2, 5, 10, 25].find((q) => H / q <= Math.max(4, Math.round(W / 90))) || 50;
  const out = [1];
  for (let v = step; v <= H; v += step) if (v > 1) out.push(v);
  return out;
}

// ───────────── 5.3 ─────────────
const SHOW_FREE = 700;

export function initFree(root) {
  const st = { seed: 1, esn: null, ffn: null, trainer: null };
  const isVisible = visibility(root);
  const { z, mu, sd } = mackey();
  const runBtn = h("button", { class: "btn primary", type: "button" }, T.trainFFNFree);
  const seedBtn = h("button", { class: "btn", type: "button" }, T.newSeeds);
  const bar = h("i");
  const progress = h("span", { class: "progress", "aria-hidden": "true", hidden: true }, bar);
  runBtn.addEventListener("click", () => trainFFN());
  seedBtn.addEventListener("click", () => {
    if (st.trainer) return;
    st.seed++;
    st.ffn = null;
    st.esn = esnFreeRun({ seed: st.seed });
    draw();
  });
  const trajBox = h("div");
  const errBox = h("div");
  const stats = h("div", { class: "stats" });
  root.replaceChildren(
    h("div", { class: "controls" }, runBtn, seedBtn, progress),
    legendRow([["ink", T.truth], ["s1", T.esnFree], ["s2", T.ffnFree]]),
    trajBox,
    h("p", { class: "panel-title" }, T.freeErrorTitle),
    errBox,
    stats,
  );

  function trainFFN() {
    if (st.trainer) return;
    const tr = (st.trainer = freeTrainer(st.seed));
    runBtn.disabled = seedBtn.disabled = true;
    progress.hidden = false;
    const step = () => {
      if (isVisible()) tr.run(30, false);
      bar.style.width = `${(100 * tr.epochs) / tr.maxEpochs}%`;
      runBtn.textContent = T.trainingEpoch(tr.epochs, tr.maxEpochs);
      if (!tr.done) return requestAnimationFrame(step);
      st.ffn = { ...ffnFreeRun(tr), time: tr.time, bestEpoch: tr.best.epoch };
      st.trainer = null;
      progress.hidden = true;
      runBtn.disabled = seedBtn.disabled = false;
      runBtn.textContent = T.trainFFNFree;
      draw();
    };
    requestAnimationFrame(step);
  }

  function draw() {
    const W = widthOf(root);
    const t0 = MG.valEnd;
    const xs = Array.from({ length: SHOW_FREE }, (_, i) => t0 + i);
    const toX = (v) => mu[0] + sd[0] * v;
    const series = [{ xs, ys: xs.map((t) => toX(z[t])), cls: "ink", label: T.truth }];
    const marks = [];
    const errs = [];
    // the error axis is in units of the test segment's standard deviation (as validSteps measures it)
    let m = 0;
    let v = 0;
    for (let i = 0; i < FREE.steps; i++) m += z[t0 + i] / FREE.steps;
    for (let i = 0; i < FREE.steps; i++) v += (z[t0 + i] - m) ** 2 / FREE.steps;
    const s = Math.sqrt(v);
    for (const [key, cls, r, name] of [["esnFree", "s1", st.esn, T.esn], ["ffnFree", "s2", st.ffn, T.ffn]]) {
      if (!r) continue;
      series.push({ xs, ys: xs.map((_, i) => toX(r.pred[i])), cls, label: T[key] });
      if (r.valid < SHOW_FREE) marks.push({ x: t0 + r.valid, cls: "now", label: name });
      errs.push({ xs, ys: xs.map((_, i) => log10(Math.abs(r.pred[i] - z[t0 + i]) / s)), cls, label: T[key] });
    }
    trajBox.replaceChildren(lineChart({ width: W, height: 230, series, marks, yFmt: fmt2, xLabel: T.timeStep, tipFmt: (v2) => fmt(v2, 3), tipTitle: (x) => `t = ${x}` }));
    errBox.replaceChildren(lineChart({
      width: W, height: 170, series: errs, hlines: [{ y: log10(FREE.thr), label: T.validLine }], marks: marks.map((m) => ({ ...m, label: null })),
      y0: -5, y1: 1, yTicks: [-4, -3, -2, -1, 0, 1], yFmt: powFmt, xLabel: T.timeStep, tipFmt: (v2) => fmt(10 ** v2, 4), tipTitle: (x) => `t = ${x}`,
    }));
    stats.replaceChildren(
      h("span", null, h("b", null, st.esn ? String(st.esn.valid) : "–"), T.validStepsOf(T.esn)),
      h("span", null, h("b", null, st.ffn ? String(st.ffn.valid) : "–"), T.validStepsOf(T.ffn)),
      st.esn ? h("span", { class: "stat-note" }, T.freeNote(st.esn.lambda, st.esn.time, st.ffn?.time, st.seed)) : null,
    );
  }

  st.esn = esnFreeRun({ seed: st.seed });
  draw();
  return { redraw: draw };
}

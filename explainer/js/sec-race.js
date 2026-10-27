// 4.1 — the race: four models, one task (Mackey–Glass, 20 steps ahead), test error against wall-clock time.
// Readout-only models are a single fit (a dot); the FFN is a curve that improves epoch by epoch.

import { h, fmt, ctl, segToggle, widthOf, visibility, ms, int } from "./util.js";
import { lineChart, legendRow, log10, powFmt } from "./charts.js";
import { mgTask, fitLinear, fitELM, fitESN, Trainer, ESN_MG, MG } from "./core.js";
import { T } from "./i18n.js";

const EPOCHS = 150;
const MODELS = [
  { key: "esn", cls: "s1" },
  { key: "elm", cls: "s4" },
  { key: "linear", cls: "muted" },
  { key: "ffn", cls: "s2" },
];

export function initRace(root) {
  const st = { L: 20, running: false };
  const task = mgTask();
  const isVisible = visibility(root);
  let runs = [];

  const startBtn = h("button", { class: "btn primary", type: "button" }, T.startRace);
  const resetBtn = h("button", { class: "btn", type: "button" }, T.reset);
  startBtn.addEventListener("click", () => !st.running && start());
  resetBtn.addEventListener("click", () => reset());
  const chartBox = h("div");
  const tableBox = h("div", { class: "scroll-x" });
  const note = h("p", { class: "sub" });

  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.windowL, segToggle([3, 10, 20].map((v) => ({ value: v, label: String(v) })), st.L, (v) => ((st.L = v), reset()), T.windowL)),
      startBtn, resetBtn),
    legendRow([["s1", T.esn], ["s4", T.elm], ["muted", T.linearAR], ["s2", T.ffn]]),
    chartBox,
    tableBox,
    note,
  );

  function reset() {
    st.running = false;
    runs = [];
    startBtn.disabled = false;
    startBtn.textContent = T.startRace;
    draw();
  }

  function start() {
    st.running = true;
    startBtn.disabled = true;
    startBtn.textContent = T.fitting;
    const seed = runs.length + 1;
    // let the button repaint before the (≈ 0.4 s) one-shot fits
    setTimeout(() => {
      const run = {
        seed,
        linear: fitLinear(task, st.L),
        elm: fitELM(task, st.L, { seed }),
        esn: fitESN(task, ESN_MG, seed),
        trainer: new Trainer(task, st.L, { hidden: [32, 32], seed, epochs: EPOCHS }),
      };
      runs.push(run);
      startBtn.textContent = T.training;
      draw();
      requestAnimationFrame(loop);
    }, 30);
  }

  function loop() {
    const run = runs.at(-1);
    if (!st.running || !run) return;
    if (isVisible()) {
      run.trainer.run(28);
      draw();
      if (run.trainer.done) {
        st.running = false;
        startBtn.disabled = false;
        startBtn.textContent = T.addSeed;
        run.ffn = run.trainer.result();
        draw();
        return;
      }
    }
    requestAnimationFrame(loop);
  }

  function draw() {
    const W = widthOf(root);
    // log time axis: a one-solve fit at 2 ms, 0.1 s and an FFN at 10 s all stay readable on any device
    const lt = (ms_) => Math.log10(Math.max(ms_ / 1000, 1e-3));
    const series = [];
    const points = [];
    const endLabels = [];
    let tMax = 2;
    let lo = Infinity;
    let hi = -Infinity;
    const seen = (v) => ((lo = Math.min(lo, v)), (hi = Math.max(hi, v)));
    for (const run of runs) {
      const hist = run.trainer.history;
      if (hist.length) tMax = Math.max(tMax, hist.at(-1).time / 1000);
      for (const m of MODELS.slice(0, 3)) tMax = Math.max(tMax, run[m.key].time / 1000);
    }
    const x0 = -3;
    const x1 = Math.ceil(Math.log10(tMax * 1.15) * 2) / 2;
    runs.forEach((run, i) => {
      const latest = i === runs.length - 1;
      const hist = run.trainer.history;
      if (hist.length) {
        series.push({
          xs: hist.map((e) => lt(e.time)), ys: hist.map((e) => log10(e.test)), cls: "s2", label: `${T.ffn} · ${T.seed} ${run.seed}`,
          thin: !latest, opacity: latest ? 1 : 0.45, noTip: !latest,
        });
        hist.forEach((e) => seen(log10(e.test)));
        // once trained, the FFN is scored at its best validation epoch: mark that point, and label it like the table
        const best = run.ffn ? hist.find((e) => e.epoch === run.ffn.bestEpoch) : null;
        if (best && latest) points.push({ x: lt(best.time), y: log10(run.ffn.err), cls: "s2", r: 6, tip: `<b>${T.ffn}</b> · ${T.bestEpoch} (${best.epoch})<br>${T.testNrmse} <span class="mono">${fmt(run.ffn.err, 5)}</span>` });
        if (latest) {
          const e = best ? run.ffn.err : hist.at(-1).test;
          endLabels.push({ y: log10(e), text: `${T.ffn} ${fmt(e, 4)}`, cls: "s2" });
        }
      }
      for (const m of MODELS.slice(0, 3)) {
        const r = run[m.key];
        const x = lt(r.time);
        const y = log10(r.err);
        seen(y);
        points.push({ x, y, cls: m.cls, r: latest ? 6 : 4, tip: `<b>${T[m.key]}</b> · ${T.seed} ${run.seed}<br>${T.testNrmse} <span class="mono">${fmt(r.err, 5)}</span><br>${T.fitTime} <span class="mono">${ms(r.time)}</span>` });
        if (latest) {
          // a guide from the moment the fit finished to the right edge: the error the FFN has to reach
          series.push({ xs: [x, x1], ys: [y, y], cls: m.cls, dash: true, thin: true, noTip: true });
          endLabels.push({ y, text: `${T[m.key]} ${fmt(r.err, 4)}`, cls: m.cls });
        }
      }
    });
    const y0 = Number.isFinite(lo) ? Math.floor(lo - 0.15) : -4;
    const y1 = Number.isFinite(hi) ? Math.max(Math.ceil(hi + 0.05), y0 + 2) : 0;
    const ticks = [];
    for (let e = y0; e <= y1; e++) ticks.push(e);
    const xTicks = [-3, -2, -1, 0, 1, 2].filter((v) => v <= x1 + 1e-9);
    chartBox.replaceChildren(lineChart({
      width: W, height: 280, series, points, endLabels, x0, x1, y0, y1, yTicks: ticks, yFmt: powFmt, xTicks,
      xFmt: (v) => timeLabel(v), xLabel: T.wallClockLog, tipFmt: (v) => fmt(10 ** v, 4), tipTitle: (x) => ms(10 ** x * 1000),
      margin: { l: 42, r: 132, t: 8, b: 34 },
    }));
    if (!runs.length) chartBox.append(h("p", { class: "sub" }, T.raceHint));

    // the table: one row per model, every seed's test error
    const rows = MODELS.map((m) => {
      const done = runs.filter((r) => (m.key === "ffn" ? r.ffn : r[m.key]));
      const errs = done.map((r) => (m.key === "ffn" ? r.ffn.err : r[m.key].err));
      const times = done.map((r) => (m.key === "ffn" ? r.ffn.time : r[m.key].time));
      const nTrained = runs.length ? (m.key === "ffn" ? runs[0].trainer.nTrained : runs[0][m.key].nTrained) : null;
      return h("tr", null,
        h("td", null, h("span", { class: `key bg-${m.cls}`, style: { marginRight: "7px" } }), T[m.key]),
        h("td", { class: "text" }, T[`${m.key}How`]),
        h("td", null, nTrained ? int(nTrained) : "–"),
        h("td", null, times.length ? ms(median(times)) : "–"),
        h("td", null, errs.length ? errs.map((e) => fmt(e, 4)).join(" · ") : "–"));
    });
    tableBox.replaceChildren(h("table", { class: "tbl" },
      h("thead", null, h("tr", null, h("th", null, T.model), h("th", { class: "text" }, T.trainedBy), h("th", null, T.trainedWeights), h("th", null, T.fitTime), h("th", null, T.testNrmsePerSeed))),
      h("tbody", null, rows)));
    note.textContent = T.raceNote(st.L, ESN_MG.N, MG.H);
  }

  draw();
  return { redraw: draw };
}

const timeLabel = (e) => ms(10 ** Math.round(e) * 1000);

function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor((s.length - 1) / 2)];
}

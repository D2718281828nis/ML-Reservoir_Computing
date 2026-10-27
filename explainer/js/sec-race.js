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
    const series = [];
    const points = [];
    const hlines = [];
    let tMax = 1.5;
    let lo = Infinity;
    let hi = -Infinity;
    const note1 = (v) => {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    };
    runs.forEach((run, i) => {
      const latest = i === runs.length - 1;
      const hist = run.trainer.history;
      if (hist.length) {
        tMax = Math.max(tMax, hist.at(-1).time / 1000);
        series.push({
          xs: hist.map((e) => e.time / 1000), ys: hist.map((e) => log10(e.test)), cls: "s2", label: `${T.ffn} · ${T.seed} ${run.seed}`,
          thin: !latest, opacity: latest ? 1 : 0.45, noTip: !latest,
        });
        hist.forEach((e) => note1(log10(e.test)));
      }
      for (const m of MODELS.slice(0, 3)) {
        const r = run[m.key];
        const y = log10(r.err);
        note1(y);
        points.push({ x: r.time / 1000, y, cls: m.cls, r: latest ? 5 : 3.5, tip: `<b>${T[m.key]}</b> · ${T.seed} ${run.seed}<br>${T.testNrmse} <span class="mono">${fmt(r.err, 5)}</span><br>${T.fitTime} <span class="mono">${ms(r.time)}</span>` });
        if (latest) hlines.push({ y }); // a dashed guide: where the FFN has to get to
      }
    });
    const y0 = Number.isFinite(lo) ? Math.floor(lo - 0.15) : -4;
    const y1 = Number.isFinite(hi) ? Math.max(Math.ceil(hi + 0.05), y0 + 2) : 0;
    const ticks = [];
    for (let e = y0; e <= y1; e++) ticks.push(e);
    chartBox.replaceChildren(lineChart({
      width: W, height: 270, series, points, hlines, x0: 0, x1: Math.ceil(tMax * 2) / 2, y0, y1, yTicks: ticks, yFmt: powFmt,
      xFmt: (v) => `${fmt(v, 1)} s`, xLabel: T.wallClock, tipFmt: (v) => fmt(10 ** v, 4), tipTitle: (x) => `${fmt(x, 2)} s`,
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

function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor((s.length - 1) / 2)];
}

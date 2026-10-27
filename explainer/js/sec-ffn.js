// 2.2 — a feed-forward network trained live: a window of Mackey–Glass in, the value 20 steps ahead out.

import { h, fmt, ctl, segToggle, slider, widthOf, visibility, robustScale, divLegend, showTip, hideTip, ms, int } from "./util.js";
import { lineChart, heatmap, legendRow, log10, powFmt } from "./charts.js";
import { mgTask, mackey, Trainer, MG } from "./core.js";
import { T } from "./i18n.js";

const SHOW = 320; // test points drawn
const EPOCHS = 150;

export function initFFN(root) {
  const st = { L: 20, hidden: 32, seed: 1, running: false };
  const task = mgTask();
  const { mu, sd } = mackey();
  const isVisible = visibility(root);
  let tr = null;

  const runBtn = h("button", { class: "btn primary", type: "button" }, T.train);
  const resetBtn = h("button", { class: "btn", type: "button" }, T.reset);
  runBtn.addEventListener("click", () => (st.running ? stop() : start()));
  resetBtn.addEventListener("click", () => reset());
  const fcBox = h("div");
  const lossBox = h("div");
  const heatBox = h("div");
  const stats = h("div", { class: "stats" });

  root.replaceChildren(
    h("div", { class: "controls" },
      slider(T.windowL, { min: 2, max: 40, step: 1, value: st.L, onInput: (v) => ((st.L = v), reset()) }),
      ctl(T.hiddenUnits, segToggle([8, 16, 32, 64].map((v) => ({ value: v, label: String(v) })), st.hidden, (v) => ((st.hidden = v), reset()), T.hiddenUnits)),
      runBtn, resetBtn),
    h("div", { class: "viz-row" },
      h("div", { class: "main-plot" },
        h("p", { class: "panel-title" }, T.ffnForecastTitle(MG.H)),
        legendRow([["ink", T.truth], ["s2", T.ffn]]),
        fcBox),
      h("div", { class: "side" },
        h("div", null, h("p", { class: "panel-title" }, T.errorPerEpoch), legendRow([["muted", T.trainSet], ["s2", T.valSet]]), lossBox),
        h("div", null, h("p", { class: "panel-title" }, T.firstLayer), heatBox))),
    stats,
  );

  function reset() {
    stop();
    st.seed++;
    tr = new Trainer(task, st.L, { hidden: [st.hidden, st.hidden], seed: st.seed, epochs: EPOCHS });
    runBtn.textContent = T.train;
    draw();
  }
  function start() {
    if (tr.done) reset();
    st.running = true;
    runBtn.textContent = T.pause;
    requestAnimationFrame(loop);
  }
  function stop() {
    st.running = false;
    runBtn.textContent = tr?.done ? T.trainAgain : T.train;
  }
  function loop() {
    if (!st.running) return;
    if (isVisible()) {
      tr.run(28);
      draw();
      if (tr.done) return stop();
    }
    requestAnimationFrame(loop);
  }

  function draw() {
    const W = widthOf(root);
    const wide = W > 760;
    const mainW = wide ? Math.round(W * 0.6) : W;
    const sideW = wide ? W - mainW - 20 : W;

    // forecast on the test months
    const P = tr.current();
    const xs = Array.from({ length: SHOW }, (_, i) => task.test[0] + i + MG.H);
    const toX = (v) => mu[0] + sd[0] * v;
    fcBox.replaceChildren(lineChart({
      width: mainW, height: 250,
      series: [
        { xs, ys: xs.map((_, i) => toX(task.Y[task.test[0] + i])), cls: "ink", label: T.truth },
        { xs, ys: xs.map((_, i) => toX(P[i])), cls: "s2", label: T.ffn },
      ],
      xLabel: T.timeStep, yFmt: (v) => fmt(v, 2), tipTitle: (x) => `t = ${x}`,
    }));

    // error curves (log scale)
    const hist = tr.history;
    const best = tr.best.epoch;
    lossBox.replaceChildren(lineChart({
      width: sideW, height: 150,
      series: hist.length ? [
        { xs: hist.map((e) => e.epoch), ys: hist.map((e) => log10(Math.sqrt(e.train))), cls: "muted", label: T.trainSet },
        { xs: hist.map((e) => e.epoch), ys: hist.map((e) => log10(Math.sqrt(e.val))), cls: "s2", label: T.valSet },
      ] : [],
      marks: best ? [{ x: best, label: T.bestEpoch }] : [],
      x0: 0, x1: EPOCHS, y0: -2.5, y1: 0.5, yTicks: [-2, -1, 0], yFmt: powFmt, xLabel: T.epoch,
      tipFmt: (v) => fmt(10 ** v, 4), tipTitle: (x) => `${T.epoch} ${x}`, margin: { l: 42, r: 10, t: 8, b: 34 },
    }));

    // first-layer weights: hidden unit × window position
    const first = tr.ffn.net.layers[0];
    const scale = robustScale(first.W);
    const heat = heatmap(first.W, first.nOut, first.nIn, scale, {
      height: wide ? 120 : 100,
      onHover: (r, c, ev) => showTip(`${T.unit} ${r + 1} · z(t${c ? `−${c}` : ""})<br><span class="mono">w = ${fmt(first.W[r * first.nIn + c], 3)}</span>`, ev),
      onLeave: hideTip,
    });
    heatBox.replaceChildren(
      h("div", { class: "heat-wrap" }, h("div", { class: "heat-y" }, h("span", null, "1"), h("span", null, String(first.nOut))), heat,
        h("div", { class: "heat-x" }, h("span", null, "z(t)"), h("span", null, `z(t−${first.nIn - 1})`))),
      h("div", { class: "sub" }, divLegend(scale)),
    );

    const last = hist.at(-1);
    const res = tr.epochs ? tr.result() : null;
    stats.replaceChildren(
      h("span", null, h("b", null, `${tr.epochs} / ${EPOCHS}`), T.epochsDone),
      h("span", null, h("b", null, int(tr.nTrained)), T.trainedWeights),
      h("span", null, h("b", null, ms(tr.time)), T.trainingTime),
      h("span", null, h("b", null, last ? fmt(last.test, 4) : "–"), T.testNrmseNow),
      res ? h("span", null, h("b", null, fmt(res.err, 4)), T.testAtBest(res.bestEpoch)) : null,
    );
  }

  reset();
  return { redraw: draw };
}

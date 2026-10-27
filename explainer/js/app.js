// Boot: formulas, navigation, then every section. Only part 5.2 needs a data file; everything else is simulated here.

import { loadJSON, placeholder, resetPalette } from "./util.js";
import { T } from "./i18n.js";
import { initFlow } from "./sec-flow.js";
import { initLoop, initGraph } from "./sec-graph.js";
import { initFFN } from "./sec-ffn.js";
import { initInside, initMemory } from "./sec-reservoir.js";
import { initRace } from "./sec-race.js";
import { initLab, initFree } from "./sec-forecast.js";
import { initCount, initLorenz } from "./sec-ngrc.js";

const ASSETS = new URL("../assets/", import.meta.url).href;
const $ = (id) => document.getElementById(id);
const sections = [];

main();

async function main() {
  setupTex();
  setupScrollSpy();
  const boot = (id, init, ...args) => {
    try {
      sections.push(init($(id), ...args));
    } catch (e) {
      fail([id], e);
    }
  };
  boot("viz-flow", initFlow);
  const flowApi = sections.at(-1)?.api;
  if (flowApi) {
    boot("viz-loop", initLoop, flowApi);
    boot("viz-graph", initGraph, flowApi);
  }
  boot("viz-ffn", initFFN);
  boot("viz-inside", initInside);
  boot("viz-memory", initMemory);
  boot("viz-race", initRace);
  boot("viz-free", initFree);
  boot("viz-count", initCount);
  boot("viz-lorenz", initLorenz);

  $("viz-lab").replaceChildren(placeholder(T.loading));
  await loadJSON(ASSETS + "series.json")
    .then((series) => boot("viz-lab", initLab, series))
    .catch((e) => fail(["viz-lab"], e));

  let lastWidth = innerWidth;
  let timer = 0;
  addEventListener("resize", () => {
    if (innerWidth === lastWidth) return;
    lastWidth = innerWidth;
    clearTimeout(timer);
    timer = setTimeout(redrawAll, 180);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    resetPalette();
    redrawAll();
  });
}

function redrawAll() {
  for (const s of sections) s?.redraw?.();
}

function fail(ids, e) {
  console.error(e);
  for (const id of ids) $(id)?.replaceChildren(placeholder(T.loadError(e.message), true));
}

// KaTeX loads async from the CDN; until then the plain-text fallback inside each .tex span stays visible.
function setupTex() {
  const render = () => {
    for (const el of document.querySelectorAll(".tex[data-tex]")) {
      try {
        window.katex.render(el.dataset.tex, el, { displayMode: !!el.closest(".formula"), throwOnError: false, strict: "ignore" });
      } catch {
        /* keep the fallback text */
      }
    }
  };
  if (window.katex) render();
  else document.getElementById("katex-js")?.addEventListener("load", render);
}

function setupScrollSpy() {
  const links = [...document.querySelectorAll("#pipeline a")];
  const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const onScroll = () => {
    const y = scrollY + 120;
    let cur = null;
    for (const t of targets) if (t.offsetTop <= y) cur = t;
    for (const a of links) a.classList.toggle("active", cur && a.getAttribute("href") === `#${cur.id}`);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

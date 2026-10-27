// DOM, formatting, color, tooltip, canvas and asset helpers shared by every section.

export const LANG = typeof document !== "undefined" && document.documentElement.lang?.startsWith("ru") ? "ru" : "en";
export const LANG_LOCALE = LANG === "ru" ? "ru-RU" : "en-US";

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  setProps(el, props);
  append(el, children);
  return el;
}

export function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  setProps(el, props);
  append(el, children);
  return el;
}

function setProps(el, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

// Number formatting with a real minus sign and no "-0.000".
export function fmt(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "–";
  if (Math.abs(v) < 0.5 * 10 ** -digits) v = 0;
  const str = v.toFixed(digits);
  return str.startsWith("-") ? "−" + str.slice(1) : str;
}
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ───────────── colors (read from CSS custom properties, so dark mode follows the page) ─────────────
let palette = null;
export function resetPalette() {
  palette = null;
}
export function colors() {
  if (palette) return palette;
  const cs = getComputedStyle(document.documentElement);
  const rgb = (name) => cs.getPropertyValue(name).trim().split(/[\s,]+/).map(Number);
  const str = (name) => cs.getPropertyValue(name).trim();
  palette = {
    ink: rgb("--c-ink"), paper: rgb("--c-paper"), neg: rgb("--c-neg"), mid: rgb("--c-mid"), pos: rgb("--c-pos"),
    s1: str("--s1"), s2: str("--s2"), s3: str("--s3"), s4: str("--s4"), muted: str("--muted"), line: str("--line"),
    lineStrong: str("--line-strong"), surface2: str("--surface-2"), surface: str("--surface"), inkCss: str("--ink"),
    ink2: str("--ink-2"), accent: str("--accent"),
  };
  return palette;
}
export const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const rgbCss = (c, a = 1) => (a === 1 ? `rgb(${c[0] | 0} ${c[1] | 0} ${c[2] | 0})` : `rgb(${c[0] | 0} ${c[1] | 0} ${c[2] | 0} / ${a})`);

// blue ← gray → red, for signed weights
export function diverging(v, scale) {
  const c = colors();
  const t = clamp(v / (scale || 1), -1, 1);
  return mix(c.mid, t < 0 ? c.neg : c.pos, Math.abs(t));
}
export function divLegend(scale, digits = 2) {
  const c = colors();
  const bar = h("span", { class: "bar", style: { background: `linear-gradient(to right, ${rgbCss(c.neg)}, ${rgbCss(c.mid)}, ${rgbCss(c.pos)})` } });
  return h("span", { class: "legend" }, fmt(-scale, digits), bar, `+${fmt(scale, digits)}`);
}

// Color scale that ignores the top 1% of magnitudes.
export function robustScale(arr, q = 0.99) {
  const a = Array.from(arr, Math.abs).filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.floor(q * a.length))] || 1e-6 : 1;
}

// ───────────── tooltip ─────────────
let tipEl = null;
export function showTip(content, ev) {
  tipEl ||= document.getElementById("tooltip");
  if (typeof content === "string") tipEl.innerHTML = content;
  else tipEl.replaceChildren(content);
  tipEl.hidden = false;
  const pad = 14;
  const r = tipEl.getBoundingClientRect();
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tipEl.style.left = `${Math.max(8, x)}px`;
  tipEl.style.top = `${Math.max(8, y)}px`;
}
export function hideTip() {
  if (tipEl) tipEl.hidden = true;
}

// ───────────── canvas ─────────────
// Size a canvas for its CSS box at the device pixel ratio; returns a context in CSS pixels.
export function fitCanvas(canvas, cssW, cssH) {
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
// content-box width (inside padding)
export function widthOf(el, fallback = 600) {
  const cs = getComputedStyle(el);
  const w = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  return Math.max(200, Math.floor(w) || fallback);
}

// ───────────── assets ─────────────
export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${url.split("/").pop()}`);
  return r.json();
}

// ───────────── seeded randomness (mulberry32 + Box–Muller) ─────────────
export function rng(seed = 1) {
  let st = seed >>> 0;
  const uniform = () => {
    st = (st + 0x6d2b79f5) | 0;
    let t = Math.imul(st ^ (st >>> 15), 1 | st);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal };
}

// Re-render only while visible; returns a function that tells whether the element is on screen.
export function visibility(el) {
  let visible = false;
  new IntersectionObserver((entries) => {
    for (const e of entries) visible = e.isIntersecting;
  }).observe(el);
  return () => visible;
}

export function segToggle(options, current, onPick, label) {
  const wrap = h("div", { class: "seg-toggle", role: "group", "aria-label": label || null });
  for (const o of options) {
    const b = h("button", { type: "button", "aria-pressed": String(o.value === current) }, o.label);
    b.addEventListener("click", () => {
      for (const x of wrap.children) x.setAttribute("aria-pressed", String(x === b));
      onPick(o.value);
    });
    wrap.append(b);
  }
  return wrap;
}

export function ctl(label, ...children) {
  return h("div", { class: "ctl" }, h("span", { class: label.length <= 2 ? "lbl sym" : "lbl" }, label), ...children);
}

export function placeholder(text, error = false) {
  return h("div", { class: `placeholder${error ? " error" : ""}` }, error ? null : h("span", { class: "spinner" }), text);
}

// Range input with a live value. onInput runs on every move; c.set(v) moves it from code.
export function slider(label, { min, max, step, value, format = (v) => String(v), onInput, wide = false }) {
  const input = h("input", { type: "range", min, max, step, value, "aria-label": label });
  const out = h("output", null, format(value));
  input.addEventListener("input", () => {
    out.textContent = format(+input.value);
    onInput(+input.value);
  });
  const c = ctl(label, input, out);
  if (wide) c.classList.add("wide");
  c.set = (v) => {
    input.value = v;
    out.textContent = format(v);
  };
  return c;
}

// Run fn at most once per animation frame, with the latest arguments.
export function perFrame(fn) {
  let pending = null;
  return (...args) => {
    const first = pending === null;
    pending = args;
    if (first) {
      requestAnimationFrame(() => {
        const a = pending;
        pending = null;
        fn(...a);
      });
    }
  };
}

const MS = LANG === "ru" ? ["мс", "с"] : ["ms", "s"];
export const ms = (t) => (t < 1 ? `${t.toFixed(2)} ${MS[0]}` : t < 1000 ? `${t.toFixed(0)} ${MS[0]}` : `${(t / 1000).toFixed(t < 1e4 ? 1 : 0)} ${MS[1]}`);
export const int = (n) => Math.round(n).toLocaleString(LANG_LOCALE);

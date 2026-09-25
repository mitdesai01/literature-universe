/* POLISY lab site. Everything is drawn from the results embedded in #polisy-data
   (findings, views, datasets, diagnostics, catalog, profiles), so new analyses appear
   without changes here. Charts use Plotly; maps are SVG drawn from the bundled state shapes. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ data
  const readJSON = (id) => { try { return JSON.parse(document.getElementById(id).textContent); } catch (e) { return null; } };
  const CFG = readJSON("polisy-config") || {};
  const DATA = readJSON("polisy-data") || {};
  const VIEWS = DATA.views || {};
  const SETS = DATA.datasets || {};
  // Tables arrive one list per column (text columns as dictionary + indices); rebuild rows once.
  for (const d of Object.values(SETS)) {
    if (!d.enc) continue;
    const cols = d.columns, n = d.n || 0;
    const vals = cols.map((c) => { const e = d.enc[c]; return e && e.dict ? e.idx.map((i) => (i < 0 ? null : e.dict[i])) : (e || []); });
    const rows = new Array(n);
    for (let i = 0; i < n; i++) { const r = {}; for (let j = 0; j < cols.length; j++) r[cols[j]] = vals[j][i] ?? null; rows[i] = r; }
    d.rows = rows;
    delete d.enc;
  }
  const LABELS = DATA.labels || {};
  const GRADES = DATA.grades || [["robust", "Robust", ""], ["suggestive", "Suggestive", ""], ["fragile", "Fragile", ""],
    ["descriptive", "Descriptive", ""], ["artifact", "Artifact", ""], ["needs data", "Needs data", ""]];
  const GRADE_NAME = Object.fromEntries(GRADES.map((g) => [g[0], g[1]]));
  const FINDINGS = (DATA.findings || []).slice().sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99) || String(a.id).localeCompare(String(b.id)));
  const POS = new Map(FINDINGS.map((f, i) => [f.id, i + 1]));
  const LEVEL_ORDER = ["occupation", "industry", "employer", "firm", "technology", "county", "metro", "state", "national"];
  const THEME_ORDER = ["Political ideology x AI", "Migration x AI", "Policy x AI", "AI adoption", "AI & innovation", "Innovation", "Migration",
    "Organizations", "Geography", "Landscape", "Measurement", "Anomalies"];
  const SOURCE_SHORT = { aioe: "AIOE", dynamic_aioe: "Dynamic AIOE", btos: "BTOS", cspp: "State policy (CSPP)", irs_migration: "IRS migration",
    patentsview: "PatentsView", vrscores: "VRscores", elections: "Election returns", geography: "CBSA geography" };
  const FONT = "Archivo, 'Helvetica Neue', Arial, sans-serif";

  // ------------------------------------------------------------------ helpers
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  }
  function svg(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
    return el;
  }
  const arr = (x) => (x == null ? [] : Array.isArray(x) ? x : [x]);
  const uniq = (xs) => [...new Set(xs)];
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const trunc = (s, n) => { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; };
  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const cap = (s) => { s = String(s || ""); return s.charAt(0).toUpperCase() + s.slice(1); };
  const themeName = (t) => String(t || "").replace(/ x /g, " × ");
  const MINUS = "−";
  const typo = (s) => String(s ?? "").replace(/(^|[\s(\[=,;:/])-(?=\d|\.\d)/g, "$1" + MINUS).replace(/ -> /g, " → ");
  function groupBy(rows, f) {
    const m = new Map();
    for (const r of rows) { const k = f(r); if (!m.has(k)) m.set(k, []); m.get(k).push(r); }
    return m;
  }
  let uid = 0;
  const nextId = (s) => `c${++uid}-${slug(s)}`;

  function fmtNum(v) {
    if (!isNum(v)) return "–";
    const a = Math.abs(v);
    let s;
    if (a >= 1000) s = Math.round(v).toLocaleString("en-US");
    else if (Number.isInteger(v)) s = String(v);
    else if (a >= 100) s = v.toFixed(0);
    else if (a >= 10) s = v.toFixed(1);
    else if (a >= 1) s = v.toFixed(2);
    else s = v.toPrecision(2);
    return s.replace("-", MINUS);
  }
  const fmtR = (r) => (isNum(r) ? (r < 0 ? MINUS : "+") + Math.abs(r).toFixed(2) : "–");

  const SHARE_RX = /(^|_)(rep_share|vote_share|rep_pred|coverage|share|female|white|black|asian|hispanic)$|^share_|_share$/;
  const INFO = new Map();
  function info(ds, col) {
    const key = ds.id + "\u0001" + col;
    if (INFO.has(key)) return INFO.get(key);
    let n = 0, num = 0, min = Infinity, max = -Infinity, len = 0;
    const seen = new Map();
    for (const r of ds.rows) {
      const v = r[col];
      if (v == null || v === "") continue;
      n++;
      if (isNum(v)) { num++; if (v < min) min = v; if (v > max) max = v; } else len += String(v).length;
      if (seen.size <= 400 || seen.has(v)) seen.set(v, (seen.get(v) || 0) + 1);
    }
    const o = {
      n, num: n > 0 && num === n, min, max, uniq: seen.size,
      cats: [...seen.entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0]),
      year: /^(year|period)$/.test(col), avgLen: n ? len / n : 0,
    };
    o.share = o.num && SHARE_RX.test(col) && !/_pp$/.test(col) && min >= 0 && max <= 1;
    INFO.set(key, o);
    return o;
  }
  function fmt(v, col, ds) {
    if (v == null || v === "") return "–";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (!isNum(v)) return String(v);
    if (/^(year|period)$/.test(col)) return String(v);
    if (ds && info(ds, col).share) return (v * 100).toFixed(1) + "%";
    return fmtNum(v);
  }
  function label(col, view) {
    if (view && view.labels && view.labels[col]) return view.labels[col];
    if (LABELS[col]) return LABELS[col];
    const s = String(col).replace(/^exp_/, "Exposure to ").replace(/_pp$/, " (points)").replace(/_/g, " ");
    return cap(s);
  }

  // ------------------------------------------------------------------ colour
  function tokens() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    return {
      paper: g("--paper"), panel: g("--panel"), sunk: g("--sunk"), ink: g("--ink"), ink2: g("--ink-2"), ink3: g("--ink-3"),
      rule: g("--rule"), accent: g("--accent"), accent2: g("--accent-2"), dem: g("--dem"), rep: g("--rep"), mid: g("--mid"),
      seqLo: g("--seq-lo"), seqHi: g("--seq-hi"), neg: g("--neg"), pos: g("--pos"), dark: g("--is-dark") === "1",
    };
  }
  function hexRGB(c) {
    c = String(c).trim();
    if (c.startsWith("rgb")) return c.match(/[\d.]+/g).slice(0, 3).map(Number);
    c = c.replace("#", "");
    if (c.length === 3) c = c.split("").map((x) => x + x).join("");
    const n = parseInt(c, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const delin = (c) => Math.round(Math.max(0, Math.min(1, c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)) * 255);
  function oklab(rgb) {
    const [r, g, b] = rgb.map(lin);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
  }
  function unlab([L, a, b]) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
      s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [delin(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s), delin(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
      delin(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)];
  }
  function mix(c1, c2, t) {
    const A = oklab(hexRGB(c1)), B = oklab(hexRGB(c2));
    const [r, g, b] = unlab(A.map((x, i) => x + (B[i] - x) * t));
    return `rgb(${r},${g},${b})`;
  }
  const luminance = (c) => { const [r, g, b] = hexRGB(c).map(lin); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
  const isPartisan = (c) => /(^|_)rep_share$|vote_share$|^rep_pred$/.test(c);
  const isPartisanChange = (c) => /^(change_pp|drift_pp|culture_gap|culture_gap_pp|demeaned_pp|resid|shift_pp|shift_agi_pp|gap_pp)$/.test(c);
  function scaleFor(col, vals, T) {
    const xs = vals.filter(isNum).sort((a, b) => a - b);
    if (!xs.length) return null;
    const q = (p) => xs[Math.round(p * (xs.length - 1))];
    let lo = q(0.02), hi = q(0.98);
    if (lo === hi) { lo = xs[0]; hi = xs[xs.length - 1]; }
    if (lo === hi) { lo -= 1; hi += 1; }
    let kind = "seq", mid = null, A, B;
    const partisan = isPartisan(col) || isPartisanChange(col);
    if (isPartisan(col)) { kind = "div"; mid = 0.5; A = T.dem; B = T.rep; }
    else if (isPartisanChange(col)) { kind = "div"; mid = 0; A = T.dem; B = T.rep; }
    else if (lo < 0 && hi > 0) { kind = "div"; mid = 0; A = T.neg; B = T.pos; }
    let f;
    if (kind === "div") {
      const span = Math.max(Math.abs(lo - mid), Math.abs(hi - mid)) || 1;
      lo = mid - span; hi = mid + span;
      f = (v) => { const t = Math.max(-1, Math.min(1, (v - mid) / span)); return t < 0 ? mix(T.mid, A, -t) : mix(T.mid, B, t); };
    } else {
      f = (v) => mix(T.seqLo, T.seqHi, 0.08 + 0.92 * Math.max(0, Math.min(1, (v - lo) / (hi - lo))));
    }
    const plotly = Array.from({ length: 11 }, (_, i) => [i / 10, f(lo + ((hi - lo) * i) / 10)]);
    return { f, lo, hi, mid, kind, plotly, partisan };
  }
  const CAT = {
    light: ["#3d6b59", "#b8862b", "#6d5a9c", "#2f8a8f", "#8a6b45", "#7f8f35", "#9b5e86", "#5f7d8c", "#4f9a6a", "#556270", "#a08a2a", "#a4788f"],
    dark: ["#7fb39c", "#e0b25a", "#a996d6", "#63c0c4", "#c4a57d", "#b5c56a", "#d49ac0", "#98b5c4", "#86cf9f", "#9aa7b4", "#d7c25b", "#d3a7bf"],
  };
  function catColors(ds, col, T) {
    const pal = T.dark ? CAT.dark : CAT.light;
    const m = new Map();
    const cats = info(ds, col).cats.map(String);
    if (/^lisa/.test(col)) {
      const fixed = { "high-high": T.rep, "low-low": T.dem, "high-low": mix(T.rep, T.mid, 0.55), "low-high": mix(T.dem, T.mid, 0.55) };
      cats.forEach((c) => m.set(c, fixed[c] || T.ink3));
      return m;
    }
    cats.forEach((c, i) => m.set(c, i < pal.length ? pal[i] : T.ink3));
    return m;
  }
  const seriesColors = (n, T) => (n === 2 ? [T.ink2, T.accent2] : (T.dark ? CAT.dark : CAT.light).slice(0, Math.max(n, 1)));

  // ------------------------------------------------------------------ grade glyphs
  function glyph(grade, big) {
    const s = svg("svg", { viewBox: "0 0 14 14", class: big ? "g g-big" : "g", "aria-hidden": "true", focusable: "false" });
    const ring = (extra) => svg("circle", Object.assign({ cx: 7, cy: 7, r: 5.6, fill: "none", stroke: "currentColor", "stroke-width": 1.5 }, extra));
    if (grade === "robust") s.append(svg("circle", { cx: 7, cy: 7, r: 6.3, fill: "currentColor" }));
    else if (grade === "suggestive") s.append(ring(), svg("path", { d: "M7 1.4 A5.6 5.6 0 0 0 7 12.6 Z", fill: "currentColor" }));
    else if (grade === "fragile") s.append(ring());
    else if (grade === "descriptive") s.append(ring(), svg("circle", { cx: 7, cy: 7, r: 2, fill: "currentColor" }));
    else if (grade === "artifact") s.append(ring(), svg("path", { d: "M3.1 10.9 L10.9 3.1", stroke: "currentColor", "stroke-width": 1.5, fill: "none" }));
    else s.append(ring({ "stroke-dasharray": "2.2 2", "stroke-width": 1.3 }));
    return s;
  }

  // ------------------------------------------------------------------ small controls
  function field(text, control, cls) {
    return h("label", { class: "field" + (cls ? " " + cls : "") }, h("span", null, text), control);
  }
  function selectEl(options, value, onChange, names, id) {
    // options: [primary, extra] lists of values
    const sel = h("select", { id: nextId(id || "select") });
    const add = (parent, vals) => vals.forEach((o) => parent.append(h("option", { value: String(o) }, names && names[o] != null ? names[o] : String(o))));
    add(sel, options[0]);
    if (options[1] && options[1].length) { const g = h("optgroup", { label: "More columns" }); add(g, options[1]); sel.append(g); }
    sel.value = String(value ?? "");
    sel.addEventListener("change", () => onChange(sel.value));
    return sel;
  }
  function select(text, options, value, onChange, view, names) {
    const nm = Object.assign({}, names || {});
    for (const o of [...options[0], ...(options[1] || [])]) if (nm[o] == null && o !== "") nm[o] = view === null ? String(o) : label(o, view);
    return field(text, selectEl(options, value, onChange, nm, text));
  }
  function searchField(text, placeholder, onInput, cls) {
    const inp = h("input", { type: "search", id: nextId(text), placeholder: placeholder || "", autocomplete: "off" });
    let t = null;
    inp.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => onInput(inp.value.trim()), 160); });
    return field(text, inp, cls);
  }
  function checkbox(text, checked, onChange) {
    const inp = h("input", { type: "checkbox", id: nextId(text) });
    inp.checked = !!checked;
    inp.addEventListener("change", () => onChange(inp.checked));
    return h("label", { class: "check" }, inp, text);
  }
  function yearSlider(years, value, onChange, play) {
    const rng = h("input", { type: "range", min: 0, max: years.length - 1, step: 1, id: nextId("year"), "aria-label": "Year" });
    rng.value = String(Math.max(0, years.indexOf(value)));
    const out = h("output", null, String(value));
    const set = (i) => { rng.value = String(i); out.textContent = String(years[i]); onChange(years[i]); };
    rng.addEventListener("input", () => set(+rng.value));
    const wrap = h("div", { class: "year-field" }, rng, out);
    if (play) {
      let timer = null;
      const btn = h("button", { class: "btn", type: "button", "aria-pressed": "false" }, "Play");
      const stop = () => { clearInterval(timer); timer = null; btn.textContent = "Play"; btn.setAttribute("aria-pressed", "false"); };
      btn.addEventListener("click", () => {
        if (timer) return stop();
        if (+rng.value >= years.length - 1) set(0);
        btn.textContent = "Pause"; btn.setAttribute("aria-pressed", "true");
        timer = setInterval(() => { const i = +rng.value + 1; if (i >= years.length || !btn.isConnected) return stop(); set(i); }, 850);
      });
      wrap.append(btn);
    }
    return h("div", { class: "field" }, h("span", null, "Year"), wrap);
  }
  let toastTimer = null;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 2400);
  }
  function copyText(text, okMsg) {
    const fallback = () => {
      const ta = h("textarea", { style: "position:fixed;left:-9999px;top:0", readonly: true });
      ta.value = text; document.body.append(ta); ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      ta.remove();
      toast(ok ? okMsg : "This browser blocked copying. Use Download CSV instead.");
    };
    try { navigator.clipboard.writeText(text).then(() => toast(okMsg), fallback); } catch (e) { fallback(); }
  }
  function csvOf(cols, rows) {
    const q = (v) => { if (v == null) return ""; const s = String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    return [cols.join(","), ...rows.map((r) => cols.map((c) => q(r[c])).join(","))].join("\n");
  }

  // ------------------------------------------------------------------ plotting core
  const hasPlotly = () => typeof window.Plotly !== "undefined";
  function layout(T, o) {
    o = Object.assign({}, o || {});
    const ax = (a) => {
      a = Object.assign({}, a || {});
      const title = Object.assign({ font: { size: 12, color: T.ink2 } }, a.title || {});
      delete a.title;
      const base = { gridcolor: T.rule, zerolinecolor: T.ink3, linecolor: T.rule, tickcolor: T.rule, automargin: true,
        tickfont: { size: 11, color: T.ink3 }, title };
      const out = Object.assign(base, a);
      for (const k of Object.keys(out)) if (out[k] === undefined) delete out[k];
      return out;
    };
    const L = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: FONT, size: 12, color: T.ink2 },
      margin: { l: 10, r: 12, t: 10, b: 10 }, hovermode: "closest", dragmode: "zoom",
      hoverlabel: { bgcolor: T.panel, bordercolor: T.rule, font: { color: T.ink, family: FONT, size: 12 }, align: "left" },
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: 11, color: T.ink2 }, bgcolor: "rgba(0,0,0,0)",
        itemsizing: "constant" },
    };
    L.xaxis = ax(o.xaxis); L.yaxis = ax(o.yaxis);
    delete o.xaxis; delete o.yaxis;
    return Object.assign(L, o);
  }
  const pconf = (v) => ({ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    toImageButtonOptions: { format: "svg", filename: "polisy-" + (v ? v.id : "chart") } });
  function noPlotly(I) {
    I.plot.replaceChildren(h("div", { class: "noplot" },
      "Charts need Plotly, which loads from cdn.jsdelivr.net. Connect to the internet or open polisy_lab_offline.html."));
  }
  function sizer(vals, maxPx = 26, minPx = 3.5) {
    let m = 0;
    for (const v of vals) if (isNum(v) && v > m) m = v;
    m = m || 1;
    return (v) => (isNum(v) && v > 0 ? minPx + (maxPx - minPx) * Math.sqrt(v / m) : minPx);
  }
  function wfit(x, y, w) {
    let sw = 0, sx = 0, sy = 0;
    for (let i = 0; i < x.length; i++) { const wi = w ? w[i] : 1; sw += wi; sx += wi * x[i]; sy += wi * y[i]; }
    const mx = sx / sw, my = sy / sw;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < x.length; i++) { const wi = w ? w[i] : 1, dx = x[i] - mx, dy = y[i] - my; sxx += wi * dx * dx; sxy += wi * dx * dy; syy += wi * dy * dy; }
    const b = sxy / sxx;
    return { b, a: my - b * mx, r: sxy / Math.sqrt(sxx * syy), n: x.length };
  }

  // Live charts: redrawn on theme change, drawn late if mounted while hidden.
  const LIVE = new Set();
  function safeDraw(I) {
    if (!I.el.isConnected) { LIVE.delete(I); return; }
    if (I.el.offsetParent === null) { I.stale = true; return; }
    I.stale = false;
    try { I.draw(); } catch (e) {
      console.error(e);
      I.plot.replaceChildren(h("div", { class: "noplot" }, "This chart could not be drawn: " + e.message));
    }
  }
  function refreshVisible(forceDraw) {
    for (const I of LIVE) {
      if (!I.el.isConnected) { LIVE.delete(I); continue; }
      if (I.el.offsetParent === null) { if (forceDraw) I.stale = true; continue; }
      if (I.stale || forceDraw) safeDraw(I);
      else if (hasPlotly()) $$(".js-plotly-plot", I.plot).forEach((p) => { try { Plotly.Plots.resize(p); } catch (e) { /* hidden */ } });
    }
  }

  const RENDER = {};
  function mountView(host, vid, opts) {
    opts = opts || {};
    const v = VIEWS[vid];
    const fig = h("figure", { class: "view", "data-view": vid });
    host.append(fig);
    if (!v) { fig.append(h("p", { class: "muted" }, "This chart is not part of this build.")); return null; }
    const ds = SETS[v.dataset] || (v.kind === "table" && SETS[vid]);
    const head = h("figcaption", { class: "view-head" }, h("span", { class: "view-title" }, v.title));
    if (opts.link) head.append(h("a", { class: "btn quiet small", href: "#v-" + vid }, "Open in Explore"));
    const controls = h("div", { class: "controls" });
    const plot = h("div", { class: "plot" });
    const live = h("span");
    const foot = h("div", { class: "view-foot" }, live, v.note ? h("span", null, typo(v.note)) : null);
    fig.append(head, controls, plot, foot);
    if (!ds || !ds.rows || !ds.rows.length) { plot.replaceChildren(h("div", { class: "noplot" }, "No rows for this chart in this build.")); return null; }
    const I = { el: fig, v, ds, controls, plot, foot, opts, stale: false, draw: () => {} };
    I.setLive = (parts) => { live.replaceChildren(...parts.filter(Boolean).map((p, i) => (i ? [" · ", p] : p)).flat()); };
    (RENDER[v.kind] || RENDER.table)(I);
    LIVE.add(I);
    safeDraw(I);
    return I;
  }

  // ---- scatter: axes, colour, size, filters, search, year and a weighted trend line
  RENDER.scatter = (I) => {
    const { v, ds, opts } = I;
    const full = !!opts.full;
    const cols = ds.columns.filter((c) => !c.startsWith("_"));
    const nums = cols.filter((c) => info(ds, c).num && !info(ds, c).year);
    const cats = cols.filter((c) => { const f = info(ds, c); return !f.num && f.uniq >= 2 && f.uniq <= 60; });
    const xs = arr(v.x).filter((c) => cols.includes(c));
    const ys = arr(v.y).filter((c) => cols.includes(c));
    if (!xs.length || !ys.length) { I.draw = () => I.plot.replaceChildren(h("div", { class: "noplot" }, "The chart's columns are missing.")); return; }
    const txt = v.text && cols.includes(v.text) ? v.text : null;
    const st = { x: xs[0], y: ys[0], color: v.color && cols.includes(v.color) ? v.color : "", size: v.size && cols.includes(v.size) ? v.size : "",
      hl: "", trend: true, f: {}, top: 0, year: null };
    const years = cols.includes("year") ? info(ds, "year").cats.filter(isNum).sort((a, b) => a - b) : [];
    if (years.length > 1) st.year = years[years.length - 1];
    const redraw = () => { safeDraw(I); };
    const lists = (primary) => (full ? [primary, nums.filter((c) => !primary.includes(c))] : [primary, []]);
    if (xs.length > 1 || full) I.controls.append(select("X axis", lists(xs), st.x, (val) => { st.x = val; redraw(); }, v));
    if (ys.length > 1 || full) I.controls.append(select("Y axis", lists(ys), st.y, (val) => { st.y = val; redraw(); }, v));
    const colorOpts = uniq([st.color, ...cats, ...(full ? nums : [])].filter(Boolean));
    if (colorOpts.length) I.controls.append(select("Colour", [["", ...colorOpts], []], st.color, (val) => { st.color = val; redraw(); }, v, { "": "None" }));
    const filt = uniq([...arr(v.filter), ...(full ? cats : [])]).filter((c) => cols.includes(c) && info(ds, c).uniq <= 60 && c !== txt);
    for (const c of filt) {
      const vals = info(ds, c).cats.map(String).sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
      I.controls.append(select(label(c, v), [["", ...vals], []], "", (val) => { st.f[c] = val; redraw(); }, null, { "": "All" }));
    }
    if (st.size) I.controls.append(select("Show", [["0", "0.5", "0.75", "0.9"], []], "0", (val) => { st.top = +val; redraw(); }, null,
      { 0: "All", 0.5: "Largest half", 0.75: "Largest quarter", 0.9: "Largest tenth" }));
    if (txt) {
      const ex = ds.rows.find((r) => r[txt]);
      I.controls.append(searchField("Find", ex ? "e.g. " + trunc(String(ex[txt]).split(/[ ,-]/)[0], 18) : "", (val) => { st.hl = val; redraw(); }));
    }
    if (years.length > 1) I.controls.append(yearSlider(years, st.year, (y) => { st.year = y; redraw(); }, false));
    I.controls.append(checkbox("Trend line", st.trend, (val) => { st.trend = val; redraw(); }));
    I.rows = () => {
      let rows = ds.rows;
      if (st.year != null) rows = rows.filter((r) => r.year === st.year);
      for (const [c, val] of Object.entries(st.f)) if (val !== "") rows = rows.filter((r) => String(r[c]) === val);
      rows = rows.filter((r) => isNum(r[st.x]) && isNum(r[st.y]));
      if (st.top > 0 && st.size) {
        const s = rows.map((r) => r[st.size]).filter(isNum).sort((a, b) => a - b);
        const thr = s[Math.floor(st.top * (s.length - 1))];
        rows = rows.filter((r) => isNum(r[st.size]) && r[st.size] >= thr);
      }
      return rows;
    };
    I.draw = () => {
      const rows = I.rows();
      if (opts.onRows) opts.onRows(rows);
      if (!hasPlotly()) return noPlotly(I);
      const T = tokens();
      const X = st.x, Y = st.y, S = st.size, C = st.color;
      const fx = info(ds, X), fy = info(ds, Y);
      const sz = S ? sizer(ds.rows.map((r) => r[S])) : () => (rows.length > 2000 ? 4.5 : 7.5);
      const type = rows.length > 2000 ? "scattergl" : "scatter";
      const hov = (r) => {
        const lines = [];
        if (txt) lines.push(`<b>${esc(trunc(r[txt], 60))}</b>`);
        lines.push(`${esc(label(X, v))}: ${esc(fmt(r[X], X, ds))}`, `${esc(label(Y, v))}: ${esc(fmt(r[Y], Y, ds))}`);
        if (S) lines.push(`${esc(label(S, v))}: ${esc(fmt(r[S], S, ds))}`);
        if (C && C !== txt) lines.push(`${esc(label(C, v))}: ${esc(fmt(r[C], C, ds))}`);
        return lines.join("<br>");
      };
      const mk = (rs, color, extra) => ({
        type, mode: "markers", x: rs.map((r) => r[X]), y: rs.map((r) => r[Y]), text: rs.map(hov), hovertemplate: "%{text}<extra></extra>",
        marker: Object.assign({ size: rs.map((r) => sz(S ? r[S] : 0)), color, opacity: 0.82, line: { width: 0.6, color: T.panel } }, extra || {}),
      });
      const traces = [];
      const catColor = C && !info(ds, C).num;
      if (C && !catColor) {
        const sc = scaleFor(C, ds.rows.map((r) => r[C]), T);
        traces.push(Object.assign(mk(rows, rows.map((r) => r[C]), {
          colorscale: sc.plotly, cmin: sc.lo, cmax: sc.hi, showscale: true,
          colorbar: { thickness: 10, len: 0.75, outlinewidth: 0, tickfont: { size: 10, color: T.ink3 }, tickformat: info(ds, C).share ? ".0%" : "",
            title: { text: esc(trunc(label(C, v), 28)), side: "right", font: { size: 11, color: T.ink2 } } },
        }), { name: label(C, v), showlegend: false }));
      } else if (catColor) {
        const colors = catColors(ds, C, T);
        const groups = groupBy(rows, (r) => (r[C] == null ? "–" : String(r[C])));
        const order = info(ds, C).cats.map(String);
        const keys = [...groups.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b));
        for (const k of keys) traces.push(Object.assign(mk(groups.get(k), colors.get(k) || T.ink3), { name: trunc(k, 34), legendgroup: k }));
      } else {
        traces.push(Object.assign(mk(rows, T.ink2), { name: label(Y, v), showlegend: false }));
      }
      const parts = [`${rows.length.toLocaleString("en-US")} shown`];
      if (st.hl && txt) {
        const q = st.hl.toLowerCase();
        const hit = rows.filter((r) => String(r[txt] ?? "").toLowerCase().includes(q));
        if (hit.length) {
          traces.push({
            type: "scatter", mode: hit.length <= 25 ? "markers+text" : "markers", x: hit.map((r) => r[X]), y: hit.map((r) => r[Y]),
            text: hit.map((r) => esc(trunc(r[txt], 32))), textposition: "top center", textfont: { size: 11, color: T.ink },
            hovertext: hit.map(hov), hovertemplate: "%{hovertext}<extra></extra>", showlegend: false, name: "found",
            marker: { size: hit.map((r) => sz(S ? r[S] : 0) + 6), color: "rgba(0,0,0,0)", line: { width: 2.2, color: T.accent2 } },
          });
        }
        parts.push(`${hit.length} match${hit.length === 1 ? "" : "es"} for “${st.hl}”`);
      }
      if (st.trend && rows.length > 3) {
        const x = rows.map((r) => r[X]), y = rows.map((r) => r[Y]);
        const w = S ? rows.map((r) => (isNum(r[S]) ? Math.max(0, r[S]) : 0)) : null;
        const fw = wfit(x, y, w), fu = wfit(x, y, null);
        let x0 = Infinity, x1 = -Infinity;
        for (const a of x) { if (a < x0) x0 = a; if (a > x1) x1 = a; }
        if (isNum(fw.b)) traces.push({ type: "scatter", mode: "lines", x: [x0, x1], y: [fw.a + fw.b * x0, fw.a + fw.b * x1],
          line: { color: T.accent2, width: 2.4 }, hoverinfo: "skip", showlegend: false, name: "trend" });
        parts.push(S ? `r = ${fmtR(fu.r)} unweighted, ${fmtR(fw.r)} weighted by ${label(S, v).toLowerCase()} (line)` : `r = ${fmtR(fu.r)}`);
      }
      const L = layout(T, {
        height: opts.height || (full ? 540 : 440), showlegend: !!catColor,
        xaxis: { title: { text: esc(label(X, v)) }, tickformat: fx.share ? ".0%" : undefined, zeroline: fx.min < 0 && fx.max > 0 },
        yaxis: { title: { text: esc(label(Y, v)) }, tickformat: fy.share ? ".0%" : undefined, zeroline: fy.min < 0 && fy.max > 0 },
      });
      Plotly.react(I.plot, traces, L, pconf(v));
      I.setLive(parts);
    };
  };

  // ---- bars: one or several series, grouped, sorted, partisan changes coloured by sign
  RENDER.bar = (I) => {
    const { v, ds, opts } = I;
    const cols = ds.columns;
    const xs = arr(v.x).filter((c) => cols.includes(c)), ys = arr(v.y).filter((c) => cols.includes(c));
    let cat, vals;
    if (xs.length === 1 && !info(ds, xs[0]).num) { cat = xs[0]; vals = ys; }
    else if (ys.length === 1 && !info(ds, ys[0]).num) { cat = ys[0]; vals = xs; }
    else { cat = xs[0]; vals = ys; }
    vals = vals.filter((c) => info(ds, c).num);
    const group = v.group && cols.includes(v.group) ? v.group : null;
    const maxLen = Math.max(...ds.rows.map((r) => String(r[cat] ?? "").length));
    const horiz = v.orientation === "h" || ds.rows.length > 12 || maxLen > 14;
    I.draw = () => {
      if (opts.onRows) opts.onRows(ds.rows);
      if (!hasPlotly()) return noPlotly(I);
      const T = tokens();
      const single = vals.length === 1 && !group;
      let rows = ds.rows.slice();
      if (single) rows.sort((a, b) => (b[vals[0]] ?? -Infinity) - (a[vals[0]] ?? -Infinity));
      const trace = (rs, col, name, color) => {
        const c = rs.map((r) => esc(trunc(r[cat], 46)));
        const val = rs.map((r) => r[col]);
        const text = rs.map((r) => `<b>${esc(r[cat])}</b><br>${esc(name)}: ${esc(fmt(r[col], col, ds))}` +
          (v.color && v.color !== col && r[v.color] != null ? `<br>${esc(label(v.color, v))}: ${esc(fmt(r[v.color], v.color, ds))}` : ""));
        return Object.assign({ type: "bar", name: trunc(name, 40), marker: { color, line: { width: 0 } }, text, hovertemplate: "%{text}<extra></extra>",
          textposition: "none" }, horiz ? { orientation: "h", y: c, x: val } : { x: c, y: val });
      };
      const traces = [];
      if (group) {
        const colors = catColors(ds, group, T);
        for (const [k, rs] of groupBy(rows, (r) => String(r[group]))) traces.push(trace(rs, vals[0], `${label(group, v)}: ${k}`, colors.get(k)));
      } else {
        const sc = seriesColors(vals.length, T);
        vals.forEach((c, i) => traces.push(trace(rows, c, label(c, v), sc[i])));
      }
      if (single) {
        const col = vals[0];
        let colors = null;
        if (v.color && cols.includes(v.color) && info(ds, v.color).num) {
          const s = scaleFor(v.color, ds.rows.map((r) => r[v.color]), T);
          colors = rows.map((r) => (isNum(r[v.color]) ? s.f(r[v.color]) : T.ink3));
        } else if (isPartisanChange(col)) colors = rows.map((r) => (r[col] < 0 ? T.dem : T.rep));
        else if (v.color && cols.includes(v.color)) { const m = catColors(ds, v.color, T); colors = rows.map((r) => m.get(String(r[v.color])) || T.ink3); }
        if (colors) traces[0].marker.color = colors;
      }
      const ncat = uniq(rows.map((r) => r[cat])).length;
      const per = group || vals.length > 1 ? 16 * Math.max(traces.length, 1) + 8 : 22;
      const shareAx = vals.every((c) => info(ds, c).share);
      const valAx = { zeroline: true, zerolinecolor: T.ink2, tickformat: shareAx ? ".0%" : undefined,
        title: { text: vals.length === 1 && !group ? esc(label(vals[0], v)) : "" } };
      const catAx = { gridcolor: "rgba(0,0,0,0)", autorange: horiz ? "reversed" : undefined, tickfont: { size: 11.5, color: T.ink2 } };
      const L = layout(T, {
        height: opts.height || (horiz ? Math.max(220, ncat * per + 80) : 380), barmode: "group", bargap: 0.25,
        showlegend: traces.length > 1, xaxis: horiz ? valAx : catAx, yaxis: horiz ? catAx : valAx,
      });
      Plotly.react(I.plot, traces, L, pconf(v));
      I.setLive([`${ncat} bars`]);
    };
  };

  // ---- coefficient (forest) plot: estimate and 95% interval by term and model
  RENDER.coef = (I) => {
    const { v, ds, opts } = I;
    const X = v.x || "coef", Y = v.y || "term", G = v.group, LO = v.lo, HI = v.hi;
    I.draw = () => {
      if (opts.onRows) opts.onRows(ds.rows);
      if (!hasPlotly()) return noPlotly(I);
      const T = tokens();
      const terms = uniq(ds.rows.map((r) => r[Y]));
      const models = G ? uniq(ds.rows.map((r) => r[G])) : [null];
      const M = models.length, band = 0.72;
      const pal = seriesColors(M, T);
      const traces = models.map((m, j) => {
        const rs = ds.rows.filter((r) => !G || r[G] === m);
        const off = M > 1 ? (j - (M - 1) / 2) * (band / M) : 0;
        const t = {
          type: "scatter", mode: "markers", name: String(m ?? "estimate"), x: rs.map((r) => r[X]), y: rs.map((r) => terms.indexOf(r[Y]) + off),
          marker: { size: 8, color: pal[j], line: { width: 1, color: T.panel } },
          text: rs.map((r) => `<b>${esc(r[Y])}</b><br>${esc(m ?? "")}<br>${fmtNum(r[X])}` + (LO && HI ? ` [${fmtNum(r[LO])}, ${fmtNum(r[HI])}]` : "") +
            (r.t != null ? `<br>t = ${fmtNum(r.t)}` : "") + (r.n != null ? ` · n = ${fmtNum(r.n)}` : "")),
          hovertemplate: "%{text}<extra></extra>",
        };
        if (LO && HI) t.error_x = { type: "data", symmetric: false, array: rs.map((r) => r[HI] - r[X]), arrayminus: rs.map((r) => r[X] - r[LO]),
          color: pal[j], thickness: 1.6, width: 0 };
        return t;
      });
      const L = layout(T, {
        height: opts.height || Math.max(220, terms.length * (M > 2 ? 44 : 34) + 90), showlegend: M > 1,
        yaxis: { tickvals: terms.map((_, i) => i), ticktext: terms.map((t) => esc(trunc(t, 48))), autorange: "reversed", zeroline: false,
          gridcolor: "rgba(0,0,0,0)", tickfont: { size: 11.5, color: T.ink2 } },
        xaxis: { title: { text: esc(v.xlabel || "Estimate with 95% interval") }, zeroline: true, zerolinecolor: T.ink2, zerolinewidth: 1.2 },
      });
      Plotly.react(I.plot, traces, L, pconf(v));
      I.setLive([`${terms.length} terms × ${M} model${M > 1 ? "s" : ""}`]);
    };
  };

  // ---- lines and stacked areas, optionally in small multiples (facet)
  function lineLike(I, area) {
    const { v, ds, opts } = I;
    const X = v.x, ys = arr(v.y).filter((c) => ds.columns.includes(c)), G = v.group && ds.columns.includes(v.group) ? v.group : null;
    const FAC = v.facet && ds.columns.includes(v.facet) ? v.facet : null;
    const TXT = v.text && ds.columns.includes(v.text) ? v.text : null;
    const facets = FAC ? uniq(ds.rows.map((r) => r[FAC])) : [null];
    const boxes = facets.map((f) => {
      const div = h("div");
      if (FAC) { const wrapper = h("div", { class: "facet" }, h("h5", null, label(f, v)), div); I.plot.append(wrapper); } else I.plot.append(div);
      return div;
    });
    if (FAC) I.plot.classList.add("facets");
    const wrapText = (s) => esc(s).replace(/(.{1,60})(\s|;|$)/g, "$1$2<br>").replace(/(<br>)+$/, "");
    I.draw = () => {
      if (opts.onRows) opts.onRows(ds.rows);
      if (!hasPlotly()) return noPlotly(I);
      const T = tokens();
      facets.forEach((f, fi) => {
        const rows = ds.rows.filter((r) => !FAC || r[FAC] === f).slice().sort((a, b) => (a[X] > b[X] ? 1 : a[X] < b[X] ? -1 : 0));
        const traces = [];
        const addLine = (rs, col, name, color) => traces.push({
          type: "scatter", mode: rs.length > 40 ? "lines" : "lines+markers", name: trunc(name, 36), x: rs.map((r) => r[X]), y: rs.map((r) => r[col]),
          line: { color, width: 2.2, shape: "linear" }, marker: { size: 5, color },
          stackgroup: area ? "one" : undefined, fillcolor: area ? color : undefined,
          text: rs.map((r) => `<b>${esc(name)}</b> · ${esc(fmt(r[X], X, ds))}<br>${esc(label(col, v))}: ${esc(fmt(r[col], col, ds))}` +
            (TXT && r[TXT] ? `<br><span style="font-size:11px">${wrapText(r[TXT])}</span>` : "")),
          hovertemplate: "%{text}<extra></extra>",
        });
        if (G) {
          const colors = catColors(ds, G, T);
          const groups = groupBy(rows, (r) => String(r[G]));
          if (v.palette === "sequential") {        // ordered groups (e.g. fifths): a light-to-dark ramp in name order
            const keys = [...groups.keys()].sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
            keys.forEach((k, i) => colors.set(k, mix(T.seqLo, T.seqHi, 0.3 + (0.7 * i) / Math.max(1, keys.length - 1))));
            for (const k of keys) addLine(groups.get(k), ys[0], k, colors.get(k));
          } else for (const [k, rs] of groups) addLine(rs, ys[0], k, colors.get(k) || T.ink3);
        } else {
          const sc = seriesColors(ys.length, T);
          ys.forEach((c, i) => addLine(rows, c, label(c, v), sc[i]));
        }
        for (const t of traces) { if (t.stackgroup === undefined) delete t.stackgroup; if (t.fillcolor === undefined) delete t.fillcolor; }
        const share = ys.every((c) => info(ds, c).share);
        const L = layout(T, {
          height: opts.height || (FAC ? 300 : 380), showlegend: traces.length > 1 && fi === 0,
          xaxis: { title: { text: FAC ? "" : esc(label(X, v)) }, tickformat: info(ds, X).year ? "d" : undefined },
          yaxis: { title: { text: FAC ? "" : esc(ys.length === 1 ? label(ys[0], v) : "") }, tickformat: share ? ".0%" : undefined,
            zeroline: true, zerolinecolor: T.ink3 },
        });
        Plotly.react(boxes[fi], traces, L, pconf(v));
      });
      I.setLive([`${ds.rows.length.toLocaleString("en-US")} points`]);
    };
  }
  RENDER.line = (I) => lineLike(I, false);
  RENDER.area = (I) => lineLike(I, true);

  // ---- network: nodes (x, y, size, colour) and edges (a, b, weight)
  RENDER.network = (I) => {
    const { v, ds, opts } = I;
    const E = SETS[v.edges];
    const idCol = v.text || "id";
    I.draw = () => {
      if (opts.onRows) opts.onRows(ds.rows);
      if (!hasPlotly()) return noPlotly(I);
      const T = tokens();
      const pos = new Map(ds.rows.map((r) => [String(r[idCol]), r]));
      const erows = E ? E.rows : [];
      const deg = new Map();
      let wmax = 0;
      for (const e of erows) { wmax = Math.max(wmax, e.weight || 1); deg.set(String(e.a), (deg.get(String(e.a)) || 0) + 1); deg.set(String(e.b), (deg.get(String(e.b)) || 0) + 1); }
      const buckets = [[[], []], [[], []], [[], []]];
      for (const e of erows) {
        const a = pos.get(String(e.a)), b = pos.get(String(e.b));
        if (!a || !b) continue;
        const s = (e.weight || 1) / (wmax || 1);
        const k = s > 0.5 ? 2 : s > 0.15 ? 1 : 0;
        buckets[k][0].push(a[v.x], b[v.x], null); buckets[k][1].push(a[v.y], b[v.y], null);
      }
      const traces = buckets.map(([x, y], k) => ({ type: "scatter", mode: "lines", x, y, hoverinfo: "skip", showlegend: false,
        line: { width: [0.6, 1.5, 3][k], color: k === 0 ? T.rule : mix(T.rule, T.ink3, k === 1 ? 0.5 : 1) } }));
      const sz = v.size ? sizer(ds.rows.map((r) => r[v.size]), 30, 5) : () => 9;
      const top = new Set(ds.rows.slice().sort((a, b) => (b[v.size] ?? 0) - (a[v.size] ?? 0)).slice(0, 24).map((r) => String(r[idCol])));
      const colors = v.color ? catColors(ds, v.color, T) : null;
      const groups = v.color ? groupBy(ds.rows, (r) => String(r[v.color])) : new Map([["nodes", ds.rows]]);
      for (const [k, rs] of groups) traces.push({
        type: "scatter", mode: "markers+text", name: k, x: rs.map((r) => r[v.x]), y: rs.map((r) => r[v.y]),
        text: rs.map((r) => (top.has(String(r[idCol])) ? esc(r[idCol]) : "")), textposition: "top center", textfont: { size: 10.5, color: T.ink2 },
        hovertext: rs.map((r) => `<b>${esc(r[idCol])}</b>` + (v.size ? `<br>${esc(label(v.size, v))}: ${esc(fmt(r[v.size], v.size, ds))}` : "") +
          `<br>links shown: ${deg.get(String(r[idCol])) || 0}` + (v.color ? `<br>${esc(label(v.color, v))}: ${esc(k)}` : "")),
        hovertemplate: "%{hovertext}<extra></extra>",
        marker: { size: rs.map((r) => sz(r[v.size])), color: colors ? colors.get(k) : T.ink2, line: { width: 1, color: T.panel } },
      });
      const hide = { showgrid: false, zeroline: false, showticklabels: false, showline: false, ticks: "" };
      Plotly.react(I.plot, traces, layout(T, { height: opts.height || 560, showlegend: !!v.color, xaxis: hide, yaxis: hide }), pconf(v));
      I.setLive([`${ds.rows.length} nodes`, `${erows.length} links`]);
    };
  };

  // ---- table
  RENDER.table = (I) => {
    const { v, ds, opts } = I;
    const t = dataTable(ds, { columns: v.columns, view: v });
    I.plot.append(t);
    I.draw = () => { if (opts.onRows) opts.onRows(ds.rows); };
  };

  // ---- maps: states (choropleth) and places (points) on one Albers USA canvas
  const TILES = { AK: [0, 0], ME: [10, 0], WI: [5, 1], VT: [9, 1], NH: [10, 1], WA: [0, 2], ID: [1, 2], MT: [2, 2], ND: [3, 2], MN: [4, 2],
    IL: [5, 2], MI: [6, 2], NY: [8, 2], MA: [9, 2], OR: [0, 3], NV: [1, 3], WY: [2, 3], SD: [3, 3], IA: [4, 3], IN: [5, 3], OH: [6, 3],
    PA: [7, 3], NJ: [8, 3], CT: [9, 3], RI: [10, 3], CA: [0, 4], UT: [1, 4], CO: [2, 4], NE: [3, 4], MO: [4, 4], KY: [5, 4], WV: [6, 4],
    VA: [7, 4], MD: [8, 4], DE: [9, 4], AZ: [1, 5], NM: [2, 5], KS: [3, 5], AR: [4, 5], TN: [5, 5], NC: [6, 5], SC: [7, 5], DC: [8, 5],
    OK: [3, 6], LA: [4, 6], MS: [5, 6], AL: [6, 6], GA: [7, 6], HI: [0, 7], TX: [3, 7], FL: [8, 7] };
  function summaryFor(ds, key) {
    // Another table keyed by the same column with one row per place (e.g. metro_summary for metro_year).
    for (const d of Object.values(SETS)) {
      if (d === ds || !d.columns.includes(key) || d.columns.includes("year")) continue;
      if (d.columns.length < 3) continue;
      const keys = new Set(d.rows.map((r) => r[key]));
      if (keys.size === d.rows.length) return d;
    }
    return null;
  }
  RENDER.choropleth = RENDER.points = (I) => {
    const { v, ds, opts } = I;
    const GEO = DATA.geo;
    const isPts = v.kind === "points";
    const cols = ds.columns;
    const measures = arr(v.value).filter((c) => cols.includes(c) && info(ds, c).num);
    if (!GEO || !measures.length) { I.draw = () => I.plot.replaceChildren(h("div", { class: "noplot" }, "No map shapes or measures in this build.")); return; }
    const YR = v.year && cols.includes(v.year) ? v.year : null;
    const key = isPts ? (v.text && cols.includes(v.text) ? v.text : cols[0]) : v.location || "state";
    const nameCol = !isPts && v.text && cols.includes(v.text) ? v.text : key;
    const S = isPts && v.size && cols.includes(v.size) ? v.size : null;
    const summ = summaryFor(ds, key);
    const st = { m: measures[0], year: null, tiles: false, sel: null };
    const byFips = new Map(GEO.states.map((s) => [s.fips, s]));
    const stateOf = (k) => (TILES[k] ? GEO.states.find((s) => s.abbr === k) : byFips.get(String(k).padStart(2, "0")));
    const yearsFor = (m) => (YR ? uniq(ds.rows.filter((r) => isNum(r[m])).map((r) => r[YR])).filter(isNum).sort((a, b) => a - b) : []);
    const details = opts.details || h("div", { class: "details", hidden: true });
    if (!opts.details) I.el.append(details);

    if (measures.length > 1) I.controls.append(select("Measure", [measures, []], st.m, (val) => { st.m = val; syncYears(); drawMap(); }, v));
    let yearBox = h("span");
    I.controls.append(yearBox);
    function syncYears() {
      const ys = yearsFor(st.m);
      if (!ys.includes(st.year)) st.year = ys.length ? ys[ys.length - 1] : null;
      const next = ys.length > 1 ? yearSlider(ys, st.year, (y) => { st.year = y; drawMap(); }, true) : h("span");
      yearBox.replaceWith(next); yearBox = next;
    }
    if (!isPts) I.controls.append(checkbox("Equal-size tiles", false, (val) => { st.tiles = val; build(); drawMap(); }));
    const findRows = () => (summ || ds).rows;
    I.controls.append(searchField("Find", isPts ? "a place" : "a state", (val) => {
      if (!val) return;
      const q = val.toLowerCase();
      const r = findRows().find((x) => String(x[nameCol] ?? x[key]).toLowerCase().includes(q) || String(x[key]).toLowerCase() === q);
      if (r) select_(String(r[key]));
    }));

    const box = h("div", { class: "mapbox" });
    const legend = h("div", { class: "maplegend" });
    const tip = h("div", { class: "tip", hidden: true });
    I.plot.append(legend, box);
    let canvas = null, shapes = new Map(), dots = new Map();

    function build() {
      box.replaceChildren(tip);
      shapes = new Map(); dots = new Map();
      const tiles = !isPts && st.tiles;
      canvas = svg("svg", { viewBox: tiles ? "0 0 682 496" : `0 0 ${GEO.width} ${GEO.height}`, role: "img", "aria-label": v.title });
      const g = svg("g");
      for (const s of GEO.states) {
        if (tiles) {
          const p = TILES[s.abbr];
          if (!p) continue;
          const t = svg("g", { class: "tile", transform: `translate(${p[0] * 62 + 2},${p[1] * 62 + 2})` });
          const rect = svg("rect", { width: 58, height: 58, rx: 3, class: "st" });
          const tx = svg("text", { x: 29, y: 34, "text-anchor": "middle" });
          tx.textContent = s.abbr;
          t.append(rect, tx);
          g.append(t);
          shapes.set(s.abbr, { el: t, fillEl: rect, text: tx, s });
        } else {
          const p = svg("path", { d: s.d, class: "st" + (isPts ? " base" : "") });
          g.append(p);
          shapes.set(s.abbr, { el: p, fillEl: p, s });
        }
      }
      canvas.append(g);
      if (isPts) {
        const gp = svg("g");
        const maxS = S ? Math.max(...ds.rows.map((r) => (isNum(r[S]) ? r[S] : 0))) || 1 : 1;
        const seen = new Map();
        for (const r of ds.rows) if (isNum(r._x) && isNum(r._y)) {
          const k = String(r[key]);
          const prev = seen.get(k);
          if (!prev || (S && (r[S] || 0) > (prev[S] || 0))) seen.set(k, r);
        }
        const list = [...seen.values()].sort((a, b) => (S ? (b[S] || 0) - (a[S] || 0) : 0));
        for (const r of list) {
          const rad = S ? 1.8 + 15 * Math.sqrt((r[S] || 0) / maxS) : 4;
          const c = svg("circle", { cx: r._x, cy: r._y, r: rad.toFixed(2), class: "pt" });
          gp.append(c);
          dots.set(String(r[key]), { el: c });
        }
        canvas.append(gp);
      }
      box.append(canvas);
      const onMove = (e) => {
        const k = e.target.closest("[data-k]") && e.target.closest("[data-k]").getAttribute("data-k");
        if (!k) { tip.hidden = true; return; }
        tip.replaceChildren(...tipContent(k));
        tip.hidden = false;
        const b = box.getBoundingClientRect();
        let x = e.clientX - b.left + 14, y = e.clientY - b.top + 14;
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        if (x + tw > b.width) x = e.clientX - b.left - tw - 10;
        if (y + th > b.height) y = e.clientY - b.top - th - 10;
        tip.style.left = Math.max(0, x) + "px"; tip.style.top = Math.max(0, y) + "px";
      };
      canvas.addEventListener("pointermove", onMove);
      canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
      canvas.addEventListener("click", (e) => { const t = e.target.closest("[data-k]"); if (t) select_(t.getAttribute("data-k")); });
      if (isPts) for (const [k, d] of dots) d.el.setAttribute("data-k", k);
      else for (const [abbr, sh] of shapes) sh.el.setAttribute("data-k", abbr);
    }
    const current = () => {
      const m = new Map();
      for (const r of ds.rows) if (!YR || r[YR] === st.year) {
        const k = isPts ? String(r[key]) : (TILES[r[key]] ? r[key] : (stateOf(r[key]) || {}).abbr);
        if (k) m.set(k, r);
      }
      return m;
    };
    let curRows = new Map();
    function tipContent(k) {
      const r = curRows.get(k);
      const nm = r ? r[nameCol] ?? k : (shapes.get(k) ? shapes.get(k).s.name : k);
      const out = [h("b", null, String(nm)), h("br")];
      out.push(`${label(st.m, v)}: ${r ? fmt(r[st.m], st.m, ds) : "no data"}`);
      if (S && r) out.push(h("br"), `${label(S, v)}: ${fmt(r[S], S, ds)}`);
      if (YR) out.push(h("br"), h("span", { class: "muted" }, String(st.year)));
      return out;
    }
    let scale = null;
    function drawMap() {
      const T = tokens();
      scale = scaleFor(st.m, ds.rows.map((r) => r[st.m]), T);
      curRows = current();
      if (opts.onRows) opts.onRows([...curRows.values()]);
      if (isPts) {
        for (const [k, d] of dots) {
          const r = curRows.get(k);
          const ok = r && isNum(r[st.m]);
          d.el.setAttribute("fill", ok ? scale.f(r[st.m]) : "none");
          d.el.setAttribute("stroke", ok ? T.panel : T.ink3);
          d.el.style.display = r ? "" : "none";
          d.el.classList.toggle("sel", st.sel === k);
        }
      } else {
        for (const [abbr, sh] of shapes) {
          const r = curRows.get(abbr);
          const ok = r && isNum(r[st.m]);
          const col = ok ? scale.f(r[st.m]) : T.sunk;
          sh.fillEl.setAttribute("fill", col);
          sh.fillEl.classList.toggle("nodata", !ok);
          if (sh.text) sh.text.setAttribute("fill", luminance(col) > 0.36 ? T.ink : "#ffffff");
          sh.el.classList.toggle("sel", st.sel === abbr);
        }
      }
      drawLegend(T);
      if (st.sel) showDetails(st.sel);
      const n = [...curRows.values()].filter((r) => isNum(r[st.m])).length;
      I.setLive([`${n} ${isPts ? "places" : "states"} with data${YR ? " in " + st.year : ""}`]);
    }
    function drawLegend(T) {
      const stops = scale.plotly.map(([p, c]) => `${c} ${Math.round(p * 100)}%`).join(", ");
      const fm = (x) => (info(ds, st.m).share ? Math.round(x * 100) + "%" : fmtNum(x));
      const ramp = h("div", { class: "ramp" },
        h("div", { class: "bar", style: `background: linear-gradient(90deg, ${stops})` }),
        h("div", { class: "ticks" }, h("span", null, fm(scale.lo)), h("span", null, fm((scale.lo + scale.hi) / 2)), h("span", null, fm(scale.hi))));
      if (scale.partisan) ramp.append(h("div", { class: "ends" }, h("span", null, "More Democratic"), h("span", null, "More Republican")));
      legend.replaceChildren(...[h("span", null, h("b", null, label(st.m, v))), ramp,
        S ? h("span", { class: "muted" }, `Circle area: ${label(S, v).toLowerCase()}`) : null,
        !isPts ? h("span", { class: "muted" }, "Grey: no data") : null].filter(Boolean));
    }
    function select_(k) {
      st.sel = k;
      if (isPts) for (const [kk, d] of dots) d.el.classList.toggle("sel", kk === k);
      else for (const [kk, sh] of shapes) sh.el.classList.toggle("sel", kk === k);
      showDetails(k);
    }
    function showDetails(k) {
      details.hidden = false;
      const T = tokens();
      const own = ds.rows.filter((r) => (isPts ? String(r[key]) === k : (TILES[r[key]] ? r[key] : (stateOf(r[key]) || {}).abbr) === k));
      const cur = curRows.get(k) || own[own.length - 1];
      const nm = cur ? cur[nameCol] ?? k : (shapes.get(k) ? shapes.get(k).s.name : k);
      const srow = summ ? summ.rows.find((r) => String(r[key]) === (cur ? String(cur[key]) : k)) : null;
      const src = srow || cur || {};
      const skip = new Set([key, nameCol, "lat", "lon", "_x", "_y", "state_fips", YR]);
      const dl = h("dl");
      for (const c of (srow ? summ.columns : ds.columns)) {
        if (skip.has(c) || src[c] == null) continue;
        dl.append(h("dt", null, label(c, v)), h("dd", null, fmt(src[c], c, srow ? summ : ds)));
      }
      const chart = h("div", { class: "detail-chart" });
      details.replaceChildren(h("span", { class: "eyebrow" }, isPts ? "Place" : "State"), h("h3", null, String(nm)), dl, chart);
      if (YR && own.length > 1 && hasPlotly()) {
        const series = own.filter((r) => isNum(r[st.m])).sort((a, b) => a[YR] - b[YR]);
        const avg = [];
        for (const [yy, rs] of groupBy(ds.rows.filter((r) => isNum(r[st.m])), (r) => r[YR])) {
          let sw = 0, s = 0;
          for (const r of rs) { const w = S && isNum(r[S]) ? r[S] : 1; sw += w; s += w * r[st.m]; }
          avg.push([yy, s / sw]);
        }
        avg.sort((a, b) => a[0] - b[0]);
        const traces = [
          { type: "scatter", mode: "lines", name: S ? "All (weighted)" : "All", x: avg.map((a) => a[0]), y: avg.map((a) => a[1]),
            line: { color: T.ink3, width: 1.6, dash: "dot" }, hovertemplate: "%{x}: %{y}<extra>all</extra>" },
          { type: "scatter", mode: "lines+markers", name: String(trunc(nm, 24)), x: series.map((r) => r[YR]), y: series.map((r) => r[st.m]),
            line: { color: T.accent2, width: 2.4 }, marker: { size: 5, color: T.accent2 }, hovertemplate: "%{x}: %{y}<extra></extra>" },
        ];
        const share = info(ds, st.m).share;
        Plotly.react(chart, traces, layout(T, { height: 220, showlegend: true, margin: { l: 6, r: 6, t: 6, b: 6 },
          legend: { orientation: "h", x: 0, y: -0.25, font: { size: 10.5, color: T.ink2 } },
          xaxis: { tickformat: "d" }, yaxis: { tickformat: share ? ".0%" : undefined, title: { text: "" } } }),
          { displayModeBar: false, responsive: true });
        chart.append(h("p", { class: "small muted" }, `${label(st.m, v)} by year`));
      }
    }
    build();
    syncYears();
    I.draw = drawMap;
  };

  // ------------------------------------------------------------------ data table
  function dataTable(ds, o) {
    o = o || {};
    const cols = (o.columns && o.columns.length ? o.columns : ds.columns).filter((c) => ds.columns.includes(c) && !c.startsWith("_"));
    const st = { sort: null, dir: -1, q: "", limit: o.limit || 150 };
    let source = o.rows || ds.rows;
    const wrap = h("div", { class: "tbl" });
    const count = h("span", { class: "count-line" });
    const filtered = () => {
      let rs = source;
      if (st.q) rs = rs.filter((r) => cols.some((c) => String(r[c] ?? "").toLowerCase().includes(st.q)));
      if (st.sort) {
        const c = st.sort, d = st.dir;
        rs = rs.slice().sort((a, b) => {
          const x = a[c], y = b[c];
          if (x == null) return 1;
          if (y == null) return -1;
          return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "en", { numeric: true })) * d;
        });
      }
      return rs;
    };
    const tools = h("div", { class: "tbl-bar" },
      h("div", { class: "filters" }, searchField("Search rows", "", (val) => { st.q = val.toLowerCase(); st.limit = o.limit || 150; render(); }), count),
      h("div", { class: "filters" },
        h("button", { class: "btn", type: "button", onclick: () => { const rs = filtered(); copyText(csvOf(cols, rs), `Copied ${rs.length.toLocaleString("en-US")} rows as CSV`); } }, "Copy CSV"),
        CFG.downloads ? h("a", { class: "btn", href: `data/${ds.id}.csv`, download: `${ds.id}.csv` }, "Download CSV") : null));
    const box = h("div", { class: "tbl-wrap" });
    const more = h("button", { class: "btn quiet", type: "button", hidden: true, onclick: () => { st.limit = Infinity; render(); } });
    wrap.append(tools, box, more);
    const numeric = new Set(cols.filter((c) => info(ds, c).num));
    const wide = new Set(cols.filter((c) => !info(ds, c).num && info(ds, c).avgLen > 38));
    function render() {
      const rs = filtered();
      const shown = rs.slice(0, st.limit);
      const thead = h("thead", null, h("tr", null, cols.map((c) => {
        const th = h("th", { class: numeric.has(c) ? "n" : null, scope: "col", title: c, tabindex: 0,
          "aria-sort": st.sort === c ? (st.dir > 0 ? "ascending" : "descending") : null }, label(c) + (st.sort === c ? (st.dir > 0 ? " ↑" : " ↓") : ""));
        const go = () => { if (st.sort === c) st.dir = -st.dir; else { st.sort = c; st.dir = numeric.has(c) ? -1 : 1; } render(); };
        th.addEventListener("click", go);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
        return th;
      })));
      const tbody = h("tbody", null, shown.map((r) => h("tr", null, cols.map((c) =>
        h("td", { class: numeric.has(c) ? "n" : wide.has(c) ? "wrap-cell" : null }, fmt(r[c], c, ds))))));
      box.replaceChildren(h("table", { class: "data" }, thead, tbody));
      count.textContent = `${rs.length.toLocaleString("en-US")} of ${source.length.toLocaleString("en-US")} rows`;
      more.hidden = rs.length <= shown.length;
      more.textContent = `Show all ${rs.length.toLocaleString("en-US")} rows`;
    }
    wrap.setRows = (rs) => { source = rs; render(); };
    render();
    return wrap;
  }

  // ------------------------------------------------------------------ findings
  const BUILD = {};
  function sectionHead(eyebrow, title, text) {
    return h("div", { class: "section-head" }, h("span", { class: "eyebrow" }, eyebrow), h("h2", null, title), text ? h("p", null, text) : null);
  }
  function sourceStatus() {
    const out = [];
    const cat = DATA.catalog || {};
    const ad = (DATA.run && DATA.run.adapters) || {};
    const names = uniq([...Object.keys(SOURCE_SHORT), ...Object.keys(cat)]);
    for (const s of names) {
      let on = null;
      if (s in ad) on = !!ad[s];
      else if (cat[s]) on = (cat[s].roles || []).some((r) => r.files > 0);
      if (on === null) continue;
      out.push({ id: s, name: SOURCE_SHORT[s] || (cat[s] && cat[s].title) || s, on });
    }
    return out;
  }
  function statChips(stats) {
    const out = [];
    for (const [k, val] of Object.entries(stats || {})) {
      if (val == null || (Array.isArray(val) && val.length > 4)) continue;   // long lists belong in the claim
      let s;
      if (Array.isArray(val)) s = "[" + val.map((x) => (isNum(x) ? fmtNum(x) : x)).join(", ") + "]";
      else if (typeof val === "object") {
        s = "first" in val && "last" in val ? `${fmtNum(val.first)} → ${fmtNum(val.last)}` :
          Object.entries(val).map(([a, b]) => `${a} ${isNum(b) ? fmtNum(b) : b}`).join(", ");
      } else s = isNum(val) ? fmtNum(val) : String(val);
      out.push(h("span", { class: "stat" }, k.replace(/_pp$/, " (pts)").replace(/_/g, " ") + " ", h("b", null, s)));
    }
    return out;
  }
  function findingEl(f) {
    const views = arr(f.views).filter((x) => VIEWS[x]);
    const art = h("article", { class: "finding", id: "f-" + f.id, "data-grade": f.strength });
    const rank = h("div", { class: "f-rank", title: "Position in the lab's priority order" }, String(POS.get(f.id)).padStart(2, "0"));
    const meta = h("div", { class: "f-meta" },
      h("span", { class: "grade" }, glyph(f.strength), GRADE_NAME[f.strength] || f.strength),
      h("span", null, themeName(f.theme)), h("span", null, levelText(f)), h("span", null, arr(f.datasets).join(" · ")));
    const next = h("div", { class: "next" });
    if (f.question) next.append(h("div", null, h("h4", null, "Research question"), h("p", null, typo(f.question))));
    if (f.next_data) next.append(h("div", null, h("h4", null, "Data to add"), h("p", null, typo(f.next_data))));
    const cav = arr(f.caveats);
    if (cav.length) next.append(h("div", null, h("h4", null, "Caveats"), h("ul", null, cav.map((c) => h("li", null, typo(c))))));
    const holder = h("div", { class: "f-views", hidden: true });
    const actions = h("div", { class: "f-actions" });
    if (views.length) {
      const btn = h("button", { class: "btn", type: "button", "aria-expanded": "false" }, `Show the evidence · ${views.length} chart${views.length > 1 ? "s" : ""}`);
      let mounted = false;
      btn.addEventListener("click", () => {
        const open = holder.hidden;
        holder.hidden = !open;
        btn.setAttribute("aria-expanded", String(open));
        btn.textContent = open ? "Hide the evidence" : `Show the evidence · ${views.length} chart${views.length > 1 ? "s" : ""}`;
        if (open && !mounted) { mounted = true; views.forEach((vid) => mountView(holder, vid, { link: true })); }
        if (open) refreshVisible(false);
      });
      actions.append(btn);
    }
    if (CFG.links !== false) actions.append(h("button", { class: "btn quiet", type: "button",
      onclick: () => copyText(location.href.split("#")[0] + "#f-" + f.id, "Link copied") }, "Copy link"));
    art.append(rank, h("div", { class: "f-body" }, meta, h("h3", null, typo(f.title)), h("p", { class: "claim" }, typo(f.claim)),
      h("div", { class: "stats" }, statChips(f.stats)), next.childNodes.length ? next : null, actions, holder));
    art.openEvidence = () => { const b = $(".f-actions .btn", art); if (b && holder.hidden) b.click(); };
    return art;
  }
  const LEVEL_ALIAS = { sector: "industry", patent: "technology", cpc: "technology", msa: "metro", cbsa: "metro" };
  const levelsOf = (f) => uniq(String(f.level || "other").split(/[\/,]/).map((x) => x.trim().toLowerCase()).filter(Boolean).map((x) => LEVEL_ALIAS[x] || x));
  const levelText = (f) => levelsOf(f).map(cap).join(" · ");
  const ordered = (vals, order) => uniq(vals).sort((a, b) => {
    const i = order.indexOf(a), j = order.indexOf(b);
    return (i < 0 ? 99 : i) - (j < 0 ? 99 : j) || String(a).localeCompare(String(b));
  });

  BUILD.findings = (root) => {
    const counts = {};
    for (const f of FINDINGS) counts[f.strength] = (counts[f.strength] || 0) + 1;
    const run = DATA.run || {};
    const src = sourceStatus();
    const head = h("header", { class: "masthead" },
      h("span", { class: "eyebrow" }, "Research theme 1"),
      h("h1", null, "Political Ideology", h("span", { class: "times" }, " × "), "AI", h("span", { class: "times" }, " × "), "Innovation"),
      h("p", { class: "lede" }, "POLISY (Politics, Organizations, Leadership, Strategy & Innovation) is an open research lab. It links VRscores measures of " +
        "workforce partisanship to open data on AI exposure, AI adoption, patenting, migration and state policy, and keeps every pattern it finds " +
        "as a graded finding, with the research question it raises and the data needed to answer it."),
      h("div", { class: "status" },
        h("span", { class: "count" }, h("b", null, String(FINDINGS.length)), "findings"),
        GRADES.filter((g) => counts[g[0]]).map((g) => h("span", { class: "count", title: g[2] }, glyph(g[0]), h("b", null, String(counts[g[0]])), g[1].toLowerCase())),
        run.finished ? h("span", { class: "muted" }, `Run ${run.finished}`) : null),
      src.length ? h("div", { class: "sources", "aria-label": "Datasets connected in this run" },
        src.map((s) => h("span", { class: "src-chip" + (s.on ? "" : " off"), title: s.on ? "Connected in this run" : "Not found in this run" },
          s.name + (s.on ? "" : " · waiting")))) : null);
    root.append(head);

    // evidence map
    const themes = ordered(FINDINGS.map((f) => f.theme), THEME_ORDER);
    const levels = ordered(FINDINGS.flatMap(levelsOf), LEVEL_ORDER);
    const tbl = h("table", { class: "matrix" });
    const setFilter = (k, val) => { FST[k] = val; const el = filterEls[k]; if (el) el.value = val; applyFilters(); listTop.scrollIntoView({ block: "start" }); };
    tbl.append(h("thead", null, h("tr", null, h("th", { scope: "col" }, "Theme / unit"),
      levels.map((l) => h("th", { scope: "col" }, h("button", { class: "hdr-btn", type: "button", onclick: () => setFilter("level", l) }, cap(l)))))));
    const tb = h("tbody");
    for (const t of themes) {
      const tr = h("tr", null, h("th", { scope: "row" }, h("button", { class: "hdr-btn", type: "button", onclick: () => setFilter("theme", t) }, themeName(t))));
      for (const l of levels) {
        const cell = h("div", { class: "cell" });
        for (const f of FINDINGS.filter((x) => x.theme === t && levelsOf(x).includes(l))) {
          cell.append(h("button", { class: "dot", type: "button", title: `${GRADE_NAME[f.strength] || f.strength}: ${f.title}`,
            "aria-label": `${GRADE_NAME[f.strength] || f.strength}: ${f.title}`,
            onclick: () => { resetFilters(); location.hash = "f-" + f.id; } }, glyph(f.strength, true)));
        }
        tr.append(h("td", null, cell));
      }
      tb.append(tr);
    }
    tbl.append(tb);
    root.append(h("section", { class: "stack" },
      sectionHead("Evidence map", "Where the lab has evidence, and where it is waiting for data",
        "Each mark is one finding, placed by research theme and unit of analysis; its shape shows how well it holds up. Select a mark to read the finding, or a row or column name to filter the list."),
      h("div", { class: "legend-grades" }, GRADES.map((g) => h("span", { title: g[2] }, glyph(g[0]), g[1]))),
      h("div", { class: "matrix-wrap" }, tbl)));

    // filters + list
    const FST = { q: "", theme: "", level: "", grade: "", ds: "", sort: "rank" };
    const filterEls = {};
    const allDs = uniq(FINDINGS.flatMap((f) => arr(f.datasets))).sort();
    const mk = (k, text, vals, names) => {
      const sel = selectEl([["", ...vals], []], "", (val) => { FST[k] = val; applyFilters(); }, Object.assign({ "": "All" }, names || {}), k);
      filterEls[k] = sel;
      return field(text, sel);
    };
    const q = h("input", { type: "search", id: "finding-search", placeholder: "Search titles, claims, questions", autocomplete: "off" });
    q.addEventListener("input", () => { FST.q = q.value.trim().toLowerCase(); applyFilters(); });
    filterEls.q = q;
    const sortSel = selectEl([["rank", "grade"], []], "rank", (val) => { FST.sort = val; order(); }, { rank: "Priority", grade: "Strength of evidence" }, "sort");
    const countLine = h("span", { class: "count-line", role: "status" });
    const reset = h("button", { class: "btn quiet", type: "button", onclick: () => { resetFilters(); } }, "Clear filters");
    const listTop = h("div", { class: "section-head" }, h("span", { class: "eyebrow" }, "Findings"),
      h("h2", null, "What the data say so far"),
      h("p", null, "Ordered by the lab's priority. Each finding states its evidence and grade, the question it opens and the data that would settle it. Open the evidence to see and filter the charts behind it."));
    const filters = h("div", { class: "filters" }, field("Search", q, "grow"),
      mk("theme", "Theme", themes, Object.fromEntries(themes.map((t) => [t, themeName(t)]))),
      mk("level", "Unit", levels, Object.fromEntries(levels.map((l) => [l, cap(l)]))),
      mk("grade", "Evidence", GRADES.map((g) => g[0]).filter((g) => counts[g]), GRADE_NAME),
      mk("ds", "Dataset", allDs), field("Order", sortSel), countLine, reset);
    const list = h("div", { class: "flist" });
    const empty = h("p", { class: "empty", hidden: true }, "No finding matches these filters.");
    const els = new Map(FINDINGS.map((f) => [f.id, findingEl(f)]));
    const gi = (g) => { const i = GRADES.findIndex((x) => x[0] === g); return i < 0 ? 99 : i; };
    function order() {
      const fs = FINDINGS.slice();
      if (FST.sort === "grade") fs.sort((a, b) => gi(a.strength) - gi(b.strength) || POS.get(a.id) - POS.get(b.id));
      list.replaceChildren(...fs.map((f) => els.get(f.id)));
    }
    function applyFilters() {
      let n = 0;
      for (const f of FINDINGS) {
        const hay = [f.title, f.claim, f.question, f.next_data, ...arr(f.caveats)].join(" ").toLowerCase();
        const ok = (!FST.theme || f.theme === FST.theme) && (!FST.level || levelsOf(f).includes(FST.level)) && (!FST.grade || f.strength === FST.grade) &&
          (!FST.ds || arr(f.datasets).includes(FST.ds)) && (!FST.q || hay.includes(FST.q));
        els.get(f.id).hidden = !ok;
        if (ok) n++;
      }
      countLine.textContent = `${n} of ${FINDINGS.length} findings`;
      empty.hidden = n > 0;
    }
    function resetFilters() {
      Object.assign(FST, { q: "", theme: "", level: "", grade: "", ds: "" });
      for (const [k, el] of Object.entries(filterEls)) el.value = "";
      applyFilters();
    }
    order();
    applyFilters();
    root.append(h("section", { class: "stack" }, listTop, filters, list, empty));
    FIND.open = (id) => {
      const el = els.get(id);
      if (!el) return;
      if (el.hidden) resetFilters();
      el.scrollIntoView({ block: "start" });
      el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
    };
  };
  const FIND = { open: () => {} };

  // ------------------------------------------------------------------ explore
  const EXPLORER = { open: () => {} };
  BUILD.explore = (root) => {
    root.append(sectionHead("Explore", "Every chart in the lab, with the rows behind it",
      "Choose the axes, colour, filters and year. Search to highlight an occupation, industry, employer or place. The table lists the rows the chart draws; sort it, search it or copy it as CSV."));
    const levelOf = (vid) => {
      const f = FINDINGS.find((x) => arr(x.views).includes(vid));
      if (f) return levelsOf(f)[0];
      const d = String(VIEWS[vid].dataset);
      return d.startsWith("occ") ? "occupation" : d.startsWith("ind") || d.startsWith("sector") ? "industry" : d.startsWith("emp") ? "employer" :
        d.startsWith("metro") ? "metro" : d.startsWith("state") || d.startsWith("landscape") ? "state" : d.startsWith("county") ? "county" : "other";
    };
    const byLevel = groupBy(Object.keys(VIEWS), levelOf);
    const used = new Set(Object.values(VIEWS).flatMap((v) => [v.dataset, v.edges]).filter(Boolean));
    const loose = Object.keys(SETS).filter((d) => !used.has(d));
    const levels = ordered([...byLevel.keys()], LEVEL_ORDER);
    const nav = h("nav", { class: "vlist", "aria-label": "Charts" });
    const links = new Map();
    for (const l of levels) {
      nav.append(h("div", null, h("h4", null, cap(l)), h("ul", null, byLevel.get(l).map((vid) => {
        const a = h("a", { href: "#v-" + vid }, VIEWS[vid].title);
        links.set(vid, a);
        return h("li", null, a);
      }))));
    }
    if (loose.length) nav.append(h("div", null, h("h4", null, "Tables without a chart"), h("ul", null, loose.map((d) => {
      const a = h("a", { href: "#v-table-" + d }, d.replace(/_/g, " "));
      links.set("table-" + d, a);
      return h("li", null, a);
    }))));
    const names = {};
    for (const l of levels) for (const vid of byLevel.get(l)) names[vid] = `${cap(l)}: ${VIEWS[vid].title}`;
    for (const d of loose) names["table-" + d] = `Table: ${d.replace(/_/g, " ")}`;
    const pick = selectEl([Object.keys(names), []], "", (val) => { location.hash = "v-" + val; }, names, "chart");
    const main = h("div", { class: "xmain" });
    root.append(h("div", { class: "explorer" }, nav, h("div", { class: "stack" }, field("Chart", pick, "vpick"), main)));
    EXPLORER.open = (vid) => {
      if (vid.startsWith("table-") && SETS[vid.slice(6)]) {
        const d = vid.slice(6);
        VIEWS[vid] = VIEWS[vid] || { id: vid, kind: "table", title: `Table: ${d.replace(/_/g, " ")}`, dataset: d };
      }
      if (!VIEWS[vid]) vid = Object.keys(VIEWS)[0];
      if (!vid) { main.replaceChildren(h("p", { class: "empty" }, "No charts in this build.")); return; }
      for (const [k, a] of links) a.setAttribute("aria-current", k === vid ? "true" : "false");
      pick.value = vid;
      main.replaceChildren();
      const v = VIEWS[vid];
      const ds = SETS[v.dataset];
      const table = ds && v.kind !== "table" ? dataTable(ds, { rows: ds.rows }) : null;
      mountView(main, vid, { full: true, onRows: (rows) => { if (table) table.setRows(rows); } });
      if (table) main.append(h("div", { class: "section-head" }, h("span", { class: "eyebrow" }, "Rows behind the chart"),
        h("p", { class: "small muted" }, `Table ${v.dataset}` + (ds.note ? ` · ${ds.note}` : ""))), table);
      const refs = FINDINGS.filter((f) => arr(f.views).includes(vid));
      if (refs.length) main.append(h("p", { class: "small muted" }, "Used by: ",
        refs.map((f, i) => [i ? ", " : "", h("a", { href: "#f-" + f.id }, typo(f.title))])));
    };
  };

  // ------------------------------------------------------------------ atlas
  const ATLAS = { open: () => {} };
  BUILD.atlas = (root) => {
    const maps = Object.values(VIEWS).filter((v) => v.kind === "choropleth" || v.kind === "points");
    root.append(sectionHead("Atlas", "Places: metros, states and counties",
      "Colour shows the measure you choose; circle area shows size. Move the year slider or press Play to watch change. Select a place for its profile and history, drawn against the average of all places."));
    if (!maps.length) { root.append(h("p", { class: "empty" }, "No maps in this build.")); return; }
    const best = (vid) => Math.min(99, ...FINDINGS.filter((f) => arr(f.views).includes(vid)).map((f) => POS.get(f.id)));
    maps.sort((a, b) => best(a.id) - best(b.id) || (a.kind === "points" ? 0 : 1) - (b.kind === "points" ? 0 : 1));
    const tabs = h("div", { class: "map-tabs", role: "group", "aria-label": "Maps" });
    const holder = h("div", { style: "min-width:0" });
    const details = h("aside", { class: "details" }, h("span", { class: "eyebrow" }, "Profile"), h("p", { class: "muted small" }, "Select a place on the map, or use Find."));
    const buttons = new Map();
    for (const v of maps) {
      const b = h("button", { class: "btn", type: "button", "aria-pressed": "false", onclick: () => { location.hash = "m-" + v.id; } }, v.title);
      buttons.set(v.id, b);
      tabs.append(b);
    }
    root.append(tabs, h("div", { class: "atlas" }, holder, details));
    ATLAS.open = (vid) => {
      if (!buttons.has(vid)) vid = maps[0].id;
      for (const [k, b] of buttons) b.setAttribute("aria-pressed", String(k === vid));
      holder.replaceChildren();
      details.replaceChildren(h("span", { class: "eyebrow" }, "Profile"), h("p", { class: "muted small" }, "Select a place on the map, or use Find."));
      mountView(holder, vid, { full: true, details });
    };
    ATLAS.open(maps[0].id);
  };

  // ------------------------------------------------------------------ data
  BUILD.data = (root) => {
    const cat = DATA.catalog || {};
    const ad = (DATA.run && DATA.run.adapters) || {};
    root.append(sectionHead("Data", "Sources, links and tables",
      "What each source is, whether this run found its files, how the sources were joined and how much of each join matched. Paths are shown as file names only."));
    const grid = h("div", { class: "src-grid" });
    for (const [id, m] of Object.entries(cat)) {
      const roles = m.roles || [];
      const on = id in ad ? !!ad[id] : roles.some((r) => r.files > 0);
      grid.append(h("div", { class: "src" + (on ? "" : " off") },
        h("span", { class: "state-pill " + (on ? "on" : "off") }, on ? "Connected" : "Waiting for files"),
        h("h3", null, m.title || id), h("p", { class: "pub" }, m.publisher || ""),
        h("dl", null,
          m.theme ? [h("dt", null, "Theme"), h("dd", null, m.theme)] : null,
          m.grain ? [h("dt", null, "Grain"), h("dd", null, m.grain)] : null,
          m.keys ? [h("dt", null, "Keys"), h("dd", { class: "mono" }, arr(m.keys).join(", "))] : null,
          m.access ? [h("dt", null, "Access"), h("dd", null, m.access)] : null,
          m.url ? [h("dt", null, "Source"), h("dd", null, h("a", { href: m.url, target: "_blank", rel: "noopener" }, m.url.replace(/^https?:\/\//, "")))] : null),
        h("div", { class: "roles" }, roles.map((r) => h("span", { class: "role" + (r.files ? "" : " miss"), title: r.paths || "no file found" },
          `${r.role}${r.files ? " · " + r.files : ""}`)))));
    }
    if (!Object.keys(cat).length) grid.append(h("p", { class: "muted" }, "Run the inventory stage to fill the source catalog."));
    root.append(grid);

    const diag = DATA.diagnostics || [];
    const dsec = h("section", { class: "stack" }, sectionHead("Linkage", "How the sources were joined",
      "Each row is one join: how much of the left table found a partner on the right, by the key named. Low shares mark where results rest on part of the data."));
    if (diag.length) {
      const t = h("table", { class: "data" }, h("thead", null, h("tr", null, ["Join", "Left → right", "Key", "Matched", "Share", "Note"].map((x) => h("th", null, x)))),
        h("tbody", null, diag.map((d) => h("tr", null,
          h("td", null, d.step), h("td", null, `${d.left} → ${d.right}`), h("td", { class: "mono" }, d.key),
          h("td", { class: "n" }, `${fmtNum(d.matched)} of ${fmtNum(d.total)} ${d.unit || ""}`),
          h("td", null, isNum(d.share) ? [h("span", { class: "share-track" }, h("i", { style: `width:${Math.round(d.share * 100)}%` })), (d.share * 100).toFixed(1) + "%"] : "–"),
          h("td", { class: "wrap-cell" }, d.note || "")))));
      dsec.append(h("div", { class: "tbl-wrap" }, t));
    } else dsec.append(h("p", { class: "muted" }, "Run the adapt and link stages to record the joins."));
    root.append(dsec);

    const used = {};
    for (const v of Object.values(VIEWS)) for (const d of [v.dataset, v.edges]) if (d) (used[d] = used[d] || []).push(v.id);
    const reg = h("table", { class: "data" }, h("thead", null, h("tr", null, ["Table", "Rows", "Columns", "Charts", ""].map((x) => h("th", null, x)))),
      h("tbody", null, Object.values(SETS).map((d) => h("tr", null, h("td", { class: "mono" }, d.id), h("td", { class: "n" }, fmtNum(d.rows.length)),
        h("td", { class: "wrap-cell" }, d.columns.filter((c) => !c.startsWith("_")).join(", ")),
        h("td", null, (used[d.id] || []).map((vid, i) => [i ? ", " : "", h("a", { href: "#v-" + vid }, vid)])),
        h("td", null, used[d.id] ? null : h("a", { href: "#v-table-" + d.id }, "Open table"))))));
    root.append(h("section", { class: "stack" }, sectionHead("Tables", "Every table behind the site", "Open any table in Explore, or take it from data/ next to index.html."),
      h("div", { class: "tbl-wrap" }, reg)));

    const prof = DATA.profiles || {};
    const psec = h("section", { class: "stack" }, sectionHead("File profiles", "What the files contain",
      "Columns, how often they are filled, how many distinct values they take and examples, from a sample of each file found."));
    for (const [k, p] of Object.entries(prof)) {
      const summ = h("summary", null, h("span", { class: "mono" }, k), " ", h("span", { class: "muted" },
        p.error ? "could not be read" : p.fields ? `${p.columns} columns · ${fmtNum(p.rows_sampled)} rows sampled · ${p.size_mb} MB` : p.kind || ""));
      const d = h("details", { class: "prof" }, summ);
      if (p.fields) d.append(h("div", { class: "tbl-wrap" }, h("table", { class: "data" },
        h("thead", null, h("tr", null, ["Column", "Filled", "Distinct", "Numeric", "Examples"].map((x) => h("th", null, x)))),
        h("tbody", null, p.fields.map((f) => h("tr", null, h("td", { class: "mono" }, f.name), h("td", { class: "n" }, Math.round(f.non_null * 100) + "%"),
          h("td", { class: "n" }, fmtNum(f.distinct)), h("td", null, f.numeric ? "yes" : ""), h("td", { class: "wrap-cell" }, arr(f.examples).join(" | "))))))));
      else if (p.error) d.append(h("p", { class: "small muted" }, p.error));
      psec.append(d);
    }
    if (!Object.keys(prof).length) psec.append(h("p", { class: "muted" }, "Run the profile stage to describe the files."));
    root.append(psec);

    const run = DATA.run || {};
    root.append(h("section", { class: "stack" }, sectionHead("Run", "This build", null),
      h("div", { class: "run" }, Object.entries(run).filter(([k]) => typeof run[k] !== "object").map(([k, v]) => h("span", null, `${k}: ${v}`)),
        run.adapters ? h("span", null, "adapters: " + Object.entries(run.adapters).map(([k, v]) => `${k} ${v ? "ok" : "not run"}`).join(", ")) : null,
        run.stages ? h("span", null, "stages: " + arr(run.stages).join(", ")) : null,
        arr(run.failed).length ? h("span", { style: "color:var(--rep)" }, "failed in this run: " + arr(run.failed).join("; ")) : null,
        h("span", null, `site: polisy_lab ${CFG.version || ""}, built ${CFG.built || ""}, Plotly ${CFG.plotly || ""}`))));
  };

  // ------------------------------------------------------------------ methods
  BUILD.methods = (root) => {
    const grades = h("section", { class: "stack" }, sectionHead("Methods", "How findings are graded", null),
      h("dl", { class: "run", style: "font-family:inherit;font-size:14px;gap:6px" }, GRADES.map((g) =>
        h("div", { style: "display:flex;gap:10px;align-items:baseline" }, glyph(g[0]), h("b", { style: "min-width:92px" }, g[1]), h("span", null, g[2])))));
    root.append(grades);
    if (!DATA.report_html) { root.append(h("p", { class: "muted" }, "Add docs/TECHNICAL_REPORT.md next to the package and rebuild to show the technical report here.")); return; }
    const art = h("article", { class: "prose", html: DATA.report_html });
    const toc = h("nav", { class: "toc", "aria-label": "Report sections" });
    $$("h2, h3", art).forEach((el) => {
      if (!el.id) el.id = "r-" + slug(el.textContent);
      const a = h("a", { href: "#methods", class: el.tagName === "H3" ? "h3" : null }, el.textContent);
      a.addEventListener("click", (e) => { e.preventDefault(); el.scrollIntoView({ block: "start" }); });
      toc.append(a);
    });
    art.addEventListener("click", (e) => {
      const a = e.target.closest("a[href^='#']");
      if (!a) return;
      const t = art.querySelector(a.getAttribute("href"));
      if (t) { e.preventDefault(); t.scrollIntoView({ block: "start" }); }
    });
    root.append(h("div", { class: "methods" }, toc, art));
  };

  // ------------------------------------------------------------------ routing, theme, boot
  const TABS = ["findings", "explore", "atlas", "data", "methods"];
  const built = {};
  let currentTab = null;
  function route() {
    const hash = decodeURIComponent(location.hash.replace(/^#/, "")) || "findings";
    let tab = hash, target = null;
    if (hash.startsWith("f-")) { tab = "findings"; target = hash.slice(2); }
    else if (hash.startsWith("v-")) { tab = "explore"; target = hash.slice(2); }
    else if (hash.startsWith("m-")) { tab = "atlas"; target = hash.slice(2); }
    if (!TABS.includes(tab)) tab = "findings";
    const switching = tab !== currentTab;
    for (const t of TABS) $("#tab-" + t).hidden = t !== tab;
    $$(".tabs a").forEach((a) => a.setAttribute("aria-current", a.dataset.tab === tab ? "page" : "false"));
    if (!built[tab]) {
      built[tab] = true;
      try { BUILD[tab]($("#tab-" + tab)); } catch (e) {
        console.error(e);
        $("#tab-" + tab).append(h("p", { class: "noplot" }, "This section could not be built: " + e.message));
      }
      if (tab === "explore" && !target) EXPLORER.open(Object.keys(VIEWS).find((k) => k === "occ-explorer") || Object.keys(VIEWS)[0] || "");
    }
    currentTab = tab;
    if (tab === "findings" && target) FIND.open(target);
    else if (tab === "explore" && target) { EXPLORER.open(target); window.scrollTo(0, 0); }
    else if (tab === "atlas" && target) ATLAS.open(target);
    else if (switching) window.scrollTo(0, 0);
    requestAnimationFrame(() => refreshVisible(false));
  }

  const root = document.documentElement;
  const hostTheme = root.getAttribute("data-theme");
  let pref = "auto";
  try { pref = localStorage.getItem("polisy-theme") || "auto"; } catch (e) { pref = "auto"; }
  const themeBtn = $("#theme-btn");
  function applyTheme() {
    if (pref === "auto") { if (hostTheme) root.setAttribute("data-theme", hostTheme); else root.removeAttribute("data-theme"); }
    else root.setAttribute("data-theme", pref);
    themeBtn.textContent = "Theme: " + pref;
  }
  themeBtn.addEventListener("click", () => {
    pref = { auto: "light", light: "dark", dark: "auto" }[pref] || "auto";
    try { localStorage.setItem("polisy-theme", pref); } catch (e) { /* storage unavailable */ }
    applyTheme();
  });
  if (pref !== "auto") applyTheme(); else themeBtn.textContent = "Theme: auto";
  let themeTimer = null;
  const onTheme = () => { clearTimeout(themeTimer); themeTimer = setTimeout(() => refreshVisible(true), 60); };
  new MutationObserver(onTheme).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", onTheme); } catch (e) { /* old browsers */ }

  const run = DATA.run || {};
  $("#foot").append(
    h("span", null, `POLISY lab · polisy_lab ${CFG.version || run.version || ""} · built ${CFG.built || ""}`),
    h("span", null, "Data: VRscores (Kagan, Frake & Hurst), AIOE (Felten, Raj & Seamans), US Census Bureau, IRS SOI, USPTO PatentsView, IPPSR CSPP. Map: US Census Bureau shapes via us-atlas."),
    CFG.offline ? h("span", null, `Offline copy with Plotly ${CFG.plotly}`) : null);

  if (!DATA.findings) {
    $("#tab-findings").append(h("p", { class: "noplot" }, "No results were embedded in this page. Rebuild it with polisy_lab.site.build_site()."));
    return;
  }
  window.addEventListener("hashchange", route);
  route();
})();

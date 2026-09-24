/* CBT Tool – front end.  Python (cbt_core.py) runs in a web worker. */
"use strict";

const ANALYST_COLORS = {
  AJ: "#4e79a7", AS: "#f28e2b", AWS: "#e15759", JPS: "#76b7b2", KJU: "#59a14f",
  KOC: "#edc948", SJB: "#b07aa1", SLK: "#ff9da7", TME: "#9c755f", VR: "#6b6ecf",
};
const METRIC_LABEL = { hull: "Ratio<sub>hull</sub>", ru: "r<sub>u,max</sub>", dgamma: "Ratio<sub>Δγ</sub>", kappa: "κ<sub>γ</sub>" };
const METRIC_TXT = { hull: "Ratio_hull", ru: "r_u,max", dgamma: "Ratio_Δγ", kappa: "κ_γ" };

const state = {
  ready: false,
  tests: [],          // {id, name, text, status: queued|running|done|error, res, series, error}
  current: null,
  cycle: 1,
  analysts: [],
  active: new Set(),
  models: null,
  queue: [],
  busy: false,
  tab: "record",
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined || !isFinite(v)) ? "–" : Number(v).toFixed(d);
const fmtG = (v) => {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
};
const fmtK = (v) => (v === null || v === undefined || !isFinite(v)) ? "–" : Number(v).toFixed(5);   // κγ: 5 decimals
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ------------------------------------------------------------------ */
/*  Math helpers (logit-normal)                                        */
/* ------------------------------------------------------------------ */
const expit = (x) => 1 / (1 + Math.exp(-x));
const logit = (p) => Math.log(p / (1 - p));
const GH = (() => { // probabilists' Gauss–Hermite via simple quadrature on ±8σ
  const n = 161, xs = [], ws = [];
  let s = 0;
  for (let i = 0; i < n; i++) { const z = -8 + 16 * i / (n - 1); const w = Math.exp(-z * z / 2); xs.push(z); ws.push(w); s += w; }
  return { xs, ws: ws.map(w => w / s) };
})();
function cbtStats(mu, sd) {
  let mean = 0;
  for (let i = 0; i < GH.xs.length; i++) mean += GH.ws[i] * expit(mu + sd * GH.xs[i]);
  return { mu_T: mu, sigma_T: sd, median: expit(mu), p16: expit(mu - sd), p84: expit(mu + sd),
           p05: expit(mu - 1.645 * sd), p95: expit(mu + 1.645 * sd), mean };
}
function combine(cyc) {
  if (!cyc.cbt) return null;
  const mus = [], sds = [];
  for (const a of state.analysts) {
    if (!state.active.has(a)) continue;
    const st = cyc.cbt.analysts[a];
    if (!st) continue;
    mus.push(st.mu_T); sds.push(st.sigma_T);
  }
  if (!mus.length) return null;
  const w = 1 / mus.length;
  const muc = mus.reduce((s, m) => s + w * m, 0);
  const within = sds.reduce((s, v) => s + w * v * v, 0);
  const between = mus.reduce((s, m) => s + w * (m - muc) ** 2, 0);
  const st = cbtStats(muc, Math.sqrt(within + between));
  st.sd_within = Math.sqrt(within); st.sd_between = Math.sqrt(between); st.n = mus.length;
  return st;
}
function behaviorLabel(m) {
  if (m < 0.2) return "clay-like";
  if (m < 0.4) return "predominantly clay-like";
  if (m <= 0.6) return "intermediate";
  if (m <= 0.8) return "predominantly sand-like";
  return "sand-like";
}

/* ------------------------------------------------------------------ */
/*  Worker                                                             */
/* ------------------------------------------------------------------ */
const worker = new Worker("js/worker.js?v=0.6");
worker.onmessage = (e) => {
  const d = e.data;
  if (d.type === "progress") {
    $("loaderText").textContent = d.step + "…";
    $("loaderBar").style.width = d.pct + "%";
  } else if (d.type === "ready") {
    state.ready = true;
    $("loaderBar").style.width = "100%";
    $("loader").classList.add("done");
    document.querySelectorAll("[data-example]").forEach(b => b.disabled = false);
    pump();
  } else if (d.type === "fatal") {
    const L = $("loader"); L.classList.add("fatal");
    $("loaderText").textContent = "Could not start the calculation engine: " + d.message;
  } else if (d.type === "result") {
    const t = state.tests.find(x => x.id === d.id);
    if (t) {
      t.res = JSON.parse(d.json);
      const L = d.buffer.byteLength / 4 / d.keys.length;
      t.series = {};
      d.keys.forEach((k, i) => { t.series[k] = new Float32Array(d.buffer, i * L * 4, L); });
      t.status = "done";
      if (!state.analysts.length) initAnalysts(t.res.analyst_names, t.res.analyst_metrics);
    }
    state.busy = false; afterRun(t); pump();
  } else if (d.type === "error") {
    const t = state.tests.find(x => x.id === d.id);
    if (t) { t.status = "error"; t.error = d.message; }
    state.busy = false; afterRun(t); pump();
  }
};
worker.postMessage({ type: "init" });

function pump() {
  if (!state.ready || state.busy) return;
  const t = state.tests.find(x => x.status === "queued");
  if (!t) return;
  state.busy = true; t.status = "running"; renderFiles();
  worker.postMessage({ type: "run", id: t.id, name: t.name, text: t.text });
}
function afterRun(t) {
  renderFiles();
  if (t && (state.current === null || state.current === t.id)) selectTest(t.id);
}

/* ------------------------------------------------------------------ */
/*  Files                                                              */
/* ------------------------------------------------------------------ */
let nextId = 1;
function addTest(name, text) {
  const t = { id: nextId++, name, text, status: "queued" };
  state.tests.push(t);
  if (state.current === null) state.current = t.id;
  renderFiles(); pump();
}
function readFiles(files) {
  [...files].forEach(f => {
    const r = new FileReader();
    r.onload = () => addTest(f.name, r.result);
    r.readAsText(f);
  });
}
$("fileInput").addEventListener("change", (e) => { readFiles(e.target.files); e.target.value = ""; });
const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => readFiles(e.dataTransfer.files));
document.querySelectorAll("[data-example]").forEach(b => b.addEventListener("click", async () => {
  const url = b.dataset.example;
  const txt = await fetch(url).then(r => r.text());
  addTest(url.split("/").pop(), txt);
}));

// "cycle 126 (γDA = 9.01%)" – every cycle reference also states its double-amplitude strain
function gdaOf(t, n) { const c = t.res.cycles[n - 1]; return c ? c.gamma_DA : null; }
function cycLabel(t, n, html = true) {
  const g = fmt(gdaOf(t, n), 2);
  return html ? `cycle ${n} (γ<sub>DA</sub> = ${g}%)` : `cycle ${n} (γDA = ${g}%)`;
}
function evalCycle(t) {
  return t.res.n_cycles;   // default: last cycle
}
function renderFiles() {
  const ul = $("fileList");
  ul.innerHTML = "";
  for (const t of state.tests) {
    const li = document.createElement("li");
    if (t.id === state.current) li.classList.add("active");
    let chip = "";
    if (t.status === "queued") chip = `<span class="chip">queued</span>`;
    else if (t.status === "running") chip = `<span class="chip">computing…</span>`;
    else if (t.status === "error") chip = `<span class="chip err">error</span>`;
    else if (!t.res.assessable) chip = `<span class="chip err" title="γDA below 6%: CBT cannot be assessed">not assessable</span>`;
    else {
      const c = t.res.cycles[evalCycle(t) - 1];
      const cb = combine(c);
      const cls = t.res.reached_target ? "chip" : "chip warn";
      chip = `<span class="${cls}" title="CBT at the last ${cycLabel(t, t.res.n_cycles, false)}${t.res.reached_target ? "" : "; γDA = 9% not reached"}">CBT ${cb ? cb.median.toFixed(2) : "–"}</span>`;
    }
    li.innerHTML = `<span class="fname" title="${esc(t.name)}">${esc(t.name)}</span>${chip}<button class="x" title="Remove">×</button>`;
    li.addEventListener("click", (e) => {
      if (e.target.classList.contains("x")) { removeTest(t.id); return; }
      selectTest(t.id);
    });
    ul.appendChild(li);
  }
}
function removeTest(id) {
  state.tests = state.tests.filter(t => t.id !== id);
  if (state.current === id) state.current = state.tests.length ? state.tests[0].id : null;
  renderFiles();
  if (state.current !== null) selectTest(state.current); else showEmpty();
}
function showEmpty() {
  $("emptyState").hidden = false; $("testView").hidden = true; $("exportCycles").disabled = true;
}

/* ------------------------------------------------------------------ */
/*  Analysts                                                           */
/* ------------------------------------------------------------------ */
function initAnalysts(names, metrics) {
  state.analysts = names; state.models = metrics;
  names.forEach(a => state.active.add(a));
  const box = $("analystList");
  box.innerHTML = names.map(a => `<label title="Metrics: ${metrics[a].map(m => METRIC_TXT[m]).join(", ")}">
      <input type="checkbox" data-analyst="${a}" checked>
      <span class="swatch" style="background:${ANALYST_COLORS[a] || "#888"}"></span>${a}</label>`).join("");
  box.querySelectorAll("input").forEach(cb => cb.addEventListener("change", () => {
    cb.checked ? state.active.add(cb.dataset.analyst) : state.active.delete(cb.dataset.analyst);
    refreshAll();
  }));
}
$("allAnalysts").onclick = () => { state.analysts.forEach(a => state.active.add(a)); syncAnalystBoxes(); refreshAll(); };
$("noAnalysts").onclick = () => { state.active.clear(); syncAnalystBoxes(); refreshAll(); };
function syncAnalystBoxes() {
  document.querySelectorAll("[data-analyst]").forEach(cb => cb.checked = state.active.has(cb.dataset.analyst));
}
function refreshAll() { renderFiles(); if (curTest()) renderTest(false); }

/* ------------------------------------------------------------------ */
/*  Test view                                                          */
/* ------------------------------------------------------------------ */
const curTest = () => state.tests.find(t => t.id === state.current && t.status === "done");

function selectTest(id) {
  state.current = id;
  renderFiles();
  const t = state.tests.find(x => x.id === id);
  if (!t) return showEmpty();
  $("emptyState").hidden = true; $("testView").hidden = false;
  if (t.status !== "done") {
    $("tName").textContent = t.name;
    $("tFacts").innerHTML = "";
    $("tBanner").innerHTML = t.status === "error"
      ? `<div class="banner err"><strong>This file could not be processed.</strong> ${esc(t.error)}</div>`
      : `<div class="banner">Computing…</div>`;
    document.querySelector(".results").style.display = "none";
    document.querySelector(".cyclebar").style.display = "none";
    $("tabs").style.display = "none";
    document.querySelectorAll(".tabpane").forEach(p => p.style.display = "none");
    $("exportCycles").disabled = true;
    return;
  }
  document.querySelector(".results").style.display = "";
  document.querySelector(".cyclebar").style.display = "";
  $("tabs").style.display = "";
  document.querySelectorAll(".tabpane").forEach(p => p.style.display = "");
  $("exportCycles").disabled = false;
  state.cycle = evalCycle(t);
  renderTest(true);
}

function renderTest(full) {
  const t = curTest(); if (!t) return;
  const r = t.res;
  $("tName").textContent = t.name;
  const gmax = Math.max(...r.cycles.map(c => c.gamma_DA || 0));
  const csr = Math.max(...r.cycles.map(c => c.CSR_cycle || 0));
  $("tFacts").innerHTML = [
    `Cycles <b>${r.n_cycles}</b>`,
    `σ′<sub>v0</sub> <b>${fmt(r.sv0, 1)} kPa</b>`,
    `CSR <b>${fmt(csr, 3)}</b>`,
    `max γ<sub>DA</sub> <b>${fmt(gmax, 2)}%</b>`,
    `γ<sub>DA</sub>=9% cycle <b>${r.target_cycle || "not reached"}</b>`,
    `N<sub>s</sub> (γ<sub>SA</sub>=3%) <b>${r.curvature.N_s ? fmt(r.curvature.N_s, 1) : "not reached"}</b>`,
  ].map(s => `<span>${s}</span>`).join("");
  const warns = [];
  const ok = r.assessable;
  $("cbtCard").style.display = ok ? "" : "none";
  document.querySelector(".results").classList.toggle("single", !ok);
  document.querySelectorAll('#tabs button[data-tab="cbt"], #tabs button[data-tab="table"]').forEach(b => b.style.display = ok ? "" : "none");
  if (!ok && (state.tab === "cbt" || state.tab === "table")) document.querySelector('#tabs button[data-tab="record"]').click();
  if (!ok) warns.push(`<b>CBT for fine-grained soils cannot be assessed.</b> The test reached a maximum double-amplitude shear strain of ${fmt(gmax, 2)}%, below the required γ<sub>DA</sub> = ${r.gamma_DA_min}%. The test record and metrics are shown for reference only.`);
  else if (!r.reached_target) warns.push(`The test did not reach γ<sub>DA</sub> = 9% (maximum ${fmt(gmax, 2)}%). Ultimate hysteretic behavior may not have developed, so CBT may be understated.`);
  if (!r.curvature.reached) warns.push("γ<sub>SA</sub> = 3% was not reached, so κ<sub>γ</sub> is evaluated in time/absolute-strain space and is not comparable with the calibration data.");
  $("tBanner").innerHTML = warns.map((w, i) => `<div class="banner ${!ok && i === 0 ? "err" : "warn"}">${w}</div>`).join("");
  const s = $("cycSlider"); s.max = r.n_cycles; $("cycInput").max = r.n_cycles;
  $("goTarget").disabled = !r.target_cycle;
  if (full) renderRecordStatic(t);
  renderCycle();
}

function setCycle(n) {
  const t = curTest(); if (!t) return;
  state.cycle = Math.max(1, Math.min(t.res.n_cycles, Math.round(n) || 1));
  renderCycle();
}
$("cycSlider").addEventListener("input", (e) => setCycle(+e.target.value));
$("cycInput").addEventListener("change", (e) => setCycle(+e.target.value));
$("prevCyc").onclick = () => setCycle(state.cycle - 1);
$("nextCyc").onclick = () => setCycle(state.cycle + 1);
$("goTarget").onclick = () => { const t = curTest(); if (t && t.res.target_cycle) setCycle(t.res.target_cycle); };
$("goLast").onclick = () => { const t = curTest(); if (t) setCycle(t.res.n_cycles); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.key === "ArrowLeft" && curTest()) { setCycle(state.cycle - 1); e.preventDefault(); }
  if (e.key === "ArrowRight" && curTest()) { setCycle(state.cycle + 1); e.preventDefault(); }
});

let rafPending = false;
function renderCycle() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; renderCycleNow(); });
}
function renderCycleNow() {
  const t = curTest(); if (!t) return;
  const c = t.res.cycles[state.cycle - 1];
  $("cycSlider").value = state.cycle; $("cycInput").value = state.cycle;
  $("cycInfo").innerHTML = `γ<sub>DA</sub> = ${fmt(c.gamma_DA, 2)}% · γ<sub>SA</sub> = ${fmt(c.gamma_SA, 2)}% · CSR = ${fmt(c.CSR_cycle, 3)}` +
    (state.cycle === t.res.target_cycle ? ` · <b>γ<sub>DA</sub> = 9% cycle</b>` : "");
  $("rCycle").innerHTML = cycLabel(t, state.cycle); $("mCycle").innerHTML = cycLabel(t, state.cycle);

  const cb = combine(c);
  if (cb) {
    $("rMedian").textContent = cb.median.toFixed(2);
    $("rLabel").textContent = behaviorLabel(cb.median);
    $("gBand").style.left = (cb.p16 * 100) + "%"; $("gBand").style.width = ((cb.p84 - cb.p16) * 100) + "%";
    $("gMed").style.left = (cb.median * 100) + "%";
    $("rKV").innerHTML = [
      ["Median", fmt(cb.median, 3)], ["Mean", fmt(cb.mean, 3)],
      ["16–84%", `${fmt(cb.p16, 2)}–${fmt(cb.p84, 2)}`], ["5–95%", `${fmt(cb.p05, 2)}–${fmt(cb.p95, 2)}`],
      ["μ<sub>T</sub>", fmt(cb.mu_T, 3)], ["σ<sub>T</sub>", fmt(cb.sigma_T, 3)],
      ["σ within", fmt(cb.sd_within, 3)], ["σ between", fmt(cb.sd_between, 3)],
      ["Analysts", `${cb.n} of ${state.analysts.length}`],
    ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
  } else {
    $("rMedian").textContent = "–"; $("rLabel").textContent = "select at least one analyst";
    $("gBand").style.width = 0; $("gMed").style.left = "-10px"; $("rKV").innerHTML = "";
  }

  const M = [
    ["r<sub>u,max</sub>", c.ru, "max. in cycle"],
    ["Ratio<sub>hull</sub>", c.hull, "A<sub>loop</sub>/A<sub>hull</sub>"],
    ["Ratio<sub>Δγ</sub>", c.dgamma, "Δγ<sub>cyc</sub>/γ<sub>FC</sub>"],
    ["κ<sub>γ</sub>", c.kappa, "whole record", fmtK],
  ];
  $("metricGrid").innerHTML = M.map(([n, v, u, f]) =>
    `<div class="metric model"><div class="mn">${n}</div><div class="mv">${(f || fmtG)(v)}</div><div class="mu">${u}</div></div>`).join("");

  if (state.tab === "record") renderRecordCycle(t, c);
  if (state.tab === "evol") renderEvol(t);
  if (state.tab === "cbt" && t.res.assessable) renderCBT(t);
  if (state.tab === "table" && t.res.assessable) renderTable(t, c);
}

/* ------------------------------------------------------------------ */
/*  Plots                                                              */
/* ------------------------------------------------------------------ */
const PCFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };
// Plot titles are intentionally omitted; axis titles and legends identify each plot.
const baseLayout = (_unused, xt, yt, extra = {}) => Object.assign({
  margin: { l: 62, r: 16, t: 16, b: 50 },
  xaxis: { title: { text: xt, font: { size: 13 } }, zeroline: false, gridcolor: "#eef0f3", linecolor: "#c9cfd8", mirror: true },
  yaxis: { title: { text: yt, font: { size: 13 } }, zeroline: false, gridcolor: "#eef0f3", linecolor: "#c9cfd8", mirror: true },
  showlegend: false, font: { family: "-apple-system, Segoe UI, Roboto, Arial, sans-serif", size: 12, color: "#1b2430" },
  paper_bgcolor: "#fff", plot_bgcolor: "#fff", hovermode: "closest",
  modebar: { orientation: "v" },
}, extra);
const ax = (which, o) => Object.assign({}, baseLayout()[which], o);
const legendBelow = (b = 110) => ({ showlegend: true, legend: { orientation: "h", x: 0, y: -0.2, yanchor: "top", font: { size: 11 } } });

// Height from the top of an element to the bottom of the window (page unscrolled),
// so every tab fits in one window at 100% zoom.
function availFrom(el) {
  const top = el.getBoundingClientRect().top + window.scrollY;
  return Math.max(340, window.innerHeight - top - 14);
}
function cellWidth(el) {
  const grid = el.parentElement;
  const cs = getComputedStyle(grid);
  const cols = grid.classList.contains("recgrid") ? (window.innerWidth <= 1100 ? 2 : 3)
    : (cs.gridTemplateColumns.split(" ").filter(Boolean).length || 1);
  const gap = parseFloat(cs.columnGap) || 0;
  return Math.floor((grid.clientWidth - gap * (cols - 1)) / cols) || 420;
}
function fixedReact(id, traces, layout, w, h) {
  const el = $(id);
  layout.autosize = false;
  layout.margin.autoexpand = false;
  layout.width = Math.round(w); layout.height = Math.round(h);
  el.style.width = layout.width + "px"; el.style.height = layout.height + "px";
  Plotly.react(id, traces, layout, Object.assign({}, PCFG, { responsive: false }));
}
// Square plotting area; two rows of record plots must fit in the window.
const REC_MARGIN = { l: 58, r: 14, t: 50, b: 46 };          // row 1 (legend above the plot)
const REC_MARGIN_2 = { l: 58, r: 14, t: 50, b: 46 };       // row 2
const REC_MARGIN_RL = (r) => ({ l: 58, r, t: 50, b: 46 });  // plots with the legend to the right
const REC_VERT = REC_MARGIN.t + REC_MARGIN.b + REC_MARGIN_2.t + REC_MARGIN_2.b;
function squareReact(id, traces, layout) {
  const el = $(id), m = layout.margin;
  // one common side length so both rows line up and fit in the window
  const side = Math.max(180, Math.min(cellWidth(el) - m.l - m.r, (availFrom(el.parentElement) - 10 - REC_VERT) / 2));
  fixedReact(id, traces, layout, side + m.l + m.r, side + m.t + m.b);
}
const legendRight = { showlegend: true, legend: { orientation: "v", x: 1.03, xanchor: "left", y: 1, yanchor: "top", font: { size: 10.5 } } };
const legendBottom = { showlegend: true, legend: { orientation: "h", x: -0.2, xanchor: "left", y: -0.24, yanchor: "top", font: { size: 10.5 } } };
const legendTop = { showlegend: true, legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: 10.5 } } };
const legendInside = (pos = "tl") => ({ showlegend: true, legend: {
  x: pos[1] === "l" ? 0.02 : 0.98, xanchor: pos[1] === "l" ? "left" : "right",
  y: pos[0] === "t" ? 0.98 : 0.02, yanchor: pos[0] === "t" ? "top" : "bottom",
  bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#dfe3e9", borderwidth: 1, font: { size: 10.5 } } });
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (curTest()) renderCycleNow(); }, 150);
});

function stride(arr, maxPts = 30000) {
  const k = Math.max(1, Math.ceil(arr.length / maxPts));
  if (k === 1) return Array.from(arr);
  const out = []; for (let i = 0; i < arr.length; i += k) out.push(arr[i]); return out;
}
const slice = (a, l1, l2) => Array.from(a.subarray(l1, l2));
// WebGL traces are drawn above SVG traces, so every trace in the record plots uses WebGL
const bg = (x, y) => ({ x, y, type: "scattergl", mode: "lines", line: { color: "#b9c0ca", width: 1 }, hoverinfo: "skip", showlegend: false, name: "Full record" });
const fg = (x, y, name, color = "#111", width = 2) => ({ x, y, type: "scattergl", mode: "lines", name, line: { color, width } });

// background traces are cached per test
function renderRecordStatic(t) {
  const S = t.series;
  t._bg = {
    strain: stride(S.strain), stress: stride(S.stress), time: stride(S.time), ru: stride(S.ru),
    svn: stride(S.sv_norm), csr: stride(S.csr), nns: stride(S.NNs), ggs: stride(S.gg_s),
  };
}
const RU_REFS = [
  { ru: 1.0, color: "#e08a1e", dash: "dash" },
  { ru: 0.95, color: "#d62728", dash: "dot" },
  { ru: 0.9, color: "#8e44ad", dash: "dashdot" },
];
const ruName = (r) => `r<sub>u</sub> = ${r.ru.toFixed(r.ru === 1 ? 1 : 2)}`;
// reference lines drawn as traces so they appear in the legend
function ruRefTraces(orient, lo, hi, useRuValue) {
  return RU_REFS.map(r => {
    const v = useRuValue ? r.ru : 1 - r.ru;
    const tr = { type: "scattergl", mode: "lines", name: ruName(r), hoverinfo: "name", line: { color: r.color, width: 1.8, dash: r.dash } };
    if (orient === "v") { tr.x = [v, v]; tr.y = [lo, hi]; } else { tr.x = [lo, hi]; tr.y = [v, v]; }
    return tr;
  });
}

function renderRecordCycle(t, c) {
  const S = t.series, B = t._bg, r = t.res;
  const l1 = c.l1, l2 = c.l2;
  const reached = r.curvature.reached;
  const cyc = `Cycle ${c.cycle} (γ<sub>DA</sub> = ${fmt(c.gamma_DA, 2)}%)`;
  const tmax = S.time[S.time.length - 1];
  const withLegend = (m, extra = {}, pos = "tl") => Object.assign({ margin: Object.assign({}, m) },
    pos === "top" ? legendTop : pos === "bottom" ? legendBottom : pos === "right" ? legendRight : pos ? legendInside(pos) : { showlegend: false }, extra);

  // 1. Stress-strain loop with convex hull
  const hx = c.hull_idx.map(i => S.strain[i]), hy = c.hull_idx.map(i => S.stress[i]);
  if (hx.length) { hx.push(hx[0]); hy.push(hy[0]); }
  squareReact("pLoop", [
    bg(B.strain, B.stress),
    { x: hx, y: hy, type: "scattergl", mode: "lines", name: `Convex hull (Ratio<sub>hull</sub> = ${fmt(c.hull, 3)})`,
      line: { color: "#e08a1e", width: 3.5 }, fill: "toself", fillcolor: "rgba(224,138,30,0.08)" },
    fg(slice(S.strain, l1, l2), slice(S.stress, l1, l2), cyc),
  ], baseLayout(null, "Shear strain, γ (%)", "Shear stress, τ (kPa)", withLegend(REC_MARGIN, {}, "top")));

  // 2. Stress path
  squareReact("pPath", [bg(B.svn, B.csr), fg(slice(S.sv_norm, l1, l2), slice(S.csr, l1, l2), cyc)],
    baseLayout(null, "σ′<sub>v</sub> / σ′<sub>v0</sub>", "τ / σ′<sub>v0</sub>", withLegend(REC_MARGIN, {
      xaxis: ax("xaxis", { range: [-0.05, 1.1], title: { text: "σ′<sub>v</sub> / σ′<sub>v0</sub>", font: { size: 13 } } }) }, null)));

  // 3. Shear strain (x) vs time (y)
  squareReact("pStrainT", [bg(B.strain, B.time), fg(slice(S.strain, l1, l2), slice(S.time, l1, l2), cyc)],
    baseLayout(null, "Shear strain, γ (%)", "Time (s)", withLegend(REC_MARGIN_2, {
      yaxis: ax("yaxis", { range: [0, tmax * 1.02], title: { text: "Time (s)", font: { size: 13 } } }) }, null)));

  // 4. σ'v/σ'v0 = 1 - r_u (bottom x) and r_u (top x, reversed) vs time
  const XR = [-0.05, 1.1];
  squareReact("pRuT", [
    bg(B.svn, B.time), fg(slice(S.sv_norm, l1, l2), slice(S.time, l1, l2), cyc),
    ...ruRefTraces("v", 0, tmax * 1.02, false),
    { x: [1 - XR[0], 1 - XR[1]], y: [0, 0], xaxis: "x2", type: "scatter", mode: "markers", marker: { opacity: 0 }, hoverinfo: "skip", showlegend: false },
  ], baseLayout(null, "", "Time (s)", withLegend(REC_MARGIN_RL(185), {
    xaxis: ax("xaxis", { range: XR, mirror: false, title: { text: "σ′<sub>v</sub> / σ′<sub>v0</sub> = 1 − r<sub>u</sub>", font: { size: 13 } } }),
    xaxis2: { overlaying: "x", side: "top", range: [1 - XR[0], 1 - XR[1]], title: { text: "r<sub>u</sub>", font: { size: 13 }, standoff: 4 },
              linecolor: "#c9cfd8", showgrid: false, zeroline: false, ticks: "outside" },
    yaxis: ax("yaxis", { range: [0, tmax * 1.02], title: { text: "Time (s)", font: { size: 13 } } }),
  }, "right")));

  // 5. Strain accumulation + kappa points
  const cv = r.curvature;
  const xl = reached ? "N / N<sub>s</sub>" : "Time (s)";
  const yl = reached ? "γ<sub>SA</sub> / 3%" : "max |γ| (%)";
  const e = Math.max(l1, l2 - 1);
  squareReact("pAcc", [
    { x: B.nns, y: B.ggs, type: "scattergl", mode: "lines", line: { color: "#111", width: 1.5 }, name: "Record", hoverinfo: "skip" },
    { x: [S.NNs[l1], S.NNs[e]], y: [S.gg_s[l1], S.gg_s[e]], type: "scattergl", mode: "markers", marker: { color: "#2468a8", size: 10, symbol: "diamond" }, name: cyc },
    { x: cv.x_s, y: cv.y_s, type: "scattergl", mode: "markers", marker: { color: "#d62728", size: 12, line: { color: "#111", width: 1 } }, name: `κ<sub>γ</sub> points (κ<sub>γ</sub> = ${fmtK(cv.kappa)})` },
  ], baseLayout(null, xl, yl, withLegend(REC_MARGIN_RL(185), {
    xaxis: ax("xaxis", { rangemode: "tozero", title: { text: xl, font: { size: 13 } } }),
    yaxis: ax("yaxis", { rangemode: "tozero", title: { text: yl, font: { size: 13 } } }),
  }, "right")));
}

// vertical marker lines as legend traces (selected cycle and γDA = 9% cycle)
function markerLines(t, xSel, xTarget, lo, hi, br = false) {
  const sep = br ? "<br>" : " ";
  const tr = [{ x: [xSel, xSel], y: [lo, hi], type: "scatter", mode: "lines", name: `Selected${sep}${cycLabel(t, state.cycle)}`,
    line: { color: "#111", width: 1.5 }, hoverinfo: "name" }];
  if (t.res.target_cycle) tr.push({ x: [xTarget, xTarget], y: [lo, hi], type: "scatter", mode: "lines",
    name: `First γ<sub>DA</sub> ≥ 9%:${sep}${cycLabel(t, t.res.target_cycle)}`, line: { color: "#2e7d4f", width: 1.8, dash: "dash" }, hoverinfo: "name" });
  return tr;
}
function paddedRange(vals) {
  const v = vals.filter(x => x !== null && x !== undefined && isFinite(x));
  let lo = Math.min(...v), hi = Math.max(...v);
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const p = (hi - lo) * 0.06 || 0.05;
  return [lo - p, hi + p];
}
function renderEvol(t) {
  const cy = t.res.cycles.map(c => c.cycle);
  const n = cy.length;
  const hline = (v, name, color, dash) => ({ x: [0, n + 1], y: [v, v], type: "scatter", mode: "lines", name, line: { color, width: 1.8, dash }, hoverinfo: "name" });
  const mk = (id, key, label, yt, refs = [], yr = null, legPos = "bl") => {
    const y = t.res.cycles.map(c => c[key]);
    const [lo, hi] = yr || paddedRange([...y, ...refs.map(r => r.y[0])]);
    const el = $(id);
    const h = (availFrom(el.parentElement) - 10) / 2;
    const w = Math.min(cellWidth(el), 1.45 * h);
    fixedReact(id, [
      { x: cy, y, type: "scatter", mode: "markers", name: label, marker: { size: 6, color: "#2468a8", opacity: 0.85 } },
      { x: [state.cycle], y: [y[state.cycle - 1]], type: "scatter", mode: "markers", showlegend: false, hoverinfo: "y", marker: { size: 11, color: "#111" } },
      ...markerLines(t, state.cycle, t.res.target_cycle, lo, hi),
      ...refs,
    ], baseLayout(null, "Cycle, N", yt, Object.assign({ margin: { l: 58, r: 14, t: 66, b: 44 },
      xaxis: ax("xaxis", { range: [0, n + 1], title: { text: "Cycle, N", font: { size: 13 } } }),
      yaxis: ax("yaxis", { range: [lo, hi], title: { text: yt, font: { size: 12 } } }) }, legendTop)), w, h);
  };
  mk("eRu", "ru", "r<sub>u,max</sub>", "r<sub>u,max</sub>", RU_REFS.map(r => hline(r.ru, ruName(r), r.color, r.dash)), [-0.05, 1.05]);
  mk("eHull", "hull", "Ratio<sub>hull</sub>", "Ratio<sub>hull</sub> = A<sub>loop</sub>/A<sub>hull</sub>", [], [0, 1.02]);
  mk("eDg", "dgamma", "Ratio<sub>Δγ</sub>", "Ratio<sub>Δγ</sub> = Δγ<sub>cyc</sub>/γ<sub>FC</sub>", [], null, "tr");
  mk("eGda", "gamma_DA", "γ<sub>DA</sub>", "Double-amplitude shear strain, γ<sub>DA</sub> (%)",
    [hline(6, "γ<sub>DA</sub> = 6% (minimum)", "#b3261e", "dot"), hline(9, "γ<sub>DA</sub> = 9% (assessment)", "#2e7d4f", "dashdot")], null, "br");
}

const legendPacked = { showlegend: true, legend: { orientation: "v", x: 1.02, xanchor: "left", y: 1, yanchor: "top", font: { size: 10.5 } } };
const CBT_MARGIN = { l: 58, r: 170, t: 24, b: 46 };
// plotting area width = 2 x height
function rectSize(id) {
  const el = $(id), m = CBT_MARGIN;
  let pw = cellWidth(el) - m.l - m.r, ph = pw / 2;
  const avail = availFrom(el.parentElement);
  if (ph + m.t + m.b > avail) { ph = avail - m.t - m.b; pw = 2 * ph; }
  return [pw + m.l + m.r, ph + m.t + m.b];
}
function renderCBT(t) {
  const cycles = t.res.cycles;
  const gda = cycles.map(c => c.gamma_DA);
  const comb = cycles.map(c => combine(c));
  const traces = [];
  if ($("showAnalystLines").checked) {
    for (const a of state.analysts) {
      if (!state.active.has(a)) continue;
      traces.push({ x: gda, y: cycles.map(c => c.cbt.analysts[a] ? c.cbt.analysts[a].median : null), type: "scatter", mode: "markers",
        name: a, marker: { color: ANALYST_COLORS[a], size: 4, opacity: 0.7 } });
    }
  }
  traces.push({ x: gda, y: comb.map(s => s && s.median), type: "scatter", mode: "markers", name: "Combined median",
    marker: { color: "#14283f", size: 7 } });
  const gSel = cycles[state.cycle - 1].gamma_DA;
  const gT = t.res.target_cycle ? cycles[t.res.target_cycle - 1].gamma_DA : null;
  // marker lines are labelled on the plot instead of in the legend, so the legend fits
  traces.push(...markerLines(t, gSel, gT, 0, 1).map(tr => Object.assign(tr, { showlegend: false })));
  const notes = [{ x: gSel, y: 1, yanchor: "bottom", xanchor: "right", text: `Selected (${fmt(gSel, 2)}%)`, showarrow: false, font: { size: 10.5, color: "#111" } }];
  if (gT !== null) notes.push({ x: gT, y: 1, yanchor: "bottom", xanchor: "left", text: `First γ<sub>DA</sub> ≥ 9%`, showarrow: false, font: { size: 10.5, color: "#2e7d4f" } });
  const gx = paddedRange(gda);
  fixedReact("cEvol", traces, baseLayout(null, "Double-amplitude shear strain, γ<sub>DA</sub> (%)", "CBT", Object.assign({
    margin: Object.assign({}, CBT_MARGIN),
    xaxis: ax("xaxis", { range: [Math.max(0, gx[0]), gx[1]], title: { text: "Double-amplitude shear strain, γ<sub>DA</sub> (%)", font: { size: 13 } } }),
    yaxis: ax("yaxis", { range: [0, 1], title: { text: "CBT", font: { size: 13 } } }), annotations: notes }, legendPacked)), ...rectSize("cEvol"));

  // distribution at current cycle
  const c = cycles[state.cycle - 1];
  const lg = $("logitSpace").checked;
  const pd = [];
  const xs = [];
  if (lg) { for (let i = 0; i <= 400; i++) xs.push(-10 + 20 * i / 400); }
  else { for (let i = 1; i < 400; i++) xs.push(i / 400); }
  const pdf = (mu, sd) => xs.map(x => {
    if (lg) return Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
    const z = (logit(x) - mu) / sd;
    return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI) * x * (1 - x));
  });
  if ($("showAnalystLines").checked) {
    for (const a of state.analysts) {
      const st = c.cbt.analysts[a];
      if (!state.active.has(a) || !st) continue;
      pd.push({ x: xs, y: pdf(st.mu_T, st.sigma_T), type: "scatter", mode: "lines", name: a, line: { color: ANALYST_COLORS[a], width: 1.2 } });
    }
  }
  const cb = combine(c);
  let ymax;
  if (cb) {
    const yc = pdf(cb.mu_T, cb.sigma_T);
    ymax = 3 * Math.max(...yc);
    pd.push({ x: xs, y: yc, type: "scatter", mode: "lines", name: `Combined,<br>${cycLabel(t, state.cycle)}`, line: { color: "#14283f", width: 3.5 } });
  }
  fixedReact("cPdf", pd, baseLayout(null, lg ? "logit(CBT)" : "CBT", "Probability density", Object.assign({
    margin: Object.assign({}, CBT_MARGIN),
    xaxis: ax("xaxis", lg ? { title: { text: "logit(CBT)", font: { size: 13 } } } : { range: [0, 1], title: { text: "CBT", font: { size: 13 } } }),
    yaxis: ax("yaxis", Object.assign({ title: { text: "Probability density", font: { size: 13 } } }, ymax ? { range: [0, ymax] } : {})),
  }, legendPacked)), ...rectSize("cPdf"));
}
$("showAnalystLines").onchange = () => { const t = curTest(); if (t) renderCBT(t); };
$("logitSpace").onchange = () => { const t = curTest(); if (t) renderCBT(t); };

function renderTable(t, c) {
  const rows = state.analysts.map(a => {
    const st = c.cbt.analysts[a];
    const on = state.active.has(a);
    if (!st) return `<tr class="off"><td>${a}</td><td colspan="8">missing metric</td></tr>`;
    return `<tr class="${on ? "" : "off"}"><td><span class="swatch" style="background:${ANALYST_COLORS[a]}"></span> ${a}</td>
      <td style="text-align:left">${state.models[a].map(m => METRIC_LABEL[m]).join(", ")}</td>
      <td>${fmt(st.mu_T)}</td><td>${fmt(st.sigma_T)}</td><td>${fmt(st.sd_param)}</td><td>${fmt(st.sd_model)}</td>
      <td>${fmt(st.median)}</td><td>${fmt(st.p16)} – ${fmt(st.p84)}</td><td>${on ? (1 / Math.max(1, state.active.size)).toFixed(3) : "0"}</td></tr>`;
  }).join("");
  const cb = combine(c);
  const comb = cb ? `<tr class="combined"><td>Combined</td><td></td><td>${fmt(cb.mu_T)}</td><td>${fmt(cb.sigma_T)}</td><td colspan="2" style="text-align:center">within ${fmt(cb.sd_within)} · between ${fmt(cb.sd_between)}</td><td>${fmt(cb.median)}</td><td>${fmt(cb.p16)} – ${fmt(cb.p84)}</td><td>1</td></tr>` : "";
  $("aTable").innerHTML = `<thead><tr><th>Analyst</th><th style="text-align:left">Predictors</th><th>μ<sub>T</sub></th><th>σ<sub>T</sub></th>
    <th>σ coef.</th><th>σ̂ model</th><th>Median CBT</th><th>16–84%</th><th>Weight</th></tr></thead><tbody>${rows}${comb}</tbody>`;
}

/* tabs */
document.querySelectorAll("#tabs button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".tabpane").forEach(p => p.classList.toggle("on", p.id === "tab-" + b.dataset.tab));
  state.tab = b.dataset.tab;
  renderCycleNow();
  window.dispatchEvent(new Event("resize"));
}));

/* ------------------------------------------------------------------ */
/*  Export                                                             */
/* ------------------------------------------------------------------ */
function download(name, text) {
  const blob = new Blob([text], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const csvVal = (v) => (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) ? "" : (typeof v === "number" ? +v.toPrecision(8) : `"${String(v).replace(/"/g, '""')}"`);
const activeList = () => state.analysts.filter(a => state.active.has(a));

$("exportCycles").onclick = () => {
  const t = curTest(); if (!t) return;
  const act = activeList();
  const cols = ["cycle", "gamma_SA", "gamma_DA", "CSR_cycle", "ru", "hull", "dgamma", "kappa", "str_acc"];
  const head = [...cols, "CBT_median", "CBT_p16", "CBT_p84", "CBT_mean", "mu_T", "sigma_T", "sigma_within", "sigma_between",
    ...act.flatMap(a => [`${a}_median`, `${a}_mu_T`, `${a}_sigma_T`])];
  const note = t.res.assessable ? "" : ` | CBT NOT ASSESSED: gamma_DA max ${fmt(t.res.gamma_DA_max, 2)}% < ${t.res.gamma_DA_min}%`;
  const lines = [`# CBT Tool v0.2 | file: ${t.name} | analysts: ${act.join(" ")} | gamma_DA=9% cycle: ${t.res.target_cycle || "not reached"}${note}`, head.join(",")];
  for (const c of t.res.cycles) {
    const cb = combine(c) || {};
    lines.push([...cols.map(k => csvVal(c[k])), csvVal(cb.median), csvVal(cb.p16), csvVal(cb.p84), csvVal(cb.mean), csvVal(cb.mu_T), csvVal(cb.sigma_T), csvVal(cb.sd_within), csvVal(cb.sd_between),
      ...act.flatMap(a => { const s = (c.cbt && c.cbt.analysts[a]) || {}; return [csvVal(s.median), csvVal(s.mu_T), csvVal(s.sigma_T)]; })].join(","));
  }
  download(t.name.replace(/\.[^.]+$/, "") + "_CBT_cycles.csv", lines.join("\n"));
};

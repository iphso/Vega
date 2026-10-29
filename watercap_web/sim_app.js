// Water capture simulator: designs are the unit; every design is evaluated under one set of conditions, at the
// fidelity you choose (Fast, Resolved, 3D quick, 3D full), and always shows its best current result.
// Engines: wcmodel.js / wcgpu.js (Fast, batched on the GPU), pyworker.js (Resolved: the campaign's Python
// under Pyodide), the StoneLab device3d worker (3D). Views: view3d.js (axisymmetric designs), sim3d.js (3D).
"use strict";
const $ = s => document.querySelector(s);
const PI = Math.PI;
const toast = (t, ms = 3500) => { const el = $("#toast"); el.textContent = t; el.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => { el.hidden = true; }, ms); };

/* ======================= families, designs ======================= */
const FAMNAME = { corona_only: "corona only", cold_spray: "cold spray", charged_spray: "charged spray", rain_curtain: "rain curtain", disc_variants: "disc variants",
  multi_tier: "multi-tier", ring_spray: "ring spray", coflow_spray: "coflow spray", charged_film: "charged film", cooled_wall: "cooled wall", tabbed_ring: "tabbed ring" };
const FAMCOL = { tabbed_ring: "#ffb547", corona_only: "#4e79a7", cold_spray: "#59a14f", charged_spray: "#e15759", rain_curtain: "#b07aa1", disc_variants: "#f28e2b",
  multi_tier: "#edc948", ring_spray: "#9c755f", coflow_spray: "#bab0ac", charged_film: "#76b7b2", cooled_wall: "#ff9da7" };
const famCol = f => FAMCOL[f] || "#8a8a94";
// the tabbed-ring electrode stack (3D only): 8 plates at 31.25 mm in a 150 mm tube, 130 m^3/h
const TABBED = { tube_diameter_m: 0.15, plates: 8, pitch_m: 0.03125, air_flow_m3_s: 130 / 3600, tipApexR: 0.0557 };
// reference LES pressure drop per pitch (StoneLab solver3d E4b, e4b_analyse.json), for the 3D quick flows
const REF_DP = { off: 0.2640, on: 0.3348, off_flip: 0.2785, on_flip: 0.3967 };
const COND0 = { inlet_T_C: 35, inlet_RH: 1, inlet_lwc_g_m3: 2, mist_mmd_um: 15, mist_gsd: 1.6, ambient_T_C: 25, seawater_T_C: null, site: "" };
const condKey = c => JSON.stringify([c.inlet_T_C, c.inlet_RH, c.inlet_lwc_g_m3, c.mist_mmd_um, c.mist_gsd, c.ambient_T_C, c.seawater_T_C]);
const CK0 = condKey(COND0);
let cond = { ...COND0 };
try { const s = JSON.parse(localStorage.getItem("wc.cond") || "null"); if (s) cond = { ...COND0, ...s }; } catch (e) { /* default */ }

// a design's parameters as a key (memoized: parameter objects are replaced, never mutated)
const DK = new WeakMap();
const dKey = d => { let k = DK.get(d); if (k === undefined) { k = JSON.stringify(Object.keys(d).sort().map(q => [q, d[q]])); DK.set(d, k); } return k; };
const designs = [];            // { id, name, fam, kind: "disc" | "tabbed", d, src: "library" | "team" }
const results = {};            // id -> tier -> { cond, dk, m, at, by }
(() => {
  const rank = { v1: 0, rerun: 1, new: 1 }, best = new Map();
  for (const r of DATA ? DATA.rows : []) {
    r.id = r.id.replace(/^brad_variants/, "disc_variants"); if (r.fam === "brad_variants") r.fam = "disc_variants";
    const q = best.get(r.id); if (!q || rank[r.set] > rank[q.set]) best.set(r.id, r);
  }
  designs.push({ id: "tabbed-ring", name: "tabbed ring · 8 plates", fam: "tabbed_ring", kind: "tabbed", src: "library", measured: 1 - 3.73 / 10.63,
    d: { tip_radius_mm: 0.1, corona_voltage_kV: 12, tab_lean: "downstream" } });
  for (const r of best.values()) {
    const d = { disc_thickness_mm: 0.3, spray_gsd: 1.5, spray_v0: 2, spray_cone_half_angle_deg: 30, wall_U_W_m2K: 0, ...r.d };
    designs.push({ id: r.id, name: r.id.replace(/_/g, " "), fam: r.fam, kind: "disc", src: "library", d });
    // the library's stored run is this design's Resolved result at the library conditions (the first sweep's
    // spray designs used the fast spray model, so only their spray-free runs count as Resolved)
    if (r.set !== "v1" || !(r.d.spray_flow_L_min > 0)) results[r.id] = { resolved: { cond: CK0, dk: dKey(d), m: fromLibrary(r.r), at: 0, by: "library" } };
  }
})();
function famsPresent() { return [...new Set(designs.map(x => x.fam))]; }

/* ======================= metrics: one shape for every tier ======================= */
// cap = water captured, fraction of the water entering (vapor + liquid); capLh = L/h; Wh = all modeled loads per
// litre captured; dp = Pa; arc = 1 - V / sparkover. x = extras for the detail table.
function metric(cap, capLh, loadsW, dp, arc, x = {}) { return { cap, capLh, Wh: capLh > 1e-3 ? loadsW / capLh : NaN, dp, arc, x: { loadsW, ...x } }; }
function fromLibrary(R) {
  return metric(1 - R.out, R.rec, R.elecW + R.fanW + (R.pumpW || 0) + (R.recoolW || 0), R.dp, R.arc,
    { "corona current": [R.Ima, "mA"], "electrical": [R.elecW, "W"], "fan": [R.fanW, "W"], "pump": [R.pumpW, "W"], "spray recooling": [R.recoolW, "W"], "vapor removed": [100 * R.vap, "%"], "liquid escaping": [100 * R.esc, "%"] });
}
function fromFast(r) {
  return metric(1 - r.outlet_total_water_frac, r.recovered_L_h, r.electrical_power_W + r.fan_power_W, r.pressure_drop_Pa, r.arc_margin,
    { "corona current": [r.corona_current_mA, "mA"], "electrical": [r.electrical_power_W, "W"], "fan": [r.fan_power_W, "W"], "vapor removed": [100 * r.vapor_removed_frac, "%"], "liquid escaping": [100 * r.liquid_escape_frac, "%"] });
}
function fromResolved(m) {
  const loads = m.electrical_power_W + m.fan_power_W + (m.pump_power_W || 0) + (m.recool_power_W || 0) + (m.wall_cooling_power_W || 0);
  return metric(1 - m.outlet_total_water_frac, m.recovered_L_h, loads, m.pressure_drop_Pa, m.arc_margin,
    { "corona current": [m.corona_current_mA, "mA"], "electrical": [m.electrical_power_W, "W"], "fan": [m.fan_power_W, "W"], "pump": [m.pump_power_W, "W"],
      "spray recooling": [m.recool_power_W, "W"], "wall cooling": [m.wall_cooling_power_W || null, "W"], "vapor removed": [100 * m.vapor_removed_frac, "%"], "liquid escaping": [100 * m.liquid_escape_frac, "%"] });
}
// psychrometrics for the 3D tier's water balance (it moves mist, not vapor)
const psat = T => 611.21 * Math.exp((18.678 - T / 234.5) * (T / (257.14 + T)));
function waterIn(Q, c) {
  const pv = c.inlet_RH * psat(c.inlet_T_C), w = 0.622 * pv / (101325 - pv), rho = (101325 - pv) / (287.05 * (c.inlet_T_C + 273.15));
  return { vapor: rho * Q * w * 3600, mist: c.inlet_lwc_g_m3 * Q * 3.6 };
}
// the inlet mist's mass in each simulated size class (lognormal MMD / GSD, bins split at the geometric midpoints)
function massWeights(diam, mmd, gsd) {
  const erf = x => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const cdf = d => d <= 0 ? 0 : d === Infinity ? 1 : 0.5 * (1 + erf(Math.log(d / mmd) / (Math.SQRT2 * Math.log(gsd))));
  const edges = [0, ...diam.slice(1).map((d, i) => Math.sqrt(d * diam[i])), Infinity];
  return diam.map((_, i) => cdf(edges[i + 1]) - cdf(edges[i]));
}
function sparkV(gap, pol) { const gc = gap * 100; return 1e3 * (pol > 0 ? 4.5 : 8.0) * (1 + 1 / Math.sqrt(gc)) * gc; }
function from3D(rows, I_uA, kV, dpPitch, c) {
  const Q = TABBED.air_flow_m3_s, w = waterIn(Q, c), wt = massWeights(rows.map(q => q.d_um), c.mist_mmd_um, c.mist_gsd);
  const mistCap = rows.reduce((a, q, i) => a + wt[i] * Math.max(0, q.reduction), 0), capLh = mistCap * w.mist;
  const elec = TABBED.plates * I_uA * 1e-6 * kV * 1e3, dp = dpPitch * TABBED.plates, fan = dp * Q / 0.6;
  return metric(capLh / (w.vapor + w.mist), capLh, elec + fan, dp, 1 - kV * 1e3 / sparkV(TABBED.tube_diameter_m / 2 - TABBED.tipApexR, 1),
    { "mist removed": [100 * mistCap, "%"], "corona current": [TABBED.plates * I_uA * 1e-3, "mA"], "electrical": [elec, "W"], "fan": [fan, "W"] });
}

/* ======================= tiers ======================= */
const TIERS = [
  { k: "fast", n: "Fast", est: "instant", badge: "FAST" },
  { k: "resolved", n: "Resolved", est: "~1–3 min", badge: "RESOLVED" },
  { k: "3dq", n: "3D quick", est: "~2 min", badge: "3D" },
  { k: "3df", n: "3D full", est: "~45 min", badge: "3D FULL" },
];
const TK = Object.fromEntries(TIERS.map(t => [t.k, t]));
// what a run will cost, from the design (the spray stage's CFD dominates Resolved when there is spray)
function estimate(x, d, t) {
  if (t.k === "resolved") return d.spray_flow_L_min > 0 ? "~3–8 min" : "~1 min";
  return t.est;
}
let D3DISC = false;   // disc designs in 3D: on once the device3d worker offers it
function tierAvail(x, t) {
  if (x.kind === "tabbed") return t === "3dq" || t === "3df";
  if (t === "fast" || t === "resolved") return true;
  return D3DISC;
}
// results live under the design's id, or under id::draft while its parameters are being edited; a result
// counts when it was made under the current conditions for exactly these parameters
const rkey = (x, d = x.d) => dKey(d) === dKey(x.d) ? x.id : x.id + "::draft";
const resOf = (x, d = x.d) => (results[rkey(x, d)] ||= {});
const valid = (x, t, d = x.d) => { const r = results[rkey(x, d)]?.[t]; return r && r.cond === condKey(cond) && r.dk === dKey(d) ? r : null; };
function best(x, d = x.d) {
  for (const t of ["3df", "3dq", "resolved", "fast"]) { const r = valid(x, t, d); if (r) return { t, r }; }
  // nothing current: the latest stale result, greyed
  for (const t of ["3df", "3dq", "resolved", "fast"]) { const r = results[rkey(x, d)]?.[t]; if (r) return { t, r, stale: true }; }
  return null;
}
const lesRuns = {};   // design key -> the last browser LES summary

/* ======================= Fast tier: the whole library on the GPU ======================= */
const IN = WCGPU.IN, S_IN = WCGPU.IN_STRIDE, S_OUT = WCGPU.OUT_STRIDE, O = Object.fromEntries(WCGPU.OUT.map((k, i) => [k, i]));
const MODEL_OPT = { nb: 4, nz: 40, outer: 3, fall_steps: 4000 };
function fastX(d, c = cond) {
  return { air_flow_m3_s: d.air_flow_m3_s, tube_diameter_m: d.tube_diameter_m, spray_section_length_m: d.spray_section_length_m, corona_section_length_m: d.corona_section_length_m,
    nozzle_height_frac: d.nozzle_height_frac, disc_diameter_frac: d.disc_diameter_frac, disc_spacing_m: d.disc_spacing_m, spray_flow_L_min: d.spray_flow_L_min, spray_T_C: d.spray_T_C,
    spray_d_um: d.spray_d_um, spray_q_frac: d.spray_q_frac <= 1.0001e-5 ? 0 : d.spray_q_frac, spray_gsd: Math.max(d.spray_gsd ?? 1.5, 1), spray_v0: d.spray_v0 ?? 2,
    corona_voltage_kV: d.corona_voltage_kV, corona_polarity: d.corona_polarity, spray_polarity_rel: d.spray_polarity_rel ?? 1,
    inlet_T_C: c.inlet_T_C, inlet_RH: c.inlet_RH, inlet_lwc_g_m3: c.inlet_lwc_g_m3, mist_mmd_um: c.mist_mmd_um, mist_gsd: c.mist_gsd };
}
let engine = null;
async function makeEngine() {
  try { const g = await WCGPU.GPUModel.create(FIT); return { run: (arr, n) => g.run(arr, n, { ...MODEL_OPT, corona: "refit" }) }; }
  catch (e) {
    const src = $("#wcm-src").textContent, OUTK = WCGPU.OUT;
    const wsrc = src + `\nconst FIT=${JSON.stringify(FIT)};const IN=${JSON.stringify(IN)};const OUTK=${JSON.stringify(OUTK)};
onmessage=e=>{const {arr,n,opt,id}=e.data;const out=new Float32Array(n*${S_OUT});for(let i=0;i<n;i++){const x={};IN.forEach((k,j)=>x[k]=arr[i*${S_IN}+j]);
const r=WCModel.simulate(x,{...opt,fit:FIT});for(let k=0;k<${S_OUT};k++)out[i*${S_OUT}+k]=r[OUTK[k]];}postMessage({out,id},[out.buffer]);};`;
    const N = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
    const ws = Array.from({ length: N }, () => new Worker(URL.createObjectURL(new Blob([wsrc], { type: "text/javascript" }))));
    let seq = 0;
    return { run(arr, n) {
      const per = Math.ceil(n / N), jobs = [];
      for (let w = 0; w < N; w++) {
        const a = w * per, b = Math.min(n, a + per); if (a >= b) continue;
        const id = ++seq;
        jobs.push(new Promise(res => { const h = e => { if (e.data.id === id) { ws[w].removeEventListener("message", h); res([a, e.data.out]); } }; ws[w].addEventListener("message", h); ws[w].postMessage({ arr: arr.slice(a * S_IN, b * S_IN), n: b - a, opt: MODEL_OPT, id }); }));
      }
      return Promise.all(jobs).then(parts => { const out = new Float32Array(n * S_OUT); for (const [a, o] of parts) out.set(o, a * S_OUT); return out; });
    } };
  }
}
async function fastAll() {
  const xs = designs.filter(x => x.kind === "disc");
  const arr = new Float32Array(xs.length * S_IN);
  xs.forEach((x, i) => WCGPU.packDesign(arr, i, fastX(x.d)));
  const out = await engine.run(arr, xs.length), ck = condKey(cond);
  xs.forEach((x, i) => {
    const r = {}; for (const k of WCGPU.OUT) r[k] = out[i * S_OUT + O[k]];
    (results[x.id] ||= {}).fast = { cond: ck, dk: dKey(x.d), m: fromFast(r), at: Date.now(), by: "" };
  });
}

/* ======================= storage: the team's store, or this browser ======================= */
const API = new URL("api/", location.href).href;
let remote = false;
async function storeLoad() {
  try {
    const r = await fetch(API + "store", { cache: "no-store" });
    if (r.ok) { const s = await r.json(); remote = true; return s; }
  } catch (e) { /* no store here */ }
  try { return JSON.parse(localStorage.getItem("wc.store") || "null") || { designs: [], results: {} }; } catch (e) { return { designs: [], results: {} }; }
}
function localSave() {
  if (remote) return;
  try {
    const team = designs.filter(x => x.src === "team").map(({ id, name, fam, kind, d }) => ({ id, name, fam, kind, d }));
    const res = {}; for (const [id, t] of Object.entries(results)) if (!id.endsWith("::draft")) for (const [k, v] of Object.entries(t)) if (k !== "fast" && v.by !== "library") (res[id] ||= {})[k] = v;
    localStorage.setItem("wc.store", JSON.stringify({ designs: team, results: res }));
  } catch (e) { toast("could not save in this browser: " + e.message); }
}
async function putDesign(x) {
  if (!remote) return localSave();
  const r = await fetch(API + "designs/" + encodeURIComponent(x.id), { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: x.name, fam: x.fam, kind: x.kind, d: x.d }) });
  if (!r.ok) toast("save failed: " + r.status);
}
async function delDesign(x) {
  if (!remote) return localSave();
  await fetch(API + "designs/" + encodeURIComponent(x.id), { method: "DELETE" });
}
async function putResult(id, t, v) {
  if (!remote) return localSave();
  fetch(API + `results/${encodeURIComponent(id)}/${t}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(v) }).catch(() => toast("result not shared (network)"));
}

/* ======================= selection, drafts ======================= */
let sel = null, draft = null;      // draft: edited parameters of sel (null = unmodified)
const curD = () => draft || sel.d;
const isDraft = () => !!draft && dKey(draft) !== dKey(sel.d);

/* ======================= Resolved tier (Pyodide worker) ======================= */
let resWorker = null, resBooted = false, resJob = 0;
const running = {};               // id|tier -> { frac, note }
function runResolved(x, d) {
  if (!resWorker) { resWorker = new Worker(URL.createObjectURL(new Blob([$("#res-src").textContent], { type: "text/javascript" }))); resBooted = false; }
  const job = ++resJob, key = x.id + "|resolved", ck = condKey(cond), dk = dKey(d), t0 = performance.now();
  const aux = { inlet_T_C: cond.inlet_T_C, inlet_RH: cond.inlet_RH, inlet_lwc_g_m3: cond.inlet_lwc_g_m3, mist_mmd_um: cond.mist_mmd_um, mist_gsd: cond.mist_gsd, ambient_T_C: cond.ambient_T_C };
  if (cond.seawater_T_C != null) aux.seawater_T_C = cond.seawater_T_C;
  const nPer = Math.max(1, Math.round(d.corona_section_length_m / d.disc_spacing_m)), spray = d.spray_flow_L_min > 0;
  running[key] = { frac: 0.02, note: "starting Python" };
  render();
  resWorker.onmessage = e => {
    const m = e.data; if (m.id !== job) return;
    if (m.type === "progress") {
      running[key] = m.stage === "load" ? { frac: 0.03, note: "starting Python" } : m.stage === "spray" ? { frac: 0.05 + 0.65 * Math.min(1, (m.spray.outer + 1) / 40), note: `spray · iteration ${m.spray.outer + 1}` } : { frac: (spray ? 0.7 : 0.05) + (spray ? 0.3 : 0.95) * (m.period + 1) / nPer, note: `corona · disc ${m.period + 1} of ${nPer}` };
      renderSide();
    } else if (m.type === "done") {
      resBooted = true; delete running[key];
      const v = { cond: ck, dk, m: fromResolved(m.result.metrics), at: Date.now(), by: "", secs: (performance.now() - t0) / 1e3 };
      resOf(x, d).resolved = v;
      if (rkey(x, d) === x.id) putResult(x.id, "resolved", v);
      render();
    } else if (m.type === "error") {
      delete running[key]; toast("Resolved run failed: " + m.message); resWorker.terminate(); resWorker = null; render();
    }
  };
  resWorker.postMessage({ type: "run", id: job, design: d, aux, preset: "resolved_coarse", seed: 0, base: new URL(".", location.href).href,
    bundle: resBooted ? null : JSON.parse($("#pybundle").textContent) });
}

/* ======================= 3D tier (device3d worker) ======================= */
let d3w = null, d3id = 0, d3ready = null, d3mesh = {}, d3busy = false;
const d3pend = {}, d3loaded = new Set();
function d3call(type, args = {}, onlog) {
  if (!d3w) {
    d3w = new Worker(URL.createObjectURL(new Blob([$("#d3-src").textContent], { type: "text/javascript" })), { type: "module" });
    d3w.onmessage = e => {
      const m = e.data;
      if (m.lost) { toast("GPU lost: " + m.lost); return; }
      if (m.log) { d3pend[m.id]?.[2]?.(m.log); return; }
      const p = d3pend[m.id]; if (!p) return;
      delete d3pend[m.id]; m.err ? p[1](new Error(m.err)) : p[0](m.ok);
    };
  }
  const id = ++d3id;
  return new Promise((res, rej) => { d3pend[id] = [res, rej, onlog]; d3w.postMessage({ id, type, ...args }); });
}
const d3Base = () => new URL("device3d", location.href).href;
async function d3load(names) { const need = names.filter(n => !d3loaded.has(n)); if (need.length) { await d3call("load", { base: d3Base(), names: need }); need.forEach(n => d3loaded.add(n)); } }
async function d3init() {
  if (!d3ready) d3ready = (async () => {
    if (!navigator.gpu) throw new Error("3D needs WebGPU (a recent Chrome or Edge)");
    const info = await d3call("init");
    if (!window.Sim3D) await new Promise(r => window.addEventListener("sim3d-ready", r, { once: true }));
    Sim3D.init($("#s3view"));
    Sim3D.setStack({ ...info.device, plateZ0: 0.013125 });
    return info;
  })();
  return d3ready;
}
async function d3show(x, d, flowKey, fieldKey, drops) {
  await d3init();
  const rTip = String(d.tip_radius_mm);
  if (!d3mesh[rTip]) d3mesh[rTip] = await d3call("mesh", { rTip });
  Sim3D.setMesh(d3mesh[rTip]);
  if (flowKey.startsWith("pre:")) await d3load([flowKey.slice(4)]);
  Sim3D.setFlow(await d3call("flowGrid", { flow: flowKey, stride: 2 }));
  if (fieldKey) { if (fieldKey.startsWith("pre:")) await d3load([fieldKey.slice(4)]); const I = await d3call("ions", { field: fieldKey, floor: 3e13, max: 45000 }); Sim3D.setIons(I.points, I.mirrored); }
  Sim3D.setDrops(drops || null);
}
const leanKey = d => d.tab_lean === "upstream" ? "_flip" : "";
async function run3D(x, d, tier) {
  const key = x.id + "|" + tier, ck = condKey(cond), dk = dKey(d), t0 = performance.now();
  if (d3busy) { toast("a 3D run is already going"); return; }
  d3busy = true;
  const note = s => { running[key] = { ...running[key], note: s.replace(/\s*\.\.\.$/, "") }; renderSide(); };
  running[key] = { frac: 0.03, note: "starting" }; render();
  try {
    await d3init();
    const fl = leanKey(d), base = "pre:off" + fl;
    let flow, field, I_uA;
    running[key].frac = 0.1;
    const tip = await d3call("tip", { rTip: String(d.tip_radius_mm), kV: [d.corona_voltage_kV], fieldKV: d.corona_voltage_kV }, note);
    I_uA = tip.rows[0]?.I_uA ?? 0;
    if (tier === "3dq") { flow = "pre:on" + fl; field = "tip"; }
    else {
      running[key].frac = 0.2; note("LES");
      const r = await d3call("les", { kV: d.corona_voltage_kV, rTip: String(d.tip_radius_mm), flip: !!fl, tEnd: 2.5, statsStart: 0.5 }, s => { const m = /t=([\d.]+)/.exec(s); if (m) running[key].frac = 0.2 + 0.7 * Math.min(1, +m[1] / 2.5); note(s.slice(0, 60)); });
      flow = r.flowKey; field = r.fieldKey || "tip"; if (r.summary.ehd) I_uA = r.summary.ehd.I_uA;
      lesRuns[rkey(x, d)] = r.summary;
    }
    running[key].frac = 0.92; note("capture");
    await d3load([...new Set([flow, base].filter(k => k.startsWith("pre:")).map(k => k.slice(4)))]);
    const cap = await d3call("mist", { field, flow, baseline: base, perClass: 20000, inertia: "none" }, note);
    const dpPitch = tier === "3df" ? lesRuns[rkey(x, d)].dp_per_pitch_Pa : REF_DP["on" + fl];
    const v = { cond: ck, dk, m: from3D(cap.rows, I_uA, d.corona_voltage_kV, dpPitch, cond), at: Date.now(), by: "", secs: (performance.now() - t0) / 1e3,
      bySize: cap.rows.map(q => [q.d_um, q.reduction]), view: { flow, field } };
    resOf(x, d)[tier] = v;
    if (rkey(x, d) === x.id) putResult(x.id, tier, v);
    if (x === sel) d3show(x, d, flow, field, cap.rows[nearestClass(cap.rows)].charged.sample).catch(e => toast(e.message));
  } catch (e) { toast("3D run failed: " + e.message); }
  delete running[key]; d3busy = false; render();
}
const nearestClass = rows => { let b = 0; rows.forEach((q, i) => { if (Math.abs(Math.log(q.d_um / cond.mist_mmd_um)) < Math.abs(Math.log(rows[b].d_um / cond.mist_mmd_um))) b = i; }); return b; };

/* ======================= list + map ======================= */
const lst = { q: "", sort: "cap", fams: null, mode: "list" };
// numbers as people read them: 2 significant figures below 100, whole numbers above, never an exponent
const num = (v, sig = 2) => !isFinite(v) ? "–" : Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (+v.toPrecision(sig)).toString();
const fmt = { cap: m => isFinite(m.cap) ? (100 * m.cap).toFixed(m.cap < 0.1 ? 1 : 0) + "%" : "–", Wh: m => isFinite(m.Wh) ? num(m.Wh) + " Wh/L" : "–", dp: m => isFinite(m.dp) ? num(m.dp) + " Pa" : "–", arc: m => isFinite(m.arc) ? m.arc.toFixed(2) : "–" };
function sortVal(x) {
  const b = best(x); if (!b) return Infinity;
  const m = b.r.m, s = b.stale ? 1e6 : 0;
  return lst.sort === "cap" ? s - m.cap : lst.sort === "Wh" ? s + (isFinite(m.Wh) ? m.Wh : 1e5) : lst.sort === "dp" ? s + m.dp : lst.sort === "arc" ? s - m.arc : 0;
}
function shown() {
  const q = lst.q.toLowerCase();
  return designs.filter(x => lst.fams.has(x.fam) && (!q || x.name.toLowerCase().includes(q) || (FAMNAME[x.fam] || "").includes(q)));
}
function renderList() {
  const xs = shown();
  const sv = new Map(xs.map(x => [x, sortVal(x)]));
  const sorted = lst.sort === "name" ? xs.slice().sort((a, b) => a.name.localeCompare(b.name)) : xs.slice().sort((a, b) => sv.get(a) - sv.get(b));
  const row = x => {
    const b = best(x), m = b?.r.m;
    return `<div class="item${b?.stale ? " stale" : ""}" data-id="${x.id}" aria-selected="${x === sel}" role="option"><span class="dot" style="background:${famCol(x.fam)}"></span><span class="nm">${esc(x.name)}</span>` +
      `${b ? `<span class="tier t-${b.t}">${TK[b.t].badge}</span>` : `<span class="tier">–</span>`}<span class="mx">${m ? `<span><b>${fmt.cap(m)}</b> captured</span><span>${fmt.Wh(m)}</span><span>${fmt.dp(m)}</span>` : "not run"}</span></div>`;
  };
  const team = sorted.filter(x => x.src === "team"), lib = sorted.filter(x => x.src !== "team");
  const LIM = 150;
  $("#list").innerHTML = (team.length ? `<div class="gtitle"><h2>Team · ${team.length}</h2></div>${team.map(row).join("")}` : "") +
    `<div class="gtitle"><h2>Library · ${lib.length}</h2></div>${lib.slice(0, LIM).map(row).join("")}` +
    (lib.length > LIM ? `<div class="more"><button type="button" class="btn" id="listmore">show all ${lib.length}</button></div>` : "");
  $("#list").querySelectorAll(".item").forEach(el => el.onclick = () => select(designs.find(x => x.id === el.dataset.id)));
  $("#listmore")?.addEventListener("click", e => { e.target.parentElement.outerHTML = lib.slice(LIM).map(row).join(""); $("#list").querySelectorAll(".item").forEach(el => el.onclick = () => select(designs.find(x => x.id === el.dataset.id))); });
  $("#fams").innerHTML = famsPresent().map(f => `<button type="button" data-f="${f}" aria-pressed="${lst.fams.has(f)}"><span class="dot" style="background:${famCol(f)}"></span>${FAMNAME[f] || f}</button>`).join("");
  $("#fams").querySelectorAll("button").forEach(b => b.onclick = e => {
    const f = b.dataset.f;
    if (e.altKey || e.metaKey) lst.fams = new Set([f]); else lst.fams.has(f) ? lst.fams.delete(f) : lst.fams.add(f);
    if (!lst.fams.size) lst.fams = new Set(famsPresent());
    renderList(); renderMap();
  });
}
let mapPts = [];
function renderMap() {
  const cv = $("#map"); if (lst.mode !== "map") { cv.hidden = true; return; }
  cv.hidden = false;
  const w = cv.parentElement.clientWidth, h = Math.round(w * 0.9), dpr = devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const M = { l: 34, r: 8, t: 8, b: 26 }, pw = w - M.l - M.r, ph = h - M.t - M.b;
  const pts = shown().map(x => { const b = best(x); return b && !b.stale && isFinite(b.r.m.Wh) && b.r.m.Wh > 0 ? { x, Wh: b.r.m.Wh, cap: b.r.m.cap } : null; }).filter(Boolean);
  const X = v => M.l + (Math.log10(v) + 1) / 4 * pw, Y = v => M.t + ph - Math.min(1, Math.max(0, v / 0.7)) * ph;   // 0.1 .. 1000 Wh/L, 0 .. 70%
  g.strokeStyle = "#2a2a30"; g.fillStyle = "#55555e"; g.font = "10px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
  for (const v of [0.1, 1, 10, 100, 1000]) { g.beginPath(); g.moveTo(X(v), M.t); g.lineTo(X(v), M.t + ph); g.stroke(); g.fillText(v >= 1 ? v : "0.1", X(v) - 6, h - 12); }
  for (const v of [0, 0.2, 0.4, 0.6]) { g.beginPath(); g.moveTo(M.l, Y(v)); g.lineTo(M.l + pw, Y(v)); g.stroke(); g.fillText((100 * v).toFixed(0) + "%", 2, Y(v) + 3); }
  g.fillText("Wh/L →", M.l + pw - 34, h - 2); g.save(); g.translate(10, M.t + 46); g.rotate(-PI / 2); g.fillText("captured →", 0, 0); g.restore();
  mapPts = [];
  for (const p of pts) { const px = X(p.Wh), py = Y(p.cap); mapPts.push({ px, py, x: p.x }); g.globalAlpha = p.x === sel ? 1 : 0.6; g.fillStyle = famCol(p.x.fam); g.beginPath(); g.arc(px, py, p.x === sel ? 5 : 2.6, 0, 7); g.fill(); }
  g.globalAlpha = 1;
}
$("#map").onclick = e => { const r = e.target.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top; let b = null, bd = 144; for (const p of mapPts) { const d = (p.px - mx) ** 2 + (p.py - my) ** 2; if (d < bd) { bd = d; b = p; } } if (b) select(b.x); };
$("#q").oninput = e => { lst.q = e.target.value; renderList(); renderMap(); };
$("#sort").onchange = e => { lst.sort = e.target.value; renderList(); };
$("#lmode").querySelectorAll("button").forEach(b => b.onclick = () => { lst.mode = b.dataset.m; $("#lmode").querySelectorAll("button").forEach(q => q.setAttribute("aria-pressed", q === b)); renderMap(); });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/* ======================= the design: header, view, parameters ======================= */
const DISC_P = [
  { g: "Tube", k: "tube_diameter_m", n: "Diameter", u: "cm", s: 100, st: 1, lo: 10, hi: 60 },
  { g: "Tube", k: "air_flow_m3_s", n: "Air flow", u: "m³/h", s: 3600, st: 10, lo: 100, hi: 2200 },
  { g: "Tube", k: "spray_section_length_m", n: "Spray section", u: "m", s: 1, st: 0.05, lo: 0.3, hi: 3 },
  { g: "Tube", k: "corona_section_length_m", n: "Corona section", u: "m", s: 1, st: 0.05, lo: 0.3, hi: 3 },
  { g: "Electrode", k: "disc_diameter_frac", n: "Disc / tube diameter", u: "", s: 1, st: 0.01, lo: 0.05, hi: 0.6 },
  { g: "Electrode", k: "disc_spacing_m", n: "Disc spacing", u: "cm", s: 100, st: 0.5, lo: 2, hi: 30 },
  { g: "Electrode", k: "disc_thickness_mm", n: "Disc thickness", u: "mm", s: 1, st: 0.05, lo: 0.1, hi: 1 },
  { g: "Electrode", k: "corona_voltage_kV", n: "Voltage", u: "kV", s: 1, st: 0.5, lo: 0, hi: 75 },
  { g: "Electrode", k: "corona_polarity", n: "Polarity", opts: [[-1, "negative"], [1, "positive"]] },
  { g: "Spray", k: "spray_flow_L_min", n: "Flow", u: "L/min", s: 1, st: 0.1, lo: 0, hi: 20 },
  { g: "Spray", k: "spray_T_C", n: "Temperature", u: "°C", s: 1, st: 0.5, lo: 5, hi: 40 },
  { g: "Spray", k: "spray_d_um", n: "Drop size", u: "µm", s: 1, st: 5, lo: 5, hi: 2000 },
  { g: "Spray", k: "spray_q_frac", n: "Drop charge", u: "× Rayleigh", s: 1, st: 0.01, lo: 0, hi: 0.5 },
  { g: "Spray", k: "nozzle_height_frac", n: "Nozzle height", u: "of section", s: 1, st: 0.05, lo: 0.02, hi: 1 },
  { g: "Spray", k: "spray_v0", n: "Launch speed", u: "m/s", s: 1, st: 0.5, lo: 1, hi: 20 },
  { g: "Spray", k: "spray_cone_half_angle_deg", n: "Cone half-angle", u: "°", s: 1, st: 1, lo: 5, hi: 60 },
  { g: "Spray", k: "spray_polarity_rel", n: "Charge vs corona", opts: [[1, "same sign"], [-1, "opposite"]] },
  { g: "Wall", k: "wall_U_W_m2K", n: "Wall cooling", u: "W/m²K", s: 1, st: 5, lo: 0, hi: 100 },
  { g: "Wall", k: "wall_coolant", n: "Coolant", opts: [["ambient", "ambient air"], ["seawater", "seawater"]], when: d => d.wall_U_W_m2K > 0 },
];
const TAB_P = [
  { g: "Electrode", k: "tip_radius_mm", n: "Tip radius", u: "mm", opts: [[0.05, "0.05 mm"], [0.1, "0.1 mm"], [0.25, "0.25 mm"], [0.5, "0.5 mm"]] },
  { g: "Electrode", k: "corona_voltage_kV", n: "Voltage", u: "kV", s: 1, st: 0.5, lo: 6, hi: 20 },
  { g: "Electrode", k: "tab_lean", n: "Tips lean", opts: [["downstream", "downstream"], ["upstream", "upstream"]] },
  { g: "Tube", fixed: "15 cm", n: "Diameter" }, { g: "Tube", fixed: "8 × 31.25 mm", n: "Plates" }, { g: "Tube", fixed: "130 m³/h", n: "Air flow" },
];
function renderParams() {
  const x = sel, d = curD(), P = x.kind === "tabbed" ? TAB_P : DISC_P, groups = {};
  for (const p of P) {
    if (p.k && d[p.k] === undefined && !p.opts) continue;
    if (p.when && !p.when(d)) continue;
    let ctl;
    if (p.fixed) ctl = `<input value="${p.fixed}" readonly aria-label="${p.n}">`;
    else if (p.opts) ctl = `<select data-k="${p.k}" aria-label="${p.n}">${p.opts.map(([v, n]) => `<option value="${v}"${String(d[p.k] ?? p.opts[0][0]) === String(v) ? " selected" : ""}>${n}</option>`).join("")}</select>`;
    else ctl = `<input type="number" data-k="${p.k}" value="${+(d[p.k] * p.s).toPrecision(3)}" step="${p.st}" min="${p.lo}" max="${p.hi}" aria-label="${p.n} (${p.u})">`;
    (groups[p.g] ||= []).push(`<label class="prow${p.fixed ? " fixed" : ""}"><span>${p.n}${p.u && !p.opts ? ` <span class="u">${p.u}</span>` : ""}</span>${ctl}</label>`);
  }
  $("#params").innerHTML = Object.entries(groups).map(([g, rows]) => `<div class="pgroup"><h2>${g}</h2>${rows.join("")}</div>`).join("");
  $("#params").querySelectorAll("[data-k]").forEach(el => el.onchange = () => {
    const p = P.find(q => q.k === el.dataset.k);
    let v = el.value;
    if (!p.opts) { v = +v / p.s; if (!isFinite(v)) return; } else if (!isNaN(+v) && typeof p.opts[0][0] === "number") v = +v;
    draft = { ...curD(), [p.k]: v };
    if (!isDraft()) draft = null;
    onDesignChange();
  });
}
function renderHead() {
  const x = sel;
  $("#dname").value = x.name; $("#dname").readOnly = x.src !== "team";
  $("#dfam").innerHTML = `<span class="dot" style="background:${famCol(x.fam)}"></span>${FAMNAME[x.fam] || x.fam}${isDraft() ? ` · <span style="color:var(--volt)">edited</span>` : ""}`;
  $("#dsave").hidden = !(x.src === "team" && isDraft());
  $("#dsavenew").textContent = x.src === "team" && !isDraft() ? "Duplicate" : "Save as new design";
  $("#ddel").hidden = x.src !== "team";
}
let axiTimer = null;
function renderView() {
  const x = sel, d = curD(), is3 = x.kind === "tabbed";
  $("#paneAxi").hidden = is3; $("#pane3d").hidden = !is3;
  if (!is3) {
    clearTimeout(axiTimer);
    axiTimer = setTimeout(() => {   // the fast model's profiles drive the axisymmetric view
      const xx = fastX(d), r = WCModel.simulate(xx, { ...MODEL_OPT, fit: FIT, profile: true });
      window.__wbLast = { x: xx, r }; window.View3D?.update(xx, r);
    }, 30);
  } else {
    const b = best(x, d), v = b && !b.stale ? b.r.view : null;
    d3show(x, d, v?.flow || "pre:on" + leanKey(d), v?.field || "pre:on" + leanKey(d), null).catch(e => toast(e.message));
  }
}

/* ======================= metrics + fidelity ======================= */
function draftFast(d) {
  const r = WCModel.simulate(fastX(d), { ...MODEL_OPT, fit: FIT });
  return { cond: condKey(cond), dk: dKey(d), m: fromFast(r), at: Date.now(), by: "" };
}
function renderSide() {
  const x = sel, d = curD();
  // the selected design's Fast result always reflects the current parameters
  if (x.kind === "disc" && !valid(x, "fast", d)) resOf(x, d).fast = draftFast(d);
  const b = best(x, d), m = b?.r.m;
  $("#ktier").textContent = b ? TK[b.t].badge + (b.stale ? " · stale" : "") : "not run";
  $("#ktier").className = "tier" + (b ? " t-" + b.t : "");
  const k = (n, v, s, cls = "") => `<div class="kpi"><span class="k">${n}</span><span class="v ${cls}">${v}</span><span class="s">${s}</span></div>`;
  $("#kpis").className = "kpis" + (!b || b.stale ? " stale" : "");
  $("#kpis").innerHTML = m ? k("Water captured", fmt.cap(m), isFinite(m.capLh) ? m.capLh.toFixed(m.capLh < 10 ? 2 : 1) + " L/h" : "") + k("Energy", num(m.Wh, 3), "Wh per litre") +
    k("Pressure drop", num(m.dp, 3), "Pa") + k("Arc margin", fmt.arc(m), m.arc < 0 ? "arcs" : "1 = far from sparkover", m.arc < 0 ? "bad" : "")
    : `<div class="kpi" style="grid-column:1/3"><span class="s">Run it at any fidelity below.</span></div>`;
  const ago = t => { const s = (Date.now() - t) / 1e3; return s < 90 ? "just now" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 129600 ? Math.round(s / 3600) + " h ago" : new Date(t).toLocaleDateString(); };
  $("#ladder").innerHTML = TIERS.map(t => {
    const av = tierAvail(x, t.k), run = running[x.id + "|" + t.k], r = valid(x, t.k, d), old = results[rkey(x, d)]?.[t.k];
    let st;
    if (!av) st = x.kind === "tabbed" ? "not for this electrode" : "coming soon";
    else if (run) st = run.note || "running";
    else if (r) st = t.k === "fast" ? `${fmt.cap(r.m)} · ${fmt.Wh(r.m)}` : `${fmt.cap(r.m)} · ${fmt.Wh(r.m)} · ${r.by === "library" ? "library" : ago(r.at)}${r.by && r.by !== "library" ? " · " + r.by.split("@")[0] : ""}`;
    else st = (old ? "out of date · " : "") + estimate(x, d, t);
    const btn = av && t.k !== "fast" && !run ? `<button type="button" class="btn${r ? "" : " primary"}" data-run="${t.k}">${r ? "Rerun" : "Run"}</button>` : "";
    return `<div class="rung${av ? "" : " na"}" data-t="${t.k}" aria-current="${b && b.t === t.k && !b.stale}"><span class="tn">${t.n}</span><span class="ts" title="${esc(st)}">${esc(st)}</span>${btn}` +
      (run ? `<span class="bar"><span style="width:${(100 * run.frac).toFixed(0)}%"></span></span>` : "") + "</div>";
  }).join("");
  $("#ladder").querySelectorAll("[data-run]").forEach(el => el.onclick = e => {
    e.stopPropagation();
    const t = el.dataset.run;
    if (t === "resolved") { if (Object.keys(running).some(k => k.endsWith("|resolved"))) return toast("a Resolved run is already going"); runResolved(x, d); }
    else run3D(x, d, t);
  });
  // details of the shown result
  let ex = "";
  if (m) {
    const rows = Object.entries(m.x).filter(([k, v]) => Array.isArray(v) && v[0] != null && isFinite(v[0]));
    ex += `<details><summary>More metrics</summary><table><tbody>${rows.map(([n, [v, u]]) => `<tr><td>${n}</td><td class="n">${num(v, 3)} ${u}</td></tr>`).join("")}</tbody></table></details>`;
  }
  if (b?.r.bySize) {
    const W = 280, row = 20, rows = b.r.bySize, H = row * rows.length + (x.measured ? 18 : 6), x0 = 44, sx = v => x0 + (W - x0 - 30) * Math.max(0, Math.min(1, v));
    ex += `<div class="capbars"><h2 style="margin-bottom:6px">Mist removed by drop size</h2><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mist removed by drop size">` +
      rows.map(([dd, v], i) => `<text x="0" y="${i * row + 13}" fill="#8a8a94" font-size="10">${dd.toFixed(0)} µm</text><rect x="${x0}" y="${i * row + 3}" width="${sx(v) - x0}" height="12" rx="2" fill="#5b9dff"/><text x="${sx(v) + 4}" y="${i * row + 13}" fill="#e8e8ec" font-size="10">${(100 * v).toFixed(0)}%</text>`).join("") +
      (x.measured ? `<line x1="${sx(x.measured)}" x2="${sx(x.measured)}" y1="0" y2="${H - 12}" stroke="#ffb547" stroke-dasharray="3 2"/><text x="${sx(x.measured)}" y="${H - 2}" fill="#ffb547" font-size="10" text-anchor="middle">bench ${(100 * x.measured).toFixed(0)}%</text>` : "") + "</svg></div>";
  }
  $("#extra").innerHTML = ex;
}
// list and map re-render at most once per frame however many results arrive
let listQueued = false;
function queueList() { if (listQueued) return; listQueued = true; requestAnimationFrame(() => { listQueued = false; renderList(); renderMap(); }); }
function render() { queueList(); if (sel) { renderHead(); renderSide(); } }
function onDesignChange() { renderHead(); renderSide(); renderView(); queueList(); }
function select(x) {
  if (!x) return;
  if (sel) delete results[sel.id + "::draft"];
  sel = x; draft = null;
  renderHead(); renderParams(); renderSide(); renderView(); queueList();
  try { localStorage.setItem("wc.sel", x.id); } catch (e) { /* convenience */ }
}

/* ======================= header actions ======================= */
$("#dname").onchange = e => { if (sel.src !== "team") return; sel.name = e.target.value.trim() || sel.name; putDesign(sel); renderList(); };
$("#dsavenew").onclick = async () => {
  const d = curD(), id = "u-" + Math.random().toString(36).slice(2, 10);
  const x = { id, name: (sel.src === "team" ? sel.name : sel.name) + (isDraft() ? " (edited)" : " (copy)"), fam: sel.fam, kind: sel.kind, d: { ...d }, src: "team", measured: sel.measured };
  // results already run for these exact parameters come along
  for (const [t, v] of Object.entries(results[rkey(sel, d)] || {})) if (v.dk === dKey(d)) (results[id] ||= {})[t] = v;
  designs.unshift(x); await putDesign(x);
  for (const [t, v] of Object.entries(results[id] || {})) if (t !== "fast") putResult(id, t, v);
  select(x); toast(remote ? "saved for the team" : "saved in this browser");
};
$("#dsave").onclick = async () => {
  const dr = results[rkey(sel, curD())] || {};
  sel.d = { ...curD() }; draft = null; results[sel.id] = dr; delete results[sel.id + "::draft"];
  await putDesign(sel); for (const [t, v] of Object.entries(dr)) if (t !== "fast") putResult(sel.id, t, v);
  select(sel); toast("saved");
};
$("#ddel").onclick = async () => {
  if (!$("#ddel").dataset.armed) { $("#ddel").dataset.armed = "1"; $("#ddel").textContent = "Delete?"; setTimeout(() => { delete $("#ddel").dataset.armed; $("#ddel").textContent = "Delete"; }, 2500); return; }
  const i = designs.indexOf(sel); designs.splice(i, 1); await delDesign(sel); select(designs[0]);
};
$("#s3layers").querySelectorAll("button").forEach(b => b.onclick = () => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", on); window.Sim3D?.setLayer(b.dataset.l, on); });

/* ======================= conditions ======================= */
function condText() {
  const c = cond;
  return (c.site ? `pilot site, ${c.site} · ` : "") + `${c.inlet_T_C.toFixed(0)} °C · ${c.inlet_RH >= 0.999 ? "saturated" : (100 * c.inlet_RH).toFixed(0) + "% RH"} · ${c.inlet_lwc_g_m3} g/m³ mist, ${c.mist_mmd_um} µm` + (c.seawater_T_C != null ? ` · sea ${c.seawater_T_C.toFixed(1)} °C` : "");
}
function renderCondPop() {
  const c = cond, F = [["inlet_T_C", "Inlet air", "°C", 1], ["inlet_RH", "Relative humidity", "%", 100], ["inlet_lwc_g_m3", "Mist loading", "g/m³", 1], ["mist_mmd_um", "Mist drop size (MMD)", "µm", 1], ["mist_gsd", "Mist size spread (GSD)", "", 1], ["ambient_T_C", "Outside air", "°C", 1]];
  $("#condpop").innerHTML = `<div class="lrow" style="justify-content:space-between"><h2>Conditions</h2><button type="button" class="btn" id="condx">Done</button></div>
    <label class="prow"><span>Site</span><select id="csite"><option value="">custom</option>${(SITE?.months || []).map(m => `<option${c.site === m ? " selected" : ""}>${m}</option>`).join("")}</select></label>
    ${F.map(([k, n, u, s]) => `<label class="prow"><span>${n} <span class="u">${u}</span></span><input type="number" data-c="${k}" data-s="${s}" value="${+(c[k] * s).toPrecision(4)}" step="any"></label>`).join("")}
    <label class="prow"><span>Seawater cooling <span class="u">°C</span></span><input type="number" id="csea" placeholder="none" value="${c.seawater_T_C ?? ""}" step="0.1"></label>
    <div class="lrow"><button type="button" class="btn" id="creset">Library conditions</button></div>`;
  $("#condpop").querySelectorAll("[data-c]").forEach(el => el.onchange = () => setCond({ ...cond, [el.dataset.c]: +el.value / +el.dataset.s, site: "" }));
  $("#csea").onchange = e => setCond({ ...cond, seawater_T_C: e.target.value === "" ? null : +e.target.value });
  $("#csite").onchange = e => {
    const m = e.target.value; if (!m) return setCond({ ...cond, site: "" });
    const s = SITE.climate[m]; setCond({ ...cond, site: m, inlet_T_C: s.exhaust_T_C, inlet_RH: 1, ambient_T_C: s.air_T_C, seawater_T_C: s.sea["20m"] });
  };
  $("#creset").onclick = () => setCond({ ...COND0 });
  $("#condx").onclick = () => { $("#condpop").hidden = true; };
}
async function setCond(c) {
  cond = c; try { localStorage.setItem("wc.cond", JSON.stringify(c)); } catch (e) { /* convenience */ }
  $("#condtxt").textContent = condText(); renderCondPop();
  await fastAll(); render(); if (sel) renderView();
}
$("#condbtn").onclick = () => { const p = $("#condpop"); p.hidden = !p.hidden; if (!p.hidden) renderCondPop(); };

/* ======================= start ======================= */
(async () => {
  lst.fams = new Set(famsPresent());
  $("#condtxt").textContent = condText();
  const st = await storeLoad();
  for (const x of st.designs || []) designs.unshift({ ...x, src: "team", measured: x.fam === "tabbed_ring" ? 1 - 3.73 / 10.63 : undefined });
  for (const [id, t] of Object.entries(st.results || {})) for (const [k, v] of Object.entries(t)) (results[id] ||= {})[k] = v;
  $("#storetxt").textContent = remote ? "team designs are shared" : "designs save in this browser";
  engine = await makeEngine();
  await fastAll();
  let first = null; try { first = designs.find(x => x.id === localStorage.getItem("wc.sel")); } catch (e) { /* convenience */ }
  select(first || designs.find(x => x.id === "tabbed-ring") || designs[0]);
})().catch(e => toast("could not start: " + e.message, 10000));

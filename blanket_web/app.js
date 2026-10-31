// Blanket Simulator page. Engine: StoneLab apps/blanket (bundled as blanket-engine.js);
// data: packs/ (OpenMC-derived cross sections) and designs.json (library + OpenMC
// references), both staged by build.py.
import { BlanketEngine, ARC_DESIGN, ARC_DEFAULTS, REGIONS, regionRaster, FLIBE_SOLUBILITY } from "./blanket-engine.js";

const $ = (id) => document.getElementById(id);
const FAST = 100_000, RESOLVED = 8_000_000;
const LABEL = { plasma: "Plasma", fw: "First wall", vv_in: "Inner vessel", channel: "Cooling channel", be: "Be multiplier", vv_out: "Outer vessel", blanket: "FLiBe tank", tank: "Tank wall", shield: "TiH₂ shield", coil: "TF coil", bore: "Bore" };
const MATERIAL = { fw: null, vv_in: "Inconel 718", channel: "FLiBe", be: "beryllium", vv_out: "Inconel 718", blanket: "FLiBe", tank: "Inconel 718", shield: "TiH₂", coil: "coil pack", plasma: "D-T plasma" };
const FW_NAME = { W: "tungsten", Be: "beryllium", CFC: "carbon fibre", Inconel: "Inconel 718" };
const COLORS = { plasma: [58, 44, 22], fw: [150, 156, 176], vv_in: [104, 110, 122], channel: [79, 179, 217], be: [126, 196, 140], vv_out: [104, 110, 122], blanket: [37, 98, 125], tank: [84, 88, 98], shield: [196, 128, 64], coil: [110, 78, 58], bore: [15, 15, 18] };

// ---- parameters
const P = (group, key, label, unit, step, min, max) => ({ group, key, label, unit, step, min, max });
const PARAMS = [
  P("Vessel", "g.fw_cm", "First wall", "cm", 0.1, 0, 5),
  P("Vessel", "fwMaterial", "Wall material", "", null),
  P("Vessel", "g.vv_in_cm", "Inner vessel", "cm", 0.1, 0.2, 5),
  P("Vessel", "g.channel_cm", "Cooling channel", "cm", 0.5, 0.5, 10),
  P("Vessel", "g.be_cm", "Be multiplier", "cm", 0.5, 0, 6),
  P("Vessel", "g.vv_out_cm", "Outer vessel", "cm", 0.5, 0.5, 8),
  P("Breeder", "li6Pct", "Li-6 enrichment", "%", null),
  P("Breeder", "solubility", "H solubility data", "", null),
  P("Breeder", "g.blanket_in_cm", "Tank, inboard", "cm", 5, 0, 60),
  P("Breeder", "g.blanket_out_cm", "Tank, outboard", "cm", 5, 10, 250),
  P("Breeder", "g.blanket_vert_cm", "Tank, top/bottom", "cm", 5, 10, 200),
  P("Machine", "g.R0", "Major radius (wall)", "cm", 1, 250, 500),
  P("Machine", "g.a", "Minor radius (wall)", "cm", 1, 80, 160),
  P("Machine", "g.kappa", "Elongation", "", 0.02, 1, 2.2),
  P("Machine", "fusionPowerMW", "Fusion power", "MW", 25, 50, 2000),
  P("Machine", "g.shield_in_cm", "Inboard shield", "cm", 1, 10, 100),
  P("Cooling", "channelVelocity", "Channel velocity", "m/s", 0.1, 0.2, 10),
  P("Cooling", "bulkFlow", "Tank flow", "m³/s", 0.05, 0.1, 10),
  P("Cooling", "inletK", "Inlet temperature", "K", 5, 740, 950),
  P("Cooling", "qPlasma", "Plasma heat on wall", "MW/m²", 0.05, 0, 2),
];
const get = (d, key) => key.startsWith("g.") ? (d.geometry[key.slice(2)] ?? ARC_DEFAULTS[key.slice(2)]) : d[key];
const set = (d, key, v) => { if (key.startsWith("g.")) d.geometry[key.slice(2)] = v; else d[key] = v; };
const fullDesign = (partial) => ({ ...structuredClone(ARC_DESIGN), ...structuredClone(partial ?? {}), geometry: { ...(partial?.geometry ?? {}) } });

// ---- state
let engine = null, lib = [], saved = [], cur = null, field = "heat", raster = null, runToken = 0;
const results = new Map(); // design json -> {fast, resolved}
const keyOf = (d) => JSON.stringify(d, Object.keys(d).sort()) + JSON.stringify(d.geometry, Object.keys(d.geometry).sort());
try { saved = JSON.parse(localStorage.getItem("blanket.designs") ?? "[]"); } catch { saved = []; }
const persist = () => { try { localStorage.setItem("blanket.designs", JSON.stringify(saved)); } catch { /* private mode */ } };

// ---- formatting
const f = (x, n = 2) => Number.isFinite(x) ? x.toFixed(n) : "—";
const sig = (x, n = 3) => !Number.isFinite(x) ? "—" : Math.abs(x) >= 1e6 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(n - 1).replace(/e\+?(-?)(\d+)/, (_, s, e) => `×10${sup(s + e)}`) : Number(x.toPrecision(n)).toLocaleString();
const sup = (t) => [...t].map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(c)] ?? (c === "-" ? "⁻" : c)).join("");
/** TF life is a deep-penetration tally: only show it when its standard error is under 30%. */
const tfOk = (m) => Number.isFinite(m.tfLifetimeFpy) && Number.isFinite(m.tfRelErr) && m.tfRelErr < 0.3 && m.histories >= 1e6;

// ---- designs list
function renderList() {
  const groups = new Map();
  for (const it of [...saved.map((s) => ({ ...s, family: "Your designs" })), ...lib]) {
    if (!groups.has(it.family)) groups.set(it.family, []);
    groups.get(it.family).push(it);
  }
  const L = $("list");
  L.innerHTML = "";
  for (const [fam, items] of groups) {
    const h = document.createElement("div"); h.className = "gtitle"; h.innerHTML = `<h2>${fam}</h2>`; L.append(h);
    for (const it of items) {
      const el = document.createElement("div");
      el.className = "item"; el.setAttribute("role", "option"); el.dataset.id = it.id;
      el.setAttribute("aria-selected", String(cur?.id === it.id));
      const r = results.get(keyOf(fullDesign(it.design)));
      const m = r?.resolved ?? r?.fast;
      el.innerHTML = `<span class="nm">${it.name}</span>${it.reference ? '<span class="ref" title="OpenMC continuous-energy reference available">REF</span>' : "<span></span>"}
        <span class="mx">${m ? `<span>TBR <b>${f(m.tbr, 3)}</b></span><span>VV <b>${f(m.thermal.vvMax, 0)}</b> K</span>${tfOk(m) ? `<span>TF <b>${f(m.tfLifetimeFpy, 1)}</b> y</span>` : ""}` : ""}</span>`;
      el.onclick = () => select(it);
      L.append(el);
    }
  }
}

function select(it) {
  cur = { id: it.id, name: it.name, base: structuredClone(it.design ?? {}), design: fullDesign(it.design), reference: it.reference ?? null, saved: it.family === "Your designs" || saved.some((s) => s.id === it.id) };
  $("dname").value = cur.name;
  $("ddel").hidden = !cur.saved;
  renderParams(); renderList(); redraw(); schedule();
}

// ---- parameters
function renderParams() {
  const box = $("params"); box.innerHTML = "";
  const base = fullDesign(cur.base);
  for (const g of [...new Set(PARAMS.map((p) => p.group))]) {
    const grp = document.createElement("div"); grp.className = "pgroup"; grp.innerHTML = `<h2>${g}</h2>`;
    for (const p of PARAMS.filter((q) => q.group === g)) {
      const row = document.createElement("label"); row.className = "prow";
      const v = get(cur.design, p.key);
      if (v !== get(base, p.key)) row.classList.add("changed");
      let input;
      if (p.key === "fwMaterial") {
        input = document.createElement("select");
        for (const m of engine.fwMaterials()) input.add(new Option(FW_NAME[m] ?? m, m, false, m === v));
        input.onchange = () => update(p.key, input.value);
      } else if (p.key === "solubility") {
        input = document.createElement("select");
        for (const [k, a] of Object.entries(FLIBE_SOLUBILITY)) input.add(new Option(a.source.split(",")[0], k, false, k === v));
        input.onchange = () => update(p.key, input.value);
      } else if (p.key === "li6Pct") {
        input = document.createElement("select");
        for (const e of engine.enrichments()) input.add(new Option(`${e}%`, String(e), false, e === v));
        input.onchange = () => update(p.key, Number(input.value));
      } else {
        input = document.createElement("input"); input.type = "number"; input.step = String(p.step); input.min = String(p.min); input.max = String(p.max); input.value = String(v);
        input.onchange = () => { const x = Number(input.value); if (Number.isFinite(x)) update(p.key, Math.min(p.max, Math.max(p.min, x))); };
      }
      row.innerHTML = `<span>${p.label}${p.unit ? ` <span class="u">${p.unit}</span>` : ""}</span>`;
      row.append(input); grp.append(row);
    }
    box.append(grp);
  }
}

function update(key, v) {
  set(cur.design, key, v);
  renderParams(); redraw(); schedule();
}

// ---- runs: Fast auto-runs on every change; Resolved on request
let timer = 0;
function schedule() { clearTimeout(timer); timer = setTimeout(() => run("fast"), 120); }

async function run(tier) {
  const token = ++runToken;
  const d = structuredClone(cur.design), k = keyOf(d);
  const r = results.get(k) ?? {};
  if (r[tier]) { results.set(k, r); show(); return; }
  const N = tier === "fast" ? FAST : RESOLVED;
  setRung(tier, 0, "running…");
  const t0 = performance.now();
  try {
    const m = await engine.evaluate(d, N, (mm) => {
      if (token !== runToken) return false;
      setRung(tier, mm.histories / N, `${sig(mm.histories, 2)} histories`);
      if (tier === "resolved") { r.live = mm; results.set(k, r); show(mm); }
    }, tier === "fast" ? 1 : 7);
    if (token !== runToken) return;
    m.seconds = (performance.now() - t0) / 1000;
    r[tier] = m; delete r.live; results.set(k, r);
    show(); renderList();
  } catch (e) { console.error(e); setRung(tier, 0, `failed: ${e.message}`); }
}

function setRung(tier, frac, text) {
  const el = document.querySelector(`.rung[data-t="${tier}"]`);
  if (!el) return;
  el.querySelector(".bar span").style.width = `${Math.round(frac * 100)}%`;
  el.querySelector(".ts").textContent = text;
}

// ---- metrics
function current() {
  const r = results.get(keyOf(cur.design));
  return { r, m: r?.resolved ?? r?.live ?? r?.fast, tier: r?.resolved ? "resolved" : r?.live ? "resolved" : r?.fast ? "fast" : null };
}

function show() {
  if (!cur) return;
  const { r, m, tier } = current();
  const K = $("kpis");
  if (!m) { K.classList.add("stale"); }
  else {
    K.classList.remove("stale");
    const cls = (v, good, warn, hiGood = true) => !Number.isFinite(v) ? "" : hiGood ? (v >= good ? "good" : v >= warn ? "warn" : "bad") : (v <= good ? "good" : v <= warn ? "warn" : "bad");
    const kpi = (k, v, c, s) => `<div class="kpi"><span class="k">${k}</span><span class="v ${c}">${v}</span><span class="s">${s}</span></div>`;
    K.innerHTML = [
      kpi("Tritium breeding ratio", f(m.tbr, m.tbrErr < 0.001 ? 4 : 3), cls(m.tbr, 1.1, 1.0), `± ${f(m.tbrErr, m.tbrErr < 0.001 ? 4 : 3)} · target ≥ 1.1`),
      m.thermal2d
        ? kpi("Peak inner vessel", `${f(m.thermal2d.peaks.inconelInner.K, 0)} K`, cls(m.thermal2d.peaks.inconelInner.K, 973, 1073, false), `Inconel 718, 2-D field · at R ${f(m.thermal2d.peaks.inconelInner.R, 2)} m, Z ${f(m.thermal2d.peaks.inconelInner.Z, 2)} m · tungsten face ${f(m.thermal2d.peaks.w.K, 0)} K · 1-D ${f(m.thermal.vvMax, 0)} K`)
        : kpi("Peak inner vessel", `${f(m.thermal.vvMax, 0)} K`, cls(m.thermal.vvMax, 973, 1073, false), `Inconel 718, 1-D estimate · channel out ${f(m.thermal.channelOutlet, 0)} K`),
      tfOk(m)
        ? kpi("TF magnet life", `${f(m.tfLifetimeFpy, 1)} y`, cls(m.tfLifetimeFpy, 10, 3), `${Number.isFinite(m.tfRelErr) ? `± ${Math.round(m.tfRelErr * 100)}% · ` : ""}full-power years to 3×10¹⁸ n/cm² on the inboard face`)
        : kpi("TF magnet life", "—", "", `too few neutrons reach the coil face in ${sig(m.histories, 2)} histories: run Resolved`),
      m.tritium
        ? kpi("Tritium in the salt", `${f(m.tritium.inventoryMg, 0)} mg`, "", `${f(m.tritiumMgPerS, 2)} mg/s bred · limits ${f(m.inventoryPlugMg, 0)}–${f(m.inventoryMixedMg, 0)} mg · wall loss ${permBand(m)}`)
        : kpi("Tritium in the salt", `${f(m.inventoryPlugMg, 0)}–${f(m.inventoryMixedMg, 0)} mg`, "", `${f(m.tritiumMgPerS, 2)} mg/s bred · ${f(m.residenceS / 60, 1)} min turnover`),
      m.damage
        ? kpi("Vessel helium limit", `${f(500 / m.damage.vvInPeakHeAppmPerFpy, 1)} y`, cls(500 / m.damage.vvInPeakHeAppmPerFpy, 5, 1.5), `to 500 appm He at the inner vessel's peak (${f(m.damage.vvInPeakHeAppmPerFpy, 0)} appm/y) · ${f(m.damage.vvInPeakDpaPerFpy, 1)} dpa/y`)
        : "",
      kpi("Energy multiplication", f(m.energyMultiplication, 3), "", `${f(m.nuclearMW, 0)} MW nuclear heat`),
      kpi("Tank outlet", `${f(m.thermal.bulkOutlet, 0)} K`, cls(m.thermal.bulkOutlet, 1000, 1100, false), `inlet ${f(cur.design.inletK, 0)} K · FLiBe freezes at 732 K`),
    ].join("");
  }
  // ladder
  const ref = refFor();
  const L = $("ladder");
  const rung = (t, name, desc, res, extra) => `<button type="button" class="rung ${extra ?? ""}" data-t="${t}" aria-current="${tier === t}"><span class="tn">${name}</span><span class="ts">${res ?? desc}</span><span class="tr">${t === "reference" ? "OpenMC" : "browser"}</span><span class="bar"><span style="width:${res ? 100 : 0}%"></span></span></button>`;
  L.innerHTML = [
    rung("fast", "Fast", `${sig(FAST, 2)} histories`, r?.fast ? `TBR ${f(r.fast.tbr, 3)} ± ${f(r.fast.tbrErr, 3)} · ${f(r.fast.seconds, 2)} s` : null),
    rung("resolved", "Resolved", `${RESOLVED / 1e6}M histories · click to run`, r?.resolved ? `TBR ${f(r.resolved.tbr, 4)} ± ${f(r.resolved.tbrErr, 4)} · ${f(r.resolved.seconds, 1)} s` : null),
    rung("reference", "Reference", "continuous-energy OpenMC: library designs only", ref ? `TBR ${f(ref.TBR, 4)} ± ${f(ref.TBR_sd, 4)}` : null, ref ? "" : "na"),
  ].join("");
  L.querySelector('[data-t="resolved"]').onclick = () => run("resolved");
  // heating bars
  const H = $("heat");
  if (m) {
    const rows = ["fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield"].filter((k) => m.heatingMW[k] !== undefined);
    const max = Math.max(...rows.map((k) => m.heatingMW[k]));
    H.innerHTML = rows.map((k) => {
      const v = m.heatingMW[k], rv = ref?.heating_MW?.[k];
      const c = COLORS[k];
      return `<div class="brow"><span>${LABEL[k]}</span><span class="track"><span style="width:${(100 * v) / max}%;background:rgb(${c})"></span>${rv !== undefined ? `<em style="left:${Math.min(100, (100 * rv) / max)}%" title="OpenMC ${f(rv, 1)} MW"></em>` : ""}</span><span class="n">${f(v, v < 10 ? 2 : 1)} MW</span></div>`;
    }).join("") + (ref?.heating_MW ? `<div class="note">Ticks: OpenMC reference.</div>` : "");
  }
  $("notes").innerHTML = notes(m);
  drawMap(m);
  if (view3d && m && !$("v3").hidden) view3d.setLive(m, cur.design);
}

/** Tank-wall permeation across the four FLiBe solubility sources, % of bred tritium. */
function permBand(m) {
  const v = Object.values(m.permeationBySolubility ?? {});
  if (!v.length) return "—";
  return `${(Math.min(...v) * 100).toPrecision(2)}–${(Math.max(...v) * 100).toPrecision(2)}%`;
}

function refFor() {
  if (!cur?.reference) return null;
  return keyOf(cur.design) === keyOf(fullDesign(cur.base)) ? cur.reference : null;
}

function notes(m) {
  const fw = cur.design.fwMaterial;
  const parts = [`Cross sections: OpenMC ENDF/B-VIII.0, 175 groups, weighted on a ${FW_NAME[fw] ?? fw}-wall design at ${engine.enrichments().reduce((a, b) => Math.abs(b - cur.design.li6Pct) < Math.abs(a - cur.design.li6Pct) ? b : a)}% Li-6.`];
  parts.push("Vessel temperature: 2-D (R, Z) conduction through the tile, vessel walls and multiplier with the cooling channel's film coefficient, driven by this run's heating map (the 1-D Sorbom eqs. 30–33 estimate is shown beside it).");
  if (m?.damage) parts.push("Vessel damage: NRT dpa and He production from OpenMC group responses (ENDF/B-VIII MT444, MT207); the peak is the average times OpenMC's peak-sector ratio. He, not dpa, sets the limit here (Sorbom's 500 appm criterion). Our dpa is 2–3× below the ARC papers' (which do not state their dpa method); our He agrees with them within ~15%.");
  if (m?.thermal2d?.warnings?.length) parts.push(`<span style="color:var(--warn)">${m.thermal2d.warnings.join("<br>")}</span>`);
  parts.push("Tritium in the salt: steady transport through the tank (potential flow, mixing-length turbulence) from the bred-tritium map, clean salt at the inlet, Sieverts permeation through the Inconel tank wall. The FLiBe solubility data disagree by ~3 decades, hence the band.");
  if (m) parts.push(`${sig(m.histories, 2)} histories · ${engine.device ? "WebGPU" : ""}`);
  return parts.join("<br>");
}

// ---- cross-section
function redraw() {
  raster = regionRaster(cur.design, 2);
  const used = new Set(raster.ids);
  $("keys").innerHTML = REGIONS.map((r, i) => used.has(i) && r !== "bore" ? `<span><i style="background:rgb(${COLORS[r]})"></i>${r === "fw" ? `First wall (${FW_NAME[cur.design.fwMaterial] ?? cur.design.fwMaterial})` : LABEL[r]}</span>` : "").join("");
  show();
}

const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const cmap = (t) => { t = Math.max(0, Math.min(1, t)) * (INFERNO.length - 1); const i = Math.min(INFERNO.length - 2, Math.floor(t)), u = t - i; return INFERNO[i].map((a, k) => a + (INFERNO[i + 1][k] - a) * u); };

let view = null; // {sx, ox, oy, ...} for hover
function drawMap(m) {
  const cv = $("cv"), box = $("view");
  const dpr = devicePixelRatio || 1;
  cv.width = box.clientWidth * dpr; cv.height = box.clientHeight * dpr;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0f0f12"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!raster) return;
  const { nr, nz, ids, h, z0 } = raster;
  // fit the R >= Rmin part (skip most of the bore)
  const g = raster.geom;
  const rLo = Math.max(0, Math.floor((g.coil.rmin - 20) / h));
  const w = nr - rLo, pad = 30 * dpr;
  const s = Math.min((cv.width - 2 * pad) / w, (cv.height - 2 * pad) / nz);
  const ox = (cv.width - w * s) / 2, oy = (cv.height - nz * s) / 2;
  view = { s, ox, oy, rLo, dpr, m };
  const img = new ImageData(w, nz);
  // density map in W/cm3 (heat) or atoms/s/cm3 (tritium)
  let dens = null, lo = 0, hi = 1;
  if (field === "temp" && m?.thermal2d?.field) {
    const F = m.thermal2d.field, G = F.grid;
    dens = new Float32Array(F.T.length);
    let mn = Infinity, mx = -Infinity;
    for (let k = 0; k < dens.length; k++) { const v = F.T[k]; dens[k] = Number.isFinite(v) ? v : 0; if (Number.isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); } }
    lo = mn; hi = mx; view.dens = dens; view.G = G; view.linear = true; view.offset = mn;
  } else if (field === "conc" && m?.tritium) {
    view.offset = 0;
    const T = m.tritium, G = T.grid;
    dens = new Float32Array(G.nr * G.nz);
    let mx = 0;
    const toMg = 3.01605e3 / 6.02214076e23;
    for (let k = 0; k < dens.length; k++) { const v = Number.isFinite(T.c[k]) ? T.c[k] * toMg : 0; dens[k] = v; if (v > mx) mx = v; }
    lo = 0; hi = mx; view.dens = dens; view.G = G; view.linear = true;
  } else if (m?.map && m.mapGrid && field !== "none" && field !== "conc") {
    view.linear = false;
    const G = m.mapGrid, off = field === "heat" ? 0 : 1, conv = field === "heat" ? m.sourceRate * 1.602176634e-19 : m.sourceRate;
    dens = new Float32Array(G.nr * G.nz);
    let mx = 0;
    for (let j = 0; j < G.nz; j++) for (let i = 0; i < G.nr; i++) {
      const R = G.r0 + (i + 0.5) * G.h;
      const v = (m.map[2 * (j * G.nr + i) + off] * conv) / (2 * Math.PI * R * G.h * G.h);
      dens[j * G.nr + i] = v; if (v > mx) mx = v;
    }
    hi = Math.log10(mx); lo = hi - 3.5;
    view.dens = dens; view.G = G;
  }
  for (let j = 0; j < nz; j++) for (let i = rLo; i < nr; i++) {
    const id = ids[j * nr + i];
    const o = 4 * ((nz - 1 - j) * w + (i - rLo));
    let c = id >= 0 ? COLORS[REGIONS[id]] : [15, 15, 18];
    if (dens && id >= 1 && id <= 6) {
      const G = view.G, R = i * h + h / 2, Z = z0 + j * h + h / 2;
      const gi = Math.floor((R - G.r0) / G.h), gj = Math.floor((Z - G.z0) / G.h);
      const v = gi >= 0 && gj >= 0 && gi < G.nr && gj < G.nz ? dens[gj * G.nr + gi] : 0;
      c = view.linear ? (v > 0 ? cmap((v - (view.offset ?? 0)) / (hi - (view.offset ?? 0))) : (id === 6 && field === "conc" ? cmap(0) : COLORS[REGIONS[id]].map((x) => x * 0.45))) : v > 0 ? cmap((Math.log10(v) - lo) / (hi - lo)) : [12, 12, 14];
    } else if (dens && id >= 0) c = c.map((x) => x * 0.45);
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  const tmp = new OffscreenCanvas(w, nz); tmp.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, ox, oy, w * s, nz * s);
  if (field === "temp" && m?.thermal2d) drawStations(ctx, m.thermal2d, 16 * dpr, 70 * dpr, Math.max(120 * dpr, ox - 40 * dpr), Math.min(260 * dpr, cv.height * 0.45), dpr);
  // axis tick: major radius
  ctx.fillStyle = "#55555e"; ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  ctx.fillText(`R = ${(g.p.R0 / 100).toFixed(2)} m`, ox + ((g.p.R0 / h) - rLo) * s - 30 * dpr, oy + nz * s + 18 * dpr);
  const L = $("legend");
  if (dens) {
    L.hidden = false;
    const grad = `linear-gradient(90deg, ${INFERNO.map((c) => `rgb(${c})`).join(",")})`;
    const ttl = { heat: "Nuclear heating, W/cm³", trit: "Tritium production, atoms/s/cm³", conc: "Tritium in the salt, mg/m³ (steady state)", temp: "Vessel temperature, K (2-D field)" }[field];
    const [a, b] = view.linear ? [lo, hi] : [10 ** lo, 10 ** hi];
    L.innerHTML = `<span>${ttl}</span><span><i style="background:${grad}"></i></span><span style="display:flex;justify-content:space-between"><span>${sig(a, 2)}</span><span>${sig(b, 2)}</span></span>`;
  } else L.hidden = true;
}

/** Inset: temperatures along the first wall from the channel inlet (blanket-thermal stations). */
function drawStations(ctx, t, x0, y0, w, h, dpr) {
  const S = t.stations, n = S.t.length;
  if (!n || w < 100 * dpr) return;
  const series = [["W face", S.wFaceK, "rgb(250,193,39)"], ["inner Inconel", S.inconelPeakK, "rgb(212,72,66)"], ["coolant", S.bulkK, "rgb(79,179,217)"]];
  let lo = Infinity, hi = -Infinity;
  for (const [, a] of series) for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  lo = Math.floor(lo / 50) * 50; hi = Math.ceil(hi / 50) * 50;
  const X = (k) => x0 + 34 * dpr + (k / (n - 1)) * (w - 40 * dpr), Y = (v) => y0 + 14 * dpr + (1 - (v - lo) / (hi - lo)) * (h - 34 * dpr);
  ctx.save();
  ctx.fillStyle = "rgba(22,22,26,0.85)"; ctx.strokeStyle = "#2a2a30"; ctx.lineWidth = dpr;
  ctx.beginPath(); ctx.roundRect(x0, y0 - 6 * dpr, w, h + 18 * dpr, 8 * dpr); ctx.fill(); ctx.stroke();
  ctx.font = `${10.5 * dpr}px ui-monospace, monospace`; ctx.fillStyle = "#8a8a94";
  ctx.fillText("along the first wall, from the channel inlet", x0 + 8 * dpr, y0 + 6 * dpr);
  for (const v of [lo, (lo + hi) / 2, hi]) { ctx.fillText(`${v}`, x0 + 4 * dpr, Y(v) + 3 * dpr); ctx.strokeStyle = "#24242a"; ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(n - 1), Y(v)); ctx.stroke(); }
  series.forEach(([lbl, a, c], si) => {
    ctx.strokeStyle = c; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    for (let k = 0; k < n; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(a[k]));
    ctx.stroke();
    ctx.fillStyle = c; ctx.fillRect(x0 + 34 * dpr + si * 118 * dpr, y0 + h + 2 * dpr, 8 * dpr, 8 * dpr);
    ctx.fillStyle = "#a3a3ad"; ctx.fillText(lbl, x0 + 46 * dpr + si * 118 * dpr, y0 + h + 10 * dpr);
  });
  ctx.restore();
}

$("cv").addEventListener("mousemove", (e) => {
  if (!view || !raster) return;
  const rect = $("cv").getBoundingClientRect();
  const x = (e.clientX - rect.left) * view.dpr, y = (e.clientY - rect.top) * view.dpr;
  const i = Math.floor((x - view.ox) / view.s) + view.rLo, jj = Math.floor((y - view.oy) / view.s), j = raster.nz - 1 - jj;
  const tip = $("tip");
  if (i < view.rLo || i >= raster.nr || j < 0 || j >= raster.nz) { tip.hidden = true; return; }
  const id = raster.ids[j * raster.nr + i];
  if (id < 0) { tip.hidden = true; return; }
  const r = REGIONS[id], R = (i + 0.5) * raster.h, Z = raster.z0 + (j + 0.5) * raster.h;
  let val = "";
  if (view.dens && id >= 1 && id <= 6) {
    const G = view.G, gi = Math.floor((R - G.r0) / G.h), gj = Math.floor((Z - G.z0) / G.h);
    const v = view.dens[gj * G.nr + gi];
    if (v > 0) val = ` · <b>${field === "temp" ? f(v, 0) : sig(v, 3)}</b> ${{ heat: "W/cm³", trit: "T/s/cm³", conc: "mg/m³", temp: "K" }[field]}`;
  }
  const mat = r === "fw" ? FW_NAME[cur.design.fwMaterial] : MATERIAL[r];
  tip.innerHTML = `<b>${LABEL[r]}</b>${mat ? ` (${mat})` : ""} · R ${(R / 100).toFixed(2)} m, Z ${(Z / 100).toFixed(2)} m${val}`;
  tip.hidden = false;
  tip.style.left = `${e.clientX - rect.left + 12}px`; tip.style.top = `${e.clientY - rect.top + 12}px`;
});
$("cv").addEventListener("mouseleave", () => { $("tip").hidden = true; });
// 3-D cutaway (StoneLab apps/blanket-view), loaded on first use and fed every result
let view3d = null, view3dLoading = null;
async function show3d(on) {
  $("v3").hidden = !on;
  $("view").classList.toggle("in3d", on);
  for (const id of ["cv", "keys", "legend"]) $(id).style.visibility = on ? "hidden" : "";
  if (!on) return;
  if (!view3d && !view3dLoading) {
    view3dLoading = (async () => {
      const { createBlanketView } = await import("./blanket-view.js");
      const reference = await (await fetch("./arc_result.json")).json();
      view3d = await createBlanketView($("v3"), { device: engine.device, reference, chrome: "bare", field: "mapHeat" });
    })().catch((e) => { $("v3").innerHTML = `<div class="err">3-D view failed: ${e.message}</div>`; });
  }
  await view3dLoading;
  const { m } = current();
  if (view3d && m) view3d.setLive(m, cur.design);
}

$("layers").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.f === "3d") {
    const on = b.getAttribute("aria-pressed") !== "true";
    b.setAttribute("aria-pressed", String(on));
    show3d(on);
    return;
  }
  if ($("v3").hidden === false) { show3d(false); $("layers").querySelector('[data-f="3d"]').setAttribute("aria-pressed", "false"); }
  field = b.dataset.f;
  for (const x of $("layers").querySelectorAll("button")) if (x.dataset.f !== "3d") x.setAttribute("aria-pressed", String(x === b));
  show();
});
new ResizeObserver(() => show()).observe($("view"));

// ---- design actions
$("dname").onchange = () => { cur.name = $("dname").value; if (cur.saved) { const s = saved.find((x) => x.id === cur.id); if (s) { s.name = cur.name; persist(); renderList(); } } };
$("dreset").onclick = () => { cur.design = fullDesign(cur.base); renderParams(); redraw(); schedule(); };
$("dsave").onclick = () => {
  const it = { id: `my-${Date.now()}`, name: `${cur.name}${cur.saved ? "" : " (edited)"}`, family: "Your designs", design: structuredClone(cur.design), reference: null };
  saved.unshift(it); persist(); select(it);
};
$("ddel").onclick = () => { saved = saved.filter((s) => s.id !== cur.id); persist(); select(lib[0]); };

// ---- boot
(async () => {
  try {
    engine = await BlanketEngine.create("./packs");
  } catch (e) {
    document.querySelector(".work").innerHTML = `<div class="err"><b>WebGPU is needed.</b><br>${e.message}<br><br>Chrome or Edge 113+ on a machine with a GPU.</div>`;
    return;
  }
  $("gpu").textContent = "WebGPU";
  lib = await (await fetch("./designs.json")).json();
  select(lib[0]);
  // fill in the list's numbers in the background, one design at a time
  for (const it of lib.slice(1)) {
    const d = fullDesign(it.design), k = keyOf(d);
    if (results.get(k)?.fast) continue;
    try { const m = await engine.evaluate(d, FAST, undefined, 1); results.set(k, { ...(results.get(k) ?? {}), fast: m }); renderList(); } catch (e) { console.warn(it.name, e); }
  }
})();

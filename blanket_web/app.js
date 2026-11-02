// Blanket Simulator page. Engine: StoneLab apps/blanket (bundled as blanket-engine.js);
// data: packs/ (OpenMC-derived cross sections) and designs.json (library + OpenMC
// references), both staged by build.py.
import { BlanketEngine, ARC_DESIGN, ARC_DEFAULTS, CONFORMAL_DEFAULTS, REGIONS, regionRaster, conformalSlice, FLIBE_SOLUBILITY } from "./blanket-engine.js";

const $ = (id) => document.getElementById(id);
const FAST = 100_000, RESOLVED = 8_000_000;
const LABEL = { plasma: "Plasma", fw: "First wall", vv_in: "Inner vessel", channel: "Cooling channel", be: "Be multiplier", vv_out: "Outer vessel", blanket: "FLiBe tank", tank: "Tank wall", shield: "TiH₂ shield", coil: "TF coil", bore: "Bore" };
const MATERIAL = { fw: null, vv_in: "Inconel 718", channel: "FLiBe", be: "beryllium", vv_out: "Inconel 718", blanket: "FLiBe", tank: "Inconel 718", shield: "TiH₂", coil: "coil pack", plasma: "D-T plasma" };
const FW_NAME = { W: "tungsten", Be: "beryllium", CFC: "carbon fibre", Inconel: "Inconel 718" };
// breeder concepts (StoneLab neutron/design.ts BREEDERS) and the coolant settings each starts
// from: inlet above its melting point, flows that carry about the same heat
const BREEDER_NAME = { flibe: "FLiBe", flinak: "FLiNaK", pbli: "PbLi", li: "lithium", hcpb: "He-cooled pebble bed" };
const BREEDER_PRESET = {
  flibe: { li6Pct: 90, inletK: 800, channelVelocity: 2, bulkFlow: 1.08 },
  flinak: { li6Pct: 90, inletK: 800, channelVelocity: 2, bulkFlow: 1.08 },
  pbli: { li6Pct: 90, inletK: 600, channelVelocity: 0.5, bulkFlow: 1.1 },
  li: { li6Pct: 7.5, inletK: 600, channelVelocity: 1, bulkFlow: 1.0 },
  hcpb: { li6Pct: 60, inletK: 573, channelVelocity: 60, bulkFlow: 70 },
};
const breederOf = (d) => d?.breeder ?? "flibe";
const isStell = (d) => d?.machine === "stellarator";
/** Layer label for the current design: the breeder names the blanket. */
function lbl(k, d = cur?.design) {
  if (k === "blanket") return `${BREEDER_NAME[breederOf(d)]} ${isStell(d) ? "blanket" : "tank"}`;
  if (k === "channel" && breederOf(d) === "hcpb") return "He channel";
  return LABEL[k];
}
const matOf = (k, d = cur?.design) => k === "fw" ? FW_NAME[d.fwMaterial] : (k === "blanket" || k === "channel") ? (breederOf(d) === "hcpb" && k === "channel" ? "helium" : BREEDER_NAME[breederOf(d)]) : MATERIAL[k];
const COLORS = { plasma: [58, 44, 22], fw: [150, 156, 176], vv_in: [104, 110, 122], channel: [79, 179, 217], be: [126, 196, 140], vv_out: [104, 110, 122], blanket: [37, 98, 125], tank: [84, 88, 98], shield: [196, 128, 64], coil: [110, 78, 58], bore: [15, 15, 18] };

// ---- parameters
const P = (group, key, label, unit, step, min, max) => ({ group, key, label, unit, step, min, max });
const TIPS = {
  "g.fw_cm": "Armour tile on the plasma side. Tungsten and beryllium multiply neutrons and raise TBR up to about 1.5 cm, then absorb.",
  fwMaterial: "Armour material facing the plasma.",
  "g.vv_in_cm": "Inconel 718 vessel wall behind the armour. It absorbs neutrons and runs hottest.",
  "g.channel_cm": "FLiBe gap between the two vessel walls that carries the first-wall heat away.",
  "g.be_cm": "Beryllium layer that multiplies neutrons (n,2n) before they reach the tank.",
  "g.vv_out_cm": "Inconel wall between the channel and the FLiBe tank.",
  li6Pct: "Lithium-6 share of the breeder's lithium. Li-6 is the isotope that breeds tritium from slow neutrons.",
  breeder: "What breeds tritium and carries the heat: FLiBe or FLiNaK salt, PbLi or lithium metal, or a helium-cooled ceramic pebble bed with beryllium. Switching loads that concept's usual Li-6 enrichment and coolant settings.",
  surface: "Plasma boundary from Vega's stellarator work (ConStellaration format), scaled to the major radius. Every layer is a uniform offset of the first wall in 3-D.",
  majorRadiusM: "Major radius the plasma shape is scaled to.",
  "l.fw_cm": "Armour tile on the plasma side.",
  "l.vv_in_cm": "Structural wall behind the armour.",
  "l.channel_cm": "Coolant gap between the two vessel walls.",
  "l.be_cm": "Beryllium neutron multiplier.",
  "l.vv_out_cm": "Wall between the channel and the blanket.",
  "l.blanket_cm": "Breeder depth, the same everywhere around the plasma.",
  "l.shield_cm": "Shield between the blanket and the magnets.",
  solubility: "Which measurement of hydrogen solubility in FLiBe sets the tank-wall permeation. The published values differ by about 1000×.",
  "g.blanket_in_cm": "Depth of FLiBe on the inboard side, next to the central magnet column.",
  "g.blanket_out_cm": "Depth of FLiBe on the outboard side.",
  "g.blanket_vert_cm": "Depth of FLiBe above and below the vessel.",
  "g.R0": "Major radius of the machine, measured at the first wall.",
  "g.a": "Minor radius of the first wall.",
  "g.kappa": "Vertical stretch of the vessel cross-section (height / width).",
  fusionPowerMW: "D-T fusion power. 80% of it leaves as 14 MeV neutrons, the blanket's source.",
  "g.shield_in_cm": "TiH₂ shield that protects the inboard legs of the TF magnets.",
  channelVelocity: "Salt speed in the cooling channel. Faster flow cools the inner vessel better.",
  bulkFlow: "Salt flow through the tank. Sets the outlet temperature and how fast tritium is carried out.",
  inletK: "Salt temperature entering the vessel. FLiBe freezes at 732 K.",
  qPlasma: "Heat flux from the plasma onto the first wall (radiation and particles).",
};
const PARAMS = [
  P("Vessel", "g.fw_cm", "First wall", "cm", 0.1, 0, 5),
  P("Vessel", "fwMaterial", "Wall material", "", null),
  P("Vessel", "g.vv_in_cm", "Inner vessel", "cm", 0.1, 0.2, 5),
  P("Vessel", "g.channel_cm", "Cooling channel", "cm", 0.5, 0.5, 10),
  P("Vessel", "g.be_cm", "Be multiplier", "cm", 0.5, 0, 6),
  P("Vessel", "g.vv_out_cm", "Outer vessel", "cm", 0.5, 0.5, 8),
  P("Breeder", "breeder", "Breeder", "", null),
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
  P("Cooling", "channelVelocity", "Channel velocity", "m/s", 0.1, 0.1, 120),
  P("Cooling", "bulkFlow", "Tank flow", "m³/s", 0.05, 0.1, 100),
  P("Cooling", "inletK", "Inlet temperature", "K", 5, 300, 950),
  P("Cooling", "qPlasma", "Plasma heat on wall", "MW/m²", 0.05, 0, 2),
];
// stellarators: a conformal build around a Fourier first wall (surfaces.json)
const SPARAMS = [
  P("Vessel", "l.fw_cm", "First wall", "cm", 0.1, 0, 5),
  P("Vessel", "fwMaterial", "Wall material", "", null),
  P("Vessel", "l.vv_in_cm", "Inner vessel", "cm", 0.1, 0.2, 5),
  P("Vessel", "l.channel_cm", "Cooling channel", "cm", 0.5, 0.5, 10),
  P("Vessel", "l.be_cm", "Be multiplier", "cm", 0.5, 0, 6),
  P("Vessel", "l.vv_out_cm", "Outer vessel", "cm", 0.5, 0.5, 8),
  P("Breeder", "breeder", "Breeder", "", null),
  P("Breeder", "li6Pct", "Li-6 enrichment", "%", null),
  P("Breeder", "l.blanket_cm", "Blanket depth", "cm", 5, 10, 200),
  P("Machine", "surface", "Plasma shape", "", null),
  P("Machine", "majorRadiusM", "Major radius", "m", 0.1, 3, 15),
  P("Machine", "fusionPowerMW", "Fusion power", "MW", 25, 50, 4000),
  P("Machine", "l.shield_cm", "Shield", "cm", 1, 10, 100),
];
const paramsOf = (d) => isStell(d) ? SPARAMS : PARAMS;
const ALLPARAMS = [...PARAMS, ...SPARAMS.filter((p) => !PARAMS.some((q) => q.key === p.key))];
let SURF = {};
const get = (d, key) => key.startsWith("g.") ? (d.geometry[key.slice(2)] ?? ARC_DEFAULTS[key.slice(2)])
  : key.startsWith("l.") ? (d.layers?.[key.slice(2)] ?? CONFORMAL_DEFAULTS[key.slice(2)])
  : key === "breeder" ? breederOf(d) : d[key];
const set = (d, key, v) => { if (key.startsWith("g.")) d.geometry[key.slice(2)] = v; else if (key.startsWith("l.")) (d.layers ??= {})[key.slice(2)] = v; else d[key] = v; };
const fullDesign = (partial) => ({ ...structuredClone(ARC_DESIGN), ...structuredClone(partial ?? {}), geometry: { ...(partial?.geometry ?? {}) }, layers: { ...(partial?.layers ?? {}) } });
const toConf = (d) => { const S = SURF[d.surface]; return { surface: { rc: S.rc, zs: S.zs, nfp: S.nfp }, majorRadiusM: d.majorRadiusM, layers: d.layers ?? {}, breeder: d.breeder, fwMaterial: d.fwMaterial, li6Pct: d.li6Pct, fusionPowerMW: d.fusionPowerMW }; };
/** One design through the engine: the tokamak path or the conformal (3-D) one. */
const evalDesign = (d, tier, N, onUpdate, seed) => isStell(d)
  ? engine.evaluateConformal(toConf(d), N, onUpdate, seed)
  : engine.evaluate(withTier(d, tier), N, onUpdate, seed);

// ---- state
let engine = null, lib = [], saved = [], cur = null, field = "heat", raster = null, runToken = 0;
const results = new Map(); // design json -> {fast, resolved}
// the tank flow per tier: Fast uses potential flow (0.15 s), Resolved the k-epsilon RANS
// (seconds per new tank shape); the results map stays keyed by the design itself
const withTier = (d, tier) => ({ ...d, tankFlow: tier === "resolved" ? "rans" : "potential" });
const keyOf = (d) => JSON.stringify(d, Object.keys(d).sort()) + JSON.stringify(d.geometry, Object.keys(d.geometry).sort()) + JSON.stringify(d.layers ?? {}, Object.keys(d.layers ?? {}).sort());
try { saved = JSON.parse(localStorage.getItem("blanket.designs") ?? "[]"); } catch { saved = []; }
// Saved designs: the team store (./api, Cloudflare KV behind Access) when the server has one,
// this browser's storage otherwise. `team` flips on when GET ./api/store answers.
let team = false;
const persist = (changed, removed) => {
  try { localStorage.setItem("blanket.designs", JSON.stringify(saved)); } catch { /* private mode */ }
  if (!team) return;
  if (changed) fetch(`./api/designs/${encodeURIComponent(changed.id)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(changed) }).catch(() => {});
  if (removed) fetch(`./api/designs/${encodeURIComponent(removed)}`, { method: "DELETE" }).catch(() => {});
};
async function loadTeamStore() {
  try {
    const r = await fetch("./api/store", { cache: "no-store" });
    if (!r.ok) return;
    const j = await r.json();
    team = true;
    saved = (j.designs ?? []).map((d) => ({ ...d, family: "Your designs" })).sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0));
    $("store").textContent = `team store · ${saved.length} saved`;
  } catch { /* no server store: keep this browser's */ }
}

// ---- formatting
const f = (x, n = 2) => Number.isFinite(x) ? x.toFixed(n) : "—";
const sig = (x, n = 3) => !Number.isFinite(x) ? "—" : Math.abs(x) >= 1e6 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(n - 1).replace(/e\+?(-?)(\d+)/, (_, s, e) => `×10${sup(s + e)}`) : Number(x.toPrecision(n)).toLocaleString();
const sup = (t) => [...t].map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(c)] ?? (c === "-" ? "⁻" : c)).join("");
const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const tipA = (t) => (t ? ` data-tip="${esc(t)}"` : "");

// ---- tooltips: any [data-tip] element; hover, keyboard focus, or tap
const gtip = $("gtip");
let tipFor = null;
function showTip(el) {
  tipFor = el; gtip.textContent = el.dataset.tip; gtip.hidden = false;
  const r = el.getBoundingClientRect(), t = gtip.getBoundingClientRect();
  const x = Math.max(8, Math.min(innerWidth - t.width - 8, r.left));
  const y = r.bottom + 6 + t.height < innerHeight ? r.bottom + 6 : Math.max(8, r.top - 6 - t.height);
  gtip.style.left = `${x}px`; gtip.style.top = `${y}px`;
}
const hideTip = () => { tipFor = null; gtip.hidden = true; };
document.addEventListener("pointerover", (e) => { if (e.pointerType === "touch") return; const el = e.target.closest?.("[data-tip]"); if (el) { if (el !== tipFor) showTip(el); } else if (tipFor) hideTip(); });
document.addEventListener("pointerdown", (e) => { const el = e.target.closest?.("[data-tip]"); if (e.pointerType === "touch" && el && el !== tipFor) showTip(el); else if (!el || e.pointerType === "touch") hideTip(); });
document.addEventListener("focusin", (e) => { const el = e.target.closest?.("[data-tip]"); if (el) showTip(el); else hideTip(); });
addEventListener("scroll", hideTip, true);

// ---- view overlays (material key, colour scale, first-wall profile): each collapses to its title
let ovMin = {};
try { ovMin = JSON.parse(localStorage.getItem("blanket.ov") ?? "{}"); } catch { ovMin = {}; }
// on a phone the first-wall chart would cover the map, so it starts folded there
const isMin = (k) => ovMin[k] ?? (k === "inset" && innerWidth < 700);
const ovSync = () => { for (const el of document.querySelectorAll(".ov")) { const mn = isMin(el.dataset.ov); el.classList.toggle("min", mn); const b = el.querySelector(".oc"); if (b) { b.textContent = mn ? "+" : "–"; b.ariaLabel = mn ? "Expand" : "Collapse"; } } };
const ovHead = (title) => `<div class="oh"><span>${title}</span><button type="button" class="oc" aria-label="Collapse">–</button></div>`;
document.querySelector(".view").addEventListener("click", (e) => {
  const h = e.target.closest(".ov .oh"); if (!h) return;
  const k = h.parentElement.dataset.ov; ovMin[k] = !isMin(k);
  try { localStorage.setItem("blanket.ov", JSON.stringify(ovMin)); } catch { /* private mode */ }
  ovSync();
  if (k === "inset" && !isMin(k)) show();
});

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
        <span class="mx">${m ? `<span>TBR <b>${f(m.tbr, 3)}</b></span>${m.thermal ? `<span>VV <b>${f(m.thermal2d?.peaks.inconelInner.K ?? m.thermal.vvMax, 0)}</b> K</span>` : ""}${tfOk(m) ? `<span>TF <b>${f(m.tfLifetimeFpy, 1)}</b> y</span>` : ""}` : ""}</span>`;
      el.onclick = () => select(it);
      L.append(el);
    }
  }
}

// ---- links: #d=<design id>&<param>=<value>... carries the selected design and its edits
// (the hash never reaches the server)
function writeHash() {
  if (!cur) return;
  const base = fullDesign(cur.base), q = new URLSearchParams({ d: cur.id });
  for (const p of paramsOf(cur.design)) { const v = get(cur.design, p.key); if (v !== get(base, p.key)) q.set(p.key, String(v)); }
  history.replaceState(null, "", `#${q}`);
}
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const it = [...saved, ...lib].find((x) => x.id === q.get("d"));
  if (!it) return false;
  select(it);
  let edited = false;
  for (const p of paramsOf(cur.design)) {
    // older links wrote geometry keys without the "g." prefix
    const raw = q.get(p.key) ?? (p.key.startsWith("g.") ? q.get(p.key.slice(2)) : null);
    if (raw === null) continue;
    let v = p.step ? Number(raw) : p.key === "li6Pct" ? Number(raw) : raw;
    if (p.step) { if (!Number.isFinite(v)) continue; v = Math.min(p.max, Math.max(p.min, v)); }
    if (p.key === "li6Pct" && !engine.enrichments(breederOf(cur.design)).includes(v)) continue;
    if (p.key === "breeder" && !engine.breeders().includes(v)) continue;
    if (p.key === "surface" && !(v in SURF)) continue;
    if (p.key === "fwMaterial" && !engine.fwMaterials().includes(v)) continue;
    if (p.key === "solubility" && !(v in FLIBE_SOLUBILITY)) continue;
    set(cur.design, p.key, v); edited = true;
  }
  if (edited) { renderParams(); redraw(); schedule(); writeHash(); }
  return true;
}

function select(it) {
  cur = { id: it.id, name: it.name, base: structuredClone(it.design ?? {}), design: fullDesign(it.design), reference: it.reference ?? null, saved: it.family === "Your designs" || saved.some((s) => s.id === it.id) };
  $("dname").value = cur.name;
  $("ddel").hidden = !cur.saved;
  if (sweep) closeSweep();
  renderParams(); renderList(); redraw(); schedule(); writeHash();
}

// ---- parameters
function renderParams() {
  const box = $("params"); box.innerHTML = "";
  const base = fullDesign(cur.base);
  const PS = paramsOf(cur.design);
  for (const g of [...new Set(PS.map((p) => p.group))]) {
    const grp = document.createElement("div"); grp.className = "pgroup"; grp.innerHTML = `<h2>${g}</h2>`;
    for (const p of PS.filter((q) => q.group === g)) {
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
        for (const e of engine.enrichments(breederOf(cur.design))) input.add(new Option(`${e}%`, String(e), false, e === v));
        input.onchange = () => update(p.key, Number(input.value));
      } else if (p.key === "breeder") {
        input = document.createElement("select");
        for (const b of engine.breeders()) input.add(new Option(BREEDER_NAME[b] ?? b, b, false, b === v));
        input.onchange = () => setBreeder(input.value);
      } else if (p.key === "surface") {
        input = document.createElement("select");
        for (const [k, S] of Object.entries(SURF)) input.add(new Option(S.label, k, false, k === v));
        input.onchange = () => update(p.key, input.value);
      } else {
        input = document.createElement("input"); input.type = "number"; input.step = String(p.step); input.min = String(p.min); input.max = String(p.max); input.value = String(v);
        input.onchange = () => { const x = Number(input.value); if (Number.isFinite(x)) update(p.key, Math.min(p.max, Math.max(p.min, x))); };
      }
      row.innerHTML = `<span${tipA(TIPS[p.key])}>${p.label}${p.unit ? ` <span class="u">${p.unit}</span>` : ""}</span>`;
      row.append(input);
      if (p.step || p.key === "li6Pct" || p.key === "fwMaterial" || p.key === "breeder") {
        const sw = document.createElement("button");
        sw.type = "button"; sw.className = "swb"; sw.innerHTML = SWEEP_ICON;
        sw.ariaLabel = `Sweep ${p.label.toLowerCase()} across its range`;
        sw.dataset.tip = `See how ${p.label.toLowerCase()} changes the results across its range`;
        sw.setAttribute("aria-pressed", String(sweep?.p.key === p.key));
        sw.onclick = (e) => { e.preventDefault(); startSweep(p); renderParams(); };
        row.append(sw);
      } else row.append(document.createElement("span"));
      grp.append(row);
    }
    box.append(grp);
  }
}

/** A breeder brings its usual enrichment and coolant settings (BREEDER_PRESET). */
function setBreeder(b) {
  if (sweep) closeSweep();
  cur.design.breeder = b;
  const pre = BREEDER_PRESET[b] ?? {};
  for (const [k, v] of Object.entries(pre)) if (!isStell(cur.design) || k === "li6Pct") cur.design[k] = v;
  if (!engine.enrichments(b).includes(cur.design.li6Pct)) cur.design.li6Pct = engine.enrichments(b)[0];
  renderParams(); redraw(); schedule(); writeHash();
}

function update(key, v) {
  if (sweep && key !== sweep.p.key) closeSweep();
  set(cur.design, key, v);
  renderParams(); redraw(); schedule(); drawSweep(); writeHash();
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
    const m = await evalDesign(d, tier, N, (mm) => {
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
    // one short line under each number; the detail lives in the label's tooltip
    const kpi = (k, v, c, sub, tip, warn) => `<div class="kpi"><span class="k"${tipA(warn ? `${tip}\n\n⚠ ${warn}` : tip)}>${k}${warn ? '<span class="w">⚠</span>' : ""}</span><span class="v ${c}">${v}</span><span class="s">${sub}</span></div>`;
    if (isStell(cur.design)) { showStell(m, kpi, cls); return showRest(r, m, tier); }
    const t2 = m.thermal2d, P2 = t2?.peaks;
    const flow = m.tritium?.flowModel === "rans" ? "Resolved: recirculating (k-epsilon) tank flow." : "Fast: potential tank flow, no recirculation. Resolved runs the recirculating flow.";
    K.innerHTML = [
      kpi("Tritium breeding ratio", f(m.tbr, m.tbrErr < 0.001 ? 4 : 3), cls(m.tbr, 1.1, 1.0), `± ${f(m.tbrErr, m.tbrErr < 0.001 ? 4 : 3)} · target 1.1`,
        "Tritium atoms bred per fusion neutron. A plant needs at least 1.1 to stay self-sufficient with margin. ± is the Monte Carlo standard error."),
      t2
        ? kpi("Peak inner vessel", `${f(P2.inconelInner.K, 0)} K`, cls(P2.inconelInner.K, 973, 1073, false), "limit ~973 K",
          `Hottest point on the inner Inconel 718 wall, from the 2-D (R, Z) heat-conduction field. Inconel 718 loses strength above about 973 K.\nAt R ${f(P2.inconelInner.R, 2)} m, Z ${f(P2.inconelInner.Z, 2)} m · tungsten face ${f(P2.w.K, 0)} K · 1-D estimate ${f(m.thermal.vvMax, 0)} K.\nOuter vessel ${f(P2.inconelOuter.K, 0)} K, set by an assumed film coefficient to the tank.`,
          t2.warnings?.length ? t2.warnings.join("\n") : null)
        : kpi("Peak inner vessel", `${f(m.thermal.vvMax, 0)} K`, cls(m.thermal.vvMax, 973, 1073, false), "1-D estimate",
          `Hottest point on the inner Inconel 718 wall (1-D estimate). Channel outlet ${f(m.thermal.channelOutlet, 0)} K.`),
      tfOk(m)
        ? kpi("TF magnet life", `${f(m.tfLifetimeFpy, 1)} y`, cls(m.tfLifetimeFpy, 10, 3), Number.isFinite(m.tfRelErr) ? `± ${Math.round(m.tfRelErr * 100)}% · full-power` : "full-power years",
          "Full-power years until the inboard face of the TF magnet reaches a fast-neutron fluence of 3×10¹⁸ n/cm².")
        : kpi("TF magnet life", "—", "", `<button type="button" class="lnk" data-run="resolved">run Resolved</button>`,
          `Full-power years until the inboard TF magnet face reaches 3×10¹⁸ n/cm². Too few of Fast's ${sig(m.histories, 2)} neutrons get through the shield to estimate it; Resolved runs enough.`),
      breederOf(cur.design) !== "flibe"
        ? kpi("Tritium in the salt", "—", "", `${f(m.tritiumMgPerS, 2)} mg/s bred`,
          `Tritium transport in the blanket is modelled for FLiBe only so far. ${BREEDER_NAME[breederOf(cur.design)]} needs its own solubility and extraction model.`)
        : m.tritium
        ? kpi("Tritium in the salt", `${f(m.tritium.inventoryMg, 0)} mg`, "", `${f(m.tritiumMgPerS, 2)} mg/s bred`,
          `Steady-state tritium dissolved in the tank's salt, before extraction.\n${flow}\nTank-wall loss ${permBand(m)} of the bred tritium across the four FLiBe solubility datasets.`,
          m.tritium.warnings?.length ? m.tritium.warnings.join("\n") : null)
        : kpi("Tritium in the salt", `${f(m.inventoryPlugMg, 0)}–${f(m.inventoryMixedMg, 0)} mg`, "", `${f(m.tritiumMgPerS, 2)} mg/s bred`,
          `Tritium held in the salt, between plug flow and a fully mixed tank. ${f(m.residenceS / 60, 1)} min turnover.`),
      m.damage
        ? kpi("Vessel helium limit", `${f(500 / m.damage.vvInPeakHeAppmPerFpy, 1)} y`, cls(500 / m.damage.vvInPeakHeAppmPerFpy, 5, 1.5), "to 500 appm He",
          `Full-power years until the inner vessel's peak reaches 500 appm helium, past which Inconel can no longer be weld-repaired (Sorbom's criterion).\n${f(m.damage.vvInPeakHeAppmPerFpy, 0)} appm He/y · ${f(m.damage.vvInPeakDpaPerFpy, 1)} dpa/y (NRT).`)
        : "",
      kpi("Energy multiplication", f(m.energyMultiplication, 3), "", `${f(m.nuclearMW, 0)} MW heat`,
        "Nuclear heat deposited in the machine per unit of neutron energy. Above 1 because Li-6 and beryllium reactions release energy."),
      kpi("Tank outlet", `${f(m.thermal.bulkOutlet, 0)} K`, cls(m.thermal.bulkOutlet, 1000, 1100, false), `inlet ${f(cur.design.inletK, 0)} K`,
        "Salt temperature leaving the tank, from the heat balance at this flow. FLiBe freezes at 732 K."),
    ].join("");
  }
  showRest(r, m, tier);
}

/** KPIs for a stellarator (conformal 3-D build): what that path computes. */
function showStell(m, kpi, cls) {
  const he = m.damage?.layers?.vv_in?.heAppmPerFpy;
  $("kpis").innerHTML = [
    kpi("Tritium breeding ratio", f(m.tbr, m.tbrErr < 0.001 ? 4 : 3), cls(m.tbr, 1.1, 1.0), `± ${f(m.tbrErr, m.tbrErr < 0.001 ? 4 : 3)} · target 1.1`,
      "Tritium atoms bred per fusion neutron, in the full 3-D stellarator geometry. ± is the Monte Carlo standard error."),
    kpi("Neutron wall load", f(m.wallLoadMWm2, 2), cls(m.wallLoadMWm2, 2, 3.5, false), `MW/m² · ${f(m.wallAreaM2, 0)} m² wall`,
      "Average 14 MeV neutron power per unit first-wall area. The local peak is higher where the plasma comes closest to the wall."),
    tfOk(m)
      ? kpi("Magnet life", `${f(m.tfLifetimeFpy, 1)} y`, cls(m.tfLifetimeFpy, 10, 3), Number.isFinite(m.tfRelErr) ? `± ${Math.round(m.tfRelErr * 100)}% · average face` : "full-power years",
        "Full-power years until the coils' plasma-facing 5 cm reaches 3×10¹⁸ n/cm² fast fluence, averaged all round. Hot spots reach it sooner.")
      : kpi("Magnet life", "—", "", `<button type="button" class="lnk" data-run="resolved">run Resolved</button>`,
        "Too few Fast neutrons get through the shield to estimate it; Resolved runs enough."),
    Number.isFinite(he)
      ? kpi("Vessel helium limit", `${f(500 / he, 1)} y`, cls(500 / he, 5, 1.5), "average, to 500 appm",
        `Full-power years until the inner vessel averages 500 appm helium (Sorbom's re-weldability criterion). Peaks are higher; their factor is not computed for 3-D shapes yet.\n${f(he, 0)} appm He/y average.`)
      : "",
    kpi("Energy multiplication", f(m.energyMultiplication, 3), "", `${f(m.nuclearMW, 0)} MW heat`,
      "Nuclear heat deposited in the machine per unit of neutron energy."),
  ].join("");
}

function showRest(r, m, tier) {
  const K = $("kpis");
  // ladder
  const ref = refFor();
  const L = $("ladder");
  const RT = { fast: "100k neutron histories on your GPU; re-runs on every edit.", resolved: "8M histories and the recirculating tank-flow solve, about 12 s. Click to run.", reference: "Continuous-energy OpenMC, run offline. Library designs only, before edits." };
  const rung = (t, name, desc, res, extra) => `<button type="button" class="rung ${extra ?? ""}" data-t="${t}" aria-current="${tier === t}"${tipA(RT[t])}><span class="tn">${name}</span><span class="ts">${res ?? desc}</span><span class="tr">${t === "reference" ? "OpenMC" : "browser"}</span><span class="bar"><span style="width:${res ? 100 : 0}%"></span></span></button>`;
  L.innerHTML = [
    rung("fast", "Fast", `${sig(FAST, 2)} histories`, r?.fast ? `TBR ${f(r.fast.tbr, 3)} ± ${f(r.fast.tbrErr, 3)} · ${f(r.fast.seconds, 2)} s` : null),
    rung("resolved", "Resolved", `${RESOLVED / 1e6}M histories · run`, r?.resolved ? `TBR ${f(r.resolved.tbr, 4)} ± ${f(r.resolved.tbrErr, 4)} · ${f(r.resolved.seconds, 1)} s` : null),
    rung("reference", "Reference", "library designs only", ref ? `TBR ${f(ref.TBR, 4)} ± ${f(ref.TBR_sd, 4)}` : null, ref ? "" : "na"),
  ].join("");
  L.querySelector('[data-t="resolved"]').onclick = () => run("resolved");
  for (const b of K.querySelectorAll("[data-run]")) b.onclick = () => run(b.dataset.run);
  // heating bars
  const H = $("heat");
  if (m) {
    const rows = ["fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield"].filter((k) => m.heatingMW[k] !== undefined);
    const max = Math.max(...rows.map((k) => m.heatingMW[k]));
    H.innerHTML = rows.map((k) => {
      const v = m.heatingMW[k], rv = ref?.heating_MW?.[k];
      const c = COLORS[k];
      return `<div class="brow"><span>${lbl(k)}</span><span class="track"><span style="width:${(100 * v) / max}%;background:rgb(${c})"></span>${rv !== undefined ? `<em style="left:${Math.min(100, (100 * rv) / max)}%" title="OpenMC ${f(rv, 1)} MW"></em>` : ""}</span><span class="n">${f(v, v < 10 ? 2 : 1)} MW</span></div>`;
    }).join("");
  }
  $("heat-h").dataset.tip = `Nuclear heat deposited in each layer, this run.${ref?.heating_MW ? " White ticks: the OpenMC reference." : ""}`;
  $("notes").innerHTML = `<a href="validation.html">How these numbers are computed and checked →</a>`;
  showSens();
  drawMap(m);
  if (view3d && m && !$("v3").hidden && !isStell(cur.design)) view3d.setLive(m, cur.design);
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

// ---- cross-section
let phi = 0, sliceToken = 0;
async function redraw() {
  const stell = isStell(cur.design);
  $("layers").hidden = stell; $("phictl").hidden = !stell;
  if (stell) {
    const tok = ++sliceToken, d = cur.design;
    const g = await engine.conformalGeometry(toConf(d));
    if (tok !== sliceToken) return;
    raster = { ...conformalSlice(g, phi, 2), stell: g, R0: d.majorRadiusM * 100 };
    $("phiv").textContent = `${Math.round((phi * 180) / Math.PI)}°`;
  } else raster = regionRaster(cur.design, 2);
  const used = new Set(raster.ids);
  $("keys").innerHTML = ovHead("Layers") + `<div class="ob">${REGIONS.map((r, i) => used.has(i) && r !== "bore" ? `<span><i style="background:rgb(${COLORS[r]})"></i>${r === "fw" ? `First wall (${FW_NAME[cur.design.fwMaterial] ?? cur.design.fwMaterial})` : lbl(r)}</span>` : "").join("")}</div>`;
  ovSync();
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
  const rLo = raster.stell ? 0 : Math.max(0, Math.floor((g.coil.rmin - 20) / h));
  if (raster.stell) m = null; // no (R, Z) maps for 3-D builds yet: materials only
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
    let c = id >= 0 && !(raster.stell && id === 10) ? COLORS[REGIONS[id]] : [15, 15, 18];
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
  const inset = $("inset");
  inset.hidden = !(field === "temp" && m?.thermal2d);
  ovSync();
  if (!inset.hidden && !isMin("inset")) {
    const ic = $("insetcv"), iw = ic.clientWidth * dpr, ih = ic.clientHeight * dpr;
    ic.width = iw; ic.height = ih;
    drawStations(ic.getContext("2d"), m.thermal2d, 0, 4 * dpr, iw, ih - 18 * dpr, dpr);
  }
  // axis tick: major radius
  ctx.fillStyle = "#55555e"; ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  const R0 = raster.stell ? raster.R0 : g.p.R0, rr0 = raster.stell ? raster.r0 : 0;
  ctx.fillText(`R = ${(R0 / 100).toFixed(2)} m`, ox + (((R0 - rr0) / h) - rLo) * s - 30 * dpr, oy + nz * s + 18 * dpr);
  const L = $("legend");
  if (dens) {
    L.hidden = false;
    const grad = `linear-gradient(90deg, ${INFERNO.map((c) => `rgb(${c})`).join(",")})`;
    const ttl = { heat: "Heating, W/cm³", trit: "Tritium bred, /s/cm³", conc: "Tritium, mg/m³", temp: "Temperature, K" }[field];
    const [a, b] = view.linear ? [lo, hi] : [10 ** lo, 10 ** hi];
    L.innerHTML = ovHead(ttl) + `<div class="ob"><span><i style="background:${grad}"></i></span><span style="display:flex;justify-content:space-between"><span>${sig(a, 2)}</span><span>${sig(b, 2)}</span></span></div>`;
    ovSync();
  } else L.hidden = true;
}

/** Inset: temperatures along the first wall from the channel inlet (blanket-thermal stations). */
function drawStations(ctx, t, x0, y0, w, h, dpr) {
  const S = t.stations, n = S.t.length;
  if (!n || w < 100 * dpr) return;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const series = [["W face", S.wFaceK, "rgb(250,193,39)"], ["inner Inconel", S.inconelPeakK, "rgb(212,72,66)"], ["coolant", S.bulkK, "rgb(79,179,217)"]];
  let lo = Infinity, hi = -Infinity;
  for (const [, a] of series) for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  lo = Math.floor(lo / 50) * 50; hi = Math.ceil(hi / 50) * 50;
  const X = (k) => x0 + 34 * dpr + (k / (n - 1)) * (w - 40 * dpr), Y = (v) => y0 + 14 * dpr + (1 - (v - lo) / (hi - lo)) * (h - 34 * dpr);
  ctx.save();
  ctx.font = `${10.5 * dpr}px ui-monospace, monospace`; ctx.fillStyle = "#8a8a94";
  for (const v of [lo, (lo + hi) / 2, hi]) { ctx.fillText(`${v}`, x0 + 4 * dpr, Y(v) + 3 * dpr); ctx.strokeStyle = "#24242a"; ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(n - 1), Y(v)); ctx.stroke(); }
  series.forEach(([lbl, a, c], si) => {
    ctx.strokeStyle = c; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    for (let k = 0; k < n; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(a[k]));
    ctx.stroke();
    ctx.fillStyle = c; ctx.fillRect(x0 + 34 * dpr + si * 96 * dpr, y0 + h + 2 * dpr, 8 * dpr, 8 * dpr);
    ctx.fillStyle = "#a3a3ad"; ctx.fillText(lbl, x0 + 46 * dpr + si * 96 * dpr, y0 + h + 10 * dpr);
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
  if (id < 0 || (raster.stell && id === 10)) { tip.hidden = true; return; }
  const r = REGIONS[id], R = (raster.stell ? raster.r0 : 0) + (i + 0.5) * raster.h, Z = raster.z0 + (j + 0.5) * raster.h;
  let val = "";
  if (view.dens && id >= 1 && id <= 6) {
    const G = view.G, gi = Math.floor((R - G.r0) / G.h), gj = Math.floor((Z - G.z0) / G.h);
    const v = view.dens[gj * G.nr + gi];
    if (v > 0) val = ` · <b>${field === "temp" ? f(v, 0) : sig(v, 3)}</b> ${{ heat: "W/cm³", trit: "T/s/cm³", conc: "mg/m³", temp: "K" }[field]}`;
  }
  const mat = matOf(r);
  tip.innerHTML = `<b>${lbl(r)}</b>${mat ? ` (${mat})` : ""} · R ${(R / 100).toFixed(2)} m, Z ${(Z / 100).toFixed(2)} m${val}`;
  tip.hidden = false;
  tip.style.left = `${e.clientX - rect.left + 12}px`; tip.style.top = `${e.clientY - rect.top + 12}px`;
});
$("cv").addEventListener("mouseleave", () => { $("tip").hidden = true; });
// 3-D cutaway (StoneLab apps/blanket-view), loaded on first use and fed every result
let view3d = null, view3dLoading = null;
async function show3d(on) {
  $("v3").hidden = !on;
  $("view").classList.toggle("in3d", on);
  for (const id of ["cv", "keys", "legend", "inset"]) $(id).style.visibility = on ? "hidden" : "";
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

// ---- sweep: one parameter across its range at the Fast tier, the coupled outputs side by side
const SWEEP_OUT = [
  { k: "Tritium breeding ratio", v: (m) => m.tbr, e: (m) => m.tbrErr, n: 3, guide: [1.1, "target 1.1"] },
  { k: "Peak inner vessel, K", v: (m) => m.thermal2d?.peaks.inconelInner.K ?? m.thermal?.vvMax ?? NaN, n: 0, guide: [973, "973 K"] },
  { k: "Tritium in the salt, mg", v: (m) => m.tritium?.inventoryMg ?? NaN, n: 0 },
  { k: "Vessel helium limit, years", v: (m) => (m.damage ? 500 / (Number.isFinite(m.damage.vvInPeakHeAppmPerFpy) ? m.damage.vvInPeakHeAppmPerFpy : m.damage.layers?.vv_in?.heAppmPerFpy) : NaN), n: 2 },
];
const SWEEP_N = 9;
// a small trend chart: "plot the results against this input"
const SWEEP_ICON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 2v11.5H14"/><path d="M4.5 10.5l3-3.5 2.5 2 3.5-4.5"/></svg>`;
let sweep = null, sweepToken = 0;
const snapTo = (x, step) => Number((Math.round(x / step) * step).toFixed(6));

async function startSweep(p) {
  const token = ++sweepToken;
  const now = get(cur.design, p.key);
  // wall material is categorical: one point per material, no line between them
  const cat = p.key === "fwMaterial" || p.key === "breeder";
  const grid = cat ? (p.key === "breeder" ? engine.breeders() : engine.fwMaterials()) : p.key === "li6Pct" ? engine.enrichments(breederOf(cur.design)) : Array.from({ length: SWEEP_N }, (_, i) => snapTo(p.min + ((p.max - p.min) * i) / (SWEEP_N - 1), p.step));
  const xs = cat ? grid : [...new Set([...grid, now])].sort((a, b) => a - b);
  sweep = { p, xs, pos: cat ? xs.map((_, j) => j) : xs, cat, ms: xs.map(() => null), base: structuredClone(cur.design), hover: -1 };
  if (!$("v3").hidden) { show3d(false); $("layers").querySelector('[data-f="3d"]').setAttribute("aria-pressed", "false"); }
  $("sw").hidden = false; $("view").classList.add("insweep");
  drawSweep();
  for (let i = 0; i < xs.length; i++) {
    const d = structuredClone(sweep.base); set(d, p.key, xs[i]);
    if (p.key === "breeder") for (const [k, v] of Object.entries(BREEDER_PRESET[xs[i]] ?? {})) if (!isStell(d) || k === "li6Pct") d[k] = v;
    const k = keyOf(d);
    let m = results.get(k)?.fast;
    if (!m) {
      try { m = await evalDesign(d, "fast", FAST, undefined, 1); } catch (e) { console.warn("sweep", xs[i], e); }
      if (m) results.set(k, { ...(results.get(k) ?? {}), fast: m });
    }
    if (token !== sweepToken) return;
    sweep.ms[i] = m ?? null; drawSweep();
  }
}

function closeSweep() {
  sweepToken++; sweep = null;
  $("sw").hidden = true; $("view").classList.remove("insweep");
}

function drawSweep() {
  if (!sweep) return;
  const { p, xs, ms, pos, cat } = sweep;
  const lab = (x) => (cat ? FW_NAME[x] ?? BREEDER_NAME[x] ?? x : sig(x, 4));
  const done = ms.filter(Boolean).length, now = get(cur.design, p.key);
  const unit = p.unit ? ` ${p.unit}` : "";
  $("sw-head").innerHTML = `<span${tipA(`${xs.length} designs at the Fast tier, ${sig(FAST, 2)} histories each, all with the same random numbers so the differences between points are real, not noise.`)}><b>${p.label}</b>${p.unit ? ` <span class="u">${p.unit}</span>` : ""} · ${xs.length} designs${done < xs.length ? ` · <b>${done}/${xs.length}</b>` : ""}</span>${sweep.hover >= 0 ? `<span class="hxv">${p.label} = ${lab(xs[sweep.hover])}${unit}</span>` : `<span class="u">click a point to apply it</span>`}`;
  const host = $("sw-charts");
  if (host.childElementCount !== SWEEP_OUT.length) host.innerHTML = SWEEP_OUT.map(() => `<div class="swc"><div class="swt"></div><svg></svg></div>`).join("");
  const x0 = cat ? -0.5 : pos[0], x1 = cat ? pos.length - 0.5 : pos[pos.length - 1];
  const nowPos = cat ? xs.indexOf(now) : now;
  SWEEP_OUT.forEach((o, ci) => {
    const cell = host.children[ci], svg = cell.querySelector("svg");
    const W = Math.max(160, cell.clientWidth - 18), H = Math.max(100, cell.clientHeight - cell.querySelector(".swt").offsetHeight - 10);
    const L = 46, R = 12, T = 10, B = 24;
    const pts = xs.map((x, j) => (ms[j] ? { j, x: pos[j], y: o.v(ms[j]), e: o.e?.(ms[j]) ?? 0 } : null)).filter((q) => q && Number.isFinite(q.y));
    const hv = sweep.hover >= 0 && ms[sweep.hover] ? o.v(ms[sweep.hover]) : null;
    cell.querySelector(".swt").innerHTML = `${o.k}${hv !== null && Number.isFinite(hv) ? ` <b>${f(hv, o.n)}</b>` : ""}`;
    if (!pts.length) { svg.innerHTML = ""; svg.setAttribute("viewBox", `0 0 ${W} ${H}`); return; }
    let lo = Math.min(...pts.map((q) => q.y - q.e)), hi = Math.max(...pts.map((q) => q.y + q.e));
    // show the guide line when it is near the data, without letting it flatten the curve
    if (o.guide) { const [g] = o.guide, span = Math.max(hi - lo, Math.abs(hi) * 0.02); if (g > lo - 0.6 * span && g < hi + 0.6 * span) { lo = Math.min(lo, g); hi = Math.max(hi, g); } }
    // at least 5% of the value: Fast-tier noise must not be stretched to full height
    { const c = (hi + lo) / 2, d = Math.max(Math.abs(c) * 0.025, 1e-3); if (hi - lo < 2 * d) { lo = c - d; hi = c + d; } }
    const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;
    const X = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R), Y = (y) => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
    const ticks = (a, b, n) => { const st = 10 ** Math.floor(Math.log10((b - a) / n || 1)); const s = [1, 2, 5, 10].map((k) => k * st).find((k) => (b - a) / k <= n) ?? st * 10; const out = []; for (let v = Math.ceil(a / s) * s; v <= b + 1e-9; v += s) out.push(Number(v.toPrecision(6))); return out; };
    const yt = ticks(lo, hi, 3), xt = cat || p.key === "li6Pct" ? pos : ticks(x0, x1, 4);
    let s = "";
    for (const v of yt) s += `<line class="g" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${L - 6}" y="${Y(v) + 3.5}" text-anchor="end">${sig(v, 3)}</text>`;
    for (const v of xt) s += `<text class="ax" x="${X(v)}" y="${H - 6}" text-anchor="middle">${cat ? FW_NAME[xs[v]] ?? xs[v] : sig(v, 3)}</text>`;
    if (o.guide && o.guide[0] >= lo && o.guide[0] <= hi) s += `<line class="gd" x1="${L}" x2="${W - R}" y1="${Y(o.guide[0])}" y2="${Y(o.guide[0])}"/><text class="gl" x="${W - R}" y="${Y(o.guide[0]) - 4}" text-anchor="end">${o.guide[1]}</text>`;
    if (nowPos >= x0) s += `<line class="cur" x1="${X(nowPos)}" x2="${X(nowPos)}" y1="${T}" y2="${H - B}"/>`;
    if (sweep.hover >= 0) s += `<line class="hx" x1="${X(pos[sweep.hover])}" x2="${X(pos[sweep.hover])}" y1="${T}" y2="${H - B}"/>`;
    if (!cat) s += `<polyline class="ln" points="${pts.map((q) => `${X(q.x)},${Y(q.y)}`).join(" ")}"/>`;
    for (const q of pts) {
      if (q.e > 0 && Y(q.y - q.e) - Y(q.y + q.e) > 3) s += `<line class="eb" x1="${X(q.x)}" x2="${X(q.x)}" y1="${Y(q.y - q.e)}" y2="${Y(q.y + q.e)}"/>`;
      s += `<circle class="pt${xs[q.j] === now ? " on" : ""}${q.j === sweep.hover ? " hv" : ""}" cx="${X(q.x)}" cy="${Y(q.y)}" r="${q.j === sweep.hover ? 5 : 4}"/>`;
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.innerHTML = s;
    svg.dataset.l = L; svg.dataset.r = W - R;
  });
}

// hover: nearest swept value across every chart; click applies it
const sweepIndexAt = (svg, clientX) => {
  const rect = svg.getBoundingClientRect(), L = Number(svg.dataset.l), Rr = Number(svg.dataset.r);
  const t = (clientX - rect.left - L) / (Rr - L), P = sweep.pos, a = sweep.cat ? -0.5 : P[0], b = sweep.cat ? P.length - 0.5 : P[P.length - 1], x = a + t * (b - a);
  let best = -1, bd = Infinity;
  P.forEach((v, j) => { if (sweep.ms[j] && Math.abs(v - x) < bd) { bd = Math.abs(v - x); best = j; } });
  return best;
};
$("sw-charts").addEventListener("pointermove", (e) => {
  const svg = e.target.closest("svg"); if (!sweep || !svg) return;
  const j = sweepIndexAt(svg, e.clientX);
  if (j === sweep.hover) return;
  sweep.hover = j; drawSweep();
});
$("sw-charts").addEventListener("pointerleave", () => { if (!sweep) return; sweep.hover = -1; drawSweep(); });
$("sw-charts").addEventListener("click", (e) => {
  const svg = e.target.closest("svg"); if (!sweep || !svg) return;
  const j = sweepIndexAt(svg, e.clientX);
  if (j >= 0) update(sweep.p.key, sweep.xs[j]);
});
$("sw-close").onclick = () => { closeSweep(); };
new ResizeObserver(() => drawSweep()).observe($("sw"));

let layersMin = false;
try { layersMin = localStorage.getItem("blanket.layersMin") === "1"; } catch { /* private mode */ }
const layersSync = () => { $("layers").classList.toggle("min", layersMin); $("lcol").textContent = layersMin ? "›" : "‹"; $("lcol").setAttribute("aria-expanded", String(!layersMin)); $("lcol").ariaLabel = layersMin ? "Show the view options" : "Collapse the view options"; };
layersSync();
$("layers").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.id === "lcol") { layersMin = !layersMin; try { localStorage.setItem("blanket.layersMin", layersMin ? "1" : "0"); } catch { /* private mode */ } layersSync(); return; }
  if (sweep) closeSweep();
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

// ---- panes: drag the dividers (double-click resets), x a pane closed (the rest take its room),
// reopen it from the header. Same scheme as the newwater simulator; a per-viewer convenience.
(() => {
  const DEF = { designs: 260, metrics: 330, view: 0.6, closed: [] }, NAMES = { designs: "Designs", view: "View", params: "Parameters", metrics: "Metrics" };
  let L = { ...DEF };
  try { L = { ...DEF, ...JSON.parse(localStorage.getItem("blanket.layout") ?? "{}") }; } catch { /* convenience */ }
  const save = () => { try { localStorage.setItem("blanket.layout", JSON.stringify(L)); } catch { /* convenience */ } };
  const pane = (k) => document.querySelector(`.pn[data-pane="${k}"]`), open = (k) => !L.closed.includes(k);
  function apply() {
    for (const k of Object.keys(NAMES)) pane(k).hidden = !open(k);
    const center = open("view") || open("params");
    $("center").hidden = !center;
    $("g1").hidden = !(open("designs") && (center || open("metrics")));
    $("g2").hidden = !(open("metrics") && center);
    $("gh").hidden = !(open("view") && open("params"));
    // the centre takes the slack; with no centre the metrics (else the designs) do
    const grow = center ? "center" : open("metrics") ? "metrics" : "designs";
    for (const k of ["designs", "metrics"]) { const el = pane(k), g = grow === k; el.style.flex = g ? "1 1 0" : "none"; el.style.width = g ? "" : `${L[k]}px`; }
    pane("view").style.flex = open("params") ? `${L.view} 1 0` : "1 1 0";
    pane("params").style.flex = open("view") ? `${1 - L.view} 1 0` : "1 1 0";
    $("reopen").innerHTML = L.closed.map((k) => `<button type="button" data-k="${k}">+ ${NAMES[k]}</button>`).join("");
    save();
  }
  for (const b of document.querySelectorAll(".pn .px")) b.onclick = () => {
    const k = b.closest(".pn").dataset.pane;
    if (Object.keys(NAMES).filter(open).length <= 1) return; // keep one pane
    L.closed = [...L.closed, k]; apply();
  };
  $("reopen").onclick = (e) => { const k = e.target.dataset?.k; if (!k) return; L.closed = L.closed.filter((x) => x !== k); apply(); };
  function drag(g, axis, onMove) {
    g.addEventListener("pointerdown", (e) => {
      e.preventDefault(); g.setPointerCapture(e.pointerId); g.classList.add("drag"); document.body.classList.add("dragging", axis);
      const x0 = e.clientX, y0 = e.clientY, s0 = { ...L };
      const mv = (ev) => { onMove(ev.clientX - x0, ev.clientY - y0, s0); apply(); };
      const up = () => { g.classList.remove("drag"); document.body.classList.remove("dragging", axis); g.removeEventListener("pointermove", mv); g.removeEventListener("pointerup", up); g.removeEventListener("pointercancel", up); };
      g.addEventListener("pointermove", mv); g.addEventListener("pointerup", up); g.addEventListener("pointercancel", up);
    });
    g.addEventListener("dblclick", () => { for (const k of ["designs", "metrics", "view"]) L[k] = DEF[k]; apply(); }); // reset sizes
  }
  const W = () => $("work").clientWidth, clampW = (v) => Math.max(180, Math.min(W() - 260, v));
  drag($("g1"), "v", (dx, dy, s0) => { L.designs = clampW(s0.designs + dx); });
  drag($("g2"), "v", (dx, dy, s0) => { L.metrics = clampW(s0.metrics - dx); });
  drag($("gh"), "h", (dx, dy, s0) => { L.view = Math.max(0.15, Math.min(0.88, s0.view + dy / Math.max(1, $("center").clientHeight - 40))); });
  apply();
})();

// ---- which layer matters: density sensitivities of TBR and the magnet fluence, one GPU run
const SENS_N = 2_000_000;
const sensCache = new Map(); // design key -> Sensitivities
let sensBusy = false;
function showSens() {
  const box = $("sens");
  if (!box || !cur) return;
  const s = sensCache.get(keyOf(cur.design));
  if (!s) {
    box.innerHTML = `<button type="button" class="lnk" id="sensgo">${sensBusy ? "computing…" : `compute (${sig(SENS_N, 2)} neutrons)`}</button>`;
    $("sensgo").onclick = runSens;
    return;
  }
  // +10% density of each layer: change in TBR (absolute) and in magnet fast fluence (%)
  // fluence falls off exponentially with a shield's density, so its +10% is exp(0.1 d ln F) - 1, not linear
  const rows = s.regions.map((r, i) => ({ r, t: 0.1 * s.dTbr[i], q: 100 * (Math.exp(0.1 * s.dlnTf[i]) - 1) })).filter((x) => x.r !== "coil" && (Math.abs(x.t) > 2e-4 || Math.abs(x.q) > 0.2));
  const mt = Math.max(...rows.map((x) => Math.abs(x.t)), 1e-6), mq = Math.max(...rows.map((x) => Math.abs(x.q)), 1e-6);
  const bar = (v, mx, c) => `<span class="track ctr"><span style="${v >= 0 ? "left:50%" : `left:${50 - (50 * Math.abs(v)) / mx}%`};width:${(50 * Math.abs(v)) / mx}%;background:${c}"></span></span>`;
  box.innerHTML = `<div class="sensh"><span></span><span>TBR</span><span>Magnet fluence</span></div>` + rows.map((x) =>
    `<div class="srow"><span>${lbl(x.r)}</span>${bar(x.t, mt, x.t >= 0 ? "var(--good)" : "var(--bad)")}<span class="n">${x.t >= 0 ? "+" : "−"}${f(Math.abs(x.t), 4)}</span>${bar(x.q, mq, x.q <= 0 ? "var(--good)" : "var(--bad)")}<span class="n">${x.q >= 0 ? "+" : "−"}${f(Math.abs(x.q), 1)}%</span></div>`).join("");
}
async function runSens() {
  if (sensBusy) return;
  sensBusy = true; showSens();
  const d = structuredClone(cur.design), k = keyOf(d);
  try { sensCache.set(k, await engine.sensitivities(isStell(d) ? toConf(d) : d, SENS_N, 5)); }
  catch (e) { console.error(e); }
  sensBusy = false; showSens();
}
$("phi").oninput = () => { phi = (Number($("phi").value) * Math.PI) / 180; redraw(); };

// ---- design actions
$("dname").onchange = () => { cur.name = $("dname").value; if (cur.saved) { const s = saved.find((x) => x.id === cur.id); if (s) { s.name = cur.name; persist(s); renderList(); } } };
$("dreset").onclick = () => { if (sweep) closeSweep(); cur.design = fullDesign(cur.base); renderParams(); redraw(); schedule(); writeHash(); };
$("dlink").onclick = async () => {
  writeHash();
  const b = $("dlink");
  try { await navigator.clipboard.writeText(location.href); b.textContent = "Link copied"; } catch { b.textContent = "Copy from the address bar"; }
  setTimeout(() => { b.textContent = "Copy link"; }, 1800);
};
$("dsave").onclick = () => {
  const it = { id: `my-${Date.now()}`, name: `${cur.name}${cur.saved ? "" : " (edited)"}`, family: "Your designs", design: structuredClone(cur.design), reference: null };
  saved.unshift(it); persist(it); select(it);
};
$("ddel").onclick = () => { const id = cur.id; saved = saved.filter((s) => s.id !== id); persist(null, id); select(lib[0]); };

// ---- boot
(async () => {
  try {
    engine = await BlanketEngine.create("./packs");
  } catch (e) {
    document.querySelector(".work").innerHTML = `<div class="err"><b>WebGPU is needed.</b><br>${e.message}<br><br>Chrome or Edge 113+ on a machine with a GPU.</div>`;
    return;
  }
  $("gpu").textContent = "WebGPU";
  await loadTeamStore();
  lib = await (await fetch("./designs.json")).json();
  try { SURF = await (await fetch("./surfaces.json")).json(); } catch { SURF = {}; }
  if (!readHash()) select(lib[0]);
  addEventListener("hashchange", () => readHash());
  // fill in the list's numbers in the background, one design at a time
  for (const it of lib.slice(1)) {
    const d = fullDesign(it.design), k = keyOf(d);
    if (results.get(k)?.fast) continue;
    try { const m = await evalDesign(d, "fast", FAST, undefined, 1); results.set(k, { ...(results.get(k) ?? {}), fast: m }); renderList(); } catch (e) { console.warn(it.name, e); }
  }
})();

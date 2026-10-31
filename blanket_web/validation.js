// Blanket validation page: every number comes from validation.json (build.py), which
// gathers the measured LIBRA data, the published ARC figures, our OpenMC runs and the
// browser checks (StoneLab scripts/validation-collect.ts). Charts are inline SVG.
const $ = (id) => document.getElementById(id);
const V = await (await fetch("./validation.json")).json();
const B = V.browser, BM = V.benchmarks;
const C = { openmc: "#3987e5", measured: "#d95926", browser: "#199e70", paper: "#c98500" };
const f = (x, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : "—");
const pct = (x, n = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(n)}%`;
const NS = "http://www.w3.org/2000/svg";

// ---------- small SVG kit
function svg(w, h) { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); return s; }
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); parent?.append(e); return e; }
function text(parent, x, y, s, cls = "axis", anchor = "start") { const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent); t.textContent = s; return t; }
/** Marker per series: circle (OpenMC), square (measured), diamond (browser), triangle (published). */
function marker(parent, kind, x, y, r = 5, fill) {
  const c = fill ?? C[kind], st = { fill: c, stroke: "#16161a", "stroke-width": 2 };
  if (kind === "openmc") return el("circle", { cx: x, cy: y, r, ...st }, parent);
  if (kind === "measured") return el("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 1.5, ...st }, parent);
  if (kind === "browser") return el("path", { d: `M${x},${y - r - 1}L${x + r + 1},${y}L${x},${y + r + 1}L${x - r - 1},${y}Z`, ...st }, parent);
  return el("path", { d: `M${x},${y - r - 1}L${x + r + 1},${y + r}L${x - r - 1},${y + r}Z`, ...st }, parent);
}
function legendItem(kind, label) {
  const s = svg(14, 14); marker(s, kind, 7, 7, 4.5);
  const span = document.createElement("span"); span.append(s, label); return span;
}
const tip = $("tip");
function hover(node, html) {
  node.style.cursor = "default";
  node.addEventListener("mousemove", (e) => { tip.innerHTML = html; tip.hidden = false; tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 12}px`; });
  node.addEventListener("mouseleave", () => { tip.hidden = true; });
}
function ticks(lo, hi, n = 5) {
  const step = 10 ** Math.floor(Math.log10((hi - lo) / n)); const m = [1, 2, 2.5, 5, 10].find((k) => (hi - lo) / (k * step) <= n) * step;
  const out = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-12; v += m) out.push(+v.toFixed(10)); return out;
}

// ---------- hero
$("chain").innerHTML = [
  ["measured", "Measured", "LIBRA BABY 1L and FLiBe-1L: tritium counted from a litre of salt"],
  ["openmc", "OpenMC (continuous energy)", "reproduces LIBRA's own models; our ARC model of the published design"],
  ["browser", "Browser (GPU multigroup)", "175 groups from OpenMC, 64-bin scattering angles, ~3M histories/s"],
].map(([k, t, s]) => `<div class="step"><b><i style="background:${C[k]}"></i>${t}</b>${s}</div>`).join("");
const packErr = B.packs.map((p) => Math.abs(p.browser[0] / p.openmc - 1));
const LO = V.libra_openmc_full;
const flibe = LO["FLiBe-1L"], baby = LO["BABY-1L"];
const corr = (repo, full) => { const b = B.libra[repo]; return [b.browser[0] * full.ours.mean / b.openmc_simplified[0], full.ours.mean / b.openmc_simplified[0]]; };
const [flibeBrowserFull] = corr("FLiBe-1L-run-1", flibe);
$("tiles").innerHTML = [
  ["Browser vs OpenMC", `≤ ${(Math.max(...packErr) * 100).toFixed(2)}%`, `TBR across ${B.packs.length} ARC cross-section sets`],
  ["FLiBe-1L, measured", `${f(flibe.measured["run-1"].value * 1e3, 2)} · ${f(flibe.measured["run-2"].value * 1e3, 2)}`, `×10⁻³ TBR · browser chain ${f(flibeBrowserFull * 1e3, 2)}`],
  ["Speed", `${f(B.libra["FLiBe-1L-run-1"].seconds, 2)} s`, "20 million histories, one experiment"],
  ["ARC magnet life", `${f(3e18 / mean(B.tf.per_run_fluence_per_fpy), 1)} y`, "published: 9 full-power years"],
].map(([k, v, s]) => `<div class="tile"><span class="k">${k}</span><span class="v">${v}</span><span class="s">${s}</span></div>`).join("");
function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }

// ---------- LIBRA
function measuredOf(full) {
  return Object.entries(full.measured).map(([run, v]) => (typeof v === "number" ? { run, value: v, error: null } : { run, value: v.value, error: v.error }));
}
function libraFig(title, sub, repo, full, live) {
  const b = B.libra[repo], [bf, ratio] = corr(repo, full);
  const rows = [
    ...measuredOf(full).map((m) => ({ k: "measured", label: `measured, ${m.run}`, v: m.value, e: m.error })),
    { k: "openmc", label: "LIBRA's OpenMC model", v: full.libra_modelled, e: null },
    { k: "openmc", label: "our OpenMC, full model", v: full.ours.mean, e: full.ours.std },
    { k: "browser", label: live ? "browser (live) × full/simplified" : "browser × full/simplified", v: live ? live * ratio : bf, e: b.browser[1] * ratio },
  ];
  const W = 460, rowH = 26, top = 8, H = top + rows.length * rowH + 34, L = 170, Rr = 18;
  const vals = rows.flatMap((r) => [r.v - (r.e ?? 0), r.v + (r.e ?? 0)]);
  const lo = Math.min(...vals) * 0.9, hi = Math.max(...vals) * 1.05;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - Rr);
  const s = svg(W, H), g = el("g", { class: "grid" }, s);
  for (const t of ticks(lo * 1e3, hi * 1e3, 5)) { const xx = x(t / 1e3); el("line", { x1: xx, x2: xx, y1: top, y2: H - 30 }, g); text(s, xx, H - 16, String(t), "axis", "middle"); }
  text(s, (L + W) / 2, H - 2, "TBR ×10⁻³", "axis-title", "middle");
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    text(s, L - 10, y + 4, r.label, "axis", "end");
    if (r.e) el("line", { x1: x(r.v - r.e), x2: x(r.v + r.e), y1: y, y2: y, stroke: C[r.k], "stroke-width": 2, "stroke-linecap": "round" }, s);
    const m = marker(s, r.k, x(r.v), y);
    const hit = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, s);
    hover(hit, `<b>${r.label}</b><br>${f(r.v * 1e3, 3)}${r.e ? ` ± ${f(r.e * 1e3, 3)}` : ""} ×10⁻³`);
    void m;
  });
  const d = document.createElement("div"); d.className = "fig";
  d.innerHTML = `<div class="ft">${title}</div><div class="fs">${sub}</div>`;
  d.append(s);
  const lg = document.createElement("div"); lg.className = "legend";
  lg.append(legendItem("measured", "measured"), legendItem("openmc", "OpenMC"), legendItem("browser", "browser"));
  d.append(lg);
  return d;
}
function drawLibra(live = {}) {
  const box = $("libra-figs"); box.innerHTML = "";
  box.append(
    libraFig("FLiBe-1L · FLiBe, natural lithium", "measured whiskers: reported 1σ", "FLiBe-1L-run-1", flibe, live["FLiBe-1L-run-1"]),
    libraFig("BABY 1L · LiF-LiCl surrogate salt", "the repository reports no per-run error", "BABY-1L-run-4", baby, live["BABY-1L-run-4"]),
  );
}
drawLibra();
{
  const r = (label, a, b, n = 3) => `<tr><td>${label}</td><td class="n">${a}</td><td class="n">${b}</td></tr>`;
  const e = (m, s) => `${f(m * 1e3, 3)}${s ? ` <span class="d">± ${f(s * 1e3, 3)}</span>` : ""}`;
  const bb = B.libra["BABY-1L-run-4"], bfl = B.libra["FLiBe-1L-run-1"];
  const [bbF, bbR] = corr("BABY-1L-run-4", baby), [flF, flR] = corr("FLiBe-1L-run-1", flibe);
  const mb = measuredOf(baby), mf = measuredOf(flibe);
  $("libra-table").innerHTML = `<tr><th>TBR ×10⁻³</th><th class="n">BABY 1L</th><th class="n">FLiBe-1L</th></tr>`
    + r("Browser, crucible stack in free space", e(...bb.browser), e(...bfl.browser))
    + r("OpenMC, the same stack (what the browser must match)", e(...bb.openmc_simplified), e(...bfl.openmc_simplified))
    + `<tr class="sub"><td>browser / OpenMC − 1</td><td class="n">${pct(bb.browser[0] / bb.openmc_simplified[0] - 1, 1)}</td><td class="n">${pct(bfl.browser[0] / bfl.openmc_simplified[0] - 1, 1)}</td></tr>`
    + r("OpenMC, LIBRA's full model (vault, generator tube, lead, table)", e(baby.ours.mean, baby.ours.std), e(flibe.ours.mean, flibe.ours.std))
    + `<tr class="sub"><td>full / stack (room ${pct(baby.ours.mean / V.baby_no_vault.no_vault.mean - 1, 1)} of it for BABY)</td><td class="n">× ${f(bbR, 3)}</td><td class="n">× ${f(flR, 3)}</td></tr>`
    + r("LIBRA's published OpenMC", e(baby.libra_modelled), e(flibe.libra_modelled))
    + r("Browser × full / stack", e(bbF), e(flF))
    + r("Measured", mb.map((m) => f(m.value * 1e3, 2)).join(" · "), mf.map((m) => `${f(m.value * 1e3, 2)} <span class="d">± ${f(m.error * 1e3, 2)}</span>`).join(" · "));
}
let engine = null;
$("runlive").onclick = async () => {
  const btn = $("runlive"), out = $("liveout");
  btn.disabled = true;
  try {
    if (!engine) { out.textContent = "starting WebGPU…"; const { BlanketEngine } = await import("./blanket-engine.js"); engine = await BlanketEngine.create("./packs"); }
    const live = {}, parts = [];
    for (const repo of ["FLiBe-1L-run-1", "BABY-1L-run-4"]) {
      const r = await engine.libra(repo, 20_000_000, (m, se, n) => { out.textContent = `${repo}: ${(n / 1e6).toFixed(1)}M histories, TBR ${f(m * 1e3, 3)} ×10⁻³`; });
      live[repo] = r.tbr;
      parts.push(`${repo.replace(/-run-\d/, "")}: ${f(r.tbr * 1e3, 3)} ± ${f(r.se * 1e3, 3)} ×10⁻³ in ${f(r.seconds, 2)} s (OpenMC ${f(r.openmc[0] * 1e3, 3)})`);
    }
    out.innerHTML = parts.join("<br>");
    drawLibra(live);
  } catch (e) { out.textContent = `WebGPU needed: ${e.message}`; }
  btn.disabled = false;
};

// ---------- browser vs OpenMC
const PACK_LABEL = (p) => p.pack === "base" ? "ARC baseline (W wall, 90% Li-6)" : p.pack.replace("fw_material-", "first wall: ").replace("li6_pct-", "Li-6 ").replace(/\.0$/, "") + (p.pack.startsWith("li6") ? "%" : "");
{
  const W = 460, rowH = 24, top = 6, L = 200, H = top + B.packs.length * rowH + 34;
  const ds = B.packs.map((p) => ({ label: PACK_LABEL(p), d: p.browser[0] / p.openmc - 1, e: p.browser[1] / p.openmc }));
  const lim = Math.max(0.01, ...ds.map((x) => Math.abs(x.d) + x.e)) * 1.15;
  const x = (v) => L + ((v + lim) / (2 * lim)) * (W - L - 14);
  const s = svg(W, H), g = el("g", { class: "grid" }, s);
  for (const t of ticks(-lim * 100, lim * 100, 6)) { el("line", { x1: x(t / 100), x2: x(t / 100), y1: top, y2: H - 30 }, g); text(s, x(t / 100), H - 16, `${t}%`, "axis", "middle"); }
  el("line", { x1: x(0), x2: x(0), y1: top, y2: H - 30, stroke: "#8a8a94", "stroke-width": 1 }, s);
  ds.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    text(s, L - 10, y + 4, r.label, "axis", "end");
    el("line", { x1: x(r.d - r.e), x2: x(r.d + r.e), y1: y, y2: y, stroke: C.browser, "stroke-width": 2, "stroke-linecap": "round" }, s);
    marker(s, "browser", x(r.d), y, 4.5);
    hover(el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, s), `<b>${r.label}</b><br>browser ${f(B.packs[i].browser[0], 4)} · OpenMC ${f(B.packs[i].openmc, 4)} · ${pct(r.d)}`);
  });
  text(s, (L + W) / 2, H - 2, "browser TBR / OpenMC TBR − 1", "axis-title", "middle");
  $("parity").append(s);
}
{
  const p = B.packs.find((q) => q.pack === "base");
  const layers = Object.entries(p.heating_MW);
  const NAME = { fw: "first wall", vv_in: "inner vessel", channel: "cooling channel", be: "Be multiplier", vv_out: "outer vessel", blanket: "FLiBe tank", tank: "tank wall", shield: "TiH₂ shield" };
  const W = 460, rowH = 24, top = 6, L = 140, H = top + layers.length * rowH + 34;
  const ds = layers.map(([k, [b, o]]) => ({ k, b, o, d: b / o - 1 }));
  const lim = Math.max(0.02, ...ds.map((x) => Math.abs(x.d))) * 1.2;
  const x = (v) => L + ((v + lim) / (2 * lim)) * (W - L - 14);
  const s = svg(W, H), g = el("g", { class: "grid" }, s);
  for (const t of ticks(-lim * 100, lim * 100, 6)) { el("line", { x1: x(t / 100), x2: x(t / 100), y1: top, y2: H - 30 }, g); text(s, x(t / 100), H - 16, `${t}%`, "axis", "middle"); }
  el("line", { x1: x(0), x2: x(0), y1: top, y2: H - 30, stroke: "#8a8a94", "stroke-width": 1 }, s);
  ds.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    text(s, L - 10, y + 4, NAME[r.k], "axis", "end");
    marker(s, "browser", x(r.d), y, 4.5);
    text(s, W, y + 4, `${f(r.b, r.b < 10 ? 2 : 1)} MW`, "axis", "end");
    hover(el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, s), `<b>${NAME[r.k]}</b><br>browser ${f(r.b, 2)} MW · OpenMC ${f(r.o, 2)} MW · ${pct(r.d, 1)}`);
  });
  text(s, (L + W - 50) / 2, H - 2, "browser / OpenMC − 1", "axis-title", "middle");
  $("heatpar").append(s);
}
$("pack-table").innerHTML = `<tr><th>Cross-section set</th><th class="n">Browser TBR</th><th class="n">OpenMC TBR</th><th class="n">Δ</th><th class="n">Worst layer heating</th></tr>`
  + B.packs.map((p) => {
    const worst = Object.entries(p.heating_MW).filter(([k]) => !["tank", "shield"].includes(k)).map(([k, [b, o]]) => [k, b / o - 1]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
    return `<tr><td>${PACK_LABEL(p)}</td><td class="n">${f(p.browser[0], 4)} <span class="d">± ${f(p.browser[1], 4)}</span></td><td class="n">${f(p.openmc, 4)}</td><td class="n">${pct(p.browser[0] / p.openmc - 1)}</td><td class="n">${worst[0]} ${pct(worst[1], 1)}</td></tr>`;
  }).join("");

// ---------- Fig. 30 small multiples
{
  const P = BM.arc_sorbom2015.fig30_TBR_vs_first_wall_cm, T = P.thickness_cm;
  const MAT = { W: "Tungsten", Be: "Beryllium", CFC: "Carbon fibre", Inconel: "Inconel 718" };
  const all = [];
  for (const m of Object.keys(MAT)) for (let i = 0; i < T.length; i++) { all.push(P[m][i]); const o = V.fig30_openmc[`${m}_${T[i].toFixed(1)}`]; if (o) all.push(o.mean); }
  for (const r of B.first_wall) all.push(r.browser[0]);
  const lo = Math.floor(Math.min(...all) * 50) / 50, hi = Math.ceil(Math.max(...all) * 50) / 50;
  for (const m of Object.keys(MAT)) {
    const W = 260, H = 200, L = 40, Rr = 8, top = 8, bot = 30;
    const x = (v) => L + (v / 2.1) * (W - L - Rr), y = (v) => top + (1 - (v - lo) / (hi - lo)) * (H - top - bot);
    const s = svg(W, H), g = el("g", { class: "grid" }, s);
    for (const t of ticks(lo, hi, 4)) { el("line", { x1: L, x2: W - Rr, y1: y(t), y2: y(t) }, g); text(s, L - 6, y(t) + 4, t.toFixed(2), "axis", "end"); }
    for (const t of [0, 0.5, 1, 1.5, 2]) text(s, x(t), H - 14, String(t), "axis", "middle");
    text(s, (L + W) / 2, H - 1, "first-wall thickness, cm", "axis-title", "middle");
    // published: line + triangles
    el("polyline", { points: T.map((t, i) => `${x(t)},${y(P[m][i])}`).join(" "), fill: "none", stroke: C.paper, "stroke-width": 2, "stroke-linejoin": "round" }, s);
    T.forEach((t, i) => { const n = marker(s, "paper", x(t), y(P[m][i]), 3.5); hover(n, `<b>${MAT[m]}, ${t} cm</b><br>published (MCNP) ${f(P[m][i], 3)}`); });
    T.forEach((t) => { const o = V.fig30_openmc[`${m}_${t.toFixed(1)}`]; if (o) { const n = marker(s, "openmc", x(t), y(o.mean), 3.5); hover(n, `<b>${MAT[m]}, ${t} cm</b><br>our OpenMC ${f(o.mean, 4)} ± ${f(o.std, 4)}`); } });
    B.first_wall.filter((r) => r.material === m).forEach((r) => { const n = marker(s, "browser", x(r.cm), y(r.browser[0]), 3.5); hover(n, `<b>${MAT[m]}, ${r.cm} cm</b><br>browser ${f(r.browser[0], 4)} ± ${f(r.browser[1], 4)}`); });
    const d = document.createElement("div"); d.className = "fig"; d.innerHTML = `<div class="ft">${MAT[m]}</div><div class="fs">TBR</div>`; d.append(s);
    $("fig30").append(d);
  }
  $("fig30-legend").append(legendItem("paper", "published (MCNP, digitised)"), legendItem("openmc", "our OpenMC"), legendItem("browser", "browser"));
  const slope = (m, src) => { const xs = T, ys = src(m); const mx = mean(xs), my = mean(ys); return xs.reduce((a, xx, i) => a + (xx - mx) * (ys[i] - my), 0) / xs.reduce((a, xx) => a + (xx - mx) ** 2, 0); };
  const ours = (m) => T.map((t) => V.fig30_openmc[`${m}_${t.toFixed(1)}`].mean), pap = (m) => P[m];
  $("fig30-note").innerHTML = `Absorbing walls match the paper to within 0.01 at every thickness (TBR per cm, ours / published: Inconel ${f(slope("Inconel", ours), 3)} / ${f(slope("Inconel", pap), 3)}, carbon ${f(slope("CFC", ours), 3)} / ${f(slope("CFC", pap), 3)}). <b>Neutron-multiplying walls gain about twice the paper's TBR per cm</b> (tungsten ${f(slope("W", ours), 3)} / ${f(slope("W", pap), 3)}, beryllium ${f(slope("Be", ours), 3)} / ${f(slope("Be", pap), 3)}). The candidates are the 8 cm tungsten divertor and the support columns, both absent from our model, and the newer nuclear data. This is open. The browser follows our OpenMC throughout; its tungsten points beyond 1.4 cm use cross sections weighted on the 1 cm wall's spectrum.`;
}

// ---------- magnet life
{
  const yrs = B.tf.per_run_fluence_per_fpy.map((v) => 3e18 / v);
  const mY = 3e18 / mean(B.tf.per_run_fluence_per_fpy);
  const W = 460, H = 150, L = 30, Rr = 14, yb = 70;
  const lo = 6, hi = 12, x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - Rr);
  const s = svg(W, H), g = el("g", { class: "grid" }, s);
  for (const t of ticks(lo, hi, 6)) { el("line", { x1: x(t), x2: x(t), y1: 18, y2: H - 34 }, g); text(s, x(t), H - 20, String(t), "axis", "middle"); }
  text(s, (L + W) / 2, H - 4, "full-power years to 3×10¹⁸ n/cm² (E > 0.1 MeV)", "axis-title", "middle");
  el("line", { x1: x(9), x2: x(9), y1: 14, y2: H - 34, stroke: C.paper, "stroke-width": 2, "stroke-dasharray": "4 3" }, s);
  marker(s, "paper", x(9), 14, 4); text(s, x(9) + 8, 18, "published 9 y", "axis");
  yrs.forEach((v, i) => { const n = marker(s, "browser", x(v), yb + ((i % 3) - 1) * 9, 4); hover(n, `<b>run ${i + 1}</b><br>${f(v, 2)} y from 8M histories`); });
  el("line", { x1: x(mY), x2: x(mY), y1: 44, y2: H - 34, stroke: C.browser, "stroke-width": 2 }, s);
  text(s, x(mY) - 6, 40, `mean ${f(mY, 1)} y`, "axis", "end");
  $("tf").append(s);
  const tu = B.tf.track_unsplit, ts = B.tf.track_split;
  const p = document.createElement("p"); p.style.fontSize = "12.5px";
  p.innerHTML = `A deep-penetration tally: geometric splitting inboard of the tank (2× per 10 cm). Checked against a run with no splitting: ${(tu[0] * 1e5).toFixed(2)} ± ${(tu[1] * 1e5).toFixed(2)} vs ${(ts[0] * 1e5).toFixed(2)} ± ${(ts[1] * 1e5).toFixed(2)} ×10⁻⁵ (200M vs 40M histories). Our coil and shield are the paper's radial build, not its full geometry.`;
  $("tf").append(p);
}

// ---------- temperatures and tritium
{
  const S = BM.arc_sorbom2015.thermal, P = B.thermal_on_paper_inputs, O = B.arc.thermal;
  $("thermal-table").innerHTML = `<tr><th></th><th class="n">Published</th><th class="n">Our model, paper's heating</th><th class="n">Our model, our heating</th></tr>`
    + `<tr><td>Channel outlet</td><td class="n">${S.channel_outlet_K} K</td><td class="n">${f(P.channelOutlet, 0)} K</td><td class="n">${f(O.channelOutlet, 0)} K</td></tr>`
    + `<tr><td>Peak inner vessel (Inconel)</td><td class="n">${S.vv_max_inconel_K} K</td><td class="n">${f(P.vvMax, 0)} K</td><td class="n">${f(O.vvMax, 0)} K</td></tr>`
    + `<tr><td>Tank outlet</td><td class="n">${S.bulk_outlet_K} K</td><td class="n">${f(P.bulkOutlet, 0)} K</td><td class="n">${f(O.bulkOutlet, 0)} K</td></tr>`
    + `<tr class="sub"><td colspan="4">Our heating sends the 1 cm tungsten tile's own ${f(B.arc.heating_MW.fw, 0)} MW through the inner vessel (heat flux ${f(O.wallHeatFlux / 1e6, 2)} vs ${f(P.wallHeatFlux / 1e6, 2)} MW/m²). The paper's heating table appears to leave the tile out: its inner-vessel 26 MW matches our ${f(B.arc.heating_MW.vv_in, 0)} MW without it.</td></tr>`;
  const F = BM.arc_festim_dark2026, A = B.arc;
  $("trit-table").innerHTML = `<tr><th></th><th class="n">FESTIM ARC (OpenMC → OpenFOAM → FESTIM)</th><th class="n">This simulator</th></tr>`
    + `<tr><td>Tritium bred (= outlet flux at steady state)</td><td class="n">1.0 mg/s</td><td class="n">${f(A.tritiumMgPerS, 2)} mg/s</td></tr>`
    + `<tr><td>Salt flow through the tank</td><td class="n">≈ 1.1 m³/s <span class="d">(1.0 mg/s at 0.90 mg/m³)</span></td><td class="n">1.08 m³/s <span class="d">(Sorbom: 18 × 0.3 m² × 0.2 m/s)</span></td></tr>`
    + `<tr><td>Tritium inventory in the salt</td><td class="n">${F.tritium_inventory_mg} mg</td><td class="n">${f(A.inventory[0], 0)}–${f(A.inventory[1], 0)} mg <span class="d">(plug flow to well mixed)</span></td></tr>`
    + `<tr><td>Time scale</td><td class="n">≈ ${F.time_to_steady_min} min to steady state</td><td class="n">${f(A.residenceS / 60, 1)} min turnover <span class="d">(${f(A.flibeM3, 0)} m³ of salt)</span></td></tr>`
    + `<tr class="sub"><td colspan="3">Both assume perfect extraction at the inlet. FESTIM resolves turbulent mixing in a 5° sector; ours is a residence-time balance, so it brackets the inventory rather than predicting it.</td></tr>`;
}

// ---------- open items
$("gaplist").innerHTML = [
  "<b>Browser vs OpenMC on the LIBRA stacks: +1.5% in both experiments.</b> Likely the 0.05 cm raster at the 0.3 cm Inconel walls, or air averaged as one region.",
  "<b>Neutron-multiplying first walls</b> gain about twice the published TBR per cm (above). The divertor and support columns are not modelled.",
  "<b>BABY run 3 and FLiBe-1L run 2</b> differ from the other runs well beyond their errors. That is LIBRA's open question, not ours; every run is shown.",
  "<b>FLiBe-1L is repository data</b> without a paper yet.",
  "<b>Design combinations</b> that mix a non-tungsten wall with a non-90% enrichment use two cross-section sets that have not been checked against OpenMC together. Variants reuse the baseline's photon-heating factors.",
  "<b>Tritium transport</b> is a residence-time balance: no spatial transport and no permeation through the vessel yet.",
  "<b>Temperatures</b> come from a 1-D slab model of the vessel channel, not a field.",
].map((s) => `<li>${s}</li>`).join("");

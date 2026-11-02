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
const BREEDER_LABEL = { flinak: "FLiNaK", pbli: "PbLi", li: "lithium", hcpb: "He-cooled pebble bed" };
const PACK_LABEL = (p) => p.pack.startsWith("breeder-") ? `breeder: ${BREEDER_LABEL[p.pack.slice(8)] ?? p.pack.slice(8)}` : p.pack === "base" ? "ARC baseline (W wall, 90% Li-6)" : p.pack.replace("fw_material-", "first wall: ").replace("li6_pct-", "Li-6 ").replace(/\.0$/, "") + (p.pack.startsWith("li6") ? "%" : "");
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
  }).join("")
  + (V.combos ?? []).map((c) => `<tr><td>combined: ${c.fw} wall + ${c.li6}% Li-6 <span class="d">(two sets)</span></td><td class="n">${f(c.browser[0], 4)} <span class="d">± ${f(c.browser[1], 4)}</span></td><td class="n">${f(c.openmc[0], 4)} <span class="d">± ${f(c.openmc[1], 4)}</span></td><td class="n">${pct(c.rel)}</td><td class="n">—</td></tr>`).join("");

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
  const S = BM.arc_sorbom2015.thermal, P = B.thermal_on_paper_inputs, O = B.arc.thermal, T2 = B.arc.thermal2d;
  $("thermal-table").innerHTML = `<tr><th></th><th class="n">Published</th><th class="n">1-D, paper's heating</th><th class="n">1-D, our heating</th><th class="n">2-D field, our heating</th></tr>`
    + `<tr><td>Channel outlet</td><td class="n">${S.channel_outlet_K} K</td><td class="n">${f(P.channelOutlet, 0)} K</td><td class="n">${f(O.channelOutlet, 0)} K</td><td class="n">${f(T2.coolant.outletK, 0)} K</td></tr>`
    + `<tr><td>Peak inner vessel (Inconel)</td><td class="n">${S.vv_max_inconel_K} K</td><td class="n">${f(P.vvMax, 0)} K</td><td class="n">${f(O.vvMax, 0)} K</td><td class="n">${f(T2.peaks.inconelInner.K, 0)} K</td></tr>`
    + `<tr><td>Tungsten face</td><td class="n">—</td><td class="n">—</td><td class="n">${f(O.fwSurface, 0)} K</td><td class="n">${f(T2.peaks.w.K, 0)} K</td></tr>`
    + `<tr><td>Tank outlet</td><td class="n">${S.bulk_outlet_K} K</td><td class="n">${f(P.bulkOutlet, 0)} K</td><td class="n">${f(O.bulkOutlet, 0)} K</td><td class="n">—</td></tr>`
    + `<tr class="sub"><td colspan="5">The 1-D model is Sorbom's eqs. 30–33 and reproduces his numbers on his inputs. Our heating sends the 1 cm tungsten tile's own ${f(B.arc.heating_MW.fw, 0)} MW through the inner vessel (heat flux ${f(O.wallHeatFlux / 1e6, 2)} vs ${f(P.wallHeatFlux / 1e6, 2)} MW/m²). The paper's heating table appears to leave the tile out: its inner-vessel 26 MW matches our ${f(B.arc.heating_MW.vv_in, 0)} MW without it. The 2-D field conducts through tile, walls and multiplier with the channel's film coefficient around the full poloidal loop, driven by the heating map's shape. Its peak sits at the outboard just below the midplane (R ${f(T2.peaks.inconelInner.R, 2)} m, Z ${f(T2.peaks.inconelInner.Z, 2)} m), where the map's heating is ${f(T2.mapPeaking?.smoothed ?? NaN, 2)}× the wall average. Energy closes to ${Math.abs(T2.energy.relResidual).toExponential(0)}; ${f(T2.ms, 0)} ms.${T2.warnings.length ? ` Flag: ${T2.warnings.join("; ")}.` : ""}</td></tr>`;
  const F = BM.arc_festim_dark2026, A = B.arc, TR = B.tritium;
  const pb = Object.values(TR.permeationBySolubility);
  $("trit-table").innerHTML = `<tr><th></th><th class="n">FESTIM ARC (OpenMC → OpenFOAM → FESTIM)</th><th class="n">This simulator</th></tr>`
    + `<tr><td>Tritium bred (= outlet flux at steady state)</td><td class="n">1.0 mg/s</td><td class="n">${f(A.tritiumMgPerS, 2)} mg/s</td></tr>`
    + `<tr><td>Salt flow through the tank</td><td class="n">≈ 1.1 m³/s <span class="d">(1.0 mg/s at 0.90 mg/m³)</span></td><td class="n">1.08 m³/s <span class="d">(Sorbom: 18 × 0.3 m² × 0.2 m/s)</span></td></tr>`
    + `<tr><td>Outlet-averaged concentration</td><td class="n">0.90 mg/m³</td><td class="n">${f(TR.outletMgPerM3, 2)} mg/m³</td></tr>`
    + `<tr><td>Tritium inventory in the salt</td><td class="n">${F.tritium_inventory_mg} mg</td><td class="n">${f(TR.inventoryMg, 0)} mg <span class="d">(${Object.entries(TR.grid).map(([k, v]) => `${k} ${f(v, 0)}`).join(" · ")})</span></td></tr>`
    + `<tr><td>Permeation through the tank wall</td><td class="n">not modelled</td><td class="n">${(Math.min(...pb) * 100).toPrecision(2)}–${(Math.max(...pb) * 100).toPrecision(2)}% of bred</td></tr>`
    + `<tr><td>Solve</td><td class="n">HPC, 5° sector, k-ω SST</td><td class="n">axisymmetric k-ε RANS, ${TR.layout} layout · Resolved tier: a few seconds for a new tank shape, cached after</td></tr>`
    + `<tr class="sub"><td colspan="3">Both put clean salt at the inlet (perfect extraction) and run the cooling channel in series with the tank: salt leaves the vessel channel as a jet at the vessel's lower inboard corner and the outlet is at the top inboard (FESTIM's Fig. 2b; Sorbom 5.4.2: the channels "exhaust to the bulk blanket"). The jet angle and outlet slot are our reading of their figure; our tank is a box with no divertor legs. Grid convergence is weak: ±20% at the page's 4 cm grid. Closure knobs: ${Object.entries(TR.turbulence).map(([k, v]) => `${k} ${f(v, 0)}`).join(", ")} mg. Layout matters more than the closure: ${Object.entries(TR.layoutSensitivity ?? {}).map(([k, v]) => `${k} ${f(v, 0)}`).join(", ")} mg. Permeation spans the FLiBe hydrogen-solubility data: ${Object.entries(TR.permeationBySolubility).map(([k, v]) => `${k.replace(/(\d{4})/, " $1")} ${(v * 100).toPrecision(2)}%`).join(", ")} (Inconel: Kishimoto 1985). The Fast tier uses the potential flow with the channel's tritium into the tank.</td></tr>`;
  // where the gap to FESTIM came from: one modelling change at a time
  if (TR.decomposition) {
    const grids = Object.keys(TR.decomposition), names = Object.keys(TR.decomposition[grids[0]]);
    $("trit-table").innerHTML += `<tr><th>Inventory, one change at a time</th>${grids.map((g) => `<th class="n">${g} grid</th>`).join("")}</tr>`
      + names.map((n) => `<tr${/default/.test(n) ? ' style="font-weight:600"' : ""}><td>${n}</td>${grids.map((g) => `<td class="n">${f(TR.decomposition[g][n].inventoryMg, 0)} mg</td>`).join("")}</tr>`).join("")
      + `<tr class="sub"><td colspan="3">FESTIM: ${F.tritium_inventory_mg} mg. Recirculation alone adds about 50 mg; sending the channel's tritium (${f(TR.channelMgPerS / TR.bredMgPerS * 100, 0)}% of production) into the tank instead of straight to the outlet adds about 80 mg. The original model did both the other way and read 133 mg.</td></tr>`;
  }
}

// ---------- vessel damage
{
  const D = B.damage, L = D.browser.layers, O = D.openmc, Pb = D.published;
  const NAME = { vv_in: "Inner vessel (1 cm Inconel 718)", vv_out: "Outer vessel (3 cm)", tank: "Tank wall" };
  const cell = (b, o, n) => `${f(b, n)} <span class="d">(${pct(b / o - 1, 1)})</span>`;
  $("damage-table").innerHTML = `<tr><th></th><th class="n">Browser</th><th class="n">OpenMC</th><th class="n">Sorbom 2015</th><th class="n">Kuang 2018</th></tr>`
    + ["vv_in", "vv_out", "tank"].map((r) => {
      const so = Pb.sorbom2015[r], ku = Pb.kuang2018[r];
      return `<tr><td>${NAME[r]}, dpa (NRT)</td><td class="n">${cell(L[r].dpaPerFpy, O[r].dpa_per_FPY, r === "tank" ? 3 : 1)}</td><td class="n">${f(O[r].dpa_per_FPY, r === "tank" ? 3 : 1)}</td><td class="n">${so ? so[0] : "—"}</td><td class="n">${ku ? ku[0] : "—"}</td></tr>`
        + `<tr><td>${NAME[r]}, He appm</td><td class="n">${cell(L[r].heAppmPerFpy, O[r].He_appm_per_FPY, r === "tank" ? 2 : 0)}</td><td class="n">${f(O[r].He_appm_per_FPY, r === "tank" ? 2 : 0)}</td><td class="n">${so ? so[1] : "—"}</td><td class="n">${ku ? ku[1] : "—"}</td></tr>`;
    }).join("")
    + `<tr><td>Inner vessel peak (outboard midplane)</td><td class="n">${f(D.browser.vvInPeakDpaPerFpy, 1)} dpa · ${f(D.browser.vvInPeakHeAppmPerFpy, 0)} appm</td><td class="n">× ${f(D.peakOverAvg.dpa, 3)} · × ${f(D.peakOverAvg.he, 3)}</td><td class="n">—</td><td class="n">—</td></tr>`
    + `<tr class="sub"><td colspan="5">Responses are OpenMC's ENDF/B-VIII damage energy (MT 444, NRT with E<sub>d</sub> 40 eV for Inconel) and He production (MT 207) per group, folded with the browser's flux. He agrees with both papers within about 15%. <b>Our dpa is 2–3× below theirs</b>; neither states its dpa method or displacement energy, so this is open. He is what binds: at the peak, Sorbom's 500 appm criterion arrives in ${f(500 / D.browser.vvInPeakHeAppmPerFpy, 1)} full-power years. NRT, not arc-dpa; no Ni-59 two-step He or burn-up.</td></tr>`;
}

// ---------- open items
// ---- generic-tool checks
{
  const X = V.extended ?? {}, rows = [];
  const row = (what, check, ours, ref, d) => rows.push(`<tr><td>${what}</td><td>${check}</td><td class="n">${ours}</td><td class="n">${ref}</td><td class="n">${d}</td></tr>`);
  const BN = { flinak: "FLiNaK", pbli: "PbLi", li: "lithium", hcpb: "He-cooled pebble bed" };
  for (const b of X.breeders ?? []) {
    const br = X.breeders_browser?.[b.breeder];
    row(`${BN[b.breeder] ?? b.breeder} breeder, ARC geometry`, "TBR vs OpenMC continuous energy", br ? `${f(br[0], 4)} <span class="d">± ${f(br[1], 4)}</span>` : "—", f(b.openmc_ce, 4), br ? pct(br[0] / b.openmc_ce - 1) : "—");
  }
  const C = X.conformal;
  if (C) {
    const g = C.grids[0], hv = Object.entries(g.heating_MW).filter(([k]) => k !== "coil" && k !== "tank" && k !== "shield").map(([, [a, b]]) => Math.abs(a / b - 1));
    row("3-D conformal path (axisymmetric case)", "TBR vs OpenMC on the same offset geometry", `${f(g.browser[0], 4)} <span class="d">± ${f(g.browser[1], 4)}</span>`, f(C.openmc_ce.TBR, 4), pct(g.rel));
    row("", "heating, first wall to blanket (worst layer)", "", "", `≤ ${(Math.max(...hv) * 100).toFixed(1)}%`);
  }
  const P = X.ports;
  if (P) {
    row(`${P.ports.n} equatorial ports, ${P.ports.width_cm / 100} × ${P.ports.height_cm / 100} m`, "TBR vs OpenMC CSG ports", `${f(P.browser.with[0], 4)}`, `${f(P.openmc.with[0], 4)} <span class="d">± ${f(P.openmc.with[1], 4)}</span>`, pct(P.browser.with[0] / P.openmc.with[0] - 1));
    row("", "TBR lost to the ports", f(P.browser.with[0] - P.browser.without[0], 3), f(P.openmc.with[0] - P.openmc.without[0], 3), "");
  }
  const S = X.sens;
  if (S) {
    const big = S.rows.filter((r) => Math.abs(r.dTBR_FD) > 0.005), w = Math.max(...big.map((r) => Math.abs(r.dTBR_LR - r.dTBR_FD)));
    row("Layer sensitivities (one run)", `dTBR/d(density) vs finite differences, ${S.histories / 1e6}M histories`, `${big.length} layers`, "", `worst |Δ| ${f(w, 4)}`);
    const sh = S.rows.find((r) => r.region === "blanket");
    if (sh) row("", "magnet fluence vs blanket density, d ln F", f(sh.dlnTF_LR, 2), f(sh.dlnTF_FD, 2), "");
  }
  const M = X.mhd;
  row("Activation, decay heat, contact dose", "OpenMC depletion (full ENDF/B-VIII.0 chain), 1 s – 100 y after 1 FPY", "", "", "≤ 1.7%, first wall – shield");
  const U = X.uq;
  if (U?.all) {
    const a = U.all, part = ["ni58", "f19", "be9", "w"].filter((k) => U[k]?.resolved).map((k) => `${{ f19: "F-19", ni58: "Ni-58", be9: "Be-9", w: "W" }[k]} ${U[k].tbr[0].toFixed(2)}%`).join(", ");
    row("Nuclear-data uncertainty on TBR", `ENDF/B-VIII.0 covariances, ${a.n} correlated samples of 9 nuclides in OpenMC`, `± ${a.tbr[0].toFixed(2)}% <span class="d">(± ${(a.tbr[0] / 100 * a.nominal).toFixed(3)})</span>`, "", `68%: ${a.tbr[1].toFixed(2)}–${a.tbr[2].toFixed(2)}%`);
    row("", `largest terms: ${part}; heating ± ${a.heat.toFixed(2)}%`, "", "", "");
  }
  if (M) row("Liquid-metal duct flow in a field", "pressure gradient vs Shercliff / Hunt series, Ha 10–10⁵", "", "", "≤ 0.2%");
  $("generic-table").innerHTML = `<tr><th>What</th><th>Checked against</th><th class="n">Ours</th><th class="n">Reference</th><th class="n">Δ</th></tr>` + rows.join("");
}

$("gaplist").innerHTML = [
  "<b>Nuclear-data uncertainty is a lower bound.</b> Sampled: Li-6, Li-7, Be-9, F-19, Ni-58 and W; the other Inconel nuclides, the shield and the coil were not, nor any angular or secondary-energy distributions (where older studies found the Be (n,2n) term dominant). Per-nuclide values agree with Aimetta et al. (PHYSOR 2022) within about 20%.",
  "<b>Other breeders</b> run in ARC's geometry with coolant properties from recalled correlations; tritium transport and wall permeation are FLiBe-only, and the pebble bed has no purge-gas model. The MHD pressure drop treats the vessel channel as a square duct (no ribs, bends, inserts or manifolds).",
  "<b>Stellarators</b> use the ARC cross sections (collapsed in a tokamak spectrum) and a uniform conformal build; no temperature, tritium or peak-damage factors yet, and the magnet fluence is the coil face's average, not its hot spot. The 3-D geometry path itself matches OpenMC on the axisymmetric case.",
  "<b>Browser vs OpenMC on the LIBRA stacks: +0.5% in both experiments</b> (1.7σ). An earlier +1.5% was the OpenMC reference's own noise: both experiments shared a seed and nearly the same geometry, so their errors were correlated; 20 million reference histories brought it to +0.5%. Halving the raster pitch to 0.025 cm changed nothing.",
  "<b>Neutron-multiplying first walls</b> gain about twice the published TBR per cm (above). The divertor and support columns are not modelled.",
  "<b>BABY run 3 and FLiBe-1L run 2</b> differ from the other runs well beyond their errors. That is LIBRA's open question, not ours; every run is shown.",
  "<b>FLiBe-1L is repository data</b> without a paper yet.",
  "<b>Design combinations</b> mixing a non-tungsten wall with a non-90% enrichment are assembled from two cross-section sets; four pairings were checked against OpenMC (all within 0.4%), not every one.",
  "<b>Tank flow</b>: k-ε RANS in the Resolved tier, potential flow in Fast. Grid convergence is weak (±20% at 4 cm), the jet and outlet placement is read off FESTIM's figure, and the tank flow rate is an open choice: our thermal tier runs the vessel channel at 1.67 m³/s next to a 1.08 m³/s bulk flow, while FESTIM sends 1.1 m³/s through both in series. If the tank sees the sum, the inventory falls roughly in proportion. Permeation covers the tank wall only (not the inner vessel into the plasma chamber), and assumes the Inconel controls it, not the salt film.",
  "<b>Vessel dpa</b> is 2–3× below the ARC papers' while He agrees; their dpa method is unstated.",
  "<b>FLiBe hydrogen solubility</b> varies by ~3 decades between published measurements. The simulator lets you pick the source and shows the band.",
  "<b>Vessel temperatures</b> are a 2-D conduction field with a 1-D channel; the bulk tank's temperature field is not solved (its outlet is an energy balance). The outer vessel's peak is set by an assumed 500 W/m²K film coefficient to the bulk tank, an order-of-magnitude estimate, so it is not a margin result; the inner vessel's peak barely depends on it. The heating map's poloidal smoothing (0–40 cm) moves the peak by about ±6 K.",
  "<b>Tritium release dynamics are not validated.</b> LIBRA's counted release over time follows from its fitted surface-transfer coefficients, and the FLiBe runs release in two stages that one coefficient cannot describe, so we compare totals (TBR) only.",
].map((s) => `<li>${s}</li>`).join("");

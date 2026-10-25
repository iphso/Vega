// Watercap film physics: everything the film shows is precomputed here once, from a scene file
// (Vega/viewer/public/data/watercap/scenes/<id>.json), into tables the layers play back as pure
// functions of time. No DOM, no three.js: runs in node for checks (node cinema/watercap_physics.js).
//
//   fields  bilinear samplers of the solved r–z fields (u axial, v radial, T, S) plus humidity w(T, S)
//   mist    per-radius rise tables z(τ) integrated through the solved axial velocity
//   drops   spray drops integrated from the nozzle with Schiller–Naumann drag in the solved air field,
//           gravity, turbulent dispersion (Ornstein–Uhlenbeck) and drop heating toward the local air
//           temperature; each ends in the sump, on the wall (then drains as film) or past the outlet

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// stateless hash in [0, 1): particle i, channel k
export function h01(i, k = 0) {
  let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(k + 0x27d4eb2f, 0xc2b2ae35);
  x ^= x >>> 15; x = Math.imul(x, 0x2c1b3c6d); x ^= x >>> 12; x = Math.imul(x, 0x297a2d39); x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}

const psat = T => 610.94 * Math.exp(17.625 * T / (T + 243.04));            // Pa (Magnus)
export const humidity = (T, S) => { const pv = Math.min(S, 1.02) * psat(T); return 1e3 * 0.622 * pv / (101325 - pv); }; // g/kg

function locate(arr, x) {
  const n = arr.length;
  if (x <= arr[0]) return [0, 0];
  if (x >= arr[n - 1]) return [n - 2, 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m; else hi = m; }
  return [lo, (x - arr[lo]) / (arr[lo + 1] - arr[lo])];
}

export function makeFields(sc) {
  const F = sc.fields, rc = F.r, zc = F.z;
  const w = F.T.map((row, i) => row.map((T, j) => humidity(T, F.S[i][j])));
  const grids = { u: F.u, v: F.v, T: F.T, S: F.S, w };
  const R = sc.geom.R;
  function sample(name, r, z) {
    const G = grids[name];
    const [i, fr] = locate(rc, Math.min(Math.abs(r), R)), [j, fz] = locate(zc, z);
    const a = G[i][j] * (1 - fz) + G[i][j + 1] * fz, b = G[i + 1][j] * (1 - fz) + G[i + 1][j + 1] * fz;
    return a * (1 - fr) + b * fr;
  }
  // cross-section (area-weighted) average at height z
  function avg(name, z) {
    let s = 0, a = 0;
    for (let i = 0; i < rc.length; i++) {
      const r0 = i ? (rc[i - 1] + rc[i]) / 2 : 0, r1 = i < rc.length - 1 ? (rc[i] + rc[i + 1]) / 2 : R, da = r1 * r1 - r0 * r0;
      s += sample(name, rc[i], z) * da; a += da;
    }
    return s / a;
  }
  const zTop = zc[zc.length - 1];
  return { sample, avg, rc, zc, zTop, wIn: avg("w", 0.02), wOut: avg("w", zTop), TIn: avg("T", 0.02), TOut: avg("T", zTop) };
}

export function geometry(sc) {
  const g = sc.geom, L = g.Ls + g.Lc;
  const discs = [];
  for (let k = 0; k < g.n_discs; k++) discs.push(g.Ls + g.spacing / 2 + k * g.spacing);
  const nz = g.nozzles[0] || { z: g.Ls, r: 0, cone_deg: 30 };
  return { R: g.R, Ls: g.Ls, Lc: g.Lc, L, rd: g.disc_r, s: g.spacing, t: g.disc_t, discs, nozzleZ: nz.z, nozzleR: nz.r, cone: nz.cone_deg };
}

/* ---------------- mist: rise tables per radial bin ---------------- */
export function mistTables(G, Fd, { bins = 18, dtau = 0.01, z0 = -0.18, zEnd = null } = {}) {
  const top = zEnd ?? G.L + 0.7, out = [];
  for (let b = 0; b < bins; b++) {
    const r = G.R * 0.965 * Math.sqrt((b + 0.5) / bins);
    const zs = [z0];
    let z = z0;
    while (z < top && zs.length < 4000) {
      // the solved axial air speed; inlet / outlet extrapolated; a floor keeps recirculation cells moving
      const u = Fd.sample("u", r, Math.min(Math.max(z, 0), Fd.zTop));
      z += Math.max(u, 0.45) * dtau; zs.push(z);
    }
    out.push({ r, z: Float32Array.from(zs), tauEnd: (zs.length - 1) * dtau, dtau });
  }
  return out;
}
export function zAt(tab, tau) {
  const x = tau / tab.dtau, i = Math.min(tab.z.length - 2, Math.max(0, Math.floor(x))), f = Math.min(1, Math.max(0, x - i));
  return tab.z[i] + (tab.z[i + 1] - tab.z[i]) * f;
}
export function tauAt(tab, z) {
  const a = tab.z; let lo = 0, hi = a.length - 1;
  if (z <= a[0]) return 0; if (z >= a[hi]) return tab.tauEnd;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= z) lo = m; else hi = m; }
  return (lo + (z - a[lo]) / (a[lo + 1] - a[lo])) * tab.dtau;
}

/* ---------------- spray drops ---------------- */
// fate: 0 sump, 1 wall (then film), 2 escaped past the outlet, 3 still suspended at tMax
export function sprayDrops(G, Fd, sc, { n = 640, seed = 67, dts = 0.02, tMax = 7, sigmaTurb = 0.1, TL = 0.05 } = {}) {
  const rnd = mulberry32(seed), gauss = () => { let u = 0, v = 0; while (u === 0) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const d0 = sc.d || {}, dMean = (d0.spray_d_um || 300) * 1.08e-6, v0 = d0.spray_v0 || 2.4, Tl0 = d0.spray_T_C ?? 10;
  const half = (G.cone || 30) / 2 * Math.PI / 180;            // cone_deg is the full cone angle
  const rhoL = 998, rhoA = 1.15, mu = 1.8e-5, kA = 0.026, cL = 4186;
  const drops = [];
  for (let i = 0; i < n; i++) {
    // Rosin–Rammler sizes, q = 2.4, clipped to what the nozzle data supports
    const d = Math.min(900e-6, Math.max(45e-6, dMean * Math.pow(-Math.log(1 - 0.999 * rnd()), 1 / 2.4)));
    const th = rnd() * 2 * Math.PI, a = half * Math.sqrt(rnd());
    let x = G.nozzleR, y = G.nozzleZ - 0.012, z = 0;
    let vx = v0 * Math.sin(a) * Math.cos(th), vz = v0 * Math.sin(a) * Math.sin(th), vy = -v0 * Math.cos(a);
    let ux = 0, uy = 0, uz = 0, T = Tl0, t = 0, fate = 3, hit = null;
    const tauP = rhoL * d * d / (18 * mu), dt = Math.min(0.002, tauP / 4);
    const samples = [x, y, z, T];
    let nextS = dts, entrainedAt = -1;
    while (t < tMax) {
      const r = Math.hypot(x, z), zc = Math.min(Math.max(y, 0), Fd.zTop);
      const ua = Fd.sample("u", r, zc), va = Fd.sample("v", r, zc), Ta = Fd.sample("T", r, zc);
      const cx = r > 1e-6 ? x / r : 0, cz = r > 1e-6 ? z / r : 0;
      // turbulent velocity seen by the drop
      const k = Math.exp(-dt / TL), s = sigmaTurb * Math.min(1, (G.R - r) / 0.03) * Math.sqrt(1 - k * k);   // damped at the wall
      ux = ux * k + s * gauss(); uy = uy * k + s * gauss(); uz = uz * k + s * gauss();
      const Ux = va * cx + ux, Uy = ua + uy, Uz = va * cz + uz;
      const rx = Ux - vx, ry = Uy - vy, rz = Uz - vz, Re = rhoA * Math.hypot(rx, ry, rz) * d / mu;
      const f = (1 + 0.15 * Math.pow(Re, 0.687)) / tauP;
      vx += rx * f * dt; vy += (ry * f - 9.81) * dt; vz += rz * f * dt;
      x += vx * dt; y += vy * dt; z += vz * dt; t += dt;
      // heating: convection (Ranz–Marshall) with a latent-heat boost while vapour condenses on the cold drop
      const Nu = 2 + 0.6 * Math.sqrt(Re) * 0.89, tauH = rhoL * cL * d * d / (6 * kA * Nu) / 3.2;
      T += (Ta - T) * (1 - Math.exp(-dt / tauH));
      if (entrainedAt < 0 && y > G.Ls + 0.04 && vy > 0) entrainedAt = t;
      if (t >= nextS) { samples.push(x, y, z, T); nextS += dts; }
      const rr = Math.hypot(x, z);
      if (rr >= G.R * 0.985) { const sc2 = G.R * 0.985 / rr; x *= sc2; z *= sc2; fate = 1; hit = { y, th: Math.atan2(z, x), T }; break; }
      if (y <= 0) { y = 0; fate = 0; break; }
      if (y >= G.L + 0.35) { fate = 2; break; }
    }
    samples.push(x, y, z, T);
    const ns = samples.length / 4, dur = (ns - 1) * dts;
    drops.push({ d, p: Float32Array.from(samples), ns, dts, dur: Math.max(dur, 1e-3), fate, hit, entrainedAt });
  }
  return drops;
}

// drop i at age τ (sim s), path phase only: [x, y, z, T]
export function dropAt(D, tau, out) {
  const x = Math.min(D.ns - 1.0001, Math.max(0, tau / D.dts)), i = Math.floor(x), f = x - i, p = D.p, a = 4 * i, b = a + 4;
  out[0] = p[a] + (p[b] - p[a]) * f; out[1] = p[a + 1] + (p[b + 1] - p[a + 1]) * f;
  out[2] = p[a + 2] + (p[b + 2] - p[a + 2]) * f; out[3] = p[a + 3] + (p[b + 3] - p[a + 3]) * f;
  return out;
}

// the wall film's flux profile: deposited drop volume draining down, normalised to its maximum
export function filmProfile(G, drops, nb = 96) {
  const prof = new Float32Array(nb);
  for (const D of drops) if (D.fate === 1 && D.hit.y < G.Ls + 0.02) {
    const k = Math.min(nb - 1, Math.max(0, Math.floor(D.hit.y / G.Ls * nb)));
    for (let j = 0; j <= k; j++) prof[j] += D.d ** 3;
  }
  const m = Math.max(...prof, 1e-30);
  for (let j = 0; j < nb; j++) prof[j] /= m;
  return prof;
}

// node check: fates and timings for the hero scene
if (typeof process !== "undefined" && process.argv?.[1]?.endsWith("watercap_physics.js")) {
  const fs = await import("node:fs");
  const sc = JSON.parse(fs.readFileSync(new URL("./watercap_scene.json", import.meta.url)));
  const G = geometry(sc), Fd = makeFields(sc);
  console.log("geom", G, "w in/out", Fd.wIn.toFixed(1), Fd.wOut.toFixed(1), "T in/out", Fd.TIn.toFixed(1), Fd.TOut.toFixed(1));
  const M = mistTables(G, Fd);
  console.log("mist rise s:", M.map(m => m.tauEnd.toFixed(2)).join(" "));
  const t0 = Date.now(), D = sprayDrops(G, Fd, sc);
  const c = [0, 0, 0, 0]; D.forEach(d => c[d.fate]++);
  console.log("drops", D.length, "ms", Date.now() - t0, "sump/wall/escaped/suspended", c.join("/"),
    "entrained", D.filter(d => d.entrainedAt >= 0).length,
    "median dur", D.map(d => d.dur).sort((a, b) => a - b)[D.length >> 1].toFixed(2),
    "wall z", D.filter(d => d.fate === 1).map(d => d.hit.y.toFixed(2)).slice(0, 20).join(","));
  const Tend = D.filter(d => d.fate === 0).map(d => d.p[d.p.length - 1]);
  console.log("sump drop T mean", (Tend.reduce((a, b) => a + b, 0) / Tend.length).toFixed(1));
}

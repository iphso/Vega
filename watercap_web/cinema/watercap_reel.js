// The watercap reel: the marketing cut of watercap_script.js. Same world, shots and camera; the
// callouts say what the simulation includes (colour = stage) instead of how each piece validates.
// The status film stays the validation page's; this one is for showing the product.
import * as THREE from "three";
import base from "./watercap_script.js";
import { softPoints, trailSegments, smooth, clamp, hash01 as h01 } from "./cinema.js";

const STAGE = { gas: "#5b9dff", spray: "#3ccfb4", mist: "#9fd8ff", corona: "#ffb547", energy: "#e8e8ec" };

// per base callout id: [stage, label, detail]
const COPY = {
  gasflow: ["gas", "k-ω SST turbulence", "The turbulent updraft is resolved from wall to axis and cross-checked against OpenFOAM."],
  dp:      ["gas", "Pressure drop", "Spray, wall and disc drag are all counted, so every design pays for the airflow it costs."],
  mist:    ["mist", "Fog transport & growth", "Every droplet size class is tracked as it grows, evaporates and settles."],
  nozzle:  ["spray", "Real nozzles", "Drop size, pressure and Cd come from measured atomizer data."],
  traj:    ["spray", "Lagrangian drops", "Thousands of drops, each with full inertia, drag and turbulent dispersion."],
  drops:   ["spray", "Drop heat & mass transfer", "Vapour condenses on every drop, with conduction inside it."],
  film:    ["spray", "Falling wall film", "Deposited spray and condensate drain to the sump, with film resistance."],
  splash:  ["mist", "Phoretic deposition", "Fog drifts to the cold wall by thermophoresis and Stefan flow."],
  flood:   ["spray", "Flooding & entrainment", "Drops the updraft carries away are followed into the corona stage."],
  onset:   ["corona", "Corona onset", "An ionization-integral onset is computed at every electrode tip."],
  vi:      ["corona", "Ion drift & space charge", "A self-consistent ion field and current, with mobility tuned for humid air."],
  charge:  ["corona", "Droplet charging", "Fog droplets charge by field and diffusion charging, then drift to the wall."],
  deposit: ["corona", "Wall capture", "Charged fog lands on the grounded wall and drains away."],
  quench:  ["corona", "Mist quenching", "Dense fog chokes the corona, and the solver finds the self-consistent state."],
  coag:    ["spray", "Spray scavenging", "Falling drops sweep fog out of the air, and the captured fog follows its drop."],
  energy:  ["energy", "Full energy accounting", "Corona, pump and recooling power are reported per litre recovered."],
};

/* ---- the 3D corona tier, in the tube: during "In 3D" the disc stack becomes the tabbed-ring stack in place.
   Plates, ion plumes and air come from the simulator's own LES tables (device3d_reel.json, baked by StoneLab
   apps/device3d/scripts/reel-asset.ts); one plate per disc, the pitch cell scaled to the tube's radius (xy) and
   disc spacing (z) */
const NA = 1400, AS = 64, TS3 = 0.025;   // air at 1/40 speed so the swirl around each tab reads
let D3 = null, GW = null;
const dec = (b, T) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };

function bake3d(a, G) {
  const S = G.R / (a.R * 1e-3) * 1e-3, Sz = G.s / (a.pitch * 1e-3) * 1e-3;   // world m per solver mm (radial, axial)
  const at = (x, y, zRel) => new THREE.Vector3(x * S, G.discs[0] + zRel * Sz, -y * S);    // zRel: mm above plate 0's mid-plane
  const P = dec(a.mesh.pos, Int16Array), N = dec(a.mesh.nrm, Int16Array), I = dec(a.mesh.idx, Uint32Array);
  const p = new Float32Array(P.length), n = new Float32Array(N.length);
  for (let i = 0; i < P.length; i += 3) {
    p[i] = P[i] / a.mesh.scale; p[i + 1] = P[i + 2] / a.mesh.scale; p[i + 2] = -P[i + 1] / a.mesh.scale;   // mm, y up
    n[i] = N[i] / 32767; n[i + 1] = N[i + 2] / 32767; n[i + 2] = -N[i + 1] / 32767;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(p, 3)); geo.setAttribute("normal", new THREE.BufferAttribute(n, 3)); geo.setIndex(new THREE.BufferAttribute(I, 1));
  const ions = Array.from(dec(a.ions.xyzn, Int16Array), v => v / a.ions.scale);
  // air tracers: integrate the LES mean flow (periodic over the pitch) once, over the whole stack; the film replays the paths
  const F = a.flow, U = dec(F.uvw, Int16Array), nd = G.discs.length, dtS = 0.0012;
  const vel = (x, y, z, o) => {
    const fx = (x - F.origin[0]) / F.h - 0.5, fy = (y - F.origin[1]) / F.h - 0.5, fz = ((z % a.pitch) + a.pitch) % a.pitch / F.hz - 0.5;
    const i0 = clamp(Math.floor(fx), 0, F.mx - 2), j0 = clamp(Math.floor(fy), 0, F.my - 2), k0 = Math.floor(fz);
    const tx = clamp(fx - i0, 0, 1), ty = clamp(fy - j0, 0, 1), tz = fz - k0;
    o[0] = o[1] = o[2] = 0;
    for (let c = 0; c < 8; c++) {
      const i = i0 + (c & 1), j = j0 + ((c >> 1) & 1), k = (((k0 + ((c >> 2) & 1)) % F.mz) + F.mz) % F.mz;
      const w = (c & 1 ? tx : 1 - tx) * ((c >> 1) & 1 ? ty : 1 - ty) * ((c >> 2) & 1 ? tz : 1 - tz), q = 3 * (i + F.mx * (j + F.my * k));
      o[0] += w * U[q]; o[1] += w * U[q + 1]; o[2] += w * U[q + 2];
    }
  };
  const path = new Float32Array(NA * AS * 3), v = [0, 0, 0];
  for (let i = 0; i < NA; i++) {
    const r = a.R * 0.96 * Math.sqrt(h01(i, 71)), th = 6.2832 * h01(i, 72);
    let x = r * Math.cos(th), y = r * Math.sin(th), z = a.plateZ + (h01(i, 73) * nd - 0.7) * a.pitch;   // solver mm
    for (let k = 0; k < AS; k++) {
      const o = 3 * (i * AS + k); path[o] = x; path[o + 1] = y; path[o + 2] = z;
      for (let s = 0; s < 4; s++) {
        vel(x, y, z, v); x += v[0] * dtS / 4; y += v[1] * dtS / 4; z += v[2] * dtS / 4;
        const rr = Math.hypot(x, y); if (rr > a.R * 0.985) { x *= a.R * 0.985 / rr; y *= a.R * 0.985 / rr; }
      }
    }
  }
  return { a, G, S, Sz, at, geo, ions, path, dtS, top: G.discs[nd - 1] + 0.5 * G.s, bot: G.discs[0] - 0.5 * G.s };
}

const device3d = {
  init(ctx) {
    const { a, G, S, at } = D3, g = this.g = new THREE.Group(); ctx.scene.add(g);
    // ion plumes, one copy per plate; brightness by log density (amber -> white hot)
    const n = D3.ions.length / 4, nd = G.discs.length; let hi = -1e9; for (let i = 3; i < D3.ions.length; i += 4) hi = Math.max(hi, D3.ions[i]);
    this.ions = softPoints(n * nd, { depthFade: [0.08, 0.25] }); this.ions.renderOrder = 6; g.add(this.ions);
    const A = this.ions.geometry.attributes;
    for (let k = 0; k < nd; k++) for (let i = 0; i < n; i++) {
      const q = 4 * i, j = k * n + i, u = clamp((D3.ions[q + 3] - (hi - 3)) / 3, 0, 1),
        jx = (h01(j, 81) - 0.5) * 0.98, jy = (h01(j, 82) - 0.5) * 0.98, jz = (h01(j, 83) - 0.5) * 0.98,   // break up the 0.98 mm cell lattice
        p = at(D3.ions[q] + jx, D3.ions[q + 1] + jy, D3.ions[q + 2] - a.plateZ + jz + k * a.pitch);
      A.position.setXYZ(j, p.x, p.y, p.z); A.color.setXYZ(j, 1.0 + 0.9 * u * u, 0.42 + 0.7 * u * u, 0.12 + 0.5 * u * u * u);
      A.size.setX(j, 1.1 * S); A.alpha.setX(j, 0.012 + 0.14 * u * u);
    }
    this.ions.commit(n * nd);
    // tab tips: the glow every plume starts from
    this.tips = softPoints(a.apices.length * nd, { depthFade: [0.08, 0.25] }); this.tips.renderOrder = 7; g.add(this.tips);
    const T = this.tips.geometry.attributes;
    for (let k = 0; k < nd; k++) a.apices.forEach((ap, i) => { const p = at(ap[0], ap[1], ap[2] - 0.25 + k * a.pitch), j = k * a.apices.length + i;
      T.position.setXYZ(j, p.x, p.y, p.z); T.color.setXYZ(j, 2.6, 1.7, 0.9); T.size.setX(j, 5 * S); });
    this.air = trailSegments(NA * 5); this.air.renderOrder = 6; g.add(this.air);
  },
  update(scene, t, dt, p) {
    const w = p.w, vis = w > 0.003; this.g.visible = vis; if (!vis) return;
    // far away the plumes of 13 plates pile up additively into a blown-out block: dim them with distance
    const c = p.ctx.camera.position, dCam = Math.hypot(c.x, c.y - 0.5 * (D3.top + D3.bot), c.z), far = clamp(1.1 / dCam, 0.12, 1);
    const glow = p.get("glow", 1) * w * far;
    this.ions.material.uniforms.uOpacity.value = glow;
    const T = this.tips.geometry.attributes;
    for (let j = 0; j < T.alpha.count; j++) T.alpha.setX(j, glow * (0.55 + 0.25 * Math.sin(t * 19 + j * 2.3) * Math.sin(t * 7.1 + j)));
    this.tips.commit();
    // air: each tracer replays its baked path on its own phase; fades in and out over its life
    this.air.visible = p.get("air3", 0) > 0.003; if (!this.air.visible) return;
    const B = this.air.geometry.attributes, life = (AS - 6) * D3.dtS, air = p.get("air3", 0) * w, a = D3.a; let s = 0;
    for (let i = 0; i < NA; i++) {
      const u = (t * TS3 / life + h01(i, 74)) % 1, f = u * (AS - 6) + 5, k0 = Math.floor(f), fr = f - k0;
      const al = air * 0.9 * Math.sin(Math.PI * u);
      const pt = k => { const o = 3 * (i * AS + k), o2 = o + 3; return D3.at(D3.path[o] + fr * (D3.path[o2] - D3.path[o]), D3.path[o + 1] + fr * (D3.path[o2 + 1] - D3.path[o + 1]), D3.path[o + 2] + fr * (D3.path[o2 + 2] - D3.path[o + 2]) - a.plateZ); };
      let prev = pt(k0);
      if (prev.y < D3.bot || prev.y > D3.top) continue;
      for (let k = 1; k <= 5; k++) {
        const q = pt(k0 - k);
        B.position.setXYZ(2 * s, prev.x, prev.y, prev.z); B.position.setXYZ(2 * s + 1, q.x, q.y, q.z);
        B.color.setXYZ(2 * s, 0.7, 0.9, 1.3); B.color.setXYZ(2 * s + 1, 0.7, 0.9, 1.3);
        B.alpha.setX(2 * s, al * (1 - (k - 1) / 5)); B.alpha.setX(2 * s + 1, al * (1 - k / 5));
        prev = q; s++;
      }
    }
    this.air.commit(s);
  },
};

// the base hardware with the tabbed-ring stack in place of the discs (one plate per disc position)
const hardware = {
  init(ctx) {
    base.layers.hardware.init.call(this, ctx);
    this.g.children.find(c => c.isInstancedMesh).visible = false;
    const steel = new THREE.MeshStandardMaterial({ color: 0x737a84, metalness: 0.55, roughness: 0.6, envMapIntensity: 0.4, side: THREE.DoubleSide });
    for (const y of D3.G.discs) {
      const m = new THREE.Mesh(D3.geo, steel);
      m.scale.setScalar(D3.S); m.position.set(0, y - 0.25 * D3.S, 0); this.g.add(m);
    }
  },
  update(scene, t, dt, p) { base.layers.hardware.update.call(this, scene, t, dt, p); },
};

const cap = (shot, kicker, title, line, inT) => ({ shot, kicker, title, line, status: null, in: inT ?? 0.4 });

function captions(ctx) {
  return [
    cap("inlet", "Inlet", "Warm, misty exhaust rises", "Saturated exhaust enters at the bottom, carrying fog and vapour."),
    cap("nozzle", "Spray stage", "A cold spray meets it", "Chilled water is sprayed down through the rising air."),
    cap("heat", "Spray stage", "Vapour condenses on every drop", "Drops run from cold blue to warm as they pull water out of the air."),
    cap("film", "Spray stage", "The wall drains it away", "Spray and condensate that reach the wall fall to the sump as a film."),
    cap("entrain", "Spray stage", "Nothing gets lost", "Fine drops the updraft lifts are followed into the next stage."),
    cap("corona", "Corona stage", "Ions stream from every tip", "High voltage on the central rod fills the gap with ions."),
    cap("charge", "Corona stage", "Fog charges and flies to the wall", "Droplets too fine for any spray get charged and driven out of the air."),
    cap("quench", "Corona stage", "Coupled, end to end", "The fog and the field shape each other, and the solver resolves both together."),
    cap("in3d", "Corona stage · 3D", "Every tip, in 3D", "Ions, ionic wind and the air around any electrode shape, solved on the GPU.", 1.0),
    cap("outlet", "Outlet", "Drier air out, water back", "Recovered water, power and pressure drop for every design."),
    cap("overview", "StoneLab", "One simulation, every stage", "Turbulence, spray, film, fog, corona and energy, solved together.", 1.2),
  ];
}

function callouts(ctx) {
  const all = base.callouts(ctx);
  // overview: the base film pins coagulation at the very top; scavenging belongs at the spray stage's top,
  // where the base pins fog transport -- swap the two anchors
  const ovc = id => all.find(c => c.style === "compact" && c.phys === id);
  const a = ovc("coag"), b = ovc("mist");
  if (a && b) [a.at, b.at] = [b.at, a.at];
  const rTip = D3.a.rOut * D3.S;
  for (const c of all) if ((c.phys === "onset" || c.phys === "quench") && typeof c.at === "function") {
    const f = c.at;
    c.at = (t, cx) => { const v = f(t, cx); if (!v) return v; const r = Math.hypot(v.x, v.z); if (r > 1e-6 && r < rTip) v.multiplyScalar(rTip / r); return v; };
  }
  return all.flatMap(c => {
    const k = COPY[c.phys];
    if (!k) return [];
    const [stage, label, detail] = k;
    let det = detail;
    if (c.phys === "drops" && typeof c.detail === "function") {
      const f = c.detail;   // keep the live "this drop: 300 µm, now 14.2 °C." readout
      det = (t, cx) => { const m = String(f(t, cx)).match(/^This drop: .*? °C\./); return m ? `${m[0]} ${detail}` : detail; };
    }
    return [{ ...c, label, detail: c.style === "compact" ? undefined : det, status: "inc", color: STAGE[stage] }];
  });
}

function shots(ctx) {
  const Ls = GW.Ls, BASE3 = { stage: 1, shell: 1, hardware: 1, mist: 0.25, plane: 0.25, spray: 1, film: 1 };
  const in3d = { id: "in3d", chapter: "In 3D", duration: 11,
    layers: { ...BASE3, device3d: 1, corona: 0 },
    params: { glow: 1, air3: [[0, 0], [1.5, 1], [10, 1], [11, 0]], cut: 0.8, legend: 0, shiftX: 0.0 },
    camera: [{ at: 2.5, target: [0, Ls + 0.42, 0], az: 236, el: 30, dist: 1.05, fov: 30 },
             { at: 7, target: [0, Ls + 0.5, 0], az: 242, el: 20, dist: 0.8, fov: 30 },
             { at: 11, target: [0, Ls + 0.55, 0], az: 246, el: 14, dist: 0.9, fov: 30 }] };
  const end = { id: "end", chapter: "The simulator", duration: 8,
    layers: { stage: 1, shell: 1, hardware: 1, mist: 0.6, plane: 0.3, spray: 0.8, film: 1, corona: 1, outlet: 0.6 },
    params: { legend: 0, hud: [[0, 1], [1.5, 0]], shiftX: [[0, 0.0], [2.5, 0.22]] },
    camera: [{ at: 8, target: [0, 1.6, 0], az: 330, el: 9, dist: 9.0, fov: 28, ease: true }],
    titleCard: { kicker: "StoneLab simulator", title: "Every design, any fidelity", sub: "Fast · Resolved · 3D quick · 3D full",
      meta: "Energy per litre · water captured · pressure drop · arc margin", in: 0.8, out: 7.6 } };
  const out = [];
  const swap = s => s.layers && "corona" in s.layers ? { ...s, layers: { ...s.layers, device3d: s.layers.corona, corona: 0 } } : s;
  for (const s0 of base.shots(ctx)) {
    let s = swap(s0);
    if (s.id === "open") s = { ...s, chapter: "The tube",
      titleCard: { ...s.titleCard, kicker: "StoneLab", title: "Water capture, simulated",
        sub: "Warm, humid exhaust goes in at the bottom. A cold spray and then a corona pull the water out of it, and drier air leaves the top.",
        meta: "Spray · film · fog · corona · energy, all in one coupled model" } };
    if (s.id === "inlet") s = { ...s, chapter: "Inlet" };
    if (s.id === "overview") s = { ...s, chapter: "What's inside", params: { ...s.params, legend: 0 } };
    if (s.id === "outlet") out.push(in3d);
    out.push(s);
  }
  out.push(swap(end));
  return out;
}

const cornerIn3d = t => { const c = D3.ctx; return c.at && t >= c.at("in3d", 0) && t < c.at("outlet", 0); };

export default {
  ...base,
  title: "Water Capture Reel",
  fileName: "watercap-reel",
  status: { inc: { color: STAGE.spray, label: "" } },
  legend: null,
  corner: t => ["StoneLab · water capture tube", D3 && cornerIn3d(t) ? "Air at 1/40 speed" : "Air and drops at ⅕ speed"],
  async setup(ctx) {
    const [w, a] = await Promise.all([base.setup(ctx), fetch(new URL("./device3d_reel.json", import.meta.url)).then(r => r.json())]);
    GW = w.G; D3 = bake3d(a, GW); D3.ctx = ctx;
    return w;
  },
  layers: { ...base.layers, hardware, device3d },
  shots, captions, callouts,
};

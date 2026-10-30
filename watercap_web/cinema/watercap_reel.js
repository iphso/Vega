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
  onset:   ["corona", "Corona onset", "An ionization-integral onset is computed on every disc edge."],
  vi:      ["corona", "Ion drift & space charge", "A self-consistent ion field and current, with mobility tuned for humid air."],
  charge:  ["corona", "Droplet charging", "Fog droplets charge by field and diffusion charging, then drift to the wall."],
  deposit: ["corona", "Wall capture", "Charged fog lands on the grounded wall and drains away."],
  quench:  ["corona", "Mist quenching", "Dense fog chokes the corona, and the solver finds the self-consistent state."],
  coag:    ["spray", "Spray scavenging", "Falling drops sweep fog out of the air, and the captured fog follows its drop."],
  energy:  ["energy", "Full energy accounting", "Corona, pump and recooling power are reported per litre recovered."],
};

/* ---- the 3D corona tier: three pitches of the tabbed-ring stack (device3d_reel.json, baked from the
   simulator's own LES tables by StoneLab apps/device3d/scripts/reel-asset.ts), drawn off to the side of the tube
   world at X3 and scaled up by K3 so the camera works at the film's usual distances */
const X3 = 40, K3 = 10, NP = 3, NA = 700, AS = 64, TS3 = 0.025;   // air at 1/40 speed: 2 m/s crosses a pitch in ~0.6 s
let D3 = null;
const dec = (b, T) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
const at3 = (x, y, z) => new THREE.Vector3(X3 + x * 1e-3 * K3, z * 1e-3 * K3, -y * 1e-3 * K3);   // solver mm -> world

function bake3d(a) {
  const P = dec(a.mesh.pos, Int16Array), N = dec(a.mesh.nrm, Int16Array), I = dec(a.mesh.idx, Uint32Array);
  const p = new Float32Array(P.length), n = new Float32Array(N.length);
  for (let i = 0; i < P.length; i += 3) {
    p[i] = P[i] / a.mesh.scale; p[i + 1] = P[i + 2] / a.mesh.scale; p[i + 2] = -P[i + 1] / a.mesh.scale;   // mm, y up
    n[i] = N[i] / 32767; n[i + 1] = N[i + 2] / 32767; n[i + 2] = -N[i + 1] / 32767;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(p, 3)); geo.setAttribute("normal", new THREE.BufferAttribute(n, 3)); geo.setIndex(new THREE.BufferAttribute(I, 1));
  const ions = Array.from(dec(a.ions.xyzn, Int16Array), v => v / a.ions.scale);
  // air tracers: integrate the LES mean flow (periodic over the pitch) once; the film plays the paths back
  const F = a.flow, U = dec(F.uvw, Int16Array), L = NP * a.pitch, dtS = 0.0012;
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
    let x = r * Math.cos(th), y = r * Math.sin(th), z = -0.4 * a.pitch + h01(i, 73) * L;
    for (let k = 0; k < AS; k++) {
      const o = 3 * (i * AS + k); path[o] = x; path[o + 1] = y; path[o + 2] = z;
      for (let s = 0; s < 4; s++) {
        vel(x, y, z, v); x += v[0] * dtS / 4; y += v[1] * dtS / 4; z += v[2] * dtS / 4;
        const rr = Math.hypot(x, y); if (rr > a.R * 0.985) { x *= a.R * 0.985 / rr; y *= a.R * 0.985 / rr; }
      }
    }
  }
  return { a, geo, ions, path, dtS, L };
}

const device3d = {
  init(ctx) {
    const a = D3.a, g = this.g = new THREE.Group(); ctx.scene.add(g);
    const steel = new THREE.MeshStandardMaterial({ color: 0x8a929c, metalness: 0.55, roughness: 0.62, envMapIntensity: 0.4, side: THREE.DoubleSide });
    for (let k = 0; k < NP; k++) {
      const m = new THREE.Mesh(D3.geo, steel);
      m.scale.setScalar(1e-3 * K3); m.position.set(X3, (k * a.pitch + a.plateZ - 0.25) * 1e-3 * K3, 0); g.add(m);
    }
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.004 * K3, 0.004 * K3, (NP + 0.6) * a.pitch * 1e-3 * K3, 32), steel);
    rod.position.set(X3, NP * a.pitch * 0.5e-3 * K3, 0); g.add(rod);
    this.tube = new THREE.ShaderMaterial({
      uniforms: { op: { value: 1 } },
      vertexShader: `varying vec3 vN; varying vec3 vV; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `uniform float op; varying vec3 vN; varying vec3 vV;
        void main() { float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 3.0); vec3 c = vec3(0.55, 0.65, 0.78) * (0.02 + 0.45 * f); gl_FragColor = vec4(c * op, (0.02 + 0.4 * f) * op); }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
    });
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(a.R * 1e-3 * K3, a.R * 1e-3 * K3, D3.L * 1e-3 * K3, 128, 1, true), this.tube);
    tube.position.set(X3, D3.L * 0.5e-3 * K3, 0); tube.renderOrder = 5; g.add(tube);
    // ion cloud, one copy per plate; brightness by log density (amber -> white hot)
    const n = D3.ions.length / 4; let hi = -1e9; for (let i = 3; i < D3.ions.length; i += 4) hi = Math.max(hi, D3.ions[i]);
    this.ions = softPoints(n * NP, { depthFade: [0.2, 0.5] }); this.ions.renderOrder = 6; g.add(this.ions);
    const A = this.ions.geometry.attributes; this.ionA = new Float32Array(n * NP);
    for (let k = 0; k < NP; k++) for (let i = 0; i < n; i++) {
      const q = 4 * i, j = k * n + i, u = clamp((D3.ions[q + 3] - (hi - 3)) / 3, 0, 1), jx = (h01(j, 81) - 0.5) * 0.98, jy = (h01(j, 82) - 0.5) * 0.98, jz = (h01(j, 83) - 0.5) * 0.98,   // break up the 0.98 mm cell lattice
        p = at3(D3.ions[q] + jx, D3.ions[q + 1] + jy, D3.ions[q + 2] + jz + k * a.pitch);
      A.position.setXYZ(j, p.x, p.y, p.z); A.color.setXYZ(j, 1.0 + 0.9 * u * u, 0.42 + 0.7 * u * u, 0.12 + 0.5 * u * u * u);
      A.size.setX(j, 0.0011 * K3); this.ionA[j] = 0.012 + 0.14 * u * u;
    }
    this.ions.commit(n * NP);
    // tab tips: the glow every plume starts from
    this.tips = softPoints(a.apices.length * NP); this.tips.renderOrder = 7; g.add(this.tips);
    a.apices.forEach((ap, i) => { for (let k = 0; k < NP; k++) { const p = at3(ap[0], ap[1], ap[2] + k * a.pitch + a.plateZ - 0.25), j = k * a.apices.length + i;
      this.tips.geometry.attributes.position.setXYZ(j, p.x, p.y, p.z); this.tips.geometry.attributes.color.setXYZ(j, 2.6, 1.7, 0.9); this.tips.geometry.attributes.size.setX(j, 0.006 * K3); } });
    this.air = trailSegments(NA * 5); this.air.renderOrder = 6; g.add(this.air);
  },
  update(scene, t, dt, p) {
    const w = p.w, vis = w > 0.003; this.g.visible = vis; if (!vis) return;
    this.tube.uniforms.op.value = w;
    const glow = p.get("glow3", 1) * w, A = this.ions.geometry.attributes;
    for (let j = 0; j < this.ionA.length; j++) A.alpha.setX(j, this.ionA[j] * glow);
    A.alpha.needsUpdate = true;
    const T = this.tips.geometry.attributes;
    for (let j = 0; j < T.alpha.count; j++) T.alpha.setX(j, glow * (0.55 + 0.25 * Math.sin(t * 19 + j * 2.3) * Math.sin(t * 7.1 + j)));
    this.tips.commit();
    // air: each tracer replays its baked path on its own phase; fades in and out over its life
    const B = this.air.geometry.attributes, life = (AS - 6) * D3.dtS, air = p.get("air3", 1) * w; let s = 0;
    for (let i = 0; i < NA; i++) {
      const u = (t * TS3 / life + h01(i, 74)) % 1, f = u * (AS - 6) + 5, k0 = Math.floor(f), fr = f - k0;
      const al = air * 0.9 * Math.sin(Math.PI * u);
      const pt = k => { const o = 3 * (i * AS + k), o2 = o + 3; return at3(D3.path[o] + fr * (D3.path[o2] - D3.path[o]), D3.path[o + 1] + fr * (D3.path[o2 + 1] - D3.path[o + 1]), D3.path[o + 2] + fr * (D3.path[o2 + 2] - D3.path[o + 2])); };
      let prev = pt(k0);
      if (prev.y < -0.001 || prev.y > D3.L * 1e-3 * K3) continue;
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

const cap = (shot, kicker, title, line, inT) => ({ shot, kicker, title, line, status: null, in: inT ?? 0.4 });

function captions(ctx) {
  return [
    cap("inlet", "Inlet", "Warm, misty exhaust rises", "Saturated exhaust enters at the bottom, carrying fog and vapour."),
    cap("nozzle", "Spray stage", "A cold spray meets it", "Chilled water is sprayed down through the rising air."),
    cap("heat", "Spray stage", "Vapour condenses on every drop", "Drops run from cold blue to warm as they pull water out of the air."),
    cap("film", "Spray stage", "The wall drains it away", "Spray and condensate that reach the wall fall to the sump as a film."),
    cap("entrain", "Spray stage", "Nothing gets lost", "Fine drops the updraft lifts are followed into the next stage."),
    cap("corona", "Corona stage", "Ions stream from every disc", "High voltage on the central rod fills the gap with ions."),
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
  const P = D3.a.pitch * 1e-3 * K3, R = D3.a.R * 1e-3 * K3;
  const in3d = { id: "in3d", chapter: "In 3D", duration: 10, cut: true,
    layers: { stage: 1, device3d: 1 },
    params: { glow3: [[0.6, 0], [2.2, 1]], air3: [[0, 0.6], [1, 1]], cut: 0.7, legend: 0, shiftX: 0.0 },
    camera: [{ at: 0, target: [X3, 1.5 * P, 0], az: 30, el: 52, dist: 4.6 * R, fov: 30, cut: true },
             { at: 5, target: [X3, 1.6 * P, 0], az: 70, el: 30, dist: 3.6 * R, fov: 30 },
             { at: 10, target: [X3, 1.5 * P, 0], az: 112, el: 14, dist: 3.4 * R, fov: 30, ease: true }] };
  const end = { id: "end", chapter: "The simulator", duration: 8,
    layers: { stage: 1, shell: 1, hardware: 1, mist: 0.6, plane: 0.3, spray: 0.8, film: 1, corona: 1, outlet: 0.6 },
    params: { legend: 0, hud: [[0, 1], [1.5, 0]], shiftX: [[0, 0.0], [2.5, 0.22]] },
    camera: [{ at: 8, target: [0, 1.6, 0], az: 330, el: 9, dist: 9.0, fov: 28, ease: true }],
    titleCard: { kicker: "StoneLab simulator", title: "Every design, any fidelity", sub: "Fast · Resolved · 3D quick · 3D full",
      meta: "Energy per litre · water captured · pressure drop · arc margin", in: 0.8, out: 7.6 } };
  const out = [];
  for (const s0 of base.shots(ctx)) {
    let s = s0;
    if (s.id === "open") s = { ...s, chapter: "The tube",
      titleCard: { ...s.titleCard, kicker: "StoneLab", title: "Water capture, simulated",
        sub: "Warm, humid exhaust goes in at the bottom. A cold spray and then a corona pull the water out of it, and drier air leaves the top.",
        meta: "Spray · film · fog · corona · energy, all in one coupled model" } };
    if (s.id === "inlet") s = { ...s, chapter: "Inlet" };
    if (s.id === "overview") s = { ...s, chapter: "What's inside", params: { ...s.params, legend: 0 } };
    if (s.id === "outlet") s = { ...s, cut: true, camera: [{ at: 0, target: [0, s.camera[0].target[1], 0], az: 238, el: 21, dist: 1.95, cut: true }, ...s.camera] };
    if (s.id === "outlet") out.push(in3d);
    out.push(s);
  }
  out.push(end);
  return out;
}

const cornerIn3d = t => { const c = D3.ctx; return c.at && t >= c.at("in3d", 0) && t < c.at("outlet", 0); };

export default {
  ...base,
  title: "Water Capture Reel",
  fileName: "watercap-reel",
  status: { inc: { color: STAGE.spray, label: "" } },
  legend: null,
  corner: t => ["StoneLab · water capture tube", D3 && cornerIn3d(t) ? "One electrode pitch × 3 · air at 1/40 speed" : "Air and drops at ⅕ speed"],
  async setup(ctx) {
    const [w, a] = await Promise.all([base.setup(ctx), fetch(new URL("./device3d_reel.json", import.meta.url)).then(r => r.json())]);
    D3 = bake3d(a); D3.ctx = ctx;
    return w;
  },
  layers: { ...base.layers, device3d },
  shots, captions, callouts,
};

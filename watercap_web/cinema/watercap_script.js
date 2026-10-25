// The watercap film: a ~90 s explainer of the water capture tube, driven by cinema.js.
// Geometry, air velocity, temperature and humidity come from the hero design's scene file
// (watercap_scene.json, a copy of Vega/viewer/public/data/watercap/scenes/rain_curtain-0067.json);
// every physics callout's status and evidence comes from ../validation_data.js, the page's single source of truth.
import * as THREE from "three";
import { softPoints, trailSegments, smooth, clamp, lerp, hash01 as h01 } from "./cinema.js";
import { geometry, makeFields, mistTables, zAt, tauAt, sprayDrops, dropAt, filmProfile } from "./watercap_physics.js";
import { ST, PHYS } from "../validation_data.js";

const TS = 0.2;                 // sim seconds per film second: air and drops at 1/5 speed
const VF = 0.32;                // wall-film drainage speed, m/s (sim)
const C = { water: 0x5b9dff, spray: 0x3ccfb4, volt: 0xffb547, glass: 0x9fb3c8, metal: 0xc4ccd6 };
const VAPOR = [[8, 12, 16], [14, 42, 54], [20, 80, 92], [40, 128, 132], [86, 176, 168], [150, 214, 200]];
const lin = c => new THREE.Color(c);                       // THREE.Color(hex) is linear (sRGB-decoded)
const P = Object.fromEntries(PHYS.map(p => [p.id, p]));

// evidence line for a callout: the first sentence when it says enough, else the whole entry
function ev(id) { const e = P[id].ev, i = e.indexOf(". "); return i > 28 ? e.slice(0, i + 1) : e; }

let W = null;                    // the world: geometry, fields, tables, built in setup()

/* ======================================================================= setup */
async function setup(ctx) {
  const sc = await (await fetch(new URL("./watercap_scene.json", import.meta.url))).json();
  const G = geometry(sc), F = makeFields(sc);
  const mist = mistTables(G, F);
  const drops = sprayDrops(G, F, sc);
  const film = filmProfile(G, drops);
  const sump = drops.filter(d => d.fate === 0), Tsump = sump.reduce((a, d) => a + d.p[d.p.length - 1], 0) / Math.max(1, sump.length);
  W = { sc, G, F, mist, drops, film, Tsump, ctx };
  return W;
}

// a point at height y, radius r, on the camera's right (+1) or left (−1) side of the axis
function side(y, r, s = 1) {
  return (t, ctx) => { const c = ctx.camera.position, a = Math.atan2(c.x, c.z) + (s > 0 ? Math.PI / 2 : -Math.PI / 2); return new THREE.Vector3(r * Math.sin(a), y, r * Math.cos(a)); };
}
const camAz = ctx => Math.atan2(ctx.camera.position.x, ctx.camera.position.z);

/* ======================================================================= layers */
// stage: background gradient, floor, lights, reflections. Always on.
const stage = {
  init(ctx) {
    const { scene, renderer } = ctx, G = W.G;
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.9999, 1.0); }`,
      fragmentShader: `varying vec2 vUv; uniform float aspect;
        vec3 l(vec3 c) { return pow(c, vec3(2.2)); }
        void main() { vec2 q = (vUv - vec2(0.5, 0.56)) * vec2(aspect, 1.0); float d = length(q);
          vec3 c = mix(l(vec3(0.085, 0.092, 0.115)), l(vec3(0.024, 0.024, 0.03)), smoothstep(0.0, 1.05, d));
          c += l(vec3(0.03, 0.045, 0.07)) * (1.0 - smoothstep(0.0, 0.6, abs(vUv.y - 0.62))) * 0.6;
          gl_FragColor = vec4(c, 1.0); }`,
      uniforms: { aspect: { value: 16 / 9 } }, depthWrite: false, depthTest: false }));
    bg.frustumCulled = false; bg.renderOrder = -1000; scene.add(bg);
    // floor: dark disc with a faint grid and a contact shadow
    const fl = new THREE.Mesh(new THREE.CircleGeometry(5, 96), new THREE.ShaderMaterial({
      vertexShader: `varying vec3 vP; void main() { vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `varying vec3 vP; uniform float R;
        float grid(vec2 p, float s) { vec2 g = abs(fract(p / s - 0.5) - 0.5) * s / fwidth(p); return 1.0 - smoothstep(0.0, 1.0, min(g.x, g.y)); }
        void main() { float r = length(vP.xy);
          float a = (1.0 - smoothstep(0.6, 4.5, r));
          vec3 c = vec3(0.012, 0.013, 0.017);
          float gl = grid(vP.xy, 0.25) * 0.5 + grid(vP.xy, 1.0) * 0.5;
          c += vec3(0.05, 0.065, 0.09) * gl * (1.0 - smoothstep(0.4, 3.2, r));
          float sh = 1.0 - smoothstep(R * 0.9, R * 3.2, r);
          c *= 1.0 - 0.85 * sh;
          gl_FragColor = vec4(c, a * 0.96); }`,
      uniforms: { R: { value: G.R } }, transparent: true, depthWrite: false }));
    fl.rotation.x = -Math.PI / 2; fl.position.y = -0.22; fl.renderOrder = -10; scene.add(fl);
    // lights and an environment for the metal
    scene.add(new THREE.HemisphereLight(0xc9d6ff, 0x0a0a10, 0.55));
    const key = new THREE.DirectionalLight(0xfff1e0, 2.2); key.position.set(3, 6, 4); scene.add(key);
    const rim = new THREE.DirectionalLight(0x7fa6ff, 1.6); rim.position.set(-4, 3, -3); scene.add(rim);
    const env = new THREE.Scene();
    env.add(new THREE.Mesh(new THREE.SphereGeometry(10, 32, 16), new THREE.ShaderMaterial({ side: THREE.BackSide,
      vertexShader: `varying vec3 vP; void main() { vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `varying vec3 vP; void main() { float y = normalize(vP).y; gl_FragColor = vec4(mix(vec3(0.02, 0.02, 0.025), vec3(0.32, 0.36, 0.45), smoothstep(-0.3, 0.9, y)), 1.0); }` })));
    for (const [x, y, z, s, c] of [[5, 4, 3, 3, 5], [-5, 2, -4, 2.5, 2.2], [0, 8, 0, 4, 1.0]]) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(s, s * 0.6), new THREE.MeshBasicMaterial({ color: new THREE.Color(c, c * 0.97, c * 0.92), side: THREE.DoubleSide }));
      m.position.set(x, y, z); m.lookAt(0, 0, 0); env.add(m);
    }
    const pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromScene(env, 0.03).texture; scene.environmentIntensity = 0.75;
    this.bg = bg;
  },
  update(scene, t, dt, p) { this.bg.material.uniforms.aspect.value = p.ctx.camera.aspect; },
};

// shell: the glass tube, cut away toward the camera, with a faint measurement grid; collars, inlet duct
const shell = {
  init(ctx) {
    const G = W.G, g = this.g = new THREE.Group(); ctx.scene.add(g);
    this.mat = new THREE.ShaderMaterial({
      uniforms: { color: { value: lin(C.glass) }, camAz: { value: 0 }, cutHalf: { value: 0.6 }, R: { value: G.R }, op: { value: 1 }, Ls: { value: G.Ls }, Lh: { value: G.L / 2 } },
      vertexShader: `varying vec3 vN; varying vec3 vV; varying vec3 vP;
        void main() { vP = position; vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `uniform vec3 color; uniform float camAz; uniform float cutHalf; uniform float R; uniform float op; uniform float Ls; uniform float Lh;
        varying vec3 vN; varying vec3 vV; varying vec3 vP;
        void main() {
          float phi = atan(vP.x, vP.z); float d = abs(mod(phi - camAz + 3.14159265, 6.2831853) - 3.14159265);
          if (d < cutHalf) discard;
          float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.5);
          float y = vP.y + Lh;
          float gz = abs(fract(y / 0.25 - 0.5) - 0.5) * 0.25 / fwidth(y);
          float s = phi * R; float sp = 6.2831853 * R / 24.0; float gt = abs(fract(s / sp - 0.5) - 0.5) * sp / fwidth(s);
          float grid = (1.0 - smoothstep(0.0, 1.0, gz)) * 0.5 + (1.0 - smoothstep(0.0, 1.0, gt)) * 0.3;
          float edge = 1.0 - smoothstep(0.0, 1.6, (d - cutHalf) * R / fwidth(d * R));
          float ls = 1.0 - smoothstep(0.0, 1.4, abs(y - Ls) / fwidth(y));
          vec3 c = color * (0.03 + 0.5 * f) + color * grid * 0.06 + vec3(0.8, 0.9, 1.0) * edge * 0.55 + vec3(1.0, 0.71, 0.28) * ls * 0.35;
          float a = 0.025 + 0.42 * f + grid * 0.05 + edge * 0.6 + ls * 0.3;
          gl_FragColor = vec4(c * op, a * op); }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
    });
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(G.R, G.R, G.L, 160, 1, true), this.mat);
    tube.position.y = G.L / 2; tube.renderOrder = 5; g.add(tube);
    const metal = new THREE.MeshStandardMaterial({ color: 0x3a3f48, metalness: 0.85, roughness: 0.38 });
    const band = (y0, y1, rr = G.R * 1.035) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(rr, rr, y1 - y0, 128, 1, true), metal); m.position.y = (y0 + y1) / 2; m.material.side = THREE.DoubleSide; g.add(m); return m; };
    band(-0.22, 0.0); band(G.L - 0.005, G.L + 0.045);
    // flanges
    for (const y of [0.0, G.L - 0.005]) { const r = new THREE.Mesh(new THREE.TorusGeometry(G.R * 1.04, 0.006, 8, 128), metal); r.rotation.x = Math.PI / 2; r.position.y = y; g.add(r); }
    // inlet duct into the base
    const duct = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.7, 48, 1, true), metal);
    duct.rotation.z = Math.PI / 2; duct.position.set(-G.R - 0.33, -0.11, 0); g.add(duct);
    const cap = new THREE.Mesh(new THREE.CircleGeometry(G.R * 1.035, 96), metal); cap.rotation.x = Math.PI / 2; cap.position.y = -0.22; g.add(cap);
    this.metal = metal;
  },
  update(scene, t, dt, p) {
    const w = p.w; this.g.visible = w > 0.002;
    const u = this.mat.uniforms; u.camAz.value = camAz(p.ctx); u.cutHalf.value = p.get("cut", 0.6) * 1.15; u.op.value = w;
  },
};

// hardware: HV rod, disc stack, nozzle and its feed, top spider with insulator
const hardware = {
  init(ctx) {
    const G = W.G, g = this.g = new THREE.Group(); ctx.scene.add(g);
    const steel = new THREE.MeshStandardMaterial({ color: 0xa9b1bc, metalness: 0.9, roughness: 0.3, envMapIntensity: 0.7 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x4a515c, metalness: 0.85, roughness: 0.35 });
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, G.Lc + 0.09, 16), steel);
    rod.position.y = G.Ls + (G.Lc + 0.09) / 2 - 0.02; g.add(rod);
    const th = Math.max(G.t, 0.0011);
    const discMat = new THREE.MeshStandardMaterial({ color: 0x6b727c, metalness: 0.8, roughness: 0.55, envMapIntensity: 0.3 });
    const discs = new THREE.InstancedMesh(new THREE.CylinderGeometry(G.rd, G.rd, th, 128), discMat, G.discs.length);
    const m = new THREE.Matrix4();
    G.discs.forEach((y, i) => discs.setMatrixAt(i, m.makeTranslation(0, y, 0)));
    g.add(discs);
    // nozzle: body + cone, fed by a pipe from the wall
    const tint = new THREE.MeshStandardMaterial({ color: 0x2a9c88, metalness: 0.6, roughness: 0.3 });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.026, 32), tint); body.position.y = G.nozzleZ + 0.015; g.add(body);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.011, 0.016, 32), tint); cone.rotation.x = Math.PI; cone.position.y = G.nozzleZ - 0.004; g.add(cone);
    const feed = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, G.R, 16), dark);
    feed.rotation.z = Math.PI / 2; feed.position.set(G.R / 2, G.nozzleZ + 0.024, 0); g.add(feed);
    // spider and insulator at the top
    const bar = new THREE.Mesh(new THREE.BoxGeometry(2 * G.R * 1.03, 0.01, 0.014), dark); bar.position.y = G.L + 0.05; g.add(bar);
    const ins = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.014, 0.07, 32), new THREE.MeshPhysicalMaterial({ color: 0xe9e4da, roughness: 0.35, clearcoat: 0.6 }));
    ins.position.y = G.L + 0.02; g.add(ins);
    for (let k = 0; k < 4; k++) { const sh = new THREE.Mesh(new THREE.CylinderGeometry(0.019, 0.019, 0.004, 32), ins.material); sh.position.y = G.L + 0.0 + k * 0.014; g.add(sh); }
  },
  update(scene, t, dt, p) { this.g.visible = p.w > 0.01; },
};

// the cut-away plane through the axis, coloured by the solved humidity field (texture a, the inlet value everywhere, is kept for scripts that want to blend)
const plane = {
  init(ctx) {
    const G = W.G, F = W.F, w0 = F.wOut, w1 = F.wIn;
    this.lo = w0; this.hi = w1;
    const paint = solved => {
      const cv = document.createElement("canvas"); cv.width = 128; cv.height = 768;
      const g = cv.getContext("2d"), img = g.createImageData(cv.width, cv.height);
      for (let row = 0; row < cv.height; row++) {
        const z = (1 - (row + 0.5) / cv.height) * G.L;
        for (let col = 0; col < cv.width; col++) {
          const r = Math.abs((col + 0.5) / cv.width * 2 - 1) * G.R;
          const w = solved ? F.sample("w", r, z) : w1;
          const u = clamp((w - w0) / (w1 - w0), 0, 1), x = u * (VAPOR.length - 1), i = Math.min(VAPOR.length - 2, Math.floor(x)), f = x - i;
          const k = 4 * (row * cv.width + col), edge = smooth((G.R - r) / (G.R * 0.06));
          for (let c = 0; c < 3; c++) img.data[k + c] = 0.8 * (VAPOR[i][c] + (VAPOR[i + 1][c] - VAPOR[i][c]) * f);
          img.data[k + 3] = (14 + 100 * Math.pow(u, 1.4)) * edge;
        }
      }
      g.putImageData(img, 0, 0);
      const tx = new THREE.CanvasTexture(cv); tx.colorSpace = THREE.SRGBColorSpace; tx.minFilter = THREE.LinearFilter; return tx;
    };
    this.mat = new THREE.ShaderMaterial({
      uniforms: { a: { value: paint(false) }, b: { value: paint(true) }, mixv: { value: 0 }, op: { value: 1 } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D a; uniform sampler2D b; uniform float mixv; uniform float op; varying vec2 vUv;
        void main() { vec4 c = mix(texture2D(a, vUv), texture2D(b, vUv), mixv); gl_FragColor = vec4(c.rgb * c.a * op * 0.9, c.a * op * 0.9); }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    this.m = new THREE.Mesh(new THREE.PlaneGeometry(2 * G.R * 0.995, G.L), this.mat);
    this.m.position.y = G.L / 2; this.m.renderOrder = 1; ctx.scene.add(this.m);
  },
  update(scene, t, dt, p) {
    this.m.visible = p.w > 0.003; this.mat.uniforms.op.value = p.w * 0.85; this.mat.uniforms.mixv.value = 1;   // the plane only ever shows the solved field; it fades in once the spray runs
    this.m.rotation.y = camAz(p.ctx);
  },
};

/* ---- mist: fog droplets riding the solved updraft; charged and driven to the wall in the corona stage */
const NM = 3000, NB = 4200;           // base mist, extra "burst" mist (the dense slug that quenches the corona)
function mistInfo(i) {
  const M = W.mist, b = Math.min(M.length - 1, Math.floor(h01(i, 1) * M.length)), tab = M[b], G = W.G;
  const per = tab.tauEnd + 0.15 + h01(i, 4) * 0.3;
  const capt = h01(i, 6) < 0.94;
  const zc = G.Ls + 0.04 + Math.min(G.Lc - 0.08, -Math.log(1 - 0.995 * h01(i, 7)) * 0.3);
  return { b, tab, per, phi: h01(i, 5) * per, th0: h01(i, 3) * 6.283, capt, zc, tauC: tauAt(tab, zc), tauLs: tauAt(tab, G.Ls), D: 0.25 + 0.45 * h01(i, 8), burst: i >= NM };
}
let MI = null;
const _v = new THREE.Vector3();
// state of mist particle i at film time t → writes into o; returns false when not present
function mistState(i, t, ctx, o) {
  const m = MI[i], G = W.G, S = t * TS;
  const tau = ((S + m.phi) % m.per + m.per) % m.per;
  const tIn = t - tau / TS;
  let slug = 1;
  if (m.burst) { slug = ctx.param("burst", t, 0) - h01(i, 9); if (slug <= 0) return false; slug = Math.min(1, slug * 4); }
  let z = zAt(m.tab, tau), r = m.tab.r * (1 + 0.035 * Math.sin(1.7 * tau + 6.283 * h01(i, 10)));
  const th = m.th0 + 0.14 * Math.sin(0.6 * tau + 6.283 * h01(i, 11));
  o.state = 0; o.sig = 0; o.a = 1;
  if (m.capt && tau > m.tauC) {
    const tC = t - (tau - m.tauC) / TS;
    if (ctx.param("glow", tC, 0) > 0.12 + 0.75 * h01(i, 12)) {
      const sig = (tau - m.tauC) / m.D;
      if (sig >= 1.25) return false;
      const zc = zAt(m.tab, m.tauC);
      z = zc + (z - zc) * 0.35;                             // the drift is mostly radial
      r = lerp(r, G.R * 0.985, Math.pow(Math.min(sig, 1), 1.7));
      o.state = sig < 1 ? 1 : 2; o.sig = sig; o.a = sig < 1 ? 1 : 1 - (sig - 1) / 0.25;
    }
  }
  if (z > G.L) { o.state = 3; o.a = 1 - smooth((z - G.L) / 0.6); }
  if (m.burst) { const k = slug * smooth((z - G.Ls + 0.35) / 0.25) * (1 - smooth((z - G.Ls - 0.35) / 0.3)); if (k <= 0.002) return false; o.a *= k; }
  o.x = r * Math.sin(th); o.y = z; o.z = r * Math.cos(th); o.r = r; o.tau = tau;
  return true;
}
const mist = {
  init(ctx) {
    MI = Array.from({ length: NM + NB }, (_, i) => mistInfo(i));
    this.pts = softPoints(NM + NB); this.pts.renderOrder = 3; ctx.scene.add(this.pts);
    this.o = {};
  },
  update(scene, t, dt, p) {
    const pts = this.pts; pts.visible = p.w > 0.003; if (!pts.visible) return;
    const A = pts.geometry.attributes, F = W.F, G = W.G, o = this.o, sol = p.get("solved", 1), wr = F.wIn - F.wOut;
    const cBlue = lin(C.water), cWarm = new THREE.Color(0.9, 0.86, 0.8), cEsc = lin(0xff9a4d);
    let n = 0;
    for (let i = 0; i < NM + NB; i++) {
      if (!mistState(i, t, p.ctx, o)) continue;
      A.position.setXYZ(n, o.x, o.y, o.z);
      let r, g, b, a = 0.42 * o.a, s = 0.0042 + 0.0022 * h01(i, 13);
      if (o.state === 1 || o.state === 2) {      // charged: an amber glint that brightens toward the wall
        const k = smooth(o.sig * 2.5), gl = 1 + 0.9 * smooth(o.sig * 4) * (1 - 0.5 * o.sig);
        r = lerp(cBlue.r, 1.0, k) * gl; g = lerp(cBlue.g, 0.62, k) * gl; b = lerp(cBlue.b, 0.22, k) * gl; a = (0.45 + 0.35 * k) * o.a; s *= 0.95;
      } else {
        const wn = lerp(1, clamp((F.sample("w", o.r, clamp(o.y, 0, G.L)) - F.wOut) / wr, 0, 1), sol);
        r = lerp(cBlue.r, cWarm.r, wn * 0.8); g = lerp(cBlue.g, cWarm.g, wn * 0.8); b = lerp(cBlue.b, cWarm.b, wn * 0.8);
        if (o.state === 3) { const k = smooth((o.y - G.L) / 0.25); r = lerp(r, cEsc.r, k); g = lerp(g, cEsc.g, k); b = lerp(b, cEsc.b, k); }
        a *= smooth((o.y + 0.02) / 0.12);
        if (MI[i].burst) { a *= 1.5; s *= 1.3; r = lerp(r, 0.85, 0.5); g = lerp(g, 0.9, 0.5); b = lerp(b, 1.0, 0.5); }
      }
      A.color.setXYZ(n, r, g, b); A.alpha.setX(n, a * p.w); A.size.setX(n, s);
      n++;
    }
    pts.commit(n);
  },
  // follow mist particle i (charged path); after it deposits, the anchor stays where it hit the wall
  track(i, hold = true) {
    const o = {}, m = MI[i];
    return (t, ctx) => {
      const ok = mistState(i, t, ctx, o);
      if (ok && (o.state === 0 || o.state === 1 || o.state === 3)) return new THREE.Vector3(o.x, o.y, o.z);
      if (!hold) return null;
      // deposited: the wall point where it landed (a pure function of t, so seeking lands on the same frame)
      const tau = ((t * TS + m.phi) % m.per + m.per) % m.per;
      if (tau < m.tauC + m.D) return ok ? new THREE.Vector3(o.x, o.y, o.z) : null;
      if (!mistState(i, t - (tau - (m.tauC + m.D * 0.999)) / TS, ctx, o)) return null;
      return new THREE.Vector3(o.x, o.y, o.z);
    };
  },
  // a capturable particle that is charged at tc, rising in view before, on the camera-facing side
  findCharged(ctx, tc, { rMin = 0.08, rMax = 0.15, zMin, zMax } = {}) {
    const G = W.G, cam = ctx.timeline.camera(tc), az = cam[3] * Math.PI / 180, o = {};
    let best = -1, bs = Infinity;
    for (let i = 0; i < NM; i++) {
      const m = MI[i];
      if (!m.capt || m.tab.r < rMin || m.tab.r > rMax) continue;
      if (!mistState(i, tc, ctx, o) || o.state !== 1 || o.sig > 0.12) continue;
      if (!mistState(i, tc - 2.0, ctx, o) || o.state !== 0) continue;
      if (zMin != null && (m.zc < zMin || m.zc > zMax)) continue;
      const th = m.th0, da = Math.abs(((th - az + Math.PI) % 6.283 + 6.283) % 6.283 - Math.PI);
      const s = Math.abs(da - 1.2) + 0.5 * o.sig;
      if (s < bs) { bs = s; best = i; }
    }
    return best;
  },
};

/* ---- spray drops: falling, warming, hitting the wall and draining as film, or carried up */
const NT = 6, TRAIL = 0.011;          // trail segments per drop, sim seconds between trail samples
function dropInfo(i) {
  const D = W.drops[i], filmDur = D.fate === 1 ? Math.max(0, D.hit.y) / VF : 0;
  const per = D.dur + filmDur + 0.25 + 0.6 * h01(i, 21);
  return { D, filmDur, per, phi: h01(i, 22) * per };
}
let DI = null;
// position of drop i at age τ (sim s) since emission: flight, then film; writes o; false if gone
function dropPos(i, tau, o) {
  const m = DI[i], D = m.D, G = W.G;
  if (tau < 0) return false;
  if (tau <= D.dur) { dropAt(D, tau, o.p); o.film = false; o.T = o.p[3]; return true; }
  if (D.fate !== 1) return false;
  const a = tau - D.dur, y = D.hit.y - VF * a;
  if (y <= 0) return false;
  const th = D.hit.th + 0.012 * Math.sin(a * 3 + i), r = G.R * 0.982;
  o.p[0] = r * Math.cos(th); o.p[1] = y; o.p[2] = r * Math.sin(th); o.film = true; o.T = D.hit.T;
  return true;
}
function dropTau(i, t, ctx) {           // age of drop i's current cycle, or −1 before the spray first starts it
  const m = DI[i], S = t * TS;
  const k = Math.floor((S - m.phi) / m.per), tau = S - m.phi - k * m.per;
  const tE = t - tau / TS;
  if (ctx.param("sprayOn", tE, 0) <= h01(i, 23) * 0.98) return -1;
  return tau;
}
const T_COLD = 9, T_WARM = 22;
function tempColor(T, out) {            // cold blue → teal → warm amber
  const u = clamp((T - T_COLD) / (T_WARM - T_COLD), 0, 1);
  const a = [0.22, 0.48, 1.15], b = [0.08, 0.72, 0.56], c = [1.2, 0.5, 0.14];
  if (u < 0.5) { const k = u / 0.5; for (let j = 0; j < 3; j++) out[j] = lerp(a[j], b[j], k); }
  else { const k = (u - 0.5) / 0.5; for (let j = 0; j < 3; j++) out[j] = lerp(b[j], c[j], k); }
  return out;
}
const spray = {
  init(ctx) {
    DI = W.drops.map((_, i) => dropInfo(i));
    const n = W.drops.length;
    this.pts = softPoints(n); this.pts.renderOrder = 4; this.pts.material.uniforms.uHard.value = 0.55; ctx.scene.add(this.pts);
    this.tr = trailSegments(n * NT); this.tr.renderOrder = 4; ctx.scene.add(this.tr);
    this.o = { p: new Float32Array(4) }; this.q = { p: new Float32Array(4) }; this.col = [0, 0, 0];
  },
  update(scene, t, dt, p) {
    const vis = p.w > 0.003; this.pts.visible = this.tr.visible = vis; if (!vis) return;
    const A = this.pts.geometry.attributes, B = this.tr.geometry.attributes, o = this.o, q = this.q, col = this.col, G = W.G, ctx = p.ctx;
    const film = lin(C.water);
    let n = 0, s = 0;
    for (let i = 0; i < DI.length; i++) {
      const tau = dropTau(i, t, ctx);
      if (tau < 0 || !dropPos(i, tau, o)) continue;
      const D = DI[i].D;
      let a = 0.95;
      if (o.film) { col[0] = film.r * 0.9; col[1] = film.g * 0.95; col[2] = film.b * 1.1; a = 0.55 * smooth(o.p[1] / 0.1); }
      else {
        tempColor(o.T, col);
        if (D.entrainedAt >= 0 && tau > D.entrainedAt) { const k = smooth((tau - D.entrainedAt) / 0.2); col[0] = lerp(col[0], 1.1, k * 0.6); col[1] = lerp(col[1], 1.2, k * 0.6); col[2] = lerp(col[2], 1.25, k * 0.6); }
        if (D.fate === 0) a *= smooth((D.dur - tau) / 0.08);
        if (D.fate >= 2) a *= smooth((D.dur - tau) / 0.25);
        a *= 1 - smooth((o.p[1] - G.Ls - 0.12) / 0.45);
        a *= smooth(tau / 0.03);
      }
      A.position.setXYZ(n, o.p[0], o.p[1], o.p[2]); A.color.setXYZ(n, col[0], col[1], col[2]);
      A.alpha.setX(n, a * p.w); A.size.setX(n, o.film ? 0.006 : 0.0055 + 9 * D.d); n++;
      // trail: the same path a little earlier, fading out
      let px = o.p[0], py = o.p[1], pz = o.p[2];
      for (let k = 1; k <= NT; k++) {
        if (!dropPos(i, tau - k * TRAIL * (o.film ? 2.5 : 1), q)) break;
        const a0 = a * (1 - (k - 1) / NT) * 0.6, a1 = a * (1 - k / NT) * 0.6;
        B.position.setXYZ(2 * s, px, py, pz); B.position.setXYZ(2 * s + 1, q.p[0], q.p[1], q.p[2]);
        B.color.setXYZ(2 * s, col[0], col[1], col[2]); B.color.setXYZ(2 * s + 1, col[0], col[1], col[2]);
        B.alpha.setX(2 * s, a0 * p.w); B.alpha.setX(2 * s + 1, a1 * p.w);
        px = q.p[0]; py = q.p[1]; pz = q.p[2]; s++;
      }
    }
    this.pts.commit(n); this.tr.commit(s);
  },
  track(i) { const o = { p: new Float32Array(4) }; return (t, ctx) => { const tau = dropTau(i, t, ctx); return tau >= 0 && dropPos(i, tau, o) ? new THREE.Vector3(o.p[0], o.p[1], o.p[2]) : null; }; },
  temp(i) { const o = { p: new Float32Array(4) }; return (t, ctx) => { const tau = dropTau(i, t, ctx); return tau >= 0 && dropPos(i, tau, o) ? o.T : null; }; },
  // a drop in flight for the whole window [t0, t1], passing pred, preferring the camera-facing side
  find(ctx, t0, t1, pred) {
    const o = { p: new Float32Array(4) }, tm = (t0 + t1) / 2, az = ctx.timeline.camera(tm)[3] * Math.PI / 180;
    let best = -1, bs = Infinity;
    for (let i = 0; i < DI.length; i++) {
      const a = dropTau(i, t0, ctx), b = dropTau(i, t1, ctx), D = DI[i].D;
      if (a < 0 || b < a || b > D.dur - 0.03) continue;
      if (!pred(D, a, b)) continue;
      dropAt(D, (a + b) / 2, o.p);
      const th = Math.atan2(o.p[0], o.p[2]), r = Math.hypot(o.p[0], o.p[2]);
      const da = Math.abs(((th - az + Math.PI) % 6.283 + 6.283) % 6.283 - Math.PI);
      const s = Math.abs(da - 1.3) + (r < 0.04 ? 1 : 0) + h01(i, 99) * 0.2;
      if (s < bs) { bs = s; best = i; }
    }
    return best;
  },
};

// wall film: a thin sheet with rivulets draining down, as thick as the deposited flux above it
const filmLayer = {
  init(ctx) {
    const G = W.G, prof = W.film;
    this.mat = new THREE.ShaderMaterial({
      uniforms: { prof: { value: Array.from(prof) }, camAz: { value: 0 }, cutHalf: { value: 0.6 }, time: { value: 0 }, op: { value: 1 }, Ls: { value: G.Ls }, front: { value: 10 }, color: { value: lin(C.water) } },
      vertexShader: `varying vec3 vP; varying vec3 vN; varying vec3 vV; void main() { vP = position; vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `uniform float prof[96]; uniform float camAz; uniform float cutHalf; uniform float time; uniform float op; uniform float Ls; uniform float front; uniform vec3 color;
        varying vec3 vP; varying vec3 vN; varying vec3 vV;
        float hash(float x) { return fract(sin(x * 127.1) * 43758.5453); }
        void main() {
          float phi = atan(vP.x, vP.z); float d = abs(mod(phi - camAz + 3.14159265, 6.2831853) - 3.14159265);
          if (d < cutHalf) discard;
          float y = vP.y + Ls * 0.5;
          float k = clamp(y / Ls * 96.0, 0.0, 95.0); int i = int(k); float pf = prof[i];
          if (y > Ls) discard;
          float lane = floor(phi * 34.0); float u = fract(phi * 34.0);
          float riv = smoothstep(0.5, 0.0, abs(u - 0.5 - 0.25 * (hash(lane) - 0.5))) * step(0.35, hash(lane + 7.0));
          float flow = 0.5 + 0.5 * sin((y + time * (0.8 + 0.6 * hash(lane + 3.0))) * 40.0 + hash(lane) * 20.0);
          float sheet = 0.12 + 0.25 * pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.0);
          float a = pf * (sheet + riv * (0.35 + 0.65 * flow * flow)) * smoothstep(front, front - 0.15, Ls - y) * op;
          gl_FragColor = vec4(color * a * 1.2, a * 0.8); }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    this.m = new THREE.Mesh(new THREE.CylinderGeometry(G.R * 0.994, G.R * 0.994, G.Ls, 160, 1, true), this.mat);
    this.m.position.y = G.Ls / 2; this.m.renderOrder = 4; ctx.scene.add(this.m);
  },
  update(scene, t, dt, p) {
    const u = this.mat.uniforms, ctx = p.ctx; this.m.visible = p.w > 0.003;
    u.camAz.value = camAz(ctx); u.cutHalf.value = p.get("cut", 0.6) * 1.15 + 0.02; u.time.value = -t * TS * VF; u.op.value = p.w;
    // the film's front: how far down it has drained since the spray (last) turned on
    let on = 0; for (let k = 0; k < 40; k++) { const tt = t - k * 0.5; if (ctx.param("sprayOn", tt, 0) > 0.5) on = k * 0.5; else break; }
    u.front.value = 0.25 + on * TS * VF * 3.0;
  },
};

/* ---- corona: glowing disc edges, ions drifting to the wall; brightness follows the "glow" param */
const NI = 64;
const corona = {
  init(ctx) {
    const G = W.G, nd = G.discs.length;
    this.rings = new THREE.InstancedMesh(new THREE.TorusGeometry(G.rd, 0.0016, 8, 160), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), nd);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
    G.discs.forEach((y, i) => { this.rings.setMatrixAt(i, m.compose(new THREE.Vector3(0, y, 0), q, new THREE.Vector3(1, 1, 1))); this.rings.setColorAt(i, new THREE.Color(0, 0, 0)); });
    this.rings.renderOrder = 6; ctx.scene.add(this.rings);
    this.halo = softPoints(nd * 48); this.halo.renderOrder = 6; ctx.scene.add(this.halo);
    this.ions = softPoints(nd * NI); this.ions.renderOrder = 6; ctx.scene.add(this.ions);
    this.tr = trailSegments(nd * NI * 2); this.tr.renderOrder = 6; ctx.scene.add(this.tr);
    this.c = new THREE.Color();
  },
  ion(d, j, t, ctx, out) {        // ion j of disc d: fans out from the edge to the wall (stylised speed)
    const G = W.G, per = 0.9 + 0.3 * h01(d * 997 + j, 31), u = t / per + h01(d * 997 + j, 32);
    const sig = u - Math.floor(u), tE = t - sig * per;
    if (ctx.param("glow", tE, 0) * 1.05 < 0.15 + 0.85 * h01(d * 997 + j + Math.floor(u) * 7919, 33)) return false;
    const r = Math.sqrt(G.rd * G.rd + (G.R * G.R - G.rd * G.rd) * sig);
    const sd = h01(d * 997 + j, 34) * 2 - 1, th = h01(d * 997 + j, 35) * 6.283;
    const y = G.discs[d] + sd * G.s * 0.47 * Math.pow((r - G.rd) / (G.R - G.rd), 0.8);
    out[0] = r * Math.sin(th); out[1] = y; out[2] = r * Math.cos(th); out[3] = sig;
    return true;
  },
  update(scene, t, dt, p) {
    const G = W.G, nd = G.discs.length, ctx = p.ctx, glow = p.get("glow", 0) * p.w;
    const vis = p.w > 0.003; this.rings.visible = this.halo.visible = this.ions.visible = this.tr.visible = vis; if (!vis) return;
    for (let d = 0; d < nd; d++) {
      const fl = 0.82 + 0.18 * Math.sin(t * 23 + d * 1.7) * Math.sin(t * 7.3 + d), k = glow * fl * 2.8;
      this.rings.setColorAt(d, this.c.setRGB(1.0 * k, 0.62 * k, 0.22 * k));
    }
    this.rings.instanceColor.needsUpdate = true;
    const H = this.halo.geometry.attributes;
    for (let d = 0, n = 0; d < nd; d++) for (let k = 0; k < 48; k++, n++) {
      const th = k / 48 * 6.283, rr = G.rd + 0.004;
      H.position.setXYZ(n, rr * Math.sin(th), G.discs[d], rr * Math.cos(th));
      H.color.setXYZ(n, 1.0, 0.6, 0.22); H.alpha.setX(n, 0.11 * glow * (0.7 + 0.3 * Math.sin(t * 11 + k * 2.1 + d)));
      H.size.setX(n, 0.02);
    }
    this.halo.commit();
    const A = this.ions.geometry.attributes, B = this.tr.geometry.attributes, o = [0, 0, 0, 0], q = [0, 0, 0, 0];
    let n = 0, s = 0;
    for (let d = 0; d < nd; d++) for (let j = 0; j < NI; j++) {
      if (!this.ion(d, j, t, ctx, o)) continue;
      const a = 0.6 * (1 - Math.pow(o[3], 3)) * smooth(o[3] / 0.05) * p.w;
      A.position.setXYZ(n, o[0], o[1], o[2]); A.color.setXYZ(n, 1.15, 0.9, 0.6); A.alpha.setX(n, a); A.size.setX(n, 0.0024); n++;
      let px = o[0], py = o[1], pz = o[2];
      for (let k = 1; k <= 2; k++) {
        if (!this.ion(d, j, t - k * 0.035, ctx, q) || q[3] > o[3]) break;
        B.position.setXYZ(2 * s, px, py, pz); B.position.setXYZ(2 * s + 1, q[0], q[1], q[2]);
        B.color.setXYZ(2 * s, 1.0, 0.75, 0.45); B.color.setXYZ(2 * s + 1, 1.0, 0.75, 0.45);
        B.alpha.setX(2 * s, a * 0.5 * (1 - (k - 1) / 2)); B.alpha.setX(2 * s + 1, a * 0.5 * (1 - k / 2));
        px = q[0]; py = q[1]; pz = q[2]; s++;
      }
    }
    this.ions.commit(n); this.tr.commit(s);
  },
};

// outlet: cleaner, drier air leaving the top as faint rising streaks
const outlet = {
  init(ctx) { this.N = 56; this.tr = trailSegments(this.N * 4); this.tr.renderOrder = 7; ctx.scene.add(this.tr); },
  update(scene, t, dt, p) {
    const vis = p.w > 0.003; this.tr.visible = vis; if (!vis) return;
    const G = W.G, B = this.tr.geometry.attributes; let s = 0;
    for (let j = 0; j < this.N; j++) {
      const u = t * 0.32 * (0.8 + 0.4 * h01(j, 41)) + h01(j, 42), sg = u - Math.floor(u);
      const r0 = G.R * 0.9 * Math.sqrt(h01(j, 43)), th = h01(j, 44) * 6.283 + 0.5 * sg + 0.15 * Math.sin(t * 0.7 + j);
      const pos = k => { const ss = sg - k * 0.035; const y = G.L + 0.02 + ss * 1.1, r = r0 * (1 + 0.9 * ss * ss); return [r * Math.sin(th), y, r * Math.cos(th), ss]; };
      for (let k = 0; k < 4; k++) {
        const a = pos(k), b = pos(k + 1); if (b[3] < 0) break;
        const al = p.w * 0.13 * Math.sin(Math.PI * clamp(a[3], 0, 1)) * (1 - k / 4);
        B.position.setXYZ(2 * s, a[0], a[1], a[2]); B.position.setXYZ(2 * s + 1, b[0], b[1], b[2]);
        B.color.setXYZ(2 * s, 0.75, 0.85, 1.0); B.color.setXYZ(2 * s + 1, 0.75, 0.85, 1.0);
        B.alpha.setX(2 * s, al); B.alpha.setX(2 * s + 1, al * 0.6); s++;
      }
    }
    this.tr.commit(s);
  },
};

/* ======================================================================= the film */
const BASE = { stage: 1, shell: 1, hardware: 1 };
function shots(ctx) {
  const G = W.G, L = G.L, Ls = G.Ls, mid = L / 2;
  return [
    { id: "open", chapter: "The device", duration: 10, fade: 0,
      layers: { ...BASE, mist: 0.9, spray: 1, film: 1, corona: 1, outlet: 0.8, plane: 0 },
      params: { cut: [[0, 0.3], [10, 0.62]], solved: 1, glow: 1, sprayOn: 1, burst: 0, legend: 0, hud: [[0, 0], [2, 0], [4, 1]], bloom: 0.95, shiftX: 0.17 },
      camera: [{ at: 0, target: [0, mid + 0.15, 0], az: -40, el: 8, dist: 8.4, fov: 28, ease: true }, { at: 10, target: [0, mid, 0], az: 30, el: 10, dist: 7.0, fov: 28 }],
      titleCard: { kicker: "Simulation film", title: "Water capture tube",
        sub: "Warm, humid exhaust goes in at the bottom. A cold spray, then a corona, pull water out of it. Cleaner, drier air leaves the top.",
        meta: `${W.sc.id} · Ø ${Math.round(2 * G.R * 1000)} mm × ${L.toFixed(1)} m · best design in the campaign`, in: 0.8, out: 9.0 } },
    { id: "inlet", chapter: "Warm exhaust", duration: 11,
      layers: { ...BASE, mist: 1, plane: 0 },
      params: { cut: 0.72, solved: 0, glow: 0, sprayOn: 0, shiftX: 0.07 },
      camera: [{ at: 0.4, target: [0, 0.62, 0], az: 58, el: 4, dist: 3.3, fov: 30 }, { at: 11, target: [0, 1.05, 0], az: 82, el: 8, dist: 3.4, fov: 30 }] },
    { id: "nozzle", chapter: "Spray stage", duration: 6,
      layers: { ...BASE, mist: 1, plane: 0, spray: 1, film: 1 },
      params: { sprayOn: [[0.9, 0], [1.5, 1]], solved: 0 },
      camera: [{ at: 0.3, target: [0, Ls - 0.18, 0], az: 92, el: 13, dist: 1.45, fov: 30 }, { at: 6, target: [0, Ls - 0.42, 0], az: 104, el: 10, dist: 1.75 }] },
    { id: "heat", duration: 7,
      layers: { ...BASE, mist: 1, plane: [[0.6, 0], [3.2, 1]], spray: 1, film: 1 },
      params: { solved: [[0.5, 0], [5, 1]], dropbar: [[0.4, 0], [1.4, 1], [6.2, 1], [7, 0]] },
      camera: [{ at: 3, target: [0, 0.95, 0], az: 114, el: 6, dist: 3.3 }, { at: 7, target: [0, 0.8, 0], az: 122, el: 5, dist: 3.3 }] },
    { id: "film", duration: 6,
      layers: { ...BASE, mist: 0.8, plane: 0.7, spray: 1, film: 1 },
      camera: [{ at: 3, target: [0.03, 0.62, 0], az: 140, el: 16, dist: 1.55 }, { at: 6, target: [0.03, 0.75, 0], az: 148, el: 14, dist: 1.6 }] },
    { id: "entrain", duration: 6,
      layers: { ...BASE, mist: 1, plane: 0.8, spray: 1, film: 1 },
      camera: [{ at: 2.5, target: [0, Ls - 0.2, 0], az: 160, el: 6, dist: 2.4 }, { at: 6, target: [0, Ls + 0.05, 0], az: 168, el: 7, dist: 2.2 }] },
    { id: "corona", chapter: "Corona stage", duration: 6,
      layers: { ...BASE, mist: 1, plane: 0.45, spray: 1, film: 1, corona: 1 },
      params: { glow: [[0.8, 0], [2.4, 1]], cut: 0.78 },
      camera: [{ at: 2, target: [0, Ls + 0.32, 0], az: 182, el: 13, dist: 1.05 }, { at: 6, target: [0, Ls + 0.36, 0], az: 192, el: 11, dist: 0.98 }] },
    { id: "charge", duration: 6,
      layers: { ...BASE, mist: 1, plane: 0.4, spray: 1, film: 1, corona: 1 },
      camera: [{ at: 3, target: [0, Ls + 0.4, 0], az: 205, el: 6, dist: 0.95 }, { at: 6, target: [0, Ls + 0.42, 0], az: 212, el: 6, dist: 0.95 }] },
    { id: "quench", duration: 8,
      layers: { ...BASE, mist: 1, plane: 0.4, spray: 1, film: 1, corona: 1 },
      params: { burst: [[0, 0], [0.6, 1], [4.2, 1], [5.6, 0]], glow: [[1.4, 1], [2.9, 0.16], [4.6, 0.16], [6.6, 1]] },
      camera: [{ at: 3, target: [0, Ls + 0.22, 0], az: 222, el: 5, dist: 1.6 }, { at: 8, target: [0, Ls + 0.28, 0], az: 232, el: 6, dist: 1.5 }] },
    { id: "outlet", chapter: "Outlet", duration: 8,
      layers: { ...BASE, mist: 1, plane: 0.3, spray: 1, film: 1, corona: 1, outlet: 1 },
      params: { burst: 0, cut: 0.7 },
      camera: [{ at: 3, target: [0, L - 0.05, 0], az: 248, el: 22, dist: 1.9 }, { at: 8, target: [0, L + 0.05, 0], az: 262, el: 24, dist: 2.0 }] },
    { id: "overview", chapter: "What's validated", duration: 14,
      layers: { ...BASE, mist: 0.75, plane: 0.35, spray: 0.9, film: 1, corona: 1, outlet: 0.7 },
      params: { cut: 0.62, legend: [[1.5, 0], [3, 1]], shiftX: 0.0 },
      camera: [{ at: 4.5, target: [0, mid, 0], az: 290, el: 7, dist: 8.0, fov: 28 }, { at: 14, target: [0, mid, 0], az: 302, el: 7, dist: 8.0, fov: 28, ease: true }] },
  ];
}

const cap = (shot, kicker, title, line, status, inT, outT) => ({ shot, kicker, title, line, status, in: inT ?? 0.4, out: outT });
function captions(ctx) {
  const d = W.sc.d, o = W.sc.o, F = W.F;
  const cnt = s => PHYS.filter(p => p.st === s).length;
  return [
    cap("inlet", "02 · Inlet · turbulent updraft validated", "Warm, misty exhaust rises",
      `It enters at ${F.TIn.toFixed(0)} °C and saturated, carrying ${F.wIn.toFixed(0)} g of vapour per kg of air. The updraft is the solved k-ω SST field, which matches OpenFOAM within 2–6%.`, "pass"),
    cap("nozzle", "03 · Spray stage · nozzle not modeled", "A cold spray from one nozzle",
      `${d.spray_T_C.toFixed(0)} °C water at ${d.spray_flow_L_min.toFixed(1)} L/min, ${Math.round(d.spray_d_um)} µm drops. Drop size is a free input today; Tokgöz's nozzle data rules out fine drops at these flows.`, "none"),
    cap("heat", "03 · Spray stage · drop heating unvalidated", "Drops warm as vapour condenses on them",
      `Cold blue to warm: drops reach the sump near ${W.Tsump.toFixed(0)} °C while the humid layer thins above them. On El-Morsi's spray cooler we capture only 18–30% of the measured cooling.`, "fail"),
    cap("film", "03 · Spray stage · wall film unvalidated", "Drops that reach the wall drain as a film",
      "The film carries deposited spray and condensate down to the sump. Not validated: El-Morsi suggests the wall carries several times the heat we model.", "fail"),
    cap("entrain", "03 · Spray stage · flooding partial", "Fine drops ride the updraft",
      "Some drops are carried up toward the corona stage. The flooding diagnostic predicts which designs oscillate (38/92 vs 1/29); there is no measured case yet.", "part"),
    cap("corona", "04 · Corona stage · ion drift & V–I validated", "Ions stream from every disc edge",
      `${Math.abs(d.corona_voltage_kV).toFixed(0)} kV on the central rod; ions drift to the grounded wall. V–I matches Lübbert within 4.5% given the onset voltage; humidity near saturation is the weak spot.`, "pass"),
    cap("charge", "04 · Corona stage · charging partly checked", "Fog droplets charge and drift to the wall",
      "Field and diffusion charging, then electrohydrodynamic drift across the gap. Indirectly checked via Lübbert; Fu 2022 could not test it (fog load unreported).", "part"),
    cap("quench", "04 · Corona stage · quenching validated", "Quenching by mist",
      "A dense fog's space charge chokes the corona, which recovers once the fog is cleared. Validated against Lübbert 2012: quench times within 6–17% over 14–28 kV.", "pass"),
    cap("outlet", "05 · Outlet · not yet validated against experiment", "Cleaner, drier air leaves",
      `Model: ${Math.round(100 * o.vapor)}% of the incoming vapour removed, ${o.recovered.toFixed(1)} L/h recovered. Designs rank reliably; absolute outlet water is not validated yet.`, "part"),
    cap("overview", "06 · Validation coverage", "What's validated, where",
      `${cnt("pass")} validated · ${cnt("part")} partial · ${cnt("fail")} failed · ${cnt("run")} running · ${cnt("none")} not modeled. The corona physics checks out; the spray stage is the open question.`, null, 1.2),
  ];
}

function callouts(ctx) {
  const G = W.G, at = ctx.at, R = G.R, Ls = G.Ls;
  const co = (id, o) => ({ status: P[id].st, label: P[id].name, detail: ev(id), phys: id, ...o });
  // tracked particles, chosen deterministically for each window
  const fallT = [at("nozzle", 2.1), at("nozzle", 6.2)];
  const dFall = spray.find(ctx, ...fallT, (D, a, b) => D.fate === 0 && D.d > 220e-6 && a < 0.12);
  const heatT = [at("heat", 0.8), at("heat", 6.8)];
  const dHeat = spray.find(ctx, ...heatT, (D, a, b) => D.fate === 0 && D.d > 250e-6 && a < 0.2);
  const entT = [at("entrain", 0.8), at("entrain", 5.8)];
  const dEnt = spray.find(ctx, ...entT, (D, a, b) => D.entrainedAt >= 0 && a > D.entrainedAt - 0.15);
  const mInlet = (() => {   // a rising mist particle in the lower tube during the inlet chapter
    const o = {}; let best = 0, bs = Infinity; const t0 = at("inlet", 4.4), t1 = at("inlet", 10.6), az = ctx.timeline.camera(t0)[3] * Math.PI / 180;
    for (let i = 0; i < NM; i++) {
      if (!mistState(i, t0, ctx, o) || o.y < 0.15 || o.y > 0.6) continue;
      const y0 = o.y; if (!mistState(i, t1, ctx, o) || o.y < y0 || o.y > 1.45) continue;
      const da = Math.abs(((MI[i].th0 - az + Math.PI) % 6.283 + 6.283) % 6.283 - Math.PI), s = Math.abs(da + 1.25 - Math.PI / 2) + Math.abs(MI[i].tab.r - 0.12) * 3;
      if (s < bs) { bs = s; best = i; }
    }
    return best;
  })();
  const mCharge = mist.findCharged(ctx, at("charge", 2.0), { rMin: 0.05, rMax: 0.13, zMin: Ls + 0.25, zMax: Ls + 0.55 });
  const temp = spray.temp(dHeat);
  const ion = (d, r, s) => side(G.discs[d] + 0.012, r, s);
  const OV = at("overview", 3.4), ovO = at("overview", 14) + 1;
  const ov = (id, y, r, s, k, extra) => co(id, { at: side(y, r, s), t0: OV + k * 0.16, t1: ovO, style: "compact", side: s > 0 ? "right" : "left", ...extra });
  return [
    // 02 inlet
    co("gasflow", { shot: "inlet", in: 2.0, out: 10.6, at: side(0.95, 0.105, 1), dy: -30 }),
    co("dp", { shot: "inlet", in: 3.4, out: 10.6, at: side(0.03, R * 0.85, -1) }),
    co("mist", { shot: "inlet", in: 4.6, out: 10.6, at: mist.track(mInlet, false) }),
    // 03 spray
    co("nozzle", { shot: "nozzle", in: 0.9, out: 5.8, at: side(G.nozzleZ, 0.006, 1), dy: -40 }),
    co("traj", { t0: fallT[0], t1: fallT[1], at: spray.track(dFall) }),
    co("drops", { t0: heatT[0], t1: heatT[1], at: spray.track(dHeat), detail: (t, ctx) => { const T = temp(t, ctx); return T == null ? ev("drops") : `This drop: ${Math.round(W.drops[dHeat].d * 1e6)} µm, now ${T.toFixed(1)} °C. ${ev("drops")}`; }, maxLines: 3 }),
    co("film", { shot: "film", in: 0.6, out: 5.8, at: side(0.55, R * 0.99, 1) }),
    co("splash", { shot: "film", in: 1.8, out: 5.8, at: side(1.0, R * 0.99, -1) }),
    co("flood", { t0: entT[0], t1: entT[1], at: spray.track(dEnt) }),
    // 04 corona
    co("onset", { shot: "corona", in: 1.6, out: 5.8, at: side(G.discs[2], G.rd, 1), dy: -20 }),
    co("vi", { shot: "corona", in: 2.8, out: 5.8, at: ion(4, (G.rd + R) / 2, -1) }),
    co("charge", { shot: "charge", in: 0.4, out: 5.8, at: mist.track(mCharge, true) }),
    co("deposit", { shot: "charge", in: 1.8, out: 5.8, at: side(Ls + 0.62, R * 0.99, -1) }),
    co("quench", { shot: "quench", in: 2.4, out: 7.6, at: side(G.discs[1], G.rd, 1) }),
    co("coag", { shot: "quench", in: 3.4, out: 7.6, at: side(Ls + 0.06, R * 0.5, -1) }),
    // 05 outlet
    co("energy", { shot: "outlet", in: 1.6, out: 7.6, at: side(G.L + 0.04, 0.01, 1), dy: -30, maxLines: 3, detail: P.energy.ev }),
    // 06 overview: everything, colour-coded by status
    ov("coag", G.L - 0.08, 0.02, 1, 0), ov("onset", G.discs[10], G.rd, 1, 1), ov("vi", G.discs[8], (G.rd + R) / 2, 1, 2),
    ov("quench", G.discs[6], G.rd * 0.6, 1, 3), ov("charge", G.discs[4], R * 0.75, 1, 4), ov("deposit", G.discs[2], R * 0.98, 1, 5),
    ov("mist", Ls + 0.05, R * 0.4, 1, 6), ov("gasflow", 0.75, R * 0.3, 1, 7),
    ov("nozzle", G.nozzleZ, 0.01, -1, 8), ov("splash", 1.55, R * 0.98, -1, 9), ov("flood", 1.75, R * 0.55, -1, 10),
    ov("traj", 1.35, R * 0.45, -1, 11), ov("drops", 1.1, R * 0.3, -1, 12), ov("film", 0.7, R * 0.98, -1, 13),
    ov("dp", 0.12, R * 0.6, -1, 14), ov("energy", -0.11, R * 1.04, -1, 15),
  ];
}

// humidity colour bar, top-right, while the cut-away plane is showing
function overlay(g, t, ui) {
  if (ui.compact) return;
  const db = ui.ctx.param("dropbar", t, 0);
  if (db > 0.02) {           // drop temperature scale while the drops' colours carry it
    const x = ui.W - 40 - 190, y = 116, w = 190, stops = [0, 0.5, 1].map(u => { const c = tempColor(T_COLD + u * (T_WARM - T_COLD), [0, 0, 0]); return c.map(v => Math.round(255 * Math.pow(Math.min(1, v), 1 / 2.2))); });
    g.globalAlpha = db; g.font = ui.font(500, 10.5, true); g.fillStyle = ui.theme.ink3; g.textAlign = "left";
    g.fillText("DROP TEMPERATURE, °C", x, y);
    const grd = g.createLinearGradient(x, 0, x + w, 0); stops.forEach((c, i) => grd.addColorStop(i / 2, `rgb(${c.join(",")})`));
    g.fillStyle = grd; g.fillRect(x, y + 8, w, 6);
    g.font = ui.font(500, 11, true); g.fillStyle = ui.theme.ink2; g.fillText(String(T_COLD), x, y + 30); g.textAlign = "right"; g.fillText(T_WARM + "+", x + w, y + 30);
    g.globalAlpha = 1;
  }
  const a = ui.ctx.layer("plane", t); if (a < 0.05) return;
  const x = ui.W - 40 - 190, y = 70, w = 190, F = W.F;
  g.globalAlpha = a;
  g.font = ui.font(500, 10.5, true); g.fillStyle = ui.theme.ink3; g.textAlign = "left";
  g.fillText("HUMIDITY ON THE CUT PLANE, G/KG", x, y);
  const grd = g.createLinearGradient(x, 0, x + w, 0);
  VAPOR.forEach((c, i) => grd.addColorStop(i / (VAPOR.length - 1), `rgb(${c.join(",")})`));
  g.fillStyle = grd; g.fillRect(x, y + 8, w, 6);
  g.font = ui.font(500, 11, true); g.fillStyle = ui.theme.ink2;
  g.fillText(F.wOut.toFixed(1), x, y + 30); g.textAlign = "right"; g.fillText(F.wIn.toFixed(1), x + w, y + 30);
  g.globalAlpha = 1;
}

export default {
  title: "Water capture tube",
  fileName: "watercap-film",
  fade: 1.4,
  posterTime: 82,
  defaults: { cut: 0.6, solved: 1, glow: 1, sprayOn: 1, burst: 0, legend: 0, hud: 1, dropbar: 0 },
  post: { bloom: 0.95, threshold: 0.9, radius: 0.85, exposure: 1.05, vignette: 0.4, grain: 0.9 },
  status: Object.fromEntries(Object.entries(ST).map(([k, s]) => [k, { color: s.hex, label: s.t }])),
  legend: Object.values(ST).map(s => ({ color: s.hex, label: s.t })),
  corner: () => [`Water capture tube · ${W.sc.id}`, "Air and drops at ⅕ speed · ions not to scale"],
  setup, shots, captions, callouts, overlay,
  layers: { stage, shell, plane, hardware, film: filmLayer, mist, spray, corona, outlet },
};

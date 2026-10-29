// The 3D sim's viewport: one electrode plate contoured from the solver's SDF, stacked over the tube's pitches,
// the corona's ion cloud, air tracers advected through the LES mean flow, and the drops' fates from a capture run.
// Solver axes (x, y across, z along the tube) are drawn as (x, z, -y): the tube stands up.
//   window.Sim3D = { init(el), setStack(device), setMesh(mesh), setIons(pts, mirrored), setFlow(grid),
//                    setDrops(sample | null), setLayer(name, on) }
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const COL = { plate: 0xb9c3cf, tube: 0x9fb3c8, air: 0x7fd6ff, wall: 0xff8a4d, plateHit: 0xffc247, out: 0x5b9dff };
const INFERNO = [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]];
const inferno = t => { t = Math.min(1, Math.max(0, t)) * (INFERNO.length - 1); const i = Math.min(INFERNO.length - 2, Math.floor(t)), u = t - i; return INFERNO[i].map((c, k) => (c + (INFERNO[i + 1][k] - c) * u) / 255); };

let renderer, scene, camera, controls, host, legendEl;
let dev = null, stack = new THREE.Group(), plates = new THREE.Group(), ionsObj = null, airObj = null, dropsObj = null, tube = null;
let flow = null, air = null, ionInfo = null;
// a soft round sprite for point clouds (the default square aliases into stripes on the 1 mm cell lattice)
const DOT = (() => {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.45, "rgba(255,255,255,0.55)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
})();
const layers = { electrode: true, ions: true, air: true, drops: true };
const AIR_N = 2600, SLOW = 0.06;   // tracers; the flow plays at 6% speed (2 m/s bulk: one pitch in ~0.25 s)

function init(el) {
  host = el;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, 1, 0.002, 20);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.enableZoom = false;   // the viewport fills the tab: the wheel zooms toward the target
  renderer.domElement.addEventListener("wheel", e => {
    e.preventDefault();
    const off = camera.position.clone().sub(controls.target).multiplyScalar(Math.exp(e.deltaY * 0.0015));
    camera.position.copy(controls.target).add(off.setLength(Math.min(Math.max(off.length(), 0.03), 5)));
  }, { passive: false });
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x101016, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 1.6); dl.position.set(2, 4, 3); scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0x8aa4ff, 0.6); dl2.position.set(-3, 1, -2); scene.add(dl2);
  scene.add(stack);
  stack.add(plates);
  legendEl = host.querySelector(".s3-legend");
  new ResizeObserver(resize).observe(host);
  resize();
  let vis = true;
  new IntersectionObserver(es => { vis = es[0].isIntersecting; }).observe(host);
  let last = performance.now();
  const frame = now => {
    const dt = Math.min(0.05, (now - last) / 1e3); last = now;
    if (vis && !document.hidden && host.offsetParent !== null) { stepAir(dt); controls.update(); renderer.render(scene, camera); }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); renderer.domElement.style.width = w + "px"; renderer.domElement.style.height = h + "px";
  camera.aspect = w / h; camera.updateProjectionMatrix();
}

// solver (x, y, z) -> scene
const S = (x, y, z, v = new THREE.Vector3()) => v.set(x, z, -y);

/** The device frame: tube radius, pitch, plate count; frames the camera on the stack. */
function setStack(d) {
  dev = d;
  if (tube) stack.remove(tube);
  const L = d.plates * d.pitch;
  if (ionsObj) { stack.remove(ionsObj); ionsObj = null; }
  if (dropsObj) { stack.remove(dropsObj); dropsObj = null; }
  const g = new THREE.CylinderGeometry(d.R, d.R, L + 0.06, 96, 1, true);
  g.translate(0, L / 2, 0);
  tube = new THREE.Mesh(g, new THREE.MeshPhysicalMaterial({ color: COL.tube, transparent: true, opacity: 0.08, roughness: 0.15, side: THREE.DoubleSide, depthWrite: false }));
  tube.renderOrder = 3;
  const edges = new THREE.Group();
  for (const y of [-0.03, L + 0.03]) {
    const c = new THREE.Mesh(new THREE.TorusGeometry(d.R, 0.0008, 6, 96), new THREE.MeshBasicMaterial({ color: COL.tube, transparent: true, opacity: 0.5 }));
    c.rotation.x = Math.PI / 2; c.position.y = y; edges.add(c);
  }
  tube.add(edges);
  stack.add(tube);
  // frame the stack: distance from its larger dimension
  const span = Math.max(L, 2.4 * d.R), dist = 2.7 * span;
  controls.target.set(0, L * 0.46, 0);
  camera.position.set(0.58 * dist, L * 0.46 + 0.42 * dist, 0.7 * dist);
}

/** The plate surface (plate-local: lower face at z = 0), drawn at each pitch; `mirrored` flips it (counter-flow frame). */
let meshGeom = null, mirroredFrame = false, plateZ = 0, centered = false;
function setMesh(m) {
  centered = m.kind === "disc";   // disc meshes sit on their mid-plane; plate meshes start at their lower face
  meshGeom = new THREE.BufferGeometry();
  const P = m.positions, N = m.normals, p = new Float32Array(P.length), n = new Float32Array(N.length);
  for (let i = 0; i < P.length; i += 3) { p[i] = P[i]; p[i + 1] = P[i + 2]; p[i + 2] = -P[i + 1]; n[i] = N[i]; n[i + 1] = N[i + 2]; n[i + 2] = -N[i + 1]; }
  meshGeom.setAttribute("position", new THREE.BufferAttribute(p, 3));
  meshGeom.setAttribute("normal", new THREE.BufferAttribute(n, 3));
  meshGeom.setIndex(new THREE.BufferAttribute(m.indices, 1));
  placePlates();
}
function placePlates() {
  plates.clear();
  if (!meshGeom || !dev) return;
  const mat = new THREE.MeshStandardMaterial({ color: COL.plate, metalness: 0.35, roughness: 0.38, side: THREE.DoubleSide });
  const lower = (plateZ || dev.plateZ0 + 0.5 * dev.thickness) - (centered ? 0 : 0.5 * dev.thickness);
  for (let k = 0; k < dev.plates; k++) {
    const o = new THREE.Mesh(meshGeom, mat);
    if (mirroredFrame) { o.scale.y = -1; o.position.y = k * dev.pitch + dev.pitch - lower; } else o.position.y = k * dev.pitch + lower;
    plates.add(o);
  }
  plates.visible = layers.electrode;
}

/** Ion cloud: (x, y, z, log10 n) per point within one pitch, repeated over the stack. */
function setIons(pts, mirrored) {
  if (ionsObj) { stack.remove(ionsObj); ionsObj.geometry.dispose(); }
  if (!pts || !dev) { ionsObj = null; legend(); return; }
  const n = pts.length / 4;
  let hi = -Infinity;
  for (let i = 3; i < pts.length; i += 4) hi = Math.max(hi, pts[i]);
  const lo = hi - 3;
  const pos = new Float32Array(3 * n * dev.plates), col = new Float32Array(3 * n * dev.plates);
  const flip = mirrored !== mirroredFrame;
  const v = new THREE.Vector3();
  for (let k = 0; k < dev.plates; k++) for (let i = 0; i < n; i++) {
    const z = flip ? dev.pitch - pts[4 * i + 2] : pts[4 * i + 2];
    S(pts[4 * i], pts[4 * i + 1], z + k * dev.pitch, v);
    const o = 3 * (k * n + i);
    pos[o] = v.x; pos[o + 1] = v.y; pos[o + 2] = v.z;
    const t = Math.max(0, (pts[4 * i + 3] - lo) / (hi - lo)), c = inferno(0.3 + 0.7 * t), a = 0.08 + 0.5 * t * t;
    col[o] = c[0] * a; col[o + 1] = c[1] * a; col[o + 2] = c[2] * a;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  ionsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.0022, map: DOT, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true }));
  ionsObj.visible = layers.ions;
  stack.add(ionsObj);
  ionInfo = { lo, hi };
  legend();
}

/** Air tracers through a downsampled mean flow {mx, my, mz, h, hz, origin, uvw, mirrored}. */
function setFlow(g) {
  flow = g;
  mirroredFrame = g.mirrored; plateZ = g.plateZ;
  placePlates();
  if (!airObj) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * AIR_N), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(6 * AIR_N), 3));
    airObj = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false }));
    stack.add(airObj);
    air = { x: new Float32Array(AIR_N), y: new Float32Array(AIR_N), z: new Float32Array(AIR_N), age: new Float32Array(AIR_N) };
    for (let i = 0; i < AIR_N; i++) spawnAir(i, Math.random() * dev.plates * dev.pitch);
  }
  airObj.visible = layers.air;
}
function spawnAir(i, z = 0) {
  const r = dev.R * 0.97 * Math.sqrt(Math.random()), th = 2 * Math.PI * Math.random();
  air.x[i] = r * Math.cos(th); air.y[i] = r * Math.sin(th); air.z[i] = z; air.age[i] = 0;
}
const vel = [0, 0, 0];
function sample(x, y, z) {
  const g = flow, Lz = g.mz * g.hz;
  const zz = ((z % Lz) + Lz) % Lz;
  const fx = (x - g.origin[0]) / g.h - 0.5, fy = (y - g.origin[1]) / g.h - 0.5, fz = zz / g.hz - 0.5;
  const i0 = Math.floor(fx), j0 = Math.floor(fy), k0 = Math.floor(fz), tx = fx - i0, ty = fy - j0, tz = fz - k0;
  vel[0] = vel[1] = vel[2] = 0;
  for (let c = 0; c < 8; c++) {
    const i = Math.min(g.mx - 1, Math.max(0, i0 + (c & 1))), j = Math.min(g.my - 1, Math.max(0, j0 + ((c >> 1) & 1))), k = ((k0 + ((c >> 2) & 1)) % g.mz + g.mz) % g.mz;
    const w = (c & 1 ? tx : 1 - tx) * ((c >> 1) & 1 ? ty : 1 - ty) * ((c >> 2) & 1 ? tz : 1 - tz), o = 3 * (i + g.mx * (j + g.my * k));
    vel[0] += w * g.uvw[o]; vel[1] += w * g.uvw[o + 1]; vel[2] += w * g.uvw[o + 2];
  }
  return vel;
}
function stepAir(dt) {
  if (!flow || !air || !airObj.visible) return;
  const h = dt * SLOW, L = dev.plates * dev.pitch, tail = 0.012 / 2.0;   // streak: the last 12 mm at 2 m/s
  const P = airObj.geometry.attributes.position.array, C = airObj.geometry.attributes.color.array, v = new THREE.Vector3();
  for (let i = 0; i < AIR_N; i++) {
    // midpoint step
    let u = sample(air.x[i], air.y[i], air.z[i]);
    const xm = air.x[i] + 0.5 * h * u[0], ym = air.y[i] + 0.5 * h * u[1], zm = air.z[i] + 0.5 * h * u[2];
    u = sample(xm, ym, zm);
    air.x[i] += h * u[0]; air.y[i] += h * u[1]; air.z[i] += h * u[2]; air.age[i] += dt;
    const r = Math.hypot(air.x[i], air.y[i]), sp = Math.hypot(u[0], u[1], u[2]);
    if (air.z[i] > L || air.z[i] < 0 || r > dev.R * 0.995 || (sp < 0.02 && air.age[i] > 1)) spawnAir(i);
    const o = 6 * i;
    S(air.x[i], air.y[i], air.z[i], v); P[o] = v.x; P[o + 1] = v.y; P[o + 2] = v.z;
    S(air.x[i] - tail * u[0], air.y[i] - tail * u[1], air.z[i] - tail * u[2], v); P[o + 3] = v.x; P[o + 4] = v.y; P[o + 5] = v.z;
    const fade = Math.min(1, air.age[i] / 0.4), b = 0.35 + 0.65 * Math.min(1, sp / 3);
    C[o] = 0.5 * b * fade; C[o + 1] = 0.84 * b * fade; C[o + 2] = 1.0 * b * fade; C[o + 3] = C[o + 4] = C[o + 5] = 0;
  }
  airObj.geometry.attributes.position.needsUpdate = true; airObj.geometry.attributes.color.needsUpdate = true;
}

/** Drop fates from a capture run: (x, y, z, fate) per parcel; fate 1 left the top, 2 hit the wall, 3 hit a plate. */
function setDrops(smp) {
  if (dropsObj) { stack.remove(dropsObj); dropsObj.geometry.dispose(); dropsObj = null; }
  if (smp && dev) {
    const L = dev.plates * dev.pitch, keep = [];
    for (let i = 0; i < smp.length / 4; i++) if (smp[4 * i + 3] === 1 || smp[4 * i + 2] <= L) keep.push(i);   // drawn stack only
    const n = keep.length, pos = new Float32Array(3 * n), col = new Float32Array(3 * n), v = new THREE.Vector3();
    const cols = { 1: new THREE.Color(COL.out), 2: new THREE.Color(COL.wall), 3: new THREE.Color(COL.plateHit) };
    for (let j = 0; j < n; j++) {
      const i = keep[j], f = smp[4 * i + 3], c = cols[f] || cols[1];
      const z = f === 1 ? L + 0.006 : smp[4 * i + 2];
      S(smp[4 * i], smp[4 * i + 1], z, v); pos[3 * j] = v.x; pos[3 * j + 1] = v.y; pos[3 * j + 2] = v.z;
      col[3 * j] = c.r; col[3 * j + 1] = c.g; col[3 * j + 2] = c.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    dropsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.0055, map: DOT, vertexColors: true, transparent: true, depthWrite: false, sizeAttenuation: true }));
    dropsObj.visible = layers.drops;
    stack.add(dropsObj);
  }
  legend();
}

function setLayer(k, on) {
  layers[k] = on;
  if (k === "electrode") plates.visible = on;
  if (k === "ions" && ionsObj) ionsObj.visible = on;
  if (k === "air" && airObj) airObj.visible = on;
  if (k === "drops" && dropsObj) dropsObj.visible = on;
}

function legend() {
  if (!legendEl) return;
  const rows = [];
  const e = v => { const k = Math.floor(v); return `${(10 ** (v - k)).toFixed(0)}e${k}`; };
  if (ionInfo && ionsObj) rows.push(`<div class="lr">ions <i style="background:linear-gradient(90deg,${INFERNO.slice(2).map(c => `rgb(${c})`).join(",")})"></i> ${e(ionInfo.lo)}–${e(ionInfo.hi)} /m³</div>`);
  if (dropsObj) rows.push(`<div class="lr"><b style="color:#5b9dff">●</b> out <b style="color:#ff8a4d">●</b> wall <b style="color:#ffc247">●</b> plate</div>`);
  legendEl.innerHTML = rows.join("");
  legendEl.hidden = !rows.length;
}

window.Sim3D = { init, setStack, setMesh, setIons, setFlow, setDrops, setLayer };
window.dispatchEvent(new Event("sim3d-ready"));

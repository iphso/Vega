// 3D view of the current design: the axisymmetric tube drawn as the object it implies.
// Fast model (every input change): geometry, a cut-away plane colored by the cross-section-averaged
// profiles along the tube, mist rising and thinning out where it is collected, spray falling from the
// nozzle, corona glow at the disc edges. A full simulation of the design shown adds its 2D fields to
// the plane (spray stage r-z fields; the corona unit cell's |E| tiled over the disc stack).
//   window.View3D = { update(x, r), setResolved(R | null), setField(k), setView(k) }
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const COL = { mist: 0x5b9dff, spray: 0x3ccfb4, escaped: 0xff9a4d, disc: 0x8a96a6, glow: 0xffb547, glass: 0x9fb3c8 };
const LUT = {
  viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37]],
  inferno: [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]],
  vapor: [[8, 12, 16], [14, 42, 54], [20, 80, 92], [40, 128, 132], [86, 176, 168], [150, 214, 200]],
  ice: [[8, 10, 18], [18, 34, 70], [32, 70, 140], [64, 120, 220], [91, 157, 255], [170, 205, 255], [235, 245, 255]],
};
function cmap(name, t) {
  const L = LUT[name]; t = Math.min(1, Math.max(0, isFinite(t) ? t : 0)) * (L.length - 1);
  const i = Math.min(L.length - 2, Math.floor(t)), u = t - i;
  return L[i].map((c, k) => c + (L[i + 1][k] - c) * u);
}
const cssGrad = name => "linear-gradient(90deg," + LUT[name].map((c, i) => `rgb(${c.map(Math.round)}) ${(100 * i / (LUT[name].length - 1)).toFixed(0)}%`).join(",") + ")";

// fields on the cut-away plane: fast = from the 1D profiles, full = from a full simulation
const FIELDS = {
  w: { n: "Humidity", unit: "g/kg", map: "vapor",
       fast: (P, k) => 1e3 * P.w[k], full: R => R.fields.w.map(v => 1e3 * v) },
  mist: { n: "Liquid in the air", unit: "% of inflow", map: "ice",
          fast: (P, k) => 100 * P.mist[k] / Math.max(P.mist_in, 1e-12),
          full: R => R.fields.sprayLiq ? R.fields.sprayLiq.map(v => Math.log10(Math.max(1e3 * v, 1e-3))) : R.fields.l.map(v => Math.log10(Math.max(1e3 * v, 1e-3))),
          fullUnit: "g/m³, log₁₀", fullLog: true },
  T: { n: "Air temperature", unit: "°C", map: "inferno", fast: (P, k) => P.T[k], full: R => R.fields.T },
  E: { n: "Electric field", unit: "kV/cm, log₁₀", map: "inferno", fullOnly: true },
  none: { n: "Off" },
};

let renderer, scene, camera, controls, host, labelsEl, legendEl, viewsEl, fieldSel, badgeEl;
let tubeGroup = new THREE.Group(), plane = null, planeTex = null, planeCanvas = null;
let geomKey = "", cur = null, resolved = null, field = "w", view = "whole";
let discMesh = null, glowMesh = null, glowLevel = 0;
const mist = { n: 2600, pts: null, r: null, th: null, z: null, st: null, t: null };
const spray = { n: 1200, pts: null, r: null, th: null, z: null, vr: null, vz: null, st: null, t: null };

function init(container) {
  host = container;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(30, 1, 0.005, 100);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  // the wheel scrolls the page; ctrl / cmd + wheel (or a pinch) zooms the view
  controls.enableZoom = false;
  renderer.domElement.addEventListener("wheel", e => {
    if (!(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    const off = camera.position.clone().sub(controls.target).multiplyScalar(Math.exp(e.deltaY * 0.0015));
    const len = Math.min(Math.max(off.length(), 0.05), 60);
    camera.position.copy(controls.target).add(off.setLength(len));
    renderNow();
  }, { passive: false });
  controls.addEventListener("change", () => { if (document.hidden) renderNow(); });
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x101016, 1.15));
  const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(2, 5, 3); scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0x8aa4ff, 0.5); dl2.position.set(-3, 2, -2); scene.add(dl2);
  scene.add(tubeGroup);
  labelsEl = host.querySelector(".v3-labels"); legendEl = host.querySelector(".v3-legend");
  viewsEl = host.querySelector(".v3-views"); fieldSel = host.querySelector(".v3-field"); badgeEl = host.querySelector(".v3-badge");
  for (const [k, f] of Object.entries(FIELDS)) {
    const o = document.createElement("option"); o.value = k; o.textContent = f.n; fieldSel.appendChild(o);
  }
  fieldSel.value = field;
  fieldSel.onchange = () => setField(fieldSel.value);
  viewsEl.querySelectorAll("button").forEach(b => b.onclick = () => setView(b.dataset.view, true));
  new ResizeObserver(resize).observe(host);
  resize();
  let vis = true;
  new IntersectionObserver(es => { vis = es[0].isIntersecting; }).observe(host);
  let last = performance.now();
  const frame = now => {
    const dt = Math.min(0.05, (now - last) / 1e3); last = now;
    if (vis && !document.hidden) { step(dt, now / 1e3); controls.update(); renderer.render(scene, camera); placeLabels(); }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// a frame now, outside the animation loop (input changes, view changes, hidden tabs)
function renderNow() { if (!renderer) return; camera.lookAt(controls.target); renderer.render(scene, camera); placeLabels(); }

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false); renderer.domElement.style.width = w + "px"; renderer.domElement.style.height = h + "px";
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  renderNow();
}

/* ---------------- geometry ---------------- */
function geometry(x) {
  const R = x.tube_diameter_m / 2, Ls = x.spray_section_length_m, Lc = x.corona_section_length_m, L = Ls + Lc;
  return { R, Ls, Lc, L, rd: x.disc_diameter_frac * R, s: x.disc_spacing_m, t: Math.max((x.disc_thickness_mm || 0.3) * 1e-3, 2e-4),
           nd: Math.max(1, Math.floor(Lc / x.disc_spacing_m)), zn: x.nozzle_height_frac * Ls, spray: x.spray_flow_L_min > 0 };
}

function ring(R, y, color, opacity) {
  const pts = [];
  for (let i = 0; i <= 128; i++) { const a = i / 128 * Math.PI * 2; pts.push(new THREE.Vector3(R * Math.cos(a), y, R * Math.sin(a))); }
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
}

function build(g) {
  scene.remove(tubeGroup);
  tubeGroup.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
  tubeGroup = new THREE.Group(); scene.add(tubeGroup);
  const { R, Ls, L } = g;
  // glass tube: clear face-on, bright at the silhouette (a Fresnel rim), so it reads as a solid tube
  const glass = new THREE.Mesh(new THREE.CylinderGeometry(R, R, L, 128, 1, true), new THREE.ShaderMaterial({
    uniforms: { color: { value: new THREE.Color(COL.glass) } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 color; varying vec3 vN; varying vec3 vV;
      void main() { float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 3.0); gl_FragColor = vec4(color, 0.03 + 0.55 * f); }`,
    transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  glass.position.y = L / 2; glass.renderOrder = 3; tubeGroup.add(glass);
  for (const [y, op] of [[0, 0.5], [L, 0.5]]) tubeGroup.add(ring(R, y, COL.glass, op));
  tubeGroup.add(ring(R * 1.001, Ls, COL.glow, 0.35));
  // the cut-away plane through the axis
  planeCanvas = document.createElement("canvas"); planeCanvas.width = 96; planeCanvas.height = 1024;
  planeTex = new THREE.CanvasTexture(planeCanvas); planeTex.colorSpace = THREE.SRGBColorSpace; planeTex.minFilter = THREE.LinearFilter;
  plane = new THREE.Mesh(new THREE.PlaneGeometry(2 * R * 0.998, L),
    new THREE.MeshBasicMaterial({ map: planeTex, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
  plane.position.y = L / 2; plane.renderOrder = 1; tubeGroup.add(plane);
  // disc stack on its rod, one instanced mesh; a glow ring at every disc edge
  const metal = new THREE.MeshStandardMaterial({ color: COL.disc, metalness: 0.85, roughness: 0.28 });
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(Math.min(0.004, g.rd * 0.15), Math.min(0.004, g.rd * 0.15), g.Lc, 16), metal);
  rod.position.y = Ls + g.Lc / 2; tubeGroup.add(rod);
  discMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(g.rd, g.rd, g.t, 96), metal, g.nd);
  glowMesh = new THREE.InstancedMesh(new THREE.TorusGeometry(g.rd, Math.max(g.t * 1.6, R * 0.012), 10, 96),
    new THREE.MeshBasicMaterial({ color: COL.glow, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }), g.nd);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  for (let i = 0; i < g.nd; i++) {
    const y = Ls + (i + 0.5) * g.s;
    discMesh.setMatrixAt(i, m.makeTranslation(0, y, 0));
    glowMesh.setMatrixAt(i, m.compose(new THREE.Vector3(0, y, 0), q, new THREE.Vector3(1, 1, 1)));
  }
  tubeGroup.add(discMesh); tubeGroup.add(glowMesh);
  // nozzle
  if (g.spray) {
    const noz = new THREE.Mesh(new THREE.ConeGeometry(Math.min(0.012, R * 0.12), Math.min(0.03, R * 0.3), 24),
      new THREE.MeshStandardMaterial({ color: COL.spray, metalness: 0.3, roughness: 0.4 }));
    noz.position.y = g.zn; tubeGroup.add(noz);
  }
  makeParticles(g);
}

function pointsMaterial(size) {
  return new THREE.PointsMaterial({ size, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending });
}
function makeParticles(g) {
  const size = Math.max(g.R * 0.05, 0.004);
  for (const P of [mist, spray]) {
    P.r = new Float32Array(P.n); P.th = new Float32Array(P.n); P.z = new Float32Array(P.n); P.st = new Uint8Array(P.n); P.t = new Float32Array(P.n);
    if (P === spray) { P.vr = new Float32Array(P.n); P.vz = new Float32Array(P.n); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * P.n), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * P.n), 3));
    P.pts = new THREE.Points(geo, pointsMaterial(P === spray ? size * 1.6 : size));
    P.pts.renderOrder = 2; P.pts.frustumCulled = false;
    tubeGroup.add(P.pts);
  }
  for (let i = 0; i < mist.n; i++) spawnMist(i, g, Math.random() * g.L);
  for (let i = 0; i < spray.n; i++) spawnSpray(i, g, Math.random());
  spray.pts.visible = g.spray;
}

/* ---------------- particles: animated, shaped by the model's profiles ---------------- */
// state: 0 rising, 1 captured (moving to the wall, fading), 2 escaped (orange past the outlet), 3 falling spray
function spawnMist(i, g, z = 0) {
  mist.r[i] = g.R * 0.97 * Math.sqrt(Math.random()); mist.th[i] = Math.random() * Math.PI * 2;
  mist.z[i] = z; mist.st[i] = 0; mist.t[i] = 0;
}
function spawnSpray(i, g, age = 0) {
  const half = 30 * Math.PI / 180, a = Math.random() * half;
  spray.r[i] = 0; spray.th[i] = Math.random() * Math.PI * 2; spray.z[i] = g.zn;
  const v = Math.min(g.L * 0.25, g.Ls * 0.6);
  spray.vr[i] = v * Math.sin(a); spray.vz[i] = -v * Math.cos(a);
  spray.st[i] = Math.random() < (cur?.r.spray_top_frac ?? 0) ? 0 : 3; spray.t[i] = -age * 1.2;
}

function profileAt(arr, P, z) {
  const k = Math.min(arr.length - 1, Math.max(0, Math.floor(z / P.dz)));
  return arr[k];
}

function step(dt, time) {
  if (!cur) return;
  const g = cur.g, P = cur.r.profile, L = g.L, rise = L / 7;     // ~7 s from inlet to outlet
  const mistFrac = z => z <= 0 ? 1 : profileAt(P.mist, P, z) / Math.max(P.mist_in, 1e-12);
  // mist
  {
    const pos = mist.pts.geometry.attributes.position.array, col = mist.pts.geometry.attributes.color.array;
    const cm = new THREE.Color(COL.mist), ce = new THREE.Color(COL.escaped);
    for (let i = 0; i < mist.n; i++) {
      if (mist.st[i] === 0) {
        const z0 = mist.z[i], z1 = z0 + rise * dt * (0.75 + 0.5 * (1 - mist.r[i] / g.R));
        const f0 = mistFrac(z0), f1 = mistFrac(Math.min(z1, L));
        if (f0 > 0 && Math.random() > Math.min(1, f1 / f0)) { mist.st[i] = 1; mist.t[i] = 0; }
        mist.z[i] = z1;
        if (z1 >= L) { mist.st[i] = 2; mist.t[i] = 0; }
      } else if (mist.st[i] === 1) {
        mist.t[i] += dt; mist.r[i] += (g.R - mist.r[i]) * Math.min(1, dt * 5);
        if (mist.t[i] > 0.6) spawnMist(i, g);
      } else if (mist.st[i] === 2) {
        mist.t[i] += dt; mist.z[i] += rise * dt;
        if (mist.t[i] > 0.9) spawnMist(i, g);
      }
      const a = mist.th[i], r = mist.r[i];
      pos[3 * i] = r * Math.cos(a); pos[3 * i + 1] = mist.z[i]; pos[3 * i + 2] = r * Math.sin(a);
      const c = mist.st[i] === 2 ? ce : cm, fade = mist.st[i] === 1 ? Math.max(0, 1 - mist.t[i] / 0.6) : mist.st[i] === 2 ? Math.max(0, 1 - mist.t[i] / 0.9) : 0.55;
      col[3 * i] = c.r * fade; col[3 * i + 1] = c.g * fade; col[3 * i + 2] = c.b * fade;
    }
    mist.pts.geometry.attributes.position.needsUpdate = true; mist.pts.geometry.attributes.color.needsUpdate = true;
  }
  // spray: falls from the nozzle to the sump; the share the model carries up rides the air to the outlet
  if (g.spray) {
    const pos = spray.pts.geometry.attributes.position.array, col = spray.pts.geometry.attributes.color.array;
    const cs = new THREE.Color(COL.spray), ce = new THREE.Color(COL.escaped);
    const sprayKg = Math.max(cur.x.spray_flow_L_min / 60000 * 998, 1e-12);
    const upFrac = z => profileAt(P.spray, P, z) / sprayKg;
    for (let i = 0; i < spray.n; i++) {
      spray.t[i] += dt;
      if (spray.t[i] < 0) { pos[3 * i + 1] = -10; continue; }
      if (spray.st[i] === 3) {
        spray.vz[i] -= 9.81 * 0.03 * dt; spray.vr[i] *= Math.exp(-dt * 2.5); spray.vz[i] = Math.max(spray.vz[i], -g.Ls * 0.3);
        spray.r[i] = Math.min(g.R * 0.98, spray.r[i] + spray.vr[i] * dt); spray.z[i] += spray.vz[i] * dt;
        if (spray.z[i] <= 0 || spray.t[i] > 7) spawnSpray(i, g);
      } else if (spray.st[i] === 0) {
        const z0 = spray.z[i], z1 = z0 + rise * dt * 0.8;
        const f0 = upFrac(Math.max(z0, g.zn)), f1 = upFrac(Math.min(z1, L));
        spray.r[i] = Math.min(g.R * 0.95, spray.r[i] + g.R * 0.4 * dt * Math.random());
        if (z0 > g.zn && f0 > 0 && Math.random() > Math.min(1, f1 / f0)) { spray.st[i] = 1; spray.t[i] = 0; }
        spray.z[i] = z1;
        if (z1 >= L) { spray.st[i] = 2; spray.t[i] = 0; }
      } else {
        spray.r[i] += (g.R - spray.r[i]) * Math.min(1, dt * 5);
        if (spray.st[i] === 2) spray.z[i] += rise * dt;
        if (spray.t[i] > 0.7) spawnSpray(i, g);
      }
      const a = spray.th[i], r = spray.r[i];
      pos[3 * i] = r * Math.cos(a); pos[3 * i + 1] = spray.z[i]; pos[3 * i + 2] = r * Math.sin(a);
      const c = spray.st[i] === 2 ? ce : cs, fade = spray.st[i] === 1 || spray.st[i] === 2 ? Math.max(0, 1 - spray.t[i] / 0.7) : 1.0;
      col[3 * i] = c.r * fade; col[3 * i + 1] = c.g * fade; col[3 * i + 2] = c.b * fade;
    }
    spray.pts.geometry.attributes.position.needsUpdate = true; spray.pts.geometry.attributes.color.needsUpdate = true;
  }
  if (glowMesh) glowMesh.material.opacity = glowLevel * (0.85 + 0.15 * Math.sin(time * 6.0));
}

/* ---------------- the cut-away plane ---------------- */
function paintPlane() {
  if (!planeCanvas || !cur) return;
  const ctx = planeCanvas.getContext("2d"), W = planeCanvas.width, H = planeCanvas.height, g = cur.g, f = FIELDS[field];
  plane.visible = field !== "none";
  if (field === "none") { legendEl.hidden = true; return; }
  const full = resolved && (field === "E" || f.full);
  badgeEl.hidden = !resolved;
  const img = ctx.createImageData(W, H);
  let vals = new Float32Array(W * H).fill(NaN), unit = f.unit, log = false;
  const zOf = row => (1 - (row + 0.5) / H) * g.L, rOf = col => Math.abs((col + 0.5) / W * 2 - 1) * g.R;
  if (field === "E") {
    if (!resolved || !resolved.corona || !resolved.corona.cell) { legendEl.hidden = false; legendEl.innerHTML = `<span>Electric field: run the full simulation to see it</span>`; ctx.clearRect(0, 0, W, H); planeTex.needsUpdate = true; return; }
    const cell = resolved.corona.cell, cg = coronaGrid(resolved);
    if (cg.nr * cg.nz === cell.Er.length) {
      for (let row = 0; row < H; row++) {
        const z = zOf(row); if (z < g.Ls) continue;
        const zz = (z - g.Ls) % g.s, j = bisect(cg.zf, zz, cg.nz);
        for (let col = 0; col < W; col++) {
          const i = bisect(cg.rf, rOf(col), cg.nr), k = i * cg.nz + j;
          vals[row * W + col] = Math.log10(Math.max(Math.hypot(cell.Er[k], cell.Ez[k]) / 1e5, 1e-3));
        }
      }
    }
    log = true;
  } else if (full) {
    const F = resolved.fields, arr = f.full(resolved), nz = F.nz, nr = F.nr, zf = zFaces(F.zc);
    for (let row = 0; row < H; row++) {
      const j = bisect(zf, zOf(row), nz);
      for (let col = 0; col < W; col++) vals[row * W + col] = arr[bisect(F.rf, rOf(col), nr) * nz + j];
    }
    if (f.fullUnit) unit = f.fullUnit;
    log = !!f.fullLog;
  } else {
    const P = cur.r.profile;
    for (let row = 0; row < H; row++) {
      const v = f.fast(P, Math.min(P.T.length - 1, Math.floor(zOf(row) / P.dz)));
      for (let col = 0; col < W; col++) vals[row * W + col] = v;
    }
  }
  const fin = Array.from(vals).filter(isFinite).sort((a, b) => a - b);
  let lo = fin.length ? fin[Math.floor(0.01 * (fin.length - 1))] : 0, hi = fin.length ? fin[Math.ceil(0.995 * (fin.length - 1))] : 1;
  if (hi - lo < 1e-9) hi = lo + 1e-6;
  for (let p = 0; p < W * H; p++) {
    const v = vals[p];
    if (!isFinite(v)) { img.data[4 * p + 3] = 0; continue; }
    const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))), c = cmap(f.map, t);
    img.data[4 * p] = c[0]; img.data[4 * p + 1] = c[1]; img.data[4 * p + 2] = c[2]; img.data[4 * p + 3] = 45 + 115 * t;
  }
  ctx.putImageData(img, 0, 0);
  planeTex.needsUpdate = true;
  const fmt = v => log ? (10 ** v).toPrecision(2) : (+v.toPrecision(3)).toString();
  legendEl.hidden = false;
  legendEl.innerHTML = `<span class="lt">${f.n}${full ? "" : " · cross-section average"}</span>` +
    `<span class="lr"><span>${fmt(lo)}</span><i style="background:${cssGrad(f.map)}"></i><span>${fmt(hi)} ${log ? unit.replace(", log₁₀", "") : unit}</span></span>`;
}
const bisect = (edges, x, n) => { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (edges[m + 1] <= x) lo = m + 1; else hi = m; } return Math.min(Math.max(lo, 0), n - 1); };
function zFaces(zc) {
  const n = zc.length, zf = new Float64Array(n + 1);
  for (let j = 1; j < n; j++) zf[j] = 0.5 * (zc[j - 1] + zc[j]);
  zf[0] = Math.max(0, 2 * zc[0] - zf[1]); zf[n] = 2 * zc[n - 1] - zf[n - 1];
  return zf;
}
// the corona unit cell's grid, rebuilt as the solver built it (ehd.graded_faces)
function gradedFaces(lo, hi, fine, hF, hC, gr) {
  const pts = fine.filter(p => lo <= p && p <= hi).sort((a, b) => a - b);
  const target = x => { if (!pts.length) return hC; let d = Infinity; for (const p of pts) d = Math.min(d, Math.abs(p - x)); return Math.min(hF + (gr - 1) * d, hC); };
  const f = [lo]; let x = lo;
  while (x < hi) { let h = target(x); h = Math.min(h, target(x + h)); x = Math.min(x + h, hi); f.push(x); }
  const n = f.length; if (n > 2 && f[n - 1] - f[n - 2] < 0.3 * (f[n - 2] - f[n - 3])) f.splice(n - 2, 1);
  return Float64Array.from(f);
}
function coronaGrid(R) {
  const d = R.inp.design, Rt = d.tube_diameter_m / 2;
  const tier = R.preset === "resolved" ? { e: 10, hc: 1e-3, g: 1.12 } : { e: 8, hc: 2e-3, g: 1.2 };
  const re = d.disc_thickness_mm * 1e-3 / 2, hf = re / tier.e, rd = d.disc_diameter_frac * Rt, s = d.disc_spacing_m;
  const rf = gradedFaces(0, Rt, [rd - re, rd, rd + re], hf, tier.hc, tier.g), zf = gradedFaces(0, s, [s / 2 - re, s / 2, s / 2 + re], hf, tier.hc, tier.g);
  return { rf, zf, nr: rf.length - 1, nz: zf.length - 1 };
}

/* ---------------- camera and labels ---------------- */
function setView(k, animate) {
  view = k;
  viewsEl.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.view === k));
  if (!cur) return;
  const g = cur.g, fov = camera.fov * Math.PI / 180;
  let target, span;
  if (k === "spray") { target = new THREE.Vector3(0, g.Ls / 2, 0); span = Math.max(g.Ls, 4 * g.R); }
  else if (k === "discs") { target = new THREE.Vector3(0, g.Ls + Math.min(3, g.nd / 2) * g.s, 0); span = Math.max(6 * g.s, 2.6 * g.R); }
  else { target = new THREE.Vector3(0, g.L / 2, 0); span = g.L * 1.08; }
  const dist = span / 2 / Math.tan(fov / 2) / Math.min(1, camera.aspect < 1 ? camera.aspect * 1.6 : 1);
  const dir = new THREE.Vector3(0.7, k === "whole" ? 0.22 : 0.38, 1).normalize();
  const to = target.clone().add(dir.multiplyScalar(dist));
  if (!animate) { camera.position.copy(to); controls.target.copy(target); renderNow(); return; }
  const fromP = camera.position.clone(), fromT = controls.target.clone(), t0 = performance.now();
  const tick = () => {
    const u = Math.min(1, (performance.now() - t0) / 650), e = u * u * (3 - 2 * u);
    camera.position.lerpVectors(fromP, to, e); controls.target.lerpVectors(fromT, target, e);
    if (u < 1) requestAnimationFrame(tick); else renderNow();
  };
  tick();
}
function placeLabels() {
  if (!cur) return;
  const g = cur.g, w = host.clientWidth, h = host.clientHeight;
  const L = [["air in", 0, g.R * 1.25], ["outlet", g.L, g.R * 1.25], ["corona discs", g.Ls + Math.min(g.Lc * 0.5, 0.3), g.R * 1.25]];
  if (g.spray) L.push(["nozzle", g.zn, g.R * 1.25]);
  const v = new THREE.Vector3();
  labelsEl.querySelectorAll("span").forEach((sp, i) => {
    const l = L[i]; if (!l) { sp.hidden = true; return; }
    v.set(l[2], l[1], 0).project(camera);
    const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h;
    sp.hidden = v.z > 1 || x < 0 || x > w || y < 0 || y > h;
    sp.textContent = l[0]; sp.style.transform = `translate(${x.toFixed(0)}px, ${y.toFixed(0)}px)`;
  });
}

/* ---------------- API ---------------- */
function update(x, r) {
  const g = geometry(x), key = JSON.stringify(g);
  cur = { x, r, g };
  if (key !== geomKey) { const first = !geomKey; geomKey = key; build(g); setView(view, !first); }
  spray.pts.visible = g.spray;
  // corona glow: brighter with more current per disc, none below onset
  const I = r.corona_current_mA, I0 = Math.max(r.corona_current_no_mist_mA, 1e-9);
  glowLevel = I > 1e-4 ? 0.25 + 0.6 * Math.min(1, Math.sqrt(I / g.nd / 0.08)) * Math.min(1, I / I0 + 0.3) : 0;
  paintPlane();
  if (!g.spray) spray.pts.visible = false;
  step(0.016, performance.now() / 1e3);
  renderNow();
}
function setResolved(R) { resolved = R; paintPlane(); renderNow(); }
function setField(k) { field = k; fieldSel.value = k; paintPlane(); renderNow(); }

const el = document.getElementById("view3d");
init(el);
window.View3D = { update, setResolved, setField, setView: k => setView(k, true),
  advance: secs => { for (let t = 0; t < secs; t += 1 / 30) step(1 / 30, t); renderNow(); },
  debug: () => ({ cam: camera.position.toArray(), target: controls.target.toArray(), kids: tubeGroup.children.length, cur: !!cur, geomKey, aspect: camera.aspect, size: renderer.getSize(new THREE.Vector2()).toArray() }) };
if (window.__wbLast) update(window.__wbLast.x, window.__wbLast.r);

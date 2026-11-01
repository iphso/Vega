// The blanket reel: a short film of the ARC-class FLiBe blanket simulator, for showing the product.
// Every picture is simulator output, baked by StoneLab packages/physics/scripts/reel-bake.ts into
// blanket_reel.json: the Resolved-tier ARC baseline (heating and tritium-production maps, the
// 2-D vessel temperature, the k-epsilon tank flow and its tritium), a Fast-tier first-wall sweep,
// and 600 analog neutron histories through the same cross sections. World units are metres,
// y up; the machine is revolved about y with a wedge cut out facing +z.
import * as THREE from "three";
import { softPoints, trailSegments, smooth, clamp } from "./cinema.js";

let D = null;              // baked data + derived tables
const CUT = 50 * Math.PI / 180;           // the wedge spans phi in (-CUT, CUT)
const PX = 0.5;                            // cut-face texture: cm per pixel
const COLORS = { fw: [150, 156, 176], vv_in: [104, 110, 122], channel: [79, 179, 217], be: [126, 196, 140], vv_out: [104, 110, 122], blanket: [37, 98, 125], tank: [84, 88, 98], shield: [196, 128, 64], coil: [110, 78, 58] };
const REG = ["plasma", "fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield", "coil", "bore"];
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const cmap = (u) => { u = clamp(u, 0, 1) * (INFERNO.length - 1); const i = Math.min(INFERNO.length - 2, Math.floor(u)), f = u - i; return INFERNO[i].map((a, k) => a + (INFERNO[i + 1][k] - a) * f); };
const f32 = (b) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
const at3 = (Rcm, Zcm, phi) => new THREE.Vector3(Rcm / 100 * Math.sin(phi), Zcm / 100, Rcm / 100 * Math.cos(phi));
const fmt = (x, n) => x.toFixed(n);

/* ---------------------------------------------------------------- geometry (a port of BlanketGeometry.region) */
function sdPoly(P, x, y) {
  const n = P.length / 2;
  let d = Infinity, inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const ax = P[2 * j], ay = P[2 * j + 1], bx = P[2 * i], by = P[2 * i + 1];
    const ex = bx - ax, ey = by - ay, wx = x - ax, wy = y - ay;
    const t = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1);
    d = Math.min(d, Math.hypot(wx - ex * t, wy - ey * t));
    if ((ay > y) !== (by > y) && x < ax + ((y - ay) * ex) / ey) inside = !inside;
  }
  return inside ? -d : d;
}
/** Nearest point on the polygon and the unit tangent there. */
function nearestOnPoly(P, x, y) {
  const n = P.length / 2; let best = Infinity, out = null;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const ax = P[2 * j], ay = P[2 * j + 1], ex = P[2 * i] - ax, ey = P[2 * i + 1] - ay, L2 = ex * ex + ey * ey;
    const t = clamp(((x - ax) * ex + (y - ay) * ey) / L2, 0, 1), px = ax + ex * t, py = ay + ey * t, d = Math.hypot(x - px, y - py);
    if (d < best) { best = d; const L = Math.sqrt(L2); out = [px, py, ex / L, ey / L]; }
  }
  return out;
}
const inBox = (b, R, Z) => R >= b.rmin && R < b.rmax && Z >= b.zmin && Z < b.zmax;
function region(g, R, Z) {
  const c = g.coil;
  if (R >= c.rmax || Z < c.zmin || Z >= c.zmax) return -1;
  if (R < c.rmin) return 10;
  if (!inBox(g.shield, R, Z)) return 9;
  if (!inBox(g.tank, R, Z)) return 8;
  if (!inBox(g.blanket, R, Z)) return 7;
  const d = sdPoly(g.poly, R, Z);
  if (d < 0) return 0;
  for (let k = 0; k < g.layerEdges.length; k++) if (d < g.layerEdges[k]) return k + 1;
  return 6;
}

/* ---------------------------------------------------------------- cut-face textures */
function bakeFaces(d) {
  const g = d.geometry, W = Math.ceil(g.coil.rmax / PX), H = Math.ceil((g.coil.zmax - g.coil.zmin) / PX);
  const ids = new Int8Array(W * H);
  for (let j = 0; j < H; j++) { const Z = g.coil.zmax - (j + 0.5) * PX; for (let i = 0; i < W; i++) ids[j * W + i] = region(g, (i + 0.5) * PX, Z); }
  const mk = (fn) => {
    const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
    const c2 = cv.getContext("2d"), img = c2.createImageData(W, H);
    for (let j = 0; j < H; j++) { const Z = g.coil.zmax - (j + 0.5) * PX; for (let i = 0; i < W; i++) {
      const k = j * W + i, rgba = fn(ids[k], (i + 0.5) * PX, Z);
      if (rgba) { img.data[4 * k] = rgba[0]; img.data[4 * k + 1] = rgba[1]; img.data[4 * k + 2] = rgba[2]; img.data[4 * k + 3] = rgba[3] ?? 255; }
    } }
    c2.putImageData(img, 0, 0);
    const tx = new THREE.CanvasTexture(cv); tx.colorSpace = THREE.SRGBColorSpace; tx.minFilter = THREE.LinearMipmapLinearFilter; tx.anisotropy = 8;
    return tx;
  };
  const sample = (grid, arr) => (R, Z) => { const i = Math.floor((R - grid.r0) / grid.h), j = Math.floor((Z - grid.z0) / grid.h); return i >= 0 && j >= 0 && i < grid.nr && j < grid.nz ? arr[j * grid.nr + i] : NaN; };
  // heating and tritium production: log scale over 3.5 decades, in the vessel layers and the tank (as the page)
  const logField = (arr) => { let mx = 0; for (const v of arr) if (v > mx) mx = v; const hi = Math.log10(mx), lo = hi - 3.5, s = sample(d.maps.grid, arr);
    return { lo: 10 ** lo, hi: mx, fn: (id, R, Z) => { if (id < 1 || id > 6) return null; const v = s(R, Z); return v >= 10 ** lo ? [...cmap((Math.log10(v) - lo) / (hi - lo)), 255] : null; } }; };
  const heat = logField(d.maps.heat), trit = logField(d.maps.trit);
  let tlo = Infinity, thi = -Infinity; for (const v of d.temp.T) if (Number.isFinite(v)) { tlo = Math.min(tlo, v); thi = Math.max(thi, v); }
  const ts = sample(d.temp.grid, d.temp.T);
  let clo = Infinity, chi = 0; for (const v of d.tank.c) if (Number.isFinite(v)) { clo = Math.min(clo, v); chi = Math.max(chi, v); }
  const cs = sample(d.tank.grid, d.tank.c);
  return {
    W, H,
    mat: mk((id) => (id < 1 || id === 10 ? null : [...COLORS[REG[id]], 255])),
    heat: mk(heat.fn), trit: mk(trit.fn),
    temp: mk((id, R, Z) => { if (id < 1 || id > 5) return null; const v = ts(R, Z); return Number.isFinite(v) ? [...cmap((v - tlo) / (thi - tlo)), 255] : null; }),
    conc: mk((id, R, Z) => { if (id !== 6) return null; const v = cs(R, Z); return Number.isFinite(v) ? [...cmap((v - clo) / (chi - clo)), 255] : null; }),
    ranges: { heat: [heat.lo, heat.hi], trit: [trit.lo, trit.hi], temp: [tlo, thi], conc: [clo, chi] },
  };
}

function bilinear(grid, arr) {   // cell-centred, NaN-aware
  return (R, Z) => {
    const fx = (R - grid.r0) / grid.h - 0.5, fz = (Z - grid.z0) / grid.h - 0.5, i = Math.floor(fx), j = Math.floor(fz), tx = fx - i, tz = fz - j;
    let a = 0, w = 0;
    for (const [di, dj, ww] of [[0, 0, (1 - tx) * (1 - tz)], [1, 0, tx * (1 - tz)], [0, 1, (1 - tx) * tz], [1, 1, tx * tz]]) {
      const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= grid.nr || jj >= grid.nz) continue;
      const v = arr[jj * grid.nr + ii]; if (Number.isFinite(v)) { a += ww * v; w += ww; }
    }
    if (w > 0) return a / w;
    // at the layer's edge the field's 5 mm mask is coarser than the geometry: take the nearest valid cell
    const ci = Math.round(fx), cj = Math.round(fz);
    for (let r = 1; r <= 3; r++) for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
      const ii = ci + di, jj = cj + dj; if (ii < 0 || jj < 0 || ii >= grid.nr || jj >= grid.nz) continue;
      const v = arr[jj * grid.nr + ii]; if (Number.isFinite(v)) return v;
    }
    return NaN;
  };
}
function bakePatch(d, Rc, Zc, size = 40, px = 0.1) {
  const g = d.geometry, n = Math.round(size / px), cv = document.createElement("canvas"); cv.width = cv.height = n;
  const c2 = cv.getContext("2d"), img = c2.createImageData(n, n), ts = bilinear(d.temp.grid, d.temp.T), [lo, hi] = D.faces.ranges.temp;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    // inside this patch (outboard, inside the tank box) every boundary is a distance from the first wall,
    // so the edges are anti-aliased from that distance rather than stepped by the texels
    const R = Rc - size / 2 + (i + 0.5) * px, Z = Zc + size / 2 - (j + 0.5) * px, k = 4 * (j * n + i);
    const dd = sdPoly(g.poly, R, Z), edge = g.layerEdges[g.layerEdges.length - 1];
    if (dd < -px) continue;
    // temperature contours run along the wall: average along the tangent at this depth (hides the 5 mm grid's saw-tooth)
    const [p0, q0, tx, ty] = nearestOnPoly(g.poly, R, Z), dc = Math.min(dd, edge - 0.25), f = dd > 1e-6 ? dc / dd : 1;
    const Rb = p0 + (R - p0) * f, Zb = q0 + (Z - q0) * f;   // pulled back inside the band near its outer edge
    let a = 0, w = 0;
    for (let q = -4; q <= 4; q++) { const x = ts(Rb + q * 0.4 * tx, Zb + q * 0.4 * ty); if (Number.isFinite(x)) { a += x; w++; } }
    const band = w ? cmap((a / w - lo) / (hi - lo)).map((x) => x * 1.1) : COLORS.fw.map((x) => x * 0.8), salt = COLORS.blanket.map((x) => x * 0.8);
    const u = clamp((dd - edge) / px + 0.5, 0, 1), c = band.map((x, m) => x + (salt[m] - x) * u);
    img.data[k] = Math.min(255, c[0]); img.data[k + 1] = Math.min(255, c[1]); img.data[k + 2] = Math.min(255, c[2]); img.data[k + 3] = Math.round(255 * clamp(dd / px + 0.5, 0, 1));
  }
  c2.putImageData(img, 0, 0);
  const tx = new THREE.CanvasTexture(cv); tx.colorSpace = THREE.SRGBColorSpace; tx.anisotropy = 8;
  return { tx, Rc, Zc, size };
}

/* ---------------------------------------------------------------- neutron paths */
const LOGE = (g) => Math.log10(Math.max(1e-3, D.groupEdgesEV[D.groupEdgesEV.length - 1 - g]));
function energyColor(g) {
  // 14 MeV white-blue -> MeV cyan -> keV amber -> thermal red
  const u = clamp((LOGE(g) + 1) / 8.2, 0, 1);
  const stops = [[1.6, 0.35, 0.2], [1.5, 0.8, 0.25], [0.35, 1.2, 1.5], [0.9, 1.3, 2.2]];
  const x = u * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  return stops[i].map((a, k) => a + (stops[i + 1][k] - a) * f);
}
function bakePaths(d) {
  const keep = [];
  for (const p of d.paths) {
    // the bake's frame is (x, y, z up); the film's is y up, with phi = atan2(x, z) as in the lathes
    const phi = Math.atan2(p.p[0], p.p[1]);
    if (Math.abs(phi) > CUT + 0.6) continue;           // the histories near the open wedge
    const pts = [], cum = [0];
    for (let k = 0; k < p.p.length; k += 3) pts.push(new THREE.Vector3(p.p[k] / 100, p.p[k + 2] / 100, p.p[k + 1] / 100));
    for (let k = 1; k < pts.length; k++) cum.push(cum[k - 1] + pts[k].distanceTo(pts[k - 1]));
    keep.push({ pts, cum, g: p.g, end: p.end });
  }
  return keep;
}

/* ---------------------------------------------------------------- tank tracers: the k-epsilon mean flow, integrated once */
function bakeTracers(d, n = 700, life = 4, steps = 80) {
  const G = d.tank.grid, ur = f32(d.tank.ur), uz = f32(d.tank.uz);
  const vel = (R, Z) => { // bilinear on cell centres, cm -> m/s
    const fx = (R - G.r0) / G.h - 0.5, fz = (Z - G.z0) / G.h - 0.5, i = Math.floor(fx), j = Math.floor(fz);
    if (i < 0 || j < 0 || i >= G.nr - 1 || j >= G.nz - 1) return null;
    const tx = fx - i, tz = fz - j; let a = 0, b = 0, w = 0;
    for (const [di, dj, ww] of [[0, 0, (1 - tx) * (1 - tz)], [1, 0, tx * (1 - tz)], [0, 1, (1 - tx) * tz], [1, 1, tx * tz]]) {
      const k = (j + dj) * G.nr + i + di; if (Number.isFinite(ur[k])) { a += ww * ur[k]; b += ww * uz[k]; w += ww; }
    }
    return w > 0.3 ? [a / w, b / w] : null;
  };
  // salt cells to seed from
  const cells = [];
  for (let j = 0; j < G.nz; j++) for (let i = 0; i < G.nr; i++) if (Number.isFinite(ur[j * G.nr + i])) cells.push([G.r0 + (i + 0.5) * G.h, G.z0 + (j + 0.5) * G.h]);
  const speeds = cells.map(([R, Z]) => { const v = vel(R, Z); return v ? Math.hypot(v[0], v[1]) : 0; }).sort((a, b) => a - b);
  const med = speeds[Math.floor(speeds.length / 2)];
  const speedUp = Math.max(1, Math.round(0.35 / Math.max(med, 1e-4)));   // shown speed ~0.35 m/s median
  const dt = life / steps, path = new Float32Array(n * (steps + 1) * 2);
  for (let q = 0; q < n; q++) {
    const c = cells[Math.floor(D.ctx.hash01(q, 11) * cells.length)];
    let R = c[0] + (D.ctx.hash01(q, 12) - 0.5) * G.h, Z = c[1] + (D.ctx.hash01(q, 13) - 0.5) * G.h;
    for (let s = 0; s <= steps; s++) {
      path[2 * (q * (steps + 1) + s)] = R; path[2 * (q * (steps + 1) + s) + 1] = Z;
      const v = vel(R, Z); if (!v) continue;
      const h = dt * speedUp * 100;   // cm per step
      const v2 = vel(R + 0.5 * h * v[0], Z + 0.5 * h * v[1]) ?? v;
      R += h * v2[0]; Z += h * v2[1];
    }
  }
  return { n, life, steps, path, speedUp, med };
}

/* ---------------------------------------------------------------- layers */
const lights = {
  init(ctx) {
    ctx.scene.add(new THREE.HemisphereLight(0xc9d6ff, 0x0a0a10, 0.5));
    const key = new THREE.DirectionalLight(0xfff1e0, 2.0); key.position.set(6, 9, 10); ctx.scene.add(key);
    const rim = new THREE.DirectionalLight(0x7fa6ff, 1.4); rim.position.set(-8, 4, -6); ctx.scene.add(rim);
  },
  update() {},
};

const machine = {
  init(ctx) {
    const g = D.geometry, grp = this.g = new THREE.Group(); ctx.scene.add(grp);
    const steel = new THREE.MeshStandardMaterial({ color: 0x5b616b, metalness: 0.8, roughness: 0.42, side: THREE.DoubleSide });
    const band = (a, b, mat) => grp.add(new THREE.Mesh(new THREE.LatheGeometry([new THREE.Vector2(a[0] / 100, a[1] / 100), new THREE.Vector2(b[0] / 100, b[1] / 100)], 160, CUT, 2 * Math.PI - 2 * CUT), mat));
    const c = g.coil;
    band([c.rmax, c.zmin], [c.rmax, c.zmax], steel); band([c.rmin, c.zmax], [c.rmax, c.zmax], steel);
    band([c.rmin, c.zmin], [c.rmax, c.zmin], steel); band([c.rmin, c.zmin], [c.rmin, c.zmax], steel);
    // the first wall all the way round (seen through the wedge), glowing with its heating when "fwglow" is up
    const ring = []; for (let k = 0; k <= g.poly.length / 2; k++) { const m = k % (g.poly.length / 2); ring.push(new THREE.Vector2(g.poly[2 * m] / 100, g.poly[2 * m + 1] / 100)); }
    const fwGeo = new THREE.LatheGeometry(ring, 200, CUT, 2 * Math.PI - 2 * CUT);
    grp.add(new THREE.Mesh(fwGeo, new THREE.MeshStandardMaterial({ color: 0x8d93a3, metalness: 0.7, roughness: 0.5, side: THREE.DoubleSide })));
    // heating on the first wall: the map's value 1 cm into the tile
    const s = D.sampleHeat, col = new Float32Array(fwGeo.attributes.position.count * 3), pos = fwGeo.attributes.position;
    let mx = 0; const vals = [];
    for (let i = 0; i < pos.count; i++) {   // 0.5 cm into the tile, outward from the plasma centre
      const R = Math.hypot(pos.getX(i), pos.getZ(i)) * 100, Z = pos.getY(i) * 100, nR = R - g.p.R0, L = Math.hypot(nR, Z) || 1;
      const v = s(R + 0.5 * nR / L, Z + 0.5 * Z / L); vals.push(v); if (v > mx) mx = v;
    }
    vals.forEach((v, i) => { const u = v > 0 ? clamp((Math.log10(v / mx) + 1.2) / 1.2, 0, 1) : 0; const c3 = cmap(u).map((x) => (x / 255) ** 2.2 * 1.1); col.set(c3, 3 * i); });
    const glowGeo = fwGeo.clone(); glowGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    this.glow = new THREE.Mesh(glowGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    grp.add(this.glow);
    // the two cut faces: materials, with the fields blended in by their layers' weights
    const F = D.faces, wM = g.coil.rmax / 100, hM = (g.coil.zmax - g.coil.zmin) / 100;
    this.faceMat = new THREE.ShaderMaterial({
      uniforms: { tMat: { value: F.mat }, tHeat: { value: F.heat }, tTrit: { value: F.trit }, tTemp: { value: F.temp }, tConc: { value: F.conc }, w: { value: new THREE.Vector4() }, uDim: { value: 1 }, uGain: { value: 1.25 }, uHole: { value: new THREE.Vector4(2, 2, 2, 2) } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D tMat, tHeat, tTrit, tTemp, tConc; uniform vec4 w, uHole; uniform float uDim, uGain; varying vec2 vUv;
        void main() { if (vUv.x > uHole.x && vUv.x < uHole.z && vUv.y > uHole.y && vUv.y < uHole.w) discard;   // under the close-up's fine patch
          vec4 m = texture2D(tMat, vUv); if (m.a < 0.5) discard; vec3 c = m.rgb * 0.8;
          vec4 h = texture2D(tHeat, vUv); c = mix(c, h.rgb * uGain, h.a * w.x);
          vec4 t = texture2D(tTrit, vUv); c = mix(c, t.rgb * uGain, t.a * w.y);
          vec4 k = texture2D(tTemp, vUv); c = mix(c, k.rgb * uGain, k.a * w.z);
          vec4 q = texture2D(tConc, vUv); c = mix(c, q.rgb * uGain, q.a * w.w);
          gl_FragColor = vec4(c * uDim, 1.0); }`,
      side: THREE.DoubleSide,
    });
    for (const phi of [CUT, -CUT]) {
      const pl = new THREE.Mesh(new THREE.PlaneGeometry(wM, hM), this.faceMat);
      pl.rotation.y = phi - Math.PI / 2;
      pl.position.set((wM / 2) * Math.sin(phi), (g.coil.zmax + g.coil.zmin) / 200, (wM / 2) * Math.cos(phi));
      grp.add(pl);
    }
    // the close-up's fine patch, 1.5 mm proud of the +CUT face on the wedge side
    const pt = D.patch, phiP = CUT - 0.15 / pt.Rc;
    this.patch = new THREE.Mesh(new THREE.PlaneGeometry(pt.size / 100, pt.size / 100), new THREE.MeshBasicMaterial({ map: pt.tx, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
    this.patch.rotation.y = phiP - Math.PI / 2;
    this.patch.position.copy(at3(pt.Rc, pt.Zc, phiP));
    grp.add(this.patch);
    this.ctx = ctx;
  },
  update(scene, t, dt, P) {
    const L = (n) => P.ctx.layer(n, t);
    this.patch.material.opacity = P.get("patch", 0); this.patch.visible = this.patch.material.opacity > 0.01;
    const g = D.geometry, pt = D.patch, hz = g.coil.zmax - g.coil.zmin, m = 0.3;   // the hole is 3 mm inside the patch
    if (this.patch.visible) this.faceMat.uniforms.uHole.value.set((pt.Rc - pt.size / 2 + m) / g.coil.rmax, (pt.Zc - pt.size / 2 + m - g.coil.zmin) / hz, (pt.Rc + pt.size / 2 - m) / g.coil.rmax, (pt.Zc + pt.size / 2 - m - g.coil.zmin) / hz);
    else this.faceMat.uniforms.uHole.value.set(2, 2, 2, 2);
    this.faceMat.uniforms.w.value.set(L("heat"), L("trit"), L("temp"), L("conc"));
    this.faceMat.uniforms.uDim.value = 1 - 0.55 * P.get("dim", 0);
    this.glow.material.opacity = P.get("fwglow", 0);
  },
};
// weight carriers for the cut-face fields (their crossfades drive the face shader)
const heat = () => {}, trit = () => {}, temp = () => {}, conc = () => {};

const plasma = {
  init(ctx) {
    const g = D.geometry, n = 5000; this.pts = softPoints(n); ctx.scene.add(this.pts);
    const P = this.pts.geometry.attributes;
    for (let i = 0; i < n; i++) {
      const rho = Math.sqrt(ctx.hash01(i, 1)) * 0.9, th = 2 * Math.PI * ctx.hash01(i, 2), ph = CUT + 0.08 + (2 * Math.PI - 2 * CUT - 0.16) * ctx.hash01(i, 3);
      const R = g.p.R0 + g.p.a * rho * Math.cos(th + Math.asin(g.p.delta) * Math.sin(th)), Z = g.p.kappa * g.p.a * rho * Math.sin(th);
      const v = at3(R, Z, ph); P.position.setXYZ(i, v.x, v.y, v.z);
      const core = 1 - rho * rho;
      P.color.setXYZ(i, 1.1 + 0.6 * core, 0.35 + 0.35 * core, 1.4 + 0.4 * core);
      P.size.setX(i, 0.3 + 0.2 * ctx.hash01(i, 4));
      this.base = this.base ?? new Float32Array(n); this.base[i] = 0.02 + 0.045 * core;
    }
    this.n = n;
  },
  update(scene, t, dt, P) {
    const A = this.pts.geometry.attributes.alpha, w = P.w * P.get("plasma", 1);
    this.pts.visible = w > 0.01;
    for (let i = 0; i < this.n; i++) A.setX(i, this.base[i] * w * (0.85 + 0.15 * Math.sin(2.1 * t + 6.28 * P.ctx.hash01(i, 5))));
    this.pts.commit();
  },
};

const neutrons = {
  init(ctx) {
    this.paths = D.paths; let nSeg = 0; for (const p of this.paths) nSeg += p.pts.length - 1;
    this.lines = trailSegments(nSeg); this.lines.material.depthTest = false; ctx.scene.add(this.lines);
    this.heads = softPoints(this.paths.length); this.heads.material.depthTest = false; ctx.scene.add(this.heads);
    this.ends = softPoints(this.paths.length); this.ends.material.depthTest = false; ctx.scene.add(this.ends);
    this.lines.renderOrder = this.heads.renderOrder = this.ends.renderOrder = 10;
    this.speed = 2.2;   // m/s on screen (a 14 MeV neutron does 5×10^7 m/s)
    this.ctx = ctx;
  },
  launch(i) { const c = this.ctx; return c.at("neutrons", 0.6) + (i / this.paths.length) * (c.at("breeding", 6) - c.at("neutrons", 0.6)); },
  update(scene, t, dt, Pm) {
    const w = Pm.w, vis = w > 0.01;
    this.lines.visible = this.heads.visible = this.ends.visible = vis;
    if (!vis) return;
    const LP = this.lines.geometry.attributes, HP = this.heads.geometry.attributes, EP = this.ends.geometry.attributes;
    let s = 0;
    this.paths.forEach((p, i) => {
      const d = (t - this.launch(i)) * this.speed, total = p.cum[p.cum.length - 1];
      for (let k = 1; k < p.pts.length; k++, s++) {
        const a = p.pts[k - 1], b = p.pts[k], c = energyColor(p.g[k - 1]);
        // fade: the segment is drawn up to the head, then fades over ~0.7 m of further travel
        const on = d > p.cum[k - 1] ? clamp(1 - (d - p.cum[k]) / 0.9, 0, 1) * 0.85 * w : 0;
        const u = clamp((d - p.cum[k - 1]) / Math.max(1e-6, p.cum[k] - p.cum[k - 1]), 0, 1);
        LP.position.setXYZ(2 * s, a.x, a.y, a.z);
        LP.position.setXYZ(2 * s + 1, a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u);
        LP.color.setXYZ(2 * s, ...c); LP.color.setXYZ(2 * s + 1, ...c);
        LP.alpha.setX(2 * s, on); LP.alpha.setX(2 * s + 1, on);
      }
      // the moving neutron
      if (d > 0 && d < total) {
        let k = 1; while (k < p.cum.length - 1 && p.cum[k] < d) k++;
        const a = p.pts[k - 1], b = p.pts[k], u = clamp((d - p.cum[k - 1]) / Math.max(1e-6, p.cum[k] - p.cum[k - 1]), 0, 1);
        HP.position.setXYZ(i, a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u);
        HP.color.setXYZ(i, ...energyColor(p.g[k - 1]).map((x) => x * 1.6)); HP.alpha.setX(i, 0.9 * w); HP.size.setX(i, 0.07);
      } else HP.alpha.setX(i, 0);
      // where it ended: bred tritium stays lit (green), absorption is a brief dim red
      const e = p.pts[p.pts.length - 1]; EP.position.setXYZ(i, e.x, e.y, e.z);
      const since = d - total;
      if (since > 0 && p.end === "bred") { EP.color.setXYZ(i, 0.35, 2.4, 0.7); EP.alpha.setX(i, w * Pm.get("bred", 1) * (0.5 + 0.5 * Math.exp(-since / 0.6))); EP.size.setX(i, 0.06 + 0.1 * Math.exp(-since / 0.3)); }
      else if (since > 0 && p.end === "absorbed") { EP.color.setXYZ(i, 1.4, 0.3, 0.2); EP.alpha.setX(i, w * 0.7 * Math.exp(-since / 0.8)); EP.size.setX(i, 0.05); }
      else EP.alpha.setX(i, 0);
    });
    this.lines.commit(); this.heads.commit(); this.ends.commit();
  },
};

const flow = {
  init(ctx) {
    this.T = D.tracers; const n = this.T.n;
    this.pts = softPoints(2 * n); ctx.scene.add(this.pts); this.pts.renderOrder = 5;
    this.ctx = ctx;
  },
  update(scene, t, dt, Pm) {
    const w = Pm.w; this.pts.visible = w > 0.01; if (!this.pts.visible) return;
    const { n, life, steps, path } = this.T, A = this.pts.geometry.attributes;
    for (let q = 0; q < n; q++) {
      const ph = Pm.ctx.hash01(q, 21) * life, tau = ((t + ph) % life + life) % life, s = Math.min(steps - 1, Math.floor((tau / life) * steps)), f = (tau / life) * steps - s;
      const o = 2 * (q * (steps + 1) + s), R = path[o] + (path[o + 2] - path[o]) * f, Z = path[o + 1] + (path[o + 3] - path[o + 1]) * f;
      const fade = Math.min(smooth(tau / 0.5), smooth((life - tau) / 0.5)) * w * 0.8;
      for (const [side, phi] of [[0, CUT - 0.004], [1, -CUT + 0.004]]) {   // on both cut faces, just proud of them
        const v = at3(R, Z, phi), i = 2 * q + side;
        A.position.setXYZ(i, v.x, v.y, v.z); A.color.setXYZ(i, 0.8, 1.5, 2.0); A.alpha.setX(i, fade * 0.7); A.size.setX(i, 0.03);
      }
    }
    this.pts.commit();
  },
};

/* ---------------------------------------------------------------- shots */
const B = () => D.baseline;
const R0 = () => D.geometry.p.R0;
// a point on a cut face (phi = +/-CUT), and the camera azimuth that looks square at that face from the wedge
const face = (Rcm, Zcm, side) => { const v = at3(Rcm, Zcm, side * CUT); return [v.x, v.y, v.z]; };
// the wedge side of the phi = +CUT face is toward decreasing phi, of the -CUT face toward increasing phi
const faceAz = (side) => -side * (90 - 50);

function shots() {
  const pk = B().peaks.inconelInner;
  return [
    { id: "open", chapter: "The blanket", duration: 10, fade: 0,
      layers: { lights: 1, machine: 1, plasma: 1 },
      params: { dim: 0, fwglow: 0, shiftX: 0.24 },
      camera: [{ at: 0, target: [0, -0.2, 0], az: -38, el: 24, dist: 25, fov: 30, ease: true }, { at: 10, target: [0, 0, 0], az: -8, el: 20, dist: 21, fov: 30 }],
      titleCard: { kicker: "StoneLab simulator", title: "FLiBe immersion blanket",
        sub: "Neutronics, heat and tritium for an ARC-class blanket, coupled, running in the browser on a GPU.",
        meta: `ARC baseline · ${B().fusionMW} MW fusion · Sorbom et al. 2015 geometry`, in: 0.8, out: 9.2 } },
    { id: "neutrons", chapter: "Neutrons", duration: 12,
      layers: { lights: 1, machine: 1, plasma: 1, neutrons: 1 },
      params: { dim: [[0, 0], [1.5, 1]], shiftX: 0 },
      camera: [{ at: 1.2, target: [0, 0, 3.6], az: 12, el: 10, dist: 9.5, fov: 30 }, { at: 12, target: [0, 0, 3.8], az: 26, el: 8, dist: 8.2, fov: 30 }] },
    { id: "heat", chapter: "Heat", duration: 9,
      layers: { lights: 1, machine: 1, plasma: 0.6, heat: 1, neutrons: 0.35 },
      params: { dim: 0 },
      camera: [{ at: 1.8, target: face(370, 0, -1), az: faceAz(-1) + 4, el: 6, dist: 17, fov: 30 }, { at: 9, target: face(380, 0, -1), az: faceAz(-1) - 6, el: 4, dist: 16, fov: 30 }] },
    { id: "breeding", chapter: "Breeding", duration: 9,
      layers: { lights: 1, machine: 1, plasma: 0.6, trit: 1, neutrons: 0.8 },
      params: { bred: 1 },
      camera: [{ at: 1.5, target: face(390, 0, -1), az: faceAz(-1) - 10, el: 5, dist: 15, fov: 30 }, { at: 9, target: face(420, 20, -1), az: faceAz(-1) - 16, el: 8, dist: 12.5, fov: 30 }] },
    { id: "vessel", chapter: "Vessel heat", duration: 10,
      layers: { lights: 1, machine: 1, temp: 1 },
      params: { stations: [[1, 0], [2.5, 1], [9, 1], [10, 0]], patch: [[0, 1], [9.6, 1], [10, 0]] },
      camera: [{ at: 0, target: face(pk.R * 100 - 4, pk.Z * 100, 1), az: faceAz(1) - 10, el: 4, dist: 0.7, fov: 30, cut: true },
               { at: 5, target: face(pk.R * 100 - 3, pk.Z * 100 + 2, 1), az: faceAz(1) - 2, el: 5, dist: 0.6, fov: 30 },
               { at: 10, target: face(pk.R * 100 - 2, pk.Z * 100 + 4, 1), az: faceAz(1) + 6, el: 6, dist: 0.52, fov: 30 }] },
    { id: "tank", chapter: "Tritium in the salt", duration: 12,
      layers: { lights: 1, machine: 1, conc: 1, flow: 1 },
      camera: [{ at: 0, target: face(395, 0, 1), az: faceAz(1) + 12, el: 6, dist: 16.5, fov: 30, cut: true }, { at: 2.4, target: face(400, 0, 1), az: faceAz(1) + 10, el: 6, dist: 16, fov: 30 }, { at: 12, target: face(430, 0, 1), az: faceAz(1) - 2, el: 4, dist: 13.5, fov: 30 }] },
    { id: "sweep", chapter: "Design sweeps", duration: 11,
      layers: { lights: 1, machine: 1, plasma: 1, trit: 0.6 },
      params: { shiftX: [[0, 0], [2, -0.2]], chart: [[1, 0], [2.2, 1], [10.2, 1], [11, 0]] },
      camera: [{ at: 2, target: [0, 0, 0], az: -20, el: 22, dist: 23, fov: 30 }, { at: 11, target: [0, 0, 0], az: 8, el: 18, dist: 22, fov: 30 }] },
    { id: "end", chapter: "The simulator", duration: 9,
      layers: { lights: 1, machine: 1, plasma: 1 },
      params: { shiftX: [[0, -0.2], [2.5, 0.3]] },
      camera: [{ at: 9, target: [0, 0, 0], az: 30, el: 16, dist: 33, fov: 30, ease: true }],
      titleCard: { kicker: "StoneLab simulator", title: "Every design, on your GPU", sub: "Fast in under a second · Resolved in seconds · OpenMC reference",
        meta: "Browser TBR within 0.55% of continuous-energy OpenMC · OpenMC checked on MIT's LIBRA breeding experiments", in: 0.8, out: 8.4 } },
  ];
}

const cap = (shot, kicker, title, line, inT = 0.6, outT) => ({ shot, kicker, title, line, status: "inc", in: inT, out: outT });
function captions() {
  const b = B(), pk = b.peaks.inconelInner;
  return [
    cap("neutrons", "02 · Neutrons", "14 MeV neutrons leave the plasma",
      "Histories sampled from the simulator's 175-group ENDF/B-VIII.0 cross sections. Blue-white is fast, amber slowed; green is a lithium atom turned to tritium."),
    cap("heat", "03 · Heat", "Where the energy lands",
      `${fmt(b.nuclearMW, 0)} MW of nuclear heat, ${fmt(b.heatingMW.fw, 0)} MW of it in the 1 cm tungsten wall. Energy multiplication ${fmt(b.M, 3)}.`),
    cap("breeding", "04 · Breeding", `Tritium breeding ratio ${fmt(b.tbr[0], 3)}`,
      `${fmt(b.tritiumMgPerS, 2)} mg of tritium a second, mostly in the first tens of centimetres of salt. 8 million histories, seconds on one GPU.`),
    cap("vessel", "05 · Vessel heat", `Inner vessel peaks at ${fmt(pk.K, 0)} K`,
      `A 2-D conduction field through tungsten, Inconel, the cooling channel and the beryllium, driven by the heating map. Hottest just below the outboard midplane.`),
    cap("tank", "06 · Tritium in the salt", `${fmt(b.inventoryMg, 0)} mg held in the tank`,
      `k-ε turbulent tank flow, the cooling channel jetting in at the bottom, tritium carried out at the top and through the Inconel wall. Flow shown at ${D.tracers.speedUp}× speed.`),
    cap("sweep", "07 · Design sweeps", "One input across its range, in seconds",
      "Ten first-wall thicknesses at the Fast tier with the same random numbers, so the curve is the design, not the noise."),
  ];
}

function callouts(ctx) {
  const g = D.geometry, b = B();
  const co = (shot, pos, label, detail, side, o = {}) => ({ shot, at: [pos.x, pos.y, pos.z], label, detail, side, status: "inc", in: 2.2, out: 9.4, ...o });
  return [
    co("open", at3(g.p.R0, 0, -0.9), "D-T plasma", `${b.fusionMW} MW fusion`, "left", { in: 3.0 }),
    co("open", at3(g.blanket.rmax - 40, 120, CUT), "FLiBe tank", "Molten salt: coolant and breeder in one", "right", { in: 3.6 }),
    co("open", at3((g.shield.rmax + g.coil.rmax) / 2, -230, CUT), "TiH₂ shield · TF coils", "Neutrons must not reach the magnets", "right", { in: 4.2 }),
    co("heat", at3(g.blanket.rmin + 6, 0, -CUT), "Inboard", "The thinnest salt: 20 cm", "left", { in: 3.0, out: 8.6, style: "compact" }),
    co("breeding", at3(g.blanket.rmax - 30, -40, -CUT), "Outboard tank", "90 cm of salt catches most of the leakage", "right", { in: 3.0, out: 8.6 }),
    co("vessel", at3(b.peaks.w.R * 100, b.peaks.w.Z * 100, CUT), "Tungsten face", `${fmt(b.peaks.w.K, 0)} K`, "left", { in: 3.0, style: "compact" }),
    co("vessel", at3(b.peaks.inconelOuter.R * 100, b.peaks.inconelOuter.Z * 100, CUT), "Outer Inconel", `${fmt(b.peaks.inconelOuter.K, 0)} K`, "left", { in: 3.6, style: "compact" }),
  ];
}

/* ---------------------------------------------------------------- 2-D overlay: colour scales, the wall profile, the sweep */
const sci = (x) => { if (x === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))), m = x / 10 ** e; return Math.abs(e) <= 3 ? String(Number(x.toPrecision(2))) : `${m.toFixed(1)}e${e}`; };
function overlay(g, t, ui) {
  if (ui.compact) return;
  const C = ui.ctx, T = ui.theme;
  // colour scale for whichever field is up
  const scales = [["heat", "NUCLEAR HEATING, W/CM³ (LOG)"], ["trit", "TRITIUM PRODUCTION, ATOMS/S/CM³ (LOG)"], ["temp", "VESSEL TEMPERATURE, K"], ["conc", "TRITIUM IN THE SALT, MG/M³"]];
  for (const [k, title] of scales) {
    const a = C.layer(k, t) * (1 - C.param("chart", t, 0)); if (a < 0.05) continue;
    const [lo, hi] = D.faces.ranges[k], x = ui.W - 40 - 220, y = 70, w = 220;
    g.globalAlpha = a; g.font = ui.font(500, 10.5, true); g.fillStyle = T.ink3; g.textAlign = "left"; g.fillText(title, x, y);
    const grd = g.createLinearGradient(x, 0, x + w, 0); INFERNO.forEach((c, i) => grd.addColorStop(i / (INFERNO.length - 1), `rgb(${c.join(",")})`));
    g.fillStyle = grd; g.fillRect(x, y + 8, w, 6);
    g.font = ui.font(500, 11, true); g.fillStyle = T.ink2; g.fillText(k === "temp" ? fmt(lo, 0) : sci(lo), x, y + 30); g.textAlign = "right"; g.fillText(k === "temp" ? fmt(hi, 0) : sci(hi), x + w, y + 30);
    g.globalAlpha = 1;
  }
  // temperatures along the first wall (the page's inset)
  const sa = C.param("stations", t, 0);
  if (sa > 0.02) {
    const S = D.temp.stations, n = S.t.length, x0 = ui.W - 40 - 380, y0 = 130, w = 380, h = 190;
    const ser = [["W face", S.wFaceK, "rgb(250,193,39)"], ["inner Inconel", S.inconelPeakK, "rgb(212,72,66)"], ["coolant", S.bulkK, "rgb(79,179,217)"]];
    let lo = Infinity, hi = -Infinity; for (const [, a] of ser) for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    lo = Math.floor(lo / 100) * 100; hi = Math.ceil(hi / 100) * 100;
    const X = (k) => x0 + 44 + (k / (n - 1)) * (w - 56), Y = (v) => y0 + 30 + (1 - (v - lo) / (hi - lo)) * (h - 64);
    g.globalAlpha = sa; g.fillStyle = T.panel; g.strokeStyle = T.rule; g.beginPath(); g.roundRect(x0, y0, w, h, 10); g.fill(); g.stroke();
    g.font = ui.font(500, 10.5, true); g.fillStyle = T.ink3; g.textAlign = "left"; g.fillText("ALONG THE FIRST WALL, K", x0 + 14, y0 + 20);
    g.font = ui.font(500, 10, true);
    for (const v of [lo, (lo + hi) / 2, hi]) { g.fillStyle = T.ink3; g.textAlign = "right"; g.fillText(String(v), x0 + 38, Y(v) + 3); g.strokeStyle = "rgba(255,255,255,0.06)"; g.beginPath(); g.moveTo(X(0), Y(v)); g.lineTo(X(n - 1), Y(v)); g.stroke(); }
    ser.forEach(([lbl, a, c], si) => {
      g.strokeStyle = c; g.lineWidth = 2; g.beginPath(); for (let k = 0; k < n; k++) (k ? g.lineTo : g.moveTo).call(g, X(k), Y(a[k])); g.stroke();
      g.fillStyle = c; g.fillRect(x0 + 44 + si * 110, y0 + h - 22, 8, 8); g.fillStyle = T.ink2; g.textAlign = "left"; g.fillText(lbl, x0 + 56 + si * 110, y0 + h - 14);
    });
    g.lineWidth = 1; g.globalAlpha = 1;
  }
  // the sweep: points appear in the order the page computes them
  const ca = C.param("chart", t, 0);
  if (ca > 0.02) {
    const S = D.sweep, t0 = C.at("sweep", 2.2), shown = clamp(Math.floor((t - t0) / 0.55) + 1, 0, S.length);
    const panel = (y0, title, val, err, n, guide) => {
      const x0 = ui.W - 40 - 520, w = 520, h = 250;
      g.fillStyle = T.panel; g.strokeStyle = T.rule; g.beginPath(); g.roundRect(x0, y0, w, h, 10); g.fill(); g.stroke();
      g.font = ui.font(500, 12); g.fillStyle = T.ink2; g.textAlign = "left"; g.fillText(title, x0 + 16, y0 + 24);
      const vs = S.map(val); let lo = Math.min(...vs.map((v, i) => v - err(i))), hi = Math.max(...vs.map((v, i) => v + err(i)));
      if (guide) { lo = Math.min(lo, guide[0]); hi = Math.max(hi, guide[0]); }
      const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
      const L = x0 + 58, Rr = x0 + w - 18, Tt = y0 + 40, Bb = y0 + h - 34;
      const X = (cm) => L + (cm / 5) * (Rr - L), Y = (v) => Tt + (1 - (v - lo) / (hi - lo)) * (Bb - Tt);
      g.font = ui.font(500, 10.5, true);
      for (let k = 0; k <= 3; k++) { const v = lo + ((hi - lo) * k) / 3; g.strokeStyle = "rgba(255,255,255,0.06)"; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(Rr, Y(v)); g.stroke(); g.fillStyle = T.ink3; g.textAlign = "right"; g.fillText(v.toFixed(n), L - 8, Y(v) + 3.5); }
      for (const cm of [0, 1, 2, 3, 4, 5]) { g.textAlign = "center"; g.fillText(String(cm), X(cm), Bb + 18); }
      g.textAlign = "right"; g.fillText("tungsten first wall, cm", Rr, Bb + 32 - 0);
      if (guide) { g.setLineDash([4, 3]); g.strokeStyle = T.ink3; g.beginPath(); g.moveTo(L, Y(guide[0])); g.lineTo(Rr, Y(guide[0])); g.stroke(); g.setLineDash([]); g.fillStyle = T.ink2; g.textAlign = "right"; g.fillText(guide[1], Rr, Y(guide[0]) - 5); }
      g.strokeStyle = "#4fb3d9"; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i < shown; i++) (i ? g.lineTo : g.moveTo).call(g, X(S[i].fw_cm), Y(vs[i])); g.stroke();
      for (let i = 0; i < shown; i++) {
        const e = err(i); if (e > 0) { g.strokeStyle = "rgba(79,179,217,0.6)"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(S[i].fw_cm), Y(vs[i] - e)); g.lineTo(X(S[i].fw_cm), Y(vs[i] + e)); g.stroke(); }
        g.fillStyle = S[i].fw_cm === 1 ? T.ink : "#4fb3d9"; g.beginPath(); g.arc(X(S[i].fw_cm), Y(vs[i]), 4.5, 0, 7); g.fill();
      }
      g.lineWidth = 1;
    };
    g.globalAlpha = ca;
    panel(80, "Tritium breeding ratio", (s) => s.tbr, (i) => S[i].tbrErr, 2, [1.1, "target 1.1"]);
    panel(350, "Peak inner vessel, K", (s) => s.vvK, () => 0, 0, null);
    g.globalAlpha = 1;
  }
}

export default {
  title: "Blanket Reel",
  fileName: "blanket-reel",
  fade: 1.3,
  posterTime: 32,
  defaults: { patch: 0, dim: 0, fwglow: 0, bred: 1, stations: 0, chart: 0, plasma: 1, legend: 0 },
  post: { bloom: 0.8, threshold: 0.9, exposure: 1.05, vignette: 0.4, grain: 0.7 },
  status: { inc: { color: "#4fb3d9", label: "" } },
  legend: null,
  corner: (t) => ["StoneLab · FLiBe blanket", "ARC baseline · Resolved tier"],
  async setup(ctx) {
    const d = await (await fetch(new URL("./blanket_reel.json", import.meta.url))).json();
    d.maps.heat = f32(d.maps.heat); d.maps.trit = f32(d.maps.trit); d.temp.T = f32(d.temp.T); d.tank.c = f32(d.tank.c);
    for (const [k, v] of Object.entries(d.temp.stations)) if (v && !Array.isArray(v) && typeof v === "object") d.temp.stations[k] = Object.values(v);   // typed arrays arrive as objects
    D = d; D.ctx = ctx;
    const G = d.maps.grid;
    D.sampleHeat = (R, Z) => { const i = Math.floor((R - G.r0) / G.h), j = Math.floor((Z - G.z0) / G.h); return i >= 0 && j >= 0 && i < G.nr && j < G.nz ? d.maps.heat[j * G.nr + i] : 0; };
    D.faces = bakeFaces(d);
    const pk = d.baseline.peaks.inconelInner; D.patch = bakePatch(d, pk.R * 100, pk.Z * 100);
    D.paths = bakePaths(d);
    D.tracers = bakeTracers(d);
    return d;
  },
  layers: { lights, machine, heat, trit, temp, conc, plasma, neutrons, flow },
  shots, captions, callouts, overlay,
};

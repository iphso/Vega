// cinema.js: a small engine for animated, captioned 3D explainers of simulations.
// A declarative script (shots, layers, callouts, captions) becomes a deterministic film:
// every frame is a pure function of the timeline time t, so ?t=12.5&paused=1 renders that exact frame.
// ES module, no build step; needs an import map that resolves "three" (vendored three.js r186).
// See README.md beside this file for the script format.
import * as THREE from "three";

/* ================================================================ utilities */
export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, u) => a + (b - a) * u;
export const smooth = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
export const smoother = x => { x = clamp(x, 0, 1); return x * x * x * (x * (x * 6 - 15) + 10); };
export const window01 = (t, t0, t1, fin = 0.5, fout = 0.5) => Math.min(smooth((t - t0) / fin), 1 - smooth((t - (t1 - fout)) / fout));
export function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function hash01(i, k = 0) {
  let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(k + 0x27d4eb2f, 0xc2b2ae35);
  x ^= x >>> 15; x = Math.imul(x, 0x2c1b3c6d); x ^= x >>> 12; x = Math.imul(x, 0x297a2d39); x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}
const fmtTime = s => { s = Math.max(0, s); const m = Math.floor(s / 60), r = Math.floor(s % 60); return `${m}:${String(r).padStart(2, "0")}`; };

/* ================================================================ shared GPU helpers for layers */
// Soft round sprites: attributes position, color (linear RGB, may exceed 1 for bloom), alpha, size (world m).
export function softPoints(n, { blending = THREE.AdditiveBlending, depthFade = [0.25, 0.6], far = 40 } = {}) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uOpacity: { value: 1 }, uScale: { value: 800 }, uNear: { value: new THREE.Vector2(...depthFade) }, uFar: { value: far }, uHard: { value: 0 } },
    vertexShader: `attribute vec3 color; attribute float alpha; attribute float size; uniform float uScale; uniform vec2 uNear; uniform float uFar;
      varying vec3 vC; varying float vA;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); float d = -mv.z;
        vC = color; vA = alpha * smoothstep(uNear.x, uNear.y, d) * (1.0 - smoothstep(uFar * 0.6, uFar, d));
        gl_PointSize = max(1.5, size * uScale / max(d, 0.01)); if (vA < 0.002) gl_PointSize = 0.0;
        gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform float uOpacity; uniform float uHard; varying vec3 vC; varying float vA;
      void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float r2 = dot(p, p); if (r2 > 1.0) discard;
        float a = mix(exp(-r2 * 3.2), 1.0 - smoothstep(0.55, 1.0, r2), uHard) * vA * uOpacity;
        gl_FragColor = vec4(vC * a, a); }`,
    transparent: true, depthWrite: false, blending: blending === THREE.AdditiveBlending ? THREE.CustomBlending : blending,
  });
  if (blending === THREE.AdditiveBlending) { mat.blendSrc = THREE.OneFactor; mat.blendDst = THREE.OneFactor; mat.blendSrcAlpha = THREE.OneFactor; mat.blendDstAlpha = THREE.OneMinusSrcAlphaFactor; }
  const pts = new THREE.Points(geo, mat);
  pts.frustumCulled = false;
  pts.userData.n = n;
  const v2 = new THREE.Vector2();
  pts.onBeforeRender = (r, s, cam) => { const rt = r.getRenderTarget(), h = rt ? rt.height : r.getDrawingBufferSize(v2).y; mat.uniforms.uScale.value = h / (2 * Math.tan(cam.fov * Math.PI / 360)); };
  pts.commit = (count = n) => { for (const k of ["position", "color", "alpha", "size"]) geo.attributes[k].needsUpdate = true; geo.setDrawRange(0, count); };
  return pts;
}
// Fading line segments (trails): 2 vertices per segment, per-vertex color and alpha, additive.
export function trailSegments(nSeg) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * nSeg), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(6 * nSeg), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(2 * nSeg), 1).setUsage(THREE.DynamicDrawUsage));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uOpacity: { value: 1 } },
    vertexShader: `attribute vec3 color; attribute float alpha; varying vec3 vC; varying float vA;
      void main() { vC = color; vA = alpha; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform float uOpacity; varying vec3 vC; varying float vA; void main() { float a = vA * uOpacity; gl_FragColor = vec4(vC * a, a); }`,
    transparent: true, depthWrite: false, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
  });
  const ls = new THREE.LineSegments(geo, mat);
  ls.frustumCulled = false;
  ls.commit = (count = nSeg) => { for (const k of ["position", "color", "alpha"]) geo.attributes[k].needsUpdate = true; geo.setDrawRange(0, 2 * count); };
  return ls;
}

/* ================================================================ post: HDR, bloom, ACES, vignette, grain */
const FS_VERT = `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
class Post {
  constructor(renderer, opts = {}) {
    this.r = renderer; this.o = { bloom: 0.9, threshold: 0.85, radius: 0.85, exposure: 1.0, vignette: 0.35, grain: 0.8, ...opts };
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2));
    this.quad = new THREE.Mesh(geo); this.quad.frustumCulled = false;
    this.scene = new THREE.Scene(); this.scene.add(this.quad);
    const mk = (frag, uniforms) => new THREE.ShaderMaterial({ vertexShader: FS_VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false });
    this.mBright = mk(`uniform sampler2D tIn; uniform float th; uniform vec2 px; varying vec2 vUv;
      void main() { vec3 c = vec3(0.0);
        c += texture2D(tIn, vUv + px * vec2(-1.0, -1.0)).rgb; c += texture2D(tIn, vUv + px * vec2(1.0, -1.0)).rgb;
        c += texture2D(tIn, vUv + px * vec2(-1.0, 1.0)).rgb; c += texture2D(tIn, vUv + px * vec2(1.0, 1.0)).rgb; c *= 0.25;
        float l = max(c.r, max(c.g, c.b)); float k = smoothstep(th, th + 0.6, l);
        gl_FragColor = vec4(min(c * k, vec3(40.0)), 1.0); }`, { tIn: { value: null }, th: { value: 1 }, px: { value: new THREE.Vector2() } });
    this.mDown = mk(`uniform sampler2D tIn; uniform vec2 px; varying vec2 vUv;
      void main() { vec3 c = texture2D(tIn, vUv + px * vec2(-1.0, -1.0)).rgb + texture2D(tIn, vUv + px * vec2(1.0, -1.0)).rgb
        + texture2D(tIn, vUv + px * vec2(-1.0, 1.0)).rgb + texture2D(tIn, vUv + px * vec2(1.0, 1.0)).rgb; gl_FragColor = vec4(c * 0.25, 1.0); }`,
      { tIn: { value: null }, px: { value: new THREE.Vector2() } });
    this.mBlur = mk(`uniform sampler2D tIn; uniform vec2 dir; varying vec2 vUv;
      void main() { vec3 c = texture2D(tIn, vUv).rgb * 0.2270270;
        c += (texture2D(tIn, vUv + dir * 1.3846153).rgb + texture2D(tIn, vUv - dir * 1.3846153).rgb) * 0.3162162;
        c += (texture2D(tIn, vUv + dir * 3.2307692).rgb + texture2D(tIn, vUv - dir * 3.2307692).rgb) * 0.0702702;
        gl_FragColor = vec4(c, 1.0); }`, { tIn: { value: null }, dir: { value: new THREE.Vector2() } });
    this.mComp = mk(`uniform sampler2D tScene; uniform sampler2D b0; uniform sampler2D b1; uniform sampler2D b2; uniform sampler2D b3; uniform sampler2D b4;
      uniform float strength; uniform float radius; uniform float exposure; uniform float vig; uniform float grain; uniform float time; uniform vec2 res; varying vec2 vUv;
      vec3 RRTAndODTFit(vec3 v) { vec3 a = v * (v + 0.0245786) - 0.000090537; vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081; return a / b; }
      vec3 aces(vec3 color) {
        const mat3 ACESInputMat = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
        const mat3 ACESOutputMat = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
        color *= 1.0 / 0.6; color = ACESInputMat * color; color = RRTAndODTFit(color); color = ACESOutputMat * color; return clamp(color, 0.0, 1.0); }
      vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
      float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
      void main() {
        vec3 c = texture2D(tScene, vUv).rgb;
        float r = radius;
        vec3 b = texture2D(b0, vUv).rgb * 1.0 + texture2D(b1, vUv).rgb * mix(1.0, 0.8, r) + texture2D(b2, vUv).rgb * mix(1.0, 0.6, r)
               + texture2D(b3, vUv).rgb * mix(1.0, 0.45, r) + texture2D(b4, vUv).rgb * mix(1.0, 0.35, r);
        c = (c + b * strength) * exposure;
        c = aces(c);
        vec2 q = (vUv - 0.5) * vec2(res.x / res.y, 1.0);
        c *= mix(1.0, smoothstep(1.25, 0.25, length(q)), vig);
        c = srgb(c);
        c += (hash(vUv * res + fract(time) * 91.7) - 0.5) * grain / 255.0;
        gl_FragColor = vec4(c, 1.0); }`,
      { tScene: { value: null }, b0: { value: null }, b1: { value: null }, b2: { value: null }, b3: { value: null }, b4: { value: null },
        strength: { value: 1 }, radius: { value: 0.8 }, exposure: { value: 1 }, vig: { value: 0.3 }, grain: { value: 1 }, time: { value: 0 }, res: { value: new THREE.Vector2(1, 1) } });
    this.w = 0; this.h = 0;
  }
  setSize(w, h) {
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.main?.dispose(); (this.mips || []).forEach(m => { m.a.dispose(); m.b.dispose(); });
    const opt = { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.main = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples: 4 });
    this.mips = [];
    let mw = w, mh = h;
    for (let i = 0; i < 5; i++) {
      mw = Math.max(1, Math.round(mw / 2)); mh = Math.max(1, Math.round(mh / 2));
      this.mips.push({ a: new THREE.WebGLRenderTarget(mw, mh, opt), b: new THREE.WebGLRenderTarget(mw, mh, opt), w: mw, h: mh });
    }
  }
  pass(mat, target) { this.quad.material = mat; this.r.setRenderTarget(target); this.r.render(this.scene, this.cam); }
  render(scene, camera, time, o = {}) {
    const O = { ...this.o, ...o }, r = this.r;
    r.setRenderTarget(this.main); r.clear(); r.render(scene, camera);
    let src = this.main.texture, sw = this.w, sh = this.h;
    this.mips.forEach((m, i) => {
      const mat = i === 0 ? this.mBright : this.mDown;
      mat.uniforms.tIn.value = src; mat.uniforms.px.value.set(0.5 / sw, 0.5 / sh);
      if (i === 0) mat.uniforms.th.value = O.threshold;
      this.pass(mat, m.a);
      this.mBlur.uniforms.tIn.value = m.a.texture; this.mBlur.uniforms.dir.value.set(1 / m.w, 0); this.pass(this.mBlur, m.b);
      this.mBlur.uniforms.tIn.value = m.b.texture; this.mBlur.uniforms.dir.value.set(0, 1 / m.h); this.pass(this.mBlur, m.a);
      src = m.a.texture; sw = m.w; sh = m.h;
    });
    const u = this.mComp.uniforms;
    u.tScene.value = this.main.texture; this.mips.forEach((m, i) => u["b" + i].value = m.a.texture);
    u.strength.value = O.bloom; u.radius.value = O.radius; u.exposure.value = O.exposure; u.vig.value = O.vignette; u.grain.value = O.grain;
    u.time.value = time; u.res.value.set(this.w, this.h);
    this.pass(this.mComp, null);
  }
}

/* ================================================================ timeline */
// A value spec is a number, or keyframes [[at, v], ...] in shot-local seconds (smoothstep between keys),
// or { keys: [[at, v], ...], ease: "linear" | "smooth" }.
function evalSpec(spec, lt) {
  if (spec == null) return undefined;
  if (typeof spec === "number" || typeof spec === "boolean") return +spec;
  const keys = Array.isArray(spec) ? spec : spec.keys, ease = Array.isArray(spec) ? smooth : spec.ease === "linear" ? x => clamp(x, 0, 1) : smooth;
  if (!keys.length) return 0;
  if (lt <= keys[0][0]) return keys[0][1];
  for (let i = 0; i < keys.length - 1; i++) {
    const [a, va] = keys[i], [b, vb] = keys[i + 1];
    if (lt < b) return lerp(va, vb, ease((lt - a) / Math.max(b - a, 1e-6)));
  }
  return keys[keys.length - 1][1];
}
const lastOf = spec => typeof spec === "number" ? spec : Array.isArray(spec) ? spec[spec.length - 1][1] : spec.keys[spec.keys.length - 1][1];

class Timeline {
  constructor(script) {
    this.fade = script.fade ?? 1.2;
    let t = 0;
    this.shots = script.shots.map((s, i) => { const o = { ...s, index: i, t0: t, t1: t + s.duration }; t = o.t1; return o; });
    this.duration = t;
    this.byId = Object.fromEntries(this.shots.filter(s => s.id).map(s => [s.id, s]));
    this.chapters = [];
    this.shots.forEach(s => { if (s.chapter) this.chapters.push({ title: s.chapter, t0: s.t0, shot: s }); });
    // params inherit from the previous shot when a shot does not set them; layers do not
    const defaults = script.defaults || {};
    let prev = { ...defaults };
    for (const s of this.shots) {
      s._p = {};
      for (const k of new Set([...Object.keys(prev), ...Object.keys(s.params || {})])) s._p[k] = s.params && k in s.params ? s.params[k] : prev[k];
      prev = {}; for (const k in s._p) prev[k] = lastOf(s._p[k]);
      s._l = s.layers || {};
    }
    this.camKeys = this.buildCamera();
  }
  at(id, dt = 0) { const s = this.byId[id]; if (!s) throw new Error("no shot " + id); return s.t0 + dt; }
  shotIndex(t) {
    const S = this.shots; if (t <= 0) return 0; if (t >= this.duration) return S.length - 1;
    let lo = 0, hi = S.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (S[m].t0 <= t) lo = m; else hi = m - 1; }
    return lo;
  }
  // crossfaded value of a param (kind "_p") or a layer weight (kind "_l") at time t
  value(kind, name, t, dflt = 0) {
    const i = this.shotIndex(t), S = this.shots, s = S[i];
    const v = sh => { const x = evalSpec(sh[kind][name], t - sh.t0); return x === undefined ? dflt : x; };
    const F = s.fade ?? this.fade, h = F / 2;
    if (i > 0 && t < s.t0 + h && !s.cut) { const u = smooth((t - (s.t0 - h)) / F); return lerp(v(S[i - 1]), v(s), u); }
    if (i < S.length - 1 && t > s.t1 - h && !S[i + 1].cut) { const n = S[i + 1], Fn = n.fade ?? this.fade; if (t > s.t1 - Fn / 2) { const u = smooth((t - (s.t1 - Fn / 2)) / Fn); return lerp(v(s), v(n), u); } }
    return v(s);
  }
  param(name, t, d = 0) { return this.value("_p", name, t, d); }
  layer(name, t) { return this.value("_l", name, t, 0); }
  buildCamera() {
    const keys = [];
    for (const s of this.shots) for (const k of s.camera || []) {
      const t = s.t0 + (k.at ?? 0), tg = k.target || [0, 0, 0];
      let az = k.az, el = k.el, dist = k.dist;
      if (k.pos) { const d = new THREE.Vector3(...k.pos).sub(new THREE.Vector3(...tg)); dist = d.length(); el = Math.asin(d.y / dist) * 180 / Math.PI; az = Math.atan2(d.x, d.z) * 180 / Math.PI; }
      keys.push({ t, v: [tg[0], tg[1], tg[2], az ?? 0, el ?? 10, dist ?? 5, k.fov ?? 30], ease: !!k.ease, cut: !!k.cut || (!!s.cut && (k.at ?? 0) === 0) });
    }
    keys.sort((a, b) => a.t - b.t);
    for (let i = 1; i < keys.length; i++) {          // unwrap azimuth
      let a = keys[i].v[3]; const p = keys[i - 1].v[3];
      while (a - p > 180) a -= 360; while (a - p < -180) a += 360; keys[i].v[3] = a;
    }
    keys.forEach((k, i) => {                          // time-aware Catmull-Rom tangents
      const A = keys[i - 1], B = keys[i + 1];
      const end = !A || !B || k.ease || k.cut || (B && B.cut);
      k.m = k.v.map((x, j) => end ? 0 : (B.v[j] - A.v[j]) / Math.max(B.t - A.t, 1e-6));
    });
    return keys;
  }
  camera(t) {
    const K = this.camKeys;
    if (!K.length) return [0, 1, 0, 0, 10, 5, 30];
    if (t <= K[0].t) return K[0].v.slice();
    if (t >= K[K.length - 1].t) return K[K.length - 1].v.slice();
    let i = 0; while (i < K.length - 2 && K[i + 1].t <= t) i++;
    const a = K[i], b = K[i + 1];
    if (b.cut) return a.v.slice();
    const h = b.t - a.t, s = (t - a.t) / h, s2 = s * s, s3 = s2 * s;
    const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    return a.v.map((x, j) => h00 * x + h10 * h * a.m[j] + h01 * b.v[j] + h11 * h * b.m[j]);
  }
}

/* ================================================================ overlay: captions, callouts, legend */
const DESIGN_W = 1600, DESIGN_H = 900;
function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
function hexA(hex, a) {
  const h = hex.replace("#", ""), n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function wrap(g, text, maxW, maxLines) {
  const words = String(text).split(/\s+/), lines = []; let cur = "";
  for (const w of words) {
    const t = cur ? cur + " " + w : w;
    if (g.measureText(t).width <= maxW || !cur) cur = t; else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  if (lines.length > maxLines) { const l = lines.slice(0, maxLines); let s = l[maxLines - 1]; while (s.length > 1 && g.measureText(s + "…").width > maxW) s = s.slice(0, -1); l[maxLines - 1] = s.replace(/[\s,;:.–-]+$/, "") + "…"; return l; }
  return lines;
}

/* ================================================================ the player */
const CSS = `
.cin { --c-bg:#0c0c0e; --c-panel:#16161a; --c-card:#1c1c21; --c-ink:#e8e8ec; --c-ink2:#8a8a94; --c-ink3:#55555e; --c-rule:#2a2a30; --c-acc:#5b9dff;
  --c-font:-apple-system,"Segoe UI",system-ui,sans-serif; --c-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  position:relative; color:var(--c-ink); font:400 13px/1.4 var(--c-font); outline:none; }
.cin-stage { position:relative; width:100%; aspect-ratio:16/9; background:#070708; border-radius:10px; overflow:hidden; border:1px solid var(--c-rule); cursor:pointer; }
.cin-stage canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
.cin-load { position:absolute; inset:0; display:grid; place-items:center; color:var(--c-ink3); font:500 11px/1 var(--c-mono); letter-spacing:.08em; text-transform:uppercase; pointer-events:none; }
.cin-big { position:absolute; left:50%; top:50%; width:64px; height:64px; margin:-32px 0 0 -32px; border-radius:50%; background:rgba(12,12,14,.55); border:1px solid rgba(255,255,255,.14);
  display:grid; place-items:center; opacity:0; transition:opacity .25s; pointer-events:none; backdrop-filter:blur(4px); }
.cin-big svg { width:22px; height:22px; fill:var(--c-ink); margin-left:3px; }
.cin.paused .cin-big { opacity:1; }
.cin.rec .cin-stage::after { content:"REC"; position:absolute; top:12px; right:14px; font:600 10px/1 var(--c-mono); color:#ff8a7a; letter-spacing:.1em; }
.cin-bar { display:flex; align-items:center; gap:10px; margin-top:10px; }
.cin-bar button { background:var(--c-card); color:var(--c-ink); border:1px solid var(--c-rule); border-radius:7px; height:30px; min-width:30px; padding:0 9px; font:500 12px/1 var(--c-font); cursor:pointer; display:inline-flex; align-items:center; gap:6px; }
.cin-bar button:hover { background:#24242b; }
.cin-bar button[aria-pressed="true"] { color:var(--c-acc); border-color:color-mix(in srgb, var(--c-acc) 45%, var(--c-rule)); }
.cin-bar svg { width:13px; height:13px; fill:currentColor; }
.cin-time { font:500 11px/1 var(--c-mono); color:var(--c-ink2); min-width:76px; text-align:center; font-variant-numeric:tabular-nums; }
.cin-scrub { position:relative; flex:1; height:30px; cursor:pointer; touch-action:none; }
.cin-scrub .tr { position:absolute; left:0; right:0; top:13px; height:4px; border-radius:2px; background:var(--c-rule); }
.cin-scrub .fl { position:absolute; left:0; top:13px; height:4px; border-radius:2px; background:var(--c-acc); }
.cin-scrub .tk { position:absolute; top:9px; width:2px; height:12px; margin-left:-1px; background:var(--c-ink3); border-radius:1px; }
.cin-scrub .hd { position:absolute; top:8px; width:14px; height:14px; margin-left:-7px; border-radius:50%; background:var(--c-ink); box-shadow:0 0 0 3px rgba(91,157,255,.25); }
.cin-scrub:focus-visible { outline:1px solid var(--c-acc); outline-offset:2px; border-radius:4px; }
.cin-chaps { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.cin-chaps button { background:none; border:1px solid var(--c-rule); color:var(--c-ink2); border-radius:14px; padding:5px 11px; font:500 11.5px/1 var(--c-font); cursor:pointer; }
.cin-chaps button:hover { color:var(--c-ink); }
.cin-chaps button.on { color:var(--c-ink); border-color:var(--c-ink3); background:var(--c-card); }
.cin-chaps button b { font:500 10px/1 var(--c-mono); color:var(--c-ink3); margin-right:6px; }
.cin-list { display:none; margin:10px 0 0; padding:0; list-style:none; gap:6px; }
.cin.compact .cin-list { display:grid; }
.cin-list .cap { font-size:13px; color:var(--c-ink2); } .cin-list .cap b { display:block; color:var(--c-ink); font-size:14px; font-weight:600; }
.cin-list li.co { display:flex; gap:8px; align-items:flex-start; background:var(--c-card); border-radius:6px; padding:7px 9px; font-size:12.5px; border-left:3px solid var(--c-ink3); }
.cin-list li.co i { font:600 10px/16px var(--c-mono); font-style:normal; min-width:16px; height:16px; border-radius:8px; text-align:center; color:#0c0c0e; }
.cin-list li.co small { display:block; color:var(--c-ink2); font-size:11.5px; }
.cin-kbd { margin-left:auto; color:var(--c-ink3); font:500 10.5px/1 var(--c-mono); white-space:nowrap; }
@media (max-width: 720px) { .cin-kbd, .cin-bar .lbl { display:none; } .cin-bar { gap:6px; } .cin-time { min-width:64px; } }
`;
const ICON = {
  play: `<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>`,
  pause: `<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>`,
  loop: `<svg viewBox="0 0 16 16"><path d="M4 5h7V3l3 3-3 3V7H4a1 1 0 0 0-1 1v1H1V8a3 3 0 0 1 3-3zm8 6H5v2l-3-3 3-3v2h7a1 1 0 0 0 1-1V7h2v1a3 3 0 0 1-3 3z"/></svg>`,
  rec: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5"/></svg>`,
  stop: `<svg viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5"/></svg>`,
};

export class Cinema {
  constructor(host, script, opts = {}) {
    this.host = host; this.script = script;
    const q = new URLSearchParams(location.search);
    this.opts = { t: q.has("t") ? +q.get("t") : null, paused: q.get("paused") === "1", ...opts };
    this.t = 0; this.playing = false; this.loop = script.loop ?? true; this.ready = false;
    this.theme = { font: '-apple-system, "Segoe UI", system-ui, sans-serif', mono: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
      ink: "#e8e8ec", ink2: "#8a8a94", ink3: "#55555e", panel: "rgba(14,14,18,0.80)", rule: "rgba(255,255,255,0.09)", ...(script.theme || {}) };
  }

  async init() {
    if (!document.getElementById("cin-css")) { const st = document.createElement("style"); st.id = "cin-css"; st.textContent = CSS; document.head.appendChild(st); }
    const H = this.host;
    H.classList.add("cin"); H.tabIndex = 0;
    H.innerHTML = `<div class="cin-stage"><canvas class="gl"></canvas><canvas class="ov"></canvas><div class="cin-load">Loading</div><div class="cin-big">${ICON.play}</div></div>
      <div class="cin-bar"><button class="pp" title="Play / pause (space)" aria-label="Play">${ICON.play}</button><span class="cin-time">0:00 / 0:00</span>
        <div class="cin-scrub" role="slider" tabindex="0" aria-label="Timeline"><div class="tr"></div><div class="fl"></div><div class="hd"></div></div>
        <button class="lp" title="Loop" aria-pressed="true">${ICON.loop}</button>
        <button class="sv" title="Record the film to a 1080p WebM">${ICON.rec}<span class="lbl">Save video</span></button></div>
      <div class="cin-chaps"></div><ul class="cin-list" aria-live="polite"></ul>`;
    this.el = sel => H.querySelector(sel);
    this.stage = this.el(".cin-stage"); this.cgl = this.el("canvas.gl"); this.cov = this.el("canvas.ov"); this.g2 = this.cov.getContext("2d");

    // Some GPU/driver combinations refuse particular context attributes; fall back step by step.
    const tries = [{ powerPreference: "high-performance" }, {}, { powerPreference: "low-power" }, { failIfMajorPerformanceCaveat: false, depth: true, stencil: false }];
    let r = null, lastErr = null;
    for (const extra of tries) {
      try { r = new THREE.WebGLRenderer({ canvas: this.cgl, antialias: false, alpha: false, preserveDrawingBuffer: false, ...extra }); break; }
      catch (e) { lastErr = e; console.warn("[cinema] WebGL context attempt failed", extra, e.message); }
    }
    if (!r) throw lastErr;
    this.renderer = r;
    r.setPixelRatio(1); r.autoClear = true; r.setClearColor(0x070708, 1);
    r.toneMapping = THREE.NoToneMapping; r.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.post = new Post(r, this.script.post || {});
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.02, 80);
    this.ctx = { THREE, cinema: this, scene: this.scene, camera: this.camera, renderer: r, script: this.script,
      param: (n, t, d) => this.tl.param(n, t ?? this.t, d), layer: (n, t) => this.tl.layer(n, t ?? this.t), hash01, mulberry32, smooth, clamp, lerp };
    if (this.script.setup) this.ctx.data = await this.script.setup(this.ctx);
    const res = v => typeof v === "function" ? v(this.ctx) : v;
    const shots = res(this.script.shots);
    this.tl = new Timeline({ ...this.script, shots });
    this.ctx.timeline = this.tl; this.ctx.at = (id, dt) => this.tl.at(id, dt);
    // layers: { name: fn(scene, t, dt, params) } or { name: { init(ctx), update(scene, t, dt, params) } }
    this.layers = Object.entries(this.script.layers || {}).map(([name, L]) => ({ name, fn: typeof L === "function" ? L : L.update.bind(L), def: L }));
    for (const L of this.layers) if (L.def.init) await L.def.init(this.ctx);
    // callouts and captions may be functions of ctx: they can query the timeline and the layers
    this.callouts = (res(this.script.callouts) || []).map((c, i) => this.resolveTimes({ ...c, n: i + 1 }));
    this.captions = [...(res(this.script.captions) || []), ...this.tl.shots.filter(s => s.caption).map(s => ({ ...s.caption, t0: s.t0, t1: s.t1, shot: undefined }))]
      .map(c => this.resolveTimes(c)).sort((a, b) => a.t0 - b.t0);
    this.titles = this.tl.shots.filter(s => s.titleCard).map(s => ({ ...s.titleCard, t0: s.t0 + (s.titleCard.in ?? 0.6), t1: s.t0 + (s.titleCard.out ?? s.duration - 0.8) }));

    this.buildUI();
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.resize();
    this.el(".cin-load").remove();
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.seek(this.opts.t ?? (reduce ? (this.script.posterTime ?? 0) : 0));
    this.ready = true; window.__cinemaReady = true;
    if (!this.opts.paused && !reduce) {
      let started = false;
      new IntersectionObserver(es => { if (es[0].isIntersecting && !started) { started = true; this.play(); } else if (!es[0].isIntersecting && this.playing && !this.rec) this.pause(); }, { threshold: 0.35 }).observe(this.stage);
    }
    let last = performance.now();
    const frame = now => {
      const dt = Math.min(0.1, (now - last) / 1e3); last = now;
      if (this.playing) {
        let t = this.t + dt;
        if (t >= this.tl.duration) { if (this.rec) { this.stopRecording(true); t = this.tl.duration; } else if (this.loop) t = 0; else { t = this.tl.duration; this.pause(); } }
        this.t = t; this.draw(dt);
      } else if (this.dirty) this.draw(0);
      this.dirty = false;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    return this;
  }

  resolveTimes(c) {
    const o = { ...c };
    const base = c.shot ? this.tl.at(c.shot) : 0;
    o.t0 = c.t0 ?? base + (c.in ?? 0); o.t1 = c.t1 ?? (c.out != null ? base + c.out : c.shot ? this.tl.byId[c.shot].t1 : this.tl.duration);
    return o;
  }

  buildUI() {
    const H = this.host, D = this.tl.duration;
    this.el(".pp").onclick = () => this.toggle();
    this.stage.onclick = () => { this.toggle(); H.focus({ preventScroll: true }); };
    const lp = this.el(".lp"); lp.setAttribute("aria-pressed", this.loop); lp.onclick = () => { this.loop = !this.loop; lp.setAttribute("aria-pressed", this.loop); };
    this.el(".sv").onclick = () => this.rec ? this.stopRecording(false) : this.record();
    const sc = this.el(".cin-scrub");
    for (const c of this.tl.chapters) if (c.t0 > 0) { const k = document.createElement("div"); k.className = "tk"; k.style.left = (100 * c.t0 / D) + "%"; k.title = c.title; sc.appendChild(k); }
    const seekX = e => { const b = sc.getBoundingClientRect(); this.seek(clamp((e.clientX - b.left) / b.width, 0, 1) * D); };
    sc.onpointerdown = e => { sc.setPointerCapture(e.pointerId); this._wasPlaying = this.playing; this.pause(); seekX(e); sc.onpointermove = seekX; };
    sc.onpointerup = () => { sc.onpointermove = null; if (this._wasPlaying) this.play(); };
    const ch = this.el(".cin-chaps");
    this.tl.chapters.forEach((c, i) => {
      const b = document.createElement("button"); b.innerHTML = `<b>${String(i + 1).padStart(2, "0")}</b>${c.title}`;
      b.onclick = () => { this.seek(c.t0 + 0.01); if (!this.playing) this.play(); };
      ch.appendChild(b);
    });
    document.addEventListener("keydown", e => {
      if (e.target.closest && e.target.closest("input,textarea,select")) return;
      const inside = H.contains(document.activeElement), b = this.stage.getBoundingClientRect();
      const inView = b.top < innerHeight * 0.6 && b.bottom > innerHeight * 0.4;
      if (!inside && !(document.activeElement === document.body && inView)) return;
      const ci = this.chapterIndex(this.t);
      if (e.key === " " || e.key === "k") { e.preventDefault(); this.toggle(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); this.seek(Math.min(D, this.t + (e.shiftKey ? 1 : 5))); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); this.seek(Math.max(0, this.t - (e.shiftKey ? 1 : 5))); }
      else if (e.key === "ArrowDown" || e.key === "]") { e.preventDefault(); const c = this.tl.chapters[ci + 1]; if (c) this.seek(c.t0 + 0.01); }
      else if (e.key === "ArrowUp" || e.key === "[") { e.preventDefault(); const c0 = this.tl.chapters[ci]; const c = this.t - c0.t0 > 2 ? c0 : this.tl.chapters[Math.max(0, ci - 1)]; this.seek(c.t0 + 0.01); }
      else if (e.key === "l") { lp.click(); }
      else if (e.key === "Home") { this.seek(0); }
    });
  }
  chapterIndex(t) { let k = 0; this.tl.chapters.forEach((c, i) => { if (c.t0 <= t + 1e-6) k = i; }); return k; }

  resize() {
    if (this.rec) return;
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, this.script.maxPixelRatio ?? 2);
    this.setPixels(Math.max(2, Math.round(w * dpr)), Math.max(2, Math.round(h * dpr)));
    this.compact = w < (this.script.compactBelow ?? 640);
    this.host.classList.toggle("compact", this.compact);
    this.dirty = true;
  }
  setPixels(W, Hh) {
    this.renderer.setSize(W, Hh, false); this.post.setSize(W, Hh);
    this.cov.width = W; this.cov.height = Hh;
    this.camera.aspect = W / Hh; this.camera.updateProjectionMatrix();
    this.pxW = W; this.pxH = Hh;
  }

  play() { if (this.t >= this.tl.duration - 1e-3) this.t = 0; this.playing = true; this.host.classList.remove("paused"); this.el(".pp").innerHTML = ICON.pause; this.el(".pp").setAttribute("aria-label", "Pause"); }
  pause() { this.playing = false; this.host.classList.add("paused"); this.el(".pp").innerHTML = ICON.play; this.el(".pp").setAttribute("aria-label", "Play"); this.dirty = true; }
  toggle() { if (this.rec) return; this.playing ? this.pause() : this.play(); }
  seek(t) { this.t = clamp(t, 0, this.tl.duration); this.draw(0); }

  /* ------------------------------------------------ one frame */
  draw(dt) {
    const t = this.t, tl = this.tl;
    const v = tl.camera(t), cam = this.camera;
    const az = v[3] * Math.PI / 180, el = v[4] * Math.PI / 180;
    cam.position.set(v[0] + v[5] * Math.cos(el) * Math.sin(az), v[1] + v[5] * Math.sin(el), v[2] + v[5] * Math.cos(el) * Math.cos(az));
    cam.fov = v[6];
    const sx = tl.param("shiftX", t, 0);              // lens shift: + moves the subject right (fraction of width)
    if (Math.abs(sx) > 1e-4) cam.setViewOffset(this.pxW, this.pxH, -sx * this.pxW, 0, this.pxW, this.pxH); else cam.clearViewOffset();
    cam.updateProjectionMatrix(); cam.lookAt(v[0], v[1], v[2]); cam.updateMatrixWorld();
    this.ctx.t = t; this.ctx.camTarget = new THREE.Vector3(v[0], v[1], v[2]); this.ctx.pxH = this.pxH; this.ctx.pxW = this.pxW;
    const pScale = this.pxH / (2 * Math.tan(cam.fov * Math.PI / 360));
    this.ctx.pointScale = pScale;
    const shot = tl.shots[tl.shotIndex(t)];
    for (const L of this.layers) {
      const w = tl.layer(L.name, t);
      const params = { w, t, shot, ctx: this.ctx, get: (n, d = 0) => tl.param(n, t, d) };
      L.fn(this.scene, t, dt, params);
    }
    const P = this.script.post || {};
    this.post.render(this.scene, cam, t, { bloom: tl.param("bloom", t, P.bloom ?? 0.9), exposure: tl.param("exposure", t, P.exposure ?? 1) });
    this.drawOverlay(t);
    if (this.rec) this.recFrame();
    this.updateUI(t);
  }

  updateUI(t) {
    const D = this.tl.duration, f = 100 * t / D;
    this.el(".cin-scrub .fl").style.width = f + "%"; this.el(".cin-scrub .hd").style.left = f + "%";
    this.el(".cin-time").textContent = `${fmtTime(t)} / ${fmtTime(D)}`;
    this.el(".cin-scrub").setAttribute("aria-valuenow", t.toFixed(1));
    const ci = this.chapterIndex(t);
    this.el(".cin-chaps").querySelectorAll("button").forEach((b, i) => b.classList.toggle("on", i === ci));
    if (this.compact) {
      const cap = this.captionAt(t), cos = this.visibleCallouts;
      const key = (cap ? cap.title : "") + "|" + cos.map(c => c.n + ":" + this.txt(c.detail, t)).join(",");
      if (key !== this._listKey) {
        this._listKey = key;
        const st = this.script.status || {};
        this.el(".cin-list").innerHTML = (cap ? `<li class="cap"><b>${cap.title}</b>${this.txt(cap.line, t) || ""}</li>` : "") + cos.map(c => {
          const s = st[c.status] || {}, col = c.color || s.color || "#8a8a94";
          return `<li class="co" style="border-left-color:${col}"><i style="background:${col}">${c.n}</i><span>${this.txt(c.label, t)}${s.label ? ` <small style="display:inline;color:${col}">· ${s.label}</small>` : ""}${c.detail ? `<small>${this.txt(c.detail, t)}</small>` : ""}</span></li>`;
        }).join("");
      }
    }
  }
  txt(v, t) { return typeof v === "function" ? v(t, this.ctx) : v; }
  captionAt(t) { let c = null; for (const k of this.captions) if (t >= k.t0 && t < k.t1) c = k; return c; }

  /* ------------------------------------------------ overlay (2D canvas, also composited into recordings) */
  drawOverlay(t) {
    const g = this.g2, W = this.pxW, H = this.pxH, U = W / DESIGN_W, T = this.theme;
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, W, H);
    g.setTransform(U, 0, 0, U, 0, 0);                // draw in design units: 1600 × 900
    const DH = H / U, compact = this.compact && !this.rec;
    const font = (w, s, mono) => `${w} ${s}px ${mono ? T.mono : T.font}`;
    g.textBaseline = "alphabetic";
    // corner marks
    if (this.script.corner && !compact) {
      const cA = this.tl.param("hud", t, 1);
      if (cA > 0.01) {
        g.globalAlpha = cA; g.font = font(500, compact ? 20 : 12.5, true); g.fillStyle = T.ink3; g.textAlign = "left";
        const [l, r] = this.txt(this.script.corner, t);
        if (l) g.fillText(l.toUpperCase(), 40, 44);
        if (r && !compact) { g.textAlign = "right"; g.fillText(r.toUpperCase(), DESIGN_W - 40, 44); }
        g.globalAlpha = 1;
      }
    }
    // title cards
    for (const c of this.titles) {
      const a = window01(t, c.t0, c.t1, 1.1, 0.9); if (a <= 0.003) continue;
      const y0 = DH * 0.40, x0 = 96, rise = (1 - smoother(clamp((t - c.t0) / 1.4, 0, 1))) * 14;
      g.globalAlpha = a; g.textAlign = "left";
      if (c.kicker) { g.font = font(500, compact ? 22 : 14, true); g.fillStyle = c.kickerColor || "#5b9dff"; g.fillText(c.kicker.toUpperCase(), x0, y0 - 66 + rise); }
      g.font = font(600, compact ? 84 : 64); g.fillStyle = T.ink; g.fillText(c.title, x0, y0 + rise);
      if (c.sub) { g.font = font(400, compact ? 32 : 22); g.fillStyle = "rgba(232,232,236,0.72)"; wrap(g, c.sub, compact ? 900 : 640, 3).forEach((l, i) => g.fillText(l, x0, y0 + 44 + rise + i * (compact ? 42 : 31))); }
      if (c.meta) { g.font = font(400, compact ? 32 : 22); const nl = c.sub ? wrap(g, c.sub, compact ? 900 : 640, 3).length : 0;
        g.font = font(500, compact ? 20 : 12.5, true); g.fillStyle = T.ink3; g.fillText(c.meta.toUpperCase(), x0, y0 + 44 + nl * (compact ? 42 : 31) + 22 + rise); }
      g.globalAlpha = 1;
    }
    this.drawCallouts(g, t, DH, compact, font);
    // captions (cross-fade)
    let capBottom = DH;
    for (const c of this.captions) {
      const a = window01(t, c.t0, c.t1, 0.7, 0.6); if (a <= 0.003) continue;
      const x0 = 56, rise = (1 - smoother(clamp((t - c.t0) / 0.9, 0, 1))) * 10;
      const s = compact ? 1.7 : 1;
      g.globalAlpha = a; g.textAlign = "left";
      g.font = font(400, 16.5 * s); const lines = compact ? [] : wrap(g, this.txt(c.line, t) || "", 760, 2);
      let y = DH - 52 - (lines.length - 1) * 23 + rise;
      // a soft dark scrim behind the caption for legibility on bright frames
      const grd = g.createLinearGradient(0, DH - 230, 0, DH); grd.addColorStop(0, "rgba(7,7,8,0)"); grd.addColorStop(1, `rgba(7,7,8,${0.55 * a})`);
      g.globalAlpha = 1; g.fillStyle = grd; g.fillRect(0, DH - 230, DESIGN_W, 230); g.globalAlpha = a;
      if (lines.length) { g.fillStyle = "rgba(232,232,236,0.74)"; lines.forEach((l, i) => g.fillText(l, x0, y + i * 23)); y -= 34 * s; } else y += 4;
      g.font = font(600, 28 * s); g.fillStyle = T.ink; g.fillText(this.txt(c.title, t), x0, y);
      if (c.kicker && !compact) {
        g.font = font(500, 12 * s, true); const k = this.txt(c.kicker, t).toUpperCase(), ky = y - 36 * s;
        if (c.status) {
          const st = (this.script.status || {})[c.status] || {}, col = c.color || st.color || T.ink3;
          g.fillStyle = col; g.beginPath(); g.arc(x0 + 4, ky - 4 * s, 4 * s, 0, 7); g.fill();
          g.fillStyle = col; g.fillText(k, x0 + 15 * s, ky);
        } else { g.fillStyle = c.kickerColor || T.ink2; g.fillText(k, x0, ky); }
      }
      capBottom = Math.min(capBottom, y - 70);
      g.globalAlpha = 1;
    }
    // legend
    const la = this.tl.param("legend", t, 0);
    if (la > 0.01 && this.script.legend) {
      const items = this.txt(this.script.legend, t);
      g.globalAlpha = la; g.font = font(500, compact ? 20 : 13); g.textAlign = "left";
      const widths = items.map(it => g.measureText(it.label).width + (compact ? 40 : 30));
      let x = DESIGN_W - 48 - widths.reduce((a, b) => a + b, 0), y = DH - 54;
      items.forEach((it, i) => {
        g.fillStyle = it.color; roundRect(g, x, y - (compact ? 14 : 10), compact ? 16 : 11, compact ? 16 : 11, 3); g.fill();
        g.fillStyle = T.ink2; g.fillText(it.label, x + (compact ? 24 : 17), y); x += widths[i];
      });
      g.globalAlpha = 1;
    }
    if (this.script.overlay) { g.save(); this.script.overlay(g, t, { ctx: this.ctx, W: DESIGN_W, H: DH, compact, font, theme: T }); g.restore(); }
  }

  drawCallouts(g, t, DH, compact, font) {
    const T = this.theme, st = this.script.status || {}, cam = this.camera, v = new THREE.Vector3();
    const items = [];
    for (const c of this.callouts) {
      const a = window01(t, c.t0, c.t1, c.fadeIn ?? 0.7, c.fadeOut ?? 0.5);
      if (a <= 0.003) continue;
      const p = typeof c.at === "function" ? c.at(t, this.ctx) : c.at;
      if (!p) continue;
      v.set(p.x ?? p[0], p.y ?? p[1], p.z ?? p[2]).project(cam);
      if (v.z > 1 || v.z < -1) continue;
      const x = (v.x + 1) / 2 * DESIGN_W, y = (1 - v.y) / 2 * DH;
      const m = 24; const vis = smooth(Math.min(x - m, DESIGN_W - m - x, y - m, DH - m - y) / 40);
      if (vis <= 0) continue;
      const s = st[c.status] || {}, col = c.color || s.color || T.ink2;
      items.push({ c, a: a * vis, x, y, col, sLabel: c.statusLabel ?? s.label });
    }
    this.visibleCallouts = items.filter(i => i.a > 0.3).map(i => i.c);
    if (!items.length) return;
    if (compact) {   // numbered dots; the list under the player carries the text
      g.textAlign = "center";
      for (const it of items) {
        g.globalAlpha = it.a; g.fillStyle = it.col; g.beginPath(); g.arc(it.x, it.y, 40, 0, 7); g.fill();
        g.strokeStyle = "rgba(7,7,8,0.85)"; g.lineWidth = 5; g.stroke();
        g.fillStyle = "#0c0c0e"; g.font = font(700, 44, true); g.fillText(String(it.c.n), it.x, it.y + 15);
      }
      g.globalAlpha = 1; return;
    }
    // measure boxes
    const PADX = 12, PADY = 9;
    for (const it of items) {
      const c = it.c, mini = c.style === "compact";
      it.label = this.txt(c.label, t); it.detail = mini ? null : this.txt(c.detail, t);
      g.font = font(600, mini ? 14.5 : 16); const lw = g.measureText(it.label).width;
      g.font = font(500, 11, true); const sw = it.sLabel && !mini ? g.measureText(it.sLabel.toUpperCase()).width : 0;
      g.font = font(400, 13.5); it.lines = it.detail ? wrap(g, it.detail, c.maxWidth ?? 285, c.maxLines ?? 2) : [];
      const dw = Math.max(0, ...it.lines.map(l => g.measureText(l).width));
      it.w = Math.min(340, Math.max(lw + (mini ? 22 : 0), sw, dw) + 2 * PADX + 3);
      it.h = mini ? 32 : PADY * 2 + (sw ? 18 : 0) + 20 + it.lines.length * 18;
    }
    // sides and columns
    const top = 64, bottom = DH - (this.captionAt(t) ? 178 : 40);
    const groups = { left: [], right: [] };
    for (const it of items) { it.side = it.c.side || (it.x > DESIGN_W * 0.5 ? "right" : "left"); groups[it.side].push(it); }
    for (const [side, G] of Object.entries(groups)) {
      if (!G.length) continue;
      const gap = 56;
      if (side === "right") {
        const col = Math.max(...G.map(i => i.x + (i.c.offset ?? gap)));
        for (const i of G) i.bx = Math.min(DESIGN_W - 32 - i.w, i.c.column ?? col);
      } else {
        const col = Math.min(...G.map(i => i.x - (i.c.offset ?? gap)));
        for (const i of G) i.bx = Math.max(32, (i.c.column ?? col) - i.w);
      }
      // vertical: desired center at anchor (+ optional dy), then resolve overlaps weighted by appearance
      G.forEach(i => { i.by = i.y + (i.c.dy ?? 0) - i.h / 2; i.eff = i.h * smooth(i.a * 1.4); });
      G.sort((a, b) => a.by - b.by);
      for (let pass = 0; pass < 3; pass++) {
        for (let k = 0; k < G.length; k++) {
          const lo = k ? G[k - 1].by + G[k - 1].eff + 8 * smooth(G[k - 1].a * 1.4) : top;
          G[k].by = Math.max(G[k].by, lo);
        }
        for (let k = G.length - 1; k >= 0; k--) {
          const hi = k < G.length - 1 ? G[k + 1].by - 8 * smooth(G[k].a * 1.4) - G[k].eff : bottom - G[k].h;
          G[k].by = Math.min(G[k].by, hi);
        }
        G.forEach(i => { i.by = Math.max(top, i.by); });
      }
    }
    // draw
    for (const it of items) {
      const c = it.c, mini = c.style === "compact", a = it.a, right = it.side === "right";
      const lineA = smooth(a * 1.6), boxA = smooth((a - 0.3) / 0.7), slide = (1 - boxA) * 10 * (right ? -1 : 1);
      const bx = it.bx + slide, by = it.by, ex = right ? bx : bx + it.w, ey = by + Math.min(it.h / 2, 22);
      // leader: anchor → elbow → box edge, drawn progressively
      const elbowX = ex + (right ? -18 : 18);
      const pts = [[it.x, it.y], [elbowX, ey], [ex, ey]];
      const seg = [Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]), Math.abs(pts[2][0] - pts[1][0])], total = seg[0] + seg[1];
      let rem = total * lineA;
      g.globalAlpha = 0.75 * Math.min(1, a * 1.3); g.strokeStyle = it.col; g.lineWidth = 1.25;
      g.beginPath(); g.moveTo(...pts[0]);
      for (let k = 0; k < 2 && rem > 0; k++) { const u = Math.min(1, rem / Math.max(seg[k], 1e-6)); g.lineTo(lerp(pts[k][0], pts[k + 1][0], u), lerp(pts[k][1], pts[k + 1][1], u)); rem -= seg[k]; }
      g.stroke();
      // anchor
      const pulse = 0.5 + 0.5 * Math.sin(t * 3.1 + c.n);
      g.globalAlpha = Math.min(1, a * 1.5) * (0.25 + 0.2 * pulse); g.fillStyle = it.col; g.beginPath(); g.arc(it.x, it.y, 7 + 2.5 * pulse, 0, 7); g.fill();
      g.globalAlpha = Math.min(1, a * 1.5); g.beginPath(); g.arc(it.x, it.y, 3.4, 0, 7); g.fill();
      g.strokeStyle = "rgba(7,7,8,0.9)"; g.lineWidth = 1.2; g.stroke();
      if (boxA <= 0.01) continue;
      // box
      g.globalAlpha = boxA;
      roundRect(g, bx, by, it.w, it.h, 6); g.fillStyle = T.panel; g.fill(); g.strokeStyle = T.rule; g.lineWidth = 1; g.stroke();
      g.save(); roundRect(g, bx, by, it.w, it.h, 6); g.clip(); g.fillStyle = it.col; g.fillRect(right ? bx : bx + it.w - 3, by, 3, it.h); g.restore();
      g.textAlign = "left";
      const tx = bx + PADX + (right ? 3 : 0);
      if (mini) {
        g.fillStyle = it.col; g.beginPath(); g.arc(tx + 4, by + 16, 4, 0, 7); g.fill();
        g.font = font(600, 14.5); g.fillStyle = T.ink; g.fillText(it.label, tx + 14, by + 21);
      } else {
        let y = by + PADY;
        if (it.sLabel) { g.font = font(500, 11, true); g.fillStyle = it.col; g.fillText(it.sLabel.toUpperCase(), tx, y + 10); y += 18; }
        g.font = font(600, 16); g.fillStyle = T.ink; g.fillText(it.label, tx, y + 15); y += 20;
        g.font = font(400, 13.5); g.fillStyle = "#a2a2ac"; it.lines.forEach((l, k) => g.fillText(l, tx, y + 14 + k * 18));
      }
    }
    g.globalAlpha = 1;
  }

  /* ------------------------------------------------ export: canvas + overlay → 1080p WebM */
  record() {
    if (!window.MediaRecorder) { alert("This browser cannot record video (no MediaRecorder)."); return; }
    const W = 1920, H = 1080;
    this.recCanvas = document.createElement("canvas"); this.recCanvas.width = W; this.recCanvas.height = H;
    this.recG = this.recCanvas.getContext("2d");
    const stream = this.recCanvas.captureStream(60);
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    const mimeType = types.find(m => MediaRecorder.isTypeSupported(m)) || "";
    this.rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 16e6 });
    this.recChunks = [];
    this.rec.ondataavailable = e => { if (e.data.size) this.recChunks.push(e.data); };
    this.rec.onstop = () => {
      if (this._recKeep) {
        const blob = new Blob(this.recChunks, { type: "video/webm" }), a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = (this.script.fileName || "film") + ".webm"; document.body.appendChild(a); a.click();
        setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
      }
      this.rec = null; this.host.classList.remove("rec");
      const sv = this.el(".sv"); sv.innerHTML = `${ICON.rec}<span class="lbl">Save video</span>`;
      this.resize(); this.pause();
    };
    this.setPixels(W, H); this.compact = false; this.host.classList.remove("compact");
    this.host.classList.add("rec");
    const sv = this.el(".sv"); sv.innerHTML = `${ICON.stop}<span class="lbl">Stop</span>`;
    this.t = 0; this.draw(0); this.rec.start(250); this.play();
  }
  recFrame() {
    const g = this.recG; g.drawImage(this.cgl, 0, 0); g.drawImage(this.cov, 0, 0);
    const sv = this.el(".sv .lbl"); if (sv) sv.textContent = `Stop · ${Math.round(100 * this.t / this.tl.duration)}%`;
  }
  stopRecording(keep) { if (!this.rec) return; this._recKeep = keep; this.playing = false; this.rec.stop(); }
}

export async function createCinema(host, script, opts) { const c = new Cinema(host, script, opts); await c.init(); return c; }

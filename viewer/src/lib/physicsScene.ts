import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { colormapRgb } from './colormap';

/** Shared DESC 3D view for the physics modes.

Geometry and |B| come from the same DESC grid in the bundle. One
field period is tiled around Z into a closed torus (θ and ζ both
wrap). Do not mix these vertices with Design-mode surfacePoints() —
DESC's θ is not the input parameterization.
*/

/** The ζ / φ cut marks, matching ZETA_HIGHLIGHT in heatmap.ts. White core
    plus a dark casing: the torus is painted in viridis or plasma, whose hues
    between them run purple → teal → green → magenta → orange → yellow, so an
    achromatic mark is the only one that survives both. */
const ZETA_MARK_3D = 0xffffff;
const ZETA_CASING_3D = 0x08080c;

/** Selection mark on the torus. The 2D cross can stay amber because it is
    a thin stroke read against a flat cell, but a dot on the shaded 3D
    surface is a solid patch of colour competing with the ramp underneath --
    amber sits inside plasma's own orange band, so the dot goes achromatic
    like the ζ marks and lets shape carry the identity instead. */
const MARK_3D = 0xffffff;

/** Meshes and lines own a geometry and a material; a plain Object3D used as
    a container does not, and traverse() hands us both. */
type Drawable = THREE.Mesh | THREE.Line;

function isDrawable(obj: THREE.Object3D): obj is Drawable {
  return (obj as Drawable).geometry !== undefined;
}

/** Release one traversed object's GPU resources, skipping any material in
    `keep` -- the module-level materials are shared across rebuilds and must
    outlive the objects that reference them. */
function disposeDrawable(obj: THREE.Object3D, keep: THREE.Material[] = []): void {
  if (!isDrawable(obj)) return;
  obj.geometry.dispose();
  const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
  for (const mat of mats) {
    if (mat && !keep.includes(mat)) mat.dispose();
  }
}

/** A marker dot standing on the colormapped torus: white core inside a dark
inverted hull, the same core/casing pair the 2D cross and the ζ rim use.

A dot centred exactly on the surface would bury its own casing: viewed
face-on, the hull's back faces fall below the surface plane and the torus
depth-tests them away. So markers carry a `lift` and are nudged toward the
camera every frame (see liftMarker) -- the ball floats just clear of the
surface, the halo survives at any camera angle, and the lift stays small
enough that the torus still occludes the mark from behind.
*/
function surfaceDot(radius, opacity = 1) {
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 20, 14),
    new THREE.MeshBasicMaterial({
      color: MARK_3D, transparent: opacity < 1, opacity,
    }),
  );
  // 1.9x, a touch fatter than the 2D cross's casing-to-core ratio: a sphere
  // shows its halo only as a rim, where the flat cross shows it along the
  // whole stroke.
  dot.add(new THREE.Mesh(
    new THREE.SphereGeometry(radius * 1.9, 20, 14),
    new THREE.MeshBasicMaterial({ color: ZETA_CASING_3D, side: THREE.BackSide }),
  ));
  dot.userData.lift = radius * 2.4;
  return dot;
}

/** Closed torus: one-FP DESC vertices, rotated NFP times, sewn in θ and ζ. */
function closedTorusGeometry(vertices, B, nTheta, nZeta, nfp, cmap, vmin, vmax) {
  const nfpN = Math.max(1, nfp | 0);
  const nZ = nZeta * nfpN;
  const nVerts = nTheta * nZ;
  const pos = new Float32Array(nVerts * 3);
  const col = new Float32Array(nVerts * 3);
  const span = (vmax - vmin) || 1;
  for (let p = 0; p < nfpN; p++) {
    const c = Math.cos((2 * Math.PI * p) / nfpN);
    const s = Math.sin((2 * Math.PI * p) / nfpN);
    for (let it = 0; it < nTheta; it++) {
      for (let iz = 0; iz < nZeta; iz++) {
        const src = it * nZeta + iz;
        const dst = it * nZ + p * nZeta + iz;
        const x = vertices[src][0];
        const y = vertices[src][1];
        const z = vertices[src][2];
        pos[dst * 3] = x * c - y * s;
        pos[dst * 3 + 1] = x * s + y * c;
        pos[dst * 3 + 2] = z;
        const [cr, cg, cb] = colormapRgb(cmap, (B[src] - vmin) / span);
        col[dst * 3] = cr / 255;
        col[dst * 3 + 1] = cg / 255;
        col[dst * 3 + 2] = cb / 255;
      }
    }
  }
  const indices = [];
  for (let it = 0; it < nTheta; it++) {
    const itNext = (it + 1) % nTheta;
    for (let iz = 0; iz < nZ; iz++) {
      const izNext = (iz + 1) % nZ;
      const a = it * nZ + iz;
      const b = itNext * nZ + iz;
      const c = itNext * nZ + izNext;
      const d = it * nZ + izNext;
      indices.push(a, b, c, a, c, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  // NOTE: there is no per-attribute colour space in three. A
  // `colorAttr.colorSpace = SRGBColorSpace` used to sit here and was a no-op
  // -- nothing in the renderer reads it -- so these colormap bytes are
  // uploaded as-is and treated as working (linear) space, while the 2D
  // heatmap's putImageData path treats the same bytes as sRGB. That makes
  // the torus read slightly lighter than the map for the same |B|.
  // Reconciling them changes every rendered colour, so it is left alone
  // here rather than folded into a typing fix.
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function lineFromXyz(xyz, color, loop) {
  const pts = xyz.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  if (loop && pts.length > 1) pts.push(pts[0].clone());
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
  return new THREE.Line(geo, mat);
}

function wrapAngle(d) {
  return Math.atan2(Math.sin(d), Math.cos(d));
}

function tiledVertex(vertices, nTheta, nZeta, nfp, it, izFull) {
  const nfpN = Math.max(1, nfp | 0);
  const nZetaN = Math.max(1, nZeta | 0);
  const p = Math.floor(izFull / nZetaN);
  const iz = izFull - p * nZetaN;
  const src = it * nZetaN + iz;
  const v = vertices[src];
  const ang = (2 * Math.PI * p) / nfpN;
  const c = Math.cos(ang), s = Math.sin(ang);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
}

/** Poloidal loop on the drawn torus at Boozer ζ — samples the mesh, so the
    overlay cannot float off the surface or chord through the hole. */
function zetaCutLoop(vertices, nTheta, nZeta, nfp, zeta) {
  const nfpN = Math.max(1, nfp | 0);
  const nZetaN = Math.max(1, nZeta | 0);
  const nZ = nZetaN * nfpN;
  const z = ((zeta % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const zIndex = (z / (2 * Math.PI)) * nZ;
  const iz0 = Math.floor(zIndex) % nZ;
  const t = zIndex - Math.floor(zIndex);
  const iz1 = (iz0 + 1) % nZ;
  const loop = [];
  for (let it = 0; it < nTheta; it++) {
    const a = tiledVertex(vertices, nTheta, nZetaN, nfpN, it, iz0);
    const b = tiledVertex(vertices, nTheta, nZetaN, nfpN, it, iz1);
    loop.push([
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ]);
  }
  return loop;
}

/** Closed poloidal loop: cylindrical φ = const ∩ flux surface (not φ+π). */
function phiCutLoop(vertices, nTheta, nZeta, nfp, phi) {
  const nfpN = Math.max(1, nfp | 0);
  const nZ = nZeta * nfpN;
  const loop = [];
  for (let it = 0; it < nTheta; it++) {
    let hit = null;
    for (let iz = 0; iz < nZ; iz++) {
      const a = tiledVertex(vertices, nTheta, nZeta, nfp, it, iz);
      const b = tiledVertex(vertices, nTheta, nZeta, nfp, it, (iz + 1) % nZ);
      const d0 = wrapAngle(Math.atan2(a[1], a[0]) - phi);
      const d1 = wrapAngle(Math.atan2(b[1], b[0]) - phi);
      if (!(d0 <= 0 && d1 > 0)) continue;
      const span = (d1 - d0) || 1e-9;
      const t = Math.min(1, Math.max(0, -d0 / span));
      hit = [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
      ];
      break;
    }
    if (hit) loop.push(hit);
  }
  return loop;
}

function phiCutPlane(loop, phi) {
  let cx = 0, cy = 0, cz = 0;
  let rMin = Infinity, rMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (const p of loop) {
    cx += p[0]; cy += p[1]; cz += p[2];
    const R = Math.hypot(p[0], p[1]);
    if (R < rMin) rMin = R;
    if (R > rMax) rMax = R;
    if (p[2] < zMin) zMin = p[2];
    if (p[2] > zMax) zMax = p[2];
  }
  const n = loop.length;
  cx /= n; cy /= n; cz /= n;
  const w = Math.max(rMax - rMin, 0.08) * 1.7;
  const h = Math.max(zMax - zMin, 0.08) * 1.7;
  const group = new THREE.Group();
  const slab = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.04),
    // Lower opacity than the old pink slab: white at 0.5 flattens the |B|
    // colours it covers, which is the field the plane is there to cut.
    new THREE.MeshBasicMaterial({
      color: ZETA_MARK_3D, transparent: true, opacity: 0.16,
      side: THREE.DoubleSide, depthWrite: false,
    }),
  );
  slab.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(slab.geometry),
    new THREE.LineBasicMaterial({ color: ZETA_MARK_3D }),
  ));
  const c = Math.cos(phi), s = Math.sin(phi);
  group.matrix.makeBasis(
    new THREE.Vector3(c, s, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(-s, c, 0),
  );
  group.matrix.setPosition(cx, cy, cz);
  group.matrixAutoUpdate = false;
  group.add(slab);
  return group;
}

function loopPoints(xyz) {
  const pts = xyz.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  if (pts.length > 2 && pts[0].distanceToSquared(pts[pts.length - 1]) < 1e-16) pts.pop();
  return pts;
}

/** Tube that only follows consecutive surface samples — no spline chords through empty space. */
function loopRibbon(xyz, color, radius, casing = ZETA_CASING_3D) {
  const pts = loopPoints(xyz);
  if (pts.length < 3) return null;
  const n = pts.length;
  const edges = [];
  for (let i = 0; i < n; i++) edges.push(pts[i].distanceTo(pts[(i + 1) % n]));
  const sorted = edges.slice().sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)] || 1;
  const maxEdge = med * 4;
  const path = new THREE.CurvePath<THREE.Vector3>();
  for (let i = 0; i < n; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    if (a.distanceTo(b) > maxEdge) continue;
    path.add(new THREE.LineCurve3(a, b));
  }
  if (!path.curves.length) return null;
  const segments = Math.max(48, path.curves.length * 6);
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(path, segments, radius, 8, false),
    new THREE.MeshBasicMaterial({ color }),
  );
  if (casing != null) {
    // Inverted hull: a fatter tube drawn back-faces-only sits behind the core
    // everywhere except the silhouette, where it reads as a dark outline. The
    // torus underneath is painted in viridis or plasma, and a bare white rim
    // would sink into either ramp's yellow end without it.
    tube.add(new THREE.Mesh(
      new THREE.TubeGeometry(path, segments, radius * 1.85, 8, false),
      new THREE.MeshBasicMaterial({ color: casing, side: THREE.BackSide }),
    ));
  }
  return tube;
}

export function createPhysicsScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0c0e);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.position.set(2, 1.5, 2);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(3, 5, 4);
  scene.add(dirLight);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dirLight2.position.set(-3, -2, -4);
  scene.add(dirLight2);

  const axes = new THREE.AxesHelper(1.5);
  scene.add(axes);

  const root = new THREE.Group();
  scene.add(root);

  // Hover ghost. It lives outside `root` so clearRoot() never disposes it,
  // and the render loop is already continuous -- moving it costs nothing,
  // where setBundle() would rebuild the whole torus on every mousemove.
  // Slightly smaller and a touch translucent, so it reads as provisional
  // next to the pinned mark without giving up any contrast.
  const hoverMarker = surfaceDot(0.026, 0.9);
  hoverMarker.visible = false;
  scene.add(hoverMarker);
  let pinnedMarker = null;

  const flatMaterial = new THREE.MeshStandardMaterial({
    color: 0x5b9dff, metalness: 0.1, roughness: 0.6, side: THREE.DoubleSide,
  });
  const colorMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.DoubleSide, toneMapped: false,
  });

  // Keep each marker floating just off the surface on the camera side, so
  // its dark halo is never swallowed by the torus it is standing on.
  const liftVec = new THREE.Vector3();
  function liftMarker(mesh) {
    const base = mesh && mesh.visible && mesh.userData.base;
    if (!base) return;
    liftVec.set(base[0], base[1], base[2]);
    liftVec.subVectors(camera.position, liftVec);
    const len = liftVec.length() || 1;
    liftVec.multiplyScalar((mesh.userData.lift || 0) / len);
    mesh.position.set(base[0] + liftVec.x, base[1] + liftVec.y, base[2] + liftVec.z);
  }

  let running = true;
  function animate() {
    if (!running) return;
    requestAnimationFrame(animate);
    controls.update();
    liftMarker(hoverMarker);
    liftMarker(pinnedMarker);
    renderer.render(scene, camera);
  }
  animate();

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w < 2 || h < 2) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);

  function clearRoot() {
    while (root.children.length) {
      const child = root.children[0];
      root.remove(child);
      child.traverse((obj) => disposeDrawable(obj, [flatMaterial, colorMaterial]));
    }
  }

  function resetView() {
    camera.position.set(2, 1.5, 2);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function setBundle(bundle, opts) {
    clearRoot();
    pinnedMarker = null;  // clearRoot just disposed it
    if (!bundle) return;
    const {
      cmap = 'viridis',
      showB = true,
      showAxis = true,
      showAxes = true,
      showCutPlane = false,
      showPhiPlane = false,
      showFieldLines = false,
      showFieldLine = false,
      cutXyz = null,
      zeta = null,
      markerXyz = null,
      highlightSeed = -1,
      phi = null,
    } = opts || {};

    axes.visible = !!showAxes;
    const meta = bundle.meta;
    const nfp = meta.nfp || 1;
    const vmin = meta.B_min;
    const vmax = meta.B_max;
    const surf = bundle.surface3d;
    if (!surf || !surf.vertices) return;
    const geometry = closedTorusGeometry(
      surf.vertices, surf.B, surf.n_theta, surf.n_zeta, nfp, cmap, vmin, vmax,
    );
    const material = showB ? colorMaterial : flatMaterial;
    colorMaterial.vertexColors = true;
    root.add(new THREE.Mesh(geometry, material));

    if (showAxis && bundle.axis && bundle.axis.xyz) {
      root.add(lineFromXyz(bundle.axis.xyz, 0xe8e8ec, false));
    }

    // The swept-ζ rim: the poloidal loop the slider is currently sitting on.
    // It is the only cut mark left on the torus -- the all-planes loops that
    // used to accompany it crowded the colormap the top view exists to show.
    if (showCutPlane) {
      const onSurface = Number.isFinite(zeta)
        ? zetaCutLoop(surf.vertices, surf.n_theta, surf.n_zeta, nfp, zeta)
        : cutXyz;
      if (onSurface && onSurface.length > 2) {
        const rim = loopRibbon(onSurface, ZETA_MARK_3D, 0.016);
        if (rim) root.add(rim);
      }
    }

    if (showPhiPlane) {
      const phiCut = Number.isFinite(phi) ? phi : ((bundle.poincare && bundle.poincare.phi) || 0);
      const loop = phiCutLoop(surf.vertices, surf.n_theta, surf.n_zeta, nfp, phiCut);
      if (loop.length > 2) {
        root.add(phiCutPlane(loop, phiCut));
        const ribbon = loopRibbon(loop, ZETA_MARK_3D, 0.018);
        if (ribbon) root.add(ribbon);
      }
    }

    const seeds = (bundle.poincare && bundle.poincare.seeds) || [];
    const surfaces = (bundle.poincare && bundle.poincare.surfaces_s) || [];
    const sMin = surfaces.length ? Math.min(...surfaces) : 0;
    const sMax = surfaces.length ? Math.max(...surfaces) : 1;
    const sSpan = (sMax - sMin) || 1;
    if (showFieldLines || showFieldLine) {
      for (let si = 0; si < seeds.length; si++) {
        const seed = seeds[si];
        if (!seed.xyz || !seed.xyz.length) continue;
        const isHi = si === highlightSeed;
        if (showFieldLine && !showFieldLines && !isHi) continue;
        const frac = (seed.s - sMin) / sSpan;
        const color = isHi ? 0xffb454 : new THREE.Color().setHSL(0.65 - 0.45 * frac, 0.65, 0.55);
        const line = lineFromXyz(seed.xyz, color, false);
        if (line.material) {
          line.material.opacity = isHi ? 0.95 : 0.35;
          line.material.transparent = true;
        }
        root.add(line);
      }
    }

    if (markerXyz && markerXyz.length === 3) {
      pinnedMarker = surfaceDot(0.034);
      pinnedMarker.userData.base = markerXyz;
      pinnedMarker.position.set(markerXyz[0], markerXyz[1], markerXyz[2]);
      root.add(pinnedMarker);
    }
  }

  /** Show the ghost at a surface point, or hide it with a null xyz. */
  function setHoverMarker(xyz) {
    const ok = !!xyz && xyz.length === 3 && xyz.every(Number.isFinite);
    if (ok) {
      hoverMarker.userData.base = xyz;
      hoverMarker.position.set(xyz[0], xyz[1], xyz[2]);
    }
    hoverMarker.visible = ok;
  }

  function dispose() {
    running = false;
    hoverMarker.traverse((obj) => disposeDrawable(obj));
    clearRoot();
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  }

  return { setBundle, setHoverMarker, resetView, resize, dispose };
}

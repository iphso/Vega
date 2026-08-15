/** 2D Boozer cut: equally spaced ζ slices of the surface in (R, Z). */

import { canvasCssSize, ZETA_CASING, ZETA_HIGHLIGHT } from '../heatmap';

export const EQUAL_CUTS = 6;

// Six hues for the six equally spaced slices. The last slot used to be
// near-white (#e8e8ec), which now belongs to the swept-ζ highlight alone --
// two white outlines in one panel is exactly the ambiguity the highlight is
// supposed to resolve -- so it is violet instead.
const SLICE_COLORS = [
  '#e74c3c',
  '#3b82f6',
  '#22c55e',
  '#22d3ee',
  '#e879f9',
  '#a78bfa',
];

function lerpPlane(a, b, t) {
  const n = a.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

export function interpolatedCut(bundle, zeta) {
  const cuts = bundle && bundle.cuts;
  if (!cuts || !cuts.zeta || !cuts.zeta.length) return null;
  const zetas = cuts.zeta;
  const nfp = bundle.meta.nfp || 1;
  const period = (2 * Math.PI) / nfp;
  const K = zetas.length;
  let z = ((zeta % period) + period) % period;
  let k0 = 0;
  for (let k = 0; k < K; k++) {
    if (zetas[k] <= z) k0 = k;
  }
  const k1 = (k0 + 1) % K;
  const z0 = zetas[k0];
  const z1 = k1 === 0 ? zetas[0] + period : zetas[k1];
  const span = (z1 - z0) || 1;
  const t = (z - z0) / span;
  const R = lerpPlane(cuts.R[k0], cuts.R[k1], t);
  const Z = lerpPlane(cuts.Z[k0], cuts.Z[k1], t);
  const xyz0 = cuts.xyz[k0];
  const xyz1 = cuts.xyz[k1];
  const xyz = xyz0.map((p, i) => [
    p[0] + (xyz1[i][0] - p[0]) * t,
    p[1] + (xyz1[i][1] - p[1]) * t,
    p[2] + (xyz1[i][2] - p[2]) * t,
  ]);
  return { R, Z, xyz, zeta: z, k0, k1, t };
}

export function equalZetaCuts(bundle, n = EQUAL_CUTS) {
  const nfp = bundle.meta.nfp || 1;
  const period = (2 * Math.PI) / nfp;
  const count = Math.max(2, n | 0);
  const slices = [];
  for (let i = 0; i < count; i++) {
    const zeta = (i * period) / count;
    const cut = interpolatedCut(bundle, zeta);
    if (cut) slices.push(cut);
  }
  return { slices, period, delta: period / count };
}

function strokeClosed(ctx, xs, ys, toX, toY) {
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const x = toX(xs[i]), y = toY(ys[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
}

function fillClosed(ctx, xs, ys, toX, toY) {
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const x = toX(xs[i]), y = toY(ys[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
}

function niceTicks(min, max, n = 4) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

function outwardAnchor(xs, ys) {
  let cx = 0, cy = 0;
  for (let i = 0; i < xs.length; i++) { cx += xs[i]; cy += ys[i]; }
  cx /= xs.length;
  cy /= ys.length;
  let best = 0, bestD = -1;
  for (let i = 0; i < xs.length; i++) {
    const d = (xs[i] - cx) ** 2 + (ys[i] - cy) ** 2;
    if (d > bestD) { bestD = d; best = i; }
  }
  const dx = xs[best] - cx;
  const dy = ys[best] - cy;
  const len = Math.hypot(dx, dy) || 1;
  return { x: xs[best] + (dx / len) * 0.04, y: ys[best] + (dy / len) * 0.04 };
}

export function drawBoozerCut(canvas, bundle, zeta, opts) {
  if (!bundle) return;
  const projection = (opts && opts.projection) || 'RZ';
  const marker = opts && opts.marker;
  const { slices } = equalZetaCuts(bundle);
  const current = interpolatedCut(bundle, zeta);
  if (!slices.length || !current) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const { w: cssW, h: cssH } = canvasCssSize(canvas);
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 44, padR = 12, padT = 22, padB = 36;
  const plotW = Math.max(10, cssW - padL - padR);
  const plotH = Math.max(10, cssH - padT - padB);

  const xsOf = (R, xyz) => (projection === 'XZ' ? xyz.map((p) => p[0]) : R);
  const ysOf = (Z, xyz) => (projection === 'XZ' ? xyz.map((p) => p[2]) : Z);

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  const consider = (xs, ys) => {
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] < xMin) xMin = xs[i];
      if (xs[i] > xMax) xMax = xs[i];
      if (ys[i] < yMin) yMin = ys[i];
      if (ys[i] > yMax) yMax = ys[i];
    }
  };
  for (const slice of slices) {
    consider(xsOf(slice.R, slice.xyz), ysOf(slice.Z, slice.xyz));
  }

  const xPad = (xMax - xMin) * 0.12 || 0.05;
  const yPad = (yMax - yMin) * 0.12 || 0.05;
  xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
  const xSpan = xMax - xMin, ySpan = yMax - yMin;
  const scale = Math.min(plotW / xSpan, plotH / ySpan);
  const ox = padL + (plotW - xSpan * scale) / 2;
  const oy = padT + (plotH - ySpan * scale) / 2;
  const toX = (v) => ox + (v - xMin) * scale;
  const toY = (v) => oy + (yMax - v) * scale;

  ctx.strokeStyle = '#2a2a30';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toX(xMin), toY(0));
  ctx.lineTo(toX(xMax), toY(0));
  ctx.moveTo(toX(xMin), toY(yMin));
  ctx.lineTo(toX(xMin), toY(yMax));
  ctx.stroke();

  ctx.fillStyle = '#8a8a94';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of niceTicks(xMin, xMax)) {
    ctx.beginPath();
    ctx.strokeStyle = '#2a2a30';
    ctx.moveTo(toX(t), toY(yMin));
    ctx.lineTo(toX(t), toY(yMin) - 4);
    ctx.stroke();
    ctx.fillStyle = '#8a8a94';
    ctx.fillText(t.toFixed(1), toX(t), toY(yMin) + 6);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of niceTicks(yMin, yMax)) {
    ctx.beginPath();
    ctx.strokeStyle = '#2a2a30';
    ctx.moveTo(toX(xMin), toY(t));
    ctx.lineTo(toX(xMin) + 4, toY(t));
    ctx.stroke();
    ctx.fillStyle = '#8a8a94';
    ctx.fillText(t.toFixed(1), toX(xMin) - 6, toY(t));
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(projection === 'XZ' ? 'X [m]' : 'R [m]', ox + xSpan * scale / 2, cssH - 6);
  ctx.save();
  ctx.translate(12, oy + ySpan * scale / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Z [m]', 0, 0);
  ctx.restore();

  const highlightZeta = opts && opts.highlightZeta;
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textBaseline = 'middle';
  if (highlightZeta) {
    for (let i = 0; i < slices.length; i++) {
      const slice = slices[i];
      const xs = xsOf(slice.R, slice.xyz);
      const ys = ysOf(slice.Z, slice.xyz);
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = SLICE_COLORS[i % SLICE_COLORS.length];
      ctx.lineWidth = 1.2;
      strokeClosed(ctx, xs, ys, toX, toY);
      ctx.globalAlpha = 1;
    }
    const curXs = xsOf(current.R, current.xyz);
    const curYs = ysOf(current.Z, current.xyz);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
    fillClosed(ctx, curXs, curYs, toX, toY);
    ctx.strokeStyle = ZETA_CASING;
    ctx.lineWidth = 6.4;
    strokeClosed(ctx, curXs, curYs, toX, toY);
    ctx.strokeStyle = ZETA_HIGHLIGHT;
    ctx.lineWidth = 2.8;
    strokeClosed(ctx, curXs, curYs, toX, toY);
    const a = outwardAnchor(curXs, curYs);
    const label = `ζ = ${current.zeta.toFixed(2)}`;
    ctx.font = 'bold 13px -apple-system, sans-serif';
    ctx.textAlign = a.x >= (xMin + xMax) / 2 ? 'left' : 'right';
    const lx = toX(a.x) + (ctx.textAlign === 'left' ? 5 : -5);
    ctx.lineJoin = 'round';
    ctx.strokeStyle = ZETA_CASING;
    ctx.lineWidth = 3.2;
    ctx.strokeText(label, lx, toY(a.y));
    ctx.fillStyle = ZETA_HIGHLIGHT;
    ctx.fillText(label, lx, toY(a.y));
  } else {
    let nearest = 0;
    let nearestD = Infinity;
    for (let i = 0; i < slices.length; i++) {
      const d = Math.abs(slices[i].zeta - current.zeta);
      if (d < nearestD) { nearestD = d; nearest = i; }
    }
    for (let i = 0; i < slices.length; i++) {
      const slice = slices[i];
      const xs = xsOf(slice.R, slice.xyz);
      const ys = ysOf(slice.Z, slice.xyz);
      const color = SLICE_COLORS[i % SLICE_COLORS.length];
      const hi = i === nearest;
      ctx.strokeStyle = color;
      ctx.lineWidth = hi ? 2.4 : 1.4;
      strokeClosed(ctx, xs, ys, toX, toY);
      const a = outwardAnchor(xs, ys);
      ctx.fillStyle = color;
      ctx.textAlign = a.x >= (xMin + xMax) / 2 ? 'left' : 'right';
      ctx.fillText(`ζ = ${slice.zeta.toFixed(2)}`, toX(a.x) + (ctx.textAlign === 'left' ? 4 : -4), toY(a.y));
    }
  }

  if (marker && marker.xyz) {
    const mx = projection === 'XZ' ? marker.xyz[0] : Math.hypot(marker.xyz[0], marker.xyz[1]);
    const my = projection === 'XZ' ? marker.xyz[2] : marker.xyz[2];
    ctx.fillStyle = '#ffb454';
    ctx.beginPath();
    ctx.arc(toX(mx), toY(my), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

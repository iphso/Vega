import { colormapRgb } from '../colormap';
import { canvasCssSize } from '../heatmap';

/** Literature Poincaré plot: punctures in the (R, Z) plane at fixed φ. */

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

export function drawPoincare(canvas, bundle, opts) {
  if (!bundle || !bundle.poincare || !bundle.poincare.seeds) return;
  const seeds = bundle.poincare.seeds;
  const cmap = (opts && opts.cmap) || 'viridis';
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const { w: cssW, h: cssH } = canvasCssSize(canvas);
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 44, padR = 12, padT = 16, padB = 36;
  const plotW = Math.max(10, cssW - padL - padR);
  const plotH = Math.max(10, cssH - padT - padB);

  let rMin = Infinity, rMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (const seed of seeds) {
    for (let i = 0; i < seed.R.length; i++) {
      if (seed.R[i] < rMin) rMin = seed.R[i];
      if (seed.R[i] > rMax) rMax = seed.R[i];
      if (seed.Z[i] < zMin) zMin = seed.Z[i];
      if (seed.Z[i] > zMax) zMax = seed.Z[i];
    }
  }
  if (!Number.isFinite(rMin)) {
    ctx.fillStyle = '#55555e';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText('no Poincaré punctures', padL, padT + 14);
    return;
  }
  const rPad = (rMax - rMin) * 0.08 || 0.05;
  const zPad = (zMax - zMin) * 0.08 || 0.05;
  rMin -= rPad; rMax += rPad; zMin -= zPad; zMax += zPad;
  const rSpan = rMax - rMin, zSpan = zMax - zMin;
  const scale = Math.min(plotW / rSpan, plotH / zSpan);
  const ox = padL + (plotW - rSpan * scale) / 2;
  const oy = padT + (plotH - zSpan * scale) / 2;
  const toX = (r) => ox + (r - rMin) * scale;
  const toY = (z) => oy + (zMax - z) * scale;

  ctx.strokeStyle = '#2a2a30';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toX(rMin), toY(0));
  ctx.lineTo(toX(rMax), toY(0));
  ctx.moveTo(toX(rMin), toY(zMin));
  ctx.lineTo(toX(rMin), toY(zMax));
  ctx.stroke();

  ctx.fillStyle = '#8a8a94';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of niceTicks(rMin, rMax)) {
    ctx.beginPath();
    ctx.strokeStyle = '#2a2a30';
    ctx.moveTo(toX(t), toY(zMin));
    ctx.lineTo(toX(t), toY(zMin) - 4);
    ctx.stroke();
    ctx.fillStyle = '#8a8a94';
    ctx.fillText(t.toFixed(1), toX(t), toY(zMin) + 6);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of niceTicks(zMin, zMax)) {
    ctx.beginPath();
    ctx.strokeStyle = '#2a2a30';
    ctx.moveTo(toX(rMin), toY(t));
    ctx.lineTo(toX(rMin) + 4, toY(t));
    ctx.stroke();
    ctx.fillStyle = '#8a8a94';
    ctx.fillText(t.toFixed(1), toX(rMin) - 6, toY(t));
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('R [m]', ox + rSpan * scale / 2, cssH - 6);
  ctx.save();
  ctx.translate(12, oy + zSpan * scale / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Z [m]', 0, 0);
  ctx.restore();

  const surfaces = bundle.poincare.surfaces_s || [];
  const sMin = surfaces.length ? Math.min(...surfaces) : 0;
  const sMax = surfaces.length ? Math.max(...surfaces) : 1;
  const sSpan = (sMax - sMin) || 1;
  const highlight = opts && Number.isInteger(opts.highlightSeed) ? opts.highlightSeed : -1;

  for (let si = 0; si < seeds.length; si++) {
    const seed = seeds[si];
    const frac = (seed.s - sMin) / sSpan;
    const [r, g, b] = colormapRgb(cmap, frac);
    const isHi = si === highlight;
    ctx.fillStyle = isHi ? '#ffb454' : `rgb(${r},${g},${b})`;
    const rad = isHi ? 2.6 : 1.6;
    for (let i = 0; i < seed.R.length; i++) {
      ctx.beginPath();
      ctx.arc(toX(seed.R[i]), toY(seed.Z[i]), rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

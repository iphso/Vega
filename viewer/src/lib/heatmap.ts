import { colormapRgb } from './colormap';

/** Canvas heatmap of a 2D scalar field + optional marching-squares contours.

Literature Boozer |B| plot: poloidal angle θ upward, toroidal angle ζ
rightward, both [0, 2π]. One field period is tiled NFP times in ζ.
*/

export const HEATMAP_PAD = { L: 42, R: 56, T: 18, B: 44 };
/* The ζ indicator is chrome, not data, and it has to stay readable on top of
   whichever colormap is live. Between them viridis and plasma sweep dark
   purple → blue → teal → green → magenta → orange → yellow, so no single hue
   is safe on both -- the old #ff2f92 sat right inside plasma's magenta
   midtones. A white core over a dark casing is achromatic, so it collides
   with neither ramp, and the casing is what carries it over the bright yellow
   ends where bare white would wash out. Every ζ mark -- map line, cut
   outline, torus rim -- uses this pair, so the panels read as one indicator. */
export const ZETA_HIGHLIGHT = '#ffffff';
export const ZETA_CASING = 'rgba(8, 8, 12, 0.88)';

export function canvasCssSize(canvas) {
  const parent = canvas.parentElement;
  const w = canvas.clientWidth || (parent && parent.clientWidth) || 0;
  const h = canvas.clientHeight || (parent && parent.clientHeight) || 0;
  return {
    w: Math.max(2, Math.round(w)),
    h: Math.max(2, Math.round(h)),
  };
}

/** Plot box in CSS px. The single source of truth for where the field
sits inside the canvas — picking, the KaTeX label overlay and the hover
chrome all have to agree with what drawHeatmap painted. */
export function heatmapGeometry(canvas) {
  const { w: cssW, h: cssH } = canvasCssSize(canvas);
  const { L: padL, R: padR, T: padT, B: padB } = HEATMAP_PAD;
  return {
    cssW, cssH, padL, padR, padT, padB,
    plotW: Math.max(10, cssW - padL - padR),
    plotH: Math.max(10, cssH - padT - padB),
  };
}

/** Marching squares over a doubly-periodic (θ, ζ) field.

|B|(θ_B, ζ_B) wraps in both angles, so the cell loops run over the full
grid and read the far edge with a modulo — otherwise every contour is cut
at θ=2π and at the last ζ column. A wrap cell's vertices are emitted at
the un-wrapped index (i+1 == nTheta) and again shifted back one period,
so the halves that leave the top/right re-enter at the bottom/left; the
caller's clip rect discards whatever lands outside the plot.
*/
function marchingSquares(field, nTheta, nZeta, level) {
  const segs = [];
  const val = (i, j) => field[(i % nTheta) * nZeta + (j % nZeta)];
  const lerp = (va, vb) => {
    const d = vb - va;
    return d === 0 ? 0.5 : (level - va) / d;
  };
  for (let i = 0; i < nTheta; i++) {
    for (let j = 0; j < nZeta; j++) {
      const v00 = val(i, j), v10 = val(i + 1, j), v01 = val(i, j + 1), v11 = val(i + 1, j + 1);
      const idx = (v00 > level ? 1 : 0) | (v10 > level ? 2 : 0) | (v11 > level ? 4 : 0) | (v01 > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const bottom = [i + lerp(v00, v10), j];
      const right = [i + 1, j + lerp(v10, v11)];
      const top = [i + lerp(v01, v11), j + 1];
      const left = [i, j + lerp(v00, v01)];
      // Cases 5 and 10 are the two saddles and must connect *oppositely*:
      // each pair of segments has to isolate the two above-level corners.
      // 5 = v00|v11 above -> (left,bottom) + (right,top).
      // 10 = v10|v01 above -> (bottom,right) + (left,top).
      const edges = {
        1: [left, bottom], 2: [bottom, right], 3: [left, right], 4: [right, top],
        5: [left, bottom, right, top], 6: [bottom, top], 7: [left, top],
        8: [left, top], 9: [bottom, top], 10: [bottom, right, left, top],
        11: [right, top], 12: [left, right], 13: [bottom, right], 14: [left, bottom],
      };
      const e = edges[idx];
      if (!e) continue;
      const wrapT = i === nTheta - 1;
      const wrapZ = j === nZeta - 1;
      for (let k = 0; k < e.length; k += 2) {
        const a = e[k], b = e[k + 1];
        segs.push(a, b);
        // Periodic ghosts so the wrapped half is drawn where it belongs.
        if (wrapT) segs.push([a[0] - nTheta, a[1]], [b[0] - nTheta, b[1]]);
        if (wrapZ) segs.push([a[0], a[1] - nZeta], [b[0], b[1] - nZeta]);
        if (wrapT && wrapZ) {
          segs.push([a[0] - nTheta, a[1] - nZeta], [b[0] - nTheta, b[1] - nZeta]);
        }
      }
    }
  }
  return segs;
}

function flattenField(field, nTheta, nZeta) {
  if (!field) return new Float32Array(nTheta * nZeta);
  if (!Array.isArray(field[0]) && !(field[0] && field[0].length)) {
    return field;
  }
  const out = new Float32Array(nTheta * nZeta);
  for (let i = 0; i < nTheta; i++) {
    for (let j = 0; j < nZeta; j++) out[i * nZeta + j] = field[i][j];
  }
  return out;
}

function tileToroidal(field, nTheta, nZeta, nfp) {
  const nfpN = Math.max(1, nfp | 0);
  if (nfpN === 1) return { field, nZeta };
  const nZ = nZeta * nfpN;
  const out = new Float32Array(nTheta * nZ);
  for (let i = 0; i < nTheta; i++) {
    for (let p = 0; p < nfpN; p++) {
      for (let j = 0; j < nZeta; j++) {
        out[i * nZ + p * nZeta + j] = field[i * nZeta + j];
      }
    }
  }
  return { field: out, nZeta: nZ };
}

/** Map a mouse event to (i, j) on the *one-period* (θ, ζ) grid. */
export function heatmapPick(canvas, clientX, clientY, nTheta, nZeta, nfp = 1) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const { padL, padT, plotW, plotH } = heatmapGeometry(canvas);
  const nfpN = Math.max(1, nfp | 0);
  const nZetaPlot = nZeta * nfpN;
  const u = (x - padL) / plotW;
  const v = 1 - (y - padT) / plotH;
  if (u < 0 || u > 1 || v < 0 || v > 1) return null;
  // Inverse of the cell-centred mapping drawHeatmap uses: sample j owns
  // the cell [j, j+1)/n, so floor() lands on the sample under the cursor.
  const jFull = Math.min(nZetaPlot - 1, Math.max(0, Math.floor(u * nZetaPlot)));
  const j = jFull % nZeta;
  const i = Math.min(nTheta - 1, Math.max(0, Math.floor(v * nTheta)));
  return { i, j, u, v };
}

/** Tick marks, plus their text unless the caller typesets it in the DOM. */
function drawAngleTicks(ctx, padL, padT, plotW, plotH, withText = true) {
  const ticks = [
    { frac: 0, label: '0' },
    { frac: 0.5, label: 'π' },
    { frac: 1, label: '2π' },
  ];
  ctx.strokeStyle = '#8a8a94';
  ctx.fillStyle = '#8a8a94';
  ctx.font = '11px -apple-system, sans-serif';
  ctx.lineWidth = 1;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of ticks) {
    const x = padL + t.frac * plotW;
    ctx.beginPath();
    ctx.moveTo(x, padT + plotH);
    ctx.lineTo(x, padT + plotH + 5);
    ctx.stroke();
    if (withText) ctx.fillText(t.label, x, padT + plotH + 7);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of ticks) {
    const y = padT + (1 - t.frac) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL - 5, y);
    ctx.lineTo(padL, y);
    ctx.stroke();
    if (withText) ctx.fillText(t.label, padL - 7, y);
  }
}

export function drawHeatmap(canvas, opts) {
  const {
    nTheta, nZeta, cmap = 'viridis', nContours = 8,
    vmin, vmax, marker, zetaLine, nfp = 1,
    xTitle = 'toroidal angle ζ',
    yTitle = 'poloidal angle θ',
    barLabel = '|B|',
    // With domLabels the caller owns every piece of text that is a symbol
    // (axis titles, tick labels, the bar label, the ζ flag) and renders it
    // as KaTeX in an overlay; canvas fillText cannot typeset θ_B, ζ_B.
    // Only the numeric colorbar limits stay on the canvas.
    domLabels = false,
  } = opts;
  const flat = flattenField(opts.field, nTheta, nZeta);
  const tiled = tileToroidal(flat, nTheta, nZeta, nfp);
  const field = tiled.field;
  const nZetaPlot = tiled.nZeta;
  const nfpN = Math.max(1, nfp | 0);

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const geo = heatmapGeometry(canvas);
  const { cssW, cssH } = geo;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, cssW, cssH);

  const { padL, padR, padT, padB, plotW, plotH } = geo;

  let lo = vmin, hi = vmax;
  if (lo == null || hi == null) {
    lo = Infinity; hi = -Infinity;
    for (let k = 0; k < field.length; k++) {
      if (field[k] < lo) lo = field[k];
      if (field[k] > hi) hi = field[k];
    }
  }
  const span = (hi - lo) || 1;

  const img = ctx.createImageData(nZetaPlot, nTheta);
  for (let i = 0; i < nTheta; i++) {
    for (let j = 0; j < nZetaPlot; j++) {
      const frac = (field[i * nZetaPlot + j] - lo) / span;
      const [r, g, b] = colormapRgb(cmap, frac);
      const y = nTheta - 1 - i;
      const off = (y * nZetaPlot + j) * 4;
      img.data[off] = r; img.data[off + 1] = g; img.data[off + 2] = b; img.data[off + 3] = 255;
    }
  }
  const tmp = document.createElement('canvas');
  tmp.width = nZetaPlot; tmp.height = nTheta;
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, padL, padT, plotW, plotH);

  // Sample j occupies the cell [j, j+1)/n of the drawn image, so its centre
  // sits at (j+0.5)/n -- the grid is endpoint=False, there is no sample at
  // 2π. Contours, markers and the ζ line all have to use this same
  // cell-centred mapping or they sit half a cell off the field they label.
  const cellX = (j) => padL + ((j + 0.5) / nZetaPlot) * plotW;
  const cellY = (i) => padT + (1 - (i + 0.5) / nTheta) * plotH;

  if (nContours > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, plotW, plotH);
    ctx.clip();
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;
    for (let c = 1; c <= nContours; c++) {
      const level = lo + (c / (nContours + 1)) * span;
      const segs = marchingSquares(field, nTheta, nZetaPlot, level);
      ctx.beginPath();
      for (let s = 0; s < segs.length; s += 2) {
        const a = segs[s], b = segs[s + 1];
        ctx.moveTo(cellX(a[1]), cellY(a[0]));
        ctx.lineTo(cellX(b[1]), cellY(b[0]));
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  ctx.strokeStyle = '#2a2a30';
  ctx.strokeRect(padL + 0.5, padT + 0.5, plotW - 1, plotH - 1);
  drawAngleTicks(ctx, padL, padT, plotW, plotH, !domLabels);
  if (!domLabels) {
    ctx.fillStyle = '#8a8a94';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(xTitle, padL + plotW / 2, cssH - 8);
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yTitle, 0, 0);
    ctx.restore();
  }

  const barX = cssW - padR + 12, barW = 10;
  for (let y = 0; y < plotH; y++) {
    const frac = 1 - y / plotH;
    const [r, g, b] = colormapRgb(cmap, frac);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(barX, padT + y, barW, 1);
  }
  ctx.fillStyle = '#8a8a94';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  if (!domLabels) ctx.fillText(barLabel, barX - 2, padT - 2);
  ctx.fillText(hi.toFixed(2), barX + barW + 3, padT + 8);
  ctx.fillText(lo.toFixed(2), barX + barW + 3, padT + plotH);

  const toPlotX = cellX;
  const toPlotY = cellY;
  let zetaFlagX = null;
  if (zetaLine != null) {
    // u is a continuous fraction of one field period, not a sample index,
    // so it maps straight onto the axis rather than through a cell centre.
    const u = Number.isFinite(zetaLine.u)
      ? Math.min(1, Math.max(0, zetaLine.u))
      : (Number.isFinite(zetaLine.j) ? (zetaLine.j + 0.5) / nZeta : null);
    if (u != null) {
      const xs = [];
      for (let p = 0; p < nfpN; p++) xs.push(padL + ((p + u) / nfpN) * plotW);
      // Casing first, core on top: the dark stroke is opaque rather than a
      // translucent glow, so the pair keeps the same contrast over viridis's
      // dark purple as over plasma's yellow.
      ctx.setLineDash([]);
      ctx.strokeStyle = ZETA_CASING;
      ctx.lineWidth = 6.5;
      for (const x of xs) {
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }
      ctx.strokeStyle = ZETA_HIGHLIGHT;
      ctx.lineWidth = 2.4;
      for (const x of xs) {
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }
      zetaFlagX = xs[0];
      if (!domLabels) {
        ctx.font = 'bold 12px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = ZETA_CASING;
        ctx.lineWidth = 3;
        ctx.strokeText('ζ', xs[0], padT - 2);
        ctx.fillStyle = ZETA_HIGHLIGHT;
        ctx.fillText('ζ', xs[0], padT - 2);
      }
    }
  }
  if (marker && Number.isFinite(marker.i) && Number.isFinite(marker.j)) {
    // Same casing trick as the ζ line: the handle keeps its amber identity
    // (it matches --handle in the UI), but bare amber on plasma's orange
    // band is nearly invisible, so it gets a dark outline underneath.
    for (const [style, width] of [[ZETA_CASING, 3.2], ['#ffb454', 1.2]]) {
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      for (let p = 0; p < nfpN; p++) {
        const x = toPlotX(p * nZeta + marker.j);
        const y = toPlotY(marker.i);
        ctx.beginPath();
        ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y);
        ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  // Plot geometry in CSS px, so a DOM overlay can pin KaTeX labels to the
  // same axes/colorbar the canvas just drew.
  return { ...geo, barX, barW, zetaFlagX };
}

import { drawHeatmap, heatmapGeometry, ZETA_HIGHLIGHT } from '../heatmap';
import { tex, B_NORM } from '../math';

const ANGLE_TICKS = [
  { frac: 0, src: '0' },
  { frac: 0.5, src: '\\pi' },
  { frac: 1, src: '2\\pi' },
];

const X_TITLE = '\\text{Boozer toroidal angle }\\zeta_B';
const Y_TITLE = '\\text{Boozer poloidal angle }\\theta_B';

function place(host, src, style, cls = 'hm-label') {
  const el = document.createElement('div');
  el.className = cls;
  el.innerHTML = tex(src);
  Object.assign(el.style, style);
  host.appendChild(el);
}

/** Pin KaTeX labels onto the axes drawHeatmap just drew.

θ_B and ζ_B carry a subscript, which canvas fillText cannot typeset -- it
would render the literal "θ_B". So the canvas draws the tick *marks* and
the overlay draws every symbol, positioned from the geometry drawHeatmap
returns (CSS px, same box as the canvas, which is inset-0 in .plot-frame).
*/
function layoutLabels(host, geo) {
  host.textContent = '';
  const { padL, padT, plotW, plotH, cssH, barX, zetaFlagX } = geo;

  for (const t of ANGLE_TICKS) {
    place(host, t.src, {
      left: `${padL + t.frac * plotW}px`,
      top: `${padT + plotH + 7}px`,
      transform: 'translateX(-50%)',
    }, 'hm-tick');
    place(host, t.src, {
      left: `${padL - 7}px`,
      top: `${padT + (1 - t.frac) * plotH}px`,
      transform: 'translate(-100%, -50%)',
    }, 'hm-tick');
  }

  place(host, X_TITLE, {
    left: `${padL + plotW / 2}px`,
    top: `${cssH - 14}px`,
    transform: 'translateX(-50%)',
  });
  // translate(-50%,-50%) rotate(...) lands the element's centre on
  // (left, top) and spins it there, so the rotated title sits in the
  // gutter regardless of how long the text is.
  place(host, Y_TITLE, {
    left: '13px',
    top: `${padT + plotH / 2}px`,
    transform: 'translate(-50%, -50%) rotate(-90deg)',
  });

  place(host, B_NORM, {
    left: `${barX - 2}px`,
    top: `${padT - 3}px`,
    transform: 'translateY(-100%)',
  }, 'hm-tick');

  if (zetaFlagX != null) {
    place(host, '\\zeta', {
      left: `${zetaFlagX}px`,
      top: `${padT - 2}px`,
      transform: 'translate(-50%, -100%)',
      color: ZETA_HIGHLIGHT,
      // The flag sits in the top padding, but the label host is inset-0 over
      // the whole frame -- a tight layout can push it onto the colormap, so
      // it carries the same dark casing the canvas line does.
      textShadow: '0 0 3px rgba(8,8,12,0.9), 0 0 1px rgba(8,8,12,0.9)',
    }, 'hm-flag');
  }
}

/** Hover chrome: a cell crosshair plus a (θ_B, ζ_B, |B|/⟨|B|⟩) readout.

Lives in its own overlay, not the label one, and is built once and then
mutated — a mousemove must not redraw the heatmap or re-run KaTeX, so the
symbols are typeset at construction and only the numbers change.
*/
const HOVER_UI = new WeakMap();

function ensureHoverUI(host) {
  let ui = HOVER_UI.get(host);
  if (ui) return ui;
  const cross = document.createElement('div');
  cross.className = 'hm-cross';
  const tip = document.createElement('div');
  tip.className = 'hm-tip';
  const vals = [];
  for (const sym of ['\\theta_B', '\\zeta_B', B_NORM]) {
    const row = document.createElement('div');
    const k = document.createElement('span');
    k.className = 'k';
    k.innerHTML = tex(sym);
    const v = document.createElement('span');
    v.className = 'v';
    row.append(k, v);
    tip.appendChild(row);
    vals.push(v);
  }
  host.append(cross, tip);
  ui = { cross, tip, vals };
  HOVER_UI.set(host, ui);
  return ui;
}

function fieldValue(bb, i, j) {
  const row = bb.B[i];
  if (row && (Array.isArray(row) || row.length !== undefined)) return row[j];
  return bb.B[i * bb.n_zeta + j];
}

/** Draw (or with pick == null, hide) the hover readout over the B map. */
export function drawBoozerBHover(host, canvas, bundle, pick) {
  if (!host) return;
  const ui = ensureHoverUI(host);
  if (!pick || !bundle || !bundle.boozer_B) {
    host.dataset.on = '0';
    return;
  }
  const bb = bundle.boozer_B;
  const nfpN = Math.max(1, (bundle.meta.nfp || 1) | 0);
  const { padL, padT, plotW, plotH, cssW } = heatmapGeometry(canvas);
  // Snap to the hovered cell of the *tiled* grid, so the crosshair sits on
  // the copy under the cursor rather than on the first field period.
  const nZetaPlot = bb.n_zeta * nfpN;
  const jFull = Math.min(nZetaPlot - 1, Math.max(0, Math.floor(pick.u * nZetaPlot)));
  const cw = plotW / nZetaPlot;
  const ch = plotH / bb.n_theta;
  const x = padL + (jFull + 0.5) * cw;
  const y = padT + plotH - (pick.i + 0.5) * ch;

  ui.cross.style.left = `${x}px`;
  ui.cross.style.top = `${y}px`;
  ui.cross.style.width = `${Math.max(6, cw)}px`;
  ui.cross.style.height = `${Math.max(6, ch)}px`;

  ui.vals[0].textContent = bb.theta[pick.i].toFixed(2);
  ui.vals[1].textContent = ((jFull * 2 * Math.PI) / nZetaPlot).toFixed(2);
  ui.vals[2].textContent = Number(fieldValue(bb, pick.i, pick.j)).toFixed(3);

  // Flip the tip to the other side near the right edge so it stays inside.
  const flip = x > padL + plotW * 0.6;
  ui.tip.style.left = `${Math.min(cssW - 8, Math.max(8, x + (flip ? -12 : 12)))}px`;
  ui.tip.style.top = `${y}px`;
  ui.tip.style.transform = `translate(${flip ? '-100%' : '0'}, -50%)`;
  host.dataset.on = '1';
}

export function drawBoozerB(canvas, bundle, opts) {
  if (!bundle || !bundle.boozer_B) return;
  const bb = bundle.boozer_B;
  const contoursOn = opts && opts.contoursOn === false ? false : true;
  const nContours = contoursOn
    ? ((opts && opts.nContours) != null ? opts.nContours : 8)
    : 0;
  const labelHost = opts && opts.labelHost;
  // Scale to *this* surface, not meta.B_min/B_max (which span every
  // surface in the bundle). A single surface occupies a sub-interval of
  // the volume range, so the global scale washes the pattern out and
  // pushes contour levels outside the range entirely -- and the |B|
  // structure on one surface is exactly what quasi-symmetry is read from.
  // Leaving vmin/vmax undefined makes drawHeatmap autoscale to the field.
  const geo = drawHeatmap(canvas, {
    field: bb.B,
    nTheta: bb.n_theta,
    nZeta: bb.n_zeta,
    nfp: bundle.meta.nfp || 1,
    cmap: (opts && opts.cmap) || 'viridis',
    nContours,
    marker: opts && opts.marker,
    zetaLine: opts && opts.zetaLine,
    domLabels: !!labelHost,
    // Boozer angles, not the geometric ones -- the distinction is the
    // whole point of the plot. Only used when there is no overlay to
    // typeset them properly.
    xTitle: 'Boozer toroidal angle ζ_B',
    yTitle: 'Boozer poloidal angle θ_B',
    // Dimensionless: the service divides |B| by its mean on this surface,
    // so the scale is backend-independent (1.0 = the surface average).
    barLabel: '|B|/⟨|B|⟩',
  });
  if (labelHost && geo) layoutLabels(labelHost, geo);
}

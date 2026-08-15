/** Shared selection across Boozer B / Cut / Poincaré lenses. */

import { tex, B_MAG } from './math';

export type PhysicsMode = 'design' | 'boozerB' | 'boozerCut' | 'poincare';

export function modeExplain(mode: PhysicsMode): { title: string; body: string; detail: string } {
  switch (mode) {
    case 'design':
      return { title: '', body: '', detail: '' };
    case 'boozerB':
      return {
        title: 'Boozer B map',
        body: `${tex(`${B_MAG}(\\theta,\\zeta)`)} — Boozer-coordinate representation of field strength.`,
        detail: 'Poloidal angle θ vs toroidal angle ζ (full 2π, field periods tiled). Hover to preview where a point sits on the 3D surface; click to pin it there.',
      };
    case 'boozerCut':
      return {
        title: 'Boozer Cut',
        body: `${tex('\\zeta = \\mathrm{const}')} — physical R–Z geometry of the flux surface.`,
        detail: `Equally spaced toroidal slices of the surface shape. Drag ${tex('\\zeta')} to highlight a slice.`,
      };
    case 'poincare':
      return {
        title: 'Poincaré',
        body: `${tex('\\varphi = \\mathrm{const}')} — physical R–Z geometry of field-line punctures.`,
        detail: 's picks the 3D/Boozer surface; Poincaré always traces s = 0.25, 0.5, 0.75, 1. Seeds = starting θ₀ per surface; punctures = hits per line.',
      };
    default: {
      const _never: never = mode;
      return _never;
    }
  }
}

export function angDiff(a: number, b: number): number {
  const t = ((a - b + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return Math.abs(t);
}

export function nearestZetaIndex(zetas: number[], zeta: number, nfp: number): number {
  const period = (2 * Math.PI) / (nfp || 1);
  let z = ((zeta % period) + period) % period;
  let k = 0;
  let best = Infinity;
  for (let i = 0; i < zetas.length; i++) {
    const d = Math.min(Math.abs(zetas[i] - z), period - Math.abs(zetas[i] - z));
    if (d < best) { best = d; k = i; }
  }
  return k;
}

export function selectionFromHeatmap(bundle, pick) {
  const bb = bundle.boozer_B;
  const i = pick.i;
  const j = pick.j;
  const xyzGrid = bb.xyz;
  const xyz = xyzGrid && xyzGrid[i] ? xyzGrid[i][j] : null;
  return {
    i, j,
    thetaB: bb.theta[i],
    zetaB: bb.zeta[j],
    xyz,
    s: bundle.meta.s,
  };
}

export function refreshSelectionXyz(bundle, sel) {
  if (!sel || !bundle || !bundle.boozer_B) return sel;
  const bb = bundle.boozer_B;
  const i = Math.min(bb.n_theta - 1, Math.max(0, sel.i));
  const j = nearestZetaIndex(bb.zeta, sel.zetaB, bundle.meta.nfp);
  const xyzGrid = bb.xyz;
  return {
    ...sel,
    i, j,
    thetaB: bb.theta[i],
    zetaB: bb.zeta[j],
    xyz: xyzGrid && xyzGrid[i] ? xyzGrid[i][j] : sel.xyz,
    s: bundle.meta.s,
  };
}

/** Field-line label θ0 = θ_B − ι ζ_B, then nearest Poincaré seed on this s. */
export function nearestPoincareSeedIndex(bundle, sel): number {
  if (!sel || !bundle || !bundle.poincare) return -1;
  const seeds = bundle.poincare.seeds || [];
  const iota = bundle.meta.iota_s || 0;
  const theta0 = sel.thetaB - iota * sel.zetaB;
  let best = -1;
  let bestD = Infinity;
  for (let k = 0; k < seeds.length; k++) {
    const seed = seeds[k];
    if (Math.abs(seed.s - sel.s) > 1e-6) continue;
    const d = angDiff(seed.theta0, theta0);
    if (d < bestD) { bestD = d; best = k; }
  }
  return best;
}

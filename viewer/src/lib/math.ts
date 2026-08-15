import katex from 'katex';

/** Field strength. B is a vector, so every label that means the magnetic
field carries the arrow -- one definition so the panels cannot drift. */
export const B_MAG = '|\\vec{B}|';

/** What the physics panels actually plot: field strength divided by its own
mean on the flux surface. DESC solves at Ψ = 1 Wb and VMEC++ at the
boundary's phiedge, so raw tesla differ between the backends by a constant.
The ratio cancels it, leaving the modulation of the field about its mean. */
export const B_NORM = '|\\vec{B}|/\\langle|\\vec{B}|\\rangle';

/** Inline TeX → HTML. Used for labels, captions, and the explain box. */
export function tex(src: string): string {
  return katex.renderToString(src, { throwOnError: false, output: 'html' });
}

/** Render every [data-math] node in root. */
export function typeset(root: ParentNode = document): void {
  root.querySelectorAll('[data-math]').forEach((el) => {
    if (!(el instanceof HTMLElement)) return;
    if (el.dataset.typeset === '1') return;
    katex.render(el.dataset.math || '', el, { throwOnError: false, output: 'html' });
    el.dataset.typeset = '1';
  });
}

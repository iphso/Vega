"""Does a SMOOTHED gradient estimate point anywhere reproducible, when the FD one does not?

THE FINDING THIS RESPONDS TO (2026-10-06, `fd_diagnostic.py --fidelity high` at
`hf_asp16`'s endpoint). The P3 objective is a STAIRCASE in our coordinates. The scan
shows `f(x + h e_i) - f(x)` holding CONSTANT across a range of h -- coord 0 gives
+1.076351e-4 at h=1e-2 and +1.074222e-4 at h=3e-2, identical to 0.2%; coord 40 gives
+4.589847e-4 and +4.490728e-4, identical to 2%; coord 20 gives +5.618904e-4 and
+5.665692e-4 at h=3e-4 and 1e-3, identical to 0.8%. A constant difference across
widening h means the function jumped once and then went flat: flat treads roughly
1e-3 to 1e-2 wide, step heights 1e-4 to 1e-3, and non-monotone (coords 20 and 40
both step back DOWN at larger h).

Consequently there is no usable finite-difference step at any scale tested. The
consecutive-gradient cosine rises monotonically with h -- 0.7050, 0.7398, 0.7530,
0.8737 -- and tops out at 0.874, i.e. ~29 degrees of disagreement, never approaching
1. That is the whole explanation of the minimax's behaviour: the LP overpredicted the
drop by 27x and from iteration 6 predicted the WRONG SIGN, because it was handed a
direction derived from step heights divided by an arbitrary denominator.

It also retires the "basin-independent barrier" reading of `hf_s0b` (+0.080991) and
`hf_asp16` (+0.093147). Both terminated when the trust region shrank below the tread
width, at which point every probe lands on a flat tread, the LP sees no descent, and
the run reports `local minimax optimum reached`. Two runs failing identically is not
two runs corroborating a physical wall.

WHAT THIS SCRIPT TESTS. Gaussian smoothing replaces f with
f_sigma(x) = E[f(x + sigma u)], u ~ N(0, I), whose gradient is
E[f(x + sigma u) u] / sigma -- estimable from function values alone, with no
differencing. If sigma exceeds the tread width the smoothed function is genuinely
smooth, and this is the standard evolution-strategies estimator. The question is not
whether the formula is correct but whether it is USABLE at an affordable sample
count, so:

  Draw TWO INDEPENDENT estimates at the same sigma and take their cosine.
  Two independent estimates of a real direction agree; two draws of noise do not.

This is the same test the FD study used, which makes the numbers directly
comparable against its 0.874 ceiling -- and it is the honest one, because a single
smoothed gradient always looks plausible and nothing about its appearance reveals
whether it would reproduce.

COST. 2 estimates x --n-samples evals per sigma, antithetic (+u, -u) so a sample
costs 2 evals and the estimator has lower variance. At --n-samples 32 that is 64
evals per estimate, 128 per sigma: comparable to one 82-point FD stencil, so a
three-sigma sweep costs about the same as three minimax iterations.

READ-OUT, pre-registered:
  - cosine >= 0.95 at some sigma: a reproducible descent direction exists where FD
    has none. That sigma is the method fix, and the minimax should be re-run with
    smoothed gradients rather than stencils.
  - cosine in 0.7-0.95: better than FD's 0.874 only if it clears it; report the
    margin and the sample count needed, do NOT declare a fix on a rounding margin.
    (The tail-not-median lesson: a threshold that decides a claim must be compared
    against the right statistic, and here the comparison is against FD's BEST pair,
    0.8737, not against its median.)
  - cosine < 0.7 everywhere: smoothing at affordable sample counts does not recover
    a direction either, and the approach needs to move to a method that never forms
    a gradient at all (model-based DFO / CMA-ES over the reduced coordinate set).

CAVEAT. One base point, one basin, phase-1 objective only. Cosine between two
estimates measures REPRODUCIBILITY, not descent: a reproducible direction can still
be a poor one. Confirming descent needs a line search along it, which this script
does not do.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402


def official_violations(m):
    iota = abs(m["edge_rotational_transform_over_n_field_periods"])
    return np.array([
        (0.25 - iota) / 0.25,
        (np.log10(m["qi"]) + 3.5) / 3.5,
        (m["edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (m["flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        (0.0 - m["vacuum_well"]) / 1e-1,
    ])


def phase1(m):
    """Sum of squared POSITIVE violations -- the same scalar fd_diagnostic used, so
    the cosines here are comparable with its 0.8737."""
    if m is None:
        return None
    return float(np.sum(np.maximum(0.0, official_violations(m)) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-npy", default="/work/output/hf_asp16_best_seed.npy")
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--coef-scales", default="/work/output/coef_scales.npy")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--n-samples", type=int, default=32,
                   help="antithetic pairs per estimate; each costs 2 evals")
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--timeout-seconds", type=float, default=1800.0)
    p.add_argument("--fidelity", default="high", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    X = np.load(a.seed_npy)
    row = X[a.seed_row]
    x0 = row[:90].astype(np.float64)
    nfp = int(row[90]) if X.shape[1] > 90 else 2
    scales = np.load(a.coef_scales).astype(np.float64)
    free = np.array([i for i in range(90) if i not in oracle.ZERO_INDICES])
    z0 = x0[free] / scales[free]
    fid = oracle.FIDELITY_PRESETS[a.fidelity]
    n = len(free)
    rng = np.random.default_rng(a.seed)

    def z_to_args(z):
        x = x0.copy()
        x[free] = z * scales[free]
        x[oracle.ZERO_INDICES] = 0.0
        return oracle.params_to_worker_args(x, {"nfp": nfp}, fid)

    def evaluate(zs):
        jobs, order = [], {}
        for i, z in enumerate(zs):
            rc, zsn, nf, fn = z_to_args(z)
            jobs.append((i, rc, zsn, nf, fn))
            order[i] = i
        out = [None] * len(zs)
        for t, ok, payload in run_batch_with_timeout(
                jobs, oracle.worker_fn, a.n_workers, a.timeout_seconds):
            if not ok:
                continue
            m = {k: payload.get(k) for k in oracle.TARGET_NAMES}
            if all(v is not None for v in m.values()):
                out[order[t]] = m
        return out

    base = evaluate([z0])[0]
    if base is None:
        print("[es] base point did not evaluate. NOTE: a failed eval is "
              "indistinguishable from a non-converging design -- check the image has "
              "`constellaration` before reading this as physics.", flush=True)
        return
    f0 = phase1(base)
    v0 = official_violations(base)
    print(f"[es] base f={f0:.8e} worst={v0.max():+.6f} nfp={nfp} n_free={n} "
          f"fidelity={a.fidelity}", flush=True)
    print(f"[es] FD reference to beat: best consecutive-gradient cosine 0.8737 "
          f"(h=1e-2 vs 2e-2)", flush=True)

    def smoothed_grad(sigma, label):
        """Antithetic Gaussian-smoothing estimate of grad f_sigma at z0."""
        U = rng.standard_normal((a.n_samples, n))
        zs, tags = [], []
        for i, u in enumerate(U):
            zs.append(z0 + sigma * u); tags.append((i, +1))
            zs.append(z0 - sigma * u); tags.append((i, -1))
        ms = evaluate(zs)
        g = np.zeros(n)
        kept = 0
        for (i, s), m in zip(tags, ms):
            f = phase1(m)
            if f is None:
                continue
            # Antithetic central form: each sample contributes (f+ - f-)/(2 sigma) u,
            # accumulated here one side at a time so a single failed eval costs that
            # side only rather than discarding the pair.
            g += (s * f / (2.0 * sigma)) * U[i]
            kept += 1
        g /= max(a.n_samples, 1)
        print(f"[es]   {label}: sigma={sigma:g} kept {kept}/{2*a.n_samples} evals "
              f"|g|={np.linalg.norm(g):.4e}", flush=True)
        return g

    print(f"\n{'sigma':>8} {'cosine(A,B)':>12} {'|gA|':>11} {'|gB|':>11} "
          f"{'|gA|/|gB|':>10}  verdict")
    results = []
    for sigma in a.sigmas:
        gA = smoothed_grad(sigma, "est A")
        gB = smoothed_grad(sigma, "est B")
        na, nb = np.linalg.norm(gA), np.linalg.norm(gB)
        cos = float(gA @ gB / (na * nb)) if na > 0 and nb > 0 else 0.0
        verdict = ("REPRODUCIBLE" if cos >= 0.95 else
                   "beats FD" if cos > 0.8737 else "no better than FD")
        print(f"{sigma:8.3g} {cos:12.4f} {na:11.4e} {nb:11.4e} "
              f"{na/max(nb,1e-30):10.3f}  {verdict}", flush=True)
        results.append((sigma, cos))

    best = max(results, key=lambda r: r[1]) if results else None
    print("\n[es] INTERPRETATION")
    if best and best[1] >= 0.95:
        print(f"[es] A reproducible direction EXISTS at sigma={best[0]:g} "
              f"(cosine {best[1]:.4f}) where finite differences top out at 0.8737. "
              f"Smoothing is the method fix: re-run the minimax with smoothed "
              f"gradients instead of the 82-point stencil, at this sigma. Confirm "
              f"with a line search along the direction before trusting descent -- "
              f"reproducible is not the same as descending.")
    elif best and best[1] > 0.8737:
        print(f"[es] sigma={best[0]:g} gives cosine {best[1]:.4f}, which beats FD's "
              f"0.8737 but does not reach 0.95. Report the margin, not a fix: at "
              f"this agreement the direction is still ~{np.degrees(np.arccos(best[1])):.0f} "
              f"degrees uncertain and a minimax LP fed it will still mispredict. "
              f"Raising --n-samples reduces variance as 1/sqrt(N); check whether the "
              f"required N is affordable before adopting.")
    else:
        print(f"[es] Smoothing at these sample counts does NOT recover a "
              f"reproducible direction (best cosine "
              f"{best[1] if best else float('nan'):.4f}). Forming a gradient at all "
              f"is the wrong move on this objective; use a method that never does -- "
              f"a RANK-based search (ALM-NGOpt / CMA-ES), which only orders "
              f"candidates, so a flat tread merely makes two of them tie.")
        # DO NOT reduce to the 29 coordinates carrying 90% of the FD column norm.
        # That set comes from `analyze_jacobian_sparsity.py`, computed from FD
        # stencils, and the staircase finding retires it: ranking coordinates by FD
        # column norm on a piecewise-constant objective ranks probe artifacts. An
        # earlier version of this script recommended exactly that -- written before
        # this experiment's own result invalidated it. The gradient-free alternative
        # was measured and is also unavailable: corpus PCA gives effective dimension
        # 32.86, needing 57 of 81 components for 95% of boundary variance
        # (`corpus_pca.py`).
        print(f"[es] NOT a coordinate reduction: the FD-derived 29-coordinate set is "
              f"invalidated by the staircase, and corpus PCA shows no low-dimensional "
              f"structure to substitute (effective dimension 32.86, 57/81 components "
              f"for 95% variance).")
    print("[es] CAVEAT: one base point, one basin, phase-1 objective. Cosine "
          "measures REPRODUCIBILITY, not descent.")


if __name__ == "__main__":
    main()

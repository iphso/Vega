"""A reduced search space derived WITHOUT gradients, from the corpus itself.

WHY THIS AND NOT THE JACOBIAN RANKING. `analyze_jacobian_sparsity.py` found 90% of
squared gradient norm in 29 of 81 coordinates and recommended searching there. That
result is computed from `hf_s0b`'s finite-difference stencil, and the 2026-10-06
staircase finding invalidates it: FD on a piecewise-constant objective returns step
height over an arbitrary denominator, so a ranking by FD column norm ranks probe
artifacts. **It must not be used to choose a reduced space.** (I pre-registered that
reduction as the fallback before knowing the result that undercuts it.)

This derives the reduction from data instead: all 3000 streamed ConStellaration
boundaries are exactly `(5, 9)`, the same window `minimax_p3.py` optimises, so the
corpus is a sample of real stellarator boundaries in precisely our coordinates. PCA
over it needs no gradient, no VMEC++ call, and cannot be confounded by resolvability
or by the staircase.

Prior support: `paper_v1`'s search cloud collapsed to ~2.4 effective dimensions, so
a low intrinsic dimension here is expected rather than surprising.

WHAT IT IS FOR. CMA-ES is the right family on a staircase because it is RANK-based:
it orders candidates and never differences them, so a flat tread merely makes two
candidates tie instead of producing a zero gradient. Its cost, however, scales badly
with dimension -- the standard population is 4 + 3 ln(n) and the covariance has
n(n+1)/2 free parameters, which at n=81 is 3321. Searching a PCA subspace is what
makes it affordable at ~600 core-seconds per evaluation.

WHITENING. Coefficients span a 540x dynamic range across the corpus, so PCA is run
on whitened coordinates using the project's existing `coef_scales.npy`. Without
that, the components are just the few largest-variance modes.

READ-OUT: the number of components needed for 95% / 99% of boundary variance is the
candidate search dimension. If that number is small (<= ~20) a reduced CMA-ES is
affordable; if it is close to 81 the corpus has no low-dimensional structure and the
dimension cannot be bought this way.

CAVEAT. Corpus variance is not the same as the directions that matter for P3
feasibility -- the corpus contains no feasible design (best worst-violation 0.3328).
This bounds where real boundaries live, not where the feasible set is. A subspace
that captures boundary variance can still omit the direction feasibility needs, so
any reduced run must be checked against an unreduced baseline before the reduction
is believed.
"""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-records", type=int, default=3000)
    p.add_argument("--coef-scales", default="/work/output/coef_scales.npy")
    p.add_argument("--out", default="/work/output/corpus_pca.npz")
    a = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("proxima-fusion/constellaration", split="train",
                      streaming=True)

    rows, nfps = [], []
    n = 0
    for rec in ds:
        if n >= a.max_records:
            break
        n += 1
        if not rec.get("boundary.is_stellarator_symmetric"):
            continue
        rc = np.asarray(rec["boundary.r_cos"], dtype=float)
        zs = np.asarray(rec["boundary.z_sin"], dtype=float)
        if rc.shape != (5, 9) or zs.shape != (5, 9):
            continue
        rows.append(np.concatenate([rc.ravel(), zs.ravel()]))
        nfps.append(int(rec["boundary.n_field_periods"]))

    X = np.asarray(rows)
    nfps = np.asarray(nfps)
    print(f"[pca] {X.shape[0]} boundaries x {X.shape[1]} coefficients")

    scales = np.load(a.coef_scales).astype(float)
    use = scales > 0
    print(f"[pca] whitening with coef_scales.npy ({int(use.sum())} nonzero scales; "
          f"the corpus spans a ~540x coefficient range, so unwhitened PCA would "
          f"just return the largest-variance modes)")
    Z = X[:, use] / scales[use]

    mu = Z.mean(axis=0)
    Zc = Z - mu
    # SVD rather than an explicit covariance: 3000x81 is small and SVD is the
    # better-conditioned route to the same components.
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2
    frac = np.cumsum(var) / var.sum()

    print(f"\n[pca] cumulative variance explained:")
    marks = {}
    for thr in (0.50, 0.80, 0.90, 0.95, 0.99, 0.999):
        k = int(np.searchsorted(frac, thr) + 1)
        marks[thr] = k
        print(f"   {thr*100:5.1f}%  ->  {k:>3} of {Z.shape[1]} components")

    # Participation ratio: a basis-free effective dimension, (sum l)^2 / sum l^2.
    # Quoted because it is comparable with `paper_v1`'s ~2.4 and does not depend on
    # choosing a variance threshold.
    pr = float(var.sum() ** 2 / np.sum(var ** 2))
    print(f"\n[pca] participation-ratio effective dimension: {pr:.2f}")
    print(f"[pca] (`paper_v1`'s search cloud collapsed to ~2.4 by the same measure)")

    print(f"\n[pca] per-nfp counts in the corpus: "
          f"{ {int(k): int((nfps == k).sum()) for k in np.unique(nfps)} }")

    np.savez_compressed(a.out, components=Vt, mean=mu, var=var,
                        scales=scales, use=use, frac=frac)
    print(f"\n[pca] wrote components -> {a.out}")

    k95, k99 = marks[0.95], marks[0.99]
    if k95 <= 20:
        print(f"[pca] VERDICT: {k95} components carry 95% of boundary variance, so a "
              f"reduced CMA-ES is affordable (population 4+3ln(n) = "
              f"{4 + 3*np.log(k95):.0f} at n={k95}, against "
              f"{4 + 3*np.log(81):.0f} at n=81).")
    else:
        print(f"[pca] VERDICT: 95% needs {k95} components and 99% needs {k99}; the "
              f"corpus has no strongly low-dimensional structure in these "
              f"coordinates and the search dimension cannot be bought this way.")
    print("[pca] CAVEAT: corpus variance is not the feasible direction. The corpus "
          "contains NO feasible design (best worst-violation 0.3328), so this bounds "
          "where real boundaries live, not where feasibility is. Validate any "
          "reduced run against an unreduced baseline.")


if __name__ == "__main__":
    main()

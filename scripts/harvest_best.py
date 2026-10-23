#!/usr/bin/env python3
"""Harvest the best designs from EVERY persisted eval, not just accepted steps.

WHY THIS EXISTS
---------------
`minimax_p3.py` updates its `<tag>_best.npy` only at the base point of each
iteration -- i.e. only along the accepted trajectory. But each iteration also
evaluates an 82-point finite-difference stencil, and every one of those points is
a real boundary with full VMEC++ physics. They are computed, persisted to
`<tag>_results.jsonl` (since the 2026-09-28 retention fix), and then ignored.

They should not be. Measured across 104,038 evals in 72 runs: **62 of 72 runs
contain a stencil point strictly better than that run's best accepted step**,
by 0.0005 to 0.0159 worst-violation. The restart ladders have therefore been
chaining from a slightly worse point than the search had already found, every
round, in 86% of runs.

The reason is structural, not a bug in the gradient: a stencil perturbs ONE
whitened coordinate at a time by fd_step. The minimax step then moves along a
direction that trades all five constraints off against each other, and that
direction is chosen to reduce the LINEARIZED worst violation. When curvature
bites, a lone single-coordinate perturbation can beat the full step. It is free
information about the local landscape that the solver's own update rule discards.

This script scans every results.jsonl and emits a seed .npy in the format
minimax_p3.py expects -- [x(90), nfp, worst] per row -- ranked by the chosen
criterion, with near-duplicate designs collapsed so a ladder does not spend
rounds re-refining the same basin.

CRITERIA
  violation : min worst_violation. Pushes toward feasibility, which is the gate.
  hv        : max (1+L_gradB)*(20-A). Pushes toward score, but only matters
              among designs that could plausibly become feasible, so this is
              filtered by --max-worst.
  joint     : rank by hv among designs under --max-worst. DEGENERATE IN PRACTICE
              -- hv anti-correlates with feasibility (raising aspect ratio eases
              the physics constraints and simultaneously destroys hv), so this
              just returns the highest-hv designs sitting exactly at the filter
              edge: the first run returned six rows at worst=0.118-0.119 against
              a 0.12 cap. Kept only for comparison; use pareto.
  pareto    : the non-dominated set in (minimize worst_violation, maximize hv),
              then spread across it. This is the right structure: the benchmark
              score is hypervolume of FEASIBLE designs, so a design is useless
              if it never reaches feasibility AND worth nothing if it arrives
              there with no hv. Neither axis alone is the objective and no fixed
              scalarization of the two is either -- we do not know in advance
              which basins can cross the 0.01 bar, so the front is the honest
              representation of what we have.
"""
import argparse
import glob
import json
import os

import numpy as np

STATE_DIR = "output/minimax_p3"


def load_all(pattern):
    """Yield (worst, hv, nfp, x, tag, kind) for every eval that has metrics."""
    for f in sorted(glob.glob(pattern)):
        tag = os.path.basename(f).replace("_results.jsonl", "")
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a killed run
            w = d.get("worst_violation")
            p = d.get("params")
            if w is None or p is None or len(p) != 90:
                continue
            yield (float(w), d.get("hv_contribution"), int(d.get("nfp", 2)),
                   np.asarray(p, dtype=np.float64), tag, d.get("kind"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=f"{STATE_DIR}/*_results.jsonl")
    ap.add_argument("--criterion", choices=["violation", "hv", "joint", "pareto"],
                    default="pareto")
    ap.add_argument("--max-worst", type=float, default=0.12,
                    help="for hv/joint: ignore designs worse than this. Default 0.12 "
                         "keeps everything within ~4x of the 0.01 feasibility bar; "
                         "observed per-basin reductions were 6-71%%, median 30%%, so "
                         "designs much worse than this are not credible candidates.")
    ap.add_argument("--n", type=int, default=12, help="rows to emit")
    ap.add_argument("--dedup-dist", type=float, default=2.0,
                    help="min L2 distance in WHITENED coords between emitted rows. "
                         "The 40 'diverse' lbfgs seeds had median pairwise distance "
                         "2.08 and turned out to be one basin, so 2.0 is the floor "
                         "below which two rows are the same design.")
    ap.add_argument("--coef-scales", default="output/coef_scales.npy")
    # NOT under output/minimax_p3 -- that directory is created by the container and
    # is root-owned, so host-side writes there fail with EACCES.
    ap.add_argument("--out", default="output/harvest_seeds.npy")
    args = ap.parse_args()

    scales = np.load(args.coef_scales).astype(np.float64)
    rows = list(load_all(args.glob))
    if not rows:
        print("no evals with metrics found")
        return
    print(f"scanned {len(rows):,} evals with metrics")

    pool = rows
    if args.criterion == "pareto":
        cand = [r for r in rows if r[1] is not None]
        # Non-dominated set in (minimize worst, maximize hv). Sort by worst
        # ascending, then sweep keeping only rows that raise the running hv max:
        # anything that does not is dominated by an earlier row that is both at
        # least as feasible and at least as valuable.
        cand.sort(key=lambda r: (r[0], -r[1]))
        front, hv_max = [], -np.inf
        for r in cand:
            if r[1] > hv_max:
                front.append(r)
                hv_max = r[1]
        print(f"{len(front):,} designs on the (worst, hv) Pareto front")
        # Walk the front from the feasible end outward: the near-feasible end is
        # where a reduction actually converts into score.
        pool = front
    elif args.criterion in ("hv", "joint"):
        pool = [r for r in rows if r[0] <= args.max_worst and r[1] is not None]
        print(f"{len(pool):,} within max_worst={args.max_worst}")
        pool.sort(key=lambda r: -r[1])
    else:
        pool.sort(key=lambda r: r[0])
    if not pool:
        print("nothing passed the filter; loosen --max-worst")
        return

    # Greedy: walk best-first, keep a row only if it is far enough from every row
    # already kept. Distance is measured in whitened coordinates because raw
    # Fourier coefficients span a 540x dynamic range across the dataset, so a raw
    # L2 distance is dominated by a handful of large-variance modes.
    # nfp is compared EXACTLY and separately: the 90 coefficients are per-field-
    # period amplitudes, so the same coefficient vector at nfp=4 and nfp=5 is two
    # completely different stellarators, arbitrarily far apart in real geometry. An
    # earlier version distanced on coefficients alone and silently collapsed
    # rp4_none (nfp=5) into s29 (nfp=4), dropping a genuine front member.
    kept = []
    for w, hv, nfp, x, tag, kind in pool:
        z = x[scales > 0] / scales[scales > 0]
        if any(nk == nfp and np.linalg.norm(z - zk) < args.dedup_dist
               for zk, nk, _ in kept):
            continue
        kept.append((z, nfp, (w, hv, nfp, x, tag, kind)))
        if len(kept) >= args.n:
            break

    print(f"\n{'#':>3} {'worst':>9} {'hv':>7} nfp  {'tag':14s} kind")
    out = []
    for i, (_, _, (w, hv, nfp, x, tag, kind)) in enumerate(kept):
        print(f"{i:>3} {w:9.5f} {hv if hv is None else f'{hv:7.1f}'} {nfp:>3}  {tag:14s} {kind}")
        out.append(np.concatenate([x, [nfp], [w]]))

    arr = np.asarray(out)
    np.save(args.out, arr)
    print(f"\nwrote {arr.shape} -> {args.out}")


if __name__ == "__main__":
    main()

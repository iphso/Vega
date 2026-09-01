"""Selects a fixed-size anchor pool for phase-2 directed search out of a
phase-1 (--seed-mode fixed) pool's own accepted output, ranked by P1
(GeometricalProblem) worst-single-constraint-violation ascending -- i.e.
"closest to actually feasible," using all three constraints jointly rather
than a single proxy metric like max_elongation alone (the smoke test showed
elongation isn't even the binding constraint here; triangularity is, so
ranking by elongation alone would pick anchors that look good on the wrong
axis). If any rows are genuinely P1-feasible, ranks those by the real
objective (lowest max_elongation) ahead of everything else; the rest are
ordered by worst_violation.

Writes a standalone bootstrap_generic_<out_tag>/{X,Y}.npy so
bootstrap_generic.py's existing --anchor-source-tag can point at it
unchanged -- no new anchor-source plumbing needed downstream.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from p1_report import p1_violations, TOL

OUT_DIR = Path("/home/slater_victoroff_aihub/external/vega/output")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source_tag")
    p.add_argument("out_tag")
    p.add_argument("--top-k", type=int, default=200)
    args = p.parse_args()

    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    elong_i = target_names.index("max_elongation")
    src = OUT_DIR / f"bootstrap_generic_{args.source_tag}"
    X, Y = np.load(src / "X.npy"), np.load(src / "Y.npy")

    v = p1_violations(Y, target_names)
    worst = v.max(axis=1)
    feasible = worst <= TOL
    n_feas = int(feasible.sum())

    # Feasible rows first (sorted by real objective), then the rest by
    # closeness to feasibility -- np.lexsort's primary key is the LAST arg.
    rank_key = np.where(feasible, Y[:, elong_i], np.inf)
    order = np.lexsort((rank_key, ~feasible))
    keep = order[:min(args.top_k, len(order))]

    out_dir = OUT_DIR / f"bootstrap_generic_{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X[keep])
    np.save(out_dir / "Y.npy", Y[keep])
    print(f"[{args.source_tag} -> {args.out_tag}] selected {len(keep)}/{len(X)} anchors "
          f"({n_feas} were genuinely P1-feasible) -> {out_dir}")
    print(f"  worst_violation range in selection: {worst[keep].min():+.4f} .. {worst[keep].max():+.4f}")
    print(f"  max_elongation range in selection:  {Y[keep, elong_i].min():.3f} .. {Y[keep, elong_i].max():.3f}")


if __name__ == "__main__":
    main()

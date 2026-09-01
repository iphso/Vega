"""Post-hoc report of constellaration.problems.GeometricalProblem (P1)
feasibility/objective against any bootstrap_generic.py vmec pool.

Deliberately NOT used as an accept/reject filter during generation (see
bootstrap_generic.py's sanity_filter, which stays purely physical-sanity) --
computed after the fact so near-misses stay visible in the pool for
diagnosis instead of silently vanishing. Matches constellaration.problems.
GeometricalProblem exactly: same three feasibility constraints, same 1%
relative-violation tolerance (_DEFAULT_RELATIVE_TOLERANCE), same objective
(minimize max_elongation) and score normalization (1=circular..10=elongated
-> 1.0..0.0).
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/slater_victoroff_aihub/external/vega/output")
TOL = 1e-2  # constellaration.problems._DEFAULT_RELATIVE_TOLERANCE

AR_UB = 4.0
TRI_UB = -0.5
IOTA_LB = 0.3


def p1_violations(Y, target_names):
    """Returns (N, 3) normalized constraint violations, same formula and
    same sign convention as GeometricalProblem._normalized_constraint_violations
    (<=TOL per column means that constraint is satisfied)."""
    ar = Y[:, target_names.index("aspect_ratio")]
    tri = Y[:, target_names.index("average_triangularity")]
    iota = np.abs(Y[:, target_names.index("edge_rotational_transform_over_n_field_periods")])
    return np.stack([
        (ar - AR_UB) / abs(AR_UB),
        (tri - TRI_UB) / abs(TRI_UB),
        (IOTA_LB - iota) / abs(IOTA_LB),
    ], axis=1)


def p1_score(max_elongation):
    return 1.0 - np.clip((max_elongation - 1.0) / 9.0, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tags", nargs="+", help="bootstrap_generic_<tag> pool(s) to report on")
    p.add_argument("--near-miss-k", type=int, default=5)
    args = p.parse_args()
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    ar_i = target_names.index("aspect_ratio")
    tri_i = target_names.index("average_triangularity")
    iota_i = target_names.index("edge_rotational_transform_over_n_field_periods")
    elong_i = target_names.index("max_elongation")

    for tag in args.tags:
        d = OUT_DIR / f"bootstrap_generic_{tag}"
        X, Y = np.load(d / "X.npy"), np.load(d / "Y.npy")
        v = p1_violations(Y, target_names)
        feasible = np.all(v <= TOL, axis=1)
        n_feas = int(feasible.sum())
        print(f"=== {tag} ({len(Y)} accepted candidates) ===")
        print(f"  P1-feasible: {n_feas}/{len(Y)} ({n_feas / len(Y):.3%})")
        if n_feas:
            best_idx = np.where(feasible)[0][np.argmin(Y[feasible, elong_i])]
            best_elong = Y[best_idx, elong_i]
            print(f"  best (lowest) max_elongation among feasible: {best_elong:.4f} "
                  f"(P1 score={p1_score(best_elong):.4f}), row index {best_idx}")
        worst = v.max(axis=1)
        near = np.argsort(worst)[:args.near_miss_k]
        print(f"  closest {args.near_miss_k} rows by worst single constraint violation:")
        for i in near:
            print(f"    idx={i:6d} ar={Y[i, ar_i]:7.3f} tri={Y[i, tri_i]:7.3f} "
                  f"iota={Y[i, iota_i]:7.3f} elong={Y[i, elong_i]:7.3f} worst_violation={worst[i]:+.4f}")
        print()


if __name__ == "__main__":
    main()

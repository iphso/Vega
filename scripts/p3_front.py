"""Assemble the P3 Pareto front from every feasible design any instance has
found, and report the ACTUAL benchmark score.

This exists because the project spent its entire history reporting
`worst_violation` as if it were progress. It isn't the score. The score is:

    MHDStableQIStellarator._score:
        feasible = [m for m in metrics if is_feasible(m)]     # ALL 5 normalized
        if not feasible: return 0.0                           # violations <= 1e-2
        X = [(-L_gradB, aspect_ratio) for m in feasible]
        return hypervolume(X, reference_point=(1.0, 20.0))

So every result this project has ever produced scored exactly **0.0** -- no
feasible design has ever been found. `worst_violation=0.2036` and
`worst_violation=10.0` have identical scores.

Two consequences that drive strategy, both easy to get wrong:

  * A single point contributes (1 + L_gradB) * (20 - A). L_gradB is HALF the
    score and is UNCONSTRAINED -- once feasible it is the cheap axis to push.
  * DENSITY IS WORTHLESS. A dominated point contributes exactly zero. Ten
    feasible designs at the same L_gradB score identically to the single one
    with the lowest aspect ratio. Only genuine trade-offs (lower A at lower
    L_gradB, or higher L_gradB at the same A) add volume.

Reads output/ngopt_alm_p3_instance/feasible_pool.jsonl (appended by every
instance of ngopt_alm_p3_instance.py as it finds feasible designs).

NOTE ON FIDELITY: the pool is filled from the LOW-fidelity search oracle, but
the real scorer calls `ConstellarationSettings.default_high_fidelity()`
unconditionally, and fidelity_check.py measured high fidelity reading ~0.017
WORSE on the active constraints. So the number this prints is an UPPER BOUND.
Use --validate to re-run the front members through the high-fidelity oracle and
get the honest score before making any claim.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "ngopt_alm_p3_instance"

REFERENCE_POINT = np.array([1.0, 20.0])  # (-L_gradB, aspect_ratio)
FEASIBLE_TOL = 1e-2

# The five OFFICIAL constraints. aspect_ratio is objective #2, not a constraint.
OFFICIAL = [
    ("edge_rotational_transform_over_n_field_periods", ">=", 0.25, True, None, False),
    ("qi", "<=", 10 ** -3.5, False, None, True),
    ("edge_magnetic_mirror_ratio", "<=", 0.25, False, None, False),
    ("flux_compression_in_regions_of_bad_curvature", "<=", 0.9, False, None, False),
    ("vacuum_well", ">=", 0.0, False, 0.1, False),
]


def official_violations(measured):
    import torch
    from optimize import paper_feasibility_violation
    out = []
    for name, op, value, use_abs, divisor, log_t in OFFICIAL:
        col = torch.tensor([[measured[name]]], dtype=torch.float32)
        out.append(float(paper_feasibility_violation(
            col[:, 0], op, value, use_abs=use_abs, divisor=divisor, log_transform=log_t
        ).item()))
    return np.array(out)


def pareto_front(points):
    """points: list of (L_gradB, aspect_ratio, payload). Minimizing
    (-L_gradB, aspect_ratio). Returns the non-dominated subset, sorted by
    aspect_ratio ascending."""
    pts = sorted(points, key=lambda p: (p[1], -p[0]))  # A asc, then L desc
    front, best_L = [], -np.inf
    for L, A, payload in pts:
        if L > best_L:                # strictly better L at a higher A
            front.append((L, A, payload))
            best_L = L
    return front


def hypervolume(front):
    """Exact 2D hypervolume vs REFERENCE_POINT for a non-dominated set."""
    if not front:
        return 0.0
    total, prev_A = 0.0, REFERENCE_POINT[1]
    # walk aspect_ratio ascending; each point owns the slab above it
    for L, A, _ in sorted(front, key=lambda p: p[1]):
        if A >= prev_A:
            continue
        total += (REFERENCE_POINT[0] - (-L)) * (prev_A - A)
        prev_A = A
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=str(STATE_DIR / "feasible_pool.jsonl"))
    p.add_argument("--validate", action="store_true",
                   help="re-evaluate the front members at HIGH fidelity (the scorer's own "
                        "setting) and report the honest score, not the low-fidelity upper bound")
    p.add_argument("--n-workers", type=int, default=8)
    args = p.parse_args()

    pool_path = Path(args.pool)
    if not pool_path.exists():
        print(f"[p3_front] no feasible pool at {pool_path}")
        print("[p3_front] BENCHMARK SCORE = 0.0  (no feasible design found yet)")
        return

    rows = []
    with open(pool_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        print("[p3_front] pool is empty -> BENCHMARK SCORE = 0.0")
        return

    points = [(r["L_gradB"], r["aspect_ratio"], r) for r in rows]
    front = pareto_front(points)
    hv = hypervolume(front)

    print(f"[p3_front] {len(rows)} feasible designs in pool, "
          f"{len(front)} on the Pareto front")
    print(f"[p3_front] LOW-FIDELITY hypervolume = {hv:.3f}  (upper bound; see --validate)\n")
    print(f"{'A':>8} {'L_gradB':>9} {'slab HV':>9} {'tag':>8} {'outer':>6}")
    print("-" * 46)
    prev_A = REFERENCE_POINT[1]
    for L, A, r in sorted(front, key=lambda p: p[1]):
        slab = (REFERENCE_POINT[0] + L) * (prev_A - A) if A < prev_A else 0.0
        print(f"{A:>8.3f} {L:>9.3f} {slab:>9.2f} {r.get('tag',''):>8} {r.get('outer',-1):>6}")
        prev_A = min(prev_A, A)

    if not args.validate:
        return

    print("\n[p3_front] re-evaluating front at HIGH fidelity (the scorer's own setting)...")
    from oracle_harness import run_batch_with_timeout
    fid = oracle.FIDELITY_PRESETS["high"]
    jobs, meta = [], {}
    for i, (L, A, r) in enumerate(sorted(front, key=lambda p: p[1])):
        params = np.array(r["params"], dtype=np.float64)
        r_cos, z_sin, nfp, fname = oracle.params_to_worker_args(
            params, {"nfp": r["nfp"]}, fid)
        jobs.append((i, r_cos, z_sin, nfp, fname))
        meta[i] = r

    validated = []
    for tag, ok, payload in run_batch_with_timeout(
        jobs, oracle.worker_fn, min(args.n_workers, len(jobs)) or 1, 1800.0
    ):
        r = meta[tag]
        if not ok:
            print(f"  A={r['aspect_ratio']:.3f}: FAILED at high fidelity ({str(payload)[:50]})")
            continue
        y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
        if any(v is None for v in y.values()):
            print(f"  A={r['aspect_ratio']:.3f}: incomplete metrics at high fidelity")
            continue
        v = official_violations(y)
        feas = bool((v <= FEASIBLE_TOL).all())
        print(f"  A={y['aspect_ratio']:.3f} L={y['minimum_normalized_magnetic_gradient_scale_length']:.3f} "
              f"worst_viol={v.max():+.5f} {'FEASIBLE' if feas else 'INFEASIBLE at high fidelity'}")
        if feas:
            validated.append((y["minimum_normalized_magnetic_gradient_scale_length"],
                              y["aspect_ratio"], r))

    hv_hi = hypervolume(pareto_front(validated))
    print(f"\n[p3_front] *** HIGH-FIDELITY BENCHMARK SCORE = {hv_hi:.3f} *** "
          f"({len(validated)}/{len(front)} front members survived)")


if __name__ == "__main__":
    main()

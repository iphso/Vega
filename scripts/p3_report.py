"""P3 (MHDStableQIStellarator) feasibility/objective report, mirroring
p2_report.py's structure for the real-vs-surrogate comparison, and using
the official paper tolerance/constraint values.

Note: P3's own official constraint set has no aspect_ratio bound at all --
the search scripts (diag_p2_constraints2.py) add one themselves as an
epsilon-constraint device to approximate P3's real 2D Pareto front (the
official objective/score is a hypervolume over (grad_scale_length,
aspect_ratio), not a single scalar) -- this report checks candidates
against the 5 constraints ConStellaration actually scores, no more.

vacuum_well's official threshold is 0.0, and the official benchmark divides
its violation by max(0.1, |0.0|) = 0.1 instead of |0.0| (see
optimize.py's paper_feasibility_violation docstring for why -- the naive
divide-by-threshold blows up at exactly zero). Reproduced here."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")
TOL = 1e-2
IOTA_LB, LOG10_QI_UB, MIRROR_UB, FLUX_UB, VACUUM_LB = 0.25, -3.5, 0.25, 0.9, 0.0
VACUUM_DIVISOR = max(0.1, abs(VACUUM_LB))


def p3_violations(y, target_names):
    iota = np.abs(y[:, target_names.index("edge_rotational_transform_over_n_field_periods")])
    qi = y[:, target_names.index("qi")]
    mirror = y[:, target_names.index("edge_magnetic_mirror_ratio")]
    flux = y[:, target_names.index("flux_compression_in_regions_of_bad_curvature")]
    vacuum = y[:, target_names.index("vacuum_well")]
    return np.stack([
        (IOTA_LB - iota) / abs(IOTA_LB),
        (np.log10(np.clip(qi, 1e-12, None)) - LOG10_QI_UB) / abs(LOG10_QI_UB),
        (mirror - MIRROR_UB) / abs(MIRROR_UB),
        (flux - FLUX_UB) / abs(FLUX_UB),
        (VACUUM_LB - vacuum) / VACUUM_DIVISOR,
    ], axis=1)


def main():
    paths = sys.argv[1:]
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    grad_i = target_names.index("minimum_normalized_magnetic_gradient_scale_length")
    ar_i = target_names.index("aspect_ratio")
    fidelity_name = oracle.FIDELITY_PRESETS["low"]

    jobs, meta = [], {}
    for path in paths:
        label = Path(path).stem
        for rank, line in enumerate(Path(path).read_text().splitlines()):
            d = json.loads(line)
            r_cos, z_sin = np.array(d["r_cos"]).reshape(5, 9), np.array(d["z_sin"]).reshape(5, 9)
            tag = (label, rank)
            jobs.append((tag, r_cos, z_sin, int(d["n_field_periods"]), fidelity_name))
            meta[tag] = d["predicted_targets"]

    print(f"validating {len(jobs)} candidates...")
    rows = []
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, 20, 45.0):
        if not ok:
            print(f"  {tag} FAILED: {str(payload)[:150]}")
            rows.append((*tag, False, None, None, None))
            continue
        y = np.array([payload.get(n) for n in target_names], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            rows.append((*tag, False, None, None, None))
            continue
        v = p3_violations(y[None, :], target_names)[0]
        worst = float(v.max())
        rows.append((*tag, True, worst, y[grad_i], y[ar_i]))

    n_ok = sum(1 for r in rows if r[2])
    feasible = [(r[3], r[4], r[5], r[0], r[1]) for r in rows if r[2] and r[3] <= TOL]
    print(f"{n_ok}/{len(rows)} converged, {len(feasible)}/{len(rows)} genuinely P3-feasible "
          f"(official 5-constraint set, no aspect_ratio bound)")
    if feasible:
        feasible.sort(reverse=True)  # maximize grad_scale_length -> best first
        for worst, grad, ar, label, rank in feasible[:10]:
            print(f"  {label} rank {rank}: grad_scale_length={grad:.4f} aspect_ratio={ar:.4f} "
                  f"worst_violation={worst:+.4f}")
        print(f"\nBEST (highest grad_scale_length among feasible): {feasible[0][3]} rank {feasible[0][4]}, "
              f"grad_scale_length={feasible[0][1]:.4f}, aspect_ratio={feasible[0][2]:.4f}")
    else:
        near = sorted([(r[3], r[0], r[1]) for r in rows if r[2]])[:10]
        print("closest non-feasible attempts (worst_violation ascending):")
        for worst, label, rank in near:
            print(f"  {label} rank {rank}: worst_violation={worst:+.4f}")


if __name__ == "__main__":
    main()

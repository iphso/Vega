"""P2 (SimpleToBuildQIStellarator) feasibility/objective report, mirroring
p1_report.py's structure for the real-vs-surrogate comparison, and using
the official paper tolerance/constraint values."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")
TOL = 1e-2
AR_UB, IOTA_LB, LOG10_QI_UB, MIRROR_UB, ELONG_UB = 10.0, 0.25, -4.0, 0.2, 5.0


def p2_violations(y, target_names):
    ar = y[:, target_names.index("aspect_ratio")]
    iota = np.abs(y[:, target_names.index("edge_rotational_transform_over_n_field_periods")])
    qi = y[:, target_names.index("qi")]
    mirror = y[:, target_names.index("edge_magnetic_mirror_ratio")]
    elong = y[:, target_names.index("max_elongation")]
    return np.stack([
        (ar - AR_UB) / abs(AR_UB),
        (IOTA_LB - iota) / abs(IOTA_LB),
        (np.log10(np.clip(qi, 1e-12, None)) - LOG10_QI_UB) / abs(LOG10_QI_UB),
        (mirror - MIRROR_UB) / abs(MIRROR_UB),
        (elong - ELONG_UB) / abs(ELONG_UB),
    ], axis=1)


def main():
    paths = sys.argv[1:]
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    grad_i = target_names.index("minimum_normalized_magnetic_gradient_scale_length")
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
            rows.append((*tag, False, None, None))
            continue
        y = np.array([payload.get(n) for n in target_names], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            rows.append((*tag, False, None, None))
            continue
        v = p2_violations(y[None, :], target_names)[0]
        worst = float(v.max())
        rows.append((*tag, True, worst, y[grad_i]))

    n_ok = sum(1 for r in rows if r[2])
    feasible = [(r[3], r[4], r[0], r[1]) for r in rows if r[2] and r[3] <= TOL]
    print(f"{n_ok}/{len(rows)} converged, {len(feasible)}/{len(rows)} genuinely P2-feasible")
    if feasible:
        feasible.sort(reverse=True)  # maximize grad_scale_length -> best first
        for worst, grad, label, rank in feasible[:10]:
            print(f"  {label} rank {rank}: grad_scale_length={grad:.4f} worst_violation={worst:+.4f}")
        print(f"\nBEST (highest grad_scale_length among feasible): {feasible[0][2]} rank {feasible[0][3]}, "
              f"grad_scale_length={feasible[0][1]:.4f}")
    else:
        near = sorted([(r[3], r[0], r[1]) for r in rows if r[2]])[:10]
        print("closest non-feasible attempts (worst_violation ascending):")
        for worst, label, rank in near:
            print(f"  {label} rank {rank}: worst_violation={worst:+.4f}")


if __name__ == "__main__":
    main()

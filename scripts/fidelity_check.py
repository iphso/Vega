import sys
sys.path.insert(0, "/work/scripts")
import numpy as np
import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout
from optimize import paper_feasibility_violation

params = np.load("/work/output/xbest_check.npy")[0]
r_cos = params[:45].reshape(5,9).astype(np.float64)
z_sin = params[45:90].reshape(5,9).astype(np.float64)
nfp = 2

P3_CONSTRAINTS = [
    ("edge_rotational_transform_over_n_field_periods", ">=", 0.25, True, None, False),
    ("qi", "<=", 10 ** -3.5, False, None, True),
    ("edge_magnetic_mirror_ratio", "<=", 0.25, False, None, False),
    ("flux_compression_in_regions_of_bad_curvature", "<=", 0.9, False, None, False),
    ("vacuum_well", ">=", 0.0, False, 0.1, False),
]

def real_violations(measured):
    import torch
    out = []
    for name, op, value, use_abs, divisor, log_t in P3_CONSTRAINTS:
        col = torch.tensor([[measured[name]]], dtype=torch.float32)
        v = paper_feasibility_violation(col[:, 0], op, value, use_abs=use_abs, divisor=divisor, log_transform=log_t)
        out.append(float(v.item()))
    return np.array(out)

if __name__ == "__main__":
    # two independent low_fidelity re-evals of the IDENTICAL point (determinism check)
    # plus one high_fidelity eval (does the answer change with more grid resolution?)
    jobs = [
        ("low_a", r_cos, z_sin, nfp, "low_fidelity"),
        ("low_b", r_cos, z_sin, nfp, "low_fidelity"),
        ("high", r_cos, z_sin, nfp, "high_fidelity"),
    ]
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, 3, 900.0):
        print(f"\n=== {tag} ===")
        if ok:
            v = real_violations(payload)
            print("worst_violation:", v.max())
            print("violations:", np.round(v, 5))
        else:
            print("FAILED:", payload)

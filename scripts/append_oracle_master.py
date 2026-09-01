"""Single validate+backup entry point for every candidate this exploration's
generate_candidates(_multigen).py produces -- deliberately replaces calling
p2_report.py/p3_report.py/p1_alm_validate.py separately, since those also
hit the real oracle on the same jsonl files: running both would spend twice
the oracle credits for the same candidates. This does one real-oracle pass
and both appends every row (converged or not -- provenance-tagged) to a
running master backup file for later retraining AND prints the same
feasibility/objective summary those report scripts would have.

Master file (default output/oracle_candidates_master.jsonl, --master to
change) is append-only, never overwritten: {r_cos, z_sin, n_field_periods,
is_stellarator_symmetric, source_generator, source_scorer, problem,
fidelity, converged, measured_targets (or null), predicted_targets}."""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402
from p1_report import p1_violations, TOL as P1_TOL  # noqa: E402
from p2_report import p2_violations, TOL as P2_TOL  # noqa: E402
from p3_report import p3_violations, TOL as P3_TOL  # noqa: E402

OUT_DIR = Path("/work/output")
VIOLATION_FN = {"p1": p1_violations, "p2": p2_violations, "p3": p3_violations}
TOL_BY_PROBLEM = {"p1": P1_TOL, "p2": P2_TOL, "p3": P3_TOL}
OBJ_NAME = {"p1": "max_elongation", "p2": "minimum_normalized_magnetic_gradient_scale_length",
            "p3": "minimum_normalized_magnetic_gradient_scale_length"}
OBJ_MINIMIZE = {"p1": True, "p2": False, "p3": False}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl_paths", nargs="+")
    p.add_argument("--source-generator", required=True, choices=["vae", "gan", "cvae", "diffusion"])
    p.add_argument("--source-scorer", required=True)
    p.add_argument("--problem", required=True, choices=["p1", "p2", "p3"])
    p.add_argument("--fidelity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--n-workers", type=int, default=20)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--master", default=str(OUT_DIR / "oracle_candidates_master.jsonl"))
    args = p.parse_args()

    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    jobs, meta = [], {}
    for path in args.jsonl_paths:
        label = Path(path).stem
        for rank, line in enumerate(Path(path).read_text().splitlines()):
            d = json.loads(line)
            r_cos, z_sin = np.array(d["r_cos"]).reshape(5, 9), np.array(d["z_sin"]).reshape(5, 9)
            tag = (label, rank)
            jobs.append((tag, r_cos, z_sin, int(d["n_field_periods"]), fidelity_name))
            meta[tag] = d

    print(f"[append_oracle_master] validating {len(jobs)} candidates "
          f"(generator={args.source_generator}, scorer={args.source_scorer}, problem={args.problem}, "
          f"fidelity={args.fidelity})...")
    viol_fn = VIOLATION_FN[args.problem]
    tol = TOL_BY_PROBLEM[args.problem]
    obj_name, minimize = OBJ_NAME[args.problem], OBJ_MINIMIZE[args.problem]
    obj_i = target_names.index(obj_name)

    n_converged = 0
    feasible_rows = []  # (obj_value, worst_violation, label, rank)
    near_misses = []    # (worst_violation, label, rank)
    now = datetime.datetime.utcnow().isoformat() + "Z"
    with open(args.master, "a") as f:
        for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
            d = meta[tag]
            row = {
                "timestamp": now,
                "source_generator": args.source_generator,
                "source_scorer": args.source_scorer,
                "problem": args.problem,
                "fidelity": args.fidelity,
                "label": tag[0], "rank": tag[1],
                "r_cos": d["r_cos"], "z_sin": d["z_sin"],
                "n_field_periods": d["n_field_periods"],
                "is_stellarator_symmetric": d.get("is_stellarator_symmetric", 1.0),
                "predicted_targets": d.get("predicted_targets"),
                "converged": bool(ok),
                "measured_targets": None,
                "worst_violation": None,
                "error": None,
            }
            if ok:
                y = {n: payload.get(n) for n in target_names}
                if all(v is not None and np.isfinite(v) for v in y.values()):
                    row["measured_targets"] = y
                    n_converged += 1
                    y_arr = np.array([[y[n] for n in target_names]])
                    worst = float(viol_fn(y_arr, target_names)[0].max())
                    row["worst_violation"] = worst
                    near_misses.append((worst, tag[0], tag[1]))
                    if worst <= tol:
                        obj_val = y[obj_name]
                        feasible_rows.append((obj_val, worst, tag[0], tag[1]))
                else:
                    row["converged"] = False
                    row["error"] = "non-finite metric"
            else:
                row["error"] = str(payload)[:300]
            f.write(json.dumps(row) + "\n")

    print(f"[append_oracle_master] {n_converged}/{len(jobs)} converged, "
          f"{len(feasible_rows)}/{len(jobs)} genuinely {args.problem.upper()}-feasible "
          f"(appended {len(jobs)} rows total to {args.master})")
    if feasible_rows:
        feasible_rows.sort(reverse=not minimize)
        for obj_val, worst, label, rank in feasible_rows[:10]:
            print(f"  FEASIBLE {label} rank {rank}: {obj_name}={obj_val:.5g} worst_violation={worst:+.4f}")
        best = feasible_rows[0]
        print(f"  BEST: {best[2]} rank {best[3]}, {obj_name}={best[0]:.5g}")
    elif near_misses:
        near_misses.sort()
        print("  closest non-feasible attempts (worst_violation ascending):")
        for worst, label, rank in near_misses[:10]:
            print(f"    {label} rank {rank}: worst_violation={worst:+.4f}")


if __name__ == "__main__":
    main()

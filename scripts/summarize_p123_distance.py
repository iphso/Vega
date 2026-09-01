"""Summarize one or more run_p123_battery.sh labels' real-oracle results:
for each (label, problem), report the best REAL feasible objective if any
converged candidate actually cleared every constraint, and otherwise the
closest real candidate by the same normalized worst_violation used to
judge feasibility -- a genuine "how far off" distance, not just "did it
work". P1's own objective (max_elongation) is reported for the P1 row;
for P2/P3 (never feasible so far) the report is centered on real
measured qi, since qi is the established blocking constraint, alongside
the overall worst_violation for full-picture context.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from p1_report import p1_violations, TOL as P1_TOL  # noqa: E402
from p2_report import p2_violations, TOL as P2_TOL  # noqa: E402
from p3_report import p3_violations, TOL as P3_TOL  # noqa: E402

OUT_DIR = Path("/work/output")
VIOLATION_FN = {"p1": p1_violations, "p2": p2_violations, "p3": p3_violations}
TOL = {"p1": P1_TOL, "p2": P2_TOL, "p3": P3_TOL}
OBJ_NAME = {"p1": "max_elongation", "p2": "minimum_normalized_magnetic_gradient_scale_length",
            "p3": "minimum_normalized_magnetic_gradient_scale_length"}
OBJ_MINIMIZE = {"p1": True, "p2": False, "p3": False}


def main():
    labels = sys.argv[1:]
    if not labels:
        print("usage: summarize_p123_distance.py <label1> [label2 ...]")
        sys.exit(1)

    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    rows_by_key = {}  # (label, problem) -> list of measured_targets dicts
    with open(OUT_DIR / "oracle_candidates_master.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("source_scorer") not in labels or not d.get("converged"):
                continue
            key = (d["source_scorer"], d["problem"])
            rows_by_key.setdefault(key, []).append(d["measured_targets"])

    header = f"{'label':40s} {'problem':5s} {'n_conv':>7s} {'n_feas':>7s} {'worst_violation':>16s} {'qi (real)':>12s} {'obj (real)':>14s}"
    print(header)
    print("-" * len(header))
    for label in labels:
        for problem in ("p1", "p2", "p3"):
            rows = rows_by_key.get((label, problem))
            if not rows:
                print(f"{label:40s} {problem:5s} {'--':>7s}")
                continue
            import numpy as np
            y = np.array([[r[n] for n in target_names] for r in rows])
            viol = VIOLATION_FN[problem](y, target_names).max(axis=1)
            best_i = viol.argmin()
            n_feas = int((viol <= TOL[problem]).sum())
            obj_i = target_names.index(OBJ_NAME[problem])
            if n_feas > 0:
                feas_mask = viol <= TOL[problem]
                obj_vals = y[feas_mask, obj_i]
                best_obj = obj_vals.min() if OBJ_MINIMIZE[problem] else obj_vals.max()
            else:
                best_obj = y[best_i, obj_i]
            qi_i = target_names.index("qi")
            print(f"{label:40s} {problem:5s} {len(rows):7d} {n_feas:7d} "
                  f"{viol[best_i]:+16.4f} {y[best_i, qi_i]:12.5g} {best_obj:14.5g}")


if __name__ == "__main__":
    main()

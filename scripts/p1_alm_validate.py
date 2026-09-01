"""Validates every candidate across a set of generate_candidates.py JSONL
outputs against the REAL oracle (not the surrogate's own prediction) --
answers three things at once: (1) does the ALM search find a feasible
design consistently across seeds, not just once, (2) how many of the
near-top candidates (not just rank 1) are also really feasible, and
(3) among everything that's REALLY feasible, what's the best (lowest)
max_elongation -- P1's actual objective, not just the binary feasibility
question this session mostly tracked so far.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout
from p1_report import p1_violations, TOL, p1_score

OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl_paths", nargs="+")
    p.add_argument("--fidelity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--n-workers", type=int, default=10)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    args = p.parse_args()

    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    elong_i = target_names.index("max_elongation")
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    jobs, meta = [], {}
    for path in args.jsonl_paths:
        seed_label = Path(path).stem
        for rank, line in enumerate(Path(path).read_text().splitlines()):
            d = json.loads(line)
            r_cos = np.array(d["r_cos"]).reshape(5, 9)
            z_sin = np.array(d["z_sin"]).reshape(5, 9)
            tag = (seed_label, rank)
            jobs.append((tag, r_cos, z_sin, int(d["n_field_periods"]), fidelity_name))
            meta[tag] = d["predicted_targets"]

    print(f"validating {len(jobs)} candidates from {len(args.jsonl_paths)} runs at fidelity={args.fidelity} "
          f"({fidelity_name})...")
    results = {}
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
        results[tag] = (ok, payload)

    rows = []
    for tag, (ok, payload) in results.items():
        seed_label, rank = tag
        if not ok:
            rows.append((seed_label, rank, False, None, None, None, None, None))
            continue
        y = np.array([payload.get(n) for n in target_names], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            rows.append((seed_label, rank, False, None, None, None, None, None))
            continue
        v = p1_violations(y[None, :], target_names)[0]
        worst = float(v.max())
        feasible = worst <= TOL
        rows.append((seed_label, rank, True, feasible, worst, y[elong_i],
                      meta[tag].get("max_elongation"), y))

    print(f"\n{'seed':10s} {'rank':4s} {'converged':9s} {'feasible':8s} {'worst_v':8s} "
          f"{'elong(measured)':15s} {'elong(predicted)':16s}")
    feasible_rows = []
    per_seed_feasible = {}
    for seed_label, rank, ok, feasible, worst, elong, pred_elong, y in sorted(rows):
        if not ok:
            print(f"{seed_label:10s} {rank:<4d} {'NO':9s}")
            continue
        print(f"{seed_label:10s} {rank:<4d} {'yes':9s} {str(feasible):8s} {worst:8.4f} "
              f"{elong:15.4f} {pred_elong:16.4f}")
        per_seed_feasible.setdefault(seed_label, 0)
        if feasible:
            per_seed_feasible[seed_label] += 1
            feasible_rows.append((elong, seed_label, rank, y))

    n_converged = sum(1 for r in rows if r[2])
    n_feasible = len(feasible_rows)
    print(f"\n{n_converged}/{len(rows)} converged in VMEC++, {n_feasible}/{len(rows)} genuinely P1-feasible")
    print(f"seeds with >=1 feasible candidate: {sum(1 for v in per_seed_feasible.values() if v > 0)}/"
          f"{len({r[0] for r in rows})}")

    if feasible_rows:
        feasible_rows.sort()
        best_elong, best_seed, best_rank, best_y = feasible_rows[0]
        print(f"\nBEST (lowest measured max_elongation among genuinely feasible): "
              f"{best_seed} rank {best_rank}, max_elongation={best_elong:.4f}, "
              f"P1 score={p1_score(best_elong):.4f}")
        for name, val in zip(target_names, best_y):
            print(f"  {name}: {val:.5f}")


if __name__ == "__main__":
    main()

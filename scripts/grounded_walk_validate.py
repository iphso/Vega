"""Oracle half of the grounded-walk round-trip (see grounded_walk_step.py for
the GPU half and its module docstring for the overall design). Runs in the
oracle-x86 service (has constellaration+vmecpp, no GPU/torch) since
grounded_walk_step.py's container (has GPU+torch, no constellaration) can't
do this part itself -- state round-trips between the two via files under
output/grounded_walk_<tag>/, driven by run_grounded_walk.sh.

Reads this round's proposed_candidates.jsonl (every fiber's proposed step,
not yet accepted), real-validates every one of them through VMEC++, and
writes validation_results.jsonl with each fiber's real convergence + measured
targets (or null) -- grounded_walk_step.py's next invocation is what actually
decides accept-vs-rollback from this file.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--fidelity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    args = p.parse_args()

    state_dir = OUT_DIR / f"grounded_walk_{args.tag}"
    candidates_path = state_dir / "proposed_candidates.jsonl"
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    rows = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    jobs, meta = [], {}
    for d in rows:
        tag = d["fiber"]
        # worker_fn requires (n_poloidal_modes, n_toroidal_modes)=(5,9) arrays, not
        # the flat 45-element lists JSONL stores them as -- append_oracle_master.py
        # already does this same reshape on read; grounded_walk_step.py writes flat
        # lists (matching generate_candidates.py's own JSONL convention) so this
        # side has to reshape too. Missing this made every validation in the first
        # real run fail with a structural/shape error, not a real physics failure.
        r_cos = np.array(d["r_cos"]).reshape(5, 9)
        z_sin = np.array(d["z_sin"]).reshape(5, 9)
        jobs.append((tag, r_cos, z_sin, d["n_field_periods"], fidelity_name))
        meta[tag] = d

    n_converged = 0
    results = []
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
        d = meta[tag]
        row = {"fiber": tag, "r_cos": d["r_cos"], "z_sin": d["z_sin"], "converged": False, "measured_targets": None}
        if ok:
            y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if all(v is not None for v in y.values()):
                row["converged"] = True
                row["measured_targets"] = y
                n_converged += 1
        results.append(row)

    results.sort(key=lambda r: r["fiber"])
    with open(state_dir / "validation_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[grounded_walk_validate] {n_converged}/{len(jobs)} converged this round")


if __name__ == "__main__":
    main()

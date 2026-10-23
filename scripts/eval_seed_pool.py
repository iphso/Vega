"""Evaluate a seed pool at low fidelity and print A / L / official violations.

Needed to design the aspect-floor replication honestly: an aspect FLOOR only does
work on a design whose aspect ratio is below it. Applying --aspect-min 13.5 to a
seed already at A=14.96 tests nothing, because the sixth constraint is slack at
iteration 0. This prints the pool so replication seeds can be chosen on the
criterion that the lever is actually engaged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from minimax_p3 import official_violations  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

NAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="/work/output/diverse_seeds.npy")
    p.add_argument("--nfp", type=int, default=0,
                   help="0 = try to read nfp from the pool's 92nd column if present")
    p.add_argument("--workers", type=int, default=12)
    # A seed is only usable for a high-fidelity run if it CONVERGES at high
    # fidelity -- the project's own leading designs do not (2026-10-05), so this
    # is now a prerequisite for seed selection and not a diagnostic.
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--out", default="/work/output/seed_pool_eval.jsonl")
    a = p.parse_args()

    pool = np.load(a.pool)
    print(f"[pool] {a.pool} shape={pool.shape} fidelity={a.fidelity}", flush=True)

    # Rows are 92 wide: 90 boundary coefficients + 2 trailing columns. The oracle
    # reads only the first 90; nfp has to come from somewhere, so probe the tail.
    jobs, nfps, params = [], {}, {}
    for i, row in enumerate(pool):
        nfp = a.nfp if a.nfp else int(round(row[90])) if pool.shape[1] > 90 else 3
        if not 1 <= nfp <= 6:
            nfp = 3
        x = np.asarray(row[:90], dtype=float).copy()
        x[oracle.ZERO_INDICES] = 0.0   # the oracle requires zero-enforced params
        rc, zsin, nfp_i, fn = oracle.params_to_worker_args(
            x, {"nfp": nfp}, oracle.FIDELITY_PRESETS[a.fidelity])
        jobs.append((i, rc, zsin, nfp_i, fn))
        nfps[i], params[i] = nfp, x

    rows = []
    for i, ok, payload in run_batch_with_timeout(
            jobs, oracle.worker_fn, a.workers, 1800):
        nfp = nfps[i]
        mt = {n: payload.get(n) for n in oracle.TARGET_NAMES} if ok else None
        if not ok or any(v is None for v in mt.values()):
            why = payload if not ok else "missing metrics"
            print(f"  [{i:2d}] nfp={nfp} NOT CONVERGED ({str(why)[:60]})", flush=True)
            rows.append({"row": i, "nfp": nfp, "converged": False})
            continue
        v = official_violations(mt)
        rows.append({"row": i, "nfp": nfp, "converged": True,
                     "A": mt["aspect_ratio"],
                     "L": mt["minimum_normalized_magnetic_gradient_scale_length"],
                     "worst": float(v.max()),
                     "binding": NAMES[int(np.argmax(v))],
                     "violations": [float(x) for x in v]})
        print(f"  [{i:2d}] nfp={nfp} A={mt['aspect_ratio']:7.4f} "
              f"L={mt['minimum_normalized_magnetic_gradient_scale_length']:6.3f} "
              f"worst={float(v.max()):8.5f} binding={NAMES[int(np.argmax(v))]}",
              flush=True)

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in rows if r.get("converged")]
    below = sorted((r for r in ok if r["A"] < 13.5), key=lambda r: r["worst"])
    print(f"\n[pool] {len(ok)}/{len(rows)} converged; "
          f"{len(below)} with A < 13.5 (floor would bind)", flush=True)
    for r in below[:8]:
        print(f"  row {r['row']:2d}  nfp={r['nfp']}  A={r['A']:7.4f}  "
              f"worst={r['worst']:.5f}  binding={r['binding']}", flush=True)


if __name__ == "__main__":
    main()

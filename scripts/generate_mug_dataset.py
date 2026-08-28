"""Bootstraps a training dataset for the mug/thermos thermal-design domain --
v3 (EXPERIMENT_LOG §50), sampling the 14-dim parameterization (variable
body shape + lid, on top of v2's materials/wall-profile/handle) instead of
v2's 8-dim one. Same role as generate_airfoil_dataset.py/
generate_torax_dataset.py; direct range sampling is still sufficient here
(no archetypal seeds needed) -- confirmed by v3's own 40/40 smoke test.

Runs entirely on host, no Docker -- mug_oracle.py's only dependency is
numpy/scipy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import mug_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Radius ranges: 15-60mm spans a small espresso cup (30mm dia) to a large
# tumbler (120mm dia) -- r_mid free to be smaller (taper), larger (belly), or
# equal to base/rim (straight cylinder) within that same range.
DEFAULT_RANGES = dict(
    r_base_mm=(15.0, 60.0),
    r_mid_mm=(15.0, 60.0),
    r_rim_mm=(15.0, 60.0),
    t_wall_rim_mm=(0.3, 5.0),
    t_wall_base_mm=(0.3, 5.0),
    struct_material_idx=(0.0, 3.0),
    t_gap_mm=(0.05, 25.0),
    insulation_material_idx=(0.0, 3.0),
    handle_length_mm=(10.0, 80.0),
    handle_diameter_mm=(3.0, 20.0),
    handle_material_idx=(0.0, 4.0),
    lid_coverage_frac=(0.0, 1.0),
    t_lid_mm=(0.5, 10.0),
    lid_material_idx=(0.0, 3.0),
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=20_000)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=280)
    p.add_argument("--timeout-seconds", type=float, default=15.0)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="mug")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    accepted_X, accepted_Y = [], []
    n_attempted, n_accepted = 0, 0
    t_start = time.perf_counter()

    def sample_batch(n):
        candidates = []  # (tag, *worker_args, params)
        for i in range(n):
            params = np.array([rng.uniform(*DEFAULT_RANGES[name]) for name in oracle.PARAM_NAMES], dtype=np.float64)
            candidates.append((i, *oracle.params_to_worker_args(params, {}, "low"), params))
        return candidates

    while n_accepted < args.target_count:
        raw = sample_batch(args.batch_size)
        jobs = [c[:-1] for c in raw]
        lookup = {c[0]: c[-1] for c in raw}

        n_attempted += len(jobs)
        for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
            if ok:
                params = lookup[tag]
                row_x = params.astype(np.float32)
                row_y = np.array([payload[name] for name in oracle.TARGET_NAMES], dtype=np.float32)
                if np.all(np.isfinite(row_y)):
                    accepted_X.append(row_x)
                    accepted_Y.append(row_y)
                    n_accepted += 1

        elapsed = time.perf_counter() - t_start
        print(f"[{args.out_tag}] attempted={n_attempted} accepted={n_accepted}/{args.target_count} "
              f"(hit rate {n_accepted / max(n_attempted, 1):.1%})  elapsed={elapsed:.0f}s")

        if len(accepted_X) >= args.checkpoint_every or n_accepted >= args.target_count:
            X = np.stack(accepted_X) if accepted_X else np.zeros((0, oracle.PARAM_DIM), dtype=np.float32)
            Y = np.stack(accepted_Y) if accepted_Y else np.zeros((0, len(oracle.TARGET_NAMES)), dtype=np.float32)
            X_path, Y_path = OUT_DIR / f"{args.out_tag}_X.npy", OUT_DIR / f"{args.out_tag}_Y.npy"
            if X_path.exists():
                X = np.concatenate([np.load(X_path), X])
                Y = np.concatenate([np.load(Y_path), Y])
            np.save(X_path, X)
            np.save(Y_path, Y)
            accepted_X.clear()
            accepted_Y.clear()
            print(f"[{args.out_tag}] checkpointed -> {X_path} ({len(X)} rows total)")

    (OUT_DIR / f"{args.out_tag}_target_names.json").write_text(json.dumps(oracle.TARGET_NAMES, indent=2))
    (OUT_DIR / f"{args.out_tag}_feature_names.json").write_text(json.dumps(oracle.PARAM_NAMES, indent=2))
    stats = {"n_accepted": n_accepted, "n_attempted": n_attempted, "hit_rate": n_accepted / max(n_attempted, 1),
              "elapsed_seconds": time.perf_counter() - t_start, "seed": args.seed}
    (OUT_DIR / f"{args.out_tag}_generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\ndone. {n_accepted} rows in {stats['elapsed_seconds']:.0f}s ({stats['hit_rate']:.1%} hit rate)")


if __name__ == "__main__":
    main()

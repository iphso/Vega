"""Bootstraps a training dataset for the mug/thermos thermal-design domain --
v6 (EXPERIMENT_LOG §59), sampling the 9-dim single-wall/lidless
parameterization. Same role as generate_airfoil_dataset.py/
generate_torax_dataset.py; direct range sampling is still sufficient here
(v3's own 40/40 smoke test already confirmed this style of domain doesn't
need archetypal seeds, and v6 didn't change that).

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
#
# Wall thickness (§59): NOT a copy of v3-v5's 0.3-5mm range -- that was
# calibrated for a thin structural shell backed by a separate real
# insulation layer. With no insulation layer at all now, the wall alone
# carries both the structural load AND all of the thermal resistance, so
# it needs a realistic single-wall range instead: 2-10mm spans a thin
# plastic cup to a thick stoneware mug.
DEFAULT_RANGES = dict(
    r_base_mm=(15.0, 60.0),
    r_mid_mm=(15.0, 60.0),
    r_rim_mm=(15.0, 60.0),
    t_wall_rim_mm=(2.0, 10.0),
    t_wall_base_mm=(2.0, 10.0),
    # §62: bounds are len(table)-1, not a fixed 3/4 -- STRUCTURAL_MATERIALS/
    # HANDLE_MATERIALS grew from 4/5 generic-category entries to 20/15 real
    # named materials (thermal_mug_spike.py), so the valid continuous-index
    # range widened along with them.
    struct_material_idx=(0.0, float(len(oracle.STRUCTURAL_MATERIALS) - 1)),
    handle_length_mm=(10.0, 80.0),
    handle_diameter_mm=(3.0, 20.0),
    handle_material_idx=(0.0, float(len(oracle.HANDLE_MATERIALS) - 1)),
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

    # §58: this script's own checkpointing APPENDS onto an existing
    # {out_tag}_X.npy/Y.npy rather than overwriting it (see below) -- a
    # real footgun caught directly, not theoretical: a v4 run interrupted
    # partway through left partial rows on disk, and a subsequent fresh v5
    # run silently merged onto them (plus the v3 rows already there before
    # that), producing a physics-inconsistent 50,240-row file before this
    # was noticed and the file deleted and regenerated clean. Loud, not
    # silent, is the fix -- append is still occasionally the right call
    # (e.g. deliberately topping up an existing pool), so this warns
    # rather than refuses.
    existing_X_path = OUT_DIR / f"{args.out_tag}_X.npy"
    if existing_X_path.exists():
        n_existing = np.load(existing_X_path).shape[0]
        print(f"[{args.out_tag}] WARNING: {existing_X_path} already has {n_existing} rows -- "
              f"this run will APPEND onto them, not replace them. If those rows are from a "
              f"different physics version (an interrupted prior run, an older simulate_vN), "
              f"delete {args.out_tag}_X.npy/{args.out_tag}_Y.npy first.")

    rng = np.random.default_rng(args.seed)

    accepted_X, accepted_Y = [], []
    n_attempted, n_accepted = 0, 0
    t_start = time.perf_counter()

    # §63: struct_material_idx/handle_material_idx are drawn as true
    # integer classes, not `rng.uniform` like every other (genuinely
    # continuous) column -- a material is a categorical choice (one named
    # entry from STRUCTURAL_MATERIALS/HANDLE_MATERIALS), not a point on a
    # spectrum, and a fractional label here would be exactly the fictional
    # "70% titanium / 30% steel" blend this whole change exists to remove.
    # thermal_mug_spike.py's simulate_v6() itself still accepts a
    # continuous index (material_props() still interpolates) -- that's
    # fine, it's a smooth physical proxy useful for e.g. a one-off
    # sensitivity sweep -- but the real DATASET, the thing every scoring/
    # generative model trains against as ground truth, must only ever
    # contain designs that correspond to an actual buildable material.
    MATERIAL_COLUMNS = {"struct_material_idx", "handle_material_idx"}

    def sample_batch(n):
        candidates = []  # (tag, *worker_args, params)
        for i in range(n):
            params = np.array([
                float(rng.integers(0, round(DEFAULT_RANGES[name][1]) + 1)) if name in MATERIAL_COLUMNS
                else rng.uniform(*DEFAULT_RANGES[name])
                for name in oracle.PARAM_NAMES
            ], dtype=np.float64)
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

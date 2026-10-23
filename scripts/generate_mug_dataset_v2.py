"""Mug domain v2, Phase A2 (EXPERIMENT_LOG §70-76): the same accept/reject
batch-sampling loop generate_mug_dataset.py already uses for the old v1
revolve oracle, now pointed at mug_openfoam_oracle_v2.py's free-form-
deformed body. Confirms directly -- not just claims -- that this whole
redesign's own stated goal works: real, non-trivial rejection, not the old
domain's structural 100% acceptance rate.

Must run inside the `openfoam` service (mug_openfoam_oracle_v2.py needs
blockMesh/surfaceFeatures/snappyHexMesh/chtMultiRegionFoam/surfaceCheck on
PATH), unlike generate_mug_dataset.py's own host-only v1 counterpart.
`run_batch_with_timeout`'s multiprocessing workers all run inside this ONE
container invocation -- no per-candidate container spin-up needed, the
container's own core count (confirmed 32 via `nproc`, §75) is what
`--n-workers` parallelizes across.

Per-candidate cost is real (§75: ~207s worst-case single-core for a
thin-wall accepted design; rejected designs are far cheaper, most caught
by the surfaceCheck self-intersection gate before any OpenFOAM invocation
at all) -- `--timeout-seconds` defaults well above that, and this script's
own default `--target-count` is deliberately modest (a real validation
run, not a production dataset -- Phase A2's own stated goal is confirming
the hit rate is real and non-trivial, not seeding training data yet).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import mug_openfoam_oracle_v2 as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Deliberately INDEPENDENT, overlapping ranges for r_inner_mm/r_outer_mm --
# the whole point of this redesign (§70) is that outer is no longer
# guaranteed >= inner by construction, so sampling them from the same
# range (rather than e.g. outer=inner+uniform(2,10)) is what actually lets
# the sampler produce (and the oracle correctly reject) a genuinely
# infeasible body, not just a body a hand-picked range happens to avoid.
DEFAULT_RANGES = dict(
    r_inner_mm=(15.0, 60.0),
    r_outer_mm=(15.0, 65.0),
    height_mm=(60.0, 150.0),
    struct_material_idx=(0.0, float(len(oracle.tms.STRUCTURAL_MATERIALS) - 1)),
    coating_material_idx=(0.0, float(len(oracle.tms.COATING_MATERIALS) - 1)),
    coating_thickness_mm=(0.0, 0.5),
    # §77: radial_offset straddles the nominal r_outer widely enough
    # (matching r_inner/r_outer's own overlapping-range philosophy above)
    # that a real fraction of sampled handles genuinely float off the
    # body's actual (possibly asymmetric) surface, not just theoretically.
    handle_theta1=(0.0, 2 * np.pi),
    handle_z1_frac=(0.1, 0.9),
    handle_radial_offset1_mm=(-25.0, 25.0),
    handle_theta2=(0.0, 2 * np.pi),
    handle_z2_frac=(0.1, 0.9),
    handle_radial_offset2_mm=(-25.0, 25.0),
    handle_bulge_mm=(-10.0, 40.0),
    handle_radius_mm=(2.0, 8.0),
    handle_material_idx=(0.0, float(len(oracle.tms.HANDLE_MATERIALS) - 1)),
    joint_material_idx=(0.0, float(len(oracle.tms.JOINT_MATERIALS) - 1)),
)
# §71's own bulge test used a single +-15mm displacement and already found
# local thickness swinging from 1.3mm to 9.3mm around the ring -- +-5mm
# per component, independently on all 72 (36 inner + 36 outer) lattice
# displacement components at once, is a genuinely wide, unbiased
# exploration of the lattice's real expressive range, not narrowed to
# avoid producing infeasible shapes (that narrowing would defeat this
# script's own purpose).
DISP_RANGE_MM = (-5.0, 5.0)

MATERIAL_COLUMNS = {"struct_material_idx", "coating_material_idx", "handle_material_idx", "joint_material_idx"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=50)
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--timeout-seconds", type=float, default=400.0)
    p.add_argument("--checkpoint-every", type=int, default=10_000)  # effectively "only at the end" for this validation scale
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="mug_v2")
    p.add_argument("--min-classifier-labels", type=int, default=80,
                    help="§91: worker-dispatched outcomes needed before the feasibility classifier trains at "
                         "all -- below this, sample_batch falls back to plain FIFO (identical to §85's own "
                         "behavior). 80 is a real floor for a GradientBoostingClassifier over 88 features to "
                         "say anything meaningful, not a tuned value.")
    p.add_argument("--oversample-factor", type=int, default=8,
                    help="§91: once trained, draw this many TIMES the batch size worth of fast-filter-passing "
                         "candidates, score them all, dispatch only the top batch_size by predicted worker-"
                         "stage success probability -- never skips the real check_feasibility_fast gate, only "
                         "reorders which fast-filter survivors get the (expensive) worker slot first.")
    args = p.parse_args()

    existing_X_path = OUT_DIR / f"{args.out_tag}_X.npy"
    if existing_X_path.exists():
        n_existing = np.load(existing_X_path).shape[0]
        print(f"[{args.out_tag}] WARNING: {existing_X_path} already has {n_existing} rows -- "
              f"this run will APPEND onto them, not replace them.")

    rng = np.random.default_rng(args.seed)

    accepted_X, accepted_Y = [], []
    n_attempted, n_accepted = 0, 0
    n_fast_rejected = 0
    fast_reject_reasons = {}
    reject_reasons = {}
    t_start = time.perf_counter()

    # §91 (direct user steer: "can we use the bootstrap and search approach
    # we've got... even just having the valid/invalid data can be good
    # enough for the generative techniques to start working"): the fast
    # filter (§85) already showed a cheap classifier trained on free
    # (params -> valid/invalid) labels enriches proposals dramatically
    # (measured: top-2% by classifier score hit 25% real-valid vs a 1.87%
    # base rate). But the ACTUAL wall-clock bottleneck here isn't the free
    # fast-filter stage (~2ms/draw) -- it's the fixed 32-worker OpenFOAM
    # pool, each job costing a real ~365s regardless of outcome (measured
    # directly from this run's own first hour: 384 worker-dispatched jobs
    # in 4375s at 32-way parallelism). So the lever that actually moves
    # wall-clock throughput is raising the fraction of WORKER-DISPATCHED
    # jobs that succeed (self-intersection + meshing + solve, currently
    # ~24%), not the fraction of raw draws that clear the free gate.
    #
    # worker_log_X/y accumulate (params, 1-if-accepted-else-0) for EVERY
    # worker-dispatched candidate this run has ever seen (real oracle
    # signal, the only kind that can label this outcome -- unlike the fast
    # filter, this can't be regenerated for free on demand). A fresh
    # GradientBoostingClassifier retrains on the whole accumulated log
    # after every batch once there are enough labels to be worth it --
    # cheap (well under a second at this scale) next to a single 365s
    # worker job. `sample_batch` below uses it to OVERSAMPLE: draw several
    # batches' worth of fast-filter survivors, score them, dispatch only
    # the top batch_size by predicted worker-stage success -- the real
    # check_feasibility_fast gate still runs on every one of them
    # unchanged, so this can only reorder which survivors get a worker
    # slot first, never let something invalid through.
    worker_log_X, worker_log_y = [], []
    clf = None
    clf_path_X = OUT_DIR / f"{args.out_tag}_worker_log_X.npy"
    clf_path_y = OUT_DIR / f"{args.out_tag}_worker_log_y.npy"
    if clf_path_X.exists():
        worker_log_X = list(np.load(clf_path_X))
        worker_log_y = list(np.load(clf_path_y))
        print(f"[{args.out_tag}] loaded {len(worker_log_X)} prior worker-outcome labels for the feasibility classifier")

    def maybe_retrain_classifier():
        nonlocal clf
        if len(worker_log_X) < args.min_classifier_labels:
            return
        y_arr = np.array(worker_log_y)
        if len(set(y_arr.tolist())) < 2:
            return  # can't train on a single class yet (e.g. all failures so far)
        new_clf = GradientBoostingClassifier(n_estimators=80, max_depth=3)
        new_clf.fit(np.array(worker_log_X), y_arr)
        clf = new_clf

    def sample_one():
        vals = []
        for name in oracle.PARAM_NAMES:
            if name in DEFAULT_RANGES:
                if name in MATERIAL_COLUMNS:
                    vals.append(float(rng.integers(0, round(DEFAULT_RANGES[name][1]) + 1)))
                else:
                    vals.append(rng.uniform(*DEFAULT_RANGES[name]))
            else:
                # every disp_inner_mm_*/disp_outer_mm_* column
                vals.append(rng.uniform(*DISP_RANGE_MM))
        return np.array(vals, dtype=np.float64)

    def fast_filter_survivors(n):
        """§85's own draw-until-n-pass loop, factored out so §91 can call
        it for a bigger oversampled pool without duplicating the logic."""
        nonlocal n_fast_rejected
        survivors = []
        while len(survivors) < n:
            params = sample_one()
            fast = oracle.check_feasibility_fast(params)
            if not fast["valid"]:
                n_fast_rejected += 1
                fast_reject_reasons[fast["reason"]] = fast_reject_reasons.get(fast["reason"], 0) + 1
                continue
            survivors.append(params)
        return survivors

    def sample_batch(n):
        """§85: pre-filters through `oracle.check_feasibility_fast()` --
        pure numpy/scipy, no Docker/OpenFOAM subprocess -- on THIS process
        before ever handing a candidate to the multiprocessing worker
        pool. Every candidate that reaches `jobs` below is one that's
        already cleared the cheap geometry/attach-point/volume gates, so
        the worker pool only ever spends real subprocess/Docker time on
        candidates worth spending it on (the remaining self-intersection
        `surfaceCheck` gate, then meshing/solving).

        §91: once the worker-outcome classifier exists, draws
        `oversample_factor * n` fast-filter survivors instead of just `n`,
        scores them all, and keeps only the top `n` by predicted
        worker-stage success probability -- the free fast-filter draws
        needed for this are negligible (§85: ~2ms each) next to what a
        single worker slot costs, so oversampling costs nothing that
        matters while raising the fraction of dispatched jobs that
        actually succeed."""
        if clf is None:
            survivors = fast_filter_survivors(n)
        else:
            pool = fast_filter_survivors(min(n * args.oversample_factor, n * 64))
            scores = clf.predict_proba(np.array(pool))[:, 1]
            order = np.argsort(-scores)[:n]
            survivors = [pool[i] for i in order]
        return [(i, *oracle.params_to_worker_args(params, {}, "low"), params) for i, params in enumerate(survivors)]

    while n_accepted < args.target_count:
        raw = sample_batch(args.batch_size)
        jobs = [c[:-1] for c in raw]
        lookup = {c[0]: c[-1] for c in raw}

        n_attempted += len(jobs)
        for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
            params = lookup[tag]
            # §91: every worker-dispatched job labels the feasibility
            # classifier's training set, accepted or not -- a real
            # oracle-verified outcome is exactly the signal a fast-filter
            # pass/fail can't provide.
            worker_log_X.append(params.astype(np.float32))
            worker_log_y.append(1 if ok else 0)
            if ok:
                row_x = params.astype(np.float32)
                row_y = np.array([payload[name] for name in oracle.TARGET_NAMES], dtype=np.float32)
                if np.all(np.isfinite(row_y)):
                    accepted_X.append(row_x)
                    accepted_Y.append(row_y)
                    n_accepted += 1
            else:
                reject_reasons[str(payload)] = reject_reasons.get(str(payload), 0) + 1

        maybe_retrain_classifier()

        elapsed = time.perf_counter() - t_start
        n_total_draws = n_attempted + n_fast_rejected
        clf_status = f"clf={'active' if clf is not None else 'warming up'} (n_labels={len(worker_log_X)})"
        print(f"[{args.out_tag}] accepted={n_accepted}/{args.target_count}  "
              f"worker_dispatched={n_attempted} (post-fast-filter hit rate {n_accepted / max(n_attempted, 1):.1%})  "
              f"fast_rejected={n_fast_rejected}  overall hit rate {n_accepted / max(n_total_draws, 1):.1%}  "
              f"{clf_status}  elapsed={elapsed:.0f}s")

        if len(accepted_X) >= args.checkpoint_every or n_accepted >= args.target_count:
            X = np.stack(accepted_X) if accepted_X else np.zeros((0, oracle.PARAM_DIM), dtype=np.float32)
            Y = np.stack(accepted_Y) if accepted_Y else np.zeros((0, len(oracle.TARGET_NAMES)), dtype=np.float32)
            X_path, Y_path = OUT_DIR / f"{args.out_tag}_X.npy", OUT_DIR / f"{args.out_tag}_Y.npy"
            if X_path.exists():
                X = np.concatenate([np.load(X_path), X])
                Y = np.concatenate([np.load(Y_path), Y])
            np.save(X_path, X)
            np.save(Y_path, Y)
            # §91: persist the worker-outcome log too, same append-don't-
            # replace convention -- this is what makes the classifier's
            # training data survive a restart instead of resetting to
            # cold-start every time (§89's own restart lost nothing
            # equivalent because it didn't exist yet).
            np.save(clf_path_X, np.array(worker_log_X, dtype=np.float32))
            np.save(clf_path_y, np.array(worker_log_y, dtype=np.int64))
            accepted_X.clear()
            accepted_Y.clear()
            print(f"[{args.out_tag}] checkpointed -> {X_path} ({len(X)} rows total)")

    (OUT_DIR / f"{args.out_tag}_target_names.json").write_text(json.dumps(oracle.TARGET_NAMES, indent=2))
    (OUT_DIR / f"{args.out_tag}_feature_names.json").write_text(json.dumps(oracle.PARAM_NAMES, indent=2))
    stats = {
        "n_accepted": n_accepted,
        "n_worker_dispatched": n_attempted,
        "n_fast_rejected": n_fast_rejected,
        "post_fast_filter_hit_rate": n_accepted / max(n_attempted, 1),
        "overall_hit_rate": n_accepted / max(n_attempted + n_fast_rejected, 1),
        "n_worker_outcome_labels": len(worker_log_X),
        "classifier_active": clf is not None,
        "reject_reasons": reject_reasons,
        "fast_reject_reasons": fast_reject_reasons,
        "elapsed_s": time.perf_counter() - t_start,
    }
    (OUT_DIR / f"{args.out_tag}_generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[{args.out_tag}] DONE: {stats}")


if __name__ == "__main__":
    main()

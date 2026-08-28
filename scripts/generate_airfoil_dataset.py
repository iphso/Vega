"""Bootstraps a training dataset for the airfoil domain -- the equivalent of
what preprocess.py does for VMEC++ by reformatting the upstream ConStellaration
HF dataset, except there's no existing CST-parameterized airfoil dataset to
reformat (AirfRANS uses the NACA family, not CST -- see EXPERIMENT_LOG §21),
so this generates one from scratch. Tractable specifically because XFOIL is
so fast (~15-45ms/call, confirmed in §21) that bulk generation is a
few-minutes job, unlike VMEC++'s dataset (bootstrapped from an existing
benchmark specifically because generating one from scratch at this project's
scale would have been impractical).

Sampling: a handful of archetypal CST seed shapes (thin/thick, symmetric/
cambered) perturbed by Gaussian noise at a range of scales. Not pure-random
CST coefficients -- confirmed in §21 that those essentially never converge
(noise std 1.2 -> 0% XFOIL convergence) the same way pure independent-per-
coefficient sampling failed for VMEC++ boundaries (§9's 0%-hit-rate finding).
Noise std is itself sampled per-candidate (not fixed) so the resulting
dataset spans a real range of "how far from a known-good shape," not one
narrow band.

Reynolds and angle of attack are treated as continuous columns of X (unlike
n_field_periods for VMEC++, which is a discrete aux one-hot) -- there's no
natural discrete conditioning variable here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import airfoil_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")

# A handful of archetypal CST seeds spanning thin/thick, symmetric/cambered --
# not meant to be exhaustive, just enough diversity that perturbing around
# several centers covers more of plausible-airfoil-space than perturbing one.
SEEDS = {
    "thin_symmetric": np.array([0.10] * 8 + [-0.10] * 8),
    "thick_symmetric": np.array([0.22] * 8 + [-0.22] * 8),
    "cambered_thin": np.array([0.14] * 8 + [-0.06] * 8),
    "cambered_thick": np.array([0.24] * 8 + [-0.12] * 8),
    "rear_loaded": np.array([0.08, 0.10, 0.14, 0.18, 0.20, 0.20, 0.18, 0.14] + [-0.06, -0.08, -0.10, -0.12, -0.14, -0.14, -0.12, -0.10]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=100_000)
    p.add_argument("--noise-std-range", type=float, nargs=2, default=(0.02, 0.18))
    p.add_argument("--reynolds-range", type=float, nargs=2, default=(1e5, 1e7))
    p.add_argument("--alpha-range", type=float, nargs=2, default=(-5.0, 15.0))
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=280)
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="airfoil")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    seed_names = list(SEEDS.keys())

    accepted_X, accepted_Y = [], []
    n_attempted, n_accepted = 0, 0
    t_start = time.perf_counter()

    def sample_batch(n):
        candidates = []  # (tag, params, aux)
        for i in range(n):
            base = SEEDS[seed_names[rng.integers(len(seed_names))]]
            noise_std = rng.uniform(*args.noise_std_range)
            params = base + rng.normal(0, noise_std, size=oracle.PARAM_DIM)
            reynolds = float(np.exp(rng.uniform(np.log(args.reynolds_range[0]), np.log(args.reynolds_range[1]))))
            alpha = float(rng.uniform(*args.alpha_range))
            aux = {"reynolds": reynolds, "mach": 0.0, "alpha": alpha}
            candidates.append((i, *oracle.params_to_worker_args(params, aux, "low"), params, reynolds, alpha))
        return candidates

    while n_accepted < args.target_count:
        raw = sample_batch(args.batch_size)
        jobs = [(tag, x, y, re_, m, al) for tag, x, y, re_, m, al, _params, _re, _al in raw]
        lookup = {tag: (params, re_, al) for tag, _x, _y, _re, _m, _al, params, re_, al in raw}

        n_attempted += len(jobs)
        for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
            if ok:
                params, reynolds, alpha = lookup[tag]
                row_x = np.concatenate([params, [reynolds, alpha]]).astype(np.float32)
                row_y = np.array([payload[name] for name in oracle.TARGET_NAMES], dtype=np.float32)
                if np.all(np.isfinite(row_y)):
                    accepted_X.append(row_x)
                    accepted_Y.append(row_y)
                    n_accepted += 1

        elapsed = time.perf_counter() - t_start
        print(f"[{args.out_tag}] attempted={n_attempted} accepted={n_accepted}/{args.target_count} "
              f"(hit rate {n_accepted / max(n_attempted, 1):.1%})  elapsed={elapsed:.0f}s")

        if len(accepted_X) >= args.checkpoint_every or n_accepted >= args.target_count:
            X = np.stack(accepted_X) if accepted_X else np.zeros((0, oracle.PARAM_DIM + 2), dtype=np.float32)
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

    feature_names = [f"upper_{i}" for i in range(8)] + [f"lower_{i}" for i in range(8)] + ["reynolds", "alpha"]
    (OUT_DIR / f"{args.out_tag}_target_names.json").write_text(json.dumps(oracle.TARGET_NAMES, indent=2))
    (OUT_DIR / f"{args.out_tag}_feature_names.json").write_text(json.dumps(feature_names, indent=2))
    stats = {"n_accepted": n_accepted, "n_attempted": n_attempted, "hit_rate": n_accepted / max(n_attempted, 1),
              "elapsed_seconds": time.perf_counter() - t_start, "seed": args.seed}
    (OUT_DIR / f"{args.out_tag}_generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\ndone. {n_accepted} rows in {stats['elapsed_seconds']:.0f}s ({stats['hit_rate']:.1%} hit rate)")


if __name__ == "__main__":
    main()

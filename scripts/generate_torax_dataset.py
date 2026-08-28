"""Bootstraps a training dataset for the TORAX domain -- the same "no existing
dataset, generate one from scratch" situation airfoils were in (§21-22), not
VMEC++'s "reformat an existing benchmark" situation (§9's PDEBench-scale HF
dataset doesn't exist for a 10-scalar tokamak transport parameterization
either). Generation is tractable, but nowhere near as cheap as airfoil's:
XFOIL is ~15-45ms/call (§21); TORAX is ~0.13-0.45s/call EVEN WARM (§31),
roughly an order of magnitude slower per candidate, plus a real (if rare)
hang risk on pathological input (§31's degenerate-candidate finding) that
neither VMEC++'s nor XFOIL's harness has to budget for at this severity.
Uses oracle_harness_persistent.run_batch_persistent, not
oracle_harness.run_batch_with_timeout -- the whole reason §31 built that
second harness.

Seeding: TWO archetypes, both derived from real bundled TORAX example
configs by direct inspection (`ToraxConfig.from_dict(...)`), not guessed --
same "confirmed working before trusted" bar as airfoil's CST seeds (§21) and
VMEC++'s boundary seeds (§9).

  - `iter_baseline`: basic_config's own RESOLVED defaults (every one of our
    10 PARAM_NAMES read back off the built ToraxConfig, not the sparse
    override dict) -- Ip=15MA, nbar=8.5e19 m^-3, chi_i=chi_e=1.0 m^2/s
    (constant-transport model), P_total=120MW, I_generic=3MA, R_major=6.2m,
    a_minor=2.0m, B_0=5.3T, elongation_LCFS=1.72. This is the exact
    configuration §31's own oracle calls were validated against.
  - `iter_flattop`: `iter_baseline` with Ip and P_total swapped for
    `iterhybrid_predictor_corrector`'s real flattop values (Ip=10.5MA,
    P_total=51MW) -- the ONLY two fields expressed in units actually
    compatible with our parameterization. Checked directly, not assumed:
    that example's `nbar` (0.8) is a Greenwald-FRACTION
    (`n_e_nbar_is_fGW: True` in its own config dict), a completely different
    quantity from `iter_baseline`'s absolute-density nbar (8.5e19) that our
    oracle's `_build_config` writes straight into `profile_conditions.nbar`
    with no such flag set -- reusing 0.8 there would mean ~0.8 particles/m^3,
    an near-vacuum plasma, not a real second density regime. Its transport
    model is `qlknn`, not the `constant` model our chi_i/chi_e override
    actually controls, so its chi values aren't comparable either. Both
    excluded from this archetype for that reason, not overlooked.
  - `iterhybrid_rampup` NOT used as a third archetype: it's a time-varying
    ramp *toward* predictor_corrector's own flattop point (`Ip:
    {0: 3e6, 80: 10.5e6}`), not a distinct steady-state regime our
    scalar (single-value, not time-profile) parameterization can represent
    -- `iter_flattop` already covers its endpoint.
  - `step_flattop_bgb` NOT used: fails outright on import in this container
    (`FileNotFoundError` on a bundled IMAS geometry file not shipped in the
    `torax` PyPI wheel) -- confirmed by actually trying it, not assumed
    working per §25/§31's "untested" flag.

Noise: ALL 10 params are strictly-positive physical quantities spanning
~8 orders of magnitude (elongation ~1 to P_total ~1e8), so perturbation is
multiplicative/log-space (`seed * exp(N(0, noise_std))`) rather than
airfoil's additive-in-linear-space noise -- additive noise at one scale
would be invisible on Ip and would blow elongation_LCFS negative. noise_std
is itself sampled per-candidate (not fixed), same reasoning as §21.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import torax_oracle as oracle
from oracle_harness_persistent import run_batch_persistent

OUT_DIR = Path("/work/output")

# Resolved directly off ToraxConfig.from_dict(basic_config.CONFIG) -- see
# module docstring. Order matches oracle.PARAM_NAMES.
ITER_BASELINE = {
    "Ip": 15_000_000.0, "nbar": 8.5e19, "chi_i": 1.0, "chi_e": 1.0,
    "P_total": 1.2e8, "I_generic": 3_000_000.0,
    "R_major": 6.2, "a_minor": 2.0, "B_0": 5.3, "elongation_LCFS": 1.72,
}
ITER_FLATTOP = {**ITER_BASELINE, "Ip": 10_500_000.0, "P_total": 5.1e7}

SEEDS = {
    "iter_baseline": np.array([ITER_BASELINE[n] for n in oracle.PARAM_NAMES]),
    "iter_flattop": np.array([ITER_FLATTOP[n] for n in oracle.PARAM_NAMES]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=300)
    p.add_argument("--noise-std-range", type=float, nargs=2, default=(0.03, 0.35))
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=160)
    p.add_argument("--timeout-seconds", type=float, default=30.0)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="torax")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    seed_names = list(SEEDS.keys())

    accepted_X, accepted_Y = [], []
    n_attempted, n_accepted = 0, 0
    t_start = time.perf_counter()

    def sample_batch(n):
        candidates = []  # (tag, overrides_dict, fidelity_name)
        lookup = {}
        for i in range(n):
            base = SEEDS[seed_names[rng.integers(len(seed_names))]]
            noise_std = rng.uniform(*args.noise_std_range)
            params = base * np.exp(rng.normal(0, noise_std, size=oracle.PARAM_DIM))
            # params_to_worker_args expects an already-resolved fidelity value
            # (see torax_oracle.py's own docstring on the bug this convention
            # mismatch caused, EXPERIMENT_LOG §35/§37) -- resolve "low" here,
            # not inside the oracle module.
            overrides, n_rho = oracle.params_to_worker_args(params, None, oracle.FIDELITY_PRESETS["low"])
            candidates.append((i, overrides, n_rho))
            lookup[i] = params
        return candidates, lookup

    while n_accepted < args.target_count:
        jobs, lookup = sample_batch(args.batch_size)
        n_attempted += len(jobs)

        for tag, ok, payload in run_batch_persistent(jobs, oracle.persistent_worker_fn, args.n_workers, args.timeout_seconds):
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
              f"(hit rate {n_accepted / max(n_attempted, 1):.1%})  elapsed={elapsed:.0f}s "
              f"({elapsed / max(n_attempted, 1):.2f}s/attempt)", flush=True)

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
            print(f"[{args.out_tag}] checkpointed -> {X_path} ({len(X)} rows total)", flush=True)

    (OUT_DIR / f"{args.out_tag}_target_names.json").write_text(json.dumps(oracle.TARGET_NAMES, indent=2))
    (OUT_DIR / f"{args.out_tag}_feature_names.json").write_text(json.dumps(oracle.PARAM_NAMES, indent=2))
    stats = {"n_accepted": n_accepted, "n_attempted": n_attempted, "hit_rate": n_accepted / max(n_attempted, 1),
              "elapsed_seconds": time.perf_counter() - t_start, "seed": args.seed}
    (OUT_DIR / f"{args.out_tag}_generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\ndone. {n_accepted} rows in {stats['elapsed_seconds']:.0f}s ({stats['hit_rate']:.1%} hit rate)")


if __name__ == "__main__":
    main()

"""Bootstraps a training dataset for the photonics domain (grating coupler,
photonics_oracle.py) -- same "no existing dataset, generate one from
scratch" situation airfoils/torax/mug were in, using
oracle_harness.run_batch_with_timeout since photonics_domain()'s own
harness_fit is confirmed "subprocess-per-candidate" (no JIT tax, see
gym_schema.py's photonics_domain() docstring), the same harness shape as
mug/airfoil/VMEC++, not TORAX's persistent-worker one. Must run inside the
`meep` compose service (real MEEP is conda-forge-only, see
photonics_oracle.py/Dockerfile.meep).

v2 (EXPERIMENT_LOG -- direct user request: "apodization and fiber-mode
overlap and variable buried-oxide thickness... give us a lot more good
variation and make it a more interesting optimization challenge"):
PARAM_DIM went 4->6 (duty_cycle split into duty_cycle_start/duty_cycle_end
for apodization, box_thickness_um promoted from a fixed constant to a real
parameter -- see photonics_oracle.py's own module docstring for the full
physics). This is a BREAKING schema change, not an extension -- the v1
4-param/5-target dataset (18,053 rows) is incompatible with v2's 6-param/
6-target shape and was archived as photonics_v1_4param_{X,Y,...} rather
than silently overwritten or appended onto (same "loud, not silent"
discipline as generate_mug_dataset.py's own existing-file warning).

Seeding: the same FIVE literature-grounded archetypes from v1
(canonical_630nm, full_etch, shallow_etch_weak, thin_soi_150nm,
near_bragg_reflective), each now given duty_cycle_start==duty_cycle_end
(the uniform/non-apodized degenerate case, matching what they were in v1)
and box_thickness_um=2.0 (v1's own former fixed constant) as their
baseline -- PLUS two new archetypes:

  - `apodized_literature`: duty_cycle_start=0.85, duty_cycle_end=0.25,
    period=0.83, wg=0.22, etch=0.07, box=2.0 -- the exact apodization
    recipe from Lomonte, Lenzini & Pernice 2021 (EXPERIMENT_LOG's own
    validation source for this domain), adapted to our 20-period grating
    (their own design used 50 periods at telecom wavelength; ours is
    shorter, a real, flagged difference, not presented as an exact replica).
  - `thin_box_interference`: box_thickness_um=0.7 (near the thin end of
    the real range), otherwise canonical_630nm's own values -- probes the
    up/down-split sensitivity to box thickness the v1 primer's validation
    work found in the literature (EXPERIMENT_LOG: designs ranging
    80%up/8.7%down to 1.7%up/86%down depending on exactly this dimension).

Each archetype is perturbed (multiplicative noise on wg_thickness/period/
etch_depth/box_thickness, additive+clip on both duty cycles) rather than
sampled exactly, for real local diversity around each real reference
point. A further UNIFORM-RANGE-sampled fraction (no archetype) covers the
rest of the space, matching mug_oracle.py's own "direct range sampling is
sufficient" finding for a domain this cheap to call -- both strategies
mixed 50/50 by candidate, not chosen once globally.

`--medium-fraction`: per-candidate, an independent coin flip sends that
fraction of rows through the medium (resolution=20) fidelity tier instead
of low. Each row's `{out_tag}_fidelity.json` entry records which tier
actually produced it, since low/medium have real, different, measured
numerical-error characteristics (EXPERIMENT_LOG: energy_closure error
~7-8% at low vs. ~1-2% at medium).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import photonics_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")

# [wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end, etch_depth_um, box_thickness_um]
SEEDS = {
    "canonical_630nm": np.array([0.22, 0.63, 0.50, 0.50, 0.07, 2.0]),
    "full_etch": np.array([0.22, 0.63, 0.50, 0.50, 0.22, 2.0]),
    "shallow_etch_weak": np.array([0.22, 0.63, 0.50, 0.50, 0.02, 2.0]),
    "thin_soi_150nm": np.array([0.15, 0.55, 0.50, 0.50, 0.05, 2.0]),
    "near_bragg_reflective": np.array([0.22, 0.28, 0.50, 0.50, 0.07, 2.0]),
    "apodized_literature": np.array([0.22, 0.83, 0.85, 0.25, 0.07, 2.0]),
    "thin_box_interference": np.array([0.22, 0.63, 0.50, 0.50, 0.07, 0.7]),
}

# Full-range bounds for the uniform-sampling half of every batch.
RANGES = dict(
    wg_thickness_um=(0.12, 0.34),
    grating_period_um=(0.25, 1.00),
    duty_cycle_start=(0.15, 0.85),
    duty_cycle_end=(0.15, 0.85),
    etch_depth_um=None,  # sampled as a fraction of wg_thickness_um below, not its own absolute range
    box_thickness_um=(0.5, 3.0),  # spans common real BOX thicknesses (thin ~0.5-0.7um to thick ~2-3um wafers)
)


def _perturb_seed(rng, base, noise_std):
    wg = base[0] * np.exp(rng.normal(0, noise_std))
    period = base[1] * np.exp(rng.normal(0, noise_std))
    dcs = float(np.clip(base[2] + rng.normal(0, noise_std), 0.05, 0.95))
    dce = float(np.clip(base[3] + rng.normal(0, noise_std), 0.05, 0.95))
    etch = base[4] * np.exp(rng.normal(0, noise_std))
    etch = min(etch, 0.98 * wg)
    box = base[5] * np.exp(rng.normal(0, noise_std))
    return np.array([wg, period, dcs, dce, etch, box])


def _sample_uniform(rng):
    wg = rng.uniform(*RANGES["wg_thickness_um"])
    period = rng.uniform(*RANGES["grating_period_um"])
    dcs = rng.uniform(*RANGES["duty_cycle_start"])
    dce = rng.uniform(*RANGES["duty_cycle_end"])
    etch = rng.uniform(0.05, 0.98) * wg  # always a physically valid fraction of the thickness just drawn
    box = rng.uniform(*RANGES["box_thickness_um"])
    return np.array([wg, period, dcs, dce, etch, box])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=3000)
    p.add_argument("--noise-std-range", type=float, nargs=2, default=(0.03, 0.25))
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=140)
    p.add_argument("--timeout-seconds", type=float, default=30.0)
    p.add_argument("--checkpoint-every", type=int, default=300)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS),
                    help="the LOW-tier fidelity every candidate not selected for --medium-fraction uses")
    p.add_argument("--medium-fraction", type=float, default=0.0,
                    help="fraction of candidates (independent per-candidate coin flip) run at medium fidelity instead of --fidelity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="photonics")
    args = p.parse_args()

    existing_X_path = OUT_DIR / f"{args.out_tag}_X.npy"
    if existing_X_path.exists():
        n_existing = np.load(existing_X_path).shape[0]
        print(f"[{args.out_tag}] WARNING: {existing_X_path} already has {n_existing} rows -- "
              f"this run will APPEND onto them, not replace them. If those rows are from the "
              f"v1 4-param schema, STOP and archive/delete first -- v2's 6-param schema is "
              f"incompatible (see this script's own module docstring).")

    rng = np.random.default_rng(args.seed)
    seed_names = list(SEEDS.keys())

    accepted_X, accepted_Y = [], []
    accepted_source = []  # which seed archetype (or "uniform") produced each accepted row, for post-hoc analysis
    accepted_fidelity = []  # which tier ("low"/"medium") actually scored each accepted row
    n_attempted, n_accepted = 0, 0
    t_start = time.perf_counter()

    def sample_batch(n):
        raw = []  # (tag, *worker_args, params, source_label, fidelity_label)
        for i in range(n):
            if rng.random() < 0.5:
                source = seed_names[rng.integers(len(seed_names))]
                params = _perturb_seed(rng, SEEDS[source], rng.uniform(*args.noise_std_range))
                # A perturbed seed can still drift outside the structurally-
                # valid range -- re-validate rather than trust the
                # perturbation, same defensive stance mug_oracle.py's own
                # worker_fn takes on its input.
                ok, _ = oracle._validate(*params)
                if not ok:
                    params = _sample_uniform(rng)
                    source = "uniform"
            else:
                params = _sample_uniform(rng)
                source = "uniform"
            fidelity = "medium" if rng.random() < args.medium_fraction else args.fidelity
            raw.append((i, *oracle.params_to_worker_args(params, {}, fidelity), params, source, fidelity))
        return raw

    while n_accepted < args.target_count:
        raw = sample_batch(args.batch_size)
        jobs = [c[:-3] for c in raw]
        lookup = {c[0]: (c[-3], c[-2], c[-1]) for c in raw}

        n_attempted += len(jobs)
        for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
            if ok:
                params, source, fidelity = lookup[tag]
                row_x = params.astype(np.float32)
                row_y = np.array([payload[name] for name in oracle.TARGET_NAMES], dtype=np.float32)
                if np.all(np.isfinite(row_y)):
                    accepted_X.append(row_x)
                    accepted_Y.append(row_y)
                    accepted_source.append(source)
                    accepted_fidelity.append(fidelity)
                    n_accepted += 1

        elapsed = time.perf_counter() - t_start
        print(f"[{args.out_tag}] attempted={n_attempted} accepted={n_accepted}/{args.target_count} "
              f"(hit rate {n_accepted / max(n_attempted, 1):.1%})  elapsed={elapsed:.0f}s "
              f"({elapsed / max(n_attempted, 1):.2f}s/attempt)", flush=True)

        if len(accepted_X) >= args.checkpoint_every or n_accepted >= args.target_count:
            X = np.stack(accepted_X) if accepted_X else np.zeros((0, oracle.PARAM_DIM), dtype=np.float32)
            Y = np.stack(accepted_Y) if accepted_Y else np.zeros((0, len(oracle.TARGET_NAMES)), dtype=np.float32)
            X_path, Y_path = OUT_DIR / f"{args.out_tag}_X.npy", OUT_DIR / f"{args.out_tag}_Y.npy"
            src_path = OUT_DIR / f"{args.out_tag}_source.json"
            fid_path = OUT_DIR / f"{args.out_tag}_fidelity.json"
            n_prior = 0
            if X_path.exists():
                n_prior = np.load(X_path).shape[0]
                X = np.concatenate([np.load(X_path), X])
                Y = np.concatenate([np.load(Y_path), Y])
            prior_source = json.loads(src_path.read_text()) if src_path.exists() else []
            sources_all = prior_source + accepted_source
            prior_fidelity = json.loads(fid_path.read_text()) if fid_path.exists() else ["low"] * n_prior
            fidelity_all = prior_fidelity + accepted_fidelity
            np.save(X_path, X)
            np.save(Y_path, Y)
            src_path.write_text(json.dumps(sources_all))
            fid_path.write_text(json.dumps(fidelity_all))
            accepted_X.clear()
            accepted_Y.clear()
            accepted_source.clear()
            accepted_fidelity.clear()
            print(f"[{args.out_tag}] checkpointed -> {X_path} ({len(X)} rows total)", flush=True)

    (OUT_DIR / f"{args.out_tag}_target_names.json").write_text(json.dumps(oracle.TARGET_NAMES, indent=2))
    (OUT_DIR / f"{args.out_tag}_feature_names.json").write_text(json.dumps(oracle.PARAM_NAMES, indent=2))
    stats = {"n_accepted": n_accepted, "n_attempted": n_attempted, "hit_rate": n_accepted / max(n_attempted, 1),
              "elapsed_seconds": time.perf_counter() - t_start, "seed": args.seed,
              "fidelity": args.fidelity, "medium_fraction": args.medium_fraction}
    (OUT_DIR / f"{args.out_tag}_generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\ndone. {n_accepted} rows in {stats['elapsed_seconds']:.0f}s ({stats['hit_rate']:.1%} hit rate)")


if __name__ == "__main__":
    main()

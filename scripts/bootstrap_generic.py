"""Generic dataset-bootstrapping loop: propose candidates via an already-
trained TARGET-CONDITIONED generator (cvae/diffusion/gan), validate through
the real oracle, accumulate every accepted design into a growing pool --
across any domain with a gym_schema.DomainSpec, not just VMEC++.

This deliberately replaces, rather than extends, this project's three
existing VMEC++-only bootstrap tools (bootstrap_loop.py, generate_and_validate.py,
gradient_walk.py), which all predate the target-conditioned generators
built in §16 onward: every one of them samples from train_vae.py's plain
UNCONDITIONAL VAE, and gradient_walk.py's/bootstrap_loop.py's "directed"
search pushes toward exactly one hardcoded metric (--directed-target) via
backprop through a separate, frozen surrogate regressor. None of them use
the conditional cvae/diffusion/gan this project has since validated for
actually understanding direction (§18-20) and not memorizing or mode-
collapsing (§26). This script uses those generators directly instead: same
random-unit-direction-in-target-space mechanism eval_random_direction_
steerability.py already validated for EVALUATING steerability, repurposed
here to DRIVE generation. All three of the old scripts also hand-roll their
own copy of the subprocess-timeout harness, predating oracle_harness.py's
consolidation -- this one uses the shared oracle_harness.run_batch_with_timeout
instead, no fourth duplicate.

Two seeding modes (--seed-mode), matching the user's "start with seeds or
not" framing:

  anchor (default) -- pick a real row from the domain's own dataset, nudge
    its OWN measured target by a random unit direction (--step-std norm),
    condition the generator on the nudged target + the anchor's own aux
    (nfp / reynolds+alpha), decode. Seeded by real data, explores outward
    from it in a fresh random direction every draw -- the "with seeds" mode.

  prior -- no real anchor at all: sample a target point directly from
    N(0, 1) in the same z-scored target space (i.e. "a plausible-ish
    target broadly within the real range," not anchored to any specific
    real row) plus a domain-appropriate random aux draw, condition the
    generator on that, decode. The "without seeds" mode -- whatever
    coverage this produces is purely a function of what the generator's
    own prior already captured, not pushed outward from a known point.

NOT included yet, by explicit user scoping: no retrain-the-generator-each-
round loop (bootstrap_loop.py's own generation structure) -- this reruns a
fixed, already-trained checkpoint every round. A real, deliberate scope cut
for a first version, not an oversight; the natural next step once this
baseline is in.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gym_schema import vmec_spec, airfoil_spec, torax_spec
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


# ---------------------------------------------------------------------------
# Domain-specific round builders -- everything downstream (oracle batching,
# accept/reject bookkeeping, checkpointing) is shared; only "how do I turn
# one round into decoded params + worker args + an X-row in this domain's
# own on-disk convention" differs, the same split eval_cvae_steerability.py
# and eval_airfoil_steerability.py already have (via each domain's own
# build_candidates), just generalized one level further here since gym_schema
# doesn't yet abstract conditioning/aux itself (flagged as open in §27).
# ---------------------------------------------------------------------------

def sample_unit_directions(n, dim, rng):
    v = rng.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


def build_round_vmec(sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std,
                      X, Y, seed_mode, step_std, n_per_round, k_per_seed, fidelity_name, rng):
    from eval_cvae_steerability import eval_space
    import vmec_oracle as oracle
    from train_vae import NFP_VALUES, nfp_one_hot

    Yz = (eval_space(Y, target_names) - t_mean) / t_std
    n_seeds = max(1, n_per_round // k_per_seed)
    jobs = []
    for s in range(n_seeds):
        if seed_mode == "anchor":
            idx = rng.integers(len(X))
            anchor_z = torch.tensor(Yz[idx:idx + 1], dtype=torch.float32)
            nfp = int(X[idx, 90])
            u = sample_unit_directions(1, n_targets, rng)[0]
            target_z = anchor_z[0].numpy() + step_std * u
        else:  # "prior"
            target_z = rng.normal(size=n_targets)
            nfp = int(rng.choice(NFP_VALUES))
        target_z_t = torch.tensor(target_z[None, :], dtype=torch.float32)
        nfp_t = torch.tensor([float(nfp)], dtype=torch.float32)
        cond = torch.cat([target_z_t, nfp_one_hot(nfp_t)], dim=-1)
        decoded = sample_fn(cond, k_per_seed)
        params = decoded * coeff_std + coeff_mean
        params[:, oracle.ZERO_INDICES] = 0.0
        for k in range(k_per_seed):
            tag = (s, k)
            x_row = np.concatenate([params[k], [float(nfp), 1.0]]).astype(np.float32)
            worker_args = oracle.params_to_worker_args(params[k], {"nfp": nfp}, fidelity_name)
            jobs.append((tag, x_row, *worker_args))
    return jobs


def build_round_airfoil(sample_fn, target_names, n_targets, coeff_dim, coeff_mean, coeff_std, t_mean, t_std,
                         re_mean, re_std, al_mean, al_std, X, Y, seed_mode, step_std, n_per_round, k_per_seed,
                         fidelity_name, rng):
    from eval_airfoil_steerability import eval_space
    import airfoil_oracle as oracle

    cd_col, lod_col = target_names.index("cd"), target_names.index("l_over_d")
    sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
    Xs, Ys = X[sane], Y[sane]
    Yz = (eval_space(Ys, target_names) - t_mean) / t_std

    n_seeds = max(1, n_per_round // k_per_seed)
    jobs = []
    for s in range(n_seeds):
        if seed_mode == "anchor":
            idx = rng.integers(len(Xs))
            anchor_z = Yz[idx]
            reynolds, alpha = float(Xs[idx, coeff_dim]), float(Xs[idx, coeff_dim + 1])
            u = sample_unit_directions(1, n_targets, rng)[0]
            target_z = anchor_z + step_std * u
        else:  # "prior"
            target_z = rng.normal(size=n_targets)
            reynolds = float(np.exp(rng.uniform(np.log(1e5), np.log(1e7))))
            alpha = float(rng.uniform(-5.0, 15.0))
        aux = torch.tensor([[(np.log(reynolds) - re_mean) / re_std, (alpha - al_mean) / al_std]], dtype=torch.float32)
        cond = torch.cat([torch.tensor(target_z[None, :], dtype=torch.float32), aux], dim=-1)
        decoded = sample_fn(cond, k_per_seed)
        params = decoded * coeff_std + coeff_mean
        for k in range(k_per_seed):
            tag = (s, k)
            x_row = np.concatenate([params[k], [reynolds, alpha]]).astype(np.float32)
            worker_args = oracle.params_to_worker_args(params[k], {"reynolds": reynolds, "mach": 0.0, "alpha": alpha}, fidelity_name)
            jobs.append((tag, x_row, *worker_args))
    return jobs


def build_round_torax(sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std,
                       log_target_names, X, Y, seed_mode, step_std, n_per_round, k_per_seed,
                       fidelity_name, rng):
    """No aux/conditioning beyond the target itself -- torax_oracle.py's own
    docstring is explicit that "everything that varies is already in
    params" (gym_schema.get_conditioning's torax branch has extra_dim=0),
    unlike VMEC's n_field_periods or airfoil's (reynolds, alpha). Simplest
    of the three round-builders as a direct result -- no extra x_row
    columns, no worker_aux dict beyond {}."""
    from steerability_generic import eval_space
    import torax_oracle as oracle

    Yz = (eval_space(Y, target_names, log_target_names) - t_mean) / t_std
    n_seeds = max(1, n_per_round // k_per_seed)
    jobs = []
    for s in range(n_seeds):
        if seed_mode == "anchor":
            idx = rng.integers(len(X))
            anchor_z = Yz[idx]
            u = sample_unit_directions(1, n_targets, rng)[0]
            target_z = anchor_z + step_std * u
        else:  # "prior"
            target_z = rng.normal(size=n_targets)
        cond = torch.tensor(target_z[None, :], dtype=torch.float32)
        decoded = sample_fn(cond, k_per_seed)
        params = decoded * coeff_std + coeff_mean
        for k in range(k_per_seed):
            tag = (s, k)
            x_row = params[k].astype(np.float32)
            worker_args = oracle.params_to_worker_args(params[k], {}, fidelity_name)
            jobs.append((tag, x_row, *worker_args))
    return jobs


# ---------------------------------------------------------------------------
# Shared loop -- domain-agnostic given a round-builder closure
# ---------------------------------------------------------------------------

def run_bootstrap(build_round_fn, worker_fn, target_names, out_dir, n_workers, timeout_s,
                   target_count, checkpoint_every, round_size, tag, sanity_filter=None,
                   batch_worker_fn=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "round_log.jsonl"
    accepted_X, accepted_Y = [], []
    rejected_X, rejected_reason = [], []
    n_attempted, n_accepted = 0, 0
    t_start = time.perf_counter()

    def _append(path, arrays):
        arr = np.concatenate([np.load(path)] + arrays) if path.exists() else np.concatenate(arrays)
        np.save(path, arr)

    def flush():
        if accepted_X:
            _append(out_dir / "X.npy", [np.stack(accepted_X)])
            _append(out_dir / "Y.npy", [np.stack(accepted_Y)])
            accepted_X.clear(); accepted_Y.clear()
        if rejected_X:
            _append(out_dir / "rejected_X.npy", [np.stack(rejected_X)])
            (out_dir / "rejected_reasons.json").write_text(json.dumps(rejected_reason[-2000:]))
            rejected_X.clear()

    round_idx = 0
    while n_accepted < target_count:
        round_idx += 1
        round_jobs = build_round_fn(round_size)  # [(tag, x_row, *worker_args), ...]
        tag_to_row = {tag: x_row for tag, x_row, *_ in round_jobs}
        oracle_jobs = [(tag, *wargs) for tag, x_row, *wargs in round_jobs]
        n_attempted += len(oracle_jobs)

        if batch_worker_fn is not None:
            # One jax.vmap call scores the WHOLE round at once (torax_oracle.
            # run_batch_vmap) instead of oracle_harness's subprocess-per-
            # candidate pool -- see gym_schema.DomainSpec.batch_worker_fn's
            # own docstring for why this is the GPU-fit path. round_size
            # must stay fixed across the whole run for this to hit the
            # compiled-function cache after round 1 (see run_batch_vmap's
            # own docstring) -- true here since round_size is a single CLI
            # arg reused every round, never varied mid-run.
            job_tags = [j[0] for j in oracle_jobs]
            job_wargs = [j[1:] for j in oracle_jobs]
            scored_results = batch_worker_fn(job_wargs)
            round_results = zip(job_tags, (r[0] for r in scored_results), (r[1] for r in scored_results))
        else:
            round_results = run_batch_with_timeout(oracle_jobs, worker_fn, n_workers, timeout_s)

        round_accepted = 0
        for job_tag, ok, payload in round_results:
            row = tag_to_row[job_tag]
            if ok:
                y = np.array([payload.get(name) for name in target_names], dtype=object)
                if all(v is not None for v in y) and np.all(np.isfinite(y.astype(np.float64))):
                    y = y.astype(np.float32)
                    # Same physical-sanity gate every training script for this domain applies
                    # (e.g. airfoil's cd>=1e-6/|l_over_d|<=300, §22) -- "the oracle ran and
                    # returned finite numbers" is NOT the same bar as "this is a physically
                    # sane design," confirmed the hard way (see EXPERIMENT_LOG §33: an early,
                    # unfiltered run of this exact loop accepted a candidate with
                    # l_over_d=61,069 before this check existed).
                    if sanity_filter is None or bool(sanity_filter(y[None, :])[0]):
                        accepted_X.append(row)
                        accepted_Y.append(y)
                        n_accepted += 1
                        round_accepted += 1
                        continue
                    rejected_X.append(row); rejected_reason.append("failed_sanity_filter")
                else:
                    rejected_X.append(row); rejected_reason.append("non_finite_metrics")
            else:
                rejected_X.append(row); rejected_reason.append(str(payload)[:80])

        elapsed = time.perf_counter() - t_start
        record = {"round": round_idx, "n_round_attempted": len(oracle_jobs), "n_round_accepted": round_accepted,
                   "n_attempted": n_attempted, "n_accepted": n_accepted,
                   "hit_rate": n_accepted / max(n_attempted, 1), "elapsed_seconds": elapsed}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{tag}] round {round_idx}: {round_accepted}/{len(oracle_jobs)} this round -- "
              f"{n_accepted}/{target_count} total accepted ({n_accepted / max(n_attempted, 1):.1%} hit rate, "
              f"{elapsed:.0f}s elapsed)")

        if len(accepted_X) >= checkpoint_every or len(rejected_X) >= checkpoint_every:
            flush()

    flush()
    print(f"[{tag}] done: {n_accepted}/{n_attempted} accepted ({n_accepted / max(n_attempted, 1):.1%}) "
          f"in {time.perf_counter() - t_start:.0f}s. pool at {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=["vmec", "airfoil", "torax"])
    p.add_argument("--model-type", default="cvae", choices=["cvae", "diffusion", "gan"])
    p.add_argument("--tag", required=True, help="checkpoint tag to load")
    p.add_argument("--seed-mode", default="anchor", choices=["anchor", "prior"])
    p.add_argument("--step-std", type=float, default=1.5,
                    help="anchor mode only: norm of the random target-space push away from the anchor's own measured target")
    p.add_argument("--target-count", type=int, default=2000)
    p.add_argument("--round-size", type=int, default=112)
    p.add_argument("--k-per-seed", type=int, default=4, help="candidates decoded per seed/anchor draw")
    p.add_argument("--fidelity", default="low")
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default=None)
    p.add_argument("--max-steps", type=int, default=200,
                    help="torax only: bounded-loop step budget for the vmap batch_worker_fn path -- keep round_size fixed across a whole run so the compiled (round_size, max_steps) function is only built once (see run_batch_vmap's docstring)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cpu")
    out_tag = args.out_tag or f"{args.domain}_{args.model_type}_{args.seed_mode}_s{args.seed}"
    out_dir = OUT_DIR / f"bootstrap_generic_{out_tag}"

    batch_worker_fn = None

    if args.domain == "torax":
        from steerability_generic import load_generative_model
        spec = torax_spec()
        fidelity_name = spec.fidelities[0].internal_name  # torax has only "low" today (§31/§35 open item)
        bundle = load_generative_model("torax", args.model_type, args.tag, dev)
        sample_fn, target_names, n_targets = bundle["sample_fn"], bundle["target_names"], bundle["n_targets"]
        coeff_mean, coeff_std = bundle["coeff_mean"], bundle["coeff_std"]
        t_mean, t_std, log_target_names = bundle["t_mean"], bundle["t_std"], bundle["log_target_names"]
        x_path, y_path = spec.dataset_paths(OUT_DIR)
        X, Y = np.load(x_path), np.load(y_path)
        print(f"[{out_tag}] domain=torax model={args.model_type} seed_mode={args.seed_mode} "
              f"fidelity={args.fidelity}={fidelity_name} step_std={args.step_std} max_steps={args.max_steps}")

        def build_round(n):
            return build_round_torax(sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std,
                                      log_target_names, X, Y, args.seed_mode, args.step_std, n, args.k_per_seed,
                                      fidelity_name, rng)
        worker_fn = spec.worker_fn
        if spec.batch_worker_fn is not None:
            batch_worker_fn = lambda wargs: spec.batch_worker_fn(wargs, max_steps=args.max_steps)
    elif args.domain == "vmec":
        from eval_cvae_steerability import load_generative_model
        spec = vmec_spec()
        fidelity_name = spec.fidelities[{"low": 0, "medium": 1, "high": 2}[args.fidelity]].internal_name
        sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std = \
            load_generative_model(args.model_type, args.tag, dev)
        X, Y = np.load(OUT_DIR / "X.npy"), np.load(OUT_DIR / "Y.npy")
        print(f"[{out_tag}] domain=vmec model={args.model_type} seed_mode={args.seed_mode} "
              f"fidelity={args.fidelity}={fidelity_name} step_std={args.step_std}")

        def build_round(n):
            return build_round_vmec(sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std,
                                     X, Y, args.seed_mode, args.step_std, n, args.k_per_seed, fidelity_name, rng)
        worker_fn = spec.worker_fn
    else:
        from eval_airfoil_steerability import load_generative_model
        spec = airfoil_spec()
        fidelity_name = spec.fidelities[0].internal_name  # airfoil has only "low" today (§21/§27 open item)
        (sample_fn, target_names, n_targets, coeff_dim, coeff_mean, coeff_std, t_mean, t_std,
         re_mean, re_std, al_mean, al_std, dataset_tag) = load_generative_model(args.model_type, args.tag, dev)
        X, Y = np.load(OUT_DIR / f"{dataset_tag}_X.npy"), np.load(OUT_DIR / f"{dataset_tag}_Y.npy")
        print(f"[{out_tag}] domain=airfoil model={args.model_type} seed_mode={args.seed_mode} "
              f"fidelity={args.fidelity}={fidelity_name} step_std={args.step_std}")

        def build_round(n):
            return build_round_airfoil(sample_fn, target_names, n_targets, coeff_dim, coeff_mean, coeff_std,
                                        t_mean, t_std, re_mean, re_std, al_mean, al_std, X, Y,
                                        args.seed_mode, args.step_std, n, args.k_per_seed, fidelity_name, rng)
        worker_fn = spec.worker_fn

    run_bootstrap(build_round, worker_fn, target_names, out_dir, args.n_workers, args.timeout_seconds,
                  args.target_count, args.checkpoint_every, args.round_size, out_tag,
                  sanity_filter=spec.sanity_filter, batch_worker_fn=batch_worker_fn)


if __name__ == "__main__":
    main()

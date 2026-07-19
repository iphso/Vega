"""Self-training / bootstrap generation loop.

Each generation: retrain the VAE (scripts/train_vae.py) on real data + every
accepted synthetic design from all prior generations, sample a new batch from
that retrained VAE, validate through the real VMEC++ oracle (reusing
scripts/generate_and_validate.py's timeout-safe subprocess runner), and
append accepted (and rejected, reason-flagged) designs to a running pool.
Repeat for --generations cycles, each bounded by --seconds-per-generation.

The real risk here -- what the user is calling "meta-overfitting" -- is that
each generation's VAE is partly trained on its own previous generation's
output, so left alone it converges toward whatever it already produced
rather than expanding coverage. Two deliberate counter-measures:

  1. Latent sampling variance inflates over generations (see --schedule
     below), pushing outward rather than re-condensing.
  2. A fixed --anchor-frac of every batch (default 10%) is instead sampled
     from --base-vae-tag (the original VAE, trained on real data only,
     never retrained) at its native std=1 -- a stable "outside view" the
     loop can't drift away from no matter how many generations run.

Two std schedules (--schedule): "linear" (std grows every generation,
optionally capped by --max-std -- confirmed empirically that this ties
coverage growth and hit-rate directly together: freezing std stabilizes
hit-rate but coverage plateaus too, see EXPERIMENT_LOG) and "cycle" (repeating
relax -> expand -> fill phases: relax to --relax-std for --cycle-relax-gens
generations to recover hit-rate and let the VAE consolidate, ramp up to that
cycle's peak over --cycle-expand-gens generations to seed new territory,
hold at the peak for --cycle-fill-gens generations to densify it, then
relax again -- with the peak itself increasing by --peak-std-growth every
cycle so the long-run trend still expands outward, just with periodic
recovery instead of a monotonic hit-rate bleed).

A third, fixed --random-frac of every batch (default 10%) bypasses the VAE
entirely: each of the 81 free coefficients sampled independently and
uniformly within its own observed range (generate_and_validate.py's
"random" sampling mode -- confirmed ~0% hit rate on its own, see
EXPERIMENT_LOG). This isn't for hit rate. Nearly all of it will be rejected,
but VMEC++'s rejection is itself a labeled data point about where the
infeasible boundary sits, sampled from a distribution with zero learned
bias -- useful precisely because it's unlike anything the VAE would ever
produce. Combined with the reject-flagging below, this is the main source
of genuinely-diverse negative examples in the pool.

A fourth, optional --directed-frac (default 0, off) replaces blind variance
expansion with gradient-*directed* movement toward --directed-target: real
anchor designs are encoded by the current generation's VAE, then walked
--directed-steps steps via ordinary PyTorch autograd through decoder ->
--surrogate-tag (a small, fast, already-trained regression net -- NOT
vmec_jax's physics-implicit-diff, which turned out slow and fragile; see
EXPERIMENT_LOG for why plain neural-net backprop is a completely different,
much cheaper thing). This is exploring toward an objective, not searching
for one final answer -- every resulting candidate still goes through the
exact same real-VMEC++ validation as everything else in this loop, because
the gradient is w.r.t. the SURROGATE's prediction, not real physics: a
design-search candidate the surrogate ensemble was confident about has
already been caught failing 2/3 real constraints once in this project (see
EXPERIMENT_LOG S8). The surrogate is also NOT retrained each generation
(unlike the VAE) -- its gradient signal reflects the data it was trained on,
not the growing pool. Per-generation logging separates the directed
sub-batch's hit rate and mean *measured* (not predicted) target value from
the rest, so it's visible whether this is actually moving real physics or
just exploiting the surrogate.

The goal of this loop is broadening the feasible-region prior (coverage and
target-range extension), not surrogate metric accuracy -- so the two things
logged per generation to generation_log.jsonl are exactly that, not RMSE:
  - coverage_fraction_under_covered: fraction of that generation's accepted
    designs whose nearest real neighbor sits in an under-covered cluster
    (same definition as scripts/make_splits.py's cluster split). If this
    trends down across generations, the loop is collapsing and needs more
    forced exploration; holding or rising means it's genuinely still
    expanding.
  - range_extension: per target, how far this generation's (and the
    cumulative pool's) accepted designs reach beyond the real dataset's
    observed [min, max] -- the direct measure of "extending the space."

Pool stored at output/bootstrap_<tag>/{X,Y}.npy (accepted, same 92-col
schema as output/X.npy) with a parallel generation_idx.npy for provenance,
and {rejected_X,rejected_reason,rejected_generation_idx}.npy for the
nonphysical side (same reason codes as generate_and_validate.py).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from generate_and_validate import (
    NFP_VALUES, REJECT_REASON_CODES, ZERO_COEFF_IDX, make_random_sampler, run_batch_with_timeout,
)

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def load_vae(tag):
    from train_vae import VAE
    dev = torch.device("cpu")
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = VAE(coeff_dim=90, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["coeff_mean"], ckpt["coeff_std"], ckpt["latent_dim"], ckpt["hidden"]


def sample_from_vae(model, coeff_mean, coeff_std, latent_dim, n, rng, std):
    from train_vae import nfp_one_hot
    z = std * torch.randn(n, latent_dim)
    nfp_batch = rng.choice(NFP_VALUES, size=n)
    nfp_t = torch.tensor(nfp_batch, dtype=torch.float32)
    with torch.no_grad():
        decoded = model.decode(z, nfp_one_hot(nfp_t)).numpy()
    coeffs = decoded * coeff_std + coeff_mean
    coeffs[:, ZERO_COEFF_IDX] = 0.0
    out = []
    for k in range(n):
        r_cos = coeffs[k, :45].reshape(5, 9).astype(np.float64)
        z_sin = coeffs[k, 45:90].reshape(5, 9).astype(np.float64)
        out.append((r_cos, z_sin, int(nfp_batch[k])))
    return out


def compute_current_std(args, g, schedule_gen):
    """Returns (current_std, phase_label, cycle_idx). "linear": monotonic
    ramp over the absolute lineage generation g, optionally capped. "cycle":
    relax (flat low) -> expand (ramp to that cycle's peak) -> fill (flat at
    peak), repeating over schedule_gen (a counter local to when --schedule
    cycle started, not the absolute lineage count), with the peak ratcheting
    up --peak-std-growth per cycle.
    """
    if args.schedule == "linear":
        current_std = 1.0 + args.std_growth * g
        if args.max_std is not None:
            current_std = min(current_std, args.max_std)
        return current_std, "linear", 0

    cycle_length = args.cycle_relax_gens + args.cycle_expand_gens + args.cycle_fill_gens
    cycle_idx = (schedule_gen - 1) // cycle_length
    pos = (schedule_gen - 1) % cycle_length
    peak_std = args.peak_std_start + args.peak_std_growth * cycle_idx

    if pos < args.cycle_relax_gens:
        return args.relax_std, "relax", cycle_idx
    if pos < args.cycle_relax_gens + args.cycle_expand_gens:
        step = pos - args.cycle_relax_gens + 1
        frac = step / args.cycle_expand_gens
        return args.relax_std + frac * (peak_std - args.relax_std), "expand", cycle_idx
    return peak_std, "fill", cycle_idx


def make_directed_sampler(current_model_bundle, surrogate, target_idx, direction,
                           n_steps, step_size, anchor_pool_X, rng):
    """Gradient-directed candidates: encode real anchors with the CURRENT
    generation's VAE, then take n_steps of plain-autograd gradient descent
    (normalized-direction steps, no adaptive shrinking -- this loop doesn't
    validate mid-walk, only the final result, same as every other sampler
    here) through decoder -> surrogate toward minimizing/maximizing
    target_idx. No VMEC++ in this function at all -- it just proposes
    candidates, exactly like the other samplers; validation happens once,
    afterward, in the main loop.

    Exposes sample.last_predicted (the surrogate's own predicted target
    value for the batch just proposed) so the caller can log mean
    predicted-by-the-surrogate vs. mean measured-by-VMEC++ -- the real
    check on whether this is moving actual physics or just exploiting the
    surrogate's blind spots.
    """
    from train_vae import nfp_one_hot
    c_model, c_mean, c_std, c_latent, _ = current_model_bundle
    coeff_mean_t = torch.tensor(c_mean, dtype=torch.float32)
    coeff_std_t = torch.tensor(c_std, dtype=torch.float32)
    sign = -1.0 if direction == "min" else 1.0

    def sample(n):
        replace = len(anchor_pool_X) < n
        anchor_rows = rng.choice(len(anchor_pool_X), size=n, replace=replace)
        anchor_coeffs = torch.tensor((anchor_pool_X[anchor_rows][:, :90] - c_mean) / c_std, dtype=torch.float32)
        anchor_nfp = torch.tensor(anchor_pool_X[anchor_rows][:, 90], dtype=torch.float32)
        nfp_cond = nfp_one_hot(anchor_nfp)
        ones = torch.ones(n, 1)

        with torch.no_grad():
            z0, _ = c_model.encode(anchor_coeffs, nfp_cond)
        z = z0.clone().requires_grad_(True)

        for _ in range(n_steps):
            coeffs = c_model.decode(z, nfp_cond) * coeff_std_t + coeff_mean_t
            pred = surrogate(torch.cat([coeffs, anchor_nfp.unsqueeze(1), ones], dim=1))
            grad, = torch.autograd.grad(pred[:, target_idx].sum(), z)
            with torch.no_grad():
                grad_norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                z = (z + sign * step_size * grad / grad_norm).detach().requires_grad_(True)

        with torch.no_grad():
            coeffs_final = c_model.decode(z, nfp_cond) * coeff_std_t + coeff_mean_t
            pred_final = surrogate(torch.cat([coeffs_final, anchor_nfp.unsqueeze(1), ones], dim=1))
        sample.last_predicted = pred_final[:, target_idx].numpy().copy()

        coeffs_np = coeffs_final.numpy()
        coeffs_np[:, ZERO_COEFF_IDX] = 0.0
        out = []
        for k in range(n):
            r_cos = coeffs_np[k, :45].reshape(5, 9).astype(np.float64)
            z_sin = coeffs_np[k, 45:90].reshape(5, 9).astype(np.float64)
            out.append((r_cos, z_sin, int(anchor_nfp[k].item())))
        return out

    sample.last_predicted = None
    return sample


def make_mixed_sampler(anchor_model_bundle, current_model_bundle, random_sampler, directed_sampler,
                        anchor_frac, random_frac, directed_frac, current_std, rng):
    """Returns sample(n) -> (candidates, tags), tags marking which
    sub-sampler produced each candidate ("anchor"/"random"/"directed"/
    "current") so the caller can report the directed sub-batch's own
    hit rate and achieved target value separately from the rest.
    """
    a_model, a_mean, a_std, a_latent, _ = anchor_model_bundle
    c_model, c_mean, c_std, c_latent, _ = current_model_bundle

    def sample(n):
        n_anchor = int(round(anchor_frac * n))
        n_random = int(round(random_frac * n))
        n_directed = int(round(directed_frac * n)) if directed_sampler else 0
        n_current = n - n_anchor - n_random - n_directed
        out, tags = [], []
        if n_anchor:
            out += sample_from_vae(a_model, a_mean, a_std, a_latent, n_anchor, rng, std=1.0)
            tags += ["anchor"] * n_anchor
        if n_random:
            out += random_sampler(n_random)
            tags += ["random"] * n_random
        if n_directed:
            out += directed_sampler(n_directed)
            tags += ["directed"] * n_directed
        if n_current:
            out += sample_from_vae(c_model, c_mean, c_std, c_latent, n_current, rng, std=current_std)
            tags += ["current"] * n_current
        combined = list(zip(out, tags))
        rng.shuffle(combined)
        if not combined:
            return [], []
        out2, tags2 = zip(*combined)
        return list(out2), list(tags2)

    return sample


def load_coverage_reference(sparse_percentile, subsample, rng):
    X = np.load(OUT_DIR / "X.npy")
    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    feature_names = json.loads((OUT_DIR / "feature_names.json").read_text())
    feat_std = np.array([meta["feature_stats"][n]["std"] for n in feature_names[:90]]).clip(min=1e-6)
    assign = np.load(OUT_DIR / "cluster_assignments.npy")
    cluster_sizes = {int(k): v for k, v in json.loads((OUT_DIR / "cluster_sizes.json").read_text()).items()}
    threshold = np.percentile(list(cluster_sizes.values()), sparse_percentile)
    sparse_clusters = {c for c, n in cluster_sizes.items() if n <= threshold}

    idx = rng.choice(len(X), size=min(subsample, len(X)), replace=False)
    X_sub = (X[idx][:, :90] / feat_std).astype(np.float32)
    assign_sub = assign[idx]
    return X_sub, assign_sub, sparse_clusters, feat_std


def coverage_fraction(Xg, X_sub, assign_sub, sparse_clusters, feat_std, chunk=200):
    if len(Xg) == 0:
        return None
    Xgn = (Xg[:, :90] / feat_std).astype(np.float32)
    nearest = np.zeros(len(Xgn), dtype=int)
    for start in range(0, len(Xgn), chunk):
        d = np.linalg.norm(Xgn[start:start + chunk, None, :] - X_sub[None, :, :], axis=2)
        nearest[start:start + chunk] = assign_sub[d.argmin(axis=1)]
    return float(np.mean([c in sparse_clusters for c in nearest]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-vae-tag", default="vae_coeffs_s0", help="real-data-only VAE, used as the fixed anchor")
    p.add_argument("--tag", default="bootstrap0")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--seconds-per-generation", type=float, default=7200.0)
    p.add_argument("--vae-epochs", type=int, default=100, help="epochs per generation's VAE retrain (warm-started, so fewer than a from-scratch fit)")
    p.add_argument("--anchor-frac", type=float, default=0.10)
    p.add_argument("--random-frac", type=float, default=0.10,
                    help="fraction of every batch sampled independently per-coefficient, bypassing the VAE "
                         "entirely -- near-0% hit rate expected, kept for the rejected-candidate pool as "
                         "zero-learned-bias negative examples, not for adding accepted designs")
    p.add_argument("--std-growth", type=float, default=0.10,
                    help="latent std inflation per generation -- the goal here is extending coverage/range, "
                         "not metric accuracy, so this defaults more aggressive than a typical exploration schedule")
    p.add_argument("--max-std", type=float, default=None,
                    help="linear schedule only: cap on current_std (1.0 + std_growth*generation) -- without "
                         "this, generation count and exploration radius are hard-tied together. Set this to "
                         "freeze std at its current level and just keep accumulating at that radius.")
    p.add_argument("--schedule", default="linear", choices=["linear", "cycle"],
                    help="linear: current_std = 1.0 + std_growth*generation (optionally capped by --max-std) -- "
                         "monotonic, so expanding reach and stabilizing hit-rate are mutually exclusive. "
                         "cycle: repeating relax -> expand -> fill phases (see --cycle-*/--relax-std/"
                         "--peak-std-* args) -- periodically bursts std outward to seed new territory, then "
                         "holds at the new peak to consolidate/densify before relaxing and bursting again, "
                         "with the peak itself ratcheting up cycle over cycle.")
    p.add_argument("--cycle-relax-gens", type=int, default=1,
                    help="cycle schedule: generations held at --relax-std (hit-rate recovery, VAE consolidation)")
    p.add_argument("--cycle-expand-gens", type=int, default=2,
                    help="cycle schedule: generations linearly ramping std from --relax-std to that cycle's peak")
    p.add_argument("--cycle-fill-gens", type=int, default=2,
                    help="cycle schedule: generations held at that cycle's peak std (densify the new frontier)")
    p.add_argument("--relax-std", type=float, default=1.5,
                    help="cycle schedule: the low point each cycle relaxes to")
    p.add_argument("--peak-std-start", type=float, default=2.2,
                    help="cycle schedule: peak std for the first cycle")
    p.add_argument("--peak-std-growth", type=float, default=0.2,
                    help="cycle schedule: how much the peak std increases each successive cycle")
    p.add_argument("--directed-frac", type=float, default=0.0,
                    help="fraction of every batch from gradient-directed search toward --directed-target "
                         "(0 = off, matching prior behavior exactly). See module docstring for what this "
                         "does and doesn't guarantee.")
    p.add_argument("--directed-target", default="max_elongation")
    p.add_argument("--directed-direction", default="min", choices=["min", "max"])
    p.add_argument("--directed-steps", type=int, default=3,
                    help="gradient steps per generation for the directed sub-batch (fresh anchors each "
                         "generation, not a persistent walk across generations)")
    p.add_argument("--directed-step-size", type=float, default=0.1, help="normalized latent step size")
    p.add_argument("--surrogate-tag", default="split_vae_prior_augmented_s0",
                    help="static surrogate checkpoint for --directed-frac -- not retrained each generation")
    p.add_argument("--sparse-percentile", type=float, default=50.0)
    p.add_argument("--coverage-subsample", type=int, default=3000)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=112)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    target_stats = json.loads((OUT_DIR / "metadata.json").read_text())["target_stats"]
    real_min = np.array([target_stats[t]["min"] for t in target_names])
    real_max = np.array([target_stats[t]["max"] for t in target_names])
    pool_dir = OUT_DIR / f"bootstrap_{args.tag}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    log_path = pool_dir / "generation_log.jsonl"

    anchor_bundle = load_vae(args.base_vae_tag)
    base_ckpt = torch.load(CKPT_DIR / f"{args.base_vae_tag}.pt", map_location="cpu")
    latent_dim, hidden = base_ckpt["latent_dim"], base_ckpt["hidden"]

    X_sub, assign_sub, sparse_clusters, feat_std = load_coverage_reference(
        args.sparse_percentile, args.coverage_subsample, rng)

    surrogate = directed_target_idx = X_anchor_pool = None
    if args.directed_frac > 0:
        from gradient_walk import load_surrogate
        surrogate, surrogate_target_names = load_surrogate(args.surrogate_tag)
        assert surrogate_target_names == target_names, \
            "surrogate's target order doesn't match target_names.json -- can't trust the target index"
        directed_target_idx = target_names.index(args.directed_target)
        X_anchor_pool = np.load(OUT_DIR / "X.npy")
        print(f"[{args.tag}] directed search active: {args.directed_direction} {args.directed_target} "
              f"({args.directed_frac:.0%} of every batch, {args.directed_steps} steps, "
              f"surrogate={args.surrogate_tag})")

    def _append(path, arrays):
        arr = np.concatenate([np.load(path)] + arrays) if path.exists() else np.concatenate(arrays)
        np.save(path, arr)

    # Auto-resume: a second invocation with the same --tag continues the same
    # lineage (generation numbering, warm-start chain, std-growth schedule)
    # instead of restarting at gen1/base-vae each time, which would silently
    # discard the exploration progression and duplicate generation numbers in
    # the log.
    # The cycle schedule needs its own counter, separate from the absolute
    # lineage generation number: it should start at 1 the first time --schedule
    # cycle is used (not silently inherit gen15's absolute count as "3 cycles
    # already happened"), but continue correctly on a later resume of the
    # cycle schedule specifically (not reset to cycle 0 every invocation,
    # which would lose the peak-ratcheting progression).
    start_gen = 1
    start_schedule_gen = 1
    prev_vae_tag = args.base_vae_tag
    if log_path.exists():
        last_record = json.loads(log_path.read_text().strip().splitlines()[-1])
        start_gen = last_record["generation"] + 1
        prev_vae_tag = last_record["vae_tag"]
        if last_record.get("schedule") == args.schedule == "cycle":
            start_schedule_gen = last_record.get("schedule_gen", 0) + 1
        print(f"[{args.tag}] resuming from generation {start_gen} (warm start from {prev_vae_tag}, "
              f"schedule_gen starting at {start_schedule_gen})")

    for g in range(start_gen, start_gen + args.generations):
        schedule_gen = start_schedule_gen + (g - start_gen)
        gen_vae_tag = f"{args.tag}_gen{g}"
        extra_x_args = []
        if (pool_dir / "X.npy").exists():
            extra_x_args = ["--extra-x", str(pool_dir / "X.npy")]
        train_cmd = [
            sys.executable, "scripts/train_vae.py",
            "--latent-dim", str(latent_dim), "--hidden", str(hidden),
            "--epochs", str(args.vae_epochs), "--tag", gen_vae_tag,
            "--warm-start", prev_vae_tag, "--seed", str(args.seed + g),
        ] + extra_x_args
        print(f"[{args.tag}] generation {g}: retraining VAE -> {gen_vae_tag} (warm start from {prev_vae_tag})")
        subprocess.run(train_cmd, check=True)
        prev_vae_tag = gen_vae_tag

        current_bundle = load_vae(gen_vae_tag)
        current_std, phase, cycle_idx = compute_current_std(args, g, schedule_gen)
        random_sampler = make_random_sampler(args, rng)
        directed_sampler = None
        if args.directed_frac > 0:
            directed_sampler = make_directed_sampler(
                current_bundle, surrogate, directed_target_idx, args.directed_direction,
                args.directed_steps, args.directed_step_size, X_anchor_pool, rng)
        sample_candidates = make_mixed_sampler(
            anchor_bundle, current_bundle, random_sampler, directed_sampler,
            args.anchor_frac, args.random_frac, args.directed_frac, current_std, rng)
        print(f"[{args.tag}] generation {g}: sampling with anchor_frac={args.anchor_frac} "
              f"random_frac={args.random_frac} directed_frac={args.directed_frac} current_std={current_std:.3f} "
              f"phase={phase} cycle={cycle_idx} schedule_gen={schedule_gen} for {args.seconds_per_generation:.0f}s")

        accepted_X, accepted_Y, gen_idx_acc = [], [], []
        rejected_X, rejected_reason, gen_idx_rej = [], [], []
        n_attempted = n_accepted = n_reject_structural = n_reject_vmec = n_timeout = n_reject_unknown = 0
        n_directed_attempted = n_directed_accepted = 0
        directed_measured_values, directed_predicted_values = [], []
        t_start = time.perf_counter()

        def flush():
            if accepted_X:
                _append(pool_dir / "X.npy", [np.stack(accepted_X)])
                _append(pool_dir / "Y.npy", [np.stack(accepted_Y)])
                _append(pool_dir / "generation_idx.npy", [np.array(gen_idx_acc, dtype=np.int32)])
                accepted_X.clear(); accepted_Y.clear(); gen_idx_acc.clear()
            if rejected_X:
                _append(pool_dir / "rejected_X.npy", [np.stack(rejected_X)])
                _append(pool_dir / "rejected_reason.npy", [np.array(rejected_reason, dtype=np.int8)])
                _append(pool_dir / "rejected_generation_idx.npy", [np.array(gen_idx_rej, dtype=np.int32)])
                rejected_X.clear(); rejected_reason.clear(); gen_idx_rej.clear()
                (pool_dir / "rejected_reason_legend.json").write_text(
                    json.dumps({v: k for k, v in REJECT_REASON_CODES.items()}, indent=2))

        gen_accepted_X, gen_accepted_Y = [], []
        while time.perf_counter() - t_start < args.seconds_per_generation:
            candidates, tags = sample_candidates(args.batch_size)
            n_attempted += len(candidates)
            if directed_sampler is not None and directed_sampler.last_predicted is not None:
                directed_predicted_values.extend(directed_sampler.last_predicted.tolist())
            # run_batch_with_timeout yields in completion order, not submission
            # order -- match each result back to its sampler tag by exact value.
            tag_lookup = {(r.tobytes(), z.tobytes(), nfp): tag for (r, z, nfp), tag in zip(candidates, tags)}
            for ok, r_cos, z_sin, nfp, payload in run_batch_with_timeout(candidates, args.n_workers, args.timeout_seconds):
                tag = tag_lookup.get((r_cos.tobytes(), z_sin.tobytes(), nfp), "unknown_source")
                if tag == "directed":
                    n_directed_attempted += 1
                row = np.concatenate([r_cos.flatten(), z_sin.flatten(), [float(nfp), 1.0]]).astype(np.float32)
                if ok:
                    y = np.array([payload[name] for name in target_names], dtype=np.float64)
                    if any(v is None for v in y) or not np.all(np.isfinite(y.astype(np.float64))):
                        n_reject_vmec += 1
                        rejected_X.append(row); rejected_reason.append(REJECT_REASON_CODES["vmec"]); gen_idx_rej.append(g)
                        continue
                    accepted_X.append(row); accepted_Y.append(y.astype(np.float32)); gen_idx_acc.append(g)
                    gen_accepted_X.append(row); gen_accepted_Y.append(y)
                    n_accepted += 1
                    if tag == "directed":
                        n_directed_accepted += 1
                        directed_measured_values.append(y[directed_target_idx])
                else:
                    if payload.startswith("structural"):
                        n_reject_structural += 1; reason = "structural"
                    elif "timed out" in payload:
                        n_timeout += 1; reason = "timeout"
                    elif payload.startswith("unknown"):
                        n_reject_unknown += 1; reason = "unknown"
                    else:
                        n_reject_vmec += 1; reason = "vmec"
                    rejected_X.append(row); rejected_reason.append(REJECT_REASON_CODES[reason]); gen_idx_rej.append(g)

            elapsed = time.perf_counter() - t_start
            print(f"[{args.tag}][gen{g}] attempted={n_attempted} accepted={n_accepted} "
                  f"(hit rate {n_accepted / max(n_attempted, 1):.1%})  "
                  f"reject: structural={n_reject_structural} vmec={n_reject_vmec} timeout={n_timeout} "
                  f"unknown={n_reject_unknown}  elapsed={elapsed:.0f}s")

            if len(accepted_X) >= args.checkpoint_every or len(rejected_X) >= args.checkpoint_every:
                flush()

        flush()

        cov = coverage_fraction(np.stack(gen_accepted_X), X_sub, assign_sub, sparse_clusters, feat_std) \
            if gen_accepted_X else None

        # Range extension is the actual metric of interest here (broadening the
        # feasible-region prior), not surrogate accuracy -- track how far this
        # generation's accepted designs, and the cumulative pool, reach beyond
        # the real dataset's observed range on each target.
        range_extension = None
        if gen_accepted_Y:
            gen_Y = np.stack(gen_accepted_Y)
            pool_Y = np.load(pool_dir / "Y.npy")
            gen_beyond_max = np.maximum(0, gen_Y.max(axis=0) - real_max)
            gen_beyond_min = np.maximum(0, real_min - gen_Y.min(axis=0))
            pool_beyond_max = np.maximum(0, pool_Y.max(axis=0) - real_max)
            pool_beyond_min = np.maximum(0, real_min - pool_Y.min(axis=0))
            range_extension = {
                name: {
                    "gen_beyond_max": float(gen_beyond_max[i]), "gen_beyond_min": float(gen_beyond_min[i]),
                    "pool_beyond_max": float(pool_beyond_max[i]), "pool_beyond_min": float(pool_beyond_min[i]),
                }
                for i, name in enumerate(target_names)
            }

        directed_stats = None
        if args.directed_frac > 0:
            directed_stats = {
                "target": args.directed_target, "direction": args.directed_direction,
                "n_attempted": n_directed_attempted, "n_accepted": n_directed_accepted,
                "hit_rate": n_directed_accepted / max(n_directed_attempted, 1),
                # predicted = surrogate's own belief, over every attempted directed
                # candidate; measured = real VMEC++ value, over only the accepted
                # ones. A large gap between these is the surrogate being wrong,
                # not the search failing.
                "predicted_target_mean": float(np.mean(directed_predicted_values)) if directed_predicted_values else None,
                "measured_target_mean": float(np.mean(directed_measured_values)) if directed_measured_values else None,
            }

        record = {
            "generation": g, "vae_tag": gen_vae_tag, "current_std": current_std,
            "schedule": args.schedule, "schedule_gen": schedule_gen, "phase": phase, "cycle_idx": cycle_idx,
            "n_attempted": n_attempted, "n_accepted": n_accepted,
            "hit_rate": n_accepted / max(n_attempted, 1),
            "n_reject_structural": n_reject_structural, "n_reject_vmec": n_reject_vmec, "n_timeout": n_timeout,
            "n_reject_unknown": n_reject_unknown,
            "directed": directed_stats,
            "coverage_fraction_under_covered": cov,
            "range_extension": range_extension,
            "elapsed_seconds": time.perf_counter() - t_start,
            "pool_size_after": (len(np.load(pool_dir / "X.npy")) if (pool_dir / "X.npy").exists() else 0),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        directed_summary = (f"  directed: {directed_stats['n_accepted']}/{directed_stats['n_attempted']} "
                             f"({directed_stats['hit_rate']:.1%})  predicted_{args.directed_target}_mean="
                             f"{directed_stats['predicted_target_mean']}  measured_{args.directed_target}_mean="
                             f"{directed_stats['measured_target_mean']}") if directed_stats else ""
        print(f"[{args.tag}] generation {g} done: {json.dumps(record)}{directed_summary}")

    print(f"[{args.tag}] all {args.generations} generations complete. pool at {pool_dir}, log at {log_path}")


if __name__ == "__main__":
    main()

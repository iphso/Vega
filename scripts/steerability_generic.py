"""Domain-agnostic validity + steerability + selectivity eval -- replaces
the two near-duplicate VMEC++-only scripts this project had
(eval_cvae_steerability.py's axis-aligned test, eval_random_direction_
steerability.py's arbitrary-direction test) with ONE script parameterized
by --domain and --direction-mode, built on gym_schema.py's DomainSpec/
Conditioning abstractions. A third domain (TORAX, or any of the others)
plugs in by writing a gym_schema spec + conditioning factory, not another
eval script -- the whole point of this generalization, per direct user
request: "adopt abstractions that allow us to do this extension in the
easiest and most straightforward way possible, assuming we're going to do
this with more oracles later."

--direction-mode axis: nudge exactly one target dim at a time (n_targets
  one-hot directions per anchor, matching eval_cvae_steerability.py's
  original test exactly -- verified bit-for-bit reproducible against it,
  see EXPERIMENT_LOG, before this replaced it).
--direction-mode arbitrary: --n-directions random unit vectors in the full
  target space per anchor, applied to every target simultaneously
  (matching eval_random_direction_steerability.py's test exactly).

Both modes share one metric now instead of two incompatible ones:
cosine similarity between requested and achieved movement (achieved delta
z-scored against the anchor's own converged-validity-candidate baseline,
never a re-verified real row -- the self-consistent-baseline discipline
established in EXPERIMENT_LOG §17 and carried forward everywhere since).
`correct_direction_rate` (cosine > 0) generalizes axis mode's old binary
sign check -- provably the same quantity when the direction is one-hot,
confirmed by construction, not just claimed. Selectivity (on-target vs.
off-target z-movement) is computed for axis mode, where "the other targets"
is a well-defined concept; arbitrary mode's cosine similarity already
captures the analogous idea (how much of the achieved movement lands along
the requested direction vs. orthogonal to it) so a separate selectivity
number isn't computed there.

Deliberately narrower than the two scripts it replaces in one respect:
only random anchor selection is implemented (farthest-point sampling was
confirmed in §17/19 to bias low for reasons unrelated to the model, and was
never actually used in any of the comparability-relevant configs across
either domain) -- dropped rather than ported, to keep the generalization
itself simpler.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gym_schema import airfoil_spec, get_conditioning, mug_spec, torax_spec, vmec_spec
from oracle_harness import run_batch_with_timeout
from oracle_harness_persistent import run_batch_persistent

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

SPEC_FACTORIES = {"vmec": vmec_spec, "airfoil": airfoil_spec, "torax": torax_spec, "mug": mug_spec}


def eval_space(Y, target_names, log_target_names):
    Ye = Y.copy()
    for name in log_target_names:
        idx = target_names.index(name)
        Ye[:, idx] = np.log(np.clip(Ye[:, idx], 1e-12, None))
    return Ye


def load_dataset(spec, dataset_tag=None):
    if spec.dataset_paths is not None:
        x_path, y_path = spec.dataset_paths(OUT_DIR)
    else:
        x_path, y_path = OUT_DIR / f"{dataset_tag}_X.npy", OUT_DIR / f"{dataset_tag}_Y.npy"
    return np.load(x_path), np.load(y_path)


def random_anchor_sample(points, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(len(points), size=n, replace=False)


def sample_unit_directions(n, dim, rng):
    v = rng.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


def load_generative_model(domain_name, model_type, tag, dev):
    spec = SPEC_FACTORIES[domain_name]()
    # weights_only=False: these are this project's own trained checkpoints
    # (numpy arrays for coeff_mean/std, not just tensors), not third-party
    # files -- needed explicitly once the torax image picked up a torch
    # version newer than 2.6 (default flipped there), see Dockerfile.torax.
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev, weights_only=False)
    conditioning = get_conditioning(domain_name, ckpt)
    coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
    target_names = ckpt["target_names"]
    n_targets = len(target_names)
    log_target_names = ckpt.get("log_target_names", spec.log_target_names)
    param_dim = spec.param_dim

    if model_type == "cvae":
        from train_cvae import CVAE
        model = CVAE(coeff_dim=param_dim, n_targets=n_targets, latent_dim=ckpt["latent_dim"],
                     hidden=ckpt["hidden"], n_nfp=conditioning.extra_dim).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        latent_dim = ckpt["latent_dim"]

        def sample_fn(cond, k):
            with torch.no_grad():
                z = torch.randn(k, latent_dim)
                return model.decode(z, cond.repeat(k, 1)).numpy()
    elif model_type == "diffusion":
        from train_diffusion import DiffusionDenoiser, ddpm_sample, make_schedule
        model = DiffusionDenoiser(coeff_dim=param_dim, n_targets=n_targets, hidden=ckpt["hidden"],
                                   time_embed_dim=ckpt["time_embed_dim"], n_nfp=conditioning.extra_dim).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        schedule = make_schedule(ckpt["T"], device=dev)

        def sample_fn(cond, k):
            return ddpm_sample(model, cond.repeat(k, 1), schedule, coeff_dim=param_dim, device=dev).numpy()
    else:
        from train_gan import Generator as GANGenerator
        model = GANGenerator(coeff_dim=param_dim, n_targets=n_targets, latent_dim=ckpt["latent_dim"],
                              hidden=ckpt["hidden"], n_nfp=conditioning.extra_dim).to(dev)
        model.load_state_dict(ckpt["generator_state_dict"])
        model.eval()
        latent_dim = ckpt["latent_dim"]

        def sample_fn(cond, k):
            with torch.no_grad():
                z = torch.randn(k, latent_dim)
                return model(z, cond.repeat(k, 1)).numpy()

    return {
        "spec": spec, "conditioning": conditioning, "sample_fn": sample_fn,
        "target_names": target_names, "log_target_names": log_target_names, "n_targets": n_targets,
        "coeff_mean": coeff_mean, "coeff_std": coeff_std, "t_mean": t_mean, "t_std": t_std,
        "param_dim": param_dim, "dataset_tag": ckpt.get("dataset_tag"),
    }


def build_candidates(sample_fn, cond, k, coeff_mean, coeff_std, zero_indices, spec, worker_aux, fidelity_name):
    decoded = sample_fn(cond, k)
    params = decoded * coeff_std + coeff_mean
    if zero_indices:
        params[:, zero_indices] = 0.0
    return [spec.params_to_worker_args(params[i], worker_aux, fidelity_name) for i in range(k)]


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(SPEC_FACTORIES))
    p.add_argument("--model-type", default="cvae", choices=["cvae", "diffusion", "gan"])
    p.add_argument("--tag", required=True)
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--direction-mode", default="axis", choices=["axis", "arbitrary"])
    p.add_argument("--n-directions", type=int, default=None,
                    help="arbitrary mode only; default = n_targets (matches axis mode's per-anchor oracle cost)")
    p.add_argument("--k-steer", type=int, default=3)
    p.add_argument("--step-std", type=float, default=1.0)
    p.add_argument("--steer-mode", default="targeted", choices=["targeted", "null"])
    p.add_argument("--fidelity", default="low")
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default=None)
    return p


def run_eval(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cpu")

    bundle = load_generative_model(args.domain, args.model_type, args.tag, dev)
    spec, conditioning = bundle["spec"], bundle["conditioning"]
    sample_fn, target_names, log_target_names, n_targets = (
        bundle["sample_fn"], bundle["target_names"], bundle["log_target_names"], bundle["n_targets"])
    coeff_mean, coeff_std, t_mean, t_std = bundle["coeff_mean"], bundle["coeff_std"], bundle["t_mean"], bundle["t_std"]
    param_dim = bundle["param_dim"]

    fidelity_map = {f.name: f.internal_name for f in spec.fidelities}
    fidelity_name = fidelity_map[args.fidelity]

    X, Y = load_dataset(spec, bundle["dataset_tag"])
    if spec.sanity_filter is not None:
        mask = spec.sanity_filter(Y)
        X, Y = X[mask], Y[mask]

    Yz = (eval_space(Y, target_names, log_target_names) - t_mean) / t_std
    anchor_idx = random_anchor_sample(Yz, args.n_anchors, seed=args.seed)
    n_directions = n_targets if args.direction_mode == "axis" else (args.n_directions or n_targets)

    print(f"[{args.tag}] domain={args.domain} model={args.model_type} direction_mode={args.direction_mode} "
          f"{args.n_anchors} anchors x {n_directions} directions, steer_mode={args.steer_mode}, "
          f"fidelity={args.fidelity}={fidelity_name}, step_std={args.step_std}, seed={args.seed}")

    jobs = []  # (tag, *worker_args)
    directions = {}  # (a_i, dir_i) -> unit vector (n_targets,)
    for a_i, idx in enumerate(anchor_idx):
        row = X[idx]
        aux = conditioning.aux_from_row(row, param_dim)
        anchor_z = Yz[idx]
        worker_aux = conditioning.worker_aux(aux)

        cond = conditioning.cond_from_target_and_aux(anchor_z, aux)
        for k, cand in enumerate(build_candidates(sample_fn, cond, args.k_validity, coeff_mean, coeff_std,
                                                    spec.zero_indices, spec, worker_aux, fidelity_name)):
            jobs.append((("validity", a_i, k), *cand))

        for dir_i in range(n_directions):
            if args.direction_mode == "axis":
                u = np.zeros(n_targets)
                u[dir_i] = -1.0  # matches the original axis-aligned test's convention: always decrease
            else:
                u = sample_unit_directions(1, n_targets, rng)[0]
            directions[(a_i, dir_i)] = u
            steer_z = anchor_z.copy()
            if args.steer_mode == "targeted":
                steer_z = steer_z + args.step_std * u
            steer_cond = conditioning.cond_from_target_and_aux(steer_z, aux)
            for k, cand in enumerate(build_candidates(sample_fn, steer_cond, args.k_steer, coeff_mean, coeff_std,
                                                        spec.zero_indices, spec, worker_aux, fidelity_name)):
                jobs.append((("steer", a_i, dir_i, k), *cand))

    print(f"[{args.tag}] {len(jobs)} oracle calls queued "
          f"({args.n_anchors * args.k_validity} validity + {args.n_anchors * n_directions * args.k_steer} steer)")

    # EXPERIMENT_LOG §37: dispatch on harness_fit rather than always using
    # oracle_harness.run_batch_with_timeout -- a subprocess-per-candidate
    # harness would pay TORAX's ~4.9s JIT-compile tax on every single
    # candidate (confirmed §25/§31), the exact situation
    # oracle_harness_persistent.py exists to avoid. Every other domain's
    # harness_fit is still "subprocess-per-candidate", so this is a no-op
    # change for vmec/airfoil.
    if spec.harness_fit == "needs-batched-worker" and spec.persistent_worker_fn is not None:
        batch_iter = run_batch_persistent(jobs, spec.persistent_worker_fn, args.n_workers, args.timeout_seconds)
    else:
        batch_iter = run_batch_with_timeout(jobs, spec.worker_fn, args.n_workers, args.timeout_seconds)

    results = {}
    n_done = n_converged = 0
    for tag, ok, payload in batch_iter:
        n_done += 1
        if ok:
            y = np.array([payload.get(name) for name in target_names], dtype=object)
            if all(v is not None for v in y) and np.all(np.isfinite(y.astype(np.float64))):
                results[tag] = y.astype(np.float64)
                n_converged += 1
        if tag not in results:
            results[tag] = None
        if n_done % 40 == 0 or n_done == len(jobs):
            print(f"[{args.tag}] oracle: {n_done}/{len(jobs)} done (hit rate so far {n_converged / n_done:.1%})")

    validity_hits = validity_attempts = 0
    n_anchors_with_valid_baseline = 0
    cosine_sims, magnitude_ratios = [], []
    on_target_abs_z_all, off_target_abs_z_all = [], []
    n_steer_attempted = n_steer_converged = n_judgeable = 0

    for a_i, idx in enumerate(anchor_idx):
        validity_ys = [results.get(("validity", a_i, k)) for k in range(args.k_validity)]
        validity_ys = [y for y in validity_ys if y is not None]
        validity_hits += len(validity_ys)
        validity_attempts += args.k_validity
        baseline_e = eval_space(np.stack(validity_ys), target_names, log_target_names).mean(axis=0) if validity_ys else None
        if baseline_e is not None:
            n_anchors_with_valid_baseline += 1

        for dir_i in range(n_directions):
            u = directions[(a_i, dir_i)]
            samples = [results.get(("steer", a_i, dir_i, k)) for k in range(args.k_steer)]
            converged = [s for s in samples if s is not None]
            n_steer_attempted += args.k_steer
            n_steer_converged += len(converged)
            if baseline_e is None:
                continue
            n_judgeable += len(converged)
            for s in converged:
                delta_z = (eval_space(s[None, :], target_names, log_target_names)[0] - baseline_e) / t_std
                denom = np.linalg.norm(delta_z) * np.linalg.norm(u)
                cos_sim = float(np.dot(delta_z, u) / denom) if denom > 1e-12 else 0.0
                cosine_sims.append(cos_sim)
                magnitude_ratios.append(float(np.linalg.norm(delta_z) / args.step_std))
                if args.direction_mode == "axis":
                    on_target_abs_z_all.append(float(abs(delta_z[dir_i])))
                    off_target_abs_z_all.append(float(np.mean(np.abs(np.delete(delta_z, dir_i)))))

    overall_validity_rate = validity_hits / max(validity_attempts, 1)
    steer_conv_rate = n_steer_converged / max(n_steer_attempted, 1)
    mean_cos = float(np.mean(cosine_sims)) if cosine_sims else None
    median_cos = float(np.median(cosine_sims)) if cosine_sims else None
    correct_dir_rate = float(np.mean([c > 0 for c in cosine_sims])) if cosine_sims else None

    print(f"\n=== {args.tag}: {args.n_anchors} anchors x {n_directions} directions ({args.direction_mode}) ===")
    print(f"  {n_anchors_with_valid_baseline}/{args.n_anchors} anchors have a usable baseline")
    print(f"  validity hit rate: {overall_validity_rate:.1%}")
    print(f"  steer-candidate convergence: {steer_conv_rate:.1%} ({n_steer_converged}/{n_steer_attempted})")
    print(f"  correct-direction rate (cos>0): {correct_dir_rate}")
    print(f"  mean/median cosine similarity: {mean_cos} / {median_cos}")

    out = {
        "domain": args.domain, "model_type": args.model_type, "tag": args.tag,
        "direction_mode": args.direction_mode, "n_anchors": args.n_anchors, "n_directions": n_directions,
        "k_validity": args.k_validity, "k_steer": args.k_steer, "step_std": args.step_std,
        "steer_mode": args.steer_mode, "fidelity": args.fidelity, "seed": args.seed,
        "overall_validity_hit_rate": overall_validity_rate,
        "n_anchors_with_valid_baseline": n_anchors_with_valid_baseline,
        "steer_candidate_convergence_rate": steer_conv_rate,
        "n_judgeable": n_judgeable,
        "correct_direction_rate": correct_dir_rate,
        "mean_cosine_similarity": mean_cos, "median_cosine_similarity": median_cos,
        "mean_magnitude_ratio": float(np.mean(magnitude_ratios)) if magnitude_ratios else None,
    }
    if args.direction_mode == "axis":
        overall_on = float(np.median(on_target_abs_z_all)) if on_target_abs_z_all else None
        overall_off = float(np.median(off_target_abs_z_all)) if off_target_abs_z_all else None
        out["overall_selectivity"] = {
            "median_on_target_abs_z": overall_on, "median_off_target_mean_abs_z": overall_off,
            "selectivity_ratio": (overall_on / overall_off) if overall_off else None,
        }

    out_path = OUT_DIR / f"steerability_generic_{args.out_tag or args.tag}_{args.direction_mode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return out


def main():
    args = build_arg_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

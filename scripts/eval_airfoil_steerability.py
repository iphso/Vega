"""Validity + steerability + selectivity for the airfoil domain -- same
methodology as eval_cvae_steerability.py (VMEC++ domain), same baseline
discipline (mean of the anchor's own converged validity candidates, never a
foreign-pipeline re-verification -- learned that lesson the hard way once
already, see EXPERIMENT_LOG §17, not repeating it here even though this
dataset is self-generated and the mistake's specific mechanism doesn't
apply the same way). Like eval_cvae_steerability.py, --model-type dispatches
across cvae/diffusion/gan (train_airfoil_cvae.py / train_airfoil_diffusion.py
/ train_airfoil_gan.py) through one shared harness -- same anchors, same
oracle, same fidelity, only the sampling mechanism differs.

Conditioning differs from the VMEC harness: Reynolds (log-space) and angle
of attack are continuous aux, concatenated directly rather than one-hot
encoded (see train_airfoil_cvae.py). Anchors are real rows from the
generated airfoil_X.npy/Y.npy dataset, picked uniformly at random (no
farthest-point option here -- §17/19 already found that biases the read
toward the domain's own extremes for reasons unrelated to the model, so it
was never worth reproducing as a default here).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import airfoil_oracle as oracle
from oracle_harness import run_batch_with_timeout
from train_airfoil_cvae import CVAE
from train_diffusion import DiffusionDenoiser, ddpm_sample, make_schedule
from train_gan import Generator as GANGenerator

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
LOG_TARGET_NAMES = ["cd"]


def eval_space(Y, target_names):
    Ye = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Ye[:, idx] = np.log(np.clip(Ye[:, idx], 1e-12, None))
    return Ye


def load_generative_model(model_type, tag, dev):
    """Mirrors eval_cvae_steerability.py's load_generative_model, adapted
    for airfoil checkpoints (dataset_tag, reynolds/alpha aux stats instead
    of nfp). Returns (sample_fn, target_names, n_targets, coeff_dim,
    coeff_mean, coeff_std, t_mean, t_std, re_mean, re_std, al_mean, al_std,
    dataset_tag) -- sample_fn(cond, k) -> (k, coeff_dim) numpy array of
    standardized params, regardless of model type."""
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    target_names = ckpt["target_names"]
    n_targets = len(target_names)
    coeff_dim = ckpt["coeff_dim"]
    coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
    re_mean, re_std = ckpt["reynolds_mean"], ckpt["reynolds_std"]
    al_mean, al_std = ckpt["alpha_mean"], ckpt["alpha_std"]

    if model_type == "cvae":
        model = CVAE(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"], n_nfp=2).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        latent_dim = ckpt["latent_dim"]

        def sample_fn(cond, k):
            with torch.no_grad():
                z = torch.randn(k, latent_dim)
                return model.decode(z, cond.repeat(k, 1)).numpy()
    elif model_type == "diffusion":
        model = DiffusionDenoiser(coeff_dim=coeff_dim, n_targets=n_targets, hidden=ckpt["hidden"],
                                   time_embed_dim=ckpt["time_embed_dim"], n_nfp=2).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        schedule = make_schedule(ckpt["T"], device=dev)

        def sample_fn(cond, k):
            return ddpm_sample(model, cond.repeat(k, 1), schedule, coeff_dim=coeff_dim, device=dev).numpy()
    else:
        model = GANGenerator(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"], n_nfp=2).to(dev)
        model.load_state_dict(ckpt["generator_state_dict"])
        model.eval()
        latent_dim = ckpt["latent_dim"]

        def sample_fn(cond, k):
            with torch.no_grad():
                z = torch.randn(k, latent_dim)
                return model(z, cond.repeat(k, 1)).numpy()

    return (sample_fn, target_names, n_targets, coeff_dim, coeff_mean, coeff_std, t_mean, t_std,
            re_mean, re_std, al_mean, al_std, ckpt["dataset_tag"])


def build_candidates(sample_fn, cond, k, coeff_mean, coeff_std, aux_phys, fidelity_name):
    decoded = sample_fn(cond, k)
    params = decoded * coeff_std + coeff_mean
    reynolds, alpha = aux_phys
    return [oracle.params_to_worker_args(params[i], {"reynolds": reynolds, "mach": 0.0, "alpha": alpha}, fidelity_name)
            for i in range(k)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", default="cvae", choices=["cvae", "diffusion", "gan"])
    p.add_argument("--tag", default="airfoil_cvae_s0", help="checkpoint tag; defaults assume --model-type cvae's naming")
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--k-steer", type=int, default=3)
    p.add_argument("--step-std", type=float, default=1.0)
    p.add_argument("--steer-mode", default="targeted", choices=["targeted", "null"])
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cpu")
    (sample_fn, target_names, n_targets, coeff_dim, coeff_mean, coeff_std, t_mean, t_std,
     re_mean, re_std, al_mean, al_std, dataset_tag) = load_generative_model(args.model_type, args.tag, dev)

    X = np.load(OUT_DIR / f"{dataset_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{dataset_tag}_Y.npy")
    Ye = eval_space(Y, target_names)
    Yz = (Ye - t_mean) / t_std

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(len(X), size=args.n_anchors, replace=False)
    print(f"[{args.tag}] {args.n_anchors} random anchors from {len(X)} generated rows, "
          f"k_validity={args.k_validity}, k_steer={args.k_steer}, step_std={args.step_std}, mode={args.steer_mode}")

    jobs = []
    for a_i, idx in enumerate(anchor_idx):
        reynolds, alpha = float(X[idx, coeff_dim]), float(X[idx, coeff_dim + 1])
        anchor_z = torch.tensor(Yz[idx:idx + 1], dtype=torch.float32)
        aux_cond = torch.tensor([[(np.log(reynolds) - re_mean) / re_std, (alpha - al_mean) / al_std]], dtype=torch.float32)
        cond = torch.cat([anchor_z, aux_cond], dim=-1)
        for k, cand in enumerate(build_candidates(sample_fn, cond, args.k_validity, coeff_mean, coeff_std, (reynolds, alpha), "low")):
            jobs.append((("validity", a_i, k), *cand))

        for d in range(n_targets):
            steer_z = anchor_z.clone()
            if args.steer_mode == "targeted":
                steer_z[0, d] -= args.step_std
            steer_cond = torch.cat([steer_z, aux_cond], dim=-1)
            for k, cand in enumerate(build_candidates(sample_fn, steer_cond, args.k_steer, coeff_mean, coeff_std, (reynolds, alpha), "low")):
                jobs.append((("steer", a_i, d, k), *cand))

    print(f"[{args.tag}] {len(jobs)} oracle calls queued "
          f"({args.n_anchors * args.k_validity} validity + {args.n_anchors * n_targets * args.k_steer} steer)")

    results = {}
    n_done, n_converged = 0, 0
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
        n_done += 1
        if ok:
            y = np.array([payload[name] for name in target_names], dtype=np.float64)
            if np.all(np.isfinite(y)):
                results[tag] = y
                n_converged += 1
        if tag not in results:
            results[tag] = None
        if n_done % 40 == 0 or n_done == len(jobs):
            print(f"[{args.tag}] oracle: {n_done}/{len(jobs)} (hit rate {n_converged / n_done:.1%})")

    steer_by_dim = {d: {"attempted": 0, "converged": 0, "judgeable": 0, "correct_direction": 0,
                         "on_target_abs_z": [], "off_target_abs_z": []} for d in range(n_targets)}
    validity_hits, validity_attempts = 0, 0
    n_anchors_with_valid_baseline = 0

    for a_i, idx in enumerate(anchor_idx):
        validity_ys = [results.get(("validity", a_i, k)) for k in range(args.k_validity)]
        validity_ys = [y for y in validity_ys if y is not None]
        validity_hits += len(validity_ys)
        validity_attempts += args.k_validity
        baseline_e = eval_space(np.stack(validity_ys), target_names).mean(axis=0) if validity_ys else None
        if baseline_e is not None:
            n_anchors_with_valid_baseline += 1

        for d in range(n_targets):
            requested_delta = -args.step_std * t_std[d]
            samples = [results.get(("steer", a_i, d, k)) for k in range(args.k_steer)]
            converged = [s for s in samples if s is not None]
            steer_by_dim[d]["attempted"] += args.k_steer
            steer_by_dim[d]["converged"] += len(converged)
            if baseline_e is not None:
                steer_by_dim[d]["judgeable"] += len(converged)
                for s in converged:
                    delta_z = (eval_space(s[None, :], target_names)[0] - baseline_e) / t_std
                    achieved_delta = delta_z[d] * t_std[d]
                    if achieved_delta < 0:
                        steer_by_dim[d]["correct_direction"] += 1
                    steer_by_dim[d]["on_target_abs_z"].append(float(abs(delta_z[d])))
                    steer_by_dim[d]["off_target_abs_z"].append(float(np.mean(np.abs(np.delete(delta_z, d)))))

    overall_validity_rate = validity_hits / max(validity_attempts, 1)
    print(f"\n=== {args.tag}: {args.n_anchors} anchors ===")
    print(f"  {n_anchors_with_valid_baseline}/{args.n_anchors} anchors have a usable baseline")
    print(f"  validity hit rate: {overall_validity_rate:.1%} ({validity_hits}/{validity_attempts})")
    print(f"\n  {'target':12s} {'converged':>10s} {'judgeable':>10s} {'correct dir':>12s} {'selectivity':>12s}")
    all_on, all_off = [], []
    for d in range(n_targets):
        s = steer_by_dim[d]
        conv_rate = s["converged"] / max(s["attempted"], 1)
        correct_rate = s["correct_direction"] / max(s["judgeable"], 1) if s["judgeable"] else float("nan")
        on_med = float(np.median(s["on_target_abs_z"])) if s["on_target_abs_z"] else float("nan")
        off_med = float(np.median(s["off_target_abs_z"])) if s["off_target_abs_z"] else float("nan")
        sel = on_med / off_med if off_med else float("nan")
        all_on.extend(s["on_target_abs_z"]); all_off.extend(s["off_target_abs_z"])
        print(f"  {target_names[d]:12s} {conv_rate:9.1%}  {s['judgeable']:10d}  {correct_rate:11.1%}  {sel:11.2f}")
    overall_sel = (np.median(all_on) / np.median(all_off)) if all_off else float("nan")

    out = {
        "model_type": args.model_type,
        "tag": args.tag, "n_anchors": args.n_anchors, "k_validity": args.k_validity, "k_steer": args.k_steer,
        "step_std": args.step_std, "steer_mode": args.steer_mode,
        "overall_validity_hit_rate": overall_validity_rate,
        "n_anchors_with_valid_baseline": n_anchors_with_valid_baseline,
        "overall_selectivity": float(overall_sel),
        "steer_by_target": {
            target_names[d]: {
                "convergence_rate": steer_by_dim[d]["converged"] / max(steer_by_dim[d]["attempted"], 1),
                "n_judgeable": steer_by_dim[d]["judgeable"],
                "correct_direction_rate": steer_by_dim[d]["correct_direction"] / max(steer_by_dim[d]["judgeable"], 1) if steer_by_dim[d]["judgeable"] else None,
            } for d in range(n_targets)
        },
        "seed": args.seed,
    }
    out_path = OUT_DIR / f"eval_airfoil_steerability_{args.out_tag or args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return out


if __name__ == "__main__":
    main()

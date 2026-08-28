"""Generalizes eval_cvae_steerability.py's axis-aligned steerability test
(nudge exactly one target, holding the other 10's *requested* value fixed)
to an arbitrary direction in the full 11-dim target space -- multiple
targets requested to shift at once, which is closer to what "move toward
this new spec" actually looks like in practice than always isolating one
axis. Reuses everything from eval_cvae_steerability.py except the direction
itself: same model loading (load_generative_model), same anchor selection,
same oracle/fidelity/timeout machinery, same baseline-from-own-validity-
candidates discipline, same seeded reproducibility.

Per anchor, --n-directions random unit vectors are drawn in the same
z-scored/log-transformed eval space the model conditions on (seeded off
--seed so this is reproducible the same way the rest of this project now
is -- see eval_cvae_steerability.py's seeding-bug fix). Each direction is
scaled by --step-std and applied to *all* 11 target dims simultaneously
(steer_z = anchor_z + step_std * unit_direction), not just one.

The axis-aligned test's two separate metrics (correct-direction: a sign
check on one axis; selectivity: on-target vs off-target movement) collapse
into one continuous, more general one here: cosine similarity between the
*requested* movement vector (the unit direction, scaled) and the *achieved*
movement vector (the candidate's full measured delta, z-scored) in the
same 11-dim space. 1.0 = moved exactly where asked; 0.0 = moved
orthogonally (no better than an arbitrary unrelated direction); negative =
moved the opposite way. A magnitude ratio (achieved norm / requested norm)
is also tracked, same spirit as the axis-aligned test's own ratio.

--null mirrors eval_cvae_steerability.py's --steer-mode null: candidates are
decoded at the anchor's own *unperturbed* target, but a "requested
direction" is still nominally assigned to each so cosine similarity can be
computed against it -- any similarity above 0 here is pure decode noise,
the chance-level floor this metric needs (an open item eval_cvae_steerability.py
didn't have an answer for; built in from the start here instead of as a
follow-up gap).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import vmec_oracle as oracle
from eval_cvae_steerability import (
    FIDELITY_PRESETS, build_candidates, eval_space, farthest_point_sample,
    load_generative_model, random_anchor_sample,
)
from oracle_harness import run_batch_with_timeout
from train_vae import nfp_one_hot

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", default="cvae", choices=["cvae", "diffusion", "gan"])
    p.add_argument("--tag", default="cvae_targets_full_s0")
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--anchor-selection", default="random", choices=["random", "farthest"])
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--n-directions", type=int, default=11, help="random directions sampled per anchor (default matches the axis-aligned test's 11-per-anchor cost)")
    p.add_argument("--k-per-direction", type=int, default=3)
    p.add_argument("--step-std", type=float, default=1.0, help="norm of the requested move, in the same z-scored/eval space as eval_cvae_steerability.py's --step-std")
    p.add_argument("--null", action="store_true", help="chance-level control: don't actually perturb, just resample at the anchor's own target and score against a nominal direction anyway")
    p.add_argument("--fidelity", default="low", choices=list(FIDELITY_PRESETS))
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default=None)
    return p


def run_eval(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fidelity_name = FIDELITY_PRESETS[args.fidelity]

    dev = torch.device("cpu")
    sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std = \
        load_generative_model(args.model_type, args.tag, dev)

    X = np.load(OUT_DIR / "X.npy")
    Y = np.load(OUT_DIR / "Y.npy")
    Ye = eval_space(Y, target_names)
    Yz = (Ye - t_mean) / t_std

    anchor_sampler = {"random": random_anchor_sample, "farthest": farthest_point_sample}[args.anchor_selection]
    anchor_idx = anchor_sampler(Yz, args.n_anchors, seed=args.seed)
    print(f"[{args.tag}] {args.n_anchors} anchors ({args.anchor_selection}), {args.n_directions} random "
          f"directions each, null={args.null}, fidelity={args.fidelity}={fidelity_name}, step_std={args.step_std}")

    jobs = []  # (tag, *worker_args)
    directions = {}  # (a_i, dir_i) -> unit vector (np.ndarray, n_targets)
    for a_i, idx in enumerate(anchor_idx):
        row = X[idx]
        nfp = int(row[90])
        aux = {"nfp": nfp}

        anchor_z = torch.tensor(Yz[idx:idx + 1], dtype=torch.float32)
        nfp_t = torch.tensor([float(nfp)], dtype=torch.float32)
        cond = torch.cat([anchor_z, nfp_one_hot(nfp_t)], dim=-1)
        for k, cand in enumerate(build_candidates(sample_fn, cond, args.k_validity, coeff_mean, coeff_std, aux, fidelity_name)):
            jobs.append((("validity", a_i, k), *cand))

        for dir_i in range(args.n_directions):
            u = rng.normal(size=n_targets)
            u = u / np.linalg.norm(u)
            directions[(a_i, dir_i)] = u
            steer_z = anchor_z.clone()
            if not args.null:
                steer_z[0] += torch.tensor(args.step_std * u, dtype=torch.float32)
            steer_cond = torch.cat([steer_z, nfp_one_hot(nfp_t)], dim=-1)
            for k, cand in enumerate(build_candidates(sample_fn, steer_cond, args.k_per_direction, coeff_mean, coeff_std, aux, fidelity_name)):
                jobs.append((("steer", a_i, dir_i, k), *cand))

    print(f"[{args.tag}] {len(jobs)} total oracle calls queued "
          f"({args.n_anchors * args.k_validity} validity + "
          f"{args.n_anchors * args.n_directions * args.k_per_direction} steer)")

    results = {}
    n_done, n_converged = 0, 0
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
        n_done += 1
        if ok:
            y = np.array([payload[name] for name in target_names], dtype=np.float64)
            if all(v is not None for v in y) and np.all(np.isfinite(y)):
                results[tag] = y
                n_converged += 1
        if tag not in results:
            results[tag] = None
        if n_done % 40 == 0 or n_done == len(jobs):
            print(f"[{args.tag}] oracle: {n_done}/{len(jobs)} done (hit rate so far {n_converged / n_done:.1%})")

    # ---- assemble ----
    validity_hits, validity_attempts = 0, 0
    n_anchors_with_valid_baseline = 0
    cosine_sims, magnitude_ratios = [], []
    n_steer_attempted, n_steer_converged, n_judgeable = 0, 0, 0
    per_anchor = []

    for a_i, idx in enumerate(anchor_idx):
        validity_ys = [results.get(("validity", a_i, k)) for k in range(args.k_validity)]
        validity_ys = [y for y in validity_ys if y is not None]
        v_hits = len(validity_ys)
        validity_hits += v_hits
        validity_attempts += args.k_validity
        baseline_e = eval_space(np.stack(validity_ys), target_names).mean(axis=0) if validity_ys else None
        if baseline_e is not None:
            n_anchors_with_valid_baseline += 1

        anchor_dirs = []
        for dir_i in range(args.n_directions):
            u = directions[(a_i, dir_i)]
            samples = [results.get(("steer", a_i, dir_i, k)) for k in range(args.k_per_direction)]
            converged = [s for s in samples if s is not None]
            n_steer_attempted += args.k_per_direction
            n_steer_converged += len(converged)
            trial = {"n_converged": len(converged), "cosine_sims": [], "magnitude_ratios": []}
            if baseline_e is not None:
                n_judgeable += len(converged)
                for s in converged:
                    delta_z = (eval_space(s[None, :], target_names)[0] - baseline_e) / t_std
                    denom = np.linalg.norm(delta_z) * np.linalg.norm(u)
                    cos_sim = float(np.dot(delta_z, u) / denom) if denom > 1e-12 else 0.0
                    mag_ratio = float(np.linalg.norm(delta_z) / args.step_std)
                    cosine_sims.append(cos_sim)
                    magnitude_ratios.append(mag_ratio)
                    trial["cosine_sims"].append(cos_sim)
                    trial["magnitude_ratios"].append(mag_ratio)
            anchor_dirs.append(trial)
        per_anchor.append({"row_index": int(idx), "nfp": int(X[idx, 90]),
                            "has_valid_baseline": baseline_e is not None,
                            "validity_hit_rate": v_hits / args.k_validity, "directions": anchor_dirs})

    overall_validity_rate = validity_hits / max(validity_attempts, 1)
    mean_cos = float(np.mean(cosine_sims)) if cosine_sims else float("nan")
    median_cos = float(np.median(cosine_sims)) if cosine_sims else float("nan")
    steer_conv_rate = n_steer_converged / max(n_steer_attempted, 1)

    print(f"\n=== {args.tag}: {args.n_anchors} anchors x {args.n_directions} random directions, "
          f"null={args.null}, fidelity={args.fidelity} ===")
    print(f"  {n_anchors_with_valid_baseline}/{args.n_anchors} anchors have a usable baseline")
    print(f"  validity hit rate (anchor's own spot): {overall_validity_rate:.1%}")
    print(f"  steer-candidate convergence: {steer_conv_rate:.1%} ({n_steer_converged}/{n_steer_attempted})")
    print(f"  mean cosine(achieved, requested) = {mean_cos:.3f}   median = {median_cos:.3f}   "
          f"(1.0 = moved exactly where asked, 0.0 = orthogonal/no better than unrelated, over n={len(cosine_sims)})")

    out = {
        "model_type": args.model_type, "tag": args.tag, "n_anchors": args.n_anchors,
        "n_directions": args.n_directions, "k_per_direction": args.k_per_direction,
        "step_std": args.step_std, "null": args.null, "fidelity": args.fidelity,
        "overall_validity_hit_rate": overall_validity_rate,
        "steer_candidate_convergence_rate": steer_conv_rate,
        "n_anchors_with_valid_baseline": n_anchors_with_valid_baseline,
        "n_judgeable": n_judgeable,
        "mean_cosine_similarity": mean_cos, "median_cosine_similarity": median_cos,
        "mean_magnitude_ratio": float(np.mean(magnitude_ratios)) if magnitude_ratios else None,
        "anchors": per_anchor, "seed": args.seed,
    }
    out_path = OUT_DIR / f"eval_random_direction_{args.out_tag or args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return out


def main():
    args = build_arg_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

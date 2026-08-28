"""A crisper first test for a target-conditioned generative model
(--model-type cvae -> scripts/train_cvae.py, diffusion -> scripts/
train_diffusion.py, or gan -> scripts/train_gan.py, all --source full) than
eval_cvae_generation.py's retrieval comparison: retrieval isn't the thing
being measured here (per explicit user feedback -- it solves a strictly
easier problem, "find the best existing match," not "synthesize something
new," so it isn't a fair competitor and was dropped from this comparison
entirely). All three model types share this exact harness -- same anchors,
same oracle, same fidelity, same null-mode control -- so their numbers are
directly comparable; only `build_candidates`'s `sample_fn` differs (one
decode() call for the VAE, one generator() call for the GAN, vs. a full DDPM
chain for diffusion). This tests two capabilities directly, per anchor
target spec drawn from the real dataset:

  1. Validity at the spot: condition on the anchor's own target vector,
     decode --k-validity candidates, check how many converge through the
     real oracle. Not checking whether a candidate reproduces the anchor's
     own coefficients -- that's not the question, and was never the point
     (that's what --k-validity>1 stochastic decodes are for: several
     different plausible answers to the same target request, not a
     reconstruction target).

  2. Steerability: for each of the 11 targets in turn, nudge *only* that
     target's conditioning value by --step-std (in the same z-scored/
     log-transformed eval space train_cvae.py conditions on) away from the
     anchor, holding every other target and nfp fixed, decode
     --k-steer candidates, validate through the oracle, and check whether
     the *measured* value of that one target actually moved in the
     requested direction (and by how much of the requested magnitude) --
     not whether the design is "better," purely whether the model's
     conditioning is causally hooked up to what comes out. Also tracks
     *selectivity*: every converged steer candidate's full 11-dim measured
     vector is compared against baseline_e on every target, not just the
     requested one, z-scored so all 11 targets are in comparable units --
     "on-target |z-delta|" (the requested dim) vs. "off-target |z-delta|"
     (mean over the other 10) isolates whether nudging one target moves
     that target *selectively*, or drags the whole vector along with it.
     A selectivity ratio near 1 means no selectivity at all (on-target and
     off-target movement are the same size); higher is better isolation.

No train/test split, no held-out target region -- this isn't measuring
generalization to an unseen region (see make_splits.py's `target-cluster`
mode and eval_cvae_generation.py for that, a separate, harder question for
later). Anchors are instead chosen to *spread out* across target space
(farthest-point sampling in the same z-scored eval space, greedy max-min
distance) so a handful of random draws can't all land in the same dominant
dense region by chance.

Every oracle call uses --fidelity, default `medium` (`from_boundary_resolution`,
VMEC++'s 25->51->99 multigrid preset) rather than the `low` preset (25->71)
that generated the dataset and was used by every prior oracle call in this
project (§8-§16) -- genuinely more resolved, at ~2.8x the per-call cost
(measured: ~17s low vs ~48s medium on a real row) but still finishes a
few-hundred-call run in minutes given parallel workers.

The steerability baseline is the mean of the anchor's own converged
--k-validity candidates (decoded by this model, validated by *this* oracle
at *this* fidelity) -- NOT a re-verification of the anchor's original real
design. An earlier version of this script did the latter and it was a real
mistake, caught by a direct "isn't VMEC++ deterministic?" question: X.npy's
r_cos/z_sin come straight from the upstream proxima-fusion/constellaration
HuggingFace dataset's own precomputed metrics (preprocess.py never re-runs
VMEC++ on them at all), only float32-downcast for storage here. Re-running
that already-float32-truncated boundary through our oracle is deterministic
(confirmed directly: identical failure, same error, same timing, on repeat
calls) but is not reproducing whatever settings/version/precision Proxima
Fusion actually used -- and for a design sitting right at VMEC's force-
tolerance edge (1e-17 at the first multigrid stage), a rounding-level input
difference is enough to flip convergence outright. §8 already saw hints of
this (up to 6% metric disagreement on a row that *did* reconverge); this
script's first version ran into the failure-mode version of the same gap
and briefly misread it as a physical "fragile edge of feasibility" finding
about the anchors themselves, rather than a reproduction artifact. Anchoring
the baseline to our own validity candidates instead removes the dependency
on reproducing Proxima Fusion's numbers entirely -- self-consistent within
this script's own model+oracle+fidelity, and it only needs *one* of
--k-validity candidates to converge rather than one specific (possibly
marginal, foreign-pipeline) real design.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout
from train_cvae import CVAE
from train_diffusion import DiffusionDenoiser, ddpm_sample, make_schedule
from train_gan import Generator as GANGenerator
from train_vae import nfp_one_hot

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

# Kept as module-level aliases so the rest of this file (and EXPERIMENT_LOG
# references to e.g. FIDELITY_PRESETS) doesn't need to change -- the actual
# domain knowledge now lives in vmec_oracle.py (see oracle_base.py for the
# interface any other domain would implement instead).
LOG_TARGET_NAMES = oracle.LOG_TARGET_NAMES
FIDELITY_PRESETS = oracle.FIDELITY_PRESETS


def eval_space(Y, target_names):
    Ye = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Ye[:, idx] = np.log(np.clip(Ye[:, idx], 1e-12, None))
    return Ye


def random_anchor_sample(points, n, seed=0):
    """Plain uniform draw, no replacement -- the default. `points` isn't
    used beyond len() (kept for a matching signature with
    farthest_point_sample)."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(points), size=n, replace=False)


def farthest_point_sample(points, n, seed=0):
    """Greedy max-min-distance selection: spreads the n anchors across
    `points` instead of picking them uniformly at random. NOT the default --
    confirmed (see EXPERIMENT_LOG §17) that this preferentially selects
    dataset outliers/extremes (maximizing spread does exactly that), which
    biases both validity and steerability low for reasons that have nothing
    to do with the model -- appropriate for a later, deliberately-harder
    stress test, not for asking "can it do this at all" on representative
    target specs."""
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(points)))]
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(selected)


def load_generative_model(model_type, tag, dev):
    """Loads a checkpoint and returns (sample_fn, target_names, n_targets,
    coeff_mean, coeff_std, t_mean, t_std) -- the one dispatch point for
    "which architecture is this," factored out so other scripts (e.g.
    eval_random_direction_steerability.py) don't reimplement the
    cVAE/diffusion/GAN loading logic. sample_fn(cond, k) -> (k, PARAM_DIM)
    numpy array of standardized params, regardless of model type."""
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    target_names = ckpt["target_names"]
    n_targets = len(target_names)
    coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]

    if model_type == "cvae":
        model = CVAE(coeff_dim=90, n_targets=n_targets, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        latent_dim = ckpt["latent_dim"]

        def sample_fn(cond, k):
            with torch.no_grad():
                z = torch.randn(k, latent_dim)
                return model.decode(z, cond.repeat(k, 1)).numpy()
    elif model_type == "diffusion":
        model = DiffusionDenoiser(coeff_dim=90, n_targets=n_targets, hidden=ckpt["hidden"],
                                   time_embed_dim=ckpt["time_embed_dim"]).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        schedule = make_schedule(ckpt["T"], device=dev)

        def sample_fn(cond, k):
            return ddpm_sample(model, cond.repeat(k, 1), schedule, device=dev).numpy()
    else:
        model = GANGenerator(coeff_dim=90, n_targets=n_targets, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
        model.load_state_dict(ckpt["generator_state_dict"])
        model.eval()
        latent_dim = ckpt["latent_dim"]

        def sample_fn(cond, k):
            with torch.no_grad():
                z = torch.randn(k, latent_dim)
                return model(z, cond.repeat(k, 1)).numpy()

    return sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std


def build_candidates(sample_fn, cond, k, coeff_mean, coeff_std, aux, fidelity_name):
    """`sample_fn(cond, k)` -> (k, PARAM_DIM) numpy array of STANDARDIZED
    params -- the one thing that differs between generative model types
    (cVAE: one decode(z, cond) call; diffusion: a full ddpm_sample chain).
    Everything downstream (unstandardize, zero the structurally-fixed
    params, hand off to the oracle's own params_to_worker_args) is both
    model-agnostic AND domain-agnostic -- oracle.ZERO_INDICES and
    oracle.params_to_worker_args are the only domain-specific pieces, both
    delegated to vmec_oracle.py (or whatever other oracle module a future
    domain provides -- see oracle_base.py)."""
    decoded = sample_fn(cond, k)
    params = decoded * coeff_std + coeff_mean
    params[:, oracle.ZERO_INDICES] = 0.0
    return [oracle.params_to_worker_args(params[i], aux, fidelity_name) for i in range(k)]


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", default="cvae", choices=["cvae", "diffusion", "gan"])
    p.add_argument("--tag", default="cvae_targets_full_s0", help="checkpoint tag; defaults assume --model-type cvae's naming")
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--anchor-selection", default="random", choices=["random", "farthest"],
                    help="'random' (default): representative, unbiased target specs -- the right choice "
                         "for 'can it do this at all'. 'farthest': deliberately spread to dataset extremes, "
                         "confirmed to bias validity/steerability low for reasons unrelated to the model "
                         "(see EXPERIMENT_LOG §17) -- a harder stress test for later, not a first read.")
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--k-steer", type=int, default=3)
    p.add_argument("--step-std", type=float, default=1.0, help="z-scored/eval-space decrease applied to one target dim at a time")
    p.add_argument("--steer-mode", default="targeted", choices=["targeted", "null"],
                    help="'targeted' (default): actually nudge the requested target dimension. "
                         "'null': resample --k-steer extra candidates at the SAME (unperturbed) anchor "
                         "target instead -- everything else identical (same baseline, same 'correct "
                         "direction' bookkeeping) -- a chance-level control. Any apparent directionality "
                         "here is pure decode-to-decode noise, not response to a request; run this at the "
                         "same anchors/seed as a 'targeted' run to see how much of that run's "
                         "correct-direction rate clears the noise floor.")
    p.add_argument("--fidelity", default="medium", choices=list(FIDELITY_PRESETS))
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default=None)
    return p


def run_eval(args):
    """Callable core, separate from the CLI entrypoint, so an orchestrator
    (e.g. run_ablation.py) can invoke many configs in-process without
    shelling out to a fresh docker/python process per config -- `args` is
    any object with the attributes build_arg_parser() defines (an
    argparse.Namespace from the CLI, or a plain object/SimpleNamespace built
    programmatically). Returns the same dict that gets written to
    out_path -- the orchestrator can use it directly instead of re-reading
    its own JSON back off disk."""
    # NOTE: --seed previously only drove anchor selection (np.random) -- the
    # actual stochastic decoding (every torch.randn(...) call in sample_fn)
    # was never seeded, so "the same config, same seed" silently decoded
    # different candidates every run. Caught the hard way: a null-mode rerun
    # of the "same" config gave a materially different chance-level number
    # for cVAE and diffusion (see EXPERIMENT_LOG's correction to §18's
    # premature "opposite-sign pattern" claim). Seeding torch here too makes
    # --seed actually reproduce a full run, not just which anchors get used.
    torch.manual_seed(args.seed)
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
    print(f"[{args.tag}] {args.n_anchors} anchors ({args.anchor_selection}) selected from {len(X)} real rows "
          f"(fidelity={args.fidelity}={fidelity_name}, k_validity={args.k_validity}, "
          f"k_steer={args.k_steer}, step_std={args.step_std})")

    # ---- build every oracle call up front, tagged, so one shared parallel
    # batch does the baseline re-verify + validity candidates + steer
    # candidates for every anchor at once (instead of n_anchors serial batches) ----
    jobs = []  # (tag, *worker_args) -- worker_args shape is oracle-specific, opaque to everything below
    for a_i, idx in enumerate(anchor_idx):
        row = X[idx]
        nfp = int(row[90])
        aux = {"nfp": nfp}
        real_params = row[:90].astype(np.float64)
        jobs.append((("baseline", a_i, None), *oracle.params_to_worker_args(real_params, aux, fidelity_name)))

        anchor_z = torch.tensor(Yz[idx:idx + 1], dtype=torch.float32)
        nfp_t = torch.tensor([float(nfp)], dtype=torch.float32)
        cond = torch.cat([anchor_z, nfp_one_hot(nfp_t)], dim=-1)
        for k, cand in enumerate(build_candidates(sample_fn, cond, args.k_validity, coeff_mean, coeff_std, aux, fidelity_name)):
            jobs.append((("validity", a_i, k), *cand))

        for d in range(n_targets):
            steer_z = anchor_z.clone()
            if args.steer_mode == "targeted":
                steer_z[0, d] -= args.step_std
            steer_cond = torch.cat([steer_z, nfp_one_hot(nfp_t)], dim=-1)
            for k, cand in enumerate(build_candidates(sample_fn, steer_cond, args.k_steer, coeff_mean, coeff_std, aux, fidelity_name)):
                jobs.append((("steer", a_i, d, k), *cand))

    print(f"[{args.tag}] {len(jobs)} total oracle calls queued "
          f"({args.n_anchors} real-design re-verify [diagnostic only, not used as the steer baseline] + "
          f"{args.n_anchors * args.k_validity} validity + {args.n_anchors * n_targets * args.k_steer} steer)")

    results = {}  # tag -> Y array or None
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

    # ---- assemble per-anchor / per-direction results ----
    # NOTE: correct_direction/magnitude_ratios are measured against the mean
    # of the anchor's own converged --k-validity candidates, NOT a
    # re-verification of the anchor's original real design (see module
    # docstring: that real row's stored metrics come from an unreproduced
    # upstream pipeline, and re-verifying it against our own oracle
    # conflates "did this foreign design happen to reconverge here" with
    # the actual question). "real_design_reconverged" is recorded purely as
    # a diagnostic. "judgeable" requires >=1 validity candidate converged
    # (to have a baseline at all) -- a much weaker requirement than the
    # anchor's single specific original row reconverging.
    anchors_out = []
    steer_by_dim = {d: {"attempted": 0, "converged": 0, "judgeable": 0, "correct_direction": 0,
                         "magnitude_ratios": [], "on_target_abs_z": [], "off_target_abs_z": []}
                     for d in range(n_targets)}
    validity_hits, validity_attempts = 0, 0
    n_anchors_with_real_design_reconverged = 0
    n_anchors_with_valid_baseline = 0

    for a_i, idx in enumerate(anchor_idx):
        real_design_y = results.get(("baseline", a_i, None))
        if real_design_y is not None:
            n_anchors_with_real_design_reconverged += 1

        validity_ys = [results.get(("validity", a_i, k)) for k in range(args.k_validity)]
        validity_ys = [y for y in validity_ys if y is not None]
        v_hits = len(validity_ys)
        validity_hits += v_hits
        validity_attempts += args.k_validity
        baseline_e = eval_space(np.stack(validity_ys), target_names).mean(axis=0) if validity_ys else None
        if baseline_e is not None:
            n_anchors_with_valid_baseline += 1

        directions = []
        for d in range(n_targets):
            requested_delta = -args.step_std * t_std[d]
            samples = [results.get(("steer", a_i, d, k)) for k in range(args.k_steer)]
            converged = [s for s in samples if s is not None]
            steer_by_dim[d]["attempted"] += args.k_steer
            steer_by_dim[d]["converged"] += len(converged)
            correct = 0
            ratios = []
            on_target_zs, off_target_zs = [], []
            if baseline_e is not None:
                steer_by_dim[d]["judgeable"] += len(converged)
                for s in converged:
                    # Full delta vector (all n_targets dims), z-scored by each
                    # target's own t_std so on-target and off-target movement
                    # are in comparable units (raw eval-space units aren't --
                    # e.g. qi deltas are O(0.01-0.05), aspect_ratio O(1-2)).
                    delta_z = (eval_space(s[None, :], target_names)[0] - baseline_e) / t_std
                    achieved_delta = delta_z[d] * t_std[d]  # back to eval-space for requested_delta comparability
                    if achieved_delta < 0:
                        correct += 1
                    ratios.append(float(achieved_delta / requested_delta))
                    on_target_zs.append(float(abs(delta_z[d])))
                    off_target_zs.append(float(np.mean(np.abs(np.delete(delta_z, d)))))
                steer_by_dim[d]["correct_direction"] += correct
                steer_by_dim[d]["magnitude_ratios"].extend(ratios)
                steer_by_dim[d]["on_target_abs_z"].extend(on_target_zs)
                steer_by_dim[d]["off_target_abs_z"].extend(off_target_zs)
            directions.append({
                "target": target_names[d], "requested_delta_eval_space": float(requested_delta),
                "n_converged": len(converged), "n_correct_direction": correct,
                "achieved_deltas_eval_space": [float(eval_space(s[None, :], target_names)[0][d] - baseline_e[d]) for s in converged] if baseline_e is not None else [],
                "on_target_abs_z": on_target_zs, "off_target_mean_abs_z": off_target_zs,
            })

        anchors_out.append({
            "row_index": int(idx), "nfp": int(X[idx, 90]),
            "real_design_reconverged_at_this_fidelity_diagnostic_only": real_design_y is not None,
            "has_valid_baseline": baseline_e is not None,
            "validity_hit_rate": v_hits / args.k_validity,
            "directions": directions,
        })

    overall_validity_rate = validity_hits / max(validity_attempts, 1)
    print(f"\n=== {args.tag}: {args.n_anchors} anchors, fidelity={args.fidelity} ===")
    print(f"  [diagnostic] {n_anchors_with_real_design_reconverged}/{args.n_anchors} anchors' ORIGINAL real "
          f"design (foreign pipeline, not used as the baseline) re-converged under our oracle at this fidelity")
    print(f"  {n_anchors_with_valid_baseline}/{args.n_anchors} anchors have a usable baseline "
          f"(>=1 of {args.k_validity} validity candidates converged)")
    print(f"  validity hit rate (candidates at the anchor's own target spot): {overall_validity_rate:.1%} "
          f"({validity_hits}/{validity_attempts})")
    print(f"\n  steerability by target ('correct dir' is out of judgeable candidates only -- baseline-valid anchors):")
    print(f"  {'target':55s} {'converged':>10s} {'judgeable':>10s} {'correct dir':>12s} {'median |achieved/requested|':>28s} {'on-target |z|':>14s} {'off-target |z|':>15s} {'selectivity':>12s}")
    all_on, all_off = [], []
    for d in range(n_targets):
        s = steer_by_dim[d]
        conv_rate = s["converged"] / max(s["attempted"], 1)
        correct_rate = s["correct_direction"] / max(s["judgeable"], 1) if s["judgeable"] else float("nan")
        med_ratio = float(np.median(np.abs(s["magnitude_ratios"]))) if s["magnitude_ratios"] else float("nan")
        on_med = float(np.median(s["on_target_abs_z"])) if s["on_target_abs_z"] else float("nan")
        off_med = float(np.median(s["off_target_abs_z"])) if s["off_target_abs_z"] else float("nan")
        selectivity = on_med / off_med if off_med and not np.isnan(off_med) else float("nan")
        all_on.extend(s["on_target_abs_z"]); all_off.extend(s["off_target_abs_z"])
        print(f"  {target_names[d]:55s} {conv_rate:9.1%}  {s['judgeable']:10d}  {correct_rate:11.1%}  {med_ratio:27.2f}  {on_med:13.2f}  {off_med:14.2f}  {selectivity:11.2f}")
    overall_on = float(np.median(all_on)) if all_on else float("nan")
    overall_off = float(np.median(all_off)) if all_off else float("nan")
    print(f"\n  overall: median on-target |z-delta|={overall_on:.2f}, median off-target mean|z-delta|={overall_off:.2f}, "
          f"selectivity={overall_on/overall_off if overall_off else float('nan'):.2f} "
          f"(>1 means the requested target moved more than the average unrequested one; 1 means no selectivity at all)")

    out = {
        "model_type": args.model_type, "tag": args.tag, "n_anchors": args.n_anchors, "k_validity": args.k_validity, "k_steer": args.k_steer,
        "step_std": args.step_std, "steer_mode": args.steer_mode, "fidelity": args.fidelity, "fidelity_preset": fidelity_name,
        "overall_validity_hit_rate": overall_validity_rate,
        "n_anchors_with_valid_baseline": n_anchors_with_valid_baseline,
        "n_anchors_with_real_design_reconverged_diagnostic_only": n_anchors_with_real_design_reconverged,
        "overall_selectivity": {
            "median_on_target_abs_z": overall_on, "median_off_target_mean_abs_z": overall_off,
            "selectivity_ratio": overall_on / overall_off if overall_off else None,
        },
        "steer_by_target": {
            target_names[d]: {
                "convergence_rate": steer_by_dim[d]["converged"] / max(steer_by_dim[d]["attempted"], 1),
                "n_judgeable": steer_by_dim[d]["judgeable"],
                "correct_direction_rate": steer_by_dim[d]["correct_direction"] / max(steer_by_dim[d]["judgeable"], 1) if steer_by_dim[d]["judgeable"] else None,
                "median_abs_magnitude_ratio": float(np.median(np.abs(steer_by_dim[d]["magnitude_ratios"]))) if steer_by_dim[d]["magnitude_ratios"] else None,
                "median_on_target_abs_z": float(np.median(steer_by_dim[d]["on_target_abs_z"])) if steer_by_dim[d]["on_target_abs_z"] else None,
                "median_off_target_mean_abs_z": float(np.median(steer_by_dim[d]["off_target_abs_z"])) if steer_by_dim[d]["off_target_abs_z"] else None,
            } for d in range(n_targets)
        },
        "anchors": anchors_out,
        "seed": args.seed,
    }
    out_path = OUT_DIR / f"eval_cvae_steerability_{args.out_tag or args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return out


def main():
    args = build_arg_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

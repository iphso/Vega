"""Evaluates a target-conditioned generation baseline (scripts/train_cvae.py)
against the real VMEC++ oracle, on target specs drawn from
output/splits/target_cluster/test.npz -- entire target-space clusters the
model never trained on (see make_splits.py's `target-cluster` mode), the
generation-direction analogue of §6's cluster-split test for the forward
(design -> metrics) surrogate.

For each of --n-specs test rows (used only for its (Y, nfp) -- its own X is
never given to the model, and is used only as a reference to confirm the
row's own metrics reproduce under the oracle, matching §8's validation-first
discipline): sample --k candidates from the cVAE decoder conditioned on that
row's target vector, validate every candidate through the real VMEC++ oracle
in parallel (reused unmodified from generate_and_validate.py -- same
subprocess-per-candidate + hard timeout approach, since the same "VMEC++ can
hang instead of failing fast" risk applies to any candidate it hasn't seen,
generated or not). Reports:

  hit rate           -- fraction of sampled candidates that converge at all.
  target error        -- among converged candidates, per-target relative
                          error between the oracle's measured metrics and
                          the spec's requested metrics, in eval space (log
                          space for LOG_TARGET_NAMES, physical units for the
                          rest -- same convention screen_reference_baselines
                          uses, for the same reason: physical-unit error on
                          e.g. max_elongation is dominated by rare outliers).
                          Best-of-k per spec (the closest of the k converged
                          candidates), then averaged across specs.

A zero-training-cost nearest-neighbor retrieval baseline is reported
alongside every run: for each spec, the closest train-split row by the same
z-scored target distance the clusters were built from. Its "target error" is
exact (it's an already-oracle-validated real design, not re-run), so it's a
floor the generative model needs to beat to be worth anything -- retrieval
has a trivial ceiling on plausibility (it always returns a real, feasible
design) but a hard floor on precision (it can't get closer than the nearest
existing point in a finite dataset).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_and_validate import ZERO_COEFF_IDX, run_batch_with_timeout
from make_splits import LOG_TARGET_NAMES
from train_cvae import CVAE
from train_vae import nfp_one_hot

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def eval_space(Y, target_names):
    """log space for LOG_TARGET_NAMES, physical units otherwise -- matches
    screen_reference_baselines.py's ranking convention (see EXPERIMENT_LOG §15:
    physical-unit RMSE on these 4 targets is dominated by rare huge-value
    outliers and doesn't differentiate models at all)."""
    Ye = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Ye[:, idx] = np.log(np.clip(Ye[:, idx], 1e-12, None))
    return Ye


def relative_error(measured_e, requested_e):
    return np.abs(measured_e - requested_e) / np.maximum(np.abs(requested_e), 1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="cvae_targets_s0")
    p.add_argument("--split", default="target_cluster")
    p.add_argument("--n-specs", type=int, default=40)
    p.add_argument("--k", type=int, default=5, help="candidates sampled per spec")
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default=None, help="results json filename; defaults to --tag")
    args = p.parse_args()

    dev = torch.device("cpu")
    ckpt = torch.load(CKPT_DIR / f"{args.tag}.pt", map_location=dev)
    target_names = ckpt["target_names"]
    model = CVAE(coeff_dim=90, n_targets=len(target_names), latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]

    train_npz = np.load(OUT_DIR / "splits" / args.split / "train.npz")
    test_npz = np.load(OUT_DIR / "splits" / args.split / "test.npz")
    X_train, Y_train = train_npz["X"], train_npz["Y"]
    X_test, Y_test = test_npz["X"], test_npz["Y"]

    rng = np.random.default_rng(args.seed)
    spec_idx = rng.choice(len(X_test), size=min(args.n_specs, len(X_test)), replace=False)
    print(f"[{args.tag}] evaluating {len(spec_idx)} target specs from {args.split}/test.npz "
          f"({len(X_test)} available), k={args.k} candidates each, against train={len(X_train)}")

    # --- nearest-neighbor retrieval baseline (train split, z-scored target space) ---
    Ye_train = eval_space(Y_train, target_names)
    Ye_test = eval_space(Y_test, target_names)
    train_z = (Ye_train - t_mean) / t_std
    test_z = (Ye_test[spec_idx] - t_mean) / t_std
    dists = np.linalg.norm(train_z[None, :, :] - test_z[:, None, :], axis=-1)
    nn_idx = dists.argmin(axis=1)
    nn_err = relative_error(Ye_train[nn_idx], Ye_test[spec_idx])
    nn_median_per_spec = np.median(nn_err, axis=1)

    # --- cVAE candidate generation ---
    Y_spec = Y_test[spec_idx]
    nfp_spec = X_test[spec_idx, 90]
    cond_targets = (eval_space(Y_spec, target_names) - t_mean) / t_std
    cond_t = torch.tensor(cond_targets, dtype=torch.float32)
    nfp_t = torch.tensor(nfp_spec, dtype=torch.float32)

    candidates = []  # (spec_i, r_cos, z_sin, nfp)
    with torch.no_grad():
        for i in range(len(spec_idx)):
            cond = torch.cat([cond_t[i:i + 1], nfp_one_hot(nfp_t[i:i + 1])], dim=-1).repeat(args.k, 1)
            z = torch.randn(args.k, ckpt["latent_dim"])
            decoded = model.decode(z, cond).numpy()
            coeffs = decoded * coeff_std + coeff_mean
            coeffs[:, ZERO_COEFF_IDX] = 0.0
            for k in range(args.k):
                r_cos = coeffs[k, :45].reshape(5, 9).astype(np.float64)
                z_sin = coeffs[k, 45:90].reshape(5, 9).astype(np.float64)
                candidates.append((i, r_cos, z_sin, int(round(nfp_spec[i]))))

    per_spec_results = {i: [] for i in range(len(spec_idx))}
    n_converged, n_attempted = 0, 0
    vmec_candidates = [(r_cos, z_sin, nfp) for _, r_cos, z_sin, nfp in candidates]
    for (spec_i, _, _, _), (ok, r_cos, z_sin, nfp, payload) in zip(
        candidates, run_batch_with_timeout(vmec_candidates, args.n_workers, args.timeout_seconds)
    ):
        n_attempted += 1
        if ok:
            y = np.array([payload[name] for name in target_names], dtype=np.float64)
            if all(v is not None for v in y) and np.all(np.isfinite(y)):
                n_converged += 1
                per_spec_results[spec_i].append(y)
        if n_attempted % 20 == 0 or n_attempted == len(candidates):
            print(f"[{args.tag}] validated {n_attempted}/{len(candidates)} "
                  f"(hit rate so far {n_converged / n_attempted:.1%})")

    best_of_k_err, hit_any = [], []
    for i in range(len(spec_idx)):
        ys = per_spec_results[i]
        hit_any.append(len(ys) > 0)
        if not ys:
            continue
        Ye_i = eval_space(np.stack(ys), target_names)
        req_e = eval_space(Y_spec[i:i + 1], target_names)[0]
        errs = relative_error(Ye_i, req_e[None, :])
        median_per_candidate = np.median(errs, axis=1)
        best_of_k_err.append(float(median_per_candidate.min()))

    hit_rate = n_converged / max(n_attempted, 1)
    spec_hit_rate = float(np.mean(hit_any))
    cvae_mean_err = float(np.mean(best_of_k_err)) if best_of_k_err else None
    nn_mean_err = float(np.mean(nn_median_per_spec))

    print(f"\n=== {args.tag} vs. nearest-neighbor retrieval, {len(spec_idx)} specs, k={args.k} ===")
    print(f"  cVAE candidate hit rate (of all {n_attempted} sampled): {hit_rate:.1%}")
    print(f"  cVAE spec-level hit rate (>=1 of {args.k} converged):    {spec_hit_rate:.1%}")
    print(f"  cVAE best-of-k median-relative-target-error (converged specs only, n={len(best_of_k_err)}): "
          f"{cvae_mean_err if cvae_mean_err is not None else float('nan'):.4f}")
    print(f"  NN-retrieval median-relative-target-error (all {len(spec_idx)} specs, always feasible): "
          f"{nn_mean_err:.4f}")

    out = {
        "tag": args.tag, "split": args.split, "n_specs": len(spec_idx), "k": args.k,
        "n_attempted": n_attempted, "n_converged": n_converged, "hit_rate": hit_rate,
        "spec_hit_rate": spec_hit_rate, "cvae_best_of_k_mean_target_error": cvae_mean_err,
        "cvae_best_of_k_per_spec_errors": best_of_k_err,
        "nn_retrieval_mean_target_error": nn_mean_err,
        "nn_retrieval_per_spec_errors": nn_median_per_spec.tolist(),
        "seed": args.seed,
    }
    out_path = OUT_DIR / f"eval_cvae_generation_{args.out_tag or args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

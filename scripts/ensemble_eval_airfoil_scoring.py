"""Airfoil analogue of ensemble_eval.py (VMEC++, §4): evaluate an ensemble
of independently-seeded train_airfoil_scoring.py checkpoints (same
architecture, different --seed) against the held-out test set -- ensemble-
averaged RMSE vs. a single member, plus the same inter-member-spread
calibration check.

Assumes all member tags share dataset-tag/split/normalization stats (only
the first member's in_mean/in_std/target_mean/target_std are used to
un-normalize -- true whenever all members were trained with the same
--seed-independent split, i.e. --source split with the same --split name,
or --source random with the *same* --seed used only for shuffling since the
train/val/test row indices for a given seed are deterministic; mixing
--source random runs with different seeds would use different test sets
per member and silently misalign rows -- not guarded against here, matching
ensemble_eval.py's own lack of a guard, since this project's ensembles have
always been "same split, different init seed" by construction).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from train_airfoil_scoring import CKPT_DIR, OUT_DIR, ScoringMLP, ScoringTrunkModel


def load_model(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    trunk_arch = ckpt.get("trunk_arch", "legacy")
    if trunk_arch == "legacy":
        model = ScoringMLP(in_dim=ckpt["in_dim"], n_targets=len(ckpt["target_names"]),
                            hidden=ckpt["hidden"], n_blocks=ckpt["n_blocks"]).to(dev)
    else:
        model = ScoringTrunkModel(in_dim=ckpt["in_dim"], n_targets=len(ckpt["target_names"]),
                                   trunk_arch=trunk_arch, hidden=ckpt["hidden"],
                                   latent_dim=ckpt.get("latent", 128), n_blocks=ckpt["n_blocks"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_test_set(ckpt):
    """Reproduces train_airfoil_scoring.py's own filter+split+normalize
    logic exactly, using the checkpoint's own recorded stats (never
    recomputed) so this evaluation can't leak or drift from what the model
    was actually trained/normalized against."""
    target_names = ckpt["target_names"]
    log_target_names = ckpt["log_target_names"]
    cd_col, lod_col = target_names.index("cd"), target_names.index("l_over_d")

    if ckpt["split"] == "split":
        split_dir = OUT_DIR / "splits" / ckpt["split_name"]
        test_npz = np.load(split_dir / "test.npz")
        X_test, Y_test = test_npz["X"], test_npz["Y"]
    else:
        X = np.load(OUT_DIR / f"{ckpt['dataset_tag']}_X.npy")
        Y = np.load(OUT_DIR / f"{ckpt['dataset_tag']}_Y.npy")
        sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
        X, Y = X[sane], Y[sane]
        n = len(X)
        rng = np.random.default_rng(ckpt["seed"])
        perm = rng.permutation(n)
        n_val = int(n * ckpt["val_frac"])
        n_test = int(n * ckpt["test_frac"])
        test_idx = perm[:n_test]
        X_test, Y_test = X[test_idx], Y[test_idx]

    return X_test, Y_test, target_names, log_target_names


def rmse_table(label, rmse, target_names):
    print(f"  {label}:")
    for name, v in zip(target_names, rmse):
        print(f"    {name:12s} {v:.5f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member-tags", nargs="+", required=True)
    p.add_argument("--baseline-tag", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    ref_ckpt = torch.load(CKPT_DIR / f"{args.member_tags[0]}.pt", map_location=dev)
    X_test, Y_test, target_names, log_target_names = load_test_set(ref_ckpt)
    in_mean, in_std = ref_ckpt["in_mean"], ref_ckpt["in_std"]
    x_test = torch.tensor((X_test - in_mean) / in_std, dtype=torch.float32, device=dev)

    def unnormalize(pred_z, ckpt):
        pred = pred_z * ckpt["target_std"] + ckpt["target_mean"]
        for name in log_target_names:
            idx = target_names.index(name)
            pred[:, idx] = np.exp(pred[:, idx])
        return pred

    preds = []
    for tag in args.member_tags:
        model, ckpt = load_model(tag, dev)
        with torch.no_grad():
            pred_z = model(x_test).cpu().numpy()
        preds.append(unnormalize(pred_z, ckpt))
    preds = np.stack(preds, axis=0)  # (M, N, T)

    ensemble_pred = preds.mean(axis=0)
    ensemble_rmse = np.sqrt(((ensemble_pred - Y_test) ** 2).mean(axis=0))
    single_rmse = np.sqrt(((preds[0] - Y_test) ** 2).mean(axis=0))

    print(f"\n=== ensemble of {len(args.member_tags)}: {args.member_tags} ===")
    rmse_table("ensemble-averaged test RMSE", ensemble_rmse, target_names)
    rmse_table(f"single member ({args.member_tags[0]}) test RMSE", single_rmse, target_names)

    if args.baseline_tag:
        base_model, base_ckpt = load_model(args.baseline_tag, dev)
        with torch.no_grad():
            base_pred_z = base_model(x_test).cpu().numpy()
        base_pred = unnormalize(base_pred_z, base_ckpt)
        base_rmse = np.sqrt(((base_pred - Y_test) ** 2).mean(axis=0))
        rmse_table(f"baseline ({args.baseline_tag}) test RMSE", base_rmse, target_names)

    # Calibration: does inter-member spread correlate with actual |error|?
    spread = preds.std(axis=0)  # (N, T)
    abs_err = np.abs(ensemble_pred - Y_test)
    print("\n  calibration check (mean |error| by inter-member spread quintile, low -> high):")
    n = Y_test.shape[0]
    for k, name in enumerate(target_names):
        order = np.argsort(spread[:, k])
        bucket_means = [abs_err[order[q * n // 5:(q + 1) * n // 5], k].mean() for q in range(5)]
        arrow = "monotonic" if all(bucket_means[i] <= bucket_means[i + 1] + 1e-12 for i in range(4)) else "not monotonic"
        print(f"    {name:12s} " + "  ".join(f"{m:9.4g}" for m in bucket_means) + f"   [{arrow}]")


if __name__ == "__main__":
    main()

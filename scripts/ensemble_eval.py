"""Evaluate an ensemble of independently-seeded regression checkpoints against
the held-out test set: per-target RMSE of the averaged prediction, compared
to a single ensemble member and to the small (230K-param) baseline. Also
checks calibration -- whether inter-member prediction spread (a free
uncertainty signal from ensembling) actually correlates with real error,
since that's what would let a surrogate-guided search decide when to trust
a prediction versus fall back to the real solver.
"""
import argparse
from pathlib import Path

import torch

from train import CKPT_DIR, OUT_DIR, DualPathMLP, load_split


def load_model(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"],
        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"],
        spatial_latent=ckpt["spatial_latent"], head_hidden=ckpt["head_hidden"],
        priority_weight=ckpt["priority_weight"], use_spatial=ckpt["use_spatial"],
        trunk_arch=ckpt.get("trunk_arch", "mlp"), trunk_blocks=ckpt.get("trunk_blocks", 3),
        use_symlog_latent=ckpt.get("use_symlog_latent", False),
        log_target_mask=ckpt.get("log_target_mask"),
        objective=ckpt.get("objective", "regression"),
    ).to(dev)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not result.unexpected_keys
    model.eval()
    return model, ckpt["target_names"]


def rmse_table(label, mse, target_names):
    print(f"  {label}:")
    for name, m in zip(target_names, mse):
        print(f"    {name:55s} {m.sqrt().item():12.5g}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member-tags", nargs="+", required=True,
                    help="checkpoint stems for the ensemble members (same architecture, different seeds)")
    p.add_argument("--baseline-tag", default=None,
                    help="optional single small-model checkpoint stem to compare against")
    p.add_argument("--split", default=None, choices=["random", "group", "cluster"],
                    help="evaluate against output/splits/<split>/test.npz instead of the default "
                         "test set (see scripts/make_splits.py)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    data_dir = (OUT_DIR / "splits" / args.split) if args.split else None
    X_test, Y_test = load_split("test", data_dir)
    X_test, Y_test = X_test.to(dev), Y_test.to(dev)

    preds = []
    target_names = None
    for tag in args.member_tags:
        model, target_names = load_model(tag, dev)
        with torch.no_grad():
            preds.append(model(X_test))
    preds = torch.stack(preds, dim=0)  # (M, N, T)

    ensemble_pred = preds.mean(dim=0)  # (N, T)
    ensemble_mse = ((ensemble_pred - Y_test) ** 2).mean(dim=0)
    single_mse = ((preds[0] - Y_test) ** 2).mean(dim=0)

    print(f"\n=== ensemble of {len(args.member_tags)}: {args.member_tags} ===")
    rmse_table("ensemble-averaged test RMSE", ensemble_mse, target_names)
    rmse_table(f"single member ({args.member_tags[0]}) test RMSE", single_mse, target_names)

    if args.baseline_tag:
        base_model, base_names = load_model(args.baseline_tag, dev)
        with torch.no_grad():
            base_pred = base_model(X_test)
        base_mse = ((base_pred - Y_test) ** 2).mean(dim=0)
        rmse_table(f"baseline ({args.baseline_tag}) test RMSE", base_mse, base_names)

    # Calibration: does inter-member spread (std across the M members, per
    # sample per target) correlate with actual |error| of the ensemble mean?
    # Bucket test samples into spread quintiles per target and report mean
    # |error| in each bucket -- a useful signal should be monotonic.
    spread = preds.std(dim=0)  # (N, T)
    abs_err = (ensemble_pred - Y_test).abs()  # (N, T)
    print("\n  calibration check (mean |error| by inter-member spread quintile, low -> high):")
    T = Y_test.shape[1]
    n = Y_test.shape[0]
    for k, name in enumerate(target_names):
        order = torch.argsort(spread[:, k])
        bucket_means = []
        for q in range(5):
            idx = order[q * n // 5:(q + 1) * n // 5]
            bucket_means.append(abs_err[idx, k].mean().item())
        arrow = "monotonic" if all(bucket_means[i] <= bucket_means[i + 1] + 1e-12 for i in range(4)) else "not monotonic"
        print(f"    {name:55s} " + "  ".join(f"{m:9.4g}" for m in bucket_means) + f"   [{arrow}]")


if __name__ == "__main__":
    main()

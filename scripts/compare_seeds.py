"""One-off: re-evaluate the 3 standard (reg_mlp_big_full_s*) and 3 joint
(vae_joint_full_s*) checkpoints from this session's architecture comparison
against the shared bootstrap_live test split, and print a per-seed +
aggregate (mean +/- std across seeds) per-target RMSE table for each branch.
Not part of the regular pipeline -- just a report generator over checkpoints
that already exist on disk.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from train import DualPathMLP, load_split
from train_vae_joint import JointVAESurrogate, evaluate as joint_evaluate

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
SPLIT = "bootstrap_live"


def eval_standard(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"],
        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"], spatial_latent=ckpt["spatial_latent"],
        head_hidden=ckpt["head_hidden"], priority_weight=ckpt["priority_weight"],
        use_spatial=ckpt["use_spatial"], trunk_arch=ckpt["trunk_arch"], trunk_blocks=ckpt["trunk_blocks"],
        use_symlog_latent=ckpt["use_symlog_latent"], log_target_mask=ckpt["log_target_mask"],
        objective=ckpt["objective"],
    ).to(dev)
    # strict=False: tolerates checkpoints saved before norm_target_mean/std
    # (train.py's --normalize-target-names) existed -- those buffers just
    # keep their identity-transform __init__ defaults, an exact no-op.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    data_dir = OUT_DIR / "splits" / SPLIT
    X_test, Y_test = load_split("test", data_dir)
    X_test, Y_test = X_test.to(dev), Y_test.to(dev)
    loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256, shuffle=False)
    mse_sum, n_batches = torch.zeros(ckpt["n_targets"], device=dev), 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            mse_sum += ((pred - yb) ** 2).mean(dim=0)
            n_batches += 1
    return (mse_sum / n_batches).sqrt().cpu(), ckpt["target_names"]


def eval_joint(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = JointVAESurrogate(
        ckpt["n_targets"], ckpt["vae_latent_dim"], ckpt["vae_hidden"], ckpt["hidden"], ckpt["latent"],
        ckpt["head_hidden"], priority_weight=torch.ones(ckpt["n_targets"]),
        log_target_mask=ckpt["log_target_mask"],
    ).to(dev)
    model.vae.load_state_dict(ckpt["vae_state_dict"])
    model.surrogate.load_state_dict(ckpt["surrogate_state_dict"])

    data_dir = OUT_DIR / "splits" / SPLIT
    X_test, Y_test = load_split("test", data_dir)
    loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256, shuffle=False)
    _, _, test_mse = joint_evaluate(model, loader, dev, ckpt["coeff_mean"].to(dev),
                                     ckpt["coeff_std"].to(dev), ckpt["beta"], ckpt["n_targets"])
    return test_mse.sqrt().cpu(), ckpt["target_names"]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    standard_rmses, target_names = [], None
    for s in (0, 1, 2):
        rmse, target_names = eval_standard(f"reg_mlp_big_full_s{s}", dev)
        standard_rmses.append(rmse)
    standard_stack = torch.stack(standard_rmses)

    joint_rmses = []
    for s in (0, 1, 2):
        rmse, _ = eval_joint(f"vae_joint_full_s{s}", dev)
        joint_rmses.append(rmse)
    joint_stack = torch.stack(joint_rmses)

    print(f"{'target':55s} {'standard (mean+-std)':>26s} {'joint (mean+-std)':>26s} {'delta':>10s}")
    for i, name in enumerate(target_names):
        s_mean, s_std = standard_stack[:, i].mean().item(), standard_stack[:, i].std().item()
        j_mean, j_std = joint_stack[:, i].mean().item(), joint_stack[:, i].std().item()
        delta = (j_mean - s_mean) / s_mean * 100
        print(f"{name:55s} {s_mean:9.5f} +- {s_std:7.5f} {j_mean:9.5f} +- {j_std:7.5f} {delta:+9.1f}%")

    s_overall = standard_stack.mean(dim=1)
    j_overall = joint_stack.mean(dim=1)
    print(f"\n{'mean over targets, per seed':55s}")
    for s in range(3):
        print(f"  seed {s}: standard {s_overall[s].item():.5f}  joint {j_overall[s].item():.5f}")
    print(f"\noverall mean RMSE: standard {s_overall.mean().item():.5f} +- {s_overall.std().item():.5f}  "
          f"joint {j_overall.mean().item():.5f} +- {j_overall.std().item():.5f}")


if __name__ == "__main__":
    main()

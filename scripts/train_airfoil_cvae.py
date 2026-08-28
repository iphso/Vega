"""Conditional VAE for the airfoil domain -- reuses train_cvae.py's CVAE
class unchanged (its "n_nfp" constructor argument is really just "extra
continuous/discrete conditioning dims," confirmed generic enough to reuse
directly rather than needing a rewrite) but with airfoil-appropriate
conditioning: n_field_periods' discrete one-hot has no equivalent here, so
Reynolds number (log-transformed, wide dynamic range) and angle of attack
are concatenated as two continuous conditioning values instead
(n_nfp=2 at construction, despite the name).

Trained on scripts/generate_airfoil_dataset.py's output (--source full
equivalent -- this is a from-scratch generated dataset, not a real one with
a held-out-region generalization question the way VMEC++'s target_cluster
split answers; every row here is already a synthetic bootstrap sample).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_cvae import CVAE

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = ["cd"]  # matches airfoil_oracle.py's own convention


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="airfoil_cvae_s0")
    p.add_argument("--dataset-tag", default="airfoil")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    X = np.load(OUT_DIR / f"{args.dataset_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{args.dataset_tag}_Y.npy")
    target_names = json.loads((OUT_DIR / f"{args.dataset_tag}_target_names.json").read_text())
    n_targets = len(target_names)

    # XFOIL's convergence flag alone doesn't guarantee physical sanity at the
    # extreme tails: a handful of candidates converge with a spuriously
    # near-zero cd (down to ~1e-12, not real 2D-airfoil drag), blowing up
    # l_over_d=cl/cd to absurd values (seen: up to 9.8e10). Confirmed narrow
    # (127/50,011 rows, 0.25%) before filtering rather than assumed --
    # dropped here as a training-time sanity filter; the real fix (a
    # physical-sanity check inside airfoil_oracle.py's worker_fn itself, not
    # just "did XFOIL's conv flag come back true") is still open.
    cd_col, lod_col = target_names.index("cd"), target_names.index("l_over_d")
    sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
    n_dropped = len(Y) - sane.sum()
    if n_dropped:
        print(f"[{args.tag}] dropping {n_dropped}/{len(Y)} rows with non-physical cd/l_over_d")
    X, Y = X[sane], Y[sane]

    coeff_dim = 16  # 8 upper + 8 lower CST weights, the first 16 columns of X
    coeffs_raw = X[:, :coeff_dim]
    reynolds_raw = X[:, coeff_dim]
    alpha_raw = X[:, coeff_dim + 1]

    coeff_mean = coeffs_raw.mean(axis=0)
    coeff_std = coeffs_raw.std(axis=0).clip(min=1e-6)

    log_reynolds = np.log(reynolds_raw)
    reynolds_mean, reynolds_std = log_reynolds.mean(), log_reynolds.std()
    alpha_mean, alpha_std = alpha_raw.mean(), alpha_raw.std()

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    targets_z = (Yt - t_mean) / t_std

    coeffs = torch.tensor((coeffs_raw - coeff_mean) / coeff_std, dtype=torch.float32)
    targets = torch.tensor(targets_z, dtype=torch.float32)
    aux = torch.tensor(np.stack([
        (log_reynolds - reynolds_mean) / reynolds_std,
        (alpha_raw - alpha_mean) / alpha_std,
    ], axis=-1), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs, targets, aux)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = CVAE(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=args.latent_dim, hidden=args.hidden, n_nfp=2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} n_targets={n_targets} "
          f"latent_dim={args.latent_dim} hidden={args.hidden} params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        recon_sum, kl_sum, n_batches = 0.0, 0.0, 0
        for xb, yb, auxb in loader:
            xb, yb, auxb = xb.to(dev), yb.to(dev), auxb.to(dev)
            cond = torch.cat([yb, auxb], dim=-1)
            opt.zero_grad()
            recon, mu, logvar = model(xb, cond)
            recon_loss = ((recon - xb) ** 2).mean(dim=0).sum()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0).sum()
            loss = recon_loss + args.beta * kl
            loss.backward()
            opt.step()
            recon_sum += recon_loss.item()
            kl_sum += kl.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  recon {recon_sum / n_batches:9.5f}  kl {kl_sum / n_batches:9.5f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim, "hidden": args.hidden, "coeff_dim": coeff_dim,
        "coeff_mean": coeff_mean, "coeff_std": coeff_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "reynolds_mean": reynolds_mean, "reynolds_std": reynolds_std,
        "alpha_mean": alpha_mean, "alpha_std": alpha_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

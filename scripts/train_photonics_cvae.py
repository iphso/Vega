"""Conditional VAE for the photonics domain -- reuses train_cvae.py's CVAE
class unchanged, `n_nfp=0`: like TORAX/mug, photonics has no discrete or
continuous AUX conditioning at all (photonics_oracle.py's own docstring:
"no aux -- no discrete/continuous conditioning variable in this
parameterization"). `cond` is therefore just the (z-scored) target vector.

No LOG_TARGET_NAMES, unlike VMEC/airfoil/TORAX: all 5 targets (up/down/
transmitted/reflected_efficiency, energy_closure) are physically bounded to
roughly [0, 1.2] (real measured range, EXPERIMENT_LOG), nothing spanning
orders of magnitude the way Q_fusion or nbar do -- no wide-dynamic-range
target needs the log treatment here. No sanity filter either:
gym_schema.py's photonics_domain() has both validity_fn and sanity_filter
set to None (no non-convergence mode, no known-bad numerical tail found
yet), matching the real 100% hit rate the actual dataset run confirmed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_cvae import CVAE

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="photonics_cvae_s0")
    p.add_argument("--dataset-tag", default="photonics")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    X = np.load(OUT_DIR / f"{args.dataset_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{args.dataset_tag}_Y.npy")
    target_names = json.loads((OUT_DIR / f"{args.dataset_tag}_target_names.json").read_text())
    n_targets = len(target_names)

    coeff_dim = X.shape[1]  # 4 (PARAM_DIM) -- no aux columns for this domain
    coeffs_raw = X.astype(np.float64)
    coeff_mean = coeffs_raw.mean(axis=0)
    coeff_std = coeffs_raw.std(axis=0).clip(min=1e-6)

    Yt = Y.astype(np.float64)
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    targets_z = (Yt - t_mean) / t_std

    coeffs = torch.tensor((coeffs_raw - coeff_mean) / coeff_std, dtype=torch.float32)
    targets = torch.tensor(targets_z, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = CVAE(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=args.latent_dim, hidden=args.hidden, n_nfp=0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} n_targets={n_targets} "
          f"latent_dim={args.latent_dim} hidden={args.hidden} params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        recon_sum, kl_sum, n_batches = 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            cond = yb
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
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

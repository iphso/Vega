"""Conditional diffusion model for the photonics domain -- reuses
train_diffusion.py's DiffusionDenoiser/make_schedule/q_sample unchanged,
n_nfp=0 (no aux -- see train_photonics_cvae.py's docstring). Data loading
and normalization copied verbatim from train_photonics_cvae.py so all
three photonics checkpoints train on identically-processed data. Same
T=200 linear-schedule DDPM as train_diffusion.py -- a capability check
(does the architecture generalize to a 6th domain), not a per-domain
quality sweep. No categorical-column auxiliary loss needed here (unlike
train_mug_diffusion.py) -- all 4 photonics design params are continuous.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_diffusion import DiffusionDenoiser, make_schedule, q_sample

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--time-embed-dim", type=int, default=32)
    p.add_argument("--T", type=int, default=200, help="diffusion timesteps")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="photonics_diffusion_s0")
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

    coeff_dim = X.shape[1]
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

    schedule = make_schedule(args.T, device=dev)
    model = DiffusionDenoiser(coeff_dim=coeff_dim, n_targets=n_targets, hidden=args.hidden,
                               time_embed_dim=args.time_embed_dim, n_nfp=0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} n_targets={n_targets} "
          f"T={args.T} hidden={args.hidden} params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            cond = yb
            t = torch.randint(0, args.T, (xb.shape[0],), device=dev)
            x_t, noise = q_sample(xb, t, schedule)
            opt.zero_grad()
            eps_pred = model(x_t, t, cond)
            loss = ((eps_pred - noise) ** 2).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  loss {loss_sum / n_batches:9.5f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": args.hidden, "time_embed_dim": args.time_embed_dim, "T": args.T,
        "coeff_dim": coeff_dim,
        "coeff_mean": coeff_mean, "coeff_std": coeff_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

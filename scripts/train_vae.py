"""Conditional VAE over the 90 free boundary Fourier coefficients, conditioned
on n_field_periods -- trained on the full real dataset (all splits combined;
this model's job is to know the real data distribution as well as possible,
not to generalize to held-out rows the way the surrogate is evaluated).

Purpose: a decoder that reliably produces plausible-looking Fourier
coefficients from a latent sample, for the data-augmentation generation loop
(scripts/generate_and_validate.py) -- VMEC++ is the actual certainty check
(a candidate is only "physically plausible" once it's confirmed to converge),
but sampling from noise directly would have a very low hit rate against that
check, given how structured/correlated real boundaries are (see EXPERIMENT_LOG
-- e.g. 9 of 90 coefficients are exactly zero by symmetry, others tightly
correlated). The VAE's job is purely to raise that hit rate by learning the
real manifold's shape first.

Coefficients are standardized (per-coefficient mean/std from metadata.json)
before training for stable reconstruction loss balance across differently-
scaled coefficients -- unlike the surrogate itself, this is a generative
model over the coefficients, not a regressor of physical targets, so the
project's "don't normalize" reasoning for the surrogate doesn't transfer here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

NFP_VALUES = [1, 2, 3, 4, 5]  # observed range, see metadata.json


def nfp_one_hot(nfp_col):
    idx = torch.tensor([NFP_VALUES.index(int(v)) for v in nfp_col.tolist()], device=nfp_col.device)
    return torch.nn.functional.one_hot(idx, num_classes=len(NFP_VALUES)).float()


class VAE(nn.Module):
    def __init__(self, coeff_dim=90, latent_dim=32, hidden=256, n_nfp=len(NFP_VALUES)):
        super().__init__()
        self.latent_dim = latent_dim
        cond_dim = n_nfp
        self.encoder = nn.Sequential(
            nn.Linear(coeff_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, coeff_dim),
        )

    def encode(self, x, cond):
        h = self.encoder(torch.cat([x, cond], dim=-1))
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z, cond):
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, cond)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = ((recon - x) ** 2).mean(dim=0).sum()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0).sum()
    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.01, help="KL weight")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="vae_coeffs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--extra-x", action="append", default=[],
                    help="additional X.npy path(s) (same 92-col schema as output/X.npy) to concatenate "
                         "with the real dataset before training -- e.g. accepted synthetic designs from "
                         "scripts/bootstrap_loop.py. Standardization stats (coeff_mean/std) still come "
                         "from the real dataset's metadata.json only, so they stay anchored across "
                         "generations rather than drifting with whatever synthetic data gets added.")
    p.add_argument("--warm-start", default=None, help="checkpoint tag to initialize weights from, instead of random init")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    X = np.load(OUT_DIR / "X.npy")
    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    feature_names = json.loads((OUT_DIR / "feature_names.json").read_text())
    coeff_mean = np.array([meta["feature_stats"][n]["mean"] for n in feature_names[:90]])
    coeff_std = np.array([meta["feature_stats"][n]["std"] for n in feature_names[:90]]).clip(min=1e-6)

    n_real = len(X)
    for path in args.extra_x:
        X = np.concatenate([X, np.load(path)])
    if args.extra_x:
        print(f"[{args.tag}] {n_real:,} real rows + {len(X) - n_real:,} extra rows from {len(args.extra_x)} "
              f"file(s) = {len(X):,} total")

    coeffs = torch.tensor((X[:, :90] - coeff_mean) / coeff_std, dtype=torch.float32)
    nfp = torch.tensor(X[:, 90], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs, nfp)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = VAE(coeff_dim=90, latent_dim=args.latent_dim, hidden=args.hidden).to(dev)
    if args.warm_start:
        warm_ckpt = torch.load(CKPT_DIR / f"{args.warm_start}.pt", map_location=dev)
        model.load_state_dict(warm_ckpt["model_state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] latent_dim={args.latent_dim} hidden={args.hidden} beta={args.beta} "
          f"params={n_params:,} rows={len(dataset):,}" + (f" warm_start={args.warm_start}" if args.warm_start else ""))

    for epoch in range(1, args.epochs + 1):
        model.train()
        recon_sum, kl_sum, n_batches = 0.0, 0.0, 0
        for xb, nfp_b in loader:
            xb, nfp_b = xb.to(dev), nfp_b.to(dev)
            cond = nfp_one_hot(nfp_b)
            opt.zero_grad()
            recon, mu, logvar = model(xb, cond)
            loss, recon_loss, kl = vae_loss(recon, xb, mu, logvar, args.beta)
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
        "latent_dim": args.latent_dim,
        "hidden": args.hidden,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "nfp_values": NFP_VALUES,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

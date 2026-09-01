"""Conditional WGAN-GP for the bracket domain -- reuses train_gan.py's
Generator/Critic/gradient_penalty UNCHANGED, same convention already
established by train_mug_gan.py for a second (also-geometric) domain.
Condition is a single scalar: the (log, z-scored) compliance the shape
achieves -- so sampling z with a chosen "cond" value is how you DIRECT
generation ("give me a shape around this compliance"), the generative half
of the bracket domain's A/B/C/D bar (oracle=warp.fem forward solve,
dataset=bracket_generate_dataset_warp.py, scoring=train_bracket_scoring.py,
generative=this file + bracket_direct_geometry.py's latent steering).

One GAN PER FORCE LAYOUT (same reasoning as train_bracket_scoring.py's
docstring -- different layouts have different cell counts, no shared input
dimensionality to condition one cross-layout model on yet).

Same WGAN-GP hyperparameters as train_gan.py/train_mug_gan.py (n_critic=5,
lr=1e-4, betas=(0.5, 0.9)) -- not tuned domain-specifically, matching the
project's own stated discipline of not giving any one generative baseline
a tuning campaign the others didn't get.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from train_gan import Generator, Critic, gradient_penalty

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", required=True, choices=["v1_symmetric", "v2_asymmetric", "candidate_12"])
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    tag = f"bracket_gan_{args.layout}_s{args.seed}"

    data = np.load(OUT_DIR / f"bracket_dataset_{args.layout}.npz")
    rho = data["rho"].astype(np.float32)              # (n_rows, n_cells) -- already in [rho_min, 1]
    compliance_log = np.log(data["compliance"].astype(np.float32))

    n_cells = rho.shape[1]
    rho_mean, rho_std = rho.mean(axis=0), rho.std(axis=0).clip(min=1e-6)
    rho_z = (rho - rho_mean) / rho_std
    c_mean, c_std = compliance_log.mean(), compliance_log.std().clip(min=1e-6)
    cond_z = (compliance_log - c_mean) / c_std

    coeffs = torch.tensor(rho_z, dtype=torch.float32)
    cond = torch.tensor(cond_z, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(coeffs, cond)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    # n_targets=1 (the single compliance conditioning scalar), n_nfp=0 (no
    # extra one-hot conditioning block -- there's nothing analogous to
    # stellarators' discrete nfp count in this domain).
    gen = Generator(coeff_dim=n_cells, n_targets=1, latent_dim=args.latent_dim, hidden=args.hidden, n_nfp=0).to(dev)
    critic = Critic(coeff_dim=n_cells, n_targets=1, hidden=args.hidden, n_nfp=0).to(dev)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    n_params = sum(p_.numel() for p_ in gen.parameters())
    print(f"[{tag}] rows={len(dataset):,} n_cells={n_cells} latent_dim={args.latent_dim} "
          f"hidden={args.hidden} n_critic={args.n_critic} generator_params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        gen.train(); critic.train()
        d_loss_sum, g_loss_sum, n_batches = 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            for _ in range(args.n_critic):
                z = torch.randn(xb.shape[0], args.latent_dim, device=dev)
                fake = gen(z, yb).detach()
                critic_opt.zero_grad()
                d_loss = critic(fake, yb).mean() - critic(xb, yb).mean() \
                    + args.lambda_gp * gradient_penalty(critic, xb, fake, yb, dev)
                d_loss.backward()
                critic_opt.step()

            z = torch.randn(xb.shape[0], args.latent_dim, device=dev)
            fake = gen(z, yb)
            gen_opt.zero_grad()
            g_loss = -critic(fake, yb).mean()
            g_loss.backward()
            gen_opt.step()

            d_loss_sum += d_loss.item(); g_loss_sum += g_loss.item(); n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  critic_loss {d_loss_sum / n_batches:9.4f}  gen_loss {g_loss_sum / n_batches:9.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "generator_state_dict": gen.state_dict(),
        "latent_dim": args.latent_dim, "hidden": args.hidden, "n_cells": n_cells,
        "rho_mean": rho_mean, "rho_std": rho_std,
        "compliance_log_mean": c_mean, "compliance_log_std": c_std,
        "layout": args.layout, "seed": args.seed,
    }, CKPT_DIR / f"{tag}.pt")
    print(f"saved {CKPT_DIR / f'{tag}.pt'}")


if __name__ == "__main__":
    main()

"""Conditional WGAN-GP for the TORAX domain -- reuses train_gan.py's
Generator/Critic/gradient_penalty unchanged, n_nfp=0 (no aux -- see
train_torax_cvae.py's docstring). Data loading and the sanity filter are
copied verbatim from train_torax_cvae.py/train_torax_diffusion.py so all
three TORAX checkpoints train on identically-processed data. Same WGAN-GP
hyperparameters as train_gan.py (n_critic=5, lr=1e-4, betas=(0.5, 0.9)) --
not tuned domain-specifically.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_gan import Generator, Critic, gradient_penalty
from train_torax_cvae import sanity_mask

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = ["Q_fusion", "tau_E"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="torax_gan_s0")
    p.add_argument("--dataset-tag", default="torax")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    X = np.load(OUT_DIR / f"{args.dataset_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{args.dataset_tag}_Y.npy")
    target_names = json.loads((OUT_DIR / f"{args.dataset_tag}_target_names.json").read_text())
    n_targets = len(target_names)

    sane = sanity_mask(X, Y, target_names)
    n_dropped = len(Y) - sane.sum()
    if n_dropped:
        print(f"[{args.tag}] dropping {n_dropped}/{len(Y)} rows outside the sanity filter (Q_fusion/T_e/H98 caps)")
    X, Y = X[sane], Y[sane]

    coeff_dim = X.shape[1]
    coeffs_raw = X

    # nbar (~1e19-1e20) squared overflows float32 during std() --
    # confirmed directly: coeff_std came back `inf` for that one column,
    # silently zeroing its z-scored signal for every row (a real, serious
    # bug caught by checking the actual computed stats, not assumed safe
    # just because training ran without an error). Compute in float64.
    coeffs_f64 = coeffs_raw.astype(np.float64)
    coeff_mean = coeffs_f64.mean(axis=0)
    coeff_std = coeffs_f64.std(axis=0).clip(min=1e-6)

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    targets_z = (Yt - t_mean) / t_std

    coeffs = torch.tensor((coeffs_raw - coeff_mean) / coeff_std, dtype=torch.float32)
    targets = torch.tensor(targets_z, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    gen = Generator(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=args.latent_dim, hidden=args.hidden, n_nfp=0).to(dev)
    critic = Critic(coeff_dim=coeff_dim, n_targets=n_targets, hidden=args.hidden, n_nfp=0).to(dev)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} n_targets={n_targets} "
          f"latent_dim={args.latent_dim} hidden={args.hidden} n_critic={args.n_critic} "
          f"generator_params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        gen.train(); critic.train()
        d_loss_sum, g_loss_sum, n_batches = 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            cond = yb

            for _ in range(args.n_critic):
                z = torch.randn(xb.shape[0], args.latent_dim, device=dev)
                fake = gen(z, cond).detach()
                critic_opt.zero_grad()
                d_loss = critic(fake, cond).mean() - critic(xb, cond).mean() \
                    + args.lambda_gp * gradient_penalty(critic, xb, fake, cond, dev)
                d_loss.backward()
                critic_opt.step()

            z = torch.randn(xb.shape[0], args.latent_dim, device=dev)
            fake = gen(z, cond)
            gen_opt.zero_grad()
            g_loss = -critic(fake, cond).mean()
            g_loss.backward()
            gen_opt.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  critic_loss {d_loss_sum / n_batches:9.4f}  gen_loss {g_loss_sum / n_batches:9.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "generator_state_dict": gen.state_dict(),
        "latent_dim": args.latent_dim, "hidden": args.hidden, "coeff_dim": coeff_dim,
        "coeff_mean": coeff_mean, "coeff_std": coeff_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

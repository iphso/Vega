"""Conditional GAN over the 90 free boundary Fourier coefficients -- a third
generative architecture family for the same target-metrics-to-design task
train_cvae.py (VAE) and train_diffusion.py (diffusion) answer. Same
conditioning (11 log+z-scored target metrics + nfp one-hot), same
--source full convention, same coefficient-standardization discipline.
Generator architecture is identical in shape to train_cvae.py's decoder
(same hidden width, same depth, same cond_dim) -- 101,466 params, which
turns out to match that decoder's own param count exactly (211,098 total
CVAE params include its encoder, which only exists for training and is
never used at generation time; 101,466 is what actually runs at sampling
time for both). train_diffusion.py's denoiser (206,810 params) is a
different comparison basis again -- it's the one network used at both
train and inference time, run T times per sample rather than once. Not
claiming perfectly matched capacity across all three, just that none of
them were arbitrarily over/under-sized relative to what the others
actually run at generation time.

WGAN-GP (Gulrajani et al. 2017), not a vanilla BCE-loss GAN: mode collapse
and non-convergence are the standard failure modes of adversarial training,
and this is meant to stand as a fair, working baseline without a tuning
campaign the other two generative baselines didn't get either (per
EXPERIMENT_LOG §18's own admission that train_diffusion.py's schedule
wasn't tuned) -- WGAN-GP's gradient-penalty critic is the standard "give
yourself the best chance of stable training with default-ish hyperparameters"
choice, not a claim that it's the best possible GAN variant for this task.

n_critic critic updates per generator update (5, WGAN-GP's own default),
lr=1e-4 and betas=(0.5, 0.9) for both optimizers (also standard WGAN-GP
defaults, not tuned here) -- deliberately not train_cvae.py/
train_diffusion.py's lr=1e-3/Adam-defaults, which is known to destabilize
adversarial training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from make_splits import LOG_TARGET_NAMES
from train_vae import NFP_VALUES, nfp_one_hot

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


class Generator(nn.Module):
    def __init__(self, coeff_dim=90, n_targets=11, latent_dim=32, hidden=256, n_nfp=len(NFP_VALUES)):
        super().__init__()
        self.latent_dim = latent_dim
        cond_dim = n_targets + n_nfp
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, coeff_dim),
        )

    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=-1))


class Critic(nn.Module):
    def __init__(self, coeff_dim=90, n_targets=11, hidden=256, n_nfp=len(NFP_VALUES)):
        super().__init__()
        cond_dim = n_targets + n_nfp
        self.net = nn.Sequential(
            nn.Linear(coeff_dim + cond_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, cond):
        return self.net(torch.cat([x, cond], dim=-1)).squeeze(-1)


def gradient_penalty(critic, real, fake, cond, device):
    eps = torch.rand(real.shape[0], 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp, cond)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interp, grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def target_cond(Y, target_names, t_mean, t_std):
    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    return (Yt - t_mean) / t_std


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
    p.add_argument("--tag", default="gan_targets")
    p.add_argument("--source", default="full", choices=["split", "full"])
    p.add_argument("--split", default="target_cluster")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    if args.source == "full":
        X, Y = np.load(OUT_DIR / "X.npy"), np.load(OUT_DIR / "Y.npy")
    else:
        train_npz = np.load(OUT_DIR / "splits" / args.split / "train.npz")
        X, Y = train_npz["X"], train_npz["Y"]
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())

    coeff_mean = X[:, :90].mean(axis=0)
    coeff_std = X[:, :90].std(axis=0).clip(min=1e-6)

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    cond_targets = target_cond(Y, target_names, t_mean, t_std)

    coeffs = torch.tensor((X[:, :90] - coeff_mean) / coeff_std, dtype=torch.float32)
    targets = torch.tensor(cond_targets, dtype=torch.float32)
    nfp = torch.tensor(X[:, 90], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs, targets, nfp)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    n_targets = len(target_names)
    gen = Generator(coeff_dim=90, n_targets=n_targets, latent_dim=args.latent_dim, hidden=args.hidden).to(dev)
    critic = Critic(coeff_dim=90, n_targets=n_targets, hidden=args.hidden).to(dev)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    n_params = sum(p.numel() for p in gen.parameters())
    source_desc = "full dataset" if args.source == "full" else f"split={args.split}/train"
    print(f"[{args.tag}] source={source_desc} latent_dim={args.latent_dim} hidden={args.hidden} "
          f"n_critic={args.n_critic} generator_params={n_params:,} rows={len(dataset):,}")

    for epoch in range(1, args.epochs + 1):
        gen.train(); critic.train()
        d_loss_sum, g_loss_sum, n_batches = 0.0, 0.0, 0
        for xb, yb, nfp_b in loader:
            xb, yb, nfp_b = xb.to(dev), yb.to(dev), nfp_b.to(dev)
            cond = torch.cat([yb, nfp_one_hot(nfp_b)], dim=-1)

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
        "latent_dim": args.latent_dim,
        "hidden": args.hidden,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "target_names": target_names,
        "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean,
        "target_std": t_std,
        "nfp_values": NFP_VALUES,
        "source": args.source,
        "split": args.split if args.source == "split" else None,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

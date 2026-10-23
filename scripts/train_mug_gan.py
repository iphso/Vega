"""Conditional WGAN-GP for the mug domain -- reuses train_gan.py's
Generator/Critic/gradient_penalty UNCHANGED (n_nfp=0, no aux -- see
train_mug_cvae.py's docstring). Data loading/normalization copied verbatim
from train_mug_cvae.py/train_mug_diffusion.py so all three mug checkpoints
train on identically-processed data. Same WGAN-GP hyperparameters as
train_gan.py (n_critic=5, lr=1e-4, betas=(0.5, 0.9)) -- not tuned
domain-specifically.

§63: same categorical fix as train_mug_cvae.py (see mug_categorical.py's
module docstring for the concrete broken-output evidence this replaces),
adapted to WGAN-GP specifically. The Generator's raw output for a
categorical block is unconstrained logits, but real data for that block is
a hard one-hot (exact 0/1) -- if the Critic saw those two representations
directly, it could trivially tell real from fake just by noticing "this
block isn't in {0,1}," a shortcut that teaches the Generator nothing about
which material is realistic. Fixed via `mug_categorical.
soften_categorical_torch()`: softmax each categorical block of the
Generator's output (differentiable, so the adversarial loss still
backprops into it) before the Critic ever sees it, leaving BOTH real and
fake in the same probability simplex. The Generator is pushed toward
confident near-one-hot corners by the adversarial loss itself; the actual
hard discrete decision is only made later, at sampling time, by
`from_model_space()`'s argmax (steerability_generic.py / any future
sampling code), never here during training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mug_categorical as mc
from train_gan import Generator, Critic, gradient_penalty

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="mug_gan_s0")
    p.add_argument("--dataset-tag", default="mug")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    X = np.load(OUT_DIR / f"{args.dataset_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{args.dataset_tag}_Y.npy")
    target_names = json.loads((OUT_DIR / f"{args.dataset_tag}_target_names.json").read_text())
    n_targets = len(target_names)

    coeff_dim = mc.MODEL_DIM  # 42, not X.shape[1]=9 -- see train_mug_cvae.py
    coeffs_model = mc.to_model_space(X)
    cont_slice, _cat_slices = mc.model_space_slices()

    cont_mean = coeffs_model[:, cont_slice].mean(axis=0)
    cont_std = coeffs_model[:, cont_slice].std(axis=0).clip(min=1e-6)
    coeffs_z = coeffs_model.copy()
    coeffs_z[:, cont_slice] = (coeffs_model[:, cont_slice] - cont_mean) / cont_std
    # categorical blocks stay exact one-hot -- real data the Critic sees
    # must be a genuine hard corner of the simplex, the thing the
    # Generator's own softmaxed output (below) is being pushed toward.

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    targets_z = (Yt - t_mean) / t_std

    coeffs = torch.tensor(coeffs_z, dtype=torch.float32)
    targets = torch.tensor(targets_z, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    gen = Generator(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=args.latent_dim, hidden=args.hidden, n_nfp=0).to(dev)
    critic = Critic(coeff_dim=coeff_dim, n_targets=n_targets, hidden=args.hidden, n_nfp=0).to(dev)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} (7 continuous + "
          f"{'+'.join(str(n) for _, _, n in mc.CATEGORICAL_SPECS)} one-hot) n_targets={n_targets} "
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
                fake = mc.soften_categorical_torch(gen(z, cond)).detach()
                critic_opt.zero_grad()
                d_loss = critic(fake, cond).mean() - critic(xb, cond).mean() \
                    + args.lambda_gp * gradient_penalty(critic, xb, fake, cond, dev)
                d_loss.backward()
                critic_opt.step()

            z = torch.randn(xb.shape[0], args.latent_dim, device=dev)
            fake = mc.soften_categorical_torch(gen(z, cond))
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
        "coeff_mean": cont_mean, "coeff_std": cont_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

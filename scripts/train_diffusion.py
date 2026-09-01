"""Conditional diffusion model over the 90 free boundary Fourier coefficients
-- a second, architecturally different generative baseline for the same
target-metrics-to-design task train_cvae.py answers with a VAE. Same
conditioning (11 log+z-scored target metrics + nfp one-hot), same
--source full convention (train on the entire real dataset, no held-out
region -- this is a capability check, not a generalization test, per
EXPERIMENT_LOG §17), same coefficient standardization discipline (stats
computed from whatever was actually trained on). Only the generative
mechanism differs: a single decoder forward pass (VAE) vs. an iterative
denoising chain (diffusion) -- the question this baseline answers is whether
the cVAE's modest steerability (§17: ~68-71% correct-direction, clearing its
own chance-level floor by ~13-16 points but far short of retrieval in dense
target-space regions) is a fundamental property of this task/data, or
specifically a VAE-architecture limitation a stronger generative model can
improve on.

Standard DDPM (Ho et al. 2020) training objective (simple noise-prediction
MSE loss, ancestral sampling at generation time), T=200 by default -- enough
steps for stable training without making eval-time generation (this model
gets sampled dozens of times per anchor in eval_cvae_steerability.py, each
requiring T sequential forward passes, batched across the k candidates in
flight) prohibitively slow; not tuned against a diffusion-specific quality
metric, matched instead to what keeps the eval harness's wall-clock budget
comparable to the cVAE's single-pass sampling -- see EXPERIMENT_LOG §24 for
the resulting ~200x inference-compute asymmetry vs. the cVAE/GAN, an
acknowledged, not fixed, gap.

Noise schedule: still linear beta, but re-tuned (beta_end 0.02 -> 0.12), NOT
the original DDPM paper's linear beta(1e-4 -> 0.02) schedule this file used
through EXPERIMENT_LOG §18-23. That schedule was tuned for T=1000; reused
unchanged at this project's T=200 it left alpha_bar at t=T-1 = 0.132 (should
be ~0 for genuinely destroyed signal) -- confirmed directly via
audit_baselines.py (§24), not assumed. A real train/sample mismatch:
training's forward process never actually produced a fully-noised input at
t=T-1, but ddpm_sample's ancestral chain starts generation from
x_T ~ N(0, I) (fully noised) regardless.

A cosine schedule (Nichol & Dhariwal 2021) was tried first and reverted --
worth recording why, not just what. Cosine reaches alpha_bar~0 by
construction for any T without needing to hand-retune a linear schedule's
beta_end, which looked like the more principled fix. But at only T=200
steps, the cosine curve's steep late-stage descent forces individual betas
near t=T up to the schedule's own clip ceiling (0.999) to compensate for
the coarser discretization -- confirmed by direct trace through
ddpm_sample's reverse loop (not assumed): the very first reverse step
(t=T-1, beta=0.999, alpha=0.001) divides eps_pred's contribution by
sqrt(alpha_t)=0.03, a ~30x one-step amplification of any prediction error.
Traced sample stats blew up from std~1 at x_T to std~480 by t=0, entirely
off-manifold, unstandardized param magnitudes in the hundreds against a
real O(0.001-1) range. This is why the very first rerun of the ablation
grid after the "fix" showed near-0% validity -- caught before the full
grid finished, not after. **Fixed with a re-tuned linear schedule instead**:
beta_end=0.12 (up from 0.02) gives alpha_bar at t=T-1 = 3.6e-6 (comfortably
below the ~1e-5 "genuinely destroyed" bar) while keeping max beta at only
0.12 (alpha_t=0.88, sqrt=0.94 -- nowhere near the cosine schedule's
division-by-near-zero danger zone). Verified directly post-fix, both the
schedule's endpoint value and a full traced ancestral sample staying
on-manifold end to end, before retraining either checkpoint on it.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from make_splits import LOG_TARGET_NAMES
from train_vae import NFP_VALUES, nfp_one_hot

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def sinusoidal_embedding(t, dim):
    """Standard transformer-style sinusoidal timestep embedding. t: (B,) long/float tensor of timesteps."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionDenoiser(nn.Module):
    def __init__(self, coeff_dim=90, n_targets=11, hidden=256, time_embed_dim=64, n_nfp=len(NFP_VALUES)):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        cond_dim = n_targets + n_nfp
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(coeff_dim + time_embed_dim + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, coeff_dim),
        )

    def forward(self, x_t, t, cond):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        return self.net(torch.cat([x_t, t_emb, cond], dim=-1))


def make_schedule(T, beta_start=1e-4, beta_end=0.12, device="cpu"):
    """Linear beta schedule, re-tuned from this file's original
    beta_end=0.02 (see module docstring for why, and why a cosine schedule
    was tried first and reverted). beta_end=0.12 gives alpha_bar at t=T-1
    of ~3.6e-6 at this project's T=200 -- comfortably past the ~1e-5
    "genuinely destroyed signal" bar the original beta_end=0.02 missed
    (0.132) -- while keeping the max single-step beta at only 0.12, far
    from the ~1 danger zone that made the cosine alternative blow up
    numerically at generation time (dividing by sqrt(alpha_t) amplifies
    error ~1.07x here vs. ~31x there)."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


def q_sample(x0, t, schedule, noise=None):
    """Forward diffusion: x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = schedule["alpha_bars"][t][:, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise


@torch.no_grad()
def ddpm_sample(model, cond, schedule, coeff_dim=90, device="cpu"):
    """Standard DDPM ancestral sampling, batched across all rows of `cond` at
    once (one denoising chain per row, T sequential model calls total for
    the whole batch -- not T calls per row)."""
    T = schedule["betas"].shape[0]
    n = cond.shape[0]
    x = torch.randn(n, coeff_dim, device=device)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch, cond)
        beta_t, alpha_t, ab_t = schedule["betas"][t], schedule["alphas"][t], schedule["alpha_bars"][t]
        mean = (x - beta_t / (1 - ab_t).sqrt() * eps_pred) / alpha_t.sqrt()
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + beta_t.sqrt() * noise
        else:
            x = mean
    return x


def target_cond(Y, target_names, t_mean, t_std):
    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    return (Yt - t_mean) / t_std


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--T", type=int, default=200, help="diffusion timesteps")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="diffusion_targets")
    p.add_argument("--source", default="full", choices=["split", "full", "augmented"],
                    help="see train_cvae.py's --source for the full rationale; 'full' is this script's "
                         "only tested path so far, matching its capability-check (not generalization) use. "
                         "'augmented': real + oracle-validated bootstrap-generated rows, see --aug-tag.")
    p.add_argument("--split", default="target_cluster")
    p.add_argument("--aug-tag", default=None,
                    help="--source augmented only: loads X_aug_<tag>.npy/Y_aug_<tag>.npy, built by "
                         "merge_bootstrap_pools.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    if args.source == "full":
        X, Y = np.load(OUT_DIR / "X.npy"), np.load(OUT_DIR / "Y.npy")
    elif args.source == "augmented":
        assert args.aug_tag, "--source augmented requires --aug-tag"
        X, Y = np.load(OUT_DIR / f"X_aug_{args.aug_tag}.npy"), np.load(OUT_DIR / f"Y_aug_{args.aug_tag}.npy")
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    schedule = make_schedule(args.T, device=dev)
    model = DiffusionDenoiser(coeff_dim=90, n_targets=len(target_names), hidden=args.hidden,
                               time_embed_dim=args.time_embed_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    source_desc = "full dataset" if args.source == "full" else f"split={args.split}/train"
    print(f"[{args.tag}] source={source_desc} T={args.T} hidden={args.hidden} "
          f"params={n_params:,} rows={len(dataset):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, yb, nfp_b in loader:
            xb, yb, nfp_b = xb.to(dev), yb.to(dev), nfp_b.to(dev)
            cond = torch.cat([yb, nfp_one_hot(nfp_b)], dim=-1)
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
        "hidden": args.hidden,
        "time_embed_dim": args.time_embed_dim,
        "T": args.T,
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

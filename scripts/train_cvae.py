"""Conditional VAE over the 90 free boundary Fourier coefficients, conditioned
on the 11 target metrics (+ n_field_periods) -- the inverse of train_vae.py's
job. Where train_vae.py answers "what does a plausible boundary look like,"
this answers "what does a plausible boundary look like *for these target
values*" -- the deterministic-regression baseline's generative counterpart:
scripts/screen_reference_baselines.py (X -> Y) predicts target metrics from a
design; this predicts (a distribution over) designs from target metrics.

Two training sources (--source):

  split (default) -- output/splits/<--split>/train.npz only (see
    make_splits.py's `target-cluster` mode). Use this when the eval needs a
    held-out-region generalization claim, e.g. eval_cvae_generation.py's
    "can it hit target regions it never saw a neighbor of" test -- training
    on val/test target-clusters would invalidate that before it starts.

  full -- the entire real dataset (output/X.npy, Y.npy), train_vae.py's own
    convention of training on everything. Use this for a pure capability
    check that isn't measuring generalization to an unseen region at all
    (e.g. eval_cvae_steerability.py: can the model hit an arbitrary
    requested target combination and does nudging one target move the
    generated design's measured metric the requested way -- neither
    question needs held-out data, and more training data only helps).

Coefficient and target normalization stats are always computed from
whichever set was actually trained on (not metadata.json's full-dataset
stats, even in --source full), same leakage discipline as §7's
--normalize-targets/--normalize-inputs flags in train.py.

The 4 wide-dynamic-range targets in LOG_TARGET_NAMES (see make_splits.py) are
log-transformed before z-scoring, both for the clustering these splits were
built from and for the conditioning vector here -- otherwise a handful of
qi/max_elongation outliers would dominate the conditioning signal the same
way they'd have dominated the clustering distance.
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


class CVAE(nn.Module):
    def __init__(self, coeff_dim=90, n_targets=11, latent_dim=32, hidden=256, n_nfp=len(NFP_VALUES)):
        super().__init__()
        self.latent_dim = latent_dim
        cond_dim = n_targets + n_nfp
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


def target_cond(Y, target_names, t_mean, t_std):
    """Log-transforms LOG_TARGET_NAMES columns then z-scores every column,
    using stats already computed on the train split (t_mean/t_std)."""
    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    return (Yt - t_mean) / t_std


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.01, help="KL weight")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="cvae_coeffs")
    p.add_argument("--source", default="split", choices=["split", "full"],
                    help="'split': output/splits/<--split>/train.npz only (for a held-out-region "
                         "generalization test, e.g. --split target_cluster). 'full': the entire real "
                         "dataset (output/X.npy, Y.npy), train_vae.py's convention -- use this when the "
                         "eval doesn't need a held-out target region (e.g. eval_cvae_steerability.py's "
                         "capability check, as opposed to eval_cvae_generation.py's generalization check).")
    p.add_argument("--split", default="target_cluster", help="--source split only: output/splits/<split>/train.npz to train on")
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = CVAE(coeff_dim=90, n_targets=len(target_names), latent_dim=args.latent_dim, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    source_desc = "full dataset" if args.source == "full" else f"split={args.split}/train"
    print(f"[{args.tag}] source={source_desc} latent_dim={args.latent_dim} hidden={args.hidden} "
          f"beta={args.beta} params={n_params:,} rows={len(dataset):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        recon_sum, kl_sum, n_batches = 0.0, 0.0, 0
        for xb, yb, nfp_b in loader:
            xb, yb, nfp_b = xb.to(dev), yb.to(dev), nfp_b.to(dev)
            cond = torch.cat([yb, nfp_one_hot(nfp_b)], dim=-1)
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

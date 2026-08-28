"""Conditional diffusion model for the airfoil domain -- reuses
train_diffusion.py's DiffusionDenoiser/make_schedule/ddpm_sample unchanged
(same generic "n_nfp is just extra conditioning dims" argument that let
train_airfoil_cvae.py reuse train_cvae.py's CVAE directly), with the same
airfoil-appropriate conditioning as train_airfoil_cvae.py: log-Reynolds and
angle of attack as two continuous aux values (n_nfp=2) instead of VMEC's
discrete nfp one-hot. Data loading, physical-sanity filtering, and
normalization discipline are copied verbatim from train_airfoil_cvae.py so
the two checkpoints are trained on identically-processed data and directly
comparable in eval_airfoil_steerability.py.

Same T=200 linear-schedule DDPM as train_diffusion.py, not tuned
domain-specifically -- this is a capability check (does the architecture
generalize to a second domain at all), not a per-domain quality sweep.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_diffusion import DiffusionDenoiser, make_schedule, q_sample

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = ["cd"]  # matches airfoil_oracle.py's own convention


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--T", type=int, default=200, help="diffusion timesteps")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="airfoil_diffusion_s0")
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

    # Same physical-sanity filter as train_airfoil_cvae.py -- see that
    # file's comment for the 127/50,011 near-zero-cd finding this guards
    # against. Kept verbatim rather than factored out so each training
    # script stays independently readable; not worth a shared-util module
    # for a two-line filter used in three places.
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

    schedule = make_schedule(args.T, device=dev)
    model = DiffusionDenoiser(coeff_dim=coeff_dim, n_targets=n_targets, hidden=args.hidden,
                               time_embed_dim=args.time_embed_dim, n_nfp=2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} n_targets={n_targets} "
          f"T={args.T} hidden={args.hidden} params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, yb, auxb in loader:
            xb, yb, auxb = xb.to(dev), yb.to(dev), auxb.to(dev)
            cond = torch.cat([yb, auxb], dim=-1)
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
        "reynolds_mean": reynolds_mean, "reynolds_std": reynolds_std,
        "alpha_mean": alpha_mean, "alpha_std": alpha_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

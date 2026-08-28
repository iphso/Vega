"""Conditional VAE for the TORAX domain -- reuses train_cvae.py's CVAE class
unchanged (same "n_nfp is really just extra conditioning dims" argument
that already let train_airfoil_cvae.py reuse it directly), but with
`n_nfp=0`: TORAX has NO discrete or continuous conditioning variable at all
(unlike VMEC's n_field_periods one-hot or airfoil's Reynolds/alpha) --
`torax_oracle.py`'s own docstring is explicit that "everything that varies
is already in params," confirmed again here via gym_schema.py's `torax`
Conditioning entry (extra_dim=0). `cond` is therefore just the (z-scored)
target vector, nothing concatenated onto it.

Params are z-scored in LINEAR space, same as every other domain (CST
coefficients for airfoil, Fourier coefficients for VMEC) -- a real, flagged
simplification: TORAX's 10 params are strictly-positive quantities
spanning ~8 orders of magnitude and were themselves SAMPLED in log-space
(generate_torax_dataset.py's `seed * exp(N(0, noise_std))`), so a
log-space z-scoring of the params (mirroring the target side's own
LOG_TARGET_NAMES treatment) would likely fit their actual shape better.
Not done here to avoid extending steerability_generic.py's shared,
domain-agnostic `build_candidates` (which un-normalizes via a single
linear `params * coeff_std + coeff_mean`, no domain-specific inverse-log
step) -- kept as linear z-scoring so the existing generic eval
infrastructure needs zero changes, flagged as a real future improvement,
not an oversight.

The sanity filter matches gym_schema.torax_spec()'s own (EXPERIMENT_LOG
§35/§37): Q_fusion<=300, T_e_volume_avg<=200, H98<=20 -- practical caps on
a smooth, gapless power-law tail (no natural "this is definitely broken"
split the way airfoil's cd<1e-6 had), dropping ~1.2% of rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_cvae import CVAE

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = ["Q_fusion", "tau_E"]  # matches torax_oracle.py's own convention


PARAM_NAMES = ["Ip", "nbar", "chi_i", "chi_e", "P_total", "I_generic", "R_major", "a_minor", "B_0", "elongation_LCFS"]
# Practical caps on the INPUT side, discovered while training this pass --
# a real gap not caught by the target-side filter alone (nor gym_schema's
# own sanity_filter, which is Y-only per its established contract, so this
# lives here rather than there): X.npy contains rows with R_major up to
# 134m, elongation_LCFS up to 27, B_0 up to 82T -- structurally-valid
# ToraxConfig overrides that still ran to a finite SimError.NO_ERROR result
# (confirmed by direct inspection, not assumed), but nowhere near any real
# tokamak concept (compare EXPERIMENT_LOG §36's real references: ITER/SPARC/
# JET all sit under R_major=6.2m, B_0=12.2T, elongation~1.7). Generous
# (~3-4x the largest real reference device on each axis), not derived from
# a measured physical limit -- same "practical cap, not a confirmed
# threshold" caveat as the Q_fusion/T_e/H98 filter.
_PARAM_CAPS = {"Ip": 6e7, "nbar": 5e20, "chi_i": 50, "chi_e": 50, "P_total": 5e8,
                "I_generic": 3e7, "R_major": 20, "a_minor": 8, "B_0": 40, "elongation_LCFS": 4}


def sanity_mask(X, Y, target_names):
    qi, tei, hi = (target_names.index(n) for n in ("Q_fusion", "T_e_volume_avg", "H98"))
    target_ok = (Y[:, qi] <= 300) & (Y[:, tei] <= 200) & (Y[:, hi] <= 20)
    param_ok = np.ones(len(X), dtype=bool)
    for i, name in enumerate(PARAM_NAMES):
        param_ok &= X[:, i] <= _PARAM_CAPS[name]
    return target_ok & param_ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="torax_cvae_s0")
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

    coeff_dim = X.shape[1]  # 10 (PARAM_DIM) -- no aux columns for this domain, unlike VMEC/airfoil
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

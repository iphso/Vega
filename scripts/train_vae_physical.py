"""Experiment (B) from the 2026-09-15 literature-review conversation, stage 1:
a VAE over the 90 free boundary Fourier coefficients, trained with an added
*physical-space* (R/Z collocation-point) reconstruction loss on top of the
usual per-coefficient MSE -- following arXiv:2608.16938's finding that this
is what makes a compressed latent space geometry-aware rather than just
statistics-aware, and (per that paper) is a big part of why their downstream
generative sampling actually worked where raw-coefficient conditioning
didn't. See EXPERIMENT_LOG for the full context: our own nae_generative_wall_
probe.py got 0/480 qi-feasible candidates from cVAE/diffusion/GAN models that
all operated directly on the raw standardized 90-dim coefficient vector.

Physical loss construction: R(theta,phi) and Z(theta,phi) are LINEAR in the
stellarator-symmetric Fourier coefficients for any fixed (theta,phi)
collocation grid --
    R(theta,phi) = sum_{m=0..4, n_idx=0..8} r_cos[m,n_idx] * cos(m*theta - (n_idx-4)*nfp*phi)
    Z(theta,phi) = sum_{m=0..4, n_idx=0..8} z_sin[m,n_idx] * sin(m*theta - (n_idx-4)*nfp*phi)
(matches nae_pilot.py's confirmed r_cos[5,9]/z_sin[5,9] layout, m outer/n
inner, n=n_idx-4). Since nfp is an integer in {1..5} for every row in this
project, the (theta,phi)->basis-vector map is exactly one of 5 fixed
matrices -- precomputed once, then applied per-batch by nfp value. This is
CHEAP (a fixed linear layer, no autodiff through actual surface geometry
needed) and exact, not an approximation of the physical-space loss idea.

Coefficients are standardized as in train_vae.py; physical loss is computed
by un-standardizing model output/target back to physical coefficients first
(the basis is defined on physical, not standardized, coefficients).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_vae import VAE, NFP_VALUES, nfp_one_hot  # noqa: E402

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

N_THETA, N_PHI = 8, 8  # collocation grid resolution per surface


def build_basis(nfp_values, n_theta=N_THETA, n_phi=N_PHI):
    """Returns dict nfp -> (basis_r [n_points,45], basis_z [n_points,45]),
    both float32 torch tensors. n_points = n_theta*n_phi."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    TH, PH = TH.flatten(), PH.flatten()  # (n_points,)
    ms = np.arange(5)
    n_idxs = np.arange(9)
    ns = n_idxs - 4
    basis = {}
    for nfp in nfp_values:
        basis_r = np.zeros((len(TH), 5, 9))
        basis_z = np.zeros((len(TH), 5, 9))
        for mi, m in enumerate(ms):
            for ni, n in enumerate(ns):
                angle = m * TH - n * nfp * PH
                basis_r[:, mi, ni] = np.cos(angle)
                basis_z[:, mi, ni] = np.sin(angle)
        basis[nfp] = (
            torch.tensor(basis_r.reshape(len(TH), 45), dtype=torch.float32),
            torch.tensor(basis_z.reshape(len(TH), 45), dtype=torch.float32),
        )
    return basis


def physical_recon_loss(recon_phys, target_phys, nfp_col, basis, dev):
    """recon_phys/target_phys: (B,90) unstandardized coeffs, split as
    r_cos[:45]/z_sin[45:]. nfp_col: (B,) int nfp values. Returns scalar MSE
    over R,Z collocation points, batched by grouping rows with equal nfp
    (basis differs per nfp, but there are only 5 possible values)."""
    loss_sum = 0.0
    n_total = 0
    for nfp in NFP_VALUES:
        mask = nfp_col == nfp
        if not mask.any():
            continue
        basis_r, basis_z = basis[nfp]
        basis_r, basis_z = basis_r.to(dev), basis_z.to(dev)
        r_recon, z_recon = recon_phys[mask, :45], recon_phys[mask, 45:]
        r_targ, z_targ = target_phys[mask, :45], target_phys[mask, 45:]
        R_recon = r_recon @ basis_r.T  # (n_mask, n_points)
        Z_recon = z_recon @ basis_z.T
        R_targ = r_targ @ basis_r.T
        Z_targ = z_targ @ basis_z.T
        loss_sum = loss_sum + ((R_recon - R_targ) ** 2).sum() + ((Z_recon - Z_targ) ** 2).sum()
        n_total += mask.sum().item()
    return loss_sum / max(n_total, 1) / (2 * N_THETA * N_PHI)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.01, help="KL weight")
    p.add_argument("--phys-weight", type=float, default=1.0, help="physical-space recon loss weight")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="vae_physical_v1")
    p.add_argument("--source-split", default="all_grounded_walk_v1",
                    help="output/splits/<name>/train.npz -- use the richest merged pool available")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    train_npz = np.load(OUT_DIR / "splits" / args.source_split / "train.npz")
    X = train_npz["X"]

    coeff_mean = X[:, :90].mean(axis=0)
    coeff_std = X[:, :90].std(axis=0).clip(min=1e-6)

    coeffs_std = torch.tensor((X[:, :90] - coeff_mean) / coeff_std, dtype=torch.float32)
    coeffs_phys = torch.tensor(X[:, :90], dtype=torch.float32)
    nfp = torch.tensor(X[:, 90], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(coeffs_std, coeffs_phys, nfp)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    basis = build_basis(NFP_VALUES)
    coeff_std_t = torch.tensor(coeff_std, dtype=torch.float32, device=dev)
    coeff_mean_t = torch.tensor(coeff_mean, dtype=torch.float32, device=dev)

    model = VAE(coeff_dim=90, latent_dim=args.latent_dim, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] latent_dim={args.latent_dim} hidden={args.hidden} beta={args.beta} "
          f"phys_weight={args.phys_weight} params={n_params:,} rows={len(dataset):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        recon_sum, kl_sum, phys_sum, n_batches = 0.0, 0.0, 0.0, 0
        for xb_std, xb_phys, nfp_b in loader:
            xb_std, xb_phys, nfp_b = xb_std.to(dev), xb_phys.to(dev), nfp_b.to(dev)
            nfp_int = nfp_b.round().long()
            cond = nfp_one_hot(nfp_b)
            opt.zero_grad()
            mu, logvar = model.encode(xb_std, cond)
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
            recon_std = model.decode(z, cond)

            recon_loss = ((recon_std - xb_std) ** 2).mean(dim=0).sum()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0).sum()

            recon_phys = recon_std * coeff_std_t + coeff_mean_t
            phys_loss = physical_recon_loss(recon_phys, xb_phys, nfp_int, basis, dev)

            loss = recon_loss + args.beta * kl + args.phys_weight * phys_loss
            loss.backward()
            opt.step()
            recon_sum += recon_loss.item()
            kl_sum += kl.item()
            phys_sum += phys_loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  recon {recon_sum / n_batches:9.5f}  "
                  f"kl {kl_sum / n_batches:9.5f}  phys {phys_sum / n_batches:9.5f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "hidden": args.hidden,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "nfp_values": NFP_VALUES,
        "phys_weight": args.phys_weight,
        "source_split": args.source_split,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

"""Experiment (B) stage 2: a conditional DDPM operating in the physical-loss
VAE's latent space (train_vae_physical.py) instead of raw 90-dim coefficient
space -- architecturally identical to train_diffusion.py's DiffusionDenoiser,
just with coeff_dim=latent_dim, and the "clean data" x0 is the frozen VAE
encoder's posterior mean (mu) rather than the raw standardized coefficients.

Also implements the fine-tuning half of arXiv:2608.16938's recipe: --finetune
-from loads a pretrained checkpoint (trained on the full/general pool) and
continues training on a small near-wall "HL set" (output/hl_nearwall_*.npy,
the 500 lowest-worst-constraint-violation real rows pooled across this
project's 5 active P3 search lanes) mixed with --replay-frac of the original
general pool, to avoid catastrophic forgetting of basic validity while
steering the model's mass toward the wall.

Sampling produces latents via ddpm_sample, decodes through the frozen VAE
decoder to get standardized coefficients, unstandardizes, and hands off to
the same oracle.params_to_worker_args / real-VMEC validation path
nae_generative_wall_probe.py used -- see that script for the harness
plumbing this one is meant to plug into (sample_fn interface).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_splits import LOG_TARGET_NAMES  # noqa: E402
from train_vae import VAE, NFP_VALUES, nfp_one_hot  # noqa: E402
from train_diffusion import (  # noqa: E402
    DiffusionDenoiser, make_schedule, q_sample, ddpm_sample, target_cond,
)

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def load_vae(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = VAE(coeff_dim=90, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def encode_dataset(vae, X, dev, batch=4096):
    """Returns mu latents (N, latent_dim) for the full dataset, using the
    frozen VAE's posterior mean (not a sample) -- standard practice for
    building a latent-diffusion training set."""
    coeff_mean = torch.tensor(vae_ckpt_stat["coeff_mean"], dtype=torch.float32, device=dev)
    coeff_std = torch.tensor(vae_ckpt_stat["coeff_std"], dtype=torch.float32, device=dev)
    mus = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch, :90], dtype=torch.float32, device=dev)
        xb_std = (xb - coeff_mean) / coeff_std
        nfp_b = torch.tensor(X[i:i + batch, 90], dtype=torch.float32, device=dev)
        cond = nfp_one_hot(nfp_b)
        mu, _ = vae.encode(xb_std, cond)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus)


vae_ckpt_stat = None  # set in main, used by encode_dataset


def load_xy(source, aug_tag=None, split="target_cluster"):
    if source == "full":
        return np.load(OUT_DIR / "X.npy"), np.load(OUT_DIR / "Y.npy")
    if source == "augmented":
        return np.load(OUT_DIR / f"X_aug_{aug_tag}.npy"), np.load(OUT_DIR / f"Y_aug_{aug_tag}.npy")
    if source == "hl_nearwall":
        return np.load(OUT_DIR / "hl_nearwall_X.npy"), np.load(OUT_DIR / "hl_nearwall_Y.npy")
    train_npz = np.load(OUT_DIR / "splits" / source / "train.npz")
    return train_npz["X"], train_npz["Y"]


def main():
    global vae_ckpt_stat
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae-tag", default="vae_physical_v1")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="latent_diffusion_v1")
    p.add_argument("--source", default="all_grounded_walk_v1",
                    help="'full', 'augmented', 'hl_nearwall', or an output/splits/<name> dir")
    p.add_argument("--aug-tag", default=None)
    p.add_argument("--finetune-from", default=None, help="pretrained latent-diffusion tag to continue from")
    p.add_argument("--replay-source", default="all_grounded_walk_v1",
                    help="source of replay rows when fine-tuning (mixed with --source at --replay-frac)")
    p.add_argument("--replay-frac", type=float, default=0.0,
                    help="fraction of each batch drawn from --replay-source instead of --source "
                         "(fine-tuning only, per arXiv:2608.16938's 30%% source-domain replay)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    vae, vae_ckpt = load_vae(args.vae_tag, dev)
    vae_ckpt_stat = vae_ckpt

    X, Y = load_xy(args.source, args.aug_tag)
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())

    print(f"[{args.tag}] encoding {len(X):,} rows through frozen {args.vae_tag} to build latent dataset...")
    latents = encode_dataset(vae, X, dev)
    latent_dim = latents.shape[1]
    lat_mean = latents.mean(axis=0)
    lat_std = latents.std(axis=0).clip(min=1e-6)
    latents_std = (latents - lat_mean) / lat_std

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    cond_targets = target_cond(Y, target_names, t_mean, t_std)

    lat_t = torch.tensor(latents_std, dtype=torch.float32)
    targ_t = torch.tensor(cond_targets, dtype=torch.float32)
    nfp_t = torch.tensor(X[:, 90], dtype=torch.float32)

    replay_lat_t = replay_targ_t = replay_nfp_t = None
    if args.replay_frac > 0:
        Xr, Yr = load_xy(args.replay_source)
        r_latents = encode_dataset(vae, Xr, dev)
        r_latents_std = (r_latents - lat_mean) / lat_std  # reuse fine-tune-set's own normalization
        Ytr = Yr.copy()
        for name in LOG_TARGET_NAMES:
            idx = target_names.index(name)
            Ytr[:, idx] = np.log(np.clip(Ytr[:, idx], 1e-12, None))
        r_cond_targets = target_cond(Yr, target_names, t_mean, t_std)
        replay_lat_t = torch.tensor(r_latents_std, dtype=torch.float32)
        replay_targ_t = torch.tensor(r_cond_targets, dtype=torch.float32)
        replay_nfp_t = torch.tensor(Xr[:, 90], dtype=torch.float32)
        print(f"[{args.tag}] replay pool: {len(Xr):,} rows from {args.replay_source}, "
              f"mixing at {args.replay_frac:.0%} per batch")

    schedule = make_schedule(args.T, device=dev)
    model = DiffusionDenoiser(coeff_dim=latent_dim, n_targets=len(target_names),
                               hidden=args.hidden, time_embed_dim=args.time_embed_dim).to(dev)
    if args.finetune_from:
        base_ckpt = torch.load(CKPT_DIR / f"{args.finetune_from}.pt", map_location=dev)
        model.load_state_dict(base_ckpt["model_state_dict"])
        print(f"[{args.tag}] fine-tuning from {args.finetune_from}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_main = len(lat_t)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] latent_dim={latent_dim} T={args.T} hidden={args.hidden} "
          f"params={n_params:,} main_rows={n_main:,}")

    batch = args.batch
    n_replay = int(batch * args.replay_frac)
    n_main_b = batch - n_replay
    n_batches_per_epoch = max(1, n_main // n_main_b)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(n_batches_per_epoch):
            idx = np.random.randint(0, n_main, size=n_main_b)
            xb, yb, nfpb = lat_t[idx], targ_t[idx], nfp_t[idx]
            if n_replay > 0:
                ridx = np.random.randint(0, len(replay_lat_t), size=n_replay)
                xb = torch.cat([xb, replay_lat_t[ridx]])
                yb = torch.cat([yb, replay_targ_t[ridx]])
                nfpb = torch.cat([nfpb, replay_nfp_t[ridx]])
            xb, yb, nfpb = xb.to(dev), yb.to(dev), nfpb.to(dev)
            cond = torch.cat([yb, nfp_one_hot(nfpb)], dim=-1)
            t = torch.randint(0, args.T, (xb.shape[0],), device=dev)
            x_t, noise = q_sample(xb, t, schedule)
            opt.zero_grad()
            eps_pred = model(x_t, t, cond)
            loss = ((eps_pred - noise) ** 2).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  loss {loss_sum / n_batches_per_epoch:9.5f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": args.hidden,
        "time_embed_dim": args.time_embed_dim,
        "T": args.T,
        "latent_dim": latent_dim,
        "lat_mean": lat_mean,
        "lat_std": lat_std,
        "target_names": target_names,
        "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean,
        "target_std": t_std,
        "nfp_values": NFP_VALUES,
        "vae_tag": args.vae_tag,
        "source": args.source,
        "finetuned_from": args.finetune_from,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

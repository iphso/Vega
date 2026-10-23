"""Conditional VAE for the mug domain -- reuses train_cvae.py's CVAE class
unchanged (same "n_nfp is really just extra conditioning dims" argument
that already let train_airfoil_cvae.py/train_torax_cvae.py reuse it
directly), with `n_nfp=0`: like TORAX, the mug domain has no discrete or
continuous AUX conditioning at all -- every design variable is already
part of `X` (see mug_oracle.py's own docstring).

§63: the model's own INTERNAL working vector is no longer identical to
`X`, though. `X` still has 9 real design columns (mug_oracle.PARAM_NAMES),
but struct_material_idx/handle_material_idx are categorical (which ONE
named material), not continuous like the other 7 -- see
mug_categorical.py's module docstring for the concrete broken-output
evidence this replaces. `mug_categorical.to_model_space(X)` one-hot-
expands those two columns (20 and 15 classes) into the model's own
`MODEL_DIM=42`-wide vector; the CVAE class itself is UNCHANGED (it has no
idea its 42 dims aren't all continuous), but this script's own loss now
treats the categorical blocks as classification heads (cross-entropy
against the true class, recovered by argmax of the real one-hot) instead
of folding them into the continuous MSE reconstruction term -- the actual
fix, not just a wider input.

No sanity filter -- unlike VMEC/airfoil/TORAX, no known-bad numerical tail
has been found in this domain yet (§48's own flagged open item; all 20,000
generated rows were finite/valid by construction). No LOG_TARGET_NAMES --
also still open (§47/§48), left empty for now; `handle_temp_60s_C`'s
floor-heavy distribution (76% within 0.1C of ambient, §48) is the target
most likely to eventually want one, not yet investigated here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mug_categorical as mc
from train_cvae import CVAE

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="mug_cvae_s0")
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

    coeff_dim = mc.MODEL_DIM  # 42: 7 continuous + 20-class + 15-class one-hot, not X.shape[1]=9
    coeffs_model = mc.to_model_space(X)
    cont_slice, cat_slices = mc.model_space_slices()

    cont_mean = coeffs_model[:, cont_slice].mean(axis=0)
    cont_std = coeffs_model[:, cont_slice].std(axis=0).clip(min=1e-6)
    coeffs_z = coeffs_model.copy()
    coeffs_z[:, cont_slice] = (coeffs_model[:, cont_slice] - cont_mean) / cont_std
    # categorical blocks are left as exact one-hot -- z-scoring a {0,1}
    # column per-class would distort argmax ordering across a row (each
    # class column would get its own mean/std from its own prior
    # frequency), which is exactly the recovered-true-class signal the
    # cross-entropy loss below depends on.

    cat_targets = np.stack(
        [np.argmax(coeffs_model[:, cs], axis=1) for _, cs in cat_slices], axis=1
    )  # (n, n_categorical_specs), int class index per column

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    targets_z = (Yt - t_mean) / t_std

    coeffs = torch.tensor(coeffs_z, dtype=torch.float32)
    targets = torch.tensor(targets_z, dtype=torch.float32)
    cats = torch.tensor(cat_targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(coeffs, targets, cats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = CVAE(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=args.latent_dim, hidden=args.hidden, n_nfp=0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} (7 continuous + "
          f"{'+'.join(str(n) for _, _, n in mc.CATEGORICAL_SPECS)} one-hot) n_targets={n_targets} "
          f"latent_dim={args.latent_dim} hidden={args.hidden} params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        recon_sum, cat_loss_sum, kl_sum, n_batches = 0.0, 0.0, 0.0, 0
        for xb, yb, catb in loader:
            xb, yb, catb = xb.to(dev), yb.to(dev), catb.to(dev)
            cond = yb
            opt.zero_grad()
            recon, mu, logvar = model(xb, cond)
            recon_loss = ((recon[:, cont_slice] - xb[:, cont_slice]) ** 2).mean(dim=0).sum()
            cat_loss = sum(F.cross_entropy(recon[:, cs], catb[:, j]) for j, (_, cs) in enumerate(cat_slices))
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0).sum()
            loss = recon_loss + cat_loss + args.beta * kl
            loss.backward()
            opt.step()
            recon_sum += recon_loss.item()
            cat_loss_sum += float(cat_loss)
            kl_sum += kl.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  recon {recon_sum / n_batches:9.5f}  "
                  f"cat_ce {cat_loss_sum / n_batches:8.4f}  kl {kl_sum / n_batches:9.5f}")

    # Real verification, not assumed: does the classification head actually
    # recover the true material, reconstructing its own training inputs
    # through the full encode/decode pass (same discipline as the scoring
    # model's own naive-baseline comparison) -- a random-guess baseline
    # would score 1/20=5.0% (struct) / 1/15=6.7% (handle).
    model.eval()
    with torch.no_grad():
        recon_all, _, _ = model(coeffs.to(dev), targets.to(dev))
        for j, (name, cs) in enumerate(cat_slices):
            pred = recon_all[:, cs].argmax(dim=1).cpu()
            acc = (pred == cats[:, j]).float().mean().item()
            n_classes = cs.stop - cs.start
            print(f"[{args.tag}] {name} reconstruction accuracy: {acc:.1%} (random guess baseline: {1 / n_classes:.1%})")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim, "hidden": args.hidden, "coeff_dim": coeff_dim,
        "coeff_mean": cont_mean, "coeff_std": cont_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

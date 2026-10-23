"""Conditional diffusion model for the mug domain -- reuses
train_diffusion.py's DiffusionDenoiser/make_schedule/q_sample/ddpm_sample
UNCHANGED, n_nfp=0 (no aux -- see train_mug_cvae.py's docstring). Data
loading and normalization copied verbatim from train_mug_cvae.py so all
three mug checkpoints train on identically-processed data. Same T=200,
already-re-tuned (§28) linear schedule as train_diffusion.py -- a
capability check (does the architecture generalize to a 4th domain), not
a per-domain quality sweep.

§63: same categorical fix as train_mug_cvae.py/train_mug_gan.py (see
mug_categorical.py's module docstring). The FIRST version of this fix
(one-hot-expand the two categorical columns via
`mug_categorical.to_model_space` and let plain Gaussian diffusion run over
those one-hot corners like any other coordinate, an "Analog Bits"-style
relaxation, no loss change) was tried and directly verified NOT to work at
this T=200/200-epoch budget -- decoded material classes on a real
conditioning batch came out uncorrelated with the true row (raw block
values flat in [-0.4, 0.5], top1-top2 gap ~0.1, indistinguishable from
noise) unlike cVAE (99.9% reconstruction accuracy via its own direct
cross-entropy loss) and the GAN (~80% top-class confidence via softmax-
before-critic adversarial pressure). Both of those work BECAUSE they have
an explicit training signal pushing the categorical block toward a
confident corner; plain noise-prediction MSE has none.

Fixed by giving diffusion the same kind of explicit signal: an auxiliary
cross-entropy loss on each categorical block of the model's own predicted
x0 (recoverable from eps_pred and x_t via the standard DDPM identity
`x0_pred = (x_t - sqrt(1-alpha_bar_t) eps_pred) / sqrt(alpha_bar_t)`),
SNR-weighted by alpha_bar_t per sample -- at large t (mostly noise),
alpha_bar_t is tiny, x0_pred is barely informed by x_t at all, and the
loss is weighted down accordingly (the model genuinely can't know the
material yet from a nearly-pure-noise input, so it isn't pushed to
pretend it does); at small t (mostly signal), alpha_bar_t is near 1 and
the loss is weighted at close to full strength, exactly where the
categorical class should be confidently recoverable. This is the same
family of technique as x0-parameterization/auxiliary-reconstruction losses
used elsewhere in the diffusion literature to sharpen semantic outputs,
not an invented one-off hack. The main noise-prediction loss is
UNCHANGED; this is purely additive.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mug_categorical as mc
from train_diffusion import DiffusionDenoiser, ddpm_sample, make_schedule, q_sample

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cat-weight", type=float, default=2.0,
                    help="scale on the auxiliary SNR-weighted categorical cross-entropy term (§63). "
                         "1.0 (unweighted) only reached ~30% top-class confidence (near cVAE/GAN's "
                         "99.9%/~80%); 5.0 overfit the auxiliary term hard enough to diverge the "
                         "continuous geometry outputs entirely (sampled radii/thickness in the "
                         "millions). 2.0 was the checked middle ground: ~47-57% confidence with "
                         "continuous outputs still in a plausible range.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="mug_diffusion_s0")
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
    cont_slice, cat_slices = mc.model_space_slices()

    cont_mean = coeffs_model[:, cont_slice].mean(axis=0)
    cont_std = coeffs_model[:, cont_slice].std(axis=0).clip(min=1e-6)
    coeffs_z = coeffs_model.copy()
    coeffs_z[:, cont_slice] = (coeffs_model[:, cont_slice] - cont_mean) / cont_std

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean = Yt.mean(axis=0)
    t_std = Yt.std(axis=0).clip(min=1e-6)
    targets_z = (Yt - t_mean) / t_std

    cat_targets = np.stack(
        [np.argmax(coeffs_model[:, cs], axis=1) for _, cs in cat_slices], axis=1
    )  # (n, n_categorical_specs), int class index per column

    coeffs = torch.tensor(coeffs_z, dtype=torch.float32)
    targets = torch.tensor(targets_z, dtype=torch.float32)
    cats = torch.tensor(cat_targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(coeffs, targets, cats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    schedule = make_schedule(args.T, device=dev)
    model = DiffusionDenoiser(coeff_dim=coeff_dim, n_targets=n_targets, hidden=args.hidden,
                               time_embed_dim=args.time_embed_dim, n_nfp=0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] rows={len(dataset):,} coeff_dim={coeff_dim} (7 continuous + "
          f"{'+'.join(str(n) for _, _, n in mc.CATEGORICAL_SPECS)} one-hot) n_targets={n_targets} "
          f"T={args.T} hidden={args.hidden} params={n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, cat_loss_sum, n_batches = 0.0, 0.0, 0
        for xb, yb, catb in loader:
            xb, yb, catb = xb.to(dev), yb.to(dev), catb.to(dev)
            cond = yb
            t = torch.randint(0, args.T, (xb.shape[0],), device=dev)
            x_t, noise = q_sample(xb, t, schedule)
            opt.zero_grad()
            eps_pred = model(x_t, t, cond)
            noise_loss = ((eps_pred - noise) ** 2).mean()

            # §63: auxiliary SNR-weighted cross-entropy on the model's own
            # predicted x0 (standard DDPM identity) -- see module docstring
            # for why plain noise-prediction MSE alone doesn't sharpen the
            # categorical blocks into a confident corner.
            ab = schedule["alpha_bars"][t][:, None]
            x0_pred = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-3)
            cat_loss = args.cat_weight * sum(
                (ab.squeeze(-1) * F.cross_entropy(x0_pred[:, cs], catb[:, j], reduction="none")).mean()
                for j, (_, cs) in enumerate(cat_slices)
            )
            loss = noise_loss + cat_loss
            loss.backward()
            opt.step()
            loss_sum += noise_loss.item()
            cat_loss_sum += float(cat_loss)
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  loss {loss_sum / n_batches:9.5f}  cat_ce(SNR-wtd) {cat_loss_sum / n_batches:8.4f}")

    # Real verification, not assumed: does the reverse process actually
    # land near a confident one-hot corner for each categorical block, or
    # does it smear across classes (which would mean the plain-Gaussian
    # treatment of a categorical variable isn't actually working)? Sampled
    # conditioned on a batch of REAL target rows -- not expected to
    # reproduce that exact row's own material (a different material can
    # legitimately hit the same target metrics), just to land confidently
    # on SOME class rather than a smeared blend.
    model.eval()
    with torch.no_grad():
        k = min(2000, len(targets))
        sampled = ddpm_sample(model, targets[:k].to(dev), schedule, coeff_dim=coeff_dim, device=dev).cpu()
        for name, cs in cat_slices:
            probs = torch.softmax(sampled[:, cs], dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()
            print(f"[{args.tag}] {name} sampled-block mean top-class softmax confidence: {confidence:.1%} "
                  f"(1/{cs.stop - cs.start}={1 / (cs.stop - cs.start):.1%} would mean no signal at all)")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": args.hidden, "time_embed_dim": args.time_embed_dim, "T": args.T,
        "coeff_dim": coeff_dim,
        "coeff_mean": cont_mean, "coeff_std": cont_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

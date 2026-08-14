"""Joint VAE + surrogate training: one shared encoder, two separate
projection heads trained together end-to-end on a single combined loss --
not a frozen pretrained VAE with a separate MLP bolted on afterward.

  encoder(coeffs, nfp) -> mu, logvar
      -> decoder(z, nfp) -> reconstructed coefficients   (VAE loss)
      -> surrogate(mu, nfp, symmetry_flag) -> target metrics  (metrics loss)

Both projection heads backprop into the same shared encoder every step, so
the latent it learns is shaped by *both* objectives -- reconstructing the
Fourier coefficients and predicting the physical targets -- unlike a VAE
pretrained on reconstruction alone with a downstream regressor frozen on
top of it. The surrogate half reuses scripts/train.py's DualPathMLP
unchanged (single-path, no spatial branch -- there's no raw r_cos/z_sin
grid in latent space), just fed [mu, nfp, symmetry_flag] instead of the
raw 92-dim input; the loss is the same learned-uncertainty-weighted sum
over targets as everywhere else in this project (no target normalization).

Companion to scripts/train.py's standard recipe (raw coefficients -> a
DualPathMLP directly, no VAE involved) -- run with the same --split and
--seed for an apples-to-apples architecture comparison. Data must respect
train/val/test split discipline throughout (the VAE reconstruction loss
included), unlike scripts/train_vae.py's standalone recipe which
deliberately trains on the full real dataset regardless of split.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from soap import SOAP
from torch.utils.data import DataLoader, TensorDataset
from train import IDX_NFP, LOG_TARGET_NAMES, DualPathMLP, load_split, print_breakdown
from train_vae import VAE, nfp_one_hot, vae_loss

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


class JointVAESurrogate(torch.nn.Module):
    def __init__(
        self,
        n_targets,
        vae_latent_dim,
        vae_hidden,
        trunk_hidden,
        trunk_latent,
        head_hidden,
        priority_weight,
        log_target_mask,
    ):
        super().__init__()
        self.vae = VAE(coeff_dim=90, latent_dim=vae_latent_dim, hidden=vae_hidden)
        self.surrogate = DualPathMLP(
            vae_latent_dim + 2,
            n_targets,
            latent_dim=trunk_latent,
            hidden=trunk_hidden,
            head_hidden=head_hidden,
            priority_weight=priority_weight,
            use_spatial=False,
            trunk_arch="mlp",
            log_target_mask=log_target_mask,
            objective="regression",
        )

    def forward(self, coeffs_std, cond, nfp_and_flag):
        mu, logvar = self.vae.encode(coeffs_std, cond)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.vae.decode(z, cond)
        pred = self.surrogate(torch.cat([mu, nfp_and_flag], dim=1))
        return recon, mu, logvar, pred


def step(model, xb, yb, coeff_mean, coeff_std, beta):
    coeffs = (xb[:, :90] - coeff_mean) / coeff_std
    cond = nfp_one_hot(xb[:, IDX_NFP])
    nfp_and_flag = xb[:, IDX_NFP : IDX_NFP + 2]
    recon, mu, logvar, pred = model(coeffs, cond, nfp_and_flag)
    v_loss, recon_l, kl_l = vae_loss(recon, coeffs, mu, logvar, beta)
    m_loss, per_task_mse = model.surrogate.weighted_loss(pred, yb)
    return v_loss + m_loss, v_loss.detach(), m_loss.detach(), per_task_mse


def evaluate(model, loader, dev, coeff_mean, coeff_std, beta, n_targets):
    model.eval()
    total_v, total_m, total_mse, n_batches = (
        0.0,
        0.0,
        torch.zeros(n_targets, device=dev),
        0,
    )
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss, v_loss, m_loss, per_task_mse = step(
                model, xb, yb, coeff_mean, coeff_std, beta
            )
            total_v += v_loss.item()
            total_m += m_loss.item()
            total_mse += per_task_mse
            n_batches += 1
    return total_v / n_batches, total_m / n_batches, total_mse / n_batches


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "soap"])
    p.add_argument("--soap-weight-decay", type=float, default=0.01)
    p.add_argument("--soap-precondition-frequency", type=int, default=50)
    p.add_argument("--soap-max-precond-dim", type=int, default=1024)
    p.add_argument("--vae-latent-dim", type=int, default=32)
    p.add_argument("--vae-hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.01, help="VAE KL weight")
    p.add_argument(
        "--hidden", type=int, default=256, help="surrogate trunk hidden width"
    )
    p.add_argument(
        "--latent", type=int, default=128, help="surrogate trunk output width"
    )
    p.add_argument(
        "--head-hidden", type=int, default=64, help="per-target head hidden width"
    )
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--log-targets", action="store_true")
    p.add_argument("--tag", default="joint")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    dev = torch.device(args.device)
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    feature_names = json.loads((OUT_DIR / "feature_names.json").read_text())
    coeff_mean = torch.tensor(
        [meta["feature_stats"][n]["mean"] for n in feature_names[:90]],
        dtype=torch.float32,
        device=dev,
    )
    coeff_std = torch.tensor(
        [meta["feature_stats"][n]["std"] for n in feature_names[:90]],
        dtype=torch.float32,
        device=dev,
    ).clamp_min(1e-6)
    data_dir = (OUT_DIR / "splits" / args.split) if args.split else None

    X_train, Y_train = load_split("train", data_dir)
    X_val, Y_val = load_split("val", data_dir)
    n_targets = Y_train.shape[1]

    log_target_mask = torch.zeros(n_targets, dtype=torch.bool)
    if args.log_targets:
        for name in LOG_TARGET_NAMES:
            log_target_mask[target_names.index(name)] = True

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=args.batch, shuffle=False
    )

    model = JointVAESurrogate(
        n_targets,
        args.vae_latent_dim,
        args.vae_hidden,
        args.hidden,
        args.latent,
        args.head_hidden,
        priority_weight=torch.ones(n_targets),
        log_target_mask=log_target_mask,
    ).to(dev)

    if args.optimizer == "soap":
        opt = SOAP(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.soap_weight_decay,
            precondition_frequency=args.soap_precondition_frequency,
            max_precond_dim=args.soap_max_precond_dim,
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{args.tag}] joint vae+surrogate  vae_latent={args.vae_latent_dim}  trunk_hidden={args.hidden}  "
        f"trunk_latent={args.latent}  optimizer={args.optimizer}  lr={args.lr}  params={n_params:,}  "
        f"seed={args.seed}  rows={len(X_train):,}"
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, v_sum, m_sum, n_batches = 0.0, 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss, v_loss, m_loss, _ = step(
                model, xb, yb, coeff_mean, coeff_std, args.beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += loss.item()
            v_sum += v_loss.item()
            m_sum += m_loss.item()
            n_batches += 1

        is_val_epoch = epoch % args.val_interval == 0 or epoch == args.epochs
        if not is_val_epoch:
            print(
                f"epoch {epoch:3d}  train_loss {loss_sum / n_batches:9.4f}  "
                f"(vae {v_sum / n_batches:8.4f}  metrics {m_sum / n_batches:8.4f})"
            )
            continue

        val_v, val_m, val_mse = evaluate(
            model, val_loader, dev, coeff_mean, coeff_std, args.beta, n_targets
        )
        val_total = val_v + val_m
        print(
            f"epoch {epoch:3d}  train_loss {loss_sum / n_batches:9.4f}  val_loss {val_total:9.4f}  "
            f"(val_vae {val_v:8.4f}  val_metrics {val_m:8.4f})  mean_val_rmse {val_mse.sqrt().mean().item():.5f}"
        )
        print_breakdown("val", val_mse, target_names)

        if val_total < best_val:
            best_val = val_total
            torch.save(
                {
                    "vae_state_dict": model.vae.state_dict(),
                    "surrogate_state_dict": model.surrogate.state_dict(),
                    "vae_latent_dim": args.vae_latent_dim,
                    "vae_hidden": args.vae_hidden,
                    "beta": args.beta,
                    "coeff_mean": coeff_mean.cpu(),
                    "coeff_std": coeff_std.cpu(),
                    "hidden": args.hidden,
                    "latent": args.latent,
                    "head_hidden": args.head_hidden,
                    "n_targets": n_targets,
                    "log_target_mask": log_target_mask,
                    "target_names": target_names,
                    "split": args.split,
                    "epoch": epoch,
                    "val_loss": val_total,
                },
                CKPT_DIR / f"{args.tag}.pt",
            )

    train_seconds = time.perf_counter() - train_start
    ckpt_path = CKPT_DIR / f"{args.tag}.pt"
    print(
        f"training done. best val_loss {best_val:.4f}  params={n_params:,}  "
        f"train_time={train_seconds:.1f}s  checkpoint saved to {ckpt_path}"
    )

    # Final test-set evaluation, using the best checkpoint.
    ckpt = torch.load(ckpt_path, map_location=dev)
    test_model = JointVAESurrogate(
        ckpt["n_targets"],
        ckpt["vae_latent_dim"],
        ckpt["vae_hidden"],
        ckpt["hidden"],
        ckpt["latent"],
        ckpt["head_hidden"],
        priority_weight=torch.ones(ckpt["n_targets"]),
        log_target_mask=ckpt["log_target_mask"],
    ).to(dev)
    test_model.vae.load_state_dict(ckpt["vae_state_dict"])
    test_model.surrogate.load_state_dict(ckpt["surrogate_state_dict"])

    X_test, Y_test = load_split("test", data_dir)
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test), batch_size=args.batch, shuffle=False
    )
    test_v, test_m, test_mse = evaluate(
        test_model,
        test_loader,
        dev,
        ckpt["coeff_mean"].to(dev),
        ckpt["coeff_std"].to(dev),
        ckpt["beta"],
        ckpt["n_targets"],
    )

    print(f"\n=== TEST (checkpoint from epoch {ckpt['epoch']}) ===")
    print(f"test_vae_loss {test_v:9.4f}  test_metrics_loss {test_m:9.4f}")
    print_breakdown("test", test_mse, ckpt["target_names"])


if __name__ == "__main__":
    main()

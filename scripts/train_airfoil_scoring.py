"""Forward scoring model for the airfoil domain: design -> predicted
performance (cl, cd, cm, l_over_d), the direction
train_airfoil_cvae.py/train_airfoil_diffusion.py/train_airfoil_gan.py all
answer in reverse. Closes a real, confirmed gap (checked directly against
the checkpoint inventory, not assumed): every forward regression checkpoint
on disk before this one is stellarator-only (`reg_mlp_*`/`reg_siren_*`/
`contrastive_*`, all trained via `train.py` against VMEC++'s 90-coefficient
representation) -- nothing analogous existed for airfoils at all.

First shipped as a deliberately-scoped plain-MLP baseline (§29), the same
place VMEC++'s own scoring work started (§1) before that domain's own
~8-session architecture search happened. This version (§32) adds a real,
if intentionally scoped-down, architecture search on top of that baseline:
trunk arch (mlp/siren/half_siren, via the now-shared `nn_trunks.py` --
train.py's own trunks, extracted so both domains use identical code, not a
reimplementation) x optimizer (adam/soap, via the same `soap.py` used by
train.py). Scoped down deliberately, not by omission: airfoils' input space
(18 dims: 16 CST coefficients + Reynolds + alpha) is far smaller than
VMEC++'s 90+5, so a single hidden width per arch is compared rather than
VMEC's own separate small-capacity-then-scale-up campaign (§2 then §4) --
if a trunk/optimizer win shows up here at all, capacity scaling is a
natural follow-up, not assumed necessary up front. `--ensemble-tags`
(see ensemble_eval_airfoil_scoring.py) covers the "free calibrated
uncertainty" half of what VMEC's own ensembling explored (§4).

Same physical-sanity filter and normalization discipline as the generation
scripts (cd log-transformed, everything z-scored on whatever's actually
trained on, never full-dataset stats -- same leakage discipline as this
project's very first normalization findings, §7).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nn_trunks import build_trunk
from soap import SOAP

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

LOG_TARGET_NAMES = ["cd"]  # matches airfoil_oracle.py's own convention


class ScoringMLP(nn.Module):
    """Kept for backward-compat with the §29 baseline checkpoint format
    (plain stacked Linear+ReLU blocks, single Linear readout -- no separate
    trunk/head split). ScoringTrunkModel below is what --trunk-arch actually
    trains against; passing --trunk-arch mlp gives an architecturally
    similar but not bit-identical model (an extra latent projection + ReLU
    before the readout, matching train.py's own trunk convention), used for
    the arch-search comparison so mlp/siren/half_siren are evaluated under
    literally the same head/loss/training loop, not just "structurally
    similar" ones.
    """

    def __init__(self, in_dim=18, n_targets=4, hidden=256, n_blocks=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ScoringTrunkModel(nn.Module):
    """trunk (mlp/siren/half_siren, from nn_trunks.build_trunk) -> single
    Linear(latent_dim, n_targets) readout. The arch-search harness for this
    domain -- every --trunk-arch choice shares this exact head/loss, so a
    win is attributable to the trunk alone.
    """

    def __init__(self, in_dim, n_targets, trunk_arch="mlp", hidden=256, latent_dim=128, n_blocks=3):
        super().__init__()
        self.trunk = build_trunk(trunk_arch, in_dim, hidden, latent_dim, n_blocks=n_blocks)
        self.head = nn.Linear(latent_dim, n_targets)

    def forward(self, x):
        return self.head(self.trunk(x))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--latent", type=int, default=128, help="--trunk-arch != legacy only: trunk output dim before the readout head")
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="airfoil_reg_mlp_s0")
    p.add_argument("--dataset-tag", default="airfoil")
    p.add_argument("--trunk-arch", default="legacy", choices=["legacy", "mlp", "siren", "half_siren"],
                    help="'legacy': §29's original ScoringMLP (stacked Linear+ReLU, no separate trunk/head "
                         "split) -- kept as the default so old invocations reproduce exactly. Any other "
                         "value uses ScoringTrunkModel (nn_trunks.build_trunk + a Linear readout head), "
                         "the arch-search harness (§32).")
    p.add_argument("--optimizer", default="adam", choices=["adam", "soap"],
                    help="soap: Shampoo-preconditioned Adam (scripts/soap.py), same as train.py's own "
                         "--optimizer soap. Usually wants a higher lr than Adam.")
    p.add_argument("--soap-weight-decay", type=float, default=0.01)
    p.add_argument("--soap-precondition-frequency", type=int, default=50)
    p.add_argument("--soap-max-precond-dim", type=int, default=1024)
    p.add_argument("--source", default="random", choices=["random", "split"],
                    help="'random': fresh random 80/10/10 split each run (this script's original behavior). "
                         "'split': load a precomputed split dir under output/splits/<--split>/ (e.g. the "
                         "cluster split from make_splits_generic.py --domain airfoil) -- lets this script "
                         "answer the same random-vs-cluster generalization-gap question §6 answered for VMEC++.")
    p.add_argument("--split", default="airfoil_cluster", help="--source split only: output/splits/<name>/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    target_names = json.loads((OUT_DIR / f"{args.dataset_tag}_target_names.json").read_text())
    n_targets = len(target_names)
    cd_col, lod_col = target_names.index("cd"), target_names.index("l_over_d")

    if args.source == "split":
        split_dir = OUT_DIR / "splits" / args.split
        train_npz, val_npz, test_npz = (np.load(split_dir / f"{s}.npz") for s in ("train", "val", "test"))
        X = np.concatenate([train_npz["X"], val_npz["X"], test_npz["X"]])
        Y = np.concatenate([train_npz["Y"], val_npz["Y"], test_npz["Y"]])
        n_tr, n_va = len(train_npz["X"]), len(val_npz["X"])
        train_idx = np.arange(0, n_tr)
        val_idx = np.arange(n_tr, n_tr + n_va)
        test_idx = np.arange(n_tr + n_va, len(X))
        print(f"[{args.tag}] loaded split={args.split} (sanity filter already applied when the split was built) "
              f"-> train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    else:
        X = np.load(OUT_DIR / f"{args.dataset_tag}_X.npy")
        Y = np.load(OUT_DIR / f"{args.dataset_tag}_Y.npy")
        # Same physical-sanity filter train_airfoil_cvae.py uses -- see that
        # file's comment for the 127/50,011 near-zero-cd finding this guards
        # against.
        sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
        n_dropped = len(Y) - sane.sum()
        if n_dropped:
            print(f"[{args.tag}] dropping {n_dropped}/{len(Y)} rows with non-physical cd/l_over_d")
        X, Y = X[sane], Y[sane]

        # Random 80/10/10 split -- the same split type VMEC++'s early work
        # (§1-5) used before §6 found it substantially overstates
        # generalization relative to a cluster split. Use --source split
        # (with make_splits_generic.py's cluster split) for the harder,
        # comparable-to-VMEC++ generalization check.
        n = len(X)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)
        n_val = int(n * args.val_frac)
        n_test = int(n * args.test_frac)
        test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
        print(f"[{args.tag}] {n:,} rows after filtering -> train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    in_mean, in_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0).clip(min=1e-6)

    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    t_mean, t_std = Yt[train_idx].mean(axis=0), Yt[train_idx].std(axis=0).clip(min=1e-6)

    def make_tensors(idx):
        x = torch.tensor((X[idx] - in_mean) / in_std, dtype=torch.float32)
        y = torch.tensor((Yt[idx] - t_mean) / t_std, dtype=torch.float32)
        return x, y

    x_train, y_train = make_tensors(train_idx)
    x_val, y_val = make_tensors(val_idx)
    x_test, y_test = make_tensors(test_idx)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    if args.trunk_arch == "legacy":
        model = ScoringMLP(in_dim=X.shape[1], n_targets=n_targets, hidden=args.hidden, n_blocks=args.n_blocks).to(dev)
    else:
        model = ScoringTrunkModel(in_dim=X.shape[1], n_targets=n_targets, trunk_arch=args.trunk_arch,
                                   hidden=args.hidden, latent_dim=args.latent, n_blocks=args.n_blocks).to(dev)
    if args.optimizer == "soap":
        opt = SOAP(model.parameters(), lr=args.lr, weight_decay=args.soap_weight_decay,
                   precondition_frequency=args.soap_precondition_frequency,
                   max_precond_dim=args.soap_max_precond_dim)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] in_dim={X.shape[1]} n_targets={n_targets} trunk_arch={args.trunk_arch} "
          f"optimizer={args.optimizer} hidden={args.hidden} n_blocks={args.n_blocks} params={n_params:,}")

    def rmse_by_target(x, y_z):
        model.eval()
        with torch.no_grad():
            pred_z = model(x.to(dev)).cpu()
        # back out of z-scoring (and the log transform for cd) into real units
        pred = pred_z.numpy() * t_std + t_mean
        real = y_z.numpy() * t_std + t_mean
        for name in LOG_TARGET_NAMES:
            idx = target_names.index(name)
            pred[:, idx] = np.exp(pred[:, idx])
            real[:, idx] = np.exp(real[:, idx])
        return np.sqrt(((pred - real) ** 2).mean(axis=0))

    best_val_rmse_mean = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            val_rmse = rmse_by_target(x_val, y_val)
            val_mean = float(val_rmse.mean())
            print(f"epoch {epoch:4d}  train_loss(z) {loss_sum / n_batches:9.5f}  "
                  f"val_rmse_mean {val_mean:9.5f}  " +
                  "  ".join(f"{n}={v:.4f}" for n, v in zip(target_names, val_rmse)))
            if val_mean < best_val_rmse_mean:
                best_val_rmse_mean = val_mean
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_rmse = rmse_by_target(x_test, y_test)
    print(f"\n[{args.tag}] final test RMSE (real units, best-val-epoch checkpoint):")
    for name, v in zip(target_names, test_rmse):
        print(f"  {name:12s} {v:.5f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": args.hidden, "n_blocks": args.n_blocks, "in_dim": X.shape[1],
        "trunk_arch": args.trunk_arch, "latent": args.latent, "optimizer": args.optimizer,
        "in_mean": in_mean, "in_std": in_std,
        "target_names": target_names, "log_target_names": LOG_TARGET_NAMES,
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
        "test_rmse": {n: float(v) for n, v in zip(target_names, test_rmse)},
        "split": args.source if args.source == "split" else "random",
        "split_name": args.split if args.source == "split" else None,
        "seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

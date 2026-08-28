"""Forward scoring model for the mug domain: design -> predicted
(temp_at_2h_C, mass_kg, touch_temp_60s_C, handle_temp_60s_C), the direction
train_mug_cvae.py/train_mug_diffusion.py/train_mug_gan.py all answer in
reverse. First baseline (§49) -- plain MLP, same deliberately-simple
starting point every other domain's scoring work began at (VMEC++'s own
§1, airfoil's §29) before any architecture search.

Compares against a computed (not assumed) naive predict-the-training-mean
baseline, same discipline as airfoil's §29 -- important here specifically
because `handle_temp_60s_C` is heavily floor-dominated (§48: 76% of rows
sit within 0.1C of ambient), so a model could look deceptively good on raw
RMSE by mostly predicting the floor; the mean-baseline comparison is what
would actually reveal that, not raw RMSE alone.

No LOG_TARGET_NAMES applied -- open item from §47/§48, not yet rigorously
checked the way VMEC/airfoil's log-targets were originally identified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


class ScoringMLP(nn.Module):
    def __init__(self, in_dim, n_targets, hidden=128, n_blocks=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="mug_reg_mlp_s0")
    p.add_argument("--dataset-tag", default="mug")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    target_names = json.loads((OUT_DIR / f"{args.dataset_tag}_target_names.json").read_text())
    n_targets = len(target_names)
    X = np.load(OUT_DIR / f"{args.dataset_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{args.dataset_tag}_Y.npy")

    n = len(X)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"[{args.tag}] {n:,} rows -> train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    in_mean, in_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0).clip(min=1e-6)
    t_mean, t_std = Y[train_idx].mean(axis=0), Y[train_idx].std(axis=0).clip(min=1e-6)

    def make_tensors(idx):
        x = torch.tensor((X[idx] - in_mean) / in_std, dtype=torch.float32)
        y = torch.tensor((Y[idx] - t_mean) / t_std, dtype=torch.float32)
        return x, y

    x_train, y_train = make_tensors(train_idx)
    x_val, y_val = make_tensors(val_idx)
    x_test, y_test = make_tensors(test_idx)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = ScoringMLP(in_dim=X.shape[1], n_targets=n_targets, hidden=args.hidden, n_blocks=args.n_blocks).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] in_dim={X.shape[1]} n_targets={n_targets} hidden={args.hidden} "
          f"n_blocks={args.n_blocks} params={n_params:,}")

    def rmse_by_target(x, y_z):
        model.eval()
        with torch.no_grad():
            pred_z = model(x.to(dev)).cpu()
        pred = pred_z.numpy() * t_std + t_mean
        real = y_z.numpy() * t_std + t_mean
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

    # naive predict-the-training-mean baseline, computed not assumed (§29's
    # own convention) -- important here specifically because handle_temp_60s_C
    # is 76% floor-dominated (§48), so raw RMSE alone could look deceptively good.
    train_mean_real = Y[train_idx].mean(axis=0)
    naive_rmse = np.sqrt(((Y[test_idx] - train_mean_real) ** 2).mean(axis=0))

    print(f"\n[{args.tag}] final test RMSE (real units, best-val-epoch checkpoint) vs. naive mean baseline:")
    for name, v, nv in zip(target_names, test_rmse, naive_rmse):
        reduction = (1 - v / nv) * 100 if nv > 0 else float("nan")
        print(f"  {name:20s} model={v:.5f}  naive_mean={nv:.5f}  error_reduction={reduction:.1f}%")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": args.hidden, "n_blocks": args.n_blocks, "in_dim": X.shape[1],
        "in_mean": in_mean, "in_std": in_std,
        "target_names": target_names, "log_target_names": [],
        "target_mean": t_mean, "target_std": t_std,
        "dataset_tag": args.dataset_tag,
        "test_rmse": {n: float(v) for n, v in zip(target_names, test_rmse)},
        "naive_rmse": {n: float(v) for n, v in zip(target_names, naive_rmse)},
        "seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac,
    }, CKPT_DIR / f"{args.tag}.pt")
    print(f"saved {CKPT_DIR / f'{args.tag}.pt'}")


if __name__ == "__main__":
    main()

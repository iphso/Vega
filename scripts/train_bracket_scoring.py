"""Forward scoring model for the bracket domain: density field (shape) ->
predicted compliance, the direction train_bracket_gan.py answers in
reverse -- same role as train_mug_scoring.py/train.py play in their own
domains. First baseline: plain MLP, same deliberately-simple starting
point every other domain's scoring work began at.

One model PER FORCE LAYOUT, not one model across layouts -- the dataset
(bracket_generate_dataset_warp.py) only covers 3 fixed layouts with
different resolutions (v1/candidate_12: 4800 cells, v2: 12544 cells), so
there's no shared input dimensionality to condition a single cross-layout
model on yet (see EXPERIMENT_LOG's bracket section for why that's a
deliberately deferred, bigger follow-up, not an oversight here).

Compares against a computed (not assumed) naive predict-the-training-mean
baseline, same discipline as every other domain's scoring script.
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
    def __init__(self, in_dim, hidden=256, n_blocks=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", required=True, choices=["v1_symmetric", "v2_asymmetric", "candidate_12"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    tag = f"bracket_scoring_{args.layout}_s{args.seed}"

    data = np.load(OUT_DIR / f"bracket_dataset_{args.layout}.npz")
    X = data["rho"].astype(np.float32)          # (n_rows, n_cells)
    y = np.log(data["compliance"].astype(np.float32))  # log-space: compliance spans ~0.25-350x, heavy-tailed

    n = len(X)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"[{tag}] {n:,} rows (n_cells={X.shape[1]}) -> train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    x_mean, x_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0).clip(min=1e-6)
    y_mean, y_std = y[train_idx].mean(), y[train_idx].std().clip(min=1e-6)

    def make_tensors(idx):
        xt = torch.tensor((X[idx] - x_mean) / x_std, dtype=torch.float32)
        yt = torch.tensor((y[idx] - y_mean) / y_std, dtype=torch.float32)
        return xt, yt

    x_train, y_train = make_tensors(train_idx)
    x_val, y_val = make_tensors(val_idx)
    x_test, y_test = make_tensors(test_idx)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                          batch_size=args.batch, shuffle=True)

    model = ScoringMLP(in_dim=X.shape[1], hidden=args.hidden, n_blocks=args.n_blocks).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[{tag}] in_dim={X.shape[1]} hidden={args.hidden} n_blocks={args.n_blocks} params={n_params:,}")

    def rmse_log_and_real(x, y_z):
        model.eval()
        with torch.no_grad():
            pred_z = model(x.to(dev)).cpu()
        pred_log = pred_z.numpy() * y_std + y_mean
        real_log = y_z.numpy() * y_std + y_mean
        rmse_log = float(np.sqrt(((pred_log - real_log) ** 2).mean()))
        rmse_real = float(np.sqrt(((np.exp(pred_log) - np.exp(real_log)) ** 2).mean()))
        return rmse_log, rmse_real

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs:
            val_rmse_log, val_rmse_real = rmse_log_and_real(x_val, y_val)
            print(f"epoch {epoch:4d}  train_loss(z) {loss_sum / n_batches:9.5f}  "
                  f"val_rmse_log {val_rmse_log:.4f}  val_rmse_compliance {val_rmse_real:.4f}")
            if val_rmse_log < best_val:
                best_val = val_rmse_log
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_rmse_log, test_rmse_real = rmse_log_and_real(x_test, y_test)

    naive_pred_log = y[train_idx].mean()
    naive_rmse_real = float(np.sqrt(((np.exp(y[test_idx]) - np.exp(naive_pred_log)) ** 2).mean()))
    reduction = (1 - test_rmse_real / naive_rmse_real) * 100 if naive_rmse_real > 0 else float("nan")
    print(f"\n[{tag}] test RMSE (real compliance units, best-val-epoch): model={test_rmse_real:.4f}  "
          f"naive_mean={naive_rmse_real:.4f}  error_reduction={reduction:.1f}%")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": args.hidden, "n_blocks": args.n_blocks, "in_dim": X.shape[1],
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "layout": args.layout, "log_space": True,
        "test_rmse_compliance": test_rmse_real, "naive_rmse_compliance": naive_rmse_real,
        "seed": args.seed,
    }, CKPT_DIR / f"{tag}.pt")
    print(f"saved {CKPT_DIR / f'{tag}.pt'}")


if __name__ == "__main__":
    main()

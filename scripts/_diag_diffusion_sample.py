import json
from pathlib import Path
import numpy as np
import torch
from train_diffusion import DiffusionDenoiser, make_schedule

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
dev = torch.device("cpu")

ckpt = torch.load(CKPT_DIR / "diffusion_targets_full_s0.pt", map_location=dev)
target_names = ckpt["target_names"]
n_targets = len(target_names)
model = DiffusionDenoiser(coeff_dim=90, n_targets=n_targets, hidden=ckpt["hidden"],
                           time_embed_dim=ckpt["time_embed_dim"]).to(dev)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
T = ckpt["T"]
schedule = make_schedule(T, device=dev)
print(f"T={T}")

from make_splits import LOG_TARGET_NAMES
from train_vae import nfp_one_hot
X = np.load(OUT_DIR / "X.npy")
Y = np.load(OUT_DIR / "Y.npy")
t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
Yt = Y.copy()
for name in LOG_TARGET_NAMES:
    idx = target_names.index(name)
    Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
Yz = (Yt - t_mean) / t_std

idx = 0
anchor_z = torch.tensor(Yz[idx:idx+1], dtype=torch.float32)
nfp_t = torch.tensor([float(X[idx, 90])], dtype=torch.float32)
cond = torch.cat([anchor_z, nfp_one_hot(nfp_t)], dim=-1)
k = 5
cond_rep = cond.repeat(k, 1)

# manual ancestral sampling with instrumentation
torch.manual_seed(0)
x = torch.randn(k, 90)
print(f"x_T stats: min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} std={x.std():.3f}")
checkpoints_t = [T-1, int(T*0.75), int(T*0.5), int(T*0.25), 10, 5, 1, 0]
with torch.no_grad():
    for t in reversed(range(T)):
        t_batch = torch.full((k,), t, dtype=torch.long)
        eps_pred = model(x, t_batch, cond_rep)
        beta_t, alpha_t, ab_t = schedule["betas"][t], schedule["alphas"][t], schedule["alpha_bars"][t]
        mean = (x - beta_t / (1 - ab_t).sqrt() * eps_pred) / alpha_t.sqrt()
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + beta_t.sqrt() * noise
        else:
            x = mean
        if t in checkpoints_t:
            print(f"t={t:4d}  beta={beta_t.item():.4f} alpha={alpha_t.item():.4f}  "
                  f"x: min={x.min().item():10.3f} max={x.max().item():10.3f} mean={x.mean().item():10.3f} "
                  f"std={x.std().item():10.3f}  eps_pred: min={eps_pred.min().item():8.3f} max={eps_pred.max().item():8.3f}")

print("\nfinal decoded (standardized coeff space) stats:")
print(f"min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f} std={x.std().item():.3f}")
print(f"any NaN: {torch.isnan(x).any().item()}  any Inf: {torch.isinf(x).any().item()}")

coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
params = x.numpy() * coeff_std + coeff_mean
print(f"\nunstandardized params stats: min={params.min():.3f} max={params.max():.3f}")
print("(real boundary coefficients are typically O(0.001-1) in magnitude -- compare against this)")

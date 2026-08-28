from pathlib import Path
import numpy as np
import torch
from train_diffusion import DiffusionDenoiser, make_schedule

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
dev = torch.device("cpu")

ckpt = torch.load(CKPT_DIR / "airfoil_diffusion_s0.pt", map_location=dev)
target_names = ckpt["target_names"]
n_targets = len(target_names)
coeff_dim = ckpt["coeff_dim"]
model = DiffusionDenoiser(coeff_dim=coeff_dim, n_targets=n_targets, hidden=ckpt["hidden"],
                           time_embed_dim=ckpt["time_embed_dim"], n_nfp=2).to(dev)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
T = ckpt["T"]
schedule = make_schedule(T, device=dev)

X = np.load(OUT_DIR / "airfoil_X.npy")
Y = np.load(OUT_DIR / "airfoil_Y.npy")
cd_col, lod_col = target_names.index("cd"), target_names.index("l_over_d")
sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
X, Y = X[sane], Y[sane]

t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
log_target_names = ckpt["log_target_names"]
Yt = Y.copy()
for name in log_target_names:
    idx = target_names.index(name)
    Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
Yz = (Yt - t_mean) / t_std

idx = 0
anchor_z = torch.tensor(Yz[idx:idx+1], dtype=torch.float32)
reynolds, alpha = X[idx, coeff_dim], X[idx, coeff_dim+1]
re_mean, re_std = ckpt["reynolds_mean"], ckpt["reynolds_std"]
al_mean, al_std = ckpt["alpha_mean"], ckpt["alpha_std"]
aux = torch.tensor([[(np.log(reynolds)-re_mean)/re_std, (alpha-al_mean)/al_std]], dtype=torch.float32)
cond = torch.cat([anchor_z, aux], dim=-1)
k = 5
cond_rep = cond.repeat(k, 1)

torch.manual_seed(0)
x = torch.randn(k, coeff_dim)
with torch.no_grad():
    for t in reversed(range(T)):
        t_batch = torch.full((k,), t, dtype=torch.long)
        eps_pred = model(x, t_batch, cond_rep)
        beta_t, alpha_t, ab_t = schedule["betas"][t], schedule["alphas"][t], schedule["alpha_bars"][t]
        mean = (x - beta_t / (1 - ab_t).sqrt() * eps_pred) / alpha_t.sqrt()
        if t > 0:
            x = mean + beta_t.sqrt() * torch.randn_like(x)
        else:
            x = mean

print(f"final decoded (standardized coeff space) stats: min={x.min().item():.3f} max={x.max().item():.3f} "
      f"mean={x.mean().item():.3f} std={x.std().item():.3f}")
print(f"any NaN: {torch.isnan(x).any().item()}  any Inf: {torch.isinf(x).any().item()}")
coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
params = x.numpy() * coeff_std + coeff_mean
print(f"unstandardized CST params stats: min={params.min():.3f} max={params.max():.3f} "
      f"(real CST upper/lower weights are typically O(0.05-0.3) in magnitude)")

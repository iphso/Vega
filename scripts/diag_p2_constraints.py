"""Which of P2's 5 constraints is actually unreachable for the ALM search,
vs which are fine? Runs the same ALM machinery as generate_candidates.py
for a handful of starts, but reports each constraint's OWN final violation
separately instead of only the all-constraints-satisfied boolean.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from optimize import load_ensemble, ensemble_predict, parse_constraint, violation
from train import load_split, IDX_NFP
from train_vae import VAE, nfp_one_hot

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = json.loads((OUT_DIR / "metadata.json").read_text())
target_stats = meta["target_stats"]

ckpt = torch.load(CKPT_DIR / "vae_coeffs_full_s0.pt", map_location=dev, weights_only=False)
vae = VAE(coeff_dim=90, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
vae.load_state_dict(ckpt["model_state_dict"])
vae.eval()
coeff_mean = torch.as_tensor(ckpt["coeff_mean"], dtype=torch.float32, device=dev)
coeff_std = torch.as_tensor(ckpt["coeff_std"], dtype=torch.float32, device=dev)

models, target_names = load_ensemble(["reg_mlp_big_full_s0", "reg_mlp_big_full_s1", "reg_mlp_big_full_s2"], dev)

constraints_spec = [
    "aspect_ratio<=10.0",
    "abs(edge_rotational_transform_over_n_field_periods)>=0.25",
    "qi<=0.0001",
    "edge_magnetic_mirror_ratio<=0.2",
    "max_elongation<=5.0",
]
constraints = []
for spec in constraints_spec:
    name, op, value, use_abs = parse_constraint(spec)
    constraints.append((target_names.index(name), op, value, target_stats[name]["std"], name, use_abs))

N = 512
torch.manual_seed(0)
z = torch.randn(N, vae.latent_dim, device=dev, requires_grad=True)
nfp_col = nfp_one_hot(torch.full((N,), 3.0, device=dev))
sym_col = torch.ones(N, 1, device=dev)
opt = torch.optim.Adam([z], lr=0.02)

for outer in range(60):
    for inner in range(30):
        opt.zero_grad()
        x = vae.decode(z, nfp_col)
        x_full = torch.cat([x, nfp_col, sym_col], dim=1)
        mean, std = ensemble_predict(models, x_full)
        loss = sum(violation(mean[:, idx], op, val, sd, use_abs) for idx, op, val, sd, _, use_abs in constraints).mean()
        loss.backward()
        opt.step()

with torch.no_grad():
    x = vae.decode(z, nfp_col)
    x_full = torch.cat([x, nfp_col, sym_col], dim=1)
    mean, std = ensemble_predict(models, x_full)
    print(f"\n=== per-constraint final relative violation across {N} starts (<=1% is 'satisfied') ===")
    for idx, op, val, sd, name, use_abs in constraints:
        col = mean[:, idx].abs() if use_abs else mean[:, idx]
        if op == "<=":
            viol = (col - val) / abs(val)
        else:
            viol = (val - col) / abs(val)
        frac_ok = (viol <= 0.01).float().mean().item()
        print(f"  {name:55s} {op} {val:<10g} frac_satisfied={frac_ok:.1%}  "
              f"viol mean={viol.mean().item():+.4f} min={viol.min().item():+.4f} max={viol.max().item():+.4f}  "
              f"(raw value mean={col.mean().item():.6g})")

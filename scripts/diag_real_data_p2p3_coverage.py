"""How close does the REAL training data ever get to P2/P3's constraint region?
Not a search -- just measures the raw dataset (no VAE, no surrogate, no gradients)
to separate "the decoder's manifold doesn't cover this" from "the surrogate can't
find it." If zero real rows come close, that's a data-coverage problem no amount
of surrogate-architecture swapping fixes; if plenty come close, the bottleneck is
squarely in the search/surrogate side.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import load_split, IDX_NFP  # noqa: E402

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
_ckpt = torch.load(CKPT_DIR / "reg_mlp_big_full_s0.pt", map_location="cpu", weights_only=False)
target_names = _ckpt["target_names"]
print("target_names order (from checkpoint, matches Y columns):", target_names)

split_dir = sys.argv[1] if len(sys.argv) > 1 else None
data_dir = (OUT_DIR / "splits" / split_dir) if split_dir else None
print(f"reading split: {split_dir or '(default output/train.npz)'}")
X_train, Y_train = load_split("train", data_dir=data_dir)
nfp3 = X_train[:, IDX_NFP] == 3.0
Y = Y_train[nfp3].numpy() if hasattr(Y_train, "numpy") else np.asarray(Y_train)[nfp3.numpy()]
print(f"n rows (nfp=3): {Y.shape[0]}")


def col(name):
    return Y[:, target_names.index(name)]


def violation_frac(viol):
    return (viol <= 0.0).mean(), viol.min(), np.median(viol)


print("\n=== P2 (SimpleToBuildQIStellarator) per-constraint, REAL data only ===")
p2 = {
    "aspect_ratio<=10.0": (col("aspect_ratio") - 10.0) / 10.0,
    "|edge_iota/nfp|>=0.25": (0.25 - np.abs(col("edge_rotational_transform_over_n_field_periods"))) / 0.25,
    "log10(qi)<=-4.0": (np.log10(np.clip(col("qi"), 1e-300, None)) - (-4.0)) / 4.0,
    "edge_mirror<=0.2": (col("edge_magnetic_mirror_ratio") - 0.2) / 0.2,
    "max_elongation<=5.0": (col("max_elongation") - 5.0) / 5.0,
}
worst_p2 = np.zeros(Y.shape[0])
for name, viol in p2.items():
    frac_ok, vmin, vmed = violation_frac(viol)
    print(f"  {name:28s} frac_satisfied={frac_ok:6.2%}  min_viol={vmin:+.4f}  median_viol={vmed:+.4f}")
    worst_p2 = np.maximum(worst_p2, viol)
n_all_ok_p2 = (worst_p2 <= 0.01).sum()
best_idx_p2 = np.argsort(worst_p2)[:5]
print(f"  ALL 5 constraints simultaneously satisfied (tol 1%): {n_all_ok_p2}/{Y.shape[0]} real rows")
print(f"  best (lowest worst-violation) row(s): worst_viol={worst_p2[best_idx_p2]}")
for i in best_idx_p2[:1]:
    print(f"    closest real row raw values: " + ", ".join(f"{n}={col(n)[i]:.5g}" for n in
          ["aspect_ratio", "edge_rotational_transform_over_n_field_periods", "qi",
           "edge_magnetic_mirror_ratio", "max_elongation"]))

print("\n=== P3 (MHDStableQIStellarator) per-constraint, REAL data only ===")
p3 = {
    "|edge_iota/nfp|>=0.25": (0.25 - np.abs(col("edge_rotational_transform_over_n_field_periods"))) / 0.25,
    "log10(qi)<=-3.5": (np.log10(np.clip(col("qi"), 1e-300, None)) - (-3.5)) / 3.5,
    "edge_mirror<=0.25": (col("edge_magnetic_mirror_ratio") - 0.25) / 0.25,
    "flux_compression<=0.9": (col("flux_compression_in_regions_of_bad_curvature") - 0.9) / 0.9,
    "vacuum_well>=0.0": (0.0 - col("vacuum_well")) / 0.1,
}
worst_p3 = np.zeros(Y.shape[0])
for name, viol in p3.items():
    frac_ok, vmin, vmed = violation_frac(viol)
    print(f"  {name:28s} frac_satisfied={frac_ok:6.2%}  min_viol={vmin:+.4f}  median_viol={vmed:+.4f}")
    worst_p3 = np.maximum(worst_p3, viol)
n_all_ok_p3 = (worst_p3 <= 0.01).sum()
best_idx_p3 = np.argsort(worst_p3)[:5]
print(f"  ALL 5 constraints simultaneously satisfied (tol 1%): {n_all_ok_p3}/{Y.shape[0]} real rows")
print(f"  best (lowest worst-violation) row(s): worst_viol={worst_p3[best_idx_p3]}")
for i in best_idx_p3[:1]:
    print(f"    closest real row raw values: " + ", ".join(f"{n}={col(n)[i]:.5g}" for n in
          ["edge_rotational_transform_over_n_field_periods", "qi",
           "edge_magnetic_mirror_ratio", "flux_compression_in_regions_of_bad_curvature", "vacuum_well"]))

print("\n=== qi distribution overall (log10), all nfp=3 real rows ===")
qi = col("qi")
logqi = np.log10(np.clip(qi, 1e-300, None))
for p in [1, 5, 10, 25, 50]:
    print(f"  p{p:02d} = {np.percentile(logqi, p):.3f}")
print(f"  min = {logqi.min():.3f}  (P2 needs <=-4.0, P3 needs <=-3.5)")

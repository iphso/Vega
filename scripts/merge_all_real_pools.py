"""One-off: merge every grounded-walk run's accumulated real_pool_X/Y.npy
(every real, VMEC++-converged step any fiber in any lane has ever had
accepted, across the whole project -- ~91k rows as of 2026-09-14) onto the
bootstrap_live base split's TRAIN partition, into a single new split. Same
"val/test stay untouched" discipline as merge_real_pool_split.py, just
generalized to every output/grounded_walk_*/real_pool_{X,Y}.npy found via
glob instead of one named pair.

Purpose: give the three target-conditioned generative models (train_cvae.py,
train_diffusion.py, train_gan.py) the best available picture of the
qi-vs-mirror_ratio wall's actual shape -- including thousands of points
right at the wall from every basin's ALM pressure, not just the original
archive dataset -- before asking them to steer toward a target that sits on
the far side of it (low qi AND low mirror_ratio jointly). No filtering by
lane/outcome: every accepted real step is real, oracle-confirmed physics,
whether or not that run ever reached full P3 feasibility.
"""
import glob
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")


def main():
    base_dir = OUT_DIR / "splits" / "bootstrap_live"
    out_dir = OUT_DIR / "splits" / "all_grounded_walk_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(base_dir / "train.npz")
    X_parts, Y_parts = [train["X"]], [train["Y"]]
    n_base = train["X"].shape[0]

    pool_dirs = sorted(glob.glob(str(OUT_DIR / "grounded_walk_*")))
    n_pools = 0
    for d in pool_dirs:
        d = Path(d)
        xf, yf = d / "real_pool_X.npy", d / "real_pool_Y.npy"
        if xf.exists() and yf.exists():
            X_parts.append(np.load(xf))
            Y_parts.append(np.load(yf))
            n_pools += 1
            print(f"  + {d.name}: {X_parts[-1].shape[0]} rows")

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    np.savez(out_dir / "train.npz", X=X, Y=Y)
    print(f"train: {n_base} base -> {X.shape[0]} rows (+{X.shape[0] - n_base} from {n_pools} grounded-walk pools)")

    for name in ("val", "test"):
        d = np.load(base_dir / f"{name}.npz")
        np.savez(out_dir / f"{name}.npz", X=d["X"], Y=d["Y"])
    print("val/test copied unchanged from bootstrap_live")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

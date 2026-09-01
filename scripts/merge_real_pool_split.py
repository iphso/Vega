"""One-off companion to grounded_walk_step.py's retrain trigger: append a
grounded walk's accumulated real_pool_X.npy/real_pool_Y.npy (every real,
VMEC++-converged step any fiber has ever had accepted, across the whole
run so far) onto a base split's TRAIN partition, same "val/test stay
untouched" discipline as build_curriculum_split.py. Separate from that
script because the source here is raw arrays already in X/Y row format
(grounded_walk_step.py builds them directly), not oracle_candidates_master.jsonl
rows that need label-filtering and target-dict-to-array conversion first.
"""
import argparse
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-split", default="bootstrap_live")
    p.add_argument("--out-split", required=True)
    p.add_argument("--pool-x", required=True)
    p.add_argument("--pool-y", required=True)
    args = p.parse_args()

    base_dir = OUT_DIR / "splits" / args.base_split
    out_dir = OUT_DIR / "splits" / args.out_split
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_x, pool_y = np.load(args.pool_x), np.load(args.pool_y)
    train = np.load(base_dir / "train.npz")
    X_train = np.concatenate([train["X"], pool_x], axis=0)
    Y_train = np.concatenate([train["Y"], pool_y], axis=0)
    np.savez(out_dir / "train.npz", X=X_train, Y=Y_train)
    print(f"train: {train['X'].shape[0]} -> {X_train.shape[0]} rows (+{pool_x.shape[0]} real grounded-walk points)")

    for name in ("val", "test"):
        d = np.load(base_dir / f"{name}.npz")
        np.savez(out_dir / f"{name}.npz", X=d["X"], Y=d["Y"])
    print(f"val/test copied unchanged from {args.base_split}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

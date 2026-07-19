"""Builds output/splits/<tag>/ = the cluster split's train.npz + generated
designs (scripts/generate_and_validate.py), val.npz/test.npz copied over
unchanged.

Safety check that matters: the generated designs were sampled toward
under-covered clusters (scripts/generate_and_validate.py) without regard to
which split those clusters ended up in. Naively adding all of them to
training would leak information about the cluster-split's held-out val/test
regions right back in via generated neighbors -- the same failure mode
diagnosed in EXPERIMENT_LOG section 6, reintroduced through the back door.
So: every generated design's nearest real neighbor's cluster is looked up in
cluster_split_membership.json, and only designs anchored in a TRAIN cluster
are added. Anything anchored in a val/test cluster is dropped for this
purpose (reported, not silently discarded).
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-split", default="cluster")
    p.add_argument("--generated-dir", default=str(OUT_DIR / "generated"))
    p.add_argument("--tag", default="cluster_augmented")
    args = p.parse_args()

    base_dir = OUT_DIR / "splits" / args.base_split
    train = np.load(base_dir / "train.npz")
    X_train, Y_train = train["X"], train["Y"]

    X = np.load(OUT_DIR / "X.npy")
    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    feature_names = json.loads((OUT_DIR / "feature_names.json").read_text())
    assign = np.load(OUT_DIR / "cluster_assignments.npy")
    membership = json.loads((OUT_DIR / "cluster_split_membership.json").read_text())

    Xg = np.load(Path(args.generated_dir) / "X.npy")
    Yg = np.load(Path(args.generated_dir) / "Y.npy")

    feat_std = np.array([meta["feature_stats"][n]["std"] for n in feature_names[:90]]).clip(min=1e-6)
    Xn = X[:, :90] / feat_std
    Xgn = Xg[:, :90] / feat_std

    nearest_cluster = np.zeros(Xgn.shape[0], dtype=int)
    chunk = 100
    for start in range(0, Xgn.shape[0], chunk):
        d = np.linalg.norm(Xgn[start:start + chunk, None, :] - Xn[None, :, :], axis=2)
        nearest_cluster[start:start + chunk] = assign[d.argmin(axis=1)]

    keep_mask = np.array([membership.get(str(c)) == "train" for c in nearest_cluster])
    n_train_anchored = keep_mask.sum()
    n_dropped = len(keep_mask) - n_train_anchored
    print(f"generated designs: {len(keep_mask)} total, {n_train_anchored} anchored in a TRAIN cluster "
          f"(kept), {n_dropped} anchored in val/test clusters (dropped -- would leak held-out regions)")

    X_aug = np.concatenate([X_train, Xg[keep_mask].astype(X_train.dtype)])
    Y_aug = np.concatenate([Y_train, Yg[keep_mask].astype(Y_train.dtype)])

    out_dir = OUT_DIR / "splits" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train.npz", X=X_aug, Y=Y_aug)
    for name in ["val", "test"]:
        data = np.load(base_dir / f"{name}.npz")
        np.savez(out_dir / f"{name}.npz", X=data["X"], Y=data["Y"])

    print(f"[{args.tag}] train={len(X_aug)} ({len(X_train)} base + {n_train_anchored} generated)  "
          f"val={len(np.load(out_dir / 'val.npz')['X'])}  test={len(np.load(out_dir / 'test.npz')['X'])}  "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()

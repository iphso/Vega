"""Build alternative train/val/test splits to check whether the default
random 80/10/10 row split overstates generalization (see EXPERIMENT_LOG for
the motivating leakage measurements: 76% of test rows under the random split
share a generation-lineage family_id with a training row, and 36% of those
have a near-exact duplicate in train at std-normalized distance < 0.01).

Three split modes, each written to output/splits/<mode>/{train,val,test}.npz
in the same X/Y format load_split() already expects:

  random  -- reproduces preprocess.py's existing split exactly (seed=42),
             for a same-format baseline to compare the other two against.
  group   -- GroupShuffleSplit-style: whole omnigenous_field_and_targets.id
             families assigned entirely to one split, never divided across
             train/val/test. Fixes the family-sharing leakage specifically.
  cluster -- k-means over the 90 (std-normalized) free coefficients, entire
             clusters assigned to one split. Tests generalization to unseen
             *regions* of coefficient space, not just unseen specific
             configurations -- the harder and more relevant test given the
             surrogate is meant to guide search beyond known configurations.
             Not fixed by the group split: same-family rows are usually
             close together, but many near-duplicates in this dataset are
             NOT same-family (16.5% of no-shared-family test rows still had
             a near-exact duplicate in train), so only a spatial split
             addresses those too.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("/work/output")


def greedy_group_assign(group_sizes, group_order, fractions=(0.8, 0.1, 0.1)):
    """Assigns groups (in group_order) to train/val/test buckets, greedily
    filling each bucket to its target row-count fraction before moving to
    the next -- standard approach for splitting variably-sized groups into
    approximately-proportioned partitions without ever dividing a group."""
    total = sum(group_sizes.values())
    targets = [f * total for f in fractions]
    buckets = [[], [], []]
    counts = [0, 0, 0]
    bucket_idx = 0
    for g in group_order:
        while bucket_idx < 2 and counts[bucket_idx] >= targets[bucket_idx]:
            bucket_idx += 1
        buckets[bucket_idx].append(g)
        counts[bucket_idx] += group_sizes[g]
    return buckets


def kmeans(x, k, n_iters=100, seed=0):
    """Simple Lloyd's-algorithm k-means (no sklearn in this image)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    centroid_idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:k]
    centroids = x[centroid_idx].clone()
    assign = None
    for _ in range(n_iters):
        dists = torch.cdist(x, centroids)
        new_assign = dists.argmin(dim=1)
        if assign is not None and (new_assign == assign).all():
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                centroids[c] = x[mask].mean(dim=0)
    return assign


def save_split(name, X, Y, train_idx, val_idx, test_idx):
    split_dir = OUT_DIR / "splits" / name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / "train.npz", X=X[train_idx], Y=Y[train_idx])
    np.savez(split_dir / "val.npz", X=X[val_idx], Y=Y[val_idx])
    np.savez(split_dir / "test.npz", X=X[test_idx], Y=Y[test_idx])
    print(f"[{name}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"-> {split_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["random", "group", "cluster"],
                    choices=["random", "group", "cluster"])
    p.add_argument("--n-clusters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    X = np.load(OUT_DIR / "X.npy")
    Y = np.load(OUT_DIR / "Y.npy")
    n_samples = X.shape[0]

    if "random" in args.modes:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_samples)
        n_train = int(n_samples * 0.8)
        n_val = int(n_samples * 0.1)
        save_split("random", X, Y, perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:])

    if "group" in args.modes:
        family_ids = json.loads((OUT_DIR / "family_ids.json").read_text())
        family_to_idx = {}
        for i, f in enumerate(family_ids):
            family_to_idx.setdefault(f, []).append(i)
        group_sizes = {f: len(idxs) for f, idxs in family_to_idx.items()}
        families = list(family_to_idx.keys())
        rng = np.random.default_rng(args.seed)
        rng.shuffle(families)
        train_g, val_g, test_g = greedy_group_assign(group_sizes, families)
        train_idx = np.array([i for f in train_g for i in family_to_idx[f]])
        val_idx = np.array([i for f in val_g for i in family_to_idx[f]])
        test_idx = np.array([i for f in test_g for i in family_to_idx[f]])
        save_split("group", X, Y, train_idx, val_idx, test_idx)
        print(f"  (families: {len(train_g)} train / {len(val_g)} val / {len(test_g)} test groups, "
              f"none shared across splits by construction)")

    if "cluster" in args.modes:
        meta = json.loads((OUT_DIR / "metadata.json").read_text())
        feature_names = json.loads((OUT_DIR / "feature_names.json").read_text())
        feat_std = np.array([meta["feature_stats"][n]["std"] for n in feature_names[:90]]).clip(min=1e-6)
        coeffs = torch.tensor(X[:, :90] / feat_std, device=dev, dtype=torch.float32)
        assign = kmeans(coeffs, args.n_clusters, seed=args.seed).cpu().numpy()

        cluster_to_idx = {}
        for i, c in enumerate(assign):
            cluster_to_idx.setdefault(int(c), []).append(i)
        cluster_sizes = {c: len(idxs) for c, idxs in cluster_to_idx.items()}
        clusters = list(cluster_to_idx.keys())
        rng = np.random.default_rng(args.seed)
        rng.shuffle(clusters)
        train_c, val_c, test_c = greedy_group_assign(cluster_sizes, clusters)
        train_idx = np.array([i for c in train_c for i in cluster_to_idx[c]])
        val_idx = np.array([i for c in val_c for i in cluster_to_idx[c]])
        test_idx = np.array([i for c in test_c for i in cluster_to_idx[c]])
        save_split("cluster", X, Y, train_idx, val_idx, test_idx)
        sizes = sorted(cluster_sizes.values())
        print(f"  ({args.n_clusters} clusters, sizes range {sizes[0]}-{sizes[-1]}, median {sizes[len(sizes)//2]}; "
              f"{len(train_c)} train / {len(val_c)} val / {len(test_c)} test clusters, "
              f"entire regions of coefficient space held out for val/test)")

        # Persisted for reuse beyond the split itself -- e.g. identifying
        # under-covered regions of coefficient space (small clusters) to
        # target with generative data augmentation, rather than recomputing
        # this k-means run from scratch elsewhere.
        np.save(OUT_DIR / "cluster_assignments.npy", assign)
        with open(OUT_DIR / "cluster_sizes.json", "w") as f:
            json.dump({str(c): n for c, n in cluster_sizes.items()}, f, indent=2)
        membership = {str(c): "train" for c in train_c}
        membership.update({str(c): "val" for c in val_c})
        membership.update({str(c): "test" for c in test_c})
        with open(OUT_DIR / "cluster_split_membership.json", "w") as f:
            json.dump(membership, f, indent=2)
        print(f"  saved cluster_assignments.npy ({len(assign)} rows), cluster_sizes.json, "
              f"and cluster_split_membership.json")


if __name__ == "__main__":
    main()

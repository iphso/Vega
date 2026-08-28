"""Domain-agnostic cluster split -- generalizes make_splits.py's `cluster`
mode (input/coefficient space -- the one that mattered enormously for
VMEC++, EXPERIMENT_LOG §6: a random split overstates generalization by
2.5-3x versus holding out entire k-means clusters of coefficient space) AND
its `target-cluster` mode (target/metric space -- tests generalization to
unseen *regions of target space*, the relevant holdout for a target-to-
design model, and doubles as the coverage_metric.py's own "which regions of
target space are under-covered" ground truth) to any gym_schema.DomainSpec,
so a second domain gets both without a second hand-written split script.
Reuses make_splits.py's own kmeans/greedy_group_assign/save_split unchanged
-- those were already domain-agnostic; only the VMEC-specific main-block
(hardcoded X[:,:90]/Y[:,:11], metadata.json's precomputed feature_stats)
wasn't.

Difference from make_splits.py's own modes: feature/target std is computed
directly from the loaded X/Y (this domain's own param_dim columns / all
target columns) rather than read from metadata.json's precomputed
feature_stats -- VMEC++ has that file, airfoils don't, and computing it
directly is one line either way, so there's no reason to require a
domain-specific metadata file just for this. Target-space z-scoring
log-transforms `spec` domains' own LOG_TARGET_NAMES convention the same way
train.py/make_splits.py's target-cluster mode already does for VMEC++ --
read from each domain oracle module's own LOG_TARGET_NAMES constant (not
part of DomainSpec itself, since it's a training/eval-space convention, not
an oracle-interface one).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gym_schema import airfoil_spec, vmec_spec
from make_splits import greedy_group_assign, kmeans, save_split

OUT_DIR = Path("/work/output")
SPEC_FACTORIES = {"vmec": vmec_spec, "airfoil": airfoil_spec}
LOG_TARGET_NAMES = {
    "vmec": ["qi", "max_elongation", "flux_compression_in_regions_of_bad_curvature",
             "minimum_normalized_magnetic_gradient_scale_length"],
    "airfoil": ["cd"],
}


def target_log_space(Y, target_names, domain):
    Ye = Y.copy()
    for name in LOG_TARGET_NAMES[domain]:
        if name in target_names:
            idx = target_names.index(name)
            Ye[:, idx] = np.log(np.clip(Ye[:, idx], 1e-12, None))
    return Ye


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(SPEC_FACTORIES))
    p.add_argument("--space", default="input", choices=["input", "target"],
                    help="input: cluster on design-vector coefficients (make_splits.py's `cluster` mode). "
                         "target: cluster on (log-transformed, z-scored) target metrics (`target-cluster` mode).")
    p.add_argument("--n-clusters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", default=None,
                    help="split dir name under output/splits/ (default: '<domain>_cluster' or '<domain>_target_cluster')")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    spec = SPEC_FACTORIES[args.domain]()
    x_path, y_path = spec.dataset_paths(OUT_DIR)
    X, Y = np.load(x_path), np.load(y_path)
    if spec.sanity_filter is not None:
        mask = spec.sanity_filter(Y)
        n_dropped = len(Y) - mask.sum()
        if n_dropped:
            print(f"[{args.domain}] dropping {n_dropped}/{len(Y)} rows failing sanity_filter before clustering")
        X, Y = X[mask], Y[mask]

    if args.space == "input":
        param_dim = spec.param_dim
        raw = X[:, :param_dim]
        feat_std = raw.std(axis=0).clip(min=1e-6)
        points = torch.tensor(raw / feat_std, device=dev, dtype=torch.float32)
        default_name = f"{args.domain}_cluster"
        region_desc = f"{args.domain} coefficient space"
    else:
        Ye = target_log_space(Y, spec.target_names, args.domain)
        t_mean, t_std = Ye.mean(axis=0), Ye.std(axis=0).clip(min=1e-6)
        points = torch.tensor((Ye - t_mean) / t_std, device=dev, dtype=torch.float32)
        default_name = f"{args.domain}_target_cluster"
        region_desc = f"{args.domain} target space"

    assign = kmeans(points, args.n_clusters, seed=args.seed).cpu().numpy()

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

    name = args.out_name or default_name
    save_split(name, X, Y, train_idx, val_idx, test_idx)
    sizes = sorted(cluster_sizes.values())
    print(f"  ({args.n_clusters} clusters, sizes range {sizes[0]}-{sizes[-1]}, median {sizes[len(sizes)//2]}; "
          f"{len(train_c)} train / {len(val_c)} val / {len(test_c)} test clusters, "
          f"entire regions of {region_desc} held out for val/test)")

    split_dir = OUT_DIR / "splits" / name
    np.save(split_dir / "cluster_assignments.npy", assign)
    if args.space == "target":
        points_np = points.cpu().numpy()
        centroids = np.stack([
            points_np[cluster_to_idx[c]].mean(axis=0) if c in cluster_to_idx else np.zeros(points_np.shape[1])
            for c in range(args.n_clusters)
        ])
        np.save(split_dir / "cluster_centroids.npy", centroids)
        np.save(split_dir / "target_z_mean.npy", t_mean)
        np.save(split_dir / "target_z_std.npy", t_std)
    (split_dir / "cluster_sizes.json").write_text(json.dumps({str(c): n for c, n in cluster_sizes.items()}, indent=2))
    print(f"  saved cluster_assignments.npy and cluster_sizes.json under {split_dir}" +
          (" (plus cluster_centroids.npy, target_z_mean.npy, target_z_std.npy)" if args.space == "target" else ""))


if __name__ == "__main__":
    main()

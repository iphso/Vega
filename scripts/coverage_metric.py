"""Success metric for bootstrap_generic.py: "good coverage over broader
sections of target space" (direct user framing), made concrete and
domain-agnostic. Two complementary numbers, not one, because "coverage" has
two genuinely different failure modes a single number would conflate:

  FILL -- does generation reach into regions of target space that were
    already possible but UNDER-represented in the real dataset (gaps
    inside the known range)? Operationalized via the same target-space
    k-means clustering make_splits_generic.py --space target already
    builds (itself a generalization of make_splits.py's VMEC-only
    `target-cluster` mode) -- clusters below the real data's own median
    cluster size are "under-covered" by definition; a good bootstrap pool
    should land in those clusters at a much higher RATE than the real data
    itself does (the natural null: if generation just resampled the real
    distribution, its per-cluster hit rate would equal the real data's own
    per-cluster share, not favor sparse clusters more than chance would).
    Directly generalizes this project's own §9 finding for VMEC++
    ("93.9% of generated designs landed nearest an under-covered cluster,
    vs. ~50% baseline") to any domain and any generator, instead of a
    one-off number computed by hand for a single VAE run.

  REACH -- does generation extend BEYOND the real dataset's existing
    convex range at all, in genuinely new directions, not just fill gaps
    inside it? Operationalized via random unit directions in (z-scored)
    target space (the same sample_unit_directions mechanism steerability_
    generic.py/bootstrap_generic.py already use to EVALUATE/DRIVE
    direction-following) -- project the real data and the generated pool
    onto each direction and compare max projections. A ratio > 1 means the
    generated pool reaches further out along that direction than any real
    design does.

Both numbers report a comparison against a NULL baseline computed the same
way (real data's own per-cluster share for FILL; a same-size random
resample of the real data itself, projected the same way, for REACH) so a
score is legible as "better/worse than doing nothing" rather than a bare,
uncalibrated number.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gym_schema import airfoil_spec, vmec_spec
from make_splits_generic import LOG_TARGET_NAMES, target_log_space

OUT_DIR = Path("/work/output")
SPEC_FACTORIES = {"vmec": vmec_spec, "airfoil": airfoil_spec}


def load_real_target_space(domain, spec):
    x_path, y_path = spec.dataset_paths(OUT_DIR)
    X, Y = np.load(x_path), np.load(y_path)
    if spec.sanity_filter is not None:
        Y = Y[spec.sanity_filter(Y)]
    return target_log_space(Y, spec.target_names, domain)


def load_bootstrap_pool(out_tag):
    pool_dir = OUT_DIR / f"bootstrap_generic_{out_tag}"
    Y = np.load(pool_dir / "Y.npy")
    return Y, pool_dir


def fill_score(domain, target_names, Yz_real, Yz_gen, n_clusters=30, seed=42, undercovered_frac=0.5):
    """Loads (or builds, if missing) the target-space cluster split's
    centroids/assignments, computes each cluster's real share, labels
    clusters below the `undercovered_frac` quantile of cluster size as
    under-covered, then measures what fraction of Yz_gen's nearest-centroid
    assignments land in one of those clusters, against the null of Yz_real's
    own share landing there (by construction, close to undercovered_frac
    itself, weighted by cluster size -- not exactly undercovered_frac since
    clusters vary in size, hence computing it directly rather than assuming)."""
    split_name = f"{domain}_target_cluster"
    split_dir = OUT_DIR / "splits" / split_name
    if not (split_dir / "cluster_centroids.npy").exists():
        raise FileNotFoundError(
            f"{split_dir}/cluster_centroids.npy missing -- run "
            f"`scripts/make_splits_generic.py --domain {domain} --space target --n-clusters {n_clusters} --seed {seed}` first"
        )
    centroids = np.load(split_dir / "cluster_centroids.npy")  # (K, T), already in real data's own z-space
    assign_real = np.load(split_dir / "cluster_assignments.npy")
    cluster_sizes = json.loads((split_dir / "cluster_sizes.json").read_text())
    k = centroids.shape[0]

    sizes_arr = np.array([cluster_sizes.get(str(c), 0) for c in range(k)])
    median_size = np.median(sizes_arr[sizes_arr > 0])
    undercovered = sizes_arr < median_size  # (K,) bool

    n_real = len(assign_real)
    real_undercovered_share = float(undercovered[assign_real].mean())

    dists_gen = np.linalg.norm(Yz_gen[:, None, :] - centroids[None, :, :], axis=2)  # (N_gen, K)
    assign_gen = dists_gen.argmin(axis=1)
    gen_undercovered_rate = float(undercovered[assign_gen].mean())

    clusters_hit_by_gen = len(set(assign_gen.tolist()))

    return {
        "n_clusters": k,
        "median_cluster_size_real": float(median_size),
        "n_undercovered_clusters": int(undercovered.sum()),
        "real_undercovered_share (null baseline)": round(real_undercovered_share, 4),
        "gen_undercovered_rate": round(gen_undercovered_rate, 4),
        "fill_lift (gen / null)": round(gen_undercovered_rate / max(real_undercovered_share, 1e-9), 3),
        "clusters_hit_by_gen": clusters_hit_by_gen,
        "clusters_hit_fraction": round(clusters_hit_by_gen / k, 3),
    }


def reach_score(Yz_real, Yz_gen, n_directions=200, seed=0):
    """Projects both the real dataset and the generated pool onto
    `n_directions` random unit vectors in target-z-space; per direction,
    compares the generated pool's max projection to the real data's own max
    projection (reach_ratio > 1 = genuinely extends past the known convex
    range in that direction) and, as the null, to a same-size bootstrap
    resample of the real data's own max projection (captures how much of
    any apparent "reach" is just generated-pool-size sampling noise, since
    max-of-N grows with N even from a fixed real distribution)."""
    rng = np.random.default_rng(seed)
    dim = Yz_real.shape[1]
    directions = rng.normal(size=(n_directions, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)

    proj_real = Yz_real @ directions.T  # (N_real, D)
    proj_gen = Yz_gen @ directions.T    # (N_gen, D)
    real_max = proj_real.max(axis=0)    # (D,)
    gen_max = proj_gen.max(axis=0)

    # null: same-size resample of real data, same directions, same projections
    n_gen = len(Yz_gen)
    resample_idx = rng.integers(0, len(Yz_real), size=n_gen)
    null_max = proj_real[resample_idx].max(axis=0)

    reach_ratio = gen_max / np.abs(real_max).clip(min=1e-9)
    null_ratio = null_max / np.abs(real_max).clip(min=1e-9)
    frac_directions_exceeded = float((gen_max > real_max).mean())
    frac_directions_exceeded_null = float((null_max > real_max).mean())

    return {
        "n_directions": n_directions,
        "mean_reach_ratio (gen_max/real_max)": round(float(reach_ratio.mean()), 4),
        "mean_reach_ratio_null (same-size real resample)": round(float(null_ratio.mean()), 4),
        "frac_directions_gen_exceeds_real_max": round(frac_directions_exceeded, 4),
        "frac_directions_null_exceeds_real_max": round(frac_directions_exceeded_null, 4),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(SPEC_FACTORIES))
    p.add_argument("--out-tag", required=True, help="bootstrap_generic.py's --out-tag (reads output/bootstrap_generic_<tag>/Y.npy)")
    p.add_argument("--n-directions", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = SPEC_FACTORIES[args.domain]()
    Yz_real = load_real_target_space(args.domain, spec)
    t_mean, t_std = Yz_real.mean(axis=0), Yz_real.std(axis=0).clip(min=1e-6)
    Yz_real = (Yz_real - t_mean) / t_std

    Y_gen, pool_dir = load_bootstrap_pool(args.out_tag)
    Yz_gen = target_log_space(Y_gen, spec.target_names, args.domain)
    Yz_gen = (Yz_gen - t_mean) / t_std  # SAME real-data stats, never the generated pool's own

    print(f"\n=== coverage_metric: domain={args.domain} pool={args.out_tag} "
          f"(n_real={len(Yz_real):,} n_gen={len(Yz_gen):,}) ===\n")

    fill = fill_score(args.domain, spec.target_names, Yz_real, Yz_gen)
    print("FILL (does generation reach into already-known but under-covered regions?):")
    for k, v in fill.items():
        print(f"  {k:45s} {v}")

    reach = reach_score(Yz_real, Yz_gen, n_directions=args.n_directions, seed=args.seed)
    print("\nREACH (does generation extend beyond the real dataset's existing range?):")
    for k, v in reach.items():
        print(f"  {k:45s} {v}")

    out = {"domain": args.domain, "out_tag": args.out_tag, "n_real": len(Yz_real), "n_gen": len(Yz_gen),
           "fill": fill, "reach": reach}
    (pool_dir / "coverage_metric.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {pool_dir / 'coverage_metric.json'}")


if __name__ == "__main__":
    main()

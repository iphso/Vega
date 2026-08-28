"""Does each generative model produce genuinely novel designs, or is it
effectively memorizing/retrieving near-copies of training rows? And does it
actually explore a range of designs per request, or mode-collapse to
(near-)one answer regardless of the latent noise? Both checked directly
against real data -- no oracle calls needed, this is pure parameter-space
geometry, so it's fast and can run standalone.

Two things measured, per (domain, architecture):

1. Novelty: for each of K decoded candidates per anchor, the Euclidean
   distance (in the checkpoint's own standardized coefficient space -- the
   same space training/conditioning happens in) to its single nearest
   neighbor among ALL real training rows. Compared against a reference: the
   same nearest-neighbor-distance computation done among REAL rows
   themselves (leave-one-out) -- "how far apart are real designs from each
   other, typically." If generated candidates sit much closer to some real
   row than real rows sit to each other, that's evidence of memorization,
   not generation. A near-zero-distance fraction is flagged explicitly as a
   literal near-duplicate count.

2. Diversity (mode collapse): for the K candidates decoded at the SAME
   anchor (same conditioning, different latent noise), the mean pairwise
   distance between them, in the same standardized space. Compared against
   the same real-to-real reference distance as a sanity floor -- if a
   model's own within-request diversity is far below the typical spacing
   between real designs, that's mode collapse (all K candidates are
   effectively the same point); comparable or higher suggests it's actually
   using its latent noise to explore, not just changing z as a formality.
"""
import json
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def chunked_min_dist(queries, refs, chunk=2000):
    """Min Euclidean distance from each row in `queries` to any row in
    `refs`, chunked over `refs` so this doesn't need queries x refs held in
    memory at once (158,685 real VMEC rows x 90 dims makes the naive
    cdist a non-starter at any real query-batch size)."""
    n_q = queries.shape[0]
    best = np.full(n_q, np.inf, dtype=np.float64)
    for start in range(0, refs.shape[0], chunk):
        block = refs[start:start + chunk]
        d = np.sqrt(((queries[:, None, :] - block[None, :, :]) ** 2).sum(axis=-1))
        best = np.minimum(best, d.min(axis=1))
    return best


def real_to_real_reference(X_std, n_sample=300, seed=0, chunk=2000):
    """Leave-one-out NN distance for a random sample of real rows against
    the rest of the real set -- the 'how far apart are real designs from
    each other' scale bar everything else gets compared to."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_std), size=min(n_sample, len(X_std)), replace=False)
    dists = np.empty(len(idx), dtype=np.float64)
    for i, qi in enumerate(idx):
        q = X_std[qi:qi + 1]
        best = np.inf
        for start in range(0, X_std.shape[0], chunk):
            block = X_std[start:start + chunk]
            block_idx = np.arange(start, start + block.shape[0])
            mask = block_idx != qi
            if not mask.any():
                continue
            d = np.sqrt(((q - block[mask]) ** 2).sum(axis=-1))
            if d.min() < best:
                best = d.min()
        dists[i] = best
    return dists


def mean_pairwise_dist(points):
    n = points.shape[0]
    if n < 2:
        return float("nan")
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def audit_domain_vmec(k_per_anchor=30, n_anchors=12, seed=0):
    print(f"\n{'=' * 70}\nVMEC++ / stellarators\n{'=' * 70}")
    from eval_cvae_steerability import load_generative_model, eval_space
    from oracle_harness import run_batch_with_timeout  # noqa: F401 (not used, kept for parity/import sanity)
    dev = torch.device("cpu")

    X = np.load(OUT_DIR / "X.npy")
    Y = np.load(OUT_DIR / "Y.npy")
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())

    tags = {"cvae": "cvae_targets_full_s0", "diffusion": "diffusion_targets_full_s0", "gan": "gan_targets_full_s0"}
    results = {}
    ref_dists = None
    for model_type, tag in tags.items():
        sample_fn, tn, n_targets, coeff_mean, coeff_std, t_mean, t_std = load_generative_model(model_type, tag, dev)
        X_std = (X[:, :90] - coeff_mean) / coeff_std
        if ref_dists is None:
            ref_dists = real_to_real_reference(X_std, n_sample=300, seed=seed)

        Ye = eval_space(Y, tn)
        Yz = (Ye - t_mean) / t_std
        from train_vae import nfp_one_hot
        rng = np.random.default_rng(seed)
        anchor_idx = rng.choice(len(X), size=n_anchors, replace=False)

        all_nn = []
        per_anchor_diversity = []
        near_dup_count = 0
        torch.manual_seed(seed)
        for a_i in anchor_idx:
            anchor_z = torch.tensor(Yz[a_i:a_i + 1], dtype=torch.float32)
            nfp_t = torch.tensor([float(X[a_i, 90])], dtype=torch.float32)
            cond = torch.cat([anchor_z, nfp_one_hot(nfp_t)], dim=-1)
            decoded = sample_fn(cond, k_per_anchor)  # already standardized coeff space
            nn_d = chunked_min_dist(decoded, X_std)
            all_nn.append(nn_d)
            per_anchor_diversity.append(mean_pairwise_dist(decoded))
            near_dup_count += int((nn_d < 0.01 * np.median(ref_dists)).sum())
        all_nn = np.concatenate(all_nn)

        results[model_type] = {
            "gen_nn_median": float(np.median(all_nn)), "gen_nn_p10": float(np.percentile(all_nn, 10)),
            "ref_nn_median": float(np.median(ref_dists)),
            "ratio_median": float(np.median(all_nn) / np.median(ref_dists)),
            "near_dup_fraction": near_dup_count / len(all_nn),
            "mean_within_anchor_diversity": float(np.mean(per_anchor_diversity)),
            "diversity_vs_ref_ratio": float(np.mean(per_anchor_diversity) / np.median(ref_dists)),
        }
        r = results[model_type]
        print(f"\n  {model_type}:")
        print(f"    generated->real NN dist:  median={r['gen_nn_median']:.3f}  p10={r['gen_nn_p10']:.3f}   "
              f"(real-to-real reference median={r['ref_nn_median']:.3f})")
        print(f"    ratio (gen/real, ~1 = as far from training data as real designs are from each other): {r['ratio_median']:.2f}")
        print(f"    near-duplicate fraction (NN dist < 1% of real-to-real median): {r['near_dup_fraction']:.1%}")
        print(f"    within-anchor diversity (mean pairwise dist among {k_per_anchor} decodes at same target): "
              f"{r['mean_within_anchor_diversity']:.3f}  (vs. real-to-real median {r['ref_nn_median']:.3f}, "
              f"ratio {r['diversity_vs_ref_ratio']:.2f})")
    return results


def audit_domain_airfoil(k_per_anchor=30, n_anchors=12, seed=0):
    print(f"\n{'=' * 70}\nXFOIL / airfoils\n{'=' * 70}")
    from eval_airfoil_steerability import load_generative_model, eval_space
    dev = torch.device("cpu")

    tags = {"cvae": "airfoil_cvae_s0", "diffusion": "airfoil_diffusion_s0", "gan": "airfoil_gan_s0"}
    results = {}
    ref_dists = None
    for model_type, tag in tags.items():
        (sample_fn, target_names, n_targets, coeff_dim, coeff_mean, coeff_std, t_mean, t_std,
         re_mean, re_std, al_mean, al_std, dataset_tag) = load_generative_model(model_type, tag, dev)

        X = np.load(OUT_DIR / f"{dataset_tag}_X.npy")
        Y = np.load(OUT_DIR / f"{dataset_tag}_Y.npy")
        cd_col, lod_col = target_names.index("cd"), target_names.index("l_over_d")
        sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
        X, Y = X[sane], Y[sane]

        X_std = (X[:, :coeff_dim] - coeff_mean) / coeff_std
        if ref_dists is None:
            ref_dists = real_to_real_reference(X_std, n_sample=300, seed=seed)

        Ye = eval_space(Y, target_names)
        Yz = (Ye - t_mean) / t_std
        rng = np.random.default_rng(seed)
        anchor_idx = rng.choice(len(X), size=n_anchors, replace=False)

        all_nn = []
        per_anchor_diversity = []
        near_dup_count = 0
        torch.manual_seed(seed)
        for a_i in anchor_idx:
            reynolds, alpha = float(X[a_i, coeff_dim]), float(X[a_i, coeff_dim + 1])
            anchor_z = torch.tensor(Yz[a_i:a_i + 1], dtype=torch.float32)
            aux = torch.tensor([[(np.log(reynolds) - re_mean) / re_std, (alpha - al_mean) / al_std]], dtype=torch.float32)
            cond = torch.cat([anchor_z, aux], dim=-1)
            decoded = sample_fn(cond, k_per_anchor)
            nn_d = chunked_min_dist(decoded, X_std)
            all_nn.append(nn_d)
            per_anchor_diversity.append(mean_pairwise_dist(decoded))
            near_dup_count += int((nn_d < 0.01 * np.median(ref_dists)).sum())
        all_nn = np.concatenate(all_nn)

        results[model_type] = {
            "gen_nn_median": float(np.median(all_nn)), "gen_nn_p10": float(np.percentile(all_nn, 10)),
            "ref_nn_median": float(np.median(ref_dists)),
            "ratio_median": float(np.median(all_nn) / np.median(ref_dists)),
            "near_dup_fraction": near_dup_count / len(all_nn),
            "mean_within_anchor_diversity": float(np.mean(per_anchor_diversity)),
            "diversity_vs_ref_ratio": float(np.mean(per_anchor_diversity) / np.median(ref_dists)),
        }
        r = results[model_type]
        print(f"\n  {model_type}:")
        print(f"    generated->real NN dist:  median={r['gen_nn_median']:.3f}  p10={r['gen_nn_p10']:.3f}   "
              f"(real-to-real reference median={r['ref_nn_median']:.3f})")
        print(f"    ratio (gen/real, ~1 = as far from training data as real designs are from each other): {r['ratio_median']:.2f}")
        print(f"    near-duplicate fraction (NN dist < 1% of real-to-real median): {r['near_dup_fraction']:.1%}")
        print(f"    within-anchor diversity (mean pairwise dist among {k_per_anchor} decodes at same target): "
              f"{r['mean_within_anchor_diversity']:.3f}  (vs. real-to-real median {r['ref_nn_median']:.3f}, "
              f"ratio {r['diversity_vs_ref_ratio']:.2f})")
    return results


def main():
    out = {"vmec": audit_domain_vmec(), "airfoil": audit_domain_airfoil()}
    out_path = OUT_DIR / "audit_novelty_diversity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

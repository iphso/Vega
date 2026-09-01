"""Directed dataset augmentation -- densifies the LOW-COMPLIANCE region of
shape-space instead of spending more samples on more uniform-random junk.
Direct follow-up: "let's generate more data - explore in a directed way -
yeah? See if we can get to a better spot."

Why directed, not just "more random": the first dataset (3,000 random DCT
shapes/layout, §101) is dominated by mediocre-to-bad shapes -- random DCT
fields rarely land anywhere near the real optimum, so the scoring surrogate
and GAN had very few LOW-compliance examples to learn the shape of that
region from (§102's steering results, 12.7-26.4x worse than SIMP, are
consistent with this: the model has a poor picture of what "good" looks
like). This script instead perturbs already-known GOOD anchors -- the SIMP
reference optimum, its own trajectory snapshots (already real, already in
the dataset), and any GAN-generated/steered candidates already on disk --
with varying noise levels, volume-rematched via the same bisection trick
random_dct_density uses. Every perturbed shape is still scored through the
REAL oracle, same as every other row in this dataset; "directed" describes
WHERE the samples are drawn from, not any relaxation of that discipline.

Appends to (does not replace) the existing bracket_dataset_<layout>.npz.
"""
import argparse
import glob
import json

import numpy as np

from bracket_generate_dataset_warp import LAYOUTS, RHO_MIN, build_layout_context, score_shape


def perturb_density(anchor_rho, rng, sigma, target_volfrac):
    """Gaussian noise around a real anchor shape, volume-rematched to
    target_volfrac via bisection on an additive offset (same idea as
    random_dct_density's threshold bisection, applied post-noise instead of
    to a synthesized field)."""
    noisy = anchor_rho + rng.normal(scale=sigma, size=anchor_rho.shape)
    lo_t, hi_t = -2.0, 2.0
    for _ in range(40):
        mid = 0.5 * (lo_t + hi_t)
        rho_try = np.clip(noisy + mid, RHO_MIN, 1.0)
        if rho_try.mean() > target_volfrac:
            hi_t = mid
        else:
            lo_t = mid
    offset = 0.5 * (lo_t + hi_t)
    return np.clip(noisy + offset, RHO_MIN, 1.0).astype(np.float32)


def gather_anchors(layout_name, existing):
    """Real, already-known-good shapes to perturb around: the SIMP
    reference optimum + its own trajectory snapshots (already in the
    dataset, source=='simp_trajectory'), plus any GAN-generated/steered
    candidates already exported for this layout (§102) -- those are worse
    than SIMP but still real oracle-scored shapes, useful anchors for
    filling in the space BETWEEN "random junk" and "true optimum"."""
    anchors = []
    traj_mask = existing["source"] == "simp_trajectory"
    for rho in existing["rho"][traj_mask]:
        anchors.append(rho)
    for path in sorted(glob.glob(f"/work/output/bracket_gan_{layout_name}_*_rho.npy")):
        anchors.append(np.load(path))
    return anchors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS.keys()), choices=list(LAYOUTS.keys()))
    ap.add_argument("--n-perturb", type=int, default=3000, help="perturbation samples per layout")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    index = json.load(open("/work/output/bracket_dataset_index.json"))
    index_by_layout = {e["layout"]: e for e in index}

    for layout_name in args.layouts:
        params = LAYOUTS[layout_name]
        target_volfrac = params["volfrac"]
        print(f"\n=== layout: {layout_name} ===  target_volfrac={target_volfrac:.3f}")

        existing = dict(np.load(f"/work/output/bracket_dataset_{layout_name}.npz"))
        anchors = gather_anchors(layout_name, existing)
        print(f"  {len(anchors)} real anchors to perturb around "
              f"({(existing['source'] == 'simp_trajectory').sum()} trajectory + "
              f"{len(anchors) - (existing['source'] == 'simp_trajectory').sum()} GAN-generated)")

        ctx = build_layout_context(params)
        rng = np.random.default_rng(args.seed + hash(layout_name) % 1000)

        new_rho, new_compliance = [], []
        for i in range(args.n_perturb):
            anchor = anchors[rng.integers(len(anchors))]
            sigma = float(rng.uniform(0.03, 0.35))
            # Mostly perturb AT the layout's real volume budget (that's the
            # region the volume-tolerance-gated steering comparison actually
            # lives in), with a wider spread on a minority of samples so the
            # surrogate still sees some off-budget shapes too.
            vf = float(rng.normal(target_volfrac, 0.03)) if rng.random() < 0.7 else float(rng.uniform(0.1, 0.5))
            vf = float(np.clip(vf, 0.05, 0.6))
            rho_np = perturb_density(anchor, rng, sigma, vf)
            compliance = score_shape(ctx, rho_np)
            new_rho.append(rho_np)
            new_compliance.append(compliance)
            if i % 200 == 0 or i == args.n_perturb - 1:
                print(f"  perturb {i:4d}/{args.n_perturb - 1}: sigma={sigma:.3f}  volfrac={rho_np.mean():.3f}  "
                      f"compliance={compliance:.4f}")

        new_rho = np.stack(new_rho).astype(np.float32)
        new_compliance = np.array(new_compliance, dtype=np.float32)
        new_source = np.array(["perturbed_anchor"] * len(new_rho))
        new_iter = np.full(len(new_rho), -1, dtype=np.int32)

        rho_all = np.concatenate([existing["rho"], new_rho])
        compliance_all = np.concatenate([existing["compliance"], new_compliance])
        source_all = np.concatenate([existing["source"], new_source])
        iter_all = np.concatenate([existing["iteration"], new_iter])

        out_path = f"/work/output/bracket_dataset_{layout_name}.npz"
        np.savez_compressed(out_path, rho=rho_all, compliance=compliance_all, source=source_all, iteration=iter_all)
        print(f"  saved {out_path}: {len(rho_all)} rows total (+{len(new_rho)} directed), "
              f"compliance range [{compliance_all.min():.4f}, {compliance_all.max():.4f}]")

        if layout_name in index_by_layout:
            e = index_by_layout[layout_name]
            e["n_rows"] = int(len(rho_all))
            e["compliance_min"] = float(compliance_all.min())
            e["compliance_max"] = float(compliance_all.max())
            e["n_perturbed_anchor"] = int(len(new_rho))

    with open("/work/output/bracket_dataset_index.json", "w") as f:
        json.dump(list(index_by_layout.values()), f, indent=2)
    print("\nsaved output/bracket_dataset_index.json")


if __name__ == "__main__":
    main()

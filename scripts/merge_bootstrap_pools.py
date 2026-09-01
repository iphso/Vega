"""Merges the real dataset with one or more oracle-validated bootstrap-
generated pools (bootstrap_generic.py's own output) into a single augmented
dataset, for retraining generative models on an expanded pool -- the
"train and push" cycle: bootstrap -> merge -> retrain -> bootstrap again
with the improved generator.

Every row in the output, real or generated, passed the exact same oracle
bar (ok=True + the domain's own sanity_filter, re-applied here rather than
trusted from accept time, in case a pool predates a sanity_filter change)
-- this is NOT training on unverified synthetic data, it's training on an
expanded pool of real, physically-validated designs, just discovered by a
generator instead of by nature/the original dataset's own collection
process. Deliberately does NOT overwrite the original X.npy/Y.npy (or
domain-prefixed equivalents) -- saves under a new tag instead, so the
original real-only dataset stays available for comparison and nothing
already-trusted gets silently mutated.

Usage:
    python3 merge_bootstrap_pools.py --domain vmec \\
        --pool-tags aicr_bootstrap_gan_s0 aicr_bootstrap_cvae_s0 aicr_bootstrap_diffusion_s0 \\
        --out-tag cycle1
    # -> output/X_aug_cycle1.npy, output/Y_aug_cycle1.npy
    # (train_cvae.py/train_diffusion.py/train_gan.py: --source augmented --aug-tag cycle1)

    python3 merge_bootstrap_pools.py --domain airfoil \\
        --pool-tags aicr_airfoil_gan_s0 aicr_airfoil_cvae_s0 aicr_airfoil_diffusion_s0 \\
        --out-tag airfoil_cycle1
    # -> output/airfoil_cycle1_X.npy, output/airfoil_cycle1_Y.npy
    # (train_airfoil_*.py: --dataset-tag airfoil_cycle1)
"""
import argparse
from pathlib import Path

import numpy as np

from gym_schema import airfoil_spec, torax_spec, vmec_spec

SPEC_FACTORIES = {"vmec": vmec_spec, "airfoil": airfoil_spec, "torax": torax_spec}
OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(SPEC_FACTORIES))
    p.add_argument("--pool-tags", nargs="+", required=True,
                    help="bootstrap_generic.py --out-tag value(s) to merge in (reads "
                         "output/bootstrap_generic_<tag>/{X,Y}.npy for each)")
    p.add_argument("--out-tag", required=True,
                    help="vmec: writes X_aug_<tag>.npy/Y_aug_<tag>.npy. airfoil/torax: writes "
                         "<tag>_X.npy/<tag>_Y.npy (pass <tag> as --dataset-tag to those domains' "
                         "train_*.py scripts directly, no extra flag needed there)")
    args = p.parse_args()

    spec = SPEC_FACTORIES[args.domain]()
    x_path, y_path = spec.dataset_paths(OUT_DIR)
    X_real, Y_real = np.load(x_path), np.load(y_path)
    if spec.sanity_filter is not None:
        mask = spec.sanity_filter(Y_real)
        n_before = len(X_real)
        X_real, Y_real = X_real[mask], Y_real[mask]
        if len(X_real) != n_before:
            print(f"[{args.domain}] real: dropped {n_before - len(X_real)} rows failing sanity_filter")
    print(f"[{args.domain}] real: {len(X_real):,} rows")

    Xs, Ys = [X_real], [Y_real]
    for tag in args.pool_tags:
        pool_dir = OUT_DIR / f"bootstrap_generic_{tag}"
        Xp, Yp = np.load(pool_dir / "X.npy"), np.load(pool_dir / "Y.npy")
        if spec.sanity_filter is not None:
            mask = spec.sanity_filter(Yp)
            n_before = len(Xp)
            Xp, Yp = Xp[mask], Yp[mask]
            if len(Xp) != n_before:
                print(f"  WARNING: {tag} had {n_before - len(Xp)} rows fail sanity_filter on "
                      f"re-check (should already have passed at accept time -- investigate if nonzero)")
        print(f"[{args.domain}] +{tag}: {len(Xp):,} rows")
        Xs.append(Xp.astype(np.float32)); Ys.append(Yp.astype(np.float32))

    X_aug = np.concatenate(Xs).astype(np.float32)
    Y_aug = np.concatenate(Ys).astype(np.float32)
    print(f"[{args.domain}] augmented total: {len(X_aug):,} rows "
          f"({len(X_real):,} real + {len(X_aug) - len(X_real):,} bootstrap-generated)")

    if args.domain == "vmec":
        out_x, out_y = OUT_DIR / f"X_aug_{args.out_tag}.npy", OUT_DIR / f"Y_aug_{args.out_tag}.npy"
    else:
        out_x, out_y = OUT_DIR / f"{args.out_tag}_X.npy", OUT_DIR / f"{args.out_tag}_Y.npy"
        # train_airfoil_*.py/train_torax_*.py's --dataset-tag convention reads
        # <tag>_target_names.json too, not just <tag>_X/Y.npy -- caught the
        # hard way (a real job failure) the first time this ran without it.
        # Target names don't change between the real dataset and any of its
        # bootstrap pools (same oracle, same TARGET_NAMES), so this is a
        # straight copy, not a re-derivation.
        tn_path = OUT_DIR / f"{args.out_tag}_target_names.json"
        tn_path.write_text((OUT_DIR / f"{args.domain}_target_names.json").read_text())
        print(f"saved {tn_path}")
    np.save(out_x, X_aug)
    np.save(out_y, Y_aug)
    print(f"saved {out_x}\nsaved {out_y}")


if __name__ == "__main__":
    main()

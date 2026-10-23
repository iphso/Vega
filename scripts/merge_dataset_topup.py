"""Generic X/Y dataset merge: concatenates a base dataset with a top-up
generation run into a NEW tag, never touching the base files -- so a
mistake here can't clobber the real production dataset (see the
bracket_generate_dataset_warp.py near-miss in the 2026-09-01 vacation
compute setup: that script overwrites its canonical output in place with
no top-up mode, which is exactly the failure this script is written to
avoid for every other domain).

Usage:
  merge_dataset_topup.py --base-tag airfoil --topup-tag airfoil_vacation_topup --out-tag airfoil_vacation_merged
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-tag", required=True)
    p.add_argument("--topup-tag", required=True)
    p.add_argument("--out-tag", required=True)
    args = p.parse_args()

    base_x, base_y = OUT_DIR / f"{args.base_tag}_X.npy", OUT_DIR / f"{args.base_tag}_Y.npy"
    top_x, top_y = OUT_DIR / f"{args.topup_tag}_X.npy", OUT_DIR / f"{args.topup_tag}_Y.npy"
    out_x, out_y = OUT_DIR / f"{args.out_tag}_X.npy", OUT_DIR / f"{args.out_tag}_Y.npy"

    for path in (base_x, base_y, top_x, top_y):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
    if out_x.exists() or out_y.exists():
        raise FileExistsError(f"{out_x} or {out_y} already exists -- refusing to overwrite, pick a new --out-tag")

    X = np.concatenate([np.load(base_x), np.load(top_x)], axis=0)
    Y = np.concatenate([np.load(base_y), np.load(top_y)], axis=0)
    if X.shape[1:] != np.load(base_x).shape[1:]:
        raise ValueError("column mismatch between base and topup X -- refusing to merge")

    np.save(out_x, X)
    np.save(out_y, Y)

    for suffix in ("target_names.json", "feature_names.json"):
        src = OUT_DIR / f"{args.base_tag}_{suffix}"
        if src.exists():
            shutil.copy(src, OUT_DIR / f"{args.out_tag}_{suffix}")

    print(f"[merge_dataset_topup] {args.base_tag} ({np.load(base_x).shape[0]} rows) + "
          f"{args.topup_tag} ({np.load(top_x).shape[0]} rows) -> {args.out_tag} ({X.shape[0]} rows)")


if __name__ == "__main__":
    main()

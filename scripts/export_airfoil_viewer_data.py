"""Exports airfoil design data for the Vega viewer, matching the stellarator
viewer's own data contract (`viewer/public/data/{meta.json,X.bin,Y.bin}`) so
the browser-side design-browser code (UMAP embedding, per-target histogram
filters, candidate list -- see `viewer/src/pages/index.astro`) works
unmodified against a second domain, per direct user request ("does Vega's
stellarator viewer generalize"). `feature_layout` is the part that makes
this format domain-agnostic: a name -> offset/shape map, rather than a
hardcoded assumption about what the columns mean.

Two sources, deliberately -- mirrors `vega_testing`'s role for VMEC++
(a small curated set, not a dump of the full training data) while adding
something the stellarator viewer's data never had: a REAL, not
synthetic-bootstrap-only, comparison set.

  - `bootstrap`: generate_airfoil_dataset.py's own output (§22), same
    sanity filter (cd>=1e-6, |l_over_d|<=300) every training/eval script
    in this project already applies.
  - `real_naca`: the 3 real NACA 4-digit airfoils x 11 angles of attack
    (§36's `real_reference_airfoils.py` output) -- genuinely real,
    non-self-generated designs with known aerodynamic behavior, not
    another synthetic draw.

Chunked, not just subsampled (EXPERIMENT_LOG §61): direct user request --
filtering down to a narrow slice of the distribution should be able to
"grab more to fill it in" instead of being stuck with whatever fraction of
one fixed random sample happened to land there. The bootstrap pool (50K+
rows) is shuffled ONCE in full (not `rng.choice` of a subset that discards
the rest) and sliced into `--chunk-size`-row chunks; chunk 0 (+ the small,
always-complete `real_naca` set, included here only) is written as
today's plain `X.bin`/`Y.bin`, chunks 1..K-1 as new sibling bootstrap-only
files the browser only fetches if a filter needs more rows than are
currently loaded. `target_range` (real per-target min/max over the FULL
bootstrap+real pool) ships in meta.json too, so histogram axes/percentiles
are correct from the first paint regardless of how many chunks load.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data/airfoil")

N_CST = 8
FEATURE_LAYOUT = {
    "cst_upper": {"offset": 0, "shape": [N_CST]},
    "cst_lower": {"offset": N_CST, "shape": [N_CST]},
    "reynolds": {"offset": 2 * N_CST},
    "alpha": {"offset": 2 * N_CST + 1},
}
TARGET_NAMES = ["cl", "cd", "cm", "l_over_d"]


def load_bootstrap_shuffled(seed):
    """The FULL sane-filtered bootstrap pool, shuffled once -- callers slice
    off whatever prefix they need (chunk 0, then more chunks on demand),
    rather than discarding everything past a fixed sample size."""
    X = np.load(OUT_DIR / "airfoil_X.npy")
    Y = np.load(OUT_DIR / "airfoil_Y.npy")
    sane = (Y[:, TARGET_NAMES.index("cd")] >= 1e-6) & (np.abs(Y[:, TARGET_NAMES.index("l_over_d")]) <= 300)
    X, Y = X[sane], Y[sane]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), Y[perm].astype(np.float32)


def load_real_naca():
    refs = json.loads((OUT_DIR / "airfoil_real_references.json").read_text())
    reynolds = refs["reynolds"]
    rows_x, rows_y = [], []
    for code, info in refs["airfoils"].items():
        params = np.array(info["params"], dtype=np.float32)
        polar = refs["polars"][code]
        for alpha_str, result in polar.items():
            if not result["ok"]:
                continue
            p = result["payload"]
            rows_x.append(np.concatenate([params, [reynolds, float(alpha_str)]]).astype(np.float32))
            rows_y.append(np.array([p[name] for name in TARGET_NAMES], dtype=np.float32))
    return np.stack(rows_x), np.stack(rows_y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int, default=2500, help="rows per chunk; also chunk 0's bootstrap share, "
                                                                   "matching this script's old --n-bootstrap default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_boot, Y_boot = load_bootstrap_shuffled(args.seed)
    X_real, Y_real = load_real_naca()
    n_boot_total = len(X_boot)
    chunk_size = args.chunk_size

    X_all = np.concatenate([X_boot, X_real])
    Y_all = np.concatenate([Y_boot, Y_real])
    target_range = [[float(Y_all[:, t].min()), float(Y_all[:, t].max())] for t in range(Y_all.shape[1])]

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunk0_boot = min(chunk_size, n_boot_total)
    # real_naca FIRST, bootstrap LAST -- so a client-side top-up (appending
    # more bootstrap rows to the end of the array) never disturbs
    # real_naca's own contiguous [start, start+count) range.
    X0 = np.concatenate([X_real, X_boot[:chunk0_boot]])
    Y0 = np.concatenate([Y_real, Y_boot[:chunk0_boot]])
    X0.astype(np.float32).tofile(VIEWER_DATA_DIR / "X.bin")
    Y0.astype(np.float32).tofile(VIEWER_DATA_DIR / "Y.bin")

    n_chunks_total = max(1, -(-n_boot_total // chunk_size))  # ceil div, over the bootstrap pool only
    for c in range(1, n_chunks_total):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_boot_total)
        X_boot[lo:hi].astype(np.float32).tofile(VIEWER_DATA_DIR / f"X.chunk{c}.bin")
        Y_boot[lo:hi].astype(np.float32).tofile(VIEWER_DATA_DIR / f"Y.chunk{c}.bin")

    meta = {
        "n": int(len(X0)),
        "n_total": int(n_boot_total + len(X_real)),
        "n_chunks": int(n_chunks_total - 1),  # extra chunks beyond the initial load, bootstrap-only
        "bootstrap_target": int(chunk_size),
        "x_cols": int(X_all.shape[1]),
        "y_cols": int(Y_all.shape[1]),
        "target_names": TARGET_NAMES,
        "target_range": target_range,
        "source_legend": {"0": "real_naca", "1": "bootstrap"},
        "source_counts": {"bootstrap": int(chunk0_boot), "real_naca": int(len(X_real))},
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote chunk 0 ({chunk0_boot} bootstrap + {len(X_real)} real_naca) + {n_chunks_total - 1} more "
          f"bootstrap chunk(s) ({n_boot_total} bootstrap rows total) -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

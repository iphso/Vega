"""Exports mug design data for the Vega viewer -- the mug counterpart to
export_airfoil_viewer_data.py/export_torax_viewer_data.py, same
`meta.json`/`X.bin`/`Y.bin` contract the stellarator viewer's
design-browser code already reads domain-agnostically.

Single source (`bootstrap`, drawn from generate_mug_dataset.py's own v6
output) -- unlike airfoil/TORAX, there's no real-reference-device analogue
yet for this domain (no named real mugs measured the way §36 did NACA
airfoils / ITER-SPARC-JET tokamaks), flagged as an open item rather than
fabricated.

Chunked, not just subsampled (EXPERIMENT_LOG §61): direct user request --
filtering down to a narrow slice of the distribution should be able to
"grab more to fill it in" instead of being stuck with whatever fraction of
one fixed random sample happened to land there. The ENTIRE eligible pool
is shuffled once (a fixed seed, not `rng.choice` of a subset that discards
the rest) and sliced into `--chunk-size`-row chunks; chunk 0 is written as
today's plain `X.bin`/`Y.bin` (so an unfiltered/mildly-filtered page load
is byte-for-byte unchanged), chunks 1..K-1 as new sibling files the
browser only fetches if a filter actually needs more rows than are
currently loaded. `target_range` (real per-target min/max over the FULL
pool, not just chunk 0) ships in meta.json too, so histogram axes/
percentiles are correct and stable from the first paint regardless of how
many chunks end up loading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data/mug")

# §59: 9-dim single-wall/lidless parameterization (no more t_gap_mm/
# insulation_material_idx/lid_coverage_frac/t_lid_mm/lid_material_idx),
# 5 targets (liquid_volume_L promoted to a real, checked one).
PARAM_NAMES = [
    "r_base_mm", "r_mid_mm", "r_rim_mm",
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
]
TARGET_NAMES = ["temp_at_2h_C", "mass_kg", "touch_temp_60s_C", "handle_temp_60s_C", "liquid_volume_L"]
FEATURE_LAYOUT = {name: {"offset": i} for i, name in enumerate(PARAM_NAMES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int, default=2500, help="rows per chunk; also chunk 0's size, "
                                                                   "matching this script's old --n-bootstrap default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(OUT_DIR / "mug_X.npy")
    Y = np.load(OUT_DIR / "mug_Y.npy")
    n_total = len(X)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)  # shuffle the WHOLE pool once, not a discard-the-rest sample
    X, Y = X[perm].astype(np.float32), Y[perm].astype(np.float32)

    target_range = [[float(Y[:, t].min()), float(Y[:, t].max())] for t in range(Y.shape[1])]

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_size
    n_chunks_total = max(1, -(-n_total // chunk_size))  # ceil div
    X[:chunk_size].tofile(VIEWER_DATA_DIR / "X.bin")
    Y[:chunk_size].tofile(VIEWER_DATA_DIR / "Y.bin")
    for c in range(1, n_chunks_total):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_total)
        X[lo:hi].tofile(VIEWER_DATA_DIR / f"X.chunk{c}.bin")
        Y[lo:hi].tofile(VIEWER_DATA_DIR / f"Y.chunk{c}.bin")

    meta = {
        "n": int(min(chunk_size, n_total)),
        "n_total": int(n_total),
        "n_chunks": int(n_chunks_total - 1),  # extra chunks beyond the initial load
        "bootstrap_target": int(chunk_size),
        "x_cols": int(X.shape[1]),
        "y_cols": int(Y.shape[1]),
        "target_names": TARGET_NAMES,
        "target_range": target_range,
        "source_legend": {"0": "bootstrap"},
        "source_counts": {"bootstrap": int(min(chunk_size, n_total))},
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote chunk 0 ({min(chunk_size, n_total)} rows) + {n_chunks_total - 1} more chunk(s) "
          f"({n_total} rows total, bootstrap only -- no real-reference mugs yet) -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

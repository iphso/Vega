"""Exports mug domain v2 (FFD body + curved handle + coating + joint,
§70-91) design data for the Vega viewer -- the v2 counterpart to
export_mug_viewer_data.py, same `meta.json`/`X.bin`/`Y.bin` contract the
existing viewer pages already read domain-agnostically, but writing to
`/data/mug_v2/` (a SEPARATE directory) rather than `/data/mug/` since v2's
88-dim FFD/handle/joint parameterization is a structurally different
geometry model from v1's 9-dim revolve+straight-handle one -- they are
NOT interchangeable, and `mug.astro`'s own client-side geometry code has
no idea what an FFD lattice or a Bezier-arc handle is (confirmed directly
before building this, not assumed -- see EXPERIMENT_LOG's own entry on
why "just show it in the existing visualizer" wasn't actually possible
without this). `mug-v2.astro` is the real, dedicated page this feeds.

Reads whichever `{tag}_X.npy`/`{tag}_Y.npy` pair is passed via --out-tag
(defaults to `mug_v2_full`, the production run) -- NOT chunked like
export_mug_viewer_data.py, since v2's real dataset is still small (tens to
low thousands of rows for the foreseeable future, not the 20k+ v1's
chunking was built for); revisit chunking if/when that changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data/mug_v2")

# Kept as a literal, ordered list (not imported from mug_openfoam_oracle_v2)
# for the same reason export_mug_viewer_data.py keeps its own PARAM_NAMES
# literal: this needs to match whatever's actually IN the .npy files on
# disk at export time, which is a real, frozen fact about already-generated
# data -- not something that should silently drift if the live oracle
# module's own PARAM_NAMES changes later. Confirmed byte-for-byte against
# mug_openfoam_oracle_v2.PARAM_NAMES at the time of writing (§91).
_N_DISP = 2 * 2 * 3 * 3  # N_LATTICE=(2,2,3), 3 components each = 36
PARAM_NAMES = (
    ["r_inner_mm", "r_outer_mm", "height_mm"]
    + [f"disp_inner_mm_{i}" for i in range(_N_DISP)]
    + [f"disp_outer_mm_{i}" for i in range(_N_DISP)]
    + ["struct_material_idx", "coating_material_idx", "coating_thickness_mm"]
    + ["handle_theta1", "handle_z1_frac", "handle_radial_offset1_mm",
       "handle_theta2", "handle_z2_frac", "handle_radial_offset2_mm",
       "handle_bulge_mm", "handle_radius_mm", "handle_material_idx",
       "joint_material_idx"]
)
TARGET_NAMES = ["touch_temp_60s_C", "temp_at_2h_C", "handle_temp_60s_C", "mass_kg", "liquid_volume_L"]
FEATURE_LAYOUT = {name: {"offset": i} for i, name in enumerate(PARAM_NAMES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-tag", default="mug_v2_full", help="which generate_mug_dataset_v2.py --out-tag to export")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(OUT_DIR / f"{args.out_tag}_X.npy")
    Y = np.load(OUT_DIR / f"{args.out_tag}_Y.npy")
    n_total = len(X)
    if X.shape[1] != len(PARAM_NAMES):
        raise SystemExit(f"{args.out_tag}_X.npy has {X.shape[1]} columns, expected {len(PARAM_NAMES)} -- "
                          f"this script's PARAM_NAMES is stale relative to whatever generated this file, fix before exporting")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    X, Y = X[perm].astype(np.float32), Y[perm].astype(np.float32)

    target_range = [[float(Y[:, t].min()), float(Y[:, t].max())] for t in range(Y.shape[1])] if n_total else \
        [[0.0, 1.0]] * len(TARGET_NAMES)

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X.tofile(VIEWER_DATA_DIR / "X.bin")
    Y.tofile(VIEWER_DATA_DIR / "Y.bin")

    meta = {
        "n": int(n_total),
        "n_total": int(n_total),
        "out_tag": args.out_tag,
        "x_cols": int(X.shape[1]),
        "y_cols": int(Y.shape[1]),
        "target_names": TARGET_NAMES,
        "target_range": target_range,
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_total} rows -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

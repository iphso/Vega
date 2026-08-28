"""Exports mug design data for the Vega viewer -- the mug counterpart to
export_airfoil_viewer_data.py/export_torax_viewer_data.py, same
`meta.json`/`X.bin`/`Y.bin` contract the stellarator viewer's
design-browser code already reads domain-agnostically.

Single source (`bootstrap`, a random subsample of generate_mug_dataset.py's
own v3 output) -- unlike airfoil/TORAX, there's no real-reference-device
analogue yet for this domain (no named real mugs measured the way §36 did
NACA airfoils / ITER-SPARC-JET tokamaks), flagged as an open item rather
than fabricated.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data/mug")

PARAM_NAMES = [
    "r_base_mm", "r_mid_mm", "r_rim_mm",
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "t_gap_mm", "insulation_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
    "lid_coverage_frac", "t_lid_mm", "lid_material_idx",
]
TARGET_NAMES = ["temp_at_2h_C", "mass_kg", "touch_temp_60s_C", "handle_temp_60s_C"]
FEATURE_LAYOUT = {name: {"offset": i} for i, name in enumerate(PARAM_NAMES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bootstrap", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(OUT_DIR / "mug_X.npy")
    Y = np.load(OUT_DIR / "mug_Y.npy")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.n_bootstrap, len(X)), replace=False)
    X, Y = X[idx].astype(np.float32), Y[idx].astype(np.float32)

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X.tofile(VIEWER_DATA_DIR / "X.bin")
    Y.tofile(VIEWER_DATA_DIR / "Y.bin")

    meta = {
        "n": int(len(X)),
        "x_cols": int(X.shape[1]),
        "y_cols": int(Y.shape[1]),
        "target_names": TARGET_NAMES,
        "source_legend": {"0": "bootstrap"},
        "source_counts": {"bootstrap": int(len(X))},
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(X)} rows (bootstrap only -- no real-reference mugs yet) -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

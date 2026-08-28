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

  - `bootstrap`: a random subsample of generate_airfoil_dataset.py's own
    output (§22), same sanity filter (cd>=1e-6, |l_over_d|<=300) every
    training/eval script in this project already applies. Subsampled (not
    the full 50K+ rows) purely for browser payload size -- X.bin/Y.bin are
    loaded as flat Float32Arrays client-side, not paginated.
  - `real_naca`: the 3 real NACA 4-digit airfoils x 11 angles of attack
    (§36's `real_reference_airfoils.py` output) -- genuinely real,
    non-self-generated designs with known aerodynamic behavior, not
    another synthetic draw.
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


def load_bootstrap(n_sample, seed):
    X = np.load(OUT_DIR / "airfoil_X.npy")
    Y = np.load(OUT_DIR / "airfoil_Y.npy")
    sane = (Y[:, TARGET_NAMES.index("cd")] >= 1e-6) & (np.abs(Y[:, TARGET_NAMES.index("l_over_d")]) <= 300)
    X, Y = X[sane], Y[sane]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    return X[idx].astype(np.float32), Y[idx].astype(np.float32)


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
    ap.add_argument("--n-bootstrap", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_boot, Y_boot = load_bootstrap(args.n_bootstrap, args.seed)
    X_real, Y_real = load_real_naca()

    X = np.concatenate([X_boot, X_real])
    Y = np.concatenate([Y_boot, Y_real])

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X.astype(np.float32).tofile(VIEWER_DATA_DIR / "X.bin")
    Y.astype(np.float32).tofile(VIEWER_DATA_DIR / "Y.bin")

    meta = {
        "n": int(len(X)),
        "x_cols": int(X.shape[1]),
        "y_cols": int(Y.shape[1]),
        "target_names": TARGET_NAMES,
        "source_legend": {"0": "bootstrap", "1": "real_naca"},
        "source_counts": {"bootstrap": int(len(X_boot)), "real_naca": int(len(X_real))},
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(X)} rows ({len(X_boot)} bootstrap + {len(X_real)} real_naca) -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

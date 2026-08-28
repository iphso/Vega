"""Exports TORAX design data for the Vega viewer -- the TORAX counterpart to
export_airfoil_viewer_data.py, same `meta.json`/`X.bin`/`Y.bin` contract
(`feature_layout` name->offset map) the stellarator viewer's design-browser
code already reads domain-agnostically.

Two sources, same "real vs. synthetic" pairing as the airfoil export:

  - `bootstrap`: a random subsample of generate_torax_dataset.py's own
    output (§35), filtered by BOTH the target-side sanity cap (Q_fusion/
    T_e_volume_avg/H98, §35/§37) and the input-side geometry cap
    (train_torax_cvae.py's _PARAM_CAPS, §37 Gap 3) -- the combined filter
    every TORAX training script already applies, not a new one invented
    here.
  - `real_devices`: the 4 real, named tokamaks from real_reference_torax.py
    (§36) -- ITER baseline, ITER hybrid flattop, SPARC, JET. Only 4 rows
    (no alpha-style sweep the way airfoils have 11 angles per shape -- this
    domain's real-reference set is real device geometry, not a swept
    condition), but genuinely real and non-self-generated, same as
    airfoils' `real_naca` source.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data/torax")

PARAM_NAMES = ["Ip", "nbar", "chi_i", "chi_e", "P_total", "I_generic",
               "R_major", "a_minor", "B_0", "elongation_LCFS"]
TARGET_NAMES = ["Q_fusion", "tau_E", "H98", "T_e_volume_avg"]
FEATURE_LAYOUT = {name: {"offset": i} for i, name in enumerate(PARAM_NAMES)}

_PARAM_CAPS = {"Ip": 6e7, "nbar": 5e20, "chi_i": 50, "chi_e": 50, "P_total": 5e8,
               "I_generic": 3e7, "R_major": 20, "a_minor": 8, "B_0": 40, "elongation_LCFS": 4}


def sanity_mask(X, Y):
    qi, tei, hi = (TARGET_NAMES.index(n) for n in ("Q_fusion", "T_e_volume_avg", "H98"))
    target_ok = (Y[:, qi] <= 300) & (Y[:, tei] <= 200) & (Y[:, hi] <= 20)
    param_ok = np.ones(len(X), dtype=bool)
    for i, name in enumerate(PARAM_NAMES):
        param_ok &= X[:, i] <= _PARAM_CAPS[name]
    return target_ok & param_ok


def load_bootstrap(n_sample, seed):
    X = np.load(OUT_DIR / "torax_X.npy")
    Y = np.load(OUT_DIR / "torax_Y.npy")
    sane = sanity_mask(X, Y)
    X, Y = X[sane], Y[sane]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    return X[idx].astype(np.float32), Y[idx].astype(np.float32)


def load_real_devices():
    refs = json.loads((OUT_DIR / "torax_real_references.json").read_text())
    rows_x, rows_y = [], []
    for name, overrides in refs["devices"].items():
        result = refs["results"][name]
        if not result["ok"]:
            continue
        p = result["payload"]
        rows_x.append(np.array([overrides[n] for n in PARAM_NAMES], dtype=np.float32))
        rows_y.append(np.array([p[n] for n in TARGET_NAMES], dtype=np.float32))
    return np.stack(rows_x), np.stack(rows_y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bootstrap", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_boot, Y_boot = load_bootstrap(args.n_bootstrap, args.seed)
    X_real, Y_real = load_real_devices()

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
        "source_legend": {"0": "bootstrap", "1": "real_devices"},
        "source_counts": {"bootstrap": int(len(X_boot)), "real_devices": int(len(X_real))},
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(X)} rows ({len(X_boot)} bootstrap + {len(X_real)} real_devices) -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

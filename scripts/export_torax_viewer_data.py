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

Chunked, not just subsampled (EXPERIMENT_LOG §61): direct user request --
filtering down to a narrow slice of the distribution should be able to
"grab more to fill it in" instead of being stuck with whatever fraction of
one fixed random sample happened to land there. The bootstrap pool is
shuffled ONCE in full (not `rng.choice` of a subset that discards the
rest) and sliced into `--chunk-size`-row chunks; chunk 0 (+ the small,
always-complete `real_devices` set, included here only) is written as
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


def load_bootstrap_shuffled(seed):
    """The FULL sane-filtered bootstrap pool, shuffled once -- callers slice
    off whatever prefix they need (chunk 0, then more chunks on demand),
    rather than discarding everything past a fixed sample size."""
    X = np.load(OUT_DIR / "torax_X.npy")
    Y = np.load(OUT_DIR / "torax_Y.npy")
    sane = sanity_mask(X, Y)
    X, Y = X[sane], Y[sane]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), Y[perm].astype(np.float32)


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
    ap.add_argument("--chunk-size", type=int, default=2500, help="rows per chunk; also chunk 0's bootstrap share, "
                                                                   "matching this script's old --n-bootstrap default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_boot, Y_boot = load_bootstrap_shuffled(args.seed)
    X_real, Y_real = load_real_devices()
    n_boot_total = len(X_boot)
    chunk_size = args.chunk_size

    X_all = np.concatenate([X_boot, X_real])
    Y_all = np.concatenate([Y_boot, Y_real])
    target_range = [[float(Y_all[:, t].min()), float(Y_all[:, t].max())] for t in range(Y_all.shape[1])]

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunk0_boot = min(chunk_size, n_boot_total)
    # real_devices FIRST, bootstrap LAST -- so a client-side top-up
    # (appending more bootstrap rows to the end of the array) never
    # disturbs real_devices's own contiguous [start, start+count) range.
    X0 = np.concatenate([X_real, X_boot[:chunk0_boot]])
    Y0 = np.concatenate([Y_real, Y_boot[:chunk0_boot]])
    X0.tofile(VIEWER_DATA_DIR / "X.bin")
    Y0.tofile(VIEWER_DATA_DIR / "Y.bin")

    n_chunks_total = max(1, -(-n_boot_total // chunk_size))  # ceil div, over the bootstrap pool only
    for c in range(1, n_chunks_total):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_boot_total)
        X_boot[lo:hi].tofile(VIEWER_DATA_DIR / f"X.chunk{c}.bin")
        Y_boot[lo:hi].tofile(VIEWER_DATA_DIR / f"Y.chunk{c}.bin")

    meta = {
        "n": int(len(X0)),
        "n_total": int(n_boot_total + len(X_real)),
        "n_chunks": int(n_chunks_total - 1),  # extra chunks beyond the initial load, bootstrap-only
        "bootstrap_target": int(chunk_size),
        "x_cols": int(X_all.shape[1]),
        "y_cols": int(Y_all.shape[1]),
        "target_names": TARGET_NAMES,
        "target_range": target_range,
        "source_legend": {"0": "real_devices", "1": "bootstrap"},
        "source_counts": {"bootstrap": int(chunk0_boot), "real_devices": int(len(X_real))},
        "feature_layout": FEATURE_LAYOUT,
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote chunk 0 ({chunk0_boot} bootstrap + {len(X_real)} real_devices) + {n_chunks_total - 1} more "
          f"bootstrap chunk(s) ({n_boot_total} bootstrap rows total) -> {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()

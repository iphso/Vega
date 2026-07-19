"""Exports accepted/labeled designs from every known data source into compact
binary files (+ a small metadata JSON) for viewer/, the browser-based 3D
candidate viewer. Binary rather than JSON for the actual float arrays --
JSON's text encoding would bloat ~250K rows x 92/11 floats considerably for
no benefit, and the browser just needs a raw ArrayBuffer to read via
Float32Array.

Rejected (nonphysical) candidates are intentionally left out of this first
version -- they have no Y labels, and folding them in means every consumer
of this export has to handle a "no metrics" case. Revisit if/when the
viewer grows a feasible/infeasible view.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data")


# "pilot_cluster" (the original 476-row k-means-targeted pilot, superseded by
# every run below it) is dropped entirely -- not exported at all. "bootstrap"
# now merges every generation of the VAE-driven exploration into one source:
# the untargeted 12h vae-prior run (its first, unlabeled-as-such cycle), the
# self-training sampling loop (bootstrap_bootstrap0), and the gradient-descent
# walks (gradient_walk_walk1) -- same lineage, same purpose, just different
# generations/mechanisms of it. Reduces the comparison to what it actually is
# now: real data vs. everything the VAE-bootstrap process has produced.
SOURCES = [
    ("real", [(OUT_DIR / "X.npy", OUT_DIR / "Y.npy")]),
    ("bootstrap", [
        (OUT_DIR / "generated_vae_prior" / "X.npy", OUT_DIR / "generated_vae_prior" / "Y.npy"),
        (OUT_DIR / "bootstrap_bootstrap0" / "X.npy", OUT_DIR / "bootstrap_bootstrap0" / "Y.npy"),
        (OUT_DIR / "gradient_walk_walk1" / "X.npy", OUT_DIR / "gradient_walk_walk1" / "Y.npy"),
    ]),
]


def main():
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())

    X_parts, Y_parts, source_codes = [], [], []
    source_legend, source_counts = {}, {}

    for code, (name, pairs) in enumerate(SOURCES):
        X_pieces, Y_pieces = [], []
        for x_path, y_path in pairs:
            if not x_path.exists():
                continue
            X = np.load(x_path).astype(np.float32)
            Y = np.load(y_path).astype(np.float32)
            assert len(X) == len(Y), f"{name} ({x_path}): X/Y length mismatch ({len(X)} vs {len(Y)})"
            X_pieces.append(X)
            Y_pieces.append(Y)
        if not X_pieces:
            continue
        X = np.concatenate(X_pieces) if len(X_pieces) > 1 else X_pieces[0]
        Y = np.concatenate(Y_pieces) if len(Y_pieces) > 1 else Y_pieces[0]
        X_parts.append(X)
        Y_parts.append(Y)
        source_codes.append(np.full(len(X), code, dtype=np.uint8))
        source_legend[code] = name
        source_counts[name] = int(len(X))
        print(f"{name}: {len(X):,} rows")

    X_all = np.concatenate(X_parts)
    Y_all = np.concatenate(Y_parts)
    source_all = np.concatenate(source_codes)

    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_all.tofile(VIEWER_DATA_DIR / "X.bin")
    Y_all.tofile(VIEWER_DATA_DIR / "Y.bin")
    source_all.tofile(VIEWER_DATA_DIR / "source.bin")

    meta = {
        "n": int(len(X_all)),
        "x_cols": int(X_all.shape[1]),
        "y_cols": int(Y_all.shape[1]),
        "target_names": target_names,
        "source_legend": source_legend,
        "source_counts": source_counts,
        "feature_layout": {
            "r_cos": {"offset": 0, "shape": [5, 9]},
            "z_sin": {"offset": 45, "shape": [5, 9]},
            "n_field_periods": {"offset": 90},
            "is_stellarator_symmetric": {"offset": 91},
        },
    }
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ntotal: {len(X_all):,} rows -> {VIEWER_DATA_DIR}")
    print(f"X.bin {X_all.nbytes/1e6:.1f}MB  Y.bin {Y_all.nbytes/1e6:.1f}MB  source.bin {source_all.nbytes/1e6:.1f}MB")


if __name__ == "__main__":
    main()

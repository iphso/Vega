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

--testing points the same write path at vega_testing/ (the VMEC-converged
candidate set, currently 308 rows) instead of the production real+bootstrap
pool, so the viewer can be developed against a small, fully-labeled set
without touching that directory. Y_vmec has 12 metric columns (the
production export drops aspect_ratio_over_edge_rotational_transform);
file_rows is preserved in meta.json as the join key back to the source dump.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

OUT_DIR = Path("/work/output")
VIEWER_DATA_DIR = Path("/work/viewer/public/data")
TESTING_DIR = Path("/work/vega_testing")


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
    (
        "bootstrap",
        [
            (
                OUT_DIR / "generated_vae_prior" / "X.npy",
                OUT_DIR / "generated_vae_prior" / "Y.npy",
            ),
            (
                OUT_DIR / "bootstrap_bootstrap0" / "X.npy",
                OUT_DIR / "bootstrap_bootstrap0" / "Y.npy",
            ),
            (
                OUT_DIR / "gradient_walk_walk1" / "X.npy",
                OUT_DIR / "gradient_walk_walk1" / "Y.npy",
            ),
        ],
    ),
]


def write_export(
    X_all: NDArray[np.float32],
    Y_all: NDArray[np.float32],
    source_all: NDArray[np.uint8],
    target_names: list[str],
    source_legend: dict[int, str],
    source_counts: dict[str, int],
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Write X/Y/source as raw binaries plus the meta.json the viewer reads."""
    VIEWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_all.tofile(VIEWER_DATA_DIR / "X.bin")
    Y_all.tofile(VIEWER_DATA_DIR / "Y.bin")
    source_all.tofile(VIEWER_DATA_DIR / "source.bin")

    meta = {
        "n": len(X_all),
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
    if extra_meta:
        meta.update(extra_meta)
    (VIEWER_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ntotal: {len(X_all):,} rows -> {VIEWER_DATA_DIR}")
    print(
        f"X.bin {X_all.nbytes / 1e6:.1f}MB  Y.bin {Y_all.nbytes / 1e6:.1f}MB  source.bin {source_all.nbytes / 1e6:.1f}MB"
    )


def export_production() -> None:
    """Export the real + bootstrap pool, one source code per group."""
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())

    X_parts: list[NDArray[np.float32]] = []
    Y_parts: list[NDArray[np.float32]] = []
    source_codes: list[NDArray[np.uint8]] = []
    source_legend: dict[int, str] = {}
    source_counts: dict[str, int] = {}

    for code, (name, pairs) in enumerate(SOURCES):
        X_pieces, Y_pieces = [], []
        for x_path, y_path in pairs:
            if not x_path.exists():
                continue
            X = np.load(x_path).astype(np.float32)
            Y = np.load(y_path).astype(np.float32)
            assert len(X) == len(Y), (
                f"{name} ({x_path}): X/Y length mismatch ({len(X)} vs {len(Y)})"
            )
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
        source_counts[name] = len(X)
        print(f"{name}: {len(X):,} rows")

    write_export(
        np.concatenate(X_parts),
        np.concatenate(Y_parts),
        np.concatenate(source_codes),
        target_names,
        source_legend,
        source_counts,
    )


def export_testing() -> None:
    """Read-only consumption of vega_testing/ -- never writes back there."""
    X = np.load(TESTING_DIR / "X.npy").astype(np.float32)
    Y = np.load(TESTING_DIR / "Y_vmec.npy").astype(np.float32)
    raw_names = np.load(TESTING_DIR / "metric_names.npy", allow_pickle=True)
    file_rows = np.load(TESTING_DIR / "file_rows.npy")
    assert len(X) == len(Y) == len(file_rows), (
        f"vega_testing length mismatch: X={len(X)} Y={len(Y)} file_rows={len(file_rows)}"
    )
    assert X.shape[1] == 92, f"expected X cols=92, got {X.shape[1]}"
    assert Y.shape[1] == len(raw_names), (
        f"Y cols ({Y.shape[1]}) != metric_names ({len(raw_names)})"
    )

    # metric_names.npy stores HuggingFace-style "metrics.qi" labels; the
    # viewer (and production target_names.json) uses the bare metric name.
    target_names = [str(n).removeprefix("metrics.") for n in raw_names]
    source_legend = {0: "vega_testing"}
    source_counts = {"vega_testing": len(X)}
    source_all = np.zeros(len(X), dtype=np.uint8)
    print(f"vega_testing: {len(X):,} rows, {len(target_names)} metrics")

    write_export(
        X,
        Y,
        source_all,
        target_names,
        source_legend,
        source_counts,
        extra_meta={"file_rows": [int(v) for v in file_rows]},
    )


def main() -> None:
    """CLI entry point: export the production pool, or --testing."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--testing",
        action="store_true",
        help="export vega_testing/ instead of the production real+bootstrap pool",
    )
    args = p.parse_args()
    if args.testing:
        export_testing()
    else:
        export_production()


if __name__ == "__main__":
    main()

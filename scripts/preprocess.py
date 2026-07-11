import json
import math
import os

import numpy as np
from datasets import load_dataset

OUT_DIR = "/work/output"

TARGET_COLS = [
    "qi",
    "vacuum_well",
    "aspect_ratio",
    "max_elongation",
    "average_triangularity",
    "axis_magnetic_mirror_ratio",
    "edge_magnetic_mirror_ratio",
    "axis_rotational_transform_over_n_field_periods",
    "edge_rotational_transform_over_n_field_periods",
    "flux_compression_in_regions_of_bad_curvature",
    "minimum_normalized_magnetic_gradient_scale_length",
    # aspect_ratio_over_edge_rotational_transform intentionally excluded: it's
    # aspect_ratio / edge_rotational_transform_over_n_field_periods, both of
    # which are already targets above, and it blows up to O(1e6) near
    # edge_rotational_transform ~ 0, which no smooth regressor can fit.
]

EXPECTED_COEFF_SHAPE = (5, 9)  # (poloidal modes, toroidal modes) established via full-dataset scan


def shape_of(v):
    shape = []
    cur = v
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(shape)


def build_feature_names():
    names = []
    for m in range(EXPECTED_COEFF_SHAPE[0]):
        for n in range(EXPECTED_COEFF_SHAPE[1]):
            names.append(f"r_cos_m{m}_n{n}")
    for m in range(EXPECTED_COEFF_SHAPE[0]):
        for n in range(EXPECTED_COEFF_SHAPE[1]):
            names.append(f"z_sin_m{m}_n{n}")
    names.append("n_field_periods")
    names.append("is_stellarator_symmetric")
    return names


def column_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    print("Loading dataset...")
    cols = [
        "boundary.r_cos",
        "boundary.z_sin",
        "boundary.n_field_periods",
        "boundary.is_stellarator_symmetric",
    ] + [f"metrics.{c}" for c in TARGET_COLS]

    ds = load_dataset("proxima-fusion/constellaration", split="train")
    ds = ds.select_columns(cols)
    data = ds[:]
    n_raw = len(data["boundary.r_cos"])
    print(f"Raw rows: {n_raw}")

    feature_rows = []
    target_rows = []

    dropped_malformed_boundary = 0
    dropped_invalid_targets = 0

    for i in range(n_raw):
        r_cos = data["boundary.r_cos"][i]
        z_sin = data["boundary.z_sin"][i]
        n_fp = data["boundary.n_field_periods"][i]
        is_sym = data["boundary.is_stellarator_symmetric"][i]

        if (
            shape_of(r_cos) != EXPECTED_COEFF_SHAPE
            or shape_of(z_sin) != EXPECTED_COEFF_SHAPE
            or n_fp is None
            or is_sym is None
        ):
            dropped_malformed_boundary += 1
            continue

        targets = [data[f"metrics.{c}"][i] for c in TARGET_COLS]
        if any(t is None or not math.isfinite(t) for t in targets):
            dropped_invalid_targets += 1
            continue

        r_flat = [v for row in r_cos for v in row]
        z_flat = [v for row in z_sin for v in row]
        feature_rows.append(r_flat + z_flat + [float(n_fp), float(bool(is_sym))])
        target_rows.append(targets)

    X = np.array(feature_rows, dtype=np.float32)
    Y = np.array(target_rows, dtype=np.float32)

    assert not np.isnan(X).any(), "NaNs found in X after filtering"
    assert not np.isnan(Y).any(), "NaNs found in Y after filtering"
    assert np.isfinite(Y).all(), "Non-finite values found in Y after filtering"

    n_samples, input_dim = X.shape
    output_dim = Y.shape[1]
    print(f"Kept rows: {n_samples} (dropped malformed boundary: {dropped_malformed_boundary}, "
          f"dropped invalid targets: {dropped_invalid_targets})")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    feature_names = build_feature_names()
    assert len(feature_names) == input_dim

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "Y.npy"), Y)

    with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(OUT_DIR, "target_names.json"), "w") as f:
        json.dump(TARGET_COLS, f, indent=2)

    feature_stats = {name: column_stats(X[:, j]) for j, name in enumerate(feature_names)}
    target_stats = {name: column_stats(Y[:, j]) for j, name in enumerate(TARGET_COLS)}

    metadata = {
        "n_samples": n_samples,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "raw_rows": n_raw,
        "dropped_malformed_boundary": dropped_malformed_boundary,
        "dropped_invalid_targets": dropped_invalid_targets,
        "fourier_coeff_shape": list(EXPECTED_COEFF_SHAPE),
        "feature_layout": {
            "r_cos": {"shape": list(EXPECTED_COEFF_SHAPE), "flatten_order": "C (row-major, m outer / n inner)", "offset": 0, "length": 45},
            "z_sin": {"shape": list(EXPECTED_COEFF_SHAPE), "flatten_order": "C (row-major, m outer / n inner)", "offset": 45, "length": 45},
            "n_field_periods": {"offset": 90, "length": 1},
            "is_stellarator_symmetric": {"offset": 91, "length": 1, "note": "1.0/0.0 flag; always 1.0 in this dataset"},
        },
        "feature_stats": feature_stats,
        "target_stats": target_stats,
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Wrote X.npy, Y.npy, feature_names.json, target_names.json, metadata.json")

    # 80/10/10 split, fixed seed
    seed = 42
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    np.savez(os.path.join(OUT_DIR, "train.npz"), X=X[train_idx], Y=Y[train_idx])
    np.savez(os.path.join(OUT_DIR, "val.npz"), X=X[val_idx], Y=Y[val_idx])
    np.savez(os.path.join(OUT_DIR, "test.npz"), X=X[test_idx], Y=Y[test_idx])

    print(f"Split -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)} (seed={seed})")


if __name__ == "__main__":
    main()

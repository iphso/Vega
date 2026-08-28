"""Noise-floor check for the 4 SOAP-trunk architectures in screen_reference_baselines.py,
per EXPERIMENT_LOG §2's standing convention: a "win" isn't trusted until it's shown to
exceed what re-running the exact same config with a different seed produces by chance
alone. §15's architecture ranking (mlp_soap best on classification at 0.961 AUC, with
siren/half_siren/attention all within 0.001; half_siren_soap best on regression at 0.176
median relative RMSE, with siren/mlp_soap close behind) was single-seed — this reruns each
architecture 3x (model-init/training seed only; the train/val/test split is fixed at the
original seed=42, exactly matching the main run, so this isolates training-noise from
data-split noise) and reports mean +/- std.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from screen_reference_baselines import (
    DATA, SEED, RAW_INPUT_ARCHS, TRUNK_ARCHS, LOG_TARGET_NAMES, DROP_METRIC,
    train_soap_model, torch_predict,
)

MODEL_SEEDS = [0, 1, 2]  # training-only seeds; data split below is fixed at SEED=42 throughout


def to_log_space(Y, log_mask):
    Y = Y.copy()
    Y[:, log_mask] = np.log(np.clip(Y[:, log_mask], 1e-12, None))
    return Y


def noise_floor_stage1(X, y):
    print("\n" + "=" * 70)
    print("NOISE FLOOR — STAGE 1: predict `converged`")
    print("=" * 70)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

    out = {}
    for arch in TRUNK_ARCHS:
        Xtr_a, Xval_a, Xte_a = (Xtr, Xval, Xte) if arch in RAW_INPUT_ARCHS else (Xtr_s, Xval_s, Xte_s)
        aucs = []
        for ms in MODEL_SEEDS:
            t0 = time.time()
            model, n_epochs = train_soap_model(arch, Xtr_a, ytr.astype(np.float32), Xval_a, yval.astype(np.float32),
                                                out_dim=1, task="classification", seed=ms)
            logits = torch_predict(model, Xte_a)[:, 0]
            proba = 1 / (1 + np.exp(-logits))
            auc = roc_auc_score(yte, proba)
            aucs.append(auc)
            print(f"  {arch}_soap seed={ms}: AUC={auc:.4f}  epochs={n_epochs}  time={time.time()-t0:.1f}s")
        mean, std = float(np.mean(aucs)), float(np.std(aucs))
        print(f"  {arch}_soap: {mean:.4f} +/- {std:.4f}  (n={len(aucs)})")
        out[f"{arch}_soap"] = {"seeds": aucs, "mean": mean, "std": std}
    return out


def noise_floor_stage2(X, metrics, converged, names):
    print("\n" + "=" * 70)
    print("NOISE FLOOR — STAGE 2: predict the 11 metrics")
    print("=" * 70)
    keep_cols = [i for i, n in enumerate(names) if n != DROP_METRIC]
    keep_names = [names[i].replace("metrics.", "") for i in keep_cols]
    log_mask = np.array([n in LOG_TARGET_NAMES for n in keep_names])

    Xc = X[converged]
    Yc = metrics[converged][:, keep_cols]
    Xtr, Xtmp, Ytr, Ytmp = train_test_split(Xc, Yc, test_size=0.2, random_state=SEED)
    Xval, Xte, Yval, Yte = train_test_split(Xtmp, Ytmp, test_size=0.5, random_state=SEED)

    x_scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(to_log_space(Ytr, log_mask))
    Xtr_s, Xval_s, Xte_s = x_scaler.transform(Xtr), x_scaler.transform(Xval), x_scaler.transform(Xte)
    Ytr_s = y_scaler.transform(to_log_space(Ytr, log_mask))
    Yval_s = y_scaler.transform(to_log_space(Yval, log_mask))
    Yte_eval = to_log_space(Yte, log_mask)

    mean_pred_eval = np.tile(to_log_space(Ytr, log_mask).mean(axis=0), (len(Yte), 1))
    rmse_mean_eval = np.sqrt(mean_squared_error(Yte_eval, mean_pred_eval, multioutput="raw_values"))

    out = {}
    for arch in TRUNK_ARCHS:
        Xtr_a, Xval_a, Xte_a = (Xtr, Xval, Xte) if arch in RAW_INPUT_ARCHS else (Xtr_s, Xval_s, Xte_s)
        medians = []
        for ms in MODEL_SEEDS:
            t0 = time.time()
            model, n_epochs = train_soap_model(arch, Xtr_a, Ytr_s.astype(np.float32), Xval_a,
                                                Yval_s.astype(np.float32), out_dim=Ytr.shape[1],
                                                task="regression", seed=ms)
            pred_eval = y_scaler.inverse_transform(torch_predict(model, Xte_a))
            rmse = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
            med = float(np.median(rmse / rmse_mean_eval))
            medians.append(med)
            print(f"  {arch}_soap seed={ms}: median_rel_rmse={med:.4f}  epochs={n_epochs}  time={time.time()-t0:.1f}s")
        mean, std = float(np.mean(medians)), float(np.std(medians))
        print(f"  {arch}_soap: {mean:.4f} +/- {std:.4f}  (n={len(medians)})")
        out[f"{arch}_soap"] = {"seeds": medians, "mean": mean, "std": std}
    return out


if __name__ == "__main__":
    X = np.load(f"{DATA}/X.npy")
    metrics = np.load(f"{DATA}/metrics.npy")
    converged = np.load(f"{DATA}/converged.npy")
    names = json.load(open(f"{DATA}/metric_names.json"))

    s1 = noise_floor_stage1(X, converged.astype(int))
    s2 = noise_floor_stage2(X, metrics, converged, names)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'arch':<20}{'stage1 AUC (mean+/-std)':>30}{'stage2 med.rel.RMSE (mean+/-std)':>36}")
    for arch in TRUNK_ARCHS:
        a = s1[f"{arch}_soap"]
        b = s2[f"{arch}_soap"]
        print(f"{arch+'_soap':<20}{a['mean']:.4f} +/- {a['std']:.4f}{'':>10}{b['mean']:.4f} +/- {b['std']:.4f}")

    json.dump({"stage1": s1, "stage2": s2}, open(f"{DATA}/noise_floor_results.json", "w"), indent=2)
    print(f"\nSaved to {DATA}/noise_floor_results.json")

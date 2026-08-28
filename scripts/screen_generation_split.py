"""How much does the screen_reference_baselines.py random-split numbers overstate
generalization? EXPERIMENT_LOG §6 found random splits leak badly on the real dataset
(2.5-3x worse aggregate RMSE under a cluster split, some targets 9-19x) because of
generation-lineage families. This pool has an even more direct temporal structure
available for free: generation_idx.npy (bootstrap_pool rows only — the gradient_walk_search
rows have no generation concept and are excluded here, same as §15's open thread flagged).

Split: train on early generations, test on late ones — the realistic deployment scenario
(train a surrogate on data collected so far, score genuinely new candidates), and a much
harder leakage bar than a random row split, since consecutive generations in a bootstrap/
self-training loop are anchored on each other's outputs (EXPERIMENT_LOG §9-10).

Only 3 models per stage (not all 8 from the reference suite) — same footprint as §6's own
"small baseline vs. production recipe" comparison, not a full re-sweep: a linear floor
(logistic/ridge), mlp_soap (best raw-input MLP per §15), attention_soap (independently
best-tuned different family, scaled input).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from screen_reference_baselines import (
    DATA, SEED, RAW_INPUT_ARCHS, LOG_TARGET_NAMES, DROP_METRIC,
    train_soap_model, torch_predict,
)

MODEL_ARCHS = ["mlp", "attention"]  # + a linear baseline, per stage, see module docstring
TRAIN_GEN_MAX = 67   # generations 1-67 -> train  (~80% of 84 generations)
VAL_GEN_MAX = 75      # 68-75 -> val (~9.5%)
# 76-84 -> test (~10.7%)


def load_split():
    X = np.load(f"{DATA}/X.npy")
    metrics = np.load(f"{DATA}/metrics.npy")
    converged = np.load(f"{DATA}/converged.npy")
    source = np.load(f"{DATA}/source.npy")
    gen = np.load(f"{DATA}/generation_idx.npy")
    names = json.load(open(f"{DATA}/metric_names.json"))

    pool = source == "bootstrap_pool"
    print(f"bootstrap_pool rows: {pool.sum()} / {len(source)} total "
          f"(excluding {(~pool).sum()} gradient_walk_search rows — no generation concept)")

    train_mask = pool & (gen >= 1) & (gen <= TRAIN_GEN_MAX)
    val_mask = pool & (gen > TRAIN_GEN_MAX) & (gen <= VAL_GEN_MAX)
    test_mask = pool & (gen > VAL_GEN_MAX)
    print(f"train (gen 1-{TRAIN_GEN_MAX}): {train_mask.sum()}  "
          f"val (gen {TRAIN_GEN_MAX+1}-{VAL_GEN_MAX}): {val_mask.sum()}  "
          f"test (gen {VAL_GEN_MAX+1}-84): {test_mask.sum()}")
    return X, metrics, converged, names, train_mask, val_mask, test_mask


def stage1_classification(X, converged, train_mask, val_mask, test_mask, seed=SEED):
    print("\n" + "=" * 70)
    print("GENERATION SPLIT — STAGE 1: predict `converged`")
    print("=" * 70)
    Xtr, ytr = X[train_mask], converged[train_mask].astype(int)
    Xval, yval = X[val_mask], converged[val_mask].astype(int)
    Xte, yte = X[test_mask], converged[test_mask].astype(int)
    print(f"base rate: train={ytr.mean():.4f} val={yval.mean():.4f} test={yte.mean():.4f}  "
          f"(train/test base-rate gap alone shows the distribution shift across generations)")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

    results = {}
    t0 = time.time()
    logreg = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    results["logistic_regression"] = logreg.predict_proba(Xte_s)[:, 1]
    print(f"  logistic_regression fit in {time.time()-t0:.1f}s")

    for arch in MODEL_ARCHS:
        Xtr_a, Xval_a, Xte_a = (Xtr, Xval, Xte) if arch in RAW_INPUT_ARCHS else (Xtr_s, Xval_s, Xte_s)
        t0 = time.time()
        model, n_epochs = train_soap_model(arch, Xtr_a, ytr.astype(np.float32), Xval_a, yval.astype(np.float32),
                                            out_dim=1, task="classification", seed=seed)
        logits = torch_predict(model, Xte_a)[:, 0]
        results[f"{arch}_soap"] = 1 / (1 + np.exp(-logits))
        print(f"  {arch}_soap fit in {time.time()-t0:.1f}s ({n_epochs} epochs)")

    print(f"\n{'model':<24}{'AUC':>8}{'accuracy':>10}{'brier':>10}")
    out = {}
    for name, proba in results.items():
        auc, acc, brier = roc_auc_score(yte, proba), accuracy_score(yte, proba > 0.5), brier_score_loss(yte, proba)
        print(f"{name:<24}{auc:>8.4f}{acc:>10.4f}{brier:>10.4f}")
        out[name] = {"auc": float(auc), "accuracy": float(acc), "brier": float(brier)}
    return out


def stage2_regression(X, metrics, converged, names, train_mask, val_mask, test_mask, seed=SEED):
    print("\n" + "=" * 70)
    print("GENERATION SPLIT — STAGE 2: predict the 11 metrics")
    print("=" * 70)
    keep_cols = [i for i, n in enumerate(names) if n != DROP_METRIC]
    keep_names = [names[i].replace("metrics.", "") for i in keep_cols]
    log_mask = np.array([n in LOG_TARGET_NAMES for n in keep_names])

    def to_log_space(Y):
        Y = Y.copy()
        Y[:, log_mask] = np.log(np.clip(Y[:, log_mask], 1e-12, None))
        return Y

    conv_train, conv_val, conv_test = train_mask & converged, val_mask & converged, test_mask & converged
    Xtr, Ytr = X[conv_train], metrics[conv_train][:, keep_cols]
    Xval, Yval = X[conv_val], metrics[conv_val][:, keep_cols]
    Xte, Yte = X[conv_test], metrics[conv_test][:, keep_cols]
    print(f"converged rows: train={len(Xtr)} val={len(Xval)} test={len(Xte)}")

    x_scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(to_log_space(Ytr))
    Xtr_s, Xval_s, Xte_s = x_scaler.transform(Xtr), x_scaler.transform(Xval), x_scaler.transform(Xte)
    Ytr_s = y_scaler.transform(to_log_space(Ytr))
    Yval_s = y_scaler.transform(to_log_space(Yval))
    Yte_eval = to_log_space(Yte)

    mean_pred_eval = np.tile(to_log_space(Ytr).mean(axis=0), (len(Yte), 1))
    rmse_mean_eval = np.sqrt(mean_squared_error(Yte_eval, mean_pred_eval, multioutput="raw_values"))

    models_eval = {}
    t0 = time.time()
    ridge = Ridge(alpha=1.0).fit(Xtr_s, Ytr_s)
    pred_eval = y_scaler.inverse_transform(ridge.predict(Xte_s))
    models_eval["ridge"] = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
    print(f"  ridge fit in {time.time()-t0:.1f}s")

    for arch in MODEL_ARCHS:
        Xtr_a, Xval_a, Xte_a = (Xtr, Xval, Xte) if arch in RAW_INPUT_ARCHS else (Xtr_s, Xval_s, Xte_s)
        t0 = time.time()
        model, n_epochs = train_soap_model(arch, Xtr_a, Ytr_s.astype(np.float32), Xval_a, Yval_s.astype(np.float32),
                                            out_dim=Ytr.shape[1], task="regression", seed=seed)
        pred_eval = y_scaler.inverse_transform(torch_predict(model, Xte_a))
        models_eval[f"{arch}_soap"] = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
        print(f"  {arch}_soap fit in {time.time()-t0:.1f}s ({n_epochs} epochs)")

    rel = {name: rmse / rmse_mean_eval for name, rmse in models_eval.items()}
    header = "".join(f"{name:>16}" for name in models_eval)
    print(f"\n{'target':<50}{'mean-pred':>12}{header}   | relative RMSE (eval space)")
    out = {}
    for i, tname in enumerate(keep_names):
        row = "".join(f"{rmse[i]:>16.4f}" for rmse in models_eval.values())
        rel_row = " / ".join(f"{rel[name][i]:.3f}" for name in models_eval)
        print(f"{tname:<50}{rmse_mean_eval[i]:>12.4f}{row}   | {rel_row}")
        out[tname] = {"mean_pred_eval": float(rmse_mean_eval[i]),
                       "relative_rmse": {name: float(rel[name][i]) for name in models_eval}}
    med_row = " / ".join(f"{np.median(rel[name]):.3f}" for name in models_eval)
    print(f"\nMEDIAN relative RMSE across 11 targets: {med_row}")
    out["_median_relative_rmse"] = {name: float(np.median(r)) for name, r in rel.items()}
    return out


if __name__ == "__main__":
    X, metrics, converged, names, train_mask, val_mask, test_mask = load_split()
    stage1 = stage1_classification(X, converged, train_mask, val_mask, test_mask)
    stage2 = stage2_regression(X, metrics, converged, names, train_mask, val_mask, test_mask)
    json.dump({"stage1_classification": stage1, "stage2_regression": stage2},
               open(f"{DATA}/generation_split_results.json", "w"), indent=2)
    print(f"\nSaved to {DATA}/generation_split_results.json")

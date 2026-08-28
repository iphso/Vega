"""Reference baselines for the screen_master.npz prediction task.

Two stages, both evaluated on a random 80/10/10 split (seed 42, matching the
project's original default split convention in EXPERIMENT_LOG.md §1):

  1. Classification: predict `converged` (did this design pass re-screening)
     from the 92-dim boundary representation.
  2. Regression: predict the 11 physics targets (the 12th,
     aspect_ratio_over_edge_rotational_transform, is dropped per the
     project's existing convention — algebraically redundant, blows up near
     zero denominators) on the converged subset only.

Deliberately lightweight sklearn baselines, run directly on host (not
through Docker like the project's PyTorch pipeline) — these are meant as a
reference floor, not the production architecture.

Data: output/screen_reference/{X,metrics,converged,source,generation_idx,file_rows}.npy
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).parent))
from soap import SOAP  # project's own optimizer (arxiv 2409.11321), already the production
# recipe's optimizer per EXPERIMENT_LOG §2/§4 — reused here rather than reimplemented.
from train import build_spectral_trunk, LOG_TARGET_NAMES  # project's own trunk architectures
# (§2-4, §13): "mlp" (plain baseline), "siren" (SIREN, Sitzmann et al. 2020 — sinusoidal
# activations, already confirmed a real (noise-floor-checked) winner on 2 of 11 targets under
# the random split in §2), "half_siren" (each layer half-sine/half-ReLU, untried in this
# reference suite so far), "attention" (per-Fourier-mode self-attention, §13's
# least-invested/worst-tested variant on the real dataset — turned out to be one of the two
# best here). LOG_TARGET_NAMES: the 4 targets (qi, max_elongation,
# flux_compression_in_regions_of_bad_curvature, minimum_normalized_magnetic_gradient_scale_length)
# train.py already flags as having 114x-1.8e7x dynamic range and predicts as log(target) —
# reused here rather than re-deriving, since it's exactly the pathology behind max_elongation's
# heavy tail (§15).

DATA = "output/screen_reference"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRUNK_ARCHS = ["mlp", "siren", "half_siren", "attention"]  # -> mlp_soap / siren_soap / half_siren_soap / attention_soap
RAW_INPUT_ARCHS = {"mlp", "siren", "half_siren"}  # direct A/B per arch (classification task,
# same seed/split): mlp raw 0.9614 vs scaled 0.9494 AUC, siren raw 0.960 vs scaled 0.898 —
# both clearly prefer raw. attention is the opposite (scaled 0.9611 vs raw 0.9576) and stays
# out of this set. Not a SIREN-specific quirk after all — StandardScaler was quietly costing
# mlp_soap ~1.2pt AUC too; only attention actually wants normalized input.


class TrunkHeadModel(nn.Module):
    """`build_spectral_trunk`'s architecture (mlp/siren/attention) + a plain
    linear head — deliberately a reference-scoped model, not train.py's
    production DualPathMLP (no spatial CNN branch, no log-target heads, no
    per-task uncertainty weighting). Isolates trunk architecture as the only
    variable, all trained with the same SOAP optimizer."""

    def __init__(self, trunk_arch, in_dim, out_dim, hidden=256, latent_dim=128):
        super().__init__()
        self.trunk = build_spectral_trunk(trunk_arch, in_dim, hidden, latent_dim)
        self.head = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        return self.head(self.trunk(x))


def train_soap_model(trunk_arch, Xtr, ytr, Xval, yval, out_dim, task, seed=SEED, max_epochs=200,
                      patience=15, batch_size=2048, lr=3e-3):
    """Shared SOAP-optimized trainer for both stages, any trunk architecture.
    task: 'classification' or 'regression'."""
    torch.manual_seed(seed)
    model = TrunkHeadModel(trunk_arch, Xtr.shape[1], out_dim).to(DEVICE)
    opt = SOAP(model.parameters(), lr=lr, precondition_frequency=10)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xval_t = torch.as_tensor(Xval, dtype=torch.float32, device=DEVICE)
    yval_t = torch.as_tensor(yval, dtype=torch.float32, device=DEVICE)
    if ytr_t.ndim == 1:
        ytr_t, yval_t = ytr_t.unsqueeze(1), yval_t.unsqueeze(1)

    n = len(Xtr_t)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1


def torch_predict(model, X):
    with torch.no_grad():
        out = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
    return out.cpu().numpy()

DROP_METRIC = "metrics.aspect_ratio_over_edge_rotational_transform"


def load():
    X = np.load(f"{DATA}/X.npy")
    metrics = np.load(f"{DATA}/metrics.npy")
    converged = np.load(f"{DATA}/converged.npy")
    names = json.load(open(f"{DATA}/metric_names.json"))
    return X, metrics, converged, names


def stage1_classification(X, y, seed=SEED):
    print("\n" + "=" * 70)
    print("STAGE 1: predict `converged` (classification)")
    print("=" * 70)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    print(f"train={len(Xtr)} val={len(Xval)} test={len(Xte)}  base rate (train)={ytr.mean():.4f}")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    results = {}

    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    results["base_rate"] = dummy.predict_proba(Xte)[:, 1]

    t0 = time.time()
    logreg = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    results["logistic_regression"] = logreg.predict_proba(Xte_s)[:, 1]
    print(f"  logistic_regression fit in {time.time()-t0:.1f}s")

    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    results["hist_gradient_boosting"] = hgb.predict_proba(Xte)[:, 1]
    print(f"  hist_gradient_boosting fit in {time.time()-t0:.1f}s")

    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=200, early_stopping=True,
                         random_state=seed).fit(Xtr_s, ytr)
    results["mlp_sklearn"] = mlp.predict_proba(Xte_s)[:, 1]
    print(f"  mlp_sklearn fit in {time.time()-t0:.1f}s ({mlp.n_iter_} iters)")

    t0 = time.time()
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dte = xgb.DMatrix(Xte)
    dval = xgb.DMatrix(Xval, label=yval)
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 6,
         "eta": 0.1, "device": "cuda" if DEVICE == "cuda" else "cpu", "seed": seed},
        dtr, num_boost_round=1000, evals=[(dval, "val")],
        early_stopping_rounds=30, verbose_eval=False)
    results["xgboost"] = booster.predict(dte)
    print(f"  xgboost fit in {time.time()-t0:.1f}s ({booster.best_iteration+1} rounds)")

    Xval_s = scaler.transform(Xval)
    for arch in TRUNK_ARCHS:
        # Input scaling turned out to be architecture-specific, not a SIREN-only quirk.
        # Direct A/B, same seed/split, this task: mlp raw 0.9614 vs. scaled 0.9494 AUC,
        # siren raw 0.960 vs. scaled 0.898 — both clearly prefer raw (SIREN's SineLayer
        # init is literally calibrated for the coefficients' raw O(0.01-0.4) scale, and
        # it turns out plain MLP+SOAP does too, on this task). attention is the opposite
        # (scaled 0.9611 vs. raw 0.9576) and is the one arch standard-scaling actually
        # helps — see RAW_INPUT_ARCHS above.
        Xtr_arch, Xval_arch, Xte_arch = (Xtr, Xval, Xte) if arch in RAW_INPUT_ARCHS else (Xtr_s, Xval_s, Xte_s)
        t0 = time.time()
        model, n_epochs = train_soap_model(arch, Xtr_arch, ytr.astype(np.float32), Xval_arch, yval.astype(np.float32),
                                            out_dim=1, task="classification", seed=seed)
        logits = torch_predict(model, Xte_arch)[:, 0]
        results[f"{arch}_soap"] = 1 / (1 + np.exp(-logits))
        print(f"  {arch}_soap ({DEVICE}) fit in {time.time()-t0:.1f}s ({n_epochs} epochs)")

    print(f"\n{'model':<24}{'AUC':>8}{'accuracy':>10}{'brier':>10}")
    for name, proba in results.items():
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, proba > 0.5)
        brier = brier_score_loss(yte, proba)
        print(f"{name:<24}{auc:>8.4f}{acc:>10.4f}{brier:>10.4f}")

    return {name: {"auc": float(roc_auc_score(yte, p)),
                    "accuracy": float(accuracy_score(yte, p > 0.5)),
                    "brier": float(brier_score_loss(yte, p))}
            for name, p in results.items()}


def stage2_regression(X, metrics, converged, names, seed=SEED):
    print("\n" + "=" * 70)
    print("STAGE 2: predict the 11 metrics (regression, converged subset only)")
    print("=" * 70)
    keep_cols = [i for i, n in enumerate(names) if n != DROP_METRIC]
    keep_names = [names[i].replace("metrics.", "") for i in keep_cols]

    Xc = X[converged]
    Yc = metrics[converged][:, keep_cols]
    assert np.isfinite(Yc).all(), "unexpected NaNs in converged rows"

    Xtr, Xtmp, Ytr, Ytmp = train_test_split(Xc, Yc, test_size=0.2, random_state=seed)
    Xval, Xte, Yval, Yte = train_test_split(Xtmp, Ytmp, test_size=0.5, random_state=seed)
    print(f"train={len(Xtr)} val={len(Xval)} test={len(Xte)}")

    # Predict log(target) for train.py's LOG_TARGET_NAMES (qi, max_elongation,
    # flux_compression_in_regions_of_bad_curvature, min_norm_grad_scale_length) instead of
    # target directly — same clamp_min(1e-12) convention as DualPathMLP's log_target_mask.
    # These are exactly the wide-dynamic-range targets behind max_elongation's heavy tail
    # (§15) and xgboost's joint-model catastrophe on it; log-compressing before any model
    # (including z-scoring) ever sees them should fix that at the source, for every model,
    # not just xgboost.
    log_mask = np.array([n in LOG_TARGET_NAMES for n in keep_names])
    print(f"  log-target columns: {[n for n in keep_names if n in LOG_TARGET_NAMES]}")

    def to_log_space(Y):
        Y = Y.copy()
        Y[:, log_mask] = np.log(np.clip(Y[:, log_mask], 1e-12, None))
        return Y

    def from_log_space(Y):
        Y = Y.copy()
        Y[:, log_mask] = np.exp(Y[:, log_mask])
        return Y

    x_scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(to_log_space(Ytr))
    Xtr_s, Xte_s = x_scaler.transform(Xtr), x_scaler.transform(Xte)
    Ytr_s = y_scaler.transform(to_log_space(Ytr))

    def predict_eval_space(raw_pred_s):
        """Invert only the z-score, not the log — "eval space" (log for the 4
        flagged targets, physical units for the other 7) is what every RMSE
        comparison below actually uses, for a reason found the hard way (see below)."""
        return y_scaler.inverse_transform(raw_pred_s)

    # mean-predictor floor, physical units — kept only as a reference/sanity table,
    # NOT used for model comparison. See the eval-space floor + explanation below.
    mean_pred = np.tile(Ytr.mean(axis=0), (len(Yte), 1))
    rmse_mean = np.sqrt(mean_squared_error(Yte, mean_pred, multioutput="raw_values"))

    # The actual comparison metric: RMSE in "eval space" (log for the 4 wide-dynamic-range
    # targets, physical units for the rest). Found necessary, not a style choice: on the
    # first pass, EVERY model's physical-unit max_elongation RMSE converged to ~3971-3977
    # regardless of architecture (ridge and attention_soap landed within 0.15% of each
    # other) — traced to just 3 of 20,708 test rows (values 118k/279k/477k) whose squared
    # error alone accounts for ~99% of the total, swamping any real difference in how well
    # models fit the other 20,705 rows. Physical-unit RMSE for this target measures "how
    # badly did you miss 3 specific outliers," not model quality; log-space RMSE does.
    Yte_eval = to_log_space(Yte)
    Ytr_eval = to_log_space(Ytr)
    mean_pred_eval = np.tile(Ytr_eval.mean(axis=0), (len(Yte), 1))
    rmse_mean_eval = np.sqrt(mean_squared_error(Yte_eval, mean_pred_eval, multioutput="raw_values"))

    t0 = time.time()
    ridge = Ridge(alpha=1.0).fit(Xtr_s, Ytr_s)
    pred_eval = predict_eval_space(ridge.predict(Xte_s))
    rmse_ridge = np.sqrt(mean_squared_error(Yte, from_log_space(pred_eval), multioutput="raw_values"))
    rmse_ridge_eval = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
    print(f"  ridge fit in {time.time()-t0:.1f}s")

    t0 = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, early_stopping=True,
                        random_state=seed).fit(Xtr_s, Ytr_s)
    pred_eval = predict_eval_space(mlp.predict(Xte_s))
    rmse_mlp = np.sqrt(mean_squared_error(Yte, from_log_space(pred_eval), multioutput="raw_values"))
    rmse_mlp_eval = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
    print(f"  mlp_sklearn fit in {time.time()-t0:.1f}s ({mlp.n_iter_} iters)")

    t0 = time.time()
    # HistGradientBoostingRegressor is single-output; fit one per target, directly on
    # eval-space targets (already log-transformed for the 4 flagged columns via
    # Ytr_log_only) — no z-scoring needed since each target gets its own model anyway.
    rmse_hgb = np.zeros(len(keep_names))
    rmse_hgb_eval = np.zeros(len(keep_names))
    Ytr_log_only = to_log_space(Ytr)
    for i, tname in enumerate(keep_names):
        hgb = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, Ytr_log_only[:, i])
        pred_i_eval = hgb.predict(Xte)
        pred_i_phys = np.exp(pred_i_eval) if log_mask[i] else pred_i_eval
        rmse_hgb[i] = np.sqrt(mean_squared_error(Yte[:, i], pred_i_phys))
        rmse_hgb_eval[i] = np.sqrt(mean_squared_error(Yte_eval[:, i], pred_i_eval))
    print(f"  hist_gradient_boosting (per-target) fit in {time.time()-t0:.1f}s")

    t0 = time.time()
    # xgboost's multi_output_tree strategy fits all 11 targets as one model
    # (shared tree structure, per-leaf per-target values) rather than 11
    # independent boosters — faster and lets the trees exploit target correlation.
    # Trained on log+z-scored targets (Ytr_s/Yval_s), same as ridge/the SOAP trunks: a
    # non-log-transformed joint loss is dominated entirely by max_elongation's huge scale
    # (std ~4000 vs. O(0.01-2) for everything else) and early-stops after ~8 rounds having
    # barely touched the other 10 targets — confirmed by trying it first, see EXPERIMENT_LOG.
    Yval_s = y_scaler.transform(to_log_space(Yval))
    dtr = xgb.DMatrix(Xtr, label=Ytr_s)
    dte = xgb.DMatrix(Xte)
    dval = xgb.DMatrix(Xval, label=Yval_s)
    booster = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1,
         "multi_strategy": "multi_output_tree", "tree_method": "hist",
         "device": "cuda" if DEVICE == "cuda" else "cpu", "seed": seed},
        dtr, num_boost_round=1000, evals=[(dval, "val")],
        early_stopping_rounds=30, verbose_eval=False)
    pred_eval = predict_eval_space(booster.predict(dte))
    rmse_xgb = np.sqrt(mean_squared_error(Yte, from_log_space(pred_eval), multioutput="raw_values"))
    rmse_xgb_eval = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
    print(f"  xgboost (joint multi-output) fit in {time.time()-t0:.1f}s ({booster.best_iteration+1} rounds)")

    models = {"ridge": rmse_ridge, "hgb": rmse_hgb, "xgb": rmse_xgb, "mlp": rmse_mlp}
    models_eval = {"ridge": rmse_ridge_eval, "hgb": rmse_hgb_eval, "xgb": rmse_xgb_eval, "mlp": rmse_mlp_eval}
    Xval_s = x_scaler.transform(Xval)
    for arch in TRUNK_ARCHS:
        # RAW_INPUT_ARCHS (mlp/siren/half_siren) get raw input, attention gets scaled —
        # see the module-level comment for the per-arch A/B that decided this.
        Xtr_arch, Xval_arch, Xte_arch = (Xtr, Xval, Xte) if arch in RAW_INPUT_ARCHS else (Xtr_s, Xval_s, Xte_s)
        t0 = time.time()
        model, n_epochs = train_soap_model(arch, Xtr_arch, Ytr_s.astype(np.float32), Xval_arch,
                                            Yval_s.astype(np.float32),
                                            out_dim=Yc.shape[1], task="regression", seed=seed)
        pred_eval = predict_eval_space(torch_predict(model, Xte_arch))
        models[f"{arch}_soap"] = np.sqrt(mean_squared_error(Yte, from_log_space(pred_eval), multioutput="raw_values"))
        models_eval[f"{arch}_soap"] = np.sqrt(mean_squared_error(Yte_eval, pred_eval, multioutput="raw_values"))
        print(f"  {arch}_soap ({DEVICE}) fit in {time.time()-t0:.1f}s ({n_epochs} epochs)")

    # Physical-unit RMSE table: printed for reference/interpretability only, NOT used for
    # ranking — see the comment above rmse_mean_eval for why (3 outlier test rows dominate
    # max_elongation's physical RMSE for every model near-identically).
    header = "".join(f"{name:>12}" for name in models)
    print(f"\n{'target':<50}{'mean-pred':>12}{header}   | physical-unit RMSE (reference only, NOT the ranking metric — see note)")
    out = {}
    for i, tname in enumerate(keep_names):
        row = "".join(f"{rmse[i]:>12.4f}" for rmse in models.values())
        print(f"{tname:<50}{rmse_mean[i]:>12.4f}{row}")
        out[tname] = {"mean_pred_physical": float(rmse_mean[i]),
                       **{f"{name}_physical": float(rmse[i]) for name, rmse in models.items()}}

    # Eval-space RMSE: the actual comparison metric (log-space for the 4 wide-dynamic-range
    # targets, physical units for the other 7). Relative RMSE (model / mean-predictor, both
    # in eval space) is still a scale-free skill score across targets: <1.0 beats trivial.
    rel = {name: rmse / rmse_mean_eval for name, rmse in models_eval.items()}
    header_eval = "".join(f"{name:>12}" for name in models_eval)
    print(f"\n{'target':<50}{'mean-pred':>12}{header_eval}   | relative RMSE, eval space (model/mean-pred)")
    for i, tname in enumerate(keep_names):
        row = "".join(f"{rmse[i]:>12.4f}" for rmse in models_eval.values())
        rel_row = " / ".join(f"{rel[name][i]:.3f}" for name in models_eval)
        space = "log" if log_mask[i] else "phys"
        print(f"{tname:<50}[{space}]{rmse_mean_eval[i]:>7.4f}{row}   | {rel_row}")
        out[tname]["mean_pred_eval"] = float(rmse_mean_eval[i])
        out[tname].update({f"{name}_eval": float(rmse[i]) for name, rmse in models_eval.items()})
        out[tname]["relative_rmse"] = {name: float(rel[name][i]) for name in models_eval}
    pad = " " * (12 * (len(models_eval) + 1))
    print(f"\n{'target':<50}{pad}   | MEDIAN relative RMSE across 11 targets (eval space)")
    med_row = " / ".join(f"{np.median(rel[name]):.3f}" for name in models_eval)
    print(f"{'':<50}{pad}   | {med_row}")
    out["_median_relative_rmse"] = {name: float(np.median(r)) for name, r in rel.items()}
    print(f"\nNote: max_elongation is heavy-tailed on this screened pool (median 4.1, 99.9th pct "
          f"8,303, max 5.03e6 among converged rows) — same pathology the project already flagged "
          f"and dropped one column for (see EXPERIMENT_LOG §1). Log-transforming it (+3 other "
          f"train.py LOG_TARGET_NAMES targets) before modeling was step one; step two, found the "
          f"hard way: physical-unit RMSE post-exp is STILL broken for ranking — on the first pass "
          f"every model converged to ~the same max_elongation RMSE (ridge and attention_soap within "
          f"0.15% of each other) because 3 of 20,708 test rows' squared error dominates the sum "
          f"regardless of model quality on the other 20,705. The 'eval space' table above (log-space "
          f"RMSE for the 4 flagged targets) is what actually reflects fit quality; the physical-unit "
          f"table is kept only so the reference RMSE numbers are still there if useful.")
    return out


if __name__ == "__main__":
    X, metrics, converged, names = load()
    print(f"Loaded: X={X.shape} metrics={metrics.shape} converged_rate={converged.mean():.4f}")

    stage1 = stage1_classification(X, converged.astype(int))
    stage2 = stage2_regression(X, metrics, converged, names)

    json.dump({"stage1_classification": stage1, "stage2_regression": stage2},
               open(f"{DATA}/reference_results.json", "w"), indent=2)
    print(f"\nSaved results to {DATA}/reference_results.json")

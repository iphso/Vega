"""Diagnostic, not a permanent eval: chases EXPERIMENT_LOG §37's flagged
finding further (direct user request) -- diffusion's steer-candidate
convergence collapses (~34-74% across step_std 0.5-2.0) while cVAE/GAN stay
much higher (~91-97%/44-92%), even though diffusion's own null-mode (no
push at all) convergence is 98.6%, matching cVAE/GAN's null numbers
closely (96.5%/94.4%). So the gap is specifically about what happens once
the conditioning target moves away from the anchor's own real target, not
an inherent diffusion sampling-quality problem.

This script reuses steerability_generic.py's own building blocks
(load_generative_model, random_anchor_sample, sample_unit_directions,
build_candidates -- same 12 anchors/seed=0/axis-mode/step_std=1.0 as the
§37 headline numbers) but, unlike that script, keeps the RAW decoded
params and the oracle's FULL failure message for every candidate instead
of collapsing to a single ok/not-ok bit -- the two pieces of information
needed to actually test a mechanism: (1) does a candidate's raw geometry
violate train_torax_cvae.py's own input-side sanity caps (§37's Gap 3
finding -- R_major/elongation/etc. pushed to absurd values), and (2) what
kind of failure is it (a clean TORAX SimError vs. a timeout/hang vs. a
structural config-build error) -- cross-tabulated per model.
"""
import json
from pathlib import Path

import numpy as np
import torch

from steerability_generic import (
    build_candidates, load_dataset, load_generative_model,
    random_anchor_sample, eval_space,
)
from oracle_harness_persistent import run_batch_persistent
from train_torax_cvae import PARAM_NAMES, _PARAM_CAPS

OUT_DIR = Path("/work/output")
N_ANCHORS = 12
K_STEER = 5  # a bit more than steerability_generic.py's default 3, for a cleaner per-model failure breakdown
STEP_STD = 1.0
SEED = 0


def cap_violations(params_row):
    return [name for i, name in enumerate(PARAM_NAMES) if params_row[i] > _PARAM_CAPS[name]]


def run_model(model_type, tag, dev):
    bundle = load_generative_model("torax", model_type, tag, dev)
    spec, conditioning = bundle["spec"], bundle["conditioning"]
    sample_fn, target_names, log_target_names, n_targets = (
        bundle["sample_fn"], bundle["target_names"], bundle["log_target_names"], bundle["n_targets"])
    coeff_mean, coeff_std, t_mean, t_std = bundle["coeff_mean"], bundle["coeff_std"], bundle["t_mean"], bundle["t_std"]
    param_dim = bundle["param_dim"]

    X, Y = load_dataset(spec, bundle["dataset_tag"])
    if spec.sanity_filter is not None:
        mask = spec.sanity_filter(Y)
        X, Y = X[mask], Y[mask]
    Yz = (eval_space(Y, target_names, log_target_names) - t_mean) / t_std
    anchor_idx = random_anchor_sample(Yz, N_ANCHORS, seed=SEED)

    fidelity_name = 25  # resolved value, matches gym_schema's FidelityLevel("low", ...) for torax

    jobs = []
    raw_params = {}  # tag -> (10,) raw unnormalized params, for later cap-violation lookup
    for a_i, idx in enumerate(anchor_idx):
        row = X[idx]
        aux = conditioning.aux_from_row(row, param_dim)
        anchor_z = Yz[idx]
        worker_aux = conditioning.worker_aux(aux)
        for dir_i in range(n_targets):
            u = np.zeros(n_targets)
            u[dir_i] = -1.0
            steer_z = anchor_z + STEP_STD * u
            cond = conditioning.cond_from_target_and_aux(steer_z, aux)
            decoded = sample_fn(cond, K_STEER)
            params = decoded * coeff_std + coeff_mean
            if spec.zero_indices:
                params[:, spec.zero_indices] = 0.0
            for k in range(K_STEER):
                tag_k = ("steer", a_i, dir_i, k)
                raw_params[tag_k] = params[k]
                jobs.append((tag_k, *spec.params_to_worker_args(params[k], worker_aux, fidelity_name)))

    results = {}
    for tag, ok, payload in run_batch_persistent(jobs, spec.persistent_worker_fn, n_workers=16, timeout_s=30.0):
        results[tag] = (ok, payload)

    return raw_params, results, target_names


def summarize(model_type, raw_params, results):
    n_total = len(results)
    n_ok = sum(1 for ok, _ in results.values() if ok)
    fail_reasons = {}
    fail_with_cap_violation = 0
    ok_with_cap_violation = 0
    fail_violations_detail = []
    for tag, (ok, payload) in results.items():
        violations = cap_violations(raw_params[tag])
        if ok:
            if violations:
                ok_with_cap_violation += 1
        else:
            if violations:
                fail_with_cap_violation += 1
                fail_violations_detail.append(violations)
            reason = str(payload)
            # bucket into a small set of categories rather than every distinct message
            if "timed out" in reason:
                bucket = "timeout/hang"
            elif "NAN_DETECTED" in reason:
                bucket = "SimError: NAN_DETECTED"
            elif "LOW_TEMPERATURE_COLLAPSE" in reason:
                bucket = "SimError: LOW_TEMPERATURE_COLLAPSE"
            elif "REACHED_MIN_DT" in reason:
                bucket = "SimError: REACHED_MIN_DT"
            elif "DID_NOT_REACH_T_FINAL" in reason:
                bucket = "SimError: DID_NOT_REACH_T_FINAL"
            elif "structural" in reason:
                bucket = "structural (config build failed)"
            else:
                bucket = f"other: {reason[:60]}"
            fail_reasons[bucket] = fail_reasons.get(bucket, 0) + 1

    n_fail = n_total - n_ok
    print(f"\n=== {model_type} ===")
    print(f"  {n_ok}/{n_total} converged ({n_ok/n_total:.1%})")
    print(f"  failure reasons ({n_fail} total):")
    for bucket, count in sorted(fail_reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:4d}  {bucket}")
    print(f"  of {n_fail} failures: {fail_with_cap_violation} ({fail_with_cap_violation/max(n_fail,1):.1%}) "
          f"violate an input-side sanity cap (§37 Gap 3)")
    print(f"  of {n_ok} successes: {ok_with_cap_violation} ({ok_with_cap_violation/max(n_ok,1):.1%}) "
          f"violate an input-side sanity cap")

    # which params get violated most often among failures
    from collections import Counter
    violation_counts = Counter(name for v in fail_violations_detail for name in v)
    if violation_counts:
        print(f"  which params get violated (among failing candidates that violate at least one):")
        for name, count in violation_counts.most_common():
            print(f"    {name}: {count}")

    # raw magnitude comparison: converged vs failed, for a couple of the most physically salient params
    for pname in ["R_major", "elongation_LCFS", "B_0", "Ip"]:
        pidx = PARAM_NAMES.index(pname)
        ok_vals = [raw_params[tag][pidx] for tag, (ok, _) in results.items() if ok]
        fail_vals = [raw_params[tag][pidx] for tag, (ok, _) in results.items() if not ok]
        print(f"  {pname}: converged median={np.median(ok_vals):.3g} (n={len(ok_vals)})  "
              f"failed median={np.median(fail_vals):.3g} (n={len(fail_vals)})")

    return {
        "model_type": model_type, "n_total": n_total, "n_ok": n_ok,
        "fail_reasons": fail_reasons,
        "fail_with_cap_violation": fail_with_cap_violation, "ok_with_cap_violation": ok_with_cap_violation,
        "n_fail": n_fail,
    }


def main():
    dev = torch.device("cpu")
    summary = {}
    for model_type in ["cvae", "diffusion", "gan"]:
        tag = f"torax_{model_type}_s0"
        raw_params, results, target_names = run_model(model_type, tag, dev)
        summary[model_type] = summarize(model_type, raw_params, results)

    (OUT_DIR / "diag_torax_diffusion_convergence.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved -> {OUT_DIR / 'diag_torax_diffusion_convergence.json'}")


if __name__ == "__main__":
    main()

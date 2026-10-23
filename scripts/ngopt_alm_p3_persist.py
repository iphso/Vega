"""ngopt_alm_p3.py variant testing one specific structural hypothesis for why
ngopt_v1/v2 both stalled (flat best_worst_violation while rho exploded, see
EXPERIMENT_LOG): the original script recreates a FRESH NGOpt optimizer every
single outer ALM iteration (budget=56, then discarded), recentered on a small
box around x_best. That means NGOpt's chosen sub-optimizer (typically a CMA
variant) never gets to build up any real internal model -- covariance
adaptation, step-size adaptation, momentum -- across the run. Each outer step
is architecturally closer to a small random-perturbation batch around x_best
than genuine gradient-free optimization. This is NOT what "ALM-NGOpt" means
in the paper: a proper ALM wrapper keeps ONE optimizer's internal state alive
across outer steps and only changes the *objective it's told* (via evolving
lambda/rho) between ask/tell batches.

This script fixes exactly that, and nothing else structurally:
  - ONE `ng.p.Array` / NGOpt optimizer created once for the whole run
    (budget = n_outer * budget_per_outer), asked/told in chunks of
    budget_per_outer real-VMEC evals so the same parallel harness still
    applies, but NEVER recreated.
  - `mutable_sigma=True` -- let NGOpt's own sub-optimizer adapt its step size
    (this is most of the actual power of CMA-style search; the original
    script disabled it).
  - Bounds are FIXED for the whole run (a generous box around the seed point,
    default sigma0 * 60 -- the same cap ngopt_v2's stall-widening eventually
    hit), not recentered/regrown every outer iteration -- recentering a fresh
    box every step is itself a symptom of restart-based design and doesn't
    make sense once the optimizer's own internal state (mean/covariance) is
    what's tracking "where we are" instead.
  - The ALM lambda/rho update logic (PHR formulation, paper_feasibility_
    violation for exact match with the benchmark) is UNCHANGED from
    ngopt_alm_p3.py -- this is a one-variable test of the persistent-
    optimizer hypothesis, not a rewrite of the whole ALM scheme.

Must run in the `train` service. Output: output/ngopt_alm_p3_persist/
{results.jsonl, <tag>_success.json, <tag>_checkpoint.json} -- separate
directory from ngopt_alm_p3.py's own output so both can run/be compared
side by side without collision.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402
from optimize import paper_feasibility_violation  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "ngopt_alm_p3_persist"

P3_CONSTRAINTS = [
    ("edge_rotational_transform_over_n_field_periods", ">=", 0.25, True, None, False),
    ("qi", "<=", 10 ** -3.5, False, None, True),
    ("edge_magnetic_mirror_ratio", "<=", 0.25, False, None, False),
    ("flux_compression_in_regions_of_bad_curvature", "<=", 0.9, False, None, False),
    ("vacuum_well", ">=", 0.0, False, 0.1, False),
]

FAIL_PENALTY = 50.0


def real_violations(measured):
    import torch
    out = []
    for name, op, value, use_abs, divisor, log_t in P3_CONSTRAINTS:
        col = torch.tensor([[measured[name]]], dtype=torch.float32)
        v = paper_feasibility_violation(col[:, 0], op, value, use_abs=use_abs, divisor=divisor, log_transform=log_t)
        out.append(float(v.item()))
    return np.array(out)


def alm_loss(violations, objective, obj_weight, lam, rho):
    penalty = np.sum(np.maximum(0.0, lam + rho * violations) ** 2 - lam ** 2) / (2 * rho)
    return -obj_weight * objective + penalty


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed-x", default=str(OUT_DIR / "hl_nearwall_X.npy"))
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--n-outer", type=int, default=150, help="number of ask/tell batches (not independent restarts)")
    p.add_argument("--budget-per-outer", type=int, default=56)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--sigma0", type=float, default=0.0008)
    p.add_argument("--bound-mult", type=float, default=60.0,
                    help="FIXED bound half-width as a multiple of sigma0, for the whole run")
    p.add_argument("--rho0", type=float, default=1.0)
    p.add_argument("--rho-growth", type=float, default=1.15)
    p.add_argument("--obj-weight", type=float, default=0.02)
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import nevergrad as ng

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"{args.tag}_results.jsonl"
    success_path = STATE_DIR / f"{args.tag}_success.json"
    checkpoint_path = STATE_DIR / f"{args.tag}_checkpoint.json"
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    n_c = len(P3_CONSTRAINTS)
    total_budget = args.n_outer * args.budget_per_outer

    X = np.load(args.seed_x)
    x0 = X[args.seed_row][:90].astype(np.float64)
    nfp = int(X[args.seed_row][90])
    print(f"[ngopt_persist] seeded from row {args.seed_row}, nfp={nfp}, total_budget={total_budget}")

    bound_width = np.maximum(args.sigma0 * args.bound_mult, 1e-6)
    lower = x0 - bound_width
    upper = x0 + bound_width
    lower[oracle.ZERO_INDICES] = -1e-6
    upper[oracle.ZERO_INDICES] = 1e-6

    # mutable_sigma=True: the actual fix under test. nevergrad requires a
    # SCALAR sigma when mutable (it becomes its own adapting parameter, not a
    # frozen per-dim vector) -- init at the empirically-safe value (see
    # ngopt_alm_p3.py's own sigma-scan docstring) and let the sub-optimizer
    # adapt it over the run instead of it being frozen forever. ZERO_INDICES
    # stay pinned regardless, since their bounds are +-1e-6 either way.
    param = ng.p.Array(init=x0, mutable_sigma=True).set_bounds(lower=lower, upper=upper)
    param.set_mutation(sigma=args.sigma0)
    optimizer = ng.optimizers.registry["NGOpt"](parametrization=param, budget=total_budget,
                                                  num_workers=args.n_workers)

    lam = np.zeros(n_c)
    rho = args.rho0
    x_best = x0.copy()
    best_worst = 1e9
    start_outer = 0

    if args.resume and checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text())
        x_best = np.array(ck["x_best"], dtype=np.float64)
        lam = np.array(ck["lam"], dtype=np.float64)
        rho = ck["rho"]
        best_worst = ck["best_worst"]
        start_outer = ck["outer"] + 1
        # NOTE: optimizer internal state is NOT restored across --resume
        # (nevergrad doesn't pickle cleanly through this harness) -- a
        # --resume run restarts the optimizer's adaptation from x_best, which
        # is a real limitation of --resume specifically, not of the no-resume
        # persistent-optimizer run this script is primarily testing.
        print(f"[ngopt_persist] resumed bookkeeping from outer={start_outer} "
              f"(optimizer state NOT restored -- fresh optimizer re-centered on x_best)")
        param2 = ng.p.Array(init=x_best, mutable_sigma=True).set_bounds(lower=lower, upper=upper)
        param2.set_mutation(sigma=args.sigma0)
        optimizer = ng.optimizers.registry["NGOpt"](parametrization=param2,
                                                      budget=total_budget - start_outer * args.budget_per_outer,
                                                      num_workers=args.n_workers)

    all_results = []
    tag_ctr = itertools.count()

    for outer in range(start_outer, args.n_outer):
        asked = [optimizer.ask() for _ in range(args.budget_per_outer)]

        jobs, cand_by_tag = [], {}
        for cand in asked:
            tag = next(tag_ctr)
            params = np.array(cand.value, dtype=np.float64)
            params[oracle.ZERO_INDICES] = 0.0
            r_cos, z_sin, job_nfp, fname = oracle.params_to_worker_args(params, {"nfp": nfp}, fidelity_name)
            jobs.append((tag, r_cos, z_sin, job_nfp, fname))
            cand_by_tag[tag] = cand

        scored = {}
        for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
            cand = cand_by_tag[tag]
            cand_params = np.array(cand.value, dtype=np.float64)
            cand_params[oracle.ZERO_INDICES] = 0.0
            row = {"outer": outer, "converged": False, "params": cand_params.tolist(), "nfp": nfp}
            fail_loss = alm_loss(np.full(n_c, FAIL_PENALTY), 0.0, args.obj_weight, lam, rho)
            if ok:
                y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
                if all(v is not None for v in y.values()):
                    viol = real_violations(y)
                    obj = y["minimum_normalized_magnetic_gradient_scale_length"]
                    loss = alm_loss(viol, obj, args.obj_weight, lam, rho)
                    worst = float(viol.max())
                    row.update({"converged": True, "measured_targets": y, "violations": viol.tolist(),
                                "worst_violation": worst, "objective": obj, "loss": loss})
                    scored[tag] = (loss, worst, cand_params, viol)
                    if worst < best_worst:
                        best_worst = worst
                        x_best = cand_params.copy()
                    if worst <= args.relative_tol:
                        success_path.write_text(json.dumps({
                            "outer": outer, "measured_targets": y, "worst_violation": worst,
                            "params": cand_params.tolist(), "nfp": nfp,
                        }, indent=2))
                        print(f"[ngopt_persist] *** REAL P3 FEASIBILITY REACHED at outer={outer} *** "
                              f"worst_violation={worst:+.4f}")
                    optimizer.tell(cand, loss)
                else:
                    scored[tag] = (fail_loss, 1e9, cand_params, None)
                    optimizer.tell(cand, fail_loss)
            else:
                scored[tag] = (fail_loss, 1e9, cand_params, None)
                optimizer.tell(cand, fail_loss)
            all_results.append(row)

        n_conv = sum(1 for v in scored.values() if v[1] < 1e9)
        if n_conv:
            best_tag = min(scored, key=lambda t: scored[t][0])
            _, worst_here, x_here, viol_here = scored[best_tag]
            if viol_here is not None:
                lam = np.maximum(0.0, lam + rho * viol_here)
        rho *= args.rho_growth

        with open(results_path, "a") as f:
            for r in all_results[-args.budget_per_outer:]:
                f.write(json.dumps(r) + "\n")
        checkpoint_path.write_text(json.dumps({
            "outer": outer, "x_best": x_best.tolist(), "lam": lam.tolist(),
            "rho": rho, "best_worst": best_worst,
        }))
        print(f"[ngopt_persist] outer {outer}: {n_conv}/{args.budget_per_outer} converged, "
              f"best_worst_violation={best_worst:.4f}, rho={rho:.3f}")

        if success_path.exists():
            print("[ngopt_persist] success reached, stopping")
            break

    print(f"[ngopt_persist] done, best_worst_violation={best_worst:.4f}")


if __name__ == "__main__":
    main()

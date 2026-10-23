"""ngopt_alm_p3_official.py, re-targeted at the ACTUAL published settings for
the paper's own P3 ALM-NGOpt run (arxiv:2506.19583 section A.4.1), fetched
and quoted directly from the paper text -- not the pip package's dataclass
defaults, which turn out to differ from what was actually run:

    NGOpt budget per outer iter k:  min(20_000, 1500 + 300*k)
    rho_i^0 = 10,  rho_max = 1e8
    constraint violation tolerance reduction factor tau = 0.8  (package
        default is 0.5 -- the real run used a LOOSER tolerance, i.e. rho
        grows less readily than the package's own default would)
    penalty increase factor sigma = 5  (package default is 2 -- the real run
        grows rho much more aggressively per-doubling-event than default)
    trust-region bounds: delta_0 = 0.33, shrink factor gamma = 0.95/outer,
        floored at delta_min = 0.05
    Run on a 96-vCPU machine for 15-24 hours (i.e. genuinely hundreds of
    thousands to low millions of real VMEC++ evaluations in a single run).

This is the actual, quantitative answer to "why did all 8 of our prior
attempts stall around worst_violation~0.26-0.28 instead of reaching 0":
every prior attempt (ngopt_v1/v2, persist/persist_cma, the 4 alternate
seeds, omnig_v1, official_v1) used a per-outer budget of 40-140 -- 10-500x
smaller than the paper's own schedule at the SAME outer iteration index.
The paper's own baseline is not claiming a different search technique
finds the answer; it's running the same ALM-NGOpt idea with 1-2 orders of
magnitude more real-VMEC compute per attempt than anything tried here so
far. This script is the first real attempt to match that, not just the
mechanics.

Two concrete changes versus ngopt_alm_p3_official.py:

1. **Bounds actually shrink now.** ngopt_alm_p3_official.py found (correctly,
   for the pip package's generic `augmented_lagrangian_runner.run`) that
   `update_augmented_lagrangian_state` never shrinks bounds unless a caller
   explicitly passes a reduced value in -- but the paper's own reported
   gamma=0.95/delta_min=0.05 settings only make sense if the ACTUAL run
   script (which may differ from what ships in the pip package today, or
   may compute+pass bounds explicitly at a call site not shown in the
   inspected source) does shrink them. Implemented directly here:
   bound_mult *= --bounds-shrink every outer iteration, floored at
   --bound-mult-min. Stall-triggered widening is DISABLED by default
   (--stall-patience defaults effectively off via a huge default) since the
   paper's own recipe monotonically narrows rather than reactively widens.

2. **CLI defaults changed to the paper's own numbers** where this script's
   flags already generalized to them (rho0, rho-increase-factor, rho-max,
   tol-reduction-factor, budget-initial/increment/max) -- no new plumbing
   needed for those, just different numbers.

Still NOT attempted: the infinity-norm spectral reparametrization of the
search space itself. Given the budget gap alone plausibly explains the
whole prior stall pattern, that is deprioritized until this run's own
result is in.

Must run in the `train` service. Output: output/ngopt_alm_p3_paper/
{results.jsonl, <tag>_success.json, <tag>_checkpoint.json}.
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
STATE_DIR = OUT_DIR / "ngopt_alm_p3_paper"

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
    """Per-constraint rho (vector), matching the official formulation:
    L = obj_term + sum_i 0.5*rho_i*(relu(lam_i/rho_i + c_i)**2 - (lam_i/rho_i)**2)
    which is algebraically identical to sum_i (relu(lam_i+rho_i*c_i)**2-lam_i**2)/(2*rho_i)
    -- same PHR term, just per-constraint rho instead of scalar."""
    penalty = np.sum((np.maximum(0.0, lam + rho * violations) ** 2 - lam ** 2) / rho)
    return -obj_weight * objective + penalty


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed-x", default=str(OUT_DIR / "hl_nearwall_X.npy"))
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--n-outer", type=int, default=100)
    p.add_argument("--budget-initial", type=int, default=1500, help="paper's own schedule: min(20000, 1500+300k)")
    p.add_argument("--budget-increment", type=int, default=300)
    p.add_argument("--budget-max", type=int, default=20000)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--sigma0", type=float, default=0.0008)
    p.add_argument("--rho0", type=float, default=10.0, help="paper's own rho_i^0=10")
    p.add_argument("--rho-increase-factor", type=float, default=5.0, help="paper's own sigma=5")
    p.add_argument("--rho-max", type=float, default=1e8, help="paper's own rho_max=1e8")
    p.add_argument("--tol-reduction-factor", type=float, default=0.8,
                    help="paper's own tau=0.8 -- rho_i only grows if max(0,c_i) > this * max(0,c_i_prev)")
    p.add_argument("--obj-weight", type=float, default=0.02)
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bound-mult-initial", type=float, default=60.0,
                    help="starting trust-region half-width as a multiple of sigma0 (analogue of paper's delta_0, "
                         "not numerically comparable since we don't reparametrize by the infinity-norm spectral "
                         "scale -- picked to match this project's own previously-proven generous starting box)")
    p.add_argument("--bounds-shrink", type=float, default=0.95, help="paper's own gamma=0.95, applied every outer")
    p.add_argument("--bound-mult-min", type=float, default=9.0,
                    help="floor for bound_mult, ~ bound_mult_initial * (delta_min/delta_0) = 60*(0.05/0.33)")
    p.add_argument("--stall-patience", type=int, default=10 ** 9,
                    help="disabled by default -- the paper's recipe monotonically shrinks bounds rather than "
                         "reactively widening on stall; set finite to re-enable this project's own fallback")
    p.add_argument("--stall-growth", type=float, default=1.6)
    p.add_argument("--max-bound-mult", type=float, default=60.0)
    args = p.parse_args()

    import nevergrad as ng

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"{args.tag}_results.jsonl"
    success_path = STATE_DIR / f"{args.tag}_success.json"
    checkpoint_path = STATE_DIR / f"{args.tag}_checkpoint.json"
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    sigma = np.full(90, args.sigma0)
    sigma[oracle.ZERO_INDICES] = 0.0

    n_c = len(P3_CONSTRAINTS)
    start_outer = 0
    bound_mult = args.bound_mult_initial
    stall_count = 0
    budget = args.budget_initial
    prev_viol_clip = np.zeros(n_c)  # max(0, c_i) from the previous outer's best point

    if args.resume and checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text())
        x_best = np.array(ck["x_best"], dtype=np.float64)
        nfp = ck["nfp"]
        lam = np.array(ck["lam"], dtype=np.float64)
        rho = np.array(ck["rho"], dtype=np.float64)
        best_worst = ck["best_worst"]
        start_outer = ck["outer"] + 1
        bound_mult = ck.get("bound_mult", bound_mult)
        stall_count = ck.get("stall_count", 0)
        budget = ck.get("budget", budget)
        prev_viol_clip = np.array(ck.get("prev_viol_clip", prev_viol_clip.tolist()))
        print(f"[ngopt_paper] resumed from checkpoint at outer={start_outer}, "
              f"best_worst={best_worst:.4f}, bound_mult={bound_mult:.1f}, budget={budget}")
    else:
        X = np.load(args.seed_x)
        x0 = X[args.seed_row][:90].astype(np.float64)
        nfp = int(X[args.seed_row][90])
        print(f"[ngopt_paper] seeded from row {args.seed_row}, nfp={nfp}")
        lam = np.zeros(n_c)
        rho = np.full(n_c, args.rho0)
        x_best = x0.copy()
        best_worst = 1e9

    all_results = []
    tag_ctr = itertools.count()

    for outer in range(start_outer, args.n_outer):
        bound_width = np.maximum(args.sigma0 * bound_mult, 1e-6)
        lower = x_best - bound_width
        upper = x_best + bound_width
        lower[oracle.ZERO_INDICES] = -1e-6
        upper[oracle.ZERO_INDICES] = 1e-6
        param = ng.p.Array(init=x_best, mutable_sigma=False).set_bounds(lower=lower, upper=upper)
        param.set_mutation(sigma=sigma)
        optimizer = ng.optimizers.registry["NGOpt"](parametrization=param, budget=budget,
                                                      num_workers=args.n_workers)
        # official recipe: explicitly inject the incumbent into the ask
        # stream every outer iteration, on top of param init=x_best.
        optimizer.suggest(x_best)

        asked = [optimizer.ask() for _ in range(budget)]

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
                        print(f"[ngopt_paper] *** REAL P3 FEASIBILITY REACHED at outer={outer} *** "
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
                viol_clip = np.maximum(0.0, viol_here)
                # official conditional per-constraint rho growth: only double
                # rho_i where this constraint's violation did NOT shrink to
                # at least half of what it was last outer iteration.
                grow_mask = viol_clip > args.tol_reduction_factor * prev_viol_clip
                rho = np.where(grow_mask, np.minimum(args.rho_max, rho * args.rho_increase_factor), rho)
                prev_viol_clip = viol_clip

        budget = int(min(args.budget_max, budget + args.budget_increment))

        # paper's own recipe: monotonic geometric shrink every outer
        # iteration, floored -- NOT reactive widening (stall-patience is
        # disabled by default in this script, see argparse help).
        bound_mult = max(args.bound_mult_min, bound_mult * args.bounds_shrink)

        if outer == start_outer:
            prev_best_worst = best_worst
        if best_worst < prev_best_worst - 1e-9:
            stall_count = 0
        else:
            stall_count += 1
        prev_best_worst = best_worst
        if stall_count >= args.stall_patience and bound_mult < args.max_bound_mult:
            bound_mult = min(bound_mult * args.stall_growth, args.max_bound_mult)
            stall_count = 0
            print(f"[ngopt_paper] outer {outer}: stalled {args.stall_patience} iters, "
                  f"widening bound_mult -> {bound_mult:.1f}")

        with open(results_path, "a") as f:
            for r in all_results[-len(asked):]:
                f.write(json.dumps(r) + "\n")
        checkpoint_path.write_text(json.dumps({
            "outer": outer, "x_best": x_best.tolist(), "nfp": nfp, "lam": lam.tolist(),
            "rho": rho.tolist(), "best_worst": best_worst, "bound_mult": bound_mult,
            "stall_count": stall_count, "budget": budget, "prev_viol_clip": prev_viol_clip.tolist(),
        }))
        print(f"[ngopt_paper] outer {outer}: {n_conv}/{len(asked)} converged, "
              f"best_worst_violation={best_worst:.4f}, rho={np.array2string(rho, precision=2)}, "
              f"bound_mult={bound_mult:.1f}, budget={budget}")

        if success_path.exists():
            print("[ngopt_paper] success reached, stopping")
            break

    print(f"[ngopt_paper] done, best_worst_violation={best_worst:.4f}")


if __name__ == "__main__":
    main()

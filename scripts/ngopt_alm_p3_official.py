"""ngopt_alm_p3.py variant that grafts on the specific mechanics of the REAL
official ALM implementation (`constellaration.optimization.augmented_lagrangian`
+ `augmented_lagrangian_runner.run`, read directly via source inspection --
see EXPERIMENT_LOG), which differ from this project's hand-rolled version in
three concrete, actionable ways. All 7 methods tried before this stalled in
the same qualitative place (see EXPERIMENT_LOG's "Consolidated conclusion");
this is the first of the two explicitly-named remaining directions.

1. **Per-constraint penalty parameters, grown conditionally, not a single
   global rho grown unconditionally.** The official code keeps a vector
   rho_i (one per constraint) and only DOUBLES rho_i for a constraint whose
   violation did not shrink to at least half its previous value:
       rho_i <- min(rho_max, 2*rho_i)   if max(0,c_i) > 0.5*max(0,c_i_prev)
       rho_i <- rho_i                    otherwise
   Our original script grows a single scalar rho *= 1.15 every outer
   iteration regardless of progress -- rho exploded into the billions/
   trillions in ngopt_v1/v2 even while genuinely still improving on some
   constraints, over-penalizing the whole vector for one stuck constraint
   and needlessly sharpening the loss landscape everywhere at once.

2. **Growing per-outer budget** (budget_initial -> budget_max via
   budget_increment each outer iteration), not a constant budget_per_outer
   forever. Early outer iterations (far from any real optimum, cheap to be
   wrong about) get a small budget; later ones (refining near the actual
   optimum) get a bigger one.

3. **Explicit `oracle.suggest(x_best)`** -- the official code injects the
   current incumbent directly into NGOpt's ask stream every outer iteration
   (on top of param `init=x_best`, which only seeds the distribution, not a
   guaranteed evaluation). Cheap, harmless, added here too.

What's NOT grafted on (explicitly, so this isn't overclaimed as a full
reproduction): the official code also reparametrizes search space by an
infinity-norm spectral scaling per Fourier mode (`compute_infinity_norm_
spectrum_scaling_fun`) before handing coordinates to NGOpt -- skipped here,
real engineering lift for uncertain payoff. And despite the `Augmented
LagrangianSettings.bounds_reduction_factor=0.95` docstring implying shrinking
trust-region bounds, direct inspection of `update_augmented_lagrangian_state`
shows bounds are actually held at `max(bounds_min, state.bounds)` --
i.e. CONSTANT unless a caller explicitly overrides them, which
`augmented_lagrangian_runner.run` never does. So "shrinking bounds" was a
docstring artifact, not real behavior -- this script keeps this project's
own proven stall-triggered widening logic instead of inventing a shrink
mechanism the official code doesn't actually use.

Must run in the `train` service. Output: output/ngopt_alm_p3_official/
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
STATE_DIR = OUT_DIR / "ngopt_alm_p3_official"

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
    p.add_argument("--n-outer", type=int, default=80)
    p.add_argument("--budget-initial", type=int, default=40)
    p.add_argument("--budget-increment", type=int, default=6)
    p.add_argument("--budget-max", type=int, default=140)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--sigma0", type=float, default=0.0008)
    p.add_argument("--rho0", type=float, default=1.0)
    p.add_argument("--rho-increase-factor", type=float, default=2.0,
                    help="official default: penalty_parameters_increase_factor")
    p.add_argument("--rho-max", type=float, default=1e8, help="official default: penalty_parameters_max")
    p.add_argument("--tol-reduction-factor", type=float, default=0.5,
                    help="official default: constraint_violation_tolerance_reduction_factor -- "
                         "rho_i only grows if max(0,c_i) > this * max(0,c_i_prev)")
    p.add_argument("--obj-weight", type=float, default=0.02)
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stall-patience", type=int, default=12)
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
    bound_mult = 10.0
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
        print(f"[ngopt_official] resumed from checkpoint at outer={start_outer}, "
              f"best_worst={best_worst:.4f}, bound_mult={bound_mult:.1f}, budget={budget}")
    else:
        X = np.load(args.seed_x)
        x0 = X[args.seed_row][:90].astype(np.float64)
        nfp = int(X[args.seed_row][90])
        print(f"[ngopt_official] seeded from row {args.seed_row}, nfp={nfp}")
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
                        print(f"[ngopt_official] *** REAL P3 FEASIBILITY REACHED at outer={outer} *** "
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
            print(f"[ngopt_official] outer {outer}: stalled {args.stall_patience} iters, "
                  f"widening bound_mult -> {bound_mult:.1f}")

        with open(results_path, "a") as f:
            for r in all_results[-budget:]:
                f.write(json.dumps(r) + "\n")
        checkpoint_path.write_text(json.dumps({
            "outer": outer, "x_best": x_best.tolist(), "nfp": nfp, "lam": lam.tolist(),
            "rho": rho.tolist(), "best_worst": best_worst, "bound_mult": bound_mult,
            "stall_count": stall_count, "budget": budget, "prev_viol_clip": prev_viol_clip.tolist(),
        }))
        print(f"[ngopt_official] outer {outer}: {n_conv}/{len(asked)} converged, "
              f"best_worst_violation={best_worst:.4f}, rho={np.array2string(rho, precision=2)}, "
              f"bound_mult={bound_mult:.1f}, budget={budget}")

        if success_path.exists():
            print("[ngopt_official] success reached, stopping")
            break

    print(f"[ngopt_official] done, best_worst_violation={best_worst:.4f}")


if __name__ == "__main__":
    main()

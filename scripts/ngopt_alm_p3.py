"""Direct reproduction of the ConStellaration paper's own winning P3 recipe --
"ALM-NGOpt": Augmented Lagrangian outer loop wrapped around Nevergrad's NGOpt
gradient-free meta-optimizer, evaluated against REAL VMEC++ every single
iteration. No learned surrogate anywhere in this loop -- that's the entire
point. Every negative result this project has hit chasing P3 (staged-vs-free
push blind-spot exploitation, the 0/480 and then 0/160 generative wall-probe
failures) has the same shape: a frozen neural network standing in for real
physics near the boundary, where real training data is thinnest and the
network is most confidently wrong. NGOpt-against-real-VMEC sidesteps that
whole failure category by construction, at the cost of every evaluation
being a real (if low-fidelity) VMEC++ solve instead of a cheap forward pass.

Confirmed directly against the actual arxiv:2506.19583 paper and the
`constellaration` package we already have installed: the paper's own
`MHDStableQIStellarator` (constellaration.problems) uses exactly the same
5 P3 constraints/thresholds this project's grounded_walk lanes use, and its
own Table 5 baseline (ALM + Nevergrad's NGOpt, real VMEC++, aspect_ratio
fixed per run as a 6th constraint) reached normalized constraint violation
as low as 0.0 -- i.e. P3 IS solvable, just not yet by anything this project
has tried.

ALM formulation: standard Powell-Hestenes-Rockafellar augmented Lagrangian
for inequality constraints c_i(x) <= 0 (positive = violated), using this
project's own paper_feasibility_violation (optimize.py) for each c_i so the
qi log-transform and vacuum_well zero-threshold handling exactly match the
official benchmark code (see that function's own docstring):

    L(x) = -obj_weight * objective(x) + (1/(2*rho)) * sum_i
           ( relu(lambda_i + rho*c_i(x))**2 - lambda_i**2 )
    lambda_i <- max(0, lambda_i + rho * c_i(x_best_this_outer_iter))

Each outer ALM iteration runs a fresh NGOpt optimizer (nevergrad), warm-
started at the best point found by the previous iteration, for a small
ask/tell budget evaluated in parallel batches through the same real-VMEC
worker pool (oracle.worker_fn / run_batch_with_timeout) every other script
in this project uses. A non-converged (VMEC failure) candidate gets a large
fixed penalty rather than being dropped, so NGOpt learns to avoid that
region of coefficient space rather than just never seeing it scored.

Must run in the `train` service (needs nevergrad + vmecpp/constellaration in
the same process; oracle-x86 has no nevergrad).

Output: output/ngopt_alm_p3/{results.jsonl, state.json}. Writes
output/ngopt_alm_p3/success.json the moment a real point satisfies all 5 P3
constraints (relative_tol), same convention as grounded_walk_step_v4.py.
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
STATE_DIR = OUT_DIR / "ngopt_alm_p3"

# (target_name, op, value, use_abs, divisor, log_transform) -- exact P3 set,
# same convention/order as grounded_walk_step_v4.py's --constraint parsing.
P3_CONSTRAINTS = [
    ("edge_rotational_transform_over_n_field_periods", ">=", 0.25, True, None, False),
    ("qi", "<=", 10 ** -3.5, False, None, True),
    ("edge_magnetic_mirror_ratio", "<=", 0.25, False, None, False),
    ("flux_compression_in_regions_of_bad_curvature", "<=", 0.9, False, None, False),
    ("vacuum_well", ">=", 0.0, False, 0.1, False),
]

FAIL_PENALTY = 50.0  # per-constraint violation assigned to a non-converged candidate


def real_violations(measured):
    """measured: dict of target_name -> float. Returns (n_c,) array, positive
    = violated, via paper_feasibility_violation for exact-match semantics."""
    import torch
    out = []
    for name, op, value, use_abs, divisor, log_t in P3_CONSTRAINTS:
        col = torch.tensor([[measured[name]]], dtype=torch.float32)
        v = paper_feasibility_violation(col[:, 0], op, value, use_abs=use_abs, divisor=divisor, log_transform=log_t)
        out.append(float(v.item()))
    return np.array(out)


def alm_loss(violations, objective, obj_weight, lam, rho):
    """Standard PHR augmented Lagrangian penalty. violations: (n_c,) signed,
    positive = infeasible. objective: real scalar to MAXIMIZE."""
    penalty = np.sum(np.maximum(0.0, lam + rho * violations) ** 2 - lam ** 2) / (2 * rho)
    return -obj_weight * objective + penalty


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed-x", default=str(OUT_DIR / "hl_nearwall_X.npy"))
    p.add_argument("--seed-y", default=str(OUT_DIR / "hl_nearwall_Y.npy"))
    p.add_argument("--seed-row", type=int, default=0, help="row index into seed-x/-y sorted by worst-violation")
    p.add_argument("--n-outer", type=int, default=60)
    p.add_argument("--budget-per-outer", type=int, default=56)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--sigma0", type=float, default=0.0008,
                    help="initial NGOpt search std per coefficient. Confirmed by direct sigma scan on the "
                         "actual seed point: this near-wall real point sits in a narrow VMEC convergence "
                         "basin -- sigma=0.005 broke ALL 8 test candidates, sigma=0.002 broke 7/8, "
                         "sigma=0.001 converged 7/8, sigma=0.0005 converged 8/8. Near-wall real points are "
                         "themselves close to marginal/degenerate equilibria, not just close to the P3 "
                         "constraint boundary -- a real, useful finding on its own.")
    p.add_argument("--sigma-whitened", type=float, default=0.0, metavar="W",
                   help="per-coordinate mutation sigma in WHITENED units (sigma = W * "
                        "coef_scales), overriding the flat raw --sigma0. 0 keeps the "
                        "historical flat behaviour. Use this at --fidelity high: the "
                        "flat raw sigma gave 7/32 convergence at outer 0 because it is "
                        "a 17x-uneven perturbation in the units the physics responds "
                        "to. 0.02 is the single-coordinate stencil step that converges "
                        "100%% at high fidelity; note NGOpt moves all coordinates at "
                        "once, so total displacement is W*sqrt(81) = 9W whitened, and "
                        "the smoothing study measured ~3%% failure at W=0.02, 22%% at "
                        "0.05, 52%% at 0.10.")
    p.add_argument("--coef-scales", default=str(OUT_DIR / "coef_scales.npy"))
    p.add_argument("--rho0", type=float, default=1.0)
    p.add_argument("--rho-growth", type=float, default=1.15)
    p.add_argument("--obj-weight", type=float, default=0.02)
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stall-patience", type=int, default=12,
                    help="outer iterations with no best_worst improvement before widening the search box")
    p.add_argument("--stall-growth", type=float, default=1.6,
                    help="bound_width multiplier applied each time stall-patience is hit")
    p.add_argument("--max-bound-mult", type=float, default=60.0,
                    help="cap on bound_width as a multiple of sigma0, however much stalling occurs")
    args = p.parse_args()

    import nevergrad as ng

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"{args.tag}_results.jsonl"
    success_path = STATE_DIR / f"{args.tag}_success.json"
    checkpoint_path = STATE_DIR / f"{args.tag}_checkpoint.json"
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    # WHITENED SIGMA. A flat raw sigma is wrong whenever the coefficients have
    # different natural scales, and across the 81 free coefficients
    # `coef_scales.npy` spans 0.00023 to 0.1234 -- a **539x** range. A single raw
    # number therefore means a per-coordinate perturbation that differs by that
    # factor in the units the physics actually responds to: at --sigma0 0.0008 with
    # bound_mult 10 the box is 0.008 raw everywhere, which is 0.065 whitened units
    # on the largest-scale coefficient and **35** on the smallest, against flat
    # treads of 1e-3 to 1e-2 and a stencil step of 0.02 that converges 100%. So the
    # small-scale coefficients were being searched in a box ~3500 treads wide while
    # the large-scale ones got ~6. Measured consequence at high fidelity:
    # `ngopt_hf1` outer 0 converged 7/32. The low-fidelity calibration this came
    # from (sigma 0.005 broke 8/8, 0.001 converged 7/8, 0.0005 converged 8/8, see
    # --sigma0's help) was taken on 8 candidates at the WRONG fidelity, where the
    # convergence basin is much wider -- the same inherited-parameter error as
    # fd_step=0.02.
    #
    # And shrinking a flat sigma cannot fix it: holding the smallest-scale
    # coefficient to 0.02 whitened forces the largest to 0.0012, below one tread,
    # so it would stop moving at all. Scaling sigma BY coef_scales makes the
    # perturbation uniform in whitened units, which is the space the trust-region
    # and tread measurements are in.
    if args.sigma_whitened > 0.0:
        scales = np.load(args.coef_scales).astype(np.float64)
        sigma = args.sigma_whitened * scales
        _nz = scales[scales > 0]
        print(f"[ngopt_alm_p3] WHITENED sigma: {args.sigma_whitened:g} whitened units "
              f"-> raw {sigma[scales > 0].min():.3e}..{sigma[scales > 0].max():.3e} "
              f"across the 81 free coefficients")
        print(f"[ngopt_alm_p3] scales span {_nz.min():.3e}..{_nz.max():.3e} "
              f"({_nz.max()/_nz.min():.0f}x). A flat raw sigma uniform at this one's "
              f"largest value would be {args.sigma_whitened * _nz.max() / _nz.min():.1f} "
              f"whitened on the smallest-scale coefficient instead of "
              f"{args.sigma_whitened:g} -- i.e. {_nz.max()/_nz.min():.0f}x the "
              f"intended perturbation there.")
    else:
        sigma = np.full(90, args.sigma0)
    sigma[oracle.ZERO_INDICES] = 0.0  # never search the structurally-zero coefficients

    n_c = len(P3_CONSTRAINTS)
    start_outer = 0
    bound_mult = 10.0  # multiple of sigma0; grown on stall, capped at --max-bound-mult
    stall_count = 0

    if args.resume and checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text())
        x_best = np.array(ck["x_best"], dtype=np.float64)
        nfp = ck["nfp"]
        lam = np.array(ck["lam"], dtype=np.float64)
        rho = ck["rho"]
        best_worst = ck["best_worst"]
        start_outer = ck["outer"] + 1
        bound_mult = ck.get("bound_mult", bound_mult)
        stall_count = ck.get("stall_count", 0)
        print(f"[ngopt_alm_p3] resumed from checkpoint at outer={start_outer}, "
              f"best_worst={best_worst:.4f}, bound_mult={bound_mult:.1f}")
    else:
        X = np.load(args.seed_x)
        x0 = X[args.seed_row][:90].astype(np.float64)
        nfp = int(X[args.seed_row][90])
        print(f"[ngopt_alm_p3] seeded from row {args.seed_row}, nfp={nfp}")
        lam = np.zeros(n_c)
        rho = args.rho0
        x_best = x0.copy()
        best_worst = 1e9

    all_results = []
    tag_ctr = itertools.count()

    # Explicit bounds are required -- without them NGOpt's auto-selected
    # sub-optimizer falls back to a quasi-random opening batch that ignores
    # the per-dim mutation sigma entirely (confirmed directly: unbounded,
    # the first `ask()` batch perturbs coefficients by O(1), which VMEC++
    # can't even parse as valid geometry -- 0/8 converged in a smoke test).
    # Bound width starts as a generous multiple of sigma0 so NGOpt has real
    # room to move without repeating the same failure the other way (a box
    # so tight it can't escape a bad local seed). ngopt_v1 (see
    # EXPERIMENT_LOG) showed the tight box needed to survive VMEC near the
    # wall can itself become a trap: best_worst_violation went dead flat at
    # 0.0917 for the last 33/200 outer iterations while rho exploded to
    # 1.4e12, the classic ALM-stuck-at-a-local-point signature, not a budget
    # problem. --stall-patience/--stall-growth widen the box once that
    # pattern is detected, instead of grinding rho forever against a box
    # too small to contain any better point.

    for outer in range(start_outer, args.n_outer):
        # The box has to be whitened for the same reason sigma does -- a flat raw
        # bound is what made the smallest-scale coefficient's box 34.8 whitened
        # units wide. `sigma` already carries the per-coordinate scale, so deriving
        # the box from it keeps the two consistent when either is whitened.
        bound_width = np.maximum(sigma * bound_mult, 1e-6) if args.sigma_whitened > 0.0 \
            else np.maximum(args.sigma0 * bound_mult, 1e-6)
        lower = x_best - bound_width
        upper = x_best + bound_width
        # ZERO_INDICES must stay a real (if tiny) interval, not a single
        # point -- nevergrad rejects lower==upper -- but sigma=0 there
        # (set above) means NGOpt's own mutation never actually moves off
        # x_best=0 in these dims regardless.
        lower[oracle.ZERO_INDICES] = -1e-6
        upper[oracle.ZERO_INDICES] = 1e-6
        param = ng.p.Array(init=x_best, mutable_sigma=False).set_bounds(lower=lower, upper=upper)
        param.set_mutation(sigma=sigma)
        optimizer = ng.optimizers.registry["NGOpt"](parametrization=param, budget=args.budget_per_outer,
                                                      num_workers=args.n_workers)
        asked = []
        for _ in range(args.budget_per_outer):
            asked.append(optimizer.ask())

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
            if ok:
                y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
                if all(v is not None for v in y.values()):
                    viol = real_violations(y)
                    obj = y["minimum_normalized_magnetic_gradient_scale_length"]
                    loss = alm_loss(viol, obj, args.obj_weight, lam, rho)
                    worst = float(viol.max())
                    row.update({"converged": True, "measured_targets": y, "violations": viol.tolist(),
                                "worst_violation": worst, "objective": obj, "loss": loss})
                    scored[tag] = (loss, worst, np.array(cand.value, dtype=np.float64), viol)
                    if worst < best_worst:
                        best_worst = worst
                        x_best = np.array(cand.value, dtype=np.float64)
                        x_best[oracle.ZERO_INDICES] = 0.0
                    if worst <= args.relative_tol:
                        success_path.write_text(json.dumps({
                            "outer": outer, "measured_targets": y, "worst_violation": worst,
                            "params": np.array(cand.value, dtype=np.float64).tolist(), "nfp": nfp,
                        }, indent=2))
                        print(f"[ngopt_alm_p3] *** REAL P3 FEASIBILITY REACHED at outer={outer} *** "
                              f"worst_violation={worst:+.4f}")
            if not row["converged"]:
                loss = alm_loss(np.full(n_c, FAIL_PENALTY), 0.0, args.obj_weight, lam, rho)
                scored[tag] = (loss, 1e9, np.array(cand.value, dtype=np.float64), None)
            all_results.append(row)
            optimizer.tell(cand, loss if row["converged"] else alm_loss(np.full(n_c, FAIL_PENALTY), 0.0, args.obj_weight, lam, rho))

        n_conv = sum(1 for v in scored.values() if v[1] < 1e9)
        # ALM multiplier update at the best-found real point this outer iter
        if n_conv:
            best_tag = min(scored, key=lambda t: scored[t][0])
            _, worst_here, x_here, viol_here = scored[best_tag]
            if viol_here is not None:
                lam = np.maximum(0.0, lam + rho * viol_here)
        rho *= args.rho_growth

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
            print(f"[ngopt_alm_p3] outer {outer}: stalled {args.stall_patience} iters, "
                  f"widening bound_mult -> {bound_mult:.1f}")

        with open(results_path, "a") as f:
            for r in all_results[-args.budget_per_outer:]:
                f.write(json.dumps(r) + "\n")
        checkpoint_path.write_text(json.dumps({
            "outer": outer, "x_best": x_best.tolist(), "nfp": nfp, "lam": lam.tolist(),
            "rho": rho, "best_worst": best_worst, "bound_mult": bound_mult, "stall_count": stall_count,
        }))
        print(f"[ngopt_alm_p3] outer {outer}: {n_conv}/{args.budget_per_outer} converged, "
              f"best_worst_violation={best_worst:.4f}, rho={rho:.3f}, bound_mult={bound_mult:.1f}")

        if success_path.exists():
            print("[ngopt_alm_p3] success reached, stopping")
            break

    print(f"[ngopt_alm_p3] done, best_worst_violation={best_worst:.4f}")


if __name__ == "__main__":
    main()

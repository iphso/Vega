"""One instance of the paper's ACTUAL P3 decomposition: ALM-NGOpt against real
VMEC++ with `aspect_ratio <= A*` added as a sixth inequality constraint.

WHY THIS EXISTS (EXPERIMENT_LOG, 2026-09-26). `ngopt_alm_p3_paper.py` ran for
6 days / 680k real evals and stalled at worst_violation=0.204 -- against a
MIS-SPECIFIED problem. Verified directly against the installed package and the
paper:

  * `constellaration.problems.MHDStableQIStellarator` is MULTI-objective --
    minimize (-minimum_normalized_magnetic_gradient_scale_length, aspect_ratio)
    -- and is scored by the HYPERVOLUME of a SET of feasible designs. Its
    `evaluate()` signature takes a LIST of boundaries, and it calls
    `ConstellarationSettings.default_high_fidelity()` unconditionally.
  * The paper (arXiv:2506.19583) decomposes it "into multiple single-objective
    problems by treating the aspect ratio as an inequality constraint", with
    A* in {6, 8, 10, 12}, each "seeded from rotating ellipse configurations",
    15-24h per instance on 96 vCPU -- reaching constraint violations
    {0.104, 0.00130, 0.0, 0.00210}, i.e. three of four essentially feasible.

`ngopt_alm_p3_paper.py` had NO aspect-ratio constraint (leaving the search a
free direction to wander in), drove a single point rather than a front, and
seeded from an omnigenity fit rather than a rotating ellipse. This script fixes
all three. The ALM machinery itself is unchanged and still the paper's numbers
-- the method was never the problem.

SEEDS. Chosen empirically by `rotating_ellipse_probe.py` (nfp=2 dominates; the
higher nfp values blow flux_compression up by 3-15x). A*=6 is the awkward one:
VMEC++/QI rejects every elongation except 1.5, and only at nfp=3, with
"Minimum of B is at the boundary".

    A*= 6: elongation=1.5 nfp=3  -> seed worst_violation 1.7314
    A*= 8: elongation=2.0 nfp=2  -> seed worst_violation 0.6013
    A*=10: elongation=3.0 nfp=2  -> seed worst_violation 0.6402
    A*=12: elongation=4.0 nfp=2  -> seed worst_violation 1.2242

Note these seeds start FAR WORSE than paper_v1's stalled incumbent (0.204).
That is expected and is the point: the omnigenity anchor sat in a
better-looking basin that provably does not contain a feasible design, and 680k
evals confirmed it (the cloud never left a ball of radius 1.7% of the seed
norm, with an effective dimensionality of 2.4).

Search runs at low fidelity for throughput; finalists MUST be re-validated at
high fidelity, which reads ~0.017 worse on the active constraints -- see
`fidelity_check.py`. Launch all four via `run_p3_instances.sh`.

Must run in the `train` service. Output: output/ngopt_alm_p3_instance/
{<tag>_results.jsonl, <tag>_success.json, <tag>_checkpoint.json}.
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
STATE_DIR = OUT_DIR / "ngopt_alm_p3_instance"

BASE_CONSTRAINTS = [
    ("edge_rotational_transform_over_n_field_periods", ">=", 0.25, True, None, False),
    ("qi", "<=", 10 ** -3.5, False, None, True),
    ("edge_magnetic_mirror_ratio", "<=", 0.25, False, None, False),
    ("flux_compression_in_regions_of_bad_curvature", "<=", 0.9, False, None, False),
    ("vacuum_well", ">=", 0.0, False, 0.1, False),
]

# Per-instance seeds from rotating_ellipse_probe.py: (elongation, nfp).
SEEDS = {6.0: (1.2, 2), 8.0: (2.0, 2), 10.0: (3.0, 2), 12.0: (4.0, 2)}

# A* <= 5 has NO viable rotating-ellipse seed at all -- every elongation in
# 1.2-2.0 x every nfp in 2-4 dies with VMEC++/QI "Minimum of B is at the
# boundary". Reaching low aspect ratio therefore needs CONTINUATION: get a
# feasible design at an A* where seeds do work, then warm-start the next
# instance one step lower via --seed-npy. That is also the region where the
# paper's own baseline failed (violation 0.104 at its hardest instance), and
# where hypervolume pays most -- a point's contribution is
# (1 + L_gradB) * (20 - A).

# Populated in main() once A* is known -- the paper's decomposition adds
# aspect_ratio as the sixth constraint, so the vector length is 6 here, not 5.
P3_CONSTRAINTS = []

FAIL_PENALTY = 50.0


def real_violations(measured):
    import torch
    out = []
    for name, op, value, use_abs, divisor, log_t in P3_CONSTRAINTS:
        col = torch.tensor([[measured[name]]], dtype=torch.float32)
        v = paper_feasibility_violation(col[:, 0], op, value, use_abs=use_abs, divisor=divisor, log_transform=log_t)
        out.append(float(v.item()))
    return np.array(out)


def rotating_ellipse_params(aspect_ratio, elongation, iota_over_nfp, nfp):
    """The paper's seed ("seeded from rotating ellipse configurations"),
    converted to this project's 90-dim row layout. Shared definition with
    rotating_ellipse_probe.py -- see that script for the viability sweep that
    picked the per-A* elongation/nfp in SEEDS."""
    from constellaration import initial_guess
    from constellaration.mhd import near_axis_configuration

    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=aspect_ratio,
        elongation=elongation,
        # takes EDGE TOTAL iota and divides by nfp internally; P3 constrains
        # iota/nfp, so scale up here.
        rotational_transform=iota_over_nfp * nfp,
        n_field_periods=nfp,
    )
    boundary = near_axis_configuration.smooth_and_set_max_mode_numbers(boundary, 4, 4)
    r_cos = np.asarray(boundary.r_cos, dtype=np.float64)
    z_sin = np.asarray(boundary.z_sin, dtype=np.float64)
    if r_cos.shape != (5, 9) or z_sin.shape != (5, 9):
        raise ValueError(f"unexpected shape r_cos={r_cos.shape} z_sin={z_sin.shape}")
    params = np.concatenate([r_cos.flatten(), z_sin.flatten()])
    params[oracle.ZERO_INDICES] = 0.0
    return params


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
    p.add_argument("--aspect-ratio-max", type=float, required=True,
                   help="A*, the paper's decomposition lever: aspect_ratio <= A* becomes "
                        "the sixth inequality constraint. Paper uses {6, 8, 10, 12}.")
    p.add_argument("--elongation", type=float, default=None,
                   help="rotating-ellipse seed elongation; default: per-A* value from "
                        "rotating_ellipse_probe.py (see SEEDS)")
    p.add_argument("--nfp", type=int, default=None,
                   help="rotating-ellipse seed field periods; default: per-A* value from SEEDS")
    p.add_argument("--seed-iota-over-nfp", type=float, default=0.28,
                   help="rotating-ellipse seed iota/nfp; over the 0.25 lower bound")
    p.add_argument("--seed-npy", default=None,
                   help="CONTINUATION: warm-start from a saved 90-dim design instead of a "
                        "rotating ellipse. Use to walk aspect ratio downward -- seed A*=N from "
                        "the feasible design found at A*=N+1 or N+2. Required below A*=6, where "
                        "no rotating ellipse survives VMEC++/QI at all.")
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--seed-nfp", type=int, default=None,
                   help="nfp for --seed-npy (default: column 90 if present, else 2)")
    p.add_argument("--whiten", action="store_true", default=True,
                   help="B1: scale per-coefficient mutation sigma AND trust-region half-width by "
                        "that coefficient's std over the real 158,685-design dataset "
                        "(output/coef_scales.npy). Without this the search is isotropic over a "
                        "basis whose natural scales span 540x, so one flat sigma is simultaneously "
                        "~13x WIDER than the entire natural spread of the tightest coefficients "
                        "(destructive noise) and ~3% of the widest (barely moves them). That is the "
                        "direct cause of paper_v1's cloud collapsing to 2.4 effective dimensions.")
    p.add_argument("--no-whiten", dest="whiten", action="store_false")
    p.add_argument("--coef-scales", default=str(OUT_DIR / "coef_scales.npy"))
    p.add_argument("--feasible-tol", type=float, default=1e-2,
                   help="the benchmark's own _DEFAULT_RELATIVE_TOLERANCE: is_feasible() requires "
                        "ALL five normalized violations <= this")
    p.add_argument("--obj-weight-feasible", type=float, default=2.0,
                   help="PHASE 2 objective weight, applied once a feasible design exists. The score "
                        "is hypervolume of (-L_gradB, aspect_ratio) vs ref (1.0, 20.0), so "
                        "L_gradB is HALF the score: +3 in L_gradB is worth ~+27% hypervolume, "
                        "comparable to adding two whole feasible designs. With the phase-1 weight "
                        "(0.02) the ALM loss is ~98% feasibility and we never optimize the axis we "
                        "are scored on.")
    p.add_argument("--n-outer", type=int, default=100)
    p.add_argument("--budget-initial", type=int, default=1500, help="paper's own schedule: min(20000, 1500+300k)")
    p.add_argument("--budget-increment", type=int, default=300)
    p.add_argument("--budget-max", type=int, default=20000)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--sigma0", type=float, default=0.004,
                   help="5x ngopt_alm_p3_paper.py's 0.0008. That value was sized to POLISH a "
                        "near-feasible omnigenity seed; these rotating-ellipse seeds start at "
                        "worst_violation 0.60-1.73 and have only 7 of 81 free coefficients "
                        "nonzero (mpol=1 shape zero-padded into the mpol=4/ntor=4 space), so "
                        "the search has to grow higher-order modes up from exactly zero rather "
                        "than nudge existing ones.")
    p.add_argument("--rho0", type=float, default=10.0, help="paper's own rho_i^0=10")
    p.add_argument("--rho-increase-factor", type=float, default=5.0, help="paper's own sigma=5")
    p.add_argument("--rho-max", type=float, default=1e8, help="paper's own rho_max=1e8")
    p.add_argument("--tol-reduction-factor", type=float, default=0.8,
                    help="paper's own tau=0.8 -- rho_i only grows if max(0,c_i) > this * max(0,c_i_prev)")
    p.add_argument("--obj-weight", type=float, default=0.02)
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bound-mult-initial", type=float, default=25.0,
                    help="starting trust-region half-width as a multiple of sigma0 (analogue of paper's delta_0, "
                         "not numerically comparable since we don't reparametrize by the infinity-norm spectral "
                         "scale -- picked to match this project's own previously-proven generous starting box)")
    p.add_argument("--bounds-shrink", type=float, default=0.95, help="paper's own gamma=0.95, applied every outer")
    p.add_argument("--bound-mult-min", type=float, default=5.0,
                    help="floor for bound_mult, keeping the paper's delta_min/delta_0 = 0.05/0.33 ratio: "
                         "25*(0.05/0.33) ~= 3.8, rounded up to 5. With sigma0=0.004 the trust-region "
                         "half-width therefore runs 0.100 -> 0.020, versus paper_v1's 0.048 -> 0.0072 -- "
                         "paper_v1 pinned at its floor by outer 36 and its whole 680k-eval cloud never "
                         "left a ball of radius 0.018, which is the failure this widening targets.")
    p.add_argument("--stall-patience", type=int, default=10 ** 9,
                    help="disabled by default -- the paper's recipe monotonically shrinks bounds rather than "
                         "reactively widening on stall; set finite to re-enable this project's own fallback")
    p.add_argument("--stall-growth", type=float, default=1.6)
    p.add_argument("--max-bound-mult", type=float, default=25.0)
    args = p.parse_args()

    import nevergrad as ng

    # The paper's decomposition: five official P3 constraints + aspect_ratio.
    # Set before n_c/real_violations are used, since both read this global.
    global P3_CONSTRAINTS
    P3_CONSTRAINTS = BASE_CONSTRAINTS + [
        ("aspect_ratio", "<=", args.aspect_ratio_max, False, None, False),
    ]

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"{args.tag}_results.jsonl"
    success_path = STATE_DIR / f"{args.tag}_success.json"
    checkpoint_path = STATE_DIR / f"{args.tag}_checkpoint.json"
    # Shared across ALL instances on purpose: the benchmark scores the
    # hypervolume of one combined set of feasible designs, so every feasible
    # point any instance finds belongs to the same pool. p3_front.py reads this.
    pool_path = STATE_DIR / "feasible_pool.jsonl"
    fidelity_name = oracle.FIDELITY_PRESETS[args.fidelity]

    free_idx = [i for i in range(90) if i not in oracle.ZERO_INDICES]
    if args.whiten:
        scales = np.load(args.coef_scales).astype(np.float64)
        # Normalize by the MEDIAN free scale so --sigma0 keeps its absolute
        # "typical step" meaning (and its already-validated 0.004 calibration);
        # whitening then only REDISTRIBUTES that step across coefficients
        # according to how much each one actually varies in the real dataset.
        sigma = args.sigma0 * scales / np.median(scales[free_idx])
        print(f"[p3_inst] whitened search: per-coefficient sigma spans "
              f"{sigma[free_idx].min():.2e} .. {sigma[free_idx].max():.2e} "
              f"(median {np.median(sigma[free_idx]):.2e}), vs flat {args.sigma0:.2e}")
    else:
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
        print(f"[p3_inst] resumed from checkpoint at outer={start_outer}, "
              f"best_worst={best_worst:.4f}, bound_mult={bound_mult:.1f}, budget={budget}")
    elif args.seed_npy:
        X = np.load(args.seed_npy)
        X = X if X.ndim == 2 else X[None, :]
        x0 = X[args.seed_row][:90].astype(np.float64)
        x0[oracle.ZERO_INDICES] = 0.0
        if args.seed_nfp is not None:
            nfp = args.seed_nfp
        elif X.shape[1] > 90:
            nfp = int(X[args.seed_row][90])
        else:
            nfp = 2
        print(f"[p3_inst] A*={args.aspect_ratio_max} CONTINUATION warm-start from "
              f"{args.seed_npy} row {args.seed_row}, nfp={nfp}")
    else:
        default_elong, default_nfp = SEEDS.get(args.aspect_ratio_max, (2.0, 2))
        elongation = args.elongation if args.elongation is not None else default_elong
        nfp = args.nfp if args.nfp is not None else default_nfp
        x0 = rotating_ellipse_params(
            args.aspect_ratio_max, elongation, args.seed_iota_over_nfp, nfp
        )
        print(f"[p3_inst] A*={args.aspect_ratio_max} seeded from rotating ellipse "
              f"(elongation={elongation}, nfp={nfp}, iota/nfp={args.seed_iota_over_nfp})")
        lam = np.zeros(n_c)
        rho = np.full(n_c, args.rho0)
        x_best = x0.copy()
        best_worst = 1e9

    all_results = []
    tag_ctr = itertools.count()
    n_feasible = 0  # designs meeting the benchmark's own is_feasible() bar

    for outer in range(start_outer, args.n_outer):
        # PHASE 2: once a feasible design exists, stop spending ~98% of the
        # loss on feasibility. L_gradB is one of the two scored axes and is
        # otherwise never optimized; the ALM penalty still holds feasibility.
        obj_weight = args.obj_weight_feasible if n_feasible else args.obj_weight
        # Per-coefficient when whitened: the trust region becomes an axis-aligned
        # box whose half-widths track each coefficient's real variability,
        # instead of a 90-dim cube that is far too wide in some directions and
        # far too narrow in others.
        bound_width = np.maximum(sigma * bound_mult, 1e-6)
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
            fail_loss = alm_loss(np.full(n_c, FAIL_PENALTY), 0.0, obj_weight, lam, rho)
            if ok:
                y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
                if all(v is not None for v in y.values()):
                    viol = real_violations(y)
                    obj = y["minimum_normalized_magnetic_gradient_scale_length"]
                    loss = alm_loss(viol, obj, obj_weight, lam, rho)
                    worst = float(viol.max())
                    row.update({"converged": True, "measured_targets": y, "violations": viol.tolist(),
                                "worst_violation": worst, "objective": obj, "loss": loss})
                    scored[tag] = (loss, worst, cand_params, viol)
                    if worst < best_worst:
                        best_worst = worst
                        x_best = cand_params.copy()
                    # Harvest: the five OFFICIAL constraints only (viol[:5]) --
                    # aspect_ratio is objective #2 in the real problem, not a
                    # constraint; A* is just this decomposition's own lever, so
                    # a design outside its A* box can still be a perfectly
                    # legitimate front member.
                    if float(viol[:5].max()) <= args.feasible_tol:
                        n_feasible += 1
                        with open(pool_path, "a") as pf:
                            pf.write(json.dumps({
                                "tag": args.tag, "outer": outer, "nfp": nfp,
                                "aspect_ratio_max": args.aspect_ratio_max,
                                "params": cand_params.tolist(),
                                "measured_targets": y,
                                "official_worst_violation": float(viol[:5].max()),
                                "L_gradB": y["minimum_normalized_magnetic_gradient_scale_length"],
                                "aspect_ratio": y["aspect_ratio"],
                            }) + "\n")
                        if n_feasible == 1:
                            print(f"[p3_inst] *** FIRST FEASIBLE DESIGN *** outer={outer} "
                                  f"L_gradB={y['minimum_normalized_magnetic_gradient_scale_length']:.3f} "
                                  f"aspect_ratio={y['aspect_ratio']:.3f} "
                                  f"-- switching to phase 2 (obj_weight {args.obj_weight} -> "
                                  f"{args.obj_weight_feasible})")
                    if worst <= args.relative_tol:
                        success_path.write_text(json.dumps({
                            "outer": outer, "measured_targets": y, "worst_violation": worst,
                            "params": cand_params.tolist(), "nfp": nfp,
                        }, indent=2))
                        print(f"[p3_inst] *** REAL P3 FEASIBILITY REACHED at outer={outer} *** "
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
            print(f"[p3_inst] outer {outer}: stalled {args.stall_patience} iters, "
                  f"widening bound_mult -> {bound_mult:.1f}")

        with open(results_path, "a") as f:
            for r in all_results[-len(asked):]:
                f.write(json.dumps(r) + "\n")
        checkpoint_path.write_text(json.dumps({
            "outer": outer, "x_best": x_best.tolist(), "nfp": nfp, "lam": lam.tolist(),
            "rho": rho.tolist(), "best_worst": best_worst, "bound_mult": bound_mult,
            "stall_count": stall_count, "budget": budget, "prev_viol_clip": prev_viol_clip.tolist(),
            "n_feasible": n_feasible,
        }))
        print(f"[p3_inst] outer {outer}: {n_conv}/{len(asked)} converged, "
              f"best_worst_violation={best_worst:.4f}, rho={np.array2string(rho, precision=2)}, "
              f"bound_mult={bound_mult:.1f}, budget={budget}, "
              f"feasible={n_feasible}, phase={2 if n_feasible else 1}")

        if success_path.exists():
            print("[p3_inst] success reached, stopping")
            break

    print(f"[p3_inst] done, best_worst_violation={best_worst:.4f}")


if __name__ == "__main__":
    main()

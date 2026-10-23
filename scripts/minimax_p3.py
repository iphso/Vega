"""Minimax SQP for P3 feasibility: attack max_i v_i(x) directly via a linear
program, instead of descending a scalarized surrogate.

WHY (EXPERIMENT_LOG, 2026-09-27). `lbfgs_refine_p3.py` was run to a genuine
stall after three configuration bugs were found and fixed:

  1. fd_step 1e-3 was 20x too small -- `fd_diagnostic.py` measured the usable
     plateau at h = 1e-2..5e-2 (consecutive-gradient cosine 0.91), while 1e-3 vs
     1e-2 agreed only at 0.53 and 1e-4 vs 1e-3 was NEGATIVE (-0.14).
  2. sum-of-squares was the wrong surrogate: it drove f down (2.6165e-2 ->
     2.4602e-2) while worst_violation went the WRONG way (0.09653 -> 0.11186).
  3. a 60s batch timeout silently zeroed gradient components -- the identical
     deterministic base point gave fd_fail 0/81 and 7/81 in two runs. Fixed at
     300s; confirmed 0/81 thereafter.

With all three fixed and a verifiably clean gradient, p-norm (p=8) L-BFGS still
stalled: 13 gradients, worst 0.09653 -> 0.09718, f flat then rising, |g| ~1.5.

THE DIAGNOSIS. At the best recoverable design the violations are
  iota -0.1905 | qi +0.0963 | mirror +0.0965 | fluxcomp +0.0871 | vwell -0.0977
Three are active and nearly TIED, and all three must fall ~10x to reach the
1e-2 feasibility bar. Any scalarization (sum of squares, p-norm, weighted sum)
picks one compromise weighting and descends that. When the active constraints'
gradients conflict pairwise, the scalarized descent direction stalls at a point
that is not a minimum of the max -- it is just a minimum of that particular
scalarization. This is the textbook failure mode that minimax methods exist for.

THE FORMULATION. At each iterate solve the linear program

    min_{d, t}  t
    s.t.        v_i + g_i . d  <=  t        for each of the 5 constraints
                -Delta <= d_j <= Delta      (trust region, whitened units)

d is then the step that reduces the WORST violation as much as the local linear
model allows, correctly trading off conflicting constraints instead of
averaging them. Classic Zoutendijk / SLP minimax step.

FREE INFORMATION. The 82-evaluation stencil already computes all eleven metrics
at every perturbed point, so all FIVE constraint gradients come out of the same
oracle batch that previously produced one scalar gradient. This is strictly more
information per VMEC call, at identical cost.

Trust region adapts on the ACTUAL measured max violation (not the linear
model): grow on success, shrink on failure, terminate when Delta collapses.

Whitened coordinates and the deterministic low-fidelity oracle as before; see
`lbfgs_refine_p3.py` for that rationale.

Must run in the `train` service. Output: output/minimax_p3/
{<tag>_trace.jsonl, <tag>_best.npy}; feasible designs are appended to
output/ngopt_alm_p3_instance/feasible_pool.jsonl for p3_front.py.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "minimax_p3"
POOL_PATH = OUT_DIR / "ngopt_alm_p3_instance" / "feasible_pool.jsonl"

FEASIBLE_TOL = 1e-2
REF_POINT = (1.0, 20.0)
CNAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]


def official_violations(m):
    """problems._normalized_constraint_violations, transcribed exactly."""
    iota = abs(m["edge_rotational_transform_over_n_field_periods"])
    return np.array([
        (0.25 - iota) / 0.25,
        (np.log10(m["qi"]) + 3.5) / 3.5,
        (m["edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (m["flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        (0.0 - m["vacuum_well"]) / 1e-1,
    ])


def hv_contribution(m):
    return ((REF_POINT[0] + m["minimum_normalized_magnetic_gradient_scale_length"])
            * (REF_POINT[1] - m["aspect_ratio"]))


# ---------------------------------------------------------------------------
# ASPECT-RATIO FLOOR: a SIXTH pseudo-constraint, used for SEARCH ONLY.
#
# WHY. At the project leader (ex_floor2 lineage, worst=0.02787) the five official
# violations are iota -0.136, qi +0.0287, mirror +0.0286, fluxcomp +0.0113,
# vwell -0.0899. The two binding constraints are qi and mirror, equalised by the
# minimax to within 5e-5, and the two with the most slack are iota and vwell.
# Measured over 36,152 evals with worst<=0.05 (A in 12.17-13.10, i.e. the leader's
# own neighbourhood):
#     spearman(A, qi violation)     = -0.669     spearman(A, iota)  = +0.472
#     spearman(A, mirror violation) = -0.091     spearman(A, fluxc) = +0.546
#     spearman(A, WORST violation)  = -0.650     spearman(A, vwell) = +0.708
# Raising A improves exactly the two constraints that bind and degrades exactly
# the three that have 9-14% of unused margin. The same sign holds at every
# quality cut (worst<=1.0: -0.641; <=0.2: -0.421; <=0.1: -0.662).
#
# WHY WE HAVE NEVER DONE IT. Aspect ratio is an OBJECTIVE in P3, not a
# constraint, so the minimax never controls it -- A floats wherever the descent
# leaves it. Every design we own with worst<=0.05 sits in A in [12.17, 13.10],
# out of 991,855 evaluated. The high-A region is unexplored by construction, not
# by evidence.
#
# WHY IT IS WORTH HYPERVOLUME. The benchmark scores hypervolume of FEASIBLE
# designs and returns exactly 0.0 when none are feasible, which is our score
# after ~1M evaluations. A feasible design at A=14 contributes (1+L)(20-14) and
# beats 0.0; an infeasible design at A=12.9 contributes nothing. Buying
# feasibility with aspect ratio is therefore strictly correct while the feasible
# set is empty, and hv can be pushed back down in A afterwards from a feasible
# starting point.
#
# WHY IT IS SEPARATE FROM official_violations(). The official 5-vector stays
# exactly as the scorer defines it: it is what gets persisted as
# "worst_violation" on every eval and what every harvest and cross-run
# comparison in this project reads. Corrupting it would silently invalidate the
# ~1M rows already on disk. The 6-vector drives the LP, the Jacobian and
# candidate selection only.
def search_violations(m, aspect_min=0.0):
    v = official_violations(m)
    if aspect_min > 0.0:
        v = np.append(v, (aspect_min - m["aspect_ratio"]) / aspect_min)
    return v


class Evaluator:
    def __init__(self, args, x_ref, scales, free, nfp, results_path):
        self.a, self.x_ref, self.scales, self.free, self.nfp = args, x_ref, scales, free, nfp
        self.fid = oracle.FIDELITY_PRESETS[args.fidelity]
        self.results_path = results_path
        self.cur_iter = -1
        self.n_evals = 0
        self.n_feasible = 0
        self._t = itertools.count()

    def z_to_x(self, z):
        x = self.x_ref.copy()
        x[self.free] = z * self.scales[self.free]
        x[oracle.ZERO_INDICES] = 0.0
        return x

    def __call__(self, zs, kind="eval"):
        """Every evaluation is PERSISTED. An earlier version of this script
        discarded them -- ~50,550 real VMEC++ evaluations across 22 runs kept
        only a per-iteration trace summary, repeating exactly the
        ngopt_v1 data loss (that run stored no `params`, so the project's
        all-time best design at 0.0917 is unrecoverable). Boundaries plus their
        eleven physics metrics are the durable asset here; the optimizer's path
        is incidental."""
        jobs, order = [], {}
        for i, z in enumerate(zs):
            rc, zsin, nfp, fn = oracle.params_to_worker_args(
                self.z_to_x(z), {"nfp": self.nfp}, self.fid)
            t = next(self._t)
            jobs.append((t, rc, zsin, nfp, fn)); order[t] = i
        out = [None] * len(zs)
        rows = []
        for t, ok, payload in run_batch_with_timeout(
            jobs, oracle.worker_fn, self.a.n_workers, self.a.timeout_seconds
        ):
            self.n_evals += 1
            i = order[t]
            if not ok:
                rows.append({"tag": self.a.tag, "iter": self.cur_iter, "kind": kind,
                             "converged": False, "nfp": self.nfp,
                             "params": self.z_to_x(zs[i]).tolist()})
                continue
            m = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if any(v is None for v in m.values()):
                rows.append({"tag": self.a.tag, "iter": self.cur_iter, "kind": kind,
                             "converged": False, "nfp": self.nfp,
                             "params": self.z_to_x(zs[i]).tolist()})
                continue
            out[i] = m
            v = official_violations(m)
            rows.append({"tag": self.a.tag, "iter": self.cur_iter, "kind": kind,
                         "converged": True, "nfp": self.nfp,
                         "params": self.z_to_x(zs[i]).tolist(),
                         "measured_targets": m, "violations": v.tolist(),
                         "worst_violation": float(v.max()),
                         "hv_contribution": hv_contribution(m)})
            if float(v.max()) <= FEASIBLE_TOL:
                self._feasible(zs[i], m)
        with open(self.results_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return out

    def _feasible(self, z, m):
        self.n_feasible += 1
        w = float(official_violations(m).max())
        with open(POOL_PATH, "a") as f:
            f.write(json.dumps({
                "tag": self.a.tag, "source": "minimax_p3", "nfp": self.nfp,
                "params": self.z_to_x(z).tolist(), "measured_targets": m,
                "official_worst_violation": w,
                "L_gradB": m["minimum_normalized_magnetic_gradient_scale_length"],
                "aspect_ratio": m["aspect_ratio"],
            }) + "\n")
        if self.n_feasible == 1:
            print(f"[minimax] *** FIRST FEASIBLE DESIGN *** worst={w:.5f} "
                  f"L_gradB={m['minimum_normalized_magnetic_gradient_scale_length']:.3f} "
                  f"A={m['aspect_ratio']:.3f} hv={hv_contribution(m):.2f}", flush=True)


def constraint_jacobian(ev, z, args):
    """All five constraint gradients from one 1+n stencil.

    Components whose perturbed solve failed are dropped from the LP's view by
    zeroing that column: a failed solve carries no slope information, and
    letting a fabricated value in would produce a spurious direction.
    """
    n = len(z)
    E = np.eye(n)
    batch = [z] + [z + args.fd_step * E[i] for i in range(n)]
    ms = ev(batch, kind="stencil")
    m0 = ms[0]
    if m0 is None:
        return None, None, None, 0, None, None
    v0 = search_violations(m0, args.aspect_min)
    hv0 = hv_contribution(m0)
    J = np.zeros((len(v0), n))
    g_hv = np.zeros(n)   # d(hypervolume contribution)/d(coefficient), free
    failed = []
    for i in range(n):
        mi = ms[i + 1]
        if mi is None:
            failed.append(i)
            continue
        J[:, i] = (search_violations(mi, args.aspect_min) - v0) / args.fd_step
        g_hv[i] = (hv_contribution(mi) - hv0) / args.fd_step

    # RETRY failed coordinates at a reduced step. The failures are not random:
    # they concentrate in a contiguous block of HIGH-ORDER poloidal modes
    # (r_cos m=3,4), where a full fd_step produces a boundary sharp enough that
    # VMEC++ cannot find an equilibrium. Observed at deep32 convergence:
    # 11/81 dead, indices 26-33 contiguous plus 52,55,81. Zeroing them leaves
    # the LP searching an effectively 70-dimensional subspace, which plausibly
    # limits how far the descent can go.
    for shrink in args.fd_retry:
        if not failed:
            break
        h = args.fd_step * shrink
        retry = [z + h * E[i] for i in failed]
        rms = ev(retry, kind="stencil_retry")
        still = []
        for i, mi in zip(failed, rms):
            if mi is None:
                still.append(i)
            else:
                J[:, i] = (search_violations(mi, args.aspect_min) - v0) / h
                g_hv[i] = (hv_contribution(mi) - hv0) / h
        failed = still
    return v0, J, m0, len(failed), g_hv, hv0


def minimax_step(v, J, delta_l2, g_hv=None, hv0=None, mode="none",
                 hv_floor=None, hv_weight=0.0):
    """LP:  min t  s.t.  v_i + J_i.d <= t,  |d|_inf <= delta_l2/sqrt(n).

    delta is an L2 BUDGET, converted to the per-coordinate box the LP needs.
    The LP saturates every coordinate, so a box of b over n coordinates yields
    ||d||_2 ~ b*sqrt(n); bounding the box directly therefore lets the TOTAL
    displacement run sqrt(81)=9x larger than intended. Measured at box=0.3:
    ||d||_2=2.65, i.e. 130x fd_step, so the LP was optimizing the linear model
    hundreds of intervals beyond where it was measured -- producing a direction
    that only became useful after the line search shrank it ~700x.
    Variables ordered [d (n), t (1)]."""
    from scipy.optimize import linprog
    n = J.shape[1]
    delta = delta_l2 / np.sqrt(n)
    c = np.zeros(n + 1); c[-1] = 1.0
    A = np.hstack([J, -np.ones((J.shape[0], 1))])   # rows = 5, or 6 with --aspect-min
    b = -v

    # Phase 1 descends max_i v_i with BOTH objectives unconstrained, so the
    # solver freely trades them away to make the physics easier -- measured
    # directly: deep32 halved the violation (0.0678 -> 0.0346) while driving
    # L_gradB 2.94 -> 1.17 and aspect ratio 11.77 -> 12.86, cutting the
    # design's hypervolume contribution 32.4 -> 15.5. Since the benchmark
    # scores hypervolume and not violation, that is progress toward a prize
    # worth half as much.
    if mode == "floor" and g_hv is not None:
        # hv0 + g_hv.d >= hv_floor   <=>   -g_hv.d <= hv0 - hv_floor
        A = np.vstack([A, np.concatenate([-g_hv, [0.0]])])
        b = np.concatenate([b, [hv0 - hv_floor]])
    elif mode == "weighted" and g_hv is not None:
        # trade a little violation progress for hv progress, in the objective
        c[:n] = -hv_weight * g_hv
    bounds = [(-delta, delta)] * n + [(None, None)]
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        return None, None
    return res.x[:n], res.x[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed-npy", default=str(OUT_DIR / "lbfgs_seeds.npy"))
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--coef-scales", default=str(OUT_DIR / "coef_scales.npy"))
    p.add_argument("--fd-step", type=float, default=2e-2,
                   help="measured plateau, see fd_diagnostic.py")
    p.add_argument("--hv-mode", default="none", choices=["none", "floor", "weighted"],
                   help="how phase 1 treats the two SCORED objectives. 'none': ignore them "
                        "(they degrade freely). 'floor': add an LP row holding "
                        "(1+L_gradB)*(20-A) at or above --hv-floor-frac of its starting value. "
                        "'weighted': add -w*g_hv to the LP objective -- MIS-SCALED and kept only "
                        "for the record; t is on the violation scale (~0.07) while hv is ~32, so "
                        "any useful w lets the hv term dominate and the run stalls immediately.\n"
                        "DEFAULT IS 'none'. The floor was briefly made default on a single A/B "
                        "(basin s32: none 0.050484 -> floor 0.032098, -36%%) and that DID NOT "
                        "REPLICATE. Paired arms across 6 basins, identical seed/budget/delta:\n"
                        "    basin L_gradB0   none      floor     floor vs none\n"
                        "      32     2.94  0.050484  0.032098      -36.4%%\n"
                        "      13     5.25  0.097839  0.100747       +3.0%%\n"
                        "      21     6.88  0.083593  0.084535       +1.1%%\n"
                        "      37     7.27  0.086448  0.086448        0.0%%\n"
                        "       4     7.70  0.214788  0.209144       -2.6%%\n"
                        "       9     9.92  0.086895  0.092689       +6.7%%\n"
                        "-> 2 wins, 1 tie, 3 losses; median effect slightly NEGATIVE. s32 is an\n"
                        "outlier and its -36%% is unexplained. A proposed mechanism (that the\n"
                        "unconstrained arm catastrophically degrades hv, so the floor prevents it)\n"
                        "was based on conflating two different runs: the -52%% hv collapse came from\n"
                        "`deep32`, not from this experiment's `hv_none` arm, which degraded only\n"
                        "-8.7%% -- comparable to every other basin.\n"
                        "WHAT DOES REPLICATE: floor protects hv slightly and reliably (ahead on hv\n"
                        "in 5 of 6 basins, +0.1%% to +12%%) at a violation cost of 0-6.7%%. A small,\n"
                        "symmetric trade-off -- not free upside. Use it deliberately when preserving\n"
                        "L_gradB matters more than the last few percent of feasibility progress.")
    p.add_argument("--hv-floor-frac", type=float, default=1.0)
    p.add_argument("--hv-weight", type=float, default=0.02)
    p.add_argument("--nullspace-hv", type=float, default=0.0, metavar="FRAC",
                   help="add a hypervolume-ascent component projected into the NULL SPACE of the\n"
                        "constraint Jacobian, with norm FRAC*||d_LP||. 0.0 (default) = off.\n"
                        "\n"
                        "WHY THIS IS NOT ANOTHER hv-mode VARIANT. --hv-mode floor/weighted both\n"
                        "make hv compete with feasibility inside the LP: floor as an extra row,\n"
                        "weighted as an objective term. Measured, floor gives a small symmetric\n"
                        "trade-off (2 wins/1 tie/3 losses over 6 paired basins) and weighted stalls\n"
                        "outright on a units mismatch. Both treat hv and the violations as rivals\n"
                        "for the same directions. The measured geometry says they largely are not.\n"
                        "\n"
                        "Pooling 1171 saved constraint Jacobians and 106,622 objective finite\n"
                        "differences recovered from the stencil evals: the constraint map has a\n"
                        "participation ratio of 4.2 effective dimensions (top 10 of 81 directions\n"
                        "= 89.8%% of all constraint sensitivity) and is dominated by m=1 modes --\n"
                        "dir1 alone is 43.5%% and is purely r_cos[m=1,n=-1..+2]. But the 35\n"
                        "directions that are constraint-NULL (<10%% of peak sensitivity) still\n"
                        "carry 41%% of the hv sensitivity and 38%% of the L_gradB sensitivity,\n"
                        "while carrying only 0.6%% of the aspect-ratio sensitivity. Spearman\n"
                        "between constraint sensitivity and hv sensitivity is just +0.579, and\n"
                        "z_sin[m=2,n=+1] / z_sin[m=3,n=-1] rank top-5 on hv while sitting near the\n"
                        "constraint null space.\n"
                        "\n"
                        "So there is a subspace in which hv can be bought at near-zero first-order\n"
                        "violation cost, and almost no aspect-ratio cost -- which is exactly the\n"
                        "trade-off that capped the leader (floor2 reached worst=0.0313 by pushing\n"
                        "A to 12.885, leaving hv at 33.2 against rp9's 89.6).\n"
                        "\n"
                        "MECHANISM. d = d_LP + FRAC*||d_LP|| * P g_hv/||P g_hv||, where\n"
                        "P = I - J^+ J. Since J P = 0, the added component changes every predicted\n"
                        "violation by exactly zero at first order, so it rides the EXISTING line\n"
                        "search and costs no extra evaluations. Second-order error is handled the\n"
                        "same way it already is for d_LP: the ladder shrinks alpha until the\n"
                        "MEASURED worst violation improves.")
    p.add_argument("--aspect-min", type=float, default=0.0, metavar="A",
                   help=(
    "Add a SIXTH pseudo-constraint (A_MIN - aspect_ratio)/A_MIN to the minimax,\n"
    "for SEARCH ONLY -- the persisted worst_violation stays the official 5-vector.\n"
    "0.0 (default) = off, exactly the historical behaviour.\n"
    "\n"
    "Aspect ratio is an OBJECTIVE in P3, so the minimax has never controlled it:\n"
    "A floats wherever descent leaves it, and all 991,855 of our evals with\n"
    "worst<=0.05 sit in A in [12.17, 13.10]. Within that very neighbourhood,\n"
    "spearman(A, qi violation) = -0.669 and spearman(A, worst) = -0.650, while\n"
    "iota/fluxcomp/vwell -- the three with 9-14%% unused margin -- worsen (+0.47,\n"
    "+0.55, +0.71). So A buys down exactly the two constraints that bind.\n"
    "\n"
    "This costs hypervolume on purpose. The scorer returns 0.0 with an empty\n"
    "feasible set, which is our score after ~1M evals, so a feasible design at\n"
    "A=14 scoring (1+L)(20-14) strictly dominates an infeasible one at A=12.9.\n"
    "Set A_MIN above the current A to push; the LP will equalise the aspect row\n"
    "against qi and mirror rather than jumping straight to it."))
    p.add_argument("--hv-tie-tol", type=float, default=0.0, metavar="TOL",
                   help="among line-search candidates within TOL of the best MEASURED worst\n"
                        "violation, select the one with the highest measured hv instead of the\n"
                        "first. 0.0 (default) preserves the historical pure-min-worst rule.\n"
                        "Without this the null-space component is invisible to selection: the\n"
                        "ladder picks on violation alone, so a candidate that gained hv at equal\n"
                        "violation loses to an arbitrary tie-mate.")
    p.add_argument("--fd-retry", type=float, nargs="*", default=[0.25, 0.06],
                   help="shrink factors to retry coordinates whose perturbed solve failed. "
                        "Recovers high-order poloidal modes that diverge at the full step.")
    p.add_argument("--n-line-search", type=int, default=9,
                   help="trial points along the LP direction, geometrically spaced from the full "
                        "trust-region step down to ~fd_step/2")
    p.add_argument("--delta0", type=float, default=0.10,
                   help="initial trust-region L2 BUDGET on the total step, in whitened units. "
                        "Sized as a few multiples of fd_step (0.02) so the LP solves for a "
                        "direction inside the region where its own linear model was measured.")
    p.add_argument("--delta-min", type=float, default=None,
                   help="Defaults to fd_step. Shrinking the trust region BELOW the "
                        "finite-difference interval is meaningless: the gradient is a secant "
                        "measured over fd_step, so extrapolating it to a finer step describes "
                        "nothing real, and fd_step cannot be reduced because fd_diagnostic.py "
                        "showed everything under ~1e-2 is noise. Seed 0's first run shrank to "
                        "6e-3 < fd_step=2e-2 and spent iterations in that meaningless regime. "
                        "The usable resolution of this objective is ~1e-2 whitened units.")
    p.add_argument("--max-stall", type=int, default=0, metavar="N",
                   help="stop after N consecutive iterations with no improvement. "
                        "0 (default) preserves the historical behaviour: shrink delta "
                        "by --shrink until it crosses --delta-min. That is fine at low "
                        "fidelity and ruinous at high -- `hf_asp16` took its last real "
                        "step at iteration 5 and then spent iterations 6-18 shrinking "
                        "0.043 -> 0.0006 at 0.85x a time, which is 1183 high-fidelity "
                        "VMEC++ evals and 6.5 of the run's 9.5 hours to confirm a local "
                        "optimum that was already unambiguous by iteration 8. Set this "
                        "for any --fidelity high run.")
    p.add_argument("--delta-max", type=float, default=0.3)
    p.add_argument("--grow", type=float, default=1.8)
    p.add_argument("--shrink", type=float, default=0.45)
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--n-workers", type=int, default=26)
    p.add_argument("--timeout-seconds", type=float, default=300.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    args = p.parse_args()
    if args.delta_min is None:
        # NOT floored at fd_step: delta bounds the LP's DIRECTION search, while
        # the geometric ladder independently controls actual step SIZE down to
        # fd_step/2. Flooring delta at fd_step terminated runs after 2-3
        # iterations, and the only config that produced a measured improvement
        # (5.3e-5) had a loose delta with the ladder doing the work.
        args.delta_min = 5e-4

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trace = STATE_DIR / f"{args.tag}_trace.jsonl"

    X = np.load(args.seed_npy)
    row = X[args.seed_row]
    x0 = row[:90].astype(np.float64)
    nfp = int(row[90]) if X.shape[1] > 90 else 2
    scales = np.load(args.coef_scales).astype(np.float64)
    free = np.array([i for i in range(90) if i not in oracle.ZERO_INDICES])
    z = x0[free] / scales[free]

    ev = Evaluator(args, x0, scales, free, nfp, STATE_DIR / f"{args.tag}_results.jsonl")
    print(f"[minimax] tag={args.tag} seed_row={args.seed_row} nfp={nfp} "
          f"n_free={len(free)} fd_step={args.fd_step}", flush=True)

    delta = args.delta0
    best = None
    stall = 0            # consecutive non-improving iterations, for --max-stall
    t0 = time.time()

    jac_log = []
    hv_start = None
    for it in range(args.max_iter):
        ev.cur_iter = it
        v, J, m0, nfail, g_hv, hv0 = constraint_jacobian(ev, z, args)
        if J is not None:
            # 5 x 81 sensitivity matrix per iterate: d(violation_i)/d(coef_j) in
            # whitened units. Reusable for surrogate fitting, conditioning
            # analysis, and identifying which coefficients actually move which
            # constraint -- independent of this particular optimizer.
            jac_log.append({"it": it, "z": z.tolist(), "v": v.tolist(),
                            "J": J.tolist()})
        if v is None:
            print("[minimax] base point failed to solve; shrinking and retrying", flush=True)
            delta *= args.shrink
            if delta < args.delta_min:
                break
            continue
        worst = float(v.max())
        # `best`, the saved _best.npy and the FEASIBLE test are all judged on the
        # OFFICIAL five constraints. With an aspect floor on, `worst` above is the
        # 6-vector max the SEARCH is minimising, which is deliberately larger
        # while A is still climbing -- it must never be mistaken for the score.
        worst_off = float(official_violations(m0).max())
        if best is None or worst_off < best[0]:
            best = (worst_off, z.copy(), m0)
            np.save(STATE_DIR / f"{args.tag}_best.npy",
                    np.concatenate([ev.z_to_x(z), [nfp], [worst_off]])[None, :])

        active = [(CNAMES + ["aspect"])[i] for i in range(len(v)) if v[i] > worst - 0.02]
        print(f"[minimax] it{it:>4} worst={worst_off:+.5f} search={worst:+.5f} "
              f"A={m0['aspect_ratio']:.3f} delta={delta:.4f} "
              f"fd_fail={nfail}/{len(free)} hv={hv0:.1f} active={','.join(active)} "
              f"evals={ev.n_evals} feasible={ev.n_feasible}", flush=True)
        with open(trace, "a") as tf:
            tf.write(json.dumps({"it": it, "worst": worst_off,
                                 "search_worst": worst, "delta": delta,
                                 "violations": v.tolist(), "fd_fail": nfail,
                                 "n_evals": ev.n_evals,
                                 "L_gradB": m0["minimum_normalized_magnetic_gradient_scale_length"],
                                 "aspect_ratio": m0["aspect_ratio"]}) + "\n")

        if worst_off <= FEASIBLE_TOL:
            print(f"[minimax] *** FEASIBLE at iteration {it} *** worst={worst_off:.6f}", flush=True)
            break

        floor = None
        if args.hv_mode == "floor":
            if hv_start is None:
                hv_start = hv0
            floor = args.hv_floor_frac * hv_start
        d, t_pred = minimax_step(v, J, delta, g_hv=g_hv, hv0=hv0,
                                 mode=args.hv_mode, hv_floor=floor,
                                 hv_weight=args.hv_weight)
        if d is None:
            print("[minimax] LP infeasible; shrinking", flush=True)
            delta *= args.shrink
            if delta < args.delta_min:
                break
            continue

        # Line search on the ACTUAL max violation, not the linear model.
        #
        # The LP always saturates the trust region (|d|inf == delta), so a fixed
        # alpha ladder like (1, .5, .25) never probes below 0.25*delta. With
        # delta >> fd_step that means every trial step is an extrapolation far
        # beyond the interval the gradient was measured over -- observed
        # directly at delta=0.6: t_pred=-0.94 (absurd) and all three trial
        # points failed VMEC outright. The ladder is therefore built to reach
        # DOWN TO roughly half an fd_step, so at least one trial always sits
        # inside the regime where the linear model has any claim to validity.
        # Scale the ladder by ||d||_2, NOT |d|inf. The LP saturates every
        # coordinate at the box bound, so |d|inf=delta over 81 coordinates means
        # ||d||_2 ~ delta*sqrt(81) = 9*delta -- a COMBINED displacement far larger
        # than the single-coordinate fd_step each partial derivative was measured
        # over. Measured directly at delta=0.3: ||d||_2 ~ 2.7 (135x fd_step), and
        # every trial step made the max violation WORSE (0.111-0.259 vs base
        # 0.0965), monotonically approaching the base value as alpha shrank.
        # The linear model's validity is governed by total displacement, so the
        # ladder must reach down to ||alpha*d||_2 ~ fd_step/2.
        # NULL-SPACE HYPERVOLUME COMPONENT.
        # P = I - J^+ J projects onto the null space of the 5x81 constraint
        # Jacobian, so J(P g_hv) = 0 and the added component moves no predicted
        # violation at first order. Built via lstsq on J J^T (5x5) rather than an
        # explicit 81x81 pinv: cheaper, and better conditioned when two constraint
        # gradients are nearly parallel -- which happens exactly at convergence,
        # where the active set equalizes and this component matters most.
        ns_retained = None
        if args.nullspace_hv > 0.0 and g_hv is not None and np.isfinite(g_hv).all():
            gh = np.asarray(g_hv, dtype=np.float64)
            lam, *_ = np.linalg.lstsq(J @ J.T, J @ gh, rcond=None)
            Pg = gh - J.T @ lam
            nPg, ngh = float(np.linalg.norm(Pg)), float(np.linalg.norm(gh))
            # Fraction of the hv gradient surviving projection. The pooled
            # measurement predicts a substantial value here; if it comes out near
            # zero at this point, the two subspaces are locally aligned after all
            # and the component correctly does nothing. Logged either way.
            ns_retained = nPg / ngh if ngh > 0 else 0.0
            dc = float(np.linalg.norm(d))
            if nPg > 1e-12 and dc > 0:
                d = d + args.nullspace_hv * dc * (Pg / nPg)

        dnorm = float(np.linalg.norm(d))
        a_min = min(1.0, max(args.fd_step / (2.0 * max(dnorm, 1e-12)), 1e-5))
        alphas = np.geomspace(1.0, a_min, args.n_line_search)
        # DEGENERATE LADDER. Once dnorm < fd_step/2 the clamp drives a_min to
        # exactly 1.0 and geomspace(1, 1, n) returns n copies of 1.0, so every
        # trial point is BIT-IDENTICAL and the iteration pays n evals for one.
        # `hf_asp16` spent iterations 10-18 like this -- the printed trial rows
        # are nine identical numbers to six decimals -- which at high fidelity is
        # 72 wasted VMEC++ evals, ~40 min of a 32-core box, per run. Collapsing
        # to the distinct alphas changes no decision (the duplicates cannot win
        # anything the survivor does not) and only stops paying for them.
        alphas = np.unique(np.round(alphas, 12))[::-1]
        if len(alphas) < args.n_line_search:
            print(f"[minimax]      ladder degenerate: |d|2={dnorm:.2e} < "
                  f"fd_step/2={args.fd_step/2:.2e}, so a_min clamped to 1.0 and "
                  f"{args.n_line_search} trials collapse to {len(alphas)} distinct "
                  f"point(s). The step is now SMALLER than the interval its own "
                  f"gradient was measured over.", flush=True)
        cands = [z + a * d for a in alphas]
        ms = ev(cands, kind="linesearch")

        # Candidates carry their LADDER INDEX so the trust-region update below
        # can tell whether the winning step came from the top or bottom of the
        # ladder. (Compacting this list silently shifts those indices and makes
        # the trust region grow/shrink on the wrong signal.)
        cand_rows = [(float(search_violations(m, args.aspect_min).max()), zc, m, k)
                     for k, (m, zc) in enumerate(zip(ms, cands)) if m is not None]

        # HARD floor: reject candidates whose MEASURED hypervolume is below the
        # floor. The LP only constrains the PREDICTED hv after the step
        # (hv0 + g_hv.d >= floor) and the line search previously accepted any
        # violation-reducing candidate without re-checking, so model error let hv
        # erode each iteration -- observed drifting 37.2 -> 34.5 -> 33.8 -> 31.5
        # against a 32.4 floor, behaving as a strong prior rather than a
        # constraint. Violations are already judged on measured values; hv now
        # is too. If NOTHING clears the floor, fall back to the unfiltered set
        # rather than stalling outright -- a stalled iteration shrinks delta and
        # loses the basin, which is worse than a small hv dip.
        if floor is not None:
            kept = [r for r in cand_rows if hv_contribution(r[2]) >= floor]
            if kept:
                cand_rows = kept

        if cand_rows:
            new_worst, new_z, new_m, win = min(cand_rows, key=lambda r: r[0])
            # Among candidates statistically indistinguishable on violation, take
            # the one worth more. Applied only when --hv-tie-tol > 0 so the
            # historical selection rule is unchanged by default.
            if args.hv_tie_tol > 0.0:
                near = [r for r in cand_rows if r[0] <= new_worst + args.hv_tie_tol]
                if near:
                    new_worst, new_z, new_m, win = max(
                        near, key=lambda r: hv_contribution(r[2]))
        else:
            new_worst, new_z, new_m, win = np.inf, None, None, 0
        # Predicted-vs-actual is the whole diagnosis. If t_pred sits well below
        # worst but the measured step does not move, the linear model is wrong
        # at this scale and no trust-region schedule can rescue it.
        trials = [round(float(search_violations(m, args.aspect_min).max()), 6) if m is not None else None
                  for m in ms]
        hv_trials = [round(hv_contribution(m), 1) if m is not None else None for m in ms]
        print(f"[minimax]      LP |d|inf={np.abs(d).max():.5f} |d|2={dnorm:.4f} "
              f"a_min={a_min:.2e} t_pred={t_pred:+.6f} "
              f"(pred drop {worst - t_pred:+.6f})  measured_best={new_worst:+.6f} "
              f"(actual drop {worst - new_worst:+.6f})  trials={trials} hv={hv_trials}"
              + ("" if ns_retained is None
                 else f" ns_retained={ns_retained:.3f}"), flush=True)

        if new_z is not None and new_worst < worst - 1e-9:
            # Trust-region update keyed on WHERE ON THE LADDER the winning step
            # was, not on predicted-vs-actual. The classic ratio test is useless
            # here: the LP's linear model overpredicts by orders of magnitude
            # (t_pred=-0.94 against a true 0.096), so `actual > 0.5*pred` never
            # fires and delta could only ever shrink -- it was about to
            # terminate at the floor WHILE the run was descending at 2e-4/iter.
            # A win high on the ladder means the region is too small; a win at
            # the bottom means the direction is only good very locally.
            z = new_z
            stall = 0
            # Indexed on the ACTUAL ladder length, not args.n_line_search: the
            # degenerate-ladder collapse above can shorten it, and the thirds
            # have to be thirds of the ladder that was really evaluated or the
            # trust region grows/shrinks on a mis-scaled signal.
            if win < len(alphas) // 3:
                delta = min(args.delta_max, delta * args.grow)
            elif win >= 2 * len(alphas) // 3:
                delta = max(args.delta_min, delta * 0.85)
        else:
            delta *= args.shrink
            stall += 1
            if args.max_stall > 0 and stall >= args.max_stall:
                print(f"[minimax] {stall} consecutive iterations with no "
                      f"improvement (--max-stall={args.max_stall}); stopping at "
                      f"worst={worst:+.6f} rather than shrinking delta={delta:.2e} "
                      f"down to {args.delta_min:.2e}", flush=True)
                break
            if delta < args.delta_min:
                print(f"[minimax] trust region collapsed below {args.delta_min}; "
                      f"local minimax optimum reached", flush=True)
                break

    if jac_log:
        np.savez_compressed(STATE_DIR / f"{args.tag}_jacobians.npz",
                            it=np.array([r["it"] for r in jac_log]),
                            z=np.array([r["z"] for r in jac_log]),
                            v=np.array([r["v"] for r in jac_log]),
                            J=np.array([r["J"] for r in jac_log]),
                            free_idx=free, scales=scales, nfp=nfp)
    dt = time.time() - t0
    print(f"\n[minimax] done in {dt/60:.1f} min, {ev.n_evals} real VMEC++ evals "
          f"(all persisted to {args.tag}_results.jsonl)")
    if best:
        w, zb, mb = best
        v = official_violations(mb)
        print(f"[minimax] best worst_violation={w:+.6f}")
        print("          " + "  ".join(f"{n}={x:+.5f}" for n, x in zip(CNAMES, v)))
        print(f"          L_gradB={mb['minimum_normalized_magnetic_gradient_scale_length']:.3f} "
              f"A={mb['aspect_ratio']:.3f} qi={mb['qi']:.6f}")
    print(f"[minimax] feasible designs found: {ev.n_feasible}")


if __name__ == "__main__":
    main()

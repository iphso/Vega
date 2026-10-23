"""L-BFGS refinement of P3 candidates, with finite-difference gradients taken
straight through the real VMEC++ solver.

WHY (EXPERIMENT_LOG, 2026-09-27). Across 822,600 real VMEC++ evaluations in 19
gradient-free runs, this project's best design is worst_violation=0.0917 --
found by `ngopt_v1` in its first 11,200 evals. The 680,300-eval `paper_v1` run
reached only 0.2036. More gradient-free compute stopped buying anything a long
time ago: only 36 designs in the entire inventory get below 0.10, and they all
come from the early small runs.

NGOpt is gradient-free in an 81-dimensional free-coefficient space. That is
reasonable for coarse exploration and hopeless for the fine convergence
feasibility actually needs (all five normalized violations <= 1e-2). The team
that topped the public leaderboard (arXiv:2608.28224, 0.9763) made exactly this
switch: they replaced their evolutionary optimizer with "local gradient based
L-BFGS", computing gradients "in parallel using a forward, two-point
finite-difference stencil straight through the solver", started from the
existing top-5 entries, and stated the limitation plainly -- the method
"requires solutions worth refining". We now have 40 of those.

WHY FINITE DIFFERENCES ARE SOUND HERE. FD gradients through a simulator are
worthless if the simulator is noisy. `fidelity_check.py` established that
VMEC++ at `low_fidelity` is BIT-FOR-BIT deterministic: two independent
re-evaluations of an identical boundary returned worst_violation identical to
all 17 significant digits. The project also runs single-threaded
(`multithreaded=False`), which `VmecSettings.max_threads` documents as required
for reproducibility. So there is no noise floor to fight.

COST. 81 free coefficients -> 82 evaluations per gradient, fully parallel. At
~10 s per eval over 32 workers that is roughly 30 s per gradient, so a few
hundred L-BFGS iterations is hours, not the days every prior approach cost.

TWO PHASES.
  Phase 1 (infeasible): minimize sum_i relu(v_i)^2 over the five OFFICIAL
    constraints. Smooth, and exactly zero once feasible.
  Phase 2 (feasible): maximize this design's own hypervolume contribution,
    (1 + L_gradB) * (20 - aspect_ratio), which is what the benchmark actually
    scores, while a stiff penalty holds feasibility. Aspect ratio is an
    OBJECTIVE in the real problem, never a constraint -- the paper's A* boxes
    are its own decomposition device and are deliberately absent here.

WHITENING. Optimization runs in z = x / scale, where scale is each
coefficient's std over the real 158,685-design dataset (output/coef_scales.npy).
Those scales span 540x, so an unwhitened L-BFGS inherits a hopelessly
ill-conditioned Hessian -- the same pathology that collapsed paper_v1's search
cloud to 2.4 effective dimensions.

Must run in the `train` service. Output: output/lbfgs_refine/
{<tag>_results.jsonl, <tag>_trace.jsonl, <tag>_best.npy}, and appends any
feasible design to output/ngopt_alm_p3_instance/feasible_pool.jsonl so
p3_front.py picks it up.
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
STATE_DIR = OUT_DIR / "lbfgs_refine"
POOL_PATH = OUT_DIR / "ngopt_alm_p3_instance" / "feasible_pool.jsonl"

FEASIBLE_TOL = 1e-2
REF_POINT = (1.0, 20.0)   # (-L_gradB, aspect_ratio)
FAIL_VALUE = 1e3


def official_violations(m):
    """problems._normalized_constraint_violations, transcribed exactly.
    Signed and NOT clipped -- negative means satisfied with margin."""
    iota = abs(m["edge_rotational_transform_over_n_field_periods"])
    return np.array([
        (0.25 - iota) / 0.25,
        (np.log10(m["qi"]) + 3.5) / 3.5,
        (m["edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (m["flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        (0.0 - m["vacuum_well"]) / max(1e-1, 0.0),
    ])


def hv_contribution(m):
    """This single design's hypervolume box against the reference point."""
    return ((REF_POINT[0] + m["minimum_normalized_magnetic_gradient_scale_length"])
            * (REF_POINT[1] - m["aspect_ratio"]))


class Oracle:
    """Batched real-VMEC++ evaluator over whitened coordinates."""

    def __init__(self, args, x_ref, scales, free_idx, nfp, results_path):
        self.args = args
        self.x_ref = x_ref
        self.scales = scales
        self.free = free_idx
        self.nfp = nfp
        self.results_path = results_path
        self.n_evals = 0
        self.n_feasible = 0
        self.best = None          # (phase1_obj, worst, z, metrics)
        self.best_hv = None       # (hv, z, metrics) among feasible
        self._tags = itertools.count()

    def z_to_x(self, z):
        x = self.x_ref.copy()
        x[self.free] = z * self.scales[self.free]
        x[oracle.ZERO_INDICES] = 0.0
        return x

    def evaluate(self, z_list):
        """One real VMEC++ call per z. Returns list of (metrics|None)."""
        fid = oracle.FIDELITY_PRESETS[self.args.fidelity]
        jobs, order = [], {}
        for i, z in enumerate(z_list):
            x = self.z_to_x(z)
            r_cos, z_sin, nfp, fname = oracle.params_to_worker_args(
                x, {"nfp": self.nfp}, fid)
            t = next(self._tags)
            jobs.append((t, r_cos, z_sin, nfp, fname))
            order[t] = i

        out = [None] * len(z_list)
        rows = []
        for t, ok, payload in run_batch_with_timeout(
            jobs, oracle.worker_fn, self.args.n_workers, self.args.timeout_seconds
        ):
            i = order[t]
            self.n_evals += 1
            if not ok:
                rows.append({"converged": False})
                continue
            m = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if any(v is None for v in m.values()):
                rows.append({"converged": False})
                continue
            out[i] = m
            v = official_violations(m)
            worst = float(v.max())
            rows.append({"converged": True, "worst_violation": worst,
                         "objective": m["minimum_normalized_magnetic_gradient_scale_length"],
                         "aspect_ratio": m["aspect_ratio"]})
            if worst <= FEASIBLE_TOL:
                self._record_feasible(z_list[i], m, worst)

        with open(self.results_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return out

    def _record_feasible(self, z, m, worst):
        self.n_feasible += 1
        hv = hv_contribution(m)
        if self.best_hv is None or hv > self.best_hv[0]:
            self.best_hv = (hv, z.copy(), m)
        with open(POOL_PATH, "a") as pf:
            pf.write(json.dumps({
                "tag": self.args.tag, "source": "lbfgs_refine", "nfp": self.nfp,
                "params": self.z_to_x(z).tolist(),
                "measured_targets": m,
                "official_worst_violation": worst,
                "L_gradB": m["minimum_normalized_magnetic_gradient_scale_length"],
                "aspect_ratio": m["aspect_ratio"],
            }) + "\n")
        if self.n_feasible == 1:
            print(f"[lbfgs] *** FIRST FEASIBLE DESIGN *** worst={worst:.5f} "
                  f"L_gradB={m['minimum_normalized_magnetic_gradient_scale_length']:.3f} "
                  f"A={m['aspect_ratio']:.3f} hv_contribution={hv:.2f}", flush=True)

    # ---- objectives -------------------------------------------------

    def phase1(self, m):
        """Smooth surrogate for max_i relu(v_i), the real feasibility condition."""
        if m is None:
            return FAIL_VALUE
        pos = np.maximum(0.0, official_violations(m))
        if self.args.objective == "sumsq":
            return float(np.sum(pos ** 2))
        p = self.args.pnorm_p
        return float(np.sum(pos ** p) ** (1.0 / p))

    def phase2(self, m):
        """Negative hypervolume contribution, with feasibility held by a stiff
        penalty. Only used once a feasible design has been seen."""
        if m is None:
            return FAIL_VALUE
        v = official_violations(m)
        pen = float(np.sum(np.maximum(0.0, v) ** 2))  # stiff quadratic is fine here
        return -hv_contribution(m) + self.args.phase2_penalty * pen

    def objective(self, m):
        use_p2 = self.args.phase2 and self.n_feasible > 0
        return self.phase2(m) if use_p2 else self.phase1(m)


def fun_and_grad(z, orc, args):
    """Value and forward-difference gradient, all 1+n_free evals in one batch.

    A perturbed point that fails VMEC gets gradient component 0 rather than a
    huge finite value: a failed solve carries no usable slope information, and
    letting FAIL_VALUE leak into the stencil would fabricate an enormous
    spurious gradient and throw L-BFGS across the domain.
    """
    n = len(z)
    batch = [z] + [z + args.fd_step * np.eye(n)[i] for i in range(n)]
    ms = orc.evaluate(batch)

    m0 = ms[0]
    f0 = orc.objective(m0)
    if m0 is None:
        # Base point unsolvable: flat, large value so the line search retreats.
        return FAIL_VALUE, np.zeros(n)

    g = np.zeros(n)
    n_fail = 0
    for i in range(n):
        mi = ms[i + 1]
        if mi is None:
            n_fail += 1
            continue
        g[i] = (orc.objective(mi) - f0) / args.fd_step

    v0 = official_violations(m0)
    worst = float(v0.max())
    if orc.best is None or f0 < orc.best[0]:
        orc.best = (f0, worst, z.copy(), m0)

    phase = 2 if (args.phase2 and orc.n_feasible > 0) else 1
    print(f"[lbfgs] eval#{orc.n_evals:>7}  f={f0:.6e}  worst={worst:+.5f}  "
          f"|g|={np.linalg.norm(g):.3e}  fd_fail={n_fail}/{n}  phase={phase}  "
          f"feasible={orc.n_feasible}", flush=True)
    with open(orc.results_path.with_name(orc.results_path.name.replace(
            "_results", "_trace")), "a") as tf:
        tf.write(json.dumps({
            "n_evals": orc.n_evals, "f": f0, "worst": worst,
            "grad_norm": float(np.linalg.norm(g)), "fd_fail": n_fail,
            "phase": phase, "L_gradB": m0["minimum_normalized_magnetic_gradient_scale_length"],
            "aspect_ratio": m0["aspect_ratio"],
        }) + "\n")
    return f0, g


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed-npy", default=str(OUT_DIR / "lbfgs_seeds.npy"),
                   help="rows of [90 params, nfp, worst_violation], best-first")
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--coef-scales", default=str(OUT_DIR / "coef_scales.npy"))
    p.add_argument("--objective", default="pnorm", choices=["pnorm", "sumsq"],
                   help="Phase-1 surrogate for the feasibility condition. Feasibility is a MAX "
                        "condition (ALL five normalized violations <= 1e-2), so sum-of-squares is "
                        "misaligned: it trades one constraint worse to improve two others. Observed "
                        "directly -- at seed 0 it drove f from 2.6165e-2 to 2.4602e-2 while "
                        "worst_violation went the WRONG way, 0.09653 -> 0.11186. `pnorm` uses "
                        "(sum relu(v)^p)^(1/p), a smooth approximation of the max whose gradient is "
                        "dominated by whichever constraint is actually binding.")
    p.add_argument("--pnorm-p", type=float, default=8.0)
    p.add_argument("--fd-step", type=float, default=2e-2,
                   help="plateau measured by fd_diagnostic.py: consecutive-gradient cosine is 0.91 "
                        "between h=1e-2 and 5e-2, but only 0.53 between 1e-3 and 1e-2, and NEGATIVE "
                        "(-0.14) between 1e-4 and 1e-3. Steps below ~1e-2 are noise.")
    p.add_argument("--fd-step-unused", type=float, default=1e-3,
                   help="forward-difference step in WHITENED units, so the real step on "
                        "coefficient i is fd_step * scale_i. Safe to make small because the "
                        "low-fidelity oracle is bit-for-bit deterministic (fidelity_check.py).")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--trust-z", type=float, default=6.0,
                   help="box half-width in whitened units (dataset stds) around the seed")
    p.add_argument("--phase2", action="store_true", default=True,
                   help="after the first feasible design, switch to maximizing hypervolume")
    p.add_argument("--no-phase2", dest="phase2", action="store_false")
    p.add_argument("--phase2-penalty", type=float, default=1e4)
    p.add_argument("--n-workers", type=int, default=26)
    p.add_argument("--timeout-seconds", type=float, default=300.0,
                   help="Generous on purpose. At 60s the SAME base point gave fd_fail=0/81 in one "
                        "run and 7/81 in another -- identical deterministic inputs, so those were "
                        "timeouts under 30-way contention, not VMEC divergence. Dropped components "
                        "are silently set to zero gradient, which corrupts the descent direction "
                        "non-deterministically. Slow-but-complete beats fast-but-lossy here.")
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    args = p.parse_args()

    from scipy.optimize import minimize

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"{args.tag}_results.jsonl"

    X = np.load(args.seed_npy)
    row = X[args.seed_row]
    x0 = row[:90].astype(np.float64)
    nfp = int(row[90]) if X.shape[1] > 90 else 2
    seed_worst = float(row[91]) if X.shape[1] > 91 else float("nan")

    scales = np.load(args.coef_scales).astype(np.float64)
    free_idx = np.array([i for i in range(90) if i not in oracle.ZERO_INDICES])
    z0 = x0[free_idx] / scales[free_idx]

    print(f"[lbfgs] tag={args.tag} seed_row={args.seed_row} nfp={nfp} "
          f"seed_worst={seed_worst:.5f}")
    print(f"[lbfgs] {len(free_idx)} free coefficients -> {len(free_idx)+1} evals per gradient")
    print(f"[lbfgs] trust box +-{args.trust_z} dataset-stds, fd_step={args.fd_step} (whitened)")

    orc = Oracle(args, x0, scales, free_idx, nfp, results_path)
    bounds = [(zi - args.trust_z, zi + args.trust_z) for zi in z0]

    t0 = time.time()
    res = minimize(
        fun_and_grad, z0, args=(orc, args), jac=True, method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": args.max_iter, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-10},
    )
    dt = time.time() - t0

    print(f"\n[lbfgs] done in {dt/60:.1f} min, {orc.n_evals} real VMEC++ evals")
    print(f"[lbfgs] scipy: {res.message}")
    if orc.best:
        f, worst, zb, mb = orc.best
        print(f"[lbfgs] best: worst_violation={worst:+.5f} (seed was {seed_worst:.5f})  "
              f"f={f:.6e}")
        print(f"        L_gradB={mb['minimum_normalized_magnetic_gradient_scale_length']:.3f} "
              f"A={mb['aspect_ratio']:.3f}  qi={mb['qi']:.6f}")
        v = official_violations(mb)
        print("        violations [iota,qi,mirror,fluxcomp,vwell] = "
              + np.array2string(v, precision=5, suppress_small=False))
        np.save(STATE_DIR / f"{args.tag}_best.npy",
                np.concatenate([orc.z_to_x(zb), [nfp], [worst]])[None, :])
    print(f"[lbfgs] feasible designs found: {orc.n_feasible}")


if __name__ == "__main__":
    main()

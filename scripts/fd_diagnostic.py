"""Is the P3 objective actually differentiable at the scales we are probing?

`lbfgs_refine_p3.py`'s first four L-BFGS iterations moved worst_violation only
0.09653 -> 0.09637 over 492 real evals, with |grad| RISING (0.80 -> 1.11) while
steps shrank, and fd_fail climbing 2/81 -> 17/81. Rising gradient norm plus a
line search that keeps rejecting is the classic signature of a finite-difference
step in the wrong regime -- the linear model does not predict the function, so
every trial step overshoots.

VMEC++ at low fidelity is bit-for-bit deterministic (fidelity_check.py), so
there is no stochastic noise floor. But determinism is not smoothness: `qi` is
computed through a Boozer transform plus spline interpolation and bounce-point
root-finding (`omnigeneity/qi.py`), any of which can be piecewise-constant or
kinked at small displacements. A deterministic staircase produces gradients that
are exactly zero almost everywhere and enormous at the steps -- which is exactly
what "|g| rising while progress stalls" looks like.

Two tests, both at the best recoverable design (ngopt_v2, worst=0.09653):

  SCAN  Walk a single coefficient across several decades of displacement and
        print the five official violations at each point. If the metrics move
        smoothly, differences scale linearly with the step. If they sit
        constant and then jump, the function is a staircase at that resolution
        and no finite-difference step will recover a useful gradient.

  GRAD  Compute the full 81-dim gradient at several fd_step values and compare
        consecutive ones by cosine similarity and norm. A well-posed step size
        shows a plateau where consecutive gradients agree (cos ~ 1). No plateau
        means there is no good step size and the approach needs rethinking.

Cheap: SCAN is ~11 evals per coordinate, GRAD is 82 per step size.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

OUT_DIR = Path("/work/output")


def official_violations(m):
    iota = abs(m["edge_rotational_transform_over_n_field_periods"])
    return np.array([
        (0.25 - iota) / 0.25,
        (np.log10(m["qi"]) + 3.5) / 3.5,
        (m["edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (m["flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        (0.0 - m["vacuum_well"]) / 1e-1,
    ])


def phase1(m):
    if m is None:
        return None
    return float(np.sum(np.maximum(0.0, official_violations(m)) ** 2))


class Batch:
    def __init__(self, x_ref, scales, free, nfp, n_workers, timeout, fidelity):
        self.x_ref, self.scales, self.free, self.nfp = x_ref, scales, free, nfp
        self.n_workers, self.timeout = n_workers, timeout
        self.fid = oracle.FIDELITY_PRESETS[fidelity]
        self._t = itertools.count()

    def z_to_x(self, z):
        x = self.x_ref.copy()
        x[self.free] = z * self.scales[self.free]
        x[oracle.ZERO_INDICES] = 0.0
        return x

    def __call__(self, zs):
        jobs, order = [], {}
        for i, z in enumerate(zs):
            rc, zs_, nfp, fn = oracle.params_to_worker_args(
                self.z_to_x(z), {"nfp": self.nfp}, self.fid)
            t = next(self._t)
            jobs.append((t, rc, zs_, nfp, fn)); order[t] = i
        out = [None] * len(zs)
        for t, ok, payload in run_batch_with_timeout(
            jobs, oracle.worker_fn, self.n_workers, self.timeout
        ):
            if not ok:
                continue
            m = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if all(v is not None for v in m.values()):
                out[order[t]] = m
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-npy", default=str(OUT_DIR / "lbfgs_seeds.npy"))
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--coef-scales", default=str(OUT_DIR / "coef_scales.npy"))
    p.add_argument("--scan-coords", type=int, nargs="+", default=[0, 5, 20],
                   help="indices into the 81 FREE coordinates to scan")
    p.add_argument("--grad-steps", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 5e-2, 2e-1])
    p.add_argument("--n-workers", type=int, default=30)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS))
    p.add_argument("--skip-grad", action="store_true")
    args = p.parse_args()

    X = np.load(args.seed_npy)
    row = X[args.seed_row]
    x0 = row[:90].astype(np.float64)
    nfp = int(row[90]) if X.shape[1] > 90 else 2
    scales = np.load(args.coef_scales).astype(np.float64)
    free = np.array([i for i in range(90) if i not in oracle.ZERO_INDICES])
    z0 = x0[free] / scales[free]
    ev = Batch(x0, scales, free, nfp, args.n_workers, args.timeout_seconds, args.fidelity)

    base = ev([z0])[0]
    if base is None:
        print("[fd] base point does not converge -- aborting")
        return
    f0 = phase1(base)
    v0 = official_violations(base)
    print(f"[fd] base: f={f0:.8e}  worst={v0.max():+.6f}")
    print(f"[fd] violations [iota,qi,mirror,fluxcomp,vwell] = "
          f"{np.array2string(v0, precision=6)}\n")

    # ---------------- SCAN ----------------
    steps = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    for c in args.scan_coords:
        zs = [z0 + s * np.eye(len(z0))[c] for s in steps]
        ms = ev(zs)
        print(f"=== SCAN free-coord {c} (real coefficient index {free[c]}, "
              f"scale {scales[free[c]]:.5f}) ===")
        print(f"{'step_z':>9} {'real dx':>11} {'df':>14} {'df/step':>12} "
              f"{'d(qi_viol)':>12} {'d(mirror)':>11}")
        for s, m in zip(steps, ms):
            if m is None:
                print(f"{s:>9.0e} {s*scales[free[c]]:>11.2e} {'VMEC FAIL':>14}")
                continue
            df = phase1(m) - f0
            dv = official_violations(m) - v0
            print(f"{s:>9.0e} {s*scales[free[c]]:>11.2e} {df:>+14.6e} "
                  f"{df/s:>+12.4e} {dv[1]:>+12.3e} {dv[2]:>+11.3e}")
        print()

    if args.skip_grad:
        return

    # ---------------- GRAD ----------------
    n = len(z0)
    grads = {}
    for h in args.grad_steps:
        zs = [z0 + h * np.eye(n)[i] for i in range(n)]
        ms = ev(zs)
        g = np.zeros(n); nf = 0
        for i, m in enumerate(ms):
            if m is None:
                nf += 1; continue
            g[i] = (phase1(m) - f0) / h
        grads[h] = (g, nf)
        print(f"[fd] grad @ h={h:<8.0e} |g|={np.linalg.norm(g):.4e}  "
              f"nonzero={int((g!=0).sum())}/{n}  fail={nf}")

    print(f"\n{'h_a':>10} {'h_b':>10} {'cosine':>9} {'|g_a|/|g_b|':>12}")
    hs = sorted(grads)
    for a, b in zip(hs, hs[1:]):
        ga, gb = grads[a][0], grads[b][0]
        na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
        cos = float(ga @ gb / (na * nb)) if na > 0 and nb > 0 else float("nan")
        print(f"{a:>10.0e} {b:>10.0e} {cos:>9.4f} {na/max(nb,1e-30):>12.3f}")
    print("\n[fd] cosine ~1 between consecutive steps = stable plateau, that h is usable.")
    print("[fd] cosine near 0 everywhere = no usable finite-difference step exists.")


if __name__ == "__main__":
    main()

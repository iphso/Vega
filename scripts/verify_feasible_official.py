"""Run the OFFICIAL P3 scorer on our feasible candidates, at high fidelity.

WHY THIS EXISTS. Everything in the search loop is evaluated with VMEC++
`low_fidelity`, which is what makes the oracle cheap (and bit-for-bit
deterministic, which the exhaustion protocol depends on). The benchmark does not
use it: `MHDStableQIStellarator.evaluate()` calls
`ConstellarationSettings.default_high_fidelity()`. A design that satisfies all
five constraints at low fidelity is therefore a CANDIDATE for a score, not a
score. This script is the only thing in the project that produces a real P3
number, and it must be the source of any claim about one.

It calls `problem.evaluate(boundaries)` directly -- not a transcription of the
constraint formulas -- so the score, the feasibility flags and the objectives all
come from the installed scorer rather than from our reading of it. (Our
transcription in `minimax_p3.official_violations` was separately checked against
the scorer source and matches, but a transcription is not what we want to trust
at the point of making a claim.)

Usage (inside the `train` service):
    python3 scripts/verify_feasible_official.py --tag asp_a13.5 --n 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402

OUT = Path("/work/output")
NAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]


def low_fidelity_violations(mt):
    return np.array([
        (0.25 - abs(mt["edge_rotational_transform_over_n_field_periods"])) / 0.25,
        (np.log10(mt["qi"]) + 3.5) / 3.5,
        (mt["edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (mt["flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        (0.0 - mt["vacuum_well"]) / 1e-1,
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", action="append", required=True,
                   help="run tag(s) in output/minimax_p3 to pull candidates from")
    p.add_argument("--n", type=int, default=8,
                   help="how many distinct candidates to verify (high fidelity is slow)")
    p.add_argument("--max-low-worst", type=float, default=1e-2,
                   help="only consider candidates whose LOW-fidelity worst violation is <= this")
    p.add_argument("--out", default="/work/output/official_verification.jsonl")
    args = p.parse_args()

    # Collect distinct feasible-at-low-fidelity candidates, best first.
    seen, pool = set(), []
    for tag in args.tag:
        f = OUT / "minimax_p3" / f"{tag}_results.jsonl"
        if not f.exists():
            print(f"[verify] missing {f}", flush=True)
            continue
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not d.get("converged"):
                continue
            mt = d.get("measured_targets")
            params = d.get("params")
            if not isinstance(mt, dict) or params is None:
                continue
            try:
                w = float(low_fidelity_violations(mt).max())
            except Exception:
                continue
            if w > args.max_low_worst:
                continue
            key = tuple(np.round(np.asarray(params, dtype=float), 10))
            if key in seen:
                continue
            seen.add(key)
            pool.append((w, np.asarray(params, dtype=float), int(d["nfp"]), mt, tag))
    pool.sort(key=lambda r: r[0])
    # Say what the cut actually was. With --max-low-worst above 1e-2 these are NOT
    # feasible designs, and printing them as "feasible" is how a mis-labelled count
    # gets quoted back later as a result -- a scan keyed on a stored violation field
    # once reported 91 "feasible" designs that were cross-lane artifacts.
    label = ("low-fidelity-FEASIBLE" if args.max_low_worst <= 1e-2
             else f"candidates with low-fidelity worst <= {args.max_low_worst:g} (NOT feasible)")
    print(f"[verify] {len(pool)} distinct {label}; "
          f"verifying the best {min(args.n, len(pool))} at HIGH fidelity", flush=True)
    if not pool:
        return

    from constellaration.geometry import surface_rz_fourier
    from constellaration.problems import MHDStableQIStellarator

    problem = MHDStableQIStellarator()
    boundaries, meta = [], []
    for w, params, nfp, mt, tag in pool[:args.n]:
        r_cos, z_sin, nfp_i, _ = oracle.params_to_worker_args(
            params, {"nfp": nfp}, "high_fidelity")
        boundaries.append(surface_rz_fourier.SurfaceRZFourier(
            r_cos=r_cos, z_sin=z_sin, n_field_periods=nfp_i,
            is_stellarator_symmetric=True))
        meta.append((w, tag, mt, params, nfp))

    # ONE DESIGN AT A TIME. `problem.evaluate(list)` runs the forward model in a
    # loop and propagates a VMEC++ non-convergence as an exception, so a single bad
    # boundary destroys the whole batch -- that is exactly how the first attempt at
    # this died, with 8 candidates in and no result out. High fidelity uses a finer
    # radial/Fourier resolution than the search, so a design that converged at
    # `low_fidelity` genuinely may not converge here; that is an outcome to record
    # per design, not an error to abort on.
    print("[verify] evaluating one design at a time at HIGH fidelity (slow)", flush=True)
    survivors, rows = [], []
    for i, (b, (w_low, tag, mt_low, params, nfp)) in enumerate(zip(boundaries, meta)):
        try:
            r1 = problem.evaluate([b])
        except Exception as e:
            print(f"  [{i}] {tag}: low_worst={w_low:.6f} -> HIGH-FIDELITY FAILED "
                  f"({type(e).__name__}: {e})", flush=True)
            rows.append({"tag": tag, "low_worst": w_low, "high_feasible": False,
                         "high_error": f"{type(e).__name__}: {e}",
                         "nfp": nfp, "params": params.tolist()})
            continue
        feas = r1.feasibility[0] if r1.feasibility is not None else None
        obj = r1.objectives[0] if r1.objectives is not None else None
        ok = bool(np.all(np.asarray(feas) <= 0)) if np.ndim(feas) else bool(feas)
        print(f"  [{i}] {tag}: low_worst={w_low:.6f} -> HIGH-FIDELITY feasible={ok} "
              f"single_point_score={r1.score}", flush=True)
        print(f"       objectives={obj}", flush=True)
        print(f"       feasibility={feas}", flush=True)
        print(f"       low-fid A={mt_low['aspect_ratio']:.4f} "
              f"L={mt_low['minimum_normalized_magnetic_gradient_scale_length']:.4f}",
              flush=True)
        rows.append({"tag": tag, "low_worst": w_low, "high_feasible": ok,
                     "objectives": None if obj is None else [float(x) for x in np.ravel(obj)],
                     "feasibility": None if feas is None else [float(x) for x in np.ravel(feas)],
                     "single_point_score": float(r1.score), "nfp": nfp,
                     "params": params.tolist()})
        if ok:
            survivors.append(b)

    nfeas = sum(1 for r in rows if r["high_feasible"])
    print(f"\n[verify] {nfeas}/{len(meta)} survived at high fidelity", flush=True)

    # The score is the hypervolume of the feasible SET, so it must be computed on
    # the survivors together -- the max of the single-point scores is a lower bound
    # on it, not the number.
    set_score = None
    if survivors:
        try:
            set_score = float(problem.evaluate(survivors).score)
            print(f"[verify] OFFICIAL SCORE over the {len(survivors)} surviving "
                  f"design(s): {set_score}", flush=True)
        except Exception as e:
            print(f"[verify] set evaluation failed ({type(e).__name__}: {e}); "
                  f"per-design scores above still stand", flush=True)
    else:
        print("[verify] no survivors -> official score remains 0.0", flush=True)

    with open(args.out, "a") as fh:
        for r in rows:
            r["set_score"] = set_score
            fh.write(json.dumps(r) + "\n")
    print(f"[verify] appended to {args.out}", flush=True)


if __name__ == "__main__":
    main()

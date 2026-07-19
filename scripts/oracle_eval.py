"""Ground-truth physics oracle for a boundary design, via VMEC++
(constellaration.forward_model) -- the same solver and metric definitions
used to generate this project's training dataset, so it plugs in directly
as a ground-truth check for the surrogate/design-search work rather than
just trusting surrogate predictions on designs outside the training
distribution.

Two fidelity levels (constellaration's own presets):
  low  (default) -- what generated this dataset. ~2-10s per call in testing.
  high -- what the ConStellaration benchmark scores against. Slower (not
          yet benchmarked here) -- meant for dialing in on already-promising
          candidates found cheaply at low fidelity, not for a first pass.

An invalid/unphysical boundary fails fast (~0.02s, a clear RuntimeError from
the VMEC++ solver, or a pydantic ValidationError if it violates stellarator-
symmetry structural constraints before ever reaching the solver) rather than
hanging -- this alone is a free, cheap "is this even physically possible"
signal, independent of getting exact metric values.

Two input modes:
  --design-json PATH   a design saved by optimize.py's --save flag (has
                        r_cos, z_sin, n_field_periods, predicted_targets).
  --row-index N         a real row from output/X.npy, for sanity-checking
                        the oracle itself against already-known metrics.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier

OUT_DIR = Path("/work/output")


def load_from_design_json(path):
    design = json.loads(Path(path).read_text())
    r_cos = np.array(design["r_cos"], dtype=np.float64).reshape(5, 9)
    z_sin = np.array(design["z_sin"], dtype=np.float64).reshape(5, 9)
    nfp = int(design["n_field_periods"])
    is_sym = bool(design["is_stellarator_symmetric"])
    comparison = design.get("predicted_targets")
    return r_cos, z_sin, nfp, is_sym, comparison


def load_from_row_index(idx):
    X = np.load(OUT_DIR / "X.npy")
    Y = np.load(OUT_DIR / "Y.npy")
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    row = X[idx]
    r_cos = row[:45].reshape(5, 9).astype(np.float64)
    z_sin = row[45:90].reshape(5, 9).astype(np.float64)
    nfp = int(row[90])
    is_sym = bool(row[91])
    comparison = {name: float(v) for name, v in zip(target_names, Y[idx])}
    return r_cos, z_sin, nfp, is_sym, comparison


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--design-json", help="path to a design saved by optimize.py --save")
    g.add_argument("--row-index", type=int, help="index into output/X.npy, for sanity-checking the oracle")
    p.add_argument("--fidelity", default="low", choices=["low", "high"])
    args = p.parse_args()

    if args.design_json:
        r_cos, z_sin, nfp, is_sym, comparison = load_from_design_json(args.design_json)
        source = args.design_json
    else:
        r_cos, z_sin, nfp, is_sym, comparison = load_from_row_index(args.row_index)
        source = f"output/X.npy row {args.row_index}"

    print(f"=== oracle_eval: {source}  (nfp={nfp}, fidelity={args.fidelity}) ===")

    try:
        boundary = surface_rz_fourier.SurfaceRZFourier(
            r_cos=r_cos, z_sin=z_sin, n_field_periods=nfp, is_stellarator_symmetric=is_sym,
        )
    except Exception as e:
        print(f"INFEASIBLE (structural validation, before ever reaching the solver): "
              f"{type(e).__name__}: {e}")
        return

    settings = None
    if args.fidelity == "high":
        settings = forward_model.ConstellarationSettings.default_high_fidelity()

    t0 = time.perf_counter()
    try:
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
    except Exception as e:
        print(f"INFEASIBLE (VMEC++ solver failed after {time.perf_counter() - t0:.2f}s): "
              f"{type(e).__name__}: {e}")
        return
    elapsed = time.perf_counter() - t0

    print(f"FEASIBLE -- VMEC++ converged in {elapsed:.2f}s\n")
    metrics_dict = metrics.model_dump()
    if comparison:
        print(f"  {'target':55s} {'oracle (ground truth)':>22s} {'surrogate/dataset':>18s} {'rel. diff':>10s}")
        for name, oracle_val in metrics_dict.items():
            if oracle_val is None or name not in comparison:
                continue
            ref = comparison[name]
            rel_diff = (oracle_val - ref) / abs(ref) if ref else float("nan")
            print(f"  {name:55s} {oracle_val:22.6g} {ref:18.6g} {rel_diff:10.2%}")
    else:
        print("  recomputed metrics (no comparison values available):")
        for name, val in metrics_dict.items():
            if val is not None:
                print(f"    {name:55s} {val:.6g}")


if __name__ == "__main__":
    main()

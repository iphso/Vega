"""Is `problem.evaluate()` broken, or are our designs genuinely unresolvable?

CONTEXT. Both of the project's leading designs -- `asp_a13.5` (low-fidelity
official worst 0.006549, all five constraints satisfied) and `ex_floor2_r21`
(0.026563, the 1M-eval leader) -- came back `RuntimeError: VMEC++ did not
converge` from `MHDStableQIStellarator.evaluate()`. That is the real scorer, so
if it stands, neither design can score at all.

Before concluding anything about our designs, the harness has to be exonerated.
`ConstellarationSettings.default_high_fidelity()` was verified field-for-field
identical to the oracle's `VmecPresetSettings(fidelity="high_fidelity")` (0
differing fields of 12), so the settings path is not the explanation. What is
still untested is whether OUR call into `problem.evaluate()` works at all.

THE CONTROL. Designs straight from the ConStellaration dataset are VMEC-converged
by construction -- that is how the dataset was produced. Run them through the
identical code path:
  - dataset designs converge, ours don't  -> the harness is fine and our designs
    are genuinely marginal at high fidelity. The low-fidelity feasible set is an
    artifact and ~1M evals have been optimising against a resolution the scorer
    does not use.
  - dataset designs ALSO fail            -> the harness is wrong, every
    high-fidelity conclusion today is void, and nothing about our designs has
    been shown.
The second outcome is the one to hope for; it is also the one that costs nothing
but a bug fix.
"""
import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", default="/work/output/highfid_harness_check.jsonl")
    a = p.parse_args()

    from datasets import load_dataset
    from constellaration.geometry import surface_rz_fourier
    from constellaration.problems import MHDStableQIStellarator

    print("[chk] streaming the ConStellaration dataset for control boundaries",
          flush=True)
    ds = load_dataset("proxima-fusion/constellaration", split="train", streaming=True)

    problem = MHDStableQIStellarator()
    rows = []
    taken = 0
    for rec in ds:
        if taken >= a.n:
            break
        # Fields are flattened with dotted names; `preprocess.py` already relies on
        # this layout, so use its convention rather than guessing at a JSON blob.
        try:
            if not rec["boundary.is_stellarator_symmetric"]:
                continue   # our whole parameterisation is stellarator-symmetric
            b = surface_rz_fourier.SurfaceRZFourier(
                r_cos=np.asarray(rec["boundary.r_cos"], dtype=float),
                z_sin=np.asarray(rec["boundary.z_sin"], dtype=float),
                n_field_periods=int(rec["boundary.n_field_periods"]),
                is_stellarator_symmetric=True)
        except KeyError:
            print(f"[chk] unexpected schema; keys = {list(rec)[:20]}", flush=True)
            return
        except Exception as e:
            print(f"[chk] could not build boundary: {type(e).__name__}: {e}", flush=True)
            continue
        taken += 1
        try:
            r = problem.evaluate([b])
            ok = True
            print(f"  [{taken-1}] dataset nfp={b.n_field_periods} "
                  f"-> CONVERGED  score={r.score}  feasibility={r.feasibility}",
                  flush=True)
            rows.append({"source": "dataset", "nfp": int(b.n_field_periods),
                         "converged": True, "score": float(r.score)})
        except Exception as e:
            ok = False
            print(f"  [{taken-1}] dataset nfp={b.n_field_periods} "
                  f"-> FAILED  {type(e).__name__}: {e}", flush=True)
            rows.append({"source": "dataset", "nfp": int(b.n_field_periods),
                         "converged": False, "error": f"{type(e).__name__}: {e}"})
        del ok

    nconv = sum(1 for r in rows if r["converged"])
    print(f"\n[chk] {nconv}/{len(rows)} dataset designs converged through "
          f"problem.evaluate()", flush=True)
    if nconv == 0:
        print("[chk] VERDICT: the harness is broken. Every high-fidelity result "
              "from today is void -- nothing has been shown about our designs.",
              flush=True)
    elif nconv == len(rows):
        print("[chk] VERDICT: the harness works. Our designs are genuinely "
              "non-convergent at the fidelity the scorer uses.", flush=True)
    else:
        print("[chk] VERDICT: mixed -- high-fidelity non-convergence is a property "
              "of boundaries generally, not of our search. Needs a rate, not a "
              "verdict: re-run with a larger --n.", flush=True)

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

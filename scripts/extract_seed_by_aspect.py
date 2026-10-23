"""Pull a seed out of the inventory by ASPECT RATIO as well as violation.

WHY THIS EXISTS. `harvest_best.py` ranks by violation or by (worst, hv) Pareto
front, and both of those push toward LOW aspect ratio, because aspect is an
objective: hv = (1+L_gradB)*(20-A), so every unit of aspect is score given away.
That is correct when picking a design to submit. It is wrong when picking a seed
for the aspect lever.

The lever's whole premise -- validated at high fidelity by `hf_asp13` iteration 2,
where mirror fell 0.0797 -> 0.0217 for +0.25 of aspect -- is that aspect ratio BUYS
constraint satisfaction. A seed for that experiment has to already sit in the
high-aspect corner, for a reason that is arithmetic rather than aesthetic: the
floor enters the minimax as the normalized row `(aspect_min - A)/aspect_min`, so
seeding a floor of 13.5 from a design at A=8.4 creates a violation of +0.377,
4.7x every real constraint. The minimax is then a pure aspect-climber and the
physics is collateral; that is exactly how `hf_asp13` went 0.080 -> 0.137 in three
iterations. The low-fidelity run the lever came from (`asp_a13.5`) started at
A=12.894 against the same floor, i.e. a mild +0.045, comparable to its real
constraints. The lever was always a gentle pressure, never a climb.

AND THE SEED MUST CONVERGE AT HIGH FIDELITY. That is the 2026-10-05 finding and it
is non-negotiable: 0/18 of our designs below worst 0.05 converge at the fidelity
that scores, so "best" by low-fidelity violation is anti-correlated with usable.
`--require-survey` restricts candidates to designs the convergence survey actually
ran at high fidelity and saw converge. Matching is by (tag, aspect, worst) rounded,
because the survey persists metrics but not parameters.
"""
import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from minimax_p3 import official_violations  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="output/minimax_p3/*_results.jsonl")
    p.add_argument("--aspect-min", type=float, default=14.5)
    p.add_argument("--aspect-max", type=float, default=15.5)
    p.add_argument("--max-worst", type=float, default=0.105)
    p.add_argument("--require-survey",
                   default="output/highfid_convergence_survey.jsonl",
                   help="only keep designs this survey saw CONVERGE at high "
                        "fidelity; '' disables the check (not recommended)")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", default="output/aspect_seed.npy")
    a = p.parse_args()

    okset = None
    if a.require_survey:
        okset = set()
        for line in open(a.require_survey):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("high_converged"):
                okset.add((d.get("tag"), round(float(d["A_low"]), 2),
                           round(float(d["worst_low"]), 4)))
        print(f"[seed] survey: {len(okset)} designs confirmed converging at high "
              f"fidelity")

    cand = []
    for f in sorted(globmod.glob(a.glob)):
        tag = Path(f).name.replace("_results.jsonl", "")
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not d.get("converged"):
                continue
            mt, params = d.get("measured_targets"), d.get("params")
            if not isinstance(mt, dict) or params is None or len(params) != 90:
                continue
            try:
                w = float(official_violations(mt).max())
                A = float(mt["aspect_ratio"])
            except Exception:
                continue
            if not (a.aspect_min <= A <= a.aspect_max) or w > a.max_worst:
                continue
            if okset is not None and (tag, round(A, 2), round(w, 4)) not in okset:
                continue
            cand.append((w, A, int(d.get("nfp", 2)),
                         np.asarray(params, dtype=float), tag))

    print(f"[seed] {len(cand)} candidates with A in [{a.aspect_min}, "
          f"{a.aspect_max}] and official worst <= {a.max_worst}"
          + (" (survey-confirmed)" if okset is not None else ""))
    if not cand:
        print("[seed] nothing matched; widen the window or drop --require-survey")
        return

    cand.sort(key=lambda r: r[0])
    out = []
    print(f"\n{'#':>3} {'worst':>9} {'A':>8} nfp  tag")
    for i, (w, A, nfp, x, tag) in enumerate(cand[:a.n]):
        print(f"{i:>3} {w:9.5f} {A:8.3f} {nfp:>3}  {tag}")
        out.append(np.concatenate([x, [nfp], [w]]))

    arr = np.asarray(out)
    np.save(a.out, arr)
    print(f"\n[seed] wrote {arr.shape} -> {a.out}")
    print(f"[seed] NOTE: pick the floor from the seed's own aspect. At A={cand[0][1]:.2f} "
          f"a floor of {cand[0][1]*1.06:.1f} is a +0.057 pressure -- below the real "
          f"constraints, so the minimax trades rather than climbs.")


if __name__ == "__main__":
    main()

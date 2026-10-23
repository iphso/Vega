"""How many of the 81 free coefficients actually carry the high-fidelity gradient?

WHY THIS IS THE DOMINANT LEVER. Measured cost of one high-fidelity minimax
iteration on 32 cores: ~60 min (hf_s0b iteration 0, 91 evals, 91/91 converged).
At the observed -2.7%/iteration, descending 0.094 -> 0.01 needs ~82 iterations
~ 3.4 days of the whole box, and the project's own history says the trust region
collapses long before that. The fidelity SCHEDULE cannot fix this: the handoff from
low fidelity sits at worst ~0.095 and we are already at 0.094, so every remaining
step is irreducibly high-fidelity.

What is left is the per-iteration cost, and that is set by 82 evals per gradient
(81 free coefficients + base). On 32 workers 82 evals is THREE waves -- and the
third carries only 18 jobs, so most of it is idle. Cutting to <=63 free
coefficients would make it two waves; <=31 would make it one. Those are 33% and
67% reductions in wall time per iteration, available with no loss if the dropped
coordinates do not carry gradient.

THE DATA IS ALREADY PAID FOR. hf_s0b's iteration-0 stencil is a real
high-fidelity Jacobian: 82 evaluations of v(z + h*e_i) around one base point. This
reconstructs J from the persisted rows and ranks coordinates by how much they move
the five official constraints. No new simulation.

PRIOR EVIDENCE THAT THIS WILL PAY. Coefficient scales span 540x across the
dataset, `paper_v1`'s search cloud collapsed to ~2.4 effective dimensions, and the
stencil's own retry comment records failures concentrating in a contiguous block of
HIGH-ORDER poloidal modes (r_cos m=3,4) where a full step makes the boundary too
sharp for VMEC. If those same high-order coordinates also carry little gradient,
dropping them is free twice over -- cheaper iterations AND fewer excursions toward
the unresolvable boundaries that caused the 2026-10-05 artifact.

CAVEAT ON READING IT. This is ONE base point in ONE basin. Column norms are a
local property; a coordinate that is flat here may matter elsewhere. The output is
a candidate reduction to TEST (run a paired reduced-vs-full iteration and compare
descent), not a reduction to adopt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from minimax_p3 import official_violations  # noqa: E402

NAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hf_s0b")
    p.add_argument("--iter", type=int, default=0)
    p.add_argument("--scales", default="/work/output/coef_scales.npy",
                   help="whitening scales; the stencil steps in whitened units")
    args = p.parse_args()

    f = Path(f"/work/output/minimax_p3/{args.tag}_results.jsonl")
    rows = []
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("iter") != args.iter:
            continue
        rows.append(d)

    stencil = [d for d in rows if d.get("kind") == "stencil" and d.get("converged")]
    print(f"[jac] {args.tag} iter {args.iter}: {len(stencil)} converged stencil rows")
    if len(stencil) < 10:
        print("[jac] too few rows"); return

    P = np.array([d["params"] for d in stencil], dtype=float)
    # The base point is the one row that is the unperturbed centre: every stencil
    # perturbs exactly one free coordinate, so the base is the coordinate-wise
    # median (81 of 82 rows agree on every coordinate).
    base = np.median(P, axis=0)

    free = np.array([i for i in range(90) if i not in oracle.ZERO_INDICES])
    recs = []
    for d, params in zip(stencil, P):
        dif = params - base
        nz = np.nonzero(np.abs(dif) > 0)[0]
        if len(nz) != 1:
            continue          # the base row itself, or an ambiguous row
        recs.append((int(nz[0]), float(dif[nz[0]]), d))

    print(f"[jac] identified {len(recs)} single-coordinate perturbations "
          f"of {len(free)} free coefficients")
    if not recs:
        print("[jac] could not identify perturbations; aborting"); return

    # v at the base point: use the row whose params equal base (if present),
    # else the mean of violations is not valid -- so require it.
    base_row = None
    for d, params in zip(stencil, P):
        if np.allclose(params, base, atol=0, rtol=0):
            base_row = d
            break
    if base_row is None:
        print("[jac] no exact base row in the stencil; using the nearest row")
        k = int(np.argmin(np.abs(P - base).sum(axis=1)))
        base_row = stencil[k]
    v0 = official_violations(base_row["measured_targets"])

    cols = {}
    for idx, h, d in recs:
        v = official_violations(d["measured_targets"])
        cols[idx] = (v - v0) / h

    J = np.zeros((5, 90))
    for idx, g in cols.items():
        J[:, idx] = g
    norms = np.linalg.norm(J[:, free], axis=0)
    order = np.argsort(-norms)

    tot = float(np.sum(norms ** 2))
    print(f"\n[jac] column-norm distribution over {len(free)} free coefficients")
    cum = 0.0
    marks = {}
    for rank, j in enumerate(order, 1):
        cum += float(norms[j] ** 2)
        frac = cum / tot if tot else 0.0
        for thr in (0.50, 0.80, 0.90, 0.95, 0.99):
            if thr not in marks and frac >= thr:
                marks[thr] = rank
    for thr in (0.50, 0.80, 0.90, 0.95, 0.99):
        r = marks.get(thr)
        print(f"   {int(thr*100):>3}% of squared gradient norm in the top "
              f"{r if r else '-'} coordinates")

    print(f"\n[jac] top 15 coordinates by norm:")
    for j in order[:15]:
        gi = free[j]
        m, n = divmod(int(gi) % 45, 9)
        which = "r_cos" if gi < 45 else "z_sin"
        print(f"   param {gi:>2} ({which} m={m} n={n-4:+d})  |col|={norms[j]:.4g}")

    print(f"\n[jac] bottom 15 (candidates to drop):")
    for j in order[-15:]:
        gi = free[j]
        m, n = divmod(int(gi) % 45, 9)
        which = "r_cos" if gi < 45 else "z_sin"
        print(f"   param {gi:>2} ({which} m={m} n={n-4:+d})  |col|={norms[j]:.4g}")

    # Wall-time arithmetic: evals per gradient = kept + 1, and on W workers the
    # wall cost is ceil((kept+1)/W) waves. The waves are what matters, not the
    # eval count -- 82 evals on 32 workers is 3 waves with the third mostly idle.
    print(f"\n[jac] WAVE ARITHMETIC on 32 workers (the actual wall-time unit):")
    for keep in (len(free), 63, 48, 31):
        if keep > len(free):
            continue
        waves = -(-(keep + 1) // 32)
        kept_frac = (float(np.sum(norms[order[:keep]] ** 2)) / tot) if tot else 0.0
        print(f"   keep {keep:>2} coords -> {keep+1:>2} evals -> {waves} wave(s), "
              f"retaining {100*kept_frac:.1f}% of gradient norm")
    print("\n[jac] CAVEAT: one base point, one basin. Column norms are local. This "
          "is a reduction to TEST with a paired reduced-vs-full iteration, not one "
          "to adopt on this evidence.")


if __name__ == "__main__":
    main()

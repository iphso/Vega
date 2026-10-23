"""Can a FEASIBLE P3 design even be written in our parameterisation?

THE HYPOTHESIS. Two high-fidelity runs with nothing in common -- `hf_s0b` (nfp=2,
A=8.4) and `hf_asp16` (nfp=4, A=15.3), independently seeded -- stalled at official
worst +0.080991 and +0.093147, within 0.012 of each other, both at 100% VMEC++
convergence, and both with the SAME three-way active set (qi, mirror, fluxcomp).
A barrier that lands in the same place from unrelated starting points, with the
same constraints binding, is more consistent with a property of the SEARCH SPACE
than with two coincidental local optima.

`minimax_p3.py` optimises `r_cos = params[:45].reshape(5, 9)` and
`z_sin = params[45:90].reshape(5, 9)`: poloidal m = 0..4, toroidal n = -4..+4, with
`ZERO_INDICES` removing the 9 coefficients fixed by stellarator symmetry and
normalisation. 81 free numbers. If designs that actually satisfy the five P3
constraints carry meaningful amplitude at |m| >= 5 or |n| >= 5, then feasibility is
OUTSIDE the set of shapes we can represent, every run is optimising against a wall,
and no improvement to fd_step, the trust region, or the iteration count can matter.

WHY THIS IS THE RIGHT TEST AND NOT ANOTHER LEVER. It is falsifiable in one
direction and decisive in the other, it needs no VMEC++ (so it costs nothing and
cannot be confounded by the convergence problem), and it is measured against the
ConStellaration dataset -- real designs, not our own optimised inventory, whose
resolvability is exactly what is in question.

READ-OUT, pre-registered:
  - If dataset designs satisfying all five constraints fit inside (5, 9) with
    negligible residual, the truncation is EXONERATED and the barrier is numerical
    (curvature / fd_step), which the high-fidelity fd_diagnostic addresses.
  - If they carry real amplitude outside (5, 9), the parameterisation is the
    ceiling and the fix is to widen it, not to search harder inside it.
  - If the dataset contains NO design satisfying all five, that is its own finding
    and this test is INCONCLUSIVE -- it would mean we have no positive example of
    P3 feasibility to compare against, which must be reported as such and not
    silently read as the exoneration branch.

CAVEAT. "Negligible residual" is a judgement, so both the per-mode amplitudes and
the fraction of total spectral energy outside the window are printed. A design
whose out-of-window energy is below the amplitude of its own smallest KEPT mode
cannot be distinguished from one that truly fits; that comparison is printed too
rather than asserted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# The five official normalized violations, kept in this file rather than imported
# so this script has no dependency on the oracle image.
def official_violations_from_metrics(m):
    iota = abs(m["edge_rotational_transform_over_n_field_periods"])
    return np.array([
        (0.25 - iota) / 0.25,
        (np.log10(m["qi"]) + 3.5) / 3.5,
        (m["edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (m["flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        (0.0 - m["vacuum_well"]) / 1e-1,
    ])


METRIC_KEYS = {
    "edge_rotational_transform_over_n_field_periods":
        "metrics.edge_rotational_transform_over_n_field_periods",
    "qi": "metrics.qi",
    "edge_magnetic_mirror_ratio": "metrics.edge_magnetic_mirror_ratio",
    "flux_compression_in_regions_of_bad_curvature":
        "metrics.flux_compression_in_regions_of_bad_curvature",
    "vacuum_well": "metrics.vacuum_well",
}

# Our window: r_cos / z_sin are (5, 9) == m in 0..4, n in -4..+4.
M_KEEP, N_HALF = 5, 4


def window_split(arr):
    """Return (in-window energy, out-of-window energy, max out-of-window |amp|).

    Dataset arrays are (n_poloidal, n_toroidal) with the toroidal axis centred on
    n=0, same convention as ours, so the window is a centred crop.
    """
    a = np.asarray(arr, dtype=float)
    nm, nn = a.shape
    nc = nn // 2                      # index of n = 0
    lo, hi = max(0, nc - N_HALF), min(nn, nc + N_HALF + 1)
    mask = np.zeros_like(a, dtype=bool)
    mask[:min(nm, M_KEEP), lo:hi] = True
    inside, outside = a[mask], a[~mask]
    return (float(np.sum(inside ** 2)), float(np.sum(outside ** 2)),
            float(np.abs(outside).max()) if outside.size else 0.0,
            float(np.abs(inside[inside != 0]).min()) if np.any(inside != 0) else 0.0,
            a.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-records", type=int, default=4000)
    p.add_argument("--bar", type=float, default=1e-2,
                   help="the official feasibility bar on every normalized violation")
    p.add_argument("--near", type=float, default=0.10,
                   help="also report designs merely NEAR feasible, as a fallback "
                        "population if nothing in the dataset is actually feasible")
    a = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("proxima-fusion/constellaration", split="train",
                      streaming=True)

    n_seen = n_metrics = 0
    feasible, near = [], []
    allw, allv = [], []
    for rec in ds:
        if n_seen >= a.max_records:
            break
        n_seen += 1
        if not rec.get("boundary.is_stellarator_symmetric"):
            continue
        try:
            m = {k: rec[v] for k, v in METRIC_KEYS.items()}
        except KeyError:
            print(f"[trunc] metrics not in this schema; keys sample = "
                  f"{[k for k in list(rec) if 'metric' in k][:12]}")
            return
        if any(v is None for v in m.values()):
            continue
        try:
            v = official_violations_from_metrics(m)
        except Exception:
            continue
        n_metrics += 1
        w = float(v.max())
        allw.append(w); allv.append(v)
        row = (w, v, rec["boundary.r_cos"], rec["boundary.z_sin"],
               int(rec["boundary.n_field_periods"]))
        if w <= a.bar:
            feasible.append(row)
        elif w <= a.near:
            near.append(row)

    print(f"[trunc] scanned {n_seen} records, {n_metrics} with all five metrics")
    print(f"[trunc] satisfying ALL FIVE at the {a.bar:g} bar: {len(feasible)}")
    print(f"[trunc] merely near-feasible (worst <= {a.near:g}): {len(near)}")

    # WHERE THE DATASET ACTUALLY SITS. A bare count of zero is unreadable: it is
    # consistent both with "the dataset nearly gets there" and with "the dataset
    # is nowhere near", and those imply opposite things about our own 0.081. The
    # dataset is a general stellarator corpus, not a set of P3 solutions, so this
    # distribution is the only honest way to place our designs against it.
    if allw:
        w = np.array(allw)
        qs = np.percentile(w, [0, 1, 5, 25, 50])
        print(f"\n[trunc] dataset worst-violation distribution over {len(w)} designs:")
        print(f"   min {qs[0]:.4f}   p1 {qs[1]:.4f}   p5 {qs[2]:.4f}   "
              f"p25 {qs[3]:.4f}   median {qs[4]:.4f}")
        print(f"   our best high-fidelity-admissible design: 0.0810 (hf_s0b)")
        print(f"   designs better than ours: {int((w <= 0.0810).sum())}/{len(w)}")
        pc = np.array(allv)
        print(f"\n[trunc] dataset per-constraint minima (which one is the real wall):")
        for i, n in enumerate(["iota", "qi", "mirror", "fluxcomp", "vwell"]):
            print(f"   {n:>9}  min {pc[:, i].min():+.4f}   median "
                  f"{np.median(pc[:, i]):+.4f}   frac satisfied "
                  f"{float((pc[:, i] <= a.bar).mean()):.3f}")

    pop, label = (feasible, "FEASIBLE") if feasible else (near, "NEAR-FEASIBLE")
    if not pop:
        print("[trunc] INCONCLUSIVE: no design at or near the bar in this sample. "
              "Cannot compare against a positive example; do NOT read this as "
              "exonerating the truncation.")
        return
    if not feasible:
        print("[trunc] NOTE: no dataset design clears the bar, so the comparison "
              "below uses the near-feasible population. That is weaker evidence "
              "and is reported as such -- it bounds where the physics is heading, "
              "not where feasibility sits.")

    print(f"\n[trunc] spectral content of {len(pop)} {label} designs vs our "
          f"(m<{M_KEEP}, |n|<={N_HALF}) window")
    print(f"{'worst':>9} {'nfp':>4} {'shape':>9} {'out/total':>10} "
          f"{'max|out|':>10} {'min|kept|':>10}  verdict")
    fr = []
    for w, v, rc, zs, nfp in sorted(pop, key=lambda r: r[0])[:25]:
        ri, ro, rmx, rmn, shp = window_split(rc)
        zi, zo, zmx, zmn, _ = window_split(zs)
        tot = ri + ro + zi + zo
        frac = (ro + zo) / tot if tot else 0.0
        mx, mn = max(rmx, zmx), min(x for x in (rmn, zmn) if x > 0) if (rmn or zmn) else 0.0
        fr.append(frac)
        verdict = ("fits" if mx <= mn else "HAS OUT-OF-WINDOW CONTENT")
        print(f"{w:9.5f} {nfp:>4} {str(shp):>9} {frac:10.3e} {mx:10.3e} "
              f"{mn:10.3e}  {verdict}")

    fr = np.array(fr)
    print(f"\n[trunc] out-of-window energy fraction: median {np.median(fr):.3e}, "
          f"max {fr.max():.3e}")
    if fr.max() < 1e-6:
        print("[trunc] VERDICT: the truncation is EXONERATED. Designs at this "
              "constraint level live inside our window, so the 0.081/0.093 barrier "
              "is numerical, not representational -- which points at fd_step and "
              "curvature, exactly what the high-fidelity fd_diagnostic measures.")
    else:
        print("[trunc] VERDICT: real amplitude sits OUTSIDE our window. The "
              "parameterisation is a candidate ceiling and widening it is a "
              "structural fix, not another lever. Confirm by re-running one stalled "
              "basin with a wider (m, n) before believing it.")


if __name__ == "__main__":
    main()

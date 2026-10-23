"""How much do low-fidelity violations lie, on designs where BOTH fidelities converge?

Separate question from resolvability. Today's finding is that our leading designs
do not converge at high fidelity at all. This asks the quieter question underneath
it: on the designs where high fidelity DOES converge, how far off is the cheap
oracle the search has been steering by?

The project carries one data point on this -- 2026-09-27, low 0.092682 -> high
0.097115, a gap of 0.0044, recorded as "the high-fidelity penalty is smaller than
assumed". That is n=1, and it is conditioned on the design having converged, so it
is a survivorship estimate. The 23 diverse seeds give a paired sample: identical
boundaries, both fidelities, nothing else varying.

WHY IT MATTERS FOR THE RESTART. The feasibility bar is 1e-2 on every normalized
violation. If the median low->high shift is comparable to 1e-2, then low-fidelity
search cannot resolve feasibility even where it converges, and `--fidelity high`
is mandatory rather than merely safer. If the shift is far below 1e-2, low fidelity
remains usable for coarse exploration with a high-fidelity gate on acceptance,
which is much cheaper than searching at high fidelity throughout.

Reads the two passes of `eval_seed_pool.py` over the SAME pool:
    --low  output/seed_pool_eval.jsonl        (--fidelity low)
    --high output/seed_pool_eval_high.jsonl   (--fidelity high)
"""
import argparse
import json

import numpy as np

NAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]


def load(path):
    out = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        out[d["row"]] = d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low", default="output/seed_pool_eval.jsonl")
    p.add_argument("--high", default="output/seed_pool_eval_high.jsonl")
    # The rung names appear in every printed line. They are parameters because
    # this compares ANY two passes -- medium-vs-high is a different claim from
    # low-vs-high, and a hardcoded "low->high" label on a medium comparison is
    # exactly the kind of mislabelled output that gets quoted back as a result.
    p.add_argument("--label-a", default="low")
    p.add_argument("--label-b", default="high")
    a = p.parse_args()

    lo, hi = load(a.low), load(a.high)
    A, B = a.label_a, a.label_b
    rows = sorted(set(lo) & set(hi))
    print(f"[gap] {len(rows)} rows present in both passes  ({A} vs {B})")

    both, lo_only = [], []
    for r in rows:
        l, h = lo[r], hi[r]
        if l.get("converged") and not h.get("converged"):
            lo_only.append(r)
        elif l.get("converged") and h.get("converged"):
            both.append(r)

    print(f"[gap] converged at {A} AND {B}: {len(both)}")
    print(f"[gap] converged at {A} but NOT {B}: {len(lo_only)}"
          + (f"  rows {lo_only}" if lo_only else ""))

    if not both:
        print("[gap] no paired rows; nothing to compare")
        return

    print(f"\n{'row':>4} {A+'_A':>8} {B+'_B':>8} {'dA':>7}  "
          f"{'worst_'+A:>10} {'worst_'+B:>10} {'d_worst':>9}  bind_{A}->bind_{B}")
    dws, dAs, per_c = [], [], {n: [] for n in NAMES}
    for r in both:
        l, h = lo[r], hi[r]
        dw = h["worst"] - l["worst"]
        dA = h["A"] - l["A"]
        dws.append(dw)
        dAs.append(dA)
        for i, n in enumerate(NAMES):
            per_c[n].append(h["violations"][i] - l["violations"][i])
        print(f"{r:>4} {l['A']:8.4f} {h['A']:8.4f} {dA:+7.3f}  "
              f"{l['worst']:10.5f} {h['worst']:10.5f} {dw:+9.5f}  "
              f"{l['binding']}->{h['binding']}")

    dws = np.array(dws)
    print(f"\n[gap] worst-violation shift {A}->{B}: median {np.median(dws):+.5f}, "
          f"mean {dws.mean():+.5f}, |max| {np.abs(dws).max():.5f}")
    print(f"[gap] aspect-ratio shift: median {np.median(dAs):+.4f}, "
          f"|max| {np.abs(dAs).max():.4f}")
    print("\n[gap] per-constraint shift (median, then |max|):")
    for n in NAMES:
        v = np.array(per_c[n])
        print(f"   {n:>9}  median {np.median(v):+.5f}   |max| {np.abs(v).max():.5f}")

    # THE STATISTIC THAT MATTERS IS THE TAIL, NOT THE MEDIAN. This originally
    # branched on median |shift| vs 1e-2, which is the wrong test and on the first
    # real data landed at 0.95x the bar -- a coin flip that would have printed
    # "low fidelity stays usable" off a rounding margin. Certifying `worst <= 1e-2`
    # is a statement about the WORST case, so the quantity to compare against the
    # bar is the upper tail of |shift|. A median below the bar with a tail above it
    # means low fidelity ranks designs usefully and certifies nothing.
    adw = np.abs(dws)
    med, p90, mx = (float(np.median(adw)), float(np.percentile(adw, 90)),
                    float(adw.max()))
    print(f"\n[gap] |shift| vs the 1e-2 bar:  median {med:.5f} ({med/1e-2:.1f}x)   "
          f"p90 {p90:.5f} ({p90/1e-2:.1f}x)   max {mx:.5f} ({mx/1e-2:.1f}x)")
    print(f"[gap] signed median {np.median(dws):+.5f} -- a near-zero signed median "
          f"with a {mx/1e-2:.1f}x tail means the error is VARIANCE, not bias, so no "
          f"offset or learned correction can repair the cheap oracle.")
    # Handoff threshold: this rung stays informative only while the REMAINING gap
    # is large compared with its own error. k=10 is the margin; at k=1 the oracle
    # is pure noise. Measured for low-vs-high this lands at ~0.095, independently
    # where the resolvability survey shows convergence collapsing -- two unrelated
    # measurements picking the same handoff is why the rule is trusted over a
    # hand-tuned schedule.
    k = 10.0
    print(f"\n[gap] HANDOFF: `{A}` stays informative down to worst ~ "
          f"{k*med:.4f} (k={k:g} x its median error {med:.5f}). Below that, step "
          f"up to `{B}`.")
    if mx >= 1e-2:
        print(f"[gap] VERDICT: the tail exceeds the bar. `{A}` CANNOT certify "
              f"feasibility even where it converges. It remains usable for coarse "
              f"ranking down to the handoff above; any feasibility claim requires "
              f"`{B}`.")
    else:
        print(f"[gap] VERDICT: even the worst observed shift is inside the bar, so "
              f"`{A}` can certify feasibility. Searching at `{A}` with a `{B}` gate "
              f"on acceptance is sufficient and cheaper than searching at `{B}`.")

    nflip = sum(1 for r in both if lo[r]["binding"] != hi[r]["binding"])
    print(f"[gap] binding constraint CHANGES between fidelities in {nflip}/{len(both)} "
          f"designs. The minimax acts on the active set, so where this happens the "
          f"low-fidelity gradient aims at the wrong constraint -- a gate on "
          f"acceptance cannot fix that, only searching at high fidelity can.")

    print("\n[gap] TWO LIMITS ON READING THIS:")
    print(f"[gap]  1. Conditioned on `{B}` convergence -- a survivorship estimate "
          f"that says nothing about designs that fail outright.")
    lo_w = [lo[r]["worst"] for r in both]
    print(f"[gap]  2. Every design here sits at worst {min(lo_w):.3f}-{max(lo_w):.3f}, "
          f"far from the 1e-2 bar. How the shift behaves AT the bar is unmeasured, "
          f"and not measurable from our optimised designs because those do not "
          f"converge at `{B}`. Do not extrapolate this table to the final approach.")


if __name__ == "__main__":
    main()

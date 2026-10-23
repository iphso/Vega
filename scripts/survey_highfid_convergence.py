"""WHERE is our design inventory resolvable by the official scorer?

THE FINDING THIS ANSWERS. Every leading design in the project fails the real
scorer with `RuntimeError: VMEC++ did not converge` -- `asp_a13.5`'s three best
(low-fidelity official worst 0.006549/0.007144/0.007602, all five constraints
satisfied) and `ex_floor2_r21`, the 1M-eval leader. Four straight ConStellaration
dataset designs run through the identical call converge fine, so the harness is
sound and the boundaries are the problem. ~1M evaluations were spent optimising
against `low_fidelity`, and the search has evidently been mining the gap between
it and the fidelity that scores.

Mechanism: the minimax has 81 free Fourier coefficients and NOTHING penalising
high-order content or resolution sensitivity. Hard optimisation against a coarse
solver is how you get boundaries that only exist at coarse resolution.

WHAT THIS MEASURES. High-fidelity convergence RATE, stratified by aspect ratio and
by how far the design was optimised (low-fidelity worst violation). The strata are
the point: a flat rate means our whole inventory is unusable and the oracle has to
change. A rate that falls as designs get better optimised is direct evidence that
the search itself destroys resolvability -- the most important thing we could learn,
because it means every future run needs a convergence guard, not a better lever.

WHY THE ORACLE AND NOT `problem.evaluate()`. `problem.evaluate()` is serial, and a
rate needs hundreds of samples. The scorer's traceback runs
`problems.py -> forward_model.forward_model`, which is the same function
`vmec_oracle.worker_fn` calls, and
`ConstellarationSettings.default_high_fidelity()` was checked field-for-field
against the oracle's `VmecPresetSettings(fidelity="high_fidelity")`: 0 differing
fields of 12. So oracle-high convergence IS scorer convergence, and it parallelises
over 32 cores. This equivalence is the one assumption here; it was verified rather
than assumed, which is the whole reason today's result was caught at all.
"""
import argparse
import glob as globmod
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from minimax_p3 import official_violations  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

A_BINS = [0, 8, 10, 12, 13, 14, 100]
# The transition is in the violation axis around worst ~ 0.1, so that region needs
# resolution. The first run used edges [.., 0.05, 0.10, 0.20, ..] and the 0.10 edge
# manufactured an apparent "high-aspect designs resolve" exception: all six designs
# in that cell sat at 0.0990-0.0999, i.e. they were 0.1 designs filed as better
# than 0.1. Finer edges here so a bin boundary cannot fake a feature again.
W_BINS = [0, 0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 1e9]


def binof(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="/work/output/minimax_p3/*_results.jsonl")
    p.add_argument("--per-cell", type=int, default=6,
                   help="samples per (aspect bin x violation bin) cell")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", default="/work/output/highfid_convergence_survey.jsonl")
    a = p.parse_args()

    # Stratified reservoir: walk every row once, keep the first --per-cell distinct
    # designs per cell. Taking the BEST per cell would bias the survey toward
    # exactly the over-optimised designs whose resolvability is in question.
    cells = defaultdict(list)
    seen = set()
    nrows = 0
    for f in sorted(globmod.glob(a.glob)):
        for line in open(f):
            nrows += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not d.get("converged"):
                continue
            mt, params = d.get("measured_targets"), d.get("params")
            if not isinstance(mt, dict) or params is None:
                continue
            try:
                w = float(official_violations(mt).max())
                A = float(mt["aspect_ratio"])
            except Exception:
                continue
            key = (binof(A, A_BINS), binof(w, W_BINS))
            if len(cells[key]) >= a.per_cell:
                continue
            h = hash(tuple(np.round(np.asarray(params, dtype=float), 8)))
            if h in seen:
                continue
            seen.add(h)
            cells[key].append((np.asarray(params, dtype=float), int(d["nfp"]),
                               A, w, d.get("tag", "?")))

    jobs, meta = [], {}
    for key, items in sorted(cells.items()):
        for params, nfp, A, w, tag in items:
            x = params.copy()
            x[oracle.ZERO_INDICES] = 0.0
            rc, zs, nfp_i, fn = oracle.params_to_worker_args(
                x, {"nfp": nfp}, oracle.FIDELITY_PRESETS["high"])
            t = len(jobs)
            jobs.append((t, rc, zs, nfp_i, fn))
            meta[t] = (key, A, w, tag, nfp)

    print(f"[srv] scanned {nrows} rows -> {len(jobs)} designs across "
          f"{len(cells)} cells, at HIGH fidelity on {a.workers} workers", flush=True)

    stats = defaultdict(lambda: [0, 0])   # cell -> [converged, total]
    rows = []
    done = 0
    for t, ok, payload in run_batch_with_timeout(
            jobs, oracle.worker_fn, a.workers, a.timeout):
        key, A, w, tag, nfp = meta[t]
        conv = bool(ok) and all(
            payload.get(n) is not None for n in oracle.TARGET_NAMES)
        stats[key][1] += 1
        stats[key][0] += int(conv)
        rows.append({"cell_A": key[0], "cell_w": key[1], "A_low": A,
                     "worst_low": w, "tag": tag, "nfp": nfp,
                     "high_converged": conv,
                     "error": None if conv else str(payload)[:200]})
        done += 1
        if done % 20 == 0:
            tot = sum(s[1] for s in stats.values())
            cv = sum(s[0] for s in stats.values())
            print(f"[srv] {done}/{len(jobs)}  running rate {cv}/{tot}", flush=True)

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print("\n[srv] HIGH-FIDELITY CONVERGENCE RATE  (rows = aspect ratio, "
          "cols = low-fidelity worst violation)", flush=True)
    hdr = "  ".join(f"{W_BINS[j]:g}-{W_BINS[j+1]:g}".rjust(11)
                    for j in range(len(W_BINS) - 1))
    print(f"{'aspect':>12}  {hdr}", flush=True)
    for i in range(len(A_BINS) - 1):
        cs = []
        for j in range(len(W_BINS) - 1):
            c, n = stats.get((i, j), [0, 0])
            cs.append((f"{c}/{n}" if n else "-").rjust(11))
        print(f"{f'{A_BINS[i]:g}-{A_BINS[i+1]:g}':>12}  " + "  ".join(cs), flush=True)

    tot = sum(s[1] for s in stats.values())
    cv = sum(s[0] for s in stats.values())
    print(f"\n[srv] overall {cv}/{tot} = {100.0*cv/max(tot,1):.1f}% converge at the "
          f"fidelity that scores", flush=True)
    print("[srv] read: a rate that FALLS as worst-violation falls means the search "
          "itself destroys resolvability, and every run needs a convergence guard.",
          flush=True)


if __name__ == "__main__":
    main()

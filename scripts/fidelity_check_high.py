"""Is HIGH-fidelity VMEC++ deterministic, and what does it cost?

WHY. The project's answer to "are finite-difference gradients sound?" is
`fidelity_check.py`: two independent `low_fidelity` re-evaluations of an identical
boundary returned worst_violation identical to all 17 significant digits, so there
is no noise floor to fight. That result is real and it is the basis of every
gradient run here.

It was measured at LOW fidelity only. Reading that file: it issues `low_a`,
`low_b` AND `high` -- two low-fidelity re-evals and exactly ONE high-fidelity
eval. So high-fidelity determinism has never been tested, and one eval cannot
test it.

This matters right now because the fix for the fidelity-gap finding (2026-10-05)
is to search at the fidelity that scores -- `minimax_p3.py --fidelity high`, which
needs no new code. That fix inherits the FD-soundness argument, and inheriting a
property verified at one fidelity is exactly the mistake that produced the
artifact in the first place ("low fidelity is deterministic" and "the
high-fidelity penalty is small" were both true and neither was the relevant
property). So: measure it, do not assume it.

ALSO MEASURES COST, which decides whether the fix is affordable. 81 free
coefficients means 82 evals per gradient. At low fidelity that is ~30 s on 32
cores and a 40-iteration run is hours. The high/low wall-time ratio turns that
into the real budget.

CONTROL BOUNDARIES come from the ConStellaration dataset, which is
VMEC-converged by construction -- our own leading designs cannot be used here
because they do not converge at high fidelity at all, which is the finding that
prompted this. Determinism is a property of the solver, not of the design, so a
dataset boundary tests it correctly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

KEYS = ["qi", "aspect_ratio", "edge_rotational_transform_over_n_field_periods",
        "edge_magnetic_mirror_ratio",
        "flux_compression_in_regions_of_bad_curvature", "vacuum_well",
        "minimum_normalized_magnetic_gradient_scale_length"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-designs", type=int, default=2)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default="/work/output/fidelity_check_high.json")
    a = p.parse_args()

    from datasets import load_dataset
    print("[fid] pulling dataset control boundaries", flush=True)
    ds = load_dataset("proxima-fusion/constellaration", split="train",
                      streaming=True)
    bnds = []
    for rec in ds:
        if len(bnds) >= a.n_designs:
            break
        if not rec["boundary.is_stellarator_symmetric"]:
            continue
        bnds.append((np.asarray(rec["boundary.r_cos"], dtype=float),
                     np.asarray(rec["boundary.z_sin"], dtype=float),
                     int(rec["boundary.n_field_periods"])))

    # Two independent re-evals of each boundary at HIGH fidelity (the determinism
    # test proper), plus one at low fidelity for the cost ratio. Run them all in
    # ONE batch so they share machine conditions.
    jobs = []
    for i, (rc, zs, nfp) in enumerate(bnds):
        jobs.append((f"d{i}_high_a", rc, zs, nfp, "high_fidelity"))
        jobs.append((f"d{i}_high_b", rc, zs, nfp, "high_fidelity"))
        jobs.append((f"d{i}_low", rc, zs, nfp, "low_fidelity"))

    print(f"[fid] {len(jobs)} evals ({len(bnds)} boundaries x high,high,low)",
          flush=True)
    got, times = {}, {}
    t_batch = time.time()
    for tag, ok, payload in run_batch_with_timeout(
            jobs, oracle.worker_fn, len(jobs), a.timeout):
        got[tag] = payload if ok else None
        times[tag] = time.time() - t_batch
        print(f"[fid]   {tag}: {'ok' if ok else 'FAILED ' + str(payload)[:80]}",
              flush=True)

    report = {"determinism": {}, "cost": {}}
    print("\n[fid] HIGH-FIDELITY DETERMINISM (two independent re-evals, "
          "identical boundary)", flush=True)
    for i in range(len(bnds)):
        A, B = got.get(f"d{i}_high_a"), got.get(f"d{i}_high_b")
        if A is None or B is None:
            print(f"  d{i}: a control boundary failed at high fidelity "
                  f"(a={A is not None} b={B is not None}) -- cannot test", flush=True)
            report["determinism"][f"d{i}"] = "control failed"
            continue
        worst_rel = 0.0
        bad = []
        for k in KEYS:
            va, vb = A.get(k), B.get(k)
            if va is None or vb is None:
                continue
            if va == vb:
                continue
            rel = abs(va - vb) / max(abs(va), 1e-300)
            worst_rel = max(worst_rel, rel)
            bad.append(f"{k}: {va!r} vs {vb!r} (rel {rel:.3e})")
        if not bad:
            print(f"  d{i}: BIT-IDENTICAL across all {len(KEYS)} metrics", flush=True)
            report["determinism"][f"d{i}"] = {"bit_identical": True}
        else:
            print(f"  d{i}: NOT bit-identical -- worst relative difference "
                  f"{worst_rel:.3e}", flush=True)
            for s in bad:
                print(f"       {s}", flush=True)
            report["determinism"][f"d{i}"] = {"bit_identical": False,
                                              "worst_rel": worst_rel,
                                              "fields": bad}

    print("\n[fid] COST. Wall time is per-batch here (all evals ran concurrently), "
          "so these are upper bounds on single-eval latency, not throughput.",
          flush=True)
    for i in range(len(bnds)):
        th = max(times.get(f"d{i}_high_a", 0), times.get(f"d{i}_high_b", 0))
        tl = times.get(f"d{i}_low", 0)
        ratio = th / tl if tl else float("nan")
        print(f"  d{i}: high {th:7.1f}s   low {tl:7.1f}s   ratio {ratio:5.1f}x",
              flush=True)
        report["cost"][f"d{i}"] = {"high_s": th, "low_s": tl, "ratio": ratio}

    print("\n[fid] READ: if high fidelity is NOT bit-identical, finite-difference "
          "gradients at high fidelity have a noise floor, the step size h must be "
          "set above it, and `--fidelity high` is NOT a drop-in fix.", flush=True)
    Path(a.out.replace("/work", ".")).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[fid] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

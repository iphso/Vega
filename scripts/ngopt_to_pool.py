"""One-off/repeatable: turn an ngopt_alm_p3.py run's real-VMEC-labeled
candidates into the same {X,Y} pool schema every other real-data source in
this project uses (output/X.npy's own 92-col schema: 90 coefficients, nfp,
a trailing 1.0 -- see output/X.npy row 0 / scripts/bootstrap_loop.py's
anchor-pool loading), plus a matching rejected_X/rejected_reason pair for
the non-converged candidates.

Why this is worth doing at all: ngopt_alm_p3's ALM+NGOpt search starts from
a single near-wall seed and does *local* gradient-free search around
whatever point currently has the lowest worst-violation. That's a
fundamentally different sampling process from the bootstrap loop's VAE-
decode-from-latent-prior sampling -- it's a directed local walk through a
specific corner of coefficient space (nfp=2, aspect_ratio~8.2, right at the
P3 wall) that the VAE's learned prior may never visit on its own, densely
labeled by real VMEC++ along the way, including exactly which nearby points
are and are not physically realizable. Feeding it into the bootstrap pool
doesn't replace the bootstrap loop's own exploration -- it's a second,
independently-sourced sample of a well the VAE-based sampler wasn't
necessarily covering, added as anchor-quality real data for the next
retrain, same as any other real_pool merge in this project
(merge_all_real_pools.py's role for the grounded-walk lanes).

Usage:
    python scripts/ngopt_to_pool.py --tag ngopt_v2 --out-tag ngopt_v2_well

Output: output/bootstrap_seed_<out-tag>/{X,Y}.npy (converged, ready to pass
as an extra --anchor-pool / merge target) and
{rejected_X,rejected_reason}.npy (non-converged, reason always "vmec" since
ngopt_alm_p3 doesn't distinguish structural/timeout at the row level).
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")
TARGET_NAMES = json.loads((OUT_DIR / "target_names.json").read_text())
REJECT_REASON_CODES = {"structural": 0, "vmec": 1, "timeout": 2, "unknown": 3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="ngopt_alm_p3.py --tag whose results.jsonl to read")
    p.add_argument("--out-tag", default=None, help="defaults to --tag")
    p.add_argument("--results-dir", default=str(OUT_DIR / "ngopt_alm_p3"))
    args = p.parse_args()
    out_tag = args.out_tag or args.tag

    results_path = Path(args.results_dir) / f"{args.tag}_results.jsonl"
    rows = [json.loads(l) for l in results_path.open() if l.strip()]
    if not rows or "params" not in rows[0]:
        raise SystemExit(f"{results_path} has no per-row 'params' field -- this run predates the "
                          f"params-logging fix, nothing to merge (see EXPERIMENT_LOG).")

    conv_X, conv_Y, rej_X, rej_reason = [], [], [], []
    for r in rows:
        params = np.array(r["params"], dtype=np.float64)
        row_x = np.concatenate([params, [float(r["nfp"])], [1.0]])
        if r["converged"]:
            y = r["measured_targets"]
            conv_X.append(row_x)
            conv_Y.append([y[n] for n in TARGET_NAMES])
        else:
            rej_X.append(row_x)
            rej_reason.append(REJECT_REASON_CODES["vmec"])

    out_dir = OUT_DIR / f"bootstrap_seed_{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    conv_X = np.array(conv_X) if conv_X else np.zeros((0, 92))
    conv_Y = np.array(conv_Y) if conv_Y else np.zeros((0, len(TARGET_NAMES)))
    rej_X = np.array(rej_X) if rej_X else np.zeros((0, 92))
    rej_reason = np.array(rej_reason, dtype=np.int64)

    np.save(out_dir / "X.npy", conv_X)
    np.save(out_dir / "Y.npy", conv_Y)
    np.save(out_dir / "rejected_X.npy", rej_X)
    np.save(out_dir / "rejected_reason.npy", rej_reason)

    print(f"[ngopt_to_pool] {args.tag}: {len(conv_X)} converged (real 'well' anchors), "
          f"{len(rej_X)} rejected (real negatives), from {len(rows)} total rows")
    print(f"       -> {out_dir}/")
    if len(conv_Y):
        qi = conv_Y[:, TARGET_NAMES.index("qi")]
        mr = conv_Y[:, TARGET_NAMES.index("edge_magnetic_mirror_ratio")]
        print(f"       qi range {qi.min():.6g}-{qi.max():.6g}, "
              f"mirror_ratio range {mr.min():.4f}-{mr.max():.4f}")


if __name__ == "__main__":
    main()

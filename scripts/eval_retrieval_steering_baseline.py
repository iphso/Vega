"""Zero-training, zero-oracle-call baseline for the steerability question in
eval_cvae_steerability.py: instead of decoding a candidate from the cVAE,
just look up the real dataset row closest (by the same z-scored/log-
transformed target distance the model conditions on) to the *requested,
perturbed* target spec, and check whether that neighbor's own already-known
real metric differs from the anchor's in the requested direction.

This isn't testing generation at all -- it's asking how much of the cVAE's
correct-direction rate is actually attributable to the model, versus simply
reflecting that real designs near each other in target space naturally
differ from each other in roughly the expected way (the same kind of local
structure the §16 retrieval-precision comparison leaned on, reused here for
a different question). No VMEC++ calls needed: every neighbor is a real,
already-labeled row, and §17's random-sample check already established that
~95% of typical rows are legitimate -- rerunning them through the oracle
would just be re-confirming that, not answering this baseline's question.

Uses the SAME anchor selection (random, same seed) and SAME z-scored eval
space (log-transform LOG_TARGET_NAMES, z-score with a --tag checkpoint's own
target_mean/target_std) as eval_cvae_steerability.py, so results line up
against a same-tag/same-seed cVAE run directly, target by target.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_cvae_steerability import eval_space, random_anchor_sample
from make_splits import LOG_TARGET_NAMES

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="cvae_targets_full_s0", help="checkpoint to borrow target_mean/target_std/anchor space from")
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--step-std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="retrieval_steering")
    args = p.parse_args()

    ckpt = torch.load(CKPT_DIR / f"{args.tag}.pt", map_location="cpu")
    target_names = ckpt["target_names"]
    n_targets = len(target_names)
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]

    X = np.load(OUT_DIR / "X.npy")
    Y = np.load(OUT_DIR / "Y.npy")
    Ye = eval_space(Y, target_names)
    Yz = (Ye - t_mean) / t_std

    anchor_idx = random_anchor_sample(Yz, args.n_anchors, seed=args.seed)
    print(f"[{args.out_tag}] {args.n_anchors} anchors (same random/seed={args.seed} selection as "
          f"eval_cvae_steerability.py) -- retrieval-only, no oracle calls")

    steer_by_dim = {d: {"n": 0, "correct": 0} for d in range(n_targets)}
    per_anchor = []
    for a_i, idx in enumerate(anchor_idx):
        anchor_z = Yz[idx]
        anchor_e = Ye[idx]
        directions = []
        for d in range(n_targets):
            query_z = anchor_z.copy()
            query_z[d] -= args.step_std
            dists = np.linalg.norm(Yz - query_z[None, :], axis=1)
            dists[idx] = np.inf  # exclude the anchor itself
            nn_idx = int(np.argmin(dists))
            achieved_delta = float(Ye[nn_idx, d] - anchor_e[d])
            correct = achieved_delta < 0
            steer_by_dim[d]["n"] += 1
            steer_by_dim[d]["correct"] += int(correct)
            directions.append({
                "target": target_names[d], "neighbor_row_index": nn_idx,
                "neighbor_distance_z_space": float(dists[nn_idx]),
                "achieved_delta_eval_space": achieved_delta, "correct_direction": correct,
            })
        per_anchor.append({"row_index": int(idx), "directions": directions})

    print(f"\n=== {args.out_tag}: {args.n_anchors} anchors, nearest-neighbor retrieval (no generation) ===")
    print(f"  {'target':55s} {'correct dir':>12s}")
    tot_n, tot_c = 0, 0
    for d in range(n_targets):
        s = steer_by_dim[d]
        rate = s["correct"] / max(s["n"], 1)
        tot_n += s["n"]
        tot_c += s["correct"]
        print(f"  {target_names[d]:55s} {rate:11.1%}")
    print(f"\n  overall: {tot_c}/{tot_n} = {tot_c/max(tot_n,1):.1%}")

    out = {
        "out_tag": args.out_tag, "tag": args.tag, "n_anchors": args.n_anchors, "step_std": args.step_std,
        "seed": args.seed,
        "overall_correct_direction_rate": tot_c / max(tot_n, 1),
        "steer_by_target": {
            target_names[d]: steer_by_dim[d]["correct"] / max(steer_by_dim[d]["n"], 1) for d in range(n_targets)
        },
        "anchors": per_anchor,
    }
    out_path = OUT_DIR / f"eval_{args.out_tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

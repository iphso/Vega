"""One-off: build a curriculum-augmented copy of an existing split by
appending real, VMEC++-converged (but not necessarily problem-feasible)
oracle_candidates_master.jsonl rows onto the TRAIN partition only -- val/test
stay untouched, copied verbatim, so evaluation against them stays honest
(these new rows are genuinely novel real data near the P2/P3 constraint
boundary, not something the model could have memorized any other way, so
leaking them into val/test would defeat the point of testing on them).

Deliberately narrow / one-off (not a general "retrain on master" tool):
selects exactly the rows this round's curriculum search produced, by
--labels (the jsonl stem tags append_oracle_master.py stored per row),
requires converged=True (guarantees measured_targets is present and
finite -- see append_oracle_master.py), and does not filter by problem
feasibility at all -- keeping real-but-off-target points is the entire
point (see conversation: "just as a quick signal of feasibility").
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-split", default="cluster")
    p.add_argument("--out-split", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="oracle_candidates_master.jsonl 'label' values to pull in")
    p.add_argument("--master", default=str(OUT_DIR / "oracle_candidates_master.jsonl"))
    args = p.parse_args()

    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    labels = set(args.labels)

    new_X, new_Y = [], []
    seen = set()  # (label, rank) -- master file is append-only, could have dup runs
    with open(args.master) as f:
        for line in f:
            d = json.loads(line)
            if d.get("label") not in labels or not d.get("converged"):
                continue
            key = (d["label"], d["rank"])
            if key in seen:
                continue
            seen.add(key)
            r_cos, z_sin = d["r_cos"], d["z_sin"]
            nfp, sym = d["n_field_periods"], d.get("is_stellarator_symmetric", 1.0)
            new_X.append(np.array(r_cos + z_sin + [nfp, sym], dtype=np.float32))
            mt = d["measured_targets"]
            new_Y.append(np.array([mt[n] for n in target_names], dtype=np.float32))

    new_X, new_Y = np.stack(new_X), np.stack(new_Y)
    print(f"pulled {len(new_X)} real-converged rows from labels={sorted(labels)}")

    base_dir = OUT_DIR / "splits" / args.base_split
    out_dir = OUT_DIR / "splits" / args.out_split
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(base_dir / "train.npz")
    X_train = np.concatenate([train["X"], new_X], axis=0)
    Y_train = np.concatenate([train["Y"], new_Y], axis=0)
    np.savez(out_dir / "train.npz", X=X_train, Y=Y_train)
    print(f"train: {train['X'].shape[0]} -> {X_train.shape[0]} rows")

    for name in ("val", "test"):
        d = np.load(base_dir / f"{name}.npz")
        np.savez(out_dir / f"{name}.npz", X=d["X"], Y=d["Y"])
    print(f"val/test copied unchanged from {args.base_split}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

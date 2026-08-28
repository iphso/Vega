"""Small unattended orchestrator for eval_random_direction_steerability.py,
same resumable/running-summary pattern as run_ablation.py. Grid: 3 models x
{targeted, null} at the standard step_std=1.0, low fidelity -- a first pass
at arbitrary-direction alignment, not a full magnitude sweep (kept modest on
purpose; extend by adding step_std values to STEP_STDS below once this first
read is in, the same way run_ablation.py's own sweep grew incrementally).
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from eval_random_direction_steerability import run_eval

OUT_DIR = Path("/work/output")

MODEL_TAGS = {
    "cvae": "cvae_targets_full_s0",
    "diffusion": "diffusion_targets_full_s0",
    "gan": "gan_targets_full_s0",
}
MODELS = ["cvae", "diffusion", "gan"]
STEP_STDS = [1.0]


def build_configs(n_anchors, k_validity, n_directions, k_per_direction, seed):
    configs = []
    for model in MODELS:
        tag = MODEL_TAGS[model]
        base = dict(model_type=model, tag=tag, n_anchors=n_anchors, anchor_selection="random",
                    k_validity=k_validity, n_directions=n_directions, k_per_direction=k_per_direction,
                    seed=seed, n_workers=24, timeout_seconds=90.0, fidelity="low")
        for step in STEP_STDS:
            step_tag = f"{step:.1f}".replace(".", "")
            out_tag = f"randdir_{model}_step{step_tag}_s{seed}"
            configs.append((out_tag, SimpleNamespace(**base, step_std=step, null=False, out_tag=out_tag)))
            out_tag_null = f"randdir_{model}_null_s{seed}"
            configs.append((out_tag_null, SimpleNamespace(**base, step_std=step, null=True, out_tag=out_tag_null)))
    return configs


def summarize(results_by_tag):
    summary = {}
    for tag, out in results_by_tag.items():
        summary[tag] = {
            "model_type": out["model_type"], "step_std": out["step_std"], "null": out["null"],
            "validity_hit_rate": out["overall_validity_hit_rate"],
            "steer_candidate_convergence_rate": out["steer_candidate_convergence_rate"],
            "mean_cosine_similarity": out["mean_cosine_similarity"],
            "median_cosine_similarity": out["median_cosine_similarity"],
            "n_judgeable": out["n_judgeable"],
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--n-directions", type=int, default=11)
    p.add_argument("--k-per-direction", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    configs = build_configs(args.n_anchors, args.k_validity, args.n_directions, args.k_per_direction, args.seed)
    print(f"[randdir-ablation] {len(configs)} configs queued: {len(MODELS)} models x "
          f"{len(STEP_STDS)} step size(s) x {{targeted, null}}")

    results_by_tag = {}
    summary_path = OUT_DIR / "random_direction_ablation_summary.json"
    t_start = time.perf_counter()

    for i, (out_tag, run_args) in enumerate(configs, 1):
        out_path = OUT_DIR / f"eval_random_direction_{out_tag}.json"
        if out_path.exists() and not args.force:
            print(f"\n[randdir-ablation] ({i}/{len(configs)}) {out_tag}: already exists, skipping")
            results_by_tag[out_tag] = json.loads(out_path.read_text())
            continue

        elapsed = time.perf_counter() - t_start
        print(f"\n[randdir-ablation] ({i}/{len(configs)}) {out_tag}: starting  "
              f"(model={run_args.model_type} step_std={run_args.step_std} null={run_args.null})  "
              f"[{elapsed/60:.1f}min elapsed so far]")
        try:
            out = run_eval(run_args)
            results_by_tag[out_tag] = out
        except Exception as e:
            print(f"[randdir-ablation] ({i}/{len(configs)}) {out_tag}: FAILED -- {type(e).__name__}: {e}")
            continue

        summary_path.write_text(json.dumps(summarize(results_by_tag), indent=2))
        print(f"[randdir-ablation] updated {summary_path} ({len(results_by_tag)}/{len(configs)} done)")

    total_elapsed = time.perf_counter() - t_start
    summary = summarize(results_by_tag)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[randdir-ablation] done. {len(results_by_tag)}/{len(configs)} configs in {total_elapsed/60:.1f} min.\n")
    print(f"{'config':30s} {'validity':>10s} {'steer-conv':>11s} {'mean cos':>10s} {'median cos':>11s}")
    for tag, s in summary.items():
        mc = f"{s['mean_cosine_similarity']:.3f}" if s['mean_cosine_similarity'] is not None else "n/a"
        mdc = f"{s['median_cosine_similarity']:.3f}" if s['median_cosine_similarity'] is not None else "n/a"
        print(f"{tag:30s} {s['validity_hit_rate']:9.1%}  {s['steer_candidate_convergence_rate']:10.1%}  {mc:>10s}  {mdc:>11s}")
    print(f"\nfull summary saved to {summary_path}")


if __name__ == "__main__":
    main()

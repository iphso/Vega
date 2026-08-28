"""Unattended orchestrator for a step-magnitude x model-architecture ablation
using eval_cvae_steerability.py's run_eval() directly (in-process, not
shelled out per config -- avoids Docker-in-Docker and re-paying model-load
time for every point in the grid).

NOTE: an earlier version of this script had `ablation_summary.json`
overwritten with only the CURRENT invocation's configs, silently clobbering
a prior run's summary (caught when a --seed 1 rerun wiped the --seed 0
summary -- the per-config eval_cvae_steerability_ablation_*.json files were
all still on disk, only the aggregated summary was lost). Fixed:
summarize_all_on_disk() now rebuilds the summary from every matching file
on disk each time, not just the configs this invocation touched.

Grid (see CONFIGS below): all 3 model types x a step_std sweep at low
fidelity, one medium-fidelity confirmation per model at step_std=1.0, one
null/chance-level control per model (null mode doesn't actually depend on
step_std -- see eval_cvae_steerability.py's docstring -- so one run per
model covers it regardless of how many step sizes are swept).

Resumable: before running a config, checks whether its output file already
exists and skips it unless --force -- a long unattended run that gets
killed partway (or is deliberately re-launched to extend the grid) picks up
where it left off rather than re-paying already-done oracle time. Writes a
running consolidated summary (output/ablation_summary.json) after every
single config completes, not just at the end, so partial results are always
inspectable while this is still running.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from eval_cvae_steerability import run_eval

OUT_DIR = Path("/work/output")

MODEL_TAGS = {
    "cvae": "cvae_targets_full_s0",
    "diffusion": "diffusion_targets_full_s0",
    "gan": "gan_targets_full_s0",
}

STEP_STDS = [0.5, 1.0, 1.5, 2.0]
MODELS = ["cvae", "diffusion", "gan"]


def build_configs(n_anchors, k_validity, k_steer, seed):
    """Returns a list of (out_tag, args_namespace) -- the grid this ablation covers.
    out_tag always carries the seed (ablation_..._s{seed}) -- a rerun at a
    different --seed is a genuine second data point, not an overwrite of the
    first; the seed-0 vs seed-1 mismatch on the null baseline is exactly what
    motivated checking this in the first place (see EXPERIMENT_LOG §19)."""
    configs = []
    for model in MODELS:
        tag = MODEL_TAGS[model]
        base = dict(model_type=model, tag=tag, n_anchors=n_anchors, anchor_selection="random",
                    k_validity=k_validity, k_steer=k_steer, seed=seed,
                    n_workers=24, timeout_seconds=90.0)

        # step-magnitude sweep, low fidelity, targeted mode
        for step in STEP_STDS:
            out_tag = f"ablation_{model}_step{step:.1f}_low_s{seed}".replace(".", "", 1)
            args = SimpleNamespace(**base, step_std=step, steer_mode="targeted",
                                    fidelity="low", out_tag=out_tag)
            configs.append((out_tag, args))

        # one medium-fidelity confirmation, step_std=1.0
        out_tag = f"ablation_{model}_step10_medium_s{seed}"
        args = SimpleNamespace(**base, step_std=1.0, steer_mode="targeted",
                                fidelity="medium", out_tag=out_tag)
        configs.append((out_tag, args))

        # one chance-level control (step_std value doesn't matter in null mode)
        out_tag = f"ablation_{model}_null_low_s{seed}"
        args = SimpleNamespace(**base, step_std=1.0, steer_mode="null",
                                fidelity="low", out_tag=out_tag)
        configs.append((out_tag, args))

    return configs


def summarize(results_by_tag):
    """Aggregate correct-direction / convergence / selectivity across the 11
    per-target rows into one number per config, matching how every table in
    EXPERIMENT_LOG §17-19 was hand-computed from the raw JSON -- done here
    once, consistently, instead of per-analysis by hand."""
    summary = {}
    for tag, out in results_by_tag.items():
        by_target = out["steer_by_target"]
        judgeable = [v["n_judgeable"] for v in by_target.values()]
        correct = [round(v["correct_direction_rate"] * v["n_judgeable"]) for v in by_target.values() if v["n_judgeable"]]
        tot_j, tot_c = sum(judgeable), sum(correct)
        summary[tag] = {
            "model_type": out["model_type"], "step_std": out["step_std"], "steer_mode": out["steer_mode"],
            "fidelity": out["fidelity"], "validity_hit_rate": out["overall_validity_hit_rate"],
            "correct_direction_rate": tot_c / tot_j if tot_j else None,
            "n_judgeable": tot_j,
            "selectivity_ratio": out["overall_selectivity"]["selectivity_ratio"],
        }
    return summary


def summarize_all_on_disk():
    """Rebuilds the summary from every eval_cvae_steerability_ablation_*.json
    on disk, not just whatever this invocation ran -- see the module
    docstring for why (a prior version silently lost --seed 0's summary
    when --seed 1 ran)."""
    results_by_tag = {}
    for path in sorted(OUT_DIR.glob("eval_cvae_steerability_ablation_*.json")):
        tag = path.stem.removeprefix("eval_cvae_steerability_")
        results_by_tag[tag] = json.loads(path.read_text())
    return summarize(results_by_tag)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--k-steer", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rerun configs even if their output file already exists")
    args = p.parse_args()

    configs = build_configs(args.n_anchors, args.k_validity, args.k_steer, args.seed)
    print(f"[ablation] {len(configs)} configs queued: {len(MODELS)} models x "
          f"({len(STEP_STDS)} step sizes + 1 medium-fidelity confirmation + 1 null control)")

    results_by_tag = {}
    summary_path = OUT_DIR / "ablation_summary.json"
    t_start = time.perf_counter()

    for i, (out_tag, run_args) in enumerate(configs, 1):
        out_path = OUT_DIR / f"eval_cvae_steerability_{out_tag}.json"
        if out_path.exists() and not args.force:
            print(f"\n[ablation] ({i}/{len(configs)}) {out_tag}: already exists, skipping (--force to rerun)")
            results_by_tag[out_tag] = json.loads(out_path.read_text())
            continue

        elapsed = time.perf_counter() - t_start
        print(f"\n[ablation] ({i}/{len(configs)}) {out_tag}: starting  "
              f"(model={run_args.model_type} step_std={run_args.step_std} mode={run_args.steer_mode} "
              f"fidelity={run_args.fidelity})  [{elapsed/60:.1f}min elapsed so far]")
        try:
            out = run_eval(run_args)
            results_by_tag[out_tag] = out
        except Exception as e:
            print(f"[ablation] ({i}/{len(configs)}) {out_tag}: FAILED -- {type(e).__name__}: {e}")
            continue

        summary_path.write_text(json.dumps(summarize_all_on_disk(), indent=2))
        print(f"[ablation] updated {summary_path} ({len(results_by_tag)}/{len(configs)} configs done)")

    total_elapsed = time.perf_counter() - t_start
    summary = summarize_all_on_disk()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[ablation] done. {len(results_by_tag)}/{len(configs)} configs completed in {total_elapsed/60:.1f} min.")
    print(f"\n{'config':40s} {'validity':>10s} {'correct-dir':>12s} {'selectivity':>12s}")
    for tag, s in summary.items():
        cd = f"{s['correct_direction_rate']:.1%}" if s['correct_direction_rate'] is not None else "n/a"
        sel = f"{s['selectivity_ratio']:.2f}" if s['selectivity_ratio'] is not None else "n/a"
        print(f"{tag:40s} {s['validity_hit_rate']:9.1%}  {cd:>11s}  {sel:>11s}")
    print(f"\nfull summary saved to {summary_path}")


if __name__ == "__main__":
    main()

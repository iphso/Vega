"""Domain-agnostic orchestrator for steerability_generic.py -- replaces the
two VMEC++-only orchestrators (run_ablation.py, run_random_direction_
ablation.py) with one, parameterized by --domain, so bringing a second
domain up to the same eval depth as the first is a config change, not a
new script. Grid: 3 model types x {axis: 4 step sizes + 1 medium-fidelity
confirmation + 1 null, arbitrary: 1 step size (matching this project's own
first-pass scoping for that test, EXPERIMENT_LOG §20) + 1 null}, per seed.

Resumable (skips configs whose output file already exists, same discipline
as run_ablation.py) and writes a running consolidated summary after every
config, not just at the end.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from steerability_generic import run_eval

OUT_DIR = Path("/work/output")

MODELS = ["cvae", "diffusion", "gan"]
STEP_STDS = [0.5, 1.0, 1.5, 2.0]


def build_configs(domain, tags, n_anchors, k_validity, k_steer, seed, n_workers, timeout_seconds):
    configs = []
    for model in MODELS:
        tag = tags[model]
        base = dict(domain=domain, model_type=model, tag=tag, n_anchors=n_anchors, k_validity=k_validity,
                    k_steer=k_steer, seed=seed, n_workers=n_workers, timeout_seconds=timeout_seconds)

        # axis mode: magnitude sweep + one medium-fidelity confirmation + one null
        for step in STEP_STDS:
            out_tag = f"ablgen_{domain}_{model}_axis_step{step:.1f}_low_s{seed}".replace(".", "", 1)
            configs.append((out_tag, SimpleNamespace(**base, direction_mode="axis", n_directions=None,
                                                        step_std=step, steer_mode="targeted", fidelity="low", out_tag=out_tag)))
        if "medium" in {f for f in _fidelity_names(domain)}:
            out_tag = f"ablgen_{domain}_{model}_axis_step10_medium_s{seed}"
            configs.append((out_tag, SimpleNamespace(**base, direction_mode="axis", n_directions=None,
                                                        step_std=1.0, steer_mode="targeted", fidelity="medium", out_tag=out_tag)))
        out_tag = f"ablgen_{domain}_{model}_axis_null_low_s{seed}"
        configs.append((out_tag, SimpleNamespace(**base, direction_mode="axis", n_directions=None,
                                                    step_std=1.0, steer_mode="null", fidelity="low", out_tag=out_tag)))

        # arbitrary-direction mode: one step size + one null, matching this
        # project's own first-pass scoping for this test (EXPERIMENT_LOG §20)
        out_tag = f"ablgen_{domain}_{model}_arbitrary_step10_low_s{seed}"
        configs.append((out_tag, SimpleNamespace(**base, direction_mode="arbitrary", n_directions=None,
                                                    step_std=1.0, steer_mode="targeted", fidelity="low", out_tag=out_tag)))
        out_tag = f"ablgen_{domain}_{model}_arbitrary_null_low_s{seed}"
        configs.append((out_tag, SimpleNamespace(**base, direction_mode="arbitrary", n_directions=None,
                                                    step_std=1.0, steer_mode="null", fidelity="low", out_tag=out_tag)))
    return configs


def _fidelity_names(domain):
    from gym_schema import airfoil_spec, vmec_spec
    spec = {"vmec": vmec_spec, "airfoil": airfoil_spec}[domain]()
    return [f.name for f in spec.fidelities]


def summarize_all_on_disk(domain):
    results = {}
    for path in sorted(OUT_DIR.glob(f"steerability_generic_ablgen_{domain}_*.json")):
        tag = path.stem.removeprefix("steerability_generic_")
        out = json.loads(path.read_text())
        results[tag] = {
            "model_type": out["model_type"], "direction_mode": out["direction_mode"],
            "step_std": out["step_std"], "steer_mode": out["steer_mode"], "fidelity": out["fidelity"],
            "seed": out["seed"], "validity_hit_rate": out["overall_validity_hit_rate"],
            "correct_direction_rate": out["correct_direction_rate"],
            "mean_cosine_similarity": out["mean_cosine_similarity"],
            "selectivity_ratio": out.get("overall_selectivity", {}).get("selectivity_ratio"),
        }
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=["vmec", "airfoil"])
    p.add_argument("--tags", nargs=3, metavar=("CVAE_TAG", "DIFFUSION_TAG", "GAN_TAG"), required=True)
    p.add_argument("--n-anchors", type=int, default=12)
    p.add_argument("--k-validity", type=int, default=10)
    p.add_argument("--k-steer", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tags = {"cvae": args.tags[0], "diffusion": args.tags[1], "gan": args.tags[2]}
    configs = build_configs(args.domain, tags, args.n_anchors, args.k_validity, args.k_steer,
                             args.seed, args.n_workers, args.timeout_seconds)
    print(f"[ablation-generic] domain={args.domain} {len(configs)} configs queued")

    t_start = time.perf_counter()
    for i, (out_tag, run_args) in enumerate(configs, 1):
        out_path = OUT_DIR / f"steerability_generic_{out_tag}_{run_args.direction_mode}.json"
        if out_path.exists() and not args.force:
            print(f"[ablation-generic] ({i}/{len(configs)}) {out_tag}: already exists, skipping")
            continue
        elapsed = time.perf_counter() - t_start
        print(f"\n[ablation-generic] ({i}/{len(configs)}) {out_tag}: starting "
              f"(model={run_args.model_type} mode={run_args.direction_mode} step_std={run_args.step_std} "
              f"steer_mode={run_args.steer_mode} fidelity={run_args.fidelity})  [{elapsed/60:.1f}min elapsed]")
        try:
            run_eval(run_args)
        except Exception as e:
            print(f"[ablation-generic] ({i}/{len(configs)}) {out_tag}: FAILED -- {type(e).__name__}: {e}")
            continue
        summary = summarize_all_on_disk(args.domain)
        (OUT_DIR / f"ablation_generic_summary_{args.domain}.json").write_text(json.dumps(summary, indent=2))

    total = time.perf_counter() - t_start
    print(f"\n[ablation-generic] done in {total/60:.1f} min. "
          f"summary at output/ablation_generic_summary_{args.domain}.json")


if __name__ == "__main__":
    main()

"""One-off re-run of every diffusion config in the VMEC++ ablation grid
after the noise-schedule fix (EXPERIMENT_LOG §24->§25) -- overwrites the
existing diffusion_targets_full_s0.pt-derived output files in place with the
now-cosine-schedule checkpoint's numbers. cVAE/GAN configs are untouched
(never had this bug).

NOTE on naming: run_ablation.py's build_configs() constructs seed-0 out_tags
as f"..._s{seed}" (i.e. "..._s0"), a convention introduced when the seed-1
rerun bug was fixed (EXPERIMENT_LOG §19-20) -- but seed 0's original files on
disk predate that fix and were never renamed, so they exist WITHOUT the
"_s0" suffix (e.g. "ablation_diffusion_step05_low.json", not
"..._step05_low_s0.json"). Calling run_ablation.py --seed 0 directly right
now would not find those old files (wrong name), and would silently
re-run cVAE/GAN too, creating a second, redundantly-named copy of every
config rather than overwriting in place. This script instead reruns ONLY
diffusion's 12 configs (6 per seed x 2 seeds), writing seed 0 outputs at the
exact pre-existing (unsuffixed) filenames and seed 1 outputs at the
pre-existing "_s1" filenames -- true in-place overwrite, no orphaned
duplicate files, no unnecessary cVAE/GAN recompute.
"""
from types import SimpleNamespace

from eval_cvae_steerability import run_eval

TAG = "diffusion_targets_full_s0"
STEP_STDS = [0.5, 1.0, 1.5, 2.0]


def configs_for_seed(seed):
    suffix = "" if seed == 0 else f"_s{seed}"
    base = dict(model_type="diffusion", tag=TAG, n_anchors=12, anchor_selection="random",
                k_validity=10, k_steer=3, seed=seed, n_workers=24, timeout_seconds=90.0)
    out = []
    for step in STEP_STDS:
        out_tag = f"ablation_diffusion_step{step:.1f}_low{suffix}".replace(".", "", 1)
        out.append((out_tag, SimpleNamespace(**base, step_std=step, steer_mode="targeted", fidelity="low", out_tag=out_tag)))
    out_tag = f"ablation_diffusion_step10_medium{suffix}"
    out.append((out_tag, SimpleNamespace(**base, step_std=1.0, steer_mode="targeted", fidelity="medium", out_tag=out_tag)))
    out_tag = f"ablation_diffusion_null_low{suffix}"
    out.append((out_tag, SimpleNamespace(**base, step_std=1.0, steer_mode="null", fidelity="low", out_tag=out_tag)))
    return out


def main():
    configs = configs_for_seed(0) + configs_for_seed(1)
    print(f"[rerun-diffusion] {len(configs)} configs queued (6 per seed x 2 seeds)")
    for i, (out_tag, args) in enumerate(configs, 1):
        print(f"\n[rerun-diffusion] ({i}/{len(configs)}) {out_tag}: starting "
              f"(step_std={args.step_std} mode={args.steer_mode} fidelity={args.fidelity} seed={args.seed})")
        run_eval(args)
    print("\n[rerun-diffusion] all done")


if __name__ == "__main__":
    main()

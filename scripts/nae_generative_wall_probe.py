"""Generative probe of the qi-vs-edge_magnetic_mirror_ratio wall: instead of
moving continuously from a real anchor (every grounded_walk_step*.py run) or
constructing QI-by-analytic-formula (nae_pilot.py), directly *ask* each of
the project's three target-conditioned generative models (train_cvae.py,
train_diffusion.py, train_gan.py) for a design whose full 11-target vector
already sits on the far side of the wall -- low qi AND low mirror_ratio
AND the other 3 P3 constraints satisfied, jointly, as a single conditioning
request -- and see what each model's learned manifold actually proposes
there, oracle-validated for real. This is a different kind of "jump" than
NAE construction: it leans on whatever correlations these models absorbed
from ~522k real points (see merge_all_real_pools.py, output/splits/
all_grounded_walk_v1 -- includes the base archive plus every grounded-walk
lane's real pool, so the models have seen dense sampling right up against
the wall from many angles) rather than a closed-form physics recipe.

All three checkpoints must already be trained on --source split --split
all_grounded_walk_v1 (see merge_all_real_pools.py) with the SAME --tag
convention this script assumes by default (cvae_targets_allgw_v1,
diffusion_targets_allgw_v1, gan_targets_allgw_v1) -- override via --tag if
different.

For each nfp in {2,3,4,5}, builds ONE explicit conditioning target
("beyond-the-wall spec", see BEYOND_WALL_TARGETS) requesting: qi=1.5e-4
(well under the 3.162e-4 threshold, itself within the real dataset's
observed range -- min 6.1e-5), edge_magnetic_mirror_ratio=0.18 (under the
0.25 threshold, within the observed range -- min 0.122), vacuum_well=0.05
(above the >=0 threshold), flux_compression=0.6 (under the 0.9 threshold),
edge_rotational_transform_over_n_field_periods=0.3*nfp (clears the abs>=0.25
threshold), minimum_normalized_magnetic_gradient_scale_length=5.0 (near the
dataset mean, not an extreme ask), remaining targets (aspect_ratio,
max_elongation, average_triangularity, axis_magnetic_mirror_ratio,
axis_rotational_transform_over_n_field_periods) at their dataset means --
i.e. "give me something ordinary in every other respect, but P3-feasible."

Decodes --k candidates per (model_type, nfp), real-oracle-validates every
one at 'low' fidelity (matches the rest of this project's grounded-walk
convention), and reports per-model-type: convergence rate, measured
qi/mirror_ratio range, how many/which satisfy each individual P3 constraint,
how many satisfy all 5 jointly, and (if any) the best (lowest) mirror_ratio
among any converged, qi-feasible candidate -- directly comparable to every
basin's number so far (0.48-0.69 gradient search, 0.34-0.39 NAE-seed ALM
push).

Output: output/nae_generative_wall_probe/results.jsonl (one row per
(model_type, nfp, candidate) attempted).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402
from eval_cvae_steerability import load_generative_model, eval_space  # noqa: E402
from train_vae import nfp_one_hot, NFP_VALUES  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "nae_generative_wall_probe"

P3_THRESH = {
    "edge_rotational_transform_over_n_field_periods": ("abs_ge", 0.25),
    "edge_magnetic_mirror_ratio": ("le", 0.25),
    "flux_compression_in_regions_of_bad_curvature": ("le", 0.9),
    "vacuum_well": ("ge", 0.0),
    "qi": ("le", 0.0003162),
}

DEFAULT_TAGS = {
    "cvae": "cvae_targets_allgw_v1",
    "diffusion": "diffusion_targets_allgw_v1",
    "gan": "gan_targets_allgw_v1",
}

DATASET_MEANS = {  # from output/metadata.json's target_stats, "ordinary elsewhere" fill-ins
    "aspect_ratio": 8.618,
    "max_elongation": 4.806,
    "average_triangularity": -0.0988,
    "axis_magnetic_mirror_ratio": 0.293,
}


def beyond_wall_target(target_names, nfp):
    y = dict(DATASET_MEANS)
    y["qi"] = 1.5e-4
    y["vacuum_well"] = 0.05
    y["edge_magnetic_mirror_ratio"] = 0.18
    y["flux_compression_in_regions_of_bad_curvature"] = 0.6
    y["edge_rotational_transform_over_n_field_periods"] = 0.3 * nfp
    y["axis_rotational_transform_over_n_field_periods"] = 0.3 * nfp
    y["minimum_normalized_magnetic_gradient_scale_length"] = 5.0
    return np.array([y[n] for n in target_names], dtype=np.float64)


def p3_satisfied(y):
    for name, (kind, thresh) in P3_THRESH.items():
        v = y[name]
        if kind == "abs_ge" and not (abs(v) >= thresh):
            return False
        if kind == "le" and not (v <= thresh):
            return False
        if kind == "ge" and not (v >= thresh):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=40, help="candidates decoded per (model_type, nfp)")
    p.add_argument("--nfps", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    for mt in DEFAULT_TAGS:
        p.add_argument(f"--{mt}-tag", default=DEFAULT_TAGS[mt])
    args = p.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device("cpu")

    jobs, meta = [], {}
    tag_ctr = itertools.count()
    for model_type in ("cvae", "diffusion", "gan"):
        ckpt_tag = getattr(args, f"{model_type}_tag")
        sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std = \
            load_generative_model(model_type, ckpt_tag, dev)
        for nfp in args.nfps:
            y_req = beyond_wall_target(target_names, nfp)
            y_req_2d = y_req[None, :]
            y_req_eval = eval_space(y_req_2d, target_names)[0]
            y_req_z = (y_req_eval - t_mean) / t_std
            nfp_idx = NFP_VALUES.index(nfp)
            nfp_oh = torch.nn.functional.one_hot(torch.tensor([nfp_idx]), num_classes=len(NFP_VALUES)).float()
            cond = torch.cat([torch.tensor(y_req_z, dtype=torch.float32)[None, :], nfp_oh], dim=-1)

            decoded = sample_fn(cond, args.k)
            params = decoded * coeff_std + coeff_mean
            params[:, oracle.ZERO_INDICES] = 0.0
            for i in range(args.k):
                tag = next(tag_ctr)
                r_cos, z_sin, job_nfp, fidelity_name = oracle.params_to_worker_args(
                    params[i], {"nfp": nfp}, oracle.FIDELITY_PRESETS["low"])
                jobs.append((tag, r_cos, z_sin, job_nfp, fidelity_name))
                meta[tag] = {"model_type": model_type, "nfp": nfp, "requested": dict(zip(target_names, y_req.tolist()))}

    print(f"[wall_probe] {len(jobs)} candidates to validate ({len(DEFAULT_TAGS)} models x {len(args.nfps)} nfps x {args.k})")

    results = []
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
        row = dict(meta[tag])
        row["converged"] = False
        row["measured_targets"] = None
        row["p3_satisfied"] = False
        if ok:
            y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if all(v is not None for v in y.values()):
                row["converged"] = True
                row["measured_targets"] = y
                row["p3_satisfied"] = p3_satisfied(y)
        results.append(row)

    with open(STATE_DIR / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== summary ===")
    for model_type in ("cvae", "diffusion", "gan"):
        sub = [r for r in results if r["model_type"] == model_type]
        conv = [r for r in sub if r["converged"]]
        p3ok = [r for r in sub if r["p3_satisfied"]]
        print(f"[{model_type}] {len(conv)}/{len(sub)} converged, {len(p3ok)} satisfy ALL 5 P3 constraints")
        if conv:
            qi = np.array([r["measured_targets"]["qi"] for r in conv])
            mr = np.array([r["measured_targets"]["edge_magnetic_mirror_ratio"] for r in conv])
            print(f"       qi range {qi.min():.6g}-{qi.max():.6g}, mirror_ratio range {mr.min():.4f}-{mr.max():.4f}")
            feas = qi <= P3_THRESH["qi"][1]
            print(f"       {feas.sum()} pass qi<=3.162e-4 on their own", end="")
            if feas.sum():
                print(f", best mirror_ratio among them: {mr[feas].min():.4f}")
            else:
                print()


if __name__ == "__main__":
    main()

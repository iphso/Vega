"""Direct test of the officially-shipped omnigenous-field construction recipe
(`constellaration.data_generation.vmec_optimization.optimize_boundary_omnigenity_vmec`,
`constellaration.omnigeneity.omnigenity_field_sampling`) against real P3
feasibility, per direct user request ("test whatever you feel we need to
reliably get to valid P3 candidates").

Why this is a fundamentally different, better-posed attempt than everything
else in this project so far: every prior method (gradient search from a real
near-wall point, generative-model wall probes, direct NGOpt-against-real-VMEC
from many seeds/basins, NAE first-order construction) searches or constructs
in the RAW 90-dim boundary Fourier coefficient space and hopes QI emerges.
This recipe instead targets a physically-structured OMNIGENOUS FIELD directly
-- mirror_ratio, edge_rotational_transform_over_n_field_periods, aspect_ratio,
and max_elongation are literal input dials, not properties to discover -- and
fits a real boundary to it via NGOpt+real-VMEC (the exact same
`optimize_boundary_omnigenity_vmec` used to generate the training dataset
this whole project trains surrogates on). The omnigenity residual being
driven toward zero during the fit is a close cousin of the actual P3 `qi`
metric (both derive from Goodman et al. 2023's QI construction), so a good
fit should produce genuinely low qi by construction, not by search luck.

What it does NOT target: vacuum_well and flux_compression_in_regions_of_bad_
curvature are outside this fit's objective entirely -- every real near-wall
point this project has ever found (NAE-seeded or gradient-search-refined)
has had trouble with exactly these two, so they remain the likely blocker
even if this recipe nails qi/mirror_ratio/iota for real. If so, the
resulting boundary is a strictly better starting point to hand to
ngopt_alm_p3.py than anything used so far (mirror_ratio and qi should already
be far closer to feasible than any prior real seed).

Converts the fitted `SurfaceRZFourier` directly to this project's own 90-dim
(r_cos flatten + z_sin flatten) row layout -- confirmed byte-for-byte
compatible in EXPERIMENT_LOG (no resampling/refit) -- and validates every
candidate through the SAME real-VMEC oracle (vmec_oracle.worker_fn) every
other script in this project uses, not the omnigenity fit's own internal
low-fidelity forward model.

Must run in the `train` service (needs constellaration's data_generation
submodule + nevergrad + vmecpp in the same process).

Output: output/omnigenity_seed/{results.jsonl, best_seed_<tag>.npy}.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "omnigenity_seed"

P3_THRESH = {
    "edge_rotational_transform_over_n_field_periods": (">=", 0.25, True),
    "qi": ("<=", 10 ** -3.5, False),
    "edge_magnetic_mirror_ratio": ("<=", 0.25, False),
    "flux_compression_in_regions_of_bad_curvature": ("<=", 0.9, False),
    "vacuum_well": (">=", 0.0, False),
}


def p3_check(y):
    out = {}
    for name, (op, thresh, use_abs) in P3_THRESH.items():
        v = abs(y[name]) if use_abs else y[name]
        out[name] = (v >= thresh) if op == ">=" else (v <= thresh)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--nfp", type=int, default=2)
    p.add_argument("--mirror-ratio", type=float, default=0.18,
                    help="target field mirror ratio, comfortably under the 0.25 P3 threshold")
    p.add_argument("--iota-over-nfp", type=float, default=0.28,
                    help="target edge_rotational_transform_over_n_field_periods, comfortably over 0.25")
    p.add_argument("--aspect-ratio", type=float, default=8.0)
    p.add_argument("--max-elongation", type=float, default=5.0)
    p.add_argument("--n-samples", type=int, default=8, help="x_lmn field-shape samples to try (mirror/iota/etc fixed)")
    p.add_argument("--max-poloidal-mode", type=int, default=4)
    p.add_argument("--max-toroidal-mode", type=int, default=4)
    p.add_argument("--budget-per-design-variable", type=int, default=100)
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from constellaration.data_generation import vmec_optimization, vmec_optimization_settings
    from constellaration.omnigeneity import omnigenity_field_sampling

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"{args.tag}_results.jsonl"

    sampler_settings = omnigenity_field_sampling.SampleOmnigenousFieldAndTargetsSettings(
        n_field_periods=args.nfp,
        mirror_ratio=args.mirror_ratio,
        edge_rotational_transform_over_n_field_periods=args.iota_over_nfp,
        aspect_ratio=args.aspect_ratio,
        max_elongation=args.max_elongation,
        seed=args.seed,
        n_samples=args.n_samples,
    )
    targets_list = sampler_settings.sample_omnigenous_fields_and_targets()
    print(f"[omnigenity_seed] sampled {len(targets_list)} field/target combos "
          f"(nfp={args.nfp}, mirror_ratio={args.mirror_ratio}, "
          f"iota/nfp={args.iota_over_nfp}, aspect_ratio={args.aspect_ratio})")

    opt_settings = vmec_optimization_settings.OmnigenousFieldVmecOptimizationSettings(
        max_poloidal_mode=args.max_poloidal_mode,
        max_toroidal_mode=args.max_toroidal_mode,
        gradient_free_budget_per_design_variable=args.budget_per_design_variable,
        verbose=False,
    )

    jobs, meta = [], {}
    tag_ctr = itertools.count()
    fidelity_name = oracle.FIDELITY_PRESETS["low"]
    fitted_params = {}

    for i, targets in enumerate(targets_list):
        print(f"[omnigenity_seed] fitting sample {i+1}/{len(targets_list)} via real-VMEC NGOpt "
              f"(this can take a while -- each fit is its own NGOpt run against real VMEC++)...")
        try:
            boundary = vmec_optimization.optimize_boundary_omnigenity_vmec(
                targets=targets, settings=opt_settings)
        except Exception as e:
            print(f"[omnigenity_seed] sample {i} fit FAILED: {e!r}")
            continue
        r_cos = np.asarray(boundary.r_cos, dtype=np.float64)
        z_sin = np.asarray(boundary.z_sin, dtype=np.float64)
        if r_cos.shape != (5, 9) or z_sin.shape != (5, 9):
            print(f"[omnigenity_seed] sample {i}: unexpected shape r_cos={r_cos.shape} z_sin={z_sin.shape}, skipping")
            continue
        params = np.concatenate([r_cos.flatten(), z_sin.flatten()])
        params[oracle.ZERO_INDICES] = 0.0
        tag = next(tag_ctr)
        fitted_params[tag] = params
        r_cos_final, z_sin_final, job_nfp, fname = oracle.params_to_worker_args(
            params, {"nfp": args.nfp}, fidelity_name)
        jobs.append((tag, r_cos_final, z_sin_final, job_nfp, fname))
        meta[tag] = {"sample": i}

    print(f"[omnigenity_seed] {len(jobs)}/{len(targets_list)} fits succeeded, validating via REAL VMEC++...")

    # BUG FOUND AND FIXED: optimize_boundary_omnigenity_vmec's own internal
    # NGOpt fit uses a ProcessPoolExecutor with multiprocessing.get_context
    # ("forkserver"). Running run_batch_with_timeout (fork-based) in the SAME
    # process afterward silently corrupts every job -- confirmed directly:
    # the exact same params, re-validated in a FRESH process, converge fine
    # with real, sane metrics; in-process right after the omnigenity fit,
    # every job comes back converged=False with no error payload at all.
    # Fix: shell out to a fresh interpreter for the validation step instead
    # of calling run_batch_with_timeout in this process.
    np.save(STATE_DIR / f"{args.tag}_fitted_params.npy",
            np.stack([fitted_params[t] for t in sorted(fitted_params)]))
    import subprocess
    validate_script = f"""
import sys, json
sys.path.insert(0, "/work/scripts")
import numpy as np
import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout

params_arr = np.load("{STATE_DIR / f'{args.tag}_fitted_params.npy'}")
jobs = []
for i, p in enumerate(params_arr):
    r_cos = p[:45].reshape(5, 9)
    z_sin = p[45:].reshape(5, 9)
    jobs.append((i, r_cos, z_sin, {args.nfp}, "{fidelity_name}"))

out = []
for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, {args.n_workers}, {args.timeout_seconds}):
    out.append({{"sample": int(tag), "converged": bool(ok),
                 "measured_targets": payload if ok else None,
                 "error": None if ok else str(payload)}})
print(json.dumps(out))
"""
    proc = subprocess.run(["python3", "-c", validate_script], capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"[omnigenity_seed] validation subprocess FAILED: {proc.stderr[-2000:]}")
        return
    validated = {r["sample"]: r for r in json.loads(proc.stdout.strip().splitlines()[-1])}

    results = []
    for tag in sorted(fitted_params):
        row = dict(meta[tag])
        row["params"] = fitted_params[tag].tolist()
        row["nfp"] = args.nfp
        row["converged"] = False
        v = validated.get(tag)
        if v and v["converged"]:
            y = v["measured_targets"]
            if all(y.get(n) is not None for n in oracle.TARGET_NAMES):
                row["converged"] = True
                row["measured_targets"] = y
                checks = p3_check(y)
                row["p3_checks"] = checks
                row["p3_satisfied"] = all(checks.values())
        results.append(row)

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    conv = [r for r in results if r["converged"]]
    p3ok = [r for r in results if r.get("p3_satisfied")]
    print(f"\n[omnigenity_seed] {len(conv)}/{len(results)} converged, {len(p3ok)} satisfy ALL 5 P3 constraints")
    if conv:
        for r in sorted(conv, key=lambda r: -sum(r["p3_checks"].values())):
            y = r["measured_targets"]
            c = r["p3_checks"]
            n_pass = sum(c.values())
            print(f"  sample {r['sample']}: {n_pass}/5 pass -- "
                  f"iota={y['edge_rotational_transform_over_n_field_periods']:+.3f}({'Y' if c['edge_rotational_transform_over_n_field_periods'] else 'n'}) "
                  f"qi={y['qi']:.5f}({'Y' if c['qi'] else 'n'}) "
                  f"mr={y['edge_magnetic_mirror_ratio']:.3f}({'Y' if c['edge_magnetic_mirror_ratio'] else 'n'}) "
                  f"fc={y['flux_compression_in_regions_of_bad_curvature']:.3f}({'Y' if c['flux_compression_in_regions_of_bad_curvature'] else 'n'}) "
                  f"vw={y['vacuum_well']:+.4f}({'Y' if c['vacuum_well'] else 'n'})")
        best = max(conv, key=lambda r: sum(r["p3_checks"].values()))
        best_x = np.concatenate([np.array(best["params"]), [float(args.nfp)], [1.0]])
        np.save(STATE_DIR / f"best_seed_{args.tag}.npy", best_x[None, :])
        print(f"[omnigenity_seed] best candidate ({sum(best['p3_checks'].values())}/5) saved to "
              f"best_seed_{args.tag}.npy for use as --seed-x in ngopt_alm_p3.py")


if __name__ == "__main__":
    main()

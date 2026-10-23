"""Probe: validate the paper's own P3 seeding recipe end-to-end before
committing compute to the four-instance ALM run.

Background (EXPERIMENT_LOG, 2026-09-26): `paper_v1` spent 6 days and 680k real
VMEC++ evals grinding to worst_violation=0.204 against a MIS-SPECIFIED problem.
The real P3 (`constellaration.problems.MHDStableQIStellarator`) is
MULTI-objective -- minimize (-L_grad_B, aspect_ratio) -- scored by the
HYPERVOLUME of a SET of feasible designs, and its `evaluate()` takes a LIST of
boundaries and runs them at `default_high_fidelity()`. The paper decomposes it
into four single-objective instances by adding `aspect_ratio <= A*` as a sixth
inequality constraint, A* in {6,8,10,12}, each "seeded from rotating ellipse
configurations", 15-24h per instance -- reaching violations
{0.104, 0.00130, 0.0, 0.00210}. We had no aspect-ratio constraint and a
different (omnigenity-fit) seed.

This probe answers the two things that must be true before the real run is
worth launching:
  1. Does `constellaration.initial_guess.generate_rotating_ellipse` round-trip
     cleanly into this project's own 90-dim (r_cos.flatten + z_sin.flatten)
     row layout, at mpol=4/ntor=4, and survive the real VMEC++ oracle?
  2. Where do those seeds actually START on all six constraints? Specifically
     their `aspect_ratio` -- a seed generated at aspect_ratio=A does not
     necessarily MEASURE at A once VMEC solves it, and if the seeds land far
     outside their own A* box the instance is mis-seeded from outer 0.

Prints a per-seed table; writes nothing the main runner depends on. Cheap
(a few dozen single evals), so run it before the multi-day commitment.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402
from optimize import paper_feasibility_violation  # noqa: E402

# The five official P3 constraints, plus aspect_ratio as the sixth that the
# paper's decomposition adds. Tuple layout matches ngopt_alm_p3_paper.py's
# own P3_CONSTRAINTS: (name, op, bound, use_abs, divisor, log_transform).
BASE_CONSTRAINTS = [
    ("edge_rotational_transform_over_n_field_periods", ">=", 0.25, True, None, False),
    ("qi", "<=", 10 ** -3.5, False, None, True),
    ("edge_magnetic_mirror_ratio", "<=", 0.25, False, None, False),
    ("flux_compression_in_regions_of_bad_curvature", "<=", 0.9, False, None, False),
    ("vacuum_well", ">=", 0.0, False, 0.1, False),
]

ASPECT_RATIOS = [6.0, 8.0, 10.0, 12.0]


def constraints_for(aspect_ratio_max):
    return BASE_CONSTRAINTS + [
        ("aspect_ratio", "<=", aspect_ratio_max, False, None, False),
    ]


def real_violations(measured, constraints):
    import torch
    out = []
    for name, op, value, use_abs, divisor, log_t in constraints:
        col = torch.tensor([[measured[name]]], dtype=torch.float32)
        v = paper_feasibility_violation(
            col[:, 0], op, value, use_abs=use_abs, divisor=divisor, log_transform=log_t
        )
        out.append(float(v.item()))
    return np.array(out)


def rotating_ellipse_params(aspect_ratio, elongation, iota_over_nfp, nfp):
    """Paper's seed, converted to this project's 90-dim row layout.

    `generate_rotating_ellipse` returns an mpol=1/ntor=1 surface; P3's design
    space here is mpol=4/ntor=4 -> r_cos and z_sin both (5, 9). The package's
    own `smooth_and_set_max_mode_numbers` does that widening (zero-padding the
    higher modes), which is the same helper `initial_guess` itself uses
    internally, so this stays on the package's supported path rather than
    hand-padding the arrays.
    """
    from constellaration import initial_guess
    from constellaration.mhd import near_axis_configuration

    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=aspect_ratio,
        elongation=elongation,
        # generate_rotating_ellipse takes the EDGE TOTAL iota and divides by
        # nfp internally; P3's constraint is on iota/nfp, so scale up here.
        rotational_transform=iota_over_nfp * nfp,
        n_field_periods=nfp,
    )
    boundary = near_axis_configuration.smooth_and_set_max_mode_numbers(boundary, 4, 4)

    r_cos = np.asarray(boundary.r_cos, dtype=np.float64)
    z_sin = np.asarray(boundary.z_sin, dtype=np.float64)
    if r_cos.shape != (5, 9) or z_sin.shape != (5, 9):
        raise ValueError(f"unexpected shape r_cos={r_cos.shape} z_sin={z_sin.shape}")

    params = np.concatenate([r_cos.flatten(), z_sin.flatten()])
    params[oracle.ZERO_INDICES] = 0.0
    return params


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aspect-ratios", type=float, nargs="+", default=ASPECT_RATIOS)
    p.add_argument("--elongations", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    p.add_argument("--nfps", type=int, nargs="+", default=[2])
    p.add_argument("--iota-over-nfp", type=float, nargs="+", default=[0.30],
                   help="comfortably over the 0.25 lower bound")
    p.add_argument("--n-workers", type=int, default=28)
    args = p.parse_args()

    specs, jobs, params_by_tag = {}, [], {}
    fidelity_name = oracle.FIDELITY_PRESETS["low"]
    tag_ctr = itertools.count()

    for A, elong, nfp, iota in itertools.product(
        args.aspect_ratios, args.elongations, args.nfps, args.iota_over_nfp
    ):
        try:
            params = rotating_ellipse_params(A, elong, iota, nfp)
        except Exception as e:
            print(f"[probe] seed A={A} elong={elong} nfp={nfp} iota={iota} "
                  f"FAILED to build: {e!r}")
            continue
        tag = next(tag_ctr)
        r_cos, z_sin, job_nfp, fname = oracle.params_to_worker_args(
            params, {"nfp": nfp}, fidelity_name
        )
        jobs.append((tag, r_cos, z_sin, job_nfp, fname))
        specs[tag] = (A, elong, nfp, iota)
        params_by_tag[tag] = params

    print(f"[probe] built {len(jobs)} rotating-ellipse seeds, evaluating via real VMEC++...")

    rows = {}
    for tag, ok, payload in run_batch_with_timeout(
        jobs, oracle.worker_fn, min(args.n_workers, len(jobs)) or 1, 120.0
    ):
        rows[tag] = (ok, payload)

    print(f"\n{'A*':>5} {'elong':>6} {'nfp':>4} {'iota':>5} {'conv':>5} {'aspect':>8} {'L_gradB':>8} "
          f"{'qi':>10} {'mirror':>7} {'fluxcomp':>9} {'vwell':>9} {'iota/nfp':>9} {'worst':>8}")
    print("-" * 115)
    ok_rows = []
    for tag in sorted(specs):
        A, elong, nfp, iota = specs[tag]
        pre = f"{A:>5.0f} {elong:>6.1f} {nfp:>4d} {iota:>5.2f}"
        ok, payload = rows.get(tag, (False, "missing"))
        if not ok:
            print(f"{pre} {'NO':>5}   {str(payload)[:55]}")
            continue
        y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
        if any(v is None for v in y.values()):
            print(f"{pre} {'PART':>5}   incomplete metrics")
            continue
        viol = real_violations(y, constraints_for(A))
        ok_rows.append((A, elong, nfp, iota, float(viol.max())))
        print(f"{pre} {'yes':>5} "
              f"{y['aspect_ratio']:>8.3f} "
              f"{y['minimum_normalized_magnetic_gradient_scale_length']:>8.3f} "
              f"{y['qi']:>10.6f} "
              f"{y['edge_magnetic_mirror_ratio']:>7.3f} "
              f"{y['flux_compression_in_regions_of_bad_curvature']:>9.3f} "
              f"{y['vacuum_well']:>9.5f} "
              f"{y['edge_rotational_transform_over_n_field_periods']:>9.3f} "
              f"{viol.max():>8.4f}")

    print("\n[probe] violation vector order: "
          "[iota, qi, mirror, flux_compression, vacuum_well, aspect_ratio]")

    # Best viable seed per A* -- this is the actual deliverable, since the
    # four-instance runner needs exactly one starting boundary per instance.
    print("\n[probe] best viable seed per A*:")
    for A in sorted({r[0] for r in ok_rows}):
        cands = [r for r in ok_rows if r[0] == A]
        best = min(cands, key=lambda r: r[4])
        print(f"  A*={best[0]:>5.0f}: elong={best[1]:.1f} nfp={best[2]} "
              f"iota/nfp={best[3]:.2f} -> worst_violation={best[4]:.4f}")
    missing = sorted(set(args.aspect_ratios) - {r[0] for r in ok_rows})
    if missing:
        print(f"  NO VIABLE SEED for A* in {missing}")


if __name__ == "__main__":
    main()

"""Generates a batch of bracket topology-optimization candidates across the
real parameter space (mount positions, load position/direction/magnitude,
volume fraction) -- direct follow-up request ("make a whole bunch of
candidates... the asymmetric one is more interesting, yeah?").

Sampling is deliberately biased AWAY from the symmetric/straight-down v1
case: our own naive-baseline check (bracket_simp_warp.py, §100) showed the
symmetric case only beats a trivial two-tube guess by 1.07x, while the
asymmetric/angled v2 case beats it by 2.37x -- symmetric point-to-point
loads are close to the textbook-obvious statics answer, so there's more
real optimization signal to see in asymmetric configs. Concretely:
  - mount_a_x sampled from the left third of the domain, mount_b_x from the
    right third (never mirrored around the domain center on purpose).
  - load_x sampled off-center between the mounts (not the midpoint).
  - load direction sampled off-vertical (tilted in x and z), not straight
    down.
Same solver, same resolution/iteration count already verified to converge
(bracket_simp_warp.py's own default: 1500 iters at res (40,20,6) is flat to
5 decimal places -- §100) -- this is a parameter sweep of the EXISTING
single-load-case solver, not a new optimizer.

Each candidate is scored against its own naive two-tube baseline
automatically (built into bracket_simp_warp.run()) and the whole batch's
results are written to one summary index (`bracket_candidates_index.json`)
that the viewer reads to populate its case list -- so a candidate that
turns out to be under-converged or uninteresting is visible in the numbers,
not hidden.
"""
import argparse
import json

import numpy as np

from bracket_export_mesh_warp import export_mesh
from bracket_simp_warp import run

BOUNDS_HI = [2.0, 1.0, 0.3]
RES = [40, 20, 6]
MOUNT_RADIUS = 0.12
LOAD_RADIUS = 0.12


def sample_candidate(rng):
    mount_a_x = rng.uniform(0.15, 0.55)
    mount_b_x = rng.uniform(1.45, 1.85)
    # Load strictly between the mounts, biased off-center (avoid the exact
    # midpoint, which is the one point where an asymmetric mount pair still
    # produces a near-symmetric load path).
    span_lo, span_hi = mount_a_x + 0.25, mount_b_x - 0.25
    mid = (span_lo + span_hi) / 2
    load_x = mid + rng.choice([-1, 1]) * rng.uniform(0.15, (span_hi - span_lo) / 2 - 0.02)
    load_x = float(np.clip(load_x, span_lo, span_hi))

    # Off-vertical load direction: mostly downward (matches gravity-loaded
    # bracket framing) but tilted in x/z so it isn't the straight-down v1
    # case. Magnitude varied too, since a bigger load changes how much the
    # volume constraint actually binds.
    tilt_deg = rng.uniform(10, 35)
    az = rng.uniform(0, 2 * np.pi)
    mag = rng.uniform(0.10, 0.20)
    down = -np.cos(np.radians(tilt_deg))
    horiz = np.sin(np.radians(tilt_deg))
    load_vec = [mag * horiz * np.cos(az), mag * down, mag * horiz * np.sin(az)]

    volfrac = float(rng.uniform(0.22, 0.35))

    return dict(
        mount_a_x=float(mount_a_x), mount_b_x=float(mount_b_x), mount_radius=MOUNT_RADIUS,
        load_x=float(load_x), load_radius=LOAD_RADIUS, load_vec=[float(v) for v in load_vec],
        bounds_hi=BOUNDS_HI, res=RES, volfrac=volfrac,
        penal=3.0, e_min=1e-3, rho_min=1e-3, vol_penalty_weight=50.0,
        n_iters=1500, lr=0.02,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="bracket_candidate")
    ap.add_argument("--n-iters-override", type=int, default=None,
                     help="For smoke-testing the sweep mechanics only -- NOT for real candidates "
                          "(1500 is the checked convergence point at this resolution, §100).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    index = []
    for i in range(args.n_candidates):
        tag = f"{args.out_prefix}_{i:02d}"
        params = sample_candidate(rng)
        if args.n_iters_override is not None:
            params["n_iters"] = args.n_iters_override
        print(f"\n=== candidate {i:02d}/{args.n_candidates - 1} ({tag}) ===")
        print({k: v for k, v in params.items() if k not in ("bounds_hi", "res")})
        meta = run(out_tag=tag, quiet=True, **params)
        export_mesh(tag)
        index.append({
            "tag": tag,
            "mount_a_x": params["mount_a_x"], "mount_b_x": params["mount_b_x"],
            "load_x": params["load_x"], "load_vec": params["load_vec"],
            "volfrac": params["volfrac"],
            "final_compliance": meta["final_compliance"],
            "final_mean_rho": meta["final_mean_rho"],
            "naive_baseline_compliance": meta["naive_baseline_compliance"],
            "naive_vs_simp_ratio": meta["naive_vs_simp_ratio"],
            # Real, physically-grounded metrics (Aluminum 6061, 200x100x30mm
            # envelope -- see bracket_postprocess.py) -- direct user feedback
            # that SIMP compliance alone isn't an accessible metric.
            "mass_kg": meta.get("mass_kg"), "max_load_kgf": meta.get("max_load_kgf"),
            "safety_factor": meta.get("safety_factor"),
            "islands_removed_fraction": meta.get("islands_removed_fraction"),
            "spans_load_path": meta.get("spans_load_path"),
        })

    index.sort(key=lambda c: -c["naive_vs_simp_ratio"])
    with open("/work/output/bracket_candidates_index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n\n=== {len(index)} candidates, ranked by naive-vs-SIMP ratio (higher = more non-trivial gain) ===")
    for c in index:
        print(f"{c['tag']}: ratio={c['naive_vs_simp_ratio']:.2f}x  compliance={c['final_compliance']:.4f}  "
              f"volfrac={c['volfrac']:.2f}  mounts=({c['mount_a_x']:.2f},{c['mount_b_x']:.2f})  load_x={c['load_x']:.2f}")
    print("\nsaved output/bracket_candidates_index.json")


if __name__ == "__main__":
    main()

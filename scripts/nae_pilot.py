"""NAE (near-axis-expansion) pilot: a fundamentally different way of getting a
boundary, meant to test whether the qi-vs-edge_magnetic_mirror_ratio wall
found by every gradient/ALM-based search so far (grounded_walk_step*.py, all
basins nfp=2/3/4) is a property of the whole design family, or an artifact of
local search never being able to leave the basin it started in.

Every experiment up to this point moves *continuously* from a real anchor
point via VAE-latent gradient steps -- it can trace whatever Pareto curve
it's already on, but it can never discover a disconnected region of
coefficient space. This script instead *constructs* boundaries directly via
constellaration.initial_guess.generate_nae(), an analytic near-axis-expansion
method (Goodman et al. 2023, "Constructing Precisely Quasi-Isodynamic
Magnetic Fields") that is QI *by construction* to first order, with
mirror_ratio as a literal dialed-in input -- not something read off after an
optimization. This is part of the official constellaration data-generation
pipeline (see constellaration/data_generation/vmec_optimization.py), but
nothing in this project has called it before now.

Sweeps (nfp, mirror_ratio, aspect_ratio) with rotational_transform set
comfortably above the P3 threshold (0.25 * nfp) so constraint C1 is not the
bottleneck, converts each resulting SurfaceRZFourier to this project's
(r_cos[5,9], z_sin[5,9], nfp) layout, and real-VMEC-validates every one
through the same oracle.worker_fn/run_batch_with_timeout path
grounded_walk_validate.py uses, at the same "low" fidelity the grounded-walk
searches use throughout, so results are directly comparable.

Output: output/nae_pilot/results.jsonl (one row per config attempted, with
converged flag + all 11 measured targets or null) and a short console
summary of the qi-vs-mirror_ratio scatter across every point that converged,
plus how many satisfy all 5 P3 constraints jointly.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

from constellaration import initial_guess  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "nae_pilot"

# P3 benchmark constraints (see EXPERIMENT_LOG.md).
P3_THRESH = {
    "edge_rotational_transform_over_n_field_periods": ("abs_ge", 0.25),
    "edge_magnetic_mirror_ratio": ("le", 0.25),
    "flux_compression_in_regions_of_bad_curvature": ("le", 0.9),
    "vacuum_well": ("ge", 0.0),
    "qi": ("le", 0.0003162),
}


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


def build_grid():
    """Every (nfp, mirror_ratio, aspect_ratio) combo to try. rotational_transform
    is set to 0.3*nfp (comfortably clears the 0.25*nfp P3 threshold, C1 is not
    what we're testing here); max_elongation fixed at a middle-of-archive value
    (dataset mean ~4.8, but the max is a long tail -- 4.0 keeps us away from
    the tail while still giving the NAE construction room)."""
    nfps = [2, 3, 4, 5]
    mirror_ratios = [0.15, 0.20, 0.25, 0.30]
    aspect_ratios = [6.0, 8.0, 10.0]
    grid = []
    for nfp, mr, ar in itertools.product(nfps, mirror_ratios, aspect_ratios):
        grid.append({
            "n_field_periods": nfp,
            "mirror_ratio": mr,
            "aspect_ratio": ar,
            "max_elongation": 4.0,
            "rotational_transform": 0.3 * nfp,
        })
    return grid


def surface_to_row(surface):
    """SurfaceRZFourier (stellarator-symmetric, max_poloidal_mode=4,
    max_toroidal_mode=4) -> flat r_cos[45], z_sin[45], both C-order
    (m outer, n inner, n=n_idx-4) -- matches scripts/preprocess.py's layout
    exactly (physics_bundle.py's Fourier-convention note confirms
    SurfaceRZFourier already uses this same synthesis)."""
    r_cos = np.asarray(surface.r_cos)
    z_sin = np.asarray(surface.z_sin)
    assert r_cos.shape == (5, 9) and z_sin.shape == (5, 9), \
        f"unexpected shape r_cos={r_cos.shape} z_sin={z_sin.shape}"
    return r_cos, z_sin


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    print(f"[nae_pilot] {len(grid)} configs to construct + validate")

    jobs, meta = [], {}
    n_construct_failed = 0
    for i, cfg in enumerate(grid):
        try:
            surface = initial_guess.generate_nae(
                aspect_ratio=cfg["aspect_ratio"],
                max_elongation=cfg["max_elongation"],
                rotational_transform=cfg["rotational_transform"],
                mirror_ratio=cfg["mirror_ratio"],
                n_field_periods=cfg["n_field_periods"],
                max_poloidal_mode=4,
                max_toroidal_mode=4,
            )
            r_cos, z_sin = surface_to_row(surface)
        except Exception as e:  # noqa: BLE001 -- NAE construction can fail for some inputs
            n_construct_failed += 1
            print(f"[nae_pilot] config {i} {cfg} FAILED to construct: {e}")
            continue
        tag = i
        jobs.append((tag, r_cos, z_sin, cfg["n_field_periods"], oracle.FIDELITY_PRESETS["low"]))
        meta[tag] = cfg | {"r_cos": r_cos.flatten().tolist(), "z_sin": z_sin.flatten().tolist()}

    print(f"[nae_pilot] {len(jobs)}/{len(grid)} constructed OK ({n_construct_failed} failed), "
          f"validating through real VMEC++...")

    n_converged = 0
    n_p3_ok = 0
    results = []
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, 28, 45.0):
        row = dict(meta[tag])
        row["converged"] = False
        row["measured_targets"] = None
        row["p3_satisfied"] = False
        if ok:
            y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if all(v is not None for v in y.values()):
                row["converged"] = True
                row["measured_targets"] = y
                n_converged += 1
                if p3_satisfied(y):
                    row["p3_satisfied"] = True
                    n_p3_ok += 1
        results.append(row)

    with open(STATE_DIR / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[nae_pilot] {n_converged}/{len(jobs)} converged, {n_p3_ok} satisfy ALL 5 P3 constraints")

    conv = [r for r in results if r["converged"]]
    if conv:
        qi = np.array([r["measured_targets"]["qi"] for r in conv])
        mr = np.array([r["measured_targets"]["edge_magnetic_mirror_ratio"] for r in conv])
        print(f"[nae_pilot] qi range: {qi.min():.6g} - {qi.max():.6g}")
        print(f"[nae_pilot] mirror_ratio range: {mr.min():.4f} - {mr.max():.4f}")
        if len(conv) > 2:
            r_corr = np.corrcoef(qi, mr)[0, 1]
            print(f"[nae_pilot] qi-vs-mirror_ratio correlation across {len(conv)} converged NAE points: r={r_corr:.4f}")
        thresh = P3_THRESH["qi"][1]
        feas = qi <= thresh
        print(f"[nae_pilot] {feas.sum()} points pass qi<=0.0003162 on their own")
        if feas.sum():
            best_i = np.argmin(mr[feas])
            best = np.array(conv)[feas][best_i]
            print(f"[nae_pilot] best mirror_ratio among qi-feasible NAE points: "
                  f"{mr[feas][best_i]:.4f} (qi={qi[feas][best_i]:.6g}, cfg={best['n_field_periods']=} "
                  f"mirror_ratio_target={best['mirror_ratio']} aspect_ratio={best['aspect_ratio']})")


if __name__ == "__main__":
    main()

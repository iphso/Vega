"""Experiment (A) from the 2026-09-15 literature-review conversation: test
whether tightening the QI-consistency order used during NAE construction
changes the realized qi-vs-edge_magnetic_mirror_ratio tradeoff at finite
aspect ratio, as a cheap first probe of "direct QI construction" methods
(Rodriguez et al. 2022/2024, arXiv:2206.08417, arXiv:2409.20328) before
committing to standing up a full 2nd-order (r2) NAE pipeline.

Important scoping note, confirmed by reading source directly: this is NOT a
true 2nd-order-in-r NAE construction. constellaration.mhd.near_axis_configuration
always builds the underlying qic.Qic object with order="r1" hardcoded (see
near_axis_configuration_to_pyqic) -- there is no r2 lever exposed anywhere in
this dependency chain. What IS exposed is `qic.Qic.min_geo_qi_consistency`'s
own `order` argument, which is unrelated to expansion order in r: it's how
many derivatives of the QI-consistency residual are also driven to zero at
the phi=0,pi/nfp points (order=1 used by constellaration's generate();
order=2 additionally zeroes the first derivative). Reusing the same free
parameters (axis Fourier coefficients + d_over_curvature), just holding the
solution to a stricter local QI condition, is a legitimate variant worth
real-oracle-validating -- but a true r2 construction (new Z2/X2 degrees of
freedom) is a separate, much bigger undertaking, still open if this doesn't
help.

Method: monkeypatch near_axis_configuration.generate's objective from
order=1 to order=2 (source-patch + exec, since order isn't a parameter of
generate() itself), sweep the same style of grid as nae_pilot.py but focused
on the mirror_ratio_target band nae_pilot's own re-mining showed mattered
(0.10-0.20, not the original 0.15-0.30), across nfp in {2,3,4} x aspect_ratio
in {6,8,10}, and real-VMEC-validate every resulting boundary exactly like
nae_pilot.py did, at the same "low" fidelity, for direct comparability.

Output: output/nae_order2_pilot/results.jsonl, same row schema as
nae_pilot.py's, plus a console summary comparing directly against the
order=1 baseline already sitting in output/nae_pilot/results.jsonl.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402

from constellaration.mhd import near_axis_configuration as nac  # noqa: E402

OUT_DIR = Path("/work/output")
STATE_DIR = OUT_DIR / "nae_order2_pilot"
BASELINE_JSONL = OUT_DIR / "nae_pilot" / "results.jsonl"

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


def build_order2_generate():
    """Source-patches near_axis_configuration.generate's hardcoded
    order=1 QI-consistency call to order=2, and returns the patched
    function. Done via source-string replace + exec (not a parameter of
    generate()) so we don't have to hand-copy the ~90-line closure."""
    import inspect
    src = inspect.getsource(nac.generate)
    target = "stellarator.min_geo_qi_consistency(order=1)"
    assert target in src, "near_axis_configuration.generate's source changed -- re-check the patch"
    patched_src = src.replace(target, "stellarator.min_geo_qi_consistency(order=2)")
    patched_src = patched_src.replace("def generate(", "def generate_order2(", 1)
    ns = dict(vars(nac))  # generate()'s closure needs the module's own globals (np, optimize, etc.)
    exec(compile(patched_src, "<order2-patch>", "exec"), ns)
    return ns["generate_order2"]


def build_grid():
    """Same shape as nae_pilot.py's build_grid, but mirror_ratio_target
    narrowed to the band nae_pilot's own re-mining showed actually lands
    close to the P3 mirror_ratio wall (0.15 measured at mr=0.26-0.34,
    unlike 0.25-0.30 which measured at mr=0.48+)."""
    nfps = [2, 3, 4]
    mirror_ratios = [0.10, 0.12, 0.15, 0.18, 0.20]
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
    r_cos = np.asarray(surface.r_cos)
    z_sin = np.asarray(surface.z_sin)
    assert r_cos.shape == (5, 9) and z_sin.shape == (5, 9), \
        f"unexpected shape r_cos={r_cos.shape} z_sin={z_sin.shape}"
    return r_cos, z_sin


def generate_nae_order2(generate_order2_fn, aspect_ratio, max_elongation,
                         rotational_transform, mirror_ratio, n_field_periods,
                         max_poloidal_mode=4, max_toroidal_mode=4):
    """Mirrors constellaration.initial_guess.generate_nae exactly, just
    calling the order=2-patched generate() instead of the real one."""
    near_axis_expansion = generate_order2_fn(
        mirror_ratio=mirror_ratio,
        min_iota=rotational_transform,
        max_elongation=max_elongation,
        torsion=1.33 / aspect_ratio,
        n_field_periods=n_field_periods,
        major_radius=1.0,
        max_toroidal_mode=max(3, max_toroidal_mode),
    )
    settings = nac.NearAxisToPlasmaBoundarySettings(minor_radius=1.0 / aspect_ratio)
    boundary = nac.near_axis_configuration_to_plasma_boundary(
        near_axis_configuration=near_axis_expansion,
        settings=settings,
    )
    return nac.smooth_and_set_max_mode_numbers(boundary, max_poloidal_mode, max_toroidal_mode)


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    generate_order2 = build_order2_generate()
    grid = build_grid()
    print(f"[nae_order2] {len(grid)} configs to construct + validate (order=2 QI consistency)")

    jobs, meta = [], {}
    n_construct_failed = 0
    for i, cfg in enumerate(grid):
        try:
            surface = generate_nae_order2(
                generate_order2,
                aspect_ratio=cfg["aspect_ratio"],
                max_elongation=cfg["max_elongation"],
                rotational_transform=cfg["rotational_transform"],
                mirror_ratio=cfg["mirror_ratio"],
                n_field_periods=cfg["n_field_periods"],
            )
            r_cos, z_sin = surface_to_row(surface)
        except Exception as e:  # noqa: BLE001
            n_construct_failed += 1
            print(f"[nae_order2] config {i} {cfg} FAILED to construct: {e}")
            continue
        tag = i
        jobs.append((tag, r_cos, z_sin, cfg["n_field_periods"], oracle.FIDELITY_PRESETS["low"]))
        meta[tag] = cfg | {"r_cos": r_cos.flatten().tolist(), "z_sin": z_sin.flatten().tolist()}

    print(f"[nae_order2] {len(jobs)}/{len(grid)} constructed OK ({n_construct_failed} failed), "
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

    print(f"[nae_order2] {n_converged}/{len(jobs)} converged, {n_p3_ok} satisfy ALL 5 P3 constraints")

    conv = [r for r in results if r["converged"]]
    if conv:
        qi = np.array([r["measured_targets"]["qi"] for r in conv])
        mr = np.array([r["measured_targets"]["edge_magnetic_mirror_ratio"] for r in conv])
        print(f"[nae_order2] qi range: {qi.min():.6g} - {qi.max():.6g}")
        print(f"[nae_order2] mirror_ratio range: {mr.min():.4f} - {mr.max():.4f}")
        if len(conv) > 2:
            r_corr = np.corrcoef(qi, mr)[0, 1]
            print(f"[nae_order2] qi-vs-mirror_ratio correlation across {len(conv)} converged points: r={r_corr:.4f}")

    if BASELINE_JSONL.exists():
        baseline = [json.loads(l) for l in BASELINE_JSONL.open()]
        base_conv = [r for r in baseline if r["converged"]]
        if base_conv:
            b_mr = np.array([r["measured_targets"]["edge_magnetic_mirror_ratio"] for r in base_conv])
            print(f"[nae_order2] baseline (order=1) mirror_ratio range for comparison: "
                  f"{b_mr.min():.4f} - {b_mr.max():.4f} (n={len(base_conv)})")


if __name__ == "__main__":
    main()

"""Generates a real surrogate-training dataset -- (density-field shape) ->
(compliance under a FIXED force layout) -- for a small, fixed set of force
layouts. Direct follow-up request after the candidate sweep: "I would
ideally like to do it for like... 2 or 3 force layouts?"

This mirrors the STRUCTURE of the original (fully deleted) bracket dataset
-- fixed boundary condition, vary the candidate shape, score every shape
through the same forward solver -- but fixes the two things that made the
original unusable: the boundary condition here is a real two-mount +
one-load layout (not a single clamped edge), and there are multiple real
layouts (not one), each one an already-validated case from earlier work
(v1's symmetric case, v2's asymmetric/angled case, and the strongest result
from the 16-candidate sweep, §101 -- picked because it beat its own naive
baseline by 3.52x, the most non-trivial shape seen so far).

Two real, physically-grounded sources of shape diversity per layout:
  1. SIMP optimization TRAJECTORY snapshots (bracket_simp_warp.run's new
     snapshot_every option) -- real intermediate structure from the
     uniform-density starting point through to the converged optimum, not
     synthetic noise.
  2. Random smooth density fields synthesized from low-frequency 3D DCT
     coefficients (same idea as the original dataset's DCT parameterization,
     generalized to 3D), volume-fraction-matched via bisection on a
     logistic threshold so every sample sits at a comparable, plausible
     material budget instead of an arbitrary one.

Every shape is scored via ONE real forward FEM solve (bracket_simp_warp's
own solve_u_for_rho, same SIMP-scaled Hookean stiffness used everywhere
else in this domain) against the geometry/BC machinery built ONCE per
layout and reused for every sample in that layout -- not rebuilt per shape.
"""
import argparse
import json

import numpy as np
import warp as wp
import warp.fem as fem
from scipy.fft import idctn

from bracket_fem_warp import build_bd_matrix, build_geometry, build_rhs, build_subdomains, volume_form
from bracket_simp_warp import run, solve_u_for_rho

# Three already-validated, real force layouts -- deliberately not new
# arbitrary configs. v1/v2 are the original curated cases; "candidate_12"
# is the strongest result (3.52x over naive) from the §101 sweep.
LAYOUTS = {
    "v1_symmetric": dict(
        mount_a_x=0.3, mount_b_x=1.7, mount_radius=0.12, load_x=1.0, load_radius=0.12,
        load_vec=[0.0, -0.15, 0.0], bounds_hi=[2.0, 1.0, 0.3], res=[40, 20, 6], volfrac=0.3,
    ),
    "v2_asymmetric": dict(
        mount_a_x=0.25, mount_b_x=1.5, mount_radius=0.12, load_x=1.3, load_radius=0.12,
        load_vec=[0.05, -0.14, 0.06], bounds_hi=[2.0, 1.0, 0.3], res=[56, 28, 8], volfrac=0.25,
    ),
    "candidate_12": dict(
        mount_a_x=0.21092484108526738, mount_b_x=1.7285281500310945, mount_radius=0.12,
        load_x=0.6906358737673945, load_radius=0.12,
        load_vec=[-0.028038036510791262, -0.1298513307311431, -0.02996391627596828],
        bounds_hi=[2.0, 1.0, 0.3], res=[40, 20, 6], volfrac=0.23139448951109312,
    ),
}
PENAL = 3.0
E_MIN = 1e-3
RHO_MIN = 1e-3
N_ITERS = 1500  # same checked-convergent setting used everywhere else in this domain (§100)
SNAPSHOT_EVERY = 50  # -> 30 trajectory snapshots per layout


def random_dct_density(res, rng, target_volfrac, low_freq=6, sharpness=None):
    """A smooth random density field synthesized from low-frequency 3D DCT
    coefficients (generalizes the original dataset's DCT-parameterized
    shapes to 3D), volume-fraction-matched to `target_volfrac` via
    bisection on a logistic threshold -- so random samples land at a
    comparable, plausible material budget instead of an arbitrary one."""
    kx, ky, kz = min(low_freq, res[0]), min(low_freq, res[1]), min(low_freq, res[2])
    coeffs = np.zeros(res, dtype=np.float64)
    coeffs[:kx, :ky, :kz] = rng.normal(size=(kx, ky, kz))
    field = idctn(coeffs, norm="ortho")
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)

    if sharpness is None:
        sharpness = rng.uniform(4.0, 20.0)  # varies from blobby to near-binary
    lo_t, hi_t = -1.0, 2.0
    for _ in range(40):
        mid_t = 0.5 * (lo_t + hi_t)
        rho_try = RHO_MIN + (1.0 - RHO_MIN) / (1.0 + np.exp(-sharpness * (field - mid_t)))
        if rho_try.mean() > target_volfrac:
            lo_t = mid_t
        else:
            hi_t = mid_t
    threshold = 0.5 * (lo_t + hi_t)
    rho = RHO_MIN + (1.0 - RHO_MIN) / (1.0 + np.exp(-sharpness * (field - threshold)))
    return rho.astype(np.float32)


def build_layout_context(params):
    wp.init()
    bounds_lo = wp.vec3(0.0, 0.0, 0.0)
    bounds_hi_v = wp.vec3(*params["bounds_hi"])
    res_v = wp.vec3i(*params["res"])
    geo = build_geometry(res=res_v, bounds_lo=bounds_lo, bounds_hi=bounds_hi_v)
    mount_a, mount_b, load_domain = build_subdomains(
        geo, mount_a_x=params["mount_a_x"], mount_b_x=params["mount_b_x"], mount_radius=params["mount_radius"],
        load_x=params["load_x"], load_radius=params["load_radius"],
    )
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
    u_test = fem.make_test(space=u_space)
    u_trial = fem.make_trial(space=u_space)
    bd_matrix = build_bd_matrix(u_space, mount_a, mount_b)
    rhs = build_rhs(u_space, load_domain, load_vec=wp.vec3(*params["load_vec"]))
    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    n_cells = rho_space.node_count()

    # Warp's own per-dof ordering for a degree-0 (piecewise-constant) space
    # is NOT a simple row-major flatten of the (i,j,k) resolution grid --
    # recover the mapping the same way bracket_export_mesh_warp.py does (via
    # each dof's own physical position), so a random field generated on a
    # plain (i,j,k) numpy grid can be reordered into the dof array
    # solve_u_for_rho actually expects.
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array(params["bounds_hi"])
    res = np.array(params["res"])
    cell_size = (hi - lo) / res
    centers = rho_space.node_positions().numpy()
    idx = np.round((centers - lo) / cell_size - 0.5).astype(int)
    assert idx.min() >= 0 and (idx.max(axis=0) < res).all(), "cell-center index recovery out of bounds"

    # Mount/load attachment regions, in grid-index space -- the EXACT
    # boundary-condition strip the solver itself uses (see
    # bracket_postprocess.boundary_strip_cells's own docstring for why this
    # replaced an earlier padded-3D-sphere approximation), shared by
    # connectivity cleanup (bracket_postprocess.clean_disconnected_islands)
    # so callers building a layout context once don't need to recompute
    # this per candidate.
    import bracket_postprocess as _bp
    cell_size_y = cell_size[1]
    mount_a_grid_idx = _bp.boundary_strip_cells(centers, idx, cell_size_y, params["mount_a_x"], params["mount_radius"], False, params["bounds_hi"][1])
    mount_b_grid_idx = _bp.boundary_strip_cells(centers, idx, cell_size_y, params["mount_b_x"], params["mount_radius"], False, params["bounds_hi"][1])
    load_grid_idx = _bp.boundary_strip_cells(centers, idx, cell_size_y, params["load_x"], params["load_radius"], True, params["bounds_hi"][1])

    return dict(u_space=u_space, u_test=u_test, u_trial=u_trial, bd_matrix=bd_matrix, rhs=rhs,
                rho_space=rho_space, n_cells=n_cells, res=tuple(params["res"]), grid_idx=idx,
                cell_centers=centers, bounds_lo=lo, bounds_hi=hi,
                mount_a_grid_idx=mount_a_grid_idx, mount_b_grid_idx=mount_b_grid_idx, load_grid_idx=load_grid_idx)


def grid_to_dofs(ctx, field3d):
    idx = ctx["grid_idx"]
    return field3d[idx[:, 0], idx[:, 1], idx[:, 2]]


def clean_shape(ctx, rho_dofs):
    """Removes disconnected 'floating island' material from a dof-ordered
    density field, using the layout context's own precomputed mount/load
    attachment regions. Returns (cleaned_rho_dofs, removed_fraction,
    spans_load_path) -- spans_load_path is False when nothing actually
    connects both mounts to the load (see bracket_postprocess.py's own
    docstring: callers must check this before trusting strength/mass
    numbers computed from the cleaned shape)."""
    import bracket_postprocess as _bp
    grid = _bp.dofs_to_grid(rho_dofs, ctx["grid_idx"], ctx["res"])
    cleaned_grid, removed_fraction, spans_load_path = _bp.clean_disconnected_islands(
        grid, ctx["mount_a_grid_idx"], ctx["mount_b_grid_idx"], ctx["load_grid_idx"])
    return _bp.grid_to_dofs(cleaned_grid, ctx["grid_idx"]), removed_fraction, spans_load_path


def score_shape(ctx, rho_np, return_u=False):
    rho_field = ctx["rho_space"].make_field()
    rho_field.dof_values = wp.array(rho_np.astype(np.float32), dtype=float)
    u = solve_u_for_rho(ctx["u_space"], ctx["u_trial"], ctx["u_test"], ctx["bd_matrix"], ctx["rhs"],
                         rho_field, PENAL, E_MIN, quiet=True)
    compliance = float(np.dot(ctx["rhs"].numpy().flatten(), u.numpy().flatten()))
    if return_u:
        return compliance, u
    return compliance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS.keys()), choices=list(LAYOUTS.keys()))
    ap.add_argument("--n-random", type=int, default=150, help="random DCT-shape samples per layout")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-iters-override", type=int, default=None,
                     help="For smoke-testing the pipeline mechanics only -- NOT for a real dataset "
                          "(1500 is the checked convergence point, §100).")
    ap.add_argument("--snapshot-every-override", type=int, default=None)
    args = ap.parse_args()
    n_iters = args.n_iters_override or N_ITERS
    snapshot_every = args.snapshot_every_override or SNAPSHOT_EVERY

    dataset_index = []
    for layout_name in args.layouts:
        params = LAYOUTS[layout_name]
        print(f"\n=== layout: {layout_name} ===  {params}")

        # 1. Real SIMP trajectory -- also gives us the actual full optimization
        # run for this layout, reused as this layout's "reference optimum".
        meta, snapshots = run(**params, penal=PENAL, e_min=E_MIN, rho_min=RHO_MIN, vol_penalty_weight=50.0,
                               n_iters=n_iters, lr=0.02, out_tag=f"dataset_{layout_name}_ref", quiet=True,
                               snapshot_every=snapshot_every)
        print(f"  reference optimum: compliance={meta['final_compliance']:.4f}  "
              f"naive_ratio={meta['naive_vs_simp_ratio']:.2f}x  {len(snapshots)} trajectory snapshots")

        rows_rho = [s[1] for s in snapshots]
        rows_compliance = [s[2] for s in snapshots]
        rows_source = ["simp_trajectory"] * len(snapshots)
        rows_iter = [s[0] for s in snapshots]

        # 2. Random DCT-parameterized shapes, scored against the SAME
        # geometry/BC context (built once, reused for every random sample --
        # no cost to rebuild the FEM machinery per shape).
        ctx = build_layout_context(params)
        rng = np.random.default_rng(args.seed + hash(layout_name) % 1000)
        for i in range(args.n_random):
            target_vf = float(rng.uniform(0.15, 0.45))
            field3d = random_dct_density(params["res"], rng, target_vf)
            rho_np = grid_to_dofs(ctx, field3d)  # reorder into the solver's actual dof order
            compliance = score_shape(ctx, rho_np)
            rows_rho.append(rho_np)
            rows_compliance.append(compliance)
            rows_source.append("dct_random")
            rows_iter.append(-1)
            if i % 25 == 0 or i == args.n_random - 1:
                print(f"  random sample {i:3d}/{args.n_random - 1}: volfrac={rho_np.mean():.3f} "
                      f"(target {target_vf:.3f})  compliance={compliance:.4f}")

        rho_array = np.stack([r.reshape(-1) for r in rows_rho]).astype(np.float32)
        compliance_array = np.array(rows_compliance, dtype=np.float32)
        out_path = f"/work/output/bracket_dataset_{layout_name}.npz"
        np.savez_compressed(
            out_path, rho=rho_array, compliance=compliance_array,
            source=np.array(rows_source), iteration=np.array(rows_iter, dtype=np.int32),
        )
        print(f"  saved {out_path}: {rho_array.shape[0]} rows, {rho_array.shape[1]} cells/row "
              f"(res {params['res']}), compliance range [{compliance_array.min():.4f}, {compliance_array.max():.4f}]")

        dataset_index.append({
            "layout": layout_name, "params": params, "n_rows": int(rho_array.shape[0]),
            "n_cells": int(rho_array.shape[1]), "res": params["res"],
            "reference_compliance": meta["final_compliance"],
            "reference_naive_ratio": meta["naive_vs_simp_ratio"],
            "compliance_min": float(compliance_array.min()), "compliance_max": float(compliance_array.max()),
            "n_simp_trajectory": len(snapshots), "n_dct_random": args.n_random,
            "npz_file": f"bracket_dataset_{layout_name}.npz",
        })

    with open("/work/output/bracket_dataset_index.json", "w") as f:
        json.dump(dataset_index, f, indent=2)
    print(f"\nsaved output/bracket_dataset_index.json ({len(dataset_index)} layouts)")


if __name__ == "__main__":
    main()

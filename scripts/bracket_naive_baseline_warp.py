"""Sanity check on whether bracket_simp_warp.py's SIMP optimization is doing
anything non-trivial, or just finding the "obviously correct" two-straight-
bars answer that connecting a point load to two point supports basically
forces. Direct user question: "the v shaped trusses aren't actual optimal
or anything are they? There's non-trivial design and optimization to be
done?"

Builds a NAIVE baseline density field by hand -- two straight cylindrical
tubes directly connecting each mount to the load point, tube radius chosen
(via bisection on the discretized grid) so its volume fraction matches the
SIMP run's target exactly, void everywhere else -- then runs it through the
EXACT SAME forward solver (same SIMP stiffness interpolation, same
boundary conditions) used to score the optimized result, so the comparison
is apples-to-apples at matched material budget. If SIMP's compliance is
meaningfully lower than this naive guess's, that's direct, checkable
evidence of real optimization gain, not just confirmation of the obvious.
"""
import argparse

import numpy as np
import warp as wp
import warp.fem as fem

from bracket_fem_warp import build_bd_matrix, build_geometry, build_rhs, build_subdomains
from bracket_simp_warp import solve_u_for_rho


def dist_to_segment(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(p - proj, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mount-a-x", type=float, default=0.3)
    ap.add_argument("--mount-b-x", type=float, default=1.7)
    ap.add_argument("--mount-radius", type=float, default=0.12)
    ap.add_argument("--load-x", type=float, default=1.0)
    ap.add_argument("--load-radius", type=float, default=0.12)
    ap.add_argument("--load-vec", type=float, nargs=3, default=[0.0, -0.15, 0.0])
    ap.add_argument("--bounds-hi", type=float, nargs=3, default=[2.0, 1.0, 0.3])
    ap.add_argument("--res", type=int, nargs=3, default=[40, 20, 6])
    ap.add_argument("--volfrac", type=float, default=0.3)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--e-min", type=float, default=1e-3)
    ap.add_argument("--simp-compliance", type=float, default=None,
                     help="the already-optimized run's final compliance, for a direct printed comparison")
    args = ap.parse_args()

    wp.init()
    bounds_lo = wp.vec3(0.0, 0.0, 0.0)
    bounds_hi_v = wp.vec3(*args.bounds_hi)
    res_v = wp.vec3i(*args.res)

    geo = build_geometry(res=res_v, bounds_lo=bounds_lo, bounds_hi=bounds_hi_v)
    mount_a, mount_b, load_domain = build_subdomains(
        geo, mount_a_x=args.mount_a_x, mount_b_x=args.mount_b_x, mount_radius=args.mount_radius,
        load_x=args.load_x, load_radius=args.load_radius,
    )
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
    u_test = fem.make_test(space=u_space)
    u_trial = fem.make_trial(space=u_space)
    bd_matrix = build_bd_matrix(u_space, mount_a, mount_b)
    rhs = build_rhs(u_space, load_domain, load_vec=wp.vec3(*args.load_vec))

    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    centers = rho_space.node_positions().numpy()

    z_mid = args.bounds_hi[2] / 2
    mount_a_pt = np.array([args.mount_a_x, 0.0, z_mid])
    mount_b_pt = np.array([args.mount_b_x, 0.0, z_mid])
    load_pt = np.array([args.load_x, args.bounds_hi[1], z_mid])

    d_a = dist_to_segment(centers, mount_a_pt, load_pt)
    d_b = dist_to_segment(centers, mount_b_pt, load_pt)
    d_min = np.minimum(d_a, d_b)

    # Bisection on tube radius so the discretized volume fraction exactly
    # matches the SIMP run's target -- a fair, matched-material-budget
    # comparison, not an eyeballed radius.
    lo_r, hi_r = 0.0, max(args.bounds_hi)
    for _ in range(50):
        mid = 0.5 * (lo_r + hi_r)
        frac = (d_min < mid).mean()
        if frac < args.volfrac:
            lo_r = mid
        else:
            hi_r = mid
    r_tube = 0.5 * (lo_r + hi_r)
    rho_np = np.where(d_min < r_tube, 1.0, 1e-3).astype(np.float32)
    achieved_volfrac = float(rho_np.mean())
    print(f"naive two-straight-tube baseline: r_tube={r_tube:.4f}, achieved volfrac={achieved_volfrac:.4f} (target {args.volfrac})")

    rho = wp.array(rho_np, dtype=float)
    rho_field = rho_space.make_field()
    rho_field.dof_values = rho

    u = solve_u_for_rho(u_space, u_trial, u_test, bd_matrix, rhs, rho_field, args.penal, args.e_min, quiet=True)
    compliance = float(np.dot(rhs.numpy().flatten(), u.numpy().flatten()))
    print(f"naive two-straight-tube compliance: {compliance:.6f}")

    if args.simp_compliance is not None:
        ratio = compliance / args.simp_compliance
        print(f"\nSIMP-optimized compliance: {args.simp_compliance:.6f}")
        print(f"naive / SIMP ratio: {ratio:.3f}x  ({'SIMP wins' if ratio > 1.02 else 'roughly tied' if ratio > 0.98 else 'naive wins (SIMP found something WORSE -- red flag)'})")


if __name__ == "__main__":
    main()

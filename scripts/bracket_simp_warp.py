"""Density-based (SIMP) topology optimization on top of bracket_fem_warp's
verified forward elasticity solve -- minimize compliance under a volume
budget, via NVIDIA Warp's autodiff (warp.fem), GPU-accelerated.

Sensitivity method: compliance minimization under a FIXED load is the
classic self-adjoint case in topology optimization -- the adjoint variable
equals -u itself, so no second linear solve is needed for the gradient
(this is why the standard 99-line SIMP reference code never solves an
adjoint system either; it's a well-established shortcut, not invented
here). Concretely, per outer iteration:
  1. Assemble K(rho) and solve u = K(rho)^-1 f via CG (forward only, no
     autodiff tracked -- rho is fixed for this step).
  2. Under a wp.Tape, recompute compliance = integral of the elasticity
     bilinear form evaluated AT THE FIXED u (a plain scalar functional of
     rho, same "loss_form of a discrete field" pattern used in NVIDIA's
     own example_elastic_shape_optimization.py) plus a volume-budget
     penalty term.
  3. tape.backward() gives d(compliance + volume_penalty)/d(rho) directly
     -- correct because u is held fixed in this pass, matching the
     self-adjoint shortcut exactly.
  4. Adam step on rho, clamp to [rho_min, 1].

Full CLI added after the first verified run -- direct follow-up request
("let's go forward and see if we can actually optimize one ourselves")
needs a genuinely re-runnable tool with real parameters, not a script with
the one already-seen case hardcoded.
"""
import argparse
import json

import numpy as np
import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.optim import Adam

import bracket_postprocess
from bracket_fem_warp import (
    LAME_LAMBDA, LAME_MU, build_bd_matrix, build_geometry, build_rhs, build_subdomains,
    volume_form,
)


def dist_to_segment(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(p - proj, axis=1)


@wp.func
def simp_scale(rho: float, penal: float, e_min: float):
    r = wp.max(rho, 0.0)
    return e_min + wp.pow(r, penal) * (1.0 - e_min)


@fem.integrand
def hooke_elasticity_form_simp(s: fem.Sample, u: fem.Field, v: fem.Field, rho: fem.Field, penal: float, e_min: float):
    scale = simp_scale(rho(s), penal, e_min)
    lam = LAME_LAMBDA * scale
    mu = LAME_MU * scale
    strain_u = fem.D(u, s)
    stress_u = 2.0 * mu * strain_u + lam * wp.trace(strain_u) * wp.identity(n=3, dtype=float)
    return wp.ddot(fem.D(v, s), stress_u)


@fem.integrand
def neg_compliance_functional(s: fem.Sample, u: fem.Field, rho: fem.Field, penal: float, e_min: float):
    """NEGATIVE of the scalar energy functional u^T K(rho) u, evaluated at a
    FIXED discrete field u (u held constant -- matching the self-adjoint
    compliance-sensitivity shortcut). The sign matters: true compliance
    C(rho) = f^T u(rho) satisfies dC/drho_e = -u^T (dK/drho_e) u (standard
    adjoint derivation, K u = f differentiated w.r.t. rho_e). Differentiating
    +u^T K(rho) u at fixed u gives +u^T (dK/drho_e) u -- the WRONG sign for
    a compliance-minimizing gradient descent step. Differentiating this
    NEGATED version gives the correctly-signed dC/drho_e directly. (Caught
    empirically: an earlier unsigned version made compliance visibly get
    WORSE over iterations while mean density collapsed -- exactly the
    signature of a flipped gradient.)"""
    scale = simp_scale(rho(s), penal, e_min)
    lam = LAME_LAMBDA * scale
    mu = LAME_MU * scale
    strain_u = fem.D(u, s)
    stress_u = 2.0 * mu * strain_u + lam * wp.trace(strain_u) * wp.identity(n=3, dtype=float)
    return -wp.ddot(strain_u, stress_u)


@fem.integrand
def mean_density_functional(s: fem.Sample, rho: fem.Field):
    return rho(s)


@fem.integrand
def von_mises_functional(s: fem.Sample, u: fem.Field, rho: fem.Field, penal: float, e_min: float, out: wp.array(dtype=float)):
    """Per-cell von Mises stress at the FINAL converged (rho, u) -- direct
    user request ("a way to view the strain and forces"). Uses the same
    SIMP-scaled stress as the forward solve (not the raw-material stress),
    so a cell at rho~rho_min genuinely reads near-zero stress rather than
    reporting the stress the solid material would carry if it were there.
    Standard von Mises from the full 3x3 stress tensor (this is a 3D solid,
    not plane stress/strain, matching bracket_fem_warp.py's own 3D Lame
    parameters)."""
    scale = simp_scale(rho(s), penal, e_min)
    lam = LAME_LAMBDA * scale
    mu = LAME_MU * scale
    strain_u = fem.D(u, s)
    stress_u = 2.0 * mu * strain_u + lam * wp.trace(strain_u) * wp.identity(n=3, dtype=float)
    s11 = stress_u[0, 0]
    s22 = stress_u[1, 1]
    s33 = stress_u[2, 2]
    s12 = stress_u[0, 1]
    s13 = stress_u[0, 2]
    s23 = stress_u[1, 2]
    vm = wp.sqrt(0.5 * ((s11 - s22) * (s11 - s22) + (s22 - s33) * (s22 - s33) + (s33 - s11) * (s33 - s11)
                         + 6.0 * (s12 * s12 + s13 * s13 + s23 * s23)))
    out[s.qp_index] = vm


@wp.kernel
def add_volume_penalty(loss: wp.array(dtype=float), vol_integral: wp.array(dtype=float), domain_volume: float, target: float, weight: float):
    # vol_integral is the RAW integral of rho over the domain (fem.integrate
    # returns integrals, not means), so it's divided by the domain's total
    # volume to get the actual mean density (volume fraction) before
    # comparing to `target`.
    mean_rho = vol_integral[0] / domain_volume
    diff = mean_rho - target
    loss[0] += weight * diff * diff


@wp.kernel
def clamp_rho(rho: wp.array(dtype=float), rho_min: float):
    i = wp.tid()
    rho[i] = wp.clamp(rho[i], rho_min, 1.0)


def solve_u_for_rho(u_space, u_trial, u_test, bd_matrix, rhs, rho_field, penal, e_min, quiet=True):
    matrix = fem.integrate(
        hooke_elasticity_form_simp,
        fields={"u": u_trial, "v": u_test, "rho": rho_field},
        values={"penal": penal, "e_min": e_min},
        output_dtype=float,
    )
    bd_rhs = wp.zeros_like(rhs)
    rhs_copy = wp.clone(rhs)
    fem.project_linear_system(matrix, rhs_copy, bd_matrix, bd_rhs)
    u = wp.zeros_like(rhs)
    fem_example_utils.bsr_cg(matrix, b=rhs_copy, x=u, quiet=quiet, tol=1e-8, max_iters=2000)
    return u


def run(mount_a_x, mount_b_x, mount_radius, load_x, load_radius, load_vec,
        bounds_hi, res, volfrac, penal, e_min, rho_min, vol_penalty_weight,
        n_iters, lr, out_tag, quiet=True, snapshot_every=None):
    """snapshot_every: if set, also returns a list of (iter, rho_np) density
    snapshots taken every `snapshot_every` iterations along the optimization
    trajectory -- used by bracket_generate_dataset_warp.py as one real,
    physically-meaningful source of diverse (non-optimal) shapes for a
    surrogate dataset (the trajectory runs from the uniform initial density
    through to the converged optimum, so it's real intermediate structure,
    not noise)."""
    wp.init()
    bounds_lo = wp.vec3(0.0, 0.0, 0.0)
    bounds_hi_v = wp.vec3(*bounds_hi)
    res_v = wp.vec3i(*res)

    geo = build_geometry(res=res_v, bounds_lo=bounds_lo, bounds_hi=bounds_hi_v)
    mount_a, mount_b, load_domain = build_subdomains(
        geo, mount_a_x=mount_a_x, mount_b_x=mount_b_x, mount_radius=mount_radius,
        load_x=load_x, load_radius=load_radius,
    )
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
    u_test = fem.make_test(space=u_space)
    u_trial = fem.make_trial(space=u_space)

    bd_matrix = build_bd_matrix(u_space, mount_a, mount_b)
    rhs = build_rhs(u_space, load_domain, load_vec=wp.vec3(*load_vec))

    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    n_cells = rho_space.node_count()
    print(f"n_cells (density dofs): {n_cells}, n_displacement_dofs: {u_space.node_count()}")

    rho = wp.array(np.full(n_cells, volfrac, dtype=np.float32), dtype=float, requires_grad=True)
    rho_field = rho_space.make_field()

    optimizer = Adam([rho], lr=lr)
    cell_domain = fem.Cells(geometry=geo)
    domain_volume_arr = wp.empty(shape=1, dtype=float)
    fem.integrate(volume_form, domain=cell_domain, output=domain_volume_arr)
    domain_volume = float(domain_volume_arr.numpy()[0])
    print(f"domain volume: {domain_volume:.4f}")

    history = []
    snapshots = [] if snapshot_every else None
    for it in range(n_iters):
        rho_field.dof_values = rho
        u = solve_u_for_rho(u_space, u_trial, u_test, bd_matrix, rhs, rho_field, penal, e_min, quiet=quiet)
        u_field = u_space.make_field()
        u_field.dof_values = u

        real_compliance = float(np.dot(rhs.numpy().flatten(), u.numpy().flatten()))
        if snapshot_every and it % snapshot_every == 0:
            snapshots.append((it, rho.numpy().copy(), real_compliance))

        rho.grad.zero_()
        loss = wp.zeros(shape=1, dtype=float, requires_grad=True)
        vol_integral = wp.zeros(shape=1, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            rho_field.dof_values = rho
            fem.integrate(
                neg_compliance_functional,
                domain=cell_domain,
                fields={"u": u_field, "rho": rho_field},
                values={"penal": penal, "e_min": e_min},
                output=loss,
            )
            fem.integrate(mean_density_functional, domain=cell_domain, fields={"rho": rho_field}, output=vol_integral)
            wp.launch(add_volume_penalty, dim=1, inputs=[loss, vol_integral, domain_volume, volfrac, vol_penalty_weight])
        tape.backward(loss=loss)

        mean_rho_val = float(vol_integral.numpy()[0]) / domain_volume
        history.append((it, real_compliance, mean_rho_val))
        if it % 10 == 0 or it == n_iters - 1:
            print(f"iter {it:4d}  real_compliance={real_compliance:.6f}  mean_rho={mean_rho_val:.4f}")

        optimizer.step([rho.grad])
        wp.launch(clamp_rho, dim=n_cells, inputs=[rho, rho_min])
        tape.zero()

    rho_np = rho.numpy()
    cell_centers = rho_space.node_positions().numpy()
    print(f"\nfinal mean rho: {rho_np.mean():.4f} (target {volfrac})")
    print(f"final rho min/max: {rho_np.min():.4f} / {rho_np.max():.4f}")
    print(f"fraction of cells with rho > 0.5: {(rho_np > 0.5).mean():.4f}")

    z_mid = bounds_hi[2] / 2
    mount_a_pt = np.array([mount_a_x, 0.0, z_mid])
    mount_b_pt = np.array([mount_b_x, 0.0, z_mid])
    load_pt = np.array([load_x, bounds_hi[1], z_mid])

    # Remove disconnected "floating island" material -- direct user
    # feedback ("the floating islands and stuff"). SIMP's e_min baseline
    # means nothing is ever discretely disconnected in the SOLVER's own
    # math, so a visually-floating high-density blob (helps compliance a
    # little via that weak baseline coupling, but isn't a real load path)
    # is a real possible outcome, not a rendering artifact -- checked and
    # fixed here, not just at render time, so every downstream number
    # (compliance, stress, mass) reflects the actually-connected shape.
    grid_idx = bracket_postprocess.compute_grid_idx(cell_centers, [0.0, 0.0, 0.0], bounds_hi, res)
    rho_grid = bracket_postprocess.dofs_to_grid(rho_np, grid_idx, res)
    cell_size_y = bounds_hi[1] / res[1]
    mount_a_idx = bracket_postprocess.boundary_strip_cells(cell_centers, grid_idx, cell_size_y, mount_a_x, mount_radius, False, bounds_hi[1])
    mount_b_idx = bracket_postprocess.boundary_strip_cells(cell_centers, grid_idx, cell_size_y, mount_b_x, mount_radius, False, bounds_hi[1])
    load_idx = bracket_postprocess.boundary_strip_cells(cell_centers, grid_idx, cell_size_y, load_x, load_radius, True, bounds_hi[1])
    cleaned_grid, removed_fraction, spans_load_path = bracket_postprocess.clean_disconnected_islands(
        rho_grid, mount_a_idx, mount_b_idx, load_idx, threshold=0.5, rho_min=rho_min)
    rho_np = bracket_postprocess.grid_to_dofs(cleaned_grid, grid_idx)
    if removed_fraction > 0:
        print(f"connectivity cleanup: removed {removed_fraction:.4f} of the domain's cells as disconnected islands")
    if not spans_load_path:
        print("WARNING: no material connects both mounts to the load at all -- this result has no real load path "
              "(SIMP essentially failed to converge to a functioning structure); strength/mass numbers below are not meaningful.")

    # Per-cell von Mises stress at the FINAL, CLEANED (rho, u) -- direct
    # user request ("a way to view the strain and forces"). Resolves u for
    # the cleaned rho (not the loop's last `u`, and not the pre-cleanup
    # rho) so every exported number is consistent with the shape actually
    # rendered/measured.
    rho = wp.array(rho_np.astype(np.float32), dtype=float)
    rho_field.dof_values = rho
    u_final = solve_u_for_rho(u_space, u_trial, u_test, bd_matrix, rhs, rho_field, penal, e_min, quiet=quiet)
    u_field_final = u_space.make_field()
    u_field_final.dof_values = u_final
    von_mises_arr = wp.zeros(shape=n_cells, dtype=float)
    fem.interpolate(
        von_mises_functional,
        at=cell_domain,
        fields={"u": u_field_final, "rho": rho_field},
        values={"penal": penal, "e_min": e_min, "out": von_mises_arr},
    )
    von_mises_np = von_mises_arr.numpy()
    print(f"von Mises stress: min={von_mises_np.min():.4f}  max={von_mises_np.max():.4f}  mean={von_mises_np.mean():.4f}")
    cleaned_compliance = float(np.dot(rhs.numpy().flatten(), u_final.numpy().flatten()))

    # Naive two-straight-tube sanity check, run automatically on every SIMP
    # run -- §100: a "converged-looking" result (real V-truss shape, smooth
    # monotonic compliance curve, volume fraction near target) is NOT proof
    # the optimizer did anything non-trivial. §100 caught a real case where
    # the "optimized" result was actually BEATEN by the obvious two-straight
    # -tubes guess at matched material budget, because 150 iterations
    # (the original default) was genuinely under-converged. Comparing
    # against this baseline on every run, not just when something looks
    # suspicious, is the fix -- an under-converged result silently claiming
    # victory is a much easier mistake to make than to notice by eye.
    d_a = dist_to_segment(cell_centers, mount_a_pt, load_pt)
    d_b = dist_to_segment(cell_centers, mount_b_pt, load_pt)
    d_min = np.minimum(d_a, d_b)
    lo_r, hi_r = 0.0, max(bounds_hi)
    for _ in range(50):
        mid_r = 0.5 * (lo_r + hi_r)
        frac = (d_min < mid_r).mean()
        if frac < volfrac:
            lo_r = mid_r
        else:
            hi_r = mid_r
    r_tube = 0.5 * (lo_r + hi_r)
    naive_rho_np = np.where(d_min < r_tube, 1.0, rho_min).astype(np.float32)
    naive_rho = wp.array(naive_rho_np, dtype=float)
    naive_rho_field = rho_space.make_field()
    naive_rho_field.dof_values = naive_rho
    naive_u = solve_u_for_rho(u_space, u_trial, u_test, bd_matrix, rhs, naive_rho_field, penal, e_min, quiet=True)
    naive_compliance = float(np.dot(rhs.numpy().flatten(), naive_u.numpy().flatten()))
    final_compliance = cleaned_compliance
    naive_ratio = naive_compliance / final_compliance
    verdict = "SIMP wins" if naive_ratio > 1.02 else "roughly tied" if naive_ratio > 0.98 else "NAIVE WINS -- likely under-converged, rerun with more --n-iters"
    print(f"\nnaive two-straight-tube baseline (matched volume fraction): compliance={naive_compliance:.6f}")
    print(f"naive / SIMP compliance ratio: {naive_ratio:.3f}x  ({verdict})")

    # Real, physically-grounded metrics (mass, max safe load) instead of
    # raw SIMP compliance -- direct user feedback ("SIMP compliance is
    # also like... not an interesting metric... how much load can it take
    # and what's the weight"). See bracket_postprocess.py's own docstring
    # for the full unit-conversion derivation (Aluminum 6061, 200x100x30mm
    # bracket envelope, both direct user choices).
    real_material_fraction = float((rho_np >= 0.5).mean())
    metrics = bracket_postprocess.real_metrics(load_vec, float(von_mises_np.max()), real_material_fraction, domain_volume)
    print(f"real units (Al6061, 200x100x30mm envelope): mass={metrics['mass_kg']*1000:.1f}g  "
          f"nominal load={metrics['nominal_load_N']:.1f}N  safety factor={metrics['safety_factor']:.2f}  "
          f"max load={metrics['max_load_kgf']:.1f}kgf")

    np.save(f"/work/output/{out_tag}_rho.npy", rho_np.astype(np.float32))
    np.save(f"/work/output/{out_tag}_cell_centers.npy", cell_centers.astype(np.float32))
    np.save(f"/work/output/{out_tag}_von_mises.npy", von_mises_np.astype(np.float32))
    meta = {
        "mount_a_x": mount_a_x, "mount_b_x": mount_b_x, "mount_radius": mount_radius,
        "load_x": load_x, "load_radius": load_radius, "load_vec": list(load_vec),
        "bounds_lo": [0.0, 0.0, 0.0], "bounds_hi": list(bounds_hi), "res": list(res),
        "volfrac": volfrac, "penal": penal, "e_min": e_min, "n_iters": n_iters,
        "final_compliance": final_compliance, "final_mean_rho": float(rho_np.mean()),
        "islands_removed_fraction": removed_fraction, "spans_load_path": spans_load_path,
        "compliance_history": [h[1] for h in history],
        "naive_baseline_compliance": naive_compliance, "naive_vs_simp_ratio": naive_ratio,
        "von_mises_min": float(von_mises_np.min()), "von_mises_max": float(von_mises_np.max()),
        "von_mises_mean": float(von_mises_np.mean()),
        **metrics,
    }
    with open(f"/work/output/{out_tag}_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"saved output/{out_tag}_rho.npy, output/{out_tag}_cell_centers.npy, output/{out_tag}_von_mises.npy, output/{out_tag}_meta.json")
    if snapshot_every:
        return meta, snapshots
    return meta


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
    ap.add_argument("--rho-min", type=float, default=1e-3)
    ap.add_argument("--vol-penalty-weight", type=float, default=50.0)
    # 150 (the original §98/§99 default) was genuinely under-converged --
    # confirmed by comparing against a naive two-straight-tube baseline
    # (bracket_naive_baseline_warp.py) at matched volume fraction: the
    # 150-iteration result actually LOST to the naive guess (0.318 vs
    # 0.277 compliance) despite using MORE material. Running to 1500
    # iterations converges to 0.259 (flat to 5 decimal places by ~iter
    # 1420), which genuinely beats the naive baseline by ~7%. 1500 is a
    # real, checked convergence point for this problem size, not a guess.
    ap.add_argument("--n-iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out-tag", default="bracket_warp_simp")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(
        mount_a_x=args.mount_a_x, mount_b_x=args.mount_b_x, mount_radius=args.mount_radius,
        load_x=args.load_x, load_radius=args.load_radius, load_vec=args.load_vec,
        bounds_hi=args.bounds_hi, res=args.res, volfrac=args.volfrac, penal=args.penal,
        e_min=args.e_min, rho_min=args.rho_min, vol_penalty_weight=args.vol_penalty_weight,
        n_iters=args.n_iters, lr=args.lr, out_tag=args.out_tag, quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

"""vmec_jax as a drop-in alternative oracle to VMEC++, computing the exact
same 11 ConStellaration metrics via the exact same constellaration analysis
code (constellaration.forward_model.forward_model, reproduced metric-by-
metric below) -- only the solve itself differs.

The MHD problem itself (mode resolution, multigrid ladder, pressure/current
profile, phiedge, ...) is built via constellaration.mhd.vmec_utils.
build_vmecpp_indata -- the exact function VMEC++'s own run_vmec() calls --
and every field is ported into vmec_jax's VmecInput, rather than re-derived
from just the boundary array (see vmec_jax_forward_model's docstring for why
that matters). indata.rbc/zbs are [m, n+ntor] (poloidal-outer, matching this
project's own r_cos/z_sin, since both descend from the same VMEC2000-lineage
convention), while vmec_jax's dense rbc/zbs are [n+ntor, m] (toroidal-outer)
-- a transpose, not a reshape, and a real bug was traced to exactly this
mismatch during evaluation (see EXPERIMENT_LOG).

vmec_jax's solve is converted back into a real constellaration
vmec_utils.VmecppWOut by round-tripping through a standard wout.nc file
(vmec_jax's own writer -> vmecpp's own reader) rather than hand-rolling an
adapter object -- this is the same wout schema VMEC++ itself produces, and
is what makes the *existing* metric code reusable unmodified.

Runs in the `oracle` docker-compose service (Dockerfile.oracle), which has
vmecpp + constellaration + vmec-jax in one environment -- confirmed
empirically that this coexistence is fine (they all want numpy>=2; the
numpy<2 pin in Dockerfile.train's `train` service was only ever for torch,
which isn't present here).
"""
import tempfile

import numpy as np


def vmec_jax_forward_model(r_cos, z_sin, nfp):
    """Returns (metrics, equilibrium) like constellaration.forward_model.forward_model,
    or (None, None) if vmec_jax's solve didn't converge (treat as a rejection,
    same as a VMEC++ non-convergence).

    The MHD problem (mode resolution, multigrid ladder, pressure/current
    profile, phiedge, ...) is not re-derived here -- it's built via
    constellaration's own vmec_utils.build_vmecpp_indata, the exact same
    function VMEC++'s own run_vmec() calls, and every field is ported
    across to vmec_jax's VmecInput. Re-deriving these independently was
    tried and is a real source of false mismatches: an earlier pass used
    mpol=5/ntor=4 (this boundary's raw stored resolution) and a single-stage
    [31]/[1e-10]/[1500] ladder, while VMEC++'s actual default settings solve
    at mpol=6/ntor=6 with a 2-stage [25,71]/[1e-17,1e-13]/[2000,20000]
    ladder and a prescribed (ncurr=1, current-free) pressure profile -- none
    of which is visible from the boundary array alone.
    """
    from vmec_jax.core.input import VmecInput
    from vmec_jax.core.multigrid import solve_multigrid
    from vmec_jax.core.wout import wout_from_state, write_wout

    import vmecpp
    from constellaration.boozer import boozer as boozer_module
    from constellaration.forward_model import ConstellarationMetrics, ConstellarationSettings
    from constellaration.geometry import radial_profile, surface_rz_fourier, surface_utils
    from constellaration.mhd import (
        geometry_utils, ideal_mhd_parameters, magnetics_utils, turbulent_transport, vmec_utils,
    )
    from constellaration.mhd import vmec_settings as vmec_settings_module
    from constellaration.omnigeneity import qi as qi_module

    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=int(nfp), is_stellarator_symmetric=True)

    settings = ConstellarationSettings()
    mhd_params = ideal_mhd_parameters.boundary_to_ideal_mhd_parameters(boundary)
    vmec_settings = vmec_settings_module.create_vmec_settings_from_preset(
        boundary, settings=settings.vmec_preset_settings)
    indata = vmec_utils.build_vmecpp_indata(
        mhd_parameters=mhd_params, boundary=boundary, vmec_settings=vmec_settings)

    # indata.rbc/zbs are [m, n+ntor] (matches this project's own r_cos/z_sin
    # layout -- both came from a VMEC2000-lineage convention) while
    # vmec_jax's dense rbc/zbs are [n+ntor, m] -- transpose, not reshape.
    inp = VmecInput(
        nfp=int(indata.nfp), mpol=int(indata.mpol), ntor=int(indata.ntor), lasym=False,
        rbc=np.asarray(indata.rbc).T, zbs=np.asarray(indata.zbs).T,
        ns_array=np.asarray(indata.ns_array), ftol_array=np.asarray(indata.ftol_array),
        niter_array=np.asarray(indata.niter_array), delt=float(indata.delt),
        phiedge=float(indata.phiedge), ncurr=int(indata.ncurr),
        pmass_type=indata.pmass_type, am=np.asarray(indata.am), pres_scale=float(indata.pres_scale),
        pcurr_type=indata.pcurr_type, ac=np.asarray(indata.ac), curtor=float(indata.curtor),
        gamma=float(indata.gamma),
    )
    result = solve_multigrid(inp)
    if not result.converged:
        return None, None

    wout = wout_from_state(inp=inp, state=result.state, niter=result.iterations,
                            fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql)

    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        write_wout(f.name, wout)
        vmecpp_wout = vmecpp.VmecWOut.from_wout_file(f.name)
    equilibrium = vmec_utils.vmecppwout_from_wout(vmecpp_wout)

    n_poloidal_points, n_toroidal_points = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=equilibrium.mpol, max_toroidal_mode=equilibrium.ntor)

    max_elongation = geometry_utils.max_elongation(
        equilibrium=equilibrium, n_poloidal_points=n_poloidal_points, n_toroidal_points=n_toroidal_points)
    average_triangularity = geometry_utils.average_triangularity(surface=boundary)

    normalized_effective_radius_on_full_grid_mesh = np.sqrt(equilibrium.normalized_toroidal_flux_full_grid_mesh)
    iota = radial_profile.InterpolatedRadialProfile(
        rho=normalized_effective_radius_on_full_grid_mesh, values=equilibrium.iotaf)
    axis_rotational_transform = float(radial_profile.evaluate_at_normalized_effective_radius(iota, np.array([0.0])))
    edge_rotational_transform = float(radial_profile.evaluate_at_normalized_effective_radius(iota, np.array([1.0])))

    vacuum_well = magnetics_utils.vacuum_well(equilibrium)
    magnetic_mirror_ratio = magnetics_utils.magnetic_mirror_ratio(equilibrium)
    axis_magnetic_mirror_ratio = float(
        radial_profile.evaluate_at_normalized_effective_radius(magnetic_mirror_ratio, np.array([0.0])))
    edge_magnetic_mirror_ratio = float(
        radial_profile.evaluate_at_normalized_effective_radius(magnetic_mirror_ratio, np.array([1.0])))

    phi_upper_bound = 2 * np.pi / equilibrium.n_field_periods / (1 + int(not equilibrium.lasym))
    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta=n_poloidal_points, n_phi=n_toroidal_points, phi_upper_bound=phi_upper_bound, include_endpoints=True)
    minimum_normalized_magnetic_gradient_scale_length = (
        np.min(magnetics_utils.normalized_magnetic_gradient_scale_length(equilibrium, theta_phi))
        * equilibrium.n_field_periods)

    boozer_settings = boozer_module.create_boozer_settings_from_equilibrium_resolution(
        mhd_equilibrium=equilibrium, settings=settings.boozer_preset_settings)
    boozer = boozer_module.run_boozer(equilibrium=equilibrium, settings=boozer_settings)
    qi_metrics = qi_module.quasi_isodynamicity_residual(boozer=boozer, settings=settings.qi_settings)
    qi_residuals = float(np.sum(qi_metrics.residuals**2))

    flux_compression = turbulent_transport.compute_flux_compression_in_regions_of_bad_curvature(
        equilibrium=equilibrium, settings=settings.turbulent_settings)

    metrics = ConstellarationMetrics(
        aspect_ratio=equilibrium.aspect,
        aspect_ratio_over_edge_rotational_transform=equilibrium.aspect / edge_rotational_transform,
        max_elongation=max_elongation,
        edge_rotational_transform_over_n_field_periods=edge_rotational_transform / equilibrium.n_field_periods,
        axis_rotational_transform_over_n_field_periods=axis_rotational_transform / equilibrium.n_field_periods,
        average_triangularity=average_triangularity,
        vacuum_well=vacuum_well,
        axis_magnetic_mirror_ratio=axis_magnetic_mirror_ratio,
        edge_magnetic_mirror_ratio=edge_magnetic_mirror_ratio,
        minimum_normalized_magnetic_gradient_scale_length=minimum_normalized_magnetic_gradient_scale_length,
        qi=qi_residuals,
        flux_compression_in_regions_of_bad_curvature=flux_compression,
    )
    return metrics, equilibrium


if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--row-index", type=int, default=0)
    args = p.parse_args()

    X = np.load("/work/output/X.npy")
    row = X[args.row_index]
    r_cos = row[:45].reshape(5, 9)
    z_sin = row[45:90].reshape(5, 9)
    nfp = int(row[90])

    t0 = time.perf_counter()
    metrics, _ = vmec_jax_forward_model(r_cos, z_sin, nfp)
    dt = time.perf_counter() - t0

    if metrics is None:
        print(f"row {args.row_index}: did not converge ({dt:.2f}s)")
    else:
        print(f"row {args.row_index}: converged in {dt:.2f}s")
        for name in ["aspect_ratio", "max_elongation", "qi", "flux_compression_in_regions_of_bad_curvature",
                     "vacuum_well", "axis_magnetic_mirror_ratio", "edge_magnetic_mirror_ratio",
                     "axis_rotational_transform_over_n_field_periods",
                     "edge_rotational_transform_over_n_field_periods", "average_triangularity",
                     "minimum_normalized_magnetic_gradient_scale_length"]:
            print(f"  {name:55s} {getattr(metrics, name)}")

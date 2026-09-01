"""3D linear-elasticity FEM solve for the bracket topology-optimization
domain, via NVIDIA Warp's warp.fem module -- replaces the self-written
scipy plane-stress 2D solver removed in full (EXPERIMENT_LOG §96) after it
was judged not physically sound (single-fixed-edge cantilever, not a real
bracket; a colormap render that could never show material as absent).

Direct user decisions this rebuild is based on: 3D (not 2D), and the domain
is fixed at TWO separate mount regions with a load applied at a third,
distinct region -- not a single clamped edge. "Use warp" was an explicit
instruction not to hand-write another solver from scratch; this module
leans on warp.fem's own elasticity/boundary-projector/linear-solve
machinery, grounded directly against NVIDIA's own current source (fetched
2026-08-31, not guessed from possibly-stale model knowledge):
  - warp/examples/fem/example_elastic_shape_optimization.py for the
    Hookean-elasticity weak form, Dirichlet boundary projectors, and the
    wp.Tape-based autodiff pattern this module's optimizer stage will reuse.
  - warp/examples/fem/example_diffusion_3d.py for the Grid3D geometry
    constructor and the project_linear_system + bsr_cg solve pattern.

§98's first version hardcoded mount/load positions as Python module-level
globals referenced directly inside @fem.integrand kernels -- fine for a
single correctness check, but Warp bakes global lookups into the compiled
kernel at JIT time, so changing them per-run isn't safe/defined. Direct
follow-up request ("let's go forward and see if we can actually optimize
one ourselves") needs real per-run parameterization, so mount/load
positions, radius, and the load vector are now explicit kernel/function
arguments threaded through everywhere, not module constants -- the same
geometry/BC/solve code now genuinely supports a different case, not just
the one symmetric config already verified.
"""
import numpy as np
import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

wp.init()

# Defaults matching §98's original verified case -- still the module's
# fallback, but every function below takes these as real arguments now.
BOUNDS_LO = wp.vec3(0.0, 0.0, 0.0)
BOUNDS_HI = wp.vec3(2.0, 1.0, 0.3)
RES = wp.vec3i(40, 20, 6)

MOUNT_A_X = 0.3
MOUNT_B_X = 1.7
MOUNT_RADIUS = 0.12
LOAD_X = 1.0
LOAD_RADIUS = 0.12
LOAD_VEC = wp.vec3(0.0, -0.15, 0.0)

E0 = 1.0
NU = 0.3
# Standard isotropic 3D Lame parameters from Young's modulus / Poisson ratio
# (NOT the plane-strain-specific vec2 formula in the 2D shape-optimization
# example above -- that one bakes in a 2D plane-strain assumption that
# doesn't apply here).
LAME_MU = E0 / (2.0 * (1.0 + NU))
LAME_LAMBDA = E0 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
LAME = wp.vec2(LAME_LAMBDA, LAME_MU)


@fem.integrand
def classify_boundary_regions(
    s: fem.Sample,
    domain: fem.Domain,
    mount_a: wp.array(dtype=int),
    mount_b: wp.array(dtype=int),
    load: wp.array(dtype=int),
    mount_a_x: float,
    mount_b_x: float,
    mount_radius: float,
    load_x: float,
    load_radius: float,
):
    pos = domain(s)
    nor = fem.normal(domain, s)
    if nor[1] < -0.5:  # bottom face (y=0)
        if wp.abs(pos[0] - mount_a_x) < mount_radius:
            mount_a[s.qp_index] = 1
        elif wp.abs(pos[0] - mount_b_x) < mount_radius:
            mount_b[s.qp_index] = 1
    elif nor[1] > 0.5:  # top face (y=1)
        if wp.abs(pos[0] - load_x) < load_radius:
            load[s.qp_index] = 1


@fem.integrand
def boundary_projector_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@wp.func
def hooke_stress(strain: wp.mat33, lame: wp.vec2):
    lam = lame[0]
    mu = lame[1]
    return 2.0 * mu * strain + lam * wp.trace(strain) * wp.identity(n=3, dtype=float)


@fem.integrand
def stress_field(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    return hooke_stress(fem.D(u, s), lame)


@fem.integrand
def hooke_elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), stress_field(s, u, lame))


@fem.integrand
def volume_form():
    return 1.0


@fem.integrand
def load_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, load: wp.vec3, inv_area: wp.array(dtype=float)):
    return wp.dot(v(s), load) * inv_area[0]


@wp.kernel
def invert_scalar(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[0] = 1.0 / x[0]


def build_geometry(res=RES, bounds_lo=BOUNDS_LO, bounds_hi=BOUNDS_HI):
    return fem.Grid3D(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)


def build_subdomains(geo, mount_a_x=MOUNT_A_X, mount_b_x=MOUNT_B_X, mount_radius=MOUNT_RADIUS,
                      load_x=LOAD_X, load_radius=LOAD_RADIUS):
    boundary = fem.BoundarySides(geo)
    mount_a_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
    mount_b_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
    load_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
    fem.interpolate(
        classify_boundary_regions,
        at=boundary,
        values={
            "mount_a": mount_a_mask, "mount_b": mount_b_mask, "load": load_mask,
            "mount_a_x": mount_a_x, "mount_b_x": mount_b_x, "mount_radius": mount_radius,
            "load_x": load_x, "load_radius": load_radius,
        },
    )
    mount_a = fem.Subdomain(boundary, element_mask=mount_a_mask)
    mount_b = fem.Subdomain(boundary, element_mask=mount_b_mask)
    load = fem.Subdomain(boundary, element_mask=load_mask)
    return mount_a, mount_b, load


def build_rhs(u_space, load_domain, load_vec=LOAD_VEC):
    load_test = fem.make_test(space=u_space, domain=load_domain)
    load_area = wp.empty(shape=1, dtype=float)
    fem.integrate(volume_form, domain=load_domain, output=load_area)
    inv_area = wp.empty(shape=1, dtype=float)
    wp.launch(invert_scalar, dim=1, inputs=[load_area, inv_area])
    return fem.integrate(load_form, fields={"v": load_test}, values={"load": load_vec, "inv_area": inv_area}, output_dtype=wp.vec3)


def build_bd_matrix(u_space, mount_a, mount_b):
    bd_matrix = None
    for sub in (mount_a, mount_b):
        bd_test = fem.make_test(space=u_space, domain=sub)
        bd_trial = fem.make_trial(space=u_space, domain=sub)
        m = fem.integrate(boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal", output_dtype=float)
        bd_matrix = m if bd_matrix is None else bd_matrix + m
    fem.normalize_dirichlet_projector(bd_matrix)
    return bd_matrix


def solve_forward(geo, u_space, mount_a, mount_b, load_domain, load_vec=LOAD_VEC, lame=LAME, degree=1, quiet=True):
    """Forward elasticity solve at uniform density (E0 everywhere). Returns
    (u_field, compliance) -- compliance = 0.5 * u^T K u, the standard
    topology-optimization objective (strain energy under load)."""
    u_test = fem.make_test(space=u_space)
    u_trial = fem.make_trial(space=u_space)

    bd_matrix = build_bd_matrix(u_space, mount_a, mount_b)
    rhs = build_rhs(u_space, load_domain, load_vec)
    matrix = fem.integrate(hooke_elasticity_form, fields={"u": u_trial, "v": u_test}, values={"lame": lame}, output_dtype=float)

    bd_rhs = wp.zeros_like(rhs)
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

    u = wp.zeros_like(rhs)
    fem_example_utils.bsr_cg(matrix, b=rhs, x=u, quiet=quiet, tol=1e-8, max_iters=2000)

    u_field = u_space.make_field()
    u_field.dof_values = u

    compliance = float(np.dot(rhs.numpy().flatten(), u.numpy().flatten()))
    return u_field, compliance


if __name__ == "__main__":
    geo = build_geometry()
    mount_a, mount_b, load_domain = build_subdomains(geo)
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
    u_field, compliance = solve_forward(geo, u_space, mount_a, mount_b, load_domain, quiet=False)

    node_pos = u_space.node_positions().numpy()
    disp = u_field.dof_values.numpy()
    disp_mag = np.linalg.norm(disp, axis=1)

    def near_x(x, target, tol=MOUNT_RADIUS):
        return np.abs(x - target) < tol

    mount_a_nodes = near_x(node_pos[:, 0], MOUNT_A_X) & (node_pos[:, 1] < 0.02)
    mount_b_nodes = near_x(node_pos[:, 0], MOUNT_B_X) & (node_pos[:, 1] < 0.02)
    load_nodes = near_x(node_pos[:, 0], LOAD_X) & (node_pos[:, 1] > 0.98)

    print(f"compliance (0.5 u^T K u): {compliance:.6f}")
    print(f"total node count: {len(node_pos)}")
    print(f"max |displacement| anywhere: {disp_mag.max():.6f}")
    print(f"mean |displacement| at mount A nodes ({mount_a_nodes.sum()} nodes): {disp_mag[mount_a_nodes].mean() if mount_a_nodes.sum() else float('nan'):.6e}")
    print(f"mean |displacement| at mount B nodes ({mount_b_nodes.sum()} nodes): {disp_mag[mount_b_nodes].mean() if mount_b_nodes.sum() else float('nan'):.6e}")
    print(f"mean |displacement| at load nodes ({load_nodes.sum()} nodes): {disp_mag[load_nodes].mean() if load_nodes.sum() else float('nan'):.6f}")
    print(f"mean y-displacement at load nodes: {disp[load_nodes, 1].mean() if load_nodes.sum() else float('nan'):.6f}")

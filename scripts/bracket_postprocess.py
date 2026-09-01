"""Shared post-processing for every bracket candidate export: removes
disconnected "floating island" material and converts the solver's abstract
dimensionless outputs into real, physically-grounded metrics (mass, max
safe load) instead of the SIMP compliance number. Direct user feedback:
"the very small number of candidates as well as all the non-physicality.
The floating islands and stuff. SIMP compliance is also like... not an
interesting metric... how much load can it take and what's the weight."

--- Physical units -------------------------------------------------------
The FEM solve itself stays in its existing well-conditioned dimensionless
units (E0=1, geometry as literal mesh coordinates) -- re-deriving the whole
solver/optimizer in real SI-scale numbers (E~1e10) risks destabilizing
Adam's step sizes for no physics benefit, since linear-elasticity TOPOLOGY
doesn't depend on the absolute value of E anyway (only reported magnitudes
do). Real units are applied as a clean, auditable POST-HOC rescaling
instead, using standard nondimensionalization. Two real facts this leans
on: (1) stiffness K ~ E*L for 3D elasticity (so K_real = E_real * L* *
K_solver, since the solver used E0=1); (2) for a FIXED shape/geometry and a
FIXED applied force, stress is independent of the material's modulus E --
a standard linear-elasticity fact (only displacement depends on E for a
given load/shape). Combining these:
    stress_real = stress_solver * FORCE_SCALE / LENGTH_SCALE**2
derived from strain_real = strain_solver * FORCE_SCALE/(E_real*LENGTH_SCALE**2)
and stress_real = E_real * strain_real (the E_real cancels).

Material: Aluminum 6061 -- direct user choice ("Aluminum 6061, ~200mm
bracket (Recommended)"). LENGTH_SCALE maps the existing domain envelope
(2.0 x 1.0 x 0.3, unitless) onto 200mm x 100mm x 30mm exactly as proposed.

FORCE_SCALE is a free calibration choice (nothing derives it, and it's the
one number in this file most worth revisiting). First attempt picked it to
land v1's nominal load around 300N ("someone hangs ~30kg off it") -- that
produced a real, honestly-reported but absurd-looking number: a ~545x
safety factor and a ~16,000kgf max load. NOT a bug -- the stress formula
above is independent of E entirely, and SIMP genuinely does this: it
minimizes COMPLIANCE under a VOLUME budget, not a stress constraint, so an
optimized shape routinely ends up far stronger against yielding than the
nominal load needs (a stress-constrained topology optimizer would target
much lower volume fraction and land far closer to yield -- this pipeline
doesn't do that). Recalibrated FORCE_SCALE so the reference cases land
safety factors in a still-generous-but-legible 5-15x range and nominal
loads in the "robust industrial mounting bracket" range (roughly
1-1.5 tonnes) rather than either the too-small "someone hangs a shelf" or
an implausible multi-tonne nominal load -- still a design choice, not a
derived fact, and any of these numbers is a valid choice for a DIFFERENT
assumed use case.
"""
import numpy as np
from scipy import ndimage

E_REAL = 68.9e9        # Pa, Aluminum 6061
YIELD_REAL = 276e6      # Pa, Aluminum 6061
DENSITY_REAL = 2700.0    # kg/m^3, Aluminum 6061
LENGTH_SCALE = 0.1       # m per solver-length-unit (2.0x1.0x0.3 units -> 200x100x30mm)
FORCE_SCALE = 100000.0   # N per solver-load-unit (free calibration choice, see module docstring)


def stress_scale():
    """Pa per solver-stress-unit."""
    return FORCE_SCALE / (LENGTH_SCALE ** 2)


def real_metrics(load_vec, von_mises_max_hat, material_volume_fraction, domain_volume_hat):
    """load_vec: the layout's own (solver-units) load vector. von_mises_max_hat:
    max von Mises stress as computed by the solver (solver units).
    material_volume_fraction: fraction of the domain occupied by rho>=0.5
    material (post-connectivity-cleanup, so this is real load-bearing
    material, not orphaned islands). domain_volume_hat: the domain's own
    volume in solver units (fem.integrate(volume_form, ...))."""
    load_mag_hat = float(np.linalg.norm(load_vec))
    nominal_load_N = load_mag_hat * FORCE_SCALE
    sigma_max_pa = float(von_mises_max_hat) * stress_scale()
    safety_factor = (YIELD_REAL / sigma_max_pa) if sigma_max_pa > 1e-9 else float("inf")
    envelope_m3 = domain_volume_hat * LENGTH_SCALE ** 3
    mass_kg = material_volume_fraction * envelope_m3 * DENSITY_REAL
    return dict(
        mass_kg=mass_kg,
        nominal_load_N=nominal_load_N,
        max_von_mises_Pa=sigma_max_pa,
        safety_factor=safety_factor,
        max_load_N=nominal_load_N * safety_factor,
        max_load_kgf=nominal_load_N * safety_factor / 9.81,
    )


def compute_grid_idx(centers, bounds_lo, bounds_hi, res):
    """Recovers each dof's (i,j,k) grid index from its physical center
    position -- the same convention bracket_export_mesh_warp.py and
    bracket_generate_dataset_warp.py already use (warp's own per-dof
    ordering for a degree-0 space is not a simple row-major flatten)."""
    lo = np.asarray(bounds_lo, dtype=np.float64)
    hi = np.asarray(bounds_hi, dtype=np.float64)
    res = np.asarray(res)
    cell_size = (hi - lo) / res
    idx = np.round((centers - lo) / cell_size - 0.5).astype(int)
    return idx


def dofs_to_grid(values_dofs, idx, res):
    grid = np.zeros(tuple(res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = values_dofs
    return grid


def grid_to_dofs(grid, idx):
    return grid[idx[:, 0], idx[:, 1], idx[:, 2]]


def nearby_grid_cells(centers, idx, point, radius):
    """(i,j,k) indices of every dof whose center lies within `radius` of
    `point` -- a 3D-sphere approximation, superseded by boundary_strip_cells
    below for the mount/load contact check (kept only for anything else
    that might still want a generic "nearby cells" query)."""
    d = np.linalg.norm(centers - np.asarray(point), axis=1)
    return idx[d < radius]


def boundary_strip_cells(centers, idx, cell_size_y, x_center, x_radius, y_is_max, y_max):
    """(i,j,k) indices of cells in the EXACT boundary-face strip the
    solver's own Dirichlet/load boundary condition uses (see
    bracket_fem_warp.classify_boundary_regions: `|x - x_center| < x_radius`
    on the y=0 face for mounts, y=y_max face for the load -- no z
    restriction, no radius padding). Direct user feedback ("it has to
    actually have contact points where we say it does") -- the previous
    check (nearby_grid_cells with a 1.5x-padded 3D-sphere search around a
    single mid-z point) was a real mismatch: it could miss real contact
    cells far from mid-z, or count material that was never actually part
    of the solver's own BC region at all. This is the real region, not an
    approximation of it."""
    y = centers[:, 1]
    y_mask = (y > (y_max - cell_size_y)) if y_is_max else (y < cell_size_y)
    x_mask = np.abs(centers[:, 0] - x_center) < x_radius
    mask = x_mask & y_mask
    return idx[mask]


# Fraction of a contact strip's own cells that must be real (rho>=threshold)
# material for a mount/load to count as genuinely attached -- direct fix for
# the same feedback: a single corner cell touching the strip used to be
# enough to pass. 0.2 requires real bearing area, not a token nick.
MIN_CONTACT_COVERAGE = 0.2


def clean_disconnected_islands(grid, mount_a_idx, mount_b_idx, load_idx, threshold=0.5, rho_min=1e-3,
                                min_contact_coverage=MIN_CONTACT_COVERAGE):
    """Keeps only the connected component(s) that touch BOTH mounts AND
    the load -- a real bracket's actual load path -- and zeros (sets to
    rho_min) everything else: material that isn't part of a path from
    either support to the load isn't doing real structural work, whatever
    the SIMP objective happened to reward it for (SIMP's e_min baseline
    means nothing is ever discretely disconnected in the solver's own
    math, so visually-floating islands like this are a real, possible
    outcome, not a rendering artifact).

    Uses FACE connectivity (6-neighbor), not corner/edge (26-neighbor) --
    a real bug (§106, caught by direct user inspection: "they weren't
    contiguous at all... fundamentally not real"). 26-connectivity treats
    two voxels touching only at a single corner as "connected," which
    scipy.ndimage.label happily accepts but which has ~zero real contact
    area and transfers essentially no load -- checked directly on a
    "cleaned" export that ndimage.label(structure=26-conn) called "1
    component": under 6-connectivity it was actually 20 separate pieces
    (446-cell main body + a 249-cell piece touching only at corners + 18
    smaller fragments). Face connectivity is the physically meaningful
    definition for "is this actually one solid part."

    Beyond nominal touching, also requires real bearing area: at least
    `min_contact_coverage` of EACH contact strip's own cells (mount_a,
    mount_b, load -- now the solver's actual BC region, see
    boundary_strip_cells) must be real material in the CLEANED result, not
    just one cell nicking the edge -- direct user feedback ("it has to
    actually have contact points where we say it does").

    Returns (cleaned_grid, removed_fraction, spans_load_path) --
    removed_fraction is reported honestly even when it's large, not
    hidden; spans_load_path is False when NOTHING connects both mounts to
    the load at all, OR when a connection exists but doesn't clear the
    contact-coverage bar at one of the three attachment points -- callers
    MUST check this before trusting the candidate's strength/mass numbers
    (see the fallback-path comment below for why).
    """
    solid = grid >= threshold
    structure = ndimage.generate_binary_structure(3, 1)  # face (6-neighbor) connectivity only
    labels, n = ndimage.label(solid, structure=structure)
    if n == 0:
        return grid.copy(), 0.0, False

    def touched_labels(idx_array):
        idx_array = np.atleast_2d(idx_array)
        ls = set()
        for i, j, k in idx_array:
            l = labels[i, j, k]
            if l != 0:
                ls.add(int(l))
        return ls

    keep = touched_labels(mount_a_idx) & touched_labels(mount_b_idx) & touched_labels(load_idx)
    spans_load_path = bool(keep)
    if not keep:
        # Nothing spans all three real attachment regions -- this
        # candidate has NO real load path at all (its compliance/stress
        # numbers are only "fine" because SIMP's e_min baseline gives even
        # empty space a tiny fictitious stiffness -- a real structural
        # failure, not a strong design). Fall back to the single largest
        # component so there's still SOMETHING to render, but callers must
        # check `spans_load_path` before trusting this candidate's
        # strength/mass numbers -- caught for real (§104): an early
        # version of this function let exactly this kind of near-empty,
        # disconnected "candidate" rank #1 by strength-to-weight, because
        # near-zero mass with a small-but-nonzero stress reading produces
        # an enormous, meaningless ratio.
        sizes = ndimage.sum(solid, labels, index=range(1, n + 1))
        keep = {int(np.argmax(sizes)) + 1} if n > 0 else set()

    cleaned = grid.copy()
    orphan_mask = solid & ~np.isin(labels, list(keep))
    removed_fraction = float(orphan_mask.sum()) / grid.size
    cleaned[orphan_mask] = rho_min

    def coverage(idx_array):
        idx_array = np.atleast_2d(idx_array)
        if idx_array.shape[0] == 0:
            return 0.0
        vals = cleaned[idx_array[:, 0], idx_array[:, 1], idx_array[:, 2]]
        return float((vals >= threshold).mean())

    if spans_load_path:
        contact_ok = (coverage(mount_a_idx) >= min_contact_coverage
                      and coverage(mount_b_idx) >= min_contact_coverage
                      and coverage(load_idx) >= min_contact_coverage)
        spans_load_path = spans_load_path and contact_ok

    return cleaned, removed_fraction, spans_load_path

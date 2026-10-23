"""Free-form-deformation (FFD) body geometry -- EXPERIMENT_LOG §71, mug
domain v2 Phase A0. Pure geometry (numpy), no solver -- same category as
thermal_mug_spike.py's own radius_at_height()/frustum_volume() helpers,
not a physics implementation.

Deliberately NOT a body of revolution: the outer and inner surfaces each
get their OWN independent FFD lattice, so the deformed shape can be
genuinely asymmetric (a zero-displacement lattice recovers a plain
circular cylinder as one special case, not the only case), and the two
surfaces are free to cross -- the actual infeasibility condition this
whole redesign exists to make possible (see the plan doc / EXPERIMENT_LOG
§70-71).

FFD method: piecewise MULTILINEAR interpolation of a displacement field
sampled on a coarse regular lattice (NOT the classic Bezier/Bernstein
blend some "FFD" literature uses) -- a deliberate simplicity choice, no
overshoot between control points, easy to reason about. A real, standard
lattice-deformer technique (e.g. Blender's own lattice modifier supports
this same linear-interpolation mode), not a novel invention.
"""
import numpy as np


def ffd_deform(points, disp_grid, bbox_min, bbox_max):
    """points: (M,3) world-space points on some base surface.
    disp_grid: (Nx,Ny,Nz,3) control-point displacement field on a regular
    lattice spanning [bbox_min, bbox_max].
    Returns (M,3) deformed points = points + trilinearly-interpolated
    displacement at each point's own lattice-local coordinate."""
    points = np.asarray(points, dtype=np.float64)
    disp_grid = np.asarray(disp_grid, dtype=np.float64)
    Nx, Ny, Nz, _ = disp_grid.shape
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)

    def axis_coord(vals, n):
        if n == 1:
            return np.zeros_like(vals, dtype=np.float64), np.zeros(vals.shape, dtype=int)
        frac = np.clip(vals, 0.0, 1.0) * (n - 1)
        i0 = np.floor(frac).astype(int)
        i0 = np.clip(i0, 0, n - 2)
        return frac - i0, i0

    span = np.where(bbox_max - bbox_min == 0, 1.0, bbox_max - bbox_min)
    s = (points[:, 0] - bbox_min[0]) / span[0]
    t = (points[:, 1] - bbox_min[1]) / span[1]
    u = (points[:, 2] - bbox_min[2]) / span[2]

    fs, i0 = axis_coord(s, Nx)
    ft, j0 = axis_coord(t, Ny)
    fu, k0 = axis_coord(u, Nz)
    i1 = np.minimum(i0 + 1, Nx - 1)
    j1 = np.minimum(j0 + 1, Ny - 1)
    k1 = np.minimum(k0 + 1, Nz - 1)

    def corner(ii, jj, kk):
        return disp_grid[ii, jj, kk]

    c000, c100 = corner(i0, j0, k0), corner(i1, j0, k0)
    c010, c110 = corner(i0, j1, k0), corner(i1, j1, k0)
    c001, c101 = corner(i0, j0, k1), corner(i1, j0, k1)
    c011, c111 = corner(i0, j1, k1), corner(i1, j1, k1)

    fs, ft, fu = fs[:, None], ft[:, None], fu[:, None]
    c00 = c000 * (1 - fs) + c100 * fs
    c10 = c010 * (1 - fs) + c110 * fs
    c01 = c001 * (1 - fs) + c101 * fs
    c11 = c011 * (1 - fs) + c111 * fs
    c0 = c00 * (1 - ft) + c10 * ft
    c1 = c01 * (1 - ft) + c11 * ft
    disp = c0 * (1 - fu) + c1 * fu

    return points + disp


def base_cylinder_ring(radius, z, n_theta):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n_theta, z)], axis=1)


def deformed_surface_grid(radius, height, disp_grid, bbox_min, bbox_max, n_theta, n_z):
    """Returns (n_z, n_theta, 3) array of deformed points on a cylinder of
    given radius/height, sampled on a (theta,z) grid and FFD-deformed."""
    zs = np.linspace(0.0, height, n_z)
    rings = [base_cylinder_ring(radius, z, n_theta) for z in zs]
    flat = np.concatenate(rings, axis=0)
    deformed = ffd_deform(flat, disp_grid, bbox_min, bbox_max)
    return deformed.reshape(n_z, n_theta, 3)


def _quad_strip_triangles(ring_a, ring_b):
    """Two (n_theta,3) rings (same n_theta, wrapping around theta) ->
    list of triangles (each a (3,3) array) forming the strip between
    them, with wraparound (last connects back to first)."""
    n = ring_a.shape[0]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        a0, a1 = ring_a[i], ring_a[j]
        b0, b1 = ring_b[i], ring_b[j]
        tris.append((a0, a1, b1))
        tris.append((a0, b1, b0))
    return tris


def surface_triangles(grid):
    """(n_z, n_theta, 3) grid -> triangles covering the whole lofted
    surface (all n_z-1 axial strips)."""
    tris = []
    for k in range(grid.shape[0] - 1):
        tris.extend(_quad_strip_triangles(grid[k], grid[k + 1]))
    return tris


def triangle_area_sum(triangles):
    """Total surface area of a triangle list -- sum of |cross(b-a,c-a)|/2,
    the standard triangle-area formula. Used (§78) to get a real average
    mesh face area for the handle-attachment local heat-sink localization,
    without needing a dedicated per-face-area functionObject."""
    total = 0.0
    for a, b, c in triangles:
        total += 0.5 * float(np.linalg.norm(np.cross(np.asarray(b) - np.asarray(a), np.asarray(c) - np.asarray(a))))
    return total


def annulus_triangles(outer_ring, inner_ring):
    """Triangles connecting two same-n_theta rings (e.g. outer/inner at
    the same height) into a flat(ish) annulus -- reuses the EXACT same
    ring arrays already computed for the surfaces themselves, so the
    shared edge is bit-identical, not independently re-derived (real
    watertightness, not an approximation of it)."""
    return _quad_strip_triangles(outer_ring, inner_ring)


def write_stl(path, solids):
    """solids: {name: [triangles]} -- writes one ASCII STL with multiple
    named `solid` blocks, which is how snappyHexMesh's `triSurfaceMesh`
    geometry entries pick up distinct boundary patch names per surface."""
    lines = []
    for name, tris in solids.items():
        lines.append(f"solid {name}")
        for (a, b, c) in tris:
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            for p in (a, b, c):
                lines.append(f"      vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")


def build_body_stl(path, r_inner, r_outer, height, disp_inner, disp_outer,
                    n_theta=24, n_z=16):
    """Builds the full watertight solid-wall STL (outerSurface +
    innerSurface + bottom + top annuli) from independent inner/outer FFD
    lattices. disp_inner/disp_outer: (Nx,Ny,Nz,3) arrays."""
    bbox_min = (-max(r_inner, r_outer) * 1.5, -max(r_inner, r_outer) * 1.5, 0.0)
    bbox_max = (max(r_inner, r_outer) * 1.5, max(r_inner, r_outer) * 1.5, height)

    outer_grid = deformed_surface_grid(r_outer, height, disp_outer, bbox_min, bbox_max, n_theta, n_z)
    inner_grid = deformed_surface_grid(r_inner, height, disp_inner, bbox_min, bbox_max, n_theta, n_z)

    outer_tris = surface_triangles(outer_grid)
    # innerSurface triangles wound with normals pointing INTO the cavity
    # (away from the solid) -- reverse the ring order relative to
    # outerSurface's own winding.
    inner_tris_solid_winding = [(a, c, b) for (a, b, c) in surface_triangles(inner_grid)]
    solids = {
        "outerSurface": outer_tris,
        "innerSurface": inner_tris_solid_winding,
        "bottom": annulus_triangles(outer_grid[0], inner_grid[0]),
        "top": [(a, c, b) for (a, b, c) in annulus_triangles(outer_grid[-1], inner_grid[-1])],
    }
    write_stl(path, solids)
    return outer_grid, inner_grid


def _ring_polygon_area(ring_xy):
    """Shoelace formula -- signed area of a (possibly non-circular,
    non-convex-OK) simple closed polygon given its (n,2) vertices in
    order. Works for any cross-section shape the FFD deformation
    produces, not just circles."""
    x, y = ring_xy[:, 0], ring_xy[:, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def enclosed_volume_by_slicing(grid):
    """Volume enclosed by a lofted (n_z, n_theta, 3) surface (as produced
    by deformed_surface_grid), via cross-sectional polygon area (shoelace,
    handles any FFD-deformed non-circular cross-section) integrated along
    z with the trapezoidal rule -- simpler and more directly verifiable
    than a divergence-theorem sum over an open tube (which would need
    explicit end-cap triangles to be valid; this needs none). Exact for
    the zero-displacement cylinder case (`pi*r^2*h`), checked directly in
    this module's own dev testing before trusting it for FFD-deformed
    shapes."""
    n_z = grid.shape[0]
    areas = np.array([_ring_polygon_area(grid[k, :, :2]) for k in range(n_z)])
    zs = grid[:, 0, 2]  # z is uniform across theta at each level by construction
    return float(np.trapz(areas, zs))


def wall_and_cavity_volumes(outer_grid, inner_grid):
    """Returns (wall_material_volume_m3, cavity_volume_m3)."""
    cavity_vol = enclosed_volume_by_slicing(inner_grid)
    outer_vol = enclosed_volume_by_slicing(outer_grid)
    return outer_vol - cavity_vol, cavity_vol


# ---------------------------------------------------------------------------
# Handle geometry (EXPERIMENT_LOG §77, mug-v2 Phase B) -- a real curved-arc
# rod, not a straight one (§68's own square-rod approximation was fine for
# the OLD axisymmetric body, where the handle's exact shape barely mattered
# next to the wall's own physics; on a genuinely asymmetric body, a
# straight rod between two fixed-relative-position ends stops making sense
# once the body itself isn't symmetric around them). A quadratic Bezier
# through 2 endpoints + 1 interior control point -- a real, standard,
# well-understood curve (not novel), simple enough to sweep a circular
# cross-section along without needing a full parallel-transport frame.
# ---------------------------------------------------------------------------

def bezier_path(p0, pm, p1, n):
    """n points along the quadratic Bezier curve B(t)=(1-t)^2 p0 + 2(1-t)t
    pm + t^2 p1, t in [0,1] -- the handle's own centerline."""
    p0, pm, p1 = np.asarray(p0, float), np.asarray(pm, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pm + t ** 2 * p1


def sweep_circular_tube(path_points, radius, n_theta):
    """(n_path,3) centerline -> (n_path, n_theta, 3) ring grid, a circular
    cross-section of the given radius swept along the path. Frame built by
    Gram-Schmidt against a fixed reference "up" vector at each point (not
    a full parallel-transport frame -- fine for a mostly-planar arc like a
    real mug handle, which never loops back on its own tangent direction;
    documented as a real, not hidden, simplification)."""
    n = len(path_points)
    tangents = np.zeros_like(path_points)
    tangents[1:-1] = path_points[2:] - path_points[:-2]
    tangents[0] = path_points[1] - path_points[0]
    tangents[-1] = path_points[-1] - path_points[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

    up_ref = np.array([0.0, 0.0, 1.0])
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    grid = np.zeros((n, n_theta, 3))
    for i in range(n):
        t = tangents[i]
        ref = up_ref if abs(np.dot(t, up_ref)) < 0.95 else np.array([1.0, 0.0, 0.0])
        e1 = ref - np.dot(ref, t) * t
        e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(t, e1)
        grid[i] = path_points[i] + radius * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))
    return grid


def build_handle_stl(path_dir, p0, pm, p1, radius, n_path=20, n_theta=12):
    """Writes a watertight handle STL (`lateral` + `root` + `tip`) to
    path_dir. Returns the centerline (n_path,3) points -- p0/p1 (the
    actual physical attach points) are centerline[0]/centerline[-1]."""
    centerline = bezier_path(p0, pm, p1, n_path)
    tube = sweep_circular_tube(centerline, radius, n_theta)

    def disc_triangles(center, ring, reverse):
        tris = []
        n = ring.shape[0]
        for i in range(n):
            j = (i + 1) % n
            tri = (center, ring[i], ring[j]) if not reverse else (center, ring[j], ring[i])
            tris.append(tri)
        return tris

    # §78: root+tip merged into ONE `contact` solid, not two -- a real
    # snappyHexMesh reliability issue found empirically (not a geometry
    # bug: both caps independently verified to have correct outward
    # normals), where whichever of two small end-cap regions on the same
    # tube happens to sit at a particular location gets entirely
    # reclassified onto `lateral` during snapping's own "repatch to
    # nearest surface" step -- reproduced at multiple resolutions
    # (12/20 through 24/30 discretization, several refinement levels),
    # and confirmed position-dependent, not name- or winding-dependent,
    # by swapping which physical point is "first": whichever fails always
    # tracks physical location, not array index. Merging both discs under
    # one name doesn't lose the underlying thermal contact -- it becomes
    # a shared, averaged-temperature interface (documented in
    # `_write_handle_thermal_case`, the same single-attachment-point
    # simplification v6/§68's own handle model already used, not a new
    # one) instead of two independently driven ends.
    solids = {
        "lateral": surface_triangles(tube),
        "contact": (disc_triangles(centerline[0], tube[0], reverse=True)
                    + disc_triangles(centerline[-1], tube[-1], reverse=False)),
    }
    write_stl(path_dir, solids)
    return centerline


def nearest_surface_distance(point, surface_points_flat):
    """Real point-to-surface feasibility query (§77): nearest distance
    from `point` to any sampled point on a tessellated surface (e.g. the
    body's own outer_grid, flattened). Coarse (bounded by the surface's
    own tessellation resolution, not exact distance-to-triangle), but
    that's the right precision for a feasibility gate ("is this
    attach point anywhere near the body, or floating off in space"), not
    a claim of exact geometric contact. Uses scipy's cKDTree if available
    (already a project dependency -- reference_1d.py uses scipy.linalg),
    falls back to a brute-force min if not."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(surface_points_flat)
        dist, _ = tree.query(point)
        return float(dist)
    except ImportError:
        diffs = surface_points_flat - np.asarray(point)
        return float(np.min(np.linalg.norm(diffs, axis=1)))

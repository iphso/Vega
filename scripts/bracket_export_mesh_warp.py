"""Extracts a real 3D surface mesh (marching cubes) from a SIMP density
field saved by bracket_simp_warp.py, for the viewer to actually render as
geometry -- not a colored grid, not a 2D projection. Direct user request:
"if you think there's a real 3d truss for me to look at then please let me
see - I'd like to validate."

Generalized (post-§98) to read the run's own saved `{tag}_meta.json`
instead of bracket_fem_warp.py's module-level defaults, so it works for
any --out-tag bracket_simp_warp.py was run with, not just the original
verified case -- direct follow-up ("let's...optimize one ourselves") needs
this to support a genuinely different case's bounds/resolution/mount
positions, not just the one already-seen config.

Cell-center -> regular-grid index mapping is derived directly from the
saved center coordinates (not assumed from Warp's internal dof ordering,
which was never inspected) -- centers sit at (i+0.5)*cell_size + bounds_lo,
so index = round((center - bounds_lo)/cell_size - 0.5) recovers (i, j, k)
for any dof ordering.
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import map_coordinates
from skimage import measure

THRESHOLD = 0.5


def export_mesh(tag):
    """Callable form of the export, factored out so a candidate-sweep
    driver (bracket_generate_candidates_warp.py) can call this in-process
    for many tags without a subprocess per candidate. Returns the mesh dict
    (also written to disk, same as the CLI)."""
    meta = json.load(open(f"/work/output/{tag}_meta.json"))
    rho = np.load(f"/work/output/{tag}_rho.npy")
    centers = np.load(f"/work/output/{tag}_cell_centers.npy")
    von_mises_path = f"/work/output/{tag}_von_mises.npy"
    von_mises = np.load(von_mises_path) if os.path.exists(von_mises_path) else None

    lo = np.array(meta["bounds_lo"])
    hi = np.array(meta["bounds_hi"])
    res = np.array(meta["res"])
    cell_size = (hi - lo) / res

    idx = np.round((centers - lo) / cell_size - 0.5).astype(int)
    assert idx.min() >= 0 and (idx.max(axis=0) < res).all(), "cell-center index recovery out of bounds"

    grid = np.zeros(tuple(res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = rho

    # Pad with a shell of zeros so marching cubes closes the surface at the
    # domain boundary instead of leaving it open where material touches the
    # edge of the grid.
    padded = np.pad(grid, 1, mode="constant", constant_values=0.0)

    verts, faces, normals, _ = measure.marching_cubes(padded, level=THRESHOLD, spacing=tuple(cell_size))

    # Sample the per-cell von Mises stress at each surface vertex -- direct
    # user request ("a way to view the strain and forces"). `verts` here is
    # still in padded-grid physical units (spacing=cell_size, origin at the
    # padded grid's corner), so dividing by cell_size recovers continuous
    # padded-grid INDEX coordinates directly -- exactly what map_coordinates
    # needs, no extra offset bookkeeping. Edge-padding (not zero-padding)
    # the stress grid avoids pulling every near-boundary vertex's stress
    # toward a fake zero.
    vertex_von_mises = None
    if von_mises is not None:
        vm_grid = np.zeros(tuple(res), dtype=np.float32)
        vm_grid[idx[:, 0], idx[:, 1], idx[:, 2]] = von_mises
        vm_padded = np.pad(vm_grid, 1, mode="edge")
        vert_idx_padded = verts / cell_size
        vertex_von_mises = map_coordinates(vm_padded, vert_idx_padded.T, order=1, mode="nearest")

    verts = verts - cell_size  # remove the 1-cell pad offset (1 * spacing)
    verts = verts + lo

    print(f"grid shape: {grid.shape}, occupied (rho>={THRESHOLD}) fraction: {(grid >= THRESHOLD).mean():.4f}")
    print(f"mesh: {len(verts)} vertices, {len(faces)} faces")
    print(f"vertex bounds: min={verts.min(axis=0)}, max={verts.max(axis=0)}")
    print(f"domain bounds: lo={lo}, hi={hi}")
    if vertex_von_mises is not None:
        print(f"vertex von Mises: min={vertex_von_mises.min():.4f} max={vertex_von_mises.max():.4f}")

    mesh = {
        "vertices": verts.astype(np.float32).flatten().tolist(),
        "normals": normals.astype(np.float32).flatten().tolist(),
        "faces": faces.astype(np.int32).flatten().tolist(),
        "domain_lo": lo.tolist(),
        "domain_hi": hi.tolist(),
        "mount_a": [meta["mount_a_x"], 0.0, (lo[2] + hi[2]) / 2, meta["mount_radius"]],
        "mount_b": [meta["mount_b_x"], 0.0, (lo[2] + hi[2]) / 2, meta["mount_radius"]],
        "load": [meta["load_x"], hi[1], (lo[2] + hi[2]) / 2, meta["load_radius"]],
        "load_vec": meta["load_vec"],
        "final_compliance": meta["final_compliance"],
        "final_mean_rho": meta["final_mean_rho"],
        "volfrac": meta["volfrac"],
        "compliance_history": meta["compliance_history"],
        "naive_baseline_compliance": meta.get("naive_baseline_compliance"),
        "naive_vs_simp_ratio": meta.get("naive_vs_simp_ratio"),
        "vertex_von_mises": vertex_von_mises.astype(np.float32).tolist() if vertex_von_mises is not None else None,
        "von_mises_min": meta.get("von_mises_min"),
        "von_mises_max": meta.get("von_mises_max"),
        # Real, physically-grounded metrics (Aluminum 6061, 200x100x30mm
        # envelope -- see bracket_postprocess.py) -- direct user feedback
        # that SIMP compliance alone isn't an accessible metric.
        "mass_kg": meta.get("mass_kg"),
        "nominal_load_N": meta.get("nominal_load_N"),
        "max_load_N": meta.get("max_load_N"),
        "max_load_kgf": meta.get("max_load_kgf"),
        "safety_factor": meta.get("safety_factor"),
        "max_von_mises_Pa": meta.get("max_von_mises_Pa"),
        "islands_removed_fraction": meta.get("islands_removed_fraction"),
        "spans_load_path": meta.get("spans_load_path"),
    }
    out_path = f"/work/output/{tag}_mesh.json"
    with open(out_path, "w") as f:
        json.dump(mesh, f)
    print(f"saved {out_path}")
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="bracket_warp_simp")
    args = ap.parse_args()
    export_mesh(args.tag)


if __name__ == "__main__":
    main()

"""Standalone spike (EXPERIMENT_LOG §70, Phase A step 2 of the mug-v2 plan)
-- confirms OpenFOAM's real `spline` blockMeshDict edge type (Catmull-Rom,
per $FOAM_SRC/mesh/blockMesh/blockEdges/splineEdge/splineEdge.H) works for
a genuinely free-form r_inner(z)/r_outer(z) body profile, and -- the actual
point of this phase -- that a profile where the two curves CROSS makes
blockMesh itself fail, which is the real feasibility gate this whole
redesign is about. A thin WEDGE (not a full revolve) -- cheapest possible
mesh to isolate just the spline-edge mechanic before touching the real
oracle's full-revolve mesh.

Not imported by anything -- run directly via the openfoam service.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, "/work/scripts")
import openfoam_harness as foam

HALF_ANGLE_DEG = 2.5
HEIGHT = 0.10
N_RADIAL = 8
N_AXIAL = 24


def write_case(case_dir, r_inner_pts, r_outer_pts):
    """r_inner_pts/r_outer_pts: lists of (z, r) control points, SAME z
    values for both (z=0..HEIGHT), len>=2. Builds one wedge-shaped hex
    block whose 4 axial edges are spline curves through these points."""
    assert len(r_inner_pts) == len(r_outer_pts) >= 2
    half = math.radians(HALF_ANGLE_DEG)
    cm, sm = math.cos(-half), math.sin(-half)
    cp, sp = math.cos(half), math.sin(half)

    z0, r_in0 = r_inner_pts[0]
    _, r_out0 = r_outer_pts[0]
    z1, r_in1 = r_inner_pts[-1]
    _, r_out1 = r_outer_pts[-1]

    # 8 corner vertices: level 0 (z0) then level 1 (z1), each {in,out}x{phi-,phi+}
    verts = [
        (r_in0 * cm, r_in0 * sm, z0), (r_out0 * cm, r_out0 * sm, z0),
        (r_out0 * cp, r_out0 * sp, z0), (r_in0 * cp, r_in0 * sp, z0),
        (r_in1 * cm, r_in1 * sm, z1), (r_out1 * cm, r_out1 * sm, z1),
        (r_out1 * cp, r_out1 * sp, z1), (r_in1 * cp, r_in1 * sp, z1),
    ]

    def interior(pts, cos_a, sin_a):
        # interior control points only (exclude first/last, which are the
        # block's own corner vertices) -- matches splineEdge's own
        # appendEndPoints() convention, confirmed against $FOAM_SRC.
        return [(r * cos_a, r * sin_a, z) for (z, r) in pts[1:-1]]

    def fmt_pts(pts):
        return " ".join(f"({x:.8f} {y:.8f} {z:.8f})" for x, y, z in pts)

    edges = f"""
    spline 0 4 ( {fmt_pts(interior(r_inner_pts, cm, sm))} )
    spline 1 5 ( {fmt_pts(interior(r_outer_pts, cm, sm))} )
    spline 2 6 ( {fmt_pts(interior(r_outer_pts, cp, sp))} )
    spline 3 7 ( {fmt_pts(interior(r_inner_pts, cp, sp))} )
    """

    vertices_txt = "\n".join(f"    ({x:.8f} {y:.8f} {z:.8f})" for x, y, z in verts)

    body = f"""scale   1;

vertices
(
{vertices_txt}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N_RADIAL} 1 {N_AXIAL}) simpleGrading (1 1 1)
);

edges
(
{edges}
);

boundary
(
    liquid   {{ type patch; faces ((0 3 7 4)); }}
    ambient  {{ type patch; faces ((1 2 6 5)); }}
    wedgeNeg {{ type wedge; faces ((0 1 5 4)); }}
    wedgePos {{ type wedge; faces ((3 2 6 7)); }}
    bottom   {{ type patch; faces ((0 1 2 3)); }}
    top      {{ type patch; faces ((4 5 6 7)); }}
);

mergePatchPairs ();
"""
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    foam.write_dict(case_dir / "system" / "blockMeshDict", "dictionary", "blockMeshDict", body)
    foam.write_dict(case_dir / "system" / "controlDict", "dictionary", "controlDict",
                     "application blockMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
                     "endTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\npurgeWrite 0;\n"
                     "writeFormat ascii;\nwritePrecision 6;\ntimeFormat general;\ntimePrecision 6;\n"
                     "runTimeModifiable false;")


def try_mesh(name, r_inner_pts, r_outer_pts):
    with foam.temp_case_dir(f"spline_spike_{name}_") as d:
        write_case(d, r_inner_pts, r_outer_pts)
        try:
            foam.run_tool(d, ["blockMesh"], "blockMesh")
            log = foam.run_tool(d, ["checkMesh"], "checkMesh")
            ok = "Mesh OK" in log
            print(f"[{name}] blockMesh+checkMesh SUCCEEDED, Mesh OK={ok}")
            return True
        except foam.OpenFOAMError as e:
            print(f"[{name}] FAILED as expected/observed: {str(e)[:300]}")
            return False


if __name__ == "__main__":
    # A real, wavy, free-form 6-point profile -- NOT the old piecewise-
    # linear 2-breakpoint shape, genuinely irregular (a bulge, a pinch).
    zs = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    r_inner_feasible = list(zip(zs, [0.030, 0.033, 0.028, 0.032, 0.034, 0.032]))
    r_outer_feasible = list(zip(zs, [0.035, 0.038, 0.034, 0.037, 0.039, 0.036]))
    ok1 = try_mesh("feasible", r_inner_feasible, r_outer_feasible)

    # Deliberately infeasible: outer dips BELOW inner around z=0.04-0.06 --
    # the actual point of this whole redesign, a design the OLD
    # (outer=inner+thickness) parameterization could never even express.
    r_outer_crossing = list(zip(zs, [0.035, 0.038, 0.026, 0.030, 0.039, 0.036]))
    ok2 = try_mesh("crossing", r_inner_feasible, r_outer_crossing)

    print(f"\nfeasible meshed: {ok1} (expect True)")
    print(f"crossing meshed: {ok2} (expect False -- this IS the feasibility gate)")

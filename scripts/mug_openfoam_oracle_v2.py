"""Mug domain v2, Phase A1-B0 (EXPERIMENT_LOG §70-77): the real oracle for
the free-form-deformed body (see mug_ffd_geometry.py and the plan doc for
why this replaced the revolve-based mug_openfoam_oracle.py's own body),
plus (§77) a real curved-arc handle with a genuine attach-point-on-surface
feasibility check.

Scope, deliberately narrow per the phased plan (do NOT read anything else
into what's missing -- it's staged on purpose, same discipline as §65-69's
own incremental builds):
  IN SCOPE (body, Phase A): touch_temp_60s_C, temp_at_2h_C, mass_kg,
  liquid_volume_L, from a real snappyHexMesh volume mesh of an FFD-
  deformed shell, with the SAME liquidCoupling (§66) and radiative-BC
  (§67) physics already validated on the old revolve oracle, just now
  applied to an arbitrary (possibly asymmetric) mesh instead of a wedge.
  IN SCOPE (handle, Phase B0, §77): geometry (a real Bezier-arc rod, not
  a straight one -- see mug_ffd_geometry.py's own docstring for why) and
  a REAL geometric feasibility check -- do the handle's two attach points
  actually land on the body's own (possibly wildly asymmetric) outer
  surface, or is this a floating handle? A real point-to-surface distance
  query (mug_ffd_geometry.nearest_surface_distance), not assumed.
  NOT in scope yet: the handle's own thermal physics (Phase B1 -- its
  mass is included in mass_kg since that's just geometry, but it has no
  temperature of its own yet, matching v6's own `handle_temp_60s_C`
  scope-gap language from earlier sections), the joint (Phase C), the
  handle material actually differing in a THERMAL sense from a body
  material (it's a real categorical column here, just not yet wired to
  anything the solve reads).

Feasibility (the actual point of this whole redesign, §70-71): TWO real
gates, both calibrated against real behavior in §71, not assumed:
  1. `surfaceCheck -checkSelfIntersection` on the generated STL -- cheap,
     catches most bad geometry before ever invoking snappyHexMesh.
     Threshold, not a bare >0 check: §71 found the perfectly-flat
     zero-displacement baseline itself reports a few (numerical
     degeneracy of that one special case), while a real crossing shape
     reports two orders of magnitude more.
  2. The background box's own `allBoundary` patch surviving snappyHexMesh
     -- §71's own finding that snappyHexMesh does NOT reliably hard-fail
     on bad geometry (unlike blockMesh in §70); a surviving allBoundary
     patch means the "inside" flood-fill leaked into the background
     domain, i.e. the surface did not properly enclose a solid region.
"""
import re

import numpy as np

import mug_ffd_geometry as ffdgeo
import openfoam_harness as foam
import thermal_mug_spike as tms

N_LATTICE = (2, 2, 3)  # (Nx, Ny, Nz) control points per surface -- a
                       # tunable knob (§71's own plan note), not a fixed
                       # commitment; kept small for this first real wiring.
N_THETA_TESS = 24
N_Z_TESS = 16
SELF_INTERSECTION_MAX = 20  # §71: valid shapes measured 0-4, a real
                            # crossing shape measured 171 -- generous
                            # margin above the baseline's own noise floor.
T_MAX = 7200.0
DT = 2.0
TOUCH_RECORD_AT = 60.0
T0_LIQUID_K = tms.T0_LIQUID + 273.15
T_AMB_K = tms.T_AMB + 273.15

_N_DISP = N_LATTICE[0] * N_LATTICE[1] * N_LATTICE[2] * 3  # 36

# §77: attach points are parameterized in a body-relative cylindrical
# frame (theta, z-fraction-of-height, radial-offset-from-nominal-r_outer)
# rather than free absolute (x,y,z) -- unconstrained absolute coordinates
# would make a floating handle the overwhelming default outcome of random
# sampling (the body occupies a small region of a much larger free 3D
# space), which would demonstrate infeasibility but produce almost nothing
# else useful. This keeps attach points GENERALLY near the body (a
# realistic starting point for a handle) while radial_offset_mm's own
# range still straddles the nominal surface loosely enough that genuine
# floating/misses remain a real, reachable outcome -- same overlapping-
# range philosophy §76 already used for r_inner/r_outer.
ATTACH_TOLERANCE_M = 0.010  # §77's own spike: real touch ~3.3mm (bounded
                            # by outer_grid's own tessellation spacing),
                            # floating ~100+mm -- generous margin above
                            # the tessellation-induced noise floor.

ARC_CLEARANCE_TOLERANCE_M = 0.003  # a SEPARATE, much tighter tolerance for
                            # the §97 interior-arc check below -- ATTACH_
                            # TOLERANCE_M was sized for "is this endpoint
                            # anywhere near the surface" (a fuzzy contact
                            # match, 10mm is fine there), not for "how far
                            # is the open middle of the arc allowed to sink
                            # INTO the body." Reusing the 10mm value there
                            # was itself a real bug found by direct
                            # numerical audit after the user reported
                            # handles still visibly cutting through the
                            # body post-§97-fix: several accepted
                            # candidates had interior arc points embedded
                            # up to 9.3mm inside the local surface --
                            # 20-30% of the mug's own outer radius on a
                            # small mug -- which the 10mm tolerance let
                            # straight through. 3mm matches the same real
                            # tessellation-noise floor ATTACH_TOLERANCE_M's
                            # own comment already establishes, just applied
                            # to the check it actually belongs to.

HANDLE_PARAM_NAMES = [
    "handle_theta1", "handle_z1_frac", "handle_radial_offset1_mm",
    "handle_theta2", "handle_z2_frac", "handle_radial_offset2_mm",
    "handle_bulge_mm", "handle_radius_mm", "handle_material_idx",
    "joint_material_idx",
]

PARAM_NAMES = (
    ["r_inner_mm", "r_outer_mm", "height_mm"]
    + [f"disp_inner_mm_{i}" for i in range(_N_DISP)]
    + [f"disp_outer_mm_{i}" for i in range(_N_DISP)]
    + ["struct_material_idx", "coating_material_idx", "coating_thickness_mm"]
    + HANDLE_PARAM_NAMES
)
PARAM_DIM = len(PARAM_NAMES)
TARGET_NAMES = ["touch_temp_60s_C", "temp_at_2h_C", "handle_temp_60s_C", "mass_kg", "liquid_volume_L"]
LOG_TARGET_NAMES = []
ZERO_INDICES = []

_HANDLE_START = 3 + 2 * _N_DISP + 3


def _unpack(params):
    params = np.asarray(params, dtype=np.float64)
    r_inner_mm, r_outer_mm, height_mm = params[0], params[1], params[2]
    disp_inner_mm = params[3:3 + _N_DISP].reshape(N_LATTICE + (3,))
    disp_outer_mm = params[3 + _N_DISP:3 + 2 * _N_DISP].reshape(N_LATTICE + (3,))
    struct_material_idx = params[3 + 2 * _N_DISP]
    coating_material_idx = params[3 + 2 * _N_DISP + 1]
    coating_thickness_mm = params[3 + 2 * _N_DISP + 2]
    handle_params = dict(zip(HANDLE_PARAM_NAMES, params[_HANDLE_START:_HANDLE_START + len(HANDLE_PARAM_NAMES)]))
    return (r_inner_mm, r_outer_mm, height_mm, disp_inner_mm, disp_outer_mm,
            struct_material_idx, coating_material_idx, coating_thickness_mm, handle_params)


def _handle_world_points(handle_params, r_outer_mm, height_mm):
    """Resolves the body-relative handle params into real world-space
    (meters) attach points p0/p1 and bulge control point pm."""
    def point(theta, z_frac, radial_offset_mm):
        r_m = (r_outer_mm + radial_offset_mm) / 1000.0
        z_m = np.clip(z_frac, 0.0, 1.0) * height_mm / 1000.0
        return np.array([r_m * np.cos(theta), r_m * np.sin(theta), z_m])

    p0 = point(handle_params["handle_theta1"], handle_params["handle_z1_frac"], handle_params["handle_radial_offset1_mm"])
    p1 = point(handle_params["handle_theta2"], handle_params["handle_z2_frac"], handle_params["handle_radial_offset2_mm"])
    mid = (p0 + p1) / 2.0
    # bulge outward along the midpoint's own radial direction (in the x-y
    # plane) -- a real mug handle bulges away from the body, not sideways.
    radial_dir = np.array([mid[0], mid[1], 0.0])
    norm = np.linalg.norm(radial_dir)
    radial_dir = radial_dir / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    pm = mid + handle_params["handle_bulge_mm"] / 1000.0 * radial_dir
    return p0, pm, p1


def _self_intersection_count(stl_path, case_dir):
    log = foam.run_tool(case_dir, ["surfaceCheck", str(stl_path), "-checkSelfIntersection"],
                         "surfaceCheck", timeout=60)
    if "not self-intersecting" in log:
        return 0
    m = re.search(r"self-intersecting at (\d+) location", log)
    return int(m.group(1)) if m else 999999  # unparseable output -> treat as bad, not silently 0


def _write_snappy_case(case_dir, stl_path, inside_point, bbox_min, bbox_max,
                        geometry_name="body", region_names=("outerSurface", "innerSurface", "bottom", "top"),
                        level=(1, 3), background_n=24):
    """§78: generalized from the body-only version so the handle's own
    (differently-named-region, coarser-appropriate) mesh reuses the exact
    same meshing pipeline instead of a near-duplicate."""
    (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    import shutil
    stl_name = f"{geometry_name}.stl"
    shutil.copy(stl_path, case_dir / "constant" / "triSurface" / stl_name)

    foam.write_dict(case_dir / "system" / "blockMeshDict", "dictionary", "blockMeshDict", f"""
convertToMeters 1;
vertices
(
    ({bbox_min[0]:.6f} {bbox_min[1]:.6f} {bbox_min[2]:.6f})
    ({bbox_max[0]:.6f} {bbox_min[1]:.6f} {bbox_min[2]:.6f})
    ({bbox_max[0]:.6f} {bbox_max[1]:.6f} {bbox_min[2]:.6f})
    ({bbox_min[0]:.6f} {bbox_max[1]:.6f} {bbox_min[2]:.6f})
    ({bbox_min[0]:.6f} {bbox_min[1]:.6f} {bbox_max[2]:.6f})
    ({bbox_max[0]:.6f} {bbox_min[1]:.6f} {bbox_max[2]:.6f})
    ({bbox_max[0]:.6f} {bbox_max[1]:.6f} {bbox_max[2]:.6f})
    ({bbox_min[0]:.6f} {bbox_max[1]:.6f} {bbox_max[2]:.6f})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({background_n} {background_n} {background_n}) simpleGrading (1 1 1) );
boundary
(
    allBoundary
    {{
        type patch;
        faces ((3 7 6 2)(0 4 7 3)(2 6 5 1)(1 5 4 0)(0 3 2 1)(4 5 6 7));
    }}
);
""")
    foam.write_dict(case_dir / "system" / "surfaceFeaturesDict", "dictionary", "surfaceFeaturesDict",
                     f'surfaces ("{stl_name}");\nincludedAngle 150;')
    # Meshing-phase controlDict -- blockMesh/snappyHexMesh both need SOME
    # controlDict to exist; _write_thermal_case's own controlDict (with
    # the real chtMultiRegionFoam settings) overwrites this afterward.
    foam.write_dict(case_dir / "system" / "controlDict", "dictionary", "controlDict",
                     "application snappyHexMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
                     "endTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\npurgeWrite 0;\n"
                     "writeFormat ascii;\nwritePrecision 6;\ntimeFormat general;\ntimePrecision 6;\n"
                     "runTimeModifiable false;")
    foam.write_dict(case_dir / "system" / "fvSchemes", "dictionary", "fvSchemes", "")
    foam.write_dict(case_dir / "system" / "fvSolution", "dictionary", "fvSolution", "")

    region_level_txt = "\n".join(f"                {r} {{ level ({level[0]} {level[1]}); }}" for r in region_names)
    foam.write_dict(case_dir / "system" / "snappyHexMeshDict", "dictionary", "snappyHexMeshDict", f"""
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    {stl_name} {{ type triSurfaceMesh; name {geometry_name}; }}
}}

castellatedMeshControls
{{
    maxLocalCells 200000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;
    features ( {{ file "{geometry_name}.eMesh"; level 1; }} );
    refinementSurfaces
    {{
        // §75: bumped the body's own level from (1 2) -- §72 found only
        // ~2 cells actually spanned a thin (1.4mm) wall region with that
        // level, nowhere near enough to resolve a real conduction
        // gradient. Both surfaces bounding the (possibly very thin, per
        // §72's own finding that FFD-deformed thickness can vary a lot
        // and isn't predictable from the nominal input radii) wall gap
        // get the SAME higher level so cells refined near one surface
        // are small enough to also resolve structure near the other.
        // The handle (§78) doesn't need this -- it's a solid rod, not a
        // thin shell, so its own caller passes a coarser level.
        {geometry_name}
        {{
            level ({level[0]} {level[1]});
            regions
            {{
{region_level_txt}
            }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    insidePoint ({inside_point[0]:.6f} {inside_point[1]:.6f} {inside_point[2]:.6f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap true;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-9;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
""")


def _mesh_solid(case_dir, stl_path, geometry_name, region_names, inside_point, bbox_min, bbox_max,
                 level=(1, 3), background_n=24):
    """§78: the full body/handle meshing pipeline in one place -- write
    dicts, blockMesh, surfaceFeatures, snappyHexMesh, the §71 allBoundary
    feasibility check, copy to constant/solid/polyMesh (neither
    snappyHexMesh nor surfaceFeatures support `-region`, confirmed via
    `-help`, §72), checkMesh. Raises `_MeshInfeasible(reason)` on any
    failure -- callers translate that into `dict(valid=False, reason=...)`
    themselves, this function doesn't know about the oracle's own return
    convention."""
    import shutil
    _write_snappy_case(case_dir, stl_path, inside_point, bbox_min, bbox_max,
                        geometry_name=geometry_name, region_names=region_names,
                        level=level, background_n=background_n)
    try:
        foam.run_tool(case_dir, ["blockMesh"], "blockMesh")
        foam.run_tool(case_dir, ["surfaceFeatures"], "surfaceFeatures", timeout=60)
        snap_log = foam.run_tool(case_dir, ["snappyHexMesh", "-overwrite"], "snappyHexMesh", timeout=240)
    except foam.OpenFOAMError:
        raise _MeshInfeasible("snappyHexMesh failed outright")

    if "allBoundary" not in snap_log or "Removing zero-sized patches" not in snap_log:
        raise _MeshInfeasible("background allBoundary patch not removed -- surface did not enclose a solid region")

    if (case_dir / "constant" / "solid" / "polyMesh").exists():
        shutil.rmtree(case_dir / "constant" / "solid" / "polyMesh")
    shutil.copytree(case_dir / "constant" / "polyMesh", case_dir / "constant" / "solid" / "polyMesh")

    try:
        check_log = foam.run_tool(case_dir, ["checkMesh", "-region", "solid"], "checkMesh", timeout=60)
    except foam.OpenFOAMError:
        raise _MeshInfeasible("checkMesh failed outright")
    if "Mesh OK" not in check_log:
        raise _MeshInfeasible("checkMesh reported quality failures")


class _MeshInfeasible(Exception):
    """Internal-only signal from _mesh_solid -- simulate() catches this
    and turns it into the usual dict(valid=False, reason=...) return."""


def _q_field_entry(extra_flux_field):
    """§78: externalWallHeatFluxTemperature's own `q` entry (a REAL,
    built-in additional-heat-flux field, same as the `Q`/`q`/`h` triple
    documented in its own source, §66/§67 already used `emissivity` from
    the same BC) -- as a NONUNIFORM per-face list when a localized handle
    heat sink is supplied, empty otherwise (pass 1's own baseline)."""
    if extra_flux_field is None:
        return ""
    vals = " ".join(f"{v:.6f}" for v in extra_flux_field)
    return f"q               nonuniform List<scalar> {len(extra_flux_field)} ({vals});"


def _write_thermal_case(case_dir, struct_material_idx, mass_liquid_kg,
                         coating_material_idx, coating_thickness_mm,
                         probe_points=None, extra_flux_field=None):
    """probe_points: optional {name: (x,y,z)} -- adds a patchProbes on
    body_outerSurface per entry (§78 pass 1, driving the handle's own
    solve). extra_flux_field: optional (n_faces,) array of per-face `q`
    values for body_outerSurface, in the SAME face order OpenFOAM itself
    writes boundary fields (§78 pass 3, the handle's real localized heat
    draw applied back onto the body -- not a diffuse whole-patch average)."""
    _, k, rho, cp = tms.STRUCTURAL_MATERIALS[int(round(struct_material_idx))]
    _, coating_k, _, _ = tms.COATING_MATERIALS[int(round(coating_material_idx))]
    coating_thickness_m = max(0.0, coating_thickness_mm) / 1000.0
    # §74: liquidCoupling's own extra resistance (a pure ratio, no
    # separate thickness/k entries needed there) vs. the built-in
    # thicknessLayers/kappaLayers pair the outer BC already has.
    coating_resistance = coating_thickness_m / coating_k if coating_thickness_m > 0 else 0.0

    foam.write_dict(case_dir / "constant" / "regionProperties", "dictionary", "regionProperties",
                     "regions\n(\n    solid       (solid)\n);", location="constant")
    foam.write_dict(case_dir / "constant" / "solid" / "thermophysicalProperties", "dictionary",
                     "thermophysicalProperties", f"""thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIsoSolid;
    thermo          eConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleInternalEnergy;
}}
mixture
{{
    specie {{ nMoles 1; molWeight 100; }}
    transport {{ kappa {k}; }}
    thermodynamics {{ Hf 0; Cv {cp}; }}
    equationOfState {{ rho {rho}; }}
}}""", location="constant/solid")
    foam.write_dict(case_dir / "constant" / "solid" / "radiationProperties", "dictionary",
                     "radiationProperties",
                     "radiationModel  none;\n\nabsorptionEmissionModel none;\n\nscatterModel    none;\n\nsootModel       none;",
                     location="constant/solid")
    foam.write_dict(case_dir / "system" / "solid" / "fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }""")
    foam.write_dict(case_dir / "system" / "solid" / "fvSolution", "dictionary", "fvSolution", """solvers
{
    "(h|e)" { solver PCG; preconditioner DIC; tolerance 1e-08; relTol 0; }
    "(h|e)Final" { $e; }
}
PIMPLE { nNonOrthogonalCorrectors 0; }
relaxationFactors { equations { ".*" 1; } }""")
    foam.write_dict(case_dir / "system" / "fvSchemes", "dictionary", "fvSchemes",
                     "ddtSchemes {}\ngradSchemes {}\ndivSchemes {}\nlaplacianSchemes {}\ninterpolationSchemes {}\nsnGradSchemes {}")
    foam.write_dict(case_dir / "system" / "fvSolution", "dictionary", "fvSolution",
                     "PIMPLE { nOuterCorrectors 1; }")

    functions_txt = ""
    if probe_points:
        blocks = []
        for name, pt in probe_points.items():
            blocks.append(f"""
    {name}
    {{
        type        patchProbes;
        libs        ("libsampling.so");
        region      solid;
        patchName   body_outerSurface;
        fields      (T);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations
        (
            ({pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f})
        );
    }}""")
        functions_txt = "\n\nfunctions\n{" + "".join(blocks) + "\n}"

    foam.write_dict(case_dir / "system" / "controlDict", "dictionary", "controlDict", f"""libs            ("libliquidCoupling.so");
application     chtMultiRegionFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {T_MAX};
deltaT          {DT};
writeControl    adjustableRunTime;
writeInterval   {TOUCH_RECORD_AT};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
adjustTimeStep  no;
runTimeModifiable false;{functions_txt}""")

    # §71: snappyHexMesh's own automatic patch-naming convention -- the
    # geometry entry name ("body", from the STL's own `body.stl` key)
    # prefixes each of the STL's named `solid` blocks.
    foam.write_dict(case_dir / "0" / "solid" / "T", "volScalarField", "T", f"""dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_AMB_K};

boundaryField
{{
    body_innerSurface
    {{
        type            liquidCoupling;
        h               {tms.H_LIQ};
        mLiquid         {mass_liquid_kg};
        cLiquid         {tms.CP_LIQUID};
        Tliquid0        {T0_LIQUID_K};
        // §74: a real glaze/enamel/liner covers a piece inside AND out --
        // same coating applied to both surfaces, via the two different
        // mechanisms each BC actually has for it (this plugin's own
        // coatingResistance vs. the built-in thicknessLayers/kappaLayers
        // below).
        coatingResistance {coating_resistance};
        value           uniform {T_AMB_K};
    }}
    body_outerSurface
    {{
        type            externalWallHeatFluxTemperature;
        h               uniform {tms.H_AIR};
        Ta              uniform {T_AMB_K};
        emissivity      {tms.EMISSIVITY_SURFACE};
        {_q_field_entry(extra_flux_field)}
        thicknessLayers ({coating_thickness_m});
        kappaLayers     ({coating_k});
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
    body_bottom {{ type zeroGradient; }}
    body_top    {{ type zeroGradient; }}
}}""", location="0/solid")


def _write_handle_thermal_case(case_dir, handle_material_idx, table_contact_K, mid_point, h_joint):
    """§78: the handle's own solve. Root+tip are ONE merged `contact`
    patch (see mug_ffd_geometry.build_handle_stl's own comment for the
    real snappyHexMesh reliability issue that forced this -- not a design
    preference), driven by a single time-series averaging the body's real
    local temperature at BOTH actual attach points. This is the same
    single-attachment-point simplification v6/§68's own handle model
    already used -- not a new one, just arrived at for a different
    reason. The body-SIDE effect (§78's own `_localized_flux_field`)
    still acts at both real attach points independently -- only the
    handle's OWN internal solve shares one driving temperature. Same
    externalWallHeatFluxTemperature Robin-BC mechanism as the body's own
    ambient patch, just with a `table` Function1 instead of a constant --
    a real, standard OpenFOAM mechanism, not custom code.

    §82: `h_joint` is the candidate's own geometry/material-derived
    contact conductance (tms.joint_conductance_w_m2k(joint_material_idx)),
    replacing the old fixed tms.H_HANDLE_CONTACT -- the joint is now a
    real categorical design choice, not a shared constant."""
    _, k, rho, cp = tms.HANDLE_MATERIALS[int(round(handle_material_idx))]
    foam.write_dict(case_dir / "constant" / "regionProperties", "dictionary", "regionProperties",
                     "regions\n(\n    solid       (solid)\n);", location="constant")
    foam.write_dict(case_dir / "constant" / "solid" / "thermophysicalProperties", "dictionary",
                     "thermophysicalProperties", f"""thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIsoSolid;
    thermo          eConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleInternalEnergy;
}}
mixture
{{
    specie {{ nMoles 1; molWeight 100; }}
    transport {{ kappa {k}; }}
    thermodynamics {{ Hf 0; Cv {cp}; }}
    equationOfState {{ rho {rho}; }}
}}""", location="constant/solid")
    foam.write_dict(case_dir / "constant" / "solid" / "radiationProperties", "dictionary",
                     "radiationProperties",
                     "radiationModel  none;\n\nabsorptionEmissionModel none;\n\nscatterModel    none;\n\nsootModel       none;",
                     location="constant/solid")
    foam.write_dict(case_dir / "system" / "solid" / "fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }""")
    foam.write_dict(case_dir / "system" / "solid" / "fvSolution", "dictionary", "fvSolution", """solvers
{
    "(h|e)" { solver PCG; preconditioner DIC; tolerance 1e-08; relTol 0; }
    "(h|e)Final" { $e; }
}
PIMPLE { nNonOrthogonalCorrectors 0; }
relaxationFactors { equations { ".*" 1; } }""")
    foam.write_dict(case_dir / "system" / "fvSchemes", "dictionary", "fvSchemes",
                     "ddtSchemes {}\ngradSchemes {}\ndivSchemes {}\nlaplacianSchemes {}\ninterpolationSchemes {}\nsnGradSchemes {}")
    foam.write_dict(case_dir / "system" / "fvSolution", "dictionary", "fvSolution",
                     "PIMPLE { nOuterCorrectors 1; }")

    foam.write_dict(case_dir / "system" / "controlDict", "dictionary", "controlDict", f"""application     chtMultiRegionFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {T_MAX};
deltaT          {DT};
writeControl    adjustableRunTime;
writeInterval   {TOUCH_RECORD_AT};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
adjustTimeStep  no;
runTimeModifiable false;

functions
{{
    handleMidProbe
    {{
        type        probes;
        libs        ("libsampling.so");
        region      solid;
        fields      (T);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations
        (
            ({mid_point[0]:.8f} {mid_point[1]:.8f} {mid_point[2]:.8f})
        );
    }}
    contactProbe
    {{
        type        patchProbes;
        libs        ("libsampling.so");
        region      solid;
        patchName   handle_contact;
        fields      (T);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations ( (0 0 0) );
    }}
}}""")

    table_contact_txt = "\n".join(f"        ({t:.4f} {v:.6f})" for t, v in table_contact_K)
    foam.write_dict(case_dir / "0" / "solid" / "T", "volScalarField", "T", f"""dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_AMB_K};

boundaryField
{{
    handle_contact
    {{
        type            externalWallHeatFluxTemperature;
        h               uniform {h_joint};
        Ta              table
(
{table_contact_txt}
        );
        thicknessLayers ();
        kappaLayers     ();
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
    handle_lateral
    {{
        // Convective loss to ambient along the handle's length -- v6's
        // own H_AIR*P_h fin term, no radiative addition (v6 doesn't
        // radiate from the handle either).
        type            externalWallHeatFluxTemperature;
        h               uniform {tms.H_AIR};
        Ta              uniform {T_AMB_K};
        thicknessLayers ();
        kappaLayers     ();
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
}}""", location="0/solid")


def _localized_flux_field(case_dir, p0, p1, q_root_w, q_tip_w, decay_length_m, outer_surface_area_m2):
    """§78: the handle's real heat draw, applied back onto the body as a
    LOCALIZED sink (a Gaussian falloff around each real attach point),
    not a diffuse whole-patch average -- the actual "convincing" part of
    this phase. Normalized by (real triangulated surface area) / (real
    mesh face count) as the average face area -- not exact per-face-area
    weighting (would need its own functionObject), a documented
    approximation, same spirit as every other honestly-flagged
    simplification in this codebase."""
    import subprocess
    subprocess.run(["postProcess", "-region", "solid", "-func", "writeCellCentres", "-time", f"{TOUCH_RECORD_AT:g}"],
                    cwd=case_dir, capture_output=True, text=True)
    t_dir = case_dir / f"{TOUCH_RECORD_AT:g}" / "solid"
    cx = foam.read_boundary_field(t_dir / "Cx")["body_outerSurface"]
    cy = foam.read_boundary_field(t_dir / "Cy")["body_outerSurface"]
    cz = foam.read_boundary_field(t_dir / "Cz")["body_outerSurface"]
    face_centers = np.stack([cx, cy, cz], axis=1)
    n_faces = len(face_centers)
    avg_face_area = outer_surface_area_m2 / n_faces

    d0 = np.linalg.norm(face_centers - np.asarray(p0), axis=1)
    d1 = np.linalg.norm(face_centers - np.asarray(p1), axis=1)
    k0 = np.exp(-(d0 / decay_length_m) ** 2)
    k1 = np.exp(-(d1 / decay_length_m) ** 2)

    # Negative = extra heat LEAVING the wall here (matching this BC's own
    # qTot sign convention, same one §67's radiative term and §69's
    # earlier local-sink experiment both already used).
    q0 = -q_root_w * k0 / (np.sum(k0) * avg_face_area)
    q1 = -q_tip_w * k1 / (np.sum(k1) * avg_face_area)
    return q0 + q1


def check_feasibility_fast(params):
    """§85: the "cheap validity checker" tier, split out as its own
    directly-callable function instead of leaving these checks buried
    inline in `simulate()`, matching this project's own fidelity-ladder
    convention (`OracleFunction.fidelities`, cheapest first) -- direct
    user request ("we should have quick, fast validity checkers that we
    can run before throwing things at the oracle for full fidelity
    simulation"). Confirmed empirically worth having, not just tidiness:
    a real 960-candidate run (§77) already showed 94.9% of rejections
    caught by exactly these checks (attach-point distance, liquid volume,
    self-intersection) before ever invoking snappyHexMesh/chtMultiRegionFoam
    -- this function is that same cheap gate sequence, now something a
    sampler can call BEFORE ever dispatching a candidate into the
    multiprocessing worker pool, not just something that happens to run
    early inside one.

    Zero OpenFOAM subprocess calls -- pure numpy/scipy (FFD tessellation +
    a cKDTree nearest-surface-point query), runs on the host, no `openfoam`
    Docker service required. Deliberately STOPS short of the
    self-intersection `surfaceCheck` gate (§71's gate #1): that one DOES
    need the OpenFOAM binary, so it stays inside `simulate()` as the next,
    slightly-more-expensive rung, not folded in here -- keeping this
    function's own "no subprocess, no Docker" property real rather than
    approximate.

    Returns `dict(valid, reason)` on rejection, or `dict(valid=True,
    outer_grid=..., inner_grid=..., wall_vol=..., cavity_vol=...,
    liquid_volume_L=..., p0=..., pm=..., p1=...)` on success -- the caller
    (`simulate()` below) reuses these instead of recomputing the same FFD
    tessellation a second time."""
    (r_inner_mm, r_outer_mm, height_mm, disp_inner_mm, disp_outer_mm,
     struct_material_idx, coating_material_idx, coating_thickness_mm, handle_params) = _unpack(params)

    if min(r_inner_mm, r_outer_mm, height_mm) <= 0:
        return dict(valid=False, reason="non-positive base geometry")
    if not (0 <= struct_material_idx <= len(tms.STRUCTURAL_MATERIALS) - 1):
        return dict(valid=False, reason="struct_material_idx out of range")
    if not (0 <= coating_material_idx <= len(tms.COATING_MATERIALS) - 1):
        return dict(valid=False, reason="coating_material_idx out of range")
    if coating_thickness_mm < 0:
        return dict(valid=False, reason="negative coating_thickness_mm")
    if not (0 <= handle_params["handle_material_idx"] <= len(tms.HANDLE_MATERIALS) - 1):
        return dict(valid=False, reason="handle_material_idx out of range")
    if handle_params["handle_radius_mm"] <= 0:
        return dict(valid=False, reason="non-positive handle_radius_mm")
    if not (0 <= handle_params["joint_material_idx"] <= len(tms.JOINT_MATERIALS) - 1):
        return dict(valid=False, reason="joint_material_idx out of range")

    r_inner, r_outer, height = r_inner_mm / 1000.0, r_outer_mm / 1000.0, height_mm / 1000.0
    disp_inner, disp_outer = disp_inner_mm / 1000.0, disp_outer_mm / 1000.0

    bbox_min = (-max(r_inner, r_outer) * 1.5, -max(r_inner, r_outer) * 1.5, 0.0)
    bbox_max = (max(r_inner, r_outer) * 1.5, max(r_inner, r_outer) * 1.5, height)
    outer_grid = ffdgeo.deformed_surface_grid(r_outer, height, disp_outer, bbox_min, bbox_max, N_THETA_TESS, N_Z_TESS)
    inner_grid = ffdgeo.deformed_surface_grid(r_inner, height, disp_inner, bbox_min, bbox_max, N_THETA_TESS, N_Z_TESS)

    wall_vol, cavity_vol = ffdgeo.wall_and_cavity_volumes(outer_grid, inner_grid)
    liquid_volume_L = cavity_vol * 1000.0
    if liquid_volume_L < tms.MIN_LIQUID_VOLUME_L or wall_vol <= 0:
        return dict(valid=False, reason="liquid cavity below minimum-viable-volume floor, or non-positive wall volume")

    # §89: a real, non-obvious combination directly caught by eyeballing a
    # peek batch -- a 43mm-thick calabash_gourd wall ("can calabash gourds
    # even be shaped?"). MEAN thickness (not the nominal r_outer-r_inner)
    # since the FFD lattice can locally distort thickness well away from
    # that nominal value -- this check is about whether the material as a
    # WHOLE plausibly forms this wall, not a per-point structural check
    # (self-intersection/liquid-volume already cover local pinch-through).
    mean_thickness_mm = float(np.mean(np.linalg.norm(outer_grid - inner_grid, axis=-1))) * 1000.0
    tmin_mm, tmax_mm = tms.wall_thickness_range_mm(struct_material_idx)
    if not (tmin_mm <= mean_thickness_mm <= tmax_mm):
        struct_name = tms.STRUCTURAL_MATERIALS[int(round(struct_material_idx))][0]
        return dict(valid=False, reason=f"implausible wall thickness for {struct_name} "
                                         f"(mean {mean_thickness_mm:.1f}mm, plausible range {tmin_mm:.1f}-{tmax_mm:.1f}mm)")

    # §77 feasibility gate: do the handle's two attach points actually
    # land on the body's REAL (possibly wildly asymmetric) outer surface?
    # A real point-to-surface distance query, not assumed -- this is the
    # "floating donut" rejection the whole Phase B redesign exists for.
    p0, pm, p1 = _handle_world_points(handle_params, r_outer_mm, height_mm)
    outer_flat = outer_grid.reshape(-1, 3)
    d0_check = ffdgeo.nearest_surface_distance(p0, outer_flat)
    d1_check = ffdgeo.nearest_surface_distance(p1, outer_flat)
    if d0_check > ATTACH_TOLERANCE_M or d1_check > ATTACH_TOLERANCE_M:
        return dict(valid=False, reason=f"handle attach point(s) not on body surface (d0={d0_check*1000:.1f}mm, d1={d1_check*1000:.1f}mm)")

    # §97: a real, deep gap the §77 check above never covered -- it only
    # verifies the handle's two ENDPOINTS touch the surface, never that
    # the ARC BETWEEN them stays outside the body. Found by user
    # inspection in the real viewer (candidates with attach points on
    # near-opposite sides of the mug and a small bulge produced a handle
    # that visibly cut straight through the mug's own interior -- "one of
    # them is just stabbing randomly through the mug") and confirmed
    # numerically, not just visually: the offending candidate's own
    # centerline sat ~30mm from the outer surface AND ~25mm from the
    # inner cavity surface AT THE SAME POINT -- only possible if that
    # point is floating in open space inside the cavity, nowhere near
    # either wall, not looping around the outside the way a real handle
    # does. Every prior gate (attach points, self-intersection, meshing)
    # is blind to this: the handle and body are built and meshed as two
    # INDEPENDENT solids, connected only by the endpoint-distance
    # constraint, so nothing anywhere previously checked what happens
    # in between.
    #
    # Fixed with a real, cheap (no OpenFOAM) geometric test: bilinearly
    # interpolate the outer surface's own LOCAL radius (distance from the
    # central axis) at each centerline sample's (theta, z), same
    # (theta,z)-grid interpolation convention `mug-v2.astro`'s own
    # applyBodyColorsGrid-equivalent JS already uses for coloring --
    # reject if any INTERIOR sample (excluding a margin near each real
    # end, which legitimately sits at/inside the surface by construction)
    # sits closer to the axis than the local outer surface, i.e. inside
    # the body's own envelope rather than looping outside it.
    n_theta_g, n_z_g = N_THETA_TESS, N_Z_TESS
    outer_radius_grid = np.linalg.norm(outer_grid[:, :, :2], axis=-1)  # (n_z, n_theta)

    def local_outer_radius(world_xyz):
        theta = np.arctan2(world_xyz[1], world_xyz[0]) % (2 * np.pi)
        theta_pos = (theta / (2 * np.pi)) * n_theta_g
        t_lo = int(np.floor(theta_pos)) % n_theta_g
        t_hi = (t_lo + 1) % n_theta_g
        t_frac = theta_pos - np.floor(theta_pos)
        z_pos = np.clip(world_xyz[2] / height, 0.0, 1.0) * (n_z_g - 1)
        z_lo = int(np.clip(np.floor(z_pos), 0, n_z_g - 1))
        z_hi = int(np.clip(z_lo + 1, 0, n_z_g - 1))
        z_frac = z_pos - z_lo
        r00, r01 = outer_radius_grid[z_lo, t_lo], outer_radius_grid[z_lo, t_hi]
        r10, r11 = outer_radius_grid[z_hi, t_lo], outer_radius_grid[z_hi, t_hi]
        r_lo = r00 + (r01 - r00) * t_frac
        r_hi = r10 + (r11 - r10) * t_frac
        return r_lo + (r_hi - r_lo) * z_frac

    centerline_check = ffdgeo.bezier_path(p0, pm, p1, n=20)
    # skip the 3 samples nearest each real end -- those legitimately sit
    # at/inside the surface (that's the actual attach disc), only the
    # ARC IN BETWEEN needs to clear the body.
    for pt in centerline_check[3:-3]:
        r_local = local_outer_radius(pt)
        r_pt = float(np.linalg.norm(pt[:2]))
        if r_pt < r_local - ARC_CLEARANCE_TOLERANCE_M:
            return dict(valid=False, reason=f"handle arc passes through the body interior "
                                             f"(point radius {r_pt*1000:.1f}mm < local surface radius {r_local*1000:.1f}mm)")

    return dict(valid=True, outer_grid=outer_grid, inner_grid=inner_grid,
                wall_vol=wall_vol, cavity_vol=cavity_vol, liquid_volume_L=liquid_volume_L,
                p0=p0, pm=pm, p1=p1)


def simulate(params):
    """params: (PARAM_DIM,) array, see PARAM_NAMES. Returns a dict with
    `valid` and, if True, TARGET_NAMES."""
    (r_inner_mm, r_outer_mm, height_mm, disp_inner_mm, disp_outer_mm,
     struct_material_idx, coating_material_idx, coating_thickness_mm, handle_params) = _unpack(params)

    fast = check_feasibility_fast(params)
    if not fast["valid"]:
        return fast
    outer_grid, inner_grid = fast["outer_grid"], fast["inner_grid"]
    wall_vol, cavity_vol, liquid_volume_L = fast["wall_vol"], fast["cavity_vol"], fast["liquid_volume_L"]
    p0, pm, p1 = fast["p0"], fast["pm"], fast["p1"]

    r_inner, r_outer, height = r_inner_mm / 1000.0, r_outer_mm / 1000.0, height_mm / 1000.0
    disp_inner, disp_outer = disp_inner_mm / 1000.0, disp_outer_mm / 1000.0

    with foam.temp_case_dir("mug_v2_body_") as case_dir:
        # check_feasibility_fast already computed outer_grid/inner_grid
        # (and validated them) above -- build_body_stl is called again
        # here only because IT'S the one that also writes the actual STL
        # to disk, which the fast checker deliberately doesn't do. Same
        # deterministic FFD math, so this reproduces identical grids --
        # a small amount of duplicated computation (~ms) traded for
        # keeping the fast checker's own "no disk I/O" property real.
        stl_path = case_dir / "body.stl"
        outer_grid, inner_grid = ffdgeo.build_body_stl(
            stl_path, r_inner, r_outer, height, disp_inner, disp_outer,
            n_theta=N_THETA_TESS, n_z=N_Z_TESS)

        centerline = ffdgeo.bezier_path(p0, pm, p1, n=20)
        arc_length_m = float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
        _, k_handle, rho_handle, cp_handle = tms.HANDLE_MATERIALS[int(round(handle_params["handle_material_idx"]))]
        handle_radius_m = handle_params["handle_radius_mm"] / 1000.0
        mass_handle_kg = np.pi * handle_radius_m ** 2 * arc_length_m * rho_handle
        outer_surface_area_m2 = ffdgeo.triangle_area_sum(ffdgeo.surface_triangles(outer_grid))

        # §71 feasibility gate #1: cheap, before ever invoking snappyHexMesh.
        try:
            n_intersect = _self_intersection_count(stl_path, case_dir)
        except foam.OpenFOAMError:
            return dict(valid=False, reason="surfaceCheck itself failed")
        if n_intersect > SELF_INTERSECTION_MAX:
            return dict(valid=False, reason=f"self-intersecting surface ({n_intersect} locations)")

        _, k, rho, cp = tms.STRUCTURAL_MATERIALS[int(round(struct_material_idx))]
        mass_kg = wall_vol * rho + mass_handle_kg
        mass_liquid_kg = cavity_vol * tms.RHO_LIQUID

        # Robust insidePoint: the (theta,z) sample with the LARGEST local
        # wall thickness, at mid-height -- avoids landing on an accidental
        # near-zero-thickness point even for an otherwise-valid design.
        k_mid = N_Z_TESS // 2
        thickness = np.linalg.norm(outer_grid[k_mid] - inner_grid[k_mid], axis=1)
        i_best = int(np.argmax(thickness))
        inside_point = (outer_grid[k_mid, i_best] + inner_grid[k_mid, i_best]) / 2.0

        margin = max(r_inner, r_outer) * 0.6
        bbox_min = (-max(r_inner, r_outer) - margin, -max(r_inner, r_outer) - margin, -margin)
        bbox_max = (max(r_inner, r_outer) + margin, max(r_inner, r_outer) + margin, height + margin)

        try:
            _mesh_solid(case_dir, stl_path, "body", ("outerSurface", "innerSurface", "bottom", "top"),
                        inside_point, bbox_min, bbox_max, level=(1, 3))
        except _MeshInfeasible as e:
            return dict(valid=False, reason=str(e))

        # --- pass 1: body, baseline BC, probing its own local temperature
        # at BOTH real attach points -- exists only to drive the handle. ---
        _write_thermal_case(case_dir, struct_material_idx, mass_liquid_kg,
                             coating_material_idx, coating_thickness_mm,
                             probe_points={"attachProbe0": p0, "attachProbe1": p1})
        try:
            foam.run_tool(case_dir, ["chtMultiRegionFoam"], "chtMultiRegionFoam", timeout=300)
        except foam.OpenFOAMError:
            return dict(valid=False, reason="chtMultiRegionFoam (pass 1) failed")

        probe0_file = case_dir / "postProcessing" / "attachProbe0" / "solid" / "0" / "T"
        probe1_file = case_dir / "postProcessing" / "attachProbe1" / "solid" / "0" / "T"
        if not probe0_file.exists() or not probe1_file.exists():
            return dict(valid=False, reason="pass 1 attach-point probes missing")
        table_p0_K = foam.read_probe_series(probe0_file)
        table_p1_K = foam.read_probe_series(probe1_file)

        # --- pass 2: the handle, its own mesh, driven by pass 1. ---
        handle_stl_path = case_dir / "handle.stl"
        mid_point = ffdgeo.build_handle_stl(handle_stl_path, p0, pm, p1, handle_radius_m,
                                             n_path=20, n_theta=12)[len(centerline) // 2]
        h_margin = handle_radius_m * 4
        h_bbox_min = (min(p0[0], p1[0], pm[0]) - h_margin, min(p0[1], p1[1], pm[1]) - h_margin, min(p0[2], p1[2], pm[2]) - h_margin)
        h_bbox_max = (max(p0[0], p1[0], pm[0]) + h_margin, max(p0[1], p1[1], pm[1]) + h_margin, max(p0[2], p1[2], pm[2]) + h_margin)

        # §78: table_p0_K/table_p1_K share the identical time grid (same
        # controlDict/DT as the pass-1 solve they both came from) --
        # index-pairing, not float-time-matching, same robust convention
        # §69 already established for this exact situation.
        table_contact_K = [(t0, (v0 + v1) / 2.0) for (t0, v0), (_, v1) in zip(table_p0_K, table_p1_K)]

        with foam.temp_case_dir("mug_v2_handle_") as handle_dir:
            try:
                _mesh_solid(handle_dir, handle_stl_path, "handle", ("lateral", "contact"),
                            mid_point, h_bbox_min, h_bbox_max, level=(1, 2), background_n=16)
            except _MeshInfeasible as e:
                return dict(valid=False, reason=f"handle mesh: {e}")

            h_joint = tms.joint_conductance_w_m2k(handle_params["joint_material_idx"])
            _write_handle_thermal_case(handle_dir, handle_params["handle_material_idx"],
                                        table_contact_K, mid_point, h_joint)
            try:
                foam.run_tool(handle_dir, ["chtMultiRegionFoam"], "chtMultiRegionFoam", timeout=300)
            except foam.OpenFOAMError:
                return dict(valid=False, reason="chtMultiRegionFoam (handle) failed")

            handle_mid_probe = handle_dir / "postProcessing" / "handleMidProbe" / "solid" / "0" / "T"
            contact_probe_file = handle_dir / "postProcessing" / "contactProbe" / "solid" / "0" / "T"
            if not handle_mid_probe.exists() or not contact_probe_file.exists():
                return dict(valid=False, reason="handle probes missing")
            handle_series = foam.read_probe_series(handle_mid_probe)
            contact_series = foam.read_probe_series(contact_probe_file)
            handle_temp_60s_C = next(v for t, v in handle_series if t >= TOUCH_RECORD_AT) - 273.15

            # §78: the handle's REAL total heat draw, time-averaged (same
            # documented constant-approximation tradeoff §69 already made
            # for the old revolve oracle's own single-attach-point
            # version), split evenly between the two real attach points
            # for the BODY-side localized sink below -- the body still
            # loses heat at BOTH real locations, only the handle's OWN
            # internal solve shares one driving temperature.
            #
            # h_joint*deltaT is a per-area FLUX (W/m^2), not a total power
            # -- multiplying by the contact disc's own real area
            # (pi*radius^2) is what actually converts it to Watts. A real
            # bug caught by a genuine solver crash (SIGABRT deep in
            # heSolidThermo::calculate(), not a clean FATAL ERROR): the
            # first version of this line skipped that area multiplication
            # entirely, feeding _localized_flux_field a flux value where
            # it expected total watts -- the resulting per-face `q`
            # (itself divided by yet another area) came out with the
            # wrong units/magnitude entirely, diverging the solve outright.
            #
            # §82: h_joint is now the candidate's OWN joint-material-
            # derived conductance (the same value _write_handle_thermal_
            # case's BC above was driven by), not the old shared
            # tms.H_HANDLE_CONTACT constant -- a mechanical dry-fit joint
            # both runs hotter at the handle (weaker coupling to the pass-1
            # table) and draws less real power from the body than a fused/
            # brazed joint would, self-consistently, from one shared value.
            contact_area_m2 = np.pi * handle_radius_m ** 2
            q_contact_total_w = contact_area_m2 * float(np.mean(
                [h_joint * (tc - th) for (_, tc), (_, th) in zip(table_contact_K, contact_series)]))
            q_root_w = q_tip_w = q_contact_total_w / 2.0

        # --- pass 3: body again, with the handle's own LOCALIZED extra
        # loss at both real attach points -- authoritative results. ---
        decay_length_m = handle_radius_m * 3
        extra_flux = _localized_flux_field(case_dir, p0, p1, q_root_w, q_tip_w, decay_length_m, outer_surface_area_m2)
        _write_thermal_case(case_dir, struct_material_idx, mass_liquid_kg,
                             coating_material_idx, coating_thickness_mm,
                             extra_flux_field=extra_flux)
        try:
            foam.run_tool(case_dir, ["chtMultiRegionFoam"], "chtMultiRegionFoam", timeout=300)
        except foam.OpenFOAMError:
            return dict(valid=False, reason="chtMultiRegionFoam (pass 3) failed")

        t60_field = case_dir / f"{TOUCH_RECORD_AT:g}" / "solid" / "T"
        t_final_field = case_dir / f"{T_MAX:g}" / "solid" / "T"
        if not t60_field.exists() or not t_final_field.exists():
            return dict(valid=False, reason="expected time directories missing")
        fields_60 = foam.read_boundary_field(t60_field)
        touch_temp_60s_C = float(fields_60["body_outerSurface"].max()) - 273.15
        fields_final = foam.read_boundary_field(t_final_field, entry="refValue")
        temp_at_2h_C = float(fields_final["body_innerSurface"][0]) - 273.15

    return dict(
        valid=True,
        touch_temp_60s_C=touch_temp_60s_C,
        temp_at_2h_C=temp_at_2h_C,
        handle_temp_60s_C=handle_temp_60s_C,
        mass_kg=float(mass_kg),
        liquid_volume_L=float(liquid_volume_L),
    )


def worker_fn(conn, *params):
    """Matches this project's usual oracle worker convention (see
    mug_oracle.py) -- not wired into oracle_harness yet (Phase A2), but
    shaped so that wiring is a drop-in, not a redesign."""
    try:
        try:
            r = simulate(np.array(params, dtype=np.float64))
        except Exception as e:
            conn.send((False, f"sim error: {e}"))
            return
        if not r["valid"]:
            conn.send((False, r.get("reason", "invalid")))
            return
        conn.send((True, {name: r[name] for name in TARGET_NAMES}))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    return tuple(float(v) for v in params)

"""OpenFOAM-backed mug wall-conduction oracle (EXPERIMENT_LOG §65-68) --
replacing thermal_mug_spike.py's self-written solver per direct user
course reversal (§64). Deliberately narrower than v6 for now, same
incremental-versioning discipline this whole domain has always used:

  IN SCOPE: the structural wall (r_base/r_mid/r_rim, t_wall_rim/
  t_wall_base, struct_material_idx) -> touch_temp_60s_C AND temp_at_2h_C
  (real transient FVM conduction, run by chtMultiRegionFoam, with the
  liquid's own lumped energy balance coupled in -- see §66), mass_kg and
  liquid_volume_L (closed-form geometry, not physics -- computed directly,
  same as v6 does).

  §66: the liquid is no longer an infinite fixed-temperature reservoir.
  The `liquid` patch uses a compiled `liquidCoupling` boundary condition
  (scripts/openfoam_libs/liquidCoupling/) that integrates
  m*c*dTliquid/dt = -sum(h*(Tliquid-Tface)*faceArea) inline, once per
  solver timestep, and reports the wedge's own liquid temperature back at
  any time via the patch's own `refValue`. A `codedMixed` inline block
  (the originally planned mechanism) turned out to be a dead end: OpenFOAM
  unconditionally refuses runtime dynamicCode compilation when running as
  root (dynamicCode::checkSecurity()'s isAdministrator() check, no
  override -- confirmed directly against $FOAM_SRC, not assumed), which is
  how this project's `openfoam` service runs. A real `wmake libso` plugin
  sidesteps that AND compiles once at image-build time instead of once per
  candidate -- see EXPERIMENT_LOG §66 for the full account and the
  independent validation (scripts/openfoam_mug_spike/reference_1d_coupled.py)
  against this exact coupling law.

  §67: the `ambient` patch now also radiates, via
  `externalWallHeatFluxTemperature`'s own built-in `emissivity` option --
  no custom code at all. Its internal formula (h + eps*sigma*(Ta+Tp)*
  (Ta^2+Tp^2), read straight out of $FOAM_SRC) is algebraically IDENTICAL
  to thermal_mug_spike.py's own `_radiative_h` linearized-Stefan-Boltzmann
  trick -- same physics, same one-timestep-lagged linearization
  convention, OpenFOAM's own built-in implementation of it rather than a
  reimplementation.

  §68: the handle. v6's own handle model is already a strong
  simplification -- a 1D fin (cross-section area A_h, perimeter P_h)
  BOTH of whose ends attach to the SAME single wall node (`patch_local`,
  the wall's mid-height outer temperature at its fixed theta=0 location)
  via a contact conductance H_HANDLE_CONTACT, losing heat laterally to
  ambient along its length. That fidelity level doesn't need a real 3D
  CAD attachment or snappyHexMesh at all -- it's realized here as its own
  SEPARATE solid region/case (`_write_handle_case`), a straight
  SQUARE-cross-section rod (side=sqrt(A_h), preserving the real handle's
  cross-sectional area exactly, hence its axial conduction resistance and
  thermal mass too -- perimeter is ~12.8% larger than the real round
  handle's, a documented, quantified simplification, not silently
  wrong) meshed with plain `blockMesh` (no wedge -- nothing about a
  straight rod is axisymmetric). Both end faces get `Ta` set to a
  `table` Function1 built from a `patchProbes` sample of the WALL run's
  own `ambient` patch at its mid-height, taken AFTER the wall case
  finishes -- a ONE-WAY coupling (the handle doesn't feed back into the
  wall's own solve this phase). This isn't an arbitrary corner cut: the
  wall's wedge mesh has only ONE cell azimuthally (matching this phase's
  own no-azimuthal-variation scope, §65), so there's no specific "wedge
  column" for a handle's small heat sink to attach back onto even if
  two-way coupling were wanted -- one-way is the natural fidelity match,
  not a downgrade from what the mesh already resolves.

  NOT in scope yet, real gaps stated plainly, not hidden:
  - No two-way handle<->wall coupling (see §68 above for why this
    matches the wall mesh's own azimuthal resolution, not a shortcut).
  - No radiative loss on the handle itself (v6 doesn't model this either
    -- its own lateral loss term is H_AIR alone, no radiative addition).

Geometry: an axisymmetric WEDGE mesh (2.5 degree half-angle, matching
common OpenFOAM practice for axisymmetric problems) -- valid here because
nothing in this phase's physics varies azimuthally (no handle yet, no
azimuthal asymmetry of any kind), so resolving a full 360 degrees would
waste cells for zero physical benefit. Reuses thermal_mug_spike.py's own
`radius_at_height` (piecewise-linear base->mid->rim profile) directly,
not a reimplementation -- guarantees the geometry this oracle solves is
IDENTICAL to what v6 assumes, not a drifted approximation. Wall thickness
is linear from t_wall_base (z=0) to t_wall_rim (z=height) across the WHOLE
height (matching simulate_v6's own `row_t_wall_mm` formula exactly) --
together with the 2-segment radius profile, the true geometry is exactly
piecewise-linear with a single breakpoint at height/2, so a 2-block wedge
mesh (vertices at z=0, height/2, height) represents the real shape
EXACTLY, not an approximation refined only by mesh resolution within each
block.
"""
import math

import numpy as np

import openfoam_harness as foam
import thermal_mug_spike as tms

N_RADIAL = 12
N_AXIAL_PER_BLOCK = 8
N_THETA = 24  # full-revolve angular resolution (§69) -- matches v6's own
              # N_THETA only for a comparable visual granularity, not to
              # chase v6 parity; measure real cost before assuming this
              # value survives at dataset-generation scale.
T_MAX = 7200.0
DT = 2.0
TOUCH_RECORD_AT = 60.0
N_HANDLE_AXIAL = 20  # matches tms.N_HANDLE's own axial resolution

T0_LIQUID_K = tms.T0_LIQUID + 273.15
T_AMB_K = tms.T_AMB + 273.15


def _wall_thickness_mm(z, height, t_wall_base_mm, t_wall_rim_mm):
    return t_wall_base_mm + (t_wall_rim_mm - t_wall_base_mm) * (z / height)


def _profile_points(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm):
    """3 (z, r_inner, r_outer) points in meters, at z=0/height/2/height --
    the 2 breakpoints a piecewise-linear wedge mesh needs to be exact."""
    height = tms.HEIGHT
    zs = [0.0, height / 2.0, height]
    pts = []
    for z in zs:
        # radius_at_height only ever uses z/height as a dimensionless
        # fraction, so passing radii in mm alongside height/z in meters is
        # fine -- the mm units just pass straight through unchanged.
        r_in_mm = tms.radius_at_height(r_base_mm, r_mid_mm, r_rim_mm, height, z)
        t_mm = _wall_thickness_mm(z, height, t_wall_base_mm, t_wall_rim_mm)
        pts.append((z, r_in_mm / 1000.0, (r_in_mm + t_mm) / 1000.0))
    return pts




def _write_common_solid_dicts(case_dir, k, rho, cp):
    """regionProperties/thermophysicalProperties/radiationProperties/
    fvSchemes/fvSolution -- identical structure for both the wall case
    and the handle case (§68), only the material numbers differ. Shared
    here instead of duplicated so the two cases can't silently drift."""
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


def _write_full_block_mesh_dict(case_dir, profile_pts, n_theta):
    """§69: a real full 360-degree revolve (N_THETA angular sectors, each
    identical in structure to the old single wedge block) -- NOT a
    snappyHexMesh attachment, just repeating the same wedge-block pattern
    all the way around and sharing vertices between neighbors so
    blockMesh treats the seams as ordinary internal faces, no `wedge`
    patches or `mergePatchPairs` needed at all. Sector i=0 is centered on
    theta=0 (spans -pi/n_theta..+pi/n_theta) to match v6's own fixed
    +X handle placement convention (thermal_mug_spike.py's
    `theta_handle_idx=0`, mug.astro's handle mesh position) -- that
    sector's own outer-surface faces get pulled out into a SEPARATE
    `ambientHandle` patch (both axial blocks) so the handle's real local
    heat sink can act on just that one patch, not the whole ring."""
    def vtx_idx(level, theta_i, io):
        return level * (n_theta * 2) + theta_i * 2 + io

    verts = [None] * (3 * n_theta * 2)
    for level, (z, r_in, r_out) in enumerate(profile_pts):
        for i in range(n_theta):
            theta = -math.pi / n_theta + i * 2 * math.pi / n_theta
            c, s = math.cos(theta), math.sin(theta)
            verts[vtx_idx(level, i, 0)] = (r_in * c, r_in * s, z)
            verts[vtx_idx(level, i, 1)] = (r_out * c, r_out * s, z)

    blocks = []
    for level in (0, 1):
        for i in range(n_theta):
            j = (i + 1) % n_theta
            v0, v1 = vtx_idx(level, i, 0), vtx_idx(level, i, 1)
            v2, v3 = vtx_idx(level, j, 1), vtx_idx(level, j, 0)
            v4, v5 = vtx_idx(level + 1, i, 0), vtx_idx(level + 1, i, 1)
            v6, v7 = vtx_idx(level + 1, j, 1), vtx_idx(level + 1, j, 0)
            blocks.append(
                f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) "
                f"({N_RADIAL} 1 {N_AXIAL_PER_BLOCK}) simpleGrading (1 1 1)"
            )

    liquid_faces, ambient_faces, ambient_handle_faces = [], [], []
    bottom_faces, top_faces = [], []
    for level in (0, 1):
        for i in range(n_theta):
            j = (i + 1) % n_theta
            v0, v1 = vtx_idx(level, i, 0), vtx_idx(level, i, 1)
            v2, v3 = vtx_idx(level, j, 1), vtx_idx(level, j, 0)
            v4, v5 = vtx_idx(level + 1, i, 0), vtx_idx(level + 1, i, 1)
            v6, v7 = vtx_idx(level + 1, j, 1), vtx_idx(level + 1, j, 0)
            liquid_faces.append(f"({v0} {v3} {v7} {v4})")
            (ambient_handle_faces if i == 0 else ambient_faces).append(f"({v1} {v2} {v6} {v5})")
            if level == 0:
                bottom_faces.append(f"({v0} {v1} {v2} {v3})")
            else:
                top_faces.append(f"({v4} {v5} {v6} {v7})")

    vertices_txt = "\n".join(f"    ({x:.8f} {y:.8f} {z:.8f})" for x, y, z in verts)
    blocks_txt = "\n".join(blocks)

    body = f"""scale   1;

vertices
(
{vertices_txt}
);

blocks
(
{blocks_txt}
);

edges ();

boundary
(
    liquid
    {{
        type    patch;
        faces   ({" ".join(liquid_faces)});
    }}
    ambient
    {{
        type    patch;
        faces   ({" ".join(ambient_faces)});
    }}
    ambientHandle
    {{
        type    patch;
        faces   ({" ".join(ambient_handle_faces)});
    }}
    bottom
    {{
        type    patch;
        faces   ({" ".join(bottom_faces)});
    }}
    top
    {{
        type    patch;
        faces   ({" ".join(top_faces)});
    }}
);

mergePatchPairs ();
"""
    foam.write_dict(case_dir / "system" / "solid" / "blockMeshDict", "dictionary", "blockMeshDict", body)


def _write_case(case_dir, struct_material_idx, mass_liquid_kg, profile_pts, extra_flux_ambient_handle=None):
    """§69: full 360-degree mesh (see _write_full_block_mesh_dict). Run
    TWICE per candidate (a 2-pass approximation of the handle<->wall
    coupling, chosen over a real mesh-conformal mappedWall interface for
    lower implementation risk -- see EXPERIMENT_LOG §69):
      pass 1 (extra_flux_ambient_handle=None): `ambientHandle` behaves
        identically to `ambient` -- this pass exists only to feed the
        handle case its own driving temperature (via wallMidProbe).
      pass 2 is the separate handle solve (unchanged from §68).
      pass 3 (extra_flux_ambient_handle=<value>): `ambientHandle` gets an
        EXTRA constant heat flux (`externalWallHeatFluxTemperature`'s own
        built-in `q` entry -- not custom code) representing the handle's
        own time-averaged heat draw computed from pass 2 -- this is the
        pass whose results are authoritative (touch_temp_60s_C/
        temp_at_2h_C), since it's the one that actually reflects the
        handle's local cooling effect on the wall.
    """
    name, k, rho, cp = tms.STRUCTURAL_MATERIALS[int(round(struct_material_idx))]
    _write_common_solid_dicts(case_dir, k, rho, cp)

    # §68/§69: the wall's LOCAL (theta=0, mid-height) outer-surface
    # temperature, sampled at every solver step, drives the handle's own
    # case -- see this module's docstring. z=height/2 is profile_pts[1]
    # (the mesh's own middle breakpoint); theta=0 is the CENTER of the
    # `ambientHandle` sector (see _write_full_block_mesh_dict).
    _, _, r_out_mid = profile_pts[1]
    height_mid = profile_pts[1][0]
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
runTimeModifiable false;

functions
{{
    wallMidProbe
    {{
        type        patchProbes;
        libs        ("libsampling.so");
        region      solid;
        patchName   ambientHandle;
        fields      (T);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations
        (
            ({r_out_mid:.8f} 0 {height_mid:.8f})
        );
    }}
}}""")

    foam.write_dict(case_dir / "0" / "solid" / "T", "volScalarField", "T", f"""dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_AMB_K};

boundaryField
{{
    liquid
    {{
        // §66: compiled liquidCoupling plugin -- Ta evolves via the
        // liquid's own lumped energy balance instead of staying fixed at
        // T0_LIQUID_K (see this module's docstring + EXPERIMENT_LOG §66).
        type            liquidCoupling;
        h               {tms.H_LIQ};
        mLiquid         {mass_liquid_kg};
        cLiquid         {tms.CP_LIQUID};
        Tliquid0        {T0_LIQUID_K};
        value           uniform {T_AMB_K};
    }}
    ambient
    {{
        // §67: `emissivity` turns on externalWallHeatFluxTemperature's
        // OWN built-in linearized-Stefan-Boltzmann term -- no custom code
        // needed at all. Read straight out of $FOAM_SRC (this module's
        // docstring has the derivation): with emissivity>0 it adds
        // sigma*(Ta+Tp)*(Ta^2+Tp^2) onto h, which is algebraically
        // IDENTICAL to thermal_mug_spike.py's own `_radiative_h` --
        // same physics, same explicit one-timestep-lagged linearization
        // (Tp is the patch's own previous-iterate face value), just
        // OpenFOAM's built-in implementation of it instead of a
        // reimplementation.
        type            externalWallHeatFluxTemperature;
        h               uniform {tms.H_AIR};
        Ta              uniform {T_AMB_K};
        emissivity      {tms.EMISSIVITY_SURFACE};
        thicknessLayers ();
        kappaLayers     ();
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
    ambientHandle
    {{
        // §69: identical to `ambient` in pass 1 (extra_flux_ambient_handle
        // is None) -- pass 3 adds a constant `q` (built-in
        // externalWallHeatFluxTemperature entry, real physical units
        // W/m^2, NOT custom code) representing the handle's own
        // time-averaged heat draw, computed from the separate handle
        // solve. Negative q = extra heat LEAVING the wall here (the
        // handle is a sink), matching this BC's own qTot sign convention
        // (read straight out of $FOAM_SRC, same source §66/§67 already
        // used).
        type            externalWallHeatFluxTemperature;
        h               uniform {tms.H_AIR};
        Ta              uniform {T_AMB_K};
        emissivity      {tms.EMISSIVITY_SURFACE};
        {"q               uniform " + repr(extra_flux_ambient_handle) + ";" if extra_flux_ambient_handle is not None else ""}
        thicknessLayers ();
        kappaLayers     ();
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
    bottom {{ type zeroGradient; }}
    top    {{ type zeroGradient; }}
}}""", location="0/solid")


def _write_handle_block_mesh_dict(case_dir, side_m, length_m):
    """A straight SQUARE-cross-section rod, side x side x length -- see
    this module's docstring for why a square (not a real round handle,
    and not snappyHexMesh) is the right fidelity match for v6's own 1D-fin
    handle model. Local axes: x=length (0..length_m), y/z=cross-section
    (+-side/2). One block, (N_HANDLE_AXIAL, 1, 1) -- root/tip end faces
    are single quads, and each of the 4 lateral faces auto-subdivides
    into N_HANDLE_AXIAL sub-faces along x, all grouped under one
    `lateral` patch (a real convective loss surface, not 4 separate
    ones -- v6's own model treats lateral loss as one combined h*P term
    too)."""
    s = side_m / 2.0
    verts = [
        (0.0, -s, -s), (length_m, -s, -s), (length_m, s, -s), (0.0, s, -s),
        (0.0, -s, s), (length_m, -s, s), (length_m, s, s), (0.0, s, s),
    ]
    vertices_txt = "\n".join(f"    ({x:.8f} {y:.8f} {z:.8f})" for x, y, z in verts)
    body = f"""scale   1;

vertices
(
{vertices_txt}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N_HANDLE_AXIAL} 1 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    root
    {{
        type    patch;
        faces   ((0 3 7 4));
    }}
    tip
    {{
        type    patch;
        faces   ((1 2 6 5));
    }}
    lateral
    {{
        type    patch;
        faces   ((0 1 5 4) (3 2 6 7) (0 1 2 3) (4 5 6 7));
    }}
);

mergePatchPairs ();
"""
    foam.write_dict(case_dir / "system" / "solid" / "blockMeshDict", "dictionary", "blockMeshDict", body)


def _write_handle_case(case_dir, handle_material_idx, wall_temp_table_K, length_m):
    _, k, rho, cp = tms.HANDLE_MATERIALS[int(round(handle_material_idx))]
    _write_common_solid_dicts(case_dir, k, rho, cp)

    table_txt = "\n".join(f"        ({t:.4f} {v:.6f})" for t, v in wall_temp_table_K)

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
            ({length_m / 2.0:.8f} 0 0)
        );
    }}
    rootProbe
    {{
        // §69: the handle's own root-face temperature over time -- with
        // the wall's local temperature (wallMidProbe, above) gives
        // H_HANDLE_CONTACT*(T_wall-T_root) at each sample, i.e. the
        // handle's REAL heat draw, fed back into the wall's pass-3 `q`.
        type        patchProbes;
        libs        ("libsampling.so");
        region      solid;
        patchName   root;
        fields      (T);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations
        (
            (0 0 0)
        );
    }}
}}""")

    foam.write_dict(case_dir / "0" / "solid" / "T", "volScalarField", "T", f"""dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_AMB_K};

boundaryField
{{
    root
    {{
        // §68: both ends attach to the SAME wall-mid-height time series
        // -- matching v6's own `T_amb_right=patch_local, T_left_conv=
        // patch_local` (a single-attachment-point approximation, not a
        // real handle loop geometry), via `Ta`'s `table` Function1
        // reading the wall run's own patchProbes output.
        type            externalWallHeatFluxTemperature;
        h               uniform {tms.H_HANDLE_CONTACT};
        Ta              table
(
{table_txt}
        );
        thicknessLayers ();
        kappaLayers     ();
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
    tip
    {{
        type            externalWallHeatFluxTemperature;
        h               uniform {tms.H_HANDLE_CONTACT};
        Ta              table
(
{table_txt}
        );
        thicknessLayers ();
        kappaLayers     ();
        kappaMethod     solidThermo;
        value           uniform {T_AMB_K};
    }}
    lateral
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


def _run_wall_pass(profile_pts, struct_material_idx, mass_liquid_kg, extra_flux_ambient_handle=None):
    """One full wall solve. Returns (touch_temp_60s_C, temp_at_2h_C,
    wall_temp_table_K) or None on any failure -- §69's `simulate()` calls
    this twice (pass 1 to drive the handle, pass 3 with the handle's own
    local extra loss folded back in)."""
    with foam.temp_case_dir("mug_openfoam_wall_") as case_dir:
        try:
            _write_full_block_mesh_dict(case_dir, profile_pts, N_THETA)
            _write_case(case_dir, struct_material_idx, mass_liquid_kg, profile_pts, extra_flux_ambient_handle)
            foam.run_tool(case_dir, ["blockMesh", "-region", "solid"], "blockMesh")
            foam.run_tool(case_dir, ["chtMultiRegionFoam"], "chtMultiRegionFoam", timeout=300)
        except foam.OpenFOAMError:
            return None

        t60_field = case_dir / f"{TOUCH_RECORD_AT:g}" / "solid" / "T"
        t_final_field = case_dir / f"{T_MAX:g}" / "solid" / "T"
        probe_file = case_dir / "postProcessing" / "wallMidProbe" / "solid" / "0" / "T"
        if not t60_field.exists() or not t_final_field.exists() or not probe_file.exists():
            return None
        fields_60 = foam.read_boundary_field(t60_field)
        touch_temp_60s_C = float(max(fields_60["ambient"].max(), fields_60["ambientHandle"].max())) - 273.15

        # §66: the liquidCoupling BC's own `refValue` IS the lumped liquid
        # temperature it integrated -- reading it back at t=T_MAX is
        # exactly v6's temp_at_2h_C (its own `final_liq_temp_C`), not an
        # approximation of it.
        fields_final = foam.read_boundary_field(t_final_field, entry="refValue")
        temp_at_2h_C = float(fields_final["liquid"][0]) - 273.15

        # Read BEFORE this case dir is cleaned up (the `with` exits at the
        # end of this function) -- pass 1's own return value is the
        # handle case's driving `Ta` table (§68/§69).
        wall_temp_table_K = foam.read_probe_series(probe_file)

    return touch_temp_60s_C, temp_at_2h_C, wall_temp_table_K


def simulate(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
             handle_length_mm, handle_diameter_mm, handle_material_idx):
    if min(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm,
           handle_length_mm, handle_diameter_mm) <= 0:
        return dict(valid=False)
    if not (0 <= struct_material_idx <= len(tms.STRUCTURAL_MATERIALS) - 1):
        return dict(valid=False)
    if not (0 <= handle_material_idx <= len(tms.HANDLE_MATERIALS) - 1):
        return dict(valid=False)

    profile_pts = _profile_points(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm)

    # mass_kg/liquid_volume_L: closed-form from geometry, same as v6 --
    # not something the FVM solve needs to tell us.
    _, k, rho, cp = tms.STRUCTURAL_MATERIALS[int(round(struct_material_idx))]
    mass_struct_kg = 0.0
    vol_liq = 0.0
    for i in range(2):
        z1, r_in1, r_out1 = profile_pts[i]
        z2, r_in2, r_out2 = profile_pts[i + 1]
        h_seg = z2 - z1
        vol_wall = tms.frustum_volume(r_out1, r_out2, h_seg) - tms.frustum_volume(r_in1, r_in2, h_seg)
        mass_struct_kg += vol_wall * rho
        vol_liq += tms.frustum_volume(r_in1, r_in2, h_seg)
    liquid_volume_L = vol_liq * 1000.0
    if liquid_volume_L < tms.MIN_LIQUID_VOLUME_L:
        return dict(valid=False)

    mass_liquid_kg = vol_liq * tms.RHO_LIQUID

    # §68: handle geometry -- square rod of the SAME cross-sectional area
    # as the real round handle (see this module's docstring for the exact
    # ~12.8% perimeter/lateral-loss-area tradeoff this implies).
    _, k_h, rho_h, cp_h = tms.HANDLE_MATERIALS[int(round(handle_material_idx))]
    length_m = handle_length_mm / 1000.0
    diameter_m = handle_diameter_mm / 1000.0
    a_h = math.pi * (diameter_m / 2) ** 2
    side_m = math.sqrt(a_h)
    mass_handle_kg = a_h * length_m * rho_h
    mass_kg = mass_struct_kg + mass_handle_kg

    # --- pass 1: wall, ambientHandle behaving like plain ambient -- exists
    # only to drive the handle case (§69's own docstring). ---
    pass1 = _run_wall_pass(profile_pts, struct_material_idx, mass_liquid_kg)
    if pass1 is None:
        return dict(valid=False)
    _, _, wall_temp_table_K = pass1

    # --- pass 2: the handle, one-way driven by pass 1 (§68). ---
    with foam.temp_case_dir("mug_openfoam_handle_") as handle_dir:
        try:
            _write_handle_block_mesh_dict(handle_dir, side_m, length_m)
            _write_handle_case(handle_dir, handle_material_idx, wall_temp_table_K, length_m)
            foam.run_tool(handle_dir, ["blockMesh", "-region", "solid"], "blockMesh")
            foam.run_tool(handle_dir, ["chtMultiRegionFoam"], "chtMultiRegionFoam", timeout=300)
        except foam.OpenFOAMError:
            return dict(valid=False)

        handle_probe_file = handle_dir / "postProcessing" / "handleMidProbe" / "solid" / "0" / "T"
        root_probe_file = handle_dir / "postProcessing" / "rootProbe" / "solid" / "0" / "T"
        if not handle_probe_file.exists() or not root_probe_file.exists():
            return dict(valid=False)
        handle_series = foam.read_probe_series(handle_probe_file)
        root_series = foam.read_probe_series(root_probe_file)
        # First sample at/after 60s -- same "first sample >= record_at"
        # convention v6's own `touch_temp_at`/`handle_temp_at` recording
        # uses, not just the nearest.
        handle_temp_60s_C = next(v for t, v in handle_series if t >= TOUCH_RECORD_AT) - 273.15

        # §69: the handle's REAL heat draw (W/m^2 at its root, using the
        # SAME H_HANDLE_CONTACT law the BC itself uses), time-averaged --
        # a constant approximation of what's really a time-varying flux,
        # applied back onto the wall in pass 3. Both series step through
        # the identical deltaT sequence (same controlDict), so zipping by
        # index (not matching on float time values) is the robust way to
        # pair them up.
        draws = [tms.H_HANDLE_CONTACT * (tw - tr) for (_, tw), (_, tr) in zip(wall_temp_table_K, root_series)]
        q_handle_draw = float(np.mean(draws)) if draws else 0.0

    # --- pass 3: wall again, now with the handle's own local extra loss
    # applied to `ambientHandle` -- this pass's results are authoritative. ---
    pass3 = _run_wall_pass(profile_pts, struct_material_idx, mass_liquid_kg, extra_flux_ambient_handle=-q_handle_draw)
    if pass3 is None:
        return dict(valid=False)
    touch_temp_60s_C, temp_at_2h_C, _ = pass3

    return dict(
        valid=True,
        touch_temp_60s_C=touch_temp_60s_C,
        temp_at_2h_C=temp_at_2h_C,
        handle_temp_60s_C=handle_temp_60s_C,
        mass_kg=float(mass_kg),
        liquid_volume_L=float(liquid_volume_L),
    )

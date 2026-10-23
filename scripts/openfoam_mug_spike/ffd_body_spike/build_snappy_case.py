"""Full mesh pipeline spike (EXPERIMENT_LOG §71, Phase A0 step 4): blockMesh
(background box) -> surfaceFeatures -> snappyHexMesh -overwrite -> checkMesh,
on the FFD-deformed body STL from ffd_geometry.py. Confirms the real
snappyHexMesh dictionary syntax (pulled from tutorials/mesh/snappyHexMesh/
flange, not guessed) works end-to-end on our own generated geometry, and
that innerSurface/outerSurface/bottom/top patches survive into the final
mesh.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/work/scripts")
import numpy as np
import openfoam_harness as foam
import mug_ffd_geometry as ffd

R_INNER, R_OUTER, HEIGHT = 0.030, 0.033, 0.10


def build_case(case_dir, disp_inner, disp_outer, insidePoint):
    (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)

    ffd.build_body_stl(case_dir / "constant" / "triSurface" / "body.stl",
                        R_INNER, R_OUTER, HEIGHT, disp_inner, disp_outer)

    margin = max(R_INNER, R_OUTER) * 0.6
    lo = -max(R_INNER, R_OUTER) - margin
    hi = max(R_INNER, R_OUTER) + margin
    zlo, zhi = -margin, HEIGHT + margin
    foam.write_dict(case_dir / "system" / "blockMeshDict", "dictionary", "blockMeshDict", f"""
convertToMeters 1;
vertices
(
    ({lo:.6f} {lo:.6f} {zlo:.6f})
    ({hi:.6f} {lo:.6f} {zlo:.6f})
    ({hi:.6f} {hi:.6f} {zlo:.6f})
    ({lo:.6f} {hi:.6f} {zlo:.6f})
    ({lo:.6f} {lo:.6f} {zhi:.6f})
    ({hi:.6f} {lo:.6f} {zhi:.6f})
    ({hi:.6f} {hi:.6f} {zhi:.6f})
    ({lo:.6f} {hi:.6f} {zhi:.6f})
);
blocks ( hex (0 1 2 3 4 5 6 7) (24 24 24) simpleGrading (1 1 1) );
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
                     'surfaces ("body.stl");\nincludedAngle 150;')

    foam.write_dict(case_dir / "system" / "snappyHexMeshDict", "dictionary", "snappyHexMeshDict", f"""
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    body.stl
    {{
        type triSurfaceMesh;
        name body;
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 200000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;
    features
    (
        {{ file "body.eMesh"; level 1; }}
    );
    refinementSurfaces
    {{
        body
        {{
            level (1 2);
            regions
            {{
                outerSurface {{ level (1 2); }}
                innerSurface {{ level (1 2); }}
                bottom       {{ level (1 2); }}
                top          {{ level (1 2); }}
            }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    insidePoint ({insidePoint[0]:.6f} {insidePoint[1]:.6f} {insidePoint[2]:.6f});
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

    foam.write_dict(case_dir / "system" / "controlDict", "dictionary", "controlDict",
                     "application snappyHexMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
                     "endTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\npurgeWrite 0;\n"
                     "writeFormat ascii;\nwritePrecision 6;\ntimeFormat general;\ntimePrecision 6;\n"
                     "runTimeModifiable false;")
    foam.write_dict(case_dir / "system" / "fvSchemes", "dictionary", "fvSchemes", "")
    foam.write_dict(case_dir / "system" / "fvSolution", "dictionary", "fvSolution", "")


if __name__ == "__main__":
    zero_disp = np.zeros((2, 2, 3, 3))
    outer_disp = np.zeros((2, 2, 3, 3))
    outer_disp[1, 1, 1] = [0.015, 0.0, 0.0]

    with foam.temp_case_dir("snappy_body_spike_") as d:
        build_case(d, zero_disp, outer_disp, insidePoint=(0.0315, 0.0, 0.05))
        try:
            print(foam.run_tool(d, ["blockMesh"], "blockMesh")[-500:])
            print(foam.run_tool(d, ["surfaceFeatures"], "surfaceFeatures", timeout=60)[-500:])
            print(foam.run_tool(d, ["snappyHexMesh", "-overwrite"], "snappyHexMesh", timeout=180)[-3000:])
            print(foam.run_tool(d, ["checkMesh"], "checkMesh", timeout=60)[-2000:])
        except foam.OpenFOAMError as e:
            print("FAILED:", str(e)[:3000])

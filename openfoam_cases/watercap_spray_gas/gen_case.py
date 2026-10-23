#!/usr/bin/env python3
"""watercap_spray_gas: the spray stage's gas phase in OpenFOAM 10, with the spray's momentum
source frozen as a body force -- the cross-check of StoneLab spray_cfd's turbulence models
(mixing length, k-omega SST) on the flow a spray actually drives.

Input: an .npz from StoneLab (rf, zf: the spray_cfd grid's radial and axial faces; Fz, Fr: the
converged spray force per cell (N); R, U, rho, L; u_in, k_in, w_in: the developed SST inlet).
Mesh: a 5-degree wedge whose cell faces contain spray_cfd's (one blockMesh block per spray_cfd
radial row, each spray_cfd cell split into --refine x --refine cells), so the force maps exactly:
the force density is constant over each spray_cfd cell, read from constant/sprayForce.dat by a
coded fvModel. Inlet: StoneLab's developed SST profile (u, k, omega) by radius (coded BCs);
outlet: p = 0, zero gradient; tube wall: no slip, low-Re (k = 0, omegaWallFunction on a y+ < 1
mesh, nutLowReWallFunction). simpleFoam, kOmegaSST, no buoyancy (the frozen comparison is
momentum only).

    python3 gen_case.py CASE --npz frozen.npz --refine 1 --scheme upwind
    ./Allrun CASE ...            (Docker: vega-openfoam:latest)
"""
import argparse
import math
import os

import numpy as np

HDR = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  10
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
"""
MU = 1.81e-5


def write(path, cls, loc, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HDR.format(cls=cls, loc=loc, obj=obj) + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--refine", type=int, default=1, help="OpenFOAM cells per spray_cfd cell, each direction")
    ap.add_argument("--scheme", default="upwind", choices=["upwind", "linearUpwind"])
    ap.add_argument("--iters", type=int, default=6000)
    a = ap.parse_args()
    d = np.load(a.npz)
    rf, zf, Fz, Fr = d["rf"], d["zf"], d["Fz"], d["Fr"]
    R, U, rho, L = float(d["R"]), float(d["U"]), float(d["rho"]), float(d["L"])
    nu = MU / rho
    nr, nz = Fz.shape
    m = a.refine
    ang = math.radians(2.5)
    vol = np.pi * (rf[1:] ** 2 - rf[:-1] ** 2)[:, None] * np.diff(zf)[None, :]
    rc = 0.5 * (rf[1:] + rf[:-1])
    os.makedirs(a.out, exist_ok=True)
    # ---- force density (m/s^2, kinematic) and inlet profiles, read by the coded fvModel / BCs
    os.makedirs(f"{a.out}/constant", exist_ok=True)
    with open(f"{a.out}/constant/sprayForce.dat", "w") as f:
        f.write(f"{nr} {nz}\n" + " ".join(f"{x:.12g}" for x in rf) + "\n" + " ".join(f"{x:.12g}" for x in zf) + "\n")
        f.write(" ".join(f"{x:.9g}" for x in (Fz / vol / rho).ravel()) + "\n")
        f.write(" ".join(f"{x:.9g}" for x in (Fr / vol / rho).ravel()) + "\n")
    with open(f"{a.out}/constant/inletProfile.dat", "w") as f:
        f.write(f"{nr}\n" + " ".join(f"{x:.12g}" for x in rc) + "\n")
        for arr in (d["u_in"], d["k_in"], d["w_in"]):
            f.write(" ".join(f"{x:.9g}" for x in arr) + "\n")
    # ---- mesh: one block per radial row, axial cells uniform (spray_cfd's zf is uniform)
    verts = [(0.0, 0.0, 0.0), (0.0, 0.0, L)]          # axis bottom/top (collapsed wedge apex)
    vid = {}

    def V(i, top, side):
        if i == 0:
            return 1 if top else 0
        key = (i, top, side)
        if key not in vid:
            r = rf[i]
            vid[key] = len(verts)
            verts.append((r * math.cos(ang), (-1 if side == 0 else 1) * r * math.sin(ang), L if top else 0.0))   # right-handed hex
        return vid[key]

    blocks, front, back, inlet, outlet, wall = [], [], [], [], [], []
    for i in range(nr):
        hexv = [V(i, 0, 0), V(i + 1, 0, 0), V(i + 1, 0, 1), V(i, 0, 1),
                V(i, 1, 0), V(i + 1, 1, 0), V(i + 1, 1, 1), V(i, 1, 1)]
        blocks.append(f"    hex ({' '.join(map(str, hexv))}) ({m} 1 {nz * m}) simpleGrading (1 1 1)")
        front.append(f"({V(i, 0, 0)} {V(i + 1, 0, 0)} {V(i + 1, 1, 0)} {V(i, 1, 0)})")
        back.append(f"({V(i, 0, 1)} {V(i, 1, 1)} {V(i + 1, 1, 1)} {V(i + 1, 0, 1)})")
        inlet.append(f"({V(i, 0, 0)} {V(i, 0, 1)} {V(i + 1, 0, 1)} {V(i + 1, 0, 0)})")
        outlet.append(f"({V(i, 1, 0)} {V(i + 1, 1, 0)} {V(i + 1, 1, 1)} {V(i, 1, 1)})")
    wall.append(f"({V(nr, 0, 0)} {V(nr, 0, 1)} {V(nr, 1, 1)} {V(nr, 1, 0)})")

    def patch(name, typ, faces):
        fl = "\n".join("            " + x for x in faces)
        return f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n{fl}\n        );\n    }}\n"

    bm = ("convertToMeters 1;\n\nvertices\n(\n" + "\n".join(f"    ({x:.12g} {y:.12g} {z:.12g})" for x, y, z in verts)
          + "\n);\n\nblocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges\n(\n);\n\nboundary\n(\n"
          + patch("front", "wedge", front) + patch("back", "wedge", back) + patch("inlet", "patch", inlet)
          + patch("outlet", "patch", outlet) + patch("tubeWall", "wall", wall)
          + "    axis\n    {\n        type empty;\n        faces\n        (\n            (0 1 1 0)\n        );\n    }\n);\n\nmergePatchPairs\n(\n);\n")
    write(f"{a.out}/system/blockMeshDict", "dictionary", "system", "blockMeshDict", bm)
    write(f"{a.out}/constant/physicalProperties", "dictionary", "constant", "physicalProperties",
          f"viscosityModel  constant;\nnu              [0 2 -1 0 0 0 0] {nu:.9g};\n")
    write(f"{a.out}/constant/momentumTransport", "dictionary", "constant", "momentumTransport",
          "simulationType RAS;\nRAS\n{\n    model           kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}\n")
    # ---- the frozen spray force: a coded fvModel (constant density over each spray_cfd cell)
    write(f"{a.out}/constant/fvModels", "dictionary", "constant", "fvModels", r"""sprayForce
{
    type            coded;
    selectionMode   all;
    field           U;

    codeInclude
    #{
        #include "IFstream.H"
    #};

    codeAddSup
    #{
        static autoPtr<vectorField> F;
        if (!F.valid())
        {
            IFstream is(mesh().time().constant()/"sprayForce.dat");
            label nr, nz;
            is >> nr >> nz;
            scalarField rf(nr + 1), zf(nz + 1), fz(nr*nz), fr(nr*nz);
            forAll(rf, i) { is >> rf[i]; }
            forAll(zf, j) { is >> zf[j]; }
            forAll(fz, n) { is >> fz[n]; }
            forAll(fr, n) { is >> fr[n]; }
            const vectorField& C = mesh().C();
            F.reset(new vectorField(C.size(), Zero));
            forAll(C, c)
            {
                const scalar r = C[c].x()/Foam::cos(degToRad(2.5));   // the wedge midplane radius (faces are chords)
                label i = 0; while (i < nr - 1 && rf[i + 1] <= r) { i++; }
                label j = 0; while (j < nz - 1 && zf[j + 1] <= C[c].z()) { j++; }
                const scalar f_r = fr[i*nz + j];
                const vector er = r > 0 ? vector(C[c].x()/r, C[c].y()/r, 0) : vector(1, 0, 0);
                F()[c] = f_r*er + vector(0, 0, fz[i*nz + j]);
            }
            Info<< "sprayForce: total " << gSum(F()*mesh().V()) << endl;
        }
        const DimensionedField<vector, volMesh> Fd
        (
            IOobject("sprayF", mesh().time().timeName(), mesh()),
            mesh(),
            dimVelocity/dimTime,
            F()
        );
        eqn += Fd;
    #};
}
""")
    # ---- fields: inlet by radius from StoneLab's developed SST profile
    def inlet_code(col, vec):
        val = "vector(0, 0, v)" if vec else "v"
        return f"""        type            codedFixedValue;
        value           uniform {"(0 0 0)" if vec else "1"};
        name            inlet_{col};
        codeInclude
        #{{
            #include "IFstream.H"
        #}};
        code
        #{{
            IFstream is(this->db().time().constant()/"inletProfile.dat");
            label n; is >> n;
            scalarField rc(n), u(n), k(n), w(n);
            forAll(rc, i) {{ is >> rc[i]; }}
            forAll(u, i) {{ is >> u[i]; }}
            forAll(k, i) {{ is >> k[i]; }}
            forAll(w, i) {{ is >> w[i]; }}
            const scalarField& prof = {col};
            const vectorField Cf(this->patch().Cf());
            {"vectorField" if vec else "scalarField"} out(Cf.size());
            forAll(Cf, f)
            {{
                const scalar r = Cf[f].x()/Foam::cos(degToRad(2.5));
                label i = 0; while (i < n - 2 && rc[i + 1] < r) {{ i++; }}
                const scalar t = min(max((r - rc[i])/(rc[i + 1] - rc[i]), 0.0), 1.0);
                const scalar v = (1 - t)*prof[i] + t*prof[i + 1];
                out[f] = {val};
            }}
            operator==(out);
        #}};"""

    def field(name, cls, dim, internal, inl, outl, wal):
        body = f"""dimensions      {dim};

internalField   {internal};

boundaryField
{{
    "(front|back)"
    {{
        type wedge;
    }}
    axis
    {{
        type empty;
    }}
    inlet
    {{
{inl}
    }}
    outlet
    {{
{outl}
    }}
    tubeWall
    {{
{wal}
    }}
}}
"""
        write(f"{a.out}/0/{name}", cls, "0", name, body)

    k0 = float(np.mean(d["k_in"]))
    w0 = float(np.median(d["w_in"]))
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", f"uniform (0 0 {U:.6g})", inlet_code("u", True),
          "        type            zeroGradient;", "        type            noSlip;")
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0", "        type            zeroGradient;",
          "        type            fixedValue;\n        value           uniform 0;", "        type            zeroGradient;")
    field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k0:.6g}", inlet_code("k", False),
          "        type            inletOutlet;\n        inletValue      uniform 1e-8;\n        value           uniform 1e-8;",
          "        type            fixedValue;\n        value           uniform 1e-12;")
    field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {w0:.6g}", inlet_code("w", False),
          "        type            inletOutlet;\n        inletValue      uniform 1;\n        value           uniform 1;",
          "        type            omegaWallFunction;\n        value           uniform 1;")
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0", "        type            calculated;\n        value           uniform 0;",
          "        type            calculated;\n        value           uniform 0;",
          "        type            nutLowReWallFunction;\n        value           uniform 0;")
    sch = "bounded Gauss upwind" if a.scheme == "upwind" else "bounded Gauss linearUpwind grad(U)"
    schk = "bounded Gauss upwind" if a.scheme == "upwind" else "bounded Gauss limitedLinear 1"
    write(f"{a.out}/system/fvSchemes", "dictionary", "system", "fvSchemes", f"""ddtSchemes {{ default steadyState; }}
gradSchemes {{ default Gauss linear; }}
divSchemes
{{
    default         none;
    div(phi,U)      {sch};
    div(phi,k)      {schk};
    div(phi,omega)  {schk};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}
""")
    write(f"{a.out}/system/fvSolution", "dictionary", "system", "fvSolution", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl { p 1e-7; U 1e-8; "(k|omega)" 1e-8; }
}
relaxationFactors
{
    equations { U 0.7; ".*" 0.6; }
    fields { p 1; }
}
""")
    write(f"{a.out}/system/controlDict", "dictionary", "system", "controlDict", f"""application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {a.iters};
deltaT          1;
writeControl    timeStep;
writeInterval   1000;
purgeWrite      2;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;

functions
{{
    pIn
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   50;
        log             yes;
        writeFields     no;
        regionType      patch;
        name            inlet;
        operation       areaAverage;
        fields          (p);
    }}
    yPlus1
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
}}
""")
    with open(f"{a.out}/case_meta.txt", "w") as f:
        f.write(f"npz {os.path.abspath(a.npz)}\nrefine {m}\nscheme {a.scheme}\ncells {nr * nz * m * m}\n")


if __name__ == "__main__":
    main()

# watercap_disc_cell: one periodic disc period of a disc-stack tube (OpenFOAM 10)

Cross-check for StoneLab's resolved cell flow (`stonelab/ehd_flow.py`, `solve_cell_flow`):
the same periodic cell, solved with stock OpenFOAM 10 (docker image `vega-openfoam:latest`).

## Geometry and physics

- Grounded tube of radius R; one thin coaxial disc (radius rd, thickness t) centred in a cell of
  length s (the disc spacing); cyclic top/bottom. No mount rod (StoneLab's `DiscStackSection` has none).
- Disc edge is square here; StoneLab rounds it (radius t/2). t is 0.37-0.52 mm, so this only
  affects the first few cells around the edge.
- Air with StoneLab's properties: rho = 1.15 kg/m^3, mu = 1.81e-5 Pa s.
- `meanVelocityForce` (fvConstraints) holds the flux at U pi R^2. Its volume-average target is
  corrected for the disc volume. The driving gradient is printed every iteration in the solver log
  (`pressure gradient = ...`, kinematic: multiply by rho to get Pa/m).
- Walls (tube, disc): no-slip, k-omega SST with `nutUSpaldingWallFunction` and `omegaWallFunction`.
  The first cell is 0.05 mm (disc) and 0.1 mm (tube) at level 1, which is y+ of about 0.3-1. Both
  sizes halve at level 2.
- Mesh: blockMesh, five blocks graded toward the disc faces, the disc edge and the tube wall.
  Default is a 5-degree axisymmetric wedge. `--sector DEG --ntheta N` makes a 3-D sector with
  rotational cyclics, for DES/LES.
- Geometry comes from `designs_params.json`, which was extracted from `Vega/watercap_web/designs.json`
  with StoneLab's `DiscStackSection` conventions: `cold_spray-0126` and `ring_spray-0017`.

## Running

    ./Allrun CASEDIR --design cold_spray-0126 --level 1                 # steady simpleFoam, SST
    NP=2 ./Allrun CASEDIR --design cold_spray-0126 --level 2 --np 2 --urelax 0.7 --trelax 0.5
    NP=4 ./Allrun CASEDIR --design cold_spray-0126 --solver pimpleFoam --endTime 1 --np 4   # URANS

`gen_case.py` writes the case (`--help` lists the options). `Allrun` runs it in Docker.
`Allrun.foam` runs inside an OpenFOAM-10 shell. It also writes the cell centres (C) and volumes (V)
at the latest time, for post-processing. An existing CASEDIR is continued from its latest time.

Post-processing, mapping onto StoneLab's grid and the mist-collection rerun are in the
StoneLab spike scratchpad (`ofcell/`: `ofread.py`, `compare.py`, `collect.py`).

## Results (2026-10-22)

Run directories, logs, mapped fields and figures are in the session scratchpad
`/tmp/claude-1001/-home-sl8rv-Projects/afc1651d-031a-4262-a8a0-20cc3523ef36/scratchpad/ofcell/`.
They are not kept here.

| Run | Cells | Wall clock | G (Pa/m) | Notes |
|---|---|---|---|---|
| cold_spray-0126 SST M1, steady | 27k | ~33 min, 1 core, 10k iterations | 1.087 | converged to 0.1%; y+ 0.5 on the tube, <0.93 on the disc |
| cold_spray-0126 SST M2, steady | 109k | 68 min, 2 cores, 3k iterations from M1 | 1.111 | M1 to M2: G +2.2%, same reversed-flow area and profiles |
| cold_spray-0126 SST URANS M1, maxCo 0.8 | 27k | 91 min, 1 core, t = 0.1 to 0.2 s | 1.076 to 1.083 | **steady**: probe std below 1e-3 m/s |
| cold_spray-0126 SST URANS M1, maxCo 3 | 27k | 109 min, 2 cores, 0.5 s | (2.3) | became unsteady at t = 0.118 s, with ±40 Pa/m gradient noise; rerunning at maxCo 0.8 stays steady, so this is **numerical**; discard |
| cold_spray-0126 3-D 60-degree sector k-omega SST DES | 393k | 2.8 h, 5 then 8 cores, 0.2 s | 0.88 (mean over 0.1-0.2 s) | **unsteady**: ~60 Hz rim-shear-layer/cavity oscillation; G swings -1.5 to 2; mean not converged (±25%) |
| ring_spray-0017 SST M1, steady | 27k | ~93 min, 1 core, 25k iterations | 0.894 | still creeping (+0.1% per 1k iterations) |
| ring_spray-0017 SST M2, steady | 109k | 75 min, 2 cores, 4k iterations from M1 | 0.846 | flat for the last 2.5k iterations |

The disc wake on ring_spray-0017 spans the whole 21 cm period, with reversed flow on the axis
everywhere. On cold_spray-0126 the core r < rd recirculates from one disc to the next. The
DES time-mean differs from the steady SST field by 4% rms of U.

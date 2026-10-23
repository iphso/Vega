# watercap_spray_gas: the spray stage's gas flow in OpenFOAM 10, with the spray force frozen

A cross-check of StoneLab `spray_cfd`'s turbulence models (`CFDConfig(turbulence="mixing_length" | "sst")`)
on the flow a spray actually drives.

- **Input:** an `.npz` holding the spray_cfd grid (`rf`, `zf`), the converged spray force per cell (`Fz`, `Fr`, N),
  `R`, `U`, `rho`, `L` and the developed SST inlet (`u_in`, `k_in`, `w_in`). The StoneLab spike scratchpad's
  `spsst/frozen.py` writes one per design, together with StoneLab's own mixing-length and SST solutions of the same
  frozen-force problem.
- **Mesh:** a 5-degree wedge, with one blockMesh block per spray_cfd radial row and each spray_cfd cell split
  `--refine` times in r and z. The force density is constant over each spray_cfd cell and is read by a coded fvModel
  from `constant/sprayForce.dat`, so the total force maps exactly.
  - Cell and face radii are taken at the wedge midplane (x / cos 2.5°), because the wedge faces are chords and the
    centroids sit about 1.4e-4 m inside the circle, which is more than the 6e-5 m wall cell.
- **Boundary conditions:**
  - Inlet: StoneLab's developed SST profile, coded by radius.
  - Outlet: p = 0, zero gradient.
  - Wall: no slip, low-Re (k = 0, omegaWallFunction on a y+ < 1 mesh, nutLowReWallFunction).
- **Solver:** simpleFoam with kOmegaSST, and no buoyancy.

Running:

    ./Allrun runs/cold_m1u --npz frozen_cold_spray-0126.npz --refine 1 --scheme upwind --iters 8000
    ./Allrun runs/cold_m3l --npz frozen_cold_spray-0126.npz --refine 3 --scheme linearUpwind --iters 15000

Results (2026-10-23) are in StoneLab `docs/realism_roadmap.md`, under "Spray stage: k-ω SST in place of the mixing length".
M1 (5k cells) takes about 10 minutes and M3 (48k cells) about 1 hour, on one core under load.

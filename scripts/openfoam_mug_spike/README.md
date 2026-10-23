# Mug OpenFOAM spike (EXPERIMENT_LOG §64)

Direct user course-reversal: `thermal_mug_spike.py`'s self-written finite-
difference solver was ruled out going forward -- "I don't want anything
self-written... we've got to reverse course on that in favor of something
that's more broadly accepted." This is the validation spike for OpenFOAM as
the replacement, at the agreed scope: solid conduction only (wall/handle),
same empirical convective-coefficient/radiative boundary conditions the
current physics already uses -- not full conjugate heat transfer (resolved
liquid/air flow), which stays a future option using the same tool.

## What's here

- `slab_case/`: a minimal `chtMultiRegionFoam` case (OpenFOAM v10, ONE
  region, `type solid`, no fluid region at all) -- a 3mm-thick plane slab
  (polypropylene), convective boundary conditions on both faces
  (`externalWallHeatFluxTemperature`, h=300/Ta=90C liquid-side,
  h=15/Ta=20C ambient-side), transient, 2 hours. The plane-slab reduction
  mirrors `thermal_mug_spike.py`'s own thin-wall approximation (constant
  area, no curvature term) -- deliberately the simplest real geometry that
  still exercises genuine through-thickness conduction, not a toy 0D case.
- `reference_1d.py`: an INDEPENDENT from-scratch 1D implicit finite-volume
  reference solver (does not import or reuse any of `thermal_mug_spike.py`'s
  code) for the identical problem -- the actual cross-check.

## Result

At matching time resolution (both dt=0.5s), OpenFOAM's real solver and the
independent reference agree to within numerical-discretization-difference
tolerance:

| quantity | reference (`reference_1d.py`) | OpenFOAM (`chtMultiRegionFoam`) |
|---|--:|--:|
| outer-surface T at 60s | 61.73 C | 61.66 C |
| outer-surface T at 2h | 75.89 C | 75.80 C |
| inner-surface T at 2h | 87.12 C | 87.21 C |

(OpenFOAM's own field files store Kelvin; subtract 273.15 for the table
above.) At dt=2s the match at 60s was looser (61.20 C) -- confirmed to be a
timestep-resolution artifact, not a structural discrepancy, by rerunning at
dt=0.5s and watching the gap close from 0.53C to 0.07C.

## Reproducing

```
docker run --rm -i -u $(id -u):$(id -g) -v "$PWD/slab_case":/work \
  openfoam/openfoam10-paraview510 << 'EOF'
source /opt/openfoam10/etc/bashrc
cd /work
blockMesh -region solid
chtMultiRegionFoam
EOF
python3 reference_1d.py
```

Notes for whoever automates this next (parametric geometry + material,
real dataset generation): the image's ENTRYPOINT does not forward `docker
run` CMD args to bash -- pipe commands via stdin (as above), not
`docker run image -c "..."`. Also needed running as `-u $(id -u):$(id -g)`
-- the image's default user can't write to a host-mounted case dir
otherwise.

## Open (not done in this spike)

- Only a flat slab, one material, fixed liquid temperature (no liquid
  energy balance), no handle, no radiation term, no automation. This
  proves OpenFOAM reproduces the right PHYSICS for this problem class; it
  does not yet reproduce the real mug's full parameterization.
- No Dockerfile/docker-compose integration yet -- run via a bare `docker
  run` against the public `openfoam/openfoam10-paraview510` image.
- No decision yet on the parametric meshing strategy for the real mug body
  (revolved-profile wall + handle) -- `blockMesh` (structured, fast, but
  needs a per-shape templated dict) vs `snappyHexMesh` (more general,
  handles the handle's 3D attachment more naturally, higher per-candidate
  meshing cost) is still open.

## `coupled_spike/` (EXPERIMENT_LOG §66)

Same slab/material/BCs as `slab_case/`, except the `liquid` patch uses the
compiled `liquidCoupling` plugin (`scripts/openfoam_libs/liquidCoupling/`)
instead of a fixed-Ta `externalWallHeatFluxTemperature` -- the liquid's own
lumped energy balance now evolves inline as the solver runs, instead of
being held at a fixed temperature for the full 2 hours. This is the
building block `mug_openfoam_oracle.py` needed to produce `temp_at_2h_C`
(this domain's own headline target) for real, not just `touch_temp_60s_C`.

`reference_1d_coupled.py` independently validates the exact coupling law
(same file/rest of methodology as `reference_1d.py`, extended with the
liquid ODE) -- see that script's own header comment for why using the
previous-timestep boundary FACE value (not a lumped cell-center
approximation) is provably the same flux a converged `mixed` BC computes,
so the two are a genuine cross-check, not coincidentally similar formulas.

Result (both dt=0.5s):

| quantity | reference (`reference_1d_coupled.py`) | OpenFOAM (`liquidCoupling`) |
|---|--:|--:|
| liquid T at 60s | 89.977 C | 89.976 C |
| liquid T at 2h | 89.499 C | 89.498 C |
| outer-surface T at 60s | 61.72 C | 61.65 C |
| outer-surface T at 2h | 75.49 C | 75.40 C |

Liquid-temperature agreement is near-exact (<0.001C); the outer-surface
residual (~0.07-0.1C) matches the same timestep-discretization artifact
already characterized in `slab_case/`'s own validation above, not a new
issue.

**Why a compiled plugin instead of `codedMixed`:** the original plan was a
`codedMixed` inline block running the ODE via OpenFOAM's own dynamicCode
runtime compilation. It's unconditionally refused when running as root
(`dynamicCode::checkSecurity()`'s `isAdministrator()` check has no
override -- confirmed directly against `$FOAM_SRC`, not assumed), which is
how this project's `openfoam` Docker service runs. A real `wmake libso`
plugin -- OpenFOAM's own standard extension mechanism -- sidesteps that
entirely and compiles once at image-build time instead of once per
candidate (see `Dockerfile.openfoam`).

**A real bug caught here, not glossed over:** the first working version of
`liquidCoupling` integrated heat loss over only the WEDGE patch's own
faces (a 2.5-degree-half-angle slice) while being handed the FULL mug's
liquid mass -- silently under-counting heat loss by ~72x (the full-circle-
to-wedge-angle ratio), so a real mug case's liquid barely cooled at all.
Caught by comparing `mug_openfoam_oracle.py`'s `temp_at_2h_C` against v6's
own and finding it implausible (liquid staying near 90C after 2 hours),
not by a crash. Fixed with an explicit `wedgeFraction` dictionary entry
the BC multiplies its own patch-local heat-loss sum by before integrating
the ODE -- see `liquidCouplingFvPatchScalarField.H`'s own comment on this.
This standalone `coupled_spike/` case is a flat slab (not a wedge), so it
never exercised that bug -- worth remembering when trusting a validation
case that doesn't share the real oracle's geometry.

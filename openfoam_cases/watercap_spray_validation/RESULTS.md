# Watercap spray-condenser: independent OpenFOAM-10 cross-check

Run date 2026-10-21. Solver: OpenFOAM Foundation v10 `reactingFoam`, using the
`clouds` fvModel (a two-way coupled `reactingCloud`), in Docker image
`vega-openfoam:latest`, which is the `openfoam/openfoam10-paraview510` base.
I set the case up from the physics only. Nothing was tuned toward the in-house
solver's numbers.

## 1. Is condensation modeled? Yes, but only after a non-default switch is turned on

Source: `$FOAM_SRC/lagrangian/parcel/submodels/Reacting/PhaseChangeModel/LiquidEvaporation/LiquidEvaporation.C`
in the container (`/opt/openfoam10/src/...`):

- Lines 74-81: `condensation_ = coeffDict().lookupOrDefault<Switch>("condensation", false)`.
  **The default is false.**
- Lines 210-217:
  ```
  // molar flux of vapour [kmol/m2/s]
  scalar Ni = kc*(Cs - Cinf);
  // limit if not permitting condensation
  if (!condensation_)
  {
      Ni = max(Ni, 0);
  }
  ```
  With the stock default, condensation is clipped to zero. A 20 C drop in
  saturated 35 C air would then exchange only sensible heat and never grow.
- The parcel side, `parcels/Templates/ReactingParcel/ReactingParcel.C`
  `calcPhaseChange`, about lines 107-121, passes a negative `dMassPC` through
  unchanged. `dMassPC = min(mass*YPhase*Y, dMassPC)` caps only the upper side,
  and `Sh -= dMassPC*dh/dt` then hands the latent heat of condensation to the
  drop. The early return `if (T < Tvap)` uses the triple-point temperature
  (273.16 K), so it does not apply here.
- `LiquidEvaporationBoil.C` cannot condense at all. Line 210 says
  `if (Xc*pc > pSat) { // saturated vapour - no phase change }`, and line 291
  says `if (Xr > 0)`. Do not use it for this case.

**This case uses `phaseChangeModel liquidEvaporation;` with `condensation true;`**
(set in `constant/cloudProperties`). The run log confirms that condensation
happens: "Mass transfer phase change" is negative and falls steadily
(cumulative -3.42e-3 kg by t = 6 s).

**I also ran the stock default (`condensation false`) as a check.** It used
the base mesh, ran 0-3 s and was averaged over 2-3 s, in 600 s of wall time
(in scratch, not kept here). Results:
- vapor_removed_frac = **0.0000**, and the phase-change mass stays exactly 0.
- Outlet mixing-cup T = **28.18 C**, at unchanged w = 0.0366 kg/kg. That is
  an outlet RH of about 147%, which is unphysical.
- Exiting drops are at 28.1 C. The spray splits 76% out of the top, 23% to
  the wall and 0% out of the bottom.
- Min mean-window Uz is -8.9 m/s.

This matches the hand estimate below. The stock default therefore gives zero
vapour removal and a gas that is only sensibly cooled.

The explanation: with condensation clipped, no vapour is removed by the
drops, so vapor_removed_frac = 0. The gas would
only cool sensibly by heating the spray, and the saturated inlet air would
become supersaturated, which a Lagrangian-only model cannot relieve because
there is no bulk-fog model. Sensible-only equilibrium, as a hand estimate:
- The gas heat capacity flow is m_da*(cp_a + w*cp_v), about 174 W/K.
- The spray's is m_s*cp_l, about 139 W/K.
- The common temperature is (174*35 + 139*20)/313 = **28.3 C**. The gas
  gives up about 0.93 kW of sensible heat.
- The humidity stays at w = 0.0366 kg/kg, so the outlet would be at RH of
  about 146%, which is unphysical.

So the stock model would report vapor_removed_frac = 0 and a gas cooling of
about 6.7 K. That under-predicts vapour removal entirely and over-predicts
the temperature drop, compared with the condensing result below (1.9 K).

## 2. Time-averaged results (window 4.0-6.0 s, base mesh 78k cells)

| quantity | inlet | outlet |
|---|---|---|
| gas mass flow [kg/s] | 0.168249 | 0.167619 |
| dry-air mass flow [kg/s] | 0.162309 | 0.162308 |
| H2O vapour mass flow [kg/s] | 5.9400e-3 | 5.3108e-3 |
| humidity ratio [kg/kg dry air] | 0.03660 | 0.03272 |
| mixing-cup T (phi-weighted) | 35.000 C | **33.09 C** |

- **vapor_removed_frac = (5.9400e-3 - 5.3108e-3)/5.9400e-3 = 0.1059**
- Independent check from the cloud: the condensation rate is 6.295e-4 kg/s,
  which is 0.1060 of the inlet vapour. This agrees with the outlet balance
  to 0.1%.
- The outlet is at equilibrium: RH = 1.000 at 33.09 C, and the exiting drops
  are at 33.07 C. My own adiabatic-equilibrium estimate, using the same
  OpenFOAM water vapour-pressure curve and constant cp and h_fg, is
  T_eq = 33.13 C and removed fraction 0.1044. The CFD result is about 1.5%
  above that. The difference is property-model detail (janaf cp, enthalpy
  difference) and the wall-lost drops, which leave hotter (33.8 C).
  **OpenFOAM reaches the adiabatic-mixing ceiling.**

Spray mass fate, as fractions of the injected 0.03333 kg/s. Exit masses
include condensate, so the fractions sum to slightly more than 1:

| fate | kg/s | fraction | mass-mean T at exit | mass-mean d (D43) | mean residence |
|---|---|---|---|---|---|
| out of the top (outlet) | 0.02617 | **0.785** | **33.07 C** | 62.5 um | 1.08 s |
| hits the wall (removed) | 0.00782 | 0.235 | 33.84 C | 75.2 um | 0.51 s |
| falls out of the bottom (inlet) | ~0 (1 parcel in 2 s) | 0.000 | - | - | - |

- Liquid hold-up in the domain is 0.032 kg (about 18,900 parcels) and stays
  constant over the window. The liquid balance closes:
  0.03333 + 0.00063 (condensed) = 0.03396 kg/s in, against 0.03399 kg/s out.
- Wall treatment: drops that touch the wall are removed (`localInteraction`,
  `escape`), that is, they become wall film or drainage. With `rebound`, most
  of that 23% would probably be re-entrained and leave through the top. This
  is a modeling choice, not a solver result.

## 3. Velocity field and pressure drop

- **Strong reverse flow below the nozzle.** The downward spray drives a gas
  jet against the 2.1 m/s upflow.
  - The time-mean axial velocity on the axis is negative from z = 0.63 to
    1.02 m. The minimum mean Uz is **-7.67 m/s on the axis at z = 0.95 m**,
    and the global minimum of UMean_z is **-7.95 m/s**.
  - The instantaneous minimum Uz over the window has mean -8.42 m/s and
    lowest -9.19 m/s.
  - The spray and jet stagnate at about z = 0.55-0.6 m, which is 0.4-0.45 m
    below the nozzle. The gas then turns and the drops are carried up the
    annulus. The peak mean Uz is +4.46 m/s near the wall at z of about 1 m.
- Mean gas state inside the spray cone (axis, z = 0.7-0.95 m): T = 24-27 C,
  Y_H2O = 0.018-0.022. This is locally saturated at the local temperature.
  Above z = 1.2 m the gas is 32.5-33 C and nearly uniform radially.
- Selected axis values (time-mean):

  | z [m] | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 1.0 | 1.05 | 1.2 | 1.5 | 2.0 | 2.45 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Uz [m/s] | 2.44 | 1.26 | -2.57 | -4.61 | -6.42 | -7.67 | -2.82 | 0.53 | 1.28 | 1.65 | 1.86 | 1.92 |
  | T [C] | 35.0 | 34.0 | 27.1 | 26.0 | 24.5 | 23.8 | 29.1 | 32.1 | 32.5 | 32.7 | 32.9 | 33.0 |

- Pressure is the area average over each patch, and p includes hydrostatics
  (the `buoyancyForce` fvModel acts on the full p).
  - The inlet minus outlet difference is 38.3 Pa.
  - The gas hydrostatic part is about 27.5 Pa (rho_in*g*L).
  - That leaves **about 10.8 Pa** of non-hydrostatic pressure drop, mostly
    from the spray. The suspended liquid weighs about 4.5 Pa over the tube
    area, and the downward injection momentum adds about 4.7 Pa. Wall
    friction is under 1 Pa.
  - Before the spray starts (t < 0.5 s), the difference is 28.0 Pa, which is
    essentially hydrostatic.

## 4. Numerical setup, cost and caveats

- **Geometry and mesh:** a 3D O-grid cylinder from blockMesh (central square
  plus 4 ring blocks). R = 0.15 m, L = 2.5 m, axis +z, g = (0 0 -9.81).
  - 12x12 cells in the central square, 10 radial cells in the ring graded
    toward the wall, 125 axial cells (2 cm): **78,000 hex cells**.
  - Max non-orthogonality 27 deg, max aspect ratio 7.5. Wall functions are
    used; the estimated wall y+ is about 15-20.
- **Why 3D instead of the requested 2D wedge:** OpenFOAM-10 Lagrangian
  tracking on a wedge is unsafe for a point full-cone injector on the axis.
  - `InjectionModel.C` (about lines 467 and 569) snaps injection positions
    to the wedge mid-plane with `constrainToMeshCentre`.
  - `particleTemplates.C:224` `hitWedgePatch` is a FatalError ("Hitting a
    wedge patch should not be possible").
  - A cone sampled in full azimuth has velocity components out of the wedge
    plane and toward negative x, across the degenerate axis.
  - The 3D mesh avoids all of this. It also represents the full cone and the
    stochastic turbulent dispersion in three dimensions, and it was cheap
    enough (42 min on 2 cores).
- **Gas side:**
  - reactingFoam, transient PIMPLE (1 outer and 2 pressure correctors), with
    heRhoThermo, a multicomponent air + H2O mixture, janaf/sutherland
    properties and a perfect gas.
  - Turbulence model k-omega SST.
  - Schemes are linearUpwind for U, limitedLinear for h and Yi, upwind for
    k and omega, and Euler in time.
- **Inlet:**
  - `flowRateInletVelocity`, volumetricFlowRate 0.15 m^3/s, with the
    **`turbulentBL` profile (1/7 power law in wall distance)**. The mean
    velocity is 2.128 m/s because the faceted inlet area is 0.07048 m^2.
  - Turbulence: k from 5% intensity (`turbulentIntensityKineticEnergyInlet`),
    omega from mixing length 0.021 m (0.07 D).
  - T = 308.15 K and Y_H2O = 0.035305 (w = 0.0366). This is RH 100% using
    OpenFOAM's own water p_v(35 C) = 5630 Pa.
- **Walls and outlet:** walls are no-slip with zeroGradient T (adiabatic).
  The outlet has p = 101325 Pa fixed, `pressureInletOutletVelocity` for U,
  and `inletOutlet` for the scalars.
- **Spray:**
  - `coneInjection` at (0 0 1.0), direction (0 0 -1), thetaInner 0,
    thetaOuter 30 deg, Umag 10 m/s, T0 293.15 K.
  - Mass flow 0.03333 kg/s from t = 0.5 s, with 20,000 parcels/s of equal
    mass (`parcelBasisType mass`).
  - Size distribution: a `general` tabulated pdf, **lognormal by mass,
    MMD 60 um, GSD 1.5, truncated at +/-3 sigma (17.8-202.5 um)**, 41 points.
    Because every parcel has the same mass, the sampled parcel-diameter pdf
    is the mass pdf. The check gives sampled MMD 60.0 um and D43 65 um.
  - Physics models: sphereDrag plus gravity, `stochasticDispersionRAS`, and
    RanzMarshall heat transfer. Mass transfer uses the Ranz-Marshall Sherwood
    number inside liquidEvaporation, with `enthalpyTransfer
    enthalpyDifference`.
  - All source terms are two-way coupled: rho, U, Yi and h, explicit.
  - No collisions, film or charge.
- **Time stepping:** adaptive, maxCo 0.8, capped at dt = 1e-3 s. Over the
  averaging window dt = 1e-3 s throughout, and the max Courant number is
  about 0.46. The cloud's own sub-cycling uses maxCo 0.3.
- **Run:**
  - 0-6 s of physical time; the gas runs alone until the spray starts at
    0.5 s.
  - 6008 steps, **2516 s wall time on 2 MPI ranks** (scotch decomposition).
  - Averages are over the last 2 s (4-6 s) with the `fieldAverage` function
    object and time-integrated patch-flux series.
- **Steadiness:**
  - The outlet vapour removal already reads 0.1061 over 2.5-3.06 s, against
    0.1059 over 4-6 s.
  - The parcel count and liquid hold-up are flat after about 2 s.
  - The time-step continuity error is about 1e-10 per step (cumulative
    -5e-6).
- **Caveats:**
  - The near-nozzle jet is resolved only by about 1 cm cells, so the peak
    reverse velocity depends on the mesh. See the mesh check below.
  - There is no droplet collision or coalescence and no wall film. Wall hits
    are removed.
  - There is no homogeneous (bulk-fog) condensation. It was hardly needed:
    the maximum local RH at t = 6 s is 1.014 instantaneous and 1.009
    time-mean, and fewer than 0.01% of cells are above 1.01.
  - A point injector puts the full spray momentum into one cell.

### Mesh sensitivity
A finer mesh was run as a check (run in scratch, not kept in this directory):
16x16 central square, 13 radial cells, 170 axial cells, **184,960 cells**
(2.4x the base mesh). It ran 0-4 s, averaged over 3-4 s, and took 3637 s
wall time on 2 ranks.

| quantity | base 78k (avg 4-6 s) | fine 185k (avg 3-4 s) |
|---|---|---|
| vapor_removed_frac (outlet) | 0.1059 | 0.1055 |
| condensation / inlet vapour (cloud) | 0.1060 | 0.1063 |
| outlet mixing-cup T | 33.09 C | 33.10 C |
| spray out of the top / wall / bottom | 0.785 / 0.235 / 0 | 0.773 / 0.243 / 0 |
| exiting drop T (top) | 33.07 C | 33.08 C |
| min mean Uz on axis (location) | -7.67 m/s (z = 0.95 m) | -7.67 m/s (z = 0.96 m) |
| global min UMean_z | -7.95 m/s | -7.94 m/s |
| reverse-flow extent on axis | z = 0.63-1.02 m | z = 0.64-1.02 m |
| inlet minus outlet p (incl. hydrostatic) | 38.3 Pa | 38.2 Pa |

All the headline numbers are mesh-independent to within about 1-2%.

## 5. Rerunning the case

The case directory is `/home/sl8rv/Projects/Vega/openfoam_cases/watercap_spray_validation/`.

- `./Allrun` runs the whole case from the host. It uses Docker,
  `vega-openfoam:latest` and 2 ranks, then runs `python3 postprocess.py` and
  writes `results_summary.txt`.
- `Allrun.foam` is the in-container driver: blockMesh, checkMesh,
  decomposePar, reactingFoam, reconstructPar, then line sampling of the
  time-averaged fields.
- `./Allclean` removes the outputs.
- `postprocess.py [t0] [t1]` prints all of the numbers above from
  `postProcessing/` and `log.reactingFoam`.
- Kept outputs:
  - time directories 0.5 through 6, with UMean, TMean, H2OMean, pMean and
    rhoMean at t = 6;
  - `postProcessing/` (patch-flux time series, min/max, axis and radial
    lines, and parcel exit records under `lagrangian/`);
  - all logs.

## Comparison with the in-house solver (for context only; not used for setup)

| quantity | in-house | OpenFOAM-10 (this run) |
|---|---|---|
| vapor_removed_frac | ~0.10 (ceiling 0.104) | 0.106 (equilibrium; my ceiling estimate 0.104) |
| outlet T | 33-34 C | 33.09 C |
| spray out of the top | nearly all | 78.5% (23.5% hits the wall, 0% falls out of the bottom) |
| min axial velocity | ~ -4.5 m/s | -7.7 m/s time-mean (-8.4 instantaneous mean, -9.2 extreme) |

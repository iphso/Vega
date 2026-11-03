# FNS Li2O and Be cylindrical assemblies: OpenMC / ENDF/B-VIII.0 benchmarks + perturbed-library runs

Date: 2026-11-03. Purpose: experimental C/E for the blanket simulator's breeder/multiplier data, and
benchmark responses on the 30 joint ND samples of the ARC UQ campaign (`blanket/uq`, set `all`) so they
can be paired with the same samples' ARC TBR for Bayesian data assimilation.

Data source: FENDL-2 benchmark compilation JAERI-M 94-014 (`blanket/refs/lit/jaerim94014/benchmarks/
jaerim94014/{li2o,be}`), cross-checked against JAERI-M 86-182 (Li2O original report, read in full by the
literature pass). Published C/E: `blanket/refs/lit/fns/` (see "Comparison" below).

## Files
- `fnsdata.py`: transcribed experimental data and uncertainty splits (with table references)
- `fnslib.py`: model (geometry, materials, FNS source, tallies), result extraction
- `responses.py`: response definitions (E, uncertainty split, how C is computed); NE213 band integrals
- `make_xs.py`: Be-metal S(a,b) (NJOY from the ENDF/B-VIII.0 TSL file) + Np-237; library XML
- `run_bench.py`: one model run -> JSON of per-bin rates; `sample_runs.py`: perturbed-library campaign
- `analyze.py`: builds `results.json` and the tables below (host python, numpy only)
- Big outputs (not in git): `~/runs/blanket/bench/fns/{nominal,samples,xs,xs_sab}`

Reproduce (from `~/Projects/Vega`; `S` = `docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp
-e OMP_NUM_THREADS=12 -v ~/runs/blanket/bench/fns:/fnsruns -v ~/runs/blanket/uq:/runs openmc`):
```
$S blanket/bench/fns/make_xs.py                     # needs /fnsruns/dl/tsl-Be-metal.endf (NNDC thermal zip)
$S blanket/bench/fns/run_bench.py li2o --fine-faces --particles 100000000 --batches 50 --seed 1 --out /fnsruns/nominal/li2o_noroom_ff.json
$S blanket/bench/fns/run_bench.py li2o --particles 100000000 --batches 50 --seed 1 --out /fnsruns/nominal/li2o_noroom.json
$S blanket/bench/fns/run_bench.py li2o --kin --particles 50000000 --batches 50 --seed 3 --out /fnsruns/nominal/li2o_noroom_kin.json
$S blanket/bench/fns/run_bench.py li2o --full-sphere --particles 20000000 --batches 40 --seed 11 --out /fnsruns/nominal/li2o_pair_noroom.json
$S blanket/bench/fns/run_bench.py li2o --room --void-room --particles 20000000 --batches 40 --seed 11 --out /fnsruns/nominal/li2o_pair_room.json
$S blanket/bench/fns/run_bench.py li2o --room --particles 30000000 --batches 50 --seed 2 --out /fnsruns/nominal/li2o_room.json
   (same six for be with 20M/20M/10M/5M/5M/10M histories)
$S blanket/bench/fns/sample_runs.py --start -1 --n 30 --particles 30000000 6000000
python3 blanket/bench/fns/analyze.py
```

## Models

### Common
- **Code/data**: OpenMC 0.16.0, ENDF/B-VIII.0 (NNDC HDF5, `blanket/xs`, 294 K). Be-metal S(a,b) from
  the ENDF/B-VIII.0 thermal sublibrary (tsl-Be-metal) processed with NJOY 2016 at 296 K (the
  openmc-data-storage VIII.0 mirror lists c_Be.h5 but the file 404s). Np-237 HDF5 fetched separately.
  Neutron transport only.
- **Detector responses** are computed as R(z) = ∫ φ(E,z) σ_det(E) dE per detector atom per source
  neutron: a track-length flux tally on thin on-axis cylinders (r < 1.5 cm) times an
  EnergyFunctionFilter holding the pointwise 294 K detector cross section **read from the same HDF5
  library the transport uses** (so a perturbed library perturbs both). This is the infinitely-dilute
  response, i.e. what the experimenters report after their self-shielding/edge corrections. Bins are
  0.5 cm, 1 mm within 1 cm of both faces; positions are linearly interpolated between bin centres and
  never across a face. T6 = Li-6 MT205 (n,Xt), T7 = Li-7 MT205, fission MT18, foils MT107/16/103/102.
- **Normalisation**: per D-T reaction (= per "source neutron" of the associated-alpha monitor).
  OpenMC ≥ 0.15 already scales tallies by total source strength.

### Source (the assumption with the largest a-priori leverage, measured to be small)
The compilation tabulates **only the 0-degree spectrum** (Table 7.21 / t8-16, 125 groups,
neutrons/bin/D-T reaction per 4π sr) and states that 0-degree emission is 1.1767 (Li2O, old target)
or 1.1262 (Be, new target) per D-T reaction; the tables sum to exactly those numbers. Its own MCNP
input (f8-8.txt) uses this spectrum as a point source with weight 1.1261 and SDIR direction biasing
into the forward cone. Baseline here = that prescription: the 0-degree spectrum, uniform in
μ ∈ [0.52, 1] around the axis (cone covering the whole front face, μ = 0.536), strength
S0 (1-0.52)/2 per D-T. No neutron outside the cone can reach the assembly without a room.
Variant `--kin`: 8 μ-bins whose D-T peak (above 13.5 MeV, blended from 12 MeV) is shifted by
non-relativistic T(d,n) kinematics (effective E_d fitted so that E_n(0°) matches the tabulated peak
mean) and whose intensity follows the CM→lab Jacobian normalised to S0 at 0°. This lowers the
energy and intensity off-axis, the physically expected direction. **Effect: |ΔC| ≤ 1-2% almost
everywhere, ≤ 0.6% on every integrated response**; it is carried as part of `geom_rel_unc`.
(JAERI-M 86-182's DOT analysis used the 0° spectrum as an isotropic source with integral 1.173; on a
point source on axis that is equivalent to our baseline inside the cone.)

### Li2O assembly (Table 7.1, JAERI-M 86-182 §2)
Equivalent cylinder r = 31.5 cm, **length 61.0 cm** (stated; 12 blocks × 5.06 cm = 60.72 cm), front
face 20 cm from the source. Three regions with the homogenised densities of Table 7.1: inner Li2O
(drawer contents, r < 2.719), SS304 drawer (2.719-2.866), outer Li2O. Aluminium frame omitted
(JAERI-M 86-182 App. A.4: effect "very small"). Off-axis Li-glass (r = 15.2 cm) tallied on a ring
14.7 < r < 15.7.

### Be assembly (JAERI MCNP-3 input f8-8.txt)
Cylinder r = 31.5 cm, 45.54 cm long (z = 20.0-65.54 cm), Be blocks 1.22149e-1 atoms/b-cm with the
MCNP input's impurities (C 7.7e-5, O 5.0e-4, Al 2.9e-4, Fe 2.5e-5; **no B, Li, Cd**), Be-metal
S(a,b); centre drawer = square SS tube, |x|,|y| in [2.46, 2.54] cm, 2.009e-2 atoms/b-cm, Be inside.

### Room return (approximate room model)
Target room per the Li2O text (both experiments used the same 80° beam line / first target room):
15 m × 15 m, target 5.5 m from the west and south walls, 7.9 m below the ceiling, 3.8 m above the
basement floor. Modelled as a concrete box (NBS-03 ordinary concrete, 2.35 g/cc, 2 m walls) with the
assembly pointing away from the near walls (assumption: Fig. 7.3 is not in the compilation). Steel
grating floor (1.8 m below), target hardware, beam line and assembly stand omitted.
The room factor f_room = C_room/C_noroom comes from a **correlated pair**: both runs at the same seed,
with the same full-sphere source, and the room interior void, so histories coincide inside the
assembly until they leak and the ratio isolates the returned neutrons (an independent room run with
air agrees within its noise). Result:
- Threshold responses (T7, NE213, fission of U-238/Th-232/Np-237, all foils except Au, flux > 10 MeV):
  f_room = 1.000 to ≤ 0.1%.
- T6 / U-235 / Au in the interior of both assemblies: consistent with 1 within ±1-2% (Li2O 22-75 cm,
  Be everywhere). The experimenters' own Li-pellet room-return estimate (Table 7.2) is 0.6-1.8% in the
  interior, 3.3% at 70.7 cm and 6.5% at 75.7 cm, i.e. within our noise.
- Significant only at the Li2O front face: Li-glass at 20.25 cm (2.5 mm inside the Li2O) ×1.30 on
  axis, ×1.70 at r = 15.2 cm. The front-face Li-pellet point (19.9 cm, i.e. in the air gap in our
  geometry) would see ×2.6 if it sat in the open, whereas the experimenters estimated only 0.6%. So
  the pellet was effectively shielded by Li2O, and its position is really "at the face".
**Decision**: C = no-room C × f_room only where f_room is resolved (> 3σ; that is the two front-face
Li-glass points); elsewhere C = no-room C and |f_room − 1| + σ goes into `geom_rel_unc`. The
Li-pellet T6 is compared with Table 7.2's "corrected for self-shielding **and room return**" column
(as JAERI and the FENDL-1 analyses did); the 80.8 cm pellet point (room return "over 200%") has no
corrected value and is excluded. All other data are uncorrected and compared with C as defined.

## Experimental uncertainties (systematic / random split)
`E_corr_group` names the shared systematic; `exp_correlations` in results.json is rebuilt from the
split (full correlation of the systematic part within a group, random parts uncorrelated, integral
responses propagated exactly from their points). Splits:
- Li-pellet LSC (T6, T7): systematic 1.81% = yield 1.5 ⊕ efficiency 1.0 ⊕ Li atoms 0.1 (Table 7.3);
  the rest of the quoted error is random.
- Li-glass in Li2O: 5.4% systematic (Table 7.4 column); the rest (fitting, self-shielding, counting)
  random. Excluded: the 80.70 cm value 8.13e-29, 10× the neighbouring points (a typo for 8.13e-30?).
- NE213 T7 and spectra: 2.34% alpha-monitor calibration ⊕ 1% efficiency common; alpha statistics and
  unfolding random. The source-intensity column of Table 7.6 is exactly √(alpha stat² + 2.34²), which
  confirms the split. The band-integral error (> 10 MeV) is not given: 3% random assumed.
- Fission chambers (Li2O): 90% of the smallest error in each series common (assumption). Be U-235:
  3.2% common (yield ~2 ⊕ deposit ~2 ⊕ efficiency ~1.5, assumption).
- Be Li-glass: 1.96% common (yield 1.9 ⊕ ⁶Li atoms 0.5, from Table 7.5), rest random.
- Foils: 2.5% efficiency common. Note: the quoted foil errors (≥ 2.7%) are smaller than efficiency
  2.5 ⊕ yield 3.0, so the 3% neutron-yield error is evidently **not included** in them (the graphite
  chapter says so explicitly). It is not added here; a user may want to add it as a common term.

## Results

C/E summary (MC 1σ ≤ 1% unless noted; full per-response table below):

<!-- AUTO:SUMMARY -->
| case | series | n | positions cm | C/E range | C/E integral or mean | E unc % |
|---|---|---|---|---|---|---|
| li2o | t6_lsc | 14 | 19.9-75.7 | 0.867-1.028 | 0.975 | 4.5-15.7 |
| li2o | t7_lsc | 8 | 19.9-45.3 | 0.963-1.716 | 1.141 | 4.2-28.1 |
| li2o | t6_lg_ax | 10 | 20.2-75.7 | 0.963-1.330 | 1.138 | 6.8-11.0 |
| li2o | t6_lg_r15 | 5 | 20.2-70.6 | 0.914-1.404 | 1.096 | 6.8-9.5 |
| li2o | t7_ne213 | 7 | 21.6-82.1 | 1.083-1.157 | 1.111 | 3.0-4.9 |
| li2o | fc_u235_f | 13 | 23.9-77.2 | 0.884-0.990 | 0.959 | 4.1-4.2 |
| li2o | fc_u238_f | 13 | 23.9-77.2 | 0.985-1.045 | 1.012 | 3.9-4.2 |
| li2o | fc_np237_f | 13 | 23.9-77.2 | 0.837-0.979 | 0.937 | 5.1-5.3 |
| li2o | fc_th232_f | 12 | 23.9-77.2 | 0.846-1.034 | 0.997 | 4.7-6.4 |
| li2o | foil_al27_na | 14 | 19.9-75.8 | 1.004-1.124 | 1.049 | 3.1-3.8 |
| li2o | foil_ni58_n2n | 14 | 19.9-75.7 | 0.898-1.080 | 0.972 | 2.8-10.3 |
| li2o | T6_lsc_int | 1 | integral | 0.990-0.990 | 0.990 | 2.3-2.3 |
| li2o | T7_lsc_int | 1 | integral | 1.062-1.062 | 1.062 | 3.2-3.2 |
| li2o | T6_lg_int | 1 | integral | 1.143-1.143 | 1.143 | 5.9-5.9 |
| li2o | T7_ne213_int | 1 | integral | 1.139-1.139 | 1.139 | 3.0-3.0 |
| li2o | phi_gt10_ne213 | 6 | 21.6-72.0 | 1.007-1.101 | 1.041 | 3.9-3.9 |
| be | t6_lg | 9 | 20.1-63.1 | 1.052-1.257 | 1.172 | 2.4-3.2 |
| be | fc_u235_f | 10 | 24.1-62.2 | 1.196-1.312 | 1.277 | 3.8-3.8 |
| be | foil_au197_ng | 6 | 20.0-65.6 | 0.842-1.172 | 1.062 | 2.7-3.0 |
| be | foil_al27_na | 6 | 20.0-65.6 | 0.937-1.089 | 1.026 | 2.7-3.7 |
| be | foil_ni58_n2n | 6 | 20.0-65.6 | 0.650-0.982 | 0.872 | 2.7-9.0 |
| be | foil_fe56_np | 6 | 20.0-65.6 | 0.989-1.078 | 1.047 | 2.7-3.7 |
| be | foil_zr90_n2n | 6 | 20.0-65.6 | 0.810-1.013 | 0.949 | 2.7-4.1 |
| be | foil_ni58_np | 6 | 20.0-65.6 | 1.006-1.197 | 1.102 | 2.7-4.1 |
| be | T6_lg_int | 1 | integral | 1.196-1.196 | 1.196 | 2.1-2.1 |
| be | U235f_int | 1 | integral | 1.282-1.282 | 1.282 | 3.3-3.3 |
| be | phi_gt10_ne213 | 8 | 21.4-62.0 | 0.852-0.949 | 0.897 | 3.9-3.9 |
<!-- /AUTO:SUMMARY -->

### Comparison with published C/E (model check, not a target)
Literature pass: `blanket/refs/lit/fns/` (FZKA 5785 Fischer 1996, FENDL-3 CRP Appendix B,
Konno 2011 PNST 2, Schnabel et al. 2023 arXiv:2311.10063, Kondo 2011 PNST 1). JAERI-M 86-182 was
read but could not be saved (bot wall).

**Li2O.** MCNP JENDL-3.2 / FENDL-1 (FZKA 5785 Table 17, against the room-corrected pellet data):
- **T6 pellets**: 0.87-0.90 at the face, 1.00-1.06 at 2-35 cm depth, 0.93-0.99 at 40-56 cm.
  **Ours (ENDF/B-VIII.0)**: 0.87 at the face, 0.99-1.03 at 2-35 cm, 0.91-1.01 at 40-56 cm.
- **T7 pellets**: 0.93 at the face, 0.98-1.08 in the front 15 cm, 1.74/1.26 at the two deep points
  (24% and 28% experimental error). **Ours**: 0.96, 0.98-1.07, **1.72/1.24**.
- **T7 by NE213**: 1.08-1.18, which FZKA 5785 attributes entirely to the JENDL-3PR1 ⁷Li(n,n't) cross
  section that the experimenters folded into the "measurement". **Ours**: 1.08-1.16. The same caveat
  applies, so treat these as a cross-section-dependent derived quantity.
- **Al(n,α)**: 1.01-1.09 (published) vs **1.02-1.06** (ours). **U-235(n,f)**: 0.94-1.04 falling to 0.94
  deep (published) vs **0.98 falling to 0.89 at 77 cm** (ours). Ni-58(n,2n), U-238 and Th-232 also
  fall in the published ranges.

Everything matches the published re-analyses to within a few %, including the odd T7 pellet point at
40.2 cm. The Li2O model is therefore not the source of any of the deviations listed below.

**Be.** Every modern analysis overpredicts the slow-neutron responses:
- **Published**, Li-glass T6 C/E: about 0.97 at the face, 1.12-1.31 inside (Schnabel 2023, MCNP6 +
  ENDF/B-VIII.0 + VIII.0 Be S(a,b): 1.12-1.15 at 5 cm, 1.21-1.26 at 10-13 cm, 1.26-1.31 at 28-30 cm,
  1.17-1.20 at 40 cm); U-235 fission 1.24-1.33; JENDL-4: T6 1.13-1.36, U-235 1.26-1.38.
  **Ours**: T6 1.12-1.26 inside, U-235 1.21-1.33, i.e. inside the ENDF/B-VIII.0 band. Our face point
  is 1.12; the published 0.95-0.99 there probably reflects their tally volume (the JAERI MCNP input's
  first Be cell is 1 cm thick) and the very steep surface gradient. We use 1 mm bins.
- Au(n,γ) inside: 1.10-1.19 (Schnabel) vs ours 1.1-1.2 (no foil self-shielding modelled).
- Threshold foils: Al, Zr(n,2n) and Fe(n,p) are 0.94-1.08, Ni(n,p) rises to 1.1-1.2 at depth, and
  Ni(n,2n) is 0.87-0.97 (0.65 ± 0.11 MC at the back face).

**Our flux above 10 MeV (NE213) is 0.85-0.95 of the measurement.** Maekawa et al. 1991 found the
same for ENDF/B-IV and LANL Be data (ENDF/B-VIII.0 Be-9 descends from LANL), whereas JENDL-3 agreed.
This is real data information about Be-9 elastic plus (n,2n) removal of 14 MeV neutrons, and so it
is relevant to the multiplier.

### Why Be overpredicts the slow responses (our reading, not adjusted for)
The 20-30% T6/U-235 overprediction is common to every library and code since 1991, so it is not
peculiar to this model. Candidates, in decreasing order of our confidence:
1. **Absorbing impurities in the FNS Be.** The specified composition contains no B/Li/Cd. Pulsed-
   neutron measurements on FNS Be (Yamauchi et al., JAERI-Conf 2004-006, abstract) put the
   macroscopic absorption about 30% above the manufacturer's value. Using Be + Al + Fe + C + O
   absorption of about 1.2e-3 /cm, one ppm of natural boron adds about 6%. A few ppm B-equivalent
   would therefore remove a large part of the excess, and it would act on thermal responses only.
   That fits the pattern: fast foils are fine, while T6, U-235 and Au are all high by a similar
   factor.
2. Be-9 (n,γ) / thermal data. JENDL-4 cut the excess by about 5% via a 10% larger (n,γ).
3. Thermal scattering law. Our VIII.0 Be S(a,b) is in.

The fast-flux deficit (> 10 MeV) is a separate, genuine Be-9 data effect.
**For assimilation: the Be T6/U-235 responses should carry an extra model uncertainty for the
unknown impurity absorption (we suggest ~10-15% on slow responses) or be excluded.** Otherwise the
fit will try to explain a material-composition problem with Be-9 cross sections. The ND samples
cannot move them much anyway (see below).

## Perturbed-library runs (set `all`, 30 samples × seeds A/B)
For each sample i, the perturbed Li6/Li7/Be9/Ni58 HDF5 files are **rebuilt from the saved factors**
(`~/runs/blanket/uq/results/all_A_factors/i.npz`; asserted identical to all_B) with
`blanket/uq/uqlib.write_perturbed`. They go into our own `~/runs/blanket/bench/fns/libs/i/` and are
used for both benchmarks and both seeds, then deleted. F-19 and W are perturbed in the ARC samples
but absent from both assemblies, so leaving them unperturbed is exact. Seeds 777 (A) and 778 (B), as
in the campaign; the nominal library was rerun at both seeds (index −1). Models: no-room baseline
model, 30M histories (Li2O) and 6M (Be) per run.

`results.json → samples.all`: `A[j][i]` = response j with sample i at seed 777 (B: 778), given as
C_nominal × Y_i(s)/Y_nom(s). This carries over the high-statistics nominal, the fine face bins and the
room factor; nominal_A/B therefore equal C. `stats` gives per response the correlated-sampling
estimator of the UQ campaign:
- D_i(s) = Y_i(s) − Y_nom(s)
- var_ND = var_i(mean_s D_i) − var_i(D_A − D_B)/4

with the residual noise, a resolved flag (ND variance > 3× noise variance), and the correlation of D_i
with the paired ARC ΔTBR.

Reading of the ND results:
- ND spread on the TPR responses is **1.2-2.0%** (1σ) in both assemblies. The MC noise left after
  pairing is 0.02-0.4% on integrals and front/mid points, and 1-1.3% only at the deepest T6 points.
  So the ND spread is resolved on every integrated response and on almost every point.
- The Li2O T6 spread (about 2%) is mostly Li-7: at 7.5% Li-6, Li-7 inelastic sets the moderation.
  The Be T6/U-235 spread (about 1.6%) is Be-9.
- Seen from these benchmarks, the 30 samples are **nearly uncorrelated with the paired ARC ΔTBR**
  (|r| ≤ 0.27 for every TPR; with 30 samples, |r| < 0.36 is noise). The two reasons:
  - ARC TBR's ND spread is dominated by Ni-58 and F-19, which are absent or minor here.
  - The Li-7 that drives Li2O T6 matters little at 90% Li-6.
- The exception is Be **Ni-58(n,p)**, with r = −0.55 and ND spread about 10%. This foil sees the
  same Ni-58 (n,p) cross section that dominates the ARC TBR uncertainty. It is the most informative
  single response for ARC here, with C/E 1.01 at the face, rising to 1.1-1.2 at depth.
- Be **flux above 10 MeV** has an ND spread of 0.3% at the face, rising to 3.5-7% at depth. The
  measured −10% deficit is therefore 1.5-3 σ_ND at depth, a Be-9 elastic/(n,2n) signal the
  assimilation can use.

Compute: about 3.3 h wall time for 124 sample runs, 8 threads; about 1.2 h for nominal/variant/room
runs, 4-12 threads, on the shared 32-core box. Library rebuild takes about 1 s per sample.

<!-- AUTO:ND -->
| response | ND rsd % | MC noise % (paired) | resolved | r with ARC ΔTBR |
|---|---|---|---|---|
| li2o:t6_lsc_22.4 | 1.31 | 0.24 | yes | -0.15 |
| li2o:t6_lsc_40.2 | 2.02 | 0.35 | yes | -0.04 |
| li2o:t6_lsc_60.5 | 2.03 | 0.99 | yes | -0.10 |
| li2o:t6_lsc_70.7 | 1.42 | 1.25 | no | -0.20 |
| li2o:t7_lsc_22.4 | 1.23 | 0.03 | yes | -0.10 |
| li2o:t7_lsc_40.2 | 1.51 | 0.21 | yes | -0.24 |
| li2o:t7_ne213_21.6 | 1.25 | 0.02 | yes | -0.08 |
| li2o:t7_ne213_41.6 | 1.55 | 0.16 | yes | -0.23 |
| li2o:t7_ne213_61.8 | 2.22 | 0.89 | yes | -0.25 |
| li2o:foil_ni58_n2n_19.85 | 1.64 | 0.01 | yes | +0.02 |
| li2o:T6_lsc_int | 1.82 | 0.13 | yes | -0.10 |
| li2o:T7_lsc_int | 1.18 | 0.03 | yes | -0.15 |
| li2o:T6_lg_int | 1.83 | 0.20 | yes | -0.09 |
| li2o:T7_ne213_int | 1.19 | 0.04 | yes | -0.18 |
| li2o:phi_gt10_ne213_21.6 | 0.15 | 0.01 | yes | +0.18 |
| li2o:phi_gt10_ne213_31.5 | 0.81 | 0.06 | yes | +0.15 |
| li2o:phi_gt10_ne213_41.6 | 1.51 | 0.11 | yes | +0.14 |
| li2o:phi_gt10_ne213_51.7 | 2.16 | 0.19 | yes | +0.12 |
| li2o:phi_gt10_ne213_61.8 | 2.79 | 0.38 | yes | +0.12 |
| li2o:phi_gt10_ne213_72.0 | 3.59 | 0.57 | yes | +0.07 |
| be:t6_lg_25.17 | 1.74 | 0.59 | yes | +0.15 |
| be:t6_lg_40.35 | 1.62 | 0.38 | yes | +0.10 |
| be:t6_lg_60.59 | 1.21 | 1.01 | no | +0.27 |
| be:fc_u235_f_26.66 | 1.48 | 0.56 | yes | +0.12 |
| be:fc_u235_f_41.90 | 1.60 | 0.30 | yes | +0.04 |
| be:fc_u235_f_57.14 | 2.04 | 0.58 | yes | +0.16 |
| be:foil_au197_ng_20.00 | 1.13 | 3.40 | no | -0.11 |
| be:foil_al27_na_20.00 | 0.35 | 0.06 | yes | -0.04 |
| be:foil_ni58_n2n_20.00 | 1.65 | 0.01 | yes | +0.02 |
| be:foil_fe56_np_20.00 | 0.33 | 0.06 | yes | -0.04 |
| be:foil_zr90_n2n_20.00 | 0.03 | 0.01 | yes | +0.03 |
| be:foil_ni58_np_20.00 | 10.23 | 0.19 | yes | -0.55 |
| be:T6_lg_int | 1.55 | 0.22 | yes | +0.13 |
| be:U235f_int | 1.61 | 0.15 | yes | +0.09 |
| be:phi_gt10_ne213_21.4 | 0.35 | 0.03 | yes | -0.10 |
| be:phi_gt10_ne213_26.5 | 0.61 | 0.08 | yes | -0.09 |
| be:phi_gt10_ne213_31.6 | 1.07 | 0.17 | yes | -0.13 |
| be:phi_gt10_ne213_32.7 | 1.15 | 0.26 | yes | -0.15 |
| be:phi_gt10_ne213_41.7 | 3.51 | 0.37 | yes | -0.05 |
| be:phi_gt10_ne213_47.9 | 3.30 | 1.11 | yes | -0.18 |
| be:phi_gt10_ne213_51.9 | 3.80 | 1.01 | yes | -0.04 |
| be:phi_gt10_ne213_62.0 | 7.34 | 2.54 | yes | -0.08 |

171 of 198 sample responses resolve ND above MC noise (ND var > 3x noise var); the full list is in the tables at the end (ND rsd marked * where not resolved).
<!-- /AUTO:ND -->

## Caveats
- Equivalent cylinders: both assemblies are square-block stacks (63 cm average diameter). The
  r_eq = 31.5 cm cylinder is the compilation's own model; it is not separately quantified here.
- Li2O length 61.0 vs 60.72 cm, the room geometry/orientation and the omitted grating floor only
  touch the last 1-2 positions (room return is resolved only at the front face).
- Li-glass T6 in Li2O is 1.11-1.28 (the 20.25 cm face point 1.33) in the front 30 cm, while the Li
  pellets at the same depths are 0.99-1.03. Both cannot be right with one calculation. JAERI-M 86-182
  §7 notes that the Li-glass ⁶Li−⁷Li subtraction over-subtracts where T7 is large (front), and that
  near-surface values are affected by room return and self-shielding. We trust the pellet data and
  flag the front Li-glass points.
- No photon transport (not needed for any response here). No detector perturbation modelled beyond
  the drawer (the data are corrected to unperturbed values). Au(n,γ) in Be is not corrected for foil
  self-shielding (foil thickness not given).
- Not done: Li2O fission-track data (Table 7.13), In-115(n,n')/(n,γ), Nb-93(n,2n)92m and Ti(n,x)Sc
  foils (no isomer production data in the transport library; IRDFF not set up). Also not done: the
  PRC low-energy spectra in Be (units/normalisation unclear), the FNS TOF slab leakage (`fns-tof/`),
  and the Be-sandwich Li2O assemblies (no such data in this compilation; JAERI-M 86-182 /
  Maekawa 1988 describe them but the numbers are not in `jaerim94014`).

## Full tables
<!-- AUTO:TABLES -->

#### li2o: every response

| id | pos cm | E | E unc % (sys) | C/E | MC % | model % | f_room | ND rsd % | MC-noise % | r(TBR) |
|---|---|---|---|---|---|---|---|---|---|---|
| t6_lsc_19.9 | 19.90 | 7.660e-29 | 4.8 (1.8) | 0.867 | 0.32 | 0.7 | 2.627 (n.a.) | 1.04 | 0.16 | -0.17 |
| t6_lsc_22.4 | 22.40 | 8.930e-29 | 4.5 (1.8) | 0.993 | 0.31 | 1.4 | 1.021 (n.a.) | 1.31 | 0.24 | -0.15 |
| t6_lsc_24.9 | 24.90 | 9.440e-29 | 4.5 (1.8) | 1.000 | 0.31 | 0.4 | 0.998 (n.a.) | 1.74 | 0.28 | -0.16 |
| t6_lsc_27.5 | 27.50 | 9.230e-29 | 4.5 (1.8) | 1.028 | 0.26 | 0.3 | 1.008 (n.a.) | 1.93 | 0.19 | -0.16 |
| t6_lsc_30.0 | 30.00 | 9.150e-29 | 4.5 (1.8) | 0.996 | 0.26 | 1.0 | 0.999 (n.a.) | 1.82 | 0.30 | -0.12 |
| t6_lsc_35.1 | 35.10 | 8.180e-29 | 4.7 (1.8) | 0.996 | 0.40 | 1.0 | 1.015 (n.a.) | 1.92 | 0.30 | -0.10 |
| t6_lsc_40.2 | 40.20 | 7.010e-29 | 4.9 (1.8) | 0.988 | 0.48 | 0.5 | 1.005 (n.a.) | 2.02 | 0.35 | -0.04 |
| t6_lsc_45.3 | 45.30 | 5.710e-29 | 5.3 (1.8) | 1.008 | 0.57 | 1.1 | 1.007 (n.a.) | 2.17 | 0.66 | -0.06 |
| t6_lsc_50.3 | 50.30 | 4.550e-29 | 5.8 (1.8) | 1.000 | 0.61 | 0.2 | 1.002 (n.a.) | 2.02 | 0.52 | -0.05 |
| t6_lsc_55.4 | 55.40 | 3.480e-29 | 6.5 (1.8) | 1.011 | 0.56 | 1.4 | 0.981 (n.a.) | 1.70 | 0.65 | -0.04 |
| t6_lsc_60.5 | 60.50 | 2.760e-29 | 7.3 (1.8) | 0.957 | 0.67 | 0.1 | 1.019 (n.a.) | 2.03 | 0.99 | -0.10 |
| t6_lsc_65.6 | 65.60 | 2.030e-29 | 8.7 (1.8) | 0.945 | 0.80 | 1.7 | 0.998 (n.a.) | 1.68 | 0.92 | -0.08 |
| t6_lsc_70.7 | 70.70 | 1.460e-29 | 10.5 (1.8) | 0.905 | 1.24 | 3.3 | 1.044 (n.a.) | 1.42* | 1.25 | -0.20 |
| t6_lsc_75.7 | 75.70 | 8.360e-30 | 15.7 (1.8) | 0.957 | 1.57 | 4.1 | 0.997 (n.a.) | 0.61* | 1.76 | -0.03 |
| t7_lsc_19.9 | 19.90 | 6.320e-29 | 4.2 (1.8) | 0.963 | 0.11 | 0.9 | 1.000 (n.a.) | 1.28 | 0.02 | -0.00 |
| t7_lsc_22.4 | 22.40 | 4.600e-29 | 4.9 (1.8) | 1.069 | 0.12 | 0.9 | 1.000 (n.a.) | 1.23 | 0.03 | -0.10 |
| t7_lsc_24.9 | 24.90 | 3.850e-29 | 5.6 (1.8) | 0.981 | 0.15 | 1.0 | 1.000 (n.a.) | 1.22 | 0.05 | -0.13 |
| t7_lsc_27.5 | 27.50 | 2.670e-29 | 6.4 (1.8) | 1.069 | 0.19 | 1.6 | 1.000 (n.a.) | 1.19 | 0.03 | -0.17 |
| t7_lsc_30.0 | 30.00 | 2.080e-29 | 7.6 (1.8) | 1.057 | 0.24 | 1.5 | 1.000 (n.a.) | 1.27 | 0.08 | -0.20 |
| t7_lsc_35.1 | 35.10 | 1.260e-29 | 10.7 (1.8) | 1.032 | 0.31 | 2.9 | 1.000 (n.a.) | 1.38 | 0.11 | -0.20 |
| t7_lsc_40.2 | 40.20 | 4.570e-30 | 23.8 (1.8) | 1.716 | 0.46 | 3.8 | 1.000 (n.a.) | 1.51 | 0.21 | -0.24 |
| t7_lsc_45.3 | 45.30 | 3.920e-30 | 28.1 (1.8) | 1.244 | 0.58 | 3.1 | 1.000 (n.a.) | 1.69 | 0.27 | -0.21 |
| t6_lg_ax_20.25 | 20.25 | 5.440e-29 | 11.0 (5.4) | 1.330 | 0.47 | 40.2 | 1.297 (n.a.) | 1.15 | 0.20 | -0.18 |
| t6_lg_ax_22.53 | 22.53 | 7.050e-29 | 9.9 (5.4) | 1.262 | 0.27 | 3.4 | 1.016 (n.a.) | 1.34 | 0.20 | -0.15 |
| t6_lg_ax_25.06 | 25.06 | 8.490e-29 | 9.1 (5.4) | 1.113 | 0.29 | 1.7 | 0.999 (n.a.) | 1.78 | 0.29 | -0.15 |
| t6_lg_ax_27.61 | 27.61 | 7.630e-29 | 10.5 (5.4) | 1.241 | 0.28 | 2.5 | 1.009 (n.a.) | 1.94 | 0.21 | -0.13 |
| t6_lg_ax_30.12 | 30.12 | 7.320e-29 | 8.3 (5.4) | 1.243 | 0.28 | 2.5 | 0.996 (n.a.) | 1.79 | 0.31 | -0.11 |
| t6_lg_ax_40.24 | 40.24 | 6.210e-29 | 7.9 (5.4) | 1.114 | 0.52 | 3.5 | 1.005 (n.a.) | 2.02 | 0.36 | -0.04 |
| t6_lg_ax_50.38 | 50.38 | 4.230e-29 | 7.1 (5.4) | 1.073 | 0.53 | 3.6 | 1.002 (n.a.) | 2.06 | 0.46 | -0.04 |
| t6_lg_ax_60.50 | 60.50 | 2.520e-29 | 7.3 (5.4) | 1.048 | 0.67 | 5.1 | 1.019 (n.a.) | 2.03 | 0.99 | -0.10 |
| t6_lg_ax_70.64 | 70.64 | 1.330e-29 | 6.9 (5.4) | 0.998 | 1.11 | 11.3 | 1.040 (n.a.) | 1.40* | 1.17 | -0.21 |
| t6_lg_ax_75.72 | 75.72 | 8.280e-30 | 6.8 (5.4) | 0.963 | 1.63 | 10.6 | 0.998 (n.a.) | 0.47* | 1.81 | -0.02 |
| t6_lg_r15_20.25 | 20.25 | 6.100e-29 | 9.5 (5.4) | 1.404 | 3.46 | 35.1 | 1.701 | 1.19 | 0.08 | -0.15 |
| t6_lg_r15_25.06 | 25.06 | 6.560e-29 | 8.3 (5.4) | 1.051 | 0.07 | 1.8 | 1.012 (n.a.) | 1.73 | 0.07 | -0.14 |
| t6_lg_r15_30.12 | 30.12 | 6.040e-29 | 8.6 (5.4) | 1.140 | 0.10 | 1.1 | 1.003 (n.a.) | 1.88 | 0.08 | -0.10 |
| t6_lg_r15_50.38 | 50.38 | 3.570e-29 | 7.7 (5.4) | 0.972 | 0.15 | 2.2 | 1.011 (n.a.) | 2.08 | 0.18 | -0.02 |
| t6_lg_r15_70.64 | 70.64 | 1.110e-29 | 6.8 (5.4) | 0.914 | 0.23 | 3.4 | 1.018 (n.a.) | 2.05 | 0.33 | +0.00 |
| t7_ne213_21.6 | 21.60 | 4.610e-29 | 4.0 (2.5) | 1.157 | 0.12 | 0.8 | 1.000 (n.a.) | 1.25 | 0.02 | -0.08 |
| t7_ne213_31.5 | 31.50 | 1.660e-29 | 3.0 (2.5) | 1.137 | 0.25 | 1.7 | 1.000 (n.a.) | 1.28 | 0.08 | -0.22 |
| t7_ne213_41.6 | 41.60 | 6.230e-30 | 3.0 (2.5) | 1.107 | 0.39 | 2.6 | 0.999 (n.a.) | 1.55 | 0.16 | -0.23 |
| t7_ne213_51.7 | 51.70 | 2.400e-30 | 3.6 (2.5) | 1.083 | 0.87 | 5.4 | 1.000 (n.a.) | 1.89 | 0.52 | -0.17 |
| t7_ne213_61.8 | 61.80 | 9.040e-31 | 4.1 (2.5) | 1.111 | 1.30 | 7.1 | 1.000 (n.a.) | 2.22 | 0.89 | -0.25 |
| t7_ne213_72.0 | 72.00 | 3.460e-31 | 4.9 (2.5) | 1.089 | 1.72 | 12.0 | 1.000 (n.a.) | 3.54 | 0.93 | -0.24 |
| t7_ne213_82.1 | 82.10 | 1.250e-31 | 4.4 (2.5) | 1.090 | 3.02 | 24.1 | 1.056 (n.a.) | 2.91* | 2.16 | -0.22 |
| fc_u235_f_23.9 | 23.90 | 5.000e-28 | 4.1 (3.7) | 0.981 | 0.12 | 0.9 | 1.001 (n.a.) | 0.49 | 0.07 | -0.06 |
| fc_u235_f_26.4 | 26.40 | 4.250e-28 | 4.2 (3.7) | 0.988 | 0.16 | 1.5 | 1.004 (n.a.) | 0.69 | 0.13 | -0.08 |
| fc_u235_f_29.0 | 29.00 | 3.670e-28 | 4.1 (3.7) | 0.981 | 0.15 | 1.1 | 1.000 (n.a.) | 0.76 | 0.13 | -0.05 |
| fc_u235_f_31.5 | 31.50 | 3.150e-28 | 4.1 (3.7) | 0.990 | 0.17 | 2.2 | 1.008 (n.a.) | 0.87 | 0.15 | -0.00 |
| fc_u235_f_36.6 | 36.60 | 2.400e-28 | 4.1 (3.7) | 0.989 | 0.24 | 1.8 | 1.000 (n.a.) | 1.08 | 0.22 | +0.02 |
| fc_u235_f_41.7 | 41.70 | 1.830e-28 | 4.1 (3.7) | 0.984 | 0.37 | 2.1 | 1.000 (n.a.) | 1.37 | 0.31 | +0.02 |
| fc_u235_f_46.7 | 46.70 | 1.400e-28 | 4.1 (3.7) | 0.971 | 0.43 | 4.2 | 1.009 (n.a.) | 1.69 | 0.34 | +0.01 |
| fc_u235_f_51.8 | 51.80 | 1.060e-28 | 4.1 (3.7) | 0.964 | 0.54 | 4.1 | 1.006 (n.a.) | 1.63 | 0.71 | -0.07 |
| fc_u235_f_56.9 | 56.90 | 7.960e-29 | 4.1 (3.7) | 0.951 | 0.58 | 3.5 | 1.005 (n.a.) | 1.84 | 0.52 | -0.03 |
| fc_u235_f_62.0 | 62.00 | 5.900e-29 | 4.1 (3.7) | 0.937 | 0.67 | 5.2 | 1.016 (n.a.) | 1.42 | 0.81 | +0.00 |
| fc_u235_f_67.1 | 67.10 | 4.180e-29 | 4.1 (3.7) | 0.929 | 0.88 | 4.6 | 1.007 (n.a.) | 0.90* | 0.96 | +0.09 |
| fc_u235_f_72.1 | 72.10 | 2.810e-29 | 4.1 (3.7) | 0.914 | 1.16 | 8.8 | 0.981 (n.a.) | 0.92* | 1.39 | -0.01 |
| fc_u235_f_77.2 | 77.20 | 1.680e-29 | 4.2 (3.7) | 0.884 | 1.59 | 11.5 | 1.001 (n.a.) | 1.59* | 1.64 | -0.10 |
| fc_u238_f_23.9 | 23.90 | 1.850e-28 | 3.9 (3.5) | 0.992 | 0.12 | 0.8 | 1.000 (n.a.) | 0.13 | 0.03 | +0.08 |
| fc_u238_f_26.4 | 26.40 | 1.410e-28 | 3.9 (3.5) | 0.991 | 0.14 | 1.4 | 1.001 (n.a.) | 0.23 | 0.04 | +0.18 |
| fc_u238_f_29.0 | 29.00 | 1.050e-28 | 3.9 (3.5) | 1.015 | 0.16 | 1.2 | 0.999 (n.a.) | 0.35 | 0.05 | +0.16 |
| fc_u238_f_31.5 | 31.50 | 8.040e-29 | 3.9 (3.5) | 1.030 | 0.20 | 1.3 | 1.000 (n.a.) | 0.41 | 0.07 | +0.11 |
| fc_u238_f_36.6 | 36.60 | 4.870e-29 | 3.9 (3.5) | 1.021 | 0.26 | 2.1 | 1.002 (n.a.) | 0.67 | 0.14 | +0.14 |
| fc_u238_f_41.7 | 41.70 | 2.970e-29 | 3.9 (3.5) | 1.024 | 0.37 | 2.4 | 0.999 (n.a.) | 0.91 | 0.16 | +0.13 |
| fc_u238_f_46.7 | 46.70 | 1.810e-29 | 3.9 (3.5) | 1.045 | 0.50 | 3.1 | 1.002 (n.a.) | 1.11 | 0.26 | +0.16 |
| fc_u238_f_51.8 | 51.80 | 1.140e-29 | 3.9 (3.5) | 1.024 | 0.69 | 3.9 | 0.998 (n.a.) | 1.38 | 0.51 | +0.13 |
| fc_u238_f_56.9 | 56.90 | 7.050e-30 | 3.9 (3.5) | 1.014 | 0.64 | 4.5 | 0.996 (n.a.) | 1.56 | 0.58 | +0.08 |
| fc_u238_f_62.0 | 62.00 | 4.420e-30 | 4.0 (3.5) | 1.018 | 0.84 | 4.3 | 0.998 (n.a.) | 1.74 | 0.58 | +0.12 |
| fc_u238_f_67.1 | 67.10 | 2.800e-30 | 4.0 (3.5) | 0.990 | 1.25 | 7.8 | 0.996 (n.a.) | 2.68 | 0.59 | +0.01 |
| fc_u238_f_72.1 | 72.10 | 1.720e-30 | 4.0 (3.5) | 1.003 | 1.30 | 8.2 | 1.001 (n.a.) | 2.71 | 0.75 | -0.04 |
| fc_u238_f_77.2 | 77.20 | 9.970e-31 | 4.2 (3.5) | 0.985 | 2.23 | 18.9 | 1.059 (n.a.) | 3.19 | 1.37 | +0.01 |
| fc_np237_f_23.9 | 23.90 | 4.710e-28 | 5.1 (4.6) | 0.931 | 0.11 | 0.7 | 0.999 (n.a.) | 0.25 | 0.04 | -0.04 |
| fc_np237_f_26.4 | 26.40 | 3.670e-28 | 5.1 (4.6) | 0.955 | 0.14 | 1.4 | 1.000 (n.a.) | 0.23 | 0.05 | +0.05 |
| fc_np237_f_29.0 | 29.00 | 2.930e-28 | 5.1 (4.6) | 0.952 | 0.14 | 1.1 | 1.000 (n.a.) | 0.22 | 0.06 | +0.09 |
| fc_np237_f_31.5 | 31.50 | 2.380e-28 | 5.1 (4.6) | 0.945 | 0.17 | 1.3 | 1.001 (n.a.) | 0.22 | 0.09 | +0.19 |
| fc_np237_f_36.6 | 36.60 | 1.510e-28 | 5.1 (4.6) | 0.959 | 0.23 | 1.6 | 1.000 (n.a.) | 0.19* | 0.16 | +0.43 |
| fc_np237_f_41.7 | 41.70 | 9.570e-29 | 5.1 (4.6) | 0.979 | 0.33 | 2.4 | 0.997 (n.a.) | 0.40 | 0.17 | +0.32 |
| fc_np237_f_46.7 | 46.70 | 6.330e-29 | 5.1 (4.6) | 0.960 | 0.41 | 2.6 | 1.000 (n.a.) | 0.52 | 0.27 | +0.34 |
| fc_np237_f_51.8 | 51.80 | 4.050e-29 | 5.1 (4.6) | 0.965 | 0.60 | 3.4 | 0.998 (n.a.) | 0.59* | 0.51 | +0.28 |
| fc_np237_f_56.9 | 56.90 | 2.680e-29 | 5.2 (4.6) | 0.932 | 0.51 | 3.9 | 0.998 (n.a.) | 0.79 | 0.45 | +0.20 |
| fc_np237_f_62.0 | 62.00 | 1.680e-29 | 5.2 (4.6) | 0.952 | 0.69 | 4.3 | 1.002 (n.a.) | 1.12 | 0.59 | +0.23 |
| fc_np237_f_67.1 | 67.10 | 1.100e-29 | 5.2 (4.6) | 0.913 | 0.89 | 6.5 | 1.006 (n.a.) | 1.70 | 0.51 | +0.10 |
| fc_np237_f_72.1 | 72.10 | 7.130e-30 | 5.3 (4.6) | 0.896 | 1.05 | 7.7 | 1.014 (n.a.) | 1.78 | 0.79 | +0.04 |
| fc_np237_f_77.2 | 77.20 | 4.310e-30 | 5.2 (4.6) | 0.837 | 1.83 | 14.3 | 1.039 (n.a.) | 2.10 | 1.03 | +0.01 |
| fc_th232_f_23.9 | 23.90 | 5.740e-29 | 4.7 (4.2) | 1.000 | 0.12 | 0.8 | 1.000 (n.a.) | 0.16 | 0.03 | +0.10 |
| fc_th232_f_26.4 | 26.40 | 4.290e-29 | 4.7 (4.2) | 1.007 | 0.14 | 1.4 | 1.000 (n.a.) | 0.29 | 0.03 | +0.15 |
| fc_th232_f_29.0 | 29.00 | 3.290e-29 | 4.7 (4.2) | 0.993 | 0.17 | 1.1 | 0.999 (n.a.) | 0.42 | 0.05 | +0.15 |
| fc_th232_f_31.5 | 31.50 | 2.450e-29 | 4.7 (4.2) | 1.026 | 0.21 | 1.3 | 1.000 (n.a.) | 0.49 | 0.07 | +0.10 |
| fc_th232_f_36.6 | 36.60 | 1.440e-29 | 4.7 (4.2) | 1.034 | 0.27 | 2.1 | 1.001 (n.a.) | 0.76 | 0.12 | +0.13 |
| fc_th232_f_41.7 | 41.70 | 8.730e-30 | 4.7 (4.2) | 1.032 | 0.37 | 2.5 | 0.999 (n.a.) | 1.00 | 0.16 | +0.10 |
| fc_th232_f_46.7 | 46.70 | 5.400e-30 | 4.7 (4.2) | 1.030 | 0.50 | 3.2 | 1.002 (n.a.) | 1.22 | 0.26 | +0.14 |
| fc_th232_f_51.8 | 51.80 | 3.290e-30 | 4.8 (4.2) | 1.034 | 0.72 | 4.2 | 0.998 (n.a.) | 1.47 | 0.48 | +0.13 |
| fc_th232_f_56.9 | 56.90 | 2.050e-30 | 5.2 (4.2) | 1.007 | 0.66 | 4.2 | 0.997 (n.a.) | 1.71 | 0.63 | +0.07 |
| fc_th232_f_67.1 | 67.10 | 7.870e-31 | 6.4 (4.2) | 1.012 | 1.29 | 8.0 | 0.997 (n.a.) | 2.80 | 0.56 | +0.00 |
| fc_th232_f_72.1 | 72.10 | 5.220e-31 | 5.6 (4.2) | 0.943 | 1.42 | 8.7 | 1.000 (n.a.) | 2.87 | 0.80 | -0.05 |
| fc_th232_f_77.2 | 77.20 | 3.300e-31 | 5.7 (4.2) | 0.846 | 2.30 | 20.2 | 1.061 (n.a.) | 3.35 | 1.45 | +0.01 |
| foil_al27_na_19.95 | 19.95 | 2.120e-29 | 3.2 (2.5) | 1.017 | 0.15 | 1.0 | 1.000 (n.a.) | 0.03 | 0.01 | +0.16 |
| foil_al27_na_22.44 | 22.44 | 1.570e-29 | 3.1 (2.5) | 1.048 | 0.12 | 0.9 | 1.000 (n.a.) | 0.24 | 0.02 | +0.14 |
| foil_al27_na_24.98 | 24.98 | 1.180e-29 | 3.1 (2.5) | 1.029 | 0.15 | 1.1 | 1.000 (n.a.) | 0.43 | 0.03 | +0.15 |
| foil_al27_na_27.52 | 27.52 | 8.610e-30 | 3.3 (2.5) | 1.041 | 0.18 | 1.5 | 1.000 (n.a.) | 0.63 | 0.03 | +0.14 |
| foil_al27_na_30.06 | 30.06 | 6.310e-30 | 3.3 (2.5) | 1.062 | 0.24 | 1.7 | 1.000 (n.a.) | 0.77 | 0.06 | +0.14 |
| foil_al27_na_35.14 | 35.14 | 3.600e-30 | 3.3 (2.5) | 1.056 | 0.34 | 3.1 | 1.000 (n.a.) | 1.07 | 0.10 | +0.14 |
| foil_al27_na_40.22 | 40.22 | 2.120e-30 | 3.5 (2.5) | 1.047 | 0.47 | 3.9 | 1.000 (n.a.) | 1.37 | 0.22 | +0.13 |
| foil_al27_na_45.30 | 45.30 | 1.280e-30 | 3.5 (2.5) | 1.044 | 0.60 | 3.0 | 1.000 (n.a.) | 1.71 | 0.18 | +0.10 |
| foil_al27_na_50.29 | 50.29 | 7.620e-31 | 3.5 (2.5) | 1.058 | 0.83 | 5.1 | 1.000 (n.a.) | 1.96 | 0.28 | +0.04 |
| foil_al27_na_55.47 | 55.47 | 4.490e-31 | 3.1 (2.5) | 1.058 | 0.80 | 5.1 | 1.000 (n.a.) | 2.34 | 0.33 | +0.12 |
| foil_al27_na_60.55 | 60.55 | 2.840e-31 | 3.8 (2.5) | 1.004 | 1.11 | 6.4 | 1.000 (n.a.) | 2.92 | 0.47 | +0.09 |
| foil_al27_na_65.63 | 65.63 | 1.670e-31 | 3.8 (2.5) | 1.040 | 1.76 | 9.0 | 1.000 (n.a.) | 3.09 | 0.90 | -0.01 |
| foil_al27_na_70.71 | 70.71 | 9.870e-32 | 3.7 (2.5) | 1.124 | 2.54 | 16.0 | 1.004 (n.a.) | 3.23 | 0.93 | +0.12 |
| foil_al27_na_75.79 | 75.79 | 6.110e-32 | 3.6 (2.5) | 1.061 | 2.52 | 20.9 | 1.058 (n.a.) | 4.07 | 1.56 | -0.09 |
| foil_ni58_n2n_19.85 | 19.85 | 8.000e-30 | 3.1 (2.5) | 0.927 | 0.14 | 1.1 | 1.000 (n.a.) | 1.64 | 0.01 | +0.02 |
| foil_ni58_n2n_22.34 | 22.34 | 5.190e-30 | 2.8 (2.5) | 0.971 | 0.14 | 1.1 | 1.000 (n.a.) | 1.61 | 0.01 | +0.04 |
| foil_ni58_n2n_24.88 | 24.88 | 3.610e-30 | 3.2 (2.5) | 0.939 | 0.16 | 1.0 | 1.000 (n.a.) | 1.62 | 0.02 | +0.06 |
| foil_ni58_n2n_27.42 | 27.42 | 2.340e-30 | 4.5 (2.5) | 0.988 | 0.19 | 1.0 | 1.000 (n.a.) | 1.65 | 0.02 | +0.08 |
| foil_ni58_n2n_29.96 | 29.96 | 1.620e-30 | 5.3 (2.5) | 0.989 | 0.21 | 1.2 | 1.000 (n.a.) | 1.70 | 0.03 | +0.10 |
| foil_ni58_n2n_35.04 | 35.04 | 8.840e-31 | 5.3 (2.5) | 0.898 | 0.32 | 2.4 | 1.000 (n.a.) | 1.83 | 0.04 | +0.14 |
| foil_ni58_n2n_40.12 | 40.12 | 4.100e-31 | 6.6 (2.5) | 0.996 | 0.29 | 2.7 | 1.000 (n.a.) | 2.03 | 0.08 | +0.15 |
| foil_ni58_n2n_45.20 | 45.20 | 2.350e-31 | 6.7 (2.5) | 0.936 | 0.74 | 4.6 | 1.000 (n.a.) | 2.24 | 0.10 | +0.18 |
| foil_ni58_n2n_50.29 | 50.29 | 1.230e-31 | 7.1 (2.5) | 0.954 | 0.95 | 5.8 | 1.000 (n.a.) | 2.42 | 0.19 | +0.18 |
| foil_ni58_n2n_55.37 | 55.37 | 6.130e-32 | 2.8 (2.5) | 1.028 | 1.15 | 6.8 | 1.000 (n.a.) | 2.67 | 0.35 | +0.16 |
| foil_ni58_n2n_60.45 | 60.45 | 3.220e-32 | 9.5 (2.5) | 1.080 | 1.49 | 8.4 | 1.000 (n.a.) | 2.64 | 0.47 | +0.15 |
| foil_ni58_n2n_65.53 | 65.53 | 2.010e-32 | 9.6 (2.5) | 0.963 | 1.88 | 11.6 | 1.000 (n.a.) | 3.11 | 0.51 | +0.08 |
| foil_ni58_n2n_70.61 | 70.61 | 1.160e-32 | 10.3 (2.5) | 0.936 | 2.76 | 13.3 | 1.000 (n.a.) | 3.10 | 0.84 | +0.11 |
| foil_ni58_n2n_75.69 | 75.69 | 6.120e-33 | 8.9 (2.5) | 1.004 | 3.78 | 24.1 | 1.000 (n.a.) | 3.20 | 0.87 | +0.10 |
| T6_lsc_int | - | 2.952e-27 | 2.3 (1.8) | 0.990 | 0.14 | 0.3 | 1.083 (n.a.) | 1.82 | 0.13 | -0.10 |
| T7_lsc_int | - | 5.369e-28 | 3.2 (1.8) | 1.062 | 0.08 | 1.0 | 1.000 (n.a.) | 1.18 | 0.03 | -0.15 |
| T6_lg_int | - | 2.538e-27 | 5.9 (5.4) | 1.143 | 0.18 | 2.6 | 1.015 (n.a.) | 1.83 | 0.20 | -0.09 |
| T7_ne213_int | - | 4.947e-28 | 3.0 (2.5) | 1.139 | 0.12 | 1.0 | 1.000 (n.a.) | 1.19 | 0.04 | -0.18 |
| phi_gt10_ne213_21.6 | 21.60 | 1.563e-04 | 3.9 (2.5) | 1.007 | 0.11 | 0.7 | 1.000 (n.a.) | 0.15 | 0.01 | +0.18 |
| phi_gt10_ne213_31.5 | 31.50 | 4.536e-05 | 3.9 (2.5) | 1.025 | 0.21 | 1.7 | 1.000 (n.a.) | 0.81 | 0.06 | +0.15 |
| phi_gt10_ne213_41.6 | 41.60 | 1.479e-05 | 3.9 (2.5) | 1.027 | 0.35 | 2.5 | 1.000 (n.a.) | 1.51 | 0.11 | +0.14 |
| phi_gt10_ne213_51.7 | 51.70 | 5.219e-06 | 3.9 (2.5) | 1.024 | 0.60 | 3.6 | 1.000 (n.a.) | 2.16 | 0.19 | +0.12 |
| phi_gt10_ne213_61.8 | 61.80 | 1.799e-06 | 3.9 (2.5) | 1.063 | 1.05 | 5.8 | 1.000 (n.a.) | 2.79 | 0.38 | +0.12 |
| phi_gt10_ne213_72.0 | 72.00 | 6.454e-07 | 3.9 (2.5) | 1.101 | 1.70 | 10.0 | 1.004 (n.a.) | 3.59 | 0.57 | +0.07 |

#### be: every response

| id | pos cm | E | E unc % (sys) | C/E | MC % | model % | f_room | ND rsd % | MC-noise % | r(TBR) |
|---|---|---|---|---|---|---|---|---|---|---|
| t6_lg_20.11 | 20.11 | 3.583e-26 | 3.1 (2.0) | 1.116 | 0.80 | 6.5 | 0.984 (n.a.) | 1.61* | 0.96 | +0.30 |
| t6_lg_25.17 | 25.17 | 2.065e-25 | 3.2 (2.0) | 1.123 | 0.38 | 4.6 | 1.018 (n.a.) | 1.74 | 0.59 | +0.15 |
| t6_lg_30.23 | 30.23 | 2.983e-25 | 3.1 (2.0) | 1.184 | 0.42 | 2.8 | 0.994 (n.a.) | 1.71 | 0.44 | +0.05 |
| t6_lg_32.76 | 32.76 | 3.226e-25 | 2.4 (2.0) | 1.202 | 0.34 | 3.0 | 0.992 (n.a.) | 1.72 | 0.42 | +0.15 |
| t6_lg_40.35 | 40.35 | 3.254e-25 | 2.4 (2.0) | 1.183 | 0.32 | 3.1 | 1.010 (n.a.) | 1.62 | 0.38 | +0.10 |
| t6_lg_47.94 | 47.94 | 2.339e-25 | 2.9 (2.0) | 1.251 | 0.40 | 2.6 | 1.005 (n.a.) | 1.57 | 0.56 | +0.05 |
| t6_lg_50.47 | 50.47 | 2.007e-25 | 2.9 (2.0) | 1.257 | 0.37 | 2.3 | 1.002 (n.a.) | 1.16 | 0.59 | +0.13 |
| t6_lg_60.59 | 60.59 | 7.863e-26 | 2.7 (2.0) | 1.178 | 0.73 | 5.2 | 1.012 (n.a.) | 1.21* | 1.01 | +0.27 |
| t6_lg_63.12 | 63.12 | 5.050e-26 | 3.0 (2.0) | 1.052 | 0.84 | 7.4 | 1.021 (n.a.) | 2.08* | 1.24 | +0.18 |
| fc_u235_f_24.12 | 24.12 | 9.267e-26 | 3.8 (3.2) | 1.312 | 0.46 | 3.9 | 1.017 (n.a.) | 1.60 | 0.55 | +0.13 |
| fc_u235_f_26.66 | 26.66 | 1.307e-25 | 3.8 (3.2) | 1.289 | 0.35 | 3.4 | 1.008 (n.a.) | 1.48 | 0.56 | +0.12 |
| fc_u235_f_29.20 | 29.20 | 1.579e-25 | 3.8 (3.2) | 1.296 | 0.48 | 4.6 | 0.975 (n.a.) | 1.37 | 0.42 | +0.03 |
| fc_u235_f_31.74 | 31.74 | 1.794e-25 | 3.8 (3.2) | 1.279 | 0.40 | 3.5 | 0.991 (n.a.) | 1.70 | 0.53 | +0.13 |
| fc_u235_f_34.28 | 34.28 | 1.906e-25 | 3.8 (3.2) | 1.272 | 0.38 | 2.8 | 0.997 (n.a.) | 1.52 | 0.44 | +0.11 |
| fc_u235_f_41.90 | 41.90 | 1.771e-25 | 3.8 (3.2) | 1.281 | 0.36 | 4.6 | 1.027 (n.a.) | 1.60 | 0.30 | +0.04 |
| fc_u235_f_46.98 | 46.98 | 1.432e-25 | 3.8 (3.2) | 1.304 | 0.33 | 3.6 | 1.013 (n.a.) | 1.61 | 0.43 | +0.05 |
| fc_u235_f_52.06 | 52.06 | 1.073e-25 | 3.8 (3.2) | 1.280 | 0.50 | 3.4 | 1.009 (n.a.) | 1.90 | 0.48 | +0.05 |
| fc_u235_f_57.14 | 57.14 | 7.071e-26 | 3.8 (3.2) | 1.263 | 0.59 | 6.2 | 0.973 (n.a.) | 2.04 | 0.58 | +0.16 |
| fc_u235_f_62.22 | 62.22 | 3.368e-26 | 3.8 (3.2) | 1.196 | 1.03 | 8.7 | 1.037 (n.a.) | 2.05* | 1.24 | +0.16 |
| foil_au197_ng_20.00 | 20.00 | 8.628e-27 | 3.0 (2.5) | 1.079 | 5.89 | 21.4 | 0.932 (n.a.) | 1.13* | 3.40 | -0.11 |
| foil_au197_ng_24.80 | 24.80 | 4.806e-26 | 2.7 (2.5) | 1.058 | 1.96 | 16.0 | 1.057 (n.a.) | 0.71* | 1.66 | +0.14 |
| foil_au197_ng_34.92 | 34.92 | 7.272e-26 | 3.0 (2.5) | 1.137 | 1.15 | 12.1 | 1.021 (n.a.) | 1.34* | 1.50 | -0.17 |
| foil_au197_ng_45.06 | 45.06 | 4.957e-26 | 2.9 (2.5) | 1.084 | 1.21 | 9.9 | 1.032 (n.a.) | 2.25* | 1.47 | +0.13 |
| foil_au197_ng_55.20 | 55.20 | 2.330e-26 | 2.9 (2.5) | 1.172 | 1.99 | 28.5 | 0.803 (n.a.) | 1.23* | 2.17 | -0.05 |
| foil_au197_ng_65.59 | 65.59 | 2.151e-27 | 3.0 (2.5) | 0.842 | 6.85 | 10.7 | 0.978 (n.a.) | 0.00* | 11.54 | -0.21 |
| foil_al27_na_20.00 | 20.00 | 2.588e-29 | 2.7 (2.5) | 0.937 | 0.31 | 1.9 | 1.002 (n.a.) | 0.35 | 0.06 | -0.04 |
| foil_al27_na_24.80 | 24.80 | 1.382e-29 | 2.9 (2.5) | 1.013 | 0.50 | 2.2 | 1.000 (n.a.) | 0.66 | 0.25 | -0.15 |
| foil_al27_na_34.92 | 34.92 | 3.750e-30 | 3.0 (2.5) | 1.024 | 0.68 | 5.6 | 0.991 (n.a.) | 1.21* | 0.75 | -0.10 |
| foil_al27_na_45.06 | 45.06 | 1.039e-30 | 3.2 (2.5) | 1.055 | 0.94 | 7.6 | 1.010 (n.a.) | 2.31 | 0.68 | -0.22 |
| foil_al27_na_55.20 | 55.20 | 2.853e-31 | 3.7 (2.5) | 1.089 | 2.71 | 17.6 | 1.015 (n.a.) | 4.18 | 2.38 | +0.09 |
| foil_al27_na_65.59 | 65.59 | 7.559e-32 | 3.2 (2.5) | 1.040 | 5.77 | 39.1 | 1.059 (n.a.) | 4.45* | 4.48 | -0.07 |
| foil_ni58_n2n_20.00 | 20.00 | 9.416e-30 | 2.9 (2.5) | 0.900 | 0.28 | 1.3 | 1.000 (n.a.) | 1.65 | 0.01 | +0.02 |
| foil_ni58_n2n_24.80 | 24.80 | 3.675e-30 | 2.7 (2.5) | 0.982 | 0.42 | 2.0 | 1.000 (n.a.) | 1.72 | 0.06 | +0.07 |
| foil_ni58_n2n_34.92 | 34.92 | 6.881e-31 | 3.6 (2.5) | 0.936 | 0.80 | 5.5 | 1.000 (n.a.) | 3.80 | 0.18 | +0.09 |
| foil_ni58_n2n_45.06 | 45.06 | 1.338e-31 | 4.3 (2.5) | 0.969 | 1.35 | 9.6 | 1.000 (n.a.) | 7.45 | 0.40 | +0.04 |
| foil_ni58_n2n_55.20 | 55.20 | 3.271e-32 | 4.5 (2.5) | 0.794 | 4.14 | 31.2 | 1.000 (n.a.) | 11.20 | 1.95 | +0.10 |
| foil_ni58_n2n_65.59 | 65.59 | 7.543e-33 | 9.0 (2.5) | 0.650 | 11.00 | 50.2 | 1.000 (n.a.) | 3.21* | 7.87 | +0.16 |
| foil_fe56_np_20.00 | 20.00 | 2.240e-29 | 2.7 (2.5) | 1.020 | 0.30 | 1.8 | 1.001 (n.a.) | 0.33 | 0.06 | -0.04 |
| foil_fe56_np_24.80 | 24.80 | 1.221e-29 | 2.7 (2.5) | 1.078 | 0.49 | 2.2 | 1.001 (n.a.) | 0.64 | 0.24 | -0.15 |
| foil_fe56_np_34.92 | 34.92 | 3.365e-30 | 2.8 (2.5) | 1.074 | 0.67 | 5.7 | 0.988 (n.a.) | 1.21* | 0.70 | -0.11 |
| foil_fe56_np_45.06 | 45.06 | 9.886e-31 | 3.0 (2.5) | 1.047 | 0.91 | 7.1 | 1.005 (n.a.) | 2.24 | 0.70 | -0.18 |
| foil_fe56_np_55.20 | 55.20 | 2.746e-31 | 3.2 (2.5) | 1.074 | 2.63 | 18.0 | 1.024 (n.a.) | 4.29 | 2.26 | +0.09 |
| foil_fe56_np_65.59 | 65.59 | 7.412e-32 | 3.7 (2.5) | 0.989 | 5.63 | 40.3 | 1.077 (n.a.) | 4.28* | 4.45 | -0.07 |
| foil_zr90_n2n_20.00 | 20.00 | 1.772e-28 | 2.7 (2.5) | 0.987 | 0.28 | 1.2 | 1.000 (n.a.) | 0.03 | 0.01 | +0.03 |
| foil_zr90_n2n_24.80 | 24.80 | 7.627e-29 | 2.7 (2.5) | 1.013 | 0.42 | 2.0 | 1.000 (n.a.) | 0.70 | 0.07 | +0.08 |
| foil_zr90_n2n_34.92 | 34.92 | 1.465e-29 | 2.7 (2.5) | 1.009 | 0.77 | 5.4 | 1.000 (n.a.) | 2.91 | 0.16 | +0.08 |
| foil_zr90_n2n_45.06 | 45.06 | 3.316e-30 | 3.4 (2.5) | 0.961 | 1.26 | 9.3 | 1.001 (n.a.) | 6.37 | 0.57 | +0.04 |
| foil_zr90_n2n_55.20 | 55.20 | 7.508e-31 | 4.1 (2.5) | 0.915 | 3.64 | 28.0 | 1.000 (n.a.) | 9.69 | 1.61 | +0.12 |
| foil_zr90_n2n_65.59 | 65.59 | 1.682e-31 | 3.4 (2.5) | 0.810 | 9.45 | 44.3 | 1.000 (n.a.) | 3.63* | 7.00 | +0.14 |
| foil_ni58_np_20.00 | 20.00 | 7.632e-29 | 2.7 (2.5) | 1.006 | 0.51 | 3.8 | 0.990 (n.a.) | 10.23 | 0.19 | -0.55 |
| foil_ni58_np_24.80 | 24.80 | 6.162e-29 | 2.7 (2.5) | 1.075 | 0.60 | 2.8 | 1.002 (n.a.) | 7.43 | 0.45 | -0.65 |
| foil_ni58_np_34.92 | 34.92 | 2.292e-29 | 2.8 (2.5) | 1.096 | 0.77 | 6.7 | 0.980 (n.a.) | 5.94 | 0.73 | -0.68 |
| foil_ni58_np_45.06 | 45.06 | 7.628e-30 | 3.4 (2.5) | 1.124 | 1.09 | 9.7 | 0.964 (n.a.) | 5.52 | 0.98 | -0.61 |
| foil_ni58_np_55.20 | 55.20 | 2.304e-30 | 3.1 (2.5) | 1.197 | 2.54 | 12.1 | 1.008 (n.a.) | 6.15 | 2.56 | -0.41 |
| foil_ni58_np_65.59 | 65.59 | 6.001e-31 | 4.1 (2.5) | 1.113 | 4.65 | 38.0 | 1.076 (n.a.) | 7.58* | 5.19 | -0.51 |
| T6_lg_int | - | 9.384e-24 | 2.1 (2.0) | 1.196 | 0.15 | 1.4 | 1.004 (n.a.) | 1.55 | 0.22 | +0.13 |
| U235f_int | - | 5.117e-24 | 3.3 (3.2) | 1.282 | 0.15 | 1.5 | 1.005 (n.a.) | 1.61 | 0.15 | +0.09 |
| phi_gt10_ne213_21.4 | 21.40 | 1.919e-04 | 3.9 (2.5) | 0.949 | 0.23 | 1.4 | 1.000 (n.a.) | 0.35 | 0.03 | -0.10 |
| phi_gt10_ne213_26.5 | 26.50 | 1.020e-04 | 3.9 (2.5) | 0.898 | 0.35 | 2.0 | 1.002 (n.a.) | 0.61 | 0.08 | -0.09 |
| phi_gt10_ne213_31.6 | 31.60 | 5.314e-05 | 3.9 (2.5) | 0.864 | 0.48 | 3.0 | 1.000 (n.a.) | 1.07 | 0.17 | -0.13 |
| phi_gt10_ne213_32.7 | 32.70 | 4.254e-05 | 3.9 (2.5) | 0.934 | 0.61 | 3.9 | 1.003 (n.a.) | 1.15 | 0.26 | -0.15 |
| phi_gt10_ne213_41.7 | 41.70 | 1.377e-05 | 3.9 (2.5) | 0.897 | 0.87 | 5.7 | 0.996 (n.a.) | 3.51 | 0.37 | -0.05 |
| phi_gt10_ne213_47.9 | 47.90 | 6.218e-06 | 3.9 (2.5) | 0.892 | 1.27 | 9.3 | 1.011 (n.a.) | 3.30 | 1.11 | -0.18 |
| phi_gt10_ne213_51.9 | 51.90 | 3.922e-06 | 3.9 (2.5) | 0.852 | 1.67 | 12.0 | 0.971 (n.a.) | 3.80 | 1.01 | -0.04 |
| phi_gt10_ne213_62.0 | 62.00 | 1.042e-06 | 3.9 (2.5) | 0.891 | 3.15 | 17.7 | 0.986 (n.a.) | 7.34 | 2.54 | -0.08 |
<!-- /AUTO:TABLES -->

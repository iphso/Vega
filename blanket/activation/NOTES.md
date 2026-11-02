# Activation, decay heat and contact dose (browser tier)

Per region (fw, vv_in, channel, be, vv_out, blanket, tank, shield, coil) after an operating
history (default 1 full-power year at 525 MW): activity, specific activity, decay heat, a
contact gamma dose-rate proxy, top contributing nuclides, at any cooling times (default
1 s, 1 min, 1 h, 1 d, 1 wk, 1 mo, 1 y, 10 y, 100 y). Browser code:
StoneLab `packages/physics/src/neutron/activation.ts` (branch `blanket-activation`).

## Method

1. **CE transport** (`arc_activation.py`): `arc_model.build()` unchanged, photons off. Per cell:
   the VITAMIN-J-175 track-length flux and, per group, the direct reaction rate of every nuclide
   initially present for every reaction type the chain lists for it. `--ww` adds static weight
   windows (lower bound 0.25 exp(-d/12 cm), d = distance outside the tank box) so the shield and
   coil converge. (OpenMC's MAGIC generator ran away on this model: tracks grew ~1.5x per
   secondary generation; killed.)
2. **Chain**: OpenMC's ENDF/B-VIII.0 depletion chain (`fetch_chain.sh`, 3820 nuclides,
   sha256 d4f607fc...). Per cell, pruned to nuclides reachable from the initial composition
   through <= 2 neutron reactions (on nuclides with data) and any number of decays, plus the
   light secondaries openmc.deplete adds per reaction ((n,t) -> H3, (n,a) -> He4, ...):
   15 (Be) to 346 (coil) nuclides.
3. **Sidecar** (`sidecar.py` -> `runs/<run>/activation_sidecar.json`, ~0.6 MB): per cell and
   reaction, 8-group microscopic cross sections collapsed in that cell's CE spectrum
   (sigma_c = sum_{g in c} R_g / sum_{g in c} phi_g): direct CE rates for initial nuclides
   (self-shielding included), OpenMC's flat-in-energy group averages of the pointwise data
   (`openmc.lib` `collapse_rate`, what `MicroXS.from_multigroup_flux` uses) for products.
   Coarse groups whose CE flux in that cell has > 5% relative s.d. use the infinite-dilution
   fine-group xs weighted by the all-cell summed spectrum (the "donor"; listed per cell as
   `donor_groups`), so a design that raises e.g. thermal flux in a thin VV layer doesn't meet
   a zero cross section. Plus per cell and nuclide the contact-dose coefficient.
   Coarse group edges (eV, snapped to VITAMIN-J): 19.64M | 12.2M | 4.97M | 1.00M | 98k | 961 |
   10.7 | 0.68 | 1e-5.
4. **GPU**: the transport kernel adds the region's track length into one of 8 per-region coarse
   accumulators at every tentative collision (`GpuTallies.actTrack`, on only with an activation
   sidecar). The workgroup reduction now runs in chunks of 48 accumulators (workgroup memory
   stays < 16 KiB). Same tallies otherwise (bit-identical TBR/damage), no measurable slowdown
   (2M histories: 0.87 s before, 0.87-0.89 s with the tally on, RTX).
5. **TS solve** (`computeActivation`): rate per atom = sum_c sigma_c S T_c / V with exact region
   volumes; dN/dt = A N by CRAM-48 (OpenMC's IPF coefficients), one exact step per constant-power
   interval and per cooling interval. Irradiation steps solve over the strongly connected
   components of the decay+reaction graph (block forward substitution); cooling steps use the
   decay graph alone (a DAG: pure forward substitution -- with the irradiation blocks, partial
   pivoting leaked ~1e-16 of the largest member into physically empty nuclides). Default
   history + 9 cooling times for 9 regions: 60-110 ms in Node.
6. **Contact dose** (FISPACT-II's formula, semi-infinite slab surface, B = 2):
   D [Sv/h] = 5.76e-10 * (B/2) * sum_lines (mu_en/rho)_air / (mu/rho)_mat * E[MeV] * S[photons/kg/s].
   Air mu_en/rho: NIST (`openmc.data.mass_energy_absorption_coefficient`); material mu/rho:
   total (coherent + incoherent + photoelectric + pair) from the ENDF/B-VIII.0 photoatomic
   library. Photon lines: the chain's decay sources (discrete; tabular continua as pseudo-lines).
   It is the dose at the surface of a thick slab of that material alone -- a ranking proxy, not
   a plant shutdown-dose map (no geometry, no shielding by other parts, no streaming).

## Results: ARC baseline, 1 FPY at 525 MW (OpenMC reference, round-off-cleaned)

Totals per region (`runs/VITAMIN-J-175_ww/reference.json`; "~0" = below 1e-9 of that region's
shutdown value or dominated by CRAM round-off, see Validation). Contact dose = the slab proxy.
H3 from Li-6(n,a)/(n,t) in the FLiBe and Be is counted as an unprocessed inventory (it is what
dominates channel/blanket/Be activity after a day); `computeActivation(..., { exclude: ["H3"] })`
drops it, and every region reports `tritiumBq` separately.

| region | mass (t) | Bq/kg 1 s | Bq/kg 1 d | Bq/kg 1 y | Bq/kg 100 y | W 1 s | W 1 d | W 1 y | W 100 y | Sv/h 1 s | Sv/h 1 d | Sv/h 1 y | Sv/h 10 y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fw | 41.3 | 5.7e+14 | 5.4e+14 | 3.5e+13 | ~0 | 6.9e+05 | 5.2e+05 | 1.9e+04 | ~0 | 5.6e+03 | 3.1e+03 | 53 | 0.0012 |
| vv_in | 17.6 | 4.6e+14 | 4.1e+14 | 8e+13 | 3.5e+10 | 1e+06 | 6.7e+05 | 5.9e+04 | 9 | 7.4e+04 | 5.1e+04 | 3.8e+03 | 5.9e+02 |
| channel | 8.4 | 6.2e+14 | 3.4e+14 | 3.2e+14 | 1.2e+12 | 1.6e+06 | 2.6e+03 | 2.5e+03 | 9.5 | 2.5e+05 | 1.8 | ~0 | ~0 |
| be | 4.1 | 1.3e+14 | 4.9e+12 | 4.7e+12 | 1.8e+10 | 1.3e+05 | 18 | 17 | 0.066 | ~0 | ~0 | ~0 | ~0 |
| vv_out | 54.5 | 2.6e+14 | 2.3e+14 | 4e+13 | 2.3e+10 | 1.8e+06 | 1.2e+06 | 1e+05 | 22 | 4.4e+04 | 3.1e+04 | 2.3e+03 | 3.5e+02 |
| blanket | 572.5 | 2.9e+13 | 1.5e+13 | 1.4e+13 | 5.4e+10 | 6.4e+06 | 7.9e+03 | 7.4e+03 | 28 | 1.5e+04 | 0.056 | ~0 | ~0 |
| tank | 105.7 | 1.4e+12 | 1.2e+12 | 1.8e+11 | 6e+08 | 2.2e+04 | 1.5e+04 | 1.6e+03 | 0.44 | 2.7e+02 | 2.1e+02 | 22 | 4.6 |
| shield | 498.7 | ~0 | 3.8e+10 | 1.9e+09 | ~0 | ~0 | 6.1e+03 | 1.2e+02 | ~0 | 24 | 19 | 0.34 | ~0 |
| coil | 2691.2 | 1.2e+08 | 3.9e+07 | 5.9e+06 | 1.6e+04 | 47 | 7.3 | 0.78 | 0.00012 | 0.016 | 0.0032 | 0.00044 | 4.2e-05 |

Dominant decay heat: fw W-187 / W-185 (1 s - 1 d), W-185 / W-181 (1 y); Inconel layers Co-58,
Mn-56 (1 s), Co-58 (1 d), Co-57 / Co-60 (1 y), Nb-94 / Ni-63 (100 y); FLiBe N-16, O-19 (1 s),
then H3; shield Sc-46/47/48 (Ti(n,p)); coil Mn-56, Cu-64, Co-58, Mn-54.

## Validation

`StoneLab packages/physics/scripts/neutron-activation.ts` -> `runs/validation_*.json`, against
`reference.py`: OpenMC 0.16 `IndependentOperator` on the same cells, one-group xs from the same
statepoint (direct for initial nuclides, `MicroXS.from_multigroup_flux` for the rest), the FULL
3820-nuclide chain, `PredictorIntegrator` (constant flux and xs: one exact CRAM-48 step), 1 FPY
then decay to each cooling time.

Two drives of the browser code:
- **(a) CE track**: the sidecar's own CE coarse-group track. Tests pruning, collapse, solver.
- **(b) GPU**: `GpuTransport` on the hist64 neutpack, 4 seeds x 4M histories (6 s), the coarse
  track from the new accumulators. The browser path end to end.

The reference needs cleaning: IPF CRAM's absolute error (~1e-15 x the total atoms) times huge
decay constants puts e.g. 7e16 Bq of "Be-8, W-160, Hf-156" into the TiH2 shield at 1 s (73% of
the reference total there), and "Tb-148, Ag-104, Pa-224" doses into the FLiBe and Be after a
week. Contributors not reachable from the cell's initial composition within 4 reactions +
decays are counted as round-off (`reach4` in reference.json); a point is scored when the
cleaned reference is >= 1e-9 of its shutdown value and round-off is < 50% of the raw total.
Physics outside the browser's depth-2 chain (reachable at depth 3-4) is reported separately
(`out_of_chain_physical_frac`): <= 1e-3 at every scored point.

Worst |rel| over scored points, 9 cooling times x (activity, heat, dose):

| region | (a) CE track | (b) GPU | fw_cm = 2 variant, baseline sidecar, GPU |
|---|---|---|---|
| fw | 0.0% | 1.7% (dose 1 y) | 5.4% (dose 1 y); heat 3.4% |
| vv_in | 0.0% | 0.3% | 2.4% (100 y) |
| channel | 0.0% | 0.2% | 1.1% |
| be | 0.0% | 0.1% | 2.4% (1 s) |
| vv_out | 0.0% | 0.7% (100 y) | 0.8% |
| blanket | 0.0% | 0.4% | 1.4% |
| tank | 0.0% | 0.5% | 1.7% (dose 100 y) |
| shield | 0.0% | 0.3% | 0.5% |
| coil | 11% (activity 10 y) | **+500 ... +1100% (fails)** | **fails** (CE track: 15%) |

The variant column is how the browser runs a geometry change: the baseline sidecar's cross
sections, the variant geometry's GPU track, against the variant's own OpenMC reference
(`runs/VITAMIN-J-175_fw_cm-2.0_ww`). The fw drift (+3-5%) is self-shielding held at the
baseline: the variant's own sidecar has the W-182/184/186 (n,gamma) coarse cross sections in
groups 4-5 (1 keV - 1 MeV) 6-12% lower than the 1 cm baseline's (and (n,2n) one-group values
differ through the spectrum), so the baseline sidecar over-produces W-185/W-187 in 2 cm of W.

### Investigated disagreements
- **Missing H3 (first pass, fixed)**: activity -55% (FLiBe, 1 s) to -100% (Be, FLiBe after 1 d),
  fw -88% at 100 y. openmc.deplete adds the light products of each reaction ((n,t) -> H3,
  (n,a) -> He4, ...; its `REACTIONS[...].secondaries`) which the chain XML doesn't list as
  targets. The sidecar now carries them per reaction.
- **Round-off in our own solver (fixed)**: e.g. 2.4e12 atoms of Zn-63 in the coil at 1 wk
  (should be 0): cooling steps were solved over the irradiation's strongly connected
  components, where partial pivoting leaks the largest member (Cu-63, ~1e30 atoms) into the
  others. Cooling now uses the decay-only ordering (a DAG: no pivoting). Remaining round-off is
  ~1e-22 relative; contributions below 1e-12 of the region's shutdown value are dropped
  (`floorRel`).
- **Coil, CE track (+5 ... +15%)**: the coil's coarse groups 0 and 7 have > 25% CE s.d. even with
  weight windows (3M histories), so they use the donor spectrum (infinite dilution, all-cell
  shape) rather than the coil's self-shielded own; Cu-63 / Mn-55 thermal capture carries it.
- **Coil, GPU (fails, x6-x11)**: the GPU coil track is 3x-50x the CE one by coarse group while the
  tank and shield agree to < 2%. Cause: the hist64 neutpack's coil cell has **sigma_t = 0 in 54 of
  175 groups** (mgxs_gen.py collapsed it from a CE run with almost no coil flux), so neutrons in
  those groups stream through the coil without colliding. Not an activation bug; it needs the
  neutpack regenerated with coil statistics (`arc_activation.py --ww`'s static weight windows
  gave ~10x the coil figure of merit). Until then the coil numbers on the GPU path are wrong;
  the TF fast-fluence tally probably is affected too (not checked).
- **Coil, unconverged CE without VR**: the first (analog, 2M histories) run had 50-100% s.d. in
  most coil groups; all coil numbers here come from the `_ww` runs.

### Contact-dose proxy vs photon transport (`dose_check.py`, `runs/dose_check.json`)
OpenMC photon MC of a thick slab with a uniform mono-energetic source, air kerma in a thin
surface layer, against the proxy: proxy / MC = 1.1-1.5 (Inconel), 1.4-1.8 (W, MC s.d. ~20%),
1.1-1.4 (TiH2), 0.8-1.5 (FLiBe) over 0.3-6.1 MeV. Conservative by ~20-50%, as expected from the
B = 2 build-up. (This check caught a x1000 unit error: FISPACT's C = 5.76e-10 is per MeV/kg/s,
not per g.)

### Not validated
- Against experiment. Public fusion decay-heat benchmarks exist (FNG Frascati 14 MeV decay
  heat measurements (Angelone et al.; Inconel-718, W, SS among them) and the JAERI/JAEA FNS
  decay-heat experiments (Maekawa et al., 1990s, dozens of materials); not attempted: they need
  the experiments' irradiation histories, sample geometry and detector response.
- Pulsed / multi-interval histories: the solver supports them (unit-tested against analytic
  and RK4 solutions) but only the 1 FPY constant history was compared with OpenMC.
- Material variants (fw_material, li6_pct) need their own sidecar (`arc_activation.py --ww --set
  fw_material=Be` etc. then `sidecar.py`), as the damage tier has; none generated yet.
- The chain's (n,gamma) etc. have no isomer branching for most targets (e.g. Co-59 -> Co-60 only,
  no Co-60m): same in the reference, so not seen by this comparison.
- The coarse-group collapse over spectra that differ more than fw_cm 1 -> 2 cm, end to end.
  Offline (`coarse_groups.py` -> `runs/coarse_groups.json`): the damage-tier variant runs' 175-group
  CE spectra folded with the baseline's per-group rates, exact vs 8-group, key producer reactions
  whose variant rate is known to < 2%: fast threshold reactions stay within ~1-4%, but **capture
  in a softened spectrum does not**: Co-59(n,gamma) in vv_in/vv_out -42 ... -58% when Li-6 drops
  to 7.5-30% (more thermal flux in the VV), W-180/182 capture in the fw up to 35%, Cr-50/Ni-62
  capture in the tank 25-31%; p95 over all scored reactions 17% (Gc = 4: 24%, Gc = 6: 21%; the
  tank worst case drops from 115% to 26% going 6 -> 8 groups, hence 8). Caveat on the test
  itself: its "exact" uses the baseline's per-fine-group xs, which are themselves poorly sampled
  where the baseline had little low-energy flux. Practical reading: geometry changes that keep
  the Li-6 enrichment are fine (the fw_cm test above); an enrichment change needs its own
  sidecar for every cell, not just the FLiBe, or Co-60 / long-term VV dose can be off by ~2x.

## Caveats
- FLiBe activity is computed as if the salt stayed in place for the whole year (no flow
  through the out-of-flux loop, no tritium extraction, no decay during transit).
- Coil and TiH2 smears as in arc_model (no REBCO/Cu structure, no impurities: real Inconel 718
  Co/Ta/B impurities would change long-term dose; the arc_model composition has 0.47 wt% Co).
- Light-element products' own reactions (H3(n,2n) etc.) are only as good as the chain.
- Reaction products without neutron data in the library (most radionuclides) don't burn up
  (same in the reference). Estimated <~1%: the FW fluence is ~2e22 n/cm2 per FPY, so a 1 b
  one-group cross section burns ~2% of a product over the year, and few products reach 1 b in
  this hard spectrum (not measured).

## Files
- `arc_activation.py` CE transport (+ `--ww`, `--set k=v`); `sidecar.py` sidecar JSON;
  `reference.py` OpenMC depletion; `common.py` shared (cells, volumes, pruning, dose);
  `coarse_groups.py`, `dose_check.py` checks; `fetch_chain.sh`.
- `runs/VITAMIN-J-175_ww/` the baseline: `activation_sidecar.json` (what the browser loads),
  `reference.json`, `meta.json`; `runs/VITAMIN-J-175_fw_cm-2.0_ww/` the variant;
  `runs/validation_*.json`; `runs/coarse_groups.json`, `runs/dose_check.json`. (The first, analog
  run `runs/VITAMIN-J-175/` -- coil unconverged -- is not committed.)

Re-run: `docker compose run --rm openmc blanket/activation/arc_activation.py --ww --particles
150000 --batches 20` (13 min, 12 threads), then `sidecar.py runs/VITAMIN-J-175_ww` and
`reference.py runs/VITAMIN-J-175_ww` (~1 min each), then in StoneLab
`npx vite-node scripts/neutron-activation.ts -- 4000000 4 <run dir> [<sidecar run dir>]`.


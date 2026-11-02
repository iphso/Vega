# Nuclear-data UQ of the ARC FLiBe blanket (TBR and nuclear heating)

Date: 2026-11-02. Model: `blanket/arc_model.py` defaults (W first wall, Inconel 718 VV, 90% Li-6 FLiBe,
1 cm Be multiplier). Nominal TBR 1.138, total nuclear heating 486.6 MW at 525 MW fusion (neutron
KERMA plus photon transport, every cell). Library: ENDF/B-VIII.0 (NNDC HDF5 in `blanket/xs`).
Covariances: ENDF/B-VIII.0 MF33, plus MF32 where present, processed by NJOY2016 ERRORR.

## Result

Relative 1-sigma uncertainty from nuclear data, in %. The brackets give the 68% interval
(chi-square on the sample spread). UL is a conservative 95% upper limit with no noise subtraction.

| set | samples | TBR ND rsd % | heating ND rsd % |
|---|---|---|---|
| Li-6 (MT 2, 105 only) | 10 | 0.00 [0-0.03], UL 0.07 (not resolved) | 0.017 [0.007-0.030] |
| Li-7 | 20 | 0.006 [0-0.016], UL 0.03 (not resolved) | 0.004 [0-0.008], UL 0.013 |
| Be-9 | 20 | **0.22** [0.19-0.27] | 0.09 [0.08-0.11] |
| F-19 | 20 | **0.44** [0.38-0.54] | **0.28** [0.24-0.34] |
| W-182/183/184/186 (first wall) | 10 | 0.17 [0.14-0.24] | 0.007 [0-0.03] (not resolved) |
| Ni-58 (Inconel VV, tank) | 20 | **0.57** [0.50-0.70] | 0.11 [0.10-0.14] |
| **all of the above, sampled jointly** | 30 | **0.67** [0.59-0.78], UL 0.88 | **0.39** [0.34-0.45], UL 0.51 |
| quadrature sum of the single-nuclide sets | | 0.78 | 0.32 |

The joint set and the quadrature sum agree within their sampling uncertainty. A 30-sample
standard deviation is itself uncertain by about ±13%. In absolute terms: **TBR = 1.138 ± 0.008
(ND, covered nuclides only) ± 0.002 (MC, single 300k run)**, and **heating = 487 ± 2 MW (ND)**.

Per-cell heating ND rsd from the joint set (`results.json` → `heating_by_cell`): FW 0.8%, inner VV
2.5%, channel 1.2%, Be 1.2%, outer VV 1.9%, blanket 1.0%, tank 4%, shield 5%, coil 21%. The coil
value is noisy: little heating reaches it and it is set by deep-penetration F-19 transport.

Indicative drivers (`attribution.json`). These are correlations of each sample's change with its mean
fast-range factor; with 20 samples, |r| below about 0.45 is noise:
- F-19 TBR: (n,α) MT107 (r = -0.80). ENDF/B-VIII.0 gives it 30% at 14 MeV.
- F-19 heating: (n,n'α) MT22 and elastic.
- Ni-58 TBR: (n,p) MT103 (r = -0.86, 18% at 13-15 MeV in this file) and inelastic MT4 (30% at
  14 MeV). The Inconel VV sits between plasma and blanket, so fast absorption there costs breeding
  directly.
- Be-9: no single reaction dominates in 20 samples. (n,2n) MT16 and (n,α) MT800 lead.
- Li-7 is negligible for a different reason. It produces only 0.67% of the TBR at 90% Li-6:
  Li-7 (n,Xt) is 0.0076 of 1.138 (`check_li7_share.py`), and scaling its n'αt levels ×1.5 raised
  that by 49%, so the perturbation does reach the tally. Li-6 has covariances only for elastic
  and (n,t), and (n,t) is a standard at 0.5%. At 90% enrichment almost every slow neutron is
  absorbed by Li-6 anyway: d ln TBR / d ln σ(n,t) = 0.07.

### Comparison with published values
- Aimetta, Abrate, Dulla, Froio, *Nuclear data uncertainty quantification in the ARC fusion
  reactor*, PHYSOR 2022, DOI 10.13182/PHYSOR22-37861 ([preprint](https://iris.polito.it/retrieve/handle/11583/2970426/608483)).
  This is the same ARC concept, with Serpent 2 and SANDY-sampled ENDF/B-VIII.0 covariances, 500 fast-TMC
  samples. Their TBR ND rsd: Li-6 0.017%, Be-9 0.263%, F-19 0.525%. Ours: Li-6 below 0.07%, Be-9
  0.22%, F-19 0.44%, so agreement is within about 20%. Volumetric power, Be-9 contribution: their
  neutron multiplier is 1.60%, channel 0.33%, blanket 0.16%; ours is 1.66%, 0.35%, 0.20%. F-19
  contribution: theirs channel 1.54%, blanket 0.89%; ours 1.16%, 0.58%. They did not treat Li-7, W
  or Ni. **Our two largest non-FLiBe terms (Ni-58 0.57%, W 0.17%) are absent from their
  study.** Their geometry and TBR (1.085) differ from ours.
- Park & Pereslavtsev (KIT INR, poster, *Statistical analysis of TBR deviations in the DEMO due
  to nuclear data uncertainties*, [link](https://publikationen.bibliothek.kit.edu/1000098785/42934649)).
  HCPB/WCLL DEMO with TENDL-2017 random files and no Li/Be: about ±2.5-2.9%. The spread between
  libraries (ENDF/B-VIII vs JEFF/FENDL/TENDL) is above ±1.2%. That is a different concept, with
  Pb/Eurofer-dominated uncertainty and much larger TENDL covariances, so it is context only.
- Youssef & Abdou (Fusion Technol., 1986) estimated a 2-6% TBR uncertainty for older evaluations.
  The Be-9 (n,2n) energy-angle distribution alone was about 4% in FLiBe. That secondary-distribution
  term is not sampled here; see the caveats.

## Method
1. **Evaluations.** We used ENDF/B-VIII.0 neutron files from NNDC (`ENDF-B-VIII.0_neutrons.zip`), kept outside git
   in `~/runs/blanket/uq/endf`. MF33 is present for all nine nuclides. Li-6 MF33 covers MT 2 and
   105 only.
2. **`njoy_cov.py`.** RECONR, then ERRORR (iwt=6, irelco=1, MF33 plus MF32), on 35 groups. Groups are
   coarse below 100 keV, 0.5-1 MeV wide up to 13 MeV, and 0.5 MeV wide around 14 MeV.
3. **`prep.py` / `uqlib.py`.** These map covariance MTs onto the reactions OpenMC transports:
   - An own-MT covariance scales that reaction.
   - A lumped covariance (851-870, from the MF33 MTL header) scales all its constituents. For Li-7,
     levels 52-82 carry the n'αt tritium (MT205 = Σ MT52-82).
   - A summation MT (4, 103-107) scales components that have no covariance of their own. For example,
     F-19 MT4 scales levels 51-71 and 91.
   - MT1 and MT3 are not sampled.

   The joint relative covariance includes all cross-MT blocks. Relative sd is capped at 100% (a few
   near-threshold groups reached 10^3-10^5 %). The matrix is projected to PSD; the negative-eigenvalue
   fraction is at most 1.9% (Ni-58). Factors are lognormal with mean 1 and the target covariance
   (moment-matched: S = ln(1+C)), so cross sections never go negative.
4. **Perturbed HDF5.** `IncidentNeutron.from_hdf5` is scaled per (MT, group) at every temperature,
   including 0 K elastic. OpenMC rebuilds total, absorption and photon production from the
   non-redundant partials at load (`Nuclide::create_derived`). The stored redundant data that the
   tallies read are rebuilt explicitly:
   - MT 4 and 103-107 as sums of their components;
   - gas production MT 203-207 from exact integer multiplicities fitted to the library. Fit error is
     at most 1e-6 for Li/Be/F. The Ni/W fits fail because their MT5 yields depend on energy; those
     are left untouched, and no tally uses them.
   - **heating KERMA MT301** as `old + Σ_r k_r(E) Δσ_r(E)`, with per-reaction k_r = K_r/σ_r from NJOY
     HEATR partial KERMAs (local=0, matching OpenMC's MT301 with photon transport). Level MTs 600-849
     use their sum's partial KERMA. Check: the sum of partials matches the library MT301 at 14 MeV
     to 0.1% for Be, F and W, 0.6% for Li-7 and 3% for Ni-58. Li-6's sampled partials (elastic and n,t) are 26% of
     its KERMA. URR probability tables are multiplicative (W), so the factors carry into them.
5. **Transport.** `run_case.py` runs the arc_model baseline with a material-filtered TBR tally and
   an unfiltered heating tally, so each total's statistical std is exact. Runs use 150k histories in
   15 batches, 8 threads, about 50 s each.
6. **Noise handling: correlated sampling.** OpenMC seeds every history from its index, so for
   modest perturbations the histories of a perturbed run track the nominal run at the same seed.
   Every sample and the nominal were run at the same two seeds (777 and 778):
   - D_i(s) = Y_i(s) - Y_nom(s);
   - var_ND = var_i(mean_s D_i) - var_i(D_A - D_B)/4.

   The residual noise per sample is 0.02-0.16% on TBR, against 0.25% for an independent run, so 20
   samples resolve contributions down to about 0.05%. The plain uncorrelated estimator (spread at
   one seed minus the mean batch variance) is also reported. It agrees wherever it can resolve
   anything: F-19 0.42 vs 0.44, Ni-58 0.54 vs 0.57, joint 0.63 vs 0.67.
7. **Sanity checks** (300k histories, seed 777; `run_checks.sh`):
   - Li-6/Li-7/Be-9/F-19 rewritten through the pipeline with unity factors give TBR 1.1383293849,
     identical to the original library to 1e-15.
   - Li-6 (n,t) ×1.10 in every group gives TBR +0.705%. The direction is right, and the small
     sensitivity (0.07) is expected at 90% enrichment.
   - Li-7 levels 52-82 ×1.5 raise Li-7 (n,Xt) by 49%.
   - `check_perturb.py` confirms MT203-207 equal the sum of their perturbed components to 1e-6, and
     that the sampled factor std matches the covariance (median ratio 1.00, 5-95% range 0.97-1.03).

Compute: 3.6 h of wall time for the 262 production runs, 0.5 h of checks, and about 15 min of
NJOY ERRORR/HEATR, on 8 threads of a shared machine.

## Caveats / not included
- **Only nine nuclides were sampled.** Inconel also has Ni-60/61/62/64, Cr, Fe, Nb, Mo and Ti. Ni-58
  alone gives 0.57% on TBR, so the rest of the Inconel probably matters at a similar level. The
  TiH2 shield, the coil and W-180 are also unsampled. **The 0.67% is therefore a lower bound on the
  ENDF/B-VIII.0 covariance-propagated uncertainty, not the total.**
- **Secondary distributions were not sampled.** Angular (MF34, present for W), energy (MF35) and
  multiplicity uncertainties are left out. These include the Be-9 (n,2n) energy-angle distribution,
  which Youssef & Abdou found to dominate older FLiBe estimates.
- **Missing covariances in ENDF/B-VIII.0.** Li-6 has none for (n,n')/(n,n'd) continuum (MT 24,
  51-81), capture or (n,p). Ni-58 has none for MT 5, 105, 111 or 112.
- **Ni-58 covariances are large.** ENDF/B-VIII.0 gives (n,p) 18% and inelastic MT4 30% at 14 MeV;
  dosimetry-grade (n,p) data (IRDFF) are much tighter. The Ni-58 term reflects the covariance file
  as distributed and may be conservative.
- **Group structure.** Factors are flat within each of 35 groups, which cannot represent
  fine-structure resonance-parameter uncertainty. MF32 is folded into the group covariance by ERRORR.
- **Covariance repairs.** Relative sd is capped at 100% and the matrices are PSD-projected (F-19 1%,
  Ni-58 1.9% of eigenvalue mass removed). The lognormal transform slightly reshapes very large
  uncertainties.
- **KERMA consistency.** The KERMA update assumes per-reaction k_r(E) is unchanged by the
  perturbation, i.e. only σ moves. Photon-production yields per reaction scale with σ automatically.
  MT444 (damage) and MT901 (local heating) are not updated; neither is used when photons are
  transported.
- **Sample counts.** 10-30 samples per set leave the std itself uncertain by ±13-24% (68%
  intervals are in the table). The W and Li-6 sets have 10 samples.
- **Tallies.** Heating is neutron plus photon energy deposition in the modelled cells. Alpha
  heating and decay heat are excluded. TBR is (n,Xt) in FLiBe.

## Files
- Pipeline:
  - `njoy_cov.py`: ERRORR
  - `covlib.py`: ERRORR tape parser
  - `prep.py`: sampling plan, HEATR partials, gas multiplicities
  - `uqlib.py`: sampler and perturbed-HDF5 writer
  - `run_case.py`
  - `sample_driver.py`
  - `run_checks.sh`, `run_campaign.sh`, `run_extend.sh`
- Analysis:
  - `analyze.py` → `results.json`
  - `attribution.py` → `attribution.json`
- Checks: `check_perturb.py`, `check_li7_share.py`
- Data:
  - `raw/results/*.jsonl`: every run, with the transport seed. Sample factors are reproducible from
    the base set name (e.g. `f19`, shared by the A and B seeds) and the index:
    `numpy.default_rng([crc32(name) % 100000, i])`.
  - `raw/plans/*.json`: the per-nuclide MT mapping, unsampled reactions and fits.
- Kept outside git, in `~/runs/blanket/uq/`: ENDF files, ERRORR tapes and covariance matrices
  (`cov/<nuc>/plan.npz`), and the per-sample factor `.npz` files. Perturbed libraries are deleted
  after each run.

Reproduce, from ~/Projects/Vega:

```
docker compose run --rm -v ~/runs/blanket/uq:/runs openmc blanket/uq/njoy_cov.py
```

then `prep.py` the same way, then `run_checks.sh`, `run_campaign.sh`, `run_extend.sh` and
`python3 blanket/uq/analyze.py`.

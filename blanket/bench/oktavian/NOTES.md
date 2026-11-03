# OKTAVIAN sphere benchmarks: OpenMC / ENDF/B-VIII.0

Date: 2026-11-03. Code: OpenMC 0.16.0, continuous energy. Library: ENDF/B-VIII.0 (NNDC HDF5 in `blanket/xs`),
plus ENDF/B-VIII.0 thermal scattering for Be metal and reactor graphite (30% porosity), processed here with
NJOY because `blanket/xs` has no S(a,b) data (`make_sab.py`, 296 K). No photon transport.

Three groups of experiments are modelled, all from the OKTAVIAN 14 MeV facility (Osaka University), as compiled
in JAERI-M 94-014 (FENDL-2 benchmark sublibrary, `blanket/refs/lit/jaerim94014/.../oktavian`):

| group | cases | measured quantity |
|---|---|---|
| TBR spheres (Cetnar, Iguchi, Nakazawa, Sugiyama, Takahashi, Sumita) | `tbr_li`, `tbr_lic`, `tbr_pbli`, `tbr_pblic`, `tbr_beli`, `tbr_belic` | integral TBR of the Li zone; local 6Li/7Li/Al reaction rates per atom (Li2CO3 pellets, Li and Pb-Li); local TPR per cm3 of Li metal (LiF TLD pairs, the other four) |
| Be and Be+Li leakage (Murakami et al., 1991) | `be57`, `be69`, `be97`, `be128`, `beli57`, `beli128` | TOF leakage spectrum per source neutron per MeV |
| LiF and W leakage (Ichihara et al., 1984-88) | `lif60`, `w40` | TOF leakage spectrum per source neutron per unit lethargy |

Outputs: `results.json` (schema below), `spectra.json` (fine-bin C and E for the 8 leakage cases, total
leakage and the 55-degree cone), raw run records in `~/runs/blanket/bench/oktavian/results/*.jsonl`.

## Geometry and materials (with sources)

Sources were found in the open literature by a search on 2026-11-03; PDFs and text extracts are in
`blanket/refs/lit/oktavian_src/`. Page numbers are the printed numbers.

### TBR spheres
- **Assembly.** One 120 cm OD natural-Li-metal sphere in SUS316 vessels built from blocks (Sugiyama et al.,
  NEACRP-A-1984-0657 pp. 2-3; Takahashi et al., JAERI-M 91-062 §5.3, p. 83 and Figs. 1-3, pp. 88-89):
  - A-block: Li shell, vessel 20-60 cm, outer wall 5 mm, inner wall 3 mm (1984 paper; the 1991 paper says 2 mm).
  - C-block: Li shell, vessel 10-20 cm, 2 mm walls. In the multiplier systems it is replaced by Pb 10-20 cm
    or Be 5.7-17.35 cm plus a 2.65 cm gap ("Gap; (2.65cm)", Fig. 3).
  - D-block: graphite 60-80 cm. **The reflector is 20 cm**, as in Figs. 2-3, Table 1.1 and Cetnar's text.
    The compilation's TLD table titles "Pb(10)-Li(40)-C(10)" and "Be(10)-Li(40)-C(10)" are labelling errors.
    Cetnar's own Fig. 10 legend already says "Be(10)" for the 11.65 cm shell.
  - Li(50) = C-block plus A-block, Li 10.2-19.8 cm and 20.3-59.5 cm.
- **Li density 0.52 g/cm3** (natural Li). Not stated explicitly anywhere found. 0.534 is nominal Li metal, and
  the block weights and filling factors (98.6% A, 98.1% C) give about 0.49-0.51 after subtracting walls. 1 sigma
  is taken as 4%. This is the dominant geometry uncertainty of every TBR.
- **Be.** These are the combined US/PRC Be shells (Chuvilin et al., INDC(CCP)-401, Tables 1-2):
  - PRC shells 5.7-12.8 cm: 1.82 g/cm3, Be 98.9 / O 0.5 / Fe 0.6 wt%.
  - US shells 12.8-17.35 cm: 1.84 g/cm3, Be 98.82 / O 0.72 / Fe 0.13 / C 0.11 wt%.
  - 1 sigma 2% on density.
- **Pb 11.34 and graphite 1.70 g/cm3: not found.** Pb is nominal metal (1 sigma 5% for block packing). Graphite
  is a typical reactor-grade block density (1 sigma 5%).
- **Not modelled (1D):**
  - the beam duct, which is plugged with Li (B-block) or graphite (E-block);
  - the target assembly and Nb cap;
  - five 1 mm SS tubes for the pellets;
  - the experimental hall.
- **Source.** Isotropic point source.
  - Li and Pb-Li pellet runs (≤1987, pulsed beam line): n-leak Table 4.2 spectrum.
  - TLD runs (1990): beli Table 2.1, the 1991 D-T source.
- **Detectors.** Pellets and TLDs are dilute detectors: the flux in 0.25 cm spherical shells is folded with
  pointwise σ(E) through an `EnergyFunctionFilter`:
  - Li6 MT105, Li7 MT205 (all tritium production), Al27 MT107.
  - In perturbed runs, σ comes from the sample's own Li6/Li7 files.
  - TPR = N6·R6 + N7·R7 with the Li-metal atom densities.
  - TBR is an `H3-production` tally over both Li cells.

### Be and Be+Li leakage
- Bare Be shells, inner radius 5.7/6.9/9.7/12.8 cm and outer 17.35 cm, built from the same US/PRC shells
  (densities as above). JAERI-Data/Code 98-024 p. 11 confirms "solid metal" shells. The beam channel through the
  shells is not modelled.
- beli57/beli128 put that Be inside the A-block Li shell (20-60 cm, SUS316). "Be (116.5 mm) + Li (400 mm)" is
  the same assembly as `tbr_beli`.
- Source: beli Table 2.1. Tally: total current through the outer surface.

### LiF and W leakage
- **W 40 cm.** Type-II vessel, surface by surface from the compilation's MCNP input (Fig. 4.21):
  - void r < 10 cm, steel 10.0-10.2, W powder 10.2-19.75, steel 19.75-19.95;
  - 11 cm re-entrant duct with a 0.2 cm wall;
  - W 4.43 g/cm3; SUS-304 Cr 18.5 / Fe 70.4 / Ni 11.1, 7.86 g/cm3 (SINBAD NEA-1553/53 abstract).
- **LiF 61 cm.** Type-I vessel, from the MCNP model of the same vessel type in JAERI-Data/Code 98-024 Fig. A-15:
  - re-entrant duct r 2.5 cm with a 0.3 cm wall, closed 2.5 cm past the centre, no central void;
  - shell 30.0-30.5 cm, steel 7.87 g/cm3;
  - LiF 1.79 g/cm3, 6Li 7.463 at% (lif60.dat).
- Source: n-leak Table 4.2 (the stated source for LiF and W), isotropic, as in the compilation's analysis.
- Tally: total leakage current over the sphere. The measurement was TOF at 55° to the beam. `C_cone55` gives
  4π times the leakage per steradian into a 120-130° polar cone (the beam travels down the duct) as a check
  of the isotropy assumption.

### Data repairs (`okt_data.py`, `tbr_tables.py`)
- **Typing errors:**
  - beli Table 2.1 "9.640e+00" → 0.964 MeV.
  - Table 2.2 "2.250e+00" after 2.251 → 2.058.
  - OCR letters in n-leak Table 4.2.
- **Be97 energy column (Table 2.4) is corrupt** at 15.5-15.1 MeV (non-monotonic). Those bins were re-spaced only
  to keep the parser consistent, and **the be97 >10 MeV band is excluded**.
- **Excluded: Li(50) 6Li pellet at 63.3 cm.** This is outside the 60 cm sphere. Its measured rate is 1/7 of the
  calculated leakage flux × σ there, which no geometry can produce from a bare sphere. Most likely a position or
  labelling problem.
- **Table 1.12 (Li-C) "TLDD exp" column** is a copy of Table 1.9's Be-Li values. Only TPR is used.

## Uncertainty model in results.json
- **`E_rel_unc`.** The compilation's total.
  - Pellet tables: statistics plus source strength. The source-strength part (5.3%) is common within a case;
    `exp_correlations` rebuilds the case matrix from it.
  - TLD TPRs: the published TPR rsd and correlation matrices (Tables 1.5, 1.8, 1.11, 1.14), including response
    function, cross-section and spectrum terms.
  - TBR: Table 1.1 RSD.
  - Be leakage bands: quadrature sum of bin statistics plus a 7.6% normalisation (the stated 6.7-8.4%), fully
    correlated within a case.
  - **LiF/W bands: statistics plus the Nb monitor (≤1%) only.** The compilation gives no detector-efficiency
    systematic. Each response carries an `E_unc_note`. Do not assimilate these at ~1%: TOF efficiency systematics
    of OKTAVIAN-type measurements are typically several %, so add an efficiency term before use.
- **`geom_rel_unc`.** Quadrature sum of 1-sigma geometry and material changes, from correlated-seed runs
  (`sens_*` against `sens_ref`), plus a detector position ±0.2 cm term from the local radial gradient. The
  per-parameter breakdown is in `results.json → geometry_sensitivity`:
  - Li density ±4%;
  - A-block inner wall 2 vs 3 mm (half the difference);
  - graphite density 5%; Pb density 5%; Be density 2%;
  - inner cavity radius ±0.5 cm (the run moves it by 1 cm);
  - LiF/W powder density 2%.

  Responses within about 1 cm of a material interface carry very large values, from the position term alone: Be-Li
  TPR at 20.3/21.4 cm (35-86%), and points at 58.2-59.7 cm next to the graphite. They are kept but should be
  given that weight.
- **`C_mc_rel_unc`.** From the 5×10^6-history nominal run (tag `nomhi`).

## Results (nominal, 5×10^6 histories per case)

### TBR spheres: integral TBR

| case | E (rsd) | C | C/E | geom 1σ | JENDL-3/ANISN C/E (Table 1.1) | idealised model C |
|---|---|---|---|---|---|---|
| Li(50) | 0.685 (5.6%) | 0.624 | **0.911** | 4.2% | 0.98 | 0.659 |
| Li(50)+C(20) | 1.13 (16%) | 0.976 | **0.864** | 3.2% | 0.91 | 1.031 |
| Pb(10)+Li(40) | 0.530 (6.0%) | 0.480 | **0.906** | 5.2% | 0.99 | 0.483 |
| Pb(10)+Li(40)+C(20) | 1.16 (8.5%) | 1.034 | **0.892** | 2.8% | 0.96 | 1.077 |
| Be(11.65)+Li(40) | 0.848 (16%) | 0.900 | **1.061** | 3.3% | 1.10 | 0.913 |
| Be(11.65)+Li(40)+C(20) | 1.51 (9.1%) | 1.365 | **0.904** | 2.0% | 0.96 | 1.426 |

MC uncertainty is at most 0.05% everywhere. The "idealised model" column is a diagnostic only (tag `diag_ideal`):
no SS316 vessel walls and Li at the full 0.534 g/cm3.
- It lands 0-3% below the published JENDL-3/ANISN calculated TBRs (0.672, 1.03, 1.11, 0.935, 1.45) and 8% below for
  Pb-Li (0.483 vs 0.527). Takahashi et al. give 0.501 for Pb-Li with JENDL-3 (JAERI-M 91-062 p. 87 Table 2) against Cetnar's 0.527.
- So the model reproduces the published analysis when given the same simplifications. Our lower C/E comes from
  two real geometry features: about 1 cm of total steel wall and the 98% filling.
- These were not tuned. With the documented geometry the TBRs sit 9-14% below experiment, except Be-Li. Every
  difference is within about 2σ of the experimental error.

### TBR spheres: local responses
- **Li(50), pellets (Table 1.15).** 7Li(n,n'α)t C/E is 0.96-1.04 at every radius, and 27Al(n,α) is 0.88-0.97.
  The fast flux and the Li-7 tritium production are right.
  - **6Li(n,α)t is 0.75-0.79 from 10 to 41 cm, falling to 0.71 at 52 cm and 0.54 at 57.7 cm.**
  - So the TBR deficit is entirely low-energy, and it grows toward the outer surface.
  - Interpretation: the 1D model has no room return. Thermal and epithermal neutrons scattered back from the
    concrete target hall are absorbed in the outer Li, which matches the radial trend. It also lacks the target
    assembly (cooling water, if any) and the Li-filled beam duct and plug.
  - A 6Li cross-section error is unlikely. Li-6(n,t) is a standard, and the ND sampling below moves rr6 by only 3%.
- **Pb(10)+Li(40), pellets (Table 1.16).** Al27(n,α) is 0.81-0.84 at 10-30 cm and 7Li 0.86-0.88 at 10-20 cm. The
  fast flux behind the Pb is about 15% low, and the deficit shrinks to 0.91-1.0 at the back.
  - The same Al rate at r = 10.0 is 0.88 in Li(50), where it is dominated by uncollided 14 MeV flux. Part of
    this is therefore common to the pellet runs (old pulsed beam line, source-strength rsd 5.3%, "angular
    distribution of source neutron increases errors of strength estimation").
  - The rest (about 0.83 vs 0.88) points at Pb transport or the unknown Pb density. Pb density −5% raises 7Li at
    41 cm by 4.6%. **Pb density is not documented**; 11.34 was assumed.
- **TLD TPRs.** In the Li zone away from interfaces, C/E is mostly 0.85-1.15. The shape follows the published
  JENDL-3 C/E with an offset of about −5%, the same wall and density effect as above. Be-Li has a high tail at
  41-57 cm (1.15-1.32), the same as JENDL-3's 1.22-1.41.
  - Interface points (Be-Li 20.3/21.4 cm, C-side 58.2-59.7 cm) are 0.58-0.85. Their position sensitivities are
    35-86%. They sit within 1-3 mm of a steel wall or the Li/graphite boundary, where the TPR changes by up to
    10× per cm. Our model puts the A-block inner wall at 20.0-20.3 cm, so "20.3" is at the wall/Li face.
  - JENDL-3 ANISN gets 1.2 at 20.3 cm, presumably without the wall. Do not assimilate these points without their
    `geom_rel_unc`.
- The graphite reflector thickness is now unambiguous. With 10 cm instead of 20 cm the TBR would drop 11-19%
  (`sens_c10`), which would put Li-C, Pb-Li-C and Be-Li-C at about 0.75-0.80. The 20 cm drawings are confirmed.

### Be and Be-Li leakage (per source neutron, bands)

| case | >10 MeV | 3-10 | 1-3 | 0.1-1 | 0.01-0.1 |
|---|---|---|---|---|---|
| Be 116.5 mm (be57) | 1.20 | 0.91 | 0.85 | 0.87 | 1.16 |
| Be 104.5 mm (be69) | 1.26 | 0.84 | 0.80 | 0.85 | 1.24 |
| Be 76.5 mm (be97) | excluded | 0.74 | 0.69 | 0.75 | 1.02 |
| Be 45.5 mm (be128) | 0.98 | 1.03 | 1.02 | 0.95 | 1.10 |
| Be 116.5 + Li 40 (beli57) | 1.25 | 0.94 | 0.86 | 0.99 | 1.30 |
| Be 45.5 + Li 40 (beli128) | 1.12 | 0.75 | 0.79 | 0.95 | 1.08 |

Experimental uncertainty is 7.6-8.8% per band, mostly normalisation, and correlated within a case. Geometry is
0-3% on bare Be and up to 5.5% on Be-Li (Li density).
- **Published comparison** (JAERI-Data/Code 98-024 Table 5.4, be57; FENDL/E-1.0 / JENDL-3.2):
  - >10 MeV: 1.35 / 1.35; ours 1.20.
  - 5-10 MeV: 0.89 / 0.95.
  - 1-5 MeV: 0.85 / 0.99.
  - 0.1-1 MeV: 0.92 / 1.04.
  - Same pattern as ours: high-energy excess, mid-energy deficit. A model or data problem would not flip sign
    between bands.
- **Over-prediction above 10 MeV for the thick shells but not the thin one** points to the OKTAVIAN Be data
  rather than the evaluation:
  - The be57/be69/be97 energy scales put the 14 MeV peak at about 15 MeV, while be128 and the source share one
    binning.
  - JAERI-Conf 96-005 pp. 81-83 documents that the OKTAVIAN Be spectra are inconsistent with the KANT and IPPE Be
    sphere data.
- **be97 is internally inconsistent.** Its measured 1-3 and 3-10 MeV leakage exceeds that of the thicker be57
  shell, which is physically backwards for a thinner multiplier (calculation: 0.19 vs 0.19-0.20). Its >10 MeV
  energy column is corrupt. Treat be97 as suspect; down-weight it or exclude it.

### LiF 61 cm and W 40 cm leakage

| case | 10-20 MeV | 3-10 | 1-3 | 0.1-1 |
|---|---|---|---|---|
| LiF this work (ENDF/B-VIII.0) | 1.064 | 0.862 | 0.843 | 0.762 |
| LiF JENDL-3.2 / FENDL/MC-1 (98-024 T5.1) | 1.137 / 1.076 | 0.932 / 0.834 (5-10) | 0.856 / 0.856 (1-5) | 0.597 / 0.680 |
| LiF JEFF-3.1 (JEFDOC-1106) | 1.09 | 0.88 (5-10) | 0.89 (1-5) | 0.76 |
| W this work | 0.912 | 0.846 | 0.855 | 0.898 |
| W JENDL-3.2 / FENDL/MC-1 | 0.906 / 0.920 | 0.604 / 0.720 (5-10) | 0.788 / 0.879 (1-5) | 1.003 / 0.843 |
| W JEFF-3.1 / FENDL NDS 2024 mean | 0.91 / – | 0.64 / 0.73 (5-10) | 0.76 / 0.92 (1-5) | 0.97 / 0.89 |

These agree with every independent MCNP analysis to within the library spread.
- The 55° cone variant differs from total leakage by at most 3% (LiF, 10-20 MeV) and 1.3% (W). The isotropic total-leakage
  convention is adequate.
- The 10-20 MeV value of 3.3 in Schnabel et al. 2024 Table XVII is not comparable; their band is evidently defined
  differently.
- The remaining 15-25% deficits at 0.1-3 MeV (LiF) and about 10-15% below 10 MeV (W) are the long-known OKTAVIAN
  pattern. They are probably partly detector efficiency, for which the compilation gives no systematic.

## ND-perturbed runs (30 joint "all" samples)

- **Libraries.** The UQ campaign deleted its libraries. They were rebuilt from the saved factors
  (`gen_libs.py` → `~/runs/blanket/bench/oktavian/libs/<i>/`). A/B factors are identical, and the rebuilt KERMA
  diagnostics equal the campaign's to 0.
- **Sampled nuclides.** Li-6, Li-7, Be-9, F-19, Ni-58, W-182/183/184/186. The Ni-58 enters here through the steel.
  The detector σ for 6Li/7Li comes from each sample's file.
- **Runs.** Every sample, and the nominal library, ran at seeds 777 (A) and 778 (B) with 6×10^5 histories per case,
  14 cases per run.
- **Estimator.** D_i(s) = Y_i(s)/Y_nom(s) − 1; ND variance = var(mean_s D_i) − var(D_A − D_B)/4, as in
  `blanket/uq`.

**ND spread is resolved above MC noise on every response.** The per-sample residual noise is 0.01-0.3% against
ND rsd 0.4-6.6%. The weakest is the Pb-Li-C TPR at 59.4 cm, 0.54% vs 0.25%. Full table: `results.json →
samples.all.nd_rel_sd / noise_rel_sd_per_sample`.

Typical ND 1σ:

| response | ND rsd | response | ND rsd |
|---|---|---|---|
| TBR Li / Li-C / Pb-Li / Pb-Li-C / Be-Li / Be-Li-C | 1.04 / 1.01 / 0.88 / 0.47 / 0.89 / 0.63% | Li(50) rr6 / rr7 (41 cm) | 3.2 / 2.0% |
| Be leakage >10 MeV / 1-3 / 0.1-1 MeV (be57) | 1.5 / 1.9 / 2.4% | Be-Li leakage >10 MeV / <0.1 MeV (beli57) | 3.1 / 3.5% |
| LiF 10-20 / 3-10 / 1-3 / 0.1-1 MeV | 6.6 / 3.9 / 3.0 / 4.3% | W 10-20 / 3-10 / 1-3 / 0.1-1 MeV | 0.5 / 1.5 / 1.1 / 1.5% |

The Al27(n,α) responses are a control. Al is not perturbed, so they move only through transport in the
perturbed Li, Fe and Ni: 0.00-0.05% at the inner radii and up to 2.8% deep in the Li. This is as expected.

Correlation with the same samples' ARC TBR (`corr_with_arc_tbr`):
- With 30 samples, |r| below about 0.36 is consistent with zero.
- Only LiF >10 MeV (+0.45), W 0.1-1 MeV (+0.37) and W >10 MeV (−0.36) reach that level. These are the F-19 and W
  channels, which drive the ARC spread.
- The TBR spheres are weakly anticorrelated with ARC TBR (−0.1 to −0.3). Their ND spread is set by Li-7 and Be-9,
  which barely move ARC TBR at 90% Li-6.
- In this sample set, then, the OKTAVIAN data constrain the ARC TBR mainly through the LiF (F-19) and W spectra.
  For Li-7/Be-9 they constrain data the ARC TBR is insensitive to.
- Experimental errors (5-16% on TBR, about 8% on Be bands) are much larger than the ND spread (≤1% on TBR), so the
  TBR spheres carry little weight in assimilation. The LiF high-energy bands have a large ND spread (4-7%) and
  small quoted errors, but see the efficiency caveat.

## Caveats
- **1D models.** The TBR spheres omit the beam duct and plugs, the target assembly, the pellet tubes and the hall.
  Room return is the most likely cause of the outer-radius 6Li deficit.
- **Densities not found:** Li (0.52 inferred), Pb (11.34) and graphite (1.70). Li density dominates the TBR
  geometry uncertainty, at about 4%.
- **Source spectra are borrowed.** The TBR readme gives none. The pellet runs use n-leak Table 4.2 (the 1987 pulsed
  line) and the TLD runs use beli Table 2.1. Source anisotropy is ignored, as in the compilation's analyses.
- **Exp TBRs are not pure measurements.** They are TPR integrals assisted by a JENDL-3 calculated shape, and the TLD
  TPRs were unfolded with a calculated spectrum (prior rsd 80%). They carry some JENDL-3 model dependence.
- **No photon transport.** Not needed: the TLD-difference TPRs are neutron-only by construction.
- **S(a,b) temperature.** S(a,b) at 296 K with neutron data at 294 K.
- **n-leak experimental uncertainties exclude detector efficiency.** See `E_unc_note`.

## Compute
- About 2 h 20 min wall time on 12 threads (10:28-12:48) for the production campaign: nominal, 9 sensitivity sets
  and 62 sample runs.
- About 15 min more for tests, the idealised diagnostic and library/S(a,b) generation.
- Nothing was skipped.

## Files and reproduction
- **Model and pipeline:**
  - `okt.py`: model, sources, tallies.
  - `okt_data.py` → `data.json`: parsed tables and repairs.
  - `tbr_tables.py`: hand-transcribed TBR tables.
  - `responses.py`: measured quantities and experimental correlations.
  - `run_bench.py`: one library, all cases.
  - `run_campaign.sh`.
  - `analyze.py` → `results.json`, `spectra.json`.
  - `gen_libs.py`, `make_sab.py`, `dc.sh` (docker wrapper; mounts `~/runs/blanket/bench/oktavian` at `/bench` and
    `~/runs/blanket/uq` read-only at `/uq`).
- **Reproduce:**
  `blanket/bench/oktavian/dc.sh blanket/bench/oktavian/make_sab.py` (after placing the ENDF TSL files, see the
  script), then `dc.sh .../gen_libs.py`, `blanket/bench/oktavian/run_campaign.sh`, and
  `T=2 dc.sh blanket/bench/oktavian/analyze.py --raw /bench/results`.
- **Literature** used for the geometry is in `blanket/refs/lit/oktavian_src/` (not committed with this directory).

## results.json schema notes
- `cases.<id>.responses[]` follows the requested schema. Extras:
  - `CE_jendl3_published`: Table 1.1 or TLD-table JENDL-3 C over our E.
  - `C_cone55`: leakage, 55° detector direction.
  - `E_unc_note`.
- `exp_correlations` contains:
  - the published TPR matrices for the TLD cases;
  - source-strength-correlated matrices for the pellet cases (TBR, rr6, rr7, Al together);
  - normalisation-correlated band matrices for the Be cases.
- `samples.all`:
  - `A[k][j]` / `B[k][j]`: response j of sample `sample_index[k]`. Absolute values; ids are `case:id`.
  - `nominal_A` / `nominal_B` (same seeds).
  - `arc_tbr_A` / `arc_tbr_B` and `arc_tbr_nominal` from `blanket/uq/raw/results`.
  - `nd_rel_sd`, `noise_rel_sd_per_sample`, `corr_with_arc_tbr`.
- `geometry_sensitivity.<case>.<id>`: each 1σ term (signed, relative).

# TF coil: browser multigroup transport vs a converged OpenMC CE reference

Question: is the browser's TF magnet life (3e18 n/cm2 over the fast (> 0.1 MeV) fluence per FPY,
averaged over the inboard coil's plasma-facing 5 cm at |Z| < 1 m: StoneLab geometry.ts
TF_FACE_CM / TF_HALF_H_CM, design.ts) right, and why did the coupled n-gamma work see the coil
flux 6x-28x high at low energy and the coil heating +283%? ARC baseline (arc_model.py defaults).

"Fast" in both codes is the VITAMIN-J-175 groups whose lower edge is >= 0.1 MeV, i.e. E > 111 keV.

## Answer

| | face fast fluence (n/cm2/FPY) | browser / CE | magnet life (FPY) |
|---|---|---|---|
| OpenMC CE reference (6 runs, 4 window sets, 5.4M histories) | 3.996e17 +- 0.9% | 1 | **7.51 +- 0.07** |
| browser, current pack `VITAMIN-J-175_hist64` (320M histories) | 3.752e17 +- 1.5% | 0.939 +- 0.016 | 8.00 +- 0.12 |
| browser, new pack `VITAMIN-J-175_hist64_ww_wa` (320M) | 3.928e17 +- 1.5% | 0.983 +- 0.017 | 7.64 +- 0.12 |
| browser, new pack `VITAMIN-J-175_hist64_ww_wa_r20` (recommended, 320M) | 3.938e17 +- 1.5% | 0.985 +- 0.017 | **7.62 +- 0.12** |

The page's ~8.0 FPY was 6.5% optimistic (~4 sd); with the fixed constants the browser agrees
with CE within 1 sd (1.7%). Paper (Sorbom 2015): 9 FPY on their (different) coil/shield
geometry. A single 8M-history "Resolved" run scatters 9.5% (1 sd) about these means.
The reference's error is the run-to-run scatter of the six runs (2.2% per run): the batch
errors (0.9-1.6% per run) understate it ~2x -- a residual heavy tail, see below.

## Reference (coil_ref.py)

OpenMC 0.16 CE, ENDF/B-VIII.0 (blanket/xs), the arc_model.py model unchanged except the face
as its own cell (coil material) and the shield and coil filled with 5 cm R shells (same
material). Tallies: face fast flux (one bin, and the same by an (R, Z) mesh), 175-group flux and
heating-local in face, coil, shield, tank, blanket, channel, fw; 175-group inboard profile
(3 cm R bins, |Z| < 1 m, from 66 cm inside the coil face to the blanket); TBR.

Variance reduction, built in three passes (ww_magic.py):
1. static windows (0.25 exp(-d / 12 cm), d outside the tank box) -> (R, Z, 5 energy bins) mesh flux
2. MAGIC windows from that (lower bound proportional to the forward flux per energy bin, 0.5 at
   the peak), only for E > 0.1 MeV and only in the inboard leg (115 <= R < 195 cm, |Z| < 300
   cm; floor 5e-3 elsewhere); slow neutrons a constant 0.25 bound (rouletted: they cannot score
   a > 0.1 MeV tally); a run with those -> its mesh flux
3. the same from pass 2's mesh -> production windows. Two shapes were used: `ww5ff` (MAGIC in the
   focus box, a hard floor outside it) and `ww_shape` (`--face=133,100,230 --lam=10`: the MAGIC
   bound in front of the face, i.e. R >= 133 cm, |Z| <= 1 m, R <= 230 cm; elsewhere the bound at
   the nearest point of that column x exp(d / 10 cm), rising smoothly away from the target).
Upper/lower 5, max_lower_bound_ratio 10, windows checked at collisions AND surface crossings.
Face fast fluence per run (1e17/FPY, batch s.d.): ww5ff 3.983 (1.0%), 3.951 (0.9%); all-bin
MAGIC 3.902 (1.6%); ww5ff re-derived from the run 1 mesh 4.140 (1.2%); ww_shape 3.948 (1.0%),
4.060 (0.9%). The shaped windows run 2.3x faster than ww5ff and avoid the hard floor, whose
roulette promotes deep-coil survivors to ~1e4 x the local window right behind the face.

What it took (each of these gave wrong or unconverged answers first):
- **Windows act only at collisions by default.** An uncollided 14 MeV neutron crosses 51 cm of
  homogeneous TiH2 without one window check, so the deep streaming component was never split:
  the face's > 10 MeV score varied 4.7x between runs and face estimates scattered 2.4-4.3e17 with
  quoted errors of 7-22%. The R shells give a surface every 5 cm; with surface checkpoints the
  figure of merit for the face rose ~60x (3% in 3e5 histories on 4 threads).
- **One fast window bin** lumps 14 MeV streamers (high importance) with built-up 0.1-1 MeV
  neutrons; 5 bins (0.1, 1, 3, 10, 20 MeV).
- **Coincident surfaces.** The face cell's outer boundary was a second ZCylinder(r = shield
  rmin), coincident with the shield box's own: a neutron crossing from the shield sits on a
  surface of ambiguous sense and OpenMC files it under the rest of the coil (same material, no
  physics change) -- the face cell tally read 8-14% low against a mesh tally in every run. The
  face now uses the shield's surface object; cell and mesh face tallies agree exactly.
- Global MAGIC including slow neutrons ran at ~30 p/s (splitting thermal neutrons in the whole
  coil box); OpenMC's built-in MAGIC iteration ran away on this model before (activation/NOTES).
- FW-CADIS on random ray (0.16) was tried and abandoned: its convert_to_multigroup material-wise
  library comes from an analog CE run that barely reaches the coil, i.e. the target itself.

Verification (vr_check.py): a 4M-history analog run, every region/band it resolves to < 10%:
production seeds 36 fast-band comparisons each (fw, channel, blanket, tank, shield, profile),
mean z +0.06 / -0.72, rms z 0.94 / 1.19, none |z| > 3. Analog face: 3.70e17 +- 35% (consistent).
The two seeds agree to 0.8% (0.6 sd). A third run with windows re-derived from seed 1's mesh
(different windows, new seed) read 4.140 (high side of the floor's heavy tail); the shaped
windows read 3.948 / 4.060. Caveat: in these face-targeted runs TBR and blanket
heating scatter beyond their batch errors (TBR z +3.9 / -0.6 vs analog: deep low-weight
particles rouletted back to full weight in the blanket); don't take blanket numbers from them.

## Browser vs reference (scripts/coil/spectra.ts in StoneLab, compare.py)

Fast bands, browser / CE (CE s.d. +- browser s.d.):

| | 0.1-1 MeV old | new | > 1 MeV old | new |
|---|---|---|---|---|
| coil face (5 cm) | 0.929 | 1.000 +- 0.007 | 0.951 | 0.961 +- 0.008 |
| whole coil | 0.788 | 0.980 +- 0.028 | 0.953 | 0.993 +- 0.045 |
| shield | 0.987 | 0.993 +- 0.001 | 0.996 | 0.997 +- 0.001 |
| tank wall | 0.985 | 0.986 +- 0.001 | 0.998 | 0.999 +- 0.001 |
| inboard shield profile, 17 bins | 0.98-0.99 | 0.99-1.00 | 1.00-1.02 | 0.997-0.999 |
| coil 9-12 cm deep | 0.88 | 1.02 | 0.86 | 0.94 |
(compare_final.json; CE = all six runs combined. The remaining > 1 MeV deficit starts inside the
coil -- the last shield bins are at 0.999 -- and is ~2% of the face fluence.)

Slow bands (against `ref_slow`, MAGIC in all five bins, inboard-focused; its face fast fluence
3.902e17 +- 1.6%), browser / CE for < 1 eV, 1 eV-1 keV, 1-100 keV:

| | old | new (`_ww_wa`) |
|---|---|---|
| coil face | 1.96, 4.61, 3.30 | 1.06, 1.03, 1.11 (+-0.01-0.03) |
| whole coil | 1.89, 18.4, 6.58 | 1.12 +- 0.15, 0.89 +- 0.06, 1.33 +- 0.05 |
| coil 9-12 cm deep | 0.65, 20.0, 6.36 | 0.82, 1.25, 1.37 |
| shield | 0.99, 0.99, 0.99 | 1.00, 1.00, 1.00 |

Coil heating-local (photon energy deposited at birth, as the coupled n-gamma work compared):
CE 1741 +- 85 W (ref_slow) / 2024 +- 117 W (static-window run, global coil sampling);
browser old 8302 W (4.1-4.8x), new 1764 +- 34 W (0.87-1.01). The deeper coil still runs high
at low energy with the new pack (whole-coil constants, one in-situ spectrum for 64 cm of coil).

## Root cause

Kernel isolation: OpenMC in multigroup mode on the current pack's own constants (pack_mgxs.py:
sigma_t, w, transfer rows and 64-bin cosines exactly as the browser samples them; coil_ref.py
--mg) gives a face fast fluence of 3.731e17 +- 1.5% against the browser's 3.752e17 +- 1.5% on
the same pack (1.006 +- 0.021), and the fast bands through the shield and coil profile agree to
1-3%. The browser kernel reproduces an independent MG code on identical data; the gap to CE is
the data.

Not the kernel, the splitting, the source or the shield: the browser follows CE through all
51 cm of TiH2 to < 1% in the fast bands with either pack, and the shield's whole-cell spectrum
agrees to 1-2% in every band. The error starts at the shield/coil interface and is in the
**coil's group constants**, which mgxs_gen.py tallied in the 2M-history analog design run where
the coil gets ~1e-5 of the flux: per-group constants from a handful of tracks (fill_mgxs.py
already patched the 54 groups with none at all).
- The weight out of a collision, w = nu-scatter sum / sigma_t (export_neutpack.py), comes from
  the analog scatter tally: w > 1 in 50 coil groups below the (n,2n) thresholds (2.0 in group 17
  from one (n,2n) event), 0 in 2. w multiplies the weight at every collision, and the thermal
  absorption it implies (1 - w) is a small difference of two noisy numbers: coil thermal group
  0.515 vs 0.923 in the converged fill. Slow coil flux 6-29x high, coil heating 4x.
- sigma_t and the transfer rows in the 0.1-3 MeV coil groups from few tracks: face fast 5%
  low, whole-coil 0.1-1 MeV 20% low.

Attribution at the face (browser / CE, 320M histories each, +-0.017): current pack 0.939;
current constants + w from absorption 0.927; converged constants + analog w 0.962; both 0.983;
both + fill constants in the 86 coil groups still > 20% s.d. 0.985.

Also not it: the fill's flat-lethargy weighting (the r20 variant swaps 86 coil groups to it:
same answer to 0.3%), hist64 cosines, H thermal treatment in TiH2 (no S(alpha,beta) in either
code; shield slow bands agree). Resonance windows (xs_windows.py, pointwise ENDF/B-VIII.0):
within the face's 5 cm, in-group Sigma_t structure changes uncollided transmission by < 5% in
every group above 10 keV, but over 20-60 cm of coil by factors 1.1-10 in the 0.1-1 MeV Fe/Cu
window groups -- the residual > 1 MeV deficit deeper in the coil (0.94 at 9-12 cm) is of that
kind; it does not reach the TF face metric.

## Fix (data pipeline; kernel unchanged)

- mgxs_gen.py `--ww`: static windows outside the tank box in the design run (design-agnostic,
  so every variant gets converged deep-cell constants); inside the tank box the bound (0.25) is
  below the analog weight 1, so nothing there changes (TBR 1.1352 -> 1.1358, layer heating
  within 0.2%). Saves ce_flux_sd. `--no-mg` skips the gate-1 MG rerun. Run dir `_ww`.
- export_neutpack.py `--wout=absorption`: w = (1 - sigma_a / sigma_t) x nu_bar with the
  track-length absorption; `--resolve=R` fills groups with flux s.d. > R; `--suffix`. Defaults
  unchanged (old packs reproduce bit for bit).
- New packs (recipe: mgxs_gen --ww, export --wout=absorption --resolve=0.2):
  `VITAMIN-J-175_hist64_ww_wa_r20` (+ `_ng`), `VITAMIN-J-175_hist64_breeder-pbli_ww_wa_r20`;
  also `_ww_wa` (+ `_ng`), `_ww_sc`, `_coilx_wa`, `_breeder-pbli_ww_wa` for the attribution. Pack 13.9 -> 14.6 MB (gz
  3.6 -> 4.4 MB: denser deep-cell transfer rows). blanket_web/build.py skips tags it doesn't know (`_ww*`), so the page still
  ships the old packs until someone switches it (and regenerates the variant packs with --ww).

Heating and TBR elsewhere (StoneLab scripts/coil/heating.ts = photon/validate.ts, 20M histories,
vs OpenMC coupled n-gamma): every layer fw .. shield and TBR unchanged within 0.2% (TBR 1.1344
-> 1.1346 vs 1.1351 +- 0.0008); coil coupled heating +82% (2.8 sd) -> -5.9% (`_ww_wa_ng`) /
-6.3% (`_ww_wa_r20_ng`) (-0.5 sd); inner VV 4 cm / no Be variant: +64% -> -9% / -11%.

## Other designs

Same reference recipe per design (static pass, then shaped windows `--face=<shield rmin - 5>,100,230`).

| design | CE face fast fluence | CE magnet life | browser old pack | browser new pack |
|---|---|---|---|---|
| ARC baseline | 3.996e17 +- 0.9% | 7.51 +- 0.07 | 0.939 +- 0.016 (8.00) | 0.983 +- 0.017 (7.64) `_ww_wa`; **0.985 +- 0.017 (7.62)** `_ww_wa_r20` |
| TiH2 shield 61 cm (inboard), baseline packs (new = `_ww_wa`) | 1.030e17 +- 1.3% | 29.1 +- 0.4 | 0.962 +- 0.023 (30.3) | 0.998 +- 0.024 (29.2) |
| PbLi breeder, its own packs | 3.515e17 +- 0.7% | 8.53 +- 0.06 | 1.15 +- 0.09 (7.42 +- 0.56) | 0.943 +- 0.023 (9.05) `_ww_wa`; **0.974 +- 0.025 (8.76)** `_ww_wa_r20` |

(browser / CE face fast fluence, browser magnet life in brackets; 320M browser histories each.
CE errors: batch, x ~1.5-2 for the residual tail seen on the baseline.)
The thick-shield case runs the baseline packs on a new geometry, as the page does for a
geometry edit: the coil constants carry over and the fix holds. The old PbLi pack shows the
failure mode at its worst: its coil had w up to 1.29 below the (n,2n) thresholds, so the
browser's face score was heavy-tailed (two of 40 seeds at 3.3x the median; 7.6% s.d. at 320M)
and the whole-coil 0.1-1 MeV flux 3-5x high. Its --ww design run (2M histories) still left every
coil group above 20% s.d. (PbLi hardly moderates, the coil sees fewer, harder neutrons): those
in-situ constants read 0.943 (coil 0.1-1 MeV 0.87-0.92, shield fine); with --resolve=0.2 the
coil takes the converged fill constants in all 175 groups and reads 0.974. Hence the recipe
below uses --resolve=0.2 (on the baseline it changes the answer by 0.2%).

## Commands

```
# reference (from ~/Projects/Vega; -v ~/runs/blanket/coil:/runs)
# (add --set key=value for a design; face = shield inner radius - 5 cm, 133 on the baseline)
coil_ref.py --ww static --particles 5e4 --batches 10 --tag p1                          # pass 1
ww_magic.py /runs/p1/result.npz /runs/ww_a.npz 0.3 --slow-lb=0.25 --face=133,100,230 --lam=10
coil_ref.py --ww /runs/ww_a.npz --particles 1e4 --batches 30 --tag p2                  # pass 2
ww_magic.py /runs/p2/result.npz /runs/ww_b.npz 0.2 --slow-lb=0.25 --face=133,100,230 --lam=10
coil_ref.py --ww /runs/ww_b.npz --particles 1e4 --batches 80 --seed 15 --tag ref_s15   # x >= 2 seeds
python3 combine.py ref ref_s15 ref_s16 ; python3 vr_check.py analog ref_s15 --fast-only
coil_ref.py --mg /work/blanket/neutpack/<pack> --ww /runs/ww_b.npz ... --tag mg       # OpenMC MG on a pack
# browser (StoneLab packages/physics)
PACK=VITAMIN-J-175_hist64_ww_wa_r20 npx vite-node scripts/coil/spectra.ts 8000000 40 tag ['{"shield_in_cm":61}']
PACK=VITAMIN-J-175_hist64_ww_wa_r20_ng npx vite-node scripts/coil/heating.ts 2000000 10
python3 compare.py ~/runs/blanket/coil/ref ~/runs/blanket/coil/browser_tag.json
# packs
mgxs_gen.py VITAMIN-J-175 hist64 [key=value] --ww --particles=40000 --batches=100 --no-mg
export_neutpack.py VITAMIN-J-175 hist64 [_key-value]_ww --wout=absorption --resolve=0.2 --suffix=_wa_r20
photon/export_photon.py VITAMIN-J-175_hist64[_key-value]_ww_wa_r20 VITAMIN-J-175_VITAMIN-J-42
```

# Data release audit (2026-09-14)

A first pass over everything in `output/` and `checkpoints/`, with an eye
toward an eventual public HuggingFace release. This is an inventory +
recommendations document, not a migration — nothing has been deleted or
moved. See "Recommended next steps" at the end for what a real release pass
would actually do.

**Headline: the real, physically-validated design data is genuinely
substantial and release-worthy.** Six real physics domains, each backed by
a real solver (not a surrogate) at generation time, totaling roughly
**640,000 oracle/solver-validated design rows** plus ~39,000 topology-
optimization density fields. The problem this audit actually found isn't
data quality — it's that the real data is currently buried under ~34GB of
internal pipeline scratch state (retrain-cycle snapshots) that duplicates
itself dozens of times over and was never meant to be seen from outside the
project.

## 1. The big finding: 34 of 38GB is redundant intermediate storage

| Location | Size | What it actually is |
|---|---|---|
| `output/splits/` | 27GB, 157 subdirectories | Per-retrain-cycle training-set snapshots (`<tag>_r1` … `<tag>_r29`, one per grounded-walk retrain). Confirmed directly: `p4_halfsiren_v1_r28` (445,698 rows) vs `_r29` (446,010 rows) — **each snapshot is a ~99.9%-duplicate of the previous one**, just the growing pool re-saved whole at every retrain instead of only the delta. |
| `checkpoints/` | 7.9GB, 278 files | Same pattern for model weights: `reg_half_siren_grounded_<tag>_r1.pt` … `_r61.pt`, 44MB each, one full snapshot per retrain cycle of a run that's still in progress. |

Neither of these is "the data" in any release-worthy sense — they're
resumability checkpoints for this project's own training loop. The actual
information (every real design ever found) already lives in a much smaller
place: each grounded-walk run's own `real_pool_X.npy`/`real_pool_Y.npy`
(one row per accepted real step, no duplication across rounds — the 18
stellarator lanes total 94,984 rows in 51MB combined, not 27GB) and the
final/best checkpoint per run. **A real release should draw from those, and
exclude `output/splits/` and all but the final checkpoint per run
entirely.**

## 2. Per-domain inventory (canonical, non-scratch data only)

| Domain | Base dataset | Generated/expansion data | Format | License chain |
|---|---|---|---|---|
| **Stellarators** (VMEC++) | `output/X.npy`/`Y.npy` — 158,685 rows × 92 features, sourced from `proxima-fusion/constellaration` (HF, **MIT** — confirmed via the dataset's own page), float32-downcast only | 18 grounded-walk lanes, 94,984 additional real, VMEC++-validated rows (`output/grounded_walk_*/real_pool_{X,Y}.npy`); NAE-constructed pilot (`output/nae_pilot/results.jsonl`, 37 rows) | `.npy` pairs, 92-dim feature vector (45 r_cos + 45 z_sin + nfp + symmetry flag), 11 named targets (`output/target_names.json`) | MIT base (attribution required) + own VMEC++ (Apache-2.0-family solver, output not derivative) |
| **Airfoils** (XFOIL) | `output/airfoil_X.npy`/`Y.npy` — 50,011 rows × 18 features, generated from scratch (CST-parameterized seeds + Gaussian perturbation, own pipeline — **not** derived from AirfRANS despite AirfRANS motivating the domain choice) | `airfoil_vacation_merged_{X,Y}.npy` (300,081 rows), `airfoil_vacation_topup_{X,Y}.npy` (250,070 rows) — generative-model bootstrap expansions | `.npy` pairs, `airfoil_target_names.json`/`airfoil_feature_names.json` | Own-generated; XFOIL is GPL-3.0 but its output data isn't a derivative work of the solver |
| **TORAX** (tokamak transport) | `output/torax_X.npy`/`Y.npy` — 15,275 rows × 10 features | `torax_vacation_merged_X.npy` (55,376 rows), `torax_vacation_topup_X.npy` (40,101 rows) | `.npy` pairs | TORAX itself Apache-2.0 (confirmed via its LICENSE file) |
| **Mug v2** (OpenFOAM thermal) | `output/mug_X.npy`/`Y.npy` — 20,160 rows × 9 features | (no vacation-expansion files found under this prefix as of this pass — confirm before release whether one exists under a different tag) | `.npy` pairs | Own physics + OpenFOAM (GPL-3.0 solver, output not derivative) |
| **Photonics** (MEEP grating coupler) | `output/photonics_X.npy`/`Y.npy` — 20,018 rows × 6 features | `photonics_vacation_merged_{X,Y}.npy` (80,078 rows), `photonics_v2_vacation_topup_{X,Y}.npy` (60,060 rows) | `.npy` pairs | MEEP license not re-confirmed this pass — flag before release |
| **Bracket** (topology optimization, Warp) | `output/bracket_dataset_v1_symmetric.npz` (13,050 samples, 4,800-dim density field), `_v2_asymmetric.npz` (13,030 × 12,544-dim), `_candidate_12.npz` (13,030 × 4,800-dim) | Individual GAN/vacation candidate meshes (`bracket_gan_*`, `bracket_vacation_candidate_*` — ~600 individual mesh+field JSON/npy sets, ~180MB) | `.npz` (rho/compliance/source/iteration arrays) + per-sample mesh JSON | Own FEM/Warp pipeline (Warp is NVIDIA, Apache-2.0) |

Every domain has machine-readable target/feature name manifests already
(`*_target_names.json`, `*_feature_names.json`, or `output/metadata.json`'s
`feature_layout` for stellarators) — that's most of a dataset card's schema
section already written, just not assembled into one.

## 3. Licensing: no blockers found, one attribution obligation

- **`proxima-fusion/constellaration` (stellarator seed data): MIT.**
  Redistributable, modification permitted, only requirement is including
  the original license/copyright notice — this project's `X.npy`/`Y.npy`
  (float32-downcast copy of upstream r_cos/z_sin + metrics) needs a NOTICE
  file pointing back to it, not a redesign.
- **AirfRANS (ODC-ODbL, share-alike)** was investigated as a possible
  bootstrap source for airfoils but **never actually incorporated** — the
  airfoil dataset was generated fresh via this project's own CST-seed +
  XFOIL pipeline. No ODbL share-alike obligation applies to any data here.
  Confirmed by reading `generate_airfoil_dataset.py`'s own docstring, not
  assumed.
- **Solvers** (XFOIL GPL-3.0, OpenFOAM GPL-3.0, VMEC++/constellaration
  package license, TORAX Apache-2.0, jax-cfd Apache-2.0, Warp Apache-2.0):
  copyleft licenses like XFOIL/OpenFOAM's GPL-3.0 govern the *software*,
  not simulation *output data* run through them — this project never
  redistributes the solver binaries themselves, only generated design
  rows, so this isn't a blocker for a data-only release. Worth a real
  lawyer's sign-off before a large public release regardless, but no
  red flag found in this pass.
- **MEEP's** license wasn't re-confirmed this pass (it's used, but no
  license note was found in `EXPERIMENT_LOG*.md` — likely GPL-2.0 like
  most MEEP distributions, same "output data isn't a derivative" reasoning
  should apply, but flag for a real check before release, not assumed here).

## 4. What's NOT release-worthy as-is (recommend excluding entirely)

- `output/splits/` (27GB) — internal retrain scratch, see §1.
- `checkpoints/*_r{1..N}.pt` intermediate snapshots (keep only the final
  checkpoint per run, if checkpoints are released at all — model weights
  are arguably a separate release from the data itself).
- `output/physics_cache` (40MB) — solver-side memoization cache, not data.
- Dev/debug artifacts mixed into the same flat `output/` directory:
  `*_run.log`, `ablation_summary.json`, `audit_baselines.json`,
  `battery_*.jsonl`, `curriculum_r*_*.jsonl`, `_diag_*` outputs — these are
  this project's own experiment-tracking exhaust, meaningful to us, noise
  to an external HF dataset viewer.
- `output/bootstrap_bootstrap0` (403MB) and other one-off/superseded
  bootstrap pool directories (`bootstrap_generic_*_test`, `*_wide`, etc.) —
  worth a pass to confirm which are superseded vs. still-canonical before
  deciding what merges into each domain's final release file.

## 5. Naming/consistency gaps an external viewer would immediately trip on

- "`vacation_merged`" / "`vacation_topup`" are meaningful to us (generated
  during the unattended vacation-compute push) but opaque to anyone else —
  a release would want these renamed to something like
  `<domain>_generated_v1` with a manifest explaining the generation method
  (bootstrap cVAE/diffusion/GAN, oracle-validated) rather than the
  internal timeline reference.
- Stellarator grounded-walk data is scattered across 18 separately-named
  lane directories (`grounded_walk_p3_v4d_nae_seed_gpu1`, etc.) — internally
  meaningful (each is a distinct search strategy/basin), but a release
  would want one consolidated `stellarators_generated.parquet` with a
  `source_run` column preserving that provenance, not 18 separate files a
  downstream user has to know to `glob()` for (this project already has
  the merge logic for this — `scripts/merge_all_real_pools.py`, built for
  training use, would need a documentation pass, not new code, to become
  the release-assembly step).
- No root-level `LICENSE` or `README.md` for the project/data at all
  currently — `EXPERIMENT_LOG.md` is this project's own working log
  (extremely valuable as *methodology* documentation for a release, "how
  was this data actually generated," but it's written for us, not a public
  reader — would need a distilled version per domain).
- No dataset card anywhere yet (HF's standard `README.md` YAML frontmatter
  + description format) — every domain already has the underlying facts
  (row counts, schema, generation method, license chain) scattered across
  `metadata.json`, `EXPERIMENT_LOG.md`, and script docstrings; assembling
  one is a writing task, not a data-generation one.

## 6. Recommended next steps (not started — flagging for prioritization)

1. **Decide the release shape**: one HF dataset repo per domain (6 repos:
   `<user>/stellarator-designs`, `<user>/airfoil-designs`, etc.) vs. one
   multi-config repo. Six independent repos matches how differently-shaped
   each domain's data actually is (scalar Fourier coefficients vs. mesh
   density fields) and lets each have its own focused dataset card.
2. **Write one assembly script per domain** that reads only canonical
   files (base + real generated pools, explicitly excluding `splits/` and
   intermediate checkpoints) and writes a clean Parquet file + a
   `README.md` dataset card (schema table, row count, generation method
   summary distilled from `EXPERIMENT_LOG.md`, license section per §3).
3. **Add a root `LICENSE`** (MIT is the natural choice, matching the
   stellarator base data and this project's own generated-data-only
   redistribution) and a `NOTICE` crediting `proxima-fusion/constellaration`.
4. **Confirm MEEP's license** and double-check whether any mug-domain
   "vacation" expansion data exists under a different naming pattern than
   assumed here before finalizing that domain's manifest.
5. Once a domain's assembly script exists, actually deleting
   `output/splits/*` and intermediate checkpoints (keeping only the
   inputs the assembly script reads) would recover the ~34GB — worth doing
   regardless of release timing, but hold off until confirmed nothing
   currently-running (the 6 active grounded-walk lanes) still needs its own
   `_idx`-tracked split for retraining.

Happy to build any of these (the per-domain assembly scripts are the
natural first real step) once you've picked a release shape — just say
which domain(s) to start with, or "all of them."

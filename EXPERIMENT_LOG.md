# Experiment log: predicting stellarator equilibrium metrics from boundary Fourier coefficients

A running record of the preprocessing pipeline, training methodology, and architecture search — condensed to conclusions and numbers, not the full blow-by-blow. Full history is in git if needed.

**Dataset** proxima-fusion/constellaration &nbsp;·&nbsp; **Task** 92-dim → 11-target regression &nbsp;·&nbsp; **Rows** 158,685 (of 182,222 raw) &nbsp;·&nbsp; **Compute** Docker + single RTX 5000 Ada

---

## Where things stand

> **⚠ The random 80/10/10 row split overstates generalization, substantially (§6).** 76% of test rows share a generation-lineage family with a training row, and holding out entire *regions* of coefficient space (a cluster split) costs 2.5-3x the aggregate test RMSE versus the random split — some targets 9-19x worse. **Worse: the "best" architecture/optimizer combo found under the random split (wide MLP + SOAP + ensemble) does not clearly win under the cluster split** — it loses to the plain 230K-param baseline on 7 of 11 individual targets, despite being ~4% better in aggregate mean, and its ensemble-disagreement calibration also degrades. Every RMSE number from before §6 was measured under the leaky split; treat all of it as an optimistic upper bound on true generalization, and treat *relative* architecture/optimizer rankings (SIREN vs. MLP, SOAP vs. Adam) as more trustworthy than the absolute numbers, since the same leakage affects everything compared under that split.
>
> **Small-capacity architecture search (random split).** A plain 3-layer MLP (230K params) is a strong baseline. SIREN activation wins two targets by a confirmed noise-floor margin (`min_norm_grad_scale_length` ~7σ, `average_triangularity` ~3σ) at the cost of `aspect_ratio` (~4σ worse) — but this does not survive scaling to 8x width, where plain MLP + SOAP beats SIREN's own wins outright (random split only — unconfirmed under cluster split).
>
> **Normalization (§7): input normalization hurts slightly (~7-9% worse); target z-scoring is roughly neutral in aggregate but meaningfully helps the two worst-generalizing targets (rotational transform, ~45-49% better) under the cluster split.** Worth adopting for those specific targets; not a fix for the generalization gap itself.
>
> **Design-search tool exists** (`scripts/optimize.py`) but was built and tuned entirely against the leaky-split ensemble — its real capability to find genuinely novel (not near-training-data) designs is unverified and, given §6, likely more limited than it looked.
>
> **Confirmed on a real design, not just a held-out split (§8): a design-search "feasible" candidate is not actually feasible.** VMEC++ (pip-installable, low-fidelity mode ~6-14s/call, the same solver that generated this dataset) is now integrated as a ground-truth oracle. Checked against a real `optimize.py` output: the design does physically exist (VMEC++ converges) but **2 of its 3 constraints are violated in reality** (`average_triangularity` −0.374 vs. required ≤−0.5; `edge_rotational_transform` 0.242 vs. required ≥0.3), and `qi` is off by 327%, despite the surrogate ensemble reporting low disagreement at that point. The oracle is fast enough (no hour-long wait) to check every future design-search candidate before trusting it.
>
> **First data-augmentation pilot done and retrained on (§9): 476 new oracle-validated designs, mixed result.** A VAE trained on the real dataset, sampled toward the under-covered clusters identified in §6 (93.9% of generated designs landed nearest an under-covered cluster, vs. ~50% baseline), validated through VMEC++ (~69% hit rate, 15 minutes for 476). Retraining the small baseline on original + 418 of these (train-cluster-anchored only, to avoid leaking the held-out val/test regions back in) **improved 6 of 11 targets — including the two hit hardest by the original gap, `axis_rot/nfp` −26% and `edge_rot/nfp` −35%, the same two targets §7's normalization fix also singled out — but made 5 others worse (`edge_magnetic_mirror_ratio` +70%, `aspect_ratio` +37%), moving the aggregate mean the wrong way (+9%).** Single seed, tiny perturbation (0.3% of training rows) — a real, specific lead worth confirming and scaling up, not yet a settled win.
>
> **Scaled up generation 60x with a different goal: physical-realism coverage, not metric accuracy (§9 cont'd).** Redirected toward broad, untargeted sampling for a follow-up use case that cares about the feasible region's breadth, not hitting specific target values. Two real bugs caught first: fully independent per-coefficient sampling (the most literal "no prior") hit **0% valid over 2,592 attempts** (real boundaries need cross-coefficient correlation the VAE provides); and the very first attempt at broad sampling hung for **2 hours** on a single pathological candidate, since `ProcessPoolExecutor` can't reclaim a stuck worker — fixed with a per-candidate subprocess + hard 45s kill-on-timeout. Landed on unconditional VAE-prior sampling (no anchor, no cluster bias) and ran it for 12 hours: **29,243 validated designs** (18.4% of the original dataset's size), 55.2% hit rate, reaching genuinely beyond the dataset's range on some targets (`aspect_ratio` up to 31.7 vs. the original data's max of 12.1).
>
> **Retrained on the full 29,243-design batch — part of the §9 pilot's pattern is real and strengthens with scale, part of it doesn't replicate.** 24,589 of the 29,243 (the rest anchored in val/test clusters, dropped) added to the cluster-split train set — an 18.5% increase, vs. the pilot's 0.3%. Two targets show a clean, monotonic trend across both sample sizes in the *same* direction — `axis_rot/nfp` −26%→**−49%** and `edge_rot/nfp` −35%→**−49%**, both improving substantially more with more data — and two others worsen further in the same way (`edge_magnetic_mirror_ratio` +70%→**+81%**, `axis_magnetic_mirror_ratio` +32%→**+65%**). That's four targets moving the same direction at two independent sample sizes — real signal, not noise. But three others don't replicate cleanly: `aspect_ratio`'s harm *shrank* (+37%→**+26%**) rather than grew, and `max_elongation`/`average_triangularity` flipped from a pilot-stage improvement to roughly flat-to-slightly-worse (−14%→**+1%**, −22%→**+4%**) — the pilot's apparent win on those two looks like it was mostly small-sample noise. Net: the aggregate mean got worse at the larger scale, not better (+9%→**+15%** over the 9 targets with a logged baseline) — more data amplified the real, specific, already-identified mechanism (rotational transform helped, mirror ratios hurt) without fixing or even clearly worsening the rest. Not a case for blind augmentation; a case for the rotational-transform-specific mechanism being worth pursuing on its own.
>
> **Built a self-training ("bootstrap") loop** (`scripts/bootstrap_loop.py`): each generation retrains the VAE on real + all synthetic accepted so far (warm-started from the previous generation, not from scratch), samples a new batch from it, validates through VMEC++, and appends accepted *and* flagged-rejected designs to a growing pool — with two anti-collapse guards (inflating latent sampling variance each generation, a fixed 10% real-data-only anchor). Run for 30 generations total (§9-10, one power-outage interruption, auto-resumed cleanly): pool now at **92,229 designs**, and since gen26, 20% of every batch is generated by gradient-descending toward minimizing `max_elongation` through the surrogate + VAE decoder (93-95% hit rate on that sub-batch, vs. 15-30% for the rest).
>
> **Gradient-guided search reached a design competitive with the paper's own benchmark (§10).** A second, standalone script (`scripts/gradient_walk.py`) runs persistent latent-space walks, validating every proposed step through real VMEC++ every round (not just at the end). Full run (`walk1`, 28 walks × 50 rounds, target `max_elongation`): **1,270 VMEC++-accepted designs, best measured 1.327** — vs. the ConStellaration paper's reported 1.27, and far past §5's design-search tool (5.88) on the same target. Not yet checked against the paper's other constraints (open thread) — the win is on `max_elongation` alone so far.
>
> **Viewer now supports comparing real vs. bootstrapped designs directly (§11).** Consolidated from 4 exported sources down to 2 — dropped the superseded `pilot_cluster` pilot batch, merged every generation of the VAE-bootstrap lineage (`vae_prior_12h`, `bootstrap_bootstrap0`, `gradient_walk_walk1`) into one `bootstrap` source (122,742 rows) against `real` (158,685) — with a new default "color by source" projection view and floating per-source cards (hover to isolate) overlaid on the plot itself. Panes are now drag-resizable. Not yet actually looked at.

---

## 1. Data & methodology

Everything runs in Docker (never on host); `preprocess`/`train` services share volume-mounted `./output`/`./checkpoints`.

| | |
|---|---|
| Raw rows | 182,222 → kept 158,685 (dropped malformed boundary + invalid targets) |
| Input dim | 92 = `r_cos`(45) + `z_sin`(45) + `n_field_periods`(1) + `is_stellarator_symmetric`(1, always 1.0) |
| Output dim | 11 (of 12 raw — `aspect_ratio_over_edge_rotational_transform` dropped, algebraically redundant and blows up near-zero denominators, std 9,882) |
| Default split | 80/10/10 random rows, seed 42 (see §6 for why this is now flagged) |

**No target normalization** (until §7): raw physical units + learned per-task uncertainty weighting (Kendall & Gal 2018, `loss = Σ exp(-log_var_k)·MSE_k + log_var_k`) instead of z-scoring — each target's own `log_var` is learned, no hand-picked constant, and it's what let a pathological target (the dropped column above) reveal itself as unlearnable rather than just unstable. **No input normalization** (until §7): inputs already zero-mean, O(0.01-0.4) per `metadata.json`. Both re-tested in §7.

Other standing conventions: validate every 5-20 epochs not every epoch; best-val-loss checkpoint (not final epoch) evaluated once against held-out test; matched epoch budgets across compared variants (300, or fewer once SOAP made convergence faster); noise floor measured via seed reruns before trusting any "win" (§2).

## 2. Small-capacity architecture search (300 epochs, random split) — noise floor confirmed

Reran `single_base` 3x (seeds only) to get real run-to-run std per target before trusting any claimed win.

| variant | params | qi | vacuum_well | aspect_ratio | max_elong. | avg_triang. | min_norm_grad |
|---|--:|--:|--:|--:|--:|--:|--:|
| single_base (MLP) | 230,550 | 0.00659 | 0.01653 | 0.08390 | 0.43205 | 0.03759 | 0.73323 |
| siren_base | 230,550 | 0.00639 | 0.01846 | 0.09816 | 0.43429 | **0.03243** | **0.68329** |

Confirmed against 3-seed noise floor: SIREN wins `min_norm_grad_scale_length` (~7σ) and `average_triangularity` (~3σ), loses `aspect_ratio` (~4σ worse) — a genuine trade, not a free win, at this capacity only (see §5, doesn't survive scale-up).

Everything else tried at this capacity — dual-path spatial CNN branch, deeper per-target heads, chained sine/ReLU blocks, mode-attention encoder, derived geometric features, log-space target/latent heads, symlog augmentation — either didn't beat the baseline once epoch budgets were matched, or wasn't confirmed beyond noise. Attention was worst on every target (untested at scale/tuning). Log-space heads specifically backfired on their own targets (`min_norm_grad` ~10σ worse, `qi` ~7σ worse) — log-space training optimizes relative/tail accuracy while the eval metric is physical-space RMSE, dominated by bulk accuracy.

## 3. Contrastive framing: pairwise better/worse instead of absolute value

For a surrogate meant to guide design search, whether config A beats config B may matter more than either's exact value. Same backbone, but the loss becomes pairwise comparisons: score diff `d = s_i − s_j` maps through a Davidson (1970) pairwise-with-ties model, `P(i>j) = e^{d/2}/(2cosh(d/2)+ν)`, `P(tie) = ν/(2cosh(d/2)+ν)`, with a learned per-target tie-propensity `ν_k` (the contrastive analogue of `log_var`). Ground-truth "tie" = a pairwise difference smaller than that target's measured noise floor (§2).

Training directly on this beat scoring a regression checkpoint at test time (mean concordance .967 vs .959, MLP) — but naive per-pair-mean training made `ν_k` collapse to "never predict tie" on 10/11 targets, since true tie-rate spans 0.07%-8.4% across targets and rare-tie targets get essentially no gradient signal. Fixed with **per-target adaptive class-balance weighting** (`--tie-weight-cap`, caps the effective majority:minority loss-weight ratio per target rather than using one flat correction strength):

| variant | mean concordance | mean tie-recall | mean tie-precision | targets w/ real tie signal |
|---|--:|--:|--:|--:|
| unweighted | .967 | .089 | .068 | 1/11 |
| full class-balance | .872 | .899 | .075 | 11/11, concordance wrecked |
| flat interpolation | .963 | .178 | .092 | 4/11 |
| **per-target adaptive (used)** | **.945** | **.538** | **.101** | **10/11** |

`minimum_normalized_magnetic_gradient_scale_length` never picked up tie signal under any scheme — consistent with it being the hardest, most heavy-tailed target throughout (not a reweighting artifact). SIREN's regression-side edge on `average_triangularity` does not show up once trained directly on the ranking objective (MLP tracks equal-or-ahead everywhere, single seed).

## 4. Scale-up: wide MLP + SOAP ensemble (random split)

Benchmarked latency before scaling rather than assuming: 230K-param model is 0.33ms/sample; even a 112M-param model is only ~1ms. Latency was never the constraint — the real ceiling is dataset size (127K rows), so capacity was spent on **ensembling** (real accuracy + free calibrated uncertainty) rather than blind width.

**8x-wide single-path MLP (7.1M params) + 3-seed ensemble** beat every 230K variant on every target, including erasing SIREN's two confirmed wins. The identical width bump on the **SIREN** trunk regressed on *every* target vs. its own 230K baseline — plausibly `SineLayer`'s `omega_0`/init scheme not transferring to 8x width (untested fix). Ensemble inter-member disagreement vs. actual error was monotonic on 11/11 targets — a real, usable "trust this prediction" signal (**at random-split evaluation only — see §6, this degrades under cluster split**).

**Optimizer: SOAP** (Shampoo-preconditioned Adam) beat Adam on every target by 4-42%, but cost ~10x more wall-clock per epoch — profiled and found the cost was an every-step O(dim³) outer-product update on the wide axes, not just the periodic eigh/QR refresh. Fixed via `--soap-max-precond-dim` (default 1024, skip preconditioning axes wider than this) + `--soap-precondition-frequency` (default 50): **4.5x speedup** (~18.5s → ~4.3s/epoch), also letting the crash-workaround `magma` linalg backend (needed only for the ≥2048-wide case) drop back to faster default cusolver. This fast recipe was carried over to retrain the contrastive objective (§3) on the same wide backbone — concordance/recall/precision all improved together (.945→.950, .538→.579, .101→.112), no tradeoff this time.

## 5. Design search against the frozen surrogate

`scripts/optimize.py`: treats the 90 free Fourier coefficients as the optimization variable, runs Adam directly against the frozen ensemble (backprop through the network, weights fixed) to minimize/maximize one target subject to inequality constraints on others. `n_field_periods` fixed per run (integer, real design variable but the model only saw integers); `is_stellarator_symmetric` fixed at 1.0.

**Caught two real exploits before trusting any result**, both the same underlying failure mode (optimizer finds a corner of coefficient space that satisfies every per-coefficient marginal bound but sits nowhere near the real, correlated data manifold, where the surrogate is unreliable):
1. Per-coefficient clamping alone → predicted `aspect_ratio = −0.64` (physically impossible). Fixed with a **nearest-real-neighbor distance penalty** (squared distance, std-normalized, to the nearest actual same-nfp training point, recomputed every step).
2. Later, adding a proper ALM (augmented-Lagrangian, per-constraint penalty + Lagrange multiplier, ported from the ConStellaration paper's own ALM-NGOpt baseline) let the growing penalty `rho` outpace the *fixed*-weight distance/trust penalties → same failure mode recurred (distance 52.6 vs. normal 0.4-1.1, another impossible prediction). Fixed by **scaling the trust/distance weights together with `rho`**, keeping their relative balance constant regardless of how hard the ALM pushes on real constraints.

Set up a directly-comparable instance (ConStellaration's "Geometric problem": minimize `max_elongation` s.t. `aspect_ratio<=4.0`, `average_triangularity<=-0.5`, `|edge_rotational_transform_over_n_field_periods|>=0.3`, confirmed from their actual source code, not the paper prose). Their best baseline reaches `max_elongation=1.27`; two of their three baselines find *zero* feasible points at all (only ~41/160k dataset points are feasible — a genuinely sparse region). After both exploit fixes: 1/128 starts feasible, `max_elongation=5.88` — real progress (0→1 feasible) but well short of their result, and not yet stress-tested across more objective/constraint combinations.

## 6. Does the random split overstate generalization? Yes, substantially.

Motivating concern: rows aren't independently generated — they come from optimization runs seeded toward specific targets, and one run can produce multiple saved boundaries (continuation iterates, perturbations). Checked the raw dataset directly (not just our preprocessed columns): `omnigenous_field_and_targets.id` (the target spec `T=(ι*,A*,E*,O*)` a run was seeded toward) has **72,536 unique values over 182,222 raw rows — 60% duplication**, group sizes 1-7. Added to `scripts/preprocess.py` as `family_ids.json`.

**Measured leakage directly under the existing random split**: 76% of test rows share a family id with a training row. Nearest-neighbor distance confirms this is geometric, not just metadata: 36.2% of family-matched test rows have a near-exact train duplicate (std-normalized distance < 0.01); even *non*-matched test rows show 16.5% — family-id sharing only explains part of the near-duplication.

Built three split variants (`scripts/make_splits.py`, `train.py --split {random,group,cluster}`): **random** (existing default, reproduced exactly), **group** (whole `family_id` groups assigned to one split, fixes identified lineage leakage), **cluster** (k-means k=100 over std-normalized coefficients, whole clusters assigned to one split — tests generalization to unseen *regions*, the harder and more relevant test given the surrogate's purpose).

**Small baseline (230K MLP, Adam, 300ep) retrained under all three:**

| target | random | group | cluster | cluster vs. random |
|---|--:|--:|--:|--:|
| qi | .00717 | .00632 | .01926 | 2.7x |
| aspect_ratio | .06814 | .08220 | .44182 | 6.5x |
| axis_rot/nfp | .01198 | .01311 | .11450 | 9.6x |
| edge_rot/nfp | .01032 | .01039 | .12763 | 12.4x |
| **mean (all 11)** | **.1243** | .1322 (+6%) | **.3141 (+153%)** | |

Group split: only ~6% worse in aggregate (real leakage, modest aggregate effect). Cluster split: 2.5x worse, with rotational-transform targets 9-12x worse — best checkpoint at epoch 20/300 vs. ~220 for the others, i.e. it stopped generalizing almost immediately once asked to extrapolate rather than interpolate.

**Production recipe (wide MLP + SOAP, 3-seed ensemble) retrained under cluster split too** — and the "best" architecture doesn't clearly win anymore:

| | random split mean | cluster split mean | ratio |
|---|--:|--:|--:|
| small baseline | .1243 | .3141 | 2.5x |
| production ensemble | .1018 | .3023 | **2.97x** |

Production is only ~4% better than the small baseline in aggregate under cluster split, and **loses outright on 7 of 11 individual targets** (qi, vacuum_well, both mirror ratios, both rotational-transform, min_norm_grad) — despite being the clean, uniform winner on literally every target under the random split. Ensemble calibration also degrades: 3/11 targets non-monotonic (spread vs. error) vs. 1/11 under random split.

**Read:** brute-force width winning cleanly, architecture mattering little, and the nearest-neighbor distance penalty being load-bearing in the design search (§5) are all consistent with a model that's largely doing sophisticated interpolation over a dense, redundant point cloud rather than learning something that extrapolates. None of the *relative* comparisons above (SIREN vs. MLP, SOAP vs. Adam, contrastive reweighting) are necessarily wrong as rankings — the same leakage affects everything compared under the random split — but no absolute RMSE number in §1-5 should be read as true generalization to a genuinely new stellarator family.

## 7. Input/output normalization, re-tested under both splits

Re-tested the §1 "no normalization" decision now that a harder, more honest eval (cluster split) exists — small baseline (230K MLP, Adam, 300ep), both splits:

| variant | random split mean | cluster split mean |
|---|--:|--:|
| baseline (no norm) | .1243 | .3141 |
| normalize-targets (z-score + plain MSE, no uncertainty weighting) | .1246 | **.2941 (−6.4%)** |
| normalize-inputs (standardize 92 features) | .1328 (+6.8%) | .3423 (+9%) |
| both | .1278 (+2.8%) | .3228 (+2.8%) |

**Input normalization hurts slightly, consistently, on both splits** — confirms the original "nothing to normalize" reasoning was right, not just untested. **Target z-scoring is roughly neutral in aggregate but not uniform per-target**: under the cluster split it specifically helps the two targets hit hardest by the generalization gap — `axis_rot/nfp` (.1145→.0629, −45%) and `edge_rot/nfp` (.1276→.0652, −49%) — while modestly hurting `axis_magnetic_mirror_ratio` and `min_norm_grad_scale_length` (+2-4%). Worth adopting per-target rather than globally; not a fix for the generalization gap itself (−6% vs. the 2.5-3x gap it's being measured against).

`--normalize-inputs`/`--normalize-targets` in `train.py`: stats computed from the TRAIN split only (no leakage into val/test), stored in the checkpoint, target predictions always un-normalized back to physical units before reporting — so RMSE stays comparable to every other number in this log regardless of what space a given model trained in.

## 8. VMEC++ as a ground-truth physics oracle

Motivation: §6 showed the surrogate is largely interpolating known configuration families, not extrapolating. The only way to actually validate (or generate data beyond) the training distribution is to check candidates against real physics, not just the surrogate.

**VMEC++ (not STELLOPT).** The ConStellaration dataset itself was generated with VMEC++ — a modern reimplementation of VMEC by the same team, pip-installable, no MPI/GPU required, with the exact same boundary Fourier representation and target metric names as this project's data. `constellaration.forward_model.forward_model(boundary, settings=...)` wraps it directly; `settings=None` defaults to **low fidelity** (what generated our dataset), `ConstellarationSettings.default_high_fidelity()` for later dialing-in on promising candidates. STELLOPT (Fortran, MPI, much heavier to integrate) was not needed.

**Docker image change, one real bug caught and fixed.** `pip install vmecpp constellaration` needed `libnetcdf-dev` + `build-essential`/`gfortran`/`ninja-build` added to `Dockerfile.train` (for `booz-xform`, a C++ extension dependency, previously entirely absent from the image). That install also silently upgraded numpy 1.26→2.2 as a resolved dependency, which **broke `torch.from_numpy` outright** (`RuntimeError: Numpy is not available` — torch 2.2.0's C extensions are compiled against numpy's 1.x ABI). Caught via an actual training smoke test, not just an import check. Fixed by pinning `numpy<2` after the physics packages install; verified empirically that `constellaration`'s declared `numpy>=2.1.3` requirement is conservative, not a real hard floor — it imports and runs correctly against 1.26.4.

**Validated the oracle against known ground truth first.** Reconstructed a real dataset row as a `SurfaceRZFourier` boundary, ran it through `forward_model` at low fidelity: recomputed metrics matched the dataset's stored values closely (most within 2%, worst was `min_norm_grad_scale_length` at ~6%) — confirms correct integration, not just "it runs." **Took 5.7s low-fidelity**, not an hour. An invalid/unphysical boundary fails in ~0.02s with a clear error (either a pydantic structural-validation error, or a fast VMEC++ solver `RuntimeError`) rather than hanging — a free, cheap "is this even physically possible" signal, exactly the fast-check-before-dialing-in workflow wanted. Built as `scripts/oracle_eval.py` (`--row-index N` for dataset sanity checks, `--design-json PATH` for `optimize.py --save` outputs, `--fidelity {low,high}`).

**Ran the actual design-search output through it — confirms the §6 concern directly, on a real optimizer-produced design, not just a held-out split.** The `optimize.py` ConStellaration geometric-problem candidate (§5, reported "1/128 feasible," `max_elongation=5.88`, low ensemble disagreement at the time) was fed to the real oracle. VMEC++ converges — the design **is** physically buildable — but:

| target | surrogate predicted | VMEC++ ground truth | rel. diff |
|---|--:|--:|--:|
| aspect_ratio (constraint ≤4.0) | 4.018 | 4.080 | 1.5% |
| max_elongation (the objective) | 5.882 | 5.174 | −12.0% |
| average_triangularity (constraint ≤−0.5) | −0.506 | **−0.374** | 26.0% — **constraint actually violated** |
| edge_rot/nfp (constraint ≥0.3) | 0.330 | **0.242** | −26.8% — **constraint actually violated** |
| qi | 0.0357 | **0.152** | **327%** |
| min_norm_grad | 1.972 | 0.392 | −80.1% |
| vacuum_well | −0.602 | −0.551 | 8.5% |

**The design the surrogate reported as feasible is not actually feasible** — 2 of 3 constraints fail against real physics, despite low ensemble disagreement at the reported optimum. This is the concrete, real-design confirmation of §6's split-based finding: the design-search tool as built cannot yet be trusted to certify feasibility on its own; every candidate needs oracle verification before it means anything.

## 9. Generative pilot: augmenting the dataset toward a fuller manifold

Goal, directly motivated by §6: stop being bound to the existing dataset's coverage. First pass at generating *new*, oracle-validated designs concentrated in the regions the dataset covers most thinly, rather than more of the same.

**Pipeline:** conditional VAE (`scripts/train_vae.py`) over the 90 free boundary coefficients, conditioned on `n_field_periods`, trained on the *full* real dataset (158,685 rows — this model's job is to know the real manifold as well as possible, not to generalize the way the surrogate is evaluated). Its only purpose is to raise the hit rate against VMEC++ above what sampling raw noise would achieve — real boundaries are highly structured (9 of 90 coefficients are exactly zero by symmetry, others tightly correlated), so undirected sampling would rarely produce anything that converges at all.

**Sampling biased toward under-covered regions** (`scripts/generate_and_validate.py`): reused §6's k-means cluster assignment (now persisted as `cluster_assignments.npy`/`cluster_sizes.json`) to identify the 50 smallest-by-row-count of the 100 clusters (≤1,334 rows, vs. up to 10,025 for the largest) as "under-covered." Each candidate: pick a random real row from an under-covered cluster as an anchor, encode it to its VAE latent mean, decode a nearby point (anchor + Gaussian noise, `--explore-std`) using the anchor's own `n_field_periods` — concentrates new samples where the dataset is thin without discarding the structure the VAE learned.

**Validated through the real VMEC++ oracle (§8) in parallel across 28 of the machine's 32 CPU cores** (`ProcessPoolExecutor`, low fidelity) — every candidate either converges (kept, with VMEC++'s own computed metrics as ground-truth labels) or fails fast, incrementally checkpointed to `output/generated/{X,Y}.npy` (same format as the original dataset) so a long run can't lose progress.

**Pilot run: 476 validated designs in 15 minutes**, ~69% hit rate (0 structural rejections, all rejections were genuine VMEC++ convergence failures — the VAE never produced a topologically-invalid candidate). Confirmed the targeting actually worked: **93.9% of generated designs have their nearest real neighbor in an under-covered cluster**, vs. the ~50% baseline that untargeted sampling would give. Target ranges stayed physically sensible and mostly within the real dataset's range — with `aspect_ratio` reaching 16.4, genuinely *beyond* the real dataset's max of 12.1, a first concrete sign of stretching past the existing manifold rather than just resampling within it.

Not yet done: retraining the surrogate on the augmented dataset to see if this actually narrows the §6 generalization gap — this pilot only builds and validates the augmentation pipeline and produces the first batch of new data.

**Retrained on it — result is real but mixed, not a clean win.** Built `scripts/augment_split.py` first to do this safely: the 476 generated designs were sampled toward under-covered clusters *without regard to which split those clusters landed in*, so naively adding all of them to training would leak §6's held-out val/test regions back in through generated neighbors — the exact failure mode §6 diagnosed, reintroduced through the back door. Fixed by looking up each generated design's nearest-real-neighbor cluster in the cluster split's train/val/test membership and keeping only the 418 (of 476) anchored in a **train** cluster; val/test stay byte-identical to the original cluster split. Retrained the small 230K baseline (same recipe as §6) on original-train + these 418 (a 0.3% increase in training rows):

| target | cluster split, no augmentation | **+ 418 generated (train-anchored only)** | change |
|---|--:|--:|--:|
| qi | .01926 | **.01421** | −26% |
| aspect_ratio | .44182 | **.60464** | **+37% worse** |
| max_elongation | .72542 | **.62338** | −14% |
| average_triangularity | .11661 | **.09152** | −22% |
| axis_magnetic_mirror | .06887 | .09076 | +32% worse |
| edge_magnetic_mirror | .06687 | **.11388** | **+70% worse** |
| axis_rot/nfp | .11450 | **.08499** | **−26%** |
| edge_rot/nfp | .12763 | **.08346** | **−35%** |
| min_norm_grad | 1.55130 | 1.86220 | +20% worse |
| **mean (all 11)** | **.3141** | **.3434** | **+9% worse** |

6 of 11 targets improved, 5 got worse (two badly: `edge_magnetic_mirror_ratio` +70%, `aspect_ratio` +37%), and the aggregate mean moved in the wrong direction. Not read as a clean failure, though: the two targets that improved *most* (`axis_rot/nfp` −26%, `edge_rot/nfp` −35%) are the exact same two that target-normalization (§7) also specifically helped (−45%/−49% there) — two independent interventions singling out the same targets is more likely a real, specific pattern (something about how these two targets are represented or generalize) than coincidence. But this is a single seed, 418 added rows is a tiny perturbation to a 132,643-row training set, and there's no measured noise floor for "how much does swapping in a different small training-set composition move RMSE by chance alone" — unlike the seed-rerun noise floor in §2, which doesn't directly apply here. Treated as a real but unconfirmed lead, not a settled result — see open threads.

**Redirected: scaled-up generation for physical-realism coverage, not metric accuracy.** Explicit change in what this is for — the follow-up use case cares about breadth of the *feasible region* itself (VMEC++ convergence as a binary signal), not matching specific target values, and isn't meant to feed a full active-learning loop yet. Two changes from the pilot: dropped cluster-targeting in favor of broad, untargeted sampling ("basically no prior"), and this needed two real fixes before it could run unattended for hours.

**First fix attempt failed outright and revealed something important.** Tried fully independent per-coefficient uniform sampling (each of the 81 free coefficients sampled uniformly within its own observed range, no VAE at all) as the most literal "no prior." **0% hit rate over 2,592 VMEC++ attempts.** Real boundaries have strong cross-coefficient correlations (smoothness, non-self-intersection) that sampling each coefficient independently destroys completely — confirms the VAE isn't just a convenience, it's load-bearing for getting *any* usable hit rate at all once cluster-targeting is removed.

**Second bug, caught the hard way: VMEC++ can hang instead of failing fast.** The very first attempt at this (before the fix above was even tried) sat running for **2 hours** on a target of 20 candidates before being killed manually — a pathological low-quality random geometry made the solver spin instead of erroring out the way the clearly-invalid cases in §8/§9 did. `ProcessPoolExecutor` has no way to reclaim a worker stuck on one non-terminating task. Replaced it with a per-candidate throwaway subprocess + hard wall-clock timeout (45s, terminate then SIGKILL) — confirmed working in testing (caught and killed a hung candidate mid-run without stalling anything else).

**Landed on unconditional VAE-prior sampling** (`--sampling-mode vae-prior`): `z ~ N(0,1)` in the VAE's latent space, no anchor, no cluster bias, `n_field_periods` uniform over {1..5} — enough real correlational structure to actually work, but not targeted at any particular region the way §9's pilot was. Quick test: 62.5% hit rate. Kicked off a 12-hour run on this basis.

**12-hour run result: 29,243 validated designs** (18.4% the size of the entire original 158,685-row dataset) from 52,976 attempts, 55.2% hit rate, 0 structural rejections, 4,141 timeouts correctly caught and killed. Clean data, no NaNs. Real reach beyond the original dataset on some targets: generated `aspect_ratio` hit 31.7 (real dataset max: 12.1), `qi` hit 0.30 (real max: 0.27). As expected for untargeted sampling, coverage skewed toward the already-dense clusters (a 3,000-point subsample: only 8.7% landed nearest an under-covered cluster, vs. 93.9% for §9's targeted pilot) — but still touched 73+ of the 100 clusters.

**Retrained on all 24,589 train-cluster-anchored designs from this batch (60x the pilot's scale): confirms part of the pilot's pattern, doesn't replicate the rest.** `axis_rot/nfp`/`edge_rot/nfp` improved substantially more with more data (−26%→−49%, −35%→−49% vs. no-augmentation baseline) and the mirror-ratio targets got substantially more worse (+32%→+65%, +70%→+81%) — four targets, two independent sample sizes, same direction each time, real signal. But `aspect_ratio`'s harm shrank instead of growing (+37%→+26%), and `max_elongation`/`average_triangularity` flipped from a pilot-stage improvement to roughly flat (−14%→+1%, −22%→+4%) — those two looked like small-sample noise, not a real effect. Aggregate mean got worse at scale, not better (+9%→+15% over the 9 targets with a logged baseline). Reads as: a specific, real, worth-pursuing mechanism on rotational-transform targets, not a case for augmenting blindly.

**Built `scripts/bootstrap_loop.py`** for a self-training loop per this session's follow-up request: each generation warm-starts the VAE on real + all synthetic accepted so far, samples a new batch (10% from the original real-data-only VAE as a fixed anti-collapse anchor, 10% pure independent-per-coefficient random for zero-learned-bias negative examples, 80% from the current generation with inflating latent variance `std_g = 1 + 0.10g`), validates through VMEC++, and appends accepted + reason-flagged-rejected designs to a growing pool with a per-generation coverage/range-extension log. `generate_and_validate.py` also now persists rejected candidates (structural/vmec/timeout/**unknown** — worker crash or died-without-result, kept distinct from a confirmed VMEC physics rejection) going forward, rather than discarding them; every attempted candidate now ends up in either the accepted or rejected pool, nothing thrown away.

**Ran it for 15 generations (~30h total) — confirms the anti-collapse guards work, and confirms std growth is the actual coverage driver, not just retraining.** Generations 1-10 (std growing 1.10→2.00): coverage rose monotonically the whole way, 6.8%→20.2%, hit rate declined gradually and gracefully (45.3%→20.4%, not a collapse), pool reached 34,400 designs, range-extension breadth grew from 2 targets to 6 (`qi`, `vacuum_well`, `aspect_ratio`, `edge_magnetic_mirror_ratio`, `axis_magnetic_mirror_ratio`, `minimum_normalized_magnetic_gradient_scale_length`). Added a `--max-std` cap (generation count and exploration radius were previously hard-tied — no way to keep running generations without also pushing std wider each time) and ran 5 more generations frozen at std=2.0: hit rate **stabilized** (19.7-21.7%, no further decline — confirms std growth, not compounding retraining instability, was what drove the earlier decline), but coverage-fraction and range-extension both **plateaued** (coverage oscillating 18.3-21.6% with no clear trend; the same 6 targets showing extension, magnitudes creeping only marginally). Pool still grew substantially in raw volume (34,400→47,776) at a stable hit rate. Reads as: std growth is the one lever that expands reach; more generations at a fixed std mostly consolidate/add volume at the existing frontier rather than push further. To keep extending coverage, std needs to keep growing (at some rate); to hold quality/hit-rate steady, cap it — those are in tension, not a bug to fix.

## 10. Gradient-guided search for low `max_elongation`: directed sampling, then a dedicated latent-space walk

**The bootstrap loop kept running past §9's last logged point (gen15, 47,776-design pool) through gen30 (92,229) — 15 more generations, previously undocumented.** Switched from the flat `--max-std 2.0` cap to a `relax → expand → fill` cycle schedule (`--schedule cycle`), with each cycle's peak std ratcheting up by 0.2 (2.2 → 2.4 → 2.6 across cycles 0-2). Coverage-of-under-covered-clusters stayed flat around 21-23% through gen16-25 (cycles 0-1) — consistent with §9's finding that generations at a fixed-ish std mostly add volume rather than push reach, now confirmed to hold under cycling too, not just a flat cap.

**Gen26 added directed sampling** (`bootstrap_loop.py --directed-frac 0.2`): 20% of every batch is generated by gradient-descending in the current generation's VAE latent space, through the decoder and the frozen surrogate, toward minimizing `max_elongation` (`n_steps=3` per proposal, reusing `gradient_walk.py`'s `load_surrogate`) — then validated through real VMEC++ exactly like every other candidate, no special treatment. Gen26-30: directed sub-batch hit rate **93-95%**, far above the 15-30% the rest of each generation was getting at that point's exploration radius, and the surrogate's own predicted mean tracked the VMEC++-measured mean within a few percent every generation (gen30: 4.199 predicted vs. 4.141 measured) — gradient-following toward a target the surrogate can see holds up much better here than the open-ended optimizer candidate that failed 2/3 unrelated constraints in §8. Pool grew 72,554 → 92,229 over these 5 generations.

**Run died mid-generation-31 in a power outage; resumed cleanly.** `bootstrap_loop.py`'s auto-resume (same `--tag`, matches on the log's last recorded generation/vae-tag/schedule state) picked back up with no data loss — confirmed by inspection, not yet re-run further.

**Built a second, standalone gradient method: `scripts/gradient_walk.py`.** Where the directed-sampling above is one-shot (3 steps, discarded and re-anchored every batch), this runs persistent walks: each of `--n-walks` walkers takes one gradient step per round with one *shared* adaptive step size, decodes and validates every walker's proposed step through real VMEC++ every round (not just at the end), and a walker whose step fails VMEC++ rolls back to its last valid point rather than wandering into infeasible territory; the shared step size shrinks if too few of a round's proposals validate. Smoke-tested (3-4 rounds, 4 walks) first, then run at full scale — **`--tag walk1`, defaults (28 walks, 50 rounds, target `max_elongation`/min, anchored on real designs' encoded latents): 1,270 VMEC++-accepted designs, mean predicted `max_elongation` fell from ~4.8 (round 1) to 2.99 (round 50), best single accepted design measured 1.327** — within range of the ConStellaration paper's own reported optimum of 1.27, and far past what §5's design-search tool reached against the frozen surrogate (5.88). Every accepted point, like every other source in this project, is a real VMEC++ result, not a surrogate prediction.

## 11. Viewer: consolidating sources, comparing real vs. bootstrap directly

**Draggable panes.** The same left/right toggle strips now double as resize handles: a plain click still collapses/expands (unchanged behavior), but dragging past a small threshold resizes the pane instead (clamped 220-640px) and the width persists across a later collapse/expand.

**Source cards moved onto the projection plot itself.** The old sidebar list of sources is gone; each source now gets a small floating card overlaid directly on the UMAP/t-SNE canvas, with a fixed per-source color and a live count of how many of the currently-plotted ~1,000-point sample belong to it. Hovering a card isolates that source's points in the plot (same dim-the-rest mechanism as before, just relocated). Added a new **"color by source"** mode to the projection's color-by control — now the default — so the plot reads as "where does each source sit" immediately, without picking a metric first.

**Consolidated the exported sources from 4 down to 2, per direct request: `real` vs. `bootstrap`.** `pilot_cluster` (476 rows, the original §9 k-means-targeted pilot, superseded by everything after it) dropped from the export entirely — left untouched on disk, just no longer included. `vae_prior_12h`, `bootstrap_bootstrap0`, and `gradient_walk_walk1` merged into one `bootstrap` source (122,742 rows total) — same VAE-driven-exploration lineage across different generations/mechanisms of it, now viewable as one thing against the original 158,685 real rows (`scripts/export_viewer_data.py`, restructured to concatenate multiple path-pairs per named source). Re-exported: 281,427 rows total.

Not yet done: actually looking at what the consolidated real-vs-bootstrap view shows (no browser tool available this session, so the frontend changes above are verified statically — JS syntax, dev-server hot-reload, served markup — not by eye).

## 12. Open threads

**Generalization / oracle validation (highest priority — everything else is downstream of this):**
- The oracle (§8) has only checked one design-search candidate. Need to run it across more `optimize.py` outputs (different objective/constraint combos, different nfp) to see if 2/3-constraints-actually-violated is typical or was one bad case.
- §9's augmentation pattern is now confirmed real for 4 of 9 comparable targets (rotational-transform improving, mirror ratios worsening, monotonically across two sample sizes) but 3 others didn't replicate (`aspect_ratio`, `max_elongation`, `average_triangularity`) — worth investigating *why* rotational-transform/mirror-ratio specifically respond to untargeted synthetic data (same two normalization singled out in §7 too) rather than assuming it generalizes to other targets.
- The bootstrap loop (§9-10) ran 30 generations total (cycle-scheduled std since gen16, directed max_elongation sampling since gen26) and confirmed coverage tracks std growth, plateauing once std stops climbing — but the resulting 92,229-design pool hasn't been retrained-and-compared against the surrogate yet, so it's unknown whether it tracks the same rotational-transform/mirror-ratio split found with the earlier 29,243-design batch, or looks different once the generator was retraining on its own synthetic output (and, since gen26, on gradient-directed output) rather than a static batch.
- `gradient_walk.py`'s best design (`max_elongation` 1.327, §10) has only had that one target checked. It hasn't been run back through `oracle_eval.py`'s full metric set against the other 10 targets' constraints, the way §8 did for the `optimize.py` candidate — and §8's whole point was that a surrogate-confident candidate can look great on the one target being optimized while failing unrelated constraints in reality. Same open question for the rest of the 1,270 accepted `walk1` designs. Also untried at high fidelity (`ConstellarationSettings.default_high_fidelity()`) — everything here is low-fidelity VMEC++ so far.
- Whether to keep running gradient-guided generation (more `bootstrap_loop` generations, more `gradient_walk` rounds/walks, or both) or treat the current pool as sufficient and move to retraining/using it, is still undecided — same open question as in §9, now sharper given `walk1`'s result landed within range of the paper's benchmark.
- The viewer's consolidated real-vs-bootstrap view (§11) exists now but hasn't actually been looked at — the qualitative question that motivated building it ("how do the bootstrapped designs compare to the real ones") is still unanswered.
- Production recipe (wide MLP + SOAP ensemble) hasn't been retrained on the augmented set yet — only the small baseline has.
- High fidelity (`ConstellarationSettings.default_high_fidelity()`) hasn't been benchmarked for wall-clock cost yet — only low fidelity (5.7-14s) has been measured.
- The 4,141 timed-out candidates from the 12h run were discarded rather than examined (predates the reject-flagging fix) — worth a look at whether they're a distinct failure mode (e.g. systematically different from the fast VMEC++ rejections) or just unlucky slow convergence, since that could inform whether the 45s cutoff is well-calibrated. Going forward, new rejects (including timeouts) are saved by both `generate_and_validate.py` and `bootstrap_loop.py`, so this is answerable for any new run.
- No mitigation attempted yet for the split-leakage finding itself (§6) — e.g. adopting group/cluster split as the new default, deduplication before splitting, or modeling the family structure explicitly during training.
- §6/§7 only tried k=100 clusters, one seed, one architecture pairing (small vs. production) — cluster count/seed unswept, and no learning-curve sweep to see if more data or different capacity closes the gap.
- The design search (§5) has never been run against a cluster-split-trained model — given §6, the nearest-neighbor trust-region penalty may be doing even more real work than it appeared.

**Architecture/optimizer (random-split only, unconfirmed under cluster split):**
- SIREN's width-scaling regression (§4) unexplained — worth revisiting `SineLayer` init for wider layers before concluding SIREN just loses at scale.
- SOAP untried on the SIREN trunk — could plausibly fix the above (fundamentally different, better-conditioned update rule).
- No regularization anywhere in this codebase (dropout, weight decay) — several variants (chained half-SIREN, attention) were plausibly capacity-starved or overfit without it.
- Attention encoder given the least investment of anything tried — worst performer as tested, but transformers are notoriously warmup/LR/data-volume sensitive.

**Contrastive framing:**
- Single seed throughout — the MLP-over-SIREN reversal on `average_triangularity` (§3) needs a noise-floor check.
- `min_norm_grad_scale_length` tie-calling never worked under any reweighting scheme — likely needs the target's own difficulty addressed first.

**Design search / ConStellaration comparison:**
- `optimize.py`'s design search is still well short of the paper's result (5.88 vs. 1.27) — untried: more starts/outer-iterations, tighter trust weighting, or accepting this as a stretch goal. Separately, `gradient_walk.py` (§10, a different tool entirely — unconstrained single-target latent walk, not the multi-constraint design search) reached 1.327 on `max_elongation` alone; not an apples-to-apples comparison until that design is checked against the paper's actual constraint set (see oracle open thread above).
- Trust-weight/distance-weight defaults confirmed to fix the one observed exploit, not stress-tested across other objective/constraint combinations.
- No comparison yet against the paper's own MLP-ensemble surrogate (Appendix A.4: 10× 3-layer/256-hidden/tanh, z-scored targets, differently-processed ~23k-row split) — would need NRMSE/R² recomputed the same way to be comparable.

---

*`scripts/preprocess.py` · `scripts/train.py` · `scripts/make_splits.py` · `scripts/optimize.py` · `scripts/oracle_eval.py` · `scripts/train_vae.py` · `scripts/generate_and_validate.py` · `scripts/augment_split.py` · `scripts/bootstrap_loop.py` · `scripts/gradient_walk.py` · `scripts/export_viewer_data.py` — all runs via Docker*

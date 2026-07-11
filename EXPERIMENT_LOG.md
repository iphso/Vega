# Experiment log: predicting stellarator equilibrium metrics from boundary Fourier coefficients

A running record of the preprocessing pipeline, training methodology, and architecture search — including which apparent wins turned out to be real signal versus run-to-run noise.

**Dataset** proxima-fusion/constellaration &nbsp;·&nbsp; **Task** 92-dim → 11-target regression &nbsp;·&nbsp; **Rows** 158,685 (of 182,222 raw) &nbsp;·&nbsp; **Compute** Docker + single RTX 5000 Ada

---

## Where things stand

> **Current best-validated finding.** A plain 3-layer MLP is a strong, well-behaved baseline that nothing has clearly beaten in aggregate. The one architecture change that survives a real noise check is swapping the trunk's activation to **SIREN** (sinusoidal, Sitzmann et al. 2020): it wins `minimum_normalized_magnetic_gradient_scale_length` by roughly 7 standard deviations of measured run-to-run noise, and `average_triangularity` by roughly 3σ — confirmed across two independent seeds with no overlap against three baseline reruns. The same swap costs `aspect_ratio` accuracy (~4σ worse), so it's a genuine trade, not a free win. Every other direction tried — a spatial CNN branch, deeper per-target heads, chained sine/ReLU blocks, a mode-attention encoder, derived geometric features, log-space target heads, symlog-latent augmentation, and combinations thereof — either doesn't clearly beat the baseline once epoch budgets are matched, or hasn't been confirmed beyond noise.

---

## 1. Data pipeline

Everything runs in Docker (never on host) per project convention; two services — `preprocess` and `train` — share the same volume-mounted `./output` and `./checkpoints`.

| | |
|---|---|
| Raw rows | 182,222 |
| Dropped | 23,537 (malformed + NaN targets) |
| Final rows | 158,685 |
| Input dim | 92 |
| Output dim | 11 (of 12 raw) |
| Split | 80 / 10 / 10, seed 42 |

**Feature layout (92 dims).** `r_cos` (45 = 5 poloidal × 9 toroidal Fourier modes) + `z_sin` (45) + `n_field_periods` (1) + `is_stellarator_symmetric` flag (1). Verified live against the actual dataset rather than assumed — `r_sin`/`z_cos` are always null (every row is stellarator-symmetric), and mode-count shape `(5, 9)` is constant across all but one malformed row.

**Target: one column dropped.** `aspect_ratio_over_edge_rotational_transform` was removed from the target set. It's algebraically `aspect_ratio / edge_rotational_transform_over_n_field_periods` — both already separate, well-behaved targets — and blows up near-singular denominators: std of 9,882 against a mean of 68, with values up to 3.4 million on 0.7% of rows. No amount of normalization fixes a target that's fundamentally ill-conditioned; dropping the redundant, unlearnable column was more honest than forcing a model to chase it.

## 2. Training methodology

Several early instincts turned out to be wrong and were corrected mid-stream — worth recording why, not just what.

**No target normalization, anywhere.** Mean/std z-scoring, robust (median/IQR) scaling, and clipping/winsorizing outliers were all tried in reasoning and rejected. A heavy-tailed target has no normalization constant that stays representative across samples — the fix belongs in the loss, not the data. Every target head is trained on raw physical units, and cross-target scale differences are handled by **learned per-task uncertainty weighting** (Kendall & Gal 2018): `loss = Σ exp(-log_var_k)·MSE_k + log_var_k`. Each task's own `log_var` is a learned parameter, so the network discovers how much to trust each target's raw-scale error itself — no hand-picked constant, and it's what let a pathological target reveal itself as fundamentally unlearnable rather than just numerically unstable.

**No input normalization either — verified, not assumed.** Checked `metadata.json` feature statistics directly: the Fourier coefficients are already zero-mean, O(0.01–0.4), nothing heavy-tailed. Standardizing would have been solving a problem that doesn't exist.

**Validate periodically, not every epoch.** `--val-interval` controls cadence (default every 5–10 epochs, plus always the final epoch) — an unnecessary full forward/backward-free pass every single epoch was pure waste once runs stretched into the hundreds of epochs.

**Held-out test set, evaluated once.** Training loops over train/val only; at the end, the *best* checkpoint (by val loss) is reloaded and run once against `test.npz` — not the final epoch's weights, which are frequently past the point of overfitting.

**Matched epoch budgets — the recurring confound.** Several early comparisons (a 620-epoch dual-path run vs. a 300-epoch single-path run, epoch-mismatched breadth experiments) turned out to be comparing training time, not architecture. Every comparison from that point on standardized on **300 epochs**.

**Noise floor — measured, not assumed.** The same config was rerun 2–3 times with different seeds (`--seed`, added specifically for this) to get actual per-target run-to-run standard deviation before trusting any architecture's apparent win. See §5.

## 3. Architecture search, in order

**01 — Single-path MLP** *(baseline)*
3-layer trunk (hidden 256 → latent 128) → per-target heads. Simple by design, per explicit instruction not to over-engineer a first pass.

**02 — Dual-path: spectral MLP + flattened spatial branch** *(regressed)*
Added a second branch: apply the actual inverse Fourier series transform (`R,Z = Σ r_cos·cos(mθ−nφ), Σ z_sin·sin(mθ−nφ)`) onto a fixed grid, flatten, encode with an MLP, concatenate. `max_elongation` got meaningfully worse (0.302 → 0.407) — flattening a periodic 2D grid destroys the locality/periodicity that made the transform worth doing.

**03 — Dual-path: circular-padded CNN spatial encoder** *(recovered)*
Replaced flatten+MLP with a small CNN using `padding_mode="circular"` (θ and ζ are both periodic angles) plus a fusion layer after concatenation. Recovered most of the `max_elongation` loss and looked like a broad win — but the comparison ran at a different epoch count than what it was measured against.

**04 — Deeper heads + explicit priority weight** *(methodologically flagged)*
Upgraded per-target heads from a single `Linear` to a 2-layer MLP, and added a 3× fixed multiplier on `min_norm_grad_scale_length` on top of the learned uncertainty weight. Improved that one target — but skewing the loss function isn't an architecture comparison, and it wasn't at a matched epoch count either. Flagged and separated out rather than left to imply a false win.

**05 — Matched 300-epoch, 4-way rebuild** *(reset)*
Unified the codebase with a `--no-spatial` toggle so single- and dual-path share identical heads/fusion/loss, and reran everything at exactly 300 epochs: `single_base` (230,550 params), `single_bigger` (431,126), `dual_base` (312,454), `dual_bigger` (562,246). Nothing clearly beat the plain MLP. `dual_bigger` had the best aggregate loss but peaked at epoch 100/300 and didn't actually win a single individual target on the held-out test set — an early-overfit checkpoint inflating one summary number, a good caution against trusting aggregate loss alone.

**06 — Breadth: SIREN, half-SIREN, attention** *(SIREN promising)*
Three different trunk activations/topologies at matched capacity and epochs. **SIREN** (sinusoidal activation) won `average_triangularity` and `min_norm_grad_scale_length` outright, lost `aspect_ratio`. **Half-SIREN** (two parallel branches, split once) was smaller (148K params) and didn't clearly beat baseline. **Attention** (Fourier modes as tokens, positional embeddings, self-attention) was worst on every single target — plausibly needs more data/epochs/tuning to be competitive, not necessarily a dead end.

**07 — Chained half-SIREN blocks** *(overfits hard)*
Redesigned half-SIREN as a stack of blocks — each layer itself half-sine/half-ReLU, both halves seeing the full previous layer's output, so the two feature types mix at every layer instead of once at the end. Bigger (296,342 params) but peaked at epoch 60/300 and degraded sharply after — the worst overfit gap seen in the whole search. Mostly worse than baseline even at its best checkpoint; would need real regularization (none exists anywhere in this codebase yet) to be viable.

**08 — Derived geometric features** *(borderline)*
Computed 11 non-learned summary statistics (R/Z extent, an elongation proxy, per toroidal angle) from the same fixed IFFT reconstruction, concatenated onto the raw input. Modest win on `max_elongation`, a small consistent regression on `min_norm_grad_scale_length` — sensible, since the derived features are purely geometric and that target is a magnetic-field property.

**09 — SIREN + geometric features combined** *(didn't compound)*
Two independently-promising changes, tried together. Result: worse than either alone on almost every target where one had shown an edge — `aspect_ratio` worse than SIREN alone, `average_triangularity` reverted fully to baseline level, `max_elongation` worse than geo-features alone. A genuinely useful negative result: combining wins here interferes rather than adds.

**10 — Noise floor** *(signal confirmed)*
Reran the baseline 2 more times (different seeds) plus one repeat each of SIREN and geo-features, plus two seeds of the combo — 6 additional 300-epoch runs. See §5 for the numbers: this is what separated SIREN's real effect from everything else's noise-level fluctuation.

**11 — Log-space decoder experiments** *(backfired on its own target)*
Two ideas for targets that aren't necessarily a linear function of the latent: (a) `symlog(z)` concatenated onto the latent before all heads, and (b) predicting `log(target)` directly for the four strictly-positive, wide-dynamic-range targets (`qi`: ~4400×, `max_elongation`: ~114×, `flux_compression`: ~216×, `min_norm_grad_scale_length`: ~1.8×10⁷×). The targeted log-space heads made `min_norm_grad_scale_length` — the target with the strongest a priori case — **~10σ worse**, and `qi` ~7σ worse. Best explanation: log-space training optimizes for relative/tail accuracy, but the evaluation metric is physical-space RMSE, dominated by bulk/typical-value accuracy — the two objectives pull in different directions. `symlog`-latent alone was gentler (no large regressions, a couple of unconfirmed single-run positives). Combining the two made it worse again, consistent with #09.

## 4. Matched-epoch comparison (300 epochs, single seed each)

Test-set RMSE at each run's best checkpoint, raw physical units. **Bold** = best in column, *italic* = worst in column.

| variant | params | best ep. | qi | vacuum_well | aspect_ratio | max_elong. | avg_triang. | ax_mag_mirror | edge_mag_mirror | ax_rot/nfp | edge_rot/nfp | flux_comp. | min_norm_grad |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| single_base | 230,550 | 190 | 0.00659 | 0.01653 | 0.08390 | **0.43205** | 0.03759 | 0.01480 | 0.01683 | 0.01303 | 0.01057 | 0.07208 | 0.73323 |
| single_bigger | 431,126 | 170 | 0.00634 | 0.01657 | **0.07202** | 0.45545 | 0.03743 | **0.01354** | **0.01603** | 0.01209 | 0.00966 | 0.06826 | 0.69854 |
| dual_base | 312,454 | 300 | **0.00622** | **0.01586** | 0.07832 | 0.43551 | 0.03461 | 0.01463 | 0.01648 | 0.01213 | **0.00963** | **0.06619** | 0.70080 |
| dual_bigger | 562,246 | 100 | 0.00670 | 0.01658 | 0.07481 | 0.46792 | 0.03983 | 0.01444 | 0.01673 | 0.01234 | 0.01036 | 0.06871 | 0.70783 |
| siren_base | 230,550 | 170 | 0.00639 | 0.01846 | 0.09816 | 0.43429 | **0.03243** | 0.01503 | 0.01725 | **0.01209** | 0.01055 | 0.06924 | **0.68329** |
| mlp_geom_features | 233,366 | 190 | 0.00645 | 0.01666 | 0.07814 | **0.39884** | 0.03887 | 0.01509 | 0.01795 | 0.01299 | 0.01063 | 0.07182 | 0.74521 |
| chained_half_siren | 296,342 | 60 | 0.00642 | 0.01710 | *0.11076* | *0.47395* | 0.03672 | *0.01666* | *0.01847* | 0.01349 | 0.01188 | 0.06893 | 0.69059 |
| attention_base | 139,318 | 250 | *0.00780* | *0.02552* | 0.13055 | *0.84834* | *0.05894* | *0.02183* | 0.02636 | *0.01990* | *0.01498* | *0.08719* | *0.81095* |

All runs: 300 epochs, no target/input normalization, learned uncertainty-weighted loss, held-out test set, best-checkpoint selection.

### Log-space decoder experiments (300 epochs, single seed each)

| variant | params | best ep. | qi | min_norm_grad | notes |
|---|--:|--:|--:|--:|---|
| symlog_latent | 320,662 | 190 | 0.00660 | 0.75579 | gentle, a couple of unconfirmed single-run positives elsewhere |
| log_targets | 230,550 | 240 | 0.00719 | 0.82479 | ~7σ / ~10σ worse than baseline on exactly the targets it was meant to help |
| symlog + log_targets | 320,662 | 220 | 0.00690 | 0.83151 | combining did not rescue the regression |

## 5. Noise floor: what's real signal vs. run-to-run variance

Three identical `single_base` reruns (seeds only differ) give the actual measurement noise per target. Everything claiming a "win" gets checked against this before being trusted.

| target | baseline mean | run-to-run std | relative |
|---|--:|--:|--:|
| qi | 0.00665 | 0.00008 | ~1.2% |
| vacuum_well | 0.01737 | 0.00121 | ~7% |
| aspect_ratio | 0.07848 | 0.00491 | ~6% |
| max_elongation | 0.45381 | 0.01900 | ~4% |
| average_triangularity | 0.03919 | 0.00187 | ~5% |
| axis_magnetic_mirror | 0.01476 | 0.00015 | ~1% |
| edge_magnetic_mirror | 0.01700 | 0.00027 | ~1.6% |
| axis_rot/nfp | 0.01326 | 0.00049 | ~3.7% |
| edge_rot/nfp | 0.01085 | 0.00042 | ~3.9% |
| flux_compression | 0.07359 | 0.00132 | ~1.8% |
| min_norm_grad | 0.73772 | 0.00904 | ~1.2% |

### Checked against the floor

| claim | effect size | seeds agree? | verdict |
|---|---|---|---|
| SIREN wins min_norm_grad | ~7σ (0.673 vs 0.738) | yes — no overlap | **real** |
| SIREN wins average_triangularity | ~3σ (0.034 vs 0.039) | yes — no overlap | **real** |
| SIREN loses aspect_ratio | ~4σ worse (0.097 vs 0.078) | yes — no overlap | **real regression** |
| geo-features wins max_elongation | ~2σ (0.415 vs 0.454) | own spread nearly closes gap | borderline |
| geo-features loses min_norm_grad | ~2σ worse | consistent, modest | likely real, small |
| SIREN+geo combo helps anything | — | worse than either alone, most targets | **no — interferes** |
| log_targets on min_norm_grad (its own target) | ~10σ worse | single run, large enough to trust | **real regression** |
| log_targets on qi (its own target) | ~7σ worse | single run, large enough to trust | **real regression** |

σ = baseline's measured run-to-run standard deviation for that target. "No overlap" means the two seeds of the claimed effect sit entirely outside the range spanned by three baseline reruns.

## 6. Open threads

- **SIREN as the working lead** — worth a third seed for extra confidence, and worth trying at larger capacity now that it's confirmed real rather than noise.
- **Attention needs real investment or should be dropped** — worst performer as tested, but transformers are notoriously sensitive to warmup/LR schedule/data volume; wasn't given a fair shake.
- **Chained half-SIREN needs regularization** — dropout or weight decay before it can be fairly judged; currently overfits before its extra capacity pays off.
- **No regularization anywhere yet** — every model in this search trains with plain Adam + grad-norm clipping only.
- **geo-features signal is unconfirmed** — sits right at the noise floor; needs a third seed to know if it's real.
- **Log-space heads: ruled out as tried** — the naive "predict log(target) for wide-range targets" idea is a confirmed loss; if revisited, would need a training-space metric that matches physical-space evaluation (e.g. weighting log-space loss by target magnitude) rather than plain log-MSE.

---

*`scripts/preprocess.py` · `scripts/train.py` — all runs via Docker*

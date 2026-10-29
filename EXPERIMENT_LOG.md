# Experiment log — current state

This file was rewritten from 99 numbered sections down to one current-state summary per domain, on direct user request ("97 is way too much... I think you're going crazy from all the weird side tracks and bad prior thoughts"). **Nothing was deleted** — the full, unabridged section-by-section history (every bug, every reverted idea, every exact number) is preserved in `EXPERIMENT_LOG_ARCHIVE.md`. This file is meant to answer "what's actually true right now and why," not "what happened in what order." If this file falls behind (another concurrent session keeps appending numbered sections the old way), that's fine — re-consolidate again rather than trying to prevent it.

A recurring, cross-domain lesson worth stating once instead of N times: **verify by reading the real convergence/failure signal from source, not from documentation or assumption** — every domain in this project (VMEC++, XFOIL, TORAX, OpenFOAM, Warp) has had at least one case where "it ran without an error" and "it produced something real" turned out to be different claims, caught only by checking actual numbers/geometry directly.

---

## Stellarators (VMEC++) — most mature domain, full A/B/C/D bar closed

**Task**: 92-dim boundary Fourier coefficients → 11-target regression. Dataset: proxima-fusion/constellaration, 158,685 rows.

- **The random 80/10/10 split overstates generalization substantially** — a cluster split (holding out entire regions of coefficient space) costs 2.5-3x the RMSE, and the "best" architecture under the random split does not clearly win under the cluster split. Every number from before this finding should be read as an optimistic upper bound; *relative* architecture rankings are more trustworthy than absolute numbers.
- Plain MLP is a strong baseline; SOAP optimizer + wide MLP + ensembling wins in aggregate. Target z-scoring meaningfully helps the two worst-generalizing targets (rotational transform) under the cluster split.
- **VMEC++ integrated as a ground-truth oracle** (pip-installable, ~6-14s/call low-fidelity) — confirmed a real design-search "feasible" candidate was NOT actually feasible (2 of 3 constraints violated, `qi` off by 327%) despite low surrogate disagreement. Oracle now checks every design-search candidate before trusting it.
- Bootstrap/self-training loops (VAE-generated, VMEC++-validated designs added back to training data) show a real, specific, non-uniform effect: rotational-transform targets improve substantially with more synthetic data, magnetic-mirror-ratio targets get worse, and the aggregate mean is not a reliable single number to optimize against — not a case for blind augmentation.
- Gradient-guided latent-space search (`gradient_walk.py`) reached `max_elongation=1.327`, beating the ConStellaration paper's own reported 1.27 baseline on that one target.
- **Cross-domain architecture ranking, confirmed repeatedly across every domain built in this project: GAN > Diffusion > cVAE** on validity and steering signal, with the GAN's generator param-matched exactly to the cVAE's decoder (not a capacity advantage). Diffusion's inference cost is ~200x the other two (200-step ancestral sampling vs. one forward pass) — worth stating next to any diffusion-wins number.
- DESC (second equilibrium code) needed an explicit trig-identity fix to a naive VMEC→DESC coefficient copy (was dropping the `sin·sin` mode partner) before it agreed with VMEC++ at all; now correlates at 0.87 on |ι|.
- Full A/B/C/D bar (oracle/dataset/scoring/generative) closed; viewer built with 3D shape + live physics.
- **Official ConStellaration P1/P2/P3 benchmark problems** (`constellaration.problems`, not this project's own approximations) now targeted directly via gradient-ALM search against a frozen surrogate ensemble + frozen VAE decoder. **P1 solved cleanly and repeatably** — real VMEC++-confirmed feasible designs, `max_elongation` 2.65-2.80 depending on scorer architecture. **P2/P3 blocked by a genuine, confirmed data problem, not a search-technique bug**: `qi` needs to reach 1e-4 (P2) / 3.16e-4 (P3), 1-1.5 orders of magnitude past anything in the real dataset — and, more specifically, the region of design space with very low `qi` and the region satisfying every OTHER P2/P3 constraint (aspect_ratio, iota, mirror, elongation/flux_compression/vacuum_well) barely overlap: minimizing `qi` alone while holding the other constraints only reaches real, VMEC-confirmed `qi`≈0.0036-0.014, still 12-140x too high.
- **Cluster-split-trained scorers must not be used for real search** (explicit standing decision) — good only for honest generalization diagnostics. Confirmed concretely: the cluster-split-trained MLP ensemble that looked fine in isolation was dramatically worse at the one problem known to be solvable at all (P1 predicted `max_elongation` 5.8-6.3 vs. the full-data ensemble's 2.8, 0/50 vs. historically-reliable real-feasible).
- **Held-out RMSE and downstream real-oracle search quality are demonstrably not the same axis** — confirmed a second, independent way: half-SIREN (same hidden=2048/latent=512 recipe as the production MLP, trained on the FULL dataset, 3 seeds) beats MLP at real search quality (P1 `max_elongation` 2.65 vs. 2.80, both real-feasible; P3 standardized worst-constraint-violation +0.378 vs. +0.442) despite no clean RMSE-based case for preferring it over MLP. Picking a production scorer by held-out RMSE alone is not sufficient; real-oracle search battery (`run_p123_battery.sh` + `summarize_p123_distance.py`) is the actual test.
- **Free push vs. staged search on P3, same architecture/ensemble/seeds — an important, clean negative result for staging as currently built.** Three strategies compared: (1) *free push* — minimize `qi` directly, other 4 constraints held fixed, one 40-iteration ALM stage; (2) *continuation* (`--qi-schedule`) — all 5 constraints enforced throughout, only `qi`'s own bound ratchets 3e-2→3.16e-4 across 5 sequential stages (200 total outer iterations), `z` warm-started between stages; (3) *sequential-constraints* (`--sequential-constraints`) — start with ONLY `qi` active (ramped the same way), then add the other 4 constraints one at a time cumulatively, each its own full stage (360 total outer iterations), nothing ever frozen once added. Real-oracle result: free push converged 89/100 (89%) in real VMEC++; **both staged approaches converged 0/20 and 0/9 (0%)** — not "less feasible," no real equilibrium at all — despite a new ensemble-disagreement trust gate (`--trust-gate-multiplier`: candidate std vs. the ensemble's own real-in-distribution-data baseline std, computed fresh each run) rejecting ~80% of each staged run's naive "surrogate-feasible" candidates first. **The trust gate helps but is not sufficient alone** — it catches members actively disagreeing, not all three members being confidently, consistently wrong together (a shared blind spot), which is what the staged survivors turned out to be. Leading hypothesis, not yet confirmed: this is a **total optimization budget** effect (200-360 outer iterations for staged vs. 40 for free push) rather than a staging-strategy problem specifically — any approach spending this many gradient steps against a frozen surrogate may eventually find and exploit a shared blind spot regardless of how the steps are organized. Isolating control (run free push for the same 200-360 total iterations, no staging) queued, not yet run.
- `generate_candidates.py` gained `--qi-schedule` (continuation/homotopy on one constraint's bound), `--sequential-constraints` (grow the active constraint set one at a time instead — hardest constraint, i.e. `qi`, first), and `--trust-gate-multiplier` (hard ensemble-disagreement gate baselined against real training data, distinct from the pre-existing soft `--trust-weight` loss penalty) — all three composable.
- **Open question, not yet resolved**: given the total-optimization-budget framing above, whether to keep tuning the ALM approach (cap total steps, lean on the curriculum real-data retrain loop, or add periodic real-VMEC spot-checks mid-search as a stronger trust signal than ensemble disagreement alone) vs. reframing the whole P2/P3 push as an RL problem (a policy trained against the same frozen surrogate would face the identical frozen-proxy-exploitation vulnerability unless it also gets one of those same groundings — so RL is not obviously a fix for this specific failure mode, though a trained, reusable conditioned policy would be a different, independently useful capability from "solve this one instance").
- **Grounded-walk post-vacation continuation (2026-09-11/12), P3, nfp=2 and nfp=4**: resumed the two best vacation-sweep lineages as true continuations (existing z/rho/y state, not fresh anchors) — `p4_halfsiren_v1` (nfp=4, 1 known real qi-pass) and `p2_halfsiren_v1_gpu1_seed1` (nfp=2, best near-miss: qi/vacuum_well/flux_compression all pass, 3/5 constraints). Confirmed via retrain-cycle-vs-row-count cross-reference that both lanes' active scorers are now trained on all known qi-passing/near-miss real rows found so far — no leakage concern, the search isn't blind to its own best finds.
- **A/B test: proximity-biased anchoring + per-fiber death/respawn vs. the original eternal-fiber/random-anchor algorithm — negative result.** Built `grounded_walk_step_v2.py` (anchors drawn from the real pool's closest-to-feasible shortlist by real constraint violation, not uniform random; each fiber tracks its own stall counter and gets killed/respawned after 40 stalled rounds — half as a jiggled clone of the current best fiber, half as a fresh proximity-biased draw) and ran it head-to-head against the plain `p2_halfsiren_v1_gpu1_seed1` control, same nfp/constraints/scorer, colocated on the same GPU. Killed after 183 rounds / 9 retrains as a clear negative: accept rate collapsed to and stayed at 3% (1/30) for 15 straight rounds (vs. the control's stable 87-97% at a comparable early round count), best-ever real violation got *worse* over the run (0.26 → 0.38, population mean 2.3 → 9.7) rather than improving, and its real pool's 4/5-constraint hit rate (19/1,805 = 1.05%) came in worse than both the nfp=4 control (177/11,030 = 1.6%) and well below the nfp=2 control (647/10,410 = 6.2%). Read: constantly killing and reseeding fibers into a proximity-biased region the current surrogate can't reliably validate yet causes thrashing (high ALM-rejection rate) rather than concentrating search productively — the mechanism churned instead of converging. Standing conclusion: **for this search, an eternal fiber population with uniform-random real-anchor initialization outperforms aggressive proximity-biased respawn** — breadth-via-restart is not obviously better than just letting an established fiber keep taking small steps.
- **Post-vacation continuation review (2026-09-12): best real lineage hasn't actually improved despite ~250 more rounds.** `p2_halfsiren_v1_gpu1_seed1`'s best candidate is still row 4126 (qi/vacuum_well/flux_compression pass; edge_rotational_transform is close, off by only 0.021; edge_magnetic_mirror_ratio is NOT close, running ~0.28-0.31 over the 0.25 limit — the real remaining wall). Only 3 of 10,410 real pool rows pass `qi` at all, and only 2 of those also pass vacuum_well+flux_compression — that set hasn't grown. What DID grow (647 rows at 4/5 constraints) turned out to be a different, easier basin (mirror+rotational-transform+vacuum_well+flux_compression satisfiable together, none of them passing qi) — real breadth, but in the wrong direction for closing the last gap. `p4_halfsiren_v1` (nfp=4) is in worse shape (its one qi-passer doesn't even satisfy vacuum_well or flux_compression). Both lanes' budgets were extended (MAX_RETRAINS 20→40, MAX_ROUNDS 2000→3000, true continuations from existing z/rho/y) rather than let them hit their original caps and go idle.
- **v3, live: quality-diversity/novelty bonus, testing "same distance from ideal but a different direction still has value."** Direct response to the hypothesis that all 30 fibers optimizing the identical ALM objective from different starts tend to climb the same ridge rather than fan out across the constraints actually blocking full feasibility (mirror_ratio, edge_rotational_transform) — which would explain why the mirror_ratio gap hasn't moved despite hundreds of rounds. `grounded_walk_step_v3.py` adds (1) a small novelty bonus to each fiber's ALM loss rewarding distance, in normalized (mirror_ratio, |edge_rotational_transform|) space, from the nearest already-known-good real point (pool rows passing ≥3/5 constraints, `--archive-min-pass`) — weighted small (`--novelty-weight 0.05`) so ALM feasibility pressure still dominates; fibers stay eternal (no death/respawn — that's the part v2 showed thrashes), only the loss is perturbed; and (2) direct per-round logging of how many cells of a coarse grid over that same space are occupied (`coverage_bins_occupied` in `round_log.jsonl`) — a real, direct "how much of the space have we explored" metric, logged regardless of whether the bonus helps. Archive pre-seeded from `p2_halfsiren_v1_gpu1_seed1`'s existing real pool (copied into the new state dir before first launch) so novelty is meaningful from round 1 instead of needing its own rounds to build one up; anchor initialization itself is still a fresh independent random draw (`seed 5`), not warm-started. Launched 2026-09-12, replacing the `seed4` plain-recipe duplicate that had briefly occupied this slot (killed after <1hr, no real signal collected — a pure breadth duplicate was a worse use of the slot than testing this new hypothesis). Too early for a read; watch `coverage_bins_occupied` trend and whether the mirror_ratio gap on its best real candidate actually closes vs. the `seed1` control over a comparable number of rounds.
- **v3 after 61 rounds/1 retrain: coverage hypothesis not supported so far, and a second lane independently confirms the qi-vs-mirror_ratio tension is structural, not a coverage gap.** `coverage_bins_occupied` stayed at exactly 30/225 for all 61 rounds while the archive (≥3/5-pass real rows) grew 3,545→3,567 and the pool gained 1,510 new real rows — every new archive-worthy point landed in an already-occupied cell, none opened new territory on the (mirror_ratio, |ι|) grid this run tracked. Process health remains fine (accept rate 63%→83%, no thrashing, unlike v2) and it's still only 1 retrain cycle, so this isn't final, but the one metric built specifically to test the diversity hypothesis hasn't moved yet. Separately, `p4_halfsiren_v1` (nfp=4, independent lineage) found its own new qi-passing row (908: qi 0.00028, mr 0.484, rt 0.232) — worse mr than the 0.25 target but notably better than the nfp=2 lineage's 0.53-0.56 cluster, yet still fails vacuum_well and flux_compression. Across two independent nfp settings and four lanes, **every qi-passing real point found anywhere in this project so far has a distinctly worse mirror_ratio than the pool's general population** (which is ~35% mr-passing overall) — the qi-vs-mirror_ratio tension looks like a real local property of the qi-passing region, not a search-coverage artifact, though the nfp=4 point being closer to the mr threshold than any nfp=2 point is the first hint it might be loosenable rather than fixed.
- **v3's coverage metric replaced: a 2D axis-aligned occupancy grid is the wrong shape of instrument.** The original `coverage_bins_occupied` metric only looked at 2 of 5 constraints (an arbitrary mirror_ratio × |ι| projection), used fixed bins with no distance/connectivity notion, and couldn't distinguish "the known-good region got denser" from "a second, disconnected region appeared." Replaced with real point-cloud topology, computed each round over the full 5-constraint (std-normalized) archive: a k-NN graph's connected-component count + largest-component fraction (is the known-good set one blob or several separate pieces), plus mean/median nearest-neighbor distance as a continuous, bin-free dispersion measure (`grounded_walk_step_v3.py`, `--topology-knn`, default k=5; `round_log.jsonl` gained `archive_n_components`, `archive_largest_component_frac`, `archive_nn_dist_mean`, `archive_nn_dist_median`, replacing `coverage_bins_occupied`). Confirmed on the live data before deploying: v3's 3,569-point archive forms ~7-9 connected components (one giant blob at 97% + several 7-42-point satellites), and critically, **all of the archive-qualifying points v3 has found on its own since launch landed in the same giant blob** — none created or joined a satellite component. This is a dimension-honest, connectivity-aware version of the same negative read the flat 2D grid gave, not an artifact of that grid's crude binning. Note this diagnostic is still short of real persistent homology (Betti-1 loops specifically would catch a "surrounded but unexplored pocket" that plain connectivity can't) — flagged as a follow-up if the connectivity read stays ambiguous, not done yet (would need `ripser` added to the image). The novelty bonus itself (the actual training-loss term) is unchanged — still the 2D mirror_ratio/|ι| descriptor; swapping it for graph-geodesic distance over the known-good archive is queued as a further step, contingent on this diagnostic actually justifying it.
- **v3 retired 2026-09-13, confirmed negative on its own instrumentation; replaced with v4, which directly attacks the qi-vs-mirror_ratio correlation instead of generic diversity.** After 217 rounds/13 retrains, v3's own topology metric (connected components + NN dispersion over the full 5-constraint archive, added specifically to test this) showed its known-good archive MORE consolidated into a single blob (3 components, 99.7% largest) than the plain control's (7 components, 99.1% largest) — the opposite of what a working diversity mechanism should produce. Two independent metrics (the earlier 2D coverage grid and this real topology read) now agree: the novelty bonus never demonstrated it opens new territory. Standing conclusion: neither v2's proximity-bias nor v3's generic novelty bonus touched the actual mirror_ratio axis specifically — killed and replaced with `grounded_walk_step_v4.py`, which does. v4 anchors fibers ONLY at the project's 3 known qi-passing real points (`--anchor-x-file`, `--anchor-jitter` for small nearby clusters instead of exact duplicates) and directly minimizes `edge_magnetic_mirror_ratio` (`--minimize`, already supported) with ONLY `qi` enforced as a constraint — deliberately isolating the two-way question before chasing all 5 constraints together. A real-VMEC pilot (15 fibers) before wiring this into the queue already shows the tension cleanly: round 1 dropped mirror_ratio from ~0.53–0.56 to 0.37–0.51 across all 15 fibers, but qi blew up 6–13x past its threshold in every one; as the ALM multiplier pulled qi back over the next several rounds (best_real_violation min: 0.21 → 0.13 → 0.033 by round 5), the fibers recovering the most qi margin gave back the most mirror_ratio gain — the 5 real rows closest to qi-feasible by round 5 all sit at mirror_ratio 0.57–0.63, *worse* than the original 0.53–0.56 anchors, not better. Real, oracle-confirmed evidence of a local Pareto trade-off, not a search-coverage artifact. Launched as the permanent `p2_v4_correlation_gpu1` queue slot (continuing from the pilot's already-collected real rows, not restarting); the open question this full run is meant to answer is whether continued ALM pressure ever finds a point materially better on this trade-off curve (qi feasible AND mirror_ratio meaningfully below 0.53) or just oscillates along the same curve indefinitely — the latter would be a strong, specific, falsified-by-data answer to "is this a hard wall."
- **v4 (nfp=2) finished, 12 rounds: a real, quantified local Pareto wall, not a coverage gap.** The driver hit its own success condition and stopped (qi was the only enforced constraint) after finding a real point with qi=0.000274 (comfortably under the 0.0003162 threshold) — but that point's mirror_ratio is 0.689, the worst value recorded anywhere in this project, well past even the original 0.53–0.56 anchors. Across all 179 real points this run explored, qi and mirror_ratio correlate at **r=-0.95** — a clean, near-linear local trade-off curve (mr running ~0.37 at the worst-qi end up to ~0.69 at the best-qi end). Nothing found beat the original 3 anchors on both axes simultaneously; 12 rounds of real ALM pressure just traced back and forth along one tight curve. Read as a genuine local Pareto wall specific to this nfp=2 basin.
- **Follow-up launched same day: is the wall basin-specific or universal?** The nfp=4 lineage's lone qi-passer (row 908, mirror_ratio 0.484) is the only known qi-passing point anywhere with a meaningfully better mirror_ratio than the nfp=2 cluster — the one hint the correlation might loosen elsewhere. Re-ran the identical single-anchor, single-constraint free-push (`grounded_walk_step_v4.py`, `--nfp 4`, same recipe) from that one point, now running as `p2_v4b_correlation_nfp4_gpu1` in the same queue slot (queue file keeps both entries in sequence, so the nfp=2 result's provenance stays on record). Framing going forward, per direct discussion: treat discovered Pareto walls as informative boundaries/edges of the feasible region to map systematically (which constraint pairs correlate strongly and where), not just obstacles to push through one at a time.
- **Third basin added same day: nfp=3, the most promising one yet, as a genuine 4th concurrent lane (`gpu1v3`).** nfp=3 has no qi-passing real point in its 15,561 real rows, but its closest miss (qi=0.00034, only ~7% over threshold) sits at mirror_ratio 0.387 -- meaningfully better than the nfp=2 basin (0.53-0.56) or the nfp=4 anchor (0.48) -- and the whole-pool qi-vs-mirror_ratio correlation is a much weaker -0.53 (vs. -0.95 traced along the nfp=2 ALM trajectory). Anchored at the 3 best near-miss real points, same exact recipe as the other two v4 runs (minimize mirror_ratio, hold qi<=0.0003162 as the sole constraint) for a clean comparison. **First attempt had a real bug, caught immediately**: initially tried the mirror-image framing (minimize qi directly, constrain mirror_ratio<=0.5) -- but 0.5 was already trivially satisfied at all 3 anchors, so the driver's "success" trigger (all enforced constraints satisfied) fired after a single round without qi or the objective ever being meaningfully tested. Reset the state dir and relaunched with the corrected, comparable recipe; confirmed round 1 now shows real objective pressure (13/15 fibers engaged, best_real_violation starting at a genuine 0.17, not an instant false success). `vega_queue_runner.sh` gained a true 4th lane (`gpu1v3`/`queue_gpu1_v3.txt`) rather than sequencing this behind the nfp=4 run, since each job is small (15 fibers) and GPU1 has headroom (~15GB of 32GB across all three colocated jobs now).
- **nfp=3 finished: third confirming basin, the promising whole-pool signal did not survive direct ALM pressure.** 15 rounds to real SUCCESS — qi=0.000235 (comfortably clears 0.0003162) but at mirror_ratio=0.672, no better than the other two basins and right in their same 0.67-0.69 range. Across the 221 real points explored, qi-vs-mirror_ratio correlation r=-0.87, close to nfp=2's -0.95. Taken with nfp=2 (r=-0.95, mr 0.69) and nfp=4 (r in -0.76 to -0.80 as it progressed, no qi-pass yet), **three independent basins now show the same strong anticorrelation and every qi-feasible point anywhere in the project (nfp=2/3/4) lands at mirror_ratio 0.48-0.69** — looking increasingly like a structural property of this design family, not a local search-coverage gap in any one basin. A global scan across every real_pool_Y.npy in the project (all lanes, all nfp, including an nfp=5 lane not otherwise tracked) confirms nothing anywhere has ever broken this: best mirror_ratio among all qi-feasible real points project-wide is still 0.4845 (the original nfp=4 anchor).
- **NAE (near-axis-expansion) construction added as a genuinely different search direction, per direct user request to test whether the wall is a search-topology artifact vs. a real constraint.** Everything tried so far (v2/v3/v4) moves *continuously* from a real anchor via VAE-latent gradient steps — it can trace whatever local Pareto curve it's already on but can never discover a disconnected region of coefficient space. `constellaration.initial_guess.generate_nae()` (part of the official constellaration package, used in its own `data_generation/vmec_optimization.py`, never called by this project before now) instead *constructs* a boundary analytically via near-axis expansion (Goodman et al. 2023) that is QI-by-construction to first order, with `mirror_ratio` as a literal input dial rather than something read off after an optimization. Confirmed the representation is exactly compatible, not merely convertible: `SurfaceRZFourier.r_cos`/`z_sin` (shape (5,9), m=0..4 × n=-4..4) IS this project's own `r_cos`/`z_sin` layout (`scripts/preprocess.py`'s flatten convention, `physics_bundle.py`'s Fourier-convention note) — no resampling or refit, a genuine 1:1, verified directly in `scripts/nae_pilot.py`'s `surface_to_row()`.
- **NAE pilot (`scripts/nae_pilot.py`), 48 configs (nfp 2-5, mirror_ratio target 0.15-0.30, aspect_ratio 6-10), real VMEC++ validated, no surrogate**: 37 constructed OK (11 failed inside `generate_nae`'s internal root-find, all at the more extreme nfp=4/5 × low-mirror-ratio combos — `f(a) and f(b) must have different signs`), 19 converged. **The qi-vs-mirror_ratio correlation flips sign here: r=+0.70**, the opposite of every gradient-search basin — real confirmation this construction method lands in a qualitatively different region of coefficient space, not just a noisier resample of the same basin. But it starts further from feasible overall: best qi found is only 0.0095 (~30x over threshold, vs. our searches' 0.0002-0.0004), and NAE seeds also fail two constraints the gradient searches were never fighting — vacuum_well is negative everywhere (need >=0, worst -0.19) and flux_compression is consistently >1.0 (need <=0.9) — `generate_nae` controls QI/iota/elongation directly but has no dial for either. 0/19 satisfy all 5 P3 constraints on their own; this was never expected to be a finished answer, just a differently-seeded starting point.
- **Launched as a 5th concurrent lane, `p3_v4d_nae_seed_gpu1`**: the best composite NAE seed (nfp=2, aspect_ratio=6, mirror_ratio_target=0.25 → real oracle qi=0.0095, mr=0.347, vacuum_well=-0.185, flux_compression=1.21) converted losslessly to a `--anchor-x-file` and pushed with `grounded_walk_step_v4.py`, but unlike every basin so far — which only had qi open — this seed has 3 of 5 constraints open, so this run enforces the FULL P3 constraint set and maximizes the real P3 objective (`minimum_normalized_magnetic_gradient_scale_length`), not a single-constraint free-push. First round already shows real, substantial movement (best_real_violation min: clamped 10.0 → 0.68 in round 0→1, 15/15 fibers converged and accepted) — too early to read the eventual mirror_ratio outcome, this run has more ground to cover than the earlier single-constraint pushes did. Added to `queue_gpu1_v3.txt` as its second entry (same lane, `gpu1v3`, reuses the existing 4th-lane infrastructure — no `vega_queue_runner.sh` changes needed since appending to an exhausted queue file just advances its index on the next cron tick).
- **NAE-seeded run (`p3_v4d_nae_seed_gpu1`) at round 88/1286 real points: the other 4 constraints get largely solved, but the wall reasserts itself on qi specifically.** mirror_ratio, vacuum_well, and flux_compression all individually reach comfortably inside their P3 thresholds somewhere in the pool (mr down to 0.135, vw up to +0.034, fc down to 0.68 — none of the gradient-only basins ever got mirror_ratio anywhere near 0.25), but qi stalls at ~0.0020 (barely moved from ~0.0021 over 88 rounds), still ~6x over threshold, and 0 points are full-P3-feasible. At the pool's own best-qi points specifically, mirror_ratio sits at 0.336-0.391 — clearly better than every gradient basin's 0.48-0.69, but the qi-vs-mirror_ratio correlation across the whole pool is back to **r=-0.73**, same sign as every other basin. Read: ALM pressure dragged this qualitatively different (NAE-constructed) starting point back onto the same underlying wall, just re-centered at a better mirror_ratio — a 4th independent confirmation the wall is structural, not a search-coverage artifact of any one starting basin, but not the "genuinely disconnected, wall-free region" outcome hoped for either.
- **Two new, genuinely different attacks launched same day (2026-09-14), per direct user request** ("attack qi directly" + "use the generative models to probe/jump past the wall"):
  1. **`p3_v4e_qi_attack_gpu0`, a new 6th lane**: anchored at the NAE-seeded run's own 5 best-qi survivors (qi 0.00199-0.00217, mr 0.370-0.387 — the tightest cluster), directly minimizes qi while holding mirror_ratio<=0.4 (loose, not required to hold at its current best) plus vacuum_well/flux_compression/iota as constraints. Tests whether qi can be pushed down an order of magnitude from this already-better-mirror_ratio region without sliding back toward 0.48-0.69. First launch attempt used 20 anchors with 15 fibers and crashed immediately (`grounded_walk_step_v4.py` requires `--n-fibers` to be a multiple of the anchor count) — fixed by trimming to the best 5 anchors (15 divides evenly), relaunched clean, round 1 confirmed healthy (15/15 accepted). New `gpu0v2` lane in `vega_queue_runner.sh` (GPU0 had full headroom, ~5.5GB of 32GB in use).
  2. **Generative-model wall probe (`scripts/nae_generative_wall_probe.py`)**: a third kind of "jump," distinct from both NAE's closed-form construction and ALM's continuous local search — directly *asking* the project's three existing target-conditioned generative models (cVAE/diffusion/GAN, `train_cvae.py`/`train_diffusion.py`/`train_gan.py`) for a design whose full 11-target vector already sits on the far side of the wall (qi=1.5e-4, mirror_ratio=0.18, vacuum_well=0.05, flux_compression=0.6, iota=0.3·nfp jointly, all within the real dataset's observed range — not extrapolating), oracle-validated for real. First retrained all three on the richest data available: `scripts/merge_all_real_pools.py` combined the base archive with every grounded-walk lane's accumulated real pool (18 lanes, +94,984 rows → 521,761 total train rows, `output/splits/all_grounded_walk_v1`) so the models have seen dense real sampling right up against the wall from many angles, not just the original archive. All three (`cvae_targets_allgw_v1`, `diffusion_targets_allgw_v1`, `gan_targets_allgw_v1`) trained concurrently on GPU1. Hit two container-plumbing bugs before it actually ran: `oracle-x86` has no torch (can't load the generative checkpoints), and re-invoking `python3` against `oracle-x86`'s own `entrypoint: ["python3"]` tried to open a literal `/work/python3` file; fixed by running it in the `train` service instead, which (per `Dockerfile.train`) has both torch AND vmecpp/constellaration installed. **Result: clean negative.** 480 real-oracle checks — cVAE 22/160 converged (14%), best qi 0.0058 (~18x over threshold); diffusion 7/160 (4%), best qi 0.011; GAN 0/160 converged at all. 0 across all three ever got near qi-feasible, let alone jointly P3-feasible. Read: asking a model to conditionally-generate a point where real qi-feasible examples barely exist near the requested mirror_ratio is mostly out-of-distribution for it — pattern-matching from ~522k real points didn't produce a shortcut past the wall, at least not from a single direct "give me one at this exact target" request (a steerability-style incremental nudge from a real anchor wasn't tried here and remains a possible follow-up).
- **`p3_v4e_qi_attack_gpu0` corrected after a real bug, then showed genuine two-axis progress.** First launch (recipe above) repeated the *exact* "trivial success" mistake first caught on nfp=3: left qi unconstrained, so the loose constraints (mr<=0.4, vw>=0, fc<=0.9, iota>=0.25) were already satisfied by the anchors, and the driver declared SUCCESS after round 1 with qi=0.00289 — no real pressure, no real progress. Corrected to match every other basin's recipe: minimize `edge_magnetic_mirror_ratio`, constrain `qi<=0.0003162` for real (the anchors miss it by ~6x). Relaunched clean. By round 17: best real point qi=0.0010, mr=0.435 — **qi down 2x from the NAE-seed anchors while holding mirror_ratio well under the 0.48-0.69 gradient-basin range** (though above the 0.37-0.39 it started from) — the first time any basin has moved meaningfully on both axes at once rather than trading one for the other.
- **Stall audit across all 6 lanes, 2026-09-14, per direct user request ("make sure they're all still useful, or get rid of the ones that aren't").** The driver's own printed/self-tracked "stall" bookkeeping (`stall_state.json`'s `rounds_since_improvement`) turned out to be an unreliable signal — it kept reporting near-zero rounds-since-improvement on lanes that had, by the metric that actually matters (the worst-violated of the 5 real P3 constraints among any point in the pool, computed directly from `real_pool_Y.npy`, not read off driver-log text), been completely flat for hundreds of rounds. Likely tracking a softer internal signal (e.g. training loss) that kept moving even once the real constraint wall stopped yielding. Checked all 6 directly:
  - **`p4_halfsiren_v1` (gpu0, the main nfp=4 hunt): STALLED.** Worst-violation frozen since ~50% through its 19,233-row real pool (~row 9,616), still running at round 910 with zero further movement since. **Retired** — killed the driver + its retrain container, wrote a `_FAILED` marker so the watchdog won't auto-restart it, documented in `queue_gpu0.txt`. GPU0 is now fully free.
  - **`p2_halfsiren_v1_gpu1_seed1` (gpu1, the main nfp=2 hunt): STALLED.** Worst-violation frozen since ~30% through its 17,200-row real pool, 813+ rounds with zero further progress. **Retired**, same treatment, documented in `queue_gpu1.txt`.
  - **`p2_v4b_correlation_nfp4_gpu1` (gpu1v2): STALLED, and already answered its own question.** Worst-violation flat since ~25% through its 1,617-row pool; the question it existed to test (does the nfp=4 basin ever find even one qi-passing point?) had already been answered — no, 0/1,617 — and three other basins (nfp=2, nfp=3, NAE-seed) have since confirmed the same structural wall more informatively. **Retired**, documented in `queue_gpu1_v2.txt`.
  - **`p3_v4d_nae_seed_gpu1` (gpu1v3): genuinely still working, kept running.** Violation trend 19.5→7.7→5.6→5.3 through 90% of its run, only flattening in the last 10% — real if slowing progress, not a stall yet.
  - **`p3_v4e_qi_attack_gpu0` (gpu0v2): kept running, currently the most promising lane** — see the qi-attack progress note above. Too young (round 50) to call any recent flatness a stall.
  - Net effect: down from 5 active lanes to 2, freeing GPU0 entirely and GPU1's base allocation (its NAE-seed lane still runs colocated) — no replacement queued yet for the freed capacity, pending direction on what to run next.
- **Freed compute redeployed same day, per direct user question ("highest value use of that compute?") rather than just adding more fibers to the existing qi-attack lane.** Mined the original NAE pilot sweep (`output/nae_pilot/results.jsonl`, still only ever used once — the single lowest-qi seed) more thoroughly: re-sorting its 19 converged rows by *measured* mirror_ratio instead of qi shows the `mirror_ratio_target=0.15` configs land far closer to the true P3 wall than the `target=0.25` seed already in use (measured mr=0.347) — nfp=2/aspect_ratio=8/target=0.15 measures at **mr=0.258**, closer to the 0.25 threshold than anything found anywhere in this project by gradient search (0.48-0.69) or the target=0.25 NAE seed's own ALM push (0.347→0.435 so far). Mechanistic reason, not a fluke: `generate_nae`'s `mirror_ratio` argument is a first-order analytic target, and the full VMEC++ equilibrium's actually-realized mr systematically overshoots it — dialing the request lower than the true threshold compensates for that gap. qi is worse at these seeds (0.019-0.030 vs 0.0095), but qi is exactly the axis the qi-attack lane already knows how to push; landing this close to the mirror_ratio wall in the first place was the missing piece. Confirmed `constellaration`'s NAE construction has no exposed expansion-order parameter (checked `near_axis_configuration.generate`'s and `generate_nae`'s signatures directly) — a genuine higher-order-NAE idea isn't available without new upstream development, so not pursued this pass. Launched 3 new lanes from 3 diverse under-used seeds (same corrected minimize-mirror_ratio/constrain-qi recipe as `p3_v4e_qi_attack_gpu0`): `p3_v4f_nae2_nfp2ar8_gpu0` (mr=0.258 seed) and `p3_v4g_nae2_nfp2ar10_gpu0` (mr=0.279 seed, aspect_ratio=10) on the now-free GPU0, plus `p3_v4h_nae2_nfp4ar10_gpu1` (mr=0.341 seed, nfp=4, and the only NAE seed anywhere that already satisfies flux_compression on its own, 0.897) colocated on GPU1. All 3 confirmed launched clean (round 0 shows real ALM pressure, `best_real_violation min=10.0000`, not another trivial-success false-positive). Now 5 active lanes total across both GPUs.

- **Status check at round 65-146 across all 5 lanes (same day)**: none stalled. `v4g` (mr=0.279 seed) hit a new project-best measured mirror_ratio of **0.263** and `v4h` (mr=0.341 seed) hit a new project-best qi of **0.00035** (right at the 0.0003162 threshold) — not jointly feasible yet (still trading off, per-lane qi-vs-mr correlation over the last 400 real rows: v4d -0.67, v4e -0.53, v4f -0.31, v4g -0.79, v4h -0.38), but the wall is being approached from both sides simultaneously for the first time.

- **Literature review + two follow-up experiments, requested directly ("is our train/push methodology strong? other paths? kick off a lit review").** Checked recent arXiv (2206.08417, 2409.20328, 2306.17506, 2506.22166, 2608.16938) for anything structurally different from local ALM search. Two ideas were concrete enough to test same-day:

  **(A) Direct QI construction via a stricter local QI-consistency order.** Confirmed by reading source directly: `constellaration.mhd.near_axis_configuration.near_axis_configuration_to_pyqic` hardcodes `qic.Qic(..., order="r1", ...)` — there is no true 2nd-order-in-r (r2) NAE construction exposed anywhere in this dependency chain, closing off Rodriguez et al.'s literal 2nd-order method without new upstream code. What *is* exposed is `qic.Qic.min_geo_qi_consistency`'s own `order` arg, which is a different knob — how many derivatives of the local QI-consistency residual are also driven to zero, not an expansion-order parameter. Patched `near_axis_configuration.generate`'s hardcoded `order=1` to `order=2` (source-string patch + exec, see `scripts/nae_order2_consistency_sweep.py`) and reran a 45-config sweep (nfp∈{2,3,4} × aspect_ratio∈{6,8,10} × mirror_ratio_target∈{0.10,0.12,0.15,0.18,0.20}, same real-VMEC-validation path as the original `nae_pilot.py`). Result: 34/45 converged, 0 satisfy all 5 P3 constraints (same as order=1). Best mirror_ratio **0.2452** (nfp=3, ar=10, target=0.10) — a new project-best, edging past order=1's 0.2578 — but at the cost of **iota collapsing to 0.042** (vs order=1's comparable-seed 0.41, an order of magnitude worse on the C1 constraint) and no improvement to flux_compression. Read honestly: this reads as the optimizer settling into a different, not-better, local optimum of the same weighted objective (elongation/iota barrier terms unchanged, QI-residual term reweighted) rather than a genuine physics improvement — the marginal mirror_ratio gain is not worth the much bigger iota hole it would leave for ALM to climb out of. **Not turned into a new anchor/lane.** Open follow-up if revisited: rerun with the iota barrier weight increased specifically for the order=2 objective.

  **(B) Domain-adaptive latent generative probe**, following arXiv:2608.16938's explanation for why our first generative attempt (`nae_generative_wall_probe.py`, 0/480 qi-feasible, 29/480 even converged) likely failed: raw 90-dim Fourier-coefficient conditioning doesn't capture the nonlinear geometric constraints that make boundaries valid, and a single model trained once on the general population never gets pushed toward the wall specifically. Two-stage retry: (1) `scripts/train_vae_physical.py` — a VAE over the same 90 coefficients, but with an added *physical-space* (R/Z collocation-point) reconstruction loss, computed exactly (not approximated) since R/Z are linear in the coefficients for any fixed (theta,phi) grid at a given nfp — 5 fixed basis matrices, one per nfp value 1-5. Trained cleanly (150 epochs, `recon 4.58 kl 91.3 phys 0.00068`). (2) `scripts/train_latent_diffusion.py` — a DDPM operating in this VAE's 32-dim latent space rather than raw coefficient space, pretrained on the full 521,761-row merged pool (150 epochs, `loss 0.147`), then fine-tuned on a 500-row "HL set" (the lowest-worst-constraint-violation real rows pooled across all 5 active P3 lanes, `output/hl_nearwall_{X,Y}.npy`) with 30% replay from the general pool (150 epochs, `loss 0.258`), per the paper's own recipe. Final real-oracle validation (`scripts/latent_diffusion_wall_probe.py`, same beyond-wall targets/nfps as the original 0/480 baseline for direct comparability): **0/160 converged — worse than the baseline, not better** (baseline at least got 29/480 through VMEC++). Root-caused directly, not just reported as a bare number: sampled the fine-tuned model's own latents and decoded them — standardized latent samples reach `|z|`≈2031 (should be O(1-5) for a healthy DDPM posterior) and decode to raw boundary coefficients as large as 28.5, versus a real coefficient range of roughly ±1 (`coeff_mean`/`coeff_std` from the VAE checkpoint). The fine-tune (500 rows, lr=3e-4, 150 epochs) diverged the denoiser's reverse process when conditioned on the far-out-of-fine-tune-distribution beyond-wall target — VMEC++ can't even parse the resulting boundaries as valid geometry, let alone evaluate them, which is exactly why convergence went to zero rather than staying at some infeasible-but-valid rate. **Concluded negative: the domain-adaptive recipe as implemented here failed harder than the naive baseline it was meant to fix.** If revisited: lower the fine-tune LR substantially and/or clip sampled latents to the training-set range before decoding, and check whether the 500-row HL set is simply too small/too far from the beyond-wall target for this size of model to fine-tune stably.

- **Side-lane finding worth flagging, not a breakthrough**: `grounded_walk_p3_v4c_correlation_gpu1` (the earlier "does mirror_ratio trade against qi from a qi-passing anchor" probe, `grounded_walk_step_v4.py`, deliberately enforcing ONLY the qi constraint — see the script's own docstring) hit its own success criterion at round 15 (`qi=0.000235`, feasible) and halted itself. This is **not** a P3-feasible point — `edge_magnetic_mirror_ratio=0.672` and `vacuum_well=-0.036` both badly fail their real thresholds, and only 1/221 real rows in this lane's pool are even qi-feasible. Read correctly, this is the answer the docstring said to watch for: mirror_ratio got *worse* (0.672) than the qi-passing anchors it started from (0.48-0.56), i.e. pushing away from the anchor along directions that keep qi feasible pushes mirror_ratio the wrong way — reinforcing the hard Pareto wall reading rather than escaping it. Not re-armed with the other 3 constraints per the docstring's own stated follow-up condition, since that condition ("materially better mirror_ratio AND qi feasible") wasn't met.

- **Status check at round 65→911-2013 across the 5 still-active P3 lanes (2 weeks later)**: wall still holds, still 0 P3-feasible anywhere. `v4h` qi crept to a new best **0.000353** (from 0.00035, essentially flat) and `v4g` mirror_ratio held at its prior best **0.2634** — both still just outside their respective thresholds (0.0003162, 0.25). Per-lane qi-vs-mirror_ratio correlation over each lane's last 400 real rows (v4d -0.67, v4e -0.29, v4f -0.31, v4g -0.79, v4h -0.38) is confounded across lanes by differing distance-to-wall and range restriction — a corrected read, pooling all 5 lanes' real data (7,432 rows) and binning by `worst_violation` (distance to the P3 boundary) instead, shows the tradeoff is actually **strongest right at the wall** (worst∈[1,2): r=-0.575; [2,3): r=-0.478) and weakens/gets noisy further out ([3,5): r=+0.129; [5,10): r=-0.299; [10,20): r=-0.164; [20,50): r=-0.041; 50+: r=+0.193) — v4e's weak per-lane -0.29 is an artifact of its window straddling several of these bins (including the sign-flipped one), not evidence of a softer tradeoff near a solution. No lane has broken the tradeoff; if anything this sharpens the case that it's real and gets more binding as you approach P3-feasibility, not less. Both literature-driven experiments (A and B above) are now concluded negative; no new lane has been spun up from either.

- **Direct user question: is P3 even solvable, or could this be a hard wall the benchmark itself never actually closed?** Checked, not assumed. Fetched the actual paper (arXiv:2506.19583, NeurIPS 2025) and the actual `constellaration` package (already installed in `oracle-x86`): its own reference baseline, **"ALM-NGOpt"** (Augmented Lagrangian wrapped around Nevergrad's gradient-free `NGOpt` meta-optimizer, evaluated against real VMEC++ every iteration — no learned surrogate anywhere), reports Table 5 solutions for **4/4 attempted P3 instances** (aspect_ratio fixed at 6/8/10/12, one run each), normalized constraint violation as low as **0.0**. Confirmed `constellaration.problems.MHDStableQIStellarator` uses the exact same 5 constraints/thresholds this project targets. **So P3 is not impossible — it's been solved, just not yet by anything this project has tried.** The structural difference from every approach here so far: their method never routes a gradient through a frozen learned proxy. Every negative result logged above (staged-vs-free-push blind-spot exploitation, both wall-probe generative failures) has the same shape — a surrogate/generative model confidently wrong exactly where real data is thinnest, right at the boundary.

- **`scripts/ngopt_alm_p3.py`, launched same day**: direct reproduction of ALM-NGOpt — outer PHR augmented-Lagrangian loop (reusing `optimize.py`'s own `paper_feasibility_violation` for exact qi-log-transform/vacuum_well-divisor parity with the official benchmark code), inner loop a fresh `nevergrad.optimizers.registry["NGOpt"]` per outer iteration warm-started at the best real point so far, every candidate scored by a REAL low-fidelity VMEC++ call through the same `oracle.worker_fn`/`run_batch_with_timeout` harness every other script here uses (28 parallel workers). Seeded from the single best real point pooled across all lanes (nfp=2, aspect_ratio≈8.2, qi=0.000677, mr=0.534, worst_violation=1.14 — both qi and mirror_ratio tied at the wall). Both `nevergrad` and `desc` (a fast JAX-differentiable equilibrium solver, a real alternative worth trying later) turned out to already be installed in this project's own container, unused until now.
  - **Two real bugs caught and fixed before launch, not glossed over**: (1) `ng.p.Array` without explicit bounds silently ignores the per-dimension mutation sigma and falls back to a quasi-random opening batch — confirmed directly (0/8 smoke-test candidates converged, all VMEC "poorly-shaped boundary" failures); fixed with explicit `set_bounds`. (2) A direct sigma scan on the actual seed point found this near-wall real point sits in a **very narrow VMEC convergence basin** — sigma=0.005 broke all 8 test candidates, 0.002 broke 7/8, 0.001 converged 7/8, 0.0005 converged 8/8. A real, useful finding on its own: the near-wall points this project has found are themselves close to marginal/degenerate equilibria, not just close to the P3 constraint boundary — which may partly explain why gradient-based search near the wall has been rough (the true loss surface near real near-wall points may be far less smooth than a frozen surrogate believes). Set `--sigma0 0.0008` accordingly.
  - Launched: `docker compose run --rm train scripts/ngopt_alm_p3.py --tag ngopt_v1 --n-outer 200 --budget-per-outer 56 --n-workers 28`. First 7 real outer iterations: convergence rate through real VMEC++ held at 70-100% (39-56 of 56 per iteration), best_worst_violation dropped **1.14 → 0.956** — already past every one of the 5 retired lanes' final values, in 7 iterations rather than weeks. Too early to call a trend, but the direction and convergence-rate stability are both real signals, not a fluke first step. Numbers to follow.

- **`ngopt_v1` ran to completion, genuine stall, not success**: all 200 outer iterations, `best_worst_violation` flattened at **0.0917** for the last 33 iterations while `rho` exploded 16B→1.4T — the classic ALM signature of a local point the current search neighborhood can't escape, not a budget problem. No `ngopt_v1_success.json`, full P3 feasibility not reached. Root cause of the flat run found while trying to act on it further: `results.jsonl` never logged the candidate coefficient vector (`params`), only measured physics targets — so v1's 11,200 real-VMEC-labeled rows, including its exact stall point, can never be resumed from or reused downstream. A one-time, unrecoverable loss for v1 specifically; fixed going forward (see below).
- **Fixes added to `ngopt_alm_p3.py` and relaunched as `ngopt_v2`**: (1) per-row `params`/`nfp` now logged (`cand.value` with `oracle.ZERO_INDICES` zeroed) so future runs' full real data is reusable; (2) `<tag>_checkpoint.json` written every outer iteration (`x_best`, `nfp`, `lam`, `rho`, `best_worst`, `bound_mult`, `stall_count`), loadable via `--resume`; (3) adaptive stall-detection — track outer iterations since last improvement, and once `--stall-patience` (12) is exceeded, widen the search box (`bound_mult`, a multiple of `sigma0`) by `--stall-growth` (1.6x), capped at `--max-bound-mult` (60.0). Launched: `--tag ngopt_v2 --n-outer 250 --budget-per-outer 56 --n-workers 28 --timeout-seconds 45`.
- **`ngopt_v2` also stalled, at a worse value than v1 (0.0965 vs 0.0917)** — a real negative finding, not glossed over. The widening mechanism fired exactly as designed (`bound_mult` 10.0→16.0→25.6→41.0→60.0 across outer iterations 145/157/169/197), but `best_worst_violation` stayed completely flat at 0.0965 for the remaining ~50+ iterations even at 6x the original box width. **This rules out "search box too tight" as the actual cause of the stall** — points to a genuine local optimum in NGOpt's search dynamics for this seed, not something escapable by more local room alone.
- **`scripts/ngopt_to_pool.py` (new)**: converts an `ngopt_alm_p3.py` run's real-VMEC-labeled `results.jsonl` (post-fix, with `params`) into the project's standard 92-col `{X,Y}` pool schema, writing to a separate `output/bootstrap_seed_<tag>/` directory rather than merging into the master pool automatically (deliberate — broad, hard-to-reverse action on a pool many things depend on). Run on `ngopt_v2`: **13,640 converged (real "well" anchors), 360 rejected, from 14,000 total rows**; qi range 0.00063-0.0052, mirror_ratio range 0.264-0.543.
- **Cross-pollination test, per direct user request ("can we plug these into the bootstrapper... does this give us a different signal we can use constructively with our base approach")**: added `--well-extra-x` to `bootstrap_loop.py`, forwarding an external real-data X.npy as an additional `--extra-x` to every generation's `train_vae.py` retrain (on top of the lineage's own accumulating pool), following the same pattern `train_vae.py --extra-x` already used elsewhere. Launched a bounded 4-generation trial: `--tag ngopt_well_probe --well-extra-x output/bootstrap_seed_ngopt_v2_well/X.npy --generations 4 --seconds-per-generation 1800`. Gen 1 confirmed the well data folded in cleanly: 158,685 real + 13,640 extra = 172,325 rows into the VAE retrain.
- **New structural hypothesis for v1/v2's stall, found by re-reading `ngopt_alm_p3.py` closely per direct user request ("test whatever you feel we need to reliably get to valid P3 candidates")**: the script recreates a FRESH NGOpt optimizer every outer ALM iteration (budget=56, then discarded, recentered on a small box around x_best) with `mutable_sigma=False`. That means NGOpt's sub-optimizer never carries any internal model (CMA covariance/step-size adaptation, momentum) across the run — every outer step is architecturally closer to a small random-perturbation batch around x_best than genuine gradient-free search. This is a materially different design from what "ALM-NGOpt" means in the paper (one persistent optimizer, only the Lagrangian objective it's told changes between batches). Built `scripts/ngopt_alm_p3_persist.py`: same ALM lambda/rho update logic (unchanged, one-variable test), but ONE optimizer created for the whole run, `mutable_sigma=True` (nevergrad requires scalar sigma once mutable, so per-dim sigma freezing was dropped — ZERO_INDICES stay pinned via bounds regardless), fixed bounds for the whole run (sigma0*60, the same cap v2's widening eventually hit) instead of recentered/regrown per outer step. Smoke-tested clean (8/8 converged per batch, no crashes). Launched real comparative run at the same scale as v2: `--tag persist_v1 --n-outer 250 --budget-per-outer 56 --n-workers 28 --timeout-seconds 45`.
- **Result: hypothesis FAILED, decisively, in the opposite direction expected.** Head-to-head at identical outer/rho checkpoints (real VMEC++, 56/56 converged each iteration for both): outer 0 v2=1.109 vs persist=1.089 (comparable, as expected from the shared seed); outer 10 v2=0.867 vs persist=1.035; outer 20 v2=0.647 vs persist=1.031; outer 30 v2=**0.415** vs persist=**1.031 (already flat since outer 20)**. v2 improved 2.5x more over the same 30 iterations and the persistent-optimizer run had already stalled by outer 20 — killed early (`docker stop`) rather than run out the remaining 220 iterations proving the same point. **Root-caused, not just reported as a bare failure**: declaring the FULL 14,000-eval budget up front changes which sub-algorithm nevergrad's `NGOpt` registry actually selects, calibrated for long-horizon pacing against a *stationary* objective — but the ALM loss is deliberately non-stationary, changing every 56 evaluations as lambda/rho update. A long-horizon optimizer paced for 14k static evals is badly miscalibrated when the objective quietly shifts under it every outer step. The original recreate-per-outer design's real virtue, previously uncredited, is that declaring budget=56 to NGOpt every time keeps its algorithm selection/pacing matched to the actual eval-count-per-stationary-objective, at the cost of discarding state. **Refined next hypothesis, tested and ALSO FAILED the same way**: built `scripts/ngopt_alm_p3_persist_cma.py`, identical to `_persist.py` except using `CMA` directly instead of `NGOpt`'s meta-selector — CMA's population size/pacing is governed by dimension (~4+3ln(90)≈17), not declared total budget, so it should sidestep the budget-miscalibration theory above if that theory is what mattered. Result: **also stalled early**, outer 10=1.058, outer 20=1.053 (flat), essentially the same value persist_v1 stalled at (1.03-1.05) and still far behind v2 (0.647 by outer 20, 0.415 by outer 30). Killed early again. Two different sub-algorithms failing at nearly the same value rules out "which algorithm nevergrad picks" as the real variable. **Better-supported reading, found by re-examining what's actually different**: the recreate-per-outer design's real virtue isn't the small budget declaration, it's that it hard-resets the sampling center to the EXACT incumbent best point every outer iteration. Any optimizer with persistent internal state (CMA or NGOpt) updates its mean via weighted recombination over several candidates, not a hard snap to the single best — a small drift off the literal best point. Given the independently-confirmed finding that this near-wall region is an extremely narrow VMEC convergence basin (sigma=0.002 already breaks most candidates, `ngopt_alm_p3.py`'s own sigma-scan), even that small drift is enough to walk the sampling center out of a basin most nearby points don't converge in at all — which is consistent with both persistent variants stalling at almost the same value rather than at genuinely different local optima. Not committing to this as final without a direct test (queued: compare real-VMEC convergence rate of candidates sampled near a persistent optimizer's drifted mean vs. candidates sampled with a hard reset at the identical incumbent best, holding sigma fixed).
- **Pivoted to the actually-unexplored lever instead of continuing to poke at optimizer internals**: v1 and v2 both only ever searched from the SAME single seed point (`--seed-row 0` default, out of 500 real near-wall rows in `hl_nearwall_X/Y.npy`). Checked directly whether genuinely diverse seeds exist rather than assuming: the first 20 rows (sorted by worst-violation) are near-duplicates of row 0 (same lineage), but real diversity exists further in — row 126 (nfp=2, qi=0.00097, mr=0.584, L2 dist 0.245 from row 0) and row 14 (nfp=4, qualitatively different field-period count, qi=0.00071, mr=0.519) are both genuinely distinct basins. Using the best-performing architecture so far (`ngopt_alm_p3.py`'s recreate-per-outer/hard-reset design — now better understood as well-matched to this domain's narrow VMEC convergence basin, not a bug), launched two parallel lanes from these diverse seeds (`ngopt_v3_seed126`, `ngopt_v3_seed14nfp4`, 150 outer x 40 budget, 14 workers each to share the 32-core host with both lanes concurrently) to test whether v1/v2's ~0.09-0.10 stall is basin-specific (seed 0's own local optimum) or reproduces regardless of starting point — the latter would be a 5th independent confirmation of the structural qi-vs-mirror_ratio wall this project keeps finding by every method (NAE seeding, generative probes, gradient search, now direct NGOpt from multiple starts).
- **Result: both lanes stalled too, at DIFFERENT floor values, neither reaching P3 feasibility.** `seed126` (nfp=2): fast drop to ~0.30 by outer 60, exactly flat by outer 100 (0.2576), still flat through outer 150 (**final 0.2562**) despite stall-widening firing 4 times (bound_mult 10→41, rho climbing to 1.27B) — same dead-flat-despite-widening signature as v1/v2. `seed14nfp4` (nfp=4): also stalled, but at a materially better floor (**final 0.1522**, flat from outer ~120 on), better than `seed126`'s 0.256 but still far short of v1/v2's own 0.0917/0.0965 floor (though NOT a clean comparison — these lanes used budget-per-outer=40/n-outer=150 = 6,000 total evals to share the 32-core host with two concurrent lanes, vs. v1/v2's 56×200-250 = 11,200-14,000 evals; the worse floor here is at least partly a smaller-budget artifact, not necessarily a harder basin). **What IS clean regardless of the budget confound: every real starting point tried (3 so far — seed 0/nfp=2, seed 126/nfp=2, seed 14/nfp=4) stalls well short of full P3 feasibility, and rho reaching into the hundreds of millions to billions with zero movement is consistent across all of them.** This rules out "wrong starting basin" as the reason v1/v2 didn't reach feasibility — it reproduces from a qualitatively different basin (different nfp) too. Combined with the optimizer-architecture tests above (persistent-state NGOpt and persistent CMA both also failed, differently), the accumulated evidence now points away from "fix the search mechanics" and back toward the qi-vs-mirror_ratio Pareto tradeoff itself (independently measured elsewhere in this log at r≈-0.5 to -0.6 right at the wall) as the actual binding constraint — no single-point local search, regardless of optimizer or starting basin, has escaped it. **Open, not yet tried**: an explicit multi-objective/Pareto approach that treats qi and mirror_ratio as a frontier to trade along rather than two independent hard constraints attacked from one point; or reusing the grounded-walk project history's own finding that NAE-constructed seeds at specific (nfp, aspect_ratio, mirror_ratio_target) combinations reach mirror_ratio much closer to 0.25 than any gradient-search basin has (0.245-0.258 there vs 0.48-0.69 here) — those seeds were never fed into ngopt_alm_p3 specifically, only into the older grounded_walk_step_v4 driver.

**Major pivot: found the officially-shipped omnigenity-based construction recipe in the `constellaration` pip package itself, read directly from source rather than assumed.** `constellaration.data_generation.vmec_optimization.optimize_boundary_omnigenity_vmec` + `constellaration.omnigeneity.omnigenity_field_sampling.SampleOmnigenousFieldAndTargetsSettings` is very likely how the actual training dataset (and plausibly the paper's own baselines) were generated — and it is a FUNDAMENTALLY different search space from every single approach in this project so far. Every prior method (gradient search from a real point, generative-model probes, NAE first-order construction, direct NGOpt-against-real-VMEC from many seeds/basins/optimizer variants) searches or constructs directly in the raw 90-dim boundary Fourier coefficient space and hopes QI-consistency emerges from the numbers. This recipe instead: (1) samples a physically-structured OMNIGENOUS FIELD with `mirror_ratio`, `edge_rotational_transform_over_n_field_periods`, `aspect_ratio`, `n_field_periods`, and `max_elongation` as literal, directly-settable input targets (not properties to search for); (2) fits a REAL boundary to that field via NGOpt + real VMEC++ (the exact same nevergrad/NGOpt machinery this project's own `ngopt_alm_p3.py` reproduces, but scalarized log-sum-of-squared-residuals rather than ALM, and fitting a physically-meaningful field target rather than the raw 5 P3 metrics directly) — driving the true QI residual toward zero by CONSTRUCTION (the omnigenity residual is a close cousin of the actual `qi` benchmark metric, both from Goodman et al. 2023), not by search luck in an unstructured space. Separately, confirmed the real official `augmented_lagrangian.py` module also differs from `ngopt_alm_p3.py`'s hand-rolled ALM in two ways worth remembering if revisited: (1) `rho` only grows (2x) when progress on the worst violation is insufficient (`constraint_violation_tolerance_reduction_factor=0.5`), not unconditionally every outer step like our `rho *= 1.15` — ours needlessly explodes rho (into the billions) even while genuinely still improving, over-penalizing long before real stalling; (2) trust-region bounds SHRINK over the run (`bounds_reduction_factor=0.95`, floored at `bounds_min=0.05`), the opposite of our stall-triggered widening — not yet reimplemented, noted for a future revision of `ngopt_alm_p3.py` if the omnigenity route doesn't pan out on its own.
- **Tried it.** Confirmed the two best NAE anchors from the grounded-walk era are still on disk in the right 92-col format (`output/nae_seed2_nfp2_ar8_anchor.npy`, mr=0.258/qi=0.023/vw=-0.074/fc=1.125 — mirror_ratio near-feasible, everything else badly off, the OPPOSITE profile from every gradient-search seed tried in ngopt_alm_p3 so far; `output/nae_seed2_nfp4_ar10_anchor.npy`, mr=0.341, nfp=4, already satisfies flux_compression on its own). Smoke-tested the nfp2ar8 seed's local VMEC basin width directly rather than assuming the gradient-search-tuned `--sigma0 0.0008` would transfer: 12/12 converged, starting `best_worst_violation=0.735` (already better than every gradient-search seed's ~1.1 start, consistent with mirror_ratio already being close). Launched both as full runs, this specific combination (real-VMEC-direct NGOpt, not the older frozen-surrogate grounded_walk driver) never tried before: `ngopt_nae_nfp2ar8` and `ngopt_nae_nfp4ar10`, 200×56, 16 workers each (32 total, matches host core count). **Result: stalled too, same shape as every other direction.** `nfp2ar8` (mirror-ratio-near-feasible start): fast early drop, faster than any gradient-search seed's early pace (0.735→0.406 by outer 30, still ahead of v2's 0.415), but slowed and flattened by outer ~100-110 at **0.247**, essentially the same floor `seed126` (a totally different, qi-near-feasible starting seed) landed at (0.256) — a 6th independent basin converging to nearly the same wall value regardless of which direction it's approached from. `nfp4ar10`: slower throughout, convergence rate itself degraded (56/56 → 17/56 by outer 60) as the search pushed into less VMEC-stable territory, flattening around 0.53 by outer 70 (worse than every other lane, likely reflecting a genuinely harder nfp=4/aspect_ratio=10 basin). Both killed early (outer 120/70) once the flat pattern was unambiguous — full 6-way summary of every direction/seed/optimizer tried this session: **v1/v2 (seed 0, qi-near-feasible) → 0.09-0.10; seed126 (alt qi-near-feasible) → 0.256; seed14nfp4 (nfp=4, qi-near-feasible) → 0.152; nfp2ar8 (mirror-near-feasible) → 0.247; nfp4ar10 (mirror-near-feasible, nfp=4) → 0.53; persistent-optimizer/CMA variants (seed 0) → 1.03-1.05 (worse, different mechanism problem).** No single-point local search from any tested basin, direction, or optimizer architecture reaches full P3 feasibility — strong, repeated evidence the blocker is the underlying multi-constraint tradeoff itself, not search mechanics or starting-point luck.
- **Then actually ran the official omnigenity recipe found above, `scripts/omnigenity_seed.py`**: sampled 3 field-shape variants (fixed `mirror_ratio=0.18`, `edge_rotational_transform_over_n_field_periods=0.28`, `aspect_ratio=8`, `max_elongation=5`, `n_field_periods=2`, mpol=4/ntor=4 to match this project's own 90-dim resolution, `budget_per_design_variable=100`, the package's own recommended default) and fit each via `optimize_boundary_omnigenity_vmec`. **Result: real, honest, mixed.** All 3 samples land on exactly **2/5 constraints**: `edge_rotational_transform_over_n_field_periods`=0.280 and `edge_magnetic_mirror_ratio`=0.243-0.247 both pass — confirming these ARE directly controllable through this recipe, as designed. But `qi` (0.010-0.015) is NOT the hoped-for breakthrough — comparable to, not better than, the NAE first-order construction's own qi (0.0095-0.03) despite the omnigenity residual being fit directly; `flux_compression` (1.08-1.23) and `vacuum_well` (-0.03 to -0.06) both fail — the SAME two constraints every other real near-wall point in this whole project has struggled with, since neither is targeted by this fit's objective at all. **Bug found and fixed along the way, not glossed over**: first validation pass reported 0/3 even converged, looking like invalid geometry — root-caused directly rather than assumed: the boundaries were completely valid (re-checked in a fresh process, real sane physics). The actual bug: `optimize_boundary_omnigenity_vmec`'s internal `ProcessPoolExecutor(mp_context=forkserver)` silently corrupts the parent process's multiprocessing state, breaking any subsequent `run_batch_with_timeout` (fork-based) call made in the SAME process. Fixed in `omnigenity_seed.py` by shelling out to a fresh interpreter for the validation step. **Net: this is the best 3-way starting point found in the whole project** — 2/5 hard-passing, plus meaningfully better qi/fc/vw than the comparable mirror-near-feasible NAE seed (`nfp2ar8`, which had mr=0.258/qi=0.023/vw=-0.074/fc=1.125). Smoke-tested at `best_worst_violation=0.4245`, dramatically better than every other seed's start (0.6-1.1 range). Saved as `output/omnigenity_seed2_anchor.npy`, launched into the proven-best `ngopt_alm_p3.py` recipe for final refinement: `ngopt_omnig_v1`, 200×56, 28 workers.
- **Result: stalled too, at nearly the same floor as the plain NAE seed (0.28 vs 0.247), despite starting from the best combined seed found anywhere in this project.** Fast early drop (0.4245→0.322→0.292 by outer 20), then flattened hard by outer 30-40 at **0.2805**, dead flat outer 40→50. Killed there. **This is the 7th independent confirmation** (v1/v2, seed126, seed14nfp4, nfp2ar8, nfp4ar10, persistent-optimizer/CMA variants, and now the best-available omnigenity-fitted seed) that single-point local search — regardless of starting basin, direction of approach, optimizer architecture, or how good the starting point already is on 2 of the 5 constraints — cannot cross the remaining distance to full P3 feasibility from real, near-wall points in this coefficient space.

**Consolidated conclusion for this whole testing pass, stated plainly**: this session tested every reasonable lever for reliably reaching P3-feasible candidates via direct real-VMEC local search — optimizer internals (persistent NGOpt, persistent CMA, both failed worse than the original recreate-per-outer design), starting-point diversity (5 different real/constructed seeds spanning nfp=2/4 and both "qi-near/mirror-far" and "mirror-near/qi-far" starting profiles), and the actual official omnigenity-based construction recipe (which nails 2/5 constraints by design but still can't close the gap on the remaining 3). All seven attempts land in the same qualitative place: fast initial progress, then a hard wall somewhere between worst_violation 0.10 and 0.28 (or worse), never reaching 0. This is now strong, repeated, method-independent evidence that the real blocker is a genuine multi-constraint tradeoff in the P3 problem's feasible-point geometry (echoing the qi-vs-mirror_ratio r≈-0.5 to -0.6 tradeoff measured directly elsewhere in this log), not a fixable gap in search technique. **Not yet tried, and the most promising remaining direction**: reusing the official `augmented_lagrangian.py`'s own recipe (conditional rho growth, shrinking trust-region bounds) rather than `ngopt_alm_p3.py`'s hand-rolled version, PLUS treating this as a genuine multi-objective/Pareto search (explicitly trading constraints against each other along a frontier) rather than a single-point descent — both noted above but not yet implemented.

- **`ngopt_well_probe` result, all 4 generations complete, 4,229 accepted real-VMEC-validated candidates: 0/4,229 satisfy all 5 P3 constraints.** Only one candidate reaches 4/5 (at the cost of qi=0.097, nowhere close on that axis). Real signal worth keeping: best qi reached, **0.00042177**, is lower than anything in NGopt's own converged "well" data (0.00063 floor) — this VAE-based sampler, seeded with NGopt's real near-wall points, reached a lower qi than NGopt's own local search ever did. But it isn't free: every one of the low-qi candidates has badly blown mirror_ratio (0.60-0.86 vs ≤0.25 required) — the same qi-vs-mirror_ratio tradeoff documented throughout this log (binned r≈-0.5 to -0.6 right at the wall) reasserts itself here too. **Conclusion: the different exploration mechanism is real (reaches places NGopt's local search didn't), but hasn't been enough to clear the wall — same fundamental multi-constraint tension is the blocker, not a lack of near-wall samples.** Not merged into the master `output/X.npy` pool — no strong case for merging given 0 full-feasible and the low-qi points being badly infeasible elsewhere.

- **Compute reallocation, per direct user request ("feel free to turn some other stuff off if we're flatlining")**: checked recent-progress (not lifetime-progress) on each of the 5 active P3 lanes directly — running-best worst_violation over the last 150 real rows specifically, not the whole run. **4 of 5 (v4d, v4e, v4f, v4g) showed exactly zero improvement in their most recent 150 rows**; only `v4h` was still moving (0.618 recent gain). Retired the 4 flat lanes (`${TAG}_FAILED` marker files, the queue runner's own intended stop mechanism — confirmed live: `vega_queue_runner.sh`'s cron immediately relaunched all 4 on its next 10-minute tick before the marker was in place, caught and killed the relaunched drivers + their now-orphaned scorer-retrain containers rather than leaving them to run to no purpose). `v4h` and the new `ngopt_v1` lane are the only two P3 lanes running now. Nothing destroyed — every retired lane's full real pool stays on disk and can be resumed if `ngopt_v1` doesn't pan out either.

---

## Airfoils (XFOIL) — full parity with stellarators

**Task**: 16-dim CST (Class-Shape Transformation) coefficients → cl/cd/cm/l_over_d, conditioned on (Reynolds, alpha).

- XFOIL wraps cleanly (GPL-3, needed a `--recurse-submodules` git clone fix for the PyPI sdist's missing Fortran source). Genuinely faster and more forgiving than VMEC++ (85-92% validity vs. VMEC++'s 55-73%).
- **A real physical-sanity gap, caught before training on it**: a small fraction of "converged" XFOIL runs return spuriously near-zero `cd` (down to 1e-12), producing `l_over_d` outliers up to 9.8e10. Fixed with a `cd < 1e-6` rejection at the oracle level — same "convergence flag alone doesn't guarantee physical sanity at the tails" lesson VMEC++ already taught, now confirmed in a second domain.
- GAN > Diffusion > cVAE ranking replicates cleanly. `cd`/`l_over_d` are consistently the weakest-steered targets across every architecture, and are also the hardest to forward-predict (90-97% error reduction over baseline for cl/cm/l_over_d, only 13% for `cd`) — two independent methods landing on the same "hard" target, real signal.
- **Cluster-split generalization does NOT degrade the way VMEC++'s does** — every target improves under a cluster split, the opposite direction. Plausibly because this domain's dataset is bootstrapped from only ~5 archetypal seed shapes, so held-out regions aren't meaningfully harder than held-out rows. Confirmed on a second seed, not a fluke.
- Architecture search (parallel to VMEC++'s own): `half_siren` trunk + Adam beats a plain MLP baseline, confirmed 3-for-3 seeds on `cd`; SOAP loses on every trunk in this domain (opposite of VMEC++'s own SOAP win) — a genuine cross-domain difference, not assumed to transfer.
- NeuralFoil (a fitted XFOIL emulator) benchmarked and explicitly NOT adopted as a stand-in oracle — reliable at rejecting genuine garbage (~120x faster than real XFOIL) but not reliable enough in the moderate-perturbation regime to trust for real scoring.
- Real reference points added: NACA 4-digit airfoils fit to CST coefficients via closed-form least squares (RMS fit error ~0.015% chord after fixing a divide-by-zero-at-the-tips bug), then run through the real XFOIL oracle — confirms textbook-correct behavior (NACA0012 exactly antisymmetric, cambered sections lift at alpha=0 with the correct-sign pitching moment).

---

## TORAX (tokamak transport) — full generative pipeline, one real unresolved diffusion failure mode

**Task**: 10 continuous scalars (plasma current, density, heat diffusivities, heating power, 4 geometry params) → Q_fusion/tau_E/H98/T_e_volume_avg.

- Apache 2.0, clean pip install, but **hard-requires Python≥3.12/numpy≥2**, incompatible with this project's other numpy<2-pinned images — needed its own standalone Docker image.
- **Real, load-bearing architectural finding**: TORAX's cost is dominated by a one-time ~3-5s JIT-compile tax per process, not the physics itself (post-compile stepping is ~2-3ms). This project's subprocess-per-candidate harness (`oracle_harness.py`) would pay that tax on every single candidate — confirmed empirically (4.92s cold vs. 0.13-0.45s warm even under real parameter variation). Needed a genuinely new persistent-worker harness (`oracle_harness_persistent.py`, workers stay alive across many candidates) rather than a drop-in.
- Also has a real hang failure mode on degenerate input (contradicting an earlier "no known hang" assumption) — confirmed by deliberately testing a near-zero-minor-radius candidate, which ran past a 30s timeout rather than raising `SimError` cleanly. Per-candidate timeout+respawn in the persistent harness is required, not defensive extra.
- Bulk dataset (15,275 rows, 93% hit rate) has a real, flagged-not-fixed extreme-value tail (`Q_fusion` up to 19,130, physically absurd for ITER-scale devices) — distinguishing "genuinely optimistic real config" from "numerical artifact" needs physics judgment this project doesn't have a ready answer for. Confirmed via real reference devices (ITER/SPARC/JET at zero noise, all landing in a sane Q_fusion 0.19-3.16 range) that the tail really is noise-driven, not reachable physics.
- **A real, deep, mechanistically-traced diffusion failure mode, unique to this domain**: diffusion has perfect validity and the strongest directional steering signal of the three architectures, but by far the worst steer-candidate convergence (~48% vs. cVAE/GAN's 85-97%). Traced to real cause via a diagnostic that kept raw failing params: diffusion's failures are categorically different from cVAE/GAN's — physically nonsensical geometry (median R_major 170m vs. real devices' <6.2m, negative elongation, Ip up to 27x ITER's real value), not mild misses. Plausible explanation: a GAN's adversarial critic provides a "does this look real" pressure diffusion's denoising objective lacks — diffusion commits fully to a requested direction (hence its precision when it works) with no guardrail against overshooting into impossible territory. Not confirmed as THE mechanism (would need a controlled ablation), and not shown to reproduce in VMEC++/airfoil.
- Full A/B/C/D bar closed; 3D nested-flux-surface viewer built.

---

## Mug v1 (superseded — self-written FD conduction, v1 through v4)

Chosen specifically as a self-written (no license risk), cheap, intuitive 5th domain — the opposite motivation from TORAX/jax-cfd's GPU-native pick. Went through 4 real physics revisions (v1: wall+insulation+handle scalars → v2: named materials + wall profile + real transient handle fin physics → v3: variable body shape (frustum) + a lid/top-loss model, closing a real correctness gap where v1/v2 lost zero heat through the open top → v4+: axial/azimuthal/radial coupled diffusion, real radiative loss). Each revision found a genuinely new, real design axis (thermal performance, structural cost/weight, handle safety, and eventually body-shape cost) that stayed nearly separable from the others — validated via correlation analysis at 20K-row dataset scale each time, not assumed.

Full A/B/C/D bar closed at v4 scale (GAN > Diffusion > cVAE ranking replicated a 4th time here), plus a full interactive 3D viewer with live temperature-over-time animation.

**Superseded entirely** starting with a direct course reversal (self-written FD physics ruled out in favor of OpenFOAM, a real broadly-accepted solver) — see Mug v2 below, which is the actively developed version of this domain now. v1's code and dataset remain on disk for provenance/regression-check purposes, not deleted, but are not where new mug work happens.

---

## Mug v2 (current, active) — OpenFOAM-backed, free-form-deformed body + curved handle + coating + joint

**The core pivot from v1**: replaced the self-written lumped/1D thermal model with real OpenFOAM (`chtMultiRegionFoam`, solid conduction only) and replaced the revolve-based body (r_base/r_mid/r_rim profile) with a genuine **3D free-form-deformation (FFD) shell** — two independent FFD lattices (inner/outer surfaces), each `(2,2,3)` control points, meshed via `snappyHexMesh`. This was a deliberate architectural rejection of any body-of-revolution parameterization: "something *can be* radially symmetric, but there's a real lack of anything interesting happening when we enforce that." A body of revolution can never be genuinely novel/asymmetric no matter how its profile is parameterized; FFD can (including genuine self-intersection as a real infeasibility mode, not guaranteed away by construction).

**Current parameterization** (`mug_openfoam_oracle_v2.py`, `PARAM_DIM=88`): r_inner/r_outer/height + 72 FFD displacement components (inner+outer lattices) + struct/coating material + coating thickness + a real curved-arc (quadratic Bezier) handle with 2 attach points (body-relative theta/z-frac/radial-offset), handle material, and a **joint material** (a first-class categorical choice — 6 real joint types from a loose mechanical press-fit, h≈52 W/m²K, up to a welded metal fusion, h≈324,000 W/m²K — replacing an old fixed `H_HANDLE_CONTACT=200` constant). Materials: 21 structural / 27 handle / 10 coating / 6 joint, each real and food-safety-reviewed (9 of an original 20 structural materials removed for lacking real bare-hot-liquid-contact use, e.g. no bare aluminum/copper/brass — real drinkware in those metals is always lined).

**Real feasibility gates, each calibrated against actual behavior, not assumed**: self-intersection (`surfaceCheck`, threshold 20 vs. baseline noise 0-4), the background mesh's own `allBoundary` patch surviving `snappyHexMesh` (which does NOT reliably hard-fail on bad geometry the way `blockMesh` does), attach-point-to-surface distance (10mm tolerance), per-material wall-thickness plausibility (a 43mm calabash-gourd wall is not a thing gourds can be — caught by literally eyeballing accepted output, not derived), and — the deepest one, found by user inspection of the real viewer — **the handle's own Bezier ARC must stay outside the body's local surface radius all along its length, not just at its two endpoints**. The endpoint-only check let handles connecting near-opposite sides of the mug with insufficient bulge cut straight through the interior cavity ("stabbing through the mug"); confirmed 3 of 5 already-"accepted" designs in one small batch were actually invalid under the corrected check. Every dataset generated before that fix was discarded and regeneration restarted from zero.

**A real, cheap two-tier fidelity split** now exists where none did before: `check_feasibility_fast()` (pure numpy/scipy, ~2ms/candidate, zero Docker/OpenFOAM dependency) runs all the cheap gates above; only candidates that clear it get dispatched into the real OpenFOAM worker pool (self-intersection subprocess check, meshing, the actual 3-pass `chtMultiRegionFoam` solve, ~300-400s each). A self-retraining `GradientBoostingClassifier` trained on real worker-stage outcomes (accepted vs. failed-at-mesh/solve) further oversamples toward likely-successful candidates before dispatch — direct answer to "can we use the bootstrap/search approach we've got even on small data," though the real payoff (`bootstrap_generic.py`/a trained conditional generator) still needs a few hundred-to-1000 oracle-scored rows first, which production generation is actively working toward.

**Handle thermal physics**: a real 3-pass coupling (body baseline solve → probe real local temperature at both attach points → handle's own separate mesh/solve driven by that → localized Gaussian-kernel heat sink applied back to the body at both real attach points). The handle's two ends share one averaged internal driving temperature (a real, documented simplification forced by a still-unexplained `snappyHexMesh` reliability quirk where one of two small end-cap regions unpredictably vanishes during meshing — worked around by merging root+tip into one contact patch, not root-caused). Joint material genuinely changes outcomes: same aluminum handle geometry measured **21.5°C (press-fit) vs. 68.0°C (welded)** at 60s touch time, purely from attachment type.

**Current status**: production dataset generation (`generate_mug_dataset_v2.py`, target 3000 rows) restarted clean after the handle-arc-through-body fix; overall hit rate is low (~0.5-1%, most of it now correctly-rejected geometry, not a regression) but the fast-filter tier makes that cheap. A dedicated viewer exists at `/mug-v2` (separate page from the old `/mug`, since the geometry model is structurally incompatible — real client-side FFD reconstruction + `THREE.QuadraticBezierCurve3`/`TubeGeometry` for the handle, no live physics recompute since a real OpenFOAM solve takes minutes). Material-library growth (11→21 structural materials etc., done in a concurrent session) means the generative models (`train_mug_cvae/gan/diffusion.py`) still need retraining against the current tables — not yet done by any session.

---

## Bracket (topology optimization) — removed once, rebuilt on Warp, currently minimal-but-correct

**First attempt (removed)**: a 2D DCT-parameterized, scipy-sparse FEA + SIMP domain, built out to a real 150k-row dataset, a trained surrogate/VAE, and a viewer — but with a single-clamped-edge cantilever boundary condition that never produced anything a user would recognize as a real bracket topology, across four consecutive viewer-fix attempts. **Fully removed** on direct user judgment ("nothing is even slightly valid... you're not even remotely close to something physically sensible") — all scripts, checkpoints, datasets, and viewer pages deleted (this log's record of what was tried was kept, only the code was removed).

**Rebuilt from scratch on NVIDIA Warp** (`warp.fem`, GPU kernel/autodiff), per explicit instruction not to hand-write another solver, with the two real design fixes the first attempt lacked: genuine 3D geometry (`Grid3D`) and a real two-fixed-mount + one-load topology (not a single clamped edge). Real 3D linear-elasticity forward solve + SIMP optimization with the standard self-adjoint compliance-sensitivity shortcut, two real bugs caught by watching the numbers misbehave (a missing sign flip that was ascending compliance instead of descending it; a raw-integral-vs-mean units bug in the volume constraint) — both fixed and verified: compliance dropped 20-33x across two real, different cases (a symmetric 2-mount case and a harder asymmetric/angled-load case), with material genuinely concentrated along the straight-line mount-to-load paths, confirmed via marching-cubes mesh export and a real THREE.js render (not a density-grid colormap this time) directly verified live in Chrome, screenshots sent to the user for their own judgment.

**Real optimization gain confirmed, not just assumed**: a fair question after the first two cases ("the v-shaped trusses aren't actual optimal or anything, are they?") -- for a single point load connecting two point supports, a straight-strut answer is close to the textbook-obvious statics guess, so a V-truss coming out doesn't by itself prove anything non-trivial happened. Built a direct test: a naive density field (two straight tubes connecting each mount to the load, radius chosen so its volume fraction exactly matches the SIMP run's target) scored through the identical forward solver. First result was bad news, reported honestly: the naive guess (0.277 compliance) actually BEAT the "optimized" result (0.318) despite the SIMP run using MORE material -- the original 150-iteration default was genuinely under-converged, not actually optimal. Fixed by running longer (confirmed real convergence: flat to 5 decimal places by ~iteration 1420) and correcting the default to 1500 iterations. Once actually converged, SIMP genuinely beats the naive baseline -- 1.07x on the symmetric case, 2.37x on the harder asymmetric/angled-load case (more asymmetry leaves more room for a non-obvious layout to win) -- and this comparison now runs and prints automatically on every `bracket_simp_warp.py` call and shows in the viewer's stats panel, not just checked once by hand.

**Viewer brought up to par with the other domain pages** (direct request: "let's go forward and see if we can actually optimize one ourselves" -> then "I'd love to get this spun up to par with the other tasks"): added a wireframe overlay (a real second `THREE.LineSegments` from `THREE.WireframeGeometry`, toggleable independently of the shaded solid) and a von Mises stress shading mode. Stress is computed per-cell at the FINAL converged (rho, u) in `bracket_simp_warp.py` (same SIMP-scaled Hookean stress as the forward solve, so void cells read near-zero rather than reporting stress the material would carry if it were there — checked: correlation with density 0.84, mean stress in solid cells ~660x the mean in void cells), then trilinearly resampled onto the marching-cubes surface's own vertices in `bracket_export_mesh_warp.py` (`scipy.ndimage.map_coordinates` against an edge-padded stress grid, reusing the exact grid-index convention already established for density) and rendered as real per-vertex `THREE.BufferGeometry` vertex colors, not a decorative gradient. Verified live in Chrome on both cases: stress concentrates exactly where it should (the sharp bend under the load, the mount contact regions), legend range matches the solver's own reported min/max exactly (0.00-2.58 symmetric case, 0.00-12.05 asymmetric case), no console errors, `pytest` 14/14.

**16-candidate sweep, biased toward asymmetric configs** (direct request: "let's just like... make a whole bunch of candidates? I guess the asymmetric one is more interesting, yeah?"): `bracket_generate_candidates_warp.py` samples mount positions (left/right thirds of the domain, never mirrored), load position (off-center between the mounts), and load direction (10-35° off vertical, random azimuth) plus volume fraction, then runs each through the existing single-load-case solver at the same resolution/iteration count already verified to converge (res (40,20,6), 1500 iters). All 16 converged cleanly and all 16 beat their own naive two-tube baseline — ratios ranged 1.09x (a near-symmetric roll of the sampler) to 3.52x (highly asymmetric mounts + off-center load), confirming the asymmetry-vs-optimization-gain relationship already seen in v1/v2 holds across a real sample, not just those two cases. The index (`bracket_candidates_index.json`, ranked by naive-vs-SIMP ratio) is read directly by the viewer to populate a "Candidate sweep" dropdown group — no hardcoding. Visually inspected the top candidate (3.52x) live in Chrome: a genuinely non-obvious branching shape, not a simple V — confirmed both mount points present via direct density query (0.71 / 0.32 mean rho at each mount region) after one was initially camera-occluded, same false alarm as the original v1 render.

**Considered, not yet built**: extending to true multi-load-case optimization (e.g. a bridge with two end supports and several concurrent/alternating span loads) — mechanically straightforward (same K(rho), multiple RHS solves, sum the self-adjoint sensitivities per case, no new adjoint machinery needed) but deliberately deferred in favor of the parameter sweep first, per direct user choice.

**Real surrogate-training dataset, 3 fixed force layouts** (direct follow-up: "I would ideally like to do it for like... 2 or 3 force layouts?"). First clarified what the ORIGINAL (fully deleted) dataset actually was, since the user asked directly: one fixed single-clamped-edge BC, shape varied via random DCT coefficients, `(shape) -> (compliance)` rows — no force-layout variation at all, which was part of why it never looked like a real bracket. `bracket_generate_dataset_warp.py` reuses that shape-varies-under-fixed-BC structure but on 3 already-validated REAL layouts (v1 symmetric, v2 asymmetric, and candidate_12 -- the 3.52x sweep winner), each with two real sources of shape diversity: (1) the actual SIMP optimization trajectory (30 snapshots per layout, from the uniform start to the converged optimum -- real intermediate structure, not synthetic), and (2) 150 random smooth density fields per layout, synthesized from low-frequency 3D DCT coefficients (generalizing the original's DCT idea to 3D) and volume-fraction-matched via bisection on a logistic threshold. Every shape scored through ONE real forward FEM solve (`solve_u_for_rho`, same SIMP-scaled stiffness used everywhere else) against geometry/BC machinery built once per layout and reused across all 180 rows for that layout. Caught and fixed one real bug immediately: warp's own per-cell dof ordering for a degree-0 field is NOT a row-major flatten of the (i,j,k) resolution grid, so a naively-flattened random field silently scored the WRONG shape against the WRONG cells -- fixed by recovering the same index-via-physical-position mapping already used in the mesh exporter (`grid_to_dofs`), verified end to end with a small smoke test before running the real 540-row batch.

Real numbers, 540 total rows across 3 layouts (`bracket_dataset_{v1_symmetric,v2_asymmetric,candidate_12}.npz` + `bracket_dataset_index.json`): each layout's reference-optimum compliance in the new run matched its previously-logged value exactly (0.2586, 1.3697, 0.5043) -- a real reproducibility check, not just a hopeful assumption. Random DCT shapes score dramatically worse than the SIMP optimum at comparable material budgets (e.g. v2: optimum compliance 1.37 vs random-shape range up to 354) -- good, expected dynamic range for a surrogate to actually learn from, not a degenerate dataset where every row looks the same.

**Dataset scaled up, then a real generative pipeline (GAN + scoring surrogate + gradient-steered "direct new geometry") closes the domain's A/B/C/D bar** (direct request: "I want to like... generate geometry and score it and be able to like... direct new geometry - as close to parity as possible with the other problems"). Before building anything, flagged a real structural gap honestly rather than assuming parity was free: this domain's "shape" is a full voxel density grid (4,800-12,544 dims), not a compact parametric vector like stellarators' 92 Fourier coefficients or airfoils' 16 CST coefficients, so it needs proportionally more data for a generative model not to just overfit -- user chose to scale up the existing 3-layout dataset (150 -> 3,000 random shapes/layout, ~9,090 total rows) rather than add more layouts or skip the GAN.

Followed the project's own established generative-model convention exactly rather than inventing a new one: `train_bracket_scoring.py` (plain MLP, same shape as `train_mug_scoring.py`, log-space compliance target, 75-79% error reduction over a computed naive-mean baseline on all 3 layouts) and `train_bracket_gan.py` (WGAN-GP reusing `train_gan.py`'s Generator/Critic UNCHANGED, same convention `train_mug_gan.py` already established for a second geometric domain -- conditioned on a single scalar, target compliance). Added `warp-lang`+`scikit-image` to `Dockerfile.train` (verified both torch+CUDA and warp+CUDA import in one container) so the real oracle and the surrogate/GAN can be composed in one process, the same "oracle lives alongside its surrogate" pattern `Dockerfile.train` already uses for vmecpp/xfoil.

`bracket_direct_geometry.py` is the actual "direct new geometry" tool, two modes: `sample` (unconditioned generation, every candidate scored through the REAL oracle, not the GAN's own opinion -- calibration checked explicitly: predicted/real compliance ratio 1.2-1.3x across all 3 layouts, "well-calibrated" but not exact) and `steer` (gradient-guided latent walk -- backprop the trained scoring surrogate's prediction through the generator back to z, walk downhill, but the REAL oracle is still the sole accept/reject test every single round, same rule `gradient_walk.py` uses elsewhere, affordable here every round instead of periodically because this domain's real oracle is cheap).

**Caught a real, embarrassing-if-shipped bug before reporting any numbers (§102)**: the first version of `steer` had no volume constraint at all, so its v2 result looked like it BEAT the real SIMP optimum (0.80x) -- until checking the volfrac showed it used 89% material vs. SIMP's 25% target. Exactly the "more material always helps compliance" trap the naive-baseline check (§100) exists to catch, just in a new place. Fixed by adding a volume penalty to the steering objective (mirrors `bracket_simp_warp.py`'s own `vol_penalty_weight` term) and separating "best candidate within volume tolerance" (the only fair number) from "best candidate at any volfrac" (kept as a visible diagnostic, never exported or claimed as a result).

**Honest final numbers, all volume-matched**: unsteered GAN samples land 3.5-11.4x worse than the real SIMP optimum for their layout; gradient-steered candidates do better but are still 12.7-26.4x worse (one layout, candidate_12, found no volume-feasible candidate at all until step count/penalty weight were increased, then landed at 20.5x). This is a real, useful, working pipeline -- mechanically at parity with the other domains (real oracle validates every candidate, never trusts the surrogate alone) -- but NOT competitive with real SIMP at this dataset scale (~3,000 rows/layout is still small for a voxel-grid generative model, consistent with the scale gap flagged before building this). Framed honestly in the viewer itself (12 GAN-generated/steered candidates in a "Generated (GAN)" dropdown group, each labeled with its real ratio vs. the SIMP reference, explain text stating plainly that it's not yet competitive) rather than only in the log.

**Directed data augmentation closed most of the quality gap (§103)** (direct follow-up: "let's generate more data - explore in a directed way - yeah? See if we can get to a better spot"). Diagnosed WHY the first dataset underperformed before just adding more of the same: 3,000 random-DCT shapes/layout is dominated by mediocre-to-bad shapes (random fields rarely land anywhere near a real optimum), so the scoring surrogate and GAN had very few low-compliance examples to learn the shape of that region from. `bracket_augment_dataset_warp.py` instead perturbs already-known GOOD anchors -- the SIMP reference optimum, its own 30 trajectory snapshots, and the previously-exported GAN candidates -- with varying Gaussian noise, volume-rematched via the same bisection trick, +5,000 rows/layout (~8,000-8,050 total/layout), every one still scored through the real oracle (directed changes WHERE samples are drawn from, never relaxes the "always validate against real physics" rule). Real, immediate signal even before retraining: the perturbation sweep alone found shapes that slightly BEAT the original SIMP reference optimum on all 3 layouts (e.g. v1: 0.2473 vs. 0.2586) -- plausible local-search luck around a not-perfectly-global SIMP optimum, not claimed as more than that.

Retrained scoring (82.8-85.9% error reduction, up from 75-79%) and GAN on the enlarged datasets, reran `steer` with the exact same volume-tolerance-gated fairness fix from §102. **Real, large improvement, all still volume-matched against SIMP**: v1 12.7x -> 6.2x, v2 26.4x -> 2.6x, candidate_12 20.5x -> 4.1x. Calibration also improved (predicted/real compliance ratio 1.08-1.11x, down from 1.2-1.3x). Visually confirmed in Chrome: the re-steered v2 candidate is now a coherent, connected bracket-like solid, not the disconnected noise-blobs the pre-augmentation version produced. All 12 viewer candidates (mesh + index) re-exported and re-copied to the viewer's "Generated (GAN)" group, explain-box copy updated with the real before/after numbers instead of a stale dataset-size claim, `pytest` 14/14.

**Second round of directed augmentation, anchored on the first round's OWN improved candidates** (direct follow-up: "let's keep going?"). Same `bracket_augment_dataset_warp.py`, but this round's GAN-generated anchors were themselves already much better (the §103 round-1 steered candidates, 2.6x-6.2x vs. SIMP, not the original raw SIMP optimum alone) -- a real bootstrap loop, same shape as the stellarator domain's own "generated+validated designs feed back into training data" pattern, applied here for the first time. +5,000 more rows/layout (~13,000-13,050 total/layout), retrained scoring (79.0-83.9% error reduction) and GAN at 400 epochs (up from 300, since there's now enough data to use it), re-steered with more steps (80, up from 40-60).

**Further real improvement, reported completely honestly including the one regression**: v1 6.2x -> 2.1x, candidate_12 4.1x -> 2.5x -- but v2 actually got slightly WORSE this round, 2.6x -> 3.1x. Not hidden or explained away -- most likely ordinary WGAN-GP training noise (adversarial training is not monotonically improving run-to-run, a well-known property, not unique to this domain) rather than a real sign that more data hurt v2 specifically, but genuinely not confirmed either way. Calibration also improved further (predicted/real compliance ratio 0.975-1.023, effectively unbiased, up from 1.08-1.11x last round and 1.2-1.3x originally). Visually confirmed in Chrome: the v1 steered candidate is now recognizably the same V-truss topology SIMP itself finds, with one small floating disconnected fragment as the visible remaining defect -- a concrete, honest picture of "close but not there," not just a compliance number.

**Viewer rebuilt to mug-v2's own 3-pane convention, fixing a real discoverability bug** (direct report: "I'm not seeing the candidates in the viewer... bring the bracket viewer more in-line with the other viewers"). Diagnosed before touching anything: the sweep/generated candidates WERE loading correctly (confirmed via DOM query -- all 16 + 12 options really were in the page), they were just invisible, buried inside a collapsed native `<select>` with two `<optgroup>`s the user would have had to click open and scroll through to ever notice -- a real UX gap, not a data bug. Rebuilt around the exact 3-pane pattern `mug-v2.astro` already established (left pane: about/pool-switch/prev-next-random navigator with a numeric index input; center: 3D view + stats; right: a full scrollable, clickable list of every case in the current pool with its key metric shown inline) rather than inventing new conventions. Three pools (Curated/Sweep/Generated) as segmented buttons instead of dropdown optgroups. Verified live in Chrome: pool-switching, prev/next/random, direct row-click, and both display toggles (stress shading, wireframe) all work with the new layout, no console errors, `pytest` 14/14.

**Connectivity cleanup + real physical-unit metrics + a real tens-of-thousands-scale candidate pool** (direct feedback: "the very small number of candidates as well as all the non-physicality. The floating islands and stuff. SIMP compliance is also like... not an interesting metric... how much load can it take and what's the weight... shouldn't we have like... tens of thousands?"). Asked the user to pick a real material/scale (Aluminum 6061, 200x100x30mm bracket envelope) and where the scale-up should come from (GAN samples, real-oracle-filtered, not more expensive SIMP runs) rather than guessing.

`bracket_postprocess.py` (new): (1) connectivity cleanup -- 26-connected component labeling on the rho>=0.5 grid, keeps only material that actually spans BOTH mounts AND the load (a real load path), zeros everything else, always re-scored through the real oracle afterward so no reported number reflects a pre-cleanup shape. (2) Physical units -- the FEM solve itself stays in its existing well-conditioned dimensionless units (E0=1) rather than being rederived in real SI scale (would risk destabilizing Adam's step sizes for no physics benefit, and topology doesn't depend on absolute E anyway); real units are a clean post-hoc rescaling using the real fact that for a FIXED shape/load, stress is independent of E. First calibration attempt (FORCE_SCALE picked for a "~300N, someone hangs a shelf" nominal load) produced a real but absurd-looking number (545x safety factor, 16,000kgf max load) -- not a bug, a genuine consequence of SIMP optimizing compliance under a volume budget rather than a stress constraint, so it's routinely far stronger than it needs to be against yielding. Recalibrated to land reference cases at a more legible 3-11x safety-factor range instead of hiding the underlying finding.

**Caught a second real bug from the FIRST large-scale run, before trusting any of it (§104)**: connectivity cleanup's fallback ("if nothing spans mount-mount-load, keep the largest component anyway so there's something to render") let a candidate with almost no material (52 vertices, 0.23% occupied) rank #1 by strength-to-weight, because near-zero mass divided into any nonzero max-load reading produces a meaningless, enormous ratio. Fixed by returning `spans_load_path` from the cleanup function and filtering non-functional candidates out of ranking entirely (not just deprioritizing them) -- confirmed on a 100-candidate rerun: mass jumped from 4g/52-vertex garbage to a real 178g connected shape once the filter was in place.

**Caught a third real bug mid-run, this one costing real compute (§105)**: an early cleanup command deleted the `dataset_<layout>_ref_meta.json` reference files a batch's SIMP-comparison printout depends on; the first full 10,000-candidate v1 run scored EVERY candidate successfully through the real oracle, then crashed on that missing file BEFORE exporting any of them -- losing the entire batch's compute because the crash happened after the expensive part, not before it. Fixed two ways: made the reference-file load fail gracefully (`load_ref_meta`, returns None instead of crashing, callers degrade the comparison string rather than losing results) AND regenerated the actual missing files. Rerun succeeded end to end.

**Real numbers, 30,000 candidates scored across 3 layouts (10,000 each), real oracle + connectivity cleanup on every single one**: 29/10,000 (v1), 2,598/10,000 (v2), 1,464/10,000 (candidate_12) were non-functional (no real load path) and excluded from ranking -- a real, meaningfully different rate per layout, not noise. Top 100/layout (300 total, ranked by strength-to-weight = max_load_kgf/mass_kg) exported as real mesh candidates for the viewer, replacing the old 12. Calibration excellent across all three (predicted/real compliance ratio 0.975-1.013). Viewer's "Generated" pool now shows real weight/max-load pills on every candidate, a sort control (strength/weight, max load, lightest) for re-ranking the 300 client-side, and the stats panel leads with weight/nominal load/safety factor/max load before the abstract compliance numbers. Verified live in Chrome: sort-by-lightest correctly resorts across all 3 layouts mixed together (150g top result), real connected geometry, no console errors, `pytest` 14/14.

**A real connectivity bug caught by direct user inspection (§106): 26-connectivity was never the right definition, and it made "cleaned" candidates that weren't actually contiguous** (direct feedback: "the examples I saw weren't contiguous at all, so they were fundamentally not real"). Checked directly rather than defending the earlier fix: `ndimage.label` with 26-neighbor (corner-touching) connectivity called a specific exported candidate "1 connected component" -- under FACE (6-neighbor) connectivity, the physically meaningful definition, that same shape was actually 20 separate pieces (a 446-cell main body, a 249-cell piece touching only at a corner, plus 18 fragments). Two voxels sharing only a corner have ~zero real contact area and transfer essentially no load; 26-connectivity was simply the wrong tool for "is this one real part," not a tuning choice. Fixed in `bracket_postprocess.py` by switching to `ndimage.generate_binary_structure(3, 1)` (strict face connectivity) and re-verified directly on the same test case (now genuinely 1 component). Also added `spans_load_path` to the exported mesh JSON itself (it was in meta.json/the index but never reached the file the viewer actually fetches).

Also fixed the same reference-file-deletion mistake a second time (own cleanup command deleted `dataset_<layout>_ref_meta.json` again) -- this time `load_ref_meta`'s §105 fix caught it gracefully (comparison text just says "unavailable" instead of crashing), so no compute was lost, unlike the first time.

**Direct feedback: "without a comparable tsne or metric filtering like the other panes"** -- added the same feature `index.astro`'s own dataset explorer has: a client-side UMAP scatter (`umap-js`, already a project dependency) over the real metrics (weight, max load, volume fraction, z-scored), color-by selector, click-to-select, plus min/max range filter rows (mug-v2's own filter-row pattern) that narrow the browse list AND feed the map -- for the sweep and generated pools (curated only has 2 items, not worth mapping). Verified live: filtering to max-load>=6000kgf correctly narrowed 16->7 sweep candidates with the map/list/selection all staying in sync.

**Re-ran the full regeneration under corrected connectivity**: curated (v1/v2) and all 16 sweep candidates confirmed genuinely well-connected even under the stricter check (SIMP's own compliance-minimization naturally avoids islands; the corner-touching problem was specific to GAN output). The 30,000-candidate bulk generation, rerun with the fix, dropped far MORE candidates as non-functional than the buggy version did -- v1 232/10,000 (up from 29), v2 4,607/10,000 (up from 2,598), candidate_12 3,217/10,000 (up from 1,464) -- a real, large difference confirming the old check was letting through candidates that should have failed. 300 curated top candidates (100/layout) re-exported and verified genuinely single-component, calibration still excellent (0.922-0.997).

**Viewer layout brought into actual structural parity with the other domain pages** (direct request: "Can we try to get the UI more in line with the other tabs?"). Checked concretely rather than guessing at "more consistent" -- diffed bracket.astro's DOM structure against index.astro/airfoil.astro/mug-v2.astro and found two real, fixable gaps: (1) every other page shows a `<div id="subtitle">` line under the nav with a live "N candidates loaded" count -- bracket had none; added, wired to update whenever any pool's index loads. (2) airfoil.astro's own map+filter convention (confirmed identical in index.astro) puts the UMAP map paired WITH the candidate navigator in the LEFT pane, and Filters alone in the RIGHT pane -- bracket had built the opposite arrangement (map+filters+browse all stacked in the right pane, pool nav alone on the left). Moved the map card to sit with Pool/nav on the left, matching that convention exactly; right pane now holds only Filters + the Browse list. Also matched both side-panes to the project's common 340px width (airfoil's own value) instead of an ad-hoc 320/400 split. Verified live: subtitle populates correctly ("318 candidates loaded (2 curated, 16 sweep, 300 generated)"), map+nav+filters+browse all still function correctly in the new positions, no console errors, `pytest` 14/14.

**Real histogram filters (not plain min/max boxes), stress-on-by-default, and the flat candidate list removed entirely** (direct follow-up: "make sure we have the histogram filters too... have the stress color turned on should be the default... I also don't want... the list of candidates... get us as close to parity with the stellarator visualizer as possible"). Read index.astro's actual filter component source rather than approximating it -- collapsible metric cards (chevron, inline current-value, percentile-vs-pool pill), a live histogram canvas per metric with draggable amber range handles (click-drag on the bars, or click a numeric label to type an exact value), a "reset zoom" per metric and "reset all" for the card set, and a match-count progress bar -- ported essentially verbatim (same drag hit-testing, same live-preview-while-dragging redraw, same click-a-handle-to-edit pattern), swapping the 11 physics targets for this domain's 3 real metrics (weight, max load, volume fraction). The flat scrollable "Browse" list is gone -- neither index.astro nor airfoil.astro have one either; navigation is now filter + UMAP-map-click + prev/next/random only, matching how those pages actually work. Shading default flipped from flat material to von Mises stress (was an extra click before, now the first thing shown). Verified live end-to-end: dragging/editing a histogram handle correctly narrows the match count (300->199 on a max-load cutoff), cascades to the current candidate, the map, and the stats panel; clicking a UMAP point selects that exact candidate; the 16-item sweep pool (a real small-N edge case) renders and filters correctly too. `pytest` 14/14, no console errors.

**Contact validation tightened at the source, not just the render (§107)** (direct feedback: "it has to actually have contact points where we say it does"). Audited the actual connectivity-check geometry rather than assuming the §106 face-connectivity fix was sufficient, and found two real remaining gaps: (1) the mount/load "contact zone" used to seed the check was a `mount_radius x 1.5` 3D-sphere centered at mid-depth (z) around a single point -- a genuine mismatch with the solver's own boundary condition, which is actually an x-strip spanning the FULL z-depth at the y=0 (mounts) or y=max (load) face (`bracket_fem_warp.classify_boundary_regions`), no z-restriction and no radius padding at all; the sphere approximation could both miss real contact far from mid-z and credit material that was never in the solver's actual BC region. (2) "touching" only required ONE cell overlapping the zone -- a single corner-nick counted as full contact. Fixed both: `bracket_postprocess.boundary_strip_cells` now uses the solver's own exact BC region (no padding, no sphere), and `clean_disconnected_islands` now additionally requires >=20% of each contact strip's own cells to be real material (`MIN_CONTACT_COVERAGE`) -- genuine bearing area, not a token touch. Verified directly on real data before trusting it: a candidate showing plausible 29-92% coverage at each of its three attachment points, confirmed by direct query, not just by the code not crashing.

Re-ran the full pipeline under the stricter check: curated (v1/v2) and all 16 sweep candidates came back byte-identical to before (real SIMP results already had genuine bearing area, expected -- compliance minimization doesn't produce thread-thin contact). The 30,000-candidate bulk generation, rerun, rejected meaningfully MORE candidates than the connectivity-only check did -- v1 599/10,000 (up from 232), v2 5,165/10,000 (up from 4,607), candidate_12 6,218/10,000 (up from 3,217) -- real evidence the coverage requirement is catching genuinely marginal-contact candidates the connectivity check alone missed. Still 3,782-9,401 valid candidates per layout, plenty for a solid top-100 selection; calibration held (0.957-1.004). 300 curated candidates re-exported and re-verified, viewer confirmed live with no console errors, `pytest` 14/14.

**Current status**: forward solve + SIMP + mesh export + viewer are real, parameterized, self-checking per run, expose real solved stress (on by default) + a wireframe inspection mode + real physical-unit metrics (Al 6061, weight/max-load/safety-factor) instead of abstract compliance, automatically clean disconnected material using the physically correct (face-connectivity + minimum-coverage) definition of real contact and re-verify through the real oracle on every export, have a real multi-candidate sweep, a real 3-layout/~39,000-row surrogate-training dataset, a full real generative pipeline (scoring surrogate + WGAN-GP + oracle-validated latent steering + a real bootstrap augmentation loop), a real 30,000-candidate-scale bulk-generation pipeline (300 curated, sortable, genuinely-attached candidates currently in the viewer), and a viewer with real draggable-histogram metric filters + a UMAP map + 3-pane layout now structurally matching the other domains' own convention (map+nav left, histogram filters right, live subtitle count, no flat list). The domain's A/B/C/D bar is mechanically closed and has genuinely, repeatedly improved on quality, physical realism, explorability, and UI consistency -- generative quality still trails real SIMP. No `oracle_base.py`/`gym_schema.py` wrapper, no cross-layout conditioning (still 3 fixed layouts, each with its own model), no multi-load-case ("bridge") solver yet.

**Standing lessons from the whole arc, worth keeping regardless of which implementation is current**: (1) a topology-optimization problem needs boundary conditions that actually resemble the part being designed (two mount points + a load, not a single clamped edge) before its output can look like what its own name claims; (2) a density-field/topology render needs to show the reader a real reconstructed surface (or represent absence as genuine absence) — a colormap over a raw grid cannot show "material removed," because nothing in it is ever actually absent, and no amount of contrast-tuning fixes that; a real marching-cubes mesh does; (3) a plausible-looking converged result (monotonic loss curve, sensible shape, volume fraction near target) is not proof of near-optimality — compare against the simplest non-trivial baseline that could beat it, on every run, not just when something looks suspicious.

---

---

## Photonics (silicon grating coupler, MEEP) — 6th real domain, `gym_schema.py`'s own "unconfirmed" row now confirmed

**Task**: `[wg_thickness_um, grating_period_um, duty_cycle, etch_depth_um]` (PARAM_DIM=4) → `up_efficiency`/`down_efficiency`/`transmitted_efficiency`/`reflected_efficiency`/`energy_closure`, a real 2D silicon-on-insulator grating coupler at 1.55um (telecom C band), N_PERIODS=20 fixed.

- Real MEEP FDTD, not a self-written EM approximation — same "use the real accepted solver" call made up front here that mug only reached after its v1→v2 pivot off self-written thermal FD onto OpenFOAM. **The `meep` PyPI package is a squatted/unrelated stub** (imports fine, contains no simulation code at all) — the real package is conda-forge-only (`pymeep`, nompi build), needing its own standalone image (`Dockerfile.meep`, `condaforge/miniforge3` base) exactly like TORAX's hard version-pin forced a standalone image rather than a pip add to an existing one.
- **`gym_schema.py`'s own `CANDIDATE_TRIAGE` table had flagged this row "harness fit: UNKNOWN... per-call cost profile is genuinely unconfirmed either way"** — now measured directly, not assumed: low fidelity (resolution=10 px/um) ~1.26-1.31s/candidate (~43 credits, normalized to airfoil-low's 1-credit anchor), medium (resolution=20) ~3.60-3.81s/candidate (~123 credits) — both include the 2 real MEEP runs a single candidate needs (see below). No JIT-compile tax the way TORAX/JAX-Fluids/Warp domains have — repeat calls in the same process showed no warm/cold speedup — so `harness_fit="subprocess-per-candidate"` is confirmed, same shape as VMEC++/XFOIL/mug, not TORAX's.
- **Two real MEEP runs per candidate, not one**: a single monitor can't cleanly separate "forward launched power" from "reflected power" at one point, so each candidate first runs a calibration pass (the same waveguide with NO grating, for a clean input-power reference) then the real grating pass — MEEP's own standard normalization technique, not invented here.
- **A real physical sanity check confirms the setup isn't buggy, found by running it rather than assuming**: `energy_closure` (up+down+transmitted+reflected, should land near 1.0 for a real energy-conserving FDTD run) came out 1.078 at low resolution and 1.015 at medium — converging toward 1.0 as resolution increases, exactly the direction real FDTD numerical error should move, not a coincidence.
- Validity is the same character as mug's, not VMEC++'s: no non-convergence failure mode exists (a fixed-simulated-time FDTD run always completes), so `ok=False` only ever means a structurally invalid design (etch depth exceeding waveguide thickness, non-positive geometry, duty cycle outside (0,1)) — checked before ever calling MEEP.
- **Scope of this FIRST pass, deliberately**: oracle + `gym_schema.py` wiring only (direct user choice, weighed against full A/B/C/D parity with the other 5 domains) — dataset/training/viewer followed in the two sessions below. `wg_thickness_um`/period/duty-cycle/etch-depth is a real, standard silicon-photonics grating-coupler parameterization, but a genuine 2D-cross-section simplification (no lateral confinement) — a future 3D version is real future work, not attempted here.

**Follow-up: real 3080-row dataset (`generate_photonics_dataset.py`) + two genuinely hard targets found, not invented (`photonics_targets.py`)**

- 100% hit rate, 3080/3000 rows, 364s wall-clock (28 parallel workers, low fidelity) — confirms the earlier single-candidate cost measurement scales cleanly; no failures at all, matching the module's own prediction that `ok=False` only happens on a structurally-invalid design (none were sampled here since the generator only ever emits physically valid combinations).
- Five literature-grounded seed archetypes (`canonical_630nm`, `full_etch`, `shallow_etch_weak`, `thin_soi_150nm`, `near_bragg_reflective`) each perturbed, mixed 50/50 with full-range uniform sampling.
- **`near_bragg_reflective` confirmed as a real hard/adversarial region, not just a hypothesis**: mean `reflected_efficiency` 0.361 across that archetype's 281 rows vs. 0.026-0.104 for every other archetype, with individual rows up to 0.99 — and the single worst row in the ENTIRE 3080-row dataset (`reflected_efficiency`=0.998) also falls in this same narrow period band (~0.25-0.33um), independently reached via uniform sampling too, not just the seeded archetype. Real physics: grating period near lambda/(2·n_eff) couples predominantly into backward (Bragg) reflection instead of radiating.
- **One hypothesis from the oracle's own docstring corrected by the real data**: `full_etch` was expected to leak power mainly downward into the substrate (removing the etch asymmetry that biases scattering upward); the actual dataset shows the opposite dominant loss channel — `full_etch`'s mean `reflected_efficiency` (0.467) is actually the HIGHEST of any archetype, higher even than `near_bragg_reflective`'s own (0.361), while its mean `down_efficiency` (0.273) is lower than `canonical_630nm`'s (0.556). Kept as a real, reported correction rather than silently fixing the docstring's prior guess.
- **The literature-canonical design is not actually the best one in this space for pure up_efficiency** — a real, checkable finding: `canonical_630nm`'s own best row only reaches `up_efficiency`=0.457, while plain uniform sampling found designs up to 0.774 (thicker waveguide ~0.32-0.34um, larger period ~0.50-0.57um, larger duty cycle/etch than the canonical shallow-etch recipe). Consistent with the canonical recipe being optimized in the literature for FIBER-mode-matched coupling efficiency at a specific angle, not for this oracle's simpler "total upward flux" metric — a real reason for the gap, not evidence the oracle is wrong, but a legitimate reason a search method needs to actually search rather than start-and-stop at the textbook point.
- **A genuinely hard multi-objective target, found by querying the dataset rather than assumed**: requiring `reflected_efficiency`<0.03 AND `down_efficiency`<0.15 simultaneously is satisfied by only 371/3080 rows (12%), and the best `up_efficiency` among THOSE rows is just 0.127 — a 6x gap versus the 0.774 unconstrained best. Suppressing both loss channels while keeping useful up-coupling is a real, measured tradeoff in this 4-dim space, not a constructed difficulty; captured as `photonics_targets.HARD_CLEAN_DIRECTIVITY`.
- `photonics_targets.py` adds three named `gym_schema.TargetTask`s for future steerability/search evals, same role as `p1_report.py`'s fixed VMEC target: `EASY_HIGH_UP_EFFICIENCY` (the dataset's own best row, a sanity baseline), `HARD_CLEAN_DIRECTIVITY` (the joint-constraint finding above), `HARD_LOW_REFLECTION_NEAR_BRAGG` (low reflection specifically within the adversarial period band).

**Full generative pipeline (cVAE/GAN/diffusion) + steerability eval + live viewer — A/B/C/D bar closed**

- `train_photonics_{cvae,gan,diffusion}.py` reuse `train_cvae.py`/`train_gan.py`/`train_diffusion.py` unchanged (`n_nfp=0`, same no-aux shape as TORAX's own trainers, not mug's categorical-column case — all 4 photonics params are continuous). No LOG_TARGET_NAMES needed: all 5 targets are physically bounded to ~[0, 1.2], nothing spanning orders of magnitude the way Q_fusion does. All three trained clean on the real 3080-row dataset (300 epochs each, tiny MLPs, ~11K-13K params).
- `eval_photonics_steerability.py` (adapted from `eval_airfoil_steerability.py`, aux handling dropped) confirms real validity across all three architectures: cVAE 95.8%, GAN 98.6%, diffusion 98.6% (72 real MEEP calls each, 12 random anchors).
- **The project's own repeated cross-domain finding (GAN > Diffusion > cVAE on steering signal) replicates on a 7th domain**: mean correct-steering-direction rate across the 4 real physical targets (excluding `energy_closure`, a diagnostic, not an independent design axis) — GAN 74.3%, Diffusion 73.3%, cVAE 68.1%. Same ranking as VMEC++/airfoil/TORAX/mug, GAN's generator param-matched to the others as usual.
- **`up_efficiency` confirmed as this domain's own hardest-to-steer target, consistently across architectures** — cVAE's correct-direction rate on it (51.4%, a near coin-flip) is its worst of any target by a wide margin; GAN/diffusion do much better (76.7%/76.5%) but still not their best target. Consistent with the earlier dataset finding that `up_efficiency` depends on a non-obvious combination of all 4 design params rather than one dominant knob (recall: the literature-canonical design wasn't even close to the dataset's own best `up_efficiency` row).
- `photonics_physics_service.py` + `/photonics` viewer page: live MEEP FDTD in the browser, not a lookup table or surrogate — every slider move triggers a real 2-run simulation (debounced ~350ms) via the `physics-photonics` compose service, drawing the actual simulated 2D cross-section (waveguide + real grating tooth rectangles, to scale) alongside the 5 live efficiencies. Verified directly in Chrome: the canonical seed reproduces the exact numbers from the original single-candidate smoke test, and the `near_bragg` seed reproduces the adversarial finding live (`reflected_efficiency`=0.941, `energy_closure`=0.999).
- `Dockerfile.meep` extended with CPU torch + fastapi/uvicorn (same reasoning as `Dockerfile.torax`'s own additions) so one container can run real MEEP oracle calls AND load a trained checkpoint in the same process — needed by both the steerability eval and the physics service.

**Dataset scaled 3080 → 18,053 rows** (direct user question: "how much data do we have... should we be generating more"). At 3080 rows this was the smallest dataset of any domain in the project both in absolute size AND density (770 rows/dim vs. 1,500-2,800 rows/dim for VMEC++/airfoil/TORAX/mug) despite having the cheapest, 100%-hit-rate oracle of any domain — a real, checkable mismatch between "how much this domain COULD generate" and "how much it HAD," not a judgment call. Topped up with `generate_photonics_dataset.py --target-count 14920 --medium-fraction 0.15`: 14,973 new rows accepted, 99.9-100% hit rate throughout, 2705s (~45 min) wall-clock — matches the original run's measured 0.12-0.18s/attempt cleanly at 5x the scale. `photonics_fidelity.json` (new, parallel to the existing `photonics_source.json`) now tracks which tier scored each row (15,843 low / 2,210 medium, ~12.2% medium — close to the requested 15%, the difference being the original 3080 rows were backfilled as 100% low) since low/medium have real, different, measured numerical-error characteristics (energy_closure error ~7-8% vs. ~1-2%) a model reading this dataset should be able to condition on. Final density: 4,513 rows/dim — now the DENSEST dataset per dimension of any domain in the project, not just no-longer-the-thinnest. Generative-model checkpoints/steerability eval retraining on the full 18,053-row set (was 3080) — in progress as of this entry, numbers to follow.

**Real field visualization added to the viewer** (direct user feedback: viewer only showed a flat 2D cross-section with no way to see anything beyond the summary efficiencies) — `photonics_oracle._run_sim` now optionally captures a DFT (frequency-domain) field monitor for `mp.Ez` at the same `fcen` the flux monitors already use, NOT a raw time-domain snapshot (which would show near-zero after the pulse decays — the DFT integrates the field's own steady-state frequency component throughout the run, the physically correct technique, same idea as the flux monitors' own DFT machinery). Only requested by `photonics_physics_service.py` (`capture_field=True`) — the dataset-generation harness (`worker_fn`) never pays the extra cost. The viewer renders `|Ez|^2` as a log-scaled heatmap (intensity spans ~3+ orders of magnitude) in the field's own real coordinate system, with the silicon structure outlined on top — the grating's position relative to the field extent is an EXACT constant (`DPML_SRC_GAP_UM + CALIB_TAIL_UM = 1.5um`) derived from the oracle's own cell layout, not fitted. Verified directly in Chrome: the canonical design shows a clean diagonal radiation lobe pattern (light diffracting up/down off the grating, as expected); the `near_bragg` design shows a visually obvious standing-wave pattern concentrated near the input with almost no radiation lobes — a real, physically distinct field signature matching its `reflected_efficiency=0.941` number, not just a different bar-chart value. The genuinely 2D nature of this domain (no lateral confinement simulated at all, see the oracle's own docstring) was surfaced directly rather than papered over with a decorative 3D extrusion.

**Real-world framing + rigor pass** (direct user request: "I don't know what we're actually trying to optimize... bring the framing and context here up to a similar level" as the other domains, then, after a casual literature comparison read as an overclaim: "you're saying our evaluation/simulation produces meaningfully wrong numbers for a known domain?"):

- Published a primer explaining the real device (a grating coupler is one of two standard ways to get light on/off a silicon photonics chip), the real 3-priority objective (maximize `up_efficiency`; keep `reflected_efficiency` low because back-reflection destabilizes the driving laser, a real failure mode, not generic loss; keep `down_efficiency` low because it's permanently wasted power), and grounded "what good looks like" against real published numbers (uniform gratings ≈54-73% fiber-coupled, apodized ≈83-86%).
- **A casual side-by-side of our dataset's best `up_efficiency` (0.774) against those literature fiber-coupling numbers read as an overclaim once directly challenged** — correctly caught by the user. The two quantities are NOT the same axis: literature numbers are fiber-coupled efficiency at a tilted angle (~8-10°, chosen specifically to dodge the near-Bragg back-reflection this domain's own `near_bragg_reflective` archetype found) with a fiber-mode overlap integral folded in; our `up_efficiency` is total upward power at normal incidence with no angle filter and no mode overlap. A mismatch between them is a metric-definition difference, not evidence either simulation is wrong — this distinction wasn't stated clearly enough the first time and needed direct correction.
- **Four independent, real validation checks assembled in response, not just reassurance**: (1) a known-zero-case test — feeding the oracle a "grating" with zero-width grooves (duty_cycle≈1.0, physically identical to a plain waveguide, no scattering mechanism at all) returns `transmitted=1.0012, up=0.00008, down=0.00035, reflected=0.0` against the theoretically exact answer (1, 0, 0, 0) — 0.16% error, decisive evidence the flux-accounting logic (the calibration-run subtraction scheme underlying every number this domain reports) isn't systematically buggy; (2) MEEP's own official documentation reproduces analytic diffraction-grating theory to within 0.4-2.1%, the same order of magnitude as this oracle's own 1-8% energy-closure error, not orders of magnitude worse; (3) an independently published paper (Lomonte, Lenzini & Pernice 2021, real full-text read via PDF, not just an abstract) uses 2D MEEP FDTD with the IDENTICAL four-quantity accounting scheme (up/down/slab-leakage/back-reflected Poynting flux monitors) this oracle arrived at independently — real methodological corroboration; (4) literature confirms the up/down split is a genuinely design-sensitive quantity, not one with a single correct answer — published designs range from 80%up/8.7%down to 1.7%up/86%down depending on tuning, with the buried-oxide thickness (interference between direct-upward and substrate-reflected-then-upward light) as the dominant lever.
- **That last check surfaced a real, previously unflagged scope gap**: `BOX_THICKNESS_UM` is a fixed constant (2.0μm) in `photonics_oracle.py`, never exposed as a design parameter or swept, despite the literature identifying it as one of the more sensitive levers on exactly the up/down split this domain's primary objective cares about. Added to the primer's "what's out of scope" list and this log rather than silently noted — real future work (a 5th design parameter), not done here.
- Primer published as an artifact (grating-coupler cross-section diagram with the 4 real energy paths labeled, literature comparison table, the 3 named hard/easy targets reframed in plain engineering terms, the validation section above, sourced footnotes) rather than left as chat-only explanation, so it's a durable reference the way the other domains' EXPERIMENT_LOG framing already is.

**v2: PARAM_DIM 4→6, apodization + variable buried-oxide thickness + real fiber-mode-overlap target** (direct user follow-up: "I would love apodization and fiber-mode overlap and variable buried-oxide thickness... give us a lot more good variation and make it a more interesting optimization challenge" — a deliberate expansion BEFORE heavy optimization work, not scope creep found along the way). A breaking schema change, not additive — the v1 4-param/5-target dataset (18,053 rows) and its 3 generative-model checkpoints archived as `*_v1_4param_*` rather than silently overwritten (same discipline as every other domain's dataset-regeneration warning).

- **Apodization**: `duty_cycle` (one scalar) replaced by `duty_cycle_start`/`duty_cycle_end`, linearly tapered across the 20 teeth — the standard literature technique (matches Lomonte et al. 2021's own "linearly diminishing the FF from 85% to 25%"), not invented. `duty_cycle_start==duty_cycle_end` recovers v1's uniform grating exactly as the degenerate case.
- **Variable buried-oxide thickness**: `box_thickness_um` (v1's own fixed 2.0μm constant, the exact gap the v1 validation work flagged) promoted to a real 6th design parameter, range 0.5-3.0μm.
- **A real fiber-mode-overlap target, `fiber_coupling_efficiency`** — and a real debugging story worth keeping, not just the final answer: the FIRST implementation evaluated the radiated beam via MEEP's near2far transform at a physical 15μm fiber standoff distance, matched against a flat-tilted-plane-wave Gaussian fiber mode. Caught as wrong by a sanity check, not inspection: an apodized design (which the literature says SHOULD improve fiber coupling) instead collapsed the overlap by >30x versus a uniform design (0.017 vs. 0.46) — physically implausible enough to investigate rather than accept. Root cause confirmed directly: at 15μm from a ~12.6μm-wide grating aperture, the beam is nowhere near the far field (Fraunhofer distance ≈ aperture²/λ ≈ 100μm for this geometry) — it still carries real wavefront curvature a flat-phase model can't represent, and that curvature matters far more for a near-normal-incidence beam (the apodized case) than a strongly-tilted one (the uniform case), explaining the lopsided collapse. **Fixed by evaluating the overlap directly at the grating's own aperture instead** (no propagation at all) — which is also literally what apodization is designed toward in the literature (Lomonte et al.'s own "mode profile AT ITS BEAM WAIST"). Re-tested: uniform 0.75, apodized 0.69 — both plausible, comparable, no collapse. Cheaper too (reuses the same DFT-field technique as the viewer's heatmap, one 1D line instead of a separate near2far monitor).
- `fiber_coupling_efficiency = up_efficiency × mode_overlap_fraction` (matches Lomonte et al.'s own decomposition) is enforced `<= up_efficiency` by construction — verified as a real invariant across every test case (uniform/apodized/thin-box/thick-box), not just assumed.
- Per-call cost essentially unchanged (1.27s low / 3.62s medium, vs. v1's 1.26-1.31s / 3.60-3.81s) — the aperture-plane DFT line adds negligible cost, confirming it was the right implementation choice on performance grounds too, not just correctness.
- Two new seed archetypes added to `generate_photonics_dataset.py`'s existing five: `apodized_literature` (Lomonte et al.'s exact recipe, period=0.83μm, adapted to our 20-period grating vs. their 50 — genuinely underperforms in our simpler non-self-imaging setup, `fiber_coupling_efficiency`≈0.001, kept as-is rather than tuned to look good, a real finding that transplanting a recipe without its own supporting design tricks doesn't automatically work) and `thin_box_interference` (box_thickness_um=0.7, probing the up/down-split sensitivity the v1 validation work found).
- Viewer (`/photonics`) and `photonics_physics_service.py` updated for all 6 parameters (two duty-cycle sliders, a box-thickness slider) and the new `fiber_coupling_efficiency` bar; verified live in Chrome — apodized_literature's poor real numbers display correctly and consistently (fiber_coupling_efficiency still `<=` up_efficiency even in this weak case).
- Fresh 20,000-row v2 dataset generation launched (`--medium-fraction 0.15`, same as v1's own top-up); generative-model retraining and steerability eval to follow once it completes — numbers not yet in this log.

**v2 dataset generated: 20,018 rows, 99.3% hit rate, 4704s (~78 min)** — real analysis, same rigor as v1's own:

- **`fiber_coupling_efficiency <= up_efficiency` holds for all 20,018 rows, zero violations** — the internal consistency invariant from the implementation's own construction, now confirmed at full dataset scale, not just the 4 hand-picked smoke-test cases.
- **Apodization has NOT yet beaten a well-chosen uniform design on raw fiber-coupling efficiency at this dataset's scale** — a real, reported (not favorable-looking) finding: best uniform row reaches `fiber_coupling_efficiency`=0.661; best apodized row (duty_cycle_start/end differing by >0.15) reaches 0.653, essentially tied. Plausible reason, not yet confirmed: this domain's N_PERIODS=20 is short compared to the 50-70 periods real apodized designs in the literature use to gradually reshape the beam — apodization's real advantage may need more periods than this domain's fixed constant currently allows to show up. Kept as an honest result, not tuned to look better.
- **The joint "clean + high-coupling" constraint got real relief from the added parameters**: requiring `reflected_efficiency`<0.03 AND `down_efficiency`<0.15 simultaneously is satisfied by 1,864/20,018 rows (9.3%, similar to v1's 12%) with best `fiber_coupling_efficiency`=0.531 among those — but that's 80% of the unconstrained best (0.661), a MUCH smaller relative gap than v1's equivalent (0.127/0.774=16%, a 6x gap). Apodization/box-thickness genuinely opened up better jointly-good regions the 4-param space couldn't reach, not just added noise dimensions.
- **A real self-correction, worth keeping**: a first pass computed the linear correlation between `box_thickness_um` and up/down directionality across the whole dataset and got r=-0.04 — read initially as "no real dependence," which would have contradicted the literature-grounded reasoning that motivated adding this parameter in the first place. Binning `box_thickness_um` within one fixed design family (`canonical_630nm`'s own perturbation cluster) instead of across the whole heterogeneous dataset revealed a real, single-hump RESONANCE underneath the coincidentally-flat linear trend: mean directionality rises from ~0.31 (box=1.0-1.3um) to a peak of ~0.366 (box=2.0-2.1um), then falls back to ~0.27-0.32 above 2.9um — a real, physically-motivated Fabry-Pérot-style interference effect, not the "no effect" a naive linear-correlation check would have concluded. Caught by checking a binned/grouped view before trusting a single summary statistic, the same "verify from source, not summary" discipline this project's own header lesson names. `HARD_BOX_THICKNESS_RESONANCE` (`photonics_targets.py`) targets landing inside this measured peak specifically.
- `photonics_targets.py` rebuilt for the new 6-target order: `EASY_HIGH_FIBER_COUPLING` (dataset's own best row), `HARD_CLEAN_FIBER_DIRECTIVITY` (the joint-constraint finding above), `HARD_LOW_REFLECTION_NEAR_BRAGG` (unchanged in spirit — near_bragg_reflective's 1,423 v2 rows still average `reflected_efficiency`=0.372, more than double any other archetype), `HARD_BOX_THICKNESS_RESONANCE` (new — landing inside the measured resonance peak, not just anywhere in the box_thickness_um range).
- Generative-model retraining (cVAE/GAN/diffusion) on the full v2 dataset launched; steerability eval numbers to follow once complete.

**v2 retraining + steerability eval complete** (`train_photonics_{cvae,gan,diffusion}.py`, 300 epochs each on the 20,018-row/6-param/6-target dataset; `eval_photonics_steerability.py --n-anchors 12 --k-validity 6 --k-steer 3 --step-std 1.0`, `--out-tag photonics_v2_{cvae,gan,diffusion}`, real MEEP low-fidelity re-simulation of every steered candidate, no proxy scoring):

| model | validity_hit_rate | selectivity | mean correct_direction_rate (6 targets) |
|---|---|---|---|
| cVAE | 0.931 | **0.861** | 0.692 |
| GAN | 0.917 | 0.759 | **0.702** |
| diffusion | **0.958** | 0.815 | 0.690 |

- **The v1 "GAN > diffusion > cVAE" ranking does NOT clearly replicate on v2** — a real, reported (not favorable-looking) finding, not the result I was expecting to write down. On v1's simpler 4-param/5-target space GAN was unambiguously best across validity, selectivity, and direction-following; here each model wins on a different axis (diffusion best validity, cVAE best selectivity, GAN best mean direction-rate, and only by 0.01 over the other two — within noise at n_anchors=12). Plausible reason, not confirmed: the richer 6-param/6-target space (apodization + box-thickness resonance) may reward GAN's mode-seeking less now that "good" designs are more numerous and less concentrated in one narrow basin, but this hasn't been tested directly (e.g. by rerunning v1's own dataset/checkpoints with n_anchors=12 to rule out this just being a seed-count artifact of the eval itself, not a real architecture-vs-domain effect).
- Per-target detail worth keeping rather than just the summary: `up_efficiency` and `reflected_efficiency` steer cleanly for all three models (correct_direction_rate 0.79-0.95); `energy_closure` and `down_efficiency` are the hardest to steer for every model (0.45-0.65) — plausibly because both are downstream/coupled consequences of the other 4 targets rather than independent axes a latent-space nudge can isolate. `fiber_coupling_efficiency` (the new v2 target) steers unevenly: cVAE 0.758, GAN 0.625, diffusion 0.528 — the newest, most nonlinear target (product of `up_efficiency` and a mode-overlap term with its own resonance structure) is also the one with the widest spread across architectures, consistent with it being a genuinely harder steering target than the four "physically-averaged" channel efficiencies.
- Raw JSON: `output/eval_photonics_steerability_photonics_v2_{cvae,gan,diffusion}.json`.

**Direct user follow-up ("test whatever you feel we need... anything that might work is good")**: implemented the first of the two remaining named directions. Read the REAL official `constellaration.optimization.augmented_lagrangian` + `augmented_lagrangian_runner.run` source directly (not just the settings docstrings this time) and found it differs from both the docstring's own claims and this project's hand-rolled `ngopt_alm_p3.py` in ways worth stating precisely:
- **Per-constraint penalty parameters (vector rho_i), grown conditionally**: `rho_i <- min(rho_max=1e8, 2*rho_i)` only if `max(0,c_i) > 0.5 * max(0,c_i_prev)` (insufficient progress on THAT constraint specifically), else unchanged. Our version grows one global scalar `rho *= 1.15` unconditionally every outer iter regardless of progress -- explains why rho exploded to 1e12 in ngopt_v1/v2 while some constraints were still genuinely improving.
- **Growing per-outer budget** (`budget_initial` -> `budget_max` via `budget_increment` each outer iter), not a constant `budget_per_outer` forever.
- **Explicit `oracle.suggest(x_best)`** injected into the ask stream every outer iteration, on top of `param(init=x_best)`.
- **Correction to our own earlier reading**: the `bounds_reduction_factor=0.95` docstring turned out to be vestigial -- `update_augmented_lagrangian_state` only ever does `bounds = max(bounds_min, state.bounds)` unless a caller explicitly passes a shrunk value, and `augmented_lagrangian_runner.run` never does. Bounds are actually held CONSTANT at `bounds_initial` for the whole run in the real implementation -- "shrinking trust-region bounds" was never real behavior to reproduce. Kept our own proven stall-triggered widening logic instead.
- **Not grafted on** (explicit scope cut): the official code reparametrizes the search space by an infinity-norm spectral scaling per Fourier mode before optimizing (`compute_infinity_norm_spectrum_scaling_fun`) -- skipped, real lift for uncertain payoff, noted as a possible future lever if this run also stalls.

Implemented as `scripts/ngopt_alm_p3_official.py`, smoke-tested clean (10/10 converged, no errors). Launched `official_v1`: seeded from the best available point in the whole project (`output/omnigenity_seed2_anchor.npy`, 2/5 constraints already passing, smoke-tested worst_violation=0.42), 150 outer x budget 40->140 (+6/outer), 28 workers.

**Result: 8th independent confirmation of the same wall, via genuinely different mechanics this time.** Trajectory: outer 0=0.4227 -> 10=0.3104 -> 20=0.2802 -> 30=0.2708 -> 40=0.2674 -> 50=0.2648 -> 60=0.2635 -> 70=0.2633. Per-constraint rho hit the 1e8 cap on 4/5 constraints by outer 30 (qi, mirror_ratio, flux_compression, vacuum_well all failing the official "improved by >=50%" progress test almost every iteration once close to this point -- in practice not meaningfully gentler than unconditional growth once a constraint is genuinely stuck, just delayed by ~1 outer iteration per doubling). Unlike ngopt_v1/v2/omnig_v1's hard flat stalls, this one shows a **smooth, decaying, asymptotic crawl** rather than a dead stop -- a real qualitative difference in optimizer behavior from the mechanics fix -- but the asymptote itself (~0.263, converging toward but clearly not reaching 0) landed in the same place as `ngopt_omnig_v1`'s hard stall (0.2805) on the identical seed. Killed at outer 70 (`docker stop`) once outer 60->70 showed <0.001 movement.

**Updated consolidated conclusion**: the official ALM mechanics (per-constraint conditional rho, growing budget, explicit incumbent-suggestion) measurably change the *shape* of convergence (smooth asymptote vs. hard stall) but not the *destination* -- both land within 0.26-0.28 of full feasibility from the best seed found in the whole project. This is now 8 independent method/seed combinations, spanning optimizer internals (persistent NGOpt/CMA), starting-point diversity (5 real/constructed seeds), the official omnigenity construction recipe, and now the official ALM update rule itself, all converging to materially the same qualitative outcome. The one concretely untried lever from the official recipe is the infinity-norm spectral reparametrization of the search space itself (`compute_infinity_norm_spectrum_scaling_fun`) -- explicitly not attempted here, real engineering lift, uncertain payoff given how consistently everything else has landed in the same place. The other remaining direction (genuine multi-objective/Pareto search rather than scalarized ALM descent) is unlikely to add much beyond what `best_worst_violation` (already tracking the max of all 5 per-constraint violations every iteration across 8 runs) has already shown: the frontier this project can reach from real, physically-motivated starting points does not cross into all-5-simultaneously-feasible territory using local search of any tested kind.

**Direct user pushback, and it was correct: "people have gotten past it, so it should be possible."** Right -- the paper's own Table 5 reports 0.0 constraint violation for exactly this ALM-NGOpt recipe on P3. Went and fetched the actual published settings (arxiv:2506.19583 §A.4.1) rather than continuing to guess from the pip package's dataclass defaults, and found the real, quantitative, almost certainly-sufficient explanation for all 8 prior stalls: **the paper's NGOpt budget per outer iteration k is `min(20_000, 1500 + 300*k)`** -- starting at 1,500 real VMEC evaluations and growing to 20,000 per outer step, run on a 96-vCPU machine for 15-24 hours (order 10^6 total real evaluations for one ALM-NGOpt run). Every one of our 8 attempts used a per-outer budget of 40-140 -- **10-500x less compute per attempt than the paper's own winning recipe at the same outer-iteration index.** The paper's own reported settings also differ from the pip package's dataclass defaults in two more ways worth recording: `rho_i^0=10` (not 1), `penalty_parameters_increase_factor=5` (not 2), `constraint_violation_tolerance_reduction_factor=0.8` (not 0.5, i.e. LOOSER than the package default, growing rho less readily per-constraint), and trust-region bounds that genuinely DO shrink geometrically (`delta_0=0.33`, `gamma=0.95/outer`, floored at `delta_min=0.05`) -- confirming our own earlier "bounds_reduction_factor is vestigial" finding was correct for the *pip package's* generic runner specifically, but the paper's actual run clearly does shrink bounds somewhere, so this new script implements that shrink directly rather than relying on the (apparently unused-in-the-real-run) package mechanism.

Implemented as `scripts/ngopt_alm_p3_paper.py` (forked from `ngopt_alm_p3_official.py`): paper's exact budget schedule/rho/tolerance numbers as CLI defaults, monotonic bound-shrink (`bound_mult *= 0.95/outer`, floored) replacing reactive stall-widening (disabled by default here since the paper's recipe doesn't use it). Smoke-tested clean. Launched `paper_v1` from the same best-available seed as `official_v1` (`omnigenity_seed2_anchor.npy`), 100 outer iterations, 28 workers. **Estimated wall-clock: ~5-6 days** (back-of-envelope: budget schedule sums to ~1.42M real VMEC calls total across 100 outer iters at this schedule; ~10s/call / 28 workers ≈ 141 hours) -- by far the largest single compute commitment this project has made toward P3, but a direct, honest test of whether the wall really is compute-budget-limited (as the paper's own success implies) rather than a genuine structural dead end (as the previous 8-attempt "consolidated conclusion" section above suggested too hastily -- that conclusion is now understood to have been about search TECHNIQUE at a roughly fixed, badly under-scaled compute budget, not about the true reachability of P3 feasibility). Checkpointed every outer iteration, so resumable/killable without losing progress. Result to follow -- will report real intermediate numbers as they come in rather than waiting for completion, given the multi-day runtime.

---

## P3 was mis-specified for the entire project — corrected 2026-09-26

**`paper_v1` is stopped.** It ran 6 days / **680,300 real VMEC++ evals** and stalled at
`worst_violation=0.2036` (low fidelity). It was not stopped because ALM-NGOpt failed. It was
stopped because it was optimizing **the wrong problem**, and had been since the beginning.

### What P3 actually is (verified against the installed package, not recollection)

`constellaration.problems.MHDStableQIStellarator`:

* **Multi-objective**: minimize `(-minimum_normalized_magnetic_gradient_scale_length, aspect_ratio)`.
* **Scored by HYPERVOLUME of a SET of feasible designs** in that 2D space. `evaluate()` takes a
  **`list[SurfaceRZFourier]`** — you submit a Pareto front, not one design.
* Calls `ConstellarationSettings.default_high_fidelity()` **unconditionally**.

The paper (arXiv:2506.19583) decomposes it "into multiple single-objective problems by treating
**the aspect ratio as an inequality constraint**", A* ∈ {6, 8, 10, 12}, each "seeded from
**rotating ellipse** configurations", 15–24h per instance on 96 vCPU. Reported violations:
**{0.104, 0.00130, 0.0, 0.00210}** — three of four essentially feasible.

The leaderboard is real and solved: top entries ~0.974 hypervolume. arXiv:2608.28224 reached
**0.9763** by L-BFGS-refining the existing public top-5 and sampling the 2D MDS subspace those ten
solutions induce — and states its own limitation plainly: *"Our refinement method is local by
construction: it inherits the basin it starts in and **requires solutions worth refining**."*

### What we were doing instead

1. **No aspect-ratio constraint** — the search had a free direction to wander in.
2. **A single point, not a front** — even a perfectly feasible single design scores poorly on hypervolume.
3. **Low-fidelity search oracle** vs. a scorer hardcoded to high fidelity — and `fidelity_check.py`
   already measured that gap at **~0.017 on exactly the active constraints**.
4. **Wrong seed** — an omnigenity fit, not a rotating ellipse.

### Evidence the old basin was a dead end

Spread analysis over all 680k rows (sampled 1-in-20):

* Every candidate sat within **0.018** of the seed, against `||seed|| = 1.06` — a ball of radius
  **1.7%** of the design's own scale. Nothing ever left it.
* Only **3 of 90** coefficients had std > 0.001.
* **PCA effective dimension 2.4** (PC1 alone 64% of variance). 680k evals explored a ~2D slice of a
  90-dim space.
* Cloud center crept out monotonically (0.0024 → 0.0165 over outers 0→62) while spread about the
  center *shrank* — a fixed-size trust region (`bound_mult` floored at 9.0 since outer 36) dragging a
  rigid bubble downhill, not exploring.

Final incumbent violated **four of five** constraints: `qi` 0.00163 (≤0.000316, **5.2× over**, and the
binding one), mirror 0.2996 (≤0.25), flux_compression 1.0797 (≤0.9), vacuum_well −0.0168 (≥0); only
`iota/nfp` 0.2996 passed. This is consistent with the long-standing finding above that the low-`qi`
region and the region satisfying the other constraints barely overlap.

**Correction to an earlier claim in this session:** the active pair was reported as
mirror_ratio/flux_compression with "qi comfortably satisfied." That was an index-order misread —
`qi` was the binding constraint throughout.

### Corrective work (`scripts/ngopt_alm_p3_instance.py`, `scripts/rotating_ellipse_probe.py`)

`rotating_ellipse_probe.py` validated the paper's seeding recipe end-to-end over
A* × elongation × nfp. Findings:

* Round-trip through `initial_guess.generate_rotating_ellipse` +
  `smooth_and_set_max_mode_numbers(b, 4, 4)` into the 90-dim layout works; `aspect_ratio` measures
  **exactly** at the requested A*, so the sixth constraint is active-at-equality from outer 0.
* **nfp=2 dominates** — nfp 3/4 inflate flux_compression 3–15×.
* **A\* = 6 is fragile**: VMEC++/QI rejects every elongation except 1.5, and only at nfp=3
  ("Minimum of B is at the boundary").
* Chosen seeds: **A\*=6** (elong 1.5, nfp 3, worst 1.7314) · **A\*=8** (elong 2.0, nfp 2, **0.6013**)
  · **A\*=10** (elong 3.0, nfp 2, 0.6402) · **A\*=12** (elong 4.0, nfp 2, 1.2242).

These start **far worse** than paper_v1's 0.204 incumbent. That is the point.

**Trust region re-sized for exploration, not polishing.** The seeds have only **7 of 81** free
coefficients nonzero (mpol=1 shape zero-padded into mpol=4/ntor=4), so the search must grow modes up
from exactly zero. `sigma0` 0.0008 → **0.004**; `bound_mult` 60→9 becomes **25→5**, keeping the
paper's `delta_min/delta_0 = 0.05/0.33` ratio. Half-width therefore runs 0.100 → 0.020 versus
paper_v1's 0.048 → 0.0072.

**Hardware reality:** 32 cores vs the paper's 96 → roughly **45–70h per instance**, so the four run
**sequentially**, not in parallel. `a8` launched first (best seed) to validate the widened trust
region before committing the rest.

**Finalists must be re-validated at high fidelity** before any feasibility claim — the scorer uses it
unconditionally and it reads ~0.017 worse.

### Sequencing

A (reproduce the paper's protocol) is deliberately first, not as the destination but because both
alternatives need what it produces. Reparametrization (the landscape *is* badly conditioned — the 2.4
effective dimensions is evidence) is a multiplier on a working search and unmeasurable without a
feasible reference point. An RL / oracle-call-minimization framing needs a seed population of
successes and a reward that isn't near-universally infeasible; this project's prior learned-generative
attempt (VAE + latent diffusion) produced **0/160 converged, worse than the 29/480 baseline**, with
latents diverging to |z|≈2031. Per arXiv:2608.28224, the learned/latent layer works *on top of* a
population of good designs. We have none. A produces the first ones.

**Abandon-A criterion:** if all four instances stall at 0.1–0.2 the way `paper_v1` did — i.e. the
paper's own recipe does not reproduce here — then the problem is genuinely our landscape, and
reparametrization becomes the priority.

---

## The whole search program has been gradient-free, and that is the bottleneck — 2026-09-27

### Full inventory, measured

**981,285 designs with complete physics metrics**: 822,600 real VMEC++ evals we generated across 19
runs (96.7% converged) plus the 158,685 shipped dataset designs. **Zero are feasible**, so the
benchmark score has been exactly **0.0** for the project's entire history. Snapshot artifact:
`https://claude.ai/code/artifact/3fc82c16-bbda-49f7-85dd-a64e957b97f3`

### The finding that reframes everything

| run | evals | best official worst-violation |
|---|---|---|
| **ngopt_v1** | **11,200** | **0.0917** |
| ngopt_v2 | 14,000 | 0.0965 |
| ngopt_v3_seed14nfp4 | 6,000 | 0.1522 |
| **paper_v1** | **680,300** | **0.2036** |
| *(shipped dataset)* | *158,685* | *0.2031* |
| a8w / a6w / a10w (whitened) | 21.8k / 24.8k / 21.8k | 0.3283 / 0.3368 / 0.3444 |

**The project's best design came from an 11,200-eval run and beat the 680,300-eval run by 2.2x.**
More gradient-free compute stopped buying anything long ago. Only **36 designs out of 981,285** get
below 0.10, and every one comes from the early small runs.

An earlier claim in this session — "NGOpt never beat the dataset" — was **wrong**; it was based on
`paper_v1` plus the recent lanes only, missing `output/ngopt_alm_p3/`. `ngopt_v1`/`v2` beat the
dataset by ~2x.

### DATA LOSS: the best design is unrecoverable

`ngopt_v1_results.jsonl` rows contain only
`[converged, loss, measured_targets, objective, outer, violations, worst_violation]` — **no
`params`** — and it is the only one of 19 runs with **no checkpoint file**. The 0.0917 design's
boundary coefficients are gone; only its metrics survive. Best *recoverable* design is `ngopt_v2`'s
0.09653 (`ngopt_v2_checkpoint.json` has `x_best`). Every script written since stores `params` per row
plus a per-outer checkpoint.

### What blocks feasibility (113,669 sampled converged designs)

`qi` is the worst constraint in **52.4%** of designs, `edge_magnetic_mirror_ratio` in **44.1%** —
together **96.5%**. flux_compression 1.7%, vacuum_well 1.6%, iota 0.2%.

Best recoverable design's five violations, all within 0.10 of their bounds:
`qi +0.0917` (0.000662 vs 3.162e-4) · `mirror +0.0899` · `fluxcomp +0.0663` · `vwell +0.0248` ·
`iota −0.1501` (satisfied). A=8.287, L_gradB=3.666.

### Corrected dataset claim

The long-standing log claim that qi needs "1-1.5 orders of magnitude past anything in the real
dataset" is **wrong**. **169 of 158,685** dataset designs meet the qi bound (best 6.1e-5, 5x better
than required). The real obstacle is non-overlap: of those 169, **1** satisfies iota, **0** mirror,
5 flux_compression, 9 vacuum_well; their worst-of-the-other-four is min 0.41, median 0.80. Both
directions stall near 0.4 — we approach with qi bad and the rest fine, the dataset's qi-feasible
designs have the reverse. Saved best-first as `output/qi_feasible_seeds.npy`.

### Results that DO hold from the A/B work

**Whitening is real and measured.** Per-coefficient sigma and trust-region width scaled by each
coefficient's dataset std (540x dynamic range; 9 of 81 coefficients carry 90% of variance). Controlled
A/B on identical seed/A*/budget: `a8w` reached 0.4279 at outer 1, which flat `a8` needed until outer
3-4 to match — **~2.5x fewer outer iterations**, at the cost of 12-15% of evals to rejected geometries.
It preconditions a gradient step too, so it carries into L-BFGS. `output/coef_scales.npy`.

Everything else from that push was negative: the A* decomposition **leaks** (every instance drifted
outside its own box — A*=6 to 7.25, A*=8 to 10.38, A*=10 to 11.90 — because `qi` dominated the
worst-violation that `x_best` tracks), and all three whitened lanes stalled at 0.33-0.38 with qi
binding and rho in the millions.

### B2 (omnigenous-field parameterization) was assessed and rejected

`_generate_initial_guess(targets)` turns out to be **literally just a rotating ellipse** —
`make_rotating_ellipse` on `aspect_ratio`, `max_elongation`, `rotational_transform`, `n_field_periods`.
It ignores `x_lmn` and `modB_spline_knot_coefficients` entirely. The omnigenity information enters
*only* via `_refine_sample`, an inner NGOpt-against-VMEC fit at ~5 min/candidate that parallelizes
internally (~2 concurrent on 32 cores → order 24 candidates/hour vs 3,500/hour now). Nested
optimization at 150x the cost. Also note `quasi_isodynamicity_residual` requires a `BoozerOutput`, so
qi cannot be evaluated without solving VMEC — there is no cheap-qi shortcut anywhere.

An intermediate claim that B2 would cost one VMEC call per eval was **wrong and retracted** before
anything was built on it.

### Current direction: L-BFGS with FD gradients (`scripts/lbfgs_refine_p3.py`)

arXiv:2608.28224 topped the leaderboard at **0.9763** by replacing its evolutionary optimizer with
"local gradient based L-BFGS", gradients "in parallel using a forward, two-point finite-difference
stencil straight through the solver", starting from the existing public top-5 — and states the
limitation: the method "requires solutions worth refining". `output/lbfgs_seeds.npy` now holds 40
distinct such designs (worst 0.0965-0.16, deduped at 1e-5).

FD is sound here **because `fidelity_check.py` proved low-fidelity VMEC++ is bit-for-bit
deterministic** (two independent re-evals identical to 17 significant digits; single-threaded per
`VmecSettings.max_threads`). No noise floor. Cost: 81 free coefficients → 82 parallel evals per
gradient ≈ 30 s on 32 cores, so hundreds of iterations is **hours, not days**.

Phase 1 minimizes `sum_i relu(v_i)^2`; on the first feasible design phase 2 switches to maximizing
`(1 + L_gradB) * (20 - aspect_ratio)` — the actual scored quantity — with a stiff feasibility penalty.
Optimization runs in whitened coordinates. No A* box: aspect ratio is an objective in the real
problem, and the paper's boxes are its own decomposition device.

### Scoring, now implemented (`scripts/p3_front.py`)

`_score` = hypervolume of **feasible designs only** in (−L_gradB, aspect_ratio) vs reference
**(1.0, 20.0)**; `is_feasible` requires all five normalized violations ≤ **1e-2**;
`evaluate()` takes a **list** and calls `default_high_fidelity()` unconditionally. Single-point
contribution is `(1 + L_gradB) * (20 - A)`. **Density is worthless** — dominated points add exactly
zero; only genuine trade-offs count. `--validate` re-runs front members at high fidelity for an honest
number, since search runs low fidelity (~0.017 optimistic).

---

## First optimizer-driven improvement since ngopt_v2: minimax SQP — 2026-09-27 (later)

**NEW PROJECT BEST: worst_violation 0.09653 -> 0.092661** (low fidelity), **0.097115** at high
fidelity. Saved as `output/project_best.npy`. Still **score 0.0** — feasibility needs <= 0.01.

### Method: minimax SQP (`scripts/minimax_p3.py`)

Feasibility is a MAX condition, so scalarizing is wrong. Each iteration solves

    min t   s.t.  v_i + g_i . d <= t  (i=1..5),   ||d||_2 <= Delta

`p-norm L-BFGS` (p=8) stalled at 0.0965 after 13 gradients; sum-of-squares was worse still, driving
f down 2.6165e-2 -> 2.4602e-2 while worst went the WRONG way 0.09653 -> 0.11186.

**All five constraint gradients come free** from the same 82-eval stencil that previously produced
one scalar gradient — 5x the information at identical oracle cost.

**Proof the method is doing its job:** at the optimum the three active constraints are driven to
near-equality — qi +0.091519, mirror +0.092525, fluxcomp +0.092682 (spread 0.0012, from an initial
spread of 0.0094). Equalizing active constraints is precisely the signature of a true minimax
optimum.

### SIX configuration errors, all traceable to ONE unpropagated measurement

`fd_diagnostic.py` measured the objective's usable scale at **~1e-2 whitened units** (consecutive-
gradient cosine 0.91 between h=1e-2 and 5e-2; only 0.53 between 1e-3 and 1e-2; **negative (-0.14)**
between 1e-4 and 1e-3). Every part of a gradient method must respect that scale. They were fixed one
at a time as each surfaced instead of being derived together:

1. `fd_step=1e-3` — 20x too small, descending a half-random direction.
2. sum-of-squares surrogate — optimized the wrong quantity (see above).
3. 60s batch timeout — silently zeroed gradient components; the IDENTICAL deterministic base point
   gave fd_fail 0/81 and 7/81 in two runs. Fixed at 300s -> 0/81 thereafter.
4. Line-search ladder fixed at (1, .5, .25) — the LP saturates the trust region, so the smallest
   step ever tried was 0.25*delta, never reaching the valid scale. Now 9 geometric points down to
   fd_step/2.
5. **Trust region bounded in L-inf, not L2** — the LP drives all 81 coordinates to their bound, so
   `|d|inf=0.3` means `||d||_2=2.65`, i.e. **130x fd_step**. Measured: t_pred=-0.94 (absurd) and all
   large trials failed VMEC outright. Delta is now an L2 budget converted internally to the box.
6. **Trust region could only shrink.** The textbook `actual > 0.5*predicted` ratio test never fires
   when the linear model overpredicts by orders of magnitude, so delta ratcheted down and terminated
   AT THE FLOOR WHILE DESCENDING at 2e-4/iteration. Now keyed on where the winning step sat on the
   ladder — a direct observable that does not require a calibrated model.

Each of these initially looked like "the method doesn't work." None were.

### Seed diversity was an illusion

`output/lbfgs_seeds.npy` (top 40 by violation) had median pairwise whitened distance **2.08**, all
nfp=2, all A~8.43, all worst 0.0965-0.0997 — **40 samples of one basin**. Selecting on "best
violation" implicitly selects one basin. `output/diverse_seeds.npy` stratifies best-per-(nfp x
aspect-ratio bucket) across our runs, checkpoints AND the dataset: 23 seeds, median distance
**15.50**, nfp 1-5, A 6.97-14.96.

### Corrected: the high-fidelity penalty is smaller than assumed

> **SUPERSEDED 2026-10-05 — do not cite this.** n=1, conditioned on the design
> having converged, and measuring the wrong statistic. The paired n=22 measurement
> (see "How much the cheap oracle lies" at the end of this file) gives |shift|
> median 0.00954 and **max 0.05363 = 5.4× the feasibility bar**, with a signed
> median of −0.0007. The error is variance, not a "penalty": it cannot be carried
> as a margin and cannot be calibrated away. The L_gradB improvement noted below
> is real and general (17/22 designs, up to +194%) but it means our *objective*
> was also mis-measured, not that high fidelity is benign.

Measured on the new best: low 0.092682 -> high 0.097115, a gap of **0.0044**, not the ~0.017
previously carried. L_gradB also *improves* at high fidelity (3.979 -> 4.697), raising the
hypothetical hv contribution 57.6 -> 65.9.

### Honest assessment of the ceiling

The gradient method extracts roughly **4%** from a basin before the trust region legitimately
collapses. Reaching feasibility from 0.093 needs a **90%** reduction. If all 23 diverse seeds yield
similar ~4% gains, the best lands near 0.089 and **none becomes feasible**. The 23-seed sweep is
running because it is cheap and one basin may behave differently, but it is not expected to produce
a feasible design. Local refinement is not the tool that closes this gap.

---

## Diagnostics: ~50,550 evals were being discarded (fixed) — 2026-09-28

**Caught by the user asking whether trajectories were being retained.** They were not.
`minimax_p3.py` as originally written persisted only a per-iteration trace summary and a
single best-design `.npy`. Every individual evaluation — 91 per iteration, ~50,550 across 22
runs — was thrown away.

This repeats the `ngopt_v1` data loss documented earlier the same day (that run stored no
`params`, which is why the project's all-time best design at 0.0917 is permanently
unrecoverable). The script was written **after** that entry was logged. Persisting the
optimizer's path while discarding the boundaries it evaluated is exactly backwards: the
trajectory is incidental, the designs plus their physics are the durable asset.

**Now persisted per run:**

* `<tag>_results.jsonl` — every evaluation: 90-dim `params`, all eleven `measured_targets`,
  the five `violations`, `worst_violation`, `hv_contribution`, `nfp`, plus `iter` and `kind`
  (`stencil` | `linesearch`). Non-converged rows retained.
* `<tag>_jacobians.npz` — the **5 x 81 sensitivity matrix per iterate**,
  d(violation_i)/d(coefficient_j) in whitened units, with `z`, `v`, `free_idx`, `scales`,
  `nfp`. Previously discarded entirely, and the most reusable by-product of the whole
  exercise: usable for surrogate fitting, conditioning analysis, and determining which
  coefficients actually move which constraint — independent of any optimizer.

Verified on restart: 82/82 rows written with all fields on the first stencil batch.

## Did we break the Pareto wall? No — we walked down it

**We did not break through or sidestep it in any way that helps the score.**

1. **Active-constraint equalization IS the wall.** At every converged point the minimax solver
   drives the active set to equality — qi +0.091519, mirror +0.092525, fluxcomp +0.092682,
   spread 0.0012 from an initial 0.0094. That is the local Pareto frontier of the constraint
   trade-off by definition: no direction improves the worst without another rising to meet it.
   Reaching it validates the solver AND confirms we are standing on the frontier.

2. **The gains are descent ALONG the frontier**, 0.09653 -> 0.072328, not through it.

3. **Part of the gain is a sidestep that costs score.** Aspect ratio is correctly unconstrained
   (it is objective #2), so the optimizer raised it to make the physics constraints easier:
   d4 A=11.60 (hv potential 42.8), d1 A=16.11 (hv 29.7), versus the A~8.43 seed (hv 56.9).
   d1 bought a 51% violation reduction by nearly doubling aspect ratio, halving what the design
   would be worth if it ever became feasible. Screening seeds are now ranked partly on
   (1+L_gradB)*(20-A) rather than violation alone.

4. **The underlying non-overlap is untouched**: 169 of 158,685 dataset designs meet the qi
   bound, **zero** satisfy the other four. Nothing produced so far is evidence those regions
   intersect.

**The non-uniform part is the remaining hope.** Basin reductions range 4% (d0) to 69% (d4), and
screen seed 1 dropped 65.6% in 14 iterations. The frontier height varies by basin; if some
basin's local frontier sits near 0.02 rather than 0.07, feasibility is reachable there. That is
what the 54-basin screen tests.

## Two claims I adopted and then had to retract — 2026-09-30

Both were stated confidently in this log or in the artifact and both are wrong. Recording them
because the pattern is identical: a single run, a plausible mechanism, no replication.

### 1. The hypervolume-floor descent rule DID NOT REPLICATE

**The claim.** Adding a hard floor on measured hypervolume contribution to the minimax step
(`--hv-mode floor`) dominates the unconstrained rule on both axes. Basis: one controlled A/B on
screen basin s32 — `none` converged at 0.050484 (hv 29.6), `floor` reached 0.032098 (hv 33.3)
and was still descending at the iteration cap. 36% better violation and 12% better hv. I made
it the default.

**The replication.** Paired arms on 6 screened basins spanning L∇B 2.94–9.92, identical seed,
budget, δ schedule and worker count, arms run back-to-back on the same machine:

| basin | L∇B₀ | none | floor | floor vs none | hv none | hv floor |
|---|---|---|---|---|---|---|
| 32 | 2.94 | 0.050484 | 0.032098 | **−36.4%** | 29.6 | 33.3 |
| 13 | 5.25 | 0.097839 | 0.100747 | +3.0% | 71.0 | 71.1 |
| 21 | 6.88 | 0.083593 | 0.084535 | +1.1% | 68.4 | 69.2 |
| 37 | 7.27 | 0.086448 | 0.086448 | 0.0% | 68.0 | 68.0 |
| 4 | 7.70 | 0.214788 | 0.209144 | −2.6% | 92.2 | 92.7 |
| 9 | 9.92 | 0.086895 | 0.092689 | +6.7% | 88.9 | 90.5 |

**2 wins, 1 tie, 3 losses, median effect slightly negative.** s32's −36.4% is still unexplained
and is now the outlier, not the rule.

**What survives** is much smaller and worth keeping: the floor protects a little hv *reliably*
(ahead in 5/6 basins, +0.1% to +12%) at a violation cost of 0–6.7%. A small symmetric trade-off,
not free upside. Default reverted to `none`; the table is in `minimax_p3.py --help` so the
reverted conclusion cannot quietly return.

**The mechanism I offered was also wrong.** I explained the floor as preventing an hv collapse,
citing −52%. That figure came from `deep32`, a different run. This experiment's own control arm
(`hv_none`) degraded only −8.7%, in line with everything else. I had conflated two runs to
manufacture a mechanism for an effect that turned out not to exist.

**Also retracted:** `hv-mode weighted` stalls outright, and the reason is a units error, not
physics — `t` lives on the violation scale (~0.07) while hv is ~32, so any weight large enough
to matter lets the hv term dominate the minimax objective entirely.

### 2. "Starting violation barely predicts final violation" — WRONG, corr = +0.73

I generalized this from two memorable outliers (d0 started best and moved least; d4 started at
0.2324 and finished best). Across the full 54-basin screen the correlation between starting and
final worst-violation is **+0.73**. Starting point is the dominant predictor, as one would
expect. Diverse seeding still helped, but through the *residual* variance around a strong trend,
not because the trend was absent. The claim as written appears in `run_screen.sh`'s header
comment and overstates the case there too.

---

## Current frontier problem: the two things we want are in different designs — 2026-10-01

Ranking all 71 refinement tags that have retained metrics makes the real obstacle explicit.
Score is hypervolume of *feasible* designs only, single-point contribution (1+L∇B)(20−A):

| | best worst-violation | hv if feasible |
|---|---|---|
| near-feasible leader `floor2` | **0.03127** | 33.2 |
| best high-hv candidate `rp9_none` | 0.08408 | **89.6** |
| `s23` | 0.35701 | 99.5 |
| `s29` | 0.17547 | 92.8 |

Everything close to feasible is worth little; everything worth a lot is far from feasible. The
leader bought much of its descent by raising A to 12.885, which is precisely the axis the score
penalises — so its own success mechanism caps its payoff.

`floor2` is a dead end for depth: `floor2` and `floor3` started from the same point with 6×
different δ₀ and both stalled at exactly 0.031 with three constraints tied. The earlier d4 ladder
likewise returned 0.072328 bit-identically on re-run. `rp9_none` is the best joint point and has
had only 30 iterations.

**Running now** (`run_hv_ladder.sh`): 3-round restart ladder × 70 iterations on `rp9_none`,
`rp21_none`, `s29`, `rp4_none` with `--hv-mode none` (floor cost rp9 6.7%). rp9 needs an 8.4×
violation reduction versus the leader's 3.1×, but the leader's gradient is provably zero and
rp9's is not. Observed per-basin reductions were 6–71% (median 30%), so neither is comfortable.

## The exhaustion rule never fires, and the real lever was the unconstrained objective — 2026-10-04

### Exhaustion: 15 rounds, strictly monotone, no termination

The restart-until-bit-identical protocol was run on the leader (`floor2`) for fifteen rounds,
each re-seeded from the best EVAL of the previous round (not its `_best.npy`, which records a
worse design in 62 of 72 runs). Every single round returned a strictly better design:

| round | best worst-violation | round | best worst-violation |
|---|---|---|---|
| 1 | 0.030701014326020522 | 9  | 0.029408305874562668 |
| 2 | 0.030545782491734785 | 10 | 0.029343259158415957 |
| 3 | 0.030480247526839532 | 11 | 0.029053937948113300 |
| 4 | 0.030304771598560393 | 12 | 0.028936433692405840 |
| 5 | 0.029731429675224370 | 13 | 0.028771499796710058 |
| 6 | 0.029622197995104190 | 14 | 0.028653277596992766 |
| 7 | 0.029563368164009445 | 15 | 0.027873 (in flight) |
| 8 | 0.029443631784031420 |    |  |

So the stopping rule I designed **does not stop**. It fired exactly twice in this project, on
`rp9_none` and `rp21_none`, and both of those were being re-seeded from the wrong point — with
harvested seeding `rp9_none` went 0.085362 -> 0.082744, i.e. progress was available and the
"exhausted" verdict on it is RETRACTED. (`rp21_none` and `s29` held up; harvested seeding is NOT
a universal win and must not be reported as one.) `s32` behaved the same way: six rounds, strictly
monotone, -7.0e-5/round, heading for ~0.048. Killed as hopeless for feasibility.

Rate on floor2 is -1.3e-4/round at ~165 min/round, so the bar is ~140 more rounds ~ 16 days of
the whole box for ONE design. **I tried to fit an asymptote and it is not identifiable.** Fitting
w_inf + C exp(-r/tau) on prefixes gives w_inf = 0.0235 (8 rounds), 0.0285 (10), 0.0278 (12),
0.0269 (14) with tau swinging 32.7 -> 8.2 -> 17.3 and residual 1.1e-4, which is the size of one
round's step. The estimate is dragged by each new round instead of predicting it. **No claim
either way about whether floor2 has a floor above 1e-2.**

### Two measurement errors caught before they became claims

1. A first pass reported **91 feasible designs**. Recomputing all five official violations
   directly from `measured_targets` across 991,855 evals gives **0**. The 91 were P1/P2 rows whose
   stored `worst_violation` field is computed against a different constraint set. Never trust that
   field across lanes; recompute. (Conventions verified against the installed scorer source:
   `(0.25-iota)/0.25`, `(log10(qi)+3.5)/3.5`, `(mirror-0.25)/0.25`, `(fluxc-0.9)/0.9`, `-vw/1e-1`.)
2. The qi gap is **log-normalized**. Reading 0.0287 as "2.9% away" understates it 7x: it means
   qi residual must fall from 3.98e-4 to 3.43e-4, i.e. **-13.8%**.

Actual remaining work at the leader: qi **-13.8%**, edge mirror ratio **-1.8%**, flux compression
-0.12%; iota and vacuum well satisfied with 13.6% and 9% of unused margin.

### The lever: aspect ratio is an OBJECTIVE, so the minimax never controlled it

P3 minimizes `(-L_gradB, aspect_ratio)` subject to five constraints. Aspect ratio is therefore
never in the constraint set, and `minimax_p3.py` lets A float wherever descent leaves it. Result:
**all 991,855 of our evals with worst<=0.05 sit in A in [12.17, 13.10]**. The high-A region is
unexplored by construction, not by evidence.

Within that same neighbourhood (n=36,152 evals, worst<=0.05):

```
spearman(A, qi violation)     = -0.669     spearman(A, iota)  = +0.472
spearman(A, mirror violation) = -0.091     spearman(A, fluxc) = +0.546
spearman(A, WORST violation)  = -0.650     spearman(A, vwell) = +0.708
```

Raising A buys down exactly the two constraints that bind and charges it to the three with unused
margin. Same sign at every quality cut (worst<=1.0: -0.641; <=0.2: -0.421; <=0.1: -0.662).

Added `--aspect-min A` to `minimax_p3.py`: a SIXTH pseudo-constraint `(A_min - A)/A_min` in the
LP/Jacobian/candidate-selection only. `official_violations()` is left untouched and every
persisted `worst_violation` stays the scorer's 5-vector, because ~1M rows on disk and every
harvest in this project read that field. A unit test of the 6-row LP caught a hardcoded `5` in
the constraint-matrix assembly that would have silently dropped the new row.

### One iteration changed the structure of the problem

`asp_smoke`, seeded from the leader, `--aspect-min 14`:

```
it=0  A=12.894  official=0.028653   iota -0.1360  qi +0.0287  mirror +0.0286  fluxc +0.0113  vwell -0.0899
it=1  A=13.074  official=0.031897   iota -0.1465  qi +0.0319  mirror +0.0096  fluxc +0.0069  vwell -0.0887
```

A +0.18 in aspect ratio put **mirror and flux compression both under the 1e-2 bar**. Four of five
official constraints satisfied, and the three-way conflicting active set that the minimax was
built to handle collapsed to a **single binding constraint, qi**. The move on mirror (0.0286 ->
0.0096) is far larger than its -0.091 correlation predicted.

**Caveat, stated up front: the official worst got WORSE at it=1 (0.028653 -> 0.031897) and qi
itself got worse.** The bet is that qi as a lone binding constraint is tractable in a way a
three-way conflict is not; that is NOT yet demonstrated. One iteration is one iteration.

`run_aspect_ladder.sh` now maps achievable official worst against imposed aspect floors
13.5 / 14 / 15 / 16, with the read-out fixed in advance, to find the LOWEST A admitting a feasible
design. The trade is correct while the feasible set is empty — the scorer returns exactly 0.0 then,
so any feasible design beats the leader, and A can be walked back down from a feasible point
afterwards. It is bounded: A>=20 contributes zero hypervolume.

Score remains **0.0**. Zero feasible designs in 991,855 evaluated.

---

## THE SEARCH HAS BEEN MINING A FIDELITY GAP — 2026-10-05

**This is the most consequential negative result in the project and it invalidates
standing conclusions, not just today's run.** Read this before trusting any
low-fidelity number anywhere above.

### What happened

The aspect-floor lever (previous section) produced the project's first designs
satisfying all five official constraints — `asp_a13.5`, official worst 0.006549,
A=13.5548, 104 feasible evals, a 5-point front. At **low fidelity**. Verified
against the installed scorer, `MHDStableQIStellarator.evaluate()`:

| design | low-fid official worst | high fidelity |
|---|---|---|
| `asp_a13.5` best | 0.006549 | **VMEC++ did not converge** |
| `asp_a13.5` #2 | 0.007144 | **VMEC++ did not converge** |
| `asp_a13.5` #3 | 0.007602 | **VMEC++ did not converge** |
| `ex_floor2_r21` (1M-eval leader) | 0.026563 | **VMEC++ did not converge** |

Not infeasible. **Unresolvable.** The scorer cannot evaluate these boundaries, and
a design it cannot evaluate contributes nothing. Score remains **0.0**.

### The harness was exonerated first, and that is the only reason this is a result

Three alternative explanations were killed before concluding anything:

1. **Batch abort.** The first attempt passed 8 candidates to one
   `problem.evaluate()` call, which loops internally and propagates the first
   non-convergence — one bad boundary destroyed the batch and produced no data.
   `verify_feasible_official.py` now evaluates one design at a time and records
   non-convergence as an outcome. *My bug, not physics.*
2. **Wrong settings path.** Hypothesis: the scorer's
   `ConstellarationSettings.default_high_fidelity()` is harsher than the oracle's
   `VmecPresetSettings(fidelity="high_fidelity")`, and the project's one
   successful high-fidelity measurement (2026-09-27, low 0.092682 → high 0.097115)
   went through the gentler path. **Checked field-for-field: 0 differing fields of
   12.** They are the same settings. That 2026-09-27 success was simply a
   different design that happened to resolve, and it never licensed the general
   inference drawn from it.
3. **Broken harness.** Four ConStellaration dataset designs through the identical
   `problem.evaluate()` call: **4/4 converged** (scores 0.0, feasibility
   0.86–2.30 — correctly infeasible). `check_highfid_harness.py`. The call works;
   our boundaries are the problem.

### Mechanism, and why it indicts the method rather than the luck

`minimax_p3.py` optimises **81 free Fourier coefficients with nothing penalising
high-order content or resolution sensitivity**. ~1M evaluations of hard descent
against a coarse solver is precisely how one obtains boundaries that exist only at
coarse resolution. This is mesh dependence, and the ordering is the tell: the
*three best* low-fidelity designs all fail, and so does the leader. **The better a
design looks at low fidelity, the more suspect it now is.**

### The reasoning error, stated plainly

Low-fidelity search was justified on `fidelity_check.py`: VMEC++ at `low_fidelity`
is bit-for-bit deterministic to 17 digits, and the high-fidelity metric *penalty*
is small (0.0044 on the then-best design). Both facts are true. **Neither is the
relevant property.** Determinism makes finite-difference gradients sound; it says
nothing about whether the scorer can evaluate the result. A small metric penalty
was measured on designs that *converged*, so it is conditioned on exactly the event
that fails here — a survivorship artifact. "Low fidelity is reliable" and
"low-fidelity optima are admissible" were treated as one claim. They are two, and
only the first was ever tested.

The aspect ladder's "first feasible design" is that error cashing out. The lever
did exactly what it was built to do; the thing it optimised was not the benchmark.

### Running now

`survey_highfid_convergence.py` — high-fidelity convergence RATE across the
inventory, stratified by aspect ratio × low-fidelity worst violation, sampled
first-per-cell rather than best-per-cell (best-per-cell would bias the survey
toward the over-optimised designs whose resolvability is the question). Parallel
via the oracle, which is legitimate *because* of the 0-of-12 settings check above.

Pre-registered read-out:
- **flat rate** → the whole inventory is unusable and the oracle must change
- **rate falls as worst-violation falls** → the search itself destroys
  resolvability; every future run needs a convergence guard, not a better lever
- **rate falls with aspect ratio** → the aspect lever is specifically harmful and
  the sixth constraint needs a resolvability companion

### The fix is viable: high fidelity IS bit-for-bit deterministic

`fidelity_check.py` — the basis of every gradient run in this project — issues
`low_a`, `low_b`, `high`: **two** low-fidelity re-evals and exactly **one**
high-fidelity eval. So the 17-digit determinism result that justifies finite
differences was never tested at the fidelity that scores, and one evaluation
cannot test determinism. `fidelity_check_high.py` closes that gap, on dataset
control boundaries (ours cannot be used — they do not converge at high fidelity,
which is the finding above; determinism is a property of the solver, not the
design):

| control | determinism | high | low | ratio |
|---|---|---|---|---|
| d0 (nfp=5) | **BIT-IDENTICAL**, all 7 metrics | 442.2 s | 35.8 s | 12.4× |
| d1 (nfp=2) | **BIT-IDENTICAL**, all 7 metrics | 403.9 s | 82.2 s | 4.9× |

So FD gradients are sound at high fidelity: no noise floor, no lower bound forced
on the step size `h`, and `--fidelity high` (already an argument of
`minimax_p3.py`, no new code) is a genuine drop-in rather than a trap.

**Cost caveat, stated because the numbers invite over-reading.** Those six evals
ran concurrently on a box at load ~40 and the figures are batch wall-time, so they
are upper bounds on single-eval latency and the 12.4×/4.9× spread is contention
noise rather than a measurement of the ratio. Trustworthy claim: **roughly
5–12×**, ~400 s/eval under load. At 82 evals per gradient that is ~20 min per
iteration on 32 workers, so a 40-iteration run is most of a day — one honest run
per day, versus ~1M cheap evals that produced nothing scoreable.

**The guard is intrinsic.** A boundary the scorer cannot resolve returns a failed
eval, which `minimax_p3.py` already persists as `converged: false` and excludes
from candidates and from `best`. Searching at high fidelity makes the artifact
unreachable by construction — no penalty term, no heuristic.

**New prerequisite for seed selection.** A seed is unusable unless it converges at
high fidelity, and the project's own leaders do not. `eval_seed_pool.py` gained
`--fidelity` for exactly this; it is a gate now, not a diagnostic.

### How much the cheap oracle lies, measured properly (n=22 paired)

The project carried ONE data point on this — 2026-09-27, low 0.092682 → high
0.097115, gap 0.0044, logged as *"the high-fidelity penalty is smaller than
assumed."* That is n=1, it is conditioned on the design having converged, and it
measures the wrong statistic. `eval_seed_pool.py` was run over the 23 diverse
seeds at **both** fidelities (`--fidelity` added for this) and compared paired by
`compare_fidelity_gap.py`. **22/23 converge at high fidelity** (row 20 does not).

| quantity | result |
|---|---|
| worst-violation \|shift\| | median **0.00954 (1.0× the bar)**, p90 0.02844 (2.8×), max **0.05363 (5.4×)** |
| signed median shift | **−0.00068** — essentially zero |
| aspect ratio shift | **0.0000 exactly** (geometric, fidelity-invariant) |
| binding constraint flips | **1/22** (row 21: qi → iota) |
| L∇B (the objective) | understated in **17/22**, median **+9.5%**, max **+193.9%** |

**The error is VARIANCE, not bias.** A near-zero signed median with a 5.4× tail
means the cheap oracle cannot be calibrated — no offset, no scale factor, no
learned correction repairs it. Low fidelity **ranks** designs usefully and
**certifies nothing**: typical error ≈ the entire feasibility margin.

**The active set moves.** Where the binding constraint differs between fidelities,
the low-fidelity gradient aims at the wrong constraint. The minimax solves
`min t s.t. v_i + g_i·d <= t` and is *defined* by its active set, so this is not a
magnitude error — a high-fidelity gate on acceptance can reject a bad design but
cannot repair a wrong descent direction. **This is what makes `--fidelity high`
mandatory rather than a cost/benefit trade.** It also means the aspect lever's
headline mechanism ("collapsed a three-way active set to qi alone") was mechanism
on an active set that may not be the real one.

**The objective was wrong too.** L∇B understated in 17/22, up to 2.9× (row 4:
2.254 → 6.623). Every "hv if feasible" figure in this log is built on
low-fidelity L∇B, including the 2026-10-01 ranking that made `rp9_none` the best
high-hv candidate at 89.6. Those rankings are not trustworthy. Note the sign: real
hv potential is *better* than recorded, even as feasibility is worse.

**A pre-registered read-out that was wrong, and the correction.**
`compare_fidelity_gap.py` originally branched on *median* |shift| vs 1e-2. The
data came back at 0.00954 — **0.95×, a rounding margin** — and it printed "low
fidelity stays usable." Wrong statistic: certifying `worst <= 1e-2` is a claim
about the worst case, so the tail is what must be compared against the bar, and
the tail is 5.4×. The script now tests the tail, reports median/p90/max, and says
explicitly that variance cannot be corrected away. Recording this because
pre-registering a read-out does not help if the registered statistic is the wrong
one.

**Two limits held to.** (1) Survivorship: conditioned on high-fidelity
convergence, so it says nothing about designs that fail outright. (2) Every design
here sits at worst 0.097–0.365, far from the bar. **How the shift behaves AT the
bar is unmeasured, and unmeasurable from our optimised designs because those do
not converge.** No extrapolation to the final approach.

### THE SURVEY: descent is the mechanism, and the frontier is ~0.1

114 designs from `minimax_p3` (228,035 rows scanned), stratified by aspect ratio ×
low-fidelity worst violation, sampled **first-per-cell** not best-per-cell —
best-per-cell would have biased the survey toward exactly the over-optimised
designs whose resolvability is the question. Evaluated at high fidelity through
the oracle, which is equivalent to the scorer by the 0-of-12 settings check.

```
A \ worst      <0.01   0.01-0.05   0.05-0.1   0.1-0.2   >0.2
  0-8            -         -           -        5/6      6/6
  8-10           -         -          0/6       6/6      6/6
 10-12           -         -          1/6       6/6      6/6
 12-13           -        0/6         0/6       2/6      6/6
 13-14          0/6       0/6         0/6       3/6      6/6
 14-100          -         -          6/6       6/6       -
overall 65/114 = 57.0%
```

**The dependence is on the violation axis, not aspect ratio.** Monotone:

| low-fid worst | high-fidelity convergence |
|---|---|
| > 0.2 | **6/6 everywhere — 100%** |
| 0.1–0.2 | mostly 6/6, degrading to 2/6–3/6 at A 12–14 |
| 0.05–0.1 | 7/30 |
| **< 0.05** | **0/18 — zero** |

This is the pre-registered outcome *"rate falls as worst-violation falls → the
search itself destroys resolvability."* **Descent is the mechanism, not the aspect
lever** — aspect ratio is barely a factor. The better a design looks at low
fidelity, the less likely it exists at the fidelity that scores.

**The project's honest frontier is ≈ 0.10**, which is where diverse-pool row 0 sits
(0.10082) and approximately where things stood on 2026-09-27. Everything below
that was the optimiser walking into the unresolvable region — reliably, because
that is precisely where the cheap solver's errors are largest and most
exploitable.

**A binning artifact I reported and then had to correct.** The cell A≥14 /
worst 0.05–0.1 read 6/6, which I flagged as "the only place well-optimised designs
resolve" and went looking for a mechanism. All six designs sit at
**0.0990–0.0999** — pressed against the top edge of the bin, i.e. 0.1 designs that
the 0.10 edge filed as better than 0.1. No high-aspect exception exists; the
pattern is cleaner than first reported. `W_BINS` is now
`[0, .01, .03, .05, .075, .10, .15, .20, inf]` so a bin boundary cannot fake a
feature again.

**Mechanism attribution, from the lane composition:**

```
FAILED:    asp_a13.5 28, deep32 12, rep_s0_floor 6, asp_a15.0 1, asp_a16.0 1, s31 1
CONVERGED (worst<0.2): hl_s29_r1 15, hl_rp21_none_r1 6, hl_rp4_none_r3 6, s36 5, ...
```

Every failure is a **refinement** lane. Everything surviving near the frontier is a
shallower run sitting at ~0.1. Depth of optimisation is the discriminant, which is
the cleanest possible confirmation that the gradient method was mining the gap.

### Baseline non-convergence is ~1 in 20, not zero

Row 20 of the untouched diverse pool fails at high fidelity. So non-convergence is
not purely manufactured by our optimiser — but the contrast stands: **~5% of
untouched diverse designs vs 100% of our optimised leaders** (4/4 tested,
including 3/3 of `asp_a13.5`'s best). Optimisation pressure did not invent the
failure mode; it drove us into it reliably. A high-fidelity run should therefore
expect occasional failed evals as normal rather than as a signal of breakage.

### `scripts/run_highfid.sh`, and the 300-second trap

`minimax_p3.py --timeout-seconds` defaults to **300**. High-fidelity evals measured
**442.2 s and 403.9 s**. Running `--fidelity high` on the default would have timed
out *every* eval, recorded all of them as `converged: false`, and produced a
confident, entirely false "high fidelity doesn't work" result **with no error
raised anywhere**. `run_highfid.sh` sets 1800 s and hard-REFUSES any value below
900 s (tested). The aspect floor is **off by default** there: it produced the
low-fidelity artifact and is unvalidated at high fidelity.

**Running now:** `hf_s0` — high fidelity, 30 iterations, 16 workers, seeded from
diverse-pool row 0, the best design that actually resolves:
**worst = 0.10082** (A=8.4298, L=4.693, binding mirror). That is approximately
where the project stood on 2026-09-27. The ~1M intervening evaluations bought a
number inside the fidelity gap, not physics.

### `hf_s0b`: high-fidelity search is stable, the grind is hopeless, the lever's premise is real — 2026-10-06

First full high-fidelity minimax run. Seeded from `hf_s0`'s iteration-0 stencil
best (the only resolvable starting point available, since every previous project
leader fails at high fidelity).

```
1820 evals, 100% CONVERGED            <- not one failure in 1820
best worst 0.080991 (iter 19)          0 feasible
17.0 h / 20 iterations = ~51 min/iter  (32 workers)
trajectory 0.094011 -> 0.080991 (-13.8%), STILL DESCENDING at the cap
plateaus at iters 3-5, 12-13, 16-18, each broken by a later step
violations: iota -0.19563  qi +0.08099  mirror +0.07772  fluxcomp +0.07676  vwell -0.12406
```

**1. The resolvability guard is intrinsic and it works.** 1820/1820 converged.
Starting from a resolvable point the search never left the resolvable region. This
settles yesterday's open question — low-fidelity descent *drove us out*; the
resolvable region is not empty. Compare 81/82 on the first stencil from the
unoptimised seed, and 0/18 for low-fidelity designs below worst 0.05.

**2. Pure high-fidelity grinding does NOT reach feasibility.** −13.8% per 20
iterations and decelerating. From 0.081 to the 1e-2 bar needs ~87.7% more, i.e.
~280 further iterations ≈ **10 days of the whole box** — and that assumes the rate
holds, which the plateaus and the project's own history (90% of a basin's gain by
iteration 19–27) say it will not. A structural lever is required, not more compute.

**3. THE ACTIVE SET REPRODUCES AT HIGH FIDELITY.** qi/mirror/fluxcomp tied within
0.004 of each other, iota and vwell carrying 0.196 and 0.124 of unused slack. That
is the *same* structure as the old low-fidelity leader. So the aspect-floor
premise — aspect ratio is an unconstrained OBJECTIVE, and raising it buys margin on
exactly those three constraints while iota/vwell absorb the cost — **is valid at
the fidelity that scores.** The lever was previously demonstrated at a fidelity
where its result was fiction; its mechanism was never shown to be fiction.

Note also `best_worst` 0.080991 (true best eval) vs the reported accepted
0.082054 — the stencil-beats-accepted-step pattern holds at high fidelity too
(62 of 72 low-fidelity runs showed it).

**Running: `hf_asp13`** — `--aspect-min 13.5` at high fidelity, 24 iterations,
seeded from `hf_s0b`'s best. This is the direct test of whether the aspect lever
was *pure artifact* or *real mechanism measured at the wrong fidelity*. 13.5 is
deliberately the value that produced the low-fidelity breakthrough, so the
comparison is like-for-like. At low fidelity one iteration of it moved mirror
0.0286 → 0.0096 and fluxcomp 0.0113 → 0.0069; if anything of that order survives
at high fidelity it is worth far more than 280 grinding iterations.

### The fidelity ladder has no usable middle rung

`eval_seed_pool.py --fidelity medium` (`from_boundary_resolution`) over the same 23
seeds, compared paired against high:

| rung vs high | median \|shift\| | p90 | max | active-set flips | cost |
|---|---|---|---|---|---|
| low | 0.00954 (1.0×) | 0.02844 | 0.05363 (5.4×) | 1/22 | <70 core-s/eval |
| **medium** | **0.00813 (0.8×)** | 0.02699 | **0.05450 (5.4×)** | **2/22** | ~96 core-s/eval |
| high | — | — | — | — | ~614 core-s/eval |

**Medium is strictly dominated.** 15% better than low on the median, *identical* in
the tail, *worse* on the active set, and slightly more expensive than low. Its
handoff threshold computes to 0.0813 against low's 0.095 — the same place. Whatever
`from_boundary_resolution` refines, it is not what makes low fidelity wrong.

**Consequence for scheduling.** There is no gradual fidelity ramp available; it is
a two-rung cliff. Low fidelity reaches worst ≈0.095 in minutes, and *everything*
below that costs full price. The ladder therefore optimises only the cheap ~5% of
the journey, which is why per-iteration economics matters more than the schedule.

**Handoff rule, now encoded in `compare_fidelity_gap.py`:** step up when
`worst < k × (this rung's median error)`, k=10. For low that gives ~0.095, which is
independently where the resolvability survey shows convergence collapsing — two
unrelated measurements choosing the same threshold. **k=10 is a judgement, not a
measurement**; the errors are measured, the comfort multiple is not. At k=5 the
handoff would move to ~0.048, already inside the region where low-fidelity designs
stop existing, so k=10 is at least consistent with the independent evidence.

### Dimensionality is the available wall-time lever (measured, untested)

`analyze_jacobian_sparsity.py` reconstructs the real high-fidelity Jacobian from
`hf_s0b`'s iteration-0 stencil — 82 evaluations already paid for, no new compute.
**90% of the squared gradient norm sits in the top 29 of 81 coordinates.**

```
keep 81 -> 82 evals -> 3 waves on 32 workers -> 100.0% of gradient norm
keep 63 -> 64 evals -> 2 waves               ->  99.3%
keep 48 -> 49 evals -> 2 waves               ->  97.1%
keep 31 -> 32 evals -> 1 wave                ->  91.2%
```

Waves, not eval counts, are the wall-time unit: 82 evals on 32 workers is three
waves with the third carrying only 18 jobs. **63 coordinates costs 0.7% of the
gradient and removes a third of every iteration.**

**A hypothesis of mine that this refuted.** I proposed dropping the high-order
`r_cos m=3,4` modes on the grounds that they are where stencil evals fail (the
boundary becomes too sharp for VMEC) and therefore probably contribute little.
Wrong — three of the top six coordinates are exactly those modes: param 41
(r_cos m=4 n=+1) |col|=116.7 is #2, param 84 (z_sin m=4 n=−1) is #3, param 29
(r_cos m=3 n=−2) is #6. **The failure-prone directions are the most important
directions.** That is likely the mechanistic heart of the fidelity artifact:
descent is pulled toward precisely the modes that break the solver. What is
genuinely droppable is the low-order end — r_cos m=0 n=0 at |col|=1.24, z_sin
m=0 n=+1 at 2.22 — near-uniform scalings the normalised constraints barely see.

Caveat recorded in the script: one base point, one basin, column norms are local.
This is a reduction to TEST with a paired reduced-vs-full iteration, not to adopt.

### Killed on this result

- `run_aspect_ladder.sh` rung A≥16.0 — pushing deeper into the highest-aspect,
  most resolution-marginal direction available.
- `run_exhaust.sh` on floor2, at round 22 (0.02681982992351425, 22 strictly
  monotone rounds, never bit-identical). Its own question was already answered
  negatively; it was also refining a design the scorer cannot evaluate.

---


---

## The aspect lever at high fidelity: mechanism confirmed, my test mis-specified — 2026-10-06

`hf_asp13` was meant to be the like-for-like high-fidelity replay of the lever that
produced the low-fidelity `asp_a13.5` artifact. It was not like-for-like, and the
error was mine.

The floor enters the minimax through `search_violations` as one extra normalized
row, `(aspect_min - A)/aspect_min` (`minimax_p3.py:131`). Its magnitude therefore
depends entirely on where the seed already sits:

| run | fidelity | seed A | floor | pseudo-violation at iter 0 | real worst at iter 0 | ratio |
|---|---|---|---|---|---|---|
| `asp_a13.5` | low | 12.894 | 13.5 | **+0.045** | 0.0285 | 1.6x |
| `hf_asp13` | high | 8.408 | 13.5 | **+0.377** | 0.0804 | **4.7x** |

At 1.6x the floor is a pressure the minimax trades against the physics. At 4.7x it
IS the objective and the physics is collateral. `hf_asp13` was not testing the
lever; it was climbing aspect ratio. It behaved exactly as that description
predicts, over three high-fidelity iterations (~80 min each):

| iter | worst | A | iota | qi | mirror | fluxcomp | vwell |
|---|---|---|---|---|---|---|---|
| 0 | +0.080363 | 8.408 | -0.19591 | +0.08036 | +0.07967 | +0.07678 | -0.12402 |
| 1 | +0.090253 | 8.490 | -0.20441 | +0.08945 | +0.07816 | +0.09025 | -0.13049 |
| 2 | +0.137299 | 8.653 | -0.22741 | +0.13730 | **+0.02167** | +0.13604 | -0.14715 |

Killed at iteration 2. **I seeded a 5.1-wide aspect gap and then read the result as
if it were the lever.** Same class of error as the median-vs-tail read-out and the
bin-edge artifact: the instrument was pointed at the wrong quantity, and the number
it returned was perfectly real.

### What the run established anyway

**The aspect -> mirror coupling reproduces at the fidelity that scores.** Mirror fell
0.0797 -> 0.0217, a 3.7x reduction, for +0.245 of aspect ratio. That is the
mechanism the low-fidelity lever ran on, measured at high fidelity for the first
time, on a 100%-converging trajectory.

**It is a trade, not a free win.** qi went 0.0804 -> 0.1373 and fluxcomp
0.0768 -> 0.1360 over the same two iterations. At low fidelity the lever looked like
it bought mirror for nothing; that appearance was the fidelity gap, since qi is the
log-normalized constraint and the one whose low->high shift was largest. The honest
statement is that **aspect ratio buys mirror and charges qi/fluxcomp**, and whether
that trade closes depends on the local Jacobian, not on the lever existing.

### `hf_asp16`: the properly specified test

A seed for this experiment has two requirements that pull against each other, and
`harvest_best.py` satisfies neither, because it ranks by violation or by the
(worst, hv) front and both push toward LOW aspect — aspect is an objective, so every
unit of it is score given away. New script `extract_seed_by_aspect.py` selects on
aspect *and* cross-references the convergence survey, so a candidate is kept only if
the survey actually ran it at high fidelity and saw it converge.

Result: **`hl_s29_r1`, A=15.091, official worst 0.09897, nfp=4, high-fidelity
converging.** Note what this is: the high-aspect corner where the low-fidelity
breakthrough happened, except this design is real at the scoring fidelity. Its
violation (0.099) is the honest frontier, and its aspect leaves hv = 4.9*(1+LgradB)
against the (1.0, 20.0) reference, so feasibility here is worth having.

Floor set to **16.0**: a +0.057 pressure against real constraints at 0.099 — a 0.6x
ratio, gentler even than `asp_a13.5`'s 1.6x, chosen deliberately so a mis-sized floor
cannot be the explanation a second time.

> **Read-out, pre-registered.** The claim under test is the TRADE, not feasibility.
> CONFIRM if mirror falls materially while qi and fluxcomp do not rise enough to
> cancel it, i.e. official worst falls below 0.099. REFUTE if worst rises or stalls
> while aspect climbs, which is `hf_asp13`'s signature at a floor that cannot be
> blamed. Feasibility is not the bar here and must not be claimed without a
> high-fidelity evaluation from the installed scorer.

### The standing lesson, restated

Every one of my errors in this investigation has been **a correct measurement of the
wrong quantity** — the median instead of the tail, a bin edge instead of a feature, a
pseudo-constraint's magnitude instead of its presence. None was a bug; all three
produced clean numbers. The guard is not more care in reading output, it is checking
what the instrument is actually pointed at before it runs.

---

## `hf_asp16`: the lever is REFUTED, and the real blocker is the linear model — 2026-10-06

The properly specified high-fidelity aspect-lever run. 1729 evals, **1729/1729
converged** (third consecutive high-fidelity run at 100%), 570.5 min, terminated by
trust-region collapse: *"local minimax optimum reached"*.

| iter | official worst | A | iota | qi | mirror | fluxcomp | vwell |
|---|---|---|---|---|---|---|---|
| 0 | +0.099117 | 15.091 | -0.26090 | +0.09912 | +0.06779 | +0.09346 | -0.06700 |
| 1 | +0.097970 | 15.169 | -0.26469 | +0.09797 | +0.06526 | +0.09576 | -0.07192 |
| 4 | +0.096356 | 15.230 | -0.26919 | +0.09636 | +0.06908 | +0.09574 | -0.07667 |
| 5 | **+0.093147** | 15.307 | -0.27383 | +0.09308 | +0.07548 | +0.09315 | -0.07754 |
| 6-18 | +0.093147 (flat) | 15.307 | — | — | — | — | — |

**Verdict against the pre-registered read-out: REFUTED.** The criterion was "mirror
falls materially while qi/fluxcomp do not cancel it". Mirror *rose*, 0.0678 ->
0.0755. Total gain was -6.0% in five iterations, then thirteen iterations of exactly
nothing.

### Why the lever did nothing: it never entered the active set

`search` equals `worst` on every printed line. The floor contributes
`(16.0 - 15.09)/16.0 = +0.057`, which sits BELOW the real constraints at 0.099, so
the sixth row was never binding and the LP never saw it. **I sized the floor to be
gentle and sized it into inertness** -- the third variation on the same error
(median-for-tail, bin-edge-for-feature, magnitude-for-presence).

But aspect rose anyway, 15.091 -> 15.307, with the floor inert. **Raising aspect
ratio is the endogenous descent direction**; the optimizer does it unprompted. The
lever was never adding a mechanism, it was re-stating what the gradient already
wanted -- which also explains why `asp_a13.5` looked so effective at low fidelity:
it was taking credit for the search's own preferred direction.

### The actual blocker: the Jacobian is measured outside the trust region

| iter | predicted drop | actual drop | \|d\|inf | fd_step | ratio |
|---|---|---|---|---|---|
| 0 | +0.065017 | **+0.002419** | 0.01111 | 0.02 | **27x over** |
| 5 | +0.043048 | +0.004832 | 0.00478 | 0.02 | 8.9x over |
| 6 | +0.047104 | **-0.011401** | 0.00478 | 0.02 | wrong SIGN |
| 12 | +0.009143 | -0.009615 | 0.00056 | 0.02 | wrong SIGN |

From iteration 6 the LP promises a large drop and the measured step makes things
**worse**. The cause is visible in the table: steps of 0.011 down to 0.0001 against
a 0.02 finite-difference probe. Every high-fidelity gradient this project has
computed was secant-measured over an interval **2x to 200x larger than the step
taken with it**.

And `fd_step=0.02` is an inherited property. It comes from `fd_diagnostic.py`
("everything under ~1e-2 is noise"), which defaults to `--fidelity low` and whose
own header reasons from *low-fidelity* determinism. High fidelity is also
bit-identical (`fidelity_check_high.py`), so there is no noise floor to respect
here and no reason the low-fidelity step size transfers. **Running
`fd_diagnostic.py --fidelity high` at `hf_asp16`'s endpoint** (grad-steps 3e-4 ..
2e-2) to find the real plateau. Note the harness lesson re-earned: the first launch
reported `base point does not converge`, which was `No module named
'constellaration'` -- the `physics` service is not the oracle image, and a failed
eval is indistinguishable from a non-converging design. Use `train`.

### The barrier is real and basin-independent

| run | nfp | A | active set | final official worst | ended by |
|---|---|---|---|---|---|
| `hf_s0b` | 2 | 8.4 | qi, mirror, fluxcomp | +0.080991 | 20 iters, decelerating |
| `hf_asp16` | 4 | 15.3 | qi, mirror, fluxcomp | +0.093147 | local optimum |

Two basins with nothing in common -- different field-period count, aspect ratio
7 apart, independently seeded -- stall within 0.012 of each other at ~8x the
feasibility bar, both at 100% convergence. **This is not a resolvability problem and
not a compute problem.** The "~280 more iterations" projection from `hf_s0b` was
measuring the deceleration of a broken linear model, not a distance to feasibility.

### Two wall-time defects fixed (neither changes a search decision)

1. **Degenerate line-search ladder.**
   `a_min = min(1.0, fd_step/(2*dnorm))` saturates at exactly 1.0 once
   `dnorm < fd_step/2`, and `geomspace(1, 1, 9)` is nine copies of 1.0 -- so all
   nine trial points are BIT-IDENTICAL. That is what the nine-identical-numbers
   trial rows from iteration 10 onward were. Now deduped to the distinct alphas,
   verified all-or-nothing (only a_min=1.0 collapses, and to exactly one point, so
   `win` is 0 either way and the trust-region signal is unchanged). The
   trust-region thirds now index `len(alphas)`, not `args.n_line_search`, because
   the original code's own comment warned that compacting this list shifts them.
2. **`--max-stall N`** -- stop after N non-improving iterations instead of shrinking
   delta by 0.85x until it crosses `delta_min`. `hf_asp16` spent iterations 6-18
   (1183 high-fidelity evals, 6.5 of 9.5 hours) confirming an optimum that was
   unambiguous by iteration 8. Default 0 in `minimax_p3.py` to preserve the
   historical low-fidelity behaviour; default 4 in `run_highfid.sh`, where it is
   always right. `FD_STEP` is now overridable there too, pending the measurement.

**Score of record: 0.0.** Best high-fidelity-admissible design: +0.080991
(`hf_s0b`, A=8.4).

---

## The barrier is NOT the parameterisation, and we are far ahead of the corpus — 2026-10-06

`check_truncation_ceiling.py` (no VMEC++, so free and unconfoundable by the
convergence problem) tested the one hypothesis that `fd_step` cannot address: that
P3 feasibility needs Fourier content outside our `(5, 9)` window, in which case
every run has been optimising against a wall.

**Exonerated by construction.** All 3000 streamed ConStellaration records have
`boundary.r_cos` and `boundary.z_sin` of shape exactly `(5, 9)` -- m = 0..4,
n = -4..+4. The benchmark's own corpus is defined in precisely the parameterisation
`minimax_p3.py` optimises. There is no out-of-window amplitude to miss.

### Where we actually stand against the corpus

| | official worst violation |
|---|---|
| dataset min (best of 3000) | 0.3328 |
| dataset p1 / p5 | 0.4159 / 0.5411 |
| dataset p25 / median | 0.9546 / 1.2587 |
| **our best high-fidelity-admissible (`hf_s0b`)** | **0.0810** |
| dataset designs better than ours | **0 / 3000** |

Our stalled design is **4.1x better than the best of 3000 reference designs**. The
0.081 wall is not underperformance; it is unexplored ground with no nearby example
to learn from. This also retires the framing that the project is "behind" -- it is
past the corpus and stuck on the last factor of 8.

### Which constraint is the wall

| constraint | dataset min | fraction satisfying the 1e-2 bar |
|---|---|---|
| qi | -0.0444 | **0.002** |
| vwell | -1.1086 | 0.051 |
| fluxcomp | -0.7405 | 0.139 |
| mirror | -0.4825 | 0.227 |
| iota | -1.8675 | 0.232 |

**Every constraint is individually attainable**, qi included -- 6 designs in 3000
clear it alone. What is absent from the corpus is the INTERSECTION, which is exactly
consistent with both high-fidelity runs stalling with qi, mirror and fluxcomp
mutually binding. The problem is the joint feasible set, not any single metric.

Caveat: first 3000 streamed records, not a random sample of the full corpus, so the
fractions are indicative and the minima are upper bounds on the corpus best.

### The chain of elimination is now closed

| candidate cause | status | evidence |
|---|---|---|
| harness / scorer bug | ruled out | 4/4 dataset designs converge through the identical call |
| resolvability | ruled out | 1820/1820, 1729/1729, 268/275 converge at high fidelity |
| parameterisation ceiling | ruled out | corpus is the same `(5, 9)` window |
| compute / iteration count | ruled out | both basins reach local minimax optima, not a time limit |
| **the linear model** | **the remaining cause** | LP overpredicts 27x; wrong SIGN from iteration 6 |

And the concrete defect inside that last row is identified: `fd_step=0.02` against
steps of 0.011 down to 0.0001, i.e. a secant measured 2-200x outside the interval it
is used in, with the 0.02 itself inherited from a LOW-fidelity noise measurement
that high-fidelity determinism makes inapplicable. The high-fidelity
`fd_diagnostic.py` re-measurement is in flight; its first point, `h=3e-4 -> |g|=7.17,
nonzero 81/81, fail 0`, already contradicts the "everything under ~1e-2 is noise"
premise that set the step.

**Score of record: 0.0.**

---

## THE OBJECTIVE IS A STAIRCASE. There is no usable gradient — 2026-10-06

`fd_diagnostic.py --fidelity high` at `hf_asp16`'s endpoint (worst +0.093147). This
is the root cause of everything above it in this log.

### The scan: flat treads

`f(x + h e_i) - f(x)` holding CONSTANT as h widens means the function jumped once
and then went flat. It does this in every coordinate scanned:

| coord | df at h | df at wider h | agreement |
|---|---|---|---|
| 0 (real idx 4) | +1.076351e-4 @ 1e-2 | +1.074222e-4 @ 3e-2 | **0.2%** |
| 40 (real idx 44) | +4.589847e-4 @ 1e-2 | +4.490728e-4 @ 3e-2 | **2%** |
| 20 (real idx 24) | +5.618904e-4 @ 3e-4 | +5.665692e-4 @ 1e-3 | **0.8%** |

Below those thresholds df is ~1e-8 to 1e-7, i.e. nil. **Flat treads roughly 1e-3 to
1e-2 wide in whitened units, step heights 1e-4 to 1e-3, and non-monotone** -- coords
20 and 40 both step back DOWN at larger h (coord 40: +4.49e-4 at 3e-2 falls to
+2.95e-4 at 1e-1). `fd_diagnostic.py`'s own docstring predicted this mechanism:
qi runs through a Boozer transform, spline interpolation and bounce-point
root-finding, any of which is piecewise-constant, and "a deterministic staircase
produces gradients that are exactly zero almost everywhere and enormous at the
steps."

### No usable finite-difference step exists at any scale tested

| h | \|g\| | pair cosine | \|g_a\|/\|g_b\| |
|---|---|---|---|
| 3e-4 | 7.1729e+0 | — | — |
| 1e-3 | 2.7654e+0 | 0.7050 (vs 3e-4) | 2.594 |
| 3e-3 | 1.6241e+0 | 0.7398 (vs 1e-3) | 1.703 |
| 1e-2 | 1.2467e+0 | 0.7530 (vs 3e-3) | 1.303 |
| 2e-2 | 1.4942e+0 | **0.8737** (vs 1e-2) | 0.834 |

All 81/81 coordinates nonzero, 0 failures, at every h -- so this is structure, not
noise, and high fidelity is bit-identical so it cannot be noise. Cosine rises
monotonically with h and **tops out at 0.874, never approaching 1**: consecutive
gradients are ~29 degrees apart at the best pair. By the script's own stated
criterion there is no usable FD step in the range probed.

This explains, completely and quantitatively, the LP's 27x overprediction and its
WRONG-SIGN predictions from iteration 6. A direction built from step heights divided
by an arbitrary denominator is not a descent direction.

### RETRACTION: the "basin-independent barrier" was a shared artifact

> **The 2026-10-06 reading of `hf_s0b` (+0.080991) and `hf_asp16` (+0.093147) as
> evidence of a real physical wall is WITHDRAWN.**

Both runs terminated when delta shrank below the tread width. At that point every
probe lands on a flat tread, df is identically zero, the LP sees no available
descent, and `minimax_p3.py` prints `local minimax optimum reached`. **Neither run
reached a local optimum. Both ran out of resolution and misreported it as
convergence.** Two runs stalling in the same place because they fail by the same
mechanism is not two independent measurements of a barrier -- and I read their
agreement as corroboration, which is the bin-edge error again: structure in the
instrument mistaken for structure in the world.

What survives from that section: the corpus comparison (our 0.0810 beats 0/3000
dataset designs, min 0.3328) and the truncation exoneration (the corpus is the same
`(5, 9)` window) are independent of this and stand.

### Also retracted: my own "fd_step is too wide" diagnosis

I proposed that `fd_step=0.02` was the defect because steps ran 0.011 down to
0.0001 -- a probe 2-200x wider than the step. The falloff exponent killed it:
|g| ~ h^-alpha with alpha = 0.79, 0.48, 0.22 decaying toward zero as h GROWS, and
the cosine improving monotonically with h. Anything smaller than 0.02 is worse, not
better. `minimax_p3.py`'s own `--delta-min` help text had the right instinct --
"shrinking the trust region BELOW the finite-difference interval is meaningless" --
and I read past it while proposing the opposite.

The low-fidelity provenance of `fd_step=0.02` was still worth flagging; it simply
turned out to be right for a reason that happens to carry over, because the
roughness scale is a property of the physics pipeline rather than of solver
settings.

### Consequence: `--max-stall` is a correctness fix, not a wall-time saving

Flooring delta and stopping on stall were logged earlier as efficiency measures.
They are not. Letting delta collapse below the tread width drives the optimizer into
a regime where its model is undefined, produces sign-wrong predictions, and ends in
a false convergence report. `hf_asp16` iterations 6-18 were not merely 1183 wasted
evals -- they were 13 iterations of a search operating below its own resolution.

### Next: `es_gradient_check.py` -- is a SMOOTHED gradient reproducible?

Gaussian smoothing, `f_sigma(x) = E[f(x + sigma u)]`, is genuinely smooth once sigma
exceeds the tread width, and its gradient is estimable from function values with no
differencing. The test is deliberately the same one FD failed, so the numbers are
comparable: **draw two INDEPENDENT estimates at one sigma and take their cosine.**
Two estimates of a real direction agree; two draws of noise do not. Antithetic
sampling, 32 pairs = 64 evals per estimate, sigmas 0.02/0.05/0.10, at high fidelity
from `hf_asp16`'s endpoint.

> **Read-out, pre-registered.** cosine >= 0.95 at some sigma: a reproducible
> direction exists where FD has none, and that sigma is the method fix. 0.8737 to
> 0.95: report the margin and the N required, do NOT declare a fix on a rounding
> margin -- the comparison is against FD's BEST pair, 0.8737, not its median.
> Below 0.7: forming a gradient is the wrong move at all, and the next step is a
> method that never forms one (model-based DFO / CMA-ES over the 29 coordinates
> carrying 90% of the column norm).
>
> Cosine measures REPRODUCIBILITY, not descent. A reproducible direction can still
> be a bad one; confirming descent needs a line search along it.

**Score of record: 0.0.** Best admissible design: +0.080991, still 4.1x better than
the best of 3000 corpus designs.

---

## Smoothing fails, and the "wall" has never been measured at the scoring fidelity — 2026-10-07

### `es_gradient_check.py`: no estimated gradient is reproducible here

Gaussian smoothing was the principled repair for a staircase: `f_sigma(x) =
E[f(x + sigma u)]` is genuinely smooth once sigma exceeds the tread width, and its
gradient needs no differencing. Tested the way FD was tested -- two INDEPENDENT
estimates at one sigma, cosine between them -- so the numbers are directly
comparable with FD's 0.8737 ceiling. 32 antithetic pairs per estimate, high
fidelity, from `hf_asp16`'s endpoint (f=2.30380520e-02, worst +0.093147).

| sigma | cosine(A,B) | \|gA\| | \|gB\| | ratio | evals kept |
|---|---|---|---|---|---|
| 0.02 | **0.4109** | 4.7340 | 3.7120 | 1.275 | 62/64, 64/64 |
| 0.05 | **0.1055** | 7.0312 | 5.1064 | 1.377 | 50/64, 57/64 |
| 0.10 | **-0.0176** | 4.1218 | 5.0099 | 0.823 | **31/64, 33/64** |

At sigma=0.10 two independent estimates of the same gradient are ORTHOGONAL. The
collapse is monotone in sigma, 0.4109 -> 0.1055 -> -0.0176.

Worse than FD at every sigma, and getting worse as sigma grows -- the opposite of
the hypothesis. Two reasons, both measured:

1. **Undersampling.** For independent estimates the expected cosine is
   `|g|^2/(|g|^2 + s^2)`, so 0.4109 implies `s^2 = 1.43|g|^2`. Reaching 0.95 needs
   `s^2 = 0.053|g|^2`: a **27x variance reduction**, i.e. ~870 pairs instead of 32,
   **~1740 evals per gradient** -- about 9 hours for one descent direction. Outside
   the affordable range by an order of magnitude before direction quality even
   enters.
2. **Direction-biased censoring.** A sample perturbs all 81 coordinates at once, so
   its displacement is `sigma*sqrt(81)` -- 0.18, 0.45, 0.90 whitened units at the
   three sigmas, against single-coordinate stencil steps of ~0.02. Failures rose
   3% -> 22% -> 52%, and they are not random: they concentrate in whichever
   directions break VMEC++, the same direction-dependence behind the original
   fidelity-gap artifact. At 52% the estimator is no longer a gradient of `f_sigma`
   but of `f` restricted to the convergent half of the ball, and when one side of an
   antithetic pair survives and the other does not, the pairing's cancellation is
   gone. This was flagged BEFORE the numbers landed, so the read-out was not
   adjusted to fit them.

Pre-registered third branch applies: **forming a gradient at all is the wrong move
on this objective.**

### Correction to my own fallback recommendation

That branch named "DFO/CMA-ES over the 29 coordinates carrying 90% of the column
norm". **That coordinate set is invalid** -- it comes from
`analyze_jacobian_sparsity.py`, computed from FD stencils, which the staircase
finding retires. Ranking coordinates by FD column norm on a piecewise-constant
objective ranks probe artifacts. I pre-registered a fallback that depended on a
result the same experiment would invalidate.

### `corpus_pca.py`: the dimension cannot be bought from the data either

The gradient-free replacement -- PCA over all 3000 corpus boundaries, which are
exactly `(5, 9)`, whitened by `coef_scales.npy`:

| variance | components of 81 |
|---|---|
| 50% | 12 |
| 80% | 33 |
| 90% | 46 |
| 95% | **57** |
| 99% | 71 |

Participation-ratio effective dimension **32.86**. No strongly low-dimensional
structure. **And this corrects a figure this project has been carrying:**
`paper_v1`'s "~2.4 effective dimensions" was the dimensionality of one optimizer's
own SEARCH CLOUD -- a symptom of a collapsed search -- not a property of the
problem. Real stellarator boundaries occupy ~33 effective dimensions. Reading 2.4 as
intrinsic dimension would justify exactly the wrong reduction.

### THE CONFOUND: every "wall" confirmation is a low-fidelity measurement

`ngopt_alm_p3.py` and all its variants default to `--fidelity low`. So do the NAE,
omnigenity, generative-probe and grounded-walk lanes. Every stall value this project
has used to argue a structural wall:

| run | method | fidelity | floor |
|---|---|---|---|
| `ngopt_v1` | ALM-NGOpt, recreate-per-outer | **low** | 0.0917 |
| `ngopt_v2` | + stall widening | **low** | 0.0965 |
| `persist_v1` | one persistent optimizer | **low** | 1.031 |
| `ngopt_v3_seed126` | diverse seed, nfp=2 | **low** | 0.2562 |
| `ngopt_v3_seed14nfp4` | diverse seed, nfp=4 | **low** | 0.1522 |
| `ngopt_nae_nfp2ar8` | NAE seed | **low** | 0.247 |
| omnigenity-fitted seed | official recipe | **low** | 0.2805 |

After 2026-10-05 -- median |low->high| shift 0.00954, p90 0.02844, tail 5.4x the
bar, active-set flips, and 0/18 of designs below worst 0.05 converging at high
fidelity -- **a low-fidelity floor cannot certify anything about feasibility.** The
log's "7th independent confirmation" is seven measurements by an instrument this
project later showed unfit for exactly this question. They are not thereby wrong;
but the wall they agree on **has never been observed at the fidelity that scores.**

Set against that: the one method ever run at high fidelity, the FD minimax, reached
**+0.080991** -- better than all seven -- while crippled by a staircase that makes
its gradients meaningless.

| | low fidelity | high fidelity |
|---|---|---|
| gradient-based | 0.0965 | **0.081** (staircase-crippled) |
| rank-based (NGOpt/CMA) | 0.0917-0.2805, x7 | **never run** |

And the paper's own ALM-NGOpt baseline reports violation as low as **0.0** on 4/4 P3
instances at aspect_ratio 6/8/10/12. `hf_s0b` sits at A=8.4, inside that range. So
the 0.08-0.09 region is demonstrably crossable by this method family at our aspect
ratio.

### Running: `ngopt_hf1`

ALM-NGOpt at **high fidelity**, seeded from `hf_s0b`'s +0.080991 point (nfp=2,
A=8.4, 100% converging), 100 outer x 32 budget = 3200 evals, 32 workers, ~10 min per
outer, ~17 h.

Why this combination and not another lever: rank-based selection only ORDERS
candidates and never differences them, so a flat tread merely makes two candidates
tie instead of producing a zero gradient -- it is structurally immune to the defect
that broke every gradient method here. And the step size is in the right regime by
measurement, not hope: `--sigma0 0.0008` is in RAW coefficient units while treads
are 1e-3 to 1e-2 WHITENED, which at scales of 0.00023-0.0039 is raw treads of 2e-7
to 4e-5 -- so NGOpt's perturbations are 20-3000x wider than a tread and cross many
per step. Unlike all seven prior runs, which crawled up from ~1.1, it starts 10x
closer to the bar.

> **Read-out.** Any movement below 0.0810 is the first high-fidelity progress since
> `hf_s0b`, and the first evidence that the staircase -- not the physics -- was the
> blocker. A flat run at 0.081 is the FIRST honest high-fidelity measurement of the
> wall, which is worth having either way. Feasibility requires a high-fidelity
> evaluation from the installed scorer and may not be claimed without one.

**Score of record: 0.0.**

---

## `ngopt_hf1` killed at outer 0: a third inherited low-fidelity parameter — 2026-10-07

ALM-NGOpt at high fidelity from `hf_s0b`'s +0.080991 seed converged **7/32 at outer
0** (best_worst 0.0828). Killed immediately -- not because it was slow, but because
it was INVALID: rank-based selection over a 22%-surviving population ranks what
CONVERGED, not what is good. That is the same direction-biased censoring that
corrupted the smoothed-gradient estimates, and it defeats the one property that made
a rank-based method the right choice.

### Cause: a flat raw sigma against a 539x range of coefficient scales

`coef_scales.npy` spans **0.00023 to 0.1234** across the 81 free coefficients. The
script used `sigma = np.full(90, args.sigma0)` and a box of `sigma0 * bound_mult`,
both flat in RAW units, so at `--sigma0 0.0008` / `bound_mult 10` the box was 0.008
raw everywhere:

| coefficient | scale | box in whitened units | vs a tread (1e-3..1e-2) |
|---|---|---|---|
| largest-scale | 0.1234 | 0.065 | ~6 treads |
| smallest-scale | 0.00023 | **35** | **~3500 treads** |

The small-scale coefficients were searched in a box 3500 treads wide while the
large-scale ones got 6. And `--sigma0 0.0008` traces to a calibration recorded in
its own help text -- "sigma=0.005 broke ALL 8 test candidates, 0.002 broke 7/8,
0.001 converged 7/8, 0.0005 converged 8/8" -- taken on **8 candidates at low
fidelity**, where the convergence basin is far wider.

Shrinking a flat sigma cannot fix this: holding the smallest-scale coefficient to
0.02 whitened forces the largest to 0.0012, below a single tread, so it would stop
moving at all. The fix is `--sigma-whitened W`, setting `sigma = W * coef_scales` and
deriving the box from it, making the perturbation uniform in the units every other
measurement here uses.

### The pattern worth naming

| parameter | value | provenance | status |
|---|---|---|---|
| `fd_step` | 0.02 | low-fidelity noise study | right by luck (roughness is physical) |
| `sigma0` | 0.0008 | 8 low-fidelity candidates | **wrong at high fidelity** |
| flat-vs-whitened sigma | flat raw | never examined | **wrong given a 539x scale range** |

**The search was configured for a solver we stopped using.** Each of these was
defensible where it was set and none was re-derived when the oracle changed. This is
the inherited-property error applied to hyperparameters, where it is worst, because
nothing ever re-reads the justification.

### Running: `ngopt_hf2`

Same as `ngopt_hf1` but `--sigma-whitened 0.02` -- the single-coordinate stencil step
that converges 100% at high fidelity. NGOpt moves all coordinates at once, so typical
displacement is `0.02*sqrt(81) = 0.18` whitened, where the smoothing study measured
~3% failure. 100 outer x 32 budget, 32 workers, ~17 h.

> **Read-out, unchanged from `ngopt_hf1`.** Movement below 0.0810 is the first
> high-fidelity progress since `hf_s0b` and the first evidence the staircase rather
> than the physics was the blocker. A flat run is the first honest high-fidelity
> measurement of the wall. **Convergence rate is now a validity precondition, not a
> performance note:** below ~90% the ranking is censored and the run must be
> re-tuned rather than read.

Also corrected `es_gradient_check.py`'s own closing recommendation, which pointed at
the FD-derived 29-coordinate reduction -- text I wrote before the experiment produced
the result that invalidates it.

**Score of record: 0.0.**

---

## `ngopt_hf2`: the whitened fix works, and the wall is a THREE-WAY tie — 2026-10-07

### The sigma fix is confirmed

Identical seed, budget and fidelity to `ngopt_hf1`; only the sigma units changed.
**7/32 -> 32/32 converged**, and 32/32 has held for all 34 outers so far. The
validity gate (>=90%, below which rank-based selection is censored and the run must
be re-tuned rather than read) passes throughout.

### Trajectory: constant steps at lengthening intervals

| step at outer | best_worst | interval since previous step | step size |
|---|---|---|---|
| seed | 0.08099 | — | — |
| 2 | 0.0782 | — | — |
| 6 | 0.0780 | 4 | 0.0002 |
| 11 | 0.0778 | 5 | 0.0002 |
| 15 | 0.0773 | 4 | 0.0005 |
| 22 | 0.0766 | 7 | 0.0007 |
| 31 | **0.0761** | 9 | 0.0005 |

Net -6.0% over 32 outers. **Intervals grow (4, 5, 4, 7, 9) while step sizes stay flat
at ~0.0005** -- logarithmic convergence. This retires an idea I floated at outer 22,
that a longer plateau buys a proportionally bigger step (rho accumulating to pay for
a tread crossing): plateau 8 returned 0.0005, smaller than plateau 6's 0.0007. It
also means my earlier linear projection of ~0.060 at outer 100 was too optimistic;
**expect ~0.072-0.073**, i.e. this configuration does not reach feasibility with more
wall time, and the shortfall is ~7x rather than a budget away.

Worth noting what this run is NOT doing: `ngopt_v1/v2` went dead-flat for 33-50+
consecutive outers with rho into the billions, and `bound_mult` grew to its 60.0 cap.
Here rho is at 116 and `bound_mult` has never left 10.0 -- the stall-widening
mechanism has never needed to fire, because no plateau has reached
`--stall-patience 12`. This is slow descent, not the historical stall.

### The finding: a three-way tie with two slack constraints at pinned aspect

Per-outer best, official violations:

| outer | qi | mirror | fluxcomp | iota | vwell | A |
|---|---|---|---|---|---|---|
| 22 | +0.07665 | +0.07615 | +0.07601 | -0.20004 | -0.13021 | 8.440 |
| 26 | +0.07695 | +0.07694 | +0.07480 | -0.19890 | -0.12774 | 8.429 |
| 31 | +0.07568 | +0.07553 | +0.07606 | -0.19898 | -0.13170 | 8.441 |

1. **qi, mirror and fluxcomp bind in near-lockstep, within 0.001-0.005.** Which one
   leads rotates between iterations; when one falls another rises to meet it.
   Reducing the max requires reducing all three AT ONCE, which is the mechanism
   behind logarithmic convergence -- almost every direction helps one and hurts
   another, so only rare directions make progress.
2. **iota (-0.199) and vwell (-0.131) have large unused slack**, and vwell is
   drifting MORE slack over the run (-0.126 -> -0.133). Two of five constraints are
   free capacity that nothing is converting into progress.
3. **Aspect ratio is pinned at 8.42-8.45** across all 34 outers. The one trade
   measured at high fidelity (`hf_asp13`: aspect buys mirror, charges qi and
   fluxcomp) sits unused -- and against a three-way tie it would be a wash anyway.

### The basin map this creates

| run | A | active set | conflict | best worst (high fidelity) |
|---|---|---|---|---|
| `ngopt_hf2` | 8.43 | qi, mirror, fluxcomp | **three-way** | **0.0761** |
| `hf_asp16` | 15.3 | qi, fluxcomp (mirror slack 0.018) | two-way | 0.0932 |

Low aspect: better absolute value, harder three-way conflict. High aspect: weaker
two-way conflict, starts 22% worse. **This is the first time the wall has been
characterised at the fidelity that scores, with 100% convergence** -- every prior
account of it was low-fidelity and framed as a two-way qi-vs-mirror anticorrelation
(r=-0.95 along `ngopt` trajectories, r=-0.87/-0.73 in other basins). A three-way tie
with two slack constraints is a materially different object from a two-way
anticorrelation, and it is what the method has to be designed against.

**Score of record: 0.0.** Best high-fidelity-admissible design: **+0.0761**
(`ngopt_hf2` outer 31), superseding `hf_s0b`'s +0.080991.

---

## PAUSED 2026-10-07 — how to resume

All search containers stopped; compute is free. Nothing is mid-write.

**State preserved.** `ngopt_hf2` stopped at outer 33 of 100, `best_worst=0.07605712`,
rho=115.8, bound_mult=10.0, stall_count=2, 32/32 converged throughout.
- checkpoint: `output/ngopt_alm_p3/ngopt_hf2_checkpoint.json`
- all 1088 evals with params: `output/ngopt_alm_p3/ngopt_hf2_results.jsonl`
- best design as a ready seed: **`output/ngopt_hf2_best_seed.npy`** (92-col, nfp=2)

**To resume exactly where it stopped** (the script supports it; checkpoint carries
lam/rho/bound_mult/stall_count):

    docker compose run --rm --name vega-ngopt-hf2 train scripts/ngopt_alm_p3.py \
      --tag ngopt_hf2 --resume --fidelity high --timeout-seconds 1800 \
      --sigma-whitened 0.02 --coef-scales /work/output/coef_scales.npy \
      --n-outer 100 --budget-per-outer 32 --n-workers 32

Expected if resumed: ~0.072-0.073 at outer 100, NOT feasible. The trajectory is
logarithmic (constant ~0.0005 steps at lengthening intervals), so finishing it buys
the clean baseline, not the bar.

**Open threads, in the order they are worth picking up.**

1. **The three-way tie is the thing to design against**, not the trajectory. qi,
   mirror and fluxcomp bind within 0.001-0.005 while iota (-0.199) and vwell
   (-0.131) sit slack and aspect is pinned at 8.43. Any method that cannot reduce
   all three at once will converge logarithmically here regardless of its internals.
2. **Untested: whether the two-way basin is easier.** `hf_asp16` at A=15.3 has
   mirror slack by 0.018, so only qi and fluxcomp bind -- a weaker conflict starting
   22% worse (0.0932). Whether a weaker conflict beats a better starting value is
   unknown and is one high-fidelity run to find out, seeded from
   `output/aspect_seed.npy`.
3. **`--max-stall` and the whitened sigma are now defaults worth keeping.**
   `run_highfid.sh` defaults `MAX_STALL=4`; `ngopt_alm_p3.py` needs
   `--sigma-whitened` passed explicitly (default 0 preserves the historical flat
   behaviour, which is WRONG at high fidelity -- 7/32 convergence).
4. **Convergence rate is a validity precondition.** Below ~90%, rank-based selection
   ranks what converged rather than what is good; re-tune sigma rather than reading
   the run.

**Do NOT reuse, all invalidated 2026-10-06/07:** the FD-derived 29-coordinate
reduction (`analyze_jacobian_sparsity.py`); any "hv if feasible" ranking computed at
low fidelity; the "~2.4 effective dimensions" figure (corpus PCA gives 32.86); and
the seven low-fidelity stall values as evidence of a structural wall.

**Score of record: 0.0.** Best high-fidelity-admissible design: **+0.0761**.

---

## Watercap (electrospray + corona water recovery from cooling-tower exhaust) — 7th domain, partner-proposed, 2026-10-20

**Task**: a grounded vertical tube with saturated, mist-laden tower exhaust flowing up. An electrospray stage (lower) sprays charged drops down against the flow to capture vapor; a corona stage (upper, a stack of electrode discs; the partner's about-to-deploy prototype) charges what remains and drives it to the wall. Objective: minimize total water (vapor + liquid) leaving the top; also power per liter, pressure drop, arc margin. 15 design params + 5 operating-condition aux (airfoil convention), 13 targets. `scripts/watercap_oracle.py`, `generate_watercap_dataset.py`, `smoke_watercap_oracle.py`, `gym_schema.watercap_domain()`.

**Why a self-written ROM again (the mug v1->v2 rule, knowingly broken):** no accepted solver covers the coupling. Stock OpenFOAM (both forks) has condensing Lagrangian spray (ESI's `LiquidEvaporation` hard-clips condensation to zero, `max(kc*(Cs-Cinf),0)`; Foundation has a `condensation` switch; `liquidEvaporationBoil` cannot condense at all), but corona ion transport, droplet field/diffusion charging, Coulomb force on an evolving parcel charge, and the ion-drag body force are all custom code. ESI v2306's `Coulomb` particle force only supports fixed q(d). Closest open base: `swtbkim/electrostaticSimpleFoam` (GPL-3.0, ESI v2412, 2 commits; Poisson + ion transport + Kaptsov/Peek, no particles). So: ROM (low/medium) now; CFD tier `FIDELITY_PRESETS["high"]` raises until built and is deliberately NOT listed in `watercap_domain()`.

**ROM**: 1D axial, turbulent well-mixed cross-section. Mist co-flows (stiff LSODA march, quasi-steady drop temperature with Kelvin-Thomson). Spray classes the upflow can carry (v_t < 0.8U, i.e. essentially every electrospray drop) join that implicit march; only drops that genuinely fall get a time-traced PSI-cell counterflow. Cone-jet scaling (Fernandez de la Mora & Loscertales), Peek onset, Townsend I-V, Pauthenier + White charging (with bounded attractive-charging forms), Deutsch wall loss, unipolar Coulomb dispersion of the spray cloud, mixing-equilibrium thermodynamic ceiling. Water balance closes to <=1.4e-3 on 274/275 uniform designs (median 1e-8). Uncalibrated, named constants: `CORONA_IV_COEFF`, `DISC_EDGE_RADIUS_M`, `SPARK_E_*`; the prototype's I-V and sparkover data replace them.

**Three bugs the sanity battery caught, all of the "ran fine, numbers wrong" kind:** (1) spray step ~100x the drop's thermal relaxation time, so cold drops condensed at mid-step temperature for far too long (balance off 15%, "utilization" 1.18 > the ceiling); (2) lagged counterflow sources for entrained drops overshot until LSODA died (fixed architecturally, not by relaxation); (3) fine mist fully evaporating in subsaturated inlet air made dm/dt finite as m->0 (45% of uniform samples failed; smooth evaporation cutoff -> 1.8%).

**Physics findings (ROM, but each rests on an identity or a bound, not a fitted constant):**
1. **Charge cannot capture vapor.** At the Rayleigh limit the Thomson term cancels the Kelvin term to (1 - 1/eps_r): p_eq/p_sat >= 1 - 1e-4 at every radius 0.1-100 um (asserted in the smoke test). Charged spray at the saturated inlet temperature removes 0.000% of the vapor. Vapor capture is bounded by spray subcooling: at 2 L/min of 20 C spray into 0.15 m^3/s of saturated 35 C air the adiabatic mixing ceiling is 10.4% of the vapor; at 10 L/min of 10 C, 50%. The Auburn group (L. Cremaschi, DOE DE-EE0009161) claimed the opposite mechanism; their own MS thesis (Morcelli 2022) measured dehumidification "always limited and lower than 1%... mainly within the uncertainty of the dew point sensor", consistent with this. Their OpenFOAM v9 model (Yel Mahi PhD 2025) is embargoed to 2030.
2. **Charging the spray HURTS vapor capture.** Drop-size x charge sweep at fixed flow/T: neutral 5-100 um drops reach the ceiling (100%); at 10% of Rayleigh charge 24-53%; at 50%, 1-10% -- Coulomb dispersion puts the spray on the wall in milliseconds, before it exchanges heat.
3. **Scaling-law electrospray at useful flow is unrealizable.** Cone-jet currents at L/min give a launch space-charge field ~4.5e8 V/m, ~190x humid-air breakdown (a lower bound -- the cone is treated as filling the tube). 67% of uniform-sampled designs arc, almost all for this reason. Feasible charge is <~1% of Rayleigh at 30 um.
4. **What the spray actually needs is collectability, not charge.** Neutral 5 um spray leaks 22% out the top through the corona stage; 30-100 um is caught. Best designs in the sweep: neutral/weakly-charged 30-100 um spray + corona, outlet total water 0.85-0.86 at 2 L/min / 20 C. Corona-only (the prototype) captures 97% of a 15 um mist at 30 kV, 1.3 mA, 38 W (uncalibrated I-V).

**Fidelity**: low ~3.0 s mean (100 credits), medium ~6.3 s (211); they agree to median 2e-5 on the objective -- discretization is converged; the gap that matters is ROM->CFD.

**Surrogate plan (lit review, prioritized for "works" over novelty)**: ROM as the low fidelity; GP (ARD Matern-5/2) on the CFD-minus-ROM residual, or ROM outputs as GP inputs (NARGP-style), checked against AR1 co-kriging and an HF-only GP + TabPFN baseline (Brevault 2020: linear MF wins at few HF samples; Johnson 2026: GPs beat TabPFN on smooth simulator outputs at n=10d-40d). PCA+GP for axial profiles. Constrained BO on arc_margin for choosing CFD runs; Kennedy-O'Hagan calibration with strong discrepancy priors once the partner's instrumented inlet/outlet data exists. Explicitly not: PINNs, neural operators at n<300.

**Open, needs the user/partner:** see the decisions list in the session summary -- spray parameterization (electrospray vs. atomizer + charge-to-mass), the CFD tier's scope, and calibration data from the prototype.

**Running**: 20k-row low-fidelity v1 dataset, `output/watercap_*`, log `logs/watercap_gen_v1.log`.

### Resolved electric model (2026-10-20, same day): `stonelab.ehd` + `stonelab.ehd_particles`, Vega tier `watercap_ehd.py`

Direct user request: expand the simulation side, build a real electric model rather than leaning on the ROM's coaxial-equivalent corona. Built in StoneLab (validated reference physics; scipy as a new optional `ehd` extra), adapted into Vega as a second `OracleFunction` (`watercap_ehd`, tiers `ehd_coarse` 1500 credits / `ehd` 7000) over the same 13 targets, so ROM and EHD rows share one schema for multi-fidelity work.

**What it solves**: axisymmetric (r, z) Poisson + unipolar ion drift-diffusion + Kaptsov/Peek corona, on a graded tensor grid with Shortley-Weller boundaries (rounded disc edge resolved, not stair-cased); one periodic unit cell per disc, marched along the section. Lagrangian droplets: signed Pauthenier (exact per-step) + White charging, inertia-free drift, pipe-flow eddy diffusivity with van Driest damping at the wall AND the discs plus the full Ito drift, gas flow = 1/7-power pipe profile + potential correction around the discs. Two-way: drop space charge into Poisson, ion capture as a sink.

**Validation (tests, all passing)**: Laplace coaxial exact to 1e-14 (log-mean radial conductance); Kaptsov coaxial corona I-V vs the exact semi-analytic solution, first-order convergence in bulk spacing (+0.84/+0.43/+0.22% at 60 kV), converges 25-200 kV; negative corona the exact mirror; zero current below onset; Pauthenier exact; well-mixed condition; Deutsch-Anderson limit within 3%.

**Numerics that had to be found, not assumed** (each measured): (1) midpoint radial differences left a grid-independent +2.6% current floor -> log-mean conductance; (2) three fixed-point Kaptsov updates each worked in one regime and failed in another (limit cycle, stall, divergence to 1e296 at 80 kV) -> Newton on the coupled (phi, rho, injection) system with an exact (FD-checked) Jacobian; (3) plain Newton predicted rho = -9e5 C/m^3 from near-singular rows in ion-free cells -> pseudo-transient continuation + physical Einstein diffusion; (4) Kaptsov is a complementarity condition -> active set, one constraint dropped at a time; (5) a max-norm residual passed with 0.4% current imbalance -> conservation-based convergence; (6) voltage continuation from 1.08x onset. Particle side: drops falling straight through discs (11% "fouling"), undamped near-disc diffusion (2-3%), missing z Ito drift (0.5%/disc) -> now 0.175% over 15 discs (~1e-4/disc, upper end of a deposition-velocity estimate). And a handoff bug: the ROM state was sampled one cell INTO its own corona, so drops arrived pre-charged and got "collected" below onset.

**Findings vs the ROM (partner-baseline: 30 cm tube, 9 cm discs of 0.3 mm, 10 cm spacing, 15 discs, 2 g/m^3 of 15 um mist, 2.1 m/s)**:
- Onset is **~33 kV** (grid-converged 33.0-33.5), not the ROM's 13.6 kV. At the ROM baseline's 30 kV this geometry does not corona at all: ROM says 97% of mist captured, EHD says 0.3%.
- ROM corona current is 4-45x too high (45 kV: 3.66 vs 0.79-1.23 mA).
- Mist quenches the corona: 20-49% current reduction in the first period (2 to 5 g/m^3), recovering downstream as mist is collected.
- Collection is strongly size- and margin-dependent near onset: 36 kV leaves 13% of 6.7 um drops (medium grid); 38 kV / 4 m/s / 0.6 m collects 33% overall. Far above onset (45 kV) everything saturates at ~100% in < 1.5 m.
- Grid: coarse tier (8 cells/edge radius, 2 mm) within ~8% of medium on current near onset, ~2% at 45 kV.
- Grid uncertainty of the medium tier itself, disc unit cell at 40 kV: 48.5 / 46.0 / 51.6 uA per period (10 cells-1 mm / 20 cells-1 mm / 10 cells-0.5 mm) -- about +-6% on current. The 95k-cell grid needed two more active-set fixes (reset the pseudo-time step for each new active set; re-admit one link at a time).

**Implication**: the partner's real disc edge geometry (and sharpness/serration) is a first-order input, and their prototype I-V curve is the single most valuable calibration datum.


### Fully resolved tier + candidate campaign (2026-10-21)

Direct user request: raise simulation fidelity and run many candidates. Built `stonelab.spray_cfd` (axisymmetric RANS + Lagrangian spray + Eulerian mist/fog + implicit wall film), Vega tier `scripts/watercap_resolved.py` (spray CFD -> radial-profile handoff -> `ehd_particles.march_section(parcels=...)`), campaign driver `scripts/watercap_campaign.py`. The ROM gained an atomizer `spray_override` (lognormal drops, charge as a fraction of Rayleigh); the EHD tier now takes Peek's air density from the inlet T (35 C: delta 0.95) and disc thickness as an input.

**Why the spray stage needed a flow solve:** radial mixing time ~R^2/(5.8 D_t) ~ 1-3 s against ~1 s residence, so "well mixed" is not a given; and a useful spray carries as much momentum as the air (2 L/min at 10 m/s: 0.33 N vs 0.37 N). A prescribed-profile version converged to a state the spray never reached (core annulus starved). With momentum coupling: reverse flow -4.5 m/s below the nozzle.

**Validation:** Colebrook friction <3%; Gnielinski wall heat transfer (entrance-corrected) within 0.8-2.4% at Re 2e4-8e4; exact closure without spray; 8 pytest cases. **Independent OpenFOAM-10 cross-check** (3D, 78k cells, reactingFoam + kOmegaSST, `openfoam_cases/watercap_spray_validation/`): vapor removed 0.106 vs ours 0.104 (ceiling 0.104), outlet T 33.09 vs 33.14 C. Disagreements: OF reverse flow -7.7 m/s (ours -4.5: mixing length vs SST), OF puts 24% of spray on the wall (ours ~0). NOTE: OF-10's `liquidEvaporation` clips condensation to zero unless `condensation true` (default run: 0.000 removed, 147% RH outlet) -- any stock OpenFOAM spray-condenser result must be checked for this.

**Numerics found, not assumed (each measured):** relaxing the Patankar reference state pins the gas (~1%/iteration); cell-local linearization converges at ~G/(G+B)/iteration in a dense core (60+ its); Anderson on (h,w) erratic; DEFECT CORRECTION (Eulerian drop-temperature transport from the trace's own cell-to-cell capacity fluxes, solved with the gas, Lagrangian sources as the correction) converges in ~10. Mist enthalpy must not ride in gas h (settling mist piled up with too little enthalpy, T=3 C, S=7). Mist trapped in 30 um sublayer cells (thinner than the drops) grew without bound next to cold walls -> interception + Wood turbulent deposition into the film. Film energy must linearize the evaporating mass flux live (lagged Gout gave film T = -62 C). These three were caught by the campaign's first ~70 runs (17% failures); those runs are kept as `output/watercap_campaign/*_suspect` and were restarted.

**Physics findings so far:**
1. Charged spray is NOT simply bad at resolved fidelity: 5% Rayleigh charge puts all spray on the wall as a cold falling film that keeps condensing (11.0% vapor removed vs 10.4% neutral), with zero carryover to the corona. The ROM counted wall-deposited spray as lost. Campaign example charged_spray-0053: ROM-spray+EHD 0.853 outlet, resolved 0.728. Film result assumes full wetting (optional `film_min_wetting` exists, default off) -- optimistic, to be bounded.
2. Neutral entrained spray all reaches the corona; dense carryover quenched corona current by 72% (cold_spray-0180), though collection still saturated.
3. A passively cooled wall (U=50 W/m2K to 15 C ambient, no spray) removes 6.2% of vapor.
4. Recooling the spray dominates energy: below ambient+5 K it needs a chiller; total specific energy 61-145 Wh/L vs 6-28 Wh/L electrical+fan alone. `specific_energy_total_Wh_L` reports it.
5. Campaign v1 (1087 designs, ROM spray + EHD corona): corona-only saturates at outlet 0.952 (all mist, no vapor); spray designs ranked essentially by chilled-water L/G and temperature (best 0.28-0.43 at L/G > 1, where the ROM is least trustworthy).

**Running:** `v2_resolved` (608 new-concept designs: ring_spray, coflow_spray, multi_tier, charged_film, cooled_wall) and `v2_rerun` (462 non-arcing v1 spray designs at the resolved tier, for ROM-vs-resolved ranking). Logs `logs/watercap_campaign_v2*.log`; `watercap_campaign.analyze()` summarizes.


### Watercap Workbench: in-browser simulator + precomputed-design browser (2026-10-21)

Direct user request: a WebGPU artifact where the partner sets the parameters on their list and gets the targets, plus a Vega-style browser of the precomputed designs. Artifact https://claude.ai/artifact/MaXSthnMH9WWHwk4joeijT (supersedes the Design Explorer EwswydsocH781api5yRXWB). Source `watercap_web/` (wcmodel.js float64 CPU model, wcgpu.js WGSL port, corona_fit.json, workbench.src.html, build.py); data snapshot `scripts/export_watercap_workbench.py` -> `watercap_web/designs.json` (1,685 runs at export: v1 1084, v2_rerun 298, v2_resolved 303; re-run export + build.py + republish as the campaign finishes).

Model changes: spray generalized from 2 classes (main + satellite) to 4: lognormal equal-volume bins with per-bin Rayleigh-fraction charge when GSD > 1 (identical to the ROM's atomizer `spray_override`, n_bins=4), launch speed an input (was fixed 2 m/s). Legacy 2-class path bit-identical (max diff 0 on 60 random designs). GPU outputs now include wall field and onset. Electrospray mode = cone-jet scaling ported line for line (flow/nozzles/conductivity/voltage -> drop size, charge, current; spray V*I added to power, V vs sparkover(R) to arc margin). Corona can be set by current (closed-form inversion of the refit clean-air I-V).

Validation: browser model (fast-model corona) vs the Python ROM on 1,050 v1 designs (water/air <= 1): total water out median |err| 5.5e-5, p90 1.6e-3 (was 3.3e-4 / 6.1e-3 with the 2-class spray). GPU f32 vs CPU f64 p90 2.4e-4. Vs stored rows with refit corona: v1 (fast spray + resolved corona) within 2 pts on 85%; fully resolved within 2 pts on only 34% (median 5.8 pts). By family (signed median, browser - resolved): coflow +0.06, ring +1.0, multi-tier +1.4, cold spray +0.6 pts, but charged film +6.6, charged spray +5.2, cooled wall +4.9, rain curtain +24 -- the browser model has no wall film, no wall cooling, no spray-driven recirculation, so it is pessimistic exactly where the resolved tier found extra capture. 123 v1 designs blow spray out the top in both models (total out > 100%); excluded from the point-error stats and counted separately.

**Viewer spike (2026-10-21):** `viewer/src/pages/watercap.astro` (served at /watercap by the vega-viewer container) + `scripts/export_watercap_viewer_data.py` -> `viewer/public/data/watercap/`. All ~1,450 campaign designs (resolved tier where available) in a concept-filterable list and a per-concept strip chart; 10 curated designs (best per concept, the baseline stack at 40 kV, the thin/large/wide disc geometry at 30 kV) carry 3D scenes: spray parcel paths recorded from the resolved spray trace (`SprayCFD.trace(record=N)`, new optional arg), mist paths traced through that design's resolved corona field (no mist coupling; display only), spray-stage T / saturation / speed / spray-density fields, and the E-field of one disc gap. Not linked from the other pages' nav yet (those files carry other sessions' uncommitted edits).


### The resolved tier in the browser: TS port of stonelab.ehd + ehd_particles + spray_cfd (2026-10-21)

Direct user request: the full physics of the multi-minute resolved run in the browser. Ported (not approximated) to TypeScript in StoneLab worktree `~/Projects/StoneLab-ehd`, branch `ehd-web` (uncommitted): `packages/physics/src/ehd/` (sparse.ts Gilbert-Peierls LU, frontal.ts multifrontal LU on a nested-dissection tree, linsolve.ts, electrostatics.ts, corona.ts, flow.ts, particles.ts, section.ts) and `src/spray/` (thermo.ts, cfd.ts, trace.ts, watercap.ts = Vega's build_cfd/corona_parcels/simulate). Worker entry `apps/watercap/worker.ts`, bundle `apps/watercap/dist/watercap-worker.js` (86 kB), inlined by `watercap_web/build.py`. Workbench v2 runs it in a Web Worker ("Run the resolved model"; designs tab "Re-run at full fidelity here").

Validation (all in `packages/physics/test/ehd|spray`, fixtures from the Python): coaxial Laplace + Kaptsov exact cases; disc unit cell (3 geometries) current/phi/rho to 1e-12 rel vs stonelab.ehd; Pauthenier/well-mixed/Deutsch; coupled march within the Python march's 4-seed spread; spray CFD with no spray reproduces Python to 1e-8..1e-7 (eps x cond floor of the graded grid); with spray inside the 3-seed spread (charged_film-0029, ring_spray-0063, cooled_wall-0000); end to end vs stored campaign rows: charged_film-0029 outlet 0.8062 vs 0.8080, current 1.5149 vs 1.5150 mA, no-mist current exact; ring_spray-0063 0.92176 vs 0.92180. In-browser re-run of charged_film-0029: 50 s (Python 37 s).

Speed: 66% of a run is dense front factorization (82 Newton factorizations for a 9-period march, ~0.75 s each at ~0.5 Gflop/s in JS); parcel tracking is ~4%, so WebGPU particle kernels are not the lever. Next: worker-parallel subtree factorization (~3x on the critical path), WASM SIMD dense kernel (~2x), both exact.

Bugs found in the Python reference along the way (not fixed there; files are untracked in ~/Projects/StoneLab):
1. tests/test_ehd_particles.py::test_pauthenier_exact_update is vacuous: pytest.approx's default abs=1e-12 dwarfs the 1e-15 C charges. field_charge_step actually returns 3.90/6.28/30.0x the exact field charge at t = 0.1/1/10 tau (explicit, unbounded diffusion-charging term over a long step). The field part is exact; the claim "Pauthenier exact" in the EHD section above holds only for it.
2. spray_cfd runs away for a dry cooled wall with no spray (cooled_wall-0000 with spray_flow=0, U=35 W/m2K): outlet w 0.0328 -> 0.0049 in 4 outer iterations (86% of the vapor 'removed'), NaN by 40. Every campaign cooled_wall row has spray, so stored results are unaffected; finding 3 above (6.2%, U=50, no spray) came from a different setup.
3. The well-mixed test's fixed 6% per-bin criterion fails ~1 seed in 10 in the centre bin (counting noise ~3.6%); the port uses a 4-sigma counting criterion.

---

*Full section-by-section history, every exact number, every reverted idea: `EXPERIMENT_LOG_ARCHIVE.md`.*

## Watercap: in-browser full-model compute (2026-10-22)

The /watercap viewer can now solve any design, or an edited one, at the campaign's `resolved_coarse` tier inside the browser tab. Nothing is precomputed.
- **How it works:** `viewer/public/watercap/compute-worker.js` runs Pyodide 0.27.7 (CPython + numpy 2.0 + scipy 1.14 in WebAssembly) in a web worker. It executes the *same* Python, `scripts/watercap_browser.py` → `watercap_resolved.simulate` + the exporter's `scene_from_solve`. The sources are served from `data/watercap/pybundle.json`, rebuilt with `python3 scripts/watercap_browser.py --bundle`.
- **Why not a WebGPU port:** a profile of one design (20 s native) shows about 50% in SuperLU sparse direct solves on about 3k cells and about 30% in the parcel tracer. Neither gains anything from a GPU at this size, and a port would drift from the reference.
- **Parity:** `charged_spray-0100` in Chrome reproduces the campaign record on all 16 outputs to the exported precision, in 72 s (57 s solver). Node+Pyodide gives the same result in 79 s, and native takes 41 s.
- **Porting fix:** wasm32 has a 32-bit `intp`, so `np.bincount` rejected the int64 class-index arrays. `spray_cfd.py:787` and `spray_tower.py:607` now use `np.intp`. Native behavior is unchanged; StoneLab's 8 spray-CFD tests pass.
- **Page:** the design parameters are editable. "Compute full model" re-solves the selected design and checks it against the campaign record. "Compute edited design" adds a new `<id>·edit<n>` row with a 3D scene and its own dot in the strip chart.
- **Inputs:** `design_inputs.json` holds the full-precision campaign inputs, because the 4-digit rounding in `designs.json` would break parity.
- **Timing:** `o.solve_s` (campaign wall time) drives a time hint, since some designs (e.g. `charged_spray-0161`, 1,269 s in the campaign) take tens of minutes.
- **Example edit:** `charged_spray-0100` with the spray at 17.9 °C instead of 13.9 °C recovers 10.6 L/h instead of 12.5 L/h, and the all-in energy drops from 131 to 108 Wh/L.

### Resolved tier: parallel factorization (2026-10-22)

StoneLab-ehd `ehd-web` 1f0749d. The corona's sparse LU now runs on a pool of
workers (parfrontal.ts): assembly subtrees below a cut go to one worker each,
the fronts above are list-scheduled by critical path. Same per-front kernel as
the serial path -> bitwise identical results (checked end to end). No
SharedArrayBuffer on claude.ai (not cross-origin isolated), so it's message
passing and the corona chain is async.

- Newton system (36k): 550 ms serial -> 191 ms (4 workers) / 162 ms (6); 8 is no
  better. Bound: critical path 0.187 of 0.864 GFlop (4.6x); one 544-wide front is 10%.
- The Jacobian pattern changed every Newton step (SparseBuilder dropped exact
  zeros of upwinded terms): 69 symbolic analyses per run -> 7 with keepZeros.
- charged_film-0029: Node 46.2 s -> 23.5 s (6 workers); browser 50 s -> 30 s.
  Python 37 s.
- Rail default design (4 L/min, 60 um spray, 2.1 m/s air): 95 s -> 67 s in Node;
  particle tracking is ~35% of the main thread there (next lever, exact with the
  per-parcel streams if the reductions keep a fixed order).

Physics finding, not a port bug (Python agrees: 1137% vs TS 1152%, current 0 vs
0, no-mist 0.80427 mA both): at the Workbench's default rail design the spray is
entrained up into the corona, quenches it completely (0 current in all 15
periods) and leaves the top -- "total water out" 11x the inlet water. The fast
model says 72%. Another instance of the fast/resolved gap at high spray loading.

Also fixed: sparse.test.ts's corona-sized system was singular (two border
constraints drawn on the same cell, seed 2); it had been failing.

## Watercap: one simulation for sweeps and the browser (Python + wasm-compute-kit) (2026-10-22)

- **Decision (user):** the browser runs the campaign's own Python, not a TS port. The Workbench's resolved tier is now Pyodide running `scripts/watercap_browser.workbench()`, which calls `watercap_resolved.simulate` and then `stonelab` spray_cfd/ehd/ehd_particles. Pyodide 0.27.7 with numpy/scipy is served as supporting files of the artifact (`watercap_web/build.py --pyodide-from`).
- **Solver:** every StoneLab grid solve (SuperLU before) goes through `stonelab/gridsolve.py` to wasm-compute-kit's `python/wck`, a multifrontal LU on the grid's nested dissection.
  - The same Rust runs natively (ctypes, SSE2) and in the browser (kernels.wasm). Solutions are bitwise equal between the two, and with StoneLab's TS FrontalLU.
  - The solver embeds patterns into a superset it has already analysed, because scipy drops zeros. That gives 57 analyses for 1,205 factorizations.
  - A global cache budget of 192 MB keeps browser memory at about 340 MB.
- **Parity:** a stored design re-run reproduces its campaign record on every exported output. Checked: charged_spray-0100, charged_film-0029 and ring_spray-0063, natively, in Node+Pyodide and in Chrome. Native and browser runs of the whole program agree to about 1e-8 relative; numpy's exp/log differ between the builds.
- **Speed:**

  | Run | SuperLU (before) | wck (now) |
  |---|---|---|
  | charged_spray-0100, native | 41 s | 27.8 s |
  | charged_spray-0100, Node+Pyodide | 79 s | 53 s |
  | charged_spray-0100, Chrome | 72 s | 40 s |
  | charged_film-0029, Chrome | | 31–34 s (the TS+wasm port took about 13 s) |
  | ring_spray-0063, Chrome | | 46 s |

## Watercap: corona particle tracking in Rust; a power undercount in the section march (2026-10-22)

- **Kernel:** `track_period` now runs wasm-compute-kit's `ehd_particles.rs` (native .so or kernels.wasm) whenever it can. The Python stays as `track_period_reference` (STONELAB_PARTICLES=python).
  - Each parcel is stepped to its fate on its own xoshiro stream, and libm provides the transcendentals.
  - Against the reference: the charging step matches to 1e-12, and wall deposition and carried space charge agree statistically (tests/test_ehd_particles.py). Paths differ by Monte Carlo.
  - Speed on charged_spray-0100: native 27.8 → 19.1 s, Node+Pyodide 53 → 37 s. Native and wasm agree to about 1e-8.
- **Bug fixed:** `march_section` broke out of the period loop once no parcels were left, and never counted the corona current of the remaining periods.
  - On charged_spray-0100 only 7 to 9 of 35 periods were marched: current 2.64 → 10.30 mA, power 135 → 529 W, energy 12.2 → 43.7 Wh/L.
  - In the stored campaign, 474 of 1,383 designs have Ima more than 20% below (1 − quench)·Ima0 (v1 392, rerun 29, new 53). Their electrical power and Wh/L are understated, so the optimizer was biased toward designs that collect the mist early.
  - Stored records need a re-run before energy numbers are quoted.

## Watercap: spray trace in Rust, a worker pool, and iph.so/newwater (2026-10-22)

- **Spray kernel:** `SprayCFD.trace` now runs wck `spray_trace.rs`, with the droplet thermodynamics ported. The Python stays as `trace_reference`, and a design run agrees with it within Monte Carlo (outputs within about 0.4%; water closure within the reference's seed-to-seed range of −1e-4 to 9e-4).
- **Parallel chunks:** both particle kernels split the parcels into 32 fixed chunks, combined in order, so the thread count never changes the bits (tested).
  - Native: ctypes threads, opt-in with WCK_THREADS.
  - Browser: a pool of helper workers, each with its own kernels.wasm, and results through a SharedArrayBuffer. This needs cross-origin isolation.
- **Timing, charged_spray-0100:**
  - Native: 41 s (morning) → 15.8 s on one thread, 14.7 s on 8.
  - Chrome, isolated page with the 8-thread pool: 22 s, was 40 s.
  - charged_film-0029: 16 s (the TS port took 13 s). ring_spray-0063: 27 s.
  - Browser-to-native ratio is about 1.35×, was 2×.
- **Hosting:** iph.so/newwater, behind Cloudflare Access through a fail-closed Pages Function, with COOP/COEP. Built by `watercap_web/build.py --site`.

## Realism spike: disc wakes, ionic wind, turbulent deposition (2026-10-22)

The roadmap is in StoneLab `docs/realism_roadmap.md`: audit, validation ladder, technology shortlist, OpenFOAM comparison.
- **Ionic wind:** the corona's force builds 10–100 Pa across the gap. But in a resolved disc-cell flow (new `stonelab/ehd_flow.py`), it changes mist collection by only about 1.5 points; pressure balances most of it.
- **Disc wakes matter much more.** Resolving the separated flow behind the discs cuts fine-mist (1–5 µm) collection by 16% (relative) on a sparse stack and roughly halves it on dense stacks of large discs (cold_spray-0126: 5 µm 49% → 28.5%). The fast model and the library over-predict collection there. Next is an OpenFOAM k-ω SST/LES cross-check of the cell.
- **Liu & Agarwal deposition:** StoneLab predicts zero, against measured V_d⁺ of 1e-4 to 0.15. Inertia-free mist can't reach the wall. Needs inertial drops with a Langevin model of the turbulence.

## watercap: realism push 2 — physics with a clear path, and a coupling bug (2026-10-22)

Details: StoneLab `docs/realism_roadmap.md` (Experiments 2–3, rungs 1–2, the Experiment 1 correction).

- **Inertial drops** (Langevin fluid velocity seen, lift, Brownian): wck `track_inertial`, native and wasm.
  - Liu & Agarwal within 2× for τ⁺ ≥ 8; low for τ⁺ 0.5–4 (the Gaussian limit).
  - Disc impaction becomes first order on dense stacks: up to 56% of the corona-section liquid lands on the discs.
- **Humidity mobility** (`stonelab.corona_air`, calibrated to Nouri 2011, validated on Ait Said 2019): corona power −4 to −26%.
- **Resolved disc-cell flow** with a Newton–Krylov solver: 4–30 s instead of 400–1,700 s. SIMPLE had falsely converged on ring_spray-0017 (G 70% low).
- **New `resolved_coarse_plus` / `resolved_plus` tiers** carry inertia, humidity mobility, growth, the resolved cell flow and self-consistent coupling. The plain tiers are unchanged.
- **Bug: corona-mist coupling.** Three fixed relaxed iterations could stop on the "quenched" side of a swing.
  - **129 of 990 stored resolved designs** showed a dead corona and outlet water above the inlet because of it. With the fix (`march_section(adaptive_iters)`), their median outlet goes from 6.5 to 0.86.
  - None beats the library's best, so the top rankings are unaffected. The library has not been regenerated yet.

## Watercap: the partner's real electrode, and how far the model is from it (2026-10-27)

Partner feedback (Brad) with a STEP of the emitter (Onshape AP242). Files + parse/onset scripts: `partner/brad_2026-10-27/`.
- **Device:** 150 mm ID grounded SS tube, 350 mm long; 8 plates at 31.25 mm pitch (stack 219 mm, centred); positive polarity, ~12 kV (30 kV supply at ~40%). Bench fan ~130 m³/h (~2 m/s, residence in the stack ~0.11 s).
- **Plate (from the STEP):** 0.5 mm sheet, an annular ring OD 90 / ID 70 mm (open centre), 3 x Ø4.3 tie-rod holes at r = 40 mm, 6 hooked arms ending in tabs bent 40° out of plane; the tab corners are the emitters, at r = 55.7 mm (gap to wall 19.3 mm), 4.4 mm above the plate.
- **Measurement:** 10.63 g/min leaving without corona, 3.73 with: 65% reduction. At 130 m³/h that is 4.2 g/kg, below any ambient humidity, so it must be liquid only (or the flow differs) -- ask. Deutsch-equivalent migration velocity 0.37 m/s.
- **The model cannot represent this device yet.** Axisymmetric smooth-disc onset (ionization criterion, 30 C, 25 g/m³): plate as a Ø90 disc 28.8 kV; as a Ø111 disc out to the tips 23.1 kV. So the current tiers predict zero corona and zero capture at 12 kV. An isolated hyperboloid tip at 19.3 mm: 4.8 / 6.0 / 8.5 / 11.6 kV for tip radius 0.05 / 0.1 / 0.25 / 0.5 mm (neighbour shielding will raise these). The tips are the whole device; they need the 3D unit cell (60° sector x one pitch).
- **Core bypass hypothesis:** the r < 35 mm core inside the rings is near field-free and carries 25% of the flow (1/7-power profile); radial mixing time (~1 s) >> residence (0.11 s). Most of the 35% that escapes may be core flow. Testable: blank the centre or measure the outlet radially.
- **Seawater ceiling:** exhaust saturated 27 C (Jan) / 34 C (Aug), seawater at 20 m 15.2 / 22.3 C: vapour ceiling 52% / 51% of the exhaust vapour if cooled to seawater itself, 42% / 41% through a closed fresh loop with a 3 K exchanger approach (`watercap_site.py` prints every month: 32-49%). Fresh water must stay in a closed loop (direct seawater contact contaminates the product).

### Seawater recooling and the pilot site (2026-10-27)

- `scripts/watercap_site.py`: the partner's monthly table as aux (`site_aux(month, intake="20m", base=...)`): saturated exhaust at the tube inlet, ambient, seawater at the intake depth. Exhaust T is partner data only for Jan (27 C) and Aug (34 C); other months are our linear interpolation in the ambient wet bulb, flagged `exhaust_interpolated`.
- `watercap_resolved`: aux `seawater_T_C` adds seawater recooling options. Spray >= sea + 3 K: seawater pumping only (20 m head, 5 K rise, eff 0.7, ~0.014 W per W rejected) plus the fresh loop's exchanger loss (50 kPa). Colder: chiller with its condenser on seawater. The cheapest option is taken (`recool_mode`). Design key `wall_coolant="seawater"` cools the wall at sea + 3 K and charges the wall heat's pumping (`wall_cooling_power_W`). StoneLab `spray_cfd` now exports `wall_heat_W` (one additive outlet key). Without `seawater_T_C`, every number is unchanged (the stored library stays reproducible).
- Check, cooled_wall-0070 at resolved_coarse (spray 2.2 L/min, wall U 60):
  | case | recovered L/h | recool mode | recool W | wall pump W | Wh/L total |
  |---|---|---|---|---|---|
  | stored aux (35 C exhaust, 25 C ambient) | 5.47 | chiller_air | 267 | 0 | 216 |
  | Aug site, spray 18.3 C | 4.72 | chiller_seawater | 293 | 0 | 255 |
  | + seawater-cooled wall | 5.07 | chiller_seawater | 291 | 14 | 240 |
  | + spray at sea + 3 K (25.3 C) | 4.00 | seawater_hx | 19 | 14 | 237 |
- **Finding:** with seawater-only recooling the whole thermal side (recool + spray pump + wall pump) is ~38 W, < 10 Wh/L; 899 W of the 237 Wh/L is the corona (this library design's ~40 kV disc operating point). Once seawater is in, energy per litre is a corona-power question, i.e. set by the electrode model. The seawater wall rejects ~1 kW for 14 W of pumping.
- Plant constants (approach, head, rise, exchanger loss) are order-of-magnitude placeholders, not a design.

## Watercap: the partner's literature, reviewed (2026-10-28)

Brad sent 38 DOIs (Cremaschi/Auburn, Reznikov, modified Kelvin-Thomson, corona condensation, MD). None had been read before; finding 1 rested on the classical bound plus Auburn's thesis. Four parallel reviews, per-paper notes stating what was actually read, synthesis in `partner/brad_literature/README.md` (uncommitted with the rest of `partner/`).

- **Finding 1 holds.** No paper shows net vapour removal from sub/saturated air without a cold sink or supersaturation. All sub-saturation theories use a dipole term added to Kelvin-Thomson (fitted to 1-10 molecule clusters; double-counts surface vapour enrichment on um drops); even at face value it gives <=1e-4 at corona-achievable charge. Auburn measured 0.6-3.1% vs 1.7% uncertainty; their 2026 growth law has no evaporation term (predicts condensation at 50% RH). Both scripts re-run and reproduce.
- **Partner's "asymptotic curve" quantified:** single-ion clusters hold ~9-100 H2O, 260-2950x latent heat at 12 kV; a 1 mA corona of such clusters is ~1e-3 g/min vs 6.9 g/min removed in his rig.
- **"Corona intensifies what's already happening" is what the literature shows:** cold walls (+16-57%, vanishing as cooling rises), nozzle expansion (ions reach but never exceed equilibrium wetness), steam-jet mixing. Ion nucleation needs S ~4 (Wilson 1897).
- **Steam-vent test bed** sits at S ~3-11, the ion-nucleation regime; cooling-tower plumes at ~1.0-1.4, so vent results will overstate tower performance. Model would need an ion-nucleus class (S >~3), steam-jet mixing with corona suppression, and an EHD multiplier at sub-dew-point walls.

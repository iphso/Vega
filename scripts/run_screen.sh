#!/bin/bash
# Wide, shallow screen across many basins -- then refine only the promising ones.
#
# WHY. Across the 10 basins run so far, the total reduction achieved varied from
# 4% (d0) to 69% (d4), with striking individual exceptions to the trend: d0
# started best (0.0965) and moved least; d4 started at 0.2324 and finished best
# (0.0723).
#
# CORRECTION (2026-09-30): an earlier version of this comment claimed starting
# violation does NOT predict final violation. That was generalized from the two
# outliers above and is wrong. Across the completed 54-basin screen the
# correlation between starting and final worst-violation is +0.73 -- starting
# point is the DOMINANT predictor. Diverse seeding still pays off, but through
# the residual variance around a strong trend, not because no trend exists.
#
# But it IS visible EARLY: d1 had already dropped 4.6% by iteration 5 and
# finished at 51%; d0 was flat from the start and finished at 4%. So a short
# run discriminates basins at ~1/6 the cost of a full one.
#
# Deep refinement of already-converged designs is exhausted: the restart ladder
# re-ran d4 twice from its own output and returned 0.072328 both times, exactly
# unchanged. Additional depth on the leaders buys nothing; breadth is the only
# remaining lever.
#
# Seeds (output/screen_seeds.npy, 54 of them) are stratified across nfp 1-5 and
# aspect ratio 5.7-11.9, ranked with a mild preference for hypervolume potential
# (1+L_gradB)*(20-A) rather than violation alone -- the leader d4 sits at only
# 42.8 and several screening seeds reach 111, so a feasible design from a
# high-potential basin would be worth far more than one from d4's.
set -u
cd "$(dirname "$0")/.." || exit 1

ITERS=${ITERS:-14}
N=$(python3 -c "import numpy as np;print(len(np.load('output/screen_seeds.npy')))")
echo "[screen] $N seeds x $ITERS iterations"

for i in $(seq 0 $((N-1))); do
  echo "[screen] === seed $i / $((N-1)) ==="
  docker compose run --rm --name "vega-sc-$i" train scripts/minimax_p3.py \
      --tag "s$i" --seed-npy /work/output/screen_seeds.npy --seed-row "$i" \
      --delta0 0.02 --shrink 0.7 --grow 1.6 --max-iter "$ITERS" 2>&1 \
    | grep -aE "minimax\] (best worst|done in|\*\*\*)" | tail -2
done
echo "[screen] ALL DONE"

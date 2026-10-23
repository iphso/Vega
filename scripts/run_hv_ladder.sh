#!/bin/bash
# Deep restart ladder on the HIGH-HYPERVOLUME candidates, not the lowest-violation ones.
#
# WHY THIS AND NOT MORE DEPTH ON THE LEADER.
# The leader (floor2, worst=0.03127) is a confirmed local minimax optimum: floor2
# and floor3 started from the same point with 6x different delta0 and both stalled
# at exactly 0.031. Three constraints are tied there (qi/mirror/fluxcomp), which is
# what active-constraint equalization looks like at convergence. More depth on it
# returns the same number, as the earlier d4 restart ladder already demonstrated
# (0.072328 twice, bit-identical).
#
# And its score ceiling is low. Hypervolume contribution is (1+L_gradB)*(20-A), so
# the leader is worth 33.2 IF it ever becomes feasible -- it bought much of its
# violation reduction by pushing aspect ratio to 12.9, which is the axis the score
# penalises. Meanwhile the screen produced basins at hv 90-100 that are still
# descending when their 14- or 30-iteration budget runs out:
#
#   tag         worst     hv     A      L_gradB  nfp   iters so far
#   rp9_none    0.08408   89.6   11.884 10.044   4     30   <- best joint point
#   rp21_none   0.08251   68.5    ~     ~        ~     30
#   s29         0.17547   92.8   12.750 11.795   4     14
#   rp4_none    0.21135   92.1    9.290  7.600   5     30
#
# A feasible design from rp9 scores 2.7x one from the leader. Both are currently
# worth exactly 0.0, so the only question is which is reachable -- and rp9 needs a
# 8.4x violation reduction where the leader needs 3.1x but has provably stopped.
# Observed reductions across 54 screened basins ran 6-71%, median 30%, so neither
# is comfortable; the difference is that one still has a non-zero gradient.
#
# DESCENT RULE: --hv-mode none, i.e. the reverted default. The hv-floor rule was
# adopted on a single A/B (s32, -36%) and then FAILED to replicate across 6 paired
# basins: 2 wins, 1 tie, 3 losses, median slightly negative. On rp9 specifically
# the floor arm was 6.7% WORSE on violation. What replicates is only that the floor
# protects a little hv (ahead in 5/6, +0.1% to +12%) at a 0-6.7% violation cost.
# Since these seeds already have the hv we want, spend the budget on violation.
#
# LADDER rather than one long run: trust-region collapse is partly a path artifact.
# Resuming from a collapsed design with a fresh trust region re-expands delta and
# re-derives curvature, and has previously found further descent (seed 0: 0.09456
# -> 0.09266 on restart). A design that keeps improving keeps getting rounds; a
# stalled one produces a bit-identical result and the next round is cheap to spot.
set -u
cd "$(dirname "$0")/.." || exit 1

ROUNDS=${ROUNDS:-3}
ITERS=${ITERS:-70}
W=${W:-24}          # 32 cores total, GPU sweep from another project also resident
SRCS=${SRCS:-"rp9_none rp21_none s29 rp4_none"}

echo "[hvl] $SRCS | $ROUNDS rounds x $ITERS iters | $W workers"
for src in $SRCS; do
  host_cur="output/minimax_p3/${src}_best.npy"
  if [ ! -f "$host_cur" ]; then
    echo "[hvl] no $host_cur, skipping $src"
    continue
  fi
  cur="/work/${host_cur}"
  prev="$src"

  for r in $(seq 1 "$ROUNDS"); do
    tag="hl_${src}_r${r}"
    echo "[hvl] === $src round $r (from $prev) ==="
    docker compose run --rm --name "vega-hvl-${src}-${r}" train scripts/minimax_p3.py \
        --tag "$tag" --seed-npy "$cur" --seed-row 0 \
        --hv-mode none \
        --delta0 0.10 --shrink 0.7 --grow 1.6 \
        --max-iter "$ITERS" --n-workers "$W" 2>&1 \
      | grep -aE "minimax\] (best worst|done in|\*\*\*)" | tail -3

    nxt="output/minimax_p3/${tag}_best.npy"
    if [ ! -f "$nxt" ]; then
      echo "[hvl] $src round $r produced no output, stopping this design"
      break
    fi
    cur="/work/${nxt}"
    prev="$tag"
  done
done
echo "[hvl] DONE"

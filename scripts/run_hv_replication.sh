#!/bin/bash
# Replication: does the hypervolume-floor descent rule beat the unconstrained
# rule in general, or only on the one seed where it was first measured?
#
# THE n=1 PROBLEM. The floor rule was adopted on a single controlled A/B from
# seed s32 (start 0.06779, hv 32.4):
#     none : converged 0.050484, hv 29.6, trust region collapsed at it 24
#     floor: reached  0.032098, hv 33.3, still descending at the it-45 cap
# 36% better violation AND 12% better hv, terminating later rather than earlier.
# Decisive on that seed. But this session has already produced two confident
# extrapolations from small samples that were flatly wrong -- a projected "~4%
# per-basin ceiling" (measured range turned out to be 4-71%) and a "geometric
# decay to an 0.040 asymptote" called two iterations before the rate
# re-accelerated 10x. So one seed is not enough to state this as a principle.
#
# DESIGN. Paired runs on N screened basins, chosen to span different starting
# violations, aspect ratios and field periods. Both arms get identical seed,
# budget, delta schedule and worker count; the ONLY difference is --hv-mode.
# Each pair runs sequentially so the two arms see the same machine conditions.
#
# READ-OUT per basin: final worst_violation and final hv for each arm. The claim
# "floor dominates" survives only if it wins on violation in most basins AND
# does not systematically lose hv. A split result (e.g. floor better on violation
# but worse on hv, or better only from high-violation starts) is the interesting
# outcome and should be reported as such rather than smoothed over.
set -u
cd "$(dirname "$0")/.." || exit 1

BASINS=${BASINS:-"9 21 13 4 6"}
ITERS=${ITERS:-30}
W=${W:-26}

echo "[repl] basins: $BASINS | $ITERS iters/arm | $W workers"
for b in $BASINS; do
  src="output/minimax_p3/s${b}_best.npy"
  if [ ! -f "$src" ]; then
    echo "[repl] no $src, skipping basin $b"
    continue
  fi
  for mode in none floor; do
    echo "[repl] === basin $b arm $mode ==="
    docker compose run --rm --name "vega-rp-${b}-${mode}" train scripts/minimax_p3.py \
        --tag "rp${b}_${mode}" --seed-npy "/work/${src}" --seed-row 0 \
        --hv-mode "$mode" --hv-floor-frac 1.0 \
        --delta0 0.05 --shrink 0.7 --grow 1.6 --max-iter "$ITERS" --n-workers "$W" 2>&1 \
      | grep -aE "minimax\] (best worst|done in|\*\*\*)" | tail -2
  done
done
echo "[repl] DONE"

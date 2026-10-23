#!/bin/bash
# PAIRED A/B: does a null-space hypervolume component buy hv at no violation cost?
#
# DESIGNED AS A REPLICATION FROM THE START. The last descent-rule idea (--hv-mode
# floor) was adopted as the default on a single basin showing -36% violation and
# +12% hv, and then failed to replicate: 2 wins, 1 tie, 3 losses across 6 paired
# basins, median slightly negative, and the mechanism I had offered for it turned
# out to conflate two different runs. So this one gets N basins and paired arms
# BEFORE any conclusion, and the read-out is fixed in advance (below).
#
# THE HYPOTHESIS AND WHY IT IS DIFFERENT FROM --hv-mode floor.
# floor and weighted both make hv COMPETE with feasibility inside the LP. The
# measured geometry says they mostly need not compete:
#   - the constraint map has participation ratio 4.2 (top 10 of 81 directions =
#     89.8% of all constraint sensitivity), dominated by m=1 modes
#   - the 35 constraint-null directions still carry 41% of hv sensitivity and 38%
#     of L_gradB sensitivity, but only 0.6% of aspect-ratio sensitivity
#   - directly, on 1224 matched (Jacobian, hv-gradient) pairs recovered from saved
#     Jacobians and stencil evals: ||P g_hv||/||g_hv|| has median 0.687, p10 0.542,
#     and exceeds 0.5 in 94% of iterations, where P = I - J^+ J
# So ~69% of the hv gradient sits outside the span of the five constraint gradients
# and is available at ZERO first-order violation cost. The step adds
# FRAC*||d_LP|| worth of it, which rides the existing line search for free because
# J P = 0 means no predicted violation changes.
#
# WHY IT MATTERS FOR THE SCORE. The benchmark scores hypervolume of FEASIBLE
# designs, (1+L_gradB)(20-A) per point. Our leader got to worst=0.0313 partly by
# raising A to 12.885, leaving hv at 33.2 while rp9 sits at 89.6 -- so the leader's
# own descent mechanism capped its payoff. Aspect-ratio sensitivity in the
# constraint-null subspace is 0.006 of the active subspace, i.e. this component can
# raise L_gradB without paying the A cost that caused that.
#
# ARMS (identical seed, budget, delta schedule, worker count; run back-to-back):
#   off : --nullspace-hv 0.0                          (current default behaviour)
#   on  : --nullspace-hv 0.5 --hv-tie-tol 1e-4
# FRAC=0.5 puts the hv component at half the norm of the LP direction. TIE_TOL=1e-4
# is ~0.3% of a typical worst violation (0.03-0.2) -- tight enough that selecting on
# hv within it is a genuine tie-break and not a disguised trade.
#
# READ-OUT, FIXED IN ADVANCE. Primary endpoint: final hv at equal-or-better final
# worst violation. The claim succeeds only if `on` does not systematically lose
# violation AND gains hv in most basins. Three outcomes are all publishable:
#   - hv up, violation flat        -> the subspace decomposition works as measured
#   - hv up, violation worse       -> it is a trade like floor was, not free
#   - hv flat                      -> ns_retained is high but the directions do not
#                                     integrate into usable finite steps (check the
#                                     logged ns_retained against measured hv delta)
# Report whichever occurs. Do not re-describe a trade-off as free upside.
set -u
cd "$(dirname "$0")/.." || exit 1

# Spans the (worst, hv) Pareto front plus the leader: low-hv/near-feasible through
# high-hv/far-from-feasible, nfp 2 and 4.
BASINS=${BASINS:-"floor2 rp9_none rp21_none s29 s4 s32"}
ITERS=${ITERS:-40}
W=${W:-26}
FRAC=${FRAC:-0.5}
TIE=${TIE:-1e-4}
WAIT_FOR=${WAIT_FOR:-1}

if [ "$WAIT_FOR" = "1" ]; then
  # Do not oversubscribe: both arms must see the same machine conditions, which is
  # the whole point of pairing. Wait for the other P3 drivers to exit first.
  while ps -eo args | grep -qE "[r]un_hv_ladder.sh|[r]un_exhaust.sh"; do
    sleep 120
  done
  echo "[ns] other P3 drivers finished; starting"
fi

echo "[ns] basins: $BASINS | $ITERS iters/arm | $W workers | FRAC=$FRAC TIE=$TIE"
for b in $BASINS; do
  # Seed each pair from the best EVAL of that run, not its _best.npy: 62 of 72 runs
  # contain a stencil point strictly better than their best accepted step, and for
  # floor2 the gap is the project's headline number (0.031271 found vs 0.031804
  # recorded). Both arms share this seed file, so pairing is preserved.
  seed="output/ns_seed_${b}.npy"
  if ! python3 scripts/harvest_best.py --criterion violation --n 1 \
        --glob "output/minimax_p3/${b}_results.jsonl" --out "$seed" >/dev/null 2>&1; then
    if [ -f "output/minimax_p3/${b}_best.npy" ]; then
      seed="output/minimax_p3/${b}_best.npy"
    else
      echo "[ns] no seed for $b, skipping"; continue
    fi
  fi

  for arm in off on; do
    if [ "$arm" = "off" ]; then
      extra="--nullspace-hv 0.0"
    else
      extra="--nullspace-hv $FRAC --hv-tie-tol $TIE"
    fi
    echo "[ns] === $b arm $arm ==="
    docker compose run --rm --name "vega-ns-${b}-${arm}" train scripts/minimax_p3.py \
        --tag "ns_${b}_${arm}" --seed-npy "/work/${seed}" --seed-row 0 \
        --hv-mode none $extra \
        --delta0 0.10 --shrink 0.7 --grow 1.6 \
        --max-iter "$ITERS" --n-workers "$W" 2>&1 \
      | grep -aE "minimax\] (best worst|done in|\*\*\*|it .*ns_retained)" | tail -3
  done
done
echo "[ns] DONE"

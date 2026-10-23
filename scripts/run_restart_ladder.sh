#!/bin/bash
# Iterated restart on the leading designs from the diverse-seed sweep.
#
# WHY RESTART RATHER THAN CONTINUE THE SWEEP.
# Trust-region collapse is partly a PATH artifact, not proof of a stationary
# point. Direct evidence: seed 0 collapsed at 0.09456, and resuming from its
# own output with a fresh trust region found a further 0.0019 before collapsing
# again at 0.09266. Each restart re-expands delta and re-derives curvature.
#
# The 13 unrun sweep seeds start at 0.273-0.383 and would need 74-81%
# reductions to beat the current leader (d4, 0.072328). The best reduction
# observed across every basin so far is 69%. Restarting leaders is the better
# expected value for the same ~17 hours of compute.
#
# Leaders carried forward (best -> worst, with hypervolume potential in
# parentheses, = (1+L_gradB)*(20-A) if the design ever becomes feasible):
#   d4  0.072328  A=11.60 L=4.10  (42.8)
#   d1  0.075132  A=16.11 L=6.65  (29.7)
#   d3  0.098535  A=11.62 L=7.16  (68.4)  <- worse violation, best hv potential
#
# Each round chains from the previous round's best, so a design that keeps
# improving keeps getting refined; a stalled one stops early.
set -u

ROUNDS=${ROUNDS:-4}
cd "$(dirname "$0")/.." || exit 1

for src in d4 d1 d3; do
  host_cur="output/minimax_p3/${src}_best.npy"
  if [ ! -f "$host_cur" ]; then
    echo "[ladder] no $host_cur, skipping $src"
    continue
  fi
  cur="/work/${host_cur}"
  prev_tag="$src"

  for r in $(seq 1 "$ROUNDS"); do
    tag="${src}r${r}"
    echo "[ladder] === $src round $r (from $prev_tag) ==="
    docker compose run --rm --name "vega-rl-${src}-${r}" train scripts/minimax_p3.py \
        --tag "$tag" --seed-npy "$cur" --seed-row 0 \
        --delta0 0.15 --shrink 0.7 --grow 1.6 --max-iter 200 2>&1 \
      | grep -aE "minimax\] (best worst|done in|\*\*\*)" | tail -3

    nxt="output/minimax_p3/${tag}_best.npy"
    if [ ! -f "$nxt" ]; then
      echo "[ladder] $src round $r produced no output, stopping this design"
      break
    fi
    cur="/work/${nxt}"
    prev_tag="$tag"
  done
done

echo "[ladder] DONE"

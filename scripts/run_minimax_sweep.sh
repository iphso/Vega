#!/bin/bash
# Sequential minimax sweep over the stratified diverse seed set, using the
# configuration validated on seed 0 (which descended 0.09653 -> 0.092682):
#   delta0=0.02  -- the working scale; larger values waste iterations shrinking
#   ladder reaching down to fd_step/2 in L2 terms
#   trust-region update keyed on ladder position, not the predicted/actual ratio
set -u
N=$(python3 -c "import numpy as np;print(len(np.load('output/diverse_seeds.npy')))")
echo "[sweep] $N diverse seeds"
for i in $(seq 0 $((N-1))); do
  echo "[sweep] === seed $i / $((N-1)) ==="
  docker compose run --rm --name vega-mm-s$i train scripts/minimax_p3.py \
      --tag d$i --seed-npy /work/output/diverse_seeds.npy --seed-row $i \
      --delta0 0.02 --shrink 0.7 --grow 1.5 --max-iter 200 2>&1 \
      | grep -aE "minimax\] (it +(0|[0-9]*0) |best worst|done in|\*\*\*)" | tail -5
done
echo "[sweep] ALL SEEDS DONE"

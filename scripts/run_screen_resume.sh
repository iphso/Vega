#!/bin/bash
# Resume the 54-basin screen, skipping basins already completed.
# Now benefits from --fd-retry, which recovers high-order poloidal-mode
# gradient columns that diverge at the full fd_step (11/81 were dead at
# deep32's convergence; retry recovers all of them for ~16 extra evals).
set -u
cd "$(dirname "$0")/.." || exit 1
REST=$(cat /tmp/claude-1001/-home-sl8rv/6038fa05-1353-4fbe-9cdd-85b85f1b7172/scratchpad/remaining.txt)
for i in $REST; do
  echo "[screen] === seed $i ==="
  docker compose run --rm --name "vega-sc-$i" train scripts/minimax_p3.py \
      --tag "s$i" --seed-npy /work/output/screen_seeds.npy --seed-row "$i" \
      --delta0 0.02 --shrink 0.7 --grow 1.6 --max-iter 14 --n-workers 15 2>&1 \
    | grep -aE "minimax\] (best worst|\*\*\*)" | tail -2
done
echo "[screen] ALL DONE"

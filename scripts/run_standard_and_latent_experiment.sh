#!/bin/bash
# One-shot orchestrator for two parallel experiments, both retraining on the
# post-5-day-run pool (output/splits/bootstrap_live, 426,777 train rows):
#
#   1. "standard": fresh (non-warm-started) VAE + fresh production-recipe
#      surrogate ensemble (reg_mlp_big_soap's architecture), then a design
#      search checkpoint against the first ConStellaration geometric task
#      (minimize max_elongation s.t. aspect_ratio<=4.0, average_triangularity
#      <=-0.5, |edge_rotational_transform/nfp|>=0.3, nfp=3) -- same recipe as
#      scripts/long_run.py's per-block retrain+search, run standalone.
#   2. "joint": scripts/train_vae_joint.py -- a VAE and the target-metrics
#      surrogate trained together end-to-end (one shared encoder, two
#      projection heads: decode-to-coefficients and predict-metrics, same
#      combined loss every step) instead of a frozen pretrained VAE with a
#      separate MLP bolted on afterward -- otherwise the latent the
#      surrogate sees was never actually shaped by the prediction task, and
#      the comparison against branch 1 doesn't isolate what it's supposed
#      to. Same --split, same per-seed --seed as branch 1, for an
#      apples-to-apples architecture comparison; VAE and surrogate use the
#      same seed as branch 1's corresponding seed but are NOT warm-started
#      from branch 1's VAE checkpoint (that one was trained on the full
#      real+pool dataset regardless of split, which would leak val/test
#      rows into this branch's starting weights).
#
# Only step 0 (branch 1's own standard VAE, needed for its design-search
# step) is shared/serial; branch 2 trains its own VAE from scratch, jointly,
# entirely within its own parallel leg. The two branches run concurrently,
# pinned to separate GPUs, and don't depend on each other.
set -e
cd /work

echo "[run] step 0: fresh VAE on real + full bootstrap pool (branch 1's search-time decoder only)"
python3 scripts/train_vae.py \
  --extra-x output/bootstrap_bootstrap0/X.npy \
  --tag vae_coeffs_full_s0

echo "[run] launching standard branch (GPU 0) and joint-vae branch (GPU 1) in parallel"

(
  set -e
  export CUDA_VISIBLE_DEVICES=0
  for s in 0 1 2; do
    echo "[standard] seed $s"
    python3 scripts/train.py --split bootstrap_live --hidden 2048 --latent 512 --head-hidden 256 \
      --no-spatial --optimizer soap --epochs 300 --val-interval 5 \
      --seed "$s" --tag "reg_mlp_big_full_s$s"
  done
  echo "[standard] design-search checkpoint"
  python3 scripts/latent_optimize.py \
    --minimize max_elongation \
    --constraint "aspect_ratio<=4.0" \
    --constraint "average_triangularity<=-0.5" \
    --constraint "abs(edge_rotational_transform_over_n_field_periods)>=0.3" \
    --nfp 3 --score-bounds 1.0 10.0 \
    --member-tags reg_mlp_big_full_s0 reg_mlp_big_full_s1 reg_mlp_big_full_s2 \
    --vae-tag vae_coeffs_full_s0 \
    --save output/geometric_candidate_standard_v2.json \
    --progress-log output/standard_retrain_progress.jsonl \
    --run-label standard_v2
  echo "[standard] branch done"
) &
STANDARD_PID=$!

(
  set -e
  export CUDA_VISIBLE_DEVICES=1
  for s in 0 1 2; do
    echo "[joint] seed $s"
    python3 scripts/train_vae_joint.py --split bootstrap_live --hidden 2048 --latent 512 --head-hidden 256 \
      --optimizer soap --epochs 300 --val-interval 5 \
      --seed "$s" --tag "vae_joint_full_s$s"
  done
  echo "[joint] branch done (no search checkpoint yet -- architecture comparison only)"
) &
LATENT_PID=$!

wait "$STANDARD_PID"
wait "$LATENT_PID"
echo "[run] both branches finished"

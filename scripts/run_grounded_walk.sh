#!/usr/bin/env bash
# Driver for the grounded-walk round-trip: alternates the GPU step
# (grounded_walk_step.py, in `train`) with real-oracle validation
# (grounded_walk_validate.py, in `oracle-x86`), persisting state as files
# under output/grounded_walk_<tag>/ since each docker compose run is a fresh
# container. Handles the retrain trigger itself (merge real_pool into a
# fresh split, fine-tune, point the walk at the new checkpoint, resume from
# each fiber's last accepted point -- never loses ground).
#
# Usage: run_grounded_walk.sh <tag> [-- extra args forwarded to grounded_walk_step.py]
# e.g.:
#   run_grounded_walk.sh p3_halfsiren_v1 -- \
#     --scorer-tag reg_half_siren_big_soap_full_s0 \
#     --constraint "abs(edge_rotational_transform_over_n_field_periods)>=0.25" \
#     --constraint "edge_magnetic_mirror_ratio<=0.25" \
#     --constraint "flux_compression_in_regions_of_bad_curvature<=0.9" \
#     --constraint "vacuum_well>=0.0" \
#     --constraint "qi<=0.0003162"
set -uo pipefail
cd "$(dirname "$0")/.."

TAG=$1; shift
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

STATE_DIR="output/grounded_walk_${TAG}"
LOG_DIR="/tmp/grounded_walk_logs"
mkdir -p "$LOG_DIR"

MAX_ROUNDS=${MAX_ROUNDS:-300}
MAX_RETRAINS=${MAX_RETRAINS:-10}
RETRAIN_EPOCHS=${RETRAIN_EPOCHS:-60}
BASE_SPLIT=${BASE_SPLIT:-bootstrap_live}
SCORER_ARCH=${SCORER_ARCH:-half_siren}

round=0
retrains=0
while (( round < MAX_ROUNDS )); do
  docker compose run --rm train scripts/grounded_walk_step.py --tag "$TAG" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/${TAG}_step.log"
  step_status=${PIPESTATUS[0]}
  if [[ $step_status -ne 0 ]]; then
    echo "[driver] grounded_walk_step.py exited $step_status -- stopping"
    break
  fi

  if [[ -f "$STATE_DIR/success.json" ]]; then
    echo "[driver] SUCCESS reached, stopping"
    cat "$STATE_DIR/success.json"
    break
  fi
  if [[ -f "$STATE_DIR/STOPPED_min_step_size" ]]; then
    echo "[driver] step size collapsed below floor, stopping"
    break
  fi

  if [[ -f "$STATE_DIR/needs_retrain.flag" ]]; then
    if [[ ! -f "$STATE_DIR/real_pool_X.npy" ]]; then
      # No fiber has EVER had a step accepted -- there is nothing real to
      # retrain on yet, so a retrain here would just reproduce the same
      # checkpoint for real wall-clock cost. Shrink the step size hard
      # instead and try stepping again from the same (never-moved) anchors.
      # NOTE: state files are created by docker containers running as root --
      # this driver runs on the HOST, so it can READ them (world-readable) but
      # cannot WRITE or DELETE them directly (needs write on the containing
      # dir, which is root-owned). Every mutation below goes through a
      # container for exactly that reason -- a host-side `echo >`/`rm -f` here
      # silently no-ops instead of erroring under `set -uo pipefail` (no `-e`),
      # which is what let an earlier run spin 500+ rounds doing nothing at all.
      cur_step=$(cat "$STATE_DIR/step_size.txt" 2>/dev/null || echo 0.01)
      docker compose run --rm preprocess -c "
import os
step = max(float('$cur_step') * 0.3, 1e-6)
open('/work/${STATE_DIR}/step_size.txt', 'w').write(str(step))
os.remove('/work/${STATE_DIR}/needs_retrain.flag')
print(f'shrunk step size {$cur_step} -> {step}')
"
      echo "[driver] no real data collected yet -- nothing to retrain on. Shrunk step size instead."
      continue
    fi
    retrains=$((retrains + 1))
    if (( retrains > MAX_RETRAINS )); then
      echo "[driver] hit max retrain cycles ($MAX_RETRAINS), stopping"
      break
    fi
    echo "[driver] === RETRAIN CYCLE $retrains ==="
    SPLIT_NAME="grounded_walk_${TAG}_r${retrains}"
    docker compose run --rm preprocess scripts/merge_real_pool_split.py \
      --base-split "$BASE_SPLIT" --out-split "$SPLIT_NAME" \
      --pool-x "/work/${STATE_DIR}/real_pool_X.npy" \
      --pool-y "/work/${STATE_DIR}/real_pool_Y.npy"

    NEW_TAG="reg_${SCORER_ARCH}_grounded_${TAG}_r${retrains}"
    SOAP_ARGS=(--optimizer soap)
    if [[ "$SCORER_ARCH" == "half_siren" ]]; then
      SOAP_ARGS+=(--soap-linalg-backend magma)
    fi
    docker compose run --rm train scripts/train.py \
      --trunk-arch "$SCORER_ARCH" "${SOAP_ARGS[@]}" \
      --hidden 2048 --latent 512 --head-hidden 256 --no-spatial \
      --split "$SPLIT_NAME" --epochs "$RETRAIN_EPOCHS" --seed 0 \
      --tag "$NEW_TAG" \
      2>&1 | tee -a "$LOG_DIR/${TAG}_retrain_${retrains}.log"

    docker compose run --rm preprocess -c "
import os
open('/work/${STATE_DIR}/current_scorer_tag.txt', 'w').write('$NEW_TAG')
os.remove('/work/${STATE_DIR}/needs_retrain.flag')
"
    echo "[driver] retrain $retrains done -- resuming with $NEW_TAG from each fiber's last accepted point"
    continue
  fi

  # A step was actually proposed (not blocked by a retrain) -- validate it for real.
  docker compose run --rm oracle-x86 scripts/grounded_walk_validate.py --tag "$TAG" \
    2>&1 | tee -a "$LOG_DIR/${TAG}_validate.log"
  val_status=${PIPESTATUS[0]}
  if [[ $val_status -ne 0 ]]; then
    echo "[driver] grounded_walk_validate.py exited $val_status -- stopping"
    break
  fi

  round=$((round + 1))
done

echo "[driver] loop ended after $round rounds, $retrains retrain cycles"

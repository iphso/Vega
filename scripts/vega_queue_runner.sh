#!/usr/bin/env bash
# Cron-driven queue runner for the vacation compute plan. Two GPU lanes
# (queue_gpu0.txt/queue_gpu1.txt) each work through a list of grounded-walk
# stellarator discovery configs (one nfp per slot); once BOTH lanes are
# exhausted, a shared phase-2 queue (queue_phase2.txt) runs one-shot
# cross-domain bulk generation jobs.
#
# Run via crontab, e.g.: */10 * * * * ~/Projects/Vega/scripts/vega_queue_runner.sh
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="/tmp/grounded_walk_logs"
STATUS_FILE="$LOG_DIR/queue_status.json"
mkdir -p "$LOG_DIR"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
esc() { sed 's/"/\\"/g'; }

MAX_RESTARTS_IN_WINDOW=5
WINDOW_SECONDS=7200

# ---------------------------------------------------------------------------
# GPU lane (phase 1): each slot is a grounded_walk driver invocation.
# Returns via global RESULT_JSON and LANE_DONE (1 if the whole queue for this
# lane is exhausted).
# ---------------------------------------------------------------------------
run_gpu_lane() {
  local LANE="$1" QUEUE_FILE="$2" IDX_FILE="$LOG_DIR/${1}_idx"
  local idx total
  idx=$(cat "$IDX_FILE" 2>/dev/null || echo 0)
  total=$(grep -vc '^\s*#\|^\s*$' "$QUEUE_FILE")
  LANE_DONE=0

  while true; do
    if (( idx >= total )); then
      LANE_DONE=1
      RESULT_JSON="{\"lane\":\"$LANE\",\"state\":\"QUEUE_EXHAUSTED\"}"
      return
    fi

    local line TAG SCRIPT ENVVARS ARGS
    line=$(grep -v '^\s*#\|^\s*$' "$QUEUE_FILE" | sed -n "$((idx+1))p")
    IFS='|' read -r TAG SCRIPT ENVVARS ARGS <<< "$line"

    local STATE_DIR="output/grounded_walk_${TAG}"
    local DRIVER_LOG="$LOG_DIR/${TAG}_driver.log"
    local RESTART_LOG="$LOG_DIR/${TAG}_restarts.log"
    local FAILED_MARKER="$LOG_DIR/${TAG}_FAILED"

    if [[ -f "$FAILED_MARKER" ]]; then
      echo "[$(ts)] [queue:$LANE] slot $TAG previously FAILED -- skipping to next" >> "$LOG_DIR/${LANE}_queue.log"
      idx=$((idx+1)); echo "$idx" > "$IDX_FILE"; continue
    fi
    if [[ -f "$STATE_DIR/success.json" ]]; then
      echo "[$(ts)] [queue:$LANE] slot $TAG reached SUCCESS -- advancing" >> "$LOG_DIR/${LANE}_queue.log"
      idx=$((idx+1)); echo "$idx" > "$IDX_FILE"; continue
    fi
    if [[ -f "$STATE_DIR/STOPPED_min_step_size" ]]; then
      echo "[$(ts)] [queue:$LANE] slot $TAG converged (step size collapsed) -- advancing" >> "$LOG_DIR/${LANE}_queue.log"
      idx=$((idx+1)); echo "$idx" > "$IDX_FILE"; continue
    fi

    if pgrep -f "$SCRIPT $TAG " > /dev/null; then
      RESULT_JSON="{\"lane\":\"$LANE\",\"tag\":\"$TAG\",\"state\":\"RUNNING\"}"
      return
    fi

    if [[ ! -f "$DRIVER_LOG" ]]; then
      # Never launched before -- start it fresh.
      echo "[$(ts)] [queue:$LANE] launching fresh slot $TAG" >> "$LOG_DIR/${LANE}_queue.log"
      # shellcheck disable=SC2086
      nohup env $ENVVARS "./scripts/$SCRIPT" "$TAG" -- $ARGS >> "$DRIVER_LOG" 2>&1 &
      disown
      RESULT_JSON="{\"lane\":\"$LANE\",\"tag\":\"$TAG\",\"state\":\"LAUNCHED\"}"
      return
    fi

    # Driver log exists but process is dead -- was this a controlled exit
    # (hit its own round/retrain budget, or the driver detected a real
    # problem) or an abrupt crash (docker daemon restart, OOM-kill, etc.)?
    if tail -n 8 "$DRIVER_LOG" | grep -qE '^\[driver\] (loop ended|SUCCESS reached|step size collapsed|grounded_walk_step(_v2|_v3|_v4)?\.py exited|grounded_walk_validate\.py exited|hit max retrain)'; then
      echo "[$(ts)] [queue:$LANE] slot $TAG finished its round/retrain budget without a terminal state file -- advancing for diversity" >> "$LOG_DIR/${LANE}_queue.log"
      idx=$((idx+1)); echo "$idx" > "$IDX_FILE"; continue
    fi

    # Abrupt crash -- restart with backoff, same slot.
    local now recent_restarts
    now=$(date +%s)
    touch "$RESTART_LOG"
    awk -v cutoff="$((now - WINDOW_SECONDS))" '$1 >= cutoff' "$RESTART_LOG" > "${RESTART_LOG}.tmp" && mv "${RESTART_LOG}.tmp" "$RESTART_LOG"
    recent_restarts=$(wc -l < "$RESTART_LOG" | tr -d ' ')

    if (( recent_restarts >= MAX_RESTARTS_IN_WINDOW )); then
      local reason="crashed $recent_restarts times in the last $((WINDOW_SECONDS/60))min -- giving up on this slot, see $DRIVER_LOG"
      echo "$reason" > "$FAILED_MARKER"
      echo "[$(ts)] [queue:$LANE] $reason -- advancing to next slot" >> "$LOG_DIR/${LANE}_queue.log"
      idx=$((idx+1)); echo "$idx" > "$IDX_FILE"; continue
    fi

    echo "$now" >> "$RESTART_LOG"
    echo "[$(ts)] [driver] not running, restarting (restart $((recent_restarts+1))/$MAX_RESTARTS_IN_WINDOW in window)" >> "$DRIVER_LOG"
    # shellcheck disable=SC2086
    nohup env $ENVVARS "./scripts/$SCRIPT" "$TAG" -- $ARGS >> "$DRIVER_LOG" 2>&1 &
    disown
    RESULT_JSON="{\"lane\":\"$LANE\",\"tag\":\"$TAG\",\"state\":\"RESTARTED\",\"restart_count_in_window\":$((recent_restarts+1))}"
    return
  done
}

# ---------------------------------------------------------------------------
# Phase 2: one-shot cross-domain generation jobs, run only once both GPU
# lanes report their queue exhausted.
# ---------------------------------------------------------------------------
run_phase2() {
  local QUEUE_FILE="scripts/queue_phase2.txt" IDX_FILE="$LOG_DIR/phase2_idx"
  local idx total
  idx=$(cat "$IDX_FILE" 2>/dev/null || echo 0)
  total=$(grep -vc '^\s*#\|^\s*$' "$QUEUE_FILE")

  while true; do
    if (( idx >= total )); then
      PHASE2_JSON="{\"state\":\"QUEUE_EXHAUSTED\"}"
      return
    fi

    local line NAME SERVICE SCRIPT_ARGS
    line=$(grep -v '^\s*#\|^\s*$' "$QUEUE_FILE" | sed -n "$((idx+1))p")
    IFS='|' read -r NAME SERVICE SCRIPT_ARGS <<< "$line"

    local EXITCODE_FILE="$LOG_DIR/phase2_${NAME}.exitcode"
    local JOB_LOG="$LOG_DIR/phase2_${NAME}.log"

    if [[ -f "$EXITCODE_FILE" ]]; then
      local code; code=$(cat "$EXITCODE_FILE")
      if [[ "$code" == "0" ]]; then
        echo "[$(ts)] [phase2] $NAME completed successfully -- advancing" >> "$LOG_DIR/phase2_queue.log"
      else
        echo "[$(ts)] [phase2] $NAME exited $code -- not retrying, advancing (see $JOB_LOG)" >> "$LOG_DIR/phase2_queue.log"
      fi
      idx=$((idx+1)); echo "$idx" > "$IDX_FILE"; continue
    fi

    if pgrep -f "phase2_run_${NAME}\.sh" > /dev/null; then
      PHASE2_JSON="{\"name\":\"$NAME\",\"state\":\"RUNNING\"}"
      return
    fi

    echo "[$(ts)] [phase2] launching $NAME ($SERVICE)" >> "$LOG_DIR/phase2_queue.log"
    cat > "$LOG_DIR/phase2_run_${NAME}.sh" <<EOF
#!/usr/bin/env bash
cd "$(pwd)"
docker compose run --rm $SERVICE $SCRIPT_ARGS
echo \$? > "$EXITCODE_FILE"
EOF
    chmod +x "$LOG_DIR/phase2_run_${NAME}.sh"
    nohup "$LOG_DIR/phase2_run_${NAME}.sh" >> "$JOB_LOG" 2>&1 &
    disown
    PHASE2_JSON="{\"name\":\"$NAME\",\"state\":\"LAUNCHED\"}"
    return
  done
}

RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu0 scripts/queue_gpu0.txt
gpu0_result="$RESULT_JSON"; gpu0_done=$LANE_DONE

RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu1 scripts/queue_gpu1.txt
gpu1_result="$RESULT_JSON"; gpu1_done=$LANE_DONE

# Third, colocated lane on GPU1: originally an A/B test of proximity-biased
# anchor selection + per-fiber death/respawn (grounded_walk_step_v2.py) --
# retired 2026-09-12 as a confirmed negative (thrashing, see
# EXPERIMENT_LOG.md). Repurposed same day for a quality-diversity/novelty
# bonus (grounded_walk_step_v3.py) -- also retired, 2026-09-13, after its own
# topology instrumentation showed it consolidating MORE than the plain
# control, not less. Now running grounded_walk_step_v4.py: anchored only at
# the project's known qi-passing real points, directly minimizing
# edge_magnetic_mirror_ratio with just qi enforced, to test whether the
# qi-vs-mirror_ratio correlation seen on every qi-passing point found so far
# is a real Pareto wall or a loosenable one. Independent tag/state dir,
# shares the GPU only for compute (each job uses ~5GB of 32GB) -- not gated
# on or gating the gpu1 lane it's being compared against.
RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu1v2 scripts/queue_gpu1_v2.txt
gpu1v2_result="$RESULT_JSON"; gpu1v2_done=$LANE_DONE

# Fourth, colocated lane on GPU1: the same qi-vs-mirror_ratio correlation
# question as gpu1v2's v4 runs, but a third basin (nfp=3) approached from
# the opposite direction -- anchored at near-miss real points with already-
# good mirror_ratio, directly minimizing qi instead. See
# queue_gpu1_v3.txt's header and EXPERIMENT_LOG.md for the full rationale.
# True concurrent job, not sequenced behind gpu1v2 -- cheap (15 fibers) and
# the GPU has headroom for a third job alongside the other two.
RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu1v3 scripts/queue_gpu1_v3.txt
gpu1v3_result="$RESULT_JSON"; gpu1v3_done=$LANE_DONE

# Fifth, colocated lane on GPU0: direct qi-attack, seeded from
# gpu1v3's NAE-seeded run's own best-qi survivors. See queue_gpu0_v2.txt's
# header and EXPERIMENT_LOG.md for the full rationale. True concurrent job,
# not sequenced behind gpu0's main lane -- cheap (15 fibers) and GPU0 has
# full headroom.
RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu0v2 scripts/queue_gpu0_v2.txt
gpu0v2_result="$RESULT_JSON"; gpu0v2_done=$LANE_DONE

# Sixth/seventh/eighth lanes, 2026-09-14: after retiring the 3 confirmed-
# stalled lanes (see EXPERIMENT_LOG.md's stall audit), redeployed the freed
# GPU0+GPU1 capacity into 3 new NAE-seeded pushes mining the original NAE
# pilot sweep's under-used mirror_ratio_target=0.15 configs (much closer to
# the true P3 mirror_ratio wall than the target=0.25 seed used so far -- see
# each queue file's own header for the full rationale).
RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu0v3 scripts/queue_gpu0_v3.txt
gpu0v3_result="$RESULT_JSON"; gpu0v3_done=$LANE_DONE

RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu0v4 scripts/queue_gpu0_v4.txt
gpu0v4_result="$RESULT_JSON"; gpu0v4_done=$LANE_DONE

RESULT_JSON=""; LANE_DONE=0
run_gpu_lane gpu1v4 scripts/queue_gpu1_v4.txt
gpu1v4_result="$RESULT_JSON"; gpu1v4_done=$LANE_DONE

PHASE2_JSON="{\"state\":\"NOT_STARTED\"}"
if (( gpu0_done == 1 && gpu1_done == 1 && gpu1v2_done == 1 && gpu1v3_done == 1 && gpu0v2_done == 1 && gpu0v3_done == 1 && gpu0v4_done == 1 && gpu1v4_done == 1 )); then
  run_phase2
fi

{
  echo "{\"checked_at\":\"$(ts)\",\"gpu0\":$gpu0_result,\"gpu1\":$gpu1_result,\"gpu1v2\":$gpu1v2_result,\"gpu1v3\":$gpu1v3_result,\"gpu0v2\":$gpu0v2_result,\"gpu0v3\":$gpu0v3_result,\"gpu0v4\":$gpu0v4_result,\"gpu1v4\":$gpu1v4_result,\"phase2\":$PHASE2_JSON}"
} > "$STATUS_FILE"

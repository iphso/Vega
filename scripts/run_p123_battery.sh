#!/usr/bin/env bash
# Repeatable "how close does this scorer ensemble get to P1/P2/P3" battery:
# same 3 searches used ad hoc during the curriculum-round-1 exploration,
# packaged so any scorer ensemble (single- or multi-seed) gets run through
# an identical protocol -- P1's own official search, plus the qi-push probe
# for P2/P3 (minimize qi directly, holding every OTHER constraint fixed --
# see EXPERIMENT_LOG: this is a distance/frontier probe, not a claim that
# qi-minimization alone would ever be the real search objective).
#
# Usage: run_p123_battery.sh <label> <n-candidates> <member-tag> [more member-tags...]
# Uses default trust-weight (ensemble disagreement penalty) -- pass a
# single member-tag and it's just 0-variance, not wrong, but a true
# ensemble is the fair comparison against reg_mlp_big_soap_cluster_s{0,1,2}.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL=$1; N_CAND=$2; shift 2
MEMBERS=("$@")
GEN_LOG_DIR=/tmp/battery_logs
mkdir -p "$GEN_LOG_DIR"

echo "[battery:$LABEL] members=${MEMBERS[*]} n_candidates=$N_CAND"

docker compose run --rm train scripts/generate_candidates.py \
  --member-tags "${MEMBERS[@]}" --n-starts 150 --n-candidates "$N_CAND" \
  --save /work/output/battery_${LABEL}_p1.jsonl \
  > "$GEN_LOG_DIR/gen_${LABEL}_p1.log" 2>&1

docker compose run --rm train scripts/generate_candidates.py \
  --member-tags "${MEMBERS[@]}" --n-starts 150 --n-candidates "$N_CAND" \
  --minimize qi \
  --constraint "aspect_ratio<=10.0" \
  --constraint "abs(edge_rotational_transform_over_n_field_periods)>=0.25" \
  --constraint "edge_magnetic_mirror_ratio<=0.2" \
  --constraint "max_elongation<=5.0" \
  --save /work/output/battery_${LABEL}_p2.jsonl \
  > "$GEN_LOG_DIR/gen_${LABEL}_p2.log" 2>&1

docker compose run --rm train scripts/generate_candidates.py \
  --member-tags "${MEMBERS[@]}" --n-starts 150 --n-candidates "$N_CAND" \
  --minimize qi \
  --constraint "abs(edge_rotational_transform_over_n_field_periods)>=0.25" \
  --constraint "edge_magnetic_mirror_ratio<=0.25" \
  --constraint "flux_compression_in_regions_of_bad_curvature<=0.9" \
  --constraint "vacuum_well>=0.0" \
  --save /work/output/battery_${LABEL}_p3.jsonl \
  > "$GEN_LOG_DIR/gen_${LABEL}_p3.log" 2>&1

echo "[battery:$LABEL] generation done, validating against real oracle..."

for problem in p1 p2 p3; do
  docker compose run --rm oracle-x86 scripts/append_oracle_master.py \
    /work/output/battery_${LABEL}_${problem}.jsonl \
    --source-generator vae --source-scorer "$LABEL" --problem "$problem" \
    --fidelity low --n-workers 20 --timeout-seconds 60 \
    > "$GEN_LOG_DIR/val_${LABEL}_${problem}.log" 2>&1
done

echo "[battery:$LABEL] BATTERY_COMPLETE"

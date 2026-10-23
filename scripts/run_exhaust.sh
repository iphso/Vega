#!/bin/bash
# EXHAUSTION PROTOCOL: restart a design until a round returns a bit-identical best.
#
# THE QUESTION: what does it take to "fully exploit" a basin? Until now we never
# had a stopping rule, so every answer about whether a design was finished was a
# guess -- and one of those guesses was wrong in a way that misdirected a day of
# compute (see below).
#
# WHY A FIXED ITERATION BUDGET CANNOT ANSWER IT. minimax_p3.py terminates when the
# trust region collapses below --delta-min, which is a property of the DESCENT PATH,
# not of the basin. Measured over every run with >=25 iterations, a basin reaches
# 90% of its eventual gain only at iteration 19-27, and 4 of 9 such runs were still
# descending at their cap:
#
#   tag         iters  start     end       red%   90%/99% at iter   at cap
#   hv_floor       45  0.06778   0.03209   52.6    27 / 42          STILL DESCENDING
#   deep32         36  0.06773   0.03630   46.4    25 / 31          plateaued
#   rp4_floor      30  0.23620   0.20924   11.4    19 / 29          STILL DESCENDING
#   rp9_none       30  0.11292   0.08627   23.6     9 / 23          plateaued
#   rp21_floor     29  0.09775   0.08446   13.6    11 / 14          STILL DESCENDING
#
# The 54-basin screen used a 14-iteration cap. It was measuring which basins descend
# FAST, and we read it as which basins descend FAR. Those are different quantities.
#
# WHY floor2 IS FIRST. The project leader (floor2, worst=0.03127, the best design in
# ~1.02M candidates) ran TEN iterations before its trust region collapsed. floor3 ran
# fourteen. I previously called floor2 "a confirmed local minimax optimum" and a
# "dead end for depth" on the grounds that floor2 and floor3 both stopped at exactly
# 0.031 from the same start with 6x different delta0. That is evidence the descent
# path reliably collapses its trust region there -- NOT that the point is stationary.
# Both runs were replications of one descent; neither was a restart from floor2's own
# output with a fresh trust region. Meanwhile hv_floor ran 45 iterations from the same
# start region and was still descending at the cap.
#
# THE STOPPING RULE. A restart re-expands delta from delta0 and re-derives curvature
# from scratch, so it can move where the collapsed run could not. If a full round
# starting FROM the previous round's output returns a best that is bit-identical to
# what it started with, the solver genuinely cannot move: same base point, same
# deterministic VMEC++ evaluations (verified bit-for-bit reproducible to 17 digits at
# low fidelity), same stencil, no progress. That is exhaustion, and it is the signal
# deep32b produced for deep32 -- 0.03229475802103021 returned to 17 digits.
#
# Each round also seeds from the best EVAL, not the best accepted step. 62 of 72 runs
# contain a finite-difference stencil point strictly better than their best accepted
# step (by 0.0005-0.0159), because a stencil perturbs one whitened coordinate while
# the minimax step trades all five constraints off against a linearised model. Every
# previous ladder round restarted from a knowingly worse point. harvest_best.py
# extracts the true best; --use-harvest turns that on.
set -u
cd "$(dirname "$0")/.." || exit 1

SRC=${SRC:-floor2}
MAX_ROUNDS=${MAX_ROUNDS:-8}
ITERS=${ITERS:-60}
W=${W:-8}                  # coexists with run_hv_ladder.sh at 24 on a 32-core box
DELTA0=${DELTA0:-0.10}
HVMODE=${HVMODE:-none}

START=${START:-1}          # resume: round START seeds from ex_${SRC}_r$((START-1))_seed.npy

cur_host="output/minimax_p3/${SRC}_best.npy"
if [ ! -f "$cur_host" ]; then
  echo "[exh] no $cur_host"; exit 1
fi
cur="/work/${cur_host}"
prev_best=""

# Round 1 seeds from the best EVAL of the source run too, not its _best.npy. For
# floor2 these differ and it matters: _best.npy (which is also project_best.npy, the
# project's leader of record) holds 0.03180435561496913, while the run actually found
# 0.03127059865990108 at a stencil point 2.1e-5 away. Starting a ladder from the
# recorded best would discard the better design before the first iteration.
if [ "$START" -le 1 ] && python3 scripts/harvest_best.py --criterion violation --n 1 \
      --glob "output/minimax_p3/${SRC}_results.jsonl" \
      --out "output/ex_${SRC}_r0_seed.npy" >/dev/null 2>&1; then
  cur="/work/output/ex_${SRC}_r0_seed.npy"
  echo "[exh] round-1 seed harvested from all evals of $SRC"
fi

if [ "$START" -gt 1 ]; then
  resume="output/ex_${SRC}_r$((START-1))_seed.npy"
  if [ ! -f "$resume" ]; then echo "[exh] no resume seed $resume"; exit 1; fi
  cur="/work/${resume}"
  echo "[exh] resuming at round $START from $resume"
fi

echo "[exh] $SRC | rounds $START..$MAX_ROUNDS x $ITERS iters | $W workers | delta0=$DELTA0"
for r in $(seq "$START" "$MAX_ROUNDS"); do
  tag="ex_${SRC}_r${r}"
  echo "[exh] === round $r (seed: $cur) ==="
  docker compose run --rm --name "vega-exh-${SRC}-${r}" train scripts/minimax_p3.py \
      --tag "$tag" --seed-npy "$cur" --seed-row 0 \
      --hv-mode "$HVMODE" \
      --delta0 "$DELTA0" --shrink 0.7 --grow 1.6 \
      --max-iter "$ITERS" --n-workers "$W" 2>&1 \
    | grep -aE "minimax\] (best worst|done in|\*\*\*)" | tail -3

  # Best over ALL evals this round, printed at full precision: the bit-identity
  # test needs every digit, so no rounding anywhere in this comparison.
  best=$(python3 - "$tag" <<'PY'
import json, sys, os
tag = sys.argv[1]
f = f"output/minimax_p3/{tag}_results.jsonl"
if not os.path.exists(f):
    print("NONE"); raise SystemExit
b = None
for line in open(f):
    try: d = json.loads(line)
    except Exception: continue
    w = d.get("worst_violation")
    if w is not None and (b is None or w < b): b = w
print(repr(b) if b is not None else "NONE")
PY
)
  echo "[exh] round $r best (full precision): $best"
  if [ "$best" = "NONE" ]; then
    echo "[exh] round $r produced no scored evals; stopping"
    break
  fi
  if [ "$best" = "$prev_best" ]; then
    echo "[exh] BIT-IDENTICAL to round $((r-1)) -> basin EXHAUSTED at $best after $r rounds"
    break
  fi
  prev_best="$best"

  # Re-seed from the best eval of this round, which may be a stencil point that the
  # trajectory never accepted.
  python3 scripts/harvest_best.py --criterion violation --n 1 \
      --glob "output/minimax_p3/${tag}_results.jsonl" \
      --out "output/ex_${SRC}_r${r}_seed.npy" >/dev/null 2>&1 \
    && cur="/work/output/ex_${SRC}_r${r}_seed.npy" \
    || cur="/work/output/minimax_p3/${tag}_best.npy"
done
echo "[exh] DONE"

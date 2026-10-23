#!/bin/bash
# SEARCH AT THE FIDELITY THAT SCORES.
#
# WHY THIS REPLACES THE LOW-FIDELITY LADDERS (EXPERIMENT_LOG, 2026-10-05). Every
# leading design in the project -- including `asp_a13.5`, which satisfied all five
# official constraints at low fidelity, and `ex_floor2_r21`, the 1M-eval leader --
# fails the real scorer with `VMEC++ did not converge`. Four dataset designs pass
# through the identical call, so the harness is sound and the boundaries are the
# problem. ~1M evaluations were spent mining the gap between `low_fidelity` and the
# fidelity that scores.
#
# WHY THE FIX IS A FLAG AND NOT A PENALTY TERM. A boundary the scorer cannot
# resolve comes back as a failed eval, which minimax_p3.py already persists as
# `converged: false` and excludes from candidates and from `best`. Searching at
# high fidelity makes the artifact UNREACHABLE BY CONSTRUCTION -- no penalty on
# high-order Fourier content, no resolvability heuristic, nothing to tune.
#
# WHY FD GRADIENTS ARE STILL SOUND HERE. `fidelity_check.py` established 17-digit
# determinism at LOW fidelity only -- it issues `low_a`, `low_b`, `high`, so one
# high-fidelity eval, which cannot test determinism. `fidelity_check_high.py` ran
# the real test on two dataset controls: BIT-IDENTICAL across all 7 metrics at high
# fidelity. No noise floor, so no lower bound is forced on the step size h.
#
# ---------------------------------------------------------------------------
# THE TIMEOUT IS NOT OPTIONAL. `--timeout-seconds` defaults to 300. High-fidelity
# evals measured 442.2 s and 403.9 s. Running --fidelity high on the default would
# time out EVERY eval, record all of them as `converged: false`, and produce a
# confident, entirely false "high fidelity doesn't work" result with no error
# raised anywhere. That is why this script sets it explicitly and why the default
# must never be relied on for a high-fidelity run.
# ---------------------------------------------------------------------------
#
# BUDGET. 81 free coefficients -> 82 evals per gradient. At ~400 s/eval over 32
# workers that is 3 waves ~ 20 min/iteration, so ~30 iterations is most of a day.
# One honest run per day, against ~1M cheap evals that produced nothing scoreable.
#
# SEEDS MUST CONVERGE AT HIGH FIDELITY. This is a new prerequisite and it excludes
# every one of the project's own leaders. `eval_seed_pool.py --fidelity high`
# produces the admissible list; pass a row from it.
set -u
cd "$(dirname "$0")/.." || exit 1

TAG=${TAG:-hf1}
POOL=${POOL:-output/diverse_seeds.npy}
ROW=${ROW:-0}
ITERS=${ITERS:-30}
W=${W:-32}
TIMEOUT=${TIMEOUT:-1800}      # >> the ~440 s measured; see the block above
ASPECT_MIN=${ASPECT_MIN:-0.0} # off by default: the lever is unvalidated at high fidelity
# fd_step=0.02 came from `fd_diagnostic.py`, which was run at LOW fidelity and
# justified by low-fidelity determinism. High fidelity is ALSO bit-identical
# (fidelity_check_high.py), so the noise argument does not carry the step size
# across -- that is an inherited property, and this project has been burned by
# those. `hf_asp16` took steps of |d|inf=0.011 down to 0.0001 against a 0.02
# probe: the Jacobian was measured 2-200x outside the region it was used in, and
# the LP overpredicted the drop by 27x. Overridable pending the high-fidelity
# re-measurement.
FD_STEP=${FD_STEP:-0.02}
# Default ON at high fidelity, unlike in minimax_p3.py where 0 preserves history.
MAX_STALL=${MAX_STALL:-4}

if [ ! -f "$POOL" ]; then echo "[hf] no pool $POOL"; exit 1; fi
if [ "$(printf '%.0f' "$TIMEOUT")" -lt 900 ]; then
  echo "[hf] REFUSING: TIMEOUT=$TIMEOUT is too low for high fidelity (~440 s/eval)."
  echo "[hf] A low timeout silently marks every eval non-converged. Use >= 900."
  exit 1
fi

extra=""
if [ "$ASPECT_MIN" != "0.0" ]; then
  extra="--aspect-min $ASPECT_MIN"
  echo "[hf] NOTE: aspect floor $ASPECT_MIN is ON. This lever produced the"
  echo "[hf] low-fidelity artifact; at high fidelity it is UNVALIDATED."
fi

echo "[hf] tag=$TAG seed=$POOL row=$ROW iters=$ITERS workers=$W timeout=${TIMEOUT}s"
docker compose run --rm --name "vega-hf-${TAG}" train scripts/minimax_p3.py \
    --tag "$TAG" --seed-npy "/work/${POOL}" --seed-row "$ROW" \
    --fidelity high --timeout-seconds "$TIMEOUT" \
    --hv-mode none $extra \
    --delta0 0.10 --shrink 0.7 --grow 1.6 \
    --fd-step "${FD_STEP:-0.02}" --max-stall "${MAX_STALL:-4}" \
    --max-iter "$ITERS" --n-workers "$W"
echo "[hf] DONE"

#!/bin/bash
# ASPECT LADDER: map achievable official worst-violation as a function of an
# imposed aspect-ratio floor. The output is a CURVE, and the quantity we need
# from it is the LOWEST A at which a design becomes feasible.
#
# WHY THIS IS THE RIGHT EXPERIMENT NOW. At the project leader (0.028653, A=12.894)
# the active set is three-way -- qi +0.0287, mirror +0.0286, fluxcomp +0.0113 --
# with iota (-0.136) and vwell (-0.090) carrying large unused margin. Fifteen
# consecutive exhaustion rounds moved the worst violation at -1.3e-4/round and
# never terminated; linear extrapolation to the 1e-2 bar is ~140 more rounds.
#
# One iteration with --aspect-min 14 did this instead (asp_smoke, verified):
#     it=0  A=12.894  qi +0.0287  mirror +0.0286  fluxc +0.0113
#     it=1  A=13.074  qi +0.0319  mirror +0.0096  fluxc +0.0069
# A +0.18 in aspect ratio put mirror AND fluxcomp under the 1e-2 bar. Four of the
# five official constraints are now satisfied and the problem has collapsed from a
# three-way conflicting active set to a SINGLE binding constraint, qi. That is a
# structural change, not an increment: a one-constraint descent has no conflicting
# -gradient stall mode, which is the exact failure the minimax was built for.
#
# WHY IT IS WORTH THE HYPERVOLUME. P3 scores hypervolume of the FEASIBLE set and
# returns exactly 0.0 when that set is empty -- our score after ~1M evals. A
# feasible design at A=14 contributes (1+L)(20-14); our infeasible leader at
# A=12.894 contributes nothing. So while the feasible set is empty, ANY feasible
# design is an infinite relative improvement, and the ladder's job is to find the
# cheapest one. Once a feasible design exists, A can be walked back DOWN from it.
#
# READ-OUT, FIXED IN ADVANCE. For each rung record final official worst, the
# binding constraint, final A, and hv. Three outcomes:
#   - some rung reaches official worst <= 1e-2  -> FIRST FEASIBLE DESIGN; the
#     lowest such A is the one to exploit, and the score stops being 0.0
#   - worst falls monotonically in A but never crosses -> extrapolate the qi-vs-A
#     curve to find the required A, and report whether it implies an hv worth
#     having (A>=20 contributes zero, so there IS a ceiling on this trade)
#   - worst does NOT fall in A -> the smoke-test step was a one-iteration
#     transient; the aspect lever is dead and this file records that it was tried
# Report whichever occurs. A feasible design with poor hv is still a WIN over 0.0,
# but do not describe it as beating the paper -- those are different claims.
set -u
cd "$(dirname "$0")/.." || exit 1

RUNGS=${RUNGS:-"13.5 14.0 15.0 16.0"}
SEED=${SEED:-output/ex_floor2_r14_seed.npy}
ITERS=${ITERS:-50}
W=${W:-12}

if [ ! -f "$SEED" ]; then echo "[asp] no seed $SEED"; exit 1; fi
echo "[asp] rungs: $RUNGS | seed=$SEED | $ITERS iters | $W workers"

for a in $RUNGS; do
  tag="asp_a${a}"
  echo "[asp] === aspect_min=$a ==="
  docker compose run --rm --name "vega-asp-${a}" train scripts/minimax_p3.py \
      --tag "$tag" --seed-npy "/work/${SEED}" --seed-row 0 \
      --hv-mode none --aspect-min "$a" \
      --delta0 0.10 --shrink 0.7 --grow 1.6 \
      --max-iter "$ITERS" --n-workers "$W" 2>&1 \
    | grep -aE "minimax\] (best worst|done in|\*\*\*)" | tail -3

  # Summarise this rung from the PERSISTED official violations, never from the
  # search 6-vector: the aspect row is a search device and is not scored.
  python3 - "$tag" <<'PY'
import json, os, sys
import numpy as np
NAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]
tag = sys.argv[1]
f = f"output/minimax_p3/{tag}_results.jsonl"
if not os.path.exists(f):
    print(f"[asp]   {tag}: no results"); raise SystemExit
best = None
nfeas = 0
for line in open(f):
    try: d = json.loads(line)
    except Exception: continue
    if not d.get("converged"): continue
    w = d.get("worst_violation")
    if w is None: continue
    if w <= 1e-2: nfeas += 1
    if best is None or w < best[0]: best = (w, d)
if best is None:
    print(f"[asp]   {tag}: no converged evals"); raise SystemExit
w, d = best
mt, v = d["measured_targets"], d["violations"]
bind = NAMES[int(np.argmax(v))]
print(f"[asp]   {tag}: official worst={w:.6f} binding={bind} "
      f"A={mt['aspect_ratio']:.4f} L={mt['minimum_normalized_magnetic_gradient_scale_length']:.4f} "
      f"hv={d['hv_contribution']:.2f} feasible_evals={nfeas}")
print("[asp]   violations: " + "  ".join(f"{n}={x:+.5f}" for n, x in zip(NAMES, v)))
if w <= 1e-2:
    print(f"[asp]   *** FEASIBLE DESIGN at aspect floor {tag} *** score contribution {d['hv_contribution']:.2f}")
PY
done
echo "[asp] DONE"

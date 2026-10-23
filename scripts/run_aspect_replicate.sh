#!/bin/bash
# DOES THE ASPECT-FLOOR LEVER REPLICATE? Paired A/B from independent seeds.
#
# WHAT HAPPENED. `asp_a13.5` reached the project's first all-five-constraints
# design at low fidelity (official worst 0.006549, A=13.5548, 104 feasible evals).
# That is n=1, from ONE seed, and that seed was `ex_floor2_r14_seed.npy` -- the
# 14th round of an exhaustion ladder on floor2, i.e. the single most heavily
# optimized point in the project. Everything about the result could be a property
# of that point rather than of the aspect floor. Two facts already argue it might be:
#   - the NEXT rung up, `asp_a14.0`, did NOT reach feasibility (0.023588, fluxcomp).
#     If more aspect ratio were simply better, 14.0 would have beaten 13.5.
#   - this project has already made exactly this mistake once. The `--hv-mode floor`
#     rule was adopted as a default off one basin's -36.4%, and the 6-basin paired
#     replication came back 2 wins / 1 tie / 3 losses, median slightly NEGATIVE.
#     s32's -36.4% remains unexplained and is now the outlier, not the rule.
#
# THE DESIGN. Paired arms, same seed, same budget, same delta schedule, same worker
# count, arms interleaved on the same machine. The ONLY difference is --aspect-min.
# A paired A/B is specifically what exposed the hv-floor error; an unpaired
# "aspect-floor runs reach lower violation" comparison cannot distinguish the lever
# from the seed, which is the entire question here.
#
# THE SEEDS are from `output/diverse_seeds.npy`, chosen off `eval_seed_pool.py` to
# be independent of floor2 and to vary what the lever has to work against -- nfp,
# aspect ratio, AND which constraint is binding at iteration 0:
#   row  0  nfp=2  A= 8.4298  worst=0.09653  binding=mirror
#   row  2  nfp=3  A= 9.7235  worst=0.20398  binding=qi
#   row 11  nfp=4  A= 9.5091  worst=0.27131  binding=vwell
# All three start far below the 13.5 floor, so the sixth constraint is active from
# iteration 0 and the lever is genuinely engaged. All three also start 4-10x worse
# than the asp_a13.5 seed did (0.0268), which matters for reading the result.
#
# READ-OUT, FIXED IN ADVANCE. Two different claims, and this run can confirm the
# first while refuting the second:
#
#   CLAIM A (mechanism): the floor reduces official worst-violation relative to the
#     paired no-floor arm. Read: sign of (floor - none) per seed. 3/3 or 2/3 with a
#     clear median makes the lever real and general. 1/3 or a near-zero median makes
#     it a property of floor2's basin, and asp_a13.5 goes in the log as a
#     single-basin result that did not generalize -- same shape as --hv-mode floor.
#
#   CLAIM B (feasibility): a floor arm reaches official worst <= 1e-2. This is NOT
#     expected here and its absence does NOT refute claim A. These seeds start at
#     0.097-0.271; asp_a13.5 needed 27 iterations from 0.0268. Reaching 1e-2 from
#     0.271 in 40 iterations would require a reduction larger than anything measured
#     in 72 runs. Do not read "no arm became feasible" as "the lever failed."
#
# The quantity that decides CLAIM A is the PAIRED DIFFERENCE, not the absolute
# level. Report it per seed with the starting point alongside, so a reader can see
# the lever separately from the seed quality.
set -u
cd "$(dirname "$0")/.." || exit 1

SEEDS=${SEEDS:-"0 2 11"}
POOL=${POOL:-output/diverse_seeds.npy}
FLOOR=${FLOOR:-13.5}
ITERS=${ITERS:-40}
W=${W:-3}

if [ ! -f "$POOL" ]; then echo "[rep] no pool $POOL"; exit 1; fi
echo "[rep] seeds: $SEEDS | floor=$FLOOR | $ITERS iters | $W workers/arm | pool=$POOL"

pids=""
for s in $SEEDS; do
  for arm in none floor; do
    tag="rep_s${s}_${arm}"
    extra=""
    [ "$arm" = "floor" ] && extra="--aspect-min $FLOOR"
    docker compose run --rm --name "vega-rep-s${s}-${arm}" train scripts/minimax_p3.py \
        --tag "$tag" --seed-npy "/work/${POOL}" --seed-row "$s" \
        --hv-mode none $extra \
        --delta0 0.10 --shrink 0.7 --grow 1.6 \
        --max-iter "$ITERS" --n-workers "$W" \
      > "/tmp/claude-1001/-home-sl8rv/6038fa05-1353-4fbe-9cdd-85b85f1b7172/scratchpad/rep_${s}_${arm}.log" 2>&1 &
    pids="$pids $!"
    echo "[rep] launched $tag (pid $!)"
  done
done

echo "[rep] waiting on$pids"
wait $pids
echo "[rep] all arms finished; summarising on the OFFICIAL five constraints"

python3 - $SEEDS <<'PY'
import json, os, sys
import numpy as np
NAMES = ["iota", "qi", "mirror", "fluxcomp", "vwell"]

def best(tag):
    """Best over ALL evals on the OFFICIAL five. The aspect row is a search device
    and is never scored, so the floor arm must be judged on the same five as the
    control or the comparison is rigged in the floor's favour."""
    f = f"output/minimax_p3/{tag}_results.jsonl"
    if not os.path.exists(f):
        return None
    b = None
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        if not d.get("converged"): continue
        w = d.get("worst_violation")
        if w is None: continue
        if b is None or w < b[0]: b = (w, d)
    return b

print(f"{'seed':>5} {'start':>9} {'none':>9} {'floor':>9} {'floor-none':>11} {'%':>7}  binding(floor)  A(floor)")
diffs = []
for s in sys.argv[1:]:
    bn, bf = best(f"rep_s{s}_none"), best(f"rep_s{s}_floor")
    if bn is None or bf is None:
        print(f"{s:>5}  MISSING ARM (none={bn is not None} floor={bf is not None})")
        continue
    wn, wf = bn[0], bf[0]
    mt, v = bf[1]["measured_targets"], bf[1]["violations"]
    bind = NAMES[int(np.argmax(v))]
    d = wf - wn
    pct = 100.0 * d / wn
    diffs.append(d)
    # start point is iteration 0 of either arm -- identical by construction
    st = None
    for line in open(f"output/minimax_p3/rep_s{s}_none_results.jsonl"):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("converged") and r.get("iter") == 0:
            st = r["worst_violation"]; break
    print(f"{s:>5} {st if st is None else f'{st:9.5f}'} {wn:9.5f} {wf:9.5f} "
          f"{d:+11.5f} {pct:+6.1f}%  {bind:>8}  {mt['aspect_ratio']:7.4f}")

if diffs:
    wins = sum(1 for d in diffs if d < 0)
    print(f"\n[rep] CLAIM A: floor better in {wins}/{len(diffs)} seeds; "
          f"median paired difference {np.median(diffs):+.5f}")
    if wins >= 2 and np.median(diffs) < 0:
        print("[rep] -> lever reproduces outside floor2's basin")
    else:
        print("[rep] -> lever does NOT generalise; asp_a13.5 is a single-basin result")
PY
echo "[rep] DONE"

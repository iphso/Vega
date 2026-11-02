#!/bin/bash
# Extension after the first pass: Ni-58 turned out to be the largest single contributor
# (n=10 too few), and the joint set's CI was wide. The driver resumes, so this only adds
# samples 10-19 (Ni-58) and 20-29 (all).
set -e
cd ~/Projects/Vega
P=${PARTICLES:-150000}; B=${BATCHES:-15}; T=${THREADS:-8}
run() { docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=$T \
          -v $HOME/runs/blanket/uq:/runs openmc blanket/uq/sample_driver.py --particles $P --batches $B "$@" \
          2>&1 | grep --line-buffered -v -E 'orphan|Container'; }
pair() { local name=$1 n=$2; shift 2
  run --set ${name}_A --factor-key $name --n $n --seed 777 --fixed-seed "$@"
  run --set ${name}_B --factor-key $name --n $n --seed 778 --fixed-seed "$@"; }
pair ni58 20 --nuclides Ni58
pair all 30 --nuclides Li6 Li7 Be9 F19 Ni58 W182 W183 W184 W186
echo EXTEND DONE

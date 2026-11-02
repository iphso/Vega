#!/bin/bash
# Production campaign: correlated sampling with two common transport seeds.
# Every sample set (and the nominal library) is run at seed A=777 and B=778 with
# 150k histories each; per sample D_i(s) = Y_i(s) - Y_nom(s) and analyze.py uses
# the A/B disagreement of D_i to measure (and subtract) the residual MC noise.
set -e
cd ~/Projects/Vega
P=${PARTICLES:-150000}; B=${BATCHES:-15}; T=${THREADS:-8}
run() { docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=$T \
          -v $HOME/runs/blanket/uq:/runs openmc blanket/uq/sample_driver.py --particles $P --batches $B "$@" \
          2>&1 | grep --line-buffered -v -E 'orphan|Container'; }
pair() { local name=$1 n=$2; shift 2
  run --set ${name}_A --factor-key $name --n $n --seed 777 --fixed-seed "$@"
  run --set ${name}_B --factor-key $name --n $n --seed 778 --fixed-seed "$@"; }
pair nom 1 --mode base
pair li7 20 --nuclides Li7
pair f19 20 --nuclides F19
pair be9 20 --nuclides Be9
pair li6 10 --nuclides Li6
pair all 20 --nuclides Li6 Li7 Be9 F19 Ni58 W182 W183 W184 W186
pair w 10 --nuclides W182 W183 W184 W186
pair ni58 10 --nuclides Ni58
echo CAMPAIGN DONE

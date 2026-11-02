#!/bin/bash
# Pipeline sanity checks, all at the same transport seed (777) so differences are exact:
#   chk_base   original library
#   chk_unity  Li6/Li7/Be9/F19 rewritten through the perturbation pipeline with factors = 1
#   chk_li6nt  Li-6 (n,t) x 1.10 in every group
#   chk_corr   2 Li-7 random samples at the same seed (does correlated sampling help?)
set -e
cd ~/Projects/Vega
T=${THREADS:-8}
run() { docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=$T \
          -v $HOME/runs/blanket/uq:/runs openmc blanket/uq/sample_driver.py "$@" 2>&1 | grep -v -E 'orphan|Container'; }
run --set chk_base --mode base --n 1 --seed 777
run --set chk_unity --mode unity --nuclides Li6 Li7 Be9 F19 --n 1 --seed 777
run --set chk_li6nt --mode li6nt10 --n 1 --seed 777
run --set chk_corr_li7 --nuclides Li7 --n 3 --seed 777 --fixed-seed
run --set chk_base2 --mode base --n 1 --seed 778
run --set chk_corr_li7_s778 --factor-key chk_corr_li7 --nuclides Li7 --n 3 --seed 778 --fixed-seed

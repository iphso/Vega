#!/bin/bash
# docker compose wrapper: mounts the run tree at /bench and the UQ tree read-only at /uq
cd ~/Projects/Vega
exec docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e UQ_ROOT=/uq -e OMP_NUM_THREADS=${T:-12} \
  -v $HOME/runs/blanket/uq:/uq:ro -v $HOME/runs/blanket/bench/oktavian:/bench openmc "$@" 2>&1 | grep --line-buffered -v -E 'orphan|Container'

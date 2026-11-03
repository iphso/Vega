#!/bin/bash
# OKTAVIAN production: high-statistics nominal, geometry sensitivities, then the 30 joint ND samples
# at the UQ campaign's paired seeds (A=777, B=778) plus the nominal library at both seeds.
# Resumable (run_bench.py skips finished (tag, case) pairs).
set -e
D=$(dirname $(readlink -f $0)); R=blanket/bench/oktavian/run_bench.py
P=${PARTICLES:-600000}
run() { $D/dc.sh $R "$@" | grep --line-buffered -v -i warn; }
run --tag nomhi --seed 12345 --particles 5000000 --batches 50
# geometry / material sensitivities (correlated with nom_A: same seed)
TBR="tbr_li tbr_lic tbr_pbli tbr_pblic tbr_beli tbr_belic"; BELI="be57 be69 be97 be128 beli57 beli128"
run --tag sens_ref     --seed 777 --particles 1000000
run --tag sens_lidens+ --seed 777 --particles 1000000 --cases $TBR beli57 beli128 --geom '{"li_dens": 0.546}'
run --tag sens_lidens- --seed 777 --particles 1000000 --cases $TBR beli57 beli128 --geom '{"li_dens": 0.494}'
run --tag sens_awall   --seed 777 --particles 1000000 --cases $TBR beli57 beli128 --geom '{"a_wall_in": 0.2}'
run --tag sens_cdens   --seed 777 --particles 1000000 --cases tbr_lic tbr_pblic tbr_belic --geom '{"c_dens": 1.785}'
run --tag sens_pbdens  --seed 777 --particles 1000000 --cases tbr_pbli tbr_pblic --geom '{"pb_dens": 10.77}'
run --tag sens_bedens  --seed 777 --particles 1000000 --cases tbr_beli tbr_belic $BELI --geom '{"be_scale": 0.98}'
run --tag sens_cavity  --seed 777 --particles 1000000 --cases tbr_li tbr_lic tbr_pbli tbr_pblic --geom '{"cblk_rin": 9.0, "pb_rin": 9.0}'
run --tag sens_c10     --seed 777 --particles 1000000 --cases tbr_lic tbr_pblic tbr_belic --geom '{"c_thick": 10.0}'
run --tag sens_lifw    --seed 777 --particles 1000000 --cases lif60 w40 --geom '{"lif_dens": 1.826, "w_dens": 4.519}'
# ND samples, paired seeds
run --tag nom_A --seed 777 --particles $P
run --tag nom_B --seed 778 --particles $P
for i in $(seq 0 29); do
  run --tag s${i}_A --seed 777 --lib $i --particles $P
  run --tag s${i}_B --seed 778 --lib $i --particles $P
done
echo CAMPAIGN DONE

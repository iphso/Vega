# Integral-experiment assimilation + Be-9 (n,2n) spectrum hardness

## assimilate.py
GLLS update of the ARC TBR from benchmark C/E (blanket/bench/*/results.json) using the 30 joint
ENDF/B-VIII.0 samples (blanket/uq, set `all`) run on both ARC and the benchmarks. Seed-crossed
covariance (A x B) removes MC noise in expectation; off-diagonals shrunk (Ledoit-Wolf); a
permutation null (ARC samples re-paired at random) is the bar any band reduction must clear.
Synthetic check (2026-11-03): a benchmark with r = 0.88 to ARC reading 1% low shifts ARC -0.38% and
narrows ±0.68% -> ±0.38% (null 95%: 1.2%); an unrelated benchmark changes nothing.

## be_sed.py: Be-9 MT16 outgoing-energy tilt
Exponential tilt of p(E'|E) per incident energy so <E'> moves by exactly `shift`; angle tables and
the (n,2n) cross section / multiplicity untouched (heating MT301 not made consistent).
Check: <E'> at 14.1 MeV 4.269 -> 4.696 / 3.842 MeV for +/-10%.

ARC baseline TBR (1 cm Be), 600k histories, seeds 11/12 (std ~0.0013-0.0018 each):

| Be (n,2n) <E'> | seed 11 | seed 12 |
|---|---|---|
| -10% | 1.13610 | 1.13615 |
| nominal | 1.13682 | 1.13791 |
| +10% | 1.13648 | 1.13798 |

Mean change: -10% -> -0.11%, +10% -> -0.01%; both within the ~0.15-0.2% noise of a difference.
So |dTBR| <~ 0.15% per 10% hardness: the Be secondary-spectrum term that dominates thick-Be
blankets in older studies is small for ARC's 1 cm multiplier. Thick-Be designs need the
benchmark constraint (FNS Be, OKTAVIAN Be spheres).

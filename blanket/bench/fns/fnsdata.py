"""Experimental data of the JAERI FNS Li2O and Be cylindrical assemblies, transcribed from the
FENDL-2 benchmark compilation JAERI-M 94-014 (blanket/refs/lit/jaerim94014/benchmarks/jaerim94014):
  li2o/read.me  (chapter 7, Tables 7.1-7.21)       be/read.me + be/t8-*.txt, be/f8-*.txt (chapter 8)

All positions are distances from the D-T target along the assembly axis [cm]; the assembly front
face is at 20.0 cm. Rates are per atom of the detector nuclide per source neutron (= per D-T
reaction, from the associated-alpha monitor).

Uncertainty split (relative, %). Each series gets a systematic part that is fully correlated within
its E_corr_group and a random part (uncorrelated). Where the compilation gives an explicit split we
use it; otherwise the assumption is stated next to the series.
"""
import math

# --------------------------------------------------------------------------- Li2O (chapter 7)
# Table 7.1 homogenised densities [atoms/b-cm]; outer radius per region [cm]
LI2O_REGIONS = {
    "inner": dict(r=2.719, Li6=4.221e-3, Li7=5.277e-2, O=2.850e-2, Fe=5.205e-4, Ni=6.296e-5,
                  Cr=1.433e-4, Mn=7.245e-6),
    "ss": dict(r=2.866, Fe=2.324e-2, Ni=2.812e-3, Cr=6.398e-3, Mn=3.295e-4),
    "outer": dict(r=31.5, Li6=4.191e-3, Li7=5.240e-2, O=2.830e-2, Fe=1.112e-3, Ni=1.352e-4,
                  Cr=3.068e-4, Mn=1.880e-5),
}
# Text: "63.0 cm in average diameter and 61.0 cm in length"; 12 standard blocks x 5.06 cm = 60.72 cm.
LI2O_LENGTH = 61.0
LI2O_SRC_0DEG = 1.1767   # Table 7.21 normalisation (= sum of the tabulated spectrum)

# Table 7.2: pos, T6 measured, err%, T6 corrected self-shielding, T6 corrected both (ss + room),
#            T7 measured, err%   (None = not given). 80.8 cm: room return "over 200 %", no corrected value.
LI2O_LSC = [
    (19.9, 7.58e-29, 4.8, 7.71e-29, 7.66e-29, 6.32e-29, 4.2),
    (22.4, 8.81e-29, 4.5, 8.98e-29, 8.93e-29, 4.60e-29, 4.9),
    (24.9, 9.30e-29, 4.5, 9.48e-29, 9.44e-29, 3.85e-29, 5.6),
    (27.5, 9.17e-29, 4.5, 9.28e-29, 9.23e-29, 2.67e-29, 6.4),
    (30.0, 9.01e-29, 4.5, 9.19e-29, 9.15e-29, 2.08e-29, 7.6),
    (35.1, 8.05e-29, 4.7, 8.23e-29, 8.18e-29, 1.26e-29, 10.7),
    (40.2, 6.89e-29, 4.9, 7.06e-29, 7.01e-29, 4.57e-30, 23.8),
    (45.3, 5.61e-29, 5.3, 5.75e-29, 5.71e-29, 3.92e-30, 28.1),
    (50.3, 4.48e-29, 5.8, 4.60e-29, 4.55e-29, None, None),
    (55.4, 3.42e-29, 6.5, 3.52e-29, 3.48e-29, None, None),
    (60.5, 2.73e-29, 7.3, 2.81e-29, 2.76e-29, None, None),
    (65.6, 2.01e-29, 8.7, 2.08e-29, 2.03e-29, None, None),
    (70.7, 1.47e-29, 10.5, 1.52e-29, 1.46e-29, None, None),
    (75.7, 8.65e-30, 15.7, 8.94e-30, 8.36e-30, None, None),
    (80.8, 7.11e-30, 18.3, 7.34e-30, None, None, None),
]
# Table 7.3 systematic: yield 1.5, efficiency 1.0, Li atoms 0.1 -> 1.81 %; rest of the quoted error random
LI2O_LSC_SYS = math.sqrt(1.5**2 + 1.0**2 + 0.1**2)

# Table 7.4 Li-glass pair (6Li TPR): z, r, TPR, total err %, syst %, random Ds/s %, random DCa/DCa %
# total err includes the fitting / self-shielding errors (footnote *1 says they are excluded from the
# 5.4 % syst column); the 80.70 cm value 8.13e-29 is 10x the 75.72 cm point and the Li-pellet back
# value (7.3e-30): almost certainly a typo for 8.13e-30 -> excluded rather than guessed.
LI2O_LG = [
    (20.25, 0.0, 5.44e-29, 11.0), (22.53, 0.0, 7.05e-29, 9.9), (25.06, 0.0, 8.49e-29, 9.1),
    (27.61, 0.0, 7.63e-29, 10.5), (30.12, 0.0, 7.32e-29, 8.3), (40.24, 0.0, 6.21e-29, 7.9),
    (50.38, 0.0, 4.23e-29, 7.1), (60.50, 0.0, 2.52e-29, 7.3), (70.64, 0.0, 1.33e-29, 6.9),
    (75.72, 0.0, 8.28e-30, 6.8),
    (20.25, 15.2, 6.10e-29, 9.5), (25.06, 15.2, 6.56e-29, 8.3), (30.12, 15.2, 6.04e-29, 8.6),
    (50.38, 15.2, 3.57e-29, 7.7), (70.64, 15.2, 1.11e-29, 6.8),
]
LI2O_LG_EXCLUDED = [(80.70, 0.0, 8.13e-29, 6.6, "suspected typo (x10), back face, room dominated")]
LI2O_LG_SYS = 5.4

# Table 7.6 NE213-unfolded 7Li TPR: pos, TPR, unfolding err %, source-intensity err %.
# Source err = sqrt(alpha statistics^2 + 2.34^2) (Table 7.11 counts reproduce it exactly), the 2.34 %
# alpha-monitor calibration being common to all points; efficiency calibration (+0.4/-1.6) % taken as
# a further 1.0 % common systematic. NOTE: this "measurement" is the unfolded spectrum folded with
# the experimenters' 7Li(n,n't) cross section (JENDL-3/ENDF/B-IV era), so it is not independent of
# the cross section being tested.
LI2O_NE213 = [(21.6, 4.61e-29, 2.1, 3.3), (31.5, 1.66e-29, 1.0, 2.7), (41.6, 6.23e-30, 1.0, 2.7),
              (51.7, 2.40e-30, 2.4, 2.5), (61.8, 9.04e-31, 3.2, 2.4), (72.0, 3.46e-31, 4.1, 2.4),
              (82.1, 1.25e-31, 3.6, 2.3)]
LI2O_NE213_SYS = math.sqrt(2.34**2 + 1.0**2)

# Table 7.7 micro fission chambers: pos, U235, U238, Np237, Th232 (value, err%)
LI2O_FC = [
    (23.9, (5.00e-28, 4.1), (1.85e-28, 3.9), (4.71e-28, 5.1), (5.74e-29, 4.7)),
    (26.4, (4.25e-28, 4.2), (1.41e-28, 3.9), (3.67e-28, 5.1), (4.29e-29, 4.7)),
    (29.0, (3.67e-28, 4.1), (1.05e-28, 3.9), (2.93e-28, 5.1), (3.29e-29, 4.7)),
    (31.5, (3.15e-28, 4.1), (8.04e-29, 3.9), (2.38e-28, 5.1), (2.45e-29, 4.7)),
    (36.6, (2.40e-28, 4.1), (4.87e-29, 3.9), (1.51e-28, 5.1), (1.44e-29, 4.7)),
    (41.7, (1.83e-28, 4.1), (2.97e-29, 3.9), (9.57e-29, 5.1), (8.73e-30, 4.7)),
    (46.7, (1.40e-28, 4.1), (1.81e-29, 3.9), (6.33e-29, 5.1), (5.40e-30, 4.7)),
    (51.8, (1.06e-28, 4.1), (1.14e-29, 3.9), (4.05e-29, 5.1), (3.29e-30, 4.8)),
    (56.9, (7.96e-29, 4.1), (7.05e-30, 3.9), (2.68e-29, 5.2), (2.05e-30, 5.2)),
    (62.0, (5.90e-29, 4.1), (4.42e-30, 4.0), (1.68e-29, 5.2), None),
    (67.1, (4.18e-29, 4.1), (2.80e-30, 4.0), (1.10e-29, 5.2), (7.87e-31, 6.4)),
    (72.1, (2.81e-29, 4.1), (1.72e-30, 4.0), (7.13e-30, 5.3), (5.22e-31, 5.6)),
    (77.2, (1.68e-29, 4.2), (9.97e-31, 4.2), (4.31e-30, 5.2), (3.30e-31, 5.7)),
]
# Table 7.9 foils (b) 27Al(n,a), (c) 58Ni(n,2n): pos, rate, err%
LI2O_AL = [(19.95, 2.12e-29, 3.2), (22.44, 1.57e-29, 3.1), (24.98, 1.18e-29, 3.1), (27.52, 8.61e-30, 3.3),
           (30.06, 6.31e-30, 3.3), (35.14, 3.60e-30, 3.3), (40.22, 2.12e-30, 3.5), (45.30, 1.28e-30, 3.5),
           (50.29, 7.62e-31, 3.5), (55.47, 4.49e-31, 3.1), (60.55, 2.84e-31, 3.8), (65.63, 1.67e-31, 3.8),
           (70.71, 9.87e-32, 3.7), (75.79, 6.11e-32, 3.6)]
LI2O_NI2N = [(19.85, 8.00e-30, 3.1), (22.34, 5.19e-30, 2.8), (24.88, 3.61e-30, 3.2), (27.42, 2.34e-30, 4.5),
             (29.96, 1.62e-30, 5.3), (35.04, 8.84e-31, 5.3), (40.12, 4.10e-31, 6.6), (45.20, 2.35e-31, 6.7),
             (50.29, 1.23e-31, 7.1), (55.37, 6.13e-32, 2.8), (60.45, 3.22e-32, 9.5), (65.53, 2.01e-32, 9.6),
             (70.61, 1.16e-32, 10.3), (75.69, 6.12e-33, 8.9)]

# --------------------------------------------------------------------------- Be (chapter 8)
# Material and geometry from the JAERI MCNP-3 input in f8-8.txt (Fig. 8.8): Be blocks 1.22149e-1
# atoms/b-cm incl. impurities; centre drawer = square SS tube, |x|,|y| in [2.46, 2.54] cm, 2.00898e-2.
BE_BLOCK = dict(Be9=1.2152e-1, C=7.7109e-5, O16=4.9813e-4, Al27=2.9013e-4, Fe=2.4678e-5)
BE_DRAWER = dict(Cr=3.9167e-3, Mn55=2.0032e-4, Fe=1.4240e-2, Ni=1.7328e-3)
BE_LENGTH = 45.54          # f8-8 surfaces 2/3 (z = 20.0 .. 65.54); text: 456 mm
BE_SRC_0DEG = 1.1262       # chapter 8 text; t8-16 sums to 1.12617 (MCNP SRC weight 1.1261)

# t8-4: 6Li TPR by Li-glass (10 mm dia x 0.1 mm), corrected for self-shielding and edge effect.
# position [mm from target], TPR, err %
BE_LG = [(201.1, 3.583e-26, 3.07), (251.7, 2.065e-25, 3.19), (302.3, 2.983e-25, 3.06),
         (327.6, 3.226e-25, 2.35), (403.5, 3.254e-25, 2.35), (479.4, 2.339e-25, 2.92),
         (504.7, 2.007e-25, 2.92), (605.9, 7.863e-26, 2.71), (631.2, 5.050e-26, 3.01)]
# Errors "estimated as similar to the Li2O / graphite slabs": syst = yield 1.9 (+) 6Li atoms 0.5 (Table 7.5)
BE_LG_SYS = math.sqrt(1.9**2 + 0.5**2)
# t8-5: U-235 micro-fission chamber (FC4A) [mm], rate, err %
BE_FC = [(241.2, 9.267e-26, 3.81), (266.6, 1.307e-25, 3.80), (292.0, 1.579e-25, 3.80),
         (317.4, 1.794e-25, 3.80), (342.8, 1.906e-25, 3.80), (419.0, 1.771e-25, 3.80),
         (469.8, 1.432e-25, 3.80), (520.6, 1.073e-25, 3.80), (571.4, 7.071e-26, 3.81),
         (622.2, 3.368e-26, 3.81)]
# assumption: yield ~2 %, deposit mass ~2 %, extrapolation/efficiency ~1.5 % common -> 3.2 % systematic
BE_FC_SYS = 3.2
# t8-6 foils; header depths 0.0, "48.0" (= 4.80, text: 248 mm from target), 14.92, 25.06, 35.20, 45.59 cm
# from the front face. Only reactions whose product is the ground state of the full MT are kept
# (no isomer data in the transport library): In115(n,n')115m, Nb93(n,2n)92m and Ti(n,x)Sc are skipped.
BE_FOIL_DEPTHS = [0.0, 4.80, 14.92, 25.06, 35.20, 45.59]
BE_FOILS = {
    "au197_ng": [(8.628e-27, 3.0), (4.806e-26, 2.7), (7.272e-26, 3.0), (4.957e-26, 2.9), (2.330e-26, 2.9), (2.151e-27, 3.0)],
    "al27_na": [(2.588e-29, 2.7), (1.382e-29, 2.9), (3.750e-30, 3.0), (1.039e-30, 3.2), (2.853e-31, 3.7), (7.559e-32, 3.2)],
    "ni58_n2n": [(9.416e-30, 2.9), (3.675e-30, 2.7), (6.881e-31, 3.6), (1.338e-31, 4.3), (3.271e-32, 4.5), (7.543e-33, 9.0)],
    "fe56_np": [(2.240e-29, 2.7), (1.221e-29, 2.7), (3.365e-30, 2.8), (9.886e-31, 3.0), (2.746e-31, 3.2), (7.412e-32, 3.7)],
    "zr90_n2n": [(1.772e-28, 2.7), (7.627e-29, 2.7), (1.465e-29, 2.7), (3.316e-30, 3.4), (7.508e-31, 4.1), (1.682e-31, 3.4)],
    "ni58_np": [(7.632e-29, 2.7), (6.162e-29, 2.7), (2.292e-29, 2.8), (7.628e-30, 3.4), (2.304e-30, 3.1), (6.001e-31, 4.1)],
}
# foil errors: detector efficiency 2.5 % is the common part (Tables 6.6 / 7.10); the quoted totals
# (>= 2.7 %) are smaller than efficiency (+) 3 % yield, so the yield error is evidently NOT included.
FOIL_SYS = 2.5

# Room (Li2O text, same FNS first target room / 80-degree beam line for Be): 15 m x 15 m;
# target to west and south walls 5.5 m, to ceiling 7.9 m, grating floor 1.8 m, basement floor 3.8 m.
ROOM = dict(near_wall=550.0, size=1500.0, ceiling=790.0, grating=180.0, basement=380.0)

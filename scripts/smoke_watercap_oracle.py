"""Physics sanity battery for watercap_oracle.py, as assertions. Each case
has a known right answer that does not depend on any uncalibrated constant:

  1. Kelvin-Thomson at the Rayleigh limit: equilibrium vapor-pressure ratio
     >= ~1 for every radius -- no stable charged drop sits below flat-water
     saturation. The claim the whole vapor-capture argument rests on.
  2. Everything off: outlet == inlet (no recovery, mass balance exact).
  3. Saturated inlet + charged spray AT the inlet temperature: no vapor
     removal, regardless of charge.
  4. Cold spray: vapor removal > 0 and never beyond the adiabatic mixing
     ceiling (ceiling_gap_frac >= 0, to tolerance).
  5. Corona only: mist is collected; opposite spray polarity runs.
  6. Water balance closes in every case.

Run: python3 scripts/smoke_watercap_oracle.py   (host or any compose service)
"""
import time
import warnings

import numpy as np

import watercap_oracle as o

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = dict(air_flow_m3_s=0.15, tube_diameter_m=0.3, spray_section_length_m=1.5, corona_section_length_m=1.5,
            nozzle_height_frac=0.9, disc_diameter_frac=0.3, disc_spacing_m=0.1, spray_flow_L_min=2.0,
            spray_n_nozzles=100, spray_T_C=20.0, spray_conductivity_mS_m=50.0, spray_voltage_kV=10.0,
            corona_voltage_kV=30.0, corona_polarity=-1.0, spray_polarity_rel=1.0)
AUX = dict(inlet_T_C=35.0, inlet_RH=1.0, inlet_lwc_g_m3=2.0, mist_mmd_um=15.0, mist_gsd=1.8)


def run(**overrides):
    x = {**BASE, **overrides}
    t = time.perf_counter()
    ok, pl = o.simulate(np.array([x[k] for k in o.PARAM_NAMES]), AUX, o.FIDELITY_PRESETS["low"])
    assert ok, pl
    print(f"  ({time.perf_counter() - t:.2f}s) vapor_removed={pl['vapor_removed_frac']:+.4f} "
          f"outlet_total={pl['outlet_total_water_frac']:.4f} ceiling_gap={pl['ceiling_gap_frac']:+.4f} "
          f"mist_collected={pl['mist_collected_frac']:.3f} residual={pl['water_balance_residual']:+.1e}")
    assert abs(pl["water_balance_residual"]) < 1e-4, "water balance does not close"
    return pl


def main():
    print("1. Kelvin-Thomson at the Rayleigh limit")
    for a in np.logspace(-7, -4, 7):
        r = float(o.kelvin_thomson_factor(a, o.rayleigh_charge(a, 20.0), 20.0))
        print(f"  a={a:.1e} m  p_eq/p_sat={r:.8f}")
        assert r >= 1 - 1e-4, "a stable charged drop would sit below flat-water saturation"

    print("2. everything off")
    pl = run(spray_flow_L_min=0.0, spray_voltage_kV=0.0, corona_voltage_kV=0.0)
    assert abs(pl["recovered_L_h"]) < 1e-6 and abs(pl["vapor_removed_frac"]) < 1e-3

    print("3. charged spray at the saturated inlet temperature")
    pl = run(spray_T_C=AUX["inlet_T_C"], corona_voltage_kV=0.0)
    assert pl["vapor_removed_frac"] < 1e-3, "charge alone removed vapor"

    print("4. cold spray")
    pl = run(corona_voltage_kV=0.0)
    assert pl["vapor_removed_frac"] > 0 and pl["ceiling_gap_frac"] > -0.01, "beat the thermodynamic ceiling"

    print("5. corona only, then opposite spray polarity")
    pl = run(spray_flow_L_min=0.0, spray_voltage_kV=0.0)
    assert pl["mist_collected_frac"] > 0.5 and pl["corona_current_mA"] > 0
    run(spray_polarity_rel=-1.0)
    print("all watercap smoke checks passed")


if __name__ == "__main__":
    main()

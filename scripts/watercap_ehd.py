"""The watercap domain's resolved-electrostatics tier (FIDELITY_PRESETS
"ehd_coarse" / "ehd"): watercap_oracle.py's ROM for the spray stage, then
StoneLab's two-way coupled axisymmetric corona + droplet model
(stonelab.ehd / stonelab.ehd_particles) for the corona stage, replacing the
ROM's coaxial-equivalent corona.

What the 2D stage resolves that the ROM cannot (each was a named,
uncalibrated constant or a missing effect in the ROM):
  - corona onset and I-V of the ACTUAL disc geometry (rounded edge of
    curvature t/2, shielding by neighbouring discs, the tube): Kaptsov +
    Peek on a Shortley-Weller boundary, validated against the exact
    coaxial solution to <0.5%. For the default 0.3 mm discs it puts onset at
    ~33 kV where the ROM's CORONA_IV_COEFF/shield guess said 13.6 kV.
  - corona quenching by the mist's own space charge (2 g/m^3 of 15 um mist
    cut the first period's current by ~20% in the first coupled runs)
  - drop charging along real trajectories in the real field, turbulent
    dispersion with near-wall damping, deposition on the wall vs the discs
  - the flow around the discs (pipe profile + potential correction)

Vapor is still the ROM's: the corona section changes humidity negligibly
(no subcooled liquid there), so the ROM's outlet humidity ratio is used.

StoneLab is a sibling repository; set STONELAB_SRC to its `src/` if it is
not at ../../StoneLab/src relative to this file. Requires scipy.
"""
import os
import sys
from pathlib import Path

import numpy as np

import watercap_oracle as rom

_STONELAB = os.environ.get("STONELAB_SRC", str(Path(__file__).resolve().parents[2] / "StoneLab" / "src"))
if _STONELAB not in sys.path:
    sys.path.insert(0, _STONELAB)

DISC_THICKNESS_M = 3e-4  # fixed hardware constant (edge curvature t/2), same as watercap_oracle.DISC_EDGE_RADIUS_M*2

# edge_cells = cells per edge radius at the disc edge; h_coarse = bulk cell
# size. Chosen from a Laplace onset-voltage grid study (onset converges to
# 33.0-33.5 kV for 0.3 mm discs in a 30 cm tube): 3 cells/5 mm gave +5%,
# 5 cells/5 mm gave -9% (not even monotone), so the coarse tier is 8 cells/
# 2 mm (32.96 kV, 13.6k cells). Section check at 36 kV, near onset where it
# is most sensitive: coarse 18.0 vs medium 19.6 uA/period, mass escaping
# 3.0% vs 2.4%; at 45 kV 89.8 vs 91.7 uA/period. Medium is within ~5% of
# 20 cells/edge radius.
EHD_PRESETS = {
    "ehd_coarse": {"edge_cells": 8, "h_coarse": 2e-3, "growth": 1.2, "n_per_class": 800, "coupling_iters": 2},
    "ehd": {"edge_cells": 10, "h_coarse": 1e-3, "growth": 1.12, "n_per_class": 1500, "coupling_iters": 3},
}


def simulate(params, aux, preset_name="ehd_coarse", rom_fidelity="low", seed=0, spray_override=None,
             disc_thickness_m=DISC_THICKNESS_M, rel_air_density=None):
    """(ok, payload) with the same TARGET_NAMES as watercap_oracle, so EHD
    rows and ROM rows share one dataset schema (and a multi-fidelity
    surrogate sees the same targets at both fidelities)."""
    from stonelab.ehd_particles import DiscStackSection, march_section

    ok, base = rom.simulate(params, aux, rom.FIDELITY_PRESETS[rom_fidelity], spray_override=spray_override)
    if not ok:
        return ok, base
    pre = EHD_PRESETS[preset_name]
    x = dict(zip(rom.PARAM_NAMES, (float(v) for v in params)))
    ci = base["corona_inlet"]
    R = x["tube_diameter_m"] / 2
    pol_c = ci["pol_c"]
    n_periods = max(1, int(round(x["corona_section_length_m"] / x["disc_spacing_m"])))
    if rel_air_density is None:  # Peek's delta at the inlet temperature (35 C: 0.95 -> onset ~4% lower than at 20 C)
        rel_air_density = (293.15 / (aux["inlet_T_C"] + 273.15))
    sec = DiscStackSection(R=R, disc_radius=x["disc_diameter_frac"] * R, disc_thickness=disc_thickness_m,
                           rel_air_density=rel_air_density,
                           spacing=x["disc_spacing_m"], n_periods=n_periods,
                           voltage=pol_c * x["corona_voltage_kV"] * 1e3, mobility=rom.ION_MOBILITY[pol_c],
                           edge_cells=pre["edge_cells"], h_coarse=pre["h_coarse"], growth=pre["growth"])
    cls = [c for c in ci["classes"] if c["number_flow"] > 0]
    if not cls:
        return ok, base
    Q = ci["U"] * np.pi * R ** 2
    radii = np.array([c["radius"] for c in cls])
    conc = np.array([c["number_flow"] for c in cls]) / Q
    q_in = np.array([c["charge"] for c in cls])
    try:
        res = march_section(sec, pol_c, ci["U"], radii, conc, q_in=q_in, n_per_class=pre["n_per_class"],
                            coupling_iters=pre["coupling_iters"], seed=seed)
    except Exception as e:  # a solver failure is ok=False, like any other oracle
        return False, f"ehd failure: {e}"
    if not all(p["converged"] for p in res.periods):
        return False, "ehd corona did not converge in every period"

    m = rom.RHO_L * 4 / 3 * np.pi * radii ** 3
    nf = conc * Q
    liq_out = float(np.sum(res.escaped_frac * nf * m))
    liq_disc = float(np.sum(res.emitter_frac * nf * m))
    I_c = float(abs(np.sum(res.current_per_period)))
    # replace the ROM's corona-stage outputs; keep its vapor and spray stage
    vapor_in, vapor_out = ci["vapor_in"], base["vapor_removed_frac"]
    w_frac_out = 1 - vapor_out
    total_in = vapor_in + ci["mist_liq_in"]
    total_out = vapor_in * w_frac_out + liq_out
    elec_W = x["corona_voltage_kV"] * 1e3 * I_c + ci["elec_W_spray"]
    p = dict(base)
    p.pop("corona_inlet", None)
    p.pop("profile", None)
    p.update({
        "outlet_total_water_frac": float(total_out / total_in),
        "liquid_escape_frac": float(liq_out / max(ci["mist_liq_in"] + ci["spray_in"], 1e-12)),
        "recovered_L_h": float((total_in - total_out) * 3600),
        "electrical_power_W": float(elec_W),
        "corona_current_mA": float(I_c * 1e3),
        "wall_field_kV_cm": float(abs(np.mean(res.wall_field_kV_cm))),
        "specific_energy_Wh_L": float((elec_W + ci["fan_W"]) / max((total_in - total_out) * 3600, 1e-3)),
        # EHD-only diagnostics
        "ehd_disc_fouling_frac": float(liq_disc / max(ci["mist_liq_in"] + ci["spray_in"], 1e-12)),
        "ehd_quenching_frac": float(1 - abs(res.current_per_period[0]) / max(abs(res.current_no_mist), 1e-30))
        if res.current_no_mist else 0.0,
        "ehd_current_no_mist_mA": float(abs(res.current_no_mist) * n_periods * 1e3),
        "ehd_n_periods": n_periods,
        "fidelity_tier": preset_name,
    })
    return True, p


def worker_fn(conn, params, aux, preset_name):
    try:
        ok, payload = simulate(params, aux, preset_name)
        if ok:
            payload.pop("profile", None)
            payload.pop("corona_inlet", None)
        conn.send((ok, payload))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    return (np.asarray(params, float), {k: float(aux[k]) for k in rom.AUX_NAMES}, fidelity_name)

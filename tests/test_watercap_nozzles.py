"""watercap_nozzles: the Wang-Lefebvre check against Tokgoz 2019 and the feasibility verdicts on the
library's characteristic designs (pure Python, no StoneLab needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import watercap_nozzles as wn  # noqa: E402

BASE = dict(tube_diameter_m=0.30, spray_gsd=1.5, spray_q_frac=0.0, spray_T_C=10.0)


def test_lefebvre_wang_reproduces_tokgoz_check():
    # StoneLab validation/tokgoz2019 data.json "our_check_vs_A1": 210, 157, 127 um (A-1, water, measured dP)
    for (m, dP), ref in zip(((0.100, 2.42e5), (0.173, 5.72e5), (0.240, 10.34e5)), (210, 157, 127)):
        smd = wn.lefebvre_wang_smd(m, dP, 5.4e-3, 41.5, 20.0)[0] * 1e6
        assert smd == pytest.approx(ref, rel=0.03)


def test_smd_from_mmd_lognormal():
    assert wn.smd_from_mmd(100e-6, 1.0) == pytest.approx(100e-6)
    assert wn.smd_from_mmd(100e-6, 1.5) == pytest.approx(92.1e-6, rel=1e-3)


def test_no_spray_is_feasible():
    assert wn.nozzle_feasibility(dict(BASE, spray_flow_L_min=0.0, spray_v0=5.0, spray_d_um=100.0))["feasible"]


def test_conventional_hollow_cone_is_feasible():
    # 6 L/min, ~190 um SMD at ~10 m/s launch: Tokgoz's own operating range
    r = wn.nozzle_feasibility(dict(BASE, spray_flow_L_min=6.0, spray_v0=10.0, spray_d_um=200.0))
    assert r["feasible"] and r["technology"] == "pressure_swirl"
    assert 1.0 < r["dP_bar"] < 2.0


def test_fine_spray_at_low_pressure_is_infeasible_but_relaxed():
    # 35 um at 12 m/s (~1.5 bar): no simplex nozzle at that pressure; reachable only at >10 bar
    r = wn.nozzle_feasibility(dict(BASE, spray_flow_L_min=8.0, spray_v0=12.0, spray_d_um=37.0))
    assert not r["feasible"] and "pressure-swirl" in r["reason"]
    assert r["feasible_relaxed"] and r["relaxed"]["dP_bar"] > 10
    assert r["relaxed"]["pump_W"] > 10 * r["pump_W_modeled"]


def test_slow_rain_is_a_jet_plate():
    r = wn.nozzle_feasibility(dict(BASE, tube_diameter_m=0.44, spray_flow_L_min=19.4, spray_v0=4.8, spray_d_um=500.0))
    assert r["feasible"] and r["technology"] == "plain_orifice" and r["regime"] == "rayleigh"
    assert r["n_nozzles_per_tier"] > 100


def test_dripping_and_overcharge_flagged():
    r = wn.nozzle_feasibility(dict(BASE, spray_flow_L_min=5.0, spray_v0=1.2, spray_d_um=150.0))
    assert not r["feasible"] and "drips" in r["reason"]
    r = wn.nozzle_feasibility(dict(BASE, spray_flow_L_min=5.0, spray_v0=10.0, spray_d_um=30.0, spray_q_frac=0.3))
    assert not r["feasible"] and "mC/kg" in r["reason"]

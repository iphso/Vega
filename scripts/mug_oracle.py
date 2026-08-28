"""oracle_base.py implementation for the mug/thermos thermal-design domain --
v3 (EXPERIMENT_LOG §50), wrapping `thermal_mug_spike.simulate_v3()`.
Supersedes v2's PARAM_DIM=8 after direct user request to "uplevel" the
domain further: a lid/top-loss model (v1/v2 modeled ZERO heat loss through
the top at all, backwards from reality -- an open liquid surface is
typically a FASTER loss pathway than an insulated side wall, not a
negligible one) and a variable body shape (radius at base/mid/rim instead
of a fixed-radius cylinder, so taper/flare/belly shapes are expressible).
See thermal_mug_spike.py's own v3 docstring for the physics and the real
initial-condition bug caught and fixed while building it.

Design vector x (PARAM_DIM=14): [r_base_mm, r_mid_mm, r_rim_mm,
t_wall_rim_mm, t_wall_base_mm, struct_material_idx, t_gap_mm,
insulation_material_idx, handle_length_mm, handle_diameter_mm,
handle_material_idx, lid_coverage_frac, t_lid_mm, lid_material_idx].
`lid_coverage_frac` continuously interpolates open-cup (0) to
fully-sealed-lid (1) rather than a discrete branch, consistent with every
other material/design choice in this domain being continuous.

Targets unchanged (temp_at_2h_C, mass_kg, touch_temp_60s_C,
handle_temp_60s_C) -- `mass_kg` now also includes lid mass when present,
`touch_temp_60s_C` now takes the max across rim/base/lid outer surfaces
instead of just rim/base.

Validity: same situation as v1/v2 -- the FD scheme is unconditionally
stable, so `ok=False` only means a structurally invalid input (any
non-positive geometry/thickness/length/diameter).
"""
from thermal_mug_spike import (  # noqa: F401
    HANDLE_MATERIALS,
    INSULATION_MATERIALS,
    PARAM_DIM_V3 as PARAM_DIM,
    PARAM_NAMES_V3 as PARAM_NAMES,
    STRUCTURAL_MATERIALS,
    simulate_v3,
)

TARGET_NAMES = ["temp_at_2h_C", "mass_kg", "touch_temp_60s_C", "handle_temp_60s_C"]
LOG_TARGET_NAMES = []
ZERO_INDICES = []
FIDELITY_PRESETS = {"low": "default"}
T_HORIZON_S = 2 * 3600.0


def worker_fn(conn, r_base_mm, r_mid_mm, r_rim_mm,
              t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
              t_gap_mm, insulation_material_idx,
              handle_length_mm, handle_diameter_mm, handle_material_idx,
              lid_coverage_frac, t_lid_mm, lid_material_idx):
    try:
        try:
            r = simulate_v3(r_base_mm, r_mid_mm, r_rim_mm,
                             t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                             t_gap_mm, insulation_material_idx,
                             handle_length_mm, handle_diameter_mm, handle_material_idx,
                             lid_coverage_frac, t_lid_mm, lid_material_idx,
                             t_max=T_HORIZON_S, record_at=60.0)
        except Exception as e:
            conn.send((False, f"sim error: {e}"))
            return
        if not r["valid"]:
            conn.send((False, "invalid input (non-finite, or non-positive geometry/thickness/length/diameter)"))
            return
        conn.send((True, {
            "temp_at_2h_C": float(r["final_liq_temp_C"]),
            "mass_kg": float(r["mass_kg"]),
            "touch_temp_60s_C": float(r["touch_temp_at_C"]),
            "handle_temp_60s_C": float(r["handle_temp_at_C"]),
        }))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    """params: (PARAM_DIM,) -- see this module's docstring for the ordering."""
    return tuple(float(v) for v in params)

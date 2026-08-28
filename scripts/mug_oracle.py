"""oracle_base.py implementation for the mug/thermos thermal-design domain --
v2 (EXPERIMENT_LOG §48), wrapping `thermal_mug_spike.simulate_v2()` instead
of v1's `simulate()`. Superseded v1's PARAM_DIM=3 (t_wall_mm, t_gap_mm,
k_gap) after direct user request for a richer parameterization: named
materials instead of one free-floating conductivity, a 2-band wall profile
(rim vs. base thickness) instead of a constant thickness, and a real handle
(weight + a genuine "does the handle get hot" safety target) -- see
thermal_mug_spike.py's own v2 docstring for the physics.

Design vector x (PARAM_DIM=8): [t_wall_rim_mm, t_wall_base_mm,
struct_material_idx, t_gap_mm, insulation_material_idx, handle_length_mm,
handle_diameter_mm, handle_material_idx]. The three `*_material_idx` params
are continuous, interpolating through STRUCTURAL_MATERIALS/
INSULATION_MATERIALS/HANDLE_MATERIALS (named real materials, sorted by
conductivity) rather than a single free-floating conductivity per v1 --
see thermal_mug_spike.material_props().

Targets: temp_at_2h_C, mass_kg (now includes wall+insulation+handle),
touch_temp_60s_C (body, worst of the two bands), handle_temp_60s_C (new --
grip-point safety). §48 found the last one is heavily floor-dominated
(most non-metal handle materials sit at ~ambient at 60s for realistic
lengths) -- a real physical finding (that's WHY those materials are used
for grips), not a data bug, but worth knowing before training on it.

Validity: same situation as v1 -- the FD scheme is unconditionally stable,
so `ok=False` only means a structurally invalid input.
"""
from thermal_mug_spike import (  # noqa: F401
    HANDLE_MATERIALS,
    INSULATION_MATERIALS,
    PARAM_DIM_V2 as PARAM_DIM,
    PARAM_NAMES_V2 as PARAM_NAMES,
    STRUCTURAL_MATERIALS,
    simulate_v2,
)

TARGET_NAMES = ["temp_at_2h_C", "mass_kg", "touch_temp_60s_C", "handle_temp_60s_C"]
LOG_TARGET_NAMES = []  # not yet examined for dynamic-range skew
ZERO_INDICES = []
FIDELITY_PRESETS = {"low": "default"}
T_HORIZON_S = 2 * 3600.0


def worker_fn(conn, t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
              t_gap_mm, insulation_material_idx,
              handle_length_mm, handle_diameter_mm, handle_material_idx):
    try:
        try:
            r = simulate_v2(t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                             t_gap_mm, insulation_material_idx,
                             handle_length_mm, handle_diameter_mm, handle_material_idx,
                             t_max=T_HORIZON_S, record_at=60.0)
        except Exception as e:
            conn.send((False, f"sim error: {e}"))
            return
        if not r["valid"]:
            conn.send((False, "invalid input (non-finite, or non-positive thickness/length/diameter)"))
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
    """params: (PARAM_DIM,) = [t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
    t_gap_mm, insulation_material_idx, handle_length_mm, handle_diameter_mm,
    handle_material_idx]."""
    return tuple(float(v) for v in params)

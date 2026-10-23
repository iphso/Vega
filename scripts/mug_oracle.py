"""oracle_base.py implementation for the mug/thermos thermal-design domain --
v6 physics (EXPERIMENT_LOG §59), wrapping `thermal_mug_spike.simulate_v6()`.

v6 is a full domain reset, not another incremental upgrade -- direct user
review of v5 found the whole foundation wrong: every prior version's design
vector included a double-wall insulation gap and a partial lid, making this
quietly an "insulated travel mug" domain rather than an actual "mug" one.
v6 drops that architecture entirely (single structural wall, no lid, no
insulation layer), fixes a real bug (the wall started AT FILL TEMPERATURE,
not ambient), adds real radiative loss (previously absent entirely), and
makes `liquid_volume_L` a genuine checked target instead of a computed-but-
ignored value. It also replaces v5's 2-cell (patch/bulk) angular
approximation with an actual 3D (axial x azimuthal x radial) grid -- see
thermal_mug_spike.py's own v6 header comment for the physics, the
shared-matrix batched-solve performance approach that keeps real 3D
affordable, and its verification. v1-v5 are left in thermal_mug_spike.py
untouched for reference, per this file's own established convention, but
are no longer used by this oracle or anything downstream of it.

Design vector x (PARAM_DIM=9): [r_base_mm, r_mid_mm, r_rim_mm,
t_wall_rim_mm, t_wall_base_mm, struct_material_idx, handle_length_mm,
handle_diameter_mm, handle_material_idx]. No lid/insulation-gap fields at
all -- not defaulted away, genuinely absent from the parameterization.

Targets (5, up from 4): temp_at_2h_C, mass_kg, touch_temp_60s_C,
handle_temp_60s_C, liquid_volume_L. The last is new -- a design that can't
hold a meaningful amount of liquid is rejected at the validity check
(`simulate_v6`'s own MIN_LIQUID_VOLUME_L), not silently included with a
tiny value.

Validity: same situation as v1-v5 -- every implicit half of the scheme is
unconditionally stable and the explicit halves (axial, azimuthal) self-select
their own sub-step counts against a computed stability bound, so `ok=False`
means either a structurally invalid input (any non-positive geometry/
thickness/length/diameter) or a liquid cavity below the minimum-viable-
volume floor.
"""
from thermal_mug_spike import (  # noqa: F401
    HANDLE_MATERIALS,
    PARAM_DIM_V6 as PARAM_DIM,
    PARAM_NAMES_V6 as PARAM_NAMES,
    STRUCTURAL_MATERIALS,
    TARGET_NAMES_V6 as TARGET_NAMES,
    simulate_v6,
)

LOG_TARGET_NAMES = []
ZERO_INDICES = []
FIDELITY_PRESETS = {"low": "default"}
T_HORIZON_S = 2 * 3600.0


def worker_fn(conn, r_base_mm, r_mid_mm, r_rim_mm,
              t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
              handle_length_mm, handle_diameter_mm, handle_material_idx):
    try:
        try:
            r = simulate_v6(r_base_mm, r_mid_mm, r_rim_mm,
                             t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                             handle_length_mm, handle_diameter_mm, handle_material_idx,
                             t_max=T_HORIZON_S, record_at=60.0)
        except Exception as e:
            conn.send((False, f"sim error: {e}"))
            return
        if not r["valid"]:
            conn.send((False, "invalid input (non-finite, non-positive geometry/thickness/length/diameter, "
                               "or liquid cavity below the minimum-viable-volume floor)"))
            return
        conn.send((True, {
            "temp_at_2h_C": float(r["final_liq_temp_C"]),
            "mass_kg": float(r["mass_kg"]),
            "touch_temp_60s_C": float(r["touch_temp_at_C"]),
            "handle_temp_60s_C": float(r["handle_temp_at_C"]),
            "liquid_volume_L": float(r["liquid_volume_L"]),
        }))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    """params: (PARAM_DIM,) -- see this module's docstring for the ordering."""
    return tuple(float(v) for v in params)

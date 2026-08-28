"""oracle_base.py implementation for the mug/thermos thermal-design domain
(EXPERIMENT_LOG §46) -- the 4th real domain module plugged into this
project's generic harness (after vmec_oracle.py, airfoil_oracle.py,
torax_oracle.py), and the first one whose underlying physics is
self-written rather than a wrapped external solver, so there's no license
question at all (§46's own reason for picking this domain).

Oracle: scripts/thermal_mug_spike.py's 1D thin-wall transient-conduction
FD model (implicit backward-Euler, verified against a closed-form
massless-wall limit and resolution-swept, see §46). This module is a thin
wrapper adding the oracle_base.py contract (worker_fn/params_to_worker_args/
TARGET_NAMES/etc.) on top of that already-verified simulate() function --
no new physics here.

Design vector x = [t_wall_mm, t_gap_mm, k_gap] (PARAM_DIM=3) -- flagged in
§46 as thin relative to every other domain here (VMEC 90+, airfoil 16,
TORAX 10); kept as-is for this first real dataset rather than expanded
preemptively, same "spike -> dataset -> THEN decide if richer" order every
other domain in this project followed.

No discrete or continuous aux conditioning (unlike VMEC's n_field_periods
or airfoil's (reynolds, alpha)) -- everything that varies is already in
params, same situation torax_oracle.py's own docstring describes.

Validity: unlike every wrapped-solver domain here, there's no external
"did not converge" flag -- the FD scheme is unconditionally stable
(backward Euler) for any positive thickness/conductivity, so a naive
`ok=False` on "time to reach 60C" not occurring within a fixed simulated
window would conflate two very different things: a broken candidate vs. a
GENUINELY EXCELLENT thermos that's still hot at the cutoff. Caught exactly
this way, not by inspection: an initial 20-candidate smoke test with
t_max=6h gave only a 20% hit rate, and every rejection was "still above
60C at 6h," not an actual instability -- i.e. the rejection was silently
biasing the dataset toward mediocre designs only, throwing away the best
part of the design space. Fixed by switching the primary target from a
threshold-crossing TIME (open-ended, needs an arbitrary cutoff) to the
LIQUID TEMPERATURE AT A FIXED TIME (2h) -- `simulate()`'s own
`final_liq_temp_C` already computes exactly this when `t_max=7200`,
always well-defined for any stable input, no rejection branch needed.
`ok=False` here now means only a genuinely invalid input (non-positive
thickness/conductivity) -- this domain's "hit rate" is expected to sit
near 100% by construction, a real and worth-noting contrast with every
wrapped-solver domain in this project (VMEC++/XFOIL/TORAX all have a
non-convergence failure mode that this one structurally doesn't).
"""
from thermal_mug_spike import PARAM_DIM, PARAM_NAMES, simulate  # noqa: F401

TARGET_NAMES = ["temp_at_2h_C", "mass_kg", "touch_temp_60s_C"]
LOG_TARGET_NAMES = []  # not yet examined for dynamic-range skew -- open item, see §46/§47
ZERO_INDICES = []  # no structurally-fixed coefficients for this parameterization
FIDELITY_PRESETS = {"low": "default"}  # single fixed grid resolution (N_WALL=20/N_GAP=40) -- no fidelity ladder yet
T_HORIZON_S = 2 * 3600.0  # the fixed evaluation time temp_at_2h_C is read at


def worker_fn(conn, t_wall_mm, t_gap_mm, k_gap):
    try:
        try:
            r = simulate(t_wall_mm, t_gap_mm, k_gap, t_max=T_HORIZON_S)
        except Exception as e:
            conn.send((False, f"sim error: {e}"))
            return
        if not r["valid"]:
            conn.send((False, "invalid input (non-finite, or non-positive thickness/conductivity)"))
            return
        conn.send((True, {
            "temp_at_2h_C": float(r["final_liq_temp_C"]),
            "mass_kg": float(r["mass_kg"]),
            "touch_temp_60s_C": float(r["touch_temp_60s_C"]),
        }))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    """params: (PARAM_DIM,) = [t_wall_mm, t_gap_mm, k_gap]. `aux` and
    `fidelity_name` accepted for interface consistency (no conditioning,
    only one fidelity tier)."""
    return (float(params[0]), float(params[1]), float(params[2]))

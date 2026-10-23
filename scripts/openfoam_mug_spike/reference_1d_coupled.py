"""Independent reference for the LIQUID-COUPLED wall BC (EXPERIMENT_LOG
§66) -- extends reference_1d.py's own from-scratch 1D plane-slab solver
with a lumped liquid energy balance, instead of reference_1d.py's fixed
T_LIQ. Still shares no code with thermal_mug_spike.py or
mug_openfoam_oracle.py.

Why this is a real independent check, not circular: the flux law used
here (`G_left*(T_liquid - T[0])`, i.e. film resistance 1/h in series with
the half-cell conduction resistance dx/2K, both evaluated against the
PREVIOUS timestep's converged state) is algebraically the exact same
quantity as what a converged `mixed` OpenFOAM boundary evaluates to:

    T_face = (h*T_liquid + gc*T0) / (h + gc),  gc = kappa*deltaCoeffs
    flux   = h*(T_liquid - T_face) = G_left*(T_liquid - T0)   [same value,
             since T_face's own definition makes the two resistors in the
             series network carry identical current]

So this script and the OpenFOAM `codedMixed` BC are two independently
written implementations of the SAME physical coupling law -- one on a
from-scratch implicit tridiagonal FD grid, the other on OpenFOAM's own
FVM/PIMPLE machinery -- exactly the kind of cross-check reference_1d.py
already did for the fixed-Ta case in §64.
"""
import numpy as np
from scipy.linalg import solve_banded

H_LIQ = 300.0
H_AIR = 15.0
T_AMB = 20.0
T0_LIQUID = 90.0

# Polypropylene, matching reference_1d.py / slab_case.
K = 0.22
RHO = 905.0
CP = 1920.0

T_WALL = 0.003
N = 60
DT = 0.5
T_MAX = 7200.0

# Liquid: matches coupled_spike test case's chosen values (a validation
# fixture, not a real mug's liquid volume -- just needs to match whatever
# the OpenFOAM test case declares).
A_LIQUID = 1e-4   # m^2, matches slab_case's y*z = 0.01*0.01 cross-section
M_LIQUID = 0.3    # kg
C_LIQUID = 4186.0  # J/(kg K), water


def run():
    dx = np.full(N, T_WALL / N)
    T = np.full(N, T_AMB)
    C = RHO * CP * dx

    G = np.empty(N - 1)
    for i in range(N - 1):
        G[i] = 1.0 / (dx[i] / (2 * K) + dx[i + 1] / (2 * K))
    G_left = 1.0 / (1.0 / H_LIQ + dx[0] / (2 * K))
    G_right = 1.0 / (1.0 / H_AIR + dx[-1] / (2 * K))

    T_liquid = T0_LIQUID
    n_steps = int(T_MAX / DT)
    touch_at_60s = None
    liquid_at_60s = None
    for step in range(n_steps):
        t_now = step * DT

        # Explicit-lagged liquid loss, using THIS step's pre-solve T[0]
        # (i.e. last timestep's converged wall temperature) -- same
        # lagging convention the coded BC uses (Tp is whatever the field
        # held before this timestep's solve).
        q_liquid = G_left * (T_liquid - T[0])  # W/m^2, positive = liquid losing heat
        T_liquid -= DT * (q_liquid * A_LIQUID) / (M_LIQUID * C_LIQUID)

        diag = C / DT
        lower = np.zeros(N)
        upper = np.zeros(N)
        rhs = C / DT * T
        for i in range(N - 1):
            diag[i] += G[i]; diag[i + 1] += G[i]
            upper[i] = -G[i]; lower[i + 1] = -G[i]
        diag[0] += G_left; rhs[0] += G_left * T_liquid
        diag[-1] += G_right; rhs[-1] += G_right * T_AMB
        ab = np.zeros((3, N))
        ab[0, 1:] = upper[:-1]; ab[1, :] = diag; ab[2, :-1] = lower[1:]
        T = solve_banded((1, 1), ab, rhs)

        if touch_at_60s is None and (t_now + DT) >= 60.0:
            touch_at_60s = T[-1]
            liquid_at_60s = T_liquid

    return touch_at_60s, liquid_at_60s, T[-1], T[0], T_liquid


if __name__ == "__main__":
    touch60, liq60, outer_2h, inner_2h, liq_2h = run()
    print(f"outer-surface T at 60s:  {touch60:.4f} C")
    print(f"liquid T at 60s:         {liq60:.4f} C")
    print(f"outer-surface T at 2h:   {outer_2h:.4f} C")
    print(f"inner-surface T at 2h:   {inner_2h:.4f} C")
    print(f"liquid T at 2h:          {liq_2h:.4f} C")

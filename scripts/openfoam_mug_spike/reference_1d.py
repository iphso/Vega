"""Independent 1D transient plane-slab conduction reference solver -- NOT
reusing any of thermal_mug_spike.py's own code, so a match against it (or
against OpenFOAM) is a real cross-check, not circular. Implicit
backward-Euler, standard finite-volume discretization, own from-scratch
implementation.

Problem: a t_wall-thick slab, material (k, rho, cp), starting at T_amb.
Left face: convective, h_liq, T_liq (fixed, liquid treated as an infinite
reservoir -- isolates wall conduction physics from the liquid's own energy
balance, deliberately, for this validation). Right face: convective,
h_air, T_amb.
"""
import numpy as np
from scipy.linalg import solve_banded

# Same values thermal_mug_spike.py uses (H_LIQ, H_AIR, T_AMB, T0_LIQUID)
H_LIQ = 300.0
H_AIR = 15.0  # a representative constant, not the temperature-dependent radiative-augmented model -- see spike notes
T_LIQ = 90.0
T_AMB = 20.0

# Polypropylene (STRUCTURAL_MATERIALS[3] in thermal_mug_spike.py)
K = 0.22
RHO = 905.0
CP = 1920.0

T_WALL = 0.003  # 3mm
N = 60          # cells across the wall thickness
DT = 0.5
T_MAX = 7200.0


def run():
    dx = np.full(N, T_WALL / N)
    T = np.full(N, T_AMB)
    C = RHO * CP * dx  # per unit area

    G = np.empty(N - 1)
    for i in range(N - 1):
        G[i] = 1.0 / (dx[i] / (2 * K) + dx[i + 1] / (2 * K))
    G_left = 1.0 / (1.0 / H_LIQ + dx[0] / (2 * K))
    G_right = 1.0 / (1.0 / H_AIR + dx[-1] / (2 * K))

    n_steps = int(T_MAX / DT)
    touch_at_60s = None
    for step in range(n_steps):
        t_now = step * DT
        diag = C / DT
        lower = np.zeros(N)
        upper = np.zeros(N)
        rhs = C / DT * T
        for i in range(N - 1):
            diag[i] += G[i]; diag[i + 1] += G[i]
            upper[i] = -G[i]; lower[i + 1] = -G[i]
        diag[0] += G_left; rhs[0] += G_left * T_LIQ
        diag[-1] += G_right; rhs[-1] += G_right * T_AMB
        ab = np.zeros((3, N))
        ab[0, 1:] = upper[:-1]; ab[1, :] = diag; ab[2, :-1] = lower[1:]
        T = solve_banded((1, 1), ab, rhs)
        if touch_at_60s is None and (t_now + DT) >= 60.0:
            touch_at_60s = T[-1]

    return touch_at_60s, T[-1], T[0]


if __name__ == "__main__":
    touch60, outer_2h, inner_2h = run()
    print(f"outer-surface T at 60s:  {touch60:.4f} C")
    print(f"outer-surface T at 2h:   {outer_2h:.4f} C")
    print(f"inner-surface T at 2h:   {inner_2h:.4f} C")

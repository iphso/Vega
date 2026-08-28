"""
Feasibility spike for a 5th Optimization Gym domain: transient heat conduction
through an insulated mug/thermos wall.

Not GPU, not a wrapped external simulator -- self-written 1D (thin-wall/planar
approximation) transient conduction, implicit (backward Euler) finite-volume,
liquid modeled as a lumped-capacitance node, ambient as a fixed-temperature
boundary. No license question: the physics model is authored here directly,
same category as VMEC's stiffness-method truss idea would be (nothing wrapped).

Design vector x = [t_wall_mm, t_gap_mm, k_gap]:
  - t_wall_mm: structural shell thickness (steel), mm
  - t_gap_mm: insulation layer thickness, mm
  - k_gap: insulation material conductivity, W/(m K) -- continuous stand-in
    for material choice (vacuum-like ~0.005 .. foam/air ~0.03 .. poor ~0.05)

Targets y:
  - retention_time_s: time for liquid to cool from 90C to 60C (None if not
    reached within t_max -- flagged, not silently clipped)
  - mass_kg: wall+insulation material mass (thin-wall volume x density),
    a cost/weight proxy -- the real design tension against retention_time
  - touch_temp_60s_C: outer wall surface temperature 60s after filling --
    a safety-relevant, genuinely time-lagged quantity distinct from the
    steady-state retention behavior (checked explicitly below, not assumed)
"""
import math
import numpy as np
from scipy.linalg import solve_banded

# --- physical constants (order-of-magnitude real values, not tuned) ---
CP_LIQUID = 4186.0        # J/(kg K), water/coffee
RHO_LIQUID = 1000.0       # kg/m3
K_STEEL = 15.0            # W/(m K), stainless steel
RHO_STEEL = 8000.0        # kg/m3
CP_STEEL = 500.0          # J/(kg K)
RHO_INSULATION = 40.0     # kg/m3, foam/aerogel-like
CP_INSULATION = 1400.0    # J/(kg K)

H_LIQ = 300.0             # W/(m2 K), liquid-side natural convection
H_AIR = 10.0              # W/(m2 K), ambient-side natural convection
T_AMB = 20.0              # deg C
T0_LIQUID = 90.0          # deg C, fill temperature
T_THRESHOLD = 60.0        # deg C, "still hot enough" cutoff

R_INNER = 0.035           # m, mug cavity radius
HEIGHT = 0.10             # m
A_INNER = 2 * math.pi * R_INNER * HEIGHT   # thin-wall approx: use inner area throughout
LIQ_VOLUME = math.pi * R_INNER**2 * HEIGHT
C_LIQ = RHO_LIQUID * LIQ_VOLUME * CP_LIQUID

N_WALL = 20
N_GAP = 40  # gap ranges wider (0.5-20mm+) than wall -- needs more nodes for the
            # early-time (60s) touch-temp transient to be resolution-converged;
            # confirmed by direct N_GAP sweep (15/40/100 -> 43.4/40.8/39.8 C at
            # t_gap=20mm), N_GAP=15 was under-resolved enough to produce a
            # non-physical non-monotonic touch_temp_60s vs. t_gap

PARAM_DIM = 3
PARAM_NAMES = ["t_wall_mm", "t_gap_mm", "k_gap"]


def simulate(t_wall_mm, t_gap_mm, k_gap, dt=5.0, t_max=6 * 3600.0,
             wall_mass_scale=1.0, record_touch_at=60.0):
    if t_wall_mm <= 0 or t_gap_mm <= 0 or k_gap <= 0:
        return dict(valid=False, retention_time_s=None, mass_kg=None, touch_temp_60s_C=None)

    t_wall = t_wall_mm / 1000.0
    t_gap = t_gap_mm / 1000.0
    n = N_WALL + N_GAP
    dx_wall = t_wall / N_WALL
    dx_gap = t_gap / N_GAP
    dx = np.concatenate([np.full(N_WALL, dx_wall), np.full(N_GAP, dx_gap)])
    k = np.concatenate([np.full(N_WALL, K_STEEL), np.full(N_GAP, k_gap)])
    rho = np.concatenate([np.full(N_WALL, RHO_STEEL), np.full(N_GAP, RHO_INSULATION)])
    cp = np.concatenate([np.full(N_WALL, CP_STEEL), np.full(N_GAP, CP_INSULATION)])
    C_cells = rho * cp * dx * A_INNER * wall_mass_scale

    G_cells = np.empty(n - 1)
    for i in range(n - 1):
        R = dx[i] / (2 * k[i] * A_INNER) + dx[i + 1] / (2 * k[i + 1] * A_INNER)
        G_cells[i] = 1.0 / R

    G_liq0 = 1.0 / (1.0 / (H_LIQ * A_INNER) + dx[0] / (2 * K_STEEL * A_INNER))
    G_air_last = 1.0 / (1.0 / (H_AIR * A_INNER) + dx[-1] / (2 * k[-1] * A_INNER))

    ndim = n + 1  # 0 = liquid, 1..n = wall/gap cells
    Ccap = np.concatenate([[C_LIQ], C_cells])

    diag = np.zeros(ndim)
    lower = np.zeros(ndim)
    upper = np.zeros(ndim)

    diag[0] = Ccap[0] / dt + G_liq0
    upper[0] = -G_liq0

    for i in range(1, ndim):
        cell = i - 1
        Gl = G_liq0 if cell == 0 else G_cells[cell - 1]
        Gr = G_air_last if cell == n - 1 else G_cells[cell]
        diag[i] = Ccap[i] / dt + Gl + Gr
        lower[i] = -Gl
        if cell != n - 1:
            upper[i] = -Gr

    ab = np.zeros((3, ndim))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]

    forcing_air = np.zeros(ndim)
    forcing_air[-1] = G_air_last * T_AMB

    T = np.full(ndim, T0_LIQUID)  # mug starts uniformly at fill temp

    n_steps = int(t_max / dt)
    retention_time = None
    touch_temp_60s = None
    touch_recorded = False

    for step in range(n_steps):
        t_now = step * dt
        rhs = Ccap / dt * T + forcing_air
        T = solve_banded((1, 1), ab, rhs)
        if not np.all(np.isfinite(T)):
            return dict(valid=False, retention_time_s=None, mass_kg=None, touch_temp_60s_C=None)
        if not touch_recorded and (t_now + dt) >= record_touch_at:
            touch_temp_60s = float(T[-1])
            touch_recorded = True
        if retention_time is None and T[0] <= T_THRESHOLD:
            retention_time = t_now + dt

    mass_kg = A_INNER * (t_wall * RHO_STEEL + t_gap * RHO_INSULATION)
    return dict(
        valid=True,
        retention_time_s=retention_time,
        mass_kg=mass_kg,
        touch_temp_60s_C=touch_temp_60s,
        final_liq_temp_C=float(T[0]),
    )


def analytic_time_constant_s(t_wall_mm, t_gap_mm, k_gap):
    """Massless-wall lumped-RC estimate, for verifying the FD implementation
    against a closed-form limit (§21/§45's own 'check against a known case'
    convention) -- NOT the model's actual prediction, which includes real
    wall/gap thermal mass."""
    t_wall = t_wall_mm / 1000.0
    t_gap = t_gap_mm / 1000.0
    R_liq = 1.0 / (H_LIQ * A_INNER)
    R_wall = t_wall / (K_STEEL * A_INNER)
    R_gap = t_gap / (k_gap * A_INNER)
    R_air = 1.0 / (H_AIR * A_INNER)
    tau = (R_liq + R_wall + R_gap + R_air) * C_LIQ
    return -tau * math.log((T_THRESHOLD - T_AMB) / (T0_LIQUID - T_AMB))


if __name__ == "__main__":
    print("=== Verification: near-massless wall/gap vs. analytic RC limit ===")
    for (tw, tg, kg) in [(1.0, 5.0, 0.03), (2.0, 10.0, 0.01)]:
        a = analytic_time_constant_s(tw, tg, kg)
        r = simulate(tw, tg, kg, wall_mass_scale=1e-6, t_max=max(a * 1.5, 14400))
        rt = r["retention_time_s"]
        if rt is None:
            print(f"  t_wall={tw}mm t_gap={tg}mm k_gap={kg}: never reached threshold (analytic={a:.1f}s)")
        else:
            print(f"  t_wall={tw}mm t_gap={tg}mm k_gap={kg}: "
                  f"FD(massless)={rt:.1f}s  analytic={a:.1f}s  "
                  f"rel_err={(rt-a)/a*100:.2f}%")

    print("\n=== Sweep 1: insulation thickness (t_wall=1mm, k_gap=0.03 foam-like) ===")
    print(f"{'t_gap_mm':>9} {'retention_s':>12} {'retention_min':>14} {'mass_kg':>9} {'touch60s_C':>11}")
    for tg in [0.0, 2.0, 5.0, 10.0, 20.0]:
        tg_eff = max(tg, 1e-3)
        r = simulate(1.0, tg_eff, 0.03)
        rt = r["retention_time_s"]
        rt_str = f"{rt:.1f}" if rt is not None else "not reached"
        rt_min_str = f"{rt/60:.1f}" if rt is not None else "-"
        print(f"{tg:>9.1f} {rt_str:>12} {rt_min_str:>14} "
              f"{r['mass_kg']:>9.4f} {r['touch_temp_60s_C']:>11.2f}")

    print("\n=== Sweep 2: insulation quality (t_wall=1mm, t_gap=5mm) ===")
    print(f"{'k_gap':>8} {'retention_s':>12} {'retention_min':>14} {'touch60s_C':>11}")
    for kg in [0.005, 0.015, 0.026, 0.05]:
        r = simulate(1.0, 5.0, kg)
        rt = r["retention_time_s"]
        rt_str = f"{rt:.1f}" if rt is not None else "not reached"
        rt_min_str = f"{rt/60:.1f}" if rt is not None else "-"
        print(f"{kg:>8.3f} {rt_str:>12} {rt_min_str:>14} "
              f"{r['touch_temp_60s_C']:>11.2f}")

    print("\n=== Redundancy check: does touch_temp_60s track retention_time 1:1, or carry independent info? ===")
    rng = np.random.default_rng(0)
    n_samples = 300
    tw = rng.uniform(0.5, 4.0, n_samples)
    tg = rng.uniform(0.5, 20.0, n_samples)
    kg = rng.uniform(0.005, 0.05, n_samples)
    retentions, touches, masses = [], [], []
    n_invalid = 0
    for i in range(n_samples):
        r = simulate(tw[i], tg[i], kg[i])
        if not r["valid"] or r["retention_time_s"] is None:
            n_invalid += 1
            continue
        retentions.append(r["retention_time_s"])
        touches.append(r["touch_temp_60s_C"])
        masses.append(r["mass_kg"])
    retentions = np.array(retentions)
    touches = np.array(touches)
    masses = np.array(masses)
    print(f"  {n_samples} random designs, {n_invalid} invalid/never-reached-threshold")
    print(f"  corr(retention_time, touch_temp_60s) = {np.corrcoef(retentions, touches)[0,1]:.3f}")
    print(f"  corr(retention_time, mass_kg)        = {np.corrcoef(retentions, masses)[0,1]:.3f}")
    print(f"  corr(touch_temp_60s, mass_kg)        = {np.corrcoef(touches, masses)[0,1]:.3f}")

    import time
    t0 = time.time()
    for i in range(50):
        simulate(tw[i], tg[i], kg[i])
    dt_batch = time.time() - t0
    print(f"\n=== Speed: {dt_batch/50*1000:.2f} ms/candidate (host CPU, single-threaded, N=30 cells) ===")

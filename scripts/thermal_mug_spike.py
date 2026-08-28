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


# ---------------------------------------------------------------------------
# v2 (EXPERIMENT_LOG §48): user asked for a richer parameterization -- named
# materials instead of one free-floating conductivity, a wall PROFILE instead
# of a constant thickness, and a real handle (weight + a "does the handle get
# hot" safety target). v1's `simulate()` is left untouched above for
# reference/regression-checking; v2 is a genuinely different design vector
# and gets its own function.
#
# Wall profile: kept to 2 axial bands (rim, base) rather than a fully
# height-resolved field -- a real generalization beyond v1's constant
# thickness (mugs often ARE thinner at the rim for a comfortable drinking
# edge and thicker at the base for stability/drop resistance) without a full
# 2D solve. Each band gets an equal half-share of the cylindrical side-wall
# area and is its own independent 1D radial+gap conduction chain, both
# drawing from the same liquid.
#
# Coupling scheme: v1 solved liquid+wall as ONE fully-implicit system (liquid
# is an unknown in the same tridiagonal solve). With 3 independent chains now
# (rim, base, handle) all touching the liquid/body, that would require a
# star-shaped (non-tridiagonal) matrix. Used explicit-implicit splitting
# instead: the liquid's own energy balance is updated EXPLICITLY each step
# from the previous step's band fluxes, and each chain (rim, base, handle) is
# solved fully implicitly using the previous step's liquid/body temperature
# as its own boundary value. Verified below that this still matches v1
# closely in the reducible case (same materials/thickness on both bands, no
# handle contribution to the liquid) -- not assumed equivalent.
#
# Handle: a genuine transient fin-conduction problem, not a quasi-steady
# shortcut -- checked first and rejected: a quasi-steady estimate (using the
# steady-state fin formula at each instant) implicitly assumes the handle's
# own thermal diffusion time is short vs. the 60s reporting window. Direct
# check: tau ~ L^2/alpha for a ~5cm steel handle is ~667s, for a low-k
# handle (precisely the "safe" designs we care about) it's even less settled
# within 60s -- neither case clears the 60s window, so the transient must be
# solved for real. Modeled as a 1D fin: implicit backward-Euler chain along
# the handle's length, WITH a lateral convective-loss term at every cell
# (h_air * perimeter * dx * (T-T_amb), the standard extended-surface/fin
# term v1's wall chains didn't need since they only lose heat at their two
# ends), Dirichlet-coupled at the base to the (previous-step) average of the
# two wall bands' outer surface temperatures, adiabatic (zero-flux) tip --
# the standard assumption for a fin whose tip area is small vs. its lateral
# area. Starts at ambient temperature, not fill temperature -- unlike the
# wall (which is assumed to instantly wet-contact the liquid at t=0), the
# handle is physically remote and only heats via conduction from the body.
# ---------------------------------------------------------------------------

# (name, k [W/(m K)], rho [kg/m3], cp [J/(kg K)]) -- sorted by k ascending in
# each table so a continuous index interpolates through real named materials,
# not just an arbitrary number range. Order-of-magnitude real values, not
# fit to any specific product.
STRUCTURAL_MATERIALS = [
    ("plastic", 0.2, 950.0, 1900.0),
    ("glass", 1.0, 2500.0, 840.0),
    ("ceramic", 1.5, 2300.0, 1050.0),
    ("steel", 15.0, 8000.0, 500.0),
]
INSULATION_MATERIALS = [
    ("vacuum", 0.005, 5.0, 500.0),
    ("aerogel", 0.015, 150.0, 1000.0),
    ("air_gap", 0.026, 1.2, 1005.0),
    ("foam", 0.03, 40.0, 1400.0),
]
HANDLE_MATERIALS = [
    ("silicone_rubber", 0.15, 1100.0, 1600.0),
    ("wood", 0.17, 600.0, 1700.0),
    ("plastic_grip", 0.2, 950.0, 1900.0),
    ("ceramic", 1.5, 2300.0, 1050.0),
    ("steel", 15.0, 8000.0, 500.0),
]

N_HANDLE = 20

PARAM_DIM_V2 = 8
PARAM_NAMES_V2 = [
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "t_gap_mm", "insulation_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
]


def material_props(idx, table):
    """idx: continuous, clamped to [0, len(table)-1] -- interpolates (k, rho,
    cp) between adjacent named reference materials sorted by k ascending."""
    idx = float(np.clip(idx, 0, len(table) - 1))
    xs = np.arange(len(table), dtype=float)
    ks = np.array([t[1] for t in table])
    rhos = np.array([t[2] for t in table])
    cps = np.array([t[3] for t in table])
    return float(np.interp(idx, xs, ks)), float(np.interp(idx, xs, rhos)), float(np.interp(idx, xs, cps))


def _step_chain(T, dx, k, rho, cp, area, dt, T_amb_right, h_right,
                 T_left_conv=None, h_left=None, T_left_dirichlet=None,
                 T_right_dirichlet=None, lateral_hP=0.0, T_lateral=T_AMB):
    """One implicit (backward-Euler) step of a 1D finite-volume conduction
    chain with fixed (not coupled-unknown) boundary values on both ends --
    either a convective left boundary (T_left_conv/h_left, what the wall
    bands use to reach the liquid) or a Dirichlet left boundary
    (T_left_dirichlet, what the handle's base end uses to reach the wall),
    a convective right boundary (T_amb_right/h_right; h_right=0 means
    adiabatic/no-flux, a free fin tip) OR a Dirichlet right boundary
    (T_right_dirichlet -- §56: the handle's SECOND attached end, once the
    loop shape got two attachment points instead of one attached end + one
    free tip), and an optional per-cell lateral convective loss term
    (lateral_hP = h*perimeter, zero for the wall bands which only lose heat
    at their two ends, nonzero for the handle which loses heat along its
    whole exposed length -- the fin equation's extra term)."""
    n = len(T)
    C = rho * cp * dx * area
    G = np.empty(n - 1)
    for i in range(n - 1):
        R = dx[i] / (2 * k[i] * area) + dx[i + 1] / (2 * k[i + 1] * area)
        G[i] = 1.0 / R

    diag = C / dt + lateral_hP * dx
    lower = np.zeros(n)
    upper = np.zeros(n)
    rhs = C / dt * T + lateral_hP * dx * T_lateral

    for i in range(n - 1):
        diag[i] += G[i]
        diag[i + 1] += G[i]
        upper[i] = -G[i]
        lower[i + 1] = -G[i]

    if T_left_dirichlet is not None:
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = T_left_dirichlet
    else:
        G_left = 1.0 / (1.0 / (h_left * area) + dx[0] / (2 * k[0] * area))
        diag[0] += G_left
        rhs[0] += G_left * T_left_conv

    if T_right_dirichlet is not None:
        diag[-1] = 1.0
        lower[-1] = 0.0
        rhs[-1] = T_right_dirichlet
    elif h_right != 0.0:
        G_right = 1.0 / (1.0 / (h_right * area) + dx[-1] / (2 * k[-1] * area))
        diag[-1] += G_right
        rhs[-1] += G_right * T_amb_right
    # h_right == 0.0 and no T_right_dirichlet: adiabatic tip, no term added

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def simulate_v2(t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 t_gap_mm, insulation_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0):
    if min(t_wall_rim_mm, t_wall_base_mm, t_gap_mm, handle_length_mm, handle_diameter_mm) <= 0:
        return dict(valid=False)

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_ins, rho_ins, cp_ins = material_props(insulation_material_idx, INSULATION_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)

    area_band = A_INNER / 2.0  # rim band + base band split the cylindrical side wall evenly

    def build_band(t_wall_mm):
        t_wall, t_gap = t_wall_mm / 1000.0, t_gap_mm / 1000.0
        dx = np.concatenate([np.full(N_WALL, t_wall / N_WALL), np.full(N_GAP, t_gap / N_GAP)])
        k = np.concatenate([np.full(N_WALL, k_struct), np.full(N_GAP, k_ins)])
        rho = np.concatenate([np.full(N_WALL, rho_struct), np.full(N_GAP, rho_ins)])
        cp = np.concatenate([np.full(N_WALL, cp_struct), np.full(N_GAP, cp_ins)])
        return dx, k, rho, cp

    dx_rim, k_rim, rho_rim, cp_rim = build_band(t_wall_rim_mm)
    dx_base, k_base, rho_base, cp_base = build_band(t_wall_base_mm)

    n_band = N_WALL + N_GAP
    T_rim = np.full(n_band, T0_LIQUID)
    T_base = np.full(n_band, T0_LIQUID)
    T_liq = T0_LIQUID

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)  # handle starts at room temp, not fill temp

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False

    for step in range(n_steps):
        t_now = step * dt
        flux_rim = H_LIQ * area_band * (T_liq - T_rim[0])
        flux_base = H_LIQ * area_band * (T_liq - T_base[0])
        T_liq_new = T_liq - dt / C_LIQ * (flux_rim + flux_base)

        T_rim_new = _step_chain(T_rim, dx_rim, k_rim, rho_rim, cp_rim, area_band, dt,
                                 T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
        T_base_new = _step_chain(T_base, dx_base, k_base, rho_base, cp_base, area_band, dt,
                                  T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)

        body_temp = 0.5 * (T_rim[-1] + T_base[-1])
        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=T_AMB, h_right=0.0, T_left_dirichlet=body_temp,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)

        ok = (np.isfinite(T_liq_new) and np.all(np.isfinite(T_rim_new))
              and np.all(np.isfinite(T_base_new)) and np.all(np.isfinite(T_handle_new)))
        if not ok:
            return dict(valid=False)

        T_liq, T_rim, T_base, T_handle = T_liq_new, T_rim_new, T_base_new, T_handle_new

        if not recorded and (t_now + dt) >= record_at:
            touch_temp_at = float(max(T_rim[-1], T_base[-1]))
            handle_temp_at = float(T_handle[-1])
            recorded = True

    mass_struct = area_band * ((t_wall_rim_mm + t_wall_base_mm) / 1000.0) * rho_struct
    mass_ins = area_band * 2 * (t_gap_mm / 1000.0) * rho_ins
    mass_handle = A_h * L_h * rho_handle
    mass_kg = mass_struct + mass_ins + mass_handle

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
    )


# ---------------------------------------------------------------------------
# v3 (EXPERIMENT_LOG §50): user asked to "uplevel" the domain further, picked
# two of four proposed directions -- a lid/top-loss model (the single
# biggest correctness gap in v1/v2: the top of the mug lost ZERO heat,
# which is backwards -- an open liquid surface is typically the FASTEST
# loss pathway on a real mug, not a negligible one) and a variable body
# shape (radius profile instead of a fixed-radius cylinder). v2's
# simulate_v2() is left untouched for reference/regression.
#
# Body shape: radius at 3 heights (base/mid/rim) instead of one constant --
# a taper (narrower base), a flare (wider rim), or a belly (wider middle)
# are all expressible, not just a straight cylinder. Total height stays
# fixed (HEIGHT=0.10m) -- varying it too is a real further idea, not done
# here (kept the parameter count from ballooning in one pass). Each
# half-height "band" is now a frustum (truncated cone), not a plain
# cylinder shell -- lateral area and volume both computed from the real
# frustum formulas, not the fixed-cylinder constants v1/v2 used.
#
# Lid: a single continuous `lid_coverage_frac` in [0,1] interpolates
# between a fully open cup (0) and a fully sealed lid (1) -- avoids a
# discrete open/lidded branch, consistent with this project's
# continuous-parameterization style everywhere else. The open fraction of
# the top loses heat directly to ambient via a temperature-DEPENDENT
# effective coefficient (h rises with liquid temperature, approximating
# evaporation's own dependence on vapor pressure, which rises with
# temperature -- a real, load-bearing simplification instead of a full
# Antoine-equation vapor-pressure model, chosen because the ballpark
# magnitude -- open coffee cools much faster than lidded, common
# experience -- matters more here than the exact curve shape). The
# covered fraction conducts through a lid (its own thickness + a material
# index, reusing STRUCTURAL_MATERIALS -- a lid is a similar kind of
# component to the wall shell, not worth a 4th materials table) to
# ambient at the plain (non-evaporative) side-wall convection coefficient.
# ---------------------------------------------------------------------------

H_AIR_TOP_OPEN = 15.0   # W/(m2 K), open-liquid-surface convection baseline --
                        # somewhat higher than the side wall's H_AIR=10 (an
                        # unobstructed buoyant plume above a hot liquid convects
                        # more freely than air along a vertical wall)
EVAP_COEFF = 60.0       # W/(m2 K)-equivalent, evaporation's contribution at
                        # full temperature difference -- an order-of-magnitude
                        # stand-in for "open coffee cools much faster than
                        # lidded," not a real vapor-pressure/Antoine-equation
                        # model; scales linearly with (T_liq-T_amb)/(T0-T_amb)
                        # so it fades as the liquid approaches ambient, same
                        # direction real evaporation rate does (falling vapor
                        # pressure difference), just not the same curve shape

N_LID = 15
N_AXIAL_BANDS = 6  # §56: was 2 (rim band + base band) -- see simulate_v3's
                   # own docstring for why this, not a finer N_WALL/N_GAP,
                   # is what actually gives the 3D view real height-wise
                   # spatial detail to show.

PARAM_DIM_V3 = 14
PARAM_NAMES_V3 = [
    "r_base_mm", "r_mid_mm", "r_rim_mm",
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "t_gap_mm", "insulation_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
    "lid_coverage_frac", "t_lid_mm", "lid_material_idx",
]


def frustum_lateral_area(r1, r2, h):
    return math.pi * (r1 + r2) * math.sqrt((r2 - r1) ** 2 + h ** 2)


def frustum_volume(r1, r2, h):
    return (math.pi * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)


def radius_at_height(r_base, r_mid, r_rim, height, z):
    """Same 2-segment (base->mid->rim) piecewise-linear profile the body
    geometry itself uses, evaluated at an arbitrary height z -- lets
    N_AXIAL_BANDS bands (or anything else) sample the real body profile at
    any height, not just the 3 design points."""
    if z <= height / 2:
        return r_base + (r_mid - r_base) * (z / (height / 2))
    return r_mid + (r_rim - r_mid) * ((z - height / 2) / (height / 2))


def simulate_v3(r_base_mm, r_mid_mm, r_rim_mm,
                 t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 t_gap_mm, insulation_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 lid_coverage_frac, t_lid_mm, lid_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0,
                 record_series=False, series_dt=None):
    """record_series=True (viewer-only -- never set by the oracle/bulk-
    generation path, which only needs the two scalar snapshots): also
    returns a time series of liquid temp + each band's outer temp + the
    handle's own full internal profile, for the viewer's
    temperature-over-time animation.

    series_dt=None (default) records every native integration step (dt) --
    §55 user feedback on the original 60s downsampling: the first minute or
    two is where almost all the interesting transient behavior actually
    happens, so a 60s stride made the animation's first couple of frames
    look like a discontinuous "flip" rather than a diffusion.

    §56: the body wall is divided into N_AXIAL_BANDS (6, up from 2 --
    rim/base) independent radial conduction chains stacked along the
    height, each its own frustum slice with wall thickness linearly
    interpolated between t_wall_base_mm/t_wall_rim_mm at that height.
    This is a real, deliberate distinction from just increasing N_WALL/
    N_GAP: those control resolution THROUGH the wall's thickness (radial),
    which does NOT add any new spatial detail ALONG the height, since the
    body's own visualization only ever reads one outer-surface value per
    band regardless of how finely that band's own radial direction is
    resolved. Going from 2 bands to 6 is what actually gives the 3D
    viewer real height-wise data to interpolate between, instead of a
    single straight-line blend across the whole body no matter how "fine"
    the mesh was inside those 2 fixed points.

    The handle is now attached at BOTH ends (T_left_dirichlet AND
    T_right_dirichlet, both driven by the same near-mid-height band) --
    §56 user feedback: the loop shape (added in §53 for realism) has two
    attachment points and no free tip, but the fin model still had one
    attached end + one free end, a real, named geometric mismatch. Fixing
    the PHYSICS to match the loop (rather than fudging the visualization
    further) makes the loop's own farthest point (from the mug) the
    genuine coolest point of a symmetric profile -- exactly what a
    real two-point-attached loop handle would do -- rather than an
    approximation of a profile that was never actually symmetric to begin
    with. `handle_temp_at_C`/`T_handle_tip` now report the PROFILE'S
    MIDDLE index (the new genuine coolest point), not the last index."""
    if series_dt is None:
        series_dt = dt
    if min(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm,
           t_gap_mm, handle_length_mm, handle_diameter_mm, t_lid_mm) <= 0:
        return dict(valid=False)
    lid_coverage_frac = float(np.clip(lid_coverage_frac, 0.0, 1.0))

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_ins, rho_ins, cp_ins = material_props(insulation_material_idx, INSULATION_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)
    k_lid, rho_lid, cp_lid = material_props(lid_material_idx, STRUCTURAL_MATERIALS)

    r_base, r_mid, r_rim = r_base_mm / 1000.0, r_mid_mm / 1000.0, r_rim_mm / 1000.0
    height = HEIGHT
    band_edges = [i * height / N_AXIAL_BANDS for i in range(N_AXIAL_BANDS + 1)]
    band_r = [radius_at_height(r_base, r_mid, r_rim, height, z) for z in band_edges]
    band_h = height / N_AXIAL_BANDS
    band_area = [frustum_lateral_area(band_r[i], band_r[i + 1], band_h) for i in range(N_AXIAL_BANDS)]
    vol_liq = sum(frustum_volume(band_r[i], band_r[i + 1], band_h) for i in range(N_AXIAL_BANDS))
    c_liq = RHO_LIQUID * vol_liq * CP_LIQUID
    a_top = math.pi * r_rim ** 2
    open_area = a_top * (1.0 - lid_coverage_frac)
    lid_area = a_top * lid_coverage_frac

    # wall thickness per band -- linear interpolation between the two real
    # design parameters, evaluated at each band's own height midpoint
    band_t_wall_mm = [t_wall_base_mm + (t_wall_rim_mm - t_wall_base_mm)
                       * ((band_edges[i] + band_edges[i + 1]) / 2 / height)
                       for i in range(N_AXIAL_BANDS)]

    def build_band(t_wall_mm):
        t_wall, t_gap = t_wall_mm / 1000.0, t_gap_mm / 1000.0
        dx = np.concatenate([np.full(N_WALL, t_wall / N_WALL), np.full(N_GAP, t_gap / N_GAP)])
        k = np.concatenate([np.full(N_WALL, k_struct), np.full(N_GAP, k_ins)])
        rho = np.concatenate([np.full(N_WALL, rho_struct), np.full(N_GAP, rho_ins)])
        cp = np.concatenate([np.full(N_WALL, cp_struct), np.full(N_GAP, cp_ins)])
        return dx, k, rho, cp

    bands = [build_band(band_t_wall_mm[i]) for i in range(N_AXIAL_BANDS)]
    n_band = N_WALL + N_GAP
    T_bands = [np.full(n_band, T0_LIQUID) for _ in range(N_AXIAL_BANDS)]
    T_liq = T0_LIQUID
    mid_band_idx = N_AXIAL_BANDS // 2  # both handle attachment points treated as near mid-height

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)
    handle_mid_idx = N_HANDLE // 2

    has_lid = lid_area > 1e-9
    if has_lid:
        t_lid = t_lid_mm / 1000.0
        dx_lid = np.full(N_LID, t_lid / N_LID)
        k_lid_arr = np.full(N_LID, k_lid)
        rho_lid_arr = np.full(N_LID, rho_lid)
        cp_lid_arr = np.full(N_LID, cp_lid)
        # Starts at ambient, not fill temperature -- like the handle (T_handle
        # above) and unlike the wall bands: a lid isn't liquid-wetted at t=0
        # the way a submerged wall shell is, it only heats via conduction
        # from the liquid/vapor below.
        T_lid = np.full(N_LID, T_AMB)
    else:
        T_lid = None

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False
    series = None
    if record_series:
        series = {"t_s": [], "T_liq": [], "T_bands_outer": [], "T_rim_outer": [], "T_base_outer": [],
                  "T_lid_outer": [], "T_handle_tip": [], "T_handle_profile": []}
        next_sample_t = 0.0

    def record_frame(t_now):
        series["t_s"].append(t_now)
        series["T_liq"].append(float(T_liq))
        series["T_bands_outer"].append([float(Tb[-1]) for Tb in T_bands])
        series["T_rim_outer"].append(float(T_bands[-1][-1]))  # convenience aliases for the 2D chart
        series["T_base_outer"].append(float(T_bands[0][-1]))
        series["T_lid_outer"].append(float(T_lid[-1]) if has_lid else None)
        series["T_handle_tip"].append(float(T_handle[handle_mid_idx]))
        series["T_handle_profile"].append(T_handle.tolist())

    for step in range(n_steps):
        t_now = step * dt
        if record_series and t_now >= next_sample_t:
            record_frame(t_now)
            next_sample_t += series_dt

        h_open = H_AIR_TOP_OPEN + EVAP_COEFF * max(0.0, T_liq - T_AMB) / (T0_LIQUID - T_AMB)
        flux_bands = sum(H_LIQ * band_area[i] * (T_liq - T_bands[i][0]) for i in range(N_AXIAL_BANDS))
        flux_open = h_open * open_area * (T_liq - T_AMB)
        flux_lid = H_LIQ * lid_area * (T_liq - T_lid[0]) if has_lid else 0.0
        T_liq_new = T_liq - dt / c_liq * (flux_bands + flux_open + flux_lid)

        T_bands_new = []
        for i in range(N_AXIAL_BANDS):
            dx_b, k_b, rho_b, cp_b = bands[i]
            T_bands_new.append(_step_chain(T_bands[i], dx_b, k_b, rho_b, cp_b, band_area[i], dt,
                                            T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ))

        body_temp = T_bands[mid_band_idx][-1]
        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=T_AMB, h_right=0.0,
                                    T_left_dirichlet=body_temp, T_right_dirichlet=body_temp,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)
        if has_lid:
            T_lid_new = _step_chain(T_lid, dx_lid, k_lid_arr, rho_lid_arr, cp_lid_arr, lid_area, dt,
                                     T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
        else:
            T_lid_new = None

        finite = (np.isfinite(T_liq_new) and all(np.all(np.isfinite(Tb)) for Tb in T_bands_new)
                  and np.all(np.isfinite(T_handle_new))
                  and (T_lid_new is None or np.all(np.isfinite(T_lid_new))))
        if not finite:
            return dict(valid=False)

        T_liq, T_bands, T_handle = T_liq_new, T_bands_new, T_handle_new
        if has_lid:
            T_lid = T_lid_new

        if not recorded and (t_now + dt) >= record_at:
            surfaces = [Tb[-1] for Tb in T_bands]
            if has_lid:
                surfaces.append(T_lid[-1])
            touch_temp_at = float(max(surfaces))
            handle_temp_at = float(T_handle[handle_mid_idx])
            recorded = True

    if record_series:
        # capture the final state too, so the animation's last frame is the
        # true end state rather than whatever the last series_dt-aligned
        # sample happened to land on
        record_frame(n_steps * dt)

    mass_struct = sum(band_area[i] * (band_t_wall_mm[i] / 1000.0) * rho_struct for i in range(N_AXIAL_BANDS))
    mass_ins = sum(band_area[i] * (t_gap_mm / 1000.0) * rho_ins for i in range(N_AXIAL_BANDS))
    mass_handle = A_h * L_h * rho_handle
    mass_lid = lid_area * (t_lid_mm / 1000.0) * rho_lid if has_lid else 0.0
    mass_kg = mass_struct + mass_ins + mass_handle + mass_lid

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
        liquid_volume_L=float(vol_liq * 1000.0),
        series=series,
        geometry=dict(r_base_m=r_base, r_mid_m=r_mid, r_rim_m=r_rim, height_m=HEIGHT,
                       has_lid=has_lid, lid_coverage_frac=lid_coverage_frac,
                       handle_length_m=L_h, handle_diameter_m=d_h),
    )


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

    print("\n=== v2 regression check: reduced to v1's case (equal bands, steel/foam, no handle feedback) ===")
    v1r = simulate(1.0, 5.0, 0.03, t_max=7200.0)
    v2r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0, t_max=7200.0)
    print(f"  v1 final_liq_temp_C={v1r['final_liq_temp_C']:.2f}  v2 final_liq_temp_C={v2r['final_liq_temp_C']:.2f}  "
          f"(explicit-liquid-coupling scheme, close-but-not-identical expected)")
    print(f"  v1 touch_temp_60s_C={v1r['touch_temp_60s_C']:.2f}  v2 touch_temp_at_C={v2r['touch_temp_at_C']:.2f}")

    print("\n=== v2 handle sanity: length sweep (steel handle, diameter=6mm -- worst case material) ===")
    for L in [5.0, 20.0, 50.0, 100.0]:
        r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, L, 6.0, 4.0, t_max=7200.0)
        print(f"  length={L:>6.1f}mm  handle_temp_60s={r['handle_temp_at_C']:.2f}C")

    print("\n=== v2 handle sanity: material sweep (length=50mm, diameter=6mm) ===")
    for name, k, rho, cp in HANDLE_MATERIALS:
        idx = [i for i, t in enumerate(HANDLE_MATERIALS) if t[0] == name][0]
        r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 50.0, 6.0, float(idx), t_max=7200.0)
        print(f"  {name:>16} (k={k:>5.2f})  handle_temp_60s={r['handle_temp_at_C']:.2f}C")

    print("\n=== v2 handle sanity: diameter sweep (length=50mm, steel) ===")
    for d in [2.0, 6.0, 12.0, 20.0]:
        r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 50.0, d, 4.0, t_max=7200.0)
        print(f"  diameter={d:>5.1f}mm  handle_temp_60s={r['handle_temp_at_C']:.2f}C  mass_kg={r['mass_kg']:.4f}")

    print("\n=== v2 speed ===")
    t0 = time.time()
    for _ in range(50):
        simulate_v2(1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/50*1000:.2f} ms/candidate")

    R_MM = R_INNER * 1000.0  # 35.0 -- v1/v2's fixed cylinder radius, for regression comparisons

    print("\n=== v3 vs. v2: a fully-sealed, well-insulated lid should APPROACH (not exactly match) v2's zero-top-loss idealization ===")
    v2r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0, t_max=7200.0)
    v3r = simulate_v3(R_MM, R_MM, R_MM, 1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0,
                       1.0, 5.0, 0.0, t_max=7200.0)  # lid_coverage=1, t_lid=5mm, plastic (lowest k available)
    print(f"  v2 (zero top loss, unphysical idealization)  final_liq_temp_C={v2r['final_liq_temp_C']:.2f}  touch_temp_at_C={v2r['touch_temp_at_C']:.2f}")
    print(f"  v3 (realistic sealed plastic lid)             final_liq_temp_C={v3r['final_liq_temp_C']:.2f}  touch_temp_at_C={v3r['touch_temp_at_C']:.2f}  "
          f"(v3 should run a bit COOLER than v2 -- a real lid still conducts some heat, v2 assumes none)")

    print("\n=== v3 lid sanity: coverage sweep (straight cylinder, same wall/gap as above) ===")
    for cov in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = simulate_v3(R_MM, R_MM, R_MM, 1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0,
                         cov, 3.0, 3.0, t_max=7200.0)  # t_lid=3mm steel lid when present
        print(f"  coverage={cov:.2f}  temp_at_2h_C={r['final_liq_temp_C']:.2f}  touch_temp_at_C={r['touch_temp_at_C']:.2f}")

    print("\n=== v3 body-shape sanity: belly sweep (r_base=r_rim=35mm fixed, r_mid varies) ===")
    for r_mid in [25.0, 35.0, 45.0, 60.0]:
        r = simulate_v3(35.0, r_mid, 35.0, 1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0,
                         0.5, 3.0, 3.0, t_max=7200.0)
        print(f"  r_mid={r_mid:>5.1f}mm  liquid_volume_L={r['liquid_volume_L']:.3f}  "
              f"mass_kg={r['mass_kg']:.4f}  temp_at_2h_C={r['final_liq_temp_C']:.2f}")

    print("\n=== v3 speed ===")
    t0 = time.time()
    for _ in range(50):
        simulate_v3(30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, 0.6, 3.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/50*1000:.2f} ms/candidate")

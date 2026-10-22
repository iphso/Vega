"""oracle_base.py implementation for the watercap domain -- a vertical,
grounded tube recovering water (vapor AND mist) from cooling-tower exhaust,
proposed by an external partner (EXPERIMENT_LOG "Watercap"): an
electrospray stage in the lower section sprays charged droplets downward
against the upflow, and a corona stage in the upper section (a stack of
electrode discs on the tube axis -- the partner's existing, about-to-deploy
prototype) charges whatever liquid remains and drives it to the wall.

THIS IS A SELF-WRITTEN REDUCED-ORDER MODEL (ROM), deliberately, and the
first domain since mug v1 to be one. The mug v1->v2 pivot established
"use the real accepted solver" -- but no accepted solver covers this
coupling out of the box (stock OpenFOAM has Lagrangian evaporation/
condensation, but no corona ion transport, no droplet field/diffusion
charging, and no Coulomb force on parcels; all three would be custom
code). So the ladder here is: `low`/`medium` = this ROM;
`ehd_coarse`/`ehd` = watercap_ehd.py, which keeps this ROM's spray stage and
replaces its corona stage with StoneLab's resolved, two-way coupled
axisymmetric corona + droplet model (validated against the exact coaxial
Kaptsov solution). A full gas-phase CFD tier (RANS + condensing spray) is
not built. The ROM is built only from textbook correlations, each named at
its point of use, so every one is a calibration target, not a hidden
assumption -- and the EHD tier has already shown its corona I-V is off by
4-45x for real 0.3 mm discs (onset 33 kV, not the 13.6 kV assumed here). The ROM is the cheap fidelity a multi-fidelity surrogate
(residual GP on CFD-minus-ROM) is built on top of -- not a stand-in for it.

Model (1D axial, cross-section well-mixed by turbulence, Re ~ 1e4-1e5):

  Air: plug flow upward, dry-air mass flow constant; state T_g(z), w(z)
    (humidity ratio). Two-way coupled to every droplet population through
    vapor and heat sources.
  Mist (inlet liquid water, lognormal by mass, N_BINS equal-mass bins):
    co-flows with the air at U - v_terminal, integrated in z together with
    the air (stiff ODE, LSODA). Droplet temperature is quasi-steady (micron
    drops relax in ~ms): it solves the convective-heat = latent-heat
    balance, including the Kelvin-Thomson surface vapor pressure.
  Spray (electrospray, counterflow): Lagrangian classes traced in TIME
    against the current air profile (drag + gravity, so small drops turn
    around and get entrained upward -- most cone-jet water drops will),
    with their own temperature ODE (they start at spray_T). Their vapor/
    heat sources feed back into the air march; outer fixed-point iteration
    with under-relaxation closes the counterflow coupling.
  Electrospray: cone-jet scaling (Fernandez de la Mora & Loscertales 1994:
    d ~ G (Q eps eps0 / K)^(1/3), I = f(eps) (gamma K Q / eps)^(1/2), f~18),
    onset voltage (Smith 1986), minimum flow rate Q0 = gamma eps eps0/(rho K).
    Per-droplet charge from I/(droplet rate), capped at 0.8 q_Rayleigh.
    Outside the cone-jet window the spray degrades to large, weakly charged
    drops (dripping) -- a flag, not a rejection.
  Spray-cloud space charge: unipolar Coulomb dispersion to the wall, loss
    rate Z rho_q / eps0 (Z = mobility), rho_q from the traced spray density.
  Scavenging of (neutral) mist by spray drops: geometric kernel x
    impaction efficiency (Calvert-type Stk/(Stk+0.25))^2 x enhancement from
    image-force attraction -- the weakest-grounded piece; flagged.
  Corona: onset by Peek's law at the disc edge, Townsend-type I-V,
    coaxial wall field including ion space charge, ion density from wall
    current density; droplet charging = Pauthenier field charging + White
    diffusion charging (signed, so opposite-polarity spray carryover is
    first neutralized, then re-charged); Deutsch turbulent-mixing wall loss
    2|w|/R with w = qE C_c / (6 pi mu a). Opposite-sign drops drift to the
    discs instead (electrode fouling, tracked separately).
  Arc limit: empirical average sparkover field for a rod-to-cylinder gap,
    polarity-dependent. UNCALIBRATED -- the partner's prototype I-V and
    sparkover data are exactly what should replace CORONA_IV_COEFF and
    SPARK_E_* below.
  Pressure drop: Blasius tube friction + per-disc annular-contraction loss
    + droplet drag reaction on the gas (integrated from the traces).

The thermodynamic point this ROM exists partly to make quantitative:
charge lowers a droplet's surface vapor pressure (Thomson term
q^2/(32 pi^2 eps0 a^4) x (1 - 1/eps_r)), but at the Rayleigh limit that
term cancels the Kelvin curvature term 2 gamma / a to within ~1%, so no
stable drop can take vapor out of air at or below saturation at its own
temperature. Vapor capture is therefore bounded by spray subcooling:
m_cond <~ m_spray cp (T_dew - T_spray) / h_fg -- the
`ceiling_gap_frac` target reports how far a design is from it.

Design vector x (PARAM_DIM=15) -- see PARAM_NAMES. The two polarities are
continuous [-1, 1] and read by sign (same treatment as mug v1's continuous
material indices). Electrospray and corona CURRENTS are outputs, not
inputs: at a given voltage, geometry and liquid conductivity they are
determined (cone-jet scaling; corona I-V).

aux (operating conditions, not design -- appended to X after PARAM_DIM,
the airfoil reynolds/alpha convention): AUX_NAMES.

Targets: TARGET_NAMES. `water_balance_residual` is the
energy_closure-style internal diagnostic (should be ~0).
"""
import numpy as np
from scipy.integrate import solve_ivp

PARAM_NAMES = [
    "air_flow_m3_s",            # volumetric air flow through one tube
    "tube_diameter_m",
    "spray_section_length_m",
    "corona_section_length_m",
    "nozzle_height_frac",       # nozzle position within the spray section, fraction of its length from the bottom
    "disc_diameter_frac",       # disc diameter / tube diameter
    "disc_spacing_m",
    "spray_flow_L_min",         # total electrospray liquid flow
    "spray_n_nozzles",          # rounded; per-nozzle flow sets the cone-jet regime
    "spray_T_C",
    "spray_conductivity_mS_m",  # spray water quality (DI ~0.1, tap ~50, tower blowdown ~200+)
    "spray_voltage_kV",
    "corona_voltage_kV",
    "corona_polarity",          # sign: +1 positive corona, -1 negative
    "spray_polarity_rel",       # sign: +1 same polarity as corona, -1 opposite
]
PARAM_DIM = len(PARAM_NAMES)
ZERO_INDICES = []
AUX_NAMES = ["inlet_T_C", "inlet_RH", "inlet_lwc_g_m3", "mist_mmd_um", "mist_gsd"]

TARGET_NAMES = [
    "outlet_total_water_frac",     # (vapor + liquid leaving top) / (vapor + liquid entering bottom) -- the main objective
    "vapor_removed_frac",          # (w_in - w_out) / w_in
    "liquid_escape_frac",          # liquid leaving top / (inlet mist + spray liquid)
    "recovered_L_h",               # inlet total water - outlet total water (can be negative: spray carryover)
    "electrical_power_W",
    "fan_power_W",
    "pressure_drop_Pa",
    "wall_field_kV_cm",
    "arc_margin",                  # 1 - V_corona / V_sparkover; <0 means the design arcs
    "corona_current_mA",
    "spray_current_uA",
    "ceiling_gap_frac",            # (adiabatic air+spray mixing ceiling - vapor removed) / inlet vapor; >= ~0
    "water_balance_residual",      # diagnostic, ~0
]
LOG_TARGET_NAMES = ["fan_power_W", "pressure_drop_Pa"]

# Fidelity = discretization. The resolved-corona tiers live in watercap_ehd.py
# (EHD_PRESETS); they share TARGET_NAMES with these.
FIDELITY_PRESETS = {
    "low": {"n_bins": 6, "n_z": 80, "outer_iters": 6, "spray_dt_frac": 0.5},
    "medium": {"n_bins": 12, "n_z": 200, "outer_iters": 12, "spray_dt_frac": 0.25},
}

# --- physical constants -----------------------------------------------------
P_ATM = 101325.0
R_D, R_V, R_U = 287.05, 461.5, 8.314
M_W = 0.018015
CP_A, CP_V, CP_L = 1006.0, 1860.0, 4186.0
RHO_L = 998.0
EPS0 = 8.854e-12
EPS_R_W = 80.0
E_CHG = 1.602e-19
K_B = 1.381e-23
G = 9.81
MU_G = 1.81e-5
K_G = 0.026
PR, SC = 0.71, 0.60

# --- fixed hardware constants (not design variables) ------------------------
NOZZLE_RADIUS_M = 2.5e-4       # 0.5 mm OD capillary
DISC_EDGE_RADIUS_M = 1.5e-4    # effective edge radius of a thin (~0.3 mm) disc -- UNCALIBRATED
SPRAY_V0_M_S = 2.0             # downward launch speed
FAN_EFF = 0.6
ION_MOBILITY = {+1: 1.4e-4, -1: 2.1e-4}   # m^2/Vs, humid air
# Calibration targets (see module docstring) -- deliberately named, NOT fitted:
CORONA_IV_COEFF = 1.0          # multiplier on the coaxial Townsend I-V; calibrate to prototype I-V
SPARK_E_INF_KV_CM = {+1: 4.5, -1: 8.0}   # long-gap average sparkover field, rod-cylinder
SPARK_E_SHORT_COEFF = 1.0      # (1 + c / sqrt(gap_cm)) short-gap correction
E_BREAKDOWN_V_M = 2.4e6       # uniform-field breakdown of humid air (~0.8 x dry 3 MV/m)
CONE_JET_MAX_Q_RATIO = 1e4     # Q/Q0 above which a single cone-jet is assumed to go multi-jet/unstable


# ---------------------------------------------------------------------------
# Thermo
# ---------------------------------------------------------------------------

def p_sat(T_C):
    """Buck (1981) saturation vapor pressure over flat water, Pa."""
    T_C = np.clip(np.asarray(T_C, dtype=float), -60.0, 150.0)
    return 611.21 * np.exp((18.678 - T_C / 234.5) * (T_C / (257.14 + T_C)))


def dp_sat_dT(T_C):
    h = 1e-3
    return (p_sat(T_C + h) - p_sat(T_C - h)) / (2 * h)


def w_from_pv(pv):
    return 0.622 * pv / (P_ATM - pv)


def pv_from_w(w):
    return w * P_ATM / (0.622 + w)


def h_fg(T_C):
    return 2.501e6 - 2370.0 * T_C


def surface_tension(T_C):
    return 0.0756 - 1.4e-4 * T_C


def vapor_diffusivity(T_C):
    return 2.11e-5 * ((T_C + 273.15) / 273.15) ** 1.94


def gas_density(T_C, w):
    pv = pv_from_w(w)
    return (P_ATM - pv) / (R_D * (T_C + 273.15)) + pv / (R_V * (T_C + 273.15))


def kelvin_thomson_factor(a, q, T_C):
    """Equilibrium surface vapor pressure / flat-water p_sat for a drop of
    radius a carrying charge q. The Thomson term carries (1 - 1/eps_r)."""
    gamma = surface_tension(T_C)
    term = 2 * gamma / a - (1 - 1 / EPS_R_W) * q ** 2 / (32 * np.pi ** 2 * EPS0 * a ** 4)
    # clipped: only nm-scale remnants of a fully evaporating drop reach it
    return np.exp(np.clip(M_W / (RHO_L * R_U * (T_C + 273.15)) * term, -5.0, 5.0))


def rayleigh_charge(a, T_C):
    return 8 * np.pi * np.sqrt(EPS0 * surface_tension(T_C) * a ** 3)


def terminal_velocity(a, rho_g):
    """Stokes with Schiller-Naumann correction, by fixed-point iteration."""
    v = 2 * RHO_L * G * a ** 2 / (9 * MU_G)
    for _ in range(8):
        re = np.abs(rho_g * v * 2 * a / MU_G)
        v = 2 * RHO_L * G * a ** 2 / (9 * MU_G * (1 + 0.15 * re ** 0.687))
    return v


def cunningham(a):
    lam = 0.066e-6
    kn = lam / a
    return 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))


# ---------------------------------------------------------------------------
# Droplet heat/mass transfer
# ---------------------------------------------------------------------------

def _transfer_coeffs(a, v_rel, rho_g):
    re = rho_g * np.abs(v_rel) * 2 * a / MU_G
    nu = 2 + 0.6 * np.sqrt(re) * PR ** (1 / 3)
    sh = 2 + 0.6 * np.sqrt(re) * SC ** (1 / 3)
    return nu, sh


def quasi_steady_drop_T(a, q, T_g, pv, v_rel, rho_g):
    """Droplet temperature where convective heat in balances latent heat
    out (the generalized wet-bulb, with Kelvin-Thomson). Newton,
    vectorized over drops."""
    nu, sh = _transfer_coeffs(a, v_rel, rho_g)
    D = vapor_diffusivity(T_g)
    rho_v_inf = pv / (R_V * (T_g + 273.15))
    T_d = np.full_like(a, T_g, dtype=float) - 0.5
    for _ in range(6):
        Tk = T_d + 273.15
        kt = kelvin_thomson_factor(a, q, T_d)
        rho_vs = p_sat(T_d) * kt / (R_V * Tk)
        drho = dp_sat_dT(T_d) * kt / (R_V * Tk) - rho_vs / Tk
        f = nu * K_G * (T_g - T_d) + h_fg(T_d) * sh * D * (rho_v_inf - rho_vs)
        df = -nu * K_G - h_fg(T_d) * sh * D * drho
        T_d = T_d - f / df
    return T_d


def evap_cutoff(mdot, m, m_ref):
    """Smoothly switches evaporation off as a drop's mass approaches zero
    (m << 1e-6 of its initial mass). Total evaporation of fine mist in
    subsaturated inlet air is real; without this the m -> 0 limit is
    singular (dm/dt stays finite while m vanishes) and LSODA fails."""
    m_floor = 1e-6 * m_ref
    return np.where(mdot < 0, mdot * (m / (m + m_floor)) ** 2, mdot)


def drop_mass_rate(a, q, T_d, T_g, pv, v_rel, rho_g):
    """dm/dt (kg/s, positive = condensation) and convective heat into the
    drop (W) at a given drop temperature."""
    nu, sh = _transfer_coeffs(a, v_rel, rho_g)
    D = vapor_diffusivity(T_g)
    rho_v_inf = pv / (R_V * (T_g + 273.15))
    rho_vs = p_sat(T_d) * kelvin_thomson_factor(a, q, T_d) / (R_V * (T_d + 273.15))
    mdot = 2 * np.pi * a * sh * D * (rho_v_inf - rho_vs)
    qconv = 2 * np.pi * a * nu * K_G * (T_g - T_d)
    return mdot, qconv


# ---------------------------------------------------------------------------
# Electrostatics
# ---------------------------------------------------------------------------

def peek_onset_field(r, rel_density=1.0):
    """Peek's law for a cylindrical conductor of radius r (m), V/m."""
    return 3.1e6 * rel_density * (1 + 0.0308 / np.sqrt(rel_density * r))


def corona_state(V, R, r_disc, spacing, polarity):
    """Onset voltage, current per unit length, wall field, ion density.
    Disc stack treated as an equivalent coaxial emitter of the disc-edge
    curvature; close disc spacing shields the edges (onset raised)."""
    gap = R - r_disc
    shield = 1.0 + np.exp(-2 * spacing / max(gap, 1e-6))
    E_on = peek_onset_field(DISC_EDGE_RADIUS_M)
    V_on = E_on * DISC_EDGE_RADIUS_M * np.log(2 * gap / DISC_EDGE_RADIUS_M) * shield
    mu = ION_MOBILITY[polarity]
    if V <= V_on:
        E_wall = V / (R * np.log(R / max(r_disc, 1e-4)))
        return dict(V_on=V_on, I_per_m=0.0, E_wall=E_wall, N_ion=0.0, mu_ion=mu)
    I_per_m = CORONA_IV_COEFF * 8 * np.pi * EPS0 * mu * V * (V - V_on) / (R ** 2 * np.log(R / max(r_disc, 1e-4)))
    # coaxial unipolar space-charge field at the wall
    E_lap = V_on / (R * np.log(R / max(r_disc, 1e-4)))
    E_wall = np.sqrt(E_lap ** 2 + I_per_m / (2 * np.pi * EPS0 * mu))
    J_wall = I_per_m / (2 * np.pi * R)
    N_ion = J_wall / (E_CHG * mu * E_wall)
    return dict(V_on=V_on, I_per_m=I_per_m, E_wall=E_wall, N_ion=N_ion, mu_ion=mu)


def sparkover_voltage(gap, polarity):
    gap_cm = gap * 100
    return 1e3 * SPARK_E_INF_KV_CM[polarity] * (1 + SPARK_E_SHORT_COEFF / np.sqrt(gap_cm)) * gap_cm


def charging_rate(a, q, cs, polarity, T_C):
    """Pauthenier field charging + White diffusion charging, signed so that
    `polarity * q` is charge of the corona's own sign."""
    if cs["N_ion"] <= 0:
        return np.zeros_like(a)
    p = 3 * EPS_R_W / (EPS_R_W + 2)
    q_s = 4 * np.pi * EPS0 * p * cs["E_wall"] * a ** 2
    tau = 4 * EPS0 / (cs["N_ion"] * E_CHG * cs["mu_ion"])
    qs_rel = polarity * q  # charge in the corona's own sign
    # Pauthenier for 0 <= q < q_s. For an opposite-sign drop (q < 0, e.g.
    # opposite-polarity spray carryover) its own field also collects ions:
    # rate ~ q_s/tau + 4|q|/tau (Gauss's law ion flux), continuous at q = 0 --
    # Pauthenier's (1 - q/q_s)^2 would over-extrapolate quadratically there.
    field = np.where(qs_rel >= q_s, 0.0,
                     np.where(qs_rel >= 0, (q_s / tau) * (1 - qs_rel / q_s) ** 2,
                              (q_s / tau) * (1 + 4 * np.abs(qs_rel) / q_s)))
    Tk = np.asarray(T_C, dtype=float) + 273.15
    c_bar = np.sqrt(8 * K_B * Tk / (np.pi * 4.8e-26))
    x = qs_rel * E_CHG / (4 * np.pi * EPS0 * a * K_B * Tk)
    # White (repulsive, x > 0) / Coulomb-attraction factor y/(1 - e^-y) (x < 0)
    y = np.clip(-x, 1e-9, 1e6)
    factor = np.where(x >= 0, np.exp(-np.clip(x, 0, 700)), y / -np.expm1(-y))
    diff = np.pi * a ** 2 * cs["N_ion"] * E_CHG * c_bar * factor
    return polarity * (field + diff)


def electrospray(Q_per, K, V, R, T_C, sign):
    """Cone-jet operating point for one nozzle. Returns droplet radius,
    per-droplet charge (signed), nozzle current, regime string."""
    gamma = surface_tension(T_C)
    V_on = np.sqrt(gamma * NOZZLE_RADIUS_M * np.cos(np.radians(49.3)) / (2 * EPS0)) * np.log(4 * R / NOZZLE_RADIUS_M)
    Q0 = gamma * EPS_R_W * EPS0 / (RHO_L * K)
    if V < V_on or Q_per <= 0:
        # no cone-jet: gravity dripping, Tate's law drop, weak induction charge
        d = (6 * NOZZLE_RADIUS_M * gamma / (RHO_L * G)) ** (1 / 3) * 2
        a = d / 2
        q = sign * 0.01 * rayleigh_charge(a, T_C)
        return a, q, 0.0, "dripping", V_on
    ratio = Q_per / Q0
    d_jet = 1.0 * (Q_per * EPS_R_W * EPS0 / K) ** (1 / 3)
    a = 1.89 * d_jet / 2
    I = 18.0 * np.sqrt(gamma * K * Q_per / EPS_R_W)
    regime = "cone_jet"
    if ratio < 1:
        regime = "below_min_flow"  # pulsating; scaling still used, flagged
    elif ratio > CONE_JET_MAX_Q_RATIO:
        regime = "multi_jet"       # flow too high for one jet: assume jets split, drops ~ as if Q/n_jets at the limit
        a = 1.89 * (CONE_JET_MAX_Q_RATIO * Q0 * EPS_R_W * EPS0 / K) ** (1 / 3) / 2
    n_rate = Q_per / (4 / 3 * np.pi * a ** 3)
    q = min(I / n_rate, 0.8 * rayleigh_charge(a, T_C))
    return a, sign * q, I, regime, V_on


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def _validate(p, aux):
    names = dict(zip(PARAM_NAMES, p))
    if not all(np.isfinite(v) for v in p) or not all(np.isfinite(aux[k]) for k in AUX_NAMES):
        return False, "non-finite input"
    for k in ("air_flow_m3_s", "tube_diameter_m", "spray_section_length_m", "corona_section_length_m",
              "disc_spacing_m", "spray_conductivity_mS_m"):
        if names[k] <= 0:
            return False, f"{k} must be positive"
    if not (0.0 < names["nozzle_height_frac"] <= 1.0):
        return False, "nozzle_height_frac must be in (0, 1]"
    if not (0.02 < names["disc_diameter_frac"] < 0.9):
        return False, "disc_diameter_frac must be in (0.02, 0.9)"
    for k in ("spray_flow_L_min", "spray_voltage_kV", "corona_voltage_kV"):
        if names[k] < 0:
            return False, f"{k} must be non-negative"
    if names["spray_n_nozzles"] < 0.5:
        return False, "spray_n_nozzles must be >= 1"
    if not (0.0 < aux["inlet_RH"] <= 1.0):
        return False, "inlet_RH must be in (0, 1]"
    if aux["inlet_lwc_g_m3"] < 0 or aux["mist_mmd_um"] <= 0 or aux["mist_gsd"] < 1.0:
        return False, "invalid mist distribution"
    return True, ""


def _mist_bins(lwc_kg_m3, mmd, gsd, n_bins):
    from scipy.stats import norm
    qs = (np.arange(n_bins) + 0.5) / n_bins
    d = mmd * np.exp(np.log(max(gsd, 1.0 + 1e-6)) * norm.ppf(qs))
    m_drop = RHO_L * np.pi * d ** 3 / 6
    n_per_m3 = (lwc_kg_m3 / n_bins) / m_drop
    return d / 2, n_per_m3


def simulate(p, aux, fid, spray_override=None):
    """Plain (ok, payload) computation, no subprocess wrapping (the
    photonics compute_efficiencies() split).

    spray_override: optional dict(d_mmd_m, gsd, q_frac, v0, n_bins) -- an
    atomizer (pressure/air-blast nozzle, optionally induction-charged)
    instead of the electrospray scaling: lognormal drops by mass, charge a
    fixed fraction of Rayleigh, no spray current or cone-jet. The
    electrospray inputs (n_nozzles, conductivity, spray voltage) are then
    ignored. Used by the candidate campaign, whose designs are atomizers
    (EXPERIMENT_LOG finding 3: cone-jet electrospray at L/min is
    unrealizable)."""
    ok, err = _validate(p, aux)
    if not ok:
        return False, f"invalid input: {err}"
    x = dict(zip(PARAM_NAMES, (float(v) for v in p)))
    n_bins, n_z, outer_iters = fid["n_bins"], fid["n_z"], fid["outer_iters"]
    pol_c = 1 if x["corona_polarity"] >= 0 else -1
    pol_s = pol_c if x["spray_polarity_rel"] >= 0 else -pol_c

    R = x["tube_diameter_m"] / 2
    A = np.pi * R ** 2
    Ls, Lc = x["spray_section_length_m"], x["corona_section_length_m"]
    L = Ls + Lc
    z_nozzle = x["nozzle_height_frac"] * Ls
    r_disc = x["disc_diameter_frac"] * R

    T_in, w_in = aux["inlet_T_C"], w_from_pv(aux["inlet_RH"] * p_sat(aux["inlet_T_C"]))
    rho_in = gas_density(T_in, w_in)
    mdot_air = rho_in * x["air_flow_m3_s"] / (1 + w_in)  # dry air kg/s
    U = x["air_flow_m3_s"] / A

    # corona
    V_c = x["corona_voltage_kV"] * 1e3
    cs = corona_state(V_c, R, r_disc, x["disc_spacing_m"], pol_c)
    I_corona = cs["I_per_m"] * Lc
    V_spark = sparkover_voltage(R - r_disc, pol_c)
    arc_margin = 1 - V_c / V_spark

    # electrospray
    n_noz = max(1, int(round(x["spray_n_nozzles"])))
    Q_tot = x["spray_flow_L_min"] / 1000 / 60
    K = x["spray_conductivity_mS_m"] * 1e-3
    a_s, q_s, I_noz, regime, V_es_on = electrospray(Q_tot / n_noz, K, x["spray_voltage_kV"] * 1e3, R, x["spray_T_C"], pol_s)
    I_spray = n_noz * I_noz
    V_spark_es = sparkover_voltage(R, pol_s)
    # main drops + satellites (15% of volume at half radius, ~same charge/area ratio)
    spray_cls = [] if Q_tot <= 0 else [(a_s, q_s, 0.85 * Q_tot), (a_s / 2, q_s / 4, 0.15 * Q_tot)]
    v0_spray = SPRAY_V0_M_S
    if spray_override is not None:
        so = spray_override
        a_b, _ = _mist_bins(1.0, so["d_mmd_m"], so.get("gsd", 1.5), int(so.get("n_bins", 4)))  # returns radii
        spray_cls = [] if Q_tot <= 0 else [(a, pol_s * so.get("q_frac", 0.0) * rayleigh_charge(a, x["spray_T_C"]),
                                             Q_tot / len(a_b)) for a in a_b]
        a_s, q_s = float(np.exp(np.mean(np.log(a_b)))), pol_s * so.get("q_frac", 0.0) * rayleigh_charge(float(np.exp(np.mean(np.log(a_b)))), x["spray_T_C"])
        I_spray, regime = 0.0, "atomizer"
        v0_spray = so.get("v0", SPRAY_V0_M_S)

    z_edges = np.linspace(0, L, n_z + 1)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    dz = L / n_z
    in_corona = zc >= Ls
    nb = n_bins
    a_m0, n_m0 = _mist_bins(aux["inlet_lwc_g_m3"] * 1e-3, aux["mist_mmd_um"] * 1e-6, aux["mist_gsd"], nb)
    Ndot_m = n_m0 * x["air_flow_m3_s"]  # mist drops / s per bin
    m0 = RHO_L * 4 / 3 * np.pi * a_m0 ** 3
    spray_in = RHO_L * Q_tot

    # Spray classes split by whether the upflow can carry them. Entrained
    # drops (v_t < 0.8 U) turn around within ~tau_p * v0 (mm) and then co-flow
    # with the air, so they join the implicit z-march (stiffly coupled to the
    # gas, own space charge computed locally). Only drops that genuinely fall
    # against the flow get the time-traced counterflow (PSI-cell) treatment,
    # where the coupling is weak because they are big and few.
    ent, fall = [], []
    for (a0, q0, Qc) in spray_cls:
        (ent if terminal_velocity(a0, rho_in) < 0.8 * U else fall).append((a0, q0, Qc))
    ne = len(ent)
    a_e0 = np.array([c[0] for c in ent])
    q_e0 = np.array([c[1] for c in ent])
    m_e0 = RHO_L * 4 / 3 * np.pi * a_e0 ** 3
    Ndot_e = np.array([RHO_L * c[2] for c in ent]) / np.maximum(m_e0, 1e-30) if ne else np.zeros(0)
    E_app0 = x["spray_voltage_kV"] * 1e3 / (R * np.log(R / NOZZLE_RADIUS_M)) if spray_override is None else 0.0

    # state layout
    iM, iQ, iF, iD = 2, 2 + nb, 2 + 2 * nb, 2 + 3 * nb
    jM, jT, jQ, jF, jW, jE = (2 + 4 * nb + k * ne for k in range(6))
    n_state = 2 + 4 * nb + 6 * ne

    S_v = np.zeros(n_z)   # falling-spray vapor source into gas, kg/s per cell (+ = evaporation)
    S_h = np.zeros(n_z)   # falling-spray sensible heat into gas, W per cell
    rho_q_fall = np.zeros(n_z)
    fall_kern = []        # (number density per cell, radius, charge, axial velocity)

    def scav_rate(a_m, v_m, collectors):
        """First-order mist loss rate (1/s) to spray collectors, each given as
        (number density, radius, charge, absolute axial velocity)."""
        tot = np.zeros_like(a_m)
        for (n_c, a_c, q_c, v_c) in collectors:
            if n_c <= 0:
                continue
            v_rel = np.maximum(np.abs(v_c - v_m), 1e-3)
            stk = RHO_L * (2 * a_m) ** 2 * v_rel / (18 * MU_G * 2 * a_c)
            e_imp = (stk / (stk + 0.25)) ** 2
            # image-force attraction, charged collector / neutral mist (Pilat-style; order of magnitude only)
            kim = q_c ** 2 * a_m ** 2 * cunningham(a_m) / (3 * np.pi ** 2 * EPS0 * MU_G * (2 * a_c) ** 5 * v_rel)
            e = np.minimum(e_imp + np.minimum((15 * np.pi / 8 * kim) ** 0.4, 1.0), 1.0)
            tot += n_c * e * np.pi * (a_c + a_m) ** 2 * v_rel
        return tot

    def unpack_ent(y):
        m = np.maximum(y[jM:jM + ne], 1e-30)
        return m, y[jT:jT + ne], y[jQ:jQ + ne], np.maximum(y[jF:jF + ne], 0.0)

    def rhs(z, y, spray_on):
        k = min(int(z / dz), n_z - 1)
        Tg, w = y[0], max(y[1], 1e-6)
        rho_g = gas_density(Tg, w)
        pv = pv_from_w(w)
        d = np.zeros(n_state)
        # entrained spray
        rho_q = rho_q_fall[k]
        collectors = [(fk[0][k], fk[1], fk[2], fk[3]) for fk in fall_kern] if z < Ls else []
        dw_e = dT_e = 0.0
        if spray_on and ne and y[jF:jF + ne].max() > 1e-9:  # a fully deposited spray is frozen, not integrated
            m_e, T_e, q_e, F_e = unpack_ent(y)
            a_e = (m_e / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)
            q_e = np.sign(q_e) * np.minimum(np.abs(q_e), rayleigh_charge(a_e, T_e))  # Rayleigh fission caps charge
            vt_e = terminal_velocity(a_e, rho_g)
            v_e = np.maximum(U - vt_e, 0.05 * U)
            n_e = Ndot_e * F_e / (A * v_e)
            rho_q += float((n_e * q_e).sum())
            mdot_e, qconv_e = drop_mass_rate(a_e, q_e, T_e, Tg, pv, vt_e, rho_g)
            mdot_e = evap_cutoff(mdot_e, m_e, m_e0)
            Zc = cunningham(a_e) / (6 * np.pi * MU_G * a_e)
            if z >= Ls:
                dq = charging_rate(a_e, q_e, cs, pol_c, Tg)
                w_dr = q_e * cs["E_wall"] * Zc * pol_c
                k_loss = 2 * np.abs(w_dr) / R
                to_elec = w_dr < 0
            else:
                dq = np.zeros(ne)
                k_loss = np.abs(q_e) * Zc * (abs(rho_q) / EPS0 + 2 * E_app0 * np.exp(-abs(z - z_nozzle) / R) / R)
                to_elec = np.zeros(ne, bool)
                collectors += list(zip(n_e, a_e, q_e, v_e))
            loss = Ndot_e * k_loss * F_e * m_e / v_e
            d[jM:jM + ne] = mdot_e / v_e
            d[jT:jT + ne] = (qconv_e + h_fg(T_e) * mdot_e) / (m_e * CP_L * v_e)
            d[jQ:jQ + ne] = dq / v_e
            d[jF:jF + ne] = -k_loss * F_e / v_e
            d[jW:jW + ne] = np.where(to_elec, 0.0, loss)
            d[jE:jE + ne] = np.where(to_elec, loss, 0.0)
            dw_e = -(Ndot_e * F_e * mdot_e / v_e).sum()
            dT_e = -(Ndot_e * F_e * qconv_e / v_e).sum()
        # mist
        m = np.maximum(y[iM:iM + nb], 1e-30)
        q = y[iQ:iQ + nb]
        F = np.maximum(y[iF:iF + nb], 0.0)
        a = (m / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)
        vt = terminal_velocity(a, rho_g)
        v = np.maximum(U - vt, 0.05 * U)
        Td = quasi_steady_drop_T(a, q, Tg, pv, vt, rho_g)
        mdot, qconv = drop_mass_rate(a, q, Td, Tg, pv, vt, rho_g)
        mdot_c = evap_cutoff(mdot, m, m0)
        qconv = qconv * np.where(mdot != 0, mdot_c / np.where(mdot != 0, mdot, 1.0), 1.0)  # quasi-steady: heat tracks latent
        mdot = mdot_c
        if z >= Ls:
            dqm = charging_rate(a, q, cs, pol_c, Tg)
            k_m = 2 * np.abs(q * cs["E_wall"] * cunningham(a) / (6 * np.pi * MU_G * a)) / R
        else:
            dqm = np.zeros(nb)
            # collector-relative velocity: difference of terminal velocities
            k_m = scav_rate(a, v, collectors) if collectors else np.zeros(nb)
        Nflow = Ndot_m * F
        d[iM:iM + nb] = mdot / v
        d[iQ:iQ + nb] = dqm / v
        d[iF:iF + nb] = -k_m * F / v
        d[iD:iD + nb] = Ndot_m * k_m * F * m / v
        d[1] = (-(Nflow * mdot / v).sum() + dw_e + S_v[k] / dz) / mdot_air
        d[0] = (-(Nflow * qconv / v).sum() + dT_e + S_h[k] / dz) / (mdot_air * (CP_A + w * CP_V))
        return d

    def march():
        y0 = np.zeros(n_state)
        y0[0], y0[1] = T_in, w_in
        y0[iM:iM + nb] = m0
        y0[iF:iF + nb] = 1.0
        atol = np.full(n_state, 1e-9)
        atol[0] = 1e-6
        atol[iM:iM + nb] = 1e-6 * m0
        atol[iQ:iQ + nb] = 1e-6 * rayleigh_charge(a_m0, T_in)
        atol[iD:iD + nb] = 1e-9 * max(spray_in + mdot_air * w_in, 1e-12)
        if ne:
            atol[jM:jM + ne] = 1e-6 * m_e0
            atol[jT:jT + ne] = 1e-6
            atol[jQ:jQ + ne] = 1e-6 * rayleigh_charge(a_e0, x["spray_T_C"])
            atol[jW:jW + ne] = atol[jE:jE + ne] = 1e-9 * max(spray_in, 1e-12)
        kw = dict(method="LSODA", rtol=1e-5, atol=atol, max_step=dz)
        pre = zc[zc < z_nozzle]
        s1 = solve_ivp(rhs, (0, z_nozzle), y0, t_eval=pre, args=(False,), **kw)
        if not s1.success:
            return None, s1.message
        y1 = s1.y[:, -1].copy() if s1.y.shape[1] and s1.t[-1] == z_nozzle else \
            solve_ivp(rhs, (0, z_nozzle), y0, args=(False,), **kw).y[:, -1].copy()
        if ne:
            y1[jM:jM + ne], y1[jT:jT + ne], y1[jQ:jQ + ne], y1[jF:jF + ne] = m_e0, x["spray_T_C"], q_e0, 1.0
        post = np.append(zc[zc >= z_nozzle], L)
        s2 = solve_ivp(rhs, (z_nozzle, L), y1, t_eval=post, args=(True,), **kw)
        if not s2.success:
            return None, s2.message
        Y = np.concatenate([s1.y, s2.y[:, :-1]], axis=1)
        return (Y, s2.y[:, -1]), ""

    fall_stats = dict(top=0.0, bottom=0.0, wall=0.0, elec=0.0, F_drag=0.0, cond=0.0)
    relax = 0.5
    n_iter = outer_iters if fall else 1
    T_g, w_g = np.full(n_z, T_in), np.full(n_z, w_in)
    rho_q_ent = np.zeros(n_z)
    for it in range(n_iter):
        if fall:
            res = _trace_falling(fall, z_nozzle, L, n_z, dz, A, R, U, T_g, w_g, rho_q_ent + rho_q_fall, in_corona,
                                 cs, pol_c, E_app0, fid["spray_dt_frac"], v0=v0_spray)
            S_v = relax * res["S_v"] + (1 - relax) * S_v if it else res["S_v"]
            S_h = relax * res["S_h"] + (1 - relax) * S_h if it else res["S_h"]
            rho_q_fall = relax * res["rho_q"] + (1 - relax) * rho_q_fall if it else res["rho_q"]
            fall_kern = res["kernels"]
            fall_stats = res["stats"]
        out, msg = march()
        if out is None:
            return False, f"ode failure: {msg}"
        Y, y_end = out
        dT_change = np.max(np.abs(Y[0] - T_g))
        T_g, w_g = Y[0], Y[1]
        if ne:
            a_e = (np.maximum(Y[jM:jM + ne], 1e-30) / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)
            v_e = np.maximum(U - terminal_velocity(a_e, rho_in), 0.05 * U)
            n_e = Ndot_e[:, None] * np.maximum(Y[jF:jF + ne], 0) / (A * v_e)
            rho_q_ent = (n_e * Y[jQ:jQ + ne]).sum(axis=0) * (zc >= z_nozzle) * (~in_corona)
        if it > 1 and dT_change < 1e-4:
            break

    # ---- droplet state entering the corona section (for the EHD tier, which
    # replaces this ROM's corona stage with a resolved 2D one)
    # last grid point BEFORE the corona section: the first one inside it has
    # already been charged by one cell of this ROM's own corona, and the EHD
    # tier then collected that pre-charge even below onset
    kc = max(int(np.searchsorted(zc, Ls)) - 1, 0)
    yc = Y[:, kc]
    inlet_cls = [{"radius": float((max(yc[iM + b], 1e-30) / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)),
                  "number_flow": float(Ndot_m[b] * max(yc[iF + b], 0.0)), "charge": float(yc[iQ + b]), "kind": "mist"}
                 for b in range(nb)]
    if ne and z_nozzle <= Ls:
        inlet_cls += [{"radius": float((max(yc[jM + c], 1e-30) / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)),
                       "number_flow": float(Ndot_e[c] * max(yc[jF + c], 0.0)), "charge": float(yc[jQ + c]),
                       "kind": "spray"} for c in range(ne)]

    # ---- outlet accounting at z = L
    w_out = y_end[1]
    m_end, F_end = y_end[iM:iM + nb], y_end[iF:iF + nb]
    mist_dep = float(y_end[iD:iD + nb].sum())
    mist_liq_out = float((Ndot_m * F_end * m_end).sum())
    mist_liq_in = float((Ndot_m * m0).sum())
    spray_stats = dict(fall_stats)
    if ne:
        ent_top = float((Ndot_e * np.maximum(y_end[jF:jF + ne], 0) * y_end[jM:jM + ne]).sum())
        ent_wall, ent_elec = float(y_end[jW:jW + ne].sum()), float(y_end[jE:jE + ne].sum())
        spray_stats["top"] += ent_top
        spray_stats["wall"] += ent_wall
        spray_stats["elec"] += ent_elec
        spray_stats["cond"] += ent_top + ent_wall + ent_elec - float((Ndot_e * m_e0).sum())
        # drag reaction of entrained drops on the gas: launch momentum reversal + suspended weight
        a_end = (np.maximum(Y[jM:jM + ne], 1e-30) / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)
        v_e = np.maximum(U - terminal_velocity(a_end, rho_in), 0.05 * U)
        susp = (Ndot_e[:, None] * np.maximum(Y[jF:jF + ne], 0) * Y[jM:jM + ne] * G / v_e * dz * (zc >= z_nozzle)).sum()
        spray_stats["F_drag"] += float(susp + (Ndot_e * m_e0 * (v0_spray + U)).sum())
    rho_q_prof = rho_q_ent + rho_q_fall
    vapor_in, vapor_out = mdot_air * w_in, mdot_air * w_out
    liq_out = mist_liq_out + spray_stats["top"]
    total_in = vapor_in + mist_liq_in
    total_out = vapor_out + liq_out

    # water balance over every in/out channel; nonzero means the ODE
    # tolerance or the falling-spray fixed point has not converged
    spray_out_all = spray_stats["top"] + spray_stats["bottom"] + spray_stats["wall"] + spray_stats["elec"]
    residual = (total_in + spray_in - total_out - mist_dep - (spray_out_all - spray_stats["top"])) / max(total_in, 1e-12)

    # Thermodynamic ceiling on vapor removal: adiabatically mix the air
    # stream with the spray stream to a common saturated equilibrium T_f
    # (electrostatic work is milliwatts against kW latent loads, and no
    # stable drop can sit below flat-water p_sat -- see module docstring).
    # Utilization = actual vapor removed / that ceiling; must be <= ~1.
    vap_max = _max_vapor_removal(mdot_air, T_in, w_in, spray_in, x["spray_T_C"])
    util = (vapor_in - mdot_air * w_out) / vap_max if vap_max > 1e-12 else 0.0

    # Spray-cloud space-charge limit: a unipolar cloud of density rho_q in a
    # grounded tube sets a wall field rho_q R / (2 eps0). Electrospray
    # scaling laws give the current a nozzle WANTS to emit; this is the
    # field that current would produce. Above breakdown the design arcs (or
    # in reality self-limits its current) -- either way the scaling-law
    # operating point is not realizable, so it is folded into arc_margin.
    # Evaluated at launch, where the cloud is densest: a charged spray
    # disperses to the wall within ~eps0/(Z rho_q), typically before the
    # first grid point, so the resolved profile alone understates it. Even
    # this treats the cone as filling the cross-section -- a lower bound.
    rho_launch = abs(float((Ndot_e * q_e0 / (A * np.maximum(U - terminal_velocity(a_e0, rho_in), 0.05 * U))).sum())) if ne else 0.0
    E_sc_wall = max(float(np.max(np.abs(rho_q_prof))), rho_launch) * R / (2 * EPS0) if spray_cls else 0.0

    # pressure drop
    rho_g = rho_in
    re_tube = rho_g * U * 2 * R / MU_G
    f_d = 0.316 * re_tube ** -0.25
    dp_fric = f_d * (L / (2 * R)) * 0.5 * rho_g * U ** 2
    beta2 = x["disc_diameter_frac"] ** 2
    n_discs = max(1, int(Lc / x["disc_spacing_m"]))
    cc = 0.61 + 0.39 * (1 - beta2) ** 3
    k_disc = (1 / (cc * (1 - beta2)) - 1) ** 2
    shield_dp = min(1.0, x["disc_spacing_m"] / (4 * x["disc_diameter_frac"] * 2 * R))
    dp_discs = n_discs * k_disc * shield_dp * 0.5 * rho_g * U ** 2
    dp_spray = abs(spray_stats["F_drag"]) / A
    dp = dp_fric + dp_discs + dp_spray
    fan_W = dp * x["air_flow_m3_s"] / FAN_EFF
    elec_W = V_c * I_corona + x["spray_voltage_kV"] * 1e3 * I_spray

    payload = {
        "outlet_total_water_frac": float(total_out / total_in) if total_in > 0 else 1.0,
        "vapor_removed_frac": float((w_in - w_out) / w_in),
        "liquid_escape_frac": float(liq_out / max(mist_liq_in + spray_in, 1e-12)),
        "recovered_L_h": float((total_in - total_out) * 3600),
        "electrical_power_W": float(elec_W),
        "fan_power_W": float(fan_W),
        "pressure_drop_Pa": float(dp),
        "wall_field_kV_cm": float(cs["E_wall"] / 1e5),
        "arc_margin": float(min(arc_margin, 1 - x["spray_voltage_kV"] * 1e3 / V_spark_es,
                                1 - E_sc_wall / E_BREAKDOWN_V_M)),
        "corona_current_mA": float(I_corona * 1e3),
        "spray_current_uA": float(I_spray * 1e6),
        "ceiling_gap_frac": float((vap_max - (vapor_in - mdot_air * w_out)) / vapor_in),
        "water_balance_residual": float(residual),
        "spray_enthalpy_utilization": float(util),  # diagnostic only: ill-conditioned as the ceiling -> 0
        # diagnostics (not targets)
        "spray_regime": regime,
        "spray_drop_d_um": float(2 * a_s * 1e6),
        "spray_charge_frac_rayleigh": float(abs(q_s) / rayleigh_charge(a_s, x["spray_T_C"])),
        "corona_onset_kV": float(cs["V_on"] / 1e3),
        "spray_space_charge_kV_cm": float(E_sc_wall / 1e5),
        "spray_onset_kV": float(V_es_on / 1e3),
        "spray_fate_frac": {k: float(spray_stats[k] / spray_in) if spray_in > 0 else 0.0
                             for k in ("top", "bottom", "wall", "elec")},
        "specific_energy_Wh_L": float((elec_W + fan_W) / max((total_in - total_out) * 3600, 1e-3)),
        "outlet_T_C": float(y_end[0]),
        "vapor_removal_ceiling_frac": float(vap_max / vapor_in) if vapor_in > 0 else 0.0,
        "outlet_RH": float(pv_from_w(w_out) / p_sat(y_end[0])),
        "n_outer_iters": it + 1,
        "spray_entrained_frac": float(sum(c[2] for c in ent) / Q_tot) if Q_tot > 0 else 0.0,
        "mist_collected_frac": float(mist_dep / max(mist_liq_in, 1e-12)),  # can exceed 1: mist grows by condensation
        "profile": {"z": zc.tolist(), "T_g": T_g.tolist(), "w": w_g.tolist(), "rho_q": rho_q_prof.tolist()},
        "corona_inlet": {"classes": inlet_cls, "w": float(yc[1]),
                         "vapor_in": float(vapor_in), "mist_liq_in": mist_liq_in, "spray_in": float(spray_in),
                         "spray_bottom": float(spray_stats["bottom"]), "U": float(U), "pol_c": pol_c,
                         "elec_W_spray": float(x["spray_voltage_kV"] * 1e3 * I_spray), "dp_Pa": float(dp),
                         "fan_W": float(fan_W)},
    }
    return True, payload



def _trace_falling(classes, z_nozzle, L, n_z, dz, A, R, U, T_g, w_g, rho_q, in_corona, cs, pol_c, E_app0, dt_frac,
                   v0=SPRAY_V0_M_S):
    """Time-domain trace of spray classes too large for the upflow to carry,
    against a fixed air profile (PSI-cell; the caller under-relaxes the
    returned sources). Exponential integrators throughout: velocity,
    temperature and surviving fraction each relax with their own time
    constant, and every source uses the exact step average."""
    S_v, S_h, rq = np.zeros(n_z), np.zeros(n_z), np.zeros(n_z)
    kernels = []
    st = dict(top=0.0, bottom=0.0, wall=0.0, elec=0.0, F_drag=0.0, cond=0.0)
    for (a0, q0, Qc) in classes:
        m0 = RHO_L * 4 / 3 * np.pi * a0 ** 3
        Ndot = RHO_L * Qc / m0
        z, v, a, Td, q, f = z_nozzle, -v0, a0, None, q0, 1.0
        k = min(int(z / dz), n_z - 1)
        Td = T_g[k] if Td is None else Td
        dens = np.zeros(n_z)
        steps = 0
        while 0 <= z < L and f > 1e-6 and steps < 20000:
            k = min(int(z / dz), n_z - 1)
            rho_g = gas_density(T_g[k], w_g[k])
            pv = pv_from_w(w_g[k])
            v_rel = U - v
            re = rho_g * abs(v_rel) * 2 * a / MU_G
            tau_p = RHO_L * (2 * a) ** 2 / (18 * MU_G * (1 + 0.15 * re ** 0.687))
            m = RHO_L * 4 / 3 * np.pi * a ** 3
            Teq = quasi_steady_drop_T(np.array([a]), np.array([q]), T_g[k], pv, v_rel, rho_g)[0]
            nu, sh = _transfer_coeffs(a, v_rel, rho_g)
            dpdT = dp_sat_dT(Td) / (R_V * (Td + 273.15))
            tau_T = m * CP_L / (2 * np.pi * a * (nu * K_G + h_fg(Td) * sh * vapor_diffusivity(T_g[k]) * dpdT))
            Zc = cunningham(a) / (6 * np.pi * MU_G * a)
            if in_corona[k]:
                dq_dt = charging_rate(np.array([a]), np.array([q]), cs, pol_c, T_g[k])[0]
                w_drift = q * cs["E_wall"] * Zc * pol_c
                k_loss = 2 * abs(w_drift) / R
                to_elec = w_drift < 0
                tau_q = 4 * EPS0 / max(cs["N_ion"] * E_CHG * cs["mu_ion"], 1e-30)
            else:
                dq_dt, to_elec, tau_q = 0.0, False, np.inf
                k_loss = abs(q) * Zc * (abs(rho_q[k]) / EPS0 + 2 * E_app0 * np.exp(-abs(z - z_nozzle) / R) / R)
            dt = dt_frac * min(dz / max(abs(v), 1e-3), 0.3 * tau_q)
            eT = np.exp(-dt / tau_T)
            T_avg = Teq + (Td - Teq) * tau_T / dt * (1 - eT)
            mdot, qconv = drop_mass_rate(a, q, T_avg, T_g[k], pv, v_rel, rho_g)
            if mdot < 0 and -mdot * dt > 0.2 * m:
                dt = 0.2 * m / -mdot
                eT = np.exp(-dt / tau_T)
            Td_new = Teq + (Td - Teq) * eT
            ek = np.exp(-k_loss * dt)
            f_new = f * ek
            f_avg = f * (1 - ek) / (k_loss * dt) if k_loss * dt > 1e-12 else f
            v_eq = U - G * tau_p
            ep = np.exp(-dt / tau_p)
            v_new = v_eq + (v - v_eq) * ep
            dzp = v_eq * dt + (v - v_eq) * tau_p * (1 - ep)
            m_new = max(m + mdot * dt, 1e-30)
            cond = mdot * Ndot * f_avg * dt
            lost = Ndot * (f * m - f_new * m_new) + cond  # closes the step's mass balance exactly
            st["elec" if to_elec else "wall"] += lost
            S_v[k] -= cond
            S_h[k] -= qconv * Ndot * f_avg * dt
            st["cond"] += cond
            st["F_drag"] += Ndot * f_avg * m * (v_new - v + G * dt)
            dens[k] += Ndot * f_avg * dt
            rq[k] += q * Ndot * f_avg * dt
            a = (m_new / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3)
            q = q + dq_dt * dt
            q = np.sign(q) * min(abs(q), rayleigh_charge(a, T_avg))
            Td, v, f = Td_new, v_new, f_new
            z += dzp
            steps += 1
        m_end = RHO_L * 4 / 3 * np.pi * a ** 3
        if z >= L:
            st["top"] += Ndot * f * m_end
        elif z < 0:
            st["bottom"] += Ndot * f * m_end
        else:
            st["wall"] += Ndot * f * m_end  # step cap reached: count as deposited
        kernels.append((dens / (A * dz), a0, q0, U - terminal_velocity(a0, gas_density(T_g[0], w_g[0]))))
    return dict(S_v=S_v, S_h=S_h, rho_q=rq / (A * dz), kernels=kernels, stats=st)

def _max_vapor_removal(mdot_air, T_in, w_in, m_spray, T_spray):
    """kg/s of vapor condensed when the air and spray reach a common
    saturated equilibrium temperature T_f (energy balance, bisection)."""
    if m_spray <= 0 or T_spray >= T_in:
        return 0.0

    def excess(T_f):
        w_f = min(w_from_pv(p_sat(T_f)), w_in)
        released = mdot_air * (CP_A + w_in * CP_V) * (T_in - T_f) + h_fg(T_f) * mdot_air * (w_in - w_f)
        return released - m_spray * CP_L * (T_f - T_spray)

    lo, hi = T_spray, T_in
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:  # air still releasing more than the spray absorbs: equilibrium is warmer
            lo = mid
        else:
            hi = mid
    T_f = 0.5 * (lo + hi)
    return max(mdot_air * (w_in - w_from_pv(p_sat(T_f))), 0.0)


def _dew_point(w):
    pv = pv_from_w(w)
    lo, hi = -40.0, 100.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if p_sat(mid) > pv:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def worker_fn(conn, params, aux, fid):
    try:
        ok, payload = simulate(params, aux, fid)
        if ok:
            payload.pop("profile", None)
            payload.pop("corona_inlet", None)
        conn.send((ok, payload))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    fid = FIDELITY_PRESETS[fidelity_name]
    return (np.asarray(params, dtype=float), {k: float(aux[k]) for k in AUX_NAMES}, fid)

"""Can real nozzles deliver the spray a watercap design asks for?

A design fixes the total spray flow Q (spray_flow_L_min), the mass-median drop
diameter (spray_d_um, lognormal of mass with spray_gsd, so the Sauter mean is
SMD = MMD exp(-ln^2(gsd)/2)), the launch speed v0 (spray_v0), the cone
half-angle and the number of tiers. A real atomizer couples them: the pressure
sets v0 AND, with the flow per nozzle, the drop size. `nozzle_feasibility`
looks for an atomizer -- technology x nozzle count x discharge coefficient --
that delivers all three at once, inside an engineering envelope, and says why
not if none does. It changes no score: `watercap_resolved.simulate` attaches
its result next to the outcomes (`nozzle`, `feasible`, `feasible_reason`).

Technologies (the correlation's uncertainty is the matching tolerance on SMD):

  pressure_swirl  (simplex hollow cone). Launch speed v0 = Cv sqrt(2 dP / rho),
      Cv 0.62-0.72 measured on pressure-swirl atomizers at 6-14 L/min (Tokgoz
      2019, StoneLab validation/tokgoz2019; the hifi tier's nozzle_cd = 0.67).
      Flow mdot = Cd A0 sqrt(2 rho dP) with Cd 0.20-0.45 on the orifice area
      (Tokgoz's A-1: 0.20-0.23; simplex range in Lefebvre & McDonell 2017, ch. 6).
      SMD from Wang & Lefebvre (1987, J. Propulsion Power 3:11):
          SMD = 4.52 (sigma mu^2 / (rho_A dP^2))^0.25 (t cos th)^0.25
              + 0.39 (sigma rho / (rho_A dP))^0.25 (t cos th)^0.75,
      film t = 3.66 (d0 mdot mu / (rho dP))^0.25 (Suyari & Lefebvre 1986) --
      the combination checked against Tokgoz's PDPA data: within ~25%.
      Envelope: dP 0.5-70 bar (below ~0.5 bar a simplex sheet is still in the
      onion/tulip stage, not a developed hollow cone; 70 bar = high-pressure
      fog pumps), orifice d0 >= 0.15 mm (the smallest commercial fog-nozzle
      orifices, with fine filtration), film t < d0/2.
  plain_orifice  (perforated plate / showerhead jets, low pressure). Jet speed
      v0 = Cv sqrt(2 dP / rho), Cv 0.97; jet diameter d_j = sqrt(Cc) d0, Cc
      0.62-1.0 (sharp-edged to short tube). Breakup regime on the gas Weber
      number We_g = rho_A v0^2 d_j / sigma (Reitz 1978; Lefebvre & McDonell
      ch. 2): Rayleigh (We_g < 0.4, and liquid We_L > 8 so it jets rather than
      drips): d = 1.89 d_j, near monodisperse (Rayleigh 1878; +-10%); first
      wind-induced (0.4-13): d ~ d_j (+-50%); beyond: not covered here.
      Envelope: d0 >= 0.15 mm, hole pitch >= 5 mm across 90% of the section.

Nozzle count per tier: 1 .. N_max, pressure-swirl cones on a >= 5 cm pitch
over 90% of the tube section (ring designs: along the ring). The resolved
model injects each tier as ONE point source (or ring); a feasible answer that
needs many nozzles is flagged `point_source_mismatch` (soft).

Electrostatic charge (spray_q_frac x Rayleigh limit): induction charging of
sprays reaches ~1-10 mC/kg (Law 2001, J. Electrostatics 51-52:25); above
10 mC/kg the design is flagged infeasible.

`feasible` = the as-modeled (Q, SMD, v0) is deliverable. `feasible_relaxed` =
the flow and SMD are deliverable by a pressure-swirl atomizer at SOME pressure
in the envelope (the model's v0 and pump power are then wrong; the required
pressure and pump power are reported). The envelope numbers are engineering
judgement from the sources above and catalog practice, not a fit; they are
parameters (`ENVELOPE`) so a sensitivity check is one call.
"""
import math

import numpy as np

RHO_L = 998.0
RHO_A = 1.15
PUMP_EFF = 0.6          # StoneLab spray_tower.PUMP_EFF
TOL_LW = 1.25           # Wang-Lefebvre vs Tokgoz 2019 PDPA: within ~25%
TOL_RAYLEIGH = 1.10
TOL_WIND1 = 1.5

ENVELOPE = dict(
    dP_min_swirl_Pa=0.5e5, dP_max_Pa=70e5, d0_min_m=0.15e-3, d0_soft_m=0.5e-3,
    pitch_swirl_m=0.05, pitch_plate_m=0.005, cv_swirl=(0.62, 0.67, 0.72), cd_swirl=(0.20, 0.30, 0.45),
    cv_orifice=0.97, cc_orifice=(0.62, 0.8, 1.0), qm_max_C_kg=10e-3,
)


def surface_tension(T_C):     # StoneLab spray_tower.surface_tension
    return 0.0756 - 1.4e-4 * T_C


def mu_liquid(T_C):           # StoneLab spray_tower.mu_liquid (Vogel)
    return 2.414e-5 * 10 ** (247.8 / (T_C + 133.15))


def smd_from_mmd(mmd, gsd):
    return mmd * math.exp(-0.5 * math.log(max(gsd, 1.0)) ** 2)


def lefebvre_wang_smd(mdot, dP, d0, half_angle_deg, T_C=20.0, rho_a=RHO_A):
    """Wang & Lefebvre (1987) SMD [m] of a pressure-swirl atomizer, film thickness after Suyari & Lefebvre."""
    sig, mu = surface_tension(T_C), mu_liquid(T_C)
    t = 3.66 * (d0 * mdot * mu / (RHO_L * dP)) ** 0.25
    tc = t * math.cos(math.radians(half_angle_deg))
    return (4.52 * (sig * mu ** 2 / (rho_a * dP ** 2)) ** 0.25 * tc ** 0.25
            + 0.39 * (sig * RHO_L / (rho_a * dP)) ** 0.25 * tc ** 0.75), t


def _n_max(c, pitch):
    R = c["tube_diameter_m"] / 2
    rr = c.get("nozzle_radial_frac", 0.0) * R * 0.95
    if rr > 0:
        return max(1, int(2 * math.pi * rr / pitch))
    return max(1, int(0.9 * math.pi * R ** 2 / pitch ** 2))


def _n_grid(n_max):
    return sorted(set(int(round(x)) for x in np.geomspace(1, n_max, 60)))


def _swirl(Q_tier, smd_req, half, T_C, n_max, env, dP_fixed=None, prefer="nozzles"):
    """Best pressure-swirl configuration per tier; dP from v0 (dP_fixed: list of (cv, dP)) or free."""
    cands = []
    dPs = dP_fixed if dP_fixed is not None else [(None, float(p)) for p in np.geomspace(env["dP_min_swirl_Pa"], env["dP_max_Pa"], 40)]
    for cv, dP in dPs:
        if not env["dP_min_swirl_Pa"] <= dP <= env["dP_max_Pa"]:
            continue
        for n in _n_grid(n_max):
            m = Q_tier / n
            for cd in env["cd_swirl"]:
                d0 = math.sqrt(4 * m / (math.pi * cd * math.sqrt(2 * RHO_L * dP)))
                if d0 < env["d0_min_m"]:
                    continue
                smd, t = lefebvre_wang_smd(m, dP, d0, half, T_C)
                if t >= d0 / 2:
                    continue
                cands.append(dict(err=abs(math.log(smd / smd_req)), n=n, dP_Pa=dP, d0_m=d0, smd_m=smd, cd=cd, cv=cv))
    if not cands:
        return None
    ok = [q for q in cands if q["err"] <= math.log(TOL_LW)]
    # within tolerance: fewest nozzles (or the lowest pressure: pump power), then the closest size
    key = (lambda q: (q["n"], q["dP_Pa"], q["err"])) if prefer == "nozzles" else (lambda q: (q["dP_Pa"], q["n"], q["err"]))
    return min(ok, key=key) if ok else min(cands, key=lambda q: q["err"])


def _swirl_range(Q_tier, half, T_C, n_max, dP_list, env):
    smds = []
    for _, dP in dP_list:
        if not env["dP_min_swirl_Pa"] <= dP <= env["dP_max_Pa"]:
            continue
        for n in (1, n_max):
            for cd in env["cd_swirl"]:
                m = Q_tier / n
                d0 = math.sqrt(4 * m / (math.pi * cd * math.sqrt(2 * RHO_L * dP)))
                if d0 >= env["d0_min_m"]:
                    smds.append(lefebvre_wang_smd(m, dP, d0, half, T_C)[0])
    return (min(smds), max(smds)) if smds else None


def _orifice(Q_tier, smd_req, v0, T_C, n_max, env):
    sig = surface_tension(T_C)
    out = []
    for cc in env["cc_orifice"]:
        for regime, ratio, tol in (("rayleigh", 1.89, TOL_RAYLEIGH), ("first_wind", 1.0, TOL_WIND1)):
            dj = smd_req / ratio
            weg, wel = RHO_A * v0 ** 2 * dj / sig, RHO_L * v0 ** 2 * dj / sig
            ok_regime = (wel > 8 and weg < 0.4) if regime == "rayleigh" else (0.4 <= weg < 13)
            d0 = dj / math.sqrt(cc)
            n = Q_tier / (RHO_L * v0 * math.pi / 4 * dj ** 2)
            # the size is set by d0, so any hole >= d0_min within tolerance works
            d0_lo = max(d0 / tol, env["d0_min_m"])
            out.append(dict(regime=regime, cc=cc, ok_regime=ok_regime, d0_m=d0, n=n, We_g=weg, We_L=wel,
                            ok=ok_regime and d0_lo <= d0 * tol and n <= n_max and n >= 1, tol=tol))
    ok = [o for o in out if o["ok"]]
    return (min(ok, key=lambda o: o["n"]) if ok else None), out


def nozzle_feasibility(c, aux=None, env=None):
    env = dict(ENVELOPE, **(env or {}))
    Q = c.get("spray_flow_L_min", 0.0) / 60e3 * RHO_L
    if Q <= 0:
        return dict(feasible=True, feasible_relaxed=True, reason="no spray", technology=None)
    T_C = c.get("spray_T_C", 20.0)
    v0 = c["spray_v0"]
    half = c.get("spray_cone_half_angle_deg", 30.0)
    tiers = max(1, int(round(c.get("spray_n_tiers", 1))))
    Qt = Q / tiers
    mmd = c["spray_d_um"] * 1e-6
    smd_req = smd_from_mmd(mmd, c.get("spray_gsd", 1.5))
    nmax_s, nmax_p = _n_max(c, env["pitch_swirl_m"]), _n_max(c, env["pitch_plate_m"])
    r = dict(smd_req_um=smd_req * 1e6, v0=v0, flow_L_min=c["spray_flow_L_min"], tiers=tiers,
             n_max_swirl_per_tier=nmax_s, notes=[])
    reasons = []
    # ---- pressure-swirl at the modeled launch speed
    dP_v = [(cv, 0.5 * RHO_L * (v0 / cv) ** 2) for cv in env["cv_swirl"]]
    r["dP_swirl_bar"] = [round(p / 1e5, 3) for _, p in dP_v]
    sw = _swirl(Qt, smd_req, half, T_C, nmax_s, env, dP_fixed=dP_v)
    rng = _swirl_range(Qt, half, T_C, nmax_s, dP_v, env)
    r["smd_range_swirl_um"] = None if rng is None else [rng[0] * 1e6, rng[1] * 1e6]
    sw_ok = sw is not None and sw["err"] <= math.log(TOL_LW)
    if sw is None:
        reasons.append(f"pressure-swirl: v0 {v0:.1f} m/s needs dP {dP_v[0][1] / 1e5:.2f}-{dP_v[-1][1] / 1e5:.2f} bar "
                       f"(envelope {env['dP_min_swirl_Pa'] / 1e5:.1f}-{env['dP_max_Pa'] / 1e5:.0f} bar)")
    elif not sw_ok:
        reasons.append(f"pressure-swirl at {sw['dP_Pa'] / 1e5:.2f} bar: SMD {rng[0] * 1e6:.0f}-{rng[1] * 1e6:.0f} um "
                       f"with 1-{nmax_s} nozzles/tier, asked {smd_req * 1e6:.0f} um")
    # ---- plain-orifice jets at the modeled launch speed
    po, po_all = _orifice(Qt, smd_req, v0, T_C, nmax_p, env)
    if po is None:
        pr = [o for o in po_all if o["regime"] == "rayleigh"][-1]
        why = ("drips (We_L %.1f < 8)" % pr["We_L"] if pr["We_L"] <= 8 else
               "We_g %.2f outside Rayleigh" % pr["We_g"] if pr["We_g"] >= 0.4 else
               "holes %.3f mm < %.2f mm" % (pr["d0_m"] * 1e3 * TOL_RAYLEIGH, env["d0_min_m"] * 1e3)
               if pr["d0_m"] * TOL_RAYLEIGH < env["d0_min_m"] else "%.0f holes/tier > %d" % (pr["n"], nmax_p))
        reasons.append(f"plain orifice: Rayleigh jets {why}")
    # ---- pick
    if sw_ok:
        r.update(technology="pressure_swirl", n_nozzles_per_tier=sw["n"], dP_bar=sw["dP_Pa"] / 1e5,
                 d0_mm=sw["d0_m"] * 1e3, smd_um=sw["smd_m"] * 1e6, cd=sw["cd"], cv=sw["cv"])
    elif po is not None:
        r.update(technology="plain_orifice", regime=po["regime"], n_nozzles_per_tier=int(math.ceil(po["n"])),
                 dP_bar=0.5 * RHO_L * (v0 / env["cv_orifice"]) ** 2 / 1e5, d0_mm=po["d0_m"] * 1e3,
                 smd_um=smd_req * 1e6, We_g=po["We_g"])
        if po["regime"] == "rayleigh" and c.get("spray_gsd", 1.5) > 1.25:
            r["notes"].append("Rayleigh jets are near monodisperse; modeled gsd %.2f" % c.get("spray_gsd", 1.5))
        if half > 15 and c.get("nozzle_radial_frac", 0.0) == 0:
            r["notes"].append("jet plate spreads little; modeled cone half-angle %.0f deg" % half)
    else:
        r["technology"] = None
    feas = r["technology"] is not None
    if r["technology"] == "pressure_swirl" and not 2e5 <= r["dP_bar"] * 1e5 <= 11e5:
        r["notes"].append("Wang-Lefebvre extrapolated: %.2f bar outside the 2.4-10 bar checked against Tokgoz 2019" % r["dP_bar"])
    if feas:
        if r["d0_mm"] < env["d0_soft_m"] * 1e3:
            r["notes"].append("orifice %.2f mm < %.1f mm: needs fine filtration" % (r["d0_mm"], env["d0_soft_m"] * 1e3))
        if r["n_nozzles_per_tier"] > 1 and c.get("nozzle_radial_frac", 0.0) == 0:
            r["notes"].append("point_source_mismatch: %d nozzles/tier modeled as one point source" % r["n_nozzles_per_tier"])
    # ---- charge
    if c.get("spray_q_frac", 0.0) > 0:
        a = mmd / 2
        qm = c["spray_q_frac"] * 8 * math.pi * math.sqrt(8.854e-12 * surface_tension(T_C) * a ** 3) / (
            RHO_L * 4 / 3 * math.pi * a ** 3)
        r["q_m_mC_kg"] = qm * 1e3
        if qm > env["qm_max_C_kg"]:
            feas = False
            reasons.append(f"charge {qm * 1e3:.1f} mC/kg > induction-charging ceiling {env['qm_max_C_kg'] * 1e3:.0f} mC/kg")
    # ---- relaxed: the flow and SMD at any pressure-swirl pressure (v0, pump power then differ)
    rel = _swirl(Qt, smd_req, half, T_C, nmax_s, env, prefer="pressure")
    rel_ok = rel is not None and rel["err"] <= math.log(TOL_LW) and not (r.get("q_m_mC_kg", 0) > env["qm_max_C_kg"] * 1e3)
    r["feasible_relaxed"] = bool(feas or rel_ok)
    if rel is not None and rel_ok:
        r["relaxed"] = dict(dP_bar=rel["dP_Pa"] / 1e5, n_nozzles_per_tier=rel["n"], d0_mm=rel["d0_m"] * 1e3,
                            v_exit=ENVELOPE["cv_swirl"][1] * math.sqrt(2 * rel["dP_Pa"] / RHO_L),
                            pump_W=Q / RHO_L * rel["dP_Pa"] / PUMP_EFF,
                            extrapolated=not 2e5 <= rel["dP_Pa"] <= 11e5)
    r["pump_W_modeled"] = Q / RHO_L * 0.5 * RHO_L * (v0 / 0.67) ** 2 / PUMP_EFF
    r["feasible"] = bool(feas)
    r["reason"] = "ok" if feas else "; ".join(reasons)
    return r

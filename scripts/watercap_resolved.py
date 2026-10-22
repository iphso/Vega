"""The watercap domain's fully resolved tier ("resolved_coarse" / "resolved"):
both stages resolved, nothing from the 1D ROM.

  spray stage   stonelab.spray_cfd: axisymmetric RANS (mixing length, van
                Driest, buoyancy) two-way coupled to Lagrangian spray
                parcels (momentum, heat, vapor), Eulerian mist bins, CCN
                fog, wall film. Resolves what the ROM's well-mixed
                cross-section cannot: where the spray goes, the air it drags
                and stirs, the air that bypasses it, fog from stratified
                mixing, condensation on the cold wall film.
  corona stage  stonelab.ehd / ehd_particles (as watercap_ehd), fed with the
                spray stage's RADIAL profiles: mist bins and fog per ring,
                and the spray parcels that actually reach the corona section,
                with their positions, sizes and charges.

Outlet water = the spray stage's outlet vapor + whatever liquid escapes the
corona stage. The corona stage changes no vapor (no subcooled liquid there),
and liquid that grows or forms in the corona section (resolved by the spray
stage over the whole tube length) is assigned its class's corona escape
fraction.

Design = watercap_campaign's atomizer candidate dict (spray geometry keys
optional: spray_cone_half_angle_deg, nozzle_radial_frac,
spray_axis_angle_deg, wall_U_W_m2K).

Energy accounting adds what the ROM leaves out: the heat the spray carries
away must be rejected before the water can be sprayed again. A spray at or
above ambient + DRY_COOLER_APPROACH_K is recooled by a dry cooler (fan power
DRY_COOLER_W_PER_W of the heat); below that it needs a chiller at
CHILLER_CARNOT_FRAC of the Carnot COP. `specific_energy_Wh_L` keeps the ROM's
definition (electrical + fan); `specific_energy_total_Wh_L` adds pump and
spray recooling.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

import watercap_oracle as rom

_STONELAB = os.environ.get("STONELAB_SRC", str(Path(__file__).resolve().parents[2] / "StoneLab" / "src"))
if _STONELAB not in sys.path:
    sys.path.insert(0, _STONELAB)

DRY_COOLER_APPROACH_K = 5.0
DRY_COOLER_W_PER_W = 0.02
CHILLER_CARNOT_FRAC = 0.5
CHILLER_LIFT_K = 5.0          # condenser above ambient, evaporator below the spray temperature

RESOLVED_PRESETS = {
    "resolved_coarse": dict(cfd=dict(nr=20, dz=0.015, n_mist_bins=4, n_spray_bins=4, n_per_bin=150, outer_iters=40,
                                     min_outer=12, simple_first=80, simple_per_outer=15, relax_src=0.5, tol=5e-4),
                            ehd="ehd_coarse", n_per_class=400),
    "resolved": dict(cfd=dict(nr=26, dz=0.01, n_mist_bins=6, n_spray_bins=6, n_per_bin=300, outer_iters=80,
                              min_outer=20, simple_first=150, simple_per_outer=25, relax_src=0.5, tol=2e-4),
                     ehd="ehd", n_per_class=1000),
}


def recool_power_W(Q_heat, T_spray, T_amb):
    if Q_heat <= 0:
        return 0.0
    if T_spray >= T_amb + DRY_COOLER_APPROACH_K:
        return DRY_COOLER_W_PER_W * Q_heat
    Tc = T_spray - CHILLER_LIFT_K + 273.15
    Th = T_amb + CHILLER_LIFT_K + 273.15
    cop = CHILLER_CARNOT_FRAC * Tc / max(Th - Tc, 1.0)
    return Q_heat / cop


def build_cfd(c, aux, cfd_kw):
    from stonelab.spray_cfd import CFDConfig, SprayCFD
    from stonelab.spray_tower import Injector, Inlet
    R = c["tube_diameter_m"] / 2
    Ls, Lc = c["spray_section_length_m"], c["corona_section_length_m"]
    U = c["air_flow_m3_s"] / (np.pi * R ** 2)
    inlet = Inlet(U=U, T_C=aux["inlet_T_C"], RH=aux["inlet_RH"], lwc_kg_m3=aux["inlet_lwc_g_m3"] * 1e-3,
                  mist_mmd=aux["mist_mmd_um"] * 1e-6, mist_gsd=aux["mist_gsd"],
                  ccn_per_m3=aux.get("ccn_per_cm3", 300.0) * 1e6)
    pol_c = 1 if c["corona_polarity"] >= 0 else -1
    pol_s = pol_c if c.get("spray_polarity_rel", 1.0) >= 0 else -pol_c
    inj = []
    flow = c["spray_flow_L_min"] / 60 * 1e-3 * rom.RHO_L
    if flow > 0:
        n_tiers = max(1, int(round(c.get("spray_n_tiers", 1))))
        for k in range(n_tiers):
            frac = c["nozzle_height_frac"] * (1 - k / n_tiers) if n_tiers > 1 else c["nozzle_height_frac"]
            z = min(frac * Ls, Ls - 0.02)
            inj.append(Injector(z=z, flow_kg_s=flow / n_tiers, T_C=c["spray_T_C"], d_mmd=c["spray_d_um"] * 1e-6,
                                gsd=c.get("spray_gsd", 1.5), v0=c["spray_v0"],
                                cone_half_angle_deg=c.get("spray_cone_half_angle_deg", 30.0),
                                r_ring=c.get("nozzle_radial_frac", 0.0) * R * 0.95,
                                axis_angle_deg=c.get("spray_axis_angle_deg", 0.0),
                                q_frac_rayleigh=c.get("spray_q_frac", 0.0), polarity=pol_s))
    kw = dict(cfd_kw)
    kw["wall_U"] = c.get("wall_U_W_m2K", 0.0)
    kw["T_amb"] = aux.get("ambient_T_C", 25.0)
    return SprayCFD(R, Ls, Ls + Lc, inlet, inj, CFDConfig(**kw)), pol_c


def corona_parcels(res, c, n_per_class, rng):
    """Inlet parcels for the corona stage from the spray stage's handoff:
    mist bins and fog per ring (positions sampled within each ring by flux),
    plus the spray parcels that reach the top, at their last crossing."""
    from stonelab.ehd_particles import Parcels
    from stonelab.spray_tower import RHO_L
    h = res.handoff
    rf, gas = h["rf"], h["gas"]
    radii, nflow, kinds, cls_mass_top = [], [], [], []
    P = {k: [] for k in ("r", "q", "a", "w", "cls")}

    def ring_parcels(nf_ring, a_ring, k, q=0.0):
        # net upward flux per ring; a ring with reverse flow across Ls hands nothing up
        nf_ring = np.maximum(nf_ring, 0.0)
        tot = nf_ring.sum()
        if tot <= 0:
            return
        p = nf_ring / tot
        idx = rng.choice(len(p), size=n_per_class, p=p)
        u = rng.random(n_per_class)
        r = np.sqrt(rf[idx] ** 2 + u * (rf[idx + 1] ** 2 - rf[idx] ** 2)) * 0.999
        P["r"].append(r)
        P["a"].append(a_ring[idx])
        P["q"].append(np.full(n_per_class, q))
        P["w"].append(np.full(n_per_class, tot / n_per_class))
        P["cls"].append(np.full(n_per_class, k))

    # mist bins
    for b in range(gas["N"].shape[0]):
        N, M = gas["N"][b], gas["M"][b]
        nf_ring = np.maximum(gas["G"] * N, 0.0)
        a_ring = np.maximum((M / np.maximum(N, 1e-300) / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3), 1e-7)
        k = len(radii)
        radii.append(float(np.sum(nf_ring * a_ring) / max(nf_ring.sum(), 1e-300)))
        nflow.append(float(nf_ring.sum()))
        kinds.append("mist")
        cls_mass_top.append(float(h["mist_top_by_bin"][b]) if len(h["mist_top_by_bin"]) > b else 0.0)
        ring_parcels(nf_ring, a_ring, k)
    # fog
    l = gas["l"]
    if np.any(l > 0):
        rho_l = l * h["rho0"] / (1 + 0.0)
        a_f = np.maximum((3 * rho_l / (4 * np.pi * RHO_L * h["ccn"])) ** (1 / 3), 5e-7)
        m_f = RHO_L * 4 / 3 * np.pi * a_f ** 3
        nf_ring = np.where(l > 0, np.maximum(gas["G"], 0.0) * l / m_f, 0.0)
        k = len(radii)
        radii.append(float(np.sum(nf_ring * a_f) / max(nf_ring.sum(), 1e-300)))
        nflow.append(float(nf_ring.sum()))
        kinds.append("fog")
        cls_mass_top.append(float(h["fog_top"]))
        ring_parcels(nf_ring, a_f, k)
    # spray reaching the top (handed over at its last upward crossing of Ls)
    sp = h["spray"]
    top = res.outlet["spray_top_by_cls"]
    if sp is not None and len(sp["r"]):
        for sc in np.unique(sp["cls"]):
            m = sp["cls"] == sc
            k = len(radii)
            radii.append(float(np.mean(sp["a"][m])))
            nflow.append(float(sp["wt"][m].sum()))
            kinds.append("spray")
            cls_mass_top.append(float(top[int(sc)]))
            P["r"].append(np.minimum(sp["r"][m], rf[-1] * 0.999))
            P["a"].append(sp["a"][m])
            P["q"].append(sp["q"][m])
            P["w"].append(sp["wt"][m])
            P["cls"].append(np.full(int(m.sum()), k))
    if not radii:
        return None
    parcels = Parcels(*(np.concatenate(P[k]) for k in ("r", "q", "a", "w", "cls")))
    parcels.cls = parcels.cls.astype(int)
    return dict(parcels=parcels, radii=np.array(radii), nflow=np.array(nflow), kinds=kinds,
                mass_top=np.array(cls_mass_top))


def simulate(c, aux, preset_name="resolved_coarse", seed=0, verbose=False):
    """(ok, payload) for a campaign candidate dict `c` and operating point `aux`."""
    import watercap_ehd as ehd
    from stonelab.ehd_particles import DiscStackSection, march_section
    pre = RESOLVED_PRESETS[preset_name]
    t0 = time.time()
    cfd, pol_c = build_cfd(c, aux, pre["cfd"])
    try:
        res = cfd.solve(verbose=verbose)
    except Exception as e:
        # buoyancy- or spray-driven recirculation can diverge SIMPLE at the default relaxation:
        # one retry with gentler relaxation and more flow iterations per outer step
        kw = dict(pre["cfd"], alpha_u=0.35, alpha_p=0.15, simple_per_outer=2 * pre["cfd"]["simple_per_outer"])
        cfd, pol_c = build_cfd(c, aux, kw)
        try:
            res = cfd.solve(verbose=verbose)
        except Exception as e2:
            return False, f"spray-stage failure: {e} / retry: {e2}"
    if not all(np.isfinite([res.outlet["vapor_out"], res.outlet["T_out"]])):
        return False, "spray-stage non-finite"
    t_cfd = time.time() - t0
    o, d = res.outlet, res.diag
    R = c["tube_diameter_m"] / 2
    U = c["air_flow_m3_s"] / (np.pi * R ** 2)
    pay = dict(spray_stage=dict(vapor_removed_frac=o["vapor_removed_frac"], fog_out_frac=o["fog_out"] / o["total_in"],
                                mist_out_frac=o["mist_out"] / o["total_in"], spray_top_frac=o["spray_top"] / max(o["spray_in"], 1e-12),
                                spray_fates=o["spray_fates"], T_out=o["T_out"], film_drain=o["film_drain"],
                                wall_condensation=o["wall_condensation"], mist_scavenged=o["mist_scavenged"],
                                dp_Pa=o["dp_Pa"], heat_absorbed_W=o["heat_absorbed_W"], pump_W=o["pump_W"]),
               cfd_diag={k: v for k, v in d.items() if k != "history"}, cfd_s=t_cfd)
    # ---- corona stage
    n_periods = max(1, int(round(c["corona_section_length_m"] / c["disc_spacing_m"])))
    rng = np.random.default_rng(seed)
    cp = corona_parcels(res, c, pre["n_per_class"], rng)
    I_c, liq_esc, liq_disc, esc_by_kind = 0.0, 0.0, 0.0, {}
    quench, I_nomist, wall_field = 0.0, 0.0, 0.0
    t1 = time.time()
    if cp is not None:
        ep = ehd.EHD_PRESETS[pre["ehd"]]
        sec = DiscStackSection(R=R, disc_radius=c["disc_diameter_frac"] * R, disc_thickness=c["disc_thickness_mm"] * 1e-3,
                               rel_air_density=293.15 / (aux["inlet_T_C"] + 273.15), spacing=c["disc_spacing_m"],
                               n_periods=n_periods, voltage=pol_c * c["corona_voltage_kV"] * 1e3,
                               mobility=rom.ION_MOBILITY[pol_c], edge_cells=ep["edge_cells"], h_coarse=ep["h_coarse"],
                               growth=ep["growth"])
        Q = U * np.pi * R ** 2
        try:
            sr = march_section(sec, pol_c, U, cp["radii"], cp["nflow"] / Q, n_per_class=pre["n_per_class"],
                               coupling_iters=ep["coupling_iters"], seed=seed, parcels=cp["parcels"])
        except Exception as e:
            return False, f"ehd failure: {e}"
        if not all(p["converged"] for p in sr.periods):
            return False, "ehd corona did not converge in every period"
        esc = sr.escaped_frac
        liq_esc = float(np.sum(esc * cp["mass_top"]))
        liq_disc = float(np.sum(sr.emitter_frac * cp["mass_top"]))
        for kd in set(cp["kinds"]):
            m = np.array([k_ == kd for k_ in cp["kinds"]])
            esc_by_kind[kd] = float(np.sum(esc[m] * cp["mass_top"][m]) / max(np.sum(cp["mass_top"][m]), 1e-30))
        I_c = float(abs(np.sum(sr.current_per_period)))
        I_nomist = float(abs(sr.current_no_mist) * n_periods)
        quench = float(1 - abs(sr.current_per_period[0]) / max(abs(sr.current_no_mist), 1e-30)) if sr.current_no_mist else 0.0
        wall_field = float(abs(np.mean(sr.wall_field_kV_cm)))
    t_ehd = time.time() - t1
    # ---- outlet, energy
    total_in = o["total_in"]
    total_out = o["vapor_out"] + liq_esc
    recovered = (total_in - total_out) * 3600
    # disc-stack pressure loss (same correlation as the ROM), on top of the resolved tube + spray
    beta2 = c["disc_diameter_frac"] ** 2
    cc = 0.61 + 0.39 * (1 - beta2) ** 3
    k_disc = (1 / (cc * (1 - beta2)) - 1) ** 2
    shield = min(1.0, c["disc_spacing_m"] / (4 * c["disc_diameter_frac"] * 2 * R))
    rho = 1.15
    dp_discs = n_periods * k_disc * shield * 0.5 * rho * U ** 2
    dp = max(o["dp_Pa"], 0.0) + dp_discs
    fan_W = dp * c["air_flow_m3_s"] / rom.FAN_EFF
    elec_W = c["corona_voltage_kV"] * 1e3 * I_c
    recool_W = recool_power_W(o["heat_absorbed_W"], c["spray_T_C"], aux.get("ambient_T_C", 25.0)) if o["spray_in"] > 0 else 0.0
    # arc: the ROM's sparkover estimate (uncalibrated) and the resolved spray-cloud field
    V_spark = rom.sparkover_voltage(R - c["disc_diameter_frac"] * R, pol_c)
    arc_margin = min(1 - c["corona_voltage_kV"] * 1e3 / V_spark, d["spray_arc_margin"])
    pay.update({
        "outlet_total_water_frac": float(total_out / total_in),
        "vapor_removed_frac": float(o["vapor_removed_frac"]),
        "liquid_escape_frac": float(liq_esc / max(o["mist_in"] + o["spray_in"], 1e-12)),
        "recovered_L_h": float(recovered),
        "electrical_power_W": float(elec_W),
        "fan_power_W": float(fan_W),
        "pump_power_W": float(o["pump_W"]),
        "recool_power_W": float(recool_W),
        "pressure_drop_Pa": float(dp),
        "wall_field_kV_cm": wall_field,
        "arc_margin": float(arc_margin),
        "corona_current_mA": I_c * 1e3,
        "corona_current_no_mist_mA": I_nomist * 1e3,
        "quenching_frac": quench,
        "disc_fouling_frac": float(liq_disc / max(o["mist_in"] + o["spray_in"], 1e-12)),
        "escape_by_kind": esc_by_kind,
        "specific_energy_Wh_L": float((elec_W + fan_W) / max(recovered, 1e-3)),
        "specific_energy_total_Wh_L": float((elec_W + fan_W + o["pump_W"] + recool_W) / max(recovered, 1e-3)),
        "water_balance_residual": float(d["water_residual"]),
        "energy_residual": float(d["energy_residual"]),
        "momentum_number": float(d["momentum_number"]),
        "ehd_s": t_ehd, "fidelity_tier": preset_name,
    })
    return True, pay

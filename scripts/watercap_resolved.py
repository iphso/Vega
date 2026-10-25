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
# The same tiers with the physics added in the 2026-10 realism spike (StoneLab
# docs/realism_roadmap.md): inertial drops with a Langevin model of the fluid
# velocity they see (disc impaction, turbulent deposition), humidity-dependent
# ion mobility (stonelab.corona_air), and corona-section drop sizes that follow
# the spray stage's condensation growth period by period, and corona-mist coupling
# iterated to a settled current (dense charged spray can quench the corona; three
# fixed iterations can stop on either side of the swing), and the resolved RANS
# flow through the disc cell (wakes, recirculation) in place of the potential flow,
# sampled from its conservative face values (Pollock) so tracers stay well mixed.
# The plain tiers stay
# as they were, so stored results remain reproducible.
PLUS = dict(inertia=True, humid_mobility=True, growth=True, adaptive_coupling=True, resolved_flow=True,
            conservative_flow=True)
# Spray stage must reach a steady state (StoneLab spray_cfd's convergence report): oscillating designs
# (e.g. near-flooding rain curtains) are rejected, and slow drifters get more outer iterations.
STEADY = dict(require_steady_spray=True, outer_iters=100)
# _plus_sst: the same with the k-omega SST cell flow (validated against OpenFOAM kOmegaSST), and
# spray stages whose steady iteration oscillates (near flooding) rerun time-accurately
# (StoneLab SprayCFD.solve_transient: persistent hold-up) and scored on their time averages.
PLUS_SST = dict(PLUS, resolved_flow="sst", resolved_dp=True, require_steady_spray=True, transient_spray=True)
# _hifi: k-omega SST in the spray stage as well (StoneLab spray_sst; matches OpenFOAM kOmegaSST on the
# spray-driven tube flow, where the mixing length misplaces the spray wake -- realism roadmap, spray-stage SST).
# Plus: Abramzon-Sirignano drops with finite liquid conductivity, and the corona coupled with common random
# numbers and a start-up load ramp, charged on the emitted current (realism roadmap, corona-mist coupling).
HIFI = dict(PLUS_SST, spray_turbulence="sst", drop_model="as_conduction", corona_crn="ramp", supply_current=True,
            # ion mobility from hydrate equilibrium + Blanc's law for the air/water-vapour mixture (StoneLab
            # corona_air, valid to saturation at 20-80 C; the Nouri fit of the other tiers saturates above
            # ~30 g/m^3). onset_humidity: StoneLab's FITTED empirical onset-vs-RH factor ("empirical_nouri",
            # blind test on Fu 2022 over-predicts the fall ~2x), kept off here.
            mobility_model="cluster_blanc", onset_humidity=None)
for _k in list(RESOLVED_PRESETS):
    RESOLVED_PRESETS[_k + "_plus"] = dict(RESOLVED_PRESETS[_k], physics=PLUS)
    _cfd = dict(RESOLVED_PRESETS[_k]["cfd"], outer_iters=max(RESOLVED_PRESETS[_k]["cfd"]["outer_iters"],
                                                            STEADY["outer_iters"]))
    RESOLVED_PRESETS[_k + "_plus_sst"] = dict(RESOLVED_PRESETS[_k], cfd=_cfd, physics=PLUS_SST)
    # SST spray stages near flooding converge slowly (cold_spray-0214: a monotone dp drift that a
    # 100-iteration cap reports as "oscillating"; settled by iteration 120-155, the same with 15 or
    # 40 SIMPLE sweeps per outer step), so the cap is 200 (converged designs stop early)
    RESOLVED_PRESETS[_k + "_hifi"] = dict(RESOLVED_PRESETS[_k], cfd=dict(_cfd, outer_iters=max(_cfd["outer_iters"], 200), nozzle_cd=0.67),  # Tokgoz 2019 measured Cd
                                          physics=HIFI)


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


def corona_parcels(res, c, n_per_class, rng, spray_clones=1):
    """Inlet parcels for the corona stage from the spray stage's handoff:
    mist bins and fog per ring (positions sampled within each ring by flux),
    plus the spray parcels that reach the top, at their last crossing.
    `spray_clones` > 1 tracks each handed-over spray parcel that many times
    (weight / clones, same entry state; their turbulent paths differ), for
    designs whose outlet hangs on the escape of a small fraction of a large
    entrained spray."""
    from stonelab.ehd_particles import Parcels
    from stonelab.spray_tower import RHO_L
    h = res.handoff
    rf, gas = h["rf"], h["gas"]
    radii, nflow, kinds, cls_mass_top = [], [], [], []
    P = {k: [] for k in ("r", "q", "a", "w", "cls", "v")}

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
        P["v"].append(np.full((n_per_class, 3), np.nan))     # mist and fog enter moving with the gas

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
            nc = max(int(spray_clones), 1)
            rep = lambda x: np.repeat(x, nc, axis=0)   # noqa: E731
            P["r"].append(rep(np.minimum(sp["r"][m], rf[-1] * 0.999)))
            P["a"].append(rep(sp["a"][m]))
            P["q"].append(rep(sp["q"][m]))
            P["w"].append(rep(sp["wt"][m]) / nc)
            P["cls"].append(np.full(int(m.sum()) * nc, k))
            if "vz" in sp:      # spray enters with its velocity where it crossed Ls (inertial tracking)
                P["v"].append(rep(np.stack([sp["vr"][m], np.zeros(int(m.sum())), sp["vz"][m]], axis=1)))
            else:
                P["v"].append(np.full((int(m.sum()) * nc, 3), np.nan))
    if not radii:
        return None
    parcels = Parcels(*(np.concatenate(P[k]) for k in ("r", "q", "a", "w", "cls")), v=np.concatenate(P["v"]))
    parcels.cls = parcels.cls.astype(int)
    return dict(parcels=parcels, radii=np.array(radii), nflow=np.array(nflow), kinds=kinds,
                mass_top=np.array(cls_mass_top))


def growth_resize(cfd, cp, Ls, spacing):
    """march_section's `resize`: mist and fog parcels follow the spray stage's
    local mean drop size (its Eulerian mist bins and CCN fog, solved over the
    whole tube) from period to period, at the ring they are in. Spray parcels
    keep their size (large drops: growth is small relative to their radius)."""
    from stonelab.spray_tower import RHO_L
    kinds = cp["kinds"]
    fields = []
    for k, kd in enumerate(kinds):
        if kd == "mist" and k < cfd.nb:
            N, M = cfd.N[k], cfd.M[k]
            fields.append(np.maximum((M / np.maximum(N, 1e-300) / (RHO_L * 4 / 3 * np.pi)) ** (1 / 3), 1e-7))
        elif kd == "fog":
            fields.append(np.maximum((3 * cfd.rho0 * cfd.l / (4 * np.pi * RHO_L * cfd.inlet.ccn_per_m3)) ** (1 / 3), 5e-7))
        else:
            fields.append(None)
    jz = lambda z: int(np.clip(np.searchsorted(cfd.zf, z) - 1, 0, cfd.nz - 1))
    state = {"j": jz(Ls - 1e-9)}

    def resize(p, parcels):
        j0, j1 = state["j"], jz(Ls + (p + 0.5) * spacing)
        state["j"] = j1
        a = parcels.a.copy()
        i = np.clip(np.searchsorted(cfd.rf, parcels.r) - 1, 0, cfd.nr - 1)
        for k, F in enumerate(fields):
            m = parcels.cls == k
            if F is None or not m.any():
                continue
            a[m] *= F[i[m], j1] / np.maximum(F[i[m], j0], 1e-300)
        return a
    return resize


def simulate(c, aux, preset_name="resolved_coarse", seed=0, verbose=False, keep=None):
    """(ok, payload) for a campaign candidate dict `c` and operating point `aux`.
    `keep`, if a dict, receives the converged spray-stage solver (`cfd`, `res`)."""
    import watercap_ehd as ehd
    from stonelab.ehd_particles import DiscStackSection, march_section
    pre = RESOLVED_PRESETS[preset_name]
    phys = pre.get("physics", {})
    t0 = time.time()
    cfd_kw = dict(pre["cfd"])
    if phys.get("spray_turbulence"):                      # "sst": k-omega SST in the spray stage too
        cfd_kw["turbulence"] = phys["spray_turbulence"]
    if phys.get("drop_model"):                            # "as_conduction": A-S film + finite liquid conductivity
        cfd_kw["drop_model"] = phys["drop_model"]
    if phys.get("n_per_bin_scale"):                       # more spray parcels (tracer noise)
        cfd_kw["n_per_bin"] = int(round(cfd_kw.get("n_per_bin", 200) * phys["n_per_bin_scale"]))
    cfd, pol_c = build_cfd(c, aux, cfd_kw)
    try:
        res = cfd.solve(verbose=verbose)
    except Exception as e:
        # buoyancy- or spray-driven recirculation can diverge SIMPLE at the default relaxation:
        # one retry with gentler relaxation and more flow iterations per outer step
        kw = dict(cfd_kw, alpha_u=0.35, alpha_p=0.15, simple_per_outer=2 * pre["cfd"]["simple_per_outer"])
        cfd, pol_c = build_cfd(c, aux, kw)
        try:
            res = cfd.solve(verbose=verbose)
        except Exception as e2:
            return False, f"spray-stage failure: {e} / retry: {e2}"
    if not all(np.isfinite([res.outlet["vapor_out"], res.outlet["T_out"]])):
        return False, "spray-stage non-finite"
    for _ in range(phys.get("drift_continue", 2) if phys.get("require_steady_spray") else 0):
        # still creeping one way at the cap (StoneLab's "drifting"): not a cycle, just not converged
        # yet -- iterate on from where it stopped
        if (res.diag.get("convergence") or {}).get("status") != "drifting":
            break
        cfd.cfg.warm_start = False
        res = cfd.solve(verbose=verbose)
    if phys.get("transient_spray") and (res.diag.get("convergence") or {}).get("status") == "oscillating":
        # no steady state: run the spray time-accurately from where the iteration stopped and score
        # its time averages (spray_status "unsteady_averaged")
        res = cfd.solve_transient(verbose=verbose)
        if not all(np.isfinite([res.outlet["vapor_out"], res.outlet["T_out"]])):
            return False, "spray-stage non-finite (transient)"
    if keep is not None:
        keep.update(cfd=cfd, res=res)
    t_cfd = time.time() - t0
    o, d = res.outlet, res.diag
    conv = d.get("convergence") or {}
    spray_status = conv.get("status", "converged" if d.get("converged") else "unknown")
    if phys.get("require_steady_spray") and spray_status == "oscillating":
        return False, "spray stage has no steady state (oscillating)"
    R = c["tube_diameter_m"] / 2
    U = c["air_flow_m3_s"] / (np.pi * R ** 2)
    pay = dict(spray_stage=dict(vapor_removed_frac=o["vapor_removed_frac"], fog_out_frac=o["fog_out"] / o["total_in"],
                                mist_out_frac=o["mist_out"] / o["total_in"], spray_top_frac=o["spray_top"] / max(o["spray_in"], 1e-12),
                                spray_fates=o["spray_fates"], T_out=o["T_out"], film_drain=o["film_drain"],
                                wall_condensation=o["wall_condensation"], mist_scavenged=o["mist_scavenged"],
                                dp_Pa=o["dp_Pa"], heat_absorbed_W=o["heat_absorbed_W"], pump_W=o["pump_W"]),
               cfd_diag={k: v for k, v in d.items() if k not in ("history", "transient")}, cfd_s=t_cfd)
    # ---- corona stage
    n_periods = max(1, int(round(c["corona_section_length_m"] / c["disc_spacing_m"])))
    rng = np.random.default_rng(seed)
    cp = corona_parcels(res, c, pre["n_per_class"], rng, spray_clones=phys.get("spray_clones", 1))
    I_c, liq_esc, liq_disc, liq_back, esc_by_kind = 0.0, 0.0, 0.0, 0.0, {}
    quench, I_nomist, wall_field = 0.0, 0.0, 0.0
    sr, quenched_periods, front = None, 0, None
    t1 = time.time()
    if cp is not None:
        ep = ehd.EHD_PRESETS[pre["ehd"]]
        mobility = rom.ION_MOBILITY[pol_c]
        if phys.get("humid_mobility"):
            from stonelab import corona_air
            hg = res.handoff["gas"]
            T_h, w_h = float(np.mean(hg["T"])), float(np.mean(hg["w"]))
            e = w_h * 101325.0 / (0.622 + w_h)
            mobility = corona_air.ion_mobility(pol_c, T_h, ah=e * 0.018015 / (8.314 * (T_h + 273.15)) * 1e3,
                                               mobility_model=phys.get("mobility_model", "nouri_fit"))
        mobility *= phys.get("mobility_scale", 1.0)          # uncertainty studies
        onset_scale = phys.get("onset_scale", 1.0)
        if phys.get("onset_humidity") not in (None, "none"):
            # StoneLab corona_onset.onset_humidity_factor: FITTED (Nouri 2011) onset-vs-RH factor, handed-over gas
            from stonelab import corona_air, corona_onset
            hg = res.handoff["gas"]
            T_h, w_h = float(np.mean(hg["T"])), float(np.mean(hg["w"]))
            e = w_h * 101325.0 / (0.622 + w_h)
            onset_scale *= corona_onset.onset_humidity_factor(
                min(100.0, 100.0 * e / corona_air.saturation_vapour_pressure(T_h)), phys["onset_humidity"])
        onset_kw = {}
        if phys.get("onset_model", "peek") != "peek":
            # stonelab.corona_onset: ionization-integral onset on the disc edge's own field, in the
            # handed-over moist gas (opt-in; the existing tiers keep Peek)
            hg = res.handoff["gas"]
            T_h, w_h = float(np.mean(hg["T"])), float(np.mean(hg["w"]))
            e = w_h * 101325.0 / (0.622 + w_h)
            onset_kw = dict(onset_model=phys["onset_model"],
                            abs_humidity=e * 0.018015 / (8.314 * (T_h + 273.15)) * 1e3)
        sec = DiscStackSection(R=R, disc_radius=c["disc_diameter_frac"] * R, disc_thickness=c["disc_thickness_mm"] * 1e-3,
                               rel_air_density=293.15 / (aux["inlet_T_C"] + 273.15), spacing=c["disc_spacing_m"],
                               n_periods=n_periods, voltage=pol_c * c["corona_voltage_kV"] * 1e3,
                               mobility=mobility, edge_cells=ep["edge_cells"], h_coarse=ep["h_coarse"],
                               growth=ep["growth"], onset_scale=onset_scale,
                               **onset_kw)
        Q = U * np.pi * R ** 2
        try:
            sr = march_section(sec, pol_c, U, cp["radii"], cp["nflow"] / Q, n_per_class=pre["n_per_class"],
                               coupling_iters=ep["coupling_iters"], seed=seed, parcels=cp["parcels"], verbose=verbose,
                               inertia=phys.get("inertia") if isinstance(phys.get("inertia"), dict) else bool(phys.get("inertia")),
                               resize=growth_resize(cfd, cp, c["spray_section_length_m"], c["disc_spacing_m"])
                               if phys.get("growth") else None,
                               adaptive_iters=16 if phys.get("adaptive_coupling") else 0,
                               cell_flow=("resolved_sst" if phys.get("resolved_flow") == "sst" else
                                          "resolved" if phys.get("resolved_flow") else "potential"),
                               conservative_flow=bool(phys.get("conservative_flow")),
                               crn=bool(phys.get("corona_crn")),
                               crn_coupling=phys["corona_crn"] if isinstance(phys.get("corona_crn"), str) else "ramp")
        except Exception as e:
            return False, f"ehd failure: {e}"
        if not all(p["converged"] for p in sr.periods):
            return False, "ehd corona did not converge in every period"
        esc = sr.escaped_frac
        liq_esc = float(np.sum(esc * cp["mass_top"]))
        liq_disc = float(np.sum(sr.emitter_frac * cp["mass_top"]))
        liq_back = float(np.sum(sr.fell_back_frac * cp["mass_top"])) if sr.fell_back_frac is not None else 0.0
        for kd in set(cp["kinds"]):
            m = np.array([k_ == kd for k_ in cp["kinds"]])
            esc_by_kind[kd] = float(np.sum(esc[m] * cp["mass_top"][m]) / max(np.sum(cp["mass_top"][m]), 1e-30))
        I_c = float(abs(np.sum(sr.current_per_period)))
        if phys.get("supply_current"):
            # the supply delivers the emitted current: ions reaching the wall AND ions charging the
            # drops (which carry that current on to the wall, the discs or out). The wall ion current
            # alone undercounts it where a dense mist takes up most of the ions.
            I_c = float(abs(np.sum([q.get("emitted", q["current"]) for q in sr.periods])))
        I_nomist = float(abs(sr.current_no_mist) * n_periods)
        quench = float(1 - abs(sr.current_per_period[0]) / max(abs(sr.current_no_mist), 1e-30)) if sr.current_no_mist else 0.0
        wall_field = float(abs(np.mean(sr.wall_field_kV_cm)))
        # quench front: dense entrained spray can hold the corona near extinction over the first
        # periods until enough of it is collected; where (if) it recovers sets the outlet of such
        # designs, and a front near the top of the stack means a small change in load flips them
        I_rel = np.abs(np.array([q.get("emitted", q["current"]) for q in sr.periods])) / max(
            abs(getattr(sr.no_mist, "current_emitted", sr.current_no_mist) or 0.0), 1e-30)
        quenched_periods = int(np.sum(I_rel < 0.1))
        rec = np.flatnonzero(I_rel >= 0.5)
        front = int(rec[0]) if len(rec) else n_periods
        if keep is not None:
            keep.update(sec=sr, cp=cp)
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
    dp_discs = dp_discs_corr = n_periods * k_disc * shield * 0.5 * rho * U ** 2
    cell_G = getattr(sr, "cell_G", None) if sr is not None else None
    if phys.get("resolved_dp") and cell_G is not None:
        # the resolved cell flow's own drag: its mean gradient less the bare tube's wall friction
        # (Blasius), which the tube model already counts
        Re = rho * U * 2 * R / 1.85e-5
        G_wall = 0.3164 * Re ** -0.25 / (2 * R) * 0.5 * rho * U ** 2
        dp_discs = n_periods * c["disc_spacing_m"] * max(cell_G - G_wall, 0.0)
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
        "disc_dp_Pa": float(dp_discs), "disc_dp_correlation_Pa": float(dp_discs_corr),
        "cell_G_Pa_m": None if cell_G is None else float(cell_G),
        "wall_field_kV_cm": wall_field,
        "arc_margin": float(arc_margin),
        "corona_current_mA": I_c * 1e3,
        "corona_current_no_mist_mA": I_nomist * 1e3,
        "quenching_frac": quench,
        "corona_quenched_periods": quenched_periods if sr is not None else 0,
        "corona_recovery_period": front if sr is not None else None,
        "corona_quench_margin_periods": (n_periods - front) if sr is not None else None,
        "disc_fouling_frac": float(liq_disc / max(o["mist_in"] + o["spray_in"], 1e-12)),
        # inertial drops (the _plus tiers) that fall back out of the corona section into the spray stage
        "corona_fell_back_frac": float(liq_back / max(o["mist_in"] + o["spray_in"], 1e-12)),
        "escape_by_kind": esc_by_kind,
        "specific_energy_Wh_L": float((elec_W + fan_W) / max(recovered, 1e-3)),
        "specific_energy_total_Wh_L": float((elec_W + fan_W + o["pump_W"] + recool_W) / max(recovered, 1e-3)),
        "water_balance_residual": float(d["water_residual"]),
        "energy_residual": float(d["energy_residual"]),
        "momentum_number": float(d["momentum_number"]),
        "ehd_s": t_ehd, "fidelity_tier": preset_name, "physics": dict(phys), "spray_status": spray_status,
        "spray_flooding": {k: v for k, v in (d.get("flooding") or {}).items() if k != "by_class"},
        "spray_unsteady": None if "transient" not in d else dict(
            # batch-means standard errors of the spray stage's own time averages (kg/s, Pa; the spray stage's
            # outlet fraction counts the spray carried up into the corona section)
            se={("spray_stage_outlet_frac" if k == "outlet_total_water_frac" else k): d["transient"]["se"].get(k)
                for k in ("outlet_total_water_frac", "vapor_out", "vapor_removed_frac", "dp_Pa", "spray_top")},
            **{k: d["transient"][k] for k in ("dp_std_rel", "w_out_std_rel", "dp_period_s", "t_avg", "wall_s")}),
    })
    # can real nozzles deliver this spray (flow, drop size, launch speed)? A flag next to the scores, not in them.
    try:
        import watercap_nozzles
        nz = watercap_nozzles.nozzle_feasibility(c, aux)
        pay.update(nozzle=nz, feasible=nz["feasible"], feasible_relaxed=nz["feasible_relaxed"], feasible_reason=nz["reason"])
    except ImportError:          # browser bundles built before the module existed
        pay.update(nozzle=None, feasible=None, feasible_relaxed=None, feasible_reason="watercap_nozzles unavailable")
    return True, pay

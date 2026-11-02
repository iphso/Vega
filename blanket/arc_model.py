"""Parametric ARC FLiBe immersion-blanket model in OpenMC (reference tier).

Geometry follows Sorbom et al. 2015 (arXiv 1409.3540), the MCNP model of
Fig. 27: an axisymmetric D-shaped double-walled vacuum vessel (W first wall |
Inconel 718 inner VV | FLiBe channel | Be multiplier | Inconel 718 outer VV)
immersed in a rectangular ("squared") FLiBe tank, surrounded by a TiH2 shield
and a coil-structure box. Inboard radial build from Fig. 2; tank and shield
extents elsewhere read off Fig. 27 (~2 cm/px, approximate).

Assumptions not in the paper, each a design knob:
  - first-wall shape: Miller D, R0=333 cm, a=113 cm (inboard face at 220 cm,
    outboard at 446 cm per Fig. 2 + Table 1), kappa=1.84, delta=0.1 (the paper
    gives no triangularity; Fig. 27 is nearly elliptical)
  - divertor (8 cm W over 17% of the lower VV) and the 18 support columns are
    omitted
  - plasma source: DT 14.1 MeV rings weighted by n^2<sigma v> on a parabolic
    profile (T0=27 keV, n0=1.8e20, Table 1), isotropic

Run: docker compose run --rm openmc blanket/arc_model.py [--particles N] [--batches B]
"""
import argparse, json, os
import numpy as np
import openmc

HERE = os.path.dirname(os.path.abspath(__file__))

# ARC Table 1
FUSION_POWER_MW = 525.0
NEUTRONS_PER_S = FUSION_POWER_MW * 1e6 / (17.59 * 1.602176634e-13)

DEFAULTS = dict(
    R0=333.0, a=113.0, kappa=1.84, delta=0.10,
    fw_material="W", fw_cm=1.0,          # first wall (PFC tiles)
    vv_in_cm=1.0, channel_cm=2.0, be_cm=1.0, vv_out_cm=3.0,
    li6_pct=None,                        # None: the breeder's default (BREEDERS)
    breeder="flibe",                     # blanket + channel fill (BREEDERS)
    blanket_in_cm=20.0, blanket_out_cm=90.0, blanket_vert_cm=70.0,
    tank_cm=3.0,
    shield_in_cm=51.0, shield_out_cm=24.0, shield_vert_cm=28.0,
    coil_cm=64.0,
    layout="box",                        # "box" (ARC's squared tank) or "conformal"
    conf_blanket_cm=70.0, conf_shield_cm=35.0,   # conformal layout: uniform offsets
    ports_n=0, ports_w_cm=0.0, ports_h_cm=0.0,   # equatorial ports (box layout): void through
                                                 # fw .. tank wall, outboard, centred on phi = k 2pi/n
)


# Breeder concepts: fill of the bulk blanket and of the VV cooling channel, and the
# Li-6 enrichment each is usually quoted at. Densities at operating temperature.
#   flibe   2LiF-BeF2, 1.94 g/cc (~900 K): ARC
#   flinak  LiF-NaF-KF 46.5-11.5-42 mol%, 2.02 g/cc (~900 K; Sohal 2010)
#   pbli    Pb-15.7Li eutectic (at%), 9.4 g/cc (~700 K; Martelli 2019), DCLL/WCLL: 90% Li-6;
#           the channel carries PbLi too (self-cooled first-wall channel, DCLL-like)
#   li      liquid lithium, 0.48 g/cc (~800 K), natural enrichment, self-cooled
#   hcpb    helium-cooled pebble bed, homogenised breeding zone, solid volume fractions:
#           Li4SiO4 15% (2.40 g/cc), Be 40% (1.85), EUROFER 10%, 8 MPa He the rest
#           (~63%-packed Li4SiO4 and Be beds plus cooling plates, EU DEMO HCPB-like);
#           60% Li-6; channel = 8 MPa He (0.0055 g/cc, ~void)
BREEDERS = {"flibe": 90.0, "flinak": 90.0, "pbli": 90.0, "li": 7.5, "hcpb": 60.0}


def breeder_material(name, li6):
    enr = dict(enrichment_target="Li6", enrichment_type="ao", enrichment=li6)
    if name == "flibe":
        m = openmc.Material(name="flibe"); m.add_elements_from_formula("F4Li2Be", **enr); m.set_density("g/cm3", 1.94)
    elif name == "flinak":
        m = openmc.Material(name="flinak")
        m.add_element("Li", 46.5, enrichment_target="Li6", enrichment_type="ao", enrichment=li6)
        m.add_element("Na", 11.5); m.add_element("K", 42.0); m.add_element("F", 100.0)
        m.set_density("g/cm3", 2.02)
    elif name == "pbli":
        m = openmc.Material(name="pbli")
        m.add_element("Pb", 84.3); m.add_element("Li", 15.7, enrichment_target="Li6", enrichment_type="ao", enrichment=li6)
        m.set_density("g/cm3", 9.4)
    elif name == "li":
        m = openmc.Material(name="lithium"); m.add_element("Li", 1.0, enrichment_target="Li6", enrichment_type="ao", enrichment=li6)
        m.set_density("g/cm3", 0.48)
    elif name == "hcpb":
        lis = openmc.Material(name="li4sio4"); lis.add_elements_from_formula("Li4SiO4", **enr); lis.set_density("g/cm3", 2.40)
        be = openmc.Material(name="be_peb"); be.add_element("Be", 1.0); be.set_density("g/cm3", 1.85)
        eu = eurofer()
        he = helium()
        m = openmc.Material.mix_materials([lis, be, eu, he], [0.15, 0.40, 0.10, 0.35], "vo", name="hcpb")
    else:
        raise ValueError(name)
    return m


def eurofer():
    eu = openmc.Material(name="eurofer97")
    for el, wt in [("Fe", 88.9), ("Cr", 9.0), ("W", 1.1), ("Mn", 0.4), ("V", 0.2), ("Ta", 0.12), ("C", 0.11)]:
        eu.add_element(el, wt, "wo")
    eu.set_density("g/cm3", 7.75)
    return eu


def helium():
    he = openmc.Material(name="helium"); he.add_element("He", 1.0); he.set_density("g/cm3", 0.0055)
    return he


def li6_of(p):
    return p["li6_pct"] if p.get("li6_pct") is not None else BREEDERS[p.get("breeder", "flibe")]


def materials(p):
    m = {}
    W = openmc.Material(name="tungsten"); W.add_element("W", 1.0); W.set_density("g/cm3", 19.3)
    inc = openmc.Material(name="inconel718")
    for el, wt in [("Ni", 52.5), ("Cr", 19.0), ("Fe", 18.5), ("Nb", 5.1), ("Mo", 3.05),
                   ("Ti", 0.9), ("Al", 0.5), ("Co", 0.47), ("C", 0.04), ("Mn", 0.18),
                   ("Si", 0.18)]:
        inc.add_element(el, wt, "wo")
    inc.set_density("g/cm3", 8.19)
    flibe = breeder_material(p["breeder"], li6_of(p))
    chan = helium() if p["breeder"] == "hcpb" else flibe
    be = openmc.Material(name="beryllium"); be.add_element("Be", 1.0); be.set_density("g/cm3", 1.85)
    tih2 = openmc.Material(name="TiH2"); tih2.add_elements_from_formula("TiH2"); tih2.set_density("g/cm3", 3.75)
    # coil pack: REBCO + Cu + steel case smeared; steel-dominated (only the
    # fluence reaching it matters here, not its own breeding)
    coil = openmc.Material(name="coil")
    for el, wt in [("Fe", 50.0), ("Cr", 12.0), ("Ni", 8.0), ("Cu", 25.0), ("Mn", 1.5), ("Mo", 1.5), ("Y", 0.5), ("Ba", 1.0), ("Gd", 0.5)]:
        coil.add_element(el, wt, "wo")
    coil.set_density("g/cm3", 7.9)
    cfc = openmc.Material(name="cfc"); cfc.add_element("C", 1.0); cfc.set_density("g/cm3", 1.9)
    if chan is not flibe:
        m["chan"] = chan
    m.update(W=W, inconel=inc, flibe=flibe, Be=be, TiH2=tih2, coil=coil, CFC=cfc)
    return m


def miller(R0, a, kappa, delta, n=96):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([R0 + a * np.cos(t + np.arcsin(delta) * np.sin(t)), kappa * a * np.sin(t)])


def rz_box(rmin, rmax, zmin, zmax):
    return (+openmc.ZCylinder(r=rmin) & -openmc.ZCylinder(r=rmax)
            & +openmc.ZPlane(zmin) & -openmc.ZPlane(zmax))


def build(p=None, particles=20000, batches=50, photons=True):
    p = {**DEFAULTS, **(p or {})}
    mats = materials(p)
    fw_mat = {"W": mats["W"], "Be": mats["Be"], "CFC": mats["CFC"], "Inconel": mats["inconel"]}[p["fw_material"]]

    # VV layers as successive offsets of the plasma-facing polygon
    fw_poly = openmc.model.Polygon(miller(p["R0"], p["a"], p["kappa"], p["delta"]), basis="rz")
    layers = [("fw", p["fw_cm"], fw_mat), ("vv_in", p["vv_in_cm"], mats["inconel"]),
              ("channel", p["channel_cm"], mats.get("chan", mats["flibe"])), ("be", p["be_cm"], mats["Be"]),
              ("vv_out", p["vv_out_cm"], mats["inconel"])]
    polys = [fw_poly]
    for _, t, _ in layers:
        polys.append(polys[-1].offset(t))
    cells = {"plasma": openmc.Cell(name="plasma", region=-polys[0])}
    for i, (name, t, mat) in enumerate(layers):
        if t > 0:
            cells[name] = openmc.Cell(name=name, fill=mat, region=+polys[i] & -polys[i + 1])
    vv_outer = polys[-1]
    if p["layout"] == "conformal":
        return _conformal_rest(p, mats, cells, polys, particles, batches, photons)

    # squared tank: extents from the VV's bounding box
    pts = miller(p["R0"], p["a"], p["kappa"], p["delta"], 720)
    vv_th = sum(t for _, t, _ in layers)
    r_in = pts[:, 0].min() - vv_th
    r_out = pts[:, 0].max() + vv_th
    z_top = pts[:, 1].max() + vv_th
    b_rmin, b_rmax = r_in - p["blanket_in_cm"], r_out + p["blanket_out_cm"]
    b_zmax = z_top + p["blanket_vert_cm"]
    blanket_box = rz_box(b_rmin, b_rmax, -b_zmax, b_zmax)
    cells["blanket"] = openmc.Cell(name="blanket", fill=mats["flibe"], region=blanket_box & +vv_outer)
    tk = p["tank_cm"]
    tank_box = rz_box(b_rmin - tk, b_rmax + tk, -b_zmax - tk, b_zmax + tk)
    cells["tank"] = openmc.Cell(name="tank", fill=mats["inconel"], region=tank_box & ~blanket_box)
    sh_box = rz_box(b_rmin - tk - p["shield_in_cm"], b_rmax + tk + p["shield_out_cm"],
                    -b_zmax - tk - p["shield_vert_cm"], b_zmax + tk + p["shield_vert_cm"])
    cells["shield"] = openmc.Cell(name="shield", fill=mats["TiH2"], region=sh_box & ~tank_box)
    c = p["coil_cm"]
    coil_rmin = max(1.0, b_rmin - tk - p["shield_in_cm"] - c)
    coil_box = rz_box(coil_rmin, b_rmax + tk + p["shield_out_cm"] + c,
                      -b_zmax - tk - p["shield_vert_cm"] - c, b_zmax + tk + p["shield_vert_cm"] + c)
    cells["coil"] = openmc.Cell(name="coil", fill=mats["coil"], region=coil_box & ~sh_box)
    if p["ports_n"] > 0:
        _carve_ports(p, cells, tank_box, polys[0])
    bound = openmc.Sphere(r=1500, boundary_type="vacuum")
    cells["void"] = openmc.Cell(name="void", region=-bound & ~coil_box)
    # the plasma polygon is inside every box; exclude VV interior from box cells
    geom = openmc.Geometry(list(cells.values()))

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches = particles, batches
    settings.photon_transport = photons
    settings.source = plasma_source(p)
    settings.output = {"tallies": False}

    tallies = openmc.Tallies()
    breed = [cells[k] for k in ("channel", "blanket")]
    t = openmc.Tally(name="TBR"); t.filters = [openmc.CellFilter(breed)]; t.scores = ["(n,Xt)"]; tallies.append(t)
    named = [cells[k] for k in ("fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield", "coil") if k in cells]
    h = openmc.Tally(name="heating"); h.filters = [openmc.CellFilter(named)]
    h.scores = ["heating"] if photons else ["heating-local"]; tallies.append(h)
    # fast fluence (>0.1 MeV) into the coil pack: ARC's magnet lifetime criterion
    f = openmc.Tally(name="coil_fast_flux")
    f.filters = [openmc.CellFilter([cells["coil"]]), openmc.EnergyFilter([0.1e6, 20e6])]
    f.scores = ["flux"]; tallies.append(f)

    model = openmc.Model(geom, openmc.Materials(mats.values()), settings, tallies)
    return model, cells, p


def _carve_ports(p, cells, tank_box, fw_poly):
    """Equatorial ports as StoneLab gpu.ts PortSpec: wedges of toroidal width ports_w_cm at the
    outboard first wall (R0 + a), |Z| < ports_h_cm / 2, R > R0, cut out of fw .. tank wall."""
    n = int(p["ports_n"])
    psi = min(np.pi / n, p["ports_w_cm"] / 2 / (p["R0"] + p["a"]))
    zlo, zhi = openmc.ZPlane(-p["ports_h_cm"] / 2), openmc.ZPlane(p["ports_h_cm"] / 2)
    out = +openmc.ZCylinder(r=p["R0"]) & +zlo & -zhi
    port = None
    for k in range(n):
        c = 2 * np.pi * k / n
        a1, a2 = c - psi, c + psi
        w = +openmc.Plane(a=-np.sin(a1), b=np.cos(a1), c=0, d=0) & -openmc.Plane(a=-np.sin(a2), b=np.cos(a2), c=0, d=0)
        port = w if port is None else (port | w)
    port = port & out
    for name in ("fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank"):
        if name in cells:
            cells[name].region = cells[name].region & ~port
    cells["port"] = openmc.Cell(name="port", region=port & tank_box & +fw_poly)


def _conformal_rest(p, mats, cells, polys, particles, batches, photons):
    """Blanket, tank wall, shield and coil as uniform offsets of the VV's outer polygon:
    the axisymmetric case of StoneLab's conformal 3-D geometry (neutron/conformal.ts),
    whose GPU path this checks."""
    vv_outer = polys[-1]
    outs = [("blanket", p["conf_blanket_cm"], mats["flibe"]), ("tank", p["tank_cm"], mats["inconel"]),
            ("shield", p["conf_shield_cm"], mats["TiH2"]), ("coil", p["coil_cm"], mats["coil"])]
    prev = vv_outer
    for name, t, mat in outs:
        nxt = prev.offset(t)
        cells[name] = openmc.Cell(name=name, fill=mat, region=+prev & -nxt)
        prev = nxt
    bound = openmc.Sphere(r=1500, boundary_type="vacuum")
    cells["void"] = openmc.Cell(name="void", region=-bound & +prev)
    geom = openmc.Geometry(list(cells.values()))
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches = particles, batches
    settings.photon_transport = photons
    settings.source = plasma_source(p)
    settings.output = {"tallies": False}
    tallies = openmc.Tallies()
    breed = [cells[k] for k in ("channel", "blanket")]
    t = openmc.Tally(name="TBR"); t.filters = [openmc.CellFilter(breed)]; t.scores = ["(n,Xt)"]; tallies.append(t)
    named = [cells[k] for k in ("fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield", "coil") if k in cells]
    h = openmc.Tally(name="heating"); h.filters = [openmc.CellFilter(named)]
    h.scores = ["heating"] if photons else ["heating-local"]; tallies.append(h)
    # fast flux in the whole coil layer (the GPU tallies only its plasma-facing 5 cm)
    f = openmc.Tally(name="coil_fast_flux")
    f.filters = [openmc.CellFilter([cells["coil"]]), openmc.EnergyFilter([0.1e6, 20e6])]
    f.scores = ["flux"]; tallies.append(f)
    model = openmc.Model(geom, openmc.Materials(mats.values()), settings, tallies)
    return model, cells, p


def plasma_source(p, n_rho=12, n_theta=24):
    """DT rings over the plasma cross-section weighted by n^2 <sigma v>(T)."""
    T0, n0, aT, an = 27.0, 1.0, 1.5, 0.5   # keV; parabolic exponents, flat-ish density
    a_pl = p["a"] - 3.0                     # plasma edge sits 3 cm inside the FW (SOL)
    sources = []
    rho = (np.arange(n_rho) + 0.5) / n_rho
    th = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi
    for r in rho:
        T = T0 * (1 - r ** 2) ** aT
        n = n0 * (1 - r ** 2) ** an
        sv = bosch_hale_dt(max(T, 0.2))
        for t in th:
            R = p["R0"] + r * a_pl * np.cos(t + np.arcsin(p["delta"]) * np.sin(t))
            Z = p["kappa"] * r * a_pl * np.sin(t)
            # area element ~ r dr dtheta (x kappa, const), volume ~ 2 pi R
            w = n * n * sv * r * R
            src = openmc.IndependentSource()
            src.space = openmc.stats.CylindricalIndependent(
                openmc.stats.Discrete([R], [1.0]), openmc.stats.Uniform(0, 2 * np.pi),
                openmc.stats.Discrete([Z], [1.0]))
            src.angle = openmc.stats.Isotropic()
            # thermal Doppler width (~0.1-0.2 MeV here) is irrelevant to breeding
            src.energy = openmc.stats.Discrete([14.08e6], [1.0])
            src.strength = w
            sources.append(src)
    # OpenMC carries source strength into particle weight: normalise so
    # tallies stay per source neutron
    tot = sum(s.strength for s in sources)
    for s in sources:
        s.strength /= tot
    return sources


def bosch_hale_dt(T):
    """<sigma v> for D-T in m^3/s, Bosch & Hale 1992, T in keV."""
    BG, mrc2 = 34.3827, 1124656.0
    C1, C2, C3, C4, C5, C6, C7 = 1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35e-2, -1.0675e-4, 1.366e-5
    theta = T / (1 - T * (C2 + T * (C4 + T * C6)) / (1 + T * (C3 + T * (C5 + T * C7))))
    xi = (BG ** 2 / (4 * theta)) ** (1 / 3)
    return C1 * theta * np.sqrt(xi / (mrc2 * T ** 3)) * np.exp(-3 * xi) * 1e-6


def run(p=None, particles=20000, batches=50, photons=True, tag="arc_baseline", threads=None):
    model, cells, p = build(p, particles, batches, photons)
    d = os.path.join(HERE, "runs", tag)
    os.makedirs(d, exist_ok=True)
    sp_path = model.run(cwd=d, threads=threads or int(os.environ.get("OMP_NUM_THREADS", "24")))
    with openmc.StatePoint(sp_path) as sp:
        tbr = sp.get_tally(name="TBR").get_pandas_dataframe()
        heat = sp.get_tally(name="heating").get_pandas_dataframe()
        ff = sp.get_tally(name="coil_fast_flux").get_pandas_dataframe()
    id2name = {c.id: n for n, c in cells.items()}
    eV_to_MW = NEUTRONS_PER_S * 1.602176634e-19 / 1e6
    out = {
        "params": p,
        "TBR": {"mean": float(tbr["mean"].sum()), "std": float(np.sqrt((tbr["std. dev."] ** 2).sum())),
                "by_cell": {id2name[int(r.cell)]: float(r["mean"]) for _, r in tbr.iterrows()}},
        "heating_MW": {id2name[int(r.cell)]: {"mean": float(r["mean"]) * eV_to_MW, "std": float(r["std. dev."]) * eV_to_MW}
                       for _, r in heat.iterrows()},
        # flux tally is track length (cm per source neutron); fluence per FPY needs volume
        "coil_fast_flux_cm_per_src": float(ff["mean"].iloc[0]),
        "neutrons_per_s": NEUTRONS_PER_S,
    }
    out["heating_MW_total"] = sum(v["mean"] for v in out["heating_MW"].values())
    json.dump(out, open(os.path.join(d, "result.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--no-photons", action="store_true")
    ap.add_argument("--tag", default="arc_baseline")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=")
        over[k] = v if k in ("fw_material", "breeder", "layout") else float(v)
    print(json.dumps(run(over, a.particles, a.batches, not a.no_photons, a.tag), indent=2))

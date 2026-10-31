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
    li6_pct=90.0,
    blanket_in_cm=20.0, blanket_out_cm=90.0, blanket_vert_cm=70.0,
    tank_cm=3.0,
    shield_in_cm=51.0, shield_out_cm=24.0, shield_vert_cm=28.0,
    coil_cm=64.0,
)


def materials(p):
    m = {}
    W = openmc.Material(name="tungsten"); W.add_element("W", 1.0); W.set_density("g/cm3", 19.3)
    inc = openmc.Material(name="inconel718")
    for el, wt in [("Ni", 52.5), ("Cr", 19.0), ("Fe", 18.5), ("Nb", 5.1), ("Mo", 3.05),
                   ("Ti", 0.9), ("Al", 0.5), ("Co", 0.47), ("C", 0.04), ("Mn", 0.18),
                   ("Si", 0.18)]:
        inc.add_element(el, wt, "wo")
    inc.set_density("g/cm3", 8.19)
    flibe = openmc.Material(name="flibe")
    flibe.add_elements_from_formula("F4Li2Be", enrichment_target="Li6",
                                    enrichment_type="ao", enrichment=p["li6_pct"])
    flibe.set_density("g/cm3", 1.94)
    be = openmc.Material(name="beryllium"); be.add_element("Be", 1.0); be.set_density("g/cm3", 1.85)
    tih2 = openmc.Material(name="TiH2"); tih2.add_elements_from_formula("TiH2"); tih2.set_density("g/cm3", 3.75)
    # coil pack: REBCO + Cu + steel case smeared; steel-dominated (only the
    # fluence reaching it matters here, not its own breeding)
    coil = openmc.Material(name="coil")
    for el, wt in [("Fe", 50.0), ("Cr", 12.0), ("Ni", 8.0), ("Cu", 25.0), ("Mn", 1.5), ("Mo", 1.5), ("Y", 0.5), ("Ba", 1.0), ("Gd", 0.5)]:
        coil.add_element(el, wt, "wo")
    coil.set_density("g/cm3", 7.9)
    cfc = openmc.Material(name="cfc"); cfc.add_element("C", 1.0); cfc.set_density("g/cm3", 1.9)
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
              ("channel", p["channel_cm"], mats["flibe"]), ("be", p["be_cm"], mats["Be"]),
              ("vv_out", p["vv_out_cm"], mats["inconel"])]
    polys = [fw_poly]
    for _, t, _ in layers:
        polys.append(polys[-1].offset(t))
    cells = {"plasma": openmc.Cell(name="plasma", region=-polys[0])}
    for i, (name, t, mat) in enumerate(layers):
        if t > 0:
            cells[name] = openmc.Cell(name=name, fill=mat, region=+polys[i] & -polys[i + 1])
    vv_outer = polys[-1]

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
        over[k] = v if k == "fw_material" else float(v)
    print(json.dumps(run(over, a.particles, a.batches, not a.no_photons, a.tag), indent=2))

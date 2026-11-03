"""OKTAVIAN (Osaka) 14 MeV sphere benchmarks in OpenMC: shared materials, sources, geometry
builders, tallies and response extraction. See NOTES.md for every geometry assumption.

Cases
  TBR spheres (tbr/readme.txt, Cetnar et al. JAERI-M 91-062):
    tbr_li, tbr_lic, tbr_pbli, tbr_pblic, tbr_beli, tbr_belic
  Be and Be+Li leakage (beli/readme.txt, Murakami et al. 1991):
    be57, be69, be97, be128 (bare Be shells, inner radius in mm, outer 173.5 mm),
    beli57, beli128 (Be shell + 40 cm Li)
  Leakage spectra (n-leak, Ichihara et al.): lif60, w40

Units of the computed responses follow the compilation exactly:
  rr6/rr7/al        reaction rate per atom per source neutron, x1e-24 (Tables 1.15/1.16)
  tpr               tritons per cm3 per source neutron (TLD tables)
  tbr               tritons per source neutron in the Li zone
  beli spectra      leakage neutrons per source neutron per MeV, bands: per source neutron
  n-leak spectra    leakage current per source neutron per unit lethargy, bands: per source neutron
"""
import json, os, copy
import numpy as np
import openmc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json")))
XS_BASE = "/work/blanket/xs/cross_sections.xml"
SAB_DIR = os.environ.get("OKT_SAB", "/bench/xs_sab")

# ----------------------------------------------------------------------------- geometry (cm)
# Defaults are the most defensible values found (sources in NOTES.md); every one of them is a
# key of GEOM so the sensitivity runs (sens.py) can move it.
GEOM = dict(
    # TBR spheres: Takahashi et al. JAERI-M 91-062 sec 5.3 (Figs 1-3, p.83) + Sugiyama et al.
    # NEACRP-A-1984-0657 (120 cm Li sphere). SUS316 vessels; A-block = Li shell 20-60 cm,
    # C-block = Li shell 10-20 cm (replaced by Pb or Be for the multiplier systems).
    a_rin=20.0, a_rout=60.0,     # A-block vessel outer surfaces
    a_wall_in=0.3,               # A inner wall: 3 mm (1984 paper); the 1991 paper says 2 mm
    a_wall_out=0.5,              # A outer wall 5 mm (both papers)
    cblk_rin=10.0, cblk_rout=20.0, cblk_wall=0.2,   # C-block Li vessel, 2 mm walls
    pb_rin=10.0, pb_rout=20.0,
    be_scale=1.0,                # multiplier on the shell densities (PRC 1.82, USA 1.84)
    c_thick=20.0,                # graphite D-block 60-80 cm (Figs 2, 3)
    li_dens=0.52,                # natural Li metal 0.534 x ~98 % filling (inferred, see NOTES)
    pb_dens=11.34, c_dens=1.70,  # not found in open sources: nominal Pb, typical reactor graphite
    li6_ao=None,                 # None: natural (OpenMC 7.59 at%)
    # leakage piles
    lif_dens=1.79, w_dens=4.43,
)


def xs_xml(base_xml, workdir, sab=True):
    """Copy a cross_sections.xml with absolute paths and the processed S(a,b) files added."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(base_xml); root = tree.getroot()
    d0 = os.path.dirname(os.path.abspath(base_xml))
    dd = root.find("directory")
    if dd is not None:
        d0 = os.path.join(d0, dd.text); root.remove(dd)
    for lib in root.findall("library"):
        p = lib.get("path")
        if not os.path.isabs(p):
            lib.set("path", os.path.join(d0, p))
    if sab:
        for name in ("c_Graphite_30p", "c_Graphite", "c_Be"):
            ET.SubElement(root, "library", materials=name, path=os.path.join(SAB_DIR, name + ".h5"), type="thermal")
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, "cross_sections.xml")
    tree.write(out)
    return out


# ----------------------------------------------------------------------------- materials
def mat_li(g, name="li"):
    m = openmc.Material(name=name)
    if g["li6_ao"] is None:
        m.add_element("Li", 1.0)
    else:
        m.add_element("Li", 1.0, enrichment_target="Li6", enrichment_type="ao", enrichment=100 * g["li6_ao"])
    m.set_density("g/cm3", g["li_dens"]); return m


def mat_ss(name="ss304", dens=7.93, comp=None):
    """SUS304. Default: typical composition. comp='okt' is the compilation's MCNP card (Fig 4.21)."""
    m = openmc.Material(name=name)
    if comp == "316":
        for el, w in (("Fe", 65.5), ("Cr", 17.0), ("Ni", 12.0), ("Mo", 2.5), ("Mn", 2.0), ("Si", 1.0)):
            m.add_element(el, w, "wo")
    elif comp == "okt":
        for el, w in (("Cr", 18.5), ("Fe", 70.4), ("Ni", 11.1)):
            m.add_element(el, w, "wo")
    else:
        for el, w in (("Fe", 69.5), ("Cr", 19.0), ("Ni", 9.5), ("Mn", 1.5), ("Si", 0.5)):
            m.add_element(el, w, "wo")
    m.set_density("g/cm3", dens); return m


BE_SHELLS = [(2.5, 5.7, "usa"), (5.7, 6.9, "prc"), (6.9, 9.7, "prc"), (9.7, 12.8, "prc"),
             (12.8, 15.25, "usa"), (15.25, 17.35, "usa")]   # INDC(CCP)-401 Table 1


def mat_be(origin, scale=1.0):
    """Combined US/PRC Be sphere shells, INDC(CCP)-401 Table 2 (wt %, density g/cm3)."""
    m = openmc.Material(name=f"be_{origin}")
    if origin == "usa":
        comp, dens = (("Be", 98.82), ("O", 0.72), ("Fe", 0.13), ("C", 0.11)), 1.84
    else:
        comp, dens = (("Be", 98.9), ("O", 0.5), ("Fe", 0.6)), 1.82
    for el, w in comp:
        m.add_element(el, w, "wo")
    m.set_density("g/cm3", dens * scale)
    m.add_s_alpha_beta("c_Be"); return m


def be_layers(rin, rout=17.35):
    return [(r1, "be_" + o) for r0, r1, o in BE_SHELLS if r0 >= rin - 1e-9 and r1 <= rout + 1e-9]


def mat_c(dens, name="graphite"):
    m = openmc.Material(name=name); m.add_element("C", 1.0); m.set_density("g/cm3", dens)
    m.add_s_alpha_beta("c_Graphite_30p"); return m


def mat_pb(dens, name="pb"):
    m = openmc.Material(name=name); m.add_element("Pb", 1.0); m.set_density("g/cm3", dens); return m


def mat_lif(dens):
    """LiF powder (lif60.dat): LiF 98.07 wt%, F 1.83 wt% (as listed), 6Li 7.463 at%."""
    m = openmc.Material(name="lif")
    li_w = 0.9807 * 6.94 / 25.94
    m.add_element("Li", li_w, "wo", enrichment_target="Li6", enrichment_type="ao", enrichment=7.463)
    m.add_element("F", 1 - li_w, "wo")
    m.set_density("g/cm3", dens); return m


def mat_w(dens):
    m = openmc.Material(name="w"); m.add_element("W", 1.0); m.set_density("g/cm3", dens); return m


# ----------------------------------------------------------------------------- sources
def source_beli():
    """Table 2.1 (1991 D-T source, per MeV per source neutron), histogram, isotropic."""
    b = sorted(DATA["beli"]["source"], key=lambda x: x["lo"])
    x = [b[0]["lo"] * 1e6] + [r["hi"] * 1e6 for r in b]
    p = [r["v"] for r in b] + [0.0]
    return openmc.stats.Tabular(x, p, "histogram")


def source_nleak():
    """Table 4.2 (pulsed beam line, LiF/W runs; per unit lethargy), histogram, isotropic."""
    b = sorted(DATA["nleak"]["source"], key=lambda x: x["lo"])
    x = [b[0]["lo"] * 1e6] + [r["hi"] * 1e6 for r in b]
    p = [r["v"] * np.log(r["hi"] / r["lo"]) / (r["hi"] - r["lo"]) for r in b] + [0.0]
    return openmc.stats.Tabular(x, p, "histogram")


def make_settings(spectrum, particles, batches, seed):
    s = openmc.Settings()
    s.run_mode = "fixed source"; s.particles = int(particles); s.batches = int(batches)
    src = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)), angle=openmc.stats.Isotropic(),
                                   energy=spectrum)
    s.source = src; s.seed = int(seed)
    s.photon_transport = False
    s.output = {"tallies": False, "summary": False}
    return s


# ----------------------------------------------------------------------------- geometry helpers
def shells(radii, fills, vacuum_r):
    """Concentric shells: radii ascending (outer radius of each region), fills list (None=void).
    Returns (geometry, surfaces dict by radius, cells list)."""
    surfs = [openmc.Sphere(r=r) for r in radii]
    outer = openmc.Sphere(r=vacuum_r, boundary_type="vacuum")
    cells, inner = [], None
    for s, f in zip(surfs, fills):
        reg = -s if inner is None else (+inner & -s)
        cells.append(openmc.Cell(fill=f, region=reg)); inner = s
    cells.append(openmc.Cell(fill=None, region=+inner & -outer))
    return openmc.Geometry(cells), surfs, cells


# ----------------------------------------------------------------------------- cases
TBR_CASES = ["tbr_li", "tbr_lic", "tbr_pbli", "tbr_pblic", "tbr_beli", "tbr_belic"]
BELI_CASES = ["be57", "be69", "be97", "be128", "beli57", "beli128"]
NLEAK_CASES = ["lif60", "w40"]
ALL_CASES = TBR_CASES + BELI_CASES + NLEAK_CASES


def a_block(g):
    return [(g["a_rin"] + g["a_wall_in"], "ss"), (g["a_rout"] - g["a_wall_out"], "li"), (g["a_rout"], "ss")]


def tbr_layers(case, g):
    """Radial layers (outer radius, material key) for the TBR spheres (1D: the beam duct, which is
    plugged with Li (B-block) or graphite (E-block), and the target assembly are not modelled)."""
    if case in ("tbr_li", "tbr_lic"):
        w = g["cblk_wall"]
        L = [(g["cblk_rin"], None), (g["cblk_rin"] + w, "ss"), (g["cblk_rout"] - w, "li"), (g["cblk_rout"], "ss")]
    elif case in ("tbr_pbli", "tbr_pblic"):
        L = [(g["pb_rin"], None), (g["pb_rout"], "pb")]
        if g["pb_rout"] < g["a_rin"] - 1e-9:
            L.append((g["a_rin"], None))
    else:
        L = [(5.7, None)] + be_layers(5.7) + [(g["a_rin"], None)]
    L += a_block(g)
    if case.endswith("c"):
        L.append((g["a_rout"] + g["c_thick"], "c"))
    return L


def build(case, xs_file, particles, batches, seed, geom=None, sigma_lib=None):
    """Returns (model, meta). meta carries what extract() needs.
    sigma_lib: dict nuclide -> HDF5 path for the detector response functions (default: base)."""
    openmc.reset_auto_ids()
    g = dict(GEOM); g.update(geom or {})
    mats = {}
    tallies = []
    meta = dict(case=case, geom=g)
    if case in TBR_CASES:
        mats = dict(li=mat_li(g), pb=mat_pb(g["pb_dens"]), be_usa=mat_be("usa", g["be_scale"]),
                    be_prc=mat_be("prc", g["be_scale"]), c=mat_c(g["c_dens"]), ss=mat_ss("ss316", 7.95, comp="316"))
        L = tbr_layers(case, g)
        radii = [r for r, _ in L]; fills = [mats[k] if k else None for _, k in L]
        rmax = radii[-1]
        geo, surfs, cells = shells(radii, fills, rmax + 5.0)
        li_cells = [c for c, (_, k) in zip(cells, L) if k == "li"]
        t = openmc.Tally(name="tbr"); t.filters = [openmc.CellFilter(li_cells)]; t.scores = ["H3-production"]
        tallies.append(t)
        # radial response mesh: 0.25 cm shells
        mesh = openmc.SphericalMesh(r_grid=np.arange(0.0, rmax + 1e-6, 0.25))
        meta["r_grid"] = list(mesh.r_grid)
        fns = response_functions(sigma_lib)
        for k, (E, s) in fns.items():
            t = openmc.Tally(name=f"rr_{k}")
            t.filters = [openmc.MeshFilter(mesh), openmc.EnergyFunctionFilter(E, s)]
            t.scores = ["flux"]; tallies.append(t)
        li = mats["li"]
        dens = li.get_nuclide_atom_densities()  # atoms/b-cm
        meta["N6"], meta["N7"] = float(dens.get("Li6", 0.0)), float(dens.get("Li7", 0.0))
        meta["layers"] = L
        spectrum = source_nleak() if case in ("tbr_li", "tbr_pbli") else source_beli()
    elif case in BELI_CASES:
        rin = {"be57": 5.7, "be69": 6.9, "be97": 9.7, "be128": 12.8, "beli57": 5.7, "beli128": 12.8}[case]
        mats = dict(be_usa=mat_be("usa", g["be_scale"]), be_prc=mat_be("prc", g["be_scale"]), li=mat_li(g),
                    ss=mat_ss("ss316", 7.95, comp="316"))
        L = [(rin, None)] + be_layers(rin)
        if case.startswith("beli"):   # Be shell inside the A-block Li shell (same assembly as tbr_beli)
            L += [(g["a_rin"], None)] + a_block(g)
        radii = [r for r, _ in L]; fills = [mats[k] if k else None for _, k in L]
        geo, surfs, cells = shells(radii, fills, radii[-1] + 5.0)
        tallies += leak_tallies(surfs[-1], DATA["beli"][case])
        spectrum = source_beli()
        meta["layers"] = L
    elif case in NLEAK_CASES:
        geo, outer_surf, L = nleak_geometry(case, g)
        tallies += leak_tallies(outer_surf, DATA["nleak"][case])
        spectrum = source_nleak()
        meta["layers"] = L
    else:
        raise KeyError(case)
    model = openmc.Model(geometry=geo, settings=make_settings(spectrum, particles // batches, batches, seed),
                         tallies=openmc.Tallies(tallies))
    model.materials = openmc.Materials(list({id(m): m for m in geo.get_all_materials().values()}.values()))
    model.materials.cross_sections = xs_file
    return model, meta


def nleak_geometry(case, g):
    """Type-I (LiF, 61 cm) and Type-II (W, 40 cm) vessels, re-entrant duct along +z.
    Type-II follows the compilation's MCNP input (Fig 4.21) surface by surface; Type-I from the
    text: ID 60 cm, 0.5 cm SUS-304 wall, 5.1 cm diameter re-entrant hole up to the centre."""
    ss = mat_ss("ss304", 7.86, comp="okt")  # SINBAD NEA-1553/53 composition and density
    if case == "w40":
        smp = mat_w(g["w_dens"])
        c1, c2 = openmc.ZCylinder(r=5.55), openmc.ZCylinder(r=5.75)
        s3, s4, s5, s6 = (openmc.Sphere(r=r) for r in (10.0, 10.2, 19.75, 19.95))  # 39.9 cm OD
        s7 = openmc.Sphere(r=60.0, boundary_type="vacuum"); p8 = openmc.ZPlane(z0=8.32)
        cells = [openmc.Cell(fill=None, region=(-s3 & -p8) | (+p8 & -c1 & -s6)),
                 openmc.Cell(fill=ss, region=(-s4 & +s3 & -p8) | (+p8 & -c2 & +c1 & -s6)),
                 openmc.Cell(fill=smp, region=(-s5 & +s4 & -p8) | (-s5 & +c2 & +p8)),
                 openmc.Cell(fill=ss, region=(-s6 & +s5 & -p8) | (-s6 & +s5 & +p8 & +c2)),
                 openmc.Cell(fill=None, region=+s6 & -s7)]
        return openmc.Geometry(cells), s6, [(10.0, None), (10.2, "ss"), (19.75, "w"), (19.95, "ss")]
    # LiF Type-I, as modelled in JAERI-Data/Code 98-024 Fig A-15 (Cu 60 cm pile): duct CX 2.5 with a
    # 0.3 cm wall (to 2.8) running in from the +z side and closed 2.5 cm past the centre (PX -2.5/-2.8),
    # shell SO 30.0/30.5, no central void; steel 7.87 g/cm3
    ss = mat_ss("ss304", 7.87, comp="okt")
    smp = mat_lif(g["lif_dens"])
    c1, c2 = openmc.ZCylinder(r=2.5), openmc.ZCylinder(r=2.8)
    p1, p2 = openmc.ZPlane(z0=-2.5), openmc.ZPlane(z0=-2.8)
    s_in, s_out = openmc.Sphere(r=30.0), openmc.Sphere(r=30.5)
    s_vac = openmc.Sphere(r=60.0, boundary_type="vacuum")
    duct_void = -c1 & +p1 & -s_out
    duct_wall = ((+c1 & -c2 & +p1) | (-c2 & +p2 & -p1)) & -s_out
    duct_all = -c2 & +p2 & -s_out
    cells = [openmc.Cell(fill=None, region=duct_void), openmc.Cell(fill=ss, region=duct_wall),
             openmc.Cell(fill=smp, region=-s_in & ~duct_all),
             openmc.Cell(fill=ss, region=+s_in & -s_out & ~duct_all),
             openmc.Cell(fill=None, region=+s_out & -s_vac)]
    return openmc.Geometry(cells), s_out, [(2.5, None), (30.0, "lif"), (30.5, "ss")]


def leak_tallies(surf, bins):
    edges = sorted({b["lo"] for b in bins} | {b["hi"] for b in bins})
    ef = openmc.EnergyFilter(np.array(edges) * 1e6)
    t = openmc.Tally(name="leak"); t.filters = [openmc.SurfaceFilter([surf]), ef]; t.scores = ["current"]
    # the beam comes down the re-entrant duct (duct on the +z side, beam direction -z): the detector
    # at 55 deg to the beam sees neutrons emitted at polar angle 125 deg; cone 120-130 deg
    t2 = openmc.Tally(name="leak55")
    t2.filters = [openmc.SurfaceFilter([surf]), openmc.PolarFilter(np.radians([0, 120, 130, 180])), ef]
    t2.scores = ["current"]
    return [t, t2]


_SIG_CACHE = {}


def response_functions(sigma_lib=None):
    """Pointwise cross sections (barns) for EnergyFunctionFilter: Li6 MT105, Li7 MT205, Al27 MT107.
    For perturbed runs, sigma_lib points Li6/Li7 at the sample's HDF5 so the detector response
    moves with the library, as it does in the calculation of the measured quantity."""
    import openmc.data
    sigma_lib = sigma_lib or {}
    out = {}
    for key, nuc, mt in (("6", "Li6", 105), ("7", "Li7", 205), ("al", "Al27", 107)):
        path = sigma_lib.get(nuc, f"/work/blanket/xs/ENDFB-8.0-NNDC_{nuc}.h5")
        if path not in _SIG_CACHE:
            _SIG_CACHE[path] = openmc.data.IncidentNeutron.from_hdf5(path)
        d = _SIG_CACHE[path]
        T = "294K"
        E = np.asarray(d.energy[T]); xs = d[mt].xs[T]
        y = xs(E)
        y = np.where(E < xs.x[0], 0.0, y)
        keep = np.ones(len(E), bool)
        out[key] = (E[keep], np.clip(y[keep], 0, None))
    return out


# ----------------------------------------------------------------------------- extraction
def _shell_value(r_grid, vals, r):
    """Linear interpolation of a shell-averaged radial profile at radius r (no extrapolation
    across a material interface is attempted; shells are 0.25 cm)."""
    rg = np.asarray(r_grid); c = 0.5 * (rg[1:] + rg[:-1])
    i = int(np.clip(np.searchsorted(rg, r) - 1, 0, len(c) - 1))
    return i


def extract(sp, meta):
    """Return dict: id -> (value, std). TBR cases: tbr, rr6@r, rr7@r, tpr@r (every 0.25 cm bin
    centre index is resolvable; positions are looked up in responses.py). Leakage: per-bin spectra."""
    case = meta["case"]
    out = {}
    if case in TBR_CASES:
        t = sp.get_tally(name="tbr"); out["tbr"] = (float(t.mean.sum()), float(np.sqrt((t.std_dev ** 2).sum())))
        rg = np.asarray(meta["r_grid"]); vol = 4 / 3 * np.pi * (rg[1:] ** 3 - rg[:-1] ** 3)
        prof = {}
        for k in ("6", "7", "al"):
            t = sp.get_tally(name=f"rr_{k}")
            prof[k] = (t.mean.ravel() / vol, t.std_dev.ravel() / vol)
        out["_profile"] = dict(r_grid=list(rg), **{k: [list(v[0]), list(v[1])] for k, v in prof.items()},
                               N6=meta["N6"], N7=meta["N7"])
    else:
        for name in ("leak", "leak55"):
            t = sp.get_tally(name=name)
            m, s = t.mean.ravel(), t.std_dev.ravel()
            if name == "leak55":
                # polar bins [0-120, 120-130, 130-180] x energy; keep 120-130 deg, scale to 4pi per sr
                ne = len(t.filters[-1].bins)
                m, s = m.reshape(3, ne)[1], s.reshape(3, ne)[1]
                dom = 2 * np.pi * (np.cos(np.radians(120)) - np.cos(np.radians(130)))
                m, s = m * 4 * np.pi / dom, s * 4 * np.pi / dom
            edges = t.filters[-1].values / 1e6
            out["_" + name] = dict(edges=list(edges), mean=list(m), std=list(s))
    return out

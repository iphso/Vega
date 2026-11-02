"""Shared pieces of the activation / decay-heat / shutdown-dose tier (see NOTES.md).

Cells, materials and volumes of the ARC baseline (../arc_model.py), the depletion
chain, its pruning, the coarse activation groups and the contact-dose constants.
Everything here is used both by the browser-sidecar generator and by the OpenMC
depletion reference, so the two read the same inputs.
"""
import os, sys
import xml.etree.ElementTree as ET
import numpy as np

ACT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ACT))
from arc_model import build, miller, NEUTRONS_PER_S, DEFAULTS, FUSION_POWER_MW  # noqa: E402,F401

CHAIN = os.path.join(ACT, "data", "chain_endf_b8.0.xml")
CELLS = ["fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield", "coil"]
FPY_S = 3.15576e7
# reaction depth kept in the browser chain: products of products (decay closure unlimited)
DEPTH = 2

# cooling times reported (s)
COOL = {"1 s": 1.0, "1 min": 60.0, "1 h": 3600.0, "1 d": 86400.0, "1 wk": 604800.0,
        "1 mo": 2.62974e6, "1 y": FPY_S, "10 y": 10 * FPY_S, "100 y": 100 * FPY_S}


def lib_nuclides():
    """Nuclides with neutron data in the cross-section library."""
    root = ET.parse(os.environ["OPENMC_CROSS_SECTIONS"]).getroot()
    return {e.get("materials") for e in root if e.tag == "library" and e.get("type") == "neutron"}


def cell_materials(over=None):
    """build() + per-cell material, nuclide atom densities (atoms/b-cm) and volumes (cm3)."""
    import openmc
    model, cells, p = build(over or None, 1000, 1, photons=False)
    mats = {n: cells[n].fill for n in CELLS if n in cells}
    vol = cell_volumes(p)
    return model, cells, p, mats, vol


def poly_moment(pts):
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    return abs(np.sum((x + x1) * (x * y1 - x1 * y)) / 6.0)


def cell_volumes(p):
    """Exact cell volumes (cm3): VV layers as solids of revolution of the mitred offset
    polygons (the construction arc_model uses), the boxes analytically."""
    import openmc
    fw_poly = openmc.model.Polygon(miller(p["R0"], p["a"], p["kappa"], p["delta"]), basis="rz")
    layers = [("fw", p["fw_cm"]), ("vv_in", p["vv_in_cm"]), ("channel", p["channel_cm"]),
              ("be", p["be_cm"]), ("vv_out", p["vv_out_cm"])]
    vol, acc = {}, 0.0
    prev = poly_moment(np.asarray(fw_poly.points))
    for n, t in layers:
        acc += t
        mo = poly_moment(np.asarray(fw_poly.offset(acc).points))
        vol[n] = 2 * np.pi * (mo - prev)
        prev = mo
    vv_solid = 2 * np.pi * prev
    pts = miller(p["R0"], p["a"], p["kappa"], p["delta"], 720)
    vv_th = acc
    b_rmin, b_rmax = pts[:, 0].min() - vv_th - p["blanket_in_cm"], pts[:, 0].max() + vv_th + p["blanket_out_cm"]
    b_z = pts[:, 1].max() + vv_th + p["blanket_vert_cm"]
    tk = p["tank_cm"]
    cyl = lambda r0, r1, z: np.pi * (r1 ** 2 - r0 ** 2) * 2 * z
    vb = cyl(b_rmin, b_rmax, b_z)
    vt = cyl(b_rmin - tk, b_rmax + tk, b_z + tk)
    s_rmin, s_rmax, s_z = b_rmin - tk - p["shield_in_cm"], b_rmax + tk + p["shield_out_cm"], b_z + tk + p["shield_vert_cm"]
    vs = cyl(s_rmin, s_rmax, s_z)
    c = p["coil_cm"]
    vc = cyl(max(1.0, s_rmin - c), s_rmax + c, s_z + c)
    vol["blanket"] = vb - vv_solid
    vol["tank"] = vt - vb
    vol["shield"] = vs - vt
    vol["coil"] = vc - vs
    return vol


def load_chain():
    import openmc.deplete as dep
    return dep.Chain.from_xml(CHAIN)


def secondaries(rtype, chain):
    """Light nuclides a reaction also makes (OpenMC's REACTIONS table: (n,t) -> H3, (n,a) ->
    He4, ...), those present in the chain -- what openmc.deplete adds to the matrix."""
    from openmc.deplete.chain import REACTIONS
    info = REACTIONS.get(rtype)
    return [x for x in (info.secondaries if info else ()) if x in chain.nuclide_dict]


def prune(chain, initial, lib, depth=DEPTH):
    """Nuclides reachable from `initial` through <= depth neutron reactions (on nuclides
    with data, their light secondaries included) and any number of decays.
    Returns {name: reaction depth}."""
    q = [(n, 0) for n in initial]
    reach = {n: 0 for n in initial}
    while q:
        n, d = q.pop()
        if n not in chain.nuclide_dict:
            continue
        nuc = chain[n]
        for m in nuc.decay_modes:
            if m.target and m.target in chain.nuclide_dict and (m.target not in reach or reach[m.target] > d):
                reach[m.target] = d
                q.append((m.target, d))
        if n in lib and d < depth:
            for r in nuc.reactions:
                if r.type == "fission":
                    continue
                for tgt in [r.target] + secondaries(r.type, chain):
                    if tgt and tgt in chain.nuclide_dict and (tgt not in reach or reach[tgt] > d + 1):
                        reach[tgt] = d + 1
                        q.append((tgt, d + 1))
    return reach


# ---- contact dose (semi-infinite slab surface), FISPACT-II's formula:
#   D [Sv/h] = C * (B/2) * sum_i (mu_en/rho)_air(E_i) / (mu/rho)_mat(E_i) * E_i * S_i
# S_i photons per KILOGRAM per s at E_i (MeV), B = 2 (build-up), C = 5.76e-10 Sv/h per MeV/kg/s
# (1 MeV/kg/s = 1.602e-13 Gy/s = 5.77e-10 Gy/h; dose_check.py verifies it against photon MC)
# units, mu in cm2/g. One half of the infinite-medium air-kerma: the surface sees a half space.
DOSE_C = 5.76e-10
DOSE_B = 2.0
def photon_lines(nuc):
    """Discrete decay photon (energy eV, photons per second per atom) of a chain nuclide."""
    src = nuc.sources.get("photon") if nuc.sources else None
    if src is None:
        return np.zeros(0), np.zeros(0)
    import openmc.stats
    def lines_of(d, w=1.0):
        if isinstance(d, openmc.stats.Discrete):
            return np.asarray(d.x, float), w * np.asarray(d.p, float)
        if isinstance(d, openmc.stats.Tabular):
            # continuum: pseudo-lines at interval midpoints carrying each interval's integral
            x, p = np.asarray(d.x, float), np.asarray(d.p, float)
            dx = np.diff(x)
            ip = p[:-1] * dx if d.interpolation == "histogram" else 0.5 * (p[:-1] + p[1:]) * dx
            return 0.5 * (x[:-1] + x[1:]), w * ip
        if isinstance(d, openmc.stats.Mixture):
            parts = [lines_of(dd, w * pw) for pw, dd in zip(d.probability, d.distribution)]
            return np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
        raise TypeError(f"{nuc.name}: unhandled photon source {type(d)}")
    return lines_of(src)


def mu_rho_material(mat, E):
    """Total mass attenuation coefficient (cm2/g, no coherent) of an openmc.Material at
    energies E (eV), from the photoatomic library."""
    import openmc.data
    root = ET.parse(os.environ["OPENMC_CROSS_SECTIONS"]).getroot()
    base = os.path.dirname(os.environ["OPENMC_CROSS_SECTIONS"])
    paths = {e.get("materials"): os.path.join(base, e.get("path")) for e in root if e.tag == "library" and e.get("type") == "photon"}
    wf = {}
    for nuc in mat.get_nuclide_atom_densities():
        w = mat.get_mass_density(nuc)
        el = openmc.data.ATOMIC_SYMBOL[openmc.data.zam(nuc)[0]]
        wf[el] = wf.get(el, 0.0) + w
    rho = sum(wf.values())
    mu = np.zeros_like(E, dtype=float)
    for el, w in wf.items():
        ph = openmc.data.IncidentPhoton.from_hdf5(paths[el])
        sig = np.zeros_like(E, dtype=float)
        for mt in (502, 504, 515, 517, 522):   # total attenuation, as FISPACT's mu_m
            if mt in ph.reactions:
                sig += ph.reactions[mt].xs(E)
        Z = openmc.data.ATOMIC_NUMBER[el]
        A = openmc.data.atomic_weight(el)
        mu += (w / rho) * sig * 1e-24 * 6.02214076e23 / A   # cm2/g
    return mu


def dose_coeffs(mat, names, lines):
    """Contact-dose coefficient per nuclide in `mat`: sum over its decay photons (>= 1 keV) of
    (mu_en/rho)_air / (mu/rho)_mat * E[MeV] * photons/s per atom. `lines[name]` = photon_lines."""
    import openmc.data
    air = openmc.data.mass_energy_absorption_coefficient("air")
    Eu = sorted({float(e) for x in names for e in lines[x][0] if e >= 1e3})
    ratio = {}
    if Eu:
        Ea = np.array(Eu)
        ratio = dict(zip(Eu, air(Ea) / mu_rho_material(mat, Ea)))
    return np.array([sum(ratio[float(e)] * e * 1e-6 * i for e, i in zip(*lines[x]) if e >= 1e3) for x in names])

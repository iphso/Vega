"""OpenMC models of the JAERI FNS Li2O and Be cylindrical assemblies (JAERI-M 94-014 ch. 7 and 8).

Geometry: equivalent cylinders (r_eq 31.5 cm) on the source axis (+z), front face 20 cm from the
point D-T source, as in the compilation's own MCNP/DOT inputs (f8-8/9/10.txt). Optional concrete
target room ("room" model) for room-return. Detector responses are computed as
    R(z) = integral phi(E, z) sigma_det(E) dE      [reactions / atom / D-T neutron]
with an EnergyFunctionFilter holding the pointwise 294 K detector cross section taken from the SAME
HDF5 library the transport uses (so perturbed libraries perturb both transport and response),
tallied on thin cylindrical mesh bins on the axis. This is the infinitely dilute detector
response, i.e. the quantity the experimenters report after their self-shielding corrections.
"""
import os, math
import xml.etree.ElementTree as ET
import numpy as np
import openmc
import openmc.data

import fnsdata as D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_XS = "/work/blanket/xs"
FNS_XS = "/fnsruns/xs"
EXTRA = {"Np237": "/fnsruns/xs_sab/ENDFB-8.0-NNDC_Np237.h5"}
SAB = {"c_Be": "/fnsruns/xs/c_Be.h5"}
REFDIR = os.path.join(HERE, "..", "..", "refs", "lit", "jaerim94014", "benchmarks", "jaerim94014")


# ------------------------------------------------------------------------------------------ data
def write_xs_xml(perturbed, out_xml):
    """Base ENDF/B-VIII.0 library (absolute paths) + Np237 + c_Be, with perturbed nuclides swapped."""
    tree = ET.parse(os.path.join(BASE_XS, "cross_sections.xml"))
    root = tree.getroot()
    for d in root.findall("directory"):
        root.remove(d)
    have = set()
    for lib in root.findall("library"):
        m = lib.get("materials")
        have.add(m)
        lib.set("path", os.path.abspath(perturbed[m]) if m in perturbed else os.path.join(BASE_XS, lib.get("path")))
    for m, p in EXTRA.items():
        if m not in have:
            ET.SubElement(root, "library", materials=m, path=perturbed.get(m, p), type="neutron")
    for m, p in SAB.items():
        if m not in have:
            ET.SubElement(root, "library", materials=m, path=p, type="thermal")
    os.makedirs(os.path.dirname(out_xml), exist_ok=True)
    tree.write(out_xml)


def xs_paths(xml):
    root = ET.parse(xml).getroot()
    return {lib.get("materials"): lib.get("path") for lib in root.findall("library")}


# detector response functions: id -> (nuclide, MT list summed)
RESPONSES = {
    "li6_t": ("Li6", [205]), "li7_t": ("Li7", [205]),
    "u235_f": ("U235", [18]), "u238_f": ("U238", [18]), "np237_f": ("Np237", [18]),
    "th232_f": ("Th232", [18]), "al27_na": ("Al27", [107]), "ni58_n2n": ("Ni58", [16]),
    "ni58_np": ("Ni58", [103]), "fe56_np": ("Fe56", [103]), "zr90_n2n": ("Zr90", [16]),
    "au197_ng": ("Au197", [102]),
}


def response_xs(rid, xml, T="294K"):
    nuc, mts = RESPONSES[rid]
    data = openmc.data.IncidentNeutron.from_hdf5(xs_paths(xml)[nuc])
    E = data.energy[T]
    y = np.zeros_like(E)
    for mt in mts:
        if mt not in data.reactions and mt == 18:
            parts = [m for m in (19, 20, 21, 38) if m in data.reactions]
        else:
            parts = [mt]
        for m in parts:
            t = data.reactions[m].xs[T]
            j = np.searchsorted(E, t.x[0])
            y[j:j + len(t.y)] += t.y
    # drop leading zeros except one (keeps the threshold edge)
    nz = np.nonzero(y)[0]
    k = max(nz[0] - 1, 0) if len(nz) else 0
    return E[k:], y[k:]


# ---------------------------------------------------------------------------------------- source
def read_source_table(case):
    """Return ascending edges [eV] and bin contents [n / bin / D-T reaction] at 0 degrees."""
    if case == "li2o":
        lines = open(os.path.join(REFDIR, "li2o", "read.me"), "rb").read().decode("latin1").splitlines()
        i = [k for k, l in enumerate(lines) if l.startswith("Table 7.21")][0]
        g = {}
        for l in lines[i + 2:i + 75]:
            t = l.split()
            if len(t) >= 3 and t[0].isdigit():
                g[int(t[0])] = (float(t[1]), float(t[2]))
            if len(t) >= 6 and t[3].isdigit():
                g[int(t[3])] = (float(t[4]), float(t[5]))
            elif len(t) == 5 and t[3].isdigit():   # "126  1.0010e-05" (bound only) after a full triplet
                g[int(t[3])] = (float(t[4]), None)
            elif len(t) == 2 and t[0].isdigit():
                g[int(t[0])] = (float(t[1]), None)
    else:
        g = {}
        for l in open(os.path.join(REFDIR, "be", "t8-16.txt")).read().replace("\r", "").splitlines()[1:]:
            t = l.split()
            if len(t) >= 2:
                g[int(t[0])] = (float(t[1]), float(t[2]) if len(t) > 2 else None)
    n = max(g)
    upper = np.array([g[k][0] for k in range(1, n + 1)])       # descending, last = lowest bound
    val = np.array([g[k][1] for k in range(1, n)])              # groups 1..n-1
    edges = upper[::-1]
    p = val[::-1]
    return edges, p


def dt_kinematics(Ed_MeV, mu):
    """Lab neutron energy [MeV] for T(d,n)4He at deuteron energy Ed, lab cosine mu (non-relativistic)."""
    md, mt, mn, ma, Q = 2.01410, 3.01605, 1.00866, 4.00260, 17.5893
    a = math.sqrt(md * mn * Ed_MeV) * mu
    b = a * a + (ma + mn) * (ma * Q + (ma - md) * Ed_MeV)
    return ((a + math.sqrt(b)) / (ma + mn)) ** 2


def lab_anisotropy(Ed_MeV, mus, n=2_000_000, seed=1):
    """dN/dOmega_lab(mu) / (1/4pi) for CM-isotropic emission (Jacobian only), by sampling."""
    md, mt, mn, ma, Q = 2.01410, 3.01605, 1.00866, 4.00260, 17.5893
    rng = np.random.default_rng(seed)
    Ecm = Ed_MeV * mt / (md + mt)
    En_cm = (Q + Ecm) * ma / (ma + mn)
    vn = math.sqrt(2 * En_cm / mn)
    vc = math.sqrt(2 * Ed_MeV * md) / (md + mt)
    mucm = rng.uniform(-1, 1, n)
    vz = vn * mucm + vc
    vperp = vn * np.sqrt(1 - mucm**2)
    mulab = vz / np.hypot(vz, vperp)
    h, e = np.histogram(mulab, bins=400, range=(-1, 1), density=True)
    c = 0.5 * (e[1:] + e[:-1])
    return np.interp(mus, c, h * 2.0)


def make_sources(case, room=False, kinematic=False, mu_cut=0.52, Ed_eff=None):
    """Point D-T source at the origin, beam axis +z. Returns (sources, total strength per D-T).

    'spec' (default) = the compilation's prescription, as in its MCNP input (f8-8: SRC weight 1.1261,
    SDIR cone towards the assembly): the 0-degree spectrum (absolute, per D-T per 4pi sr) everywhere
    in the forward cone. mu_cut 0.52 < 20/sqrt(20^2+31.5^2) = 0.536, i.e. the whole front face.
    room=True adds the rest of the sphere, carrying the remaining 1 - S0(1-mu_cut)/2 neutrons per
    D-T with the same spectrum, so that the total is one neutron per D-T reaction.
    kinematic=True replaces the flat cone by 8 cosine bins whose peak energy follows D-T kinematics
    and whose intensity follows the CM->lab Jacobian, normalised to S0 at 0 degrees.
    """
    S0 = D.LI2O_SRC_0DEG if case == "li2o" else D.BE_SRC_0DEG
    edges, p = read_source_table(case)
    width = np.diff(edges)
    sp = []

    def energy_dist(scale=1.0):
        if scale == 1.0:
            x = edges
        else:   # shift the D-T peak (above 13.5 MeV) by `scale`, blend to 1 between 12 and 13.5 MeV
            w = np.clip((edges - 12e6) / 1.5e6, 0, 1)
            x = edges * (1 + (scale - 1) * w)
        dens = np.append(p / np.diff(x), 0.0)
        return openmc.stats.Tabular(x, dens, interpolation="histogram")

    def cone(mu_lo, mu_hi, strength, scale=1.0):
        s = openmc.IndependentSource()
        s.space = openmc.stats.Point((0, 0, 0))
        s.angle = openmc.stats.PolarAzimuthal(openmc.stats.Uniform(mu_lo, mu_hi),
                                              openmc.stats.Uniform(0, 2 * math.pi), reference_uvw=(0, 0, 1))
        s.energy = energy_dist(scale)
        s.strength = strength
        s.particle = "neutron"
        return s

    if not kinematic:
        sp.append(cone(mu_cut, 1.0, S0 * (1 - mu_cut) / 2))
    else:
        # effective deuteron energy: the one whose 0-degree D-T energy matches the table's peak mean
        pk = edges[:-1] > 13.4e6
        Emean0 = float(np.sum(p[pk] * 0.5 * (edges[:-1] + edges[1:])[pk]) / p[pk].sum()) / 1e6
        if Ed_eff is None:
            lo, hi = 0.01, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if dt_kinematics(mid, 1.0) < Emean0 else (lo, mid)
            Ed_eff = 0.5 * (lo + hi)
        mub = np.linspace(mu_cut, 1.0, 9)
        J1 = lab_anisotropy(Ed_eff, [0.999])[0]
        for a, b in zip(mub[:-1], mub[1:]):
            m = 0.5 * (a + b)
            I = S0 * lab_anisotropy(Ed_eff, [m])[0] / J1
            sp.append(cone(a, b, I * (b - a) / 2, dt_kinematics(Ed_eff, m) / dt_kinematics(Ed_eff, 1.0)))
    if room:
        fwd = sum(s.strength for s in sp)
        sp.append(cone(-1.0, mu_cut, 1.0 - fwd))
    return sp, sum(s.strength for s in sp)


# ------------------------------------------------------------------------------------- materials
def mat(name, comp, trace=None):
    m = openmc.Material(name=name)
    for k, v in comp.items():
        if k in ("r",):
            continue
        if k[-1].isdigit():
            m.add_nuclide(k, v)
        else:
            m.add_element(k, v)
    m.set_density("sum")
    return m


def concrete():
    m = openmc.Material(name="concrete")   # ordinary concrete, NBS 03 (PNNL-15870 rev1), 2.35 g/cc
    for el, w in [("H", 0.008485), ("C", 0.050064), ("O", 0.473483), ("Mg", 0.024183), ("Al", 0.036063),
                  ("Si", 0.145100), ("S", 0.002970), ("K", 0.001697), ("Ca", 0.246924), ("Fe", 0.011031)]:
        m.add_element(el, w, "wo")
    m.set_density("g/cm3", 2.35)
    return m


def air():
    m = openmc.Material(name="air")
    m.add_element("N", 0.755268, "wo"); m.add_element("O", 0.231781, "wo"); m.add_element("Ar", 0.012827, "wo")
    m.set_density("g/cm3", 0.001205)
    return m


# -------------------------------------------------------------------------------------- geometry
def room_shell(inner_region_cells, mats, room):
    """Concrete box room around the assembly. Model axes: z = beam/assembly axis, y = up.
    Target 5.5 m from two walls (taken as the -x and -z walls, i.e. the assembly points away
    from the near south/west walls, an assumption: Fig. 7.3 is not in the compilation), 9.5 m
    from the other two, 7.9 m below the ceiling, 3.8 m above the basement floor. Walls 2 m.
    The steel grating floor 1.8 m below the target and the target/beam-line hardware are omitted.
    Returns the cells of air + concrete and the outer boundary region."""
    R = D.ROOM
    T = 200.0
    xi = (-R["near_wall"], R["size"] - R["near_wall"])
    zi = xi
    yi = (-R["basement"], R["ceiling"])
    inner = openmc.model.RectangularParallelepiped(xi[0], xi[1], yi[0], yi[1], zi[0], zi[1])
    outer = openmc.model.RectangularParallelepiped(xi[0] - T, xi[1] + T, yi[0] - T, yi[1] + T, zi[0] - T, zi[1] + T,
                                                   boundary_type="vacuum")
    return inner, outer


def build(case, room=False, kinematic=False, particles=100_000, batches=20, seed=1, zstep=0.5,
          xs_xml=None, energy_tally=True, full_sphere=False, room_air=True, fine_faces=False):
    """full_sphere: emit the backward (non-cone) neutrons also without a room, so that a no-room run
    and a room run at the same seed sample identical source particles; room_air=False leaves the room
    interior void, so histories stay identical inside the assembly up to the first leak and the
    room/no-room ratio is a correlated (low-noise) estimate of the room-return factor."""
    """Return (model, info). info carries what result extraction needs."""
    is_li2o = case == "li2o"
    L = D.LI2O_LENGTH if is_li2o else D.BE_LENGTH
    z0, z1 = 20.0, 20.0 + L
    R = 31.5
    xs_xml = xs_xml or os.path.join(FNS_XS, "cross_sections.xml")

    cells, mats = [], []
    pz0, pz1 = openmc.ZPlane(z0), openmc.ZPlane(z1)
    cyl = openmc.ZCylinder(r=R)
    slab = +pz0 & -pz1 & -cyl
    if is_li2o:
        mi, ms, mo = (mat("li2o_inner", D.LI2O_REGIONS["inner"]), mat("ss_drawer", D.LI2O_REGIONS["ss"]),
                      mat("li2o_outer", D.LI2O_REGIONS["outer"]))
        mats += [mi, ms, mo]
        c1, c2 = openmc.ZCylinder(r=D.LI2O_REGIONS["inner"]["r"]), openmc.ZCylinder(r=D.LI2O_REGIONS["ss"]["r"])
        cells += [openmc.Cell(name="inner", fill=mi, region=+pz0 & -pz1 & -c1),
                  openmc.Cell(name="drawer", fill=ms, region=+pz0 & -pz1 & +c1 & -c2),
                  openmc.Cell(name="outer", fill=mo, region=+pz0 & -pz1 & +c2 & -cyl)]
    else:
        mb = mat("be", D.BE_BLOCK); mb.add_s_alpha_beta("c_Be")
        md = mat("ss_drawer", D.BE_DRAWER)
        mats += [mb, md]
        sq_in = openmc.model.RectangularPrism(4.92, 4.92)    # |x|,|y| < 2.46
        sq_out = openmc.model.RectangularPrism(5.08, 5.08)   # |x|,|y| < 2.54
        cells += [openmc.Cell(name="be_in_drawer", fill=mb, region=+pz0 & -pz1 & -sq_in),
                  openmc.Cell(name="drawer", fill=md, region=+pz0 & -pz1 & +sq_in & -sq_out),
                  openmc.Cell(name="be", fill=mb, region=+pz0 & -pz1 & +sq_out & -cyl)]
    if room:
        inner, outer = room_shell(None, mats, D.ROOM)
        a, c = air(), concrete()
        mats += [a, c]
        cells += [openmc.Cell(name="room_air", fill=a if room_air else None, region=-inner & ~slab),
                  openmc.Cell(name="walls", fill=c, region=+inner & -outer)]
    else:
        # no room (as in the compilation's MCNP model): void around the assembly
        outer = openmc.Sphere(r=200.0, boundary_type="vacuum")
        cells += [openmc.Cell(name="void", region=-outer & ~slab)]
    geom = openmc.Geometry(openmc.Universe(cells=cells))

    sources, norm = make_sources(case, room=room or full_sphere, kinematic=kinematic)
    st = openmc.Settings()
    st.run_mode = "fixed source"
    st.source = sources
    st.particles = int(particles)
    st.batches = int(batches)
    st.seed = int(seed)
    st.photon_transport = False
    st.output = {"tallies": False}

    mats_ = openmc.Materials(mats)
    mats_.cross_sections = xs_xml

    # tallies: thin on-axis cylinders (r < 1.5 cm) on a zstep grid; off-axis ring (Li2O Li-glass)
    zlo, zhi = 18.0, (84.0 if is_li2o else 68.0)
    zg = np.round(np.arange(zlo, zhi + 1e-9, zstep), 6)
    if fine_faces:   # 1 mm bins within ~1 cm of both faces (steep thermal gradients at the surfaces)
        fine = np.concatenate([np.arange(z0 - 0.5, z0 + 1.0 + 1e-9, 0.1), np.arange(z1 - 1.0, z1 + 0.5 + 1e-9, 0.1)])
        zg = np.unique(np.round(np.concatenate([zg[(zg < z0 - 0.5) | ((zg > z0 + 1.0) & (zg < z1 - 1.0)) | (zg > z1 + 0.5)],
                                                fine]), 6))
    meshes = {"axis": openmc.CylindricalMesh(r_grid=[0.0, 1.5], z_grid=zg, name="axis")}
    if is_li2o:
        meshes["ring15"] = openmc.CylindricalMesh(r_grid=[14.7, 15.7], z_grid=zg, name="ring15")
    rids = (["li6_t", "li7_t", "u235_f", "u238_f", "np237_f", "th232_f", "al27_na", "ni58_n2n"] if is_li2o else
            ["li6_t", "u235_f", "au197_ng", "al27_na", "ni58_n2n", "ni58_np", "fe56_np", "zr90_n2n"])
    tallies = []
    for rid in rids:
        E, y = response_xs(rid, xs_xml)
        ef = openmc.EnergyFunctionFilter(E, y)
        for mname, mesh in meshes.items():
            if mname != "axis" and rid != "li6_t":
                continue
            t = openmc.Tally(name=f"{rid}@{mname}")
            t.filters = [openmc.MeshFilter(mesh), ef]
            t.scores = ["flux"]
            tallies.append(t)
    if energy_tally:
        eb = np.unique(np.concatenate([[1e-5, 0.625, 1e3, 1e5], np.geomspace(0.1e6, 20e6, 101)]))
        t = openmc.Tally(name="spectrum@axis2")
        mesh2 = openmc.CylindricalMesh(r_grid=[0.0, 1.5], z_grid=np.arange(18.0, zhi + 1e-9, 2.0), name="axis2")
        t.filters = [openmc.MeshFilter(mesh2), openmc.EnergyFilter(eb)]
        t.scores = ["flux"]
        tallies.append(t)
    model = openmc.model.Model(geometry=geom, materials=mats_, settings=st, tallies=openmc.Tallies(tallies))
    info = dict(case=case, room=room, kinematic=kinematic, norm=norm, full_sphere=full_sphere, room_air=room_air, fine_faces=fine_faces, zgrid=zg.tolist(), rids=rids,
                meshes={k: [float(m.r_grid[0]), float(m.r_grid[-1])] for k, m in meshes.items()},
                length=L, particles=particles, batches=batches, seed=seed, xs=xs_xml)
    return model, info


def extract(sp_path, info):
    """Per-mesh-bin rates [per atom per D-T neutron] with std, for every response tally."""
    zg = np.array(info["zgrid"])
    out = {"zc": (0.5 * (zg[1:] + zg[:-1])).tolist()}
    with openmc.StatePoint(sp_path) as sp:
        for t in sp.tallies.values():
            if "@" not in t.name:
                continue
            rid, mname = t.name.split("@")
            if rid == "spectrum":
                ef = t.filters[1].bins
                out["spectrum"] = dict(ebins=ef[:, 0].tolist() + [float(ef[-1, 1])],
                                       zgrid=t.filters[0].mesh.z_grid.tolist(),
                                       mean=t.mean.reshape(-1, len(ef)).tolist(),
                                       std=t.std_dev.reshape(-1, len(ef)).tolist())
                vol = np.pi * 1.5**2 * np.diff(np.array(t.filters[0].mesh.z_grid))
                out["spectrum"]["vol"] = vol.tolist()
                continue
            r0, r1 = info["meshes"][mname]
            vol = np.pi * (r1**2 - r0**2) * np.diff(zg)
            # OpenMC >= 0.15 already scales tallies by the total source strength (= per D-T
            # reaction here); sigma is in barns -> 1e-24 cm^2
            m = t.mean.ravel() * 1e-24 / vol
            s = t.std_dev.ravel() * 1e-24 / vol
            out[t.name] = dict(mean=m.tolist(), std=s.tolist())
    return out


def at_positions(res, key, positions):
    """Linear interpolation between mesh-bin centres; variance from the two weights.
    Never interpolates across the assembly faces (z0 = 20, z1 = 20 + L): a position inside the
    assembly (z0 <= p <= z1) only uses bins inside it, one outside only bins outside it; within half a
    bin of a face this means the nearest bin on the same side (constant extrapolation)."""
    zc = np.array(res["zc"]); m = np.array(res[key]["mean"]); s = np.array(res[key]["std"])
    z0, z1 = 20.0, 20.0 + res["info"]["length"]
    vals, sds = [], []
    for p in positions:
        ok = (zc >= z0) & (zc <= z1) if z0 <= p <= z1 else (zc < z0) | (zc > z1)
        if p < z0:
            ok &= zc < z0
        elif p > z1:
            ok &= zc > z1
        idx = np.nonzero(ok)[0]
        zz = zc[idx]
        if p <= zz[0] or p >= zz[-1]:
            k = idx[0] if p <= zz[0] else idx[-1]
            vals.append(m[k]); sds.append(s[k]); continue
        jj = int(np.searchsorted(zz, p) - 1)
        j = idx[jj]
        assert idx[jj + 1] == j + 1
        w = (p - zc[j]) / (zc[j + 1] - zc[j])
        vals.append((1 - w) * m[j] + w * m[j + 1])
        sds.append(math.sqrt(((1 - w) * s[j]) ** 2 + (w * s[j + 1]) ** 2))
    return np.array(vals), np.array(sds)

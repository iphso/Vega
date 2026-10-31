"""The LIBRA experiments (BABY 1L, FLiBe-1L) in a form the browser transport can run.

LIBRA's own model = crucible stack (axisymmetric about the crucible axis) + the
A-325 generator tube, lead bricks, supports and table + the MIT vault. The
browser tier takes only the axisymmetric stack in free space. This script builds
exactly that simplified model in OpenMC and writes everything the browser needs:

  1. CE TBR of the simplified model (what the browser must reproduce)
  2. per-cell 175-group hist64 cross sections from that run (+ tritium coefficients)
  3. an (R, Z) raster of cell ids about the crucible axis (outside the 50 cm sphere = escape)
  4. the A-325 source as angle bins (cosine to the generator axis) x group spectra

The chain to measurement is then:
  browser(simplified) vs OpenMC CE(simplified)       -- this transport's accuracy
  OpenMC(simplified) vs OpenMC(LIBRA full, vault)    -- what the simplification drops
  LIBRA full vs measured                             -- the experiment

Run: docker compose run --rm openmc blanket/libra_simple.py <repo> [particles]
  repo: BABY-1L-run-4 or FLiBe-1L-run-1
"""
import json, os, sys
import numpy as np
import openmc, openmc.mgxs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else "BABY-1L-run-4"
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
sys.path.insert(0, os.path.join(HERE, f"ext/{REPO}/analysis/neutron"))
from openmc_model import baby_geometry
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond
from libra_toolbox.neutronics import materials as LM

GS = "VITAMIN-J-175"
NMU = 64
THREADS = int(os.environ.get("OMP_NUM_THREADS", "24"))
d = os.path.join(HERE, "runs", "libra", REPO)
os.makedirs(d, exist_ok=True)
groups = openmc.mgxs.EnergyGroups(GS)

x_c, y_c, z_c = 587.0, 60.0, 100.0
sphere, breeder_cell, cells = baby_geometry(x_c, y_c, z_c)
sphere.boundary_type = "vacuum"
# the non-axisymmetric parts (generator tube wall, lead bricks, HDPE supports,
# table) become air; the generator's vacuum interior too
air = LM.Air
for c in cells:
    if c.fill is None or c.fill is LM.SS304 or c.fill is LM.Lead or c.fill is LM.HDPE:
        c.fill = air
# the table under the source is an epoxy slab at z < z_c - 28: also not part of the stack
for c in cells:
    bb = c.region.bounding_box
    if c.fill is LM.Epoxy and bb.upper_right[2] <= z_c - 28.0 + 1e-6:
        c.fill = air
# merge cells by material so the browser sees one region per material
mats = {}
for c in cells:
    mats.setdefault(c.fill.name, []).append(c)
names = {}
for m, cs in mats.items():
    for c in cs:
        names[c.id] = m
breeder = breeder_cell.fill.name
print("materials:", {m: len(cs) for m, cs in mats.items()}, "breeder:", breeder)

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = A325_generator_diamond((x_c, y_c, z_c - 5.635), (1, 0, 0))
settings.particles, settings.batches = NP, 40
settings.output = {"tallies": False}
geom = openmc.Geometry(cells)
model = openmc.Model(geom, openmc.Materials({c.fill for c in cells}), settings)

# domains: one per material (cells of a material share spectrum weighting; the
# breeder and the thin Inconel layers dominate and are single regions anyway)
domains = [mats[m] for m in mats]
lib = openmc.mgxs.Library(geom)
lib.energy_groups = groups
lib.mgxs_types = ["total", "absorption", "nu-scatter matrix", "multiplicity matrix"]
lib.domain_type = "material"
lib.domains = [cs[0].fill for cs in domains]
lib.correction = None
lib.scatter_format = "histogram"
lib.histogram_bins = NMU
lib.by_nuclide = False
lib.build_library()
model.tallies = openmc.Tallies()
lib.add_to_tallies_file(model.tallies, merge=True)
mf = openmc.MaterialFilter(lib.domains)
ef = openmc.EnergyFilter(groups.group_edges)
rr = openmc.Tally(name="rr"); rr.filters = [mf, ef]; rr.scores = ["flux", "(n,Xt)"]
model.tallies.append(rr)
tb = openmc.Tally(name="TBR"); tb.filters = [openmc.CellFilter([breeder_cell])]; tb.scores = ["(n,Xt)"]
model.tallies.append(tb)
sp_path = model.run(cwd=os.path.join(d, "ce"), threads=THREADS)

with openmc.StatePoint(sp_path) as sp:
    lib.load_from_statepoint(sp)
    t = sp.get_tally(name="TBR").get_pandas_dataframe()
    tbr, tbr_sd = float(t["mean"].iloc[0]), float(t["std. dev."].iloc[0])
    r = sp.get_tally(name="rr")
    flux = r.get_values(scores=["flux"]).reshape(len(domains), -1)[:, ::-1]
    trit = r.get_values(scores=["(n,Xt)"]).reshape(len(domains), -1)[:, ::-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sig_t = np.where(flux > 0, trit / flux, 0.0)
mat_names = [cs[0].fill.name for cs in domains]
mglib = lib.create_mg_library(xs_type="macro", xsdata_names=[f"cell_{n}" for n in mat_names])
mglib.export_to_hdf5(os.path.join(d, "mgxs.h5"))
np.savez(os.path.join(d, "coeffs.npz"), sig_tritium=sig_t, kerma=np.zeros_like(sig_t),
         ce_flux=flux, cells=np.array(mat_names), group_edges=groups.group_edges)

# ---- (R, Z) raster of material ids about the crucible axis
H = 0.05
Rmax, Zlo, Zhi = 50.0, z_c - 50.0, z_c + 50.0
nr, nz = int(Rmax / H), int((Zhi - Zlo) / H)
# one slice through the axis (C++ id_map: 2M points in one call); the stack is
# axisymmetric and the dropped parts are air at every angle, so the +y half-plane is it
model.export_to_model_xml(os.path.join(d, "ce", "model.xml"))
idm = model.id_map(origin=(x_c, y_c + Rmax / 2, z_c), width=(Rmax, Zhi - Zlo), pixels=(nr, nz), basis="yz")
mat_ids = idm[:, :, 2][::-1, :]          # rows top->bottom -> z ascending
id2name = {m.id: m.name for m in model.materials}
idx = {m: i for i, m in enumerate(mat_names)}
ids = np.full((nz, nr), 255, dtype=np.uint8)
for mid, name in id2name.items():
    if name in idx:
        ids[mat_ids == mid] = idx[name]
Rg = (np.arange(nr) + 0.5) * H
Zg = Zlo + (np.arange(nz) + 0.5) * H - z_c
ids[(Rg[None, :] ** 2 + Zg[:, None] ** 2) >= 50.0 ** 2] = 255
print("raster cells per material:", {m: int((ids == k).sum()) for m, k in idx.items()})
ids.tofile(os.path.join(d, "raster.u8"))

# ---- A-325 source as angle bins x group CDFs (relative to the generator axis +x)
srcs = A325_generator_diamond((x_c, y_c, z_c - 5.635), (1, 0, 0))
edges = groups.group_edges
bins = []
for s in srcs:
    mu = s.angle.mu
    e, p = np.asarray(s.energy.x), np.asarray(s.energy.p)
    # linear-linear tabular -> group integrals
    gp = np.zeros(len(edges) - 1)
    fine = np.linspace(e.min(), e.max(), 4000)
    pf = np.interp(fine, e, p)
    gi = np.clip(np.searchsorted(edges, fine) - 1, 0, len(edges) - 2)
    np.add.at(gp, gi, pf)
    gp = gp[::-1] / gp.sum()           # group 0 = highest energy
    bins.append({"mu_lo": float(mu.a), "mu_hi": float(mu.b), "strength": float(s.strength), "group_p": gp.tolist()})
src = {"position_rz": [0.0, -5.635], "axis": [1, 0, 0], "bins": bins}

meta = {"repo": REPO, "breeder": breeder, "materials": mat_names, "ce_TBR": [tbr, tbr_sd],
        "raster": {"h": H, "nr": nr, "nz": nz, "z0": -50.0, "file": "raster.u8", "outside": 255,
                   "note": "z relative to the crucible stack's z_c; R about the crucible axis"},
        "source": src, "particles": NP * 40}
json.dump(meta, open(os.path.join(d, "libra.json"), "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("repo", "breeder", "materials", "ce_TBR")}, indent=1))

"""Group constants for the deep-shield materials in EVERY group, to fill the groups the design
runs (mgxs_gen.py) leave empty.

The ARC run's tank wall, shield and coil cells see no neutrons in some groups (the coil: 54 of
175, 45 of them above 0.1 MeV; tank and shield: the 7 above 14 MeV), so their tallied cross
sections there are zero and the transport would let a neutron in such a group stream through
them unattenuated (the TF-face fluence is up to ~1.5x high that way). Here each material sits in
its own 50 cm sphere around an isotropic point source flat in lethargy (1e-5 eV .. 20 MeV), so
every group is populated; export_neutpack.py takes a cell's constants from here only in the
groups where its own design run had no flux.

Run: docker compose run --rm openmc blanket/fill_mgxs.py [GROUPS] [FORMAT]   (VITAMIN-J-175 hist64)
"""
import os, sys
import numpy as np
import openmc, openmc.mgxs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_model import materials, DEFAULTS, HERE

GS = sys.argv[1] if len(sys.argv) > 1 else "VITAMIN-J-175"
FMT = sys.argv[2] if len(sys.argv) > 2 else "hist64"
HIST = int(FMT[4:]) if FMT.startswith("hist") else 0
LEG = int(FMT[3:]) if FMT.startswith("leg") else 0
THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
d = os.path.join(HERE, "runs", "mgxs", f"fill_{GS}_{FMT}")
os.makedirs(d, exist_ok=True)

mats = materials(dict(DEFAULTS))
ROLE = {"tank": mats["inconel"], "shield": mats["TiH2"], "coil": mats["coil"]}
# ~flat in lethargy: equal weight per decade, uniform within it
EDGES = [1e-5 * 10 ** k for k in range(12)] + [2e7]
FLAT_LETHARGY = openmc.stats.Mixture([float(np.log10(b / a)) for a, b in zip(EDGES[:-1], EDGES[1:])],
                                     [openmc.stats.Uniform(a, b) for a, b in zip(EDGES[:-1], EDGES[1:])])
cells, srcs = {}, []
for k, (name, mat) in enumerate(ROLE.items()):
    x0 = 400.0 * k
    s = openmc.Sphere(x0=x0, r=50.0)
    cells[name] = openmc.Cell(name=name, fill=mat, region=-s)
    srcs.append(openmc.IndependentSource(space=openmc.stats.Point((x0, 0, 0)), angle=openmc.stats.Isotropic(),
                                         energy=FLAT_LETHARGY, strength=1.0))
outer = openmc.Sphere(r=2000.0, boundary_type="vacuum")
void = openmc.Cell(name="void", region=-outer & ~cells["tank"].region & ~cells["shield"].region & ~cells["coil"].region)
geom = openmc.Geometry(list(cells.values()) + [void])
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.particles, settings.batches = 100_000, 30
settings.source = srcs
settings.photon_transport = False
settings.output = {"tallies": False}

groups = openmc.mgxs.EnergyGroups(GS)
domains = list(cells.values())
lib = openmc.mgxs.Library(geom)
lib.energy_groups = groups
lib.mgxs_types = ["total", "absorption", "nu-scatter matrix", "multiplicity matrix"]
lib.domain_type = "cell"
lib.domains = domains
lib.correction = None
if HIST:
    lib.scatter_format, lib.histogram_bins = "histogram", HIST
else:
    lib.scatter_format, lib.legendre_order = "legendre", LEG
lib.by_nuclide = False
lib.build_library()
tallies = openmc.Tallies()
lib.add_to_tallies_file(tallies, merge=True)
rr = openmc.Tally(name="rr")
rr.filters = [openmc.CellFilter(domains), openmc.EnergyFilter(groups.group_edges)]
rr.scores = ["flux", "(n,Xt)", "heating-local"]
tallies.append(rr)
model = openmc.Model(geom, openmc.Materials(set(ROLE.values())), settings, tallies)
sp_path = model.run(cwd=os.path.join(d, "ce"), threads=THREADS)
with openmc.StatePoint(sp_path) as sp:
    lib.load_from_statepoint(sp)
    t = sp.get_tally(name="rr")
    flux = t.get_values(scores=["flux"]).reshape(len(domains), -1)[:, ::-1]
    trit = t.get_values(scores=["(n,Xt)"]).reshape(len(domains), -1)[:, ::-1]
    heat = t.get_values(scores=["heating-local"]).reshape(len(domains), -1)[:, ::-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sig_t = np.where(flux > 0, trit / flux, 0.0)
    kerma = np.where(flux > 0, heat / flux, 0.0)
names = [c.name for c in domains]
mglib = lib.create_mg_library(xs_type="macro", xsdata_names=[f"cell_{n}" for n in names])
mglib.export_to_hdf5(os.path.join(d, "mgxs.h5"))
np.savez(os.path.join(d, "coeffs.npz"), sig_tritium=sig_t, kerma=kerma, ce_flux=flux, cells=np.array(names), group_edges=groups.group_edges)
print("empty groups per cell:", {n: int((flux[i] <= 0).sum()) for i, n in enumerate(names)})
print("wrote", d)

"""Gate 1: how much does collapsing to a coarse group structure cost?

1. Continuous-energy (CE) ARC baseline -> per-cell macroscopic MGXS
   (total, absorption, P3 nu-scatter + multiplicity) plus per-cell group
   tritium-production and local-heating coefficients from CE reaction-rate /
   flux ratios.
2. Rerun the identical geometry in OpenMC multigroup mode; TBR and heating are
   group flux x the CE-derived coefficients.
3. Compare. The difference is the group-collapse error alone (same code, same
   geometry), the floor any browser multigroup tier inherits.

Run: docker compose run --rm openmc blanket/mgxs_gen.py [GROUPS] [FORMAT] [key=value ...]
  (default VITAMIN-J-175 leg3; overrides as in arc_model.DEFAULTS, e.g. fw_material=Be li6_pct=30)

NOT VITAMIN-J-42: that is the VITAMIN-J *photon* structure (1 keV-50 MeV); every
neutron below 1 keV falls outside it (a first attempt lost 6% of TBR that way).
"""
import json, os, sys
import numpy as np
import openmc, openmc.mgxs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_model import build, HERE, NEUTRONS_PER_S

GS = sys.argv[1] if len(sys.argv) > 1 else "VITAMIN-J-175"
# angular format: "leg3" (P3 Legendre) or "hist64" (64-bin cosine histogram)
FMT = sys.argv[2] if len(sys.argv) > 2 else "leg3"
LEG = int(FMT[3:]) if FMT.startswith("leg") else 0
HIST = int(FMT[4:]) if FMT.startswith("hist") else 0
# design overrides as key=value (same keys as arc_model.DEFAULTS); they tag the run
OVER = {}
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    OVER[k] = v if k == "fw_material" else float(v)
TAG = "".join(f"_{k}-{v}" for k, v in OVER.items())
THREADS = int(os.environ.get("OMP_NUM_THREADS", "24"))
d = os.path.join(HERE, "runs", "mgxs", f"{GS}_{FMT}{TAG}")
os.makedirs(d, exist_ok=True)
groups = openmc.mgxs.EnergyGroups(GS)
EV_TO_MW = NEUTRONS_PER_S * 1.602176634e-19 / 1e6

# ---- 1. CE run + library -------------------------------------------------
model, cells, p = build(OVER or None, particles=40000, batches=50, photons=False)
domains = [c for k, c in cells.items() if c.fill is not None]
names = {c.id: k for k, c in cells.items()}
lib = openmc.mgxs.Library(model.geometry)
lib.energy_groups = groups
lib.mgxs_types = ["total", "absorption", "nu-scatter matrix", "multiplicity matrix"]
lib.domain_type = "cell"
lib.domains = domains
lib.correction = None
if HIST:
    lib.scatter_format = "histogram"
    lib.histogram_bins = HIST
else:
    lib.scatter_format = "legendre"
    lib.legendre_order = LEG
lib.by_nuclide = False
lib.build_library()
model.tallies = openmc.Tallies()
lib.add_to_tallies_file(model.tallies, merge=True)
ef = openmc.EnergyFilter(groups.group_edges)
cf = openmc.CellFilter(domains)
rr = openmc.Tally(name="rr"); rr.filters = [cf, ef]; rr.scores = ["flux", "(n,Xt)", "heating-local"]
model.tallies.append(rr)
sp_path = model.run(cwd=os.path.join(d, "ce"), threads=THREADS)

with openmc.StatePoint(sp_path) as sp:
    lib.load_from_statepoint(sp)
    t = sp.get_tally(name="rr")
    flux = t.get_values(scores=["flux"]).reshape(len(domains), -1)
    trit = t.get_values(scores=["(n,Xt)"]).reshape(len(domains), -1)
    heat = t.get_values(scores=["heating-local"]).reshape(len(domains), -1)
# OpenMC energy filter bins are low->high; MGXS group 1 is the highest energy
flux, trit, heat = flux[:, ::-1], trit[:, ::-1], heat[:, ::-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sig_t = np.where(flux > 0, trit / flux, 0.0)      # tritium production, 1/cm
    kerma = np.where(flux > 0, heat / flux, 0.0)      # eV/cm
ce = {
    "TBR_by_cell": {names[c.id]: float(trit[i].sum()) for i, c in enumerate(domains)},
    "TBR": float(trit[[i for i, c in enumerate(domains) if names[c.id] in ("channel", "blanket")]].sum()),
    "heating_MW": {names[c.id]: float(heat[i].sum() * EV_TO_MW) for i, c in enumerate(domains)},
}

mglib = lib.create_mg_library(xs_type="macro", xsdata_names=[f"cell_{names[c.id]}" for c in domains])
mglib.export_to_hdf5(os.path.join(d, "mgxs.h5"))
np.savez(os.path.join(d, "coeffs.npz"), sig_tritium=sig_t, kerma=kerma, ce_flux=flux,
         cells=np.array([names[c.id] for c in domains]), group_edges=groups.group_edges)

# ---- 2. MG run on the identical geometry ---------------------------------
mats = []
for c in domains:
    m = openmc.Material(name=f"mg_{names[c.id]}")
    m.set_density("macro", 1.0)
    m.add_macroscopic(f"cell_{names[c.id]}")
    mats.append(m)
    c.fill = m
mg = openmc.Model(model.geometry, openmc.Materials(mats), model.settings)
mg.materials.cross_sections = os.path.join(d, "mgxs.h5")
mg.settings.energy_mode = "multi-group"
mg.settings.particles = 40000
mg.settings.batches = 50
# MG mode samples source energy into the group containing 14.08 MeV
mft = openmc.Tally(name="mgflux")
mft.filters = [openmc.CellFilter(domains), openmc.EnergyFilter(groups.group_edges)]
mft.scores = ["flux"]
mg.tallies = openmc.Tallies([mft])
sp_path = mg.run(cwd=os.path.join(d, "mg"), threads=THREADS)
with openmc.StatePoint(sp_path) as sp:
    mgflux = sp.get_tally(name="mgflux").get_values(scores=["flux"]).reshape(len(domains), -1)[:, ::-1]
np.save(os.path.join(d, "mg_flux.npy"), mgflux)
mg_res = {
    "TBR": float(sum((mgflux[i] * sig_t[i]).sum() for i, c in enumerate(domains) if names[c.id] in ("channel", "blanket"))),
    "TBR_by_cell": {names[c.id]: float((mgflux[i] * sig_t[i]).sum()) for i, c in enumerate(domains)},
    "heating_MW": {names[c.id]: float((mgflux[i] * kerma[i]).sum() * EV_TO_MW) for i, c in enumerate(domains)},
}

out = {"groups": GS, "format": FMT, "design": OVER, "ce": ce, "mg": mg_res,
       "TBR_rel_err": mg_res["TBR"] / ce["TBR"] - 1,
       "heating_rel_err": {k: mg_res["heating_MW"][k] / v - 1 if v else None for k, v in ce["heating_MW"].items()}}
json.dump(out, open(os.path.join(d, "gate1.json"), "w"), indent=2)
print(json.dumps(out, indent=2))

"""Probe OpenMC 0.16 photon tallies on a small slab problem before building the
multigroup photon library (photon_gen.py): which scores/filters work for photons,
where secondary photons are born (neutron, photon, electron, positron events), and
whether production x energy closes the neutron heating-local - heating gap.

Run: docker compose run --rm -v ~/runs/blanket/photon:/runs openmc blanket/photon/probe.py [material]
"""
import json, os, sys
import numpy as np
import openmc, openmc.mgxs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arc_model import materials, DEFAULTS

# 0.16's Filter.__eq__ runs np.allclose over bins, which fails on ParticleProductionFilter's
# (particle, lo, hi) tuples when two of them meet in Tallies.to_xml_element: compare by identity
openmc.ParticleProductionFilter.__eq__ = lambda a, b: a is b
openmc.ParticleProductionFilter.__hash__ = lambda a: id(a)

MAT = sys.argv[1] if len(sys.argv) > 1 else "inconel"
mats = materials(dict(DEFAULTS))
m = mats[MAT]
PG = openmc.mgxs.EnergyGroups("VITAMIN-J-42").group_edges
NG = np.logspace(-5, np.log10(2e7), 12)
s = openmc.Sphere(r=30, boundary_type="vacuum")
c = openmc.Cell(fill=m, region=-s)
geom = openmc.Geometry([c])
st = openmc.Settings(run_mode="fixed source", particles=20000, batches=10, photon_transport=True)
src = openmc.IndependentSource(space=openmc.stats.Point(), angle=openmc.stats.Isotropic(), energy=openmc.stats.Discrete([14.08e6], [1]))
st.source = src
cf = openmc.CellFilter([c])
P = lambda *p: openmc.ParticleFilter(list(p))
T = []
def tally(name, filters, scores, est=None):
    t = openmc.Tally(name=name); t.filters = filters; t.scores = scores
    if est: t.estimator = est
    T.append(t)
tally("n", [cf, P("neutron"), openmc.EnergyFilter(NG)], ["flux", "heating", "heating-local"])
tally("nprod", [cf, P("neutron"), openmc.EnergyFilter(NG), openmc.ParticleProductionFilter("photon", PG)], ["events"], "analog")
tally("g", [cf, P("photon"), openmc.EnergyFilter(PG)], ["flux", "total", "heating"])
tally("gsc", [cf, P("photon"), openmc.EnergyFilter(PG), openmc.EnergyoutFilter(PG)], ["scatter"], "analog")
tally("gprod", [cf, P("photon"), openmc.EnergyFilter(PG), openmc.ParticleProductionFilter("photon", PG)], ["events"], "analog")
tally("eprod", [cf, P("electron", "positron"), openmc.ParticleProductionFilter("photon", PG)], ["events"], "analog")
tally("heat", [cf, P("neutron", "photon", "electron", "positron")], ["heating"])
tally("heatall", [cf], ["heating"])
tally("grx", [cf, P("photon"), openmc.EnergyFilter(PG)], ["coherent-scatter", "incoherent-scatter", "photoelectric", "pair-production"])
model = openmc.Model(geom, openmc.Materials([m]), st, openmc.Tallies(T))
d = f"/runs/probe_{MAT}"
sp = openmc.StatePoint(model.run(cwd=d, threads=int(os.environ.get("OMP_NUM_THREADS", "8"))))
def g(n, sc=None):
    t = sp.get_tally(name=n)
    k = t.scores.index(sc) if sc else 0
    return t.mean[:, 0, k]
nf, nh, nhl = g("n", "flux"), g("n", "heating"), g("n", "heating-local")
mid = np.sqrt(PG[:-1] * PG[1:])
prod = g("nprod").reshape(len(NG) - 1, len(PG) - 1)
print("neutron heating", nh.sum(), "heating-local", nhl.sum(), "diff", nhl.sum() - nh.sum())
print("n->photon count", prod.sum(), "energy(mid)", (prod * mid).sum())
gf, gt, gh = g("g", "flux"), g("g", "total"), g("g", "heating")
gsc = g("gsc").reshape(len(PG) - 1, len(PG) - 1)
gpr = g("gprod").reshape(len(PG) - 1, len(PG) - 1)
epr2 = g("eprod").reshape(2, len(PG) - 1)
print("photons from electrons", epr2[0].sum(), "E", (epr2[0] * mid).sum(), "from positrons", epr2[1].sum(), "E", (epr2[1] * mid).sum())
epr = epr2.sum(0)
print("photon collisions", gt.sum(), "scatter", gsc.sum(), "photon->photon prod", gpr.sum(), "e+- -> photon", epr.sum(), "E", (epr * mid).sum(), "511 grp share", epr[np.searchsorted(PG, 511e3) - 1] / max(epr.sum(), 1e-30))
print("heating by particle", g("heat"), "total", g("heatall"), "photon heating(tl)", gh.sum())
rx = sp.get_tally(name="grx").mean[:, 0, :]
print("photon rx sums coh/incoh/pe/pair", rx.sum(0))
np.savez(f"{d}/probe.npz", prod=prod, gsc=gsc, gpr=gpr, epr=epr, gf=gf, gt=gt, gh=gh, rx=rx, nf=nf, nh=nh, nhl=nhl)

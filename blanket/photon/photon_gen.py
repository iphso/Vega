"""Multigroup photon data for coupled neutron-gamma transport in the browser (StoneLab
packages/physics/src/neutron, photon extension of the neutpack; export_photon.py).

OpenMC 0.16's openmc.mgxs is neutron-only, so everything here is built from raw tallies on a
continuous-energy (CE) coupled n-gamma run of the design (arc_model.build, photons on), as
ratios to the tallied flux of the same run, group by group:

  neutron side (VITAMIN-J-175, per cell):
    kerma_n[g]        'heating' with ParticleFilter(neutron) / neutron flux: the NJOY MT301
                      KERMA, which leaves out the photon energy ('heating-local' = MT901 puts it
                      back as local deposition). neutron + photon heating = coupled total.
    prod[g, h]        photons born into photon group h per unit neutron flux in group g:
                      ParticleProductionFilter('photon', photon groups) x 'events' (analog,
                      the filter weights by the secondary's weight), 1/cm
  photon side (VITAMIN-J-42, per cell):
    sigma_t[h]        'total' / photon flux
    kerma[h]          'heating' with ParticleFilter(photon) / photon flux: OpenMC's photon
                      heating (MT525 heating number; with the default thick-target
                      bremsstrahlung electron treatment every e-/e+ deposits at the photon's
                      collision -- probe.py: electron and positron events score nothing)
    scat[h, h', mu]   analog 'scatter' (coherent + incoherent) x Energyout x Mu (64 bins)
    sec[h, h']        photons born at photon collisions (fluorescence, TTB bremsstrahlung,
                      annihilation pairs -- all attributed to the photon event under TTB),
                      ParticleProductionFilter x 'events'; emitted isotropically in transport
  Cells whose group is under-sampled in the design run (relative std of the group flux
  > REL_MAX: the deep tank wall / shield / coil, and photon groups few photons reach) take that
  group's constants from a fill run: each material alone in a 50 cm sphere around point
  sources flat in lethargy, neutrons 1e-5 eV..20 MeV and photons 1 keV..20 MeV (as
  fill_mgxs.py does for the neutron constants).

Photon groups: VITAMIN-J-42 (1 keV .. 50 MeV), the photon half of the VITAMIN-J / FENDL
fusion shielding libraries (the same family as the neutron VITAMIN-J-175), ~0.1-0.15 lethargy
wide above 1 MeV where capture and inelastic gammas live, with edges at 0.45/0.51 MeV around
the annihilation line and through the W K-edge (69.5 keV) region; 42 x 42 transfers keep the
GPU tables small.

Run: docker compose run --rm -v ~/runs/blanket/photon:/runs openmc blanket/photon/photon_gen.py \
         [key=value ...]          (arc_model.DEFAULTS overrides, e.g. vv_in_cm=4 be_cm=0)
     env PARTICLES, BATCHES (default 40000 x 50), FILL_PARTICLES (default 50000 x 20)
Output: /runs/VITAMIN-J-175_VITAMIN-J-42<TAG>/photon.npz + result.json (CE coupled heating by
cell, split neutron / photon, and TBR), consumed by export_photon.py.
"""
import json, os, sys
import numpy as np
import openmc, openmc.mgxs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from arc_model import build, NEUTRONS_PER_S

# 0.16's Filter.__eq__ runs np.allclose over bins, which fails on ParticleProductionFilter's
# (particle, lo, hi) tuples when two of them meet in Tallies.to_xml_element: compare by identity
openmc.ParticleProductionFilter.__eq__ = lambda a, b: a is b
openmc.ParticleProductionFilter.__hash__ = lambda a: id(a)

NGS, PGS = "VITAMIN-J-175", "VITAMIN-J-42"
NMU = 64
REL_MAX = 0.05
OVER = {}
for kv in sys.argv[1:]:
    k, v = kv.split("=")
    OVER[k] = v if k in ("fw_material", "breeder", "layout") else float(v)
TAG = "".join(f"_{k}-{v}" for k, v in OVER.items())
THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
OUT = os.environ.get("PHOTON_RUNS", "/runs")
d = os.path.join(OUT, f"{NGS}_{PGS}{TAG}")
os.makedirs(d, exist_ok=True)
NE = openmc.mgxs.EnergyGroups(NGS).group_edges
PE = openmc.mgxs.EnergyGroups(PGS).group_edges
G, P = len(NE) - 1, len(PE) - 1
EV_TO_MW = NEUTRONS_PER_S * 1.602176634e-19 / 1e6


def photon_tallies(domains):
    cf = openmc.CellFilter(domains)
    pn, pg = openmc.ParticleFilter(["neutron"]), openmc.ParticleFilter(["photon"])
    ne, pe = openmc.EnergyFilter(NE), openmc.EnergyFilter(PE)
    T = []

    def tally(name, filters, scores, est=None):
        t = openmc.Tally(name=name); t.filters = filters; t.scores = scores
        if est: t.estimator = est
        T.append(t)
    tally("n", [cf, pn, ne], ["flux", "heating", "heating-local", "(n,Xt)"])
    tally("nprod", [cf, pn, ne, openmc.ParticleProductionFilter("photon", PE)], ["events"], "analog")
    tally("g", [cf, pg, pe], ["flux", "total", "heating"])
    tally("gsc", [cf, pg, pe, openmc.EnergyoutFilter(PE), openmc.MuFilter(NMU)], ["scatter"], "analog")
    tally("gsec", [cf, pg, pe, openmc.ParticleProductionFilter("photon", PE)], ["events"], "analog")
    tally("heat_split", [cf, openmc.ParticleFilter(["neutron", "photon", "electron", "positron"])], ["heating"])
    return T


def read(sp, nd):
    """Per-domain group constants as ratios to the same run's flux; rel. std of the fluxes."""
    def t(name):
        x = sp.get_tally(name=name)
        return x.mean[:, 0, :], x.std_dev[:, 0, :], x.scores
    out = {}
    m, s, sc = t("n")
    m = m.reshape(nd, G, -1); s = s.reshape(nd, G, -1)
    out["nflux"] = m[:, :, sc.index("flux")]
    out["nflux_rel"] = np.where(out["nflux"] > 0, s[:, :, sc.index("flux")] / np.maximum(out["nflux"], 1e-300), np.inf)
    out["nheat"] = m[:, :, sc.index("heating")]
    out["nheat_local"] = m[:, :, sc.index("heating-local")]
    out["ntrit"] = m[:, :, sc.index("(n,Xt)")]
    out["nprod"] = t("nprod")[0].reshape(nd, G, P)
    m, s, sc = t("g")
    m = m.reshape(nd, P, -1); s = s.reshape(nd, P, -1)
    out["gflux"] = m[:, :, 0]
    out["gflux_rel"] = np.where(out["gflux"] > 0, s[:, :, 0] / np.maximum(out["gflux"], 1e-300), np.inf)
    out["gtot"] = m[:, :, 1]
    out["gheat"] = m[:, :, 2]
    out["gsc"] = t("gsc")[0].reshape(nd, P, P, NMU)
    out["gsec"] = t("gsec")[0].reshape(nd, P, P)
    hs = sp.get_tally(name="heat_split")
    out["heat_split"] = hs.mean[:, 0, 0].reshape(nd, 4)
    out["heat_split_sd"] = hs.std_dev[:, 0, 0].reshape(nd, 4)
    # energy filters are low -> high; groups here run high -> low (group 0 = highest energy)
    for k in ("nflux", "nflux_rel", "nheat", "nheat_local", "ntrit"):
        out[k] = out[k][:, ::-1]
    out["nprod"] = out["nprod"][:, ::-1, ::-1]
    for k in ("gflux", "gflux_rel", "gtot", "gheat"):
        out[k] = out[k][:, ::-1]
    out["gsc"] = out["gsc"][:, ::-1, ::-1, :]
    out["gsec"] = out["gsec"][:, ::-1, ::-1]
    return out


def ratios(r):
    """Group constants: tallies / flux (zero where the flux is)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        nf, gf = r["nflux"], r["gflux"]
        nn = np.where(nf > 0, 1 / nf, 0.0)
        gg = np.where(gf > 0, 1 / gf, 0.0)
        return dict(
            kerma_n=r["nheat"] * nn, kerma_local=r["nheat_local"] * nn, prod=r["nprod"] * nn[:, :, None],
            p_sigma_t=r["gtot"] * gg, p_kerma=r["gheat"] * gg,
            p_scat=r["gsc"] * gg[:, :, None, None], p_sec=r["gsec"] * gg[:, :, None],
        )


# ---- 1. the design, coupled n-gamma CE --------------------------------------------------
NP = int(os.environ.get("PARTICLES", "40000")); NB = int(os.environ.get("BATCHES", "50"))
model, cells, p = build(OVER or None, particles=NP, batches=NB, photons=True)
domains = [c for k, c in cells.items() if c.fill is not None]
names = [next(k for k, c in cells.items() if c is dom) for dom in domains]
model.tallies.extend(photon_tallies(domains))
# REUSE=1: take the design run's statepoint from a previous run of the same design (the fill is
# always rerun)
prev = os.path.join(d, "ce", f"statepoint.{NB}.h5")
sp_path = prev if os.environ.get("REUSE") == "1" and os.path.exists(prev) else model.run(cwd=os.path.join(d, "ce"), threads=THREADS)
with openmc.StatePoint(sp_path) as sp:
    des = read(sp, len(domains))
    heat = sp.get_tally(name="heating")
    hcells = [int(b) for b in heat.filters[0].bins]
    coupled = {names[[dd.id for dd in domains].index(cid)]: (float(heat.mean[i, 0, 0]) * EV_TO_MW, float(heat.std_dev[i, 0, 0]) * EV_TO_MW)
               for i, cid in enumerate(hcells)}
    tb = sp.get_tally(name="TBR")
    tbr = (float(tb.mean.sum()), float(np.sqrt((tb.std_dev ** 2).sum())))
    runtime = float(sp.runtime["total"])
cr = ratios(des)

# ---- 2. fill: each material alone in a sphere, flat-lethargy neutron + photon sources ------
mats = {}
for n, dom in zip(names, domains):
    mats.setdefault(dom.fill.name, (dom.fill, []))[1].append(n)
NEDGES = [1e-5 * 10 ** k for k in range(12)] + [2e7]
# photons up to the structure's top edge: capture of a 14 MeV neutron emits up to ~20 MeV
# (W: 14 + S_n ~ 21 MeV), so the 20-30 MeV group is populated and needs constants everywhere
# (a fill to 20 MeV left it at sigma_t = 0: such photons streamed from the VV to the coil)
PEDGES = [1e3 * 10 ** k for k in range(5)] + [float(PE[-1])]
flat = lambda E: openmc.stats.Mixture([float(np.log10(b / a)) for a, b in zip(E[:-1], E[1:])],
                                      [openmc.stats.Uniform(a, b) for a, b in zip(E[:-1], E[1:])])
fcells, srcs = [], []
for k, (mname, (mat, _)) in enumerate(mats.items()):
    x0 = 400.0 * k
    c = openmc.Cell(name=f"fill_{mname}", fill=mat, region=-openmc.Sphere(x0=x0, r=50.0))
    fcells.append(c)
    for part, E in (("neutron", NEDGES), ("photon", PEDGES)):
        srcs.append(openmc.IndependentSource(space=openmc.stats.Point((x0, 0, 0)), angle=openmc.stats.Isotropic(),
                                             energy=flat(E), strength=1.0, particle=part))
outer = openmc.Sphere(r=400.0 * len(fcells) + 500, boundary_type="vacuum")
reg = -outer
for c in fcells:
    reg = reg & ~c.region
geom = openmc.Geometry(fcells + [openmc.Cell(name="void", region=reg)])
st = openmc.Settings(run_mode="fixed source", photon_transport=True, source=srcs,
                     particles=int(os.environ.get("FILL_PARTICLES", "50000")), batches=20)
st.output = {"tallies": False}
fm = openmc.Model(geom, openmc.Materials([m for m, _ in mats.values()]), st, openmc.Tallies(photon_tallies(fcells)))
with openmc.StatePoint(fm.run(cwd=os.path.join(d, "fill"), threads=THREADS)) as sp:
    fil = read(sp, len(fcells))
fr = ratios(fil)

# ---- 3. merge: design constants where the design sampled the group, fill elsewhere -------
fidx = {n: list(mats).index(cells[n].fill.name) for n in names}
nfilled, pfilled = {}, {}
for i, n in enumerate(names):
    j = fidx[n]
    bad_n = des["nflux_rel"][i] > REL_MAX
    bad_p = des["gflux_rel"][i] > REL_MAX
    for k in ("kerma_n", "kerma_local"):
        cr[k][i, bad_n] = fr[k][j, bad_n]
    cr["prod"][i, bad_n] = fr["prod"][j, bad_n]
    for k in ("p_sigma_t", "p_kerma"):
        cr[k][i, bad_p] = fr[k][j, bad_p]
    cr["p_scat"][i, bad_p] = fr["p_scat"][j, bad_p]
    cr["p_sec"][i, bad_p] = fr["p_sec"][j, bad_p]
    nfilled[n], pfilled[n] = int(bad_n.sum()), int(bad_p.sum())

np.savez_compressed(os.path.join(d, "photon.npz"), cells=np.array(names), neutron_edges=NE, photon_edges=PE, nmu=NMU,
                    **cr, ce_nflux=des["nflux"], ce_gflux=des["gflux"], ce_nheat_local=des["nheat_local"])
mid = np.sqrt(PE[:-1] * PE[1:])[::-1]
res = {
    "design": OVER, "particles": NP * NB, "runtime_s": runtime, "TBR": tbr,
    "heating_MW": {n: {"mean": v[0], "std": v[1]} for n, v in coupled.items()},
    "heating_split_MW": {n: {"neutron": float(des["heat_split"][i, 0] * EV_TO_MW), "photon": float(des["heat_split"][i, 1] * EV_TO_MW),
                             "electron": float(des["heat_split"][i, 2] * EV_TO_MW), "positron": float(des["heat_split"][i, 3] * EV_TO_MW),
                             "neutron_std": float(des["heat_split_sd"][i, 0] * EV_TO_MW), "photon_std": float(des["heat_split_sd"][i, 1] * EV_TO_MW)}
                         for i, n in enumerate(names)},
    "local_heating_MW": {n: float(des["nheat_local"][i].sum() * EV_TO_MW) for i, n in enumerate(names)},
    # photon energy born per source neutron (group-midpoint energies) vs heating-local - heating
    "photon_energy_check_MeV": {n: [float((des["nprod"][i] * mid[None, :]).sum() / 1e6), float((des["nheat_local"][i] - des["nheat"][i]).sum() / 1e6)]
                                for i, n in enumerate(names)},
    "groups_filled": {"neutron": nfilled, "photon": pfilled},
}
json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

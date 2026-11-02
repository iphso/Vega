"""Checks the contact-dose proxy (FISPACT-II's semi-infinite-slab formula, B = 2) against
an OpenMC photon transport of the same situation: a thick slab of the material with a
uniform isotropic photon source, air-kerma rate at its surface from the tallied photon
fluence (track length in a thin void layer on the surface, central patch), with the NIST
air mass energy-absorption coefficient:
    D [Gy/h] = 3600 * 1.602e-13 * 1e3 * sum_E phi(E) E[MeV] (mu_en/rho)_air(E)
(Gy of air ~ Sv for this proxy, as FISPACT treats it). Source normalised to 1 photon / kg / s.

Run: docker compose run --rm openmc blanket/activation/dose_check.py
Writes runs/dose_check.json.
"""
import json, os
import numpy as np
import openmc, openmc.data
from common import ACT, DOSE_C, DOSE_B, mu_rho_material
from arc_model import materials, DEFAULTS

HALF = 60.0      # slab half-thickness, cm (>> mean free path of a 2 MeV photon in any of these)
LAT = 300.0      # lateral half-width, cm
PATCH = 30.0     # central surface patch, cm
GAP = 0.1        # void tally layer thickness, cm


def mc_dose(mat, E_eV, particles=200000, batches=10):
    top = openmc.ZPlane(z0=0.0)
    bot = openmc.ZPlane(z0=-2 * HALF)
    sides = [openmc.XPlane(-LAT), openmc.XPlane(LAT), openmc.YPlane(-LAT), openmc.YPlane(LAT)]
    box = +sides[0] & -sides[1] & +sides[2] & -sides[3]
    patch = (+openmc.XPlane(-PATCH) & -openmc.XPlane(PATCH) & +openmc.YPlane(-PATCH) & -openmc.YPlane(PATCH))
    ztop = openmc.ZPlane(z0=GAP)
    outer = openmc.Sphere(r=2000, boundary_type="vacuum")
    slab = openmc.Cell(fill=mat, region=box & -top & +bot)
    layer = openmc.Cell(region=patch & +top & -ztop)
    rest = openmc.Cell(region=-outer & ~(box & -top & +bot) & ~(patch & +top & -ztop))
    geom = openmc.Geometry([slab, layer, rest])
    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.particles, s.batches = particles, batches
    s.photon_transport = True
    s.electron_treatment = "ttb"
    src = openmc.IndependentSource()
    src.space = openmc.stats.Box([-LAT, -LAT, -2 * HALF], [LAT, LAT, 0.0])
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([E_eV], [1.0])
    src.particle = "photon"
    s.source = src
    s.output = {"tallies": False}
    eb = np.geomspace(1e3, E_eV * 1.001, 200)
    t = openmc.Tally()
    t.filters = [openmc.CellFilter([layer]), openmc.EnergyFilter(eb), openmc.ParticleFilter(["photon"])]
    t.scores = ["flux"]
    model = openmc.Model(geom, openmc.Materials([mat]), s, openmc.Tallies([t]))
    d = os.path.join(ACT, "runs", "dose_check", f"{mat.name}_{int(E_eV)}")
    os.makedirs(d, exist_ok=True)
    sp = model.run(cwd=d, threads=int(os.environ.get("OMP_NUM_THREADS", "8")))
    with openmc.StatePoint(sp) as f:
        tt = f.tallies[t.id]
        phi = tt.mean.ravel()                           # cm per source photon
        sd = tt.std_dev.ravel()
    vol_layer = (2 * PATCH) ** 2 * GAP
    mass_g = mat.get_mass_density() * (2 * LAT) ** 2 * 2 * HALF
    Em = np.sqrt(eb[1:] * eb[:-1])
    air = openmc.data.mass_energy_absorption_coefficient("air")
    # fluence rate per (photon/kg/s): track / V per source photon, times mass_kg source photons/s
    mass_kg = mass_g / 1000
    w = phi / vol_layer * mass_kg * Em * 1e-6 * air(Em)
    D = 3600 * 1.602176634e-13 * 1e3 * w.sum()
    rel = np.sqrt(((sd / vol_layer * mass_kg * Em * 1e-6 * air(Em)) ** 2).sum()) / w.sum()
    return float(D), float(rel)


def formula(mat, E_eV):
    air = openmc.data.mass_energy_absorption_coefficient("air")
    return float(DOSE_C * DOSE_B / 2 * air(np.array([E_eV]))[0] / mu_rho_material(mat, np.array([E_eV]))[0] * E_eV * 1e-6)


def main():
    m = materials(DEFAULTS)
    out = []
    for key in ("inconel", "W", "TiH2", "flibe"):
        for E in (0.3e6, 0.8e6, 1.25e6, 2.5e6, 6.1e6):
            mc, rel = mc_dose(m[key], E)
            f = formula(m[key], E)
            row = {"material": m[key].name, "E_MeV": E / 1e6, "mc_Gy_h_per_photon_kg_s": mc, "mc_rel_sd": rel,
                   "proxy": f, "proxy_over_mc": f / mc}
            print(row)
            out.append(row)
    json.dump({"note": __doc__.split("\n\n")[0], "rows": out}, open(os.path.join(ACT, "runs", "dose_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

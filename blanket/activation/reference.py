"""OpenMC depletion reference for the activation tier: IndependentOperator on the ARC
baseline cells with the CE fluxes and one-group cross sections of arc_activation.py's run,
the FULL ENDF/B-VIII.0 chain (3820 nuclides, no pruning), 1 full-power year at the design
source rate, then decay to each cooling time (OpenMC's CRAM-48 throughout).

One-group cross sections, as get_microxs_and_flux(reaction_rate_mode="flux",
reaction_rate_opts=initial nuclides) would make them from the same statepoint: direct CE
rate / CE flux for the nuclides initially in each cell, MicroXS.from_multigroup_flux (the
cell's 175-group CE flux, flat-in-energy group averages of the pointwise data) for every
other chain nuclide with data.

Contact dose from the resulting inventories with common.dose_coeffs (the same formula and
photon data the browser sidecar uses).

Run: docker compose run --rm openmc blanket/activation/reference.py [RUN_DIR]
Writes RUN_DIR/reference.json.
"""
import argparse, json, os, time
import numpy as np
import openmc, openmc.deplete as dep
from common import ACT, CHAIN, COOL, FPY_S, prune, load_chain, lib_nuclides, cell_materials, photon_lines, dose_coeffs, DOSE_C, DOSE_B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(ACT, "runs", "VITAMIN-J-175"))
    ap.add_argument("--fpy", type=float, default=1.0)
    a = ap.parse_args()
    openmc.config["chain_file"] = CHAIN
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    t = np.load(os.path.join(a.run, "tallies.npz"))
    edges = t["group_edges"]
    flux = t["flux"]                                    # [cell, g], group 0 = highest
    chain, lib = load_chain(), lib_nuclides()
    named = meta["cells"]
    _, _, _, mats, _ = cell_materials(meta["design"] or None)
    with_data = [n.name for n in chain.nuclides if n.name in lib]
    reactions = chain.reactions

    materials, fluxes, micros = [], [], []
    t0 = time.time()
    with openmc.lib.TemporarySession():
        for ci, n in enumerate(named):
            m = mats[n].clone()
            m.name = n
            m.depletable = True
            m.volume = meta["volumes_cm3"][n]
            phi = flux[ci]
            mx = dep.MicroXS.from_multigroup_flux(edges, phi[::-1].copy(), CHAIN, 293.6, with_data, reactions)
            # direct CE one-group xs for the nuclides initially present
            mname = meta["materials"][n]["name"]
            info = meta["rr_tallies"][mname]
            rr = t[f"rr_{mname}"][info["cells"].index(n)]        # [g, nuc, score]
            dens = meta["materials"][n]["atoms_b_cm"]
            data = mx.data.copy()
            for i, x in enumerate(info["nuclides"]):
                for j, s in enumerate(info["scores"]):
                    if s in reactions:
                        data[with_data.index(x), reactions.index(s), 0] = rr[:, i, j].sum() / (dens[x] * phi.sum())
            micros.append(dep.MicroXS(data, with_data, reactions))
            materials.append(m)
            fluxes.append(float(phi.sum()))
    print(f"microxs {time.time() - t0:.0f}s")

    S = meta["neutrons_per_s"]
    cool = list(COOL.values())
    steps = [a.fpy * FPY_S] + list(np.diff([0.0] + cool))
    rates = [S] + [0.0] * len(cool)
    op = dep.IndependentOperator(openmc.Materials(materials), fluxes, micros, CHAIN, normalization_mode="source-rate")
    integ = dep.PredictorIntegrator(op, steps, source_rates=rates, timestep_units="s")
    out_dir = os.path.join(a.run, "depletion")
    os.makedirs(out_dir, exist_ok=True)
    cwd = os.getcwd(); os.chdir(out_dir)
    t0 = time.time()
    integ.integrate()
    os.chdir(cwd)
    print(f"depletion {time.time() - t0:.0f}s")
    res = dep.Results(os.path.join(out_dir, "depletion_results.h5"))

    lines = {x.name: photon_lines(x) for x in chain.nuclides}
    names_all = [x.name for x in chain.nuclides]
    half = {x.name: x.half_life for x in chain.nuclides}
    out = {"half_life_s": {}, "cooling_labels": list(COOL), "cooling_s": cool, "fpy": a.fpy, "neutrons_per_s": S,
           "chain": os.path.basename(CHAIN), "chain_nuclides": len(chain), "cells": {}}
    for ci, n in enumerate(named):
        m = materials[ci]
        _, act = res.get_activity(m, units="Bq", by_nuclide=False)
        _, actn = res.get_activity(m, units="Bq", by_nuclide=True)
        _, heat = res.get_decay_heat(m, units="W", by_nuclide=False)
        _, heatn = res.get_decay_heat(m, units="W", by_nuclide=True)
        # atoms of every nuclide at every step
        atoms = {}
        for x in names_all:
            try:
                _, nx = res.get_atoms(m, x)
            except (KeyError, ValueError):
                continue
            if np.any(nx > 0):
                atoms[x] = nx
        keys = [x for x in atoms if np.any(lines[x][1] > 0)]
        dk = dict(zip(keys, dose_coeffs(mats[n], keys, lines)))
        mass_g = mats[n].get_mass_density() * m.volume
        dose = np.zeros(len(steps) + 1)
        dose_by = {}
        for x in keys:
            d = DOSE_C * DOSE_B / 2 * dk[x] * atoms[x] / (mass_g / 1000)   # per kg
            dose += d
            dose_by[x] = d
        cells = {}
        idx = list(range(2, len(steps) + 1))        # step 0 = start, 1 = end of irradiation
        # every contributor above 1e-7 of the total (the comparison separates CRAM round-off
        # in exotic short-lived nuclides from physics outside the browser's pruned chain)
        def byn(d, tot):
            return {x: float(v) for x, v in d.items() if tot > 0 and v > 1e-7 * tot}
        cell = {"volume_cm3": m.volume, "mass_kg": mass_g / 1000, "total_flux_cm_per_src": fluxes[ci],
                "activity_Bq": [float(act[k]) for k in idx], "decay_heat_W": [float(heat[k]) for k in idx],
                "contact_dose_Sv_h": [float(dose[k]) for k in idx],
                "shutdown": {"activity_Bq": float(act[1]), "decay_heat_W": float(heat[1]), "contact_dose_Sv_h": float(dose[1])},
                "by_nuclide_activity": [byn(actn[k], act[k]) for k in idx],
                "by_nuclide_heat": [byn(heatn[k], heat[k]) for k in idx],
                "by_nuclide_dose": [byn({x: v[k] for x, v in dose_by.items()}, dose[k]) for k in idx]}
        # what is physically reachable at all: <= 4 neutron reactions + decays from the
        # initial composition (the browser keeps <= 2). Contributors outside this set are
        # CRAM round-off (IPF's absolute error ~1e-15 x the total atoms, times a large lambda)
        cell["reach4"] = sorted(prune(chain, list(meta["materials"][n]["atoms_b_cm"]), lib, depth=4))
        out["cells"][n] = cell
        print(n, "A(1 d) %.3e Bq  H(1 d) %.3e W  D(1 d) %.3e Sv/h" % (cell["activity_Bq"][3], cell["decay_heat_W"][3], cell["contact_dose_Sv_h"][3]))
    used = {x for c in out["cells"].values() for q in ("by_nuclide_activity", "by_nuclide_heat", "by_nuclide_dose") for d in c[q] for x in d}
    out["half_life_s"] = {x: half[x] for x in sorted(used)}
    json.dump(out, open(os.path.join(a.run, "reference.json"), "w"), indent=1)
    print("wrote", os.path.join(a.run, "reference.json"))


if __name__ == "__main__":
    main()

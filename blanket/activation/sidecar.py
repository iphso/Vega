"""tallies.npz (arc_activation.py) -> activation_sidecar.json for the browser
(StoneLab packages/physics/src/neutron/activation.ts). Format in that file's header and
NOTES.md.

Per cell: the pruned chain (common.prune), the initial atom densities, Gc-group microscopic
cross sections of every chain reaction of every nuclide with neutron data, collapsed in the
cell's own CE spectrum -- direct CE rate / CE flux for the nuclides initially present (the
tallies carry self-shielding), OpenMC's flat-in-energy group average of the pointwise data
(openmc.lib collapse_rate, what MicroXS.from_multigroup_flux uses) for products -- and the
contact-dose coefficient of every nuclide in that cell's material.

Run: docker compose run --rm openmc blanket/activation/sidecar.py [RUN_DIR] [--edges e1,e2,..]
     (--edges: coarse-group boundaries in eV, descending, inside 1e-5 .. 2e7)
"""
import argparse, json, os
import numpy as np
import openmc, openmc.lib, openmc.data
from openmc.data import REACTION_MT
from common import (ACT, CELLS, load_chain, lib_nuclides, prune, secondaries, photon_lines,
                    dose_coeffs, DOSE_C, DOSE_B, cell_materials)

# default coarse groups (eV, descending interior edges): see coarse_groups.py / NOTES.md
DEFAULT_EDGES = [12.0e6, 5.0e6, 1.0e6, 1.0e5, 1.0e3, 10.0, 0.625]
RESOLVED = 0.25


def coarse_map(edges_asc, interior_desc):
    """Fine group (0 = highest) -> coarse group, coarse boundaries snapped to fine edges."""
    G = len(edges_asc) - 1
    hi = edges_asc[::-1]          # hi[g] = upper edge of fine group g
    lo = hi[1:]                    # lower edge of fine group g
    cmap = np.zeros(G, int)
    snapped = []
    for e in interior_desc:
        k = int(np.argmin(np.abs(np.log(hi[1:G] / e)))) + 1   # nearest fine boundary
        snapped.append(float(hi[k]))
    for g in range(G):
        cmap[g] = sum(1 for s in snapped if lo[g] < s * (1 - 1e-9))
    return cmap, snapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(ACT, "runs", "VITAMIN-J-175"))
    ap.add_argument("--edges", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    t = np.load(os.path.join(a.run, "tallies.npz"))
    edges = t["group_edges"]                      # ascending
    G = len(edges) - 1
    interior = [float(x) for x in a.edges.split(",")] if a.edges else DEFAULT_EDGES
    cmap, snapped = coarse_map(edges, interior)
    Gc = int(cmap.max()) + 1
    flux, flux_sd = t["flux"], t["flux_sd"]       # [cell, g], group 0 = highest
    chain, lib = load_chain(), lib_nuclides()
    named = meta["cells"]
    _, _, _, mats, _ = cell_materials(meta["design"] or None)

    # ---- per-cell pruned chains and the global nuclide table
    reach = {}
    for n in named:
        init = list(meta["materials"][n]["atoms_b_cm"])
        reach[n] = sorted(prune(chain, init, lib), key=lambda x: chain.nuclide_dict[x])
    glob = sorted({x for n in named for x in reach[n]}, key=lambda x: chain.nuclide_dict[x])
    gidx = {x: i for i, x in enumerate(glob)}
    nuclides = []
    for x in glob:
        nu = chain[x]
        nuclides.append({"name": x, "half_life_s": nu.half_life if nu.half_life else None,
                         "decay_energy_eV": float(nu.decay_energy or 0.0),
                         "decay": [[int(gidx[m.target]), float(m.branching_ratio)] for m in nu.decay_modes
                                   if m.target in gidx]})

    # branching ratios of multi-target reactions must sum to 1 (openmc.deplete removes the
    # parent once at the full rate; we remove it once per target at sigma * br)
    for x in glob:
        tot = {}
        for r in chain[x].reactions:
            tot[r.type] = tot.get(r.type, 0.0) + (r.branching_ratio if r.branching_ratio is not None else 1.0)
        for k, v in tot.items():
            if abs(v - 1) > 1e-6:
                print("WARNING branching", x, k, v)

    # ---- fine-group infinite-dilution xs of every (nuclide, reaction): products always; the
    # initial nuclides for coarse groups the cell's own CE tally doesn't resolve (below)
    need = set()
    for n in named:
        for x in reach[n]:
            if x in lib:
                for r in chain[x].reactions:
                    if r.type != "fission":
                        need.add((x, r.type))
    fine = {}
    with openmc.lib.TemporarySession():
        loaded = {}
        for (x, rt) in sorted(need):
            if x not in loaded:
                loaded[x] = openmc.lib.load_nuclide(x)
            nu, mt = loaded[x], REACTION_MT[rt]
            s = np.zeros(G)
            for g in range(G):                      # ascending energy order for collapse_rate
                f = np.zeros(G); f[g] = 1.0
                s[g] = nu.collapse_rate(mt, 293.6, edges, f)
            fine[(x, rt)] = s[::-1]                 # group 0 = highest

    # ---- contact dose coefficients per cell material
    lines = {x: photon_lines(chain[x]) for x in glob}
    # Donor spectrum for unresolved coarse groups: the sum over all cells (the blanket
    # dominates: well sampled at every energy). A coarse group counts as unresolved in a cell
    # when its CE flux there has a relative standard deviation above RESOLVED (thin VV layers
    # see few thermal neutrons in the baseline; a softer design can raise them a lot, and a
    # zero there would read as no capture at all).
    donor = flux.sum(axis=0)
    cells_out = {}
    for ci, n in enumerate(named):
        mname = meta["materials"][n]["name"]
        dens = meta["materials"][n]["atoms_b_cm"]
        rt_info = meta["rr_tallies"][mname]
        rr = t[f"rr_{mname}"][rt_info["cells"].index(n)]          # [g, nuc, score]
        phi = flux[ci]
        phic = np.array([phi[cmap == c].sum() for c in range(Gc)])
        rsdc = np.array([np.sqrt((flux_sd[ci][cmap == c] ** 2).sum()) / phic[c] if phic[c] > 0 else np.inf for c in range(Gc)])
        unres = rsdc > RESOLVED
        donor_c = np.array([donor[cmap == c].sum() for c in range(Gc)])
        loc = {x: i for i, x in enumerate(reach[n])}
        reactions = []
        for x in reach[n]:
            if x not in lib:
                continue
            for r in chain[x].reactions:
                if r.type == "fission":
                    continue
                br = float(r.branching_ratio) if r.branching_ratio is not None else 1.0
                sg = fine[(x, r.type)]
                if x in rt_info["nuclides"] and r.type in rt_info["scores"]:
                    Rg = rr[:, rt_info["nuclides"].index(x), rt_info["scores"].index(r.type)]
                    Rc = np.array([Rg[cmap == c].sum() for c in range(Gc)])
                    sig = np.where(phic > 0, Rc / np.where(phic > 0, phic, 1) / dens[x], 0.0)   # barns
                else:
                    sig = np.array([(sg[cmap == c] * phi[cmap == c]).sum() for c in range(Gc)])
                    sig = np.where(phic > 0, sig / np.where(phic > 0, phic, 1), 0.0)
                for c in np.nonzero(unres)[0]:
                    sig[c] = (sg[cmap == c] * donor[cmap == c]).sum() / donor_c[c]
                sig = sig * br
                if not np.any(sig > 0):
                    continue
                reactions.append([int(loc[x]), int(loc.get(r.target, -1)), r.type, [float(v) for v in sig],
                                  [int(loc[y]) for y in secondaries(r.type, chain)]])
        dose_k = list(dose_coeffs(mats[n], reach[n], lines))
        cells_out[n] = {
            "material": mname, "density_g_cm3": meta["materials"][n]["density_g_cm3"],
            "volume_cm3": meta["volumes_cm3"][n],
            "nuclides": [int(gidx[x]) for x in reach[n]],
            "n0": [float(dens.get(x, 0.0)) * 1e24 for x in reach[n]],
            "reactions": reactions, "dose_k": [float(v) for v in dose_k],
            "ce_track": [float(v) for v in phic],
            "donor_groups": [int(c) for c in np.nonzero(unres)[0]],
            "ce_track_rel_sd": [float(np.sqrt((flux_sd[ci][cmap == c] ** 2).sum()) / phic[c]) if phic[c] > 0 else 0.0
                                for c in range(Gc)],
        }
        print(n, mname, "chain", len(reach[n]), "reactions", len(reactions))
    out = {
        "format_version": 1, "groups": meta["groups"], "G": G, "Gc": Gc,
        "coarse_of_group": [int(v) for v in cmap], "coarse_edges_eV": [float(edges[-1])] + snapped + [float(edges[0])],
        "group_order": "group 0 = highest energy (as the neutpack); coarse_edges_eV descending",
        "design": meta["design"], "histories": meta["particles"] * meta["batches"],
        "neutrons_per_s": meta["neutrons_per_s"], "chain": "chain_endf_b8.0.xml (OpenMC, ENDF/B-VIII.0)",
        "dose": {"C": DOSE_C, "B": DOSE_B,
                 "formula": "Sv/h = C * B/2 * sum_i N_i * dose_k_i / mass_kg; dose_k = sum_lines (mu_en/rho)_air/(mu/rho)_mat * E[MeV] * photons/s/atom"},
        "units": {"n0": "atoms/cm3", "sigma": "barns", "ce_track": "cm per source neutron"},
        "nuclides": nuclides, "cells": cells_out,
    }
    path = a.out or os.path.join(a.run, "activation_sidecar.json")
    json.dump(out, open(path, "w"))
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.2f} MB", "Gc", Gc, "edges", out["coarse_edges_eV"])


if __name__ == "__main__":
    main()

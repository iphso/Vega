"""CE transport for the activation tier: one OpenMC fixed-source run of the ARC baseline
(../arc_model.py build(), unchanged) tallying, per cell, the VITAMIN-J-175 flux and the
direct (n,x) reaction rates of every nuclide initially present, per group, for every
reaction type the depletion chain lists for it. Both the browser sidecar (sidecar.py)
and the OpenMC depletion reference (reference.py) are built from this one statepoint.

Run: docker compose run --rm openmc blanket/activation/arc_activation.py \
        [--particles N] [--batches B] [--set key=value ...]
Writes runs/VITAMIN-J-175[_tags]/{statepoint, tallies.npz, meta.json}.
"""
import argparse, json, os, time
import numpy as np
import openmc, openmc.mgxs
from common import ACT, CELLS, cell_materials, load_chain, lib_nuclides, miller, NEUTRONS_PER_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=100000)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ww", action="store_true",
                    help="weight windows (MAGIC, generated in a first run) so the tank wall, shield and coil converge")
    ap.add_argument("--ww-l", type=float, default=12.0)
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=")
        over[k] = v if k == "fw_material" else float(v)
    tag = "".join(f"_{k}-{v}" for k, v in over.items()) + ("_ww" if a.ww else "")
    out = os.path.join(ACT, "runs", f"VITAMIN-J-175{tag}")
    os.makedirs(out, exist_ok=True)

    model, cells, p, mats, vol = cell_materials(over)
    model.settings.particles, model.settings.batches = a.particles, a.batches
    chain, lib = load_chain(), lib_nuclides()
    edges = openmc.mgxs.EnergyGroups("VITAMIN-J-175").group_edges
    ef = openmc.EnergyFilter(edges)
    named = [n for n in CELLS if n in cells]
    tallies = openmc.Tallies()
    tf = openmc.Tally(name="flux")
    tf.filters = [openmc.CellFilter([cells[n] for n in named]), ef]
    tf.scores = ["flux"]
    tallies.append(tf)
    # one direct reaction-rate tally per material (cells sharing it share the tally)
    by_mat = {}
    for n in named:
        by_mat.setdefault(mats[n].name, []).append(n)
    meta = {"cells": named, "groups": "VITAMIN-J-175", "particles": a.particles, "batches": a.batches,
            "design": over, "params": p, "volumes_cm3": {n: vol[n] for n in named},
            "neutrons_per_s": NEUTRONS_PER_S, "materials": {}, "rr_tallies": {}}
    for mname, cl in by_mat.items():
        m = mats[cl[0]]
        dens = m.get_nuclide_atom_densities()
        nucs = [x for x in dens if x in lib and x in chain.nuclide_dict]
        scores = sorted({r.type for x in nucs for r in chain[x].reactions if r.type != "fission"})
        t = openmc.Tally(name=f"rr_{mname}")
        t.filters = [openmc.CellFilter([cells[n] for n in cl]), ef]
        t.nuclides = nucs
        t.scores = scores
        tallies.append(t)
        meta["rr_tallies"][mname] = {"cells": cl, "nuclides": nucs, "scores": scores}
    for n in named:
        meta["materials"][n] = {"name": mats[n].name, "density_g_cm3": mats[n].get_mass_density(),
                                "atoms_b_cm": {k: float(v) for k, v in mats[n].get_nuclide_atom_densities().items()}}
    model.tallies = tallies
    model.settings.output = {"tallies": False}
    threads = int(os.environ.get("OMP_NUM_THREADS", "12"))
    if a.ww:
        # Static weight windows on an (R, Z) mesh: lower bound 0.25 exp(-d / WW_L) with d the
        # distance outside the tank box (the shield and coil), energy-independent. (OpenMC's
        # MAGIC generator on this model ran away: tracks grew ~1.5x per secondary generation.)
        # WW_L ~ the fast-flux e-folding length through TiH2 + the coil pack (the baseline
        # run's shield/coil volume-averaged fluxes give ~9-14 cm).
        pts = miller(p["R0"], p["a"], p["kappa"], p["delta"], 720)
        vv = sum(p[k] for k in ("fw_cm", "vv_in_cm", "channel_cm", "be_cm", "vv_out_cm"))
        tk = p["tank_cm"]
        t_rmin = pts[:, 0].min() - vv - p["blanket_in_cm"] - tk
        t_rmax = pts[:, 0].max() + vv + p["blanket_out_cm"] + tk
        t_z = pts[:, 1].max() + vv + p["blanket_vert_cm"] + tk
        rmax = t_rmax + p["shield_out_cm"] + p["coil_cm"]
        zmax = t_z + p["shield_vert_cm"] + p["coil_cm"]
        h = 5.0
        rg = np.arange(0.0, rmax + h, h)
        zg = np.arange(-zmax - h, zmax + h, h)
        rc, zc = 0.5 * (rg[1:] + rg[:-1]), 0.5 * (zg[1:] + zg[:-1])
        R, Z = np.meshgrid(rc, zc, indexing="ij")                # mesh order: r fastest
        d = np.maximum.reduce([t_rmin - R, R - t_rmax, np.abs(Z) - t_z, np.zeros_like(R)])
        lower = np.maximum(0.25 * np.exp(-d / a.ww_l), 1e-9)
        mesh = openmc.CylindricalMesh(r_grid=rg, phi_grid=[0, 2 * np.pi], z_grid=zg)
        lb = lower[:, None, :, None]                             # (r, phi, z, energy)
        model.settings.weight_windows = [openmc.WeightWindows(mesh, lb, upper_bound_ratio=5.0, particle_type="neutron")]
        model.settings.weight_windows_on = True
        model.settings.max_history_splits = 500
        meta_ww = {"L_cm": a.ww_l, "mesh_cm": h}

    if a.ww:
        meta["weight_windows"] = meta_ww
    t0 = time.time()
    sp_path = model.run(cwd=out, threads=threads)
    meta["seconds"] = time.time() - t0

    arrs = {"group_edges": edges}
    with openmc.StatePoint(sp_path) as sp:
        f = sp.get_tally(name="flux")
        G = len(edges) - 1
        arrs["flux"] = f.get_values(scores=["flux"]).reshape(len(named), G)[:, ::-1]          # group 0 = highest
        arrs["flux_sd"] = f.get_values(scores=["flux"], value="std_dev").reshape(len(named), G)[:, ::-1]
        for mname, info in meta["rr_tallies"].items():
            t = sp.get_tally(name=f"rr_{mname}")
            v = t.mean.reshape(len(info["cells"]), G, len(info["nuclides"]), len(info["scores"]))
            arrs[f"rr_{mname}"] = v[:, ::-1]                                                # [cell, g, nuc, score]
    np.savez_compressed(os.path.join(out, "tallies.npz"), **arrs)
    meta["statepoint"] = os.path.basename(sp_path)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("done", out, f"{meta['seconds']:.0f}s")


if __name__ == "__main__":
    main()

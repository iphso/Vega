"""Vacuum-vessel radiation damage (NRT dpa, He appm) on the ARC baseline, OpenMC CE.

Imports the reference geometry from ../arc_model.py unchanged, then:
  - splits the inner VV (Inconel 718) into NSUB radial sublayers (offsets of the
    first-wall polygon, so the outer bounds are the original cell's surfaces);
  - tallies per cell x VITAMIN-J-175 group: flux, damage-energy (MT 444, NJOY
    HEATR damage energy, Lindhard-partitioned), (n,Xa) (MT 207, total He-4
    production);
  - tallies a 1 cm cylindrical (R, Z) mesh x inner-VV sublayer, for the peak;
  - writes the browser sidecar coeffs_damage.npz: per-cell 175-group response
    functions R_g = rate_g / flux_g (CE rate/flux ratios, the same construction
    and the same cell keys/order as mgxs_gen.py's coeffs.npz; group 0 = highest).

NRT: dpa = 0.8 * T_dam / (2 E_d), E_d = 40 eV for Inconel 718 (Ni/Fe/Cr, ASTM
E693 practice); applied to the summed damage energy (the sub-threshold PKA
region is negligible in this spectrum).

Run: docker compose run --rm openmc blanket/damage/arc_damage.py [--particles N]
     [--batches B] [--set key=value ...]      (AICR: run.sbatch, same args)
"""
import argparse, json, os, sys
import numpy as np
import openmc, openmc.mgxs

DMG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(DMG))
from arc_model import build, miller, NEUTRONS_PER_S, DEFAULTS  # noqa: E402

FPY_S = 3.15576e7
ED_EV = {"inconel718": 40.0, "tungsten": 90.0, "beryllium": 31.0, "coil": 40.0}  # ASTM E693-style
NRT_K = 0.8 / 2.0
NSUB = 4
SECTOR_DEG = 10.0


def poly_moment(pts):
    """Integral of R dA over a closed polygon in (R, Z) (shoelace first moment)."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    return abs(np.sum((x + x1) * cr) / 6.0)


def inside(pts, R, Z):
    """Even-odd point-in-polygon, vectorised over points."""
    res = np.zeros(R.shape, bool)
    n = len(pts)
    for i in range(n):
        ax, az = pts[i - 1]
        bx, bz = pts[i]
        c = (az > Z) != (bz > Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = ax + (Z - az) * (bx - ax) / (bz - az)
        res ^= c & (R < xc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=")
        over[k] = v if k == "fw_material" else float(v)
    tag = "".join(f"_{k}-{v}" for k, v in over.items())
    outdir = os.path.join(DMG, "runs", f"VITAMIN-J-175{tag}")
    os.makedirs(outdir, exist_ok=True)

    model, cells, p = build(over or None, a.particles, a.batches, photons=False)
    inc = cells["vv_in"].fill
    # ---- split vv_in radially
    fw_poly = openmc.model.Polygon(miller(p["R0"], p["a"], p["kappa"], p["delta"]), basis="rz")
    t0, dt = p["fw_cm"], p["vv_in_cm"] / NSUB
    offs = [fw_poly.offset(t0 + k * dt) for k in range(NSUB + 1)]   # offs[0], offs[-1] for volumes only
    base_region = cells["vv_in"].region
    subs = []
    for k in range(NSUB):
        reg = base_region
        if k > 0:
            reg = reg & +offs[k]
        if k < NSUB - 1:
            reg = reg & -offs[k + 1]
        subs.append(openmc.Cell(name=f"vv_in_{k}", fill=inc, region=reg))
    cell_list = [c for n, c in cells.items() if n != "vv_in"] + subs
    model.geometry = openmc.Geometry(cell_list)
    # exact sublayer volumes (solids of revolution of mitred offset polygons)
    mom = [poly_moment(np.asarray(o.points)) for o in offs]
    sub_vol = np.array([2 * np.pi * (mom[k + 1] - mom[k]) for k in range(NSUB)])

    # ---- tallies
    groups = openmc.mgxs.EnergyGroups("VITAMIN-J-175")
    edges = groups.group_edges
    named = [n for n in ("fw", "vv_in", "channel", "be", "vv_out", "blanket", "tank", "shield", "coil") if n in cells]
    domains = [cells[n] for n in named if n != "vv_in"] + subs
    tallies = openmc.Tallies()
    tg = openmc.Tally(name="grp")
    tg.filters = [openmc.CellFilter(domains), openmc.EnergyFilter(edges)]
    tg.scores = ["flux", "damage-energy", "(n,Xa)"]
    tallies.append(tg)
    tt = openmc.Tally(name="TBR")
    tt.filters = [openmc.CellFilter([cells["channel"], cells["blanket"]])]
    tt.scores = ["(n,Xt)"]
    tallies.append(tt)
    rmin = np.floor(min(np.asarray(offs[0].points)[:, 0])) - 1
    rmax = np.ceil(max(np.asarray(offs[-1].points)[:, 0])) + 1
    zmax = np.ceil(max(np.asarray(offs[-1].points)[:, 1])) + 1
    rg = np.arange(rmin, rmax + 0.5, 1.0)
    zg = np.arange(-zmax, zmax + 0.5, 1.0)
    if not a.no_mesh:
        mesh = openmc.CylindricalMesh(r_grid=rg, z_grid=zg, phi_grid=[0.0, 2 * np.pi])
        tm = openmc.Tally(name="mesh")
        tm.filters = [openmc.MeshFilter(mesh), openmc.CellFilter(subs)]
        tm.scores = ["damage-energy", "(n,Xa)"]
        tallies.append(tm)
    model.tallies = tallies
    model.settings.output = {"tallies": False}
    sp_path = model.run(cwd=outdir, threads=int(os.environ.get("OMP_NUM_THREADS", "24")))

    S = NEUTRONS_PER_S
    with openmc.StatePoint(sp_path) as sp:
        g = sp.get_tally(name="grp")
        shp = (len(domains), len(edges) - 1)
        val = {s: g.get_values(scores=[s]).reshape(shp)[:, ::-1] for s in g.scores}   # group 0 = highest
        sd = {s: g.get_values(scores=[s], value="std_dev").reshape(shp)[:, ::-1] for s in g.scores}
        tbr = float(sp.get_tally(name="TBR").mean.sum())
        if not a.no_mesh:
            m = sp.get_tally(name="mesh")
            nmr, nmz = len(rg) - 1, len(zg) - 1
            # MeshFilter bins: r fastest, then phi, then z; then the cell filter
            mesh_d = m.get_values(scores=["damage-energy"]).reshape(nmz, nmr, NSUB)
            mesh_h = m.get_values(scores=["(n,Xa)"]).reshape(nmz, nmr, NSUB)
            mesh_dsd = m.get_values(scores=["damage-energy"], value="std_dev").reshape(nmz, nmr, NSUB)
            mesh_hsd = m.get_values(scores=["(n,Xa)"], value="std_dev").reshape(nmz, nmr, NSUB)

    dnames = [n for n in named if n != "vv_in"] + [f"vv_in_{k}" for k in range(NSUB)]
    # merge the sublayers back into vv_in (keyed/ordered like coeffs.npz)
    def merged(arr):
        rows = {n: arr[i] for i, n in enumerate(dnames)}
        rows["vv_in"] = sum(rows[f"vv_in_{k}"] for k in range(NSUB))
        return np.array([rows[n] for n in named])
    flux, dmg, he = merged(val["flux"]), merged(val["damage-energy"]), merged(val["(n,Xa)"])
    flux_var = merged(sd["flux"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_dmg = np.where(flux > 0, dmg / flux, 0.0)   # eV/cm
        r_he = np.where(flux > 0, he / flux, 0.0)     # 1/cm

    mats = {n: cells[n].fill for n in named}
    ndens = {n: sum(mats[n].get_nuclide_atom_densities().values()) * 1e24 for n in named}   # atoms/cm3
    ed = {n: ED_EV.get(mats[n].name) for n in named}

    # cell volumes: vv layers from polygon moments; the others aren't needed for dpa
    vv_layers = [("fw", p["fw_cm"]), ("vv_in", p["vv_in_cm"]), ("channel", p["channel_cm"]),
                 ("be", p["be_cm"]), ("vv_out", p["vv_out_cm"])]
    vol, acc, prev = {}, 0.0, poly_moment(np.asarray(fw_poly.points))
    for n, t in vv_layers:
        acc += t
        mo = poly_moment(np.asarray(fw_poly.offset(acc).points))
        vol[n] = 2 * np.pi * (mo - prev)
        prev = mo
    # tank wall volume (box shell, analytic)
    pts = miller(p["R0"], p["a"], p["kappa"], p["delta"], 720)
    vv_th = sum(t for _, t in vv_layers)
    b_rmin, b_rmax = pts[:, 0].min() - vv_th - p["blanket_in_cm"], pts[:, 0].max() + vv_th + p["blanket_out_cm"]
    b_z = pts[:, 1].max() + vv_th + p["blanket_vert_cm"]
    tk = p["tank_cm"]
    cyl = lambda r0, r1, h: np.pi * (r1 ** 2 - r0 ** 2) * h
    vol["tank"] = cyl(b_rmin - tk, b_rmax + tk, 2 * (b_z + tk)) - cyl(b_rmin, b_rmax, 2 * b_z)

    def rates(n, D, H, V):
        N = ndens[n]
        out = {"volume_cm3": V,
               "He_appm_per_FPY": H * S * FPY_S / (N * V) * 1e6,
               "damage_energy_eV_per_src": D}
        if ed[n]:
            out["dpa_per_FPY"] = NRT_K / ed[n] * D * S * FPY_S / (N * V)
        return out

    res = {"params": p, "histories": a.particles * a.batches, "TBR": tbr, "neutrons_per_s": S,
           "nrt": {"k": 0.8, "Ed_eV": ed}, "layers": {}, "vv_in_sublayers": []}
    for n in ("fw", "vv_in", "vv_out", "tank"):
        i = named.index(n)
        D, H = dmg[i].sum(), he[i].sum()
        Dsd = np.sqrt((sd["damage-energy"][[dnames.index(x) for x in ([n] if n != "vv_in" else [f"vv_in_{k}" for k in range(NSUB)])]] ** 2).sum())
        r = rates(n, float(D), float(H), float(vol[n]))
        r["rel_sd_damage"] = float(Dsd / D) if D > 0 else None
        r["fast_flux_gt0.1MeV"] = float(flux[i][np.array(edges[::-1][1:]) >= 0.1e6].sum() * S / vol[n])
        r["thermal_flux_lt0.625eV"] = float(flux[i][np.array(edges[::-1][:-1]) <= 0.625].sum() * S / vol[n])
        r["total_flux"] = float(flux[i].sum() * S / vol[n])
        res["layers"][n] = r
    for k in range(NSUB):
        i = dnames.index(f"vv_in_{k}")
        res["vv_in_sublayers"].append({"depth_cm": [k * dt, (k + 1) * dt],
                                       **rates("vv_in", float(val["damage-energy"][i].sum()), float(val["(n,Xa)"][i].sum()), float(sub_vol[k]))})

    # ---- peak: aggregate mesh elements into poloidal sectors about (R0, 0)
    if not a.no_mesh:
        nsamp = 8
        rc, zc = 0.5 * (rg[1:] + rg[:-1]), 0.5 * (zg[1:] + zg[:-1])
        RR, ZZ = np.meshgrid(rc, zc)                                # [nz, nr]
        # elements near the layer only
        o0, o1 = np.asarray(offs[0].points), np.asarray(offs[-1].points)
        near = inside(o1 * [1, 1], RR, ZZ) & ~inside(np.asarray(fw_poly.offset(max(t0 - 1.5, 0.01)).points), RR, ZZ)
        near |= (mesh_d.sum(axis=2) > 0)
        jj, ii = np.nonzero(near)
        sub_off = (np.arange(nsamp) + 0.5) / nsamp - 0.5
        dR, dZ = [x.ravel() for x in np.meshgrid(sub_off, sub_off)]
        Rs = RR[jj, ii][:, None] + dR[None, :]
        Zs = ZZ[jj, ii][:, None] + dZ[None, :]
        ins = np.stack([inside(np.asarray(o.points), Rs, Zs) for o in offs])  # [NSUB+1, ne, ns]
        wv = 2 * np.pi * Rs / nsamp ** 2                                       # cm3 per subsample
        evol = np.stack([((ins[k + 1] & ~ins[k]) * wv).sum(axis=1) for k in range(NSUB)], axis=1)  # [ne, NSUB]
        # rescale each sublayer's sampled volume to its exact volume (sampling error ~0.1%)
        evol *= sub_vol / evol.sum(axis=0)
        ang = np.degrees(np.arctan2(ZZ[jj, ii], RR[jj, ii] - p["R0"])) % 360.0
        nsec = int(round(360 / SECTOR_DEG))
        sec = np.minimum((ang / SECTOR_DEG).astype(int), nsec - 1)
        ed_d, ed_h = mesh_d[jj, ii], mesh_h[jj, ii]
        ed_dv, ed_hv = mesh_dsd[jj, ii] ** 2, mesh_hsd[jj, ii] ** 2
        N = ndens["vv_in"]
        prof = []
        for s in range(nsec):
            msk = sec == s
            Dk, Hk, Vk = ed_d[msk].sum(0), ed_h[msk].sum(0), evol[msk].sum(0)
            Dsd = np.sqrt(ed_dv[msk].sum(0))
            dpa = NRT_K / ed["vv_in"] * Dk * S * FPY_S / (N * Vk)
            heA = Hk * S * FPY_S / (N * Vk) * 1e6
            prof.append({"theta_deg": [s * SECTOR_DEG, (s + 1) * SECTOR_DEG],
                         "dpa_per_FPY": dpa.tolist(), "He_appm_per_FPY": heA.tolist(),
                         "rel_sd_dpa": (Dsd / np.where(Dk > 0, Dk, 1)).tolist(),
                         "dpa_layer_avg": float(NRT_K / ed["vv_in"] * Dk.sum() * S * FPY_S / (N * Vk.sum())),
                         "He_layer_avg": float(Hk.sum() * S * FPY_S / (N * Vk.sum()) * 1e6)})
        res["vv_in_poloidal"] = prof
        k_pk = int(np.argmax([q["dpa_per_FPY"][0] for q in prof]))
        k_pa = int(np.argmax([q["dpa_layer_avg"] for q in prof]))
        res["vv_in_peak"] = {
            "sector_plasma_side": prof[k_pk]["theta_deg"], "dpa_per_FPY": prof[k_pk]["dpa_per_FPY"][0],
            "He_appm_per_FPY": prof[k_pk]["He_appm_per_FPY"][0], "rel_sd": prof[k_pk]["rel_sd_dpa"][0],
            "sector_layer_avg": prof[k_pa]["theta_deg"], "dpa_layer_avg": prof[k_pa]["dpa_layer_avg"],
            "He_layer_avg": prof[k_pa]["He_layer_avg"],
            "peak_over_avg_dpa": prof[k_pa]["dpa_layer_avg"] / res["layers"]["vv_in"]["dpa_per_FPY"],
            "peak_over_avg_He": prof[k_pa]["He_layer_avg"] / res["layers"]["vv_in"]["He_appm_per_FPY"],
            "note": "theta = atan2(Z, R - R0): 0 outboard midplane, 90 top, 180 inboard midplane",
        }
        np.savez_compressed(os.path.join(outdir, "mesh.npz"), r_grid=rg, z_grid=zg, damage=mesh_d, he=mesh_h,
                            damage_sd=mesh_dsd, he_sd=mesh_hsd)

    # first-wall neutron wall loading (14.1 MeV power over the plasma-facing area)
    fp = np.asarray(fw_poly.points)
    seg = np.roll(fp, -1, axis=0) - fp
    area = float(np.sum(2 * np.pi * (fp[:, 0] + np.roll(fp, -1, axis=0)[:, 0]) / 2 * np.hypot(seg[:, 0], seg[:, 1]))) * 1e-4
    res["fw_area_m2"] = area
    res["neutron_wall_loading_MW_m2"] = S * 14.06 * 1.602176634e-13 / 1e6 / area

    # ---- browser sidecar (same cells / order / group convention as coeffs.npz)
    np.savez(os.path.join(outdir, "coeffs_damage.npz"), sig_damage=r_dmg, sig_he=r_he, ce_flux=flux,
             ce_flux_rel_sd=np.sqrt(flux_var) / np.where(flux > 0, flux, 1),
             cells=np.array(named), group_edges=edges,
             atom_density=np.array([ndens[n] for n in named]),
             Ed_eV=np.array([ed[n] or 0.0 for n in named]))
    man = {
        "groups": "VITAMIN-J-175", "group_order": "group 0 = highest energy (edges ascend, as coeffs.npz)",
        "design": over, "cells": named,
        "arrays": {
            "sig_damage": "[cell, G] damage energy (MT 444) per unit flux, eV/cm; sum_g track_g * sig_damage_g = eV of damage energy per source neutron in the cell",
            "sig_he": "[cell, G] He-4 production (MT 207, (n,Xa)) per unit flux, 1/cm; sum_g track_g * sig_he_g = He atoms per source neutron",
            "atom_density": "[cell] atoms/cm3", "Ed_eV": "[cell] NRT displacement energy, 0 = no dpa for this cell",
        },
        "convert": {
            "dpa_per_FPY": "0.8 / (2 Ed) * D * S * FPY / (N V)", "He_appm_per_FPY": "1e6 * H * S * FPY / (N V)",
            "S": "neutrons/s = P_fus / 17.59 MeV", "FPY_s": FPY_S, "V": "cell volume, cm3",
        },
        "vv_in_peak_over_avg": res.get("vv_in_peak", {}).get("peak_over_avg_dpa"),
        "vv_in_peak_over_avg_He": res.get("vv_in_peak", {}).get("peak_over_avg_He"),
        "vv_in_peak_sector_deg": res.get("vv_in_peak", {}).get("sector_layer_avg"),
        "vv_in_plasma_side_over_avg": (res["vv_in_sublayers"][0]["dpa_per_FPY"] / res["layers"]["vv_in"]["dpa_per_FPY"]),
        "histories": a.particles * a.batches,
    }
    json.dump(man, open(os.path.join(outdir, "coeffs_damage.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(outdir, "result.json"), "w"), indent=1)
    short = {k: v for k, v in res.items() if k not in ("vv_in_poloidal", "params")}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()

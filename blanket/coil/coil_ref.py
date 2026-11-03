"""Converged CE reference at the TF coil of the ARC baseline (arc_model.py build(), photons off):
fast (>0.1 MeV) fluence on the coil's inboard plasma-facing 5 cm (|Z| < 1 m: StoneLab
geometry.ts TF_FACE_CM / TF_HALF_H_CM, design.ts), 175-group spectra in that face, the whole
coil, the shield and the tank wall, an inboard radial profile of 175-group spectra (3 cm bins
from 66 cm inside the coil's face out to the blanket, |Z| < 1 m), coil heating-local, TBR.

The face is its own cell (same material as the rest of the coil): splitting a cell changes no
physics. Geometry, materials and source are arc_model.py's.

Variance reduction (--ww):
  none      analog
  static    lower bound 0.25 exp(-d / L) energy-independent, d = distance outside the tank box
            (activation/arc_activation.py's windows; L = 12 cm)
  <path>    a weight-window file written by ww_magic.py from an earlier run's mesh flux (.npz)
The shield and coil are filled with 5 cm R shells and windows are checked at surface crossings
too (shell_fills; --no-shells for the old behaviour): see NOTES.md for why both matter.

Run (from ~/Projects/Vega):
  docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=8 \
     -v ~/runs/blanket/coil:/runs openmc blanket/coil/coil_ref.py --ww static --particles 1e5 --batches 10 --tag s0
Writes /runs/<tag>/{statepoint, result.npz, result.json}.
"""
import argparse, json, os, sys, time
import numpy as np
import openmc, openmc.mgxs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from arc_model import build, miller, NEUTRONS_PER_S

TF_FACE_CM, TF_HALF_H_CM = 5.0, 100.0      # StoneLab geometry.ts
PROF_H, PROF_NC, PROF_N = 3.0, 22, 40       # profile: 3 cm bins, 22 inside the face's outer radius
FPY_S = 3.15576e7
EDGES = openmc.mgxs.EnergyGroups("VITAMIN-J-175").group_edges
WW_E = [1e-5, 1e5, 1e6, 3e6, 1e7, 2e7]       # energy bins of the (R, Z) mesh flux that ww_magic.py reads


def boxes(p):
    pts = miller(p["R0"], p["a"], p["kappa"], p["delta"], 720)
    vv = sum(p[k] for k in ("fw_cm", "vv_in_cm", "channel_cm", "be_cm", "vv_out_cm"))
    b = dict(rmin=pts[:, 0].min() - vv - p["blanket_in_cm"], rmax=pts[:, 0].max() + vv + p["blanket_out_cm"])
    b["z"] = pts[:, 1].max() + vv + p["blanket_vert_cm"]
    tk = p["tank_cm"]
    t = dict(rmin=b["rmin"] - tk, rmax=b["rmax"] + tk, z=b["z"] + tk)
    s = dict(rmin=t["rmin"] - p["shield_in_cm"], rmax=t["rmax"] + p["shield_out_cm"], z=t["z"] + p["shield_vert_cm"])
    c = dict(rmin=max(1.0, s["rmin"] - p["coil_cm"]), rmax=s["rmax"] + p["coil_cm"], z=s["z"] + p["coil_cm"])
    return b, t, s, c


def vol_box(x):
    return np.pi * (x["rmax"] ** 2 - x["rmin"] ** 2) * 2 * x["z"]


def model_for(over=None):
    model, cells, p = build(over or None, 1000, 1, photons=False)
    b, t, s, c = boxes(p)
    sr = s["rmin"]
    # the face's outer boundary IS the shield box's inner cylinder (the same surface object): a
    # second, coincident ZCylinder(r=sr) leaves a neutron crossing from the shield on a surface
    # whose sense is ambiguous, and OpenMC then files it under the rest of the coil (same
    # material, so no physics changes, but the face cell tally read 8-14% low vs a mesh tally)
    outer = [x for x in cells["shield"].region.get_surfaces().values()
             if isinstance(x, openmc.ZCylinder) and abs(x.r - sr) < 1e-9]
    assert len(outer) == 1, outer
    face_reg = (+openmc.ZCylinder(r=sr - TF_FACE_CM) & -outer[0]
                & +openmc.ZPlane(-TF_HALF_H_CM) & -openmc.ZPlane(TF_HALF_H_CM))
    face = openmc.Cell(name="coil_face", fill=cells["coil"].fill, region=cells["coil"].region & face_reg)
    cells["coil"].region = cells["coil"].region & ~face_reg
    model.geometry = openmc.Geometry(list(cells.values()) + [face])
    cells["coil_face"] = face
    vols = {"coil_face": np.pi * (sr ** 2 - (sr - TF_FACE_CM) ** 2) * 2 * TF_HALF_H_CM,
            "tank": vol_box(t) - vol_box(b), "shield": vol_box(s) - vol_box(t)}
    vols["coil"] = vol_box(c) - vol_box(s)        # whole coil incl. the face
    return model, cells, p, (b, t, s, c), vols


def shell_fills(cells, bx, role_mat=None, h=5.0):
    """Fill the shield and coil cells with universes of concentric R shells (h cm, aligned on
    the shield's inner radius), all of the cell's own material: no physics changes, but a
    neutron streaming radially through the inboard leg now crosses a surface every h cm, where
    OpenMC can apply weight windows (settings.weight_window_checkpoints surface). Without it an
    uncollided 14 MeV neutron crosses 51 cm of TiH2 without a single window check (windows act
    only at collisions by default): the face's > 10 MeV score was a rare, unsplit event and the
    estimates scattered 4.7x between runs. CellFilters on the parent cells still score inside.
    role_mat: optional role -> material (multigroup mode), else the cells' own fills."""
    b, t, s, c = bx
    sr = s["rmin"]
    radii = sorted({float(r) for r in sr + h * np.arange(-int((sr - c["rmin"]) // h), int((t["rmin"] - sr) // h) + 1)
                    if c["rmin"] < r < t["rmin"] and abs(r - sr) > 1e-6 and abs(r - (sr - TF_FACE_CM)) > 1e-6})
    cyl = [openmc.ZCylinder(r=r) for r in radii]
    for role in ("shield", "coil", "coil_face"):
        mat = (role_mat or {}).get("coil" if role == "coil_face" else role)
        if role == "coil_face":
            if mat is not None:
                cells[role].fill = mat
            continue
        m = mat if mat is not None else cells[role].fill
        pieces = [openmc.Cell(fill=m, region=-cyl[0])]
        pieces += [openmc.Cell(fill=m, region=+cyl[i] & -cyl[i + 1]) for i in range(len(cyl) - 1)]
        pieces += [openmc.Cell(fill=m, region=+cyl[-1])]
        cells[role].fill = openmc.Universe(cells=pieces)
    return radii


def ww_mesh(bx, h=5.0):
    b, t, s, c = bx
    rg = np.arange(0.0, c["rmax"] + h, h)
    zg = np.arange(-c["z"] - h, c["z"] + 2 * h, h)
    return openmc.CylindricalMesh(r_grid=rg, phi_grid=[0, 2 * np.pi], z_grid=zg), rg, zg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=float, default=1e5)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--ww", default="none")
    ap.add_argument("--ww-l", type=float, default=12.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-history-splits", type=int, default=2000)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--mg", default=None, help="neutpack dir: OpenMC multigroup mode on the browser's own constants (pack_mgxs.py)")
    ap.add_argument("--no-shells", action="store_true", help="no R shells / surface window checks (the first runs)")
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=")
        over[k] = v if k in ("fw_material", "breeder", "layout") else float(v)
    out = os.path.join("/runs", a.tag)
    os.makedirs(out, exist_ok=True)
    model, cells, p, bx, vols = model_for(over)
    b, t, s, c = bx
    if a.mg:
        from pack_mgxs import to_mgxs
        h5 = os.path.join(out, "pack_mgxs.h5")
        to_mgxs(a.mg, h5)
        mats = {}
        for n, cell in cells.items():
            if cell.fill is None:
                continue
            role = "coil" if n == "coil_face" else n
            if role not in mats:
                m = openmc.Material(name=f"mg_{role}"); m.set_density("macro", 1.0); m.add_macroscopic(f"cell_{role}")
                mats[role] = m
            if role not in ("shield", "coil"):
                cell.fill = mats[role]
        model.materials = openmc.Materials(mats.values())
        model.materials.cross_sections = h5
        model.settings.energy_mode = "multi-group"
    sr = s["rmin"]
    if not a.no_shells:
        shell_fills(cells, bx, mats if a.mg else None)
        model.settings.weight_window_checkpoints = {"collision": True, "surface": True}
    elif a.mg:
        for role in ("shield", "coil", "coil_face"):
            cells[role].fill = mats["coil" if role == "coil_face" else role]
    st = model.settings
    st.particles, st.batches, st.seed = int(a.particles), a.batches, a.seed
    st.output = {"tallies": False}

    ef = openmc.EnergyFilter(EDGES)
    names = ["coil_face", "coil", "shield", "tank", "blanket", "channel", "fw"]
    cf = openmc.CellFilter([cells[n] for n in names])
    T = openmc.Tallies()
    t1 = openmc.Tally(name="cell_spec"); t1.filters = [cf, ef]; t1.scores = ["flux"] if a.mg else ["flux", "heating-local"]; T.append(t1)
    if not a.mg:
        t2 = openmc.Tally(name="tbr"); t2.filters = [openmc.CellFilter([cells["channel"], cells["blanket"]])]; t2.scores = ["(n,Xt)"]; T.append(t2)
    # face fast flux as one bin (its s.d. is honest; the group sum's is not: groups share histories)
    e_fast = float(EDGES[np.searchsorted(EDGES, 0.1e6 * (1 - 1e-9))])
    t5 = openmc.Tally(name="face_fast"); t5.filters = [openmc.CellFilter([cells["coil_face"]]), openmc.EnergyFilter([e_fast, float(EDGES[-1])])]
    t5.scores = ["flux"]; T.append(t5)
    # the same, by an (R, Z) mesh (independent of cell identification)
    fm = openmc.CylindricalMesh(r_grid=[sr - TF_FACE_CM, sr], phi_grid=[0, 2 * np.pi], z_grid=[-TF_HALF_H_CM, TF_HALF_H_CM])
    t6 = openmc.Tally(name="face_fast_mesh"); t6.filters = [openmc.MeshFilter(fm), openmc.EnergyFilter([e_fast, float(EDGES[-1])])]
    t6.scores = ["flux"]; T.append(t6)
    prof_r = sr + PROF_H * (np.arange(PROF_N + 1) - PROF_NC)
    pm = openmc.CylindricalMesh(r_grid=prof_r, phi_grid=[0, 2 * np.pi], z_grid=[-TF_HALF_H_CM, TF_HALF_H_CM])
    t3 = openmc.Tally(name="prof"); t3.filters = [openmc.MeshFilter(pm), ef]; t3.scores = ["flux"]; T.append(t3)
    wm, rg, zg = ww_mesh(bx)
    t4 = openmc.Tally(name="wwmesh"); t4.filters = [openmc.MeshFilter(wm), openmc.EnergyFilter(WW_E)]; t4.scores = ["flux"]; T.append(t4)
    model.tallies = T

    meta = {"tag": a.tag, "ww": a.ww, "particles": int(a.particles), "batches": a.batches, "design": over, "params": p,
            "volumes_cm3": {k: float(v) for k, v in vols.items()}, "prof_r_cm": prof_r.tolist(), "shield_rmin": sr, "neutrons_per_s": NEUTRONS_PER_S}
    if a.ww == "static":
        R, Z = np.meshgrid(0.5 * (rg[1:] + rg[:-1]), 0.5 * (zg[1:] + zg[:-1]), indexing="ij")
        d = np.maximum.reduce([t["rmin"] - R, R - t["rmax"], np.abs(Z) - t["z"], np.zeros_like(R)])
        lb = np.maximum(0.25 * np.exp(-d / a.ww_l), 1e-9)[:, None, :, None]
        st.weight_windows = [openmc.WeightWindows(wm, lb, upper_bound_ratio=5.0, particle_type="neutron")]
        meta["ww_l"] = a.ww_l
    elif a.ww != "none":
        z = np.load(a.ww)
        mesh = openmc.CylindricalMesh(r_grid=z["r_grid"], phi_grid=[0, 2 * np.pi], z_grid=z["z_grid"])
        st.weight_windows = [openmc.WeightWindows(mesh, z["lower"], upper_bound_ratio=float(z["ratio"]),
                                                  energy_bounds=z["energy_bounds"], particle_type="neutron",
                                                  max_lower_bound_ratio=10.0)]
        meta["ww_file"] = a.ww
    if a.ww != "none":
        st.weight_windows_on = True
        st.max_history_splits = a.max_history_splits
    t0 = time.time()
    sp_path = model.run(cwd=out, threads=int(os.environ.get("OMP_NUM_THREADS", "8")))
    meta["seconds"] = time.time() - t0

    G = len(EDGES) - 1
    with openmc.StatePoint(sp_path) as sp:
        meta["n_realizations"] = int(sp.n_realizations)
        x = sp.get_tally(name="cell_spec")
        fl = x.get_values(scores=["flux"]).reshape(len(names), G)[:, ::-1]
        fls = x.get_values(scores=["flux"], value="std_dev").reshape(len(names), G)[:, ::-1]
        if a.mg:
            hl = hls = np.zeros_like(fl); tbr = tbrs = float("nan")
        else:
            hl = x.get_values(scores=["heating-local"]).reshape(len(names), G)[:, ::-1]
            hls = x.get_values(scores=["heating-local"], value="std_dev").reshape(len(names), G)[:, ::-1]
            tb = sp.get_tally(name="tbr")
            tbr = float(tb.mean.sum()); tbrs = float(np.sqrt((tb.std_dev ** 2).sum()))
        f5 = sp.get_tally(name="face_fast")
        ff1, ff1s = float(f5.mean.ravel()[0]), float(f5.std_dev.ravel()[0])
        f6 = sp.get_tally(name="face_fast_mesh")
        ffm, ffms = float(f6.mean.ravel()[0]), float(f6.std_dev.ravel()[0])
        pr = sp.get_tally(name="prof")
        pf = pr.get_values(scores=["flux"]).reshape(PROF_N, G)[:, ::-1]
        pfs = pr.get_values(scores=["flux"], value="std_dev").reshape(PROF_N, G)[:, ::-1]
        w = sp.get_tally(name="wwmesh")
        # mesh bins: r fastest, then phi, then z
        wf = w.get_values(scores=["flux"]).reshape(len(zg) - 1, len(rg) - 1, len(WW_E) - 1)
        wfs = w.get_values(scores=["flux"], value="std_dev").reshape(len(zg) - 1, len(rg) - 1, len(WW_E) - 1)
        # per-batch face fast flux for a heavy-tail check (the batch means of the fast tally)
    # fast = groups whose lower edge >= 0.1 MeV (group 0 = highest)
    lo = EDGES[::-1][1:]
    fast = lo >= 0.1e6 * (1 - 1e-9)
    i_face = names.index("coil_face")
    ff = fl[i_face][fast].sum()
    ffs = float(np.sqrt((fls[i_face][fast] ** 2).sum()))   # groups are correlated (same histories): an underestimate
    phi = ff / vols["coil_face"]
    S = NEUTRONS_PER_S
    meta["face_fast_track_cm_per_src"] = float(ff)
    meta["face_fast_fluence_per_fpy"] = float(phi * S * FPY_S)
    meta["face_fast_rel_sd_groupsum"] = float(ffs / ff) if ff > 0 else None
    meta["face_fast_track_1bin"] = [ff1, ff1s]
    meta["face_fast_track_mesh"] = [ffm, ffms]
    meta["face_fast_rel_sd"] = ff1s / ff1 if ff1 > 0 else None
    meta["mg_pack"] = a.mg
    meta["magnet_life_fpy"] = float(3e18 / (phi * S * FPY_S)) if ff > 0 else None
    meta["TBR"] = [tbr, tbrs]
    eVtoW = S * 1.602176634e-19
    meta["heating_local_W"] = {n: [float(hl[i].sum() * eVtoW), float(np.sqrt((hls[i] ** 2).sum()) * eVtoW)] for i, n in enumerate(names)}
    np.savez_compressed(os.path.join(out, "result.npz"), names=np.array(names), flux=fl, flux_sd=fls, heat=hl, heat_sd=hls,
                        prof=pf, prof_sd=pfs, prof_r=prof_r, ww_flux=wf, ww_flux_sd=wfs, ww_r=rg, ww_z=zg, ww_e=np.array(WW_E), group_edges=EDGES)
    json.dump(meta, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("face_fast_fluence_per_fpy", "face_fast_rel_sd", "face_fast_rel_sd_groupsum", "magnet_life_fpy", "TBR", "seconds")}))


if __name__ == "__main__":
    main()

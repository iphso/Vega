"""Export watercap campaign results + per-design 3D scenes for the viewer page
(viewer/src/pages/watercap.astro).

  public/data/watercap/designs.json   every evaluated design: concept, key
                                      inputs, key outputs (resolved tier where
                                      available, else ROM spray + EHD corona)
  public/data/watercap/scenes/<id>.json  for a curated set (best design per
                                      concept + the baseline disc stack + the
                                      best disc geometry): re-runs the resolved
                                      spray stage with parcel-path recording and
                                      traces mist drops through the resolved
                                      corona field of that design's disc stack.

Scenes are a visualization re-run, not new results: the spray stage is the
same `resolved_coarse` solve; the corona drop paths are traced in the
corona field WITHOUT mist coupling (the campaign numbers include it).

    python3 scripts/export_watercap_viewer_data.py --workers 10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import watercap_campaign as wc  # noqa: E402
import watercap_oracle as rom  # noqa: E402
import watercap_resolved as wr  # noqa: E402  (puts StoneLab on sys.path)

OUT = HERE.parent / "viewer" / "public" / "data" / "watercap"

CONCEPT = {"corona_only": "corona", "brad_variants": "corona", "cold_spray": "spray", "multi_tier": "tiers",
           "coflow_spray": "coflow", "charged_spray": "charged", "charged_film": "charged", "ring_spray": "ring",
           "rain_curtain": "rain", "cooled_wall": "wall", "disc": "corona"}
CONCEPT_NAME = {"corona": "Disc stack only", "rain": "Large-drop curtain", "coflow": "Upward spray",
                "charged": "Charged spray onto the wall", "spray": "Cold spray, centre nozzle",
                "tiers": "Spray at several heights", "ring": "Ring of wall nozzles", "wall": "Cooled wall"}
DESIGN_KEYS = ["air_flow_m3_s", "tube_diameter_m", "spray_section_length_m", "corona_section_length_m",
               "nozzle_height_frac", "disc_diameter_frac", "disc_spacing_m", "disc_thickness_mm", "corona_voltage_kV",
               "corona_polarity", "spray_flow_L_min", "spray_T_C", "spray_d_um", "spray_q_frac", "spray_v0",
               "spray_cone_half_angle_deg", "nozzle_radial_frac", "spray_axis_angle_deg", "spray_n_tiers", "wall_U_W_m2K"]


def _r(v, n=4):
    return None if v is None else float(round(float(v), n))


def resolved_out(q):
    """The viewer's outcome keys from a watercap_resolved.simulate payload."""
    return dict(outlet=q["outlet_total_water_frac"], vapor=q["vapor_removed_frac"],
                escape=q["liquid_escape_frac"], recovered=q["recovered_L_h"],
                energy=q["specific_energy_Wh_L"], energy_total=q["specific_energy_total_Wh_L"],
                power=q["electrical_power_W"], current=q["corona_current_mA"],
                quench=q.get("quenching_frac"), arc=q["arc_margin"],
                closure=q["water_balance_residual"],
                fog=q["spray_stage"]["fog_out_frac"], film=q["spray_stage"]["film_drain"],
                momentum=q["momentum_number"], dp=q["pressure_drop_Pa"],
                recool=q["recool_power_W"])


def collect(full=None):
    """One row per design; the resolved tier wins where both exist."""
    rows = {}
    for r in wc.load("v1"):
        if not isinstance(r.get("ehd"), dict):
            continue
        e = r["ehd"]
        rows[r["id"]] = dict(id=r["id"], family=r["family"], tier="ehd", design=r["design"],
                             out=dict(outlet=e["outlet_total_water_frac"], vapor=e["vapor_removed_frac"],
                                      escape=e["liquid_escape_frac"], recovered=e["recovered_L_h"],
                                      energy=e["specific_energy_Wh_L"], energy_total=e["specific_energy_Wh_L"],
                                      power=e["electrical_power_W"], current=e["corona_current_mA"],
                                      quench=e.get("ehd_quenching_frac"), arc=e["arc_margin"],
                                      closure=e.get("water_balance_residual")))
    for rn in ("v2_rerun", "v2_resolved"):
        p = wc.OUT / rn / "results.jsonl"
        if not p.exists():
            continue
        for r in wc.load(rn):
            if not r.get("res_ok"):
                continue
            rows[r["id"]] = dict(id=r["id"], family=r["family"], tier="resolved", design=r["design"],
                                 out=dict(resolved_out(r["res"]), solve_s=r.get("res_s")))
    out = []
    for r in rows.values():
        if full is not None:
            full[r["id"]] = {k: v for k, v in r["design"].items() if k not in ("id", "family")}
        c = CONCEPT[r["family"]]
        out.append(dict(id=r["id"], family=r["family"], concept=c, tier=r["tier"],
                        d={k: _r(r["design"].get(k)) for k in DESIGN_KEYS if r["design"].get(k) is not None},
                        o={k: _r(v) for k, v in r["out"].items()}))
    return out


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

def corona_tracks(R, fr, thickness, spacing, Lc, V, polarity, U, T_in=35.0, sizes=(5e-6, 15e-6, 40e-6), n_each=16,
                  seed=0):
    """Mist drops traced through the resolved corona field of one disc stack
    (one periodic cell solved, tiled over the section); also the field map
    of that cell. Same particle physics as stonelab.ehd_particles."""
    import stonelab.ehd_particles as ep
    import watercap_ehd as ehd
    from stonelab.ehd import solve_corona
    pre = ehd.EHD_PRESETS["ehd_coarse"]
    n_per = max(1, int(round(Lc / spacing)))
    sec = ep.DiscStackSection(R=R, disc_radius=fr * R, disc_thickness=thickness, spacing=spacing, n_periods=n_per,
                              rel_air_density=293.15 / (T_in + 273.15), voltage=polarity * V * 1e3,
                              mobility=rom.ION_MOBILITY[polarity], edge_cells=pre["edge_cells"],
                              h_coarse=min(pre["h_coarse"], spacing / 12), growth=pre["growth"])
    g, es = sec.build()
    flow = ep.PipeFlow(U=U, R=R)
    ur, uz = ep.flow_around_emitters(es, flow)
    sol = solve_corona(es, sec.mobility, polarity, gas_velocity=(ur, uz))
    Emag = np.hypot(sol.E_r, sol.E_z)
    smp = ep.FieldSampler(g, {"Er": sol.E_r, "Ez": sol.E_z, "E": Emag, "n": np.abs(sol.rho_ion) / ep.E_CHARGE,
                              "ur": ur, "uz": uz})
    NR, NZ = 48, 40
    Rg, Zg = np.meshgrid((np.arange(NR) + 0.5) / NR * R, (np.arange(NZ) + 0.5) / NZ * spacing, indexing="ij")
    f = smp(Rg.ravel(), Zg.ravel())
    emap = np.round(np.asarray(f["E"]).reshape(NR, NZ) / 1e5, 2)          # kV/cm
    rng = np.random.default_rng(seed)
    a = np.repeat(np.array(sizes) / 2, n_each)
    r = np.tile(np.linspace(0.06, 0.97, n_each) * R, len(sizes))
    z = np.zeros_like(r)
    q = np.zeros_like(r)
    Zm = ep.cunningham(a) / (6 * np.pi * ep.MU_AIR * a)
    vs = (2 * ep.RHO_WATER * ep.G * a ** 2 / (9 * ep.MU_AIR)) * ep.cunningham(a)
    sdf = es.conductors[0].sdf
    L, s_ = n_per * spacing, spacing
    alive = np.ones(len(r), bool)
    fate = ["escaped"] * len(r)
    tracks = [[(0.0, float(r[i] / R))] for i in range(len(r))]
    h_r, h_z = np.diff(g.r_faces), np.diff(g.z_faces)
    dt_max = 0.25 * s_ / flow.U_max
    step = 0

    def Deff(rq, zq):
        d = flow.D_t(rq)
        y = np.maximum(sdf(rq, zq), 0.0)
        return d * (1 - np.exp(-y * flow.u_star / ep.NU_AIR / 26)) ** 2 + 1e-12

    while alive.any() and step < 120000:
        idx = np.flatnonzero(alive)
        zi = np.mod(z[idx], s_)
        ff = smp(r[idx], zi)
        vr = ff["ur"] + Zm[idx] * q[idx] * ff["Er"]
        vz = ff["uz"] + Zm[idx] * q[idx] * ff["Ez"] - vs[idx]
        hd = 1e-5
        D = Deff(r[idx], zi)
        dDr = (Deff(r[idx] + hd, zi) - Deff(np.maximum(r[idx] - hd, 0), zi)) / (r[idx] + hd - np.maximum(r[idx] - hd, 0))
        dDz = (Deff(r[idx], zi + hd) - Deff(r[idx], zi - hd)) / (2 * hd)
        ci = np.clip(np.searchsorted(g.r_faces, r[idx]) - 1, 0, g.nr - 1)
        cj = np.clip(np.searchsorted(g.z_faces, zi) - 1, 0, g.nz - 1)
        h = np.minimum(h_r[ci], h_z[cj])
        dt = np.minimum(dt_max, 0.4 * h / np.maximum(np.hypot(vr, vz) + np.sqrt(2 * D / dt_max), 1e-12))
        qn, _ = ep.field_charge_step(a[idx], q[idx], ff["E"], ff["n"], sec.mobility, polarity, dt)
        rn = np.abs(r[idx] + (vr + dDr + D / np.maximum(r[idx], 1e-9)) * dt + np.sqrt(2 * D * dt) * rng.standard_normal(len(idx)))
        zn = np.maximum(z[idx] + (vz + dDz) * dt + np.sqrt(2 * D * dt) * rng.standard_normal(len(idx)), 0.0)
        q[idx], r[idx], z[idx] = qn, rn, zn
        wall = rn >= R
        disc = (sdf(np.minimum(rn, R), np.mod(zn, s_)) < 0) & ~wall
        out = (zn >= L) & ~wall & ~disc
        for k, i in enumerate(idx):
            tracks[i].append((float(min(z[i], L)), float(min(r[i], R) / R)))
            if wall[k]:
                fate[i] = "wall"
            elif disc[k]:
                fate[i] = "disc"
        alive[idx[wall | disc | out]] = False
        step += 1
    tr = []
    for i, t in enumerate(tracks):
        t = np.array(t)
        keep = np.unique(np.concatenate([np.linspace(0, len(t) - 1, min(len(t), 160)).astype(int), [len(t) - 1]]))
        tr.append(dict(d_um=round(2e6 * a[i]), fate=fate[i], zr=np.round(t[keep], 4).tolist()))
    return dict(tracks=tr, emap=emap.tolist(), emap_R=R, emap_s=s_, current_uA_period=abs(sol.current_wall) * 1e6)


def scene(job):
    row, design = job
    c = dict(design)
    aux = dict(wc.BASE_AUX, ambient_T_C=25.0)
    t0 = time.time()
    cfd, pol_c = wr.build_cfd(c, aux, wr.RESOLVED_PRESETS["resolved_coarse"]["cfd"])
    res = cfd.solve()
    return scene_from_solve(row, c, cfd, res, t0)


def scene_from_solve(row, c, cfd, res, t0):
    """Scene dict from a converged spray-stage solve (re-traced with path recording) + corona tracks."""
    spray_paths = []
    if any(j.flow_kg_s > 0 for j in cfd.inj):
        cfd.trace(np.random.default_rng(3), cfd.gauss_field(cfd.src["rhoq"]), record=90)
        for p in cfd.paths:
            p = np.array(p)
            keep = np.unique(np.concatenate([np.linspace(0, len(p) - 1, min(len(p), 140)).astype(int), [len(p) - 1]]))
            spray_paths.append(np.round(p[keep], 4).tolist())
    f = res.fields
    zs = np.linspace(0, cfd.L, 140)
    def resample(F):  # (nr, nz) -> (nr, 140), rounded
        return np.round(np.array([np.interp(zs, cfd.zc, F[i]) for i in range(cfd.nr)]), 4).tolist()
    S = f["S"]
    liq = f["spray_liq"] if f["spray_liq"] is not None else np.zeros_like(S)
    fields = dict(r=np.round(cfd.rc, 5).tolist(), z=np.round(zs, 4).tolist(), T=resample(f["T"]),
                  S=resample(S), u=resample(f["u"]), v=resample(f["v"]), spray_g_m3=resample(liq * 1e3))
    R = c["tube_diameter_m"] / 2
    U = c["air_flow_m3_s"] / (np.pi * R ** 2)
    pol = 1 if c["corona_polarity"] >= 0 else -1
    cor = corona_tracks(R, c["disc_diameter_frac"], c["disc_thickness_mm"] * 1e-3, c["disc_spacing_m"],
                        c["corona_section_length_m"], c["corona_voltage_kV"], pol, U)
    inj = [dict(z=j.z, r=j.r_ring, axis_deg=j.axis_angle_deg, cone_deg=j.cone_half_angle_deg) for j in cfd.inj]
    return dict(id=row["id"], concept=row["concept"], d=row["d"], o=row["o"],
                geom=dict(R=R, Ls=c["spray_section_length_m"], Lc=c["corona_section_length_m"],
                          disc_r=c["disc_diameter_frac"] * R, disc_t=c["disc_thickness_mm"] * 1e-3,
                          spacing=c["disc_spacing_m"], n_discs=max(1, int(round(c["corona_section_length_m"] / c["disc_spacing_m"]))),
                          nozzles=inj),
                spray_paths=spray_paths, fields=fields, corona=cor, seconds=round(time.time() - t0, 1))


def curate(rows, top=1):
    """Top-N (most water recovered, non-arcing, closed) designs per concept + baseline + best disc geometry."""
    pick = {}
    for r in rows:
        o = r["o"]
        if (o.get("arc") or -1) <= 0 or abs(o.get("closure") or 0) > 0.02:
            continue
        if r["concept"] != "corona" and r["tier"] != "resolved":
            continue
        pick.setdefault(r["concept"], []).append(r)
    chosen = [r for v in pick.values() for r in sorted(v, key=lambda r: -r["o"]["recovered"])[:top]]
    base = dict(wc.BRAD, corona_voltage_kV=40.0, id="baseline-40kV", family="brad_variants")
    best_disc = dict(wc.BRAD, disc_thickness_mm=0.1, disc_diameter_frac=0.6, disc_spacing_m=0.2, corona_voltage_kV=30.0,
                     id="disc-geometry-30kV", family="disc")
    extra = []
    for d in (base, best_disc):
        extra.append(dict(id=d["id"], family=d["family"], concept="corona", tier="scene-only",
                          d={k: _r(d.get(k)) for k in DESIGN_KEYS if d.get(k) is not None}, o={}))
    return chosen + extra


def full_design(row):
    """The campaign's full design dict for a row (scene-only rows rebuild it)."""
    if row["tier"] == "scene-only":
        d = dict(wc.BRAD)
        d.update({k: v for k, v in row["d"].items()})
        d.update(id=row["id"], family=row["family"])
        return d
    for rn in ("v2_resolved", "v2_rerun", "v1"):
        for r in wc.load(rn):
            if r["id"] == row["id"]:
                return r["design"]
    raise KeyError(row["id"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--designs-only", action="store_true")
    ap.add_argument("--top", type=int, default=1, help="scenes per concept")
    ap.add_argument("--force", action="store_true", help="rebuild scenes that already exist")
    a = ap.parse_args()
    (OUT / "scenes").mkdir(parents=True, exist_ok=True)
    full = {}
    rows = collect(full)
    # full-precision inputs, so the viewer's in-browser compute reproduces campaign records exactly
    json.dump(full, open(OUT / "design_inputs.json", "w"), separators=(",", ":"))
    json.dump(dict(concepts=CONCEPT_NAME, designs=rows, generated=time.strftime("%Y-%m-%d %H:%M"),
                   aux=wc.BASE_AUX), open(OUT / "designs.json", "w"), separators=(",", ":"))
    print(len(rows), "designs ->", OUT / "designs.json")
    if a.designs_only:
        sys.exit(0)
    cur = curate(rows, a.top)
    index = [dict(id=r["id"], concept=r["concept"]) for r in cur if (OUT / "scenes" / f"{r['id']}.json").exists()]
    todo = [r for r in cur if a.force or not (OUT / "scenes" / f"{r['id']}.json").exists()]
    if a.force:
        index = []
    jobs = [(r, full_design(r)) for r in todo]
    print(len(cur), "scenes wanted,", len(jobs), "to build", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        for sc in ex.map(scene, jobs):
            json.dump(sc, open(OUT / "scenes" / f"{sc['id']}.json", "w"), separators=(",", ":"))
            index.append(dict(id=sc["id"], concept=sc["concept"]))
            json.dump(index, open(OUT / "scenes" / "index.json", "w"))  # viewer picks scenes up as they land
            print("scene", sc["id"], sc["seconds"], "s", flush=True)
    json.dump(index, open(OUT / "scenes" / "index.json", "w"))

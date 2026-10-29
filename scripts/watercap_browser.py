"""In-browser compute for the watercap viewer page (viewer/src/pages/watercap.astro).

The page's "Compute" button runs THIS code -- the campaign's own
`resolved_coarse` tier (watercap_resolved.simulate: spray-stage RANS +
parcels, then the resolved corona march) and the exporter's scene builder --
under Pyodide (CPython + numpy + scipy compiled to WebAssembly) in a web
worker. Same source, same seeds, so a design from the campaign reproduces its
campaign record; an edited design gets the same tier it would have had in the
campaign.

    python3 scripts/watercap_browser.py --bundle   # -> viewer/public/data/watercap/pybundle.json
    python3 scripts/watercap_browser.py --check charged_spray-0100   # native run of compute()

The bundle is the Python sources the worker writes into Pyodide's filesystem;
re-run --bundle after changing any of them.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STONELAB = HERE.parents[1] / "StoneLab" / "src" / "stonelab"
VEGA_FILES = ["watercap_oracle.py", "watercap_ehd.py", "watercap_resolved.py", "watercap_campaign.py",
              "export_watercap_viewer_data.py", "watercap_browser.py"]
STONELAB_FILES = ["spray_tower.py", "spray_cfd.py", "ehd.py", "ehd_particles.py", "gridsolve.py", "corona_air.py", "ehd_flow.py",
                  "corona_onset.py", "spray_sst.py", "ccn.py"]  # + their relative imports
# the grid solves' kernels (stonelab.gridsolve finds them as a sibling checkout): kernels.wasm under Pyodide
WCK = HERE.parents[1] / "wasm-compute-kit" / "python" / "wck"
WCK_FILES = ["__init__.py", "_backend.py", "_kernels_wasm.py", "particles.py", "spray.py"]


def compute(design_json, row_json="{}"):
    """JSON in (design overrides on top of the campaign baseline; viewer row
    for id/concept), JSON out: the full simulate payload, the viewer's outcome
    row and a 3D scene."""
    import watercap_campaign as wc
    import watercap_resolved as wr
    import export_watercap_viewer_data as ex
    c = dict(wc.BRAD)
    c.update(json.loads(design_json))
    row = dict(id="custom", concept="corona", d={}, o={})
    row.update(json.loads(row_json))
    aux = dict(wc.BASE_AUX, ambient_T_C=25.0)
    from stonelab import gridsolve
    gridsolve.reset()
    t0 = time.time()
    print("@stage spray", flush=True)
    keep = {}
    ok, pay = wr.simulate(c, aux, "resolved_coarse", keep=keep, verbose=True)
    if not ok:
        return json.dumps(dict(ok=False, error=str(pay), seconds=time.time() - t0))
    t_sim = time.time() - t0
    print("@stage paths", flush=True)
    o = {k: ex._r(v) for k, v in ex.resolved_out(pay).items()}
    row.update(o=o, d={k: ex._r(c.get(k)) for k in ex.DESIGN_KEYS if c.get(k) is not None})
    sc = ex.scene_from_solve(row, c, keep["cfd"], keep["res"], time.time())
    return json.dumps(dict(ok=True, o=o, res=wc._jsonable(pay), scene=sc, seconds=time.time() - t0, sim_seconds=t_sim,
                           solver=solver_info()))


def workbench(design_json, aux_json, preset="resolved_coarse", seed=0):
    """The Workbench's resolved tier (watercap_web/workbench.src.html): this
    campaign's own simulate() on a design + operating point, the result in the
    shape the page draws (metrics, spray fields, corona periods and fields).
    A stored design passed verbatim reproduces its campaign record."""
    import numpy as np
    import watercap_campaign as wc
    import watercap_resolved as wr
    c = dict(wc.BRAD)
    c.update(json.loads(design_json))
    aux = dict(wc.BASE_AUX)
    aux.update(json.loads(aux_json))
    from stonelab import gridsolve
    gridsolve.reset()          # one run's solver memory and stats at a time
    t0 = time.time()
    print("@stage spray", flush=True)
    keep = {}
    ok, pay = wr.simulate(c, aux, preset, seed=int(seed), keep=keep, verbose=True)
    if not ok:
        return json.dumps(dict(ok=False, error=str(pay), seconds=time.time() - t0, solver=solver_info()))
    f32 = lambda a: [float(x) for x in np.asarray(a, dtype=np.float32).ravel()]
    res, sr, cp = keep["res"], keep.get("sec"), keep.get("cp")
    F, d = res.fields, res.diag
    fields = dict(nr=len(F["rc"]), nz=len(F["zc"]), rc=f32(F["rc"]), zc=f32(F["zc"]), rf=f32(res.handoff["rf"]),
                  **{k: f32(F[k]) for k in ("T", "w", "S", "l", "u", "v")},
                  sprayLiq=f32(F["spray_liq"]) if F.get("spray_liq") is not None else None)
    corona = None
    if sr is not None:
        nm = sr.no_mist
        corona = dict(currents=[float(x) for x in sr.current_per_period],
                      periods=[dict(current=float(p["current"]), toParticles=float(p["to_particles"]),
                                    wall=[float(x) for x in p["wall"]]) for p in sr.periods],
                      collected=[float(x) for x in sr.collected_frac], escaped=[float(x) for x in sr.escaped_frac],
                      emitter=[float(x) for x in sr.emitter_frac], noMistCurrent=float(sr.current_no_mist),
                      wallField=[float(x) for x in sr.wall_field_kV_cm],
                      cell=dict(Er=f32(nm.E_r), Ez=f32(nm.E_z), rho=f32(nm.rho_ion), phi=f32(nm.phi)) if nm is not None else None)
    classes = dict(radii=[float(x) for x in cp["radii"]], nflow=[float(x) for x in cp["nflow"]], kinds=list(cp["kinds"]),
                   massTop=[float(x) for x in cp["mass_top"]]) if cp is not None else None
    metrics = {k: v for k, v in wc._jsonable(pay).items() if k not in ("spray_stage", "cfd_diag")}
    return json.dumps(dict(ok=True, metrics=metrics, fields=fields, corona=corona, classes=classes,
                           sprayDiag=dict(nOuter=int(d["n_outer"]), waterResidual=float(d["water_residual"]),
                                          recirculating=bool(d["recirculating"])),
                           seconds=time.time() - t0, solver=solver_info()))


def solver_info():
    """Which kernels did the grid solves: "native" / "wasm" (wasm-compute-kit) or "scipy" (the fallback)."""
    from stonelab import gridsolve
    w = gridsolve.wck()
    tot = {}
    for st in gridsolve.stats().values():
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
    b = w.backend() if w else None
    heap = b.heap_bytes() if b is not None and hasattr(b, "heap_bytes") else None
    return dict(kernels=b.kind if b else "scipy", heap_MB=round(heap / 2 ** 20) if heap else None,
                threads=b.threads if b else 1, **tot)


def bundle(out=HERE.parent / "viewer" / "public" / "data" / "watercap" / "pybundle.json"):
    """Write the bundle to `out` (None: return it)."""
    files = {f"vega/scripts/{f}": (HERE / f).read_text() for f in VEGA_FILES}
    # stonelab's own __init__ imports the whole package (flow, elastic, ...); the worker needs four modules
    files["StoneLab/src/stonelab/__init__.py"] = ""
    files.update({f"StoneLab/src/stonelab/{f}": (STONELAB / f).read_text() for f in STONELAB_FILES})
    # runstore (run recording) needs fcntl, which Pyodide lacks; the browser never records, so its
    # solver hook is the pass-through it is natively when recording is off
    files["StoneLab/src/stonelab/runstore.py"] = "def hook(solver):\n    return lambda fn: fn\n"
    files.update({f"wasm-compute-kit/python/wck/{f}": (WCK / f).read_text() for f in WCK_FILES})
    b = dict(files=files, built=time.strftime("%Y-%m-%d %H:%M"))
    if out is None:
        return b
    json.dump(b, open(out, "w"))
    print(len(files), "files ->", out, f"({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--bundle"]:
        bundle()
    elif sys.argv[1:2] == ["--workbench"]:
        # native run of workbench() on a stored Workbench row: python3 scripts/watercap_browser.py --workbench <id>
        sys.path.insert(0, str(HERE))
        rows = json.load(open(HERE.parent / "watercap_web" / "designs.json"))["rows"]
        row = next(r for r in sorted(rows, key=lambda r: r["set"] == "v1") if r["id"] == sys.argv[2])  # a fully resolved record first
        r = json.loads(workbench(json.dumps(row["d"]), json.dumps(row["a"])))
        print(json.dumps(dict(ok=r["ok"], error=r.get("error"), seconds=r["seconds"], solver=r["solver"],
                              out=r["metrics"]["outlet_total_water_frac"] if r["ok"] else None, stored=row["r"]["out"])))
    elif sys.argv[1:2] == ["--check"]:
        sys.path.insert(0, str(HERE))
        full = json.load(open(HERE.parent / "viewer" / "public" / "data" / "watercap" / "design_inputs.json"))
        did = sys.argv[2]
        r = json.loads(compute(json.dumps(full[did]), json.dumps(dict(id=did))))
        print(json.dumps(dict(ok=r["ok"], o=r.get("o"), seconds=r["seconds"], solver=r.get("solver")), indent=1))

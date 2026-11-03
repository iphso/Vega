"""Stage the blanket simulator: page + engine bundle + gzipped cross-section packs +
a design library with OpenMC reference values where they exist.

    python3 blanket_web/build.py [--out DIR]     (default blanket_web/dist)

Engine: ~/Projects/StoneLab-blanket/apps/blanket/dist/blanket-engine.js (override
BLANKET_ENGINE); build it with `node apps/blanket/build.mjs` there.
Packs: blanket/neutpack/VITAMIN-J-175_hist64[_fw_material-X | _li6_pct-Y].
"""
import argparse, gzip, json, os, re, shutil
from pathlib import Path

H = Path(__file__).resolve().parent
V = H.parent
ENGINE = Path(os.environ.get("BLANKET_ENGINE", Path.home() / "Projects/StoneLab-blanket/apps/blanket/dist/blanket-engine.js"))
GS = "VITAMIN-J-175_hist64"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(H / "dist"))
out = Path(ap.parse_args().out)
(out / "packs").mkdir(parents=True, exist_ok=True)

# ---- packs
packs = []
for d in sorted((V / "blanket/neutpack").glob(f"{GS}*")):
    tag = d.name[len(GS):]
    if tag.endswith("_ng"):
        continue  # coupled n-gamma twin of a pack (blanket/photon/export_photon.py): staged in its place below
    fw, li6, breeder = "W", 90.0, None
    if m := re.match(r"_fw_material-(\w+)$", tag):
        fw = m.group(1)
    elif m := re.match(r"_li6_pct-([\d.]+)$", tag):
        li6 = float(m.group(1))
    elif m := re.match(r"_breeder-(\w+)$", tag):
        # other breeder concepts at their usual enrichment (arc_model.BREEDERS)
        breeder = m.group(1)
        li6 = {"flibe": 90.0, "flinak": 90.0, "pbli": 90.0, "li": 7.5, "hcpb": 60.0}[breeder]
    elif tag:
        continue
    dst = out / "packs" / (tag.lstrip("_") or "base")
    dst.mkdir(exist_ok=True)
    ng = d.with_name(d.name + "_ng")
    if (ng / "manifest.json").exists() and not os.environ.get("BLANKET_NO_NG"):
        d = ng  # same neutron data + photon extension (old readers ignore it)
    shutil.copy(d / "manifest.json", dst / "manifest.json")
    raw = (d / "data.bin").read_bytes()
    gz = gzip.compress(raw, 9, mtime=0)
    (dst / "data.bin.gz").write_bytes(gz)
    # vessel dpa / He responses for this pack (blanket/damage, session sl8rv-41's sidecar)
    side = V / "blanket/damage/runs" / f"VITAMIN-J-175{tag}" / "damage_sidecar.json"
    if side.exists():
        shutil.copy(side, dst / "damage.json")
    packs.append({"dir": dst.name, "fw": fw, "li6": li6, "bytes": len(gz), "damage": side.exists(), **({"breeder": breeder} if breeder else {})})
    print(f"pack {dst.name}: fw={fw} li6={li6} {len(gz)/1e6:.2f} MB gz{' (n-gamma)' if d.name.endswith('_ng') else ''}")
json.dump({"groups": GS, "packs": packs}, open(out / "packs" / "packs.json", "w"), indent=1)
# activation / decay heat responses (blanket/activation, baseline materials and spectra)
act = V / "blanket/activation/runs/VITAMIN-J-175_ww/activation_sidecar.json"
if act.exists():
    shutil.copy(act, out / "packs" / "activation.json")

# LIBRA experiment packs (raster + point source), for the validation page's live runs
for d in sorted((V / "blanket/neutpack").glob("libra_*")):
    dst = out / "packs" / d.name
    dst.mkdir(exist_ok=True)
    shutil.copy(d / "manifest.json", dst / "manifest.json")
    (dst / "data.bin.gz").write_bytes(gzip.compress((d / "data.bin").read_bytes(), 9, mtime=0))
    (dst / "raster.u8.gz").write_bytes(gzip.compress((d / "raster.u8").read_bytes(), 9, mtime=0))
    print(f"pack {d.name}")


# ---- design library with OpenMC references
sweep = json.load(open(V / "blanket/runs/fig30_sweep.json"))
base = json.load(open(V / "blanket/runs/arc_baseline/result.json"))
def ref_fw(mat, cm):
    k = f"{mat}_{cm:.1f}"
    return {"TBR": sweep[k]["mean"], "TBR_sd": sweep[k]["std"], "source": "OpenMC CE, Fig. 30 sweep (300k histories)"} if k in sweep else None
LIB = [
    ("ARC baseline", "Baseline", {}, {"TBR": base["TBR"]["mean"], "TBR_sd": base["TBR"]["std"], "heating_MW": {k: v["mean"] for k, v in base["heating_MW"].items()}, "source": "OpenMC CE with photons (1M histories)"}),
    ("Tungsten wall, 0.4 cm", "First wall", {"geometry": {"fw_cm": 0.4}}, ref_fw("W", 0.4)),
    ("Tungsten wall, 2 cm", "First wall", {"geometry": {"fw_cm": 2.0}}, ref_fw("W", 2.0)),
    ("Beryllium wall", "First wall", {"fwMaterial": "Be"}, ref_fw("Be", 1.0)),
    ("Carbon-fibre wall", "First wall", {"fwMaterial": "CFC"}, ref_fw("CFC", 1.0)),
    ("Bare Inconel wall", "First wall", {"fwMaterial": "Inconel"}, ref_fw("Inconel", 1.0)),
    ("No Be multiplier", "Vessel", {"geometry": {"be_cm": 0}}, None),
    ("Wide cooling channel", "Vessel", {"geometry": {"channel_cm": 4}}, None),
    ("Natural lithium", "Breeder", {"li6Pct": 7.5}, None),
    ("30% Li-6", "Breeder", {"li6Pct": 30}, None),
    ("60% Li-6", "Breeder", {"li6Pct": 60}, None),
    ("Thin inboard tank", "Tank", {"geometry": {"blanket_in_cm": 10}}, None),
    ("Deep outboard tank", "Tank", {"geometry": {"blanket_out_cm": 140}}, None),
    ("Larger machine (R0 3.6 m)", "Machine", {"geometry": {"R0": 363}}, None),
]
# other breeder concepts in the ARC geometry, with coolant presets (inlet above each one's
# melting point, flows that carry ~the same heat): only where their packs exist
packs_have = {p.get("breeder") for p in packs}
BREEDER_LIB = [
    ("PbLi breeder", {"breeder": "pbli", "li6Pct": 90, "inletK": 600, "channelVelocity": 0.5, "bulkFlow": 1.1}),
    ("Liquid lithium", {"breeder": "li", "li6Pct": 7.5, "inletK": 600, "channelVelocity": 1, "bulkFlow": 1.0}),
    ("FLiNaK salt", {"breeder": "flinak", "li6Pct": 90, "inletK": 800}),
    ("He-cooled pebble bed", {"breeder": "hcpb", "li6Pct": 60, "inletK": 573, "channelVelocity": 60, "bulkFlow": 70}),
]
LIB += [(n, "Breeders", d, None) for n, d in BREEDER_LIB if d["breeder"] in packs_have]
# stellarators: a conformal build around a Fourier first wall (StoneLab neutron/conformal.ts)
surfaces = {}
for f in sorted((V / "blanket/stellarators").glob("*.json")):
    b = json.load(open(f))
    surfaces[b["id"]] = {"label": b["label"], "nfp": b["nfp"], "rc": b["r_cos"], "zs": b["z_sin"], "source": b["source"]}
json.dump(surfaces, open(out / "surfaces.json", "w"))
STELL = [
    ("Stellarator, 3 periods", {"machine": "stellarator", "surface": "vega-p1-nfp3", "majorRadiusM": 6.8, "fusionPowerMW": 1000}),
    ("Stellarator, 5 periods", {"machine": "stellarator", "surface": "bench-nfp5", "majorRadiusM": 8.0, "fusionPowerMW": 1000}),
]
LIB += [(n, "Stellarators", d, None) for n, d in STELL if d["surface"] in surfaces]
lib = [{"id": f"lib-{i}", "name": n, "family": f, "design": d, "reference": r} for i, (n, f, d, r) in enumerate(LIB)]
json.dump(lib, open(out / "designs.json", "w"), indent=1)

# ---- validation data: measured, published, OpenMC (ours) and browser numbers in one file
R = V / "blanket/runs"
val = {
    "benchmarks": json.load(open(V / "blanket/benchmarks.json")),
    "browser": json.load(open(R / "validation_browser.json")),
    "fig30_openmc": sweep,
    "arc_openmc": base,
    "libra_openmc_full": {
        "BABY-1L": json.load(open(R / "baby1l/tbr.json")),
        "FLiBe-1L": json.load(open(R / "flibe1l/tbr.json")),
    },
    "baby_no_vault": json.load(open(R / "baby1l_novault/tbr.json")),
    "libra_release": {},
    "combos": json.load(open(R / "combo_check.json")) if (R / "combo_check.json").exists() else [],
}
# LIBRA's counted tritium (IV + OV bubblers, cumulative Bq) per run, with the run's own
# neutron rate, irradiation schedule and fitted transfer coefficients
for rd in sorted((V / "blanket/ext").glob("*-1L-run-*")):
    pd = json.load(open(rd / "data/processed_data.json"))
    c = pd["cumulative_tritium_release"]
    # IV and OV bubblers are sampled on their own schedules: put both cumulative series
    # on the union of sampling times (linear between samples, 0 before the first)
    import numpy as np
    ser = {k: (np.array(c[k]["sampling_times"]["value"]), np.array(c[k]["total"]["value"])) for k in ("IV", "OV")}
    days = sorted(set(np.concatenate([t for t, _ in ser.values()]).tolist()))
    tot = [float(sum(np.interp(d, t, v, left=0.0) for t, v in ser.values())) for d in days]
    val["libra_release"][rd.name] = {
        "days": days, "measured_Bq": tot,
        "neutron_rate": pd["neutron_rate_used_in_model"]["value"],
        "irradiations_s": [[i["start_time"]["value"], i["stop_time"]["value"]] for i in pd["irradiations"]],
        "k_top": pd["k_top"]["value"], "k_wall": pd["k_wall"]["value"],
        "radius_m": pd["modelled_baby_radius"]["value"] / 100, "height_m": pd["modelled_baby_height"]["value"] * 1e3 / 1e2,
        "measured_TBR": pd["measured_TBR"]["value"],
    }
# ---- checks of the generic-tool additions (breeders, 3-D, ports, sensitivities, MHD)
def jl(path):
    path = Path(path)
    return json.load(open(path)) if path.exists() else None
ext = {"breeders": [], "conformal": jl(R / "conformal_check.json"), "ports": jl(R / "ports_check.json"),
       "sens": jl(R / "sens_check.json"), "breeders_browser": jl(R / "breeder_check.json"),
       "mhd": jl(Path.home() / "Projects/StoneLab-blanket/packages/physics/src/mhd/validation.json")}
ph = jl(Path.home() / "runs/blanket/photon/photon_check.json")
if ph:
    ext["photon"] = [{"name": c["name"], "rows": {k: {"openmc": r["openmc"][0], "factor": r.get("factor_err"), "coupled": r.get("coupled_err")} for k, r in c["rows"].items()}} for c in ph["cases"]]
fc = jl(V / "blanket/fuelcycle/table6.json")
if fc:
    ext["fuelcycle"] = [{"case": k, "tbr": v["TBR_r"], "tbr_paper": v["paper_TBR_r"], "isu": v["I_startup_kg"], "isu_paper": v["paper_I_startup_kg"]} for k, v in fc.items()]
# integral benchmarks (blanket/bench/*): C/E per case, ND spread from the joint samples, and the
# assimilation onto the ARC TBR (blanket/assim)
ext["bench"] = []
BENCH_EXCLUDE = {"oktavian:be97"}  # internally inconsistent (bench/oktavian/NOTES.md); assim.py drops it too
for bn in ("oktavian", "fns"):
    br = jl(V / f"blanket/bench/{bn}/results.json")
    if not br:
        continue
    cases = []
    for cid, c in br["cases"].items():
        if f"{bn}:{cid}" in BENCH_EXCLUDE:
            continue
        rs = [r for r in c["responses"] if r.get("C") and r.get("E")]
        cases.append({"case": cid, "kinds": sorted({r["id"].split("@")[0].split("_")[0] for r in rs}),
                      "tbr": next(([r["E"], r["E_rel_unc"], r["C"], r.get("geom_rel_unc")] for r in rs if r["id"] == "tbr"), None),
                      "tpr": [[r["id"][:2], r["C"] / r["E"]] for r in rs if r["id"] in ("T6_lsc_int", "T7_lsc_int")],
                      "ce": [r["C"] / r["E"] for r in rs if r["id"] != "tbr" and not r["id"].startswith("leak") and not r["id"].endswith("_int")], "leak_ce": [r["C"] / r["E"] for r in rs if r["id"].startswith("leak")]})
    smp = br.get("samples", {}).get("all", {})
    ndl = smp.get("nd_rel_sd") or [smp.get("stats", {}).get(i, {}).get("nd_rel_sd") for i in smp.get("response_ids", [])]
    nd = [x for i, x in zip(smp.get("response_ids", []), ndl) if x is not None and ":al@" not in i]  # Al(n,a): unperturbed control
    tnd = [x for i, x in zip(smp.get("response_ids", []), ndl) if x is not None and i.endswith(":tbr")]
    ext["bench"].append({"name": bn, "cases": cases, "nd": [min(nd), max(nd)] if nd else None, "nd_tbr": [min(tnd), max(tnd)] if tnd else None,
                         "assim": (jl(V / f"blanket/assim/{bn}.json") or {}).get(bn)})
ext["bench_combined"] = (jl(V / "blanket/assim/combined.json") or {}).get("oktavian+fns")
uq = jl(V / "blanket/uq/results.json")
if uq:
    ext["uq"] = {k: {"tbr": [v["TBR"]["nd_rsd_pct"], *v["TBR"]["nd_rsd_pct_68"]], "resolved": v["TBR"]["resolved"],
                     "heat": v["heating_total_MW"]["nd_rsd_pct"], "n": v["TBR"]["n"], "nominal": v["TBR"]["nominal"]}
                 for k, v in uq.items() if isinstance(v, dict) and isinstance(v.get("TBR"), dict)}
for d in sorted((R / "mgxs").glob(f"{GS}_breeder-*")):
    g = jl(d / "gate1.json")
    if g:
        ext["breeders"].append({"breeder": d.name.split("breeder-")[1], "openmc_ce": g["ce"]["TBR"], "openmc_mg": g["mg"]["TBR"]})
val["extended"] = ext
json.dump(val, open(out / "validation.json", "w"), indent=1)

shutil.copy(ENGINE, out / "blanket-engine.js")
# the 3-D cutaway view (StoneLab apps/blanket-view, bare chrome) and the OpenMC result it shows as reference
VIEW = Path(os.environ.get("BLANKET_VIEW", Path.home() / "Projects/StoneLab-blanket-app/apps/blanket-view/dist/blanket-view.js"))
if VIEW.exists():
    shutil.copy(VIEW, out / "blanket-view.js")
    shutil.copy(V / "blanket/runs/arc_baseline/result.json", out / "arc_result.json")
for f in ["index.html", "app.js", "validation.html", "validation.js", "reel.html"]:
    shutil.copy(H / f, out / f)
# the reel: cinema engine + script + baked data (StoneLab packages/physics/scripts/reel-bake.ts), three.js
for sub in ["cinema", "vendor"]:
    (out / sub).mkdir(exist_ok=True)
    for f in (H / sub).iterdir():
        shutil.copy(f, out / sub / f.name)
print(f"staged {out}")

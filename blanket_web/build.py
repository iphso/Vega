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
    fw, li6 = "W", 90.0
    if m := re.match(r"_fw_material-(\w+)$", tag):
        fw = m.group(1)
    elif m := re.match(r"_li6_pct-([\d.]+)$", tag):
        li6 = float(m.group(1))
    elif tag:
        continue
    dst = out / "packs" / (tag.lstrip("_") or "base")
    dst.mkdir(exist_ok=True)
    shutil.copy(d / "manifest.json", dst / "manifest.json")
    raw = (d / "data.bin").read_bytes()
    gz = gzip.compress(raw, 9)
    (dst / "data.bin.gz").write_bytes(gz)
    packs.append({"dir": dst.name, "fw": fw, "li6": li6, "bytes": len(gz)})
    print(f"pack {dst.name}: fw={fw} li6={li6} {len(gz)/1e6:.2f} MB gz")
json.dump({"groups": GS, "packs": packs}, open(out / "packs" / "packs.json", "w"), indent=1)

# LIBRA experiment packs (raster + point source), for the validation page's live runs
for d in sorted((V / "blanket/neutpack").glob("libra_*")):
    dst = out / "packs" / d.name
    dst.mkdir(exist_ok=True)
    shutil.copy(d / "manifest.json", dst / "manifest.json")
    (dst / "data.bin.gz").write_bytes(gzip.compress((d / "data.bin").read_bytes(), 9))
    (dst / "raster.u8.gz").write_bytes(gzip.compress((d / "raster.u8").read_bytes(), 9))
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
}
json.dump(val, open(out / "validation.json", "w"), indent=1)

shutil.copy(ENGINE, out / "blanket-engine.js")
for f in ["index.html", "app.js", "validation.html", "validation.js"]:
    shutil.copy(H / f, out / f)
print(f"staged {out}")

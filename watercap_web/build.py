"""Inline the model, GPU kernel, corona fit, stored designs and the resolved tier
(pyworker.js + the campaign's Python sources) into workbench.html.
Refresh the stored designs first with scripts/export_watercap_workbench.py.
The published artifact also carries Pyodide's files under pyodide/:
    python3 watercap_web/build.py --pyodide-from <a pyodide 0.27.7 dist with numpy/scipy/openblas>
stages them into watercap_web/pyodide/ (checked against pyodide-lock.json)."""
import json
import sys
from pathlib import Path

H = Path(__file__).parent
s = (H / "workbench.src.html").read_text()
fit = json.dumps(json.load(open(H / "corona_fit.json")), separators=(",", ":"))
designs = (H / "designs.json").read_text()
# the resolved tier: the campaign's Python itself (scripts/watercap_browser.py bundles it with
# stonelab and wasm-compute-kit's Python binding), run under Pyodide by pyworker.js
sys.path.insert(0, str(H.parent / "scripts"))
import watercap_browser  # noqa: E402
import watercap_site  # noqa: E402
site = dict(months=watercap_site.MONTHS, depths={"surface": "surface", "20m": "20 m", "30m": "30 m"},
            climate={m: dict(air_T_C=c[0], exhaust_T_C=round(watercap_site.exhaust_T_C(m), 2), interpolated=m not in watercap_site.EXHAUST_REF,
                             sea={k: c[i] for k, i in watercap_site.SEA_DEPTHS.items()}) for m, c in watercap_site.CLIMATE.items()})
esc = lambda t: t.replace("</script", "<\\/script")
resworker = esc((H / "pyworker.js").read_text())
# the 3D device tier: StoneLab's solver3d bundled as a module worker (StoneLab apps/device3d, `node apps/device3d/build.mjs`)
import os  # noqa: E402
D3 = Path(os.environ.get("DEVICE3D_WORKER", H.parent.parent / "StoneLab-3d" / "apps" / "device3d" / "dist" / "device3d-worker.js"))
device3d = esc(D3.read_text())
pybundle = esc(json.dumps(watercap_browser.bundle(None), separators=(",", ":")))
for key, val in [("/*__WCMODEL__*/", (H / "wcmodel.js").read_text()), ("/*__WCGPU__*/", (H / "wcgpu.js").read_text()),
                 ("/*__FIT__*/null", fit), ("/*__DESIGNS__*/null", designs), ("/*__SITE__*/null", json.dumps(site, separators=(",", ":"))), ("/*__RESWORKER__*/", resworker), ("/*__DEVICE3D__*/", device3d),
                 ("/*__PYBUNDLE__*/null", pybundle), ("/*__VIEW3D__*/", esc((H / "view3d.js").read_text()))]:
    assert s.count(key) == 1, key
    s = s.replace(key, val)
(H / "workbench.html").write_text(s)

PYODIDE_CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
PYODIDE_PACKAGES = ["numpy", "scipy", "openblas"]


def stage_pyodide(src, dst=H / "pyodide"):
    import hashlib
    import shutil
    src = Path(src)
    lock = json.load(open(src / "pyodide-lock.json"))
    assert lock["info"]["version"] == "0.27.7", lock["info"]["version"]
    dst.mkdir(exist_ok=True)
    for f in PYODIDE_CORE:
        shutil.copyfile(src / f, dst / f)
    for p in PYODIDE_PACKAGES:
        e = lock["packages"][p]
        b = (src / e["file_name"]).read_bytes()
        assert hashlib.sha256(b).hexdigest() == e["sha256"], f"{e['file_name']}: sha256 does not match the lockfile"
        (dst / e["file_name"]).write_bytes(b)
    files = sorted(x.name for x in dst.iterdir())
    print("pyodide/:", ", ".join(f"{x} {(dst / x).stat().st_size / 1e6:.1f} MB" for x in files))
    return files


def stage_site(dst):
    """The page as <dst>/index.html with pyodide/ beside it (iph.so/newwater: site repo public/newwater)."""
    import shutil
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(H / "workbench.html", dst / "index.html")
    src = H / "pyodide"
    if not src.is_dir():
        raise SystemExit("no pyodide/: run with --pyodide-from first")
    (dst / "pyodide").mkdir(exist_ok=True)
    (dst / "vendor").mkdir(exist_ok=True)
    for f in sorted((H / "vendor").iterdir()):         # three.js (MIT), for the 3D view
        shutil.copyfile(f, dst / "vendor" / f.name)
    for f in sorted(src.iterdir()):
        shutil.copyfile(f, dst / "pyodide" / f.name)
    print("site:", dst, sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6, "MB")


if "--pyodide-from" in sys.argv:
    stage_pyodide(sys.argv[sys.argv.index("--pyodide-from") + 1])
if "--site" in sys.argv:
    stage_site(sys.argv[sys.argv.index("--site") + 1])
print(f"workbench.html {len(s) / 1e3:.0f} kB")

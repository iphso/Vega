"""NJOY RECONR -> ERRORR: relative multigroup covariances (MF33, incl. MF32 resonance
contributions) for the UQ nuclides, on the GROUPS structure below.

Run in the openmc container:  blanket/uq/njoy_cov.py [nuclide ...]
ENDF inputs:  /runs/endf/n-ZZZ_Sy_AAA.endf (ENDF/B-VIII.0 neutron sublibrary, NNDC)
Outputs:      /runs/cov/<nuc>/tape22 (ERRORR covariance tape, ASCII) + njoy log
"""
import os, sys, subprocess, shutil

ENDF = {
    "Li6": "n-003_Li_006.endf", "Li7": "n-003_Li_007.endf", "Be9": "n-004_Be_009.endf",
    "F19": "n-009_F_019.endf", "Ni58": "n-028_Ni_058.endf",
    "W182": "n-074_W_182.endf", "W183": "n-074_W_183.endf", "W184": "n-074_W_184.endf",
    "W186": "n-074_W_186.endf",
}
# 35 groups (eV): coarse below 100 keV (1/v + few resonances), ~0.5-1 MeV above, finer at 14 MeV
GROUPS = [1e-5, 0.1, 0.5, 1.0, 10.0, 100.0, 1e3, 1e4, 3e4, 1e5, 2e5, 3e5, 5e5, 7e5, 1e6,
          1.5e6, 2e6, 2.5e6, 3e6, 3.5e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 10e6, 11e6, 12e6,
          13e6, 13.5e6, 14e6, 14.5e6, 15e6, 17e6, 20e6]
ROOT = os.environ.get("UQ_ROOT", "/runs")


def mat_of(path):
    with open(path) as f:
        f.readline()
        return int(f.readline()[66:70])


def run(nuc):
    src = os.path.join(ROOT, "endf", ENDF[nuc])
    d = os.path.join(ROOT, "cov", nuc)
    os.makedirs(d, exist_ok=True)
    shutil.copy(src, os.path.join(d, "tape20"))
    mat = mat_of(src)
    g = " ".join(f"{e:.6e}" for e in GROUPS)
    inp = f"""reconr
20 21 /
'{nuc} pendf' /
{mat} 0 /
0.001 /
0 /
errorr
20 21 0 22 0 /
{mat} 1 6 0 1 /
0 0.0 /
0 33 1 /
{len(GROUPS)-1} /
{g} /
stop
"""
    open(os.path.join(d, "input"), "w").write(inp)
    with open(os.path.join(d, "input")) as fi, open(os.path.join(d, "njoy.log"), "w") as fo:
        r = subprocess.run(["njoy"], stdin=fi, stdout=fo, stderr=subprocess.STDOUT, cwd=d)
    ok = os.path.exists(os.path.join(d, "tape22")) and os.path.getsize(os.path.join(d, "tape22")) > 0
    print(nuc, "mat", mat, "rc", r.returncode, "ok" if ok else "FAILED", flush=True)
    for t in ("tape20",):
        os.remove(os.path.join(d, t))


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ENDF)):
        run(n)

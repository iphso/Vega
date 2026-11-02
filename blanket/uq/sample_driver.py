"""Random-sampling ("TMC-lite") driver: per sample, draw correlated multigroup factors
for the chosen nuclides, write perturbed HDF5 files + a cross_sections.xml, run the ARC
baseline, append the result to /runs/results/<set>.jsonl (resumable).

  blanket/uq/sample_driver.py --set li7 --nuclides Li7 --n 20
  blanket/uq/sample_driver.py --set noise --mode base --n 6          # unperturbed library, new seeds
  blanket/uq/sample_driver.py --set chk_unpert --mode unity --nuclides Li6 Li7 Be9 F19 --n 1
  blanket/uq/sample_driver.py --set chk_li6nt --mode li6nt10 --n 1
"""
import argparse, json, os, shutil, sys, time, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uqlib  # noqa: E402
from run_case import run_case  # noqa: E402

ROOT = uqlib.UQ_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--nuclides", nargs="*", default=[])
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--mode", default="sample", choices=["sample", "base", "unity", "li6nt10"])
ap.add_argument("--particles", type=int, default=300_000)
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--seed", type=int, default=None, help="transport seed of sample 0 (default: from set name)")
ap.add_argument("--factor-key", default=None, help="name whose factor draws to reuse (default: --set)")
ap.add_argument("--fixed-seed", action="store_true", help="same transport seed for every sample")
a = ap.parse_args()

threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
os.makedirs(os.path.join(ROOT, "results", a.set + "_factors"), exist_ok=True)
res_path = os.path.join(ROOT, "results", a.set + ".jsonl")
done = set()
if os.path.exists(res_path):
    done = {json.loads(l)["i"] for l in open(res_path) if l.strip()}
key = zlib.crc32((a.factor_key or a.set).encode()) % 100_000
seed0 = a.seed if a.seed is not None else 1000 + key * 100
nuclides = a.nuclides if a.mode != "li6nt10" else ["Li6"]
plans = {n: uqlib.load_plan(n) for n in nuclides}
samplers = {}
for n, (meta, bounds, C, _) in plans.items():
    samplers[n], neg = uqlib.lognormal_sampler(C)

for i in range(a.n):
    if i in done:
        continue
    rng = np.random.default_rng([key, i])
    libdir = os.path.join(ROOT, "libs", a.set, str(i))
    os.makedirs(libdir, exist_ok=True)
    pert, fsave, diag = {}, {}, {}
    t0 = time.time()
    for n in nuclides:
        meta, bounds, C, _ = plans[n]
        npar, ng = len(meta["params"]), len(bounds) - 1
        if a.mode == "sample":
            f = samplers[n](rng).reshape(npar, ng)
        else:
            f = np.ones((npar, ng))
            if a.mode == "li6nt10":
                f[[p for p, (m, _) in enumerate(meta["params"]) if m == 105][0]] = 1.10
        out = os.path.join(libdir, f"{n}.h5")
        diag[n] = uqlib.write_perturbed(n, f, out, plans[n])
        pert[n] = out; fsave[n] = f
    if a.mode == "base":
        xml = os.path.join(uqlib.BASE_XS, "cross_sections.xml")
    else:
        xml = os.path.join(libdir, "cross_sections.xml")
        uqlib.write_xs_xml(pert, xml)
    t_lib = time.time() - t0
    seed = seed0 if a.fixed_seed else seed0 + i
    r = run_case(xml, os.path.join(ROOT, "runs", a.set, str(i)), a.particles, a.batches, seed, threads)
    r.update(i=i, set=a.set, mode=a.mode, nuclides=nuclides, lib_s=t_lib, diag=diag)
    np.savez_compressed(os.path.join(ROOT, "results", a.set + "_factors", f"{i}.npz"), **fsave)
    with open(res_path, "a") as fo:
        fo.write(json.dumps(r) + "\n")
    shutil.rmtree(libdir, ignore_errors=True)
    print(f"[{a.set} {i}] TBR {r['TBR']:.5f} +- {r['TBR_std']:.5f}  heat {r['heat_MW']:.2f} +- {r['heat_MW_std']:.2f} MW  "
          f"(lib {t_lib:.0f}s, run {r['wall_s']:.0f}s)", flush=True)

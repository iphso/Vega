"""FNS benchmarks on the nuclear-data-perturbed `all` libraries of the ARC UQ campaign.

For each sample i the library is REBUILT from the saved factors
(~/runs/blanket/uq/results/all_A_factors/i.npz; identical factors for all_B) with
blanket/uq/uqlib.write_perturbed, into our own directory /fnsruns/libs/<i>/ (the campaign deleted its
libraries; ~/runs/blanket/uq is only read). Only the perturbed nuclides that occur in these models are
rebuilt: Li6, Li7 (Li2O), Be9 (Be), Ni58 (stainless drawer/cladding, Ni foils). F-19 and W are
perturbed in the ARC samples but are absent from both assemblies, so leaving them unperturbed is exact.
Each library is used for both benchmarks and both seeds (777 = A, 778 = B, as in the campaign),
then deleted. Sample -1 = nominal library. Resumable.

  docker compose run --rm -v ~/runs/blanket/bench/fns:/fnsruns -v ~/runs/blanket/uq:/runs:ro openmc \
      blanket/bench/fns/sample_runs.py --n 30
"""
import argparse, json, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "uq"))
import uqlib  # noqa: E402
import fnslib  # noqa: E402
from run_bench import run  # noqa: E402

NUCS = ["Li6", "Li7", "Be9", "Ni58"]
SEEDS = {"A": 777, "B": 778}

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--start", type=int, default=-1)
ap.add_argument("--cases", nargs="*", default=["li2o", "be"])
ap.add_argument("--particles", nargs=2, type=int, default=[20_000_000, 5_000_000], help="li2o be")
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--out", default="/fnsruns/samples")
a = ap.parse_args()
PART = dict(zip(["li2o", "be"], a.particles))
OUT = a.out
os.makedirs(OUT, exist_ok=True)
plans = {n: uqlib.load_plan(n) for n in NUCS}

for i in range(a.start, a.n):
    todo = [(c, s) for c in a.cases for s in SEEDS
            if not os.path.exists(os.path.join(OUT, f"{c}_{s}_{i}.json"))]
    if not todo:
        continue
    libdir = f"/fnsruns/libs/{i}"
    t0 = time.time()
    if i < 0:
        xml = os.path.join(fnslib.FNS_XS, "cross_sections.xml")
    else:
        os.makedirs(libdir, exist_ok=True)
        fa = np.load(f"/runs/results/all_A_factors/{i}.npz")
        fb = np.load(f"/runs/results/all_B_factors/{i}.npz")
        pert = {}
        for n in NUCS:
            assert np.array_equal(fa[n], fb[n])
            pert[n] = os.path.join(libdir, f"{n}.h5")
            uqlib.write_perturbed(n, fa[n], pert[n], plans[n])
        xml = os.path.join(libdir, "cross_sections.xml")
        fnslib.write_xs_xml(pert, xml)
    tlib = time.time() - t0
    for c, s in todo:
        r = run(c, room=False, particles=PART[c], batches=a.batches, seed=SEEDS[s], xs=xml,
                work=f"/fnsruns/work/s_{c}_{s}_{i}")
        r["info"].update(sample=i, set=f"all_{s}", lib_s=tlib)
        json.dump(r, open(os.path.join(OUT, f"{c}_{s}_{i}.json"), "w"))
        print(f"[{i} {c} {s}] wall {r['info']['wall_s']:.0f}s (lib {tlib:.0f}s)", flush=True)
    if i >= 0:
        shutil.rmtree(libdir, ignore_errors=True)

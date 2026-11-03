"""Run OKTAVIAN cases against one library and append raw results to /bench/results/<tag>.jsonl.

  run_bench.py --tag nom_A --seed 777 --particles 200000 [--lib base|<sample index>] [--cases ...]
               [--geom '{"li_rin": 11}']
Each line: {case, tag, seed, lib, particles, wall_s, geom, out:{...extract()...}}.
Resumable: (case, tag) pairs already present are skipped.
"""
import argparse, json, os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--particles", type=int, default=200_000)
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--lib", default="base")
ap.add_argument("--cases", nargs="*", default=okt.ALL_CASES)
ap.add_argument("--geom", default="{}")
ap.add_argument("--particles-scale", default="{}", help='per-case multiplier, e.g. {"tbr_li": 0.5}')
a = ap.parse_args()
threads = int(os.environ.get("OMP_NUM_THREADS", "12"))
geom = json.loads(a.geom); pscale = json.loads(a.particles_scale)
os.makedirs("/bench/results", exist_ok=True)
res = f"/bench/results/{a.tag}.jsonl"
done = set()
if os.path.exists(res):
    done = {json.loads(l)["case"] for l in open(res) if l.strip()}
if a.lib == "base":
    base_xml, siglib = okt.XS_BASE, None
else:
    d = f"/bench/libs/{a.lib}"
    base_xml = os.path.join(d, "cross_sections.xml")
    siglib = {"Li6": os.path.join(d, "Li6.h5"), "Li7": os.path.join(d, "Li7.h5")}
work = f"/bench/work/{a.tag}"
xs = okt.xs_xml(base_xml, work)
for case in a.cases:
    if case in done:
        continue
    n = int(a.particles * pscale.get(case, 1.0))
    model, meta = okt.build(case, xs, n, a.batches, a.seed, geom, siglib)
    wd = os.path.join(work, case)
    os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    sp_path = model.run(cwd=wd, threads=threads, output=False)
    wall = time.time() - t0
    import openmc
    with openmc.StatePoint(sp_path) as sp:
        out = okt.extract(sp, meta)
    rec = dict(case=case, tag=a.tag, seed=a.seed, lib=a.lib, particles=n, batches=a.batches, wall_s=wall,
               geom=geom, out=out)
    with open(res, "a") as f:
        f.write(json.dumps(rec) + "\n")
    shutil.rmtree(wd, ignore_errors=True)
    v = out.get("tbr")
    print(f"[{a.tag}] {case}: {wall:.0f}s" + (f" TBR {v[0]:.4f}+-{v[1]:.4f}" if v else ""), flush=True)
shutil.rmtree(work, ignore_errors=True)

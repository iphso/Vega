"""Run the ARC baseline (blanket/arc_model.py defaults) against a given cross_sections.xml.

Returns TBR and total nuclear heating (neutron KERMA + photon transport) per source
neutron with their statistical std. Used by sample_driver.py for every UQ sample.
"""
import json, os, sys, time, shutil
import numpy as np
import openmc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import arc_model  # noqa: E402


def run_case(xs_xml, workdir, particles=300_000, batches=20, seed=1, threads=8, photons=True):
    model, cells, p = arc_model.build(None, particles // batches, batches, photons)
    model.materials.cross_sections = xs_xml
    # single-bin totals so the statistical std of the summed score is exact
    flibe = [m for m in model.materials if m.name == "flibe"][0]
    t = openmc.Tally(name="TBR_tot"); t.filters = [openmc.MaterialFilter([flibe])]; t.scores = ["(n,Xt)"]
    h = openmc.Tally(name="heat_tot"); h.scores = ["heating" if photons else "heating-local"]
    model.tallies += [t, h]
    model.settings.seed = int(seed)
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    sp_path = model.run(cwd=workdir, threads=threads, output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        tt = sp.get_tally(name="TBR_tot"); ht = sp.get_tally(name="heat_tot")
        heat = sp.get_tally(name="heating")
        tbr_mean, tbr_std = float(tt.mean.ravel()[0]), float(tt.std_dev.ravel()[0])
        heat_mean, heat_std = float(ht.mean.ravel()[0]), float(ht.std_dev.ravel()[0])
        tbr_cells = float(sp.get_tally(name="TBR").mean.sum())
        heat_cells = dict(zip(heat.filters[0].bins, heat.mean.ravel()))
    id2name = {c.id: n for n, c in cells.items()}
    eV_to_MW = arc_model.NEUTRONS_PER_S * 1.602176634e-19 / 1e6
    out = dict(TBR=tbr_mean, TBR_cellsum=tbr_cells, TBR_std=tbr_std, heat_eV=heat_mean, heat_eV_std=heat_std,
               heat_MW=heat_mean * eV_to_MW, heat_MW_std=heat_std * eV_to_MW,
               heat_cells_MW={id2name.get(int(k), str(k)): float(v) * eV_to_MW for k, v in heat_cells.items()},
               particles=particles, batches=batches, seed=int(seed), wall_s=wall, xs=xs_xml)
    for f in os.listdir(workdir):
        if f.endswith(".h5") or f.endswith(".xml"):
            os.remove(os.path.join(workdir, f))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--xs", default="/work/blanket/xs/cross_sections.xml")
    ap.add_argument("--particles", type=int, default=300_000)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default="/runs/runs/test")
    a = ap.parse_args()
    print(json.dumps(run_case(a.xs, a.work, a.particles, a.batches, a.seed, int(os.environ.get("OMP_NUM_THREADS", "8"))), indent=1))

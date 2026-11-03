"""Rebuild the 30 joint ("all") nuclear-data-perturbed libraries of the ARC UQ campaign.

The campaign (blanket/uq/sample_driver.py) deleted each perturbed library after its run but
saved the sampled group factors per sample (results/all_A_factors/<i>.npz; the B set holds
identical factors, checked below). Feeding the same factors through uqlib.write_perturbed
reproduces the same HDF5 files, so sample i here is the same perturbation as ARC sample i.

Run (from ~/Projects/Vega):
  docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e UQ_ROOT=/uq \
     -v ~/runs/blanket/uq:/uq:ro -v ~/runs/blanket/bench/oktavian:/bench \
     openmc blanket/bench/oktavian/gen_libs.py [i0 i1 ...]
Writes /bench/libs/<i>/{<nuc>.h5, cross_sections.xml}.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/work/blanket/uq")
import uqlib  # noqa: E402

NUCS = ["Li6", "Li7", "Be9", "F19", "Ni58", "W182", "W183", "W184", "W186"]
idx = [int(x) for x in sys.argv[1:]] or list(range(30))
plans = {n: uqlib.load_plan(n) for n in NUCS}
for i in idx:
    fa = np.load(f"/uq/results/all_A_factors/{i}.npz")
    fb = np.load(f"/uq/results/all_B_factors/{i}.npz")
    assert all(np.array_equal(fa[k], fb[k]) for k in NUCS), f"A/B factors differ for sample {i}"
    d = f"/bench/libs/{i}"
    xml = os.path.join(d, "cross_sections.xml")
    if os.path.exists(xml):
        print("skip", i, flush=True); continue
    os.makedirs(d, exist_ok=True)
    pert, diag = {}, {}
    for n in NUCS:
        out = os.path.join(d, f"{n}.h5")
        diag[n] = uqlib.write_perturbed(n, fa[n], out, plans[n])
        pert[n] = out
    uqlib.write_xs_xml(pert, xml + ".tmp")
    os.replace(xml + ".tmp", xml)
    # the campaign stored the same diagnostic per sample: compare to prove the rebuild is identical
    ref = [json.loads(l) for l in open("/work/blanket/uq/raw/results/all_A.jsonl") if json.loads(l)["i"] == i][0]["diag"]
    dev = max(abs((diag[n].get("kerma_rel_change_14MeV") or 0) - (ref[n].get("kerma_rel_change_14MeV") or 0)) for n in NUCS)
    print(f"sample {i}: written, max |kerma diag - campaign| = {dev:.2e}", flush=True)

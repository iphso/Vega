"""Reproduce Sorbom 2015 Fig. 30: TBR vs first-wall material and thickness.
Neutron-only (TBR needs no photon transport).

Run: docker compose run --rm openmc blanket/fig30_sweep.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_model import run, HERE

out_path = os.path.join(HERE, "runs", "fig30_sweep.json")
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for mat in ["W", "Be", "CFC", "Inconel"]:
    for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        key = f"{mat}_{t}"
        if key in res:
            continue
        r = run({"fw_material": mat, "fw_cm": t}, particles=10000, batches=30, photons=False,
                tag=f"fig30/{key}")
        res[key] = r["TBR"]
        json.dump(res, open(out_path, "w"), indent=2)
        print(key, r["TBR"]["mean"], flush=True)

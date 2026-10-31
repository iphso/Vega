"""coeffs_damage.npz -> damage_sidecar.json for the browser (StoneLab src/neutron/damage.ts).

Run (host python + numpy): python3 blanket/damage/sidecar_json.py [RUN_DIR ...]
  default: every blanket/damage/runs/VITAMIN-J-175*/
"""
import glob, json, os, sys
import numpy as np

DMG = os.path.dirname(os.path.abspath(__file__))
dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(DMG, "runs", "VITAMIN-J-175*")))
for d in dirs:
    npz = os.path.join(d, "coeffs_damage.npz")
    if not os.path.exists(npz):
        continue
    c = np.load(npz)
    man = json.load(open(os.path.join(d, "coeffs_damage.json")))
    res = json.load(open(os.path.join(d, "result.json")))
    cells = [str(x) for x in c["cells"]]
    out = {
        "format_version": 1, "groups": man["groups"], "G": int(c["sig_damage"].shape[1]),
        "group_order": man["group_order"], "design": man["design"], "histories": man["histories"],
        "nrt_k": 0.8, "fpy_s": man["convert"]["FPY_s"],
        "vv_in_peak_over_avg": man["vv_in_peak_over_avg"], "vv_in_peak_over_avg_He": man["vv_in_peak_over_avg_He"],
        "vv_in_peak_sector_deg": man["vv_in_peak_sector_deg"],
        "vv_in_plasma_side_over_avg": man["vv_in_plasma_side_over_avg"],
        "cells": {n: {"sig_damage": [float(v) for v in c["sig_damage"][i]],
                      "sig_he": [float(v) for v in c["sig_he"][i]],
                      "atom_density": float(c["atom_density"][i]), "Ed_eV": float(c["Ed_eV"][i])}
                  for i, n in enumerate(cells)},
        "ce": {n: {k: v for k, v in res["layers"][n].items()} for n in res["layers"]},
    }
    json.dump(out, open(os.path.join(d, "damage_sidecar.json"), "w"))
    print("wrote", os.path.join(d, "damage_sidecar.json"))

"""Combine independent coil_ref.py runs (different seeds, same model and windows) into one
result dir that compare.py reads: history-weighted means, s.d. combined as independent.

  python3 combine.py <out dir> <run dir> <run dir> ...      (host python, numpy only)
"""
import json, os, sys
import numpy as np

out, runs = sys.argv[1], sys.argv[2:]
Z = [np.load(f"{r}/result.npz") for r in runs]
M = [json.load(open(f"{r}/result.json")) for r in runs]
w = np.array([m["particles"] * m["batches"] for m in M], float)
w /= w.sum()
arrs = {}
for k in Z[0].files:
    if k.endswith("_sd"):
        arrs[k] = np.sqrt(sum((wi * z[k]) ** 2 for wi, z in zip(w, Z)))
    elif Z[0][k].dtype.kind == "f" and k not in ("prof_r", "ww_r", "ww_z", "ww_e", "group_edges"):
        arrs[k] = sum(wi * z[k] for wi, z in zip(w, Z))
    else:
        arrs[k] = Z[0][k]
os.makedirs(out, exist_ok=True)
np.savez_compressed(f"{out}/result.npz", **arrs)
meta = dict(M[0])
meta["combined_from"] = runs
meta["particles"] = int(sum(m["particles"] * m["batches"] for m in M)); meta["batches"] = 1
for k in ("face_fast_track_1bin", "face_fast_track_mesh"):
    if all(k in m for m in M):
        meta[k] = [float(sum(wi * m[k][0] for wi, m in zip(w, M))), float(np.sqrt(sum((wi * m[k][1]) ** 2 for wi, m in zip(w, M))))]
for k in ("face_fast_fluence_per_fpy",):
    meta[k] = float(sum(wi * m[k] for wi, m in zip(w, M)))
meta["face_fast_rel_sd"] = meta["face_fast_track_1bin"][1] / meta["face_fast_track_1bin"][0]
meta["seeds_face_fast_fluence"] = [m["face_fast_fluence_per_fpy"] for m in M]
# run-to-run scatter: with a residual heavy tail the batch s.d. understates the error, so for
# >= 3 runs also report the s.d. of the mean from the spread of the per-run values
v = np.array(meta["seeds_face_fast_fluence"])
if len(v) >= 3:
    emp = float(v.std(ddof=1) / np.sqrt(len(v)) / v.mean())
    meta["face_fast_rel_sd_batch"] = meta["face_fast_track_1bin"][1] / meta["face_fast_track_1bin"][0]
    meta["face_fast_rel_sd_runs"] = emp
    meta["face_fast_rel_sd"] = max(emp, meta["face_fast_rel_sd_batch"])
meta["magnet_life_fpy"] = 3e18 / meta["face_fast_fluence_per_fpy"]
meta["magnet_life_fpy_sd"] = meta["magnet_life_fpy"] * meta["face_fast_rel_sd"]
meta["TBR"] = [float(sum(wi * m["TBR"][0] for wi, m in zip(w, M))), float(np.sqrt(sum((wi * m["TBR"][1]) ** 2 for wi, m in zip(w, M))))]
meta["seconds"] = float(sum(m["seconds"] for m in M))
json.dump(meta, open(f"{out}/result.json", "w"), indent=1)
print(f"{out}: face fast fluence {meta['face_fast_fluence_per_fpy']:.4e} ± {100 * meta['face_fast_rel_sd']:.2f}% -> magnet life "
      f"{meta['magnet_life_fpy']:.3f} ± {meta['magnet_life_fpy_sd']:.3f} FPY; per seed {[f'{x:.4e}' for x in meta['seeds_face_fast_fluence']]}")

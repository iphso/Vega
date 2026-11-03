"""Append the photon extension (photon_gen.py) to a neutpack: coupled n-gamma heating in the
browser transport (StoneLab packages/physics/src/neutron, neutpack.ts PhotonExt).

The neutron pack's manifest + data.bin are copied unchanged into <pack><SUFFIX>; the photon
arrays are appended to data.bin and described under manifest["photon"] (version 1). A reader
that does not know the key loads the pack exactly as before (every neutron array keeps its
offset); a pack without the key runs neutron-only with its photon_factor.

manifest["photon"] = {version, groups, P, nmu, group_edges_eV (ascending), source, ce, cells}
  cells[name] (same names as the neutron cells; group 0 = highest energy throughout):
    neutron side, G = the neutron pack's groups
      kerma_n[G]          neutron KERMA without the photon energy (MT301 'heating'), eV/cm:
                          replaces kerma x photon_factor when photons are transported
      prod_yield[G]       photons born per unit neutron flux, 1/cm (sum over photon groups)
      prod_row_ptr[G+1]   sparse rows: photon groups prod_h[k] with cumulative probability
      prod_h, prod_cdf    prod_cdf[k] along each neutron group's row (emitted isotropically)
    photon side, P photon groups -- the same layout as a neutron cell
      sigma_t[P]          total, 1/cm
      nu_out[P]           photons out of a collision x sigma_t (scatter + secondaries), 1/cm
      row_ptr, tr_g, tr_cdf, mu_cdf
                          sparse transfers and nmu-bin cosine CDF per transfer (secondary
                          photons isotropic, mixed into the scatter histogram by rate)
      kerma[P]            photon heating, eV/cm

Run: docker compose run --rm -v ~/runs/blanket/photon:/runs openmc blanket/photon/export_photon.py \
         <neutron pack> [photon run dir name] [--suffix _ng]
     e.g. VITAMIN-J-175_hist64 VITAMIN-J-175_VITAMIN-J-42
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NP = os.path.join(HERE, "..", "neutpack")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SUFFIX = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else "_ng"
if "--suffix" in sys.argv:
    args.remove(SUFFIX)
PACK = args[0]
RUN = args[1] if len(args) > 1 else "VITAMIN-J-175_VITAMIN-J-42"
src_dir = os.path.join(NP, PACK)
run_dir = os.path.join(os.environ.get("PHOTON_RUNS", "/runs"), RUN)
out = os.path.join(NP, PACK + SUFFIX)
os.makedirs(out, exist_ok=True)

man = json.load(open(os.path.join(src_dir, "manifest.json")))
data = open(os.path.join(src_dir, "data.bin"), "rb").read()
man.pop("photon", None)
offset = len(data) // 4
blobs = [data]
ph = np.load(os.path.join(run_dir, "photon.npz"))
res = json.load(open(os.path.join(run_dir, "result.json")))
G, NMU = man["G"], int(ph["nmu"])
assert ph["prod"].shape[1] == G, f"photon run has {ph['prod'].shape[1]} neutron groups, pack {G}"
assert np.allclose(ph["neutron_edges"], man["group_edges_eV"]), "neutron group edges differ"
P = ph["p_sigma_t"].shape[1]


def put(arr):
    global offset
    a = np.ascontiguousarray(arr, dtype="<f4").ravel()
    blobs.append(a.tobytes())
    o = offset
    offset += a.size
    return {"offset": o, "length": int(a.size)}


def sparse_rows(mat, wmu=None):
    """Rows of a nonnegative [n_in, n_out] matrix as (row_ptr, out index, cumulative prob[, mu cdf])."""
    tot = mat.sum(axis=1)
    rp, idx, cdf, mus = [0], [], [], []
    for g in range(mat.shape[0]):
        nz = np.nonzero(mat[g] > 0)[0]
        if tot[g] > 0 and len(nz):
            c = np.cumsum(mat[g, nz]) / tot[g]
            c[-1] = 1.0
            idx += nz.tolist(); cdf += c.tolist()
            if wmu is not None:
                mus.append(wmu[g, nz])
        rp.append(len(idx))
    return np.array(rp), np.array(idx), np.array(cdf), (np.concatenate(mus) if mus else np.zeros((0, NMU + 1)))


cells = {}
names = [str(c) for c in ph["cells"]]
# every photon group that can be populated needs a cross section in every cell, or photons in it
# stream through unattenuated
pop = (ph["prod"].sum(axis=(0, 1)) > 0) | (ph["p_sec"].sum(axis=(0, 1)) > 0) | (ph["p_scat"].sum(axis=(0, 1, 3)) > 0)
for n in man["cells"]:
    hole = np.nonzero(pop & (ph["p_sigma_t"][names.index(n)] <= 0))[0]
    if len(hole):
        raise SystemExit(f"cell {n}: photon groups {hole.tolist()} are populated but have sigma_t = 0 (rerun photon_gen's fill)")
for name in man["cells"]:
    if name not in names:
        raise SystemExit(f"photon run has no cell {name}")
    i = names.index(name)
    prod = np.clip(ph["prod"][i], 0, None)
    prp, ph_h, ph_cdf, _ = sparse_rows(prod)
    st = ph["p_sigma_t"][i]
    scat = np.clip(ph["p_scat"][i], 0, None)              # [P, P, NMU]
    sec = np.clip(ph["p_sec"][i], 0, None)                # [P, P]
    w = scat + sec[:, :, None] / NMU                      # secondaries isotropic
    T = w.sum(axis=2)
    tw = np.where(T[:, :, None] > 0, w / np.where(T[:, :, None] > 0, T[:, :, None], 1), 1.0 / NMU)
    mcdf = np.concatenate([np.zeros((P, P, 1)), np.cumsum(tw, axis=2)], axis=2)
    mcdf[:, :, -1] = 1.0
    rp, tg, tc, mu = sparse_rows(T, mcdf)
    cells[name] = {
        "kerma_n": put(ph["kerma_n"][i]), "prod_yield": put(prod.sum(axis=1)),
        "prod_row_ptr": put(prp), "prod_h": put(ph_h), "prod_cdf": put(ph_cdf),
        "sigma_t": put(st), "nu_out": put(T.sum(axis=1)), "row_ptr": put(rp), "tr_g": put(tg), "tr_cdf": put(tc),
        "mu_cdf": put(mu), "kerma": put(ph["p_kerma"][i]),
    }
man["photon"] = {
    "version": 1, "groups": "VITAMIN-J-42", "P": int(P), "nmu": NMU,
    "group_edges_eV": [float(e) for e in ph["photon_edges"]],
    "source": RUN, "ce": {"heating_MW": res["heating_MW"], "heating_split_MW": res["heating_split_MW"], "TBR": res["TBR"]},
    "cells": cells,
}
with open(os.path.join(out, "data.bin"), "wb") as f:
    for b in blobs:
        f.write(b)
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
added = offset * 4 - len(data)
print(f"wrote {out}: +{added / 1e6:.2f} MB photon data ({P} photon groups) on {len(data) / 1e6:.2f} MB")

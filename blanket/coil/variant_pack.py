"""Experiment packs for the TF-coil investigation: copy a neutpack and change chosen cells'
constants, writing <neutpack>/<src>_<suffix> (never in place).

  --wout absorption   weight out of a collision from the track-length absorption,
                      w = (1 - sigma_a / sigma_t) * nu_bar, nu_bar = sum nu-scatter / sum scatter,
                      instead of the analog nu-scatter sum / sigma_t (export_neutpack.py), in groups
                      with flux in the design run; the fill's for groups the export filled
  --from <mgxs run dir> --cells a,b   replace these cells' constants wholesale (sigma_t, transfer
                      rows, cosines, w) with that run's (e.g. a weight-window run's in-situ ones)

Run: docker compose run --rm --user $(id -u):$(id -g) -e HOME=/tmp openmc blanket/coil/variant_pack.py \
        --pack VITAMIN-J-175_hist64 --mgxs runs/mgxs/VITAMIN-J-175_hist64 --wout absorption --cells tank,shield,coil --suffix wa
"""
import argparse, json, os, shutil
import numpy as np
import openmc

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..")


def lib_arrays(lib, name, G):
    x = lib.get_by_name(f"cell_{name}")
    st = np.asarray(x.total[0]).reshape(G)
    sa = np.asarray(x.absorption[0]).reshape(G)
    ns = np.asarray(x.scatter_matrix[0]).reshape(G, G, -1)
    mm = np.asarray(x.multiplicity_matrix[0]).reshape(G, G) if x.multiplicity_matrix is not None else np.ones((G, G))
    return st, sa, ns, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--mgxs", required=True, help="the mgxs run the pack was exported from (relative to blanket/)")
    ap.add_argument("--fill", default="runs/mgxs/fill_VITAMIN-J-175_hist64")
    ap.add_argument("--wout", default=None)
    ap.add_argument("--cells", default="tank,shield,coil")
    ap.add_argument("--suffix", required=True)
    a = ap.parse_args()
    src = os.path.join(B, "neutpack", a.pack)
    dst = os.path.join(B, "neutpack", f"{a.pack}_{a.suffix}")
    os.makedirs(dst, exist_ok=True)
    m = json.load(open(os.path.join(src, "manifest.json")))
    data = np.fromfile(os.path.join(src, "data.bin"), "<f4").copy()
    G = m["G"]
    lib = openmc.MGXSLibrary.from_hdf5(os.path.join(B, a.mgxs, "mgxs.h5"))
    co = np.load(os.path.join(B, a.mgxs, "coeffs.npz"))
    cnames = [str(c) for c in co["cells"]]
    fl = openmc.MGXSLibrary.from_hdf5(os.path.join(B, a.fill, "mgxs.h5"))
    log = {}
    for name in a.cells.split(","):
        cm = m["cells"][name]
        sl = lambda k: slice(cm[k]["offset"], cm[k]["offset"] + cm[k]["length"])
        st, sa, ns, mm = lib_arrays(lib, name, G)
        empty = co["ce_flux"][cnames.index(name)] <= 0
        fst, fsa, fns, fmm = lib_arrays(fl, name, G)
        st = np.where(empty, fst, st); sa = np.where(empty, fsa, sa)
        ns = np.where(empty[:, None, None], fns, ns); mm = np.where(empty[:, None], fmm, mm)
        if a.wout == "absorption":
            p0 = np.clip(ns.sum(axis=2), 0, None)
            s_plain = np.where(mm > 0, p0 / np.where(mm > 0, mm, 1), 0).sum(axis=1)
            nubar = np.where(s_plain > 0, p0.sum(axis=1) / np.where(s_plain > 0, s_plain, 1), 1.0)
            nu_new = np.clip(st - sa, 0, None) * nubar
            old = data[sl("nu_out")].copy()
            # groups without a single sampled scatter keep their (zero) row: nothing to scale
            nu_new = np.where(old > 0, nu_new, old)
            data[sl("nu_out")] = nu_new.astype("<f4")
            w_old = np.where(st > 0, old / st, 0); w_new = np.where(st > 0, nu_new / st, 0)
            log[name] = {"max_abs_dw": float(np.abs(w_new - w_old).max()), "groups_w_gt_1_old": int((w_old[20:] > 1).sum()),
                         "groups_w_gt_1_new": int((w_new[20:] > 1).sum())}
    data.tofile(os.path.join(dst, "data.bin"))
    m["variant"] = {"from": a.pack, "wout": a.wout, "cells": a.cells, "log": log}
    json.dump(m, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    print(json.dumps(m["variant"], indent=1))


if __name__ == "__main__":
    main()

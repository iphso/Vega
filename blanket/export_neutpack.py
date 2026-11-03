"""Export a gate-1 multigroup library to a .neutpack for the browser transport
(StoneLab packages/physics/src/neutron).

Layout: <out>/manifest.json + <out>/data.bin (little-endian float32; integer
fields stored as exact floats, all < 2^24). Per cell ("material role"), sparse in
the outgoing group (only ~17% of (g, g') pairs ever transfer):
  sigma_t[G]                 total, 1/cm
  nu_out[G]                  sum over g' of the P0 nu-scatter (weight out x sigma_t)
  row_ptr[G+1]               transfers of incoming group g are entries row_ptr[g]..row_ptr[g+1]
  tr_g[nnz], tr_cdf[nnz]     outgoing group and cumulative probability along the row
  mu_cdf[nnz*(NMU+1)]        scattering-cosine histogram CDF per transfer, NMU uniform
                             bins on [-1, 1]; sample a bin, then uniform inside it.
                             From a histogram library directly; from Legendre moments
                             via the pdf at bin midpoints, negatives clipped
  sigma_tritium[G], kerma[G] tritium production (1/cm) and local heating (eV/cm)
                             from CE reaction-rate / flux ratios
  photon_factor (manifest)   coupled n-gamma heating / local (kerma) heating per cell,
                             from an OpenMC photon-transport run of the same design
                             (runs/arc_photon<TAG>/result.json, else the baseline
                             runs/arc_baseline): gammas carry ~30% of the heat out of
                             the inner VV into the tile, multiplier and bulk
Group 0 is the highest energy.

Run: docker compose run --rm openmc blanket/export_neutpack.py [GROUPS] [FORMAT] [TAG] [--wout=absorption] [--resolve=R] [--suffix=S]
     --wout=absorption  weight out of a collision w = (1 - sigma_a / sigma_t) x nu_bar from the
                        track-length absorption (nu_bar = sum nu-scatter / sum scatter), instead of
                        the analog nu-scatter sum / sigma_t (default "scatter", as before). The
                        analog sum is noise where the design run saw few collisions: the ARC
                        coil had w > 1 in 50 groups below the (n,2n) thresholds, w = 2 in one, and
                        the thermal absorption implied by w is a small difference of two noisy numbers
     --resolve=R        deep cells (those fill_mgxs.py covers) also take the fill's constants in
                        groups whose design-run flux has rel. s.d. > R (coeffs.npz ce_flux_sd,
                        mgxs_gen.py writes it), not only in empty ones
     --suffix=S         appended to the output pack directory name
     docker compose run --rm openmc blanket/export_neutpack.py --libra <repo>
       (runs/libra/<repo> from libra_simple.py -> neutpack/libra_<repo>, plus its raster
        and source in libra.json; no photon factors: TBR only)
"""
import json, os, sys
import numpy as np
import openmc

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = dict((a[2:].split("=") + [""])[:2] for a in sys.argv[1:] if a.startswith("--") and "=" in a)
sys.argv = [a for a in sys.argv if not (a.startswith("--") and "=" in a)]
WOUT = FLAGS.get("wout", "scatter")
RESOLVE = float(FLAGS["resolve"]) if "resolve" in FLAGS else None
SUFFIX = FLAGS.get("suffix", "")
LIBRA = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--libra" else None
if LIBRA:
    GS, FMT, TAG = "VITAMIN-J-175", "hist64", ""
    src = os.path.join(HERE, "runs", "libra", LIBRA)
    out = os.path.join(HERE, "neutpack", f"libra_{LIBRA}")  # LIBRA may carry a study suffix, e.g. BABY-1L-run-4_h025
else:
    GS = sys.argv[1] if len(sys.argv) > 1 else "VITAMIN-J-175"
    FMT = sys.argv[2] if len(sys.argv) > 2 else "hist64"
    TAG = sys.argv[3] if len(sys.argv) > 3 else ""   # e.g. "_fw_material-Be", as mgxs_gen names its run dir
    src = os.path.join(HERE, "runs", "mgxs", f"{GS}_{FMT}{TAG}")
    out = os.path.join(HERE, "neutpack", f"{GS}_{FMT}{TAG}{SUFFIX}")
os.makedirs(out, exist_ok=True)

lib = openmc.MGXSLibrary.from_hdf5(os.path.join(src, "mgxs.h5"))
co = np.load(os.path.join(src, "coeffs.npz"))
cells = [str(c) for c in co["cells"]]
G = lib.energy_groups.num_groups
NMU = int(FMT[4:]) if FMT.startswith("hist") else 64
mid = -1 + (np.arange(NMU) + 0.5) * 2 / NMU

blobs, manifest = [], {"format_version": 2, "groups": GS, "format": FMT, "G": G, "nmu": NMU,
                       "group_edges_eV": [float(e) for e in co["group_edges"]],
                       "cells": {}}
offset = 0


def put(arr):
    global offset
    a = np.ascontiguousarray(arr, dtype="<f4").ravel()
    blobs.append(a.tobytes())
    o = offset
    offset += a.size
    return {"offset": o, "length": int(a.size)}


# groups a design run left empty in the deep cells (tank wall, shield, coil) take the
# constants of the same material in every group (fill_mgxs.py); without them a neutron in
# such a group would cross the cell unattenuated
FILL = os.path.join(HERE, "runs", "mgxs", f"fill_{GS}_{FMT}")
fill = None
if not LIBRA and os.path.exists(os.path.join(FILL, "mgxs.h5")):
    fill = (openmc.MGXSLibrary.from_hdf5(os.path.join(FILL, "mgxs.h5")), np.load(os.path.join(FILL, "coeffs.npz")))
    manifest["filled"] = {}

for i, name in enumerate(cells):
    xs = lib.get_by_name(f"cell_{name}")
    st = np.asarray(xs.total[0]).reshape(G).copy()
    # XSdata.scatter_matrix holds the NU-scatter matrix already (create_mg_library
    # writes "nu-scatter matrix" there; OpenMC divides by multiplicity_matrix to get
    # the plain scatter). [G_in, G_out, angular]
    ns = np.asarray(xs.scatter_matrix[0]).reshape(G, G, -1).copy()
    sig_trit, kerm = co["sig_tritium"][i].copy(), co["kerma"][i].copy()
    fnames = [str(c) for c in fill[1]["cells"]] if fill else []
    sa = np.asarray(xs.absorption[0]).reshape(G).copy()
    mm = np.asarray(xs.multiplicity_matrix[0]).reshape(G, G).copy() if xs.multiplicity_matrix is not None else np.ones((G, G))
    if name in fnames:
        bad = co["ce_flux"][i] <= 0
        if RESOLVE is not None and "ce_flux_sd" in co.files:
            f, fs = co["ce_flux"][i], co["ce_flux_sd"][i]
            bad |= fs > RESOLVE * np.where(f > 0, f, 0)
        empty = np.nonzero(bad)[0]
        if len(empty):
            fx, fj = fill[0].get_by_name(f"cell_{name}"), fnames.index(name)
            fst = np.asarray(fx.total[0]).reshape(G)
            fns = np.asarray(fx.scatter_matrix[0]).reshape(G, G, -1)
            st[empty] = fst[empty]; ns[empty] = fns[empty]
            sa[empty] = np.asarray(fx.absorption[0]).reshape(G)[empty]
            if fx.multiplicity_matrix is not None:
                mm[empty] = np.asarray(fx.multiplicity_matrix[0]).reshape(G, G)[empty]
            sig_trit[empty] = fill[1]["sig_tritium"][fj][empty]; kerm[empty] = fill[1]["kerma"][fj][empty]
            manifest["filled"][name] = len(empty)
    if FMT.startswith("hist"):
        p0 = ns.sum(axis=2)
        w = np.clip(ns, 0.0, None)
    else:
        p0 = ns[:, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            mom = np.where(p0[:, :, None] > 0, ns / p0[:, :, None], 0.0)
        L = mom.shape[2]
        Pm = np.array([np.polynomial.legendre.Legendre.basis(l)(mid) for l in range(L)])
        w = np.clip(np.einsum("abl,l,lm->abm", mom, (2 * np.arange(L) + 1) / 2, Pm), 0.0, None)
    tot = w.sum(axis=2, keepdims=True)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1), 1.0 / NMU)   # no angular data -> isotropic
    cdf = np.concatenate([np.zeros((G, G, 1)), np.cumsum(w, axis=2)], axis=2)
    cdf[:, :, -1] = 1.0
    p0 = np.clip(p0, 0.0, None)
    nu_out = p0.sum(axis=1)
    if WOUT == "absorption":
        # scale each row to (sigma_t - sigma_a) x nu_bar; rows with no sampled transfer stay empty
        plain = np.where(mm > 0, p0 / np.where(mm > 0, mm, 1.0), 0.0).sum(axis=1)
        nubar = np.where(plain > 0, nu_out / np.where(plain > 0, plain, 1.0), 1.0)
        target = np.clip(st - sa, 0.0, None) * nubar
        scale = np.where(nu_out > 0, target / np.where(nu_out > 0, nu_out, 1.0), 0.0)
        p0 = p0 * scale[:, None]
        nu_out = p0.sum(axis=1)
    row_ptr, tr_g, tr_cdf, mu_rows = [0], [], [], []
    for g in range(G):
        nzg = np.nonzero(p0[g] > 0)[0]
        if nu_out[g] > 0:
            c = np.cumsum(p0[g, nzg]) / nu_out[g]
            c[-1] = 1.0
            tr_g += nzg.tolist(); tr_cdf += c.tolist(); mu_rows.append(cdf[g, nzg])
        row_ptr.append(len(tr_g))
    mu_sparse = np.concatenate(mu_rows) if mu_rows else np.zeros((0, NMU + 1))
    manifest["cells"][name] = {
        "sigma_t": put(st), "nu_out": put(nu_out), "row_ptr": put(np.array(row_ptr)),
        "tr_g": put(np.array(tr_g)), "tr_cdf": put(np.array(tr_cdf)), "mu_cdf": put(mu_sparse),
        "sigma_tritium": put(sig_trit), "kerma": put(kerm),
    }

if LIBRA:
    import shutil
    meta = json.load(open(os.path.join(src, "libra.json")))
    shutil.copy(os.path.join(src, "raster.u8"), os.path.join(out, "raster.u8"))
    manifest["libra"] = meta
    with open(os.path.join(out, "data.bin"), "wb") as f:
        for b in blobs:
            f.write(b)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"wrote {out}: {G} groups, {len(cells)} cells, {offset * 4 / 1e6:.2f} MB + raster")
    sys.exit(0)

# photon redistribution factors
gate = json.load(open(os.path.join(src, "gate1.json")))
ph_path = os.path.join(HERE, "runs", f"arc_photon{TAG[:-3] if TAG.endswith('_ww') else TAG}", "result.json")
if not os.path.exists(ph_path):
    ph_path = os.path.join(HERE, "runs", "arc_baseline", "result.json")
ph = json.load(open(ph_path))["heating_MW"]
manifest["photon_source"] = os.path.relpath(ph_path, HERE)
manifest["export_flags"] = {"wout": WOUT, "resolve": RESOLVE}
for name in cells:
    local = gate["ce"]["heating_MW"].get(name, 0.0)
    coupled = ph.get(name, {}).get("mean", 0.0)
    # below ~0.1 MW (coil, ~1e-3 MW) both tallies are noise: leave local deposition
    manifest["cells"][name]["photon_factor"] = coupled / local if local > 0.1 and coupled > 0 else 1.0

with open(os.path.join(out, "data.bin"), "wb") as f:
    for b in blobs:
        f.write(b)
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(f"wrote {out}: {G} groups, {len(cells)} cells, {offset * 4 / 1e6:.2f} MB")

"""Build the per-nuclide sampling plan (see uqlib docstring): params, joint covariance,
HEATR partial KERMA coefficients, gas-production multiplicities.

Run in the openmc container after njoy_cov.py:  blanket/uq/prep.py [nuclide ...]
"""
import json, os, subprocess, sys, shutil
import numpy as np
from scipy.optimize import nnls
import openmc.data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covlib import read_errorr, lumped_map  # noqa: E402
from njoy_cov import ENDF, mat_of  # noqa: E402
import uqlib  # noqa: E402

ROOT = uqlib.UQ_ROOT


def heatr_partials(nuc, mts):
    """NJOY RECONR -> HEATR (local=0) partial kermas mt+300; returns {mt: (E, k_r)} in eV."""
    import endf
    src = os.path.join(ROOT, "endf", ENDF[nuc]); mat = mat_of(src)
    d = os.path.join(ROOT, "cov", nuc, "heatr"); os.makedirs(d, exist_ok=True)
    shutil.copy(src, os.path.join(d, "tape20"))
    out = {}
    # mt+300 must fit ENDF's 3-digit MT: level-resolved 600-849 use their sum's partial
    hmap = {m: (103 + (m - 600) // 50 if 600 <= m < 850 else m) for m in mts}
    req = sorted(set(hmap.values()) | {3})
    chunks = [req[i:i + 6] for i in range(0, len(req), 6)]   # NJOY2016 heatr: <=6 partials per pass
    inp = f"reconr\n20 21 /\n'{nuc}' /\n{mat} 0 /\n0.001 /\n0 /\n"
    for ci, ch in enumerate(chunks):
        inp += f"heatr\n20 21 {30 + ci} 0 /\n{mat} {len(ch)} 0 0 0 0 /\n{' '.join(str(m + 300) for m in ch)} /\n"
    inp += "stop\n"
    open(os.path.join(d, "input"), "w").write(inp)
    with open(os.path.join(d, "input")) as fi, open(os.path.join(d, "njoy.log"), "w") as fo:
        subprocess.run(["njoy"], stdin=fi, stdout=fo, stderr=subprocess.STDOUT, cwd=d, check=True)
    for ci, ch in enumerate(chunks):
        m = endf.Material(os.path.join(d, f"tape{30 + ci}"))
        sec = m.section_data
        for mt in ch:
            if (3, mt + 300) not in sec or (3, mt) not in sec:
                print(f"  {nuc}: no partial kerma for MT{mt}", flush=True)
                continue
            K = sec[3, mt + 300]["sigma"]; s = sec[3, mt]["sigma"]
            E = np.asarray(K.x, float)
            sv = s(E)
            k = np.where(sv > 0, np.asarray(K.y, float) / np.where(sv > 0, sv, 1), 0.0)
            out[mt] = (E, k)
        if ci == len(chunks) - 1 or ci == 0:
            tot = sec[3, 301]["sigma"]
            out["_total"] = (np.asarray(tot.x, float), np.asarray(tot.y, float))
        os.remove(os.path.join(d, f"tape{30 + ci}"))
    res = {"_total": out.pop("_total")}
    for m in mts:
        h = hmap[m]
        if h in out:
            res[m] = out[h]
        elif 3 in out:
            print(f"  {nuc}: MT{m}: no partial kerma (MT{h + 300}); using nonelastic k = K303/sigma3", flush=True)
            res[m] = out[3]
    for t in ("tape20", "tape21"):
        if os.path.exists(os.path.join(d, t)):
            os.remove(os.path.join(d, t))
    return res


def gas_fit(data, T="294K"):
    E = data.energy[T]
    nonred = [mt for mt, rx in data.reactions.items() if not rx.redundant and T in rx.xs]
    cols = {}
    for mt in nonred:
        tab = data.reactions[mt].xs[T]; j = np.searchsorted(E, tab.x[0])
        c = np.zeros(len(E)); c[j:j + len(tab.y)] = tab.y; cols[mt] = c
    res = {}
    for g in (203, 204, 205, 206, 207):
        if g not in data.reactions:
            continue
        tab = data.reactions[g].xs[T]; j = np.searchsorted(E, tab.x[0])
        y = np.zeros(len(E)); y[j:j + len(tab.y)] = tab.y
        if y.max() <= 0:
            continue
        cand = [mt for mt in nonred if mt not in (2,) and (cols[mt] * (y > 0)).sum() > 0]
        w = 1.0 / np.maximum(y, y.max() * 1e-8)
        A = np.array([cols[m] * w for m in cand]).T
        coef, _ = nnls(A, y * w)
        coef_r = {int(m): float(np.round(c * 2) / 2) for m, c in zip(cand, coef) if np.round(c * 2) / 2 > 0}
        fit = sum((c * cols[m] for m, c in coef_r.items()), np.zeros(len(E)))
        mask = y > y.max() * 1e-4
        err = float(np.max(np.abs(fit[mask] - y[mask]) / y[mask])) if mask.any() else 0.0
        res[g] = (coef_r, err)
    return res


def prep(nuc):
    d = os.path.join(ROOT, "cov", nuc)
    bounds, gxs, cov = read_errorr(os.path.join(d, "tape22"))
    lump = lumped_map(os.path.join(ROOT, "endf", ENDF[nuc]))
    data = openmc.data.IncidentNeutron.from_hdf5(uqlib.base_h5(nuc))
    nonred = {int(mt) for mt, rx in data.reactions.items() if not rx.redundant}
    covmts = sorted({int(a) for (a, b) in cov if a == b})
    params = uqlib.build_params(covmts, lump, nonred)
    ng = len(bounds) - 1
    C, sd_raw = uqlib.assemble_cov(params, cov, ng)
    C, neg = uqlib.psd(C)
    targets = sorted({r for _, t in params for r in t})
    unc = [int(x) for x in sorted(nonred - set(targets))]
    print(f"{nuc}: {len(params)} params, {len(targets)} reactions sampled; no covariance for {unc}; "
          f"PSD neg-eig fraction {neg:.2e}; sd>100% capped in {int((sd_raw > 1).sum())} (param,group) cells", flush=True)
    kp = heatr_partials(nuc, targets)
    kE_tot, K_tot = kp.pop("_total")
    # check: sum of partial kermas of sampled reactions vs total kerma at 14 MeV
    s14 = sum(np.interp(14.1e6, E, k) * float(data.reactions[mt].xs["294K"](14.1e6)) for mt, (E, k) in kp.items())
    k14 = float(np.interp(14.1e6, kE_tot, K_tot))
    lib14 = float(data.reactions[301].xs["294K"](14.1e6))
    print(f"  KERMA @14.1 MeV: HEATR total {k14:.4g}, library MT301 {lib14:.4g}, sum of sampled partials {s14:.4g} eV-b", flush=True)
    gas = gas_fit(data)
    for g, (c, err) in gas.items():
        print(f"  MT{g} = " + " + ".join(f"{v:g}*MT{k}" for k, v in c.items()) + f"   (max rel fit err {err:.2e})", flush=True)
    # common energy grid for k_r storage
    kE = np.unique(np.concatenate([E for E, _ in kp.values()])) if kp else np.array([1e-5, 2e7])
    z = {"bounds": bounds, "cov": C, "sd_raw": sd_raw, "kE": kE}
    for mt, (E, k) in kp.items():
        z[f"k_{mt}"] = np.interp(kE, E, k)
    np.savez_compressed(os.path.join(d, "plan.npz"), **z)
    meta = dict(nuclide=nuc, params=params, unsampled_reactions=unc, psd_neg_fraction=neg,
                gas={g: c for g, (c, _) in gas.items()}, gas_fit_err={g: e for g, (_, e) in gas.items()},
                kerma14=dict(heatr_total=k14, library=lib14, sampled_partials=s14),
                group_xs_mts=sorted(gxs))
    json.dump(meta, open(os.path.join(d, "plan.json"), "w"), indent=1)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ENDF)):
        prep(n)

"""Turn raw run records (run_bench.py) into the measured responses of each OKTAVIAN case.

responses(case, rec) -> list of dicts {id, quantity, position_cm, energy_band_MeV, C, C_std}
experiment(case)     -> list of dicts {id, ..., E, E_rel_unc, E_corr_group} in the same order
exp_corr(case)       -> (ids, matrix) or None
"""
import json, os
import numpy as np
import tbr_tables as TT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json")))

BELI_NORM = 0.076   # 6.7-8.4 % normalisation of neutron intensity (beli readme); midpoint
NLEAK_NORM = 0.01   # Nb foil monitor, "less than 1 % (0.4 to 1 %)", not included in the tables
BANDS_BELI = [(10.0, 99.0), (3.0, 10.0), (1.0, 3.0), (0.1, 1.0), (0.01, 0.1)]
BANDS_NLEAK = [(10.0, 99.0), (3.0, 10.0), (1.0, 3.0), (0.1, 1.0)]
# responses kept out of results.json (see NOTES.md): the Li(50) pellet at 63.3 cm lies outside the
# 60 cm sphere and its 6Li rate is 1/7 of the calculated leakage flux there (position or labelling
# problem); the be97 >10 MeV band has a corrupted energy column (non-monotonic upper energies).
EXCLUDE = {"tbr_li": {"rr6@63.3"}, "be97": {"leak_10.1-16.8MeV"}}
TABLE_OF = {"be57": "be57", "be69": "be69", "be97": "be97", "be128": "be128",
            "beli57": "beli57", "beli128": "beli128", "lif60": "lif60", "w40": "w40"}


def _bin(rg, r):
    rg = np.asarray(rg)
    return int(np.clip(np.searchsorted(rg, r, side="right") - 1, 0, len(rg) - 2))


def _bins(case):
    if case in ("lif60", "w40"):
        return DATA["nleak"][case], "lethargy"
    return DATA["beli"][case], "energy"


def band_bins(case, lo, hi):
    """Experimental bins inside [lo, hi] (MeV), snapping the band to bin edges: a bin belongs
    to the band when its centre (log) lies inside."""
    bins, _ = _bins(case)
    sel = [b for b in bins if lo <= np.sqrt(b["lo"] * b["hi"]) < hi]
    return sel


def bands(case):
    out = []
    for lo, hi in (BANDS_NLEAK if case in ("lif60", "w40") else BANDS_BELI):
        sel = band_bins(case, lo, hi)
        if len(sel) < 2:
            continue
        out.append((min(b["lo"] for b in sel), max(b["hi"] for b in sel), sel))
    return out


def experiment(case):
    return [r for r in _experiment(case) if r["id"] not in EXCLUDE.get(case, ())]


def _experiment(case):
    rows = []
    if case in TT.TBR:
        t = TT.TBR[case]
        rows.append(dict(id="tbr", quantity="TBR (tritons per source neutron, Li zone)", position_cm=None,
                         energy_band_MeV=None, E=t["E"], E_rel_unc=t["rsd"], E_corr_group=f"src:{case}"))
        if case in TT.PELLET:
            for r in TT.PELLET[case]:
                pos = r[0]
                for key, iv, ie, q in (("rr6", 1, 2, "6Li(n,a)T per atom per source n x1e-24"),
                                       ("rr7", 3, 4, "7Li(n,n'a)T per atom per source n x1e-24"),
                                       ("al", 7, 8, "27Al(n,a) per atom per source n x1e-24")):
                    if r[iv] is None:
                        continue
                    rows.append(dict(id=f"{key}@{pos}", quantity=q, position_cm=pos, energy_band_MeV=None,
                                     E=r[iv], E_rel_unc=r[ie] / 100, E_corr_group=f"src:{case}"))
        if case in TT.TLD:
            T = TT.TLD[case]
            for (pos, cj3, e), rsd in zip(T["rows"], T["rsd"]):
                rows.append(dict(id=f"tpr@{pos}", quantity="local TPR, tritons per cm3 per source n (Li metal)",
                                 position_cm=pos, energy_band_MeV=None, E=e, E_rel_unc=rsd / 100,
                                 E_corr_group=f"tld:{case}", calc_jendl3=cj3))
    else:
        bins, kind = _bins(case)
        for lo, hi, sel in bands(case):
            w = [(b["hi"] - b["lo"]) if kind == "energy" else np.log(b["hi"] / b["lo"]) for b in sel]
            E = sum(b["v"] * wi for b, wi in zip(sel, w))
            stat = np.sqrt(sum((b["e"] * wi) ** 2 for b, wi in zip(sel, w))) / E
            norm = BELI_NORM if kind == "energy" else NLEAK_NORM
            rows.append(dict(id=f"leak_{lo:.3g}-{hi:.3g}MeV", quantity="leakage neutrons per source neutron in band",
                             position_cm=None, energy_band_MeV=[lo, hi], E=E,
                             E_rel_unc=float(np.sqrt(stat ** 2 + norm ** 2)), E_stat=float(stat), E_norm=norm,
                             E_corr_group=f"norm:{case}",
                             E_unc_note=("statistics + 6.7-8.4% intensity normalisation (midpoint used)" if kind == "energy" else
                                         "statistics + Nb monitor (<1%) ONLY: the compilation gives no detector-efficiency "
                                         "systematic; treat as a lower bound")))
    return rows


def exp_corr(case):
    rows = experiment(case)
    if case in TT.TLD:
        T = TT.TLD[case]
        ids = [f"tpr@{r[0]}" for r in T["rows"]]
        return ids, TT.corr_matrix(T["upper"]).tolist()
    if case in TT.PELLET:
        # stated: errors = uncorrelated statistics + common source-strength systematic
        src = TT.TBR[case]["src"]
        ids = [r["id"] for r in rows]
        u = np.array([r["E_rel_unc"] for r in rows])
        cov = np.full((len(u), len(u)), src ** 2)
        np.fill_diagonal(cov, u ** 2)
        s = np.sqrt(np.diag(cov))
        return ids, (cov / np.outer(s, s)).tolist()
    if case in TT.TBR:
        return None
    ids = [r["id"] for r in rows]
    u = np.array([r["E_rel_unc"] for r in rows]); norm = rows[0]["E_norm"]
    cov = np.full((len(u), len(u)), norm ** 2); np.fill_diagonal(cov, u ** 2)
    s = np.sqrt(np.diag(cov))
    return ids, (cov / np.outer(s, s)).tolist()


def calc(case, rec, which="leak"):
    """Calculated value and MC std of every response id of experiment(case)."""
    out = rec["out"]; res = {}
    if case in TT.TBR:
        res["tbr"] = tuple(out["tbr"])
        p = out["_profile"]; rg = p["r_grid"]
        for r in experiment(case):
            if r["position_cm"] is None:
                continue
            i = _bin(rg, r["position_cm"])
            key = r["id"].split("@")[0]
            if key in ("rr6", "rr7", "al"):
                k = {"rr6": "6", "rr7": "7", "al": "al"}[key]
                res[r["id"]] = (p[k][0][i] * 1e24 * 1e-24, p[k][1][i])  # b/cm2 per source == table units x1e-24
            else:   # tpr = N6 rr6 + N7 rr7 (atoms/b-cm x b/cm2 = 1/cm3)
                v = p["N6"] * p["6"][0][i] + p["N7"] * p["7"][0][i]
                s = np.hypot(p["N6"] * p["6"][1][i], p["N7"] * p["7"][1][i])
                res[r["id"]] = (v, s)
    else:
        L = out["_" + which]; e = np.asarray(L["edges"]); m = np.asarray(L["mean"]); sd = np.asarray(L["std"])
        lo_e, hi_e = e[:-1], e[1:]
        for r in experiment(case):
            lo, hi = r["energy_band_MeV"]
            k = (lo_e >= lo * (1 - 1e-9)) & (hi_e <= hi * (1 + 1e-9))
            res[r["id"]] = (float(m[k].sum()), float(np.sqrt((sd[k] ** 2).sum())))
    return res


def spectrum(case, rec, which="leak"):
    """Fine-bin C and E on the experimental binning (for plots/NOTES)."""
    bins, kind = _bins(case)
    L = rec["out"]["_" + which]; e = np.asarray(L["edges"]); m = np.asarray(L["mean"]); sd = np.asarray(L["std"])
    rows = []
    for b in bins:
        k = (e[:-1] >= b["lo"] * (1 - 1e-9)) & (e[1:] <= b["hi"] * (1 + 1e-9))
        w = (b["hi"] - b["lo"]) if kind == "energy" else np.log(b["hi"] / b["lo"])
        rows.append(dict(lo=b["lo"], hi=b["hi"], E=b["v"], E_err=b["e"], C=float(m[k].sum() / w),
                         C_err=float(np.sqrt((sd[k] ** 2).sum()) / w)))
    return rows

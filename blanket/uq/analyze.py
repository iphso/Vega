"""Summarise the UQ campaign (run_campaign.sh) into blanket/uq/results.json.

Design: correlated sampling. Every perturbed sample i and the nominal library are run
at the same two transport seeds A, B. Because OpenMC seeds each history's random stream
from its index, histories under a small cross-section perturbation stay largely
correlated with the nominal ones, so the paired difference
    D_i(s) = Y_i(s) - Y_nom(s)
carries far less MC noise than Y_i itself (measured: ~0.01-0.02% vs 0.17-0.2% for TBR).
    Dbar_i       = (D_i(A) + D_i(B)) / 2
    var_noise    = var_i (D_i(A) - D_i(B)) / 4         (noise variance of Dbar_i; the nominal run's
                   unshared noise is common to all i, offsets D_i and cancels here)
    var_ND       = var_i(Dbar_i) - var_noise
    rel_ND       = sqrt(var_ND) / mean(Y_nom)
68% / 95% intervals from chi-square(n-1) on var_i(Dbar_i). Also reported: the plain
uncorrelated estimator (spread of Y_i at seed A minus the mean batch variance) as a check.

python3 blanket/uq/analyze.py      (host; needs numpy + scipy)
"""
import json, os, sys
import numpy as np
from scipy import stats

ROOT = os.environ.get("UQ_ROOT", os.path.expanduser("~/runs/blanket/uq"))
HERE = os.path.dirname(os.path.abspath(__file__))
SETS = [("li7", "Li-7"), ("f19", "F-19"), ("be9", "Be-9"), ("li6", "Li-6"),
        ("w", "W-182/183/184/186"), ("ni58", "Ni-58"), ("all", "all of the above, jointly")]
QTY = [("TBR", "TBR_std", "TBR"), ("heat_MW", "heat_MW_std", "heating_total_MW")]


def load(name):
    p = os.path.join(ROOT, "results", name + ".jsonl")
    if not os.path.exists(p):
        return {}
    return {r["i"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def estimate(DA, DB, nom, YA, sA):
    n = len(DA)
    Db = 0.5 * (DA + DB)
    s2 = Db.var(ddof=1)
    # variance (not mean square) of A-B: the nominal run's own unshared noise is common to
    # every sample at a seed and only offsets D_i, it does not spread them
    vn = float(np.var(DA - DB, ddof=1) / 4)
    rel = lambda v: float(np.sqrt(max(v, 0.0)) / nom * 100)
    lo68, hi68 = [s2 * (n - 1) / stats.chi2.ppf(q, n - 1) - vn for q in (0.84, 0.16)]
    # conservative 95% upper limit: no noise subtraction (noise estimate itself is noisy)
    hi95 = s2 * (n - 1) / stats.chi2.ppf(0.05, n - 1)
    unc = YA.var(ddof=1) - np.mean(sA ** 2)
    return dict(n=n, nominal=float(nom), nd_rsd_pct=rel(s2 - vn), nd_rsd_pct_68=[rel(lo68), rel(hi68)],
                nd_rsd_pct_95_upper=rel(hi95), resolved=bool(lo68 > 0),
                residual_noise_rsd_pct=rel(vn), single_run_stat_rsd_pct=float(np.sqrt(np.mean(sA ** 2)) / nom * 100),
                mean_shift_pct=float(Db.mean() / nom * 100), mean_shift_se_pct=float(np.sqrt(s2 / n) / nom * 100),
                min_max_pct=[float(Db.min() / nom * 100), float(Db.max() / nom * 100)],
                uncorrelated_estimator_rsd_pct=rel(unc))


def summarize(name):
    A, B = load(name + "_A"), load(name + "_B")
    nA, nB = load("nom_A")[0], load("nom_B")[0]
    idx = sorted(set(A) & set(B))
    if len(idx) < 3:
        return None
    out = dict(nuclides=A[idx[0]]["nuclides"], n_samples=len(idx), particles_per_run=A[idx[0]]["particles"],
               cpu_wall_h=sum(A[i]["wall_s"] + B[i]["wall_s"] + A[i]["lib_s"] + B[i]["lib_s"] for i in idx) / 3600)
    for key, skey, lab in QTY:
        DA = np.array([A[i][key] - nA[key] for i in idx]); DB = np.array([B[i][key] - nB[key] for i in idx])
        YA = np.array([A[i][key] for i in idx]); sA = np.array([A[i][skey] for i in idx])
        out[lab] = estimate(DA, DB, 0.5 * (nA[key] + nB[key]), YA, sA)
    # per-cell heating (no batch std stored per cell: residual noise from A/B as above)
    cells = {}
    for c in nA["heat_cells_MW"]:
        DA = np.array([A[i]["heat_cells_MW"][c] - nA["heat_cells_MW"][c] for i in idx])
        DB = np.array([B[i]["heat_cells_MW"][c] - nB["heat_cells_MW"][c] for i in idx])
        nom = 0.5 * (nA["heat_cells_MW"][c] + nB["heat_cells_MW"][c])
        Db = 0.5 * (DA + DB); v = Db.var(ddof=1) - np.var(DA - DB, ddof=1) / 4
        cells[c] = dict(nominal_MW=float(nom), nd_rsd_pct=float(np.sqrt(max(v, 0)) / nom * 100))
    out["heating_by_cell"] = cells
    return out


if __name__ == "__main__":
    res = {}
    for s, lab in SETS:
        o = summarize(s)
        if not o:
            continue
        o["label"] = lab
        res[s] = o
        for _, _, q in QTY:
            t = o[q]
            print(f"{s:5s} n={t['n']:2d} {q:17s} ND {t['nd_rsd_pct']:.3f}% [68%: {t['nd_rsd_pct_68'][0]:.3f}-{t['nd_rsd_pct_68'][1]:.3f}, "
                  f"95% UL {t['nd_rsd_pct_95_upper']:.3f}]  noise {t['residual_noise_rsd_pct']:.4f}%  "
                  f"(single-run stat {t['single_run_stat_rsd_pct']:.3f}%, uncorrelated est {t['uncorrelated_estimator_rsd_pct']:.3f}%)  "
                  f"shift {t['mean_shift_pct']:+.3f}+-{t['mean_shift_se_pct']:.3f}%")
    indiv = [k for k in res if k != "all"]
    if indiv:
        for _, _, q in QTY:
            qs = float(np.sqrt(sum(res[k][q]["nd_rsd_pct"] ** 2 for k in indiv)))
            res.setdefault("quadrature_sum_of_individual", {})[q] = qs
            print(f"quadrature sum of individual sets, {q}: {qs:.3f}%")
    # sanity checks (300k histories, seed 777)
    chk = {}
    b, u, l = load("chk_base").get(0), load("chk_unity").get(0), load("chk_li6nt").get(0)
    if b and u and l:
        chk = dict(unperturbed_pipeline_TBR=[b["TBR"], u["TBR"]], unperturbed_pipeline_identical=abs(b["TBR"] - u["TBR"]) < 1e-12,
                   li6_nt_plus10pct_dTBR_pct=(l["TBR"] / b["TBR"] - 1) * 100,
                   li6_nt_plus10pct_sensitivity=(l["TBR"] / b["TBR"] - 1) / 0.10,
                   li6_nt_plus10pct_dheat_pct=(l["heat_MW"] / b["heat_MW"] - 1) * 100,
                   li6_nt_plus10pct_single_run_stat_pct=b["TBR_std"] / b["TBR"] * 100)
        res["sanity_checks"] = chk
        print(json.dumps(chk, indent=1))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)

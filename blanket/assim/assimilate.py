"""Integral-experiment data assimilation for the ARC TBR (generalised linear least squares, the
"representativity" method of criticality safety; here with sampled sensitivities).

The nuclear-data UQ (blanket/uq) drew S joint perturbations of the ENDF/B-VIII.0 data and ran
ARC on each; the benchmarks (blanket/bench/*/results.json) ran the same S perturbations. Each
sample is run twice with independent transport seeds (A, B), so the cross-covariance
cov(x_A, y_B) estimates the nuclear-data covariance with the Monte Carlo noise removed in
expectation.

With l = ln(response / nominal) and d = ln(E / C) the benchmark discrepancies:
    Sigma = ND covariance of [benchmarks b, ARC t] (from the samples)
    G     = Sigma_bb + V_exp + V_mc + V_geom + V_model
    ln TBR_post - ln TBR_prior = Sigma_tb G^-1 d
    var_post = Sigma_tt - Sigma_tb G^-1 Sigma_bt
    chi2 = d^T G^-1 d (consistency of the experiments with the prior and its stated errors)
    representativity r_i = Sigma_ti / sqrt(Sigma_tt Sigma_ii)

Guards (S = 30 is small):
- Sigma_bb is shrunk toward its diagonal (Ledoit-Wolf intensity, estimated);
- a permutation null: re-pair ARC samples with benchmark samples at random S_perm times and
  redo the update. Any variance reduction the real pairing shows must stand clear of that
  null's spread, or it is a chance correlation, not information.
- V_model covers the nuclides the samples do not perturb (O, C, Pb, Fe, Cr ... and every
  secondary distribution): a stated per-response relative error (--model-unc), reported with
  chi2 so it can be judged.

Run: python3 blanket/assim/assimilate.py [--bench oktavian fns] [--model-unc 0.03]
"""
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # small matrices; threads only thrash
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
EFF = 0.05  # relative detector-efficiency systematic added where a spectrum quotes none


def load_arc(set_name="all"):
    rows = {}
    for seed in "AB":
        for line in open(os.path.join(B, "uq/raw/results", f"{set_name}_{seed}.jsonl")):
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["i"], {})[seed] = r["TBR"]
    idx = sorted(i for i in rows if len(rows[i]) == 2)
    return idx, np.array([rows[i]["A"] for i in idx]), np.array([rows[i]["B"] for i in idx])


def load_bench(names):
    """-> list of responses (dicts) and sample arrays A, B of shape (S, n)."""
    resp, A, Bs = [], [], []
    for name in names:
        p = os.path.join(B, "bench", name, "results.json")
        if not os.path.exists(p):
            print(f"  (no {p})"); continue
        R = json.load(open(p))
        by_id = {}
        for case, c in R["cases"].items():
            for r in c["responses"]:
                by_id[r["id"]] = by_id[f"{case}:{r['id']}"] = dict(r, case=case, bench=name)
        smp = R.get("samples", {}).get("all")
        if not smp:
            print(f"  ({name}: no perturbed-library samples)"); continue
        ids = smp["response_ids"]
        a, b = np.array(smp["A"], float), np.array(smp["B"], float)
        if a.shape[0] == len(ids) and a.shape[1] != len(ids):
            a, b = a.T, b.T
        for k, i in enumerate(ids):
            if i not in by_id:
                print(f"  ({name}: sample response {i} has no C/E entry, skipped)"); continue
            resp.append(by_id[i]); A.append(a[:, k]); Bs.append(b[:, k])
        corr = R.get("exp_correlations", {})
        for case, cm in corr.items():
            for r in resp:
                if r["bench"] == name and r["case"] == case and r["id"] in cm["ids"]:
                    r["_corr"] = (name, case, cm["ids"], np.array(cm["matrix"], float))
    return resp, (np.array(A).T if A else None), (np.array(Bs).T if Bs else None)


def drop(r):
    """Responses left out of the fit (reasons in bench/*/NOTES.md)."""
    i, c, b = r["id"], r["case"], r["bench"]
    return (i.endswith("_int")                                   # FNS integrals: sums of points already in
            or (b == "fns" and i.startswith("t7_ne213"))         # NE213 T7 has the JENDL 7Li folded in
            or (b == "fns" and c == "li2o" and i.startswith("t6_lg") and (r.get("position_cm") or 99) < 50)  # Li-glass over-subtraction in front
            # Ni-58(n,p) inside the Be: C/E climbs 1.0 -> 1.2 with depth, a Be transport signature the
            # samples cannot represent (Be-9 secondary distributions are not sampled). Kept at the face only;
            # with all six the fit reads it as Ni-58 data and shifts ARC TBR +0.27% (p 0.002).
            or (b == "fns" and c == "be" and "ni58_np" in i and (r.get("position_cm") or 0) > 20.5))


def extra_model(r):
    """Extra relative model error: FNS Be thermal responses run 20-30% high in every library since 1991,
    most likely unmodelled absorbing impurities (boron) in the Be, not Be-9 data."""
    if r["bench"] == "fns" and r["case"] == "be" and r["id"].split("_")[0] in ("t6", "fc", "foil") \
            and any(k in r["id"] for k in ("t6_lg", "u235", "au197")):
        return 0.15
    return 0.0


def exp_cov(resp):
    """Experimental relative covariance: published correlation matrices where given, else a fully
    correlated share per E_corr_group (the group's systematic is taken as the smallest
    E_rel_unc in the group, the rest uncorrelated)."""
    n = len(resp)
    sd = np.array([r["E_rel_unc"] for r in resp])
    C = np.diag(sd ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = resp[i], resp[j]
            if "_corr" in ri and "_corr" in rj and ri["_corr"][:2] == rj["_corr"][:2]:
                ids, M = ri["_corr"][2], ri["_corr"][3]
                rho = M[ids.index(ri["id"]), ids.index(rj["id"])]
                if rho == 0:
                    rho = M[ids.index(rj["id"]), ids.index(ri["id"])]
                C[i, j] = C[j, i] = rho * sd[i] * sd[j]
            elif ri.get("E_corr_group") and ri.get("E_corr_group") == rj.get("E_corr_group") and ri["bench"] == rj["bench"]:
                g = [r["E_rel_unc"] for r in resp if r.get("E_corr_group") == ri["E_corr_group"] and r["bench"] == ri["bench"]]
                s = min(g)
                C[i, j] = C[j, i] = s * s
    # spectra whose quoted error is statistics only (no detector-efficiency systematic): add EFF,
    # fully correlated within the case
    lb = [k for k, r in enumerate(resp) if "lower bound" in r.get("E_unc_note", "")]
    for i in lb:
        for j in lb:
            if (resp[i]["bench"], resp[i]["case"]) == (resp[j]["bench"], resp[j]["case"]):
                C[i, j] += EFF ** 2
    return C


def nd_cov(LA, LB):
    """Seed-crossed covariance (MC noise cancels in expectation), symmetrised and made PSD."""
    S = LA.shape[0]
    a, b = LA - LA.mean(0), LB - LB.mean(0)
    C = (a.T @ b + b.T @ a) / (2 * (S - 1))
    w, V = np.linalg.eigh(C)
    return (V * np.clip(w, 0, None)) @ V.T


def shrink(C, LA, LB):
    """Ledoit-Wolf style shrinkage of the off-diagonal toward zero (intensity from the sample
    variance of the covariance estimates)."""
    S = LA.shape[0]
    x = 0.5 * ((LA - LA.mean(0)) + (LB - LB.mean(0)))
    D = np.diag(np.diag(C))
    off = C - D
    if not np.any(off):
        return C, 0.0
    var = np.zeros_like(C)
    for s in range(S):
        var += (np.outer(x[s], x[s]) - C) ** 2
    var /= S * (S - 1)
    lam = float(np.clip(var[~np.eye(len(C), dtype=bool)].sum() / (off ** 2).sum(), 0, 1))
    return D + (1 - lam) * off, lam


def update(Sig, d, V, t=-1):
    b = [k for k in range(len(Sig)) if k != (t % len(Sig))]
    Sbb, Stb, Stt = Sig[np.ix_(b, b)], Sig[t, b], Sig[t, t]
    G = Sbb + V
    Gi = np.linalg.inv(G)
    return dict(shift=float(Stb @ Gi @ d), var_prior=float(Stt), var_post=float(Stt - Stb @ Gi @ Stb),
                chi2=float(d @ Gi @ d), dof=len(d))


def run(names, model_unc, n_perm=2000, set_name="all", only=None, quiet=False):
    idx, tA, tB = load_arc(set_name)
    resp, bA, bB = load_bench(names)
    if bA is None:
        return None
    if only:
        keep = [k for k, r in enumerate(resp) if only(r)]
        resp, bA, bB = [resp[k] for k in keep], bA[:, keep], bB[:, keep]
    S = min(len(idx), bA.shape[0])
    tA, tB, bA, bB = tA[:S], tB[:S], bA[:S], bB[:S]
    t0 = 0.5 * (tA.mean() + tB.mean())
    b0 = 0.5 * (bA.mean(0) + bB.mean(0))
    LA = np.column_stack([np.log(bA / b0), np.log(tA / t0)])
    LB = np.column_stack([np.log(bB / b0), np.log(tB / t0)])
    Sig, lam = shrink(nd_cov(LA, LB), LA, LB)
    d = np.array([np.log(r["E"] / r["C"]) for r in resp])
    V = exp_cov(resp) + np.diag([r.get("C_mc_rel_unc", 0) ** 2 + (r.get("geom_rel_unc") or 0) ** 2 + model_unc ** 2 + extra_model(r) ** 2 for r in resp])
    u = update(Sig, d, V)
    rep = Sig[-1, :-1] / np.sqrt(Sig[-1, -1] * np.clip(np.diag(Sig)[:-1], 1e-30, None))
    # permutation null: break the ARC <-> benchmark pairing
    rng = np.random.default_rng(1)
    red = []
    for _ in range(n_perm):
        p = rng.permutation(S)
        LAp = np.column_stack([LA[:, :-1], LA[p, -1]]); LBp = np.column_stack([LB[:, :-1], LB[p, -1]])
        Sp, _ = shrink(nd_cov(LAp, LBp), LAp, LBp)
        up = update(Sp, d, V)
        red.append(1 - np.sqrt(up["var_post"] / up["var_prior"]))
    red = np.array(red)
    real = 1 - np.sqrt(u["var_post"] / u["var_prior"])
    sd_t = float(np.sqrt(max(np.mean((LA[:, -1] - LA[:, -1].mean()) * (LB[:, -1] - LB[:, -1].mean())) * S / (S - 1), 0)))
    out = dict(benchmarks=names, n_responses=len(resp), samples=S, shrinkage=lam, model_unc=model_unc,
               # the joint PSD projection (S << n) inflates the ARC diagonal by an amount that depends on
               # the benchmark set; quote the band on ARC's own seed-crossed prior, narrowed by the fit's fraction
               arc_tbr_prior=t0, arc_nd_rsd_prior_pct=100 * sd_t,
               arc_nd_rsd_post_pct=100 * sd_t * (1 - real), arc_tbr_post=t0 * np.exp(u["shift"]),
               joint_prior_pct=100 * np.sqrt(u["var_prior"]),
               shift_pct=100 * u["shift"], chi2=u["chi2"], dof=u["dof"],
               band_reduction=real, null_reduction_mean=float(red.mean()), null_reduction_p95=float(np.quantile(red, 0.95)),
               p_value=float((red >= real).mean()),
               responses=[dict(id=r["id"], bench=r["bench"], case=r["case"], CE=r["C"] and r["E"] and r["C"] / r["E"],
                               representativity=float(rep[k]), nd_rsd_pct=float(100 * np.sqrt(Sig[k, k]))) for k, r in enumerate(resp)])
    if not quiet:
        print(f"{'+'.join(names)}: {len(resp)} responses, S={S}, shrink {lam:.2f}, model unc {model_unc:.0%}")
        print(f"  ARC TBR {t0:.4f} ±{out['arc_nd_rsd_prior_pct']:.2f}% -> {out['arc_tbr_post']:.4f} ±{out['arc_nd_rsd_post_pct']:.2f}% "
              f"(shift {out['shift_pct']:+.2f}%), chi2/dof {u['chi2']:.1f}/{u['dof']}")
        print(f"  band narrowed {real:.1%}; permutation null mean {red.mean():.1%}, 95% {np.quantile(red, .95):.1%} -> p = {out['p_value']:.3f}")
        top = sorted(out["responses"], key=lambda r: -abs(r["representativity"]))[:6]
        for r in top:
            print(f"    {r['bench']:9s} {r['id']:28s} C/E {r['CE']:.3f}  ND {r['nd_rsd_pct']:.2f}%  r {r['representativity']:+.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=["oktavian", "fns"])
    ap.add_argument("--model-unc", type=float, default=0.03)
    ap.add_argument("--exclude-case", nargs="*", default=["oktavian:be97"],
                    help="bench:case pairs left out (be97 is internally inconsistent, see bench/oktavian/NOTES.md)")
    ap.add_argument("--json")
    a = ap.parse_args()
    excl = set(a.exclude_case)
    res = {}
    for names in [[n] for n in a.bench] + ([a.bench] if len(a.bench) > 1 else []):
        r = run(names, a.model_unc, only=lambda r: f"{r['bench']}:{r['case']}" not in excl and not drop(r))
        if r:
            res["+".join(names)] = r
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)

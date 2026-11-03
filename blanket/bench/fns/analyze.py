"""Build results.json (C/E, experimental correlations, perturbed-sample responses) and print tables.

  python3 blanket/bench/fns/analyze.py          (host python, numpy only; reads ~/runs/blanket/bench/fns)

Calculated value C for every response:
  C = C_noroom(spec source) * f_room,  f_room = C_room / C_noroom (separate room-model runs)
except the Li-pellet T6, whose experimental values are already corrected for room return
(Table 7.2 'both effects' column): there C = C_noroom. f_room is ~1 for all threshold responses.
The no-room model is the compilation's own MCNP model (f8-8); room return is a multiplicative
correction from the approximate concrete-room model (the room model alone is ~3x noisier per CPU s).
f_room comes from a CORRELATED pair (same seed, same full-sphere source, void room interior) and is
applied only where it is resolved (|f_room-1| > 3 sigma); elsewhere C = C_noroom.
geom_rel_unc = sqrt(kin^2 + room_u^2): kin = C_kin/C_spec - 1 (D-T kinematic vs flat 0-degree cone
source), room_u = 0.5|f_room-1| where applied, |f_room-1| + sigma_f where not applied (0 for the
room-corrected Li-pellet T6).
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import types
for m in ("openmc", "openmc.data"):   # responses imports fnslib -> openmc; not needed on the host
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["openmc"].data = sys.modules["openmc.data"]
import responses as RS  # noqa: E402

RUNS = os.path.expanduser("~/runs/blanket/bench/fns")
N_SAMPLES = 30


def load(p):
    return json.load(open(os.path.join(RUNS, p)))


_coarse = {}


def C_nr_coarse(case, i):
    """C from the coarse-binned no-room run (same binning as the kinematic-source variant)."""
    if case not in _coarse:
        R = RS.li2o_all(False) if case == "li2o" else RS.be_all()
        _coarse[case] = RS.compute(R, load(f"nominal/{case}_noroom.json"))
    return _coarse[case][i][0]


def case_table(case):
    ff = f"nominal/{case}_noroom_ff.json"   # same model, 1 mm bins near the faces
    nr = load(ff) if os.path.exists(os.path.join(RUNS, ff)) else load(f"nominal/{case}_noroom.json")
    kn = load(f"nominal/{case}_noroom_kin.json")
    # room factor from the correlated pair (same seed, same full-sphere source, void room interior)
    rm, pnr = load(f"nominal/{case}_pair_room.json"), load(f"nominal/{case}_pair_noroom.json")
    rm_air = load(f"nominal/{case}_room.json")   # independent run with room air: consistency check only
    R_nr = RS.li2o_all(False) if case == "li2o" else RS.be_all()
    C_nr, C_rm, C_kn = RS.compute(R_nr, nr), None, RS.compute(R_nr, kn)
    C_pnr = RS.compute(R_nr, pnr)
    C_air = RS.compute(R_nr, rm_air)
    if case == "li2o":
        R_rm = RS.li2o_all(True)    # same ids, pellet T6 E = self-shielding-only column
        C_rm = RS.compute(R_rm, rm)
        E_rm = {r["id"]: r for r in R_rm}
    else:
        C_rm = RS.compute(R_nr, rm)
        E_rm = {r["id"]: r for r in R_nr}
    rows = []
    for r in R_nr:
        i = r["id"]
        c0, s0 = C_nr[i]
        if i not in C_rm:
            continue
        cr, sr = C_rm[i]
        cp, sp_ = C_pnr[i]
        f_room = cr / cp if cp > 0 else float("nan")
        # upper bound (treats the correlated pair as independent)
        f_sd = abs(f_room) * math.hypot(sr / cr if cr else 0, sp_ / cp if cp else 0)
        f_air = C_air[i][0] / c0 if c0 > 0 else float("nan")
        room_corrected_E = i.startswith("t6_lsc") or i == "T6_lsc_int"
        # apply the room factor only where it is resolved (> 3 sigma, sigma an upper bound);
        # elsewhere C = C_noroom and |f_room - 1| + f_sd enters the modelling uncertainty instead
        apply_room = (not room_corrected_E) and abs(f_room - 1) > 3 * f_sd
        C = c0 * f_room if apply_room else c0
        C_sd = C * math.hypot(s0 / c0, f_sd / f_room) if apply_room else s0
        kin = C_kn[i][0] / C_nr_coarse(case, i) - 1
        room_u = 0.5 * abs(f_room - 1) if apply_room else (0.0 if room_corrected_E else min(abs(f_room - 1) + f_sd, 0.5))
        geom = math.hypot(kin, room_u)
        rows.append(dict(r, C=C, C_mc=C_sd / C, f_room=f_room, f_room_sd=f_sd, kin=kin, geom=geom, apply_room=apply_room,
                         room_corrected_E=room_corrected_E, C_noroom=c0, C_room=cr, f_room_air_indep=f_air,
                         E_room_col=E_rm[i]["E"] if i in E_rm else None))
    return rows


def exp_corr(rows):
    ids = [r["id"] for r in rows]
    n = len(rows)
    M = np.zeros((n, n))
    for a, ra in enumerate(rows):
        for b, rb in enumerate(rows):
            if a == b:
                M[a, b] = ra["tot"] ** 2
            elif ra["corr_group"] == rb["corr_group"]:
                M[a, b] = ra["sys"] * rb["sys"]
    # integrals are linear combinations of the point values: rebuild their rows exactly
    idx = {i: k for k, i in enumerate(ids)}
    pts = [k for k, r in enumerate(rows) if "integral" not in r]
    W = np.zeros((n, n))
    for k, r in enumerate(rows):
        if "integral" in r:
            for i, w in r["integral"]:
                W[k, idx[i]] = w * rows[idx[i]]["E"] / r["E"]
        else:
            W[k, k] = 1.0
    Mp = M.copy()
    for k, r in enumerate(rows):   # zero integral rows before propagating
        if "integral" in r:
            Mp[k, :] = 0; Mp[:, k] = 0
    Cov = W @ Mp @ W.T
    for k in pts:
        Cov[k, k] = M[k, k]
    sd = np.sqrt(np.diag(Cov))
    return ids, (Cov / np.outer(sd, sd)).round(4).tolist()


def samples(case, rows):
    R = RS.li2o_all(False) if case == "li2o" else RS.be_all()
    Cbest = {r["id"]: r["C"] for r in rows}
    # sample value = best nominal C x Y_i(s) / Y_nom(s): ratio to the nominal library at the same seed
    # (index -1), so the high-statistics nominal, fine face bins and room factor carry over exactly
    out = {}
    for s in ("A", "B"):
        raw = []
        for i in range(-1, N_SAMPLES):
            p = os.path.join(RUNS, "samples", f"{case}_{s}_{i}.json")
            raw.append(RS.compute(R, json.load(open(p))) if os.path.exists(p) else None)
        if raw[0] is None:
            out[s] = [None] * len(raw); continue
        out[s] = [None if Y is None else {k: Cbest[k] * Y[k][0] / raw[0][k][0] for k in Cbest} for Y in raw]
    return out


def nd_stats(ids, A, B, nomA, nomB):
    """Correlated-sampling estimator of the UQ campaign (blanket/uq/NOTES.md method 6):
    D_i(s) = Y_i(s) - Y_nom(s);  var_ND = var_i(mean_s D_i) - var_i(D_A - D_B)/4.
    Returns per response: ND rel sd, residual MC noise rel sd of mean_s D_i, resolved flag, and the
    correlation of the response with the paired ARC TBR (same i, same seeds)."""
    tbr = {}
    for s in ("A", "B"):
        rows = [json.loads(l) for l in open(os.path.expanduser(f"~/runs/blanket/uq/results/all_{s}.jsonl"))]
        nom = json.loads(open(os.path.expanduser(f"~/runs/blanket/uq/results/nom_{s}.jsonl")).readline())["TBR"]
        tbr[s] = np.array([r["TBR"] for r in sorted(rows, key=lambda r: r["i"])][:N_SAMPLES]) - nom
    dT = 0.5 * (tbr["A"] + tbr["B"])
    out = {}
    for k, i in enumerate(ids):
        a, b = np.array(A[k]), np.array(B[k])
        y0 = 0.5 * (nomA[k] + nomB[k])
        DA, DB = a - nomA[k], b - nomB[k]
        Dm = 0.5 * (DA + DB)
        noise_var = np.var(DA - DB, ddof=1) / 4
        tot_var = np.var(Dm, ddof=1)
        nd_var = tot_var - noise_var
        out[i] = dict(nd_rel_sd=float(math.sqrt(max(nd_var, 0)) / y0), noise_rel_sd=float(math.sqrt(noise_var) / y0),
                      total_rel_sd=float(math.sqrt(tot_var) / y0),
                      resolved=bool(nd_var > 3 * noise_var),
                      corr_with_TBR=float(np.corrcoef(Dm, dT)[0, 1]),
                      seedAB_disagreement_rel=float(abs(nomA[k] - nomB[k]) / y0))
    return out


def main():
    res = {"benchmark": "fns", "cases": {}, "exp_correlations": {}, "samples": {},
           "notes": "see blanket/bench/fns/NOTES.md"}
    sample_ids, A, B, nomA, nomB = [], [], [], [], []
    for case, desc, src in [
        ("li2o", "FNS 60 cm-thick Li2O cylindrical assembly (1983), r_eq 31.5 cm, 14 MeV point source 20 cm in front",
         ["JAERI-M 94-014 ch.7 (li2o/read.me Tables 7.1-7.21)", "JAERI-M 86-182 (Maekawa et al. 1986)"]),
        ("be", "FNS Be cylindrical assembly (1988), r_eq 31.5 cm, 45.54 cm thick, source 20 cm in front",
         ["JAERI-M 94-014 ch.8 (be/read.me, t8-4..6, t8-16, MCNP input f8-8.txt)",
          "Maekawa et al., Fusion Technol. 19 (1991) 1949"]),
    ]:
        rows = case_table(case)
        resp = []
        for r in rows:
            resp.append({"id": r["id"], "quantity": r["quantity"], "position_cm": r["position_cm"],
                         "energy_band_MeV": [b / 1e6 for b in r["band"]] if r.get("band") else None, "E": r["E"], "E_rel_unc": round(r["tot"], 5),
                         "E_sys_rel_unc": round(r["sys"], 5), "E_rnd_rel_unc": round(r["rnd"], 5),
                         "E_corr_group": r["corr_group"], "C": r["C"], "C_mc_rel_unc": round(r["C_mc"], 5),
                         "CE": r["C"] / r["E"], "geom_rel_unc": round(r["geom"], 5),
                         "f_room": round(r["f_room"], 5), "f_room_applied": r["apply_room"], "f_room_sd_upper": round(r["f_room_sd"], 5),
                         "source_kinematics_effect": round(r["kin"], 5),
                         "note": r["note"] + ("" if not r["room_corrected_E"] else "; C from the no-room model")})
        res["cases"][case] = {"description": desc, "geometry_sources": src, "responses": resp}
        ids, M = exp_corr(rows)
        res["exp_correlations"][case] = {"ids": ids, "matrix": M}
        sm = samples(case, rows)
        if all(v is not None for v in sm["A"] + sm["B"]):
            keys = [r["id"] for r in rows]
            sample_ids += [f"{case}:{k}" for k in keys]
            A += [[sm["A"][i + 1][k] for i in range(N_SAMPLES)] for k in keys]
            B += [[sm["B"][i + 1][k] for i in range(N_SAMPLES)] for k in keys]
            nomA += [sm["A"][0][k] for k in keys]
            nomB += [sm["B"][0][k] for k in keys]
        else:
            print(f"[{case}] samples incomplete: {sum(v is not None for v in sm['A'] + sm['B'])}/{2 * (N_SAMPLES + 1)}")
    if sample_ids:
        res["samples"]["all"] = {
            "response_ids": sample_ids,
            "layout": "A[j][i] = response j (response_ids order) with ARC UQ sample i (0..29) of set all_A "
                      "(transport seed 777); B likewise with seed 778. Same i = same perturbation as "
                      "blanket/uq/raw/results/all_{A,B}.jsonl. Values = nominal C (cases) x Y_i/Y_nom at the same seed; nominal_A/B therefore equal C.",
            "nominal_A": nomA, "nominal_B": nomB, "A": A, "B": B,
            "stats": nd_stats(sample_ids, A, B, nomA, nomB)}
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    return res


def write_tables(res):
    """Regenerate the tables between the AUTO markers in NOTES.md."""
    stats = res["samples"].get("all", {}).get("stats", {})
    L = []
    for case, c in res["cases"].items():
        L.append(f"\n#### {case}: every response\n")
        L.append("| id | pos cm | E | E unc % (sys) | C/E | MC % | model % | f_room | ND rsd % | MC-noise % | r(TBR) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in c["responses"]:
            st = stats.get(f"{case}:{r['id']}")
            nd = (f"{100*st['nd_rel_sd']:.2f}{'' if st['resolved'] else '*'} | {100*st['noise_rel_sd']:.2f} | "
                  f"{st['corr_with_TBR']:+.2f}") if st else "- | - | -"
            pos = "-" if r["position_cm"] is None else f"{r['position_cm']:.2f}"
            L.append(f"| {r['id']} | {pos} | {r['E']:.3e} | {100*r['E_rel_unc']:.1f} ({100*r['E_sys_rel_unc']:.1f}) | "
                     f"{r['CE']:.3f} | {100*r['C_mc_rel_unc']:.2f} | {100*r['geom_rel_unc']:.1f} | "
                     f"{r['f_room']:.3f}{'' if r['f_room_applied'] else ' (n.a.)'} | {nd} |")
    # series summary
    S = ["| case | series | n | positions cm | C/E range | C/E integral or mean | E unc % |", "|---|---|---|---|---|---|---|"]
    for case, c in res["cases"].items():
        series = {}
        for r in c["responses"]:
            key = r["id"] if r["position_cm"] is None else r["id"].rsplit("_", 1)[0]
            series.setdefault(key, []).append(r)
        for k, rs in series.items():
            ce = [r["CE"] for r in rs]
            pos = [r["position_cm"] for r in rs if r["position_cm"] is not None]
            S.append(f"| {case} | {k} | {len(rs)} | {f'{min(pos):.1f}-{max(pos):.1f}' if pos else 'integral'} | "
                     f"{min(ce):.3f}-{max(ce):.3f} | {np.mean(ce):.3f} | "
                     f"{100*min(r['E_rel_unc'] for r in rs):.1f}-{100*max(r['E_rel_unc'] for r in rs):.1f} |")
    # ND summary: integrals + every 'resolved' fraction per series
    N = []
    if stats:
        N.append("| response | ND rsd % | MC noise % (paired) | resolved | r with ARC ΔTBR |")
        N.append("|---|---|---|---|---|")
        for k, st in stats.items():
            rid = k.split(":", 1)[1]
            if rid[0].isupper() or rid.startswith("phi_gt10") or any(rid.endswith(x) for x in
                    ("_22.4", "_40.2", "_60.5", "_70.7", "_21.6", "_41.6", "_61.8", "_25.17", "_40.35", "_60.59",
                     "_26.66", "_41.90", "_57.14", "_19.85", "_20.00")):
                N.append(f"| {k} | {100*st['nd_rel_sd']:.2f} | {100*st['noise_rel_sd']:.2f} | "
                         f"{'yes' if st['resolved'] else 'no'} | {st['corr_with_TBR']:+.2f} |")
        nres = sum(st["resolved"] for st in stats.values())
        N.append(f"\n{nres} of {len(stats)} sample responses resolve ND above MC noise (ND var > 3x noise var); "
                 "the full list is in the tables at the end (ND rsd marked * where not resolved).")
    p = os.path.join(HERE, "NOTES.md")
    if not os.path.exists(p):
        return
    t = open(p).read()
    for tag, body in (("TABLES", L), ("SUMMARY", S), ("ND", N)):
        a, b = f"<!-- AUTO:{tag} -->", f"<!-- /AUTO:{tag} -->"
        if a in t and b in t:
            t = t[:t.index(a) + len(a)] + "\n" + "\n".join(body) + "\n" + t[t.index(b):]
    open(p, "w").write(t)


if __name__ == "__main__":
    res = main()
    write_tables(res)
    for case, c in res["cases"].items():
        print(f"\n=== {case}")
        for r in c["responses"]:
            print(f"{r['id']:26s} E={r['E']:.3e} ±{100*r['E_rel_unc']:4.1f}%  C/E={r['CE']:.3f} "
                  f"mc {100*r['C_mc_rel_unc']:4.1f}%  f_room {r['f_room']:.3f}±{r['f_room_sd_upper']:.3f}  kin {100*r['source_kinematics_effect']:+5.1f}%")

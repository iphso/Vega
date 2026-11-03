"""Build results.json (and spectra.json) from the raw OKTAVIAN runs in ~/runs/blanket/bench/oktavian/results.

  python3 blanket/bench/oktavian/analyze.py [--raw ~/runs/blanket/bench/oktavian/results]

Nominal C: tag nomhi (5M histories). Geometry uncertainty: sens_* tags against sens_ref (same seed,
so differences are correlated-sampling differences), each scaled to a 1-sigma parameter change
(SENS below), plus a detector-position term (+-POS_SIGMA cm through the local radial gradient),
combined in quadrature. ND samples: tags s<i>_A/B (seed 777/778) and nom_A/B; per response
D_i(s) = Y_i(s)/Y_nom(s) - 1, ND variance = var_i(mean_s D_i) - var_i(D_A - D_B)/4.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import responses as R  # noqa: E402
import okt  # noqa: E402

# (tag, cases it applies to, scale of the run's change to a 1-sigma change, label)
SENS = [
    ("sens_lidens+", None, 0.8, "Li density +5% (1 sigma = 4%)"),
    ("sens_lidens-", None, 0.8, "Li density -5%"),
    ("sens_awall", None, 0.5, "A-block inner wall 3 vs 2 mm (half-difference)"),
    ("sens_cdens", None, 1.0, "graphite density +5% (unknown; 1 sigma)"),
    ("sens_pbdens", None, 1.0, "Pb density -5% (block packing unknown; 1 sigma)"),
    ("sens_bedens", None, 1.0, "Be density -2% (1 sigma)"),
    ("sens_cavity", None, 0.5, "inner cavity radius 10 -> 9 cm (1 sigma = 0.5 cm)"),
    ("sens_lifw", None, 1.0, "LiF / W powder density +2% (1 sigma)"),
]
POS_SIGMA = 0.2   # cm, detector position
SOURCES = {
    "tbr": ["JAERI-M 94-014 tbr/readme.txt (Tables 1.1-1.16)",
            "Cetnar et al., JAERI-M 91-062 sec 5.2 pp.70-80 (systems, Table 2 TBR)",
            "Takahashi et al., JAERI-M 91-062 sec 5.3 pp.81-89 (Figs 1-3: radii 10/20/60/80 cm, Be 5.7-17.35 cm + 2.65 cm gap, SUS316 vessels, wall thicknesses p.83)",
            "Sugiyama et al., NEACRP-A-1984-0657 (120 cm Li sphere: natural Li metal, block weights/filling factors, wall thicknesses)",
            "Chuvilin et al., INDC(CCP)-401 (1996) Tables 1-2 (US/PRC Be shells: radii, densities, impurities)"],
    "beli": ["JAERI-M 94-014 beli/readme.txt (Tables 2.1-2.7)",
             "Chuvilin et al., INDC(CCP)-401 (1996) Tables 1-2 (Be shells)",
             "Takahashi et al., JAERI-M 91-062 sec 5.3 (A-block Li shell 20-60 cm, SUS316 walls)",
             "Maekawa et al., JAERI-Data/Code 98-024 p.11 (solid metal shells, NE-218 at ~11 m)"],
    "lif60": ["JAERI-M 94-014 n-leak/report.txt + lif60.dat (Type-I vessel, LiF composition, Table 4.2 source)",
              "JAERI-Data/Code 98-024 Appendix Fig A-15 (Type-I MCNP surfaces: duct r 2.5/2.8, closed 2.5 cm past centre, shell 30.0/30.5)"],
    "w40": ["JAERI-M 94-014 n-leak/report.txt + w40.dat (Type-II vessel, Fig 4.21 MCNP input, Table 4.2 source)",
            "SINBAD NEA-1553/53 OKTAVIAN W abstract (39.9 cm OD, 0.2 cm SUS-304, steel 7.86 g/cm3, W 4.43 g/cm3)"],
}
DESC = {
    "tbr_li": "Li(50): natural Li metal 10-60 cm in SUS316 vessels (C-block 10-20, A-block 20-60); Li2CO3 pellets",
    "tbr_lic": "Li(50)+C(20): as tbr_li with graphite 60-80 cm; LiF TLD pairs (UD-136N/137N)",
    "tbr_pbli": "Pb(10)+Li(40): Pb 10-20 cm, Li A-block 20-60 cm; Li2CO3 pellets",
    "tbr_pblic": "Pb(10)+Li(40)+C(20): Pb 10-20, Li 20-60, graphite 60-80 cm; TLDs",
    "tbr_beli": "Be(11.65)+Li(40): Be 5.7-17.35, gap, Li A-block 20-60 cm; TLDs",
    "tbr_belic": "Be(11.65)+Li(40)+C(20): Be 5.7-17.35, Li 20-60, graphite 60-80 cm; TLDs",
    "be57": "Be shell 5.7-17.35 cm (116.5 mm) leakage spectrum, TOF",
    "be69": "Be shell 6.9-17.35 cm (104.5 mm) leakage spectrum, TOF",
    "be97": "Be shell 9.7-17.35 cm (76.5 mm) leakage spectrum, TOF",
    "be128": "Be shell 12.8-17.35 cm (45.5 mm) leakage spectrum, TOF",
    "beli57": "Be 5.7-17.35 cm inside the 40 cm Li A-block: leakage spectrum, TOF",
    "beli128": "Be 12.8-17.35 cm inside the 40 cm Li A-block: leakage spectrum, TOF",
    "lif60": "LiF powder 61 cm sphere (Type-I vessel) leakage spectrum, TOF at 55 deg",
    "w40": "W powder 40 cm sphere (Type-II vessel, 20 cm void) leakage spectrum, TOF at 55 deg",
}


def load(raw, tag):
    p = os.path.join(raw, tag + ".jsonl")
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["case"]: json.loads(l) for l in open(p) if l.strip()}


def pos_unc(case, rec, rid, pos):
    """Relative change for a +-POS_SIGMA shift of a point detector, from the radial profile."""
    p = rec["out"]["_profile"]; rg = np.asarray(p["r_grid"])
    key = rid.split("@")[0]
    def val(r):
        i = R._bin(rg, r)
        if key == "tpr":
            return p["N6"] * p["6"][0][i] + p["N7"] * p["7"][0][i]
        return p[{"rr6": "6", "rr7": "7", "al": "al"}[key]][0][i]
    v0, vp, vm = val(pos), val(pos + POS_SIGMA), val(pos - POS_SIGMA)
    return float(max(abs(vp - v0), abs(vm - v0)) / v0) if v0 > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/runs/blanket/bench/oktavian/results"))
    a = ap.parse_args()
    nom = load(a.raw, "nomhi")
    ref = load(a.raw, "sens_ref")
    sens = {t: load(a.raw, t) for t, *_ in SENS}
    c10 = load(a.raw, "sens_c10")
    out = dict(benchmark="oktavian", library="ENDF/B-VIII.0 (NNDC HDF5) + ENDF/B-VIII.0 TSL (c_Be, c_Graphite_30p)",
               code="OpenMC 0.16.0, continuous energy", cases={}, exp_correlations={}, samples={},
               geometry_sensitivity={}, notes="see NOTES.md")
    spectra = {}
    for case in okt.ALL_CASES:
        if case not in nom:
            continue
        rec = nom[case]; C = R.calc(case, rec)
        C55 = R.calc(case, rec, "leak55") if case not in okt.TBR_CASES else {}
        resp, gs = [], {}
        for e in R.experiment(case):
            rid = e["id"]; c, cs = C[rid]
            terms = {}
            if case in ref:
                c_ref = R.calc(case, ref[case])[rid][0]
                for tag, _, scale, label in SENS:
                    if case in sens[tag]:
                        terms[label] = (R.calc(case, sens[tag][case])[rid][0] / c_ref - 1) * scale
            if e["position_cm"] is not None:
                pu = pos_unc(case, rec, rid, e["position_cm"])
                if pu is not None:
                    terms[f"detector position +-{POS_SIGMA} cm"] = pu
            if case in c10 and case in ref:
                gs.setdefault(rid, {})["info: graphite 10 instead of 20 cm (not in budget)"] = \
                    R.calc(case, c10[case])[rid][0] / R.calc(case, ref[case])[rid][0] - 1
            # the two Li-density runs bracket one parameter: combine as their mean |change|
            li = [v for k, v in terms.items() if k.startswith("Li density")]
            parts = [v for k, v in terms.items() if not k.startswith("Li density")]
            if li:
                parts.append(np.mean(np.abs(li)))
            geom = float(np.sqrt(sum(v * v for v in parts))) if terms else None
            gs.setdefault(rid, {}).update({k: float(v) for k, v in terms.items()})
            r = dict(id=rid, quantity=e["quantity"], position_cm=e["position_cm"], energy_band_MeV=e["energy_band_MeV"],
                     E=e["E"], E_rel_unc=e["E_rel_unc"], E_corr_group=e["E_corr_group"], C=c, C_mc_rel_unc=cs / c,
                     CE=c / e["E"], geom_rel_unc=geom)
            if rid in C55:   # 4 pi x leakage per steradian in the detector direction (sensitivity only)
                r["C_cone55"] = C55[rid][0]; r["C_cone55_mc_rel_unc"] = C55[rid][1] / C55[rid][0]
            if "E_unc_note" in e:
                r["E_unc_note"] = e["E_unc_note"]
            if "calc_jendl3" in e:
                r["CE_jendl3_published"] = e["calc_jendl3"] / e["E"]
            if rid == "tbr":
                r["CE_jendl3_published"] = R.TT.TBR[case]["ce_j3"]
            resp.append(r)
        kind = "tbr" if case.startswith("tbr") else ("beli" if case.startswith("be") else case)
        out["cases"][case] = dict(description=DESC[case], geometry_sources=SOURCES[kind], responses=resp)
        out["geometry_sensitivity"][case] = gs
        ec = R.exp_corr(case)
        if ec:
            out["exp_correlations"][case] = dict(ids=ec[0], matrix=ec[1])
        if case not in okt.TBR_CASES:
            spectra[case] = dict(total=R.spectrum(case, rec, "leak"), cone55=R.spectrum(case, rec, "leak55"))
    # ---------------------------------------------------------------- ND samples
    nA, nB = load(a.raw, "nom_A"), load(a.raw, "nom_B")
    ids, YA, YB, NA, NB = [], [], [], [], []
    done = [i for i in range(30) if len(load(a.raw, f"s{i}_A")) == len(okt.ALL_CASES)
            and len(load(a.raw, f"s{i}_B")) == len(okt.ALL_CASES)]
    if nA and done:
        for case in okt.ALL_CASES:
            for e in R.experiment(case):
                ids.append(f"{case}:{e['id']}")
        def vec(recs):
            v = []
            for case in okt.ALL_CASES:
                C = R.calc(case, recs[case])
                v += [C[e["id"]][0] for e in R.experiment(case)]
            return v
        NA, NB = vec(nA), vec(nB)
        for i in done:
            YA.append(vec(load(a.raw, f"s{i}_A"))); YB.append(vec(load(a.raw, f"s{i}_B")))
        YA, YB, NA, NB = map(np.asarray, (YA, YB, NA, NB))
        DA, DB = YA / NA - 1, YB / NB - 1
        Dm = 0.5 * (DA + DB)
        noise_var = np.var(DA - DB, axis=0, ddof=1) / 4
        nd_var = np.var(Dm, axis=0, ddof=1) - noise_var
        # ARC TBR of the same samples (blanket/uq/raw/results), same seeds
        uq = os.path.join(HERE, "../../uq/raw/results")
        arc = {s: {json.loads(l)["i"]: json.loads(l)["TBR"] for l in open(os.path.join(uq, f"all_{s}.jsonl"))} for s in "AB"}
        arcn = {s: json.loads(open(os.path.join(uq, f"nom_{s}.jsonl")).readline())["TBR"] for s in "AB"}
        arc_A = np.array([arc["A"][i] for i in done]); arc_B = np.array([arc["B"][i] for i in done])
        arc_D = 0.5 * (arc_A / arcn["A"] - 1 + arc_B / arcn["B"] - 1)
        corr = [float(np.corrcoef(Dm[:, j], arc_D)[0, 1]) for j in range(len(ids))]
        out["samples"]["all"] = dict(
            sample_index=done, seeds={"A": 777, "B": 778}, response_ids=ids,
            layout="A[k][j] = response j of sample sample_index[k] at seed A (absolute value, same units as cases)",
            A=YA.tolist(), B=YB.tolist(), nominal_A=NA.tolist(), nominal_B=NB.tolist(),
            arc_tbr_A=arc_A.tolist(), arc_tbr_B=arc_B.tolist(), arc_tbr_nominal={"A": arcn["A"], "B": arcn["B"]},
            nd_rel_sd=np.sqrt(np.clip(nd_var, 0, None)).tolist(),
            noise_rel_sd_per_sample=np.sqrt(noise_var).tolist(),
            corr_with_arc_tbr=corr,
            particles_per_run=nA[okt.ALL_CASES[0]]["particles"])
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    json.dump(spectra, open(os.path.join(HERE, "spectra.json"), "w"))
    # ------------------------------------------------------------- console summary
    for case, cd in out["cases"].items():
        print(f"\n{case}")
        for r in cd["responses"]:
            g = r["geom_rel_unc"]
            print(f"  {r['id']:24s} E {r['E']:.4g} +-{100*r['E_rel_unc']:.1f}%  C {r['C']:.4g} (mc {100*r['C_mc_rel_unc']:.2f}%"
                  f", geom {100*g if g is not None else float('nan'):.1f}%)  C/E {r['CE']:.3f}"
                  + (f"  [JENDL-3 pub {r['CE_jendl3_published']:.2f}]" if "CE_jendl3_published" in r else ""))
    if out["samples"]:
        s = out["samples"]["all"]
        print(f"\nND samples: {len(s['sample_index'])}  (ND rsd / per-sample noise rsd / corr with ARC TBR)")
        for j, rid in enumerate(s["response_ids"]):
            print(f"  {rid:34s} ND {100*s['nd_rel_sd'][j]:6.2f}%  noise {100*s['noise_rel_sd_per_sample'][j]:5.2f}%  r {s['corr_with_arc_tbr'][j]:+.2f}")


if __name__ == "__main__":
    main()

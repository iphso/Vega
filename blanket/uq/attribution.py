"""Indicative attribution within a nuclide set: correlation of each sample's TBR / heating
change (Dbar_i, see analyze.py) with that sample's mean log-factor of each covariance MT
over the fast range (1-15 MeV) and below 1 MeV. With 20 samples |r| < ~0.45 is noise."""
import json, os, sys
import numpy as np
ROOT = os.path.expanduser("~/runs/blanket/uq")
L = lambda n: {r["i"]: r for r in map(json.loads, open(f"{ROOT}/results/{n}.jsonl"))}
out = {}
for s, nuc in [("f19", "F19"), ("be9", "Be9"), ("li6", "Li6"), ("ni58", "Ni58")]:
    if not os.path.exists(f"{ROOT}/results/{s}_B.jsonl"):
        continue
    A, B, nA, nB = L(s + "_A"), L(s + "_B"), L("nom_A")[0], L("nom_B")[0]
    meta = json.load(open(f"{ROOT}/cov/{nuc}/plan.json"))
    b = np.load(f"{ROOT}/cov/{nuc}/plan.npz")["bounds"]
    fast = (b[:-1] >= 1e6) & (b[1:] <= 15e6); slow = b[1:] <= 1e6
    idx = sorted(set(A) & set(B))
    res = {}
    for q in ("TBR", "heat_MW"):
        D = np.array([0.5 * (A[i][q] - nA[q] + B[i][q] - nB[q]) for i in idx])
        rows = []
        for p, (mt, _) in enumerate(meta["params"]):
            for lab, m in (("1-15MeV", fast), ("<1MeV", slow)):
                x = np.array([np.log(np.load(f"{ROOT}/results/{s}_A_factors/{i}.npz")[nuc][p][m]).mean() for i in idx])
                if x.std() > 1e-6:
                    rows.append((f"MT{mt} {lab}", float(np.corrcoef(x, D)[0, 1])))
        rows.sort(key=lambda r: -abs(r[1]))
        res[q] = rows[:6]
        print(s, q, ", ".join(f"{k}: r={v:+.2f}" for k, v in rows[:6]))
    out[s] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "attribution.json"), "w"), indent=1)

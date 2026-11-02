"""How much do the browser's coarse activation groups lose when the spectrum moves?

The sidecar collapses each reaction to Gc groups in the BASELINE cell's own CE spectrum.
A design change moves the spectrum; the browser then has the right coarse-group track
but the baseline within-group shape. This script measures that error on real spectra:
the 175-group CE fluxes of the damage-tier design variants (../damage/runs/*/coeffs_damage.npz,
same cells, same groups), folded with the baseline's per-group direct reaction rates
(rate_g / flux_g, self-shielded, from runs/VITAMIN-J-175_ww/tallies.npz):
    exact  = sum_g sigma_g phi'_g,      coarse = sum_c sigma_c(baseline) Phi'_c
for the reactions that make the dominant activation products, for several candidate
coarse-group structures. (Self-shielding changes with the variant are not in this test:
it holds sigma_g at the baseline; it isolates the coarse-group collapse.)

Run (host python + numpy): python3 blanket/activation/coarse_groups.py
Writes runs/coarse_groups.json.
"""
import glob, json, os
import numpy as np

ACT = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(ACT, "runs", "VITAMIN-J-175_ww")
meta = json.load(open(os.path.join(base, "meta.json")))
t = np.load(os.path.join(base, "tallies.npz"))
edges = t["group_edges"]                         # ascending
hi = edges[::-1]                                  # hi[g] upper edge of fine group g (0 = highest)
G = len(edges) - 1
named = meta["cells"]

MAX_SD = 0.02   # only reactions whose variant rate is known to 2% (1 sigma) are scored
KEY = {   # dominant producers in reference.json (top heat/activity/dose contributors)
    "tungsten": [("W186", "(n,gamma)"), ("W184", "(n,gamma)"), ("W186", "(n,2n)"), ("W182", "(n,2n)"), ("W180", "(n,gamma)"), ("W182", "(n,p)"), ("W184", "(n,a)")],
    "inconel718": [("Ni58", "(n,p)"), ("Ni58", "(n,2n)"), ("Ni60", "(n,p)"), ("Fe54", "(n,p)"), ("Fe56", "(n,p)"), ("Cr50", "(n,gamma)"),
                   ("Co59", "(n,gamma)"), ("Mn55", "(n,gamma)"), ("Ni62", "(n,gamma)"), ("Nb93", "(n,2n)"), ("Nb93", "(n,gamma)"), ("Ni64", "(n,gamma)"), ("Fe54", "(n,a)")],
    "flibe": [("F19", "(n,2n)"), ("F19", "(n,a)"), ("F19", "(n,p)"), ("Li6", "(n,t)"), ("Li7", "(n,gamma)"), ("Be9", "(n,a)")],
    "beryllium": [("Be9", "(n,a)"), ("Be9", "(n,t)")],
    "TiH2": [("Ti48", "(n,p)"), ("Ti46", "(n,p)"), ("Ti47", "(n,p)"), ("Ti49", "(n,np)"), ("Ti50", "(n,gamma)"), ("H2", "(n,gamma)")],
    "coil": [("Cu63", "(n,gamma)"), ("Cu65", "(n,2n)"), ("Mn55", "(n,gamma)"), ("Fe54", "(n,p)"), ("Ni58", "(n,p)"), ("Fe56", "(n,p)"), ("Ba134", "(n,2n)"), ("Gd158", "(n,gamma)")],
}
CANDIDATES = {
    "4: 14 MeV | 1 MeV | 1 keV | thermal": [12.0e6, 1.0e6, 1.0e3],
    "6": [12.0e6, 5.0e6, 1.0e6, 1.0e5, 1.0e3],
    "8 (sidecar default)": [12.0e6, 5.0e6, 1.0e6, 1.0e5, 1.0e3, 10.0, 0.625],
    "8 alt (finer fast)": [13.0e6, 8.0e6, 3.0e6, 1.0e6, 1.0e5, 1.0e3, 1.0],
}


def cmap_for(interior):
    snapped = []
    for e in interior:
        k = int(np.argmin(np.abs(np.log(hi[1:G] / e)))) + 1
        snapped.append(hi[k])
    lo = hi[1:]
    return np.array([sum(1 for s in snapped if lo[g] < s * (1 - 1e-9)) for g in range(G)])


variants = sorted(d for d in glob.glob(os.path.join(ACT, "..", "damage", "runs", "VITAMIN-J-175_*")) if os.path.exists(os.path.join(d, "coeffs_damage.npz")))
out = {"note": __doc__.split("\n\n")[1], "variants": [os.path.basename(v) for v in variants], "candidates": {}}
for cname, interior in CANDIDATES.items():
    cm = cmap_for(interior)
    Gc = cm.max() + 1
    rows = []
    for v in variants:
        z = np.load(os.path.join(v, "coeffs_damage.npz"))
        vcells = [str(x) for x in z["cells"]]
        vflux = z["ce_flux"]                       # [cell, 175], group 0 = highest
        vrsd = z["ce_flux_rel_sd"]
        tag = os.path.basename(v)
        for ci, n in enumerate(named):
            mname = meta["materials"][n]["name"]
            if "fw_material" in tag and n == "fw":
                continue                           # different material: needs its own sidecar
            info = meta["rr_tallies"][mname]
            rr = t[f"rr_{mname}"][info["cells"].index(n)]
            phi0 = t["flux"][ci]
            phi1 = vflux[vcells.index(n)]
            rsd1 = vrsd[vcells.index(n)]
            # a material change in this cell (Li-6 enrichment in the FLiBe, the first-wall
            # material) needs that design's own sidecar, as the damage tier does
            if ("li6_pct" in tag and mname == "flibe") or ("fw_material" in tag and n == "fw"):
                continue
            for (x, s) in KEY.get(mname, []):
                if x not in info["nuclides"] or s not in info["scores"]:
                    continue
                Rg = rr[:, info["nuclides"].index(x), info["scores"].index(s)]
                sg = np.where(phi0 > 0, Rg / np.where(phi0 > 0, phi0, 1), 0.0)       # macroscopic per unit flux
                exact = (sg * phi1).sum()
                sc = np.array([Rg[cm == c].sum() / phi0[cm == c].sum() if phi0[cm == c].sum() > 0 else 0 for c in range(Gc)])
                Phi1 = np.array([phi1[cm == c].sum() for c in range(Gc)])
                coarse = (sc * Phi1).sum()
                # MC uncertainty of the variant's exact rate (its own flux noise; group-independent)
                sd = np.sqrt(((sg * phi1 * rsd1) ** 2).sum()) / exact if exact > 0 else np.inf
                if exact > 0 and sd < MAX_SD:
                    rows.append({"variant": tag, "cell": n, "reaction": f"{x}{s}", "rel": float(coarse / exact - 1), "mc_rel_sd": float(sd)})
    by = {}
    for r in rows:
        by.setdefault(r["cell"], []).append(abs(r["rel"]))
    out.setdefault("by_cell", {})[cname] = {c: {"max": float(max(v)), "median": float(np.median(v))} for c, v in by.items()}
    print("   per cell max|rel| %:", {c: round(max(v) * 100, 2) for c, v in by.items()})
    rows = [r for r in rows if r["cell"] not in ("coil",)]   # coil: CE spectra are MC noise (see NOTES)
    rel = np.array([r["rel"] for r in rows])
    worst = sorted(rows, key=lambda r: -abs(r["rel"]))[:8]
    out["candidates"][cname] = {"Gc": int(Gc), "n": len(rows), "excluded": "coil", "max_abs_rel": float(np.abs(rel).max()),
                                "p95_abs_rel": float(np.percentile(np.abs(rel), 95)), "worst": worst}
    print(f"{cname:40s} Gc={Gc}  n={len(rows)}  max|rel|={np.abs(rel).max()*100:.2f}%  p95={np.percentile(np.abs(rel),95)*100:.2f}%")
    for w in worst[:4]:
        print("     ", w)
json.dump(out, open(os.path.join(ACT, "runs", "coarse_groups.json"), "w"), indent=1)

"""Nuclear-data UQ helpers: sampling plan per nuclide, correlated factor sampling,
and writing perturbed OpenMC HDF5 nuclide files.

Plan (per nuclide), built by prep.py and stored in /runs/cov/<nuc>/plan.npz + plan.json:
  params   list of (cov_mt, [HDF5 reaction MTs it scales])
             - a cov MT that is itself a non-redundant HDF5 reaction scales that reaction
             - a lumped cov MT (851-870) scales its constituents (ENDF MF33 header MTL)
             - a summation cov MT (4, 103-107) scales those components that have no
               covariance of their own (e.g. F-19 MT4 -> levels 51-71, 91)
             - MT1/MT3 (total / nonelastic) are not sampled: they are implied by partials
  cov      joint relative covariance over (param, group), incl. cross-MT blocks,
           relative sd capped at 100% (near-threshold groups blow up), PSD-projected
  kerma    per-reaction KERMA coefficient k_r(E) = K_r/sigma_r from NJOY HEATR (0 K,
           photons transported, i.e. the convention of OpenMC's MT301)
  gas      multiplicities c_r of each reaction in MT203-207 (fit to the library data)

Perturbation of a sample with group factors f[p, g]:
  sigma_r(E) <- f[p, g(E)] sigma_r(E)                (every temperature incl. 0 K elastic)
  redundant sums (4, 16, 103-107, ... kept in HDF5) <- old + sum of component deltas
  MT203-207 <- old + sum_r c_r delta_r
  MT301 (heating KERMA) <- old + sum_r k_r(E) delta_r
OpenMC rebuilds total/absorption/photon production from the non-redundant partials at load.
"""
import json, os
import numpy as np

UQ_ROOT = os.environ.get("UQ_ROOT", "/runs")
BASE_XS = "/work/blanket/xs"
LUMPED = range(851, 871)
SUMS = {4: list(range(50, 92)), 103: list(range(600, 650)), 104: list(range(650, 700)),
        105: list(range(700, 750)), 106: list(range(750, 800)), 107: list(range(800, 850))}


def base_h5(nuc):
    return os.path.join(BASE_XS, f"ENDFB-8.0-NNDC_{nuc}.h5")


def build_params(covmts, lump, nonred):
    params, covered = [], set()
    for m in covmts:
        if m in (1, 3) or m in SUMS and m not in nonred:
            continue
        if m in LUMPED:
            t = sorted(c for c, l in lump.items() if l == m and c in nonred)
        elif m in nonred and m not in lump:
            t = [m]
        else:
            t = []
        if t:
            params.append((m, t)); covered.update(t)
    for m in covmts:
        if m in SUMS and m not in nonred:
            t = sorted(c for c in SUMS[m] if c in nonred and c not in covered)
            if t:
                params.append((m, t)); covered.update(t)
    return params


def assemble_cov(params, cov, ng, sd_cap=1.0):
    n = len(params) * ng
    C = np.zeros((n, n))
    for i, (a, _) in enumerate(params):
        for j, (b, _) in enumerate(params):
            if (a, b) in cov:
                M = cov[(a, b)]
            elif (b, a) in cov:
                M = cov[(b, a)].T
            else:
                continue
            C[i * ng:(i + 1) * ng, j * ng:(j + 1) * ng] = M
    C = 0.5 * (C + C.T)
    sd = np.sqrt(np.clip(np.diag(C), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(np.outer(sd, sd) > 0, C / np.outer(sd, sd), 0.0)
    R = np.clip(R, -1, 1)
    np.fill_diagonal(R, (sd > 0).astype(float))
    sdc = np.minimum(sd, sd_cap)
    C = R * np.outer(sdc, sdc)
    return C, sd


def psd(C):
    w, V = np.linalg.eigh(C)
    neg = -w[w < 0].sum() / max(w[w > 0].sum(), 1e-30)
    w = np.clip(w, 0, None)
    return (V * w) @ V.T, float(neg)


def lognormal_sampler(C):
    """Multiplicative factors with mean 1 and relative covariance C (moment matched):
    ln f ~ N(-diag(S)/2, S), S = ln(1 + C)."""
    S = np.log1p(np.clip(C, -0.999, None))
    S, neg = psd(S)
    w, V = np.linalg.eigh(S)
    L = V * np.sqrt(np.clip(w, 0, None))
    mu = -0.5 * np.diag(S)
    def draw(rng):
        return np.exp(mu + L @ rng.standard_normal(len(mu)))
    return draw, neg


def load_plan(nuc):
    d = os.path.join(UQ_ROOT, "cov", nuc)
    meta = json.load(open(os.path.join(d, "plan.json")))
    z = np.load(os.path.join(d, "plan.npz"))
    kerma = {int(k[2:]): (z["kE"], z[k]) for k in z.files if k.startswith("k_")}
    return meta, z["bounds"], z["cov"], kerma


def write_perturbed(nuc, factors, out_path, plan=None):
    """factors: array [n_params, n_groups] (ones = unperturbed). Returns diagnostic dict."""
    import openmc.data
    meta, bounds, _, kerma = plan or load_plan(nuc)
    params = [(m, t) for m, t in meta["params"]]
    # gas production (MT203-207) is only rebuilt where the multiplicity fit is exact; the
    # Ni/W fits fail (energy-dependent MT5 yields) but no tally uses their gas production
    gas = {int(k): {int(r): c for r, c in v.items()} for k, v in meta["gas"].items()
           if meta["gas_fit_err"][k] < 1e-3}
    ng = len(bounds) - 1
    data = openmc.data.IncidentNeutron.from_hdf5(base_h5(nuc))
    rxs = data.reactions
    fac_of = {}
    for p, (m, targets) in enumerate(params):
        for r in targets:
            fac_of[r] = factors[p]
    temps = list(data.energy)  # includes '0K'
    diag = {}
    for T in temps:
        E = data.energy[T]
        g = np.clip(np.searchsorted(bounds, E, side="right") - 1, 0, ng - 1)
        delta = {}
        for r, f in fac_of.items():
            if r not in rxs or T not in rxs[r].xs:
                continue
            tab = rxs[r].xs[T]
            j = np.searchsorted(E, tab.x[0])
            old = np.asarray(tab.y, float)
            new = old * f[g[j:j + len(old)]]
            tab.y = new
            full = np.zeros(len(E)); full[j:j + len(old)] = new - old
            delta[r] = full
        if not delta or T == "0K":
            continue
        for mt, rx in rxs.items():
            if not rx.redundant or T not in rx.xs:
                continue
            tab = rx.xs[T]
            j = np.searchsorted(E, tab.x[0]); n = len(tab.y)
            if 203 <= mt <= 207 and mt not in gas:
                continue
            if mt in gas:
                d = sum(c * delta[r] for r, c in gas[mt].items() if r in delta)
            elif mt == 301:
                d = 0
                for r, dr in delta.items():
                    if r in kerma:
                        kE, k = kerma[r]
                        d = d + dr * np.interp(E, kE, k, left=0.0, right=k[-1])
                diag["kerma_rel_change_14MeV"] = float(np.interp(14.1e6, E, d) / np.interp(14.1e6, tab.x, tab.y)) if T == "294K" else diag.get("kerma_rel_change_14MeV")
            else:
                comps = data.get_reaction_components(mt)
                if comps == [mt]:
                    continue  # 444/901 etc: damage / local heating not used by the tallies
                d = sum(delta[c] for c in comps if c in delta)
            if isinstance(d, np.ndarray):
                tab.y = np.clip(np.asarray(tab.y, float) + d[j:j + n], 0, None)
    if os.path.exists(out_path):
        os.remove(out_path)
    data.export_to_hdf5(out_path, "w")
    return diag


def write_xs_xml(perturbed, out_xml):
    """Copy the base cross_sections.xml, pointing perturbed nuclides at new files."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(os.path.join(BASE_XS, "cross_sections.xml"))
    root = tree.getroot()
    for lib in root.findall("library"):
        mats = lib.get("materials")
        if mats in perturbed:
            lib.set("path", os.path.abspath(perturbed[mats]))
        else:
            lib.set("path", os.path.join(BASE_XS, lib.get("path")))
    for d in root.findall("directory"):
        root.remove(d)
    tree.write(out_xml)

"""How much does the energy structure of Sigma_t INSIDE a VITAMIN-J-175 group matter for
penetration through the coil pack, TiH2 shield and Inconel tank? From the pointwise ENDF/B-VIII.0
(294 K) macroscopic total of each material, per group (1/E weighting inside the group):
  S_nr   = 1 / <1/Sigma>                  narrow-resonance flux-weighted total (what a
                                          flux-weighted group constant tends to in a thick region)
  S_eff  = -ln <exp(-Sigma x)> / x        the constant that reproduces uncollided transmission
                                          over x cm of the material
and the uncollided-transmission ratio exp(-(S_nr - S_eff) x): < 1 means one group constant
attenuates more than the real, windowed cross section does.

Run: docker compose run --rm openmc blanket/coil/xs_windows.py  -> runs/coil_xs_windows.json
"""
import json, os, sys
import numpy as np
import openmc, openmc.data, openmc.mgxs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from arc_model import materials, DEFAULTS

lib = openmc.data.DataLibrary.from_xml(os.environ["OPENMC_CROSS_SECTIONS"])
EDGES = openmc.mgxs.EnergyGroups("VITAMIN-J-175").group_edges
_cache = {}


def total_xs(nuc):
    if nuc not in _cache:
        n = openmc.data.IncidentNeutron.from_hdf5(lib.get_by_material(nuc)["path"])
        E = n.energy["294K"]
        tot = np.zeros_like(E)
        for mt, rx in n.reactions.items():
            if not rx.redundant and mt != 1 and "294K" in rx.xs:
                tot += rx.xs["294K"](E)
        _cache[nuc] = (E, tot)
    return _cache[nuc]


def macro(mat, grid):
    s = np.zeros_like(grid)
    for nuc, dens in mat.get_nuclide_atom_densities().items():
        E, t = total_xs(nuc)
        s += dens * np.interp(grid, E, t)
    return s


def main():
    mats = materials(dict(DEFAULTS))
    out = {}
    for role, mat, xs_ in (("coil", mats["coil"], (5, 20, 60)), ("shield", mats["TiH2"], (5, 20, 50)), ("tank", mats["inconel"], (3,))):
        res = []
        for g in range(len(EDGES) - 1):
            lo, hi = EDGES[g], EDGES[g + 1]
            if lo < 1e3:
                continue
            u = np.linspace(np.log(lo), np.log(hi), 4000)
            S = macro(mat, np.exp(u))                        # uniform in lethargy = 1/E weighting
            snr = 1.0 / np.mean(1.0 / S)
            row = {"lo": lo, "hi": hi, "S_mean": float(S.mean()), "S_nr": float(snr), "S_min": float(S.min()), "S_max": float(S.max())}
            for x in xs_:
                seff = -np.log(np.mean(np.exp(-S * x))) / x
                row[f"S_eff_{x}"] = float(seff)
                row[f"T_ratio_{x}"] = float(np.exp(-(snr - seff) * x))
            res.append(row)
        out[role] = res
        print(role)
        for r in res:
            if r["lo"] >= 1e4:
                ks = [k for k in r if k.startswith("T_ratio")]
                print(f"  {r['lo']:.3e}-{r['hi']:.3e}  S_nr {r['S_nr']:.3f} range {r['S_min']:.3f}-{r['S_max']:.3f}  " + " ".join(f"{k[8:]}cm {r[k]:.2f}" for k in ks))
    json.dump(out, open(os.path.join(HERE, "..", "runs", "coil_xs_windows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

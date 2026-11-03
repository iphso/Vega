"""Browser (StoneLab scripts/coil/spectra.ts JSON) vs OpenMC (coil_ref.py result.npz) at the TF
coil: face fast fluence and magnet life, band-integrated spectra (ratio browser / OpenMC with
its 1-sigma) in the TF face, coil, shield and tank wall, and along the inboard profile.

  python3 compare.py <openmc run dir> <browser json> [more browser jsons ...] [--json out.json]
Host python (numpy only).
"""
import json, sys
import numpy as np

BANDS = [("<1 eV", 0, 1.0), ("1 eV-1 keV", 1.0, 1e3), ("1-100 keV", 1e3, 1e5), ("0.1-1 MeV", 1e5, 1e6), (">1 MeV", 1e6, 3e7)]


def load_ref(d):
    z = np.load(f"{d}/result.npz")
    m = json.load(open(f"{d}/result.json"))
    names = list(z["names"])
    fl, fs = z["flux"], z["flux_sd"]
    rows = {"coil_face": (fl[names.index("coil_face")], fs[names.index("coil_face")]),
            "coil": (fl[names.index("coil")] + fl[names.index("coil_face")], np.hypot(fs[names.index("coil")], fs[names.index("coil_face")])),
            "shield": (fl[names.index("shield")], fs[names.index("shield")]),
            "tank": (fl[names.index("tank")], fs[names.index("tank")])}
    for i in range(z["prof"].shape[0]):
        rows[f"prof{i}"] = (z["prof"][i], z["prof_sd"][i])
    return z["group_edges"], rows, m, z["prof_r"]


def load_browser(p):
    b = json.load(open(p))
    G = b["G"]
    sp, sd = np.array(b["spectra"]).reshape(-1, G), np.array(b["spectra_sd"]).reshape(-1, G)
    return {n: (sp[i], sd[i]) for i, n in enumerate(b["rows"])}, b


def band(edges, f, s, lo, hi):
    """Sum over groups inside [lo, hi) (group 0 highest); s.d. as if groups independent (the
    seeds' s.d. on the browser side is per group across seeds, so also a sum in quadrature)."""
    e_lo = edges[::-1][1:]
    k = (e_lo >= lo * (1 - 1e-9)) & (e_lo < hi * (1 - 1e-9))
    return f[k].sum(), np.sqrt((s[k] ** 2).sum())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if jout in args:
        args.remove(jout)
    edges, ref, meta, prof_r = load_ref(args[0])
    out = {"openmc": args[0], "openmc_face_fast_fluence_per_fpy": meta["face_fast_fluence_per_fpy"],
           "openmc_magnet_life_fpy": meta["magnet_life_fpy"], "openmc_face_fast_rel_sd": meta.get("face_fast_rel_sd", meta.get("face_fast_rel_sd_groupsum")),
           "browser": []}
    print(f"OpenMC {args[0]}: face fast fluence {meta['face_fast_fluence_per_fpy']:.3e}/FPY ± {100 * out['openmc_face_fast_rel_sd']:.1f}%, "
          f"magnet life {meta['magnet_life_fpy']:.2f} FPY, TBR {meta['TBR'][0]:.4f}")
    for bp in args[1:]:
        br, b = load_browser(bp)
        fl, fls = b["face_fast_fluence_per_fpy"]
        r = fl / meta["face_fast_fluence_per_fpy"]
        rs = r * np.hypot(fls / fl, out["openmc_face_fast_rel_sd"])
        print(f"\n== {bp}  ({b['histories'] / 1e6:.0f}M histories)\n  face fast fluence browser/OpenMC = {r:.3f} ± {rs:.3f}; "
              f"magnet life browser {b['magnet_life_fpy'][0]:.2f} ± {b['magnet_life_fpy'][1]:.2f} vs OpenMC {meta['magnet_life_fpy']:.2f} FPY")
        rec = {"file": bp, "pack": b["pack"], "fast_ratio": [r, rs], "magnet_life": b["magnet_life_fpy"], "bands": {}}
        print("  region      " + "".join(f"{n:>18s}" for n, _, _ in BANDS))
        regs = ["coil_face", "coil", "shield", "tank"] + [f"prof{i}" for i in range(0, 40, 3)]
        for reg in regs:
            row = []
            for n, lo, hi in BANDS:
                a, sa = band(edges, *br[reg], lo, hi)
                o, so = band(edges, *ref[reg], lo, hi)
                q = a / o if o > 0 else np.nan
                qs = q * np.hypot(sa / a if a > 0 else 0, so / o if o > 0 else 0)
                row.append((q, qs))
            lab = reg if not reg.startswith("prof") else f"R {prof_r[int(reg[4:])]:.0f}-{prof_r[int(reg[4:]) + 1]:.0f}"
            print(f"  {lab:12s}" + "".join(f"{q:9.3f} ±{qs:6.3f} " if np.isfinite(q) else f"{'-':>18s}" for q, qs in row))
            rec["bands"][reg] = row
        out["browser"].append(rec)
    if jout:
        json.dump(out, open(jout, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

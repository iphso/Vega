"""Weight windows must not bias: an analog coil_ref.py run against weight-window runs, in every
region / group band where the analog run itself resolves the answer (rel. s.d. < 10%): cell
spectra (fw, channel, blanket, tank wall, shield), the inboard profile bins, heating-local, TBR.
Reports z = (VR - analog) / combined s.d. per comparison and their spread.

  python3 vr_check.py <analog dir> <vr dir> [<vr dir> ...] [--fast-only]   (host python, numpy)
--fast-only: compare only > 0.1 MeV bands (runs whose windows roulette slow neutrons).
"""
import json, sys
import numpy as np

BANDS = [("<1 eV", 0, 1.0), ("1eV-1keV", 1.0, 1e3), ("1-100keV", 1e3, 1e5), ("0.1-1MeV", 1e5, 1e6), (">1MeV", 1e6, 3e7)]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
fast_only = "--fast-only" in sys.argv


def load(d):
    z = np.load(f"{d}/result.npz"); m = json.load(open(f"{d}/result.json"))
    names = list(z["names"])
    rows = {n: (z["flux"][i], z["flux_sd"][i]) for i, n in enumerate(names) if n in ("fw", "channel", "blanket", "tank", "shield")}
    for i in range(z["prof"].shape[0]):
        rows[f"prof{i}"] = (z["prof"][i], z["prof_sd"][i])
    return z["group_edges"][::-1][1:], rows, m


lo, A, am = load(args[0])
out = {}
for d in args[1:]:
    _, B, bm = load(d)
    zs, worst = [], []
    for reg in A:
        for n, a, b in BANDS:
            if fast_only and a < 1e5:
                continue
            k = (lo >= a * (1 - 1e-9)) & (lo < b * (1 - 1e-9))
            fa, sa = A[reg][0][k].sum(), np.sqrt((A[reg][1][k] ** 2).sum())
            fb, sb = B[reg][0][k].sum(), np.sqrt((B[reg][1][k] ** 2).sum())
            if fa <= 0 or sa / fa > 0.10 or fb <= 0:
                continue
            z = (fb - fa) / np.hypot(sa, sb)
            zs.append(z)
            worst.append((abs(z), reg, n, fb / fa))
    zs = np.array(zs)
    worst.sort(reverse=True)
    tz = (bm["TBR"][0] - am["TBR"][0]) / np.hypot(bm["TBR"][1], am["TBR"][1])
    hz = {r: (bm["heating_local_W"][r][0] - am["heating_local_W"][r][0]) / np.hypot(bm["heating_local_W"][r][1], am["heating_local_W"][r][1])
          for r in ("fw", "channel", "blanket", "tank", "shield") if am["heating_local_W"][r][1] < 0.1 * am["heating_local_W"][r][0]}
    print(f"{d} vs {args[0]}: {len(zs)} band comparisons, mean z {zs.mean():+.2f}, rms z {np.sqrt((zs ** 2).mean()):.2f}, "
          f"|z|>3: {(abs(zs) > 3).sum()} (expect {len(zs) * 0.0027:.1f}); TBR z {tz:+.2f}; heating z {({k: round(v, 2) for k, v in hz.items()})}")
    print("   worst:", [(f"{r}:{n}", round(z, 2), round(q, 3)) for z, r, n, q in worst[:4]])
    out[d] = {"n": len(zs), "mean_z": float(zs.mean()), "rms_z": float(np.sqrt((zs ** 2).mean())), "tbr_z": float(tz),
              "heating_z": hz, "worst": [(r, n, float(z), float(q)) for z, r, n, q in worst[:6]]}
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)

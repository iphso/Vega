"""Parse the OKTAVIAN tables of the JAERI-M 94-014 / FENDL-2 compilation into data.json.

Sources (blanket/refs/lit/jaerim94014/benchmarks/jaerim94014/oktavian/):
  beli/readme.txt   Tables 2.1-2.7 (D-T source, Be and Be-Li leakage spectra; n/source/MeV)
  n-leak/report.txt Table 4.2 (D-T source for LiF, Mn, Cu, Mo, W; per unit lethargy)
  n-leak/lif60.dat, w40.dat (leakage per unit lethargy per source neutron)
  tbr/readme.txt    Tables 1.1-1.16 (transcribed by hand below: the tables are irregular)

OCR/typing repairs (every one is listed in data.json["repairs"]):
  * letter O / l in numbers ("2.044OE-02", "l.OO90E-03"), "~" for "+", "1 2160E-02"
  * beli Table 2.1 "9.640e+00" between 1.058 and 0.878 MeV -> 9.640e-01
  * beli Table 2.2 "2.250e+00" after "2.251e+00": second entry read as 2.058e+00 (the
    sibling Table 2.3 has 2.259/2.058 at the same place)
  * beli Table 2.4 15.46-15.08 MeV: upper energies are non-monotonic (1.522, 1.524, 1.522...);
    the 7 bins between 15.46 and 15.08 MeV are re-spaced uniformly (flagged; the >10 MeV band
    of that case therefore carries an extra width uncertainty)
  * beli Table 2.5 last row duplicates Table 2.2's last row (7.014 at 2.856 keV); kept but the
    coarse bands do not go below 10 keV.
"""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../refs/lit/jaerim94014/benchmarks/jaerim94014/oktavian")
repairs = []


def num(s):
    t = s.replace("O", "0").replace("o", "0").replace("l", "1").replace("~", "+").replace(" ", "." if re.match(r'^\d \d', s) else "")
    if t.lower().endswith("e") or t == "":
        raise ValueError(s)
    if t != s:
        repairs.append(f"{s!r} -> {t!r}")
    return float(t)


def beli_tables():
    txt = open(os.path.join(SRC, "beli/readme.txt"), encoding="latin-1").read().replace("\r", "")
    parts = re.split(r"\nTable (2\.\d)\s", txt)
    out = {}
    for k in range(1, len(parts), 2):
        name, body = parts[k], parts[k + 1]
        if name in out:   # the 'List of Tables' block precedes the data
            pass
        rows = []
        for line in body.split("\n"):
            f = line.split()
            if not f or not re.match(r"^\d", f[0]):
                continue
            if len(f) == 1:
                rows.append((num(f[0]), None, None)); break
            if len(f) >= 3 and re.match(r"^[\d.]+e[+-]\d+$", f[0]):
                rows.append((num(f[0]), num(f[1]), num(f[2])))
        if len(rows) > 10:
            out[name] = rows
    return out


def to_bins(rows):
    """rows: (upper energy MeV, value, err) ... last row (lower edge, None)."""
    up = [r[0] for r in rows]
    lo = up[1:]
    return [dict(hi=up[i], lo=lo[i], v=rows[i][1], e=rows[i][2]) for i in range(len(rows) - 1)]


def nleak_source():
    txt = open(os.path.join(SRC, "n-leak/report.txt"), encoding="latin-1").read().replace("\r", "")
    seg = txt.split("Table 4.2 Neutron source spectrum for LiF, Mn, Cu, Mo and W.")[1].split("Table 4.3")[0]
    bins = []
    for line in seg.split("\n"):
        f = line.split("\t") if "\t" in line else line.split()
        f = [x.strip() for x in f if x.strip()]
        if len(f) < 4 or not re.match(r"^\d", f[0]):
            continue
        bins.append(dict(lo=num(f[0]), hi=num(f[1]), v=num(f[2]), e=num(f[3])))
    return bins


def nleak_spec(fn):
    bins = []
    for line in open(os.path.join(SRC, "n-leak", fn), encoding="latin-1").read().replace("\r", "").split("\n"):
        f = line.split()
        if len(f) == 4 and re.match(r"^[\d.]+$", f[0]):
            bins.append(dict(hi=float(f[0]), lo=float(f[1]), v=float(f[2]), e=float(f[3])))
    return bins


def main():
    bt = beli_tables()
    # --- documented repairs
    t21 = [list(r) for r in bt["2.1"]]
    for r in t21:
        if r[0] == 9.640:
            r[0] = 0.9640; repairs.append("beli T2.1 upper energy 9.640e+00 -> 9.640e-01")
    t22 = [list(r) for r in bt["2.2"]]
    for k in range(1, len(t22)):
        if t22[k][0] == 2.250 and t22[k - 1][0] == 2.251:
            t22[k][0] = 2.058; repairs.append("beli T2.2 upper energy 2.250e+00 (after 2.251) -> 2.058e+00")
    t24 = [list(r) for r in bt["2.4"]]
    i0 = [r[0] for r in t24].index(15.46); i1 = [r[0] for r in t24].index(15.08)
    for j, e in enumerate(np.linspace(15.46, 15.08, i1 - i0 + 1)):
        t24[i0 + j][0] = round(float(e), 4)
    repairs.append(f"beli T2.4 upper energies of rows {i0}..{i1} (15.46..15.08 MeV, non-monotonic) re-spaced uniformly")
    bt["2.1"], bt["2.2"], bt["2.4"] = t21, t22, t24
    beli = {k: to_bins(v) for k, v in bt.items()}
    for k, b in beli.items():
        bad = [x for x in b if not x["hi"] > x["lo"]]
        assert not bad, (k, bad[:3])
    data = dict(
        beli=dict(source=beli["2.1"], be57=beli["2.2"], be69=beli["2.3"], be97=beli["2.4"], be128=beli["2.5"],
                  beli57=beli["2.6"], beli128=beli["2.7"],
                  units="n per source neutron per MeV (bins given by upper energy; last lower edge from table end)",
                  norm_unc=[0.067, 0.084]),
        nleak=dict(source=nleak_source(), lif60=nleak_spec("lif60.dat"), w40=nleak_spec("w40.dat"),
                   units="leakage current per source neutron per unit lethargy"),
        repairs=repairs)
    # sanity: integrals
    for k in ["source", "be57", "be128", "beli57"]:
        b = data["beli"][k]
        data["beli"].setdefault("integral", {})[k] = sum(x["v"] * (x["hi"] - x["lo"]) for x in b)
    for k in ["source", "lif60", "w40"]:
        b = data["nleak"][k]
        data["nleak"].setdefault("integral", {})[k] = sum(x["v"] * np.log(x["hi"] / x["lo"]) for x in b)
    json.dump(data, open(os.path.join(HERE, "data.json"), "w"), indent=0)
    print(json.dumps({"beli": data["beli"]["integral"], "nleak": data["nleak"]["integral"]}, indent=1))
    print("\n".join(repairs))


if __name__ == "__main__":
    main()

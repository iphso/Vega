#!/usr/bin/env python3
"""Post-process the watercap spray validation run.

Reads postProcessing/*/0/*.dat time series and log.reactingFoam (cloud
statistics) and prints time-averaged outlet results over [t0, t1].

Usage: python3 postprocess.py [t0] [t1]   (defaults: averageStart from
controlDict .. last time in the log)
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_dat(name):
    files = sorted(glob.glob(os.path.join(HERE, "postProcessing", name, "*", "*.dat")),
                   key=lambda f: float(os.path.basename(os.path.dirname(f))))
    rows = []
    for f in files:
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            line = line.replace("(", " ").replace(")", " ")
            rows.append([float(x) for x in line.split()])
    a = np.array(rows)
    # drop duplicate times from restarts (keep last)
    _, idx = np.unique(a[::-1, 0], return_index=True)
    return a[::-1][idx]


def tavg(a, col, t0, t1):
    m = (a[:, 0] >= t0) & (a[:, 0] <= t1)
    t, y = a[m, 0], a[m, col]
    return np.trapezoid(y, t) / (t[-1] - t[0])


def parse_log():
    log = open(os.path.join(HERE, "log.reactingFoam")).read()
    out = []
    for blk in re.split(r"\nTime = ", log)[1:]:
        t = float(blk.split("s", 1)[0])
        d = {"t": t}
        m = re.search(r"mass introduced\s+=\s+(\S+)", blk)
        if not m:
            continue
        d["inj"] = float(m.group(1))
        d["sys"] = float(re.search(r"Current mass in system\s+=\s+(\S+)", blk).group(1))
        d["npar"] = int(re.search(r"Current number of parcels\s+=\s+(\S+)", blk).group(1))
        d["pc"] = float(re.search(r"Mass transfer phase change\s+=\s+(\S+)", blk).group(1))
        for p in ("walls", "inlet", "outlet"):
            m = re.search(r"patch %s\n\s+- escape\s+=\s+\S+, (\S+)" % p, blk)
            d[p] = float(m.group(1))
        out.append(d)
    return out


def parcel_stats(t0, t1):
    """Mass-weighted stats of parcels that left through each patch in [t0, t1]
    (from the patchPostProcessing cloud function)."""
    base = os.path.join(HERE, "postProcessing", "lagrangian", "cloud", "patchPostProcessing1")
    res = {}
    for patch in ("outlet", "inlet", "walls"):
        rows = []
        for f in glob.glob(os.path.join(base, "*", patch + ".post")):
            for line in open(f):
                if line.startswith("#"):
                    continue
                tok = line.replace("(", " ").replace(")", " ").split()
                if len(tok) < 32:
                    continue
                t = float(tok[0]); nP = float(tok[17]); d = float(tok[18])
                rho = float(tok[23]); T = float(tok[29]); age = float(tok[24])
                rows.append((t, nP * rho * np.pi / 6 * d**3, d, T, age, float(tok[22])))
        a = np.array(rows) if rows else np.zeros((0, 6))
        a = a[(a[:, 0] >= t0) & (a[:, 0] <= t1)] if len(a) else a
        if len(a) == 0:
            res[patch] = None
            continue
        w = a[:, 1]
        res[patch] = dict(n=len(a), mdot=w.sum() / (t1 - t0),
                          T=(w * a[:, 3]).sum() / w.sum(),
                          d43=(w * a[:, 2]).sum() / w.sum(),
                          age=(w * a[:, 4]).sum() / w.sum(),
                          Uz=(w * a[:, 5]).sum() / w.sum())
    return res


def main():
    cd = open(os.path.join(HERE, "system", "controlDict")).read()
    t0 = float(re.search(r"averageStart\s+(\S+);", cd).group(1))
    cl = parse_log()
    t1 = cl[-1]["t"] if cl else read_dat("outletFlux")[-1, 0]
    if len(sys.argv) > 1:
        t0 = float(sys.argv[1])
    if len(sys.argv) > 2:
        t1 = float(sys.argv[2])
    print(f"Averaging window: {t0:.3f} .. {t1:.3f} s")

    inF = read_dat("inletFlux"); outF = read_dat("outletFlux")
    inV = read_dat("inletVapour"); outV = read_dat("outletVapour")
    inT = read_dat("inletT"); outT = read_dat("outletT")
    inP = read_dat("inletP"); outP = read_dat("outletP")
    minU = read_dat("minU"); maxU = read_dat("maxU")

    mIn = -tavg(inF, 1, t0, t1); mOut = tavg(outF, 1, t0, t1)
    vIn = -tavg(inV, 1, t0, t1); vOut = tavg(outV, 1, t0, t1)
    TIn = tavg(inT, 1, t0, t1); TOut = tavg(outT, 1, t0, t1)
    pIn = tavg(inP, 1, t0, t1); pOut = tavg(outP, 1, t0, t1)
    print(f"gas mass flow   in {mIn:.6f}  out {mOut:.6f} kg/s")
    print(f"vapour flow     in {vIn:.6e}  out {vOut:.6e} kg/s")
    print(f"vapor_removed_frac (outlet vs inlet) = {(vIn - vOut) / vIn:.5f}")
    dryIn = mIn - vIn; dryOut = mOut - vOut
    print(f"dry-air flow    in {dryIn:.6f}  out {dryOut:.6f} kg/s")
    print(f"humidity ratio  in {vIn/dryIn:.5f}  out {vOut/dryOut:.5f} kg/kg")
    print(f"mixing-cup T    in {TIn - 273.15:.3f} C  out {TOut - 273.15:.3f} C")
    rho = mIn / 0.15  # inlet gas density from volumetric flow 0.15 m3/s
    print(f"p area-avg      in {pIn:.2f}  out {pOut:.2f} Pa; dp = {pIn - pOut:.2f} Pa "
          f"(hydrostatic rho*g*L ~ {rho*9.81*2.5:.1f} Pa)")
    m = (minU[:, 0] >= t0) & (minU[:, 0] <= t1)
    print(f"min Uz (instantaneous) over window: mean {minU[m, 3].mean():.3f}, "
          f"lowest {minU[m, 3].min():.3f} m/s; max Uz mean {maxU[m, 3].mean():.3f} m/s")

    if cl:
        t = np.array([d["t"] for d in cl])
        i0 = np.searchsorted(t, t0); i1 = np.searchsorted(t, t1, side="right") - 1
        a, b = cl[i0], cl[i1]
        dt = b["t"] - a["t"]
        dinj = b["inj"] - a["inj"]
        print(f"spray injected in window: {dinj/dt:.5f} kg/s")
        pcr = (b["pc"] - a["pc"]) / dt
        print(f"phase-change rate (cloud, +evap/-cond): {pcr:.4e} kg/s "
              f"-> vapour removed by condensation = {-pcr / vIn:.5f} of inlet vapour")
        for p in ("outlet", "inlet", "walls"):
            r = (b[p] - a[p]) / dt
            print(f"spray leaving via {p:6s}: {r:.5f} kg/s = {r / (dinj/dt):.4f} of injected")
        print(f"liquid mass in domain: {a['sys']:.5f} -> {b['sys']:.5f} kg; parcels {b['npar']}")
        print(f"cumulative to t={b['t']:.3f}: injected {b['inj']:.4f}, phase change {b['pc']:.4e}, "
              f"outlet {b['outlet']:.4f}, inlet {b['inlet']:.4f}, walls {b['walls']:.4f}, "
              f"in system {b['sys']:.4f} kg")
        ps = parcel_stats(a["t"], b["t"])
        for p, r in ps.items():
            if r is None:
                print(f"parcels via {p:6s}: none")
            else:
                print(f"parcels via {p:6s}: {r['n']} parcels, {r['mdot']:.5f} kg/s, mass-mean T "
                      f"{r['T'] - 273.15:.2f} C, mass-mean d {r['d43']*1e6:.1f} um, "
                      f"mean age {r['age']:.2f} s, mean Uz {r['Uz']:.2f} m/s")


if __name__ == "__main__":
    main()

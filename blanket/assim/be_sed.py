"""Be-9 (n,2n) secondary-energy hardness perturbation: the term the covariance sampling (blanket/uq)
could not reach (ENDF/B-VIII.0 has no MF35 / energy-distribution covariances for Be-9).

MT16's outgoing neutrons are a correlated energy-angle table: for each incident energy E a pdf
p(E'|E) and, per outgoing energy, an angular distribution. The perturbation reweights the energy
pdf with an exponential tilt and leaves every angle table alone:

    p'(E'|E) = p(E'|E) exp(a (E' - <E'>_E) / <E'>_E) / Z

with `a` solved per incident energy so the mean outgoing energy changes by exactly `shift`
(e.g. +0.10 = 10% harder). The (n,2n) cross section and multiplicity are untouched, so the
neutron balance is unchanged and only the energy of the two emitted neutrons moves. MT301 (heating)
is NOT made consistent: use this for tritium / flux responses, not heating.

Run (in the openmc container):
  blanket/assim/be_sed.py --shift 0.10 --out /work/blanket/runs/be_sed/p10
writes Be9.h5 + a cross_sections.xml (base library with Be-9 replaced).
"""
import argparse, os, shutil
import numpy as np
import openmc.data
import xml.etree.ElementTree as ET

BASE_XS = "/work/blanket/xs"


def tilt(eo, shift):
    """Reweight one Tabular outgoing-energy pdf so its mean moves by `shift` (relative)."""
    x, p = np.asarray(eo.x, float), np.asarray(eo.p, float)
    hist = eo.interpolation == "histogram"
    if len(x) < 3 or p.sum() <= 0:
        return eo, 0.0
    w = np.diff(x)

    def moments(pp):
        if hist:
            m0 = (pp[:-1] * w).sum(); m1 = (pp[:-1] * w * 0.5 * (x[:-1] + x[1:])).sum()
        else:
            m0 = (0.5 * (pp[:-1] + pp[1:]) * w).sum()
            m1 = (w * (pp[:-1] * (2 * x[:-1] + x[1:]) + pp[1:] * (x[:-1] + 2 * x[1:])) / 6).sum()
        return m0, m1 / m0

    m0, mean = moments(p)
    target = mean * (1 + shift)
    lo, hi = -50.0, 50.0
    for _ in range(100):
        a = 0.5 * (lo + hi)
        q = p * np.exp(a * (x - mean) / mean)
        if moments(q)[1] < target:
            lo = a
        else:
            hi = a
    q = p * np.exp(a * (x - mean) / mean)
    n0, _ = moments(q)
    q = q / n0
    if hist:
        c = np.concatenate([[0.0], np.cumsum(q[:-1] * w)])
    else:
        c = np.concatenate([[0.0], np.cumsum(0.5 * (q[:-1] + q[1:]) * w)])
    out = openmc.stats.Tabular(x, q, eo.interpolation)
    try:
        out.c = c
    except AttributeError:
        out._c = c
    return out, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nuc = openmc.data.IncidentNeutron.from_hdf5(os.path.join(BASE_XS, "ENDFB-8.0-NNDC_Be9.h5"))
    prod = [p for p in nuc.reactions[16].products if p.particle == "neutron"][0]
    dist = prod.distribution[0]
    means = []
    for i, eo in enumerate(dist.energy_out):
        if isinstance(eo, openmc.stats.Tabular):
            new, alpha = tilt(eo, a.shift)
            means.append((dist.energy[i], alpha))
            dist.energy_out[i] = new
    path = os.path.join(a.out, "Be9.h5")
    if os.path.exists(path):
        os.remove(path)
    nuc.export_to_hdf5(path)
    tree = ET.parse(os.path.join(BASE_XS, "cross_sections.xml"))
    root = tree.getroot()
    for lib in root.findall("library"):
        if lib.get("materials") == "Be9":
            lib.set("path", os.path.abspath(path))
        elif not os.path.isabs(lib.get("path")):
            lib.set("path", os.path.join(BASE_XS, lib.get("path")))
    tree.write(os.path.join(a.out, "cross_sections.xml"))
    # read back: mean outgoing energy at ~14 MeV before/after
    chk = openmc.data.IncidentNeutron.from_hdf5(path).reactions[16].products[0].distribution[0]
    base = openmc.data.IncidentNeutron.from_hdf5(os.path.join(BASE_XS, "ENDFB-8.0-NNDC_Be9.h5")).reactions[16].products[0].distribution[0]
    k = int(np.argmin(np.abs(np.asarray(base.energy) - 14.1e6)))
    mean = lambda t: float((np.asarray(t.p)[:-1] * np.diff(t.x) * 0.5 * (np.asarray(t.x)[:-1] + np.asarray(t.x)[1:])).sum() / (np.asarray(t.p)[:-1] * np.diff(t.x)).sum())
    print(f"mean E' at {base.energy[k]/1e6:.2f} MeV: {mean(base.energy_out[k])/1e6:.3f} -> {mean(chk.energy_out[k])/1e6:.3f} MeV, cdf end {np.asarray(chk.energy_out[k].c)[-1]:.6f}")
    E = [e for e, _ in means]
    print(f"Be9 MT16: {len(means)} incident energies tilted ({min(E)/1e6:.2f}-{max(E)/1e6:.2f} MeV), "
          f"alpha range {min(x for _, x in means):+.3f}..{max(x for _, x in means):+.3f}; wrote {a.out}")


if __name__ == "__main__":
    main()

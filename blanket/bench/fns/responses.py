"""Benchmark response definitions: experimental values with uncertainty split, and how each is
computed from a run_bench result (fnslib.extract output).

Each response: id, quantity, rid (detector reaction), mesh, position_cm (distance from target),
E, E_rel_unc (total, fraction), sys (fraction, fully correlated within corr_group), corr_group,
and optionally 'integral': a trapezoid over a list of positions (same rule applied to C and E).
"""
import math
import numpy as np
import os
import fnsdata as D
import fnslib


def _split(tot_pct, sys_pct):
    sys_pct = min(sys_pct, tot_pct)
    return tot_pct / 100, sys_pct / 100, math.sqrt(max(tot_pct**2 - sys_pct**2, 0)) / 100


def li2o_responses(room_model):
    """room_model=True: compare Li-pellet T6 with the column corrected for self-shielding only;
    False: with the column corrected for self-shielding AND room return (Table 7.2)."""
    R = []
    col = 3 if room_model else 4
    for row in D.LI2O_LSC:
        p, err = row[0], row[2]
        if row[col] is None:
            continue
        tot, s, r = _split(err, D.LI2O_LSC_SYS)
        R.append(dict(id=f"t6_lsc_{p:.1f}", quantity="6Li(n,t) TPR, Li2O pellet + LSC [T/6Li atom/source n]",
                      rid="li6_t", mesh="axis", position_cm=p, E=row[col], tot=tot, sys=s, rnd=r,
                      corr_group="li2o_lsc_sys", note="Table 7.2, " + ("self-shielding corrected" if room_model
                                                                      else "self-shielding + room-return corrected")))
    for row in D.LI2O_LSC:
        p = row[0]
        if row[5] is None:
            continue
        tot, s, r = _split(row[6], D.LI2O_LSC_SYS)
        R.append(dict(id=f"t7_lsc_{p:.1f}", quantity="7Li(n,n't) TPR, Li2O pellet + LSC [T/7Li atom/source n]",
                      rid="li7_t", mesh="axis", position_cm=p, E=row[5], tot=tot, sys=s, rnd=r,
                      corr_group="li2o_lsc_sys", note="Table 7.2 (no room correction needed: threshold)"))
    for z, rr, v, err in D.LI2O_LG:
        tot, s, r = _split(err, D.LI2O_LG_SYS)
        R.append(dict(id=f"t6_lg_{'ax' if rr == 0 else 'r15'}_{z:.2f}", quantity="6Li(n,t) TPR, Li-glass pair [T/6Li atom/source n]",
                      rid="li6_t", mesh="axis" if rr == 0 else "ring15", position_cm=z, E=v, tot=tot, sys=s, rnd=r,
                      corr_group="li2o_lg_sys", note=f"Table 7.4, r = {rr} cm, not room corrected"))
    for p, v, unf, src in D.LI2O_NE213:
        alpha = math.sqrt(max(src**2 - 2.34**2, 0))
        rnd = math.sqrt(unf**2 + alpha**2)
        sys_ = D.LI2O_NE213_SYS
        tot = math.sqrt(rnd**2 + sys_**2)
        R.append(dict(id=f"t7_ne213_{p:.1f}", quantity="7Li(n,n't) TPR from NE213 spectrum [T/7Li atom/source n]",
                      rid="li7_t", mesh="axis", position_cm=p, E=v, tot=tot / 100, sys=sys_ / 100, rnd=rnd / 100,
                      corr_group="li2o_ne213_sys", note="Table 7.6; folded with the experimenters' 7Li cross section"))
    for i, (rid, nm) in enumerate([("u235_f", "U-235"), ("u238_f", "U-238"), ("np237_f", "Np-237"), ("th232_f", "Th-232")]):
        errs = [row[1 + i][1] for row in D.LI2O_FC if row[1 + i]]
        smin = min(errs)
        for row in D.LI2O_FC:
            if not row[1 + i]:
                continue
            v, err = row[1 + i]
            tot, s, r = _split(err, smin * 0.9)
            R.append(dict(id=f"fc_{rid}_{row[0]:.1f}", quantity=f"{nm} fission rate, micro fission chamber [f/atom/source n]",
                          rid=rid, mesh="axis", position_cm=row[0], E=v, tot=tot, sys=s, rnd=r,
                          corr_group=f"li2o_fc_{rid}_sys", note="Table 7.7; split assumed: 90% of the smallest error common"))
    for rid, tab, nm in [("al27_na", D.LI2O_AL, "27Al(n,a)24Na"), ("ni58_n2n", D.LI2O_NI2N, "58Ni(n,2n)57Ni")]:
        for p, v, err in tab:
            tot, s, r = _split(err, D.FOIL_SYS)
            R.append(dict(id=f"foil_{rid}_{p:.2f}", quantity=f"{nm} foil rate [r/atom/source n]", rid=rid, mesh="axis",
                          position_cm=p, E=v, tot=tot, sys=s, rnd=r, corr_group="li2o_foil_sys",
                          note="Table 7.9; quoted error appears to exclude the 3% yield error"))
    return R


def be_responses():
    R = []
    for p, v, err in D.BE_LG:
        tot, s, r = _split(err, D.BE_LG_SYS)
        R.append(dict(id=f"t6_lg_{p/10:.2f}", quantity="6Li(n,t) TPR, Li-glass [T/6Li atom/source n]", rid="li6_t",
                      mesh="axis", position_cm=p / 10, E=v, tot=tot, sys=s, rnd=r, corr_group="be_lg_sys",
                      note="t8-4; split assumed from Li2O Table 7.5"))
    for p, v, err in D.BE_FC:
        tot, s, r = _split(err, D.BE_FC_SYS)
        R.append(dict(id=f"fc_u235_f_{p/10:.2f}", quantity="U-235 fission rate, micro fission chamber [f/atom/source n]",
                      rid="u235_f", mesh="axis", position_cm=p / 10, E=v, tot=tot, sys=s, rnd=r, corr_group="be_fc_sys",
                      note="t8-5; split assumed"))
    names = {"au197_ng": "197Au(n,g)198Au", "al27_na": "27Al(n,a)24Na", "ni58_n2n": "58Ni(n,2n)57Ni",
             "fe56_np": "56Fe(n,p)56Mn", "zr90_n2n": "90Zr(n,2n)89Zr", "ni58_np": "58Ni(n,p)58Co"}
    for rid, vals in D.BE_FOILS.items():
        for d, (v, err) in zip(D.BE_FOIL_DEPTHS, vals):
            tot, s, r = _split(err, D.FOIL_SYS)
            R.append(dict(id=f"foil_{rid}_{20 + d:.2f}", quantity=f"{names[rid]} foil rate [r/atom/source n]", rid=rid,
                          mesh="axis", position_cm=20.0 + d, E=v, tot=tot, sys=s, rnd=r, corr_group="be_foil_sys",
                          note="t8-6" + ("; Au foil thickness unknown, no self-shielding correction modelled" if rid == "au197_ng" else "")))
    return R


# ------------------------------------------------------------------ NE213 spectra: flux above 10 MeV
LI2O_SPEC_POS = [21.6, 31.5, 41.6, 51.7, 61.8, 72.0, 82.1]          # Tables 7.14-7.20
BE_SPEC = [("t8-7", 21.4), ("t8-8", 26.5), ("t8-9", 31.6), ("t8-10", 32.7), ("t8-11", 41.7),
           ("t8-12", 47.9), ("t8-13", 51.9), ("t8-14", 62.0)]       # Tables 8.7-8.14, mm -> cm
BAND = (10e6, 20e6)
# The compilation gives per-bin unfolding errors only (strongly anticorrelated between bins); the error
# of the band integral is not given. Assumed: 3 % random (unfolding) + 2.34 % alpha-monitor calibration
# and 1 % efficiency common (Tables 7.11/7.12 analogue).
SPEC_RND, SPEC_SYS = 3.0, math.sqrt(2.34**2 + 1.0**2)


def _band(lines):
    E, p = [], []
    for l in lines:
        if l.startswith("Table"):
            break
        t = l.split()
        if len(t) >= 3:
            try:
                E.append(float(t[0])); p.append(float(t[1]))
            except ValueError:
                pass
    E, p = np.array(E), np.array(p)
    du = np.gradient(np.log(E))
    m = E >= BAND[0]
    return float(np.sum(p[m] * du[m]))


def spectrum_measured(case):
    out = []
    if case == "li2o":
        L = open(os.path.join(fnslib.REFDIR, "li2o", "read.me"), "rb").read().decode("latin1").splitlines()
        heads = [k for k, l in enumerate(L) if l.startswith("Table 7.") and "Neutron spectrum in" in l][:7]
        for k, z in zip(heads, LI2O_SPEC_POS):
            out.append((z, _band(L[k + 1:k + 100])))
    else:
        for f, z in BE_SPEC:
            out.append((z, _band(open(os.path.join(fnslib.REFDIR, "be", f + ".txt")).read().splitlines()[1:])))
    return out


def spectrum_responses(case, length):
    R = []
    for z, v in spectrum_measured(case):
        if z > 20.0 + length:
            continue   # Li2O 82.1 cm: behind the back face, inside a 2 cm tally bin that straddles it
        tot = math.hypot(SPEC_RND, SPEC_SYS)
        R.append(dict(id=f"phi_gt10_ne213_{z:.1f}", quantity="neutron flux above 10 MeV, NE213 unfolded [n/cm2/source n]",
                      rid="spectrum", mesh="axis2", position_cm=z, E=v, tot=tot / 100, sys=SPEC_SYS / 100,
                      rnd=SPEC_RND / 100, corr_group=f"{case}_ne213_sys", band=list(BAND),
                      note="integral of the unfolded flux per lethargy over E >= 10 MeV; integral error assumed"))
    return R


def _spec_at(res, z):
    sp = res["spectrum"]
    eb = np.array(sp["ebins"]); zg = np.array(sp["zgrid"]); vol = np.array(sp["vol"])
    M = np.array(sp["mean"]) / vol[:, None]; S = np.array(sp["std"]) / vol[:, None]
    lo, hi = np.log(np.maximum(eb[:-1], BAND[0])), np.log(np.minimum(eb[1:], BAND[1]))
    w = np.clip((hi - lo) / np.log(eb[1:] / eb[:-1]), 0, 1)
    band, bsd = M @ w, np.sqrt((S**2) @ (w**2))
    zc = 0.5 * (zg[1:] + zg[:-1])
    z0, z1 = 20.0, 20.0 + res["info"]["length"]
    inside = (zg[:-1] >= z0 - 1e-9) & (zg[1:] <= z1 + 1e-9)
    idx = np.nonzero(inside)[0]
    zz = zc[idx]
    j = int(np.clip(np.searchsorted(zz, z) - 1, 0, len(zz) - 2))
    a, b = idx[j], idx[j + 1]
    t = (z - zc[a]) / (zc[b] - zc[a])
    val = math.exp((1 - t) * math.log(band[a]) + t * math.log(band[b]))   # log-linear (exponential attenuation)
    sd = val * math.hypot((1 - t) * bsd[a] / band[a], t * bsd[b] / band[b])
    return val, sd


def integral_responses(R, specs):
    """specs: list of (new_id, quantity, prefix filter function). Trapezoid over the measured positions."""
    out = []
    for nid, q, sel in specs:
        pts = sorted([r for r in R if sel(r)], key=lambda r: r["position_cm"])
        z = np.array([r["position_cm"] for r in pts])
        w = np.zeros(len(z))
        w[:-1] += 0.5 * np.diff(z); w[1:] += 0.5 * np.diff(z)
        E = float(np.sum(w * [r["E"] for r in pts]))
        sys_abs = float(np.sum(w * [r["E"] * r["sys"] for r in pts]))
        rnd_abs = float(math.sqrt(np.sum((w * [r["E"] * r["rnd"] for r in pts]) ** 2)))
        out.append(dict(id=nid, quantity=q, integral=[(r["id"], float(wi)) for r, wi in zip(pts, w)],
                        position_cm=None, E=E, tot=math.hypot(sys_abs, rnd_abs) / E, sys=sys_abs / E, rnd=rnd_abs / E,
                        corr_group=pts[0]["corr_group"], rid=pts[0]["rid"], mesh="axis",
                        note=f"trapezoid over {len(pts)} measured positions {z[0]:.1f}-{z[-1]:.1f} cm [unit x cm]"))
    return out


def compute(R, res):
    """Fill C and C_mc for every response from a run_bench result."""
    out = {}
    for r in R:
        if "integral" in r:
            continue
        if r["rid"] == "spectrum":
            out[r["id"]] = _spec_at(res, r["position_cm"])
            continue
        c, s = fnslib.at_positions(res, f"{r['rid']}@{r['mesh']}", [r["position_cm"]])
        out[r["id"]] = (float(c[0]), float(s[0]))
    for r in R:
        if "integral" in r:
            c = sum(w * out[i][0] for i, w in r["integral"])
            s = math.sqrt(sum((w * out[i][1]) ** 2 for i, w in r["integral"]))
            out[r["id"]] = (c, s)
    return out


def li2o_all(room_model):
    R = li2o_responses(room_model)
    R += integral_responses(R, [
        ("T6_lsc_int", "integral of 6Li TPR (Li pellets) over depth", lambda r: r["id"].startswith("t6_lsc_")),
        ("T7_lsc_int", "integral of 7Li TPR (Li pellets) over depth", lambda r: r["id"].startswith("t7_lsc_")),
        ("T6_lg_int", "integral of 6Li TPR (Li-glass, axis) over depth", lambda r: r["id"].startswith("t6_lg_ax_")),
        ("T7_ne213_int", "integral of 7Li TPR (NE213) over depth", lambda r: r["id"].startswith("t7_ne213_")),
    ])
    return R + spectrum_responses("li2o", D.LI2O_LENGTH)


def be_all():
    R = be_responses()
    R += integral_responses(R, [
        ("T6_lg_int", "integral of 6Li TPR (Li-glass) over depth", lambda r: r["id"].startswith("t6_lg_")),
        ("U235f_int", "integral of U-235 fission rate over depth", lambda r: r["id"].startswith("fc_u235_f_")),
    ])
    return R + spectrum_responses("be", D.BE_LENGTH)

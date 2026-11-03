"""Residence-time model (RTM) of an ARC-class D-T fuel cycle -> required TBR and start-up inventory.

Meschini, Ferry, Delaporte-Mathurin, Whyte, Nucl. Fusion 63 (2023) 126005, appendix A (eqs A.1-A.17,
tables 3, 4, A.12, A.13), itself adapted from Abdou et al., Nucl. Fusion 61 (2021) 013001.

Components (the paper's numbering): 1 breeding zone (LIB + channels), 2 tritium extraction (TES),
3 first wall, 4 divertor, 5 heat exchanger, 6 detritiation, 7 vacuum pumps, 8 fuel clean-up,
9 isotope separation (ISS), 10 storage, 12 membrane between TES and storage (11, fueling, is not
a reservoir: it is the outflow of 10).

Two places where the printed appendix is ambiguous and what is assumed here:
- eq A.12 prints dI12/dt = (1 - eta_TES) I2/tau2 - ..., the same coefficient as the HX feed (A.6);
  table A.12 has the TES -> membrane flow as eta_TES I2/tau2, used here (otherwise extracted
  tritium would never leave the TES path and eta would not appear in the balance).
- tau_12 (membrane) and the baseline eta_TES are not given; tau_12 = 1 h and eta_TES = 0.9 here
  (the paper states eta_TES moves TBR_r by < 1% between 0.4 and 0.95).

Plant availability is the paper's pulse source averaged over a pulse: every plasma-side flow
(breeding, implantation, pumping, fueling) is multiplied by AF. The paper notes this changes only
hour-scale dynamics (its footnote 11); `--pulsed` integrates the square wave to check that.

The system is linear with constant coefficients, so the averaged case is propagated exactly with a
matrix exponential on a log-spaced time grid.
"""
import argparse, json, math
import numpy as np
from scipy.linalg import expm

H, DAY, YR = 3600.0, 86400.0, 365.25 * 86400.0
LAM = math.log(2) / (12.32 * YR)

BASE = dict(
    Nburn=9.3e-7,      # kg/s, 525 MW
    TBE=0.02, AF=0.7, fDIR=0.3, tp_h=4.0, td_yr=2.0, tres_h=24.0, q=0.25, eps=1e-4,
    tau_BZ_h=1.25, tau_TES_h=24.0, tau_FW_s=1000.0, tau_div_s=1000.0, tau_HX_s=1000.0,
    tau_VP_s=600.0, tau_DS_h=1.0, tau_mem_h=1.0, eta_TES=0.9,
    f51=0.33, f53=0.33, f56=1e-4, f96=0.1, fp3=1e-4, fp4=1e-4,
)
IDX = {k: i for i, k in enumerate(["BZ", "TES", "FW", "div", "HX", "DS", "VP", "FC", "ISS", "storage", "mem"])}


def taus(p):
    # processing time t_p = tau_FC + tau_ISS, split in the paper's 0.3 : 3.7 ratio
    return dict(BZ=p["tau_BZ_h"] * H, TES=p["tau_TES_h"] * H, FW=p["tau_FW_s"], div=p["tau_div_s"], HX=p["tau_HX_s"],
                DS=p["tau_DS_h"] * H, VP=p["tau_VP_s"], FC=p["tp_h"] * H * 0.3 / 4.0, ISS=p["tp_h"] * H * 3.7 / 4.0,
                mem=p["tau_mem_h"] * H)


def system(p, TBR):
    """dI/dt = A I + on * (b0 + TBR b1), on = plasma-on fraction (AF when averaged)."""
    t = taus(p)
    n = len(IDX)
    A = np.zeros((n, n)); b0 = np.zeros(n); b1 = np.zeros(n)
    eps = {k: (0.0 if k in ("FW", "div", "storage") else p["eps"]) for k in IDX}
    for k, i in IDX.items():
        if k == "storage":
            A[i, i] = -LAM
        else:
            A[i, i] = -((1 + eps[k]) / t[k] + LAM)

    def flow(src, dst, frac):
        A[IDX[dst], IDX[src]] += frac / t[src]

    f54 = 1.0 - p["f51"] - p["f53"] - p["f56"]
    flow("FW", "BZ", 1); flow("div", "BZ", 1); flow("HX", "BZ", p["f51"])
    flow("BZ", "TES", 1)
    flow("HX", "FW", p["f53"]); flow("HX", "div", f54)
    flow("TES", "HX", 1 - p["eta_TES"]); flow("TES", "mem", p["eta_TES"])
    flow("HX", "DS", p["f56"]); flow("ISS", "DS", p["f96"])
    flow("VP", "FC", 1 - p["fDIR"]); flow("DS", "ISS", 1); flow("FC", "ISS", 1)
    flow("ISS", "storage", 1 - p["f96"]); flow("VP", "storage", p["fDIR"]); flow("mem", "storage", 1)
    inj = p["Nburn"] / p["TBE"]
    b1[IDX["BZ"]] = p["Nburn"]
    b0[IDX["FW"]] = p["fp3"] * inj
    b0[IDX["div"]] = p["fp4"] * inj
    b0[IDX["VP"]] = (1 - p["TBE"] - p["fp3"] - p["fp4"]) * inj
    b0[IDX["storage"]] = -inj
    return A, b0, b1


def time_grid(td):
    return np.unique(np.concatenate([[0.0], np.geomspace(60.0, td, 600), [td]]))


def propagate(A, b, ts, x0):
    """Exact solution of x' = A x + b on the grid ts (piecewise steps)."""
    n = len(x0)
    M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n] = b
    xs = [np.array(x0, float)]
    z = np.append(x0, 1.0)
    for dt in np.diff(ts):
        z = expm(M * dt) @ z
        xs.append(z[:n].copy())
    return np.array(xs)


def solve(p, tol=1e-6):
    """TBR_r and I_startup: storage >= I_res at all t, storage(td) = 2 I_startup (eqs 37, 38)."""
    td = p["td_yr"] * YR
    ts = time_grid(td)
    Ires = p["Nburn"] / p["TBE"] * p["q"] * p["tres_h"] * H
    s = IDX["storage"]
    g = np.exp(-LAM * ts)  # storage response to a unit start-up inventory (linear system)

    def storage(TBR):
        A, b0, b1 = system(p, TBR)
        return propagate(A, p["AF"] * (b0 + TBR * b1), ts, np.zeros(len(IDX)))

    def startup_for(h):
        return float(np.max((Ires - h) / g))

    def resid(TBR):
        h = storage(TBR)[:, s]
        I0 = startup_for(h)
        return I0 * g[-1] + h[-1] - 2 * I0, I0

    lo, hi = 1.0, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r, _ = resid(mid)
        (lo, hi) = (mid, hi) if r < 0 else (lo, mid)
        if hi - lo < tol:
            break
    TBR = 0.5 * (lo + hi)
    _, I0 = resid(TBR)
    A, b0, b1 = system(p, TBR)
    xs = propagate(A, p["AF"] * (b0 + TBR * b1), ts, np.eye(len(IDX))[s] * I0)
    k = int(np.argmin(xs[:, s]))
    return dict(TBR_r=TBR, I_startup_kg=I0, I_res_kg=Ires, t_infl_d=ts[k] / DAY,
                inventories_at_td_kg={kk: float(xs[-1, i]) for kk, i in IDX.items()})


def pulsed_check(p, TBR, I0, t_on=1800.0):
    """Square-wave operation: on for AF * t_on of each t_on period (the paper's pulse source)."""
    A, b0, b1 = system(p, TBR)
    n = len(IDX)
    b = b0 + TBR * b1
    M = np.zeros((n + 1, n + 1)); M[:n, :n] = A
    Mon = M.copy(); Mon[:n, n] = b
    a_on, a_off = p["AF"] * t_on, (1 - p["AF"]) * t_on
    Pon, Poff = expm(Mon * a_on), expm(M * a_off)
    P = Poff @ Pon
    z = np.append(np.eye(n)[IDX["storage"]] * I0, 1.0)
    td = p["td_yr"] * YR
    nper = int(td // t_on)
    mins = np.inf
    for i in range(nper):
        z1 = Pon @ z
        mins = min(mins, z1[IDX["storage"]], z[IDX["storage"]])
        z = Poff @ z1
        if i > 20000:  # past the inflection the minimum is behind us; jump ahead
            z = np.linalg.matrix_power(P, nper - i - 1) @ z
            break
    return dict(min_storage_kg=float(mins), storage_td_kg=float(z[IDX["storage"]]))


TABLE6 = {  # Meschini 2023 table 6: (overrides, TBR_r, I_startup)
    "baseline td=1": (dict(td_yr=1.0), 1.113, 1.14),
    "baseline td=2": (dict(td_yr=2.0), 1.067, 1.14),
    "td=1 FC": (dict(td_yr=1.0, AF=0.9, fDIR=0.7, tp_h=1.0), 1.064, 1.12),
    "td=1 PO": (dict(td_yr=1.0, TBE=0.1), 1.029, 0.33),
    "td=1 both": (dict(td_yr=1.0, AF=0.9, fDIR=0.7, tp_h=1.0, TBE=0.1), 1.022, 0.30),
    "td=2 FC": (dict(td_yr=2.0, AF=0.9, fDIR=0.7, tp_h=1.0), 1.039, 1.12),
    "td=2 PO": (dict(td_yr=2.0, TBE=0.1), 1.017, 0.33),
    "td=2 both": (dict(td_yr=2.0, AF=0.9, fDIR=0.7, tp_h=1.0, TBE=0.1), 1.012, 0.30),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulsed", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    print(f"{'case':16s} {'TBR_r':>7s} {'paper':>7s} {'I_su kg':>8s} {'paper':>6s} {'t_infl d':>8s}")
    for name, (ov, tbr_ref, isu_ref) in TABLE6.items():
        p = dict(BASE, **ov)
        r = solve(p)
        out[name] = dict(r, paper_TBR_r=tbr_ref, paper_I_startup_kg=isu_ref, params=ov)
        print(f"{name:16s} {r['TBR_r']:7.3f} {tbr_ref:7.3f} {r['I_startup_kg']:8.2f} {isu_ref:6.2f} {r['t_infl_d']:8.2f}")
        if a.pulsed:
            pc = pulsed_check(p, r["TBR_r"], r["I_startup_kg"])
            print(f"   pulsed: min storage {pc['min_storage_kg']:.3f} (I_res {r['I_res_kg']:.3f}), at td {pc['storage_td_kg']:.3f} (2 I_su {2*r['I_startup_kg']:.3f})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)

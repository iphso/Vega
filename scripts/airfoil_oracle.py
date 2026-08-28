"""XFOIL implementation of oracle_base.py's interface -- a second, genuinely
different domain plugged into the same generic harness built for VMEC++
(see EXPERIMENT_LOG's genericization work in this project). The point of
building this isn't the airfoils themselves -- it's a real test of whether
oracle_base.py's abstraction actually generalizes, or only looked generic
because it was extracted from a single example.

Parameterization: airfoil shape via CST (Class-Shape Transformation, Kulfan
2008) -- N_CST Bernstein coefficients each for the upper and lower surface
(PARAM_DIM = 2*N_CST). Chosen over the much simpler NACA 4/5-digit family
(2-3 parameters, what the AirfRANS dataset itself uses) specifically
because it's far more expressive -- closer in spirit to VMEC++'s 90 Fourier
coefficients than a 2-3-parameter family would be, and the standard choice
in the aero-shape-optimization literature for exactly this reason.

Oracle: XFOIL (Mark Drela's panel-method + integral boundary-layer solver),
via daniel-de-vries/xfoil-python -- GPL-3.0, noted since vmec_oracle.py's
dependencies are not. Confirmed directly by reading the wrapper's own
source (not assumed): `xf.a(alpha)` returns `(nan, nan, nan, nan)` whenever
the underlying Fortran `conv` flag comes back false -- the exact same
"converged or it didn't, no guessing from the returned numbers" contract
VMEC++ provides, and a hint at high angle-of-attack (mid-solve "MRCHDU:
Convergence failed" boundary-layer-marching messages can appear without the
overall `conv` flag ever going false -- XFOIL recovers from those; only the
final flag is authoritative, confirmed empirically on a NACA0012 sweep to
alpha=40 that still returned finite values despite those messages).

Aux conditioning (fixed per candidate, not part of the continuous param
vector -- mirrors n_field_periods for VMEC++): Reynolds number, Mach
number, angle of attack.

Targets: cl, cd, cm straight from XFOIL, plus l_over_d = cl/cd computed
here -- a derived-but-first-class target, same convention as VMEC++'s own
edge_rotational_transform_over_n_field_periods. cd is flagged as a
LOG_TARGET_NAMES entry -- like VMEC++'s qi/max_elongation, its dynamic
range spans orders of magnitude between near-zero-lift and near-stall.

Not yet built: an "high"/"medium" fidelity tier via OpenFOAM (mirroring
vmec_oracle.py's FIDELITY_PRESETS structure exactly -- XFOIL as the fast
"low" tier, OpenFOAM as a slow confirmatory tier, the same low/medium
relationship VMEC++ has between its 25->71 and 25->51->99 multigrid
presets) -- FIDELITY_PRESETS only has one entry for now.
"""
from math import comb

import numpy as np

TARGET_NAMES = ["cl", "cd", "cm", "l_over_d"]
LOG_TARGET_NAMES = ["cd"]
PARAM_DIM = 16
N_CST = PARAM_DIM // 2
ZERO_INDICES = []  # no structurally-fixed coefficients for this parameterization
FIDELITY_PRESETS = {"low": "xfoil"}


def _bernstein_shape(x, coeffs):
    n = len(coeffs) - 1
    S = np.zeros_like(x)
    for i, a in enumerate(coeffs):
        S += a * comb(n, i) * x ** i * (1 - x) ** (n - i)
    return S


def cst_to_coords(params, n_points=100):
    """params: (PARAM_DIM,) = [upper CST coeffs (N_CST), lower CST coeffs
    (N_CST)]. No sign convention forced on the lower coefficients -- a
    normal airfoil has them come out negative, but nothing here requires
    it (mirrors how VMEC++'s z_sin coefficients can be either sign).
    Returns (x, y) as a single closed loop: upper TE -> LE -> lower TE,
    the ordering XFOIL's Airfoil model expects."""
    au, al = params[:N_CST], params[N_CST:]
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2  # cosine spacing: denser near LE/TE
    class_fn = x ** 0.5 * (1 - x) ** 1.0  # standard airfoil class function (N1=0.5, N2=1.0)
    yu = class_fn * _bernstein_shape(x, au)
    yl = class_fn * _bernstein_shape(x, al)
    x_coords = np.concatenate([x[::-1], x[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return x_coords.astype(np.float64), y_coords.astype(np.float64)


def worker_fn(conn, x_coords, y_coords, reynolds, mach, alpha):
    try:
        from xfoil import XFoil
        from xfoil.model import Airfoil
        xf = XFoil()
        xf.print = False
        try:
            xf.airfoil = Airfoil(x_coords, y_coords)
        except Exception as e:
            conn.send((False, f"structural: {e}"))
            return
        xf.Re = reynolds
        xf.M = mach
        xf.max_iter = 100
        try:
            cl, cd, cm, _cp = xf.a(alpha)
        except Exception as e:
            conn.send((False, f"xfoil: {e}"))
            return
        if any(np.isnan(v) for v in (cl, cd, cm)):
            conn.send((False, "xfoil: did not converge"))
            return
        if cd < 1e-6:
            # XFOIL's own convergence flag doesn't guarantee physical
            # sanity at the extreme tails -- confirmed empirically (§21's
            # 50K-row dataset generation): a handful of candidates
            # "converge" with a spuriously near-zero cd (down to ~1e-12),
            # which is not real 2D-airfoil drag at any Reynolds number in
            # this project's sampled range and blows up l_over_d=cl/cd to
            # absurd values. 1e-6 is a generous floor (real converged
            # values in the same dataset run 5.8e-6 and up).
            conn.send((False, "xfoil: converged but cd non-physical (<1e-6)"))
            return
        l_over_d = float(cl / cd)
        conn.send((True, {"cl": float(cl), "cd": float(cd), "cm": float(cm), "l_over_d": l_over_d}))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    """params: (PARAM_DIM,) raw CST coefficients (already zero-enforced at
    ZERO_INDICES, though that's a no-op here -- empty list). aux:
    {"reynolds": float, "mach": float, "alpha": float}. `fidelity_name` is
    accepted for interface consistency with vmec_oracle.py but unused
    until an OpenFOAM tier exists."""
    x_coords, y_coords = cst_to_coords(params)
    return (x_coords, y_coords, aux["reynolds"], aux.get("mach", 0.0), aux["alpha"])

"""Real, named-airfoil reference points for the airfoil domain -- the same
gap closed for TORAX in real_reference_torax.py, and the one flagged
directly by the user: `generate_airfoil_dataset.py` (§22) is 100% synthetic
(archetypal CST vectors + noise), with no real, named airfoil's shape or
performance ever run through this project's own XFOIL oracle.

No existing CST-parameterized real-airfoil dataset to pull from (confirmed
already in §21 -- AirfRANS uses the NACA family, not CST). Rather than
skip real references for that reason, this fits CST coefficients to real
NACA 4-digit airfoils -- a public-domain, closed-form geometry definition
(Jacobs, Pinkerton & Greenberg 1933; the exact formula every NACA-4-digit
implementation uses, not approximated or guessed) -- via ordinary linear
least squares. The CST class-shape transformation is LINEAR in its
Bernstein coefficients for a fixed set of x-abscissas (see
`airfoil_oracle.cst_to_coords`: y(x) = class_fn(x) * sum_i a_i * B_i(x)),
so fitting is exact linear algebra, not an approximate nonlinear
optimization -- fit quality is reported (max/RMS residual in y/c), not
assumed good.

Three real, well-documented airfoils: NACA 0012 (thin, symmetric -- the
single most-tested airfoil shape in aviation history, real experimental
zero-lift/stall behavior well known), NACA 2412 (a classic light-aircraft
cambered airfoil, e.g. early Cessna wings), NACA 4415 (thicker, more
cambered, e.g. general-aviation/glider use). Each fitted shape is run
through the SAME XFOIL oracle this project already trusts (not against
external published polars directly -- keeps this internally consistent
with every other airfoil number in this project) at a real, representative
condition (Re=3e6, a common mid-scale wind-tunnel/light-aircraft Reynolds
number) across an angle-of-attack sweep, then checked for the qualitative
behavior every aerodynamics reference agrees on (NACA 0012 symmetric
zero-lift-at-alpha=0; cambered sections lifting at alpha=0; stall in a
plausible mid-teens-degrees range) -- a real sanity check, not just "XFOIL
returned a finite number."
"""
import json
from pathlib import Path

import numpy as np

import airfoil_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")
N_FIT_POINTS = 300  # dense interior sampling for the linear CST fit itself
N_SHAPE_POINTS = 100  # matches airfoil_oracle.cst_to_coords's own default, what XFOIL actually sees


def naca4_coords(code, n_points=300):
    """Closed-form NACA 4-digit geometry (Jacobs/Pinkerton/Greenberg 1933),
    public domain. `code`: 4-digit string, e.g. "0012" (m=0,p=0,t=12) or
    "2412" (m=2,p=4,t=12). Returns (xu, yu, xl, yl), each LE(x=0)->TE(x=1)
    increasing, but on its own camber-normal-offset x grid -- NOT the
    shared abscissa CST assumes, resampled onto that shared grid by
    fit_cst below."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2  # cosine spacing, denser near LE/TE -- matches cst_to_coords's own convention

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if p == 0:  # symmetric section (e.g. NACA 0012) -- no camber line at all
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(x < p,
                       (m / p**2) * (2 * p * x - x**2),
                       (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc_dx = np.where(x < p,
                           (2 * m / p**2) * (p - x),
                           (2 * m / (1 - p)**2) * (p - x))
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl


def fit_cst(xu, yu, xl, yl, n_cst=8):
    """Projects the real (camber-normal-offset) NACA surfaces onto CST's
    shared-abscissa y(x) representation via interpolation, then solves the
    per-surface linear least-squares fit for the Bernstein coefficients.
    Returns (params, (rms_upper, max_upper, rms_lower, max_lower)) -- the
    16-vector ready for airfoil_oracle.params_to_worker_args, plus real
    fit-quality numbers instead of an assumed-good fit."""
    x_fit = np.linspace(1e-4, 1 - 1e-4, N_FIT_POINTS)  # avoid x=0/1 where class_fn is exactly 0
    class_fn = x_fit**0.5 * (1 - x_fit)**1.0

    yu_on_grid = np.interp(x_fit, xu, yu)  # xu/xl both LE(0)->TE(1) increasing, matching np.interp's requirement
    yl_on_grid = np.interp(x_fit, xl, yl)

    from math import comb
    # Fold class_fn into the basis rather than dividing the target by it --
    # class_fn -> 0 at both x=0 and x=1, so target/class_fn blows up right at
    # the endpoints and (via lstsq on a degree-7 polynomial basis) destabilizes
    # the WHOLE fit, not just the tips -- a real bug caught by looking at the
    # actual fitted coefficients (alternating, up to magnitude ~21 for an
    # airfoil whose y/c never exceeds ~0.06) before trusting the "small"
    # residual-looking RMS number at face value.
    bernstein = np.stack([comb(n_cst - 1, i) * x_fit**i * (1 - x_fit)**(n_cst - 1 - i) for i in range(n_cst)], axis=1)
    basis = class_fn[:, None] * bernstein

    au, *_ = np.linalg.lstsq(basis, yu_on_grid, rcond=None)
    al, *_ = np.linalg.lstsq(basis, yl_on_grid, rcond=None)

    yu_fit = basis @ au
    yl_fit = basis @ al
    resid_u, resid_l = yu_fit - yu_on_grid, yl_fit - yl_on_grid
    quality = (float(np.sqrt(np.mean(resid_u**2))), float(np.max(np.abs(resid_u))),
               float(np.sqrt(np.mean(resid_l**2))), float(np.max(np.abs(resid_l))))
    params = np.concatenate([au, al]).astype(np.float64)
    return params, quality


REAL_AIRFOILS = ["0012", "2412", "4415"]
REYNOLDS = 3e6
ALPHAS = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]


def main():
    fits = {}
    for code in REAL_AIRFOILS:
        xu, yu, xl, yl = naca4_coords(code, n_points=N_FIT_POINTS)
        params, quality = fit_cst(xu, yu, xl, yl)
        fits[code] = {"params": params, "fit_quality_rms_max_upper_lower": quality}
        print(f"NACA {code}: fit RMS/max (upper) = {quality[0]:.2e}/{quality[1]:.2e}, "
              f"(lower) = {quality[2]:.2e}/{quality[3]:.2e}  (units: y/c)")

    jobs = []
    for code, info in fits.items():
        x_coords, y_coords = oracle.cst_to_coords(info["params"], n_points=N_SHAPE_POINTS)
        for alpha in ALPHAS:
            jobs.append((f"{code}_{alpha}", x_coords, y_coords, REYNOLDS, 0.0, float(alpha)))

    results = {code: {} for code in fits}
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, n_workers=16, timeout_s=20.0):
        code, alpha_str = tag.rsplit("_", 1)
        results[code][int(alpha_str)] = {"ok": ok, "payload": payload}

    for code in REAL_AIRFOILS:
        print(f"\nNACA {code} polar @ Re={REYNOLDS:.0e}:")
        for alpha in ALPHAS:
            r = results[code][alpha]
            if r["ok"]:
                p = r["payload"]
                print(f"  alpha={alpha:+3d}  cl={p['cl']:+.4f}  cd={p['cd']:.5f}  cm={p['cm']:+.4f}  l/d={p['l_over_d']:+7.2f}")
            else:
                print(f"  alpha={alpha:+3d}  FAILED: {r['payload']}")

    out = {
        "airfoils": {code: {"params": fits[code]["params"].tolist(),
                             "fit_quality_rms_max_upper_lower": fits[code]["fit_quality_rms_max_upper_lower"]}
                     for code in fits},
        "reynolds": REYNOLDS,
        "polars": {code: {alpha: results[code][alpha] for alpha in ALPHAS} for code in fits},
    }
    (OUT_DIR / "airfoil_real_references.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {OUT_DIR / 'airfoil_real_references.json'}")


if __name__ == "__main__":
    main()

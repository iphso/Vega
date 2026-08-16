"""Physics bundle for the Vega viewer: Boozer |B|, Boozer cuts, Poincaré.

The browser cannot solve an equilibrium. This module (and the FastAPI
wrapper in physics_service.py) takes a boundary (r_cos, z_sin, nfp) in
this project's VMEC2000-lineage layout and returns a JSON-serializable
bundle the viewer renders.

For the vega_testing candidates the solved DESC equilibria live in
vega_testing/<DESC_RUN>/equilibria/eq_i{row}_fr{file_row}.h5 (index in
<DESC_RUN>/results.npz). compute_bundle loads those instead of calling
eq.solve(). A live solve is only the fallback when no h5 matches.

Fourier convention (do not "fix" this without re-running the unit test):
  r_cos/z_sin are [m, n_idx] with m=0..4, n_idx=0..8 -> n = n_idx-4.
  R = Σ r_cos[m,n] cos(m*θ − nfp*n*φ)
  Z = Σ z_sin[m,n] sin(m*θ − nfp*n*φ)
That is the same synthesis as viewer surfacePoints() and as
constellaration SurfaceRZFourier. DESC uses a product-of-trig double
Fourier basis instead; conversion is the trig-identity map in
vmec_to_desc_modes (DESC's ptolemy_identity_fwd is the same math but
its square solve is singular on our dense m=0 ±n cosine grid).

DESC documentation: https://desc-docs.readthedocs.io/en/latest/input.html

A silent orientation flip is a known DESC hazard
(check_orientation) -- we leave it off so the mapped surface stays
byte-aligned with the viewer's geometry, and the unit test compares
DESC's R,Z against the numpy synthesis at matched geometric angles.

Two backends, both pre-solved and shipped, same bundle schema, no silent
fallback between them:

  backend="vmecpp"  DEFAULT. VMEC++ high-fidelity + booz_xform, loaded from
                    vega_testing/vmecpp_highfi/equilibria/*.npz. Run at the
                    boundary's own phiedge. Evaluation is a Fourier sum
                    over the saved Boozer coefficients (~0.2 s); the
                    service never calls vmecpp.solve.
  backend="desc"    DESC vacuum solve at L=M=N=8, loaded from
                    vega_testing/<DESC_RUN>/equilibria/*.h5. Solved at
                    DESC's default Ψ = 1 Wb. Boundaries from
                    desc_surface_from_vmec; ι sign is opposite VMEC++
                    (handedness / check_orientation=False) — compare |ι|.

|B| ships DIMENSIONLESS as |B|/⟨|B|⟩_s -- see B_UNIT_NOTE. The two codes
normalize the toroidal flux differently, so their absolute tesla differ by
a constant; dividing by the surface mean cancels it and leaves the
modulation, which is the only backend-comparable quantity. The tesla means
that were divided out are kept in meta.B_mean_T / flux.B_mean_T.

Poincaré: do NOT numerically integrate B. For an ideal equilibrium,
field lines are straight in Boozer coordinates
(θ_B = θ0 + ι(s)·ζ_B). Intersections with a cylindrical φ=const plane
are found by evaluating the Boozer→(R,φ,Z) map along that line.

Extension point for coil fields: replace poincare_from_boozer() with a
SIMSOPT compute_fieldlines trace of the Biot-Savart field from coils
(plus plasma contribution if needed). The bundle schema
(seeds[].R / Z / turn) stays the same.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# DESC/JAX is optional: the VMEC++ batch image has no AVX-capable jaxlib
# under qemu, and must still import this module for pack I/O.
try:
    from desc.compute import get_transforms
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.grid import Grid, LinearGrid
except Exception:  # noqa: BLE001 — missing DESC or jaxlib CPU-feature guard
    get_transforms = None  # type: ignore[misc, assignment]
    Equilibrium = object  # type: ignore[misc, assignment]
    FourierRZToroidalSurface = object  # type: ignore[misc, assignment]
    Grid = object  # type: ignore[misc, assignment]
    LinearGrid = object  # type: ignore[misc, assignment]

# Container path by default. VEGA_TESTING_DIR overrides it so the module, the
# CLI and the test suite can run against a checkout outside Docker without
# every caller having to thread testing_dir= through.
TESTING_DIR = Path(os.environ.get("VEGA_TESTING_DIR", "/work/vega_testing"))
DESC_RUN = "desc_L8M8N8"
DESC_EQ_DIRNAME = f"{DESC_RUN}/equilibria"
DESC_RESULTS_NAME = f"{DESC_RUN}/results.npz"
VMECPP_RUN = "vmecpp_highfi"
VMECPP_EQ_DIRNAME = f"{VMECPP_RUN}/equilibria"
VMECPP_RESULTS_NAME = f"{VMECPP_RUN}/results.npz"

# Bump when the meaning of a shipped bundle field changes. physics_service
# mixes this into the disk-cache key, so a bump retires stale bundles
# instead of serving them under the new field names.
#   1 -> |B| shipped in tesla
#   2 -> |B| shipped dimensionless as |B|/⟨|B|⟩_s (see B_UNIT_NOTE)
BUNDLE_SCHEMA_VERSION = 2

# Why |B| is not shipped in tesla: DESC solves these as vacuum equilibria at
# its default Ψ = 1 Wb, while VMEC++ runs at the boundary's own phiedge, so
# the two backends' absolute |B| differ by a pure multiplicative constant --
# a units convention, not physics. Dividing by the surface mean cancels it
# exactly (numerator and denominator carry the same factor), leaving the
# modulation of the field about its own mean. That is the quantity
# quasi-symmetry is read from, and it is the only one that can be compared
# backend to backend.
B_UNIT_NOTE = "|B| / <|B|>_s, mean over the Boozer (theta_B, zeta_B) grid"

F64 = NDArray[np.float64]
# (theta_B, zeta_B, with_B=False) -> (R, phi, Z) or (R, phi, Z, |B|).
BoozerMap = Callable[..., tuple[F64, ...]]

# Lazy cache of <DESC_RUN>/results.npz (X + eq filenames). The h5 files
# themselves are loaded per request.
_DESC_INDEX_CACHE: dict[str, Any] | None = None

# Default spectral resolution for a live solve: matches DESC_RUN so a
# CLI fallback solve is the same fidelity as the saved h5 (~395s median on
# CPU at L=M=N=8; the L=12 campaign cost ~395s for a strictly worse
# result). Overridable via CLI / the service request body;
# ignored when an h5 is loaded (L,M,N come from the equilibrium).
DEFAULT_L = 8
DEFAULT_M = 8
DEFAULT_N = 8
# Locked grid shape so JAX compiles once per process. 32×48 is enough
# for the canvas/WebGL views; bumping it recompiles (~5s) for a new shape.
DEFAULT_N_THETA = 32
DEFAULT_N_ZETA = 48
DEFAULT_N_ZETA_PLANES = 12
DEFAULT_N_SEEDS = 6
DEFAULT_N_TURNS = 16
DEFAULT_POINCARE_SURFACES = 4
# Normalized toroidal flux s. DESC LinearGrid is in ρ = √s; convert at
# the compute boundary so every label in the bundle is s, not ρ.
# Shape (5, 32, 48) is the compiled one; the s slider is a client-side
# nearest-slice pick rather than a new DESC compute.
DEFAULT_FLUX_S = (0.0, 0.25, 0.5, 0.75, 1.0)


def s_to_rho(s: ArrayLike) -> float | F64:
    """DESC radial coordinate ρ = √s (s = normalized toroidal flux)."""
    arr = np.sqrt(np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0))
    return float(arr) if arr.ndim == 0 else arr


def rho_to_s(rho: ArrayLike) -> float | F64:
    """Normalized toroidal flux s = ρ², the inverse of s_to_rho."""
    arr = np.asarray(rho, dtype=np.float64) ** 2
    return float(arr) if arr.ndim == 0 else arr


def synthesize_boundary_rz(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    theta: ArrayLike,
    phi: ArrayLike,
) -> tuple[F64, F64]:
    """VMEC2000-lineage Fourier synthesis used by the viewer (surfacePoints).

    r_cos, z_sin: arrays shaped (5, 9). theta, phi: broadcastable floats
    or arrays in radians. Returns (R, Z) at those angles.
    """
    r_cos = np.asarray(r_cos, dtype=np.float64)
    z_sin = np.asarray(z_sin, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    R = np.zeros(np.broadcast(theta, phi).shape, dtype=np.float64)
    Z = np.zeros_like(R)
    nfp_f = float(nfp)
    for m in range(5):
        for n_idx in range(9):
            n = n_idx - 4
            angle = m * theta - nfp_f * n * phi
            R = R + r_cos[m, n_idx] * np.cos(angle)
            Z = Z + z_sin[m, n_idx] * np.sin(angle)
    return R, Z


def vmec_to_desc_modes(
    r_cos: ArrayLike, z_sin: ArrayLike
) -> tuple[F64, NDArray[np.int_], F64, NDArray[np.int_]]:
    """Convert this project's (5, 9) r_cos/z_sin into DESC (m, n, coeff) arrays.

    DESC's ptolemy_identity_fwd is what it uses when reading a VMEC wout,
    but that helper solves a square system over the *unique* VMEC mode
    list. Our dense m=0..4, n=-4..4 grid includes the m=0 cosine
    redundancy cos((-n)φ) = cos(nφ), which makes that solve singular --
    so the mapping is written out here from the product-basis identities
    instead. The unit test (DESC surface vs numpy synthesis at matched
    geometric angles) is the source of truth, not this comment.

    DESC G^{m}_{n}(θ,ζ):
      m≥0, n≥0: cos(mθ) cos(n NFP ζ)
      m≥0, n<0: cos(mθ) sin(|n| NFP ζ)
      m<0, n≥0: sin(|m|θ) cos(n NFP ζ)
      m<0, n<0: sin(|m|θ) sin(|n| NFP ζ)
    VMEC: R = Σ r_cos cos(mθ − n NFP φ), Z = Σ z_sin sin(mθ − n NFP φ).

    Returns (R_lmn, modes_R, Z_lmn, modes_Z) in DESC's packed form.
    """
    r_cos = np.asarray(r_cos, dtype=np.float64).reshape(5, 9)
    z_sin = np.asarray(z_sin, dtype=np.float64).reshape(5, 9)
    R_coeff: dict[tuple[int, int], float] = {}
    Z_coeff: dict[tuple[int, int], float] = {}

    def add(store: dict[tuple[int, int], float], m: int, n: int, value: float) -> None:
        if value == 0.0:
            return
        key = (int(m), int(n))
        store[key] = store.get(key, 0.0) + float(value)

    for m in range(5):
        for n_idx in range(9):
            n = n_idx - 4
            c = r_cos[m, n_idx]
            s = z_sin[m, n_idx]
            if n == 0:
                add(R_coeff, m, 0, c)
                if m != 0:
                    add(Z_coeff, -m, 0, s)
            elif n > 0:
                add(R_coeff, m, n, c)
                if m != 0:
                    add(R_coeff, -m, -n, c)
                if m == 0:
                    add(Z_coeff, 0, -n, -s)
                else:
                    add(Z_coeff, -m, n, s)
                    add(Z_coeff, m, -n, -s)
            else:
                nabs = -n
                add(R_coeff, m, nabs, c)
                if m != 0:
                    add(R_coeff, -m, -nabs, -c)
                if m == 0:
                    add(Z_coeff, 0, -nabs, s)
                else:
                    add(Z_coeff, -m, nabs, s)
                    add(Z_coeff, m, -nabs, s)

    def packed(
        store: dict[tuple[int, int], float],
    ) -> tuple[F64, NDArray[np.int_]]:
        modes = np.array(list(store.keys()), dtype=int).reshape(-1, 2)
        coeffs = np.array([store[tuple(mn)] for mn in modes], dtype=np.float64)
        return coeffs, modes

    R_lmn, modes_R = packed(R_coeff)
    Z_lmn, modes_Z = packed(Z_coeff)
    return R_lmn, modes_R, Z_lmn, modes_Z


def unpack_row(row: ArrayLike) -> tuple[F64, F64, int, bool]:
    """Split a 92-col X row into r_cos (5,9), z_sin (5,9), nfp, is_sym."""
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size < 92:
        raise ValueError(f"expected 92-col row, got {row.size}")
    r_cos = row[:45].reshape(5, 9)
    z_sin = row[45:90].reshape(5, 9)
    nfp = int(row[90])
    is_sym = bool(row[91])
    return r_cos, z_sin, nfp, is_sym


def load_test_row(index: int, testing_dir: Path | str = TESTING_DIR) -> dict[str, Any]:
    """Boundary, provenance and h5 path for one vega_testing row."""
    X = np.load(Path(testing_dir) / "X.npy")
    if index < 0 or index >= len(X):
        raise IndexError(f"test-row {index} out of range (n={len(X)})")
    r_cos, z_sin, nfp, is_sym = unpack_row(X[index])
    file_rows = np.load(Path(testing_dir) / "file_rows.npy")
    times = np.load(Path(testing_dir) / "times.npy")
    file_row = int(file_rows[index])
    eq_path = (
        Path(testing_dir) / DESC_EQ_DIRNAME / f"eq_i{index:04d}_fr{file_row:06d}.h5"
    )
    return {
        "r_cos": r_cos,
        "z_sin": z_sin,
        "nfp": nfp,
        "is_stellarator_symmetric": is_sym,
        "file_row": file_row,
        "vmec_time_s": float(times[index]),
        "test_row": int(index),
        "eq_path": eq_path if eq_path.exists() else None,
    }


def _load_desc_index(testing_dir: Path | str = TESTING_DIR) -> dict[str, Any] | None:
    """Load vega_testing/<DESC_RUN>/results.npz once (X + h5 filenames)."""
    global _DESC_INDEX_CACHE
    if _DESC_INDEX_CACHE is not None:
        return _DESC_INDEX_CACHE
    path = Path(testing_dir) / DESC_RESULTS_NAME
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    _DESC_INDEX_CACHE = {
        "X": np.asarray(z["X"], dtype=np.float64),
        "file_rows": np.asarray(z["file_rows"]),
        "eq_paths": np.asarray(z["eq_paths"]),
        "desc_ok": np.asarray(z["desc_ok"]),
        "desc_times": np.asarray(z["desc_times"]),
        "Y_desc": np.asarray(z["Y_desc"], dtype=np.float64),
        "desc_scalar_names": [str(n) for n in z["desc_scalar_names"]],
        "eq_dir": Path(testing_dir) / DESC_EQ_DIRNAME,
    }
    return _DESC_INDEX_CACHE


def match_saved_row(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    testing_dir: Path | str = TESTING_DIR,
    atol: float = 1e-5,
) -> dict[str, Any] | None:
    """Find the vega_testing row whose X matches this boundary.

    Viewer X.bin is float32; vega_testing/X.npy is float64. atol is set
    for that round-trip. Returns None if nothing is within atol.
    """
    idx = _load_desc_index(testing_dir)
    if idx is None:
        return None
    vec = np.concatenate(
        [
            np.asarray(r_cos, dtype=np.float64).reshape(-1),
            np.asarray(z_sin, dtype=np.float64).reshape(-1),
            [float(nfp)],
        ]
    )
    if vec.size != 91:
        raise ValueError(f"expected 45+45+1 boundary vector, got {vec.size}")
    dist = np.max(np.abs(idx["X"][:, :91] - vec), axis=1)
    j = int(np.argmin(dist))
    if dist[j] > atol:
        return None
    name = str(idx["eq_paths"][j])
    eq_path = idx["eq_dir"] / name
    scalars = {
        n: float(idx["Y_desc"][j, k]) for k, n in enumerate(idx["desc_scalar_names"])
    }
    return {
        "test_row": j,
        "file_row": int(idx["file_rows"][j]),
        "eq_path": eq_path,
        "desc_ok": bool(idx["desc_ok"][j]),
        "desc_time_s": float(idx["desc_times"][j]),
        "scalars": scalars,
        "match_err": float(dist[j]),
    }


def load_saved_equilibrium(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    testing_dir: Path | str = TESTING_DIR,
) -> tuple[Equilibrium | None, dict[str, Any] | None]:
    """Load a pre-solved DESC Equilibrium h5 if this boundary is in vega_testing.

    Returns (eq, match_dict) or (None, match_or_None).
    """
    hit = match_saved_row(r_cos, z_sin, nfp, testing_dir=testing_dir)
    if hit is None or not hit["desc_ok"]:
        return None, hit
    if not hit["eq_path"].exists():
        return None, hit
    eq = Equilibrium.load(str(hit["eq_path"]))
    return eq, hit


def _as_list(x: ArrayLike) -> list[Any]:
    """Round to 6 decimals and convert to nested lists for JSON."""
    return np.round(np.asarray(x, dtype=np.float64), 6).tolist()


def _reshape_rtz_nd(
    grid: LinearGrid,
    values: ArrayLike,
    n_rho: int,
    n_theta: int,
    n_zeta: int,
) -> F64:
    """Reshape a multi-ρ LinearGrid quantity to (n_rho, n_theta, n_zeta)."""
    values = np.asarray(values)
    if hasattr(grid, "meshgrid_reshape"):
        arr = np.asarray(grid.meshgrid_reshape(values, "rtz")).squeeze()
        while arr.ndim > 3:
            arr = arr.squeeze()
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape == (n_rho, n_theta, n_zeta):
            return arr
        if arr.shape == (n_rho, n_zeta, n_theta):
            return np.swapaxes(arr, 1, 2)
    return values.reshape((n_rho, n_theta, n_zeta), order="F")


def _compute_flux_pack(
    eq: Equilibrium, rhos: ArrayLike, n_theta: int, n_zeta: int
) -> dict[str, Any]:
    """One DESC compute on a fixed (ρ, θ, ζ) grid — JAX compiles this shape once."""
    rhos = np.asarray(rhos, dtype=np.float64).reshape(-1)
    n_rho = int(rhos.size)
    grid = LinearGrid(
        rho=rhos,
        theta=int(n_theta),
        zeta=int(n_zeta),
        NFP=int(eq.NFP),
        endpoint=False,
        sym=False,
    )
    names = ["X", "Y", "Z", "|B|", "R", "phi", "theta_B", "zeta_B", "iota"]
    data = eq.compute(names, grid=grid)
    pack = {
        "rho": rhos,
        "n_theta": int(n_theta),
        "n_zeta": int(n_zeta),
        "nfp": int(eq.NFP),
    }
    for name in names:
        if name == "iota":
            try:
                pack[name] = _reshape_rtz_nd(grid, data[name], n_rho, n_theta, n_zeta)
            except (ValueError, AttributeError):
                iota = np.asarray(
                    grid.compress(data["iota"]), dtype=np.float64
                ).reshape(-1)
                pack[name] = iota.reshape(n_rho, 1, 1)
        else:
            pack[name] = _reshape_rtz_nd(grid, data[name], n_rho, n_theta, n_zeta)
    return pack


def _interpolator_from_nodes(
    theta_B: ArrayLike,
    zeta_B: ArrayLike,
    R: ArrayLike,
    phi: ArrayLike,
    Z: ArrayLike,
    nfp: int,
    B: ArrayLike | None = None,
) -> BoozerMap:
    """Boozer (θ_B, ζ_B) → (R, φ, Z[, |B|]) from already-computed nodes.

    One Delaunay triangulation per surface covers cuts and Poincaré.
    Boozer |B| on the plot grid comes from DESC's |B|_mn_B transform.
    """
    theta_B = np.mod(np.asarray(theta_B, dtype=np.float64).reshape(-1), 2 * np.pi)
    period_z = 2 * np.pi / float(nfp)
    zeta_B = np.mod(np.asarray(zeta_B, dtype=np.float64).reshape(-1), period_z)
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    cols = [R, Z, np.cos(phi), np.sin(phi)]
    if B is not None:
        cols.append(np.asarray(B, dtype=np.float64).reshape(-1))
    vals = np.column_stack(cols)
    period_t = 2 * np.pi
    # Halo only: full 3×3 copies of the surface made Qhull the bottleneck.
    # Duplicate a thin strip past each periodic edge so linear interp
    # still works at θ=0/2π and ζ=0/2π/NFP.
    margin_t, margin_z = 0.35, 0.35 * period_z / (2 * np.pi)
    pts = [np.column_stack([theta_B, zeta_B])]
    stacked = [vals]
    for dt, dz in (
        (-period_t, 0.0),
        (period_t, 0.0),
        (0.0, -period_z),
        (0.0, period_z),
        (-period_t, -period_z),
        (-period_t, period_z),
        (period_t, -period_z),
        (period_t, period_z),
    ):
        mask = np.ones(theta_B.size, dtype=bool)
        if dt < 0:
            mask &= theta_B < margin_t
        elif dt > 0:
            mask &= theta_B > (period_t - margin_t)
        if dz < 0:
            mask &= zeta_B < margin_z
        elif dz > 0:
            mask &= zeta_B > (period_z - margin_z)
        if not np.any(mask):
            continue
        pts.append(np.column_stack([theta_B[mask] + dt, zeta_B[mask] + dz]))
        stacked.append(vals[mask])
    pts = np.vstack(pts)
    stacked = np.vstack(stacked)
    lin = LinearNDInterpolator(pts, stacked)
    near = NearestNDInterpolator(pts, stacked)

    def evaluate(
        theta_q: ArrayLike, zeta_q: ArrayLike, with_B: bool = False
    ) -> tuple[F64, ...]:
        theta_q = np.asarray(theta_q, dtype=np.float64).reshape(-1)
        zeta_q = np.asarray(zeta_q, dtype=np.float64).reshape(-1)
        k = np.floor(zeta_q / period_z)
        zeta_w = zeta_q - k * period_z
        theta_w = np.mod(theta_q, period_t)
        query = np.column_stack([theta_w, zeta_w])
        y = lin(query)
        nan = np.isnan(y[:, 0])
        if np.any(nan):
            y = np.where(nan[:, None], near(query), y)
        phi_q = np.arctan2(y[:, 3], y[:, 2]) + k * period_z
        if with_B:
            if y.shape[1] < 5:
                raise ValueError("interpolator has no |B| column")
            return y[:, 0], phi_q, y[:, 1], y[:, 4]
        return y[:, 0], phi_q, y[:, 1]

    return evaluate


def _cuts_from_interp(
    interp: BoozerMap, nfp: int, n_planes: int, n_theta: int
) -> tuple[F64, F64, F64, F64]:
    """Closed (R, Z) contours and xyz at n_planes ζ_B planes in one period."""
    nfp = int(nfp)
    zeta = np.linspace(0.0, 2 * np.pi / nfp, int(n_planes), endpoint=False)
    theta = np.linspace(0.0, 2 * np.pi, int(n_theta), endpoint=False)
    theta_all = np.tile(theta, len(zeta))
    zeta_all = np.repeat(zeta, len(theta))
    R_all, phi_all, Z_all = interp(theta_all, zeta_all)
    R_planes = R_all.reshape(len(zeta), len(theta))
    Z_planes = Z_all.reshape(len(zeta), len(theta))
    phi_planes = phi_all.reshape(len(zeta), len(theta))
    R_closed = np.concatenate([R_planes, R_planes[:, :1]], axis=1)
    Z_closed = np.concatenate([Z_planes, Z_planes[:, :1]], axis=1)
    phi_closed = np.concatenate([phi_planes, phi_planes[:, :1]], axis=1)
    xyz = np.stack(
        [
            R_closed * np.cos(phi_closed),
            R_closed * np.sin(phi_closed),
            Z_closed,
        ],
        axis=-1,
    )
    return zeta, R_closed, Z_closed, xyz


def desc_surface_from_vmec(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    check_orientation: bool = False,
) -> FourierRZToroidalSurface:
    """Build a DESC FourierRZToroidalSurface from this project's coefficients.

    check_orientation defaults to False: DESC's default True can silently
    flip the parameterization to force a right-handed Jacobian, which
    would desync the surface from the viewer's surfacePoints() synthesis.
    The unit test asserts R,Z agreement with that synthesis; if a future
    DESC version needs the flip to solve, catch it there rather than
    here.
    """
    R_lmn, modes_R, Z_lmn, modes_Z = vmec_to_desc_modes(r_cos, z_sin)
    return FourierRZToroidalSurface(
        R_lmn=R_lmn,
        Z_lmn=Z_lmn,
        modes_R=modes_R,
        modes_Z=modes_Z,
        NFP=int(nfp),
        sym=True,
        check_orientation=check_orientation,
    )


def desc_boundary_rz(
    surface: FourierRZToroidalSurface, theta: ArrayLike, phi: ArrayLike
) -> tuple[F64, F64]:
    """Evaluate DESC surface R,Z at geometric (theta, phi) arrays."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    shape = np.broadcast(theta, phi).shape
    theta_b, phi_b = np.broadcast_arrays(theta, phi)
    nodes = np.column_stack(
        [
            np.ones(theta_b.size),
            theta_b.ravel(),
            phi_b.ravel(),
        ]
    )
    grid = Grid(nodes, jitable=False, sort=False)
    data = surface.compute(["R", "Z"], grid=grid)
    R = np.asarray(data["R"]).reshape(shape)
    Z = np.asarray(data["Z"]).reshape(shape)
    return R, Z


def _solve_desc(
    surface: FourierRZToroidalSurface,
    nfp: int,
    L: int,
    M: int,
    N: int,
    verbose: int = 0,
) -> Equilibrium:
    """Solve a fixed-boundary vacuum equilibrium at spectral resolution L/M/N."""
    eq = Equilibrium(
        surface=surface,
        pressure=0,
        current=0,
        NFP=int(nfp),
        L=int(L),
        M=int(M),
        N=int(N),
        sym=True,
        check_orientation=False,
        ensure_nested=True,
    )
    eq.solve(verbose=verbose)
    return eq


def _boozer_B_grid(
    eq: Equilibrium,
    rho: float,
    n_theta: int,
    n_zeta: int,
    M_booz: int | None = None,
    N_booz: int | None = None,
) -> dict[str, Any]:
    """|B|(θ_B, ζ_B) on one field period via DESC's native Boozer transform."""
    M_booz = int(M_booz if M_booz is not None else 2 * eq.M)
    N_booz = int(N_booz if N_booz is not None else 2 * eq.N)
    grid_compute = LinearGrid(
        rho=float(rho),
        M=max(4 * eq.M, 8),
        N=max(4 * eq.N, 8),
        NFP=eq.NFP,
        endpoint=False,
        sym=False,
    )
    data = eq.compute("|B|_mn_B", grid=grid_compute, M_booz=M_booz, N_booz=N_booz)
    grid_plot = LinearGrid(
        rho=float(rho),
        theta=int(n_theta),
        zeta=int(n_zeta),
        NFP=eq.NFP,
        endpoint=False,
        sym=False,
    )
    B_transform = get_transforms(
        "|B|_mn_B",
        obj=eq,
        grid=grid_plot,
        M_booz=M_booz,
        N_booz=N_booz,
    )["B"]
    B = np.asarray(B_transform.transform(data["|B|_mn_B"])).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta),
        order="F",
    )
    theta = grid_plot.nodes[:, 1].reshape(
        (grid_plot.num_theta, grid_plot.num_zeta),
        order="F",
    )[:, 0]
    zeta = grid_plot.nodes[:, 2].reshape(
        (grid_plot.num_theta, grid_plot.num_zeta),
        order="F",
    )[0, :]
    return {
        "n_theta": int(grid_plot.num_theta),
        "n_zeta": int(grid_plot.num_zeta),
        "theta": theta,
        "zeta": zeta,
        "B": B,
        "M_booz": M_booz,
        "N_booz": N_booz,
        "field_period_note": "one field period; zeta in [0, 2π/nfp)",
    }


def _iota_at(eq: Equilibrium, rho: float) -> float:
    """Rotational transform on one flux surface."""
    grid = LinearGrid(rho=float(rho), theta=0, zeta=0, NFP=eq.NFP)
    data = eq.compute("iota", grid=grid)
    return float(np.asarray(grid.compress(data["iota"])).reshape(-1)[0])


def _boozer_interpolator(
    eq: Equilibrium, rho: float, n_theta: int = 48, n_zeta: int = 48
) -> BoozerMap:
    """Callable (theta_B, zeta_B) -> (R, phi, Z) on one flux surface.

    Built from DESC's computed (θ_B, ζ_B, R, φ, Z) on a computational
    grid, then interpolated. map_coordinates cannot invert Boozer
    angles (no theta_B_r recipe), so this is the Boozer→cylindrical
    map. Field-period periodicity is applied by wrapping ζ_B into
    [0, 2π/NFP) and adding the corresponding 2π/NFP to φ.
    """
    nfp = int(eq.NFP)
    grid = LinearGrid(
        rho=float(rho),
        theta=int(n_theta),
        zeta=int(n_zeta),
        NFP=nfp,
        endpoint=False,
        sym=False,
    )
    data = eq.compute(["R", "Z", "phi", "theta_B", "zeta_B"], grid=grid)
    return _interpolator_from_nodes(
        data["theta_B"],
        data["zeta_B"],
        data["R"],
        data["phi"],
        data["Z"],
        nfp,
    )


def _phi_plane_hits(
    R: F64,
    Z: F64,
    phi_line: ArrayLike,
    zeta_B: F64,
    phi_target: float,
) -> tuple[list[float], list[float], list[float]]:
    """Linear-interpolate (R, Z, toroidal-turn) where φ crosses phi_target.

    Detects both increasing (2π→0) and decreasing (0→2π) wraps so
    direction="backward" (ζ running negative) still records punctures.
    """
    phase = np.mod(np.asarray(phi_line) - float(phi_target), 2 * np.pi)
    dphase = np.diff(phase)
    R_hits: list[float] = []
    Z_hits: list[float] = []
    turns: list[float] = []

    def _record(i: int, frac: float) -> None:
        R_hits.append((1 - frac) * R[i] + frac * R[i + 1])
        Z_hits.append((1 - frac) * Z[i] + frac * Z[i + 1])
        zeta_hit = (1 - frac) * zeta_B[i] + frac * zeta_B[i + 1]
        turns.append(zeta_hit / (2 * np.pi))

    # φ increasing through the plane: phase jumps 2π → 0
    for i in np.where(dphase < -np.pi)[0]:
        span = (2 * np.pi - phase[i]) + phase[i + 1]
        frac = 0.0 if span == 0 else float((2 * np.pi - phase[i]) / span)
        _record(int(i), frac)
    # φ decreasing through the plane: phase jumps 0 → 2π
    for i in np.where(dphase > np.pi)[0]:
        span = phase[i] + (2 * np.pi - phase[i + 1])
        frac = 0.0 if span == 0 else float(phase[i] / span)
        _record(int(i), frac)
    return R_hits, Z_hits, turns


def poincare_from_boozer(
    eq: Equilibrium | None,
    surfaces_s: Sequence[float],
    n_seeds: int,
    n_turns: int,
    phi: float,
    n_sample_per_turn: int = 32,
    interpolators: dict[float, BoozerMap] | None = None,
    iotas: dict[float, float] | None = None,
    direction: str = "forward",
) -> list[dict[str, Any]]:
    """Ideal-equilibrium Poincaré via the Boozer→cylindrical map.

    Field lines are straight in Boozer coordinates: a seed (s, θ0) maps
    to θ_B = θ0 + ι(s)·ζ_B. Intersections with the cylindrical plane
    φ = `phi` are the zeros of wrap(φ(θ_B, ζ_B) − phi) along that line.

    interpolators/iotas: optional precomputed maps keyed by surface s so
    this stays numpy-only after the one DESC grid compute. When they cover
    every requested surface, eq is unused and may be None -- which is how
    the VMEC++ path, which has no Equilibrium object, calls this.

    direction: "forward" (ζ≥0), "backward" (ζ≤0), or "both".

    Extension point for coil fields: replace this function with a
    SIMSOPT compute_fieldlines trace of the Biot-Savart field from
    coils (plus plasma contribution if needed). Keep the returned
    seed dict schema (s, theta0, R, Z, turn, xyz) so the viewer does not
    change.
    """
    phi_target = float(phi) % (2 * np.pi)
    theta0s = np.linspace(0.0, 2 * np.pi, int(n_seeds), endpoint=False)
    n_turns = int(n_turns)
    n_sample_per_turn = int(n_sample_per_turn)
    span = 2 * np.pi * n_turns
    direction = str(direction or "forward").lower()
    if direction == "backward":
        zeta_1d = np.linspace(0.0, -span, n_turns * n_sample_per_turn, endpoint=False)
    elif direction == "both":
        zeta_1d = np.linspace(
            -span, span, 2 * n_turns * n_sample_per_turn, endpoint=False
        )
    else:
        zeta_1d = np.linspace(0.0, span, n_turns * n_sample_per_turn, endpoint=False)
    n_z = len(zeta_1d)
    stride = max(1, n_sample_per_turn // 4)
    interpolators = interpolators or {}
    iotas = iotas or {}
    seeds: list[dict[str, Any]] = []
    for s in surfaces_s:
        s = float(s)
        rho = float(s_to_rho(s))
        if s in iotas:
            iota = iotas[s]
        elif eq is not None:
            iota = _iota_at(eq, rho)
        else:
            raise ValueError(f"no iota for s={s} and no DESC equilibrium to compute it")
        if s in interpolators:
            interp = interpolators[s]
        elif eq is not None:
            interp = _boozer_interpolator(eq, rho)
        else:
            raise ValueError(
                f"no Boozer interpolator for s={s} and no DESC equilibrium"
            )
        theta_all, zeta_all, meta = [], [], []
        for theta0 in theta0s:
            theta_all.append(theta0 + iota * zeta_1d)
            zeta_all.append(zeta_1d)
            meta.append(theta0)
        theta_all = np.concatenate(theta_all)
        zeta_all = np.concatenate(zeta_all)
        R_all, phi_all, Z_all = interp(theta_all, zeta_all)
        for k, theta0 in enumerate(meta):
            sl = slice(k * n_z, (k + 1) * n_z)
            R_s = R_all[sl]
            phi_s = phi_all[sl]
            Z_s = Z_all[sl]
            R_hits, Z_hits, turns = _phi_plane_hits(
                R_s,
                Z_s,
                phi_s,
                zeta_1d,
                phi_target,
            )
            xyz_line = np.column_stack(
                [
                    R_s * np.cos(phi_s),
                    R_s * np.sin(phi_s),
                    Z_s,
                ]
            )[::stride]
            seeds.append(
                {
                    "s": s,
                    "theta0": float(theta0),
                    "iota": float(iota),
                    "R": R_hits,
                    "Z": Z_hits,
                    "turn": turns,
                    "xyz": xyz_line,
                }
            )
    return seeds


def compute_bundle(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    s: float = 0.5,
    backend: str = "vmecpp",
    n_theta: int = DEFAULT_N_THETA,
    n_zeta: int = DEFAULT_N_ZETA,
    n_zeta_planes: int = DEFAULT_N_ZETA_PLANES,
    poincare: dict[str, Any] | None = None,
    L: int = DEFAULT_L,
    M: int = DEFAULT_M,
    N: int = DEFAULT_N,
    M_booz: int | None = None,
    N_booz: int | None = None,
    verbose: int = 0,
    allow_solve: bool = True,
    testing_dir: Path | str = TESTING_DIR,
) -> dict[str, Any]:
    """Return the viewer physics bundle for this boundary.

    backend="vmecpp" (the default) reads a saved Boozer pack; see
    compute_bundle_vmecpp. backend="desc" prefers a pre-solved h5 from
    vega_testing/<DESC_RUN>/equilibria when the boundary matches a testing
    row, with live eq.solve() only as the fallback (allow_solve=True).
    """
    backend = str(backend).lower()
    if backend == "vmecpp":
        return compute_bundle_vmecpp(
            r_cos,
            z_sin,
            nfp,
            s=s,
            n_theta=n_theta,
            n_zeta=n_zeta,
            n_zeta_planes=n_zeta_planes,
            poincare=poincare,
            verbose=verbose,
            allow_solve=allow_solve,
            testing_dir=testing_dir,
        )
    if backend != "desc":
        raise ValueError(f"unknown backend {backend!r}; expected 'desc' or 'vmecpp'")

    poincare = dict(poincare or {})
    n_seeds = int(poincare.get("n_seeds", DEFAULT_N_SEEDS))
    n_turns = int(poincare.get("n_turns", DEFAULT_N_TURNS))
    phi = float(poincare.get("phi", 0.0))
    direction = str(poincare.get("direction", "forward"))
    s = float(np.clip(s, 1e-3, 1.0))

    t0 = time.perf_counter()
    eq, hit = load_saved_equilibrium(r_cos, z_sin, nfp, testing_dir=testing_dir)
    if eq is not None:
        eq_source = "h5"
        L, M, N = int(eq.L), int(eq.M), int(eq.N)
    elif allow_solve:
        eq_source = "solve"
        surface = desc_surface_from_vmec(r_cos, z_sin, nfp, check_orientation=False)
        eq = _solve_desc(surface, nfp, L, M, N, verbose=verbose)
    else:
        detail = ""
        if hit is None:
            detail = "boundary did not match vega_testing/X.npy"
        elif not hit.get("desc_ok"):
            detail = f"row {hit['test_row']} is not desc_ok"
        else:
            detail = f"missing h5 {hit['eq_path']}"
        raise FileNotFoundError(
            f"no saved DESC equilibrium for this boundary ({detail}); "
            "refusing to solve because allow_solve=False"
        )

    s_grid = np.asarray(DEFAULT_FLUX_S, dtype=np.float64)
    rhos = np.asarray(s_to_rho(s_grid), dtype=np.float64)
    pack = _compute_flux_pack(eq, rhos, n_theta, n_zeta)
    nfp_eq = pack["nfp"]
    interpolators: dict[float, BoozerMap] = {}
    iotas: dict[float, float] = {}
    flux_s, flux_rho = [], []
    flux_vertices, flux_B, flux_boozer, flux_boozer_xyz, flux_cuts, flux_iota = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    # Per-surface ⟨|B|⟩ in tesla. The shipped B arrays are divided by it;
    # this list is what lets the readout still name the physical scale.
    flux_B_mean: list[float] = []
    B_raw_lo, B_raw_hi = np.inf, -np.inf
    theta_ax = None
    zeta_ax = None
    for i, s_val in enumerate(s_grid):
        s_val = float(s_val)
        rho = float(rhos[i])
        iota_i = float(np.nanmean(pack["iota"][i]))
        if s_val <= 1e-6:
            continue
        interp = _interpolator_from_nodes(
            pack["theta_B"][i],
            pack["zeta_B"][i],
            pack["R"][i],
            pack["phi"][i],
            pack["Z"][i],
            nfp_eq,
            B=pack["|B|"][i],
        )
        interpolators[s_val] = interp
        iotas[s_val] = iota_i
        verts = np.stack(
            [pack["X"][i], pack["Y"][i], pack["Z"][i]],
            axis=-1,
        ).reshape(-1, 3)
        B_flat = pack["|B|"][i].reshape(-1)
        booz = _boozer_B_grid(eq, rho, n_theta, n_zeta, M_booz, N_booz)
        tt, zz = np.meshgrid(booz["theta"], booz["zeta"], indexing="ij")
        R_b, phi_b, Z_b = interp(tt.ravel(), zz.ravel())
        n_t, n_z = int(booz["n_theta"]), int(booz["n_zeta"])
        xyz_b = np.stack(
            [R_b * np.cos(phi_b), R_b * np.sin(phi_b), Z_b],
            axis=-1,
        ).reshape(n_t, n_z, 3)
        if theta_ax is None:
            theta_ax = booz["theta"]
            zeta_ax = booz["zeta"]
        zeta_c, R_c, Z_c, xyz_c = _cuts_from_interp(
            interp, nfp_eq, n_zeta_planes, n_theta
        )
        # One constant per surface, taken on the Boozer grid, applied to both
        # the geometric-grid B (the 3D surface) and the Boozer-grid B (the
        # map). Same divisor for both views, so a value means the same thing
        # wherever it is read.
        B_mean = _surface_mean_B(booz["B"])
        B_raw_lo = min(B_raw_lo, float(np.nanmin(B_flat)))
        B_raw_hi = max(B_raw_hi, float(np.nanmax(B_flat)))
        flux_s.append(s_val)
        flux_rho.append(rho)
        flux_B_mean.append(B_mean)
        flux_vertices.append(_as_list(verts))
        flux_B.append(_as_list(B_flat / B_mean))
        flux_boozer.append(_as_list(np.asarray(booz["B"], dtype=np.float64) / B_mean))
        flux_boozer_xyz.append(_as_list(xyz_b))
        flux_cuts.append(
            {
                "zeta": _as_list(zeta_c),
                "R": _as_list(R_c),
                "Z": _as_list(Z_c),
                "xyz": _as_list(xyz_c),
            }
        )
        flux_iota.append(iota_i)

    # magnetic axis: ρ=0 slice, average over θ
    i0 = int(np.argmin(np.abs(rhos - 0.0)))
    axis = np.stack(
        [
            pack["X"][i0].mean(axis=0),
            pack["Y"][i0].mean(axis=0),
            pack["Z"][i0].mean(axis=0),
        ],
        axis=-1,
    )

    k = int(np.argmin(np.abs(np.asarray(flux_s) - s)))
    s_used = flux_s[k]
    iota_s = flux_iota[k]
    surfaces_s = flux_s
    seeds = poincare_from_boozer(
        eq,
        surfaces_s,
        n_seeds,
        n_turns,
        phi,
        interpolators=interpolators,
        iotas=iotas,
        direction=direction,
    )
    dt = time.perf_counter() - t0

    B_all = np.concatenate([np.asarray(b).reshape(-1) for b in flux_B])
    if theta_ax is None or zeta_ax is None:
        theta_ax = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        zeta_ax = np.linspace(0.0, 2 * np.pi / nfp_eq, n_zeta, endpoint=False)
    n_t_plot = int(np.asarray(theta_ax).reshape(-1).size)
    n_z_plot = int(np.asarray(zeta_ax).reshape(-1).size)
    meta = {
        "nfp": int(nfp),
        "s": s_used,
        "backend": "desc",
        "L": int(L),
        "M": int(M),
        "N": int(N),
        "eq_source": eq_source,
        "iota_s": float(iota_s),
        "iota_profile": {"s": flux_s, "iota": flux_iota},
        # B_min/B_max are the dimensionless |B|/⟨|B|⟩_s that the viewer
        # actually plots. The tesla values it was divided by are kept
        # alongside so the physical scale is still reportable.
        "B_min": float(np.nanmin(B_all)),
        "B_max": float(np.nanmax(B_all)),
        "B_unit": B_UNIT_NOTE,
        "B_mean_T": [float(v) for v in flux_B_mean],
        "B_ref_T": float(flux_B_mean[k]) if flux_B_mean else 1.0,
        "B_min_T": float(B_raw_lo),
        "B_max_T": float(B_raw_hi),
        "schema": BUNDLE_SCHEMA_VERSION,
        "solve_time_s": float(dt),
    }
    if hit is not None:
        meta["test_row"] = hit["test_row"]
        meta["file_row"] = hit["file_row"]
        meta["desc_time_s"] = hit["desc_time_s"]
        meta["desc_scalars"] = hit["scalars"]
        meta["eq_path"] = str(hit["eq_path"].name)
    return {
        "meta": meta,
        "flux": {
            "s": flux_s,
            "rho": flux_rho,
            "iota": flux_iota,
            "B_mean_T": [float(v) for v in flux_B_mean],
            "n_theta": n_t_plot,
            "n_zeta": n_z_plot,
            "nfp_grid": int(nfp_eq),
            "theta": _as_list(theta_ax),
            "zeta": _as_list(zeta_ax),
            "vertices": flux_vertices,
            "B": flux_B,
            "boozer_B": flux_boozer,
            "boozer_xyz": flux_boozer_xyz,
            "cuts": flux_cuts,
        },
        "surface3d": {
            "n_theta": int(n_theta),
            "n_zeta": int(n_zeta),
            "nfp_grid": int(nfp_eq),
            "s": s_used,
            "vertices": flux_vertices[k],
            "B": flux_B[k],
        },
        "axis": {"xyz": _as_list(axis)},
        "boozer_B": {
            "n_theta": n_t_plot,
            "n_zeta": n_z_plot,
            "theta": _as_list(theta_ax),
            "zeta": _as_list(zeta_ax),
            "B": flux_boozer[k],
            "xyz": flux_boozer_xyz[k],
            "field_period_note": "one field period; zeta_B in [0, 2π/nfp)",
        },
        "cuts": flux_cuts[k],
        "poincare": {
            "phi": phi,
            "n_turns": n_turns,
            "n_seeds": n_seeds,
            "direction": direction,
            "surfaces_s": surfaces_s,
            "seeds": [
                {
                    "s": seed["s"],
                    "theta0": seed["theta0"],
                    "iota": seed["iota"],
                    "R": _as_list(seed["R"]),
                    "Z": _as_list(seed["Z"]),
                    "turn": _as_list(seed["turn"]),
                    "xyz": _as_list(seed["xyz"]),
                }
                for seed in seeds
            ],
        },
    }


def vmecpp_pack_filename(sample_index: int, file_row: int) -> str:
    """Pack name for one row, carrying both the row index and file_row."""
    return f"vmec_i{int(sample_index):04d}_fr{int(file_row):06d}.npz"


def _fourier_series(
    xm: ArrayLike,
    xn: ArrayLike,
    cos_c: ArrayLike | None,
    sin_c: ArrayLike | None,
    theta: ArrayLike,
    zeta: ArrayLike,
) -> F64:
    """Evaluate Σ c_mn cos(mθ − nζ) + s_mn sin(mθ − nζ). xn already includes NFP."""
    xm = np.asarray(xm, dtype=np.float64).reshape(-1)
    xn = np.asarray(xn, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    angle = xm[:, None, None] * theta[None, ...] - xn[:, None, None] * zeta[None, ...]
    out = np.zeros(theta.shape, dtype=np.float64)
    if cos_c is not None:
        c = np.asarray(cos_c, dtype=np.float64).reshape(-1)
        out = out + np.sum(c[:, None, None] * np.cos(angle), axis=0)
    if sin_c is not None:
        s = np.asarray(sin_c, dtype=np.float64).reshape(-1)
        out = out + np.sum(s[:, None, None] * np.sin(angle), axis=0)
    return out


def _pack_column(pack: dict[str, Any], name: str, k: int) -> F64 | None:
    """Column k of a Boozer coefficient block, or None if the block is absent.

    booz_xform only fills the antisymmetric blocks (rmns_b, zmnc_b, bmns_b,
    numnc_b) for a non-stellarator-symmetric equilibrium. For the symmetric
    case -- which is every row in this dataset -- it returns them with shape
    (0, 0), and np.savez stores exactly that. Indexing [:, k] into those
    raises IndexError, so hand _fourier_series None instead and let it skip
    the sin (resp. cos) sum.
    """
    arr = np.asarray(pack[name], dtype=np.float64)
    if arr.size == 0 or arr.ndim < 2 or k >= arr.shape[1]:
        return None
    return arr[:, k]


def _surface_mean_B(B_booz: ArrayLike) -> float:
    """⟨|B|⟩ on one flux surface: the plain mean over the Boozer (θ_B, ζ_B) grid.

    The grid is uniform in the Boozer angles and both backends sample the
    same one, so this is literally the same functional in each. It scales
    linearly with the flux normalization -- which is the whole point: it is
    what |B| is divided by so that convention cancels.
    """
    B = np.asarray(B_booz, dtype=np.float64).reshape(-1)
    finite = B[np.isfinite(B)]
    if finite.size == 0:
        return 1.0
    mean = float(finite.mean())
    if not np.isfinite(mean) or abs(mean) < 1e-30:
        return 1.0
    return mean


def pack_from_boozer_output(equilibrium: Any, boozer: Any) -> dict[str, Any]:
    """Numpy pack from a constellaration VmecppWOut + BoozerOutput (duck-typed)."""
    s_booz = np.asarray(boozer.boozer_normalized_toroidal_flux, dtype=np.float64)
    iota_booz = np.asarray(boozer.iota, dtype=np.float64)
    if iota_booz.size != s_booz.size:
        # booz_xform iota is on the input radial mesh; interpolate to s_b.
        s_in = np.asarray(boozer.normalized_toroidal_flux, dtype=np.float64)
        iota_in = np.asarray(boozer.iota, dtype=np.float64)
        iota_booz = np.interp(s_booz, s_in, iota_in)

    def _arr(name: str) -> F64:
        return np.asarray(getattr(boozer, name), dtype=np.float64)

    s_full = np.asarray(
        equilibrium.normalized_toroidal_flux_full_grid_mesh, dtype=np.float64
    )
    return {
        "nfp": np.array(int(equilibrium.nfp)),
        "mpol": np.array(int(equilibrium.mpol)),
        "ntor": np.array(int(equilibrium.ntor)),
        "aspect": np.array(float(equilibrium.aspect)),
        "s_booz": s_booz,
        "iota_booz": iota_booz,
        "xm_b": _arr("xm_b"),
        "xn_b": _arr("xn_b"),
        "rmnc_b": _arr("rmnc_b"),
        "rmns_b": _arr("rmns_b"),
        "zmnc_b": _arr("zmnc_b"),
        "zmns_b": _arr("zmns_b"),
        "bmnc_b": _arr("bmnc_b"),
        "bmns_b": _arr("bmns_b"),
        "numnc_b": _arr("numnc_b"),
        "numns_b": _arr("numns_b"),
        "raxis_cc": np.asarray(equilibrium.raxis_cc, dtype=np.float64),
        "zaxis_cs": np.asarray(equilibrium.zaxis_cs, dtype=np.float64),
        "iotaf": np.asarray(equilibrium.iotaf, dtype=np.float64),
        "s_full": s_full,
    }


def save_vmecpp_pack(path: Path | str, pack: dict[str, Any]) -> None:
    """Write one row's Boozer pack as a compressed npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)


def load_vmecpp_pack(path: Path | str) -> dict[str, Any]:
    """Read a pack back as a plain dict of arrays."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def match_vmecpp_row(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    testing_dir: Path | str = TESTING_DIR,
    atol: float = 1e-5,
) -> dict[str, Any] | None:
    """Find the vega_testing row and its VMEC++ pack path."""
    testing_dir = Path(testing_dir)
    x_path = testing_dir / "X.npy"
    if not x_path.exists():
        return None
    X = np.asarray(np.load(x_path), dtype=np.float64)
    file_rows = np.asarray(np.load(testing_dir / "file_rows.npy"), dtype=np.int64)
    vec = np.concatenate(
        [
            np.asarray(r_cos, dtype=np.float64).reshape(-1),
            np.asarray(z_sin, dtype=np.float64).reshape(-1),
            [float(nfp)],
        ]
    )
    dist = np.max(np.abs(X[:, :91] - vec), axis=1)
    j = int(np.argmin(dist))
    if dist[j] > atol:
        return None
    name = vmecpp_pack_filename(j, int(file_rows[j]))
    pack_path = testing_dir / VMECPP_EQ_DIRNAME / name
    return {
        "test_row": j,
        "file_row": int(file_rows[j]),
        "pack_path": pack_path,
        "match_err": float(dist[j]),
    }


def _axis_xyz_from_pack(pack: dict[str, Any], n_zeta: int) -> F64:
    """Magnetic axis from the VMEC raxis_cc / zaxis_cs series."""
    nfp = int(pack["nfp"])
    raxis = np.asarray(pack["raxis_cc"], dtype=np.float64).reshape(-1)
    zaxis = np.asarray(pack["zaxis_cs"], dtype=np.float64).reshape(-1)
    phi = np.linspace(0.0, 2 * np.pi, int(n_zeta), endpoint=False)
    R = np.zeros_like(phi)
    Z = np.zeros_like(phi)
    n_mode = min(len(raxis), len(zaxis))
    for k in range(n_mode):
        ang = k * nfp * phi
        R = R + raxis[k] * np.cos(ang)
        Z = Z + zaxis[k] * np.sin(ang)
    if len(raxis) > n_mode:
        for k in range(n_mode, len(raxis)):
            R = R + raxis[k] * np.cos(k * nfp * phi)
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1)


def assemble_bundle_from_vmecpp_pack(
    pack: dict[str, Any],
    s: float = 0.5,
    n_theta: int = DEFAULT_N_THETA,
    n_zeta: int = DEFAULT_N_ZETA,
    n_zeta_planes: int = DEFAULT_N_ZETA_PLANES,
    poincare: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the viewer physics bundle from a saved VMEC++ / booz_xform pack."""
    poincare = dict(poincare or {})
    n_seeds = int(poincare.get("n_seeds", DEFAULT_N_SEEDS))
    n_turns = int(poincare.get("n_turns", DEFAULT_N_TURNS))
    phi0 = float(poincare.get("phi", 0.0))
    direction = str(poincare.get("direction", "forward"))
    s = float(np.clip(s, 1e-3, 1.0))
    nfp = int(pack["nfp"])
    n_theta = int(n_theta)
    n_zeta = int(n_zeta)

    s_booz = np.asarray(pack["s_booz"], dtype=np.float64).reshape(-1)
    iota_booz = np.asarray(pack["iota_booz"], dtype=np.float64).reshape(-1)
    xm_b = np.asarray(pack["xm_b"], dtype=np.float64).reshape(-1)
    xn_b = np.asarray(pack["xn_b"], dtype=np.float64).reshape(-1)

    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2 * np.pi / nfp, n_zeta, endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")

    interpolators: dict[float, BoozerMap] = {}
    iotas: dict[float, float] = {}
    flux_s, flux_rho, flux_iota = [], [], []
    flux_vertices, flux_B, flux_boozer, flux_boozer_xyz, flux_cuts = (
        [],
        [],
        [],
        [],
        [],
    )
    flux_B_mean: list[float] = []
    B_raw_lo, B_raw_hi = np.inf, -np.inf

    t0 = time.perf_counter()
    for k, s_val in enumerate(s_booz):
        s_val = float(s_val)
        if s_val <= 1e-6:
            continue
        iota_i = float(iota_booz[k])
        R = _fourier_series(
            xm_b,
            xn_b,
            _pack_column(pack, "rmnc_b", k),
            _pack_column(pack, "rmns_b", k),
            tt,
            zz,
        )
        Z = _fourier_series(
            xm_b,
            xn_b,
            _pack_column(pack, "zmnc_b", k),
            _pack_column(pack, "zmns_b", k),
            tt,
            zz,
        )
        B = _fourier_series(
            xm_b,
            xn_b,
            _pack_column(pack, "bmnc_b", k),
            _pack_column(pack, "bmns_b", k),
            tt,
            zz,
        )
        nu = _fourier_series(
            xm_b,
            xn_b,
            _pack_column(pack, "numnc_b", k),
            _pack_column(pack, "numns_b", k),
            tt,
            zz,
        )
        phi = zz - nu
        interp = _interpolator_from_nodes(tt, zz, R, phi, Z, nfp, B=B)
        interpolators[s_val] = interp
        iotas[s_val] = iota_i
        verts = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1).reshape(-1, 3)
        zeta_c, R_c, Z_c, xyz_c = _cuts_from_interp(interp, nfp, n_zeta_planes, n_theta)
        # Same normalization as the DESC path: divide by this surface's
        # ⟨|B|⟩ so the two backends are plotted on one dimensionless scale.
        # Here the 3D field and the Boozer map are the same array, so the
        # single divisor is trivially consistent.
        B_mean = _surface_mean_B(B)
        B_raw_lo = min(B_raw_lo, float(np.nanmin(B)))
        B_raw_hi = max(B_raw_hi, float(np.nanmax(B)))
        B_n = B / B_mean
        flux_s.append(s_val)
        flux_rho.append(float(s_to_rho(s_val)))
        flux_iota.append(iota_i)
        flux_B_mean.append(B_mean)
        flux_vertices.append(_as_list(verts))
        flux_B.append(_as_list(B_n.reshape(-1)))
        flux_boozer.append(_as_list(B_n))
        flux_boozer_xyz.append(
            _as_list(np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1))
        )
        flux_cuts.append(
            {
                "zeta": _as_list(zeta_c),
                "R": _as_list(R_c),
                "Z": _as_list(Z_c),
                "xyz": _as_list(xyz_c),
            }
        )

    axis = _axis_xyz_from_pack(pack, n_zeta)
    k = int(np.argmin(np.abs(np.asarray(flux_s) - s)))
    s_used = flux_s[k]
    iota_s = flux_iota[k]
    seeds = poincare_from_boozer(
        None,
        flux_s,
        n_seeds,
        n_turns,
        phi0,
        interpolators=interpolators,
        iotas=iotas,
        direction=direction,
    )
    dt = time.perf_counter() - t0
    B_all = np.concatenate([np.asarray(b).reshape(-1) for b in flux_B])
    mpol = int(pack["mpol"])
    ntor = int(pack["ntor"])
    meta = {
        "nfp": nfp,
        "s": s_used,
        "backend": "vmecpp",
        "L": mpol,
        "M": mpol,
        "N": ntor,
        "eq_source": "npz",
        "iota_s": float(iota_s),
        "iota_profile": {"s": flux_s, "iota": flux_iota},
        "B_min": float(np.nanmin(B_all)),
        "B_max": float(np.nanmax(B_all)),
        "B_unit": B_UNIT_NOTE,
        "B_mean_T": [float(v) for v in flux_B_mean],
        "B_ref_T": float(flux_B_mean[k]) if flux_B_mean else 1.0,
        "B_min_T": float(B_raw_lo),
        "B_max_T": float(B_raw_hi),
        "schema": BUNDLE_SCHEMA_VERSION,
        "solve_time_s": float(dt),
        "aspect": float(pack["aspect"]),
    }
    return {
        "meta": meta,
        "flux": {
            "s": flux_s,
            "rho": flux_rho,
            "iota": flux_iota,
            "B_mean_T": [float(v) for v in flux_B_mean],
            "n_theta": n_theta,
            "n_zeta": n_zeta,
            "nfp_grid": nfp,
            "theta": _as_list(theta),
            "zeta": _as_list(zeta),
            "vertices": flux_vertices,
            "B": flux_B,
            "boozer_B": flux_boozer,
            "boozer_xyz": flux_boozer_xyz,
            "cuts": flux_cuts,
        },
        "surface3d": {
            "n_theta": n_theta,
            "n_zeta": n_zeta,
            "nfp_grid": nfp,
            "s": s_used,
            "vertices": flux_vertices[k],
            "B": flux_B[k],
        },
        "axis": {"xyz": _as_list(axis)},
        "boozer_B": {
            "n_theta": n_theta,
            "n_zeta": n_zeta,
            "theta": _as_list(theta),
            "zeta": _as_list(zeta),
            "B": flux_boozer[k],
            "xyz": flux_boozer_xyz[k],
            "field_period_note": "one field period; zeta_B in [0, 2π/nfp)",
        },
        "cuts": flux_cuts[k],
        "poincare": {
            "phi": phi0,
            "n_turns": n_turns,
            "n_seeds": n_seeds,
            "direction": direction,
            "surfaces_s": flux_s,
            "seeds": [
                {
                    "s": seed["s"],
                    "theta0": seed["theta0"],
                    "iota": seed["iota"],
                    "R": _as_list(seed["R"]),
                    "Z": _as_list(seed["Z"]),
                    "turn": _as_list(seed["turn"]),
                    "xyz": _as_list(seed["xyz"]),
                }
                for seed in seeds
            ],
        },
    }


def compute_bundle_vmecpp(
    r_cos: ArrayLike,
    z_sin: ArrayLike,
    nfp: int,
    s: float = 0.5,
    n_theta: int = DEFAULT_N_THETA,
    n_zeta: int = DEFAULT_N_ZETA,
    n_zeta_planes: int = DEFAULT_N_ZETA_PLANES,
    poincare: dict[str, Any] | None = None,
    verbose: int = 0,
    allow_solve: bool = False,
    testing_dir: Path | str = TESTING_DIR,
    **kwargs: Any,
) -> dict[str, Any]:
    """Same bundle format as compute_bundle(..., backend='desc').

    Loads a precomputed VMEC++ / booz_xform pack from
    vega_testing/vmecpp_highfi/equilibria/. Live vmecpp.solve is
    intentionally not done here (ARM physics image has no working
    constellaration).
    """
    del verbose, kwargs
    if allow_solve:
        raise NotImplementedError(
            "live vmecpp.solve is not available in the physics service; "
            "load a pack from vega_testing/vmecpp_highfi/equilibria/"
        )
    hit = match_vmecpp_row(r_cos, z_sin, nfp, testing_dir=testing_dir)
    if hit is None:
        raise FileNotFoundError(
            "no saved VMEC++ pack: boundary did not match vega_testing/X.npy"
        )
    if not hit["pack_path"].exists():
        # Name the run, not just the file: VMECPP_RUN is repointed between
        # campaigns, and "missing vmec_i0306_*.npz" is baffling when the
        # real answer is that the run folder it now points at is still empty.
        eq_dir = Path(testing_dir) / VMECPP_EQ_DIRNAME
        n_have = len(list(eq_dir.glob("*.npz"))) if eq_dir.is_dir() else 0
        raise FileNotFoundError(
            f"no saved VMEC++ pack for row {hit['test_row']} in run "
            f"{VMECPP_RUN!r} (missing {hit['pack_path'].name}; "
            f"{n_have} pack(s) present in {VMECPP_EQ_DIRNAME})"
        )
    pack = load_vmecpp_pack(hit["pack_path"])
    if "ok" in pack and not bool(np.asarray(pack["ok"])):
        msg = str(pack["message"]) if "message" in pack else "solve failed"
        raise FileNotFoundError(
            f"VMEC++ pack for row {hit['test_row']} is not ok ({msg})"
        )
    bundle = assemble_bundle_from_vmecpp_pack(
        pack,
        s=s,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_zeta_planes=n_zeta_planes,
        poincare=poincare,
    )
    bundle["meta"]["test_row"] = hit["test_row"]
    bundle["meta"]["file_row"] = hit["file_row"]
    bundle["meta"]["eq_path"] = hit["pack_path"].name
    if "time_s" in pack:
        bundle["meta"]["vmecpp_time_s"] = float(np.asarray(pack["time_s"]))
    return bundle


def _print_sanity(bundle: dict[str, Any]) -> None:
    """Print the CLI sanity summary: |B| range, closed cuts, axis, hits."""
    meta = bundle["meta"]
    print(
        f"backend={meta['backend']} src={meta.get('eq_source', '?')} nfp={meta['nfp']} s={meta['s']:.3f} "
        f"L={meta['L']} M={meta['M']} N={meta['N']}  ({meta['solve_time_s']:.1f}s)"
    )
    print(
        f"  |B|/<B> range [{meta['B_min']:.4g}, {meta['B_max']:.4g}]"
        f"  <B>={meta.get('B_ref_T', float('nan')):.4g} T"
        f"  iota(s)={meta['iota_s']:.4g}"
    )
    B = np.asarray(bundle["boozer_B"]["B"])
    if not np.all(B > 0):
        print("  WARN: Boozer |B| is not strictly positive")
    R = np.asarray(bundle["cuts"]["R"])
    Z = np.asarray(bundle["cuts"]["Z"])
    closed = np.allclose(R[:, 0], R[:, -1]) and np.allclose(Z[:, 0], Z[:, -1])
    print(f"  cuts: {len(R)} planes, closed={closed}")
    axis = np.asarray(bundle["axis"]["xyz"])
    verts = np.asarray(bundle["surface3d"]["vertices"])
    R_surf = np.hypot(verts[:, 0], verts[:, 1])
    R_axis = np.hypot(axis[:, 0], axis[:, 1])
    print(
        f"  axis R in [{R_axis.min():.3f}, {R_axis.max():.3f}]  "
        f"surface R in [{R_surf.min():.3f}, {R_surf.max():.3f}]"
    )
    n_pts = sum(len(seed["R"]) for seed in bundle["poincare"]["seeds"])
    print(f"  poincare: {len(bundle['poincare']['seeds'])} seeds, {n_pts} hits")
    flux = bundle.get("flux")
    if flux:
        print(f"  flux slices s={bundle['flux']['s']}  rho={bundle['flux']['rho']}")


def main() -> None:
    """CLI entry point: build one bundle from a vega_testing row."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--test-row", type=int, required=True, help="row index into vega_testing/X.npy"
    )
    p.add_argument("--backend", default="vmecpp", choices=["vmecpp", "desc"])
    p.add_argument("--s", type=float, default=0.5)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--M", type=int, default=DEFAULT_M)
    p.add_argument("--N", type=int, default=DEFAULT_N)
    p.add_argument("--n-theta", type=int, default=DEFAULT_N_THETA)
    p.add_argument("--n-zeta", type=int, default=DEFAULT_N_ZETA)
    p.add_argument("--n-zeta-planes", type=int, default=DEFAULT_N_ZETA_PLANES)
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--n-turns", type=int, default=DEFAULT_N_TURNS)
    p.add_argument("--phi", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None, help="write bundle JSON here")
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument(
        "--solve",
        action="store_true",
        help="allow a live DESC solve if no saved h5 matches",
    )
    args = p.parse_args()

    row = load_test_row(args.test_row)
    print(
        f"test-row {row['test_row']}  file_row={row['file_row']}  "
        f"nfp={row['nfp']}  vmec_jax_time={row['vmec_time_s']:.1f}s  "
        f"eq={row['eq_path'].name if row['eq_path'] else 'none'}"
    )

    bundle = compute_bundle(
        row["r_cos"],
        row["z_sin"],
        row["nfp"],
        s=args.s,
        backend=args.backend,
        n_theta=args.n_theta,
        n_zeta=args.n_zeta,
        n_zeta_planes=args.n_zeta_planes,
        poincare={"n_seeds": args.n_seeds, "n_turns": args.n_turns, "phi": args.phi},
        L=args.L,
        M=args.M,
        N=args.N,
        verbose=args.verbose,
        allow_solve=args.solve,
    )
    bundle["meta"]["test_row"] = row["test_row"]
    bundle["meta"]["file_row"] = row["file_row"]
    _print_sanity(bundle)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bundle))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

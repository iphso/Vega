"""Fourier-convention mapping into DESC -- the transpose/handedness class
of bug this repo has already hit once (see EXPERIMENT_LOG and the
vmec_jax_oracle.py docstring). These tests compare DESC's boundary R,Z
against a direct VMEC2000-lineage synthesis at matched geometric angles,
and against the viewer's surfacePoints() formula on a vega_testing row.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from physics_bundle import (
    TESTING_DIR,
    VMECPP_EQ_DIRNAME,
    _pack_column,
    _surface_mean_B,
    compute_bundle_vmecpp,
    desc_boundary_rz,
    desc_surface_from_vmec,
    load_saved_equilibrium,
    load_test_row,
    match_saved_row,
    synthesize_boundary_rz,
    unpack_row,
)

# TESTING_DIR comes from physics_bundle (VEGA_TESTING_DIR, else the container
# path) rather than being restated here: most tests below call helpers that
# use that default, so a second copy could disagree with what is under test.
#
# The dataset is ~112MB and lives outside the repo, so a plain `pytest` on a
# checkout without it should say "no data" once, not raise FileNotFoundError
# out of every test body. This is module-wide, so the two pure-function tests
# at the bottom (_pack_column, _surface_mean_B) ride along and skip too --
# worth splitting them into their own module if they ever need to run
# standalone.
pytestmark = pytest.mark.skipif(
    not (TESTING_DIR / "X.npy").exists(),
    reason=f"no vega_testing dataset at {TESTING_DIR} (set VEGA_TESTING_DIR)",
)

# Fastest nfp=3 row in vega_testing (times.npy); nfp=3 is the bulk of the set.
TEST_ROW = 306
TOL = 1e-6

# VMECPP_RUN is repointed between campaigns (lowfi -> highfi), and a run
# that has been declared but not yet delivered has an empty equilibria/.
# Skip the pack-backed tests in that window rather than reporting a failure
# for data that is still being computed -- but only when the directory is
# genuinely empty, so a broken assembler still fails loudly.
_VMECPP_EQ_DIR = TESTING_DIR / VMECPP_EQ_DIRNAME
_HAVE_PACKS = _VMECPP_EQ_DIR.is_dir() and any(_VMECPP_EQ_DIR.glob("*.npz"))
needs_packs = pytest.mark.skipif(
    not _HAVE_PACKS,
    reason=f"no VMEC++ packs in {VMECPP_EQ_DIRNAME} (campaign not landed yet)",
)


@pytest.fixture(scope="module")
def row() -> dict[str, Any]:
    return load_test_row(TEST_ROW, testing_dir=TESTING_DIR)


def test_unpack_matches_feature_layout() -> None:
    X = np.load(TESTING_DIR / "X.npy")
    r_cos, z_sin, nfp, is_sym = unpack_row(X[TEST_ROW])
    assert r_cos.shape == (5, 9)
    assert z_sin.shape == (5, 9)
    assert nfp == int(X[TEST_ROW, 90])
    # col 91 is is_stellarator_symmetric; always 1.0 in this set
    assert is_sym is True
    assert np.allclose(X[TEST_ROW, 91], 1.0)


def test_numpy_synthesis_is_surface_points_formula(row: dict[str, Any]) -> None:
    """The JS viewer loops m outer, n_idx inner, n = n_idx-4, angle =
    m*theta - nfp*n*phi. Keep the Python port identical, not a rewrite.
    """
    r_cos, z_sin, nfp = row["r_cos"], row["z_sin"], row["nfp"]
    theta = np.array([0.0, 0.7, 2.1, 5.9])
    phi = np.array([0.0, 0.4, 1.8, 4.2])
    R, Z = synthesize_boundary_rz(r_cos, z_sin, nfp, theta, phi)

    R_js = np.zeros_like(theta)
    Z_js = np.zeros_like(theta)
    r_flat = r_cos.reshape(-1)
    z_flat = z_sin.reshape(-1)
    for m in range(5):
        for n_idx in range(9):
            n = n_idx - 4
            angle = m * theta - nfp * n * phi
            R_js += r_flat[m * 9 + n_idx] * np.cos(angle)
            Z_js += z_flat[m * 9 + n_idx] * np.sin(angle)
    assert np.max(np.abs(R - R_js)) < 1e-15
    assert np.max(np.abs(Z - Z_js)) < 1e-15


def test_desc_boundary_matches_numpy_synthesis(row: dict[str, Any]) -> None:
    r_cos, z_sin, nfp = row["r_cos"], row["z_sin"], row["nfp"]
    surface = desc_surface_from_vmec(r_cos, z_sin, nfp, check_orientation=False)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    R_np, Z_np = synthesize_boundary_rz(r_cos, z_sin, nfp, tt, pp)
    R_desc, Z_desc = desc_boundary_rz(surface, tt, pp)
    assert np.max(np.abs(R_desc - R_np)) < TOL
    assert np.max(np.abs(Z_desc - Z_np)) < TOL


def test_golden_desc_vs_viewer_surface_points(row: dict[str, Any]) -> None:
    """Boundary only, matched geometric (θ, φ). 1e-6 is the agreed budget."""
    r_cos, z_sin, nfp = row["r_cos"], row["z_sin"], row["nfp"]
    surface = desc_surface_from_vmec(r_cos, z_sin, nfp, check_orientation=False)
    n_theta, n_phi = 8, 12
    theta = (np.arange(n_theta) / n_theta) * 2 * np.pi
    phi = (np.arange(n_phi) / n_phi) * 2 * np.pi
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    R_view, Z_view = synthesize_boundary_rz(r_cos, z_sin, nfp, tt, pp)
    R_desc, Z_desc = desc_boundary_rz(surface, tt, pp)
    err = max(np.max(np.abs(R_desc - R_view)), np.max(np.abs(Z_desc - Z_view)))
    assert err < TOL, f"DESC vs surfacePoints mismatch {err} (tol {TOL})"


def test_check_orientation_true_does_not_silently_flip(row: dict[str, Any]) -> None:
    """If DESC's orientation check flips the surface, the viewer-aligned
    mapping is wrong and we need to know before trusting any Boozer plot.
    """
    r_cos, z_sin, nfp = row["r_cos"], row["z_sin"], row["nfp"]
    surface = desc_surface_from_vmec(r_cos, z_sin, nfp, check_orientation=True)
    theta = np.array([0.3, 1.2, 4.0])
    phi = np.array([0.1, 2.0, 5.0])
    R_np, Z_np = synthesize_boundary_rz(r_cos, z_sin, nfp, theta, phi)
    R_desc, Z_desc = desc_boundary_rz(surface, theta, phi)
    err = max(np.max(np.abs(R_desc - R_np)), np.max(np.abs(Z_desc - Z_np)))
    assert err < TOL, (
        f"check_orientation=True flipped/changed the surface (err={err}); "
        "keep it False in desc_surface_from_vmec"
    )


def test_match_saved_row_finds_h5(row: dict[str, Any]) -> None:
    hit = match_saved_row(row["r_cos"], row["z_sin"], row["nfp"])
    assert hit is not None
    assert hit["test_row"] == TEST_ROW
    assert hit["file_row"] == row["file_row"]
    assert hit["desc_ok"] is True
    assert hit["eq_path"].exists()
    assert hit["eq_path"].name == f"eq_i{TEST_ROW:04d}_fr{row['file_row']:06d}.h5"


def test_load_saved_equilibrium_skips_solve(row: dict[str, Any]) -> None:
    eq, hit = load_saved_equilibrium(row["r_cos"], row["z_sin"], row["nfp"])
    assert hit["test_row"] == TEST_ROW
    assert eq is not None
    assert int(eq.NFP) == row["nfp"]


# The two tests below validate the *saved artifacts*, not the mapping code.
# Campaign history: first L8 hardcoding NFP=3 and writing VMEC coeffs into
# DESC's basis directly; second L8 fixed NFP but still wrong boundaries;
# third L8 (job 62951210) builds via desc_surface_from_vmec.


def test_saved_desc_equilibria_have_the_rows_nfp() -> None:
    """Every saved h5 must carry its own row's field-period count."""
    X = np.load(TESTING_DIR / "X.npy")
    # Rows spanning several nfp values -- an nfp=3-only sample cannot see
    # a hardcoded NFP=3.
    bad = []
    for i in _sample_rows(X, n=8):
        r_cos, z_sin, nfp, _ = unpack_row(X[i])
        eq, _ = load_saved_equilibrium(r_cos, z_sin, nfp, testing_dir=TESTING_DIR)
        if eq is not None and int(eq.NFP) != int(nfp):
            bad.append((i, int(nfp), int(eq.NFP)))
    assert not bad, f"(row, nfp, saved NFP): {bad}"


def test_saved_desc_boundary_matches_the_requested_boundary() -> None:
    """The solved h5 must be the boundary we asked for, not another shape.

    Tolerance is deliberately loose (1e-3 m against a ~0.13 m minor radius):
    this is looking for a wrong surface, not for spectral round-off.
    """
    X = np.load(TESTING_DIR / "X.npy")
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    worst = []
    for i in _sample_rows(X, n=8):
        r_cos, z_sin, nfp, _ = unpack_row(X[i])
        eq, _ = load_saved_equilibrium(r_cos, z_sin, nfp, testing_dir=TESTING_DIR)
        if eq is None:
            continue
        R_true, Z_true = synthesize_boundary_rz(r_cos, z_sin, nfp, tt, pp)
        R_h5, Z_h5 = desc_boundary_rz(eq.surface, tt, pp)
        err = max(np.max(np.abs(R_h5 - R_true)), np.max(np.abs(Z_h5 - Z_true)))
        worst.append((i, int(nfp), float(err)))
    over = [w for w in worst if w[2] > 1e-3]
    assert not over, f"saved boundary != requested boundary (row, nfp, err_m): {over}"


def _sample_rows(X: np.ndarray, n: int = 8) -> list[int]:
    """Row indices covering as many distinct nfp values as the set holds."""
    nfps = X[:, 90].astype(int)
    picked: list[int] = []
    for value in sorted(set(nfps.tolist())):
        picked.extend(np.flatnonzero(nfps == value)[:2].tolist())
    return picked[:n] or [0]


@needs_packs
def test_vmecpp_backend_loads_saved_pack(row: dict[str, Any]) -> None:
    """The pack path has to actually build a bundle, not just exist.

    Regression: booz_xform returns the antisymmetric blocks (rmns_b, zmnc_b,
    bmns_b, numnc_b) with shape (0, 0) for a stellarator-symmetric run, which
    is every row here. Indexing [:, k] into those raised IndexError and took
    the whole backend down -- _pack_column returns None instead.
    """
    bundle = compute_bundle_vmecpp(
        row["r_cos"],
        row["z_sin"],
        row["nfp"],
        allow_solve=False,
        testing_dir=TESTING_DIR,
    )
    meta = bundle["meta"]
    assert meta["backend"] == "vmecpp"
    assert meta["test_row"] == TEST_ROW
    assert meta["nfp"] == row["nfp"]
    # A real equilibrium, not a degenerate one.
    assert np.isfinite(meta["iota_s"]) and abs(meta["iota_s"]) > 1e-4
    assert len(bundle["flux"]["s"]) == 4


def test_vmecpp_pack_column_handles_empty_antisymmetric_blocks() -> None:
    """(0, 0) blocks must read as absent, not as an index error."""
    pack = {
        "bmnc_b": np.ones((7, 4)),
        "bmns_b": np.zeros((0, 0)),
    }
    assert _pack_column(pack, "bmns_b", 0) is None
    col = _pack_column(pack, "bmnc_b", 2)
    assert col is not None and col.shape == (7,)
    # Out-of-range column is absent too, not an IndexError.
    assert _pack_column(pack, "bmnc_b", 9) is None


def test_vmecpp_backend_missing_pack_raises() -> None:
    """A boundary that is not a testing row still fails loudly."""
    rng = np.random.default_rng(0)
    with pytest.raises(FileNotFoundError, match="VMEC\\+\\+|vmecpp"):
        compute_bundle_vmecpp(
            rng.normal(size=(5, 9)),
            rng.normal(size=(5, 9)),
            3,
            allow_solve=False,
            testing_dir=TESTING_DIR,
        )


def test_surface_mean_B_helper() -> None:
    assert _surface_mean_B(np.full((4, 6), 2.5)) == pytest.approx(2.5)
    # Degenerate input must not produce a divide-by-zero divisor.
    assert _surface_mean_B(np.zeros((3, 3))) == 1.0
    assert _surface_mean_B(np.full((3, 3), np.nan)) == 1.0


@needs_packs
def test_vmecpp_B_is_normalized_to_surface_mean(row: dict[str, Any]) -> None:
    """|B| ships dimensionless: every surface averages to 1 by construction.

    This is what makes the two backends comparable -- DESC solves at Ψ=1 Wb
    and VMEC++ at the boundary's phiedge, so raw tesla differ by a constant
    that this division cancels.
    """
    bundle = compute_bundle_vmecpp(
        row["r_cos"],
        row["z_sin"],
        row["nfp"],
        allow_solve=False,
        testing_dir=TESTING_DIR,
    )
    flux = bundle["flux"]
    for k, s_val in enumerate(flux["s"]):
        B = np.asarray(flux["boozer_B"][k], dtype=np.float64)
        # _as_list rounds to 6 decimals for JSON, so the mean lands on 1
        # to within that quantization, not to machine precision.
        assert np.mean(B) == pytest.approx(1.0, abs=1e-5), f"s={s_val}"
    # The tesla scale that was divided out is still reported.
    means = bundle["meta"]["B_mean_T"]
    assert len(means) == len(flux["s"])
    assert all(np.isfinite(m) and m > 0 for m in means)
    assert bundle["meta"]["B_unit"].startswith("|B| / <|B|>")

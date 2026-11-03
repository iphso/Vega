"""Weight windows from an earlier coil_ref.py run's (R, Z) mesh flux (MAGIC: lower bound
proportional to the forward flux, per energy bin, 0.5 at the flux peak), with the guards that
OpenMC's built-in MAGIC iteration lacked on this model (activation/arc_activation.py notes it
ran away, tracks growing ~1.5x per secondary generation):
  - only voxels whose flux has rel. s.d. < MAX_RE are trusted; the rest take the lowest
    trusted bound within the coil box, or are left without a window (bound -1) when outside it
  - the bound's log is smoothed (3x3 box) and no voxel may sit more than RATIO_MAX below a
    neighbour (iterated to a fixed point), so no single step splits a particle more than ~10x
  - a floor at FLOOR

Run: python ww_magic.py /runs/<tag>/result.npz /runs/<out>.npz   (inside the openmc container)
"""
import sys
import numpy as np

MAX_RE, RATIO_MAX, FLOOR, UPPER = 0.3, 4.0, 1e-12, 5.0


def build(src, dst, max_re=MAX_RE, focus=None, rest_floor=1e-2, slow_lb=None, face=None, lam=10.0):
    """focus = (rmax, zmax[, rmin]): MAGIC bounds only inside rmin <= R < rmax, |Z| < zmax (the inboard leg);
    elsewhere the bound may not fall below rest_floor x its peak (neutrons there are rouletted
    rather than split: they cannot reach the TF face)."""
    z = np.load(src)
    f, s = z["ww_flux"], z["ww_flux_sd"]            # [z, r, e]
    rg, zg = z["ww_r"], z["ww_z"]
    nz, nr, ne = f.shape
    lb = np.full(f.shape, np.nan)
    for e in range(ne):
        fe, se = f[:, :, e], s[:, :, e]
        ok = (fe > 0) & (se < max_re * fe)
        lb[:, :, e] = np.where(ok, 0.5 * fe / fe[ok].max(), np.nan)
    # smooth log bound over trusted voxels (3x3), then fill untrusted voxels from neighbours
    L = np.log(lb)
    for _ in range(200):
        pad = np.pad(L, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan)
        nb = np.stack([pad[1 + dz:1 + dz + nz, 1 + dr:1 + dr + nr] for dz in (-1, 0, 1) for dr in (-1, 0, 1)])
        fill = np.isnan(L) & np.any(~np.isnan(nb), axis=0)
        if not fill.any():
            break
        L = np.where(fill, np.nanmin(nb, axis=0), L)
    pad = np.pad(L, ((1, 1), (1, 1), (0, 0)), mode="edge")
    L = np.nanmean(np.stack([pad[1 + dz:1 + dz + nz, 1 + dr:1 + dr + nr] for dz in (-1, 0, 1) for dr in (-1, 0, 1)]), axis=0)
    # neighbour ratio limit: L >= max(neighbour) - log(RATIO_MAX)
    lr = np.log(RATIO_MAX)
    for _ in range(1000):
        pad = np.pad(L, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nbmax = np.max(np.stack([pad[1 + dz:1 + dz + nz, 1 + dr:1 + dr + nr] for dz in (-1, 0, 1) for dr in (-1, 0, 1)]), axis=0)
        L2 = np.maximum(L, nbmax - lr)
        if np.allclose(L2, L):
            break
        L = L2
    if face is not None:
        # importance-shaped (CADIS-like) instead of a hard focus floor: in front of the face
        # (R >= face_rmin, |Z| <= halfh) the MAGIC bound; elsewhere the bound at the nearest point
        # of that column times exp(d / lam), d the (R, Z) distance to it -- rising smoothly away
        # from the target, never below the MAGIC bound there. A hard floor next to the face
        # (focus) roulettes deep-coil neutrons and promotes the survivors to ~1e4 x the local
        # window; the few that backscatter into the face are outliers that bias small samples low.
        frmin, fhalf, frmax = face if len(face) > 2 else (*face, np.inf)
        rc, zc = 0.5 * (rg[1:] + rg[:-1]), 0.5 * (zg[1:] + zg[:-1])
        Rr, Zz = np.meshgrid(rc, zc)                           # [z, r]
        ir = np.clip(np.searchsorted(rc, np.clip(Rr, frmin, frmax)), 0, nr - 1)
        jz = np.clip(np.searchsorted(zc, np.clip(Zz, -fhalf, fhalf)), 0, nz - 1)
        d = np.hypot(np.maximum(frmin - Rr, 0.0) + np.maximum(Rr - frmax, 0.0), np.maximum(np.abs(Zz) - fhalf, 0.0))
        for e in range(ne):
            Le = L[:, :, e].copy()
            L[:, :, e] = np.minimum(np.maximum(Le, Le[jz, ir] + d / lam), np.log(0.5))
    if focus is not None:
        rc, zc = 0.5 * (rg[1:] + rg[:-1]), 0.5 * (zg[1:] + zg[:-1])
        inside = (rc[None, :] < focus[0]) & (np.abs(zc)[:, None] < focus[1])
        if len(focus) > 2:      # (rmax, zmax, rmin): no splitting behind R = rmin either
            inside &= rc[None, :] >= focus[2]
        L = np.where(inside[:, :, None], L, np.maximum(L, np.log(0.5 * rest_floor)))
    lb = np.maximum(np.exp(L), FLOOR)
    lb = np.where(np.isnan(lb), -1.0, lb)
    if slow_lb is not None:
        # fast-targeted windows: below 0.1 MeV one constant bound, so slow neutrons (which cannot
        # score a > 0.1 MeV tally) are rouletted out of the shield and coil instead of followed
        lb[:, :, 0] = slow_lb
    lower = np.transpose(lb, (1, 0, 2))[:, None, :, :]       # (r, phi, z, e)
    eb = z["ww_e"] if "ww_e" in z.files else np.array([1e-5, 1e5, 2e7])
    np.savez(dst, lower=lower, r_grid=rg, z_grid=zg, energy_bounds=eb, ratio=UPPER)
    print("lower bound range per energy bin:", [(float(lb[:, :, e].min()), float(lb[:, :, e].max())) for e in range(ne)])


if __name__ == "__main__":
    # ww_magic.py src dst [max_re] [--focus=rmax,zmax[,rmin]] [--rest-floor=f] [--slow-lb=x] [--face=rmin,halfh] [--lam=cm]
    fl = dict((x[2:].split("=") + [""])[:2] for x in sys.argv[1:] if x.startswith("--"))
    a = [x for x in sys.argv if not x.startswith("--")]
    focus = tuple(float(v) for v in fl["focus"].split(",")) if "focus" in fl else None
    build(a[1], a[2], float(a[3]) if len(a) > 3 else MAX_RE, focus, float(fl.get("rest-floor") or 1e-2),
          float(fl["slow-lb"]) if "slow-lb" in fl else None,
          tuple(float(v) for v in fl["face"].split(",")) if "face" in fl else None, float(fl.get("lam") or 10.0))

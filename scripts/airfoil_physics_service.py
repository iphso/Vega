"""FastAPI wrapper around a single XFOIL call, for the airfoil viewer page
-- the airfoil counterpart to physics_service.py's VMEC++/DESC wrapper.
Deliberately simpler: XFOIL converges in ~15-45ms (EXPERIMENT_LOG §21),
several orders of magnitude faster than a VMEC++/DESC equilibrium solve,
so there is no job-polling/background-worker machinery here -- one POST,
one synchronous XFOIL call, one response. An in-memory LRU-ish cache still
exists (repeat requests for the same candidate, e.g. re-opening it in the
UI, are common) but there's no disk cache or job-id protocol to build.

Returns the airfoil's own (x, y) surface coordinates (so the browser
doesn't need to reimplement the CST Bernstein-polynomial synthesis in JS --
`airfoil_oracle.cst_to_coords` is the one source of truth) plus XFOIL's
real pressure-coefficient distribution (`get_cp_distribution`) alongside
cl/cd/cm -- the "shape + physics field" pairing analogous to the
stellarator viewer's "3D surface + |B|".
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/work/scripts")
import airfoil_oracle as oracle  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VIEWER_ORIGINS = ("http://localhost:4321", "http://127.0.0.1:4321")

app = FastAPI(title="vega-airfoil-physics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VIEWER_ORIGINS),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_cache: dict[tuple, dict[str, Any]] = {}
_CACHE_MAX = 512


class PhysicsRequest(BaseModel):
    params: list[float]  # 16 CST coeffs (8 upper + 8 lower)
    reynolds: float
    alpha: float
    mach: float = 0.0


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/physics")
def post_physics(req: PhysicsRequest) -> dict[str, Any]:
    if len(req.params) != oracle.PARAM_DIM:
        raise HTTPException(status_code=400, detail=f"expected {oracle.PARAM_DIM} params, got {len(req.params)}")

    key = (tuple(round(p, 6) for p in req.params), round(req.reynolds, 3), round(req.alpha, 4), round(req.mach, 4))
    if key in _cache:
        return {**_cache[key], "cached": True}

    from xfoil import XFoil
    from xfoil.model import Airfoil

    import numpy as np

    params = np.array(req.params, dtype=np.float64)
    x_coords, y_coords = oracle.cst_to_coords(params)

    xf = XFoil()
    xf.print = False
    try:
        xf.airfoil = Airfoil(x_coords, y_coords)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"structural: {e}")
    xf.Re = req.reynolds
    xf.M = req.mach
    xf.max_iter = 100

    try:
        cl, cd, cm, _cp = xf.a(req.alpha)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"xfoil: {e}")

    converged = not any(np.isnan(v) for v in (cl, cd, cm))
    cp_x = cp_vals = None
    if converged:
        try:
            cp_x, cp_vals = xf.get_cp_distribution()
            cp_x, cp_vals = cp_x.tolist(), cp_vals.tolist()
        except Exception:
            cp_x = cp_vals = None

    result = {
        "shape": {"x": x_coords.tolist(), "y": y_coords.tolist()},
        "converged": converged,
        "cl": None if not converged else float(cl),
        "cd": None if not converged else float(cd),
        "cm": None if not converged else float(cm),
        "cp": None if cp_x is None else {"x": cp_x, "cp": cp_vals},
        "cached": False,
    }
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = result
    return result

"""FastAPI wrapper around a single mug simulation, for the mug viewer page --
the mug counterpart to airfoil_physics_service.py/torax_physics_service.py.
Deliberately the simplest of the three: pure numpy/scipy, ~250ms/call
(EXPERIMENT_LOG §48), no JIT-compile tax (unlike TORAX) and no external
solver process (unlike XFOIL) -- one POST, one synchronous
`simulate_v3(..., record_series=True)` call, one response.

Returns real body geometry (r_base/r_mid/r_rim/height/handle dims, meters
-- so the browser can render the actual frustum-based shape of revolution
rather than reimplementing the geometry math in JS) plus the downsampled
temperature time series (§50's own addition to simulate_v3 for exactly
this purpose) for the viewer's time slider/animation, alongside the same
scalar targets used elsewhere in this project.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

sys.path.insert(0, "/work/scripts")
import mug_oracle as oracle  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VIEWER_ORIGINS = ("http://localhost:4321", "http://127.0.0.1:4321")

app = FastAPI(title="vega-mug-physics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VIEWER_ORIGINS),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_cache: dict[tuple, dict[str, Any]] = {}
_CACHE_MAX = 256


class PhysicsRequest(BaseModel):
    params: list[float]  # the 14 PARAM_NAMES, see mug_oracle.py


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/physics")
def post_physics(req: PhysicsRequest) -> dict[str, Any]:
    if len(req.params) != oracle.PARAM_DIM:
        raise HTTPException(status_code=400, detail=f"expected {oracle.PARAM_DIM} params, got {len(req.params)}")

    key = tuple(round(p, 6) for p in req.params)
    if key in _cache:
        return {**_cache[key], "cached": True}

    try:
        r = oracle.simulate_v3(*req.params, t_max=oracle.T_HORIZON_S, record_at=60.0, record_series=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"mug sim: {e}")

    if not r["valid"]:
        raise HTTPException(status_code=422, detail="invalid input (non-finite, or non-positive geometry/thickness/length/diameter)")

    result = {
        "geometry": r["geometry"],
        "series": r["series"],
        "targets": {
            "temp_at_2h_C": r["final_liq_temp_C"],
            "mass_kg": r["mass_kg"],
            "touch_temp_60s_C": r["touch_temp_at_C"],
            "handle_temp_60s_C": r["handle_temp_at_C"],
        },
        "liquid_volume_L": r["liquid_volume_L"],
        "cached": False,
    }
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = {k: v for k, v in result.items() if k != "cached"}
    return result

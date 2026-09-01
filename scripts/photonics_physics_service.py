"""FastAPI wrapper around a single photonics (grating coupler) MEEP call,
for the photonics viewer page -- the photonics counterpart to
torax_physics_service.py/airfoil_physics_service.py. Runs `meep` in the
SAME process as fastapi/uvicorn (unlike the dataset-generation harness,
which spawns a fresh subprocess per candidate via oracle_harness.py) --
calls photonics_oracle.compute_efficiencies() directly, synchronously, no
job-polling machinery, matching airfoil's "converges fast enough for one
POST = one call" pattern rather than VMEC++/DESC's disk-cache-plus-job-id
one. Real per-call cost (EXPERIMENT_LOG) is ~1.3-3.7s depending on
fidelity -- slow enough to feel in a UI, so the viewer's own JS debounces
slider input rather than firing a request per pixel of drag, but not slow
enough to need async polling.

v2: request/geometry updated for photonics_oracle.py's 6-param schema
(apodization via duty_cycle_start/duty_cycle_end, box_thickness_um
promoted from a fixed constant to a real field on every request/response --
see photonics_oracle.py's own module docstring). Returns the same 6
efficiencies (now including fiber_coupling_efficiency) photonics_oracle.py's
worker_fn does, plus the real cross-section geometry (waveguide extent,
apodized grating tooth rectangles, box-oxide thickness) so the browser
draws the ACTUAL simulated structure to scale, not a schematic guess --
same "shape + physics" pairing as airfoil_physics_service.py's (x,y) CST
coordinates.
"""
from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "/work/scripts")
import photonics_oracle as oracle  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VIEWER_ORIGINS = ("http://localhost:4321", "http://127.0.0.1:4321")

app = FastAPI(title="vega-photonics-physics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VIEWER_ORIGINS),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_cache: dict[tuple, dict[str, Any]] = {}
_CACHE_MAX = 512


class PhysicsRequest(BaseModel):
    wg_thickness_um: float
    grating_period_um: float
    duty_cycle_start: float
    duty_cycle_end: float
    etch_depth_um: float
    box_thickness_um: float
    fidelity: str = "low"


def _geometry(wg_thickness_um: float, grating_period_um: float, duty_cycle_start: float, duty_cycle_end: float,
              etch_depth_um: float, box_thickness_um: float) -> dict[str, Any]:
    """Real (apodized) tooth rectangles for the grating region, same
    layout math as photonics_oracle._build_geometry/_tooth_widths -- the
    viewer draws exactly what was simulated, not an approximation of it."""
    grating_len = oracle.N_PERIODS * grating_period_um
    tooth_widths = oracle._tooth_widths(grating_period_um, duty_cycle_start, duty_cycle_end)
    teeth = []
    for i, tooth_w in enumerate(tooth_widths):
        tooth_x0 = i * grating_period_um
        teeth.append({"x0": round(float(tooth_x0), 5), "width": round(float(tooth_w), 5)})
    return {
        "wg_thickness_um": wg_thickness_um,
        "grating_period_um": grating_period_um,
        "duty_cycle_start": duty_cycle_start,
        "duty_cycle_end": duty_cycle_end,
        "etch_depth_um": etch_depth_um,
        "box_thickness_um": box_thickness_um,
        "grating_length_um": grating_len,
        "n_periods": oracle.N_PERIODS,
        "teeth": teeth,  # each tooth spans [x0, x0+width) at full wg_thickness_um; the gap after each tooth is etched to etch_depth_um
        "wavelength_um": oracle.WAVELENGTH_UM,
    }


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/physics")
def post_physics(req: PhysicsRequest) -> dict[str, Any]:
    if req.fidelity not in oracle.FIDELITY_PRESETS:
        raise HTTPException(status_code=400, detail=f"fidelity must be one of {list(oracle.FIDELITY_PRESETS)}")

    key = (round(req.wg_thickness_um, 6), round(req.grating_period_um, 6),
           round(req.duty_cycle_start, 6), round(req.duty_cycle_end, 6),
           round(req.etch_depth_um, 6), round(req.box_thickness_um, 6), req.fidelity)
    if key in _cache:
        return {**_cache[key], "cached": True}

    resolution = oracle.FIDELITY_PRESETS[req.fidelity]
    ok, payload = oracle.compute_efficiencies(req.wg_thickness_um, req.grating_period_um,
                                                req.duty_cycle_start, req.duty_cycle_end,
                                                req.etch_depth_um, req.box_thickness_um, resolution,
                                                capture_field=True)

    field = payload.pop("field", None) if ok else None
    result = {
        "geometry": _geometry(req.wg_thickness_um, req.grating_period_um, req.duty_cycle_start,
                               req.duty_cycle_end, req.etch_depth_um, req.box_thickness_um),
        "ok": ok,
        "targets": payload if ok else None,
        "field": field,  # real |Ez|^2 DFT field at fcen, from the grating run -- see photonics_oracle.py's docstring
        "error": None if ok else payload,
        "cached": False,
    }
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = {k: v for k, v in result.items() if k != "cached"}
    return result

"""FastAPI wrapper around a single TORAX simulation, for the TORAX viewer
page -- the TORAX counterpart to airfoil_physics_service.py/physics_service.py.

Unlike airfoil's oracle, a single TORAX call pays a real one-time JIT-compile
tax (~4.9s cold, confirmed EXPERIMENT_LOG §25/§31/§37) the first time this
PROCESS calls `run_simulation`, then drops to ~0.1-0.5s for every later call
in the same process even as real parameters vary. Handled here by warming
the JIT at import time (one throwaway call before the app starts serving)
rather than building airfoil's simple fully-synchronous-with-no-warmup
pattern OR physics_service.py's full job-polling machinery -- a single
uvicorn worker process (no --workers, matching physics-airfoil's own config)
keeps that warm cache for its whole lifetime, so once started this can stay
synchronous like the airfoil service, not needing VMEC++/DESC's disk-cache-
plus-job-id protocol (that exists because a DESC solve itself can take
seconds to tens of seconds EVERY call, not just the first).

Returns the full radial-profile time history (T_e, T_i, n_e, j_total on the
25 cell centers; q_face on the 26 face points; both across all ~22
timesteps) -- confirmed available by direct inspection before this was
written, not assumed -- plus the final-timestep scalar targets
(Q_fusion/tau_E/H98/T_e_volume_avg) already used elsewhere in this project.
"""
from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "/work/scripts")
import torax_oracle as oracle  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VIEWER_ORIGINS = ("http://localhost:4321", "http://127.0.0.1:4321")

app = FastAPI(title="vega-torax-physics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VIEWER_ORIGINS),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_cache: dict[tuple, dict[str, Any]] = {}
_CACHE_MAX = 256


class PhysicsRequest(BaseModel):
    overrides: dict[str, float]  # the 10 PARAM_NAMES -> value


def _run_full(overrides: dict[str, float]) -> dict[str, Any]:
    from torax._src.orchestration.run_simulation import run_simulation
    from torax._src.state import SimError
    import numpy as np

    n_rho = oracle.FIDELITY_PRESETS["low"]
    cfg = oracle._build_config(overrides, n_rho)
    _data_tree, hist = run_simulation(cfg, progress_bar=False)

    if hist.sim_error != SimError.NO_ERROR:
        return {"ok": False, "error": f"torax: {hist.sim_error.name}"}

    ppo_last = hist.post_processed_outputs[-1]
    targets = {name: float(getattr(ppo_last, name)) for name in oracle.TARGET_NAMES}
    if not all(np.isfinite(v) for v in targets.values()):
        return {"ok": False, "error": "torax: non-finite output"}

    times = np.asarray(hist.times).tolist()
    rho_cell = np.asarray(hist.rho_cell_norm).tolist()
    rho_face = np.asarray(hist.rho_face_norm).tolist()

    def cell_series(attr):
        return [np.asarray(getattr(cp, attr).value).tolist() for cp in hist.core_profiles]

    def face_series(attr):
        return [np.asarray(getattr(cp, attr)).tolist() for cp in hist.core_profiles]

    profiles = {
        "times": times,
        "rho_cell_norm": rho_cell,
        "rho_face_norm": rho_face,
        # T_e/T_i in keV, n_e in m^-3, j_total in A/m^2, q_face dimensionless --
        # TORAX's own native units, not rescaled (matches how this project's
        # scalar targets are already reported unscaled elsewhere).
        "T_e": cell_series("T_e"),
        "T_i": cell_series("T_i"),
        "n_e": cell_series("n_e"),
        "j_total": [np.asarray(cp.j_total).tolist() for cp in hist.core_profiles],
        "q_face": face_series("q_face"),
    }
    return {"ok": True, "targets": targets, "profiles": profiles}


@app.on_event("startup")
def _warm_jit() -> None:
    """One throwaway call so the FIRST real request doesn't pay the ~4.9s
    cold-compile tax -- confirmed necessary (§25/§31), not precautionary."""
    import torax.examples.basic_config as bc  # noqa: F401
    default_overrides = {"Ip": 15_000_000.0, "nbar": 8.5e19, "chi_i": 1.0, "chi_e": 1.0,
                          "P_total": 1.2e8, "I_generic": 3_000_000.0,
                          "R_major": 6.2, "a_minor": 2.0, "B_0": 5.3, "elongation_LCFS": 1.72}
    try:
        _run_full(default_overrides)
    except Exception:
        pass  # warmup is best-effort; a real failure will surface on the first real request


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/physics")
def post_physics(req: PhysicsRequest) -> dict[str, Any]:
    missing = [n for n in oracle.PARAM_NAMES if n not in req.overrides]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing params: {missing}")

    key = tuple(round(req.overrides[n], 6) for n in oracle.PARAM_NAMES)
    if key in _cache:
        return {**_cache[key], "cached": True}

    try:
        result = _run_full(req.overrides)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"torax: {e}")

    result["cached"] = False
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = {k: v for k, v in result.items() if k != "cached"}
    return result

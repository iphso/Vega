"""FastAPI wrapper around physics_bundle for the Vega viewer.

POST /physics starts (or returns) a bundle. For the vega_testing
candidates this loads the pre-solved DESC h5 under
vega_testing/<DESC_RUN>/equilibria/, or a VMEC++ / booz_xform pack under
vega_testing/vmecpp_highfi/equilibria/ when backend=vmecpp. Cached JSON
payloads return 200 immediately; in-flight work returns 202 + a
job id. Cache key is sha256 of the canonical request JSON, stored under
output/physics_cache/.

Cold request is one DESC compute on a locked (5×32×48) grid plus numpy
interpolation; repeats hit the disk cache. backend=vmecpp is much cheaper
-- a Fourier sum over saved Boozer coefficients, no solve at all. The s
slider is client-side. CORS is open to the viewer origin only.

|B| in the returned bundle is dimensionless (|B|/⟨|B|⟩ on the surface) so
the two backends share one scale; BUNDLE_SCHEMA_VERSION is mixed into the
cache key so a change to that meaning retires stale bundles on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from physics_bundle import (
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_L,
    DEFAULT_M,
    DEFAULT_N,
    DEFAULT_N_SEEDS,
    DEFAULT_N_THETA,
    DEFAULT_N_TURNS,
    DEFAULT_N_ZETA,
    DEFAULT_N_ZETA_PLANES,
    DESC_RESULTS_NAME,
    DESC_RUN,
    TESTING_DIR,
    VMECPP_RESULTS_NAME,
    VMECPP_RUN,
    compute_bundle,
)
from pydantic import BaseModel, Field

# Container path by default; VEGA_PHYSICS_CACHE overrides it, matching
# VEGA_TESTING_DIR in physics_bundle so the service can run outside Docker.
CACHE_DIR = Path(
    os.environ.get("VEGA_PHYSICS_CACHE", "/work/output/physics_cache")
)
VIEWER_ORIGINS = (
    "http://localhost:4321",
    "http://127.0.0.1:4321",
)

app = FastAPI(title="vega-physics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VIEWER_ORIGINS),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="physics")
_lock = threading.Lock()
# cache_key -> {status, job_id, bundle, error}
_jobs: dict[str, dict[str, Any]] = {}


class PoincareSpec(BaseModel):
    n_seeds: int = DEFAULT_N_SEEDS
    n_turns: int = DEFAULT_N_TURNS
    phi: float = 0.0
    surfaces_s: list[float] | None = None
    direction: str = "forward"


class PhysicsRequest(BaseModel):
    r_cos: list[list[float]]
    z_sin: list[list[float]]
    nfp: int
    s: float = 0.5
    # VMEC++ by default: it returns a physical rotational transform on every
    # shipped row, DESC on a minority of them.
    backend: str = "vmecpp"
    n_theta: int = DEFAULT_N_THETA
    n_zeta: int = DEFAULT_N_ZETA
    n_zeta_planes: int = DEFAULT_N_ZETA_PLANES
    poincare: PoincareSpec = Field(default_factory=PoincareSpec)
    L: int = DEFAULT_L
    M: int = DEFAULT_M
    N: int = DEFAULT_N


def _canonical(req: PhysicsRequest) -> str:
    # s is a client-side nearest-slice pick from bundle.flux; do not
    # fragment the disk cache when the slider moves.
    payload = req.model_dump()
    payload.pop("s", None)
    # Mixed into the key so a schema bump retires every stale bundle on disk.
    # Without it the v1 tesla-valued bundles would keep being served under
    # the v2 field names, and |B| would silently be plotted on two scales.
    payload["schema"] = BUNDLE_SCHEMA_VERSION
    # Likewise the run folders. The request body names a *backend*, not the
    # campaign behind it, so repointing VMECPP_RUN (lowfi -> highfi) or
    # DESC_RUN leaves keys unchanged and the old campaign's bundles keep
    # being served -- a highfi request answered with lowfi data, cached=True,
    # no error anywhere. Naming the runs makes a repoint retire them.
    payload["desc_run"] = DESC_RUN
    payload["vmecpp_run"] = VMECPP_RUN
    # ...and the run *contents*, not just its name. A campaign re-run in
    # place keeps the folder name, so the name alone is not enough: the
    # third DESC L=M=N=8 campaign overwrote the second under the same name
    # and the service kept serving the old iota=0 bundles, cached=True.
    # Stamping results.npz retires them the moment a re-run lands.
    payload["run_stamps"] = _run_stamps()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _run_stamps() -> dict[str, str]:
    """(size, mtime) of each run's results.npz -- a cheap content fingerprint."""
    stamps: dict[str, str] = {}
    for name, results in (
        (DESC_RUN, TESTING_DIR / DESC_RESULTS_NAME),
        (VMECPP_RUN, TESTING_DIR / VMECPP_RESULTS_NAME),
    ):
        try:
            st = results.stat()
            stamps[name] = f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            stamps[name] = "missing"
    return stamps


def _cache_key(req: PhysicsRequest) -> str:
    return hashlib.sha256(_canonical(req).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_cache(key: str, bundle: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(key).with_suffix(".tmp")
    tmp.write_text(json.dumps(bundle))
    tmp.replace(_cache_path(key))


def _run_job(key: str, req: PhysicsRequest) -> None:
    """Build one bundle on the worker thread and record the outcome."""
    try:
        bundle = compute_bundle(
            np.asarray(req.r_cos, dtype=np.float64),
            np.asarray(req.z_sin, dtype=np.float64),
            req.nfp,
            s=req.s,
            backend=req.backend,
            n_theta=req.n_theta,
            n_zeta=req.n_zeta,
            n_zeta_planes=req.n_zeta_planes,
            poincare=req.poincare.model_dump(),
            L=req.L,
            M=req.M,
            N=req.N,
            allow_solve=False,
        )
        bundle["meta"]["cached"] = False
        bundle["meta"]["cache_key"] = key
        _write_cache(key, bundle)
        with _lock:
            _jobs[key] = {
                "status": "done",
                "job_id": key,
                "bundle": bundle,
                "error": None,
            }
    # Blind by design: a solver or DESC failure has to reach the client as a
    # job error, not kill the single worker thread and hang every poller.
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        with _lock:
            _jobs[key] = {
                "status": "error",
                "job_id": key,
                "bundle": None,
                "error": err,
            }


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


# response_model=None on both routes: the `dict | JSONResponse` return is a
# deliberate "200 with the bundle, or 202 with a job id", and FastAPI refuses
# to build a response model from that union (it is not a Pydantic type).
# Without this the module raises FastAPIError at import and uvicorn's reloader
# holds the port open with nothing behind it -- the service looks up, and
# every request hangs instead of failing.
@app.post("/physics", response_model=None)
def post_physics(req: PhysicsRequest) -> dict[str, Any] | JSONResponse:
    """Return a cached bundle, or start one and answer 202 + job id."""
    key = _cache_key(req)
    cached = _read_cache(key)
    if cached is not None:
        cached.setdefault("meta", {})["cached"] = True
        return cached

    with _lock:
        job = _jobs.get(key)
        if job is None or job["status"] == "error":
            _jobs[key] = {
                "status": "running",
                "job_id": key,
                "bundle": None,
                "error": None,
            }
            _executor.submit(_run_job, key, req)
        elif job["status"] == "done" and job["bundle"] is not None:
            return job["bundle"]

    return JSONResponse(
        status_code=202,
        content={"job_id": key, "status": "running"},
    )


@app.get("/physics/jobs/{job_id}", response_model=None)
def get_job(job_id: str) -> dict[str, Any] | JSONResponse:
    """Poll a job: 200 with the bundle, 202 while running, 404/500 otherwise."""
    cached = _read_cache(job_id)
    if cached is not None:
        cached.setdefault("meta", {})["cached"] = True
        return cached
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        if job["status"] == "running":
            return JSONResponse(
                status_code=202,
                content={"job_id": job_id, "status": "running"},
            )
        if job["status"] == "error":
            raise HTTPException(status_code=500, detail=job["error"])
        return job["bundle"]

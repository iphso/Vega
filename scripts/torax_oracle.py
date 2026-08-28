"""TORAX (Google DeepMind's tokamak core transport simulator) implementation
of oracle_base.py's interface -- the third domain, and a real test of
gym_schema.py's abstraction (§27-30) against something meaningfully
different from VMEC++/XFOIL: a JIT-compiled JAX simulator whose cost is
dominated by a one-time compile tax, not the physics itself (confirmed
directly, EXPERIMENT_LOG §25/§31: ~4.9s cold vs. ~0.13-0.45s warm in the
same process, even when real parameters change between calls).

Design vector (10 continuous scalars, chosen per direct user request --
"scalar knobs + geometry" -- over both the 6-scalar-only and full-physics-
parameterization alternatives): plasma current (Ip), line-averaged density
(nbar), ion/electron heat diffusivity (chi_i, chi_e, constant-model
transport), auxiliary heating power (P_total), generic current drive
(I_generic), and 4 geometry scalars (R_major, a_minor, B_0,
elongation_LCFS) -- a simple circular-geometry tokamak, not a real
equilibrium file. All 10 confirmed to build and run correctly via
`ToraxConfig.from_dict` overrides on top of the bundled `basic_config`
example, not assumed from the config schema alone.

Targets (4, matching this project's now-standard 4-target airfoil
convention rather than VMEC++'s 11): Q_fusion (fusion gain -- the classic
headline tokamak metric), tau_E (energy confinement time), H98 (confinement
quality relative to the standard H98 scaling), T_e_volume_avg
(volume-averaged electron temperature). All four are real
PostProcessedOutputs fields, confirmed by direct inspection of a real run's
output, not invented.

Because of the JIT-tax finding, this domain does NOT use oracle_harness.py's
worker_fn/run_batch_with_timeout contract -- see
oracle_harness_persistent.py's module docstring for why a persistent
worker pool is the correct harness here, not a compromise. `worker_fn`
below is kept for interface parity /completeness (a fresh-subprocess-per-
candidate caller could still use it, just slowly), but
`persistent_worker_fn` is the one actually meant to be used, via
`run_batch_persistent`.
"""
import copy

import numpy as np

TARGET_NAMES = ["Q_fusion", "tau_E", "H98", "T_e_volume_avg"]
LOG_TARGET_NAMES = ["Q_fusion", "tau_E"]  # both strictly positive, wide dynamic range near sub-ignition configs

PARAM_NAMES = ["Ip", "nbar", "chi_i", "chi_e", "P_total", "I_generic",
               "R_major", "a_minor", "B_0", "elongation_LCFS"]
PARAM_DIM = len(PARAM_NAMES)
ZERO_INDICES = []  # no structurally-fixed dims in this parameterization

# Only one fidelity tier for now (n_rho=25, the bundled example's own
# default radial grid resolution) -- matches airfoil's own single-tier
# start (EXPERIMENT_LOG §21) rather than assuming a multi-fidelity ladder
# before one's actually built and measured. n_rho is a real, natural
# fidelity knob for a later medium/high tier (coarser/finer radial grid,
# the same spirit as VMEC++'s multigrid resolution presets) -- not built
# yet.
FIDELITY_PRESETS = {"low": 25}


def _build_config(overrides, n_rho):
    """overrides: dict with PARAM_NAMES' keys -> physical values.
    Raises on a structurally-invalid config (caught by the caller)."""
    import torax.examples.basic_config as bc
    from torax._src.torax_pydantic.model_config import ToraxConfig

    d = copy.deepcopy(bc.CONFIG)
    d["profile_conditions"] = {**d.get("profile_conditions", {}),
                               "Ip": float(overrides["Ip"]), "nbar": float(overrides["nbar"])}
    d["transport"] = {**d.get("transport", {}),
                      "chi_i": float(overrides["chi_i"]), "chi_e": float(overrides["chi_e"])}
    d["sources"] = copy.deepcopy(d.get("sources", {}))
    d["sources"]["generic_heat"] = {**d["sources"].get("generic_heat", {}),
                                     "P_total": float(overrides["P_total"])}
    d["sources"]["generic_current"] = {**d["sources"].get("generic_current", {}),
                                        "I_generic": float(overrides["I_generic"])}
    d["geometry"] = {**d.get("geometry", {}), "geometry_type": "circular", "n_rho": int(n_rho),
                     "R_major": float(overrides["R_major"]), "a_minor": float(overrides["a_minor"]),
                     "B_0": float(overrides["B_0"]), "elongation_LCFS": float(overrides["elongation_LCFS"])}
    return ToraxConfig.from_dict(d)


def _run_one(overrides, n_rho):
    """Returns (ok, payload_dict_or_error_string). Shared by both worker_fn
    (one-shot subprocess) and persistent_worker_fn (long-lived process) --
    the actual physics call is identical either way, only the process
    lifecycle around it differs.

    Takes an already-RESOLVED n_rho (e.g. 25), not a fidelity name string --
    matching vmec_oracle.py's/airfoil_oracle.py's own established
    convention: every caller in this project resolves
    `FIDELITY_PRESETS[name]` itself before calling into worker_fn/
    params_to_worker_args (confirmed directly: eval_cvae_steerability.py,
    eval_random_direction_steerability.py, and gym_schema.py's
    `FidelityLevel(name, FIDELITY_PRESETS[name], ...)` all do this),
    so worker_fn never does its own internal name lookup for any other
    domain. A real bug WAS caught and fixed here (EXPERIMENT_LOG §35): the
    original generate_torax_dataset.py called params_to_worker_args with
    the raw fidelity KEY string "low" instead of the resolved value, so
    `int(n_rho)` in _build_config raised on every single call. The first
    fix made THIS function resolve the name internally instead -- worked
    for that one caller, but deviated from the established convention and
    would have broken the NEXT one (steerability_generic.py's
    gym_schema-based fidelity resolution already passes the resolved
    value, per FidelityLevel's own docstring) -- caught while extending to
    that script, reverted back to the convention every other domain
    follows, and fixed at the actual call sites
    (generate_torax_dataset.py, real_reference_torax.py) instead."""
    from torax._src.orchestration.run_simulation import run_simulation
    from torax._src.state import SimError

    try:
        cfg = _build_config(overrides, n_rho)
    except Exception as e:
        return False, f"structural: {e}"

    try:
        _data_tree, history = run_simulation(cfg, progress_bar=False)
    except Exception as e:
        return False, f"torax: {e}"

    if history.sim_error != SimError.NO_ERROR:
        return False, f"torax: {history.sim_error.name}"

    ppo = history.post_processed_outputs[-1]
    payload = {name: float(getattr(ppo, name)) for name in TARGET_NAMES}
    if not all(np.isfinite(v) for v in payload.values()):
        return False, "torax: non-finite output"
    return True, payload


def worker_fn(conn, overrides, n_rho):
    """One-shot subprocess entry point -- oracle_harness.py's own
    run_batch_with_timeout contract, kept for interface parity. `n_rho` is
    an already-resolved fidelity value (e.g. 25), not a name -- the caller
    resolves FIDELITY_PRESETS[name] itself, same convention as every other
    domain module. Pays the full JIT-compile tax on every call; only use
    this for a single exploratory call, never a real batch (see module
    docstring)."""
    try:
        ok, payload = _run_one(overrides, n_rho)
        conn.send((ok, payload))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


_VMAP_JIT_CACHE = {}


def _make_batch_step_fns(overrides_list, n_rho):
    """Builds one SimulationStepFn per candidate via the ordinary (cheap,
    ~0s) make_step_fn path, filtering out any candidate that fails to even
    construct a valid ToraxConfig (bad pydantic values) before stacking --
    these can't be vmapped through a shared pytree shape, so they're
    rejected up front with an explicit reason, same spirit as
    vmec_oracle.py's own structural-failure branch. Returns (step_fns,
    ok_indices, reasons) where `reasons` maps original-index -> error
    string for anything dropped here."""
    from torax import experimental as exp
    step_fns, ok_indices, reasons = [], [], {}
    for i, overrides in enumerate(overrides_list):
        try:
            cfg = _build_config(overrides, n_rho)
            step_fns.append(exp.make_step_fn(cfg))
            ok_indices.append(i)
        except Exception as e:
            reasons[i] = f"structural: {e}"
    return step_fns, ok_indices, reasons


def run_batch_vmap(worker_args_list, max_steps=200):
    """Batched GPU oracle call for a whole bootstrap round at once --
    EXPERIMENT_LOG's real vmap-batching spike: jax.vmap(experimental.
    run_loop_jit) over N independently-built SimulationStepFn pytrees,
    stacked via jax.tree_util.tree_map(jnp.stack, ...). Measured ~150x
    throughput vs. the per-candidate subprocess path at batch=256 (0.90s
    warm wall-clock for the WHOLE batch, vs. ~0.55s PER CANDIDATE
    sequentially), and verified bit-exact (~1e-15 relative error) against
    the non-batched oracle._run_one reference for the same configs.

    `worker_args_list`: list of (overrides_dict, n_rho) tuples -- exactly
    what params_to_worker_args() produces per candidate (same convention
    build_candidates()/oracle_harness.py callers already use), NOT raw
    param arrays. All candidates in one call must share the same n_rho --
    true within a single bootstrap round, since fidelity is round-wide,
    not per-candidate.

    Returns a list of (ok, payload_or_reason) aligned 1:1 with
    `worker_args_list`'s own order -- same (ok, payload) shape worker_fn's
    single-candidate callers already expect, so bootstrap_generic.py's
    accept/reject bookkeeping doesn't need a special case for this domain.

    The real fix for naive-GPU's "recompiles every single call" problem
    (see the spike's own cold/warm confusion, caught and fixed): this
    caches the compiled jax.jit(vmap(...)) function by (batch_size,
    max_steps). As long as every round in a bootstrap run uses the SAME
    round_size, only the FIRST round pays the ~15s one-time compile tax;
    every later round hits the cache and runs in the flat ~0.8-0.9s warm
    regime the spike measured, regardless of batch size. A caller that
    varies round_size between rounds defeats this -- keep it fixed."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from torax import experimental as exp

    n = len(worker_args_list)
    n_rho = worker_args_list[0][1] if n else None
    overrides_list = [wa[0] for wa in worker_args_list]
    step_fns, ok_indices, reasons = _make_batch_step_fns(overrides_list, n_rho)
    results = [None] * n
    for i, reason in reasons.items():
        results[i] = (False, reason)

    if not step_fns:
        return results

    cache_key = (len(step_fns), max_steps)
    if cache_key not in _VMAP_JIT_CACHE:
        _VMAP_JIT_CACHE[cache_key] = jax.jit(
            jax.vmap(lambda sf: exp.run_loop_jit(sf, max_steps=max_steps))
        )
    jit_fn = _VMAP_JIT_CACHE[cache_key]

    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *step_fns)
    _, ppo_hist, final_i = jax.block_until_ready(jit_fn(stacked))
    final_i_np = np.asarray(final_i)

    for local_b, orig_i in enumerate(ok_indices):
        idx = int(final_i_np[local_b])
        if idx >= max_steps:
            # Never hit step_fn.is_done() within budget -- the bounded-loop
            # analogue of SimError.DID_NOT_REACH_T_FINAL. run_loop_jit
            # itself doesn't surface a SimError the way the higher-level
            # run_simulation()/run_simulation_jitted() wrappers do (it's a
            # lower-level primitive -- confirmed by reading its source,
            # not assumed), so this step-count heuristic is the substitute;
            # it will NOT catch a SimError that still finishes within
            # max_steps (e.g. NEGATIVE_CORE_PROFILES) -- same known-gap
            # class as the input-side param caps flagged in §37/38, not
            # yet closed here either.
            results[orig_i] = (False, f"torax: did not converge within max_steps={max_steps} (final_i={idx})")
            continue
        vals = {name: float(np.asarray(getattr(ppo_hist, name))[local_b, idx]) for name in TARGET_NAMES}
        if not all(np.isfinite(v) for v in vals.values()):
            results[orig_i] = (False, "torax: non-finite output")
        else:
            results[orig_i] = (True, vals)
    return results


def persistent_worker_fn(conn):
    """Long-lived process entry point for oracle_harness_persistent.py's
    run_batch_persistent -- imports torax ONCE, then loops handling many
    candidates in the same warm process, which is the whole point (see
    oracle_harness_persistent.py's module docstring for the ~10-38x
    same-process speedup this is designed to actually capture)."""
    import torax.examples.basic_config as bc  # noqa: F401 -- warms the import before the first candidate
    while True:
        msg = conn.recv()
        if msg is None:
            conn.close()
            return
        overrides, n_rho = msg  # n_rho already resolved by the caller, see worker_fn's docstring
        try:
            ok, payload = _run_one(overrides, n_rho)
            conn.send((ok, payload))
        except Exception as e:
            conn.send((False, f"unknown: {e}"))


def params_to_worker_args(params, aux, n_rho):
    """params: (10,) float array, already unstandardized, in PARAM_NAMES
    order. aux: unused (no discrete conditioning variable in this domain,
    unlike VMEC's n_field_periods or airfoil's Reynolds/alpha -- everything
    that varies is already in `params`). `n_rho`: an already-resolved
    fidelity value (e.g. FIDELITY_PRESETS["low"] == 25), NOT a name string
    -- the caller resolves the name first, same convention as
    vmec_oracle.py/airfoil_oracle.py's own params_to_worker_args (see
    worker_fn's docstring for the bug this caught). Returns
    (overrides_dict, n_rho) matching what persistent_worker_fn/worker_fn
    expect."""
    overrides = {name: float(params[i]) for i, name in enumerate(PARAM_NAMES)}
    return (overrides, n_rho)

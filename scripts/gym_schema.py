"""Formal schema for what this project has been informally building since
oracle_base.py: not a collection of physics simulators, but a family of
constrained inverse-design / design-navigation environments sharing one
shape --

    x in X  --oracle(s)-->  y = f(x) in R^m,  with validity v(x) in {0,1}

-- across possibly several fidelities f_0, f_1, ... of different cost, and
three distinct tasks a method can be asked to solve against that shape:

  1. Target (inverse design): given y*, produce valid x with f(x) ~ y*.
     What eval_cvae_steerability.py's "validity" half already measures.
  2. Navigate (local design geometry): given (x, dy), produce valid x' with
     f(x') - f(x) ~ dy. What eval_cvae_steerability.py's "steerability"
     half and eval_random_direction_steerability.py already measure.
  3. Budgeted discovery: given a goal and a finite oracle budget (spent
     faster on expensive fidelities), reach the goal as cheaply as
     possible. NOT implemented anywhere in this project yet -- the
     concrete new thing this schema exists to make buildable.

This module formalizes the DATA the first two already pass around
implicitly (oracle_base.py's prose contract) into real types, and adds the
two concepts needed for the third task that nothing here has today: an
explicit validity signal separate from "did the oracle not crash," and a
cost attached to each fidelity so a budget means something.

Existing domain modules (vmec_oracle.py, airfoil_oracle.py) are NOT
rewritten against this file -- they still directly implement oracle_base.py's
worker_fn/params_to_worker_args contract, and DomainSpec below is a thin
wrapper around exactly that contract (see vmec_spec()/airfoil_spec() at the
bottom for the real adapters, using this project's own already-measured
numbers, not invented ones). A new domain can either implement
oracle_base.py's contract directly (as both existing ones do) and get
wrapped, or build a DomainSpec natively -- both are first-class.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Core data shapes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single point in design space X. `params` is the flat, continuous
    part every generative model in this project actually produces (already
    unstandardized, already zero-enforced at a domain's ZERO_INDICES).
    `aux` is small discrete/continuous conditioning that travels alongside
    params but isn't decoded by the generative model itself -- VMEC's
    n_field_periods, airfoil's (reynolds, alpha). Optional `tag` is for
    harness bookkeeping (oracle_harness.run_batch_with_timeout's existing
    completion-order-not-submission-order contract), not part of the
    design itself."""
    params: "np.ndarray"
    aux: dict = field(default_factory=dict)
    tag: Optional[object] = None


@dataclass
class FidelityLevel:
    """One rung of a domain's oracle ladder. `cost_credits` is wall-clock
    seconds per call, normalized so the CHEAPEST call in this whole project
    (airfoil/XFOIL, ~30ms) is 1 credit -- see the worked examples at the
    bottom for where every number below comes from; none are invented.
    `internal_name` is the domain-specific opaque string oracle_base.py's
    FIDELITY_PRESETS already passes through unexamined."""
    name: str
    internal_name: str
    cost_credits: float
    measured_from: str  # one-line provenance -- where this number came from


@dataclass
class OracleResult:
    """What one oracle call returns -- a strict superset of oracle_base.py's
    existing (ok, payload) pair. `valid` is deliberately a SEPARATE field
    from `ok`: `ok=False` means the oracle itself failed/timed out/crashed
    (VMEC++ non-convergence, XFOIL non-convergence, a TORAX SimError).
    `valid=False` with `ok=True` means the oracle ran fine and returned
    real metrics, but a domain-specific sanity/legality check on the RESULT
    rejects it anyway -- e.g. airfoil_oracle.py's cd<1e-6 rejection
    (EXPERIMENT_LOG §22), which historically got silently folded into the
    oracle's own ok flag rather than kept as a distinct concept. Keeping
    them separate is new here, not a description of what the two existing
    domain modules currently do (they still only expose `ok`; a caller
    wrapping them into this schema derives `valid` from
    `DomainSpec.validity_fn`, see below)."""
    ok: bool
    valid: bool
    metrics: Optional[dict] = None
    error: Optional[str] = None
    fidelity: Optional[str] = None
    cost_credits: float = 0.0


class WorkerFn(Protocol):
    def __call__(self, conn, *worker_args) -> None: ...


@dataclass
class DomainSpec:
    """Generalizes oracle_base.py's prose contract into one structured
    object. Everything above `validity_fn` maps 1:1 onto an existing
    domain module's module-level constants/functions -- see vmec_spec()/
    airfoil_spec() below for the real, filled-in adapters. `validity_fn`
    and the cost-bearing `FidelityLevel`s are the two genuinely NEW pieces
    this schema adds that neither existing domain module has today."""
    name: str
    target_names: list
    log_target_names: list
    param_dim: int
    zero_indices: list
    fidelities: list  # list[FidelityLevel], cheapest first
    worker_fn: WorkerFn
    params_to_worker_args: Callable
    validity_fn: Optional[Callable[[dict], bool]] = None
    # `validity_fn(metrics_dict) -> bool`, applied on top of `ok=True`
    # results. None means "ok is the only validity signal this domain has"
    # -- true of both vmec_oracle.py and airfoil_oracle.py's CURRENT ok
    # flag (airfoil's cd<1e-6 check lives inside worker_fn today, not here
    # -- a real domain would want it moved out to this field so `ok` means
    # purely "the solver ran" and `valid` means "the result is legitimate,"
    # not conflated as they are right now).
    harness_fit: str = "subprocess-per-candidate"  # or "needs-batched-worker"
    harness_fit_note: str = ""
    sanity_filter: Optional[Callable] = None
    # sanity_filter(Y: np.ndarray) -> bool mask, applied before training/eval
    # on top of "ok" -- e.g. airfoil_oracle's cd<1e-6/l_over_d>300 rejection
    # (EXPERIMENT_LOG §22), currently duplicated as inline code in every
    # airfoil training/eval script rather than expressed once here. None
    # (VMEC++'s case) means no post-hoc filter beyond the oracle's own ok.
    dataset_paths: Optional[Callable] = None
    # dataset_paths(out_dir) -> (X_path, Y_path) -- domain dataset file
    # naming isn't uniform today (X.npy/Y.npy for VMEC++ vs.
    # airfoil_X.npy/airfoil_Y.npy for airfoils); this makes that a single
    # lookup instead of a hardcoded path repeated in every script.
    persistent_worker_fn: Optional[Callable] = None
    # EXPERIMENT_LOG §37: the batched-worker counterpart to `worker_fn`,
    # populated only when harness_fit=="needs-batched-worker" (today: only
    # torax_oracle.persistent_worker_fn). A caller should branch on
    # harness_fit and use oracle_harness_persistent.run_batch_persistent
    # with THIS function instead of oracle_harness.run_batch_with_timeout
    # when it's set -- worker_fn alone still works for any domain but pays
    # the full per-candidate cost this field exists to avoid.
    batch_worker_fn: Optional[Callable] = None
    # EXPERIMENT_LOG's TORAX-on-GPU vmap spike: a genuinely different shape
    # from persistent_worker_fn above -- ONE process scores a whole ROUND
    # at once via jax.vmap (torax_oracle.run_batch_vmap), not N processes
    # each scoring candidates one at a time. Signature:
    # batch_worker_fn(worker_args_list) -> list[(ok, payload)], where
    # worker_args_list is a list of (overrides_dict, n_rho) tuples -- i.e.
    # exactly what params_to_worker_args() already produces per candidate,
    # just batched into a list instead of called one at a time -- aligned
    # 1:1 with worker_args_list's own order. Populated only for domains
    # that actually have a vmappable oracle (today: only torax_oracle.py's
    # experimental.run_loop_jit API) -- a caller should prefer this over
    # persistent_worker_fn when both are set and a GPU is available, since
    # it captures the real GPU win (~150x at batch=256 in the spike) that a
    # persistent one-candidate-at-a-time worker pool cannot.


# ---------------------------------------------------------------------------
# Conditioning -- the piece §27/§28/§29 all flagged as still missing: how a
# domain turns (a requested target vector, some auxiliary discrete/continuous
# state) into the actual tensor a generative model conditions on. VMEC++ uses
# a discrete n_field_periods one-hot; airfoils use continuous
# (log-Reynolds, alpha), z-scored with stats computed at TRAINING time and
# stored in the checkpoint -- which is exactly why this can't be a pure
# DomainSpec-level constant the way target_names/param_dim are: it needs
# access to a loaded checkpoint's own stats. get_conditioning(domain_name,
# ckpt) is the one entry point every script should use instead of each
# hand-rolling its own aux-handling branch (which is what every domain-
# specific eval/train script so far has actually done).
# ---------------------------------------------------------------------------

@dataclass
class Conditioning:
    name: str
    extra_dim: int
    cond_from_target_and_aux: Callable  # (target_z: (n_targets,) ndarray, aux: dict) -> torch.Tensor, shape (1, n_targets+extra_dim)
    aux_from_row: Callable               # (X_row: ndarray, param_dim: int) -> aux dict, reading a real dataset row's own conditioning state
    sample_aux: Callable                 # (rng: np.random.Generator) -> aux dict, for prior-mode sampling with no real anchor
    worker_aux: Callable                 # (aux: dict) -> dict, the physical aux params_to_worker_args expects (identity for VMEC, adds mach=0.0 for airfoil)


def get_conditioning(domain_name, ckpt):
    import torch as _torch

    if domain_name == "vmec":
        from train_vae import NFP_VALUES, nfp_one_hot

        def cond_from_target_and_aux(target_z, aux):
            target_t = _torch.tensor(target_z[None, :], dtype=_torch.float32)
            nfp_t = _torch.tensor([float(aux["nfp"])], dtype=_torch.float32)
            return _torch.cat([target_t, nfp_one_hot(nfp_t)], dim=-1)

        return Conditioning(
            name="vmec", extra_dim=len(NFP_VALUES),
            cond_from_target_and_aux=cond_from_target_and_aux,
            aux_from_row=lambda row, param_dim: {"nfp": int(row[param_dim])},
            sample_aux=lambda rng: {"nfp": int(rng.choice(NFP_VALUES))},
            worker_aux=lambda aux: {"nfp": aux["nfp"]},
        )

    if domain_name == "airfoil":
        re_mean, re_std = ckpt["reynolds_mean"], ckpt["reynolds_std"]
        al_mean, al_std = ckpt["alpha_mean"], ckpt["alpha_std"]

        def cond_from_target_and_aux(target_z, aux):
            target_t = _torch.tensor(target_z[None, :], dtype=_torch.float32)
            aux_t = _torch.tensor([[(np.log(aux["reynolds"]) - re_mean) / re_std,
                                     (aux["alpha"] - al_mean) / al_std]], dtype=_torch.float32)
            return _torch.cat([target_t, aux_t], dim=-1)

        def aux_from_row(row, param_dim):
            return {"reynolds": float(row[param_dim]), "alpha": float(row[param_dim + 1])}

        def sample_aux(rng):
            return {"reynolds": float(np.exp(rng.uniform(np.log(1e5), np.log(1e7)))),
                    "alpha": float(rng.uniform(-5.0, 15.0))}

        return Conditioning(
            name="airfoil", extra_dim=2,
            cond_from_target_and_aux=cond_from_target_and_aux,
            aux_from_row=aux_from_row, sample_aux=sample_aux,
            worker_aux=lambda aux: {"reynolds": aux["reynolds"], "mach": 0.0, "alpha": aux["alpha"]},
        )

    if domain_name == "torax":
        # No discrete or continuous conditioning variable at all here --
        # unlike VMEC's n_field_periods or airfoil's (reynolds, alpha),
        # torax_oracle.py's own docstring is explicit that "everything that
        # varies is already in params," so aux is genuinely empty (extra_dim=0)
        # rather than a Conditioning this domain happens not to use yet.
        def cond_from_target_and_aux(target_z, aux):
            return _torch.tensor(target_z[None, :], dtype=_torch.float32)

        return Conditioning(
            name="torax", extra_dim=0,
            cond_from_target_and_aux=cond_from_target_and_aux,
            aux_from_row=lambda row, param_dim: {},
            sample_aux=lambda rng: {},
            worker_aux=lambda aux: {},
        )

    raise ValueError(f"no Conditioning registered for domain {domain_name!r}")


# ---------------------------------------------------------------------------
# Tasks -- the three things a method can be asked to do against a DomainSpec
# ---------------------------------------------------------------------------

@dataclass
class TargetTask:
    """Inverse design: hit y_star. What eval_cvae_steerability.py's
    "validity at the anchor's own target" half already measures, just not
    packaged as a standalone, serializable task object -- currently the
    target is always "whatever an anchor's real row happens to have,"
    picked inline inside the eval script rather than specified up front."""
    y_star: "np.ndarray"
    aux: dict = field(default_factory=dict)
    tolerance: Optional["np.ndarray"] = None  # per-target absolute tolerance; None = "oracle validity is the only bar"


@dataclass
class NavigateTask:
    """Local design-geometry navigation: from x0, achieve f(x0) + dy.
    What eval_cvae_steerability.py's steerability half (axis-aligned) and
    eval_random_direction_steerability.py (arbitrary direction) already
    measure -- this unifies both into one task shape, since "axis-aligned"
    is just dy with all but one entry zero."""
    x0: Candidate
    y0: "np.ndarray"  # f(x0) -- passed in rather than re-queried, since the eval scripts already compute this as the "baseline" (see eval_cvae_steerability.py's docstring on why it's the anchor's OWN converged-candidate mean, not a re-verified real row)
    dy: "np.ndarray"


@dataclass
class BudgetedDiscoveryTask:
    """NOT implemented by anything in this project yet. Given a goal
    (either a TargetTask or a scalar objective to extremize) and a finite
    credit budget spent against `DomainSpec.fidelities`' costs, reach the
    goal as cheaply as possible -- tests whether a method can decide what
    to evaluate and at what fidelity, not just whether it can generate
    plausible candidates. This is the piece of the user's "Optimization
    Gym" framing with no existing analogue in eval_cvae_steerability.py or
    eval_random_direction_steerability.py; building it means a genuinely
    new harness loop (propose candidate -> pick fidelity -> spend credits
    -> observe -> repeat until budget exhausted or goal met), not an
    extension of the existing validity/steerability one."""
    goal: TargetTask
    budget_credits: float


@dataclass
class TaskResult:
    """Common scoring across all three task types, so a method's numbers
    are comparable across domains without each domain inventing its own
    metric names the way eval_cvae_steerability.py (correct_direction_rate,
    selectivity_ratio) and eval_random_direction_steerability.py (cosine
    similarity) currently do independently."""
    target_error: Optional[float] = None       # None for tasks with no explicit target (open-ended discovery)
    validity_rate: float = 0.0                  # fraction of proposed candidates with valid=True
    oracle_credits_spent: float = 0.0
    n_evaluations: int = 0
    diversity: Optional[float] = None           # mean pairwise distance among proposed candidates -- see audit_novelty_diversity.py, §26, for the only place this is currently measured, and only as a standalone audit, not a task-scoring field
    pareto_improvement: Optional[float] = None  # only meaningful for multi-objective goals; unused by anything built so far


# ---------------------------------------------------------------------------
# Worked adapters: the two domains this project actually has, wrapped --
# every number here is measured, not invented (see EXPERIMENT_LOG for each).
# ---------------------------------------------------------------------------

def vmec_spec():
    import vmec_oracle as m
    return DomainSpec(
        name="vmec_stellarator",
        target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        fidelities=[
            FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=333,
                           measured_from="~5-15s/call through oracle_harness.py, EXPERIMENT_LOG §16-17, normalized to airfoil-low's ~30ms=1 credit"),
            FidelityLevel("medium", m.FIDELITY_PRESETS["medium"], cost_credits=1600,
                           measured_from="~48s/call, EXPERIMENT_LOG §17 (2.8x low fidelity's own measured cost)"),
            FidelityLevel("high", m.FIDELITY_PRESETS["high"], cost_credits=float("nan"),
                           measured_from="never actually called in this project -- FIDELITY_PRESETS has the name, no run has ever used it, cost unmeasured"),
        ],
        worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
        validity_fn=None,  # ok IS the only validity signal VMEC++ has today
        harness_fit="subprocess-per-candidate",
        harness_fit_note="genuinely fits -- VMEC++ can hang on pathological input (EXPERIMENT_LOG §9), so a killable fresh subprocess per candidate is the right shape, not a compromise.",
        sanity_filter=None,  # no post-hoc filter needed beyond the oracle's own ok
        dataset_paths=lambda out_dir: (out_dir / "X.npy", out_dir / "Y.npy"),
    )


def torax_spec():
    """Real, running third domain (EXPERIMENT_LOG §31) -- the actual test of
    this schema against something meaningfully different from VMEC++/
    XFOIL: harness_fit is genuinely "needs-batched-worker" here, not a
    hypothesis the way it still is for the other 7 candidates in
    CANDIDATE_TRIAGE below. `worker_fn` is present for interface parity
    only; the GPU vmap spike found a genuinely better option than the
    persistent CPU worker pool this docstring originally pointed to --
    `batch_worker_fn` (torax_oracle.run_batch_vmap) scores a whole
    bootstrap round in one jax.vmap call instead of one candidate at a
    time, ~150x throughput vs. the subprocess path at batch=256, verified
    bit-exact against the non-batched reference. Kept `persistent_worker_fn`
    below too (still correct, just superseded for bulk generation) since it
    remains the right choice on a CPU-only node with no GPU available."""
    import torax_oracle as m
    return DomainSpec(
        name="torax_tokamak",
        target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        fidelities=[
            FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=164,
                           measured_from="~4.9s cold-compile first call, EXPERIMENT_LOG §31, normalized to airfoil-low's ~30ms=1 credit -- "
                                          "but this number is misleading for either warm harness: a persistent-CPU-worker warm call is "
                                          "~0.13-0.45s (~4-15 credits, 10-38x cheaper than cold), and the GPU vmap batch_worker_fn is "
                                          "~150x cheaper still at batch=256 (~3.5ms/candidate warm) because the compile tax is paid once "
                                          "per (batch_size, max_steps) shape, not once per candidate."),
        ],
        worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
        persistent_worker_fn=m.persistent_worker_fn,
        batch_worker_fn=m.run_batch_vmap,
        validity_fn=None,  # ok (sim_error == NO_ERROR) is the only validity signal built so far
        harness_fit="needs-batched-worker",
        harness_fit_note="confirmed, not hypothesized: same config in the same process runs ~38x faster on repeat calls "
                          "(4.92s -> 0.13s) on CPU, and the GPU vmap path (batch_worker_fn) goes further still -- ~150x vs. "
                          "the subprocess path at batch=256, flat ~0.8-0.9s wall-clock per round from batch=4 through 256 "
                          "(GPU not yet saturated at 256). oracle_harness.py's subprocess-per-candidate pattern would pay the "
                          "full cold-compile cost on every candidate. Prefer batch_worker_fn (torax_oracle.run_batch_vmap) "
                          "when a GPU is available; persistent_worker_fn (oracle_harness_persistent.py) otherwise.",
        # EXPERIMENT_LOG §35/§37: unlike airfoil's cd<1e-6 (a clean bimodal
        # split -- real values 5.8e-6 and up vs. a broken ~1e-12 cluster),
        # the generated dataset's Q_fusion/T_e_volume_avg/H98 tail is a
        # smooth, gapless power-law continuum (checked directly, no natural
        # break found) -- near-marginal-heating configs can genuinely
        # produce a huge but not obviously-nonphysical ratio. No confirmed
        # "this is definitely a bug" line exists the way it did for airfoil,
        # so these are practical caps (~100x the real-reference-device max
        # of §36, not a measured physical ceiling), dropping 190/15,275 rows
        # (1.2%) -- flagged as a placeholder pending real physics judgment,
        # not a confirmed-correct threshold.
        sanity_filter=lambda Y: (Y[:, m.TARGET_NAMES.index("Q_fusion")] <= 300)
                                  & (Y[:, m.TARGET_NAMES.index("T_e_volume_avg")] <= 200)
                                  & (Y[:, m.TARGET_NAMES.index("H98")] <= 20),
        dataset_paths=lambda out_dir: (out_dir / "torax_X.npy", out_dir / "torax_Y.npy"),
    )


def airfoil_spec():
    import airfoil_oracle as m
    return DomainSpec(
        name="xfoil_airfoil",
        target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        fidelities=[
            FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=1,
                           measured_from="~15-45ms/call, EXPERIMENT_LOG §21 -- the cheapest call in this project, hence the credit=1 normalization anchor"),
            # no medium/high tier -- EXPERIMENT_LOG §21's own open item: an OpenFOAM
            # second tier was proposed, never built.
        ],
        worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
        validity_fn=None,  # cd<1e-6 rejection currently lives INSIDE worker_fn (folded into ok), not as a separable validity_fn -- see the DomainSpec docstring above on why that's a real, if minor, schema violation worth fixing if this module is ever wired up for real
        harness_fit="subprocess-per-candidate",
        harness_fit_note="genuinely fits -- XFOIL is fast enough (ms-scale) that process-spawn overhead is a rounding error, not a compromise the way it would be for a JIT-compiled oracle.",
        sanity_filter=lambda Y: (Y[:, m.TARGET_NAMES.index("cd")] >= 1e-6) & (np.abs(Y[:, m.TARGET_NAMES.index("l_over_d")]) <= 300),
        dataset_paths=lambda out_dir: (out_dir / "airfoil_X.npy", out_dir / "airfoil_Y.npy"),
    )


def mug_spec():
    """4th real domain (EXPERIMENT_LOG §46-47) -- the first one that's
    self-written physics rather than a wrapped external solver, and the
    first one deliberately picked FOR being cheap/CPU-only rather than in
    spite of it (§46: walked back §44's own GPU-first framing). validity_fn
    is None for a genuinely different reason than VMEC++'s (mug_oracle.py
    has NO non-convergence failure mode at all -- the FD scheme is
    unconditionally stable -- so `ok` is only ever False on a structurally
    invalid input, not a solver failure)."""
    import mug_oracle as m
    return DomainSpec(
        name="mug_thermal",
        target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        fidelities=[
            FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=1.3,
                           measured_from="~40ms/candidate host CPU single-threaded (§46), normalized to airfoil-low's ~30ms=1 credit"),
        ],
        worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
        validity_fn=None,  # no non-convergence mode exists here -- ok=False means a structurally invalid input, not a solver failure
        harness_fit="subprocess-per-candidate",
        harness_fit_note="genuinely fits, same reasoning as airfoil's -- ~40ms/call, subprocess-spawn overhead isn't the bottleneck.",
        sanity_filter=None,  # no known-bad numerical tail found yet (unlike airfoil's cd<1e-6 or TORAX's Q_fusion tail) -- not yet stress-tested at real dataset scale
        dataset_paths=lambda out_dir: (out_dir / "mug_X.npy", out_dir / "mug_Y.npy"),
    )


# ---------------------------------------------------------------------------
# Triage of the 7 candidate domains proposed against this schema -- answers
# "which of these are genuinely clean vs. forced into the framework"
# directly, using only things confirmed in this project already (the
# TORAX/JAX-JIT-tax finding, EXPERIMENT_LOG §25) plus public information
# about each tool, NOT run/installed here. Treat the "harness_fit" column
# as a hypothesis to confirm the same way TORAX's actually was, not a
# measured fact the way vmec_spec()/airfoil_spec()'s numbers are.
# ---------------------------------------------------------------------------
CANDIDATE_TRIAGE = """
domain              | natural PARAM_DIM | validity != ok?        | harness fit (hypothesis)              | confidence
---------------------|--------------------|-------------------------|-----------------------------------------|------------
Structural/Warp FEM  | yes (mesh/field)  | yes (disconnected/      | NEEDS BATCHED WORKER -- Warp is a JIT-  | medium --
                      |                    | inverted elements are   | compiled framework (CUDA graphs), same  | not run
                      |                    | a real, separate         | JIT-tax shape as TORAX (EXPERIMENT_LOG  | here yet
                      |                    | failure mode from        | §25) is hypothesized, not confirmed --
                      |                    | "solver didn't converge")| needs the same real spike TORAX got
                      |                    |                          | before trusting "GPU: Excellent" means
                      |                    |                          | "fits oracle_harness.py"
Photonics/Meep       | yes (dielectric    | yes (fab constraints    | UNKNOWN -- Meep is FDTD, not JIT-       | low
                      | geometry)          | separate from sim        | compiled the way Warp/JAX-FEM/JAX-      |
                      |                    | convergence)             | Fluids are; per-call cost profile is    |
                      |                    |                          | genuinely unconfirmed either way        |
Wind turbine/WISDEM  | yes (chord/twist/  | yes (structural limits   | LIKELY FITS -- CPU-based multi-         | low-medium
                      | structure vector)  | vs. solver failure)      | disciplinary optimization stack, not    |
                      |                    |                          | JIT-compiled -- probably closer to      |
                      |                    |                          | VMEC++'s subprocess-per-candidate shape |
                      |                    |                          | than to TORAX's, but unconfirmed        |
Battery/PyBaMM       | yes (electrode/    | yes (thermal runaway/    | LIKELY FITS for SPM/SPMe tiers (fast,   | low-medium
                      | material params)   | degradation limits)      | CPU); DFN tier's cost is unconfirmed --
                      |                    |                          | this is the one domain that ALREADY has |
                      |                    |                          | a real, named fidelity ladder (SPM ->   |
                      |                    |                          | SPMe -> DFN -> thermal/degradation) as  |
                      |                    |                          | part of its own design, closest fit to  |
                      |                    |                          | this schema's ladder concept of any     |
                      |                    |                          | candidate on the list                   |
Compressible CFD/     | yes (body/nozzle   | yes                      | NEEDS BATCHED WORKER -- JAX-native,     | medium --
JAX-Fluids            | geometry)          |                          | same hypothesis as Warp/JAX-FEM         | not run
                      |                    |                          |                                          | here yet
Porous/Warp Darcy     | yes (level-set     | yes (disconnected flow   | NEEDS BATCHED WORKER -- same Warp JIT   | medium --
                      | geometry)          | paths)                   | hypothesis as structural/Warp FEM above | not run
                      |                    |                          |                                          | here yet
Spacecraft/Tudat      | yes (trajectory/   | yes (stability/fuel      | LIKELY FITS -- CPU orbital mechanics,   | low
                      | orbit params)      | limits)                  | not JIT-compiled -- probably closest to |
                      |                    |                          | VMEC++'s shape of anything on the list  |
"""

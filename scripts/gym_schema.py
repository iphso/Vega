"""Formal schema for what this project has been informally building since
oracle_base.py: not a collection of physics simulators, but a family of
constrained inverse-design / design-navigation environments sharing one
shape --

    x in X  --oracle(s)-->  y = f(x) in R^m,  with validity v(x) in {0, 1}

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

Four nouns, cleanly separated (this is the refactor over this file's
original DomainSpec-does-everything shape -- see git history if you want
the single-class version):

  Gymbro   -- a generative model paired with the search strategy that
              actually moves through its space (generate a candidate,
              move it toward a direction). optimize.py's raw-coefficient
              Adam+ALM search and latent_optimize.py's VAE-latent Adam+ALM
              search are two DIFFERENT Gymbros against the same Domain,
              not two configurations of one thing.
  Oracle   -- a solver PACKAGE: one or more OracleFunctions sharing one
              output metric namespace, plus that package's own legality
              checks (validity_fn, sanity_filter). VMEC++ and XFOIL are
              each an Oracle with exactly one OracleFunction today (their
              "fidelity" ladder is cost tiers of that SAME function, not
              several functions). The concept exists for an OpenFOAM-
              shaped package that genuinely offers several distinct
              callable solvers (e.g. steady-state vs transient), which
              would be one Oracle with several OracleFunctions -- not
              several Oracles, and not fidelity tiers of one function.
  Domain   -- a design space X (param_dim, zero_indices) plus every Oracle
              that can score a point in it, and what merges their results
              into one scored Candidate. Every domain here today registers
              exactly one Oracle; the concept exists for e.g. a stellarator
              Domain someday running VMEC++ for physics metrics AND a
              separate geometry-validity Oracle side by side.
  Candidate / OracleResult -- unchanged from before: a point in X, and
              what one Oracle call returns for it.

Existing domain modules (vmec_oracle.py, airfoil_oracle.py, torax_oracle.py,
mug_oracle.py, photonics_oracle.py) are NOT rewritten against this file -- they still directly
implement oracle_base.py's worker_fn/params_to_worker_args contract, and
OracleFunction below is a thin wrapper around exactly that contract (see
vmec_domain()/airfoil_domain() etc. at the bottom for the real adapters,
using this project's own already-measured numbers, not invented ones).
A new domain can either implement oracle_base.py's contract directly (as
all four existing ones do) and get wrapped, or build a Domain/Oracle
natively -- both are first-class.

Six other scripts (steerability_generic.py, make_splits_generic.py,
run_steerability_ablation_generic.py, coverage_metric.py,
bootstrap_generic.py, merge_bootstrap_pools.py) already import the old
*_spec() factories and read a flat DomainSpec-shaped attribute surface
(spec.worker_fn, spec.target_names, spec.param_dim, ...) directly. Domain
below keeps that exact surface as pass-through properties onto
oracles[0].functions[0] so none of those six needed to change for this
refactor; the *_spec() names themselves are kept as deprecated aliases of
the new *_domain() factories for the same reason.
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
    """One cost tier of a single OracleFunction. `cost_credits` is
    wall-clock seconds per call, normalized so the CHEAPEST call in this
    whole project (airfoil/XFOIL, ~30ms) is 1 credit -- see the worked
    examples at the bottom for where every number below comes from; none
    are invented. `internal_name` is the domain-specific opaque string
    oracle_base.py's FIDELITY_PRESETS already passes through unexamined."""
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
    `Oracle.validity_fn`, see below)."""
    ok: bool
    valid: bool
    metrics: Optional[dict] = None
    error: Optional[str] = None
    fidelity: Optional[str] = None
    oracle: Optional[str] = None  # which Oracle produced this, once a Domain has more than one
    cost_credits: float = 0.0


class WorkerFn(Protocol):
    def __call__(self, conn, *worker_args) -> None: ...


# ---------------------------------------------------------------------------
# OracleFunction / Oracle -- one callable solver entry point, and the
# package of one-or-more of them that share an output metric namespace.
# ---------------------------------------------------------------------------

@dataclass
class OracleFunction:
    """One callable entry point an Oracle exposes. Everything in this
    project today has exactly one per Oracle -- its `fidelities` ladder is
    cost tiers of that SAME function, not several genuinely different
    functions. An OpenFOAM-shaped package that offers several distinct
    solvers (steady-state vs transient, different physics couplings) as
    separately callable would give its Oracle several OracleFunctions
    instead -- this is the piece that lets that happen without inventing a
    second Oracle or overloading the fidelity ladder to mean something it
    doesn't."""
    name: str
    fidelities: list  # list[FidelityLevel], cheapest first
    worker_fn: WorkerFn
    params_to_worker_args: Callable
    harness_fit: str = "subprocess-per-candidate"  # or "needs-batched-worker"
    harness_fit_note: str = ""
    persistent_worker_fn: Optional[Callable] = None
    # EXPERIMENT_LOG §37: the batched-worker counterpart to `worker_fn`,
    # populated only when harness_fit=="needs-batched-worker" (today: only
    # torax_oracle.persistent_worker_fn). A caller should branch on
    # harness_fit and use oracle_harness_persistent.run_batch_persistent
    # with THIS function instead of oracle_harness.run_batch_with_timeout
    # when it's set -- worker_fn alone still works for any function but
    # pays the full per-candidate cost this field exists to avoid.
    batch_worker_fn: Optional[Callable] = None
    # EXPERIMENT_LOG's TORAX-on-GPU vmap spike: a genuinely different shape
    # from persistent_worker_fn above -- ONE process scores a whole ROUND
    # at once via jax.vmap (torax_oracle.run_batch_vmap), not N processes
    # each scoring candidates one at a time. Signature:
    # batch_worker_fn(worker_args_list) -> list[(ok, payload)], where
    # worker_args_list is a list of (overrides_dict, n_rho) tuples -- i.e.
    # exactly what params_to_worker_args() already produces per candidate,
    # just batched into a list instead of called one at a time -- aligned
    # 1:1 with worker_args_list's own order. Populated only for functions
    # that actually have a vmappable oracle (today: only torax_oracle.py's
    # experimental.run_loop_jit API) -- a caller should prefer this over
    # persistent_worker_fn when both are set and a GPU is available, since
    # it captures the real GPU win (~150x at batch=256 in the spike) that a
    # persistent one-candidate-at-a-time worker pool cannot.


@dataclass
class Oracle:
    """A solver package: one or more OracleFunctions sharing one output
    metric namespace (`target_names`), plus that package's own legality
    checks. `validity_fn`/`sanity_filter` apply to whichever function was
    actually called -- they're properties of the package's metric space,
    not of any one function."""
    name: str
    target_names: list
    log_target_names: list
    functions: list  # list[OracleFunction]
    validity_fn: Optional[Callable[[dict], bool]] = None
    # `validity_fn(metrics_dict) -> bool`, applied on top of `ok=True`
    # results. None means "ok is the only validity signal this oracle has"
    # -- true of both vmec_oracle.py and airfoil_oracle.py's CURRENT ok
    # flag (airfoil's cd<1e-6 check lives inside worker_fn today, not here
    # -- a real domain would want it moved out to this field so `ok` means
    # purely "the solver ran" and `valid` means "the result is legitimate,"
    # not conflated as they are right now).
    sanity_filter: Optional[Callable] = None
    # sanity_filter(Y: np.ndarray) -> bool mask, applied before training/eval
    # on top of "ok" -- e.g. airfoil_oracle's cd<1e-6/l_over_d>300 rejection
    # (EXPERIMENT_LOG §22), currently duplicated as inline code in every
    # airfoil training/eval script rather than expressed once here. None
    # (VMEC++'s case) means no post-hoc filter beyond the oracle's own ok.

    @property
    def default(self) -> OracleFunction:
        """The function to call when a caller hasn't named one -- every
        Oracle in this project has exactly one today, so this is just
        `functions[0]`."""
        return self.functions[0]

    def call(self, params, aux, fidelity_name, function_name=None):
        """Resolve a function (default: the package's only one) and its
        named fidelity, and return the (worker_fn, worker_args) pair a
        harness (oracle_harness.py / oracle_harness_persistent.py) runs.
        Does not itself spawn the subprocess/worker -- that's still the
        harness's job, unchanged."""
        fn = self.default if function_name is None else next(
            f for f in self.functions if f.name == function_name)
        fidelity = next(fl for fl in fn.fidelities if fl.name == fidelity_name)
        return fn.worker_fn, fn.params_to_worker_args(params, aux, fidelity.internal_name)


# ---------------------------------------------------------------------------
# Domain -- a design space plus every Oracle that can score a point in it.
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """A design space X (`param_dim`, `zero_indices`) plus every Oracle
    that can score a point in it. Every domain here today registers
    exactly one Oracle -- the concept exists for e.g. a stellarator Domain
    someday running VMEC++ for physics metrics AND a separate geometry-
    validity Oracle side by side, with Domain being what merges both
    Oracles' results into one scored Candidate instead of a caller
    juggling them by hand."""
    name: str
    param_dim: int
    zero_indices: list
    oracles: list  # list[Oracle]
    dataset_paths: Optional[Callable] = None
    # dataset_paths(out_dir) -> (X_path, Y_path) -- domain dataset file
    # naming isn't uniform today (X.npy/Y.npy for VMEC++ vs.
    # airfoil_X.npy/airfoil_Y.npy for airfoils); this makes that a single
    # lookup instead of a hardcoded path repeated in every script.
    model_dim: Optional[int] = None
    # EXPERIMENT_LOG §63: the width of vector the generative models
    # (CVAE/GAN/diffusion) actually operate over, when it differs from
    # param_dim. None (every domain before mug) means "identical to
    # param_dim" -- every design coordinate in VMEC++/airfoil/TORAX is
    # genuinely continuous, so the model's own working vector already IS
    # the design vector, 1:1. Mug's struct_material_idx/handle_material_idx
    # are categorical (which ONE named material), not continuous, so its
    # generative models' internal vector one-hot-expands those two columns
    # (mug_categorical.MODEL_DIM=42, vs. param_dim=9) -- see
    # mug_categorical.py's module docstring for why.
    from_model_space: Optional[Callable] = None
    # from_model_space(decoded_model_output, coeff_mean, coeff_std) -> real
    # param_dim-width design rows. None (every domain before mug) means the
    # old universal `decoded * coeff_std + coeff_mean` linear destandardize
    # steerability_generic.py always did -- still correct whenever
    # model_dim == param_dim. Mug's version (mug_categorical.
    # from_model_space) destandardizes the continuous columns the same way
    # AND hard-argmaxes each categorical one-hot block back to a single
    # real material class -- the actual fix for a generated design reading
    # as a fictional blended material (e.g. "soda_lime_glass/earthenware")
    # instead of a real, single, nameable one.

    @property
    def oracle(self) -> Oracle:
        """The Oracle to use when a caller hasn't picked one -- every
        Domain here today has exactly one registered, so this is just
        `oracles[0]`. Raises if that's no longer true, rather than
        silently picking one, once a Domain genuinely has several."""
        assert len(self.oracles) == 1, (
            f"{self.name} has {len(self.oracles)} oracles registered -- "
            f"iterate .oracles directly instead of using the single-oracle shortcut")
        return self.oracles[0]

    def evaluate(self, params, aux, fidelity_name="low"):
        """Score one candidate against every registered Oracle, merging
        each Oracle's metrics into a single dict (keyed by oracle name
        once a Domain has more than one). The multi-oracle counterpart to
        calling a single worker_fn directly; with today's always-one-
        Oracle domains this is exactly that Oracle's own result,
        unwrapped -- see the *_domain() adapters below for the real ones."""
        if len(self.oracles) == 1:
            return self.oracles[0].call(params, aux, fidelity_name)
        return {o.name: o.call(params, aux, fidelity_name) for o in self.oracles}

    # -- Flat pass-through surface, kept for the six scripts that already
    # import a *_spec() factory and read spec.worker_fn / spec.target_names
    # / spec.param_dim / etc directly (steerability_generic.py,
    # make_splits_generic.py, run_steerability_ablation_generic.py,
    # coverage_metric.py, bootstrap_generic.py, merge_bootstrap_pools.py).
    # None of them needed to change for this refactor; new code should
    # prefer domain.oracle.default.worker_fn etc. so it's explicit which
    # Oracle/OracleFunction is being read.
    @property
    def target_names(self): return self.oracle.target_names

    @property
    def log_target_names(self): return self.oracle.log_target_names

    @property
    def fidelities(self): return self.oracle.default.fidelities

    @property
    def worker_fn(self): return self.oracle.default.worker_fn

    @property
    def params_to_worker_args(self): return self.oracle.default.params_to_worker_args

    @property
    def harness_fit(self): return self.oracle.default.harness_fit

    @property
    def harness_fit_note(self): return self.oracle.default.harness_fit_note

    @property
    def persistent_worker_fn(self): return self.oracle.default.persistent_worker_fn

    @property
    def batch_worker_fn(self): return self.oracle.default.batch_worker_fn

    @property
    def validity_fn(self): return self.oracle.validity_fn

    @property
    def sanity_filter(self): return self.oracle.sanity_filter


# Deprecated alias -- every caller of this module was written against the
# name "DomainSpec" before this refactor split it into Domain/Oracle/
# OracleFunction. Domain's flat pass-through properties above make it a
# drop-in replacement, so this alias is the whole migration path.
DomainSpec = Domain


# ---------------------------------------------------------------------------
# Gymbro -- a generative model paired with the search strategy that
# actually moves through its space. The piece this schema had no name for
# before: optimize.py's raw-coefficient Adam+ALM search and
# latent_optimize.py's VAE-latent Adam+ALM search are two DIFFERENT
# Gymbros against the SAME Domain (vmec_stellarator), not two
# configurations of one search script.
# ---------------------------------------------------------------------------

@dataclass
class Gymbro:
    """A generative model + the search strategy that moves through its
    space. `generate` is Task 1's (Target) machinery: produce a fresh
    candidate, optionally steered toward a target. `move` is Task 2's
    (Navigate): given a candidate and a desired metric delta, produce a
    nearby one. Both are optional -- a Gymbro with no standalone prior-
    sampling mode leaves `generate` unset, same for a pure sampler with no
    direction-following.

    NOT wired up as literal callables here for the two real Gymbros below
    -- optimize.py and latent_optimize.py implement this shape as inline
    CLI flow (argparse -> load frozen ensemble -> ALM loop over Adam
    steps), not as an importable generate()/move() pair. GYMBROS below
    documents the two concrete instances of the concept the same way
    vmec_domain() etc. are the real Oracle/Domain adapters -- turning
    `generate`/`move` into real callables (so a caller could invoke a
    Gymbro without shelling out to a CLI script) is future work, not done
    by this refactor."""
    name: str
    domain: str  # which Domain.name this Gymbro searches against
    search_space: str  # "raw_coefficients" | "vae_latent"
    search_space_dim: Optional[int] = None  # None when it's just the domain's own param_dim
    surrogate_tags: list = field(default_factory=list)  # frozen ensemble member checkpoints optimized against
    validates_against_oracle: bool = False  # True: top-k re-checked for real before reporting a winner (latent_optimize.py); False: surrogate-feasible is the final answer (optimize.py)
    generate: Optional[Callable] = None
    move: Optional[Callable] = None
    script: str = ""  # where this Gymbro's actual loop lives today


GYMBROS = {
    "coeff_alm": Gymbro(
        name="coeff_alm", domain="vmec_stellarator", search_space="raw_coefficients",
        search_space_dim=90,
        surrogate_tags=["reg_mlp_big_soap_s0", "reg_mlp_big_soap_s1", "reg_mlp_big_soap_s2"],
        validates_against_oracle=False,
        script="optimize.py",
    ),
    "latent_alm": Gymbro(
        name="latent_alm", domain="vmec_stellarator", search_space="vae_latent",
        search_space_dim=None,  # VAE latent_dim, read off the auto-detected bootstrap checkpoint at run time
        surrogate_tags=["reg_mlp_big_soap_s0", "reg_mlp_big_soap_s1", "reg_mlp_big_soap_s2"],
        validates_against_oracle=True,
        script="latent_optimize.py",
    ),
}


# ---------------------------------------------------------------------------
# Conditioning -- the piece §27/§28/§29 all flagged as still missing: how a
# domain turns (a requested target vector, some auxiliary discrete/continuous
# state) into the actual tensor a generative model conditions on. VMEC++ uses
# a discrete n_field_periods one-hot; airfoils use continuous
# (log-Reynolds, alpha), z-scored with stats computed at TRAINING time and
# stored in the checkpoint -- which is exactly why this can't be a pure
# Domain-level constant the way target_names/param_dim are: it needs
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

    if domain_name == "mug":
        # No discrete or continuous conditioning variable, same reasoning as
        # torax below -- mug_oracle.py's own docstring is explicit that
        # every design variable (including the 3 continuous material-choice
        # indices) is already part of the 8-dim param vector.
        def cond_from_target_and_aux(target_z, aux):
            return _torch.tensor(target_z[None, :], dtype=_torch.float32)

        return Conditioning(
            name="mug", extra_dim=0,
            cond_from_target_and_aux=cond_from_target_and_aux,
            aux_from_row=lambda row, param_dim: {},
            sample_aux=lambda rng: {},
            worker_aux=lambda aux: {},
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
# Tasks -- the three things a method can be asked to do against a Domain
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
    credit budget spent against a Domain's Oracles' fidelity costs, reach
    the goal as cheaply as possible -- tests whether a method can decide
    what to evaluate and at what fidelity, not just whether it can
    generate plausible candidates. This is the piece of the user's
    "Optimization Gym" framing with no existing analogue in
    eval_cvae_steerability.py or eval_random_direction_steerability.py;
    building it means a genuinely new harness loop (propose candidate ->
    pick fidelity -> spend credits -> observe -> repeat until budget
    exhausted or goal met), not an extension of the existing
    validity/steerability one."""
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
# Worked adapters: the four domains this project actually has, wrapped --
# every number here is measured, not invented (see EXPERIMENT_LOG for each).
# Each wraps exactly one Oracle with exactly one OracleFunction; see the
# module docstring for why the extra structure exists even though nothing
# here uses more than one of either yet.
# ---------------------------------------------------------------------------

def vmec_domain():
    import vmec_oracle as m
    return Domain(
        name="vmec_stellarator",
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        oracles=[Oracle(
            name="vmec++",
            target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
            functions=[OracleFunction(
                name="vmec++",
                fidelities=[
                    FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=333,
                                   measured_from="~5-15s/call through oracle_harness.py, EXPERIMENT_LOG §16-17, normalized to airfoil-low's ~30ms=1 credit"),
                    FidelityLevel("medium", m.FIDELITY_PRESETS["medium"], cost_credits=1600,
                                   measured_from="~48s/call, EXPERIMENT_LOG §17 (2.8x low fidelity's own measured cost)"),
                    FidelityLevel("high", m.FIDELITY_PRESETS["high"], cost_credits=1800,
                                   measured_from="~54s/call (p1_alm_validate.py, 2026-08-30) -- first time this project has ever "
                                                  "actually called high_fidelity; a SINGLE measurement, not averaged over many calls "
                                                  "like low/medium's own numbers, so treat this as a rough estimate pending a real bulk "
                                                  "measurement. Notably NOT the order-of-magnitude-slower cost one might expect from "
                                                  "an untested tier -- comparable to medium's own ~48s."),
                ],
                worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
                harness_fit="subprocess-per-candidate",
                harness_fit_note="genuinely fits -- VMEC++ can hang on pathological input (EXPERIMENT_LOG §9), so a killable fresh subprocess per candidate is the right shape, not a compromise.",
            )],
            validity_fn=None,  # ok IS the only validity signal VMEC++ has today
            sanity_filter=None,  # no post-hoc filter needed beyond the oracle's own ok
        )],
        dataset_paths=lambda out_dir: (out_dir / "X.npy", out_dir / "Y.npy"),
    )


def torax_domain():
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
    return Domain(
        name="torax_tokamak",
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        oracles=[Oracle(
            name="torax",
            target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
            functions=[OracleFunction(
                name="torax",
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
                harness_fit="needs-batched-worker",
                harness_fit_note="confirmed, not hypothesized: same config in the same process runs ~38x faster on repeat calls "
                                  "(4.92s -> 0.13s) on CPU, and the GPU vmap path (batch_worker_fn) goes further still -- ~150x vs. "
                                  "the subprocess path at batch=256, flat ~0.8-0.9s wall-clock per round from batch=4 through 256 "
                                  "(GPU not yet saturated at 256). oracle_harness.py's subprocess-per-candidate pattern would pay the "
                                  "full cold-compile cost on every candidate. Prefer batch_worker_fn (torax_oracle.run_batch_vmap) "
                                  "when a GPU is available; persistent_worker_fn (oracle_harness_persistent.py) otherwise.",
            )],
            validity_fn=None,  # ok (sim_error == NO_ERROR) is the only validity signal built so far
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
        )],
        dataset_paths=lambda out_dir: (out_dir / "torax_X.npy", out_dir / "torax_Y.npy"),
    )


def airfoil_domain():
    import airfoil_oracle as m
    return Domain(
        name="xfoil_airfoil",
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        oracles=[Oracle(
            name="xfoil",
            target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
            functions=[OracleFunction(
                name="xfoil",
                fidelities=[
                    FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=1,
                                   measured_from="~15-45ms/call, EXPERIMENT_LOG §21 -- the cheapest call in this project, hence the credit=1 normalization anchor"),
                    # no medium/high tier -- EXPERIMENT_LOG §21's own open item: an OpenFOAM
                    # second tier was proposed, never built. (That tier, if built, is exactly
                    # the "several OracleFunctions on one Oracle" case this schema now has
                    # room for -- OpenFOAM's own several solvers, not another fidelity rung.)
                ],
                worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
                harness_fit="subprocess-per-candidate",
                harness_fit_note="genuinely fits -- XFOIL is fast enough (ms-scale) that process-spawn overhead is a rounding error, not a compromise the way it would be for a JIT-compiled oracle.",
            )],
            validity_fn=None,  # cd<1e-6 rejection currently lives INSIDE worker_fn (folded into ok), not as a separable validity_fn -- see the Oracle docstring above on why that's a real, if minor, schema violation worth fixing if this module is ever wired up for real
            sanity_filter=lambda Y: (Y[:, m.TARGET_NAMES.index("cd")] >= 1e-6) & (np.abs(Y[:, m.TARGET_NAMES.index("l_over_d")]) <= 300),
        )],
        dataset_paths=lambda out_dir: (out_dir / "airfoil_X.npy", out_dir / "airfoil_Y.npy"),
    )


def mug_domain():
    """4th real domain (EXPERIMENT_LOG §46-48) -- the first one that's
    self-written physics rather than a wrapped external solver, and the
    first one deliberately picked FOR being cheap/CPU-only rather than in
    spite of it (§46: walked back §44's own GPU-first framing). validity_fn
    is None for a genuinely different reason than VMEC++'s (mug_oracle.py
    has NO non-convergence failure mode at all -- the FD scheme is
    unconditionally stable -- so `ok` is only ever False on a structurally
    invalid input, not a solver failure). §48: PARAM_DIM went 3->8 (named
    materials + a 2-band wall profile + a real handle) after direct user
    request for a richer parameterization -- pulled from mug_oracle.py
    directly, not hardcoded, so this adapter didn't need to change.
    (Note: the in-progress mug_categorical/model_dim generalization -- a
    one-hot categorical-material encoding for the generative models -- is
    not yet committed on this branch; this adapter still matches the
    already-shipped mug_oracle.py contract.)"""
    import mug_oracle as m
    return Domain(
        name="mug_thermal",
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        oracles=[Oracle(
            name="mug_fd",
            target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
            functions=[OracleFunction(
                name="mug_fd",
                fidelities=[
                    FidelityLevel("low", m.FIDELITY_PRESETS["low"], cost_credits=7.6,
                                   measured_from="~227ms/candidate host CPU single-threaded (§48, v2's 3 coupled chains vs. v1's 1), normalized to airfoil-low's ~30ms=1 credit"),
                ],
                worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
                harness_fit="subprocess-per-candidate",
                harness_fit_note="genuinely fits, same reasoning as airfoil's -- subprocess-spawn overhead isn't the bottleneck even at v2's ~227ms/call.",
            )],
            validity_fn=None,  # no non-convergence mode exists here -- ok=False means a structurally invalid input, not a solver failure
            sanity_filter=None,  # no known-bad numerical tail found yet (unlike airfoil's cd<1e-6 or TORAX's Q_fusion tail) -- not yet stress-tested at real dataset scale
        )],
        dataset_paths=lambda out_dir: (out_dir / "mug_X.npy", out_dir / "mug_Y.npy"),
    )


# Real measured numbers (smoke_photonics_oracle.py, 3 runs, host CPU,
# `meep` compose service) -- NOT invented, same convention as every other
# FidelityLevel.measured_from in this file. Each candidate pays for TWO
# real MEEP runs (calibration + grating, see photonics_oracle.py's
# docstring), already included in these per-candidate numbers.
PHOTONICS_LOW_COST_CREDITS = 43
PHOTONICS_LOW_COST_PROVENANCE = ("~1.26-1.31s/candidate (resolution=10 px/um, 2 MEEP runs), "
                                  "smoke_photonics_oracle.py, normalized to airfoil-low's ~30ms=1 credit")
PHOTONICS_MEDIUM_COST_CREDITS = 123
PHOTONICS_MEDIUM_COST_PROVENANCE = ("~3.60-3.81s/candidate (resolution=20 px/um, 2 MEEP runs), "
                                     "smoke_photonics_oracle.py, normalized the same way -- ~2.9x low's own "
                                     "cost for 2x the resolution in each of 2 dimensions, i.e. sub-quadratic "
                                     "in practice for this cell size, not the naive 4x")


def photonics_domain():
    """6th real domain, and the first answer to this file's own
    CANDIDATE_TRIAGE row below ("Photonics/Meep... harness fit: UNKNOWN...
    per-call cost profile is genuinely unconfirmed either way") -- a real
    2D silicon grating-coupler design task, run through actual conda-forge
    MEEP FDTD (Dockerfile.meep/`meep` compose service), not a self-written
    EM approximation -- same "use the real accepted solver" choice as
    mug's v1->v2 pivot onto OpenFOAM, made up front here rather than after
    a self-written version is built and later superseded.

    harness_fit="subprocess-per-candidate": measured directly (not
    hypothesized the way TORAX's initially was) -- a single candidate's two
    required MEEP runs (calibration + grating, see photonics_oracle.py's
    docstring on why both are needed) show no meaningful warm/cold repeat-
    call speedup the way TORAX's JIT compile tax did, so there's no
    "persistent worker" win to chase here; a fresh subprocess per candidate
    is the right shape, same reasoning as VMEC++/XFOIL/mug.

    validity_fn=None for the same reason as mug's: no FDTD non-convergence
    mode exists (a fixed simulated-time run always completes), so `ok` is
    only ever False on a structurally invalid design (see
    photonics_oracle._validate) -- not a solver failure the way VMEC++'s
    `ok` can be."""
    import photonics_oracle as m
    return Domain(
        name="photonics_grating",
        param_dim=m.PARAM_DIM, zero_indices=m.ZERO_INDICES,
        oracles=[Oracle(
            name="meep_fdtd",
            target_names=m.TARGET_NAMES, log_target_names=m.LOG_TARGET_NAMES,
            functions=[OracleFunction(
                name="meep_fdtd",
                fidelities=[
                    FidelityLevel("low", "low", cost_credits=PHOTONICS_LOW_COST_CREDITS,
                                   measured_from=PHOTONICS_LOW_COST_PROVENANCE),
                    FidelityLevel("medium", "medium", cost_credits=PHOTONICS_MEDIUM_COST_CREDITS,
                                   measured_from=PHOTONICS_MEDIUM_COST_PROVENANCE),
                ],
                worker_fn=m.worker_fn, params_to_worker_args=m.params_to_worker_args,
                harness_fit="subprocess-per-candidate",
                harness_fit_note="measured, not hypothesized (unlike the other 7 CANDIDATE_TRIAGE rows below, "
                                  "still unconfirmed) -- see this function's own docstring.",
            )],
            validity_fn=None,  # no non-convergence mode; ok=False only on a structurally invalid design (see photonics_oracle._validate)
            sanity_filter=None,  # no known-bad numerical tail found yet -- not stress-tested at real dataset scale
        )],
        dataset_paths=lambda out_dir: (out_dir / "photonics_X.npy", out_dir / "photonics_Y.npy"),
    )


# Deprecated aliases -- kept so the six scripts that already
# `from gym_schema import vmec_spec, airfoil_spec, torax_spec, mug_spec`
# don't need to change for this refactor. Prefer the *_domain() names above
# in new code.
vmec_spec = vmec_domain
torax_spec = torax_domain
airfoil_spec = airfoil_domain
mug_spec = mug_domain


# ---------------------------------------------------------------------------
# Triage of the 7 candidate domains proposed against this schema -- answers
# "which of these are genuinely clean vs. forced into the framework"
# directly, using only things confirmed in this project already (the
# TORAX/JAX-JIT-tax finding, EXPERIMENT_LOG §25) plus public information
# about each tool, NOT run/installed here. Treat the "harness_fit" column
# as a hypothesis to confirm the same way TORAX's actually was, not a
# measured fact the way vmec_domain()/airfoil_domain()'s numbers are.
# ---------------------------------------------------------------------------
CANDIDATE_TRIAGE = """
domain              | natural PARAM_DIM | validity != ok?        | harness fit (hypothesis)              | confidence
---------------------|--------------------|-------------------------|-----------------------------------------|------------
Structural/Warp FEM  | yes (density field | yes (disconnected/      | UNKNOWN -- Warp JIT-compile cost         | low-medium
                      | or shape params)   | inverted elements)      | unconfirmed, same open question as       | -- attempted
                      |                    |                         | JAX-Fluids/Warp-Darcy below. A self-     | and fully
                      |                    |                         | written CPU plane-stress SIMP version    | reverted;
                      |                    |                         | was built and fully removed (single-     | see
                      |                    |                         | fixed-edge cantilever domain, not a      | EXPERIMENT_LOG
                      |                    |                         | real two-mount-point bracket envelope --  | for why
                      |                    |                         | user judgment on both the domain shape   |
                      |                    |                         | and the resulting viewer). Warp/GPU/3D   |
                      |                    |                         | remains unconfirmed for any future,      |
                      |                    |                         | better-scoped attempt at this row.       |
Photonics/Meep       | BUILT -- see        | ok=False only on a      | CONFIRMED subprocess-per-candidate --    | high --
                      | photonics_domain() | structurally invalid    | no JIT tax the way JAX/Warp domains      | built, run,
                      | above (PARAM_DIM=6)| design (photonics_      | have; low=~43 credits (~1.3s/candidate, | measured
                      |                    | oracle._validate), no   | 2 MEEP runs), medium=~123 credits       |
                      |                    | non-convergence mode    | (~3.7s/candidate), smoke_photonics_     |
                      |                    | (a fixed-simulated-time | oracle.py -- real numbers, not a        |
                      |                    | FDTD run always         | hypothesis, unlike every other row      |
                      |                    | completes)              | below.                                  |
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
                      |                    |                          | part of its own design, and (via that   |
                      |                    |                          | ladder) the closest existing analogue   |
                      |                    |                          | to a genuinely multi-OracleFunction      |
                      |                    |                          | Oracle of anything on the list          |
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

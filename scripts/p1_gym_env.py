"""P1BudgetedDiscoveryEnv: a real gymnasium.Env implementing gym_schema.py's
BudgetedDiscoveryTask -- that dataclass's own docstring calls this "the
piece of the user's 'Optimization Gym' framing with no existing analogue...
building it means a genuinely new harness loop (propose candidate -> pick
fidelity -> spend credits -> observe -> repeat until budget exhausted or
goal met)". This is that loop, targeting the P1 (GeometricalProblem) region
this session already searched extensively by hand (see EXPERIMENT_LOG-style
history: GAN/cVAE/diffusion sampling techniques plateaued at best worst-
violation ~0.155-0.41 across ~24,000 real oracle calls total and never
reached feasibility, while a gradient-ALM search against a frozen surrogate
found genuinely feasible designs in ~50-150 oracle calls, 3/3 seeds).

Action = Dict(propose: Discrete(7), fidelity: Discrete(2)):
  `propose` chooses WHICH technique produces the next candidate -- an RL
  policy learning to SEQUENCE/MIX the toolkit this session already built and
  compared, not learning raw stellarator design from scratch:
    0 gan_fixed / 1 cvae_fixed / 2 diffusion_fixed  -- condition directly at
      the P1 target point (phase 1's technique, this session's own best for
      cVAE specifically).
    3 gan_tri / 4 cvae_tri / 5 diffusion_tri -- push from the CURRENT BEST
      candidate's own measured target, directed toward lower
      average_triangularity (this session's own best directed technique for
      GAN/diffusion specifically -- but actively hurts cVAE, a real
      per-architecture asymmetry a policy needs to learn, not assume).
    6 gradient_alm -- runs generate_candidates.py (optimize.py's exact ALM
      machinery, unchanged) for a small number of starts/outer-iters against
      the frozen surrogate ensemble, ZERO real oracle cost for the search
      itself, returns its single best surrogate-favored candidate. Reusing
      that script directly (subprocess call, reading its JSONL output)
      rather than re-deriving DualPathMLP's own feature-preprocessing
      pipeline by hand here -- getting that silently wrong would produce
      confidently-garbage proposals, and the script already has it right.
  `fidelity` chooses which REAL oracle tier to spend credits validating the
  proposed candidate at: 0=low (333 credits, gym_schema.vmec_spec()'s own
  measured ~5-15s/call), 1=medium (1600 credits, ~48s/call). "high" is
  excluded -- gym_schema.vmec_spec() records its cost as NaN (never
  actually run in this project), unusable in a budget model.

Observation (14-dim float32): the current best-so-far candidate's own
measured metrics (11, z-scored the same way build_fixed_target_z does),
its worst P1 constraint violation, fraction of credit budget spent, fraction
of max_steps used.

Reward: improvement in best-so-far worst-violation since the last step
(positive = genuinely closer to feasible) minus a small per-credit cost
(so two policies reaching the same result don't score equally if one spent
more budget), plus a one-time terminal bonus on reaching real P1
feasibility. A wasted/non-converging oracle call still costs its credits
and yields no violation improvement -- exactly like a real budget.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent))
from eval_cvae_steerability import load_generative_model
from bootstrap_generic import build_fixed_target_z, build_direction_z
import vmec_oracle as oracle
from oracle_harness import run_batch_with_timeout
from p1_report import p1_violations, TOL

SCRIPTS_DIR = Path(__file__).parent
OUT_DIR = Path("/work/output")

FIXED_TARGET_OVERRIDES = {
    "aspect_ratio": 3.0, "average_triangularity": -0.6,
    "edge_rotational_transform_over_n_field_periods": 0.45, "max_elongation": 2.0,
}
FIXED_NFP = 3

PROPOSERS = ["gan_fixed", "cvae_fixed", "diffusion_fixed", "gan_tri", "cvae_tri", "diffusion_tri", "gradient_alm"]
GENERATOR_NAMES = {"gan_fixed": "gan", "cvae_fixed": "cvae", "diffusion_fixed": "diffusion",
                    "gan_tri": "gan", "cvae_tri": "cvae", "diffusion_tri": "diffusion"}
FIDELITY_NAMES = ["low", "medium", "high"]
# gym_schema.vmec_spec()'s own measured numbers -- not invented here. "high"
# was unmeasured (NaN) until 2026-08-30's p1_alm_validate.py run -- a single
# ~54s call, comparable to medium's own ~48s rather than an order of
# magnitude worse, so it's now usable in a budget model (see gym_schema.py's
# own updated comment for the caveat: single-measurement estimate).
FIDELITY_CREDITS = {"low": 333, "medium": 1600, "high": 1800}
FIDELITY_INTERNAL = {"low": oracle.FIDELITY_PRESETS["low"], "medium": oracle.FIDELITY_PRESETS["medium"],
                      "high": oracle.FIDELITY_PRESETS["high"]}

CREDIT_COST_SCALE = 1e-4  # a low-fidelity call (333 credits) costs 0.033 reward, small vs a real violation step
FEASIBLE_BONUS = 5.0
WORST_VIOLATION_SENTINEL = 10.0  # "nothing evaluated yet" placeholder, well above any real violation seen this session


class P1BudgetedDiscoveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, budget_credits: float = 6000.0, max_steps: int = 30,
                 timeout_seconds: float = 120.0, alm_n_starts: int = 32, seed: int | None = None):
        # 120s default (not 45s): medium (~48s measured) and high (~54s
        # measured) fidelity calls would otherwise systematically time out
        # against a 45s ceiling -- caught by actually exercising the high-
        # fidelity action end to end, not by inspection (it returned
        # ok=False/ate its full credits for nothing on the very first real
        # test). 45s only ever worked because every prior use of this
        # timeout in this session was low-fidelity-only.
        super().__init__()
        self.budget_credits = budget_credits
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.alm_n_starts = alm_n_starts

        self.action_space = spaces.Dict({
            "propose": spaces.Discrete(len(PROPOSERS)),
            "fidelity": spaces.Discrete(len(FIDELITY_NAMES)),
        })
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self._load_generators()
        self._np_rng = np.random.default_rng(seed)

    def _load_generators(self):
        dev = torch.device("cpu")
        self._bundles = {}
        for name, tag in [("gan", "gan_targets_full_s0"), ("cvae", "cvae_targets_full_s0"),
                           ("diffusion", "diffusion_targets_full_s0")]:
            sample_fn, target_names, n_targets, coeff_mean, coeff_std, t_mean, t_std = \
                load_generative_model(name, tag, dev)
            self._bundles[name] = dict(sample_fn=sample_fn, coeff_mean=coeff_mean, coeff_std=coeff_std)
        self.target_names = target_names  # same order for every generator (shared oracle/target_names.json)
        self.n_targets = n_targets
        self.t_mean, self.t_std = t_mean, t_std
        self.fixed_target_z = build_fixed_target_z(
            FIXED_TARGET_OVERRIDES, self.target_names, oracle.LOG_TARGET_NAMES, t_mean, t_std)
        self.tri_direction_z = build_direction_z({"average_triangularity": -1.0}, self.target_names)
        self._elong_i = self.target_names.index("max_elongation")

    def _to_z(self, y_raw: np.ndarray) -> np.ndarray:
        y = y_raw.copy().astype(np.float64)
        for name in oracle.LOG_TARGET_NAMES:
            idx = self.target_names.index(name)
            y[idx] = np.log(np.clip(y[idx], 1e-12, None))
        return (y - np.asarray(self.t_mean)) / np.asarray(self.t_std)

    def _decode_one(self, generator: str, target_z: np.ndarray, nfp: int) -> np.ndarray:
        b = self._bundles[generator]
        cond = torch.cat([
            torch.tensor(target_z[None, :], dtype=torch.float32),
            _nfp_one_hot(nfp),
        ], dim=-1)
        decoded = b["sample_fn"](cond, 1)
        params = decoded[0] * np.asarray(b["coeff_std"]) + np.asarray(b["coeff_mean"])
        params[oracle.ZERO_INDICES] = 0.0
        return params.astype(np.float64)

    def _propose_gradient_alm(self) -> np.ndarray:
        save_path = OUT_DIR / f"_gym_env_alm_scratch_{self._np_rng.integers(1_000_000)}.jsonl"
        cmd = [
            "python3", str(SCRIPTS_DIR / "generate_candidates.py"),
            "--minimize", "max_elongation",
            "--constraint", "aspect_ratio<=4.0",
            "--constraint", "average_triangularity<=-0.5",
            "--constraint", "abs(edge_rotational_transform_over_n_field_periods)>=0.3",
            "--nfp", str(FIXED_NFP),
            "--n-starts", str(self.alm_n_starts), "--n-candidates", "1",
            "--seed", str(int(self._np_rng.integers(1_000_000))),
            "--save", str(save_path), "--device", "cpu",
        ]
        subprocess.run(cmd, cwd=str(SCRIPTS_DIR), capture_output=True, timeout=300, check=False)
        if not save_path.exists():
            # No surrogate-feasible candidate found this call -- fall back to the
            # fixed-target technique rather than crash the episode over it.
            return self._decode_one("cvae", self.fixed_target_z, FIXED_NFP)
        line = save_path.read_text().splitlines()[0]
        save_path.unlink()
        design = json.loads(line)
        params = np.concatenate([design["r_cos"], design["z_sin"]]).astype(np.float64)
        return params

    def _propose(self, proposer: str, best_y_raw: np.ndarray | None) -> np.ndarray:
        if proposer == "gradient_alm":
            return self._propose_gradient_alm()
        if proposer.endswith("_fixed"):
            target_z = self.fixed_target_z
        else:  # "_tri"
            anchor_z = self._to_z(best_y_raw) if best_y_raw is not None else self.fixed_target_z
            target_z = anchor_z + 1.0 * self.tri_direction_z  # step_std=1.0, matches this session's own phase-2 runs
        return self._decode_one(GENERATOR_NAMES[proposer], target_z, FIXED_NFP)

    def _evaluate(self, params: np.ndarray, fidelity: str):
        r_cos, z_sin = params[:45].reshape(5, 9), params[45:90].reshape(5, 9)
        job = (0, r_cos, z_sin, FIXED_NFP, FIDELITY_INTERNAL[fidelity])
        _, ok, payload = next(iter(run_batch_with_timeout([job], oracle.worker_fn, 1, self.timeout_seconds)))
        if not ok:
            return None
        y = np.array([payload.get(name) for name in self.target_names], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            return None
        return y.astype(np.float32)

    def _obs(self):
        y_z = self._to_z(self.best_y) if self.best_y is not None else np.zeros(self.n_targets)
        return np.concatenate([
            y_z.astype(np.float32),
            [self.best_worst_violation, self.credits_spent / self.budget_credits, self.step_count / self.max_steps],
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        self.best_y = None
        self.best_worst_violation = WORST_VIOLATION_SENTINEL
        self.credits_spent = 0.0
        self.step_count = 0
        self.n_evaluations = 0
        self.n_valid = 0
        return self._obs(), {}

    def step(self, action):
        proposer = PROPOSERS[int(action["propose"])]
        fidelity = FIDELITY_NAMES[int(action["fidelity"])]

        params = self._propose(proposer, self.best_y)
        y = self._evaluate(params, fidelity)

        self.credits_spent += FIDELITY_CREDITS[fidelity]
        self.step_count += 1
        self.n_evaluations += 1

        old_worst = self.best_worst_violation
        newly_feasible = False
        if y is not None:
            self.n_valid += 1
            worst = float(p1_violations(y[None, :], self.target_names).max())
            if worst < self.best_worst_violation:
                if worst <= TOL and self.best_worst_violation > TOL:
                    newly_feasible = True
                self.best_worst_violation = worst
                self.best_y = y

        reward = (old_worst - self.best_worst_violation) - CREDIT_COST_SCALE * FIDELITY_CREDITS[fidelity]
        if newly_feasible:
            reward += FEASIBLE_BONUS

        terminated = self.best_worst_violation <= TOL
        truncated = (self.credits_spent >= self.budget_credits) or (self.step_count >= self.max_steps)

        info = {
            "proposer": proposer, "fidelity": fidelity, "ok": y is not None,
            "worst_violation": self.best_worst_violation,
            "oracle_credits_spent": self.credits_spent,
            "n_evaluations": self.n_evaluations,
            "validity_rate": self.n_valid / max(self.n_evaluations, 1),
        }
        return self._obs(), reward, terminated, truncated, info


def _nfp_one_hot(nfp: int) -> torch.Tensor:
    from train_vae import NFP_VALUES, nfp_one_hot
    return nfp_one_hot(torch.tensor([float(nfp)], dtype=torch.float32))

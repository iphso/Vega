"""Smoke test for p1_gym_env.py: construct the env, run a handful of
random-policy steps, confirm shapes/types are gymnasium-correct and that
credits/violations/info move the way they should -- before trusting it for
anything longer (an actual RL training run), same discipline as this whole
session's other smoke-tests-before-launching.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from p1_gym_env import P1BudgetedDiscoveryEnv

env = P1BudgetedDiscoveryEnv(budget_credits=3000, max_steps=6, seed=0)
obs, info = env.reset(seed=0)
print(f"reset OK: obs.shape={obs.shape} dtype={obs.dtype} finite={np.isfinite(obs).all()}")
assert obs.shape == (14,)
assert env.observation_space.contains(obs)

rng = np.random.default_rng(0)
total_reward = 0.0
for i in range(6):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"step {i}: proposer={info['proposer']:14s} fidelity={info['fidelity']:6s} ok={info['ok']!s:5s} "
          f"worst_violation={info['worst_violation']:.4f} credits={info['oracle_credits_spent']:.0f} "
          f"reward={reward:+.4f} terminated={terminated} truncated={truncated}")
    assert env.observation_space.contains(obs), f"obs out of space at step {i}: {obs}"
    assert np.isfinite(obs).all()
    if terminated or truncated:
        break

print(f"\ntotal_reward={total_reward:.4f}  final worst_violation={info['worst_violation']:.4f}  "
      f"credits_spent={info['oracle_credits_spent']:.0f}/{env.budget_credits}  "
      f"n_evaluations={info['n_evaluations']}  validity_rate={info['validity_rate']:.2f}")
print("\nALL CHECKS PASSED")

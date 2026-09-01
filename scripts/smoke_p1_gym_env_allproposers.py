import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from p1_gym_env import P1BudgetedDiscoveryEnv, PROPOSERS

env = P1BudgetedDiscoveryEnv(budget_credits=1e9, max_steps=100, alm_n_starts=8, seed=0)
env.reset(seed=0)

for i, proposer in enumerate(PROPOSERS):
    action = {"propose": i, "fidelity": 0}  # low fidelity throughout, cheapest/fastest
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"proposer={proposer:14s} ok={info['ok']!s:5s} worst_violation={info['worst_violation']:.4f} "
          f"reward={reward:+.4f} terminated={terminated}")

print("\nALL 7 PROPOSERS EXERCISED WITHOUT CRASHING")

"""Direct single-candidate smoke test for photonics_oracle.py -- run inside
the `meep` compose service (real conda-forge MEEP, not importable on the
host). Prints the real per-call wall-clock time at both fidelity tiers so
gym_schema.py's photonics_domain() FIDELITY_PRESETS cost_credits can be
filled in with a measured number, not an invented one (same convention as
every other domain's FidelityLevel.measured_from)."""
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import photonics_oracle as oracle

# A plausible, mid-range real design: 220nm SOI thickness (standard
# process node), ~630nm period / 50%-50% duty cycle (uniform, no
# apodization -- the degenerate case), 70nm partial etch, 2.0um box oxide
# (v1's own former fixed constant, now the design's midpoint default).
TEST_PARAMS = (0.22, 0.63, 0.5, 0.5, 0.07, 2.0)

for fidelity in ["low", "medium"]:
    args = oracle.params_to_worker_args(TEST_PARAMS, {}, fidelity)
    parent_conn, child_conn = mp.Pipe()
    t0 = time.time()
    p = mp.Process(target=oracle.worker_fn, args=(child_conn, *args))
    p.start()
    ok, payload = parent_conn.recv()
    p.join(timeout=300)
    elapsed = time.time() - t0
    print(f"[{fidelity}] elapsed={elapsed:.2f}s ok={ok} payload={payload}")

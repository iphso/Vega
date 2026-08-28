"""Domain-agnostic parallel-batch-with-hard-timeout scoring harness.

Originally three separate near-duplicate implementations (generate_and_validate.py,
the first version of eval_cvae_steerability.py, this file's true origin) all doing
the same thing for VMEC++ specifically: run a per-candidate subprocess (not a
persistent pool worker, so a hung candidate can be hard-killed without taking
the pool down with it -- see generate_and_validate.py's docstring for the
concrete VMEC++-hangs-instead-of-failing story that motivated this), race it
against a wall-clock timeout, and yield (tag, ok, payload) as each candidate
finishes -- in COMPLETION order, not submission order, so callers must key
results by the caller-supplied `tag`, never by position.

Consolidated here as the one domain-agnostic version: nothing in this module
knows what a "candidate" or a "score" actually means. Any domain that scores
a parameter vector into named metrics -- or fails to -- plugs in by providing
a `worker_fn(conn, *worker_args)` that does the actual scoring and sends
`(ok, payload)` back over `conn`. See oracle_base.py for the fuller interface
a domain is expected to implement (this function only needs `worker_fn`
itself); vmec_oracle.py is the reference implementation for this project's
actual domain (VMEC++ stellarator boundaries).
"""
import multiprocessing as mp
import time


def run_batch_with_timeout(candidates, worker_fn, n_workers, timeout_s):
    """candidates: list of (tag, *worker_args) tuples -- `tag` is never
    passed to worker_fn, it's purely for the caller to identify which
    result is which once they arrive out of order. Yields (tag, ok, payload)
    as each candidate finishes or times out."""
    pending = list(candidates)
    running = {}  # pid -> (process, conn, tag, start_time)
    while pending or running:
        while pending and len(running) < n_workers:
            tag, *worker_args = pending.pop()
            parent_conn, child_conn = mp.Pipe(duplex=False)
            proc = mp.Process(target=worker_fn, args=(child_conn, *worker_args))
            proc.start()
            child_conn.close()
            running[proc.pid] = (proc, parent_conn, tag, time.perf_counter())
        finished_pids = []
        for pid, (proc, conn, tag, t0) in running.items():
            if conn.poll(0.05):
                try:
                    ok, payload = conn.recv()
                except EOFError:
                    ok, payload = False, "unknown: worker died without a result"
                conn.close()
                proc.join(timeout=2)
                yield (tag, ok, payload)
                finished_pids.append(pid)
            elif time.perf_counter() - t0 > timeout_s:
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                conn.close()
                yield (tag, False, f"timed out after {timeout_s:.0f}s")
                finished_pids.append(pid)
        for pid in finished_pids:
            del running[pid]
        if not finished_pids:
            time.sleep(0.05)

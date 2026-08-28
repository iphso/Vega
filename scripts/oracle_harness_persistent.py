"""Persistent-worker counterpart to oracle_harness.py's run_batch_with_timeout
-- the genuinely new harness shape TORAX needs and neither VMEC++ nor
XFOIL do, confirmed empirically before writing any of this (not assumed):
a TORAX run costs ~4.9s the first time in a fresh process (JIT compilation
dominates) but ~0.13-0.43s for every subsequent call in the SAME process,
even when real parameters change between calls -- a ~10-38x difference.
oracle_harness.py's subprocess-per-candidate pattern pays that ~4.9s
compile tax on every single candidate; this file exists specifically to
not do that.

VMEC++'s harness is subprocess-per-candidate for a real reason (it can
hang on pathological input, EXPERIMENT_LOG §9 -- a fresh, killable process
per candidate is the right shape there, not a compromise). TORAX has no
known hang problem (its failure mode is a clean SimError, confirmed by
reading source, §25) but paying a 10-38x tax per candidate would be worse
than useless for bulk generation. Different domain, different constraint,
different harness -- this is the concrete case gym_schema.DomainSpec's
`harness_fit` field was added to distinguish.

Design: spawn `n_workers` long-lived processes up front, each running
`persistent_worker_fn(conn)` -- a function that does its own (expensive,
one-time) imports, then loops forever: receive one candidate's worker_args
over `conn`, compute, send back (ok, payload), repeat, until it receives
the sentinel `None` and exits. Candidates are distributed round-robin
across the live worker pool rather than one-shot-per-candidate.

A per-candidate timeout still exists (TORAX CAN be slow on a genuinely
pathological config even if it doesn't hang outright), but the response to
a timeout here is different from oracle_harness.py's: killing a hung
WORKER discards its warm JIT cache along with it, so that worker is
terminated and respawned (paying the compile tax again on ITS next
candidate), while every other worker in the pool keeps its own cache and
keeps running -- a batch-wide hang is avoided without needing to treat
every candidate as disposable.
"""
import multiprocessing as mp
import time


def run_batch_persistent(candidates, persistent_worker_fn, n_workers, timeout_s):
    """candidates: list of (tag, *worker_args). Yields (tag, ok, payload) as
    each candidate finishes, NOT necessarily in submission order (workers
    run at different speeds depending on what's already warm in their own
    JIT cache) -- same "key by tag, not position" discipline as
    oracle_harness.run_batch_with_timeout."""
    pending = list(candidates)

    def spawn_worker():
        parent_conn, child_conn = mp.Pipe(duplex=True)
        proc = mp.Process(target=persistent_worker_fn, args=(child_conn,))
        proc.start()
        child_conn.close()
        return proc, parent_conn

    workers = [spawn_worker() for _ in range(n_workers)]
    # idle_workers holds (proc, conn) tuples ready for a new candidate;
    # busy_workers maps proc.pid -> (proc, conn, tag, start_time) for
    # in-flight ones.
    idle_workers = list(workers)
    busy_workers = {}

    while pending or busy_workers:
        while pending and idle_workers:
            tag, *worker_args = pending.pop()
            proc, conn = idle_workers.pop()
            conn.send(worker_args)
            busy_workers[proc.pid] = (proc, conn, tag, time.perf_counter())

        finished_pids = []
        for pid, (proc, conn, tag, t0) in busy_workers.items():
            if conn.poll(0.05):
                try:
                    ok, payload = conn.recv()
                except EOFError:
                    ok, payload = False, "unknown: worker died without a result"
                    # worker is gone -- respawn a replacement rather than
                    # returning it to the idle pool.
                    new_proc, new_conn = spawn_worker()
                    idle_workers.append((new_proc, new_conn))
                    yield (tag, ok, payload)
                    finished_pids.append(pid)
                    continue
                idle_workers.append((proc, conn))
                yield (tag, ok, payload)
                finished_pids.append(pid)
            elif time.perf_counter() - t0 > timeout_s:
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                conn.close()
                # this worker's warm JIT cache is gone with it -- respawn a
                # fresh one so the pool stays at n_workers, but the new
                # worker starts cold (pays the compile tax on its next job).
                new_proc, new_conn = spawn_worker()
                idle_workers.append((new_proc, new_conn))
                yield (tag, False, f"timed out after {timeout_s:.0f}s")
                finished_pids.append(pid)
        for pid in finished_pids:
            del busy_workers[pid]
        if not finished_pids:
            time.sleep(0.05)

    # shut down the pool cleanly
    for proc, conn in idle_workers:
        try:
            conn.send(None)
        except (BrokenPipeError, EOFError):
            pass
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

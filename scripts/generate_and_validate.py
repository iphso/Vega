"""Data-augmentation generation loop: sample candidate boundaries, validate
each through the real VMEC++ oracle in parallel across CPU cores, and
incrementally store every design that actually converges -- along with its
real computed metrics -- as new labeled data.

Two sampling modes:

  vae-cluster (default) -- sample latents from a VAE trained on the real
    dataset (scripts/train_vae.py), biased toward under-covered clusters
    (the smaller clusters from the §6/make_splits.py k-means split): pick a
    real row from an under-covered cluster as an anchor, encode it to its
    VAE latent mean, decode a nearby point (anchor + Gaussian noise,
    --explore-std) using the anchor's own n_field_periods. High hit rate
    (~69% in testing) since the VAE has learned the real manifold's shape;
    biased toward filling gaps in known coverage, not exploring past it.
    Output: output/generated/{X,Y}.npy.

  random -- no VAE, no clustering, "basically no prior": each of the 81 free
    coefficients sampled independently and uniformly within its own observed
    [min, max] (metadata.json feature_stats), n_field_periods uniform over
    the observed {1..5}. Doesn't respect the real cross-coefficient
    correlations at all, so the hit rate against VMEC++ is expected to be
    much lower than vae-cluster -- but VMEC++ itself is the ground truth
    here (not a learnable surrogate that could be fooled), so there's no
    adversarial-exploit risk the way there was in optimize.py: a converged
    result just is physically consistent, low precision or not. The point
    isn't hitting specific metric values, it's a broad, unbiased empirical
    read on where the truly-feasible region of coefficient space is, not
    just resampling/filling gaps in what this dataset already covers.
    Output: output/generated_random/{X,Y}.npy (kept separate from
    vae-cluster's output for clean provenance -- very different sampling
    distributions, shouldn't be silently conflated).

Writes incrementally (every --checkpoint-every accepted designs) so a long
run doesn't lose progress if interrupted.

Rejected candidates are kept too, not discarded: every design that fails
(structural, genuine VMEC++ non-convergence, timeout, or unknown -- a worker
crash/died-without-result, kept distinct from "vmec" because it isn't
actually a confirmed physics rejection) is saved to <gen_dir>/rejected_X.npy
alongside a parallel rejected_reason.npy code (see REJECT_REASON_CODES /
rejected_reason_legend.json in the same dir). Every attempted candidate ends
up in either X.npy or rejected_X.npy -- none of the VMEC++ compute is
thrown away, even the ambiguous cases.
These have no physics labels (Y) -- that's the point, they're the
"this didn't work" side of the feasible/infeasible boundary, useful for
characterizing that boundary directly (e.g. a feasibility classifier) rather
than just discarding the negative examples every run currently threw away.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("/work/output")

# Structurally-always-zero coefficients (see EXPERIMENT_LOG / metadata.json
# feature_stats): r_cos m=0,n<4 and z_sin m=0 entirely. Hard-enforced on
# every candidate rather than trusting the sampler to land on exact zeros.
ZERO_COEFF_IDX = [0, 1, 2, 3, 45, 46, 47, 48, 49]
NFP_VALUES = [1, 2, 3, 4, 5]
REJECT_REASON_CODES = {"structural": 0, "vmec": 1, "timeout": 2, "unknown": 3}


def _validate_entry(conn, r_cos, z_sin, nfp):
    """Runs in its own throwaway subprocess (not a persistent pool worker) so
    the caller can hard-kill it on timeout: VMEC++ occasionally hangs on a
    pathological low-quality geometry (confirmed empirically -- pure random
    per-coefficient sampling produced a candidate that ran 2+ hours instead
    of failing fast the way structurally-invalid ones do) rather than
    raising. A persistent worker pool (e.g. ProcessPoolExecutor) has no clean
    way to reclaim a worker stuck on one non-terminating task; a fresh
    process per candidate can just be killed and discarded.
    """
    try:
        from constellaration import forward_model
        from constellaration.geometry import surface_rz_fourier
        try:
            boundary = surface_rz_fourier.SurfaceRZFourier(
                r_cos=r_cos, z_sin=z_sin, n_field_periods=nfp, is_stellarator_symmetric=True,
            )
        except Exception as e:
            conn.send((False, f"structural: {e}"))
            return
        try:
            metrics, _ = forward_model.forward_model(boundary)
        except Exception as e:
            conn.send((False, f"vmec: {e}"))
            return
        conn.send((True, metrics.model_dump()))
    except Exception as e:
        # Anything else (import failure, unhandled exception in constellaration
        # itself, etc.) -- distinct from a genuine VMEC++ convergence failure:
        # we don't actually know if this candidate is physically infeasible or
        # if the tooling just broke on it, so it's tagged "unknown" rather than
        # folded into "vmec" and treated as a confirmed physics rejection.
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def run_batch_with_timeout(candidates, n_workers, timeout_s):
    """Validates `candidates` (each an (r_cos, z_sin, nfp) tuple), up to
    n_workers concurrently, each in its own subprocess with a hard wall-clock
    timeout (SIGTERM then SIGKILL if it doesn't exit). Yields
    (ok, r_cos, z_sin, nfp, payload) as each candidate finishes or times out.
    """
    pending = list(candidates)
    running = {}  # pid -> (process, conn, candidate, start_time)
    while pending or running:
        while pending and len(running) < n_workers:
            cand = pending.pop()
            parent_conn, child_conn = mp.Pipe(duplex=False)
            proc = mp.Process(target=_validate_entry, args=(child_conn, *cand))
            proc.start()
            child_conn.close()
            running[proc.pid] = (proc, parent_conn, cand, time.perf_counter())

        finished_pids = []
        for pid, (proc, conn, cand, t0) in running.items():
            if conn.poll(0.05):
                try:
                    ok, payload = conn.recv()
                except EOFError:
                    # Worker process died (e.g. killed by the OOM killer) without
                    # sending anything back -- also "unknown", not a confirmed
                    # physics rejection.
                    ok, payload = False, "unknown: worker died without a result"
                conn.close()
                proc.join(timeout=2)
                yield (ok, *cand, payload)
                finished_pids.append(pid)
            elif time.perf_counter() - t0 > timeout_s:
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                conn.close()
                yield (False, *cand, f"vmec: timed out after {timeout_s:.0f}s")
                finished_pids.append(pid)
        for pid in finished_pids:
            del running[pid]
        if not finished_pids:
            time.sleep(0.05)


def make_vae_cluster_sampler(args, rng):
    from train_vae import VAE, nfp_one_hot

    dev = torch.device("cpu")
    ckpt = torch.load(f"/work/checkpoints/{args.vae_tag}.pt", map_location=dev)
    model = VAE(coeff_dim=90, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]

    X = np.load(OUT_DIR / "X.npy")
    assign = np.load(OUT_DIR / "cluster_assignments.npy")
    cluster_sizes = {int(k): v for k, v in json.loads((OUT_DIR / "cluster_sizes.json").read_text()).items()}
    threshold = np.percentile(list(cluster_sizes.values()), args.sparse_percentile)
    sparse_clusters = {c for c, n in cluster_sizes.items() if n <= threshold}
    anchor_pool = np.where(np.isin(assign, list(sparse_clusters)))[0]
    print(f"[{args.tag}] {len(sparse_clusters)}/{len(cluster_sizes)} clusters at/below "
          f"p{args.sparse_percentile} (size<={threshold:.0f}) -- anchor pool: {len(anchor_pool)} rows "
          f"of {len(X)} total")

    anchor_coeffs = torch.tensor((X[anchor_pool][:, :90] - coeff_mean) / coeff_std, dtype=torch.float32)
    anchor_nfp = torch.tensor(X[anchor_pool][:, 90], dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model.encode(anchor_coeffs, nfp_one_hot(anchor_nfp))

    def sample(n):
        picks = rng.integers(0, len(anchor_pool), size=n)
        z = mu[picks] + args.explore_std * torch.randn(n, mu.shape[1])
        nfp_batch = anchor_nfp[picks]
        with torch.no_grad():
            decoded = model.decode(z, nfp_one_hot(nfp_batch)).numpy()
        coeffs = decoded * coeff_std + coeff_mean
        coeffs[:, ZERO_COEFF_IDX] = 0.0
        out = []
        for k in range(n):
            r_cos = coeffs[k, :45].reshape(5, 9).astype(np.float64)
            z_sin = coeffs[k, 45:90].reshape(5, 9).astype(np.float64)
            out.append((r_cos, z_sin, int(nfp_batch[k].item())))
        return out

    return sample


def make_vae_prior_sampler(args, rng):
    """Unconditional sampling from the VAE's own prior (z ~ N(0,1), nfp
    drawn uniformly) -- not anchored to any specific real point or cluster,
    so it isn't biased toward known-sparse OR known-dense regions the way
    vae-cluster is. Confirmed necessary empirically: make_random_sampler's
    fully-uncorrelated per-coefficient sampling had a 0% hit rate over 2,592
    VMEC++ attempts (real boundaries have strong cross-coefficient
    correlations -- smoothness, non-self-intersection -- that independent
    sampling destroys entirely). This keeps enough of the learned
    correlational structure to actually produce valid geometries, while
    still being a broad, untargeted "prior" sample rather than a
    region-targeted one.
    """
    from train_vae import VAE, nfp_one_hot

    dev = torch.device("cpu")
    ckpt = torch.load(f"/work/checkpoints/{args.vae_tag}.pt", map_location=dev)
    model = VAE(coeff_dim=90, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    coeff_mean, coeff_std = ckpt["coeff_mean"], ckpt["coeff_std"]
    latent_dim = ckpt["latent_dim"]
    print(f"[{args.tag}] VAE prior sampling: z ~ N(0,1) in the {latent_dim}-dim latent space "
          f"(no anchor, no cluster targeting), nfp uniform over {NFP_VALUES}.")

    def sample(n):
        z = torch.randn(n, latent_dim)
        nfp_batch = rng.choice(NFP_VALUES, size=n)
        nfp_t = torch.tensor(nfp_batch, dtype=torch.float32)
        with torch.no_grad():
            decoded = model.decode(z, nfp_one_hot(nfp_t)).numpy()
        coeffs = decoded * coeff_std + coeff_mean
        coeffs[:, ZERO_COEFF_IDX] = 0.0
        out = []
        for k in range(n):
            r_cos = coeffs[k, :45].reshape(5, 9).astype(np.float64)
            z_sin = coeffs[k, 45:90].reshape(5, 9).astype(np.float64)
            out.append((r_cos, z_sin, int(nfp_batch[k])))
        return out

    return sample


def make_random_sampler(args, rng):
    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    feature_names = json.loads((OUT_DIR / "feature_names.json").read_text())
    lo = np.array([meta["feature_stats"][n]["min"] for n in feature_names[:90]])
    hi = np.array([meta["feature_stats"][n]["max"] for n in feature_names[:90]])
    print(f"[{args.tag}] pure random sampling: 81 free coefficients uniform in their own observed "
          f"[min,max], nfp uniform over {NFP_VALUES}. No VAE, no clustering, no bias toward existing data.")

    def sample(n):
        coeffs = rng.uniform(lo, hi, size=(n, 90))
        coeffs[:, ZERO_COEFF_IDX] = 0.0
        nfp_batch = rng.choice(NFP_VALUES, size=n)
        out = []
        for k in range(n):
            r_cos = coeffs[k, :45].reshape(5, 9)
            z_sin = coeffs[k, 45:90].reshape(5, 9)
            out.append((r_cos, z_sin, int(nfp_batch[k])))
        return out

    return sample


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampling-mode", default="vae-cluster", choices=["vae-cluster", "vae-prior", "random"])
    p.add_argument("--vae-tag", default="vae_coeffs_s0", help="vae-cluster mode only")
    p.add_argument("--target-count", type=int, default=400, help="stop once this many designs validate")
    p.add_argument("--sparse-percentile", type=float, default=50.0, help="vae-cluster mode only")
    p.add_argument("--explore-std", type=float, default=0.5, help="vae-cluster mode only")
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=112, help="candidates submitted per round (multiple of n-workers)")
    p.add_argument("--timeout-seconds", type=float, default=45.0,
                    help="hard per-candidate wall-clock limit (terminate+kill the subprocess if exceeded) -- "
                         "VMEC++ can occasionally hang on a low-quality geometry instead of failing fast, "
                         "confirmed to happen under --sampling-mode random. Comfortably above the ~6-14s "
                         "seen for legitimate low-fidelity solves.")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--max-seconds", type=float, default=None, help="optional wall-clock cap instead of/in addition to target-count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run0")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    gen_dir_name = {
        "vae-cluster": "generated", "vae-prior": "generated_vae_prior", "random": "generated_random",
    }[args.sampling_mode]
    gen_dir = OUT_DIR / gen_dir_name
    gen_dir.mkdir(parents=True, exist_ok=True)

    sampler_factory = {
        "vae-cluster": make_vae_cluster_sampler,
        "vae-prior": make_vae_prior_sampler,
        "random": make_random_sampler,
    }[args.sampling_mode]
    sample_candidates = sampler_factory(args, rng)

    accepted_X, accepted_Y = [], []
    rejected_X, rejected_reason = [], []
    n_attempted, n_accepted, n_reject_structural, n_reject_vmec, n_reject_unknown = 0, 0, 0, 0, 0
    t_start = time.perf_counter()

    def _append(path, arrays_to_concat):
        Xc = np.concatenate([np.load(path)] + arrays_to_concat) if path.exists() else np.concatenate(arrays_to_concat)
        np.save(path, Xc)

    def flush():
        if accepted_X:
            _append(gen_dir / "X.npy", [np.stack(accepted_X)])
            _append(gen_dir / "Y.npy", [np.stack(accepted_Y)])
            accepted_X.clear()
            accepted_Y.clear()
        if rejected_X:
            _append(gen_dir / "rejected_X.npy", [np.stack(rejected_X)])
            _append(gen_dir / "rejected_reason.npy", [np.array(rejected_reason, dtype=np.int8)])
            rejected_X.clear()
            rejected_reason.clear()
            (gen_dir / "rejected_reason_legend.json").write_text(
                json.dumps({v: k for k, v in REJECT_REASON_CODES.items()}, indent=2))

    n_timeout = 0
    while n_accepted < args.target_count:
        if args.max_seconds and (time.perf_counter() - t_start) > args.max_seconds:
            print(f"[{args.tag}] hit --max-seconds budget, stopping")
            break
        candidates = sample_candidates(args.batch_size)
        n_attempted += len(candidates)
        for ok, r_cos, z_sin, nfp, payload in run_batch_with_timeout(candidates, args.n_workers, args.timeout_seconds):
            row = np.concatenate([r_cos.flatten(), z_sin.flatten(), [float(nfp), 1.0]]).astype(np.float32)
            if ok:
                y = np.array([payload[name] for name in target_names], dtype=np.float64)
                if any(v is None for v in y) or not np.all(np.isfinite(y.astype(np.float64))):
                    n_reject_vmec += 1
                    rejected_X.append(row)
                    rejected_reason.append(REJECT_REASON_CODES["vmec"])
                    continue
                accepted_X.append(row)
                accepted_Y.append(y.astype(np.float32))
                n_accepted += 1
            else:
                if payload.startswith("structural"):
                    n_reject_structural += 1
                    reason = "structural"
                elif "timed out" in payload:
                    n_timeout += 1
                    reason = "timeout"
                elif payload.startswith("unknown"):
                    n_reject_unknown += 1
                    reason = "unknown"
                else:
                    n_reject_vmec += 1
                    reason = "vmec"
                rejected_X.append(row)
                rejected_reason.append(REJECT_REASON_CODES[reason])

        elapsed = time.perf_counter() - t_start
        print(f"[{args.tag}] attempted={n_attempted} accepted={n_accepted}/{args.target_count} "
              f"(hit rate {n_accepted / max(n_attempted, 1):.1%})  "
              f"reject: structural={n_reject_structural} vmec={n_reject_vmec} timeout={n_timeout} "
              f"unknown={n_reject_unknown}  elapsed={elapsed:.0f}s")

        if len(accepted_X) >= args.checkpoint_every or len(rejected_X) >= args.checkpoint_every:
            flush()
            print(f"[{args.tag}] checkpointed to {gen_dir}")

    flush()
    stats = {
        "tag": args.tag, "sampling_mode": args.sampling_mode, "target_count": args.target_count,
        "n_accepted": n_accepted, "n_attempted": n_attempted,
        "n_reject_structural": n_reject_structural, "n_reject_vmec": n_reject_vmec, "n_timeout": n_timeout,
        "n_reject_unknown": n_reject_unknown,
        "timeout_seconds": args.timeout_seconds,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    if args.sampling_mode == "vae-cluster":
        stats.update({"vae_tag": args.vae_tag, "sparse_percentile": args.sparse_percentile, "explore_std": args.explore_std})
    elif args.sampling_mode == "vae-prior":
        stats.update({"vae_tag": args.vae_tag})
    stats_path = gen_dir / f"run_stats_{args.tag}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\ndone. accepted {n_accepted} of {n_attempted} attempted "
          f"({n_accepted / max(n_attempted, 1):.1%} hit rate) in {stats['elapsed_seconds']:.0f}s")
    print(f"stats saved to {stats_path}")


if __name__ == "__main__":
    main()

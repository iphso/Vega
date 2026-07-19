"""Gradient-guided candidate generation: backprop a real, trained target
metric's prediction through the surrogate and the VAE decoder back to the
VAE's latent space, and walk downhill on it -- a second, cheap source of
candidates alongside the sampling-based bootstrap loop, using neural-net
autodiff instead of vmec_jax's physics-implicit-diff (which turned out
slow and fragile at our resolution; this is neither, since both networks
are small, ordinary, already-trained PyTorch modules -- no equilibrium
solve in the loop at all).

The gradient is w.r.t. the SURROGATE's prediction, not real physics -- see
EXPERIMENT_LOG (S8): a design-search candidate the surrogate ensemble was
confident about failed 2/3 real VMEC++ constraints. Treat every step this
script takes as a proposal, not an answer: real VMEC++ is still the sole
acceptance test, every round, for every walk.

Adaptive step size (matches the user's exact spec): one shared step size
across all walks. Each round, decode+validate all walks' proposed steps
through real VMEC++; a walk whose step is invalid rolls back to its last
valid point (but keeps trying from there next round) rather than wandering
off into infeasible territory. If too small a fraction of the round's
proposed steps were valid, shrink the shared step size -- "looking for
gaps in the gradient wall": each walk hugs the edge of the feasible region
while pushing toward lower `--target`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import DualPathMLP  # noqa: E402
from train_vae import VAE, nfp_one_hot  # noqa: E402
from generate_and_validate import ZERO_COEFF_IDX, REJECT_REASON_CODES, run_batch_with_timeout  # noqa: E402

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def load_vae(tag):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location="cpu", weights_only=False)
    model = VAE(coeff_dim=90, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["coeff_mean"], ckpt["coeff_std"]


def load_surrogate(tag):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location="cpu", weights_only=False)
    model = DualPathMLP(
        in_dim=ckpt["in_dim"], n_targets=ckpt["n_targets"], latent_dim=ckpt["latent_dim"],
        hidden=ckpt["hidden"], spatial_latent=ckpt["spatial_latent"], head_hidden=ckpt["head_hidden"],
        use_spatial=ckpt["use_spatial"], trunk_arch=ckpt["trunk_arch"], trunk_blocks=ckpt.get("trunk_blocks", 3),
        use_symlog_latent=ckpt["use_symlog_latent"], log_target_mask=ckpt["log_target_mask"],
        objective=ckpt["objective"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    assert ckpt["feature_mean"] is None and ckpt["target_mean"] is None, \
        "this script assumes an unnormalized checkpoint -- add de-normalization before using one that isn't"
    assert not ckpt["log_target_mask"].any(), \
        "this script assumes no log-masked targets -- add exp() on the target head before using one that has them"
    return model, ckpt["target_names"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae-tag", default="vae_coeffs_s0")
    p.add_argument("--surrogate-tag", default="split_vae_prior_augmented_s0")
    p.add_argument("--target", default="max_elongation", help="which of the 11 targets to minimize")
    p.add_argument("--n-walks", type=int, default=28)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--step-size", type=float, default=0.1, help="initial shared step size in latent space")
    p.add_argument("--step-shrink", type=float, default=0.5)
    p.add_argument("--valid-frac-threshold", type=float, default=0.5,
                    help="shrink step size if this round's valid fraction falls below it")
    p.add_argument("--min-step-size", type=float, default=1e-4, help="stop once step size shrinks below this")
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="walk0")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    vae, coeff_mean, coeff_std = load_vae(args.vae_tag)
    coeff_mean_t = torch.tensor(coeff_mean, dtype=torch.float32)
    coeff_std_t = torch.tensor(coeff_std, dtype=torch.float32)
    surrogate, target_names = load_surrogate(args.surrogate_tag)
    target_idx = target_names.index(args.target)
    target_names_json = json.loads((OUT_DIR / "target_names.json").read_text())
    assert target_names == target_names_json, "surrogate target order doesn't match target_names.json"

    # Anchor each walk to a distinct real design's encoded latent mean --
    # starts every walk from a point already known to be physically valid.
    X_real = np.load(OUT_DIR / "X.npy")
    anchor_rows = rng.choice(len(X_real), size=args.n_walks, replace=False)
    anchor_coeffs = torch.tensor((X_real[anchor_rows][:, :90] - coeff_mean) / coeff_std, dtype=torch.float32)
    anchor_nfp = torch.tensor(X_real[anchor_rows][:, 90], dtype=torch.float32)
    with torch.no_grad():
        z, _ = vae.encode(anchor_coeffs, nfp_one_hot(anchor_nfp))
    z = z.clone().requires_grad_(True)
    nfp_cond = nfp_one_hot(anchor_nfp)  # fixed per walk for its whole trajectory

    pool_dir = OUT_DIR / f"gradient_walk_{args.tag}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    log_path = pool_dir / "round_log.jsonl"

    accepted_X, accepted_Y = [], []
    rejected_X, rejected_reason = [], []

    def flush():
        def _append(path, arrays):
            arr = np.concatenate([np.load(path)] + arrays) if path.exists() else np.concatenate(arrays)
            np.save(path, arr)
        if accepted_X:
            _append(pool_dir / "X.npy", [np.stack(accepted_X)])
            _append(pool_dir / "Y.npy", [np.stack(accepted_Y)])
            accepted_X.clear(); accepted_Y.clear()
        if rejected_X:
            _append(pool_dir / "rejected_X.npy", [np.stack(rejected_X)])
            _append(pool_dir / "rejected_reason.npy", [np.array(rejected_reason, dtype=np.int8)])
            rejected_X.clear(); rejected_reason.clear()
            (pool_dir / "rejected_reason_legend.json").write_text(
                json.dumps({v: k for k, v in REJECT_REASON_CODES.items()}, indent=2))

    step_size = args.step_size
    t_start = time.perf_counter()

    for round_idx in range(1, args.n_rounds + 1):
        if step_size < args.min_step_size:
            print(f"[{args.tag}] step size below floor, stopping at round {round_idx}")
            break

        decoded = vae.decode(z, nfp_cond)
        coeffs = decoded * coeff_std_t + coeff_mean_t
        pred = surrogate(torch.cat([coeffs, anchor_nfp.unsqueeze(1), torch.ones(args.n_walks, 1)], dim=1))
        loss = pred[:, target_idx].sum()
        grad, = torch.autograd.grad(loss, z)

        with torch.no_grad():
            grad_norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z_proposed = z - step_size * grad / grad_norm

            coeffs_proposed = (vae.decode(z_proposed, nfp_cond) * coeff_std_t + coeff_mean_t).numpy()
        coeffs_proposed[:, ZERO_COEFF_IDX] = 0.0

        candidates = []
        for i in range(args.n_walks):
            r_cos = coeffs_proposed[i, :45].reshape(5, 9).astype(np.float64)
            z_sin = coeffs_proposed[i, 45:90].reshape(5, 9).astype(np.float64)
            candidates.append((r_cos, z_sin, int(anchor_nfp[i].item())))

        pred_elong_before = pred[:, target_idx].detach().numpy()

        # run_batch_with_timeout yields results in completion order, not
        # submission order (workers finish at different times) -- submit all
        # n_walks candidates together (using the full worker pool) and match
        # each result back to its walk index by exact value, rather than
        # validating one at a time and losing the parallelism.
        ordered_results = [None] * args.n_walks
        for ok, r_cos, z_sin, nfp, payload in run_batch_with_timeout(candidates, args.n_workers, args.timeout_seconds):
            for i, (cr, cz, cn) in enumerate(candidates):
                if ordered_results[i] is None and cn == nfp and np.array_equal(cr, r_cos) and np.array_equal(cz, z_sin):
                    ordered_results[i] = (ok, payload)
                    break

        n_valid = 0
        for i, (ok, payload) in enumerate(ordered_results):
            row = np.concatenate([coeffs_proposed[i, :90], [float(anchor_nfp[i].item()), 1.0]]).astype(np.float32)
            if ok:
                y = np.array([payload[name] for name in target_names], dtype=np.float64)
                if any(v is None for v in y) or not np.all(np.isfinite(y.astype(np.float64))):
                    rejected_X.append(row); rejected_reason.append(REJECT_REASON_CODES["vmec"])
                    continue
                n_valid += 1
                accepted_X.append(row); accepted_Y.append(y.astype(np.float32))
                with torch.no_grad():
                    z[i] = z_proposed[i]
            else:
                if payload.startswith("structural"):
                    reason = "structural"
                elif "timed out" in payload:
                    reason = "timeout"
                elif payload.startswith("unknown"):
                    reason = "unknown"
                else:
                    reason = "vmec"
                rejected_X.append(row); rejected_reason.append(REJECT_REASON_CODES[reason])

        frac_valid = n_valid / args.n_walks
        if frac_valid < args.valid_frac_threshold:
            step_size *= args.step_shrink

        record = {
            "round": round_idx, "step_size": step_size, "n_valid": n_valid, "n_walks": args.n_walks,
            "frac_valid": frac_valid, "pred_target_mean_before_step": float(pred_elong_before.mean()),
            "elapsed_seconds": time.perf_counter() - t_start,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{args.tag}] round {round_idx}: valid {n_valid}/{args.n_walks} ({frac_valid:.0%})  "
              f"step_size={step_size:.5f}  pred_{args.target}_mean={pred_elong_before.mean():.4f}")

        if len(accepted_X) >= 100 or len(rejected_X) >= 100:
            flush()

    flush()
    print(f"[{args.tag}] done. pool at {pool_dir}, log at {log_path}")


if __name__ == "__main__":
    main()

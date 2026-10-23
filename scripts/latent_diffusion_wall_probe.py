"""Experiment (B), stage 3: real-oracle validation of the fine-tuned latent-
diffusion model (train_vae_physical.py + train_latent_diffusion.py) against
the exact same "beyond-the-wall" conditioning targets nae_generative_wall_
probe.py used for the raw-coefficient cVAE/diffusion/GAN baselines (0/480
qi-feasible there) -- so this result is directly comparable, not a new
easier test.

Pipeline: build cond = [beyond-wall target (log+z-scored), nfp one-hot] ->
ddpm_sample in the fine-tuned model's 32-dim latent space -> unstandardize
latent (lat_mean/lat_std from the checkpoint) -> VAE decoder -> standardized
90-dim coefficients -> unstandardize via the VAE's own coeff_mean/coeff_std
-> zero ZERO_INDICES -> real-VMEC validate via the same
oracle.params_to_worker_args/run_batch_with_timeout path every other probe
in this project uses, at 'low' fidelity.

Must run in the `train` container (needs torch AND vmecpp/constellaration in
the same process -- same reason nae_generative_wall_probe.py does).

Output: output/latent_diffusion_wall_probe/results.jsonl.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import vmec_oracle as oracle  # noqa: E402
from oracle_harness import run_batch_with_timeout  # noqa: E402
from train_vae import VAE, nfp_one_hot, NFP_VALUES  # noqa: E402
from train_diffusion import DiffusionDenoiser, make_schedule, ddpm_sample  # noqa: E402
from nae_generative_wall_probe import beyond_wall_target, p3_satisfied, P3_THRESH  # noqa: E402
from eval_cvae_steerability import eval_space  # noqa: E402

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")
STATE_DIR = OUT_DIR / "latent_diffusion_wall_probe"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae-tag", default="vae_physical_v1")
    p.add_argument("--diffusion-tag", default="latent_diffusion_v1_hlft")
    p.add_argument("--k", type=int, default=40, help="candidates decoded per nfp")
    p.add_argument("--nfps", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device("cpu")

    vae_ckpt = torch.load(CKPT_DIR / f"{args.vae_tag}.pt", map_location=dev)
    vae = VAE(coeff_dim=90, latent_dim=vae_ckpt["latent_dim"], hidden=vae_ckpt["hidden"]).to(dev)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    coeff_mean = torch.tensor(vae_ckpt["coeff_mean"], dtype=torch.float32)
    coeff_std = torch.tensor(vae_ckpt["coeff_std"], dtype=torch.float32)

    diff_ckpt = torch.load(CKPT_DIR / f"{args.diffusion_tag}.pt", map_location=dev)
    model = DiffusionDenoiser(coeff_dim=diff_ckpt["latent_dim"], n_targets=len(diff_ckpt["target_names"]),
                               hidden=diff_ckpt["hidden"], time_embed_dim=diff_ckpt["time_embed_dim"]).to(dev)
    model.load_state_dict(diff_ckpt["model_state_dict"])
    model.eval()
    schedule = make_schedule(diff_ckpt["T"], device=dev)
    target_names = diff_ckpt["target_names"]
    t_mean, t_std = diff_ckpt["target_mean"], diff_ckpt["target_std"]
    lat_mean = torch.tensor(diff_ckpt["lat_mean"], dtype=torch.float32)
    lat_std = torch.tensor(diff_ckpt["lat_std"], dtype=torch.float32)

    jobs, meta = [], {}
    tag_ctr = itertools.count()
    for nfp in args.nfps:
        y_req = beyond_wall_target(target_names, nfp)
        y_req_eval = eval_space(y_req[None, :], target_names)[0]
        y_req_z = (y_req_eval - t_mean) / t_std
        nfp_idx = NFP_VALUES.index(nfp)
        nfp_oh_1 = torch.nn.functional.one_hot(torch.tensor([nfp_idx]), num_classes=len(NFP_VALUES)).float()
        cond = torch.cat([torch.tensor(y_req_z, dtype=torch.float32)[None, :], nfp_oh_1], dim=-1)
        cond_batch = cond.repeat(args.k, 1)

        with torch.no_grad():
            lat_std_sample = ddpm_sample(model, cond_batch, schedule, coeff_dim=diff_ckpt["latent_dim"], device=dev)
            lat_phys = lat_std_sample * lat_std + lat_mean
            nfp_oh_k = nfp_oh_1.repeat(args.k, 1)
            decoded_std = vae.decode(lat_phys, nfp_oh_k)
            params = (decoded_std * coeff_std + coeff_mean).numpy()
        params[:, oracle.ZERO_INDICES] = 0.0

        for i in range(args.k):
            tag = next(tag_ctr)
            r_cos, z_sin, job_nfp, fidelity_name = oracle.params_to_worker_args(
                params[i], {"nfp": nfp}, oracle.FIDELITY_PRESETS["low"])
            jobs.append((tag, r_cos, z_sin, job_nfp, fidelity_name))
            meta[tag] = {"nfp": nfp, "requested": dict(zip(target_names, y_req.tolist()))}

    print(f"[latent_diff_probe] {len(jobs)} candidates to validate ({len(args.nfps)} nfps x {args.k})")

    results = []
    for tag, ok, payload in run_batch_with_timeout(jobs, oracle.worker_fn, args.n_workers, args.timeout_seconds):
        row = dict(meta[tag])
        row["converged"] = False
        row["measured_targets"] = None
        row["p3_satisfied"] = False
        if ok:
            y = {n: payload.get(n) for n in oracle.TARGET_NAMES}
            if all(v is not None for v in y.values()):
                row["converged"] = True
                row["measured_targets"] = y
                row["p3_satisfied"] = p3_satisfied(y)
        results.append(row)

    with open(STATE_DIR / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    conv = [r for r in results if r["converged"]]
    p3ok = [r for r in results if r["p3_satisfied"]]
    print(f"\n[latent_diff_probe] {len(conv)}/{len(results)} converged, {len(p3ok)} satisfy ALL 5 P3 constraints")
    if conv:
        qi = np.array([r["measured_targets"]["qi"] for r in conv])
        mr = np.array([r["measured_targets"]["edge_magnetic_mirror_ratio"] for r in conv])
        print(f"       qi range {qi.min():.6g}-{qi.max():.6g}, mirror_ratio range {mr.min():.4f}-{mr.max():.4f}")
        feas = qi <= P3_THRESH["qi"][1]
        print(f"       {feas.sum()} pass qi<=3.162e-4 on their own", end="")
        if feas.sum():
            print(f", best mirror_ratio among them: {mr[feas].min():.4f}")
        else:
            print()


if __name__ == "__main__":
    main()

"""Direct audit of whether the cVAE/diffusion/GAN comparison is actually a
fair one -- checked empirically against the real checkpoints and data, not
reasoned about in the abstract. Prompted by a direct user question ("are our
baseline method treatments valid? param/compute matched?") after §18-23
repeatedly found diffusion/GAN beating the cVAE; worth ruling out that the
gap is a methodology artifact before trusting it as architectural.

Checks four things, each with a real pass/fail reading printed at the end:

1. Param count parity at generation time (what actually runs to produce one
   candidate) -- not total params, which would unfairly penalize the cVAE
   for its train-only encoder.
2. Training compute parity -- WGAN-GP's n_critic=5 means "200 epochs" is
   not the same number of gradient steps across architectures.
3. Inference compute parity -- diffusion's T-step sampling chain vs. one
   forward pass for cVAE/GAN, run at *every* candidate generated in every
   eval script in this project.
4. Two candidate correctness gaps specific to how each architecture is
   *used* at sampling time, not just how it's trained:
   - cVAE: --beta=0.01 is a light KL penalty; if the encoder's aggregate
     posterior doesn't actually land close to N(0, I), then decode(z ~
     N(0, I), cond) at generation time is sampling z from a distribution
     the decoder was never really trained to decode well -- an
     eval-time/train-time mismatch specific to this architecture.
   - diffusion: T=200 with the standard T=1000-tuned linear beta schedule
     (1e-4 -> 0.02) reused unchanged -- check whether alpha_bar at t=T-1 is
     actually close enough to 0 that q(x_{T-1} | x_0) resembles pure noise.
     If not, ancestral sampling starting from x_T ~ N(0, I) starts from a
     distribution training never actually produced.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_cvae import CVAE
from train_diffusion import DiffusionDenoiser, make_schedule
from train_gan import Generator as GANGenerator

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def audit_domain(domain_name, cvae_tag, diffusion_tag, gan_tag, coeff_dim, n_targets, extra_cond_dim,
                  X_path, Y_path, coeff_slice, cond_builder):
    print(f"\n{'=' * 70}\n{domain_name}\n{'=' * 70}")
    dev = torch.device("cpu")

    cvae_ckpt = torch.load(CKPT_DIR / f"{cvae_tag}.pt", map_location=dev)
    diff_ckpt = torch.load(CKPT_DIR / f"{diffusion_tag}.pt", map_location=dev)
    gan_ckpt = torch.load(CKPT_DIR / f"{gan_tag}.pt", map_location=dev)

    cvae = CVAE(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=cvae_ckpt["latent_dim"],
                hidden=cvae_ckpt["hidden"], n_nfp=extra_cond_dim).to(dev)
    cvae.load_state_dict(cvae_ckpt["model_state_dict"])
    cvae.eval()

    diff = DiffusionDenoiser(coeff_dim=coeff_dim, n_targets=n_targets, hidden=diff_ckpt["hidden"],
                              time_embed_dim=diff_ckpt["time_embed_dim"], n_nfp=extra_cond_dim).to(dev)
    diff.load_state_dict(diff_ckpt["model_state_dict"])
    diff.eval()

    gan = GANGenerator(coeff_dim=coeff_dim, n_targets=n_targets, latent_dim=gan_ckpt["latent_dim"],
                        hidden=gan_ckpt["hidden"], n_nfp=extra_cond_dim).to(dev)
    gan.load_state_dict(gan_ckpt["generator_state_dict"])
    gan.eval()

    # ---- 1. param count at generation time ----
    cvae_decoder_params = n_params(cvae.decoder)
    diff_params = n_params(diff)
    gan_gen_params = n_params(gan)
    print(f"\n[1] Params actually run to produce ONE candidate:")
    print(f"    cVAE decoder:  {cvae_decoder_params:>8,}  (encoder, {n_params(cvae) - cvae_decoder_params:,} params, is train-only -- excluded, correctly)")
    print(f"    diffusion net: {diff_params:>8,}  (same network runs T={diff_ckpt['T']} times per sample -- see [3])")
    print(f"    GAN generator: {gan_gen_params:>8,}")
    spread = max(cvae_decoder_params, diff_params, gan_gen_params) / min(cvae_decoder_params, diff_params, gan_gen_params)
    print(f"    -> spread (max/min): {spread:.2f}x")

    # ---- 2. training compute parity ----
    n_rows = np.load(X_path).shape[0]
    batch = 512
    n_batches = n_rows // batch
    epochs = 200
    cvae_fwd_bwd_steps = n_batches * epochs  # one fwd+bwd through encoder+decoder per batch
    diff_fwd_bwd_steps = n_batches * epochs  # one fwd+bwd through the single denoiser per batch
    n_critic = 5
    gan_critic_steps = n_batches * epochs * n_critic  # critic fwd+bwd, n_critic x per batch
    gan_gen_steps = n_batches * epochs  # generator fwd+bwd, 1x per batch
    print(f"\n[2] Training-time network updates over {epochs} epochs ({n_rows:,} rows, batch={batch}, {n_batches}/epoch):")
    print(f"    cVAE:      {cvae_fwd_bwd_steps:>10,} encoder+decoder update steps")
    print(f"    diffusion: {diff_fwd_bwd_steps:>10,} denoiser update steps")
    print(f"    GAN:       {gan_gen_steps:>10,} generator update steps + {gan_critic_steps:>10,} critic update steps "
          f"(critic sees {n_critic}x the gradient steps of the generator itself)")
    print(f"    -> 'same 200 epochs' is NOT the same training compute across architectures: "
          f"GAN's critic alone gets {gan_critic_steps / cvae_fwd_bwd_steps:.1f}x the update steps of the cVAE/diffusion nets.")

    # ---- 3. inference compute parity ----
    T = diff_ckpt["T"]
    print(f"\n[3] Forward passes needed to generate ONE candidate at eval time:")
    print(f"    cVAE:      1  (single decode call)")
    print(f"    GAN:       1  (single generator call)")
    print(f"    diffusion: {T}  (full ancestral sampling chain, sequential)")
    print(f"    -> diffusion gets {T}x the inference-time compute of cVAE/GAN for every single candidate "
          f"scored in every eval script in this project. This is a real, acknowledged asymmetry "
          f"(train_diffusion.py's own docstring: T was picked for wall-clock parity with the eval harness's "
          f"time budget, not FLOP parity) -- diffusion's steerability/validity edge over the cVAE could be "
          f"partly 'more compute per sample', not purely architecture.")

    # ---- 4a. cVAE prior-matching check ----
    X = np.load(X_path)
    Y = np.load(Y_path)
    if domain_name.startswith("XFOIL"):
        # Same physical-sanity filter train_airfoil_cvae.py applies before
        # training -- omitting it here (an earlier version of this script
        # did) feeds the ~127/50,011 known-bad outlier rows (§22: cd down to
        # ~1e-12, l_over_d up to 9.8e10) through the SAME z-scoring stats
        # that were computed on the filtered set, producing enormous
        # out-of-distribution conditioning values and a spurious-looking
        # "blown up latent space" result that was actually just this bug.
        target_names_tmp = json.loads((OUT_DIR / "airfoil_target_names.json").read_text()) if (OUT_DIR / "airfoil_target_names.json").exists() else None
        cd_col = target_names_tmp.index("cd") if target_names_tmp else 1
        lod_col = target_names_tmp.index("l_over_d") if target_names_tmp else 3
        sane = (Y[:, cd_col] >= 1e-6) & (np.abs(Y[:, lod_col]) <= 300)
        n_dropped = len(Y) - sane.sum()
        print(f"\n    [filter] dropping {n_dropped}/{len(Y)} non-physical rows before the prior-matching check "
              f"(same filter train_airfoil_cvae.py trained on)")
        X, Y = X[sane], Y[sane]
    coeffs_raw = X[:, coeff_slice]
    coeff_mean, coeff_std = cvae_ckpt["coeff_mean"], cvae_ckpt["coeff_std"]
    coeffs = torch.tensor((coeffs_raw - coeff_mean) / coeff_std, dtype=torch.float32)
    cond = cond_builder(X, Y, cvae_ckpt)
    with torch.no_grad():
        mu, logvar = cvae.encode(coeffs, cond)
    mu_np, std_np = mu.numpy(), (0.5 * logvar).exp().numpy()
    per_dim_mu_mean = mu_np.mean(axis=0)   # ideal: 0
    per_dim_mu_std = mu_np.std(axis=0)     # ideal: 1 (aggregate posterior variance should also contribute ~1 total with within-sample var)
    mean_within_sample_std = std_np.mean()  # ideal: close to 1 if posterior ~ prior
    print(f"\n[4a] cVAE aggregate posterior vs. the N(0, I) prior it's sampled from at generation time "
          f"(beta={0.01}, KL only lightly penalized):")
    print(f"     mean of per-dim posterior means (ideal 0):      {per_dim_mu_mean.mean():+.3f}  "
          f"(range {per_dim_mu_mean.min():+.3f} to {per_dim_mu_mean.max():+.3f})")
    print(f"     mean of per-dim posterior mean-STDs across data (ideal 1, this is spread of mu, not within-sample std): "
          f"{per_dim_mu_std.mean():.3f}  (range {per_dim_mu_std.min():.3f} to {per_dim_mu_std.max():.3f})")
    print(f"     mean within-sample std exp(0.5*logvar) (ideal ~1): {mean_within_sample_std:.3f}")
    total_var = (per_dim_mu_std ** 2 + std_np.mean(axis=0) ** 2).mean()
    print(f"     total aggregate variance per dim (mu-spread^2 + mean within-sample-var^2, ideal ~1): {total_var:.3f}")

    # ---- 4b. diffusion schedule sanity ----
    schedule = make_schedule(T)
    alpha_bar_final = schedule["alpha_bars"][-1].item()
    print(f"\n[4b] Diffusion schedule sanity (T={T}, linear beta 1e-4->0.02, the standard T=1000 schedule reused unadjusted):")
    print(f"     alpha_bar at t=T-1: {alpha_bar_final:.4f}  "
          f"(sqrt(alpha_bar)={math.sqrt(alpha_bar_final):.3f} residual signal, "
          f"sqrt(1-alpha_bar)={math.sqrt(1 - alpha_bar_final):.3f} noise)")
    print(f"     -> at T=1000 (paper default) this is ~1e-5 (fully destroyed). At T={T}, "
          f"{'this is NOT close to fully destroyed -- a real train/sample mismatch' if alpha_bar_final > 0.01 else 'this is reasonably close to fully destroyed'}: "
          f"ancestral sampling starts x_T from pure N(0, I), but training's forward process at t=T-1 never "
          f"actually reaches that little residual signal.")

    return {
        "param_spread": spread,
        "gan_critic_step_multiple": gan_critic_steps / cvae_fwd_bwd_steps,
        "diffusion_inference_multiple": T,
        "cvae_prior_mean_mag": float(abs(per_dim_mu_mean.mean())),
        "cvae_prior_total_var": float(total_var),
        "diffusion_alpha_bar_final": alpha_bar_final,
    }


def vmec_cond_builder(X, Y, ckpt):
    from make_splits import LOG_TARGET_NAMES
    from train_vae import nfp_one_hot
    target_names = ckpt["target_names"]
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
    Yt = Y.copy()
    for name in LOG_TARGET_NAMES:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    targets_z = (Yt - t_mean) / t_std
    nfp = torch.tensor(X[:, 90], dtype=torch.float32)
    return torch.cat([torch.tensor(targets_z, dtype=torch.float32), nfp_one_hot(nfp)], dim=-1)


def airfoil_cond_builder(X, Y, ckpt):
    target_names = ckpt["target_names"]
    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
    log_target_names = ckpt["log_target_names"]
    coeff_dim = ckpt["coeff_dim"]
    Yt = Y.copy()
    for name in log_target_names:
        idx = target_names.index(name)
        Yt[:, idx] = np.log(np.clip(Yt[:, idx], 1e-12, None))
    targets_z = (Yt - t_mean) / t_std
    reynolds_raw, alpha_raw = X[:, coeff_dim], X[:, coeff_dim + 1]
    log_reynolds = np.log(reynolds_raw)
    re_mean, re_std = ckpt["reynolds_mean"], ckpt["reynolds_std"]
    al_mean, al_std = ckpt["alpha_mean"], ckpt["alpha_std"]
    aux = np.stack([(log_reynolds - re_mean) / re_std, (alpha_raw - al_mean) / al_std], axis=-1)
    return torch.cat([torch.tensor(targets_z, dtype=torch.float32), torch.tensor(aux, dtype=torch.float32)], dim=-1)


def main():
    results = {}
    results["vmec"] = audit_domain(
        "VMEC++ / stellarators", "cvae_targets_full_s0", "diffusion_targets_full_s0", "gan_targets_full_s0",
        coeff_dim=90, n_targets=11, extra_cond_dim=5,
        X_path=OUT_DIR / "X.npy", Y_path=OUT_DIR / "Y.npy", coeff_slice=slice(0, 90),
        cond_builder=vmec_cond_builder,
    )
    results["airfoil"] = audit_domain(
        "XFOIL / airfoils", "airfoil_cvae_s0", "airfoil_diffusion_s0", "airfoil_gan_s0",
        coeff_dim=16, n_targets=4, extra_cond_dim=2,
        X_path=OUT_DIR / "airfoil_X.npy", Y_path=OUT_DIR / "airfoil_Y.npy", coeff_slice=slice(0, 16),
        cond_builder=airfoil_cond_builder,
    )
    out_path = OUT_DIR / "audit_baselines.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

"""generate_candidates.py generalized to also gradient-ALM-search through the
GAN and cVAE generators, not just the VAE decoder -- same frozen scorer
ensemble, same ALM optimization loop, same paper-matched feasibility
definition and JSONL output shape (drop-in compatible with p1_report.py/
p2_report.py/p3_report.py/append_oracle_master.py). Diffusion is
deliberately NOT included here: its generation is iterative denoising, not
a single differentiable z->x map, so it can't be gradient-searched the same
way without backprop-through-sampling (a separate, heavier project) --
compare it instead via direct target-conditioned generation
(bootstrap_generic.py --seed-mode fixed), a different technique evaluated
alongside these, not folded into the same search loop.

Generator-specific handling:
  --generator vae   z0 seeded from encoding real (near-)anchor coefficients
                     (matches generate_candidates.py exactly); cond = nfp
                     one-hot only; the target is reached purely through the
                     ALM loss on z, never told to the generator directly.
  --generator gan/cvae  z0 ~ N(0, I) (their own training-time prior, no
                     natural "encode a real anchor" step for GAN); cond =
                     [z-scored target vector, nfp one-hot], with the target
                     vector FIXED for the whole run via build_fixed_target_z
                     (reused verbatim from bootstrap_generic.py) -- i.e. the
                     generator is handed the actual desired target directly
                     as conditioning, and ALM only searches the residual
                     latent z. This tests a genuinely different hypothesis
                     than the VAE case: does target-conditioning + gradient
                     refinement reach further off-manifold than gradient
                     search with no target-conditioning at all?
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from optimize import (  # noqa: E402
    OUT_DIR, CKPT_DIR, load_ensemble, ensemble_predict, parse_constraint,
    paper_feasibility_violation, violation,
)
from gradient_walk import load_vae  # noqa: E402
from train_vae import nfp_one_hot  # noqa: E402
from train import load_split, IDX_NFP  # noqa: E402
from generate_and_validate import ZERO_COEFF_IDX  # noqa: E402
from bootstrap_generic import build_fixed_target_z, parse_kv_floats  # noqa: E402
from train_gan import Generator as GanGenerator  # noqa: E402
from train_cvae import CVAE  # noqa: E402


def load_gan(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev, weights_only=False)
    gen = GanGenerator(coeff_dim=90, n_targets=len(ckpt["target_names"]),
                        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen, ckpt


def load_cvae_gen(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev, weights_only=False)
    model = CVAE(coeff_dim=90, n_targets=len(ckpt["target_names"]),
                 latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", required=True, choices=["vae", "gan", "cvae"])
    p.add_argument("--generator-tag", default=None,
                    help="defaults to vae_coeffs_full_s0 / gan_targets_full_s0 / cvae_targets_full_s0")
    p.add_argument("--target-override", default=None,
                    help="gan/cvae only: raw-space target overrides fed as conditioning, e.g. "
                         "'aspect_ratio=4.0,average_triangularity=-0.5,"
                         "edge_rotational_transform_over_n_field_periods=0.3,max_elongation=2.8'. "
                         "Unspecified metrics default to the population mean (z=0), matching "
                         "bootstrap_generic.py's --seed-mode=fixed convention exactly.")
    p.add_argument("--member-tags", nargs="+",
                    default=["reg_mlp_big_full_s0", "reg_mlp_big_full_s1", "reg_mlp_big_full_s2"])
    p.add_argument("--minimize", default="max_elongation")
    p.add_argument("--maximize", default=None)
    p.add_argument("--constraint", action="append", default=None)
    p.add_argument("--nfp", type=int, default=3)
    p.add_argument("--n-starts", type=int, default=150)
    p.add_argument("--n-candidates", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--alm-outer-iters", type=int, default=40)
    p.add_argument("--alm-inner-steps", type=int, default=20)
    p.add_argument("--alm-rho0", type=float, default=10.0)
    p.add_argument("--alm-rho-max", type=float, default=1e9)
    p.add_argument("--alm-tau", type=float, default=0.8)
    p.add_argument("--alm-sigma", type=float, default=5.0)
    p.add_argument("--trust-weight", type=float, default=0.1)
    p.add_argument("--latent-weight", type=float, default=0.05)
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=str(OUT_DIR / "generated_candidates_multigen.jsonl"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.constraint is None:
        args.constraint = ["aspect_ratio<=4.0", "average_triangularity<=-0.5",
                            "abs(edge_rotational_transform_over_n_field_periods)>=0.3"]
    if bool(args.minimize) == bool(args.maximize):
        raise ValueError("pass exactly one of --minimize or --maximize")

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    target_stats = meta["target_stats"]

    models, target_names = load_ensemble(args.member_tags, dev)
    target_std = torch.tensor([target_stats[n]["std"] for n in target_names], device=dev)

    obj_name = args.minimize or args.maximize
    obj_idx = target_names.index(obj_name)
    obj_sign = 1.0 if args.minimize else -1.0
    obj_std = target_stats[obj_name]["std"]

    constraints = []
    for spec in args.constraint:
        name, op, value, use_abs = parse_constraint(spec)
        constraints.append((target_names.index(name), op, value, target_stats[name]["std"], name, use_abs))
    n_c = len(constraints)

    def constraint_tilde(mean):
        if n_c == 0:
            return torch.zeros((mean.shape[0], 0), device=mean.device)
        return torch.stack(
            [paper_feasibility_violation(
                mean[:, idx_c], op, value, use_abs=use_abs,
                divisor=(max(0.1, abs(value)) if _name == "vacuum_well" else None))
             for idx_c, op, value, _std_c, _name, use_abs in constraints],
            dim=1,
        )

    zero_mask = torch.ones(90, device=dev)
    zero_mask[ZERO_COEFF_IDX] = 0.0
    seed_nfp = torch.full((args.n_starts,), float(args.nfp), device=dev)
    nfp_cond = nfp_one_hot(seed_nfp)
    nfp_col = seed_nfp.unsqueeze(1)
    sym_col = torch.ones((args.n_starts, 1), device=dev)

    if args.generator == "vae":
        gen_tag = args.generator_tag or "vae_coeffs_full_s0"
        vae, coeff_mean, coeff_std = load_vae(gen_tag)
        vae = vae.to(dev)
        coeff_mean_t = torch.tensor(coeff_mean, dtype=torch.float32, device=dev)
        coeff_std_t = torch.tensor(coeff_std, dtype=torch.float32, device=dev)

        X_train, Y_train = load_split("train")
        same_nfp = X_train[:, IDX_NFP] == float(args.nfp)
        pool = X_train[same_nfp] if same_nfp.any() else X_train
        pool_Y = Y_train[same_nfp] if same_nfp.any() else Y_train
        if constraints:
            real_violation = torch.zeros(pool_Y.shape[0])
            for idx_c, op, value, std_c, _name, use_abs in constraints:
                real_violation += violation(pool_Y[:, idx_c], op, value, std_c, use_abs=use_abs)
            n_guided = args.n_starts // 2
            n_random = args.n_starts - n_guided
            candidate_pool = torch.argsort(real_violation)[:max(n_guided * 4, 16)]
            guided_idx = candidate_pool[torch.randint(0, candidate_pool.shape[0], (n_guided,))]
            random_idx = torch.randint(0, pool.shape[0], (n_random,))
            idx = torch.cat([guided_idx, random_idx])
        else:
            idx = torch.randint(0, pool.shape[0], (args.n_starts,))
        seed_coeffs = ((pool[idx][:, :90].to(dev) - coeff_mean_t) / coeff_std_t)
        with torch.no_grad():
            z0, _ = vae.encode(seed_coeffs, nfp_cond)

        def decode_x(z):
            return (vae.decode(z, nfp_cond) * coeff_std_t + coeff_mean_t) * zero_mask

    else:
        gen_tag = args.generator_tag or (f"{args.generator}_targets_full_s0")
        loader = load_gan if args.generator == "gan" else load_cvae_gen
        gen_model, ckpt = loader(gen_tag, dev)
        coeff_mean_t = torch.tensor(ckpt["coeff_mean"], dtype=torch.float32, device=dev)
        coeff_std_t = torch.tensor(ckpt["coeff_std"], dtype=torch.float32, device=dev)
        gan_target_names = ckpt["target_names"]
        t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
        log_target_names = ckpt["log_target_names"]

        overrides = parse_kv_floats(args.target_override)
        fixed_z = build_fixed_target_z(overrides, gan_target_names, log_target_names, t_mean, t_std)
        target_cond_row = torch.tensor(fixed_z, dtype=torch.float32, device=dev)
        target_cond_batch = target_cond_row.unsqueeze(0).expand(args.n_starts, -1)
        gen_cond = torch.cat([target_cond_batch, nfp_cond], dim=-1)
        print(f"[generate_candidates_multigen] {args.generator} conditioned on raw-space overrides "
              f"{overrides} (unspecified metrics at population mean)")

        z0 = torch.randn(args.n_starts, gen_model.latent_dim, device=dev)

        def decode_x(z):
            raw = gen_model.decode(z, gen_cond) if args.generator == "cvae" else gen_model(z, gen_cond)
            return (raw * coeff_std_t + coeff_mean_t) * zero_mask

    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.lr)
    rho = torch.full((args.n_starts, n_c), args.alm_rho0, device=dev)
    y = torch.zeros((args.n_starts, n_c), device=dev)
    prev_c = None

    print(f"[generate_candidates_multigen] generator={args.generator} ({gen_tag})  {args.n_starts} starts, "
          f"{'minimize' if args.minimize else 'maximize'} {obj_name}, nfp={args.nfp}, members={args.member_tags}")

    for outer in range(args.alm_outer_iters):
        for _inner in range(args.alm_inner_steps):
            opt.zero_grad()
            x = decode_x(z)
            x_full = torch.cat([x, nfp_col, sym_col], dim=1)
            mean, std = ensemble_predict(models, x_full)

            loss = obj_sign * mean[:, obj_idx] / obj_std
            penalty_scale = (rho.amax(dim=1) / args.alm_rho0) if n_c > 0 else torch.ones(args.n_starts, device=dev)
            if n_c > 0:
                c = constraint_tilde(mean)
                inner_term = torch.relu(y + rho * c)
                loss = loss + ((inner_term ** 2 - y ** 2) / (2 * rho)).sum(dim=1)
            if args.latent_weight > 0:
                latent_pen = (z ** 2).mean(dim=1)
                loss = loss + args.latent_weight * penalty_scale * latent_pen
            if args.trust_weight > 0:
                trust_pen = (std / target_std).mean(dim=1)
                loss = loss + args.trust_weight * penalty_scale * trust_pen

            loss.sum().backward()
            opt.step()

        if n_c > 0:
            with torch.no_grad():
                x = decode_x(z)
                x_full = torch.cat([x, nfp_col, sym_col], dim=1)
                mean, _ = ensemble_predict(models, x_full)
                c = constraint_tilde(mean)
                if prev_c is not None:
                    shrunk_enough = c <= args.alm_tau * prev_c
                    rho = torch.where(shrunk_enough, rho, (rho * args.alm_sigma).clamp(max=args.alm_rho_max))
                y = torch.relu(y + rho * c)
                prev_c = c

    with torch.no_grad():
        x = decode_x(z)
        x_full = torch.cat([x, nfp_col, sym_col], dim=1)
        mean, std = ensemble_predict(models, x_full)

        c_final = constraint_tilde(mean)
        feasible = (c_final <= args.relative_tol).all(dim=1) if n_c > 0 else torch.ones(args.n_starts, dtype=torch.bool, device=dev)
        obj_values = obj_sign * mean[:, obj_idx]
        n_feasible = feasible.sum().item()

        feasible_idx = torch.nonzero(feasible, as_tuple=True)[0]
        ranked = feasible_idx[torch.argsort(obj_values[feasible_idx])]
        keep = ranked[:min(args.n_candidates, n_feasible)].tolist()

        print(f"\n=== {n_feasible}/{args.n_starts} starts surrogate-feasible "
              f"(relative_tol={args.relative_tol}) -- saving {len(keep)} (NOT VMEC++-validated) ===")
        if len(keep) < args.n_candidates:
            print(f"[warn] only {len(keep)} surrogate-feasible candidates available, "
                  f"short of the requested {args.n_candidates} -- try a larger --n-starts")

        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            for rank, start_idx in enumerate(keep):
                row = x[start_idx].cpu().tolist()
                design = {
                    "rank": rank,
                    "r_cos": row[:45], "z_sin": row[45:90],
                    "n_field_periods": args.nfp, "is_stellarator_symmetric": 1.0,
                    "generator": args.generator, "generator_tag": gen_tag,
                    "validated": False,
                    "predicted_targets": {name: mean[start_idx, k].item() for k, name in enumerate(target_names)},
                    "predicted_targets_std": {name: std[start_idx, k].item() for k, name in enumerate(target_names)},
                }
                f.write(json.dumps(design) + "\n")

        if keep:
            best_obj = mean[keep[0], obj_idx].item()
            worst_obj = mean[keep[-1], obj_idx].item()
            print(f"predicted {obj_name} range among saved: best {best_obj:.6g}, worst {worst_obj:.6g}")
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()

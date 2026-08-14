"""Bulk-generate surrogate-feasible candidates for the same latent-space
design search as scripts/latent_optimize.py, WITHOUT the real-VMEC++
validation step -- purely surrogate-predicted, not checked. Reuses the
exact same ALM optimization loop (same optimize.py machinery, same
paper-matched feasibility definition), just run with more multi-starts and,
instead of validating a top-k subset for real, dumps every surrogate-
feasible candidate (up to --n-candidates, best predicted objective first)
straight to a JSONL file: one design per line, same r_cos/z_sin/nfp/
predicted-targets shape as latent_optimize.py's --save, minus the
measured_targets/really_feasible fields (there's no real check here).

Defaults to this session's confirmed-best model: the standard (raw-
coefficient) reg_mlp_big_full_s{0,1,2} ensemble + vae_coeffs_full_s0, and
the first ConStellaration geometric task (minimize max_elongation s.t.
aspect_ratio<=4.0, average_triangularity<=-0.5,
|edge_rotational_transform/nfp|>=0.3, nfp=3).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from generate_and_validate import ZERO_COEFF_IDX
from gradient_walk import load_vae
from optimize import (
    OUT_DIR,
    ensemble_predict,
    load_ensemble,
    paper_feasibility_violation,
    parse_constraint,
    violation,
)
from train import IDX_NFP, load_split
from train_vae import nfp_one_hot


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--member-tags",
        nargs="+",
        default=["reg_mlp_big_full_s0", "reg_mlp_big_full_s1", "reg_mlp_big_full_s2"],
    )
    p.add_argument("--vae-tag", default="vae_coeffs_full_s0")
    p.add_argument("--minimize", default="max_elongation")
    p.add_argument("--maximize", default=None)
    p.add_argument(
        "--constraint",
        action="append",
        default=[
            "aspect_ratio<=4.0",
            "average_triangularity<=-0.5",
            "abs(edge_rotational_transform_over_n_field_periods)>=0.3",
        ],
    )
    p.add_argument("--nfp", type=int, default=3)
    p.add_argument(
        "--n-starts", type=int, default=150, help="multi-start population size"
    )
    p.add_argument(
        "--n-candidates",
        type=int,
        default=100,
        help="max surrogate-feasible candidates to save",
    )
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
    p.add_argument(
        "--save", default=str(OUT_DIR / "generated_candidates_unvalidated.jsonl")
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if bool(args.minimize) == bool(args.maximize):
        raise ValueError("pass exactly one of --minimize or --maximize")

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    target_stats = meta["target_stats"]

    vae, coeff_mean, coeff_std = load_vae(args.vae_tag)
    vae = vae.to(dev)
    coeff_mean_t = torch.tensor(coeff_mean, dtype=torch.float32, device=dev)
    coeff_std_t = torch.tensor(coeff_std, dtype=torch.float32, device=dev)
    zero_mask = torch.ones(90, device=dev)
    zero_mask[ZERO_COEFF_IDX] = 0.0

    models, target_names = load_ensemble(args.member_tags, dev)
    target_std = torch.tensor(
        [target_stats[n]["std"] for n in target_names], device=dev
    )

    obj_name = args.minimize or args.maximize
    obj_idx = target_names.index(obj_name)
    obj_sign = 1.0 if args.minimize else -1.0
    obj_std = target_stats[obj_name]["std"]

    constraints = []
    for spec in args.constraint:
        name, op, value, use_abs = parse_constraint(spec)
        constraints.append(
            (
                target_names.index(name),
                op,
                value,
                target_stats[name]["std"],
                name,
                use_abs,
            )
        )
    n_c = len(constraints)

    def constraint_tilde(mean):
        if n_c == 0:
            return torch.zeros((mean.shape[0], 0), device=mean.device)
        return torch.stack(
            [
                paper_feasibility_violation(mean[:, idx_c], op, value, use_abs=use_abs)
                for idx_c, op, value, _std_c, _name, use_abs in constraints
            ],
            dim=1,
        )

    X_train, Y_train = load_split("train")
    same_nfp = X_train[:, IDX_NFP] == float(args.nfp)
    pool = X_train[same_nfp] if same_nfp.any() else X_train
    pool_Y = Y_train[same_nfp] if same_nfp.any() else Y_train

    if constraints:
        real_violation = torch.zeros(pool_Y.shape[0])
        for idx_c, op, value, std_c, _name, use_abs in constraints:
            real_violation += violation(
                pool_Y[:, idx_c], op, value, std_c, use_abs=use_abs
            )
        n_guided = args.n_starts // 2
        n_random = args.n_starts - n_guided
        candidate_pool = torch.argsort(real_violation)[: max(n_guided * 4, 16)]
        guided_idx = candidate_pool[
            torch.randint(0, candidate_pool.shape[0], (n_guided,))
        ]
        random_idx = torch.randint(0, pool.shape[0], (n_random,))
        idx = torch.cat([guided_idx, random_idx])
    else:
        idx = torch.randint(0, pool.shape[0], (args.n_starts,))

    seed_coeffs = (pool[idx][:, :90].to(dev) - coeff_mean_t) / coeff_std_t
    seed_nfp = torch.full((args.n_starts,), float(args.nfp), device=dev)
    with torch.no_grad():
        z0, _ = vae.encode(seed_coeffs, nfp_one_hot(seed_nfp))
    z = z0.clone().requires_grad_(True)
    nfp_cond = nfp_one_hot(seed_nfp)
    nfp_col = seed_nfp.unsqueeze(1)
    sym_col = torch.ones((args.n_starts, 1), device=dev)

    def decode_x(z):
        return (vae.decode(z, nfp_cond) * coeff_std_t + coeff_mean_t) * zero_mask

    opt = torch.optim.Adam([z], lr=args.lr)
    rho = torch.full((args.n_starts, n_c), args.alm_rho0, device=dev)
    y = torch.zeros((args.n_starts, n_c), device=dev)
    prev_c = None

    print(
        f"[generate_candidates] {args.n_starts} starts, minimize {obj_name}, nfp={args.nfp}, "
        f"vae={args.vae_tag}, members={args.member_tags}"
    )

    for outer in range(args.alm_outer_iters):
        for _inner in range(args.alm_inner_steps):
            opt.zero_grad()
            x = decode_x(z)
            x_full = torch.cat([x, nfp_col, sym_col], dim=1)
            mean, std = ensemble_predict(models, x_full)

            loss = obj_sign * mean[:, obj_idx] / obj_std
            penalty_scale = (
                (rho.amax(dim=1) / args.alm_rho0)
                if n_c > 0
                else torch.ones(args.n_starts, device=dev)
            )
            if n_c > 0:
                c = constraint_tilde(mean)
                inner_term = torch.relu(y + rho * c)
                loss = loss + ((inner_term**2 - y**2) / (2 * rho)).sum(dim=1)
            if args.latent_weight > 0:
                latent_pen = (z**2).mean(dim=1)
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
                    rho = torch.where(
                        shrunk_enough,
                        rho,
                        (rho * args.alm_sigma).clamp(max=args.alm_rho_max),
                    )
                y = torch.relu(y + rho * c)
                prev_c = c

    with torch.no_grad():
        x = decode_x(z)
        x_full = torch.cat([x, nfp_col, sym_col], dim=1)
        mean, std = ensemble_predict(models, x_full)

        c_final = constraint_tilde(mean)
        if n_c > 0:
            feasible = (c_final <= args.relative_tol).all(dim=1)
        else:
            feasible = torch.ones(args.n_starts, dtype=torch.bool, device=dev)
        obj_values = obj_sign * mean[:, obj_idx]
        n_feasible = feasible.sum().item()

        feasible_idx = torch.nonzero(feasible, as_tuple=True)[0]
        ranked = feasible_idx[torch.argsort(obj_values[feasible_idx])]
        keep = ranked[: min(args.n_candidates, n_feasible)].tolist()

        print(
            f"\n=== {n_feasible}/{args.n_starts} starts surrogate-feasible "
            f"(relative_tol={args.relative_tol}) -- saving {len(keep)} (NOT VMEC++-validated) ==="
        )
        if len(keep) < args.n_candidates:
            print(
                f"[warn] only {len(keep)} surrogate-feasible candidates available, "
                f"short of the requested {args.n_candidates} -- try a larger --n-starts"
            )

        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            for rank, start_idx in enumerate(keep):
                row = x[start_idx].cpu().tolist()
                design = {
                    "rank": rank,
                    "r_cos": row[:45],
                    "z_sin": row[45:90],
                    "n_field_periods": args.nfp,
                    "is_stellarator_symmetric": 1.0,
                    "vae_tag": args.vae_tag,
                    "validated": False,
                    "predicted_targets": {
                        name: mean[start_idx, k].item()
                        for k, name in enumerate(target_names)
                    },
                    "predicted_targets_std": {
                        name: std[start_idx, k].item()
                        for k, name in enumerate(target_names)
                    },
                }
                f.write(json.dumps(design) + "\n")

        if keep:
            best_obj = mean[keep[0], obj_idx].item()
            worst_obj = mean[keep[-1], obj_idx].item()
            print(
                f"predicted {obj_name} range among saved: best {best_obj:.6g}, worst {worst_obj:.6g}"
            )
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()

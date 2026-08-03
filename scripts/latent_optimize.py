"""Design search against the frozen surrogate ensemble, run in the VAE's
latent space instead of raw 90-dim Fourier-coefficient space -- and, unlike
`optimize.py`, ending in a real VMEC++ check rather than trusting the
surrogate's own verdict.

Same underlying constrained-optimization machinery as `optimize.py` (ALM,
augmented-Lagrangian per-constraint penalty/multiplier, same paper-matched
feasibility definition) -- reused directly from there, not reimplemented.
What's different:

  - The optimization variable is a latent vector z, decoded through a VAE
    checkpoint every step (`x = vae.decode(z, nfp_cond)`) rather than the 90
    coefficients directly. Defaults to the *latest bootstrap-loop
    generation's* VAE (auto-detected from checkpoints/, e.g.
    `bootstrap0_gen30`) -- self-trained across real + ~90K synthetic
    VMEC++-validated designs (see EXPERIMENT_LOG §9-10), a much richer model
    of the feasible manifold than the original real-data-only VAE. Staying
    in this VAE's latent space is the replacement for optimize.py's
    nearest-real-neighbor distance penalty: instead of explicitly penalizing
    distance to a real point, the decoder itself only knows how to produce
    points that look like the (now much larger) training distribution.
  - A small penalty on ||z||^2 (`--latent-weight`) keeps the search from
    drifting into a region of latent space the decoder never saw during
    training (its role is directly analogous to optimize.py's
    --distance-weight, just measured in latent space instead of coefficient
    space).
  - After the ALM search, the top `--top-k` candidates by surrogate-reported
    objective/feasibility are validated for real through VMEC++ (reusing
    generate_and_validate.py's subprocess-per-candidate/timeout machinery) --
    the whole point of §8's earlier finding is that a surrogate-feasible
    candidate is not necessarily really feasible, so this script doesn't
    report a "winner" without checking. Predicted vs. measured values are
    printed side by side for every validated candidate, and the saved design
    (if any) is the best one that's *really* feasible, not just
    surrogate-feasible.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from optimize import (  # noqa: E402
    CKPT_DIR, OUT_DIR, load_ensemble, ensemble_predict, parse_constraint,
    violation, paper_feasibility_violation, elongation_style_score, FEATURE_NAMES_90,
)
from gradient_walk import load_vae  # noqa: E402
from train_vae import nfp_one_hot  # noqa: E402
from train import load_split, IDX_NFP  # noqa: E402
from generate_and_validate import run_batch_with_timeout, ZERO_COEFF_IDX  # noqa: E402

GEN_TAG_RE = re.compile(r"^(.+)_gen(\d+)\.pt$")


def latest_bootstrap_vae_tag(bootstrap_tag, fallback="vae_coeffs_s0"):
    """Scans checkpoints/ for {bootstrap_tag}_gen{N}.pt and returns the tag
    with the highest N -- the most-recently-trained generation of the
    self-training loop's VAE, i.e. the one that's seen the most synthetic
    data. Falls back to the original real-data-only VAE if the bootstrap
    loop was never run under this tag."""
    best_gen, best_tag = -1, None
    for path in CKPT_DIR.glob(f"{bootstrap_tag}_gen*.pt"):
        m = GEN_TAG_RE.match(path.name)
        if m and m.group(1) == bootstrap_tag:
            gen = int(m.group(2))
            if gen > best_gen:
                best_gen, best_tag = gen, path.stem
    if best_tag is None:
        print(f"[latent_optimize] no {bootstrap_tag}_gen*.pt checkpoints found -- "
              f"falling back to {fallback!r}")
        return fallback
    print(f"[latent_optimize] using {best_tag} (generation {best_gen} of the {bootstrap_tag} bootstrap loop)")
    return best_tag


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member-tags", nargs="+",
                    default=["reg_mlp_big_soap_s0", "reg_mlp_big_soap_s1", "reg_mlp_big_soap_s2"])
    p.add_argument("--vae-tag", default=None,
                    help="VAE checkpoint to search in the latent space of. Default: auto-detect the "
                         "latest generation of --bootstrap-tag's self-training loop.")
    p.add_argument("--bootstrap-tag", default="bootstrap0", help="used for --vae-tag auto-detection only")
    p.add_argument("--minimize", default=None, help="target name to minimize")
    p.add_argument("--maximize", default=None, help="target name to maximize (mutually exclusive with --minimize)")
    p.add_argument("--constraint", action="append", default=[],
                    help="e.g. --constraint 'aspect_ratio<=4.0' --constraint 'abs(edge_rotational_transform_over_n_field_periods)>=0.3'. Repeatable.")
    p.add_argument("--nfp", type=int, required=True, help="fixed n_field_periods for this search")
    p.add_argument("--n-starts", type=int, default=64, help="multi-start population size")
    p.add_argument("--lr", type=float, default=0.02, help="Adam lr on the latent vector (latent space is "
                                                            "smaller/smoother than raw coefficient space, so this "
                                                            "can run a bit hotter than optimize.py's default)")
    p.add_argument("--alm-outer-iters", type=int, default=40)
    p.add_argument("--alm-inner-steps", type=int, default=20)
    p.add_argument("--alm-rho0", type=float, default=10.0)
    p.add_argument("--alm-rho-max", type=float, default=1e9)
    p.add_argument("--alm-tau", type=float, default=0.8)
    p.add_argument("--alm-sigma", type=float, default=5.0)
    p.add_argument("--trust-weight", type=float, default=0.1,
                    help="penalty on ensemble inter-member disagreement, same role as in optimize.py")
    p.add_argument("--latent-weight", type=float, default=0.05,
                    help="penalty on mean(z^2) -- keeps z near the VAE's N(0,1) prior, i.e. inside the "
                         "region the decoder actually learned. This is this script's analogue of "
                         "optimize.py's --distance-weight; 0 disables it (not recommended).")
    p.add_argument("--relative-tol", type=float, default=1e-2,
                    help="ConStellaration benchmark's own feasibility tolerance, same definition as optimize.py")
    p.add_argument("--score-bounds", type=float, nargs=2, default=None, metavar=("LOWER", "UPPER"))
    p.add_argument("--top-k", type=int, default=8,
                    help="how many surrogate-ranked candidates to actually validate through real VMEC++ "
                         "-- feasible-first by surrogate objective, then least-surrogate-infeasible filling "
                         "any remaining slots, so near-misses get checked too, not just the reported best")
    p.add_argument("--n-workers", type=int, default=28, help="parallel VMEC++ validation workers")
    p.add_argument("--timeout-seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="path to save the best REALLY-feasible design as JSON")
    p.add_argument("--progress-log", default=None,
                    help="optional path to append one JSON line summarizing this run's outcome -- for "
                         "tracking whether repeated checkpoint runs (e.g. one per block of a long "
                         "bootstrap+retrain loop) are actually converging, without re-reading full output")
    p.add_argument("--run-label", default=None, help="free-text label stored in --progress-log's line, "
                                                       "e.g. a block/generation number from the caller")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if bool(args.minimize) == bool(args.maximize):
        raise ValueError("pass exactly one of --minimize or --maximize")

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    target_stats = meta["target_stats"]

    vae_tag = args.vae_tag or latest_bootstrap_vae_tag(args.bootstrap_tag)
    vae, coeff_mean, coeff_std = load_vae(vae_tag)
    vae = vae.to(dev)
    coeff_mean_t = torch.tensor(coeff_mean, dtype=torch.float32, device=dev)
    coeff_std_t = torch.tensor(coeff_std, dtype=torch.float32, device=dev)
    zero_mask = torch.ones(90, device=dev)
    zero_mask[ZERO_COEFF_IDX] = 0.0

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
            [paper_feasibility_violation(mean[:, idx_c], op, value, use_abs=use_abs)
             for idx_c, op, value, _std_c, _name, use_abs in constraints],
            dim=1,
        )

    # Multi-start init: encode real, same-nfp training rows into this VAE's
    # latent space -- same rationale as optimize.py's real-point seeding
    # (start somewhere the model has actual support), and when constraints
    # are given, half the starts are seeded from the real points closest to
    # satisfying them (ranked by true target values), the rest random, for
    # the same reasons optimize.py does it.
    X_train, Y_train = load_split("train")
    same_nfp = X_train[:, IDX_NFP] == float(args.nfp)
    pool = X_train[same_nfp] if same_nfp.any() else X_train
    pool_Y = Y_train[same_nfp] if same_nfp.any() else Y_train
    if not same_nfp.any():
        print(f"[warn] no training rows with n_field_periods == {args.nfp}; seeding from all rows instead")

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
        print(f"[seed] {n_guided} starts from the {candidate_pool.shape[0]} real same-nfp points closest "
              f"to feasible (min true violation {real_violation[candidate_pool[0]].item():.4g}), {n_random} random")
    else:
        idx = torch.randint(0, pool.shape[0], (args.n_starts,))

    seed_coeffs = ((pool[idx][:, :90].to(dev) - coeff_mean_t) / coeff_std_t)
    seed_nfp = torch.full((args.n_starts,), float(args.nfp), device=dev)
    with torch.no_grad():
        z0, _ = vae.encode(seed_coeffs, nfp_one_hot(seed_nfp))
    z = z0.clone().requires_grad_(True)
    nfp_cond = nfp_one_hot(seed_nfp)  # fixed per start for the whole run, matches optimize.py's fixed nfp
    nfp_col = seed_nfp.unsqueeze(1)
    sym_col = torch.ones((args.n_starts, 1), device=dev)

    def decode_x(z):
        return (vae.decode(z, nfp_cond) * coeff_std_t + coeff_mean_t) * zero_mask

    opt = torch.optim.Adam([z], lr=args.lr)
    rho = torch.full((args.n_starts, n_c), args.alm_rho0, device=dev)
    y = torch.zeros((args.n_starts, n_c), device=dev)
    prev_c = None

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
        if n_c > 0:
            feasible = (c_final <= args.relative_tol).all(dim=1)
            max_violation = c_final.clamp(min=0.0).amax(dim=1)
        else:
            feasible = torch.ones(args.n_starts, dtype=torch.bool, device=dev)
            max_violation = torch.zeros(args.n_starts, device=dev)
        obj_values = obj_sign * mean[:, obj_idx]
        n_feasible = feasible.sum().item()

        # Rank: surrogate-feasible starts first (by objective, best first),
        # then infeasible starts filling any remaining top-k slots (by how
        # close to feasible, closest first) -- so near-misses get a real
        # VMEC++ check too, not just whatever the surrogate is most sure of.
        feasible_order = torch.argsort(torch.where(feasible, obj_values, torch.full_like(obj_values, float("inf"))))
        infeasible_order = torch.argsort(torch.where(feasible, torch.full_like(max_violation, float("inf")), max_violation))
        # feasible_order's first n_feasible entries are the real (non-inf-padded)
        # ones; infeasible_order's first (n_starts - n_feasible) entries are its
        # real ones, in the same "real entries sort first, inf padding sorts
        # last" pattern -- slice each to just its real portion before concatenating.
        ranked = torch.cat([feasible_order[:n_feasible], infeasible_order[:args.n_starts - n_feasible]])
        top_idx = ranked[:min(args.top_k, args.n_starts)].tolist()

        print(f"\n=== latent design search: {'minimize' if args.minimize else 'maximize'} {obj_name}, "
              f"nfp={args.nfp}, vae={vae_tag}, {n_feasible}/{args.n_starts} starts surrogate-feasible "
              f"(relative_tol={args.relative_tol}) ===")
        print(f"validating top {len(top_idx)} candidates against real VMEC++...")

        x_np = x[top_idx].cpu().numpy().astype(np.float64)
        nfp_val = int(args.nfp)
        candidates = []
        for row in x_np:
            r_cos = row[:45].reshape(5, 9)
            z_sin = row[45:90].reshape(5, 9)
            candidates.append((r_cos, z_sin, nfp_val))

        results = {}  # local rank -> (ok, payload)
        for ok, r_cos, z_sin, nfp, payload in run_batch_with_timeout(candidates, args.n_workers, args.timeout_seconds):
            for local_rank, (cr, cz, cn) in enumerate(candidates):
                if local_rank not in results and cn == nfp and np.array_equal(cr, r_cos) and np.array_equal(cz, z_sin):
                    results[local_rank] = (ok, payload)
                    break

        best_real = None  # (local_rank, measured_y_dict, real_feasible)
        best_real_infeasible_fallback = None
        for local_rank, start_idx in enumerate(top_idx):
            ok, payload = results.get(local_rank, (False, "unknown: no result"))
            print(f"\n  --- candidate {local_rank + 1}/{len(top_idx)} (start #{start_idx}, "
                  f"surrogate {'feasible' if feasible[start_idx] else 'infeasible'}) ---")
            if not ok:
                print(f"  VMEC++: FAILED ({payload})")
                continue
            measured = {name: payload[name] for name in target_names}
            real_c = []
            for idx_c, op, value, _std_c, name, use_abs in constraints:
                mv = measured[name]
                col = abs(mv) if use_abs else mv
                if op == "<=":
                    raw = col - value
                elif op == ">=":
                    raw = value - col
                else:
                    raw = abs(col - value)
                real_c.append(raw / abs(value))
            real_feasible = all(v <= args.relative_tol for v in real_c) if n_c > 0 else True
            print(f"  {obj_name} (objective): predicted {mean[start_idx, obj_idx].item():.6g}  "
                  f"measured {measured[obj_name]:.6g}")
            for k, (idx_c, op, value, _std_c, name, use_abs) in enumerate(constraints):
                label = f"abs({name})" if use_abs else name
                status = "OK" if real_c[k] <= args.relative_tol else f"VIOLATED (rel. viol. {real_c[k]:.4g})"
                print(f"  constraint {label} {op} {value}: predicted {mean[start_idx, idx_c].item():.6g}  "
                      f"measured {measured[name]:.6g}  [{status}]")
            print(f"  really feasible (measured, per paper's definition): {real_feasible}")

            if real_feasible:
                obj_measured_signed = obj_sign * measured[obj_name]
                if best_real is None or obj_measured_signed < best_real[3]:
                    best_real = (local_rank, start_idx, measured, obj_measured_signed)
            else:
                mv_worst = max(real_c) if real_c else 0.0
                if best_real_infeasible_fallback is None or mv_worst < best_real_infeasible_fallback[3]:
                    best_real_infeasible_fallback = (local_rank, start_idx, measured, mv_worst)

        print(f"\n=== result: {n_feasible}/{args.n_starts} surrogate-feasible, "
              f"{sum(1 for v in results.values() if v[0])}/{len(top_idx)} validated candidates converged "
              f"in VMEC++, {'1' if best_real else '0'}/{len(top_idx)} really feasible ===")

        winner = None
        if best_real:
            local_rank, start_idx, measured, _ = best_real
            print(f"BEST REALLY-FEASIBLE candidate: start #{start_idx}, "
                  f"{obj_name}={measured[obj_name]:.6g}")
            winner = (start_idx, measured, True)
        elif best_real_infeasible_fallback:
            local_rank, start_idx, measured, worst = best_real_infeasible_fallback
            print(f"[warn] no really-feasible candidate among the top {len(top_idx)} validated -- "
                  f"reporting the least-infeasible one instead (max relative violation {worst:.4g}), "
                  f"start #{start_idx}")
            winner = (start_idx, measured, False)
        else:
            print("[warn] no candidate converged in VMEC++ at all -- try more --top-k, a different --nfp, "
                  "or loosening constraints")

        if args.score_bounds and winner:
            lower_b, upper_b = args.score_bounds
            start_idx, measured, real_feasible = winner
            score = elongation_style_score(measured[obj_name], lower_b, upper_b, minimize=bool(args.minimize))
            print(f"ConStellaration-style score (measured, bounds {lower_b},{upper_b}): "
                  f"{score if real_feasible else 0.0:.4f}"
                  + ("" if real_feasible else "  (0 by the paper's own rule -- infeasible)"))

        if args.save and winner:
            start_idx, measured, real_feasible = winner
            row = x[start_idx].cpu().tolist()
            design = {
                "r_cos": row[:45], "z_sin": row[45:90],
                "n_field_periods": args.nfp, "is_stellarator_symmetric": 1.0,
                "vae_tag": vae_tag,
                "really_feasible": real_feasible,
                "measured_targets": measured,
                "predicted_targets": {name: mean[start_idx, k].item() for k, name in enumerate(target_names)},
                "predicted_targets_std": {name: std[start_idx, k].item() for k, name in enumerate(target_names)},
            }
            Path(args.save).write_text(json.dumps(design, indent=2))
            print(f"\nsaved{'  (really feasible)' if real_feasible else ' (NOT really feasible -- see warning above)'} "
                  f"to {args.save}")

        if args.progress_log:
            closest_violation = 0.0 if best_real else (
                best_real_infeasible_fallback[3] if best_real_infeasible_fallback else None)
            record = {
                "run_label": args.run_label,
                "vae_tag": vae_tag,
                "member_tags": args.member_tags,
                "objective": obj_name,
                "n_starts": args.n_starts,
                "n_surrogate_feasible": n_feasible,
                "n_validated": len(top_idx),
                "n_vmec_converged": sum(1 for v in results.values() if v[0]),
                "n_really_feasible": 1 if best_real else 0,
                "best_measured_objective": winner[1][obj_name] if winner else None,
                "closest_relative_violation": closest_violation,
            }
            with open(args.progress_log, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"\nprogress logged to {args.progress_log}")


if __name__ == "__main__":
    main()

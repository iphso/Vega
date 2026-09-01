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
from optimize import (  # noqa: E402
    OUT_DIR, load_ensemble, ensemble_predict, parse_constraint, paper_feasibility_violation, violation,
)
from gradient_walk import load_vae  # noqa: E402
from train_vae import nfp_one_hot  # noqa: E402
from train import load_split, IDX_NFP  # noqa: E402
from generate_and_validate import ZERO_COEFF_IDX  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member-tags", nargs="+",
                    default=["reg_mlp_big_full_s0", "reg_mlp_big_full_s1", "reg_mlp_big_full_s2"])
    p.add_argument("--vae-tag", default="vae_coeffs_full_s0")
    p.add_argument("--minimize", default="max_elongation")
    p.add_argument("--maximize", default=None)
    # default=None (not the P1 constraint list) deliberately -- argparse's
    # action="append" APPENDS to a non-empty default rather than replacing
    # it, so any --constraint passed on the CLI used to silently ALSO keep
    # enforcing P1's own constraints underneath it (found the hard way:
    # every P2/P3 attempt against this script was actually solving P1+P2 or
    # P1+P3 jointly, not P2/P3 alone -- see EXPERIMENT_LOG-style history).
    # None here + the explicit fallback below preserves the old "no
    # --constraint given at all -> defaults to P1" convenience without that
    # trap.
    p.add_argument("--constraint", action="append", default=None)
    p.add_argument("--qi-schedule", type=float, nargs="+", default=None,
                    help="Continuation/homotopy over the qi constraint (ConStellaration paper's own "
                         "approach to their hard multi-objective problem: decompose into a SEQUENCE of "
                         "increasingly strict sub-problems -- e.g. their Table 5 sweeps aspect_ratio<=6,8,10,12 "
                         "-- rather than throwing the full constraint set at the optimizer in one shot). "
                         "One of --constraint must name 'qi'; its bound is replaced by each value in this "
                         "list in turn (loosest first), tightening in stages. z (and its VAE-decoded design) "
                         "carries over between stages -- only rho/y/the Adam optimizer reset -- so every "
                         "stage warm-starts from the previous stage's best point instead of re-anchoring to "
                         "a random real design. Every OTHER constraint (mirror, vacuum_well, aspect_ratio, "
                         "etc.) stays enforced at its real target throughout, unlike a free qi-minimization "
                         "probe -- this is meant to fix exactly that failure mode (a candidate that wins on "
                         "qi alone while every other constraint drifts arbitrarily far off).")
    p.add_argument("--sequential-constraints", action="store_true",
                    help="Alternative to plain --qi-schedule continuation: instead of enforcing every "
                         "constraint from stage 1 and only tightening qi's bound, start with ONLY qi "
                         "active (still ramped via --qi-schedule, hardest-first since it's the known "
                         "universal blocker), then add the remaining constraints one at a time, in the "
                         "order given to --constraint, each as its own full stage. Once a constraint "
                         "joins the active set it is never dropped or frozen -- it keeps real ALM "
                         "pressure on it (can still drift a little as later constraints are added) "
                         "rather than being locked exactly, matching plain coordinate-descent-style "
                         "constraint satisfaction rather than a fixed-point lock-in. Requires --qi-schedule.")
    p.add_argument("--trust-gate-multiplier", type=float, default=5.0,
                    help="A candidate's ensemble disagreement (std across --member-tags) is compared "
                         "against the ensemble's OWN typical disagreement on real training data of the "
                         "same nfp (computed once, see --trust-baseline-samples). A candidate is only "
                         "counted surrogate-feasible if its worst-target disagreement ratio is <= this "
                         "multiplier -- i.e. it isn't dramatically more uncertain than the ensemble "
                         "normally is in-distribution. This is a hard gate (excludes from --n-candidates "
                         "entirely), distinct from --trust-weight's soft loss penalty during the search "
                         "itself. Set to a large number (e.g. 1e9) to disable.")
    p.add_argument("--trust-baseline-samples", type=int, default=5000,
                    help="how many real training rows (same nfp) to sample when computing the "
                         "in-distribution ensemble-disagreement baseline for --trust-gate-multiplier")
    p.add_argument("--nfp", type=int, default=3)
    p.add_argument("--n-starts", type=int, default=150, help="multi-start population size")
    p.add_argument("--n-candidates", type=int, default=100, help="max surrogate-feasible candidates to save")
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
    p.add_argument("--save", default=str(OUT_DIR / "generated_candidates_unvalidated.jsonl"))
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

    vae, coeff_mean, coeff_std = load_vae(args.vae_tag)
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

    # Mutable (list, not tuple) per-constraint records -- --qi-schedule rewrites
    # the qi entry's `value` in place between stages while constraint_tilde
    # (a closure over this same list) keeps reading it live, so every other
    # constraint's bound is completely untouched by staging.
    constraints = []
    for spec in args.constraint:
        name, op, value, use_abs = parse_constraint(spec)
        constraints.append([target_names.index(name), op, value, target_stats[name]["std"], name, use_abs])
    n_c = len(constraints)

    qi_constraint_i = None
    if args.qi_schedule:
        qi_matches = [i for i, c in enumerate(constraints) if c[4] == "qi"]
        assert qi_matches, "--qi-schedule requires a '--constraint qi<=...' entry to stage"
        qi_constraint_i = qi_matches[0]
        # Whatever bound --constraint passed for qi was just a placeholder to get an
        # entry into the list -- the schedule's own loosest value is authoritative,
        # including for the guided-seeding pool selection below (real_violation).
        constraints[qi_constraint_i][2] = args.qi_schedule[0]
    assert not args.sequential_constraints or qi_constraint_i is not None, \
        "--sequential-constraints requires --qi-schedule (qi is always tackled first)"

    def constraint_tilde(mean, active_idx=None):
        """active_idx=None (default): every constraint, as always. A subset list is
        --sequential-constraints' mechanism for "this constraint isn't active yet" --
        anything not in active_idx contributes nothing to the ALM loss at all (not
        just a loose bound), matching real coordinate-descent-style constraint
        satisfaction rather than a fixed-point lock-in on already-satisfied ones."""
        idx_list = range(n_c) if active_idx is None else active_idx
        if not idx_list:
            return torch.zeros((mean.shape[0], 0), device=mean.device)
        return torch.stack(
            [paper_feasibility_violation(
                mean[:, constraints[i][0]], constraints[i][1], constraints[i][2], use_abs=constraints[i][5],
                # vacuum_well's threshold is exactly 0.0 -- dividing by abs(value)
                # there is a divide-by-zero that NaNs the whole ALM loop (see
                # optimize.py's paper_feasibility_violation docstring). Matches
                # the official benchmark's own np.maximum(1e-1, bound) workaround.
                divisor=(max(0.1, abs(constraints[i][2])) if constraints[i][4] == "vacuum_well" else None),
                # qi spans 2+ orders of magnitude relative to its threshold for
                # every real design seen so far -- without this, its linear
                # violation is 10-300x every other constraint's, dominating the
                # ALM gradient. Matches p2_report.py/p3_report.py's own qi
                # violation exactly (they've always used log10; this loss hadn't).
                log_transform=(constraints[i][4] == "qi"))
             for i in idx_list],
            dim=1,
        )

    X_train, Y_train = load_split("train")
    same_nfp = X_train[:, IDX_NFP] == float(args.nfp)
    pool = X_train[same_nfp] if same_nfp.any() else X_train
    pool_Y = Y_train[same_nfp] if same_nfp.any() else Y_train

    # In-distribution ensemble-disagreement baseline for --trust-gate-multiplier:
    # "how uncertain is this ensemble normally, on real data it wasn't pushed away
    # from" -- a candidate whose disagreement is way outside that baseline is
    # flagged as having pushed past where the surrogate can be trusted, independent
    # of whether its point-estimate happens to land inside every constraint bound
    # (exactly the failure mode behind the 0/87 real-VMEC-convergence collapse: the
    # ensemble's point estimate agreed the design was feasible, but real geometry
    # wasn't). Sampled, not run over the full pool -- this is a cheap one-off
    # estimate, not something that needs every row.
    with torch.no_grad():
        n_sample = min(args.trust_baseline_samples, pool.shape[0])
        sample_idx = torch.randperm(pool.shape[0])[:n_sample]
        sample_x = pool[sample_idx].to(dev)  # already raw 92-dim [r_cos, z_sin, nfp, symmetry] -- model's native input
        _, sample_std = ensemble_predict(models, sample_x)
        baseline_std = sample_std.mean(dim=0).clamp_min(1e-8)  # (n_targets,)

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
    seed_nfp = torch.full((args.n_starts,), float(args.nfp), device=dev)
    with torch.no_grad():
        z0, _ = vae.encode(seed_coeffs, nfp_one_hot(seed_nfp))
    z = z0.clone().requires_grad_(True)
    nfp_cond = nfp_one_hot(seed_nfp)
    nfp_col = seed_nfp.unsqueeze(1)
    sym_col = torch.ones((args.n_starts, 1), device=dev)

    def decode_x(z):
        return (vae.decode(z, nfp_cond) * coeff_std_t + coeff_mean_t) * zero_mask

    # Build the stage list. Each stage is (label, active_idx, qi_bound_or_None).
    # - No --qi-schedule: single stage, everything active, identical to the old
    #   unstaged behavior.
    # - --qi-schedule alone ("continuation"): every constraint active in every
    #   stage; only qi's own bound ratchets loosest-to-strictest. z carries over
    #   between stages (only rho/y/Adam reset) -- warm-started, not independent
    #   restarts.
    # - --qi-schedule + --sequential-constraints ("coordinate descent"): stage 1
    #   has ONLY qi active, ramped the same way continuation does; once qi's own
    #   ramp finishes, one more constraint joins the active set per subsequent
    #   stage (in --constraint's given order), each getting a full stage of its
    #   own, until all are active. A constraint that joined 2 stages ago is NOT
    #   frozen -- it keeps real ALM pressure the whole time, it's just no longer
    #   the newest addition.
    stages = []
    if qi_constraint_i is None:
        stages.append(("unstaged", list(range(n_c)), None))
    else:
        qi_only = [qi_constraint_i]
        always_active = qi_only if args.sequential_constraints else list(range(n_c))
        for qi_bound in args.qi_schedule:
            stages.append((f"qi<={qi_bound:.4g}", always_active, qi_bound))
        if args.sequential_constraints:
            cumulative = list(qi_only)
            for i in range(n_c):
                if i == qi_constraint_i:
                    continue
                cumulative = cumulative + [i]
                name = constraints[i][4]
                stages.append((f"+{name}", list(cumulative), None))

    print(f"[generate_candidates] {args.n_starts} starts, minimize {obj_name}, nfp={args.nfp}, "
          f"vae={args.vae_tag}, members={args.member_tags}, mode="
          + ("sequential-constraints" if args.sequential_constraints else
             ("continuation" if qi_constraint_i is not None else "unstaged")))

    for stage_i, (label, active_idx, qi_bound) in enumerate(stages):
        if qi_bound is not None:
            constraints[qi_constraint_i][2] = qi_bound
        n_active = len(active_idx)
        print(f"[generate_candidates] stage {stage_i + 1}/{len(stages)} ({label}): "
              f"active constraints = {[constraints[i][4] for i in active_idx]}")

        opt = torch.optim.Adam([z], lr=args.lr)
        rho = torch.full((args.n_starts, n_active), args.alm_rho0, device=dev)
        y = torch.zeros((args.n_starts, n_active), device=dev)
        prev_c = None

        for outer in range(args.alm_outer_iters):
            for _inner in range(args.alm_inner_steps):
                opt.zero_grad()
                x = decode_x(z)
                x_full = torch.cat([x, nfp_col, sym_col], dim=1)
                mean, std = ensemble_predict(models, x_full)

                loss = obj_sign * mean[:, obj_idx] / obj_std
                penalty_scale = (rho.amax(dim=1) / args.alm_rho0) if n_active > 0 else torch.ones(args.n_starts, device=dev)
                if n_active > 0:
                    c = constraint_tilde(mean, active_idx)
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

            if n_active > 0:
                with torch.no_grad():
                    x = decode_x(z)
                    x_full = torch.cat([x, nfp_col, sym_col], dim=1)
                    mean, _ = ensemble_predict(models, x_full)
                    c = constraint_tilde(mean, active_idx)
                    if prev_c is not None:
                        shrunk_enough = c <= args.alm_tau * prev_c
                        rho = torch.where(shrunk_enough, rho, (rho * args.alm_sigma).clamp(max=args.alm_rho_max))
                    y = torch.relu(y + rho * c)
                    prev_c = c

    with torch.no_grad():
        x = decode_x(z)
        x_full = torch.cat([x, nfp_col, sym_col], dim=1)
        mean, std = ensemble_predict(models, x_full)

        c_final = constraint_tilde(mean)  # always the FULL constraint set, regardless of staging mode
        if n_c > 0:
            constraint_ok = (c_final <= args.relative_tol).all(dim=1)
        else:
            constraint_ok = torch.ones(args.n_starts, dtype=torch.bool, device=dev)

        # Trust gate: reject on disagreement even when the point estimate clears
        # every constraint -- see baseline_std above. This is exactly the check
        # that would have caught the half-SIREN staged-P3 run (0/87 converged in
        # real VMEC despite 87/500 "surrogate-feasible") before spending oracle
        # budget confirming it the expensive way.
        disagreement_ratio = std / baseline_std
        max_disagreement_ratio = disagreement_ratio.max(dim=1).values
        trust_ok = max_disagreement_ratio <= args.trust_gate_multiplier

        feasible = constraint_ok & trust_ok
        obj_values = obj_sign * mean[:, obj_idx]
        n_feasible = feasible.sum().item()
        n_constraint_ok_only = (constraint_ok & ~trust_ok).sum().item()

        feasible_idx = torch.nonzero(feasible, as_tuple=True)[0]
        ranked = feasible_idx[torch.argsort(obj_values[feasible_idx])]
        keep = ranked[:min(args.n_candidates, n_feasible)].tolist()

        print(f"\n=== {n_feasible}/{args.n_starts} starts surrogate-feasible "
              f"(relative_tol={args.relative_tol}, trust_gate_multiplier={args.trust_gate_multiplier}) "
              f"-- saving {len(keep)} (NOT VMEC++-validated) ===")
        if n_constraint_ok_only:
            print(f"[trust-gate] {n_constraint_ok_only} more starts cleared every constraint but were "
                  f"REJECTED for disagreeing with itself {args.trust_gate_multiplier}x+ more than the "
                  f"ensemble normally does on real data -- these would otherwise have counted as "
                  f"surrogate-feasible under the old (ungated) rule")
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
                    "vae_tag": args.vae_tag,
                    "validated": False,
                    "predicted_targets": {name: mean[start_idx, k].item() for k, name in enumerate(target_names)},
                    "predicted_targets_std": {name: std[start_idx, k].item() for k, name in enumerate(target_names)},
                    "max_disagreement_ratio": max_disagreement_ratio[start_idx].item(),
                }
                f.write(json.dumps(design) + "\n")

        if keep:
            best_obj = mean[keep[0], obj_idx].item()
            worst_obj = mean[keep[-1], obj_idx].item()
            print(f"predicted {obj_name} range among saved: best {best_obj:.6g}, worst {worst_obj:.6g}")
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()

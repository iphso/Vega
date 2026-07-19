"""Gradient-based design search against the trained surrogate ensemble.

Treats the 90 free Fourier boundary coefficients as the optimization
variable, freezes the trained ensemble (reg_mlp_big_soap_s{0,1,2} by
default), and runs Adam directly on the coefficients (backprop through the
frozen network) to minimize/maximize one target subject to inequality
constraints on the others, handled via a primal-dual augmented-Lagrangian
scheme (a lightweight version of the ConStellaration paper's own ALM-NGOpt
baseline, Algorithm 1: per-constraint penalty rho and multiplier y, updated
in outer iterations around inner Adam steps -- rho only grows when that
constraint's violation isn't shrinking fast enough, rather than one fixed
penalty weight for the whole run). Simplified relative to the paper: no
explicit shrinking trust-region step (delta) since Adam + the distance/trust
penalties below already constrain how far each step can move; the primal
subproblem is solved approximately (a fixed number of inner Adam steps)
rather than to convergence.

Design choices baked in, not exposed as knobs (see EXPERIMENT_LOG for why):
  - n_field_periods is fixed per run (an integer, chosen upfront, per
    --nfp) rather than optimized continuously -- it's a real design
    variable but must physically be a positive integer, and the model only
    ever saw integer values, so letting it float would mean taking
    gradients through fractional-nfp inputs the surrogate never learned.
  - is_stellarator_symmetric is fixed at 1.0 always -- the only value it
    takes anywhere in the training data (std 0), so it's not a free
    variable in any meaningful sense.
  - Every coefficient is clamped after each step to its observed
    [min, max] over the training set (metadata.json feature_stats) -- keeps
    the search inside the region the surrogate was actually trained on
    rather than extrapolating into unconstrained Fourier-coefficient space.
    This also transparently pins the 9 structurally-always-zero coefficients
    (their min == max == 0) without any special-casing.
  - The ensemble's own inter-member disagreement is added as a soft
    trust-region penalty (--trust-weight), each target's disagreement
    normalized by that target's own natural std (target_stats) before
    averaging across targets, since spread is in wildly different raw units
    per target otherwise (see EXPERIMENT_LOG for the noise-floor context).
"""
import argparse
import json
import re
from pathlib import Path

import torch

from train import CKPT_DIR, DualPathMLP, IDX_NFP, OUT_DIR, load_split

FEATURE_NAMES_90 = (
    [f"r_cos_m{m}_n{n}" for m in range(5) for n in range(9)]
    + [f"z_sin_m{m}_n{n}" for m in range(5) for n in range(9)]
)

CONSTRAINT_RE = re.compile(
    r"^(abs\()?([A-Za-z_]\w*)\)?\s*(<=|>=|==)\s*(-?[\d.eE+-]+)$"
)


def load_ensemble(tags, dev):
    models, target_names = [], None
    for tag in tags:
        ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
        assert ckpt["objective"] == "regression", f"{tag} is not a regression checkpoint"
        model = DualPathMLP(
            ckpt["in_dim"], ckpt["n_targets"],
            latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"],
            spatial_latent=ckpt["spatial_latent"], head_hidden=ckpt["head_hidden"],
            priority_weight=ckpt["priority_weight"], use_spatial=ckpt["use_spatial"],
            trunk_arch=ckpt.get("trunk_arch", "mlp"), trunk_blocks=ckpt.get("trunk_blocks", 3),
            use_symlog_latent=ckpt.get("use_symlog_latent", False),
            log_target_mask=ckpt.get("log_target_mask"),
        ).to(dev)
        result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        assert not result.unexpected_keys
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models.append(model)
        target_names = target_names or ckpt["target_names"]
        assert target_names == ckpt["target_names"], "ensemble members must share target order"
    return models, target_names


def ensemble_predict(models, x_full):
    preds = torch.stack([m(x_full) for m in models], dim=0)  # (M, B, T)
    return preds.mean(dim=0), preds.std(dim=0)


def parse_constraint(spec):
    m = CONSTRAINT_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"can't parse constraint {spec!r}, expected e.g. 'qi<=0.02' or "
            f"'abs(edge_rotational_transform_over_n_field_periods)>=0.3'"
        )
    use_abs, name, op, value = m.group(1) is not None, m.group(2), m.group(3), float(m.group(4))
    return name, op, value, use_abs


def violation(mean_col, op, value, std, use_abs=False):
    """Squared-penalty violation used in the optimization loss: unsigned,
    normalized by the target's own dataset std (for balanced gradients across
    differently-scaled targets), zero when satisfied."""
    col = mean_col.abs() if use_abs else mean_col
    if op == "<=":
        raw = torch.relu(col - value)
    elif op == ">=":
        raw = torch.relu(value - col)
    else:
        raw = (col - value).abs()
    return raw / std


def paper_feasibility_violation(mean_col, op, value, use_abs=False):
    """Exact normalized constraint violation formula from the ConStellaration
    benchmark's problems.py (_normalized_constraint_violations): signed,
    normalized by the constraint threshold's own magnitude rather than the
    target's dataset std. A design is feasible there iff this is <= a
    relative tolerance (paper default 1e-2) for every constraint -- used
    here only for final reporting, to make feasibility directly comparable
    to the paper's own definition rather than this script's internal
    (differently-normalized) optimization-loss weighting."""
    col = mean_col.abs() if use_abs else mean_col
    if op == "<=":
        raw = col - value
    elif op == ">=":
        raw = value - col
    else:
        raw = (col - value).abs()
    return raw / abs(value)


def elongation_style_score(value, lower_bound, upper_bound, minimize):
    """Generic version of problems.py's score formulas: linearly rescale
    value into [0, 1] between the given bounds (clipped), inverted if the
    objective is being minimized (matches GeometricalProblem: score = 1 -
    normalize(max_elongation, 1, 10); SimpleToBuildQIStellarator instead
    maximizes, using normalize(...) directly, un-inverted)."""
    normalized = (value - lower_bound) / (upper_bound - lower_bound)
    normalized = max(0.0, min(1.0, normalized))
    return 1.0 - normalized if minimize else normalized


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member-tags", nargs="+",
                    default=["reg_mlp_big_soap_s0", "reg_mlp_big_soap_s1", "reg_mlp_big_soap_s2"])
    p.add_argument("--minimize", default=None, help="target name to minimize")
    p.add_argument("--maximize", default=None, help="target name to maximize (mutually exclusive with --minimize)")
    p.add_argument("--constraint", action="append", default=[],
                    help="e.g. --constraint 'qi<=0.02' --constraint 'vacuum_well>=-0.2'. Repeatable.")
    p.add_argument("--nfp", type=int, required=True, help="fixed n_field_periods for this search")
    p.add_argument("--n-starts", type=int, default=64, help="multi-start population size")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alm-outer-iters", type=int, default=40,
                    help="ALM outer iterations (primal solve -> dual update -> penalty growth), "
                         "matches the paper's geometric-problem setting")
    p.add_argument("--alm-inner-steps", type=int, default=20,
                    help="Adam steps per outer iteration approximately solving the augmented-"
                         "Lagrangian primal subproblem (eq. 8 in the paper)")
    p.add_argument("--alm-rho0", type=float, default=10.0, help="initial penalty parameter, per constraint")
    p.add_argument("--alm-rho-max", type=float, default=1e9, help="cap on the penalty parameter")
    p.add_argument("--alm-tau", type=float, default=0.8,
                    help="penalty only grows if new_violation > tau * old_violation, i.e. shrinking "
                         "slower than this factor per outer iteration")
    p.add_argument("--alm-sigma", type=float, default=5.0, help="penalty growth multiplier when triggered")
    p.add_argument("--trust-weight", type=float, default=0.1,
                    help="penalty on ensemble inter-member disagreement (normalized per-target by "
                         "that target's natural std) -- 0 disables the trust-region term")
    p.add_argument("--distance-weight", type=float, default=1.0,
                    help="penalty on squared distance (per-coefficient std-normalized) to the "
                         "nearest same-nfp training point -- the main defense against the optimizer "
                         "exploiting a corner of coefficient-space the surrogate never saw real data "
                         "in. 0 disables it (not recommended, see optimize.py module docstring).")
    p.add_argument("--relative-tol", type=float, default=1e-2,
                    help="ConStellaration benchmark's own feasibility tolerance: a constraint is "
                         "satisfied iff (violation / |threshold|) <= this (paper default 1e-2). This "
                         "is also what the ALM loop itself targets (same normalization as eq. 2).")
    p.add_argument("--score-bounds", type=float, nargs=2, default=None, metavar=("LOWER", "UPPER"),
                    help="if given, also report a ConStellaration-style [0,1] score for the objective "
                         "(see elongation_style_score) -- e.g. --score-bounds 1.0 10.0 matches "
                         "GeometricalProblem's max_elongation score.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="optional path to save the winning design as JSON")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if bool(args.minimize) == bool(args.maximize):
        raise ValueError("pass exactly one of --minimize or --maximize")

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    feature_stats = meta["feature_stats"]
    target_stats = meta["target_stats"]
    lo = torch.tensor([feature_stats[n]["min"] for n in FEATURE_NAMES_90], device=dev)
    hi = torch.tensor([feature_stats[n]["max"] for n in FEATURE_NAMES_90], device=dev)

    models, target_names = load_ensemble(args.member_tags, dev)
    target_std = torch.tensor([target_stats[n]["std"] for n in target_names], device=dev)

    obj_name = args.minimize or args.maximize
    obj_idx = target_names.index(obj_name)
    obj_sign = 1.0 if args.minimize else -1.0
    obj_std = target_stats[obj_name]["std"]

    constraints = []
    for spec in args.constraint:
        name, op, value, use_abs = parse_constraint(spec)
        constraints.append(
            (target_names.index(name), op, value, target_stats[name]["std"], name, use_abs)
        )

    # Multi-start init: real training rows at the requested nfp, so the
    # search starts from points the surrogate has actual support around,
    # not an arbitrary point in coefficient space. When constraints are
    # given, half the starts are seeded from the real (same-nfp) points
    # already closest to satisfying them (ranked by true target values, not
    # surrogate predictions) rather than uniformly at random -- for a
    # problem whose feasible region is a small fraction of the dataset (the
    # ConStellaration geometric problem's feasible set is ~41 of ~160k
    # points), starting from a uniformly random real point may simply never
    # get close. The other half stays random, for diversity / escaping local
    # optima the guided half might share.
    X_train, Y_train = load_split("train")
    same_nfp = X_train[:, IDX_NFP] == float(args.nfp)
    pool = X_train[same_nfp] if same_nfp.any() else X_train
    pool_Y = Y_train[same_nfp] if same_nfp.any() else Y_train
    if not same_nfp.any():
        print(f"[warn] no training rows with n_field_periods == {args.nfp}; "
              f"seeding from all rows instead (nfp will be overwritten either way)")

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
        print(f"[seed] {n_guided} starts from the {candidate_pool.shape[0]} real same-nfp points "
              f"closest to feasible (min true violation {real_violation[candidate_pool[0]].item():.4g}), "
              f"{n_random} random")
    else:
        idx = torch.randint(0, pool.shape[0], (args.n_starts,))
    x = pool[idx][:, :90].clone().to(dev).requires_grad_(True)
    nfp_col = torch.full((args.n_starts, 1), float(args.nfp), device=dev)
    sym_col = torch.ones((args.n_starts, 1), device=dev)

    # Nearest-real-neighbor distance penalty: per-coefficient marginal
    # clamping (lo/hi above) is NOT enough on its own -- a point can satisfy
    # every individual coefficient's observed range while still sitting in a
    # corner of that box nowhere near the actual (much lower-dimensional,
    # correlated) data manifold, which is exactly where a learned surrogate's
    # predictions become unreliable (confirmed empirically: an early version
    # of this search found a "solution" with a physically impossible
    # negative aspect_ratio). Penalize squared distance, in per-coefficient
    # std-normalized units, to the nearest same-nfp training point --
    # recomputed every step since the candidate moves.
    pool_free = pool[:, :90].to(dev)
    feat_std = torch.tensor([feature_stats[n]["std"] for n in FEATURE_NAMES_90], device=dev).clamp_min(1e-6)
    pool_norm = pool_free / feat_std

    opt = torch.optim.Adam([x], lr=args.lr)

    n_c = len(constraints)
    # Per-start, per-constraint penalty (rho) and Lagrange multiplier (y),
    # both starting at the paper's own geometric-problem values. c-tilde
    # uses the same signed, threshold-relative normalization as the paper's
    # feasibility check (paper_feasibility_violation), not the std-based
    # `violation()` (that's only used above for real-data seed ranking).
    rho = torch.full((args.n_starts, n_c), args.alm_rho0, device=dev)
    y = torch.zeros((args.n_starts, n_c), device=dev)
    prev_c = None

    def constraint_tilde(mean):
        if n_c == 0:
            return torch.zeros((mean.shape[0], 0), device=mean.device)
        return torch.stack(
            [paper_feasibility_violation(mean[:, idx_c], op, value, use_abs=use_abs)
             for idx_c, op, value, _std_c, _name, use_abs in constraints],
            dim=1,
        )

    for outer in range(args.alm_outer_iters):
        for _inner in range(args.alm_inner_steps):
            opt.zero_grad()
            x_full = torch.cat([x, nfp_col, sym_col], dim=1)
            mean, std = ensemble_predict(models, x_full)

            loss = obj_sign * mean[:, obj_idx] / obj_std
            # Scale the trust/distance penalties with how hard each start's own
            # constraints are currently being enforced (rho relative to rho0),
            # not a fixed weight. Otherwise, once rho grows toward alm_rho_max
            # to force constraint satisfaction, a fixed trust/distance weight
            # becomes negligible by comparison and the optimizer is free to
            # "satisfy" the constraints by exploiting an out-of-distribution
            # region the surrogate is simply wrong in (confirmed empirically:
            # this happened, with predicted min_norm_grad going negative again
            # and distance-to-real-data jumping ~50x, before this fix).
            penalty_scale = (rho.amax(dim=1) / args.alm_rho0) if n_c > 0 else torch.ones(args.n_starts, device=dev)
            if n_c > 0:
                c = constraint_tilde(mean)
                inner_term = torch.relu(y + rho * c)
                loss = loss + ((inner_term ** 2 - y ** 2) / (2 * rho)).sum(dim=1)
            if args.distance_weight > 0:
                nn_dist_sq = torch.cdist(x / feat_std, pool_norm).min(dim=1).values ** 2
                loss = loss + args.distance_weight * penalty_scale * nn_dist_sq
            if args.trust_weight > 0:
                trust_pen = (std / target_std).mean(dim=1)
                loss = loss + args.trust_weight * penalty_scale * trust_pen

            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(lo, hi)

        if n_c > 0:
            with torch.no_grad():
                x_full = torch.cat([x, nfp_col, sym_col], dim=1)
                mean, _ = ensemble_predict(models, x_full)
                c = constraint_tilde(mean)
                if prev_c is not None:
                    shrunk_enough = c <= args.alm_tau * prev_c
                    rho = torch.where(shrunk_enough, rho, (rho * args.alm_sigma).clamp(max=args.alm_rho_max))
                y = torch.relu(y + rho * c)
                prev_c = c

    with torch.no_grad():
        x_full = torch.cat([x, nfp_col, sym_col], dim=1)
        mean, std = ensemble_predict(models, x_full)

        # Feasibility and tie-breaking both use the paper's own definitions:
        # is_feasible <=> every c-tilde <= relative_tol; compute_feasibility
        # (used here to rank infeasible candidates) is the max (infinity-norm)
        # over constraints of the normalized violation -- both straight from
        # ConStellaration's problems.py.
        c_final = constraint_tilde(mean)  # (n_starts, n_c)
        if n_c > 0:
            feasible = (c_final <= args.relative_tol).all(dim=1)
            max_violation = c_final.clamp(min=0.0).amax(dim=1)
        else:
            feasible = torch.ones(args.n_starts, dtype=torch.bool, device=dev)
            max_violation = torch.zeros(args.n_starts, device=dev)

        obj_values = obj_sign * mean[:, obj_idx]
        n_feasible = feasible.sum().item()
        if n_feasible > 0:
            candidate_scores = torch.where(feasible, obj_values, torch.full_like(obj_values, float("inf")))
            best = candidate_scores.argmin().item()
        else:
            print(f"[warn] no fully feasible candidate out of {args.n_starts} starts within "
                  f"relative-tol={args.relative_tol}; reporting least-infeasible (smallest max "
                  f"normalized violation) instead")
            best = max_violation.argmin().item()

        nn_dist = torch.cdist(x / feat_std, pool_norm).min(dim=1).values

        print(f"\n=== design search: {'minimize' if args.minimize else 'maximize'} {obj_name}, "
              f"nfp={args.nfp}, {n_feasible}/{args.n_starts} starts feasible "
              f"(paper's relative_tol={args.relative_tol}) ===")
        print(f"best candidate (start #{best}):")
        print(f"  {obj_name} (objective): {mean[best, obj_idx].item():.6g}  "
              f"(ensemble std {std[best, obj_idx].item():.3g})")
        print(f"  distance to nearest same-nfp training point (std-normalized): {nn_dist[best].item():.3g}")
        for k, (idx_c, op, value, _std_c, name, use_abs) in enumerate(constraints):
            pv = c_final[best, k].item()
            status = "OK" if pv <= args.relative_tol else f"VIOLATED (relative viol. {pv:.4g})"
            label = f"abs({name})" if use_abs else name
            print(f"  constraint {label} {op} {value}: predicted {mean[best, idx_c].item():.6g}  [{status}]")
        print(f"  feasible per paper's exact definition: {bool(feasible[best].item())}")
        if args.score_bounds:
            lower_b, upper_b = args.score_bounds
            score = elongation_style_score(mean[best, obj_idx].item(), lower_b, upper_b, minimize=bool(args.minimize))
            print(f"  ConStellaration-style score (bounds {lower_b},{upper_b}): "
                  f"{score:.4f}  (0 if any constraint infeasible per paper definition)")
            if not bool(feasible[best].item()):
                print(f"  -> reported score would be 0.0 by the paper's own scoring rule (infeasible)")
        print("  all predicted targets:")
        for k, name in enumerate(target_names):
            print(f"    {name:55s} {mean[best, k].item():12.5g}  (+/- {std[best, k].item():.3g})")

        if args.save:
            design = {
                "r_cos": x[best, :45].tolist(),
                "z_sin": x[best, 45:90].tolist(),
                "n_field_periods": args.nfp,
                "is_stellarator_symmetric": 1.0,
                "predicted_targets": {name: mean[best, k].item() for k, name in enumerate(target_names)},
                "predicted_targets_std": {name: std[best, k].item() for k, name in enumerate(target_names)},
            }
            Path(args.save).write_text(json.dumps(design, indent=2))
            print(f"\nsaved winning design to {args.save}")


if __name__ == "__main__":
    main()

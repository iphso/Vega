"""GPU half of the grounded-walk round-trip (see grounded_walk_validate.py for
the other half, and run_grounded_walk.sh for the driver loop that alternates
them). Direct response to: "I like the idea that we start at a bunch of
seeds that we know are valid and basically try to walk them toward the
target -- checking every so often to see that we're still on a valid path.
If we legitimately get to a point where all of the candidate fibers cross
out of validity, then we know we need to retrain -- with the idea that the
progressive information up to that point can let us push forward."

v3 variant -- direct response to: "even if something isn't strictly closer
or better there's benefit in having something that has the same distance
from ideal, but pointed in a different direction... having more points like
that means we're doing better." All 30 fibers in the original script
optimize the SAME ALM objective toward the SAME constraint target from
different starting points -- nothing rewards a fiber for reaching a
genuinely different region of the still-failing constraints, so they tend to
climb toward the same ridge rather than fan out. This is quality-diversity /
novelty search (MAP-Elites is the classic instance): adds (1) a small novelty
bonus to each fiber's ALM loss rewarding distance, in a 2D "behavior
descriptor" space of the two constraints actually blocking full feasibility
(edge_magnetic_mirror_ratio, edge_rotational_transform_over_n_field_periods),
from the nearest already-known-good real point (real pool rows passing
>=--archive-min-pass of 5 constraints) -- weighted small so the main
feasibility/objective pressure still dominates, this is a lateral nudge, not
a redirect (the fiber-death/respawn approach in v2 forcibly replaced fibers
and that thrashed badly -- see EXPERIMENT_LOG.md -- so v3 deliberately keeps
fibers eternal and only perturbs the loss), and (2) direct per-round
instrumentation of the archive's actual point-cloud topology -- literally
"how much of the space have we explored" -- logged regardless of whether the
novelty bonus turns out to help, so the hypothesis can be judged on data.

Revision: the original coverage metric here was a coarse 2D axis-aligned grid
over just (mirror_ratio, edge_rotational_transform) -- an arbitrary
projection, no notion of distance or connectivity, and it can't tell "one
blob got denser" from "a second disconnected blob appeared." Replaced with a
k-NN graph over the archive in the FULL 5-constraint space (connected
components -- is the known-good set one blob or several separate pieces --
plus mean/median nearest-neighbor distance as a continuous, bin-free
dispersion measure). Still coarse compared to real persistent homology
(Betti-1 loops specifically would catch a "surrounded but unexplored pocket"
that pure connectivity can't), but it's a real, dimension-honest topology
signal computed from scipy already sitting in the container, not a proxy for
one.

Unlike generate_candidates.py (many inner steps per outer ALM iteration, few
outer iterations, real validation only at the very end -- exactly what let
the earlier staged runs drift 200-360 steps into a shared surrogate blind
spot with zero real-world check along the way), this takes ONE small step
per round and never advances a fiber past a step real VMEC++ didn't confirm:

  each round: propose one step for every fiber -> (next process invocation,
  after grounded_walk_validate.py has real-checked them) accept fibers whose
  step converged, roll the rest back to their last accepted point -> if a
  round's accept rate craters, shrink the shared step size (same adaptive
  mechanism gradient_walk.py already uses for the single-target case) -> if
  EVERY fiber fails in the same round, stop stepping entirely and signal the
  driver to retrain on everything real collected so far before resuming.

Single scorer model, not an ensemble -- the whole point of grounding every
step in real VMEC++ is that we no longer need the ensemble's indirect
disagreement signal to know when to distrust a step; we just ask the oracle.
State lives entirely in files under output/grounded_walk_<tag>/ since this
script is invoked fresh (new container) once per round by the driver.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from optimize import parse_constraint, paper_feasibility_violation  # noqa: E402
from gradient_walk import load_vae  # noqa: E402
from train_vae import nfp_one_hot  # noqa: E402
from train import load_split, IDX_NFP, DualPathMLP  # noqa: E402
from generate_and_validate import ZERO_COEFF_IDX  # noqa: E402

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")


def load_scorer(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"], latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"],
        spatial_latent=ckpt["spatial_latent"], head_hidden=ckpt["head_hidden"],
        priority_weight=ckpt["priority_weight"], use_spatial=ckpt["use_spatial"],
        trunk_arch=ckpt.get("trunk_arch", "mlp"), trunk_blocks=ckpt.get("trunk_blocks", 3),
        use_symlog_latent=ckpt.get("use_symlog_latent", False), log_target_mask=ckpt.get("log_target_mask"),
        objective=ckpt["objective"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)  # strict=False: see train.py's norm buffers note
    model.eval()
    return model, ckpt["target_names"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="run name -- state lives at output/grounded_walk_<tag>/")
    p.add_argument("--scorer-tag", default="reg_half_siren_big_soap_full_s0")
    p.add_argument("--vae-tag", default="vae_coeffs_full_s0")
    p.add_argument("--minimize", default="")
    p.add_argument("--maximize", default="minimum_normalized_magnetic_gradient_scale_length")
    p.add_argument("--constraint", action="append", required=True)
    p.add_argument("--nfp", type=int, default=3)
    p.add_argument("--n-fibers", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01, help="initial shared step size")
    p.add_argument("--step-shrink", type=float, default=0.7)
    p.add_argument("--valid-frac-threshold", type=float, default=0.3,
                    help="shrink step size if this round's accept fraction falls below it")
    p.add_argument("--min-step-size", type=float, default=1e-5)
    p.add_argument("--alm-inner-steps", type=int, default=20, help="gradient steps taken per round, before validating")
    p.add_argument("--alm-rho0", type=float, default=10.0)
    p.add_argument("--alm-rho-max", type=float, default=1e9)
    p.add_argument("--alm-tau", type=float, default=0.8)
    p.add_argument("--alm-sigma", type=float, default=5.0)
    p.add_argument("--latent-weight", type=float, default=0.05)
    p.add_argument("--obj-ramp-start", type=float, default=0.3,
                    help="Per-fiber objective weight ramps from 0 to 1 as that fiber's best-EVER real "
                         "worst_violation goes from this value down to --obj-ramp-end -- find feasibility "
                         "first, optimize the real objective (grad-scale-length) only once actually close. "
                         "Motivated by a real, confirmed failure mode: with the objective always at full "
                         "weight, fibers kept finding real-VMEC-convergent designs (looked 'healthy') while "
                         "steadily drifting AWAY from feasibility over rounds, because grad-scale-length "
                         "maximization is in genuine physical tension with P2/P3's constraint set (see the "
                         "paper's own Table 5 Pareto-front experiment) -- the objective was winning a fight "
                         "it should not have even been in until near the end.")
    p.add_argument("--obj-ramp-end", type=float, default=0.05)
    p.add_argument("--stall-retrain-rounds", type=int, default=15,
                    help="force a retrain (same as the all-fibers-failed trigger) if the population's "
                         "best-ever real worst_violation hasn't improved in this many rounds -- a slow "
                         "plateau on a static surrogate is just as real a signal to refresh it as an "
                         "outright collapse, it just doesn't announce itself as loudly. Set to a huge "
                         "number to disable and rely on the collapse trigger alone (the original behavior).")
    p.add_argument("--relative-tol", type=float, default=1e-2)
    p.add_argument("--novelty-weight", type=float, default=0.05,
                    help="weight on the diversity/novelty bonus (distance, in the 2D mirror_ratio x "
                         "edge_rotational_transform descriptor space, from the nearest already-known-good "
                         "real point) added to each fiber's ALM loss. 0 disables it (falls back to the "
                         "original algorithm's behavior modulo the coverage logging).")
    p.add_argument("--archive-min-pass", type=int, default=3,
                    help="a real pool row counts as 'known-good' (eligible for the novelty archive / "
                         "coverage grid) once it satisfies at least this many of the 5 constraints.")
    p.add_argument("--archive-max-size", type=int, default=4000,
                    help="cap on archive points used for the nearest-neighbor distance each inner step "
                         "(random subsample if the real known-good pool exceeds this) -- keeps the "
                         "cdist cheap regardless of how large the pool grows.")
    p.add_argument("--topology-knn", type=int, default=5,
                    help="k for the k-NN graph built over the archive (full 5-constraint space, std-"
                         "normalized) each round -- used for the connected-components and nearest-"
                         "neighbor-distance instrumentation. Replaces the earlier 2D grid-occupancy "
                         "metric, which couldn't distinguish 'denser' from 'a new disconnected blob'.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if bool(args.minimize) == bool(args.maximize):
        raise ValueError("pass exactly one of --minimize or --maximize")

    state_dir = OUT_DIR / f"grounded_walk_{args.tag}"
    state_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    meta = json.loads((OUT_DIR / "metadata.json").read_text())
    target_stats = meta["target_stats"]

    vae, coeff_mean, coeff_std = load_vae(args.vae_tag)
    vae = vae.to(dev)
    coeff_mean_t = torch.tensor(coeff_mean, dtype=torch.float32, device=dev)
    coeff_std_t = torch.tensor(coeff_std, dtype=torch.float32, device=dev)
    zero_mask = torch.ones(90, device=dev)
    zero_mask[ZERO_COEFF_IDX] = 0.0

    scorer_tag_path = state_dir / "current_scorer_tag.txt"
    scorer_tag = scorer_tag_path.read_text().strip() if scorer_tag_path.exists() else args.scorer_tag
    scorer_tag_path.write_text(scorer_tag)
    model, target_names = load_scorer(scorer_tag, dev)

    obj_name = args.minimize or args.maximize
    obj_idx = target_names.index(obj_name)
    obj_sign = 1.0 if args.minimize else -1.0
    obj_std = target_stats[obj_name]["std"]

    constraints = []
    for spec in args.constraint:
        name, op, value, use_abs = parse_constraint(spec)
        constraints.append((target_names.index(name), op, value, name, use_abs))
    n_c = len(constraints)

    def constraint_tilde(mean):
        if n_c == 0:
            return torch.zeros((mean.shape[0], 0), device=mean.device)
        return torch.stack(
            [paper_feasibility_violation(mean[:, idx_c], op, value, use_abs=use_abs,
                                          divisor=(max(0.1, abs(value)) if name == "vacuum_well" else None),
                                          log_transform=(name == "qi"))
             for idx_c, op, value, name, use_abs in constraints],
            dim=1,
        )

    round_path = state_dir / "round.txt"
    round_num = int(round_path.read_text()) if round_path.exists() else 0
    step_size_path = state_dir / "step_size.txt"
    step_size = float(step_size_path.read_text()) if step_size_path.exists() else args.lr

    z_path = state_dir / "z.pt"
    rho_path, y_path = state_dir / "rho.pt", state_dir / "y.pt"
    pool_x_path, pool_y_path = state_dir / "real_pool_X.npy", state_dir / "real_pool_Y.npy"
    val_results_path = state_dir / "validation_results.jsonl"
    needs_retrain_flag = state_dir / "needs_retrain.flag"
    success_path = state_dir / "success.json"

    # --- Novelty/diversity archive + coverage instrumentation (v3) ---
    # Behavior descriptor = the two constraints actually blocking full P3
    # feasibility on every known good lineage so far (mirror_ratio, edge
    # rotational transform), normalized by each target's own dataset std so
    # neither dimension dominates the distance purely by scale. Built fresh
    # from the real pool every invocation (this script is stateless per
    # container run, same as everything else here) -- cheap, pool sizes here
    # are thousands of rows, not millions.
    mr_name, rt_name = "edge_magnetic_mirror_ratio", "edge_rotational_transform_over_n_field_periods"
    mr_idx_t, rt_idx_t = target_names.index(mr_name), target_names.index(rt_name)
    mr_std, rt_std = target_stats[mr_name]["std"], target_stats[rt_name]["std"]

    def hard_pass_count(Y):
        n = np.zeros(Y.shape[0], dtype=np.int64)
        for idx_c, op, value, name, use_abs in constraints:
            col = np.abs(Y[:, idx_c]) if use_abs else Y[:, idx_c]
            if op == "<=":
                ok = col <= value
            elif op == ">=":
                ok = col >= value
            else:
                ok = np.isclose(col, value)
            n += ok.astype(np.int64)
        return n

    def topology_stats(X, k):
        """k-NN graph over X (rows already std-normalized) -> (n_components,
        largest_component_frac, mean_nn_dist, median_nn_dist). Cheap (archive
        sizes here are thousands of rows, not millions) -- real, dimension-
        honest topology signal instead of an axis-aligned occupancy grid:
        connected components tells us whether the known-good set is one blob
        or several separate pieces, and NN distance is a continuous,
        bin-free stand-in for "how spread out is it," computed over the FULL
        5-constraint space rather than an arbitrary 2D projection of it."""
        n = X.shape[0]
        kk = min(k, n - 1)
        if n < 2:
            return 1, 1.0, 0.0, 0.0
        tree = cKDTree(X)
        d, i = tree.query(X, k=kk + 1)
        nn_dist = d[:, 1:].mean(axis=1)
        rows = np.repeat(np.arange(n), kk)
        cols_ = i[:, 1:].ravel()
        data = d[:, 1:].ravel()
        A = csr_matrix((data, (rows, cols_)), shape=(n, n))
        A = A.maximum(A.T)
        ncomp, labels = connected_components(A, directed=False)
        comp_sizes = np.bincount(labels)
        return int(ncomp), float(comp_sizes.max() / n), float(nn_dist.mean()), float(np.median(nn_dist))

    constraint_idx_t = [idx_c for idx_c, op, value, name, use_abs in constraints]
    constraint_std = np.array([target_stats[name]["std"] for _, _, _, name, _ in constraints])
    constraint_absmask = np.array([use_abs for *_, use_abs in constraints])

    archive_desc = None  # (K, 2) tensor on dev, normalized (mirror_ratio, |edge_rt|) -- still the 2D
                          # descriptor the novelty BONUS itself uses (see below); None if nothing yet.
    archive_n_components, archive_largest_frac = 1, 1.0
    archive_nn_dist_mean, archive_nn_dist_median = 0.0, 0.0
    if pool_y_path.exists() and n_c > 0:
        pool_Y = np.load(pool_y_path)
        if pool_Y.shape[0] > 0:
            good_mask = hard_pass_count(pool_Y) >= args.archive_min_pass
            good_Y = pool_Y[good_mask]
            if good_Y.shape[0] > 0:
                # Topology instrumentation over the FULL 5-constraint space,
                # std-normalized so no one constraint's scale dominates --
                # logged every round regardless of whether the novelty bonus
                # is even on, so the hypothesis can be judged from data.
                full_raw = good_Y[:, constraint_idx_t]
                full_raw = np.where(constraint_absmask, np.abs(full_raw), full_raw)
                full_norm = full_raw / constraint_std
                (archive_n_components, archive_largest_frac,
                 archive_nn_dist_mean, archive_nn_dist_median) = topology_stats(full_norm, args.topology_knn)

                desc_raw = np.stack([good_Y[:, mr_idx_t], np.abs(good_Y[:, rt_idx_t])], axis=1)
                desc_norm = desc_raw / np.array([mr_std, rt_std])
                if desc_norm.shape[0] > args.archive_max_size:
                    sel = np.random.RandomState(0).choice(desc_norm.shape[0], args.archive_max_size, replace=False)
                    desc_norm = desc_norm[sel]
                archive_desc = torch.tensor(desc_norm, dtype=torch.float32, device=dev)

    torch.manual_seed(args.seed)

    if not z_path.exists():
        # First-ever invocation: seed every fiber at a distinct REAL, physically
        # valid design's encoded latent mean (same anchoring gradient_walk.py
        # already uses for the single-target case) -- not a random real row and
        # not the surrogate's own idea of "feasible", an actual dataset member.
        X_train, _ = load_split("train")
        same_nfp = X_train[:, IDX_NFP] == float(args.nfp)
        pool = X_train[same_nfp] if same_nfp.any() else X_train
        anchor_rows = torch.randperm(pool.shape[0])[:args.n_fibers]
        anchor_coeffs = ((pool[anchor_rows][:, :90].to(dev) - coeff_mean_t) / coeff_std_t)
        anchor_nfp = torch.full((args.n_fibers,), float(args.nfp), device=dev)
        with torch.no_grad():
            z0, _ = vae.encode(anchor_coeffs, nfp_one_hot(anchor_nfp))
        torch.save(z0.cpu(), z_path)
        torch.save(torch.full((args.n_fibers, n_c), args.alm_rho0), rho_path)
        torch.save(torch.zeros((args.n_fibers, n_c)), y_path)
        # Sentinel "haven't seen a real measurement yet" value, comfortably above
        # --obj-ramp-start so a fresh fiber starts at objective weight 0 (pure
        # feasibility-seeking) until it actually earns a real data point.
        torch.save(torch.full((args.n_fibers,), 10.0), state_dir / "best_real_violation.pt")
        print(f"[grounded_walk] initialized {args.n_fibers} fibers from real nfp={args.nfp} anchors")

    z = torch.load(z_path, map_location=dev)
    rho = torch.load(rho_path, map_location=dev)
    y = torch.load(y_path, map_location=dev)
    best_real_violation = torch.load(state_dir / "best_real_violation.pt", map_location=dev)
    nfp_t = torch.full((args.n_fibers,), float(args.nfp), device=dev)
    nfp_cond = nfp_one_hot(nfp_t)
    nfp_col = nfp_t.unsqueeze(1)
    sym_col = torch.ones((args.n_fibers, 1), device=dev)

    def decode_x(zz):
        return (vae.decode(zz, nfp_cond) * coeff_std_t + coeff_mean_t) * zero_mask

    # Process the PREVIOUS round's validation results, if any -- accept fibers
    # whose proposed step converged (their proposed z becomes their real z),
    # roll everyone else back to what's already saved in z.pt (their last
    # ACCEPTED point -- we never overwrote it with the unconfirmed proposal).
    if val_results_path.exists():
        results = [json.loads(line) for line in val_results_path.read_text().splitlines()]
        proposed_z = torch.load(state_dir / "proposed_z.pt", map_location=dev)
        n_accepted = 0
        pool_x_new, pool_y_new = [], []
        for r in results:
            i = r["fiber"]
            if r["converged"]:
                n_accepted += 1
                z[i] = proposed_z[i]
                mt = r["measured_targets"]
                x_row = np.array(r["r_cos"] + r["z_sin"] + [args.nfp, 1.0], dtype=np.float32)
                y_row = np.array([mt[n] for n in target_names], dtype=np.float32)
                pool_x_new.append(x_row)
                pool_y_new.append(y_row)
                # Real-measured constraint violation drives this fiber's ALM
                # multiplier update -- more honest than trusting the surrogate's
                # own belief about the point it just got real confirmation on.
                # Goes through paper_feasibility_violation itself now (single
                # source of truth, including its qi log_transform) rather than a
                # separately hand-rolled numpy formula -- the earlier hand-rolled
                # version had exactly the same missing-log10-on-qi bug as
                # constraint_tilde, just duplicated instead of shared.
                y_row_t = torch.tensor([[mt[n] for n in target_names]], dtype=torch.float32, device=dev)
                c_real = torch.cat([
                    paper_feasibility_violation(y_row_t[:, idx_c], op, value, use_abs=use_abs,
                                                 divisor=(max(0.1, abs(value)) if name == "vacuum_well" else None),
                                                 log_transform=(name == "qi"))
                    for idx_c, op, value, name, use_abs in constraints
                ]) if n_c else torch.zeros(0, device=dev)
                y[i] = torch.relu(y[i] + rho[i] * c_real)
                worst = float(c_real.max().item()) if n_c else 0.0
                best_real_violation[i] = min(best_real_violation[i].item(), worst)
                if worst <= args.relative_tol:
                    success_path.write_text(json.dumps({
                        "fiber": i, "round": round_num, "measured_targets": mt, "worst_violation": worst,
                    }, indent=2))
                    print(f"[grounded_walk] *** FIBER {i} REACHED REAL FEASIBILITY at round {round_num} *** "
                          f"worst_violation={worst:+.4f}")
        if pool_x_new:
            _append_pool(pool_x_path, np.stack(pool_x_new))
            _append_pool(pool_y_path, np.stack(pool_y_new))

        frac_valid = n_accepted / args.n_fibers
        print(f"[grounded_walk] round {round_num}: accepted {n_accepted}/{args.n_fibers} ({frac_valid:.0%})")
        record = {"round": round_num, "n_accepted": n_accepted, "n_fibers": args.n_fibers,
                  "frac_valid": frac_valid, "step_size": step_size, "scorer_tag": scorer_tag,
                  "archive_n_components": archive_n_components,
                  "archive_largest_component_frac": archive_largest_frac,
                  "archive_nn_dist_mean": archive_nn_dist_mean,
                  "archive_nn_dist_median": archive_nn_dist_median,
                  "archive_size": 0 if archive_desc is None else int(archive_desc.shape[0])}
        with open(state_dir / "round_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        torch.save(best_real_violation.cpu(), state_dir / "best_real_violation.pt")

        # Stall-triggered retrain: a plateau on a static surrogate is just as
        # real a signal to refresh it as an outright collapse, it just doesn't
        # announce itself as loudly (accept rate can stay perfectly healthy the
        # whole time -- that's exactly what happened here, real data kept
        # accumulating for 40+ rounds with zero improvement in the best-ever
        # point). Added mid-run rather than only at design time, once the data
        # actually showed a plateau rather than the transient dips/recoveries
        # earlier rounds also produced -- initializes from whatever the current
        # global best already is the first time this code runs, so it doesn't
        # spuriously fire on its very first check.
        stall_path = state_dir / "stall_state.json"
        global_best = float(best_real_violation.min().item())
        if stall_path.exists():
            stall = json.loads(stall_path.read_text())
            if global_best < stall["best"] - 1e-6:
                stall = {"best": global_best, "rounds_since_improvement": 0}
            else:
                stall["rounds_since_improvement"] += 1
        else:
            stall = {"best": global_best, "rounds_since_improvement": 0}
        if stall["rounds_since_improvement"] >= args.stall_retrain_rounds:
            print(f"[grounded_walk] no improvement in best-ever real violation "
                  f"({global_best:.4f}) for {stall['rounds_since_improvement']} rounds -- "
                  f"flagging for retrain (same as an accept-rate collapse).")
            # Reset the counter itself, not just the flag -- a real bug in the
            # first version of this code left it incrementing past the
            # threshold forever once first crossed, so every subsequent round
            # re-triggered a retrain immediately regardless of whether the just-
            # refreshed surrogate had had any real chance to improve on it yet
            # (caught directly: "for 15/16/17 rounds" printed on three
            # consecutive rounds around a single retrain). The new model
            # deserves its own full --stall-retrain-rounds window.
            stall["rounds_since_improvement"] = 0
            needs_retrain_flag.touch()
            torch.save(z.cpu(), z_path)
            val_results_path.unlink()
            stall_path.write_text(json.dumps(stall))
            return
        stall_path.write_text(json.dumps(stall))

        if n_accepted == 0:
            print(f"[grounded_walk] EVERY fiber failed this round -- flagging for retrain, "
                  f"not taking a new step. z stays at each fiber's last accepted point.")
            needs_retrain_flag.touch()
            torch.save(z.cpu(), z_path)
            val_results_path.unlink()
            return
        if frac_valid < args.valid_frac_threshold:
            step_size = max(step_size * args.step_shrink, args.min_step_size)
            print(f"[grounded_walk] accept rate below threshold -- shrinking step size to {step_size:.6g}")
        val_results_path.unlink()

    if success_path.exists():
        print("[grounded_walk] success already recorded, nothing further to do")
        return
    if step_size < args.min_step_size:
        print(f"[grounded_walk] step size below floor ({args.min_step_size}), stopping")
        (state_dir / "STOPPED_min_step_size").touch()
        return

    # Per-fiber objective weight: 0 while this fiber's best-ever REAL violation
    # is still above --obj-ramp-start (pure feasibility-seeking, no pressure to
    # also improve the objective yet), ramping linearly to 1 by --obj-ramp-end.
    # A fresh fiber (sentinel 10.0, never yet real-measured) starts at 0. This is
    # the fix for the confirmed failure mode where an always-on objective term
    # fought the constraints and won over many rounds (see grounded_walk's own
    # git history / EXPERIMENT_LOG -- fibers kept finding real designs while
    # steadily drifting away from feasibility).
    obj_weight = ((args.obj_ramp_start - best_real_violation) /
                  (args.obj_ramp_start - args.obj_ramp_end)).clamp(0.0, 1.0)

    # Take this round's step: one round = args.alm_inner_steps gradient updates
    # on the ALM loss from each fiber's currently-accepted z, THEN hand the
    # proposed (not yet accepted) result to grounded_walk_validate.py.
    z = z.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=step_size)
    for _ in range(args.alm_inner_steps):
        opt.zero_grad()
        x = decode_x(z)
        x_full = torch.cat([x, nfp_col, sym_col], dim=1)
        mean = model(x_full)
        loss = obj_weight * obj_sign * mean[:, obj_idx] / obj_std
        if n_c > 0:
            c = constraint_tilde(mean)
            inner_term = torch.relu(y + rho * c)
            loss = loss + ((inner_term ** 2 - y ** 2) / (2 * rho)).sum(dim=1)
        if args.latent_weight > 0:
            loss = loss + args.latent_weight * (z ** 2).mean(dim=1)
        if archive_desc is not None and args.novelty_weight > 0:
            # Reward distance from the nearest already-known-good real point
            # in (mirror_ratio, |edge_rotational_transform|) space -- a small
            # lateral nudge to fan fibers out across the still-failing
            # constraints instead of all converging on the same solution,
            # NOT a redirect away from feasibility (the ALM term above still
            # dominates; this is deliberately weighted small).
            desc = torch.stack([mean[:, mr_idx_t], mean[:, rt_idx_t].abs()], dim=1)
            desc = desc / torch.tensor([mr_std, rt_std], device=dev)
            novelty_dist = torch.cdist(desc, archive_desc).min(dim=1).values
            loss = loss - args.novelty_weight * novelty_dist
        loss.sum().backward()
        opt.step()

    prev_c_path = state_dir / "prev_c.pt"
    with torch.no_grad():
        x = decode_x(z)
        if n_c > 0:
            x_full = torch.cat([x, nfp_col, sym_col], dim=1)
            mean = model(x_full)
            c = constraint_tilde(mean)
            # Same conditional ALM growth rule as generate_candidates.py -- only
            # grow rho on a constraint that ISN'T shrinking fast enough, not
            # unconditionally every round. An earlier version of this script grew
            # rho by alm_sigma every single round regardless of progress, which
            # would have made the penalty (and gradient magnitude on any still-
            # violated constraint) balloon by 5^round -- a real bug, caught before
            # it could confound results, not the actual cause of the first real
            # run's 0% accept rate (that was grounded_walk_validate.py's missing
            # r_cos/z_sin reshape, a separate and unrelated bug).
            prev_c = torch.load(prev_c_path) if prev_c_path.exists() else None
            if prev_c is not None:
                shrunk_enough = c <= args.alm_tau * prev_c
                rho = torch.where(shrunk_enough, rho, (rho * args.alm_sigma).clamp(max=args.alm_rho_max))
            torch.save(c, prev_c_path)
            # Standard ALM dual update, for EVERY fiber, using the surrogate's
            # estimate on this round's proposal -- regardless of whether that
            # proposal ends up accepted once validated. A real bug in the first
            # working version of this script updated y ONLY for fibers whose step
            # got accepted (see the real-measured update below), leaving a
            # rejected fiber's multipliers frozen -- meaning a fiber that kept
            # finding real-but-off-target designs got progressively WEAKER
            # constraint pressure over time instead of stronger, and the observed
            # result was exactly that: real acceptance rate stayed healthy while
            # worst_violation drifted steadily worse round over round (0.47 -> 0.65
            # over the first ~6 rounds) instead of improving. This restores the
            # corrective pressure generate_candidates.py's own ALM loop always had.
            y = torch.relu(y + rho * c)

    torch.save(z.detach().cpu(), state_dir / "proposed_z.pt")
    torch.save(rho.cpu(), rho_path)
    torch.save(y.cpu(), y_path)
    round_path.write_text(str(round_num + 1))
    step_size_path.write_text(str(step_size))

    with open(state_dir / "proposed_candidates.jsonl", "w") as f:
        with torch.no_grad():
            rows = x.cpu().tolist()
        for i, row in enumerate(rows):
            f.write(json.dumps({"fiber": i, "r_cos": row[:45], "z_sin": row[45:90],
                                 "n_field_periods": args.nfp}) + "\n")
    n_engaged = int((obj_weight > 0).sum().item())
    archive_n = 0 if archive_desc is None else int(archive_desc.shape[0])
    print(f"[grounded_walk] round {round_num} step proposed, {args.n_fibers} candidates ready to validate "
          f"({n_engaged}/{args.n_fibers} fibers close enough to feasible to have any objective pressure, "
          f"best_real_violation min={best_real_violation.min().item():.4f}, "
          f"novelty archive={archive_n} points, topology={archive_n_components} component(s) "
          f"({archive_largest_frac:.0%} in largest), nn_dist mean/median="
          f"{archive_nn_dist_mean:.4f}/{archive_nn_dist_median:.4f})")


def _append_pool(path, arr):
    full = np.concatenate([np.load(path), arr]) if path.exists() else arr
    np.save(path, full)


if __name__ == "__main__":
    main()

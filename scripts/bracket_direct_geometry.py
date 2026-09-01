"""Generate + score + DIRECT new bracket geometry -- the closing piece of
the domain's A/B/C/D bar (oracle=bracket_fem_warp.py's real warp.fem solve,
dataset=bracket_generate_dataset_warp.py, scoring=train_bracket_scoring.py,
generative=train_bracket_gan.py). Direct request: "I want to like...
generate geometry and score it and be able to like... direct new geometry
- as close to parity as possible with the other problems."

Two modes, both for a single fixed force layout (see EXPERIMENT_LOG for why
cross-layout conditioning isn't built yet):

  --mode sample: unconditioned generation. Sample z ~ N(0,1) at a chosen
  target compliance (the GAN's conditioning scalar), decode to a density
  field, and score EVERY candidate through the REAL oracle (the actual
  warp.fem forward solve) -- not just the GAN's own implicit sense of
  "plausible." Reports predicted-vs-real compliance so GAN miscalibration
  is visible, not assumed away, same discipline as vmec_jax candidates
  being checked against real VMEC++ every round.

  --mode steer: gradient-guided latent walk (same idea as gradient_walk.py,
  simplified since this domain's real oracle is cheap enough to call every
  round instead of needing a surrogate-only inner loop) -- backprop the
  trained SCORING surrogate's prediction through the GAN generator back to
  z, walk downhill toward lower predicted compliance, and validate every
  step against the real oracle. A step that makes the REAL compliance worse
  (surrogate/GAN disagreeing with real physics) is rejected and the walk
  rolls back -- exactly the "never trust the surrogate alone" pattern used
  everywhere else in this project.

Every accepted/generated candidate that beats a threshold is exported as a
real mesh (bracket_export_mesh_warp.export_mesh) so it can be inspected in
the viewer next to the SIMP-optimized cases, not just reported as numbers.
"""
import argparse
import json

import numpy as np
import torch
import warp as wp

import bracket_postprocess
from bracket_export_mesh_warp import export_mesh
from bracket_generate_dataset_warp import LAYOUTS, build_layout_context, clean_shape, score_shape
from bracket_simp_warp import von_mises_functional
import warp.fem as fem
from train_bracket_scoring import ScoringMLP
from train_gan import Generator

OUT_DIR = "/work/output"
CKPT_DIR = "/work/checkpoints"
RHO_MIN = 1e-3


def load_generator(layout, seed=0):
    ckpt = torch.load(f"{CKPT_DIR}/bracket_gan_{layout}_s{seed}.pt", map_location="cpu", weights_only=False)
    gen = Generator(coeff_dim=ckpt["n_cells"], n_targets=1, latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"], n_nfp=0)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    return gen, ckpt


def load_scoring(layout, seed=0):
    ckpt = torch.load(f"{CKPT_DIR}/bracket_scoring_{layout}_s{seed}.pt", map_location="cpu", weights_only=False)
    model = ScoringMLP(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"], n_blocks=ckpt["n_blocks"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def decode_to_rho(gen, gan_ckpt, z, cond_z):
    """GAN output -> a real, clamped density field (physically usable, not
    just 'whatever the generator happened to emit') -- the generator was
    never architecturally bounded to [rho_min, 1] (Generator is reused
    UNCHANGED from train_gan.py, same convention as train_mug_gan.py), so
    clamping here is the domain-specific fix-up, applied the same way
    mug's softmax fix-up is applied AFTER the unchanged Generator, not by
    modifying it."""
    raw = gen(z, cond_z)
    rho = raw * torch.tensor(gan_ckpt["rho_std"]) + torch.tensor(gan_ckpt["rho_mean"])
    return rho.clamp(RHO_MIN, 1.0)


def predict_compliance(scoring_model, scoring_ckpt, rho):
    x = (rho - torch.tensor(scoring_ckpt["x_mean"])) / torch.tensor(scoring_ckpt["x_std"])
    pred_log_z = scoring_model(x)
    pred_log = pred_log_z * scoring_ckpt["y_std"] + scoring_ckpt["y_mean"]
    return pred_log  # still log-space


def load_ref_meta(layout):
    """Never let a missing/stale reference file crash a batch AFTER the
    expensive real-oracle scoring has already run (§105: an earlier
    version hard-crashed here, losing 10,000 already-scored candidates
    that were never exported because the crash happened before the
    export loop). Returns None if unavailable -- callers degrade the
    SIMP-ratio comparison gracefully rather than losing everything."""
    try:
        return json.load(open(f"{OUT_DIR}/dataset_{layout}_ref_meta.json"))
    except FileNotFoundError:
        return None


def register_in_index(entry):
    """Append one exported candidate's summary to a persistent index file
    the viewer reads (bracket_generated_index.json) -- read-modify-write
    since each --mode invocation is a separate process/container, same
    pattern bracket_generate_candidates_warp.py's index uses, but appended
    to across multiple runs instead of written once at the end of a single
    batch."""
    path = f"{OUT_DIR}/bracket_generated_index.json"
    try:
        index = json.load(open(path))
    except FileNotFoundError:
        index = []
    index = [e for e in index if e["tag"] != entry["tag"]]  # replace if re-exported
    index.append(entry)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def score_cleaned(ctx, params, rho_np):
    """Cleans disconnected islands, re-scores through the real oracle, and
    computes real physical-unit metrics -- the one function every export
    path (sample/steer/bulk) funnels through so cleanup is never skipped
    for a candidate that ends up in the viewer."""
    rho_clean, removed_fraction, spans_load_path = clean_shape(ctx, rho_np)
    real_compliance, u = score_shape(ctx, rho_clean, return_u=True)

    von_mises_arr = wp.zeros(shape=ctx["n_cells"], dtype=float)
    rho_field = ctx["rho_space"].make_field()
    rho_field.dof_values = wp.array(rho_clean.astype(np.float32), dtype=float)
    u_field = ctx["u_space"].make_field()
    u_field.dof_values = u
    from bracket_generate_dataset_warp import PENAL, E_MIN
    fem.interpolate(von_mises_functional, at=fem.Cells(geometry=ctx["u_space"].geometry),
                     fields={"u": u_field, "rho": rho_field}, values={"penal": PENAL, "e_min": E_MIN, "out": von_mises_arr})
    von_mises_np = von_mises_arr.numpy()

    domain_volume_hat = float(np.prod(params["bounds_hi"]))  # rectangular domain -- exact, no need to re-integrate
    material_fraction = float((rho_clean >= 0.5).mean())
    metrics = bracket_postprocess.real_metrics(params["load_vec"], float(von_mises_np.max()), material_fraction, domain_volume_hat)
    if not spans_load_path:
        # No material connects both mounts to the load at all -- this
        # "candidate" can't take any real load, whatever its stress
        # reading says (SIMP's/the GAN's e_min-baseline math still solves
        # fine even when nothing structurally connects -- caught for real,
        # §104). Zero the strength numbers so it can never win a
        # strength-based ranking; compliance/mass stay real for
        # transparency, but max_load speaks the actual truth.
        metrics["max_load_kgf"] = 0.0
        metrics["max_load_N"] = 0.0
        metrics["safety_factor"] = 0.0
    return dict(rho=rho_clean, u=u, compliance=real_compliance, von_mises=von_mises_np,
                removed_fraction=removed_fraction, spans_load_path=spans_load_path, metrics=metrics)


def export_candidate(ctx, layout, params, tag, scored, mode, predicted_compliance=None):
    """Writes the files bracket_export_mesh_warp.export_mesh(tag) expects,
    matching bracket_simp_warp.py's own meta.json schema so the SAME export
    path (and the SAME viewer code) that renders SIMP-optimized cases also
    renders GAN-generated ones -- no viewer-side special-casing. `scored`
    is a score_cleaned() result -- island-cleaned, re-scored, real-unit
    metrics already attached."""
    rho_np, von_mises_np, real_compliance = scored["rho"], scored["von_mises"], scored["compliance"]

    np.save(f"{OUT_DIR}/{tag}_rho.npy", rho_np.astype(np.float32))
    np.save(f"{OUT_DIR}/{tag}_cell_centers.npy", ctx["cell_centers"].astype(np.float32))
    np.save(f"{OUT_DIR}/{tag}_von_mises.npy", von_mises_np.astype(np.float32))
    meta = {
        "mount_a_x": params["mount_a_x"], "mount_b_x": params["mount_b_x"], "mount_radius": params["mount_radius"],
        "load_x": params["load_x"], "load_radius": params["load_radius"], "load_vec": params["load_vec"],
        "bounds_lo": [0.0, 0.0, 0.0], "bounds_hi": params["bounds_hi"], "res": params["res"],
        "volfrac": float(rho_np.mean()), "n_iters": 0,
        "final_compliance": real_compliance, "final_mean_rho": float(rho_np.mean()),
        "islands_removed_fraction": scored["removed_fraction"], "spans_load_path": scored["spans_load_path"],
        "compliance_history": [real_compliance],  # no iterative trajectory -- direct one-shot generation
        "naive_baseline_compliance": None, "naive_vs_simp_ratio": None,
        "von_mises_min": float(von_mises_np.min()), "von_mises_max": float(von_mises_np.max()),
        "von_mises_mean": float(von_mises_np.mean()),
        "source": "gan_generated",
        **scored["metrics"],
    }
    with open(f"{OUT_DIR}/{tag}_meta.json", "w") as f:
        json.dump(meta, f)
    mesh_ok = True
    try:
        export_mesh(tag)
    except ValueError as e:
        # A poorly-trained/low-quality generated candidate can have NO
        # cell above the rho>=0.5 marching-cubes threshold anywhere (all
        # material below threshold) -- a real possible outcome for a
        # generator this early, not a bug to hide. Report it honestly
        # instead of crashing the whole batch over one bad candidate.
        mesh_ok = False
        print(f"  [{tag}] mesh export skipped -- no surface at threshold 0.5 "
              f"(max rho={rho_np.max():.4f}): {e}")

    ref_meta = load_ref_meta(layout)
    register_in_index({
        "tag": tag, "layout": layout, "mode": mode, "mesh_ok": mesh_ok,
        "real_compliance": real_compliance, "predicted_compliance": predicted_compliance,
        "volfrac": float(rho_np.mean()), "islands_removed_fraction": scored["removed_fraction"],
        "simp_reference_compliance": ref_meta["final_compliance"] if ref_meta else None,
        "ratio_vs_simp": (real_compliance / ref_meta["final_compliance"]) if ref_meta else None,
        **scored["metrics"],
    })
    return meta


RANK_KEYS = {
    # (higher_is_better, extractor) -- extractor reads a score_cleaned() result
    "strength_to_weight": (True, lambda s: s["metrics"]["max_load_kgf"] / max(s["metrics"]["mass_kg"], 1e-9)),
    "max_load": (True, lambda s: s["metrics"]["max_load_kgf"]),
    "lightest": (False, lambda s: s["metrics"]["mass_kg"]),
    "compliance": (False, lambda s: s["compliance"]),
}


def cmd_sample(args, ctx, params, gen, gan_ckpt, scoring_model, scoring_ckpt):
    """Direct follow-up: "shouldn't we have like... tens of thousands?" --
    samples a large batch from the GAN, cleans + real-oracle-scores EVERY
    one (not just a lucky few), and ranks by an accessible physical metric
    (default: strength-to-weight = max_load_kgf/mass_kg) instead of the
    raw SIMP compliance number. Only the top `--n-export` actually get a
    full mesh export (tens of thousands of mesh files isn't a browsable
    viewer pool) -- everything else is still scored, just not saved as a
    viewer candidate."""
    target_log = np.log(args.target_compliance) if args.target_compliance else gan_ckpt["compliance_log_mean"]
    cond_z_val = (target_log - gan_ckpt["compliance_log_mean"]) / gan_ckpt["compliance_log_std"]
    higher_better, rank_fn = RANK_KEYS[args.rank_by]
    print(f"[sample] n={args.n_candidates}  rank_by={args.rank_by}  target_compliance={args.target_compliance}  cond_z={cond_z_val:.3f}")

    torch.manual_seed(args.seed)
    batch_size = 256
    results = []
    n_done = 0
    while n_done < args.n_candidates:
        n = min(batch_size, args.n_candidates - n_done)
        z = torch.randn(n, gan_ckpt["latent_dim"])
        cond_z = torch.full((n, 1), float(cond_z_val))
        with torch.no_grad():
            rho_batch = decode_to_rho(gen, gan_ckpt, z, cond_z)
            pred_log = predict_compliance(scoring_model, scoring_ckpt, rho_batch)
        for i in range(n):
            rho_np = rho_batch[i].numpy()  # dof-ordered -- the GAN was trained directly on dof-ordered rho rows
            scored = score_cleaned(ctx, params, rho_np)
            scored["predicted_compliance"] = float(np.exp(pred_log[i].item()))
            results.append(scored)
        n_done += n
        if n_done % (batch_size * 4) == 0 or n_done == args.n_candidates:
            print(f"  scored {n_done}/{args.n_candidates}")

    n_total = len(results)
    functional = [r for r in results if r["spans_load_path"]]
    n_dropped = n_total - len(functional)
    print(f"[sample] {n_dropped}/{n_total} candidates dropped from ranking -- no material connects both "
          f"mounts to the load at all (not real brackets, whatever their raw compliance says)")
    results = functional
    if not results:
        print("[sample] every candidate failed the connectivity check -- nothing to rank or export. "
              "This layout's GAN likely needs more training epochs or more directed data.")
        return

    results.sort(key=rank_fn, reverse=higher_better)
    ref_meta = load_ref_meta(args.layout)
    best = results[0]
    ref_str = f"(SIMP ref: {ref_meta['final_compliance']:.4f})" if ref_meta else "(SIMP ref: unavailable)"
    print(f"\n[sample] {len(results)} candidates scored. Best by {args.rank_by}: "
          f"compliance={best['compliance']:.4f} {ref_str}  "
          f"mass={best['metrics']['mass_kg']*1000:.0f}g  max_load={best['metrics']['max_load_kgf']:.0f}kgf  "
          f"islands_removed={best['removed_fraction']:.4f}")

    exported = []
    for rank, scored in enumerate(results[:args.n_export]):
        tag = f"bracket_gan_{args.layout}_sample_{rank:02d}"
        export_candidate(ctx, args.layout, params, tag, scored, mode="sample",
                          predicted_compliance=scored["predicted_compliance"])
        exported.append(tag)
    print(f"[sample] exported top {len(exported)} (by {args.rank_by}): {exported}")

    mean_pred = np.mean([r["predicted_compliance"] for r in results])
    mean_real = np.mean([r["compliance"] for r in results])
    calib_ratio = mean_pred / mean_real
    print(f"[sample] calibration check: mean predicted/real compliance ratio = {calib_ratio:.3f} "
          f"({'well-calibrated' if 0.7 < calib_ratio < 1.4 else 'surrogate is notably miscalibrated -- trust the real numbers above, not the predicted ones'})")


def cmd_steer(args, ctx, params, gen, gan_ckpt, scoring_model, scoring_ckpt):
    """Steers toward lower compliance AT (approximately) the layout's own
    SIMP volume-fraction target -- caught the hard way (§102): an earlier
    version of this loop had no volume penalty at all, so it "beat" the
    SIMP reference on the v2 case (0.80x) purely by using 89% material vs.
    SIMP's 25% target -- the exact same "more material always helps
    compliance" trap the naive-baseline check exists to catch (§100), not
    a real win. Now the differentiable loss includes a volume penalty
    (mirrors bracket_simp_warp.py's own vol_penalty_weight term), AND the
    "best" candidate actually reported/exported must be within
    `--vol-tolerance` of the target volfrac -- an off-budget candidate with
    lower raw compliance is tracked separately for visibility but never
    presented as beating anything."""
    target_volfrac = params["volfrac"]
    vol_tol = args.vol_tolerance * target_volfrac

    torch.manual_seed(args.seed)
    z = torch.randn(args.n_walks, gan_ckpt["latent_dim"], requires_grad=True)
    cond_z = torch.zeros(args.n_walks, 1)  # unconditioned starting point; the walk itself finds "lower"

    step_size = args.step_size
    best_feasible = None      # lowest real compliance WITHIN volume tolerance -- the only one that's a fair comparison
    best_overall = None       # lowest real compliance seen at all, regardless of volfrac -- diagnostic only

    def in_tol(volfrac):
        return abs(volfrac - target_volfrac) <= vol_tol

    for round_idx in range(1, args.n_steps + 1):
        rho_batch = decode_to_rho(gen, gan_ckpt, z, cond_z)
        pred_log = predict_compliance(scoring_model, scoring_ckpt, rho_batch)
        vol_penalty = args.vol_penalty_weight * ((rho_batch.mean(dim=1) - target_volfrac) ** 2)
        loss = (pred_log + vol_penalty).sum()
        grad, = torch.autograd.grad(loss, z)

        with torch.no_grad():
            grad_norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z_proposed = z - step_size * grad / grad_norm
            rho_proposed = decode_to_rho(gen, gan_ckpt, z_proposed, cond_z)
            rho_prev = decode_to_rho(gen, gan_ckpt, z, cond_z)

        n_improved = 0
        real_compliances = []
        for i in range(args.n_walks):
            rho_np = rho_proposed[i].numpy()
            real_c, u = score_shape(ctx, rho_np, return_u=True)
            volfrac = float(rho_np.mean())
            real_compliances.append(real_c)
            if best_overall is None or real_c < best_overall[0]:
                best_overall = (real_c, rho_np.copy(), u, volfrac)
            if in_tol(volfrac) and (best_feasible is None or real_c < best_feasible[0]):
                best_feasible = (real_c, rho_np.copy(), u, volfrac)

            # Accept the step only if it improves the SAME penalized
            # objective the gradient is descending (real compliance + real
            # volume penalty) -- the surrogate/GAN's gradient is a
            # proposal, the real oracle is still the sole acceptance test
            # (gradient_walk.py's own rule).
            prev_rho_np = rho_prev[i].numpy()
            prev_c, _ = score_shape(ctx, prev_rho_np, return_u=True)
            prev_volfrac = float(prev_rho_np.mean())
            score_new = real_c + args.vol_penalty_weight * (volfrac - target_volfrac) ** 2
            score_prev = prev_c + args.vol_penalty_weight * (prev_volfrac - target_volfrac) ** 2
            if score_new < score_prev:
                with torch.no_grad():
                    z[i] = z_proposed[i]
                n_improved += 1

        print(f"round {round_idx:3d}/{args.n_steps}: {n_improved}/{args.n_walks} walks improved  "
              f"step_size={step_size:.4f}  "
              f"best_feasible={'n/a' if best_feasible is None else f'{best_feasible[0]:.4f} (volfrac {best_feasible[3]:.3f})'}  "
              f"mean_real_compliance={np.mean(real_compliances):.4f}")
        if n_improved / args.n_walks < 0.3:
            step_size *= 0.6

    ref_meta = load_ref_meta(args.layout)
    ref_str = f"SIMP reference compliance={ref_meta['final_compliance']:.4f} at volfrac={ref_meta['final_mean_rho']:.3f}" if ref_meta else "SIMP reference unavailable"
    print(f"\n[steer] target volfrac={target_volfrac:.3f} (+/-{vol_tol:.3f})  {ref_str}")
    if best_overall is not None:
        print(f"[steer] best compliance seen at ANY volfrac (diagnostic only, NOT a fair comparison): "
              f"{best_overall[0]:.4f} at volfrac={best_overall[3]:.3f}")
    if best_feasible is None:
        print(f"[steer] no candidate ever landed within the volume tolerance -- nothing fair to export. "
              f"Try more --n-steps or a larger --vol-penalty-weight.")
        return
    ratio_str = f"ratio vs SIMP reference={best_feasible[0] / ref_meta['final_compliance']:.2f}x" if ref_meta else "ratio vs SIMP: n/a"
    print(f"[steer] best compliance WITHIN volume tolerance (the fair number): {best_feasible[0]:.4f} "
          f"at volfrac={best_feasible[3]:.3f}  {ratio_str}")
    tag = f"bracket_gan_{args.layout}_steered"
    # Clean + re-score before export -- the exploration loop above uses the
    # raw (uncleaned) shape for speed (islands rarely change WHICH walk is
    # best), but the actual exported/viewer candidate gets the same
    # connectivity cleanup + real-unit metrics every other export path uses.
    scored = score_cleaned(ctx, params, best_feasible[1])
    if scored["removed_fraction"] > 0:
        print(f"[steer] connectivity cleanup on final candidate removed {scored['removed_fraction']:.4f} of the domain")
    export_candidate(ctx, args.layout, params, tag, scored, mode="steer")
    print(f"[steer] exported best FEASIBLE candidate as '{tag}': mass={scored['metrics']['mass_kg']*1000:.0f}g  "
          f"max_load={scored['metrics']['max_load_kgf']:.0f}kgf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True, choices=list(LAYOUTS.keys()))
    ap.add_argument("--mode", required=True, choices=["sample", "steer"])
    ap.add_argument("--seed", type=int, default=0)
    # sample mode
    ap.add_argument("--n-candidates", type=int, default=64)
    ap.add_argument("--n-export", type=int, default=3)
    ap.add_argument("--target-compliance", type=float, default=None)
    ap.add_argument("--rank-by", default="strength_to_weight", choices=list(RANK_KEYS.keys()))
    # steer mode
    ap.add_argument("--n-walks", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--step-size", type=float, default=0.3)
    ap.add_argument("--vol-penalty-weight", type=float, default=30.0,
                     help="pushes generated shapes toward the layout's own SIMP volfrac target -- "
                          "without this, lower compliance is trivially achieved by just using more material (§102)")
    ap.add_argument("--vol-tolerance", type=float, default=0.15,
                     help="fraction of target volfrac allowed before a candidate is excluded from "
                          "the 'best feasible' comparison against SIMP")
    args = ap.parse_args()

    params = LAYOUTS[args.layout]
    ctx = build_layout_context(params)
    gen, gan_ckpt = load_generator(args.layout, args.seed)
    scoring_model, scoring_ckpt = load_scoring(args.layout, args.seed)

    if args.mode == "sample":
        cmd_sample(args, ctx, params, gen, gan_ckpt, scoring_model, scoring_ckpt)
    else:
        cmd_steer(args, ctx, params, gen, gan_ckpt, scoring_model, scoring_ckpt)


if __name__ == "__main__":
    main()

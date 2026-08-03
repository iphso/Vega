"""Unattended multi-day orchestrator: alternates growing the bootstrap pool,
retraining the production surrogate ensemble on it, and checkpointing the
design search against the fresh ensemble+VAE -- so a redirected bootstrap
target (see EXPERIMENT_LOG) actually gets acted on, not just accumulated.

Each block:
  1. `bootstrap_loop.py --generations N` -- N more self-training generations,
     directed sampling biased toward --directed-target/--directed-direction
     (default: maximize edge_rotational_transform_over_n_field_periods, the
     target the design search's own real-VMEC++ check identified as the
     binding failure mode). Auto-resumes from wherever the pool left off.
  2. `augment_split.py` -- rebuilds a train split = cluster-split base +
     every train-cluster-anchored design in the (now bigger) pool.
  3. `train.py` x3 seeds -- retrains the actual production ensemble
     (reg_mlp_big_soap's architecture) on that split, tagged per-block so
     history isn't overwritten. This is the step that makes the redirected
     bootstrap sampling actually pay off -- more data alone doesn't help
     until the model that gets searched against is refit on it.
  4. `latent_optimize.py` against the fresh ensemble + fresh VAE, on the
     same ConStellaration geometric problem checked earlier this session --
     appends one line to --progress-log so the whole run's trend is a single
     `tail`/`jq` away rather than requiring a re-read of every block's full
     output.

Stops after --budget-hours or --max-blocks, whichever comes first. Each
subprocess call is allowed to fail without killing the run -- a warning is
printed and the loop moves to the next phase/block, since the highest-value
asset (the growing, VMEC++-validated pool, via bootstrap_loop.py's own
auto-resume/incremental-checkpoint machinery) survives a single bad step
either way.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path("/work/output")


def run(cmd, label):
    print(f"[long_run] $ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"[long_run] WARNING: {label} exited {result.returncode} after {dt:.0f}s -- continuing anyway", flush=True)
    else:
        print(f"[long_run] {label} done in {dt:.0f}s", flush=True)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget-hours", type=float, default=120.0, help="wall-clock budget, default 5 days")
    p.add_argument("--max-blocks", type=int, default=20, help="hard cap regardless of time budget")
    p.add_argument("--gens-per-block", type=int, default=6, help="~12-13h at this project's observed ~2.1-2.3h/generation")
    p.add_argument("--bootstrap-tag", default="bootstrap0")
    p.add_argument("--directed-target", default="edge_rotational_transform_over_n_field_periods")
    p.add_argument("--directed-direction", default="max", choices=["min", "max"])
    p.add_argument("--directed-frac", type=float, default=0.2)
    p.add_argument("--max-std", type=float, default=3.5, help="cycle-schedule peak cap -- see bootstrap_loop.py")
    p.add_argument("--retrain-tag-prefix", default="reg_mlp_big_edgerot")
    p.add_argument("--retrain-epochs", type=int, default=300, help="matches the original reg_mlp_big_soap_* recipe")
    p.add_argument("--retrain-optimizer", default="adam", choices=["adam", "soap"],
                    help="adam is much cheaper per-step than soap (no preconditioner computation) -- "
                         "this loop cares about wall-clock iteration speed across many blocks over the "
                         "run's budget, not squeezing out the last bit of per-block surrogate accuracy, "
                         "so adam is the default here even though soap was the production choice "
                         "(reg_mlp_big_soap_s{0,1,2}) when speed wasn't the constraint.")
    p.add_argument("--leak-check", action="store_true",
                    help="run augment_split.py's full nearest-real-neighbor cluster-membership check "
                         "(defends against the §6 leakage failure mode, ~65min+ and growing at this pool "
                         "size) instead of the default fast path that just includes every generated design "
                         "unconditionally. Off by default -- this loop's surrogate is only steering search "
                         "direction each block, not being reported as a trustworthy generalization number; "
                         "rerun augment_split.py without --no-leak-check separately before trusting one.")
    p.add_argument("--search-nfp", type=int, default=3, help="matches the nfp used in this session's design-search checks")
    p.add_argument("--progress-log", default=str(OUT_DIR / "long_run_progress.jsonl"))
    args = p.parse_args()

    t_start = time.perf_counter()
    budget_s = args.budget_hours * 3600
    block = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_s:
            print(f"[long_run] budget of {args.budget_hours}h exhausted ({elapsed / 3600:.1f}h elapsed), stopping")
            break
        if block >= args.max_blocks:
            print(f"[long_run] hit --max-blocks={args.max_blocks}, stopping")
            break
        block += 1
        print(f"\n[long_run] ===== block {block} start, {elapsed / 3600:.1f}h elapsed of "
              f"{args.budget_hours}h budget =====", flush=True)

        run([sys.executable, "scripts/bootstrap_loop.py",
             "--tag", args.bootstrap_tag, "--schedule", "cycle", "--max-std", str(args.max_std),
             "--directed-frac", str(args.directed_frac),
             "--directed-target", args.directed_target, "--directed-direction", args.directed_direction,
             "--generations", str(args.gens_per_block)],
            f"block {block} bootstrap ({args.gens_per_block} generations)")

        split_tag = "bootstrap_live"
        augment_cmd = [sys.executable, "scripts/augment_split.py",
                        "--generated-dir", str(OUT_DIR / f"bootstrap_{args.bootstrap_tag}"),
                        "--tag", split_tag]
        if not args.leak_check:
            augment_cmd.append("--no-leak-check")
        split_ok = run(augment_cmd, f"block {block} augment_split")

        member_tags = []
        if split_ok:
            for seed in (0, 1, 2):
                tag = f"{args.retrain_tag_prefix}_block{block}_s{seed}"
                retrain_ok = run([sys.executable, "scripts/train.py",
                                   "--split", split_tag, "--hidden", "2048", "--latent", "512",
                                   "--head-hidden", "256", "--no-spatial", "--optimizer", args.retrain_optimizer,
                                   "--epochs", str(args.retrain_epochs), "--val-interval", "5",
                                   "--seed", str(seed), "--tag", tag],
                                  f"block {block} retrain seed {seed}")
                if retrain_ok:
                    member_tags.append(tag)

        if len(member_tags) == 3:
            run([sys.executable, "scripts/latent_optimize.py",
                 "--minimize", "max_elongation",
                 "--constraint", "aspect_ratio<=4.0",
                 "--constraint", "average_triangularity<=-0.5",
                 "--constraint", "abs(edge_rotational_transform_over_n_field_periods)>=0.3",
                 "--nfp", str(args.search_nfp), "--score-bounds", "1.0", "10.0",
                 "--member-tags", *member_tags,
                 "--bootstrap-tag", args.bootstrap_tag,
                 "--save", str(OUT_DIR / f"latent_geometric_candidate_block{block}.json"),
                 "--progress-log", args.progress_log,
                 "--run-label", f"block{block}"],
                f"block {block} design-search checkpoint")
        else:
            print(f"[long_run] block {block}: retrain incomplete ({len(member_tags)}/3 seeds succeeded) -- "
                  f"skipping this block's design-search checkpoint, pool growth is unaffected", flush=True)

        elapsed = time.perf_counter() - t_start
        print(f"[long_run] ===== block {block} done, {elapsed / 3600:.2f}h elapsed =====", flush=True)

    print(f"[long_run] finished after {block} blocks")


if __name__ == "__main__":
    main()

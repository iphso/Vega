"""Evaluate an ensemble of independently-seeded contrastive checkpoints
(--objective contrastive) against the held-out test set: probability-averaged
3-way accuracy, concordance, and tie precision/recall (see
contrastive_ensemble_eval_metrics for why probabilities are averaged rather
than raw scores). Also reports each single member alone for comparison.
"""
import argparse

import torch

from train import (
    CKPT_DIR,
    DualPathMLP,
    contrastive_ensemble_eval_metrics,
    contrastive_eval_metrics,
    load_split,
    noise_floor_eps,
    print_contrastive_metrics,
)


def load_model(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    assert ckpt["objective"] == "contrastive", f"{tag} is not a contrastive checkpoint"
    eps = noise_floor_eps(ckpt["target_names"]).to(dev)
    model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"],
        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"],
        spatial_latent=ckpt["spatial_latent"], head_hidden=ckpt["head_hidden"],
        priority_weight=ckpt["priority_weight"], use_spatial=ckpt["use_spatial"],
        trunk_arch=ckpt.get("trunk_arch", "mlp"), trunk_blocks=ckpt.get("trunk_blocks", 3),
        use_symlog_latent=ckpt.get("use_symlog_latent", False),
        log_target_mask=ckpt.get("log_target_mask"),
        objective="contrastive", noise_floor_eps=eps,
    ).to(dev)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not result.unexpected_keys
    model.eval()
    return model, ckpt["target_names"], eps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member-tags", nargs="+", required=True,
                    help="checkpoint stems for the ensemble members (same architecture, different seeds)")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    X_test, Y_test = load_split("test")
    X_test, Y_test = X_test.to(dev), Y_test.to(dev)

    member_scores, member_log_nus, target_names, eps = [], [], None, None
    for tag in args.member_tags:
        model, target_names, eps = load_model(tag, dev)
        with torch.no_grad():
            member_scores.append(model(X_test))
        member_log_nus.append(model.log_nu.detach())

    print(f"\n=== single members: {args.member_tags} ===")
    for tag, scores in zip(args.member_tags, member_scores):
        idx = args.member_tags.index(tag)
        metrics = contrastive_eval_metrics(scores, Y_test, eps, log_nu=member_log_nus[idx], chunk=args.chunk)
        print_contrastive_metrics(tag, metrics, target_names)

    ensemble_metrics = contrastive_ensemble_eval_metrics(
        member_scores, member_log_nus, Y_test, eps, chunk=args.chunk
    )
    print(f"\n=== ensemble of {len(args.member_tags)} (probability-averaged) ===")
    print_contrastive_metrics("ensemble", ensemble_metrics, target_names)


if __name__ == "__main__":
    main()

"""Experiment 1: contrastive estimation at test time from an already-trained
absolute-regression checkpoint.

No retraining. Takes a regression checkpoint's predicted physical-unit
values, forms all in-batch pairs against the noise-floor epsilon (the same
ground-truth tie rule used everywhere else in the contrastive experiments),
and reports how well "bigger / smaller / basically the same" calls made from
those raw predicted values line up with the ground truth. This is the
baseline the from-scratch contrastive model (train.py --objective
contrastive) has to beat.
"""
import argparse
from pathlib import Path

import torch

from train import (
    CKPT_DIR,
    DualPathMLP,
    compute_geom_features,
    contrastive_eval_metrics,
    load_split,
    noise_floor_eps,
    print_contrastive_metrics,
)


def load_regression_model(tag, dev):
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev)
    model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"],
        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"],
        spatial_latent=ckpt["spatial_latent"], head_hidden=ckpt["head_hidden"],
        priority_weight=ckpt["priority_weight"], use_spatial=ckpt["use_spatial"],
        trunk_arch=ckpt.get("trunk_arch", "mlp"), trunk_blocks=ckpt.get("trunk_blocks", 3),
        use_symlog_latent=ckpt.get("use_symlog_latent", False),
        log_target_mask=ckpt.get("log_target_mask"),
    ).to(dev)
    # Older checkpoints predate the log_target_mask buffer; the reconstructed
    # model already defaults it to all-False (no log-space targets) via
    # log_target_mask=ckpt.get(...) above, so a missing key here is expected
    # and harmless -- non-strict load, but report anything else missing.
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    unexpected_missing = [k for k in result.missing_keys if k != "log_target_mask"]
    assert not unexpected_missing and not result.unexpected_keys, (
        result.missing_keys, result.unexpected_keys
    )
    model.eval()
    return model, ckpt["target_names"], ckpt.get("geom_features", False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", default=["single_base", "siren_base"],
                    help="checkpoint filename stems (without .pt) to evaluate")
    p.add_argument("--chunk", type=int, default=512, help="within-chunk all-pairs size for eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    X_test, Y_test = load_split("test")

    for tag in args.tags:
        model, target_names, geom_features = load_regression_model(tag, dev)
        eps = noise_floor_eps(target_names).to(dev)

        Xt = X_test
        if geom_features:
            Xt = torch.cat([Xt, compute_geom_features(Xt)], dim=1)
        Xt, Yt = Xt.to(dev), Y_test.to(dev)

        with torch.no_grad():
            scores = model(Xt)
            if model.log_target_mask.any():
                scores = scores.clone()
                scores[:, model.log_target_mask] = torch.exp(scores[:, model.log_target_mask])

        metrics = contrastive_eval_metrics(scores, Yt, eps, log_nu=None, chunk=args.chunk)

        print(f"\n=== {tag} (regression checkpoint, contrastive-at-test-time, no retraining) ===")
        print_contrastive_metrics(tag, metrics, target_names)


if __name__ == "__main__":
    main()

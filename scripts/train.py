import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from soap import SOAP

OUT_DIR = Path("/work/output")
CKPT_DIR = Path("/work/checkpoints")

# Structural constants tied to the preprocessing feature layout
# (scripts/preprocess.py): r_cos flat[0:45], z_sin flat[45:90],
# n_field_periods at [90], symmetry flag at [91].
N_MODES_M = 5
N_MODES_N = 9
N_COEFFS = N_MODES_M * N_MODES_N  # 45
IDX_R_COS = slice(0, N_COEFFS)
IDX_Z_SIN = slice(N_COEFFS, 2 * N_COEFFS)
IDX_NFP = 2 * N_COEFFS

# Angular grid the fixed spectral->spatial (inverse Fourier series) transform
# is evaluated on for the spatial-domain encoder branch.
GRID_NU = 16
GRID_NV = 16

# Targets with strictly-positive, wide multiplicative dynamic range (checked
# against output/metadata.json target_stats: max/min ratios of ~114x-1.8e7x),
# good candidates for predicting log(target) instead of target directly.
LOG_TARGET_NAMES = [
    "qi",
    "max_elongation",
    "flux_compression_in_regions_of_bad_curvature",
    "minimum_normalized_magnetic_gradient_scale_length",
]


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


# Per-target "basically the same" tolerance for the contrastive experiments:
# the measured run-to-run standard deviation of 3 identical single_base
# reruns (EXPERIMENT_LOG.md section 5). A pairwise difference smaller than
# this is indistinguishable from what a rerun of the exact same config would
# produce by chance alone -- a defensible, already-measured definition of
# "tie" rather than a hand-picked relative percentage.
NOISE_FLOOR_BY_NAME = {
    "qi": 0.00008,
    "vacuum_well": 0.00121,
    "aspect_ratio": 0.00491,
    "max_elongation": 0.01900,
    "average_triangularity": 0.00187,
    "axis_magnetic_mirror_ratio": 0.00015,
    "edge_magnetic_mirror_ratio": 0.00027,
    "axis_rotational_transform_over_n_field_periods": 0.00049,
    "edge_rotational_transform_over_n_field_periods": 0.00042,
    "flux_compression_in_regions_of_bad_curvature": 0.00132,
    "minimum_normalized_magnetic_gradient_scale_length": 0.00904,
}


def noise_floor_eps(target_names):
    return torch.tensor([NOISE_FLOOR_BY_NAME[n] for n in target_names])


def all_pairs_labels(Y, eps):
    """All C(B,2) unordered within-batch pairs and their ground-truth label.

    Y: (B, T) targets, eps: (T,) tie tolerance. Returns (i_idx, j_idx, label)
    with label (P, T) in {-1, 0, +1}: +1 row i's target is bigger, -1 row j's
    is bigger, 0 "basically the same" (|diff| within eps).
    """
    B = Y.shape[0]
    i_idx, j_idx = torch.triu_indices(B, B, offset=1, device=Y.device)
    diff = Y[i_idx] - Y[j_idx]  # (P, T)
    label = torch.zeros_like(diff)
    label = torch.where(diff > eps, torch.ones_like(diff), label)
    label = torch.where(diff < -eps, -torch.ones_like(diff), label)
    return i_idx, j_idx, label


def davidson_log_probs(d, log_nu):
    """Bradley-Terry-with-ties (Davidson, 1970) log-probabilities.

    d = s_i - s_j (P, T) is the model's own score difference (not
    necessarily in physical units). log_nu (T,) is a learned per-target log
    tie-propensity -- the contrastive analogue of this project's learned
    log_var uncertainty weight: instead of hand-picking one epsilon for the
    model to call ties by, each target learns how much of a score gap in its
    own units counts as "basically the same", the same way log_var lets each
    target learn how much to trust its own raw-scale error.
    Returns (log_p_win_i, log_p_win_j, log_p_tie), each (P, T).
    """
    log_nu = log_nu.expand_as(d)
    log_Z = torch.logsumexp(torch.stack([d / 2, -d / 2, log_nu], dim=0), dim=0)
    return d / 2 - log_Z, -d / 2 - log_Z, log_nu - log_Z


def contrastive_predicted_label(d, eps=None, log_nu=None):
    """Predicted {-1, 0, +1} label from a score difference d (P, T).

    With log_nu given (a Davidson-trained contrastive model): argmax over the
    model's own {win_i, tie, win_j} probabilities. Without it (a plain
    regression model's raw physical-unit output, scored at test time with no
    learned tie sense of its own): threshold d directly against the same
    noise-floor eps used to build ground truth, for the most direct
    apples-to-apples comparison between the two experiments.
    """
    if log_nu is not None:
        log_p_win_i, log_p_win_j, log_p_tie = davidson_log_probs(d, log_nu)
        stacked = torch.stack([log_p_win_j, log_p_tie, log_p_win_i], dim=0)  # order -1,0,+1
        return stacked.argmax(dim=0).float() - 1.0
    label = torch.zeros_like(d)
    label = torch.where(d > eps, torch.ones_like(d), label)
    label = torch.where(d < -eps, -torch.ones_like(d), label)
    return label


def contrastive_eval_metrics(scores, Y, eps, log_nu=None, chunk=512, seed=0):
    """Per-target contrastive metrics over one deterministic sweep of
    within-chunk all-pairs: rows shuffled once, then partitioned
    sequentially into chunks (same in-batch-all-pairs construction as
    training), so every row participates in exactly one chunk's C(chunk,2)
    pairs rather than being resampled at random.

    Returns a dict of (T,) tensors:
    - acc3: 3-way exact-match accuracy against noise-floor ground truth.
    - concordance: accuracy restricted to pairs where ground truth is NOT a
      tie -- "did we get bigger/smaller right", independent of tie-calling,
      the threshold-free ranking-quality number.
    - tie_precision / tie_recall: how well "basically the same" calls match
      noise-floor ground-truth ties.
    - tie_rate_true: fraction of ground-truth pairs that are actually ties,
      for context (accuracy alone is misleading if this is near 0 or 1).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(scores.shape[0], generator=g).to(scores.device)
    scores, Y = scores[perm], Y[perm]

    T = scores.shape[1]
    correct3 = torch.zeros(T, device=scores.device)
    nontie_total = torch.zeros(T, device=scores.device)
    nontie_correct = torch.zeros(T, device=scores.device)
    tie_true = torch.zeros(T, device=scores.device)
    tie_pred = torch.zeros(T, device=scores.device)
    tie_true_and_pred = torch.zeros(T, device=scores.device)
    total = torch.zeros(T, device=scores.device)

    for start in range(0, scores.shape[0] - 1, chunk):
        s_c = scores[start:start + chunk]
        y_c = Y[start:start + chunk]
        if s_c.shape[0] < 2:
            continue
        i_idx, j_idx, true_label = all_pairs_labels(y_c, eps)
        d = s_c[i_idx] - s_c[j_idx]
        pred_label = contrastive_predicted_label(d, eps=eps, log_nu=log_nu)

        match = (pred_label == true_label).float()
        correct3 += match.sum(dim=0)
        total += true_label.shape[0]

        nontie_mask = (true_label != 0).float()
        nontie_total += nontie_mask.sum(dim=0)
        nontie_correct += (match * nontie_mask).sum(dim=0)

        tie_true_mask = (true_label == 0).float()
        tie_pred_mask = (pred_label == 0).float()
        tie_true += tie_true_mask.sum(dim=0)
        tie_pred += tie_pred_mask.sum(dim=0)
        tie_true_and_pred += (tie_true_mask * tie_pred_mask).sum(dim=0)

    return {
        "acc3": correct3 / total.clamp_min(1),
        "concordance": nontie_correct / nontie_total.clamp_min(1),
        "tie_precision": tie_true_and_pred / tie_pred.clamp_min(1),
        "tie_recall": tie_true_and_pred / tie_true.clamp_min(1),
        "tie_rate_true": tie_true / total.clamp_min(1),
    }


def contrastive_ensemble_eval_metrics(member_scores, member_log_nus, Y, eps, chunk=512, seed=0):
    """Same metrics as contrastive_eval_metrics, but for an ensemble of
    independently-trained contrastive models. Each member has its own score
    scale and its own learned log_nu (they're not calibrated to a shared
    scale the way regression physical-unit outputs are), so averaging raw
    scores across members before applying any one member's log_nu wouldn't
    be principled. Instead: each member computes its own {win_i, tie, win_j}
    probabilities from its own (score, log_nu) pair, per pair; those
    probabilities are averaged across members (standard probability-averaging
    ensembling for a classifier); the ensemble's predicted label is the
    argmax of the averaged probabilities.

    member_scores: list of (N, T) tensors, one per member, in the same row
    order as Y (this function applies its own shuffle to Y and every
    member's scores together, consistent with contrastive_eval_metrics).
    member_log_nus: list of (T,) tensors, one per member.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(Y.shape[0], generator=g).to(Y.device)
    Y = Y[perm]
    member_scores = [scores[perm] for scores in member_scores]

    T = Y.shape[1]
    correct3 = torch.zeros(T, device=Y.device)
    nontie_total = torch.zeros(T, device=Y.device)
    nontie_correct = torch.zeros(T, device=Y.device)
    tie_true = torch.zeros(T, device=Y.device)
    tie_pred = torch.zeros(T, device=Y.device)
    tie_true_and_pred = torch.zeros(T, device=Y.device)
    total = torch.zeros(T, device=Y.device)

    n = Y.shape[0]
    for start in range(0, n - 1, chunk):
        y_c = Y[start:start + chunk]
        if y_c.shape[0] < 2:
            continue
        i_idx, j_idx, true_label = all_pairs_labels(y_c, eps)

        p_win_i = torch.zeros_like(true_label)
        p_win_j = torch.zeros_like(true_label)
        p_tie = torch.zeros_like(true_label)
        for scores, log_nu in zip(member_scores, member_log_nus):
            s_c = scores[start:start + chunk]
            d = s_c[i_idx] - s_c[j_idx]
            log_p_win_i, log_p_win_j, log_p_tie = davidson_log_probs(d, log_nu)
            p_win_i += log_p_win_i.exp()
            p_win_j += log_p_win_j.exp()
            p_tie += log_p_tie.exp()
        stacked = torch.stack([p_win_j, p_tie, p_win_i], dim=0)  # order -1,0,+1
        pred_label = stacked.argmax(dim=0).float() - 1.0

        match = (pred_label == true_label).float()
        correct3 += match.sum(dim=0)
        total += true_label.shape[0]

        nontie_mask = (true_label != 0).float()
        nontie_total += nontie_mask.sum(dim=0)
        nontie_correct += (match * nontie_mask).sum(dim=0)

        tie_true_mask = (true_label == 0).float()
        tie_pred_mask = (pred_label == 0).float()
        tie_true += tie_true_mask.sum(dim=0)
        tie_pred += tie_pred_mask.sum(dim=0)
        tie_true_and_pred += (tie_true_mask * tie_pred_mask).sum(dim=0)

    return {
        "acc3": correct3 / total.clamp_min(1),
        "concordance": nontie_correct / nontie_total.clamp_min(1),
        "tie_precision": tie_true_and_pred / tie_pred.clamp_min(1),
        "tie_recall": tie_true_and_pred / tie_true.clamp_min(1),
        "tie_rate_true": tie_true / total.clamp_min(1),
    }


class SineLayer(nn.Module):
    """SIREN sinusoidal layer (Sitzmann et al. 2020), with their init scheme:
    first layer uses a wide uniform range (high frequency content), hidden
    layers use a narrower range scaled by 1/omega_0 to keep the pre-activation
    distribution stable through depth.
    """

    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_f, 1 / in_f)
            else:
                bound = math.sqrt(6 / in_f) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenTrunk(nn.Module):
    """Pure sinusoidal-activation trunk. Final projection is a plain linear
    layer (no sine) since we want unrestricted regression features out, not
    a value bounded by sin's range.
    """

    def __init__(self, in_dim, hidden, latent_dim, first_omega=30.0, hidden_omega=1.0):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(in_dim, hidden, is_first=True, omega_0=first_omega),
            SineLayer(hidden, hidden, is_first=False, omega_0=hidden_omega),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class HalfSirenBlock(nn.Module):
    """One layer, split down the middle: half the output units come from a
    sine activation, half from ReLU. Both halves see the FULL input to the
    block (including the other half's output from the previous block), so
    sine-derived and ReLU-derived features actually get to interact and
    recombine at every layer, not just once at the very end.
    """

    def __init__(self, in_dim, out_dim, is_first=False, first_omega=30.0, hidden_omega=1.0):
        super().__init__()
        half_out = out_dim // 2
        self.sine = SineLayer(in_dim, half_out, is_first=is_first,
                               omega_0=first_omega if is_first else hidden_omega)
        self.relu = nn.Sequential(nn.Linear(in_dim, out_dim - half_out), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.sine(x), self.relu(x)], dim=-1)


class HalfSirenTrunk(nn.Module):
    """A chain of HalfSirenBlocks, each full-width (default `hidden`), each
    half sine / half ReLU, stacked so depth (and therefore capacity) is
    controllable via n_blocks -- unlike a single split-once-at-the-end
    design, this lets sine and ReLU features mix across every layer.
    """

    def __init__(self, in_dim, hidden, latent_dim, n_blocks=3, first_omega=30.0, hidden_omega=1.0):
        super().__init__()
        blocks = []
        d_in = in_dim
        for i in range(n_blocks):
            blocks.append(HalfSirenBlock(d_in, hidden, is_first=(i == 0),
                                          first_omega=first_omega, hidden_omega=hidden_omega))
            d_in = hidden
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        return torch.relu(self.out_proj(self.blocks(x)))


class ModeAttentionEncoder(nn.Module):
    """Treats each of the 45 (m, n) Fourier modes as a token -- its
    [r_cos, z_sin] coefficient pair -- with a learned positional embedding,
    plus one extra token for [n_field_periods, symmetry_flag]. Self-attention
    lets the network learn which modes interact with which directly, instead
    of only through a dense matmul over the flattened vector.
    """

    def __init__(self, latent_dim, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        self.token_proj = nn.Linear(2, d_model)
        self.extra_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(N_COEFFS, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        r_cos = x[:, IDX_R_COS]  # (batch, 45)
        z_sin = x[:, IDX_Z_SIN]  # (batch, 45)
        tokens = torch.stack([r_cos, z_sin], dim=-1)  # (batch, 45, 2)
        tokens = self.token_proj(tokens) + self.pos_embed.unsqueeze(0)

        extra = x[:, IDX_NFP:IDX_NFP + 2]  # n_field_periods, symmetry flag
        extra_tok = self.extra_proj(extra).unsqueeze(1)
        tokens = torch.cat([tokens, extra_tok], dim=1)  # (batch, 46, d_model)

        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return torch.relu(self.out_proj(pooled))


def build_spectral_trunk(arch, in_dim, hidden, latent_dim, n_blocks=3):
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )
    if arch == "siren":
        return SirenTrunk(in_dim, hidden, latent_dim)
    if arch == "half_siren":
        return HalfSirenTrunk(in_dim, hidden, latent_dim, n_blocks=n_blocks)
    if arch == "attention":
        return ModeAttentionEncoder(latent_dim)
    raise ValueError(f"unknown trunk arch: {arch}")


class SpatialEncoderCNN(nn.Module):
    """Encodes the (R, Z) boundary grid as a 2-channel image on a torus.

    theta and zeta are both periodic angles, so plain zero-padding would
    invent a fake discontinuity at the grid edge; circular padding makes
    the convolution respect that periodicity. Strided-by-pooling downsampling
    builds up receptive field so later layers see global shape (overall
    elongation/extent), not just per-point coefficients.
    """

    def __init__(self, grid_nu, grid_nv, out_dim=64, channels=(16, 32, 32)):
        super().__init__()
        layers = []
        c_in = 2
        n_pools = 0
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, padding_mode="circular"),
                nn.ReLU(),
                nn.AvgPool2d(2),
            ]
            c_in = c_out
            n_pools += 1
        assert grid_nu % (2 ** n_pools) == 0 and grid_nv % (2 ** n_pools) == 0, (
            "GRID_NU/GRID_NV must be divisible by 2**len(channels) for this pooling schedule"
        )
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c_in, out_dim)

    def forward(self, grid):  # grid: (batch, 2, Nu, Nv)
        h = self.conv(grid)
        h = self.pool(h).flatten(1)
        return torch.relu(self.proj(h))


class DualPathMLP(nn.Module):
    """Two parallel encoder branches, fused before the target heads.

    - spectral branch: a vanilla MLP directly on the raw input (Fourier
      coefficients r_cos/z_sin, n_field_periods, symmetry flag).
    - spatial branch: the same r_cos/z_sin/n_field_periods first run
      through a fixed, non-learned inverse Fourier series transform
          R(theta, zeta) = sum_mn r_cos[m,n] * cos(m*theta - n*Nfp*zeta)
          Z(theta, zeta) = sum_mn z_sin[m,n] * sin(m*theta - n*Nfp*zeta)
      onto a fixed angular grid, producing an actual spatial-domain
      boundary shape as a 2-channel image, which a small circular-CNN
      (SpatialEncoderCNN) then encodes -- preserving the 2D/periodic
      structure that a flatten+MLP would throw away.

    The two latents are concatenated and passed through a small fusion
    layer (so heads see a genuinely mixed representation, not just two
    disjoint chunks) before the per-target heads. Targets are never
    normalized; scale differences across the 11 targets are instead
    handled by learned per-task uncertainty weighting (Kendall & Gal
    2018) in `weighted_loss`.

    Set use_spatial=False to disable the spatial branch entirely (a plain
    single-path MLP on the raw input, everything else -- heads, fusion,
    loss -- identical), for apples-to-apples ablations.

    Two optional decoder-side changes, for targets that aren't necessarily
    a linear function of the latent:
    - use_symlog_latent: concatenate symlog(z) = sign(z)*log1p(|z|) onto z
      before the heads, so each head's own weights can decide how much to
      lean on a log-compressed view of the latent, without any per-target
      special-casing.
    - log_target_mask: a bool mask over targets (see LOG_TARGET_NAMES) whose
      heads predict log(target) instead of target directly -- for targets
      with huge multiplicative dynamic range (e.g. min_norm_grad_scale_len
      spans ~7 orders of magnitude), asking a linear head to represent that
      directly is a much harder function to fit than its log.
    """

    def __init__(self, in_dim, n_targets, latent_dim=128, hidden=256, spatial_latent=64,
                 head_hidden=64, priority_weight=None, use_spatial=True, trunk_arch="mlp",
                 trunk_blocks=3, use_symlog_latent=False, log_target_mask=None,
                 objective="regression", noise_floor_eps=None):
        super().__init__()
        self.objective = objective
        self.use_spatial = use_spatial
        self.use_symlog_latent = use_symlog_latent
        self.trunk_spectral = build_spectral_trunk(trunk_arch, in_dim, hidden, latent_dim, n_blocks=trunk_blocks)

        if use_spatial:
            self.spatial_encoder = SpatialEncoderCNN(GRID_NU, GRID_NV, out_dim=spatial_latent)
            combined_dim = latent_dim + spatial_latent
        else:
            self.spatial_encoder = None
            combined_dim = latent_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(),
        )
        head_in_dim = combined_dim * 2 if use_symlog_latent else combined_dim
        # Small 2-layer MLP per target instead of a single linear readout --
        # a scalar equilibrium property isn't necessarily a linear function
        # of the fused latent, and a single Linear(latent, 1) was likely
        # underfitting the harder targets.
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
            for _ in range(n_targets)
        ])
        self.log_vars = nn.Parameter(torch.zeros(n_targets))
        if objective == "contrastive":
            # Learned per-target log tie-propensity for the Davidson pairwise
            # model (see davidson_log_probs) -- only meaningful, and only
            # present, for contrastively-trained checkpoints so old
            # regression checkpoints keep loading with a strict state dict.
            self.log_nu = nn.Parameter(torch.zeros(n_targets))
            assert noise_floor_eps is not None, "contrastive objective requires noise_floor_eps"
            self.register_buffer("noise_floor_eps", noise_floor_eps)
        if priority_weight is None:
            priority_weight = torch.ones(n_targets)
        self.register_buffer("priority_weight", priority_weight)
        if log_target_mask is None:
            log_target_mask = torch.zeros(n_targets, dtype=torch.bool)
        self.register_buffer("log_target_mask", log_target_mask)

        m = torch.arange(N_MODES_M).float()
        n = torch.arange(N_MODES_N).float() - (N_MODES_N - 1) / 2  # centered, e.g. -4..4
        theta = torch.arange(GRID_NU).float() / GRID_NU * 2 * math.pi
        zeta = torch.arange(GRID_NV).float() / GRID_NV * 2 * math.pi
        self.register_buffer("m_grid", m)
        self.register_buffer("n_grid", n)
        self.register_buffer("theta_grid", theta)
        self.register_buffer("zeta_grid", zeta)

    def spectral_to_spatial(self, r_cos, z_sin, nfp):
        # r_cos, z_sin: (batch, M, N); nfp: (batch,) -> R, Z: (batch, Nu, Nv)
        m = self.m_grid.view(1, N_MODES_M, 1, 1, 1)
        n = self.n_grid.view(1, 1, N_MODES_N, 1, 1)
        theta = self.theta_grid.view(1, 1, 1, GRID_NU, 1)
        zeta = self.zeta_grid.view(1, 1, 1, 1, GRID_NV)
        nfp_ = nfp.view(-1, 1, 1, 1, 1)
        arg = m * theta - n * nfp_ * zeta  # (batch, M, N, Nu, Nv)
        R = torch.einsum("bmn,bmnuv->buv", r_cos, torch.cos(arg))
        Z = torch.einsum("bmn,bmnuv->buv", z_sin, torch.sin(arg))
        return R, Z

    def encode(self, x):
        latent_spectral = self.trunk_spectral(x)
        if not self.use_spatial:
            return self.fusion(latent_spectral)

        r_cos = x[:, IDX_R_COS].view(-1, N_MODES_M, N_MODES_N)
        z_sin = x[:, IDX_Z_SIN].view(-1, N_MODES_M, N_MODES_N)
        nfp = x[:, IDX_NFP]
        R, Z = self.spectral_to_spatial(r_cos, z_sin, nfp)
        grid = torch.stack([R, Z], dim=1)  # (batch, 2, Nu, Nv)
        latent_spatial = self.spatial_encoder(grid)
        return self.fusion(torch.cat([latent_spectral, latent_spatial], dim=1))

    def predict(self, z):
        if self.use_symlog_latent:
            z = torch.cat([z, symlog(z)], dim=-1)
        return torch.cat([head(z) for head in self.heads], dim=1)

    def forward(self, x):
        return self.predict(self.encode(x))

    def weighted_loss(self, pred, target):
        # pred is in "mixed" space: heads flagged in log_target_mask output
        # log(target), everything else outputs target directly. Training
        # loss operates in that mixed space; per_task_mse (for reporting/
        # checkpoint selection) is always converted back to physical units
        # so it stays comparable across every other experiment in this
        # search.
        if self.log_target_mask.any():
            target_train = target.clone()
            target_train[:, self.log_target_mask] = torch.log(
                target[:, self.log_target_mask].clamp_min(1e-12)
            )
        else:
            target_train = target

        sq_err = (pred - target_train) ** 2  # (batch, n_targets), mixed log/physical space
        per_task_mse = sq_err.mean(dim=0)  # (n_targets,)
        precision = torch.exp(-self.log_vars)
        # priority_weight is a fixed (non-learned) multiplier on top of the
        # automatically-learned uncertainty weight, for targets we care
        # about more than the automatic scheme alone would reflect.
        per_task_loss = self.priority_weight * (precision * per_task_mse + self.log_vars)

        if self.log_target_mask.any():
            pred_phys = pred.clone()
            pred_phys[:, self.log_target_mask] = torch.exp(pred[:, self.log_target_mask])
            per_task_mse_report = ((pred_phys - target) ** 2).mean(dim=0).detach()
        else:
            per_task_mse_report = per_task_mse.detach()

        return per_task_loss.sum(), per_task_mse_report

    def contrastive_loss(self, scores, targets, weight_cap=5.0):
        """Davidson pairwise NLL over all in-batch pairs (see all_pairs_labels
        / davidson_log_probs). `scores` is this model's raw forward() output
        -- the same per-target scalar head used for regression, just trained
        as a comparator instead of a value estimate.

        Per-target adaptive class reweighting. True tie-rate varies by two
        orders of magnitude across targets (~0.07% for axis_magnetic_mirror_
        ratio vs ~8.4% for average_triangularity), so neither extreme of a
        single global knob works everywhere: the plain per-pair mean (no
        reweighting) is the correct MLE, but gives the tie class essentially
        no gradient signal on the rare-tie targets; full class-balance (equal
        1/3 weight per class regardless of count) fixes that but overcorrects
        hard where it wasn't needed -- confirmed empirically at a flat
        alpha=1: tie recall ~0 -> ~0.9, but concordance fell ~10 points
        because the model starts crying "tie" on plenty of genuine win/lose
        pairs too. A flat intermediate alpha=0.5 split the difference but
        still under-served the rarest-tie targets while presumably
        over-correcting easier ones.

        So make the correction strength itself a function of each target's
        own live tie-rate, from the very same in-batch counts already being
        computed here, rather than one global constant. Per target: let
        ratio = n_max/n_min over the three per-target class counts (usually
        n_max is one of the win_i/win_j classes and n_min is the rare tie
        class). Solve for the interpolation weight `alpha_k` (see
        weight_cap) that would bring the *effective* majority:minority
        weight ratio down to at most `weight_cap`, instead of either leaving
        it at its natural (possibly huge) value or fully flattening it to 1.
        A target with a naturally small ratio (already-servable ties, e.g.
        average_triangularity) gets alpha_k near 0 -- little to no
        correction needed. A target with a huge natural ratio (vanishingly
        rare ties) gets alpha_k pushed close to 1 -- strong correction,
        without hand-tuning a per-target constant.

        Unlike weighted_loss, per-task terms need no learned uncertainty
        weighting to combine across targets: a log-likelihood in nats is
        already scale-free, regardless of a target's raw physical units, so
        summing per-task loss directly is a fair combination on its own.
        """
        i_idx, j_idx, true_label = all_pairs_labels(targets, self.noise_floor_eps)
        d = scores[i_idx] - scores[j_idx]
        log_p_win_i, log_p_win_j, log_p_tie = davidson_log_probs(d, self.log_nu)
        nll = torch.where(
            true_label > 0.5, -log_p_win_i,
            torch.where(true_label < -0.5, -log_p_win_j, -log_p_tie),
        )
        mask_win_i = (true_label > 0.5).float()
        mask_win_j = (true_label < -0.5).float()
        mask_tie = 1.0 - mask_win_i - mask_win_j

        n_win_i = mask_win_i.sum(dim=0).clamp_min(1)
        n_win_j = mask_win_j.sum(dim=0).clamp_min(1)
        n_tie = mask_tie.sum(dim=0).clamp_min(1)
        mean_win_i = (nll * mask_win_i).sum(dim=0) / n_win_i
        mean_win_j = (nll * mask_win_j).sum(dim=0) / n_win_j
        mean_tie = (nll * mask_tie).sum(dim=0) / n_tie

        # weight_c = n_c ** (1 - alpha_k), solved per target so the resulting
        # max:min weight ratio is exactly min(natural ratio, weight_cap).
        n_max = torch.maximum(torch.maximum(n_win_i, n_win_j), n_tie)
        n_min = torch.minimum(torch.minimum(n_win_i, n_win_j), n_tie)
        ratio = n_max / n_min
        target_ratio = ratio.clamp(max=weight_cap)
        alpha_k = 1.0 - torch.log(target_ratio) / torch.log(ratio).clamp_min(1e-8)
        alpha_k = alpha_k.clamp(0.0, 1.0)

        w_win_i = n_win_i ** (1 - alpha_k)
        w_win_j = n_win_j ** (1 - alpha_k)
        w_tie = n_tie ** (1 - alpha_k)
        w_sum = w_win_i + w_win_j + w_tie
        per_task_loss = (w_win_i * mean_win_i + w_win_j * mean_win_j + w_tie * mean_tie) / w_sum
        return per_task_loss.sum(), per_task_loss.detach()


def load_split(name, data_dir=None):
    data = np.load((data_dir or OUT_DIR) / f"{name}.npz")
    return torch.from_numpy(data["X"]).float(), torch.from_numpy(data["Y"]).float()


def compute_geom_features(X, grid_nu=16, grid_nv=16, chunk=4096):
    """Cheap, non-learned summary statistics of the reconstructed (R, Z)
    boundary shape -- extent/spread per toroidal angle, aggregated over
    angle. Not exact physics formulas (aspect ratio/elongation/triangularity
    have specific VMEC/DESC definitions we don't reproduce here) -- just
    generic geometric descriptors the network would otherwise have to
    re-derive from raw coefficients via many layers of learned matmuls.
    Computed once over the whole split (chunked to bound memory), not
    per-batch during training, since it's a fixed deterministic transform.
    """
    m = torch.arange(N_MODES_M).float().view(1, N_MODES_M, 1, 1, 1)
    n = (torch.arange(N_MODES_N).float() - (N_MODES_N - 1) / 2).view(1, 1, N_MODES_N, 1, 1)
    theta = (torch.arange(grid_nu).float() / grid_nu * 2 * math.pi).view(1, 1, 1, grid_nu, 1)
    zeta = (torch.arange(grid_nv).float() / grid_nv * 2 * math.pi).view(1, 1, 1, 1, grid_nv)

    feats_list = []
    for start in range(0, X.shape[0], chunk):
        Xc = X[start:start + chunk]
        r_cos = Xc[:, IDX_R_COS].view(-1, N_MODES_M, N_MODES_N)
        z_sin = Xc[:, IDX_Z_SIN].view(-1, N_MODES_M, N_MODES_N)
        nfp = Xc[:, IDX_NFP].view(-1, 1, 1, 1, 1)
        arg = m * theta - n * nfp * zeta
        R = torch.einsum("bmn,bmnuv->buv", r_cos, torch.cos(arg))
        Z = torch.einsum("bmn,bmnuv->buv", z_sin, torch.sin(arg))

        R_range = R.amax(dim=1) - R.amin(dim=1)  # (batch, Nv), per toroidal angle
        Z_range = Z.amax(dim=1) - Z.amin(dim=1)
        elong_proxy = Z_range / R_range.clamp_min(1e-6)
        R_mean_theta = R.mean(dim=1)
        Z_mean_theta = Z.mean(dim=1)

        feats = torch.stack([
            R_range.mean(1), R_range.amax(1), R_range.amin(1),
            Z_range.mean(1), Z_range.amax(1), Z_range.amin(1),
            elong_proxy.mean(1), elong_proxy.amax(1),
            R_mean_theta.mean(1), R_mean_theta.std(1),
            Z_mean_theta.std(1),
        ], dim=1)  # (batch, 11)
        feats_list.append(feats)
    return torch.cat(feats_list, dim=0)


def run_eval(model, loader, dev, n_targets, loss_fn):
    model.eval()
    loss_sum, batches = 0.0, 0
    metric_sum = torch.zeros(n_targets)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss, per_task_metric = loss_fn(pred, yb)
            loss_sum += loss.item()
            metric_sum += per_task_metric.cpu()
            batches += 1
    return loss_sum / batches, metric_sum / batches


def print_breakdown(label, mse, target_names):
    print(f"  per-target {label} RMSE (raw physical units):")
    for name, m in zip(target_names, mse):
        print(f"    {name:55s} {m.sqrt().item():12.5g}")


def print_contrastive_breakdown(label, per_task_nll, target_names):
    print(f"  per-target {label} class-balanced pairwise NLL (nats, Davidson tie model):")
    for name, v in zip(target_names, per_task_nll):
        print(f"    {name:55s} {v.item():12.5g}")


def print_contrastive_metrics(label, metrics, target_names):
    print(f"  per-target {label} contrastive metrics:")
    print(f"    {'target':55s} {'acc3':>8s} {'concord':>8s} {'tie_P':>8s} {'tie_R':>8s} {'tie_rate':>8s}")
    for k, name in enumerate(target_names):
        print(f"    {name:55s} {metrics['acc3'][k]:8.4f} {metrics['concordance'][k]:8.4f} "
              f"{metrics['tie_precision'][k]:8.4f} {metrics['tie_recall'][k]:8.4f} "
              f"{metrics['tie_rate_true'][k]:8.4f}")


def make_normalized_mse_loss(target_mean, target_std):
    """Plain unweighted MSE-sum loss in z-scored target space, for the
    --normalize-targets ablation -- no learned uncertainty weighting, since
    z-scoring already puts every target on a comparable scale by
    construction. Reported per-task metric is always un-normalized back to
    physical units first, so it stays comparable to every other RMSE number
    in EXPERIMENT_LOG regardless of what space the model was trained in.
    """
    def loss_fn(pred, target_norm):
        per_task_mse_norm = ((pred - target_norm) ** 2).mean(dim=0)
        loss = per_task_mse_norm.sum()
        pred_phys = pred * target_std + target_mean
        target_phys = target_norm * target_std + target_mean
        per_task_mse_report = ((pred_phys - target_phys) ** 2).mean(dim=0).detach()
        return loss, per_task_mse_report
    return loss_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "soap"],
                    help="adam: existing default. soap: Shampoo-preconditioned Adam "
                         "(https://arxiv.org/abs/2409.11321, scripts/soap.py) -- second-order-ish, "
                         "usually wants a higher lr than Adam (paper default 3e-3) and has real "
                         "per-step overhead from eigh/QR on each layer's preconditioner.")
    p.add_argument("--soap-weight-decay", type=float, default=0.01)
    p.add_argument("--soap-precondition-frequency", type=int, default=50,
                    help="how often (in steps) to refresh the preconditioner eigenbasis via QR. "
                         "Profiled: 10->50 alone is roughly a 1.6x step-time speedup at hidden=2048.")
    p.add_argument("--soap-max-precond-dim", type=int, default=1024,
                    help="skip preconditioning any parameter axis wider than this (falls back to "
                         "plain per-axis Adam-like scaling on that axis, no rotation). This is the "
                         "big lever: an axis this wide costs an O(dim^3) outer-product update EVERY "
                         "step, not just at each precondition_frequency refresh -- capping below the "
                         "current 2048-wide trunk layers cut per-step time roughly in half by itself.")
    p.add_argument("--soap-normalize-grads", action="store_true",
                    help="per SOAP's own docs, helps at large precondition_frequency (~100) but hurts "
                         "at small (~10) -- off by default since our default frequency (50) is in between "
                         "and untested either way.")
    p.add_argument("--soap-linalg-backend", default="default", choices=["default", "magma"],
                    help="torch.backends.cuda.preferred_linalg_library() for SOAP's eigh/QR calls. "
                         "'default' (cusolver) is faster but crashes (illegal CUDA memory access) on "
                         "~2048x2048+ matrices in this project's pytorch/cuda image -- only needed if "
                         "--soap-max-precond-dim is raised back up past ~1536 or so.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--spatial-latent", type=int, default=64)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--val-interval", type=int, default=5, help="epochs between validation passes")
    p.add_argument("--priority-target", default=None,
                    help="optional target name to upweight in the loss beyond the automatic "
                         "uncertainty weighting -- off by default, this skews the objective and "
                         "isn't a fair architecture comparison")
    p.add_argument("--priority-weight", type=float, default=1.0)
    p.add_argument("--no-spatial", action="store_true",
                    help="disable the spatial (IFFT + CNN) branch -- plain single-path MLP")
    p.add_argument("--trunk-arch", default="mlp", choices=["mlp", "siren", "half_siren", "attention"],
                    help="spectral-branch trunk architecture")
    p.add_argument("--trunk-blocks", type=int, default=3, help="number of chained blocks for half_siren")
    p.add_argument("--geom-features", action="store_true",
                    help="augment input with derived geometric summary stats of the reconstructed boundary")
    p.add_argument("--symlog-latent", action="store_true",
                    help="concat symlog(z) onto the latent before heads")
    p.add_argument("--log-targets", action="store_true",
                    help="predict log(target) for wide-dynamic-range targets (see LOG_TARGET_NAMES)")
    p.add_argument("--objective", default="regression", choices=["regression", "contrastive"],
                    help="regression: fit target values directly (default). contrastive: train the "
                         "same per-target heads as a pairwise bigger/smaller/basically-the-same "
                         "comparator via a Davidson tie model over in-batch pairs, instead of "
                         "matching absolute values. Forces single-path (--no-spatial) -- trunk-arch "
                         "comparison only, per current scope.")
    p.add_argument("--tie-weight-cap", type=float, default=5.0,
                    help="contrastive objective only: caps the effective majority:minority "
                         "per-class loss weight ratio at this value, per target, derived live from "
                         "each target's own in-batch class counts -- rather than one flat "
                         "reweighting strength for every target regardless of how rare its ties "
                         "actually are. See contrastive_loss docstring.")
    p.add_argument("--normalize-inputs", action="store_true",
                    help="standardize the 92 input features (per-feature mean/std from the TRAIN "
                         "split only) before feeding the trunk. Project default has been not to "
                         "(inputs already zero-mean, O(0.01-0.4), see metadata.json) -- this flag "
                         "exists to actually test that assumption rather than continue to assume it.")
    p.add_argument("--normalize-targets", action="store_true",
                    help="z-score targets (per-target mean/std from the TRAIN split only) and train "
                         "with a plain unweighted MSE sum instead of the learned uncertainty-weighted "
                         "loss -- tests the project's other standing assumption (raw physical units + "
                         "learned per-task weighting beats normalization). Regression objective only. "
                         "Reported RMSE is always un-normalized back to physical units for comparability "
                         "with every other number in EXPERIMENT_LOG.")
    p.add_argument("--tag", default="best", help="checkpoint filename stem, for running multiple experiments without clobbering each other")
    p.add_argument("--seed", type=int, default=None,
                    help="random seed for model init + data shuffling, for noise-floor / repeatability checks")
    p.add_argument("--split", default=None,
                    help="load train/val/test.npz from output/splits/<split>/ instead of the default "
                         "output/ location. 'random'/'group'/'cluster' come from scripts/make_splits.py; "
                         "any other directory under output/splits/ (e.g. a manually-augmented one) works too.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev = torch.device(args.device)
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())
    data_dir = (OUT_DIR / "splits" / args.split) if args.split else None

    X_train, Y_train = load_split("train", data_dir)
    X_val, Y_val = load_split("val", data_dir)

    # Project default has been no input/target normalization (see
    # EXPERIMENT_LOG methodology notes) -- stats computed from TRAIN only,
    # to avoid leaking val/test statistics into the transform.
    feature_mean = feature_std = None
    if args.normalize_inputs:
        feature_mean = X_train.mean(dim=0)
        feature_std = X_train.std(dim=0).clamp_min(1e-6)
        X_train = (X_train - feature_mean) / feature_std
        X_val = (X_val - feature_mean) / feature_std

    target_mean = target_std = None
    if args.normalize_targets:
        assert args.objective == "regression", "--normalize-targets only applies to the regression objective"
        target_mean = Y_train.mean(dim=0)
        target_std = Y_train.std(dim=0).clamp_min(1e-6)
        Y_train = (Y_train - target_mean) / target_std
        Y_val = (Y_val - target_mean) / target_std

    if args.geom_features:
        X_train = torch.cat([X_train, compute_geom_features(X_train)], dim=1)
        X_val = torch.cat([X_val, compute_geom_features(X_val)], dim=1)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch, shuffle=False)

    n_targets = Y_train.shape[1]
    priority_weight = torch.ones(n_targets)
    if args.priority_target:
        priority_weight[target_names.index(args.priority_target)] = args.priority_weight

    log_target_mask = torch.zeros(n_targets, dtype=torch.bool)
    if args.log_targets:
        for name in LOG_TARGET_NAMES:
            log_target_mask[target_names.index(name)] = True

    use_spatial = not args.no_spatial
    eps = None
    if args.objective == "contrastive":
        if use_spatial:
            print("[contrastive] forcing single-path (--no-spatial) -- trunk-arch comparison only")
            use_spatial = False
        eps = noise_floor_eps(target_names).to(dev)

    model = DualPathMLP(
        X_train.shape[1], n_targets,
        latent_dim=args.latent, hidden=args.hidden, spatial_latent=args.spatial_latent,
        head_hidden=args.head_hidden, priority_weight=priority_weight,
        use_spatial=use_spatial, trunk_arch=args.trunk_arch, trunk_blocks=args.trunk_blocks,
        use_symlog_latent=args.symlog_latent, log_target_mask=log_target_mask,
        objective=args.objective, noise_floor_eps=eps,
    ).to(dev)
    if args.optimizer == "soap":
        torch.backends.cuda.preferred_linalg_library(args.soap_linalg_backend)
        opt = SOAP(model.parameters(), lr=args.lr, weight_decay=args.soap_weight_decay,
                   precondition_frequency=args.soap_precondition_frequency,
                   max_precond_dim=args.soap_max_precond_dim,
                   normalize_grads=args.soap_normalize_grads)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.objective == "contrastive":
        loss_fn = lambda pred, yb: model.contrastive_loss(pred, yb, weight_cap=args.tie_weight_cap)
        breakdown_fn = print_contrastive_breakdown
    elif args.normalize_targets:
        loss_fn = make_normalized_mse_loss(target_mean.to(dev), target_std.to(dev))
        breakdown_fn = print_breakdown
    else:
        loss_fn = model.weighted_loss
        breakdown_fn = print_breakdown

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] architecture={'dual-path' if use_spatial else 'single-path'}  "
          f"objective={args.objective}  trunk={args.trunk_arch}  optimizer={args.optimizer}  lr={args.lr}  "
          f"in_dim={X_train.shape[1]}  params={n_params:,}  seed={args.seed}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss, _ = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

        is_val_epoch = epoch % args.val_interval == 0 or epoch == args.epochs
        if not is_val_epoch:
            print(f"epoch {epoch:3d}  train_nll {train_loss:9.4f}")
            continue

        val_loss, val_metric = run_eval(model, val_loader, dev, n_targets, loss_fn)
        summary = f"  mean_val_rmse {val_metric.sqrt().mean().item():.5f}" if args.objective == "regression" else ""
        print(f"epoch {epoch:3d}  train_nll {train_loss:9.4f}  val_nll {val_loss:9.4f}{summary}")
        breakdown_fn("val", val_metric, target_names)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_dim": X_train.shape[1],
                    "n_targets": n_targets,
                    "latent_dim": args.latent,
                    "hidden": args.hidden,
                    "spatial_latent": args.spatial_latent,
                    "head_hidden": args.head_hidden,
                    "priority_weight": priority_weight,
                    "use_spatial": use_spatial,
                    "trunk_arch": args.trunk_arch,
                    "trunk_blocks": args.trunk_blocks,
                    "geom_features": args.geom_features,
                    "use_symlog_latent": args.symlog_latent,
                    "log_target_mask": log_target_mask,
                    "objective": args.objective,
                    "split": args.split,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "target_names": target_names,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                CKPT_DIR / f"{args.tag}.pt",
            )

    train_seconds = time.perf_counter() - train_start
    ckpt_path = CKPT_DIR / f"{args.tag}.pt"
    print(f"training done. best val_loss {best_val:.4f}  params={n_params:,}  "
          f"train_time={train_seconds:.1f}s  checkpoint saved to {ckpt_path}")

    # Final test-set evaluation, using the best checkpoint (not necessarily
    # the last epoch's weights).
    ckpt = torch.load(ckpt_path, map_location=dev)
    test_eps = noise_floor_eps(ckpt["target_names"]).to(dev) if ckpt["objective"] == "contrastive" else None
    test_model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"],
        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"], spatial_latent=ckpt["spatial_latent"],
        head_hidden=ckpt["head_hidden"], priority_weight=ckpt["priority_weight"],
        use_spatial=ckpt["use_spatial"], trunk_arch=ckpt["trunk_arch"], trunk_blocks=ckpt["trunk_blocks"],
        use_symlog_latent=ckpt["use_symlog_latent"], log_target_mask=ckpt["log_target_mask"],
        objective=ckpt["objective"], noise_floor_eps=test_eps,
    ).to(dev)
    test_model.load_state_dict(ckpt["model_state_dict"])

    # Rebind loss_fn to test_model, not the training-loop `model` -- when the
    # best checkpoint isn't the final epoch (routine for contrastive runs,
    # e.g. best epoch 130/300 above), `model` in memory holds later-epoch
    # log_vars/log_nu than what's in ckpt, and weighted_loss/contrastive_loss
    # read those off `self`. Evaluating test_model's predictions through the
    # training model's stale loss params would silently corrupt exactly the
    # per-target numbers being reported here.
    if ckpt["objective"] == "contrastive":
        test_loss_fn = lambda pred, yb: test_model.contrastive_loss(pred, yb, weight_cap=args.tie_weight_cap)
    elif ckpt["target_mean"] is not None:
        test_loss_fn = make_normalized_mse_loss(ckpt["target_mean"].to(dev), ckpt["target_std"].to(dev))
    else:
        test_loss_fn = test_model.weighted_loss

    X_test, Y_test = load_split("test", data_dir)
    if ckpt["geom_features"]:
        X_test = torch.cat([X_test, compute_geom_features(X_test)], dim=1)
    X_test, Y_test = X_test.to(dev), Y_test.to(dev)
    if ckpt["feature_mean"] is not None:
        X_test = (X_test - ckpt["feature_mean"]) / ckpt["feature_std"]
    if ckpt["target_mean"] is not None:
        Y_test = (Y_test - ckpt["target_mean"]) / ckpt["target_std"]
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.batch, shuffle=False)
    test_loss, test_metric = run_eval(test_model, test_loader, dev, n_targets, test_loss_fn)

    print(f"\n=== TEST (checkpoint from epoch {ckpt['epoch']}) ===")
    if ckpt["objective"] == "contrastive":
        print(f"test_nll {test_loss:9.4f}")
        print_contrastive_breakdown("test", test_metric, target_names)
        with torch.no_grad():
            test_scores = test_model(X_test)
        full_metrics = contrastive_eval_metrics(test_scores, Y_test, test_eps, log_nu=test_model.log_nu)
        print_contrastive_metrics("test (full sweep)", full_metrics, target_names)
    else:
        print(f"test_nll {test_loss:9.4f}  mean_test_rmse {test_metric.sqrt().mean().item():.5f}")
        print_breakdown("test", test_metric, target_names)


if __name__ == "__main__":
    main()

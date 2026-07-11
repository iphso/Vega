import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
                 trunk_blocks=3, use_symlog_latent=False, log_target_mask=None):
        super().__init__()
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


def load_split(name):
    data = np.load(OUT_DIR / f"{name}.npz")
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


def run_eval(model, loader, dev, n_targets):
    model.eval()
    loss_sum, batches = 0.0, 0
    mse_sum = torch.zeros(n_targets)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss, per_task_mse = model.weighted_loss(pred, yb)
            loss_sum += loss.item()
            mse_sum += per_task_mse.cpu()
            batches += 1
    return loss_sum / batches, mse_sum / batches


def print_breakdown(label, mse, target_names):
    print(f"  per-target {label} RMSE (raw physical units):")
    for name, m in zip(target_names, mse):
        print(f"    {name:55s} {m.sqrt().item():12.5g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
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
    p.add_argument("--tag", default="best", help="checkpoint filename stem, for running multiple experiments without clobbering each other")
    p.add_argument("--seed", type=int, default=None,
                    help="random seed for model init + data shuffling, for noise-floor / repeatability checks")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev = torch.device(args.device)
    target_names = json.loads((OUT_DIR / "target_names.json").read_text())

    X_train, Y_train = load_split("train")
    X_val, Y_val = load_split("val")

    # No input normalization: the Fourier coefficients are already all
    # O(0.01-0.4) with zero mean (see metadata.json feature_stats) and
    # n_field_periods is O(1-5) -- nothing here needs rescaling, so we don't.

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

    model = DualPathMLP(
        X_train.shape[1], n_targets,
        latent_dim=args.latent, hidden=args.hidden, spatial_latent=args.spatial_latent,
        head_hidden=args.head_hidden, priority_weight=priority_weight,
        use_spatial=not args.no_spatial, trunk_arch=args.trunk_arch, trunk_blocks=args.trunk_blocks,
        use_symlog_latent=args.symlog_latent, log_target_mask=log_target_mask,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] architecture={'dual-path' if not args.no_spatial else 'single-path'}  "
          f"trunk={args.trunk_arch}  in_dim={X_train.shape[1]}  params={n_params:,}  seed={args.seed}")

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
            loss, _ = model.weighted_loss(pred, yb)
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

        val_loss, val_mse = run_eval(model, val_loader, dev, n_targets)
        mean_val_rmse = val_mse.sqrt().mean().item()
        print(f"epoch {epoch:3d}  train_nll {train_loss:9.4f}  val_nll {val_loss:9.4f}  "
              f"mean_val_rmse {mean_val_rmse:.5f}")
        print_breakdown("val", val_mse, target_names)

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
                    "use_spatial": not args.no_spatial,
                    "trunk_arch": args.trunk_arch,
                    "trunk_blocks": args.trunk_blocks,
                    "geom_features": args.geom_features,
                    "use_symlog_latent": args.symlog_latent,
                    "log_target_mask": log_target_mask,
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
    test_model = DualPathMLP(
        ckpt["in_dim"], ckpt["n_targets"],
        latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"], spatial_latent=ckpt["spatial_latent"],
        head_hidden=ckpt["head_hidden"], priority_weight=ckpt["priority_weight"],
        use_spatial=ckpt["use_spatial"], trunk_arch=ckpt["trunk_arch"], trunk_blocks=ckpt["trunk_blocks"],
        use_symlog_latent=ckpt["use_symlog_latent"], log_target_mask=ckpt["log_target_mask"],
    ).to(dev)
    test_model.load_state_dict(ckpt["model_state_dict"])

    X_test, Y_test = load_split("test")
    if ckpt["geom_features"]:
        X_test = torch.cat([X_test, compute_geom_features(X_test)], dim=1)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.batch, shuffle=False)
    test_loss, test_mse = run_eval(test_model, test_loader, dev, n_targets)

    print(f"\n=== TEST (checkpoint from epoch {ckpt['epoch']}) ===")
    print(f"test_nll {test_loss:9.4f}  mean_test_rmse {test_mse.sqrt().mean().item():.5f}")
    print_breakdown("test", test_mse, target_names)


if __name__ == "__main__":
    main()

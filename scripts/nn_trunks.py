"""Domain-agnostic regression-trunk architectures, extracted out of
train.py (VMEC++'s original architecture search, EXPERIMENT_LOG §2/§4) so a
second domain can run the same architecture comparison without copy-pasting
the trunk classes themselves. Only the genuinely domain-agnostic pieces move
here: SineLayer/SirenTrunk/HalfSirenBlock/HalfSirenTrunk take a plain
(in_dim, hidden, latent_dim) and know nothing about Fourier-mode layout or
any other domain-specific structure. `ModeAttentionEncoder` and the
spatial/IFFT branch in train.py's `DualPathMLP` stay put -- they're
genuinely VMEC-specific (they tokenize (m,n) Fourier modes / reconstruct a
boundary grid) and have no analogue in a domain like airfoils, whose whole
input already IS a flat parameter vector.

train.py imports its trunk classes from here (verified behavior-identical
before this replaced the inline copies -- see EXPERIMENT_LOG §32); any new
domain's own scoring-model script (e.g. train_airfoil_scoring.py) can build
directly on `build_trunk` instead of reinventing SIREN/half-SIREN from
scratch.
"""
import math

import torch
from torch import nn


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


def build_trunk(arch, in_dim, hidden, latent_dim, n_blocks=3):
    """arch in {"mlp", "siren", "half_siren"}. Returns a module in_dim -> latent_dim.
    ("attention" is intentionally not here -- it's VMEC-Fourier-mode-specific,
    see module docstring; train.py keeps its own ModeAttentionEncoder.)
    """
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
    raise ValueError(f"unknown trunk arch: {arch} (build_spectral_trunk in train.py also supports 'attention')")

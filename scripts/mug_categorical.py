"""Categorical<->continuous conversion for the mug domain's generative
models (EXPERIMENT_LOG §63).

The problem this fixes: `struct_material_idx`/`handle_material_idx` are a
genuinely categorical choice (which ONE named material, from
thermal_mug_spike.py's STRUCTURAL_MATERIALS/HANDLE_MATERIALS tables) but
were parameterized as one more continuous coordinate, the same way radius
or wall thickness is. Every consumer of that continuous value (the FD
physics's own linear interpolation, and -- the part that actually broke --
the generative models' plain MSE-regressed reconstruction) treated
"material_idx=6.795" as meaningful, when decoded it's actually a 79.5%
soda_lime_glass / 20.5% earthenware blend: a real material's own listed k/
rho/cp fed straight into the FD solver as if a mug could be cast from that
mixture. Confirmed live from the §62 checkpoint before this fix: decoded
cVAE samples included struct_material_idx=6.795 (-> "soda_lime_glass/
earthenware") and handle_material_idx=11.256 (-> "tpe_grip/stoneware"),
plus one sample (-0.729) outside the valid range entirely, silently
clamped rather than flagged.

The fix, precedented elsewhere in this project: VMEC++'s `nfp` (field-
period count) is exactly this same shape of problem -- a genuinely
discrete design choice -- and train_vae.py/train_gan.py/train_diffusion.py
already handle it via `nfp_one_hot()`, a one-hot CONDITIONING vector, never
folded into the continuously-reconstructed coefficient vector. Mug's own
case differs in one way that matters: nfp is only ever a CONDITION (pick a
field-period count, generate a design that matches it); a mug's material
choice is something we want the model to genuinely PROPOSE as part of the
design itself, not something fixed beforehand. So instead of one-hot
CONDITIONING, this gives each categorical column its own one-hot slot in
the model's OWN working vector, reconstructed via a real classification
head (cross-entropy for cVAE, softmax-before-critic for the GAN, plain
diffusion over the one-hot corners for the diffusion model -- see each
train_mug_*.py's own comments) instead of one scalar regressed by MSE.

Deliberately does NOT touch train_cvae.py/train_gan.py/train_diffusion.py
-- confirmed by reading them first that none of the three shared model
classes (CVAE/Generator+Critic/DiffusionDenoiser) compute their own loss;
every domain wrapper already does that itself. So this is domain-specific
plumbing consumed by train_mug_cvae.py/train_mug_gan.py/
train_mug_diffusion.py and, via gym_schema.py's `mug_spec()`, by
steerability_generic.py's generic decode step -- zero risk to VMEC++/
airfoil/TORAX, which never see this module.

`thermal_mug_spike.py`'s own continuous-index interpolation (`material_props()`)
is left in place, not removed -- it's genuinely useful as a smooth proxy
for one-off sensitivity sweeps, and is harmless once nothing in the real
dataset/generative pipeline ever calls it with a non-integer value again
(enforced downstream: generate_mug_dataset.py samples true integers,
`from_model_space()` below always returns a hard integer class).
"""
import numpy as np

import mug_oracle as oracle

# (name, column index in mug_oracle.PARAM_NAMES, number of classes)
CATEGORICAL_SPECS = [
    ("struct_material_idx", oracle.PARAM_NAMES.index("struct_material_idx"), len(oracle.STRUCTURAL_MATERIALS)),
    ("handle_material_idx", oracle.PARAM_NAMES.index("handle_material_idx"), len(oracle.HANDLE_MATERIALS)),
]
_CAT_COLUMNS = {col for _, col, _ in CATEGORICAL_SPECS}
CONTINUOUS_COLUMNS = [i for i in range(oracle.PARAM_DIM) if i not in _CAT_COLUMNS]
N_CONTINUOUS = len(CONTINUOUS_COLUMNS)
MODEL_DIM = N_CONTINUOUS + sum(n_classes for _, _, n_classes in CATEGORICAL_SPECS)


def to_model_space(X):
    """(n, PARAM_DIM) real design rows -> (n, MODEL_DIM) raw-continuous |
    one-hot | one-hot. Continuous columns are returned UN-standardized --
    callers z-score them exactly as before, just over CONTINUOUS_COLUMNS
    instead of the full width. Categorical columns MUST already hold exact
    integer class indices (generate_mug_dataset.py samples them that way
    as of §63); a fractional or out-of-range value here is a real upstream
    bug, not something to silently clamp, so it raises."""
    n = len(X)
    blocks = [X[:, CONTINUOUS_COLUMNS].astype(np.float32)]
    for name, col, n_classes in CATEGORICAL_SPECS:
        raw = X[:, col]
        idx = np.rint(raw).astype(np.int64)
        if not np.all(np.abs(raw - idx) < 1e-6) or not np.all((idx >= 0) & (idx < n_classes)):
            bad = raw[(np.abs(raw - idx) >= 1e-6) | (idx < 0) | (idx >= n_classes)]
            raise ValueError(f"{name}: expected exact integer classes in [0, {n_classes}), got e.g. {bad[:5]}")
        onehot = np.zeros((n, n_classes), dtype=np.float32)
        onehot[np.arange(n), idx] = 1.0
        blocks.append(onehot)
    return np.concatenate(blocks, axis=1)


def model_space_slices():
    """Column ranges within a MODEL_DIM-wide row: (continuous_slice,
    [(name, class_slice), ...]) -- shared by every train_mug_*.py that
    needs to pull the continuous block and each categorical block back out
    of a concatenated model-space tensor (real for training targets, or a
    model's own raw output for decoding/sampling)."""
    cont = slice(0, N_CONTINUOUS)
    cat_slices = []
    start = N_CONTINUOUS
    for name, _col, n_classes in CATEGORICAL_SPECS:
        cat_slices.append((name, slice(start, start + n_classes)))
        start += n_classes
    return cont, cat_slices


def from_model_space(decoded, cont_mean, cont_std):
    """(n, MODEL_DIM) raw model output (z-scored continuous | categorical
    logits | categorical logits) -> (n, PARAM_DIM) real design rows, in
    mug_oracle.PARAM_NAMES order -- the one place a generated design's
    material columns are forced back to a real, single, nameable material
    (hard argmax) instead of the old continuous blend. `cont_mean`/
    `cont_std` are the CONTINUOUS-columns-only stats saved in the
    checkpoint (length N_CONTINUOUS, not PARAM_DIM)."""
    decoded = np.asarray(decoded)
    cont_slice, cat_slices = model_space_slices()
    cont_real = decoded[:, cont_slice] * cont_std + cont_mean

    out = np.zeros((decoded.shape[0], oracle.PARAM_DIM), dtype=np.float32)
    out[:, CONTINUOUS_COLUMNS] = cont_real
    for (name, col, _n_classes), (_, cat_slice) in zip(CATEGORICAL_SPECS, cat_slices):
        out[:, col] = np.argmax(decoded[:, cat_slice], axis=1).astype(np.float32)
    return out


def soften_categorical_torch(x):
    """train_mug_gan.py only: softmax each categorical block of a torch
    tensor IN PLACE OF a hard one-hot, leaving the continuous block
    untouched. WGAN-GP's critic sees this (for the generator's fake
    samples) against real data's actual hard one-hot -- without it, the
    critic could trivially tell fake from real just by noticing "this
    block isn't in {0,1}," a shortcut that teaches the generator nothing
    about which material is realistic. Kept differentiable (plain softmax,
    not a hard argmax) since the WGAN-GP loss needs to backprop through
    it; the generator is pushed toward confident near-one-hot corners by
    the adversarial loss itself, and the actual discrete decision is only
    made later, at sampling time, by `from_model_space()`'s argmax."""
    cont_slice, cat_slices = model_space_slices()
    import torch
    parts = [x[:, cont_slice]]
    for _name, cat_slice in cat_slices:
        parts.append(torch.softmax(x[:, cat_slice], dim=-1))
    return torch.cat(parts, dim=-1)

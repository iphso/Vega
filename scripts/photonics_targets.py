"""Named TargetTasks for the photonics domain (gym_schema.TargetTask), same
role as p1_report.py's fixed target for VMEC++'s P1 problem -- a concrete,
reusable y_star a generative model or search method can be asked to hit,
grounded in this domain's own real 20,018-row v2 dataset
(generate_photonics_dataset.py, EXPERIMENT_LOG), not invented numbers.

v2: PARAM_DIM went 4->6 (apodization + variable buried-oxide thickness,
see photonics_oracle.py's own module docstring) and TARGET_NAMES gained
`fiber_coupling_efficiency` -- every target below rebuilt against the new
6-target order, not just resized. `tolerance` on each is a fraction of
that target's own achieving-row's local spread, same convention as v1.

Order matches photonics_oracle.TARGET_NAMES:
[up_efficiency, down_efficiency, transmitted_efficiency, reflected_efficiency,
fiber_coupling_efficiency, energy_closure].

  EASY_HIGH_FIBER_COUPLING: the best real fiber_coupling_efficiency found in
    the whole 20,018-row dataset (0.661, a UNIFORM -- not apodized --
    design: wg=0.306um, period=0.446um, duty~0.6/0.57, etch=0.179um,
    box=2.24um). NOT hard: the best apodized row (0.653) lands almost as
    high, so a method with reasonable coverage of the space should reach
    near this without much trouble. A real, checkable finding worth
    stating directly: at this dataset's scale, apodization has NOT yet
    beaten a well-chosen uniform design on raw fiber-coupling efficiency --
    included as the "easy" baseline the harder targets below compare
    against, not because it's the interesting result on its own.

  HARD_CLEAN_FIBER_DIRECTIVITY: high fiber_coupling_efficiency AND
    reflected_efficiency<0.03 AND down_efficiency<0.15 simultaneously --
    genuinely harder to satisfy than EASY above (1,864/20,018 rows, ~9.3%,
    clear the two loss-channel constraints; best fiber_coupling_efficiency
    among those is 0.531), but a MUCH SMALLER relative gap than v1's own
    equivalent up_efficiency version had (0.531/0.661=80% of the
    unconstrained best here, vs. v1's 0.127/0.774=16%) -- a real, measured
    finding: the extra apodization/box-thickness degrees of freedom didn't
    just add noise to this domain, they opened up genuinely better
    jointly-good regions of the design space that the older 4-param
    version couldn't reach.

  HARD_LOW_REFLECTION_NEAR_BRAGG: unchanged in spirit from v1 -- near-zero
    reflected_efficiency specifically in the near-Bragg period band
    (~0.25-0.33um), still confirmed as a real adversarial resonance in the
    v2 dataset (near_bragg_reflective's own 1,423 rows average
    reflected_efficiency=0.372, more than DOUBLE any other archetype's).

  HARD_BOX_THICKNESS_RESONANCE: a genuinely new v2 target, found by
    querying the dataset rather than assumed -- within a fixed design
    family (canonical_630nm's own perturbation cluster), the up/down
    directionality (up/(up+down)) vs. box_thickness_um relationship is a
    real, nonlinear, single-hump RESONANCE, not a monotonic trend: mean
    directionality rises from ~0.31 at box=1.0-1.3um to a peak of ~0.366
    around box=2.0-2.1um, then falls back to ~0.27-0.32 above box=2.9um.
    A naive linear correlation across the dataset's full sampled range
    reads as near-zero (r=-0.04) -- checked directly, and initially
    misread as "no real dependence" before this binned analysis caught
    the resonance underneath a coincidentally-flat linear trend. This
    target asks a method to land IN the resonance peak, not just anywhere
    in the box_thickness_um range -- a real test of whether a search
    method can find a narrow, nonlinear sweet spot a linear-correlation-
    based sensitivity analysis would miss entirely.
"""
import numpy as np

from gym_schema import TargetTask

TARGET_NAMES = ["up_efficiency", "down_efficiency", "transmitted_efficiency",
                 "reflected_efficiency", "fiber_coupling_efficiency", "energy_closure"]

EASY_HIGH_FIBER_COUPLING = TargetTask(
    y_star=np.array([0.741, 0.095, 0.124, 0.081, 0.661, 1.042]),
    tolerance=np.array([0.10, 0.10, 0.15, 0.10, 0.05, 0.05]),
)

HARD_CLEAN_FIBER_DIRECTIVITY = TargetTask(
    y_star=np.array([0.75, 0.10, 0.15, 0.02, 0.55, 1.00]),
    # Deliberately optimistic on fiber_coupling_efficiency -- the best real
    # row satisfying the constraint pair only reached 0.531, so 0.55 asks
    # a real method to beat this dataset's own best under constraint, not
    # just match it (same reasoning as v1's HARD_CLEAN_DIRECTIVITY).
    tolerance=np.array([0.15, 0.05, 0.20, 0.02, 0.08, 0.08]),
)

HARD_LOW_REFLECTION_NEAR_BRAGG = TargetTask(
    y_star=np.array([0.10, 0.20, 0.60, 0.02, 0.03, 1.00]),
    aux={"grating_period_um_range": (0.25, 0.33)},  # not consumed by photonics_oracle.py today -- documents the real constraint this target is only meaningful under, since PARAM_DIM has no aux slot for it (period is itself one of the 6 design dims)
    tolerance=np.array([0.08, 0.15, 0.20, 0.02, 0.03, 0.08]),
)

HARD_BOX_THICKNESS_RESONANCE = TargetTask(
    y_star=np.array([0.40, 0.55, 0.05, 0.05, 0.30, 1.00]),
    aux={"box_thickness_um_range": (1.95, 2.10)},  # the real measured resonance peak within the canonical_630nm family -- not consumed by photonics_oracle.py today, same "documents a constraint outside the param/aux slot" reasoning as HARD_LOW_REFLECTION_NEAR_BRAGG
    tolerance=np.array([0.08, 0.08, 0.10, 0.05, 0.08, 0.08]),
)

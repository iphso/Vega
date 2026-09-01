"""oracle_base.py implementation for the photonics domain -- a 2D silicon
grating coupler, the 6th real domain (gym_schema.py's own CANDIDATE_TRIAGE
table had flagged this as "Photonics/Meep... harness fit: UNKNOWN... per-
call cost profile is genuinely unconfirmed either way"; this module and its
FIDELITY_PRESETS entry in gym_schema.py's photonics_domain() are that
confirmation, measured the same way TORAX's was, not assumed).

Physics (real, standard silicon-photonics grating-coupler geometry, run
through actual MEEP FDTD -- not a self-written approximation, chosen
deliberately over a self-written EM solver given a real broadly-accepted
open solver exists, the same reasoning that moved mug off its own
self-written FD conduction and onto OpenFOAM): a 2D vertical cross-section
(x = propagation direction along the waveguide, y = vertical/thickness
axis) of a silicon-on-insulator waveguide (n_Si=3.45) on a SiO2 box oxide
(n_SiO2=1.44) under air cladding (n=1.0), with N_PERIODS=20 rectangular
grooves etched into the top of the waveguide over a GRATING region,
illuminated by an eigenmode source launched into the waveguide at
WAVELENGTH_UM=1.55 (telecom C band). This is the standard "effective
2D cross-section" grating-coupler simplification used throughout the
literature (ignores lateral/out-of-plane confinement -- a real 3D device
would need a much more expensive 3D run, deliberately out of scope; see
gym_schema.py's photonics_domain() docstring).

v2 (EXPERIMENT_LOG -- direct user request, after the v1 primer surfaced
"what would make this a more interesting optimization problem"): THREE
real physics additions over v1, not just parameter cosmetics --

  1. APODIZATION: `duty_cycle` (a single scalar) replaced by
     `duty_cycle_start`/`duty_cycle_end`, linearly tapered across the
     N_PERIODS teeth -- the standard apodization technique in the
     literature (matches Lomonte et al. 2021's own "linearly diminishing
     the FF from 85% at the beginning of the grating to 25% at the end"),
     not an invented scheme. A uniform grating (v1's only option) is still
     reachable as the degenerate case duty_cycle_start==duty_cycle_end.
  2. VARIABLE BOX THICKNESS: `box_thickness_um` (v1's BOX_THICKNESS_UM
     constant) promoted to a real design parameter. Real reason this
     matters, confirmed via literature while validating v1 (EXPERIMENT_LOG):
     the up/down radiated split depends sensitively on interference between
     light diffracted straight up and light diffracted down then reflected
     back up off the substrate -- published designs range from
     80%up/8.7%down to 1.7%up/86%down depending on exactly this dimension.
     v1 held it fixed at 2.0um and never explored it; a real, previously
     flagged gap, now closed.
  3. FIBER-MODE OVERLAP: a new `fiber_coupling_efficiency` target -- NOT a
     raw time-domain field, and NOT the same quantity as `up_efficiency`.
     `up_efficiency` is total power radiated upward in any direction; real
     fiber coupling additionally depends on how well the radiated beam's
     actual spatial/phase profile overlaps a single-mode fiber's Gaussian
     mode (SMF-28, MFD=10.4um at 1550nm) -- the exact caveat the v1 primer
     flagged ("up_efficiency is an upper bound on true fiber-coupling
     efficiency"). Computed via a DFT field line AT THE GRATING'S OWN
     APERTURE (`_mode_overlap_efficiency`, see its own docstring for the
     real debugging history: an EARLIER version tried MEEP's near2far
     transform evaluated at a physical fiber standoff distance, which
     turned out to sit in the Fresnel near-field regime for this aperture
     size and collapsed nonsensically for apodized designs -- reverted in
     favor of the aperture-plane evaluation below, which is also what
     apodization is actually designed toward in the literature). The
     fiber is assumed OPTIMALLY ALIGNED to each design's own actual
     radiated beam (centroid position + propagation tilt measured directly
     from the simulated near-field data itself, not a fixed universal
     angle or an approximate analytic Bragg-angle formula) -- matching how
     real experiments align a fiber empirically to a device's own measured
     output, and avoiding compounding one approximation (analytic angle
     formula) on top of another.

Design vector x (PARAM_DIM=6): [wg_thickness_um, grating_period_um,
duty_cycle_start, duty_cycle_end, etch_depth_um, box_thickness_um].
N_PERIODS and the source wavelength remain fixed constants -- kept out of
the parameterization the same way mug_oracle.py's v6 kept its material
tables as a small fixed enum rather than continuous, to keep the space
tractable while still real.

Targets (6): up_efficiency / down_efficiency / transmitted_efficiency /
reflected_efficiency (all net power crossing a monitor line, normalized to
the launched input power measured in a SEPARATE calibration run of the
straight, un-etched waveguide -- MEEP's own standard normalization
technique, since a single run can't cleanly separate "forward launched"
from "reflected" power at one monitor without a clean reference) /
fiber_coupling_efficiency (see point 3 above -- always <= up_efficiency,
itself a real internal consistency check, not just a modeling choice)
plus energy_closure = the sum of the first four (excluding
fiber_coupling_efficiency, which is a further subdivision of
up_efficiency, not a separate energy channel), a real physical diagnostic
that should land near 1.0 if the FDTD run actually conserved energy.

Validity: `ok=False` for a structurally invalid input (non-positive
thickness/period/etch depth/box thickness, either duty cycle outside
(0, 1), or etch_depth exceeding wg_thickness -- you cannot etch deeper
than the layer is thick). No FDTD non-convergence mode exists the way
VMEC++/airfoil have one -- a fixed simulated-time run always completes --
so this is the same character of validity as mug_oracle.py's (`ok` only
ever False on invalid input, not a solver failure), not VMEC++'s.
"""
import numpy as np

PARAM_DIM = 6
PARAM_NAMES = ["wg_thickness_um", "grating_period_um", "duty_cycle_start", "duty_cycle_end",
                "etch_depth_um", "box_thickness_um"]
ZERO_INDICES = []
TARGET_NAMES = ["up_efficiency", "down_efficiency", "transmitted_efficiency",
                 "reflected_efficiency", "fiber_coupling_efficiency", "energy_closure"]
LOG_TARGET_NAMES = []

# Fixed constants, not design variables (see module docstring).
WAVELENGTH_UM = 1.55
N_PERIODS = 20
N_SI = 3.45
N_SIO2 = 1.44
N_AIR = 1.0
PML_UM = 1.0
DPML_SRC_GAP_UM = 0.5        # source sits this far inside the left PML
CALIB_TAIL_UM = 1.0          # extra straight waveguide after the grating region

# Fiber-mode-overlap constant (see module docstring, point 3, and
# _mode_overlap_efficiency's own docstring for why this is evaluated at
# the grating aperture, not a physical fiber standoff distance). SMF-28
# is the standard single-mode telecom fiber; its MFD at 1550nm is a real,
# widely-cited datasheet value, not fitted.
FIBER_MFD_UM = 10.4

# Resolution (pixels/um) is the one thing that varies by fidelity tier --
# the actual per-call-cost lever confirmed by measurement (see
# gym_schema.py's photonics_domain() for the real numbers), the same "one
# knob, several cost tiers" shape as vmec_oracle.py's FIDELITY_PRESETS.
FIDELITY_PRESETS = {"low": 10, "medium": 20}


def _validate(wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end, etch_depth_um, box_thickness_um):
    vals = (wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end, etch_depth_um, box_thickness_um)
    if not all(np.isfinite(v) for v in vals):
        return False, "non-finite input"
    if wg_thickness_um <= 0:
        return False, "wg_thickness_um must be positive"
    if grating_period_um <= 0:
        return False, "grating_period_um must be positive"
    if not (0.0 < duty_cycle_start < 1.0):
        return False, "duty_cycle_start must be in (0, 1)"
    if not (0.0 < duty_cycle_end < 1.0):
        return False, "duty_cycle_end must be in (0, 1)"
    if etch_depth_um <= 0:
        return False, "etch_depth_um must be positive"
    if etch_depth_um > wg_thickness_um:
        return False, "etch_depth_um cannot exceed wg_thickness_um"
    if box_thickness_um <= 0:
        return False, "box_thickness_um must be positive"
    return True, None


def _tooth_widths(grating_period_um, duty_cycle_start, duty_cycle_end):
    """Linear apodization taper across N_PERIODS teeth -- duty_cycle_start
    at tooth 0, duty_cycle_end at tooth N_PERIODS-1, matching the standard
    literature convention (see module docstring point 1). Degenerates to a
    uniform grating (v1's only option) when duty_cycle_start==duty_cycle_end."""
    if N_PERIODS == 1:
        fracs = np.array([duty_cycle_start])
    else:
        fracs = np.linspace(duty_cycle_start, duty_cycle_end, N_PERIODS)
    return fracs * grating_period_um


def _build_geometry(mp, wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end, etch_depth_um,
                     sx, sy, wg_y_center, with_grating, grating_x0, grating_x1):
    """Shared cell geometry for both the calibration (with_grating=False,
    plain straight waveguide) and real (with_grating=True) runs -- kept as
    one function so the two runs are guaranteed to differ ONLY in the
    grating teeth, not in any other accidental geometry drift between them."""
    si = mp.Medium(index=N_SI)
    sio2 = mp.Medium(index=N_SIO2)

    geometry = [
        # Full-height background block below the waveguide = box oxide,
        # everything above = air cladding (the air cladding fill is
        # implicit -- MEEP's default_material below).
        mp.Block(size=mp.Vector3(sx, sy, mp.inf), center=mp.Vector3(0, -sy / 2, 0), material=sio2),
        # Waveguide core, full simulation length.
        mp.Block(size=mp.Vector3(sx, wg_thickness_um, mp.inf),
                  center=mp.Vector3(0, wg_y_center, 0), material=si),
    ]
    if with_grating:
        # Etch N_PERIODS rectangular grooves (air) into the TOP of the
        # core over [grating_x0, grating_x1) -- each tooth's own width is
        # duty_cycle(i) * grating_period_um, apodized per _tooth_widths.
        tooth_widths = _tooth_widths(grating_period_um, duty_cycle_start, duty_cycle_end)
        etch_y_center = wg_y_center + wg_thickness_um / 2 - etch_depth_um / 2
        for i, tooth_w in enumerate(tooth_widths):
            groove_x0 = grating_x0 + i * grating_period_um + tooth_w
            groove_x1 = grating_x0 + (i + 1) * grating_period_um
            w = groove_x1 - groove_x0
            if w <= 1e-6:
                continue
            geometry.append(mp.Block(
                size=mp.Vector3(w, etch_depth_um, mp.inf),
                center=mp.Vector3(groove_x0 + w / 2, etch_y_center, 0),
                material=mp.air,
            ))
    return geometry


def _mode_overlap_efficiency(ez, xs, k0):
    """Real fiber-mode-overlap calculation (module docstring, point 3).
    Takes the near field DIRECTLY AT THE GRATING'S OWN APERTURE (not
    propagated to a standoff distance -- see the real debugging history
    below), finds the beam's own actual centroid position and local
    propagation tilt from the simulated data itself (not an analytic
    Bragg-angle formula), builds an ideal SMF-28 Gaussian mode matched to
    that same centroid/tilt, and returns the normalized overlap fraction
    in [0, 1] -- dimensionless, multiplied against up_efficiency (a real
    physical power fraction) by the caller.

    ORIGINALLY implemented via MEEP's near2far transform, evaluated at a
    real fiber standoff distance (Z_FIBER_STANDOFF_UM=15um) -- REVERTED
    after a direct check: at 15um from a ~12.6um-wide aperture, the beam
    is nowhere near the far field (Fraunhofer distance ~aperture^2/lambda
    ~100um), so it still carries real wavefront curvature a flat-tilted-
    plane-wave model can't represent. Confirmed as the actual bug, not
    guessed: an apodized design (which SHOULD improve fiber coupling per
    the literature) instead collapsed the standoff-based overlap by >30x
    versus a uniform design (0.017 vs 0.46) -- physically implausible.
    Evaluating at the aperture plane instead (same principle apodization
    design in the literature actually optimizes toward -- Lomonte et al.
    2021's own "mode profile AT ITS BEAM WAIST") gave both designs
    plausible, comparable overlap fractions (uniform 0.75, apodized 0.69)
    -- kept as the real implementation, and cheaper too (reuses the same
    DFT field capture technique as the viewer's heatmap, no separate
    near2far monitor needed)."""
    power = np.abs(ez) ** 2
    total_power = power.sum()
    if total_power <= 0 or not np.isfinite(total_power):
        return 0.0

    x0 = float((xs * power).sum() / total_power)  # beam centroid -- where the fiber should be centered
    phase = np.unwrap(np.angle(ez))
    weights = power / power.max()
    kx = float(np.polyfit(xs, phase, 1, w=weights)[0])  # beam's own measured propagation tilt
    kx = np.clip(kx, -k0, k0)  # a physical transverse wavevector can't exceed k0

    w0 = FIBER_MFD_UM / 2.0
    fiber_mode = np.exp(-((xs - x0) ** 2) / w0 ** 2) * np.exp(1j * kx * (xs - x0))

    overlap = np.abs(np.sum(ez * np.conj(fiber_mode))) ** 2
    norm = np.sum(np.abs(ez) ** 2) * np.sum(np.abs(fiber_mode) ** 2)
    if norm <= 0:
        return 0.0
    return float(np.clip(overlap / norm, 0.0, 1.0))


def _run_sim(mp, resolution, wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end,
             etch_depth_um, box_thickness_um, with_grating, capture_field=False, capture_fiber_overlap=False):
    fcen = 1.0 / WAVELENGTH_UM
    df = 0.2 * fcen  # broad enough pulse bandwidth for a clean single-frequency readout at fcen

    grating_len = N_PERIODS * grating_period_um
    sx = 2 * PML_UM + 2 * DPML_SRC_GAP_UM + grating_len + 2 * CALIB_TAIL_UM
    sy = 2 * PML_UM + 2 * box_thickness_um  # symmetric margin above/below for radiated-field monitors
    wg_y_center = 0.0

    x0 = -sx / 2
    src_x = x0 + PML_UM + DPML_SRC_GAP_UM
    grating_x0 = src_x + CALIB_TAIL_UM
    grating_x1 = grating_x0 + grating_len
    in_mon_x = src_x + 0.2 * CALIB_TAIL_UM  # just right of the source, before the grating
    out_mon_x = grating_x1 + 0.5 * CALIB_TAIL_UM

    geometry = _build_geometry(mp, wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end,
                                 etch_depth_um, sx, sy, wg_y_center, with_grating, grating_x0, grating_x1)

    sources = [mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(src_x, wg_y_center, 0),
        size=mp.Vector3(0, sy - 2 * PML_UM, 0),
        eig_band=1, eig_match_freq=True, eig_parity=mp.NO_PARITY,
    )]

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, 0),
        resolution=resolution,
        boundary_layers=[mp.PML(PML_UM)],
        geometry=geometry,
        sources=sources,
        default_material=mp.air,
        dimensions=2,
    )

    in_mon = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(in_mon_x, wg_y_center, 0),
                                                       size=mp.Vector3(0, sy - 2 * PML_UM, 0)))
    out_mon = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(out_mon_x, wg_y_center, 0),
                                                        size=mp.Vector3(0, sy - 2 * PML_UM, 0)))
    up_y = wg_y_center + wg_thickness_um / 2 + 0.5 * box_thickness_um
    down_y = wg_y_center - wg_thickness_um / 2 - 0.5 * box_thickness_um
    up_mon = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(0, up_y, 0),
                                                       size=mp.Vector3(grating_len, 0, 0)))
    down_mon = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(0, down_y, 0),
                                                         size=mp.Vector3(grating_len, 0, 0)))

    # 1D DFT field line AT THE GRATING'S OWN APERTURE (y=up_y, no
    # propagation) for the fiber-mode-overlap calculation (point 3, module
    # docstring) -- see _mode_overlap_efficiency's own docstring for why
    # this replaced an earlier near2far-at-a-standoff-distance approach
    # (that one evaluated the beam in the Fresnel/near-field regime where
    # a flat-tilted-plane-wave model is wrong, confirmed by an apodized
    # design collapsing to a nonsensical near-zero overlap). Cheap: same
    # DFT machinery as the viewer's own field heatmap, just a 1D line
    # instead of the full 2D grid. The dataset-generation harness pays for
    # this on every candidate since fiber_coupling_efficiency is a real
    # dataset target, unlike capture_field (viewer-only).
    aperture_dft = None
    if capture_fiber_overlap:
        aperture_window = grating_len + 10.0  # margin beyond the grating itself, catches the beam's own natural spread
        aperture_dft = sim.add_dft_fields([mp.Ez], fcen, 0, 1,
                                            where=mp.Volume(center=mp.Vector3(0, up_y, 0),
                                                             size=mp.Vector3(aperture_window, 0, 0)))

    # Frequency-domain (DFT) field monitor, accumulated over the whole run
    # at the SAME fcen the flux monitors use -- the physically correct way
    # to get a meaningful steady-state field pattern out of a pulsed
    # source (a raw time-domain Ez snapshot after the pulse has decayed
    # would show close to nothing; this integrates the field's own
    # fcen-frequency component throughout the run instead, same DFT
    # machinery the flux monitors already use). Only for the viewer's
    # field-visualization request (photonics_physics_service.py) -- the
    # dataset-generation harness (worker_fn, capture_field=False always)
    # never pays this extra memory/compute cost.
    dft_fields = None
    if capture_field:
        field_region = mp.Volume(center=mp.Vector3(0, wg_y_center, 0),
                                  size=mp.Vector3(sx - 2 * PML_UM, sy - 2 * PML_UM, 0))
        dft_fields = sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=field_region)

    decay_pt = mp.Vector3(out_mon_x, wg_y_center, 0)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, decay_pt, 1e-4))

    fluxes = (mp.get_fluxes(in_mon)[0], mp.get_fluxes(out_mon)[0],
              mp.get_fluxes(up_mon)[0], mp.get_fluxes(down_mon)[0])

    overlap_frac = None
    if capture_fiber_overlap:
        k0 = 2 * np.pi * fcen
        ez_ap = sim.get_dft_array(aperture_dft, mp.Ez, 0).reshape(-1)
        xs_ap = np.linspace(-aperture_window / 2, aperture_window / 2, len(ez_ap))
        overlap_frac = _mode_overlap_efficiency(ez_ap, xs_ap, k0)

    field = None
    if capture_field:
        ez = sim.get_dft_array(dft_fields, mp.Ez, 0)  # complex, shape (nx, ny)
        intensity = (np.abs(ez) ** 2).astype(np.float64)
        field = {
            "intensity": intensity.tolist(),  # [x][y], real >=0 -- |Ez|^2 at fcen
            "extent_um": [-(sx - 2 * PML_UM) / 2, (sx - 2 * PML_UM) / 2,
                           -(sy - 2 * PML_UM) / 2, (sy - 2 * PML_UM) / 2],  # [xmin, xmax, ymin, ymax], relative to the waveguide's own vertical center
            "box_thickness_um": box_thickness_um,
        }

    return fluxes, field, overlap_frac


def compute_efficiencies(wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end, etch_depth_um,
                          box_thickness_um, resolution, capture_field=False):
    """Plain (ok, payload_or_error) computation, no subprocess/Pipe wrapping
    -- the shared core `worker_fn` (subprocess harness path) and
    photonics_physics_service.py (in-process viewer path, same process
    already has `meep` imported once at startup) both call this directly,
    so the actual physics/validation logic exists in exactly one place.
    fiber_coupling_efficiency (point 3, module docstring) is ALWAYS
    computed -- it's a real dataset target, not viewer-only, unlike
    capture_field (the visualization heatmap, which payload carries under
    "field" only when explicitly requested)."""
    ok, err = _validate(wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end,
                         etch_depth_um, box_thickness_um)
    if not ok:
        return False, f"invalid input: {err}"
    try:
        import meep as mp
    except Exception as e:
        return False, f"meep import error: {e}"
    try:
        (p_launched, _, _, _), _, _ = _run_sim(mp, resolution, wg_thickness_um, grating_period_um,
                                                 duty_cycle_start, duty_cycle_end, etch_depth_um,
                                                 box_thickness_um, with_grating=False)
        (p_in, p_out, p_up, p_down), field, overlap_frac = _run_sim(
            mp, resolution, wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end,
            etch_depth_um, box_thickness_um, with_grating=True,
            capture_field=capture_field, capture_fiber_overlap=True)
    except Exception as e:
        return False, f"meep sim error: {e}"
    if p_launched <= 0 or not np.isfinite(p_launched):
        return False, f"invalid calibration launch power: {p_launched}"
    # Net rightward flux at in_mon, with the grating present, is
    # p_launched MINUS whatever reflected back leftward through that
    # same point (see module docstring on why this needs the separate
    # calibration run rather than one monitor alone).
    reflected = max(0.0, (p_launched - p_in) / p_launched)
    up_eff = max(0.0, p_up / p_launched)
    down_eff = max(0.0, abs(p_down) / p_launched)  # down-flux monitor's own +y convention reads negative for downward power
    trans_eff = max(0.0, p_out / p_launched)
    # fiber_coupling_efficiency = (real power fraction going up) x (dimensionless
    # mode-shape match) -- decomposition matches Lomonte et al. 2021's own
    # "product of the normalized light upward-emitted... and the overlap
    # integral" (EXPERIMENT_LOG). By construction this is <= up_eff, a real
    # internal consistency check, not just a modeling convenience.
    fiber_eff = up_eff * (overlap_frac if overlap_frac is not None else 0.0)
    payload = {
        "up_efficiency": float(up_eff),
        "down_efficiency": float(down_eff),
        "transmitted_efficiency": float(trans_eff),
        "reflected_efficiency": float(reflected),
        "fiber_coupling_efficiency": float(fiber_eff),
        "energy_closure": float(up_eff + down_eff + trans_eff + reflected),
    }
    if capture_field:
        payload["field"] = field
    return True, payload


def worker_fn(conn, wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end, etch_depth_um,
              box_thickness_um, resolution):
    try:
        ok, payload = compute_efficiencies(wg_thickness_um, grating_period_um, duty_cycle_start, duty_cycle_end,
                                             etch_depth_um, box_thickness_um, resolution)
        conn.send((ok, payload))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    """params: (PARAM_DIM,) -- see this module's docstring for the ordering.
    `aux` is unused (empty dict, same as mug/torax -- no discrete/continuous
    conditioning variable in this parameterization, see gym_schema.py's
    get_conditioning for the photonics entry)."""
    resolution = FIDELITY_PRESETS[fidelity_name]
    return tuple(float(v) for v in params) + (resolution,)

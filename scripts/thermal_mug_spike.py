"""
Feasibility spike for a 5th Optimization Gym domain: transient heat conduction
through an insulated mug/thermos wall.

Not GPU, not a wrapped external simulator -- self-written 1D (thin-wall/planar
approximation) transient conduction, implicit (backward Euler) finite-volume,
liquid modeled as a lumped-capacitance node, ambient as a fixed-temperature
boundary. No license question: the physics model is authored here directly,
same category as VMEC's stiffness-method truss idea would be (nothing wrapped).

Design vector x = [t_wall_mm, t_gap_mm, k_gap]:
  - t_wall_mm: structural shell thickness (steel), mm
  - t_gap_mm: insulation layer thickness, mm
  - k_gap: insulation material conductivity, W/(m K) -- continuous stand-in
    for material choice (vacuum-like ~0.005 .. foam/air ~0.03 .. poor ~0.05)

Targets y:
  - retention_time_s: time for liquid to cool from 90C to 60C (None if not
    reached within t_max -- flagged, not silently clipped)
  - mass_kg: wall+insulation material mass (thin-wall volume x density),
    a cost/weight proxy -- the real design tension against retention_time
  - touch_temp_60s_C: outer wall surface temperature 60s after filling --
    a safety-relevant, genuinely time-lagged quantity distinct from the
    steady-state retention behavior (checked explicitly below, not assumed)
"""
import math
import numpy as np
from scipy.linalg import solve_banded

# --- physical constants (order-of-magnitude real values, not tuned) ---
CP_LIQUID = 4186.0        # J/(kg K), water/coffee
RHO_LIQUID = 1000.0       # kg/m3
K_STEEL = 15.0            # W/(m K), stainless steel
RHO_STEEL = 8000.0        # kg/m3
CP_STEEL = 500.0          # J/(kg K)
RHO_INSULATION = 40.0     # kg/m3, foam/aerogel-like
CP_INSULATION = 1400.0    # J/(kg K)

H_LIQ = 300.0             # W/(m2 K), liquid-side natural convection
H_AIR = 10.0              # W/(m2 K), ambient-side natural convection
T_AMB = 20.0              # deg C
T0_LIQUID = 90.0          # deg C, fill temperature
T_THRESHOLD = 60.0        # deg C, "still hot enough" cutoff

R_INNER = 0.035           # m, mug cavity radius
HEIGHT = 0.10             # m
A_INNER = 2 * math.pi * R_INNER * HEIGHT   # thin-wall approx: use inner area throughout
LIQ_VOLUME = math.pi * R_INNER**2 * HEIGHT
C_LIQ = RHO_LIQUID * LIQ_VOLUME * CP_LIQUID

N_WALL = 20
N_GAP = 40  # gap ranges wider (0.5-20mm+) than wall -- needs more nodes for the
            # early-time (60s) touch-temp transient to be resolution-converged;
            # confirmed by direct N_GAP sweep (15/40/100 -> 43.4/40.8/39.8 C at
            # t_gap=20mm), N_GAP=15 was under-resolved enough to produce a
            # non-physical non-monotonic touch_temp_60s vs. t_gap

PARAM_DIM = 3
PARAM_NAMES = ["t_wall_mm", "t_gap_mm", "k_gap"]


def simulate(t_wall_mm, t_gap_mm, k_gap, dt=5.0, t_max=6 * 3600.0,
             wall_mass_scale=1.0, record_touch_at=60.0):
    if t_wall_mm <= 0 or t_gap_mm <= 0 or k_gap <= 0:
        return dict(valid=False, retention_time_s=None, mass_kg=None, touch_temp_60s_C=None)

    t_wall = t_wall_mm / 1000.0
    t_gap = t_gap_mm / 1000.0
    n = N_WALL + N_GAP
    dx_wall = t_wall / N_WALL
    dx_gap = t_gap / N_GAP
    dx = np.concatenate([np.full(N_WALL, dx_wall), np.full(N_GAP, dx_gap)])
    k = np.concatenate([np.full(N_WALL, K_STEEL), np.full(N_GAP, k_gap)])
    rho = np.concatenate([np.full(N_WALL, RHO_STEEL), np.full(N_GAP, RHO_INSULATION)])
    cp = np.concatenate([np.full(N_WALL, CP_STEEL), np.full(N_GAP, CP_INSULATION)])
    C_cells = rho * cp * dx * A_INNER * wall_mass_scale

    G_cells = np.empty(n - 1)
    for i in range(n - 1):
        R = dx[i] / (2 * k[i] * A_INNER) + dx[i + 1] / (2 * k[i + 1] * A_INNER)
        G_cells[i] = 1.0 / R

    G_liq0 = 1.0 / (1.0 / (H_LIQ * A_INNER) + dx[0] / (2 * K_STEEL * A_INNER))
    G_air_last = 1.0 / (1.0 / (H_AIR * A_INNER) + dx[-1] / (2 * k[-1] * A_INNER))

    ndim = n + 1  # 0 = liquid, 1..n = wall/gap cells
    Ccap = np.concatenate([[C_LIQ], C_cells])

    diag = np.zeros(ndim)
    lower = np.zeros(ndim)
    upper = np.zeros(ndim)

    diag[0] = Ccap[0] / dt + G_liq0
    upper[0] = -G_liq0

    for i in range(1, ndim):
        cell = i - 1
        Gl = G_liq0 if cell == 0 else G_cells[cell - 1]
        Gr = G_air_last if cell == n - 1 else G_cells[cell]
        diag[i] = Ccap[i] / dt + Gl + Gr
        lower[i] = -Gl
        if cell != n - 1:
            upper[i] = -Gr

    ab = np.zeros((3, ndim))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]

    forcing_air = np.zeros(ndim)
    forcing_air[-1] = G_air_last * T_AMB

    T = np.full(ndim, T0_LIQUID)  # mug starts uniformly at fill temp

    n_steps = int(t_max / dt)
    retention_time = None
    touch_temp_60s = None
    touch_recorded = False

    for step in range(n_steps):
        t_now = step * dt
        rhs = Ccap / dt * T + forcing_air
        T = solve_banded((1, 1), ab, rhs)
        if not np.all(np.isfinite(T)):
            return dict(valid=False, retention_time_s=None, mass_kg=None, touch_temp_60s_C=None)
        if not touch_recorded and (t_now + dt) >= record_touch_at:
            touch_temp_60s = float(T[-1])
            touch_recorded = True
        if retention_time is None and T[0] <= T_THRESHOLD:
            retention_time = t_now + dt

    mass_kg = A_INNER * (t_wall * RHO_STEEL + t_gap * RHO_INSULATION)
    return dict(
        valid=True,
        retention_time_s=retention_time,
        mass_kg=mass_kg,
        touch_temp_60s_C=touch_temp_60s,
        final_liq_temp_C=float(T[0]),
    )


def analytic_time_constant_s(t_wall_mm, t_gap_mm, k_gap):
    """Massless-wall lumped-RC estimate, for verifying the FD implementation
    against a closed-form limit (§21/§45's own 'check against a known case'
    convention) -- NOT the model's actual prediction, which includes real
    wall/gap thermal mass."""
    t_wall = t_wall_mm / 1000.0
    t_gap = t_gap_mm / 1000.0
    R_liq = 1.0 / (H_LIQ * A_INNER)
    R_wall = t_wall / (K_STEEL * A_INNER)
    R_gap = t_gap / (k_gap * A_INNER)
    R_air = 1.0 / (H_AIR * A_INNER)
    tau = (R_liq + R_wall + R_gap + R_air) * C_LIQ
    return -tau * math.log((T_THRESHOLD - T_AMB) / (T0_LIQUID - T_AMB))


# ---------------------------------------------------------------------------
# v2 (EXPERIMENT_LOG §48): user asked for a richer parameterization -- named
# materials instead of one free-floating conductivity, a wall PROFILE instead
# of a constant thickness, and a real handle (weight + a "does the handle get
# hot" safety target). v1's `simulate()` is left untouched above for
# reference/regression-checking; v2 is a genuinely different design vector
# and gets its own function.
#
# Wall profile: kept to 2 axial bands (rim, base) rather than a fully
# height-resolved field -- a real generalization beyond v1's constant
# thickness (mugs often ARE thinner at the rim for a comfortable drinking
# edge and thicker at the base for stability/drop resistance) without a full
# 2D solve. Each band gets an equal half-share of the cylindrical side-wall
# area and is its own independent 1D radial+gap conduction chain, both
# drawing from the same liquid.
#
# Coupling scheme: v1 solved liquid+wall as ONE fully-implicit system (liquid
# is an unknown in the same tridiagonal solve). With 3 independent chains now
# (rim, base, handle) all touching the liquid/body, that would require a
# star-shaped (non-tridiagonal) matrix. Used explicit-implicit splitting
# instead: the liquid's own energy balance is updated EXPLICITLY each step
# from the previous step's band fluxes, and each chain (rim, base, handle) is
# solved fully implicitly using the previous step's liquid/body temperature
# as its own boundary value. Verified below that this still matches v1
# closely in the reducible case (same materials/thickness on both bands, no
# handle contribution to the liquid) -- not assumed equivalent.
#
# Handle: a genuine transient fin-conduction problem, not a quasi-steady
# shortcut -- checked first and rejected: a quasi-steady estimate (using the
# steady-state fin formula at each instant) implicitly assumes the handle's
# own thermal diffusion time is short vs. the 60s reporting window. Direct
# check: tau ~ L^2/alpha for a ~5cm steel handle is ~667s, for a low-k
# handle (precisely the "safe" designs we care about) it's even less settled
# within 60s -- neither case clears the 60s window, so the transient must be
# solved for real. Modeled as a 1D fin: implicit backward-Euler chain along
# the handle's length, WITH a lateral convective-loss term at every cell
# (h_air * perimeter * dx * (T-T_amb), the standard extended-surface/fin
# term v1's wall chains didn't need since they only lose heat at their two
# ends), Dirichlet-coupled at the base to the (previous-step) average of the
# two wall bands' outer surface temperatures, adiabatic (zero-flux) tip --
# the standard assumption for a fin whose tip area is small vs. its lateral
# area. Starts at ambient temperature, not fill temperature -- unlike the
# wall (which is assumed to instantly wet-contact the liquid at t=0), the
# handle is physically remote and only heats via conduction from the body.
# ---------------------------------------------------------------------------

# (name, k [W/(m K)], rho [kg/m3], cp [J/(kg K)]) -- sorted by k ascending in
# each table so a continuous index interpolates through real named materials,
# not just an arbitrary number range.
#
# EXPERIMENT_LOG §62: expanded from 4 generic category buckets (plastic/
# glass/ceramic/steel) each, at direct user request for something more
# specific and realistic than one representative value per broad category.
# Every row below is now a SPECIFIC named material (polypropylene, not
# "plastic"; borosilicate glass, not "glass"; 6061 aluminum, not a bare
# "metal"), with k/rho/cp taken from real published values for that
# material specifically -- not fit to any one commercial mug/kettle
# product, but no longer an arbitrary round number standing in for a whole
# category either. Sourced from Wikipedia's "List of thermal
# conductivities" plus standard heat-transfer-reference (Cengel/Incropera-
# style appendix) density/specific-heat figures for each named material;
# where a real material's own literature range allowed some room (e.g.
# several hardwoods all cluster near k=0.15-0.19), the exact value chosen
# was nudged slightly within that real range so the table stays strictly
# increasing in k -- the ordinal "more conductive" meaning of a rising
# index (relied on throughout this file and mug.astro) requires that, and
# every value used is still inside that material's real documented range.
#
# A material that legitimately appears in both STRUCTURAL_MATERIALS and
# HANDLE_MATERIALS (stoneware, 304 stainless, 6061 aluminum) uses the
# IDENTICAL (k, rho, cp) triple in both tables -- it's the same physical
# substance either way (a stoneware mug with a stoneware handle, a steel
# travel mug with a steel handle), not two different fits to the same name.
STRUCTURAL_MATERIALS = [
    # EXPERIMENT_LOG §73: filtered from the original 20 down to 11 --
    # direct user request to make sure every material a mug BODY can be
    # made of (i.e. the one in direct, sustained contact with ~90C liquid
    # for up to 2 hours) is one real products actually use bare, not just
    # thermally plausible. Removed (real reasons, not a rounding error):
    # polystyrene/abs_plastic (not food-grade for hot contact, styrene/
    # additive migration risk), acrylic_pmma (heat-deflection temp ~80-
    # 100C is at or below this domain's own liquid temp held for hours --
    # a cold-drink material), melamine_resin (explicit consumer-safety
    # guidance against hot-liquid use, formaldehyde migration risk),
    # carbon_steel/cast_iron (no real bare-metal drinkware use, rust/
    # leaching with sustained liquid contact), brass/copper (real
    # drinkware in these metals is always nickel/stainless-LINED, never
    # bare -- coffee's acidity accelerates leaching), aluminum_6061 (real
    # aluminum drinkware is anodized/lined, bare aluminum leaching is the
    # standard reason it's avoided). HANDLE_MATERIALS left untouched --
    # a handle doesn't contact the liquid, so food-contact-material
    # standards don't apply the same way.
    #
    # EXPERIMENT_LOG §78: 5 more added, same "real bare-food-safe product
    # exists" bar as §73, not §73's own excluded metals reconsidered --
    # copper/brass/carbon-steel/cast-iron/bare-aluminum stay out for the
    # same leaching/food-contact reasons (nothing here gives v6's FD oracle
    # a coating layer to make a lined version of them physically honest --
    # see the COATING_MATERIALS/lining note below for where that boundary
    # actually is today). peek_polymer and high_density_polyethylene are
    # real hot-fill-rated food-grade plastics (PEEK reusable "unbreakable"
    # cups; HDPE hot-beverage tumblers) filling the gap between the
    # existing nylon and the glass cluster. glass_ceramic (Pyroceram-style
    # cookware/drinkware) and zirconia_ceramic (yttria-stabilized, used in
    # premium ceramic kitchenware) are real bare-ceramic bodies distinct
    # from the clay-based earthenware/stoneware/porcelain/bone_china
    # already here. pewter_leadfree (modern lead-free Britannia-metal
    # alloy) is the real traditional/still-produced drinkware metal this
    # table was otherwise missing -- unlike §73's excluded metals, solid
    # pewter tankards/mugs are a genuine bare-metal product category, not
    # one that's always lined in practice.
    #
    # EXPERIMENT_LOG §79: 3 more, direct user request to keep expanding.
    # birch_wood is this table's first organic material -- a real bare-
    # wood body (Nordic kuksa-style carved cups), traditionally food-safed
    # with an oil or wax treatment rather than a fired glaze; see the new
    # beeswax_seal entry in COATING_MATERIALS below for the matching real
    # treatment, so a kuksa is now expressible as (birch_wood,
    # beeswax_seal) rather than forcing a ceramic-style glaze onto a wood
    # body. silicone_rigid is the real, distinct product category of
    # rigid/collapsible food-grade-silicone cups (rated near-boiling by
    # real manufacturers) -- deliberately close in (k, rho, cp) to
    # HANDLE_MATERIALS' silicone_rubber, since it's genuinely the same
    # substance, same convention as stoneware/stainless/titanium already
    # appearing identically in both tables. alumina_ceramic (96% Al2O3
    # technical ceramic) is a real, food-safe, distinctly MORE conductive
    # bare ceramic than anything else in this table -- premium technical
    # ceramic drinkware exists, and it is not thermally interchangeable
    # with the clay-based/glass-ceramic entries already here.
    #
    # EXPERIMENT_LOG §80: 2 more, direct user request to keep expanding.
    # calabash_gourd is a second, genuinely different bare-organic vessel
    # from birch_wood -- specifically the real, still-widely-used dried
    # gourd shell yerba mate is traditionally served in (see the new
    # sterling_silver HANDLE_MATERIALS entry below for the actual matching
    # real product: silver-rimmed mate gourds). anodized_aluminum is the
    # real, legitimate answer to §73's own bare-aluminum exclusion: an
    # anodized oxide layer is what makes aluminum drinkware/cookware food-
    # safe in practice (hard-anodized camping mugs are a genuine, common
    # product), distinct from the untreated bare aluminum §73 correctly
    # kept out. Shares aluminum_6061's exact (k, rho, cp) triple with
    # HANDLE_MATERIALS -- the anodize layer is microns thick and doesn't
    # meaningfully change bulk conduction, so this IS the same substance,
    # same shared-triple convention as birch/silicone/titanium above.
    ("calabash_gourd", 0.06, 300.0, 1800.0),
    ("birch_wood", 0.16, 670.0, 1700.0),
    ("silicone_rigid", 0.20, 1200.0, 1600.0),
    ("polypropylene", 0.22, 905.0, 1920.0),
    ("nylon_pa6", 0.25, 1140.0, 1700.0),
    ("peek_polymer", 0.27, 1320.0, 1340.0),
    ("high_density_polyethylene", 0.48, 955.0, 1900.0),
    ("soda_lime_glass", 0.90, 2500.0, 840.0),
    ("earthenware", 0.95, 1900.0, 900.0),
    ("borosilicate_glass", 1.14, 2230.0, 830.0),
    ("stoneware", 1.30, 2200.0, 1050.0),
    ("fused_silica_glass", 1.45, 2200.0, 740.0),
    ("porcelain", 1.60, 2400.0, 1085.0),
    ("bone_china", 1.80, 2450.0, 1100.0),
    ("glass_ceramic", 2.00, 2600.0, 800.0),
    ("zirconia_ceramic", 2.60, 6000.0, 480.0),
    ("stainless_steel_304", 16.2, 8000.0, 500.0),
    ("titanium_grade2", 17.0, 4510.0, 523.0),
    ("alumina_ceramic", 30.0, 3900.0, 880.0),
    ("pewter_leadfree", 50.0, 7300.0, 210.0),
    ("anodized_aluminum", 167.0, 2700.0, 896.0),
]

# EXPERIMENT_LOG §89: a real, non-obvious combination directly caught by
# eyeballing a small peek batch -- a 43mm-thick calabash_gourd wall
# ("is that... a thing? Can calabash gourds even be shaped?"). Nothing
# upstream of this was wrong: r_outer/r_inner are sampled from one shared
# generic range regardless of material, exactly per §76's own deliberate
# "overlapping ranges, materials don't guarantee feasibility" philosophy
# -- but that philosophy was only ever meant to let GEOMETRY be
# infeasible (self-intersecting, sub-floor liquid volume), not to let a
# material be asked to form a wall shape real material of that kind could
# never plausibly have. This is deliberately NOT a manufacturing-process
# simulation (direct user steer: "we're not trying to evaluate
# manufacturing processes yet") -- just a coarse, honest plausibility
# band per material, keyed by name so it can't silently drift out of sync
# if STRUCTURAL_MATERIALS is reordered. Values are engineering judgment
# (a real dried gourd shell, a thrown/cast ceramic wall, a stamped metal
# mug wall, a molded polymer wall each have a real, very different
# typical thickness range) -- same "not measured, an order-of-magnitude
# stand-in" honesty this file already uses for H_HANDLE_CONTACT/
# HANDLE_PATCH_FRAC above, not a cited manufacturing spec.
STRUCTURAL_WALL_THICKNESS_RANGE_MM = {
    "calabash_gourd": (1.5, 6.0),          # a real dried gourd shell --
                                            # thin, fixed by the plant, not
                                            # something a wall thickness
                                            # parameter can freely inflate
    "birch_wood": (3.0, 15.0),             # hand-carved wood cup (kuksa-
                                            # style) -- thicker than a
                                            # ceramic/glass wall, still
                                            # bounded
    "silicone_rigid": (2.0, 12.0),
    "polypropylene": (1.0, 8.0),
    "nylon_pa6": (1.0, 8.0),
    "peek_polymer": (1.0, 10.0),
    "high_density_polyethylene": (1.0, 8.0),
    "soda_lime_glass": (2.0, 8.0),
    "earthenware": (3.0, 12.0),            # thrown/fired pottery wall
    "borosilicate_glass": (2.0, 8.0),
    "stoneware": (3.0, 12.0),
    "fused_silica_glass": (2.0, 8.0),
    "porcelain": (2.0, 8.0),
    "bone_china": (1.5, 6.0),              # famously thin-walled by design
    "glass_ceramic": (2.0, 8.0),
    "zirconia_ceramic": (1.5, 8.0),        # strong technical ceramic, can
                                            # run thinner than earthenware
    "stainless_steel_304": (0.5, 3.0),     # stamped/drawn sheet metal
    "titanium_grade2": (0.5, 3.0),
    "alumina_ceramic": (2.0, 10.0),
    "pewter_leadfree": (1.0, 5.0),
    "anodized_aluminum": (0.8, 4.0),
}
# Generic fallback for any STRUCTURAL_MATERIALS entry a future expansion
# adds without a matching entry above -- deliberately wide (not silently
# permissive of literally anything, but wide enough not to falsely reject
# a real material nobody's assigned real bounds to yet). A KeyError would
# be the "safer" failure mode in the abstract, but would take down dataset
# generation the next time someone adds a material and forgets this table
# -- same real, already-seen failure shape as forgetting to update
# mug_categorical.py -- so this fails open instead, loudly documented here
# rather than silently.
_DEFAULT_WALL_THICKNESS_RANGE_MM = (1.0, 20.0)


def wall_thickness_range_mm(struct_material_idx):
    name = STRUCTURAL_MATERIALS[int(round(struct_material_idx))][0]
    return STRUCTURAL_WALL_THICKNESS_RANGE_MM.get(name, _DEFAULT_WALL_THICKNESS_RANGE_MM)


INSULATION_MATERIALS = [
    ("vacuum", 0.005, 5.0, 500.0),
    ("aerogel", 0.015, 150.0, 1000.0),
    ("air_gap", 0.026, 1.2, 1005.0),
    ("foam", 0.03, 40.0, 1400.0),
]
# §74: thin surface coatings -- direct user request ("glazes or whatever
# else is traditionally used"), and directly relevant to §73's food-safety
# fix: a glaze is exactly WHY porous earthenware/stoneware are food-safe
# and waterproof as drinkware at all (raw fired clay is porous). Modeled
# as a thin EXTRA series thermal resistance (openfoam_libs/liquidCoupling
# for the inner/liquid-facing surface, the built-in
# externalWallHeatFluxTemperature `thicknessLayers`/`kappaLayers` fields
# -- already present, unused until now -- for the outer surface), NOT a
# separately meshed region: real glazes/enamels are ~0.1-0.5mm, far too
# thin to mesh affordably at dataset-generation scale. (k, rho, cp) --
# rho/cp only matter if this table is ever used somewhere that models the
# coating's own thermal mass; the coating BCs themselves only use k
# (via thickness/k, a pure resistance, no capacitance -- a real, standard
# simplification for a thin coating, same one thicknessLayers/kappaLayers
# itself makes).
COATING_MATERIALS = [
    ("ceramic_glaze", 1.1, 2500.0, 840.0),      # generic vitreous glaze --
                                                 # covers majolica/tin-glaze/
                                                 # salt-glaze/most traditional
                                                 # pottery glazes, which are
                                                 # thermally near-identical
                                                 # thin glass layers
    ("vitreous_enamel", 1.0, 2500.0, 800.0),    # glass-based coating fused
                                                 # onto METAL (classic
                                                 # enamelware camping mugs)
    ("epoxy_resin_liner", 0.2, 1200.0, 1500.0), # modern insulated-metal-mug
                                                 # liner -- genuinely lower k
                                                 # than the glass-based ones
    # EXPERIMENT_LOG §78: 4 more, same "real, thermally distinct lining/
    # coating a real drinkware product uses" bar as §74. These are also
    # this table's actual answer to §73's own excluded-bare-metal note
    # above ("real drinkware in these metals is always nickel/stainless-
    # LINED, never bare") -- a copper-bodied mug is real specifically
    # BECAUSE it's tin- or nickel-lined, not bare; that's now a real
    # option here even though bare copper still correctly isn't in
    # STRUCTURAL_MATERIALS. Indexed by round(), not material_props()'s
    # continuous interpolation (see mug_openfoam_oracle_v2.py's own
    # int(round(coating_material_idx)) usage) -- these don't need to stay
    # k-ascending the way STRUCTURAL_MATERIALS/HANDLE_MATERIALS do, kept
    # loosely ordered here only for readability.
    ("tin_lining", 65.0, 7300.0, 228.0),        # traditional lining for
                                                 # copper cookware/vessels --
                                                 # the historic answer to
                                                 # "how is a copper mug
                                                 # actually food-safe"
    ("nickel_lining", 90.9, 8900.0, 444.0),     # modern equivalent -- the
                                                 # standard lining on real
                                                 # copper "Moscow Mule" mugs
    ("silver_lining", 429.0, 10490.0, 235.0),   # heritage/luxury tea-and-
                                                 # coffee service lining --
                                                 # real, if rare, product
                                                 # category
    ("food_grade_lacquer", 0.19, 1150.0, 1400.0),  # thin protective/
                                                    # cosmetic spray coating
                                                    # on metal drinkware --
                                                    # genuinely lower k than
                                                    # the metal linings above,
                                                    # a different real use
                                                    # case from epoxy_resin_
                                                    # liner's structural
                                                    # insulated-bottle role
    # EXPERIMENT_LOG §79: 2 more, direct user request to keep expanding.
    ("beeswax_seal", 0.084, 960.0, 2000.0),     # pre-modern waterproofing/
                                                 # food-safing treatment for
                                                 # wood/earthenware vessels --
                                                 # the real match for the new
                                                 # birch_wood STRUCTURAL_
                                                 # MATERIALS entry above, and
                                                 # this table's lowest-k, only
                                                 # non-mineral/non-metal entry
    ("gold_lining", 317.0, 19300.0, 129.0),     # ultra-luxury heritage tea-
                                                 # service lining -- real,
                                                 # rarer than silver_lining,
                                                 # the top of this table's
                                                 # now real luxury-lining tier
    # EXPERIMENT_LOG §80: 1 more, direct user request to keep expanding.
    ("shellac_finish", 0.25, 1100.0, 1500.0),   # traditional natural-resin
                                                 # wood/gourd finish -- a
                                                 # real second option
                                                 # alongside beeswax_seal for
                                                 # calabash_gourd/birch_wood
                                                 # bodies, not a duplicate:
                                                 # a harder, more durable
                                                 # historical alternative to
                                                 # a wax seal
]
HANDLE_MATERIALS = [
    # EXPERIMENT_LOG §78: 7 more added to the original 15 -- handles don't
    # touch the liquid, so (unlike STRUCTURAL_MATERIALS) food-contact
    # status was never the filter here; these fill in real handle/grip/
    # bail materials the original table was missing, not a relaxation of
    # any prior standard. cork and leather_wrapped are real traditional
    # kettle/tankard grip materials; phenolic_resin (Bakelite-style) is
    # the classic heat-resistant handle material still used on pots and
    # kettles; carbon_fiber_composite is a real modern premium handle
    # material (isotropic k approximation, a real simplification -- true
    # carbon fiber is strongly anisotropic, but no caller here needs
    # direction-resolved conduction through a handle this thin); cast_iron,
    # brass, and copper are real bare BAIL/handle metals (a cast-iron
    # tetsubin's integral loop handle, a brass or copper kettle handle) --
    # legitimate here in a way bare copper/brass/cast-iron are NOT in
    # STRUCTURAL_MATERIALS, since a handle never needs the food-contact
    # food-safety bar the body does.
    #
    # EXPERIMENT_LOG §79: 3 more, direct user request to keep expanding.
    # rattan_wrap is a real traditional woven-natural-fiber grip wrap
    # (cane/wicker-wrapped handles on ceramic teapots and some traditional
    # coffee vessels), thermally near cork but a genuinely different real
    # material/construction. glass_filled_nylon is the real engineering-
    # plastic handle material actual kettles/appliance handles are molded
    # from -- distinctly more conductive than pure nylon_pa6/tpe_grip
    # thanks to its glass-fiber loading, not a duplicate of either.
    # titanium_grade2 (identical (k, rho, cp) triple to STRUCTURAL_
    # MATERIALS' own entry, same substance, same shared-triple convention
    # as stoneware/stainless_steel_304/aluminum_6061 already in both
    # tables) is real -- ultralight titanium bail/loop handles are a
    # genuine backpacking-mug product category.
    #
    # EXPERIMENT_LOG §80: 2 more, direct user request to keep expanding.
    # horn (buffalo/ox horn) is a real traditional handle material distinct
    # from every wood/rubber/grip entry already here -- artisanal
    # horn-handled drinkware and utensils are a genuine, if niche, craft
    # category. sterling_silver shares silver_lining's exact triple (same
    # substance, same shared-triple convention) -- real ornate historic
    # tea-service handles, and the actual match for a silver-rimmed
    # calabash_gourd mate cup (STRUCTURAL_MATERIALS, above).
    ("cork", 0.045, 240.0, 2000.0),
    ("rattan_wrap", 0.07, 300.0, 1500.0),
    ("natural_rubber", 0.134, 920.0, 1900.0),
    ("leather_wrapped", 0.15, 860.0, 1500.0),
    ("bamboo", 0.155, 700.0, 1300.0),
    ("neoprene", 0.16, 1230.0, 2140.0),
    ("walnut", 0.165, 650.0, 1700.0),
    ("oak", 0.17, 750.0, 1700.0),
    ("epdm_rubber", 0.175, 1200.0, 1980.0),
    ("maple", 0.18, 705.0, 1700.0),
    ("beech", 0.185, 750.0, 1700.0),
    ("horn", 0.19, 1300.0, 1300.0),
    ("silicone_rubber", 0.20, 1200.0, 1600.0),
    ("abs_grip", 0.21, 1050.0, 1400.0),
    ("polypropylene_grip", 0.22, 905.0, 1920.0),
    ("tpe_grip", 0.24, 1100.0, 1800.0),
    ("phenolic_resin", 0.28, 1350.0, 1600.0),
    ("glass_filled_nylon", 0.30, 1350.0, 1500.0),
    ("stoneware", 1.30, 2200.0, 1050.0),
    ("carbon_fiber_composite", 6.0, 1600.0, 1000.0),
    ("stainless_steel_304", 16.2, 8000.0, 500.0),
    ("titanium_grade2", 17.0, 4510.0, 523.0),
    ("cast_iron", 52.0, 7200.0, 460.0),
    ("brass", 115.0, 8500.0, 380.0),
    ("aluminum_6061", 167.0, 2700.0, 896.0),
    ("copper", 390.0, 8960.0, 385.0),
    ("sterling_silver", 429.0, 10490.0, 235.0),
]

# EXPERIMENT_LOG §82 (mug v2, Phase C): the joint itself as a first-class
# categorical choice, not a fixed H_HANDLE_CONTACT=200 constant shared by
# every candidate regardless of how the handle is actually attached. A
# ceramic handle luted on with slip before firing is a genuinely different
# thermal joint from a modern handle that's epoxied or press-fit onto a
# finished body -- the old constant flattened all of that into one
# order-of-magnitude guess. Modeled the same way §74's coatings are: a
# thin resistive interface layer (k, thickness) rather than a separately
# meshed region -- real bondlines/gaps here are sub-millimeter, same
# affordability argument as a glaze. `joint_conductance_w_m2k()` below
# converts (k, thickness) -> h = k/thickness once, at the oracle's own
# call site, so this table's SHAPE matches COATING_MATERIALS (an
# interface, not a bulk material -- rho/cp are genuinely irrelevant here,
# a thin bondline's own thermal mass is negligible next to the handle/body
# either side of it, so this table is only 3 columns instead of 4).
# Deliberately real, physically distinct joint types spanning nearly 4
# orders of magnitude in the resulting h -- from a loose mechanical fit
# (mostly thermally isolating, the handle runs close to its OWN ambient-
# convection equilibrium) to a genuinely fused/continuous joint (no real
# interface at all, conducts almost as well as the bulk material itself):
JOINT_MATERIALS = [
    # (name, k [W/(m K)], bond_thickness [mm])
    ("mechanical_press_fit_dry", 0.026, 0.5),      # trapped-air-gap contact
                                                    # resistance (k = still
                                                    # air, same value as
                                                    # INSULATION_MATERIALS'
                                                    # air_gap) -- a snap-fit
                                                    # or friction-fit handle
                                                    # with no bonding agent
                                                    # at all, real surface
                                                    # roughness leaves mostly
                                                    # air-filled microscopic
                                                    # gaps -> h ~ 52 W/m2K
    ("silicone_gasket_clamp", 0.20, 1.0),          # a compliant silicone
                                                    # gasket/boot clamped
                                                    # between handle and
                                                    # body (k matches
                                                    # HANDLE_MATERIALS'
                                                    # silicone_rubber) ->
                                                    # h ~ 200 W/m2K, kept
                                                    # deliberately equal to
                                                    # the old fixed constant
                                                    # so that constant isn't
                                                    # orphaned -- this IS
                                                    # what it was standing
                                                    # in for
    ("structural_epoxy_adhesive", 0.20, 0.3),      # a real bonded joint (k
                                                    # matches COATING_
                                                    # MATERIALS' epoxy_
                                                    # resin_liner, same
                                                    # substance), thinner
                                                    # bondline than a
                                                    # compliant gasket ->
                                                    # h ~ 667 W/m2K
    ("brazed_metal_joint", 50.0, 0.15),            # silver-braze filler
                                                    # alloy joining two
                                                    # metal parts (a brazed
                                                    # stainless/titanium
                                                    # handle) -> h ~ 333000
                                                    # W/m2K -- a thin,
                                                    # highly-conductive
                                                    # metal filler layer, on
                                                    # par with the fused
                                                    # joints below despite
                                                    # being a genuinely
                                                    # separate material
    ("ceramic_slip_fusion", 1.30, 0.05),           # traditional ceramic
                                                    # handle luted on with
                                                    # slip before firing --
                                                    # becomes literally one
                                                    # continuous fired body,
                                                    # no real interface;
                                                    # modeled as a nominal
                                                    # near-zero-thickness
                                                    # layer of the body's
                                                    # own stoneware k ->
                                                    # h ~ 26000 W/m2K
    ("welded_metal_joint", 16.2, 0.05),            # true metal fusion (TIG/
                                                    # laser-welded integral
                                                    # loop handle) -- same
                                                    # "no real interface"
                                                    # logic as ceramic_slip_
                                                    # fusion, stainless_
                                                    # steel_304's own k ->
                                                    # h ~ 324000 W/m2K,
                                                    # effectively conducting
                                                    # as well as the bulk
                                                    # metal itself
]


def joint_conductance_w_m2k(joint_material_idx):
    """h = k/thickness for the (k, thickness) interface-layer model above --
    the single call site the oracle uses instead of the old fixed
    H_HANDLE_CONTACT constant. Indexed by round(), same categorical
    convention as COATING_MATERIALS (not material_props()'s continuous
    interpolation -- a joint TYPE is a discrete choice, not a blend)."""
    _, k, thickness_mm = JOINT_MATERIALS[int(round(joint_material_idx))]
    return k / (thickness_mm / 1000.0)


N_HANDLE = 20

PARAM_DIM_V2 = 8
PARAM_NAMES_V2 = [
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "t_gap_mm", "insulation_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
]


def material_props(idx, table):
    """idx: continuous, clamped to [0, len(table)-1] -- interpolates (k, rho,
    cp) between adjacent named reference materials sorted by k ascending."""
    idx = float(np.clip(idx, 0, len(table) - 1))
    xs = np.arange(len(table), dtype=float)
    ks = np.array([t[1] for t in table])
    rhos = np.array([t[2] for t in table])
    cps = np.array([t[3] for t in table])
    return float(np.interp(idx, xs, ks)), float(np.interp(idx, xs, rhos)), float(np.interp(idx, xs, cps))


def _step_chain(T, dx, k, rho, cp, area, dt, T_amb_right, h_right,
                 T_left_conv=None, h_left=None, T_left_dirichlet=None,
                 T_right_dirichlet=None, lateral_hP=0.0, T_lateral=T_AMB,
                 extra_right_G=0.0, extra_right_T=T_AMB):
    """One implicit (backward-Euler) step of a 1D finite-volume conduction
    chain with fixed (not coupled-unknown) boundary values on both ends --
    either a convective left boundary (T_left_conv/h_left, what the wall
    bands use to reach the liquid) or a Dirichlet left boundary
    (T_left_dirichlet, what the handle's base end uses to reach the wall),
    a convective right boundary (T_amb_right/h_right; h_right=0 means
    adiabatic/no-flux, a free fin tip) OR a Dirichlet right boundary
    (T_right_dirichlet -- §56: the handle's SECOND attached end, once the
    loop shape got two attachment points instead of one attached end + one
    free tip), and an optional per-cell lateral convective loss term
    (lateral_hP = h*perimeter, zero for the wall bands which only lose heat
    at their two ends, nonzero for the handle which loses heat along its
    whole exposed length -- the fin equation's extra term).

    `extra_right_G`/`extra_right_T` (§58): an OPTIONAL second, PARALLEL
    conductance at the right boundary, on top of whichever primary right
    condition is active -- e.g. a wall cell that loses heat to ambient air
    AND separately conducts to a handle root attached at that same point.
    Only meaningful alongside a convective (not Dirichlet) primary right
    boundary; ignored (by construction, added after the branch above) if
    T_right_dirichlet clamps the cell instead. Solving this IMPLICITLY
    (rather than as an explicit per-step correction elsewhere) is the
    whole reason this exists: a handle-contact flux applied explicitly
    against a single wall cell's own (often tiny) thermal mass blew up
    immediately in testing -- see EXPERIMENT_LOG §58.

    `T` may be 2D (n_cell, n_batch) instead of 1D (n_cell,) -- §59: every
    azimuthal column at a given height shares the IDENTICAL banded matrix
    (same geometry/material/dt/boundary type regardless of angle, only the
    per-column temperature history differs), so `scipy.linalg.solve_banded`
    can solve every column in ONE call via its native multiple-right-hand-
    side support, rather than one call per column -- this is what keeps a
    real 3D (axial x azimuthal x radial) grid's cost in the same ballpark
    as v4's 2D one, instead of scaling with however many azimuthal cells
    are added. `T_left_conv`/`T_amb_right`/etc. stay scalars either way
    (nothing in the shared-matrix case varies by column by construction);
    only `T` (and therefore `rhs`) actually becomes 2D."""
    n = len(T)
    C = rho * cp * dx * area
    G = np.empty(n - 1)
    for i in range(n - 1):
        R = dx[i] / (2 * k[i] * area) + dx[i + 1] / (2 * k[i + 1] * area)
        G[i] = 1.0 / R

    diag = C / dt + lateral_hP * dx
    lower = np.zeros(n)
    upper = np.zeros(n)
    if T.ndim == 1:
        rhs = C / dt * T + lateral_hP * dx * T_lateral
    else:
        rhs = (C / dt)[:, None] * T + (lateral_hP * dx)[:, None] * T_lateral

    for i in range(n - 1):
        diag[i] += G[i]
        diag[i + 1] += G[i]
        upper[i] = -G[i]
        lower[i + 1] = -G[i]

    if T_left_dirichlet is not None:
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = T_left_dirichlet
    else:
        G_left = 1.0 / (1.0 / (h_left * area) + dx[0] / (2 * k[0] * area))
        diag[0] += G_left
        rhs[0] += G_left * T_left_conv

    if T_right_dirichlet is not None:
        diag[-1] = 1.0
        lower[-1] = 0.0
        rhs[-1] = T_right_dirichlet
    elif h_right != 0.0:
        G_right = 1.0 / (1.0 / (h_right * area) + dx[-1] / (2 * k[-1] * area))
        diag[-1] += G_right
        rhs[-1] += G_right * T_amb_right
        if extra_right_G != 0.0:
            diag[-1] += extra_right_G
            rhs[-1] += extra_right_G * extra_right_T
    # h_right == 0.0 and no T_right_dirichlet: adiabatic tip, no term added
    # (extra_right_G has no primary term to add alongside in this branch --
    # not a configuration any caller uses)

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def simulate_v2(t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 t_gap_mm, insulation_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0):
    if min(t_wall_rim_mm, t_wall_base_mm, t_gap_mm, handle_length_mm, handle_diameter_mm) <= 0:
        return dict(valid=False)

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_ins, rho_ins, cp_ins = material_props(insulation_material_idx, INSULATION_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)

    area_band = A_INNER / 2.0  # rim band + base band split the cylindrical side wall evenly

    def build_band(t_wall_mm):
        t_wall, t_gap = t_wall_mm / 1000.0, t_gap_mm / 1000.0
        dx = np.concatenate([np.full(N_WALL, t_wall / N_WALL), np.full(N_GAP, t_gap / N_GAP)])
        k = np.concatenate([np.full(N_WALL, k_struct), np.full(N_GAP, k_ins)])
        rho = np.concatenate([np.full(N_WALL, rho_struct), np.full(N_GAP, rho_ins)])
        cp = np.concatenate([np.full(N_WALL, cp_struct), np.full(N_GAP, cp_ins)])
        return dx, k, rho, cp

    dx_rim, k_rim, rho_rim, cp_rim = build_band(t_wall_rim_mm)
    dx_base, k_base, rho_base, cp_base = build_band(t_wall_base_mm)

    n_band = N_WALL + N_GAP
    T_rim = np.full(n_band, T0_LIQUID)
    T_base = np.full(n_band, T0_LIQUID)
    T_liq = T0_LIQUID

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)  # handle starts at room temp, not fill temp

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False

    for step in range(n_steps):
        t_now = step * dt
        flux_rim = H_LIQ * area_band * (T_liq - T_rim[0])
        flux_base = H_LIQ * area_band * (T_liq - T_base[0])
        T_liq_new = T_liq - dt / C_LIQ * (flux_rim + flux_base)

        T_rim_new = _step_chain(T_rim, dx_rim, k_rim, rho_rim, cp_rim, area_band, dt,
                                 T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
        T_base_new = _step_chain(T_base, dx_base, k_base, rho_base, cp_base, area_band, dt,
                                  T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)

        body_temp = 0.5 * (T_rim[-1] + T_base[-1])
        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=T_AMB, h_right=0.0, T_left_dirichlet=body_temp,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)

        ok = (np.isfinite(T_liq_new) and np.all(np.isfinite(T_rim_new))
              and np.all(np.isfinite(T_base_new)) and np.all(np.isfinite(T_handle_new)))
        if not ok:
            return dict(valid=False)

        T_liq, T_rim, T_base, T_handle = T_liq_new, T_rim_new, T_base_new, T_handle_new

        if not recorded and (t_now + dt) >= record_at:
            touch_temp_at = float(max(T_rim[-1], T_base[-1]))
            handle_temp_at = float(T_handle[-1])
            recorded = True

    mass_struct = area_band * ((t_wall_rim_mm + t_wall_base_mm) / 1000.0) * rho_struct
    mass_ins = area_band * 2 * (t_gap_mm / 1000.0) * rho_ins
    mass_handle = A_h * L_h * rho_handle
    mass_kg = mass_struct + mass_ins + mass_handle

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
    )


# ---------------------------------------------------------------------------
# v3 (EXPERIMENT_LOG §50): user asked to "uplevel" the domain further, picked
# two of four proposed directions -- a lid/top-loss model (the single
# biggest correctness gap in v1/v2: the top of the mug lost ZERO heat,
# which is backwards -- an open liquid surface is typically the FASTEST
# loss pathway on a real mug, not a negligible one) and a variable body
# shape (radius profile instead of a fixed-radius cylinder). v2's
# simulate_v2() is left untouched for reference/regression.
#
# Body shape: radius at 3 heights (base/mid/rim) instead of one constant --
# a taper (narrower base), a flare (wider rim), or a belly (wider middle)
# are all expressible, not just a straight cylinder. Total height stays
# fixed (HEIGHT=0.10m) -- varying it too is a real further idea, not done
# here (kept the parameter count from ballooning in one pass). Each
# half-height "band" is now a frustum (truncated cone), not a plain
# cylinder shell -- lateral area and volume both computed from the real
# frustum formulas, not the fixed-cylinder constants v1/v2 used.
#
# Lid: a single continuous `lid_coverage_frac` in [0,1] interpolates
# between a fully open cup (0) and a fully sealed lid (1) -- avoids a
# discrete open/lidded branch, consistent with this project's
# continuous-parameterization style everywhere else. The open fraction of
# the top loses heat directly to ambient via a temperature-DEPENDENT
# effective coefficient (h rises with liquid temperature, approximating
# evaporation's own dependence on vapor pressure, which rises with
# temperature -- a real, load-bearing simplification instead of a full
# Antoine-equation vapor-pressure model, chosen because the ballpark
# magnitude -- open coffee cools much faster than lidded, common
# experience -- matters more here than the exact curve shape). The
# covered fraction conducts through a lid (its own thickness + a material
# index, reusing STRUCTURAL_MATERIALS -- a lid is a similar kind of
# component to the wall shell, not worth a 4th materials table) to
# ambient at the plain (non-evaporative) side-wall convection coefficient.
# ---------------------------------------------------------------------------

H_AIR_TOP_OPEN = 15.0   # W/(m2 K), open-liquid-surface convection baseline --
                        # somewhat higher than the side wall's H_AIR=10 (an
                        # unobstructed buoyant plume above a hot liquid convects
                        # more freely than air along a vertical wall)
EVAP_COEFF = 60.0       # W/(m2 K)-equivalent, evaporation's contribution at
                        # full temperature difference -- an order-of-magnitude
                        # stand-in for "open coffee cools much faster than
                        # lidded," not a real vapor-pressure/Antoine-equation
                        # model; scales linearly with (T_liq-T_amb)/(T0-T_amb)
                        # so it fades as the liquid approaches ambient, same
                        # direction real evaporation rate does (falling vapor
                        # pressure difference), just not the same curve shape

N_LID = 15
N_AXIAL_BANDS = 6  # §56: was 2 (rim band + base band) -- see simulate_v3's
                   # own docstring for why this, not a finer N_WALL/N_GAP,
                   # is what actually gives the 3D view real height-wise
                   # spatial detail to show.

PARAM_DIM_V3 = 14
PARAM_NAMES_V3 = [
    "r_base_mm", "r_mid_mm", "r_rim_mm",
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "t_gap_mm", "insulation_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
    "lid_coverage_frac", "t_lid_mm", "lid_material_idx",
]


def frustum_lateral_area(r1, r2, h):
    return math.pi * (r1 + r2) * math.sqrt((r2 - r1) ** 2 + h ** 2)


def frustum_volume(r1, r2, h):
    return (math.pi * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)


def radius_at_height(r_base, r_mid, r_rim, height, z):
    """Same 2-segment (base->mid->rim) piecewise-linear profile the body
    geometry itself uses, evaluated at an arbitrary height z -- lets
    N_AXIAL_BANDS bands (or anything else) sample the real body profile at
    any height, not just the 3 design points."""
    if z <= height / 2:
        return r_base + (r_mid - r_base) * (z / (height / 2))
    return r_mid + (r_rim - r_mid) * ((z - height / 2) / (height / 2))


def simulate_v3(r_base_mm, r_mid_mm, r_rim_mm,
                 t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 t_gap_mm, insulation_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 lid_coverage_frac, t_lid_mm, lid_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0,
                 record_series=False, series_dt=None):
    """record_series=True (viewer-only -- never set by the oracle/bulk-
    generation path, which only needs the two scalar snapshots): also
    returns a time series of liquid temp + each band's outer temp + the
    handle's own full internal profile, for the viewer's
    temperature-over-time animation.

    series_dt=None (default) records every native integration step (dt) --
    §55 user feedback on the original 60s downsampling: the first minute or
    two is where almost all the interesting transient behavior actually
    happens, so a 60s stride made the animation's first couple of frames
    look like a discontinuous "flip" rather than a diffusion.

    §56: the body wall is divided into N_AXIAL_BANDS (6, up from 2 --
    rim/base) independent radial conduction chains stacked along the
    height, each its own frustum slice with wall thickness linearly
    interpolated between t_wall_base_mm/t_wall_rim_mm at that height.
    This is a real, deliberate distinction from just increasing N_WALL/
    N_GAP: those control resolution THROUGH the wall's thickness (radial),
    which does NOT add any new spatial detail ALONG the height, since the
    body's own visualization only ever reads one outer-surface value per
    band regardless of how finely that band's own radial direction is
    resolved. Going from 2 bands to 6 is what actually gives the 3D
    viewer real height-wise data to interpolate between, instead of a
    single straight-line blend across the whole body no matter how "fine"
    the mesh was inside those 2 fixed points.

    The handle is now attached at BOTH ends (T_left_dirichlet AND
    T_right_dirichlet, both driven by the same near-mid-height band) --
    §56 user feedback: the loop shape (added in §53 for realism) has two
    attachment points and no free tip, but the fin model still had one
    attached end + one free end, a real, named geometric mismatch. Fixing
    the PHYSICS to match the loop (rather than fudging the visualization
    further) makes the loop's own farthest point (from the mug) the
    genuine coolest point of a symmetric profile -- exactly what a
    real two-point-attached loop handle would do -- rather than an
    approximation of a profile that was never actually symmetric to begin
    with. `handle_temp_at_C`/`T_handle_tip` now report the PROFILE'S
    MIDDLE index (the new genuine coolest point), not the last index."""
    if series_dt is None:
        series_dt = dt
    if min(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm,
           t_gap_mm, handle_length_mm, handle_diameter_mm, t_lid_mm) <= 0:
        return dict(valid=False)
    lid_coverage_frac = float(np.clip(lid_coverage_frac, 0.0, 1.0))

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_ins, rho_ins, cp_ins = material_props(insulation_material_idx, INSULATION_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)
    k_lid, rho_lid, cp_lid = material_props(lid_material_idx, STRUCTURAL_MATERIALS)

    r_base, r_mid, r_rim = r_base_mm / 1000.0, r_mid_mm / 1000.0, r_rim_mm / 1000.0
    height = HEIGHT
    band_edges = [i * height / N_AXIAL_BANDS for i in range(N_AXIAL_BANDS + 1)]
    band_r = [radius_at_height(r_base, r_mid, r_rim, height, z) for z in band_edges]
    band_h = height / N_AXIAL_BANDS
    band_area = [frustum_lateral_area(band_r[i], band_r[i + 1], band_h) for i in range(N_AXIAL_BANDS)]
    vol_liq = sum(frustum_volume(band_r[i], band_r[i + 1], band_h) for i in range(N_AXIAL_BANDS))
    c_liq = RHO_LIQUID * vol_liq * CP_LIQUID
    a_top = math.pi * r_rim ** 2
    open_area = a_top * (1.0 - lid_coverage_frac)
    lid_area = a_top * lid_coverage_frac

    # wall thickness per band -- linear interpolation between the two real
    # design parameters, evaluated at each band's own height midpoint
    band_t_wall_mm = [t_wall_base_mm + (t_wall_rim_mm - t_wall_base_mm)
                       * ((band_edges[i] + band_edges[i + 1]) / 2 / height)
                       for i in range(N_AXIAL_BANDS)]

    def build_band(t_wall_mm):
        t_wall, t_gap = t_wall_mm / 1000.0, t_gap_mm / 1000.0
        dx = np.concatenate([np.full(N_WALL, t_wall / N_WALL), np.full(N_GAP, t_gap / N_GAP)])
        k = np.concatenate([np.full(N_WALL, k_struct), np.full(N_GAP, k_ins)])
        rho = np.concatenate([np.full(N_WALL, rho_struct), np.full(N_GAP, rho_ins)])
        cp = np.concatenate([np.full(N_WALL, cp_struct), np.full(N_GAP, cp_ins)])
        return dx, k, rho, cp

    bands = [build_band(band_t_wall_mm[i]) for i in range(N_AXIAL_BANDS)]
    n_band = N_WALL + N_GAP
    T_bands = [np.full(n_band, T0_LIQUID) for _ in range(N_AXIAL_BANDS)]
    T_liq = T0_LIQUID
    mid_band_idx = N_AXIAL_BANDS // 2  # both handle attachment points treated as near mid-height

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)
    handle_mid_idx = N_HANDLE // 2

    has_lid = lid_area > 1e-9
    if has_lid:
        t_lid = t_lid_mm / 1000.0
        dx_lid = np.full(N_LID, t_lid / N_LID)
        k_lid_arr = np.full(N_LID, k_lid)
        rho_lid_arr = np.full(N_LID, rho_lid)
        cp_lid_arr = np.full(N_LID, cp_lid)
        # Starts at ambient, not fill temperature -- like the handle (T_handle
        # above) and unlike the wall bands: a lid isn't liquid-wetted at t=0
        # the way a submerged wall shell is, it only heats via conduction
        # from the liquid/vapor below.
        T_lid = np.full(N_LID, T_AMB)
    else:
        T_lid = None

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False
    series = None
    if record_series:
        series = {"t_s": [], "T_liq": [], "T_bands_outer": [], "T_rim_outer": [], "T_base_outer": [],
                  "T_lid_outer": [], "T_handle_tip": [], "T_handle_profile": []}
        next_sample_t = 0.0

    def record_frame(t_now):
        series["t_s"].append(t_now)
        series["T_liq"].append(float(T_liq))
        series["T_bands_outer"].append([float(Tb[-1]) for Tb in T_bands])
        series["T_rim_outer"].append(float(T_bands[-1][-1]))  # convenience aliases for the 2D chart
        series["T_base_outer"].append(float(T_bands[0][-1]))
        series["T_lid_outer"].append(float(T_lid[-1]) if has_lid else None)
        series["T_handle_tip"].append(float(T_handle[handle_mid_idx]))
        series["T_handle_profile"].append(T_handle.tolist())

    for step in range(n_steps):
        t_now = step * dt
        if record_series and t_now >= next_sample_t:
            record_frame(t_now)
            next_sample_t += series_dt

        h_open = H_AIR_TOP_OPEN + EVAP_COEFF * max(0.0, T_liq - T_AMB) / (T0_LIQUID - T_AMB)
        flux_bands = sum(H_LIQ * band_area[i] * (T_liq - T_bands[i][0]) for i in range(N_AXIAL_BANDS))
        flux_open = h_open * open_area * (T_liq - T_AMB)
        flux_lid = H_LIQ * lid_area * (T_liq - T_lid[0]) if has_lid else 0.0
        T_liq_new = T_liq - dt / c_liq * (flux_bands + flux_open + flux_lid)

        T_bands_new = []
        for i in range(N_AXIAL_BANDS):
            dx_b, k_b, rho_b, cp_b = bands[i]
            T_bands_new.append(_step_chain(T_bands[i], dx_b, k_b, rho_b, cp_b, band_area[i], dt,
                                            T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ))

        body_temp = T_bands[mid_band_idx][-1]
        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=T_AMB, h_right=0.0,
                                    T_left_dirichlet=body_temp, T_right_dirichlet=body_temp,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)
        if has_lid:
            T_lid_new = _step_chain(T_lid, dx_lid, k_lid_arr, rho_lid_arr, cp_lid_arr, lid_area, dt,
                                     T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
        else:
            T_lid_new = None

        finite = (np.isfinite(T_liq_new) and all(np.all(np.isfinite(Tb)) for Tb in T_bands_new)
                  and np.all(np.isfinite(T_handle_new))
                  and (T_lid_new is None or np.all(np.isfinite(T_lid_new))))
        if not finite:
            return dict(valid=False)

        T_liq, T_bands, T_handle = T_liq_new, T_bands_new, T_handle_new
        if has_lid:
            T_lid = T_lid_new

        if not recorded and (t_now + dt) >= record_at:
            surfaces = [Tb[-1] for Tb in T_bands]
            if has_lid:
                surfaces.append(T_lid[-1])
            touch_temp_at = float(max(surfaces))
            handle_temp_at = float(T_handle[handle_mid_idx])
            recorded = True

    if record_series:
        # capture the final state too, so the animation's last frame is the
        # true end state rather than whatever the last series_dt-aligned
        # sample happened to land on
        record_frame(n_steps * dt)

    mass_struct = sum(band_area[i] * (band_t_wall_mm[i] / 1000.0) * rho_struct for i in range(N_AXIAL_BANDS))
    mass_ins = sum(band_area[i] * (t_gap_mm / 1000.0) * rho_ins for i in range(N_AXIAL_BANDS))
    mass_handle = A_h * L_h * rho_handle
    mass_lid = lid_area * (t_lid_mm / 1000.0) * rho_lid if has_lid else 0.0
    mass_kg = mass_struct + mass_ins + mass_handle + mass_lid

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
        liquid_volume_L=float(vol_liq * 1000.0),
        series=series,
        geometry=dict(r_base_m=r_base, r_mid_m=r_mid, r_rim_m=r_rim, height_m=HEIGHT,
                       has_lid=has_lid, lid_coverage_frac=lid_coverage_frac,
                       handle_length_m=L_h, handle_diameter_m=d_h),
    )


# ---------------------------------------------------------------------------
# v4 (EXPERIMENT_LOG §57): real coupled 2D (axial x radial) wall conduction,
# replacing v3's N_AXIAL_BANDS independent chains. Direct user observation,
# checked and confirmed rather than assumed: v3's bands never exchanged heat
# with each other -- only with the shared liquid bulk and ambient -- so
# nothing ever actually diffused vertically over time; a locally-hot band
# just relaxed toward its own local steady state in isolation. Worse, the
# viewer's LatheGeometry only ever had 3 profile points (2 straight
# segments), so only 3 of v3's 6 band values ever reached a distinct
# vertex -- the rest were invisible regardless of how good the physics was.
# This section fixes the physics; the viewer fix (finer Lathe profile) is
# separate, in mug.astro.
#
# Same 14-dim parameter vector as v3 (PARAM_DIM_V3/PARAM_NAMES_V3,
# unchanged) -- purely an internal-solve upgrade, not a new design axis.
#
# Scheme: implicit-radial / explicit-axial operator splitting, not full
# ADI -- chosen for lower implementation risk and because it needs no
# changes to `_step_chain` at all. Each outer dt=5s step: (1) the liquid's
# explicit update, generalized from a 6-way to an N_AXIAL_NODES-way flux
# sum; (2) one _step_chain() radial solve per row, byte-for-byte the same
# call v3 already makes per band, just N_AXIAL_NODES times instead of 6;
# (3) a NEW explicit, vectorized axial diffusion correction
# (_axial_diffuse) across the whole grid at once. The radial direction
# stays implicit because it has to -- its cells are sub-millimeter and
# steel's diffusivity is high, so an explicit radial step would need
# thousands of substeps (this is exactly why v1 already used backward
# Euler radially). The axial direction is coarser (dz = height/24, several
# mm) so explicit is workable there with a small number of substeps,
# computed from a real stability bound rather than guessed.
# ---------------------------------------------------------------------------

N_AXIAL_NODES = 20  # real coupled axial resolution -- "dramatically more
                    # than 6, not extreme," the same reasoning §56 used
                    # picking N_AXIAL_BANDS=6 over 2. Checked directly, not
                    # assumed: 24 measured ~2.1s/candidate (record_series=True,
                    # the viewer's own call pattern) -- over this project's own
                    # ~2s single-candidate target -- 20 measured ~1.8s, a real
                    # speed/resolution tradeoff point, not a convergence-swept
                    # value (same open-item category as N_AXIAL_BANDS was).


def _axial_diffuse(T, k_cell, C_grid, r_center, dx_grid, dz, dt):
    """Explicit, fully-vectorized finite-volume diffusion step ACROSS the
    z-axis of a (N_AXIAL_NODES, n_cell) grid -- the real spatial coupling
    v3's independent bands never had. Thin-shell approximation (consistent
    with this whole model's stated thin-wall convention): the cross-section
    carrying axial flux at radial cell x between two adjacent rows is the
    annular ring area 2*pi*r*dx(x), not a full disk. Adiabatic (zero-flux)
    top/bottom ends -- no base-plate or open-rim-edge axial loss term, same
    scope limit as v3 never modeling a base disc.

    Explicit -> needs its own stability bound. Computed directly from the
    assembled conductances/capacities every call (not assumed -- same
    "checked directly" spirit as N_GAP's own convergence-tested resolution,
    §21) and sub-cycled enough times that the caller's outer dt never
    exceeds it, with a 0.9 safety margin.
    """
    n_z = T.shape[0]
    if n_z < 2:
        return T
    r_mid = 0.5 * (r_center[:-1] + r_center[1:])                        # (n_z-1,)
    dx_avg = 0.5 * (dx_grid[:-1, :] + dx_grid[1:, :])                   # (n_z-1, n_cell)
    G = k_cell[None, :] * (2 * math.pi * r_mid[:, None] * dx_avg) / dz  # (n_z-1, n_cell)

    G_sum = np.zeros_like(C_grid)
    G_sum[:-1, :] += G
    G_sum[1:, :] += G
    mask = G_sum > 0
    dt_max = float(np.min(C_grid[mask] / G_sum[mask])) if np.any(mask) else np.inf
    n_sub = max(1, math.ceil(dt / (0.9 * dt_max))) if np.isfinite(dt_max) else 1
    dt_sub = dt / n_sub

    Tc = T.copy()
    for _ in range(n_sub):
        flux = np.zeros_like(Tc)
        flux[:-1, :] += G * (Tc[1:, :] - Tc[:-1, :])
        flux[1:, :] += G * (Tc[:-1, :] - Tc[1:, :])
        Tc += dt_sub / C_grid * flux
    return Tc


def simulate_v4(r_base_mm, r_mid_mm, r_rim_mm,
                 t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 t_gap_mm, insulation_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 lid_coverage_frac, t_lid_mm, lid_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0,
                 record_series=False, series_dt=None):
    """See this module's v4 header comment. Identical parameter vector and
    identical target definitions to simulate_v3 -- only the wall's internal
    solve changes, from N_AXIAL_BANDS independent radial chains to a real
    coupled (N_AXIAL_NODES x radial) 2D field."""
    if series_dt is None:
        series_dt = dt
    if min(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm,
           t_gap_mm, handle_length_mm, handle_diameter_mm, t_lid_mm) <= 0:
        return dict(valid=False)
    lid_coverage_frac = float(np.clip(lid_coverage_frac, 0.0, 1.0))

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_ins, rho_ins, cp_ins = material_props(insulation_material_idx, INSULATION_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)
    k_lid, rho_lid, cp_lid = material_props(lid_material_idx, STRUCTURAL_MATERIALS)

    r_base, r_mid, r_rim = r_base_mm / 1000.0, r_mid_mm / 1000.0, r_rim_mm / 1000.0
    height = HEIGHT
    n_z = N_AXIAL_NODES
    edges = [i * height / n_z for i in range(n_z + 1)]
    edge_r = [radius_at_height(r_base, r_mid, r_rim, height, z) for z in edges]
    dz = height / n_z
    row_area = [frustum_lateral_area(edge_r[i], edge_r[i + 1], dz) for i in range(n_z)]
    row_r_center = np.array([radius_at_height(r_base, r_mid, r_rim, height, (edges[i] + edges[i + 1]) / 2)
                              for i in range(n_z)])
    vol_liq = sum(frustum_volume(edge_r[i], edge_r[i + 1], dz) for i in range(n_z))
    c_liq = RHO_LIQUID * vol_liq * CP_LIQUID
    a_top = math.pi * r_rim ** 2
    open_area = a_top * (1.0 - lid_coverage_frac)
    lid_area = a_top * lid_coverage_frac

    row_t_wall_mm = np.array([t_wall_base_mm + (t_wall_rim_mm - t_wall_base_mm)
                               * ((edges[i] + edges[i + 1]) / 2 / height) for i in range(n_z)])

    n_band = N_WALL + N_GAP
    # materials don't vary by height in this model (only the structural
    # layer's THICKNESS does, via row_t_wall_mm) -- k/rho/cp per radial cell
    # are the same for every row, exactly like v3's per-band arrays already
    # were identical across bands.
    k_cell = np.concatenate([np.full(N_WALL, k_struct), np.full(N_GAP, k_ins)])
    rho_cell = np.concatenate([np.full(N_WALL, rho_struct), np.full(N_GAP, rho_ins)])
    cp_cell = np.concatenate([np.full(N_WALL, cp_struct), np.full(N_GAP, cp_ins)])
    dx_grid = np.stack([
        np.concatenate([np.full(N_WALL, (row_t_wall_mm[i] / 1000.0) / N_WALL),
                         np.full(N_GAP, (t_gap_mm / 1000.0) / N_GAP)])
        for i in range(n_z)
    ])  # (n_z, n_band)

    T_grid = np.full((n_z, n_band), T0_LIQUID)
    T_liq = T0_LIQUID
    mid_row_idx = n_z // 2

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)
    handle_mid_idx = N_HANDLE // 2

    has_lid = lid_area > 1e-9
    if has_lid:
        t_lid = t_lid_mm / 1000.0
        dx_lid = np.full(N_LID, t_lid / N_LID)
        k_lid_arr = np.full(N_LID, k_lid)
        rho_lid_arr = np.full(N_LID, rho_lid)
        cp_lid_arr = np.full(N_LID, cp_lid)
        T_lid = np.full(N_LID, T_AMB)
    else:
        T_lid = None

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False
    series = None
    if record_series:
        series = {"t_s": [], "T_liq": [], "T_bands_outer": [], "T_rim_outer": [], "T_base_outer": [],
                  "T_lid_outer": [], "T_handle_tip": [], "T_handle_profile": []}
        next_sample_t = 0.0

    def record_frame(t_now):
        series["t_s"].append(t_now)
        series["T_liq"].append(float(T_liq))
        series["T_bands_outer"].append(T_grid[:, -1].tolist())
        series["T_rim_outer"].append(float(T_grid[-1, -1]))
        series["T_base_outer"].append(float(T_grid[0, -1]))
        series["T_lid_outer"].append(float(T_lid[-1]) if has_lid else None)
        series["T_handle_tip"].append(float(T_handle[handle_mid_idx]))
        series["T_handle_profile"].append(T_handle.tolist())

    row_area_arr = np.array(row_area)

    for step in range(n_steps):
        t_now = step * dt
        if record_series and t_now >= next_sample_t:
            record_frame(t_now)
            next_sample_t += series_dt

        h_open = H_AIR_TOP_OPEN + EVAP_COEFF * max(0.0, T_liq - T_AMB) / (T0_LIQUID - T_AMB)
        flux_rows = float(np.sum(H_LIQ * row_area_arr * (T_liq - T_grid[:, 0])))
        flux_open = h_open * open_area * (T_liq - T_AMB)
        flux_lid = H_LIQ * lid_area * (T_liq - T_lid[0]) if has_lid else 0.0
        T_liq_new = T_liq - dt / c_liq * (flux_rows + flux_open + flux_lid)

        T_radial = np.empty_like(T_grid)
        for i in range(n_z):
            T_radial[i] = _step_chain(T_grid[i], dx_grid[i], k_cell, rho_cell, cp_cell, row_area[i], dt,
                                       T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)

        C_grid = rho_cell[None, :] * cp_cell[None, :] * dx_grid * row_area_arr[:, None]
        T_grid_new = _axial_diffuse(T_radial, k_cell, C_grid, row_r_center, dx_grid, dz, dt)

        body_temp = T_grid[mid_row_idx, -1]
        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=T_AMB, h_right=0.0,
                                    T_left_dirichlet=body_temp, T_right_dirichlet=body_temp,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)
        if has_lid:
            T_lid_new = _step_chain(T_lid, dx_lid, k_lid_arr, rho_lid_arr, cp_lid_arr, lid_area, dt,
                                     T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
        else:
            T_lid_new = None

        finite = (np.isfinite(T_liq_new) and np.all(np.isfinite(T_grid_new))
                  and np.all(np.isfinite(T_handle_new))
                  and (T_lid_new is None or np.all(np.isfinite(T_lid_new))))
        if not finite:
            return dict(valid=False)

        T_liq, T_grid, T_handle = T_liq_new, T_grid_new, T_handle_new
        if has_lid:
            T_lid = T_lid_new

        if not recorded and (t_now + dt) >= record_at:
            surfaces = T_grid[:, -1].tolist()
            if has_lid:
                surfaces.append(T_lid[-1])
            touch_temp_at = float(max(surfaces))
            handle_temp_at = float(T_handle[handle_mid_idx])
            recorded = True

    if record_series:
        record_frame(n_steps * dt)

    mass_struct = sum(row_area[i] * (row_t_wall_mm[i] / 1000.0) * rho_struct for i in range(n_z))
    mass_ins = sum(row_area[i] * (t_gap_mm / 1000.0) * rho_ins for i in range(n_z))
    mass_handle = A_h * L_h * rho_handle
    mass_lid = lid_area * (t_lid_mm / 1000.0) * rho_lid if has_lid else 0.0
    mass_kg = mass_struct + mass_ins + mass_handle + mass_lid

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
        liquid_volume_L=float(vol_liq * 1000.0),
        series=series,
        geometry=dict(r_base_m=r_base, r_mid_m=r_mid, r_rim_m=r_rim, height_m=HEIGHT,
                       has_lid=has_lid, lid_coverage_frac=lid_coverage_frac,
                       handle_length_m=L_h, handle_diameter_m=d_h),
    )


# ---------------------------------------------------------------------------
# v5 (EXPERIMENT_LOG §58): a real, conservative handle<->wall coupling plus
# the one genuine angular asymmetry this model has -- a small "patch" of the
# circumference near the handle attachment, distinct from the rest of the
# ring ("bulk"). Direct user follow-up after v4 shipped: is 3D diffusion
# possible, or is the model stuck at axisymmetric (height x radial)? Answer
# worked out with the user first -- full dense 3D (a real angle dimension
# resolved everywhere) would multiply v4's own cost by however many angular
# nodes are added, for a payoff that's ONLY REAL near the handle: every
# other part of this model (uniform ambient convection, a solid-of-
# revolution body, a well-mixed lumped liquid) is already correctly
# azimuthally uniform, so a full ring would spend most of its compute
# reproducing the same value at every angle. The handle is the one place
# angle actually matters, and even there v4 didn't model it as a real 3D
# effect at all -- the handle was Dirichlet-coupled to an abstract
# "representative" body temperature, drawing heat for free (not subtracted
# from the wall's own energy balance anywhere).
#
# So this fixes BOTH gaps at once, cheaply: the wall is split into two
# angular cells -- `T_patch` (a wedge of the circumference, HANDLE_PATCH_
# FRAC wide, centered on the handle's own fixed attachment angle) and
# `T_bulk` (the rest) -- not a full ring. Cost: roughly 2x v4 (one extra
# full body solve), not the Nx a dense ring would need. The handle now
# draws real, conservative conductive flux from the patch (subtracted from
# the patch's own energy balance, same step it's added to the handle) via
# a new solid-solid contact conductance, instead of a free Dirichlet clamp.
# Same 14-dim parameter vector as v3/v4 -- still an internal-solve upgrade.
# ---------------------------------------------------------------------------

HANDLE_PATCH_FRAC = 0.15  # fraction of the mug's circumference treated as
                          # "near the handle" -- an order-of-magnitude
                          # engineering stand-in for the handle's real
                          # thermal footprint on the wall (like EVAP_COEFF/
                          # H_AIR_TOP_OPEN above), not measured against a
                          # real handle's actual contact patch. ~54 degrees.
H_HANDLE_CONTACT = 200.0  # W/(m2 K), solid-solid contact conductance at the
                          # handle's two root attachments -- another
                          # order-of-magnitude stand-in (a snug molded/
                          # press-fit joint), not a measured value.
N_LATERAL_EDGES = 2       # a wedge patch touches the rest of the ring at
                          # exactly 2 azimuthal boundaries -- this one IS
                          # exact, not an engineering guess.


def _lateral_exchange(T_a, T_b, G, C_a, C_b, dt):
    """Explicit 2-node exchange between two same-shaped (N_AXIAL_NODES,
    n_cell) grids representing angularly-separated regions of the same
    ring -- the coarsest possible azimuthal discretization (2 cells: patch
    vs. bulk, not a full ring), because this model has exactly one real
    source of angular asymmetry (the handle) and a full ring elsewhere
    would only ever compute the same value at every angle. Same explicit,
    stability-bound-computed-directly style as _axial_diffuse (a 2-node
    exchange's own stability limit: dt < 2/(G*(1/C_a+1/C_b)), same 0.9
    safety margin)."""
    rate = G * (1.0 / C_a + 1.0 / C_b)
    mask = rate > 0
    dt_max = float(np.min(1.0 / rate[mask])) if np.any(mask) else np.inf
    n_sub = max(1, math.ceil(dt / (0.9 * dt_max))) if np.isfinite(dt_max) else 1
    dt_sub = dt / n_sub
    Ta, Tb = T_a.copy(), T_b.copy()
    for _ in range(n_sub):
        flux = G * (Tb - Ta)
        Ta = Ta + dt_sub / C_a * flux
        Tb = Tb - dt_sub / C_b * flux
    return Ta, Tb


def simulate_v5(r_base_mm, r_mid_mm, r_rim_mm,
                 t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 t_gap_mm, insulation_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 lid_coverage_frac, t_lid_mm, lid_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0,
                 record_series=False, series_dt=None):
    """See this module's v5 header comment. Identical parameter vector to
    v3/v4 -- only the wall's internal solve changes again: v4's single
    (N_AXIAL_NODES x radial) field is split into `T_bulk` (most of the
    circumference) and `T_patch` (a HANDLE_PATCH_FRAC-wide wedge at the
    handle's own attachment angle), exchanging heat laterally, with the
    handle now a real conductive drain on the patch instead of a free
    Dirichlet clamp on an abstract body temperature."""
    if series_dt is None:
        series_dt = dt
    if min(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm,
           t_gap_mm, handle_length_mm, handle_diameter_mm, t_lid_mm) <= 0:
        return dict(valid=False)
    lid_coverage_frac = float(np.clip(lid_coverage_frac, 0.0, 1.0))

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_ins, rho_ins, cp_ins = material_props(insulation_material_idx, INSULATION_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)
    k_lid, rho_lid, cp_lid = material_props(lid_material_idx, STRUCTURAL_MATERIALS)

    r_base, r_mid, r_rim = r_base_mm / 1000.0, r_mid_mm / 1000.0, r_rim_mm / 1000.0
    height = HEIGHT
    n_z = N_AXIAL_NODES
    edges = [i * height / n_z for i in range(n_z + 1)]
    edge_r = [radius_at_height(r_base, r_mid, r_rim, height, z) for z in edges]
    dz = height / n_z
    row_area_full = np.array([frustum_lateral_area(edge_r[i], edge_r[i + 1], dz) for i in range(n_z)])
    row_r_center = np.array([radius_at_height(r_base, r_mid, r_rim, height, (edges[i] + edges[i + 1]) / 2)
                              for i in range(n_z)])
    vol_liq = sum(frustum_volume(edge_r[i], edge_r[i + 1], dz) for i in range(n_z))
    c_liq = RHO_LIQUID * vol_liq * CP_LIQUID
    a_top = math.pi * r_rim ** 2
    open_area = a_top * (1.0 - lid_coverage_frac)
    lid_area = a_top * lid_coverage_frac

    # patch/bulk split -- SAME materials/thickness/geometry per height
    # (nothing in this model varies by angle except which wedge the handle
    # sits in), only the ANGULAR FRACTION of the ring each accounts for
    # differs. patch_frac + bulk_frac == 1 exactly, so summed areas/masses
    # reproduce v4's own totals (checked below in __main__).
    patch_frac = HANDLE_PATCH_FRAC
    bulk_frac = 1.0 - patch_frac
    row_area_bulk = row_area_full * bulk_frac
    row_area_patch = row_area_full * patch_frac
    patch_half_angle = patch_frac * math.pi  # frac of the FULL circle (2*pi) -> half-width = frac*pi

    row_t_wall_mm = np.array([t_wall_base_mm + (t_wall_rim_mm - t_wall_base_mm)
                               * ((edges[i] + edges[i + 1]) / 2 / height) for i in range(n_z)])

    n_band = N_WALL + N_GAP
    # materials/thicknesses are identical for both wedges (only the angular
    # fraction differs) -- built once, shared.
    k_cell = np.concatenate([np.full(N_WALL, k_struct), np.full(N_GAP, k_ins)])
    rho_cell = np.concatenate([np.full(N_WALL, rho_struct), np.full(N_GAP, rho_ins)])
    cp_cell = np.concatenate([np.full(N_WALL, cp_struct), np.full(N_GAP, cp_ins)])
    dx_grid = np.stack([
        np.concatenate([np.full(N_WALL, (row_t_wall_mm[i] / 1000.0) / N_WALL),
                         np.full(N_GAP, (t_gap_mm / 1000.0) / N_GAP)])
        for i in range(n_z)
    ])  # (n_z, n_band)

    # lateral (azimuthal) patch<->bulk conductance, per (z, radial cell):
    # the real cross-section carrying that flux is the interface strip
    # (row height dz x radial cell thickness dx(x)), doubled for the
    # patch's 2 real edges (N_LATERAL_EDGES). The conduction PATH LENGTH
    # is the one genuinely approximate part (an exact full-ring FV mesh
    # would use equal-sized angular cells; a 2-cell patch/bulk split can't
    # -- the bulk is treated as a near-infinite reservoir right at its own
    # edge, so only the patch's own half-width sets the effective path),
    # flagged the same way EVAP_COEFF/H_AIR_TOP_OPEN are.
    L_char = np.maximum(row_r_center * patch_half_angle, 1e-6)  # (n_z,)
    G_az = k_cell[None, :] * (N_LATERAL_EDGES * dz * dx_grid) / L_char[:, None]  # (n_z, n_band)

    T_bulk = np.full((n_z, n_band), T0_LIQUID)
    T_patch = np.full((n_z, n_band), T0_LIQUID)
    T_liq = T0_LIQUID
    mid_row_idx = n_z // 2

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)
    handle_mid_idx = N_HANDLE // 2
    # solid-solid contact conductance at each handle root (dx_h/k_h are
    # uniform arrays, so both ends use the same formula) -- same series
    # (contact film) + (half the end cell's own conduction resistance)
    # convention _step_chain already uses for the wall<->liquid/air links.
    G_handle_end = 1.0 / (1.0 / (H_HANDLE_CONTACT * A_h) + dx_h[0] / (2 * k_h[0] * A_h))

    has_lid = lid_area > 1e-9
    if has_lid:
        t_lid = t_lid_mm / 1000.0
        dx_lid = np.full(N_LID, t_lid / N_LID)
        k_lid_arr = np.full(N_LID, k_lid)
        rho_lid_arr = np.full(N_LID, rho_lid)
        cp_lid_arr = np.full(N_LID, cp_lid)
        T_lid = np.full(N_LID, T_AMB)
    else:
        T_lid = None

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False
    series = None
    if record_series:
        series = {"t_s": [], "T_liq": [], "T_bulk_outer": [], "T_patch_outer": [],
                  "T_rim_outer": [], "T_base_outer": [],
                  "T_lid_outer": [], "T_handle_tip": [], "T_handle_profile": []}
        next_sample_t = 0.0

    def record_frame(t_now):
        series["t_s"].append(t_now)
        series["T_liq"].append(float(T_liq))
        series["T_bulk_outer"].append(T_bulk[:, -1].tolist())
        series["T_patch_outer"].append(T_patch[:, -1].tolist())
        series["T_rim_outer"].append(float(T_bulk[-1, -1]))  # convenience aliases (bulk = "away from handle")
        series["T_base_outer"].append(float(T_bulk[0, -1]))
        series["T_lid_outer"].append(float(T_lid[-1]) if has_lid else None)
        series["T_handle_tip"].append(float(T_handle[handle_mid_idx]))
        series["T_handle_profile"].append(T_handle.tolist())

    for step in range(n_steps):
        t_now = step * dt
        if record_series and t_now >= next_sample_t:
            record_frame(t_now)
            next_sample_t += series_dt

        h_open = H_AIR_TOP_OPEN + EVAP_COEFF * max(0.0, T_liq - T_AMB) / (T0_LIQUID - T_AMB)
        flux_bulk = float(np.sum(H_LIQ * row_area_bulk * (T_liq - T_bulk[:, 0])))
        flux_patch = float(np.sum(H_LIQ * row_area_patch * (T_liq - T_patch[:, 0])))
        flux_open = h_open * open_area * (T_liq - T_AMB)
        flux_lid = H_LIQ * lid_area * (T_liq - T_lid[0]) if has_lid else 0.0
        T_liq_new = T_liq - dt / c_liq * (flux_bulk + flux_patch + flux_open + flux_lid)

        # handle<->patch coupling references: BOTH sides use the OTHER's
        # value from the START of this step (same Jacobi-style convention
        # every other cross-chain coupling in this model already uses --
        # e.g. the wall rows above read T_liq from before this step's own
        # update). Symmetric handle (both roots -> the same patch
        # reference) means T_handle[0] == T_handle[-1] always (§56), so
        # either end works as "the" handle-side reference.
        patch_local = T_patch[mid_row_idx, -1]
        handle_local = T_handle[0]

        T_bulk_radial = np.empty_like(T_bulk)
        T_patch_radial = np.empty_like(T_patch)
        for i in range(n_z):
            T_bulk_radial[i] = _step_chain(T_bulk[i], dx_grid[i], k_cell, rho_cell, cp_cell, row_area_bulk[i], dt,
                                            T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
            if i == mid_row_idx:
                # real conductive drain to the handle, folded in as a SECOND
                # implicit parallel conductance at this row's own outer
                # (ambient-facing) boundary -- §58's fix for the instability
                # an earlier explicit version of this hit: that cell's own
                # thermal mass is tiny (it's one radial cell of one row of
                # a small angular wedge), so pulling a real flux out of it
                # as an explicit correction blew up immediately; solving it
                # as part of the SAME implicit system this row's own
                # ambient-convection term already uses is unconditionally
                # stable regardless of how small that mass is.
                T_patch_radial[i] = _step_chain(T_patch[i], dx_grid[i], k_cell, rho_cell, cp_cell,
                                                 row_area_patch[i], dt,
                                                 T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ,
                                                 extra_right_G=2 * G_handle_end, extra_right_T=handle_local)
            else:
                T_patch_radial[i] = _step_chain(T_patch[i], dx_grid[i], k_cell, rho_cell, cp_cell,
                                                 row_area_patch[i], dt,
                                                 T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)

        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=patch_local, h_right=H_HANDLE_CONTACT,
                                    T_left_conv=patch_local, h_left=H_HANDLE_CONTACT,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)

        C_bulk_grid = rho_cell[None, :] * cp_cell[None, :] * dx_grid * row_area_bulk[:, None]
        C_patch_grid = rho_cell[None, :] * cp_cell[None, :] * dx_grid * row_area_patch[:, None]
        T_bulk_axial = _axial_diffuse(T_bulk_radial, k_cell, C_bulk_grid, row_r_center * bulk_frac, dx_grid, dz, dt)
        T_patch_axial = _axial_diffuse(T_patch_radial, k_cell, C_patch_grid, row_r_center * patch_frac, dx_grid, dz, dt)

        T_bulk_new, T_patch_new = _lateral_exchange(T_bulk_axial, T_patch_axial, G_az, C_bulk_grid, C_patch_grid, dt)

        if has_lid:
            T_lid_new = _step_chain(T_lid, dx_lid, k_lid_arr, rho_lid_arr, cp_lid_arr, lid_area, dt,
                                     T_amb_right=T_AMB, h_right=H_AIR, T_left_conv=T_liq, h_left=H_LIQ)
        else:
            T_lid_new = None

        finite = (np.isfinite(T_liq_new) and np.all(np.isfinite(T_bulk_new)) and np.all(np.isfinite(T_patch_new))
                  and np.all(np.isfinite(T_handle_new))
                  and (T_lid_new is None or np.all(np.isfinite(T_lid_new))))
        if not finite:
            return dict(valid=False)

        T_liq, T_bulk, T_patch, T_handle = T_liq_new, T_bulk_new, T_patch_new, T_handle_new
        if has_lid:
            T_lid = T_lid_new

        if not recorded and (t_now + dt) >= record_at:
            surfaces = T_bulk[:, -1].tolist() + T_patch[:, -1].tolist()
            if has_lid:
                surfaces.append(T_lid[-1])
            touch_temp_at = float(max(surfaces))
            handle_temp_at = float(T_handle[handle_mid_idx])
            recorded = True

    if record_series:
        record_frame(n_steps * dt)

    mass_struct = sum(row_area_full[i] * (row_t_wall_mm[i] / 1000.0) * rho_struct for i in range(n_z))
    mass_ins = sum(row_area_full[i] * (t_gap_mm / 1000.0) * rho_ins for i in range(n_z))
    mass_handle = A_h * L_h * rho_handle
    mass_lid = lid_area * (t_lid_mm / 1000.0) * rho_lid if has_lid else 0.0
    mass_kg = mass_struct + mass_ins + mass_handle + mass_lid

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
        liquid_volume_L=float(vol_liq * 1000.0),
        series=series,
        geometry=dict(r_base_m=r_base, r_mid_m=r_mid, r_rim_m=r_rim, height_m=HEIGHT,
                       has_lid=has_lid, lid_coverage_frac=lid_coverage_frac,
                       handle_length_m=L_h, handle_diameter_m=d_h,
                       handle_patch_frac=patch_frac),
    )


# ---------------------------------------------------------------------------
# v6 (EXPERIMENT_LOG §59): a full domain reset, not another incremental
# patch. Direct user review of v5 found the whole foundation was wrong, not
# just under-resolved:
#
#   1. EVERY sampled mug had a double-wall insulation gap (t_gap_mm) and a
#      partial lid, since v3 (EXPERIMENT_LOG §50) -- this domain has
#      quietly been "insulated travel mug," never actually "mug." Direct
#      user call: drop the whole double-wall/lid architecture, not
#      reweight it.
#   2. The wall started AT FILL TEMPERATURE (T0_LIQUID), not ambient --
#      confirmed by inspection, a real bug backwards from the entire point
#      of watching heat spread INTO a cold body (the lid/handle already
#      correctly started at ambient; the wall never did).
#   3. No radiative loss term anywhere -- only convective terms existed.
#   4. `liquid_volume_L` was computed every version back to v3 and never
#      once checked or exposed -- a mug that can't hold coffee passed as
#      "valid" exactly like a normal one.
#   5. v5's 2-cell (patch/bulk) angular split was a cost-motivated
#      approximation, not real 3D -- rejected outright on review ("I'm not
#      interested anymore in your suggestions that the 2d axi-symmetric
#      approach is acceptable").
#
# This version fixes all five at once, since the foundation was being
# rebuilt anyway:
#
# Parameterization: 9 dims (down from 14) -- t_gap_mm, insulation_
# material_idx, lid_coverage_frac, t_lid_mm, lid_material_idx are GONE, not
# defaulted away. One structural material carries the wall end to end.
#
# Grid: genuinely 3D -- N_AXIAL_NODES (height) x N_THETA (a real periodic
# ring, not 2 cells) x N_WALL_CELLS (radial, through one homogeneous
# material now instead of two, which is what makes real angular
# resolution affordable -- see below).
#
# The performance move that keeps this from being an N_THETA-x slowdown:
# every azimuthal wedge at a given height shares the IDENTICAL banded
# conduction matrix (same geometry/material/boundary type regardless of
# angle) except the one wedge the handle attaches to. _step_chain now
# accepts a 2D (n_cell, n_theta) T and solves every column in ONE
# scipy.linalg.solve_banded call via its native multiple-right-hand-side
# support, instead of one call per column. The radial step's call count
# per timestep stays at N_AXIAL_NODES, same order as v4/v5, regardless of
# N_THETA -- the handle's own column gets one small separate single-column
# re-solve carrying its extra (real, conservative, §58-style) terms.
#
# Initial condition: the wall grid starts at T_AMB everywhere (matching
# how the handle/lid already correctly did) -- liquid still starts at
# T0_LIQUID, since that part was always right (the coffee is what's hot,
# not the vessel).
#
# Radiative loss: Stefan-Boltzmann, via the standard implicit-linearization
# trick (h_rad = eps*sigma*(T_prev+T_amb)*(T_prev^2+T_amb^2), added
# directly into the existing linear h_right/ambient term) -- keeps it
# compatible with the shared-matrix batched solve (a per-row RING-AVERAGE
# h_rad feeds the shared solve; the handle's own column re-solve gets its
# own exact local value) and avoids re-triggering the exact explicit-
# instability-against-a-tiny-cell bug §58 hit with the handle's flux.
# Applied to the wall's outer surface and the liquid's own open top
# (always fully open now -- no lid).
#
# Liquid volume: a real 5th target (TARGET_NAMES_V6), and validity now
# also requires it clear MIN_LIQUID_VOLUME_L -- a design that can't hold a
# meaningful amount of liquid is rejected, not silently included.
# ---------------------------------------------------------------------------

N_THETA = 24                # real periodic ring resolution -- affordable
                            # specifically BECAUSE of the shared-matrix
                            # batched solve above (cost is ~independent of
                            # N_THETA); checked directly via the speed test
                            # below, not assumed.
N_WALL_CELLS = 24           # radial resolution through the single wall
                            # material (replaces N_WALL+N_GAP=60 -- one
                            # homogeneous layer needs far fewer cells than
                            # two very-different-conductivity ones did).
EMISSIVITY_SURFACE = 0.9    # order-of-magnitude stand-in (most non-
                            # polished-metal mug materials -- ceramic/
                            # glass/plastic -- sit around here; a mirror-
                            # polished steel mug would radiate far less, a
                            # real per-material table not built this pass,
                            # same scope-limit spirit as EVAP_COEFF/
                            # H_AIR_TOP_OPEN).
SIGMA_SB = 5.670374419e-8   # W/(m^2 K^4), Stefan-Boltzmann constant -- a
                            # real physical constant, not an engineering
                            # stand-in.
MIN_LIQUID_VOLUME_L = 0.05  # ~one espresso shot -- a design below this
                            # can't usefully hold a beverage, rejected as
                            # invalid rather than silently included.

PARAM_DIM_V6 = 9
PARAM_NAMES_V6 = [
    "r_base_mm", "r_mid_mm", "r_rim_mm",
    "t_wall_rim_mm", "t_wall_base_mm", "struct_material_idx",
    "handle_length_mm", "handle_diameter_mm", "handle_material_idx",
]
TARGET_NAMES_V6 = ["temp_at_2h_C", "mass_kg", "touch_temp_60s_C", "handle_temp_60s_C", "liquid_volume_L"]


def _radiative_h(T_prev_C, T_amb_C=T_AMB, emissivity=EMISSIVITY_SURFACE):
    """Linearized radiative 'convective-equivalent' coefficient: the
    standard trick for keeping Stefan-Boltzmann's real T^4 dependence
    compatible with an otherwise-linear implicit solve. Exact at T_prev; a
    good approximation for the next small step since T_prev is only
    dt=5s stale. Units match h_left/h_right (W/(m^2 K)), so it can be
    added DIRECTLY onto an existing convective coefficient instead of
    needing its own boundary-condition machinery."""
    T_prev_K = T_prev_C + 273.15
    T_amb_K = T_amb_C + 273.15
    return emissivity * SIGMA_SB * (T_prev_K + T_amb_K) * (T_prev_K ** 2 + T_amb_K ** 2)


def _axial_diffuse_3d(T, k_cell, C_grid, r_center_wedge, dx_grid, dz, dt):
    """3D (n_z, n_cell, n_theta) analogue of _axial_diffuse -- couples
    adjacent AXIAL rows only (azimuthal coupling is _azimuthal_diffuse's
    own separate job). `r_center_wedge` is the row's radius already
    scaled by the wedge's own share of the ring (r/N_THETA, same
    pre-scaling convention v5 used for its patch/bulk fractions), so the
    cross-section this computes is one wedge's own, not the full ring's."""
    n_z = T.shape[0]
    if n_z < 2:
        return T
    r_mid = 0.5 * (r_center_wedge[:-1] + r_center_wedge[1:])
    dx_avg = 0.5 * (dx_grid[:-1, :] + dx_grid[1:, :])
    G = k_cell[None, :] * (2 * math.pi * r_mid[:, None] * dx_avg) / dz  # (n_z-1, n_cell)

    G_sum = np.zeros_like(C_grid)
    G_sum[:-1, :] += G
    G_sum[1:, :] += G
    mask = G_sum > 0
    dt_max = float(np.min(C_grid[mask] / G_sum[mask])) if np.any(mask) else np.inf
    n_sub = max(1, math.ceil(dt / (0.9 * dt_max))) if np.isfinite(dt_max) else 1
    dt_sub = dt / n_sub

    Tc = T.copy()
    Gb = G[:, :, None]
    Cb = C_grid[:, :, None]
    for _ in range(n_sub):
        flux = np.zeros_like(Tc)
        flux[:-1, :, :] += Gb * (Tc[1:, :, :] - Tc[:-1, :, :])
        flux[1:, :, :] += Gb * (Tc[:-1, :, :] - Tc[1:, :, :])
        Tc = Tc + dt_sub / Cb * flux
    return Tc


def _azimuthal_diffuse(T, k_cell, dx_grid, C_grid, row_r_center, dz, n_theta, dt):
    """Real periodic ring diffusion across the azimuthal axis (axis=2) of
    a (n_z, n_cell, n_theta) grid -- §59's replacement for v5's 2-cell
    patch/bulk approximation. Every adjacent pair of wedges at a given
    (z, radial cell) is geometrically IDENTICAL (equal angular width), so
    unlike v5's patch/bulk conductance (which needed an approximate
    characteristic-length stand-in because one "cell" was almost the
    whole ring), this is a real, exact finite-volume discretization: the
    interface between adjacent wedges has area (dz*dx(x)) and the
    conduction path length is the true arc length between wedge centers,
    r(z)*dtheta. Same explicit, stability-bound-computed-directly style as
    _axial_diffuse_3d; periodic neighbors via np.roll instead of open
    ends."""
    dtheta = 2 * math.pi / n_theta
    G = k_cell[None, :] * (dz * dx_grid) / (row_r_center[:, None] * dtheta)  # (n_z, n_cell)
    G_sum = 2 * G  # a uniform ring -- every wedge has exactly 2 identical neighbors
    mask = G_sum > 0
    dt_max = float(np.min(C_grid[mask] / G_sum[mask])) if np.any(mask) else np.inf
    n_sub = max(1, math.ceil(dt / (0.9 * dt_max))) if np.isfinite(dt_max) else 1
    dt_sub = dt / n_sub

    Tc = T.copy()
    Gb = G[:, :, None]
    Cb = C_grid[:, :, None]
    for _ in range(n_sub):
        flux = Gb * (np.roll(Tc, -1, axis=2) + np.roll(Tc, 1, axis=2) - 2 * Tc)
        Tc = Tc + dt_sub / Cb * flux
    return Tc


def simulate_v6(r_base_mm, r_mid_mm, r_rim_mm,
                 t_wall_rim_mm, t_wall_base_mm, struct_material_idx,
                 handle_length_mm, handle_diameter_mm, handle_material_idx,
                 dt=5.0, t_max=2 * 3600.0, record_at=60.0,
                 record_series=False, series_dt=None):
    """See this module's v6 header comment. 9-dim parameter vector (no
    lid/insulation-gap at all), a genuine 3D (axial x azimuthal x radial)
    grid via a shared-matrix batched implicit radial solve, an ambient
    wall initial condition, real radiative loss, and liquid_volume_L as a
    checked 5th target."""
    if series_dt is None:
        series_dt = dt
    if min(r_base_mm, r_mid_mm, r_rim_mm, t_wall_rim_mm, t_wall_base_mm,
           handle_length_mm, handle_diameter_mm) <= 0:
        return dict(valid=False)

    k_struct, rho_struct, cp_struct = material_props(struct_material_idx, STRUCTURAL_MATERIALS)
    k_handle, rho_handle, cp_handle = material_props(handle_material_idx, HANDLE_MATERIALS)

    r_base, r_mid, r_rim = r_base_mm / 1000.0, r_mid_mm / 1000.0, r_rim_mm / 1000.0
    height = HEIGHT
    n_z = N_AXIAL_NODES
    n_theta = N_THETA
    edges = [i * height / n_z for i in range(n_z + 1)]
    edge_r = [radius_at_height(r_base, r_mid, r_rim, height, z) for z in edges]
    dz = height / n_z
    row_area_full = np.array([frustum_lateral_area(edge_r[i], edge_r[i + 1], dz) for i in range(n_z)])
    row_area_wedge = row_area_full / n_theta
    row_r_center = np.array([radius_at_height(r_base, r_mid, r_rim, height, (edges[i] + edges[i + 1]) / 2)
                              for i in range(n_z)])
    vol_liq = sum(frustum_volume(edge_r[i], edge_r[i + 1], dz) for i in range(n_z))
    liquid_volume_L = vol_liq * 1000.0
    if liquid_volume_L < MIN_LIQUID_VOLUME_L:
        return dict(valid=False)
    c_liq = RHO_LIQUID * vol_liq * CP_LIQUID
    a_top = math.pi * r_rim ** 2  # always fully open now -- no lid

    row_t_wall_mm = np.array([t_wall_base_mm + (t_wall_rim_mm - t_wall_base_mm)
                               * ((edges[i] + edges[i + 1]) / 2 / height) for i in range(n_z)])

    n_cell = N_WALL_CELLS
    k_cell = np.full(n_cell, k_struct)
    rho_cell = np.full(n_cell, rho_struct)
    cp_cell = np.full(n_cell, cp_struct)
    dx_grid = np.tile((row_t_wall_mm / 1000.0 / n_cell)[:, None], (1, n_cell))  # (n_z, n_cell)

    # T_grid axes: (axial row, radial cell, azimuthal wedge) -- chosen so
    # T_grid[z] is directly (n_cell, n_theta), exactly what the batched
    # _step_chain call wants with no reshape.
    T_grid = np.full((n_z, n_cell, n_theta), T_AMB)  # §59: wall starts at AMBIENT, not fill temp
    T_liq = T0_LIQUID
    mid_row_idx = n_z // 2
    theta_handle_idx = 0  # matches the viewer's own fixed +X handle placement

    L_h = handle_length_mm / 1000.0
    d_h = handle_diameter_mm / 1000.0
    A_h = math.pi * (d_h / 2) ** 2
    P_h = math.pi * d_h
    dx_h = np.full(N_HANDLE, L_h / N_HANDLE)
    k_h = np.full(N_HANDLE, k_handle)
    rho_h = np.full(N_HANDLE, rho_handle)
    cp_h = np.full(N_HANDLE, cp_handle)
    T_handle = np.full(N_HANDLE, T_AMB)
    handle_mid_idx = N_HANDLE // 2
    G_handle_end = 1.0 / (1.0 / (H_HANDLE_CONTACT * A_h) + dx_h[0] / (2 * k_h[0] * A_h))

    C_grid = rho_cell[None, :] * cp_cell[None, :] * dx_grid * row_area_wedge[:, None]  # (n_z, n_cell)
    r_center_wedge = row_r_center / n_theta

    n_steps = int(t_max / dt)
    touch_temp_at = None
    handle_temp_at = None
    recorded = False
    series = None
    if record_series:
        series = {"t_s": [], "T_liq": [], "T_wall_outer": [], "T_rim_outer": [], "T_base_outer": [],
                  "T_handle_tip": [], "T_handle_profile": []}
        next_sample_t = 0.0

    def record_frame(t_now):
        series["t_s"].append(t_now)
        series["T_liq"].append(float(T_liq))
        series["T_wall_outer"].append(T_grid[:, -1, :].tolist())  # (n_z, n_theta) per frame
        series["T_rim_outer"].append(float(np.mean(T_grid[-1, -1, :])))
        series["T_base_outer"].append(float(np.mean(T_grid[0, -1, :])))
        series["T_handle_tip"].append(float(T_handle[handle_mid_idx]))
        series["T_handle_profile"].append(T_handle.tolist())

    for step in range(n_steps):
        t_now = step * dt
        if record_series and t_now >= next_sample_t:
            record_frame(t_now)
            next_sample_t += series_dt

        # radiative-linearized effective coefficient for the OPEN LIQUID
        # TOP -- the liquid has real thermal mass, so (unlike the thin
        # wall cells) this stays fully explicit and safe, the same
        # convention every prior version already used for the liquid's
        # own update.
        h_open = (H_AIR_TOP_OPEN + EVAP_COEFF * max(0.0, T_liq - T_AMB) / (T0_LIQUID - T_AMB)
                  + _radiative_h(T_liq))
        flux_wall = float(np.sum(H_LIQ * row_area_wedge[:, None] * (T_liq - T_grid[:, 0, :])))
        flux_open = h_open * a_top * (T_liq - T_AMB)
        T_liq_new = T_liq - dt / c_liq * (flux_wall + flux_open)

        # radial implicit step -- ONE shared-matrix batched solve per
        # axial row (all n_theta columns at once), using a per-row
        # RING-AVERAGE radiative coefficient (see this section's own
        # header comment); the handle's own column gets an exact
        # single-column re-solve right after.
        T_radial = np.empty_like(T_grid)
        for i in range(n_z):
            h_rad_avg = float(np.mean(_radiative_h(T_grid[i, -1, :])))
            T_radial[i] = _step_chain(T_grid[i], dx_grid[i], k_cell, rho_cell, cp_cell, row_area_wedge[i], dt,
                                       T_amb_right=T_AMB, h_right=H_AIR + h_rad_avg,
                                       T_left_conv=T_liq, h_left=H_LIQ)

        patch_local = T_grid[mid_row_idx, -1, theta_handle_idx]
        handle_local = T_handle[0]
        h_rad_local = _radiative_h(patch_local)
        T_radial[mid_row_idx, :, theta_handle_idx] = _step_chain(
            T_grid[mid_row_idx, :, theta_handle_idx], dx_grid[mid_row_idx], k_cell, rho_cell, cp_cell,
            row_area_wedge[mid_row_idx], dt,
            T_amb_right=T_AMB, h_right=H_AIR + h_rad_local, T_left_conv=T_liq, h_left=H_LIQ,
            extra_right_G=2 * G_handle_end, extra_right_T=handle_local)

        T_handle_new = _step_chain(T_handle, dx_h, k_h, rho_h, cp_h, A_h, dt,
                                    T_amb_right=patch_local, h_right=H_HANDLE_CONTACT,
                                    T_left_conv=patch_local, h_left=H_HANDLE_CONTACT,
                                    lateral_hP=H_AIR * P_h, T_lateral=T_AMB)

        T_axial = _axial_diffuse_3d(T_radial, k_cell, C_grid, r_center_wedge, dx_grid, dz, dt)
        T_new = _azimuthal_diffuse(T_axial, k_cell, dx_grid, C_grid, row_r_center, dz, n_theta, dt)

        finite = (np.isfinite(T_liq_new) and np.all(np.isfinite(T_new)) and np.all(np.isfinite(T_handle_new)))
        if not finite:
            return dict(valid=False)

        T_liq, T_grid, T_handle = T_liq_new, T_new, T_handle_new

        if not recorded and (t_now + dt) >= record_at:
            touch_temp_at = float(np.max(T_grid[:, -1, :]))
            handle_temp_at = float(T_handle[handle_mid_idx])
            recorded = True

    if record_series:
        record_frame(n_steps * dt)

    mass_struct = sum(row_area_full[i] * (row_t_wall_mm[i] / 1000.0) * rho_struct for i in range(n_z))
    mass_handle = A_h * L_h * rho_handle
    mass_kg = mass_struct + mass_handle

    return dict(
        valid=True,
        final_liq_temp_C=float(T_liq),
        mass_kg=float(mass_kg),
        touch_temp_at_C=touch_temp_at,
        handle_temp_at_C=handle_temp_at,
        liquid_volume_L=float(liquid_volume_L),
        series=series,
        geometry=dict(r_base_m=r_base, r_mid_m=r_mid, r_rim_m=r_rim, height_m=HEIGHT,
                       handle_length_m=L_h, handle_diameter_m=d_h, n_theta=n_theta),
    )


if __name__ == "__main__":
    print("=== Verification: near-massless wall/gap vs. analytic RC limit ===")
    for (tw, tg, kg) in [(1.0, 5.0, 0.03), (2.0, 10.0, 0.01)]:
        a = analytic_time_constant_s(tw, tg, kg)
        r = simulate(tw, tg, kg, wall_mass_scale=1e-6, t_max=max(a * 1.5, 14400))
        rt = r["retention_time_s"]
        if rt is None:
            print(f"  t_wall={tw}mm t_gap={tg}mm k_gap={kg}: never reached threshold (analytic={a:.1f}s)")
        else:
            print(f"  t_wall={tw}mm t_gap={tg}mm k_gap={kg}: "
                  f"FD(massless)={rt:.1f}s  analytic={a:.1f}s  "
                  f"rel_err={(rt-a)/a*100:.2f}%")

    print("\n=== Sweep 1: insulation thickness (t_wall=1mm, k_gap=0.03 foam-like) ===")
    print(f"{'t_gap_mm':>9} {'retention_s':>12} {'retention_min':>14} {'mass_kg':>9} {'touch60s_C':>11}")
    for tg in [0.0, 2.0, 5.0, 10.0, 20.0]:
        tg_eff = max(tg, 1e-3)
        r = simulate(1.0, tg_eff, 0.03)
        rt = r["retention_time_s"]
        rt_str = f"{rt:.1f}" if rt is not None else "not reached"
        rt_min_str = f"{rt/60:.1f}" if rt is not None else "-"
        print(f"{tg:>9.1f} {rt_str:>12} {rt_min_str:>14} "
              f"{r['mass_kg']:>9.4f} {r['touch_temp_60s_C']:>11.2f}")

    print("\n=== Sweep 2: insulation quality (t_wall=1mm, t_gap=5mm) ===")
    print(f"{'k_gap':>8} {'retention_s':>12} {'retention_min':>14} {'touch60s_C':>11}")
    for kg in [0.005, 0.015, 0.026, 0.05]:
        r = simulate(1.0, 5.0, kg)
        rt = r["retention_time_s"]
        rt_str = f"{rt:.1f}" if rt is not None else "not reached"
        rt_min_str = f"{rt/60:.1f}" if rt is not None else "-"
        print(f"{kg:>8.3f} {rt_str:>12} {rt_min_str:>14} "
              f"{r['touch_temp_60s_C']:>11.2f}")

    print("\n=== Redundancy check: does touch_temp_60s track retention_time 1:1, or carry independent info? ===")
    rng = np.random.default_rng(0)
    n_samples = 300
    tw = rng.uniform(0.5, 4.0, n_samples)
    tg = rng.uniform(0.5, 20.0, n_samples)
    kg = rng.uniform(0.005, 0.05, n_samples)
    retentions, touches, masses = [], [], []
    n_invalid = 0
    for i in range(n_samples):
        r = simulate(tw[i], tg[i], kg[i])
        if not r["valid"] or r["retention_time_s"] is None:
            n_invalid += 1
            continue
        retentions.append(r["retention_time_s"])
        touches.append(r["touch_temp_60s_C"])
        masses.append(r["mass_kg"])
    retentions = np.array(retentions)
    touches = np.array(touches)
    masses = np.array(masses)
    print(f"  {n_samples} random designs, {n_invalid} invalid/never-reached-threshold")
    print(f"  corr(retention_time, touch_temp_60s) = {np.corrcoef(retentions, touches)[0,1]:.3f}")
    print(f"  corr(retention_time, mass_kg)        = {np.corrcoef(retentions, masses)[0,1]:.3f}")
    print(f"  corr(touch_temp_60s, mass_kg)        = {np.corrcoef(touches, masses)[0,1]:.3f}")

    import time
    t0 = time.time()
    for i in range(50):
        simulate(tw[i], tg[i], kg[i])
    dt_batch = time.time() - t0
    print(f"\n=== Speed: {dt_batch/50*1000:.2f} ms/candidate (host CPU, single-threaded, N=30 cells) ===")

    print("\n=== v2 regression check: reduced to v1's case (equal bands, steel/foam, no handle feedback) ===")
    v1r = simulate(1.0, 5.0, 0.03, t_max=7200.0)
    v2r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0, t_max=7200.0)
    print(f"  v1 final_liq_temp_C={v1r['final_liq_temp_C']:.2f}  v2 final_liq_temp_C={v2r['final_liq_temp_C']:.2f}  "
          f"(explicit-liquid-coupling scheme, close-but-not-identical expected)")
    print(f"  v1 touch_temp_60s_C={v1r['touch_temp_60s_C']:.2f}  v2 touch_temp_at_C={v2r['touch_temp_at_C']:.2f}")

    print("\n=== v2 handle sanity: length sweep (steel handle, diameter=6mm -- worst case material) ===")
    for L in [5.0, 20.0, 50.0, 100.0]:
        r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, L, 6.0, 4.0, t_max=7200.0)
        print(f"  length={L:>6.1f}mm  handle_temp_60s={r['handle_temp_at_C']:.2f}C")

    print("\n=== v2 handle sanity: material sweep (length=50mm, diameter=6mm) ===")
    for name, k, rho, cp in HANDLE_MATERIALS:
        idx = [i for i, t in enumerate(HANDLE_MATERIALS) if t[0] == name][0]
        r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 50.0, 6.0, float(idx), t_max=7200.0)
        print(f"  {name:>16} (k={k:>5.2f})  handle_temp_60s={r['handle_temp_at_C']:.2f}C")

    print("\n=== v2 handle sanity: diameter sweep (length=50mm, steel) ===")
    for d in [2.0, 6.0, 12.0, 20.0]:
        r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 50.0, d, 4.0, t_max=7200.0)
        print(f"  diameter={d:>5.1f}mm  handle_temp_60s={r['handle_temp_at_C']:.2f}C  mass_kg={r['mass_kg']:.4f}")

    print("\n=== v2 speed ===")
    t0 = time.time()
    for _ in range(50):
        simulate_v2(1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/50*1000:.2f} ms/candidate")

    R_MM = R_INNER * 1000.0  # 35.0 -- v1/v2's fixed cylinder radius, for regression comparisons

    print("\n=== v3 vs. v2: a fully-sealed, well-insulated lid should APPROACH (not exactly match) v2's zero-top-loss idealization ===")
    v2r = simulate_v2(1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0, t_max=7200.0)
    v3r = simulate_v3(R_MM, R_MM, R_MM, 1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0,
                       1.0, 5.0, 0.0, t_max=7200.0)  # lid_coverage=1, t_lid=5mm, plastic (lowest k available)
    print(f"  v2 (zero top loss, unphysical idealization)  final_liq_temp_C={v2r['final_liq_temp_C']:.2f}  touch_temp_at_C={v2r['touch_temp_at_C']:.2f}")
    print(f"  v3 (realistic sealed plastic lid)             final_liq_temp_C={v3r['final_liq_temp_C']:.2f}  touch_temp_at_C={v3r['touch_temp_at_C']:.2f}  "
          f"(v3 should run a bit COOLER than v2 -- a real lid still conducts some heat, v2 assumes none)")

    print("\n=== v3 lid sanity: coverage sweep (straight cylinder, same wall/gap as above) ===")
    for cov in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = simulate_v3(R_MM, R_MM, R_MM, 1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0,
                         cov, 3.0, 3.0, t_max=7200.0)  # t_lid=3mm steel lid when present
        print(f"  coverage={cov:.2f}  temp_at_2h_C={r['final_liq_temp_C']:.2f}  touch_temp_at_C={r['touch_temp_at_C']:.2f}")

    print("\n=== v3 body-shape sanity: belly sweep (r_base=r_rim=35mm fixed, r_mid varies) ===")
    for r_mid in [25.0, 35.0, 45.0, 60.0]:
        r = simulate_v3(35.0, r_mid, 35.0, 1.0, 1.0, 3.0, 5.0, 3.0, 30.0, 6.0, 4.0,
                         0.5, 3.0, 3.0, t_max=7200.0)
        print(f"  r_mid={r_mid:>5.1f}mm  liquid_volume_L={r['liquid_volume_L']:.3f}  "
              f"mass_kg={r['mass_kg']:.4f}  temp_at_2h_C={r['final_liq_temp_C']:.2f}")

    print("\n=== v3 speed ===")
    t0 = time.time()
    for _ in range(50):
        simulate_v3(30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, 0.6, 3.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/50*1000:.2f} ms/candidate")

    print("\n=== v4 unit check: _axial_diffuse on a uniform grid is a true no-op ===")
    n_cell_test = N_WALL + N_GAP
    k_cell_t = np.concatenate([np.full(N_WALL, K_STEEL), np.full(N_GAP, 0.03)])
    rho_cell_t = np.concatenate([np.full(N_WALL, RHO_STEEL), np.full(N_GAP, RHO_INSULATION)])
    cp_cell_t = np.concatenate([np.full(N_WALL, CP_STEEL), np.full(N_GAP, CP_INSULATION)])
    dx_row_t = np.concatenate([np.full(N_WALL, 0.001 / N_WALL), np.full(N_GAP, 0.005 / N_GAP)])
    dx_grid_t = np.tile(dx_row_t, (N_AXIAL_NODES, 1))
    r_center_t = np.full(N_AXIAL_NODES, R_INNER)
    area_t = np.full(N_AXIAL_NODES, A_INNER / N_AXIAL_NODES)
    C_grid_t = rho_cell_t[None, :] * cp_cell_t[None, :] * dx_grid_t * area_t[:, None]
    T_uniform = np.full((N_AXIAL_NODES, n_cell_test), 55.0)
    out_uniform = _axial_diffuse(T_uniform, k_cell_t, C_grid_t, r_center_t, dx_grid_t, HEIGHT / N_AXIAL_NODES, 5.0)
    print(f"  max deviation from input (should be ~0): {np.max(np.abs(out_uniform - T_uniform)):.2e}")

    print("\n=== v4 unit check: _axial_diffuse on a gradient grid smooths it and conserves energy ===")
    T_grad = np.tile(np.linspace(90.0, 20.0, N_AXIAL_NODES)[:, None], (1, n_cell_test))
    energy_before = float(np.sum(C_grid_t * T_grad))
    out_grad = _axial_diffuse(T_grad, k_cell_t, C_grid_t, r_center_t, dx_grid_t, HEIGHT / N_AXIAL_NODES, 5.0)
    energy_after = float(np.sum(C_grid_t * out_grad))
    spread_before = float(np.std(T_grad[:, -1]))
    spread_after = float(np.std(out_grad[:, -1]))
    print(f"  energy before={energy_before:.6e}  after={energy_after:.6e}  "
          f"rel_diff={(energy_after-energy_before)/energy_before:.2e} (should be ~0, adiabatic ends)")
    print(f"  outer-cell std-dev before={spread_before:.4f}C  after={spread_after:.4f}C "
          f"(should shrink -- diffusion smooths the gradient)")

    print("\n=== v4 reducible case: straight cylinder + uniform wall thickness -- every row is thermally "
          "identical, so axial coupling should be a no-op and v4 should reproduce v3 closely ===")
    v3_uniform = simulate_v3(R_MM, R_MM, R_MM, 1.5, 1.5, 3.0, 5.0, 1.5, 30.0, 6.0, 4.0, 0.5, 3.0, 3.0, t_max=7200.0)
    v4_uniform = simulate_v4(R_MM, R_MM, R_MM, 1.5, 1.5, 3.0, 5.0, 1.5, 30.0, 6.0, 4.0, 0.5, 3.0, 3.0, t_max=7200.0)
    print(f"  v3 (6 independent bands)          final_liq_temp_C={v3_uniform['final_liq_temp_C']:.3f}  "
          f"touch_temp_at_C={v3_uniform['touch_temp_at_C']:.3f}  mass_kg={v3_uniform['mass_kg']:.5f}")
    print(f"  v4 ({N_AXIAL_NODES} coupled rows)              final_liq_temp_C={v4_uniform['final_liq_temp_C']:.3f}  "
          f"touch_temp_at_C={v4_uniform['touch_temp_at_C']:.3f}  mass_kg={v4_uniform['mass_kg']:.5f}")

    print("\n=== v4 physical sanity: thin/poor-conductor wall (so the WALL, not the insulation gap, "
          "dominates outer resistance) with real rim/base thickness asymmetry -- v4's per-row outer "
          f"temps should form a real {N_AXIAL_NODES}-value gradient, not v3's coarse 6-value step ===")
    args_asym = (30.0, 35.0, 40.0,  # r_base, r_mid, r_rim
                 1.0, 6.0, 0.0,      # t_wall_rim_mm=1 (thin), t_wall_base_mm=6 (thick), plastic (poor conductor)
                 1.0, 0.0,           # t_gap_mm=1 (thin, so it doesn't swamp the wall's own resistance), vacuum-like
                 30.0, 6.0, 4.0,     # handle
                 0.0, 3.0, 0.0)      # no lid
    v3_asym = simulate_v3(*args_asym, t_max=300.0, record_series=True, series_dt=None)
    v4_asym = simulate_v4(*args_asym, t_max=300.0, record_series=True, series_dt=None)
    mid_frame_v3 = len(v3_asym["series"]["t_s"]) // 2
    mid_frame_v4 = len(v4_asym["series"]["t_s"]) // 2
    print(f"  t={v3_asym['series']['t_s'][mid_frame_v3]:.0f}s  v3 outer temps (6 bands, base->rim): "
          f"{[round(t, 2) for t in v3_asym['series']['T_bands_outer'][mid_frame_v3]]}")
    print(f"  t={v4_asym['series']['t_s'][mid_frame_v4]:.0f}s  v4 outer temps ({N_AXIAL_NODES} rows, base->rim): "
          f"{[round(t, 2) for t in v4_asym['series']['T_bands_outer'][mid_frame_v4]]}")

    print("\n=== v4 speed ===")
    t0 = time.time()
    n_trial = 10
    for _ in range(n_trial):
        simulate_v4(30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, 0.6, 3.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/n_trial*1000:.2f} ms/candidate (scalar-only, no series)")
    t0 = time.time()
    for _ in range(n_trial):
        simulate_v4(30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, 0.6, 3.0, 2.0, t_max=7200.0,
                    record_series=True)
    print(f"  {(time.time()-t0)/n_trial*1000:.2f} ms/candidate (record_series=True, the viewer's own call pattern)")

    print("\n=== v5 unit check: _lateral_exchange on a uniform pair is a true no-op ===")
    G_test = np.full((N_AXIAL_NODES, n_cell_test), 0.01)
    Ca_test = C_grid_t.copy()
    Cb_test = C_grid_t * 1.3  # deliberately unequal capacities -- shouldn't matter for a uniform pair
    Ta_uniform = np.full((N_AXIAL_NODES, n_cell_test), 55.0)
    Tb_uniform = np.full((N_AXIAL_NODES, n_cell_test), 55.0)
    out_a, out_b = _lateral_exchange(Ta_uniform, Tb_uniform, G_test, Ca_test, Cb_test, 5.0)
    print(f"  max deviation from input (should be ~0): "
          f"{max(np.max(np.abs(out_a-Ta_uniform)), np.max(np.abs(out_b-Tb_uniform))):.2e}")

    print("\n=== v5 unit check: _lateral_exchange on a real gradient converges the pair and conserves energy ===")
    Ta_grad = np.full((N_AXIAL_NODES, n_cell_test), 80.0)
    Tb_grad = np.full((N_AXIAL_NODES, n_cell_test), 30.0)
    energy_before = float(np.sum(Ca_test * Ta_grad) + np.sum(Cb_test * Tb_grad))
    out_a, out_b = _lateral_exchange(Ta_grad, Tb_grad, G_test, Ca_test, Cb_test, 5.0)
    energy_after = float(np.sum(Ca_test * out_a) + np.sum(Cb_test * out_b))
    print(f"  gap before={float(np.mean(Tb_grad-Ta_grad)):.4f}C  after={float(np.mean(out_b-out_a)):.4f}C "
          f"(should shrink toward 0)")
    print(f"  energy before={energy_before:.6e}  after={energy_after:.6e}  "
          f"rel_diff={(energy_after-energy_before)/energy_before:.2e} (should be ~0)")

    print("\n=== v5 invariance: mass_kg (a pure geometry/material sum) must exactly match v4's, "
          "since splitting the ring into patch+bulk doesn't change its total area ===")
    args_default = (30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, 0.6, 3.0, 2.0)
    v4_default = simulate_v4(*args_default, t_max=600.0)
    v5_default = simulate_v5(*args_default, t_max=600.0)
    print(f"  v4 mass_kg={v4_default['mass_kg']:.6f}  v5 mass_kg={v5_default['mass_kg']:.6f}  "
          f"(diff={abs(v4_default['mass_kg']-v5_default['mass_kg']):.2e})")

    print("\n=== v5 physical sanity: with a real handle, the PATCH should run measurably cooler than the "
          "BULK at the same height -- it alone feeds the handle's real conductive drain ===")
    args_handle = (30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 60.0, 10.0, 4.0, 0.0, 3.0, 0.0)  # long steel handle, open cup
    v5_handle = simulate_v5(*args_handle, t_max=1800.0, record_series=True, series_dt=None)
    s5 = v5_handle["series"]
    mid_frame = len(s5["t_s"]) // 2
    bulk_mid = s5["T_bulk_outer"][mid_frame][N_AXIAL_NODES // 2]
    patch_mid = s5["T_patch_outer"][mid_frame][N_AXIAL_NODES // 2]
    print(f"  t={s5['t_s'][mid_frame]:.0f}s  bulk mid-row outer={bulk_mid:.3f}C  "
          f"patch mid-row outer={patch_mid:.3f}C  (patch should be COOLER -- it's the one feeding the handle)")

    print("\n=== v5 reducible limit: with contact conductance ~0 (handle effectively disconnected), "
          "patch and bulk should be indistinguishable -- the ONLY thing that should ever separate them ===")
    saved_contact = H_HANDLE_CONTACT
    globals()["H_HANDLE_CONTACT"] = 1e-9
    v5_nohandle = simulate_v5(*args_handle, t_max=1800.0, record_series=True, series_dt=None)
    globals()["H_HANDLE_CONTACT"] = saved_contact
    s5n = v5_nohandle["series"]
    bulk_mid_n = s5n["T_bulk_outer"][mid_frame][N_AXIAL_NODES // 2]
    patch_mid_n = s5n["T_patch_outer"][mid_frame][N_AXIAL_NODES // 2]
    print(f"  t={s5n['t_s'][mid_frame]:.0f}s  bulk mid-row outer={bulk_mid_n:.4f}C  "
          f"patch mid-row outer={patch_mid_n:.4f}C  (should now match closely)")

    print("\n=== v5 speed ===")
    t0 = time.time()
    for _ in range(n_trial):
        simulate_v5(30.0, 35.0, 40.0, 1.0, 1.5, 2.0, 8.0, 1.5, 40.0, 8.0, 2.0, 0.6, 3.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/n_trial*1000:.2f} ms/candidate (scalar-only, no series)")

    print("\n=== v6 unit check: batched _step_chain matches a per-column loop exactly ===")
    n_theta_test = N_THETA
    n_cell_test6 = N_WALL_CELLS
    dx_row = np.full(n_cell_test6, 0.003 / n_cell_test6)
    k_row = np.full(n_cell_test6, 1.5)
    rho_row = np.full(n_cell_test6, 2300.0)
    cp_row = np.full(n_cell_test6, 1050.0)
    rng6 = np.random.default_rng(1)
    T_cols = 20.0 + rng6.uniform(-2, 2, size=(n_cell_test6, n_theta_test))  # slightly different per column
    area_test = 0.0011
    batched = _step_chain(T_cols, dx_row, k_row, rho_row, cp_row, area_test, 5.0,
                           T_amb_right=T_AMB, h_right=H_AIR + 5.0, T_left_conv=90.0, h_left=H_LIQ)
    looped = np.stack([
        _step_chain(T_cols[:, j], dx_row, k_row, rho_row, cp_row, area_test, 5.0,
                    T_amb_right=T_AMB, h_right=H_AIR + 5.0, T_left_conv=90.0, h_left=H_LIQ)
        for j in range(n_theta_test)
    ], axis=1)
    print(f"  max |batched - looped| (should be ~0): {np.max(np.abs(batched - looped)):.2e}")

    print("\n=== v6 unit check: _azimuthal_diffuse on a uniform ring is a true no-op; "
          "on a real perturbation it smooths and conserves energy ===")
    n_z_test = N_AXIAL_NODES
    k_cell6 = np.full(n_cell_test6, 1.5)
    dx_grid6 = np.full((n_z_test, n_cell_test6), 0.003 / n_cell_test6)
    C_grid6 = np.full((n_z_test, n_cell_test6), 0.02)
    r_center6 = np.full(n_z_test, 0.035)
    T_uniform6 = np.full((n_z_test, n_cell_test6, N_THETA), 40.0)
    out6 = _azimuthal_diffuse(T_uniform6, k_cell6, dx_grid6, C_grid6, r_center6, HEIGHT / n_z_test, N_THETA, 5.0)
    print(f"  max deviation from input (should be ~0): {np.max(np.abs(out6 - T_uniform6)):.2e}")

    T_pert6 = np.full((n_z_test, n_cell_test6, N_THETA), 40.0)
    T_pert6[:, :, 0] = 80.0  # one hot wedge, rest at 40 -- exactly the handle's own situation
    energy_before6 = float(np.sum(C_grid6[:, :, None] * T_pert6))
    out_pert6 = _azimuthal_diffuse(T_pert6, k_cell6, dx_grid6, C_grid6, r_center6, HEIGHT / n_z_test, N_THETA, 5.0)
    energy_after6 = float(np.sum(C_grid6[:, :, None] * out_pert6))
    print(f"  hot wedge before=80.0C  after={out_pert6[0,0,0]:.4f}C  neighbor after={out_pert6[0,0,1]:.4f}C "
          f"(should move toward each other)")
    print(f"  energy before={energy_before6:.6e}  after={energy_after6:.6e}  "
          f"rel_diff={(energy_after6-energy_before6)/energy_before6:.2e} (should be ~0)")

    print("\n=== v6 physical sanity: wall starts at ambient and genuinely HEATS UP near the liquid line, "
          "not just cools down from an already-hot state ===")
    args_v6 = (30.0, 35.0, 40.0, 3.0, 5.0, 2.0, 40.0, 8.0, 2.0)
    r6 = simulate_v6(*args_v6, t_max=600.0, record_series=True, series_dt=None)
    s6 = r6["series"]
    for t_probe in [0.0, 30.0, 120.0, 600.0]:
        f = min(range(len(s6["t_s"])), key=lambda i: abs(s6["t_s"][i] - t_probe))
        rim_avg = float(np.mean(s6["T_wall_outer"][f][-1]))
        print(f"  t={s6['t_s'][f]:>4.0f}s  rim-row outer avg temp={rim_avg:.3f}C "
              f"(t=0 should read ~{T_AMB}C, then rise)")

    print("\n=== v6 physical sanity: the handle's own azimuthal wedge runs measurably cooler "
          "than the rest of the ring, at real grid resolution (not a 2-cell approximation) ===")
    mid_f = len(s6["t_s"]) // 2
    ring_at_mid = np.array(s6["T_wall_outer"][mid_f][N_AXIAL_NODES // 2])
    handle_col = ring_at_mid[0]
    rest_avg = float(np.mean(ring_at_mid[1:]))
    print(f"  t={s6['t_s'][mid_f]:.0f}s  handle-wedge outer={handle_col:.3f}C  "
          f"rest-of-ring avg outer={rest_avg:.3f}C  (handle wedge should run cooler)")

    print("\n=== v6 validity check: a design that can't hold a meaningful amount of liquid is rejected ===")
    # Note: r_base/r_mid/r_rim ARE the liquid cavity's own radii in this
    # model (wall thickness is added OUTSIDE them, doesn't eat into the
    # cavity) -- so within the real (15-60mm) sampling range and this
    # domain's fixed HEIGHT=0.10m, even the smallest straight cylinder
    # already holds ~70.7mL, comfortably above MIN_LIQUID_VOLUME_L=50mL.
    # The floor is correctly wired but currently INERT against the real
    # range (nothing that small is ever actually sampled) -- confirmed
    # directly below rather than assumed. Using an out-of-range toy radius
    # here just to exercise the code path itself.
    tiny = simulate_v6(5.0, 5.0, 5.0, 3.0, 3.0, 2.0, 40.0, 8.0, 2.0, t_max=600.0)
    real_min = simulate_v6(15.0, 15.0, 15.0, 3.0, 3.0, 2.0, 40.0, 8.0, 2.0, t_max=600.0)
    print(f"  toy tiny mug (r=5mm, out-of-range): valid={tiny['valid']} (should be False)")
    print(f"  real range minimum (r=15mm): valid={real_min['valid']}  "
          f"liquid_volume_L={real_min['liquid_volume_L']:.4f} (should be True, ~0.0707L -- "
          f"the floor doesn't reject anything the real sampling range can produce)")

    print("\n=== v6 radiative-linearization sanity: linearized flux tracks the real T^4 flux closely ===")
    T_test_C = 70.0
    h_lin = _radiative_h(T_test_C)
    flux_lin = h_lin * (T_test_C - T_AMB)
    T_test_K, T_amb_K = T_test_C + 273.15, T_AMB + 273.15
    flux_exact = EMISSIVITY_SURFACE * SIGMA_SB * (T_test_K ** 4 - T_amb_K ** 4)
    print(f"  at T={T_test_C}C: linearized flux/area={flux_lin:.3f} W/m2  exact flux/area={flux_exact:.3f} W/m2  "
          f"rel_diff={(flux_lin-flux_exact)/flux_exact:.2e}")

    print("\n=== v6 speed ===")
    t0 = time.time()
    n_trial6 = 5
    for _ in range(n_trial6):
        simulate_v6(30.0, 35.0, 40.0, 3.0, 5.0, 2.0, 40.0, 8.0, 2.0, t_max=7200.0)
    print(f"  {(time.time()-t0)/n_trial6*1000:.2f} ms/candidate (scalar-only, no series)")
    t0 = time.time()
    for _ in range(n_trial6):
        simulate_v6(30.0, 35.0, 40.0, 3.0, 5.0, 2.0, 40.0, 8.0, 2.0, t_max=7200.0, record_series=True)
    print(f"  {(time.time()-t0)/n_trial6*1000:.2f} ms/candidate (record_series=True, the viewer's own call pattern)")

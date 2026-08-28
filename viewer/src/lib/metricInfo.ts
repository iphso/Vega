/** Plain-language explanations + real reference values for every target
metric shown across the three viewers. Shared because none of the 20 names
collide across domains (stellarator/airfoil/TORAX), so one lookup covers
all three pages' filter panels AND the three per-domain glossary pages.

Sourced from real, checkable places, not written from vague recollection:
the 12 stellarator metrics' definitions and "why it matters" lines are
paraphrased directly from `constellaration.problems`' own docstrings
(`GeometricalProblem`, `SimpleToBuildQIStellarator`, `MHDStableQIStellarator`
-- Proxima Fusion's own benchmark-problem definitions, inspected directly
in this project's `train` image rather than assumed), and their numeric
references are those problems' own real constraint bounds. The airfoil and
TORAX reference numbers are this project's own previously-computed real
values (EXPERIMENT_LOG §36's NACA polars and §36/§40's real-device TORAX
runs), not textbook lookups.

`short`: one plain sentence for the hover-bubble in the viewers themselves.
`blurb`/`reference`: the full explanation, shown only on each domain's own
glossary page (`/glossary/{stellarator,airfoil,torax}`) -- moved there
per direct user feedback that the inline expand-to-read version was too
much text living inside the filter panel. */

export interface MetricInfo {
  short: string;
  blurb: string;
  reference: string;
}

export const DOMAIN_METRICS: Record<string, string[]> = {
  stellarator: [
    'aspect_ratio', 'aspect_ratio_over_edge_rotational_transform', 'average_triangularity',
    'axis_magnetic_mirror_ratio', 'axis_rotational_transform_over_n_field_periods',
    'edge_magnetic_mirror_ratio', 'edge_rotational_transform_over_n_field_periods',
    'flux_compression_in_regions_of_bad_curvature', 'max_elongation',
    'minimum_normalized_magnetic_gradient_scale_length', 'qi', 'vacuum_well',
  ],
  airfoil: ['cl', 'cd', 'cm', 'l_over_d'],
  torax: ['Q_fusion', 'tau_E', 'H98', 'T_e_volume_avg'],
};

export const METRIC_INFO: Record<string, MetricInfo> = {
  // --- Stellarator (VMEC++/ConStellaration) ---
  aspect_ratio: {
    short: 'Major radius over minor radius — how "fat" or "thin" the torus is; lower is more compact but harder to satisfy the other constraints.',
    blurb: 'Major radius over minor radius (R₀/a) — how "fat" or "thin" the torus is. Lower means a more compact, potentially cheaper machine, but it gets harder to satisfy the other constraints below at low aspect ratio.',
    reference: 'ConStellaration’s own benchmark problems require ≤4.0 (compact-geometry problem) or ≤10.0 (simple-to-build problem) — real constraint bounds, not a rule of thumb.',
  },
  aspect_ratio_over_edge_rotational_transform: {
    short: 'Aspect ratio per unit of edge magnetic twist — a compactness score; lower means more confining twist for how fat the torus is.',
    blurb: 'Aspect ratio divided by the edge rotational transform — a "compactness per unit of magnetic twist" score. Lower means the design gets more confining twist for how fat the torus is.',
    reference: 'This is literally what ConStellaration’s GeometricalProblem optimizes: it scores designs by how close to circular (see max_elongation) they can get while holding this ratio down.',
  },
  average_triangularity: {
    short: 'How "D-shaped" the cross-section is; negative means it pinches inward rather than bulging into a rounder D.',
    blurb: 'How "D-shaped" the poloidal cross-section is, averaged around the torus — negative values mean the shape pinches inward (indented) rather than bulging into a rounder D.',
    reference: 'ConStellaration’s GeometricalProblem requires ≤−0.5 — real designs in that benchmark are pushed toward negative triangularity, not incidental.',
  },
  axis_magnetic_mirror_ratio: {
    short: 'Field-strength variation along a field line at the magnetic axis; lower means fewer particles get trapped there.',
    blurb: 'Ratio of the strongest to weakest field strength |B| along a field line AT THE MAGNETIC AXIS. A field that varies less along its own field lines traps fewer particles in local "magnetic mirrors" — trapped particles are a major stellarator-specific confinement loss channel (tokamaks don’t have this problem the same way).',
    reference: 'No axis-specific bound in the benchmark problems (only the edge value is directly constrained, see edge_magnetic_mirror_ratio) — lower is still better here, just not a formally scored constraint.',
  },
  edge_magnetic_mirror_ratio: {
    short: 'Same field-strength-variation idea, measured at the plasma edge, where it matters most for confining the outer plasma.',
    blurb: 'Same idea as the axis version, but measured at the plasma edge, where it matters most for confining particles near the boundary before they’re lost.',
    reference: 'ConStellaration’s QI-stellarator problems require ≤0.2–0.25 — real constraint bounds.',
  },
  axis_rotational_transform_over_n_field_periods: {
    short: 'How much field lines twist poloidally per toroidal trip, at the axis, normalized so it’s comparable across designs.',
    blurb: 'Rotational transform (ι — how many times a field line winds poloidally per toroidal trip around the machine) at the magnetic axis, normalized by the number of field periods so it’s comparable across designs with different field-period counts. This "twist" is what averages out particle drifts — without it, confinement collapses.',
    reference: 'No axis-specific bound in the benchmark problems (only the edge value is directly constrained, see below).',
  },
  edge_rotational_transform_over_n_field_periods: {
    short: 'Same twist, measured at the edge — has to stay high enough all the way out or confinement collapses.',
    blurb: 'Same idea, measured at the plasma edge — the twist that has to survive all the way to the boundary for the field to actually confine the outer plasma.',
    reference: 'Every ConStellaration benchmark problem requires ≥0.25–0.3 here — too little edge twist and the design is infeasible by definition.',
  },
  flux_compression_in_regions_of_bad_curvature: {
    short: 'A geometric proxy for turbulent transport in the field’s destabilizing regions; lower is better.',
    blurb: 'A geometric proxy for turbulent transport: how much magnetic flux gets squeezed together specifically in the parts of the torus where the field curvature is destabilizing (rather than stabilizing). Lower means less of a free ride for turbulence to grow.',
    reference: 'ConStellaration’s MHDStableQIStellarator problem requires ≤0.9.',
  },
  max_elongation: {
    short: 'How vertically stretched the cross-section gets at its worst point; closer to 1 (circular) is easier to build and more stable.',
    blurb: 'How vertically stretched the poloidal cross-section gets, at its most-stretched point anywhere around the torus. More elongation can pack in more plasma volume per unit machine size, but pushes coil complexity and can destabilize the plasma — there’s a real ceiling.',
    reference: 'ConStellaration’s GeometricalProblem literally scores designs from 1 (circular — best) to 10 (very elongated — worst); the simple-to-build problem caps it at ≤5.0.',
  },
  minimum_normalized_magnetic_gradient_scale_length: {
    short: 'The worst-case sharpness of the field’s variation; bigger (smoother) means easier, cheaper coils.',
    blurb: 'The WORST-CASE (smallest) length scale over which the magnetic field varies, normalized to be comparable across designs. A small value means the field changes very sharply somewhere — that’s exactly the kind of feature that’s expensive or impossible to build real coils for. Bigger (smoother) is better.',
    reference: 'This is literally what ConStellaration’s two QI-stellarator problems try to MAXIMIZE — one scores it from 0 (poor) to 1 (optimal, i.e. easiest to build).',
  },
  qi: {
    short: 'How far this field is from "quasi-isodynamic" — the symmetry property this whole optimization campaign is chasing; closer to 0 is better.',
    blurb: 'The quasi-isodynamicity residual — how far this field is from "QI," a symmetry property where trapped-particle orbits average out to zero net radial drift. This is the actual innovation the whole ConStellaration campaign is chasing: a real QI field would dramatically cut the neoclassical transport losses that limit stellarators today. Closer to 0 is better.',
    reference: 'ConStellaration’s QI-stellarator problems require log₁₀(qi) ≤ −3.5 to −4 — i.e. a qi residual on the order of 1e-4 or smaller to count as "QI enough."',
  },
  vacuum_well: {
    short: 'Depth of the field’s own restoring "bowl" against instabilities, before any plasma pressure is applied; non-negative is required.',
    blurb: 'The depth of the magnetic "well" in the field with no plasma pressure yet applied — like a ball sitting in a bowl vs. balanced on a hill. A real (non-negative) well acts as a built-in restoring force against ideal-MHD pressure-driven instabilities.',
    reference: 'ConStellaration’s MHDStableQIStellarator problem requires ≥0 — a negative well is an infeasible design in that benchmark.',
  },

  // --- Airfoils (XFOIL) ---
  cl: {
    short: 'Lift coefficient — how much lift this section generates, normalized so it’s comparable across speeds and sizes.',
    blurb: 'Lift coefficient — how much lift this 2D section generates, normalized by dynamic pressure and chord so it’s comparable across speeds/sizes. This is the number that has to add up (via wing area and speed) to support an aircraft’s weight.',
    reference: 'This project’s own real NACA 0012 polar (§36): cl=0.000 at α=0° (symmetric), rising to 1.15 by α=10°; cambered NACA 2412 already sits at cl=0.234 at α=0°. Typical cruise range for a clean section is roughly 0.3–0.6; usable lift tops out somewhere past 1.2–1.6 before stall.',
  },
  cd: {
    short: 'Drag coefficient — the price paid in drag for that lift; lower is better at a given cl.',
    blurb: 'Drag coefficient — the price paid, in drag, for that lift. You want this as small as possible at the cl you actually need; it rises sharply near stall as the flow starts separating.',
    reference: 'This project’s own real NACA 0012 polar (§36): cd=0.00513 at α=0° (near-minimum), climbing to 0.011–0.02 by α=10–16° as stall approaches — the normal shape for a clean 2D section at this Reynolds number.',
  },
  cm: {
    short: 'Pitching-moment coefficient — the aerodynamic twisting force on the section, which has to be trimmed out to fly straight.',
    blurb: 'Pitching-moment coefficient (about the quarter-chord) — the aerodynamic twisting force trying to rotate the section nose-up or nose-down. A real aircraft’s tail (or the structure) has to counteract this to stay trimmed, so it matters for stability, not just efficiency.',
    reference: 'This project’s own real polars (§36): symmetric NACA 0012 sits near cm=0 at low angles; cambered NACA 2412/4415 carry a real nose-down moment around −0.05 to −0.09 throughout their whole alpha range — the textbook cambered-section signature.',
  },
  l_over_d: {
    short: 'Lift-to-drag ratio — the single best "how efficient is this shape" number for cruise; higher is better.',
    blurb: 'Lift-to-drag ratio (cl/cd) — the single best "how efficient is this shape" number for cruise: how much lift you get per unit of drag paid. Higher is better. (This is a 2D section number — a real 3D wing’s L/D is lower once induced drag, not modeled here, is added.)',
    reference: 'This project’s own real polars (§36): NACA 2412/4415 peak around L/D≈114–149 near α=4–6° — XFOIL 2D predictions like this run optimistic vs. real flight L/D, which is expected and not a red flag on its own.',
  },

  // --- TORAX (tokamak transport) ---
  Q_fusion: {
    short: 'Fusion power out over heating power in — the headline "does this reactor concept work" number; Q=1 is breakeven.',
    blurb: 'Fusion gain — fusion power produced divided by external heating power put in. Q=1 is "breakeven" (fusion output matches heating input); Q→∞ is "ignition" (the plasma sustains itself with no external heating at all). This is the single headline number for "does this reactor concept actually work."',
    reference: 'ITER’s real design target is Q≈10. JET’s real 1997 record was Q≈0.67 (65MJ, briefly). This project’s own real-device runs (§36/§40) land far below that: ITER-baseline-geometry Q_fusion=3.16, SPARC-geometry Q_fusion=0.19 — both run through this project’s own generic (non-device-tuned) heating/density settings, not each device’s own real predicted performance, so these aren’t claims about the real machines’ actual targets.',
  },
  tau_E: {
    short: 'Energy confinement time, in seconds — how long the plasma holds its heat before it leaks out; longer is better.',
    blurb: 'Energy confinement time, in seconds — how long the plasma holds onto its thermal energy before it leaks out, absent more heating. Longer means the magnetic "bottle" is doing a better job of holding heat in.',
    reference: 'ITER targets roughly 3–6s. Present real tokamaks (JET, DIII-D) typically run tenths of a second up to about 1s. This project’s own ITER-baseline-geometry run (§36): tau_E=1.38s.',
  },
  H98: {
    short: 'Confinement quality relative to the standard empirical scaling law; H98=1 matches the historical baseline.',
    blurb: 'Confinement quality relative to the H98y2 empirical scaling law — a formula fit to a large multi-device database predicting confinement time from a design’s size/field/current/etc. H98=1 means "behaving exactly as that standard scaling predicts" (this is what an ordinary H-mode plasma scores by definition); above 1 is better than the historical baseline, well below 1 usually signals a lower-confinement regime.',
    reference: 'This project’s own ITER-baseline-geometry run (§36): H98=0.66 — below the H98=1 reference line, again because the run uses generic transport settings rather than ITER’s own tuned scenario.',
  },
  T_e_volume_avg: {
    short: 'Volume-averaged electron temperature, in keV — a direct proxy for "is the plasma hot enough to fuse."',
    blurb: 'Volume-averaged electron temperature, in keV (1 keV ≈ 11.6 million °C). Fusion reaction rate depends steeply on temperature, so this is a direct proxy for "is the plasma actually hot enough to fuse" — not just confined, but hot.',
    reference: 'Fusion-relevant core temperatures are typically 10–20+ keV. This project’s own ITER-baseline-geometry run (§36): T_e_volume_avg=7.2 keV (volume-AVERAGED, so noticeably below the peak core temperature a radial profile would show at its center).',
  },
};

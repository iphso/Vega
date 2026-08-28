"""Real, named-device reference points for the TORAX domain -- direct
response to the gap the user flagged: the stellarator domain's dataset is
literally copied from real ConStellaration designs (`preprocess.py`), but
`generate_torax_dataset.py` (§35) is 100% synthetic noise-perturbation
around two archetypes, with no real, unperturbed reference target values
ever recorded anywhere.

Three real devices, geometry/current/field values from actual published
specs (not fabricated -- ITER's are already load-bearing in this project
via `basic_config`/`iterhybrid_predictor_corrector`, §31/§35; SPARC/JET
confirmed via direct web search this pass, sources in EXPERIMENT_LOG):

  - `iter_baseline` / `iter_flattop`: same as generate_torax_dataset.py's
    own seeds (§35) -- R_major=6.2m, a_minor=2.0m, B_0=5.3T are ITER's real
    published geometry, Ip/P_total the two real scenario variants already
    documented there. Included here specifically to record their UNPERTURBED
    (zero noise) real target values, which §35's dataset never isolates
    (every row there has noise applied).
  - `sparc`: R_major=1.85m, a_minor=0.57m, B_0=12.2T, Ip=8.7MA -- Commonwealth
    Fusion Systems' published SPARC design (high-field, compact, D-T,
    targeting Q>2). A genuinely different regime from ITER (4x the field,
    3.3x smaller major radius), not just a scaled variant.
  - `jet`: R_major=2.96m, a_minor=1.25m, B_0=3.45T, Ip=4.8MA (D-shaped
    plasma current) -- the real, decades-operated Joint European Torus.

Honesty about what's real and what isn't: only R_major/a_minor/B_0/Ip are
drawn from each device's actual published specs. nbar/chi_i/chi_e/P_total/
I_generic have no equally citable single "real" value for SPARC/JET readily
available here, so they're left at `iter_baseline`'s own defaults (TORAX's
own bundled basic_config values) for every device rather than fabricated --
these three reference points test "does a real device's real geometry/
current/field, run through our exact parameterization, produce sane
targets," not a full real-scenario replication.
"""
import json
from pathlib import Path

import numpy as np

import torax_oracle as oracle
from oracle_harness_persistent import run_batch_persistent

OUT_DIR = Path("/work/output")

# Baseline transport/heating knobs shared by every device below (TORAX's own
# basic_config defaults -- see generate_torax_dataset.py's ITER_BASELINE).
_SHARED = {"nbar": 8.5e19, "chi_i": 1.0, "chi_e": 1.0, "P_total": 1.2e8, "I_generic": 3_000_000.0}

REAL_DEVICES = {
    "iter_baseline": {**_SHARED, "Ip": 15_000_000.0,
                       "R_major": 6.2, "a_minor": 2.0, "B_0": 5.3, "elongation_LCFS": 1.72},
    "iter_flattop": {**_SHARED, "Ip": 10_500_000.0, "P_total": 5.1e7,
                      "R_major": 6.2, "a_minor": 2.0, "B_0": 5.3, "elongation_LCFS": 1.72},
    "sparc": {**_SHARED, "Ip": 8_700_000.0,
              "R_major": 1.85, "a_minor": 0.57, "B_0": 12.2, "elongation_LCFS": 1.72},
    "jet": {**_SHARED, "Ip": 4_800_000.0,
            "R_major": 2.96, "a_minor": 1.25, "B_0": 3.45, "elongation_LCFS": 1.72},
}


def main():
    jobs = [(name, overrides, oracle.FIDELITY_PRESETS["low"]) for name, overrides in REAL_DEVICES.items()]
    results = {}
    for tag, ok, payload in run_batch_persistent(jobs, oracle.persistent_worker_fn, n_workers=4, timeout_s=60.0):
        results[tag] = {"ok": ok, "payload": payload}
        status = "OK" if ok else "FAILED"
        print(f"[{tag}] {status}: {payload}")

    (OUT_DIR / "torax_real_references.json").write_text(json.dumps(
        {"devices": REAL_DEVICES, "results": results}, indent=2))
    print(f"\nsaved -> {OUT_DIR / 'torax_real_references.json'}")


if __name__ == "__main__":
    main()

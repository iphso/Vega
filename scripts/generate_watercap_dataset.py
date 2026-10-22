"""Bootstraps a dataset for the watercap domain (watercap_oracle.py: the
electrospray + corona water-recovery tube). Same shape as
generate_photonics_dataset.py (oracle_harness.run_batch_with_timeout,
subprocess-per-candidate), except this domain HAS operating-condition aux
(inlet T/RH/mist loading/mist size), which is appended to X after
PARAM_DIM -- the airfoil reynolds/alpha convention -- so
`{tag}_feature_names.json` is PARAM_NAMES + AUX_NAMES.

Pure numpy/scipy; runs on the host or in any compose service (no MEEP/
OpenFOAM dependency at the ROM fidelities).

Sampling: half the batch perturbs literature/prototype archetypes, half is
uniform over RANGES. Two sampling choices that matter:
  - air flow is sampled as a face VELOCITY times the sampled tube area, not
    as a flow rate, so U stays in a physical 0.5-4 m/s band for every
    diameter;
  - spray flow is sampled as a liquid-to-gas MASS ratio (0-1 kg/kg, spray
    scrubber range), with a 20% chance of exactly zero (corona-only, the
    partner's current prototype).

Rows whose design arcs (arc_margin < 0) are KEPT: ok=True, real metrics,
the validity boundary is information a generator needs. gym_schema's
watercap_domain().validity_fn is what marks them invalid.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import watercap_ehd
import watercap_oracle as oracle
from oracle_harness import run_batch_with_timeout

OUT_DIR = Path("/work/output")

# Partner-described baseline: 30 cm tube, ~2 m/s, corona stack above an
# electrospray stage, saturated 35 C tower exhaust with plume mist.
BASE = dict(air_flow_m3_s=0.15, tube_diameter_m=0.3, spray_section_length_m=1.5, corona_section_length_m=1.5,
            nozzle_height_frac=0.9, disc_diameter_frac=0.3, disc_spacing_m=0.1, spray_flow_L_min=2.0,
            spray_n_nozzles=100, spray_T_C=20.0, spray_conductivity_mS_m=50.0, spray_voltage_kV=10.0,
            corona_voltage_kV=30.0, corona_polarity=-1.0, spray_polarity_rel=1.0)
BASE_AUX = dict(inlet_T_C=35.0, inlet_RH=1.0, inlet_lwc_g_m3=2.0, mist_mmd_um=15.0, mist_gsd=1.8)
SEEDS = {
    "prototype_corona_only": {**BASE, "spray_flow_L_min": 0.0, "spray_voltage_kV": 0.0},
    "combined_baseline": BASE,
    "cold_makeup_spray": {**BASE, "spray_flow_L_min": 8.0, "spray_T_C": 12.0},
    "di_water_low_current": {**BASE, "spray_conductivity_mS_m": 0.2, "spray_n_nozzles": 20},
}

RANGES = dict(
    face_velocity_m_s=(0.5, 4.0),
    tube_diameter_m=(0.10, 0.60),
    spray_section_length_m=(0.3, 3.0),
    corona_section_length_m=(0.3, 3.0),
    nozzle_height_frac=(0.3, 1.0),
    disc_diameter_frac=(0.05, 0.6),
    disc_spacing_m=(0.02, 0.3),
    liquid_gas_ratio=(0.0, 1.0),
    log10_n_nozzles=(0.0, 3.0),
    spray_T_C=(5.0, 40.0),
    log10_conductivity_mS_m=(-1.0, np.log10(300.0)),
    spray_voltage_kV=(0.0, 30.0),
    corona_voltage_kV=(0.0, 60.0),
    polarity=(-1.0, 1.0),
)
AUX_RANGES = dict(
    inlet_T_C=(20.0, 45.0),
    inlet_RH=(0.85, 1.0),
    log10_lwc_g_m3=(-1.0, np.log10(5.0)),
    log10_mmd_um=(np.log10(3.0), np.log10(40.0)),
    mist_gsd=(1.2, 2.5),
)


def _dry_air_kg_s(flow, aux):
    w = oracle.w_from_pv(aux["inlet_RH"] * oracle.p_sat(aux["inlet_T_C"]))
    return oracle.gas_density(aux["inlet_T_C"], w) * flow / (1 + w)


def _sample_aux(rng):
    u = {k: rng.uniform(*v) for k, v in AUX_RANGES.items()}
    return dict(inlet_T_C=u["inlet_T_C"], inlet_RH=u["inlet_RH"], inlet_lwc_g_m3=10 ** u["log10_lwc_g_m3"],
                mist_mmd_um=10 ** u["log10_mmd_um"], mist_gsd=u["mist_gsd"])


def _sample_uniform(rng, aux):
    u = {k: rng.uniform(*v) for k, v in RANGES.items()}
    D = u["tube_diameter_m"]
    flow = u["face_velocity_m_s"] * np.pi * D ** 2 / 4
    lg = 0.0 if rng.random() < 0.2 else u["liquid_gas_ratio"]
    x = dict(air_flow_m3_s=flow, tube_diameter_m=D, spray_section_length_m=u["spray_section_length_m"],
             corona_section_length_m=u["corona_section_length_m"], nozzle_height_frac=u["nozzle_height_frac"],
             disc_diameter_frac=u["disc_diameter_frac"], disc_spacing_m=u["disc_spacing_m"],
             spray_flow_L_min=lg * _dry_air_kg_s(flow, aux) / oracle.RHO_L * 1000 * 60,
             spray_n_nozzles=float(np.round(10 ** u["log10_n_nozzles"])), spray_T_C=u["spray_T_C"],
             spray_conductivity_mS_m=10 ** u["log10_conductivity_mS_m"], spray_voltage_kV=u["spray_voltage_kV"],
             corona_voltage_kV=u["corona_voltage_kV"], corona_polarity=rng.uniform(-1, 1),
             spray_polarity_rel=rng.uniform(-1, 1))
    return np.array([x[k] for k in oracle.PARAM_NAMES])


def _perturb_seed(rng, seed, noise_std):
    x = dict(seed)
    for k in ("air_flow_m3_s", "tube_diameter_m", "spray_section_length_m", "corona_section_length_m",
              "disc_spacing_m", "spray_flow_L_min", "spray_n_nozzles", "spray_conductivity_mS_m",
              "spray_voltage_kV", "corona_voltage_kV"):
        x[k] = x[k] * np.exp(rng.normal(0, noise_std))
    x["spray_n_nozzles"] = max(1.0, float(np.round(x["spray_n_nozzles"])))
    x["nozzle_height_frac"] = float(np.clip(x["nozzle_height_frac"] + rng.normal(0, noise_std), 0.3, 1.0))
    x["disc_diameter_frac"] = float(np.clip(x["disc_diameter_frac"] * np.exp(rng.normal(0, noise_std)), 0.05, 0.6))
    x["spray_T_C"] = float(np.clip(x["spray_T_C"] + rng.normal(0, 20 * noise_std), 5.0, 40.0))
    return np.array([x[k] for k in oracle.PARAM_NAMES])


def _perturb_aux(rng, noise_std):
    a = dict(BASE_AUX)
    a["inlet_T_C"] = float(np.clip(a["inlet_T_C"] + rng.normal(0, 20 * noise_std), 20, 45))
    a["inlet_RH"] = float(np.clip(1.0 - abs(rng.normal(0, noise_std / 2)), 0.85, 1.0))
    a["inlet_lwc_g_m3"] *= float(np.exp(rng.normal(0, noise_std)))
    a["mist_mmd_um"] *= float(np.exp(rng.normal(0, noise_std)))
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=20000)
    p.add_argument("--noise-std-range", type=float, nargs=2, default=(0.05, 0.4))
    p.add_argument("--n-workers", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=280)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--fidelity", default="low", choices=list(oracle.FIDELITY_PRESETS) + list(watercap_ehd.EHD_PRESETS),
                   help="low/medium: ROM. ehd_coarse/ehd: ROM spray stage + StoneLab resolved corona (30s-5min/row)")
    p.add_argument("--medium-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="watercap")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = out_dir / f"{args.out_tag}_X.npy"
    if existing.exists():
        print(f"[{args.out_tag}] WARNING: {existing} already has {np.load(existing).shape[0]} rows -- "
              f"this run will APPEND onto them. Archive first if the schema changed.")

    rng = np.random.default_rng(args.seed)
    seed_names = list(SEEDS)
    acc = dict(X=[], Y=[], source=[], fidelity=[], regime=[])
    n_attempted = n_accepted = n_failed = 0
    fail_reasons = {}
    t0 = time.perf_counter()

    def sample_batch(n):
        raw = []
        for i in range(n):
            if rng.random() < 0.5:
                source = seed_names[rng.integers(len(seed_names))]
                ns = rng.uniform(*args.noise_std_range)
                params, aux = _perturb_seed(rng, SEEDS[source], ns), _perturb_aux(rng, ns)
            else:
                source = "uniform"
                aux = _sample_aux(rng)
                params = _sample_uniform(rng, aux)
            ok, _ = oracle._validate(params, aux)
            if not ok:
                aux = _sample_aux(rng)
                params, source = _sample_uniform(rng, aux), "uniform"
            fid = "medium" if rng.random() < args.medium_fraction else args.fidelity
            mod = watercap_ehd if fid in watercap_ehd.EHD_PRESETS else oracle
            raw.append((i, *mod.params_to_worker_args(params, aux, fid), params, aux, source, fid))
        return raw

    def checkpoint(final=False):
        if not acc["X"] and not final:
            return
        X = np.stack(acc["X"]) if acc["X"] else np.zeros((0, oracle.PARAM_DIM + len(oracle.AUX_NAMES)), np.float32)
        Y = np.stack(acc["Y"]) if acc["Y"] else np.zeros((0, len(oracle.TARGET_NAMES)), np.float32)
        Xp, Yp = out_dir / f"{args.out_tag}_X.npy", out_dir / f"{args.out_tag}_Y.npy"
        meta_p = out_dir / f"{args.out_tag}_rowmeta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"source": [], "fidelity": [], "spray_regime": []}
        if Xp.exists():
            X, Y = np.concatenate([np.load(Xp), X]), np.concatenate([np.load(Yp), Y])
        meta["source"] += acc["source"]
        meta["fidelity"] += acc["fidelity"]
        meta["spray_regime"] += acc["regime"]
        np.save(Xp, X)
        np.save(Yp, Y)
        meta_p.write_text(json.dumps(meta))
        for k in acc:
            acc[k].clear()
        print(f"[{args.out_tag}] checkpointed -> {Xp} ({len(X)} rows total)", flush=True)

    while n_accepted < args.target_count:
        raw = sample_batch(args.batch_size)
        jobs = [c[:4] for c in raw]
        lookup = {c[0]: c[4:] for c in raw}
        n_attempted += len(jobs)
        worker = watercap_ehd.worker_fn if args.fidelity in watercap_ehd.EHD_PRESETS else oracle.worker_fn
        for tag, ok, payload in run_batch_with_timeout(jobs, worker, args.n_workers, args.timeout_seconds):
            if not ok:
                n_failed += 1
                key = str(payload)[:60]
                fail_reasons[key] = fail_reasons.get(key, 0) + 1
                continue
            params, aux, source, fid = lookup[tag]
            row_y = np.array([payload[n] for n in oracle.TARGET_NAMES], dtype=np.float32)
            if not np.all(np.isfinite(row_y)):
                n_failed += 1
                continue
            acc["X"].append(np.concatenate([params, [aux[k] for k in oracle.AUX_NAMES]]).astype(np.float32))
            acc["Y"].append(row_y)
            acc["source"].append(source)
            acc["fidelity"].append(fid)
            acc["regime"].append(payload["spray_regime"])
            n_accepted += 1
        el = time.perf_counter() - t0
        print(f"[{args.out_tag}] attempted={n_attempted} accepted={n_accepted}/{args.target_count} "
              f"failed={n_failed} elapsed={el:.0f}s ({el / max(n_attempted, 1):.3f}s/attempt wall)", flush=True)
        if len(acc["X"]) >= args.checkpoint_every:
            checkpoint()
    checkpoint(final=True)

    (out_dir / f"{args.out_tag}_target_names.json").write_text(json.dumps(oracle.TARGET_NAMES, indent=2))
    (out_dir / f"{args.out_tag}_feature_names.json").write_text(json.dumps(oracle.PARAM_NAMES + oracle.AUX_NAMES, indent=2))
    stats = {"n_accepted": n_accepted, "n_attempted": n_attempted, "n_failed": n_failed,
             "fail_reasons": fail_reasons, "elapsed_seconds": time.perf_counter() - t0, "seed": args.seed,
             "fidelity": args.fidelity, "medium_fraction": args.medium_fraction}
    (out_dir / f"{args.out_tag}_generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\ndone. {n_accepted} rows, {n_failed} failures {fail_reasons}")


if __name__ == "__main__":
    main()

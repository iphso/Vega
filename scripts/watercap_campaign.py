"""Watercap candidate campaign: many designs, grouped into concept families,
each evaluated at the best fidelity available for it.

Tiers per candidate (all recorded):
  rom   -- watercap_oracle "low" (1D well-mixed spray stage + ROM corona)
  ehd   -- watercap_ehd "ehd_coarse": the ROM's spray stage, then the
           resolved two-way coupled axisymmetric corona + droplet stage

Sprays are atomizers (lognormal drops, charge as a fraction of Rayleigh),
through watercap_oracle.simulate(spray_override=...): cone-jet
electrospray at L/min is unrealizable (EXPERIMENT_LOG, finding 3), so the
campaign does not spend runs on it.

    python3 scripts/watercap_campaign.py --run v1 --workers 30
    python3 scripts/watercap_campaign.py --run v1 --report

Results append to output/watercap_campaign/<run>/results.jsonl (one line per
candidate, resumable: finished ids are skipped).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watercap_oracle as rom  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / "watercap_campaign"

BASE_AUX = dict(inlet_T_C=35.0, inlet_RH=1.0, inlet_lwc_g_m3=2.0, mist_mmd_um=15.0, mist_gsd=1.6)

# The partner's prototype (EXPERIMENT_LOG "partner-baseline")
BRAD = dict(air_flow_m3_s=0.15, tube_diameter_m=0.30, spray_section_length_m=0.5, corona_section_length_m=1.5,
            nozzle_height_frac=1.0, disc_diameter_frac=0.30, disc_spacing_m=0.10, spray_flow_L_min=0.0,
            spray_T_C=20.0, corona_voltage_kV=40.0, corona_polarity=-1.0, spray_polarity_rel=1.0,
            disc_thickness_mm=0.3, spray_d_um=60.0, spray_gsd=1.5, spray_q_frac=0.0, spray_v0=10.0)

# name: (low, high, scale)
CORONA = dict(air_flow_m3_s=(0.08, 0.30, "lin"), tube_diameter_m=(0.20, 0.45, "lin"),
              corona_section_length_m=(0.5, 3.0, "lin"), disc_diameter_frac=(0.15, 0.6, "lin"),
              disc_spacing_m=(0.04, 0.25, "log"), corona_voltage_kV=(20.0, 70.0, "lin"),
              corona_polarity=(-1.0, 1.0, "sign"), disc_thickness_mm=(0.1, 1.0, "log"))
SPRAY = dict(spray_section_length_m=(0.5, 2.5, "lin"), nozzle_height_frac=(0.3, 1.0, "lin"),
             spray_flow_L_min=(0.5, 10.0, "log"), spray_T_C=(5.0, 25.0, "lin"), spray_d_um=(20.0, 200.0, "log"),
             spray_v0=(3.0, 20.0, "log"))
FAMILIES = {
    "corona_only": dict(n=384, ranges=CORONA, fixed=dict(spray_flow_L_min=0.0, spray_section_length_m=0.5)),
    "cold_spray": dict(n=320, ranges=CORONA | SPRAY, fixed=dict(spray_q_frac=0.0)),
    "charged_spray": dict(n=192, ranges=CORONA | SPRAY | dict(spray_q_frac=(0.005, 0.3, "log"),
                                                               spray_polarity_rel=(-1.0, 1.0, "sign")), fixed={}),
    "rain_curtain": dict(n=128, ranges=CORONA | dict(spray_section_length_m=(0.5, 2.5, "lin"),
                                                     spray_flow_L_min=(2.0, 20.0, "log"), spray_T_C=(5.0, 25.0, "lin"),
                                                     spray_d_um=(300.0, 1500.0, "log"), spray_v0=(1.0, 5.0, "lin")),
                         fixed=dict(nozzle_height_frac=1.0, spray_q_frac=0.0)),
    "brad_variants": dict(n=64, ranges=dict(corona_voltage_kV=(28.0, 60.0, "lin"), corona_polarity=(-1.0, 1.0, "sign"),
                                            air_flow_m3_s=(0.08, 0.30, "lin")), fixed={}),
}


# Campaign v2 (resolved tier): concepts only the resolved spray stage can rank -- where the spray
# goes, what the wall film does, and a passively cooled tube. `ambient_T_C` is an operating input.
SPRAY2 = dict(spray_section_length_m=(0.5, 2.5, "lin"), spray_flow_L_min=(0.5, 10.0, "log"),
              spray_T_C=(5.0, 25.0, "lin"), spray_d_um=(20.0, 300.0, "log"), spray_v0=(2.0, 15.0, "log"),
              spray_cone_half_angle_deg=(10.0, 45.0, "lin"))
FAMILIES2 = {
    "ring_spray": dict(n=128, ranges=CORONA | SPRAY2 | dict(nozzle_height_frac=(0.4, 1.0, "lin"),
                                                           nozzle_radial_frac=(0.6, 1.0, "lin"),
                                                           spray_axis_angle_deg=(15.0, 75.0, "lin")),
                       fixed=dict(spray_q_frac=0.0)),
    "coflow_spray": dict(n=128, ranges=CORONA | SPRAY2 | dict(nozzle_height_frac=(0.02, 0.3, "lin")),
                         fixed=dict(spray_q_frac=0.0, spray_axis_angle_deg=180.0)),
    "multi_tier": dict(n=128, ranges=CORONA | SPRAY2 | dict(spray_n_tiers=(1.5, 3.49, "lin"),
                                                           nozzle_height_frac=(0.6, 1.0, "lin")),
                       fixed=dict(spray_q_frac=0.0)),
    "charged_film": dict(n=128, ranges=CORONA | SPRAY2 | dict(nozzle_height_frac=(0.4, 1.0, "lin"),
                                                             spray_q_frac=(0.01, 0.3, "log"),
                                                             spray_polarity_rel=(-1.0, 1.0, "sign")), fixed={}),
    "cooled_wall": dict(n=96, ranges=CORONA | dict(spray_section_length_m=(1.0, 3.0, "lin"),
                                                   wall_U_W_m2K=(5.0, 100.0, "log"), spray_flow_L_min=(0.0, 3.0, "lin"),
                                                   spray_T_C=(10.0, 25.0, "lin"), spray_d_um=(40.0, 200.0, "log"),
                                                   nozzle_height_frac=(0.5, 1.0, "lin")),
                        fixed=dict(spray_q_frac=0.0)),
}


def sample_family(name, seed=0):
    from scipy.stats import qmc
    fam = FAMILIES[name] if name in FAMILIES else FAMILIES2[name]
    keys = list(fam["ranges"])
    u = qmc.Sobol(len(keys), scramble=True, seed=seed + zlib.crc32(name.encode()) % 1000).random(fam["n"])
    out = []
    for i, row in enumerate(u):
        c = dict(BRAD)
        c.update(fam["fixed"])
        for k, v in zip(keys, row):
            lo, hi, sc = fam["ranges"][k]
            c[k] = lo * (hi / lo) ** v if sc == "log" else (lo + (hi - lo) * v if sc == "lin" else (1.0 if v >= 0.5 else -1.0))
        c["id"] = f"{name}-{i:04d}"
        c["family"] = name
        out.append(c)
    return out


def to_rom(c):
    x = dict(c)
    x.setdefault("spray_n_nozzles", 1.0)
    x.setdefault("spray_conductivity_mS_m", 50.0)
    x.setdefault("spray_voltage_kV", 0.0)
    p = np.array([float(x[k]) for k in rom.PARAM_NAMES])
    so = dict(d_mmd_m=c["spray_d_um"] * 1e-6, gsd=c.get("spray_gsd", 1.5), q_frac=c["spray_q_frac"], v0=c["spray_v0"],
              n_bins=4)
    return p, so


SCALARS = ["outlet_total_water_frac", "vapor_removed_frac", "liquid_escape_frac", "recovered_L_h", "electrical_power_W",
           "fan_power_W", "pressure_drop_Pa", "wall_field_kV_cm", "arc_margin", "corona_current_mA",
           "ceiling_gap_frac", "water_balance_residual", "specific_energy_Wh_L", "outlet_T_C",
           "vapor_removal_ceiling_frac", "mist_collected_frac", "spray_entrained_frac", "corona_onset_kV",
           "spray_space_charge_kV_cm", "ehd_disc_fouling_frac", "ehd_quenching_frac", "ehd_current_no_mist_mA"]


def evaluate(c, aux=None, ehd_preset="ehd_coarse"):
    import watercap_ehd as ehd
    aux = dict(BASE_AUX if aux is None else aux)
    p, so = to_rom(c)
    rec = dict(id=c["id"], family=c["family"], design=c, aux=aux)
    t0 = time.time()
    ok, r = rom.simulate(p, aux, rom.FIDELITY_PRESETS["low"], spray_override=so)
    rec["rom_s"] = time.time() - t0
    rec["rom_ok"] = ok
    rec["rom"] = {k: r[k] for k in SCALARS if k in r} if ok else str(r)
    if ok:
        rec["rom"]["spray_fate_frac"] = r.get("spray_fate_frac")
    t0 = time.time()
    ok2, r2 = ehd.simulate(p, aux, ehd_preset, spray_override=so, disc_thickness_m=c["disc_thickness_mm"] * 1e-3)
    rec["ehd_s"] = time.time() - t0
    rec["ehd_ok"] = ok2
    rec["ehd"] = {k: r2[k] for k in SCALARS if k in r2} if ok2 else str(r2)
    rec["ehd_preset"] = ehd_preset
    return rec


def evaluate_resolved(c, aux=None, preset="resolved_coarse"):
    import watercap_resolved as wr
    aux = dict(BASE_AUX if aux is None else aux)
    aux.setdefault("ambient_T_C", 25.0)
    rec = dict(id=c["id"], family=c["family"], design=c, aux=aux, tier=preset)
    t0 = time.time()
    ok, r = wr.simulate(c, aux, preset)
    rec["res_s"] = time.time() - t0
    rec["res_ok"] = ok
    rec["res"] = r if ok else str(r)
    return rec


def _worker(args):
    c, aux, preset = args
    try:
        if preset.startswith("resolved"):
            return evaluate_resolved(c, aux, preset)
        return evaluate(c, aux, preset)
    except Exception as e:
        return dict(id=c["id"], family=c["family"], design=c, error=f"{e}\n{traceback.format_exc()[-800:]}")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def candidates_from(run_name, families, only_noarc=True):
    """Re-evaluate the designs of an earlier campaign run (e.g. v1's spray families at the resolved tier)."""
    out = []
    for r in load(run_name):
        if r["family"] not in families or not isinstance(r.get("ehd"), dict):
            continue
        if only_noarc and r["ehd"]["arc_margin"] <= 0:
            continue
        c = dict(r["design"])
        out.append(c)
    return out


def run(run_name, workers, families=None, preset="ehd_coarse", ids=None, rerun=None):
    d = OUT / run_name
    d.mkdir(parents=True, exist_ok=True)
    res_path = d / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.open():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    cands = []
    if rerun:
        src_run, fams = rerun.split(":")
        cands += candidates_from(src_run, fams.split(","))
    for f in (families or ([] if rerun else FAMILIES)):
        cands += sample_family(f)
    if ids:
        cands = [c for c in cands if c["id"] in ids]
    todo = [c for c in cands if c["id"] not in done]
    # expensive (many periods) first, so the tail of the run is short jobs
    todo.sort(key=lambda c: -c["corona_section_length_m"] / c["disc_spacing_m"])
    json.dump(dict(families=FAMILIES, base_aux=BASE_AUX, n=len(cands), preset=preset), open(d / "config.json", "w"),
              indent=1)
    print(f"{len(cands)} candidates, {len(todo)} to run, {workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, maxtasksperchild=20) as pool, res_path.open("a") as fh:
        for k, rec in enumerate(pool.imap_unordered(_worker, [(c, None, preset) for c in todo])):
            fh.write(json.dumps(_jsonable(rec)) + "\n")
            fh.flush()
            if k % 20 == 0 or k == len(todo) - 1:
                el = time.time() - t0
                res = rec.get("ehd", rec.get("res"))
                val = res.get("outlet_total_water_frac") if isinstance(res, dict) else rec.get("error", res)
                print(f"[{k + 1}/{len(todo)}] {el / 60:.1f} min, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min "
                      f"last={rec['id']} out={val}", flush=True)


def load(run_name):
    return [json.loads(l) for l in (OUT / run_name / "results.jsonl").open()]


def report(run_name):
    recs = load(run_name)
    print(f"{len(recs)} records")
    fams = sorted(set(r["family"] for r in recs))
    for f in fams:
        rs = [r for r in recs if r["family"] == f]
        ok = [r for r in rs if r.get("ehd_ok") and isinstance(r.get("ehd"), dict)]
        noarc = [r for r in ok if r["ehd"]["arc_margin"] > 0]
        print(f"\n== {f}: {len(rs)} run, {len(ok)} ehd ok, {len(noarc)} non-arcing")
        if not noarc:
            continue
        noarc.sort(key=lambda r: r["ehd"]["outlet_total_water_frac"])
        for r in noarc[:6]:
            e, m, dsn = r["ehd"], r["rom"], r["design"]
            print(f"  {r['id']}: out {e['outlet_total_water_frac']:.3f} (rom {m['outlet_total_water_frac']:.3f}) "
                  f"vap {e['vapor_removed_frac']:.3f} esc {e['liquid_escape_frac']:.3f} I {e['corona_current_mA']:.2f} mA "
                  f"P {e['electrical_power_W']:.0f} W  kWh/L {e['specific_energy_Wh_L'] / 1e3:.3f} arc {e['arc_margin']:.2f} | "
                  f"V {dsn['corona_voltage_kV']:.0f} pol {dsn['corona_polarity']:+.0f} t {dsn['disc_thickness_mm']:.2f}mm "
                  f"Q {dsn['air_flow_m3_s']:.2f} D {dsn['tube_diameter_m']:.2f} Lc {dsn['corona_section_length_m']:.1f} "
                  f"s {dsn['disc_spacing_m']:.3f} df {dsn['disc_diameter_frac']:.2f} spray {dsn['spray_flow_L_min']:.1f}L/m "
                  f"{dsn['spray_T_C']:.0f}C {dsn['spray_d_um']:.0f}um q{dsn['spray_q_frac']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--preset", default="ehd_coarse")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rerun", help="RUN:fam1,fam2 -- re-evaluate an earlier run's (non-arcing) designs")
    a = ap.parse_args()
    if a.report:
        report(a.run)
    elif a.smoke:
        for f in FAMILIES:
            c = sample_family(f)[0]
            t = time.time()
            r = _worker((c, None, "ehd_coarse"))
            print(f, round(time.time() - t, 1), "s", r.get("error") or (r["rom"].get("outlet_total_water_frac") if isinstance(r["rom"], dict) else r["rom"],
                                                          r["ehd"].get("outlet_total_water_frac") if isinstance(r["ehd"], dict) else r["ehd"]))
    else:
        run(a.run, a.workers, a.families, a.preset, rerun=a.rerun)


def analyze(runs=("v1", "v2_rerun", "v2_resolved"), out_json=None):
    """Cross-run summary: per-family bests at each tier, Pareto fronts on
    (outlet water, total specific energy), and ROM/EHD-tier vs resolved
    agreement on the designs evaluated at both."""
    from scipy.stats import spearmanr
    tiers = {}
    for rn in runs:
        p = OUT / rn / "results.jsonl"
        if p.exists():
            tiers[rn] = load(rn)
    S = {}
    v1 = {r["id"]: r for r in tiers.get("v1", [])}
    res = {}
    for rn in ("v2_rerun", "v2_resolved"):
        for r in tiers.get(rn, []):
            if r.get("res_ok"):
                res[r["id"]] = r
    # agreement on shared designs
    shared = [i for i in res if i in v1 and isinstance(v1[i].get("ehd"), dict)]
    if shared:
        a = np.array([v1[i]["rom"]["outlet_total_water_frac"] for i in shared])
        b = np.array([v1[i]["ehd"]["outlet_total_water_frac"] for i in shared])
        c = np.array([res[i]["res"]["outlet_total_water_frac"] for i in shared])
        va = np.array([v1[i]["ehd"]["vapor_removed_frac"] for i in shared])
        vc = np.array([res[i]["res"]["vapor_removed_frac"] for i in shared])
        S["shared"] = dict(n=len(shared), spearman_rom_vs_res=float(spearmanr(a, c)[0]),
                           spearman_ehd_vs_res=float(spearmanr(b, c)[0]),
                           median_abs_outlet_ehd_vs_res=float(np.median(np.abs(b - c))),
                           p90_abs_outlet_ehd_vs_res=float(np.percentile(np.abs(b - c), 90)),
                           median_vapor_res_minus_wellmixed=float(np.median(vc - va)),
                           by_family={})
        for f in sorted(set(v1[i]["family"] for i in shared)):
            m = np.array([v1[i]["family"] == f for i in shared])
            S["shared"]["by_family"][f] = dict(n=int(m.sum()), spearman=float(spearmanr(b[m], c[m])[0]) if m.sum() > 3 else None,
                                               median_vapor_delta=float(np.median(vc[m] - va[m])),
                                               median_outlet_delta=float(np.median(c[m] - b[m])))
    # per family best at the resolved tier (non-arcing, converged, closure ok)
    best = {}
    for i, r in res.items():
        p = r["res"]
        if p["arc_margin"] <= 0 or abs(p["water_balance_residual"]) > 0.02:
            continue
        best.setdefault(r["family"], []).append((p["outlet_total_water_frac"], i))
    S["best_resolved"] = {f: sorted(v)[:5] for f, v in best.items()}
    if out_json:
        json.dump(S, open(out_json, "w"), indent=1, default=float)
    return S

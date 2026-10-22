"""Snapshot the watercap candidate campaign for the Watercap Workbench page.

Reads output/watercap_campaign/{v1,v2_rerun,v2_resolved}/results.jsonl and
writes watercap_web/designs.json: one compact record per successful run,
short keys, 4 significant figures. Re-run as the campaign progresses, then
rebuild the page with watercap_web/build.py.

    python scripts/export_watercap_workbench.py
"""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "output" / "watercap_campaign"
OUT = ROOT / "watercap_web" / "designs.json"

# set key -> (directory, stored-result key, tier label)
SETS = {
    "v1": ("v1", "ehd", "fast spray + resolved corona"),
    "rerun": ("v2_rerun", "res", "resolved spray + resolved corona"),
    "new": ("v2_resolved", "res", "resolved spray + resolved corona"),
}
DESIGN_KEYS = ["air_flow_m3_s", "tube_diameter_m", "spray_section_length_m", "corona_section_length_m",
               "nozzle_height_frac", "disc_diameter_frac", "disc_spacing_m", "spray_flow_L_min", "spray_T_C",
               "spray_d_um", "spray_q_frac", "corona_voltage_kV", "corona_polarity", "spray_polarity_rel",
               "disc_thickness_mm", "spray_gsd", "spray_v0", "spray_cone_half_angle_deg", "spray_axis_angle_deg",
               "spray_n_tiers", "nozzle_radial_frac", "wall_U_W_m2K"]
AUX_KEYS = ["inlet_T_C", "inlet_RH", "inlet_lwc_g_m3", "mist_mmd_um", "mist_gsd"]
# stored result -> short key
RES_KEYS = {"outlet_total_water_frac": "out", "vapor_removed_frac": "vap", "liquid_escape_frac": "esc",
            "recovered_L_h": "rec", "electrical_power_W": "elecW", "fan_power_W": "fanW", "pump_power_W": "pumpW",
            "recool_power_W": "recoolW", "pressure_drop_Pa": "dp", "wall_field_kV_cm": "Ewall", "arc_margin": "arc",
            "corona_current_mA": "Ima", "specific_energy_Wh_L": "spec", "specific_energy_total_Wh_L": "specTot",
            "quenching_frac": "quench", "disc_fouling_frac": "foul", "corona_onset_kV": "Von"}
V1_EXTRA = {"ehd_quenching_frac": "quench", "ehd_disc_fouling_frac": "foul", "ehd_current_no_mist_mA": "Ima0"}


def r4(v):
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{v:.4g}")


def r10(v):
    """design inputs keep 10 significant figures: the page re-runs stored designs exactly"""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{v:.10g}")


def pick(src, keys):
    out = {}
    for k, s in keys.items():
        if k in src and r4(src[k]) is not None:
            out[s] = r4(src[k])
    return out


def main():
    rows, counts = [], {}
    for set_key, (d, rk, tier) in SETS.items():
        n = 0
        for line in open(CAMP / d / "results.jsonl"):
            r = json.loads(line)
            if not r.get(f"{rk}_ok"):
                continue
            res = r[rk]
            row = {"id": r["id"], "fam": r["family"], "set": set_key,
                   "d": {k: r10(r["design"][k]) for k in DESIGN_KEYS if k in r["design"]},
                   "a": {k: r10(r["aux"][k]) for k in AUX_KEYS + ["ambient_T_C"] if k in r["aux"]},
                   "r": pick(res, RES_KEYS)}
            if set_key == "v1":
                row["r"].update(pick(res, V1_EXTRA))
                row["r"].pop("Von", None)  # the ehd tier copies the fast model's onset field; not a resolved value
                if r.get("rom_ok"):
                    row["rom"] = pick(r["rom"], {"outlet_total_water_frac": "out", "corona_current_mA": "Ima",
                                                 "liquid_escape_frac": "esc", "corona_onset_kV": "Von"})
            else:
                if "corona_current_no_mist_mA" in res:
                    row["r"]["Ima0"] = r4(res["corona_current_no_mist_mA"])
                ss = res.get("spray_stage", {})
                row["ss"] = pick(ss, {"vapor_removed_frac": "vap", "T_out": "Tout", "film_drain": "film"})
                row["ss"]["recirc"] = bool(res.get("cfd_diag", {}).get("recirculating", False))
            rows.append(row)
            n += 1
        counts[set_key] = n
    meta = {"exported": time.strftime("%Y-%m-%d %H:%M"), "counts": counts,
            "tiers": {k: v[2] for k, v in SETS.items()}}
    OUT.write_text(json.dumps({"meta": meta, "rows": rows}, separators=(",", ":")))
    print(f"{len(rows)} rows {counts} -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

"""The partner's pilot site (coastal, licensed seawater cooling), as watercap aux
operating conditions per month. Source: partner feedback 2026-10-27
(EXPERIMENT_LOG "the partner's real electrode"):

- Air: ERA5-Land, hourly, 2016-2025, monthly means.
- Sea surface: Open-Meteo marine archive, 2022-2025.
- 20 m and 30 m: World Ocean Atlas 2023 climatology offshore of the site.

The partner's 10 MW reference cell leaves saturated at ~27 C in January and ~34 C
in August; those are the only two exhaust temperatures given. Other months are
interpolated linearly in the ambient wet bulb through those two points (a
counterflow tower's exhaust tracks the wet bulb). That interpolation is ours, an
assumption, not partner data.

Mist content of the exhaust is unknown (the partner has none yet), so
`inlet_lwc_g_m3` is left to the caller.
"""

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# month: (air T C, RH %, wet bulb C, humidity ratio g/kg, SST C, sea 20 m C, sea 30 m C)
CLIMATE = {
    "Jan": (11.4, 77, 9.2, 6.5, 16.1, 15.2, 15.2),
    "Feb": (12.1, 77, 9.8, 6.7, 15.2, 14.6, 14.6),
    "Mar": (13.3, 75, 10.8, 7.1, 15.4, 14.5, 14.4),
    "Apr": (15.7, 74, 12.7, 8.1, 16.8, 15.2, 15.0),
    "May": (19.5, 71, 15.9, 9.8, 19.3, 16.7, 16.0),
    "Jun": (24.9, 62, 19.4, 11.9, 23.7, 18.4, 16.8),
    "Jul": (28.2, 57, 21.4, 13.3, 27.5, 20.2, 17.7),
    "Aug": (28.0, 62, 22.0, 14.3, 28.0, 22.3, 18.9),
    "Sep": (24.6, 71, 20.4, 13.5, 26.8, 22.9, 20.0),
    "Oct": (20.5, 77, 17.6, 11.5, 24.2, 22.2, 20.6),
    "Nov": (16.6, 79, 14.3, 9.3, 21.5, 19.4, 19.1),
    "Dec": (13.0, 78, 10.9, 7.3, 18.3, 17.2, 17.1),
}
EXHAUST_REF = {"Jan": 27.0, "Aug": 34.0}   # partner's reference cell, saturated
SEA_DEPTHS = {"surface": 4, "20m": 5, "30m": 6}


def exhaust_T_C(month):
    if month in EXHAUST_REF:
        return EXHAUST_REF[month]
    (wa, ta), (wb, tb) = [(CLIMATE[m][2], EXHAUST_REF[m]) for m in ("Jan", "Aug")]
    return ta + (CLIMATE[month][2] - wa) * (tb - ta) / (wb - wa)


def site_aux(month, intake="20m", base=None):
    """watercap aux for `month`: saturated tower exhaust at the tube inlet, ambient air, seawater at the
    intake depth. `base` supplies the keys the site does not fix (mist loading and size)."""
    c = CLIMATE[month]
    aux = dict(base or {})
    aux.update(inlet_T_C=exhaust_T_C(month), inlet_RH=1.0, ambient_T_C=c[0], seawater_T_C=c[SEA_DEPTHS[intake]],
               site_month=month, exhaust_interpolated=month not in EXHAUST_REF)
    return aux


if __name__ == "__main__":
    import numpy as np
    ps = lambda T: 611.2 * np.exp(17.62 * T / (243.12 + T))
    w = lambda T: 0.622 * ps(T) / (101325 - ps(T))
    print("month  exhaust  ambient  sea20m  vapor ceiling (cool to sea20m + 3 K approach)")
    for m in MONTHS:
        a = site_aux(m)
        ceil = max(0.0, 1 - w(a["seawater_T_C"] + 3.0) / w(a["inlet_T_C"]))
        print(f"{m:5s}  {a['inlet_T_C']:5.1f}{'*' if a['exhaust_interpolated'] else ' '}   {a['ambient_T_C']:5.1f}   "
              f"{a['seawater_T_C']:5.1f}   {100 * ceil:4.0f}%")
    print("* exhaust temperature interpolated in wet bulb between the partner's Jan and Aug points")

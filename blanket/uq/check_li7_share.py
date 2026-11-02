"""How much of the TBR comes from Li-7, and does scaling the Li-7 (n,n't alpha) levels
by 1.5 move the (n,Xt) tally as expected? Same seed, per-nuclide TBR tally."""
import os, sys, json
import numpy as np, openmc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import uqlib, arc_model

def run(xml, tag, n=100_000):
    model, cells, p = arc_model.build(None, n // 10, 10, True)
    model.materials.cross_sections = xml
    model.settings.seed = 777
    flibe = [m for m in model.materials if m.name == "flibe"][0]
    t = openmc.Tally(name="tn"); t.filters = [openmc.MaterialFilter([flibe])]
    t.nuclides = ["Li6", "Li7", "Be9", "F19"]; t.scores = ["(n,Xt)"]
    model.tallies.append(t)
    d = f"/runs/runs/li7share_{tag}"
    sp = model.run(cwd=d, threads=8, output=False)
    with openmc.StatePoint(sp) as s:
        df = s.get_tally(name="tn").get_pandas_dataframe()
    return df[["nuclide", "score", "mean", "std. dev."]].to_string()

meta, bounds, C, k = plan = uqlib.load_plan("Li7")
f = np.ones((len(meta["params"]), len(bounds) - 1))
for p_, (m, _) in enumerate(meta["params"]):
    if 853 <= m <= 859:
        f[p_] = 1.5
d = "/runs/libs/li7share"; os.makedirs(d, exist_ok=True)
uqlib.write_perturbed("Li7", f, f"{d}/Li7.h5", plan)
uqlib.write_xs_xml({"Li7": f"{d}/Li7.h5"}, f"{d}/cross_sections.xml")
print("NOMINAL\n" + run("/work/blanket/xs/cross_sections.xml", "nom"))
print("Li7 levels 52-82 x1.5\n" + run(f"{d}/cross_sections.xml", "x15"))

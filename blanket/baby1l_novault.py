"""How much of the BABY 1L TBR is room return? Same BABY geometry and source,
vault removed: the 50 cm air sphere gets a vacuum boundary.

Run: docker compose run --rm openmc blanket/baby1l_novault.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ext/BABY-1L-run-4/analysis/neutron"))
import openmc
from openmc_model import baby_geometry
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond

run_dir = os.path.join(HERE, "runs/baby1l_novault")
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

x_c, y_c, z_c = 587, 60, 100
sphere, cllif_cell, cells = baby_geometry(x_c, y_c, z_c)
sphere.boundary_type = "vacuum"
mats = openmc.Materials({c.fill for c in cells if c.fill is not None})

settings = openmc.Settings()
settings.source = A325_generator_diamond((x_c, y_c, z_c - 5.635), (1, 0, 0))
settings.batches, settings.inactive, settings.particles = 100, 0, int(1e4)
settings.run_mode = "fixed source"
settings.output = {"tallies": False}

tbr = openmc.Tally(name="TBR")
tbr.scores = ["(n,Xt)"]
tbr.filters = [openmc.CellFilter(cllif_cell)]

model = openmc.Model(openmc.Geometry(cells), mats, settings, openmc.Tallies([tbr]))
model.run(threads=int(os.environ.get("OMP_NUM_THREADS", "24")))
sp = openmc.StatePoint(f"statepoint.{settings.batches}.h5")
t = sp.get_tally(name="TBR").get_pandas_dataframe()
out = {"no_vault": {"mean": float(t["mean"].iloc[0]), "std": float(t["std. dev."].iloc[0])}}
json.dump(out, open("tbr.json", "w"), indent=2)
print(json.dumps(out, indent=2))

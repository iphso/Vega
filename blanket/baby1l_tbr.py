"""Gate 0: reproduce a LIBRA run's modelled TBR with their own open OpenMC
model, then compare to the measured runs of that campaign.

Run: docker compose run --rm openmc blanket/baby1l_tbr.py [particles_per_batch] [repo]
  repo: BABY-1L-run-4 (default, ClLiF; compares runs 1-4) or FLiBe-1L-run-1
  (natural-Li FLiBe; compares FLiBe-1L runs 1-2)
"""
import json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[2] if len(sys.argv) > 2 else "BABY-1L-run-4"
CAMPAIGN = REPO.rsplit("-run-", 1)[0]
sys.path.insert(0, os.path.join(HERE, f"ext/{REPO}/analysis/neutron"))
import openmc
from openmc_model import baby_model

run_dir = os.path.join(HERE, "runs", CAMPAIGN.lower().replace("-", ""))
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

model = baby_model()
if len(sys.argv) > 1 and sys.argv[1] != "-":
    model.settings.particles = int(sys.argv[1])
model.settings.output = {"tallies": False}
model.run(threads=int(os.environ.get("OMP_NUM_THREADS", "24")))
sp = openmc.StatePoint(f"statepoint.{model.settings.batches}.h5")
t = sp.get_tally(name="TBR").get_pandas_dataframe()
tbr, sd = float(t["mean"].iloc[0]), float(t["std. dev."].iloc[0])

measured = {}
for f in sorted(glob.glob(os.path.join(HERE, f"ext/{CAMPAIGN}-run-*/data/processed_data.json"))):
    run = f.split(f"{CAMPAIGN}-")[1].split("/")[0]
    d = json.load(open(f))
    measured[run] = {"value": d["measured_TBR"]["value"], "error": d.get("measured_TBR_error", {}).get("value")}
    libra_model = d.get("modelled_TBR", {}).get("mean") or d.get("TBR_used_in_model", {}).get("value")
out = {"campaign": CAMPAIGN, "ours": {"mean": tbr, "std": sd}, "libra_modelled": libra_model, "measured": measured}
json.dump(out, open("tbr.json", "w"), indent=2)
print(json.dumps(out, indent=2))

"""Geometry parity: OpenMC's cell at random (R, Z) points of the ARC model, for
the TypeScript BlanketGeometry to reproduce (StoneLab test/neutron).

Run: docker compose run --rm openmc blanket/geom_points.py [N]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_model import build, HERE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
model, cells, p = build(None, photons=False)
name_of = {c.id: k for k, c in cells.items()}
rng = np.random.default_rng(7)
# half the points uniformly in the coil bounding box, half in a band around the VV
R = np.concatenate([rng.uniform(1, 700, N // 2), rng.uniform(200, 470, N // 2)])
Z = np.concatenate([rng.uniform(-500, 500, N // 2), rng.uniform(-230, 230, N // 2)])
out = []
for r, z in zip(R, Z):
    found = model.geometry.find((float(r), 0.0, float(z)))
    out.append(name_of[found[-1].id] if found else "outside")
json.dump({"params": p, "R": R.tolist(), "Z": Z.tolist(), "cell": out},
          open(os.path.join(HERE, "runs", "geom_points.json"), "w"))
from collections import Counter
print(Counter(out))

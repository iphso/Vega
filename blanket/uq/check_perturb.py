"""Offline consistency checks of write_perturbed: unity factors reproduce the library;
random factors keep MT205/MT4/... equal to the sum of their (perturbed) components."""
import os, sys
import numpy as np
import openmc.data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uqlib

out = os.path.join(uqlib.UQ_ROOT, "tmp")
for nuc in sys.argv[1:]:
    plan = uqlib.load_plan(nuc)
    meta, bounds, C, kerma = plan
    npar, ng = len(meta["params"]), len(bounds) - 1
    base = openmc.data.IncidentNeutron.from_hdf5(uqlib.base_h5(nuc))
    uqlib.write_perturbed(nuc, np.ones((npar, ng)), f"{out}/{nuc}_1.h5", plan)
    u = openmc.data.IncidentNeutron.from_hdf5(f"{out}/{nuc}_1.h5")
    md = max(np.max(np.abs(np.asarray(u.reactions[m].xs[T].y) - np.asarray(base.reactions[m].xs[T].y)))
             for m in base.reactions if m in u.reactions for T in base.reactions[m].xs)
    print(nuc, "unity: max |dy| =", md, " reactions", len(base.reactions), len(u.reactions))
    draw, neg = uqlib.lognormal_sampler(C)
    F = np.array([draw(np.random.default_rng(k)) for k in range(400)])
    sd = np.sqrt(np.diag(C)); fs = F.std(0)
    i = np.argmax(sd * (sd < 0.3))
    print(f"  sampler: mean {F.mean():.4f}; sd check (largest <30% cell) target {sd[i]:.4f} got {fs[i]:.4f}")
    f = F[0].reshape(npar, ng)
    uqlib.write_perturbed(nuc, f, f"{out}/{nuc}_p.h5", plan)
    p = openmc.data.IncidentNeutron.from_hdf5(f"{out}/{nuc}_p.h5")
    T = "294K"; E = p.energy[T]
    def full(d, m):
        t = d.reactions[m].xs[T]; j = np.searchsorted(E, t.x[0]); y = np.zeros(len(E)); y[j:j+len(t.y)] = t.y; return y
    for g, comp in meta["gas"].items():
        if meta["gas_fit_err"][g] > 1e-3: continue
        g = int(g); s = sum(c * full(p, int(r)) for r, c in comp.items())
        y = full(p, g); m = y > 1e-6 * y.max()
        print(f"  MT{g} vs sum of perturbed comps: max rel err {np.max(np.abs(s[m]-y[m])/y[m]):.2e}")
    for m in (2, 105, 16, 52, 301, 205):
        if m in p.reactions:
            r = p.reactions[m].xs[T](14.1e6) / max(base.reactions[m].xs[T](14.1e6), 1e-30)
            r2 = p.reactions[m].xs[T](0.0253) / max(base.reactions[m].xs[T](0.0253), 1e-30)
            print(f"  MT{m}: ratio at 14.1 MeV {r:.4f}, at thermal {r2:.4f}")
    for fn in (f"{out}/{nuc}_1.h5", f"{out}/{nuc}_p.h5"):
        os.remove(fn)

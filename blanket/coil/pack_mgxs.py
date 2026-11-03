"""A browser .neutpack (export_neutpack.py layout) back to an OpenMC MGXS library, so OpenMC's
multigroup mode transports exactly the constants the browser kernel samples: sigma_t, the
weight out of a collision w = nu_out / sigma_t, the sparse transfer rows and their 64-bin
cosine histograms. OpenMC MG samples absorption with probability sigma_a / sigma_t and
multiplies the weight by the multiplicity on scattering, so with
  sigma_a = max(0, sigma_t - nu_out),  sigma_s = sigma_t - sigma_a,  multiplicity = nu_out / sigma_s
its expected weight out of a collision is the browser's w, also where w > 1.
"""
import json, os
import numpy as np
import openmc, openmc.mgxs


def read_pack(d):
    m = json.load(open(os.path.join(d, "manifest.json")))
    data = np.fromfile(os.path.join(d, "data.bin"), "<f4")
    G, NMU = m["G"], m["nmu"]
    cells = {}
    for name, cm in m["cells"].items():
        g = lambda k: data[cm[k]["offset"]:cm[k]["offset"] + cm[k]["length"]]
        st, nu = g("sigma_t").astype(float), g("nu_out").astype(float)
        rp, tg, tc = g("row_ptr").astype(int), g("tr_g").astype(int), g("tr_cdf").astype(float)
        mu = g("mu_cdf").astype(float).reshape(-1, NMU + 1)
        ns = np.zeros((G, G, NMU))
        for gi in range(G):
            a, b = rp[gi], rp[gi + 1]
            if b <= a:
                continue
            p = np.diff(np.concatenate([[0.0], tc[a:b]]))
            for k in range(a, b):
                ns[gi, tg[k]] += nu[gi] * p[k - a] * np.diff(mu[k])
        cells[name] = dict(st=st, nu=nu, ns=ns)
    return m, cells


def to_mgxs(pack_dir, out_h5):
    m, cells = read_pack(pack_dir)
    G = m["G"]
    groups = openmc.mgxs.EnergyGroups(np.array(m["group_edges_eV"]))
    lib = openmc.MGXSLibrary(groups)
    for name, c in cells.items():
        x = openmc.XSdata(f"cell_{name}", groups, representation="isotropic")
        x.order = c["ns"].shape[2]
        x.scatter_format = "histogram"
        st, nu = c["st"], c["nu"]
        sa = np.maximum(0.0, st - nu)
        ss = st - sa
        mult = np.where(ss > 0, nu / np.where(ss > 0, ss, 1.0), 1.0)
        x.set_total(st)
        x.set_absorption(sa)
        x.set_scatter_matrix(c["ns"])                       # nu-scatter, as create_mg_library writes it
        x.set_multiplicity_matrix(np.repeat(np.maximum(mult, 1e-30)[:, None], G, axis=1))
        lib.add_xsdata(x)
    lib.export_to_hdf5(out_h5)
    return lib

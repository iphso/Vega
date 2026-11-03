"""Build the extra data the FNS benchmarks need on top of blanket/xs (ENDF/B-VIII.0 NNDC HDF5):
  - c_Be: Be-metal thermal scattering, NJOY-processed from the ENDF/B-VIII.0 thermal sublibrary
    (tsl-Be-metal.endf) at 296 K. The openmc-data-storage VIII.0 mirror lists c_Be.h5 but 404s.
  - Np237 (ENDF/B-VIII.0 NNDC HDF5, fetched by openmc_data_downloader into /fnsruns/xs_sab).
and write /fnsruns/xs/cross_sections.xml = base library + those two.

Run:  docker compose run --rm -v ~/runs/blanket/bench/fns:/fnsruns -v ~/runs/blanket/uq:/runs \
          openmc blanket/bench/fns/make_xs.py
"""
import os
import openmc.data
import fnslib

OUT = "/fnsruns/xs"
os.makedirs(OUT, exist_ok=True)
sab = os.path.join(OUT, "c_Be.h5")
if not os.path.exists(sab):
    d = openmc.data.ThermalScattering.from_njoy(
        "/runs/endf/n-004_Be_009.endf", "/fnsruns/dl/tsl-Be-metal.endf",
        temperatures=[296.0], stdout=True)
    d.name = "c_Be"
    d.export_to_hdf5(sab, "w")
    print("wrote", sab, d.temperatures, d.nuclides)
fnslib.write_xs_xml({}, os.path.join(OUT, "cross_sections.xml"))
print(open(os.path.join(OUT, "cross_sections.xml")).read()[-600:])

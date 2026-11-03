"""Thermal scattering laws for graphite and Be metal (ENDF/B-VIII.0 TSL at 296 K, the lowest tabulated temperature; NJOY via openmc.data).
blanket/xs has no S(a,b) files; the OpenMC data repository mirror lacks them too, so they are
processed here from the NNDC ENDF/B-VIII.0 thermal sublibrary into /bench/xs_sab/*.h5.
Reactor graphite 30% porosity is used for the reflector blocks (closest TSL to ~1.6-1.7 g/cm3
blocks); crystalline graphite is also processed for a sensitivity check."""
import openmc.data
D = "/bench/xs_sab"
for name, tsl, n in [("c_Graphite_30p", "tsl-reactor-graphite-30P.endf", "n-006_C_012.endf"),
                     ("c_Graphite", "tsl-crystalline-graphite.endf", "n-006_C_012.endf"),
                     ("c_Be", "tsl-Be-metal.endf", "n-004_Be_009.endf")]:
    t = openmc.data.ThermalScattering.from_njoy(f"{D}/{n}", f"{D}/{tsl}", temperatures=[296.0], stdout=False)
    t.name = name
    t.export_to_hdf5(f"{D}/{name}.h5", "w")
    print(name, t.nuclides, list(t.kTs))

#!/bin/sh
# The ENDF/B-VIII.0 depletion chain OpenMC publishes (openmc-data-storage/openmc_data),
# 3820 nuclides with decay photon spectra. ~27 MB, not committed.
set -e
cd "$(dirname "$0")/data"
curl -sSLf -o chain_endf_b8.0.xml https://raw.githubusercontent.com/openmc-data-storage/openmc_data/main/src/openmc_data/depletion/chain_endf_b8.0.xml
sha256sum chain_endf_b8.0.xml

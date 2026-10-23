#!/bin/bash
# Mesh CASE and initialise it from SOURCE's latest time (mapFields -consistent; same geometry).
#   ./map_from.sh CASE SOURCE      (both on the host; runs in Docker)
set -e
C="$(readlink -f "$1")"; SRC="$(readlink -f "$2")"
IMAGE=${IMAGE:-vega-openfoam:latest}
docker run --rm -u "$(id -u):$(id -g)" -v "$C":/case -v "$SRC":/src -w /case --entrypoint bash "$IMAGE" -c \
  '. /opt/openfoam10/etc/bashrc && blockMesh > log.blockMesh 2>&1 && checkMesh > log.checkMesh 2>&1 &&
   mapFields /src -consistent -sourceTime latestTime > log.mapFields 2>&1 && rm -f 0/C 0/Cx 0/Cy 0/Cz 0/V 0/yPlus; tail -4 log.mapFields'
T=$(ls "$SRC" | grep -E '^[0-9.]+$' | sort -g | tail -1)
mkdir -p "$C/0/uniform"; cp "$SRC/$T/uniform/momentumForceProperties" "$C/0/uniform/" 2>/dev/null || true

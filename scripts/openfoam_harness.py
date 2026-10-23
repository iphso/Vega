"""Generic OpenFOAM case-running/parsing utilities (EXPERIMENT_LOG §65) --
domain-agnostic on purpose. Direct user intent: "we'll probably ultimately
want multiple oracles eventually that use openfoam as a backend," so the
genuinely reusable 20% (run a region-based case, read a boundary field's
face values back out) lives here, separate from `mug_openfoam_oracle.py`'s
mug-specific geometry/material/target logic. Nothing here knows what a mug
is.

Assumes it's running INSIDE the `openfoam` service (Dockerfile.openfoam --
`/entry_fixed.sh` already sources OpenFOAM's own bashrc, so `blockMesh`/
`chtMultiRegionFoam`/etc. are just on PATH; this module doesn't re-source
anything or manage Docker itself).
"""
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


class OpenFOAMError(RuntimeError):
    """A meshing or solve step failed -- .log holds the tool's own stdout+stderr."""

    def __init__(self, step, returncode, log):
        super().__init__(f"{step} failed (exit {returncode}); last 2000 chars:\n{log[-2000:]}")
        self.step = step
        self.returncode = returncode
        self.log = log


@contextmanager
def temp_case_dir(prefix="foam_case_"):
    """A fresh scratch case directory under /tmp, cleaned up on exit
    regardless of success/failure -- every candidate in a dataset-
    generation run gets its own, never reused (no risk of a previous
    candidate's mesh/fields leaking into the next one)."""
    d = tempfile.mkdtemp(prefix=prefix)
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def write_dict(path, foam_class, obj_name, body, location=None):
    """Writes one OpenFOAM dictionary/field file. `body` is the dictionary
    content as a plain string (caller's responsibility to get the syntax
    right) -- this just wraps it in the required FoamFile header, which
    every one of these files needs identically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    loc_line = f'    location    "{location}";\n' if location else ""
    path.write_text(
        "FoamFile\n{\n    format      ascii;\n    class       "
        f"{foam_class};\n{loc_line}    object      {obj_name};\n}}\n\n{body}\n"
    )


def read_probe_series(probe_path):
    """Parses a `probes`/`patchProbes` functionObject output file (§68 --
    one probe location only, this project's own use case so far) into a
    list of (time, value) tuples. Format: `# Probe 0 (x y z)` then
    `# Time  0` then whitespace-separated `time value` rows -- confirmed
    directly against a real run's output before trusting this, not
    guessed from docs."""
    rows = []
    for line in probe_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t_str, v_str = line.split()[:2]
        rows.append((float(t_str), float(v_str)))
    return rows


def run_tool(case_dir, args, step_name=None, timeout=120):
    """Runs one OpenFOAM utility/solver (e.g. ["blockMesh", "-region",
    "solid"] or ["chtMultiRegionFoam"]) with cwd=case_dir. Raises
    OpenFOAMError (not a bare CalledProcessError) on nonzero exit OR
    timeout -- both are real candidate-level failures a caller's own
    try/except ok=False handling (this project's usual oracle_harness
    convention) should catch the same way."""
    step_name = step_name or args[0]

    def _as_str(x):
        # §75 bug: subprocess.TimeoutExpired's own .stdout/.stderr can
        # still be bytes even with text=True on the parent run() call (a
        # real, observed Python 3.8 quirk, not hypothetical -- caught
        # when a genuinely slow solve hit this exact path and the
        # TypeError masked the real OpenFOAMError). Decode defensively
        # regardless of which type shows up.
        if x is None:
            return ""
        return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x

    try:
        result = subprocess.run(
            args, cwd=case_dir, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise OpenFOAMError(step_name, -1, _as_str(e.stdout) + _as_str(e.stderr) + "\n[TIMEOUT]")
    log = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "FOAM FATAL ERROR" in log:
        raise OpenFOAMError(step_name, result.returncode, log)
    return log


# Matches `<patchName>\n{\n ... <entry> ... uniform <number>;\n}` at the
# boundaryField top level, brace-depth-aware so it doesn't grab a nested
# sub-dict's own same-named entry (e.g. externalWallHeatFluxTemperature's
# `Ta { type uniform; value ...; }` sub-block, one level deeper) -- this
# tripped up a naive regex during the initial validation spike.
def read_boundary_field(field_path, entry="value"):
    """Parses a written field file (e.g. `<time>/<region>/T`) and returns
    {patchName: numpy array of face values} for every patch with an
    `entry` field, uniform or nonuniform. Region-agnostic, field-name-
    agnostic -- works for T or any other volScalarField. `entry` defaults
    to "value" (every patch has one) but any `mixed`-derived BC's other
    scalar entries work too -- e.g. "refValue" to read back a coupled
    BC's own internal state (mug_openfoam_oracle.py's liquidCoupling BC
    uses this to report temp_at_2h_C, §66)."""
    import numpy as np

    text = field_path.read_text()
    m = re.search(r"boundaryField\s*\{", text)
    if not m:
        raise ValueError(f"no boundaryField found in {field_path}")
    body = text[m.end():]

    out = {}
    i = 0
    while i < len(body):
        pm = re.match(r"\s*([A-Za-z0-9_.]+)\s*\{", body[i:])
        if not pm:
            i += 1
            continue
        patch_name = pm.group(1)
        block_start = i + pm.end()
        depth = 1
        j = block_start
        while depth > 0 and j < len(body):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        patch_body = body[block_start:j - 1]
        # top-level `entry` in THIS patch's own body -- exclude anything
        # inside a nested `{...}` sub-dict by stripping those first.
        stripped = re.sub(r"\{[^{}]*\}", "", patch_body)
        vm = re.search(rf"\b{entry}\s+uniform\s+([-+0-9.eE]+)\s*;", stripped)
        lm = re.search(rf"\b{entry}\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\(([^)]*)\)", patch_body, re.S)
        if lm:
            out[patch_name] = np.array([float(x) for x in lm.group(2).split()])
        elif vm:
            out[patch_name] = np.array([float(vm.group(1))])
        i = j
    return out

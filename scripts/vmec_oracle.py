"""VMEC++/ConStellaration implementation of oracle_base.py's interface --
the one domain this project actually has, kept separate from the generic
harness (eval_cvae_steerability.py, oracle_harness.py) so a future different
domain (anything of the shape "parameters get scored into named metrics,
possibly failing") can plug in beside it without touching the harness.

Consolidates what used to be duplicated across generate_and_validate.py and
eval_cvae_steerability.py's own first version: the per-candidate subprocess
worker (VMEC++ can hang instead of failing fast on a pathological boundary --
see generate_and_validate.py's docstring for the concrete story), the
fidelity-preset names used throughout this project, and the structurally-
fixed coefficient indices.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/work/output")

TARGET_NAMES = json.loads((OUT_DIR / "target_names.json").read_text())

# Kept in sync with make_splits.py's own copy (duplicated rather than
# imported so this module has no dependency on the splits/clustering code
# path -- an oracle shouldn't need to know how its own metrics get used
# downstream).
LOG_TARGET_NAMES = [
    "qi", "max_elongation", "flux_compression_in_regions_of_bad_curvature",
    "minimum_normalized_magnetic_gradient_scale_length",
]

PARAM_DIM = 90

# r_cos m=0,n<4 and z_sin m=0 entirely -- structurally always zero by
# stellarator symmetry (see EXPERIMENT_LOG / metadata.json feature_stats).
ZERO_INDICES = [0, 1, 2, 3, 45, 46, 47, 48, 49]

FIDELITY_PRESETS = {"low": "low_fidelity", "medium": "from_boundary_resolution", "high": "high_fidelity"}


def worker_fn(conn, r_cos, z_sin, nfp, fidelity_name):
    """Runs in its own throwaway subprocess -- see oracle_harness.py. Sends
    (True, metrics_dict) or (False, error_string) over `conn`."""
    try:
        from constellaration import forward_model
        from constellaration.geometry import surface_rz_fourier
        from constellaration.mhd.vmec_settings import VmecPresetSettings
        try:
            boundary = surface_rz_fourier.SurfaceRZFourier(
                r_cos=r_cos, z_sin=z_sin, n_field_periods=nfp, is_stellarator_symmetric=True,
            )
        except Exception as e:
            conn.send((False, f"structural: {e}"))
            return
        settings = forward_model.ConstellarationSettings(
            vmec_preset_settings=VmecPresetSettings(fidelity=fidelity_name)
        )
        try:
            metrics, _ = forward_model.forward_model(boundary, settings=settings)
        except Exception as e:
            conn.send((False, f"vmec: {e}"))
            return
        conn.send((True, metrics.model_dump()))
    except Exception as e:
        conn.send((False, f"unknown: {e}"))
    finally:
        conn.close()


def params_to_worker_args(params, aux, fidelity_name):
    """params: (90,) float array, already unstandardized and already
    zero-enforced at ZERO_INDICES. aux: {"nfp": int}."""
    r_cos = params[:45].reshape(5, 9).astype(np.float64)
    z_sin = params[45:90].reshape(5, 9).astype(np.float64)
    return (r_cos, z_sin, int(aux["nfp"]), fidelity_name)

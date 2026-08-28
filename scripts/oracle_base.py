"""The interface a domain module must implement to plug into
eval_cvae_steerability.py's generic ablation harness (validity + steerability
+ selectivity, across arbitrary generative-model architectures) via
oracle_harness.py's domain-agnostic subprocess-timeout batch runner.

The idea this project actually cares about generalizing: "some parameters
get scored into some named metrics, possibly failing outright" is not
specific to stellarator boundaries and VMEC++ -- it's the same shape as
scoring a molecule's descriptors through a property predictor, a circuit's
component values through a SPICE sim, a control policy's gains through a
stability check, anything with a params -> scores(or failure) oracle and a
question of "can a generative model conditioned on desired scores produce
valid params that hit them." vmec_oracle.py is the one implementation this
project actually has; this file exists so it isn't the only one *possible*.

A domain module (see vmec_oracle.py for the reference implementation) must
provide:

  TARGET_NAMES: list[str]
      Names of the scored metrics, in the fixed order every metrics vector
      uses throughout the harness.

  LOG_TARGET_NAMES: list[str] (subset of TARGET_NAMES, may be empty)
      Metrics whose dynamic range is wide enough that raw-space z-scoring/
      clustering/distance would be dominated by rare outliers -- log-
      transformed before any of that. See make_splits.py's own docstring
      for why this project's 4 flagged targets needed it.

  PARAM_DIM: int
      Dimensionality of the flat parameter vector a generative model
      produces and the oracle scores.

  ZERO_INDICES: list[int] (may be empty)
      Indices into the param vector that are structurally required to be
      exactly zero (e.g. by a symmetry constraint) -- hard-enforced on every
      generated candidate rather than trusting a generative model to land on
      exact zeros itself.

  FIDELITY_PRESETS: dict[str, str]
      Named fidelity levels -> whatever internal setting identifier the
      domain's own scorer understands. Callers pass the dict key
      (e.g. "low"/"medium"/"high"); the value is domain-specific and opaque
      to the harness.

  def worker_fn(conn, *worker_args) -> None
      Runs in its own throwaway subprocess (see oracle_harness.py -- some
      domains, VMEC++ among them, can hang instead of failing fast on a
      pathological input, so a persistent worker pool isn't safe; a fresh
      subprocess per candidate can just be killed). Must send exactly one
      `(True, metrics_dict)` or `(False, error_string)` over `conn`, and
      `conn.close()` in a `finally` block. `metrics_dict` keys must be a
      superset of TARGET_NAMES (extra keys are fine and ignored).

  def params_to_worker_args(params, aux, fidelity_name) -> tuple
      Converts a raw (already unstandardized, already zero-enforced) flat
      parameter vector plus `aux` (a domain-specific dict of any auxiliary
      discrete conditioning the generative model also produces alongside
      the continuous parameters -- e.g. this project's `n_field_periods`,
      which isn't part of the 90-dim Fourier-coefficient vector but is still
      needed to score a candidate) into the exact positional args
      `worker_fn` expects after `conn`. The harness builds each oracle job
      as `(tag, *params_to_worker_args(...))` and hands the list straight to
      oracle_harness.run_batch_with_timeout.
"""

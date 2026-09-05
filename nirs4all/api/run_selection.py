"""Pre-execution selection for the general run API, without executing operators.

The explicit portable profile remains strict. Automatic selection is a capability
decision, never a retry after a backend, data validation, or operator failure.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from nirs4all.pipeline.engine import ENGINE_ENV_VAR, Engine, resolve_engine


def select_run_engine(
    engine: str | None,
    pipeline: Any,
    dataset: Any,
    *,
    allow_fallback: bool = False,
    **options: Any,
) -> Engine:
    """Respect selectors, otherwise choose the portable or general DAG profile.

    No data are loaded or fitted, no runtime is probed, and missing native
    dependencies cannot cause an implicit change of engine. Invalid portable
    data remain errors in the selected engine rather than triggering retries.
    """
    selected = resolve_engine(engine, allow_fallback=allow_fallback)
    if engine is not None or os.environ.get(ENGINE_ENV_VAR, "").strip():
        return selected
    if not isinstance(pipeline, list) or not isinstance(dataset, Mapping):
        return "dag-ml"
    keys = set(dataset)
    if not {"X", "y", "sample_ids"} <= keys or keys - {"X", "y", "sample_ids", "target_names", "groups", "metadata"}:
        return "dag-ml"
    if (
        options.get("tuning") is not None
        or options.get("calibration") is not None
        or options.get("save_artifacts", True) is not True
        or options.get("save_charts") is True
        or options.get("plots_visible", False)
        or options.get("refit", True) is not True
        or any(options.get(key) is not None for key in ("cache", "project", "results_path"))
        or options.get("report_naming", "nirs") != "nirs"
        or set(options.get("runner_kwargs", {})) - {"methods_library_path"}
    ):
        return "dag-ml"

    from nirs4all.pipeline.dagml.raw_training_lowerer import _portable_methods_pls_params, _supported_linear_steps

    from .native_archive_training import _extract_portable_methods_hpo

    seed = options.get("random_state")
    # These functions only validate/normalize declarations. They do not invoke
    # split(), fit(), transform(), load an archive, or inspect a runtime.
    try:
        portable, _ = _extract_portable_methods_hpo(pipeline, seed=12345 if seed is None else seed)
        steps, _, _ = _supported_linear_steps(portable)
        _portable_methods_pls_params(steps)
    except (TypeError, ValueError, NotImplementedError):
        return "dag-ml"
    return "native"

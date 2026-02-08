#!/usr/bin/env python3
"""
CI example launcher with fast-mode runtime caps.

This wrapper runs an example script while applying conservative speed caps
for CI use only. It keeps user-facing examples unchanged when launched
directly (e.g., `python example.py`).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import runpy
import sys
from pathlib import Path
from typing import Any


FAST_ENV = "NIRS4ALL_EXAMPLE_FAST"
FAST_DEFAULT = "1"


def _set_thread_limits() -> None:
    # Keep worker processes lightweight and reduce CPU oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _is_fast_mode() -> bool:
    value = os.environ.get(FAST_ENV, FAST_DEFAULT).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _cap_int(value: Any, cap: int) -> Any:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return value
    return min(ivalue, cap)


def _cap_float(value: Any, cap: float) -> Any:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return value
    return min(fvalue, cap)


def _cap_param_map(params: dict[str, Any]) -> dict[str, Any]:
    capped = copy.deepcopy(params)
    int_caps = {
        "n_splits": 2,
        "n_repeats": 1,
        "n_estimators": 16,
        "max_depth": 8,
        "max_iter": 100,
        "epochs": 5,
        "num_epochs": 5,
        "batch_size": 128,
        "cv": 2,
    }
    float_caps = {
        "max_factor": 1.5,
        "ref_percentage": 0.75,
    }

    for key, cap in int_caps.items():
        if key in capped:
            capped[key] = _cap_int(capped[key], cap)
    for key, cap in float_caps.items():
        if key in capped:
            capped[key] = _cap_float(capped[key], cap)
    return capped


def _shrink_list(items: list[Any], cap: int) -> list[Any]:
    if len(items) <= cap:
        return items
    return items[:cap]


def _looks_like_model_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if "model" in step:
        return True
    if "_or_" in step:
        values = step.get("_or_")
        return isinstance(values, list)
    return False


def _optimize_object(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_optimize_object(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_optimize_object(x) for x in obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            lowered = key.lower()
            optimized = _optimize_object(value)

            if lowered in {"n_trials", "trials"}:
                out[key] = _cap_int(optimized, 1)
                continue
            if lowered in {"n_splits", "n_repeats"}:
                out[key] = _cap_int(optimized, 2 if lowered == "n_splits" else 1)
                continue
            if lowered in {"count", "pick", "bins"}:
                if lowered == "bins":
                    out[key] = _cap_int(optimized, 3)
                else:
                    out[key] = _cap_int(optimized, 1)
                continue
            if lowered == "target_size":
                out[key] = _cap_int(optimized, 10)
                continue
            if lowered in {"max_iter", "epochs", "num_epochs"}:
                out[key] = _cap_int(optimized, 100 if lowered == "max_iter" else 5)
                continue
            if lowered == "n_estimators":
                out[key] = _cap_int(optimized, 16)
                continue
            if lowered in {"_or_", "transformers", "feature_augmentation"} and isinstance(optimized, list):
                out[key] = _shrink_list(optimized, 2)
                continue
            if lowered in {"dataset", "datasets"} and isinstance(optimized, list):
                out[key] = _shrink_list(optimized, 2)
                continue
            if lowered == "model_params" and isinstance(optimized, dict):
                params = copy.deepcopy(optimized)
                # Keep search spaces tiny in CI.
                for pkey, pvalue in list(params.items()):
                    if isinstance(pvalue, list):
                        params[pkey] = _shrink_list(pvalue, 2)
                    elif isinstance(pvalue, tuple) and pvalue:
                        kind = pvalue[0]
                        if kind in {"int", "int_log"} and len(pvalue) >= 3:
                            params[pkey] = (kind, pvalue[1], min(pvalue[2], pvalue[1] + 3))
                        elif kind in {"float", "float_log"} and len(pvalue) >= 3:
                            params[pkey] = (kind, pvalue[1], pvalue[2])
                out[key] = params
                continue

            out[key] = optimized
        return out

    # sklearn-like objects: cap known heavy params.
    if hasattr(obj, "get_params") and hasattr(obj, "set_params"):
        try:
            params = obj.get_params(deep=False)
            updates = _cap_param_map(params)
            delta = {k: v for k, v in updates.items() if params.get(k) != v}
            if delta:
                obj.set_params(**delta)
        except Exception:
            pass
    return obj


def _optimize_pipeline_spec(pipeline: Any) -> Any:
    optimized = _optimize_object(pipeline)
    if isinstance(optimized, list):
        # Single pipeline: keep a small model set for CI speed.
        if optimized and not isinstance(optimized[0], list):
            model_count = 0
            slim_steps = []
            for step in optimized:
                if isinstance(step, str) and "chart" in step.lower():
                    # Visualization steps are useful for tutorials but expensive for CI.
                    continue
                if _looks_like_model_step(step):
                    model_count += 1
                    if model_count > 2:
                        continue
                slim_steps.append(step)
            optimized = slim_steps
        else:
            # List of pipelines.
            optimized = _shrink_list(optimized, 3)
    return optimized


def _optimize_dataset_spec(dataset: Any) -> Any:
    optimized = _optimize_object(dataset)
    if isinstance(optimized, list):
        optimized = _shrink_list(optimized, 2)
    return optimized


def _patch_nirs4all_fast_mode() -> None:
    import nirs4all
    from nirs4all.pipeline.runner import PipelineRunner
    run_api = importlib.import_module("nirs4all.api.run")

    original_run = run_api.run
    original_pr_init = PipelineRunner.__init__
    original_pr_run = PipelineRunner.run

    def fast_run(pipeline: Any, dataset: Any, **kwargs: Any) -> Any:
        pipeline = _optimize_pipeline_spec(pipeline)
        dataset = _optimize_dataset_spec(dataset)

        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("show_spinner", False)
        kwargs.setdefault("show_progress_bar", False)
        kwargs.setdefault("plots_visible", False)
        kwargs.setdefault("save_charts", False)
        return original_run(pipeline, dataset, **kwargs)

    def fast_pr_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("show_spinner", False)
        kwargs.setdefault("show_progress_bar", False)
        kwargs.setdefault("plots_visible", False)
        kwargs.setdefault("save_charts", False)
        original_pr_init(self, *args, **kwargs)

    def fast_pr_run(self: Any, pipeline: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
        pipeline = _optimize_pipeline_spec(pipeline)
        dataset = _optimize_dataset_spec(dataset)
        kwargs["max_generation_count"] = min(int(kwargs.get("max_generation_count", 10000)), 128)
        return original_pr_run(self, pipeline, dataset, *args, **kwargs)

    run_api.run = fast_run
    nirs4all.run = fast_run
    PipelineRunner.__init__ = fast_pr_init
    PipelineRunner.run = fast_pr_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run examples with CI fast mode.")
    parser.add_argument("example", help="Path to the example script")
    parser.add_argument("example_args", nargs=argparse.REMAINDER, help="Args for the example script")
    args = parser.parse_args()

    _set_thread_limits()

    example_path = Path(args.example).resolve()
    if not example_path.exists():
        print(f"Example not found: {example_path}", file=sys.stderr)
        return 2

    # Keep imports and relative paths consistent with direct script execution.
    if "examples" in example_path.parts:
        idx = example_path.parts.index("examples")
        examples_dir = Path(*example_path.parts[: idx + 1])
    else:
        examples_dir = example_path.parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    if _is_fast_mode():
        _patch_nirs4all_fast_mode()

    # Forward argv as if the example was launched directly.
    sys.argv = [str(example_path)] + args.example_args

    try:
        runpy.run_path(str(example_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

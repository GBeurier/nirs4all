from __future__ import annotations

import json
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import nirs4all
from tests.integration.parity import cases_generators  # noqa: F401 - registers generator cases
from tests.integration.parity._datasets import dataset_path
from tests.integration.parity._registry import get as get_case


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _run_timed(engine: str, artifacts_dir: Path) -> tuple[Any, float, list[str]]:
    case = get_case("generator_zip_paired")
    kwargs: dict[str, Any] = {
        "engine": engine,
        "verbose": 0,
        "save_artifacts": False,
        "save_charts": False,
        "random_state": 42,
        "refit": True,
    }
    if engine == "dag-ml":
        kwargs["results_path"] = str(artifacts_dir / "native-results")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        result = nirs4all.run(case.pipeline_factory(), dataset_path(case.dataset_key), **kwargs)
        elapsed = time.perf_counter() - start
    warning_messages = [str(warning.message) for warning in caught]
    if engine == "dag-ml":
        fallback_warnings = [message for message in warning_messages if "falling back to the legacy engine" in message]
        assert not fallback_warnings, fallback_warnings
        assert result._is_dagml_engine(), "engine='dag-ml' must not report a legacy fallback result"  # noqa: SLF001
        native_results_dir = getattr(result, "_dagml_results_dir", None)
        assert native_results_dir is not None, "engine='dag-ml' with results_path must expose native results"
        assert Path(native_results_dir).exists(), native_results_dir
    return result, elapsed, warning_messages


def _result_summary(result: Any, elapsed_seconds: float, warning_messages: list[str]) -> dict[str, Any]:
    best = result.best
    return {
        "elapsed_seconds": elapsed_seconds,
        "warnings": warning_messages,
        "best_rmse": float(result.best_rmse),
        "best_score": float(result.best_score),
        "cv_best_score": float(result.cv_best_score),
        "num_predictions": int(result.num_predictions),
        "selected": {
            "model_name": str(best.get("model_name") or ""),
            "model_classname": str(best.get("model_classname") or ""),
            "metric": str(best.get("metric") or ""),
            "fold_id": best.get("fold_id"),
            "variant_id": best.get("variant_id"),
            "prediction_rows": int(_as_vector(best.get("y_pred")).shape[0]),
        },
    }


def test_generate_family(artifacts_dir: Path) -> None:
    case = get_case("generator_zip_paired")
    legacy, legacy_seconds, legacy_warnings = _run_timed("legacy", artifacts_dir)
    dagml, dagml_seconds, dagml_warnings = _run_timed("dag-ml", artifacts_dir)

    legacy_pred = _as_vector(legacy.best["y_pred"])
    dagml_pred = _as_vector(dagml.best["y_pred"])
    legacy_true = _as_vector(legacy.best["y_true"])
    dagml_true = _as_vector(dagml.best["y_true"])

    assert legacy_pred.shape == dagml_pred.shape
    assert legacy_true.shape == dagml_true.shape

    parity = {
        "best_rmse_abs": abs(float(legacy.best_rmse) - float(dagml.best_rmse)),
        "best_score_abs": abs(float(legacy.best_score) - float(dagml.best_score)),
        "cv_best_score_abs": abs(float(legacy.cv_best_score) - float(dagml.cv_best_score)),
        "prediction_abs_max": float(np.max(np.abs(legacy_pred - dagml_pred))) if legacy_pred.size else 0.0,
        "target_abs_max": float(np.max(np.abs(legacy_true - dagml_true))) if legacy_true.size else 0.0,
        "prediction_rows": int(legacy_pred.shape[0]),
        "tolerance": 1e-8,
    }
    assert parity["best_rmse_abs"] <= 1e-9
    assert parity["best_score_abs"] <= 1e-9
    assert parity["prediction_abs_max"] <= parity["tolerance"]
    assert parity["target_abs_max"] <= parity["tolerance"]

    speedup = legacy_seconds / dagml_seconds if dagml_seconds > 0 else None
    family = {
        "schema_version": "n4a.e2e.pipeline_generation_performance/v1",
        "status": "passed",
        "repo": "nirs4all",
        "git_head": _git_head(),
        "case": {
            "name": case.name,
            "description": case.description,
            "dataset_key": case.dataset_key,
            "keywords": list(case.keywords),
            "capabilities": list(case.capabilities),
            "expected_min_predictions": case.expected_min_predictions,
            "variants": [
                {"n_components": 5, "scale": True},
                {"n_components": 10, "scale": False},
                {"n_components": 15, "scale": True},
            ],
        },
        "runs": {
            "legacy": _result_summary(legacy, legacy_seconds, legacy_warnings),
            "dag_ml": _result_summary(dagml, dagml_seconds, dagml_warnings),
        },
        "parity": parity,
        "performance": {
            "legacy_seconds": legacy_seconds,
            "dag_ml_seconds": dagml_seconds,
            "legacy_over_dag_ml_ratio": speedup,
            "verdict": "dag_ml_faster" if speedup is not None and speedup > 1.0 else "no_speedup_on_this_run",
        },
        "native_results_dir": str(artifacts_dir / "native-results"),
    }

    (artifacts_dir / "pipeline-family.json").write_text(json.dumps(family, indent=2) + "\n", encoding="utf-8")

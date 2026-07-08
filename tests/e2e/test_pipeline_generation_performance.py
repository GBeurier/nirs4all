from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import nirs4all
from nirs4all.data import DatasetConfigs
from tests.integration.parity import cases_generators  # noqa: F401 - registers generator cases
from tests.integration.parity._datasets import dataset_path
from tests.integration.parity._registry import get as get_case

_WEB_WASM_PREDICTION_TOLERANCE = 5e-4


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _assert_native_results_dir(value: Any) -> Path:
    assert value is not None, "engine='dag-ml' with results_path must expose native results"
    native_results_dir = Path(value)
    assert native_results_dir.is_dir(), native_results_dir
    for filename in ("manifest.json", "score_set.json", "predictions.parquet"):
        assert (native_results_dir / filename).is_file(), native_results_dir / filename
    artifact_files = sorted((native_results_dir / "artifacts").glob("*.joblib"))
    assert artifact_files, native_results_dir / "artifacts"
    return native_results_dir


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _class_path(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(path: str) -> type[Any]:
    module_name, _, class_name = path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"invalid class path {path!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, class_name)
    if not isinstance(value, type):
        raise TypeError(f"{path!r} did not resolve to a class")
    return value


def _step_descriptor(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        descriptor: dict[str, Any] = {}
        for key, value in step.items():
            if key == "model" and isinstance(value, type):
                descriptor[key] = {"class": _class_path(value)}
            else:
                descriptor[key] = value
        return descriptor

    params: dict[str, Any] = {}
    get_params = getattr(step, "get_params", None)
    if callable(get_params):
        params = {key: value for key, value in get_params(deep=False).items() if _jsonable(value)}
    if not params:
        for name, parameter in inspect.signature(step.__class__).parameters.items():
            if name == "self" or parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
                continue
            if hasattr(step, name):
                value = getattr(step, name)
                if _jsonable(value):
                    params[name] = value
    return {"class": _class_path(step.__class__), "params": params}


def _step_from_descriptor(step: dict[str, Any]) -> Any:
    if "class" in step:
        return _import_class(step["class"])(**dict(step.get("params") or {}))
    if "model" in step and isinstance(step["model"], dict):
        hydrated = {key: value for key, value in step.items() if key != "model"}
        hydrated["model"] = _import_class(step["model"]["class"])
        return hydrated
    return dict(step)


def _variants_from_descriptor(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for step in pipeline:
        zipped = step.get("_zip_")
        if isinstance(zipped, dict) and zipped:
            keys = list(zipped)
            values = [list(zipped[key]) for key in keys]
            if not values:
                return []
            lengths = {len(items) for items in values}
            if len(lengths) != 1:
                raise ValueError(f"uneven _zip_ descriptor lengths: {sorted(lengths)}")
            return [dict(zip(keys, items, strict=True)) for items in zip(*values, strict=True)]
    return []


def _candidate_descriptor(case: Any) -> dict[str, Any]:
    pipeline = [_step_descriptor(step) for step in case.pipeline_factory()]
    return {
        "schema_version": "n4a.e2e.generated_pipeline_candidate.v1",
        "scenario_id": "e2e-pipeline-generation-performance-compare",
        "status": "passed",
        "case_name": case.name,
        "description": case.description,
        "dataset_key": case.dataset_key,
        "keywords": list(case.keywords),
        "capabilities": list(case.capabilities),
        "expected_min_predictions": case.expected_min_predictions,
        "variants": _variants_from_descriptor(pipeline),
        "pipeline": pipeline,
    }


def _pipeline_from_descriptor(descriptor: dict[str, Any]) -> list[Any]:
    return [_step_from_descriptor(step) for step in descriptor["pipeline"]]


def _class_sequence(descriptor: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for step in descriptor["pipeline"]:
        if isinstance(step, dict) and isinstance(step.get("class"), str):
            sequence.append(step["class"])
        elif isinstance(step, dict) and isinstance(step.get("model"), dict):
            sequence.append(step["model"]["class"])
    return sequence


def _selected_model_identity(selected: dict[str, Any]) -> str:
    return str(selected.get("model_name") or selected.get("model_classname") or "")


def _sample_indices(best: dict[str, Any]) -> list[int]:
    values = np.asarray(best.get("sample_indices"), dtype=int).reshape(-1)
    return [int(value) for value in values]


def _selected_generator_choices(best: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = best.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        return []
    choices = metadata.get("generator_choices")
    return choices if isinstance(choices, list) else []


def _prediction_oracle(
    legacy: Any,
    dagml: Any,
    legacy_pred: np.ndarray,
    dagml_pred: np.ndarray,
    legacy_true: np.ndarray,
    dagml_true: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    legacy_indices = _sample_indices(legacy.best)
    dagml_indices = _sample_indices(dagml.best)
    assert legacy_indices == dagml_indices
    legacy_pred = legacy_pred.reshape(-1)
    dagml_pred = dagml_pred.reshape(-1)
    legacy_true = legacy_true.reshape(-1)
    dagml_true = dagml_true.reshape(-1)
    assert len(legacy_indices) == legacy_pred.shape[0] == dagml_pred.shape[0] == legacy_true.shape[0] == dagml_true.shape[0]

    rows = []
    for i, sample_index in enumerate(legacy_indices):
        actual = float(legacy_true[i])
        dagml_actual = float(dagml_true[i])
        legacy_value = float(legacy_pred[i])
        dagml_value = float(dagml_pred[i])
        rows.append(
            {
                "sample_index": sample_index,
                "sample_id": str(sample_index),
                "actual": actual,
                "dag_ml_actual": dagml_actual,
                "legacy_predicted": legacy_value,
                "dag_ml_predicted": dagml_value,
                "legacy_residual": legacy_value - actual,
                "dag_ml_residual": dagml_value - dagml_actual,
                "prediction_abs_delta": abs(legacy_value - dagml_value),
                "target_abs_delta": abs(actual - dagml_actual),
            }
        )

    return {
        "schema_version": "n4a.e2e.python_dagml_prediction_oracle.v1",
        "status": "passed",
        "partition": str(legacy.best.get("partition") or "test"),
        "selected": {
            "legacy_model": _selected_model_identity(legacy.best),
            "dag_ml_model": _selected_model_identity(dagml.best),
            "metric": str(legacy.best.get("metric") or ""),
            "generator_choices": _selected_generator_choices(legacy.best),
        },
        "rows": rows,
        "prediction_rows": len(rows),
        "tolerance": tolerance,
        "web_wasm_tolerance": _WEB_WASM_PREDICTION_TOLERANCE,
        "prediction_abs_max": max((row["prediction_abs_delta"] for row in rows), default=0.0),
        "target_abs_max": max((row["target_abs_delta"] for row in rows), default=0.0),
    }


def _web_dataset_fixture(dataset_key: str) -> dict[str, Any]:
    dataset = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    x = np.asarray(dataset.x({}, layout="2d"), dtype=float)
    y = _as_vector(dataset.y({}))
    sample_ids = [str(int(sample)) for sample in dataset.index_column("sample", {})]
    train_samples = {str(int(sample)) for sample in dataset.index_column("sample", {"partition": "train"})}
    test_samples = {str(int(sample)) for sample in dataset.index_column("sample", {"partition": "test"})}
    partitions = ["test" if sample in test_samples else "train" if sample in train_samples else "train" for sample in sample_ids]
    assert x.shape[0] == y.shape[0] == len(sample_ids) == len(partitions)
    assert "test" in partitions
    return {
        "schema_version": "n4a.e2e.web_materialized_dataset.v1",
        "status": "passed",
        "dataset_key": dataset_key,
        "targetName": "y",
        "taskType": "regression",
        "nSamples": int(x.shape[0]),
        "nFeatures": int(x.shape[1]),
        "axis": list(range(int(x.shape[1]))),
        "axisUnit": "index",
        "sampleIds": sample_ids,
        "partitions": partitions,
        "X": x.reshape(-1).tolist(),
        "y": y.tolist(),
    }


def _run_timed(engine: str, artifacts_dir: Path, pipeline: list[Any], dataset_key: str) -> tuple[Any, float, list[str]]:
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
        result = nirs4all.run(pipeline, dataset_path(dataset_key), **kwargs)
        elapsed = time.perf_counter() - start
    warning_messages = [str(warning.message) for warning in caught]
    if engine == "dag-ml":
        fallback_warnings = [message for message in warning_messages if "falling back to the legacy engine" in message]
        assert not fallback_warnings, fallback_warnings
        assert result._is_dagml_engine(), "engine='dag-ml' must not report a legacy fallback result"  # noqa: SLF001
        _assert_native_results_dir(getattr(result, "_dagml_results_dir", None))
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
    candidate = _candidate_descriptor(case)
    candidate_path = artifacts_dir / "pipeline-candidate.n4a.json"
    _write_json(candidate_path, candidate)
    reopened_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_sha256 = _stable_hash(candidate)
    reopened_candidate_sha256 = _stable_hash(reopened_candidate)

    legacy, legacy_seconds, legacy_warnings = _run_timed(
        "legacy",
        artifacts_dir,
        _pipeline_from_descriptor(reopened_candidate),
        reopened_candidate["dataset_key"],
    )
    dagml, dagml_seconds, dagml_warnings = _run_timed(
        "dag-ml",
        artifacts_dir,
        _pipeline_from_descriptor(reopened_candidate),
        reopened_candidate["dataset_key"],
    )

    legacy_pred = _as_vector(legacy.best["y_pred"])
    dagml_pred = _as_vector(dagml.best["y_pred"])
    legacy_true = _as_vector(legacy.best["y_true"])
    dagml_true = _as_vector(dagml.best["y_true"])

    assert legacy_pred.shape == dagml_pred.shape
    assert legacy_true.shape == dagml_true.shape
    assert legacy_pred.size >= case.expected_min_predictions
    assert dagml_pred.size >= case.expected_min_predictions
    assert legacy.num_predictions >= case.expected_min_predictions
    assert dagml.num_predictions >= case.expected_min_predictions

    parity = {
        "best_rmse_abs": abs(float(legacy.best_rmse) - float(dagml.best_rmse)),
        "best_score_abs": abs(float(legacy.best_score) - float(dagml.best_score)),
        "cv_best_score_abs": abs(float(legacy.cv_best_score) - float(dagml.cv_best_score)),
        "prediction_abs_max": float(np.max(np.abs(legacy_pred - dagml_pred))) if legacy_pred.size else 0.0,
        "target_abs_max": float(np.max(np.abs(legacy_true - dagml_true))) if legacy_true.size else 0.0,
        "prediction_rows": int(legacy_pred.shape[0]),
        "tolerance": 1e-8,
    }
    prediction_oracle = _prediction_oracle(legacy, dagml, legacy_pred, dagml_pred, legacy_true, dagml_true, parity["tolerance"])
    assert prediction_oracle["prediction_abs_max"] <= prediction_oracle["tolerance"]
    assert prediction_oracle["target_abs_max"] <= prediction_oracle["tolerance"]
    web_dataset = _web_dataset_fixture(reopened_candidate["dataset_key"])
    web_dataset_path = artifacts_dir / "dataset-web-oracle.json"
    _write_json(web_dataset_path, web_dataset)
    web_dataset_sha256 = _stable_hash(web_dataset)
    assert parity["best_rmse_abs"] <= 1e-9
    assert parity["best_score_abs"] <= 1e-9
    assert parity["prediction_abs_max"] <= parity["tolerance"]
    assert parity["target_abs_max"] <= parity["tolerance"]

    speedup = legacy_seconds / dagml_seconds if dagml_seconds > 0 else None
    legacy_summary = _result_summary(legacy, legacy_seconds, legacy_warnings)
    dagml_summary = _result_summary(dagml, dagml_seconds, dagml_warnings)
    python_open_pipeline = {
        "schema_version": "n4a.e2e.python_open_pipeline.v1",
        "scenario_id": "e2e-pipeline-generation-performance-compare",
        "status": "passed",
        "pipeline_reopened": True,
        "candidate_path": candidate_path.name,
        "candidate_sha256": candidate_sha256,
        "reopened_candidate_sha256": reopened_candidate_sha256,
        "candidate_hash_match": reopened_candidate_sha256 == candidate_sha256,
        "case_name": reopened_candidate["case_name"],
        "dataset_key": reopened_candidate["dataset_key"],
        "class_sequence": _class_sequence(reopened_candidate),
        "class_sequence_sha256": _stable_hash(_class_sequence(reopened_candidate)),
        "variant_count": len(reopened_candidate["variants"]),
        "variant_count_match": len(reopened_candidate["variants"]) == len(candidate["variants"]) and len(reopened_candidate["variants"]) > 0,
        "selected_model_match": _selected_model_identity(legacy_summary["selected"]) == _selected_model_identity(dagml_summary["selected"]),
        "selected_metric_match": legacy_summary["selected"]["metric"] == dagml_summary["selected"]["metric"],
        "prediction_rows_match": legacy_summary["selected"]["prediction_rows"] == dagml_summary["selected"]["prediction_rows"],
        "prediction_rows": legacy_summary["selected"]["prediction_rows"],
    }
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
            "variants": candidate["variants"],
        },
        "runs": {
            "legacy": legacy_summary,
            "dag_ml": dagml_summary,
        },
        "parity": parity,
        "prediction_oracle": prediction_oracle,
        "web_dataset": {
            "path": web_dataset_path.name,
            "sha256": web_dataset_sha256,
            "n_samples": web_dataset["nSamples"],
            "n_features": web_dataset["nFeatures"],
        },
        "performance": {
            "legacy_seconds": legacy_seconds,
            "dag_ml_seconds": dagml_seconds,
            "legacy_over_dag_ml_ratio": speedup,
            "verdict": "dag_ml_faster" if speedup is not None and speedup > 1.0 else "no_speedup_on_this_run",
        },
        "python_open_pipeline": python_open_pipeline,
        "native_results_dir": str(getattr(dagml, "_dagml_results_dir")),
    }

    _write_json(artifacts_dir / "pipeline-family.json", family)
    assert candidate_path.exists()
    assert python_open_pipeline["candidate_hash_match"] is True
    assert python_open_pipeline["variant_count_match"] is True
    assert python_open_pipeline["selected_model_match"] is True
    assert python_open_pipeline["selected_metric_match"] is True
    assert python_open_pipeline["prediction_rows_match"] is True

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.transforms import FirstDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from tests.integration.parity._datasets import dataset_path

SCENARIO_ID = "e2e-multisource-branching-stacking-replay"
PIPELINE_NAME = "multisource_duplication_stacking_native_replay"
SCORE_TOLERANCE = 1e-3
PREDICTION_TOLERANCE = 1e-8


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"expected finite score, got {value!r}")
    return number


def _pipeline() -> list[Any]:
    return [
        KFold(n_splits=3, shuffle=True, random_state=42),
        {
            "branch": [
                [SNV(), {"model": PLSRegression(n_components=8)}],
                [MSC(), {"model": PLSRegression(n_components=8)}],
                [FirstDerivative(), {"model": PLSRegression(n_components=8)}],
            ]
        },
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


def _pipeline_contract() -> dict[str, Any]:
    return {
        "schema_version": "n4a.e2e.pipeline.multisource_stacking.v1",
        "scenario_id": SCENARIO_ID,
        "name": PIPELINE_NAME,
        "dataset": {"key": "multi", "source_count": 3},
        "engine_contract": {
            "native_engine": "dag-ml",
            "legacy_engine_is_oracle_for_current_python_package": False,
            "legacy_note": (
                "The current legacy by_source stacking case remains fallback-only because its scores diverge "
                "from the leakage-safe direct stacking oracle. This scenario uses a duplication-branch "
                "stacking shape that dag-ml already supports natively and validates it against a direct "
                "Python/sklearn oracle."
            ),
        },
        "pipeline": [
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3, "shuffle": True, "random_state": 42}},
            {
                "branch": [
                    [{"class": "nirs4all.operators.transforms.StandardNormalVariate"}, {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 8}}}],
                    [{"class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"}, {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 8}}}],
                    [{"class": "nirs4all.operators.transforms.FirstDerivative"}, {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 8}}}],
                ]
            },
            {"merge": "predictions"},
            {"model": {"class": "sklearn.linear_model.Ridge", "params": {"alpha": 1.0, "random_state": 42}}},
        ],
    }


def _load_multi_dataset() -> Any:
    return DatasetConfigs(dataset_path("multi")).get_dataset_at(0)


def _branch_specs(pipeline_contract: dict[str, Any]) -> list[dict[str, Any]]:
    transform_factories: dict[str, Callable[[], Any]] = {
        "nirs4all.operators.transforms.StandardNormalVariate": SNV,
        "nirs4all.operators.transforms.MultiplicativeScatterCorrection": MSC,
        "nirs4all.operators.transforms.FirstDerivative": FirstDerivative,
    }
    short_names = {
        "nirs4all.operators.transforms.StandardNormalVariate": "SNV",
        "nirs4all.operators.transforms.MultiplicativeScatterCorrection": "MSC",
        "nirs4all.operators.transforms.FirstDerivative": "FirstDerivative",
    }
    specs: list[dict[str, Any]] = []
    for branch in pipeline_contract["pipeline"][1]["branch"]:
        transform_class = branch[0]["class"]
        n_components = int(branch[1]["model"]["params"]["n_components"])
        specs.append(
            {
                "name": f"{short_names[transform_class]}_PLS{n_components}",
                "transform_class": transform_class,
                "factory": transform_factories[transform_class],
                "n_components": n_components,
                "model_class": branch[1]["model"]["class"],
            }
        )
    return specs


def _direct_stacking_oracle(pipeline_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline_contract = pipeline_contract or _pipeline_contract()
    dataset = _load_multi_dataset()
    train = dataset.index_column("sample", {"partition": "train"})
    test = dataset.index_column("sample", {"partition": "test"})
    split_params = pipeline_contract["pipeline"][0]["params"]
    final_model_params = pipeline_contract["pipeline"][-1]["model"]["params"]
    branch_specs = _branch_specs(pipeline_contract)
    branch_descriptors = [{key: value for key, value in spec.items() if key != "factory"} for spec in branch_specs]
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(**split_params).split(train)]

    def x(ids: list[int]) -> np.ndarray:
        return np.asarray(dataset.x({"sample": [int(sample) for sample in ids]}, layout="2d", concat_source=True))

    def y(ids: list[int]) -> np.ndarray:
        stored = dataset.index_column("sample", {"sample": [int(sample) for sample in ids]})
        row_of = {int(sample): row for row, sample in enumerate(stored)}
        values = np.asarray(dataset.y({"sample": [int(sample) for sample in ids]}), dtype=float).reshape(len(stored), -1)
        return values[[row_of[int(sample)] for sample in ids], 0]

    def fit_branch(spec: dict[str, Any], sample_ids: list[int]) -> tuple[Any, PLSRegression]:
        transform = spec["factory"]()
        transformed = transform.fit_transform(x(sample_ids))
        model = PLSRegression(n_components=spec["n_components"]).fit(transformed, y(sample_ids))
        return transform, model

    def predict_branch(fitted: tuple[Any, PLSRegression], sample_ids: list[int]) -> np.ndarray:
        transform, model = fitted
        return np.asarray(model.predict(transform.transform(x(sample_ids)))).ravel()

    truth: dict[int, float] = {}
    meta_oof: dict[int, float] = {}
    base_oof: dict[str, dict[int, float]] = {spec["name"]: {} for spec in branch_specs}
    fold_summaries: list[dict[str, Any]] = []

    for fold_index, (train_ids, val_ids) in enumerate(folds):
        val_ids = [int(sample) for sample in val_ids]
        meta_columns: list[np.ndarray] = []
        base_scores: dict[str, float] = {}
        for spec in branch_specs:
            branch_name = spec["name"]
            fitted = fit_branch(spec, train_ids)
            predictions = predict_branch(fitted, val_ids)
            meta_columns.append(predictions)
            for position, sample_id in enumerate(val_ids):
                base_oof[branch_name][sample_id] = float(predictions[position])
            base_scores[branch_name] = float(np.sqrt(mean_squared_error(y(val_ids), predictions)))

        x_meta = np.column_stack(meta_columns)
        y_meta = y(val_ids)
        meta_predictions = Ridge(**final_model_params).fit(x_meta, y_meta).predict(x_meta)
        for position, sample_id in enumerate(val_ids):
            truth[sample_id] = float(y_meta[position])
            meta_oof[sample_id] = float(meta_predictions[position])
        fold_summaries.append(
            {
                "fold_id": fold_index,
                "sample_ids": val_ids,
                "meta_rmse": float(np.sqrt(mean_squared_error(y_meta, meta_predictions))),
                "base_rmse": base_scores,
            }
        )

    keys = sorted(truth)
    cv_rmse = float(np.sqrt(mean_squared_error([truth[key] for key in keys], [meta_oof[key] for key in keys])))
    meta_model = Ridge(**final_model_params).fit(
        np.array([[base_oof[spec["name"]][sample_id] for spec in branch_specs] for sample_id in keys]),
        np.array([truth[sample_id] for sample_id in keys]),
    )
    refit_branches = [fit_branch(spec, train) for spec in branch_specs]
    test_meta = np.column_stack([predict_branch(fitted, test) for fitted in refit_branches])
    test_predictions = meta_model.predict(test_meta)
    best_rmse = float(np.sqrt(mean_squared_error(y(test), test_predictions)))

    return {
        "schema_version": "n4a.e2e.oof_ledger.v1",
        "scenario_id": SCENARIO_ID,
        "oracle": "direct Python sklearn stacking over nirs4all multi-source fused feature view",
        "pipeline_sha256": _stable_hash(pipeline_contract["pipeline"]),
        "branch_descriptors": branch_descriptors,
        "branch_identity_sha256": _stable_hash(branch_descriptors),
        "folds": fold_summaries,
        "scores": {"cv_best_score": cv_rmse, "best_rmse": best_rmse},
        "test": {
            "sample_ids": [int(sample) for sample in test],
            "predictions": [float(value) for value in test_predictions],
            "targets": [float(value) for value in y(test)],
        },
    }


def _result_summary(result: Any, warning_messages: list[str]) -> dict[str, Any]:
    rows = result.predictions.filter_predictions(load_arrays=True)
    return {
        "is_dagml": bool(getattr(result, "_is_dagml_engine", lambda: False)()),
        "best_rmse": _finite(result.best_rmse),
        "best_score": _finite(result.best_score),
        "cv_best_score": _finite(result.cv_best_score),
        "num_predictions": int(result.num_predictions),
        "native_results_dir": str(getattr(result, "_dagml_results_dir", "")),
        "warnings": warning_messages,
        "rows": [
            {
                "fold_id": str(row.get("fold_id")),
                "partition": str(row.get("partition")),
                "model_name": str(row.get("model_name")),
                "val_score": row.get("val_score"),
                "test_score": row.get("test_score"),
                "train_score": row.get("train_score"),
            }
            for row in rows
            if str(row.get("model_name")).startswith("MetaModel") or str(row.get("model_name")).startswith("by_source_MetaModel")
        ],
    }


def test_multisource_stacking_replay(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipeline_contract = _pipeline_contract()
    oracle = _direct_stacking_oracle(pipeline_contract)

    native_root = artifacts_dir / "native-results"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            _pipeline(),
            dataset_path("multi"),
            engine="dag-ml",
            verbose=0,
            save_artifacts=False,
            save_charts=False,
            random_state=42,
            results_path=native_root,
        )
    warning_messages = [str(warning.message) for warning in caught]
    assert not any("falling back to the legacy engine" in message for message in warning_messages)

    summary = _result_summary(result, warning_messages)
    assert summary["is_dagml"] is True
    cv_best_score_delta = abs(summary["cv_best_score"] - oracle["scores"]["cv_best_score"])
    best_rmse_delta = abs(summary["best_rmse"] - oracle["scores"]["best_rmse"])
    assert cv_best_score_delta <= SCORE_TOLERANCE
    assert best_rmse_delta <= SCORE_TOLERANCE

    native_dir = Path(summary["native_results_dir"])
    assert native_dir.is_dir(), summary
    for filename in ("manifest.json", "score_set.json", "predictions.parquet"):
        assert (native_dir / filename).exists(), f"missing native result artifact {filename}"
    native_predictions = pl.read_parquet(native_dir / "predictions.parquet")
    native_meta_test = native_predictions.filter(
        (pl.col("model_name") == "MetaModel_Ridge")
        & (pl.col("partition") == "test")
        & (pl.col("refit_context") == "standalone")
    )
    assert native_meta_test.height == 1
    native_meta_row = native_meta_test.row(0, named=True)
    assert native_meta_row["arrays_present"] is True
    native_sample_ids = [int(sample) for sample in native_meta_row["sample_indices"]]
    native_by_sample = dict(zip(native_sample_ids, np.asarray(native_meta_row["y_pred"], dtype=float), strict=True))
    native_targets_by_sample = dict(zip(native_sample_ids, np.asarray(native_meta_row["y_true"], dtype=float), strict=True))
    oracle_by_sample = dict(zip(oracle["test"]["sample_ids"], oracle["test"]["predictions"], strict=True))
    oracle_targets_by_sample = dict(zip(oracle["test"]["sample_ids"], oracle["test"]["targets"], strict=True))
    assert set(native_by_sample) == set(oracle_by_sample)
    native_prediction_delta = max(
        abs(native_by_sample[sample] - oracle_by_sample[sample])
        for sample in sorted(oracle_by_sample)
    )
    native_target_delta = max(
        abs(native_targets_by_sample[sample] - oracle_targets_by_sample[sample])
        for sample in sorted(oracle_targets_by_sample)
    )
    assert native_prediction_delta <= PREDICTION_TOLERANCE
    assert native_target_delta <= PREDICTION_TOLERANCE

    pipeline_contract["sha256"] = _stable_hash(pipeline_contract["pipeline"])
    replay_manifest = {
        "schema_version": "n4a.e2e.multisource_stacking_replay.v1",
        "scenario_id": SCENARIO_ID,
        "status": "python_oracle_and_native_ready",
        "pipeline": pipeline_contract,
        "oracle_scores": oracle["scores"],
        "dagml_native": summary,
        "parity": {
            "status": "passed",
            "score_tolerance": SCORE_TOLERANCE,
            "cv_best_score_delta": cv_best_score_delta,
            "best_rmse_delta": best_rmse_delta,
            "within_tolerance": True,
        },
        "known_boundaries": [
            "The richer by_source stacking legacy case remains fallback-only because native leakage-safe stacking diverges from the current legacy summary.",
            "This scenario validates multisource input plus native duplication-branch stacking without changing the existing by_source fallback boundary.",
        ],
    }

    oof_ledger = {
        **oracle,
        "status": "passed",
        "parity_ok": True,
        "within_tolerance": True,
        "score_tolerance": SCORE_TOLERANCE,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "cv_best_score_delta": cv_best_score_delta,
        "best_rmse_delta": best_rmse_delta,
    }

    replay_path = artifacts_dir / "stacking-replay.n4a.json"
    python_open_ledger_path = artifacts_dir / "python-open-ledger.json"
    python_rerun_ledger_path = artifacts_dir / "python-rerun-ledger.json"
    _write_json(replay_path, replay_manifest)
    reopened_replay = json.loads(replay_path.read_text(encoding="utf-8"))
    reopened_pipeline = reopened_replay["pipeline"]
    branch_descriptors = [
        {
            "branch_index": index,
            "transform_class": branch[0]["class"],
            "model_class": branch[1]["model"]["class"],
            "n_components": branch[1]["model"]["params"]["n_components"],
        }
        for index, branch in enumerate(reopened_pipeline["pipeline"][1]["branch"])
    ]
    python_open_pipeline = {
        "schema_version": "n4a.e2e.python_open_pipeline.v1",
        "scenario_id": SCENARIO_ID,
        "status": "passed",
        "pipeline_reopened": True,
        "replay_manifest": replay_path.name,
        "reopened_scenario_id": reopened_replay.get("scenario_id"),
        "scenario_id_match": reopened_replay.get("scenario_id") == SCENARIO_ID,
        "replay_manifest_sha256": _file_hash(replay_path),
        "pipeline_sha256": pipeline_contract["sha256"],
        "reopened_pipeline_sha256": _stable_hash(reopened_pipeline["pipeline"]),
        "pipeline_hash_match": _stable_hash(reopened_pipeline["pipeline"]) == pipeline_contract["sha256"],
        "name": reopened_pipeline["name"],
        "name_match": reopened_pipeline["name"] == PIPELINE_NAME,
        "dataset_key": reopened_pipeline["dataset"]["key"],
        "source_count": reopened_pipeline["dataset"]["source_count"],
        "source_count_match": reopened_pipeline["dataset"]["source_count"] == 3,
        "branch_count": len(branch_descriptors),
        "branch_count_match": len(branch_descriptors) == 3,
        "branch_descriptors": branch_descriptors,
        "branch_identity_sha256": _stable_hash(branch_descriptors),
    }
    _write_json(python_open_ledger_path, python_open_pipeline)

    rerun_oracle = _direct_stacking_oracle(reopened_pipeline)
    original_predictions = np.asarray(oracle["test"]["predictions"], dtype=float)
    rerun_predictions = np.asarray(rerun_oracle["test"]["predictions"], dtype=float)
    original_targets = np.asarray(oracle["test"]["targets"], dtype=float)
    rerun_targets = np.asarray(rerun_oracle["test"]["targets"], dtype=float)
    prediction_shape_match = original_predictions.shape == rerun_predictions.shape
    target_shape_match = original_targets.shape == rerun_targets.shape
    prediction_max_abs_delta = (
        float(np.max(np.abs(original_predictions - rerun_predictions))) if prediction_shape_match and original_predictions.size else None
    )
    target_max_abs_delta = float(np.max(np.abs(original_targets - rerun_targets))) if target_shape_match and original_targets.size else None
    cv_best_score_delta_rerun = abs(float(oracle["scores"]["cv_best_score"]) - float(rerun_oracle["scores"]["cv_best_score"]))
    best_rmse_delta_rerun = abs(float(oracle["scores"]["best_rmse"]) - float(rerun_oracle["scores"]["best_rmse"]))
    fold_hash = _stable_hash(oracle["folds"])
    rerun_fold_hash = _stable_hash(rerun_oracle["folds"])
    finite_predictions = bool(np.all(np.isfinite(rerun_predictions)) and np.all(np.isfinite(rerun_targets)))
    python_rerun_pipeline = {
        "schema_version": "n4a.e2e.python_rerun_pipeline.v1",
        "scenario_id": SCENARIO_ID,
        "status": "passed",
        "pipeline_reopened": True,
        "replay_manifest_reopened": True,
        "python_rerun_executed": True,
        "finite_predictions": finite_predictions,
        "prediction_rows": int(rerun_predictions.size),
        "replay_manifest_sha256": _file_hash(replay_path),
        "pipeline_sha256": pipeline_contract["sha256"],
        "reopened_pipeline_sha256": _stable_hash(reopened_pipeline["pipeline"]),
        "pipeline_hash_match": _stable_hash(reopened_pipeline["pipeline"]) == pipeline_contract["sha256"],
        "branch_identity_sha256": oracle["branch_identity_sha256"],
        "rerun_branch_identity_sha256": rerun_oracle["branch_identity_sha256"],
        "branch_hash_match": rerun_oracle["branch_identity_sha256"] == oracle["branch_identity_sha256"],
        "fold_sha256": fold_hash,
        "rerun_fold_sha256": rerun_fold_hash,
        "fold_hash_match": rerun_fold_hash == fold_hash,
        "prediction_shape_match": prediction_shape_match,
        "prediction_max_abs_delta": prediction_max_abs_delta,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "target_shape_match": target_shape_match,
        "target_max_abs_delta": target_max_abs_delta,
        "target_tolerance": PREDICTION_TOLERANCE,
        "cv_best_score_delta": cv_best_score_delta_rerun,
        "best_rmse_delta": best_rmse_delta_rerun,
        "score_tolerance": PREDICTION_TOLERANCE,
    }
    python_rerun_pipeline["status"] = (
        "passed"
        if (
            finite_predictions
            and python_rerun_pipeline["pipeline_hash_match"]
            and python_rerun_pipeline["branch_hash_match"]
            and python_rerun_pipeline["fold_hash_match"]
            and prediction_max_abs_delta is not None
            and prediction_max_abs_delta <= PREDICTION_TOLERANCE
            and target_max_abs_delta is not None
            and target_max_abs_delta <= PREDICTION_TOLERANCE
            and cv_best_score_delta_rerun <= PREDICTION_TOLERANCE
            and best_rmse_delta_rerun <= PREDICTION_TOLERANCE
        )
        else "failed"
    )
    _write_json(python_rerun_ledger_path, python_rerun_pipeline)
    _write_json(artifacts_dir / "oof-ledger.json", oof_ledger)

    assert replay_path.exists()
    assert python_open_ledger_path.exists()
    assert python_open_pipeline["scenario_id_match"] is True
    assert python_open_pipeline["pipeline_hash_match"] is True
    assert python_open_pipeline["name_match"] is True
    assert python_open_pipeline["source_count_match"] is True
    assert python_open_pipeline["branch_count_match"] is True
    assert python_rerun_ledger_path.exists()
    assert python_rerun_pipeline["status"] == "passed"
    assert python_rerun_pipeline["prediction_rows"] > 0
    assert python_rerun_pipeline["prediction_max_abs_delta"] <= PREDICTION_TOLERANCE
    assert python_rerun_pipeline["best_rmse_delta"] <= PREDICTION_TOLERANCE
    assert (artifacts_dir / "oof-ledger.json").exists()
    assert json.loads((artifacts_dir / "oof-ledger.json").read_text(encoding="utf-8"))["status"] == "passed"

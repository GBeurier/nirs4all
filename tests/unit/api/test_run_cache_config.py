"""Regression tests for run() cache configuration defaults."""

from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from nirs4all.config.cache_config import CacheConfig
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml.tuning_contracts import DagMLTuningNotImplementedError


class _DummyRunner:
    """Minimal PipelineRunner test double for run() unit tests."""

    instances: list[_DummyRunner] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cache_config = None
        self.calls: list[dict[str, object]] = []
        self.workspace_path = kwargs.get("workspace_path", "/tmp/dummy")
        _DummyRunner.instances.append(self)

    def run(self, pipeline, dataset, pipeline_name, refit, **kwargs):
        self.calls.append(
            {
                "pipeline": pipeline,
                "dataset": dataset,
                "pipeline_name": pipeline_name,
                "refit": refit,
                "kwargs": kwargs,
            }
        )
        return Predictions(), {}

    def close(self):
        pass


class _RunTuningEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any, **_kwargs: Any) -> _RunTuningEstimator:
        self.fit_X_ = X
        self.fit_y_ = y
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), float(self.alpha))


class _RunTuningScaleTransformer(BaseEstimator):
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def fit(self, X: Any, y: Any | None = None) -> _RunTuningScaleTransformer:
        self.fit_y_ = y
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float) * float(self.factor)


class _RunTuningLinearEstimator(BaseEstimator):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _RunTuningLinearEstimator:
        self.fit_X_ = np.asarray(X, dtype=float)
        self.fit_y_ = np.asarray(y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0] + float(self.alpha)


class _StringifiesAs:
    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


class _RunTuningIdentityAwareEstimator(BaseEstimator):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any, **kwargs: Any) -> _RunTuningIdentityAwareEstimator:
        self.fit_kwargs_ = dict(kwargs)
        return self

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> np.ndarray:
        identities_match = _as_list(sample_ids) == ["score-a", "score-b"] and _as_list(groups) == ["batch-1", "batch-2"] and _as_list(metadata) == [{"site": "north"}, {"site": "south"}]
        if identities_match:
            return np.full(len(X), float(self.alpha))
        return np.full(len(X), 99.0)


class _RunTuningFitIdentityAwareEstimator(BaseEstimator):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> _RunTuningFitIdentityAwareEstimator:
        self.fit_identities_match_ = _as_list(sample_ids) == ["train-a", "train-b"] and _as_list(groups) == ["train-g1", "train-g2"] and _as_list(metadata) == [{"site": "train-1"}, {"site": "train-2"}]
        return self

    def predict(self, X: Any) -> np.ndarray:
        if getattr(self, "fit_identities_match_", False):
            return np.full(len(X), float(self.alpha))
        return np.full(len(X), 99.0)


class _RunTuningFitAndScoreIdentityAwareEstimator(BaseEstimator):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> _RunTuningFitAndScoreIdentityAwareEstimator:
        self.fit_identities_match_ = _as_list(sample_ids) == ["train-a", "train-b"] and _as_list(groups) == ["train-g1", "train-g2"] and _as_list(metadata) == [{"site": "train-1"}, {"site": "train-2"}]
        return self

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> np.ndarray:
        score_identities_match = _as_list(sample_ids) == ["score-a", "score-b"] and _as_list(groups) == ["batch-1", "batch-2"] and _as_list(metadata) == [{"site": "north"}, {"site": "south"}]
        if getattr(self, "fit_identities_match_", False) and score_identities_match:
            return np.full(len(X), float(self.alpha))
        return np.full(len(X), 99.0)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def _make_run_tuning_spectro_dataset() -> SpectroDataset:
    dataset = SpectroDataset("native-tuning")
    dataset.add_samples(np.asarray([[1.0], [2.0]], dtype=float), indexes={"partition": "train"})
    dataset.add_targets(np.asarray([1.0, 2.0], dtype=float))
    dataset.add_metadata(
        np.asarray(
            [
                ["train-a", "train-g1", "train-1"],
                ["train-b", "train-g2", "train-2"],
            ],
            dtype=object,
        ),
        headers=["sample_id", "group_id", "site"],
    )
    return dataset


def _make_run_tuning_spectro_dataset_with_metadata(
    *,
    name: str,
    sample_ids: list[str],
    groups: list[str],
    sites: list[str],
    x_values: list[float],
    y_values: list[float],
) -> SpectroDataset:
    dataset = SpectroDataset(name)
    dataset.add_samples(np.asarray([[value] for value in x_values], dtype=float), indexes={"partition": "train"})
    dataset.add_targets(np.asarray(y_values, dtype=float))
    dataset.add_metadata(
        np.asarray(
            [[sample_id, group, site] for sample_id, group, site in zip(sample_ids, groups, sites, strict=True)],
            dtype=object,
        ),
        headers=["sample_id", "group_id", "site"],
    )
    return dataset


def _write_run_tuning_dataset_config(
    tmp_path,
    *,
    name: str,
    sample_ids: list[str],
    groups: list[str],
    sites: list[str],
    x_values: list[float],
    y_values: list[float],
) -> str:
    folder = tmp_path / name
    folder.mkdir()
    x_path = folder / "X_train.csv"
    y_path = folder / "Y_train.csv"
    metadata_path = folder / "M_train.csv"
    config_path = folder / "dataset.json"

    x_rows = "\n".join(f"{value};{value + 0.1}" for value in x_values)
    y_rows = "\n".join(str(value) for value in y_values)
    metadata_rows = "\n".join(f"{sample_id};{group};{site}" for sample_id, group, site in zip(sample_ids, groups, sites, strict=True))
    x_path.write_text(f"1000;1100\n{x_rows}\n", encoding="utf-8")
    y_path.write_text(f"target\n{y_rows}\n", encoding="utf-8")
    metadata_path.write_text(f"sample_id;group_id;site\n{metadata_rows}\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "name": name,
                "task_type": "regression",
                "train_x": str(x_path),
                "train_y": str(y_path),
                "train_group": str(metadata_path),
                "global_params": {"delimiter": ";", "has_header": True},
            }
        ),
        encoding="utf-8",
    )
    return str(config_path)


def test_run_sets_default_cache_config_when_cache_none(monkeypatch):
    """run(cache=None) should still initialize a default CacheConfig."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    run_module.run(
        pipeline=[{"model": "DummyModel"}],
        dataset="dummy_dataset",
        cache=None,
        engine="legacy",  # this test exercises legacy PipelineRunner cache wiring (pinned explicitly: the dag-ml path builds no runner)
        verbose=0,
    )

    runner = _DummyRunner.instances[-1]
    assert isinstance(runner.cache_config, CacheConfig)
    assert runner.cache_config.step_cache_enabled is False
    assert runner.cache_config.use_cow_snapshots is True


def test_run_uses_explicit_cache_config(monkeypatch):
    """run(cache=...) should preserve the caller-provided CacheConfig."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)
    explicit_cache = CacheConfig(step_cache_enabled=True, use_cow_snapshots=False)

    run_module.run(
        pipeline=[{"model": "DummyModel"}],
        dataset="dummy_dataset",
        cache=explicit_cache,
        engine="legacy",  # legacy PipelineRunner cache wiring (pinned explicitly: the dag-ml path builds no runner)
        verbose=0,
    )

    runner = _DummyRunner.instances[-1]
    assert runner.cache_config is explicit_cache


def test_run_keeps_nested_list_pipeline_as_single_pipeline(monkeypatch):
    """Nested list steps should not be reinterpreted as a batch of pipelines."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    nested_pipeline = [
        [{"class": "sklearn.preprocessing.StandardScaler"}],
        {"model": {"class": "sklearn.linear_model.Ridge"}},
    ]

    run_module.run(
        pipeline=nested_pipeline,
        dataset="dummy_dataset",
        engine="legacy",  # legacy PipelineRunner pipeline-normalization (pinned explicitly: the dag-ml path builds no runner)
        verbose=0,
    )

    runner = _DummyRunner.instances[-1]
    assert len(runner.calls) == 1
    assert runner.calls[0]["pipeline"] == nested_pipeline


def test_run_tuning_keyword_is_reserved_and_fail_closed(monkeypatch):
    """run(tuning=...) should fail before legacy runner kwargs can absorb it."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    with pytest.raises(DagMLTuningNotImplementedError, match=r"run\(tuning=\.\.\.\)") as excinfo:
        run_module.run(
            pipeline=[{"model": "DummyModel"}],
            dataset="dummy_dataset",
            tuning={"engine": "n4m", "space": {"model.n_components": [2, 3]}},
            engine="legacy",
            verbose=0,
        )

    assert excinfo.value.tuning_spec.engine == "n4m"
    assert len(excinfo.value.tuning_spec.fingerprint) == 64
    assert "PipelineObjective" in excinfo.value.available_internal
    assert "prediction score extractor for PipelineObjective" in excinfo.value.available_internal
    assert "single-estimator and linear transformer→estimator PipelineObjective compiler" in excinfo.value.available_internal
    assert "WorkspaceStore TuningResult persistence" in excinfo.value.available_internal
    assert "RunResult tuning evidence projection" in excinfo.value.available_internal
    assert "explicit winner prediction projection into RunResult" in excinfo.value.available_internal
    assert "linear compile→tune→refit→RunResult orchestration" in excinfo.value.available_internal
    assert "public run(tuning=...) single-estimator/linear array subset" in excinfo.value.available_internal
    assert "public run(tuning=...) resume integration for broader pipeline shapes" in excinfo.value.missing_gates
    assert _DummyRunner.instances == []


def test_run_tuning_single_estimator_array_subset_executes_without_legacy_runner(monkeypatch):
    """The first public run(tuning=...) subset routes through the native single-estimator lane."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "X": np.asarray([[30.0], [40.0]]),
                "y_true": np.asarray([0.0, 0.0]),
                "score": 0.2,
                "metric": "rmse",
                "sample_ids": ["test-a", "test-b"],
                "model_name": "RunTuningWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.2)
    assert result.best["model_name"] == "RunTuningWinner"
    assert result.best_score == pytest.approx(0.2)


def test_run_tuning_force_params_enqueue_first_public_trial(monkeypatch):
    """force_params should traverse the public run(tuning=...) subset."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.2},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_result.trials[0].params == {"alpha": 0.2}
    assert result.tuning_result.trials[0].value == pytest.approx(0.2)


def test_run_tuning_score_data_can_use_temporary_conformal_dev_metric(monkeypatch):
    """R10: tuning can score candidates with a temporary dev conformal calibrator."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "conformal_mean_width",
            "direction": "minimize",
            "score_data": {
                "X": np.asarray([[10.0], [20.0]]),
                "y": np.asarray([0.0, 0.0]),
                "conformal_coverage": 0.8,
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0], [5.0], [6.0]]),
                    "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
                },
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.4)
    assert result.tuning_result.trials[0].diagnostics["metric"] == "conformal_mean_width"
    assert result.tuning_result.trials[0].diagnostics["score_family"] == "conformal"
    assert result.tuning_result.trials[0].diagnostics["score_extractor"] == "conformal_temporary_calibration"
    assert result.tuning_result.trials[0].diagnostics["final_calibration_scope"] == "unmodified_by_score_data"


def test_run_tuning_linear_transformer_chain_executes_without_legacy_runner(monkeypatch):
    """The public subset accepts a linear transformer→estimator chain."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[
            {"name": "scale", "transform": _RunTuningScaleTransformer(factor=10.0)},
            {"model": _RunTuningLinearEstimator(alpha=0.0)},
        ],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"model.alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([10.2, 20.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([30.2, 40.2]),
                "score": 0.0,
                "metric": "rmse",
                "sample_ids": ["linear-a", "linear-b"],
                "model_name": "RunTuningLinearWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"model.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [30.2, 40.2])
    assert result.best["model_name"] == "RunTuningLinearWinner"


def test_run_tuning_public_steps_mapping_executes_without_legacy_runner(monkeypatch):
    """The public subset accepts the documented {'steps': [...]} pipeline mapping."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={
            "name": "mapped-linear",
            "steps": [
                {"name": "scale", "transform": _RunTuningScaleTransformer(factor=10.0)},
                {"name": "ridge", "model": _RunTuningLinearEstimator(alpha=0.0)},
            ],
        },
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([10.2, 20.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([30.2, 40.2]),
                "score": 0.0,
                "metric": "rmse",
                "sample_ids": ["mapped-linear-a", "mapped-linear-b"],
                "model_name": "RunTuningMappedLinearWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [30.2, 40.2])
    assert result.best["model_name"] == "RunTuningMappedLinearWinner"


def test_run_tuning_public_pipeline_mapping_executes_without_legacy_runner(monkeypatch):
    """The public subset also accepts the legacy {'pipeline': [...]} wrapper."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={
            "name": "pipeline-wrapper-linear",
            "pipeline": [
                {"name": "scale", "transform": _RunTuningScaleTransformer(factor=10.0)},
                {"name": "ridge", "model": _RunTuningLinearEstimator(alpha=0.0)},
            ],
        },
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([10.2, 20.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([30.2, 40.2]),
                "score": 0.0,
                "metric": "rmse",
                "sample_ids": ["pipeline-wrapper-a", "pipeline-wrapper-b"],
                "model_name": "RunTuningPipelineWrapperWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [30.2, 40.2])
    assert result.best["model_name"] == "RunTuningPipelineWrapperWinner"


def test_run_tuning_pipeline_mapping_rejects_ambiguous_steps_and_pipeline(monkeypatch):
    """Pipeline wrapper mappings must not contain both 'steps' and 'pipeline'."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    with pytest.raises(ValueError, match="either 'steps' or 'pipeline', not both"):
        run_module.run(
            pipeline={
                "steps": [{"name": "ridge", "model": _RunTuningLinearEstimator(alpha=0.0)}],
                "pipeline": [{"name": "ridge", "model": _RunTuningLinearEstimator(alpha=0.0)}],
            },
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"ridge.alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([10.2, 20.2])},
            },
            verbose=0,
        )
    assert _DummyRunner.instances == []


def test_run_tuning_sklearn_pipeline_accepts_dotted_space_paths(monkeypatch):
    """sklearn Pipeline objects can use nirs4all dotted tuning paths."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=Pipeline(
            [
                ("scale", _RunTuningScaleTransformer(factor=10.0)),
                ("ridge", _RunTuningLinearEstimator(alpha=0.0)),
            ]
        ),
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([10.2, 20.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([30.2, 40.2]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningSklearnPipelineWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [30.2, 40.2])
    assert result.best["model_name"] == "RunTuningSklearnPipelineWinner"


def test_run_tuning_sklearn_like_step_tuples_execute_without_legacy_runner(monkeypatch):
    """sklearn-like (name, step) tuples use the same dotted tuning paths."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[
            ("scale", _RunTuningScaleTransformer(factor=10.0)),
            ("ridge", _RunTuningLinearEstimator(alpha=0.0)),
        ],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([10.2, 20.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([30.2, 40.2]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningTuplePipelineWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [30.2, 40.2])
    assert result.best["model_name"] == "RunTuningTuplePipelineWinner"


def test_run_tuning_sklearn_class_mapping_steps_execute_without_legacy_runner(monkeypatch):
    """sklearn class-path mappings are accepted for the linear native subset."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[
            {"name": "scale", "class": "sklearn.preprocessing.StandardScaler", "params": {"with_mean": False}},
            {"name": "ridge", "model": _RunTuningLinearEstimator(alpha=0.0)},
        ],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"scale.with_mean": [False], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([2.2, 4.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([6.2, 8.2]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningClassMappingWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"scale.with_mean": False, "ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [6.2, 8.2])
    assert result.best["model_name"] == "RunTuningClassMappingWinner"


def test_run_tuning_explicit_sklearn_string_steps_execute_without_legacy_runner(monkeypatch):
    """Explicit sklearn.* string steps are accepted without enabling short aliases."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[("dummy", "sklearn.dummy.DummyRegressor")],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"dummy.strategy": ["mean"]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([15.0, 15.0])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([15.0, 15.0]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningSklearnStringStepWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"dummy.strategy": "mean"}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [15.0, 15.0])
    assert result.best["model_name"] == "RunTuningSklearnStringStepWinner"


def test_run_tuning_direct_sklearn_string_model_executes_without_legacy_runner(monkeypatch):
    """A direct sklearn.* string model is accepted in the native tuning subset."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline="sklearn.dummy.DummyRegressor",
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"strategy": ["mean"]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([15.0, 15.0])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([15.0, 15.0]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningDirectSklearnStringModelWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"strategy": "mean"}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [15.0, 15.0])
    assert result.best["model_name"] == "RunTuningDirectSklearnStringModelWinner"


def test_run_tuning_declarative_sklearn_class_pipeline_executes_without_legacy_runner(monkeypatch):
    """A fully declarative sklearn class-path linear pipeline stays native."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={
            "steps": [
                {"name": "scale", "class": "sklearn.preprocessing.StandardScaler", "params": {"with_mean": False}},
                {"name": "ridge", "class": "sklearn.linear_model.Ridge", "params": {"fit_intercept": False}},
            ]
        },
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([2.0, 4.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"scale.with_mean": [False], "ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([2.0, 4.0])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([6.0, 8.0]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningDeclarativeClassPipelineWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"scale.with_mean": False, "ridge.alpha": 0.0}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [6.0, 8.0], atol=1e-12)
    assert result.best["model_name"] == "RunTuningDeclarativeClassPipelineWinner"


def test_run_tuning_direct_declarative_sklearn_class_model_executes_without_legacy_runner(monkeypatch):
    """A single declarative sklearn class-path model can be tuned natively."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={"name": "ridge", "class": "sklearn.linear_model.Ridge", "params": {"fit_intercept": False}},
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([2.0, 4.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([2.0, 4.0])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([6.0, 8.0]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningDirectDeclarativeClassModelWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.0}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [6.0, 8.0], atol=1e-12)
    assert result.best["model_name"] == "RunTuningDirectDeclarativeClassModelWinner"


def test_run_tuning_sklearn_string_model_mapping_with_params_executes_without_legacy_runner(monkeypatch):
    """A final {'model': 'sklearn.*', 'params': ...} mapping stays native."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={
            "steps": [
                {
                    "name": "ridge",
                    "model": "sklearn.linear_model.Ridge",
                    "params": {"fit_intercept": False},
                }
            ]
        },
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([2.0, 4.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([2.0, 4.0])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([6.0, 8.0]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningSklearnStringModelParamsWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.0}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [6.0, 8.0], atol=1e-12)
    assert result.best["model_name"] == "RunTuningSklearnStringModelParamsWinner"


def test_run_tuning_direct_sklearn_string_model_mapping_with_params_executes_without_legacy_runner(monkeypatch):
    """A direct {'model': 'sklearn.*', 'params': ...} mapping stays native."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={
            "name": "ridge",
            "model": "sklearn.linear_model.Ridge",
            "params": {"fit_intercept": False},
        },
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([2.0, 4.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([2.0, 4.0])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([6.0, 8.0]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningDirectSklearnStringModelParamsWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.0}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [6.0, 8.0], atol=1e-12)
    assert result.best["model_name"] == "RunTuningDirectSklearnStringModelParamsWinner"


def test_run_tuning_sklearn_string_transform_mapping_with_params_executes_without_legacy_runner(monkeypatch):
    """A preprocessing {'transform': 'sklearn.*', 'params': ...} mapping stays native."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline={
            "steps": [
                {
                    "name": "scale",
                    "transform": "sklearn.preprocessing.StandardScaler",
                    "params": {"with_mean": False},
                },
                {"name": "ridge", "model": _RunTuningLinearEstimator(alpha=0.0)},
            ]
        },
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"scale.with_mean": [False], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([2.2, 4.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([6.2, 8.2]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningSklearnStringTransformParamsWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"scale.with_mean": False, "ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [6.2, 8.2])
    assert result.best["model_name"] == "RunTuningSklearnStringTransformParamsWinner"


def test_run_tuning_linear_passthrough_steps_execute_without_legacy_runner(monkeypatch):
    """Explicit passthrough preprocessing steps stay in the native linear subset."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[
            ("identity", "passthrough"),
            {"name": "also_identity", "transform": None},
            ("ridge", _RunTuningLinearEstimator(alpha=0.0)),
        ],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"ridge.alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.2, 2.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([3.2, 4.2]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningPassthroughWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [3.2, 4.2])
    assert result.best["model_name"] == "RunTuningPassthroughWinner"


def test_run_tuning_rejects_stringified_structured_passthrough_step_without_legacy_runner(monkeypatch):
    """Structured passthrough requires a literal JSON string value, not __str__."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    with pytest.raises(DagMLTuningNotImplementedError, match=r"run\(tuning=\.\.\.\) is reserved") as excinfo:
        run_module.run(
            pipeline=[
                {"name": "identity", "transform": {"kind": _StringifiesAs("passthrough")}},
                ("ridge", _RunTuningLinearEstimator(alpha=0.0)),
            ],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"ridge.alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.2, 2.2])},
            },
            verbose=0,
        )

    assert isinstance(excinfo.value.__cause__, TypeError)
    assert "preprocessing steps must expose transform()" in str(excinfo.value.__cause__)
    assert _DummyRunner.instances == []


def test_run_tuning_can_search_preprocessing_passthrough_without_legacy_runner(monkeypatch):
    """A named preprocessing step can be toggled to passthrough by the tuning space."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[
            ("scale", _RunTuningScaleTransformer(factor=10.0)),
            ("ridge", _RunTuningLinearEstimator(alpha=0.0)),
        ],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"scale": ["passthrough"], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.2, 2.2])},
            "winner": {
                "X": np.asarray([[3.0], [4.0]]),
                "y_true": np.asarray([3.2, 4.2]),
                "score": 0.0,
                "metric": "rmse",
                "model_name": "RunTuningSearchedPassthroughWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"scale": "passthrough", "ridge.alpha": 0.2}
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [3.2, 4.2])
    assert result.best["model_name"] == "RunTuningSearchedPassthroughWinner"


def test_run_tuning_can_search_structured_preprocessing_passthrough_without_legacy_runner(tmp_path, monkeypatch):
    """Structured JSON-native passthrough specs should survive Optuna and refit."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[
            ("scale", _RunTuningScaleTransformer(factor=10.0)),
            ("ridge", _RunTuningLinearEstimator(alpha=0.0)),
        ],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([10.0, 20.0])),
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning=nirs4all.NativeTuning(
            engine="optuna",
            space={"scale": [nirs4all.TuningPassthrough()], "ridge.alpha": [0.2]},
            sampler="grid",
            n_trials=1,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(X=np.asarray([[1.0], [2.0]]), y=np.asarray([1.2, 2.2])),
            winner=nirs4all.TuningWinner(
                X=np.asarray([[3.0], [4.0]]),
                y_true=np.asarray([3.2, 4.2]),
                score=0.0,
                metric="rmse",
                model_name="RunTuningStructuredPassthroughWinner",
            ),
            workspace_tuning_id="structured-passthrough-tune",
        ),
        verbose=0,
    )
    restored = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", "structured-passthrough-tune")

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"scale": {"kind": "passthrough"}, "ridge.alpha": 0.2}
    assert restored.to_dict() == result.tuning_result.to_dict()
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [3.2, 4.2])
    assert result.best["model_name"] == "RunTuningStructuredPassthroughWinner"


def test_run_tuning_score_data_transports_explicit_identities(monkeypatch):
    """score_data ids/groups/metadata should reach compatible predict() methods."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningIdentityAwareEstimator(alpha=0.0)}],
        dataset={
            "X": np.asarray([[1.0], [2.0]]),
            "y": np.asarray([1.0, 2.0]),
            "sample_ids": ["train-a", "train-b"],
            "groups": ["train-g1", "train-g2"],
            "metadata": [{"site": "train-1"}, {"site": "train-2"}],
        },
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {
                "X": np.asarray([[10.0], [20.0]]),
                "y": np.asarray([5.0, 5.0]),
                "sample_ids": ["score-a", "score-b"],
                "groups": ["batch-1", "batch-2"],
                "metadata": [{"site": "north"}, {"site": "south"}],
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_score_data_tuple_transports_explicit_identities(monkeypatch):
    """score_data tuples can carry X/y plus prediction identities in fixed order."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningIdentityAwareEstimator(alpha=0.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": (
                np.asarray([[10.0], [20.0]]),
                np.asarray([5.0, 5.0]),
                ["score-a", "score-b"],
                ["batch-1", "batch-2"],
                [{"site": "north"}, {"site": "south"}],
            ),
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_score_data_json_array_transports_explicit_identities(monkeypatch):
    """score_data JSON arrays use the same positional contract as tuples."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningIdentityAwareEstimator(alpha=0.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": [
                np.asarray([[10.0], [20.0]]),
                np.asarray([5.0, 5.0]),
                ["score-a", "score-b"],
                ["batch-1", "batch-2"],
                [{"site": "north"}, {"site": "south"}],
            ],
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_score_data_spectro_dataset_mapping_transports_identities(monkeypatch):
    """score_data can select an explicit SpectroDataset scoring cohort."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningIdentityAwareEstimator(alpha=0.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {
                "dataset": _make_run_tuning_spectro_dataset(),
                "selector": {"partition": "train"},
                "sample_id_column": "sample_id",
                "group_column": "group_id",
                "metadata_columns": ["site"],
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_typed_score_data_spectro_dataset_transports_identities(monkeypatch):
    """Typed TuningScoreData(dataset=...) executes the same selected scoring cohort."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    score_dataset = _make_run_tuning_spectro_dataset_with_metadata(
        name="typed-score",
        sample_ids=["score-a", "score-b"],
        groups=["batch-1", "batch-2"],
        sites=["north", "south"],
        x_values=[10.0, 20.0],
        y_values=[0.0, 0.0],
    )

    result = run_module.run(
        pipeline=[{"model": _RunTuningIdentityAwareEstimator(alpha=0.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning=nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.0, 5.0]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                dataset=score_dataset,
                selector={"partition": "train"},
                sample_id_column="sample_id",
                group_column="group_id",
                metadata_columns=["site"],
            ),
        ),
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 0.0}


def test_run_tuning_dataset_and_score_data_config_paths_transport_identities(tmp_path, monkeypatch):
    """dataset and score_data can use explicit config/path-backed selected cohorts."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    fit_config = _write_run_tuning_dataset_config(
        tmp_path,
        name="fit_dataset",
        sample_ids=["train-a", "train-b"],
        groups=["train-g1", "train-g2"],
        sites=["train-1", "train-2"],
        x_values=[1.0, 2.0],
        y_values=[1.0, 2.0],
    )
    score_config = _write_run_tuning_dataset_config(
        tmp_path,
        name="score_dataset",
        sample_ids=["score-a", "score-b"],
        groups=["batch-1", "batch-2"],
        sites=["north", "south"],
        x_values=[10.0, 20.0],
        y_values=[5.0, 5.0],
    )

    result = run_module.run(
        pipeline=[{"model": _RunTuningFitAndScoreIdentityAwareEstimator(alpha=0.0)}],
        dataset={
            "dataset": fit_config,
            "selector": {"partition": "train"},
            "sample_id_column": "sample_id",
            "group_column": "group_id",
            "metadata_columns": ["site"],
        },
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {
                "dataset": score_config,
                "selector": {"partition": "train"},
                "sample_id_column": "sample_id",
                "group_column": "group_id",
                "metadata_columns": ["site"],
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 0.0}


def test_run_tuning_score_data_spectro_dataset_requires_selector() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match=r"score_data.*explicit selector"):
        run_module.run(
            pipeline=[{"model": _RunTuningIdentityAwareEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"dataset": _make_run_tuning_spectro_dataset()},
            },
            verbose=0,
        )


def test_run_tuning_score_data_spectro_dataset_rejects_mixed_arrays() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match=r"score_data.*must not also provide X/y"):
        run_module.run(
            pipeline=[{"model": _RunTuningIdentityAwareEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {
                    "dataset": _make_run_tuning_spectro_dataset(),
                    "selector": {"partition": "train"},
                    "X": np.asarray([[10.0], [20.0]]),
                    "y": np.asarray([5.0, 5.0]),
                },
            },
            verbose=0,
        )


def test_run_tuning_score_data_tuple_or_list_rejects_ambiguous_extra_fields() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="score_data tuple/list supports at most"):
        run_module.run(
            pipeline=[{"model": _RunTuningIdentityAwareEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": (
                    np.asarray([[10.0], [20.0]]),
                    np.asarray([5.0, 5.0]),
                    ["score-a", "score-b"],
                    None,
                    None,
                    {"extra": "ambiguous"},
                ),
            },
            verbose=0,
        )


def test_run_tuning_score_data_rejects_ambiguous_aliases(monkeypatch) -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    base_tuning = {
        "engine": "optuna",
        "space": {"alpha": [0.0]},
        "sampler": "grid",
        "n_trials": 1,
        "metric": "rmse",
        "direction": "minimize",
    }
    base_kwargs = {
        "pipeline": [{"model": _RunTuningIdentityAwareEstimator()}],
        "dataset": (np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        "engine": "dag-ml",
        "verbose": 0,
    }

    with pytest.raises(ValueError, match="score_data features"):
        run_module.run(
            **base_kwargs,
            tuning={
                **base_tuning,
                "score_data": {
                    "X": np.asarray([[10.0], [20.0]]),
                    "X_score": np.asarray([[10.0], [20.0]]),
                    "y": np.asarray([5.0, 5.0]),
                },
            },
        )

    with pytest.raises(ValueError, match="score_data sample_ids"):
        run_module.run(
            **base_kwargs,
            tuning={
                **base_tuning,
                "score_data": {
                    "X": np.asarray([[10.0], [20.0]]),
                    "y": np.asarray([5.0, 5.0]),
                    "sample_ids": ["score-a", "score-b"],
                    "physical_sample_ids": ["score-a", "score-b"],
                },
            },
        )

    with pytest.raises(ValueError, match="score_data conformal_calibration"):
        run_module.run(
            **base_kwargs,
            tuning={
                **base_tuning,
                "metric": "conformal_mean_width",
                "score_data": {
                    "X": np.asarray([[10.0], [20.0]]),
                    "y": np.asarray([5.0, 5.0]),
                    "conformal_calibration": {
                        "X": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
                        "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
                    },
                    "conformal_score_calibration": {
                        "X": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
                        "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
                    },
                },
            },
        )

    with pytest.raises(ValueError, match="score_data.conformal_calibration features"):
        run_module.run(
            **base_kwargs,
            tuning={
                **base_tuning,
                "metric": "conformal_mean_width",
                "score_data": {
                    "X": np.asarray([[10.0], [20.0]]),
                    "y": np.asarray([5.0, 5.0]),
                    "conformal_calibration": {
                        "X": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
                        "X_calibration": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
                        "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
                    },
                },
            },
        )


def test_run_tuning_dataset_tuple_transports_explicit_fit_identities(monkeypatch):
    """Tuple datasets can carry X/y plus fit identities in fixed order."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningFitIdentityAwareEstimator(alpha=0.0)}],
        dataset=(
            np.asarray([[1.0], [2.0]]),
            np.asarray([1.0, 2.0]),
            ["train-a", "train-b"],
            ["train-g1", "train-g2"],
            [{"site": "train-1"}, {"site": "train-2"}],
        ),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_dataset_json_array_transports_explicit_fit_identities(monkeypatch):
    """JSON array datasets use the same positional contract as tuples."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningFitIdentityAwareEstimator(alpha=0.0)}],
        dataset=[
            np.asarray([[1.0], [2.0]]),
            np.asarray([1.0, 2.0]),
            ["train-a", "train-b"],
            ["train-g1", "train-g2"],
            [{"site": "train-1"}, {"site": "train-2"}],
        ],
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_spectro_dataset_mapping_transports_selected_fit_identities(monkeypatch):
    """SpectroDataset support requires an explicit selector and identity columns."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningFitIdentityAwareEstimator(alpha=0.0)}],
        dataset={
            "dataset": _make_run_tuning_spectro_dataset(),
            "selector": {"partition": "train"},
            "sample_id_column": "sample_id",
            "group_column": "group_id",
            "metadata_columns": ["site"],
        },
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.tuning_best_params == {"alpha": 5.0}


def test_run_tuning_rejects_bare_spectro_dataset() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="must use an explicit mapping"):
        run_module.run(
            pipeline=[{"model": _RunTuningFitIdentityAwareEstimator()}],
            dataset=_make_run_tuning_spectro_dataset(),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
            },
            verbose=0,
        )


def test_run_tuning_spectro_dataset_mapping_requires_selector() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="requires an explicit selector"):
        run_module.run(
            pipeline=[{"model": _RunTuningFitIdentityAwareEstimator()}],
            dataset={"dataset": _make_run_tuning_spectro_dataset()},
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
            },
            verbose=0,
        )


def test_run_tuning_dataset_tuple_rejects_misaligned_identities() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="dataset sample_ids contains 1 rows"):
        run_module.run(
            pipeline=[{"model": _RunTuningFitIdentityAwareEstimator()}],
            dataset=(
                np.asarray([[1.0], [2.0]]),
                np.asarray([1.0, 2.0]),
                ["train-a"],
            ),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
            },
            verbose=0,
        )


def test_run_tuning_dataset_tuple_or_list_rejects_ambiguous_extra_fields() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="tuple/list supports at most"):
        run_module.run(
            pipeline=[{"model": _RunTuningFitIdentityAwareEstimator()}],
            dataset=(
                np.asarray([[1.0], [2.0]]),
                np.asarray([1.0, 2.0]),
                ["train-a", "train-b"],
                None,
                None,
                {"extra": "ambiguous"},
            ),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([5.0, 5.0])},
            },
            verbose=0,
        )


def test_run_tuning_score_data_rejects_misaligned_identities() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="score sample_ids contains 1 rows"):
        run_module.run(
            pipeline=[{"model": _RunTuningIdentityAwareEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.0]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {
                    "X": np.asarray([[10.0], [20.0]]),
                    "y": np.asarray([5.0, 5.0]),
                    "sample_ids": ["score-a"],
                },
            },
            verbose=0,
        )


def test_run_tuning_winner_accepts_physical_sample_id_alias(monkeypatch):
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "X": np.asarray([[30.0], [40.0]]),
                "y_true": np.asarray([0.0, 0.0]),
                "score": 0.2,
                "metric": "rmse",
                "physical_sample_ids": ["phys-a", "phys-b"],
            },
        },
        verbose=0,
    )

    assert result.best["metadata"]["physical_sample_id"] == ["phys-a", "phys-b"]


def test_run_tuning_winner_rejects_ambiguous_aliases(monkeypatch):
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    with pytest.raises(ValueError, match="winner features"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
                "winner": {
                    "X": np.asarray([[30.0], [40.0]]),
                    "winner_x": np.asarray([[30.0], [40.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "score": 0.2,
                    "metric": "rmse",
                    "sample_ids": ["phys-a", "phys-b"],
                },
            },
            verbose=0,
        )

    with pytest.raises(ValueError, match="winner sample_ids"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
                "winner": {
                    "X": np.asarray([[30.0], [40.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "score": 0.2,
                    "metric": "rmse",
                    "sample_ids": ["phys-a", "phys-b"],
                    "physical_sample_ids": ["phys-a", "phys-b"],
                },
            },
            verbose=0,
        )


def test_run_tuning_winner_spectro_dataset_mapping_projects_selected_cohort(monkeypatch):
    """winner can select an explicit SpectroDataset cohort for projection/calibration."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "dataset": _make_run_tuning_spectro_dataset(),
                "selector": {"partition": "train"},
                "sample_id_column": "sample_id",
                "group_column": "group_id",
                "metadata_columns": ["site"],
                "score": 0.2,
                "metric": "rmse",
                "model_name": "SpectroWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.best["model_name"] == "SpectroWinner"
    assert result.best["metadata"]["physical_sample_id"] == ["train-a", "train-b"]
    assert result.best["metadata"]["group"] == ["train-g1", "train-g2"]
    assert result.best["metadata"]["site"] == ["train-1", "train-2"]
    np.testing.assert_allclose(result.best["y_pred"], [0.2, 0.2])


def test_run_tuning_typed_winner_spectro_dataset_projects_selected_cohort(monkeypatch):
    """Typed TuningWinner(dataset=...) projects metadata from the selected cohort."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    winner_dataset = _make_run_tuning_spectro_dataset_with_metadata(
        name="typed-winner",
        sample_ids=["winner-a", "winner-b"],
        groups=["winner-g1", "winner-g2"],
        sites=["winner-1", "winner-2"],
        x_values=[30.0, 40.0],
        y_values=[0.0, 0.0],
    )

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning=nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.2]},
            sampler="grid",
            n_trials=1,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(X=np.asarray([[10.0], [20.0]]), y=np.asarray([0.0, 0.0])),
            winner=nirs4all.TuningWinner(
                dataset=winner_dataset,
                selector={"partition": "train"},
                sample_id_column="sample_id",
                group_column="group_id",
                metadata_columns=["site"],
                score=0.2,
                metric="rmse",
                model_name="TypedSpectroWinner",
            ),
        ),
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.best["model_name"] == "TypedSpectroWinner"
    assert result.best["metadata"]["physical_sample_id"] == ["winner-a", "winner-b"]
    assert result.best["metadata"]["group"] == ["winner-g1", "winner-g2"]
    assert result.best["metadata"]["site"] == ["winner-1", "winner-2"]
    np.testing.assert_allclose(result.best["y_pred"], [0.2, 0.2])


def test_run_tuning_winner_config_path_projects_selected_cohort(tmp_path, monkeypatch):
    """winner can select an explicit config/path-backed cohort for projection."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    winner_config = _write_run_tuning_dataset_config(
        tmp_path,
        name="winner_dataset",
        sample_ids=["winner-a", "winner-b"],
        groups=["winner-g1", "winner-g2"],
        sites=["winner-1", "winner-2"],
        x_values=[30.0, 40.0],
        y_values=[0.0, 0.0],
    )

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "dataset": winner_config,
                "selector": {"partition": "train"},
                "sample_id_column": "sample_id",
                "group_column": "group_id",
                "metadata_columns": ["site"],
                "score": 0.2,
                "metric": "rmse",
                "model_name": "ConfigPathWinner",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert result.best["model_name"] == "ConfigPathWinner"
    assert result.best["metadata"]["physical_sample_id"] == ["winner-a", "winner-b"]
    assert result.best["metadata"]["group"] == ["winner-g1", "winner-g2"]
    assert result.best["metadata"]["site"] == ["winner-1", "winner-2"]
    np.testing.assert_allclose(result.best["y_pred"], [0.2, 0.2])


def test_run_tuning_winner_spectro_dataset_requires_selector() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match=r"winner.*explicit selector"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
                "winner": {
                    "dataset": _make_run_tuning_spectro_dataset(),
                    "score": 0.2,
                    "metric": "rmse",
                },
            },
            verbose=0,
        )


def test_run_tuning_winner_spectro_dataset_rejects_mixed_arrays() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match=r"winner.*must not also provide X/y"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "metric": "rmse",
                "direction": "minimize",
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
                "winner": {
                    "dataset": _make_run_tuning_spectro_dataset(),
                    "selector": {"partition": "train"},
                    "X": np.asarray([[30.0], [40.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "score": 0.2,
                    "metric": "rmse",
                },
            },
            verbose=0,
        )


def test_run_tuning_can_return_integrated_conformal_result(tmp_path, monkeypatch):
    """run(tuning={..., calibration={...}}) should chain winner calibration."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "X": np.asarray([[30.0], [40.0], [50.0], [60.0]]),
                "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "score": 0.2,
                "metric": "rmse",
                "sample_ids": ["cal-a", "cal-b", "cal-c", "cal-d"],
                "model_name": "RunTuningCalibratedWinner",
            },
            "workspace_tuning_id": "run-calibrated-tune",
            "calibration": {
                "y_pred": np.asarray([1.0, 2.0]),
                "prediction_sample_ids": ["pred-a", "pred-b"],
                "coverage": 0.8,
                "workspace_conformal_id": "run-calibrated-conformal",
                "workspace_metadata": {"source": "run-tuning-calibration-test"},
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.run.tuning_best_params == {"alpha": 0.2}
    assert result.run.best["model_name"] == "RunTuningCalibratedWinner"
    assert result.calibrated.conformal_guarantee_status is not None
    assert result.calibrated.conformal_guarantee_status["status"] == "active"
    np.testing.assert_allclose(result.calibrated.interval(0.8).lower, [0.8, 1.8])
    np.testing.assert_allclose(result.calibrated.interval(0.8).upper, [1.2, 2.2])
    restored_tuning = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", "run-calibrated-tune")
    restored_calibrated = nirs4all.load_workspace_calibrated_result(tmp_path / "workspace", "run-calibrated-conformal")
    assert restored_tuning.to_dict() == result.run.tuning_result.to_dict()
    assert restored_calibrated.fingerprint == result.calibrated.metadata["calibrated_result_fingerprint"]


def test_run_tuning_workspace_conformal_reload_preserves_winner_calibration_not_score_data(tmp_path, monkeypatch):
    """Reloaded conformal workspace rows must preserve winner-derived intervals."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "X": np.asarray([[30.0], [40.0], [50.0], [60.0]]),
                "y_true": np.asarray([10.0, 10.0, 10.0, 10.0]),
                "score": 9.8,
                "metric": "rmse",
                "sample_ids": ["cal-a", "cal-b", "cal-c", "cal-d"],
                "model_name": "RunTuningWideCalibrationWinner",
            },
            "workspace_tuning_id": "run-wide-calibrated-tune",
            "calibration": {
                "y_pred": np.asarray([1.0, 2.0]),
                "prediction_sample_ids": ["pred-a", "pred-b"],
                "coverage": 0.8,
                "result_metadata": {"operator": "workspace-unit"},
                "workspace_conformal_id": "run-wide-calibrated-conformal",
            },
        },
        verbose=0,
    )

    restored = nirs4all.load_workspace_calibrated_result(tmp_path / "workspace", "run-wide-calibrated-conformal")

    assert _DummyRunner.instances == []
    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    np.testing.assert_allclose(result.calibrated.interval(0.8).lower, [-8.8, -7.8])
    np.testing.assert_allclose(result.calibrated.interval(0.8).upper, [10.8, 11.8])
    restored_interval = restored.to_dict()["prediction"]["intervals"][0]
    np.testing.assert_allclose(restored_interval["lower"], [-8.8, -7.8])
    np.testing.assert_allclose(restored_interval["upper"], [10.8, 11.8])
    restored_prediction = restored.to_predict_result()
    np.testing.assert_allclose(restored_prediction.interval(0.8).lower, [-8.8, -7.8])
    np.testing.assert_allclose(restored_prediction.interval(0.8).upper, [10.8, 11.8])
    assert restored.fingerprint == result.calibrated.metadata["calibrated_result_fingerprint"]
    assert restored_prediction.metadata["calibrated_result_fingerprint"] == restored.fingerprint
    assert result.calibrated.metadata["operator"] == "workspace-unit"
    assert restored.metadata["operator"] == "workspace-unit"
    assert result.calibrated.tuning_calibration_source == {
        "source": "tuning.winner",
        "score_data_role": "hpo_objective_only",
        "score_data_used": False,
    }
    assert restored.tuning_calibration_source == result.calibrated.tuning_calibration_source
    assert restored_prediction.tuning_calibration_source == result.calibrated.tuning_calibration_source


def test_run_tuning_accepts_top_level_calibration_alias(monkeypatch):
    """run(..., calibration={...}) should alias tuning.calibration for PC9."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "X": np.asarray([[30.0], [40.0], [50.0], [60.0]]),
                "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "score": 0.2,
                "metric": "rmse",
                "sample_ids": ["cal-a", "cal-b", "cal-c", "cal-d"],
                "model_name": "TopLevelCalibrationWinner",
            },
        },
        calibration=nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0, 2.0]),
            prediction_sample_ids=["pred-a", "pred-b"],
            coverage=0.8,
        ),
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.run.best["model_name"] == "TopLevelCalibrationWinner"
    assert result.calibrated.conformal_guarantee_status is not None
    assert result.calibrated.conformal_guarantee_status["status"] == "active"
    np.testing.assert_allclose(result.calibrated.interval(0.8).lower, [0.8, 1.8])
    np.testing.assert_allclose(result.calibrated.interval(0.8).upper, [1.2, 2.2])


def test_run_top_level_calibration_fails_closed_without_tuning() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(NotImplementedError, match=r"requires run\(tuning="):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"]},
            verbose=0,
        )


def test_run_top_level_calibration_rejects_nested_duplicate() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match=r"either as run\(calibration=\.\.\.\) or tuning\.calibration"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
                "calibration": {"y_pred": [1.0], "prediction_sample_ids": ["pred-a"]},
            },
            calibration={"y_pred": [2.0], "prediction_sample_ids": ["pred-b"]},
            verbose=0,
        )


def test_run_tuning_calibration_rejects_manual_calibration_data(monkeypatch) -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    base_tuning = {
        "engine": "optuna",
        "space": {"alpha": [0.2]},
        "sampler": "grid",
        "n_trials": 1,
        "metric": "rmse",
        "direction": "minimize",
        "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
        "winner": {
            "X": np.asarray([[30.0], [40.0], [50.0], [60.0]]),
            "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
            "score": 0.2,
            "metric": "rmse",
            "sample_ids": ["cal-a", "cal-b", "cal-c", "cal-d"],
        },
    }
    base_kwargs = {
        "pipeline": [{"model": _RunTuningEstimator(alpha=1.0)}],
        "dataset": (np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        "engine": "dag-ml",
        "verbose": 0,
    }

    with pytest.raises(ValueError, match="must not include calibration_data"):
        run_module.run(
            **base_kwargs,
            tuning={
                **base_tuning,
                "calibration": {
                    "calibration_data": {"y_true": [0.0]},
                    "y_pred": [1.0],
                    "prediction_sample_ids": ["pred-a"],
                },
            },
        )

    with pytest.raises(ValueError, match="must not include calibration_data"):
        run_module.run(
            **base_kwargs,
            tuning=base_tuning,
            calibration={
                "calibration_data": {"y_true": [0.0]},
                "y_pred": [1.0],
                "prediction_sample_ids": ["pred-a"],
            },
        )


def test_run_tuning_spectro_winner_can_chain_integrated_conformal_result(tmp_path, monkeypatch):
    """SpectroDataset winner projection should be valid conformal calibration evidence."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "winner": {
                "dataset": _make_run_tuning_spectro_dataset(),
                "selector": {"partition": "train"},
                "sample_id_column": "sample_id",
                "group_column": "group_id",
                "metadata_columns": ["site"],
                "score": 0.2,
                "metric": "rmse",
                "model_name": "SpectroCalibratedWinner",
            },
            "workspace_tuning_id": "run-spectro-calibrated-tune",
            "calibration": {
                "y_pred": np.asarray([1.0, 2.0]),
                "prediction_sample_ids": ["pred-a", "pred-b"],
                "coverage": 0.5,
                "workspace_conformal_id": "run-spectro-calibrated-conformal",
            },
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.run.best["model_name"] == "SpectroCalibratedWinner"
    assert result.run.best["metadata"]["physical_sample_id"] == ["train-a", "train-b"]
    assert result.run.best["metadata"]["group"] == ["train-g1", "train-g2"]
    assert result.run.best["metadata"]["site"] == ["train-1", "train-2"]
    assert result.calibrated.conformal_guarantee_status is not None
    assert result.calibrated.conformal_guarantee_status["status"] == "active"
    np.testing.assert_allclose(result.calibrated.interval(0.5).lower, [0.2, 1.2])
    np.testing.assert_allclose(result.calibrated.interval(0.5).upper, [1.8, 2.8])
    restored_calibrated = nirs4all.load_workspace_calibrated_result(tmp_path / "workspace", "run-spectro-calibrated-conformal")
    assert restored_calibrated.fingerprint == result.calibrated.metadata["calibrated_result_fingerprint"]


def test_run_tuning_single_estimator_array_subset_persists_workspace_result(tmp_path, monkeypatch):
    """run(tuning=...) should persist TuningResult when workspace_path is supplied."""
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    result = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset={
            "X": np.asarray([[1.0], [2.0]]),
            "y": np.asarray([1.0, 2.0]),
            "sample_ids": ["train-a", "train-b"],
        },
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning={
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
            "workspace_tuning_id": "run-tuning-main",
            "workspace_metadata": {"source": "run-tuning-test"},
        },
        verbose=0,
    )
    restored = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", "run-tuning-main")

    assert _DummyRunner.instances == []
    assert result.tuning_id == "run-tuning-main"
    assert result.tuning_best_params == {"alpha": 0.2}
    assert restored.to_dict() == result.tuning_result.to_dict()


def test_run_tuning_workspace_tuning_id_alias_is_fail_closed(tmp_path, monkeypatch):
    import nirs4all

    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    base_tuning = {
        "engine": "optuna",
        "space": {"alpha": [0.2]},
        "sampler": "grid",
        "n_trials": 1,
        "metric": "rmse",
        "direction": "minimize",
        "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
    }
    base_kwargs = {
        "pipeline": [{"model": _RunTuningEstimator(alpha=1.0)}],
        "dataset": (np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        "engine": "dag-ml",
        "workspace_path": tmp_path / "workspace",
        "verbose": 0,
    }

    result = run_module.run(
        **base_kwargs,
        tuning={
            **base_tuning,
            "tuning_id": "alias-tune-main",
        },
    )

    restored = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", "alias-tune-main")
    assert result.tuning_id == "alias-tune-main"
    assert restored.to_dict() == result.tuning_result.to_dict()

    with pytest.raises(ValueError, match="workspace_tuning_id"):
        run_module.run(
            **base_kwargs,
            tuning={
                **base_tuning,
                "workspace_tuning_id": "canonical-tune",
                "tuning_id": "alias-tune",
            },
        )


def test_run_tuning_single_estimator_resume_reuses_workspace_result(tmp_path, monkeypatch):
    """resume=True should reload the completed TuningResult instead of reoptimizing."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    tuning = {
        "engine": "optuna",
        "space": {"alpha": [0.9, 0.2]},
        "sampler": "grid",
        "n_trials": 2,
        "metric": "rmse",
        "direction": "minimize",
        "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([0.0, 0.0])},
        "workspace_tuning_id": "resume-tune-main",
    }
    first = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning=tuning,
        verbose=0,
    )
    resumed = run_module.run(
        pipeline=[{"model": _RunTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        workspace_path=tmp_path / "workspace",
        tuning={
            **tuning,
            "resume": True,
            # If the optimizer ran again, this score cohort would prefer alpha=0.9.
            "score_data": {"X": np.asarray([[10.0], [20.0]]), "y": np.asarray([1.0, 1.0])},
        },
        verbose=0,
    )

    assert _DummyRunner.instances == []
    assert first.tuning_best_params == {"alpha": 0.2}
    assert resumed.tuning_id == "resume-tune-main"
    assert resumed.tuning_best_params == {"alpha": 0.2}
    assert resumed.tuning_result.to_dict() == first.tuning_result.to_dict()


def test_run_tuning_resume_requires_workspace_identifier() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="workspace_path and tuning.workspace_tuning_id"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "resume": True,
                "score_data": {"X": [[10.0]], "y": [0.0]},
            },
            verbose=0,
        )


def test_run_tuning_dagml_subset_requires_explicit_score_data() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="score_data"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={"engine": "optuna", "space": {"alpha": [0.2]}, "sampler": "grid", "n_trials": 1},
            verbose=0,
        )


def test_run_tuning_rejects_unsupported_runtime_keys() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(ValueError, match="does not support keys"):
        run_module.run(
            pipeline=[{"model": _RunTuningEstimator()}],
            dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
            engine="dag-ml",
            tuning={
                "engine": "optuna",
                "space": {"alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
                "score_data": {"X": [[10.0]], "y": [0.0]},
                "unknown_runtime": {"y_pred": [1.0]},
            },
            verbose=0,
        )

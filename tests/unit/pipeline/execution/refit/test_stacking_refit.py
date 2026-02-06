"""Tests for the stacking refit executor (Tasks 3.3, 3.4, 3.5).

Covers:
- Task 3.3: execute_stacking_refit() two-step design (base models + meta-model)
- Task 3.4: Mixed merge handling (features + predictions branches)
- Task 3.5: GPU-aware serialization (_is_gpu_model, sequential dispatch, cleanup)
- Orchestrator integration for stacking dispatch
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.analysis.topology import PipelineTopology, analyze_topology
from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
from nirs4all.pipeline.execution.refit.executor import RefitResult
from nirs4all.pipeline.execution.refit.stacking_refit import (
    _any_branch_has_gpu_model,
    _classify_branch_type,
    _cleanup_gpu_memory,
    _create_meta_dataset,
    _extract_in_sample_predictions,
    _extract_model_class_path,
    _extract_model_from_steps,
    _extract_post_merge_steps,
    _extract_pre_branch_steps,
    _extract_preprocessing_from_branch,
    _extract_test_score_from_predictions,
    _find_branch_step,
    _find_merge_step,
    _is_gpu_model,
    _relabel_stacking_predictions,
    _replace_splitter,
    execute_stacking_refit,
)
from nirs4all.pipeline.storage.store_schema import REFIT_CONTEXT_STACKING

# =========================================================================
# Helpers
# =========================================================================


class _DummyModel:
    """Minimal sklearn-like model for testing."""

    def __init__(self, n_components: int = 5, alpha: float = 1.0) -> None:
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "alpha": self.alpha}


class _DummySplitter:
    """Minimal CV splitter for testing."""

    def split(self, X, y=None, groups=None):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        half = n // 2
        yield list(range(half)), list(range(half, n))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1


class _DummyDataset:
    """Minimal SpectroDataset stand-in for testing."""

    def __init__(self, n_train: int = 50) -> None:
        self._n_train = n_train
        self.name = "test_dataset"
        self.aggregate = None
        self.aggregate_method = None
        self.aggregate_exclude_outliers = False
        self.repetition = None

    def x(self, selector: dict, layout: str = "2d") -> np.ndarray:
        return np.random.randn(self._n_train, 10)

    def y(self, selector: dict | None = None) -> np.ndarray:
        return np.random.randn(self._n_train)

    def features_sources(self) -> int:
        return 1

    def __deepcopy__(self, memo):
        new = _DummyDataset(self._n_train)
        new.name = self.name
        return new


def _make_stacking_steps() -> list[Any]:
    """Create a typical stacking pipeline step list."""
    return [
        {"class": "sklearn.preprocessing.MinMaxScaler"},
        {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
        {"branch": [
            [{"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}}}],
            [{"model": {"class": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": 100}}}],
        ]},
        {"merge": "predictions"},
        {"model": {"class": "sklearn.linear_model.Ridge", "params": {"alpha": 1.0}}},
    ]


def _make_stacking_refit_config(**overrides) -> RefitConfig:
    """Create a RefitConfig for stacking tests."""
    defaults = {
        "expanded_steps": _make_stacking_steps(),
        "best_params": {},
        "variant_index": 0,
        "generator_choices": [],
        "pipeline_id": "test-pipeline-id",
        "metric": "rmse",
        "best_score": 0.05,
    }
    defaults.update(overrides)
    return RefitConfig(**defaults)


def _make_stacking_topology() -> PipelineTopology:
    """Create a topology descriptor for a stacking pipeline."""
    return analyze_topology(_make_stacking_steps())


def _make_mock_executor() -> MagicMock:
    """Create a mock PipelineExecutor."""
    executor = MagicMock()
    executor.initialize_context.return_value = MagicMock()
    executor.execute.return_value = None
    return executor


# =========================================================================
# Task 3.5: GPU model detection
# =========================================================================


class TestIsGpuModel:
    """Tests for _is_gpu_model helper."""

    def test_sklearn_model_not_gpu(self):
        """sklearn models are not GPU models."""
        assert _is_gpu_model({"class": "sklearn.linear_model.Ridge"}) is False

    def test_tensorflow_model_is_gpu(self):
        """TensorFlow models are detected as GPU."""
        assert _is_gpu_model({"class": "tensorflow.keras.Sequential"}) is True

    def test_keras_model_is_gpu(self):
        """Keras models are detected as GPU."""
        assert _is_gpu_model({"class": "keras.models.Sequential"}) is True

    def test_pytorch_model_is_gpu(self):
        """PyTorch models are detected as GPU."""
        assert _is_gpu_model({"class": "torch.nn.Module"}) is True

    def test_jax_model_is_gpu(self):
        """JAX models are detected as GPU."""
        assert _is_gpu_model({"class": "jax.experimental.stax.serial"}) is True

    def test_flax_model_is_gpu(self):
        """Flax models are detected as GPU."""
        assert _is_gpu_model({"class": "flax.linen.Module"}) is True

    def test_string_class_path(self):
        """String class paths work."""
        assert _is_gpu_model("torch.nn.Linear") is True
        assert _is_gpu_model("sklearn.linear_model.Ridge") is False

    def test_none_input(self):
        """None returns False."""
        assert _is_gpu_model(None) is False

    def test_empty_dict(self):
        """Empty dict returns False."""
        assert _is_gpu_model({}) is False

    def test_live_instance(self):
        """Live instances have their class inspected."""
        model = _DummyModel()
        assert _is_gpu_model(model) is False


class TestExtractModelClassPath:
    """Tests for _extract_model_class_path helper."""

    def test_string_input(self):
        """String inputs are returned as-is."""
        assert _extract_model_class_path("sklearn.linear_model.Ridge") == "sklearn.linear_model.Ridge"

    def test_dict_with_class(self):
        """Dict inputs use the 'class' key."""
        assert _extract_model_class_path({"class": "foo.Bar"}) == "foo.Bar"

    def test_dict_without_class(self):
        """Dict without 'class' returns empty string."""
        assert _extract_model_class_path({"params": {}}) == ""

    def test_live_instance(self):
        """Live instances return module.qualname."""
        model = _DummyModel()
        path = _extract_model_class_path(model)
        assert "_DummyModel" in path

    def test_none_returns_empty(self):
        """None returns empty string."""
        assert _extract_model_class_path(None) == ""


class TestAnyBranchHasGpuModel:
    """Tests for _any_branch_has_gpu_model helper."""

    def test_no_gpu_models(self):
        """All sklearn branches -> False."""
        branches = [
            [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
            [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
        ]
        assert _any_branch_has_gpu_model(branches) is False

    def test_one_gpu_model(self):
        """One GPU branch -> True."""
        branches = [
            [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
            [{"model": {"class": "tensorflow.keras.Sequential"}}],
        ]
        assert _any_branch_has_gpu_model(branches) is True

    def test_empty_branches(self):
        """Empty branch list -> False."""
        assert _any_branch_has_gpu_model([]) is False


class TestCleanupGpuMemory:
    """Tests for _cleanup_gpu_memory helper."""

    def test_does_not_raise(self):
        """Cleanup never raises even if frameworks are not installed."""
        _cleanup_gpu_memory()  # Should not raise


# =========================================================================
# Task 3.4: Branch classification for mixed merge
# =========================================================================


class TestClassifyBranchType:
    """Tests for _classify_branch_type helper."""

    def test_string_merge_predictions(self):
        """String merge 'predictions' -> all branches are predictions."""
        merge_step = {"merge": "predictions"}
        assert _classify_branch_type(0, merge_step) == "predictions"
        assert _classify_branch_type(1, merge_step) == "predictions"

    def test_string_merge_features(self):
        """String merge 'features' -> all branches are features."""
        merge_step = {"merge": "features"}
        assert _classify_branch_type(0, merge_step) == "features"

    def test_mixed_merge_predictions_branch(self):
        """Mixed merge dict: branch in predictions list."""
        merge_step = {"merge": {"predictions": [0], "features": [1]}}
        assert _classify_branch_type(0, merge_step) == "predictions"

    def test_mixed_merge_features_branch(self):
        """Mixed merge dict: branch in features list."""
        merge_step = {"merge": {"predictions": [0], "features": [1]}}
        assert _classify_branch_type(1, merge_step) == "features"

    def test_mixed_merge_unknown_branch(self):
        """Mixed merge dict: branch not in either list."""
        merge_step = {"merge": {"predictions": [0], "features": [1]}}
        assert _classify_branch_type(2, merge_step) == "unknown"

    def test_none_merge_step(self):
        """None merge step defaults to predictions."""
        assert _classify_branch_type(0, None) == "predictions"

    def test_empty_dict_merge(self):
        """Empty dict merge defaults to predictions."""
        merge_step = {"merge": {}}
        assert _classify_branch_type(0, merge_step) == "predictions"


# =========================================================================
# Step extraction helpers
# =========================================================================


class TestFindBranchStep:
    """Tests for _find_branch_step helper."""

    def test_finds_branch(self):
        """Finds branch step and its index."""
        steps = _make_stacking_steps()
        result = _find_branch_step(steps)
        assert result is not None
        idx, step = result
        assert idx == 2
        assert "branch" in step

    def test_no_branch(self):
        """Returns None when no branch exists."""
        steps = [{"class": "sklearn.preprocessing.MinMaxScaler"}]
        assert _find_branch_step(steps) is None


class TestFindMergeStep:
    """Tests for _find_merge_step helper."""

    def test_finds_merge_after_branch(self):
        """Finds the merge step after the branch index."""
        steps = _make_stacking_steps()
        merge = _find_merge_step(steps, after=2)
        assert merge is not None
        assert merge["merge"] == "predictions"

    def test_no_merge(self):
        """Returns None when no merge exists."""
        steps = [{"branch": [[], []]}]
        assert _find_merge_step(steps, after=0) is None


class TestExtractPreBranchSteps:
    """Tests for _extract_pre_branch_steps helper."""

    def test_extracts_preprocessing(self):
        """Extracts steps before the branch."""
        steps = _make_stacking_steps()
        pre = _extract_pre_branch_steps(steps, branch_idx=2)
        assert len(pre) == 2  # MinMaxScaler + KFold
        assert pre[0]["class"] == "sklearn.preprocessing.MinMaxScaler"

    def test_branch_at_start(self):
        """Returns empty list if branch is at index 0."""
        steps = [{"branch": [[], []]}]
        assert _extract_pre_branch_steps(steps, branch_idx=0) == []


class TestExtractPostMergeSteps:
    """Tests for _extract_post_merge_steps helper."""

    def test_extracts_meta_model(self):
        """Extracts steps after the merge."""
        steps = _make_stacking_steps()
        post = _extract_post_merge_steps(steps, branch_idx=2)
        assert len(post) == 1  # Ridge meta-model
        assert "model" in post[0]

    def test_no_merge(self):
        """Returns empty list if no merge step found."""
        steps = [{"branch": [[], []]}]
        assert _extract_post_merge_steps(steps, branch_idx=0) == []


class TestExtractModelFromSteps:
    """Tests for _extract_model_from_steps helper."""

    def test_dict_model_step(self):
        """Extracts model from dict step with 'model' key."""
        steps = [{"model": {"class": "sklearn.linear_model.Ridge"}}]
        model = _extract_model_from_steps(steps)
        assert model == {"class": "sklearn.linear_model.Ridge"}

    def test_live_model_instance(self):
        """Extracts live model instance."""
        model = _DummyModel()
        steps = [model]
        assert _extract_model_from_steps(steps) is model

    def test_no_model(self):
        """Returns None when no model found."""
        steps = [{"class": "sklearn.preprocessing.MinMaxScaler"}]
        assert _extract_model_from_steps(steps) is None

    def test_skips_splitter(self):
        """Splitters are not returned as models."""
        steps = [_DummySplitter()]
        assert _extract_model_from_steps(steps) is None


class TestExtractPreprocessingFromBranch:
    """Tests for _extract_preprocessing_from_branch helper."""

    def test_filters_model_steps(self):
        """Removes model steps, keeps preprocessing."""
        branch_steps = [
            {"class": "nirs4all.operators.transforms.SNV"},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]
        result = _extract_preprocessing_from_branch(branch_steps)
        assert len(result) == 1
        assert result[0]["class"] == "nirs4all.operators.transforms.SNV"

    def test_empty_branch(self):
        """Empty branch returns empty list."""
        assert _extract_preprocessing_from_branch([]) == []

    def test_model_only_branch(self):
        """Branch with only model returns empty list."""
        branch_steps = [{"model": {"class": "PLS"}}]
        assert _extract_preprocessing_from_branch(branch_steps) == []


# =========================================================================
# Prediction helpers
# =========================================================================


class TestRelabelStackingPredictions:
    """Tests for _relabel_stacking_predictions helper."""

    def test_base_model_labeling(self):
        """Base model predictions get stacking labels."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="train",
        )

        _relabel_stacking_predictions(preds, branch_index=0)

        entry = preds._buffer[0]
        assert entry["fold_id"] == "final"
        assert entry["refit_context"] == REFIT_CONTEXT_STACKING
        assert entry["metadata"]["stacking_role"] == "base_model"
        assert entry["metadata"]["stacking_branch"] == 0

    def test_meta_model_labeling(self):
        """Meta-model predictions get stacking labels."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds", model_name="Ridge", fold_id="fold_0", partition="train",
        )

        _relabel_stacking_predictions(preds, branch_index=None, is_meta=True)

        entry = preds._buffer[0]
        assert entry["fold_id"] == "final"
        assert entry["refit_context"] == REFIT_CONTEXT_STACKING
        assert entry["metadata"]["stacking_role"] == "meta_model"

    def test_empty_predictions(self):
        """No error on empty predictions."""
        preds = Predictions()
        _relabel_stacking_predictions(preds, branch_index=0)  # Should not raise


class TestExtractInSamplePredictions:
    """Tests for _extract_in_sample_predictions helper."""

    def test_extracts_numpy_predictions(self):
        """Extracts y_pred as numpy array."""
        preds = Predictions()
        y_pred = np.array([1.0, 2.0, 3.0])
        preds.add_prediction(
            dataset_name="ds", model_name="PLS", fold_id="fold_0",
            partition="train", y_pred=y_pred,
        )

        result = _extract_in_sample_predictions(preds)
        assert result is not None
        np.testing.assert_array_equal(result, y_pred)

    def test_extracts_list_predictions(self):
        """Converts list predictions to numpy."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds", model_name="PLS", fold_id="fold_0",
            partition="train", y_pred=[1.0, 2.0, 3.0],
        )

        result = _extract_in_sample_predictions(preds)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returns_none_for_no_predictions(self):
        """Returns None when no y_pred is present."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds", model_name="PLS", fold_id="fold_0",
            partition="train",
        )
        assert _extract_in_sample_predictions(preds) is None

    def test_returns_none_for_empty(self):
        """Returns None for empty predictions."""
        preds = Predictions()
        assert _extract_in_sample_predictions(preds) is None


class TestExtractTestScoreFromPredictions:
    """Tests for _extract_test_score_from_predictions helper."""

    def test_extracts_test_score(self):
        """Extracts test_score from entry."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds", model_name="Ridge", fold_id="final",
            partition="train", test_score=0.08,
        )
        assert _extract_test_score_from_predictions(preds) == pytest.approx(0.08)

    def test_returns_none_when_missing(self):
        """Returns None when no test_score."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds", model_name="Ridge", fold_id="final",
            partition="train",
        )
        assert _extract_test_score_from_predictions(preds) is None


class TestReplaceSplitter:
    """Tests for _replace_splitter helper."""

    def test_replaces_splitter(self):
        """Replaces a splitter step with _FullTrainFoldSplitter."""
        from nirs4all.pipeline.execution.refit.executor import _FullTrainFoldSplitter

        splitter = _DummySplitter()
        steps = [{"class": "sklearn.preprocessing.MinMaxScaler"}, splitter]
        dataset = _DummyDataset(n_train=42)

        result = _replace_splitter(steps, dataset)
        assert isinstance(result[1], _FullTrainFoldSplitter)

    def test_no_splitter(self):
        """Returns unchanged steps when no splitter present."""
        steps = [{"class": "sklearn.preprocessing.MinMaxScaler"}]
        dataset = _DummyDataset()
        result = _replace_splitter(steps, dataset)
        assert len(result) == 1


class TestCreateMetaDataset:
    """Tests for _create_meta_dataset helper."""

    def test_creates_dataset_with_meta_features(self):
        """Creates a SpectroDataset with the given meta-features."""
        meta_X = np.random.randn(50, 3)
        original = _DummyDataset(n_train=50)

        meta_ds = _create_meta_dataset(meta_X, original)

        assert meta_ds.name == "test_dataset_meta"
        # Should have training samples
        X = meta_ds.x({"partition": "train"}, layout="2d")
        assert X.shape == (50, 3)


# =========================================================================
# Task 3.3: execute_stacking_refit (two-step design)
# =========================================================================


class TestExecuteStackingRefit:
    """Tests for the execute_stacking_refit function."""

    def test_success_with_stacking_pipeline(self):
        """Stacking refit completes successfully."""
        config = _make_stacking_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        # Make executor.execute add a prediction with y_pred to simulate
        # base model output
        call_count = [0]

        def mock_execute(**kwargs):
            call_count[0] += 1
            ps = kwargs["prediction_store"]
            # First two calls are base models, third is meta-model
            if call_count[0] <= 2:
                ps.add_prediction(
                    dataset_name="ds",
                    model_name=f"base_{call_count[0]}",
                    fold_id="fold_0",
                    partition="train",
                    y_pred=np.random.randn(50),
                )
            else:
                ps.add_prediction(
                    dataset_name="ds",
                    model_name="meta",
                    fold_id="fold_0",
                    partition="train",
                    test_score=0.05,
                )

        executor.execute.side_effect = mock_execute

        result = execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        assert result.success is True
        assert result.refit_context == REFIT_CONTEXT_STACKING
        assert result.metric == "rmse"
        assert result.test_score == pytest.approx(0.05)
        # Phase should be reset to CV
        assert runtime_context.phase == ExecutionPhase.CV

    def test_executor_called_per_branch_plus_meta(self):
        """Executor is called once per base branch + once for meta-model."""
        config = _make_stacking_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        # Simulate predictions from each call
        call_count = [0]

        def mock_execute(**kwargs):
            call_count[0] += 1
            ps = kwargs["prediction_store"]
            ps.add_prediction(
                dataset_name="ds",
                model_name=f"model_{call_count[0]}",
                fold_id="fold_0",
                partition="train",
                y_pred=np.random.randn(50),
            )

        executor.execute.side_effect = mock_execute

        execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        # 2 base branches + 1 meta-model = 3 calls
        assert executor.execute.call_count == 3

    def test_pipeline_names_include_branch_and_meta(self):
        """Pipeline names reflect branch index and meta-model."""
        config = _make_stacking_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        runtime_context.pipeline_name = "my_pipeline"
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        # Simulate predictions
        call_count = [0]

        def mock_execute(**kwargs):
            call_count[0] += 1
            ps = kwargs["prediction_store"]
            ps.add_prediction(
                dataset_name="ds",
                model_name=f"model_{call_count[0]}",
                fold_id="fold_0",
                partition="train",
                y_pred=np.random.randn(50),
            )

        executor.execute.side_effect = mock_execute

        execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        # Check config_name arguments
        call_names = [c.kwargs["config_name"] for c in executor.execute.call_args_list]
        assert "my_pipeline_stacking_refit_base_0" in call_names
        assert "my_pipeline_stacking_refit_base_1" in call_names
        assert "my_pipeline_stacking_refit_meta" in call_names

    def test_no_branch_returns_failure(self):
        """Returns failure when no branch step found."""
        config = _make_stacking_refit_config(expanded_steps=[
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ])
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = PipelineTopology()

        result = execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        assert result.success is False

    def test_base_model_failure_aborts_refit(self):
        """If a base model fails, refit aborts and phase is reset."""
        config = _make_stacking_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        executor.execute.side_effect = RuntimeError("Training failed")
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        result = execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        assert result.success is False
        assert runtime_context.phase == ExecutionPhase.CV

    def test_refit_sets_phase_to_refit(self):
        """Runtime context phase is set to REFIT during execution."""
        config = _make_stacking_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        phases_seen = []

        call_count = [0]

        def capture_phase(**kwargs):
            call_count[0] += 1
            phases_seen.append(kwargs["runtime_context"].phase)
            ps = kwargs["prediction_store"]
            ps.add_prediction(
                dataset_name="ds",
                model_name=f"model_{call_count[0]}",
                fold_id="fold_0",
                partition="train",
                y_pred=np.random.randn(50),
            )

        executor.execute.side_effect = capture_phase

        execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        assert all(p == ExecutionPhase.REFIT for p in phases_seen)
        # Phase reset after completion
        assert runtime_context.phase == ExecutionPhase.CV

    def test_deep_copies_dataset(self):
        """Dataset is deep-copied so caller's data is not mutated."""
        config = _make_stacking_refit_config()
        original_dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        call_count = [0]
        datasets_seen = []

        def track_datasets(**kwargs):
            call_count[0] += 1
            datasets_seen.append(kwargs["dataset"])
            ps = kwargs["prediction_store"]
            ps.add_prediction(
                dataset_name="ds",
                model_name=f"model_{call_count[0]}",
                fold_id="fold_0",
                partition="train",
                y_pred=np.random.randn(50),
            )

        executor.execute.side_effect = track_datasets

        execute_stacking_refit(
            refit_config=config,
            dataset=original_dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        # All datasets passed should be copies, not the original
        for ds in datasets_seen:
            assert ds is not original_dataset

    def test_predictions_relabeled_as_stacking(self):
        """All predictions get fold_id='final' and refit_context='stacking'."""
        config = _make_stacking_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = _make_stacking_topology()

        call_count = [0]

        def mock_execute(**kwargs):
            call_count[0] += 1
            ps = kwargs["prediction_store"]
            ps.add_prediction(
                dataset_name="ds",
                model_name=f"model_{call_count[0]}",
                fold_id="fold_0",
                partition="train",
                y_pred=np.random.randn(50),
            )

        executor.execute.side_effect = mock_execute

        execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

        for entry in prediction_store._buffer:
            assert entry["fold_id"] == "final"
            assert entry["refit_context"] == REFIT_CONTEXT_STACKING


# =========================================================================
# Task 3.4: Mixed merge handling
# =========================================================================


class TestMixedMergeRefit:
    """Tests for mixed merge handling in stacking refit."""

    def test_mixed_merge_steps(self):
        """Mixed merge pipeline classifies branches correctly."""
        steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                [{"class": "sklearn.preprocessing.StandardScaler"}],
            ]},
            {"merge": {"predictions": [0], "features": [1]}},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]

        assert _classify_branch_type(0, steps[3]) == "predictions"
        assert _classify_branch_type(1, steps[3]) == "features"


# =========================================================================
# Task 3.5: Sequential vs parallel GPU dispatch
# =========================================================================


class TestGpuDispatch:
    """Tests for GPU-aware sequential dispatch."""

    def test_sequential_for_gpu_models(self):
        """GPU models are detected, triggering sequential execution."""
        branches = [
            [{"model": {"class": "torch.nn.Sequential"}}],
            [{"model": {"class": "sklearn.linear_model.Ridge"}}],
        ]
        assert _any_branch_has_gpu_model(branches) is True

    def test_parallel_for_cpu_models(self):
        """All CPU models allow parallel execution."""
        branches = [
            [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
            [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
        ]
        assert _any_branch_has_gpu_model(branches) is False

    def test_cleanup_called_after_gpu_model(self):
        """GPU memory cleanup is called after GPU model refit."""
        config = _make_stacking_refit_config(expanded_steps=[
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                [{"model": {"class": "torch.nn.Sequential"}}],
                [{"model": {"class": "sklearn.linear_model.Ridge"}}],
            ]},
            {"merge": "predictions"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ])
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = analyze_topology(config.expanded_steps)

        call_count = [0]

        def mock_execute(**kwargs):
            call_count[0] += 1
            ps = kwargs["prediction_store"]
            ps.add_prediction(
                dataset_name="ds",
                model_name=f"model_{call_count[0]}",
                fold_id="fold_0",
                partition="train",
                y_pred=np.random.randn(50),
            )

        executor.execute.side_effect = mock_execute

        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit._cleanup_gpu_memory"
        ) as mock_cleanup:
            execute_stacking_refit(
                refit_config=config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=None,
                executor=executor,
                prediction_store=prediction_store,
                topology=topology,
            )

            # Cleanup should have been called for the GPU branch (branch 0)
            assert mock_cleanup.call_count >= 1


# =========================================================================
# Orchestrator integration
# =========================================================================


class TestOrchestratorStackingDispatch:
    """Tests for stacking dispatch in PipelineOrchestrator._execute_refit_pass."""

    def test_stacking_pipeline_dispatches_to_stacking_refit(self, tmp_path):
        """Stacking pipeline now dispatches to execute_stacking_refit."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            mode="train",
        )

        store = orchestrator.store

        # Create a run with a completed stacking pipeline
        run_id = store.begin_run("test", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="stacking_pipe",
            expanded_config=_make_stacking_steps(),
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h1",
        )

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "Ridge", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="Ridge",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="Ridge",
            model_class="Ridge",
            fold_id="fold_0",
            partition="val",
            val_score=0.10,
            test_score=0.12,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=100,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
        )

        store.complete_pipeline(
            pipeline_id=pipeline_id,
            best_val=0.10,
            best_test=0.12,
            metric="rmse",
            duration_ms=100,
        )
        store.complete_run(run_id, summary={})

        executor = MagicMock()

        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit.execute_stacking_refit"
        ) as mock_stacking:
            mock_stacking.return_value = RefitResult(success=True, predictions_count=0)

            orchestrator._execute_refit_pass(
                run_id=run_id,
                dataset=_DummyDataset(),
                executor=executor,
                artifact_registry=MagicMock(),
                run_dataset_predictions=Predictions(),
                run_predictions=Predictions(),
            )

            # execute_stacking_refit should have been called
            mock_stacking.assert_called_once()

        store.close()

    def test_non_stacking_pipeline_uses_simple_refit(self, tmp_path):
        """Non-stacking pipeline still uses execute_simple_refit."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            mode="train",
        )

        store = orchestrator.store

        # Create a run with a non-stacking pipeline
        simple_steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}}},
        ]
        run_id = store.begin_run("test", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="simple_pipe",
            expanded_config=simple_steps,
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h1",
        )

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="PLS",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="PLS",
            model_class="PLS",
            fold_id="fold_0",
            partition="val",
            val_score=0.10,
            test_score=0.12,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=100,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
        )

        store.complete_pipeline(
            pipeline_id=pipeline_id,
            best_val=0.10,
            best_test=0.12,
            metric="rmse",
            duration_ms=100,
        )
        store.complete_run(run_id, summary={})

        executor = MagicMock()

        with patch(
            "nirs4all.pipeline.execution.refit.execute_simple_refit"
        ) as mock_simple:
            mock_simple.return_value = RefitResult(success=True, predictions_count=0)

            orchestrator._execute_refit_pass(
                run_id=run_id,
                dataset=_DummyDataset(),
                executor=executor,
                artifact_registry=MagicMock(),
                run_dataset_predictions=Predictions(),
                run_predictions=Predictions(),
            )

            mock_simple.assert_called_once()

        store.close()

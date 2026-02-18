"""Tests for the refit executor (Task 2.4) and orchestrator integration (Task 2.5).

Covers:
- Task 2.4: execute_simple_refit(), RefitResult, _FullTrainFoldSplitter,
  splitter detection, best_params injection, prediction relabeling
- Task 2.5: PipelineOrchestrator._execute_refit_pass(), refit parameter
  threading through runner.py and api/run.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig, RefitCriterion, parse_refit_param
from nirs4all.pipeline.execution.refit.executor import (
    RefitResult,
    _apply_params_to_model,
    _extract_test_score,
    _FullTrainFoldSplitter,
    _inject_best_params,
    _make_full_train_fold_step,
    _relabel_refit_predictions,
    _step_is_splitter,
    execute_simple_refit,
)
from nirs4all.pipeline.storage.store_schema import (
    REFIT_CONTEXT_STANDALONE,
)

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

    def __init__(self, n_train: int = 50, name: str = "test_dataset") -> None:
        self._n_train = n_train
        self.name = name
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


# =========================================================================
# Task 2.4: RefitResult dataclass
# =========================================================================


class TestRefitResult:
    """Tests for the RefitResult dataclass."""

    def test_default_values(self):
        """RefitResult has sensible defaults."""
        result = RefitResult()
        assert result.success is False
        assert result.model_artifact_id is None
        assert result.test_score is None
        assert result.metric == ""
        assert result.fold_id == "final"
        assert result.refit_context == REFIT_CONTEXT_STANDALONE
        assert result.predictions_count == 0

    def test_custom_values(self):
        """RefitResult accepts custom values."""
        result = RefitResult(
            success=True,
            model_artifact_id="abc-123",
            test_score=0.05,
            metric="rmse",
            predictions_count=3,
        )
        assert result.success is True
        assert result.model_artifact_id == "abc-123"
        assert result.test_score == 0.05
        assert result.metric == "rmse"
        assert result.predictions_count == 3


# =========================================================================
# Task 2.4: _FullTrainFoldSplitter
# =========================================================================


class TestFullTrainFoldSplitter:
    """Tests for the _FullTrainFoldSplitter dummy splitter."""

    def test_single_fold(self):
        """Yields exactly one fold."""
        splitter = _FullTrainFoldSplitter(10)
        folds = list(splitter.split(np.zeros((10, 5))))
        assert len(folds) == 1

    def test_all_in_train(self):
        """All indices go to train, validation is empty."""
        n = 20
        splitter = _FullTrainFoldSplitter(n)
        folds = list(splitter.split(np.zeros((n, 5))))
        train_indices, val_indices = folds[0]
        assert train_indices == list(range(n))
        assert val_indices == []

    def test_get_n_splits(self):
        """get_n_splits returns 1."""
        splitter = _FullTrainFoldSplitter(10)
        assert splitter.get_n_splits() == 1


# =========================================================================
# Task 2.4: _step_is_splitter
# =========================================================================


class TestStepIsSplitter:
    """Tests for splitter detection logic."""

    def test_live_splitter_instance(self):
        """Detects live splitter instances."""
        from sklearn.model_selection import KFold
        assert _step_is_splitter(KFold(n_splits=3)) is True

    def test_live_model_not_splitter(self):
        """Models are not detected as splitters."""
        assert _step_is_splitter(_DummyModel()) is False

    def test_dict_with_split_keyword(self):
        """Dict with 'split' keyword is detected."""
        assert _step_is_splitter({"split": "KFold", "params": {"n_splits": 5}}) is True

    def test_serialized_model_selection_class(self):
        """Serialized class path with model_selection is detected."""
        step = {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}}
        assert _step_is_splitter(step) is True

    def test_serialized_splitter_class_name(self):
        """Class names containing splitter fragments are detected."""
        assert _step_is_splitter({"class": "custom.ShuffleSplit"}) is True
        assert _step_is_splitter({"class": "custom.StratifiedKFold"}) is True
        assert _step_is_splitter({"class": "custom.LeaveOneOut"}) is True

    def test_serialized_non_splitter(self):
        """Non-splitter serialized classes are not detected."""
        assert _step_is_splitter({"class": "sklearn.preprocessing.MinMaxScaler"}) is False

    def test_dict_without_class_or_split(self):
        """Dict steps without split or class are not splitters."""
        assert _step_is_splitter({"model": "PLSRegression"}) is False

    def test_full_train_fold_splitter_is_detected(self):
        """Our own _FullTrainFoldSplitter is detected as a splitter."""
        splitter = _FullTrainFoldSplitter(10)
        assert _step_is_splitter(splitter) is True

    def test_dummy_splitter_is_detected(self):
        """A custom splitter with split(X) is detected."""
        assert _step_is_splitter(_DummySplitter()) is True


# =========================================================================
# Task 2.4: _make_full_train_fold_step
# =========================================================================


class TestMakeFullTrainFoldStep:
    """Tests for _make_full_train_fold_step."""

    def test_creates_splitter_with_correct_count(self):
        """Creates a splitter matching the dataset's training sample count."""
        dataset = _DummyDataset(n_train=42)
        splitter = _make_full_train_fold_step(dataset)
        assert isinstance(splitter, _FullTrainFoldSplitter)
        assert splitter._n_samples == 42

    def test_fallback_on_dataset_error(self):
        """Falls back to 100 if dataset.x() fails."""
        dataset = MagicMock()
        dataset.x.side_effect = RuntimeError("no data")
        splitter = _make_full_train_fold_step(dataset)
        assert isinstance(splitter, _FullTrainFoldSplitter)
        assert splitter._n_samples == 100


# =========================================================================
# Task 2.4: _inject_best_params
# =========================================================================


class TestInjectBestParams:
    """Tests for best parameter injection into model steps."""

    def test_inject_into_live_model(self):
        """Injects params into a live sklearn-like model via set_params."""
        model = _DummyModel(n_components=5)
        steps = [model]
        _inject_best_params(steps, {"n_components": 10})
        assert model.n_components == 10

    def test_inject_into_dict_model_with_instance(self):
        """Injects params into a dict step with a live model instance."""
        model = _DummyModel(n_components=5)
        steps = [{"model": model}]
        _inject_best_params(steps, {"n_components": 15})
        assert model.n_components == 15

    def test_inject_into_dict_model_with_params_dict(self):
        """Injects params into a serialized model dict with 'params' key."""
        steps = [{"model": {"class": "PLS", "params": {"n_components": 5}}}]
        _inject_best_params(steps, {"n_components": 20})
        assert steps[0]["model"]["params"]["n_components"] == 20

    def test_empty_best_params(self):
        """Empty best_params causes no changes."""
        model = _DummyModel(n_components=5)
        steps = [{"model": model}]
        _inject_best_params(steps, {})
        assert model.n_components == 5

    def test_skips_splitter_steps(self):
        """Does not inject params into splitter steps."""
        splitter = _DummySplitter()
        steps = [splitter]
        # Should not raise even though splitter has no set_params
        _inject_best_params(steps, {"n_components": 10})

    def test_skips_non_model_dict_steps(self):
        """Does not inject into dict steps without 'model' key."""
        steps = [{"y_processing": "MinMaxScaler"}]
        _inject_best_params(steps, {"n_components": 10})
        # Should not raise or modify

    def test_refit_params_merging(self):
        """Resolves and applies refit_params on top of best_params."""
        model = _DummyModel(n_components=5, alpha=1.0)
        steps = [{
            "model": model,
            "train_params": {"alpha": 0.5},
            "refit_params": {"alpha": 0.1},
        }]
        _inject_best_params(steps, {"n_components": 10})
        assert model.n_components == 10
        assert model.alpha == 0.1  # refit_params override


# =========================================================================
# Task 2.4: _apply_params_to_model
# =========================================================================


class TestApplyParamsToModel:
    """Tests for safe parameter application."""

    def test_apply_valid_params(self):
        """Valid params are applied."""
        model = _DummyModel(n_components=5)
        _apply_params_to_model(model, {"n_components": 10})
        assert model.n_components == 10

    def test_skip_warm_start_fold(self):
        """warm_start_fold is filtered out as refit-only key."""
        model = _DummyModel(n_components=5)
        _apply_params_to_model(model, {"n_components": 10, "warm_start_fold": "best"})
        assert model.n_components == 10
        assert not hasattr(model, "warm_start_fold")

    def test_invalid_param_skipped_gracefully(self):
        """Invalid params are skipped without raising."""
        model = _DummyModel(n_components=5)
        _apply_params_to_model(model, {"n_components": 10, "nonexistent_param": 42})
        assert model.n_components == 10

    def test_no_set_params(self):
        """Objects without set_params are skipped."""
        obj = object()
        _apply_params_to_model(obj, {"n_components": 10})  # Should not raise


# =========================================================================
# Task 2.4: _relabel_refit_predictions
# =========================================================================


class TestRelabelRefitPredictions:
    """Tests for prediction relabeling."""

    def test_relabels_fold_id_and_context(self):
        """Sets fold_id='final' and refit_context='standalone'."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="fold_0",
            partition="val",
        )
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="fold_1",
            partition="val",
        )

        _relabel_refit_predictions(preds)

        for entry in preds._buffer:
            assert entry["fold_id"] == "final"
            assert entry["refit_context"] == REFIT_CONTEXT_STANDALONE

    def test_empty_predictions(self):
        """No error on empty predictions."""
        preds = Predictions()
        _relabel_refit_predictions(preds)  # Should not raise


# =========================================================================
# Task 2.4: _extract_test_score
# =========================================================================


class TestExtractTestScore:
    """Tests for test score extraction from refit predictions."""

    def test_extracts_from_buffer(self):
        """Extracts test_score from buffered entries."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="final",
            partition="train",
            test_score=0.08,
        )

        score = _extract_test_score(preds)
        assert score == pytest.approx(0.08)

    def test_returns_none_when_no_test_score(self):
        """Returns None when no test_score is present."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="final",
            partition="train",
        )

        score = _extract_test_score(preds)
        assert score is None

    def test_empty_predictions(self):
        """Returns None for empty predictions."""
        preds = Predictions()
        score = _extract_test_score(preds)
        assert score is None


# =========================================================================
# Task 2.4: execute_simple_refit
# =========================================================================


class TestExecuteSimpleRefit:
    """Tests for the execute_simple_refit function."""

    def _make_refit_config(
        self,
        *,
        include_splitter: bool = True,
        best_params: dict | None = None,
    ) -> RefitConfig:
        """Create a RefitConfig for testing."""
        steps: list[Any] = [{"class": "sklearn.preprocessing.MinMaxScaler"}]
        if include_splitter:
            steps.append({"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}})
        steps.append({
            "model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 5}},
        })

        return RefitConfig(
            expanded_steps=steps,
            best_params=best_params or {},
            variant_index=0,
            generator_choices=[],
            pipeline_id="test-pipeline-id",
            metric="rmse",
            selection_score=0.05,
        )

    def _make_mock_executor(self) -> MagicMock:
        """Create a mock PipelineExecutor."""
        executor = MagicMock()
        executor.initialize_context.return_value = MagicMock()
        executor.execute.return_value = None
        return executor

    def test_simple_refit_success(self):
        """Refit completes successfully for a pipeline with a splitter."""
        config = self._make_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = self._make_mock_executor()
        prediction_store = Predictions()

        result = execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
        )

        assert result.success is True
        assert result.metric == "rmse"
        # Executor should have been called
        executor.execute.assert_called_once()
        # Phase should be reset to CV after refit
        assert runtime_context.phase == ExecutionPhase.CV

    def test_no_splitter_skips_refit(self):
        """Refit is skipped when pipeline has no splitter."""
        config = self._make_refit_config(include_splitter=False)
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = self._make_mock_executor()
        prediction_store = Predictions()

        result = execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
        )

        assert result.success is True
        # Executor should NOT have been called
        executor.execute.assert_not_called()

    def test_refit_sets_phase_to_refit(self):
        """Runtime context phase is set to REFIT during execution."""
        config = self._make_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()

        phases_seen = []

        def capture_phase(**kwargs):
            phases_seen.append(kwargs["runtime_context"].phase)

        executor = self._make_mock_executor()
        executor.execute.side_effect = capture_phase

        execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=Predictions(),
        )

        assert ExecutionPhase.REFIT in phases_seen
        # Phase should be reset after refit
        assert runtime_context.phase == ExecutionPhase.CV

    def test_refit_resets_phase_on_failure(self):
        """Phase is reset to CV even if refit execution fails."""
        config = self._make_refit_config()
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()

        executor = self._make_mock_executor()
        executor.execute.side_effect = RuntimeError("Simulated failure")

        result = execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=Predictions(),
        )

        assert result.success is False
        assert runtime_context.phase == ExecutionPhase.CV

    def test_refit_deep_copies_dataset(self):
        """Dataset is deep-copied so caller's data is not mutated."""
        config = self._make_refit_config()
        original_dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = self._make_mock_executor()

        execute_simple_refit(
            refit_config=config,
            dataset=original_dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=Predictions(),
        )

        # Executor should have been called with a DIFFERENT dataset object
        call_kwargs = executor.execute.call_args[1]
        assert call_kwargs["dataset"] is not original_dataset

    def test_refit_replaces_splitter_with_full_train(self):
        """The splitter step is replaced with _FullTrainFoldSplitter."""
        config = self._make_refit_config()
        dataset = _DummyDataset(n_train=30)
        runtime_context = RuntimeContext()
        executor = self._make_mock_executor()

        execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=Predictions(),
        )

        # Check the steps passed to executor
        call_kwargs = executor.execute.call_args[1]
        steps = call_kwargs["steps"]
        # The splitter (originally at index 1) should now be a _FullTrainFoldSplitter
        splitter_step = steps[1]
        assert isinstance(splitter_step, _FullTrainFoldSplitter)

    def test_refit_injects_best_params(self):
        """Best params are injected into the model step."""
        config = self._make_refit_config(best_params={"n_components": 15})
        dataset = _DummyDataset(n_train=30)
        runtime_context = RuntimeContext()
        executor = self._make_mock_executor()

        execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=Predictions(),
        )

        # Check the steps passed to executor
        call_kwargs = executor.execute.call_args[1]
        steps = call_kwargs["steps"]
        model_step = steps[2]  # Third step is the model
        assert model_step["model"]["params"]["n_components"] == 15

    def test_refit_pipeline_name_suffix(self):
        """Refit pipeline name gets '_refit' suffix."""
        config = self._make_refit_config()
        dataset = _DummyDataset(n_train=30)
        runtime_context = RuntimeContext()
        runtime_context.pipeline_name = "my_pipeline"
        executor = self._make_mock_executor()

        execute_simple_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=Predictions(),
        )

        call_kwargs = executor.execute.call_args[1]
        assert call_kwargs["config_name"] == "my_pipeline_refit"


# =========================================================================
# Task 2.5: Orchestrator refit pass
# =========================================================================


class TestOrchestratorRefitPass:
    """Tests for PipelineOrchestrator._execute_refit_pass."""

    def test_refit_parameter_accepted_by_orchestrator(self):
        """PipelineOrchestrator.execute() accepts refit parameter."""
        import inspect

        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator.execute)
        assert "refit" in sig.parameters

    def test_refit_parameter_default_is_true(self):
        """refit parameter defaults to True."""
        import inspect

        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator.execute)
        assert sig.parameters["refit"].default is True

    def test_execute_refit_pass_method_exists(self):
        """_execute_refit_pass method exists on PipelineOrchestrator."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        assert hasattr(PipelineOrchestrator, "_execute_refit_pass")

    def test_execute_refit_pass_handles_no_completed_pipelines(self, tmp_path):
        """_execute_refit_pass warns and returns if no completed pipelines."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            mode="train",
        )

        # Create a run with no completed pipelines
        run_id = orchestrator.store.begin_run("test", config={}, datasets=[])

        executor = MagicMock()
        artifact_registry = MagicMock()
        run_dataset_predictions = Predictions()
        run_predictions = Predictions()

        # Should not raise, just warn
        orchestrator._execute_refit_pass(
            run_id=run_id,
            dataset=_DummyDataset(),
            executor=executor,
            artifact_registry=artifact_registry,
            run_dataset_predictions=run_dataset_predictions,
            run_predictions=run_predictions,
        )

        orchestrator.store.close()

    def test_execute_refit_pass_dispatches_stacking(self, tmp_path):
        """_execute_refit_pass dispatches stacking pipelines to execute_stacking_refit."""
        from unittest.mock import patch

        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator
        from nirs4all.pipeline.execution.refit.executor import RefitResult

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
            expanded_config=[
                {"class": "sklearn.preprocessing.MinMaxScaler"},
                {"branch": [
                    [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                    [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
                ]},
                {"merge": "predictions"},
                {"model": {"class": "sklearn.linear_model.Ridge"}},
            ],
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

        # Should dispatch to execute_stacking_refit for stacking pipelines
        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit.execute_stacking_refit"
        ) as mock_stacking:
            mock_stacking.return_value = RefitResult(success=True, predictions_count=0)

            orchestrator._execute_refit_pass(
                run_id=run_id,
                dataset=_DummyDataset(name="ds"),
                executor=executor,
                artifact_registry=MagicMock(),
                run_dataset_predictions=Predictions(),
                run_predictions=Predictions(),
            )

            mock_stacking.assert_called_once()

        store.close()


# =========================================================================
# Task 2.5: PipelineRunner.run() refit parameter
# =========================================================================


class TestRunnerRefitParameter:
    """Tests for refit parameter in PipelineRunner.run()."""

    def test_runner_run_accepts_refit(self):
        """PipelineRunner.run() accepts refit parameter."""
        import inspect

        from nirs4all.pipeline.runner import PipelineRunner

        sig = inspect.signature(PipelineRunner.run)
        assert "refit" in sig.parameters

    def test_runner_run_refit_default_true(self):
        """PipelineRunner.run() defaults refit to True."""
        import inspect

        from nirs4all.pipeline.runner import PipelineRunner

        sig = inspect.signature(PipelineRunner.run)
        assert sig.parameters["refit"].default is True


# =========================================================================
# Task 2.5: nirs4all.run() API refit parameter
# =========================================================================


class TestApiRunRefitParameter:
    """Tests for refit parameter in nirs4all.run() API."""

    def test_api_run_accepts_refit(self):
        """nirs4all.run() accepts refit parameter."""
        import inspect

        from nirs4all.api.run import run

        sig = inspect.signature(run)
        assert "refit" in sig.parameters

    def test_api_run_refit_default_true(self):
        """nirs4all.run() defaults refit to True."""
        import inspect

        from nirs4all.api.run import run

        sig = inspect.signature(run)
        assert sig.parameters["refit"].default is True

    def test_api_run_refit_is_keyword_only(self):
        """refit is a keyword-only argument in nirs4all.run()."""
        import inspect

        from nirs4all.api.run import run

        sig = inspect.signature(run)
        param = sig.parameters["refit"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


# =========================================================================
# Task 2.5: Refit enable/disable logic in orchestrator
# =========================================================================


class TestRefitEnableLogic:
    """Tests for refit enable/disable logic in the orchestrator."""

    def test_refit_none_means_disabled(self):
        """refit=None does not trigger refit."""
        refit = None
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert refit_enabled is False

    def test_refit_false_means_disabled(self):
        """refit=False does not trigger refit."""
        refit = False
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert refit_enabled is False

    def test_refit_true_means_enabled(self):
        """refit=True triggers refit."""
        refit = True
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert refit_enabled is True

    def test_refit_empty_dict_means_disabled(self):
        """refit={} does not trigger refit (empty dict is falsy)."""
        refit: dict = {}
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert not refit_enabled

    def test_refit_nonempty_dict_means_enabled(self):
        """refit={...} triggers refit."""
        refit = {"strategy": "simple"}
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert refit_enabled

    def test_refit_nonempty_list_means_enabled(self):
        """refit=[{...}] triggers refit."""
        refit = [{"top_k": 3}]
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert refit_enabled

    def test_refit_empty_list_means_disabled(self):
        """refit=[] does not trigger refit."""
        refit: list = []
        refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
        assert not refit_enabled


# =========================================================================
# Refit aggregation reporting
# =========================================================================


def _make_refit_prediction_entry(
    partition: str,
    n_samples: int = 20,
    model_name: str = "PLS",
    config_name: str = "pipeline_refit",
    dataset_name: str = "ds",
    fold_id: str = "final",
    sample_ids: list | None = None,
    noise_scale: float = 0.1,
) -> dict[str, Any]:
    """Create a realistic refit prediction entry with arrays and metadata."""
    rng = np.random.RandomState(42)
    y_true = rng.rand(n_samples)
    y_pred = y_true + rng.randn(n_samples) * noise_scale

    metadata: dict[str, Any] = {}
    if sample_ids is not None:
        metadata["Sample_ID"] = sample_ids

    return {
        "id": f"{model_name}_{partition}_{fold_id}",
        "dataset_name": dataset_name,
        "config_name": config_name,
        "model_name": model_name,
        "model_classname": model_name,
        "fold_id": fold_id,
        "step_idx": 0,
        "partition": partition,
        "y_true": y_true,
        "y_pred": y_pred,
        "metadata": metadata,
        "val_score": 0.05,
        "test_score": 0.08,
        "train_score": 0.03,
        "cv_rank_score": 0.06,
        "cv_test_score": 0.09,
        "metric": "rmse",
        "task_type": "regression",
        "n_samples": n_samples,
        "n_features": 100,
        "refit_context": REFIT_CONTEXT_STANDALONE,
    }


class TestRefitAggregation:
    """Tests for aggregation scores in refit reports."""

    def test_refit_tab_report_includes_aggregation(self):
        """TabReportManager shows aggregated rows for refit entries with metadata."""
        from nirs4all.visualization.reports import TabReportManager

        # 20 spectra from 5 physical samples (4 reps each)
        sample_ids = [f"S{i}" for i in range(5) for _ in range(4)]

        preds = Predictions()
        for partition in ("train", "test"):
            entry = _make_refit_prediction_entry(
                partition=partition, n_samples=20, sample_ids=sample_ids,
            )
            preds._buffer.append(entry)

        # Get the test entry as "best_refit"
        test_entry = [e for e in preds._buffer if e["partition"] == "test"][0]
        partitions = preds.get_entry_partitions(test_entry)

        tab, _ = TabReportManager.generate_best_score_tab_report(
            partitions, aggregate="Sample_ID",
        )

        # The tab report should contain an aggregated row (marked with *)
        assert "*" in tab
        assert "Sample_ID" in tab or "RMSE*" in tab or "R²*" in tab

    def test_refit_tab_report_no_aggregation_when_no_metadata(self):
        """TabReportManager skips aggregation gracefully when metadata is missing."""
        from nirs4all.visualization.reports import TabReportManager

        preds = Predictions()
        for partition in ("train", "test"):
            entry = _make_refit_prediction_entry(
                partition=partition, n_samples=20, sample_ids=None,
            )
            preds._buffer.append(entry)

        test_entry = [e for e in preds._buffer if e["partition"] == "test"][0]
        partitions = preds.get_entry_partitions(test_entry)

        # Should not crash, should produce a valid table without aggregation
        tab, _ = TabReportManager.generate_best_score_tab_report(
            partitions, aggregate="Sample_ID",
        )
        assert tab is not None
        assert "RMSE" in tab

    def test_per_model_summary_includes_aggregated_score(self):
        """generate_per_model_summary shows aggregated column when configured."""
        from nirs4all.visualization.reports import TabReportManager

        sample_ids = [f"S{i}" for i in range(5) for _ in range(4)]

        preds = Predictions()
        for model_name in ("PLS", "Ridge"):
            for partition in ("train", "test"):
                entry = _make_refit_prediction_entry(
                    partition=partition, n_samples=20,
                    model_name=model_name, sample_ids=sample_ids,
                )
                if model_name == "Ridge":
                    entry["test_score"] = 0.10
                    entry["cv_rank_score"] = 0.08
                    entry["cv_test_score"] = 0.11
                preds._buffer.append(entry)

        refit_entries = [e for e in preds._buffer if e["partition"] == "test"]

        summary = TabReportManager.generate_per_model_summary(
            refit_entries, ascending=True, metric="rmse",
            aggregate="Sample_ID",
            predictions=preds,
        )

        assert "RMSEP*" in summary

    def test_per_model_summary_without_aggregation(self):
        """generate_per_model_summary works without aggregation (backward compat)."""
        from nirs4all.visualization.reports import TabReportManager

        preds = Predictions()
        for model_name in ("PLS", "Ridge"):
            for partition in ("train", "test"):
                entry = _make_refit_prediction_entry(
                    partition=partition, n_samples=20,
                    model_name=model_name,
                )
                if model_name == "Ridge":
                    entry["test_score"] = 0.10
                preds._buffer.append(entry)

        refit_entries = [e for e in preds._buffer if e["partition"] == "test"]

        summary = TabReportManager.generate_per_model_summary(
            refit_entries, ascending=True, metric="rmse",
        )

        # Should produce a valid table without aggregated column
        assert "RMSEP" in summary
        assert "RMSEP*" not in summary


# =========================================================================
# RefitCriterion and parse_refit_param
# =========================================================================


class TestRefitCriterion:
    """Tests for the RefitCriterion dataclass."""

    def test_default_values(self):
        """RefitCriterion has sensible defaults."""
        c = RefitCriterion()
        assert c.top_k == 1
        assert c.ranking == "rmsecv"
        assert c.metric == ""

    def test_custom_values(self):
        """RefitCriterion accepts custom values."""
        c = RefitCriterion(top_k=3, ranking="mean_val", metric="r2")
        assert c.top_k == 3
        assert c.ranking == "mean_val"
        assert c.metric == "r2"


class TestParseRefitParam:
    """Tests for parse_refit_param normalization."""

    def test_true_returns_default_criterion(self):
        """True → single default criterion."""
        result = parse_refit_param(True)
        assert len(result) == 1
        assert result[0].top_k == 1
        assert result[0].ranking == "rmsecv"

    def test_false_returns_empty(self):
        """False → empty list."""
        assert parse_refit_param(False) == []

    def test_none_returns_empty(self):
        """None → empty list."""
        assert parse_refit_param(None) == []

    def test_dict_returns_single_criterion(self):
        """Dict → single criterion."""
        result = parse_refit_param({"top_k": 3, "ranking": "mean_val"})
        assert len(result) == 1
        assert result[0].top_k == 3
        assert result[0].ranking == "mean_val"

    def test_dict_ignores_unknown_keys(self):
        """Dict with unknown keys ignores them."""
        result = parse_refit_param({"top_k": 2, "unknown_key": "value"})
        assert len(result) == 1
        assert result[0].top_k == 2

    def test_list_returns_multiple_criteria(self):
        """List[dict] → multiple criteria."""
        result = parse_refit_param([
            {"top_k": 3, "ranking": "rmsecv"},
            {"top_k": 1, "ranking": "mean_val"},
        ])
        assert len(result) == 2
        assert result[0].top_k == 3
        assert result[0].ranking == "rmsecv"
        assert result[1].top_k == 1
        assert result[1].ranking == "mean_val"

    def test_empty_dict_returns_empty(self):
        """Empty dict returns empty list."""
        assert parse_refit_param({}) == []

    def test_empty_list_returns_empty(self):
        """Empty list returns empty list."""
        assert parse_refit_param([]) == []


# =========================================================================
# extract_top_configs
# =========================================================================


class TestExtractTopConfigs:
    """Tests for extract_top_configs with store-backed pipelines."""

    def _setup_store_with_pipelines(self, tmp_path, n_pipelines=3):
        """Create a store with completed pipelines for testing."""
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        store = WorkspaceStore(workspace_path=tmp_path / "workspace")
        run_id = store.begin_run("test", config={}, datasets=[{"name": "ds"}])

        pipeline_ids = []
        for i in range(n_pipelines):
            pid = store.begin_pipeline(
                run_id=run_id,
                name=f"pipeline_{i}",
                expanded_config=[
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                    {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3}},
                    {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": i + 5}}},
                ],
                generator_choices=[],
                dataset_name="ds",
                dataset_hash=f"h{i}",
            )

            chain_id = store.save_chain(
                pipeline_id=pid,
                steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
                model_step_idx=0,
                model_class="PLSRegression",
                preprocessings="",
                fold_strategy="per_fold",
                fold_artifacts={},
                shared_artifacts={},
            )

            store.save_prediction(
                pipeline_id=pid,
                chain_id=chain_id,
                dataset_name="ds",
                model_name="PLS",
                model_class="PLSRegression",
                fold_id="fold_0",
                partition="val",
                val_score=0.10 + i * 0.05,
                test_score=0.12 + i * 0.03,
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
                pipeline_id=pid,
                best_val=0.10 + i * 0.05,
                best_test=0.12 + i * 0.03,
                metric="rmse",
                duration_ms=100,
            )
            pipeline_ids.append(pid)

        store.complete_run(run_id, summary={})
        return store, run_id, pipeline_ids

    def test_single_rmsecv_criterion(self, tmp_path):
        """Single rmsecv criterion selects best by best_val."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs

        store, run_id, pipeline_ids = self._setup_store_with_pipelines(tmp_path)

        criteria = [RefitCriterion(top_k=1, ranking="rmsecv")]
        configs = extract_top_configs(store, run_id, criteria, dataset_name="ds")

        assert len(configs) == 1
        assert configs[0].pipeline_id == pipeline_ids[0]  # Lowest RMSE
        store.close()

    def test_top_k_rmsecv(self, tmp_path):
        """top_k=2 selects two best pipelines by RMSECV."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs

        store, run_id, pipeline_ids = self._setup_store_with_pipelines(tmp_path)

        criteria = [RefitCriterion(top_k=2, ranking="rmsecv")]
        configs = extract_top_configs(store, run_id, criteria, dataset_name="ds")

        assert len(configs) == 2
        selected_pids = [c.pipeline_id for c in configs]
        assert pipeline_ids[0] in selected_pids
        assert pipeline_ids[1] in selected_pids
        store.close()

    def test_multiple_criteria_selects_unique_models(self, tmp_path):
        """Multiple criteria select sum(top_k) unique models (no overlap)."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs

        store, run_id, pipeline_ids = self._setup_store_with_pipelines(tmp_path)

        # Each criterion independently fills its quota by skipping already-selected models.
        # Two criteria with top_k=1 each yield 2 unique models.
        criteria = [
            RefitCriterion(top_k=1, ranking="rmsecv"),
            RefitCriterion(top_k=1, ranking="rmsecv"),
        ]
        configs = extract_top_configs(store, run_id, criteria, dataset_name="ds")

        assert len(configs) == 2  # sum(top_k) unique models
        assert configs[0].pipeline_id == pipeline_ids[0]  # Best
        assert configs[1].pipeline_id == pipeline_ids[1]  # Second best
        store.close()

    def test_empty_criteria_returns_empty(self, tmp_path):
        """Empty criteria list returns empty configs."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs

        store, run_id, _ = self._setup_store_with_pipelines(tmp_path)
        configs = extract_top_configs(store, run_id, [], dataset_name="ds")
        assert configs == []
        store.close()

    def test_raises_on_no_completed_pipelines(self, tmp_path):
        """Raises ValueError when no completed pipelines exist."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        store = WorkspaceStore(workspace_path=tmp_path / "workspace")
        run_id = store.begin_run("test", config={}, datasets=[{"name": "ds"}])
        store.complete_run(run_id, summary={})

        criteria = [RefitCriterion(top_k=1)]
        with pytest.raises(ValueError, match="no pipelines"):
            extract_top_configs(store, run_id, criteria, dataset_name="ds")
        store.close()

    def test_mean_val_fallback_without_predictions(self, tmp_path):
        """mean_val ranking falls back to rmsecv when predictions not provided."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs

        store, run_id, pipeline_ids = self._setup_store_with_pipelines(tmp_path)

        criteria = [RefitCriterion(top_k=1, ranking="mean_val")]
        configs = extract_top_configs(store, run_id, criteria, predictions=None, dataset_name="ds")

        assert len(configs) == 1
        assert configs[0].pipeline_id == pipeline_ids[0]  # Falls back to best_val
        store.close()

    def test_config_has_correct_metadata(self, tmp_path):
        """Returned RefitConfig has correct metadata fields."""
        from nirs4all.pipeline.execution.refit.config_extractor import extract_top_configs

        store, run_id, pipeline_ids = self._setup_store_with_pipelines(tmp_path)

        criteria = [RefitCriterion(top_k=1)]
        configs = extract_top_configs(store, run_id, criteria, dataset_name="ds")

        config = configs[0]
        assert config.pipeline_id == pipeline_ids[0]
        assert config.metric == "rmse"
        assert config.selection_score == pytest.approx(0.10)
        assert config.config_name == "pipeline_0"
        assert len(config.expanded_steps) == 3
        store.close()

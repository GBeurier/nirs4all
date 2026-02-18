"""Tests for P4-A: Advanced Refit (Tasks 4.1, 4.2, 4.5).

Covers:
- Task 4.1: Nested stacking refit (recursive dispatch)
- Task 4.2: Separation branch + multi-source refit
- Task 4.5: Branches without merge (competing branch winner selection)
"""

from __future__ import annotations

import copy
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
    DEFAULT_MAX_STACKING_DEPTH,
    _branch_contains_stacking,
    _infer_ascending_for_metric,
    _select_winning_branch,
    execute_competing_branches_refit,
    execute_separation_refit,
    execute_stacking_refit,
)
from nirs4all.pipeline.storage.store_schema import REFIT_CONTEXT_STACKING


# =========================================================================
# Helpers
# =========================================================================


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

    def __deepcopy__(self, memo):
        new = _DummyDataset(self._n_train, name=self.name)
        return new


def _make_config(**overrides) -> RefitConfig:
    """Create a RefitConfig with sensible defaults."""
    defaults = {
        "expanded_steps": [],
        "best_params": {},
        "variant_index": 0,
        "generator_choices": [],
        "pipeline_id": "test-pipeline-id",
        "metric": "rmse",
        "selection_score": 0.05,
    }
    defaults.update(overrides)
    return RefitConfig(**defaults)


def _make_mock_executor() -> MagicMock:
    """Create a mock PipelineExecutor."""
    executor = MagicMock()
    executor.initialize_context.return_value = MagicMock()
    executor.execute.return_value = None
    return executor


# =========================================================================
# Task 4.1: Nested stacking refit
# =========================================================================


class TestBranchContainsStacking:
    """Tests for _branch_contains_stacking helper."""

    def test_simple_branch_no_stacking(self):
        """Simple branch without merge is not stacking."""
        branch_steps = [
            {"class": "sklearn.preprocessing.SNV"},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]
        assert _branch_contains_stacking(branch_steps) is False

    def test_branch_with_merge_predictions(self):
        """Branch containing merge: predictions is stacking."""
        branch_steps = [
            {"branch": [
                [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
            ]},
            {"merge": "predictions"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]
        assert _branch_contains_stacking(branch_steps) is True

    def test_branch_with_merge_features(self):
        """Branch with merge: features is NOT stacking."""
        branch_steps = [
            {"branch": [
                [{"class": "sklearn.preprocessing.SNV"}],
                [{"class": "sklearn.preprocessing.MSC"}],
            ]},
            {"merge": "features"},
        ]
        assert _branch_contains_stacking(branch_steps) is False

    def test_branch_with_merge_concat(self):
        """Branch with merge: concat is NOT stacking."""
        branch_steps = [
            {"branch": {"by_metadata": "site"}},
            {"merge": "concat"},
        ]
        assert _branch_contains_stacking(branch_steps) is False

    def test_branch_with_mixed_merge_containing_predictions(self):
        """Branch with mixed merge containing predictions is stacking."""
        branch_steps = [
            {"branch": [
                [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                [{"class": "sklearn.preprocessing.StandardScaler"}],
            ]},
            {"merge": {"predictions": [0], "features": [1]}},
        ]
        assert _branch_contains_stacking(branch_steps) is True

    def test_deeply_nested_stacking(self):
        """Detects stacking nested inside a nested branch."""
        branch_steps = [
            {"branch": [
                [
                    {"branch": [
                        [{"model": {"class": "sklearn.linear_model.Ridge"}}],
                        [{"model": {"class": "sklearn.svm.SVR"}}],
                    ]},
                    {"merge": "predictions"},
                    {"model": {"class": "sklearn.linear_model.Ridge"}},
                ],
            ]},
            {"merge": "features"},
        ]
        assert _branch_contains_stacking(branch_steps) is True

    def test_empty_branch(self):
        """Empty branch list is not stacking."""
        assert _branch_contains_stacking([]) is False

    def test_non_dict_steps(self):
        """Non-dict steps (live objects) are not stacking."""
        assert _branch_contains_stacking(["something", 42]) is False


class TestNestedStackingRefit:
    """Tests for nested stacking refit (Task 4.1)."""

    def test_depth_limit_constant(self):
        """DEFAULT_MAX_STACKING_DEPTH is 3."""
        assert DEFAULT_MAX_STACKING_DEPTH == 3

    def test_execute_stacking_refit_accepts_depth_params(self):
        """execute_stacking_refit accepts max_depth and _current_depth."""
        import inspect
        sig = inspect.signature(execute_stacking_refit)
        assert "max_depth" in sig.parameters
        assert "_current_depth" in sig.parameters
        assert sig.parameters["max_depth"].default == DEFAULT_MAX_STACKING_DEPTH
        assert sig.parameters["_current_depth"].default == 0

    def test_nested_stacking_pipeline_detected(self):
        """Nested stacking pipeline topology has max_stacking_depth > 1."""
        steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                # Branch 0: itself a stacking pipeline
                [
                    {"branch": [
                        [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                        [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
                    ]},
                    {"merge": "predictions"},
                    {"model": {"class": "sklearn.linear_model.Ridge"}},
                ],
                # Branch 1: simple model
                [{"model": {"class": "sklearn.linear_model.Ridge"}}],
            ]},
            {"merge": "predictions"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]
        topo = analyze_topology(steps)
        assert topo.has_stacking
        assert topo.max_stacking_depth == 2

    def test_recursive_call_with_nested_branch(self):
        """Stacking refit recursively handles nested stacking branches."""
        # Nested stacking: outer branch 0 has an inner stacking pipeline
        nested_steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                [
                    {"branch": [
                        [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                        [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
                    ]},
                    {"merge": "predictions"},
                    {"model": {"class": "sklearn.linear_model.Ridge"}},
                ],
                [{"model": {"class": "sklearn.linear_model.Ridge"}}],
            ]},
            {"merge": "predictions"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]

        config = _make_config(expanded_steps=nested_steps)
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        runtime_context.pipeline_name = "nested_test"
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = analyze_topology(nested_steps)

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

        # Should have made multiple executor calls:
        # Inner stacking: 2 base models + 1 meta = 3
        # Outer: 1 for branch 1 + 1 for outer meta = 2
        # Total: 5
        assert executor.execute.call_count >= 4
        assert runtime_context.phase == ExecutionPhase.CV  # Phase reset

    def test_depth_limit_prevents_infinite_recursion(self):
        """Depth limit prevents infinite recursion for nested stacking."""
        nested_steps = [
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                [
                    {"branch": [
                        [{"model": {"class": "sklearn.linear_model.Ridge"}}],
                        [{"model": {"class": "sklearn.svm.SVR"}}],
                    ]},
                    {"merge": "predictions"},
                    {"model": {"class": "sklearn.linear_model.Ridge"}},
                ],
                [{"model": {"class": "sklearn.linear_model.Ridge"}}],
            ]},
            {"merge": "predictions"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]

        config = _make_config(expanded_steps=nested_steps)
        dataset = _DummyDataset(n_train=50)
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()
        prediction_store = Predictions()
        topology = analyze_topology(nested_steps)

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

        # Set max_depth=1 so nested stacking falls back to simple refit
        result = execute_stacking_refit(
            refit_config=config,
            dataset=dataset,
            context=None,
            runtime_context=runtime_context,
            artifact_registry=None,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
            max_depth=1,
            _current_depth=0,
        )

        # With max_depth=1, nested branch should NOT recurse
        # It should fall through to simple base model refit
        assert runtime_context.phase == ExecutionPhase.CV


# =========================================================================
# Task 4.2: Separation branch refit
# =========================================================================


class TestSeparationRefit:
    """Tests for separation branch refit (Task 4.2)."""

    def test_separation_with_stacking_delegates_to_stacking(self):
        """Separation + stacking delegates to execute_stacking_refit."""
        topology = PipelineTopology(
            has_separation_branch=True,
            has_stacking=True,
            has_multi_source=True,
        )
        config = _make_config()
        dataset = _DummyDataset()
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()

        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit"
            ".execute_stacking_refit"
        ) as mock_stacking:
            mock_stacking.return_value = RefitResult(success=True)

            result = execute_separation_refit(
                refit_config=config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=None,
                executor=executor,
                prediction_store=Predictions(),
                topology=topology,
            )

            mock_stacking.assert_called_once()
            assert result.success is True

    def test_simple_separation_delegates_to_simple_refit(self):
        """Simple separation delegates to execute_simple_refit."""
        topology = PipelineTopology(
            has_separation_branch=True,
            has_stacking=False,
        )
        config = _make_config()
        dataset = _DummyDataset()
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()

        with patch(
            "nirs4all.pipeline.execution.refit.executor"
            ".execute_simple_refit"
        ) as mock_simple:
            mock_simple.return_value = RefitResult(success=True)

            result = execute_separation_refit(
                refit_config=config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=None,
                executor=executor,
                prediction_store=Predictions(),
                topology=topology,
            )

            mock_simple.assert_called_once()
            assert result.success is True

    def test_separation_topology_detected(self):
        """Separation branches are correctly detected in topology."""
        steps = [
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": {"by_metadata": "site"}},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
            {"merge": "concat"},
        ]
        topo = analyze_topology(steps)
        assert topo.has_separation_branch is True

    def test_multi_source_topology_detected(self):
        """Multi-source branches are correctly detected in topology."""
        steps = [
            {"branch": {"by_source": True, "steps": {
                "NIR": [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                "markers": [{"model": {"class": "sklearn.linear_model.Ridge"}}],
            }}},
            {"merge": "concat"},
        ]
        topo = analyze_topology(steps)
        assert topo.has_separation_branch is True
        assert topo.has_multi_source is True

    def test_separation_with_mixed_merge_delegates_to_stacking(self):
        """Separation + mixed merge delegates to stacking refit."""
        topology = PipelineTopology(
            has_separation_branch=True,
            has_mixed_merge=True,
        )
        config = _make_config()
        dataset = _DummyDataset()
        runtime_context = RuntimeContext()
        executor = _make_mock_executor()

        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit"
            ".execute_stacking_refit"
        ) as mock_stacking:
            mock_stacking.return_value = RefitResult(success=True)

            execute_separation_refit(
                refit_config=config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=None,
                executor=executor,
                prediction_store=Predictions(),
                topology=topology,
            )

            mock_stacking.assert_called_once()


# =========================================================================
# Task 4.5: Branches without merge (competing branches)
# =========================================================================


class TestInferAscending:
    """Tests for _infer_ascending_for_metric helper."""

    def test_rmse_is_ascending(self):
        assert _infer_ascending_for_metric("rmse") is True

    def test_r2_is_not_ascending(self):
        assert _infer_ascending_for_metric("r2") is False

    def test_accuracy_is_not_ascending(self):
        assert _infer_ascending_for_metric("accuracy") is False

    def test_empty_metric_defaults_to_ascending(self):
        assert _infer_ascending_for_metric("") is True


class TestSelectWinningBranch:
    """Tests for _select_winning_branch helper."""

    def test_returns_zero_when_store_is_none(self):
        config = _make_config()
        assert _select_winning_branch(config, None, [[], []]) == 0

    def test_returns_zero_when_no_pipeline_id(self):
        config = _make_config(pipeline_id="")
        assert _select_winning_branch(config, MagicMock(), [[], []]) == 0

    def test_returns_zero_when_no_predictions(self):
        import polars as pl
        store = MagicMock()
        store.query_predictions.return_value = pl.DataFrame()
        config = _make_config()
        assert _select_winning_branch(config, store, [[], []]) == 0

    def test_selects_branch_with_lowest_rmse(self):
        import polars as pl
        store = MagicMock()
        # Branch 0: val_score 0.15, branch 1: val_score 0.10 (better for RMSE)
        store.query_predictions.return_value = pl.DataFrame({
            "branch_id": [0, 0, 1, 1],
            "val_score": [0.15, 0.16, 0.10, 0.11],
        })
        config = _make_config(metric="rmse")
        result = _select_winning_branch(config, store, [[], []])
        assert result == 1  # Branch 1 has lower RMSE

    def test_selects_branch_with_highest_r2(self):
        import polars as pl
        store = MagicMock()
        # Branch 0: val_score 0.85, branch 1: val_score 0.92 (better for R2)
        store.query_predictions.return_value = pl.DataFrame({
            "branch_id": [0, 0, 1, 1],
            "val_score": [0.85, 0.84, 0.92, 0.91],
        })
        config = _make_config(metric="r2")
        result = _select_winning_branch(config, store, [[], []])
        assert result == 1  # Branch 1 has higher R2

    def test_clamps_to_valid_range(self):
        import polars as pl
        store = MagicMock()
        # branch_id=99 is out of range for a 2-branch pipeline
        store.query_predictions.return_value = pl.DataFrame({
            "branch_id": [99],
            "val_score": [0.10],
        })
        config = _make_config(metric="rmse")
        result = _select_winning_branch(config, store, [[], []])
        assert result == 0  # Clamped to 0


class TestCompetingBranchesRefit:
    """Tests for execute_competing_branches_refit (Task 4.5)."""

    def test_no_branch_falls_back_to_simple_refit(self):
        """No branch step → falls back to simple refit."""
        config = _make_config(expanded_steps=[
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ])
        topology = PipelineTopology()

        with patch(
            "nirs4all.pipeline.execution.refit.executor"
            ".execute_simple_refit"
        ) as mock_simple:
            mock_simple.return_value = RefitResult(success=True)

            result = execute_competing_branches_refit(
                refit_config=config,
                dataset=_DummyDataset(),
                context=None,
                runtime_context=RuntimeContext(),
                artifact_registry=None,
                executor=_make_mock_executor(),
                prediction_store=Predictions(),
                topology=topology,
            )

            mock_simple.assert_called_once()
            assert result.success is True

    def test_single_branch_falls_back_to_simple_refit(self):
        """Single branch → falls back to simple refit."""
        config = _make_config(expanded_steps=[
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"branch": [
                [{"model": {"class": "sklearn.linear_model.Ridge"}}],
            ]},
        ])
        topology = PipelineTopology(has_branches_without_merge=True)

        with patch(
            "nirs4all.pipeline.execution.refit.executor"
            ".execute_simple_refit"
        ) as mock_simple:
            mock_simple.return_value = RefitResult(success=True)

            result = execute_competing_branches_refit(
                refit_config=config,
                dataset=_DummyDataset(),
                context=None,
                runtime_context=RuntimeContext(),
                artifact_registry=None,
                executor=_make_mock_executor(),
                prediction_store=Predictions(),
                topology=topology,
            )

            mock_simple.assert_called_once()

    def test_winning_branch_selected_and_flattened(self):
        """Winning branch is selected and pipeline is flattened."""
        steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                [{"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}],
                [{"model": {"class": "sklearn.ensemble.RandomForestRegressor"}}],
            ]},
            # No merge step!
        ]
        config = _make_config(expanded_steps=steps)
        topology = PipelineTopology(has_branches_without_merge=True)

        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit"
            "._select_winning_branch",
            return_value=1,
        ), patch(
            "nirs4all.pipeline.execution.refit.executor"
            ".execute_simple_refit"
        ) as mock_simple:
            mock_simple.return_value = RefitResult(success=True)

            result = execute_competing_branches_refit(
                refit_config=config,
                dataset=_DummyDataset(),
                context=None,
                runtime_context=RuntimeContext(),
                artifact_registry=None,
                executor=_make_mock_executor(),
                prediction_store=Predictions(),
                topology=topology,
            )

            # Check that simple refit was called with flattened steps
            call_kwargs = mock_simple.call_args.kwargs
            flat_config = call_kwargs["refit_config"]
            # The flattened steps should contain:
            # pre-branch + winning branch steps (branch 1) + post-branch
            # Pre-branch: MinMaxScaler + KFold = 2
            # Winning branch: RandomForestRegressor = 1
            # Post-branch: none (no post-branch steps)
            assert len(flat_config.expanded_steps) == 3
            # The model in the flattened pipeline should be RF (branch 1)
            model_step = flat_config.expanded_steps[2]
            assert "model" in model_step
            assert "RandomForest" in model_step["model"]["class"]

    def test_merge_step_removed_from_flattened_pipeline(self):
        """Merge step is removed when flattening competing branches."""
        steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"branch": [
                [{"model": {"class": "sklearn.linear_model.Ridge"}}],
                [{"model": {"class": "sklearn.svm.SVR"}}],
            ]},
            {"merge": "concat"},  # Should be removed
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]
        config = _make_config(expanded_steps=steps)
        topology = PipelineTopology(has_branches_without_merge=True)

        with patch(
            "nirs4all.pipeline.execution.refit.stacking_refit"
            "._select_winning_branch",
            return_value=0,
        ), patch(
            "nirs4all.pipeline.execution.refit.executor"
            ".execute_simple_refit"
        ) as mock_simple:
            mock_simple.return_value = RefitResult(success=True)

            execute_competing_branches_refit(
                refit_config=config,
                dataset=_DummyDataset(),
                context=None,
                runtime_context=RuntimeContext(),
                artifact_registry=None,
                executor=_make_mock_executor(),
                prediction_store=Predictions(),
                topology=topology,
            )

            flat_config = mock_simple.call_args.kwargs["refit_config"]
            # No merge step should remain
            for step in flat_config.expanded_steps:
                if isinstance(step, dict):
                    assert "merge" not in step


# =========================================================================
# Orchestrator dispatch
# =========================================================================


class TestOrchestratorDispatch:
    """Tests for the orchestrator's refit strategy dispatch."""

    def test_branches_without_merge_dispatches_to_competing(self, tmp_path):
        """Branches without merge dispatch to competing branches refit."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            mode="train",
        )
        store = orchestrator.store

        # Build competing branch steps (no merge)
        steps = [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": [
                [{"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}}}],
                [{"model": {"class": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": 50}}}],
            ]},
            # No merge step
        ]

        run_id = store.begin_run("test", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="competing_pipe",
            expanded_config=steps,
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h1",
        )

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{
                "step_idx": 0,
                "operator_class": "PLS",
                "params": {},
                "artifact_id": None,
                "stateless": False,
            }],
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
            "nirs4all.pipeline.execution.refit.stacking_refit"
            ".execute_competing_branches_refit"
        ) as mock_competing:
            mock_competing.return_value = RefitResult(
                success=True, predictions_count=0
            )

            orchestrator._execute_refit_pass(
                run_id=run_id,
                dataset=_DummyDataset(name="ds"),
                executor=executor,
                artifact_registry=MagicMock(),
                run_dataset_predictions=Predictions(),
                run_predictions=Predictions(),
            )

            mock_competing.assert_called_once()

        store.close()

    def test_separation_branch_dispatches_to_separation(self, tmp_path):
        """Separation branches dispatch to separation refit."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            mode="train",
        )
        store = orchestrator.store

        # Separation branch steps
        steps = [
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"branch": {"by_metadata": "site"}},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}}},
            {"merge": "concat"},
        ]

        run_id = store.begin_run("test", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="separation_pipe",
            expanded_config=steps,
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h1",
        )

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{
                "step_idx": 0,
                "operator_class": "PLS",
                "params": {},
                "artifact_id": None,
                "stateless": False,
            }],
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
            "nirs4all.pipeline.execution.refit.stacking_refit"
            ".execute_separation_refit"
        ) as mock_separation:
            mock_separation.return_value = RefitResult(
                success=True, predictions_count=0
            )

            orchestrator._execute_refit_pass(
                run_id=run_id,
                dataset=_DummyDataset(name="ds"),
                executor=executor,
                artifact_registry=MagicMock(),
                run_dataset_predictions=Predictions(),
                run_predictions=Predictions(),
            )

            mock_separation.assert_called_once()

        store.close()

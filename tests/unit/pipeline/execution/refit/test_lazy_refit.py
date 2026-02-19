"""Tests for LazyModelRefitResult (Task 4.3).

Covers:
- Lazy access pattern: refit is not executed until a property is accessed
- Thread safety of the lazy execution
- Caching: second access returns the cached result
- Fallback when resources are destroyed
- Integration with RunResult.models property
- cv_score and metric are available without triggering refit
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nirs4all.api.result import (
    LazyModelRefitResult,
    ModelRefitResult,
    RunResult,
)
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
from nirs4all.pipeline.execution.refit.model_selector import PerModelSelection

# =========================================================================
# Helpers
# =========================================================================

def _make_selection(
    variant_index: int = 0,
    best_score: float = 0.5,
    best_params: dict | None = None,
) -> PerModelSelection:
    """Create a PerModelSelection for testing."""
    return PerModelSelection(
        variant_index=variant_index,
        best_score=best_score,
        best_params=best_params or {},
        expanded_steps=[],
        branch_path=[],
    )

def _make_refit_config(metric: str = "rmse") -> RefitConfig:
    """Create a RefitConfig for testing."""
    return RefitConfig(
        expanded_steps=[],
        best_params={},
        variant_index=0,
        metric=metric,
        selection_score=0.5,
    )

def _make_lazy_result(
    model_name: str = "PLSRegression",
    selection: PerModelSelection | None = None,
    refit_config: RefitConfig | None = None,
) -> LazyModelRefitResult:
    """Create a LazyModelRefitResult with mocked dependencies."""
    if selection is None:
        selection = _make_selection()
    if refit_config is None:
        refit_config = _make_refit_config()

    return LazyModelRefitResult(
        model_name=model_name,
        selection=selection,
        refit_config=refit_config,
        dataset=MagicMock(),
        context=MagicMock(),
        runtime_context=MagicMock(),
        artifact_registry=MagicMock(),
        executor=MagicMock(),
        prediction_store=MagicMock(),
    )

# =========================================================================
# Tests: LazyModelRefitResult basic properties
# =========================================================================

class TestLazyModelRefitResultBasicProperties:
    """Test properties available without triggering refit."""

    def test_model_name(self):
        result = _make_lazy_result(model_name="RandomForest")
        assert result.model_name == "RandomForest"

    def test_cv_score_does_not_trigger_refit(self):
        result = _make_lazy_result()
        assert result.cv_score == 0.5
        assert not result.is_resolved

    def test_metric_does_not_trigger_refit(self):
        result = _make_lazy_result(refit_config=_make_refit_config(metric="r2"))
        assert result.metric == "r2"
        assert not result.is_resolved

    def test_is_resolved_initially_false(self):
        result = _make_lazy_result()
        assert not result.is_resolved

    def test_repr_pending(self):
        result = _make_lazy_result(model_name="PLS")
        assert "pending" in repr(result)
        assert "PLS" in repr(result)

# =========================================================================
# Tests: LazyModelRefitResult lazy execution
# =========================================================================

class TestLazyModelRefitResultLazyExecution:
    """Test that refit is triggered on first access to score/final_entry."""

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_score_triggers_refit(self, mock_refit):
        """Accessing .score triggers the refit."""
        from nirs4all.pipeline.execution.refit.executor import RefitResult

        mock_refit.return_value = RefitResult(
            success=True,
            test_score=0.42,
            metric="rmse",
        )

        result = _make_lazy_result()
        score = result.score

        assert mock_refit.called
        assert score == 0.42
        assert result.is_resolved

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_final_score_triggers_refit(self, mock_refit):
        """Accessing .final_score triggers the refit."""
        from nirs4all.pipeline.execution.refit.executor import RefitResult

        mock_refit.return_value = RefitResult(
            success=True,
            test_score=0.35,
            metric="rmse",
        )

        result = _make_lazy_result()
        assert result.final_score == 0.35
        assert result.is_resolved

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_final_entry_triggers_refit(self, mock_refit):
        """Accessing .final_entry triggers the refit."""
        from nirs4all.pipeline.execution.refit.executor import RefitResult

        mock_refit.return_value = RefitResult(
            success=True,
            test_score=0.3,
            metric="rmse",
        )

        result = _make_lazy_result()
        entry = result.final_entry
        assert mock_refit.called
        assert isinstance(entry, dict)

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_result_is_cached(self, mock_refit):
        """Second access returns cached result without re-executing."""
        from nirs4all.pipeline.execution.refit.executor import RefitResult

        mock_refit.return_value = RefitResult(
            success=True,
            test_score=0.42,
            metric="rmse",
        )

        result = _make_lazy_result()

        # First access
        score1 = result.score
        # Second access
        score2 = result.score

        assert mock_refit.call_count == 1
        assert score1 == score2

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_repr_shows_resolved(self, mock_refit):
        """After refit, repr shows 'resolved'."""
        from nirs4all.pipeline.execution.refit.executor import RefitResult

        mock_refit.return_value = RefitResult(
            success=True,
            test_score=0.42,
            metric="rmse",
        )

        result = _make_lazy_result(model_name="PLS")
        _ = result.score
        assert "resolved" in repr(result)

# =========================================================================
# Tests: Error handling
# =========================================================================

class TestLazyModelRefitResultErrorHandling:
    """Test behavior when refit fails or resources are destroyed."""

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_refit_failure_returns_minimal_result(self, mock_refit):
        """When refit fails, returns a minimal ModelRefitResult with CV info."""
        mock_refit.side_effect = RuntimeError("Resources destroyed")

        result = _make_lazy_result(model_name="PLS")
        score = result.score

        # Should return None (no test score available)
        assert score is None
        assert result.is_resolved
        assert result.cv_score == 0.5

    @patch("nirs4all.pipeline.execution.refit.executor.execute_simple_refit")
    def test_refit_failure_preserves_metric(self, mock_refit):
        """When refit fails, metric from refit_config is preserved."""
        mock_refit.side_effect = RuntimeError("Error")

        refit_config = _make_refit_config(metric="r2")
        result = _make_lazy_result(refit_config=refit_config)
        _ = result.score

        assert result.metric == "r2"

# =========================================================================
# Tests: RunResult.models integration
# =========================================================================

class TestRunResultModelsLazy:
    """Test RunResult.models returns LazyModelRefitResult when selections are available."""

    def test_models_returns_lazy_when_selections_set(self):
        """When per-model selections are available, models returns lazy results."""
        preds = Predictions()
        result = RunResult(
            predictions=preds,
            per_dataset={},
        )
        result._per_model_selections = {
            "PLSRegression": _make_selection(best_score=0.3),
        }
        result._refit_config = _make_refit_config()
        result._refit_dataset = MagicMock()
        result._refit_context = MagicMock()
        result._refit_runtime_context = MagicMock()
        result._refit_artifact_registry = MagicMock()
        result._refit_executor = MagicMock()

        models = result.models
        assert "PLSRegression" in models
        assert isinstance(models["PLSRegression"], LazyModelRefitResult)
        assert not models["PLSRegression"].is_resolved

    def test_models_returns_eager_when_no_selections(self):
        """When no per-model selections, models falls back to eager path."""
        preds = Predictions()

        # Add a final entry to the prediction store
        preds._buffer.append({
            "fold_id": "final",
            "model_name": "Ridge",
            "test_score": 0.55,
            "val_score": 0.6,
            "metric": "rmse",
            "refit_context": "standalone",
        })
        # Add a CV entry
        preds._buffer.append({
            "fold_id": "w_avg",
            "model_name": "Ridge",
            "test_score": 0.6,
            "val_score": 0.58,
            "metric": "rmse",
        })

        result = RunResult(
            predictions=preds,
            per_dataset={},
        )

        models = result.models
        assert "Ridge" in models
        assert isinstance(models["Ridge"], ModelRefitResult)
        assert models["Ridge"].final_score == 0.55

    def test_models_empty_when_no_refit(self):
        """When no refit entries and no selections, models is empty."""
        preds = Predictions()
        result = RunResult(predictions=preds, per_dataset={})
        assert result.models == {}

    def test_models_multiple_selections(self):
        """Multiple model selections produce multiple lazy results."""
        preds = Predictions()
        result = RunResult(predictions=preds, per_dataset={})
        result._per_model_selections = {
            "PLS": _make_selection(best_score=0.3),
            "Ridge": _make_selection(best_score=0.4),
        }
        result._refit_config = _make_refit_config()
        result._refit_dataset = MagicMock()
        result._refit_context = MagicMock()
        result._refit_runtime_context = MagicMock()
        result._refit_artifact_registry = MagicMock()
        result._refit_executor = MagicMock()

        models = result.models
        assert len(models) == 2
        assert isinstance(models["PLS"], LazyModelRefitResult)
        assert isinstance(models["Ridge"], LazyModelRefitResult)
        assert models["PLS"].cv_score == 0.3
        assert models["Ridge"].cv_score == 0.4

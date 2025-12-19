"""
Unit tests for PredictionAnalyzer default_aggregate parameter.

Tests Phase 4 of the aggregation feature: Visualization Defaults.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer


class TestDefaultAggregateParameter:
    """Test default_aggregate parameter in PredictionAnalyzer."""

    @pytest.fixture
    def mock_predictions(self):
        """Create a mock Predictions object."""
        predictions = Mock(spec=Predictions)
        predictions.num_predictions = 10
        predictions.get_datasets.return_value = ['test_dataset']
        predictions.get_unique_values.return_value = ['classification']
        predictions.top.return_value = []
        predictions.clear_caches = Mock()
        predictions.get_cache_stats.return_value = {}
        return predictions

    def test_default_aggregate_none_by_default(self, mock_predictions):
        """Test that default_aggregate is None when not specified."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer.default_aggregate is None

    def test_default_aggregate_set_via_constructor(self, mock_predictions):
        """Test setting default_aggregate via constructor."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate='sample_id')
        assert analyzer.default_aggregate == 'sample_id'

    def test_default_aggregate_y_value(self, mock_predictions):
        """Test setting default_aggregate to 'y'."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate='y')
        assert analyzer.default_aggregate == 'y'


class TestResolveAggregate:
    """Test _resolve_aggregate helper method."""

    @pytest.fixture
    def mock_predictions(self):
        """Create a mock Predictions object."""
        predictions = Mock(spec=Predictions)
        predictions.num_predictions = 10
        predictions.get_datasets.return_value = ['test_dataset']
        predictions.clear_caches = Mock()
        predictions.get_cache_stats.return_value = {}
        return predictions

    def test_resolve_aggregate_returns_default_when_none(self, mock_predictions):
        """Test that _resolve_aggregate returns default when aggregate is None."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate='sample_id')
        result = analyzer._resolve_aggregate(None)
        assert result == 'sample_id'

    def test_resolve_aggregate_returns_explicit_value(self, mock_predictions):
        """Test that _resolve_aggregate returns explicit value over default."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate='sample_id')
        result = analyzer._resolve_aggregate('batch_id')
        assert result == 'batch_id'

    def test_resolve_aggregate_empty_string_disables(self, mock_predictions):
        """Test that empty string disables aggregation even with default set."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate='sample_id')
        result = analyzer._resolve_aggregate('')
        assert result is None

    def test_resolve_aggregate_none_when_no_default(self, mock_predictions):
        """Test that _resolve_aggregate returns None when no default is set."""
        analyzer = PredictionAnalyzer(mock_predictions)
        result = analyzer._resolve_aggregate(None)
        assert result is None

    def test_resolve_aggregate_explicit_none_uses_default(self, mock_predictions):
        """Test that explicitly passing None uses default."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate='ID')
        result = analyzer._resolve_aggregate(None)
        assert result == 'ID'


class TestVisualizationMethodsUseDefaultAggregate:
    """Test that visualization methods use default_aggregate."""

    @pytest.fixture
    def analyzer_with_default(self):
        """Create analyzer with default_aggregate and mocked dependencies."""
        predictions = Mock(spec=Predictions)
        predictions.num_predictions = 10
        predictions.get_datasets.return_value = ['dataset1']
        predictions.clear_caches = Mock()
        predictions.get_cache_stats.return_value = {}
        predictions.top.return_value = []
        predictions.get_unique_values.return_value = []

        analyzer = PredictionAnalyzer(predictions, default_aggregate='sample_id')
        return analyzer

    @patch('nirs4all.visualization.predictions.TopKComparisonChart')
    def test_plot_top_k_uses_default_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that plot_top_k uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()  # Mock figure
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_top_k(k=5)

        # Check that render was called with effective_aggregate
        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') == 'sample_id'

    @patch('nirs4all.visualization.predictions.TopKComparisonChart')
    def test_plot_top_k_override_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that explicit aggregate overrides default."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_top_k(k=5, aggregate='other_id')

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') == 'other_id'

    @patch('nirs4all.visualization.predictions.TopKComparisonChart')
    def test_plot_top_k_disable_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that empty string disables aggregation."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_top_k(k=5, aggregate='')

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') is None

    @patch('nirs4all.visualization.predictions.ScoreHistogramChart')
    def test_plot_histogram_uses_default_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that plot_histogram uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_histogram()

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') == 'sample_id'

    @patch('nirs4all.visualization.predictions.HeatmapChart')
    def test_plot_heatmap_uses_default_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that plot_heatmap uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_heatmap('x_var', 'y_var')

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') == 'sample_id'

    @patch('nirs4all.visualization.predictions.CandlestickChart')
    def test_plot_candlestick_uses_default_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that plot_candlestick uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_candlestick('model_name')

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') == 'sample_id'


class TestPipelineRunnerLastAggregate:
    """Test PipelineRunner.last_aggregate property."""

    def test_last_aggregate_none_before_run(self):
        """Test that last_aggregate is None before any run."""
        from nirs4all.pipeline import PipelineRunner

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        assert runner.last_aggregate is None

    def test_last_aggregate_none_when_not_set(self):
        """Test that last_aggregate is None when dataset has no aggregate."""
        from nirs4all.pipeline import PipelineRunner

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        # Before running, _last_aggregate_column should be None
        assert runner._last_aggregate_column is None


class TestOrchestratorAggregateTracking:
    """Test that orchestrator tracks last_aggregate_column."""

    def test_orchestrator_has_last_aggregate_attribute(self):
        """Test that orchestrator has last_aggregate_column attribute."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            workspace_path="/tmp/test",
            verbose=0,
            save_artifacts=False,
            save_charts=False
        )

        assert hasattr(orchestrator, 'last_aggregate_column')
        assert orchestrator.last_aggregate_column is None


class TestConfusionMatrixAggregate:
    """Test confusion matrix with default aggregate."""

    @pytest.fixture
    def analyzer_with_default(self):
        """Create analyzer with default_aggregate."""
        predictions = Mock(spec=Predictions)
        predictions.num_predictions = 10
        predictions.get_datasets.return_value = ['dataset1']
        predictions.clear_caches = Mock()
        predictions.get_cache_stats.return_value = {}
        predictions.top.return_value = []
        predictions.get_unique_values.return_value = ['classification']

        return PredictionAnalyzer(predictions, default_aggregate='sample_id')

    @patch('nirs4all.visualization.predictions.ConfusionMatrixChart')
    def test_plot_confusion_matrix_uses_default_aggregate(self, mock_chart_class, analyzer_with_default):
        """Test that plot_confusion_matrix uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer_with_default.plot_confusion_matrix(k=3)

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get('aggregate') == 'sample_id'

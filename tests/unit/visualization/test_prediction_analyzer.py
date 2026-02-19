"""
Unit tests for PredictionAnalyzer.

Tests cover:
- Constructor initialization and parameter handling
- Cache management (clear_cache, get_cache_stats)
- Resolve helpers (_resolve_aggregate, _resolve_aggregate_method, _resolve_aggregate_exclude_outliers)
- Plot methods dispatch to correct chart classes (top_k, histogram, heatmap, candlestick, confusion_matrix)
- get_cached_predictions caching behavior
"""

from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.charts import ChartConfig
from nirs4all.visualization.predictions import PredictionAnalyzer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predictions():
    """Basic mock Predictions object."""
    preds = Mock(spec=Predictions)
    preds.num_predictions = 5
    preds.get_datasets.return_value = ["dataset_a"]
    preds.get_unique_values.return_value = ["regression"]
    preds.top.return_value = []
    return preds

@pytest.fixture
def analyzer(mock_predictions):
    """Default PredictionAnalyzer with no default_aggregate."""
    return PredictionAnalyzer(mock_predictions)

@pytest.fixture
def analyzer_with_default(mock_predictions):
    """PredictionAnalyzer with default_aggregate='sample_id'."""
    return PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")

# ---------------------------------------------------------------------------
# TestPredictionAnalyzerInit
# ---------------------------------------------------------------------------

class TestPredictionAnalyzerInit:
    """Test PredictionAnalyzer initialization."""

    def test_default_aggregate_none_by_default(self, mock_predictions):
        """Test that default_aggregate is None when not specified."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer.default_aggregate is None

    def test_default_aggregate_set(self, mock_predictions):
        """Test setting default_aggregate via constructor."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        assert analyzer.default_aggregate == "sample_id"

    def test_default_aggregate_method_none(self, mock_predictions):
        """Test that default_aggregate_method is None by default."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer.default_aggregate_method is None

    def test_default_aggregate_method_set(self, mock_predictions):
        """Test setting default_aggregate_method."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_method="median")
        assert analyzer.default_aggregate_method == "median"

    def test_default_aggregate_exclude_outliers_false(self, mock_predictions):
        """Test that default_aggregate_exclude_outliers is False by default."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer.default_aggregate_exclude_outliers is False

    def test_default_aggregate_exclude_outliers_set(self, mock_predictions):
        """Test setting default_aggregate_exclude_outliers."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_exclude_outliers=True)
        assert analyzer.default_aggregate_exclude_outliers is True

    def test_predictions_stored(self, mock_predictions):
        """Test that predictions object is stored."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer.predictions is mock_predictions

    def test_default_config_created(self, mock_predictions):
        """Test that a default ChartConfig is created when none provided."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert isinstance(analyzer.config, ChartConfig)

    def test_custom_config_stored(self, mock_predictions):
        """Test that a custom ChartConfig is stored."""
        config = ChartConfig()
        analyzer = PredictionAnalyzer(mock_predictions, config=config)
        assert analyzer.config is config

    def test_dataset_name_override(self, mock_predictions):
        """Test dataset_name_override is stored."""
        analyzer = PredictionAnalyzer(mock_predictions, dataset_name_override="my_dataset")
        assert analyzer.dataset_name_override == "my_dataset"

    def test_output_dir_default(self, mock_predictions):
        """Test that output_dir has a default value."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer.output_dir is not None
        assert isinstance(analyzer.output_dir, str)

    def test_output_dir_custom(self, mock_predictions):
        """Test setting a custom output_dir."""
        analyzer = PredictionAnalyzer(mock_predictions, output_dir="/tmp/my_figs")
        assert analyzer.output_dir == "/tmp/my_figs"

# ---------------------------------------------------------------------------
# TestResolveAggregate
# ---------------------------------------------------------------------------

class TestResolveAggregate:
    """Test _resolve_aggregate helper method."""

    def test_returns_default_when_none_given(self, mock_predictions):
        """Test that _resolve_aggregate returns default when None is given."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        assert analyzer._resolve_aggregate(None) == "sample_id"

    def test_returns_explicit_value_over_default(self, mock_predictions):
        """Test that explicit value overrides default."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        assert analyzer._resolve_aggregate("batch_id") == "batch_id"

    def test_empty_string_disables_aggregation(self, mock_predictions):
        """Test that empty string disables aggregation even with default set."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        assert analyzer._resolve_aggregate("") is None

    def test_none_when_no_default(self, mock_predictions):
        """Test returns None when no default is set and None is given."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer._resolve_aggregate(None) is None

    def test_explicit_none_uses_default(self, mock_predictions):
        """Test that explicitly passing None uses default."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="ID")
        assert analyzer._resolve_aggregate(None) == "ID"

# ---------------------------------------------------------------------------
# TestResolveAggregateMethod
# ---------------------------------------------------------------------------

class TestResolveAggregateMethod:
    """Test _resolve_aggregate_method helper method."""

    def test_returns_default_when_none(self, mock_predictions):
        """Test that default method is used when None is given."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_method="median")
        assert analyzer._resolve_aggregate_method(None) == "median"

    def test_returns_explicit_value(self, mock_predictions):
        """Test that explicit method overrides default."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_method="median")
        assert analyzer._resolve_aggregate_method("mean") == "mean"

    def test_returns_none_when_no_default(self, mock_predictions):
        """Test returns None when no default and None given."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer._resolve_aggregate_method(None) is None

# ---------------------------------------------------------------------------
# TestResolveAggregateExcludeOutliers
# ---------------------------------------------------------------------------

class TestResolveAggregateExcludeOutliers:
    """Test _resolve_aggregate_exclude_outliers helper method."""

    def test_returns_default_when_none(self, mock_predictions):
        """Test that default exclude_outliers is used when None is given."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_exclude_outliers=True)
        assert analyzer._resolve_aggregate_exclude_outliers(None) is True

    def test_returns_explicit_value_false(self, mock_predictions):
        """Test that explicit False overrides default True."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_exclude_outliers=True)
        assert analyzer._resolve_aggregate_exclude_outliers(False) is False

    def test_returns_explicit_value_true(self, mock_predictions):
        """Test that explicit True overrides default False."""
        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_exclude_outliers=False)
        assert analyzer._resolve_aggregate_exclude_outliers(True) is True

    def test_default_false(self, mock_predictions):
        """Test that default is False when not specified."""
        analyzer = PredictionAnalyzer(mock_predictions)
        assert analyzer._resolve_aggregate_exclude_outliers(None) is False

# ---------------------------------------------------------------------------
# TestCacheManagement
# ---------------------------------------------------------------------------

class TestCacheManagement:
    """Test cache management methods."""

    def test_get_cache_stats_returns_dict(self, analyzer):
        """Test that get_cache_stats returns a dictionary."""
        stats = analyzer.get_cache_stats()
        assert isinstance(stats, dict)
        assert "analyzer_cache" in stats

    def test_clear_cache_works(self, analyzer):
        """Test that clear_cache does not raise."""
        analyzer.clear_cache()  # Should not raise

    def test_clear_cache_calls_predictions_clear(self, mock_predictions):
        """Test that clear_cache calls predictions.clear_caches if available."""
        mock_predictions.clear_caches = Mock()
        analyzer = PredictionAnalyzer(mock_predictions)
        analyzer.clear_cache()
        mock_predictions.clear_caches.assert_called_once()

    def test_clear_cache_handles_no_clear_caches_on_predictions(self, mock_predictions):
        """Test that clear_cache works even if predictions has no clear_caches."""
        # Remove clear_caches from mock_predictions spec
        if hasattr(mock_predictions, 'clear_caches'):
            del mock_predictions.clear_caches
        analyzer = PredictionAnalyzer(mock_predictions)
        analyzer.clear_cache()  # Should not raise

# ---------------------------------------------------------------------------
# TestPlotTopK
# ---------------------------------------------------------------------------

class TestPlotTopK:
    """Test plot_top_k method dispatches to TopKComparisonChart."""

    @patch("nirs4all.visualization.predictions.TopKComparisonChart")
    def test_plot_top_k_calls_chart(self, mock_chart_class, mock_predictions):
        """Test that plot_top_k creates and renders TopKComparisonChart."""
        mock_chart = Mock()
        mock_figure = Mock()
        mock_chart.render.return_value = mock_figure
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions)
        result = analyzer.plot_top_k(k=5)

        mock_chart.render.assert_called_once()
        assert result is mock_figure

    @patch("nirs4all.visualization.predictions.TopKComparisonChart")
    def test_plot_top_k_uses_default_aggregate(self, mock_chart_class, mock_predictions):
        """Test that plot_top_k uses default_aggregate when no aggregate given."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_top_k(k=5)

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") == "sample_id"

    @patch("nirs4all.visualization.predictions.TopKComparisonChart")
    def test_plot_top_k_explicit_aggregate_overrides_default(self, mock_chart_class, mock_predictions):
        """Test that explicit aggregate overrides default."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_top_k(k=5, aggregate="batch_id")

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") == "batch_id"

    @patch("nirs4all.visualization.predictions.TopKComparisonChart")
    def test_plot_top_k_empty_string_disables_aggregate(self, mock_chart_class, mock_predictions):
        """Test that empty string disables aggregate in plot_top_k."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_top_k(k=5, aggregate="")

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") is None

# ---------------------------------------------------------------------------
# TestPlotHistogram
# ---------------------------------------------------------------------------

class TestPlotHistogram:
    """Test plot_histogram method dispatches to ScoreHistogramChart."""

    @patch("nirs4all.visualization.predictions.ScoreHistogramChart")
    def test_plot_histogram_calls_chart(self, mock_chart_class, mock_predictions):
        """Test that plot_histogram creates and renders ScoreHistogramChart."""
        mock_chart = Mock()
        mock_figure = Mock()
        mock_chart.render.return_value = mock_figure
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions)
        result = analyzer.plot_histogram()

        mock_chart.render.assert_called_once()
        assert result is mock_figure

    @patch("nirs4all.visualization.predictions.ScoreHistogramChart")
    def test_plot_histogram_uses_default_aggregate(self, mock_chart_class, mock_predictions):
        """Test that plot_histogram uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_histogram()

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") == "sample_id"

# ---------------------------------------------------------------------------
# TestPlotHeatmap
# ---------------------------------------------------------------------------

class TestPlotHeatmap:
    """Test plot_heatmap method dispatches to HeatmapChart."""

    @patch("nirs4all.visualization.predictions.HeatmapChart")
    def test_plot_heatmap_calls_chart(self, mock_chart_class, mock_predictions):
        """Test that plot_heatmap creates and renders HeatmapChart."""
        mock_chart = Mock()
        mock_figure = Mock()
        mock_chart.render.return_value = mock_figure
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions)
        result = analyzer.plot_heatmap("x_var", "y_var")

        mock_chart.render.assert_called_once()
        assert result is mock_figure

    @patch("nirs4all.visualization.predictions.HeatmapChart")
    def test_plot_heatmap_uses_default_aggregate(self, mock_chart_class, mock_predictions):
        """Test that plot_heatmap uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_heatmap("x_var", "y_var")

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") == "sample_id"

# ---------------------------------------------------------------------------
# TestPlotCandlestick
# ---------------------------------------------------------------------------

class TestPlotCandlestick:
    """Test plot_candlestick method dispatches to CandlestickChart."""

    @patch("nirs4all.visualization.predictions.CandlestickChart")
    def test_plot_candlestick_calls_chart(self, mock_chart_class, mock_predictions):
        """Test that plot_candlestick creates and renders CandlestickChart."""
        mock_chart = Mock()
        mock_figure = Mock()
        mock_chart.render.return_value = mock_figure
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions)
        result = analyzer.plot_candlestick("model_name")

        mock_chart.render.assert_called_once()
        assert result is mock_figure

    @patch("nirs4all.visualization.predictions.CandlestickChart")
    def test_plot_candlestick_uses_default_aggregate(self, mock_chart_class, mock_predictions):
        """Test that plot_candlestick uses default_aggregate."""
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_candlestick("model_name")

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") == "sample_id"

# ---------------------------------------------------------------------------
# TestPlotConfusionMatrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    """Test plot_confusion_matrix method dispatches to ConfusionMatrixChart."""

    @patch("nirs4all.visualization.predictions.ConfusionMatrixChart")
    def test_plot_confusion_matrix_calls_chart(self, mock_chart_class, mock_predictions):
        """Test that plot_confusion_matrix creates and renders ConfusionMatrixChart."""
        mock_predictions.get_unique_values.return_value = ["classification"]
        mock_chart = Mock()
        mock_figure = Mock()
        mock_chart.render.return_value = mock_figure
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions)
        result = analyzer.plot_confusion_matrix(k=3)

        mock_chart.render.assert_called_once()
        assert result is mock_figure

    @patch("nirs4all.visualization.predictions.ConfusionMatrixChart")
    def test_plot_confusion_matrix_uses_default_aggregate(self, mock_chart_class, mock_predictions):
        """Test that plot_confusion_matrix uses default_aggregate."""
        mock_predictions.get_unique_values.return_value = ["classification"]
        mock_chart = Mock()
        mock_chart.render.return_value = Mock()
        mock_chart_class.return_value = mock_chart

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate="sample_id")
        analyzer.plot_confusion_matrix(k=3)

        call_kwargs = mock_chart.render.call_args[1]
        assert call_kwargs.get("aggregate") == "sample_id"

# ---------------------------------------------------------------------------
# TestGetCachedPredictions
# ---------------------------------------------------------------------------

class TestGetCachedPredictions:
    """Test get_cached_predictions caching behavior."""

    def test_first_call_queries_predictions(self, mock_predictions):
        """Test that the first call fetches from predictions."""
        mock_predictions.top.return_value = []

        analyzer = PredictionAnalyzer(mock_predictions)
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")

        mock_predictions.top.assert_called_once()

    def test_second_call_uses_cache(self, mock_predictions):
        """Test that the second call with same params uses cache."""
        mock_predictions.top.return_value = []

        analyzer = PredictionAnalyzer(mock_predictions)
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")

        # top() should only be called once (second call hits cache)
        assert mock_predictions.top.call_count == 1

    def test_different_params_different_cache_entries(self, mock_predictions):
        """Test that different params create different cache entries."""
        mock_predictions.top.return_value = []

        analyzer = PredictionAnalyzer(mock_predictions)
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")
        analyzer.get_cached_predictions(n=5, rank_metric="r2")

        # top() should be called twice (different rank_metrics)
        assert mock_predictions.top.call_count == 2

    def test_clear_cache_invalidates_predictions(self, mock_predictions):
        """Test that clear_cache forces re-fetch on next call."""
        mock_predictions.top.return_value = []

        analyzer = PredictionAnalyzer(mock_predictions)
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")
        analyzer.clear_cache()
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")

        # top() should be called twice after cache clear
        assert mock_predictions.top.call_count == 2

    def test_uses_default_aggregate_method(self, mock_predictions):
        """Test that get_cached_predictions uses default_aggregate_method."""
        mock_predictions.top.return_value = []

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_method="median")
        analyzer.get_cached_predictions(n=5, rank_metric="rmse")

        call_kwargs = mock_predictions.top.call_args[1]
        assert call_kwargs.get("aggregate_method") == "median"

    def test_explicit_aggregate_method_overrides_default(self, mock_predictions):
        """Test that explicit aggregate_method overrides default."""
        mock_predictions.top.return_value = []

        analyzer = PredictionAnalyzer(mock_predictions, default_aggregate_method="median")
        analyzer.get_cached_predictions(n=5, rank_metric="rmse", aggregate_method="mean")

        call_kwargs = mock_predictions.top.call_args[1]
        assert call_kwargs.get("aggregate_method") == "mean"

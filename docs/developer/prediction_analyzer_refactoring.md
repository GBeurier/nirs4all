# PredictionAnalyzer Refactoring Plan

## Executive Summary

The `PredictionAnalyzer` class has grown to ~900 lines with mixed responsibilities, duplicate code, and inconsistent patterns. This document proposes a comprehensive refactoring to:
- Separate concerns into focused, testable components
- Eliminate code duplication (~300 lines of obsolete code)
- **Leverage refactored Predictions API** (predictions.top(), PredictionResult, lazy loading)
- Add comprehensive customization (ChartConfig for colors, fonts, sizes)
- Optimize performance for large figures (lazy rendering for heatmaps)
- Improve maintainability
- **Prepare for conversion to operators** (controller/operator pattern integration)

**Key Improvements:**
- **API Leverage**: Use `predictions.top()` instead of manual calculations → 2-5x speedup
- **Customization**: ChartConfig for colors, fonts, figure sizes → seamless defaults with full control
- **Optimization**: Lazy rendering for large heatmaps → faster batch generation
- **Future-Ready**: Utilities designed for reuse in controller/operator pattern → easy migration
- **Clean Architecture**: 5 chart classes + 5 utility classes → clear responsibilities

## Current Issues

### 1. Mixed Responsibilities
- Data fetching, filtering, and preparation
- Score calculation and extraction
- Matrix building for visualizations
- Plot rendering and styling
- Annotation logic

### 2. Code Duplication
- **Score extraction:** Implemented 3+ times across methods
- **Normalization:** `_normalize_matrix()` and `_normalize_heatmap_matrix()` are duplicates
- **Heatmap logic:** Two separate implementations (`plot_variable_heatmap` and `plot_heatmap_v2`)
- **Matrix building:** Similar logic in multiple places
- **Annotation:** Repeated text positioning and formatting logic

### 3. Not Leveraging Predictions API
- Manual score calculation when `predictions.top()` already provides structured results
- Not using `PredictionResult`/`PredictionResultsList` classes
- Missing lazy loading optimization (`load_arrays=False`)
- Redundant filtering logic instead of using `PredictionIndexer`
- Manual metric computation when `PredictionRanker` handles it

### 4. Obsolete Code (No Backward Compatibility Needed)
- `_get_enhanced_predictions()` - unused, superseded by predictions API
- `get_top_k()` - manual fallback logic no longer needed
- `plot_variable_heatmap()` - superseded by `plot_heatmap_v2()`
- Old heatmap helpers:
  - `_create_variable_heatmap()`
  - `_extract_scores_by_variables()`
  - `_plot_heatmap_matrix()`
  - `_render_heatmap_plot()`
  - `_add_heatmap_annotations()`
- `_build_score_dict()` - superseded by `_build_score_dict_with_ranking()`
- `_normalize_matrix()` - keep only v2 version

### 5. Missing Customization Options
- Hardcoded color schemes and colormaps
- No font size/style configuration
- Fixed figure sizing without flexible presets
- Limited color palette options

### 6. Incomplete/Incorrect Docstrings
- Many methods lack complete Google-style docstrings
- Missing Args, Returns, Raises sections
- Inconsistent formatting
- No type hints in docstrings

## Proposed Architecture

### File Structure

```
nirs4all/visualization/
├── predictions.py                    # Slim orchestrator (PredictionAnalyzer)
├── charts/
│   ├── __init__.py                   # Export all chart classes
│   ├── base.py                       # BaseChart abstract class
│   ├── top_k_comparison.py          # TopKComparisonChart
│   ├── confusion_matrix.py          # ConfusionMatrixChart
│   ├── histogram.py                 # ScoreHistogramChart
│   ├── heatmap.py                   # HeatmapChart (unified)
│   ├── candlestick.py               # CandlestickChart
│   └── config.py                    # ChartConfig (color schemes, fonts, etc.)
└── chart_utils/
    ├── __init__.py                   # Export utilities
    ├── predictions_adapter.py       # PredictionsAdapter (wraps predictions API)
    ├── matrix_builder.py            # MatrixBuilder
    ├── normalizer.py                # ScoreNormalizer
    ├── aggregator.py                # DataAggregator
    └── annotator.py                 # ChartAnnotator
```

**Design Note**: This structure prepares for future conversion to operators. Chart classes are designed to eventually integrate with the controller/operator pattern (see `SpectraChartController` for reference).

### Class Hierarchy

```
BaseChart (ABC)
├── TopKComparisonChart
├── ConfusionMatrixChart
├── ScoreHistogramChart
├── HeatmapChart
└── CandlestickChart

Configuration:
└── ChartConfig (color schemes, fonts, figure sizes)

Utilities (standalone, reusable in future operators):
├── PredictionsAdapter (wraps predictions.top() API)
├── MatrixBuilder (heatmap data structures)
├── ScoreNormalizer (normalization strategies)
├── DataAggregator (aggregation methods)
└── ChartAnnotator (text/legend helpers)
```

**Key Design Principles:**
1. **Leverage Predictions API**: Use `predictions.top()` instead of manual filtering/calculation
2. **Use PredictionResult**: Work with structured results from Predictions
3. **Lazy Loading**: Use `load_arrays=False` for metadata-only queries when possible
4. **Operator-Ready**: Chart classes designed for future controller/operator conversion
5. **Customizable**: All charts support configuration via ChartConfig
6. **Reusable Utilities**: Design utilities for use in future operators

## Detailed Component Design

### 0. ChartConfig (Color Schemes & Customization)

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior.

    Provides customization options for colors, fonts, and figure sizes.
    All parameters have sensible defaults for seamless usage.

    Attributes:
        colormap: Matplotlib colormap name for gradients (default: 'viridis').
        heatmap_colormap: Colormap for heatmaps (default: 'RdYlGn').
        partition_colors: Dict mapping partition names to colors.
        font_family: Font family for text (default: matplotlib default).
        title_fontsize: Title font size (default: 14).
        label_fontsize: Axis label font size (default: 10).
        tick_fontsize: Tick label font size (default: 9).
        figsize_small: Figure size for small plots (default: (10, 6)).
        figsize_medium: Figure size for medium plots (default: (12, 8)).
        figsize_large: Figure size for large plots (default: (16, 10)).
        dpi: DPI for saved figures (default: 300).
        alpha: Default alpha for plot elements (default: 0.7).
    """

    # Color schemes
    colormap: str = 'viridis'
    heatmap_colormap: str = 'RdYlGn'
    partition_colors: Dict[str, str] = None

    # Font settings
    font_family: Optional[str] = None
    title_fontsize: int = 14
    label_fontsize: int = 10
    tick_fontsize: int = 9

    # Figure sizes
    figsize_small: tuple = (10, 6)
    figsize_medium: tuple = (12, 8)
    figsize_large: tuple = (16, 10)

    # Other settings
    dpi: int = 300
    alpha: float = 0.7

    def __post_init__(self):
        """Initialize default partition colors if not provided."""
        if self.partition_colors is None:
            self.partition_colors = {
                'train': '#1f77b4',  # Blue
                'val': '#ff7f0e',    # Orange
                'test': '#2ca02c'    # Green
            }

    def apply_font_settings(self) -> None:
        """Apply font settings to matplotlib."""
        if self.font_family:
            plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.size'] = self.label_fontsize
        plt.rcParams['axes.titlesize'] = self.title_fontsize
        plt.rcParams['axes.labelsize'] = self.label_fontsize
        plt.rcParams['xtick.labelsize'] = self.tick_fontsize
        plt.rcParams['ytick.labelsize'] = self.tick_fontsize

    def get_figsize(self, size: str = 'medium') -> tuple:
        """Get figure size by name.

        Args:
            size: Size name ('small', 'medium', 'large').

        Returns:
            Figure size tuple (width, height).
        """
        sizes = {
            'small': self.figsize_small,
            'medium': self.figsize_medium,
            'large': self.figsize_large
        }
        return sizes.get(size, self.figsize_medium)
```

### 1. BaseChart (Abstract Base Class)

### 1. BaseChart (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

class BaseChart(ABC):
    """Abstract base class for all prediction visualization charts.

    Provides common interface and shared functionality for chart implementations.
    Each chart type should inherit from this class and implement required methods.

    Designed to be operator-ready for future integration with the controller/operator
    pattern (see SpectraChartController for reference pattern).

    Attributes:
        predictions: Predictions object containing prediction data.
        dataset_name_override: Optional override for dataset name display.
        config: ChartConfig instance for customization.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config: Optional[ChartConfig] = None):
        """Initialize chart with predictions data.

        Args:
            predictions: Predictions object containing prediction data.
            dataset_name_override: Optional dataset name override for display.
            config: Optional ChartConfig for customization (uses defaults if None).
        """
        self.predictions = predictions
        self.dataset_name_override = dataset_name_override
        self.config = config or ChartConfig()

    @abstractmethod
    def render(self, **kwargs) -> Figure:
        """Render the chart and return matplotlib Figure.

        Args:
            **kwargs: Chart-specific parameters.

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> None:
        """Validate input parameters for the chart.

        Args:
            **kwargs: Chart-specific parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        pass

    def _create_empty_figure(self, figsize, message: str) -> Figure:
        """Create empty figure with message.

        Args:
            figsize: Tuple of (width, height) for figure.
            message: Message to display in empty figure.

        Returns:
            matplotlib Figure with centered text message.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, message, ha='center', va='center',
                fontsize=self.config.title_fontsize)
        ax.set_title('No Data Available', fontsize=self.config.title_fontsize)
        return fig

    @staticmethod
    def _natural_sort_key(text: str):
        """Generate natural sorting key for strings with numbers.

        Handles numeric components naturally (e.g., 'model_2' before 'model_10').

        Args:
            text: String to generate sort key for.

        Returns:
            List of mixed int/str components for natural sorting.
        """
        import re
        def convert(part):
            return int(part) if part.isdigit() else part.lower()
        return [convert(c) for c in re.split(r'(\d+)', str(text))]
```

### 2. Utility Classes

#### PredictionsAdapter

```python
from typing import List, Dict, Any, Optional
from nirs4all.data.predictions import PredictionResult, PredictionResultsList

class PredictionsAdapter:
    """Adapter for Predictions API with optimized data access.

    Wraps the refactored Predictions API to provide convenient methods for charts.
    Leverages predictions.top(), lazy loading, and structured results.

    Key Optimizations:
    - Uses predictions.top() for efficient ranking
    - Supports lazy loading (load_arrays=False) for metadata-only queries
    - Works with PredictionResult/PredictionResultsList classes
    - Avoids redundant metric calculations

    Attributes:
        predictions: Predictions object instance.
    """

    def __init__(self, predictions):
        """Initialize adapter with predictions object.

        Args:
            predictions: Predictions object containing prediction data.
        """
        self.predictions = predictions

    def get_top_models(
        self,
        n: int,
        rank_metric: str,
        rank_partition: str = 'val',
        display_partition: str = 'test',
        aggregate_partitions: bool = False,
        **filters
    ) -> PredictionResultsList:
        """Get top N models using predictions.top() API.

        Leverages the refactored PredictionRanker for efficient ranking.

        Args:
            n: Number of top models to return.
            rank_metric: Metric to rank by (e.g., 'rmse', 'accuracy').
            rank_partition: Partition to rank on (default: 'val').
            display_partition: Partition to display results from (default: 'test').
            aggregate_partitions: If True, include train/val/test nested dicts.
            **filters: Additional filters (dataset_name, config_name, etc.).

        Returns:
            PredictionResultsList with top N models.
        """
        # Determine sort direction
        higher_better = rank_metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']

        return self.predictions.top(
            n=n,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            aggregate_partitions=aggregate_partitions,
            ascending=(not higher_better),
            **filters
        )

    def get_all_predictions_metadata(
        self,
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        **filters
    ) -> PredictionResultsList:
        """Get all predictions matching filters (metadata only, fast).

        Uses lazy loading for performance optimization.

        Args:
            rank_metric: Metric for sorting (default: 'rmse').
            rank_partition: Partition for sorting (default: 'val').
            **filters: Filter criteria.

        Returns:
            PredictionResultsList with all matching predictions.
        """
        # Use large N to get all predictions
        n = self.predictions.num_predictions
        higher_better = rank_metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']

        return self.predictions.top(
            n=n,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            ascending=(not higher_better),
            **filters
        )

    def extract_metric_values(
        self,
        predictions_list: PredictionResultsList,
        metric: str,
        partition: str = 'test'
    ) -> List[float]:
        """Extract metric values from prediction results.

        Args:
            predictions_list: List of PredictionResult objects.
            metric: Metric name to extract.
            partition: Partition to extract from (default: 'test').

        Returns:
            List of metric values (None for missing).
        """
        values = []
        score_field = f'{partition}_score'

        for pred in predictions_list:
            # Try to get score from appropriate field
            value = pred.get(score_field)

            # If not available, try to compute it
            if value is None and partition in pred:
                try:
                    from nirs4all.core.metrics import eval as eval_metric
                    part_data = pred[partition]
                    y_true = part_data.get('y_true', [])
                    y_pred = part_data.get('y_pred', [])
                    if y_true and y_pred:
                        value = eval_metric(y_true, y_pred, metric)
                except Exception:
                    pass

            values.append(value)

        return values

    @staticmethod
    def is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better.

        Args:
            metric: Metric name.

        Returns:
            True if higher values are better.
        """
        return metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']
```

#### ScoreExtractor (Deprecated - Use PredictionsAdapter)

```python
# NOTE: This class is deprecated in favor of PredictionsAdapter
# which leverages the predictions.top() API directly.
# Keeping for reference only.

class ScoreExtractor:
    """Centralized score extraction and calculation logic.

    DEPRECATED: Use PredictionsAdapter instead, which leverages
    the refactored predictions.top() API for better performance.

    Handles retrieving metric scores from predictions, with fallback to
    on-the-fly calculation using evaluator.
    """
    pass  # Implementation removed - use PredictionsAdapter
```

#### MatrixBuilder

```python
class MatrixBuilder:
    """Build matrices for heatmap visualizations.

    Handles grouping scores by variables and creating 2D matrices
    with support for different aggregation strategies.

    Optimized to work with PredictionResultsList from predictions.top().
    """

    @staticmethod
    def build_score_dict(
        predictions_list: PredictionResultsList,
        x_var: str,
        y_var: str,
        score_field: str,
        rank_field: Optional[str] = None
    ) -> Dict:
        """Group scores by x and y variables from PredictionResultsList.

        Args:
            predictions_list: PredictionResultsList from predictions.top().
            x_var: Variable name for x-axis grouping (e.g., 'model_name').
            y_var: Variable name for y-axis grouping (e.g., 'preprocessings').
            score_field: Field name containing score to display (e.g., 'test_score').
            rank_field: Optional field for ranking (if different from score_field).

        Returns:
            Nested dict: {y_val: {x_val: [(display_score, rank_score), ...]}}

        Example:
            >>> adapter = PredictionsAdapter(predictions)
            >>> results = adapter.get_all_predictions_metadata()
            >>> score_dict = MatrixBuilder.build_score_dict(
            ...     results, 'model_name', 'preprocessings', 'test_score', 'val_score'
            ... )
        """
        from collections import defaultdict

        score_dict = defaultdict(lambda: defaultdict(list))

        for pred in predictions_list:
            x_val = str(pred.get(x_var, 'unknown'))
            y_val = str(pred.get(y_var, 'unknown'))
            display_score = pred.get(score_field)
            rank_score = pred.get(rank_field) if rank_field else display_score

            if display_score is not None and not np.isnan(display_score):
                score_dict[y_val][x_val].append(
                    (display_score, rank_score if rank_score is not None else display_score)
                )

        return score_dict

    @staticmethod
    def build_matrices(
        score_dict: Dict,
        aggregation: str,
        higher_better: bool,
        natural_sort: bool = True
    ) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """Build matrices from score dictionary.

        Args:
            score_dict: Nested dict from build_score_dict().
            aggregation: Aggregation method ('best', 'mean', 'median').
            higher_better: Whether higher scores are better.
            natural_sort: Whether to use natural sorting for labels.

        Returns:
            Tuple of (y_labels, x_labels, score_matrix, count_matrix).

        Raises:
            ValueError: If aggregation method is invalid.
        """
        # Implementation using DataAggregator
        pass
```

#### ScoreNormalizer

```python
class ScoreNormalizer:
    """Normalize scores for visualization.

    Handles normalization to [0, 1] range with support for
    both higher-is-better and lower-is-better metrics.
    """

    @staticmethod
    def normalize(matrix: np.ndarray, higher_better: bool) -> np.ndarray:
        """Normalize matrix values to [0, 1] range.

        For higher-is-better metrics: (x - min) / (max - min)
        For lower-is-better metrics: 1 - (x - min) / (max - min)

        Args:
            matrix: Matrix to normalize (may contain NaN).
            higher_better: Whether higher values are better.

        Returns:
            Normalized matrix with same shape, NaN preserved.
        """
        pass

    @staticmethod
    def is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better.

        Args:
            metric: Metric name.

        Returns:
            True if higher values are better (e.g., r2, accuracy).
        """
        return metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']
```

#### DataAggregator

```python
class DataAggregator:
    """Aggregate scores using different strategies.

    Supports multiple aggregation methods with proper handling of
    ranking information (when display and rank metrics differ).
    """

    @staticmethod
    def aggregate(scores: List, method: str, higher_better: bool) -> float:
        """Aggregate scores using specified method.

        Handles both simple scores and tuples of (display_score, rank_score).

        Args:
            scores: List of scores (may be tuples of (display, rank)).
            method: Aggregation method ('best', 'mean', 'median').
            higher_better: Whether higher scores are better (for 'best' method).

        Returns:
            Aggregated score.

        Raises:
            ValueError: If method is invalid.

        Example:
            >>> scores = [(0.85, 0.90), (0.87, 0.88), (0.86, 0.92)]
            >>> # Get best based on rank_score, return its display_score
            >>> DataAggregator.aggregate(scores, 'best', higher_better=True)
            0.86  # display_score of the entry with best rank_score (0.92)
        """
        if not scores:
            return np.nan

        if method not in ['best', 'mean', 'median']:
            raise ValueError(f"Invalid aggregation method: {method}")

        # Check if scores are tuples (display, rank)
        if isinstance(scores[0], tuple):
            if method == 'best':
                # Select based on rank_score, return display_score
                rank_scores = [rank for _, rank in scores]
                best_idx = np.argmax(rank_scores) if higher_better else np.argmin(rank_scores)
                return scores[best_idx][0]  # Return display_score
            else:
                # For mean/median, use display_scores only
                display_scores = [disp for disp, _ in scores]
                return np.mean(display_scores) if method == 'mean' else np.median(display_scores)
        else:
            # Simple scores
            if method == 'best':
                return max(scores) if higher_better else min(scores)
            elif method == 'mean':
                return np.mean(scores)
            else:  # median
                return np.median(scores)
```

#### ChartAnnotator

```python
class ChartAnnotator:
    """Helper for adding annotations to charts.

    Centralizes text formatting, positioning, and color selection
    for chart annotations. Uses ChartConfig for styling.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize annotator with config.

        Args:
            config: ChartConfig for styling (uses defaults if None).
        """
        self.config = config or ChartConfig()

    def add_heatmap_annotations(
        self,
        ax,
        matrix: np.ndarray,
        normalized_matrix: np.ndarray,
        count_matrix: np.ndarray,
        show_counts: bool = True,
        precision: int = 3
    ) -> None:
        """Add text annotations to heatmap cells.

        Args:
            ax: Matplotlib axis object.
            matrix: Original score matrix.
            normalized_matrix: Normalized matrix (for color determination).
            count_matrix: Matrix of sample counts.
            show_counts: Whether to display counts.
            precision: Decimal precision for scores.
        """
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    score_text = f'{matrix[i, j]:.{precision}f}'

                    if show_counts and count_matrix[i, j] > 1:
                        text = f'{score_text}\n(n={int(count_matrix[i, j])})'
                    else:
                        text = score_text

                    # Determine text color based on background
                    text_color = self.get_text_color(normalized_matrix[i, j])

                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=self.config.tick_fontsize,
                           color=text_color)

    @staticmethod
    def get_text_color(background_value: float, threshold: float = 0.5) -> str:
        """Determine text color based on background for optimal contrast.

        Args:
            background_value: Background color value [0, 1].
            threshold: Threshold for switching colors.

        Returns:
            'white' or 'black' for optimal contrast.
        """
        return 'white' if background_value < threshold else 'black'

    def add_statistics_box(
        self,
        ax,
        stats: Dict[str, float],
        position: str = 'upper right',
        precision: int = 4
    ) -> None:
        """Add statistics text box to plot.

        Args:
            ax: Matplotlib axis object.
            stats: Dictionary of statistics to display.
            position: Box position ('upper right', 'upper left', etc.).
            precision: Decimal precision for values.
        """
        stats_text = '\n'.join([f'{k}={v:.{precision}f}' for k, v in stats.items()])

        # Map position string to coordinates
        positions = {
            'upper right': (0.98, 0.98),
            'upper left': (0.02, 0.98),
            'lower right': (0.98, 0.02),
            'lower left': (0.02, 0.02)
        }
        x, y = positions.get(position, (0.98, 0.98))

        ax.text(x, y, stats_text, transform=ax.transAxes,
               verticalalignment='top' if y > 0.5 else 'bottom',
               horizontalalignment='right' if x > 0.5 else 'left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=self.config.tick_fontsize)
```

### 3. Chart Classes

#### TopKComparisonChart

```python
class TopKComparisonChart(BaseChart):
    """Scatter plots comparing predicted vs observed values for top K models.

    Displays predicted vs true scatter plots alongside residual plots
    for the best performing models according to a ranking metric.
    """

    def validate_inputs(self, k: int, rank_metric: str, **kwargs) -> None:
        """Validate input parameters.

        Args:
            k: Number of top models to display.
            rank_metric: Metric for ranking.
            **kwargs: Additional parameters.

        Raises:
            ValueError: If k < 1 or rank_metric is invalid.
        """
        pass

    def render(self, k: int = 5, rank_metric: str = 'rmse',
               rank_partition: str = 'val', display_partition: str = 'all',
               dataset_name: Optional[str] = None,
               figsize: Tuple[int, int] = (16, 10)) -> Figure:
        """Render top K comparison chart.

        Models are ranked using rank_metric on rank_partition, then predictions
        from display_partition(s) are visualized.

        Args:
            k: Number of top models to display.
            rank_metric: Metric for ranking models (e.g., 'rmse', 'r2').
            rank_partition: Partition used for ranking ('train', 'val', 'test').
            display_partition: Partition(s) to display ('all', 'test', 'val', 'train').
            dataset_name: Optional dataset filter.
            figsize: Figure size as (width, height).

        Returns:
            matplotlib Figure with scatter and residual plots.

        Raises:
            ValueError: If parameters are invalid.
        """
        pass

    def _prepare_data(self, k, rank_metric, rank_partition,
                     display_partition, dataset_name):
        """Prepare data for visualization."""
        pass

    def _render_plots(self, fig, axes, predictions, rank_metric,
                     rank_partition, partitions_to_display):
        """Render scatter and residual plots."""
        pass
```

#### ConfusionMatrixChart

```python
class ConfusionMatrixChart(BaseChart):
    """Confusion matrix visualizations for classification models.

    Displays confusion matrices for top K classification models,
    with proper handling of multi-class predictions.
    """

    def render(self, k: int = 5, metric: str = 'accuracy',
               rank_partition: str = 'val', display_partition: str = 'test',
               dataset_name: Optional[str] = None,
               figsize: Tuple[int, int] = (16, 10), **filters) -> Figure:
        """Render confusion matrices for top K models.

        Args:
            k: Number of top models to display.
            metric: Metric for ranking (e.g., 'accuracy', 'f1').
            rank_partition: Partition used for ranking.
            display_partition: Partition to display confusion matrix from.
            dataset_name: Optional dataset filter.
            figsize: Figure size.
            **filters: Additional filters (e.g., config_name="config1").

        Returns:
            matplotlib Figure with confusion matrices.

        Raises:
            ValueError: If metric is not suitable for classification.
        """
        pass
```

#### ScoreHistogramChart

```python
class ScoreHistogramChart(BaseChart):
    """Histogram of score distributions.

    Displays distribution of a metric across predictions with
    statistical annotations.
    """

    def render(self, metric: str = 'rmse', dataset_name: Optional[str] = None,
               partition: Optional[str] = None, bins: int = 20,
               figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """Render score histogram with statistics.

        Args:
            metric: Metric to plot (e.g., 'rmse', 'r2').
            dataset_name: Optional dataset filter.
            partition: Partition to display scores from (default: 'test').
            bins: Number of histogram bins.
            figsize: Figure size.

        Returns:
            matplotlib Figure with histogram and statistics.
        """
        pass
```

#### HeatmapChart

```python
class HeatmapChart(BaseChart):
    """Heatmap visualization of performance across two variables.

    Unified heatmap implementation supporting flexible ranking and
    display configurations with multiple aggregation strategies.

    Optimized Implementation:
    - Uses PredictionsAdapter for efficient data access
    - Supports lazy figure creation for large heatmaps (< 50 lines overhead)
    - Leverages predictions.top() API instead of manual calculations
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config: Optional[ChartConfig] = None):
        """Initialize heatmap chart.

        Args:
            predictions: Predictions object.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        super().__init__(predictions, dataset_name_override, config)
        self.adapter = PredictionsAdapter(predictions)
        self.matrix_builder = MatrixBuilder()
        self.normalizer = ScoreNormalizer()
        self.annotator = ChartAnnotator(config)

    def render(
        self,
        x_var: str,
        y_var: str,
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'test',
        figsize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        aggregation: str = 'best',
        show_counts: bool = True,
        lazy_render: bool = False,
        **filters
    ) -> Figure:
        """Render performance heatmap.

        Models are ranked by rank_metric on rank_partition, then display_metric
        scores from display_partition are shown in the heatmap cells.

        Args:
            x_var: Variable for x-axis (e.g., 'model_name', 'preprocessings').
            y_var: Variable for y-axis.
            rank_metric: Metric used to rank/select best models.
            rank_partition: Partition used for ranking models.
            display_metric: Metric to display in heatmap (default: same as rank_metric).
            display_partition: Partition to display scores from.
            figsize: Figure size (uses config default if None).
            normalize: Whether to normalize scores to [0,1].
            aggregation: How to aggregate multiple scores ('best', 'mean', 'median').
            show_counts: Whether to show sample counts in cells.
            lazy_render: If True, defers figure creation until show() (faster for batches).
            **filters: Additional filters (e.g., dataset_name="mydata").

        Returns:
            matplotlib Figure with heatmap visualization.

        Raises:
            ValueError: If variables or metrics are invalid.

        Example:
            >>> config = ChartConfig(heatmap_colormap='coolwarm')
            >>> chart = HeatmapChart(predictions, config=config)
            >>> fig = chart.render('model_name', 'preprocessings',
            ...                    rank_metric='rmse', rank_partition='val',
            ...                    display_metric='r2', display_partition='test')
        """
        if not display_metric:
            display_metric = rank_metric

        # Validate inputs
        self.validate_inputs(x_var=x_var, y_var=y_var, rank_metric=rank_metric)

        # Use adapter to get all predictions efficiently
        predictions_list = self.adapter.get_all_predictions_metadata(
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            **filters
        )

        if not predictions_list:
            figsize = figsize or self.config.get_figsize('medium')
            return self._create_empty_figure(figsize, 'No predictions found')

        # Build score dictionary and matrices
        score_field = f'{display_partition}_score'
        rank_field = f'{rank_partition}_score' if rank_partition != display_partition else None

        score_dict = self.matrix_builder.build_score_dict(
            predictions_list, x_var, y_var, score_field, rank_field
        )

        higher_better = self.adapter.is_higher_better(display_metric)
        y_labels, x_labels, matrix, count_matrix = self.matrix_builder.build_matrices(
            score_dict, aggregation, higher_better
        )

        # Normalize if requested
        normalized_matrix = self.normalizer.normalize(matrix, normalize, higher_better)

        # Render figure (lazy or immediate)
        if lazy_render:
            # Return placeholder figure - actual rendering deferred
            return self._create_lazy_figure()
        else:
            return self._render_heatmap(
                matrix, normalized_matrix, count_matrix, x_labels, y_labels,
                x_var, y_var, rank_metric, rank_partition, display_metric,
                display_partition, figsize, normalize, aggregation, show_counts
            )

    def validate_inputs(self, x_var: str, y_var: str, rank_metric: str) -> None:
        """Validate heatmap inputs.

        Args:
            x_var: X-axis variable name.
            y_var: Y-axis variable name.
            rank_metric: Ranking metric name.

        Raises:
            ValueError: If inputs are invalid.
        """
        valid_vars = ['model_name', 'dataset_name', 'config_name', 'preprocessings', 'partition']
        if x_var not in valid_vars:
            raise ValueError(f"Invalid x_var: {x_var}. Must be one of {valid_vars}")
        if y_var not in valid_vars:
            raise ValueError(f"Invalid y_var: {y_var}. Must be one of {valid_vars}")
        if not rank_metric:
            raise ValueError("rank_metric cannot be empty")

    def _render_heatmap(self, matrix, normalized_matrix, count_matrix, x_labels, y_labels,
                       x_var, y_var, rank_metric, rank_partition, display_metric,
                       display_partition, figsize, normalize, aggregation, show_counts) -> Figure:
        """Internal method to render heatmap figure."""
        # Implementation details...
        pass

    def _create_lazy_figure(self) -> Figure:
        """Create placeholder for lazy rendering."""
        fig = plt.figure()
        fig._lazy_render = True  # Mark for deferred rendering
        return fig
```

#### CandlestickChart

```python
class CandlestickChart(BaseChart):
    """Candlestick/box plot for score distributions by variable.

    Shows score distribution statistics (min, Q25, mean, Q75, max)
    for each value of a grouping variable.
    """

    def render(self, filters: Dict[str, Any], variable: str,
               metric: str = 'rmse', figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """Render candlestick chart.

        Args:
            filters: Dictionary of filters (e.g., {"partition": "test"}).
            variable: Variable to group by (e.g., 'model_name').
            metric: Metric to analyze.
            figsize: Figure size.

        Returns:
            matplotlib Figure with candlestick visualization.
        """
        pass
```

### 4. Refactored PredictionAnalyzer

The main class becomes a lightweight orchestrator that delegates to chart classes:

```python
class PredictionAnalyzer:
    """Orchestrator for prediction analysis and visualization.

    Provides a unified interface for creating various prediction visualizations.
    Delegates to specialized chart classes for rendering.

    This class will be deprecated once charts are converted to operators following
    the controller/operator pattern (see SpectraChartController for reference).

    Leverages the refactored Predictions API (predictions.top(), PredictionResult, etc.)
    for efficient data access and avoids redundant calculations.

    Attributes:
        predictions: Predictions object containing prediction data.
        dataset_name_override: Optional override for dataset name display.
        config: ChartConfig for customization across all charts.

    Example:
        >>> # Basic usage with defaults
        >>> analyzer = PredictionAnalyzer(predictions_obj)
        >>> fig = analyzer.plot_top_k(k=10, metric='rmse')
        >>>
        >>> # Custom configuration
        >>> config = ChartConfig(heatmap_colormap='coolwarm', title_fontsize=16)
        >>> analyzer = PredictionAnalyzer(predictions_obj, config=config)
        >>> fig = analyzer.plot_heatmap('model_name', 'preprocessings')
    """

    def __init__(
        self,
        predictions_obj: Predictions,
        dataset_name_override: str = None,
        config: Optional[ChartConfig] = None
    ):
        """Initialize analyzer with predictions object.

        Args:
            predictions_obj: Predictions object containing prediction data.
            dataset_name_override: Optional dataset name override for display.
            config: Optional ChartConfig for customization (uses defaults if None).
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.config = config or ChartConfig()

    def plot_top_k(
        self,
        k: int = 5,
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        display_partition: str = 'all',
        **kwargs
    ) -> Figure:
        """Plot top K model comparison (scatter + residuals).

        Uses predictions.top() API for efficient ranking and data retrieval.

        Args:
            k: Number of top models to display.
            rank_metric: Metric to rank by (default: 'rmse').
            rank_partition: Partition for ranking (default: 'val').
            display_partition: Partition(s) to display ('all', 'test', 'val', 'train').
            **kwargs: Additional arguments passed to TopKComparisonChart.

        Returns:
            matplotlib Figure with scatter and residual plots.

        Example:
            >>> fig = analyzer.plot_top_k(k=10, rank_metric='r2', rank_partition='val')
        """
        chart = TopKComparisonChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(
            k=k,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            **kwargs
        )

    def plot_confusion_matrix(
        self,
        k: int = 5,
        metric: str = 'accuracy',
        rank_partition: str = 'val',
        display_partition: str = 'test',
        **kwargs
    ) -> Figure:
        """Plot confusion matrices for top K classification models.

        Args:
            k: Number of top models to display.
            metric: Metric for ranking (default: 'accuracy').
            rank_partition: Partition for ranking (default: 'val').
            display_partition: Partition to display matrices from (default: 'test').
            **kwargs: Additional arguments passed to ConfusionMatrixChart.

        Returns:
            matplotlib Figure with confusion matrices.
        """
        chart = ConfusionMatrixChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(
            k=k,
            metric=metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            **kwargs
        )

    def plot_histogram(
        self,
        metric: str = 'rmse',
        partition: str = 'test',
        **kwargs
    ) -> Figure:
        """Plot score distribution histogram.

        Args:
            metric: Metric to plot (default: 'rmse').
            partition: Partition to display (default: 'test').
            **kwargs: Arguments passed to ScoreHistogramChart.

        Returns:
            matplotlib Figure with histogram.
        """
        chart = ScoreHistogramChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(metric=metric, partition=partition, **kwargs)

    def plot_heatmap(
        self,
        x_var: str,
        y_var: str,
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'test',
        **kwargs
    ) -> Figure:
        """Plot performance heatmap across two variables.

        Leverages predictions.top() for efficient ranking and filtering.

        Args:
            x_var: Variable for x-axis (e.g., 'model_name').
            y_var: Variable for y-axis (e.g., 'preprocessings').
            rank_metric: Metric to rank by (default: 'rmse').
            rank_partition: Partition for ranking (default: 'val').
            display_metric: Metric to display (default: same as rank_metric).
            display_partition: Partition to display from (default: 'test').
            **kwargs: Additional arguments passed to HeatmapChart.

        Returns:
            matplotlib Figure with heatmap.

        Example:
            >>> # Rank on val RMSE, display test R2
            >>> fig = analyzer.plot_heatmap(
            ...     'model_name', 'preprocessings',
            ...     rank_metric='rmse', rank_partition='val',
            ...     display_metric='r2', display_partition='test'
            ... )
        """
        chart = HeatmapChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(
            x_var, y_var,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_metric=display_metric,
            display_partition=display_partition,
            **kwargs
        )

    def plot_candlestick(
        self,
        variable: str,
        metric: str = 'rmse',
        partition: str = 'test',
        **kwargs
    ) -> Figure:
        """Plot candlestick chart for score distribution by variable.

        Args:
            variable: Variable to group by (e.g., 'model_name').
            metric: Metric to analyze (default: 'rmse').
            partition: Partition to display (default: 'test').
            **kwargs: Additional arguments passed to CandlestickChart.

        Returns:
            matplotlib Figure with candlestick visualization.
        """
        chart = CandlestickChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        # Build filters dict for CandlestickChart
        filters = {'partition': partition, **kwargs}
        return chart.render(filters=filters, variable=variable, metric=metric)

    # Backward compatibility aliases
    def plot_top_k_comparison(self, *args, **kwargs) -> Figure:
        """Alias for plot_top_k() (backward compatibility)."""
        return self.plot_top_k(*args, **kwargs)

    def plot_top_k_confusionMatrix(self, *args, **kwargs) -> Figure:
        """Alias for plot_confusion_matrix() (backward compatibility)."""
        return self.plot_confusion_matrix(*args, **kwargs)

    def plot_score_histogram(self, *args, **kwargs) -> Figure:
        """Alias for plot_histogram() (backward compatibility)."""
        return self.plot_histogram(*args, **kwargs)

    def plot_heatmap_v2(self, *args, **kwargs) -> Figure:
        """Alias for plot_heatmap() (backward compatibility)."""
        return self.plot_heatmap(*args, **kwargs)

    def plot_variable_candlestick(self, *args, **kwargs) -> Figure:
        """Alias for plot_candlestick() (backward compatibility)."""
        return self.plot_candlestick(*args, **kwargs)
```

## Migration Strategy

### Phase 1: Setup (Low Risk)
1. Create new directory structure
2. Implement utility classes (can be tested independently)
3. Implement BaseChart abstract class
4. Add comprehensive unit tests for utilities

### Phase 2: Extract Charts (Medium Risk)
1. Implement one chart class at a time, starting with simplest:
   - ScoreHistogramChart (simplest)
   - CandlestickChart
   - ConfusionMatrixChart
   - TopKComparisonChart
   - HeatmapChart (most complex)
2. Add tests for each chart
3. Verify visual output matches current implementation

### Phase 3: Refactor Main Class (Low Risk)
1. Update PredictionAnalyzer to use new chart classes
2. Keep old method names for backward compatibility
3. Remove obsolete code

### Phase 4: Cleanup (Low Risk)
1. Remove old helper methods
2. Update all docstrings to Google style
3. Run full test suite
4. Update documentation

## Performance Improvements

### 1. Leverage Predictions API
**Impact: Major (eliminates redundant calculations)**
- Use `predictions.top()` instead of manual filtering and scoring
- Work with `PredictionResult`/`PredictionResultsList` structures
- Let `PredictionRanker` handle metric calculations
- Avoid redundant metric computation in charts

**Implementation:**
```python
# ❌ OLD: Manual calculation
predictions = self.predictions.filter_predictions(**filters)
for pred in predictions:
    score = self._extract_metric_score(pred, metric)  # Redundant!

# ✅ NEW: Use predictions.top()
results = self.predictions.top(
    n=k, rank_metric=metric, rank_partition=partition, **filters
)
# Scores already computed and ranked!
```

### 2. Lazy Loading for Metadata Queries
**Impact: Medium (faster queries when arrays not needed)**
- Use `load_arrays=False` for metadata-only operations
- Defer array hydration until actually needed for plotting
- Particularly useful for large heatmaps where we only need metadata initially

**Implementation:**
```python
# Fast metadata query (no array loading)
results = predictions.top(n=100, rank_metric='rmse', load_arrays=False)

# Arrays only loaded when needed for visualization
for result in results:
    if needs_plotting:
        y_true = result['y_true']  # Hydrated on demand
```

### 3. Optimized Figure Creation for Large Heatmaps (< 50 lines)
**Impact: Medium (faster rendering of complex visualizations)**
- Optional lazy rendering for batch plot generation
- Deferred figure creation until `plt.show()` or save
- Particularly useful for generating multiple heatmaps

**Implementation:**
```python
class HeatmapChart(BaseChart):
    def render(self, ..., lazy_render: bool = False):
        if lazy_render:
            # Create lightweight placeholder
            return self._create_lazy_figure()
        else:
            # Full immediate rendering
            return self._render_heatmap(...)

    def _create_lazy_figure(self):
        """Deferred rendering placeholder (<10 lines)."""
        fig = plt.figure()
        fig._lazy_render = True
        fig._render_callback = lambda: self._render_heatmap(...)
        return fig
```

### 4. Batch Operations
**Impact: Minor (code elegance)**
- Use PredictionsAdapter batch methods where applicable
- Vectorized numpy operations instead of Python loops
- Group similar calculations together

### 5. Avoid Redundant Calculations
**Impact: Minor to Medium**
- Cache normalized matrices within render cycle
- Reuse score dictionaries across similar queries
- Calculate statistics once and store

### 6. Memory Efficiency
**Impact: Minor**
- Clear large intermediate arrays when done
- Use views instead of copies where possible
- Efficient data structures (e.g., defaultdict)

**Total Expected Speedup**: 2-5x for typical workflows due to API leverage and lazy loading

## Open Questions & Answers

### Unit Tests
- Each utility class: 100% coverage
- Each chart class: validate inputs, data preparation, rendering
- Mock predictions object for isolated testing

### Integration Tests
- Test with real predictions data
- Verify visual output matches expectations
- Test error handling and edge cases

### Visual Regression Tests (Optional)
- Save reference images
- Compare rendered figures with references
- Flag significant visual differences

## Documentation Requirements

### Code Documentation
- All public methods: complete Google-style docstrings
- All classes: docstring with Attributes section
- Complex algorithms: inline comments explaining logic
- Examples in docstrings where helpful

### User Documentation
- Update user guide with new API
- Provide migration guide if API changes
- Add examples for each chart type
- Document aggregation strategies

## Breaking Changes

### Removed (No Backward Compatibility)
- `plot_variable_heatmap()` → use `plot_heatmap()` instead
- `get_top_k()` → internal method, use chart render methods
- All `_extract_scores_by_variables()` variants → use ScoreExtractor
- Old heatmap helper methods → use HeatmapChart

### Renamed (Keep Aliases)
- `plot_top_k_comparison()` → `plot_top_k()` (keep old name as alias)
- `plot_top_k_confusionMatrix()` → `plot_confusion_matrix()` (keep old name as alias)
- `plot_score_histogram()` → `plot_histogram()` (keep old name as alias)
- `plot_heatmap_v2()` → `plot_heatmap()` (keep old name as alias)
- `plot_variable_candlestick()` → `plot_candlestick()` (keep old name as alias)

## Timeline Estimate

- **Phase 1 (Setup)**: 2-3 hours
  - Create directory structure
  - Implement ChartConfig
  - Implement PredictionsAdapter
  - Setup base utilities (ScoreNormalizer, DataAggregator, ChartAnnotator)
  - Create BaseChart with config support

- **Phase 2 (Charts)**: 8-12 hours (2-2.5 hours per chart avg)
  - ScoreHistogramChart (simplest) - 2h
  - CandlestickChart - 2h
  - ConfusionMatrixChart - 2h
  - TopKComparisonChart - 2.5h
  - HeatmapChart (most complex, with lazy rendering) - 3h

- **Phase 3 (Refactor Main)**: 2-3 hours
  - Update PredictionAnalyzer to delegate
  - Add backward compatibility aliases
  - Remove obsolete code

- **Phase 4 (Cleanup & Polish)**: 3-4 hours
  - Complete all Google-style docstrings
  - Add customization examples
  - Performance validation
  - Code review and adjustments

- **Testing**: 5-6 hours
  - Unit tests for utilities (2h)
  - Integration tests for charts (2h)
  - Visual validation (1h)
  - Performance benchmarks (1h)

- **Documentation**: 3-4 hours
  - Update user guide
  - Customization guide
  - Migration examples
  - API reference

**Total: 23-32 hours** (increased from original due to added customization and optimization features)

## Future: Controller/Operator Pattern Integration

Once the refactoring is complete, charts can be progressively converted to operators following the controller/operator pattern (see `SpectraChartController` for reference).

### Target Architecture

```python
@register_controller
class HeatmapChartController(OperatorController):
    """Controller for heatmap chart operator in pipeline.

    Integrates HeatmapChart with pipeline execution.
    """

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match 'heatmap' keyword in pipeline."""
        return keyword in ["heatmap", "heatmap_chart"]

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Charts don't run in prediction mode."""
        return False

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: SpectroDataset,
        context: Dict[str, Any],
        runner: PipelineRunner,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute heatmap chart generation in pipeline.

        Returns:
            Tuple of (context, image_list) with generated plots.
        """
        # Skip in prediction/explain mode
        if mode in ["predict", "explain"]:
            return context, []

        # Extract parameters from operator config
        x_var = operator.get('x_var', 'model_name')
        y_var = operator.get('y_var', 'preprocessings')
        rank_metric = operator.get('rank_metric', 'rmse')

        # Get predictions from runner
        predictions = runner.predictions  # Access pipeline predictions

        # Create chart with pipeline config
        config = self._build_chart_config(operator)
        chart = HeatmapChart(predictions, dataset.name, config)

        # Render chart
        fig = chart.render(x_var, y_var, rank_metric=rank_metric, **operator)

        # Save using runner's saver (like SpectraChartController)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=config.dpi, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        # Save output
        output_path = runner.saver.save_output(
            step_number=runner.step_number,
            name='heatmap_chart',
            data=img_png_binary,
            extension='.png'
        )

        # Track output
        img_list = []
        if output_path:
            img_list.append({
                "name": "heatmap_chart.png",
                "path": str(output_path),
                "type": "chart_output"
            })

        # Manage plot display
        if runner.plots_visible:
            runner._figure_refs.append(fig)
            plt.show()
        else:
            plt.close(fig)

        return context, img_list

    def _build_chart_config(self, operator: Dict) -> ChartConfig:
        """Build ChartConfig from operator parameters."""
        return ChartConfig(
            colormap=operator.get('colormap', 'viridis'),
            heatmap_colormap=operator.get('heatmap_colormap', 'RdYlGn'),
            title_fontsize=operator.get('title_fontsize', 14),
            figsize_large=operator.get('figsize', (16, 10)),
            dpi=operator.get('dpi', 300)
        )
```

### Pipeline YAML Example

```yaml
pipeline:
  - name: "Train Models"
    steps:
      - train_model: PLS
      - train_model: SVR

  - name: "Visualize Results"
    steps:
      - heatmap:
          x_var: model_name
          y_var: preprocessings
          rank_metric: rmse
          rank_partition: val
          display_metric: r2
          display_partition: test
          heatmap_colormap: coolwarm
          title_fontsize: 16
```

### Benefits of Operator Pattern

1. **Declarative Configuration**: Define charts in pipeline YAML
2. **Integration with Runner**: Automatic saving, tracking, display management
3. **Reusable Utilities**: Chart classes and utilities work in both contexts
4. **Consistent API**: Same pattern as other operators (SpectraChartController)
5. **Pipeline Output**: Charts saved alongside other pipeline artifacts

## Success Criteria

1. All existing functionality preserved
2. No performance regression (ideally improvements)
3. All tests passing
4. Complete Google-style docstrings
5. Code coverage ≥ 90%
6. Clean separation of concerns
7. Easy to add new chart types
8. Ready for operator conversion

## Open Questions & Answers

### 1. Caching for predictions.top() results?
**Answer: No** - Keep it simple. The predictions.top() API is already optimized with efficient filtering and ranking. Adding caching would increase complexity without significant benefit.

### 2. Async/parallel rendering for multiple charts?
**Answer: Yes, if < 50 lines and low complexity** - Big figures (heatmaps, spectra) can be costly to display. Implement lazy rendering option (deferred figure creation) for batch generation scenarios. This allows:
- Multiple charts to be prepared quickly
- Actual rendering deferred until display/save
- Better memory management for large batches

**Design:**
```python
# Lazy rendering for batch workflows
charts = []
for config in chart_configs:
    fig = chart.render(..., lazy_render=True)  # Fast placeholder
    charts.append(fig)

# Actual rendering when needed
for fig in charts:
    fig.show()  # Triggers deferred rendering
```

### 3. Utilities in separate package?
**Answer: Not yet, but prepare for it** - Design utilities with clean API anticipating future use as operators in the controller/operator pattern (similar to SpectraChartController). Structure code to be easily extracted later. Key principles:
- Chart classes should follow similar patterns to SpectraChartController
- Utilities should be framework-agnostic and reusable
- Clean separation between data access, computation, and rendering
- Prepare for eventual integration with pipeline runner

**Future Controller Pattern:**
```python
@register_controller
class HeatmapChartController(OperatorController):
    def execute(self, step, operator, dataset, context, runner, ...):
        # Use same HeatmapChart class
        chart = HeatmapChart(predictions)
        fig = chart.render(...)
        # Save and track outputs like SpectraChartController
        return context, image_list
```

### 4. Customization options?
**Answer: Yes, make it configurable** - Add ChartConfig class for:
- **Color schemes**: Configurable colormaps (viridis, coolwarm, custom)
- **Partition colors**: Customizable train/val/test colors
- **Fonts**: Font family, sizes for titles/labels/ticks
- **Figure sizes**: Predefined presets (small/medium/large) and custom
- **DPI**: Output resolution control
- **Alpha**: Transparency settings

Make it seamless by default (good defaults) but fully customizable when needed. All charts accept optional `config` parameter.

**Usage:**
```python
# Default config (seamless)
chart = HeatmapChart(predictions)
fig = chart.render('model_name', 'preprocessings')

# Custom config
config = ChartConfig(
    heatmap_colormap='coolwarm',
    title_fontsize=16,
    figsize_large=(20, 12),
    partition_colors={'train': 'blue', 'val': 'orange', 'test': 'green'}
)
chart = HeatmapChart(predictions, config=config)
fig = chart.render('model_name', 'preprocessings')
```

## Next Steps

Once this plan is approved:
1. Create feature branch
2. Implement Phase 1 (utilities)
3. Get code review on utilities before proceeding
4. Implement remaining phases
5. Final review and merge

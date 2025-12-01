"""YChartController - Y values histogram visualization with train/test split and folds."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
import io

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep

@register_controller
class YChartController(OperatorController):

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword in ["y_chart", "chart_y"]

    @classmethod
    def use_multi_source(cls) -> bool:
        return False  # Y values don't depend on source

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Chart controllers should skip execution during prediction mode."""
        return False

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: Any,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple['ExecutionContext', Any]:
        """
        Execute y values histogram visualization.

        If cross-validation folds exist (more than 1 fold), displays a grid showing:
        - One histogram per fold validation set
        - One histogram for the test partition (if available)

        Otherwise, displays a simple train vs test histogram.

        Returns:
            Tuple of (context, StepOutput)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, StepOutput()

        # Get folds from dataset
        folds = dataset.folds

        # Check if we have multiple CV folds (not just a single train/test split)
        has_cv_folds = folds is not None and len(folds) > 1

        if has_cv_folds:
            # Grid mode: show each fold's validation set + test partition
            fig, chart_name = self._create_fold_grid_histogram(dataset, context, folds)
        else:
            # Simple mode: train vs test
            local_context = context.with_partition("train")
            y_train = dataset.y(local_context.selector)

            local_context = context.with_partition("test")
            y_test = dataset.y(local_context.selector)

            y_all = dataset.y(context.selector)

            fig, _ = self._create_bicolor_histogram(y_train, y_test, y_all)
            chart_name = "Y_distribution_train_test"

        # Save plot to memory buffer as PNG binary
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        # Create StepOutput with the chart
        step_output = StepOutput(
            outputs=[(img_png_binary, chart_name, "png")]
        )

        if runtime_context.step_runner.plots_visible:
            # Store figure reference - user will call plt.show() at the end
            runtime_context.step_runner._figure_refs.append(fig)
            plt.show()
        else:
            plt.close(fig)

        return context, step_output

    def _create_fold_grid_histogram(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        folds: List[Tuple[List[int], List[int]]]
    ) -> Tuple[Any, str]:
        """Create a grid of histograms showing Y distribution for each fold validation set and test."""
        n_folds = len(folds)

        # Check if test partition exists
        test_context = context.with_partition("test")
        y_test = dataset.y(test_context.selector)
        has_test = y_test is not None and len(y_test) > 0

        # Calculate grid dimensions
        n_plots = n_folds + (1 if has_test else 0)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Flatten axes for easy indexing
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Get train partition context for y values
        train_context = context.with_partition("train")

        # Get base sample IDs for the train partition
        base_sample_ids = dataset._indexer.x_indices(train_context, include_augmented=False)

        # Get all y values for determining common bins
        y_train_all = dataset.y(train_context.selector, include_augmented=False)
        y_train_flat = y_train_all.flatten() if y_train_all.ndim > 1 else y_train_all

        # Combine with test for common range
        if has_test:
            y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
            y_all = np.concatenate([y_train_flat, y_test_flat])
        else:
            y_all = y_train_flat

        # Determine if data is categorical or continuous
        unique_values = np.unique(y_all)
        is_categorical = len(unique_values) <= 20 or y_all.dtype.kind in {'U', 'S', 'O'}

        # Compute common bins for continuous data
        if not is_categorical:
            y_min, y_max = y_all.min(), y_all.max()
            n_bins = min(30, max(10, len(unique_values) // 2))
            common_bins = np.linspace(y_min, y_max, n_bins + 1)
        else:
            common_bins = None

        # Get colormap
        viridis_cmap = cm.get_cmap('viridis')

        # Plot each fold's validation set
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ax = axes[fold_idx]

            if len(val_idx) == 0:
                ax.text(0.5, 0.5, 'No validation samples', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f'Fold {fold_idx + 1} - Validation')
                continue

            # Map fold indices to sample IDs and get y values
            val_idx_arr = np.array(val_idx)
            try:
                val_sample_ids = base_sample_ids[val_idx_arr]
            except IndexError:
                val_sample_ids = val_idx_arr

            y_val = dataset.y({"sample": val_sample_ids.tolist()}, include_augmented=False)
            y_val_flat = y_val.flatten() if y_val.ndim > 1 else y_val

            # Also get train y for this fold (for stacked visualization)
            train_idx_arr = np.array(train_idx)
            try:
                train_sample_ids = base_sample_ids[train_idx_arr]
            except IndexError:
                train_sample_ids = train_idx_arr

            y_train_fold = dataset.y({"sample": train_sample_ids.tolist()}, include_augmented=False)
            y_train_fold_flat = y_train_fold.flatten() if y_train_fold.ndim > 1 else y_train_fold

            # Plot histogram
            if is_categorical:
                self._plot_categorical_fold(ax, y_train_fold_flat, y_val_flat, unique_values, viridis_cmap)
            else:
                self._plot_continuous_fold(ax, y_train_fold_flat, y_val_flat, common_bins, viridis_cmap)

            ax.set_title(f'Fold {fold_idx + 1} - Val (n={len(y_val_flat)})', fontsize=11)
            ax.set_xlabel('Y Values')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot test partition if available
        if has_test:
            ax = axes[n_folds]
            y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test

            # For test, show against the full training set
            if is_categorical:
                self._plot_categorical_fold(ax, y_train_flat, y_test_flat, unique_values, viridis_cmap)
            else:
                self._plot_continuous_fold(ax, y_train_flat, y_test_flat, common_bins, viridis_cmap)

            ax.set_title(f'Test Partition (n={len(y_test_flat)})', fontsize=11, color='darkred')
            ax.set_xlabel('Y Values')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        # Main title
        fig.suptitle(f'Y Distribution: {n_folds} Folds' + (' + Test' if has_test else ''),
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        chart_name = f"Y_distribution_{n_folds}folds" + ("_test" if has_test else "")
        return fig, chart_name

    def _plot_categorical_fold(self, ax, y_train: np.ndarray, y_val: np.ndarray,
                               unique_values: np.ndarray, cmap) -> None:
        """Plot categorical histogram for a single fold."""
        train_counts = np.zeros(len(unique_values))
        val_counts = np.zeros(len(unique_values))

        for i, val in enumerate(unique_values):
            train_counts[i] = np.sum(y_train == val)
            val_counts[i] = np.sum(y_val == val)

        x_pos = np.arange(len(unique_values))
        width = 0.35

        train_color = cmap(0.9)
        val_color = cmap(0.1)

        ax.bar(x_pos - width / 2, train_counts, width, label='Train', color=train_color, alpha=0.7)
        ax.bar(x_pos + width / 2, val_counts, width, label='Val/Test', color=val_color, alpha=0.9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(val) for val in unique_values], rotation=45, fontsize=8)

    def _plot_continuous_fold(self, ax, y_train: np.ndarray, y_val: np.ndarray,
                              bins: np.ndarray, cmap) -> None:
        """Plot continuous histogram for a single fold."""
        train_color = cmap(0.9)
        val_color = cmap(0.1)

        ax.hist(y_train, bins=bins, label='Train', color=train_color, alpha=0.5, edgecolor='none')
        ax.hist(y_val, bins=bins, label='Val/Test', color=val_color, alpha=0.8, edgecolor='none')

    def _create_bicolor_histogram(self, y_train: np.ndarray, y_test: np.ndarray, y_all: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Create a bicolor histogram showing train/test distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))

        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
        y_all_flat = y_all.flatten() if y_all.ndim > 1 else y_all

        # Determine if data is categorical or continuous
        unique_values = np.unique(y_all_flat)
        is_categorical = len(unique_values) <= 20 or y_all_flat.dtype.kind in {'U', 'S', 'O'}

        if is_categorical:
            # Categorical data: grouped bar plot
            self._create_categorical_bicolor_plot(ax, y_train_flat, y_test_flat, unique_values)
            ax.set_xlabel('Y Categories')
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([str(val) for val in unique_values], rotation=45)
            title = 'Y Distribution: Train vs Test (Categorical)'
        else:
            # Continuous data: overlapping histograms
            self._create_continuous_bicolor_plot(ax, y_train_flat, y_test_flat)
            ax.set_xlabel('Y Values')
            title = 'Y Distribution: Train vs Test (Continuous)'

        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics text for both splits with 0.1/0.9 viridis colors
        train_stats = f'Train (n={len(y_train_flat)}):\nMean: {np.mean(y_train_flat):.3f}\nStd: {np.std(y_train_flat):.3f}'
        test_stats = f'Test (n={len(y_test_flat)}):\nMean: {np.mean(y_test_flat):.3f}\nStd: {np.std(y_test_flat):.3f}'

        # Use 0.1/0.9 positions from viridis colormap
        viridis_cmap = cm.get_cmap('viridis')
        train_color = viridis_cmap(0.9)  # Bright yellow-green for train
        test_color = viridis_cmap(0.1)   # Dark purple-blue for test

        ax.text(0.02, 0.98, train_stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=train_color, edgecolor='black'),
                color='black')
        ax.text(0.02, 0.75, test_stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=test_color, edgecolor='white'),
                color='white')

        plot_info = {
            'title': title,
            'figure_size': (12, 6),
            'n_train': len(y_train_flat),
            'n_test': len(y_test_flat)
        }

        return fig, plot_info

    def _create_categorical_bicolor_plot(self, ax, y_train: np.ndarray, y_test: np.ndarray, unique_values: np.ndarray):
        """Create stacked bar plot for categorical data."""
        # Count occurrences for each category in train and test sets
        train_counts = np.zeros(len(unique_values))
        test_counts = np.zeros(len(unique_values))

        for i, val in enumerate(unique_values):
            train_counts[i] = np.sum(y_train == val)
            test_counts[i] = np.sum(y_test == val)

        # Create stacked bars with 0.1/0.9 viridis colors, no borders
        x_pos = np.arange(len(unique_values))
        width = 0.8  # Wider bars for better stacking visibility

        # Use 0.1 and 0.9 positions from viridis colormap
        viridis_cmap = cm.get_cmap('viridis')
        train_color = viridis_cmap(0.9)  # Bright yellow-green
        test_color = viridis_cmap(0.1)   # Dark purple-blue

        # Create stacked bars: TRAIN at bottom, TEST on top, no borders, full color intensity
        bars_train = ax.bar(x_pos, train_counts, width, label='Train',
                            color=train_color)
        bars_test = ax.bar(x_pos, test_counts, width, bottom=train_counts, label='Test',
                           color=test_color)

    def _create_continuous_bicolor_plot(self, ax, y_train: np.ndarray, y_test: np.ndarray):
        """Create overlapping histograms for continuous data."""
        # Handle empty arrays
        if len(y_train) == 0 and len(y_test) == 0:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='red')
            return

        # If one dataset is empty, just plot the other one
        if len(y_train) == 0:
            viridis_cmap = cm.get_cmap('viridis')
            test_color = viridis_cmap(0.1)  # Dark purple-blue for test
            ax.hist(y_test, bins=30, label='Test', color=test_color)
            return

        if len(y_test) == 0:
            viridis_cmap = cm.get_cmap('viridis')
            train_color = viridis_cmap(0.9)  # Bright yellow-green for train
            ax.hist(y_train, bins=30, label='Train', color=train_color)
            return        # Determine common bin edges for both distributions
        y_min = min(np.min(y_train), np.min(y_test))
        y_max = max(np.max(y_train), np.max(y_test))

        n_bins = min(30, max(10, len(np.unique(np.concatenate([y_train, y_test]))) // 2))
        bins = np.linspace(y_min, y_max, n_bins + 1)

        # Create overlapping histograms with 0.1/0.9 viridis colors, no borders
        viridis_cmap = cm.get_cmap('viridis')
        train_color = viridis_cmap(0.9)  # Bright yellow-green
        test_color = viridis_cmap(0.1)   # Dark purple-blue

        ax.hist(y_train, bins=bins, label='Train', color=train_color)
        ax.hist(y_test, bins=bins, label='Test', color=test_color)

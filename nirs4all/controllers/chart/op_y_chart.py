"""YChartController - Y values histogram visualization with train/test split."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
import io

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

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
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute y values histogram visualization with train/test split.
        Skips execution in prediction mode.

        Returns:
            Tuple of (context, image_list) where image_list contains plot metadata
        """
        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, []

        # Initialize image list to track generated plots
        img_list = []

        local_context = context.copy()
        y = dataset.y(local_context)
        local_context["partition"] = "train"
        y_train = dataset.y(local_context)
        local_context["partition"] = "test"
        y_test = dataset.y(local_context)

        # print(len(y), len(y_train), len(y_test))

        fig, _ = self._create_bicolor_histogram(y_train, y_test, y)
        chart_name = "Y_distribution_train_test.png"

        # Save plot to memory buffer as PNG binary
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        img_list.append((chart_name, img_png_binary))

        if runner.plots_visible:
            # Store figure reference - user will call plt.show() at the end
            runner._figure_refs.append(fig)
            plt.show()
        else:
            plt.close(fig)

        return context, img_list

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

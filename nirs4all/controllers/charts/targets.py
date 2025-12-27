"""YChartController - Y values histogram visualization with train/test split and folds."""

from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING
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

        Supports optional parameters via dict syntax:
            {"chart_y": {"include_excluded": True, "highlight_excluded": True}}

        Args:
            include_excluded: If True, include excluded samples in visualization
            highlight_excluded: If True, show excluded samples as separate histogram

        Returns:
            Tuple of (context, StepOutput)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, StepOutput()

        # Extract configuration from step
        step = step_info.original_step
        include_excluded = False
        highlight_excluded = False
        layout = 'standard'

        if isinstance(step, dict):
            for key in ["y_chart", "chart_y"]:
                if key in step:
                    config = step[key] if isinstance(step[key], dict) else {}
                    include_excluded = config.get("include_excluded", False)
                    highlight_excluded = config.get("highlight_excluded", False)
                    layout = config.get("layout", "standard")
                    if layout not in ('standard', 'stacked', 'staggered'):
                        raise ValueError(f"Unknown layout: {layout}. Use 'standard', 'stacked', or 'staggered'.")
                    break

        # Get folds from dataset
        folds = dataset.folds

        # Check if we have multiple CV folds (not just a single train/test split)
        has_cv_folds = folds is not None and len(folds) > 1

        if has_cv_folds:
            # Grid mode: show each fold's validation set + test partition
            fig, chart_name = self._create_fold_grid_histogram(dataset, context, folds, layout=layout)
        else:
            # Simple mode: train vs test
            local_context = context.with_partition("train")

            # Get y values with optional excluded samples
            if include_excluded:
                train_indices = dataset._indexer.x_indices(  # noqa: SLF001
                    local_context.selector, include_augmented=True, include_excluded=True
                )
                y_train = dataset.y(
                    {"sample": train_indices.tolist(), "y": local_context.state.y_processing},
                    include_excluded=True
                )

                # Get excluded y values separately if highlighting
                if highlight_excluded:
                    included_indices = dataset._indexer.x_indices(  # noqa: SLF001
                        local_context.selector, include_augmented=True, include_excluded=False
                    )
                    excluded_mask = np.isin(train_indices, included_indices, invert=True)
                    excluded_indices = train_indices[excluded_mask]
                    if len(excluded_indices) > 0:
                        y_train_excluded = dataset.y(
                            {"sample": excluded_indices.tolist(), "y": local_context.state.y_processing},
                            include_excluded=True
                        )
                    else:
                        y_train_excluded = np.array([])
                else:
                    y_train_excluded = None
            else:
                y_train = dataset.y(local_context)
                y_train_excluded = None

            local_context = context.with_partition("test")
            if include_excluded:
                test_indices = dataset._indexer.x_indices(  # noqa: SLF001
                    local_context.selector, include_augmented=True, include_excluded=True
                )
                y_test = dataset.y(
                    {"sample": test_indices.tolist(), "y": local_context.state.y_processing},
                    include_excluded=True
                ) if len(test_indices) > 0 else np.array([])
            else:
                y_test = dataset.y(local_context)

            y_all = dataset.y(context, include_excluded=include_excluded)

            fig, _ = self._create_bicolor_histogram(y_train, y_test, y_all, y_excluded=y_train_excluded, layout=layout)
            chart_name = "Y_distribution_train_test"
            if include_excluded:
                chart_name += "_with_excluded"
            if layout != 'standard':
                chart_name += f"_{layout}"

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
        folds: List[Tuple[List[int], List[int]]],
        layout: Literal['standard', 'stacked', 'staggered'] = 'standard'
    ) -> Tuple[Any, str]:
        """Create a grid of histograms showing Y distribution for each fold validation set and test."""
        n_folds = len(folds)

        # Check if test partition exists
        test_context = context.with_partition("test")
        y_test = dataset.y(test_context)
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
        y_train_all = dataset.y(train_context, include_augmented=False)
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
        viridis_cmap = plt.colormaps['viridis']

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

            y_val = dataset.y({"sample": val_sample_ids.tolist(), "y": context.state.y_processing}, include_augmented=False)
            y_val_flat = y_val.flatten() if y_val.ndim > 1 else y_val

            # Also get train y for this fold (for stacked visualization)
            train_idx_arr = np.array(train_idx)
            try:
                train_sample_ids = base_sample_ids[train_idx_arr]
            except IndexError:
                train_sample_ids = train_idx_arr

            y_train_fold = dataset.y({"sample": train_sample_ids.tolist(), "y": context.state.y_processing}, include_augmented=False)
            y_train_fold_flat = y_train_fold.flatten() if y_train_fold.ndim > 1 else y_train_fold

            # Plot histogram
            if is_categorical:
                self._plot_categorical_fold(ax, y_train_fold_flat, y_val_flat, unique_values, viridis_cmap, layout=layout)
            else:
                self._plot_continuous_fold(ax, y_train_fold_flat, y_val_flat, common_bins, viridis_cmap, layout=layout)

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
                self._plot_categorical_fold(ax, y_train_flat, y_test_flat, unique_values, viridis_cmap, layout=layout)
            else:
                self._plot_continuous_fold(ax, y_train_flat, y_test_flat, common_bins, viridis_cmap, layout=layout)

            ax.set_title(f'Test Partition (n={len(y_test_flat)})', fontsize=11, color='darkred')
            ax.set_xlabel('Y Values')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        # Main title
        layout_note = f' [{layout}]' if layout != 'standard' else ''
        fig.suptitle(f'Y Distribution: {n_folds} Folds{layout_note}' + (' + Test' if has_test else ''),
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        chart_name = f"Y_distribution_{n_folds}folds" + ("_test" if has_test else "")
        if layout != 'standard':
            chart_name += f"_{layout}"
        return fig, chart_name

    def _plot_categorical_fold(self, ax, y_train: np.ndarray, y_val: np.ndarray,
                               unique_values: np.ndarray, cmap,
                               layout: Literal['standard', 'stacked', 'staggered'] = 'standard') -> None:
        """Plot categorical histogram for a single fold.

        Args:
            ax: Matplotlib axes object.
            y_train: Training y values.
            y_val: Validation/test y values.
            unique_values: Unique category values.
            cmap: Colormap to use.
            layout: Layout style ('standard', 'stacked', 'staggered').
        """
        train_counts = np.zeros(len(unique_values))
        val_counts = np.zeros(len(unique_values))

        for i, val in enumerate(unique_values):
            train_counts[i] = np.sum(y_train == val)
            val_counts[i] = np.sum(y_val == val)

        x_pos = np.arange(len(unique_values))
        train_color = cmap(0.9)
        val_color = cmap(0.1)

        if layout == 'staggered':
            # Side-by-side bars
            width = 0.35
            ax.bar(x_pos - width / 2, train_counts, width, label='Train', color=train_color, alpha=0.7)
            ax.bar(x_pos + width / 2, val_counts, width, label='Val/Test', color=val_color, alpha=0.9)
        elif layout == 'stacked':
            # Stacked bars
            width = 0.6
            ax.bar(x_pos, train_counts, width, label='Train', color=train_color, alpha=0.7)
            ax.bar(x_pos, val_counts, width, bottom=train_counts, label='Val/Test', color=val_color, alpha=0.9)
        else:  # standard - overlapping (same as staggered for categorical)
            width = 0.35
            ax.bar(x_pos - width / 2, train_counts, width, label='Train', color=train_color, alpha=0.7)
            ax.bar(x_pos + width / 2, val_counts, width, label='Val/Test', color=val_color, alpha=0.9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(val) for val in unique_values], rotation=45, fontsize=8)

    def _plot_continuous_fold(self, ax, y_train: np.ndarray, y_val: np.ndarray,
                              bins: np.ndarray, cmap,
                              layout: Literal['standard', 'stacked', 'staggered'] = 'standard') -> None:
        """Plot continuous histogram for a single fold.

        Args:
            ax: Matplotlib axes object.
            y_train: Training y values.
            y_val: Validation/test y values.
            bins: Bin edges for histogram.
            cmap: Colormap to use.
            layout: Layout style ('standard', 'stacked', 'staggered').
        """
        train_color = cmap(0.9)
        val_color = cmap(0.1)

        if layout == 'stacked':
            # Stacked histograms
            ax.hist([y_train, y_val], bins=bins, label=['Train', 'Val/Test'],
                    color=[train_color, val_color], alpha=0.8, edgecolor='none',
                    histtype='barstacked')
        elif layout == 'staggered':
            # Side-by-side bars using numpy histogram and bar plot
            train_counts, bin_edges = np.histogram(y_train, bins=bins)
            val_counts, _ = np.histogram(y_val, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            bar_width = bin_width * 0.4
            ax.bar(bin_centers - bar_width / 2, train_counts, bar_width,
                   label='Train', color=train_color, alpha=0.7)
            ax.bar(bin_centers + bar_width / 2, val_counts, bar_width,
                   label='Val/Test', color=val_color, alpha=0.9)
        else:  # standard - overlapping
            ax.hist(y_train, bins=bins, label='Train', color=train_color, alpha=0.5, edgecolor='none')
            ax.hist(y_val, bins=bins, label='Val/Test', color=val_color, alpha=0.8, edgecolor='none')

    def _create_bicolor_histogram(self, y_train: np.ndarray, y_test: np.ndarray, y_all: np.ndarray,
                                   y_excluded: Optional[np.ndarray] = None,
                                   layout: Literal['standard', 'stacked', 'staggered'] = 'standard'
                                   ) -> Tuple[Any, Dict[str, Any]]:
        """
        Create a bicolor histogram showing train/test distribution.

        Args:
            y_train: Train partition y values
            y_test: Test partition y values
            y_all: All y values (for range calculation)
            y_excluded: Optional excluded samples y values for highlighting
            layout: Layout style ('standard', 'stacked', 'staggered')
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
        y_all_flat = y_all.flatten() if y_all.ndim > 1 else y_all

        # Flatten excluded if provided
        y_excluded_flat = None
        if y_excluded is not None and len(y_excluded) > 0:
            y_excluded_flat = y_excluded.flatten() if y_excluded.ndim > 1 else y_excluded

        # Determine if data is categorical or continuous
        unique_values = np.unique(y_all_flat)
        is_categorical = len(unique_values) <= 20 or y_all_flat.dtype.kind in {'U', 'S', 'O'}

        if is_categorical:
            # Categorical data: grouped bar plot
            self._create_categorical_bicolor_plot(ax, y_train_flat, y_test_flat, unique_values, y_excluded_flat, layout=layout)
            ax.set_xlabel('Y Categories')
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([str(val) for val in unique_values], rotation=45)
            layout_note = f' [{layout}]' if layout != 'standard' else ''
            title = f'Y Distribution: Train vs Test (Categorical){layout_note}'
        else:
            # Continuous data: overlapping histograms
            self._create_continuous_bicolor_plot(ax, y_train_flat, y_test_flat, y_excluded_flat, layout=layout)
            ax.set_xlabel('Y Values')
            layout_note = f' [{layout}]' if layout != 'standard' else ''
            title = f'Y Distribution: Train vs Test (Continuous){layout_note}'

        if y_excluded_flat is not None and len(y_excluded_flat) > 0:
            title += f' [Excluded: {len(y_excluded_flat)}]'

        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics text for both splits with 0.1/0.9 viridis colors
        train_stats = f'Train (n={len(y_train_flat)}):\nMean: {np.mean(y_train_flat):.3f}\nStd: {np.std(y_train_flat):.3f}'
        test_stats = f'Test (n={len(y_test_flat)}):\nMean: {np.mean(y_test_flat):.3f}\nStd: {np.std(y_test_flat):.3f}'

        # Use 0.1/0.9 positions from viridis colormap
        viridis_cmap = plt.colormaps['viridis']
        train_color = viridis_cmap(0.9)  # Bright yellow-green for train
        test_color = viridis_cmap(0.1)   # Dark purple-blue for test

        ax.text(0.02, 0.98, train_stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=train_color, edgecolor='black'),
                color='black')
        ax.text(0.02, 0.75, test_stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=test_color, edgecolor='white'),
                color='white')

        # Add excluded stats if present
        if y_excluded_flat is not None and len(y_excluded_flat) > 0:
            excluded_stats = f'Excluded (n={len(y_excluded_flat)}):\nMean: {np.mean(y_excluded_flat):.3f}\nStd: {np.std(y_excluded_flat):.3f}'
            ax.text(0.02, 0.52, excluded_stats, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7, edgecolor='darkred'),
                    color='white')

        plot_info = {
            'title': title,
            'figure_size': (12, 6),
            'n_train': len(y_train_flat),
            'n_test': len(y_test_flat),
            'n_excluded': len(y_excluded_flat) if y_excluded_flat is not None else 0
        }

        return fig, plot_info

    def _create_categorical_bicolor_plot(self, ax, y_train: np.ndarray, y_test: np.ndarray,
                                          unique_values: np.ndarray,
                                          y_excluded: Optional[np.ndarray] = None,
                                          layout: Literal['standard', 'stacked', 'staggered'] = 'standard'):
        """Create bar plot for categorical data.

        Args:
            ax: Matplotlib axes object.
            y_train: Training y values.
            y_test: Test y values.
            unique_values: Unique category values.
            y_excluded: Optional excluded y values.
            layout: Layout style ('standard', 'stacked', 'staggered').
        """
        # Count occurrences for each category in train and test sets
        train_counts = np.zeros(len(unique_values))
        test_counts = np.zeros(len(unique_values))
        excluded_counts = np.zeros(len(unique_values))

        for i, val in enumerate(unique_values):
            train_counts[i] = np.sum(y_train == val)
            test_counts[i] = np.sum(y_test == val)
            if y_excluded is not None:
                excluded_counts[i] = np.sum(y_excluded == val)

        x_pos = np.arange(len(unique_values))

        # Use 0.1 and 0.9 positions from viridis colormap
        viridis_cmap = plt.colormaps['viridis']
        train_color = viridis_cmap(0.9)  # Bright yellow-green
        test_color = viridis_cmap(0.1)   # Dark purple-blue

        if layout == 'staggered':
            # Side-by-side bars
            width = 0.35
            ax.bar(x_pos - width / 2, train_counts, width, label='Train', color=train_color)
            ax.bar(x_pos + width / 2, test_counts, width, label='Test', color=test_color)
            # Add excluded bars if present (offset further)
            if y_excluded is not None and np.sum(excluded_counts) > 0:
                ax.bar(x_pos + width * 1.5, excluded_counts, width, label='Excluded',
                       color='red', alpha=0.7, hatch='//')
        elif layout == 'stacked':
            # Stacked bars
            width = 0.6
            ax.bar(x_pos, train_counts, width, label='Train', color=train_color)
            ax.bar(x_pos, test_counts, width, bottom=train_counts, label='Test', color=test_color)
            # Add excluded bars if present (stacked on top)
            if y_excluded is not None and np.sum(excluded_counts) > 0:
                ax.bar(x_pos, excluded_counts, width, bottom=train_counts + test_counts,
                       label='Excluded', color='red', alpha=0.7, hatch='//')
        else:  # standard - same as original stacked behavior
            width = 0.8
            ax.bar(x_pos, train_counts, width, label='Train', color=train_color)
            ax.bar(x_pos, test_counts, width, bottom=train_counts, label='Test', color=test_color)
            # Add excluded bars if present (with hatching pattern)
            if y_excluded is not None and np.sum(excluded_counts) > 0:
                ax.bar(x_pos, excluded_counts, width, bottom=train_counts + test_counts,
                       label='Excluded', color='red', alpha=0.7, hatch='//')

    def _create_continuous_bicolor_plot(self, ax, y_train: np.ndarray, y_test: np.ndarray,
                                         y_excluded: Optional[np.ndarray] = None,
                                         layout: Literal['standard', 'stacked', 'staggered'] = 'standard'):
        """Create histograms for continuous data.

        Args:
            ax: Matplotlib axes object.
            y_train: Training y values.
            y_test: Test y values.
            y_excluded: Optional excluded y values.
            layout: Layout style ('standard', 'stacked', 'staggered').
        """
        # Handle empty arrays
        if len(y_train) == 0 and len(y_test) == 0:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='red')
            return

        # Collect all data for bin calculation
        all_data = []
        if len(y_train) > 0:
            all_data.append(y_train)
        if len(y_test) > 0:
            all_data.append(y_test)
        if y_excluded is not None and len(y_excluded) > 0:
            all_data.append(y_excluded)

        combined = np.concatenate(all_data) if all_data else np.array([])

        # If one dataset is empty, just plot the other one
        if len(y_train) == 0:
            viridis_cmap = plt.colormaps['viridis']
            test_color = viridis_cmap(0.1)  # Dark purple-blue for test
            bins = np.linspace(np.min(y_test), np.max(y_test), 31)
            ax.hist(y_test, bins=bins, label='Test', color=test_color, alpha=0.7)
            # Add excluded histogram if present (with hatching)
            if y_excluded is not None and len(y_excluded) > 0:
                ax.hist(y_excluded, bins=bins, label='Excluded', color='red', alpha=0.5, hatch='//')
            return

        if len(y_test) == 0:
            viridis_cmap = plt.colormaps['viridis']
            train_color = viridis_cmap(0.9)  # Bright yellow-green for train
            bins = np.linspace(np.min(y_train), np.max(y_train), 31)
            ax.hist(y_train, bins=bins, label='Train', color=train_color, alpha=0.7)
            # Add excluded histogram if present (with hatching)
            if y_excluded is not None and len(y_excluded) > 0:
                ax.hist(y_excluded, bins=bins, label='Excluded', color='red', alpha=0.5, hatch='//')
            return

        # Determine common bin edges for all distributions
        y_min = np.min(combined)
        y_max = np.max(combined)

        n_bins = min(30, max(10, len(np.unique(combined)) // 2))
        bins = np.linspace(y_min, y_max, n_bins + 1)

        # Create histograms with 0.1/0.9 viridis colors based on layout
        viridis_cmap = plt.colormaps['viridis']
        train_color = viridis_cmap(0.9)  # Bright yellow-green
        test_color = viridis_cmap(0.1)   # Dark purple-blue

        if layout == 'stacked':
            # Stacked histograms
            ax.hist([y_train, y_test], bins=bins, label=['Train', 'Test'],
                    color=[train_color, test_color], alpha=0.8, histtype='barstacked')
            # Add excluded histogram if present (on top)
            if y_excluded is not None and len(y_excluded) > 0:
                ax.hist(y_excluded, bins=bins, label='Excluded', color='red', alpha=0.5, hatch='//')
        elif layout == 'staggered':
            # Side-by-side bars using numpy histogram and bar plot
            train_counts, bin_edges = np.histogram(y_train, bins=bins)
            test_counts, _ = np.histogram(y_test, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            bar_width = bin_width * 0.4
            ax.bar(bin_centers - bar_width / 2, train_counts, bar_width,
                   label='Train', color=train_color, alpha=0.7)
            ax.bar(bin_centers + bar_width / 2, test_counts, bar_width,
                   label='Test', color=test_color, alpha=0.9)
            # Add excluded bars if present
            if y_excluded is not None and len(y_excluded) > 0:
                excluded_counts, _ = np.histogram(y_excluded, bins=bins)
                ax.bar(bin_centers + bar_width * 1.5, excluded_counts, bar_width,
                       label='Excluded', color='red', alpha=0.5, hatch='//')
        else:  # standard - overlapping
            ax.hist(y_train, bins=bins, label='Train', color=train_color, alpha=0.7)
            ax.hist(y_test, bins=bins, label='Test', color=test_color, alpha=0.7)
            # Add excluded histogram if present (with hatching)
            if y_excluded is not None and len(y_excluded) > 0:
                ax.hist(y_excluded, bins=bins, label='Excluded', color='red', alpha=0.5, hatch='//')

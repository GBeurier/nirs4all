"""FoldChartController - Visualizes cross-validation folds with y-value color coding."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Union
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import copy
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.config.context import ExecutionContext
import io

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.config.context import ExecutionContext


@register_controller
class FoldChartController(OperatorController):

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "fold_chart" or keyword == "chart_fold" or keyword.startswith("fold_")

    @classmethod
    def use_multi_source(cls) -> bool:
        return False  # Fold visualization is dataset-wide, not source-specific

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
        Execute fold visualization showing train/test splits with y-value color coding.
        Skips execution in prediction mode.

        Returns:
            Tuple of (context, StepOutput)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, StepOutput()

        # print(f"Executing fold charts for step: {step}, keyword: {context.metadata.keyword}")

        # Check if using metadata column for colors (keyword like "chart_columnName")
        keyword = context.metadata.keyword
        metadata_column = None
        if keyword.startswith("fold_") and keyword != "chart_fold" and keyword != "fold_chart":
            metadata_column = keyword[5:]  # Extract column name after "fold_"
            if runtime_context.step_runner.verbose > 0:
                logger.info(f"Using metadata column '{metadata_column}' for color coding")

        # Determine which partition to use (default to train if not specified)
        partition = context.selector.partition or "train"
        if partition not in ["train", "test"]:
            logger.warning(f"Invalid partition '{partition}'. Using 'train' instead.")
            partition = "train"

        # Get data for visualization
        local_context = context.with_partition(partition)

        # Get base sample IDs for the partition used for splitting (usually train)
        # This maps relative indices (0..N) to actual sample IDs
        base_sample_ids = dataset._indexer.x_indices(local_context, include_augmented=False)

        # Get folds from dataset
        folds = dataset.folds

        # Detect if we have a simple train/test split (1 fold) vs actual CV folds (multiple folds)
        # Single fold from SPXYSplitter should be treated as train/test split, not CV
        is_simple_split = folds is not None and len(folds) == 1
        original_folds_for_chart = dataset.folds  # Keep track of original for later

        # For simple train/test split, check if test partition actually exists
        # (could be that the fold created test, or test was created separately)
        if is_simple_split:
            test_ctx = {"partition": "test"}
            test_indices = dataset._indexer.x_indices(test_ctx, include_augmented=True)
            # If test exists and fold has a non-empty test set, this is a simple split
            # We should visualize it as train/test bars, not as a "fold"
            if len(test_indices) > 0 and len(folds[0][1]) > 0:
                # This is a train/test split from SPXYSplitter - treat as no CV folds
                # We'll reconstruct the visualization to show train and test as separate bars
                folds = None  # Clear folds to trigger fallback logic
                original_folds_for_chart = None  # Don't pass original folds to chart (not CV mode)

        # Fallback logic: If no folds, create a simple train/test split visualization
        if not folds:
            # print(f"{INFO}No CV folds found. Creating visualization from train/test partition.")

            # Try to get train and test data - INCLUDE AUGMENTED SAMPLES
            train_context = context.with_partition("train")
            test_context = context.with_partition("test")

            train_indices = dataset._indexer.x_indices(train_context, include_augmented=True)  # noqa: SLF001
            test_indices = dataset._indexer.x_indices(test_context, include_augmented=True)  # noqa: SLF001

            if len(test_indices) > 0:
                # We have both train and test data - create a single "fold" with absolute indices
                # Use absolute indices from the concatenated array: train is [0..N], test is [N..N+M]
                train_abs = list(range(len(train_indices)))
                test_abs = list(range(len(train_indices), len(train_indices) + len(test_indices)))
                folds = [(train_abs, test_abs)]
                logger.debug(f"  Using train ({len(train_indices)} samples including augmented) and test ({len(test_indices)} samples) partitions.")
            elif len(train_indices) > 0:
                # Only train data exists - show it as a single bar
                folds = [(list(range(len(train_indices))), [])]
                logger.debug(f"  Only train partition available ({len(train_indices)} samples including augmented).")
            else:
                logger.warning("No data available for visualization.")
                return context, StepOutput()

        # Get values for color coding (either y or metadata column)
        # For CV folds: get all data for proper indexing
        # For fallback (train/test): need to handle both base and augmented samples
        if metadata_column:
            # Use metadata column for colors
            if dataset.folds:
                # CV folds mode: need all data including augmented since indices refer to full dataset
                # Create selector without partition constraint to get all data
                all_selector = context.selector.with_partition(None)
                color_values = dataset.metadata_column(metadata_column, all_selector, include_augmented=True)

            else:
                train_ctx = context.with_partition("train")
                test_ctx = context.with_partition("test")

                # Get metadata using origin mapping (like y() does)
                train_x_idx = dataset._indexer.x_indices(train_ctx, include_augmented=True)
                test_x_idx = dataset._indexer.x_indices(test_ctx, include_augmented=True)

                # Map to origins and get metadata
                base_meta = dataset._metadata.get_column(metadata_column)
                train_origin_indices = np.array([dataset._indexer.get_origin_for_sample(int(idx)) for idx in train_x_idx])
                test_origin_indices = np.array([dataset._indexer.get_origin_for_sample(int(idx)) for idx in test_x_idx])

                meta_train = base_meta[train_origin_indices]
                meta_test = base_meta[test_origin_indices]

                # Concatenate train and test for visualization
                if len(meta_test) > 0:
                    color_values = np.concatenate([meta_train, meta_test])
                else:
                    color_values = meta_train

        else:
            # Use y values for colors (default behavior)
            if dataset.folds:
                # CV folds mode: need all data including augmented since indices refer to full dataset
                # Create selector without partition constraint to get all data
                all_selector = context.selector.with_partition(None)
                color_values = dataset.y(all_selector, include_augmented=True)
            else:
                # Fallback mode: get train and test separately and concatenate
                train_ctx = context.with_partition("train")
                test_ctx = context.with_partition("test")

                y_train = dataset.y(train_ctx, include_augmented=True)
                y_test = dataset.y(test_ctx, include_augmented=True)

                # Concatenate train and test for visualization
                if len(y_test) > 0:
                    color_values = np.concatenate([y_train, y_test])
                else:
                    color_values = y_train

        color_values_flat = color_values.flatten() if color_values.ndim > 1 else color_values

        # --- Debug Print ---
        logger.debug("--- Fold Chart Class Distribution (Train Partition) ---")
        # Always fetch train partition for this debug print to compare with sample augmentation
        train_debug_context = context.with_partition("train")

        if metadata_column:
            debug_values = dataset.metadata_column(metadata_column, train_debug_context.selector, include_augmented=True)
        else:
            debug_values = dataset.y(train_debug_context.selector, include_augmented=True)

        debug_values_flat = debug_values.flatten() if debug_values.ndim > 1 else debug_values

        unique_vals = np.unique(debug_values_flat)
        if len(unique_vals) < 50:
            counts = Counter(debug_values_flat)
            for label, count in sorted(counts.items()):
                logger.debug(f"  Class {label}: {count}")
        else:
            logger.debug(f"  Continuous values: {len(debug_values_flat)} samples, {len(unique_vals)} unique values.")
        logger.debug("-------------------------------------")
        # -------------------

        # Create fold visualization
        # Pass original_folds_for_chart (which is None for simple splits) instead of dataset.folds
        fig, plot_info = self._create_fold_chart(folds, color_values_flat, len(color_values_flat), partition, original_folds_for_chart, dataset, metadata_column, base_sample_ids)

        # Save plot to memory buffer as PNG binary
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        # Create filename with partition info
        fold_suffix = f"{len(folds)}folds" if dataset.folds else "traintest_split"
        metadata_suffix = f"_{metadata_column}" if metadata_column else ""
        image_name = f"fold_visualization_{fold_suffix}_{partition}{metadata_suffix}"

        # Create StepOutput with the chart
        step_output = StepOutput(
            outputs=[(img_png_binary, image_name, "png")]
        )

        if runtime_context.step_runner.plots_visible:
            # Store figure reference - user will call plt.show() at the end
            runtime_context.step_runner._figure_refs.append(fig)
            plt.show()
        else:
            plt.close(fig)

        return context, step_output

    def _create_fold_chart(self, folds: List[Tuple[List[int], List[int]]], y_values: np.ndarray, n_samples: int, partition: str = "train",
                           original_folds: List = None, dataset: 'SpectroDataset' = None, metadata_column: str = None,
                           base_sample_ids: np.ndarray = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Create a fold visualization chart with stacked bars showing y-value distribution.

        Args:
            folds: List of (train_indices, test_indices) tuples
            y_values: Target values for color coding
            n_samples: Total number of samples
            partition: Which partition to visualize ('train' or 'test')
            original_folds: Original folds from dataset (to distinguish CV from simple split)
            dataset: The dataset object (used to check for test partition when CV folds exist)
            metadata_column: Optional metadata column name to use for color coding instead of y values

        Returns:
            Tuple of (figure, plot_info)
        """
        n_folds = len(folds)
        is_cv_folds = original_folds is not None and len(original_folds) > 0

        # Check if there's a test partition to display (when CV folds exist)
        test_partition_indices = None
        if is_cv_folds and dataset is not None:
            test_ctx = {"partition": "test"}
            test_partition_indices = dataset._indexer.x_indices(test_ctx)
            if len(test_partition_indices) > 0:
                test_partition_indices = test_partition_indices.tolist()
            else:
                test_partition_indices = None

        # Calculate figure width including test partition if present
        extra_bars = 1 if test_partition_indices else 0
        fig_width = max(12, (n_folds + extra_bars) * 3)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

        # Create colormap - use discrete contrastive colors for metadata and classification, continuous for regression

        # Check if we should use discrete colormap (metadata column or classification task)
        is_classification_task = dataset is not None and dataset.is_classification
        use_discrete_colormap = metadata_column is not None or is_classification_task

        if use_discrete_colormap:
            # For metadata or classification: use discrete, highly contrastive colors
            # Get unique values and create a discrete colormap
            unique_values = np.unique(y_values)
            n_unique = len(unique_values)

            # Use a highly contrastive colormap (tab20, tab20b, tab20c for many categories)
            if n_unique <= 10:
                colormap = plt.colormaps['tab10'].resampled(n_unique)
            elif n_unique <= 20:
                colormap = plt.colormaps['tab20'].resampled(n_unique)
            else:
                # For many categories, use a combination or hsv
                colormap = plt.colormaps['hsv'].resampled(n_unique)

            # Create a mapping from actual values to discrete indices
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}

            # Normalize to discrete indices [0, 1, 2, ...] / n_unique
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_values])

            # For discrete values, y_min and y_max are index boundaries
            y_min, y_max = 0, n_unique - 1
        else:
            # For continuous y values (regression): use continuous colormap
            y_min, y_max = y_values.min(), y_values.max()
            colormap = plt.colormaps['viridis']
            # Normalize y values to [0, 1] for colormap
            if y_max != y_min:
                y_normalized = (y_values - y_min) / (y_max - y_min)
            else:
                y_normalized = np.zeros_like(y_values)

        bar_width = 0.8
        gap_between_folds = 0.4

        # Prepare discrete-value-specific parameters (metadata or classification)
        is_discrete_values = use_discrete_colormap  # Reuse the flag we already computed
        value_to_index_map = None
        n_unique_values = 1

        if is_discrete_values:
            unique_values = np.unique(y_values)
            n_unique_values = len(unique_values)
            value_to_index_map = {val: idx for idx, val in enumerate(unique_values)}

        # Get train y-indices including augmented samples (mapped to their origins)
        y_indices = dataset._indexer.y_indices({"partition": "train"}, include_augmented=True)  # noqa: SLF001

        # For fallback mode, y_values contains augmented samples, so we need to handle the mapping differently
        # Get the label from y_values for train partition only (not test)
        if not is_cv_folds and len(y_values) != len(y_indices):
            # Fallback mode with augmented samples: y_values may include augmented samples
            # Get the label from y_values for train partition only (not test)
            train_len = len(folds[0][0]) if folds else len(y_indices)
            y_meta = y_values[:train_len]
        else:
            # CV folds or no augmentation: use direct indexing with y_indices
            y_meta = y_values[y_indices]

        # unique_meta, counts = np.unique(y_meta, return_counts=True)
        # for val, count in zip(unique_meta, counts):
        #     print(f"{val}: {count}")
        # print('-' * 20)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Position des barres pour ce fold
            base_pos = fold_idx * (2 + gap_between_folds)
            train_pos = base_pos
            test_pos = base_pos + 1

            # Get y-values for train and test
            # In CV mode: expand fold indices to include augmented samples, then map to origins
            # In fallback mode: direct indexing (indices already into concatenated array)
            if is_cv_folds:
                # CV folds mode: folds contain base sample indices. Expand to include augmented samples.
                train_idx_arr = np.array(train_idx) if isinstance(train_idx, list) else train_idx
                test_idx_arr = np.array(test_idx) if isinstance(test_idx, list) else test_idx

                # Map relative indices to actual sample IDs (origins)
                if base_sample_ids is not None:
                    try:
                        train_origins = base_sample_ids[train_idx_arr]
                        test_origins = base_sample_ids[test_idx_arr]
                    except IndexError:
                        # Fallback if indices don't match
                        logger.warning(f"Fold indices out of bounds for partition '{partition}'. Using indices as origins.")
                        train_origins = train_idx_arr
                        test_origins = test_idx_arr
                else:
                    train_origins = train_idx_arr
                    test_origins = test_idx_arr

                train_idx_list = train_origins.tolist() if hasattr(train_origins, 'tolist') else list(train_origins)
                test_idx_list = test_origins.tolist() if hasattr(test_origins, 'tolist') else list(test_origins)

                # Expand to include augmented samples for each base sample in the fold
                train_augmented = dataset._indexer.get_augmented_for_origins(train_idx_list)  # noqa: SLF001
                test_augmented = dataset._indexer.get_augmented_for_origins(test_idx_list)  # noqa: SLF001

                # Combine base and augmented samples
                train_all_idx = train_idx_list + train_augmented.tolist()
                test_all_idx = test_idx_list + test_augmented.tolist()

                # Fetch values directly using sample IDs
                if metadata_column:
                    train_y = dataset.metadata_column(metadata_column, {"sample": train_all_idx}, include_augmented=True)
                    test_y = dataset.metadata_column(metadata_column, {"sample": test_all_idx}, include_augmented=True)
                else:
                    train_y = dataset.y({"sample": train_all_idx}, include_augmented=True)
                    test_y = dataset.y({"sample": test_all_idx}, include_augmented=True)

                # Ensure 1D arrays
                if train_y.ndim > 1:
                    train_y = train_y.flatten()
                if test_y.ndim > 1:
                    test_y = test_y.flatten()
            else:
                # Fallback mode: use direct indexing (fold indices are indices into concatenated array)
                train_y = y_values[train_idx]
                test_y = y_values[test_idx] if len(test_idx) > 0 else np.array([])

            # Sort for visualization
            train_sorted_indices = np.argsort(train_y)
            train_y_sorted = train_y[train_sorted_indices]

            # Traiter les données de test
            if len(test_y) > 0:
                test_sorted_indices = np.argsort(test_y)
                test_y_sorted = test_y[test_sorted_indices]
            else:
                test_y_sorted = np.array([])

            # Créer les barres empilées pour TRAIN
            self._create_stacked_bar(ax, train_pos, train_y_sorted, colormap,
                                   y_min, y_max, bar_width, f'Train F{fold_idx}',
                                   is_discrete_values, value_to_index_map, n_unique_values)

            # Créer les barres empilées pour TEST (only if test data exists)
            if len(test_idx) > 0:
                self._create_stacked_bar(ax, test_pos, test_y_sorted, colormap,
                                       y_min, y_max, bar_width, f'Test F{fold_idx}',
                                       is_discrete_values, value_to_index_map, n_unique_values)

            # Ajouter les labels au-dessus des barres
            train_label = 'Train' if not is_cv_folds else f'T{fold_idx}'
            ax.text(train_pos, len(train_y) + 1, f'{train_label}\n({len(train_y)})',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

            if len(test_idx) > 0:
                test_label = 'Test' if not is_cv_folds else f'V{fold_idx}'
                ax.text(test_pos, len(test_y) + 1, f'{test_label}\n({len(test_y)})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add test partition bar if CV folds exist and test data is available
        if test_partition_indices:
            # Position after all folds
            test_partition_pos = n_folds * (2 + gap_between_folds)

            # Get test partition y values
            test_partition_y = y_values[test_partition_indices]
            test_sorted_indices = np.argsort(test_partition_y)
            test_y_sorted = test_partition_y[test_sorted_indices]

            # Create stacked bar for test partition
            self._create_stacked_bar(ax, test_partition_pos, test_y_sorted, colormap,
                                   y_min, y_max, bar_width, 'Test Partition',
                                   is_discrete_values, value_to_index_map, n_unique_values)

            # Add label
            ax.text(test_partition_pos, len(test_y_sorted) + 1, f'Test\n({len(test_y_sorted)})',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')

        # Configuration des axes
        if is_cv_folds and test_partition_indices:
            xlabel = 'CV Folds (T=Train, V=Validation) + Test Partition'
        elif is_cv_folds:
            xlabel = 'Folds (T=Train, V=Validation)'
        else:
            xlabel = 'Data Split'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)

        if is_cv_folds:
            title = f'Distribution Across {n_folds} CV Folds (Partition: {partition.upper()})\n'
        else:
            title = f'Distribution - Train/Test Split (Partition: {partition.upper()})\n'

        # Adjust title based on whether using metadata, classification, or regression
        if metadata_column:
            color_label = f'metadata "{metadata_column}"'
            # For metadata, show range or unique count based on data type
            unique_values = np.unique(y_values)
            # Try to determine if values are numeric
            try:
                # Attempt to convert first unique value to float
                float(unique_values[0])
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if len(unique_values) <= 20:
                title_suffix = f'{len(unique_values)} unique values'
            elif is_numeric:
                title_suffix = f'{int(y_min)} - {int(y_max)}, {len(unique_values)} unique'
            else:
                title_suffix = f'{len(unique_values)} unique string values'
        elif is_classification_task:
            color_label = 'class labels'
            unique_values = np.unique(y_values)
            if len(unique_values) <= 20:
                title_suffix = f'{len(unique_values)} unique classes'
            else:
                title_suffix = f'{len(unique_values)} unique classes'
        else:
            color_label = 'target values (y)'
            title_suffix = f'{y_min:.2f} - {y_max:.2f}'

        ax.set_title(title + f'(Colors represent {color_label}: {title_suffix})',
                    fontsize=14)

        # Configurer les ticks x
        x_positions = []
        x_labels = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            base_pos = fold_idx * (2 + gap_between_folds)
            if is_cv_folds:
                x_positions.extend([base_pos, base_pos + 1] if len(test_idx) > 0 else [base_pos])
                x_labels.extend([f'T{fold_idx}', f'V{fold_idx}'] if len(test_idx) > 0 else [f'T{fold_idx}'])
            else:
                x_positions.extend([base_pos, base_pos + 1] if len(test_idx) > 0 else [base_pos])
                x_labels.extend(['Train', 'Test'] if len(test_idx) > 0 else ['Train'])

        # Add test partition to x-axis if present
        if test_partition_indices:
            test_partition_pos = n_folds * (2 + gap_between_folds)
            x_positions.append(test_partition_pos)
            x_labels.append('Test')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45)

        # Ajouter des séparateurs visuels entre les folds
        for fold_idx in range(1, n_folds):
            separator_pos = fold_idx * (2 + gap_between_folds) - gap_between_folds / 2
            ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.9, linewidth=1)

        # Add separator before test partition if present (lighter to distinguish from folds)
        if test_partition_indices:
            separator_pos = n_folds * (2 + gap_between_folds) - gap_between_folds / 2
            ax.axvline(x=separator_pos, color='gray', linestyle=':', alpha=0.9, linewidth=1)

        # Ajouter colorbar
        if metadata_column or is_classification_task:
            # For metadata or classification: discrete colorbar with distinct boundaries
            unique_values = np.unique(y_values)
            n_unique = len(unique_values)

            # Create discrete colormap for colorbar
            if n_unique <= 10:
                cmap_discrete = plt.colormaps['tab10'].resampled(n_unique)
            elif n_unique <= 20:
                cmap_discrete = plt.colormaps['tab20'].resampled(n_unique)
            else:
                cmap_discrete = plt.colormaps['hsv'].resampled(n_unique)

            # Create boundaries between discrete values
            boundaries = np.arange(n_unique + 1) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, cmap_discrete.N)

            mappable = cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
            mappable.set_array(np.arange(n_unique))

            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=30,
                              boundaries=boundaries, ticks=np.arange(n_unique))
            # Set tick labels to actual metadata values or class labels
            if n_unique <= 20:
                # Show all labels if not too many
                cbar.ax.set_yticklabels([str(val) for val in unique_values])
            else:
                # Show subset of labels if too many
                step = max(1, n_unique // 10)
                cbar.set_ticks(np.arange(0, n_unique, step).tolist())
                cbar.ax.set_yticklabels([str(unique_values[i]) for i in range(0, n_unique, step)])

            if metadata_column:
                cbar.set_label(f'Metadata: {metadata_column}', fontsize=12)
            else:
                cbar.set_label('Y Labels', fontsize=12)
        else:
            # For continuous y values (regression): continuous colorbar
            mappable = cm.ScalarMappable(cmap=colormap)
            mappable.set_array(y_values)
            mappable.set_clim(y_min, y_max)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label('Target Values (y)', fontsize=12)

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plot_info = {
            'title': f'Fold Distribution ({n_folds} folds)',
            'n_folds': n_folds,
            'n_samples': n_samples,
            'y_range': (float(y_min), float(y_max))
        }

        return fig, plot_info

    def _create_stacked_bar(self, ax, position, y_values_sorted, colormap,
                           y_min, y_max, bar_width, label, is_discrete_values=False, value_to_index=None, n_unique=1):
        """
        Create a single stacked bar where each segment represents one sample.

        Args:
            ax: Matplotlib axis
            position: X position of the bar
            y_values_sorted: Y values sorted in ascending order
            colormap: Colormap for coloring segments
            y_min, y_max: Min and max y values for normalization
            bar_width: Width of the bar
            label: Label for the bar
            is_discrete_values: Whether values are discrete (metadata or classification) or continuous (regression)
            value_to_index: Dictionary mapping values to discrete indices (for discrete values)
            n_unique: Number of unique values (for discrete values)
        """
        # Normaliser les valeurs pour le colormap
        if is_discrete_values and value_to_index is not None:
            # For discrete values: use discrete indices
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_values_sorted])
        else:
            # For continuous y values
            if y_max != y_min:
                y_normalized = (y_values_sorted - y_min) / (y_max - y_min)
            else:
                y_normalized = np.zeros_like(y_values_sorted)

        # Créer chaque segment de la barre empilée
        for i, (y_val, y_norm) in enumerate(zip(y_values_sorted, y_normalized)):
            color = colormap(y_norm)

            # Create darker edge color (same hue but darker)
            rgb = mcolors.to_rgb(color)
            darker_color = tuple(max(0, c * 0.6) for c in rgb)

            ax.bar(position, 1, bottom=i, width=bar_width,
                   color=color, edgecolor=darker_color, linewidth=0.5)



"""FoldChartController - Visualizes cross-validation folds with y-value color coding."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import copy
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
import io

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


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
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Execute fold visualization showing train/test splits with y-value color coding.
        Skips execution in prediction mode.

        Returns:
            Tuple of (context, image_list) where image_list contains plot binaries
        """
        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, []

        # print(f"Executing fold charts for step: {step}, keyword: {context.get('keyword', '')}")

        # Check if using metadata column for colors (keyword like "chart_columnName")
        keyword = context.get('keyword', '')
        metadata_column = None
        if keyword.startswith("fold_") and keyword != "chart_fold" and keyword != "fold_chart":
            metadata_column = keyword[5:]  # Extract column name after "fold_"
            if runner.verbose > 0:
                print(f"ℹ️ Using metadata column '{metadata_column}' for color coding")

        # Determine which partition to use (default to train if not specified)
        partition = context.get("partition", "train")
        if partition not in ["train", "test"]:
            print(f"⚠️ Invalid partition '{partition}'. Using 'train' instead.")
            partition = "train"

        # Get data for visualization
        local_context = copy.deepcopy(context)
        local_context["partition"] = partition

        # Get folds from dataset
        folds = dataset.folds
        use_absolute_indices = False  # Flag to indicate if folds contain absolute indices

        # Fallback logic: If no folds, create a simple train/test split visualization
        if not folds:
            print("ℹ️ No CV folds found. Creating visualization from train/test partition.")

            # Try to get train and test data
            train_context = copy.deepcopy(context)
            train_context["partition"] = "train"
            test_context = copy.deepcopy(context)
            test_context["partition"] = "test"

            train_indices = dataset._indexer.x_indices(train_context)
            test_indices = dataset._indexer.x_indices(test_context)

            if len(test_indices) > 0:
                # We have both train and test data - create a single "fold" with absolute indices
                # Use absolute indices: train is [0..177], test is [178..207]
                train_abs = list(range(len(train_indices)))
                test_abs = list(range(len(train_indices), len(train_indices) + len(test_indices)))
                folds = [(train_abs, test_abs)]
                print(f"  Using train ({len(train_indices)} samples) and test ({len(test_indices)} samples) partitions.")
            elif len(train_indices) > 0:
                # Only train data exists - show it as a single bar
                folds = [(list(range(len(train_indices))), [])]
                print(f"  Only train partition available ({len(train_indices)} samples).")
            else:
                print("⚠️ No data available for visualization.")
                return context, []

        # Get values for color coding (either y or metadata column)
        # For CV folds: get all data for proper indexing
        # For fallback (train/test): get data for the specific partition
        if metadata_column:
            # Use metadata column for colors
            if dataset.folds:
                # CV folds mode: need all data since indices refer to full dataset
                all_context = copy.deepcopy(context)
                all_context["partition"] = "all"
                color_values = dataset.metadata_column(metadata_column, all_context)
            else:
                # Fallback mode: get train and test separately and concatenate
                train_ctx = copy.deepcopy(context)
                train_ctx["partition"] = "train"
                test_ctx = copy.deepcopy(context)
                test_ctx["partition"] = "test"

                meta_train = dataset.metadata_column(metadata_column, train_ctx)
                meta_test = dataset.metadata_column(metadata_column, test_ctx)

                # Concatenate train and test for visualization
                if len(meta_test) > 0:
                    color_values = np.concatenate([meta_train, meta_test])
                else:
                    color_values = meta_train
        else:
            # Use y values for colors (default behavior)
            if dataset.folds:
                # CV folds mode: need all data since indices refer to full dataset
                all_context = copy.deepcopy(context)
                all_context["partition"] = "all"
                color_values = dataset.y(all_context)
            else:
                # Fallback mode: get train and test separately and concatenate
                train_ctx = copy.deepcopy(context)
                train_ctx["partition"] = "train"
                test_ctx = copy.deepcopy(context)
                test_ctx["partition"] = "test"

                y_train = dataset.y(train_ctx)
                y_test = dataset.y(test_ctx)

                # Concatenate train and test for visualization
                if len(y_test) > 0:
                    color_values = np.concatenate([y_train, y_test])
                else:
                    color_values = y_train

        color_values_flat = color_values.flatten() if color_values.ndim > 1 else color_values

        # Create fold visualization
        fig, plot_info = self._create_fold_chart(folds, color_values_flat, len(color_values_flat), partition, dataset.folds, dataset, metadata_column)

        # Save plot to memory buffer as PNG binary
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        # Create filename with partition info
        fold_suffix = f"{len(folds)}folds" if dataset.folds else "traintest_split"
        metadata_suffix = f"_{metadata_column}" if metadata_column else ""
        image_name = f"fold_visualization_{fold_suffix}_{partition}{metadata_suffix}.png"
        img_list = [(image_name, img_png_binary)]

        if runner.plots_visible:
            # Store figure reference - user will call plt.show() at the end
            runner._figure_refs.append(fig)
            plt.show()
        else:
            plt.close(fig)

        return context, img_list

    def _create_fold_chart(self, folds: List[Tuple[List[int], List[int]]], y_values: np.ndarray, n_samples: int, partition: str = "train",
                           original_folds: List = None, dataset: 'SpectroDataset' = None, metadata_column: str = None) -> Tuple[Any, Dict[str, Any]]:
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

        # Create colormap - use discrete contrastive colors for metadata, continuous for y values

        if metadata_column:
            # For metadata: use discrete, highly contrastive colors
            # Get unique values and create a discrete colormap
            unique_values = np.unique(y_values)
            n_unique = len(unique_values)

            # Use a highly contrastive colormap (tab20, tab20b, tab20c for many categories)
            if n_unique <= 10:
                colormap = cm.get_cmap('tab10', n_unique)
            elif n_unique <= 20:
                colormap = cm.get_cmap('tab20', n_unique)
            else:
                # For many categories, use a combination or hsv
                colormap = cm.get_cmap('hsv', n_unique)

            # Create a mapping from actual values to discrete indices
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}

            # Normalize to discrete indices [0, 1, 2, ...] / n_unique
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_values])

            # For metadata, y_min and y_max are index boundaries (not used for string values)
            y_min, y_max = 0, n_unique - 1
        else:
            # For y values: use continuous colormap
            y_min, y_max = y_values.min(), y_values.max()
            colormap = cm.get_cmap('viridis')
            # Normalize y values to [0, 1] for colormap
            if y_max != y_min:
                y_normalized = (y_values - y_min) / (y_max - y_min)
            else:
                y_normalized = np.zeros_like(y_values)

        bar_width = 0.8
        gap_between_folds = 0.4

        # Prepare metadata-specific parameters
        is_metadata = metadata_column is not None
        value_to_index_map = None
        n_unique_values = 1

        if is_metadata:
            unique_values = np.unique(y_values)
            n_unique_values = len(unique_values)
            value_to_index_map = {val: idx for idx, val in enumerate(unique_values)}

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Position des barres pour ce fold
            base_pos = fold_idx * (2 + gap_between_folds)
            train_pos = base_pos
            test_pos = base_pos + 1

            # Traiter les données d'entraînement
            train_y = y_values[train_idx]
            train_sorted_indices = np.argsort(train_y)
            train_y_sorted = train_y[train_sorted_indices]

            # Traiter les données de test
            test_y = y_values[test_idx]
            test_sorted_indices = np.argsort(test_y)
            test_y_sorted = test_y[test_sorted_indices]

            # Créer les barres empilées pour TRAIN
            self._create_stacked_bar(ax, train_pos, train_y_sorted, colormap,
                                   y_min, y_max, bar_width, f'Train F{fold_idx}',
                                   is_metadata, value_to_index_map, n_unique_values)

            # Créer les barres empilées pour TEST (only if test data exists)
            if len(test_idx) > 0:
                self._create_stacked_bar(ax, test_pos, test_y_sorted, colormap,
                                       y_min, y_max, bar_width, f'Test F{fold_idx}',
                                       is_metadata, value_to_index_map, n_unique_values)

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
                                   is_metadata, value_to_index_map, n_unique_values)

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

        # Adjust title based on whether using metadata or y values
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
            ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)

        # Add separator before test partition if present (lighter to distinguish from folds)
        if test_partition_indices:
            separator_pos = n_folds * (2 + gap_between_folds) - gap_between_folds / 2
            ax.axvline(x=separator_pos, color='gray', linestyle=':', alpha=0.3, linewidth=1)

        # Ajouter colorbar
        if metadata_column:
            # For metadata: discrete colorbar with distinct boundaries
            unique_values = np.unique(y_values)
            n_unique = len(unique_values)

            # Create discrete colormap for colorbar
            if n_unique <= 10:
                cmap_discrete = cm.get_cmap('tab10', n_unique)
            elif n_unique <= 20:
                cmap_discrete = cm.get_cmap('tab20', n_unique)
            else:
                cmap_discrete = cm.get_cmap('hsv', n_unique)

            # Create boundaries between discrete values
            boundaries = np.arange(n_unique + 1) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, cmap_discrete.N)

            mappable = cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
            mappable.set_array(np.arange(n_unique))

            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=30,
                              boundaries=boundaries, ticks=np.arange(n_unique))            # Set tick labels to actual metadata values
            if n_unique <= 20:
                # Show all labels if not too many
                cbar.ax.set_yticklabels([str(val) for val in unique_values])
            else:
                # Show subset of labels if too many
                step = max(1, n_unique // 10)
                cbar.set_ticks(np.arange(0, n_unique, step).tolist())
                cbar.ax.set_yticklabels([str(unique_values[i]) for i in range(0, n_unique, step)])

            cbar.set_label(f'Metadata: {metadata_column}', fontsize=12)
        else:
            # For y values: continuous colorbar
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
                           y_min, y_max, bar_width, label, is_metadata=False, value_to_index=None, n_unique=1):
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
            is_metadata: Whether values are metadata (discrete) or continuous y values
            value_to_index: Dictionary mapping values to discrete indices (for metadata)
            n_unique: Number of unique values (for metadata)
        """
        # Normaliser les valeurs pour le colormap
        if is_metadata and value_to_index is not None:
            # For metadata: use discrete indices
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
            ax.bar(position, 1, bottom=i, width=bar_width,
                  color=color, edgecolor='white', linewidth=0.5)

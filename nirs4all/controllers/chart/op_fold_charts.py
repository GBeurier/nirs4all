"""FoldChartController - Visualizes cross-validation folds with y-value color coding."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        return keyword == "fold_chart" or keyword == "chart_fold"

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

        # Get data for visualization
        local_context = copy.deepcopy(context)
        local_context["partition"] = "train"  # Use train data for fold visualization

        # Get y values for color coding
        y = dataset.y(local_context)
        y_flat = y.flatten() if y.ndim > 1 else y

        # Get folds from dataset
        folds = dataset.folds
        if not folds:
            print("⚠️ No folds found in dataset. Run cross-validation first.")
            return context, []

        # print(f"Found {len(folds)} folds to visualize")

        # Create fold visualization
        fig, plot_info = self._create_fold_chart(folds, y_flat, len(y_flat))

        # Save plot to memory buffer as PNG binary
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        # Create filename
        image_name = f"fold_visualization_{len(folds)}folds.png"
        img_list = [(image_name, img_png_binary)]

        # Keep figure open if plots are visible, close otherwise
        if not runner.plots_visible:
            plt.close(fig)

        return context, img_list

    def _create_fold_chart(self, folds: List[Tuple[List[int], List[int]]],
                          y_values: np.ndarray, n_samples: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Create a fold visualization chart with stacked bars showing y-value distribution.

        Args:
            folds: List of (train_indices, test_indices) tuples
            y_values: Target values for color coding
            n_samples: Total number of samples

        Returns:
            Tuple of (figure, plot_info)
        """
        n_folds = len(folds)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(max(12, n_folds * 3), 8))

        # Create colormap
        colormap = cm.get_cmap('viridis')
        y_min, y_max = y_values.min(), y_values.max()

        # Normalize y values to [0, 1] for colormap
        if y_max != y_min:
            y_normalized = (y_values - y_min) / (y_max - y_min)
        else:
            y_normalized = np.zeros_like(y_values)

        bar_width = 0.8
        gap_between_folds = 0.4

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
                                   y_min, y_max, bar_width, f'Train F{fold_idx}')

            # Créer les barres empilées pour TEST
            self._create_stacked_bar(ax, test_pos, test_y_sorted, colormap,
                                   y_min, y_max, bar_width, f'Test F{fold_idx}')

            # Ajouter les labels au-dessus des barres
            ax.text(train_pos, len(train_y) + 1, f'T{fold_idx}\n({len(train_y)})',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(test_pos, len(test_y) + 1, f'V{fold_idx}\n({len(test_y)})',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Configuration des axes
        ax.set_xlabel('Folds (T=Train, V=Validation)', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(f'Y-Value Distribution Across {n_folds} Folds\n'
                    f'(Colors represent target values: {y_min:.2f} - {y_max:.2f})',
                    fontsize=14)

        # Configurer les ticks x
        x_positions = []
        x_labels = []
        for fold_idx in range(n_folds):
            base_pos = fold_idx * (2 + gap_between_folds)
            x_positions.extend([base_pos, base_pos + 1])
            x_labels.extend([f'T{fold_idx}', f'V{fold_idx}'])

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45)

        # Ajouter des séparateurs visuels entre les folds
        for fold_idx in range(1, n_folds):
            separator_pos = fold_idx * (2 + gap_between_folds) - gap_between_folds/2
            ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)

        # Ajouter colorbar
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
                           y_min, y_max, bar_width, label):
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
        """
        # Normaliser les valeurs y pour le colormap
        if y_max != y_min:
            y_normalized = (y_values_sorted - y_min) / (y_max - y_min)
        else:
            y_normalized = np.zeros_like(y_values_sorted)

        # Créer chaque segment de la barre empilée
        for i, (y_val, y_norm) in enumerate(zip(y_values_sorted, y_normalized)):
            color = colormap(y_norm)
            ax.bar(position, 1, bottom=i, width=bar_width,
                  color=color, edgecolor='white', linewidth=0.5)
"""SpectralDistributionController - Spectral envelope visualization for train/test/folds."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.header_units import get_axis_label
import io

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class SpectralDistributionController(OperatorController):
    """Controller for spectral distribution envelope visualization.

    Shows envelope (min/max/mean/IQR) for train vs test partitions,
    with optional per-fold visualization when cross-validation folds exist.
    """

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword in ["spectra_dist", "spectral_distribution", "spectra_envelope"]

    @classmethod
    def use_multi_source(cls) -> bool:
        return True  # Handle each source separately

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
        """Execute spectral distribution envelope visualization.

        Creates envelope plots showing min/max/mean/IQR for train vs test.
        If CV folds exist, creates a grid showing each fold.

        Returns:
            Tuple of (context, StepOutput)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, StepOutput()

        outputs = []

        # Get spectra data - shape (samples, processings, features)
        spectra_data = dataset.x(context.selector, "3d", False)

        if not isinstance(spectra_data, list):
            spectra_data = [spectra_data]

        # Get folds
        folds = dataset.folds
        has_cv_folds = folds is not None and len(folds) > 1

        # Get colormap colors
        viridis_cmap = plt.colormaps['viridis']
        train_color = viridis_cmap(0.85)
        train_color_dark = viridis_cmap(0.95)
        test_color = viridis_cmap(0.15)
        test_color_dark = viridis_cmap(0.05)

        for sd_idx, x in enumerate(spectra_data):
            processing_ids = dataset.features_processings(sd_idx)
            n_processings = x.shape[1]

            # Get headers for x-axis
            spectra_headers = dataset.headers(sd_idx)
            current_n_features = x.shape[2]

            # Use headers if they match
            if spectra_headers and len(spectra_headers) == current_n_features:
                try:
                    wavelengths = np.array([float(h) for h in spectra_headers])
                    x_label = self._get_x_label(dataset, sd_idx)
                except (ValueError, TypeError):
                    wavelengths = np.arange(current_n_features)
                    x_label = 'Feature Index'
            else:
                wavelengths = np.arange(current_n_features)
                x_label = 'Feature Index'

            if has_cv_folds:
                # Grid mode: one plot per fold + test
                fig, chart_name = self._create_fold_grid_spectral(
                    dataset, context, folds, x, wavelengths, x_label,
                    processing_ids, sd_idx, train_color, train_color_dark,
                    test_color, test_color_dark
                )
            else:
                # Simple mode: train vs test
                fig, chart_name = self._create_simple_spectral(
                    dataset, context, x, wavelengths, x_label,
                    processing_ids, sd_idx, train_color, train_color_dark,
                    test_color, test_color_dark
                )

            # Save plot to memory buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_png_binary = img_buffer.getvalue()
            img_buffer.close()

            # Add source suffix if multi-source
            if dataset.is_multi_source():
                chart_name += f"_src{sd_idx}"

            outputs.append((img_png_binary, chart_name, "png"))

            if runtime_context.step_runner.plots_visible:
                runtime_context.step_runner._figure_refs.append(fig)
                plt.show()
            else:
                plt.close(fig)

        return context, StepOutput(outputs=outputs)

    def _get_x_label(self, dataset: 'SpectroDataset', source_idx: int) -> str:
        """Get appropriate x-axis label based on header unit."""
        try:
            header_unit = dataset.header_unit(source_idx)
            return get_axis_label(header_unit)
        except (AttributeError, IndexError):
            return 'Feature Index'

    def _create_simple_spectral(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        x: np.ndarray,
        wavelengths: np.ndarray,
        x_label: str,
        processing_ids: List[str],
        source_idx: int,
        train_color, train_color_dark, test_color, test_color_dark
    ) -> Tuple[Any, str]:
        """Create simple train vs test spectral distribution plot."""
        n_processings = x.shape[1]

        # Calculate grid for processings
        n_cols = min(3, n_processings)
        n_rows = (n_processings + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

        if n_processings == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Get train and test indices
        train_context = context.with_partition("train")
        test_context = context.with_partition("test")

        train_indices = dataset._indexer.x_indices(train_context, include_augmented=False)
        test_indices = dataset._indexer.x_indices(test_context, include_augmented=False)

        for proc_idx in range(n_processings):
            ax = axes[proc_idx]
            x_2d = x[:, proc_idx, :]

            X_train = x_2d[train_indices]
            X_test = x_2d[test_indices] if len(test_indices) > 0 else None

            self._plot_spectral_distribution(
                ax, X_train, X_test, wavelengths, x_label,
                processing_ids[proc_idx],
                train_color, train_color_dark, test_color, test_color_dark
            )

        # Hide unused subplots
        for idx in range(n_processings, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{dataset.name} - Spectral Distribution: Train vs Test',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        return fig, "spectral_distribution_train_test"

    def _create_fold_grid_spectral(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        folds: List[Tuple[List[int], List[int]]],
        x: np.ndarray,
        wavelengths: np.ndarray,
        x_label: str,
        processing_ids: List[str],
        source_idx: int,
        train_color, train_color_dark, test_color, test_color_dark
    ) -> Tuple[Any, str]:
        """Create grid of spectral distribution plots for each fold."""
        n_folds = len(folds)
        n_processings = x.shape[1]

        # Check if test partition exists
        test_context = context.with_partition("test")
        test_indices = dataset._indexer.x_indices(test_context, include_augmented=False)
        has_test = len(test_indices) > 0

        n_plots = n_folds + (1 if has_test else 0)

        # For simplicity with multiple processings, show only first processing in grid
        # or use the first processing
        proc_idx = 0
        x_2d = x[:, proc_idx, :]

        # Calculate grid dimensions
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Get base sample IDs for mapping fold indices
        train_context = context.with_partition("train")
        base_sample_ids = dataset._indexer.x_indices(train_context, include_augmented=False)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ax = axes[fold_idx]

            if len(val_idx) == 0:
                ax.text(0.5, 0.5, 'No validation samples', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f'Fold {fold_idx + 1}')
                continue

            # Map fold indices to actual sample indices
            train_idx_arr = np.array(train_idx)
            val_idx_arr = np.array(val_idx)

            try:
                train_sample_idx = base_sample_ids[train_idx_arr]
                val_sample_idx = base_sample_ids[val_idx_arr]
            except IndexError:
                train_sample_idx = train_idx_arr
                val_sample_idx = val_idx_arr

            X_train = x_2d[train_sample_idx]
            X_val = x_2d[val_sample_idx]

            self._plot_spectral_distribution(
                ax, X_train, X_val, wavelengths, x_label,
                f'Fold {fold_idx + 1}',
                train_color, train_color_dark, test_color, test_color_dark,
                legend_labels=('Train', 'Val')
            )

        # Plot test partition if available
        if has_test:
            ax = axes[n_folds]
            train_indices_all = dataset._indexer.x_indices(train_context, include_augmented=False)

            X_train_all = x_2d[train_indices_all]
            X_test = x_2d[test_indices]

            self._plot_spectral_distribution(
                ax, X_train_all, X_test, wavelengths, x_label,
                'Test Partition',
                train_color, train_color_dark, test_color, test_color_dark
            )
            ax.set_title('Test Partition', fontsize=11, color='darkred')

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        proc_name = processing_ids[proc_idx] if processing_ids else "raw"
        fig.suptitle(f'{dataset.name} - Spectral Distribution ({proc_name}): {n_folds} Folds',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        return fig, f"spectral_distribution_{n_folds}folds"

    def _plot_spectral_distribution(
        self,
        ax,
        X_train: np.ndarray,
        X_test: np.ndarray | None,
        wavelengths: np.ndarray,
        x_label: str,
        title: str,
        train_color, train_color_dark, test_color, test_color_dark,
        legend_labels: Tuple[str, str] = ('Train', 'Test')
    ) -> None:
        """Plot spectral distribution showing envelope (min/max/mean/IQR) for train and test.

        Args:
            ax: Matplotlib axis to plot on
            X_train: Training spectra array (n_samples, n_features)
            X_test: Test spectra array (n_samples, n_features) or None
            wavelengths: Array of wavelength/feature values for x-axis
            x_label: Label for x-axis
            title: Subplot title
            train_color: Color for train envelope
            train_color_dark: Darker color for train mean line
            test_color: Color for test envelope
            test_color_dark: Darker color for test mean line
            legend_labels: Labels for train and test in legend
        """
        # Calculate statistics for train
        train_mean = np.mean(X_train, axis=0)
        train_min = np.min(X_train, axis=0)
        train_max = np.max(X_train, axis=0)
        train_p25 = np.percentile(X_train, 25, axis=0)
        train_p75 = np.percentile(X_train, 75, axis=0)

        # Plot train envelope
        ax.fill_between(wavelengths, train_min, train_max,
                        alpha=0.15, color=train_color, label=f'{legend_labels[0]} (min-max)')
        ax.fill_between(wavelengths, train_p25, train_p75,
                        alpha=0.3, color=train_color, label=f'{legend_labels[0]} (IQR)')
        ax.plot(wavelengths, train_mean, color=train_color_dark, linewidth=2,
                label=f'{legend_labels[0]} mean', alpha=0.95)

        # Plot test envelope if available
        if X_test is not None and len(X_test) > 0:
            test_mean = np.mean(X_test, axis=0)
            test_min = np.min(X_test, axis=0)
            test_max = np.max(X_test, axis=0)
            test_p25 = np.percentile(X_test, 25, axis=0)
            test_p75 = np.percentile(X_test, 75, axis=0)

            ax.fill_between(wavelengths, test_min, test_max,
                            alpha=0.15, color=test_color, label=f'{legend_labels[1]} (min-max)')
            ax.fill_between(wavelengths, test_p25, test_p75,
                            alpha=0.3, color=test_color, label=f'{legend_labels[1]} (IQR)')
            ax.plot(wavelengths, test_mean, color=test_color_dark, linewidth=2,
                    label=f'{legend_labels[1]} mean', alpha=0.95)

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

"""SpectraChartController - Unified 2D and 3D spectra visualization controller."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
import io
if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

@register_controller
class SpectraChartController(OperatorController):

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword in ["chart_2d", "chart_3d", "2d_chart", "3d_chart"]

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Chart controllers should skip execution during prediction mode."""
        return False

    @staticmethod
    def _shorten_processing_name(processing_name: str) -> str:
        """Shorten preprocessing names for chart titles."""
        replacements = [
            ("raw_", ""),
            ("SavitzkyGolay", "SG"),
            ("MultiplicativeScatterCorrection", "MSC"),
            ("StandardNormalVariate", "SNV"),
            ("FirstDerivative", "1stDer"),
            ("SecondDerivative", "2ndDer"),
            ("Detrend", "Detr"),
            ("Gaussian", "Gauss"),
            ("Haar", "Haar"),
            ("LogTransform", "Log"),
            ("MinMaxScaler", "MinMax"),
            ("RobustScaler", "Rbt"),
            ("StandardScaler", "Std"),
            ("QuantileTransformer", "Quant"),
            ("PowerTransformer", "Pow"),
            # ("_", ""),
        ]
        for long, short in replacements:
            processing_name = processing_name.replace(long, short)

        # replace expr _<digit>_ with | then remaining _<digits> with nothing
        processing_name = re.sub(r'_\d+_', '>', processing_name)
        processing_name = re.sub(r'_\d+', '', processing_name)
        return processing_name

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
        Execute spectra visualization for both 2D and 3D plots.
        Skips execution in prediction mode.

        Returns:
            Tuple of (context, image_list) where image_list contains plot metadata
        """
        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, []

        is_3d = (step == "chart_3d") or (step == "3d_chart")

        # Initialize image list to track generated plots
        img_list = []
        local_context = context.copy()
        spectra_data = dataset.x(local_context, "3d", False)
        y = dataset.y(local_context)

        if not isinstance(spectra_data, list):
            spectra_data = [spectra_data]

        # Sort samples by y values (from lower to higher)
        y_flat = y.flatten() if y.ndim > 1 else y
        sorted_indices = np.argsort(y_flat)
        y_sorted = y_flat[sorted_indices]

        for sd_idx, x in enumerate(spectra_data):
            processing_ids = dataset.features_processings(sd_idx)
            n_processings = x.shape[1]

            # Debug: print what we got
            if runner.verbose > 0:
                print(f"   Source {sd_idx}: {n_processings} processings: {processing_ids}")
                print(f"   Data shape: {x.shape}")

            # Calculate subplot grid (prefer horizontal layout)
            n_cols = min(3, n_processings)  # Max 3 columns
            n_rows = (n_processings + n_cols - 1) // n_cols

            # Create figure with subplots for all preprocessings
            fig_width = 6 * n_cols
            fig_height = 5 * n_rows
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Main title with dataset name (no emoji to avoid encoding issues)
            chart_type = "3D Spectra" if is_3d else "2D Spectra"
            main_title = f"{dataset.name} - {chart_type}"
            if dataset.is_multi_source():
                main_title += f" (Source {sd_idx})"
            fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

            # Create subplots for each processing
            for processing_idx in range(n_processings):
                processing_name = processing_ids[processing_idx]
                short_name = self._shorten_processing_name(processing_name)

                # Get 2D data for this processing: (samples, features)
                x_2d = x[:, processing_idx, :]
                x_sorted = x_2d[sorted_indices]

                # Get headers for this specific processing (may differ after resampling)
                # Headers are shared across all processings in a source, so we check if they match
                spectra_headers = dataset.headers(sd_idx)
                current_n_features = x_2d.shape[1]

                # Only use headers if they match the current number of features
                if spectra_headers and len(spectra_headers) == current_n_features:
                    processing_headers = spectra_headers
                else:
                    # Headers don't match - likely after dimension-changing operation
                    processing_headers = None

                if runner.verbose > 0 and processing_idx == 0:
                    print(f"   Headers available: {len(spectra_headers) if spectra_headers else 0}, features: {current_n_features}")

                # Create subplot
                if is_3d:
                    ax = fig.add_subplot(n_rows, n_cols, processing_idx + 1, projection='3d')
                    self._plot_3d_spectra(ax, x_sorted, y_sorted, short_name, processing_headers)
                else:
                    ax = fig.add_subplot(n_rows, n_cols, processing_idx + 1)
                    self._plot_2d_spectra(ax, x_sorted, y_sorted, short_name, processing_headers)

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save plot to memory buffer as PNG binary
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_png_binary = img_buffer.getvalue()
            img_buffer.close()

            # Create filename
            image_name = "2D" if not is_3d else "3D"
            image_name += "_Chart"
            if dataset.is_multi_source():
                image_name += f"_src{sd_idx}"
            image_name += ".png"
            img_list.append((image_name, img_png_binary))

            if runner.plots_visible:
                # Store figure reference - user will call plt.show() at the end
                runner._figure_refs.append(fig)
                plt.show()
            else:
                plt.close(fig)

        return context, img_list

    def _plot_2d_spectra(self, ax, x_sorted: np.ndarray, y_sorted: np.ndarray, processing_name: str, headers: Optional[List[str]] = None) -> None:
        """Plot 2D spectra on given axis."""
        # Create feature indices (wavelengths)
        n_features = x_sorted.shape[1]

        # Use headers if available, otherwise fall back to indices
        if headers and len(headers) == n_features:
            # Try to convert headers to numeric values for wavelengths
            try:
                x_values = np.array([float(h) for h in headers])
                x_label = 'Wavelength (cm-1)'
            except (ValueError, TypeError):
                # If headers are not numeric, use them as categorical labels
                x_values = np.arange(n_features)
                x_label = 'Features'
        else:
            x_values = np.arange(n_features)
            x_label = 'Features'

        # Create colormap for gradient based on y values
        colormap = plt.colormaps.get_cmap('viridis')
        y_min, y_max = y_sorted.min(), y_sorted.max()

        # Normalize y values to [0, 1] for colormap
        if y_max != y_min:
            y_normalized = (y_sorted - y_min) / (y_max - y_min)
        else:
            y_normalized = np.zeros_like(y_sorted)

        # Plot each spectrum as a 2D line with gradient colors
        for i, spectrum in enumerate(x_sorted):
            color = colormap(y_normalized[i])
            ax.plot(x_values, spectrum,
                    color=color, alpha=0.7, linewidth=1)

        # Force axis order to prevent matplotlib from auto-sorting
        if len(x_values) > 1 and x_values[0] > x_values[-1]:
            # Descending order - set limits to force this display
            ax.set_xlim(x_values[0], x_values[-1])

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)

        # Subtitle with preprocessing name and dimensions
        subtitle = f"{processing_name}\n({len(y_sorted)} samples × {x_sorted.shape[1]} features)"
        ax.set_title(subtitle, fontsize=10)

        # Add colorbar to show the y-value gradient
        mappable = cm.ScalarMappable(cmap=colormap)
        mappable.set_array(y_sorted)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10)
        cbar.set_label('y', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    def _plot_3d_spectra(self, ax, x_sorted: np.ndarray, y_sorted: np.ndarray, processing_name: str, headers: Optional[List[str]] = None) -> None:
        """Plot 3D spectra on given axis."""
        # Create feature indices (wavelengths)
        n_features = x_sorted.shape[1]

        # Use headers if available, otherwise fall back to indices
        if headers and len(headers) == n_features:
            # Try to convert headers to numeric values for wavelengths
            try:
                x_values = np.array([float(h) for h in headers])
                x_label = 'Wavelength (cm-1)'
            except (ValueError, TypeError):
                # If headers are not numeric, use them as categorical labels
                x_values = np.arange(n_features)
                x_label = 'Features'
        else:
            x_values = np.arange(n_features)
            x_label = 'Features'

        # Create colormap for gradient based on y values
        colormap = plt.colormaps.get_cmap('viridis')
        y_min, y_max = y_sorted.min(), y_sorted.max()

        # Normalize y values to [0, 1] for colormap
        if y_max != y_min:
            y_normalized = (y_sorted - y_min) / (y_max - y_min)
        else:
            y_normalized = np.zeros_like(y_sorted)

        # Plot each spectrum as a line in 3D space with gradient colors
        for i, (spectrum, y_val) in enumerate(zip(x_sorted, y_sorted)):
            color = colormap(y_normalized[i])
            ax.plot(x_values, [y_val] * n_features, spectrum,
                    color=color, alpha=0.7, linewidth=1)

        # Force axis order to prevent matplotlib from auto-sorting
        if len(x_values) > 1 and x_values[0] > x_values[-1]:
            # Descending order - set limits to force this display
            ax.set_xlim(x_values[0], x_values[-1])

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('y (sorted)', fontsize=9)
        ax.set_zlabel('Intensity', fontsize=9)

        # Subtitle with preprocessing name and dimensions
        subtitle = f"{processing_name}\n({len(y_sorted)} samples × {x_sorted.shape[1]} features)"
        ax.set_title(subtitle, fontsize=10)

        # Add colorbar to show the y-value gradient
        mappable = cm.ScalarMappable(cmap=colormap)
        mappable.set_array(y_sorted)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10, pad=0.1)
        cbar.set_label('y', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

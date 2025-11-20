"""SpectraChartController - Unified 2D and 3D spectra visualization controller."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import re
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
import io
if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep

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
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple['ExecutionContext', List[Dict[str, Any]]]:
        """
        Execute spectra visualization for both 2D and 3D plots.
        Skips execution in prediction mode.

        Returns:
            Tuple of (context, image_list) where image_list contains plot metadata
        """
        # Extract step config for compatibility
        step = step_info.original_step

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, []

        is_3d = (step == "chart_3d") or (step == "3d_chart")

        # Initialize image list to track generated plots
        img_list = []
        # Use context directly as it is immutable-ish and we only read from it
        spectra_data = dataset.x(context.selector, "3d", False)
        y = dataset.y(context.selector)

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

                # Get header unit for this source
                try:
                    header_unit = dataset.header_unit(sd_idx)
                except (AttributeError, IndexError):
                    # Fall back to default if header_unit method not available
                    header_unit = "cm-1"

                # Create subplot
                is_classification = dataset.is_classification
                if is_3d:
                    ax = fig.add_subplot(n_rows, n_cols, processing_idx + 1, projection='3d')
                    self._plot_3d_spectra(ax, x_sorted, y_sorted, short_name, processing_headers, header_unit, is_classification)
                else:
                    ax = fig.add_subplot(n_rows, n_cols, processing_idx + 1)
                    self._plot_2d_spectra(ax, x_sorted, y_sorted, short_name, processing_headers, header_unit, is_classification)

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

            # Save the chart as a human-readable output file
            output_path = runner.saver.save_output(
                step_number=runner.step_number,
                name=image_name.replace('.png', ''),  # Name without extension
                data=img_png_binary,
                extension='.png'
            )

            # Add to image list for tracking (only if saved)
            if output_path:
                img_list.append({
                    "name": image_name,
                    "path": str(output_path),
                    "type": "chart_output"
                })

            if runner.plots_visible:
                # Store figure reference - user will call plt.show() at the end
                runner._figure_refs.append(fig)
                plt.show()
            else:
                plt.close(fig)

        return context, img_list

    def _plot_2d_spectra(self, ax, x_sorted: np.ndarray, y_sorted: np.ndarray, processing_name: str, headers: Optional[List[str]] = None, header_unit: str = "cm-1", is_classification: bool = False) -> None:
        """Plot 2D spectra on given axis."""
        # Create feature indices (wavelengths)
        n_features = x_sorted.shape[1]

        # Use headers if available, otherwise fall back to indices
        if headers and len(headers) == n_features:
            # Try to convert headers to numeric values for wavelengths
            try:
                x_values = np.array([float(h) for h in headers])
                # Determine x-axis label based on header unit
                if header_unit == "cm-1":
                    x_label = 'Wavenumber (cm⁻¹)'
                elif header_unit == "nm":
                    x_label = 'Wavelength (nm)'
                else:
                    x_label = 'Features'
            except (ValueError, TypeError):
                # If headers are not numeric, use them as categorical labels
                x_values = np.arange(n_features)
                x_label = 'Features'
        else:
            x_values = np.arange(n_features)
            x_label = 'Features'

        # Create colormap - discrete for classification, continuous for regression
        if is_classification:
            # Use discrete colormap for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            if n_unique <= 10:
                colormap = cm.get_cmap('tab10', n_unique)
            elif n_unique <= 20:
                colormap = cm.get_cmap('tab20', n_unique)
            else:
                colormap = cm.get_cmap('hsv', n_unique)

            # Create mapping from actual values to discrete indices
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_sorted])
            y_min, y_max = 0, n_unique - 1
        else:
            # Use continuous colormap for regression
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
        subtitle = f"{processing_name} - ({len(y_sorted)} samples × {x_sorted.shape[1]} features)"
        ax.set_title(subtitle, fontsize=10)

        # Add colorbar to show the y-value gradient
        if is_classification:
            # Discrete colorbar for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            import matplotlib.colors as mcolors
            boundaries = np.arange(n_unique + 1) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, colormap.N)

            mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
            mappable.set_array(np.arange(n_unique))

            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10,
                              boundaries=boundaries, ticks=np.arange(n_unique))

            # Set tick labels to actual class values
            if n_unique <= 20:
                cbar.ax.set_yticklabels([str(val) for val in unique_values])
            else:
                step = max(1, n_unique // 10)
                cbar.set_ticks(np.arange(0, n_unique, step).tolist())
                cbar.ax.set_yticklabels([str(unique_values[i]) for i in range(0, n_unique, step)])
        else:
            # Continuous colorbar for regression
            mappable = cm.ScalarMappable(cmap=colormap)
            mappable.set_array(y_sorted)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10)

        cbar.set_label('y', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    def _plot_3d_spectra(self, ax, x_sorted: np.ndarray, y_sorted: np.ndarray, processing_name: str, headers: Optional[List[str]] = None, header_unit: str = "cm-1", is_classification: bool = False) -> None:
        """Plot 3D spectra on given axis."""
        # Create feature indices (wavelengths)
        n_features = x_sorted.shape[1]

        # Use headers if available, otherwise fall back to indices
        if headers and len(headers) == n_features:
            # Try to convert headers to numeric values for wavelengths
            try:
                x_values = np.array([float(h) for h in headers])
                # Determine x-axis label based on header unit
                if header_unit == "cm-1":
                    x_label = 'Wavenumber (cm⁻¹)'
                elif header_unit == "nm":
                    x_label = 'Wavelength (nm)'
                else:
                    x_label = 'Features'
            except (ValueError, TypeError):
                # If headers are not numeric, use them as categorical labels
                x_values = np.arange(n_features)
                x_label = 'Features'
        else:
            x_values = np.arange(n_features)
            x_label = 'Features'

        # Create colormap - discrete for classification, continuous for regression
        if is_classification:
            # Use discrete colormap for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            if n_unique <= 10:
                colormap = cm.get_cmap('tab10', n_unique)
            elif n_unique <= 20:
                colormap = cm.get_cmap('tab20', n_unique)
            else:
                colormap = cm.get_cmap('hsv', n_unique)

            # Create mapping from actual values to discrete indices
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_sorted])
            y_min, y_max = 0, n_unique - 1
        else:
            # Use continuous colormap for regression
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
        subtitle = f"{processing_name} - ({len(y_sorted)} samples × {x_sorted.shape[1]} features)"
        ax.set_title(subtitle, fontsize=10)

        # Add colorbar to show the y-value gradient
        if is_classification:
            # Discrete colorbar for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            import matplotlib.colors as mcolors
            boundaries = np.arange(n_unique + 1) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, colormap.N)

            mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
            mappable.set_array(np.arange(n_unique))

            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10, pad=0.1,
                              boundaries=boundaries, ticks=np.arange(n_unique))

            # Set tick labels to actual class values
            if n_unique <= 20:
                cbar.ax.set_yticklabels([str(val) for val in unique_values])
            else:
                step = max(1, n_unique // 10)
                cbar.set_ticks(np.arange(0, n_unique, step).tolist())
                cbar.ax.set_yticklabels([str(unique_values[i]) for i in range(0, n_unique, step)])
        else:
            # Continuous colorbar for regression
            mappable = cm.ScalarMappable(cmap=colormap)
            mappable.set_array(y_sorted)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=10, pad=0.1)

        cbar.set_label('y', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

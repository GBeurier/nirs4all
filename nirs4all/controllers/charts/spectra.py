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
from nirs4all.core.logging import get_logger
from nirs4all.utils.header_units import get_x_values_and_label, apply_x_axis_limits
import io

logger = get_logger(__name__)
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
        runtime_context: Any,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple['ExecutionContext', Any]:
        """
        Execute spectra visualization for both 2D and 3D plots.
        Skips execution in prediction mode.

        Supports optional parameters via dict syntax:
            {"chart_2d": {"include_excluded": True, "highlight_excluded": True}}

        Args:
            include_excluded: If True, include excluded samples in visualization
            highlight_excluded: If True, highlight excluded samples with different style

        Returns:
            Tuple of (context, StepOutput)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Extract step config for compatibility
        step = step_info.original_step

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, StepOutput()

        # Check if step is a dict with configuration
        is_3d = False
        include_excluded = False
        highlight_excluded = False

        if isinstance(step, dict):
            for key in ["chart_2d", "chart_3d", "2d_chart", "3d_chart"]:
                if key in step:
                    is_3d = key in ["chart_3d", "3d_chart"]
                    config = step[key] if isinstance(step[key], dict) else {}
                    include_excluded = config.get("include_excluded", False)
                    highlight_excluded = config.get("highlight_excluded", False)
                    break
        else:
            is_3d = (step == "chart_3d") or (step == "3d_chart")

        # Initialize image list to track generated plots
        img_list = []

        # Get sample indices (respecting include_excluded setting)
        sample_indices = dataset._indexer.x_indices(  # noqa: SLF001
            context.selector, include_augmented=True, include_excluded=include_excluded
        )

        # Get excluded mask for highlighting if needed
        excluded_mask = None
        if highlight_excluded and include_excluded:
            # Get indices of excluded samples
            all_indices = dataset._indexer.x_indices(  # noqa: SLF001
                context.selector, include_augmented=True, include_excluded=True
            )
            included_indices = dataset._indexer.x_indices(  # noqa: SLF001
                context.selector, include_augmented=True, include_excluded=False
            )
            excluded_mask = np.isin(all_indices, included_indices, invert=True)

        # Use context directly as it is immutable-ish and we only read from it
        # Use include_excluded for data retrieval if specified
        selector_with_excluded = context.selector
        if include_excluded:
            # Modify selector to include excluded samples
            selector_with_excluded = {"sample": sample_indices.tolist()}

        spectra_data = dataset.x(selector_with_excluded, "3d", False, include_excluded=include_excluded)
        y = dataset.y(selector_with_excluded, include_excluded=include_excluded)

        if not isinstance(spectra_data, list):
            spectra_data = [spectra_data]

        # Sort samples by y values (from lower to higher)
        y_flat = y.flatten() if y.ndim > 1 else y
        sorted_indices = np.argsort(y_flat)
        y_sorted = y_flat[sorted_indices]

        # Sort excluded mask if present
        excluded_sorted = None
        if excluded_mask is not None:
            excluded_sorted = excluded_mask[sorted_indices]

        for sd_idx, x in enumerate(spectra_data):
            processing_ids = dataset.features_processings(sd_idx)
            n_processings = x.shape[1]

            # Debug: print what we got
            if runtime_context.step_runner.verbose > 0:
                logger.debug(f"   Source {sd_idx}: {n_processings} processings: {processing_ids}")
                logger.debug(f"   Data shape: {x.shape}")

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

                if runtime_context.step_runner.verbose > 0 and processing_idx == 0:
                    logger.debug(f"   Headers available: {len(spectra_headers) if spectra_headers else 0}, features: {current_n_features}")

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
                    self._plot_3d_spectra(ax, x_sorted, y_sorted, short_name, processing_headers, header_unit, is_classification, excluded_sorted)
                else:
                    ax = fig.add_subplot(n_rows, n_cols, processing_idx + 1)
                    self._plot_2d_spectra(ax, x_sorted, y_sorted, short_name, processing_headers, header_unit, is_classification, excluded_sorted)

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
            # image_name += ".png" # Extension handled by StepOutput tuple

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

            # Since we iterate over sources, we might have multiple charts.
            # However, StepOutput currently supports a list of outputs.
            # But the loop structure here returns after the first source if we just return step_output.
            # Wait, the original code was iterating but returning `img_list` which accumulated results?
            # Ah, the original code had `img_list.append` inside the loop, but then returned `img_list` AFTER the loop?
            # Let's check the original code again.

            # Original code:
            # for sd_idx, x in enumerate(spectra_data):
            #    ...
            #    if output_path: img_list.append(...)
            #    ...
            # return context, img_list

            # So I need to accumulate outputs in a list and return one StepOutput at the end.

            img_list.append((img_png_binary, image_name, "png"))

        return context, StepOutput(outputs=img_list)

    def _plot_2d_spectra(
        self,
        ax,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        processing_name: str,
        headers: Optional[List[str]] = None,
        header_unit: str = "cm-1",
        is_classification: bool = False,
        excluded_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot 2D spectra on given axis.

        Args:
            ax: Matplotlib axis
            x_sorted: Sorted spectra data
            y_sorted: Sorted target values
            processing_name: Name of processing for title
            headers: Optional wavelength headers
            header_unit: Unit for headers (cm-1, nm, etc.)
            is_classification: Whether this is a classification task
            excluded_mask: Optional boolean mask where True = excluded sample
        """
        # Get feature x-values and axis label using centralized utility
        n_features = x_sorted.shape[1]
        x_values, x_label = get_x_values_and_label(headers, header_unit, n_features)

        # Create colormap - discrete for classification, continuous for regression
        if is_classification:
            # Use discrete colormap for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            if n_unique <= 10:
                colormap = plt.colormaps['tab10'].resampled(n_unique)
            elif n_unique <= 20:
                colormap = plt.colormaps['tab20'].resampled(n_unique)
            else:
                colormap = plt.colormaps['hsv'].resampled(n_unique)

            # Create mapping from actual values to discrete indices
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_sorted])
            y_min, y_max = 0, n_unique - 1
        else:
            # Use continuous colormap for regression
            colormap = plt.colormaps['viridis']
            y_min, y_max = y_sorted.min(), y_sorted.max()

            # Normalize y values to [0, 1] for colormap
            if y_max != y_min:
                y_normalized = (y_sorted - y_min) / (y_max - y_min)
            else:
                y_normalized = np.zeros_like(y_sorted)

        # Count excluded samples for subtitle
        n_excluded = 0
        if excluded_mask is not None:
            n_excluded = excluded_mask.sum()

        # Plot each spectrum as a 2D line with gradient colors
        for i, spectrum in enumerate(x_sorted):
            color = colormap(y_normalized[i])

            # Check if this sample is excluded and should be highlighted
            if excluded_mask is not None and excluded_mask[i]:
                # Highlight excluded samples with dashed red line
                ax.plot(x_values, spectrum,
                        color='red', alpha=0.8, linewidth=1.5, linestyle='--')
            else:
                ax.plot(x_values, spectrum,
                        color=color, alpha=0.7, linewidth=1)

        # Apply x-axis limits to preserve data ordering
        apply_x_axis_limits(ax, x_values)

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)

        # Subtitle with preprocessing name and dimensions
        subtitle = f"{processing_name} - ({len(y_sorted)} samples × {x_sorted.shape[1]} features)"
        if n_excluded > 0:
            subtitle += f" [{n_excluded} excluded]"
        ax.set_title(subtitle, fontsize=10)

        # Add colorbar to show the y-value gradient
        if is_classification:
            # Discrete colorbar for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            boundaries = np.arange(n_unique + 1) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, colormap.N)

            mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
            mappable.set_array(np.arange(n_unique))

            cbar = plt.colorbar(
                mappable, ax=ax, shrink=0.8, aspect=10,
                boundaries=boundaries, ticks=np.arange(n_unique)
            )

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

    def _plot_3d_spectra(
        self,
        ax,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        processing_name: str,
        headers: Optional[List[str]] = None,
        header_unit: str = "cm-1",
        is_classification: bool = False,
        excluded_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot 3D spectra on given axis.

        Args:
            ax: Matplotlib 3D axis
            x_sorted: Sorted spectra data
            y_sorted: Sorted target values
            processing_name: Name of processing for title
            headers: Optional wavelength headers
            header_unit: Unit for headers (cm-1, nm, etc.)
            is_classification: Whether this is a classification task
            excluded_mask: Optional boolean mask where True = excluded sample
        """
        # Get feature x-values and axis label using centralized utility
        n_features = x_sorted.shape[1]
        x_values, x_label = get_x_values_and_label(headers, header_unit, n_features)

        # Create colormap - discrete for classification, continuous for regression
        if is_classification:
            # Use discrete colormap for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            if n_unique <= 10:
                colormap = plt.colormaps['tab10'].resampled(n_unique)
            elif n_unique <= 20:
                colormap = plt.colormaps['tab20'].resampled(n_unique)
            else:
                colormap = plt.colormaps['hsv'].resampled(n_unique)

            # Create mapping from actual values to discrete indices
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}
            y_normalized = np.array([value_to_index[val] / max(n_unique - 1, 1) for val in y_sorted])
            y_min, y_max = 0, n_unique - 1
        else:
            # Use continuous colormap for regression
            colormap = plt.colormaps['viridis']
            y_min, y_max = y_sorted.min(), y_sorted.max()

            # Normalize y values to [0, 1] for colormap
            if y_max != y_min:
                y_normalized = (y_sorted - y_min) / (y_max - y_min)
            else:
                y_normalized = np.zeros_like(y_sorted)

        # Count excluded samples for subtitle
        n_excluded = 0
        if excluded_mask is not None:
            n_excluded = excluded_mask.sum()

        # Plot each spectrum as a line in 3D space with gradient colors
        for i, (spectrum, y_val) in enumerate(zip(x_sorted, y_sorted)):
            color = colormap(y_normalized[i])

            # Check if this sample is excluded and should be highlighted
            if excluded_mask is not None and excluded_mask[i]:
                # Highlight excluded samples with dashed red line
                ax.plot(x_values, [y_val] * n_features, spectrum,
                        color='red', alpha=0.8, linewidth=1.5, linestyle='--')
            else:
                ax.plot(x_values, [y_val] * n_features, spectrum,
                        color=color, alpha=0.7, linewidth=1)

        # Apply x-axis limits to preserve data ordering
        apply_x_axis_limits(ax, x_values)

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('y (sorted)', fontsize=9)
        ax.set_zlabel('Intensity', fontsize=9)

        # Subtitle with preprocessing name and dimensions
        subtitle = f"{processing_name} - ({len(y_sorted)} samples × {x_sorted.shape[1]} features)"
        if n_excluded > 0:
            subtitle += f" [{n_excluded} excluded]"
        ax.set_title(subtitle, fontsize=10)

        # Add colorbar to show the y-value gradient
        if is_classification:
            # Discrete colorbar for classification
            unique_values = np.unique(y_sorted)
            n_unique = len(unique_values)

            boundaries = np.arange(n_unique + 1) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, colormap.N)

            mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
            mappable.set_array(np.arange(n_unique))

            cbar = plt.colorbar(
                mappable, ax=ax, shrink=0.8, aspect=10, pad=0.1,
                boundaries=boundaries, ticks=np.arange(n_unique)
            )

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

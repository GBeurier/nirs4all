"""SpectraChartController - Unified 2D and 3D spectra visualization controller."""

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
class SpectraChartController(OperatorController):

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword in ["chart_2d", "chart_3d"]

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute spectra visualization for both 2D and 3D plots.

        Returns:
            Tuple of (context, image_list) where image_list contains plot metadata
        """
        is_3d = step == "chart_3d"

        # Initialize image list to track generated plots
        img_list = []

        local_context = context.copy()
        spectra_data = dataset.x(local_context, "3d", False)
        y = dataset.y(local_context)

        if not isinstance(spectra_data, list):
            spectra_data = [spectra_data]

        for sd_idx, x in enumerate(spectra_data):
            # Sort samples by y values (from lower to higher)
            y_flat = y.flatten() if y.ndim > 1 else y
            sorted_indices = np.argsort(y_flat)
            processing_ids = dataset.features_processings(sd_idx)

            # Process each processing type in the 3D data
            for processing_idx in range(x.shape[1]):
                processing_name = processing_ids[processing_idx]

                # Get 2D data for this processing: (samples, features)
                x_2d = x[:, processing_idx, :]

                # Sort the data by y values
                x_sorted = x_2d[sorted_indices]
                y_sorted = y_flat[sorted_indices]

                if is_3d:
                    fig, plot_info = self._create_3d_plot(
                        x_sorted, y_sorted, processing_name, sd_idx, dataset.is_multi_source()
                    )
                else:
                    fig, plot_info = self._create_2d_plot(
                        x_sorted, y_sorted, processing_name, sd_idx, dataset.is_multi_source()
                    )

                # Save plot to memory buffer as PNG binary
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img_png_binary = img_buffer.getvalue()  # Get PNG binary data directly
                img_buffer.close()

                # # Add plot metadata to image list
                # img_info = {
                #     'plot_type': '3D' if is_3d else '2D',
                #     'source_idx': sd_idx,
                #     'processing_idx': processing_idx,
                #     'n_samples': len(y_sorted),
                #     'n_features': x_2d.shape[1],
                #     'y_range': (float(y_sorted.min()), float(y_sorted.max())),
                #     'image_base64': img_base64,
                #     'title': plot_info['title'],
                #     'figure_size': plot_info['figure_size']
                # }
                # img_list.append(img_info)

                # Show the plot
                # plt.show()
                image_name = "2D" if not is_3d else "3D"
                image_name += f"-{processing_name}_samples"
                if dataset.is_multi_source():
                    image_name += f" (src {sd_idx})"
                image_name += ".png"
                img_list.append((image_name, img_png_binary))
                plt.close(fig)

        return context, img_list

    def _create_2d_plot(self, x_sorted: np.ndarray, y_sorted: np.ndarray,
                        processing_name: str, sd_idx: int, is_multi_source: bool) -> Tuple[Any, Dict[str, Any]]:
        """Create 2D spectra plot."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # Create feature indices (wavelengths)
        n_features = x_sorted.shape[1]
        feature_indices = np.arange(n_features)

        # Create colormap for gradient based on y values
        colormap = cm.get_cmap('viridis')
        y_min, y_max = y_sorted.min(), y_sorted.max()

        # Normalize y values to [0, 1] for colormap
        if y_max != y_min:
            y_normalized = (y_sorted - y_min) / (y_max - y_min)
        else:
            y_normalized = np.zeros_like(y_sorted)

        # Plot each spectrum as a 2D line with gradient colors
        for i, spectrum in enumerate(x_sorted):
            color = colormap(y_normalized[i])
            ax.plot(feature_indices, spectrum,
                    color=color, alpha=0.7, linewidth=1)

        ax.set_xlabel('x (features)')
        ax.set_ylabel('Intensity')
        title = f"Samples: ({len(y_sorted)}, {x_sorted.shape[1]}), Process: {processing_name}"
        if is_multi_source:
            title = title + f", Src: {sd_idx}"
        ax.set_title(title, fontsize=10)

        # Add colorbar to show the y-value gradient
        mappable = cm.ScalarMappable(cmap=colormap)
        mappable.set_array(y_sorted)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('y (targets)')

        plot_info = {
            'title': title,
            'figure_size': (12, 8)
        }

        return fig, plot_info

    def _create_3d_plot(self, x_sorted: np.ndarray, y_sorted: np.ndarray,
                        processing_name: str, sd_idx: int, is_multi_source: bool) -> Tuple[Any, Dict[str, Any]]:
        """Create 3D spectra plot."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create feature indices (wavelengths)
        n_features = x_sorted.shape[1]
        feature_indices = np.arange(n_features)

        # Create colormap for gradient based on y values
        colormap = cm.get_cmap('viridis')
        y_min, y_max = y_sorted.min(), y_sorted.max()

        # Normalize y values to [0, 1] for colormap
        if y_max != y_min:
            y_normalized = (y_sorted - y_min) / (y_max - y_min)
        else:
            y_normalized = np.zeros_like(y_sorted)

        # Plot each spectrum as a line in 3D space with gradient colors
        for i, (spectrum, y_val) in enumerate(zip(x_sorted, y_sorted)):
            color = colormap(y_normalized[i])
            ax.plot(feature_indices, [y_val] * n_features, spectrum,
                    color=color, alpha=0.7, linewidth=1)

        ax.set_xlabel('x (features)')
        ax.set_ylabel('y (sorted)')
        ax.set_zlabel('Intensity')
        title = f"Samples: ({len(y_sorted)}, {x_sorted.shape[1]}), Process: {processing_name}"
        if is_multi_source:
            title = title + f", Src: {sd_idx}"
        ax.set_title(title, fontsize=10)

        # Add colorbar to show the y-value gradient
        mappable = cm.ScalarMappable(cmap=colormap)
        mappable.set_array(y_sorted)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('y (targets)')

        plot_info = {
            'title': title,
            'figure_size': (12, 8)
        }

        return fig, plot_info

"""DummyController.py - A dummy controller for testing purposes in the nirs4all pipeline."""

from typing import Any, Dict, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from sklearn.base import TransformerMixin
if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

@register_controller
class SpectraChartController3D(OperatorController):

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "chart_3d"

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
    ):
        print(f"Executing spectra charts for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")

        local_context = context.copy()
        # local_context["partition"] = "train"
        print("Local context:", local_context)
        spectra_data = dataset.x(local_context, "3d", False)
        y = dataset.y(local_context)

        if not isinstance(spectra_data, list):
            spectra_data = [spectra_data]

        for sd_idx, x in enumerate(spectra_data):
            print("Spectra data shape:", x.shape, "Y shape:", y.shape)

            # Sort samples by y values (from lower to higher)
            y_flat = y.flatten() if y.ndim > 1 else y
            sorted_indices = np.argsort(y_flat)

            # Process each processing type in the 3D data
            for processing_idx in range(x.shape[1]):
                # Get 2D data for this processing: (samples, features)
                x_2d = x[:, processing_idx, :]

                # Sort the data by y values
                x_sorted = x_2d[sorted_indices]
                y_sorted = y_flat[sorted_indices]

                # Create 3D plot
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Create feature indices (wavelengths)
                n_features = x_2d.shape[1]
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

                ax.set_xlabel('Feature Index (Wavelength)')
                ax.set_ylabel('Target Values (sorted)')
                ax.set_zlabel('Spectral Intensity')
                ax.set_title(f'3D Spectra Visualization - Processing {processing_idx}\n'
                             f'Data Source: {sd_idx}, Samples: {len(y_sorted)}')

                # Add colorbar to show the y-value gradient
                mappable = cm.ScalarMappable(cmap=colormap)
                mappable.set_array(y_sorted)
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
                cbar.set_label('Target Values')

                plt.show()

        return context

"""AugmentationChartController - Visualizes augmentation effects on spectra."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import io
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.utils.header_units import get_x_values_and_label, apply_x_axis_limits

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class AugmentationChartController(OperatorController):
    """
    Controller for visualizing augmentation effects on spectra.

    Supports two visualization modes:
    1. augment_chart: Shows original vs augmented samples overlaid with different colors
    2. augment_details_chart: Shows a grid with raw data and each augmentation type separately
    """

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword in ["augment_chart", "augmentation_chart", "augment_details_chart", "augmentation_details_chart"]

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

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
        Execute augmentation visualization.

        Returns:
            Tuple of (context, StepOutput)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Extract step config for compatibility
        step = step_info.original_step
        keyword = context.metadata.keyword

        # Skip execution in prediction mode
        if mode == "predict" or mode == "explain":
            return context, StepOutput()

        is_details = keyword in ["augment_details_chart", "augmentation_details_chart"]

        # Get configuration from step if it's a dict
        config = {}
        if isinstance(step, dict):
            config = step.get(keyword, {})

        alpha_original = config.get("alpha_original", 0.8)
        alpha_augmented = config.get("alpha_augmented", 0.4)
        max_samples = config.get("max_samples", 50)  # Limit samples for readability

        # Initialize image list
        img_list = []

        # Get train context
        train_context = context.with_partition("train")

        # Get spectra data for visualization (use first source, first processing)
        spectra_data = dataset.x(train_context.selector, "3d", False)
        if not isinstance(spectra_data, list):
            spectra_data = [spectra_data]

        for sd_idx, x in enumerate(spectra_data):
            # Get base and augmented sample indices
            base_indices = dataset._indexer.x_indices(train_context.selector, include_augmented=False)
            all_indices = dataset._indexer.x_indices(train_context.selector, include_augmented=True)

            # Separate augmented indices
            base_set = set(base_indices.tolist() if hasattr(base_indices, 'tolist') else list(base_indices))
            augmented_indices = [idx for idx in all_indices if idx not in base_set]

            n_base = len(base_indices)
            n_augmented = len(augmented_indices)

            if runtime_context.step_runner.verbose > 0:
                logger.debug(f"   Source {sd_idx}: {n_base} base samples, {n_augmented} augmented samples")

            # Get first processing (raw or first preprocessed)
            processing_ids = dataset.features_processings(sd_idx)

            if is_details:
                # Details mode: show raw + each augmentation type
                fig = self._create_details_chart(
                    x, base_indices, augmented_indices, all_indices,
                    processing_ids, dataset, sd_idx,
                    alpha_original, alpha_augmented, max_samples
                )
                image_name = "Augmentation_Details_Chart"
            else:
                # Overlay mode: show original vs augmented overlaid
                fig = self._create_overlay_chart(
                    x, base_indices, augmented_indices, all_indices,
                    processing_ids, dataset, sd_idx,
                    alpha_original, alpha_augmented, max_samples
                )
                image_name = "Augmentation_Chart"

            if dataset.is_multi_source():
                image_name += f"_src{sd_idx}"

            # Save plot to memory buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_png_binary = img_buffer.getvalue()
            img_buffer.close()

            img_list.append((img_png_binary, image_name, "png"))

            if runtime_context.step_runner.plots_visible:
                runtime_context.step_runner._figure_refs.append(fig)
                plt.show()
            else:
                plt.close(fig)

        return context, StepOutput(outputs=img_list)

    def _create_overlay_chart(
        self,
        x: np.ndarray,
        base_indices: np.ndarray,
        augmented_indices: List[int],
        all_indices: np.ndarray,
        processing_ids: List[str],
        dataset: 'SpectroDataset',
        source_idx: int,
        alpha_original: float,
        alpha_augmented: float,
        max_samples: int
    ) -> Figure:
        """
        Create overlay chart showing original (blue) and augmented (orange) samples.
        """
        n_processings = x.shape[1]

        # Calculate subplot grid
        n_cols = min(3, n_processings)
        n_rows = (n_processings + n_cols - 1) // n_cols

        fig_width = 6 * n_cols
        fig_height = 5 * n_rows
        fig = plt.figure(figsize=(fig_width, fig_height))

        main_title = f"{dataset.name} - Augmentation Overlay"
        if dataset.is_multi_source():
            main_title += f" (Source {source_idx})"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

        # Map sample indices to array positions
        all_indices_list = all_indices.tolist() if hasattr(all_indices, 'tolist') else list(all_indices)
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices_list)}

        # Limit samples for readability
        base_indices_list = base_indices.tolist() if hasattr(base_indices, 'tolist') else list(base_indices)
        if len(base_indices_list) > max_samples:
            np.random.seed(42)
            base_indices_list = list(np.random.choice(base_indices_list, max_samples, replace=False))

        if len(augmented_indices) > max_samples:
            np.random.seed(42)
            augmented_indices = list(np.random.choice(augmented_indices, max_samples, replace=False))

        # Get headers
        spectra_headers = dataset.headers(source_idx)
        try:
            header_unit = dataset.header_unit(source_idx)
        except (AttributeError, IndexError):
            header_unit = "cm-1"

        for processing_idx in range(n_processings):
            processing_name = self._shorten_processing_name(processing_ids[processing_idx])

            ax = fig.add_subplot(n_rows, n_cols, processing_idx + 1)

            # Get 2D data for this processing
            x_2d = x[:, processing_idx, :]
            n_features = x_2d.shape[1]

            # Determine x-axis values
            if spectra_headers and len(spectra_headers) == n_features:
                try:
                    x_values = np.array([float(h) for h in spectra_headers])
                    x_label = 'Wavenumber (cm⁻¹)' if header_unit == "cm-1" else 'Wavelength (nm)' if header_unit == "nm" else 'Features'
                except (ValueError, TypeError):
                    x_values = np.arange(n_features)
                    x_label = 'Features'
            else:
                x_values = np.arange(n_features)
                x_label = 'Features'

            # Plot original samples (blue)
            for idx in base_indices_list:
                if idx in idx_to_pos:
                    pos = idx_to_pos[idx]
                    ax.plot(x_values, x_2d[pos], color='steelblue', alpha=alpha_original, linewidth=0.8)

            # Plot augmented samples (orange)
            for idx in augmented_indices:
                if idx in idx_to_pos:
                    pos = idx_to_pos[idx]
                    ax.plot(x_values, x_2d[pos], color='darkorange', alpha=alpha_augmented, linewidth=0.8)

            # Force axis order
            if len(x_values) > 1 and x_values[0] > x_values[-1]:
                ax.set_xlim(x_values[0], x_values[-1])

            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel('Intensity', fontsize=9)

            subtitle = f"{processing_name}"
            ax.set_title(subtitle, fontsize=10)

            # Add legend only to first subplot
            if processing_idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='steelblue', linewidth=2, label=f'Original ({len(base_indices_list)})'),
                    Line2D([0], [0], color='darkorange', linewidth=2, label=f'Augmented ({len(augmented_indices)})')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout(rect=(0, 0, 1, 0.92), h_pad=4.0)  # type: ignore[arg-type]
        return fig

    def _create_details_chart(
        self,
        x: np.ndarray,
        base_indices: np.ndarray,
        augmented_indices: List[int],
        all_indices: np.ndarray,
        processing_ids: List[str],
        dataset: 'SpectroDataset',
        source_idx: int,
        alpha_original: float,
        alpha_augmented: float,
        max_samples: int
    ) -> Figure:
        """
        Create details chart showing raw on top-left, then each augmented transformation separately.

        This groups augmented samples by their transformer (via origin mapping and metadata).
        """
        _n_processings = x.shape[1]  # Available for future use

        # Get augmentation info from dataset indexer
        # Group augmented samples by their transformer
        transformer_groups = self._group_augmented_by_transformer(dataset, augmented_indices)

        # Calculate layout: raw + one subplot per transformer
        n_transformers = len(transformer_groups)
        n_subplots = 1 + n_transformers  # Raw + each transformer

        # For details, we show first processing only but multiple transformers
        n_cols = min(3, n_subplots)
        n_rows = (n_subplots + n_cols - 1) // n_cols

        fig_width = 6 * n_cols
        fig_height = 5 * n_rows
        fig = plt.figure(figsize=(fig_width, fig_height))

        main_title = f"{dataset.name} - Augmentation Details"
        if dataset.is_multi_source():
            main_title += f" (Source {source_idx})"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

        # Map sample indices to array positions
        all_indices_list = all_indices.tolist() if hasattr(all_indices, 'tolist') else list(all_indices)
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices_list)}

        # Get first processing data (or raw)
        x_2d = x[:, 0, :]  # First processing
        n_features = x_2d.shape[1]

        # Get headers and determine x-axis values using centralized utility
        spectra_headers = dataset.headers(source_idx)
        try:
            header_unit = dataset.header_unit(source_idx)
        except (AttributeError, IndexError):
            header_unit = "cm-1"

        x_values, x_label = get_x_values_and_label(spectra_headers, header_unit, n_features)

        base_indices_list = base_indices.tolist() if hasattr(base_indices, 'tolist') else list(base_indices)

        # Limit samples
        if len(base_indices_list) > max_samples:
            np.random.seed(42)
            base_indices_list = list(np.random.choice(base_indices_list, max_samples, replace=False))

        # Plot 1: Raw/Original samples only
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        for idx in base_indices_list:
            if idx in idx_to_pos:
                pos = idx_to_pos[idx]
                ax1.plot(x_values, x_2d[pos], color='steelblue', alpha=alpha_original, linewidth=0.8)

        apply_x_axis_limits(ax1, x_values)

        ax1.set_xlabel(x_label, fontsize=9)
        ax1.set_ylabel('Intensity', fontsize=9)
        processing_name = self._shorten_processing_name(processing_ids[0])
        ax1.set_title(f"Original ({len(base_indices_list)} samples) - {processing_name}", fontsize=10)

        # Get colormap for different transformers
        cmap = plt.colormaps['Set2']
        colors = cmap(np.linspace(0, 1, max(n_transformers, 1)))

        # Plot each transformer group
        for t_idx, (transformer_name, aug_indices) in enumerate(transformer_groups.items()):
            ax = fig.add_subplot(n_rows, n_cols, t_idx + 2)

            # Plot original in background (light gray)
            for idx in base_indices_list:
                if idx in idx_to_pos:
                    pos = idx_to_pos[idx]
                    ax.plot(x_values, x_2d[pos], color='lightgray', alpha=0.3, linewidth=0.5)

            # Limit augmented samples
            aug_indices_limited = aug_indices
            if len(aug_indices) > max_samples:
                np.random.seed(42 + t_idx)
                aug_indices_limited = list(np.random.choice(aug_indices, max_samples, replace=False))

            # Plot augmented samples for this transformer
            for idx in aug_indices_limited:
                if idx in idx_to_pos:
                    pos = idx_to_pos[idx]
                    ax.plot(x_values, x_2d[pos], color=colors[t_idx], alpha=alpha_augmented, linewidth=0.8)

            apply_x_axis_limits(ax, x_values)

            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel('Intensity', fontsize=9)
            ax.set_title(f"{transformer_name} ({len(aug_indices)} samples)", fontsize=10)

        plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)  # type: ignore[arg-type]
        return fig

    def _group_augmented_by_transformer(
        self,
        dataset: 'SpectroDataset',
        augmented_indices: List[int]
    ) -> Dict[str, List[int]]:
        """
        Group augmented samples by their transformer type.

        Uses the 'augmentation' column in the indexer to identify transformer types.

        Returns a dict: {transformer_name: [sample_indices]}
        """
        import polars as pl
        groups: Dict[str, List[int]] = {}

        if not augmented_indices:
            return groups

        # Get augmentation info from the indexer DataFrame
        df = dataset._indexer.df  # noqa: SLF001

        # Filter to only augmented samples
        aug_df = df.filter(pl.col("sample").is_in(augmented_indices))

        # Group by augmentation type
        for row in aug_df.iter_rows(named=True):
            sample_id = row["sample"]
            aug_type = row.get("augmentation", None)

            if aug_type is None:
                transformer_name = "Augmented"
            else:
                # Extract class name from augmentation ID if it contains class info
                transformer_name = str(aug_type)
                # Shorten common names
                if "Rotate_Translate" in transformer_name:
                    transformer_name = "Rotate_Translate"
                elif "Spline" in transformer_name:
                    # Extract spline type
                    if "Y_Perturbations" in transformer_name:
                        transformer_name = "Spline_Y"
                    elif "X_Perturbations" in transformer_name:
                        transformer_name = "Spline_X"
                    elif "Curve" in transformer_name:
                        transformer_name = "Spline_Curve"
                    elif "Simplification" in transformer_name:
                        transformer_name = "Spline_Simplify"
                    else:
                        transformer_name = "Spline"
                elif "Random" in transformer_name:
                    transformer_name = "Random_Op"

            if transformer_name not in groups:
                groups[transformer_name] = []
            groups[transformer_name].append(sample_id)

        # If no groups found, put all in one group
        if not groups and augmented_indices:
            groups["Augmented"] = augmented_indices

        return groups

    @staticmethod
    def _shorten_processing_name(processing_name: str) -> str:
        """Shorten preprocessing names for chart titles."""
        import re
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
        ]
        for long, short in replacements:
            processing_name = processing_name.replace(long, short)

        processing_name = re.sub(r'_\d+_', '>', processing_name)
        processing_name = re.sub(r'_\d+', '', processing_name)
        return processing_name

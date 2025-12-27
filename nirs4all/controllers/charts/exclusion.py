"""
ExclusionChartController - Visualizes excluded vs included samples.

This controller creates 2D scatter plots showing which samples have been
marked as excluded by sample filtering operations. Useful for understanding
filtering decisions and identifying patterns in excluded data.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
import io

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.steps.runtime import RuntimeContext


@register_controller
class ExclusionChartController(OperatorController):
    """
    Controller for visualizing sample exclusions.

    Creates 2D scatter plots using PCA to show the relationship between
    included and excluded samples. Supports coloring by:
    - Exclusion status (included vs excluded)
    - Target values (y)
    - Exclusion reason

    Pipeline syntax:
        "exclusion_chart"  # Basic exclusion visualization

        {"exclusion_chart": {"color_by": "y"}}  # Color by target values

        {"exclusion_chart": {"color_by": "reason"}}  # Color by exclusion reason

        {"exclusion_chart": {
            "n_components": 3,  # Use 3D PCA
            "show_legend": True,
            "title": "Custom Title"
        }}
    """

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match exclusion_chart keyword."""
        return keyword in ["exclusion_chart", "chart_exclusion"]

    @classmethod
    def use_multi_source(cls) -> bool:
        """Operates at dataset level."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Chart controllers skip during prediction."""
        return False

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', Any]:
        """
        Execute exclusion visualization.

        Creates a 2D (or 3D) scatter plot showing included vs excluded samples
        using PCA for dimensionality reduction.

        Args:
            step_info: Parsed step containing operator and configuration
            dataset: Dataset to visualize
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index (unused)
            mode: Execution mode
            loaded_binaries: Pre-loaded binaries (unused)
            prediction_store: External prediction store (unused)

        Returns:
            Tuple of (context, StepOutput with chart image)
        """
        from nirs4all.pipeline.execution.result import StepOutput

        # Skip in prediction mode
        if mode in ("predict", "explain"):
            return context, StepOutput()

        # Extract configuration
        step = step_info.original_step
        if isinstance(step, dict) and "exclusion_chart" in step:
            config = step["exclusion_chart"] if isinstance(step["exclusion_chart"], dict) else {}
        elif isinstance(step, dict) and "chart_exclusion" in step:
            config = step["chart_exclusion"] if isinstance(step["chart_exclusion"], dict) else {}
        else:
            config = {}

        color_by = config.get("color_by", "status")  # "status", "y", "reason"
        n_components = config.get("n_components", 2)
        show_legend = config.get("show_legend", True)
        custom_title = config.get("title", None)
        partition = config.get("partition", "train")

        # Get exclusion summary first to check if there are any exclusions
        summary = dataset._indexer.get_exclusion_summary()  # noqa: SLF001

        if summary["total_excluded"] == 0:
            if runtime_context.step_runner.verbose > 0:
                logger.info("   ExclusionChart: No excluded samples to visualize")
            return context, StepOutput()

        # Create the visualization
        img_list = self._create_exclusion_chart(
            dataset=dataset,
            context=context,
            partition=partition,
            color_by=color_by,
            n_components=n_components,
            show_legend=show_legend,
            custom_title=custom_title,
            runtime_context=runtime_context,
        )

        return context, StepOutput(outputs=img_list)

    def _create_exclusion_chart(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        partition: str,
        color_by: str,
        n_components: int,
        show_legend: bool,
        custom_title: Optional[str],
        runtime_context: 'RuntimeContext',
    ) -> List[Tuple[bytes, str, str]]:
        """
        Create exclusion visualization chart.

        Args:
            dataset: Dataset to visualize
            context: Execution context
            partition: Partition to visualize ("train", "test", or None for all)
            color_by: How to color points ("status", "y", "reason")
            n_components: PCA components (2 or 3)
            show_legend: Whether to show legend
            custom_title: Optional custom title
            runtime_context: Runtime context for figure management

        Returns:
            List of (image_bytes, name, format) tuples
        """
        # Build selector for partition
        if partition:
            base_selector = context.selector.with_partition(partition)
        else:
            base_selector = context.selector

        # Get included samples
        included_indices = dataset._indexer.x_indices(  # noqa: SLF001
            base_selector, include_augmented=False, include_excluded=False
        )

        # Get excluded samples
        excluded_indices = dataset._indexer.x_indices(  # noqa: SLF001
            base_selector, include_augmented=False, include_excluded=True
        )
        # Filter to only excluded ones
        excluded_mask = np.isin(excluded_indices, included_indices, invert=True)
        excluded_only_indices = excluded_indices[excluded_mask]

        if len(excluded_only_indices) == 0:
            return []

        # Combine for PCA
        all_indices = np.concatenate([included_indices, excluded_only_indices])
        is_excluded = np.concatenate([
            np.zeros(len(included_indices), dtype=bool),
            np.ones(len(excluded_only_indices), dtype=bool)
        ])

        # Get X data for all samples (included + excluded)
        # Must use include_excluded=True since we're visualizing excluded samples
        X_all_raw = dataset.x(
            {"sample": all_indices.tolist()},
            layout="2d",
            concat_source=True,
            include_excluded=True
        )
        # Ensure X_all is a 2D numpy array
        if isinstance(X_all_raw, list):
            X_all = np.vstack(X_all_raw)
        else:
            X_all = X_all_raw

        # Get y values if needed
        y_all: Optional[np.ndarray] = None
        if color_by == "y":
            y_raw = dataset.y({"sample": all_indices.tolist()}, include_excluded=True)
            if y_raw is not None:
                y_all = y_raw.flatten() if y_raw.ndim > 1 else y_raw

        # Get exclusion reasons if needed
        reasons: Optional[np.ndarray] = None
        if color_by == "reason":
            excluded_df = dataset._indexer.get_excluded_samples(base_selector)  # noqa: SLF001
            # Build reason lookup
            reason_lookup: Dict[int, str] = {}
            for row in excluded_df.to_dicts():
                sample_id = row["sample"]
                reason = row["exclusion_reason"] if row["exclusion_reason"] else "unspecified"
                reason_lookup[sample_id] = reason

            reason_list: List[str] = []
            for i, idx in enumerate(all_indices):
                if is_excluded[i]:
                    reason_list.append(reason_lookup.get(int(idx), "unspecified"))
                else:
                    reason_list.append("included")
            reasons = np.array(reason_list)

        # Apply PCA
        n_components = min(n_components, X_all.shape[1], X_all.shape[0])
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_all)

        # Create figure
        if n_components >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(12, 10))

        # Plot based on color_by mode
        if color_by == "status":
            self._plot_by_status(ax, X_reduced, is_excluded, n_components)
        elif color_by == "y" and y_all is not None:
            self._plot_by_y(ax, X_reduced, is_excluded, y_all, n_components, dataset.is_classification)
        elif color_by == "reason" and reasons is not None:
            self._plot_by_reason(ax, X_reduced, is_excluded, reasons, n_components)
        else:
            # Default to status (also fallback if y_all or reasons are None)
            self._plot_by_status(ax, X_reduced, is_excluded, n_components)

        # Set labels
        var_explained = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=11)
        if n_components >= 3:
            ax.set_zlabel(f'PC3 ({var_explained[2]:.1f}%)', fontsize=11)

        # Title
        n_included = len(included_indices)
        n_excluded = len(excluded_only_indices)
        if custom_title:
            title = custom_title
        else:
            pct = 100 * n_excluded / (n_included + n_excluded)
            title = "Sample Exclusion Visualization\n"
            title += f"Included: {n_included} | Excluded: {n_excluded} ({pct:.1f}%)"

        ax.set_title(title, fontsize=14, fontweight='bold')

        if show_legend:
            ax.legend(loc='upper right', fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to buffer
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_png_binary = img_buffer.getvalue()
        img_buffer.close()

        image_name = f"exclusion_chart_{partition}_{color_by}"

        if runtime_context.step_runner.plots_visible:
            runtime_context.step_runner._figure_refs.append(fig)
            plt.show()
        else:
            plt.close(fig)

        return [(img_png_binary, image_name, "png")]

    def _plot_by_status(
        self,
        ax: Any,
        X_reduced: np.ndarray,
        is_excluded: np.ndarray,
        n_components: int
    ) -> None:
        """Plot samples colored by exclusion status."""
        # Use viridis-compatible colors: green-ish for included, red-ish for excluded
        included_color = "#2ecc71"  # Green
        excluded_color = "#e74c3c"  # Red

        included_mask = ~is_excluded
        excluded_mask = is_excluded

        if n_components >= 3:
            ax.scatter(
                X_reduced[included_mask, 0],
                X_reduced[included_mask, 1],
                X_reduced[included_mask, 2],
                c=included_color, alpha=0.6, s=50, label=f'Included ({included_mask.sum()})',
                edgecolors='white', linewidth=0.5
            )
            ax.scatter(
                X_reduced[excluded_mask, 0],
                X_reduced[excluded_mask, 1],
                X_reduced[excluded_mask, 2],
                c=excluded_color, alpha=0.8, s=80, label=f'Excluded ({excluded_mask.sum()})',
                edgecolors='black', linewidth=1, marker='x'
            )
        else:
            ax.scatter(
                X_reduced[included_mask, 0],
                X_reduced[included_mask, 1],
                c=included_color, alpha=0.6, s=50, label=f'Included ({included_mask.sum()})',
                edgecolors='white', linewidth=0.5
            )
            ax.scatter(
                X_reduced[excluded_mask, 0],
                X_reduced[excluded_mask, 1],
                c=excluded_color, alpha=0.8, s=80, label=f'Excluded ({excluded_mask.sum()})',
                edgecolors='black', linewidth=1, marker='x'
            )

    def _plot_by_y(
        self,
        ax: Any,
        X_reduced: np.ndarray,
        is_excluded: np.ndarray,
        y_values: np.ndarray,
        n_components: int,
        is_classification: bool
    ) -> None:
        """Plot samples colored by target value, with excluded highlighted."""
        # Determine colormap
        if is_classification:
            unique_y = np.unique(y_values)
            n_unique = len(unique_y)
            if n_unique <= 10:
                cmap = plt.colormaps['tab10'].resampled(n_unique)
            else:
                cmap = plt.colormaps['viridis']

            # Normalize y to [0, 1]
            y_to_idx = {v: i for i, v in enumerate(unique_y)}
            y_norm = np.array([y_to_idx[v] / max(n_unique - 1, 1) for v in y_values])
        else:
            cmap = plt.colormaps['viridis']
            y_min, y_max = y_values.min(), y_values.max()
            if y_max > y_min:
                y_norm = (y_values - y_min) / (y_max - y_min)
            else:
                y_norm = np.zeros_like(y_values)

        colors = cmap(y_norm)

        included_mask = ~is_excluded
        excluded_mask = is_excluded

        # Plot included samples
        if n_components >= 3:
            ax.scatter(
                X_reduced[included_mask, 0],
                X_reduced[included_mask, 1],
                X_reduced[included_mask, 2],
                c=colors[included_mask], alpha=0.6, s=50,
                edgecolors='white', linewidth=0.5
            )
            # Plot excluded with X markers
            ax.scatter(
                X_reduced[excluded_mask, 0],
                X_reduced[excluded_mask, 1],
                X_reduced[excluded_mask, 2],
                c=colors[excluded_mask], alpha=0.9, s=100,
                edgecolors='black', linewidth=2, marker='X'
            )
        else:
            sc = ax.scatter(
                X_reduced[included_mask, 0],
                X_reduced[included_mask, 1],
                c=colors[included_mask], alpha=0.6, s=50,
                edgecolors='white', linewidth=0.5
            )
            ax.scatter(
                X_reduced[excluded_mask, 0],
                X_reduced[excluded_mask, 1],
                c=colors[excluded_mask], alpha=0.9, s=100,
                edgecolors='black', linewidth=2, marker='X'
            )

            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap)
            sm.set_array(y_values)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Target (y)', fontsize=10)

        # Add legend for markers
        included_patch = mpatches.Patch(color='gray', alpha=0.6, label=f'Included ({included_mask.sum()})')
        excluded_marker = Line2D(
            [0], [0], marker='X', color='w', markerfacecolor='gray',
            markersize=10, markeredgecolor='black', markeredgewidth=2,
            label=f'Excluded ({excluded_mask.sum()})', linestyle='None'
        )
        ax.legend(handles=[included_patch, excluded_marker], loc='upper right')

    def _plot_by_reason(
        self,
        ax: Any,
        X_reduced: np.ndarray,
        is_excluded: np.ndarray,
        reasons: np.ndarray,
        n_components: int
    ) -> None:
        """Plot samples colored by exclusion reason."""
        unique_reasons = np.unique(reasons)
        n_reasons = len(unique_reasons)

        # Use tab10/tab20 for discrete reasons
        if n_reasons <= 10:
            cmap = plt.colormaps['tab10'].resampled(n_reasons)
        else:
            cmap = plt.colormaps['tab20'].resampled(n_reasons)

        reason_to_idx = {r: i for i, r in enumerate(unique_reasons)}
        reason_colors: Dict[str, Any] = {r: cmap(i / max(n_reasons - 1, 1)) for i, r in enumerate(unique_reasons)}

        # Special color for included (RGB tuple)
        reason_colors["included"] = (0.18, 0.8, 0.44, 1.0)  # Green: #2ecc71

        for reason in unique_reasons:
            mask = reasons == reason
            color = reason_colors[reason]
            marker = 'o' if reason == "included" else 'X'
            size = 50 if reason == "included" else 80
            alpha = 0.6 if reason == "included" else 0.8
            edgecolor = 'white' if reason == "included" else 'black'
            linewidth = 0.5 if reason == "included" else 1

            if n_components >= 3:
                ax.scatter(
                    X_reduced[mask, 0],
                    X_reduced[mask, 1],
                    X_reduced[mask, 2],
                    c=[color], alpha=alpha, s=size, label=f'{reason} ({mask.sum()})',
                    edgecolors=edgecolor, linewidth=linewidth, marker=marker
                )
            else:
                ax.scatter(
                    X_reduced[mask, 0],
                    X_reduced[mask, 1],
                    c=[color], alpha=alpha, s=size, label=f'{reason} ({mask.sum()})',
                    edgecolors=edgecolor, linewidth=linewidth, marker=marker
                )

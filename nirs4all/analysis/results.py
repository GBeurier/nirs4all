"""
Transfer Selection Result Classes.

This module provides dataclasses for storing and visualizing transfer
preprocessing selection results.

Supports both object-based and string-based preprocessing definitions.

Classes:
    TransferResult: Result from evaluating a single preprocessing.
    TransferSelectionResults: Full results from selection process with
        ranking, visualization, and export methods.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.analysis.transfer_utils import (
    format_pipeline_name,
    get_base_preprocessings,
    get_transform_name,
    normalize_preprocessing,
)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class TransferResult:
    """
    Result from evaluating a single preprocessing for transfer.

    Attributes:
        name: Pipeline display name (e.g., 'StandardNormalVariate>FirstDerivative').
        pipeline_type: Type of pipeline ('single', 'stacked', or 'augmented').
        components: List of component names (e.g., ['StandardNormalVariate', 'FirstDerivative']).
        transfer_score: Combined transfer metric score (higher is better).
        metrics: Dictionary of individual metric values.
        improvement_pct: Percentage improvement over raw baseline.
        signal_score: Optional supervised validation score (Stage 4).
        transforms: Optional list of actual transformer objects (for object-based results).
    """

    name: str
    pipeline_type: str
    components: List[str]
    transfer_score: float
    metrics: Dict[str, float]
    improvement_pct: float
    signal_score: Optional[float] = None
    transforms: Optional[List[Any]] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        valid_types = ("single", "stacked", "augmented")
        if self.pipeline_type not in valid_types:
            raise ValueError(
                f"pipeline_type must be one of {valid_types}, "
                f"got '{self.pipeline_type}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "pipeline_type": self.pipeline_type,
            "components": self.components,
            "transfer_score": self.transfer_score,
            "metrics": self.metrics,
            "improvement_pct": self.improvement_pct,
            "signal_score": self.signal_score,
        }

    def get_transforms(
        self,
        preprocessings: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Get the transformer objects for this result.

        If transforms are already stored, returns them directly.
        Otherwise, resolves component names from the preprocessings dict.

        Args:
            preprocessings: Optional name->object mapping for resolution.

        Returns:
            List of transformer instances.
        """
        if self.transforms is not None:
            return [deepcopy(t) for t in self.transforms]

        # Resolve from component names
        if preprocessings is None:
            preprocessings = get_base_preprocessings()

        return [
            deepcopy(normalize_preprocessing(name, preprocessings))
            for name in self.components
        ]


@dataclass
class TransferSelectionResults:
    """
    Full results from transfer preprocessing selection.

    Provides access to ranked recommendations, timing information,
    and various output formats for integration with nirs4all pipelines.

    Attributes:
        ranking: List of TransferResult sorted by transfer_score (best first).
        raw_metrics: Baseline metrics computed on raw (unprocessed) data.
        timing: Dictionary of execution time per stage.
    """

    ranking: List[TransferResult]
    raw_metrics: Dict[str, float]
    timing: Dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> TransferResult:
        """
        Get the best recommendation.

        Returns:
            TransferResult with highest transfer score.

        Raises:
            ValueError: If no results are available.
        """
        if not self.ranking:
            raise ValueError("No results available")
        return self.ranking[0]

    def top_k(self, k: int = 5) -> List[TransferResult]:
        """
        Get top-K recommendations.

        Args:
            k: Number of top results to return.

        Returns:
            List of top-K TransferResult objects.
        """
        return self.ranking[:k]

    def to_preprocessing_list(
        self,
        top_k: int = 10,
        preprocessings: Optional[Dict[str, Any]] = None,
    ) -> List[List[Any]]:
        """
        Convert top-K results to a list of preprocessing transform pipelines.

        Each result is converted to a list of transformer instances that can
        be directly used in nirs4all pipeline's feature_augmentation.

        Args:
            top_k: Number of top results to convert.
            preprocessings: Optional dict mapping names to transformers.
                Uses get_base_preprocessings() if not provided.
                Not needed if results already store transform objects.

        Returns:
            List of preprocessing pipelines, where each pipeline is a list
            of transformer instances. For stacked pipelines like SNV>D1,
            returns [[SNV(), D1()], ...].

        Example:
            >>> results = selector.fit(X_train, X_test)
            >>> pp_list = results.to_preprocessing_list(top_k=5)
            >>> # pp_list = [[SNV()], [MSC()], [SNV(), D1()], ...]
            >>>
            >>> # Use in pipeline:
            >>> pipeline = [
            ...     {"feature_augmentation": {"_or_": pp_list, "pick": 1}},
            ...     {"model": PLSRegression()},
            ... ]
        """
        if preprocessings is None:
            preprocessings = get_base_preprocessings()

        result_list: List[List[Any]] = []
        top_results = self.ranking[:top_k]

        for r in top_results:
            try:
                # Use the result's get_transforms method
                pipeline_transforms = r.get_transforms(preprocessings)
                if pipeline_transforms:
                    result_list.append(pipeline_transforms)
            except (KeyError, ValueError):
                # Skip if transforms can't be resolved
                continue

        return result_list

    def to_pipeline_spec(
        self, top_k: int = 1, use_augmentation: bool = False
    ) -> Union[str, List[str], Dict[str, List[str]]]:
        """
        Convert results to nirs4all pipeline specification.

        Args:
            top_k: Number of top recommendations to include.
            use_augmentation: If True and top_k > 1, return augmentation spec.

        Returns:
            Pipeline specification usable in nirs4all:
            - Single string for top_k=1: "snv>d1"
            - List for multiple without augmentation: ["snv", "d1"]
            - Dict for augmentation: {"feature_augmentation": ["snv", "d1>msc"]}

        Example:
            >>> results.to_pipeline_spec()
            'snv'
            >>> results.to_pipeline_spec(top_k=3, use_augmentation=True)
            {'feature_augmentation': ['snv', 'd1', 'msc']}
        """
        if not self.ranking:
            return "identity"

        top_results = self.ranking[:top_k]

        if top_k == 1:
            return top_results[0].name

        names = [r.name for r in top_results]

        if use_augmentation:
            return {"feature_augmentation": names}
        else:
            return names

    def summary(self, top_k: int = 5) -> str:
        """
        Generate human-readable summary.

        Args:
            top_k: Number of top results to include in summary.

        Returns:
            Formatted summary string.
        """
        lines = ["=" * 60]
        lines.append("TRANSFER PREPROCESSING SELECTION RESULTS")
        lines.append("=" * 60)

        # Timing summary
        total_time = sum(self.timing.values())
        lines.append(f"\nTotal time: {total_time:.2f}s")
        for stage, t in self.timing.items():
            lines.append(f"  - {stage}: {t:.2f}s")

        # Raw baseline
        lines.append("\nBaseline (raw) metrics:")
        for metric, value in self.raw_metrics.items():
            if not np.isnan(value):
                lines.append(f"  - {metric}: {value:.4f}")

        # Top results
        n_display = min(top_k, len(self.ranking))
        lines.append(f"\nTop {n_display} recommendations:")
        lines.append("-" * 60)

        for i, result in enumerate(self.ranking[:top_k], 1):
            name_display = format_pipeline_name(result.name)
            lines.append(f"\n{i}. {name_display}")
            lines.append(f"   Type: {result.pipeline_type}")
            lines.append(f"   Transfer Score: {result.transfer_score:.4f}")
            lines.append(f"   Improvement: {result.improvement_pct:.1f}%")
            if result.signal_score is not None:
                lines.append(f"   Signal Score: {result.signal_score:.4f}")

        lines.append("\n" + "=" * 60)
        lines.append(f"Best recommendation: {self.best.name}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.

        Returns:
            DataFrame with columns for name, type, scores, and metrics.

        Raises:
            ImportError: If pandas is not available.
        """
        import pandas as pd

        rows = []
        for r in self.ranking:
            row = {
                "name": r.name,
                "pipeline_type": r.pipeline_type,
                "transfer_score": r.transfer_score,
                "improvement_pct": r.improvement_pct,
                "signal_score": r.signal_score,
            }
            row.update(r.metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_ranking(
        self,
        top_k: int = 15,
        show_signal_score: bool = True,
        figsize: Tuple[int, int] = (14, 8),
    ):
        """
        Plot ranked bar chart of preprocessing recommendations.

        Args:
            top_k: Number of top results to display.
            show_signal_score: Include signal score if available.
            figsize: Figure size as (width, height).

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        top_results = self.ranking[:top_k]

        # Prepare data
        names = [format_pipeline_name(r.name) for r in top_results]
        scores = [r.transfer_score for r in top_results]
        improvements = [r.improvement_pct for r in top_results]
        types = [r.pipeline_type for r in top_results]

        # Color by type
        type_colors = {
            "single": "#3498db",
            "stacked": "#2ecc71",
            "augmented": "#9b59b6",
        }
        colors = [type_colors.get(t, "#95a5a6") for t in types]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Transfer scores
        ax = axes[0]
        y_pos = np.arange(len(names))
        bars = ax.barh(
            y_pos, scores, color=colors, alpha=0.8,
            edgecolor="black", linewidth=0.5
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Transfer Score", fontsize=11, fontweight="bold")
        ax.set_title(
            "Transfer Score Ranking\n(Higher = Better)",
            fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor("#f8f9fa")
        ax.invert_yaxis()  # Best at top

        # Add value labels
        for bar, val in zip(bars, scores):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=8
            )

        # Plot 2: Improvement percentages
        ax = axes[1]
        bar_colors = ["green" if x > 0 else "red" for x in improvements]
        bars = ax.barh(
            y_pos, improvements, color=bar_colors, alpha=0.7,
            edgecolor="black", linewidth=0.5
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Improvement (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Distance Reduction\n(+ = Closer Datasets)",
            fontsize=12, fontweight="bold"
        )
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor("#f8f9fa")
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, improvements):
            label_x = val + (2 if val > 0 else -2)
            ha = "left" if val > 0 else "right"
            ax.text(
                label_x, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha=ha, va="center", fontsize=8
            )

        # Add legend for types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=type_colors["single"], label="Single"),
            Patch(facecolor=type_colors["stacked"], label="Stacked"),
            Patch(facecolor=type_colors["augmented"], label="Augmented"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right",
            bbox_to_anchor=(0.99, 0.99), fontsize=10
        )

        plt.suptitle(
            "Transfer Preprocessing Selection Results",
            fontsize=14, fontweight="bold", y=0.99
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def plot_metrics_comparison(
        self,
        top_k: int = 10,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 10),
    ):
        """
        Plot comparison of all metrics for top-K preprocessings.

        Args:
            top_k: Number of top results to display.
            metrics: Specific metrics to plot. Default: all available.
            figsize: Figure size as (width, height).

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = [
                "centroid_distance", "cka_similarity", "spread_distance",
                "grassmann_distance", "rv_coefficient", "procrustes_disparity"
            ]

        top_results = self.ranking[:top_k]

        # Filter available metrics
        available_metrics = []
        for m in metrics:
            if any(m in r.metrics for r in top_results):
                available_metrics.append(m)

        if not available_metrics:
            raise ValueError("No metrics available in results")

        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        names = [format_pipeline_name(r.name) for r in top_results]
        y_pos = np.arange(len(names))

        # Distance metrics: lower is better
        distance_metrics = {
            "centroid_distance", "grassmann_distance",
            "procrustes_disparity", "spread_distance"
        }

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            values = [r.metrics.get(metric, np.nan) for r in top_results]

            # Color based on metric type
            is_distance = metric in distance_metrics
            if is_distance:
                cmap = plt.cm.RdYlGn_r  # Red=high (bad), Green=low (good)
            else:
                cmap = plt.cm.RdYlGn  # Red=low (bad), Green=high (good)

            # Normalize values for coloring
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                vmin, vmax = min(valid_values), max(valid_values)
                norm_values = [
                    (v - vmin) / (vmax - vmin + 1e-10) for v in values
                ]
                colors = [
                    cmap(nv) if not np.isnan(v) else "#cccccc"
                    for v, nv in zip(values, norm_values)
                ]
            else:
                colors = ["#cccccc"] * len(values)

            ax.barh(
                y_pos, values, color=colors, alpha=0.8,
                edgecolor="black", linewidth=0.5
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)

            # Format metric name
            metric_display = metric.replace("_", " ").title()
            direction = "Lower=Better" if is_distance else "Higher=Better"
            ax.set_title(
                f"{metric_display}\n({direction})",
                fontsize=10, fontweight="bold"
            )
            ax.grid(axis="x", alpha=0.3)
            ax.set_facecolor("#f8f9fa")
            ax.invert_yaxis()

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Metrics Comparison for Top Preprocessings",
            fontsize=14, fontweight="bold", y=0.995
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        return fig

    def plot_improvement_heatmap(
        self,
        top_k: int = 15,
        figsize: Tuple[int, int] = (12, 10),
    ):
        """
        Plot heatmap of metric improvements vs raw data.

        Args:
            top_k: Number of top results to display.
            figsize: Figure size as (width, height).

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        top_results = self.ranking[:top_k]

        # Metrics to compare
        metrics = [
            "centroid_distance", "cka_similarity", "grassmann_distance",
            "rv_coefficient", "procrustes_disparity", "spread_distance"
        ]

        # Distance metrics: improvement = (raw - pp) / raw (positive = reduction)
        # Similarity metrics: improvement = (pp - raw) / raw (positive = increase)
        distance_metrics = {
            "centroid_distance", "grassmann_distance",
            "procrustes_disparity", "spread_distance"
        }

        names = [format_pipeline_name(r.name) for r in top_results]
        data = []

        for r in top_results:
            row = []
            for m in metrics:
                raw_val = self.raw_metrics.get(m, np.nan)
                pp_val = r.metrics.get(m, np.nan)

                if np.isnan(raw_val) or np.isnan(pp_val):
                    row.append(np.nan)
                elif m in distance_metrics:
                    # Distance: reduction is good (positive)
                    improvement = (raw_val - pp_val) / (abs(raw_val) + 1e-10)
                    row.append(improvement)
                else:
                    # Similarity: increase is good (positive)
                    improvement = (pp_val - raw_val) / (abs(raw_val) + 1e-10)
                    row.append(improvement)
            data.append(row)

        data = np.array(data)

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)

        # Labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(
            [m.replace("_", "\n").title() for m in metrics],
            fontsize=9, rotation=0
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Improvement (+ = Better)", fontsize=10)

        # Add text annotations
        for i in range(len(names)):
            for j in range(len(metrics)):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.5 else "black"
                    ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=8
                    )

        ax.set_title(
            "Metric Improvements vs Raw Data\n"
            "(Green = Improvement, Red = Degradation)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()

        return fig

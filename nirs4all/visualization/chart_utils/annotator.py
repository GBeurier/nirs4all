"""
ChartAnnotator - Helper for adding annotations to charts.
"""
import numpy as np
from typing import Optional, List, Dict, Any, Union
from nirs4all.visualization.charts.config import ChartConfig


class ChartAnnotator:
    """Helper for adding annotations to charts.

    Centralizes text formatting, positioning, and color selection
    for chart annotations. Uses ChartConfig for styling.

    Attributes:
        config: ChartConfig instance for customization.
    """

    def __init__(self, config: Optional[Union[ChartConfig, Dict[str, Any]]] = None):
        """Initialize annotator with config.

        Args:
            config: Optional ChartConfig or dict for customization.
                   If a dict is provided, it will be used to create a ChartConfig.
        """
        if isinstance(config, dict):
            self.config = ChartConfig(**config)
        else:
            self.config = config or ChartConfig()

    def add_heatmap_annotations(
        self,
        ax,
        matrix: np.ndarray,
        normalized_matrix: np.ndarray,
        count_matrix: np.ndarray,
        x_labels: List,
        y_labels: List,
        show_counts: bool = True,
        precision: int = 3
    ) -> None:
        """Add text annotations to heatmap cells.

        Args:
            ax: Matplotlib axes object.
            matrix: Original score matrix.
            normalized_matrix: Normalized matrix for color selection.
            count_matrix: Matrix of sample counts.
            x_labels: List of x-axis labels.
            y_labels: List of y-axis labels.
            show_counts: Whether to show sample counts.
            precision: Number of decimal places for scores.
        """
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                value = matrix[i, j]
                if not np.isnan(value):
                    normalized_value = normalized_matrix[i, j]
                    text_color = self.get_text_color(normalized_value)

                    # Format score text
                    score_text = f'{value:.{precision}f}'

                    # Add count if requested
                    if show_counts and count_matrix[i, j] > 1:
                        score_text += f'\n(n={int(count_matrix[i, j])})'

                    ax.text(j, i, score_text, ha='center', va='center',
                           color=text_color, fontsize=self.config.annotation_fontsize)

    @staticmethod
    def get_text_color(background_value: float, threshold: float = 0.5) -> str:
        """Determine text color based on background for optimal contrast.

        Args:
            background_value: Normalized background value (0-1).
            threshold: Threshold for switching from white to black text.

        Returns:
            Color string (always 'black' for consistency).
        """
        # Always return black for consistent, professional appearance
        return 'black'

    def add_statistics_box(
        self,
        ax,
        values: List[float],
        position: str = 'upper right',
        precision: int = 4
    ) -> None:
        """Add statistics text box to plot.

        Args:
            ax: Matplotlib axes object.
            values: List of values to compute statistics from.
            position: Position string for text box placement.
            precision: Number of decimal places.
        """
        if not values:
            return

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        stats_text = (
            f'n={len(values)}\n'
            f'μ={mean_val:.{precision}f}\n'
            f'σ={std_val:.{precision}f}\n'
            f'min={min_val:.{precision}f}\n'
            f'max={max_val:.{precision}f}'
        )

        # Determine position coordinates
        pos_map = {
            'upper right': (0.98, 0.98),
            'upper left': (0.02, 0.98),
            'lower right': (0.98, 0.02),
            'lower left': (0.02, 0.02)
        }
        x, y = pos_map.get(position, (0.98, 0.98))
        ha = 'right' if x > 0.5 else 'left'
        va = 'top' if y > 0.5 else 'bottom'

        ax.text(x, y, stats_text, transform=ax.transAxes,
                verticalalignment=va, horizontalalignment=ha,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=self.config.annotation_fontsize)

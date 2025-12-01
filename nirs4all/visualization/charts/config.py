"""
ChartConfig - Configuration for chart appearance and behavior.

Provides customization options for colors, fonts, and figure sizes.
All parameters have sensible defaults for seamless usage.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
import matplotlib.pyplot as plt


@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior.

    Provides customization options for colors, fonts, and figure sizes.
    All parameters have sensible defaults for seamless usage.

    Attributes:
        colormap: Matplotlib colormap name for gradients (default: 'viridis').
        heatmap_colormap: Colormap for heatmaps (default: 'RdYlGn').
        partition_colors: Dict mapping partition names to colors.
        font_family: Font family for all text (default: matplotlib default).
        title_fontsize: Font size for titles (default: 14).
        label_fontsize: Font size for axis labels (default: 10).
        tick_fontsize: Font size for tick labels (default: 9).
        legend_fontsize: Font size for legend text (default: 9).
        annotation_fontsize: Font size for text annotations inside charts (default: 9).
        figsize_small: Small figure size (default: (10, 6)).
        figsize_medium: Medium figure size (default: (12, 8)).
        figsize_large: Large figure size (default: (16, 10)).
        dpi: Output resolution (default: 300).
        alpha: Default alpha for plot elements (default: 0.7).
    """

    # Color schemes
    colormap: str = 'viridis'
    heatmap_colormap: str = 'RdYlGn'
    partition_colors: Optional[Dict[str, str]] = None

    # Font settings
    font_family: Optional[str] = None
    title_fontsize: int = 14
    label_fontsize: int = 10
    tick_fontsize: int = 9
    legend_fontsize: int = 9
    annotation_fontsize: int = 9

    # Figure sizes
    figsize_small: tuple = (10, 4)
    figsize_medium: tuple = (12, 8)
    figsize_large: tuple = (16, 10)

    # Other settings
    dpi: int = 300
    alpha: float = 0.7

    def __post_init__(self):
        """Initialize default partition colors if not provided."""
        if self.partition_colors is None:
            self.partition_colors = {
                'train': '#1f77b4',  # Blue
                'val': '#ff7f0e',    # Orange
                'test': '#2ca02c'    # Green
            }

    def apply_font_settings(self) -> None:
        """Apply font settings to matplotlib rcParams."""
        if self.font_family:
            plt.rcParams['font.family'] = self.font_family
        plt.rcParams['axes.titlesize'] = self.title_fontsize
        plt.rcParams['axes.labelsize'] = self.label_fontsize
        plt.rcParams['xtick.labelsize'] = self.tick_fontsize
        plt.rcParams['ytick.labelsize'] = self.tick_fontsize
        plt.rcParams['legend.fontsize'] = self.legend_fontsize

    def get_figsize(self, size: str = 'medium') -> tuple:
        """Get figure size by name.

        Args:
            size: Size name ('small', 'medium', 'large').

        Returns:
            Tuple of (width, height).
        """
        sizes = {
            'small': self.figsize_small,
            'medium': self.figsize_medium,
            'large': self.figsize_large
        }
        return sizes.get(size, self.figsize_medium)

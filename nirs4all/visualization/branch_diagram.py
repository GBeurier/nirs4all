"""
Branch Diagram - DAG visualization for pipeline branching structure.

DEPRECATED: This module is deprecated in favor of pipeline_diagram.py which
provides a more comprehensive visualization of the entire pipeline structure.

The BranchDiagram and plot_branch_diagram are now aliases for PipelineDiagram
and plot_pipeline_diagram respectively.

Example:
    >>> from nirs4all.visualization.pipeline_diagram import PipelineDiagram
    >>> diagram = PipelineDiagram(pipeline_steps, predictions)
    >>> fig = diagram.render()
    >>> fig.savefig('pipeline_diagram.png')
"""

import warnings
from typing import Any, Dict, Optional, Tuple

from matplotlib.figure import Figure

# Import from new module
from nirs4all.visualization.pipeline_diagram import (
    PipelineDiagram,
    plot_pipeline_diagram,
)

__all__ = ['BranchDiagram', 'plot_branch_diagram', 'PipelineDiagram', 'plot_pipeline_diagram']


class BranchDiagram(PipelineDiagram):
    """DEPRECATED: Use PipelineDiagram instead.

    This class is kept for backward compatibility only.
    It wraps PipelineDiagram with the old API.
    """

    def __init__(
        self,
        predictions: Any = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize BranchDiagram (deprecated).

        Args:
            predictions: Predictions object with branch metadata.
            config: Optional dict for customization.
        """
        warnings.warn(
            "BranchDiagram is deprecated. Use PipelineDiagram instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(pipeline_steps=None, predictions=predictions, config=config)

    def render(
        self,
        show_metrics: Optional[bool] = None,
        metric: Optional[str] = None,
        partition: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """Render the branch diagram (deprecated API).

        Args:
            show_metrics: Override config's show_metrics setting.
            metric: Override metric to display.
            partition: Override partition for metrics.
            figsize: Override figure size.
            title: Optional title for the diagram.

        Returns:
            matplotlib Figure object.
        """
        # Map old show_metrics to new show_shapes
        return super().render(
            show_shapes=show_metrics if show_metrics is not None else self._show_shapes,
            figsize=figsize,
            title=title,
        )


def plot_branch_diagram(
    predictions: Any = None,
    show_metrics: bool = True,
    metric: str = 'rmse',
    partition: str = 'test',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Figure:
    """DEPRECATED: Use plot_pipeline_diagram instead.

    Args:
        predictions: Predictions object with branch metadata.
        show_metrics: Whether to show metrics in nodes.
        metric: Metric to display (default: 'rmse').
        partition: Partition for metrics (default: 'test').
        figsize: Figure size tuple.
        title: Optional title for the diagram.
        config: Additional configuration dict.

    Returns:
        matplotlib Figure object.
    """
    warnings.warn(
        "plot_branch_diagram is deprecated. Use plot_pipeline_diagram instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return plot_pipeline_diagram(
        pipeline_steps=None,
        predictions=predictions,
        show_shapes=show_metrics,
        figsize=figsize,
        title=title,
        config=config,
    )

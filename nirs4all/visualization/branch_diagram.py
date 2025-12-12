"""
Branch Diagram - DAG visualization for pipeline branching structure.

This module provides visualization tools for displaying the branching
structure of a pipeline as a directed acyclic graph (DAG).

The diagram shows:
- Shared preprocessing steps before branching
- Branch nodes with their preprocessing chains
- Post-branch model steps
- Prediction counts and performance metrics per branch

Example:
    >>> from nirs4all.visualization.branch_diagram import BranchDiagram
    >>> diagram = BranchDiagram(predictions)
    >>> fig = diagram.render()
    >>> fig.savefig('branch_diagram.png')
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np


class BranchDiagram:
    """Create DAG visualization for pipeline branching structure.

    Renders a visual diagram showing the branching topology of a pipeline,
    including shared steps, branch-specific steps, and post-branch models.

    Attributes:
        predictions: Predictions object containing prediction data.
        config: Optional dict for customization.
    """

    # Default colors for different node types
    NODE_COLORS = {
        'shared': '#3498db',      # Blue for shared preprocessing
        'branch': '#2ecc71',      # Green for branch nodes
        'model': '#e74c3c',       # Red for model nodes
        'split': '#9b59b6',       # Purple for split nodes
        'default': '#95a5a6',     # Gray for unknown
    }

    # Box styles
    BOX_STYLE = "round,rounding_size=10"
    BOX_PAD = 0.3

    def __init__(
        self,
        predictions,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize BranchDiagram.

        Args:
            predictions: Predictions object with branch metadata.
            config: Optional dict for customization:
                - figsize: Tuple for figure size
                - fontsize: Base font size
                - node_width: Width of nodes
                - node_height: Height of nodes
                - show_metrics: Whether to show metrics in nodes
                - metric: Metric to display (default: 'rmse')
                - partition: Partition for metrics (default: 'test')
        """
        self.predictions = predictions
        self.config = config or {}

        # Default configuration
        self._figsize = self.config.get('figsize', (12, 8))
        self._fontsize = self.config.get('fontsize', 10)
        self._node_width = self.config.get('node_width', 2.0)
        self._node_height = self.config.get('node_height', 0.8)
        self._show_metrics = self.config.get('show_metrics', True)
        self._metric = self.config.get('metric', 'rmse')
        self._partition = self.config.get('partition', 'test')

    def render(
        self,
        show_metrics: Optional[bool] = None,
        metric: Optional[str] = None,
        partition: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """Render the branch diagram.

        Args:
            show_metrics: Override config's show_metrics setting.
            metric: Override metric to display.
            partition: Override partition for metrics.
            figsize: Override figure size.
            title: Optional title for the diagram.

        Returns:
            matplotlib Figure object.
        """
        # Apply overrides - ensure proper types
        effective_show_metrics = show_metrics if show_metrics is not None else self._show_metrics
        effective_metric = metric if metric is not None else self._metric
        effective_partition = partition if partition is not None else self._partition
        effective_figsize = figsize if figsize is not None else self._figsize

        # Extract branch information from predictions
        branch_info = self._extract_branch_info()

        if not branch_info['branches']:
            # No branches - show simple message
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, 'No branching structure detected',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Calculate layout
        layout = self._compute_layout(branch_info)

        # Create figure
        fig, ax = plt.subplots(figsize=effective_figsize)
        ax.set_aspect('equal')

        # Draw the diagram
        self._draw_nodes(ax, layout, branch_info, effective_show_metrics,
                         effective_metric, effective_partition)
        self._draw_edges(ax, layout, branch_info)

        # Configure axes
        ax.axis('off')

        # Set title
        if title is None:
            n_branches = len(branch_info['branches'])
            n_preds = sum(b['count'] for b in branch_info['branches'].values())
            title = f"Pipeline Branching Structure\n{n_branches} branches • {n_preds} predictions"
        ax.set_title(title, fontsize=self._fontsize + 2, fontweight='bold')

        # Adjust limits
        x_min, x_max, y_min, y_max = self._get_bounds(layout)
        padding = 0.5
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        plt.tight_layout()
        return fig

    def _extract_branch_info(self) -> Dict[str, Any]:
        """Extract branching structure from predictions.

        Returns:
            Dictionary with:
                - branches: Dict mapping branch_name to info (id, count, metrics)
                - shared_steps: List of shared preprocessing step names
                - post_branch_steps: List of post-branch model names
                - n_folds: Number of folds
        """
        info = {
            'branches': {},
            'shared_steps': [],
            'post_branch_steps': [],
            'n_folds': 0,
        }

        # Get unique branch names and IDs
        try:
            branch_names = self.predictions.get_unique_values('branch_name')
            if not branch_names or branch_names == [None]:
                return info
        except (ValueError, KeyError):
            return info

        # Get fold count
        try:
            fold_ids = self.predictions.get_unique_values('fold_id')
            info['n_folds'] = len([f for f in fold_ids if f is not None])
        except (ValueError, KeyError):
            info['n_folds'] = 1

        # Get model names for post-branch steps
        try:
            model_names = self.predictions.get_unique_values('model_name')
            if model_names:
                info['post_branch_steps'] = [m for m in model_names if m]
        except (ValueError, KeyError):
            pass

        # Get shared preprocessing (from first prediction if available)
        try:
            preprocessings = self.predictions.get_unique_values('preprocessings')
            if preprocessings:
                # Find common prefix across all preprocessing chains
                shared = self._find_common_prefix(preprocessings)
                if shared:
                    info['shared_steps'] = shared.split(' > ')
        except (ValueError, KeyError):
            pass

        # Extract branch details
        for branch_name in branch_names:
            if branch_name is None:
                continue

            branch_preds = self.predictions.filter_predictions(branch_name=branch_name)
            if not branch_preds:
                continue

            # Get branch ID from first prediction
            first_pred = branch_preds[0] if branch_preds else {}
            branch_id = first_pred.get('branch_id', 0)

            # Collect metrics across folds
            metrics = self._collect_branch_metrics(branch_preds)

            info['branches'][branch_name] = {
                'id': branch_id,
                'count': len(branch_preds),
                'metrics': metrics,
            }

        return info

    def _find_common_prefix(self, preprocessings: List[str]) -> str:
        """Find common prefix across all preprocessing chains.

        Args:
            preprocessings: List of preprocessing chain strings (e.g., "MinMax > SNV").

        Returns:
            Common prefix string.
        """
        if not preprocessings:
            return ""

        # Filter None values
        valid = [p for p in preprocessings if p]
        if not valid:
            return ""

        # Split into parts
        parts_list = [p.split(' > ') for p in valid]
        if not parts_list:
            return ""

        # Find common prefix
        common = []
        min_len = min(len(parts) for parts in parts_list)

        for i in range(min_len):
            first = parts_list[0][i]
            if all(parts[i] == first for parts in parts_list):
                common.append(first)
            else:
                break

        return ' > '.join(common)

    def _collect_branch_metrics(
        self,
        branch_preds: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Collect aggregate metrics for a branch.

        Args:
            branch_preds: List of predictions for a branch.

        Returns:
            Dictionary with mean metrics across folds.
        """
        from nirs4all.core import metrics as evaluator

        metrics_values = defaultdict(list)

        for pred in branch_preds:
            partitions = pred.get('partitions', {})

            for partition_name, partition_data in partitions.items():
                if not partition_data:
                    continue

                # Try to get common metrics
                for metric in ['rmse', 'r2', 'mae', 'balanced_accuracy', 'f1']:
                    value = partition_data.get(metric)
                    if value is None:
                        y_true = partition_data.get('y_true')
                        y_pred = partition_data.get('y_pred')
                        if y_true is not None and y_pred is not None:
                            try:
                                value = evaluator.eval(y_true, y_pred, metric)
                            except Exception:
                                continue

                    # Validate value is numeric before appending
                    if value is not None and isinstance(value, (int, float, np.floating, np.integer)):
                        key = f'{partition_name}_{metric}'
                        metrics_values[key].append(float(value))

        # Compute means
        metrics = {}
        for key, values in metrics_values.items():
            if values:
                metrics[key] = np.mean(values)

        return metrics

    def _compute_layout(
        self,
        branch_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute node positions for the diagram.

        Layout structure:
        - Row 0: Split node (if folds > 1)
        - Row 1: Shared preprocessing steps
        - Row 2: Branch node
        - Row 3: Individual branches (spread horizontally)
        - Row 4: Post-branch models

        Args:
            branch_info: Branch information from _extract_branch_info.

        Returns:
            Dictionary mapping node IDs to position info.
        """
        layout = {}
        y_spacing = 1.5
        x_spacing = 2.5

        current_y = 0
        center_x = 0

        # Add split node if multiple folds
        if branch_info['n_folds'] > 1:
            layout['split'] = {
                'x': center_x,
                'y': current_y,
                'type': 'split',
                'label': f"Splitter\n({branch_info['n_folds']} folds)",
            }
            current_y -= y_spacing

        # Add shared preprocessing steps
        for i, step in enumerate(branch_info['shared_steps']):
            node_id = f'shared_{i}'
            layout[node_id] = {
                'x': center_x,
                'y': current_y,
                'type': 'shared',
                'label': step,
            }
            current_y -= y_spacing

        # Add branch junction node
        if branch_info['branches']:
            layout['branch_junction'] = {
                'x': center_x,
                'y': current_y,
                'type': 'branch',
                'label': 'Branch',
            }
            current_y -= y_spacing

        # Add individual branches spread horizontally
        n_branches = len(branch_info['branches'])
        if n_branches > 0:
            x_start = -(n_branches - 1) * x_spacing / 2

            for i, (branch_name, branch_data) in enumerate(
                sorted(branch_info['branches'].items(), key=lambda x: x[1]['id'])
            ):
                x = x_start + i * x_spacing
                node_id = f'branch_{branch_data["id"]}'

                # Format label with metrics if available
                label = branch_name
                if self._show_metrics:
                    metric_key = f'{self._partition}_{self._metric}'
                    if metric_key in branch_data['metrics']:
                        value = branch_data['metrics'][metric_key]
                        label += f"\n{self._metric}: {value:.3f}"

                layout[node_id] = {
                    'x': x,
                    'y': current_y,
                    'type': 'branch',
                    'label': label,
                    'count': branch_data['count'],
                }

            current_y -= y_spacing

        # Add post-branch models
        if branch_info['post_branch_steps']:
            model_label = ' + '.join(branch_info['post_branch_steps'][:3])
            if len(branch_info['post_branch_steps']) > 3:
                model_label += '...'

            layout['model'] = {
                'x': center_x,
                'y': current_y,
                'type': 'model',
                'label': model_label,
            }

        return layout

    def _draw_nodes(
        self,
        ax: Axes,
        layout: Dict[str, Dict[str, Any]],
        branch_info: Dict[str, Any],
        show_metrics: bool,
        metric: str,
        partition: str
    ) -> None:
        """Draw nodes on the diagram.

        Args:
            ax: Matplotlib axes.
            layout: Node layout from _compute_layout.
            branch_info: Branch information.
            show_metrics: Whether to show metrics.
            metric: Metric to display.
            partition: Partition for metrics.
        """
        for node_id, node in layout.items():
            x, y = node['x'], node['y']
            node_type = node['type']
            label = node['label']

            # Get color based on type
            color = self.NODE_COLORS.get(node_type, self.NODE_COLORS['default'])

            # Draw box
            box_width = self._node_width
            box_height = self._node_height

            # Adjust height for multi-line labels
            n_lines = label.count('\n') + 1
            box_height *= (1 + 0.3 * (n_lines - 1))

            rect = FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width, box_height,
                boxstyle=self.BOX_STYLE,
                facecolor=color,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Add text
            ax.text(
                x, y, label,
                ha='center', va='center',
                fontsize=self._fontsize,
                color='white',
                fontweight='bold',
                wrap=True,
            )

            # Add count badge for branch nodes
            if 'count' in node and node_type == 'branch':
                count = node['count']
                badge_x = x + box_width / 2 - 0.2
                badge_y = y + box_height / 2 - 0.1
                ax.text(
                    badge_x, badge_y, f'n={count}',
                    ha='right', va='top',
                    fontsize=self._fontsize - 2,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#34495e', alpha=0.9),
                )

    def _draw_edges(
        self,
        ax: Axes,
        layout: Dict[str, Dict[str, Any]],
        branch_info: Dict[str, Any]
    ) -> None:
        """Draw edges connecting nodes.

        Args:
            ax: Matplotlib axes.
            layout: Node layout from _compute_layout.
            branch_info: Branch information.
        """
        # Define edge connections based on layout structure
        # Split -> shared -> branch_junction -> branches -> model

        node_ids = list(layout.keys())

        # Sequential connections for linear chain
        linear_nodes = ['split'] + [f'shared_{i}' for i in range(len(branch_info['shared_steps']))]
        linear_nodes = [n for n in linear_nodes if n in layout]

        if 'branch_junction' in layout:
            linear_nodes.append('branch_junction')

        # Draw linear connections
        for i in range(len(linear_nodes) - 1):
            self._draw_arrow(ax, layout, linear_nodes[i], linear_nodes[i + 1])

        # Connect branch junction to individual branches
        if 'branch_junction' in layout:
            for node_id in layout:
                if node_id.startswith('branch_') and node_id != 'branch_junction':
                    self._draw_arrow(ax, layout, 'branch_junction', node_id)

        # Connect branches to model
        if 'model' in layout:
            for node_id in layout:
                if node_id.startswith('branch_') and node_id != 'branch_junction':
                    self._draw_arrow(ax, layout, node_id, 'model')

    def _draw_arrow(
        self,
        ax: Axes,
        layout: Dict[str, Dict[str, Any]],
        from_id: str,
        to_id: str
    ) -> None:
        """Draw an arrow between two nodes.

        Args:
            ax: Matplotlib axes.
            layout: Node layout.
            from_id: Source node ID.
            to_id: Target node ID.
        """
        if from_id not in layout or to_id not in layout:
            return

        from_node = layout[from_id]
        to_node = layout[to_id]

        # Calculate connection points (bottom of source, top of target)
        from_x, from_y = from_node['x'], from_node['y'] - self._node_height / 2
        to_x, to_y = to_node['x'], to_node['y'] + self._node_height / 2

        arrow = FancyArrowPatch(
            (from_x, from_y), (to_x, to_y),
            arrowstyle='-|>',
            color='#2c3e50',
            linewidth=1.5,
            mutation_scale=15,
            connectionstyle='arc3,rad=0',
        )
        ax.add_patch(arrow)

    def _get_bounds(
        self,
        layout: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, float, float, float]:
        """Get bounding box for the diagram.

        Args:
            layout: Node layout.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max).
        """
        if not layout:
            return -1, 1, -1, 1

        x_coords = [n['x'] for n in layout.values()]
        y_coords = [n['y'] for n in layout.values()]

        x_min = min(x_coords) - self._node_width
        x_max = max(x_coords) + self._node_width
        y_min = min(y_coords) - self._node_height
        y_max = max(y_coords) + self._node_height

        return x_min, x_max, y_min, y_max


def plot_branch_diagram(
    predictions,
    show_metrics: bool = True,
    metric: str = 'rmse',
    partition: str = 'test',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Figure:
    """Convenience function to create a branch diagram.

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

    Example:
        >>> from nirs4all.visualization.branch_diagram import plot_branch_diagram
        >>> fig = plot_branch_diagram(predictions, metric='r2')
        >>> fig.savefig('branch_diagram.png')
    """
    cfg = config or {}
    diagram = BranchDiagram(predictions, config=cfg)
    return diagram.render(
        show_metrics=show_metrics,
        metric=metric,
        partition=partition,
        figsize=figsize,
        title=title,
    )

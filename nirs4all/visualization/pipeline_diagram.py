"""
Pipeline Diagram - DAG visualization for pipeline execution structure.

This module provides visualization tools for displaying the complete
pipeline structure as a directed acyclic graph (DAG).

The diagram shows:
- All pipeline steps with operator names
- Dataset shape at each step (samples × processings × features)
- Branching and merging points
- Model training steps
- Cross-validation splitters

Shape notation: S×P×F
- S = samples
- P = processings (preprocessing views)
- F = features (wavelengths/columns)

Example:
    >>> from nirs4all.visualization.pipeline_diagram import PipelineDiagram
    >>> diagram = PipelineDiagram(pipeline_steps, predictions)
    >>> fig = diagram.render()
    >>> fig.savefig('pipeline_diagram.png')
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class PipelineNode:
    """Represents a node in the pipeline DAG.

    Attributes:
        id: Unique node identifier
        step_index: Pipeline step index (1-based)
        label: Display label for the node
        node_type: Type of node (preprocessing, model, splitter, branch, merge, etc.)
        shape_before: Dataset shape before this step (samples, processings, features)
        shape_after: Dataset shape after this step
        input_layout_shape: 2D layout shape before step (samples, features)
        output_layout_shape: 2D layout shape after step (samples, features)
        features_shape: List of 3D per-source shapes (samples, processings, features)
        branch_id: Branch ID if inside a branch (None if not)
        branch_name: Branch name if inside a branch
        substep_index: Index within a branch's substeps
        parent_ids: List of parent node IDs
        children_ids: List of child node IDs
        duration_ms: Execution duration in milliseconds (from trace)
        metadata: Additional node metadata
    """
    id: str
    step_index: int
    label: str
    node_type: str = "preprocessing"
    shape_before: Optional[Tuple[int, int, int]] = None
    shape_after: Optional[Tuple[int, int, int]] = None
    input_layout_shape: Optional[Tuple[int, int]] = None
    output_layout_shape: Optional[Tuple[int, int]] = None
    features_shape: Optional[List[Tuple[int, int, int]]] = None
    branch_id: Optional[int] = None
    branch_name: str = ""
    substep_index: Optional[int] = None
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineDiagram:
    """Create DAG visualization for pipeline execution structure.

    Renders a visual diagram showing the complete pipeline topology,
    including all steps, shapes, branches, and models.

    Attributes:
        pipeline_steps: List of pipeline step definitions
        predictions: Optional Predictions object with execution data
        execution_trace: Optional ExecutionTrace with actual runtime shapes
        config: Optional dict for customization
    """

    # Node colors by type - lighter colors for black text readability
    NODE_COLORS = {
        'preprocessing': '#aed6f1',      # Light blue
        'feature_augmentation': '#a3e4d7',  # Light teal
        'sample_augmentation': '#a9dfbf',   # Light green
        'concat_transform': '#d7bde2',   # Light purple
        'y_processing': '#f9e79f',       # Light yellow
        'splitter': '#d2b4de',           # Light purple
        'branch': '#abebc6',             # Light green
        'merge': '#82e0aa',              # Medium green
        'source_branch': '#a3e4d7',      # Light teal
        'merge_sources': '#7dcea0',      # Medium teal
        'model': '#f5b7b1',              # Light red/pink
        'input': '#d5dbdb',              # Light gray
        'output': '#cacfd2',             # Gray
        'default': '#d5d8dc',            # Light gray
    }

    # Node shapes by type (for future SVG export)
    NODE_SHAPES = {
        'preprocessing': 'round',
        'model': 'rectangle',
        'splitter': 'diamond',
        'branch': 'hexagon',
        'merge': 'hexagon',
        'input': 'ellipse',
        'output': 'ellipse',
    }

    def __init__(
        self,
        pipeline_steps: Optional[List[Any]] = None,
        predictions: Any = None,
        execution_trace: Any = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize PipelineDiagram.

        Args:
            pipeline_steps: List of pipeline step definitions
            predictions: Optional Predictions object with execution data
            execution_trace: Optional ExecutionTrace with runtime shapes
            config: Optional dict for customization:
                - figsize: Tuple for figure size
                - fontsize: Base font size
                - node_width: Width of nodes
                - node_height: Height of nodes
                - show_shapes: Whether to show shape info
                - compact: Use compact node labels
        """
        self.pipeline_steps = pipeline_steps or []
        self.predictions = predictions
        self.execution_trace = execution_trace
        self.config = config or {}

        # Default configuration - smaller fonts
        self._figsize = self.config.get('figsize', (14, 10))
        self._fontsize = self.config.get('fontsize', 7)
        self._node_width = self.config.get('node_width', 2.2)
        self._node_height = self.config.get('node_height', 0.7)
        self._show_shapes = self.config.get('show_shapes', True)
        self._compact = self.config.get('compact', False)

        # Build the DAG
        self.nodes: Dict[str, PipelineNode] = {}
        self.edges: List[Tuple[str, str]] = []

    def render(
        self,
        show_shapes: Optional[bool] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        initial_shape: Optional[Tuple[int, int, int]] = None
    ) -> Figure:
        """Render the pipeline diagram.

        Args:
            show_shapes: Override config's show_shapes setting
            figsize: Override figure size
            title: Optional title for the diagram
            initial_shape: Initial dataset shape (samples, processings, features)

        Returns:
            matplotlib Figure object
        """
        # Apply overrides
        effective_show_shapes = show_shapes if show_shapes is not None else self._show_shapes
        effective_figsize = figsize if figsize is not None else self._figsize

        # Only build DAG if not already built (e.g., from_trace already built it)
        if not self.nodes:
            self._build_dag(initial_shape=initial_shape)

        if not self.nodes:
            # No steps - show simple message
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, 'No pipeline steps to visualize',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Calculate layout
        layout = self._compute_layout()

        # Create figure
        fig, ax = plt.subplots(figsize=effective_figsize)

        # Draw the diagram
        self._draw_edges(ax, layout)
        self._draw_nodes(ax, layout, effective_show_shapes)

        # Configure axes
        ax.set_aspect('equal')
        ax.axis('off')

        # Set title
        if title is None:
            n_steps = len([n for n in self.nodes.values() if n.node_type != 'input'])
            title = f"Pipeline Structure ({n_steps} steps)"
        ax.set_title(title, fontsize=self._fontsize + 2, fontweight='medium', pad=20, color='#2c3e50')

        # Adjust limits
        x_min, x_max, y_min, y_max = self._get_bounds(layout)
        padding = 1.0
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add legend
        self._add_legend(ax)

        plt.tight_layout()
        return fig

    @classmethod
    def from_trace(
        cls,
        execution_trace: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> 'PipelineDiagram':
        """Create a PipelineDiagram from an ExecutionTrace.

        This builds the diagram using actual runtime data including
        measured shapes at each step.

        Args:
            execution_trace: ExecutionTrace object from pipeline execution
            config: Optional configuration dict

        Returns:
            PipelineDiagram instance ready for rendering

        Example:
            >>> from nirs4all.visualization import PipelineDiagram
            >>> diagram = PipelineDiagram.from_trace(trace)
            >>> fig = diagram.render(title="Execution Trace")
        """
        diagram = cls(execution_trace=execution_trace, config=config)
        diagram._build_dag_from_trace()
        return diagram

    def _build_dag_from_trace(self) -> None:
        """Build the DAG from an ExecutionTrace object with actual shapes."""
        if not self.execution_trace:
            return

        self.nodes.clear()
        self.edges.clear()

        steps = getattr(self.execution_trace, 'steps', [])
        if not steps:
            return

        # Create input node from first step's input shape
        first_step = steps[0] if steps else None
        input_layout = getattr(first_step, 'input_shape', None) if first_step else None
        input_features = getattr(first_step, 'input_features_shape', None) if first_step else None

        input_node = PipelineNode(
            id="input",
            step_index=0,
            label="Dataset",
            node_type="input",
            output_layout_shape=input_layout,
            features_shape=input_features,
        )
        self.nodes["input"] = input_node

        # Track edges by step
        current_node_ids = ["input"]
        branch_stacks: Dict[tuple, List[str]] = {}  # branch_path -> node_ids
        in_branch_mode = False  # Track if we're inside a branch
        last_pre_branch_node = "input"  # Track the node before entering branches

        for step in steps:
            step_index = getattr(step, 'step_index', 0)
            operator_class = getattr(step, 'operator_class', '') or ''
            operator_type = getattr(step, 'operator_type', '') or ''
            branch_path = tuple(getattr(step, 'branch_path', []) or [])
            branch_name = getattr(step, 'branch_name', '') or ''
            duration_ms = getattr(step, 'duration_ms', 0.0)
            substep_idx = getattr(step, 'substep_index', None)

            # Get shapes from trace
            output_layout = getattr(step, 'output_shape', None)
            output_features = getattr(step, 'output_features_shape', None)

            # Determine node type from operator info
            node_type = self._classify_operator_type(operator_type, operator_class)

            # Format the label from operator info
            label = self._format_trace_label(operator_class, operator_type, step)

            # Create node ID
            if branch_path:
                node_id = f"step_{step_index}_b{'_'.join(map(str, branch_path))}"
            else:
                node_id = f"step_{step_index}"

            # Handle substep index for unique node IDs
            if substep_idx is not None:
                node_id += f"_s{substep_idx}"

            # Determine parent nodes based on branch path and operator type
            op_type_lower = operator_type.lower() if operator_type else ''

            if branch_path and substep_idx is not None:
                # Branch substep - chain within the same branch
                if substep_idx == 0:
                    # First substep - connect to the pre-branch node (current main path)
                    parent_nodes = current_node_ids.copy() if current_node_ids else [last_pre_branch_node]
                else:
                    # Chain to previous substep in same branch
                    prev_substep_id = f"step_{step_index}_b{'_'.join(map(str, branch_path))}_s{substep_idx - 1}"
                    if prev_substep_id in self.nodes:
                        parent_nodes = [prev_substep_id]
                    else:
                        parent_nodes = current_node_ids.copy() if current_node_ids else [last_pre_branch_node]
                in_branch_mode = True
            elif branch_path:
                # Branch step without substep index (e.g., post-branch steps like splitter)
                # Connect to the last node in this branch path
                # First try exact branch path, then try with first element only
                bp_tuple = branch_path
                parent_found = False
                while len(bp_tuple) > 0:
                    if bp_tuple in branch_stacks:
                        parent_nodes = branch_stacks[bp_tuple]
                        parent_found = True
                        break
                    # Try partial branch path (for deep branches)
                    bp_tuple = bp_tuple[:-1]

                if not parent_found:
                    # Check if there are any matching branch substeps we should connect to
                    branch_id = branch_path[0] if branch_path else 0
                    # Find all substep nodes in this branch
                    matching_leaves = []
                    for bpath, node_ids in branch_stacks.items():
                        if bpath and bpath[0] == branch_id:
                            matching_leaves.extend(node_ids)
                    parent_nodes = matching_leaves if matching_leaves else current_node_ids
                in_branch_mode = True
            elif op_type_lower == 'merge' and in_branch_mode and branch_stacks:
                # Merge step exiting branch mode - connect to ALL branch leaf nodes
                all_branch_leaves = []
                for bpath, node_ids in branch_stacks.items():
                    all_branch_leaves.extend(node_ids)
                parent_nodes = all_branch_leaves if all_branch_leaves else current_node_ids
                in_branch_mode = False
                branch_stacks.clear()  # Clear branch stacks after merge
            elif op_type_lower == 'merge' and in_branch_mode:
                # Merge without branch_stacks - connect to previous node
                parent_nodes = current_node_ids
                in_branch_mode = False
            elif op_type_lower == 'branch':
                # Branch entry - remember current node as branch parent
                parent_nodes = current_node_ids
                last_pre_branch_node = current_node_ids[0] if current_node_ids else "input"
                in_branch_mode = True
            else:
                parent_nodes = current_node_ids
                if not in_branch_mode:
                    last_pre_branch_node = current_node_ids[0] if current_node_ids else "input"

            # Create the node
            node = PipelineNode(
                id=node_id,
                step_index=step_index,
                label=label,
                node_type=node_type,
                output_layout_shape=output_layout,
                features_shape=output_features,
                branch_id=branch_path[-1] if branch_path else None,
                branch_name=branch_name,
                substep_index=substep_idx,
                parent_ids=list(parent_nodes) if parent_nodes else [],
                duration_ms=duration_ms,
            )
            self.nodes[node_id] = node

            # Add edges from parents
            for parent_id in (parent_nodes or []):
                if parent_id in self.nodes:
                    self.edges.append((parent_id, node_id))

            # Update tracking
            if branch_path:
                branch_stacks[branch_path] = [node_id]
            else:
                current_node_ids = [node_id]

    def _format_trace_label(
        self,
        operator_class: str,
        operator_type: str,
        step: Any
    ) -> str:
        """Format a label for display from trace step info.

        Creates a readable label by preferring operator_class when available,
        with fallbacks to operator_type or step index.

        Args:
            operator_class: Class name from trace (may be 'dict', 'list', etc.)
            operator_type: Operator type from trace
            step: The ExecutionStep object

        Returns:
            Human-readable label string
        """
        step_index = getattr(step, 'step_index', 0)
        op_type_lower = operator_type.lower() if operator_type else ''

        # Generic Python types to avoid using directly
        generic_types = {'dict', 'list', 'tuple', 'str', 'int', 'config', 'NoneType', ''}

        # Special handling for merge and branch - include the mode/strategy
        if op_type_lower == 'merge':
            if operator_class and operator_class.lower() not in generic_types:
                # operator_class is something like 'predictions' or 'features'
                return f"Merge ({operator_class})"
            return "Merge"

        if op_type_lower == 'branch':
            if operator_class and operator_class.lower() not in generic_types:
                return f"Branch: {operator_class}"
            return "Branch"

        if op_type_lower == 'source_branch':
            return "Source Branch"

        if op_type_lower == 'merge_sources':
            if operator_class and operator_class.lower() not in generic_types:
                return f"Merge Sources ({operator_class})"
            return "Merge Sources"

        # If operator_class is meaningful (not a generic Python type), use it
        if operator_class and operator_class.lower() not in generic_types:
            # Shorten long operator class names if needed
            if len(operator_class) > 30:
                return operator_class[:27] + "..."
            return operator_class

        # Fallback to formatted operator_type
        type_labels = {
            'preprocessing': 'Preprocessing',
            'y_processing': 'Y Processing',
            'feature_augmentation': 'Feature Aug',
            'sample_augmentation': 'Sample Aug',
            'concat_transform': 'Concat',
            'model': 'Model',
            'meta_model': 'Meta Model',
            'splitter': 'Splitter',
            'branch': 'Branch',
            'merge': 'Merge',
            'source_branch': 'Source Branch',
            'merge_sources': 'Merge Sources',
            'transform': 'Transform',
            'operator': 'Operator',
            'config': 'Config',
        }

        if operator_type:
            label = type_labels.get(op_type_lower, operator_type.title())
            return label

        return f"Step {step_index}"

    def _classify_operator_type(self, op_type: str, op_class: str) -> str:
        """Classify operator into a node type for coloring.

        Args:
            op_type: Operator type from trace
            op_class: Operator class name

        Returns:
            Node type string for coloring
        """
        op_type_lower = op_type.lower()
        op_class_lower = op_class.lower()

        if 'model' in op_type_lower or 'meta_model' in op_type_lower:
            return 'model'
        elif 'splitter' in op_type_lower or 'fold' in op_class_lower or 'split' in op_class_lower:
            return 'splitter'
        elif 'branch' in op_type_lower:
            return 'branch'
        elif 'merge' in op_type_lower:
            return 'merge'
        elif 'y_processing' in op_type_lower:
            return 'y_processing'
        elif 'feature_augmentation' in op_type_lower:
            return 'feature_augmentation'
        elif 'sample_augmentation' in op_type_lower:
            return 'sample_augmentation'
        elif 'concat_transform' in op_type_lower:
            return 'concat_transform'
        elif 'source_branch' in op_type_lower:
            return 'source_branch'
        elif 'merge_sources' in op_type_lower:
            return 'merge_sources'
        else:
            return 'preprocessing'

    def _build_dag(self, initial_shape: Optional[Tuple[int, int, int]] = None) -> None:
        """Build the DAG from pipeline steps.

        Args:
            initial_shape: Initial dataset shape
        """
        self.nodes.clear()
        self.edges.clear()

        if not self.pipeline_steps:
            # Try to infer from predictions
            if self.predictions:
                self._build_dag_from_predictions()
            return

        # Default initial shape
        current_shape = initial_shape or (100, 1, 1000)

        # Create input node
        input_node = PipelineNode(
            id="input",
            step_index=0,
            label="Dataset",
            node_type="input",
            shape_after=current_shape,
        )
        self.nodes["input"] = input_node

        # Track current node IDs for edge connections
        current_node_ids = ["input"]
        branch_stacks: List[List[str]] = []  # Stack of lists of node IDs per branch level

        step_index = 0
        for step in self.pipeline_steps:
            step_index += 1
            step_info = self._parse_step(step, step_index)

            if step_info is None:
                continue

            node_type = step_info['type']
            label = step_info['label']
            keyword = step_info.get('keyword', '')

            # Handle branching
            if node_type == 'branch':
                # Create branch node
                branch_node = PipelineNode(
                    id=f"step_{step_index}_branch",
                    step_index=step_index,
                    label="Branch",
                    node_type="branch",
                    shape_before=current_shape,
                    shape_after=current_shape,
                    parent_ids=current_node_ids.copy(),
                )
                self.nodes[branch_node.id] = branch_node

                # Add edges from current nodes to branch
                for parent_id in current_node_ids:
                    self.edges.append((parent_id, branch_node.id))

                # Create nodes for each branch
                branches = step_info.get('branches', {})
                branch_node_ids = []

                for branch_id, (branch_name, branch_steps) in enumerate(branches.items()):
                    # Create branch entry node
                    entry_id = f"step_{step_index}_b{branch_id}_entry"
                    entry_label = branch_name if isinstance(branch_name, str) else f"Branch {branch_id}"
                    entry_node = PipelineNode(
                        id=entry_id,
                        step_index=step_index,
                        label=entry_label,
                        node_type="branch",
                        branch_id=branch_id,
                        branch_name=entry_label,
                        shape_before=current_shape,
                        shape_after=current_shape,
                        parent_ids=[branch_node.id],
                    )
                    self.nodes[entry_id] = entry_node
                    self.edges.append((branch_node.id, entry_id))

                    # Process branch substeps
                    branch_current = [entry_id]
                    branch_shape = current_shape
                    for substep_idx, substep in enumerate(branch_steps):
                        substep_info = self._parse_step(substep, step_index)
                        if substep_info:
                            substep_id = f"step_{step_index}_b{branch_id}_s{substep_idx}"
                            substep_node = PipelineNode(
                                id=substep_id,
                                step_index=step_index,
                                label=substep_info['label'],
                                node_type=substep_info['type'],
                                branch_id=branch_id,
                                branch_name=entry_label,
                                substep_index=substep_idx,
                                shape_before=branch_shape,
                                shape_after=self._estimate_shape_after(branch_shape, substep_info),
                                parent_ids=branch_current.copy(),
                            )
                            self.nodes[substep_id] = substep_node
                            for parent in branch_current:
                                self.edges.append((parent, substep_id))
                            branch_current = [substep_id]
                            branch_shape = substep_node.shape_after

                    branch_node_ids.extend(branch_current)

                # Push branch context
                branch_stacks.append(branch_node_ids)
                current_node_ids = branch_node_ids

            elif node_type == 'merge':
                # Create merge node
                merge_node = PipelineNode(
                    id=f"step_{step_index}_merge",
                    step_index=step_index,
                    label="Merge",
                    node_type="merge",
                    shape_before=current_shape,
                    shape_after=self._estimate_merge_shape(current_shape, step_info),
                    parent_ids=current_node_ids.copy(),
                )
                self.nodes[merge_node.id] = merge_node

                # Add edges from all branch ends to merge
                for parent_id in current_node_ids:
                    self.edges.append((parent_id, merge_node.id))

                # Pop branch context
                if branch_stacks:
                    branch_stacks.pop()

                current_node_ids = [merge_node.id]
                current_shape = merge_node.shape_after

            else:
                # Regular step
                node_id = f"step_{step_index}"
                new_shape = self._estimate_shape_after(current_shape, step_info)

                node = PipelineNode(
                    id=node_id,
                    step_index=step_index,
                    label=label,
                    node_type=node_type,
                    shape_before=current_shape,
                    shape_after=new_shape,
                    parent_ids=current_node_ids.copy(),
                    metadata={'keyword': keyword} if keyword else {},
                )
                self.nodes[node_id] = node

                # Add edges from current nodes
                for parent_id in current_node_ids:
                    self.edges.append((parent_id, node_id))

                current_node_ids = [node_id]
                current_shape = new_shape

    def _parse_step(self, step: Any, step_index: int) -> Optional[Dict[str, Any]]:
        """Parse a pipeline step into structured info.

        Args:
            step: Pipeline step definition
            step_index: Step index

        Returns:
            Dictionary with step info or None if unrecognized
        """
        # Handle None or empty
        if step is None:
            return None

        # Handle string steps (chart commands, etc.)
        if isinstance(step, str):
            if 'chart' in step.lower():
                return {'type': 'chart', 'label': step}
            return {'type': 'default', 'label': step}

        # Handle class (not instance)
        if isinstance(step, type):
            class_name = step.__name__
            return self._classify_operator(class_name, {})

        # Handle instance (has __class__)
        if hasattr(step, '__class__') and not isinstance(step, dict):
            class_name = step.__class__.__name__
            return self._classify_operator(class_name, {})

        # Handle dict steps
        if isinstance(step, dict):
            # Check for known keywords
            keywords = [
                'preprocessing', 'y_processing', 'feature_augmentation',
                'sample_augmentation', 'concat_transform', 'branch',
                'merge', 'source_branch', 'merge_sources', 'model',
                'split', 'name', 'merge_predictions'
            ]

            for keyword in keywords:
                if keyword in step:
                    return self._parse_keyword_step(keyword, step)

            # Check if it's a model dict
            if 'model' in step:
                return self._parse_keyword_step('model', step)

            # Generic dict step
            return {'type': 'default', 'label': str(list(step.keys())[0]) if step else '?'}

        # Handle list (could be a substep list)
        if isinstance(step, (list, tuple)):
            if len(step) == 1:
                return self._parse_step(step[0], step_index)
            return {'type': 'chain', 'label': f"[{len(step)} ops]"}

        return {'type': 'default', 'label': '?'}

    def _parse_keyword_step(self, keyword: str, step: Dict) -> Dict[str, Any]:
        """Parse a keyword-based step.

        Args:
            keyword: Step keyword
            step: Step dictionary

        Returns:
            Parsed step info
        """
        value = step.get(keyword)

        if keyword == 'preprocessing':
            op_name = self._get_operator_name(value)
            return {'type': 'preprocessing', 'label': op_name, 'keyword': keyword}

        elif keyword == 'y_processing':
            op_name = self._get_operator_name(value)
            return {'type': 'y_processing', 'label': f"y: {op_name}", 'keyword': keyword}

        elif keyword == 'feature_augmentation':
            if isinstance(value, list):
                ops = [self._get_operator_name(v) for v in value[:3]]
                label = "FA: " + ", ".join(ops)
                if len(value) > 3:
                    label += f"... (+{len(value)-3})"
            else:
                label = f"FA: {self._get_operator_name(value)}"
            action = step.get('action', 'add')
            return {'type': 'feature_augmentation', 'label': label, 'action': action, 'keyword': keyword}

        elif keyword == 'sample_augmentation':
            if isinstance(value, dict):
                transformers = value.get('transformers', [])
                count = value.get('count', '?')
                label = f"SA: {len(transformers)} aug ×{count}"
            else:
                label = "Sample Aug"
            return {'type': 'sample_augmentation', 'label': label, 'keyword': keyword}

        elif keyword == 'concat_transform':
            if isinstance(value, list):
                ops = [self._get_operator_name(v) for v in value]
                label = "Concat: " + "+".join(ops)
            elif isinstance(value, dict) and 'operations' in value:
                ops = [self._get_operator_name(v) for v in value['operations']]
                label = "Concat: " + "+".join(ops)
            else:
                label = "Concat Transform"
            return {'type': 'concat_transform', 'label': label, 'keyword': keyword}

        elif keyword == 'branch':
            branches = {}
            if isinstance(value, dict):
                for k, v in value.items():
                    if not k.startswith('_'):
                        branches[k] = v if isinstance(v, list) else [v]
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    branches[f"Branch {i}"] = v if isinstance(v, list) else [v]
            return {'type': 'branch', 'label': 'Branch', 'branches': branches, 'keyword': keyword}

        elif keyword == 'merge':
            merge_type = 'features' if value == 'features' else 'predictions'
            return {'type': 'merge', 'label': f"Merge ({merge_type})", 'merge_type': merge_type, 'keyword': keyword}

        elif keyword == 'merge_predictions':
            return {'type': 'merge', 'label': 'Merge Predictions', 'merge_type': 'predictions', 'keyword': keyword}

        elif keyword == 'source_branch':
            return {'type': 'source_branch', 'label': 'Source Branch', 'keyword': keyword}

        elif keyword == 'merge_sources':
            strategy = value if isinstance(value, str) else 'concat'
            return {'type': 'merge_sources', 'label': f"Merge Sources ({strategy})", 'keyword': keyword}

        elif keyword == 'model':
            model_name = step.get('name', self._get_operator_name(value))
            return {'type': 'model', 'label': model_name, 'keyword': keyword}

        elif keyword == 'split':
            splitter = value
            splitter_name = self._get_operator_name(splitter)
            return {'type': 'splitter', 'label': splitter_name, 'keyword': keyword}

        elif keyword == 'name':
            # Named step - look for model
            if 'model' in step:
                return {'type': 'model', 'label': step['name'], 'keyword': 'model'}
            return {'type': 'default', 'label': step['name']}

        return {'type': 'default', 'label': keyword, 'keyword': keyword}

    def _classify_operator(self, class_name: str, config: Dict) -> Dict[str, Any]:
        """Classify an operator by its class name.

        Args:
            class_name: Operator class name
            config: Operator configuration

        Returns:
            Step info dictionary
        """
        # Splitters
        splitter_names = ['KFold', 'StratifiedKFold', 'GroupKFold', 'ShuffleSplit',
                         'StratifiedShuffleSplit', 'GroupShuffleSplit', 'LeaveOneOut',
                         'LeaveOneGroupOut', 'TimeSeriesSplit']
        if class_name in splitter_names:
            return {'type': 'splitter', 'label': class_name}

        # Models
        model_indicators = ['Regressor', 'Classifier', 'Regression', 'SVC', 'SVR',
                           'LinearModel', 'Tree', 'Forest', 'Boost', 'Network',
                           'MLP', 'CNN', 'RNN', 'LSTM', 'Ridge', 'Lasso', 'Elastic',
                           'PLS', 'PCR', 'KNN', 'Naive', 'Bayes']
        for indicator in model_indicators:
            if indicator in class_name:
                return {'type': 'model', 'label': class_name}

        # Scalers
        if 'Scaler' in class_name or 'Normalizer' in class_name:
            return {'type': 'preprocessing', 'label': class_name}

        # NIRS transforms
        nirs_transforms = ['SNV', 'StandardNormalVariate', 'MSC', 'MultiplicativeScatterCorrection',
                          'FirstDerivative', 'SecondDerivative', 'SavitzkyGolay', 'Detrend',
                          'Gaussian', 'SmoothSignal', 'Baseline']
        if class_name in nirs_transforms:
            return {'type': 'preprocessing', 'label': class_name}

        # Decomposition
        if class_name in ['PCA', 'TruncatedSVD', 'NMF', 'ICA', 'FactorAnalysis']:
            return {'type': 'preprocessing', 'label': class_name}

        # Default
        return {'type': 'preprocessing', 'label': class_name}

    def _get_operator_name(self, op: Any) -> str:
        """Get a human-readable name for an operator.

        Args:
            op: Operator instance or class

        Returns:
            Operator name string
        """
        if op is None:
            return "None"
        if isinstance(op, str):
            return op
        if isinstance(op, type):
            return op.__name__
        if hasattr(op, '__class__'):
            return op.__class__.__name__
        return str(op)[:20]

    def _estimate_shape_after(
        self,
        shape_before: Tuple[int, int, int],
        step_info: Dict[str, Any]
    ) -> Tuple[int, int, int]:
        """Estimate the dataset shape after a step.

        Args:
            shape_before: Shape before the step (samples, processings, features)
            step_info: Step information

        Returns:
            Estimated shape after the step
        """
        if shape_before is None:
            return (100, 1, 1000)

        samples, processings, features = shape_before
        step_type = step_info.get('type', 'default')

        if step_type == 'feature_augmentation':
            # Feature augmentation adds processings
            action = step_info.get('action', 'add')
            if action == 'extend':
                # Adds new processings
                processings += 2  # Estimate
            elif action == 'replace':
                # Multiplies processings
                processings *= 2  # Estimate
            else:  # add
                processings *= 2  # Estimate

        elif step_type == 'sample_augmentation':
            # Sample augmentation adds samples
            samples = int(samples * 1.5)  # Estimate

        elif step_type == 'concat_transform':
            # Concat reduces features
            features = 50  # Estimate for PCA/SVD concat

        elif step_type == 'model':
            # Model doesn't change shape
            pass

        elif step_type == 'splitter':
            # Splitter creates folds but doesn't change shape
            pass

        return (samples, processings, features)

    def _estimate_merge_shape(
        self,
        shape_before: Tuple[int, int, int],
        step_info: Dict[str, Any]
    ) -> Tuple[int, int, int]:
        """Estimate shape after merge.

        Args:
            shape_before: Shape before merge
            step_info: Merge step info

        Returns:
            Estimated shape after merge
        """
        samples, processings, features = shape_before
        merge_type = step_info.get('merge_type', 'predictions')

        if merge_type == 'features':
            # Concatenate features from branches
            features *= 3  # Estimate for 3 branches
            processings = 1
        else:  # predictions
            # Stack predictions as features
            features = 3  # 1 prediction per branch (estimate 3 branches)
            processings = 1

        return (samples, processings, features)

    def _build_dag_from_predictions(self) -> None:
        """Build DAG from predictions object when no pipeline steps provided."""
        if not self.predictions:
            return

        # Try to get execution info from predictions
        try:
            preprocessings = self.predictions.get_unique_values('preprocessings')
            models = self.predictions.get_unique_values('model_name')
            branches = self.predictions.get_unique_values('branch_name')
        except (ValueError, KeyError):
            return

        # Create input node
        input_node = PipelineNode(
            id="input",
            step_index=0,
            label="Dataset",
            node_type="input",
        )
        self.nodes["input"] = input_node
        current_id = "input"

        # Add preprocessing summary
        if preprocessings:
            pp_list = [p for p in preprocessings if p]
            if pp_list:
                pp_node = PipelineNode(
                    id="preprocessing",
                    step_index=1,
                    label=f"Preprocessing\n({len(pp_list)} views)",
                    node_type="preprocessing",
                    parent_ids=[current_id],
                )
                self.nodes[pp_node.id] = pp_node
                self.edges.append((current_id, pp_node.id))
                current_id = pp_node.id

        # Add branches if present
        if branches and len([b for b in branches if b]) > 1:
            branch_node = PipelineNode(
                id="branches",
                step_index=2,
                label=f"Branches\n({len(branches)})",
                node_type="branch",
                parent_ids=[current_id],
            )
            self.nodes[branch_node.id] = branch_node
            self.edges.append((current_id, branch_node.id))
            current_id = branch_node.id

        # Add models
        if models:
            model_list = [m for m in models if m]
            model_node = PipelineNode(
                id="models",
                step_index=3,
                label=f"Models\n({', '.join(model_list[:3])}{'...' if len(model_list) > 3 else ''})",
                node_type="model",
                parent_ids=[current_id],
            )
            self.nodes[model_node.id] = model_node
            self.edges.append((current_id, model_node.id))

    def _compute_layout(self) -> Dict[str, Dict[str, Any]]:
        """Compute node positions using topological sort and layering.

        Branches maintain their column positions throughout their execution,
        with nodes stacked vertically in their assigned columns.

        Returns:
            Dictionary mapping node IDs to position info
        """
        layout = {}

        if not self.nodes:
            return layout

        # Compute layers using topological sort
        layers = self._compute_layers()

        # Position nodes
        y_spacing = 1.8
        x_spacing = 2.8

        # Assign fixed column positions for each branch
        # Collect all unique branch_ids
        branch_ids = set()
        for node in self.nodes.values():
            if node.branch_id is not None:
                branch_ids.add(node.branch_id)

        # Sort branch IDs and create column mapping
        sorted_branches = sorted(branch_ids)
        n_branches = len(sorted_branches)
        branch_column = {bid: i for i, bid in enumerate(sorted_branches)}

        for layer_idx, layer_nodes in enumerate(layers):
            y = -layer_idx * y_spacing

            # Separate nodes into branched and non-branched
            branched_nodes = [(nid, self.nodes[nid].branch_id) for nid in layer_nodes
                              if self.nodes[nid].branch_id is not None]
            unbranched_nodes = [nid for nid in layer_nodes
                                if self.nodes[nid].branch_id is None]

            # Position branched nodes in their fixed columns
            if n_branches > 0:
                branch_x_start = -(n_branches - 1) * x_spacing / 2
                for node_id, bid in branched_nodes:
                    col = branch_column[bid]
                    x = branch_x_start + col * x_spacing
                    layout[node_id] = {
                        'x': x,
                        'y': y,
                        'node': self.nodes[node_id],
                    }

            # Position unbranched nodes centered
            if unbranched_nodes:
                n_unbranched = len(unbranched_nodes)
                x_start = -(n_unbranched - 1) * x_spacing / 2
                for i, node_id in enumerate(unbranched_nodes):
                    x = x_start + i * x_spacing
                    layout[node_id] = {
                        'x': x,
                        'y': y,
                        'node': self.nodes[node_id],
                    }

        return layout

    def _compute_layers(self) -> List[List[str]]:
        """Compute node layers using topological ordering.

        Returns:
            List of lists, where each inner list contains node IDs for that layer
        """
        # Build adjacency and in-degree
        in_degree = {node_id: 0 for node_id in self.nodes}
        adj = defaultdict(list)

        for from_id, to_id in self.edges:
            adj[from_id].append(to_id)
            in_degree[to_id] += 1

        # Find roots (nodes with no parents)
        roots = [node_id for node_id, degree in in_degree.items() if degree == 0]

        # BFS layering
        layers = []
        current_layer = roots
        visited = set()

        while current_layer:
            layers.append(current_layer)
            visited.update(current_layer)

            next_layer = []
            for node_id in current_layer:
                for child_id in adj[node_id]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0 and child_id not in visited:
                        next_layer.append(child_id)

            current_layer = next_layer

        return layers

    def _format_shape_display(self, node: PipelineNode) -> List[str]:
        """Format shape information for display in a node.

        Shows:
        - For single source: S×P×F (samples × preprocessings × features)
        - For multi-source: Source count + total features
        - Always shows 2D layout shape when available

        Args:
            node: The pipeline node with shape info

        Returns:
            List of formatted shape strings
        """
        shape_lines = []

        # 3D per-source shapes are preferred (samples × preprocessings × features)
        if node.features_shape:
            if len(node.features_shape) == 1:
                # Single source: show full S×P×F
                ss, pp, ff = node.features_shape[0]
                shape_lines.append(f"{ss}×{pp}×{ff}")
            else:
                # Multi-source: show per-source summary
                n_sources = len(node.features_shape)
                total_features = 0
                total_pp = 0
                n_samples = node.features_shape[0][0] if node.features_shape else 0

                source_details = []
                for i, (s, p, f) in enumerate(node.features_shape):
                    total_features += f * p  # Features after layout expansion
                    total_pp = max(total_pp, p)
                    # Short summary per source
                    source_details.append(f"S{i}:{p}×{f}")

                # Show sources summary
                if n_sources <= 2:
                    # Show all sources inline
                    shape_lines.append(f"{n_samples} samples")
                    shape_lines.append(" | ".join(source_details))
                else:
                    # Show aggregate for many sources
                    shape_lines.append(f"{n_samples}×{n_sources}src")
                    shape_lines.append(f"Σ {total_features} feat")

        elif node.output_layout_shape:
            # Fallback to 2D layout: (samples, features)
            s, f = node.output_layout_shape
            shape_lines.append(f"({s}, {f})")

        elif node.shape_after:
            # Fallback to 3D estimated shape (static diagram)
            s, p, f = node.shape_after
            shape_lines.append(f"{s}×{p}×{f}")

        return shape_lines

    def _draw_nodes(
        self,
        ax: Axes,
        layout: Dict[str, Dict[str, Any]],
        show_shapes: bool
    ) -> None:
        """Draw nodes on the diagram.

        Args:
            ax: Matplotlib axes
            layout: Node layout
            show_shapes: Whether to show shape info
        """
        for node_id, pos_info in layout.items():
            x, y = pos_info['x'], pos_info['y']
            node = pos_info['node']

            # Get color
            color = self.NODE_COLORS.get(node.node_type, self.NODE_COLORS['default'])

            # Build label with improved shape display
            label_lines = [node.label]

            if show_shapes:
                shape_lines = self._format_shape_display(node)
                label_lines.extend(shape_lines)

            # Add score for model nodes
            score = node.metadata.get('best_score')
            if score is not None and node.node_type == 'model':
                label_lines.append(f"★ {score:.2f}")

            n_lines = len(label_lines)

            # Adjust height for multi-line
            box_height = self._node_height * (1 + 0.3 * (n_lines - 1))
            box_width = self._node_width

            # Store dimensions for edge drawing
            pos_info['width'] = box_width
            pos_info['height'] = box_height

            # Draw node box
            rect = FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width, box_height,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor=color,
                edgecolor='#34495e',
                linewidth=1.0,
                alpha=0.95,
            )
            ax.add_patch(rect)

            # Calculate vertical positions for text lines
            total_text_height = n_lines * 0.18
            start_y = y + total_text_height / 2 - 0.09

            # Draw operator label (first line)
            ax.text(
                x, start_y,
                node.label,
                ha='center', va='center',
                fontsize=self._fontsize,
                color='#1a252f',
                fontweight='medium',
            )

            # Draw additional info (shapes, score)
            if len(label_lines) > 1:
                for i, line in enumerate(label_lines[1:], 1):
                    line_y = start_y - i * 0.16
                    # Use different style for score line
                    if line.startswith('★'):
                        ax.text(
                            x, line_y,
                            line,
                            ha='center', va='center',
                            fontsize=self._fontsize - 0.5,
                            color='#27ae60',
                            fontweight='bold',
                        )
                    else:
                        ax.text(
                            x, line_y,
                            line,
                            ha='center', va='center',
                            fontsize=self._fontsize - 1,
                            color='#5d6d7e',
                            fontweight='normal',
                        )

    def _draw_edges(
        self,
        ax: Axes,
        layout: Dict[str, Dict[str, Any]]
    ) -> None:
        """Draw edges connecting nodes.

        Args:
            ax: Matplotlib axes
            layout: Node layout
        """
        for from_id, to_id in self.edges:
            if from_id not in layout or to_id not in layout:
                continue

            from_pos = layout[from_id]
            to_pos = layout[to_id]

            # Get node dimensions (use defaults if not computed yet)
            from_height = from_pos.get('height', self._node_height)
            to_height = to_pos.get('height', self._node_height)

            # Calculate connection points
            from_x, from_y = from_pos['x'], from_pos['y'] - from_height / 2
            to_x, to_y = to_pos['x'], to_pos['y'] + to_height / 2

            # Draw arrow
            ax.annotate(
                '',
                xy=(to_x, to_y),
                xytext=(from_x, from_y),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='#7f8c8d',
                    linewidth=1.5,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle='arc3,rad=0.0',
                )
            )

    def _get_bounds(
        self,
        layout: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, float, float, float]:
        """Get bounding box for the diagram.

        Args:
            layout: Node layout

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        if not layout:
            return -1, 1, -1, 1

        x_coords = [p['x'] for p in layout.values()]
        y_coords = [p['y'] for p in layout.values()]

        x_min = min(x_coords) - self._node_width
        x_max = max(x_coords) + self._node_width
        y_min = min(y_coords) - self._node_height * 2
        y_max = max(y_coords) + self._node_height * 2

        return x_min, x_max, y_min, y_max

    def _add_legend(self, ax: Axes) -> None:
        """Add a legend showing node type colors.

        Args:
            ax: Matplotlib axes
        """
        legend_items = [
            ('Input/Output', self.NODE_COLORS['input']),
            ('Preprocessing', self.NODE_COLORS['preprocessing']),
            ('Feature Aug', self.NODE_COLORS['feature_augmentation']),
            ('Sample Aug', self.NODE_COLORS['sample_augmentation']),
            ('Y Processing', self.NODE_COLORS['y_processing']),
            ('Splitter', self.NODE_COLORS['splitter']),
            ('Branch/Merge', self.NODE_COLORS['branch']),
            ('Model', self.NODE_COLORS['model']),
        ]

        patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in legend_items
        ]

        ax.legend(
            handles=patches,
            loc='upper right',
            fontsize=self._fontsize - 1,
            framealpha=0.9,
            ncol=2,
        )


def plot_pipeline_diagram(
    pipeline_steps: Optional[List[Any]] = None,
    predictions: Any = None,
    show_shapes: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    initial_shape: Optional[Tuple[int, int, int]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Figure:
    """Convenience function to create a pipeline diagram.

    Args:
        pipeline_steps: List of pipeline step definitions
        predictions: Optional Predictions object with execution data
        show_shapes: Whether to show shape info in nodes
        figsize: Figure size tuple
        title: Optional title for the diagram
        initial_shape: Initial dataset shape (samples, processings, features)
        config: Additional configuration dict

    Returns:
        matplotlib Figure object

    Example:
        >>> from nirs4all.visualization.pipeline_diagram import plot_pipeline_diagram
        >>> fig = plot_pipeline_diagram(pipeline, initial_shape=(189, 1, 2151))
        >>> fig.savefig('pipeline_diagram.png')
    """
    cfg = config or {}
    diagram = PipelineDiagram(pipeline_steps, predictions, config=cfg)
    return diagram.render(
        show_shapes=show_shapes,
        figsize=figsize,
        title=title,
        initial_shape=initial_shape,
    )


# Backward compatibility alias
BranchDiagram = PipelineDiagram
plot_branch_diagram = plot_pipeline_diagram

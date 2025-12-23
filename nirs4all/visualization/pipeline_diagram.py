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

    # Professional color palette (Fill, Border)
    NODE_STYLES = {
        'preprocessing':        ('#E3F2FD', '#1976D2'),  # Blue 50 / 700
        'feature_augmentation': ('#E0F2F1', '#00796B'),  # Teal 50 / 700
        'sample_augmentation':  ('#E8F5E9', '#388E3C'),  # Green 50 / 700
        'concat_transform':     ('#F3E5F5', '#7B1FA2'),  # Purple 50 / 700
        'y_processing':         ('#FFF8E1', '#FFA000'),  # Amber 50 / 700
        'splitter':             ('#F3E5F5', '#7B1FA2'),  # Purple 50 / 700
        'branch':               ('#E0F2F1', '#00796B'),  # Teal 50 / 700
        'merge':                ('#E0F2F1', '#00796B'),  # Teal 50 / 700
        'source_branch':        ('#E0F2F1', '#00796B'),  # Teal 50 / 700
        'merge_sources':        ('#E0F2F1', '#00796B'),  # Teal 50 / 700
        'model':                ('#FFEBEE', '#D32F2F'),  # Red 50 / 700
        'input':                ('#FAFAFA', '#616161'),  # Gray 50 / 700
        'output':               ('#FAFAFA', '#616161'),  # Gray 50 / 700
        'default':              ('#ECEFF1', '#455A64'),  # Blue Grey 50 / 700
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

        # Default configuration - optimized for publication
        self._figsize = self.config.get('figsize', (12, 8))
        self._fontsize = self.config.get('fontsize', 9)
        self._node_width = self.config.get('node_width', 2.5)
        self._node_height = self.config.get('node_height', 0.8)
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
        ax.set_title(title, fontsize=self._fontsize + 4, fontweight='bold', pad=20, color='#263238')

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
        predictions: Any = None,
        config: Optional[Dict[str, Any]] = None
    ) -> 'PipelineDiagram':
        """Create a PipelineDiagram from an ExecutionTrace.

        This builds the diagram using actual runtime data including
        measured shapes at each step.

        Args:
            execution_trace: ExecutionTrace object from pipeline execution
            predictions: Optional Predictions object to enrich nodes with scores
            config: Optional configuration dict

        Returns:
            PipelineDiagram instance ready for rendering

        Example:
            >>> from nirs4all.visualization import PipelineDiagram
            >>> diagram = PipelineDiagram.from_trace(trace)
            >>> fig = diagram.render(title="Execution Trace")
        """
        diagram = cls(
            execution_trace=execution_trace,
            predictions=predictions,
            config=config
        )
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

        # Pre-process: collect shapes from steps for inheritance
        step_shapes = {}  # step_index -> (input_features, output_features)
        for step in steps:
            idx = getattr(step, 'step_index', 0)
            input_f = getattr(step, 'input_features_shape', None)
            output_f = getattr(step, 'output_features_shape', None)
            if input_f or output_f:
                step_shapes[idx] = (input_f, output_f)

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
        in_source_branch_mode = False  # Track if we're inside source_branch (before merge)
        source_branch_node_ids: List[str] = []  # Track source branch node IDs
        n_sources = 0  # Number of sources in source_branch mode
        last_pre_branch_node = "input"  # Track the node before entering branches
        last_known_features_shape = input_features  # Track last known shape for inheritance

        for step_i, step in enumerate(steps):
            step_index = getattr(step, 'step_index', 0)
            operator_class = getattr(step, 'operator_class', '') or ''
            operator_type = getattr(step, 'operator_type', '') or ''
            branch_path = tuple(getattr(step, 'branch_path', []) or [])
            branch_name = getattr(step, 'branch_name', '') or ''
            duration_ms = getattr(step, 'duration_ms', 0.0)
            substep_idx = getattr(step, 'substep_index', None)
            operator_config = getattr(step, 'operator_config', {}) or {}

            # Get shapes from trace
            input_layout = getattr(step, 'input_shape', None)
            output_layout = getattr(step, 'output_shape', None)
            input_features = getattr(step, 'input_features_shape', None)
            output_features = getattr(step, 'output_features_shape', None)

            # Check for source_branch and expand it
            is_source_branch = operator_config.get('source_branch', False)
            if is_source_branch:
                # Get number of sources from operator_config
                n_sources = operator_config.get('n_sources', 0)
                # Create expanded nodes for source_branch
                self._expand_source_branch(
                    step, step_index, steps, step_i,
                    current_node_ids, branch_stacks, last_known_features_shape
                )
                # Update current_node_ids to point to the source branch nodes
                source_branch_ids = [nid for nid in self.nodes.keys()
                                     if nid.startswith(f"step_{step_index}_src")]
                source_branch_ids.sort()  # Ensure consistent ordering
                if source_branch_ids:
                    current_node_ids = source_branch_ids
                    source_branch_node_ids = source_branch_ids.copy()
                    in_branch_mode = True
                    in_source_branch_mode = True
                continue

            # Check if this step runs on multiple sources (inside source_branch mode)
            # and should be expanded to show one node per source
            artifacts = getattr(step, 'artifacts', None)
            by_source = {}
            if artifacts:
                if hasattr(artifacts, 'by_source'):
                    by_source = artifacts.by_source or {}
                elif isinstance(artifacts, dict):
                    by_source = artifacts.get('by_source', {})

            op_type_lower = operator_type.lower() if operator_type else ''

            # Check if this is a merge step that exits source_branch mode
            if in_source_branch_mode and op_type_lower == 'merge' and not branch_path:
                # This is the source branch merge - exit source_branch mode
                self._create_merge_node_for_source_branch(
                    step, step_index, operator_class, current_node_ids,
                    input_layout, output_layout, input_features, output_features
                )
                # Exit source_branch mode
                in_source_branch_mode = False
                source_branch_node_ids = []
                current_node_ids = [f"step_{step_index}"]
                in_branch_mode = False
                branch_stacks.clear()
                continue

            # Expand step if in source_branch mode and step runs on multiple sources
            if in_source_branch_mode and len(by_source) > 1 and not branch_path:
                # Expand this step into per-source nodes
                self._expand_step_for_sources(
                    step, step_index, operator_class, operator_type,
                    by_source, current_node_ids, input_features, output_features
                )
                # Update current_node_ids to point to the expanded nodes
                expanded_ids = [nid for nid in self.nodes.keys()
                                if nid.startswith(f"step_{step_index}_src")]
                expanded_ids.sort()
                if expanded_ids:
                    current_node_ids = expanded_ids
                    source_branch_node_ids = expanded_ids.copy()
                continue

            # Fallback for features shape if missing but layout exists
            if output_features is None and output_layout is not None:
                s, f = output_layout
                output_features = [(s, 1, f)]

            # If output_features still missing, try input_features as fallback
            if output_features is None and input_features is not None:
                output_features = input_features

            # Derive model output shape from predictions (model output is prediction, not feature concat)
            node_type_tmp = self._classify_operator_type(operator_type, operator_class)
            derived_shape = None
            if node_type_tmp == 'model' and self.predictions is not None:
                derived_shape = self._derive_model_input_shape_from_predictions(
                    step_index=step_index,
                    branch_path=branch_path,
                    branch_name=branch_name,
                    operator_class=operator_class,
                    operator_config=operator_config
                )

            # For model nodes, always prefer derived shape (represents prediction output, not input features)
            if derived_shape and node_type_tmp == 'model':
                output_features = [derived_shape]
                output_layout = (derived_shape[0], derived_shape[2])

            # For branch substeps with no shape, try to inherit from step shapes
            if output_features is None and branch_path:
                # Try to get shape from next non-branch step
                for future_step in steps[step_i + 1:]:
                    future_bp = tuple(getattr(future_step, 'branch_path', []) or [])
                    if not future_bp:  # Non-branch step
                        future_input = getattr(future_step, 'input_features_shape', None)
                        if future_input:
                            output_features = future_input
                            break

            # If still no shape, use last known
            if output_features is None and last_known_features_shape:
                output_features = last_known_features_shape

            # Same fallback for input shapes
            if input_features is None and input_layout is not None:
                s, f = input_layout
                input_features = [(s, 1, f)]

            # Update last known shape if we have output
            if output_features and not branch_path:
                last_known_features_shape = output_features

            # Extract metadata including score and custom name
            metadata = {}
            if operator_config and 'n_splits' in operator_config:
                metadata['n_splits'] = operator_config['n_splits']

            score = getattr(step, 'score', None)
            if score is None and hasattr(step, 'metadata'):
                score = step.metadata.get('score')
            if score is not None:
                metadata['best_score'] = score

            # Try to get custom name from artifacts metadata
            artifacts = getattr(step, 'artifacts', None)
            if artifacts and hasattr(artifacts, 'metadata'):
                art_meta = artifacts.metadata if hasattr(artifacts, 'metadata') else {}
                if isinstance(art_meta, dict):
                    custom_name = art_meta.get('custom_name', '')
                    if custom_name:
                        metadata['custom_name'] = custom_name

            # Enrich model nodes with best score from predictions if available
            if metadata.get('best_score') is None and self.predictions is not None:
                model_name = metadata.get('custom_name')
                if not model_name and isinstance(operator_config, dict):
                    model_name = operator_config.get('name') or operator_config.get('model_name')

                best_score = self._get_best_score_from_predictions(
                    step_index=step_index,
                    branch_path=branch_path,
                    branch_name=branch_name,
                    model_name=model_name,
                    operator_class=operator_class
                )
                if best_score is not None:
                    metadata['best_score'] = best_score

            # Determine node type from operator info
            node_type = node_type_tmp

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
                bp_tuple = branch_path
                parent_found = False
                while len(bp_tuple) > 0:
                    if bp_tuple in branch_stacks:
                        parent_nodes = branch_stacks[bp_tuple]
                        parent_found = True
                        break
                    bp_tuple = bp_tuple[:-1]

                if not parent_found:
                    branch_id = branch_path[0] if branch_path else 0
                    matching_leaves = []
                    for bpath, node_ids in branch_stacks.items():
                        if bpath and bpath[0] == branch_id:
                            matching_leaves.extend(node_ids)
                    parent_nodes = matching_leaves if matching_leaves else current_node_ids
                in_branch_mode = True
            elif op_type_lower == 'merge' and in_branch_mode and branch_stacks:
                # Merge step exiting branch mode - connect to the LEAF nodes of each branch
                # Group by top-level branch ID and take the most recent (shallowest path length for same branch)
                branch_leaves_by_id: Dict[int, str] = {}
                for bpath, node_ids in branch_stacks.items():
                    if not bpath:
                        continue
                    top_branch_id = bpath[0]
                    # Prefer shorter paths (they are more recent, like MetaModel at bp=[0] vs substep at bp=[0,0])
                    current_depth = len(bpath)
                    if node_ids:
                        latest_node = node_ids[-1]
                        if top_branch_id not in branch_leaves_by_id:
                            branch_leaves_by_id[top_branch_id] = (current_depth, latest_node)
                        else:
                            existing_depth, _ = branch_leaves_by_id[top_branch_id]
                            if current_depth < existing_depth:
                                # Shorter path = more recent (e.g., MetaModel at [0] vs substep at [0,0])
                                branch_leaves_by_id[top_branch_id] = (current_depth, latest_node)

                all_branch_leaves = [node_id for _, node_id in branch_leaves_by_id.values()]
                parent_nodes = all_branch_leaves if all_branch_leaves else current_node_ids
                in_branch_mode = False
                branch_stacks.clear()
            elif op_type_lower == 'merge' and in_branch_mode:
                parent_nodes = current_node_ids
                in_branch_mode = False
            elif op_type_lower == 'branch':
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
                input_layout_shape=input_layout,
                output_layout_shape=output_layout,
                features_shape=output_features if output_features else input_features,
                branch_id=branch_path[-1] if branch_path else None,
                branch_name=branch_name,
                substep_index=substep_idx,
                parent_ids=list(parent_nodes) if parent_nodes else [],
                duration_ms=duration_ms,
                metadata=metadata,
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

    def _get_best_score_from_predictions(
        self,
        step_index: int,
        branch_path: List[int],
        branch_name: str,
        model_name: Optional[str],
        operator_class: str
    ) -> Optional[float]:
        """Return best score (prefer test of best val; fallback to best available)."""
        if self.predictions is None:
            return None

        filter_kwargs = {'step_idx': step_index, 'load_arrays': False}
        if branch_path:
            filter_kwargs['branch_id'] = branch_path[-1]
        if branch_name:
            filter_kwargs['branch_name'] = branch_name

        try:
            preds = self.predictions.filter_predictions(**filter_kwargs)
        except Exception:
            preds = []

        # Fallback: if nothing matched with branch filters, try without them
        if not preds:
            try:
                slim_kwargs = {'step_idx': step_index, 'load_arrays': False}
                preds = self.predictions.filter_predictions(**slim_kwargs)
            except Exception:
                preds = []

        if model_name:
            preds = [p for p in preds if p.get('model_name') == model_name]

        # If nothing yet, try class match
        if not preds and operator_class:
            preds = [p for p in preds if p.get('model_classname') == operator_class]

        # Last resort: drop model filters and keep step match only
        if not preds:
            try:
                preds = self.predictions.filter_predictions(step_idx=step_index, load_arrays=False)
            except Exception:
                preds = []

        if not preds:
            return None

        lower_tokens = (
            'rmse', 'mae', 'mse', 'mape', 'msle',
            'logloss', 'log_loss', 'loss', 'error', 'hinge'
        )

        def is_higher_better(metric_name: str) -> bool:
            metric_lower = (metric_name or '').lower()
            return not any(tok in metric_lower for tok in lower_tokens)

        # Rank entries by val_score when present; otherwise by test_score/train_score
        def primary_score(pred: Dict[str, Any]) -> Optional[float]:
            if pred.get('val_score') is not None:
                return pred.get('val_score')
            if pred.get('test_score') is not None:
                return pred.get('test_score')
            return pred.get('train_score')

        best_entry: Optional[Dict[str, Any]] = None
        higher_is_better: Optional[bool] = None

        for pred in preds:
            score_candidate = primary_score(pred)
            if score_candidate is None:
                continue

            if higher_is_better is None:
                higher_is_better = is_higher_better(pred.get('metric', ''))

            if best_entry is None:
                best_entry = pred
                continue

            current_best = primary_score(best_entry)
            if current_best is None:
                best_entry = pred
                continue

            if higher_is_better:
                if score_candidate > current_best:
                    best_entry = pred
            else:
                if score_candidate < current_best:
                    best_entry = pred

        if best_entry is None:
            return None

        # Display priority: test -> val -> train
        for key in ('test_score', 'val_score', 'train_score'):
            if best_entry.get(key) is not None:
                return best_entry.get(key)
        return None

    def _get_total_samples_from_trace(self) -> Optional[int]:
        """Get total sample count from execution trace.

        Looks for sample count after sample_augmentation or at the splitter step,
        which reflects the true dataset size including augmented samples.
        """
        if not self.execution_trace:
            return None

        steps = getattr(self.execution_trace, 'steps', [])
        total_samples = None

        # Look for sample count in order of preference:
        # 1. splitter step (most reliable - shows full dataset before CV split)
        # 2. sample_augmentation output (if present)
        # 3. Any step with features shape
        for step in steps:
            op_type = getattr(step, 'operator_type', '') or ''
            output_features = getattr(step, 'output_features_shape', None)

            if op_type.lower() == 'splitter' and output_features:
                # Splitter shows the full dataset size
                total_samples = output_features[0][0]
                break

        # Fallback: look for sample_augmentation or any step with features
        if total_samples is None:
            for step in steps:
                op_type = getattr(step, 'operator_type', '') or ''
                output_features = getattr(step, 'output_features_shape', None)

                if op_type.lower() == 'sample_augmentation' and output_features:
                    total_samples = output_features[0][0]
                    break

        # Final fallback: first step with output features
        if total_samples is None:
            for step in steps:
                output_features = getattr(step, 'output_features_shape', None)
                if output_features and output_features[0]:
                    total_samples = output_features[0][0]
                    break

        return total_samples

    def _derive_model_input_shape_from_predictions(
        self,
        step_index: int,
        branch_path: List[int],
        branch_name: str,
        operator_class: str,
        operator_config: Dict[str, Any],
    ) -> Optional[Tuple[int, int, int]]:
        """Infer model INPUT 3D shape (samples, processings, features) from predictions metadata.

        For models, we want to show the input shape (what the model receives),
        not the output shape (predictions). The n_features field in predictions
        stores the input feature count.

        Sample count is derived from the execution trace to include augmented samples.
        """
        if self.predictions is None:
            return None

        filter_kwargs = {'step_idx': step_index, 'load_arrays': False}
        if branch_path:
            filter_kwargs['branch_id'] = branch_path[-1]
        if branch_name:
            filter_kwargs['branch_name'] = branch_name

        try:
            preds = self.predictions.filter_predictions(**filter_kwargs)
        except Exception:
            preds = []

        if not preds:
            try:
                preds = self.predictions.filter_predictions(step_idx=step_index, load_arrays=False)
            except Exception:
                preds = []

        if not preds:
            return None

        # Get feature count from predictions metadata
        feature_counts: List[int] = []
        for pred in preds:
            n_features = pred.get('n_features')
            if n_features not in (None, 0):
                feature_counts.append(int(n_features))

        # Use minimum n_features (merged features are typically smaller than concat)
        feature_count = min(feature_counts) if feature_counts else None

        if feature_count is None:
            return None

        # Get total sample count from trace (includes augmented samples)
        sample_count = self._get_total_samples_from_trace()

        # Fallback: calculate from predictions partitions if trace not available
        if sample_count is None:
            partition_counts: Dict[str, int] = {}
            for pred in preds:
                n_samples = int(pred.get('n_samples') or 0)
                partition = str(pred.get('partition') or '').lower()
                if partition and n_samples > 0:
                    partition_counts[partition] = max(partition_counts.get(partition, 0), n_samples)

            if partition_counts:
                if 'train' in partition_counts:
                    sample_count = partition_counts['train']
                    if 'test' in partition_counts:
                        sample_count += partition_counts['test']
                    elif 'predict' in partition_counts:
                        sample_count += partition_counts['predict']
                else:
                    sample_count = sum(partition_counts.values())

        if sample_count is None or sample_count == 0:
            return None

        # Return as 3D shape (samples, 1 processing, features)
        return (sample_count, 1, int(feature_count))

    def _expand_source_branch(
        self,
        step: Any,
        step_index: int,
        all_steps: List[Any],
        step_i: int,
        current_node_ids: List[str],
        branch_stacks: Dict[tuple, List[str]],
        last_known_shape: Optional[List[Tuple[int, int, int]]]
    ) -> None:
        """Expand a source_branch step into per-source nodes.

        Args:
            step: The source_branch step
            step_index: Step index
            all_steps: All execution steps
            step_i: Index of current step in all_steps
            current_node_ids: Current parent node IDs
            branch_stacks: Branch tracking dict
            last_known_shape: Last known features shape
        """
        artifacts = getattr(step, 'artifacts', None)
        if not artifacts:
            return

        # Get by_chain to understand what transformers are in each source
        by_chain = {}
        if hasattr(artifacts, 'by_chain'):
            by_chain = artifacts.by_chain
        elif isinstance(artifacts, dict):
            by_chain = artifacts.get('by_chain', {})

        if not by_chain:
            return

        # Parse chain keys to group by source
        # Format: s5.MinMaxScaler[src=0,sub=0], s5.PCA[src=2,sub=8]
        source_ops: Dict[int, List[Tuple[int, str]]] = {}  # source_id -> [(substep, class_name), ...]
        for chain_key in by_chain.keys():
            # Extract source and class from chain key
            if '[src=' in chain_key:
                parts = chain_key.split('[')
                class_part = parts[0].split('.')[-1]  # e.g., "MinMaxScaler"
                src_part = parts[1] if len(parts) > 1 else ''
                if 'src=' in src_part:
                    src_str = src_part.split('src=')[1].split(',')[0].split(']')[0]
                    try:
                        src_id = int(src_str)
                    except ValueError:
                        continue
                    sub_str = src_part.split('sub=')[1].split(']')[0] if 'sub=' in src_part else '0'
                    try:
                        sub_id = int(sub_str)
                    except ValueError:
                        sub_id = 0

                    if src_id not in source_ops:
                        source_ops[src_id] = []
                    source_ops[src_id].append((sub_id, class_part))

        if not source_ops:
            return

        # Get shapes from next step if available
        next_step_shapes = None
        if step_i + 1 < len(all_steps):
            next_step = all_steps[step_i + 1]
            next_step_shapes = getattr(next_step, 'input_features_shape', None)

        # Create a node for each source branch
        for src_id in sorted(source_ops.keys()):
            ops_list = source_ops[src_id]
            # Sort by substep and get unique class names
            ops_list.sort(key=lambda x: x[0])
            seen_classes = []
            for _, cls_name in ops_list:
                if cls_name not in seen_classes:
                    seen_classes.append(cls_name)

            # Create label
            if len(seen_classes) <= 2:
                label = f"Src{src_id}: {' → '.join(seen_classes)}"
            else:
                label = f"Src{src_id}: {seen_classes[0]}...{seen_classes[-1]}"

            # Get shape for this source
            src_shape = None
            if next_step_shapes and src_id < len(next_step_shapes):
                src_shape = [next_step_shapes[src_id]]
            elif last_known_shape and src_id < len(last_known_shape):
                src_shape = [last_known_shape[src_id]]

            node_id = f"step_{step_index}_src{src_id}"
            node = PipelineNode(
                id=node_id,
                step_index=step_index,
                label=label,
                node_type='source_branch',
                branch_id=src_id,
                branch_name=f"source_{src_id}",
                features_shape=src_shape,
                parent_ids=current_node_ids.copy(),
            )
            self.nodes[node_id] = node

            # Add edges from parents
            for parent_id in current_node_ids:
                if parent_id in self.nodes:
                    self.edges.append((parent_id, node_id))

            # Track in branch stacks
            branch_stacks[(src_id,)] = [node_id]

    def _expand_step_for_sources(
        self,
        step: Any,
        step_index: int,
        operator_class: str,
        operator_type: str,
        by_source: Dict[int, List[str]],
        current_node_ids: List[str],
        input_features: Optional[List[Tuple[int, int, int]]],
        output_features: Optional[List[Tuple[int, int, int]]]
    ) -> None:
        """Expand a step that runs on multiple sources into per-source nodes.

        Args:
            step: The execution step
            step_index: Step index
            operator_class: Class name
            operator_type: Operator type
            by_source: Artifacts by source
            current_node_ids: Current parent node IDs (should be source branch nodes)
            input_features: Input feature shapes
            output_features: Output feature shapes
        """
        # Create a node for each source
        for src_id in sorted(by_source.keys()):
            # Get shape for this source
            src_shape = None
            if output_features and src_id < len(output_features):
                src_shape = [output_features[src_id]]
            elif input_features and src_id < len(input_features):
                src_shape = [input_features[src_id]]

            # Find the parent node for this source
            parent_id = None
            for pid in current_node_ids:
                if f"_src{src_id}" in pid:
                    parent_id = pid
                    break
            if parent_id is None and current_node_ids:
                # Fallback: use the src_id-th parent if available
                if src_id < len(current_node_ids):
                    parent_id = current_node_ids[src_id]
                else:
                    parent_id = current_node_ids[0]

            node_id = f"step_{step_index}_src{src_id}"
            label = f"Src{src_id}: {operator_class}"
            node = PipelineNode(
                id=node_id,
                step_index=step_index,
                label=label,
                node_type=self._classify_operator_type(operator_type, operator_class),
                branch_id=src_id,
                branch_name=f"source_{src_id}",
                features_shape=src_shape,
                parent_ids=[parent_id] if parent_id else [],
            )
            self.nodes[node_id] = node

            # Add edge from parent
            if parent_id and parent_id in self.nodes:
                self.edges.append((parent_id, node_id))

    def _create_merge_node_for_source_branch(
        self,
        step: Any,
        step_index: int,
        operator_class: str,
        current_node_ids: List[str],
        input_layout: Optional[Tuple[int, int]],
        output_layout: Optional[Tuple[int, int]],
        input_features: Optional[List[Tuple[int, int, int]]],
        output_features: Optional[List[Tuple[int, int, int]]]
    ) -> None:
        """Create a merge node that collects from all source branches.

        Args:
            step: The execution step
            step_index: Step index
            operator_class: Class name
            current_node_ids: Current parent node IDs (source branch nodes)
            input_layout: Input layout shape
            output_layout: Output layout shape
            input_features: Input feature shapes
            output_features: Output feature shapes
        """
        node_id = f"step_{step_index}"
        node = PipelineNode(
            id=node_id,
            step_index=step_index,
            label="Merge Sources",
            node_type='merge_sources',
            input_layout_shape=input_layout,
            output_layout_shape=output_layout,
            features_shape=output_features if output_features else input_features,
            parent_ids=current_node_ids.copy(),
        )
        self.nodes[node_id] = node

        # Add edges from all source branch nodes
        for parent_id in current_node_ids:
            if parent_id in self.nodes:
                self.edges.append((parent_id, node_id))

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
        branch_name = getattr(step, 'branch_name', '') or ''
        branch_path = getattr(step, 'branch_path', []) or []
        substep_idx = getattr(step, 'substep_index', None)

        # Use branch_path to derive branch context if branch_name is empty
        if not branch_name and branch_path:
            # Format as "B0", "B1", etc. for compact display
            branch_name = f"B{branch_path[0]}"

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
            class_label = operator_class
            if len(class_label) > 25:
                class_label = class_label[:22] + "..."

            # For branch substeps, prepend abbreviated branch name
            if branch_name and substep_idx is not None:
                # Abbreviate branch name for compact display
                short_branch = branch_name[:12] + ".." if len(branch_name) > 14 else branch_name
                return f"[{short_branch}] {class_label}"

            return class_label

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
            # For branch substeps with generic type, still show branch context
            if branch_name and substep_idx is not None:
                short_branch = branch_name[:12] + ".." if len(branch_name) > 14 else branch_name
                return f"[{short_branch}] {label}"
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
            features_shape=[current_shape],
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
            if node_type in ('branch', 'source_branch'):
                # Create branch node
                branch_node = PipelineNode(
                    id=f"step_{step_index}_branch",
                    step_index=step_index,
                    label=label,
                    node_type=node_type,
                    shape_before=current_shape,
                    shape_after=current_shape,
                    features_shape=[current_shape],
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
                        node_type=node_type,
                        branch_id=branch_id,
                        branch_name=entry_label,
                        shape_before=current_shape,
                        shape_after=current_shape,
                        features_shape=[current_shape],
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
                            new_substep_shape = self._estimate_shape_after(branch_shape, substep_info)

                            sub_metadata = {}
                            if substep_info.get('n_splits'):
                                sub_metadata['n_splits'] = substep_info['n_splits']

                            substep_node = PipelineNode(
                                id=substep_id,
                                step_index=step_index,
                                label=substep_info['label'],
                                node_type=substep_info['type'],
                                branch_id=branch_id,
                                branch_name=entry_label,
                                substep_index=substep_idx,
                                shape_before=branch_shape,
                                shape_after=new_substep_shape,
                                features_shape=[new_substep_shape],
                                parent_ids=branch_current.copy(),
                                metadata=sub_metadata,
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

            elif node_type in ('merge', 'merge_sources'):
                # Create merge node
                new_merge_shape = self._estimate_merge_shape(current_shape, step_info)
                merge_node = PipelineNode(
                    id=f"step_{step_index}_merge",
                    step_index=step_index,
                    label=label,
                    node_type=node_type,
                    shape_before=current_shape,
                    shape_after=new_merge_shape,
                    features_shape=[new_merge_shape],
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

                metadata = {'keyword': keyword} if keyword else {}
                if step_info.get('n_splits'):
                    metadata['n_splits'] = step_info['n_splits']

                node = PipelineNode(
                    id=node_id,
                    step_index=step_index,
                    label=label,
                    node_type=node_type,
                    shape_before=current_shape,
                    shape_after=new_shape,
                    features_shape=[new_shape],
                    parent_ids=current_node_ids.copy(),
                    metadata=metadata,
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
            info = self._classify_operator(class_name, {})
            if info['type'] == 'splitter':
                n_splits = getattr(step, 'n_splits', None)
                if n_splits:
                    info['n_splits'] = n_splits
            return info

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
            op_count = 1
            if isinstance(value, list):
                op_count = len(value)
                ops = [self._get_operator_name(v) for v in value[:3]]
                label = "FA: " + ", ".join(ops)
                if len(value) > 3:
                    label += f"... (+{len(value)-3})"
            else:
                label = f"FA: {self._get_operator_name(value)}"
            action = step.get('action', 'add')
            return {
                'type': 'feature_augmentation',
                'label': label,
                'action': action,
                'keyword': keyword,
                'op_count': op_count
            }

        elif keyword == 'sample_augmentation':
            aug_count = 1
            if isinstance(value, dict):
                transformers = value.get('transformers', [])
                count = value.get('count', 1)
                # Handle 'count' being a string '?' or similar
                try:
                    aug_count = int(count)
                except (ValueError, TypeError):
                    aug_count = 1

                label = f"SA: {len(transformers)} aug ×{count}"
            else:
                label = "Sample Aug"
            return {
                'type': 'sample_augmentation',
                'label': label,
                'keyword': keyword,
                'aug_count': aug_count
            }

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
            branches = {}
            if isinstance(value, dict):
                for k, v in value.items():
                    branches[str(k)] = v if isinstance(v, list) else [v]
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    branches[f"Source {i}"] = v if isinstance(v, list) else [v]
            return {'type': 'source_branch', 'label': 'Source Branch', 'branches': branches, 'keyword': keyword}

        elif keyword == 'merge_sources':
            strategy = value if isinstance(value, str) else 'concat'
            return {'type': 'merge_sources', 'label': f"Merge Sources ({strategy})", 'keyword': keyword}

        elif keyword == 'model':
            model_name = step.get('name', self._get_operator_name(value))
            return {'type': 'model', 'label': model_name, 'keyword': keyword}

        elif keyword == 'split':
            splitter = value
            splitter_name = self._get_operator_name(splitter)
            n_splits = getattr(splitter, 'n_splits', None)
            return {'type': 'splitter', 'label': splitter_name, 'keyword': keyword, 'n_splits': n_splits}

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
            op_count = step_info.get('op_count', 1)

            if action == 'extend':
                # Adds new processings
                processings += op_count
            elif action == 'replace':
                # Replaces processings
                processings = op_count
            else:  # add
                # Multiplies processings (each existing processing gets N new versions)
                processings *= op_count

        elif step_type == 'sample_augmentation':
            # Sample augmentation adds samples
            aug_count = step_info.get('aug_count', 1)
            # Usually adds N augmented samples per original sample
            # So total = original + (original * aug_count)
            samples = samples + (samples * aug_count)

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

    def _estimate_node_width(self, node: PipelineNode) -> float:
        """Estimate node width based on label length."""
        label_lines = [node.label]
        if self._show_shapes:
            label_lines.extend(self._format_shape_display(node))

        # Add score line estimate
        if node.metadata.get('best_score') is not None and node.node_type == 'model':
             label_lines.append("★ 0.00")

        max_text_len = max([len(l) for l in label_lines]) if label_lines else 0
        # Increased factor from 0.16 to 0.22 for better fit
        calc_width = 2.0 + max_text_len * 0.22
        return max(self._node_width, calc_width)

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

        # Calculate dynamic spacing based on max node width
        max_width = 0
        for node in self.nodes.values():
            w = self._estimate_node_width(node)
            if w > max_width:
                max_width = w

        # Dynamic spacing with minimum
        x_spacing = max(2.8, max_width + 0.5)
        y_spacing = 1.8

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
        - For single source: S×P×F (samples × processings × features)
        - For multi-source: Source count + total features
        - Always shows 2D layout shape when available
        - Model scores when available

        Args:
            node: The pipeline node with shape info

        Returns:
            List of formatted shape strings
        """
        shape_lines = []

        show_features_shape = True
        # For model nodes, prefer concise 2D shape (stacked features/preds) to avoid misleading 3D shapes
        if node.node_type == 'model':
            show_features_shape = False

        # Format: (samples, [p, f], [p, f]...)
        if show_features_shape and node.features_shape:
            parts = []
            n_samples = node.features_shape[0][0]
            parts.append(str(n_samples))

            for s, p, f in node.features_shape:
                parts.append(f"[{p}, {f}]")

            shape_str = f"({', '.join(parts)})"
            shape_lines.append(shape_str)

        # Total 2D: (samples, features)
        if node.output_layout_shape:
            s, f = node.output_layout_shape
            shape_lines.append(f"2D: ({s}, {f})")
        elif node.input_layout_shape and not shape_lines:
            # Fallback to input shape if output not available
            s, f = node.input_layout_shape
            shape_lines.append(f"2D: ({s}, {f})")
        elif node.features_shape:
            # Compute 2D from features_shape as a last resort
            n_samples = node.features_shape[0][0]
            total_features = sum(p * f for (_, p, f) in node.features_shape)
            shape_lines.append(f"2D: ({n_samples}, {total_features})")

        # Fallback if no trace info but we have estimated shape
        if not shape_lines and node.shape_after:
            s, p, f = node.shape_after
            shape_lines.append(f"({s}, [{p}, {f}])")

        # Final fallback: show input shape if we have nothing
        if not shape_lines and node.shape_before:
            s, p, f = node.shape_before
            shape_lines.append(f"in: ({s}, [{p}, {f}])")

        # Score is displayed separately in _draw_nodes for model nodes
        # Only show score in shape lines for non-model nodes
        # if node.node_type != 'model' and node.metadata:
        #     best_score = node.metadata.get('best_score')
        #     if best_score is not None:
        #         if isinstance(best_score, float):
        #             shape_lines.append(f"score: {best_score:.4f}")
        #         else:
        #             shape_lines.append(f"score: {best_score}")

        # Display fold info for splitters
        if node.node_type == 'splitter':
            n_splits = node.metadata.get('n_splits')
            if n_splits:
                shape_lines.append(f"{n_splits} folds")

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

            # Get style
            fill_color, border_color = self.NODE_STYLES.get(node.node_type, self.NODE_STYLES['default'])

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

            # Calculate width based on text length
            max_text_len = max([len(l) for l in label_lines]) if label_lines else 0
            # Base width + char width factor
            calc_width = 2.0 + max_text_len * 0.22
            box_width = max(self._node_width, calc_width)

            # Adjust height for multi-line
            line_height = 0.35
            box_height = self._node_height + (n_lines - 1) * line_height

            # Store dimensions for edge drawing
            pos_info['width'] = box_width
            pos_info['height'] = box_height

            # Draw shadow (offset)
            shadow_offset = 0.05
            shadow = FancyBboxPatch(
                (x - box_width / 2 + shadow_offset, y - box_height / 2 - shadow_offset),
                box_width, box_height,
                boxstyle="round,pad=0.1,rounding_size=0.2",
                facecolor='#000000',
                edgecolor='none',
                alpha=0.1,
                zorder=1
            )
            ax.add_patch(shadow)

            # Draw node box
            rect = FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width, box_height,
                boxstyle="round,pad=0.1,rounding_size=0.2",
                facecolor=fill_color,
                edgecolor=border_color,
                linewidth=1.5,
                alpha=1.0,
                zorder=2
            )
            ax.add_patch(rect)

            # Calculate vertical positions for text lines
            start_y = y + (n_lines - 1) * line_height / 2

            # Draw operator label (first line)
            ax.text(
                x, start_y,
                node.label,
                ha='center', va='center',
                fontsize=self._fontsize,
                color='#263238',  # Dark Blue Grey
                fontweight='bold',
                zorder=3
            )

            # Draw additional info (shapes, score)
            if len(label_lines) > 1:
                for i, line in enumerate(label_lines[1:], 1):
                    line_y = start_y - i * line_height
                    # Use different style for score line
                    if line.startswith('★'):
                        ax.text(
                            x, line_y,
                            line,
                            ha='center', va='center',
                            fontsize=self._fontsize,
                            color='#D32F2F',  # Red
                            fontweight='bold',
                            zorder=3
                        )
                    else:
                        ax.text(
                            x, line_y,
                            line,
                            ha='center', va='center',
                            fontsize=self._fontsize - 1.5,
                            color='#546E7A',  # Blue Grey
                            fontfamily='monospace',
                            fontweight='normal',
                            zorder=3
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

            # Determine curve based on horizontal offset
            dx = to_x - from_x
            rad = 0.0
            if abs(dx) > 0.1:
                # Curve slightly for non-vertical connections
                rad = 0.1 if dx > 0 else -0.1

            # Draw arrow
            ax.annotate(
                '',
                xy=(to_x, to_y),
                xytext=(from_x, from_y),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='#546E7A',  # Blue Grey
                    linewidth=1.5,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle=f'arc3,rad={rad}',
                    mutation_scale=15,
                ),
                zorder=0
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
            ('Input/Output', self.NODE_STYLES['input']),
            ('Preprocessing', self.NODE_STYLES['preprocessing']),
            ('Feature Aug', self.NODE_STYLES['feature_augmentation']),
            ('Sample Aug', self.NODE_STYLES['sample_augmentation']),
            ('Y Processing', self.NODE_STYLES['y_processing']),
            ('Splitter', self.NODE_STYLES['splitter']),
            ('Branch/Merge', self.NODE_STYLES['branch']),
            ('Model', self.NODE_STYLES['model']),
        ]

        patches = []
        for label, (fill, border) in legend_items:
            patch = mpatches.Patch(
                facecolor=fill,
                edgecolor=border,
                label=label,
                linewidth=1.0
            )
            patches.append(patch)

        ax.legend(
            handles=patches,
            loc='upper right',
            fontsize=self._fontsize - 1,
            framealpha=0.95,
            edgecolor='#CFD8DC',
            ncol=2,
        )


def plot_pipeline_diagram(
    pipeline_steps: Optional[List[Any]] = None,
    predictions: Any = None,
    show_shapes: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    initial_shape: Optional[Tuple[int, int, int]] = None,
    config: Optional[Dict[str, Any]] = None,
    execution_trace: Any = None
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
        execution_trace: Optional ExecutionTrace object

    Returns:
        matplotlib Figure object

    Example:
        >>> from nirs4all.visualization.pipeline_diagram import plot_pipeline_diagram
        >>> fig = plot_pipeline_diagram(pipeline, initial_shape=(189, 1, 2151))
        >>> fig.savefig('pipeline_diagram.png')
    """
    cfg = config or {}
    diagram = PipelineDiagram(pipeline_steps, predictions, execution_trace=execution_trace, config=cfg)
    if execution_trace:
        diagram._build_dag_from_trace()

    return diagram.render(
        show_shapes=show_shapes,
        figsize=figsize,
        title=title,
        initial_shape=initial_shape,
    )


# Backward compatibility alias
BranchDiagram = PipelineDiagram
plot_branch_diagram = plot_pipeline_diagram

"""
Branch Controller for pipeline branching.

This controller enables splitting a pipeline into multiple parallel sub-pipelines
("branches"), each with its own preprocessing context (X transformations, Y processing),
while sharing common upstream state (splits, initial preprocessing).

Steps declared after a branch block execute on each branch independently.

Example:
    >>> pipeline = [
    ...     ShuffleSplit(n_splits=5),
    ...     {"branch": [
    ...         [SNV(), PCA(n_components=10)],
    ...         [MSC(), FirstDerivative()],
    ...     ]},
    ...     PLSRegression(n_components=5),  # Runs on BOTH branches
    ... ]

Generator syntax is also supported:
    >>> pipeline = [
    ...     ShuffleSplit(n_splits=3),
    ...     {"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}},  # 3 branches
    ...     PLSRegression(n_components=5),
    ... ]
"""

import copy
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.pipeline.config.generator import (
    expand_spec,
    is_generator_node,
)
from nirs4all.pipeline.execution.result import StepOutput
from nirs4all.utils.emoji import BRANCH, CHECK, CROSS

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class BranchController(OperatorController):
    """Controller for pipeline branching.

    Implements the branching mechanism that allows multiple preprocessing
    chains to be evaluated independently within a single pipeline execution.

    Key behaviors:
        - Creates independent context copies for each branch
        - Executes branch steps sequentially within each branch
        - Stores branch contexts in context.custom["branch_contexts"]
        - Post-branch steps iterate over all branch contexts

    Attributes:
        priority: Controller priority (lower = higher priority). Set to 5
            to execute before most other controllers.
    """

    priority = 5  # High priority to catch branch keyword early

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the branch controller.

        Args:
            step: Original step configuration
            operator: Deserialized operator (may be list of branch definitions)
            keyword: Step keyword

        Returns:
            True if keyword is "branch"
        """
        return keyword == "branch"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Branch controller supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Branch controller should execute in prediction mode to reconstruct branches."""
        return True

    def execute(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute the branch step.

        Creates independent contexts for each branch, executes branch-specific
        steps, and stores branch contexts for post-branch iteration.

        In predict/explain mode, only executes the target branch specified in
        runtime_context.target_model.branch_id for efficiency.

        Args:
            step_info: Parsed step containing branch definitions
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput with collected artifacts)
        """
        # Get branch definitions from step
        branch_defs = self._parse_branch_definitions(step_info)

        if not branch_defs:
            print(f"{CROSS} No branch definitions found, skipping branch step")
            return context, StepOutput()

        n_branches = len(branch_defs)

        # In predict/explain mode, filter to only the target branch
        target_branch_id = None
        target_branch_name = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")
            target_branch_name = runtime_context.target_model.get("branch_name")

            if target_branch_id is not None:
                # Filter branch_defs to only the target branch
                if target_branch_id < len(branch_defs):
                    branch_defs = [branch_defs[target_branch_id]]
                    print(f"{BRANCH}Predict mode: executing only branch {target_branch_id} ({target_branch_name or 'unnamed'})")
                else:
                    raise ValueError(
                        f"Target branch_id={target_branch_id} not found in pipeline. "
                        f"Pipeline has {n_branches} branches (0-{n_branches-1}). "
                        f"The model may have been trained with a different branch configuration."
                    )
            else:
                print(f"{BRANCH}Creating {n_branches} branches (predict mode, no target branch specified)")
        else:
            print(f"{BRANCH}Creating {n_branches} branches")

        # Store the initial context as a snapshot point
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Snapshot the dataset's feature state before branching
        # This is necessary because branches modify the shared dataset
        initial_features_snapshot = self._snapshot_features(dataset)

        # Initialize list to collect branch contexts
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []

        # In predict mode, we need to set the operation counter to match training.
        # Each branch should start at the operation count it had during training.
        # We calculate this by counting operations in prior branches.
        initial_op_count = runtime_context.operation_count

        # Execute each branch
        # In predict mode with filtered branches, we need to preserve original branch_id
        for idx, branch_def in enumerate(branch_defs):
            # Use original branch_id if we're in predict mode and have filtered
            if target_branch_id is not None:
                branch_id = target_branch_id
                # In predict mode for a specific branch, set operation counter to match training
                # Each processing in a branch increments the op counter by 1
                # We need to advance the counter as if prior branches had run
                if mode in ("predict", "explain"):
                    # Advance operation counter for skipped branches
                    # Each branch with one transformer step uses 1 operation per processing
                    # For branch N, we need initial_op_count + N operations from prior branches
                    runtime_context.operation_count = initial_op_count + branch_id
            else:
                branch_id = idx

            branch_name = branch_def.get("name", f"branch_{branch_id}")
            branch_steps = branch_def.get("steps", [])

            print(f"  {BRANCH}Branch {branch_id}: {branch_name}")

            # Create isolated context for this branch
            branch_context = initial_context.copy()

            # Build branch_path by appending to parent's branch_path
            parent_branch_path = context.selector.branch_path or []
            new_branch_path = parent_branch_path + [branch_id]

            branch_context.selector = branch_context.selector.with_branch(
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=new_branch_path
            )

            # Reset processing to initial state for this branch
            branch_context.selector.processing = copy.deepcopy(initial_processing)

            # Restore dataset features to initial state for this branch
            # This ensures each branch starts from the same feature state
            self._restore_features(dataset, initial_features_snapshot)

            # In predict/explain mode, load branch-specific binaries
            branch_binaries = loaded_binaries
            if mode in ("predict", "explain") and runtime_context.artifact_loader:
                branch_binaries = runtime_context.artifact_loader.get_step_binaries(
                    runtime_context.step_number, branch_id=branch_id
                )
                if not branch_binaries:
                    # Fall back to non-branch binaries if no branch-specific ones exist
                    branch_binaries = loaded_binaries

            # Execute branch steps sequentially
            for substep in branch_steps:
                if runtime_context.step_runner:
                    runtime_context.substep_number += 1
                    result = runtime_context.step_runner.execute(
                        step=substep,
                        dataset=dataset,
                        context=branch_context,
                        runtime_context=runtime_context,
                        loaded_binaries=branch_binaries,
                        prediction_store=prediction_store
                    )
                    branch_context = result.updated_context
                    all_artifacts.extend(result.artifacts)

            # Snapshot features AFTER branch processing completes
            # This captures the feature state produced by this branch's transformers
            # Post-branch steps (e.g., model) need this to use correct features per branch
            branch_features_snapshot = self._snapshot_features(dataset)

            # Store the final context for this branch
            branch_contexts.append({
                "branch_id": branch_id,
                "name": branch_name,
                "context": branch_context,
                "generator_choice": branch_def.get("generator_choice"),
                "features_snapshot": branch_features_snapshot
            })

            print(f"  {CHECK} Branch {branch_id} ({branch_name}) completed")

        # Store branch contexts in custom dict for post-branch iteration
        # Merge with any existing branch contexts (for nested branches)
        existing_branches = context.custom.get("branch_contexts", [])
        if existing_branches:
            # Nested branching: multiply existing branches with new ones
            new_branch_contexts = self._multiply_branch_contexts(
                existing_branches, branch_contexts
            )
        else:
            new_branch_contexts = branch_contexts

        # Update context with branch contexts
        # Use the last branch's context as the "current" context
        # but store all contexts for post-branch iteration
        result_context = context.copy()
        result_context.custom["branch_contexts"] = new_branch_contexts

        # Mark that we are in branching mode
        result_context.custom["in_branch_mode"] = True

        # Collect generator choices from branches for serialization
        branch_generator_choices = [
            {"branch": branch_def.get("generator_choice")}
            for branch_def in branch_defs
            if branch_def.get("generator_choice") is not None
        ]

        print(f"{CHECK} Branch step completed with {len(new_branch_contexts)} branch(es)")

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata={
                "branch_count": len(new_branch_contexts),
                "branch_generator_choices": branch_generator_choices
            }
        )

    def _parse_branch_definitions(
        self,
        step_info: "ParsedStep"
    ) -> List[Dict[str, Any]]:
        """Parse branch definitions from step configuration.

        Supports multiple syntaxes:
            - List of lists: [[step1, step2], [step3]]
            - Dict with names: {"snv_pca": [SNV(), PCA()], "msc": [MSC()]}
            - List of dicts: [{"name": "a", "steps": [...]}, ...]
            - Generator syntax: {"_or_": [SNV(), MSC()]} or {"_range_": [...]}

        Args:
            step_info: Parsed step containing branch definitions

        Returns:
            Normalized list of branch definitions with 'name' and 'steps'
        """
        # Get the raw branch definition from original step
        raw_def = step_info.original_step.get("branch", [])

        if not raw_def:
            return []

        # Case 0: Generator syntax - expand before processing
        if isinstance(raw_def, dict) and is_generator_node(raw_def):
            return self._expand_generator_branches(raw_def)

        # Case 1: Dict with named branches {"name": [steps], ...}
        if isinstance(raw_def, dict):
            # Check if any value contains generator syntax
            expanded_branches = []
            for name, steps in raw_def.items():
                if isinstance(steps, dict) and is_generator_node(steps):
                    # Expand generator within named branch
                    expanded_steps = expand_spec(steps)
                    for i, exp_step in enumerate(expanded_steps):
                        branch_name = f"{name}_{self._generate_step_name(exp_step, i)}"
                        expanded_branches.append({
                            "name": branch_name,
                            "steps": exp_step if isinstance(exp_step, list) else [exp_step],
                            "generator_choice": exp_step
                        })
                else:
                    expanded_branches.append({
                        "name": name,
                        "steps": steps if isinstance(steps, list) else [steps]
                    })
            return expanded_branches

        # Case 2: List format
        if isinstance(raw_def, list):
            result = []
            for i, item in enumerate(raw_def):
                # Sub-case 2a: Dict with explicit name and steps
                if isinstance(item, dict) and "steps" in item:
                    steps = item["steps"]
                    # Check for generator in steps
                    if isinstance(steps, dict) and is_generator_node(steps):
                        expanded_steps = expand_spec(steps)
                        for j, exp_step in enumerate(expanded_steps):
                            result.append({
                                "name": f"{item.get('name', f'branch_{i}')}_{j}",
                                "steps": exp_step if isinstance(exp_step, list) else [exp_step],
                                "generator_choice": exp_step
                            })
                    else:
                        result.append({
                            "name": item.get("name", f"branch_{i}"),
                            "steps": steps
                        })
                # Sub-case 2b: Dict with generator syntax (inside list)
                elif isinstance(item, dict) and is_generator_node(item):
                    expanded = expand_spec(item)
                    for j, exp_item in enumerate(expanded):
                        branch_name = self._generate_step_name(exp_item, i * 100 + j)
                        result.append({
                            "name": branch_name,
                            "steps": exp_item if isinstance(exp_item, list) else [exp_item],
                            "generator_choice": exp_item
                        })
                # Sub-case 2c: List of steps (anonymous branch)
                elif isinstance(item, list):
                    # Check if the list contains generator nodes
                    expanded_list = self._expand_list_with_generators(item)
                    if len(expanded_list) > 1:
                        for j, exp_item in enumerate(expanded_list):
                            result.append({
                                "name": f"branch_{i}_{j}",
                                "steps": exp_item,
                                "generator_choice": exp_item
                            })
                    else:
                        result.append({
                            "name": f"branch_{i}",
                            "steps": expanded_list[0] if expanded_list else item
                        })
                # Sub-case 2d: Single step (wrap in list)
                else:
                    result.append({
                        "name": f"branch_{i}",
                        "steps": [item]
                    })
            return result

        # Fallback: treat as single branch with single step
        return [{"name": "branch_0", "steps": [raw_def]}]

    def _expand_generator_branches(
        self,
        generator_node: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Expand a generator node into branch definitions.

        Handles:
            - {"_or_": [SNV(), MSC(), D1()]} -> 3 branches
            - {"_or_": [[SNV(), PCA()], [MSC()]]} -> 2 branches with multi-step
            - {"_range_": [5, 15, 5], "param": "n_components", "model": PLS} -> 3 branches

        Args:
            generator_node: Dict with generator keywords (_or_, _range_, etc.)

        Returns:
            List of branch definitions with 'name' and 'steps'
        """
        expanded = expand_spec(generator_node)

        result = []
        for i, item in enumerate(expanded):
            branch_name = self._generate_step_name(item, i)
            # Ensure steps is always a list
            if isinstance(item, list):
                steps = item
            else:
                steps = [item]

            result.append({
                "name": branch_name,
                "steps": steps,
                "generator_choice": item
            })

        return result

    def _expand_list_with_generators(
        self,
        items: List[Any]
    ) -> List[List[Any]]:
        """Expand a list that may contain generator nodes.

        Each generator node in the list is expanded, and the Cartesian product
        of all expansions is returned.

        Args:
            items: List of steps, some of which may be generator nodes

        Returns:
            List of expanded step lists
        """
        from itertools import product

        expanded_items = []
        for item in items:
            if isinstance(item, dict) and is_generator_node(item):
                expanded_items.append(expand_spec(item))
            else:
                expanded_items.append([item])

        # Compute Cartesian product
        result = []
        for combo in product(*expanded_items):
            result.append(list(combo))

        return result if result else [items]

    def _generate_step_name(
        self,
        step: Any,
        index: int
    ) -> str:
        """Generate a human-readable name for a step or list of steps.

        Args:
            step: A step dict, class instance, or list of steps
            index: Fallback index if name cannot be extracted

        Returns:
            A descriptive branch name
        """
        if isinstance(step, list):
            # Multiple steps - combine names
            names = [self._get_single_step_name(s) for s in step]
            # Filter out None/empty names
            names = [n for n in names if n]
            if names:
                return "_".join(names[:3])  # Limit to first 3 steps
            return f"branch_{index}"

        return self._get_single_step_name(step) or f"branch_{index}"

    def _get_single_step_name(
        self,
        step: Any
    ) -> Optional[str]:
        """Extract a short name from a single step.

        Args:
            step: A step configuration (dict, string, etc.)

        Returns:
            Short name or None if not extractable
        """
        if step is None:
            return None

        if isinstance(step, str):
            return step

        if isinstance(step, dict):
            # Check for 'name' key first (explicit naming)
            if "name" in step:
                return step["name"]

            # Check for 'class' key (serialized format)
            if "class" in step:
                class_name = step["class"]
                # Extract short class name from full path
                if isinstance(class_name, str) and "." in class_name:
                    return class_name.split(".")[-1]
                return str(class_name).split(".")[-1].replace("'>", "")

            # Check for model key
            if "model" in step:
                return self._get_single_step_name(step["model"])

            # Check for preprocessing key
            if "preprocessing" in step:
                return self._get_single_step_name(step["preprocessing"])

            # Use first key as hint
            keys = [k for k in step.keys() if not k.startswith("_")]
            if keys:
                return keys[0]

        # For class instances, try to get class name
        if hasattr(step, "__class__"):
            return step.__class__.__name__

        return None

    def _multiply_branch_contexts(
        self,
        existing: List[Dict[str, Any]],
        new: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Multiply existing branch contexts with new branches (for nesting).

        Creates Cartesian product of branch contexts for nested branching.
        Uses branch_path for tracking the full hierarchy.

        Args:
            existing: List of existing branch context dicts
            new: List of new branch context dicts

        Returns:
            Combined list of branch contexts with hierarchical branch_path
        """
        result = []
        flattened_id = 0

        for parent in existing:
            parent_id = parent["branch_id"]
            parent_name = parent["name"]
            parent_context = parent["context"]
            parent_branch_path = parent_context.selector.branch_path or [parent_id]

            for child in new:
                child_id = child["branch_id"]
                child_name = child["name"]
                child_context = child["context"]

                # Create combined context
                combined_context = child_context.copy()

                # Build nested branch_path: parent_path + child_id
                combined_branch_path = parent_branch_path + [child_id]

                combined_context.selector = combined_context.selector.with_branch(
                    branch_id=flattened_id,  # Keep flattened ID for backward compat
                    branch_name=f"{parent_name}_{child_name}",
                    branch_path=combined_branch_path
                )

                result.append({
                    "branch_id": flattened_id,
                    "name": f"{parent_name}_{child_name}",
                    "context": combined_context,
                    "parent_branch_id": parent_id,
                    "child_branch_id": child_id,
                    "branch_path": combined_branch_path
                })
                flattened_id += 1

        return result

    def _snapshot_features(self, dataset: "SpectroDataset") -> List[Any]:
        """Create a deep copy of the dataset's feature sources.

        This is used to restore features to their initial state before each branch,
        ensuring branches operate on independent copies of the data.

        Args:
            dataset: The dataset to snapshot

        Returns:
            A deep copy of the feature sources list
        """
        return copy.deepcopy(dataset._features.sources)

    def _restore_features(
        self,
        dataset: "SpectroDataset",
        snapshot: List[Any]
    ) -> None:
        """Restore the dataset's feature sources from a snapshot.

        Args:
            dataset: The dataset to restore
            snapshot: The previously snapshotted feature sources
        """
        dataset._features.sources = copy.deepcopy(snapshot)

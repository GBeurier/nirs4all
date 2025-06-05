import numpy as np
from typing import Any, Dict, List, Union, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ScopeState:
    """Represents a pipeline scope state that can be pushed/popped."""
    filters: Dict[str, Any]
    branch: int
    processing_level: int
    augmentation_level: int
    source_config: Optional[Dict[str, Any]] = None


class PipelineContext:
    """
    Enhanced context object to track complex pipeline state and manage scoping.

    Handles:
    - Nested scoping with push/pop semantics
    - Complex filtering and data selection
    - Branch management and state tracking
    - Processing history and augmentation tracking
    - Source management (split, merge, dispatch)
    - Centroid and clustering context
    """

    def __init__(self):
        # Core state
        self.current_filters: Dict[str, Any] = {}
        self.predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Scope stack for nested contexts
        self.scope_stack: List[ScopeState] = []

        # Branch management
        self.branch_stack: List[int] = [0]
        self.branch_data_cache: Dict[int, Any] = {}  # Cache for branch-specific data

        # Processing and augmentation tracking
        self.processing_history: List[Dict[str, Any]] = []
        self.current_processing_level = 0
        self.current_augmentation_level = 0

        # Source management
        self.source_stack: List[Dict[str, Any]] = []
        self.active_sources: Optional[List[int]] = None  # None = all sources
        self.source_merge_mode = False

        # Clustering and centroid context
        self.centroid_mode = False
        self.centroid_groups: Dict[int, List[int]] = {}  # group_id -> [sample_ids]
        self.group_centroids: Dict[int, int] = {}  # group_id -> centroid_sample_id

        # Advanced state
        self.fold_context: Optional[Dict[str, Any]] = None
        self.stacking_context: Optional[Dict[str, Any]] = None
        self.dispatch_mode = False

        # Performance optimization
        self._filter_cache: Dict[str, Any] = {}
        self._cache_dirty = True

    def push_scope(self, **new_filters) -> ScopeState:
        """
        Push a new scope with additional filters.
        Returns the previous state for restoration.
        """
        # Save current state
        current_state = ScopeState(
            filters=self.current_filters.copy(),
            branch=self.current_branch,
            processing_level=self.current_processing_level,
            augmentation_level=self.current_augmentation_level,
            source_config=self._get_current_source_config()
        )

        self.scope_stack.append(current_state)

        # Apply new filters
        self.current_filters.update(new_filters)
        self._cache_dirty = True

        return current_state

    def pop_scope(self) -> Optional[ScopeState]:
        """Restore previous scope state."""
        if not self.scope_stack:
            return None

        previous_state = self.scope_stack.pop()

        # Restore state
        self.current_filters = previous_state.filters
        self.branch_stack[-1] = previous_state.branch
        self.current_processing_level = previous_state.processing_level
        self.current_augmentation_level = previous_state.augmentation_level

        if previous_state.source_config:
            self._restore_source_config(previous_state.source_config)

        self._cache_dirty = True
        return previous_state

    def push_filters(self, **filters) -> Dict[str, Any]:
        """Add temporary filters (lightweight version of push_scope)."""
        old_filters = self.current_filters.copy()
        self.current_filters.update(filters)
        self._cache_dirty = True
        return old_filters

    def pop_filters(self, old_filters: Dict[str, Any]):
        """Restore previous filters."""
        self.current_filters = old_filters
        self._cache_dirty = True

    def apply_filters(self, filters: Dict[str, Any]):
        """Apply context filters - update current filters with new ones."""
        self.current_filters.update(filters)
        self._cache_dirty = True

    # Branch management
    @property
    def current_branch(self) -> int:
        return self.branch_stack[-1]

    def push_branch(self, branch: int):
        """Push a new branch context."""
        self.branch_stack.append(branch)

        # Add branch filter
        old_branch = self.current_filters.get("branch")
        self.current_filters["branch"] = branch
        self._cache_dirty = True

        return old_branch

    def pop_branch(self) -> Optional[int]:
        """Pop branch context."""
        if len(self.branch_stack) > 1:
            old_branch = self.branch_stack.pop()

            # Update branch filter
            self.current_filters["branch"] = self.current_branch
            self._cache_dirty = True

            return old_branch
        return None

    def create_branches(self, branch_count: int) -> List[int]:
        """Create multiple branches and return branch IDs."""
        branches = []
        for i in range(branch_count):
            branch_id = self.current_branch * 1000 + i  # Hierarchical branch IDs
            branches.append(branch_id)
        return branches

    # Cluster management
    def push_cluster(self, cluster_config: Dict[str, Any]):
        """
        Push a new cluster context.

        Args:
            cluster_config: Configuration for cluster context
                - cluster_id: Identifier for the cluster
                - cluster_filters: Filters to apply for this cluster
                - cluster_operation: Operation type (fit, predict, etc.)
        """
        current_state = ScopeState(
            filters=self.current_filters.copy(),
            branch=self.current_branch,
            processing_level=self.current_processing_level,
            augmentation_level=self.current_augmentation_level,
            source_config={
                "active_sources": self.active_sources,
                "merge_mode": self.source_merge_mode
            }
        )
        self.scope_stack.append(current_state)

        # Apply cluster-specific filters
        cluster_filters = cluster_config.get("cluster_filters", {})
        self.current_filters.update(cluster_filters)

        # Track cluster in processing history
        self.processing_history.append({
            "action": "push_cluster",
            "cluster_config": cluster_config,
            "timestamp": datetime.now(),
            "branch": self.current_branch,
            "level": self.current_processing_level
        })

    def pop_cluster(self):
        """
        Pop the current cluster context and restore previous state.

        Returns:
            Dict[str, Any]: The cluster state that was popped
        """
        if not self.scope_stack:
            raise RuntimeError("Cannot pop cluster context: no cluster contexts on stack")

        # Get the last processing history entry for this cluster
        cluster_state = None
        for entry in reversed(self.processing_history):
            if entry.get("action") == "push_cluster":
                cluster_state = entry.get("cluster_config", {})
                break

        # Restore previous state
        previous_state = self.scope_stack.pop()
        self.current_filters = previous_state.filters
        self.current_branch = previous_state.branch
        self.current_processing_level = previous_state.processing_level
        self.current_augmentation_level = previous_state.augmentation_level

        if previous_state.source_config:
            self._restore_source_config(previous_state.source_config)

        # Track cluster pop in history
        self.processing_history.append({
            "action": "pop_cluster",
            "timestamp": datetime.now(),
            "branch": self.current_branch,
            "level": self.current_processing_level
        })

        return cluster_state or {}

    # Source management
    def push_source_split(self, source_indices: List[int]):
        """Split pipeline by sources - each source becomes a branch."""
        source_config = {
            "type": "split",
            "active_sources": self.active_sources,
            "source_indices": source_indices,
            "merge_mode": self.source_merge_mode
        }
        self.source_stack.append(source_config)

        # Set active sources
        self.active_sources = source_indices
        self.source_merge_mode = False

    def push_source_merge(self):
        """Merge all sources for subsequent operations."""
        source_config = {
            "type": "merge",
            "active_sources": self.active_sources,
            "merge_mode": self.source_merge_mode
        }
        self.source_stack.append(source_config)

        self.source_merge_mode = True
        self.active_sources = None  # All sources

    def pop_source_context(self):
        """Restore previous source context."""
        if self.source_stack:
            config = self.source_stack.pop()
            self.active_sources = config["active_sources"]
            self.source_merge_mode = config["merge_mode"]

    def _get_current_source_config(self) -> Dict[str, Any]:
        """Get current source configuration."""
        return {
            "active_sources": self.active_sources,
            "merge_mode": self.source_merge_mode
        }

    def _restore_source_config(self, config: Dict[str, Any]):
        """Restore source configuration."""
        self.active_sources = config["active_sources"]
        self.source_merge_mode = config["merge_mode"]

    # Clustering and centroids
    def enable_centroid_mode(self, centroid_groups: Dict[int, List[int]],
                           group_centroids: Dict[int, int]):
        """Enable centroid-based operations."""
        self.centroid_mode = True
        self.centroid_groups = centroid_groups
        self.group_centroids = group_centroids

    def disable_centroid_mode(self):
        """Disable centroid mode."""
        self.centroid_mode = False
        self.centroid_groups = {}
        self.group_centroids = {}

    def propagate_to_groups(self, centroid_operation_result: Dict[int, Any]) -> Dict[int, List[int]]:
        """
        Propagate centroid operation results to group members.

        Args:
            centroid_operation_result: {centroid_sample_id: result}

        Returns:
            {group_id: [affected_sample_ids]}
        """
        if not self.centroid_mode:
            return {}

        group_propagation = {}
        for group_id, centroid_id in self.group_centroids.items():
            if centroid_id in centroid_operation_result:
                group_propagation[group_id] = self.centroid_groups.get(group_id, [])

        return group_propagation

    # Processing and augmentation tracking
    def increment_processing_level(self, operation_hash: str):
        """Track processing level changes."""
        self.current_processing_level += 1
        self.processing_history.append({
            "level": self.current_processing_level,
            "operation_hash": operation_hash,
            "timestamp": datetime.now(),
            "branch": self.current_branch,
            "filters": self.current_filters.copy()
        })

    def increment_augmentation_level(self, augmentation_type: str):
        """Track augmentation level changes."""
        self.current_augmentation_level += 1
        self.processing_history.append({
            "level": self.current_augmentation_level,
            "augmentation_type": augmentation_type,
            "timestamp": datetime.now(),
            "branch": self.current_branch,
            "filters": self.current_filters.copy()
        })

    # Special context modes
    def enter_dispatch_mode(self):
        """Enter dispatch mode for parallel branch execution."""
        self.dispatch_mode = True

    def exit_dispatch_mode(self):
        """Exit dispatch mode."""
        self.dispatch_mode = False

    def set_fold_context(self, fold_config: Dict[str, Any]):
        """Set fold-specific context."""
        self.fold_context = fold_config

    def clear_fold_context(self):
        """Clear fold context."""
        self.fold_context = None

    def set_stacking_context(self, stack_config: Dict[str, Any]):
        """Set stacking-specific context."""
        self.stacking_context = stack_config

    def clear_stacking_context(self):
        """Clear stacking context."""
        self.stacking_context = None

    # Predictions management
    def add_predictions(self, model_name: str, predictions: Dict[str, Any]):
        """Store model predictions with context."""
        prediction_entry = {
            **predictions,
            "branch": self.current_branch,
            "processing_level": self.current_processing_level,
            "filters": self.current_filters.copy(),
            "timestamp": datetime.now()
        }
        self.predictions[model_name].append(prediction_entry)

    def get_predictions(self) -> Dict[str, Any]:
        """Get all stored predictions."""
        flattened = {}
        for model_name, pred_list in self.predictions.items():
            if pred_list:
                # Take the most recent predictions for each model
                flattened[model_name] = pred_list[-1]
        return flattened

    def get_predictions_for_branch(self, branch: int) -> Dict[str, Any]:
        """Get predictions for specific branch."""
        branch_predictions = {}
        for model_name, pred_list in self.predictions.items():
            for pred in pred_list:
                if pred.get("branch") == branch:
                    branch_predictions[model_name] = pred
                    break
        return branch_predictions

    # Utility methods
    def get_effective_filters(self) -> Dict[str, Any]:
        """Get effective filters considering all context."""
        if self._cache_dirty:
            effective_filters = self.current_filters.copy()

            # Add branch filter if not explicit
            if "branch" not in effective_filters:
                effective_filters["branch"] = self.current_branch

            # Add processing level if relevant
            if self.current_processing_level > 0:
                effective_filters["min_processing_level"] = 0
                effective_filters["max_processing_level"] = self.current_processing_level

            # Add source filters if active
            if self.active_sources is not None:
                effective_filters["active_sources"] = self.active_sources
                effective_filters["source_merge_mode"] = self.source_merge_mode

            # Add centroid filters if active
            if self.centroid_mode:
                effective_filters["centroid_mode"] = True
                effective_filters["centroid_groups"] = self.centroid_groups

            self._filter_cache = effective_filters
            self._cache_dirty = False

        return self._filter_cache.copy()

    def reset(self):
        """Reset context to initial state."""
        self.current_filters = {}
        self.predictions = defaultdict(list)
        self.scope_stack = []
        self.branch_stack = [0]
        self.branch_data_cache = {}
        self.processing_history = []
        self.current_processing_level = 0
        self.current_augmentation_level = 0
        self.source_stack = []
        self.active_sources = None
        self.source_merge_mode = False
        self.centroid_mode = False
        self.centroid_groups = {}
        self.group_centroids = {}
        self.fold_context = None
        self.stacking_context = None
        self.dispatch_mode = False
        self._filter_cache = {}
        self._cache_dirty = True

    def __repr__(self) -> str:
        return (f"PipelineContext(filters={self.current_filters}, "
                f"branch={self.current_branch}, "
                f"processing_level={self.current_processing_level}, "
                f"scope_depth={len(self.scope_stack)})")
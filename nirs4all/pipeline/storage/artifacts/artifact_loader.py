"""
Artifact Loader - Load artifacts by ID or execution context.

This module provides the ArtifactLoader class which replaces the legacy BinaryLoader.
It supports:
- Loading by artifact ID
- Loading by step/branch/fold context
- Transitive dependency resolution for stacking
- Per-fold model loading for CV averaging
- LRU caching for efficient reuse
- Lazy loading (artifacts only loaded when accessed)

The loader works with centralized storage at workspace/binaries/<dataset>/
and reads artifact metadata from v2 manifests.
"""

import logging
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nirs4all.pipeline.storage.artifacts.artifact_persistence import from_bytes
from nirs4all.pipeline.storage.artifacts.types import (
    ArtifactRecord,
    ArtifactType,
)
from nirs4all.pipeline.storage.artifacts.utils import (
    get_binaries_path,
    parse_artifact_id,
    artifact_id_matches_context,
)


logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache with configurable max size.

    Uses OrderedDict for O(1) access and LRU eviction.
    """

    def __init__(self, max_size: int = 100):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting oldest if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            # Update existing key and move to end
            self._cache[key] = value
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                # Evict oldest item
                self._cache.popitem(last=False)
            self._cache[key] = value

    def contains(self, key: str) -> bool:
        """Check if key is in cache without updating LRU order."""
        return key in self._cache

    def remove(self, key: str) -> None:
        """Remove item from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


class ArtifactLoader:
    """Load artifacts by ID or execution context.

    This class provides efficient loading of artifacts from centralized storage,
    with support for:
    - Direct loading by artifact ID
    - Context-based loading (step/branch/fold)
    - Dependency resolution for stacking meta-models
    - Per-fold model loading for cross-validation ensemble
    - LRU caching to avoid redundant I/O

    The loader uses lazy loading - artifacts are only deserialized when
    actually accessed via load_by_id() or related methods.

    Attributes:
        workspace: Workspace root path
        dataset: Dataset name
        binaries_dir: Path to centralized binaries
        results_dir: Path to results directory (for manifest reference)
        artifacts: Dictionary mapping artifact_id to ArtifactRecord

    Example:
        >>> loader = ArtifactLoader.from_manifest(manifest, results_dir)
        >>> model = loader.load_by_id("0001:3:all")
        >>> fold_models = loader.load_fold_models(step_index=3)
    """

    # Default cache size (number of artifacts)
    DEFAULT_CACHE_SIZE = 100

    def __init__(
        self,
        workspace: Path,
        dataset: str,
        results_dir: Optional[Path] = None,
        cache_size: int = DEFAULT_CACHE_SIZE
    ):
        """Initialize artifact loader.

        Args:
            workspace: Workspace root path
            dataset: Dataset name
            results_dir: Optional results directory path
            cache_size: Maximum number of artifacts to cache (default: 100)
        """
        self.workspace = Path(workspace)
        self.dataset = dataset
        self.results_dir = results_dir or self.workspace / "runs"

        # Centralized binaries directory
        self.binaries_dir = get_binaries_path(self.workspace, dataset)

        # Artifact index
        self._artifacts: Dict[str, ArtifactRecord] = {}

        # Content hash to artifact ID mapping (for deduplication)
        self._by_content_hash: Dict[str, str] = {}

        # LRU cache for loaded objects (artifact_id -> object)
        self._cache = LRUCache(max_size=cache_size)

        # Dependency graph for resolution
        self._dependencies: Dict[str, List[str]] = {}

    def load_by_id(self, artifact_id: str) -> Any:
        """Load a single artifact by its ID.

        Uses LRU cache to avoid redundant disk I/O. Artifacts are loaded
        lazily on first access.

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            Deserialized artifact object

        Raises:
            KeyError: If artifact ID not found
            FileNotFoundError: If artifact file doesn't exist
        """
        # Check LRU cache first
        cached = self._cache.get(artifact_id)
        if cached is not None:
            logger.debug(f"Cache hit for artifact: {artifact_id}")
            return cached

        # Get record
        record = self._artifacts.get(artifact_id)
        if record is None:
            raise KeyError(f"Artifact not found: {artifact_id}")

        # Load from disk (lazy loading)
        logger.debug(f"Loading artifact from disk: {artifact_id}")
        obj = self._load_artifact(record)

        # Cache and return
        self._cache.put(artifact_id, obj)
        return obj

    def load_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        fold_id: Optional[int] = None,
        pipeline_id: Optional[str] = None
    ) -> List[Tuple[str, Any]]:
        """Load all artifacts for a step context.

        Returns artifacts matching the specified step, branch path, and fold.
        If branch_path is provided, includes both branch-specific and shared
        (pre-branch) artifacts.

        Args:
            step_index: Step number to load
            branch_path: Optional branch path filter
            fold_id: Optional fold ID filter
            pipeline_id: Optional pipeline ID filter

        Returns:
            List of (artifact_id, loaded_object) tuples
        """
        results = []
        branch_path = branch_path or []

        for artifact_id, record in self._artifacts.items():
            # Check pipeline_id if specified
            if pipeline_id is not None and record.pipeline_id != pipeline_id:
                continue

            # Check step_index
            if record.step_index != step_index:
                continue

            # Check branch_path
            # Include if:
            # - record has no branch (shared/pre-branch artifact)
            # - record branch matches request
            if record.branch_path:
                if record.branch_path != branch_path:
                    continue

            # Check fold_id
            # Include if:
            # - fold_id not specified (load all folds)
            # - record has no fold (shared across folds)
            # - record fold matches request
            if fold_id is not None and record.fold_id is not None:
                if record.fold_id != fold_id:
                    continue

            # Load and add to results
            try:
                obj = self.load_by_id(artifact_id)
                results.append((artifact_id, obj))
            except (FileNotFoundError, IOError) as e:
                logger.warning(f"Failed to load artifact {artifact_id}: {e}")

        return results

    def load_with_dependencies(
        self,
        artifact_id: str
    ) -> Dict[str, Any]:
        """Load an artifact and all its transitive dependencies.

        Returns a dictionary mapping artifact IDs to loaded objects,
        in topological order (dependencies before dependents).

        Args:
            artifact_id: Starting artifact ID

        Returns:
            Dictionary of {artifact_id: loaded_object}

        Raises:
            KeyError: If artifact or dependency not found
            ValueError: If cycle detected in dependencies
        """
        # Resolve all dependencies (topologically sorted)
        dep_ids = self._resolve_dependencies(artifact_id)

        # Load in order
        result = {}
        for dep_id in dep_ids:
            result[dep_id] = self.load_by_id(dep_id)

        # Load the main artifact last
        result[artifact_id] = self.load_by_id(artifact_id)

        return result

    def load_fold_models(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        pipeline_id: Optional[str] = None
    ) -> List[Tuple[int, Any]]:
        """Load all fold-specific model artifacts for CV averaging.

        Returns models for all folds at the specified step, sorted by fold_id.

        Args:
            step_index: Step number where models are
            branch_path: Optional branch path filter
            pipeline_id: Optional pipeline ID filter

        Returns:
            List of (fold_id, loaded_model) tuples, sorted by fold_id
        """
        results = []
        branch_path = branch_path or []

        for artifact_id, record in self._artifacts.items():
            # Check pipeline_id if specified
            if pipeline_id is not None and record.pipeline_id != pipeline_id:
                continue

            # Check step_index
            if record.step_index != step_index:
                continue

            # Only model types
            if record.artifact_type not in (ArtifactType.MODEL, ArtifactType.META_MODEL):
                continue

            # Must be fold-specific
            if record.fold_id is None:
                continue

            # Check branch_path
            if branch_path and record.branch_path != branch_path:
                continue

            # Load and add to results
            try:
                obj = self.load_by_id(artifact_id)
                results.append((record.fold_id, obj))
            except (FileNotFoundError, IOError) as e:
                logger.warning(f"Failed to load fold model {artifact_id}: {e}")

        # Sort by fold_id
        return sorted(results, key=lambda x: x[0])

    def load_meta_model_with_sources(
        self,
        artifact_id: str,
        validate_branch: bool = True
    ) -> Tuple[Any, List[Tuple[str, Any]], List[str]]:
        """Load a meta-model and its source models.

        For stacking, loads the meta-model and all source models it depends on,
        preserving the feature column order as specified in meta_config.

        Args:
            artifact_id: Meta-model artifact ID
            validate_branch: If True, validate branch context matches

        Returns:
            Tuple of (meta_model, [(source_id, source_model), ...], feature_columns)
            where source_models are in the correct order for feature construction

        Raises:
            KeyError: If artifact not found
            ValueError: If artifact is not a meta-model or if branch validation fails
        """
        record = self._artifacts.get(artifact_id)
        if record is None:
            raise KeyError(f"Artifact not found: {artifact_id}")

        if record.artifact_type != ArtifactType.META_MODEL:
            raise ValueError(f"Artifact {artifact_id} is not a meta-model")

        # Load meta-model
        meta_model = self.load_by_id(artifact_id)

        # Load source models in order from meta_config
        source_models = []
        feature_columns = []

        if record.meta_config:
            # Use meta_config for ordered source loading
            if record.meta_config.source_models:
                for source_info in record.meta_config.source_models:
                    source_id = source_info.get("artifact_id")
                    if source_id:
                        # Validate branch context if requested
                        if validate_branch:
                            self._validate_branch_context(record, source_id)

                        source_model = self.load_by_id(source_id)
                        source_models.append((source_id, source_model))

            # Get feature columns
            if record.meta_config.feature_columns:
                feature_columns = record.meta_config.feature_columns
        else:
            # Fallback: use depends_on list (may not preserve order)
            logger.warning(
                f"Meta-model {artifact_id} has no meta_config. "
                "Using depends_on for source models (order may not be preserved)."
            )
            for dep_id in record.depends_on:
                dep_record = self._artifacts.get(dep_id)
                if dep_record and dep_record.artifact_type in (
                    ArtifactType.MODEL, ArtifactType.META_MODEL
                ):
                    if validate_branch:
                        self._validate_branch_context(record, dep_id)
                    source_model = self.load_by_id(dep_id)
                    source_models.append((dep_id, source_model))
                    feature_columns.append(f"{dep_record.class_name}_pred")

        return meta_model, source_models, feature_columns

    def _validate_branch_context(
        self,
        meta_record: ArtifactRecord,
        source_id: str
    ) -> None:
        """Validate that source model's branch context is compatible.

        For stacking, source models should either:
        - Be from the same branch path as the meta-model
        - Be from a shared (pre-branch) context (empty branch_path)

        Args:
            meta_record: The meta-model's artifact record
            source_id: The source model artifact ID to validate

        Raises:
            ValueError: If branch contexts are incompatible
        """
        source_record = self._artifacts.get(source_id)
        if source_record is None:
            raise ValueError(f"Source model not found: {source_id}")

        meta_branch = meta_record.branch_path
        source_branch = source_record.branch_path

        # Source can be from same branch or shared (pre-branch)
        if source_branch and source_branch != meta_branch:
            # Check if source is a prefix of meta (valid for nested branches)
            is_prefix = (
                len(source_branch) < len(meta_branch)
                and meta_branch[:len(source_branch)] == source_branch
            )
            if not is_prefix:
                raise ValueError(
                    f"Branch context mismatch for meta-model {meta_record.artifact_id}: "
                    f"source {source_id} is from branch {source_branch}, "
                    f"but meta-model is from branch {meta_branch}. "
                    f"Source models must be from the same branch or a parent branch."
                )

    def load_meta_model_for_prediction(
        self,
        artifact_id: str,
        X: Any = None
    ) -> Tuple[Any, List[Tuple[str, Any]], List[str]]:
        """Load a meta-model and its sources, ready for prediction.

        This method loads the complete stacking ensemble and validates
        that all components are compatible for prediction.

        Args:
            artifact_id: Meta-model artifact ID
            X: Optional input features for validation

        Returns:
            Tuple of (meta_model, source_models, feature_columns)
            where source_models is list of (artifact_id, model) tuples
            in the correct order for feature construction

        Raises:
            KeyError: If artifact or source models not found
            ValueError: If artifact is not a meta-model
        """
        meta_model, source_models, feature_columns = self.load_meta_model_with_sources(
            artifact_id, validate_branch=True
        )

        # Validate feature columns match source count
        if len(feature_columns) != len(source_models):
            logger.warning(
                f"Feature column count ({len(feature_columns)}) does not match "
                f"source model count ({len(source_models)}). "
                "Feature columns may need regeneration."
            )
            # Regenerate feature columns from source models
            feature_columns = []
            for source_id, source_model in source_models:
                feature_columns.append(f"{source_model.__class__.__name__}_pred")

        return meta_model, source_models, feature_columns

    def get_step_binaries(
        self,
        step_id: int,
        branch_id: Optional[int] = None,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[str, Any]]:
        """Legacy-compatible method for loading step binaries.

        This method provides backward compatibility with the BinaryLoader API.
        Prefer using load_for_step() for new code.

        Returns names in a format compatible with controller lookup patterns:
        - For models with fold_id: "ClassName_<op_num>" where op_num = step*100 + fold
        - For shared models: "ClassName_<op_num>" where op_num = step*100
        - For y_transformers (ENCODER type): "y_ClassName_<op_num>"
        - For x_transformers (TRANSFORMER type): "ClassName_<op_num>"

        Args:
            step_id: Step identifier (supports int or "step_substep" format)
            branch_id: Optional branch ID (converts to branch_path [branch_id])
            branch_path: Optional full branch path for nested branches (takes precedence over branch_id)

        Returns:
            List of (name, loaded_object) tuples
        """
        # Handle "step_substep" format
        if isinstance(step_id, str):
            if "_" in step_id:
                step_index = int(step_id.split("_")[0])
            else:
                step_index = int(step_id)
        else:
            step_index = step_id

        # Use branch_path if provided, otherwise convert branch_id to branch_path
        if branch_path is not None:
            effective_branch_path = branch_path
        elif branch_id is not None:
            effective_branch_path = [branch_id]
        else:
            effective_branch_path = []

        # Load artifacts for step
        artifacts = self.load_for_step(
            step_index=step_index,
            branch_path=effective_branch_path if effective_branch_path else None
        )

        # Convert to (name, object) format with controller-compatible names
        # We track operation counts by class name to match the original training pattern
        # where each class gets sequential operation numbers
        class_op_counts: Dict[str, int] = {}
        results = []

        for artifact_id, obj in artifacts:
            record = self._artifacts.get(artifact_id)
            if record:
                class_name = record.class_name
                artifact_type = record.artifact_type

                # Get or initialize operation count for this class
                if class_name not in class_op_counts:
                    class_op_counts[class_name] = 0
                op_index = class_op_counts[class_name]
                class_op_counts[class_name] += 1

                # Build operation number for naming:
                # - For models with fold_id: use step*100 + fold for ModelLoader compatibility
                # - For transformers: use sequential counter matching original training
                if artifact_type in (ArtifactType.MODEL, ArtifactType.META_MODEL):
                    if record.fold_id is not None:
                        op_num = step_index * 100 + record.fold_id
                    else:
                        op_num = step_index * 100
                else:
                    # For transformers, use step*100 + sequential index within step
                    # This ensures unique names that can be matched by class name search
                    op_num = step_index * 100 + op_index

                # Apply naming prefix based on artifact type
                # - ENCODER: y_transformers use "y_ClassName_N" pattern
                # - TRANSFORMER: x_transformers use "ClassName_N" pattern
                # - MODEL/META_MODEL: use "ClassName_N" pattern
                if artifact_type == ArtifactType.ENCODER:
                    name = f"y_{class_name}_{op_num}"
                else:
                    name = f"{class_name}_{op_num}"
            else:
                name = "unknown"
            results.append((name, obj))

        return results

    def has_binaries_for_step(
        self,
        step_number: int,
        substep_number: Optional[int] = None,
        branch_id: Optional[int] = None
    ) -> bool:
        """Check if binaries exist for a specific step.

        Legacy-compatible method for checking artifact availability.

        Args:
            step_number: The main step number
            substep_number: Ignored (kept for compatibility)
            branch_id: Optional branch ID to check

        Returns:
            True if artifacts exist for this step
        """
        branch_path = [branch_id] if branch_id is not None else None

        for record in self._artifacts.values():
            if record.step_index != step_number:
                continue

            if branch_path is not None:
                if record.branch_path and record.branch_path != branch_path:
                    continue

            return True

        return False

    def import_from_manifest(
        self,
        manifest: Dict[str, Any],
        results_dir: Optional[Path] = None
    ) -> None:
        """Import artifact records from a manifest.

        Loads v2 format manifests into the loader's index.

        Args:
            manifest: Manifest dictionary
            results_dir: Optional results directory override
        """
        if results_dir:
            self.results_dir = Path(results_dir)

        artifacts_section = manifest.get("artifacts", {})

        # Handle v2 format with "items" list
        if isinstance(artifacts_section, dict) and "items" in artifacts_section:
            items = artifacts_section.get("items", [])
        elif isinstance(artifacts_section, list):
            # Legacy v1 format - list directly
            items = artifacts_section
        else:
            items = []

        for item in items:
            # Handle both v1 and v2 formats
            if isinstance(item, dict):
                # Try v2 format first
                if "artifact_id" in item:
                    record = ArtifactRecord.from_dict(item)
                else:
                    # v1 format - convert
                    record = self._convert_v1_artifact(item)

                self._artifacts[record.artifact_id] = record
                self._by_content_hash[record.content_hash] = record.artifact_id

                # Track dependencies
                if record.depends_on:
                    self._dependencies[record.artifact_id] = record.depends_on

    def get_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Get artifact record by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            ArtifactRecord or None if not found
        """
        return self._artifacts.get(artifact_id)

    def get_all_records(self) -> List[ArtifactRecord]:
        """Get all artifact records.

        Returns:
            List of all ArtifactRecords
        """
        return list(self._artifacts.values())

    def clear_cache(self) -> None:
        """Clear the object cache to free memory."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state.

        Returns:
            Dictionary with cache statistics including:
            - cached_count: Number of cached artifacts
            - max_size: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - total_artifacts: Total number of registered artifacts
            - artifacts_by_type: Count by artifact type
        """
        cache_stats = self._cache.stats
        return {
            "cached_count": cache_stats["size"],
            "max_size": cache_stats["max_size"],
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "hit_rate": cache_stats["hit_rate"],
            "total_artifacts": len(self._artifacts),
            "artifacts_by_type": self._count_by_type(),
        }

    def set_cache_size(self, max_size: int) -> None:
        """Set the maximum cache size.

        If new size is smaller than current cache, oldest items are evicted.

        Args:
            max_size: New maximum cache size
        """
        if max_size < self._cache.size:
            # Need to evict some items
            while self._cache.size > max_size:
                self._cache._cache.popitem(last=False)
        self._cache._max_size = max_size

    def preload_artifacts(
        self,
        artifact_ids: Optional[List[str]] = None,
        artifact_types: Optional[List[ArtifactType]] = None
    ) -> int:
        """Preload artifacts into cache.

        Useful for warming the cache before prediction or when you know
        which artifacts will be needed.

        Args:
            artifact_ids: Specific artifact IDs to preload (default: all)
            artifact_types: Filter by artifact types (default: all)

        Returns:
            Number of artifacts loaded
        """
        count = 0

        if artifact_ids is not None:
            ids_to_load = artifact_ids
        else:
            ids_to_load = list(self._artifacts.keys())

        for artifact_id in ids_to_load:
            # Skip if already cached
            if self._cache.contains(artifact_id):
                continue

            record = self._artifacts.get(artifact_id)
            if record is None:
                continue

            # Filter by type if specified
            if artifact_types is not None and record.artifact_type not in artifact_types:
                continue

            try:
                obj = self._load_artifact(record)
                self._cache.put(artifact_id, obj)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to preload artifact {artifact_id}: {e}")

        return count

    @classmethod
    def from_manifest(
        cls,
        manifest: Dict[str, Any],
        results_dir: Path
    ) -> 'ArtifactLoader':
        """Create an ArtifactLoader from a pipeline manifest.

        Factory method for easy creation from manifest data.

        Args:
            manifest: Pipeline manifest dictionary
            results_dir: Path to results directory (manifest.yaml's parent)

        Returns:
            Initialized ArtifactLoader instance
        """
        # Determine workspace and dataset from results_dir
        # Expected structure: workspace/runs/<date>_<dataset>/<pipeline_id>/
        results_dir = Path(results_dir)

        # Try to find workspace root
        # results_dir is typically workspace/runs/YYYY-MM-DD_dataset/pipeline_id/
        # We need to go up to workspace (3 levels from pipeline_id folder)

        # First check: is parent's parent named "runs"?
        if results_dir.parent.parent.name == "runs":
            # results_dir = workspace/runs/<date>/pipeline_id
            workspace = results_dir.parent.parent.parent
        elif results_dir.parent.name == "runs":
            # results_dir = workspace/runs/<date> (no pipeline_id in path)
            workspace = results_dir.parent.parent
        else:
            # Fallback: assume results_dir is directly in workspace
            workspace = results_dir.parent

        # Extract dataset from manifest or directory name
        dataset = manifest.get("dataset", "")
        if not dataset:
            # Parse from parent directory name (YYYY-MM-DD_dataset)
            # Go up from pipeline_id folder to date_dataset folder
            if results_dir.parent.parent.name == "runs":
                dir_name = results_dir.parent.name
            else:
                dir_name = results_dir.name
            if "_" in dir_name:
                dataset = "_".join(dir_name.split("_")[1:])
            else:
                dataset = dir_name

        loader = cls(workspace, dataset, results_dir)
        loader.import_from_manifest(manifest, results_dir)

        return loader

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _load_artifact(self, record: ArtifactRecord) -> Any:
        """Load artifact binary from disk.

        Args:
            record: ArtifactRecord with path and format

        Returns:
            Deserialized object

        Raises:
            FileNotFoundError: If artifact file doesn't exist
        """
        # Try centralized binaries first (v2)
        artifact_path = self.binaries_dir / record.path
        if artifact_path.exists():
            content = artifact_path.read_bytes()
            return from_bytes(content, record.format)

        # Fallback: try _binaries in results_dir (v1 compatibility)
        legacy_path = self.results_dir / "_binaries" / record.path
        if legacy_path.exists():
            content = legacy_path.read_bytes()
            return from_bytes(content, record.format)

        raise FileNotFoundError(
            f"Artifact not found: {record.path}\n"
            f"Searched in:\n"
            f"  - {artifact_path}\n"
            f"  - {legacy_path}"
        )

    def _resolve_dependencies(
        self,
        artifact_id: str,
        visited: Optional[set] = None,
        stack: Optional[set] = None,
        _is_root: bool = True
    ) -> List[str]:
        """Resolve transitive dependencies in topological order.

        Args:
            artifact_id: Starting artifact
            visited: Set of already-processed artifacts
            stack: Stack for cycle detection
            _is_root: Internal flag to track if this is the top-level call

        Returns:
            List of dependency artifact IDs in topological order
            (excludes the starting artifact itself)

        Raises:
            ValueError: If cycle detected
        """
        if visited is None:
            visited = set()
        if stack is None:
            stack = set()

        if artifact_id in stack:
            raise ValueError(f"Cycle detected in dependency graph at {artifact_id}")

        if artifact_id in visited:
            return []

        stack.add(artifact_id)
        result = []

        # Get dependencies from record or tracking dict
        record = self._artifacts.get(artifact_id)
        deps = record.depends_on if record else self._dependencies.get(artifact_id, [])

        for dep_id in deps:
            result.extend(self._resolve_dependencies(dep_id, visited, stack, _is_root=False))

        stack.remove(artifact_id)
        visited.add(artifact_id)
        result.append(artifact_id)

        # Only exclude the starting artifact at the top level
        if _is_root:
            return result[:-1]
        return result

    def _convert_v1_artifact(self, item: Dict[str, Any]) -> ArtifactRecord:
        """Convert v1 artifact format to ArtifactRecord.

        Args:
            item: v1 format artifact dictionary

        Returns:
            ArtifactRecord instance
        """
        # Build a synthetic artifact_id from v1 fields
        step = item.get("step", 0)
        branch_id = item.get("branch_id")
        branch_path = [branch_id] if branch_id is not None else []

        # For v1, we don't have pipeline_id, so use a placeholder
        pipeline_id = "legacy"
        artifact_id = f"{pipeline_id}:{step}:all"
        if branch_path:
            artifact_id = f"{pipeline_id}:{':'.join(map(str, branch_path))}:{step}:all"

        # Determine artifact type from name or format
        name = item.get("name", "")
        if "model" in name.lower():
            artifact_type = ArtifactType.MODEL
        elif "scaler" in name.lower() or "transform" in name.lower():
            artifact_type = ArtifactType.TRANSFORMER
        elif "split" in name.lower():
            artifact_type = ArtifactType.SPLITTER
        elif "encoder" in name.lower():
            artifact_type = ArtifactType.ENCODER
        else:
            artifact_type = ArtifactType.MODEL

        # Extract class name from path if available
        path = item.get("path", "")
        class_name = path.split("_")[0] if "_" in path else name

        return ArtifactRecord(
            artifact_id=artifact_id,
            content_hash=item.get("hash", ""),
            path=path,
            pipeline_id=pipeline_id,
            branch_path=branch_path,
            step_index=step,
            fold_id=None,
            artifact_type=artifact_type,
            class_name=class_name,
            depends_on=[],
            format=item.get("format", "joblib"),
            format_version=item.get("format_version", ""),
            nirs4all_version=item.get("nirs4all_version", ""),
            size_bytes=item.get("size", 0),
            created_at=item.get("saved_at", ""),
            params={},
        )

    def _count_by_type(self) -> Dict[str, int]:
        """Count artifacts by type.

        Returns:
            Dictionary of {type_name: count}
        """
        counts: Dict[str, int] = {}
        for record in self._artifacts.values():
            type_name = record.artifact_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

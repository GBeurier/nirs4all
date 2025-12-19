"""
Artifact Registry V3 - Central registry for artifact management.

This module provides the ArtifactRegistry class which serves as the central
hub for artifact operations in the V3 artifacts system:

- Chain-based artifact identification for complete execution path tracking
- Content-addressed storage with global deduplication
- Dependency graph tracking for stacking/transfer
- Cleanup utilities for orphan detection and deletion

V3 Key Changes:
- Uses OperatorChain for artifact identification instead of V2 ID format
- Chain hash-based artifact IDs for deterministic identification
- Unified handling of branching, multi-source, stacking, and bundles

The registry works with centralized storage at workspace/binaries/<dataset>/
and coordinates with ManifestManager for manifest updates.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from nirs4all.pipeline.storage.artifacts.artifact_persistence import (
    to_bytes,
    from_bytes,
    _format_to_extension,
    _get_library_version,
    _get_nirs4all_version,
)
from nirs4all.pipeline.storage.artifacts.types import (
    ArtifactRecord,
    ArtifactType,
    MetaModelConfig,
)
from nirs4all.pipeline.storage.artifacts.operator_chain import (
    OperatorChain,
    OperatorNode,
    generate_artifact_id_v3,
    compute_chain_hash,
)
from nirs4all.pipeline.storage.artifacts.utils import (
    compute_content_hash,
    generate_filename,
    get_binaries_path,
    get_short_hash,
)


logger = logging.getLogger(__name__)


class DependencyGraph:
    """Tracks artifact dependencies for stacking and transfer.

    Maintains a directed graph where edges represent "depends on" relationships.
    Supports transitive dependency resolution with cycle detection.
    """

    def __init__(self):
        """Initialize empty dependency graph."""
        # Map: artifact_id -> list of artifact_ids it depends on
        self._dependencies: Dict[str, List[str]] = {}
        # Reverse map: artifact_id -> list of artifact_ids that depend on it
        self._dependents: Dict[str, List[str]] = {}

    def add_dependency(self, artifact_id: str, depends_on: str) -> None:
        """Add a dependency relationship.

        Args:
            artifact_id: The dependent artifact
            depends_on: The artifact being depended upon
        """
        if artifact_id not in self._dependencies:
            self._dependencies[artifact_id] = []

        if depends_on not in self._dependencies[artifact_id]:
            self._dependencies[artifact_id].append(depends_on)

        # Update reverse mapping
        if depends_on not in self._dependents:
            self._dependents[depends_on] = []

        if artifact_id not in self._dependents[depends_on]:
            self._dependents[depends_on].append(artifact_id)

    def add_dependencies(self, artifact_id: str, depends_on: List[str]) -> None:
        """Add multiple dependencies at once.

        Args:
            artifact_id: The dependent artifact
            depends_on: List of artifacts being depended upon
        """
        for dep in depends_on:
            self.add_dependency(artifact_id, dep)

    def get_dependencies(self, artifact_id: str) -> List[str]:
        """Get direct dependencies of an artifact.

        Args:
            artifact_id: Artifact to query

        Returns:
            List of artifact IDs this artifact depends on
        """
        return self._dependencies.get(artifact_id, []).copy()

    def get_dependents(self, artifact_id: str) -> List[str]:
        """Get artifacts that directly depend on this artifact.

        Args:
            artifact_id: Artifact to query

        Returns:
            List of artifact IDs that depend on this artifact
        """
        return self._dependents.get(artifact_id, []).copy()

    def resolve_dependencies(
        self,
        artifact_id: str,
        max_depth: int = 100
    ) -> List[str]:
        """Get all transitive dependencies (topologically sorted).

        Returns dependencies in order suitable for loading - dependencies
        before dependents.

        Args:
            artifact_id: Starting artifact
            max_depth: Maximum recursion depth (prevents cycles)

        Returns:
            List of all dependencies in topological order

        Raises:
            ValueError: If cycle detected or max depth exceeded
        """
        visited: Set[str] = set()
        result: List[str] = []
        stack: Set[str] = set()  # For cycle detection

        def visit(aid: str, depth: int) -> None:
            if depth > max_depth:
                raise ValueError(
                    f"Maximum dependency depth ({max_depth}) exceeded. "
                    "Possible cycle in dependency graph."
                )

            if aid in stack:
                raise ValueError(f"Cycle detected in dependency graph at {aid}")

            if aid in visited:
                return

            stack.add(aid)

            for dep in self.get_dependencies(aid):
                visit(dep, depth + 1)

            stack.remove(aid)
            visited.add(aid)
            result.append(aid)

        # Start traversal (don't include the starting artifact itself)
        for dep in self.get_dependencies(artifact_id):
            visit(dep, 0)

        return result

    def remove_artifact(self, artifact_id: str) -> None:
        """Remove an artifact and its edges from the graph.

        Args:
            artifact_id: Artifact to remove
        """
        # Remove from dependencies
        if artifact_id in self._dependencies:
            del self._dependencies[artifact_id]

        # Remove from other artifacts' dependency lists
        for deps in self._dependencies.values():
            if artifact_id in deps:
                deps.remove(artifact_id)

        # Remove from dependents
        if artifact_id in self._dependents:
            del self._dependents[artifact_id]

        # Remove from other artifacts' dependent lists
        for deps in self._dependents.values():
            if artifact_id in deps:
                deps.remove(artifact_id)

    def clear(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()
        self._dependents.clear()


class ArtifactRegistry:
    """Central registry for artifact management (V3).

    Provides:
    - Chain-based ID generation for complete execution path tracking
    - Content-addressed storage with deduplication
    - Dependency graph for stacking/transfer
    - Cleanup utilities

    V3 Key Changes:
    - Uses OperatorChain for artifact identification
    - Chain hash-based artifact IDs for deterministic identification
    - Chain path stored in ArtifactRecord for complete traceability
    - Lookup by chain path for prediction replay

    The registry coordinates between:
    - Centralized binaries at workspace/binaries/<dataset>/
    - Per-run manifests with artifact references
    - Dependency tracking for complex pipelines

    Attributes:
        workspace: Workspace root path
        dataset: Current dataset name
        binaries_dir: Path to centralized binaries
        dependency_graph: Dependency tracking graph
        pipeline_id: Current pipeline identifier for chain generation
    """

    def __init__(
        self,
        workspace: Path,
        dataset: str,
        manifest_manager: Optional[Any] = None,
        pipeline_id: str = ""
    ):
        """Initialize artifact registry.

        Args:
            workspace: Workspace root path
            dataset: Dataset name for this registry
            manifest_manager: Optional ManifestManager for manifest updates
            pipeline_id: Pipeline identifier for V3 ID generation
        """
        self.workspace = Path(workspace)
        self.dataset = dataset
        self.manifest_manager = manifest_manager
        self.pipeline_id = pipeline_id

        # Centralized binaries directory - created lazily when artifacts are saved
        self.binaries_dir = get_binaries_path(self.workspace, dataset)
        # Note: Directory is created in _ensure_binaries_dir() when first artifact is saved

        # In-memory registries
        self._artifacts: Dict[str, ArtifactRecord] = {}
        self._by_content_hash: Dict[str, str] = {}  # hash -> artifact_id
        self._by_path: Dict[str, str] = {}  # path -> artifact_id
        self._by_chain_path: Dict[str, str] = {}  # chain_path -> artifact_id (V3)

        # Dependency tracking
        self.dependency_graph = DependencyGraph()

        # Run-specific tracking for cleanup on failure
        self._current_run_artifacts: List[str] = []

    def generate_id(
        self,
        chain: Union[OperatorChain, str],
        fold_id: Optional[int] = None,
        pipeline_id: Optional[str] = None
    ) -> str:
        """Generate deterministic V3 artifact ID from operator chain.

        V3 Format: {pipeline_id}${chain_hash}:{fold_id}

        Args:
            chain: OperatorChain or chain path string
            fold_id: CV fold (None for shared)
            pipeline_id: Pipeline identifier (uses self.pipeline_id if None)

        Returns:
            V3 Artifact ID string

        Examples:
            >>> registry.generate_id(chain, fold_id=0)
            '0001_pls$a1b2c3d4e5f6:0'
            >>> registry.generate_id("s1.MinMaxScaler>s3.PLS", fold_id=None)
            '0001_pls$7f8e9d0c1b2a:all'
        """
        pid = pipeline_id or self.pipeline_id
        return generate_artifact_id_v3(pid, chain, fold_id)

    def register_with_chain(
        self,
        obj: Any,
        chain: Union[OperatorChain, str],
        artifact_type: ArtifactType,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        fold_id: Optional[int] = None,
        substep_index: Optional[int] = None,
        depends_on: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        meta_config: Optional[MetaModelConfig] = None,
        format_hint: Optional[str] = None,
        custom_name: Optional[str] = None,
        pipeline_id: Optional[str] = None
    ) -> ArtifactRecord:
        """Register and persist an artifact using V3 chain-based identification.

        This is the primary registration method for V3. It generates a deterministic
        artifact ID from the operator chain and stores the chain path for later lookup.

        Args:
            obj: Object to persist (model, transformer, etc.)
            chain: OperatorChain or chain path string
            artifact_type: Classification (model, transformer, etc.)
            step_index: Pipeline step index (1-based)
            branch_path: List of branch indices (empty for non-branching)
            source_index: Multi-source index (None for single source)
            fold_id: CV fold (None for shared artifacts)
            substep_index: Substep index for [model1, model2]
            depends_on: List of artifact IDs this depends on
            params: Model parameters for inspection
            meta_config: Meta-model configuration (for stacking)
            format_hint: Optional serialization format hint
            custom_name: User-defined name for the artifact

        Returns:
            ArtifactRecord with full metadata

        Raises:
            ValueError: If object cannot be serialized or if meta-model
                dependencies are missing
        """
        depends_on = depends_on or []
        params = params or {}
        branch_path = branch_path or []
        pid = pipeline_id or self.pipeline_id

        # Get chain path string
        if isinstance(chain, OperatorChain):
            chain_path = chain.to_path()
        else:
            chain_path = chain

        # Generate V3 artifact ID
        artifact_id = self.generate_id(chain_path, fold_id, pid)

        # Validate dependencies for meta-models
        if artifact_type == ArtifactType.META_MODEL:
            self._validate_meta_model_dependencies(
                artifact_id, depends_on, meta_config
            )

        # Serialize object
        content, format_name = to_bytes(obj, format_hint)

        # Compute content hash
        content_hash = compute_content_hash(content)

        # Check for existing artifact with same content (deduplication)
        existing_path = self._find_existing_by_hash(content_hash)

        if existing_path:
            # Reuse existing file
            path = existing_path
            logger.debug(f"Deduplication: reusing {path} for {artifact_id}")
        else:
            # Generate new filename and save
            extension = _format_to_extension(format_name)
            class_name = obj.__class__.__name__
            filename = generate_filename(
                artifact_type.value,
                class_name,
                content_hash,
                extension
            )
            path = filename

            # Write to binaries directory (create lazily if needed)
            self.binaries_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.binaries_dir / filename
            if not artifact_path.exists():
                artifact_path.write_bytes(content)
                logger.debug(f"Saved artifact: {artifact_path}")

        # Create V3 record with chain_path
        record = ArtifactRecord(
            artifact_id=artifact_id,
            content_hash=content_hash,
            path=path,
            chain_path=chain_path,
            source_index=source_index,
            pipeline_id=pid,
            branch_path=branch_path,
            step_index=step_index,
            substep_index=substep_index,
            fold_id=fold_id,
            artifact_type=artifact_type,
            class_name=obj.__class__.__name__,
            custom_name=custom_name or "",
            depends_on=depends_on,
            format=format_name,
            format_version=_get_library_version(obj),
            nirs4all_version=_get_nirs4all_version(),
            size_bytes=len(content),
            created_at=datetime.now(timezone.utc).isoformat(),
            params=params,
            meta_config=meta_config,
            version=3,  # V3 schema version
        )

        # Update in-memory registries
        self._artifacts[artifact_id] = record
        self._by_content_hash[content_hash] = artifact_id
        self._by_path[path] = artifact_id
        self._by_chain_path[chain_path] = artifact_id  # V3 chain lookup

        # Track dependencies
        if depends_on:
            self.dependency_graph.add_dependencies(artifact_id, depends_on)

        # Track for run cleanup
        self._current_run_artifacts.append(artifact_id)

        return record

    def get_by_chain(
        self,
        chain: Union[OperatorChain, str],
        fold_id: Optional[int] = None
    ) -> Optional[ArtifactRecord]:
        """Get artifact by exact chain path match.

        Args:
            chain: OperatorChain or chain path string
            fold_id: Optional fold ID to filter (None = any fold)

        Returns:
            ArtifactRecord or None if not found
        """
        if isinstance(chain, OperatorChain):
            chain_path = chain.to_path()
        else:
            chain_path = chain

        artifact_id = self._by_chain_path.get(chain_path)
        if artifact_id:
            record = self._artifacts.get(artifact_id)
            if record and (fold_id is None or record.fold_id == fold_id):
                return record
        return None

    def get_chain_prefix(
        self,
        prefix: str,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None
    ) -> List[ArtifactRecord]:
        """Get all artifacts whose chain path starts with the given prefix.

        Useful for finding all artifacts in a chain for prediction replay.

        Args:
            prefix: Chain path prefix to match
            branch_path: Optional branch path filter
            source_index: Optional source index filter

        Returns:
            List of matching ArtifactRecords
        """
        results = []
        for chain_path, artifact_id in self._by_chain_path.items():
            if chain_path.startswith(prefix):
                record = self._artifacts.get(artifact_id)
                if record:
                    # Apply filters
                    if branch_path is not None and record.branch_path != branch_path:
                        continue
                    if source_index is not None and record.source_index != source_index:
                        continue
                    results.append(record)
        return results

    def register(
        self,
        obj: Any,
        artifact_id: str,
        artifact_type: ArtifactType,
        depends_on: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        meta_config: Optional[MetaModelConfig] = None,
        format_hint: Optional[str] = None,
        custom_name: Optional[str] = None,
        chain_path: str = "",
        source_index: Optional[int] = None
    ) -> ArtifactRecord:
        """Register and persist an artifact.

        Serializes the object, stores in centralized binaries (with deduplication),
        and creates an ArtifactRecord.

        Note: This method accepts pre-generated artifact IDs for backward compatibility.
        For new code, use register_with_chain() which generates IDs from OperatorChain.

        Args:
            obj: Object to persist (model, transformer, etc.)
            artifact_id: Pre-generated artifact ID (V3 format: pipeline$hash:fold)
            artifact_type: Classification (model, transformer, etc.)
            depends_on: List of artifact IDs this depends on
            params: Model parameters for inspection
            meta_config: Meta-model configuration (for stacking)
            format_hint: Optional serialization format hint
            custom_name: User-defined name for the artifact (e.g., "Q5_PLS_10")
            chain_path: V3 operator chain path (required for full traceability)
            source_index: Multi-source index (None for single source)

        Returns:
            ArtifactRecord with full metadata

        Raises:
            ValueError: If object cannot be serialized or if meta-model
                dependencies are missing
        """
        depends_on = depends_on or []
        params = params or {}

        # Validate dependencies for meta-models
        if artifact_type == ArtifactType.META_MODEL:
            self._validate_meta_model_dependencies(
                artifact_id, depends_on, meta_config
            )

        # Extract pipeline_id and fold_id from V3 artifact ID
        from nirs4all.pipeline.storage.artifacts.operator_chain import (
            parse_artifact_id_v3,
            is_v3_artifact_id,
        )

        if is_v3_artifact_id(artifact_id):
            pipeline_id, _, fold_id = parse_artifact_id_v3(artifact_id)
            # Extract step_index and branch_path from chain_path if available
            step_index = 0
            branch_path: List[int] = []
            substep_index = None
            if chain_path:
                # Parse last node for step_index and branch_path
                last_node_str = chain_path.split(">")[-1] if ">" in chain_path else chain_path
                node = OperatorNode.from_key(last_node_str)
                step_index = node.step_index
                branch_path = node.branch_path
                substep_index = node.substep_index
        else:
            # Fallback: Extract from chain_path or use defaults
            pipeline_id = self.pipeline_id or artifact_id.split(":")[0]
            fold_id = None
            step_index = 0
            branch_path = []
            substep_index = None
            if chain_path:
                last_node_str = chain_path.split(">")[-1] if ">" in chain_path else chain_path
                if last_node_str:
                    try:
                        node = OperatorNode.from_key(last_node_str)
                        step_index = node.step_index
                        branch_path = node.branch_path
                        substep_index = node.substep_index
                    except ValueError:
                        pass

        # Serialize object
        content, format_name = to_bytes(obj, format_hint)

        # Compute content hash
        content_hash = compute_content_hash(content)

        # Check for existing artifact with same content (deduplication)
        existing_path = self._find_existing_by_hash(content_hash)

        if existing_path:
            # Reuse existing file
            path = existing_path
            logger.debug(f"Deduplication: reusing {path} for {artifact_id}")
        else:
            # Generate new filename and save
            extension = _format_to_extension(format_name)
            class_name = obj.__class__.__name__
            filename = generate_filename(
                artifact_type.value,
                class_name,
                content_hash,
                extension
            )
            path = filename

            # Write to binaries directory (create lazily if needed)
            self.binaries_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.binaries_dir / filename
            if not artifact_path.exists():
                artifact_path.write_bytes(content)
                logger.debug(f"Saved artifact: {artifact_path}")

        # Create V3 record with chain_path
        record = ArtifactRecord(
            artifact_id=artifact_id,
            content_hash=content_hash,
            path=path,
            chain_path=chain_path,
            source_index=source_index,
            pipeline_id=pipeline_id,
            branch_path=branch_path,
            step_index=step_index,
            substep_index=substep_index,
            fold_id=fold_id,
            artifact_type=artifact_type,
            class_name=obj.__class__.__name__,
            custom_name=custom_name or "",
            depends_on=depends_on,
            format=format_name,
            format_version=_get_library_version(obj),
            nirs4all_version=_get_nirs4all_version(),
            size_bytes=len(content),
            created_at=datetime.now(timezone.utc).isoformat(),
            params=params,
            meta_config=meta_config,
            version=3,  # V3 schema version
        )

        # Update in-memory registries
        self._artifacts[artifact_id] = record
        self._by_content_hash[content_hash] = artifact_id
        self._by_path[path] = artifact_id
        if chain_path:
            self._by_chain_path[chain_path] = artifact_id  # V3 chain lookup

        # Track dependencies
        if depends_on:
            self.dependency_graph.add_dependencies(artifact_id, depends_on)

        # Track for run cleanup
        self._current_run_artifacts.append(artifact_id)

        return record

    def resolve(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Resolve artifact ID to record.

        Args:
            artifact_id: Artifact ID to resolve

        Returns:
            ArtifactRecord or None if not found
        """
        return self._artifacts.get(artifact_id)

    def resolve_by_hash(self, content_hash: str) -> Optional[ArtifactRecord]:
        """Resolve content hash to artifact record.

        Args:
            content_hash: Content hash to look up

        Returns:
            ArtifactRecord or None if not found
        """
        artifact_id = self._by_content_hash.get(content_hash)
        if artifact_id:
            return self._artifacts.get(artifact_id)
        return None

    def get_dependencies(self, artifact_id: str) -> List[str]:
        """Get direct dependencies of an artifact.

        Args:
            artifact_id: Artifact to query

        Returns:
            List of artifact IDs
        """
        return self.dependency_graph.get_dependencies(artifact_id)

    def resolve_dependencies(self, artifact_id: str) -> List[ArtifactRecord]:
        """Get all transitive dependencies as records.

        Args:
            artifact_id: Starting artifact

        Returns:
            List of ArtifactRecords in topological order
        """
        dep_ids = self.dependency_graph.resolve_dependencies(artifact_id)
        return [self._artifacts[aid] for aid in dep_ids if aid in self._artifacts]

    def register_meta_model(
        self,
        obj: Any,
        artifact_id: str,
        source_model_ids: List[str],
        feature_columns: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        format_hint: Optional[str] = None
    ) -> ArtifactRecord:
        """Register a stacking meta-model with source model references.

        Convenience method for registering meta-models that automatically:
        - Creates the MetaModelConfig with ordered source model references
        - Sets up dependency tracking to source models
        - Validates that all source models exist

        Args:
            obj: The meta-model object to persist
            artifact_id: Pre-generated artifact ID for the meta-model
            source_model_ids: Ordered list of source model artifact IDs
            feature_columns: Optional feature column names matching source order
            params: Optional meta-model parameters
            format_hint: Optional serialization format hint

        Returns:
            ArtifactRecord for the registered meta-model

        Raises:
            ValueError: If any source model is not found in the registry

        Example:
            >>> meta_config_record = registry.register_meta_model(
            ...     obj=ridge_meta_model,
            ...     artifact_id="0001:5:all",
            ...     source_model_ids=["0001:3:all", "0001:4:all"],
            ...     feature_columns=["PLSRegression_pred", "RandomForestRegressor_pred"]
            ... )
        """
        # Build source_models list with feature indices
        source_models = []
        for idx, source_id in enumerate(source_model_ids):
            source_models.append({
                "artifact_id": source_id,
                "feature_index": idx
            })

        # Generate feature_columns from source model class names if not provided
        if feature_columns is None:
            feature_columns = []
            for source_id in source_model_ids:
                record = self.resolve(source_id)
                if record:
                    feature_columns.append(f"{record.class_name}_pred")
                else:
                    feature_columns.append(f"source_{source_id}_pred")

        # Create meta config
        meta_config = MetaModelConfig(
            source_models=source_models,
            feature_columns=feature_columns
        )

        # Register with meta-model type
        return self.register(
            obj=obj,
            artifact_id=artifact_id,
            artifact_type=ArtifactType.META_MODEL,
            depends_on=source_model_ids,
            params=params,
            meta_config=meta_config,
            format_hint=format_hint
        )

    def get_artifacts_for_step(
        self,
        pipeline_id: str,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        fold_id: Optional[int] = None
    ) -> List[ArtifactRecord]:
        """Get all artifacts for a specific step context.

        Args:
            pipeline_id: Pipeline to query
            step_index: Step number
            branch_path: Optional branch filter
            fold_id: Optional fold filter

        Returns:
            List of matching ArtifactRecords
        """
        results = []
        for record in self._artifacts.values():
            if record.pipeline_id != pipeline_id:
                continue
            if record.step_index != step_index:
                continue
            if branch_path is not None and record.branch_path != branch_path:
                continue
            if fold_id is not None and record.fold_id != fold_id:
                continue
            results.append(record)
        return results

    def get_fold_models(
        self,
        pipeline_id: str,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[ArtifactRecord]:
        """Get all fold-specific model artifacts for CV averaging.

        Args:
            pipeline_id: Pipeline to query
            step_index: Model step number
            branch_path: Optional branch filter

        Returns:
            List of per-fold model ArtifactRecords
        """
        results = []
        for record in self._artifacts.values():
            if record.pipeline_id != pipeline_id:
                continue
            if record.step_index != step_index:
                continue
            if record.artifact_type not in (ArtifactType.MODEL, ArtifactType.META_MODEL):
                continue
            if record.fold_id is None:
                continue  # Skip shared artifacts
            if branch_path is not None and record.branch_path != branch_path:
                continue
            results.append(record)

        # Sort by fold_id
        return sorted(results, key=lambda r: r.fold_id or 0)

    def load_artifact(self, record: ArtifactRecord) -> Any:
        """Load artifact binary from disk.

        Args:
            record: ArtifactRecord with path and format

        Returns:
            Deserialized object

        Raises:
            FileNotFoundError: If artifact file doesn't exist
        """
        artifact_path = self.binaries_dir / record.path
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        content = artifact_path.read_bytes()
        return from_bytes(content, record.format)

    def import_from_manifest(
        self,
        manifest: Dict[str, Any],
        results_dir: Path
    ) -> None:
        """Import artifact records from a manifest.

        Loads V3 format manifests into the registry, building all indexes
        including the chain_path index for V3 lookups.

        Args:
            manifest: Manifest dictionary
            results_dir: Path to results directory
        """
        artifacts_section = manifest.get("artifacts", {})

        # Handle v2/v3 format with "items" list
        if isinstance(artifacts_section, dict) and "items" in artifacts_section:
            items = artifacts_section.get("items", [])
        elif isinstance(artifacts_section, list):
            # Legacy v1 format - list directly
            items = artifacts_section
        else:
            items = []

        for item in items:
            record = ArtifactRecord.from_dict(item)
            self._artifacts[record.artifact_id] = record
            self._by_content_hash[record.content_hash] = record.artifact_id
            self._by_path[record.path] = record.artifact_id

            # V3: Index by chain_path if available
            if record.chain_path:
                self._by_chain_path[record.chain_path] = record.artifact_id

            # Rebuild dependency graph
            if record.depends_on:
                self.dependency_graph.add_dependencies(
                    record.artifact_id,
                    record.depends_on
                )

    def export_to_manifest(self) -> Dict[str, Any]:
        """Export registry to manifest V3 format.

        Returns:
            Dictionary suitable for manifest artifacts section
        """
        return {
            "schema_version": "3.0",
            "items": [record.to_dict() for record in self._artifacts.values()]
        }

    def get_all_records(self) -> List[ArtifactRecord]:
        """Get all registered artifacts.

        Returns:
            List of all ArtifactRecords
        """
        return list(self._artifacts.values())

    # =========================================================================
    # Cleanup Utilities
    # =========================================================================

    def find_orphaned_artifacts(self, scan_all_manifests: bool = True) -> List[str]:
        """Find artifact files not referenced by any manifest.

        Scans binaries directory and compares with all referenced artifacts
        from manifests in the workspace.

        Args:
            scan_all_manifests: If True, scan all manifests in workspace/runs/.
                If False, only check against in-memory registry.

        Returns:
            List of orphaned filenames
        """
        if not self.binaries_dir.exists():
            return []

        # Get all files in binaries directory
        all_files = {
            f.name for f in self.binaries_dir.iterdir()
            if f.is_file()
        }

        # Get referenced files
        if scan_all_manifests:
            referenced = self._scan_all_manifest_references()
        else:
            referenced = {record.path for record in self._artifacts.values()}

        # Find orphans
        orphans = all_files - referenced
        return sorted(orphans)

    def _scan_all_manifest_references(self) -> Set[str]:
        """Scan all manifests in workspace for artifact references.

        Searches workspace/runs/<dataset>/**/manifest.yaml for artifact paths.

        Returns:
            Set of referenced artifact filenames
        """
        import yaml

        referenced: Set[str] = set()

        # Add in-memory references
        referenced.update(record.path for record in self._artifacts.values())

        # Scan all manifest files in runs directory
        runs_dir = self.workspace / "runs"
        if not runs_dir.exists():
            return referenced

        # Look for manifests in this dataset's runs
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if this run belongs to our dataset
            # Run dirs are named: YYYY-MM-DD_dataset
            if f"_{self.dataset}" not in run_dir.name:
                continue

            # Find all manifest.yaml files in this run
            for manifest_path in run_dir.glob("**/manifest.yaml"):
                try:
                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)

                    # Extract artifact paths from manifest
                    artifacts_section = manifest.get("artifacts", {})

                    # Handle v2 format with "items" list
                    if isinstance(artifacts_section, dict):
                        items = artifacts_section.get("items", [])
                    elif isinstance(artifacts_section, list):
                        items = artifacts_section
                    else:
                        items = []

                    for item in items:
                        if isinstance(item, dict) and "path" in item:
                            referenced.add(item["path"])

                except Exception as e:
                    logger.warning(f"Error reading manifest {manifest_path}: {e}")

        return referenced

    def delete_orphaned_artifacts(
        self,
        dry_run: bool = True,
        scan_all_manifests: bool = True
    ) -> Tuple[List[str], int]:
        """Delete artifacts not referenced by any manifest.

        Args:
            dry_run: If True, only report what would be deleted
            scan_all_manifests: If True, scan all manifests before deletion

        Returns:
            Tuple of (deleted_files, bytes_freed)
        """
        orphans = self.find_orphaned_artifacts(scan_all_manifests=scan_all_manifests)
        deleted = []
        bytes_freed = 0

        for filename in orphans:
            filepath = self.binaries_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                if not dry_run:
                    filepath.unlink()
                    logger.info(f"Deleted orphaned artifact: {filename}")
                deleted.append(filename)
                bytes_freed += size

        if not dry_run and deleted:
            logger.info(
                f"Cleaned up {len(deleted)} orphaned artifacts, "
                f"freed {bytes_freed / 1024:.1f} KB"
            )

        return deleted, bytes_freed

    def delete_pipeline_artifacts(
        self,
        pipeline_id: str,
        delete_files: bool = False
    ) -> int:
        """Delete all artifacts for a specific pipeline.

        Args:
            pipeline_id: Pipeline to delete artifacts for
            delete_files: If True, also delete the binary files from disk

        Returns:
            Number of artifacts deleted
        """
        to_delete = [
            aid for aid, record in self._artifacts.items()
            if record.pipeline_id == pipeline_id
        ]

        count = 0
        for artifact_id in to_delete:
            record = self._artifacts.pop(artifact_id, None)
            if record:
                # Remove from indexes
                self._by_content_hash.pop(record.content_hash, None)
                self._by_path.pop(record.path, None)
                self.dependency_graph.remove_artifact(artifact_id)

                # Optionally delete file if not referenced elsewhere
                if delete_files:
                    # Check if any other artifact uses this file
                    if record.path not in self._by_path:
                        filepath = self.binaries_dir / record.path
                        if filepath.exists():
                            filepath.unlink()
                            logger.info(f"Deleted artifact file: {record.path}")

                count += 1

        return count

    def cleanup_failed_run(self) -> int:
        """Clean up artifacts from a failed run.

        Deletes artifacts registered during the current run.
        Called automatically on exception.

        Returns:
            Number of artifacts cleaned up
        """
        count = 0
        for artifact_id in self._current_run_artifacts:
            record = self._artifacts.pop(artifact_id, None)
            if record:
                self._by_content_hash.pop(record.content_hash, None)
                self._by_path.pop(record.path, None)
                self.dependency_graph.remove_artifact(artifact_id)
                count += 1

        self._current_run_artifacts.clear()
        if count > 0:
            logger.info(f"Cleaned up {count} artifacts from failed run")
        return count

    def purge_dataset_artifacts(self, confirm: bool = False) -> Tuple[int, int]:
        """Delete ALL artifacts for this dataset.

        This is a destructive operation that removes all artifacts in the
        binaries directory for this dataset, regardless of manifest references.

        Args:
            confirm: Must be True to actually delete files

        Returns:
            Tuple of (files_deleted, bytes_freed)

        Raises:
            ValueError: If confirm is False
        """
        if not confirm:
            raise ValueError(
                "Purge requires confirm=True. This will delete ALL artifacts "
                f"for dataset '{self.dataset}'."
            )

        if not self.binaries_dir.exists():
            return 0, 0

        files_deleted = 0
        bytes_freed = 0

        for filepath in self.binaries_dir.iterdir():
            if filepath.is_file():
                size = filepath.stat().st_size
                filepath.unlink()
                files_deleted += 1
                bytes_freed += size
                logger.info(f"Purged artifact: {filepath.name}")

        # Clear in-memory state
        self._artifacts.clear()
        self._by_content_hash.clear()
        self._by_path.clear()
        self.dependency_graph.clear()
        self._current_run_artifacts.clear()

        logger.info(
            f"Purged {files_deleted} artifacts for dataset '{self.dataset}', "
            f"freed {bytes_freed / 1024 / 1024:.2f} MB"
        )

        return files_deleted, bytes_freed

    def start_run(self) -> None:
        """Start tracking a new run for cleanup purposes."""
        self._current_run_artifacts.clear()

    def end_run(self) -> None:
        """End run tracking (successful completion)."""
        self._current_run_artifacts.clear()

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _validate_meta_model_dependencies(
        self,
        artifact_id: str,
        depends_on: List[str],
        meta_config: Optional[MetaModelConfig]
    ) -> None:
        """Validate dependencies for meta-model artifacts.

        Ensures that all source models referenced by a meta-model exist
        in the registry. This is critical for stacking pipelines where
        the meta-model depends on predictions from source models.

        Args:
            artifact_id: The meta-model artifact ID being registered
            depends_on: List of dependency artifact IDs
            meta_config: Meta-model configuration with source model refs

        Raises:
            ValueError: If any source model is not found in the registry
        """
        missing_deps = []

        # Check dependencies from depends_on list
        for dep_id in depends_on:
            if dep_id not in self._artifacts:
                missing_deps.append(dep_id)

        # Check source models from meta_config
        if meta_config and meta_config.source_models:
            for source_info in meta_config.source_models:
                source_id = source_info.get("artifact_id")
                if source_id and source_id not in self._artifacts:
                    missing_deps.append(source_id)

        if missing_deps:
            missing_str = ", ".join(missing_deps)
            raise ValueError(
                f"Cannot register meta-model {artifact_id}: "
                f"missing source model dependencies: {missing_str}. "
                f"Ensure source models are registered before the meta-model."
            )

    def _find_existing_by_hash(self, content_hash: str) -> Optional[str]:
        """Find existing artifact path by content hash.

        Checks both in-memory registry and filesystem for deduplication.

        Args:
            content_hash: Content hash to look up

        Returns:
            Existing path or None
        """
        # Check in-memory registry first
        if content_hash in self._by_content_hash:
            artifact_id = self._by_content_hash[content_hash]
            record = self._artifacts.get(artifact_id)
            if record:
                return record.path

        # Check filesystem for existing files with this hash
        short_hash = get_short_hash(content_hash)
        for filepath in self.binaries_dir.glob(f"*_{short_hash}.*"):
            # Verify it's actually the same content
            # (short hash collision is unlikely but possible)
            return filepath.name

        return None

    def get_stats(self, scan_all_manifests: bool = True) -> Dict[str, Any]:
        """Get storage statistics.

        Args:
            scan_all_manifests: If True, scan all manifests for accurate stats

        Returns:
            Dictionary with storage stats including:
            - total_artifacts: Number of registered artifacts
            - unique_files: Number of unique binary files
            - total_size_bytes: Total size of all artifacts
            - deduplication_ratio: Ratio of saved space from deduplication
            - by_type: Count of artifacts by type
            - orphaned_count: Number of orphaned files
            - disk_usage_bytes: Actual disk usage in binaries directory
        """
        total_artifacts = len(self._artifacts)
        unique_files = len(set(r.path for r in self._artifacts.values()))
        total_size = sum(r.size_bytes for r in self._artifacts.values())

        # Count by type
        by_type: Dict[str, int] = {}
        for record in self._artifacts.values():
            type_name = record.artifact_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        # Scan filesystem for actual disk usage
        disk_usage = 0
        file_count = 0
        if self.binaries_dir.exists():
            for f in self.binaries_dir.iterdir():
                if f.is_file():
                    disk_usage += f.stat().st_size
                    file_count += 1

        # Find orphaned artifacts
        orphaned = self.find_orphaned_artifacts(scan_all_manifests=scan_all_manifests)
        orphaned_size = sum(
            (self.binaries_dir / f).stat().st_size
            for f in orphaned
            if (self.binaries_dir / f).exists()
        )

        return {
            "total_artifacts": total_artifacts,
            "unique_files": unique_files,
            "total_size_bytes": total_size,
            "deduplication_ratio": (
                1 - unique_files / total_artifacts
                if total_artifacts > 0 else 0
            ),
            "by_type": by_type,
            "orphaned_count": len(orphaned),
            "orphaned_size_bytes": orphaned_size,
            "disk_usage_bytes": disk_usage,
            "disk_file_count": file_count,
            "dataset": self.dataset,
            "binaries_path": str(self.binaries_dir),
        }

"""
Artifact type definitions for the V3 artifacts system.

This module defines the core data structures for artifact management:
- ArtifactType: Enum for artifact classification
- ArtifactRecord: Complete artifact metadata for manifest storage

The V3 artifacts system uses operator chains for complete execution path
tracking, enabling deterministic artifact IDs that work correctly with
branching, multi-source, stacking, and cross-validation.

Key V3 improvements:
- OperatorChain tracking for full execution path
- Source index tracking for multi-source pipelines
- Chain hash-based artifact IDs for deterministic identification
- Unified handling of all edge cases (branching, stacking, bundles)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain


class ArtifactType(str, Enum):
    """Classification of artifact types.

    Each type has specific handling:
    - model: Trained ML models (sklearn, tensorflow, pytorch, etc.)
    - transformer: Fitted preprocessors (scalers, feature extractors)
    - splitter: Train/test split configuration (for reproducibility)
    - encoder: Label encoders, y-scalers
    - meta_model: Stacking meta-models with source model dependencies
    """

    MODEL = "model"
    TRANSFORMER = "transformer"
    SPLITTER = "splitter"
    ENCODER = "encoder"
    META_MODEL = "meta_model"

    def __str__(self) -> str:
        return self.value


@dataclass
class MetaModelConfig:
    """Configuration for meta-model source tracking.

    Stores the ordered source models that feed into a stacking meta-model,
    along with their feature column mapping.

    Attributes:
        source_models: Ordered list of source model artifact IDs with feature indices
        feature_columns: Feature column names in the meta-model input order
    """

    source_models: List[Dict[str, Any]] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "source_models": self.source_models,
            "feature_columns": self.feature_columns
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaModelConfig":
        """Create from dictionary."""
        return cls(
            source_models=data.get("source_models", []),
            feature_columns=data.get("feature_columns", [])
        )


@dataclass
class ArtifactRecord:
    """Complete artifact metadata for manifest storage (V3).

    This record contains all metadata needed to:
    - Uniquely identify an artifact via operator chain
    - Load the artifact from centralized storage
    - Resolve dependencies for stacking/transfer
    - Track serialization format and library versions

    V3 Format:
        artifact_id: "{pipeline_id}${chain_hash}:{fold_id}"
        chain_path: Full operator chain path string

    Attributes:
        artifact_id: Unique, deterministic ID based on chain hash
                     Format: "{pipeline_id}${chain_hash}:{fold_id}"
        content_hash: SHA256 hash of binary content (for deduplication)
        path: Relative path in binaries/<dataset>/ directory

        # Chain tracking (V3)
        chain_path: Serialized operator chain path
        source_index: Multi-source index (None for single source)

        # Context
        pipeline_id: Parent pipeline ID (e.g., "0001_pls_abc123")
        branch_path: Branch hierarchy as list of indices (empty = pre-branch)
        step_index: Logical step index within execution
        substep_index: Index within substep (for [model1, model2])
        fold_id: CV fold identifier (None = shared across folds)

        # Classification
        artifact_type: Type classification (model, transformer, etc.)
        class_name: Python class name (e.g., "PLSRegression")
        custom_name: User-defined name for the artifact

        # Dependencies
        depends_on: List of artifact_ids this artifact depends on

        # Serialization
        format: Serialization format (joblib, pickle, keras, etc.)
        format_version: Library version string
        nirs4all_version: nirs4all version that created this artifact
        size_bytes: Size of serialized binary in bytes
        created_at: ISO timestamp of creation

        # Metadata
        params: Hyperparameters for models
        meta_config: Configuration for meta-models
        version: Schema version (3 for V3)
    """

    # Identification
    artifact_id: str
    content_hash: str

    # Location
    path: str

    # Chain tracking (V3)
    chain_path: str = ""
    source_index: Optional[int] = None

    # Context
    pipeline_id: str = ""
    branch_path: List[int] = field(default_factory=list)
    step_index: int = 0
    substep_index: Optional[int] = None
    fold_id: Optional[int] = None

    # Classification
    artifact_type: ArtifactType = ArtifactType.MODEL
    class_name: str = ""
    custom_name: str = ""

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Serialization
    format: str = "joblib"
    format_version: str = ""
    nirs4all_version: str = ""

    # Metadata
    size_bytes: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    params: Dict[str, Any] = field(default_factory=dict)

    # Meta-model specific
    meta_config: Optional[MetaModelConfig] = None

    # Schema version
    version: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization.

        Handles enum conversion and nested dataclass serialization.

        Returns:
            Dictionary suitable for YAML safe_dump
        """
        data = {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "path": self.path,
            "chain_path": self.chain_path,
            "source_index": self.source_index,
            "pipeline_id": self.pipeline_id,
            "branch_path": self.branch_path,
            "step_index": self.step_index,
            "substep_index": self.substep_index,
            "fold_id": self.fold_id,
            "artifact_type": str(self.artifact_type),
            "class_name": self.class_name,
            "custom_name": self.custom_name,
            "depends_on": self.depends_on,
            "format": self.format,
            "format_version": self.format_version,
            "nirs4all_version": self.nirs4all_version,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "params": self.params,
            "version": self.version,
        }

        if self.meta_config is not None:
            data["meta_config"] = self.meta_config.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRecord":
        """Create ArtifactRecord from dictionary.

        Args:
            data: Dictionary from YAML manifest

        Returns:
            ArtifactRecord instance
        """
        # Handle artifact_type enum
        artifact_type_value = data.get("artifact_type", "model")
        if isinstance(artifact_type_value, str):
            artifact_type = ArtifactType(artifact_type_value)
        else:
            artifact_type = artifact_type_value

        # Handle meta_config
        meta_config = None
        if "meta_config" in data and data["meta_config"] is not None:
            meta_config = MetaModelConfig.from_dict(data["meta_config"])

        return cls(
            artifact_id=data.get("artifact_id", ""),
            content_hash=data.get("content_hash", ""),
            path=data.get("path", ""),
            chain_path=data.get("chain_path", ""),
            source_index=data.get("source_index"),
            pipeline_id=data.get("pipeline_id", ""),
            branch_path=data.get("branch_path", []),
            step_index=data.get("step_index", 0),
            substep_index=data.get("substep_index"),
            fold_id=data.get("fold_id"),
            artifact_type=artifact_type,
            class_name=data.get("class_name", ""),
            custom_name=data.get("custom_name", ""),
            depends_on=data.get("depends_on", []),
            format=data.get("format", "joblib"),
            format_version=data.get("format_version", ""),
            nirs4all_version=data.get("nirs4all_version", ""),
            size_bytes=data.get("size_bytes", 0),
            created_at=data.get("created_at", ""),
            params=data.get("params", {}),
            meta_config=meta_config,
            version=data.get("version", 3),
        )

    @property
    def short_hash(self) -> str:
        """Get short version of content hash for filenames.

        Returns:
            First 12 characters of hash (after sha256: prefix if present)
        """
        hash_value = self.content_hash
        if hash_value.startswith("sha256:"):
            hash_value = hash_value[7:]
        return hash_value[:12]

    @property
    def chain_hash(self) -> str:
        """Get chain hash from artifact ID (V3 format).

        Returns:
            Chain hash portion of the artifact ID, or empty if not V3 format
        """
        if "$" in self.artifact_id:
            # V3 format: pipeline$hash:fold
            rest = self.artifact_id.split("$", 1)[1]
            return rest.rsplit(":", 1)[0]
        return ""

    @property
    def is_branch_specific(self) -> bool:
        """Check if artifact is branch-specific.

        Returns:
            True if artifact belongs to a specific branch path
        """
        return len(self.branch_path) > 0

    @property
    def is_fold_specific(self) -> bool:
        """Check if artifact is fold-specific.

        Returns:
            True if artifact belongs to a specific CV fold
        """
        return self.fold_id is not None

    @property
    def is_source_specific(self) -> bool:
        """Check if artifact is source-specific.

        Returns:
            True if artifact belongs to a specific source in multi-source
        """
        return self.source_index is not None

    @property
    def is_meta_model(self) -> bool:
        """Check if artifact is a meta-model.

        Returns:
            True if artifact is a stacking meta-model
        """
        return self.artifact_type == ArtifactType.META_MODEL

    def get_branch_path_str(self) -> str:
        """Get branch path as string.

        Returns:
            Colon-separated branch indices or empty string
        """
        if not self.branch_path:
            return ""
        return ":".join(str(b) for b in self.branch_path)

    def get_fold_str(self) -> str:
        """Get fold ID as string.

        Returns:
            Fold ID as string or "all" for shared artifacts
        """
        return str(self.fold_id) if self.fold_id is not None else "all"

    def matches_context(
        self,
        step_index: Optional[int] = None,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        fold_id: Optional[int] = None,
    ) -> bool:
        """Check if artifact matches a given context.

        Args:
            step_index: Step to match (None = any)
            branch_path: Branch path to match (None = any)
            source_index: Source index to match (None = any)
            fold_id: Fold ID to match (None = any)

        Returns:
            True if artifact matches all specified filters
        """
        if step_index is not None and self.step_index != step_index:
            return False
        if branch_path is not None and self.branch_path != branch_path:
            return False
        if source_index is not None and self.source_index != source_index:
            return False
        if fold_id is not None and self.fold_id != fold_id:
            return False
        return True

    def __repr__(self) -> str:
        name_part = self.custom_name if self.custom_name else self.class_name
        return (
            f"ArtifactRecord(id={self.artifact_id!r}, "
            f"type={self.artifact_type.value}, "
            f"name={name_part!r})"
        )

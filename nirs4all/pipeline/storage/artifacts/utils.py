"""
Utility functions for artifact identification and path handling (V3).

This module provides utility functions for the V3 artifact system,
primarily for file path handling, content hashing, and artifact ID
utilities.

For the core V3 artifact ID functions (compute_chain_hash, generate_artifact_id_v3,
parse_artifact_id_v3, is_v3_artifact_id), use the operator_chain module directly.

V3 Artifact ID Format:
    "{pipeline_id}${chain_hash}:{fold_id}"

Examples:
    - "0001_pls$a1b2c3d4e5f6:all"  - Shared artifact
    - "0001_pls$7f8e9d0c1b2a:0"    - Fold 0 artifact
    - "0001_pls$3c4d5e6f7a8b:1"    - Fold 1 artifact
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Import core V3 functions from operator_chain to avoid duplication
from nirs4all.pipeline.storage.artifacts.operator_chain import (
    compute_chain_hash,
    generate_artifact_id_v3,
    parse_artifact_id_v3,
    is_v3_artifact_id,
)


@dataclass
class ExecutionPath:
    """Represents the execution context for an artifact (V3).

    Captures all context needed to uniquely identify an artifact
    within a pipeline execution.

    Attributes:
        pipeline_id: Pipeline identifier (e.g., "0001_pls_abc123")
        chain_path: Full operator chain path string
        branch_path: List of branch indices for nested branching
        step_index: Logical step number within current branch
        source_index: Multi-source index (None for single source)
        fold_id: CV fold identifier (None for shared artifacts)
        substep_index: Substep index (for [model1, model2])
    """

    pipeline_id: str
    chain_path: str = ""
    branch_path: List[int] = None
    step_index: int = 0
    source_index: Optional[int] = None
    fold_id: Optional[int] = None
    substep_index: Optional[int] = None

    def __post_init__(self):
        if self.branch_path is None:
            self.branch_path = []

    def to_artifact_id(self) -> str:
        """Convert execution path to V3 artifact ID string.

        Returns:
            V3 Artifact ID in format "{pipeline_id}${chain_hash}:{fold_id}"
        """
        chain_hash = compute_chain_hash(self.chain_path)
        fold_str = str(self.fold_id) if self.fold_id is not None else "all"
        return f"{self.pipeline_id}${chain_hash}:{fold_str}"

    @classmethod
    def from_artifact_id_v3(cls, artifact_id: str, chain_path: str = "") -> "ExecutionPath":
        """Create ExecutionPath from V3 artifact ID string.

        Args:
            artifact_id: V3 artifact ID to parse
            chain_path: Full chain path (required for complete reconstruction)

        Returns:
            ExecutionPath instance
        """
        pipeline_id, chain_hash, fold_id = parse_artifact_id_v3(artifact_id)
        return cls(
            pipeline_id=pipeline_id,
            chain_path=chain_path,
            fold_id=fold_id,
        )


def parse_artifact_id(
    artifact_id: str
) -> Tuple[str, List[int], int, Optional[int], Optional[int]]:
    """Parse an artifact ID into its components (V3 only).

    V3 format: {pipeline_id}${chain_hash}:{fold_id}

    Args:
        artifact_id: V3 artifact ID to parse

    Returns:
        Tuple of (pipeline_id, branch_path, step_index, fold_id, sub_index)
        For V3: step_index will be 0, branch_path empty (use ArtifactRecord for full info)

    Raises:
        ValueError: If artifact ID format is not V3
    """
    if not is_v3_artifact_id(artifact_id):
        raise ValueError(
            f"Invalid artifact ID format: {artifact_id!r}. "
            f"Expected V3 format: pipeline_id$chain_hash:fold_id. "
            f"V2 artifact format is no longer supported."
        )

    pipeline_id, chain_hash, fold_id = parse_artifact_id_v3(artifact_id)
    # For V3, detailed info (step, branch, substep) is in ArtifactRecord
    return pipeline_id, [], 0, fold_id, None


def generate_filename(
    artifact_type: str,
    class_name: str,
    content_hash: str,
    extension: str = "joblib"
) -> str:
    """Generate artifact filename from components.

    New format: <type>_<class>_<short_hash>.<ext>

    Args:
        artifact_type: Artifact type (model, transformer, etc.)
        class_name: Python class name
        content_hash: Full SHA256 hash (will be truncated)
        extension: File extension (default: joblib)

    Returns:
        Filename string

    Examples:
        >>> generate_filename("model", "PLSRegression", "abc123def456")
        "model_PLSRegression_abc123def456.joblib"
    """
    # Use first 12 chars of hash (after prefix if present)
    hash_value = content_hash
    if hash_value.startswith("sha256:"):
        hash_value = hash_value[7:]
    short_hash = hash_value[:12]

    return f"{artifact_type}_{class_name}_{short_hash}.{extension}"


def parse_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Parse artifact filename into components.

    Handles new format: <type>_<class>_<short_hash>.<ext>
    Also handles legacy format: <class>_<short_hash>.<ext>

    Args:
        filename: Filename to parse

    Returns:
        Tuple of (artifact_type, class_name, short_hash) or None if invalid
    """
    # Remove extension
    name = Path(filename).stem

    # Try new format: type_class_hash
    parts = name.split("_")

    if len(parts) >= 3:
        # New format
        artifact_type = parts[0]
        short_hash = parts[-1]
        class_name = "_".join(parts[1:-1])
        return artifact_type, class_name, short_hash
    elif len(parts) == 2:
        # Legacy format: class_hash (no type prefix)
        class_name = parts[0]
        short_hash = parts[1]
        return "", class_name, short_hash

    return None


def compute_content_hash(content: bytes) -> str:
    """Compute SHA256 hash of binary content.

    Args:
        content: Binary content to hash

    Returns:
        Full SHA256 hash with "sha256:" prefix
    """
    hash_value = hashlib.sha256(content).hexdigest()
    return f"sha256:{hash_value}"


def get_short_hash(content_hash: str, length: int = 12) -> str:
    """Extract short hash from full content hash.

    Args:
        content_hash: Full hash (with or without sha256: prefix)
        length: Number of characters to return (default: 12)

    Returns:
        Short hash string
    """
    hash_value = content_hash
    if hash_value.startswith("sha256:"):
        hash_value = hash_value[7:]
    return hash_value[:length]


def get_binaries_path(workspace: Path, dataset: str) -> Path:
    """Get the centralized binaries directory for a dataset.

    New architecture stores artifacts at workspace/binaries/<dataset>/

    Args:
        workspace: Workspace root path
        dataset: Dataset name

    Returns:
        Path to binaries directory
    """
    return workspace / "binaries" / dataset


def validate_artifact_id(artifact_id: str) -> bool:
    """Validate artifact ID format (V3 only).

    Args:
        artifact_id: Artifact ID to validate

    Returns:
        True if valid V3 format, False otherwise
    """
    if not is_v3_artifact_id(artifact_id):
        return False
    try:
        parse_artifact_id_v3(artifact_id)
        return True
    except (ValueError, IndexError):
        return False


def extract_pipeline_id_from_artifact_id(artifact_id: str) -> str:
    """Extract pipeline ID from artifact ID (V2 or V3).

    Args:
        artifact_id: Full artifact ID

    Returns:
        Pipeline ID component
    """
    if is_v3_artifact_id(artifact_id):
        return artifact_id.split("$")[0]
    return artifact_id.split(":")[0]


def extract_fold_id_from_artifact_id(artifact_id: str) -> Optional[int]:
    """Extract fold ID from artifact ID (V2 or V3).

    Args:
        artifact_id: Full artifact ID

    Returns:
        Fold ID or None if "all"
    """
    if is_v3_artifact_id(artifact_id):
        _, _, fold_id = parse_artifact_id_v3(artifact_id)
        return fold_id
    raise ValueError(f"V2 artifact format not supported: {artifact_id}")


def artifact_id_matches_context(
    artifact_id: str,
    pipeline_id: Optional[str] = None,
    branch_path: Optional[List[int]] = None,
    step_index: Optional[int] = None,
    fold_id: Optional[int] = None
) -> bool:
    """Check if a V3 artifact ID matches a given context.

    Partial matching is supported - only specified parameters are checked.
    Note: branch_path and step_index matching requires ArtifactRecord access.

    Args:
        artifact_id: V3 artifact ID to check
        pipeline_id: Expected pipeline ID (None = don't check)
        branch_path: Expected branch path (ignored for V3 - use ArtifactRecord)
        step_index: Expected step index (ignored for V3 - use ArtifactRecord)
        fold_id: Expected fold ID (None = don't check)

    Returns:
        True if artifact matches specified criteria, False otherwise
    """
    if not is_v3_artifact_id(artifact_id):
        return False  # V2 not supported

    try:
        aid_pipeline, _, aid_fold = parse_artifact_id_v3(artifact_id)
        if pipeline_id is not None and aid_pipeline != pipeline_id:
            return False
        if fold_id is not None and aid_fold != fold_id:
            return False
        # branch_path and step_index require ArtifactRecord for V3
        return True
    except ValueError:
        return False

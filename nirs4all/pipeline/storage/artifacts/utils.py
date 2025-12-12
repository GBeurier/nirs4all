"""
Utility functions for artifact identification and path handling.

This module provides functions for generating and parsing deterministic
artifact IDs based on execution path. The execution path captures the
full context of artifact creation:

- pipeline_id: Which pipeline created this artifact
- branch_path: Hierarchy of branch indices for nested branching
- step_index: Logical step number within the branch
- fold_id: CV fold for per-fold artifacts

Artifact ID Format:
    "{pipeline_id}:{branch_path}:{step_index}:{fold_id}"

Examples:
    - "0001_pls:3:all"         - Step 3, no branch, shared across folds
    - "0001_pls:0:3:0"         - Branch 0, step 3, fold 0
    - "0001_pls:0:2:3:all"     - Branch 0→2, step 3, shared across folds
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass
class ExecutionPath:
    """Represents the execution context for an artifact.

    Captures all context needed to uniquely identify an artifact
    within a pipeline execution.

    Attributes:
        pipeline_id: Pipeline identifier (e.g., "0001_pls_abc123")
        branch_path: List of branch indices for nested branching
        step_index: Logical step number within current branch
        fold_id: CV fold identifier (None for shared artifacts)
    """

    pipeline_id: str
    branch_path: List[int]
    step_index: int
    fold_id: Optional[int] = None

    def to_artifact_id(self) -> str:
        """Convert execution path to artifact ID string.

        Returns:
            Artifact ID in format "{pipeline_id}:{branch_path}:{step_index}:{fold_id}"
        """
        return generate_artifact_id(
            pipeline_id=self.pipeline_id,
            branch_path=self.branch_path,
            step_index=self.step_index,
            fold_id=self.fold_id
        )

    @classmethod
    def from_artifact_id(cls, artifact_id: str) -> "ExecutionPath":
        """Create ExecutionPath from artifact ID string.

        Args:
            artifact_id: Artifact ID to parse

        Returns:
            ExecutionPath instance
        """
        pipeline_id, branch_path, step_index, fold_id, _sub_index = parse_artifact_id(artifact_id)
        return cls(
            pipeline_id=pipeline_id,
            branch_path=branch_path,
            step_index=step_index,
            fold_id=fold_id
        )


def generate_artifact_id(
    pipeline_id: str,
    branch_path: List[int],
    step_index: int,
    fold_id: Optional[int] = None,
    sub_index: Optional[int] = None
) -> str:
    """Generate a deterministic artifact ID from execution context.

    The artifact ID uniquely identifies an artifact within a pipeline
    execution, encoding the full execution path.

    Format: "{pipeline_id}:{branch_path_str}:{step_index}:{fold_str}"
    Or with sub_index: "{pipeline_id}:{branch_path_str}:{step_index}.{sub_index}:{fold_str}"

    Args:
        pipeline_id: Pipeline identifier (e.g., "0001_pls_abc123")
        branch_path: List of branch indices (e.g., [0, 2] for nested branching)
        step_index: Step number within the current branch
        fold_id: CV fold identifier (None for shared artifacts)
        sub_index: Sub-operation index for multiple artifacts at same step

    Returns:
        Artifact ID string

    Examples:
        >>> generate_artifact_id("0001_pls", [], 3, None)
        "0001_pls:3:all"
        >>> generate_artifact_id("0001_pls", [0], 3, 0)
        "0001_pls:0:3:0"
        >>> generate_artifact_id("0001_pls", [0, 2], 3, None)
        "0001_pls:0:2:3:all"
        >>> generate_artifact_id("0001_pls", [], 3, None, sub_index=0)
        "0001_pls:3.0:all"
    """
    # Build components
    parts = [pipeline_id]

    # Add branch path components
    for branch_idx in branch_path:
        parts.append(str(branch_idx))

    # Add step index (with optional sub_index)
    if sub_index is not None:
        parts.append(f"{step_index}.{sub_index}")
    else:
        parts.append(str(step_index))

    # Add fold ID
    fold_str = str(fold_id) if fold_id is not None else "all"
    parts.append(fold_str)

    return ":".join(parts)


def parse_artifact_id(
    artifact_id: str
) -> Tuple[str, List[int], int, Optional[int], Optional[int]]:
    """Parse an artifact ID into its components.

    Args:
        artifact_id: Artifact ID to parse

    Returns:
        Tuple of (pipeline_id, branch_path, step_index, fold_id, sub_index)

    Raises:
        ValueError: If artifact ID format is invalid

    Examples:
        >>> parse_artifact_id("0001_pls:3:all")
        ("0001_pls", [], 3, None, None)
        >>> parse_artifact_id("0001_pls:0:3:0")
        ("0001_pls", [0], 3, 0, None)
        >>> parse_artifact_id("0001_pls:0:2:3:all")
        ("0001_pls", [0, 2], 3, None, None)
        >>> parse_artifact_id("0001_pls:3.1:all")
        ("0001_pls", [], 3, None, 1)
    """
    parts = artifact_id.split(":")

    if len(parts) < 3:
        raise ValueError(
            f"Invalid artifact ID format: {artifact_id!r}. "
            "Expected at least 3 parts: pipeline_id:step_index:fold_id"
        )

    # First part is always pipeline_id
    pipeline_id = parts[0]

    # Last part is always fold_id
    fold_str = parts[-1]
    fold_id = None if fold_str == "all" else int(fold_str)

    # Second to last is step_index (may include sub_index as step.sub)
    step_str = parts[-2]
    sub_index = None
    if "." in step_str:
        step_part, sub_part = step_str.split(".", 1)
        step_index = int(step_part)
        sub_index = int(sub_part)
    else:
        step_index = int(step_str)

    # Everything in between is branch_path
    branch_path = []
    if len(parts) > 3:
        branch_parts = parts[1:-2]
        branch_path = [int(b) for b in branch_parts]

    return pipeline_id, branch_path, step_index, fold_id, sub_index


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
    """Validate artifact ID format.

    Args:
        artifact_id: Artifact ID to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_artifact_id(artifact_id)
        return True
    except (ValueError, IndexError):
        return False


def extract_pipeline_id_from_artifact_id(artifact_id: str) -> str:
    """Extract pipeline ID from artifact ID.

    Args:
        artifact_id: Full artifact ID

    Returns:
        Pipeline ID component
    """
    return artifact_id.split(":")[0]


def artifact_id_matches_context(
    artifact_id: str,
    pipeline_id: Optional[str] = None,
    branch_path: Optional[List[int]] = None,
    step_index: Optional[int] = None,
    fold_id: Optional[int] = None
) -> bool:
    """Check if an artifact ID matches a given context.

    Partial matching is supported - only specified parameters are checked.

    Args:
        artifact_id: Artifact ID to check
        pipeline_id: Expected pipeline ID (None = don't check)
        branch_path: Expected branch path (None = don't check)
        step_index: Expected step index (None = don't check)
        fold_id: Expected fold ID (None = don't check)

    Returns:
        True if artifact matches all specified criteria
    """
    try:
        aid_pipeline, aid_branch, aid_step, aid_fold, _sub = parse_artifact_id(artifact_id)
    except ValueError:
        return False

    if pipeline_id is not None and aid_pipeline != pipeline_id:
        return False

    if branch_path is not None and aid_branch != branch_path:
        return False

    if step_index is not None and aid_step != step_index:
        return False

    if fold_id is not None and aid_fold != fold_id:
        return False

    return True

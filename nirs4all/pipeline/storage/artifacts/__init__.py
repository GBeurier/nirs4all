"""Artifact management module.

This module provides the v2 artifacts system with:
- ArtifactRecord: Complete artifact metadata dataclass
- ArtifactType: Enum for artifact classification
- ArtifactRegistry: Central registry for artifact management
- ArtifactLoader: Load artifacts by ID or execution context
- Utility functions for ID generation and path handling
"""

from .types import ArtifactRecord, ArtifactType, MetaModelConfig
from .artifact_registry import ArtifactRegistry, DependencyGraph
from .artifact_loader import ArtifactLoader
from .utils import (
    ExecutionPath,
    generate_artifact_id,
    parse_artifact_id,
    generate_filename,
    parse_filename,
    compute_content_hash,
    get_short_hash,
    get_binaries_path,
    validate_artifact_id,
)

__all__ = [
    # v2 types
    'ArtifactRecord',
    'ArtifactType',
    'MetaModelConfig',
    # v2 registry
    'ArtifactRegistry',
    'DependencyGraph',
    # v2 loader
    'ArtifactLoader',
    # v2 utilities
    'ExecutionPath',
    'generate_artifact_id',
    'parse_artifact_id',
    'generate_filename',
    'parse_filename',
    'compute_content_hash',
    'get_short_hash',
    'get_binaries_path',
    'validate_artifact_id',
]

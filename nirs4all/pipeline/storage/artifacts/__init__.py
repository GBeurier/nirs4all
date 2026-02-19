"""Artifact management module (V3).

This module provides the V3 artifacts system with:
- ArtifactRecord: Complete artifact metadata dataclass with chain tracking
- ArtifactType: Enum for artifact classification
- ArtifactRegistry: Central registry for artifact management
- ArtifactLoader: Load artifacts by ID or execution context
- OperatorNode/OperatorChain: V3 operator path tracking
- Utility functions for ID generation and path handling
"""

from .artifact_loader import ArtifactLoader
from .artifact_registry import ArtifactRegistry, DependencyGraph
from .operator_chain import (
    OperatorChain,
    OperatorNode,
    compute_chain_hash,
    generate_artifact_id_v3,
    is_v3_artifact_id,
    parse_artifact_id_v3,
)
from .types import ArtifactRecord, ArtifactType, MetaModelConfig
from .utils import (
    ExecutionPath,
    artifact_id_matches_context,
    compute_content_hash,
    extract_fold_id_from_artifact_id,
    extract_pipeline_id_from_artifact_id,
    generate_filename,
    get_binaries_path,
    get_short_hash,
    parse_artifact_id,
    parse_filename,
    validate_artifact_id,
)

__all__ = [
    # V3 types
    'ArtifactRecord',
    'ArtifactType',
    'MetaModelConfig',
    # V3 operator chain
    'OperatorNode',
    'OperatorChain',
    'compute_chain_hash',
    'generate_artifact_id_v3',
    'parse_artifact_id_v3',
    'is_v3_artifact_id',
    # Registry
    'ArtifactRegistry',
    'DependencyGraph',
    # Loader
    'ArtifactLoader',
    # Utilities
    'ExecutionPath',
    'parse_artifact_id',
    'generate_filename',
    'parse_filename',
    'compute_content_hash',
    'get_short_hash',
    'get_binaries_path',
    'validate_artifact_id',
    'extract_pipeline_id_from_artifact_id',
    'extract_fold_id_from_artifact_id',
    'artifact_id_matches_context',
]

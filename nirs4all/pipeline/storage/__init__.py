"""Storage and persistence layer for pipeline artifacts and results.

This module handles all I/O operations including:
- DuckDB-backed workspace storage (WorkspaceStore)
- Chain construction from execution traces (ChainBuilder)
- Chain replay for prediction from stored chains
- Pipeline artifact storage (binary models, scalers, etc.)
- Pipeline template library
"""

from .array_store import ArrayStore
from .chain_builder import ChainBuilder
from .chain_replay import replay_chain
from .library import PipelineLibrary
from .migration import MigrationReport, migrate_arrays_to_parquet, verify_migrated_store
from .workspace_store import WorkspaceStore

__all__ = [
    'ArrayStore',
    'MigrationReport',
    'WorkspaceStore',
    'ChainBuilder',
    'migrate_arrays_to_parquet',
    'replay_chain',
    'verify_migrated_store',
    'PipelineLibrary',
]

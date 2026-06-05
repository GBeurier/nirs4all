"""Storage and persistence layer for pipeline artifacts and results.

This module handles all I/O operations including:
- SQLite-backed workspace storage (WorkspaceStore)
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

# Inversion of control: register WorkspaceStore as the data layer's store backend so
# nirs4all.data.predictions can open stores from a path without importing the pipeline
# layer. This keeps the dependency direction correct (pipeline -> data); the data layer
# only holds a TYPE_CHECKING reference to WorkspaceStore. See Predictions.register_store_backend.
from nirs4all.data.predictions import Predictions as _Predictions

_Predictions.register_store_backend(WorkspaceStore)
del _Predictions

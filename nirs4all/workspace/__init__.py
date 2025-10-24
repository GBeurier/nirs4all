"""
Workspace management for nirs4all.

This module provides the workspace architecture for organizing experimental runs,
library storage, catalog management, and exports.
"""

from .workspace_manager import WorkspaceManager
from .run_manager import RunManager
from .library_manager import LibraryManager
from .schemas import PipelineEntry, RunConfig, RunSummary

__all__ = [
    "WorkspaceManager",
    "RunManager",
    "LibraryManager",
    "PipelineEntry",
    "RunConfig",
    "RunSummary",
]

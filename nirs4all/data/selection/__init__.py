"""
Selection module for dataset configuration.

This module provides flexible column and row selection for dataset loading,
supporting multiple selection syntaxes (index, name, range, regex, exclusion).

Classes:
    ColumnSelector: Select columns from a DataFrame using various methods
    RowSelector: Select rows from a DataFrame using various methods
    SampleLinker: Link samples across multiple files by key column
    RoleAssigner: Assign columns to data roles (features, targets, metadata)
"""

from .column_selector import ColumnSelector, ColumnSelectionError
from .row_selector import RowSelector, RowSelectionError
from .sample_linker import SampleLinker, LinkingError
from .role_assigner import RoleAssigner, RoleAssignmentError

__all__ = [
    "ColumnSelector",
    "ColumnSelectionError",
    "RowSelector",
    "RowSelectionError",
    "SampleLinker",
    "LinkingError",
    "RoleAssigner",
    "RoleAssignmentError",
]

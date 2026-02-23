"""
Selection module for dataset configuration.

This module provides flexible column and row selection for dataset loading,
supporting multiple selection syntaxes (index, name, range, regex, exclusion).

Classes:
    ColumnSelector: Select columns from a DataFrame using various methods
    RowSelector: Select rows from a DataFrame using various methods
    SampleLinker: Link samples across multiple files by key column
    RoleAssigner: Assign columns to data roles (features, targets, metadata)

Functions:
    random_sample: Select random sample indices from an array
    stratified_sample: Select stratified sample indices based on y-value bins
    kmeans_sample: Select representative samples via MiniBatchKMeans clustering
"""

from .column_selector import ColumnSelectionError, ColumnSelector
from .role_assigner import RoleAssigner, RoleAssignmentError
from .row_selector import RowSelectionError, RowSelector
from .sample_linker import LinkingError, SampleLinker
from .sampling import kmeans_sample, random_sample, stratified_sample

__all__ = [
    "ColumnSelector",
    "ColumnSelectionError",
    "RowSelector",
    "RowSelectionError",
    "SampleLinker",
    "LinkingError",
    "RoleAssigner",
    "RoleAssignmentError",
    "random_sample",
    "stratified_sample",
    "kmeans_sample",
]

"""
Partition module for dataset configuration.

This module provides flexible partition assignment for dataset loading,
supporting static, column-based, percentage-based, and index-based partition methods.

Classes:
    PartitionAssigner: Assign rows to train/test/predict partitions
    PartitionError: Raised when partition assignment fails

Supported partition methods:
    - Static: Assign entire file to a partition
    - Column-based: Partition based on column values
    - Percentage-based: Split by percentage with optional shuffle/stratify
    - Index-based: Explicit index lists or external files
"""

from .partition_assigner import (
    PartitionAssigner,
    PartitionError,
    PartitionResult,
    PartitionSpec,
)

__all__ = [
    "PartitionAssigner",
    "PartitionError",
    "PartitionResult",
    "PartitionSpec",
]

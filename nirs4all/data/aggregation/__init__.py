"""
Aggregation module for dataset configuration.

This module provides aggregation functionality for sample data,
including custom aggregation functions and column-based aggregation.
"""

from .aggregator import (
    Aggregator,
    AggregationConfig,
    AggregationMethod,
    aggregate_data,
)

__all__ = [
    "Aggregator",
    "AggregationConfig",
    "AggregationMethod",
    "aggregate_data",
]

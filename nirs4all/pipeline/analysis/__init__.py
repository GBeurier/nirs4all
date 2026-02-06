"""Pipeline analysis utilities.

Provides structural analysis of pipeline configurations, including
topology detection for branching, merging, stacking, and model placement.
"""

from .topology import (
    ModelNodeInfo,
    PipelineTopology,
    analyze_topology,
)

__all__ = [
    "PipelineTopology",
    "ModelNodeInfo",
    "analyze_topology",
]

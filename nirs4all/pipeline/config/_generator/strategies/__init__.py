"""Strategy pattern implementation for generator expansion.

This package provides a modular, extensible architecture for handling
different generator node types through the Strategy pattern.

Phase 2 Strategies:
    - RangeStrategy: Handles _range_ nodes for numeric sequences
    - OrStrategy: Handles _or_ nodes with pick/arrange/size semantics

Phase 3 Strategies:
    - LogRangeStrategy: Handles _log_range_ nodes for logarithmic sequences
    - GridStrategy: Handles _grid_ nodes for Cartesian product expansion
    - ZipStrategy: Handles _zip_ nodes for parallel iteration
    - ChainStrategy: Handles _chain_ nodes for sequential ordered expansion
    - SampleStrategy: Handles _sample_ nodes for statistical sampling

Phase 4+ Strategies:
    - CartesianStrategy: Handles _cartesian_ nodes for staged pipeline expansion

Usage:
    from ._generator.strategies import get_strategy, ExpansionStrategy

    strategy = get_strategy(node)
    if strategy:
        result = strategy.expand(node, seed)
        count = strategy.count(node)
"""

from .base import ExpansionStrategy
from .registry import get_strategy, register_strategy

# Phase 2 strategies
from .range_strategy import RangeStrategy
from .or_strategy import OrStrategy

# Phase 3 strategies
from .log_range_strategy import LogRangeStrategy
from .grid_strategy import GridStrategy
from .zip_strategy import ZipStrategy
from .chain_strategy import ChainStrategy
from .sample_strategy import SampleStrategy

# Phase 4+ strategies
from .cartesian_strategy import CartesianStrategy

__all__ = [
    # Base class
    "ExpansionStrategy",
    # Registry functions
    "get_strategy",
    "register_strategy",
    # Phase 2 strategies
    "RangeStrategy",
    "OrStrategy",
    # Phase 3 strategies
    "LogRangeStrategy",
    "GridStrategy",
    "ZipStrategy",
    "ChainStrategy",
    "SampleStrategy",
    # Phase 4+ strategies
    "CartesianStrategy",
]

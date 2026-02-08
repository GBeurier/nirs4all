"""Refit infrastructure for pipeline execution.

This package provides utilities for the refit phase (Phase 2+), where
the winning pipeline configuration from cross-validation is retrained
on the full training set.
"""

from .config_extractor import RefitConfig, extract_per_model_configs, extract_winning_config
from .executor import RefitResult, execute_simple_refit
from .model_selector import PerModelSelection, select_best_per_model
from .stacking_refit import (
    execute_competing_branches_refit,
    execute_separation_refit,
    execute_stacking_refit,
)

__all__ = [
    "PerModelSelection",
    "RefitConfig",
    "RefitResult",
    "execute_competing_branches_refit",
    "execute_separation_refit",
    "execute_simple_refit",
    "execute_stacking_refit",
    "extract_per_model_configs",
    "extract_winning_config",
    "select_best_per_model",
]

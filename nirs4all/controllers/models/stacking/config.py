"""
Configuration classes for training set reconstruction.

This module provides configuration dataclasses used by the TrainingSetReconstructor
for controlling how out-of-fold predictions are collected and processed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class ReconstructorConfig:
    """Configuration for TrainingSetReconstructor.

    Controls internal behavior of the reconstruction process,
    separate from the user-facing StackingConfig.

    Attributes:
        validate_fold_alignment: Whether to validate fold structure consistency.
        validate_sample_coverage: Whether to validate sample coverage.
        log_warnings: Whether to log warnings during reconstruction.
        max_missing_fold_ratio: Maximum ratio of missing folds before error (0.0-1.0).
        allow_partial_sources: Allow reconstruction if some sources have partial data.
        feature_name_pattern: Pattern for generating feature column names.
            Supports: {model_name}, {fold_id}, {classname}, {step_idx}
        excluded_fold_ids: Set of fold_id values to exclude (e.g., {'avg', 'w_avg'}).

    Example:
        >>> config = ReconstructorConfig(
        ...     validate_fold_alignment=True,
        ...     log_warnings=True,
        ...     feature_name_pattern="{model_name}_pred"
        ... )
    """

    validate_fold_alignment: bool = True
    validate_sample_coverage: bool = True
    log_warnings: bool = True
    max_missing_fold_ratio: float = 0.0
    allow_partial_sources: bool = False
    feature_name_pattern: str = "{model_name}_pred"
    excluded_fold_ids: Set[str] = field(default_factory=lambda: {'avg', 'w_avg'})

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.max_missing_fold_ratio <= 1.0:
            raise ValueError(
                f"max_missing_fold_ratio must be between 0 and 1, "
                f"got {self.max_missing_fold_ratio}"
            )

        # Convert excluded_fold_ids to set if needed
        if not isinstance(self.excluded_fold_ids, set):
            self.excluded_fold_ids = set(self.excluded_fold_ids)

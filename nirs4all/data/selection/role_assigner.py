"""
Role assigner for dataset configuration.

This module provides role assignment for DataFrame columns, assigning them
to features (X), targets (Y), or metadata roles with validation to prevent
overlap.

Example:
    >>> assigner = RoleAssigner()
    >>> result = assigner.assign(df, {
    ...     "features": "2:-1",
    ...     "targets": -1,
    ...     "metadata": [0, 1]
    ... })
    >>> print(result.features)  # Features DataFrame
    >>> print(result.targets)   # Targets DataFrame
    >>> print(result.metadata)  # Metadata DataFrame
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .column_selector import ColumnSelector, ColumnSelectionError, ColumnSpec


class RoleAssignmentError(Exception):
    """Raised when role assignment fails."""
    pass


@dataclass
class RoleAssignmentResult:
    """Result of role assignment.

    Attributes:
        features: DataFrame containing feature columns (X).
        targets: DataFrame containing target columns (Y).
        metadata: DataFrame containing metadata columns.
        feature_indices: Indices of feature columns in original DataFrame.
        target_indices: Indices of target columns in original DataFrame.
        metadata_indices: Indices of metadata columns in original DataFrame.
    """
    features: Optional[pd.DataFrame]
    targets: Optional[pd.DataFrame]
    metadata: Optional[pd.DataFrame]
    feature_indices: List[int]
    target_indices: List[int]
    metadata_indices: List[int]

    @property
    def X(self) -> Optional[pd.DataFrame]:
        """Alias for features."""
        return self.features

    @property
    def y(self) -> Optional[pd.DataFrame]:
        """Alias for targets."""
        return self.targets


# Type alias for role specification
RoleSpec = Dict[str, ColumnSpec]


class RoleAssigner:
    """Assign columns to data roles (features, targets, metadata).

    Validates that:
    - No column is assigned to multiple roles
    - At least features are assigned
    - Indices are valid

    Supports the same column selection syntax as ColumnSelector.

    Example:
        >>> assigner = RoleAssigner()
        >>> result = assigner.assign(df, {
        ...     "features": "2:-1",       # All columns except first 2 and last
        ...     "targets": -1,            # Last column
        ...     "metadata": [0, 1]        # First 2 columns
        ... })
    """

    def __init__(
        self,
        case_sensitive: bool = True,
        allow_overlap: bool = False,
    ):
        """Initialize the role assigner.

        Args:
            case_sensitive: Whether column name matching is case-sensitive.
            allow_overlap: Whether to allow columns in multiple roles.
                If True, columns can be duplicated across roles.
                If False (default), raises error on overlap.
        """
        self.selector = ColumnSelector(case_sensitive=case_sensitive)
        self.allow_overlap = allow_overlap

    def assign(
        self,
        df: pd.DataFrame,
        roles: RoleSpec,
    ) -> RoleAssignmentResult:
        """Assign columns to roles.

        Args:
            df: The DataFrame to assign roles from.
            roles: Dictionary mapping role names to column selections.
                Supported roles: "features", "targets", "metadata"
                Also accepts: "x" (alias for features), "y" (alias for targets)

        Returns:
            RoleAssignmentResult with separated DataFrames.

        Raises:
            RoleAssignmentError: If assignment is invalid (overlap, missing features).
        """
        # Normalize role keys
        normalized_roles = self._normalize_role_keys(roles)

        # Get selections for each role
        feature_spec = normalized_roles.get("features")
        target_spec = normalized_roles.get("targets")
        metadata_spec = normalized_roles.get("metadata")

        # Resolve each selection
        feature_indices: List[int] = []
        target_indices: List[int] = []
        metadata_indices: List[int] = []

        try:
            if feature_spec is not None:
                result = self.selector.select(df, feature_spec)
                feature_indices = result.indices
        except ColumnSelectionError as e:
            raise RoleAssignmentError(f"Error selecting features: {e}")

        try:
            if target_spec is not None:
                result = self.selector.select(df, target_spec)
                target_indices = result.indices
        except ColumnSelectionError as e:
            raise RoleAssignmentError(f"Error selecting targets: {e}")

        try:
            if metadata_spec is not None:
                result = self.selector.select(df, metadata_spec)
                metadata_indices = result.indices
        except ColumnSelectionError as e:
            raise RoleAssignmentError(f"Error selecting metadata: {e}")

        # Check for overlap (unless allowed)
        if not self.allow_overlap:
            self._check_overlap(feature_indices, target_indices, metadata_indices, df)

        # Create result DataFrames
        features_df = df.iloc[:, feature_indices].copy() if feature_indices else None
        targets_df = df.iloc[:, target_indices].copy() if target_indices else None
        metadata_df = df.iloc[:, metadata_indices].copy() if metadata_indices else None

        return RoleAssignmentResult(
            features=features_df,
            targets=targets_df,
            metadata=metadata_df,
            feature_indices=feature_indices,
            target_indices=target_indices,
            metadata_indices=metadata_indices,
        )

    def assign_auto(
        self,
        df: pd.DataFrame,
        target_columns: Optional[ColumnSpec] = None,
        metadata_columns: Optional[ColumnSpec] = None,
    ) -> RoleAssignmentResult:
        """Auto-assign roles with specified targets and metadata.

        Features are automatically set to all remaining columns.

        Args:
            df: The DataFrame to assign roles from.
            target_columns: Column selection for targets (Y).
            metadata_columns: Column selection for metadata.

        Returns:
            RoleAssignmentResult with separated DataFrames.
        """
        n_cols = len(df.columns)
        used_indices = set()

        # Resolve targets
        target_indices: List[int] = []
        if target_columns is not None:
            try:
                result = self.selector.select(df, target_columns)
                target_indices = result.indices
                used_indices.update(target_indices)
            except ColumnSelectionError as e:
                raise RoleAssignmentError(f"Error selecting targets: {e}")

        # Resolve metadata
        metadata_indices: List[int] = []
        if metadata_columns is not None:
            try:
                result = self.selector.select(df, metadata_columns)
                metadata_indices = result.indices
                used_indices.update(metadata_indices)
            except ColumnSelectionError as e:
                raise RoleAssignmentError(f"Error selecting metadata: {e}")

        # Features are all remaining columns
        feature_indices = [i for i in range(n_cols) if i not in used_indices]

        if not feature_indices:
            raise RoleAssignmentError(
                "No columns remaining for features after assigning targets and metadata."
            )

        # Create result DataFrames
        features_df = df.iloc[:, feature_indices].copy()
        targets_df = df.iloc[:, target_indices].copy() if target_indices else None
        metadata_df = df.iloc[:, metadata_indices].copy() if metadata_indices else None

        return RoleAssignmentResult(
            features=features_df,
            targets=targets_df,
            metadata=metadata_df,
            feature_indices=feature_indices,
            target_indices=target_indices,
            metadata_indices=metadata_indices,
        )

    def extract_y_from_x(
        self,
        df: pd.DataFrame,
        y_columns: ColumnSpec,
    ) -> RoleAssignmentResult:
        """Extract target columns from a features DataFrame.

        This is useful when Y columns are embedded in the X data.

        Args:
            df: DataFrame containing both features and targets.
            y_columns: Column selection for targets to extract.

        Returns:
            RoleAssignmentResult with features (remaining) and targets (extracted).
        """
        return self.assign_auto(df, target_columns=y_columns)

    def _normalize_role_keys(self, roles: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize role keys to standard names."""
        normalized = {}

        key_mappings = {
            # Features
            "features": "features",
            "feature": "features",
            "x": "features",
            "X": "features",
            "inputs": "features",
            "input": "features",
            # Targets
            "targets": "targets",
            "target": "targets",
            "y": "targets",
            "Y": "targets",
            "outputs": "targets",
            "output": "targets",
            "labels": "targets",
            "label": "targets",
            # Metadata
            "metadata": "metadata",
            "meta": "metadata",
            "group": "metadata",
            "groups": "metadata",
            "m": "metadata",
            "M": "metadata",
            "info": "metadata",
        }

        for key, value in roles.items():
            normalized_key = key_mappings.get(key.lower(), key.lower())
            if normalized_key in ("features", "targets", "metadata"):
                normalized[normalized_key] = value
            # Ignore unknown keys

        return normalized

    def _check_overlap(
        self,
        feature_indices: List[int],
        target_indices: List[int],
        metadata_indices: List[int],
        df: pd.DataFrame,
    ) -> None:
        """Check for column overlap between roles."""
        feature_set = set(feature_indices)
        target_set = set(target_indices)
        metadata_set = set(metadata_indices)

        # Check feature-target overlap
        ft_overlap = feature_set & target_set
        if ft_overlap:
            overlap_cols = [df.columns[i] for i in ft_overlap]
            raise RoleAssignmentError(
                f"Columns assigned to both features and targets: {overlap_cols}. "
                f"Use allow_overlap=True if this is intentional."
            )

        # Check feature-metadata overlap
        fm_overlap = feature_set & metadata_set
        if fm_overlap:
            overlap_cols = [df.columns[i] for i in fm_overlap]
            raise RoleAssignmentError(
                f"Columns assigned to both features and metadata: {overlap_cols}. "
                f"Use allow_overlap=True if this is intentional."
            )

        # Check target-metadata overlap
        tm_overlap = target_set & metadata_set
        if tm_overlap:
            overlap_cols = [df.columns[i] for i in tm_overlap]
            raise RoleAssignmentError(
                f"Columns assigned to both targets and metadata: {overlap_cols}. "
                f"Use allow_overlap=True if this is intentional."
            )

    def validate_roles(
        self,
        df: pd.DataFrame,
        roles: RoleSpec,
    ) -> List[str]:
        """Validate a role specification without performing assignment.

        Args:
            df: The DataFrame to validate against.
            roles: Role specification to validate.

        Returns:
            List of warning messages (empty if no warnings).

        Raises:
            RoleAssignmentError: If role specification is invalid.
        """
        warnings = []

        # Try assignment to validate
        try:
            result = self.assign(df, roles)
        except RoleAssignmentError:
            raise

        # Check for common issues
        if result.features is None or result.features.empty:
            warnings.append("No feature columns assigned.")

        if result.targets is not None and len(result.target_indices) > 10:
            warnings.append(
                f"Many target columns ({len(result.target_indices)}). "
                f"This may indicate a configuration error."
            )

        # Check if all columns are assigned
        n_cols = len(df.columns)
        assigned = set(result.feature_indices + result.target_indices + result.metadata_indices)
        unassigned = n_cols - len(assigned)
        if unassigned > 0:
            warnings.append(f"{unassigned} columns not assigned to any role.")

        return warnings

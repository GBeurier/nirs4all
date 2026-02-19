"""
Tests for RoleAssigner class.

Tests role assignment for features, targets, and metadata.
"""

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.selection import RoleAssigner, RoleAssignmentError


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "sample_name": ["A", "B", "C", "D", "E"],
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_3": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })

@pytest.fixture
def spectral_df():
    """Create a spectral-like DataFrame."""
    # First two columns are metadata, last is target, rest are features
    data = {
        "sample_id": list(range(10)),
        "batch": ["A"] * 5 + ["B"] * 5,
    }
    # Add 100 spectral features
    for i in range(1000, 1100):
        data[str(i)] = np.random.randn(10)
    data["protein_content"] = np.random.rand(10) * 10

    return pd.DataFrame(data)

class TestRoleAssignerBasic:
    """Basic role assignment tests."""

    def test_assign_all_roles(self, sample_df):
        """Test assigning all roles explicitly."""
        assigner = RoleAssigner()
        result = assigner.assign(sample_df, {
            "features": [2, 3, 4],
            "targets": [5],
            "metadata": [0, 1]
        })

        assert result.features is not None
        assert result.targets is not None
        assert result.metadata is not None

        assert list(result.features.columns) == ["feature_1", "feature_2", "feature_3"]
        assert list(result.targets.columns) == ["target"]
        assert list(result.metadata.columns) == ["id", "sample_name"]

    def test_assign_features_only(self, sample_df):
        """Test assigning only features."""
        assigner = RoleAssigner()
        result = assigner.assign(sample_df, {
            "features": "2:-1"
        })

        assert result.features is not None
        assert result.targets is None
        assert result.metadata is None

        assert list(result.features.columns) == ["feature_1", "feature_2", "feature_3"]

    def test_assign_with_range_syntax(self, sample_df):
        """Test assigning roles with range syntax."""
        assigner = RoleAssigner()
        result = assigner.assign(sample_df, {
            "features": "2:-1",
            "targets": -1,
            "metadata": ":2"
        })

        assert len(result.feature_indices) == 3
        assert len(result.target_indices) == 1
        assert len(result.metadata_indices) == 2

class TestRoleAssignerAliases:
    """Tests for role key aliases."""

    def test_x_alias_for_features(self, sample_df):
        """Test 'x' as alias for features."""
        assigner = RoleAssigner()
        result = assigner.assign(sample_df, {
            "x": [2, 3, 4]
        })

        assert result.features is not None
        assert result.X is not None  # Alias property
        assert len(result.feature_indices) == 3

    def test_y_alias_for_targets(self, sample_df):
        """Test 'y' as alias for targets."""
        assigner = RoleAssigner()
        result = assigner.assign(sample_df, {
            "y": -1
        })

        assert result.targets is not None
        assert result.y is not None  # Alias property
        assert len(result.target_indices) == 1

    def test_various_aliases(self, sample_df):
        """Test various aliases for roles."""
        assigner = RoleAssigner()

        # Test different alias combinations
        aliases = [
            {"X": [2], "Y": [5], "M": [0]},
            {"inputs": [2], "outputs": [5], "meta": [0]},
            {"feature": [2], "target": [5], "group": [0]},
        ]

        for alias_set in aliases:
            result = assigner.assign(sample_df, alias_set)
            assert result.features is not None
            assert result.targets is not None
            assert result.metadata is not None

class TestRoleAssignerOverlap:
    """Tests for overlap detection."""

    def test_overlap_raises_by_default(self, sample_df):
        """Test that overlapping assignments raise error."""
        assigner = RoleAssigner()

        with pytest.raises(RoleAssignmentError, match="assigned to both"):
            assigner.assign(sample_df, {
                "features": [0, 1, 2],
                "targets": [2]  # Overlaps with features
            })

    def test_overlap_allowed_when_enabled(self, sample_df):
        """Test that overlap is allowed when enabled."""
        assigner = RoleAssigner(allow_overlap=True)

        result = assigner.assign(sample_df, {
            "features": [0, 1, 2],
            "targets": [2]  # Overlaps with features
        })

        assert 2 in result.feature_indices
        assert 2 in result.target_indices

class TestRoleAssignerAuto:
    """Tests for auto role assignment."""

    def test_assign_auto_with_targets(self, sample_df):
        """Test auto assignment with only targets specified."""
        assigner = RoleAssigner()
        result = assigner.assign_auto(sample_df, target_columns=-1)

        # Features should be all columns except target
        assert len(result.feature_indices) == 5
        assert 5 not in result.feature_indices  # Target column
        assert result.target_indices == [5]

    def test_assign_auto_with_metadata(self, sample_df):
        """Test auto assignment with targets and metadata specified."""
        assigner = RoleAssigner()
        result = assigner.assign_auto(
            sample_df,
            target_columns=-1,
            metadata_columns=[0, 1]
        )

        # Features should be remaining columns
        assert set(result.feature_indices) == {2, 3, 4}
        assert result.target_indices == [5]
        assert result.metadata_indices == [0, 1]

    def test_assign_auto_no_features_left_raises(self, sample_df):
        """Test error when no columns left for features."""
        assigner = RoleAssigner()

        with pytest.raises(RoleAssignmentError, match="No columns remaining"):
            # Assign all columns to targets or metadata
            assigner.assign_auto(
                sample_df,
                target_columns=list(range(6)),
            )

class TestRoleAssignerExtractY:
    """Tests for extracting Y from X."""

    def test_extract_y_from_x(self, sample_df):
        """Test extracting target columns from features."""
        assigner = RoleAssigner()
        result = assigner.extract_y_from_x(sample_df, y_columns=-1)

        assert result.features is not None
        assert result.targets is not None

        # Target column should not be in features
        assert "target" not in result.features.columns
        assert "target" in result.targets.columns

class TestRoleAssignerSpectral:
    """Tests with spectral-like data."""

    def test_assign_spectral_data(self, spectral_df):
        """Test assigning roles in spectral data."""
        assigner = RoleAssigner()
        result = assigner.assign(spectral_df, {
            "metadata": [0, 1],      # sample_id, batch
            "features": "2:-1",       # All spectral columns
            "targets": -1             # protein_content
        })

        assert len(result.metadata_indices) == 2
        assert len(result.feature_indices) == 100  # 100 spectral columns
        assert len(result.target_indices) == 1

    def test_auto_assign_spectral_data(self, spectral_df):
        """Test auto assignment with spectral data."""
        assigner = RoleAssigner()
        result = assigner.assign_auto(
            spectral_df,
            target_columns=-1,
            metadata_columns=":2"
        )

        assert len(result.metadata_indices) == 2
        assert len(result.feature_indices) == 100
        assert len(result.target_indices) == 1

class TestRoleAssignerValidation:
    """Tests for role validation."""

    def test_validate_roles(self, sample_df):
        """Test role validation."""
        assigner = RoleAssigner()
        warnings = assigner.validate_roles(sample_df, {
            "features": "2:-1",
            "targets": -1,
        })

        # Should have warning about unassigned columns
        assert any("not assigned" in w for w in warnings)

    def test_validate_too_many_targets_warning(self, sample_df):
        """Test warning for many target columns."""
        # Create df with many columns
        df = pd.DataFrame(np.random.randn(5, 20))

        assigner = RoleAssigner()
        warnings = assigner.validate_roles(df, {
            "features": [0],
            "targets": list(range(1, 15))  # 14 targets
        })

        assert any("Many target columns" in w for w in warnings)

class TestRoleAssignerCaseSensitivity:
    """Tests for case sensitivity in role assignment."""

    def test_case_insensitive_column_selection(self, sample_df):
        """Test case-insensitive column name selection."""
        assigner = RoleAssigner(case_sensitive=False)
        result = assigner.assign(sample_df, {
            "features": ["FEATURE_1", "FEATURE_2"]
        })

        assert result.feature_indices == [2, 3]
        assert result.features.columns.tolist() == ["feature_1", "feature_2"]

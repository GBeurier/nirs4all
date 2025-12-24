"""
Unit tests for MetadataPartitionerController.

Tests the metadata partitioner branching functionality including:
- Controller matching for metadata_partitioner syntax
- Partition config parsing and validation
- Partition group building with value grouping
- Sample partitioning by metadata column
- min_samples filtering
- Per-branch CV application
- Integration with branch contexts
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from nirs4all.controllers.data.metadata_partitioner import (
    MetadataPartitionerController,
    MetadataPartitionConfig,
    _parse_metadata_partition_config,
    _build_partition_groups,
)
from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext,
    RuntimeContext
)
from nirs4all.pipeline.execution.result import StepOutput
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestMetadataPartitionerControllerMatches:
    """Test MetadataPartitionerController.matches() method."""

    def test_matches_metadata_partitioner_by_keyword(self):
        """Should match when 'by' is 'metadata_partitioner' at step level."""
        step = {"branch": [Mock()], "by": "metadata_partitioner", "column": "site"}
        assert MetadataPartitionerController.matches(step, None, "branch") is True

    def test_matches_metadata_partitioner_in_branch_dict(self):
        """Should match when 'by' is 'metadata_partitioner' inside branch dict."""
        step = {"branch": {"by": "metadata_partitioner", "column": "site", "steps": []}}
        assert MetadataPartitionerController.matches(step, None, "branch") is True

    def test_not_matches_sample_partitioner(self):
        """Should not match sample_partitioner syntax."""
        step = {"branch": {"by": "sample_partitioner", "filter": {}}}
        assert MetadataPartitionerController.matches(step, None, "branch") is False

    def test_not_matches_regular_branch(self):
        """Should not match regular branch syntax."""
        step = {"branch": [["step1"], ["step2"]]}
        assert MetadataPartitionerController.matches(step, None, "branch") is False

    def test_not_matches_named_branch(self):
        """Should not match named branch syntax."""
        step = {"branch": {"snv": ["snv"], "msc": ["msc"]}}
        assert MetadataPartitionerController.matches(step, None, "branch") is False

    def test_not_matches_non_branch_keyword(self):
        """Should not match other keywords."""
        step = {"preprocessing": {"by": "metadata_partitioner"}}
        assert MetadataPartitionerController.matches(step, None, "preprocessing") is False


class TestMetadataPartitionerPriority:
    """Test MetadataPartitionerController priority."""

    def test_priority_value(self):
        """MetadataPartitionerController should have priority 3."""
        assert MetadataPartitionerController.priority == 3


class TestMetadataPartitionConfig:
    """Test MetadataPartitionConfig dataclass."""

    def test_valid_config(self):
        """Create valid config with required parameters."""
        config = MetadataPartitionConfig(
            column="site",
            branch_steps=[Mock()],
        )
        assert config.column == "site"
        assert len(config.branch_steps) == 1
        assert config.min_samples == 1
        assert config.cv is None
        assert config.group_values is None

    def test_config_with_all_params(self):
        """Create config with all parameters."""
        mock_cv = Mock()
        config = MetadataPartitionConfig(
            column="variety",
            branch_steps=[Mock(), Mock()],
            cv=mock_cv,
            min_samples=20,
            group_values={"others": ["C", "D", "E"]},
        )
        assert config.column == "variety"
        assert len(config.branch_steps) == 2
        assert config.cv is mock_cv
        assert config.min_samples == 20
        assert config.group_values == {"others": ["C", "D", "E"]}

    def test_empty_column_raises_error(self):
        """Empty column should raise ValueError."""
        with pytest.raises(ValueError, match="column must be specified"):
            MetadataPartitionConfig(column="", branch_steps=[])

    def test_min_samples_zero_raises_error(self):
        """min_samples < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            MetadataPartitionConfig(column="site", branch_steps=[], min_samples=0)


class TestParseMetadataPartitionConfig:
    """Test _parse_metadata_partition_config function."""

    def test_parse_simple_syntax(self):
        """Parse simple syntax with column at step level."""
        step = {
            "branch": [Mock()],
            "by": "metadata_partitioner",
            "column": "site",
        }
        config = _parse_metadata_partition_config(step)
        assert config.column == "site"
        assert len(config.branch_steps) == 1

    def test_parse_nested_syntax(self):
        """Parse syntax with config inside branch dict."""
        step = {
            "branch": {
                "by": "metadata_partitioner",
                "column": "variety",
                "steps": [Mock(), Mock()],
                "min_samples": 30,
            }
        }
        config = _parse_metadata_partition_config(step)
        assert config.column == "variety"
        assert len(config.branch_steps) == 2
        assert config.min_samples == 30

    def test_parse_with_cv(self):
        """Parse config with CV splitter."""
        mock_cv = Mock()
        step = {
            "branch": [Mock()],
            "by": "metadata_partitioner",
            "column": "site",
            "cv": mock_cv,
        }
        config = _parse_metadata_partition_config(step)
        assert config.cv is mock_cv

    def test_parse_with_group_values(self):
        """Parse config with value grouping."""
        step = {
            "branch": [Mock()],
            "by": "metadata_partitioner",
            "column": "percentage",
            "group_values": {
                "zero": [0],
                "low": [5, 10, 15],
                "high": [20, 25, 30],
            },
        }
        config = _parse_metadata_partition_config(step)
        assert config.group_values == {
            "zero": [0],
            "low": [5, 10, 15],
            "high": [20, 25, 30],
        }

    def test_missing_column_raises_error(self):
        """Missing column should raise ValueError."""
        step = {
            "branch": [Mock()],
            "by": "metadata_partitioner",
        }
        with pytest.raises(ValueError, match="requires 'column' parameter"):
            _parse_metadata_partition_config(step)


class TestBuildPartitionGroups:
    """Test _build_partition_groups function."""

    def test_no_grouping(self):
        """Without group_values, each unique value becomes a partition."""
        unique_values = ["A", "B", "C"]
        groups = _build_partition_groups(unique_values, None)
        assert groups == {"A": ["A"], "B": ["B"], "C": ["C"]}

    def test_with_grouping(self):
        """With group_values, specified values are grouped together."""
        unique_values = ["A", "B", "C", "D", "E"]
        group_values = {"others": ["C", "D", "E"]}
        groups = _build_partition_groups(unique_values, group_values)

        assert "others" in groups
        assert groups["others"] == ["C", "D", "E"]
        # Ungrouped values should have individual partitions
        assert groups["A"] == ["A"]
        assert groups["B"] == ["B"]
        assert "C" not in groups  # C is in "others"
        assert "D" not in groups  # D is in "others"
        assert "E" not in groups  # E is in "others"

    def test_multiple_groups(self):
        """Multiple group_values groups."""
        unique_values = [0, 5, 10, 15, 20, 25, 30]
        group_values = {
            "zero": [0],
            "low": [5, 10, 15],
            "high": [20, 25, 30],
        }
        groups = _build_partition_groups(unique_values, group_values)

        assert groups["zero"] == [0]
        assert groups["low"] == [5, 10, 15]
        assert groups["high"] == [20, 25, 30]
        # No ungrouped values
        assert len(groups) == 3

    def test_numeric_values_as_keys(self):
        """Numeric values should be converted to string keys."""
        unique_values = [1, 2, 3]
        groups = _build_partition_groups(unique_values, None)
        assert groups == {"1": [1], "2": [2], "3": [3]}


class TestBranchTypeDetection:
    """Test BranchType detection for metadata_partitioner."""

    def test_detect_metadata_partitioner_type(self):
        """Should detect METADATA_PARTITIONER branch type."""
        from nirs4all.controllers.models.stacking import detect_branch_type, BranchType

        context = Mock()
        context.custom = {"metadata_partitioner_active": True}
        context.selector = Mock()
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.METADATA_PARTITIONER

    def test_is_disjoint_branch_for_metadata_partitioner(self):
        """Should return True for metadata_partitioner context."""
        from nirs4all.controllers.models.stacking import is_disjoint_branch

        context = Mock()
        context.custom = {
            "metadata_partitioner_active": True,
            "metadata_partition": {
                "column": "site",
                "partition_value": "A",
                "sample_indices": [0, 1, 2],
            },
        }
        context.selector = Mock()
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        assert is_disjoint_branch(context) is True

    def test_is_disjoint_branch_for_preprocessing(self):
        """Should return False for regular preprocessing branch."""
        from nirs4all.controllers.models.stacking import is_disjoint_branch

        context = Mock()
        context.custom = {
            "in_branch_mode": True,
        }
        context.selector = Mock()
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        assert is_disjoint_branch(context) is False


class TestGetDisjointBranchInfo:
    """Test get_disjoint_branch_info function."""

    def test_get_info_for_metadata_partitioner(self):
        """Should return partition info for metadata_partitioner."""
        from nirs4all.controllers.models.stacking import get_disjoint_branch_info

        context = Mock()
        context.custom = {
            "metadata_partitioner_active": True,
            "metadata_partition": {
                "column": "site",
                "partition_value": "site_A",
                "partition_values": ["A"],
                "sample_indices": [0, 1, 2, 3, 4],
                "train_sample_indices": [0, 1, 2],
                "n_samples": 5,
                "n_train_samples": 3,
            },
        }
        context.selector = Mock()
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        info = get_disjoint_branch_info(context)
        assert info is not None
        assert info["partition_type"] == "metadata"
        assert info["column"] == "site"
        assert info["partition_value"] == "site_A"
        assert info["sample_indices"] == [0, 1, 2, 3, 4]
        assert info["n_samples"] == 5
        assert info["n_train_samples"] == 3

    def test_get_info_for_sample_partitioner(self):
        """Should return partition info for sample_partitioner."""
        from nirs4all.controllers.models.stacking import get_disjoint_branch_info

        context = Mock()
        context.custom = {
            "sample_partitioner_active": True,
            "sample_partition": {
                "partition_type": "inliers",
                "sample_indices": [1, 2, 3, 4],
                "n_samples": 4,
                "filter_config": {"method": "y_outlier"},
            },
        }
        context.selector = Mock()
        context.selector.branch_id = 1
        context.selector.branch_path = [1]

        info = get_disjoint_branch_info(context)
        assert info is not None
        assert info["partition_type"] == "sample"
        assert info["partition_value"] == "inliers"
        assert info["sample_indices"] == [1, 2, 3, 4]
        assert info["n_samples"] == 4

    def test_get_info_for_non_disjoint_returns_none(self):
        """Should return None for non-disjoint branch."""
        from nirs4all.controllers.models.stacking import get_disjoint_branch_info

        context = Mock()
        context.custom = {"in_branch_mode": True}
        context.selector = Mock()
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        info = get_disjoint_branch_info(context)
        assert info is None


class TestStackingCompatibility:
    """Test stacking compatibility for metadata_partitioner."""

    def test_is_stacking_compatible_within_partition(self):
        """Should be stacking compatible within same partition."""
        from nirs4all.controllers.models.stacking import is_stacking_compatible

        context = Mock()
        context.custom = {"metadata_partitioner_active": True}
        context.selector = Mock()
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        assert is_stacking_compatible(context) is True

"""Unit tests for SourceBranchController and SourceBranchConfig.

Tests the source branching feature which enables per-source pipeline
execution for multi-source datasets.
"""

import pytest
import numpy as np

from nirs4all.operators.data.merge import (
    SourceBranchConfig,
)
from nirs4all.controllers.data.source_branch import (
    SourceBranchController,
    SourceBranchConfigParser,
)


class TestSourceBranchConfig:
    """Tests for SourceBranchConfig dataclass."""

    def test_auto_mode(self):
        """Test auto mode configuration."""
        config = SourceBranchConfig(source_pipelines="auto")
        assert config.is_auto_mode()
        assert config.merge_after is True
        assert config.merge_strategy == "concat"

    def test_dict_mode_basic(self):
        """Test basic dictionary mode."""
        config = SourceBranchConfig(
            source_pipelines={
                "NIR": ["step1", "step2"],
                "markers": ["step3"],
            }
        )
        assert not config.is_auto_mode()
        assert "NIR" in config.source_pipelines
        assert "markers" in config.source_pipelines
        assert len(config.source_pipelines["NIR"]) == 2

    def test_dict_mode_with_default(self):
        """Test dictionary mode with default pipeline."""
        source_pipelines = {
            "NIR": ["snv"],
            "_default_": ["minmax"],
        }
        config = SourceBranchConfig(source_pipelines=source_pipelines)

        # _default_ should be extracted to default_pipeline
        assert config.default_pipeline == ["minmax"]
        assert "_default_" not in config.source_pipelines

    def test_dict_mode_with_merge_options(self):
        """Test dictionary mode with merge options."""
        source_pipelines = {
            "NIR": ["snv"],
            "_merge_after_": False,
            "_merge_strategy_": "stack",
        }
        config = SourceBranchConfig(source_pipelines=source_pipelines)

        assert config.merge_after is False
        assert config.merge_strategy == "stack"
        assert "_merge_after_" not in config.source_pipelines
        assert "_merge_strategy_" not in config.source_pipelines

    def test_invalid_merge_strategy(self):
        """Test that invalid merge strategy raises error."""
        with pytest.raises(ValueError, match="merge_strategy"):
            SourceBranchConfig(
                source_pipelines="auto",
                merge_strategy="invalid"
            )

    def test_invalid_string_source_pipelines(self):
        """Test that invalid string source_pipelines raises error."""
        with pytest.raises(ValueError, match="'auto'"):
            SourceBranchConfig(source_pipelines="invalid")

    def test_get_pipeline_for_source_auto(self):
        """Test get_pipeline_for_source in auto mode."""
        config = SourceBranchConfig(source_pipelines="auto")

        result = config.get_pipeline_for_source("NIR", 0)
        assert result == []  # Empty list for auto mode (passthrough with isolation)

    def test_get_pipeline_for_source_by_name(self):
        """Test get_pipeline_for_source by name."""
        config = SourceBranchConfig(
            source_pipelines={
                "NIR": ["snv", "savgol"],
                "markers": ["variance_threshold"],
            }
        )

        result = config.get_pipeline_for_source("NIR", 0)
        assert result == ["snv", "savgol"]

        result = config.get_pipeline_for_source("markers", 1)
        assert result == ["variance_threshold"]

    def test_get_pipeline_for_source_by_index(self):
        """Test get_pipeline_for_source by index."""
        config = SourceBranchConfig(
            source_pipelines={
                0: ["snv"],
                1: ["minmax"],
            }
        )

        result = config.get_pipeline_for_source("unknown", 0)
        assert result == ["snv"]

        result = config.get_pipeline_for_source("unknown", 1)
        assert result == ["minmax"]

    def test_get_pipeline_for_source_fallback(self):
        """Test get_pipeline_for_source fallback to default."""
        config = SourceBranchConfig(
            source_pipelines={"NIR": ["snv"]},
            default_pipeline=["minmax"]
        )

        # NIR should get its specific pipeline
        result = config.get_pipeline_for_source("NIR", 0)
        assert result == ["snv"]

        # Unknown source should get default
        result = config.get_pipeline_for_source("markers", 1)
        assert result == ["minmax"]

    def test_get_all_source_mappings_auto(self):
        """Test get_all_source_mappings in auto mode."""
        config = SourceBranchConfig(source_pipelines="auto")
        available = ["NIR", "markers", "Raman"]

        result = config.get_all_source_mappings(available)

        assert len(result) == 3
        assert result["NIR"] == []
        assert result["markers"] == []
        assert result["Raman"] == []

    def test_get_all_source_mappings_dict(self):
        """Test get_all_source_mappings with dict config."""
        config = SourceBranchConfig(
            source_pipelines={
                "NIR": ["snv"],
                "markers": ["variance_threshold"],
            },
            default_pipeline=["minmax"]
        )
        available = ["NIR", "markers", "Raman"]

        result = config.get_all_source_mappings(available)

        assert result["NIR"] == ["snv"]
        assert result["markers"] == ["variance_threshold"]
        assert result["Raman"] == ["minmax"]  # Falls back to default

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = SourceBranchConfig(
            source_pipelines={"NIR": ["snv"]},
            merge_after=False,
            merge_strategy="stack"
        )

        result = config.to_dict()

        assert result["merge_after"] is False
        assert result["merge_strategy"] == "stack"
        assert "source_pipelines" in result

    def test_to_dict_auto(self):
        """Test serialization of auto mode."""
        config = SourceBranchConfig(source_pipelines="auto")

        result = config.to_dict()

        assert result["source_pipelines"] == "auto"


class TestSourceBranchConfigParser:
    """Tests for SourceBranchConfigParser."""

    def test_parse_string_auto(self):
        """Test parsing 'auto' string."""
        config = SourceBranchConfigParser.parse("auto")

        assert isinstance(config, SourceBranchConfig)
        assert config.is_auto_mode()

    def test_parse_string_invalid(self):
        """Test parsing invalid string raises error."""
        with pytest.raises(ValueError, match="Unknown source_branch mode"):
            SourceBranchConfigParser.parse("invalid")

    def test_parse_dict_simple(self):
        """Test parsing simple dictionary."""
        config = SourceBranchConfigParser.parse({
            "NIR": ["snv"],
            "markers": ["minmax"],
        })

        assert isinstance(config, SourceBranchConfig)
        assert config.source_pipelines["NIR"] == ["snv"]
        assert config.source_pipelines["markers"] == ["minmax"]

    def test_parse_dict_with_special_keys(self):
        """Test parsing dictionary with special keys."""
        config = SourceBranchConfigParser.parse({
            "NIR": ["snv"],
            "_default_": ["minmax"],
            "_merge_after_": False,
            "_merge_strategy_": "dict",
        })

        assert config.default_pipeline == ["minmax"]
        assert config.merge_after is False
        assert config.merge_strategy == "dict"
        assert "_default_" not in config.source_pipelines

    def test_parse_dict_single_step(self):
        """Test parsing dictionary with single step (not list)."""
        config = SourceBranchConfigParser.parse({
            "NIR": "snv",  # Single step, not list
        })

        # Should be normalized to list
        assert config.source_pipelines["NIR"] == ["snv"]

    def test_parse_dict_none_value(self):
        """Test parsing dictionary with None value (passthrough)."""
        config = SourceBranchConfigParser.parse({
            "NIR": None,
        })

        # None should become empty list (passthrough)
        assert config.source_pipelines["NIR"] == []

    def test_parse_existing_config(self):
        """Test parsing already parsed SourceBranchConfig."""
        original = SourceBranchConfig(source_pipelines="auto")
        config = SourceBranchConfigParser.parse(original)

        assert config is original  # Should return same instance

    def test_parse_invalid_type(self):
        """Test parsing invalid type raises error."""
        with pytest.raises(ValueError, match="Invalid source_branch config type"):
            SourceBranchConfigParser.parse(123)


class TestSourceBranchController:
    """Tests for SourceBranchController."""

    def test_matches_keyword(self):
        """Test that controller matches 'source_branch' keyword."""
        assert SourceBranchController.matches({}, None, "source_branch")
        assert not SourceBranchController.matches({}, None, "branch")
        assert not SourceBranchController.matches({}, None, "merge")

    def test_use_multi_source(self):
        """Test controller supports multi-source datasets."""
        assert SourceBranchController.use_multi_source() is True

    def test_supports_prediction_mode(self):
        """Test controller supports prediction mode."""
        assert SourceBranchController.supports_prediction_mode() is True

    def test_get_step_names_empty(self):
        """Test _get_step_names with empty list."""
        controller = SourceBranchController()
        result = controller._get_step_names([])
        assert result == ""

    def test_get_step_names_with_classes(self):
        """Test _get_step_names with class instances."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        controller = SourceBranchController()
        result = controller._get_step_names([MinMaxScaler(), StandardScaler()])

        assert "MinMaxScaler" in result
        assert "StandardScaler" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

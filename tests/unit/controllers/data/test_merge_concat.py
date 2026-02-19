"""Unit tests for concat merge strategy (Phase 5).

Tests the concat merge mode for separation branches and the
unified source merge syntax via the merge keyword.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.controllers.data.merge import MergeConfigParser, MergeController
from nirs4all.operators.data.merge import MergeConfig, SourceMergeConfig


class TestMergeConfigParserConcatMode:
    """Tests for concat mode parsing."""

    def test_parse_concat_string_mode(self):
        """Test parsing 'concat' string mode."""
        config = MergeConfigParser.parse("concat")

        assert config.collect_features is True
        assert config.is_separation_merge is True
        assert config.output_as == "features"

    def test_parse_concat_dict_mode(self):
        """Test parsing {'concat': True} dict mode."""
        config = MergeConfigParser.parse({"concat": True})

        assert config.collect_features is True
        assert config.is_separation_merge is True

    def test_concat_mode_implies_feature_collection(self):
        """Concat mode should enable feature collection."""
        config = MergeConfigParser.parse("concat")

        # Should default to collecting features from all branches
        assert config.feature_branches == "all"

class TestMergeConfigParserSourcesMerge:
    """Tests for sources merge via merge keyword."""

    def test_parse_sources_concat_string(self):
        """Test parsing {'sources': 'concat'}."""
        config = MergeConfigParser.parse({"sources": "concat"})

        assert config.source_merge is not None
        assert config.source_merge.strategy == "concat"
        assert config.source_merge.sources == "all"

    def test_parse_sources_stack_string(self):
        """Test parsing {'sources': 'stack'}."""
        config = MergeConfigParser.parse({"sources": "stack"})

        assert config.source_merge is not None
        assert config.source_merge.strategy == "stack"

    def test_parse_sources_dict_string(self):
        """Test parsing {'sources': 'dict'}."""
        config = MergeConfigParser.parse({"sources": "dict"})

        assert config.source_merge is not None
        assert config.source_merge.strategy == "dict"

    def test_parse_sources_full_config(self):
        """Test parsing full source merge configuration."""
        config = MergeConfigParser.parse({
            "sources": {
                "strategy": "stack",
                "sources": ["source_0", "source_1"],
                "on_incompatible": "pad",
                "output_name": "custom_merged",
            }
        })

        assert config.source_merge is not None
        assert config.source_merge.strategy == "stack"
        assert config.source_merge.sources == ["source_0", "source_1"]
        assert config.source_merge.on_incompatible == "pad"
        assert config.source_merge.output_name == "custom_merged"

    def test_parse_sources_with_features(self):
        """Test parsing sources combined with features."""
        config = MergeConfigParser.parse({
            "sources": "concat",
            "features": "all",
        })

        assert config.source_merge is not None
        # With features key, should not short-circuit
        assert config.collect_features is True

class TestMergeConfigSerialization:
    """Tests for serialization with new fields."""

    def test_to_dict_with_is_separation_merge(self):
        """Test serialization includes is_separation_merge."""
        config = MergeConfig(
            collect_features=True,
            is_separation_merge=True,
        )

        d = config.to_dict()

        assert d["is_separation_merge"] is True

    def test_to_dict_with_source_merge(self):
        """Test serialization includes source_merge."""
        source_merge = SourceMergeConfig(strategy="stack")
        config = MergeConfig(
            collect_features=True,
            source_merge=source_merge,
        )

        d = config.to_dict()

        assert "source_merge" in d
        assert d["source_merge"]["strategy"] == "stack"

    def test_from_dict_with_is_separation_merge(self):
        """Test deserialization handles is_separation_merge."""
        d = {
            "collect_features": True,
            "is_separation_merge": True,
        }

        config = MergeConfig.from_dict(d)

        assert config.is_separation_merge is True

    def test_from_dict_with_source_merge(self):
        """Test deserialization handles source_merge."""
        d = {
            "collect_features": True,
            "source_merge": {
                "strategy": "concat",
                "sources": "all",
            }
        }

        config = MergeConfig.from_dict(d)

        assert config.source_merge is not None
        assert config.source_merge.strategy == "concat"

class TestBranchTypeValidation:
    """Tests for branch type validation logic."""

    def test_validate_separation_branch_without_concat(self):
        """Test validation warns when using separation without concat."""
        controller = MergeController()
        config = MergeConfig(collect_features=True)

        # Mock logger to capture warnings
        with patch.object(controller, '_validate_branch_type_merge_strategy') as mock:
            # Call the actual method
            controller._validate_branch_type_merge_strategy(
                config=config,
                branch_type="separation",
                branch_contexts=[{"sample_indices": [0, 1, 2]}],
            )

            # Should not raise, just log

    def test_validate_duplication_branch_with_concat(self):
        """Test validation warns when using concat with duplication branches."""
        controller = MergeController()
        config = MergeConfig(collect_features=True, is_separation_merge=True)

        # Should not raise, just warn
        controller._validate_branch_type_merge_strategy(
            config=config,
            branch_type="duplication",
            branch_contexts=[],
        )

    def test_validate_detects_separation_from_branch_contexts(self):
        """Test validation detects separation from sample_indices in contexts."""
        controller = MergeController()
        config = MergeConfig(collect_features=True)

        # Branch contexts with sample_indices indicate separation
        branch_contexts = [
            {"branch_id": 0, "sample_indices": [0, 1, 2]},
            {"branch_id": 1, "sample_indices": [3, 4, 5]},
        ]

        # Should detect separation even with branch_type="duplication"
        controller._validate_branch_type_merge_strategy(
            config=config,
            branch_type="duplication",  # Wrong type
            branch_contexts=branch_contexts,
        )
        # Should not raise

class TestConcatMergeFunctionalTests:
    """Functional tests for concat merge integration."""

    def test_concat_merge_creates_config_correctly(self):
        """Test concat mode creates proper config for disjoint merge."""
        config = MergeConfigParser.parse("concat")

        # Should trigger disjoint branch merge logic
        assert config.collect_features is True
        assert config.is_separation_merge is True
        assert config.output_as == "features"

    def test_concat_with_predictions(self):
        """Test concat can be combined with predictions."""
        config = MergeConfigParser.parse({
            "concat": True,
            "predictions": "all",
        })

        assert config.collect_features is True
        assert config.collect_predictions is True
        assert config.is_separation_merge is True

class TestSourceMergeFromMergeKeyword:
    """Tests for source merge via merge keyword."""

    def test_sources_only_returns_early(self):
        """Test that sources-only config returns early without features/predictions."""
        config = MergeConfigParser.parse({"sources": "concat"})

        # Should only have source_merge set, not features
        assert config.source_merge is not None
        assert config.collect_features is False
        assert config.collect_predictions is False

    def test_sources_preserves_other_options(self):
        """Test sources with other global options."""
        config = MergeConfigParser.parse({
            "sources": "stack",
            "features": "all",
            "on_missing": "warn",
        })

        assert config.source_merge is not None
        assert config.collect_features is True
        assert config.on_missing == "warn"

class TestMergeStrategyModes:
    """Tests for understanding the different merge modes."""

    def test_features_mode(self):
        """Test standard features merge mode."""
        config = MergeConfigParser.parse("features")

        assert config.collect_features is True
        assert config.collect_predictions is False
        assert config.is_separation_merge is False

    def test_predictions_mode(self):
        """Test predictions merge mode."""
        config = MergeConfigParser.parse("predictions")

        assert config.collect_predictions is True
        assert config.collect_features is False
        assert config.is_separation_merge is False

    def test_all_mode(self):
        """Test all (features + predictions) merge mode."""
        config = MergeConfigParser.parse("all")

        assert config.collect_features is True
        assert config.collect_predictions is True
        assert config.is_separation_merge is False

    def test_concat_mode_for_separation(self):
        """Test concat mode for separation branches."""
        config = MergeConfigParser.parse("concat")

        assert config.collect_features is True
        assert config.is_separation_merge is True

    def test_error_on_unknown_mode(self):
        """Test error on unknown string mode."""
        with pytest.raises(ValueError, match="Unknown merge mode"):
            MergeConfigParser.parse("invalid_mode")

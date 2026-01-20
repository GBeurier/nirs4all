"""Integration tests for merge strategies (Phase 5).

Tests the concat merge mode configuration parsing and the
unified source merge syntax.

Note: Full integration tests for separation branches with by_tag
require Phase 4 branch execution to be complete.
"""

import pytest
import numpy as np
from unittest.mock import patch

import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    SavitzkyGolay,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler


class TestMergeAutoDetectsBranchType:
    """Tests for branch type auto-detection in merge."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        return nirs4all.generate.regression(n_samples=50, random_state=42)

    def test_features_merge_with_duplication_branch(self, simple_dataset):
        """Test features merge with standard duplication branches."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": "features"},
            {"model": PLSRegression(n_components=5)},
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=simple_dataset,
            verbose=0,
        )

        assert result is not None
        assert hasattr(result, 'best_rmse')

    def test_concat_with_duplication_branch(self, simple_dataset):
        """Test concat merge with duplication branches still works."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": "concat"},  # Using concat with duplication branches
            {"model": PLSRegression(n_components=5)},
        ]

        # Should still complete successfully (concat falls back to features mode)
        result = nirs4all.run(
            pipeline=pipeline,
            dataset=simple_dataset,
            verbose=0,
        )

        assert result is not None


class TestSourceMergeUnifiedSyntax:
    """Tests for source merge via merge keyword configuration."""

    def test_sources_concat_syntax(self):
        """Test {"merge": {"sources": "concat"}} syntax is parsed."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse({"sources": "concat"})

        assert config.source_merge is not None
        assert config.source_merge.strategy == "concat"

    def test_sources_stack_syntax(self):
        """Test {"merge": {"sources": "stack"}} syntax is parsed."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse({"sources": "stack"})

        assert config.source_merge is not None
        assert config.source_merge.strategy == "stack"

    def test_sources_dict_syntax(self):
        """Test {"merge": {"sources": "dict"}} syntax is parsed."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse({"sources": "dict"})

        assert config.source_merge is not None
        assert config.source_merge.strategy == "dict"

    def test_sources_with_full_config(self):
        """Test sources with full configuration dict."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse({
            "sources": {
                "strategy": "stack",
                "sources": ["source_0", "source_1"],
                "on_incompatible": "pad",
                "output_name": "merged_features",
            }
        })

        assert config.source_merge is not None
        assert config.source_merge.strategy == "stack"
        assert config.source_merge.sources == ["source_0", "source_1"]
        assert config.source_merge.on_incompatible == "pad"
        assert config.source_merge.output_name == "merged_features"


class TestMergeModesCombination:
    """Tests for combining different merge modes."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset."""
        return nirs4all.generate.regression(n_samples=50, random_state=42)

    def test_features_and_predictions_merge(self, dataset):
        """Test merge with both features and predictions."""
        pipeline = [
            {"branch": [[SNV(), PLSRegression(5)], [MSC(), PLSRegression(5)]]},
            {"merge": "all"},  # Both features and predictions
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
        )

        assert result is not None

    def test_predictions_only_merge(self, dataset):
        """Test predictions-only merge."""
        pipeline = [
            {"branch": [[SNV(), PLSRegression(5)], [MSC(), PLSRegression(5)]]},
            {"merge": "predictions"},
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
        )

        assert result is not None

    def test_features_only_merge(self, dataset):
        """Test features-only merge."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": "features"},
            {"model": PLSRegression(n_components=5)},
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
        )

        assert result is not None


class TestMergeDictConfig:
    """Tests for dict-style merge configuration."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset."""
        return nirs4all.generate.regression(n_samples=50, random_state=42)

    def test_dict_features_config(self, dataset):
        """Test dict config for features merge."""
        pipeline = [
            {"branch": [[SNV()], [MSC()], [SavitzkyGolay()]]},
            {"merge": {"features": [0, 1]}},  # Only first two branches
            {"model": PLSRegression(n_components=5)},
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
        )

        assert result is not None

    def test_dict_concat_config_parsed_correctly(self):
        """Test dict config for concat merge is parsed correctly."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse({"concat": True})

        assert config.collect_features is True
        assert config.is_separation_merge is True


class TestConcatModeConfigParsing:
    """Tests for concat merge mode configuration parsing."""

    def test_concat_string_mode_parsed(self):
        """Test 'concat' string mode is parsed correctly."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse("concat")

        assert config.collect_features is True
        assert config.is_separation_merge is True
        assert config.output_as == "features"

    def test_concat_dict_mode_parsed(self):
        """Test {'concat': True} dict mode is parsed correctly."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        config = MergeConfigParser.parse({"concat": True})

        assert config.collect_features is True
        assert config.is_separation_merge is True

    def test_concat_mode_serialization(self):
        """Test concat mode survives serialization."""
        from nirs4all.operators.data.merge import MergeConfig

        config = MergeConfig(
            collect_features=True,
            is_separation_merge=True,
        )

        # Round-trip serialization
        d = config.to_dict()
        restored = MergeConfig.from_dict(d)

        assert restored.is_separation_merge is True
        assert restored.collect_features is True


class TestMergeStrategyValidation:
    """Tests for merge strategy validation."""

    def test_unknown_mode_raises_error(self):
        """Test unknown merge mode raises error."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        with pytest.raises(ValueError, match="Unknown merge mode"):
            MergeConfigParser.parse("invalid_mode")

    def test_valid_modes_accepted(self):
        """Test all valid modes are accepted."""
        from nirs4all.controllers.data.merge import MergeConfigParser

        valid_modes = ["features", "predictions", "all", "concat"]

        for mode in valid_modes:
            config = MergeConfigParser.parse(mode)
            assert config is not None

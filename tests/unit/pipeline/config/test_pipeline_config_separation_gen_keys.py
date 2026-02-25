"""Unit tests for _has_gen_keys() with separation branches.

Verifies that generator keywords inside separation branches (by_source,
by_tag, by_metadata, by_filter) are correctly detected at the PipelineConfigs
level, while duplication branch content is still skipped.
"""

import pytest

from nirs4all.pipeline.config.pipeline_config import PipelineConfigs


class TestHasGenKeysSeparationBranches:
    """Test _has_gen_keys detection for separation branch generators."""

    def test_by_source_with_or_detected(self):
        """_or_ inside by_source steps should be detected."""
        config = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"_or_": ["SNV", "MSC"]}, "PLS"],
                    "X2": ["StandardScaler"],
                },
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_tag_with_range_detected(self):
        """_range_ inside by_tag steps should be detected."""
        config = [
            {"branch": {
                "by_tag": "outlier",
                "steps": [{"class": "PLS", "params": {"n_components": {"_range_": [1, 10]}}}],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_metadata_with_grid_detected(self):
        """_grid_ inside by_metadata steps should be detected."""
        config = [
            {"branch": {
                "by_metadata": "site",
                "steps": [{"_grid_": {"alpha": [0.1, 1.0], "n_components": [5, 10]}}],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_filter_with_sample_detected(self):
        """_sample_ inside by_filter steps should be detected."""
        config = [
            {"branch": {
                "by_filter": "quality > 0.5",
                "steps": [{"class": "Ridge", "params": {"alpha": {"_sample_": {"distribution": "log_uniform", "from": 0.001, "to": 10, "num": 5}}}}],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_source_with_cartesian_detected(self):
        """_cartesian_ inside by_source steps should be detected."""
        config = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "NIR": [{"_cartesian_": [{"_or_": ["SNV", "MSC"]}, {"_or_": ["SG", None]}]}],
                    "markers": ["StandardScaler"],
                },
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_source_with_log_range_detected(self):
        """_log_range_ inside by_source steps should be detected."""
        config = [
            {"branch": {
                "by_source": True,
                "steps": [{"class": "Ridge", "params": {"alpha": {"_log_range_": [0.001, 10, 5]}}}],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_source_with_zip_detected(self):
        """_zip_ inside by_source steps should be detected."""
        config = [
            {"branch": {
                "by_source": True,
                "steps": [{"_zip_": {"alpha": [0.1, 1.0], "n_components": [5, 10]}}],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_by_source_with_chain_detected(self):
        """_chain_ inside by_source steps should be detected."""
        config = [
            {"branch": {
                "by_source": True,
                "steps": [{"_chain_": ["SNV", "MSC", "Detrend"]}],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_separation_branch_no_generators_not_detected(self):
        """Separation branch without generators should return False."""
        config = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": ["SNV", "PLS"],
                    "X2": ["StandardScaler"],
                },
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is False

    def test_by_tag_no_generators_not_detected(self):
        """by_tag branch without generators should return False."""
        config = [
            {"branch": {
                "by_tag": "outlier",
                "steps": ["MinMaxScaler"],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is False


class TestHasGenKeysDuplicationBranchStillSkipped:
    """Verify duplication branches are still skipped (no regression)."""

    def test_duplication_list_branch_with_or_skipped(self):
        """_or_ inside duplication branch (list) should be skipped."""
        config = [
            {"branch": [
                [{"_or_": ["SNV", "MSC"]}],
                ["StandardScaler"],
            ]},
        ]
        assert PipelineConfigs._has_gen_keys(config) is False

    def test_duplication_named_branch_with_or_skipped(self):
        """_or_ inside duplication branch (named dict) should be skipped."""
        config = [
            {"branch": {
                "snv_path": [{"_or_": ["SNV", "MSC"]}],
                "scaler_path": ["StandardScaler"],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is False

    def test_duplication_branch_generator_at_top_skipped(self):
        """Generator as the branch value (duplication) should be skipped."""
        config = [
            {"branch": {"_or_": ["SNV", "MSC"]}},
        ]
        assert PipelineConfigs._has_gen_keys(config) is False

    def test_top_level_generator_still_detected(self):
        """Generator at pipeline level (not in branch) should still be detected."""
        config = [
            {"_or_": ["SNV", "MSC"]},
            "PLS",
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

    def test_mixed_pipeline_with_separation_and_top_level_gen(self):
        """Pipeline with both top-level generator and separation branch."""
        config = [
            {"_or_": ["MinMaxScaler", "StandardScaler"]},
            {"branch": {
                "by_source": True,
                "steps": ["PLS"],
            }},
        ]
        assert PipelineConfigs._has_gen_keys(config) is True

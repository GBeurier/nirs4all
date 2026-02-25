"""Unit tests for PipelineConfigs expansion with separation branches.

Verifies that PipelineConfigs correctly expands generator keywords inside
separation branches into multiple top-level pipeline configurations.
"""

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.pipeline.config.pipeline_config import PipelineConfigs


class TestSeparationBranchExpansion:
    """Test PipelineConfigs expansion for separation branch generators."""

    def test_by_source_or_expands_to_correct_count(self):
        """by_source + _or_ (3 choices) should produce 3 configs."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"_or_": [StandardScaler(), MinMaxScaler(), Ridge()]}, PLSRegression(5)],
                    "X2": [StandardScaler()],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.steps) == 3
        assert configs.has_configurations is True

    def test_by_source_range_expands(self):
        """by_source + _range_ should produce N configs."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": {"_range_": [2, 6]}}}],
                    "X2": [StandardScaler()],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        # _range_ [2, 6] produces [2, 3, 4, 5, 6] = 5 values
        assert len(configs.steps) == 5

    def test_by_source_per_source_generators_cartesian(self):
        """Generators in multiple sources produce Cartesian product."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"_or_": [StandardScaler(), MinMaxScaler()]}],
                    "X2": [{"_or_": [StandardScaler(), MinMaxScaler(), Ridge()]}],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        # 2 choices for X1 × 3 choices for X2 = 6 configs
        assert len(configs.steps) == 6

    def test_by_source_cartesian_expands(self):
        """by_source + _cartesian_ should expand correctly."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"_cartesian_": [
                        {"_or_": [StandardScaler(), MinMaxScaler()]},
                        {"_or_": [None, Ridge()]},
                    ]}],
                    "X2": [StandardScaler()],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        # _cartesian_ of 2 × 2 = 4 configs
        assert len(configs.steps) == 4

    def test_by_tag_or_expands(self):
        """by_tag + _or_ should produce correct configs."""
        pipeline = [
            {"branch": {
                "by_tag": "outlier",
                "steps": [{"_or_": [StandardScaler(), MinMaxScaler()]}],
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.steps) == 2

    def test_by_metadata_grid_expands(self):
        """by_metadata + _grid_ should produce Cartesian product."""
        pipeline = [
            {"branch": {
                "by_metadata": "site",
                "steps": [{"_grid_": {"alpha": [0.1, 1.0], "n_components": [5, 10]}}],
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        # _grid_ of 2 × 2 = 4 configs
        assert len(configs.steps) == 4

    def test_by_filter_log_range_expands(self):
        """by_filter + _log_range_ should produce correct configs."""
        pipeline = [
            {"branch": {
                "by_filter": "quality > 0.5",
                "steps": [{"class": "sklearn.linear_model.Ridge", "params": {"alpha": {"_log_range_": [0.001, 1.0, 4]}}}],
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        # _log_range_ [0.001, 1.0, 4] produces 4 values
        assert len(configs.steps) == 4

    def test_separation_no_generators_single_config(self):
        """Separation branch without generators should produce 1 config."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [StandardScaler(), PLSRegression(5)],
                    "X2": [StandardScaler()],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.steps) == 1
        assert configs.has_configurations is False

    def test_expanded_configs_preserve_structure(self):
        """Expanded configs should preserve separation branch structure."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"_or_": [StandardScaler(), MinMaxScaler()]}],
                    "X2": [StandardScaler()],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.steps) == 2

        for config in configs.steps:
            # Each config is a list with one step (the branch step)
            assert len(config) == 1
            step = config[0]
            assert "branch" in step
            branch = step["branch"]
            assert branch["by_source"] is True
            assert "steps" in branch
            assert "X1" in branch["steps"]
            assert "X2" in branch["steps"]
            # X1 should have a single operator (no generator), X2 unchanged
            x1_steps = branch["steps"]["X1"]
            assert isinstance(x1_steps, list)
            assert len(x1_steps) == 1
            # No generator keywords should remain
            x1_step = x1_steps[0]
            assert not isinstance(x1_step, dict) or "_or_" not in x1_step

    def test_by_source_shared_steps_with_generator(self):
        """by_source with shared steps (list) containing generators."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": [{"_or_": [StandardScaler(), MinMaxScaler()]}],
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.steps) == 2

    def test_generator_choices_tracked(self):
        """Generator choices should be tracked for separation branches."""
        pipeline = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "X1": [{"_or_": [StandardScaler(), MinMaxScaler()]}],
                    "X2": [StandardScaler()],
                },
            }},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.generator_choices) == 2
        # Each choice list should be non-empty
        for choices in configs.generator_choices:
            assert len(choices) > 0


class TestDuplicationBranchExpansionUnchanged:
    """Verify duplication branch expansion behavior is unchanged."""

    def test_duplication_branch_not_expanded_at_config_level(self):
        """Duplication branches with generators should NOT be expanded at config level."""
        pipeline = [
            {"branch": [
                [{"_or_": [StandardScaler(), MinMaxScaler()]}],
                [Ridge()],
            ]},
            {"model": PLSRegression(5)},
        ]
        configs = PipelineConfigs(pipeline, "test")
        # Should be 1 config — generators inside duplication branches
        # are handled at runtime by BranchController
        assert len(configs.steps) == 1

    def test_named_duplication_branch_not_expanded(self):
        """Named duplication branches with generators should NOT be expanded."""
        pipeline = [
            {"branch": {
                "snv_path": [{"_or_": [StandardScaler(), MinMaxScaler()]}],
                "scaler_path": [Ridge()],
            }},
            {"model": PLSRegression(5)},
        ]
        configs = PipelineConfigs(pipeline, "test")
        assert len(configs.steps) == 1

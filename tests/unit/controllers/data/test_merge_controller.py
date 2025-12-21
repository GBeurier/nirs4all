"""
Integration tests for MergeController Phases 2 and 3.

Tests:
- Basic branch exit after merge
- Branch validation (valid/invalid indices)
- Multiple keyword support (merge, merge_sources, merge_predictions)
- Context clearing after merge
- Feature collection from branch snapshots (Phase 3)

These tests verify the Phase 2 and 3 implementation from the branching_concat_merge_design.
"""

import pytest
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data._features.feature_source import FeatureSource
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.controllers.data.merge import MergeController, MergeConfigParser
from nirs4all.operators.data.merge import MergeConfig


def create_test_dataset(n_samples: int = 50, n_features: int = 30, seed: int = 42) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_merge")
    dataset.add_samples(X, indexes={"partition": "train"})
    dataset.add_targets(y)

    return dataset


def create_mock_feature_snapshot(n_samples: int = 50, n_features: int = 30, seed: int = 42) -> list:
    """Create a mock feature snapshot (list of FeatureSource objects)."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)

    fs = FeatureSource()
    fs.add_samples(X)

    return [fs]


class TestMergeControllerMatching:
    """Test MergeController.matches() for different keywords."""

    def test_matches_merge_keyword(self):
        """Test that controller matches 'merge' keyword."""
        assert MergeController.matches({}, None, "merge") is True

    def test_matches_merge_sources_keyword(self):
        """Test that controller matches 'merge_sources' keyword."""
        assert MergeController.matches({}, None, "merge_sources") is True

    def test_matches_merge_predictions_keyword(self):
        """Test that controller matches 'merge_predictions' keyword."""
        assert MergeController.matches({}, None, "merge_predictions") is True

    def test_does_not_match_other_keywords(self):
        """Test that controller doesn't match unrelated keywords."""
        assert MergeController.matches({}, None, "branch") is False
        assert MergeController.matches({}, None, "model") is False
        assert MergeController.matches({}, None, "preprocessing") is False
        assert MergeController.matches({}, None, "concat_transform") is False


class TestMergeConfigParsing:
    """Test MergeConfigParser for Phase 2."""

    def test_parse_features_string(self):
        """Test parsing 'features' string mode."""
        config = MergeConfigParser.parse("features")
        assert config.collect_features is True
        assert config.collect_predictions is False

    def test_parse_predictions_string(self):
        """Test parsing 'predictions' string mode."""
        config = MergeConfigParser.parse("predictions")
        assert config.collect_predictions is True
        assert config.collect_features is False

    def test_parse_all_string(self):
        """Test parsing 'all' string mode."""
        config = MergeConfigParser.parse("all")
        assert config.collect_features is True
        assert config.collect_predictions is True

    def test_parse_dict_with_features_list(self):
        """Test parsing dict with specific feature branches."""
        config = MergeConfigParser.parse({"features": [0, 2]})
        assert config.collect_features is True
        assert config.feature_branches == [0, 2]

    def test_parse_dict_with_predictions_list(self):
        """Test parsing dict with specific prediction branches."""
        config = MergeConfigParser.parse({"predictions": [1]})
        assert config.collect_predictions is True
        assert config.prediction_branches == [1]

    def test_parse_mixed_mode(self):
        """Test parsing mixed features + predictions."""
        config = MergeConfigParser.parse({
            "features": [1],
            "predictions": [0]
        })
        assert config.collect_features is True
        assert config.collect_predictions is True
        assert config.feature_branches == [1]
        assert config.prediction_branches == [0]


class TestMergeControllerBranchValidation:
    """Test branch validation in MergeController."""

    def create_mock_branch_contexts(self, n_branches: int):
        """Create mock branch contexts for testing."""
        return [
            {
                "branch_id": i,
                "name": f"branch_{i}",
                "features_snapshot": None
            }
            for i in range(n_branches)
        ]

    def test_validate_valid_branch_indices(self):
        """Test that valid branch indices pass validation."""
        controller = MergeController()
        config = MergeConfig(
            collect_features=True,
            feature_branches=[0, 1]
        )
        branch_contexts = self.create_mock_branch_contexts(3)

        # Should not raise
        controller._validate_branches(config, branch_contexts)

    def test_validate_invalid_branch_index_raises(self):
        """Test that invalid branch index raises ValueError."""
        controller = MergeController()
        config = MergeConfig(
            collect_features=True,
            feature_branches=[0, 5]  # 5 is invalid for 3 branches
        )
        branch_contexts = self.create_mock_branch_contexts(3)

        with pytest.raises(ValueError, match="Invalid branch index"):
            controller._validate_branches(config, branch_contexts)

    def test_validate_all_branches(self):
        """Test that 'all' is valid for any number of branches."""
        controller = MergeController()
        config = MergeConfig(
            collect_features=True,
            feature_branches="all"
        )
        branch_contexts = self.create_mock_branch_contexts(5)

        # Should not raise
        controller._validate_branches(config, branch_contexts)

    def test_validate_branch_by_name(self):
        """Test branch validation by name."""
        controller = MergeController()
        branch_contexts = [
            {"branch_id": 0, "name": "snv_path"},
            {"branch_id": 1, "name": "msc_path"},
        ]

        config = MergeConfig(
            collect_features=True,
            feature_branches="all"  # String 'all' should work
        )

        # Should not raise
        controller._validate_branches(config, branch_contexts)

    def test_error_message_includes_available_indices(self):
        """Test that error message shows available branch indices."""
        controller = MergeController()
        config = MergeConfig(
            collect_features=True,
            feature_branches=[10]
        )
        branch_contexts = self.create_mock_branch_contexts(3)

        with pytest.raises(ValueError) as exc_info:
            controller._validate_branches(config, branch_contexts)

        assert "Available indices" in str(exc_info.value)
        assert "[0, 1, 2]" in str(exc_info.value)


class TestMergeSourcesAndPredictionsKeywords:
    """Test that merge_sources and merge_predictions keywords are handled."""

    def test_merge_sources_single_source_is_noop(self):
        """Test that merge_sources on single-source dataset is a no-op with warning."""
        # Phase 9: merge_sources with single source logs warning and returns no-op
        from nirs4all.pipeline.steps.parser import ParsedStep

        controller = MergeController()

        # Create minimal mock objects
        class MockStepInfo:
            keyword = "merge_sources"
            original_step = {"merge_sources": "concat"}

        class MockContext:
            custom = {"branch_contexts": [], "in_branch_mode": False}
            selector = {}
            def copy(self):
                return MockContext()

        step_info = MockStepInfo()
        dataset = create_test_dataset()  # Single source dataset
        context = MockContext()

        # Single source dataset - should return no-op, not error
        result_context, output = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=None,
        )

        # Check metadata indicates no-op
        assert output.metadata.get("source_merge") == "no-op"
        assert output.metadata.get("reason") == "single_source_dataset"

    def test_merge_predictions_requires_prediction_store(self):
        """Test that merge_predictions requires prediction_store."""
        controller = MergeController()

        class MockStepInfo:
            keyword = "merge_predictions"
            original_step = {"merge_predictions": "all"}

        class MockContext:
            custom = {"branch_contexts": [], "in_branch_mode": False}
            selector = {}
            state = type('obj', (object,), {'step_number': 10})()
            def copy(self):
                return MockContext()
            def with_partition(self, partition):
                return self

        step_info = MockStepInfo()
        dataset = create_test_dataset()
        context = MockContext()

        # Without prediction_store, should raise ValueError
        with pytest.raises(ValueError, match="merge_predictions requires prediction_store"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=None,
                prediction_store=None,  # Explicitly None
            )


class TestMergeControllerExitsBranchMode:
    """Test that merge properly exits branch mode."""

    def test_merge_clears_branch_contexts(self):
        """Test that merge clears branch contexts from context."""
        controller = MergeController()

        class MockStepInfo:
            keyword = "merge"
            original_step = {"merge": "features"}

        dataset = create_test_dataset()

        # Create mock branch contexts with feature snapshots
        branch_contexts = [
            {
                "branch_id": 0,
                "name": "branch_0",
                "features_snapshot": create_mock_feature_snapshot(50, 30, seed=42)
            },
            {
                "branch_id": 1,
                "name": "branch_1",
                "features_snapshot": create_mock_feature_snapshot(50, 30, seed=43)
            },
        ]

        class MockContext:
            def __init__(self):
                self.custom = {
                    "branch_contexts": branch_contexts,
                    "in_branch_mode": True
                }

            def copy(self):
                new = MockContext()
                new.custom = dict(self.custom)
                return new

        step_info = MockStepInfo()
        context = MockContext()

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=None,
        )

        # Verify branch mode was exited
        assert result_context.custom["branch_contexts"] == []
        assert result_context.custom["in_branch_mode"] is False

    def test_merge_without_branch_mode_raises(self):
        """Test that merge without active branch mode raises error."""
        controller = MergeController()

        class MockStepInfo:
            keyword = "merge"
            original_step = {"merge": "features"}

        class MockContext:
            custom = {"branch_contexts": [], "in_branch_mode": False}
            def copy(self):
                new = MockContext()
                new.custom = dict(self.custom)
                return new

        step_info = MockStepInfo()
        dataset = create_test_dataset()
        context = MockContext()

        with pytest.raises(ValueError, match="requires active branch contexts"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=None,
            )


class TestMergeOutputMetadata:
    """Test metadata returned by merge step."""

    def test_merge_output_includes_mode(self):
        """Test that output metadata includes merge mode."""
        controller = MergeController()

        class MockStepInfo:
            keyword = "merge"
            original_step = {"merge": "features"}

        dataset = create_test_dataset()
        branch_contexts = [
            {
                "branch_id": 0,
                "name": "branch_0",
                "features_snapshot": create_mock_feature_snapshot(50, 30, seed=42)
            },
        ]

        class MockContext:
            def __init__(self):
                self.custom = {
                    "branch_contexts": branch_contexts,
                    "in_branch_mode": True
                }

            def copy(self):
                new = MockContext()
                new.custom = dict(self.custom)
                return new

        step_info = MockStepInfo()
        context = MockContext()

        _, output = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=None,
        )

        assert output.metadata["merge_mode"] == "features"

    def test_merge_output_includes_branches(self):
        """Test that output metadata includes branch information."""
        controller = MergeController()

        class MockStepInfo:
            keyword = "merge"
            original_step = {"merge": {"features": [0, 1]}}

        dataset = create_test_dataset()
        branch_contexts = [
            {
                "branch_id": 0,
                "name": "branch_0",
                "features_snapshot": create_mock_feature_snapshot(50, 30, seed=42)
            },
            {
                "branch_id": 1,
                "name": "branch_1",
                "features_snapshot": create_mock_feature_snapshot(50, 30, seed=43)
            },
        ]

        class MockContext:
            def __init__(self):
                self.custom = {
                    "branch_contexts": branch_contexts,
                    "in_branch_mode": True
                }

            def copy(self):
                new = MockContext()
                new.custom = dict(self.custom)
                return new

        step_info = MockStepInfo()
        context = MockContext()

        _, output = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=None,
        )

        assert output.metadata["feature_branches"] == [0, 1]


# ============================================================================
# Phase 6 Unit Tests: AsymmetricBranchAnalyzer
# ============================================================================

from nirs4all.controllers.data.merge import (
    BranchAnalysisResult,
    AsymmetryReport,
    AsymmetricBranchAnalyzer,
)


class TestBranchAnalysisResult:
    """Test BranchAnalysisResult dataclass."""

    def test_branch_analysis_result_creation(self):
        """Test creating a BranchAnalysisResult."""
        result = BranchAnalysisResult(
            branch_id=0,
            branch_name="test_branch",
            has_models=True,
            model_names=["PLSRegression"],
            model_count=1,
            feature_dim=30,
            has_features=True,
        )

        assert result.branch_id == 0
        assert result.branch_name == "test_branch"
        assert result.has_models is True
        assert result.model_names == ["PLSRegression"]
        assert result.model_count == 1
        assert result.feature_dim == 30
        assert result.has_features is True

    def test_branch_analysis_result_without_models(self):
        """Test BranchAnalysisResult for branch without models."""
        result = BranchAnalysisResult(
            branch_id=1,
            branch_name="feature_only_branch",
            has_models=False,
            model_names=[],
            model_count=0,
            feature_dim=50,
            has_features=True,
        )

        assert result.has_models is False
        assert result.model_count == 0
        assert result.model_names == []

    def test_branch_analysis_result_with_multiple_models(self):
        """Test BranchAnalysisResult with multiple models."""
        result = BranchAnalysisResult(
            branch_id=0,
            branch_name="multi_model",
            has_models=True,
            model_names=["PLSRegression", "Ridge", "Lasso"],
            model_count=3,
            feature_dim=30,
            has_features=True,
        )

        assert result.model_count == 3
        assert len(result.model_names) == 3
        assert "PLSRegression" in result.model_names


class TestAsymmetryReport:
    """Test AsymmetryReport dataclass."""

    def test_asymmetry_report_symmetric(self):
        """Test AsymmetryReport for symmetric branches."""
        report = AsymmetryReport(
            is_asymmetric=False,
            has_model_asymmetry=False,
            has_model_count_asymmetry=False,
            has_feature_dim_asymmetry=False,
            branches_with_models=[0, 1],
            branches_without_models=[],
            model_counts={0: 1, 1: 1},
            feature_dims={0: 30, 1: 30},
            summary="Branches are symmetric",
        )

        assert report.is_asymmetric is False
        assert report.has_model_asymmetry is False
        assert len(report.branches_without_models) == 0

    def test_asymmetry_report_model_asymmetry(self):
        """Test AsymmetryReport with model presence asymmetry."""
        report = AsymmetryReport(
            is_asymmetric=True,
            has_model_asymmetry=True,
            has_model_count_asymmetry=True,
            has_feature_dim_asymmetry=False,
            branches_with_models=[0],
            branches_without_models=[1],
            model_counts={0: 1, 1: 0},
            feature_dims={0: 30, 1: 30},
            summary="Model presence asymmetry: branches [0] have models, branches [1] have only features",
        )

        assert report.is_asymmetric is True
        assert report.has_model_asymmetry is True
        assert report.branches_with_models == [0]
        assert report.branches_without_models == [1]

    def test_asymmetry_report_feature_dim_asymmetry(self):
        """Test AsymmetryReport with feature dimension asymmetry."""
        report = AsymmetryReport(
            is_asymmetric=True,
            has_model_asymmetry=False,
            has_model_count_asymmetry=False,
            has_feature_dim_asymmetry=True,
            branches_with_models=[0, 1],
            branches_without_models=[],
            model_counts={0: 1, 1: 1},
            feature_dims={0: 30, 1: 50},
            summary="Feature dimension asymmetry: branch 0: 30 features, branch 1: 50 features",
        )

        assert report.is_asymmetric is True
        assert report.has_feature_dim_asymmetry is True
        assert report.feature_dims[0] != report.feature_dims[1]


class TestAsymmetricBranchAnalyzer:
    """Test AsymmetricBranchAnalyzer utility class."""

    def create_branch_contexts(self, branches_config: list) -> list:
        """Create branch contexts from configuration.

        Args:
            branches_config: List of dicts with 'has_features', 'feature_dim' keys.

        Returns:
            List of branch context dictionaries.
        """
        contexts = []
        for i, config in enumerate(branches_config):
            ctx = {
                "branch_id": i,
                "name": f"branch_{i}",
            }

            if config.get("has_features", False):
                # Create mock feature snapshot with specified dimension
                feature_dim = config.get("feature_dim", 30)
                n_samples = config.get("n_samples", 50)
                np.random.seed(i)
                X = np.random.randn(n_samples, feature_dim)
                fs = FeatureSource()
                fs.add_samples(X)
                ctx["features_snapshot"] = [fs]
            else:
                ctx["features_snapshot"] = None

            contexts.append(ctx)

        return contexts

    def create_mock_context(self, step_number: int = 10):
        """Create mock ExecutionContext."""
        class MockState:
            def __init__(self, step_num):
                self.step_number = step_num

        class MockContext:
            def __init__(self, step_num):
                self.state = MockState(step_num)
                self.custom = {}

        return MockContext(step_number)

    def create_mock_prediction_store(self, branch_models: dict):
        """Create mock prediction store.

        Args:
            branch_models: Dict mapping branch_id to list of model names.
        """
        class MockPredictionStore:
            def __init__(self, branch_models):
                self._branch_models = branch_models

            def filter_predictions(self, branch_id=None, partition=None, load_arrays=False):
                models = self._branch_models.get(branch_id, [])
                return [
                    {"model_name": name, "step_idx": 5, "branch_id": branch_id}
                    for name in models
                ]

        return MockPredictionStore(branch_models)

    def test_analyze_branch_with_features_no_models(self):
        """Test analyzing a branch with features but no models."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30}
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({})

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        result = analyzer.analyze_branch(0)

        assert result.branch_id == 0
        assert result.has_features is True
        assert result.has_models is False
        assert result.model_count == 0
        assert result.model_names == []

    def test_analyze_branch_with_models(self):
        """Test analyzing a branch that has models."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30}
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression", "Ridge"]
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        result = analyzer.analyze_branch(0)

        assert result.has_models is True
        assert result.model_count == 2
        assert "PLSRegression" in result.model_names
        assert "Ridge" in result.model_names

    def test_analyze_branch_missing_returns_empty(self):
        """Test analyzing a missing branch returns empty result."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30}
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({})

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        result = analyzer.analyze_branch(99)  # Non-existent branch

        assert result.branch_id == 99
        assert result.has_models is False
        assert result.has_features is False
        assert result.branch_name is None

    def test_analyze_branch_caches_result(self):
        """Test that analyze_branch caches results."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30}
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({0: ["PLS"]})

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)

        result1 = analyzer.analyze_branch(0)
        result2 = analyzer.analyze_branch(0)

        assert result1 is result2
        assert 0 in analyzer._analysis_cache

    def test_analyze_all_symmetric_branches(self):
        """Test analyze_all with symmetric branches (same models, same features)."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: ["PLSRegression"],
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        report = analyzer.analyze_all()

        assert report.is_asymmetric is False
        assert report.has_model_asymmetry is False
        assert report.has_model_count_asymmetry is False
        assert report.branches_with_models == [0, 1]
        assert report.branches_without_models == []

    def test_analyze_all_model_asymmetry(self):
        """Test analyze_all detects model presence asymmetry."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: [],  # No models in branch 1
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        report = analyzer.analyze_all()

        assert report.is_asymmetric is True
        assert report.has_model_asymmetry is True
        assert report.branches_with_models == [0]
        assert report.branches_without_models == [1]

    def test_analyze_all_model_count_asymmetry(self):
        """Test analyze_all detects model count asymmetry."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression", "Ridge"],  # 2 models
            1: ["Lasso"],  # 1 model
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        report = analyzer.analyze_all()

        assert report.is_asymmetric is True
        assert report.has_model_count_asymmetry is True
        assert report.model_counts[0] == 2
        assert report.model_counts[1] == 1

    def test_analyze_all_feature_dim_asymmetry(self):
        """Test analyze_all detects feature dimension asymmetry."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 50},  # Different dimension
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: ["PLSRegression"],
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        report = analyzer.analyze_all()

        assert report.is_asymmetric is True
        assert report.has_feature_dim_asymmetry is True

    def test_analyze_all_empty_branches(self):
        """Test analyze_all with no branches."""
        analyzer = AsymmetricBranchAnalyzer([], None, self.create_mock_context())
        report = analyzer.analyze_all()

        assert report.is_asymmetric is False
        assert report.summary == "No branches to analyze."

    def test_analyze_all_summary_content(self):
        """Test that summary contains relevant asymmetry information."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: [],
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        report = analyzer.analyze_all()

        assert "Model presence asymmetry" in report.summary
        assert "[0]" in report.summary or "0" in report.summary
        assert "[1]" in report.summary or "1" in report.summary

    def test_suggest_mixed_merge_asymmetric(self):
        """Test suggest_mixed_merge generates valid suggestion."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: [],  # No models - feature only
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        suggestion = analyzer.suggest_mixed_merge()

        assert suggestion is not None
        assert "predictions" in suggestion
        assert "features" in suggestion
        assert "[0]" in suggestion
        assert "[1]" in suggestion
        assert "mixed merge" in suggestion.lower()

    def test_suggest_mixed_merge_symmetric_returns_none(self):
        """Test suggest_mixed_merge returns None for symmetric branches."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: ["PLSRegression"],
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        suggestion = analyzer.suggest_mixed_merge()

        assert suggestion is None

    def test_suggest_mixed_merge_three_branches(self):
        """Test suggest_mixed_merge with three branches."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
            1: [],  # No models
            2: [],  # No models
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        suggestion = analyzer.suggest_mixed_merge()

        assert suggestion is not None
        # Branch 0 should be in predictions, branches 1 and 2 in features
        assert '"predictions": [0]' in suggestion
        assert '"features": [1, 2]' in suggestion

    def test_analyzer_without_prediction_store(self):
        """Test analyzer works without prediction store (all branches have no models)."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},
            {"has_features": True, "feature_dim": 30},
        ])
        context = self.create_mock_context()

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, None, context)
        result = analyzer.analyze_branch(0)

        assert result.has_models is False
        assert result.model_names == []

    def test_analyze_multiple_branches_various_configs(self):
        """Test analyzing branches with various configurations."""
        branch_contexts = self.create_branch_contexts([
            {"has_features": True, "feature_dim": 30},   # Branch 0: features + model
            {"has_features": True, "feature_dim": 50},   # Branch 1: features only
            {"has_features": False},                      # Branch 2: no features, no model
        ])
        context = self.create_mock_context()
        prediction_store = self.create_mock_prediction_store({
            0: ["PLSRegression"],
        })

        analyzer = AsymmetricBranchAnalyzer(branch_contexts, prediction_store, context)
        report = analyzer.analyze_all()

        assert report.is_asymmetric is True
        assert report.has_model_asymmetry is True
        assert 0 in report.branches_with_models
        assert 1 in report.branches_without_models
        assert 2 in report.branches_without_models


# =============================================================================
# Phase 9: Source Merge Tests
# =============================================================================

class TestSourceMergeConfig:
    """Tests for SourceMergeConfig dataclass (Phase 9)."""

    def test_default_config(self):
        """Test default configuration values."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig()
        assert config.strategy == "concat"
        assert config.sources == "all"
        assert config.on_incompatible == "error"
        assert config.output_name == "merged"

    def test_strategy_validation(self):
        """Test that invalid strategies raise ValueError."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        with pytest.raises(ValueError, match="strategy must be one of"):
            SourceMergeConfig(strategy="invalid")

    def test_on_incompatible_validation(self):
        """Test that invalid on_incompatible raises ValueError."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        with pytest.raises(ValueError, match="on_incompatible must be one of"):
            SourceMergeConfig(on_incompatible="invalid")

    def test_empty_sources_list_validation(self):
        """Test that empty sources list raises ValueError."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        with pytest.raises(ValueError, match="sources list cannot be empty"):
            SourceMergeConfig(sources=[])

    def test_get_source_indices_all(self):
        """Test resolving 'all' sources."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig(sources="all")
        indices = config.get_source_indices(["NIR", "markers", "Raman"])
        assert indices == [0, 1, 2]

    def test_get_source_indices_by_name(self):
        """Test resolving sources by name."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig(sources=["NIR", "Raman"])
        indices = config.get_source_indices(["NIR", "markers", "Raman"])
        assert indices == [0, 2]

    def test_get_source_indices_by_index(self):
        """Test resolving sources by index."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig(sources=[0, 2])
        indices = config.get_source_indices(["NIR", "markers", "Raman"])
        assert indices == [0, 2]

    def test_get_source_indices_invalid_name_raises(self):
        """Test that invalid source name raises ValueError."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig(sources=["NIR", "Unknown"])
        with pytest.raises(ValueError, match="Source name 'Unknown' not found"):
            config.get_source_indices(["NIR", "markers"])

    def test_get_source_indices_invalid_index_raises(self):
        """Test that out-of-range source index raises ValueError."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig(sources=[0, 5])
        with pytest.raises(ValueError, match="Source index 5 out of range"):
            config.get_source_indices(["NIR", "markers"])

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        config = SourceMergeConfig(
            strategy="stack",
            sources=["NIR", "MIR"],
            on_incompatible="flatten",
            output_name="combined",
        )

        data = config.to_dict()
        restored = SourceMergeConfig.from_dict(data)

        assert restored.strategy == config.strategy
        assert restored.sources == config.sources
        assert restored.on_incompatible == config.on_incompatible
        assert restored.output_name == config.output_name


class TestSourceMergeStrategies:
    """Tests for source merge strategy enums (Phase 9)."""

    def test_source_merge_strategy_values(self):
        """Test SourceMergeStrategy enum values."""
        from nirs4all.operators.data.merge import SourceMergeStrategy

        assert SourceMergeStrategy.CONCAT.value == "concat"
        assert SourceMergeStrategy.STACK.value == "stack"
        assert SourceMergeStrategy.DICT.value == "dict"

    def test_source_incompatible_strategy_values(self):
        """Test SourceIncompatibleStrategy enum values."""
        from nirs4all.operators.data.merge import SourceIncompatibleStrategy

        assert SourceIncompatibleStrategy.ERROR.value == "error"
        assert SourceIncompatibleStrategy.FLATTEN.value == "flatten"
        assert SourceIncompatibleStrategy.PAD.value == "pad"
        assert SourceIncompatibleStrategy.TRUNCATE.value == "truncate"

    def test_config_get_strategy(self):
        """Test getting strategy as enum."""
        from nirs4all.operators.data.merge import SourceMergeConfig, SourceMergeStrategy

        config = SourceMergeConfig(strategy="stack")
        assert config.get_strategy() == SourceMergeStrategy.STACK

    def test_config_get_incompatible_strategy(self):
        """Test getting incompatible strategy as enum."""
        from nirs4all.operators.data.merge import SourceMergeConfig, SourceIncompatibleStrategy

        config = SourceMergeConfig(on_incompatible="flatten")
        assert config.get_incompatible_strategy() == SourceIncompatibleStrategy.FLATTEN


class TestMergeSourcesConfigParsing:
    """Tests for merge_sources configuration parsing."""

    def test_parse_simple_string(self):
        """Test parsing simple strategy string."""
        controller = MergeController()
        config = controller._parse_source_merge_config("concat")
        assert config.strategy == "concat"
        assert config.sources == "all"

    def test_parse_dict_config(self):
        """Test parsing dict configuration."""
        controller = MergeController()
        config = controller._parse_source_merge_config({
            "strategy": "stack",
            "sources": [0, 1],
            "on_incompatible": "flatten",
        })
        assert config.strategy == "stack"
        assert config.sources == [0, 1]
        assert config.on_incompatible == "flatten"

    def test_parse_already_parsed_config(self):
        """Test that already parsed config is returned as-is."""
        from nirs4all.operators.data.merge import SourceMergeConfig

        controller = MergeController()
        original = SourceMergeConfig(strategy="dict")
        config = controller._parse_source_merge_config(original)
        assert config is original

    def test_parse_invalid_type_raises(self):
        """Test that invalid config type raises ValueError."""
        controller = MergeController()
        with pytest.raises(ValueError, match="Invalid merge_sources config type"):
            controller._parse_source_merge_config(123)

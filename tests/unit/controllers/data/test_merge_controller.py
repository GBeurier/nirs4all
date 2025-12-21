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

    def test_merge_sources_not_implemented(self):
        """Test that merge_sources raises NotImplementedError (Phase 9)."""
        # This is a Phase 9 feature - should raise NotImplementedError
        from nirs4all.pipeline.steps.parser import ParsedStep

        controller = MergeController()

        # Create minimal mock objects
        class MockStepInfo:
            keyword = "merge_sources"
            original_step = {"merge_sources": "concat"}

        class MockContext:
            custom = {"branch_contexts": [], "in_branch_mode": False}
            def copy(self):
                return MockContext()

        step_info = MockStepInfo()
        dataset = create_test_dataset()
        context = MockContext()

        with pytest.raises(NotImplementedError, match="Phase 9"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=None,
            )

    def test_merge_predictions_not_implemented(self):
        """Test that merge_predictions raises NotImplementedError (Phase 9)."""
        controller = MergeController()

        class MockStepInfo:
            keyword = "merge_predictions"
            original_step = {"merge_predictions": "average"}

        class MockContext:
            custom = {"branch_contexts": [], "in_branch_mode": False}
            def copy(self):
                return MockContext()

        step_info = MockStepInfo()
        dataset = create_test_dataset()
        context = MockContext()

        with pytest.raises(NotImplementedError, match="Phase 9"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=None,
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

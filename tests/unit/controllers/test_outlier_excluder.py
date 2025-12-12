"""
Unit tests for OutlierExcluderController.

Tests the outlier excluder branching functionality including:
- Controller matching for outlier_excluder syntax
- Strategy parsing and filter creation
- Branch name generation from strategies
- Outlier exclusion mask application
- Exclusion metadata tracking
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from nirs4all.controllers.data.outlier_excluder import (
    OutlierExcluderController,
    _create_outlier_filter,
    _strategy_to_name,
)
from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext,
    RuntimeContext
)
from nirs4all.pipeline.execution.result import StepOutput, StepResult
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestOutlierExcluderControllerMatches:
    """Test OutlierExcluderController.matches() method."""

    def test_matches_outlier_excluder_by_keyword(self):
        """Should match when 'by' is 'outlier_excluder'."""
        step = {"branch": {"by": "outlier_excluder", "strategies": []}}
        assert OutlierExcluderController.matches(step, None, "branch") is True

    def test_matches_with_strategies(self):
        """Should match with various strategies defined."""
        step = {
            "branch": {
                "by": "outlier_excluder",
                "strategies": [
                    None,
                    {"method": "isolation_forest"},
                    {"method": "mahalanobis", "threshold": 3.0}
                ]
            }
        }
        assert OutlierExcluderController.matches(step, None, "branch") is True

    def test_not_matches_regular_branch(self):
        """Should not match regular branch syntax."""
        step = {"branch": [["step1"], ["step2"]]}
        assert OutlierExcluderController.matches(step, None, "branch") is False

    def test_not_matches_named_branch(self):
        """Should not match named branch syntax."""
        step = {"branch": {"snv": ["snv"], "msc": ["msc"]}}
        assert OutlierExcluderController.matches(step, None, "branch") is False

    def test_not_matches_other_by_values(self):
        """Should not match other 'by' values."""
        step = {"branch": {"by": "some_other_splitter", "strategies": []}}
        assert OutlierExcluderController.matches(step, None, "branch") is False

    def test_not_matches_non_branch_keyword(self):
        """Should not match other keywords."""
        step = {"preprocessing": {"by": "outlier_excluder"}}
        assert OutlierExcluderController.matches(step, None, "preprocessing") is False

    def test_matches_with_empty_strategies(self):
        """Should match with empty strategies list."""
        step = {"branch": {"by": "outlier_excluder", "strategies": []}}
        assert OutlierExcluderController.matches(step, None, "branch") is True


class TestOutlierExcluderControllerPriority:
    """Test OutlierExcluderController priority."""

    def test_priority_higher_than_branch_controller(self):
        """OutlierExcluderController should have priority higher than BranchController."""
        # BranchController has priority 5, OutlierExcluderController should be 4
        # priority is a class attribute, not a method
        assert OutlierExcluderController.priority == 4


class TestStrategyParsing:
    """Test strategy parsing and filter creation."""

    def test_strategy_to_name_none(self):
        """None strategy should return 'baseline'."""
        name = _strategy_to_name(None, 0)
        assert name == "baseline"

    def test_strategy_to_name_isolation_forest(self):
        """Isolation forest strategy naming."""
        strategy = {"method": "isolation_forest", "contamination": 0.1}
        name = _strategy_to_name(strategy, 0)
        assert "if" in name
        assert "0.1" in name

    def test_strategy_to_name_mahalanobis(self):
        """Mahalanobis strategy naming."""
        strategy = {"method": "mahalanobis", "threshold": 3.0}
        name = _strategy_to_name(strategy, 0)
        assert "mahal" in name
        assert "3.0" in name

    def test_strategy_to_name_lof(self):
        """LOF strategy naming."""
        strategy = {"method": "lof", "contamination": 0.05}
        name = _strategy_to_name(strategy, 0)
        assert "lof" in name
        assert "0.05" in name

    def test_strategy_to_name_leverage(self):
        """Leverage strategy naming."""
        strategy = {"method": "leverage", "threshold": 2.0}
        name = _strategy_to_name(strategy, 0)
        assert "lev" in name

    def test_strategy_to_name_pca_residual(self):
        """PCA residual strategy naming."""
        strategy = {"method": "pca_residual", "n_components": 10, "threshold": 3.0}
        name = _strategy_to_name(strategy, 0)
        assert "pca_q" in name

    def test_strategy_to_name_method_only(self):
        """Strategy with just method name uses index."""
        strategy = {"method": "isolation_forest"}
        name = _strategy_to_name(strategy, 5)
        assert "if" in name
        assert "5" in name

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_create_outlier_filter_isolation_forest(self, mock_filter_class):
        """Create isolation forest filter."""
        strategy = {"method": "isolation_forest", "contamination": 0.1}
        mock_filter_instance = Mock()
        mock_filter_class.return_value = mock_filter_instance

        result = _create_outlier_filter(strategy)

        mock_filter_class.assert_called_once()
        call_kwargs = mock_filter_class.call_args[1]
        assert call_kwargs["method"] == "isolation_forest"
        assert call_kwargs["contamination"] == 0.1

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_create_outlier_filter_mahalanobis(self, mock_filter_class):
        """Create mahalanobis filter."""
        strategy = {"method": "mahalanobis", "threshold": 3.5}
        mock_filter_instance = Mock()
        mock_filter_class.return_value = mock_filter_instance

        result = _create_outlier_filter(strategy)

        mock_filter_class.assert_called_once()
        call_kwargs = mock_filter_class.call_args[1]
        assert call_kwargs["method"] == "mahalanobis"
        assert call_kwargs["threshold"] == 3.5

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_create_outlier_filter_lof(self, mock_filter_class):
        """Create LOF filter."""
        strategy = {"method": "lof", "contamination": 0.05}
        mock_filter_instance = Mock()
        mock_filter_class.return_value = mock_filter_instance

        result = _create_outlier_filter(strategy)

        mock_filter_class.assert_called_once()
        call_kwargs = mock_filter_class.call_args[1]
        assert call_kwargs["method"] == "lof"
        assert call_kwargs["contamination"] == 0.05

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_create_outlier_filter_leverage_mapped(self, mock_filter_class):
        """Leverage is mapped to pca_leverage."""
        strategy = {"method": "leverage", "threshold": 2.0}
        mock_filter_instance = Mock()
        mock_filter_class.return_value = mock_filter_instance

        result = _create_outlier_filter(strategy)

        call_kwargs = mock_filter_class.call_args[1]
        assert call_kwargs["method"] == "pca_leverage"

    def test_create_outlier_filter_unknown_raises(self):
        """Unknown method should raise ValueError."""
        strategy = {"method": "unknown_method"}

        with pytest.raises(ValueError, match="Unknown outlier method"):
            _create_outlier_filter(strategy)


class TestOutlierExcluderExecution:
    """Test OutlierExcluderController.execute() method."""

    @pytest.fixture
    def controller(self):
        return OutlierExcluderController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with spectral data."""
        dataset = Mock()
        dataset.name = "test_dataset"

        # Mock indexer with sample indices
        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataset._indexer = indexer

        # Mock X data
        X = np.random.randn(10, 100)  # 10 samples, 100 features
        dataset.x.return_value = X
        dataset.y.return_value = np.random.randn(10)

        # Mock features for snapshot/restore
        dataset._features = Mock()
        dataset._features.sources = []

        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0
        runtime.step_number = 1
        runtime.operation_count = 0
        runtime.saver = None

        def next_op():
            runtime.operation_count += 1
            return runtime.operation_count
        runtime.next_op = next_op

        return runtime

    def test_execute_baseline_strategy(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with baseline (None) strategy should not exclude samples."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [None]
                }
            },
            metadata={}
        )

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should have branch_contexts
        assert "branch_contexts" in result_context.custom
        assert len(result_context.custom["branch_contexts"]) == 1

        # Baseline should have no exclusions
        branch = result_context.custom["branch_contexts"][0]
        assert branch["name"] == "baseline"

        # Check exclusion info
        branch_context = branch["context"]
        if "outlier_exclusion" in branch_context.custom:
            exclusion = branch_context.custom["outlier_exclusion"]
            assert exclusion.get("n_excluded", 0) == 0

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_execute_isolation_forest_strategy(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with isolation_forest strategy should detect outliers."""
        # Mock the filter to mark some samples as outliers
        mock_filter = Mock()
        mock_mask = np.array([True, True, False, True, True, True, True, True, False, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "isolation_forest", "contamination": 0.1}]
                }
            },
            metadata={}
        )

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]
        assert len(branches) == 1

        # Check exclusion was applied
        branch_context = branches[0]["context"]
        assert "outlier_exclusion" in branch_context.custom
        exclusion = branch_context.custom["outlier_exclusion"]
        assert exclusion["n_excluded"] == 2  # indices 2 and 8 are False
        assert np.array_equal(exclusion["mask"], mock_mask)

    def test_execute_multiple_strategies(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with multiple strategies creates multiple branches."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [
                        None,
                        {"method": "isolation_forest"},
                        {"method": "mahalanobis"}
                    ]
                }
            },
            metadata={}
        )

        # Mock XOutlierFilter for the non-None strategies
        with patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter") as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_mask.return_value = np.ones(10, dtype=bool)
            mock_filter_class.return_value = mock_filter

            result_context, output = controller.execute(
                step_info=step_info,
                dataset=mock_dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="train"
            )

        branches = result_context.custom["branch_contexts"]
        assert len(branches) == 3

        # Check branch info - names use abbreviated forms
        names = [b["name"] for b in branches]
        assert "baseline" in names
        # Uses abbreviated names: "if" for isolation_forest, "mahal" for mahalanobis
        assert any("if" in n for n in names)
        assert any("mahal" in n for n in names)

    def test_branch_contexts_have_correct_selector(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Each branch context should have correct branch_id/branch_name in selector."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [None, {"method": "lof"}]
                }
            },
            metadata={}
        )

        with patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter") as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_mask.return_value = np.ones(10, dtype=bool)
            mock_filter_class.return_value = mock_filter

            result_context, output = controller.execute(
                step_info=step_info,
                dataset=mock_dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="train"
            )

        branches = result_context.custom["branch_contexts"]
        for branch in branches:
            ctx = branch["context"]
            assert ctx.selector.branch_id == branch["branch_id"]
            assert ctx.selector.branch_name == branch["name"]

    def test_branch_mode_flag_set(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute should set in_branch_mode flag."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [None]
                }
            },
            metadata={}
        )

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        assert result_context.custom.get("in_branch_mode") is True


class TestOutlierExcluderPredict:
    """Test OutlierExcluderController in prediction mode."""

    def test_supports_prediction_mode(self):
        """Controller should support prediction mode."""
        assert OutlierExcluderController.supports_prediction_mode() is True

    def test_use_multi_source(self):
        """Controller should support multi-source datasets."""
        assert OutlierExcluderController.use_multi_source() is True


class TestExclusionMaskApplication:
    """Test outlier exclusion mask handling in branch contexts."""

    @pytest.fixture
    def controller(self):
        return OutlierExcluderController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with spectral data."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        dataset._indexer = indexer

        X = np.random.randn(5, 50)
        dataset.x.return_value = X
        dataset.y.return_value = np.random.randn(5)

        dataset._features = Mock()
        dataset._features.sources = []

        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0
        runtime.step_number = 1
        runtime.operation_count = 0
        runtime.saver = None
        runtime.next_op = Mock(return_value=1)
        return runtime

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_exclusion_stored_in_context(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Exclusion info should be stored in context.custom."""
        # Mock filter that excludes 2 samples
        mock_filter = Mock()
        mock_mask = np.array([True, False, True, False, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "isolation_forest"}]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        branch_context = branch["context"]

        assert "outlier_exclusion" in branch_context.custom
        exclusion = branch_context.custom["outlier_exclusion"]
        assert np.array_equal(exclusion["mask"], mock_mask)
        assert exclusion["n_excluded"] == 2

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_exclusion_rate_calculation(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Exclusion rate should be calculated correctly in exclusion_info."""
        mock_filter = Mock()
        mock_mask = np.array([True, False, True, True, False])  # 2 excluded out of 5
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "lof"}]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        exclusion_info = branch["exclusion_info"]
        assert exclusion_info["exclusion_rate"] == pytest.approx(0.4)  # 2/5


class TestExclusionMetadata:
    """Test exclusion metadata in branch contexts."""

    @pytest.fixture
    def controller(self):
        return OutlierExcluderController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([10, 20, 30, 40, 50])
        dataset._indexer = indexer

        X = np.random.randn(5, 50)
        dataset.x.return_value = X
        dataset.y.return_value = np.random.randn(5)

        dataset._features = Mock()
        dataset._features.sources = []

        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0
        runtime.step_number = 1
        runtime.operation_count = 0
        runtime.saver = None
        runtime.next_op = Mock(return_value=1)
        return runtime

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_exclusion_count_accessible(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Exclusion count should be accessible from branch context."""
        mock_filter = Mock()
        mock_mask = np.array([True, True, False, True, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "mahalanobis"}]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        branch_context = branch["context"]

        exclusion = branch_context.custom["outlier_exclusion"]
        assert exclusion["n_excluded"] == 1

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_excluded_indices_tracked(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Excluded sample indices should be tracked in exclusion_info."""
        mock_filter = Mock()
        mock_mask = np.array([True, False, True, False, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "lof"}]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        exclusion_info = branch["exclusion_info"]

        # Indices 20 and 40 should be excluded (positions 1 and 3 where mask is False)
        assert 20 in exclusion_info["excluded_indices"]
        assert 40 in exclusion_info["excluded_indices"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def controller(self):
        return OutlierExcluderController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        dataset._indexer = indexer

        X = np.random.randn(5, 50)
        dataset.x.return_value = X
        dataset.y.return_value = np.random.randn(5)

        dataset._features = Mock()
        dataset._features.sources = []

        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0
        runtime.step_number = 1
        runtime.operation_count = 0
        runtime.saver = None
        runtime.next_op = Mock(return_value=1)
        return runtime

    def test_empty_strategies_list(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Empty strategies list should return early."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": []
                }
            },
            metadata={}
        )

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should return unchanged context
        assert "branch_contexts" not in result_context.custom or result_context.custom.get("branch_contexts") == []

    def test_unknown_method_raises_error(self):
        """Unknown outlier detection method should raise error."""
        strategy = {"method": "unknown_method"}

        with pytest.raises(ValueError, match="Unknown outlier method"):
            _create_outlier_filter(strategy)

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_no_samples_to_exclude(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """All samples valid (no exclusions) should work correctly."""
        mock_filter = Mock()
        mock_mask = np.ones(5, dtype=bool)  # All True - no exclusions
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "isolation_forest"}]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        exclusion_info = branch["exclusion_info"]

        assert exclusion_info["n_excluded"] == 0
        assert exclusion_info["exclusion_rate"] == 0.0

    @patch("nirs4all.controllers.data.outlier_excluder.XOutlierFilter")
    def test_all_samples_excluded(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """All samples excluded should be handled (edge case)."""
        mock_filter = Mock()
        mock_mask = np.zeros(5, dtype=bool)  # All False - all excluded
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [{"method": "mahalanobis"}]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        exclusion_info = branch["exclusion_info"]

        assert exclusion_info["n_excluded"] == 5
        assert exclusion_info["exclusion_rate"] == 1.0

    def test_baseline_strategy_no_exclusion(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Baseline (None) strategy should not exclude any samples."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "outlier_excluder",
                    "strategies": [None]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branch = result_context.custom["branch_contexts"][0]
        assert branch["name"] == "baseline"

        exclusion_info = branch["exclusion_info"]
        assert exclusion_info["n_excluded"] == 0


class TestBranchMultiplication:
    """Test nested branch context multiplication."""

    @pytest.fixture
    def controller(self):
        return OutlierExcluderController()

    def test_multiply_branch_contexts(self, controller):
        """Test multiplication of existing and new branch contexts."""
        # Existing branches
        existing = [
            {"branch_id": 0, "name": "A", "context": Mock(), "exclusion_info": {}},
            {"branch_id": 1, "name": "B", "context": Mock(), "exclusion_info": {}},
        ]

        # New branches
        new = [
            {"branch_id": 0, "name": "X", "context": Mock(), "exclusion_info": {}},
            {"branch_id": 1, "name": "Y", "context": Mock(), "exclusion_info": {}},
        ]

        # Mock context copy method
        for item in existing + new:
            mock_ctx = Mock()
            mock_selector = Mock()
            mock_selector.branch_id = None
            mock_selector.branch_name = None
            mock_ctx.selector = mock_selector
            mock_ctx.copy = Mock(return_value=mock_ctx)
            item["context"] = mock_ctx

        result = controller._multiply_branch_contexts(existing, new)

        # Should have 2 x 2 = 4 combinations
        assert len(result) == 4

        # Check names are combined
        names = [r["name"] for r in result]
        assert "A_X" in names
        assert "A_Y" in names
        assert "B_X" in names
        assert "B_Y" in names

        # Check flattened IDs are sequential
        ids = [r["branch_id"] for r in result]
        assert ids == [0, 1, 2, 3]

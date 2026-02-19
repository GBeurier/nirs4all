"""
Unit tests for Phase 7.2: Cross-Branch Stacking.

Tests cover:
- CrossBranchCompatibility enum
- BranchPredictionInfo dataclass
- CrossBranchValidationResult dataclass
- CrossBranchValidator validation logic
- Feature alignment across branches
- Sample overlap detection
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from nirs4all.controllers.models.stacking.branch_validator import BranchType
from nirs4all.controllers.models.stacking.crossbranch import (
    BranchPredictionInfo,
    CrossBranchCompatibility,
    CrossBranchValidationResult,
    CrossBranchValidator,
    validate_all_branches_scope,
)
from nirs4all.controllers.models.stacking.exceptions import (
    BranchFeatureAlignmentError,
    IncompatibleBranchSamplesError,
)
from nirs4all.operators.models.meta import BranchScope


class TestCrossBranchCompatibility:
    """Test CrossBranchCompatibility enum."""

    def test_compatible_value(self):
        """COMPATIBLE should indicate full compatibility."""
        assert CrossBranchCompatibility.COMPATIBLE.value == "compatible"

    def test_compatible_with_alignment(self):
        """COMPATIBLE_WITH_ALIGNMENT should indicate alignment needed."""
        compat = CrossBranchCompatibility.COMPATIBLE_WITH_ALIGNMENT
        assert compat.value == "compatible_with_alignment"

    def test_incompatible_samples(self):
        """INCOMPATIBLE_SAMPLES should indicate sample mismatch."""
        compat = CrossBranchCompatibility.INCOMPATIBLE_SAMPLES
        assert compat.value == "incompatible_samples"

    def test_incompatible_partitions(self):
        """INCOMPATIBLE_PARTITIONS should indicate partition mismatch."""
        compat = CrossBranchCompatibility.INCOMPATIBLE_PARTITIONS
        assert compat.value == "incompatible_partitions"

    def test_not_applicable(self):
        """NOT_APPLICABLE should indicate cross-branch not needed."""
        compat = CrossBranchCompatibility.NOT_APPLICABLE
        assert compat.value == "not_applicable"

class TestBranchPredictionInfo:
    """Test BranchPredictionInfo dataclass."""

    def test_basic_info(self):
        """Basic branch prediction info should store all fields."""
        info = BranchPredictionInfo(
            branch_id=0,
            branch_name="branch_0",
            model_names=["PLS", "RF"],
            sample_indices={1, 2, 3, 4, 5},
            n_samples=5,
            n_folds=5,
            branch_type=BranchType.PREPROCESSING
        )
        assert info.branch_id == 0
        assert info.branch_name == "branch_0"
        assert len(info.model_names) == 2
        assert len(info.sample_indices) == 5
        assert info.n_folds == 5

    def test_sample_partitioner_branch(self):
        """Sample partitioner branch should have disjoint samples."""
        info = BranchPredictionInfo(
            branch_id=1,
            branch_name="partition_A",
            model_names=["PLS"],
            sample_indices={1, 2, 3},
            n_samples=3,
            n_folds=3,
            branch_type=BranchType.SAMPLE_PARTITIONER
        )
        assert info.branch_type == BranchType.SAMPLE_PARTITIONER

class TestCrossBranchValidationResult:
    """Test CrossBranchValidationResult dataclass."""

    def test_compatible_result(self):
        """Compatible result should have is_compatible=True."""
        result = CrossBranchValidationResult(
            is_compatible=True,
            compatibility=CrossBranchCompatibility.COMPATIBLE,
            errors=[],
            warnings=[],
            branches={},
            common_samples=set(),
        )
        assert result.is_compatible is True
        assert result.compatibility == CrossBranchCompatibility.COMPATIBLE

    def test_incompatible_result(self):
        """Incompatible result should have is_compatible=False."""
        result = CrossBranchValidationResult(
            is_compatible=False,
            compatibility=CrossBranchCompatibility.INCOMPATIBLE_SAMPLES,
            errors=["Sample mismatch between branches"],
            warnings=[],
            branches={},
            common_samples=set(),
        )
        assert result.is_compatible is False
        assert len(result.errors) == 1

    def test_add_error(self):
        """add_error should append to errors list."""
        result = CrossBranchValidationResult()
        result.add_error("Test error")
        assert "Test error" in result.errors
        assert result.is_compatible is False

    def test_add_warning(self):
        """add_warning should append to warnings list."""
        result = CrossBranchValidationResult()
        result.add_warning("Test warning")
        assert "Test warning" in result.warnings

    def test_total_models_property(self):
        """total_models should sum models across branches."""
        result = CrossBranchValidationResult(
            branches={
                0: BranchPredictionInfo(
                    branch_id=0,
                    branch_name="b0",
                    model_names=["PLS", "RF"],
                    sample_indices={1, 2},
                    n_samples=2,
                    n_folds=2,
                    branch_type=BranchType.PREPROCESSING
                ),
                1: BranchPredictionInfo(
                    branch_id=1,
                    branch_name="b1",
                    model_names=["XGB"],
                    sample_indices={1, 2},
                    n_samples=2,
                    n_folds=2,
                    branch_type=BranchType.PREPROCESSING
                ),
            }
        )
        assert result.total_models == 3

class TestCrossBranchValidator:
    """Test CrossBranchValidator class."""

    @pytest.fixture
    def mock_prediction_store(self):
        """Create a mock prediction store."""
        store = Mock()
        store.filter_predictions = Mock(return_value=[])
        return store

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock()
        context.selector = Mock()
        context.selector.branch_id = None
        context.state = Mock()
        context.state.step_number = 5
        return context

    @pytest.fixture
    def mock_candidates(self):
        """Create mock model candidates."""
        from nirs4all.operators.models.selection import ModelCandidate

        c1 = ModelCandidate(
            model_name='PLS',
            model_classname='PLSRegression',
            step_idx=1,
            branch_id=0,
            branch_name='main'
        )
        c2 = ModelCandidate(
            model_name='RF',
            model_classname='RandomForestRegressor',
            step_idx=2,
            branch_id=0,
            branch_name='main'
        )
        return [c1, c2]

    def test_validator_initialization(self, mock_prediction_store):
        """Validator should initialize with prediction_store."""
        validator = CrossBranchValidator(
            prediction_store=mock_prediction_store,
            log_warnings=False
        )
        assert validator.log_warnings is False

    def test_single_branch_not_applicable(
        self, mock_prediction_store, mock_context, mock_candidates
    ):
        """Single branch should return NOT_APPLICABLE."""
        mock_prediction_store.filter_predictions.return_value = [
            {'model_name': 'PLS', 'branch_id': 0, 'step_idx': 1},
        ]

        validator = CrossBranchValidator(
            prediction_store=mock_prediction_store
        )

        result = validator.validate_cross_branch_stacking(
            source_candidates=mock_candidates,
            context=mock_context
        )

        assert result.compatibility == CrossBranchCompatibility.NOT_APPLICABLE

class TestFeatureAlignment:
    """Test feature alignment functionality."""

    @pytest.fixture
    def mock_prediction_store(self):
        """Create a mock prediction store."""
        store = Mock()
        store.filter_predictions = Mock(return_value=[])
        return store

    def test_align_branch_features_same_samples(self, mock_prediction_store):
        """Feature alignment with same samples should succeed."""
        validator = CrossBranchValidator(
            prediction_store=mock_prediction_store
        )

        # Features from two branches
        branch_features = {
            0: np.array([[1.0], [2.0], [3.0]]),
            1: np.array([[4.0], [5.0], [6.0]]),
        }

        branch_sample_indices = {
            0: [0, 1, 2],
            1: [0, 1, 2],
        }

        target_sample_indices = [0, 1, 2]

        aligned, valid_mask = validator.align_branch_features(
            branch_features=branch_features,
            branch_sample_indices=branch_sample_indices,
            target_sample_indices=target_sample_indices
        )

        assert aligned.shape == (3, 2)
        assert np.all(valid_mask)

    def test_align_branch_features_partial_overlap(self, mock_prediction_store):
        """Feature alignment with partial overlap should handle NaNs."""
        validator = CrossBranchValidator(
            prediction_store=mock_prediction_store
        )

        # Features from two branches with different samples
        branch_features = {
            0: np.array([[1.0], [2.0]]),
            1: np.array([[4.0], [5.0]]),
        }

        branch_sample_indices = {
            0: [0, 1],
            1: [1, 2],
        }

        target_sample_indices = [0, 1, 2]

        aligned, valid_mask = validator.align_branch_features(
            branch_features=branch_features,
            branch_sample_indices=branch_sample_indices,
            target_sample_indices=target_sample_indices
        )

        assert aligned.shape == (3, 2)
        # Sample 1 should be valid (in both branches)
        assert valid_mask[1] == True  # noqa: E712 - numpy bool comparison

class TestGetCrossBranchSources:
    """Test get_cross_branch_sources method."""

    @pytest.fixture
    def mock_prediction_store(self):
        """Create a mock prediction store."""
        store = Mock()
        store.filter_predictions = Mock(return_value=[])
        return store

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock()
        context.selector = Mock()
        context.selector.branch_id = None
        context.state = Mock()
        context.state.step_number = 5
        return context

    def test_get_cross_branch_sources_deduplicates(
        self, mock_prediction_store, mock_context
    ):
        """Should deduplicate candidates by model_name and branch_id."""
        from nirs4all.operators.models.selection import ModelCandidate

        validator = CrossBranchValidator(
            prediction_store=mock_prediction_store
        )

        # Create candidates with duplicates
        candidates = [
            ModelCandidate('PLS', 'PLSRegression', 1, None, 0, 'main'),
            ModelCandidate('PLS', 'PLSRegression', 1, None, 0, 'main'),
            ModelCandidate('RF', 'RandomForestRegressor', 2, None, 0, 'main'),
        ]

        result = validator.get_cross_branch_sources(
            source_candidates=candidates,
            context=mock_context
        )

        # Should have only 2 unique candidates
        unique_keys = {(c.model_name, c.branch_id) for c in result}
        assert len(unique_keys) == 2

class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def mock_prediction_store(self):
        """Create a mock prediction store."""
        store = Mock()
        store.filter_predictions = Mock(return_value=[])
        return store

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock()
        context.selector = Mock()
        context.selector.branch_id = None
        context.state = Mock()
        context.state.step_number = 5
        return context

    def test_validate_all_branches_scope_function(
        self, mock_prediction_store, mock_context
    ):
        """Test the convenience function for ALL_BRANCHES validation."""
        from nirs4all.operators.models.selection import ModelCandidate

        candidates = [
            ModelCandidate('PLS', 'PLSRegression', 1, None, 0, 'main'),
        ]

        result = validate_all_branches_scope(
            prediction_store=mock_prediction_store,
            source_candidates=candidates,
            context=mock_context
        )

        assert isinstance(result, CrossBranchValidationResult)

class TestCrossBranchExceptions:
    """Test cross-branch stacking exceptions."""

    def test_incompatible_branch_samples_error(self):
        """IncompatibleBranchSamplesError should contain branch info."""
        with pytest.raises(IncompatibleBranchSamplesError) as exc_info:
            raise IncompatibleBranchSamplesError(
                branches={0: 50, 1: 30},
                overlap_matrix=None
            )

        error_str = str(exc_info.value)
        # Should mention sample counts
        assert "50" in error_str or "30" in error_str

    def test_branch_feature_alignment_error(self):
        """BranchFeatureAlignmentError should contain feature counts."""
        with pytest.raises(BranchFeatureAlignmentError) as exc_info:
            raise BranchFeatureAlignmentError(
                expected_features=5,
                branch_features={0: 5, 1: 3}
            )

        error_str = str(exc_info.value)
        # Should mention expected features and branch counts
        assert "5" in error_str

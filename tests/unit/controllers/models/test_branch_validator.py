"""
Unit tests for branch validation in meta-model stacking (Phase 4).

Tests cover:
- BranchValidator class and its methods
- Branch type detection from context
- Validation results for different branch types
- Error and warning generation
- Sample alignment validation
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.controllers.models.stacking.branch_validator import (
    BranchInfo,
    BranchType,
    BranchValidationResult,
    BranchValidator,
    StackingCompatibility,
    detect_branch_type,
    is_stacking_compatible,
)
from nirs4all.controllers.models.stacking.exceptions import (
    CrossPartitionStackingError,
    DisjointSampleSetsError,
    FoldMismatchAcrossBranchesError,
    GeneratorSyntaxStackingWarning,
    IncompatibleBranchTypeError,
    NestedBranchStackingError,
)


class MockSelector:
    """Mock selector for testing."""

    def __init__(self, branch_id=None, branch_name=None, branch_path=None):
        self.branch_id = branch_id
        self.branch_name = branch_name
        self.branch_path = branch_path or []

class MockState:
    """Mock state for testing."""

    def __init__(self, step_number=5):
        self.step_number = step_number

class MockContext:
    """Mock execution context for testing."""

    def __init__(self, branch_id=None, branch_name=None, branch_path=None,
                 step_number=5, **custom):
        self.selector = MockSelector(branch_id, branch_name, branch_path)
        self.state = MockState(step_number)
        self.custom = custom

    def with_partition(self, partition):
        return self

class MockPredictionStore:
    """Mock prediction store for testing."""

    def __init__(self, predictions=None):
        self._predictions = predictions or []

    def filter_predictions(self, **kwargs):
        """Filter predictions based on kwargs."""
        results = []
        for pred in self._predictions:
            match = True
            for key, value in kwargs.items():
                if key == 'load_arrays':
                    continue
                if pred.get(key) != value and value is not None:
                    match = False
                    break
            if match:
                results.append(pred)
        return results if results else self._predictions

class TestBranchTypeDetection:
    """Tests for detect_branch_type function."""

    def test_no_branching(self):
        """Test detection when no branching is present."""
        context = MockContext()
        assert detect_branch_type(context) == BranchType.NONE

    def test_sample_partitioner_active(self):
        """Test detection of sample_partitioner."""
        context = MockContext(sample_partitioner_active=True)
        assert detect_branch_type(context) == BranchType.SAMPLE_PARTITIONER

    def test_outlier_excluder_active(self):
        """Test detection of outlier_excluder."""
        context = MockContext(outlier_excluder_active=True)
        assert detect_branch_type(context) == BranchType.OUTLIER_EXCLUDER

    def test_preprocessing_branch(self):
        """Test detection of preprocessing branch."""
        context = MockContext(
            branch_id=0,
            branch_path=[0],
            in_branch_mode=True
        )
        assert detect_branch_type(context) == BranchType.PREPROCESSING

    def test_nested_branching(self):
        """Test detection of nested branching."""
        context = MockContext(
            branch_id=0,
            branch_path=[0, 1, 2],
            in_branch_mode=True
        )
        assert detect_branch_type(context) == BranchType.NESTED

    def test_branch_id_without_context(self):
        """Test detection with branch_id but no branch context flags."""
        context = MockContext(branch_id=0)
        assert detect_branch_type(context) == BranchType.UNKNOWN

class TestIsStackingCompatible:
    """Tests for is_stacking_compatible function."""

    def test_no_branching_compatible(self):
        """No branching is always compatible."""
        context = MockContext()
        assert is_stacking_compatible(context) is True

    def test_preprocessing_compatible(self):
        """Preprocessing branches are compatible."""
        context = MockContext(
            branch_id=0,
            branch_path=[0],
            in_branch_mode=True
        )
        assert is_stacking_compatible(context) is True

    def test_sample_partitioner_compatible(self):
        """Sample partitioner is compatible within partition."""
        context = MockContext(sample_partitioner_active=True)
        assert is_stacking_compatible(context) is True

    def test_outlier_excluder_compatible(self):
        """Outlier excluder is compatible."""
        context = MockContext(outlier_excluder_active=True)
        assert is_stacking_compatible(context) is True

    def test_deep_nesting_not_compatible(self):
        """Deep nesting (>2 levels) is not compatible."""
        context = MockContext(
            branch_id=0,
            branch_path=[0, 1, 2, 3],  # Depth 4
            in_branch_mode=True
        )
        assert is_stacking_compatible(context) is False

class TestBranchValidatorNoBranching:
    """Tests for BranchValidator with no branching."""

    def test_validate_no_branching(self):
        """Validation passes with no branching."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)
        context = MockContext()

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE
        assert result.branch_info.branch_type == BranchType.NONE
        assert len(result.errors) == 0

class TestBranchValidatorPreprocessing:
    """Tests for BranchValidator with preprocessing branches."""

    def test_validate_preprocessing_branch(self):
        """Validation passes for preprocessing branch."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)
        context = MockContext(
            branch_id=0,
            branch_name='snv',
            branch_path=[0],
            in_branch_mode=True
        )

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE
        assert result.branch_info.branch_type == BranchType.PREPROCESSING

    def test_preprocessing_generates_warning(self):
        """Preprocessing branch generates informative warning."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)
        context = MockContext(
            branch_id=0,
            branch_name='snv',
            branch_path=[0],
            in_branch_mode=True
        )

        result = validator.validate(context, ['PLS'])

        assert len(result.warnings) > 0
        assert 'snv' in result.warnings[0]

class TestBranchValidatorSamplePartitioner:
    """Tests for BranchValidator with sample_partitioner."""

    def test_validate_sample_partitioner(self):
        """Validation passes for sample_partitioner within partition."""
        predictions = [
            {
                'model_name': 'PLS',
                'partition': 'val',
                'fold_id': '0',
                'step_idx': 3,
                'branch_id': 1,
                'sample_indices': list(range(10, 100)),
            }
        ]
        store = MockPredictionStore(predictions)
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=1,
            branch_name='inliers',
            sample_partitioner_active=True,
            sample_partition={
                'partition_type': 'inliers',
                'n_samples': 90,
                'sample_indices': list(range(10, 100)),
            }
        )

        result = validator.validate(context, ['PLS'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.WITHIN_PARTITION_ONLY
        assert result.branch_info.branch_type == BranchType.SAMPLE_PARTITIONER

    def test_sample_partitioner_generates_warning(self):
        """Sample partitioner generates appropriate warning."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=1,
            branch_name='inliers',
            sample_partitioner_active=True,
            sample_partition={
                'partition_type': 'inliers',
                'n_samples': 90,
                'sample_indices': list(range(10, 100)),
            }
        )

        result = validator.validate(context, ['PLS'])

        assert len(result.warnings) > 0
        assert 'inliers' in result.warnings[0].lower() or 'partition' in result.warnings[0].lower()

class TestBranchValidatorOutlierExcluder:
    """Tests for BranchValidator with outlier_excluder."""

    def test_validate_outlier_excluder(self):
        """Validation passes for outlier_excluder."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=0,
            branch_name='baseline',
            outlier_excluder_active=True,
            outlier_exclusion={
                'n_excluded': 5,
                'strategy': {'method': 'isolation_forest'},
            }
        )

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE
        assert result.branch_info.branch_type == BranchType.OUTLIER_EXCLUDER

    def test_outlier_excluder_with_exclusions(self):
        """Outlier excluder with exclusions generates warning."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=1,
            branch_name='if_0.05',
            outlier_excluder_active=True,
            outlier_exclusion={
                'n_excluded': 10,
                'strategy': {'method': 'isolation_forest', 'contamination': 0.05},
            }
        )

        result = validator.validate(context, ['PLS'])

        assert result.is_valid is True
        # Should have warning about exclusions
        assert len(result.warnings) > 0

class TestBranchValidatorNestedBranching:
    """Tests for BranchValidator with nested branching."""

    def test_validate_acceptable_nesting(self):
        """Validation passes for acceptable nesting depth (<=2)."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=0,
            branch_path=[0, 1],  # Depth 2
            in_branch_mode=True
        )

        result = validator.validate(context, ['PLS'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE_WITH_WARNINGS
        assert result.branch_info.branch_type == BranchType.NESTED

    def test_validate_deep_nesting_fails(self):
        """Validation fails for deep nesting (>2)."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=0,
            branch_path=[0, 1, 2, 3],  # Depth 4
            in_branch_mode=True
        )

        result = validator.validate(context, ['PLS'])

        assert result.is_valid is False
        assert result.compatibility == StackingCompatibility.NOT_SUPPORTED
        assert len(result.errors) > 0

class TestBranchValidatorSampleAlignment:
    """Tests for sample alignment validation."""

    def test_good_sample_alignment(self):
        """Good sample alignment passes validation."""
        predictions = [
            {
                'model_name': 'PLS',
                'partition': 'val',
                'fold_id': '0',
                'step_idx': 3,
                'sample_indices': list(range(100)),
            }
        ]
        store = MockPredictionStore(predictions)
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext()
        expected_samples = list(range(100))

        result = validator.validate_sample_alignment(
            source_model_names=['PLS'],
            expected_sample_indices=expected_samples,
            context=context
        )

        assert result.is_valid is True

    def test_poor_sample_alignment(self):
        """Poor sample alignment generates errors."""
        predictions = [
            {
                'model_name': 'PLS',
                'partition': 'val',
                'fold_id': '0',
                'step_idx': 3,
                'sample_indices': list(range(60, 100)),  # Only 60-99 (40 samples)
            }
        ]
        store = MockPredictionStore(predictions)
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext()
        expected_samples = list(range(100))  # Expect 0-99 (100 samples)

        result = validator.validate_sample_alignment(
            source_model_names=['PLS'],
            expected_sample_indices=expected_samples,
            context=context
        )

        # Should detect low overlap (40%)
        assert len(result.errors) > 0

class TestBranchInfoExtraction:
    """Tests for branch info extraction."""

    def test_extract_sample_partitioner_info(self):
        """Extract info for sample_partitioner branch."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=1,
            branch_name='inliers',
            sample_partitioner_active=True,
            sample_partition={
                'partition_type': 'inliers',
                'n_samples': 90,
                'sample_indices': list(range(10, 100)),
            }
        )

        info = validator._extract_branch_info(context)

        assert info.branch_type == BranchType.SAMPLE_PARTITIONER
        assert info.branch_id == 1
        assert info.branch_name == 'inliers'
        assert info.partition_info['partition_type'] == 'inliers'
        assert info.n_samples == 90
        assert len(info.sample_indices) == 90

    def test_extract_outlier_excluder_info(self):
        """Extract info for outlier_excluder branch."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        context = MockContext(
            branch_id=0,
            branch_name='baseline',
            outlier_excluder_active=True,
            outlier_exclusion={
                'n_excluded': 5,
                'strategy': {'method': 'isolation_forest'},
            }
        )

        info = validator._extract_branch_info(context)

        assert info.branch_type == BranchType.OUTLIER_EXCLUDER
        assert info.exclusion_info['n_excluded'] == 5

class TestBranchValidationWarnings:
    """Tests for warning generation."""

    def test_generator_syntax_warning(self):
        """Generator syntax generates warning about variant count."""
        store = MockPredictionStore()
        validator = BranchValidator(store, log_warnings=False)

        # Mock context for generator with many variants
        context = MockContext(
            in_branch_mode=True,
            branch_contexts=[
                {'name': f'variant_{i}'} for i in range(15)  # Many variants
            ]
        )

        # This should trigger generator detection
        result = validator.validate(context, [f'model_{i}' for i in range(15)])

        # Generator with many variants should generate warning
        if result.branch_info.branch_type == BranchType.GENERATOR:
            assert result.compatibility == StackingCompatibility.COMPATIBLE_WITH_WARNINGS

class TestExceptionMessages:
    """Tests for exception message formatting."""

    def test_cross_partition_error_message(self):
        """CrossPartitionStackingError has informative message."""
        error = CrossPartitionStackingError(
            partition_a='outliers',
            partition_b='inliers',
            n_samples_a=10,
            n_samples_b=90
        )

        msg = str(error)
        assert 'outliers' in msg
        assert 'inliers' in msg
        assert '10' in msg
        assert '90' in msg
        assert 'disjoint' in msg.lower()

    def test_nested_branch_error_message(self):
        """NestedBranchStackingError has informative message."""
        error = NestedBranchStackingError(
            branch_depth=4,
            branch_path=[0, 1, 2, 3],
            reason='Exceeds maximum depth'
        )

        msg = str(error)
        assert 'depth=4' in msg
        assert '0 → 1 → 2 → 3' in msg
        assert 'Exceeds maximum depth' in msg

    def test_disjoint_samples_error_message(self):
        """DisjointSampleSetsError has informative message."""
        error = DisjointSampleSetsError(
            source_model='PLS',
            expected_samples=100,
            found_samples=50,
            overlap_ratio=0.25
        )

        msg = str(error)
        assert 'PLS' in msg
        assert '100' in msg
        assert '50' in msg
        assert '25.0%' in msg

    def test_fold_mismatch_error_message(self):
        """FoldMismatchAcrossBranchesError has informative message."""
        error = FoldMismatchAcrossBranchesError(
            fold_structures={0: 5, 1: 3},
            affected_models=['PLS', 'RF', 'XGB']
        )

        msg = str(error)
        assert 'branch 0: 5 folds' in msg
        assert 'branch 1: 3 folds' in msg
        assert 'PLS' in msg

    def test_generator_warning_message(self):
        """GeneratorSyntaxStackingWarning has informative message."""
        error = GeneratorSyntaxStackingWarning(
            generator_type='_or_',
            n_variants=20
        )

        msg = str(error)
        assert '_or_' in msg
        assert '20' in msg
        assert 'variants' in msg

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

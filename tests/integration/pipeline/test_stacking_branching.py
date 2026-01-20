"""
Integration tests for meta-model stacking with branching (Phase 4).

Tests cover:
- Preprocessing branch stacking (same samples, different features)
- Sample partitioner branch stacking (different sample subsets)
- Outlier excluder branch stacking (same samples, different exclusions)
- Generator syntax with stacking
- Nested branching with stacking
- Error cases and edge scenarios

These tests ensure that meta-model stacking works correctly with all
supported branch types and provides clear error messages for unsupported
scenarios.
"""

import pytest
import numpy as np
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold

# Stacking imports
from nirs4all.controllers.models.stacking import (
    BranchValidator,
    BranchType,
    BranchInfo,
    BranchValidationResult,
    StackingCompatibility,
    detect_branch_type,
    is_stacking_compatible,
    # Exceptions
    IncompatibleBranchTypeError,
    CrossPartitionStackingError,
    NestedBranchStackingError,
    DisjointSampleSetsError,
)
from nirs4all.operators.models.meta import (
    MetaModel,
    StackingConfig,
    CoverageStrategy,
    BranchScope,
)


class TestBranchTypeDetection:
    """Tests for branch type detection from execution context."""

    def test_detect_no_branching(self):
        """Test detection of no branching."""
        context = self._create_mock_context()
        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.NONE

    def test_detect_sample_partitioner(self):
        """Test detection of sample_partitioner branching."""
        context = self._create_mock_context()
        context.custom['sample_partitioner_active'] = True
        context.custom['sample_partition'] = {
            'partition_type': 'outliers',
            'n_samples': 10,
            'sample_indices': list(range(10)),
        }

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.SAMPLE_PARTITIONER

    def test_detect_outlier_excluder(self):
        """Test detection of outlier_excluder branching."""
        context = self._create_mock_context()
        context.custom['outlier_excluder_active'] = True
        context.custom['outlier_exclusion'] = {
            'n_excluded': 5,
            'strategy': {'method': 'isolation_forest'},
        }

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.OUTLIER_EXCLUDER

    def test_detect_preprocessing_branch(self):
        """Test detection of preprocessing branching."""
        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_id = 0
        context.selector.branch_name = 'snv'
        context.selector.branch_path = [0]

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.PREPROCESSING

    def test_detect_nested_branching(self):
        """Test detection of nested branching."""
        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_id = 0
        context.selector.branch_path = [0, 1, 2]  # Depth 3

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.NESTED

    def test_is_stacking_compatible_no_branch(self):
        """Test stacking compatibility with no branching."""
        context = self._create_mock_context()
        assert is_stacking_compatible(context) is True

    def test_is_stacking_compatible_preprocessing(self):
        """Test stacking compatibility with preprocessing branch."""
        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_id = 0
        context.selector.branch_path = [0]

        assert is_stacking_compatible(context) is True

    def test_is_stacking_compatible_sample_partitioner(self):
        """Test stacking compatibility with sample_partitioner."""
        context = self._create_mock_context()
        context.custom['sample_partitioner_active'] = True

        assert is_stacking_compatible(context) is True  # Within partition

    def test_is_stacking_compatible_deep_nesting(self):
        """Test stacking compatibility with deep nesting."""
        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_path = [0, 1, 2, 3]  # Depth 4 (too deep)

        # Deep nesting should still return True but with warnings
        assert is_stacking_compatible(context) is False

    def _create_mock_context(self):
        """Create a mock execution context."""
        context = MagicMock()
        context.custom = {}
        context.selector = MagicMock()
        context.selector.branch_id = None
        context.selector.branch_name = None
        context.selector.branch_path = []
        context.state = MagicMock()
        context.state.step_number = 5
        return context


class TestBranchValidator:
    """Tests for BranchValidator class."""

    def test_validate_no_branching(self):
        """Test validation with no branching."""
        prediction_store = self._create_mock_prediction_store()
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE
        assert result.branch_info.branch_type == BranchType.NONE

    def test_validate_preprocessing_branch(self):
        """Test validation with preprocessing branch."""
        prediction_store = self._create_mock_prediction_store()
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_id = 0
        context.selector.branch_name = 'snv'
        context.selector.branch_path = [0]

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE
        assert result.branch_info.branch_type == BranchType.PREPROCESSING

    def test_validate_sample_partitioner(self):
        """Test validation with sample_partitioner."""
        prediction_store = self._create_mock_prediction_store()
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        context.custom['sample_partitioner_active'] = True
        context.custom['sample_partition'] = {
            'partition_type': 'inliers',
            'n_samples': 90,
            'sample_indices': list(range(10, 100)),
        }
        context.selector.branch_id = 1
        context.selector.branch_name = 'inliers'

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.WITHIN_PARTITION_ONLY
        assert result.branch_info.branch_type == BranchType.SAMPLE_PARTITIONER

    def test_validate_outlier_excluder(self):
        """Test validation with outlier_excluder."""
        prediction_store = self._create_mock_prediction_store()
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        context.custom['outlier_excluder_active'] = True
        context.custom['outlier_exclusion'] = {
            'n_excluded': 5,
            'strategy': {'method': 'isolation_forest'},
        }
        context.selector.branch_id = 0
        context.selector.branch_name = 'baseline'

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE
        assert result.branch_info.branch_type == BranchType.OUTLIER_EXCLUDER

    def test_validate_nested_branching_acceptable_depth(self):
        """Test validation with acceptable nested branching depth."""
        prediction_store = self._create_mock_prediction_store()
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_path = [0, 1]  # Depth 2 (acceptable)

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is True
        assert result.compatibility == StackingCompatibility.COMPATIBLE_WITH_WARNINGS
        assert result.branch_info.branch_type == BranchType.NESTED

    def test_validate_nested_branching_too_deep(self):
        """Test validation with nested branching too deep."""
        prediction_store = self._create_mock_prediction_store()
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        context.custom['in_branch_mode'] = True
        context.selector.branch_path = [0, 1, 2, 3]  # Depth 4 (too deep)

        result = validator.validate(context, ['PLS', 'RF'])

        assert result.is_valid is False
        assert result.compatibility == StackingCompatibility.NOT_SUPPORTED
        assert len(result.errors) > 0

    def test_validate_sample_alignment(self):
        """Test sample alignment validation."""
        # Predictions only for samples 60-99 (40 samples out of 100 = 40% overlap < 50%)
        prediction_store = self._create_mock_prediction_store(
            sample_indices=list(range(60, 100))
        )
        validator = BranchValidator(prediction_store, log_warnings=False)

        context = self._create_mock_context()
        context.selector.branch_id = None

        # Expected samples 0-99, but predictions only for 60-99 (40% overlap)
        expected_samples = list(range(100))
        result = validator.validate_sample_alignment(
            source_model_names=['PLS'],
            expected_sample_indices=expected_samples,
            context=context
        )

        # Should detect low overlap (< 50%)
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def _create_mock_context(self):
        """Create a mock execution context."""
        context = MagicMock()
        context.custom = {}
        context.selector = MagicMock()
        context.selector.branch_id = None
        context.selector.branch_name = None
        context.selector.branch_path = []
        context.state = MagicMock()
        context.state.step_number = 5
        return context

    def _create_mock_prediction_store(self, sample_indices=None):
        """Create a mock prediction store."""
        store = MagicMock()

        def filter_predictions(**kwargs):
            result = [{
                'model_name': kwargs.get('model_name', 'PLS'),
                'partition': kwargs.get('partition', 'val'),
                'fold_id': '0',
                'step_idx': 3,
                'branch_id': kwargs.get('branch_id'),
                'sample_indices': sample_indices or list(range(100)),
            }]
            return result

        store.filter_predictions = filter_predictions
        return store


class TestBranchingExceptions:
    """Tests for branching-related exception classes."""

    def test_cross_partition_stacking_error(self):
        """Test CrossPartitionStackingError message."""
        error = CrossPartitionStackingError(
            partition_a='outliers',
            partition_b='inliers',
            n_samples_a=10,
            n_samples_b=90
        )

        assert 'outliers' in str(error)
        assert 'inliers' in str(error)
        assert '10' in str(error)
        assert '90' in str(error)
        assert error.partition_a == 'outliers'
        assert error.partition_b == 'inliers'

    def test_nested_branch_stacking_error(self):
        """Test NestedBranchStackingError message."""
        error = NestedBranchStackingError(
            branch_depth=4,
            branch_path=[0, 1, 2, 3],
            reason='Too deep for stacking'
        )

        assert 'depth=4' in str(error)
        assert '0 → 1 → 2 → 3' in str(error)
        assert 'Too deep' in str(error)
        assert error.branch_depth == 4

    def test_incompatible_branch_type_error(self):
        """Test IncompatibleBranchTypeError message."""
        error = IncompatibleBranchTypeError(
            branch_type='unknown',
            reason='Branch type not supported',
            suggestions=['Use explicit source_models', 'Simplify pipeline']
        )

        assert 'unknown' in str(error)
        assert 'not supported' in str(error)
        assert 'Suggestions' in str(error)
        assert len(error.suggestions) == 2

    def test_disjoint_sample_sets_error(self):
        """Test DisjointSampleSetsError message."""
        error = DisjointSampleSetsError(
            source_model='PLS',
            expected_samples=100,
            found_samples=50,
            overlap_ratio=0.0
        )

        assert 'PLS' in str(error)
        assert '100' in str(error)
        assert '50' in str(error)
        assert '0.0%' in str(error)


class TestBranchInfoDataclass:
    """Tests for BranchInfo dataclass."""

    def test_branch_info_defaults(self):
        """Test BranchInfo default values."""
        info = BranchInfo(branch_type=BranchType.NONE)

        assert info.branch_type == BranchType.NONE
        assert info.branch_id is None
        assert info.branch_name is None
        assert info.branch_path == []
        assert info.partition_info is None
        assert info.exclusion_info is None
        assert info.sample_indices is None
        assert info.n_samples is None
        assert info.is_nested is False
        assert info.nesting_depth == 0

    def test_branch_info_sample_partitioner(self):
        """Test BranchInfo for sample_partitioner."""
        info = BranchInfo(
            branch_type=BranchType.SAMPLE_PARTITIONER,
            branch_id=1,
            branch_name='inliers',
            partition_info={'partition_type': 'inliers'},
            sample_indices=list(range(90)),
            n_samples=90
        )

        assert info.branch_type == BranchType.SAMPLE_PARTITIONER
        assert info.branch_id == 1
        assert info.n_samples == 90

    def test_branch_info_nested(self):
        """Test BranchInfo for nested branching."""
        info = BranchInfo(
            branch_type=BranchType.NESTED,
            branch_path=[0, 1, 2],
            is_nested=True,
            nesting_depth=3
        )

        assert info.branch_type == BranchType.NESTED
        assert info.is_nested is True
        assert info.nesting_depth == 3


class TestBranchValidationResult:
    """Tests for BranchValidationResult dataclass."""

    def test_add_error(self):
        """Test adding errors to validation result."""
        result = BranchValidationResult(
            is_valid=True,
            compatibility=StackingCompatibility.COMPATIBLE,
            branch_info=BranchInfo(branch_type=BranchType.NONE)
        )

        assert result.is_valid is True
        result.add_error("Test error message")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error message"

    def test_add_warning(self):
        """Test adding warnings to validation result."""
        result = BranchValidationResult(
            is_valid=True,
            compatibility=StackingCompatibility.COMPATIBLE,
            branch_info=BranchInfo(branch_type=BranchType.NONE)
        )

        result.add_warning("Test warning message")

        assert result.is_valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning message"


class TestStackingWithBranchTypes:
    """Integration tests for MetaModel operator with branch types."""

    def test_meta_model_operator_creation(self):
        """Test MetaModel operator creation."""
        meta = MetaModel(
            model=Ridge(),
            source_models="all",
            stacking_config=StackingConfig(
                branch_scope=BranchScope.CURRENT_ONLY
            )
        )

        assert meta.model is not None
        assert meta.source_models == "all"
        assert meta.stacking_config.branch_scope == BranchScope.CURRENT_ONLY

    def test_meta_model_with_explicit_sources(self):
        """Test MetaModel with explicit source model list."""
        meta = MetaModel(
            model=Ridge(),
            source_models=["PLS", "RandomForest", "XGBoost"]
        )

        assert meta.source_models == ["PLS", "RandomForest", "XGBoost"]

    def test_stacking_config_branch_scope(self):
        """Test StackingConfig branch scope options."""
        config_current = StackingConfig(branch_scope=BranchScope.CURRENT_ONLY)
        config_all = StackingConfig(branch_scope=BranchScope.ALL_BRANCHES)
        config_specified = StackingConfig(branch_scope=BranchScope.SPECIFIED)

        assert config_current.branch_scope == BranchScope.CURRENT_ONLY
        assert config_all.branch_scope == BranchScope.ALL_BRANCHES
        assert config_specified.branch_scope == BranchScope.SPECIFIED


class TestStackingCompatibilityEnum:
    """Tests for StackingCompatibility enum."""

    def test_compatibility_values(self):
        """Test StackingCompatibility enum values."""
        assert StackingCompatibility.COMPATIBLE.value == "compatible"
        assert StackingCompatibility.COMPATIBLE_WITH_WARNINGS.value == "compatible_with_warnings"
        assert StackingCompatibility.WITHIN_PARTITION_ONLY.value == "within_partition_only"
        assert StackingCompatibility.NOT_SUPPORTED.value == "not_supported"


class TestBranchTypeEnum:
    """Tests for BranchType enum."""

    def test_branch_type_values(self):
        """Test BranchType enum values."""
        assert BranchType.NONE.value == "none"
        assert BranchType.PREPROCESSING.value == "preprocessing"
        assert BranchType.SAMPLE_PARTITIONER.value == "sample_partitioner"
        assert BranchType.OUTLIER_EXCLUDER.value == "outlier_excluder"
        assert BranchType.GENERATOR.value == "generator"
        assert BranchType.NESTED.value == "nested"
        assert BranchType.UNKNOWN.value == "unknown"


class TestV2SeparationBranchDetection:
    """Tests for v2 BranchController separation branch detection.

    The new unified BranchController sets different context flags:
    - custom['branch_type'] = 'separation'
    - custom['separation_type'] = 'by_tag' | 'by_metadata' | 'by_filter' | 'by_source'
    - custom['sample_partition'] = {'sample_indices': [...], 'n_samples': N, ...}
    """

    def test_detect_by_tag_separation(self):
        """Test detection of by_tag separation branch."""
        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_tag'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(50)),
            'n_samples': 50,
            'separation_key': 'y_outlier_iqr',
        }

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.SAMPLE_PARTITIONER

    def test_detect_by_metadata_separation(self):
        """Test detection of by_metadata separation branch."""
        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_metadata'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(30)),
            'n_samples': 30,
            'separation_key': 'site',
        }

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.METADATA_PARTITIONER

    def test_detect_by_filter_separation(self):
        """Test detection of by_filter separation branch."""
        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_filter'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(80)),
            'n_samples': 80,
            'separation_key': 'YOutlierFilter',
        }

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.OUTLIER_EXCLUDER

    def test_detect_by_source_separation(self):
        """Test detection of by_source separation branch."""
        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_source'
        context.custom['in_source_branch_mode'] = True

        branch_type = detect_branch_type(context)
        assert branch_type == BranchType.PREPROCESSING

    def test_is_stacking_compatible_by_tag(self):
        """Test stacking compatibility with by_tag separation."""
        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_tag'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(50)),
            'n_samples': 50,
        }

        # by_tag creates disjoint samples, compatible within partition
        assert is_stacking_compatible(context) is True

    def test_is_stacking_compatible_by_source(self):
        """Test stacking compatibility with by_source separation."""
        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_source'

        # by_source is per-source preprocessing, fully compatible
        assert is_stacking_compatible(context) is True

    def test_validate_by_metadata_separation(self):
        """Test validation with by_metadata separation branch."""
        from nirs4all.controllers.models.stacking import (
            is_disjoint_branch,
            get_disjoint_branch_info,
        )

        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_metadata'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(30)),
            'n_samples': 30,
            'separation_key': 'site',
        }
        context.selector.branch_name = 'site_A'

        # by_metadata should be disjoint
        assert is_disjoint_branch(context) is True

        # Should return disjoint info
        info = get_disjoint_branch_info(context)
        assert info is not None
        assert info['partition_type'] == 'metadata'
        assert info['separation_type'] == 'by_metadata'
        assert info['column'] == 'site'
        assert info['partition_value'] == 'site_A'
        assert info['n_samples'] == 30

    def test_validate_by_tag_separation(self):
        """Test validation with by_tag separation branch."""
        from nirs4all.controllers.models.stacking import (
            is_disjoint_branch,
            get_disjoint_branch_info,
        )

        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_tag'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(20)),
            'n_samples': 20,
            'separation_key': 'y_outlier_iqr',
        }
        context.selector.branch_name = 'outliers'

        # by_tag should be disjoint
        assert is_disjoint_branch(context) is True

        # Should return disjoint info
        info = get_disjoint_branch_info(context)
        assert info is not None
        assert info['partition_type'] == 'tag'
        assert info['separation_type'] == 'by_tag'
        assert info['tag_name'] == 'y_outlier_iqr'
        assert info['partition_value'] == 'outliers'
        assert info['n_samples'] == 20

    def test_validate_by_filter_separation(self):
        """Test validation with by_filter separation branch."""
        from nirs4all.controllers.models.stacking import (
            is_disjoint_branch,
            get_disjoint_branch_info,
        )

        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_filter'
        context.custom['sample_partition'] = {
            'sample_indices': list(range(80)),
            'n_samples': 80,
            'separation_key': 'YOutlierFilter',
        }
        context.selector.branch_name = 'passing'

        # by_filter should be disjoint
        assert is_disjoint_branch(context) is True

        # Should return disjoint info
        info = get_disjoint_branch_info(context)
        assert info is not None
        assert info['partition_type'] == 'filter'
        assert info['separation_type'] == 'by_filter'
        assert info['filter_class'] == 'YOutlierFilter'
        assert info['partition_value'] == 'passing'
        assert info['n_samples'] == 80

    def test_validate_by_source_not_disjoint(self):
        """Test that by_source separation is NOT disjoint (all samples, different features)."""
        from nirs4all.controllers.models.stacking import (
            is_disjoint_branch,
            get_disjoint_branch_info,
        )

        context = self._create_mock_context()
        context.custom['branch_type'] = 'separation'
        context.custom['separation_type'] = 'by_source'
        context.selector.branch_name = 'NIR'

        # by_source is NOT disjoint - all samples, different features
        assert is_disjoint_branch(context) is False

        # Should return None for non-disjoint
        info = get_disjoint_branch_info(context)
        assert info is None

    def _create_mock_context(self):
        """Create a mock execution context for v2 separation branch testing."""
        context = MagicMock()
        context.custom = {}
        context.selector = MagicMock()
        context.selector.branch_id = 0
        context.selector.branch_name = None
        context.selector.branch_path = [0]
        context.state = MagicMock()
        context.state.step_number = 5
        return context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

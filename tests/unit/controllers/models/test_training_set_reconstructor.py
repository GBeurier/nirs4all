"""
Unit tests for Phase 2 stacking components: TrainingSetReconstructor.

Tests cover:
- TrainingSetReconstructor OOF prediction collection
- FoldAlignmentValidator fold structure validation
- Coverage strategies (STRICT, DROP_INCOMPLETE, IMPUTE_*)
- Test prediction aggregation (MEAN, WEIGHTED_MEAN, BEST_FOLD)
- Branch-aware reconstruction
- ValidationResult error/warning handling
- ReconstructionResult data structure
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.controllers.models.stacking import (
    FoldAlignmentValidator,
    ReconstructionResult,
    ReconstructorConfig,
    TrainingSetReconstructor,
    ValidationResult,
)
from nirs4all.operators.models.meta import (
    BranchScope,
    CoverageStrategy,
    StackingConfig,
    TestAggregation,
)

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@dataclass
class MockSelector:
    """Mock selector with branch_id and partition."""
    branch_id: int | None = None
    branch_name: str | None = None
    partition: str = "train"

@dataclass
class MockState:
    """Mock execution state."""
    step_number: int = 5
    mode: str = "train"

class MockExecutionContext:
    """Mock ExecutionContext for testing."""

    def __init__(self, step_number=5, branch_id=None, branch_name=None, mode="train"):
        self.selector = MockSelector(branch_id=branch_id, branch_name=branch_name)
        self.state = MockState(step_number=step_number, mode=mode)
        self.custom = {}

    def with_partition(self, partition):
        """Return copy with updated partition."""
        new_ctx = MockExecutionContext(
            step_number=self.state.step_number,
            branch_id=self.selector.branch_id,
            branch_name=self.selector.branch_name,
            mode=self.state.mode,
        )
        new_ctx.selector.partition = partition
        new_ctx.custom = self.custom.copy()
        return new_ctx

class MockIndexer:
    """Mock dataset indexer."""

    def __init__(self, n_train=80, n_test=20):
        self.n_train = n_train
        self.n_test = n_test

    def x_indices(self, selector, include_augmented=True, include_excluded=False):
        if selector.partition == "test":
            return list(range(self.n_train, self.n_train + self.n_test))
        return list(range(self.n_train))

class MockDataset:
    """Mock SpectroDataset for testing."""

    def __init__(self, n_train=80, n_test=20):
        self._indexer = MockIndexer(n_train, n_test)
        self._y_train = np.random.randn(n_train)
        self._y_test = np.random.randn(n_test)

    def y(self, selector, include_augmented=True, include_excluded=False):
        if selector.partition == "test":
            return self._y_test
        return self._y_train

class MockPredictionStore:
    """Mock Predictions store for testing."""

    def __init__(self):
        self._predictions = []

    def add_prediction(
        self,
        model_name: str,
        partition: str,
        fold_id: int,
        step_idx: int,
        sample_indices: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray | None = None,
        branch_id: int | None = None,
        branch_name: str | None = None,
        val_score: float | None = None,
        **kwargs
    ):
        """Add a prediction entry."""
        pred = {
            'model_name': model_name,
            'partition': partition,
            'fold_id': fold_id,
            'step_idx': step_idx,
            'sample_indices': sample_indices,
            'y_pred': y_pred,
            'y_true': y_true or y_pred,
            'branch_id': branch_id,
            'branch_name': branch_name,
            'val_score': val_score,
            **kwargs
        }
        self._predictions.append(pred)

    def filter_predictions(
        self,
        model_name=None,
        partition=None,
        fold_id=None,
        step_idx=None,
        branch_id=None,
        branch_name=None,
        load_arrays=True,
        **kwargs
    ):
        """Filter predictions by criteria."""
        results = []
        for pred in self._predictions:
            if model_name is not None and pred.get('model_name') != model_name:
                continue
            if partition is not None and pred.get('partition') != partition:
                continue
            if fold_id is not None and pred.get('fold_id') != fold_id:
                continue
            if step_idx is not None and pred.get('step_idx') != step_idx:
                continue
            if branch_id is not None and pred.get('branch_id') != branch_id:
                continue
            if branch_name is not None and pred.get('branch_name') != branch_name:
                continue
            results.append(pred.copy())
        return results

def create_mock_predictions_5fold(
    prediction_store: MockPredictionStore,
    model_name: str,
    step_idx: int,
    n_train: int = 80,
    n_test: int = 20,
    n_folds: int = 5,
    branch_id: int | None = None,
    val_scores: list | None = None,
):
    """Create 5-fold cross-validation predictions for a model.

    Each fold has 16 validation samples for 80 total training samples.
    """
    samples_per_fold = n_train // n_folds

    if val_scores is None:
        val_scores = [0.9 - 0.1 * i for i in range(n_folds)]

    for fold_id in range(n_folds):
        # Validation indices for this fold
        val_start = fold_id * samples_per_fold
        val_end = val_start + samples_per_fold
        val_indices = np.arange(val_start, val_end)

        # Mock predictions (just fold_id * 0.1 + sample_idx * 0.01 for testing)
        y_pred = fold_id * 0.1 + val_indices * 0.01

        prediction_store.add_prediction(
            model_name=model_name,
            partition='val',
            fold_id=fold_id,
            step_idx=step_idx,
            sample_indices=val_indices,
            y_pred=y_pred,
            branch_id=branch_id,
            val_score=val_scores[fold_id],
        )

        # Test predictions (same for all folds, but with slight variation)
        test_indices = np.arange(n_train, n_train + n_test)
        test_pred = fold_id * 0.1 + test_indices * 0.01

        prediction_store.add_prediction(
            model_name=model_name,
            partition='test',
            fold_id=fold_id,
            step_idx=step_idx,
            sample_indices=test_indices,
            y_pred=test_pred,
            branch_id=branch_id,
            val_score=val_scores[fold_id],
        )

# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_makes_invalid(self):
        """Test that adding error makes result invalid."""
        result = ValidationResult()
        result.add_error("TEST_ERROR", "Test error message")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "TEST_ERROR"

    def test_add_warning_stays_valid(self):
        """Test that adding warning keeps result valid."""
        result = ValidationResult()
        result.add_warning("TEST_WARNING", "Test warning message")
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].code == "TEST_WARNING"

    def test_merge_results(self):
        """Test merging two validation results."""
        result1 = ValidationResult()
        result1.add_error("ERROR1", "Error 1")
        result1.add_warning("WARNING1", "Warning 1")

        result2 = ValidationResult()
        result2.add_error("ERROR2", "Error 2")

        result1.merge(result2)

        assert len(result1.errors) == 2
        assert len(result1.warnings) == 1
        assert result1.is_valid is False

    def test_format_errors(self):
        """Test formatting errors as string."""
        result = ValidationResult()
        result.add_error("ERROR1", "First error")
        result.add_error("ERROR2", "Second error")

        formatted = result.format_errors()
        assert "ERROR1" in formatted
        assert "First error" in formatted
        assert "ERROR2" in formatted

    def test_format_warnings(self):
        """Test formatting warnings as string."""
        result = ValidationResult()
        result.add_warning("WARNING1", "First warning")

        formatted = result.format_warnings()
        assert "WARNING1" in formatted
        assert "First warning" in formatted

# =============================================================================
# ReconstructorConfig Tests
# =============================================================================

class TestReconstructorConfig:
    """Test ReconstructorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReconstructorConfig()
        assert config.validate_fold_alignment is True
        assert config.validate_sample_coverage is True
        assert config.log_warnings is True
        assert config.max_missing_fold_ratio == 0.0
        assert 'avg' in config.excluded_fold_ids
        assert 'w_avg' in config.excluded_fold_ids

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReconstructorConfig(
            validate_fold_alignment=False,
            max_missing_fold_ratio=0.2,
            feature_name_pattern="{model_name}_{classname}"
        )
        assert config.validate_fold_alignment is False
        assert config.max_missing_fold_ratio == 0.2
        assert config.feature_name_pattern == "{model_name}_{classname}"

    def test_invalid_max_missing_fold_ratio(self):
        """Test that invalid max_missing_fold_ratio raises error."""
        with pytest.raises(ValueError, match="max_missing_fold_ratio"):
            ReconstructorConfig(max_missing_fold_ratio=1.5)

    def test_excluded_fold_ids_list_converted_to_set(self):
        """Test that excluded_fold_ids list is converted to set."""
        config = ReconstructorConfig(excluded_fold_ids=['avg', 'test'])
        assert isinstance(config.excluded_fold_ids, set)
        assert 'avg' in config.excluded_fold_ids
        assert 'test' in config.excluded_fold_ids

# =============================================================================
# FoldAlignmentValidator Tests
# =============================================================================

class TestFoldAlignmentValidator:
    """Test FoldAlignmentValidator."""

    def test_validate_consistent_folds(self):
        """Test validation passes for consistent folds."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2)
        create_mock_predictions_5fold(store, "RF", step_idx=3)

        validator = FoldAlignmentValidator(store)
        context = MockExecutionContext(step_number=5)

        result = validator.validate(["PLS", "RF"], context)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_no_source_models_error(self):
        """Test validation error when no source models provided."""
        store = MockPredictionStore()
        validator = FoldAlignmentValidator(store)
        context = MockExecutionContext()

        result = validator.validate([], context)

        assert result.is_valid is False
        assert any(e.code == "NO_SOURCE_MODELS" for e in result.errors)

    def test_validate_no_fold_data_error(self):
        """Test validation error when no fold data found."""
        store = MockPredictionStore()  # Empty store
        validator = FoldAlignmentValidator(store)
        context = MockExecutionContext(step_number=5)

        result = validator.validate(["NonExistent"], context)

        assert result.is_valid is False
        assert any(e.code == "NO_FOLD_DATA" for e in result.errors)

    def test_validate_different_fold_counts_error(self):
        """Test validation error when models have different fold counts."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2, n_folds=5)
        create_mock_predictions_5fold(store, "RF", step_idx=3, n_folds=3)

        validator = FoldAlignmentValidator(store)
        context = MockExecutionContext(step_number=5)

        result = validator.validate(["PLS", "RF"], context)

        assert result.is_valid is False
        assert any(e.code == "FOLD_COUNT_MISMATCH" for e in result.errors)

    def test_validate_branch_filtering(self):
        """Test validation filters by branch_id."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2, branch_id=0)
        create_mock_predictions_5fold(store, "RF", step_idx=3, branch_id=1)

        validator = FoldAlignmentValidator(store)
        context = MockExecutionContext(step_number=5, branch_id=0)

        result = validator.validate(["PLS", "RF"], context)

        # RF should not be found in branch 0
        # This should still work but with partial data
        assert result.is_valid is True  # Not an error, just uses available data

# =============================================================================
# TrainingSetReconstructor Tests
# =============================================================================

class TestTrainingSetReconstructor:
    """Test TrainingSetReconstructor."""

    def test_reconstruct_basic(self):
        """Test basic OOF reconstruction with 5-fold CV."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2)
        create_mock_predictions_5fold(store, "RF", step_idx=3)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS", "RF"],
            stacking_config=StackingConfig(coverage_strategy=CoverageStrategy.STRICT),
        )

        dataset = MockDataset(n_train=80, n_test=20)
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # Check result structure
        assert isinstance(result, ReconstructionResult)
        assert result.X_train_meta.shape == (80, 2)
        assert result.X_test_meta.shape == (20, 2)
        assert len(result.feature_names) == 2
        assert result.source_models == ["PLS", "RF"]
        assert result.n_folds == 5
        assert result.coverage_ratio == 1.0

    def test_reconstruct_no_source_models_error(self):
        """Test that empty source models raises error."""
        store = MockPredictionStore()

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=[],
        )

        dataset = MockDataset()
        context = MockExecutionContext()

        with pytest.raises(ValueError, match="No source model names"):
            reconstructor.reconstruct(dataset, context)

    def test_reconstruct_feature_order(self):
        """Test that feature order matches source_model_names order."""
        store = MockPredictionStore()
        # Create predictions with different values
        for fold in range(5):
            val_indices = np.arange(fold * 16, (fold + 1) * 16)

            store.add_prediction(
                model_name="ModelA",
                partition='val',
                fold_id=fold,
                step_idx=2,
                sample_indices=val_indices,
                y_pred=np.ones(16) * 1.0,  # All 1s
            )
            store.add_prediction(
                model_name="ModelB",
                partition='val',
                fold_id=fold,
                step_idx=3,
                sample_indices=val_indices,
                y_pred=np.ones(16) * 2.0,  # All 2s
            )
            # Test predictions
            test_indices = np.arange(80, 100)
            store.add_prediction(
                model_name="ModelA",
                partition='test',
                fold_id=fold,
                step_idx=2,
                sample_indices=test_indices,
                y_pred=np.ones(20) * 1.0,
            )
            store.add_prediction(
                model_name="ModelB",
                partition='test',
                fold_id=fold,
                step_idx=3,
                sample_indices=test_indices,
                y_pred=np.ones(20) * 2.0,
            )

        # Specify order: ModelB, ModelA
        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["ModelB", "ModelA"],
        )

        dataset = MockDataset(n_train=80, n_test=20)
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # Column 0 should be ModelB (2.0), Column 1 should be ModelA (1.0)
        assert np.allclose(result.X_train_meta[:, 0], 2.0)
        assert np.allclose(result.X_train_meta[:, 1], 1.0)

# =============================================================================
# Coverage Strategy Tests
# =============================================================================

class TestCoverageStrategies:
    """Test coverage strategy implementations."""

    def _create_partial_predictions(self, store, missing_folds=None):
        """Create predictions with some missing folds."""
        missing_folds = missing_folds or []
        n_folds = 5
        n_train = 80
        samples_per_fold = n_train // n_folds

        for fold_id in range(n_folds):
            if fold_id in missing_folds:
                continue  # Skip this fold to create missing data

            val_start = fold_id * samples_per_fold
            val_end = val_start + samples_per_fold
            val_indices = np.arange(val_start, val_end)
            y_pred = np.ones(samples_per_fold) * fold_id

            store.add_prediction(
                model_name="PLS",
                partition='val',
                fold_id=fold_id,
                step_idx=2,
                sample_indices=val_indices,
                y_pred=y_pred,
            )

        # Add test predictions (complete)
        test_indices = np.arange(80, 100)
        for fold_id in range(n_folds):
            store.add_prediction(
                model_name="PLS",
                partition='test',
                fold_id=fold_id,
                step_idx=2,
                sample_indices=test_indices,
                y_pred=np.ones(20),
            )

    def test_coverage_strict_complete(self):
        """Test STRICT strategy with complete coverage."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(coverage_strategy=CoverageStrategy.STRICT),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)
        assert result.coverage_ratio == 1.0
        assert result.valid_train_mask.all()

    def test_coverage_strict_incomplete_raises(self):
        """Test STRICT strategy raises error with incomplete coverage."""
        store = MockPredictionStore()
        self._create_partial_predictions(store, missing_folds=[2])

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(coverage_strategy=CoverageStrategy.STRICT),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        with pytest.raises(ValueError, match="Incomplete OOF coverage"):
            reconstructor.reconstruct(dataset, context)

    def test_coverage_drop_incomplete(self):
        """Test DROP_INCOMPLETE strategy masks missing samples."""
        store = MockPredictionStore()
        self._create_partial_predictions(store, missing_folds=[2])

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(
                coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
                min_coverage_ratio=0.5,  # Allow up to 50% dropped
            ),
            reconstructor_config=ReconstructorConfig(log_warnings=False),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # 16 samples (1 fold) should be dropped
        assert result.valid_train_mask.sum() == 64
        assert (~result.valid_train_mask).sum() == 16

    def test_coverage_impute_zero(self):
        """Test IMPUTE_ZERO strategy fills missing with zeros."""
        store = MockPredictionStore()
        self._create_partial_predictions(store, missing_folds=[2])

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(coverage_strategy=CoverageStrategy.IMPUTE_ZERO),
            reconstructor_config=ReconstructorConfig(log_warnings=False),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # All samples should be valid
        assert result.valid_train_mask.all()

        # Missing fold (32-48) should be zeros
        assert np.allclose(result.X_train_meta[32:48, 0], 0.0)

    def test_coverage_impute_mean(self):
        """Test IMPUTE_MEAN strategy fills missing with column mean."""
        store = MockPredictionStore()
        self._create_partial_predictions(store, missing_folds=[2])

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(coverage_strategy=CoverageStrategy.IMPUTE_MEAN),
            reconstructor_config=ReconstructorConfig(log_warnings=False),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # All samples should be valid
        assert result.valid_train_mask.all()

        # Missing samples should have mean of available values
        # Available folds: 0, 1, 3, 4 with values 0, 1, 3, 4
        # Mean should be (0 + 1 + 3 + 4) / 4 * 16 samples each... actually per-fold value
        # Each fold has 16 samples with value = fold_id
        # Mean = (16*0 + 16*1 + 16*3 + 16*4) / 64 = 128/64 = 2.0
        expected_mean = 2.0
        assert np.allclose(result.X_train_meta[32:48, 0], expected_mean)

# =============================================================================
# Test Aggregation Strategy Tests
# =============================================================================

class TestTestAggregationStrategies:
    """Test test prediction aggregation strategies."""

    def _create_predictions_with_varied_scores(self, store):
        """Create predictions with different validation scores."""
        n_train = 80
        n_test = 20
        n_folds = 5
        samples_per_fold = n_train // n_folds

        val_scores = [0.9, 0.8, 0.95, 0.7, 0.85]  # Fold 2 is best

        for fold_id in range(n_folds):
            val_start = fold_id * samples_per_fold
            val_end = val_start + samples_per_fold
            val_indices = np.arange(val_start, val_end)

            store.add_prediction(
                model_name="PLS",
                partition='val',
                fold_id=fold_id,
                step_idx=2,
                sample_indices=val_indices,
                y_pred=np.ones(samples_per_fold),
                val_score=val_scores[fold_id],
            )

            # Test predictions vary by fold
            test_indices = np.arange(n_train, n_train + n_test)
            test_pred = np.ones(n_test) * (fold_id + 1)  # 1, 2, 3, 4, 5

            store.add_prediction(
                model_name="PLS",
                partition='test',
                fold_id=fold_id,
                step_idx=2,
                sample_indices=test_indices,
                y_pred=test_pred,
                val_score=val_scores[fold_id],
            )

        return val_scores

    def test_test_aggregation_mean(self):
        """Test MEAN aggregation averages test predictions."""
        store = MockPredictionStore()
        self._create_predictions_with_varied_scores(store)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(test_aggregation=TestAggregation.MEAN),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # Mean of 1, 2, 3, 4, 5 = 3.0
        expected_mean = 3.0
        assert np.allclose(result.X_test_meta[:, 0], expected_mean)

    def test_test_aggregation_best_fold(self):
        """Test BEST_FOLD uses prediction from best-scoring fold."""
        store = MockPredictionStore()
        val_scores = self._create_predictions_with_varied_scores(store)

        # Best fold is fold 2 (score 0.95)
        best_fold = np.argmax(val_scores)
        assert best_fold == 2

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(test_aggregation=TestAggregation.BEST_FOLD),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # Best fold is 2, predictions are fold_id + 1 = 3
        expected_value = 3.0
        assert np.allclose(result.X_test_meta[:, 0], expected_value)

    def test_test_aggregation_weighted_mean(self):
        """Test WEIGHTED_MEAN weights by validation scores."""
        store = MockPredictionStore()
        val_scores = self._create_predictions_with_varied_scores(store)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(test_aggregation=TestAggregation.WEIGHTED_MEAN),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        # Weighted mean: sum(score * pred) / sum(score)
        # Scores: [0.9, 0.8, 0.95, 0.7, 0.85], Preds: [1, 2, 3, 4, 5]
        scores = np.array(val_scores)
        preds = np.array([1, 2, 3, 4, 5])
        expected = np.average(preds, weights=scores)

        assert np.allclose(result.X_test_meta[:, 0], expected)

# =============================================================================
# Branch-Aware Reconstruction Tests
# =============================================================================

class TestBranchAwareReconstruction:
    """Test branch-aware reconstruction features."""

    def test_branch_filtering(self):
        """Test that reconstruction filters by branch_id."""
        store = MockPredictionStore()

        # Create predictions for branch 0
        create_mock_predictions_5fold(store, "PLS", step_idx=2, branch_id=0)

        # Create predictions for branch 1 with different values
        for fold in range(5):
            val_indices = np.arange(fold * 16, (fold + 1) * 16)
            store.add_prediction(
                model_name="PLS",
                partition='val',
                fold_id=fold,
                step_idx=2,
                sample_indices=val_indices,
                y_pred=np.ones(16) * 999.0,  # Different value
                branch_id=1,
            )
            test_indices = np.arange(80, 100)
            store.add_prediction(
                model_name="PLS",
                partition='test',
                fold_id=fold,
                step_idx=2,
                sample_indices=test_indices,
                y_pred=np.ones(20) * 999.0,
                branch_id=1,
            )

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5, branch_id=0)

        result = reconstructor.reconstruct(dataset, context)

        # Should use branch 0 predictions, not 999.0 values
        assert not np.any(result.X_train_meta == 999.0)
        assert not np.any(result.X_test_meta == 999.0)

    def test_validate_branch_compatibility(self):
        """Test branch compatibility validation."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2, branch_id=0)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
        )

        context = MockExecutionContext(step_number=5, branch_id=0, branch_name="snv")

        result = reconstructor.validate_branch_compatibility(context)

        # Should be valid since we have predictions in branch 0
        assert result.is_valid is True

    def test_validate_branch_compatibility_no_predictions_warning(self):
        """Test warning when no predictions in current branch."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2, branch_id=1)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
        )

        context = MockExecutionContext(step_number=5, branch_id=0, branch_name="snv")

        result = reconstructor.validate_branch_compatibility(context)

        # Should have a warning about no predictions in branch
        assert any(w.code == "NO_BRANCH_PREDICTIONS" for w in result.warnings)

# =============================================================================
# Feature Name Generation Tests
# =============================================================================

class TestFeatureNameGeneration:
    """Test feature name generation."""

    def test_default_feature_names(self):
        """Test default feature name pattern."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2)
        create_mock_predictions_5fold(store, "RF", step_idx=3)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS", "RF"],
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        assert result.feature_names == ["PLS_pred", "RF_pred"]

    def test_custom_feature_name_pattern(self):
        """Test custom feature name pattern."""
        store = MockPredictionStore()
        create_mock_predictions_5fold(store, "PLS", step_idx=2)

        reconstructor = TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            reconstructor_config=ReconstructorConfig(
                feature_name_pattern="{model_name}_feature"
            ),
        )

        dataset = MockDataset()
        context = MockExecutionContext(step_number=5)

        result = reconstructor.reconstruct(dataset, context)

        assert result.feature_names == ["PLS_feature"]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

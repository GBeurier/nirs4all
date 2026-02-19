"""
Unit tests for Phase 5: Classification Support in Meta-Model Stacking.

Tests cover:
- StackingTaskType enum and ClassificationInfo dataclass
- TaskTypeDetector task type detection from predictions
- ClassificationFeatureExtractor binary and multiclass probability extraction
- FeatureNameGenerator feature naming with class indices
- MetaFeatureInfo feature importance tracking
- Integration with TrainingSetReconstructor
"""

from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from nirs4all.controllers.models.stacking.classification import (
    ClassificationFeatureExtractor,
    ClassificationInfo,
    FeatureNameGenerator,
    MetaFeatureInfo,
    StackingTaskType,
    TaskTypeDetector,
    build_meta_feature_info,
)


class TestStackingTaskType:
    """Test StackingTaskType enum."""

    def test_regression_is_not_classification(self):
        """Regression task type should not be classification."""
        assert StackingTaskType.REGRESSION.is_classification is False

    def test_binary_classification_is_classification(self):
        """Binary classification should be classification."""
        assert StackingTaskType.BINARY_CLASSIFICATION.is_classification is True

    def test_multiclass_is_classification(self):
        """Multiclass classification should be classification."""
        assert StackingTaskType.MULTICLASS_CLASSIFICATION.is_classification is True

    def test_unknown_is_not_classification(self):
        """Unknown task type should not be classification."""
        assert StackingTaskType.UNKNOWN.is_classification is False

    def test_binary_n_classes(self):
        """Binary classification should have n_classes = 2."""
        assert StackingTaskType.BINARY_CLASSIFICATION.n_classes == 2

    def test_multiclass_n_classes_none(self):
        """Multiclass n_classes should be None (variable)."""
        assert StackingTaskType.MULTICLASS_CLASSIFICATION.n_classes is None

    def test_regression_n_classes_none(self):
        """Regression n_classes should be None."""
        assert StackingTaskType.REGRESSION.n_classes is None

class TestClassificationInfo:
    """Test ClassificationInfo dataclass."""

    def test_is_classification_true_for_binary(self):
        """is_classification property for binary classification."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        assert info.is_classification is True
        assert info.is_binary is True
        assert info.is_multiclass is False

    def test_is_classification_true_for_multiclass(self):
        """is_classification property for multiclass."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=5,
            has_probabilities=True
        )
        assert info.is_classification is True
        assert info.is_binary is False
        assert info.is_multiclass is True

    def test_is_classification_false_for_regression(self):
        """is_classification property for regression."""
        info = ClassificationInfo(
            task_type=StackingTaskType.REGRESSION,
            n_classes=None,
            has_probabilities=False
        )
        assert info.is_classification is False
        assert info.is_binary is False
        assert info.is_multiclass is False

    def test_n_features_per_model_regression(self):
        """Regression should have 1 feature per model."""
        info = ClassificationInfo(
            task_type=StackingTaskType.REGRESSION,
            n_classes=None,
            has_probabilities=False
        )
        assert info.get_n_features_per_model(use_proba=False) == 1
        assert info.get_n_features_per_model(use_proba=True) == 1

    def test_n_features_per_model_binary_no_proba(self):
        """Binary without proba should have 1 feature."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        assert info.get_n_features_per_model(use_proba=False) == 1

    def test_n_features_per_model_binary_with_proba(self):
        """Binary with proba should have 1 feature (positive class only)."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        assert info.get_n_features_per_model(use_proba=True) == 1

    def test_n_features_per_model_multiclass_with_proba(self):
        """Multiclass with proba should have n_classes features."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=5,
            has_probabilities=True
        )
        assert info.get_n_features_per_model(use_proba=True) == 5

    def test_n_features_per_model_multiclass_no_proba(self):
        """Multiclass without proba should have 1 feature (y_pred)."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=5,
            has_probabilities=True
        )
        assert info.get_n_features_per_model(use_proba=False) == 1

class TestTaskTypeDetector:
    """Test TaskTypeDetector class."""

    def _create_mock_prediction_store(
        self,
        predictions: list[dict[str, Any]]
    ) -> Mock:
        """Create a mock prediction store with given predictions."""
        store = Mock()
        store.filter_predictions = Mock(return_value=predictions)
        return store

    def _create_mock_context(self, step_number: int = 5) -> Mock:
        """Create a mock execution context."""
        context = Mock()
        context.selector = Mock()
        context.selector.branch_id = None
        context.state = Mock()
        context.state.step_number = step_number
        return context

    def test_detect_regression_from_task_type(self):
        """Detect regression task type from prediction metadata."""
        predictions = [{
            'model_name': 'PLSRegression',
            'step_idx': 2,
            'fold_id': 0,
            'task_type': 'regression',
            'y_proba': None,
        }]
        store = self._create_mock_prediction_store(predictions)
        context = self._create_mock_context()

        detector = TaskTypeDetector(store)
        info = detector.detect(['PLSRegression'], context)

        assert info.task_type == StackingTaskType.REGRESSION
        assert info.is_classification is False

    def test_detect_binary_classification_from_task_type(self):
        """Detect binary classification from prediction metadata."""
        predictions = [{
            'model_name': 'LogisticRegression',
            'step_idx': 2,
            'fold_id': 0,
            'task_type': 'binary_classification',
            'y_proba': np.array([[0.3, 0.7], [0.6, 0.4]]),
        }]
        store = self._create_mock_prediction_store(predictions)
        context = self._create_mock_context()

        detector = TaskTypeDetector(store)
        info = detector.detect(['LogisticRegression'], context)

        assert info.task_type == StackingTaskType.BINARY_CLASSIFICATION
        assert info.n_classes == 2
        assert info.has_probabilities is True

    def test_detect_multiclass_classification(self):
        """Detect multiclass classification from prediction metadata."""
        predictions = [{
            'model_name': 'RandomForestClassifier',
            'step_idx': 2,
            'fold_id': 0,
            'task_type': 'multiclass_classification',
            'y_proba': np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]),
        }]
        store = self._create_mock_prediction_store(predictions)
        context = self._create_mock_context()

        detector = TaskTypeDetector(store)
        info = detector.detect(['RandomForestClassifier'], context)

        assert info.task_type == StackingTaskType.MULTICLASS_CLASSIFICATION
        assert info.n_classes == 3
        assert info.has_probabilities is True

    def test_detect_unknown_when_no_predictions(self):
        """Return unknown when no predictions found."""
        store = self._create_mock_prediction_store([])
        context = self._create_mock_context()

        detector = TaskTypeDetector(store)
        info = detector.detect(['NonexistentModel'], context)

        assert info.task_type == StackingTaskType.UNKNOWN

class TestClassificationFeatureExtractor:
    """Test ClassificationFeatureExtractor class."""

    def test_extract_regression_features(self):
        """Extract y_pred for regression task."""
        info = ClassificationInfo(
            task_type=StackingTaskType.REGRESSION,
            n_classes=None,
            has_probabilities=False
        )
        extractor = ClassificationFeatureExtractor(info, use_proba=False)

        pred = {
            'y_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'y_proba': None
        }
        features = extractor.extract_features(pred, n_samples=5)

        np.testing.assert_array_equal(features, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert features.shape == (5,)

    def test_extract_binary_proba_2d(self):
        """Extract positive class probability from 2D y_proba."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        extractor = ClassificationFeatureExtractor(info, use_proba=True)

        pred = {
            'y_pred': np.array([0, 1, 1, 0]),
            'y_proba': np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])
        }
        features = extractor.extract_features(pred, n_samples=4)

        # Should extract column 1 (positive class)
        np.testing.assert_array_almost_equal(features, [0.2, 0.7, 0.6, 0.1])
        assert features.shape == (4,)

    def test_extract_binary_proba_1d(self):
        """Extract probability from 1D y_proba (already positive class)."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        extractor = ClassificationFeatureExtractor(info, use_proba=True)

        pred = {
            'y_pred': np.array([0, 1, 1, 0]),
            'y_proba': np.array([0.2, 0.7, 0.6, 0.1])  # Already 1D
        }
        features = extractor.extract_features(pred, n_samples=4)

        np.testing.assert_array_almost_equal(features, [0.2, 0.7, 0.6, 0.1])
        assert features.shape == (4,)

    def test_extract_multiclass_proba(self):
        """Extract all class probabilities for multiclass."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=3,
            has_probabilities=True
        )
        extractor = ClassificationFeatureExtractor(info, use_proba=True)

        pred = {
            'y_pred': np.array([0, 1, 2]),
            'y_proba': np.array([
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6]
            ])
        }
        features = extractor.extract_features(pred, n_samples=3)

        expected = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ])
        np.testing.assert_array_almost_equal(features, expected)
        assert features.shape == (3, 3)

    def test_fallback_to_y_pred_when_no_proba(self):
        """Fall back to y_pred when y_proba is not available."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=False
        )
        extractor = ClassificationFeatureExtractor(info, use_proba=True)

        pred = {
            'model_name': 'TestModel',
            'y_pred': np.array([0, 1, 1, 0]),
            'y_proba': None
        }

        with pytest.warns(UserWarning, match="use_proba=True but no y_proba available"):
            features = extractor.extract_features(pred, n_samples=4)

        np.testing.assert_array_equal(features, [0, 1, 1, 0])

    def test_get_n_features(self):
        """Test get_n_features method."""
        info_binary = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        extractor_binary = ClassificationFeatureExtractor(info_binary, use_proba=True)
        assert extractor_binary.get_n_features() == 1

        info_multi = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=5,
            has_probabilities=True
        )
        extractor_multi = ClassificationFeatureExtractor(info_multi, use_proba=True)
        assert extractor_multi.get_n_features() == 5

class TestFeatureNameGenerator:
    """Test FeatureNameGenerator class."""

    def test_generate_regression_names(self):
        """Generate simple names for regression."""
        info = ClassificationInfo(
            task_type=StackingTaskType.REGRESSION,
            n_classes=None,
            has_probabilities=False
        )
        generator = FeatureNameGenerator(info, use_proba=False)

        names = generator.generate_names(['PLSRegression', 'RandomForest'])

        assert names == ['PLSRegression_pred', 'RandomForest_pred']

    def test_generate_binary_names_no_proba(self):
        """Generate names for binary classification without proba."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        generator = FeatureNameGenerator(info, use_proba=False)

        names = generator.generate_names(['LogisticRegression', 'RandomForest'])

        assert names == ['LogisticRegression_pred', 'RandomForest_pred']

    def test_generate_binary_names_with_proba(self):
        """Generate names for binary classification with proba."""
        info = ClassificationInfo(
            task_type=StackingTaskType.BINARY_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        generator = FeatureNameGenerator(info, use_proba=True)

        names = generator.generate_names(['LogisticRegression', 'RandomForest'])

        # Should include _proba_1 suffix for positive class
        assert names == ['LogisticRegression_proba_1', 'RandomForest_proba_1']

    def test_generate_multiclass_names_with_proba(self):
        """Generate names for multiclass with proba (one per class)."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=3,
            has_probabilities=True
        )
        generator = FeatureNameGenerator(info, use_proba=True)

        names = generator.generate_names(['RandomForest'])

        expected = [
            'RandomForest_proba_0',
            'RandomForest_proba_1',
            'RandomForest_proba_2'
        ]
        assert names == expected

    def test_generate_multiclass_names_multiple_models(self):
        """Generate names for multiclass with multiple models."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=2,
            has_probabilities=True
        )
        generator = FeatureNameGenerator(info, use_proba=True)

        names = generator.generate_names(['ModelA', 'ModelB'])

        expected = [
            'ModelA_proba_0', 'ModelA_proba_1',
            'ModelB_proba_0', 'ModelB_proba_1'
        ]
        assert names == expected

    def test_get_feature_importance_mapping(self):
        """Test feature importance mapping."""
        info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=3,
            has_probabilities=True
        )
        generator = FeatureNameGenerator(info, use_proba=True)

        mapping = generator.get_feature_importance_mapping(['ModelA', 'ModelB'])

        assert 'ModelA' in mapping
        assert 'ModelB' in mapping
        assert len(mapping['ModelA']) == 3
        assert len(mapping['ModelB']) == 3

class TestMetaFeatureInfo:
    """Test MetaFeatureInfo dataclass."""

    def test_get_model_for_feature(self):
        """Test getting source model for a feature."""
        info = MetaFeatureInfo(
            feature_names=['ModelA_pred', 'ModelB_pred'],
            source_models=['ModelA', 'ModelB'],
            feature_to_model={'ModelA_pred': 'ModelA', 'ModelB_pred': 'ModelB'},
            classification_info=ClassificationInfo(
                task_type=StackingTaskType.REGRESSION
            )
        )

        assert info.get_model_for_feature('ModelA_pred') == 'ModelA'
        assert info.get_model_for_feature('ModelB_pred') == 'ModelB'
        assert info.get_model_for_feature('Unknown') is None

    def test_aggregate_importance_by_model(self):
        """Test aggregating feature importances by source model."""
        info = MetaFeatureInfo(
            feature_names=['ModelA_proba_0', 'ModelA_proba_1', 'ModelB_proba_0', 'ModelB_proba_1'],
            source_models=['ModelA', 'ModelB'],
            feature_to_model={
                'ModelA_proba_0': 'ModelA',
                'ModelA_proba_1': 'ModelA',
                'ModelB_proba_0': 'ModelB',
                'ModelB_proba_1': 'ModelB',
            },
            classification_info=ClassificationInfo(
                task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
                n_classes=2
            ),
            n_features_per_model={'ModelA': 2, 'ModelB': 2}
        )

        feature_importances = {
            'ModelA_proba_0': 0.3,
            'ModelA_proba_1': 0.2,
            'ModelB_proba_0': 0.35,
            'ModelB_proba_1': 0.15,
        }

        model_importance = info.aggregate_importance_by_model(feature_importances)

        assert model_importance['ModelA'] == pytest.approx(0.5)
        assert model_importance['ModelB'] == pytest.approx(0.5)

class TestBuildMetaFeatureInfo:
    """Test build_meta_feature_info helper function."""

    def test_build_for_regression(self):
        """Build meta feature info for regression task."""
        classification_info = ClassificationInfo(
            task_type=StackingTaskType.REGRESSION,
            n_classes=None,
            has_probabilities=False
        )

        info = build_meta_feature_info(
            source_model_names=['PLS', 'RF'],
            classification_info=classification_info,
            use_proba=False
        )

        assert info.feature_names == ['PLS_pred', 'RF_pred']
        assert info.source_models == ['PLS', 'RF']
        assert info.get_model_for_feature('PLS_pred') == 'PLS'
        assert info.get_model_for_feature('RF_pred') == 'RF'

    def test_build_for_multiclass_with_proba(self):
        """Build meta feature info for multiclass with probabilities."""
        classification_info = ClassificationInfo(
            task_type=StackingTaskType.MULTICLASS_CLASSIFICATION,
            n_classes=3,
            has_probabilities=True
        )

        info = build_meta_feature_info(
            source_model_names=['RF', 'LR'],
            classification_info=classification_info,
            use_proba=True
        )

        expected_names = [
            'RF_proba_0', 'RF_proba_1', 'RF_proba_2',
            'LR_proba_0', 'LR_proba_1', 'LR_proba_2'
        ]
        assert info.feature_names == expected_names
        assert info.n_features_per_model['RF'] == 3
        assert info.n_features_per_model['LR'] == 3

        # All RF features should map to RF
        assert info.get_model_for_feature('RF_proba_0') == 'RF'
        assert info.get_model_for_feature('RF_proba_1') == 'RF'
        assert info.get_model_for_feature('RF_proba_2') == 'RF'

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

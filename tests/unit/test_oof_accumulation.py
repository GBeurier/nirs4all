"""
Unit tests for OOF prediction accumulation/averaging fix.

Verifies that when samples appear in multiple validation folds (e.g.,
with RepeatedKFold), the final OOF predictions are the AVERAGE across
all appearances, not the last overwritten value.

Tests cover:
- 1D regression case via _collect_oof_predictions_with_proba
- 2D classification probability case via _collect_oof_predictions_with_proba
- Standard KFold (no regression: each sample appears once)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from nirs4all.controllers.models.stacking import (
    TrainingSetReconstructor,
    ReconstructorConfig,
)
from nirs4all.operators.models.meta import StackingConfig


class MockPredictionStore:
    """Mock Predictions store for testing OOF accumulation."""

    def __init__(self):
        self._predictions = []

    def add_prediction(self, **kwargs):
        self._predictions.append(kwargs)

    def filter_predictions(
        self,
        model_name=None,
        partition=None,
        fold_id=None,
        step_idx=None,
        branch_id=None,
        load_arrays=True,
        **kwargs
    ):
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
            results.append(pred)
        return results


# =============================================================================
# Tests for _collect_oof_predictions_with_proba (1D and 2D)
# =============================================================================


class TestCollectOofPredictionsWithProba:
    """Test OOF accumulation in _collect_oof_predictions_with_proba."""

    def _make_reconstructor(self, store):
        return TrainingSetReconstructor(
            prediction_store=store,
            source_model_names=["PLS"],
            stacking_config=StackingConfig(),
            reconstructor_config=ReconstructorConfig(),
        )

    def _make_classification_info(self, task_type="regression", n_classes=None, has_probabilities=False):
        from nirs4all.controllers.models.stacking.classification import ClassificationInfo, StackingTaskType
        type_map = {
            "regression": StackingTaskType.REGRESSION,
            "binary": StackingTaskType.BINARY_CLASSIFICATION,
            "multiclass": StackingTaskType.MULTICLASS_CLASSIFICATION,
        }
        return ClassificationInfo(
            task_type=type_map[task_type],
            n_classes=n_classes,
            has_probabilities=has_probabilities,
        )

    def test_1d_regression_averages_predictions(self):
        """1D regression: samples in multiple folds get averaged."""
        store = MockPredictionStore()

        # Sample 0 appears in folds 0 and 1 with predictions 2.0 and 4.0 -> avg 3.0
        # Sample 1 appears in folds 0 and 1 with predictions 6.0 and 8.0 -> avg 7.0
        store.add_prediction(
            model_name="PLS", partition="val", fold_id=0, step_idx=1,
            sample_indices=np.array([0, 1]), y_pred=np.array([2.0, 6.0]),
        )
        store.add_prediction(
            model_name="PLS", partition="val", fold_id=1, step_idx=1,
            sample_indices=np.array([0, 1]), y_pred=np.array([4.0, 8.0]),
        )

        reconstructor = self._make_reconstructor(store)
        id_to_pos = {0: 0, 1: 1}
        classification_info = self._make_classification_info("regression")

        oof_preds, n_folds = reconstructor._collect_oof_predictions_with_proba(
            model_name="PLS",
            branch_id=None,
            max_step=10,
            id_to_pos=id_to_pos,
            n_samples=2,
            use_proba=False,
            classification_info=classification_info,
        )

        assert n_folds == 2
        assert oof_preds.ndim == 1
        np.testing.assert_allclose(oof_preds[0], 3.0)
        np.testing.assert_allclose(oof_preds[1], 7.0)

    def test_2d_multiclass_averages_predictions(self):
        """2D multiclass probabilities: samples in multiple folds get averaged per class."""
        store = MockPredictionStore()

        n_classes = 3

        # Sample 0 in fold 0: proba [0.1, 0.3, 0.6]
        # Sample 0 in fold 1: proba [0.3, 0.5, 0.2]
        # Expected avg:       [0.2, 0.4, 0.4]
        #
        # Sample 1 in fold 0: proba [0.8, 0.1, 0.1]
        # Sample 1 in fold 1: proba [0.6, 0.3, 0.1]
        # Expected avg:       [0.7, 0.2, 0.1]

        store.add_prediction(
            model_name="PLS", partition="val", fold_id=0, step_idx=1,
            sample_indices=np.array([0, 1]),
            y_pred=np.array([0, 0]),
            y_proba=np.array([[0.1, 0.3, 0.6], [0.8, 0.1, 0.1]]),
        )
        store.add_prediction(
            model_name="PLS", partition="val", fold_id=1, step_idx=1,
            sample_indices=np.array([0, 1]),
            y_pred=np.array([2, 0]),
            y_proba=np.array([[0.3, 0.5, 0.2], [0.6, 0.3, 0.1]]),
        )

        reconstructor = self._make_reconstructor(store)
        id_to_pos = {0: 0, 1: 1}
        classification_info = self._make_classification_info(
            "multiclass", n_classes=n_classes, has_probabilities=True
        )

        oof_preds, n_folds = reconstructor._collect_oof_predictions_with_proba(
            model_name="PLS",
            branch_id=None,
            max_step=10,
            id_to_pos=id_to_pos,
            n_samples=2,
            use_proba=True,
            classification_info=classification_info,
        )

        assert n_folds == 2
        assert oof_preds.ndim == 2
        assert oof_preds.shape == (2, n_classes)
        np.testing.assert_allclose(oof_preds[0], [0.2, 0.4, 0.4])
        np.testing.assert_allclose(oof_preds[1], [0.7, 0.2, 0.1])

    def test_standard_kfold_2d_no_regression(self):
        """Standard KFold with 2D probabilities: each sample once, no averaging needed."""
        store = MockPredictionStore()

        n_classes = 2

        # 4 samples, 2 folds, each sample exactly once
        store.add_prediction(
            model_name="PLS", partition="val", fold_id=0, step_idx=1,
            sample_indices=np.array([0, 1]),
            y_pred=np.array([1, 0]),
            y_proba=np.array([[0.2, 0.8], [0.9, 0.1]]),
        )
        store.add_prediction(
            model_name="PLS", partition="val", fold_id=1, step_idx=1,
            sample_indices=np.array([2, 3]),
            y_pred=np.array([0, 1]),
            y_proba=np.array([[0.7, 0.3], [0.4, 0.6]]),
        )

        reconstructor = self._make_reconstructor(store)
        id_to_pos = {0: 0, 1: 1, 2: 2, 3: 3}
        classification_info = self._make_classification_info(
            "binary", n_classes=n_classes, has_probabilities=True
        )

        oof_preds, n_folds = reconstructor._collect_oof_predictions_with_proba(
            model_name="PLS",
            branch_id=None,
            max_step=10,
            id_to_pos=id_to_pos,
            n_samples=4,
            use_proba=True,
            classification_info=classification_info,
        )

        assert n_folds == 2
        # Binary with use_proba=True returns 1D (positive class probability)
        # because ClassificationInfo.get_n_features_per_model returns 1 for binary
        assert oof_preds.ndim == 1
        # For binary, the extractor returns positive class probability (column 1)
        np.testing.assert_allclose(oof_preds[0], 0.8)
        np.testing.assert_allclose(oof_preds[1], 0.1)
        np.testing.assert_allclose(oof_preds[2], 0.3)
        np.testing.assert_allclose(oof_preds[3], 0.6)

    def test_2d_missing_samples_stay_nan(self):
        """2D case: samples not in any fold remain NaN."""
        store = MockPredictionStore()

        n_classes = 3

        store.add_prediction(
            model_name="PLS", partition="val", fold_id=0, step_idx=1,
            sample_indices=np.array([0]),
            y_pred=np.array([1]),
            y_proba=np.array([[0.2, 0.5, 0.3]]),
        )

        reconstructor = self._make_reconstructor(store)
        id_to_pos = {0: 0, 1: 1}
        classification_info = self._make_classification_info(
            "multiclass", n_classes=n_classes, has_probabilities=True
        )

        oof_preds, _ = reconstructor._collect_oof_predictions_with_proba(
            model_name="PLS",
            branch_id=None,
            max_step=10,
            id_to_pos=id_to_pos,
            n_samples=2,
            use_proba=True,
            classification_info=classification_info,
        )

        assert oof_preds.shape == (2, n_classes)
        np.testing.assert_allclose(oof_preds[0], [0.2, 0.5, 0.3])
        assert np.all(np.isnan(oof_preds[1]))


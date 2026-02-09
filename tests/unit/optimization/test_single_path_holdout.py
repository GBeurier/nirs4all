"""Unit tests for BUG-3 regression: single-path must use holdout split.

Verifies that when no folds are available, optimization does NOT use
X_train == X_val (which causes overfitting).
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from nirs4all.optimization.optuna import OptunaManager


class TestSinglePathHoldout:
    """Tests that single-path optimization uses a proper holdout split."""

    def test_single_path_uses_holdout_split(self):
        """BUG-3 regression: X_val must differ from X_train in single-path."""
        manager = OptunaManager()

        # Create mock dataset and controller
        dataset = MagicMock()
        dataset.task_type = "regression"
        context = MagicMock()
        controller = MagicMock()
        controller._get_model_instance.return_value = MagicMock()
        controller._prepare_data.side_effect = lambda X, y, ctx: (X, y)
        controller._train_model.return_value = MagicMock()
        controller._evaluate_model.return_value = 0.5
        controller.process_hyperparameters = MagicMock(side_effect=lambda x: x)

        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)

        finetune_params = {
            "n_trials": 1,
            "verbose": 0,
            "approach": "single",
            "model_params": {"alpha": [0.1, 1.0]},
        }

        # Patch _optimize_single to capture the args it receives
        original_optimize_single = manager._optimize_single
        captured_args = {}

        def capture_optimize_single(ds, mc, X_tr, y_tr, X_v, y_v, *args, **kwargs):
            captured_args["X_train_shape"] = X_tr.shape
            captured_args["X_val_shape"] = X_v.shape
            captured_args["X_train_ptr"] = id(X_tr)
            captured_args["X_val_ptr"] = id(X_v)
            return {"alpha": 0.1}

        with patch.object(manager, '_optimize_single', side_effect=capture_optimize_single):
            manager.finetune(
                dataset=dataset,
                model_config={},
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                folds=None,
                finetune_params=finetune_params,
                context=context,
                controller=controller,
            )

        # X_train passed to _optimize_single should be smaller (80% of original)
        assert captured_args["X_train_shape"][0] == 80
        assert captured_args["X_val_shape"][0] == 20
        # They must be different arrays
        assert captured_args["X_train_ptr"] != captured_args["X_val_ptr"]

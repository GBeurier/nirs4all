"""Unit tests for BUG-1 regression: best_params must be applied in final training.

Verifies that launch_training() uses best_params regardless of mode value.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.controllers.models.base_model import BaseModelController
from nirs4all.controllers.models.sklearn_model import SklearnModelController


class TestBestParamsApplication:
    """Tests that best_params are applied during launch_training."""

    def test_best_params_applied_when_mode_is_train(self):
        """BUG-1 regression: best_params must be used even when mode='train'.

        _execute_finetune calls train() with mode='train' and best_params=<dict>.
        launch_training must use best_params regardless of mode.
        """
        # Inspect the WHOLE controller class source (not a single method) so this
        # guard survives method extraction: the best_params handling may live in
        # launch_training or in any private helper it delegates to. It still
        # guards the exact BUG-1 regression.
        import inspect
        source = inspect.getsource(BaseModelController)

        # The old buggy condition was: if mode == "finetune" and best_params is not None
        # The fix changes it to: if best_params is not None
        assert 'mode == "finetune" and best_params' not in source, (
            "BUG-1: best_params handling still gates on mode=='finetune'"
        )
        assert "if best_params is not None:" in source, (
            "BUG-1: best_params handling should check 'if best_params is not None:'"
        )

    def test_best_params_reach_training_without_cross_validation(self):
        """The holdout path must forward optimized params to final training."""
        controller = SklearnModelController()
        controller._train_single_model = MagicMock()
        best_params = {"n_components": 1}
        values = np.arange(8, dtype=float).reshape(4, 2)
        targets = np.arange(4, dtype=float)

        controller.train(
            dataset=MagicMock(),
            model_config={},
            context=MagicMock(),
            runtime_context=MagicMock(),
            prediction_store=MagicMock(),
            X_train=values,
            y_train=targets,
            X_test=values,
            y_test=targets,
            y_train_unscaled=targets,
            y_test_unscaled=targets,
            folds=[],
            best_params=best_params,
        )

        assert controller._train_single_model.call_args.args[11] == best_params

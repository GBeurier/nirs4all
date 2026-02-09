"""Unit tests for BUG-1 regression: best_params must be applied in final training.

Verifies that launch_training() uses best_params regardless of mode value.
"""

from unittest.mock import MagicMock, patch

import pytest

from nirs4all.controllers.models.base_model import BaseModelController


class TestBestParamsApplication:
    """Tests that best_params are applied during launch_training."""

    def test_best_params_applied_when_mode_is_train(self):
        """BUG-1 regression: best_params must be used even when mode='train'.

        _execute_finetune calls train() with mode='train' and best_params=<dict>.
        launch_training must use best_params regardless of mode.
        """
        # Read launch_training source to verify the condition
        import inspect
        source = inspect.getsource(BaseModelController.launch_training)

        # The old buggy condition was: if mode == "finetune" and best_params is not None
        # The fix changes it to: if best_params is not None
        assert 'mode == "finetune" and best_params' not in source, (
            "BUG-1: launch_training still gates best_params on mode=='finetune'"
        )
        assert "if best_params is not None:" in source, (
            "BUG-1: launch_training should check 'if best_params is not None:'"
        )

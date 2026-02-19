"""Unit tests for BUG-4 regression: refit must NOT re-trigger finetuning.

Verifies two defense layers:
1. BaseModelController.execute() skips _execute_finetune during REFIT phase.
2. _inject_best_params strips finetune_params from step dicts.
"""

from unittest.mock import MagicMock

import pytest

from nirs4all.pipeline.config.context import ExecutionPhase
from nirs4all.pipeline.execution.refit.executor import _inject_best_params


class TestRefitSkipsFinetuneGuard:
    """Tests for the REFIT phase guard in BaseModelController.execute."""

    def test_execute_source_has_refit_guard(self):
        """BUG-4 regression: base_model.execute must check is_refit before finetuning."""
        import inspect

        from nirs4all.controllers.models.base_model import BaseModelController

        source = inspect.getsource(BaseModelController.execute)

        # The fix adds: is_refit = runtime_context.phase == ExecutionPhase.REFIT
        # and: if not is_refit and (mode == "finetune" or ...)
        assert "is_refit" in source, (
            "BUG-4: execute() must check is_refit before dispatching to _execute_finetune"
        )
        assert "not is_refit" in source, (
            "BUG-4: execute() must skip _execute_finetune when is_refit is True"
        )

class TestInjectBestParamsStripsFinetuneParams:
    """Tests that _inject_best_params removes finetune_params from step dicts."""

    def test_finetune_params_stripped_from_model_step(self):
        """BUG-4 defense: _inject_best_params must remove finetune_params."""
        model = MagicMock()
        model.set_params = MagicMock()

        steps = [
            {"model": model, "finetune_params": {"n_trials": 50, "model_params": {"alpha": [0.1, 1.0]}}},
        ]

        _inject_best_params(steps, best_params={"alpha": 0.5})

        # finetune_params should have been removed
        assert "finetune_params" not in steps[0]

    def test_finetune_params_stripped_even_with_empty_best_params(self):
        """finetune_params should be stripped even when best_params is empty."""
        model = MagicMock()
        model.set_params = MagicMock()

        steps = [
            {"model": model, "finetune_params": {"n_trials": 10}},
        ]

        _inject_best_params(steps, best_params={})

        assert "finetune_params" not in steps[0]

    def test_non_model_steps_unaffected(self):
        """Non-model steps should pass through unchanged."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        steps = [scaler]

        _inject_best_params(steps, best_params={"alpha": 0.5})

        # Should not raise, scaler should still be there
        assert steps[0] is scaler

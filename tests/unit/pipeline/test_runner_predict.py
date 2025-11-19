import os
import random
from unittest.mock import MagicMock

import numpy as np
import pytest

from nirs4all.pipeline.runner import PipelineRunner, init_global_random_state
from nirs4all.pipeline.config.context import ExecutionContext


def test_init_global_random_state_controls_entropy():
    """Test that init_global_random_state properly seeds random generators."""
    # Clear any prior hash seed to ensure the function sets it.
    os.environ.pop("PYTHONHASHSEED", None)

    init_global_random_state(123)
    np_val1 = np.random.rand()
    py_val1 = random.random()

    init_global_random_state(123)
    np_val2 = np.random.rand()
    py_val2 = random.random()

    assert np_val1 == np_val2
    assert py_val1 == py_val2
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_run_steps_with_sequential_execution(monkeypatch):
    """Test that run_steps processes multiple steps sequentially."""
    runner = PipelineRunner(save_files=False, enable_tab_reports=False)
    dataset = MagicMock()
    context = ExecutionContext(custom={"value": 0})
    steps = [{"model": "a"}, {"model": "b"}]
    call_order = []

    from nirs4all.pipeline.steps.step_runner import StepRunner, StepResult

    def fake_execute(self, step, dataset, context, runner, loaded_binaries=None, prediction_store=None):
        step_id = step.get("model", "unknown")
        context.custom["value"] += 1
        call_order.append((step_id, context.custom["value"]))
        return StepResult(updated_context=context, artifacts=[])

    monkeypatch.setattr(StepRunner, "execute", fake_execute)

    result = runner.run_steps(steps, dataset, context, execution="sequential")

    assert call_order == [("a", 1), ("b", 2)]
    assert result.custom["value"] == 2


def test_run_step_none_returns_context(tmp_path):
    """Test that run_step with None step returns context unchanged."""
    runner = PipelineRunner(workspace_path=tmp_path / "workspace_none", save_files=False, enable_tab_reports=False)
    dataset = MagicMock()
    context = ExecutionContext(custom={"value": 1})

    from nirs4all.data.predictions import Predictions
    prediction_store = Predictions()

    result = runner.run_step(None, dataset, context, prediction_store)

    assert result is context

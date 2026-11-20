"""Inference capture cannot leak into a later pipeline sharing runtime objects."""

from types import SimpleNamespace

import pytest
from sklearn.linear_model import Ridge

from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, PipelineState, RuntimeContext, StepMetadata
from nirs4all.pipeline.execution.executor import PipelineExecutor


@pytest.mark.parametrize("fail", [False, True])
def test_capture_scope_restores_shared_runtime_on_success_and_failure(monkeypatch, fail):
    executor = PipelineExecutor(step_runner=SimpleNamespace(), save_artifacts=True)
    old_predictors = {"existing-model": object()}
    runtime = RuntimeContext(artifact_registry=object(), fitted_predictors=old_predictors)
    context = ExecutionContext(selector=DataSelector(), state=PipelineState(), metadata=StepMetadata())
    dataset = SimpleNamespace(name="capture_scope")
    observed = []
    monkeypatch.setattr(executor, "_compute_pipeline_hash", lambda steps: "metadata-only")
    monkeypatch.setattr(executor, "_begin_pipeline", lambda *args: (None, None, "metadata-only"))
    monkeypatch.setattr(executor, "_finalize_pipeline", lambda *args, **kwargs: None)

    def execute_steps(steps, data, ctx, active_runtime, *args):
        observed.append(active_runtime.capture_fitted_predictor)
        if active_runtime.capture_fitted_predictor:
            active_runtime.fitted_predictors["new-model"] = object()
            if fail:
                raise RuntimeError("capture scope test failure")
        return ctx, data

    monkeypatch.setattr(executor, "_execute_steps", execute_steps)
    steps = [{"model": MetaModel(Ridge())}]
    if fail:
        with pytest.raises(RuntimeError, match="capture scope test failure"):
            executor.execute(steps, "meta", dataset, context, runtime)
    else:
        executor.execute(steps, "meta", dataset, context, runtime)
    assert runtime.capture_fitted_predictor is False
    assert runtime.fitted_training_steps is None
    assert runtime.fitted_predictors is old_predictors
    assert set(old_predictors) == {"existing-model"}
    executor.execute([{"branch": [[Ridge()], [Ridge()]]}], "other", dataset, context, runtime)
    assert observed == [True, False]

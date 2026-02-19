import pytest

import nirs4all
import nirs4all.pipeline as pipeline_module
from nirs4all.pipeline import (
    DatasetInfo,
    Run,
    RunConfig,
    RunStatus,
    TemplateInfo,
    get_metric_info,
    is_better_score,
)


def test_run_roundtrip_and_transitions() -> None:
    run = Run(
        name="phase3-adoption",
        templates=[TemplateInfo(id="tpl1", name="Template 1", expansion_count=2)],
        datasets=[DatasetInfo(name="d1", path="/tmp/d1.csv")],
        config=RunConfig(metric="rmse"),
    )

    assert run.status == RunStatus.QUEUED
    assert run.total_pipeline_configs == 2
    assert run.total_results_expected == 2

    run.transition_to(RunStatus.RUNNING)
    run.transition_to(RunStatus.COMPLETED)

    assert run.started_at is not None
    assert run.completed_at is not None

    loaded = Run.from_dict(run.to_dict())
    assert loaded.name == run.name
    assert loaded.status == RunStatus.COMPLETED
    assert loaded.total_results_expected == 2

def test_run_invalid_transition_raises() -> None:
    run = Run(name="invalid")
    with pytest.raises(ValueError):
        run.transition_to(RunStatus.COMPLETED)

def test_metric_metadata_and_comparison_helpers() -> None:
    rmse_info = get_metric_info("rmse")
    assert rmse_info["higher_is_better"] is False
    assert is_better_score(0.1, 0.5, "rmse") is True
    assert is_better_score(0.5, 0.1, "rmse") is False

    acc_info = get_metric_info("accuracy")
    assert acc_info["higher_is_better"] is True
    assert is_better_score(0.9, 0.8, "accuracy") is True

def test_run_entities_are_public_api() -> None:
    assert hasattr(pipeline_module, "Run")
    assert hasattr(pipeline_module, "RunStatus")
    assert hasattr(nirs4all, "Run")
    assert nirs4all.RunStatus.QUEUED.value == "queued"

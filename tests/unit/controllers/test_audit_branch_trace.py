"""Parent and child artifacts retain their real branch trace ownership."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters import XOutlierFilter
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.execution.executor import PipelineExecutor, _StepArtifactValue
from nirs4all.pipeline.steps.step_runner import StepRunner
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.trace.recorder import TraceRecorder


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("tuple_child", [False, True])
def test_filter_parent_artifact_precedes_children_and_has_single_trace_owner(tmp_path, nested, tuple_child):
    rng = np.random.default_rng(43)
    dataset = SpectroDataset("branch_trace")
    dataset.add_samples(rng.normal(size=(24, 10)), {"partition": "train"})
    dataset.add_samples(rng.normal(size=(8, 10)), {"partition": "test"})
    dataset.add_targets(rng.normal(size=32))
    raw_train_x = dataset.x({"partition": "train"}).copy()
    raw_train_y = dataset.y({"partition": "train"}).copy()
    filter_step = {"branch": {
        "by_filter": XOutlierFilter(method="isolation_forest", contamination=0.25, random_state=43),
        "steps": [{"concat_transform": [StandardScaler(), PCA(2)]}] if tuple_child else [StandardScaler()],
    }}
    step = {"branch": [[filter_step]]} if nested else filter_step
    registry = ArtifactRegistry(workspace=tmp_path, dataset=dataset.name, pipeline_id="trace_pipeline")
    recorder = TraceRecorder(pipeline_uid="trace_pipeline")
    runner = StepRunner(verbose=0, show_spinner=False)
    runtime = RuntimeContext(step_runner=runner, artifact_registry=registry, trace_recorder=recorder,
                             pipeline_name="trace_pipeline", pipeline_uid="trace_pipeline", step_number=2)
    context = ExecutionContext()
    context.selector.partition = None
    executor = PipelineExecutor(runner, save_charts=False)
    executor.step_number = 2
    persisted = []
    executor._execute_single_step(step, dataset, context, runtime, all_artifacts=persisted)
    trace = recorder.finalize()
    records = registry.get_artifacts_for_step("trace_pipeline", 2)
    filter_records = [record for record in records if (record.custom_name or "").startswith("branch_filter_")]
    assert len(filter_records) == 1
    filter_record = filter_records[0]
    wrapped = registry.load_artifact(filter_record)
    assert isinstance(wrapped, _StepArtifactValue)
    assert isinstance(wrapped.value, XOutlierFilter)
    mask = wrapped.value.get_mask(raw_train_x, raw_train_y)
    assert mask.shape == (24,)
    assert mask.any() and (~mask).any()

    filter_owners = [step for step in trace.steps if filter_record.artifact_id in step.artifacts.artifact_ids]
    assert len(filter_owners) == 1
    owner = filter_owners[0]
    assert owner.operator_type == "branch"
    assert owner.operator_config["separation_type"] == "by_filter"
    assert owner.branch_path == ([0] if nested else [])
    assert owner.produces_branches
    if nested:
        outer = [step for step in trace.steps if step.produces_branches and not step.branch_path]
        assert len(outer) == 1
        assert trace.steps.index(outer[0]) < trace.steps.index(owner)

    parent_index = trace.steps.index(owner)
    child_steps = [step for step in trace.steps if step.artifacts.artifact_ids and step is not owner]
    assert len(child_steps) == 2
    for child in child_steps:
        assert trace.steps.index(child) > parent_index
        assert len(child.branch_path) == len(owner.branch_path) + 1
        assert child.branch_path[:-1] == owner.branch_path
        assert filter_record.artifact_id not in child.artifacts.artifact_ids
        assert child.artifacts.artifact_ids
    trace_ids = [artifact_id for step in trace.steps for artifact_id in step.artifacts.artifact_ids]
    assert len(trace_ids) == len(set(trace_ids))
    assert set(trace_ids) == {record.artifact_id for record in records}
    assert set(trace_ids) == {artifact["artifact_id"] for artifact in persisted}
    assert recorder.current_step is None
    assert recorder.current_branch_path() == []
    with pytest.raises(RuntimeError, match="no step is active"):
        recorder.record_artifact("unowned")

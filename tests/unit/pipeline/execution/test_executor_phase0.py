"""Phase 0 regression tests for PipelineExecutor."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import polars as pl

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.data.raw_multisource import RawMultiSourceDataset
from nirs4all.data.relations import RepetitionSpec
from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.execution.result import StepResult
from nirs4all.pipeline.steps.step_runner import StepRunner


def test_execute_passes_dataset_content_hash_to_begin_pipeline():
    """begin_pipeline should receive dataset.content_hash(), not pipeline hash."""
    store = Mock()
    store.begin_pipeline.return_value = "pipeline-1"

    dataset = Mock()
    dataset.name = "dataset_a"
    dataset.content_hash.return_value = "dataset-hash-123"
    dataset.short_preprocessings_str.return_value = ""

    runtime_context = RuntimeContext(store=store, run_id="run-1")
    context = Mock()

    executor = PipelineExecutor(step_runner=Mock(), mode="train")
    executor._execute_steps = Mock(return_value=(context, dataset))

    executor.execute(
        steps=[],
        config_name="cfg",
        dataset=dataset,
        context=context,
        runtime_context=runtime_context,
        prediction_store=Predictions(),
        original_template=[{"_or_": ["SNV", "MSC"]}, {"model": "PLSRegression"}],
    )

    assert store.begin_pipeline.call_count == 1
    assert store.begin_pipeline.call_args.kwargs["dataset_hash"] == "dataset-hash-123"
    assert store.begin_pipeline.call_args.kwargs["original_template"] == [{"_or_": ["SNV", "MSC"]}, {"model": "PLSRegression"}]


def test_execute_steps_promotes_rep_fusion_dataset_override():
    raw = RawMultiSourceDataset.from_sources(
        RepetitionSpec(sample_id="sid", link_by="sid"),
        {"A": np.array([[1.0], [3.0]]), "B": np.array([[10.0], [20.0]])},
        {"A": ["S1", "S2"], "B": ["S1", "S2"]},
        targets_by_source={"A": [10.0, 20.0]},
    )
    executor = PipelineExecutor(step_runner=StepRunner(), mode="train")
    context = executor.initialize_context(raw)

    context, dataset = executor._execute_steps(
        steps=[{"rep_fusion": "per_source_aggregate"}],
        dataset=raw,
        context=context,
        runtime_context=None,
        prediction_store=Predictions(),
        all_artifacts=[],
    )

    assert isinstance(dataset, SpectroDataset)
    assert "dataset_override" not in context.custom
    assert context.aggregate_column == "physical_sample_id"
    assert context.selector.processing == [["raw"]]
    np.testing.assert_allclose(dataset.x({"partition": "train"}), np.array([[1.0, 10.0], [3.0, 20.0]]))
    np.testing.assert_allclose(dataset.y({"partition": "train", "y": "raw"}).ravel(), np.array([10.0, 20.0]))


def test_execute_minimal_promotes_dataset_override_between_steps():
    base = SpectroDataset("base")
    base.add_samples(np.array([[0.0]]), {"partition": "train"})
    override = SpectroDataset("override")
    override.add_samples(np.array([[1.0]]), {"partition": "train"})

    class _Runner:
        def __init__(self) -> None:
            self.seen: list[SpectroDataset] = []

        def execute(self, step, dataset, context, runtime_context, loaded_binaries=None, prediction_store=None):
            self.seen.append(dataset)
            if len(self.seen) == 1:
                updated = context.copy()
                updated.custom["dataset_override"] = override
                return StepResult(updated_context=updated)
            return StepResult(updated_context=context)

    class _Step:
        def __init__(self, step_index: int) -> None:
            self.step_index = step_index
            self.substep_index = None
            self.branch_path = []

    class _Minimal:
        steps = [_Step(1), _Step(2)]
        model_step_index = None

    runner = _Runner()
    executor = PipelineExecutor(step_runner=runner, mode="predict")
    context = executor.initialize_context(base)

    executor.execute_minimal(
        steps=[{"first": True}, {"second": True}],
        minimal_pipeline=_Minimal(),
        dataset=base,
        context=context,
        runtime_context=RuntimeContext(),
        prediction_store=Predictions(),
    )

    assert runner.seen == [base, override]


def test_flush_predictions_uses_refit_runtime_overrides():
    """Refit runtime labels should be applied before saving predictions."""
    store = Mock()
    store.transaction.return_value = nullcontext()  # flush() wraps its body in a transaction
    store.get_chains_for_pipeline.return_value = pl.DataFrame(
        {
            "chain_id": ["chain-1"],
            "model_step_idx": [0],
            "branch_path": [None],
            "model_class": ["DummyModel"],
            "preprocessings": [""],
        }
    )
    store.save_prediction.return_value = "pred-1"

    prediction_store = Predictions()
    prediction_store.add_prediction(
        dataset_name="dataset_a",
        model_name="DummyModel",
        model_classname="DummyModel",
        fold_id="fold_0",
        partition="test",
        metric="rmse",
        y_true=np.array([1.0]),
        y_pred=np.array([1.0]),
    )

    runtime_context = RuntimeContext(
        phase=ExecutionPhase.REFIT,
        refit_fold_id="final",
        refit_context_name="standalone",
    )

    executor = PipelineExecutor(step_runner=Mock(), mode="train")
    executor._flush_predictions_to_store(
        store=store,
        pipeline_id="pipeline-1",
        prediction_store=prediction_store,
        runtime_context=runtime_context,
    )

    kwargs = store.save_prediction.call_args.kwargs
    assert kwargs["fold_id"] == "final"
    assert kwargs["refit_context"] == "standalone"

def test_record_dataset_shapes_is_gated_by_verbose_level():
    """Shape tracing should not materialize arrays unless verbose>=2."""
    executor = PipelineExecutor(step_runner=Mock(), mode="train", verbose=1)

    dataset = Mock()
    context = Mock()
    context.selector = {"partition": "train", "processing": [["raw"]]}
    runtime_context = Mock()
    runtime_context.trace_recorder = object()

    executor._record_dataset_shapes(
        dataset=dataset,
        context=context,
        runtime_context=runtime_context,
        is_input=True,
    )

    dataset.x.assert_not_called()

def test_record_dataset_shapes_reuses_cached_selector_shapes():
    """Repeated shape tracing for unchanged selector should reuse cache."""
    executor = PipelineExecutor(step_runner=Mock(), mode="train", verbose=2)

    dataset = Mock()
    dataset.num_samples = 5
    source = Mock()
    source.num_processings = 1
    source.num_features = 3
    features = Mock()
    features.sources = [source]
    dataset._features = features
    dataset.x.side_effect = [
        np.ones((5, 3), dtype=float),   # 2D layout
        np.ones((5, 1, 3), dtype=float),  # 3D source layout
    ]

    context = Mock()
    context.selector = {"partition": "train", "processing": [["raw"]]}
    runtime_context = Mock()
    runtime_context.trace_recorder = object()

    executor._record_dataset_shapes(dataset, context, runtime_context, is_input=True)
    executor._record_dataset_shapes(dataset, context, runtime_context, is_input=False)

    # First call materializes 2D + 3D once; second call should hit cache.
    assert dataset.x.call_count == 2
    runtime_context.record_input_shapes.assert_called_once()
    runtime_context.record_output_shapes.assert_called_once()

def test_flush_predictions_uses_preindexed_chain_matching():
    """Chain selection should resolve by step/branch/class/preprocessing."""
    store = Mock()
    store.transaction.return_value = nullcontext()  # flush() wraps its body in a transaction
    store.get_chains_for_pipeline.return_value = pl.DataFrame(
        {
            "chain_id": ["chain-0", "chain-1", "chain-2", "chain-3"],
            "model_step_idx": [0, 2, 2, 2],
            "branch_path": [None, "[0]", "[1]", "[1]"],
            "model_class": ["DummyModel", "OtherModel", "DummyModel", "DummyModel"],
            "preprocessings": ["", "pp_a", "pp_target", "pp_other"],
        }
    )
    store.save_prediction.return_value = "pred-1"

    prediction_store = Predictions()
    prediction_store.add_prediction(
        dataset_name="dataset_a",
        model_name="DummyModel",
        model_classname="DummyModel",
        fold_id="fold_0",
        partition="test",
        metric="rmse",
        y_true=np.array([1.0]),
        y_pred=np.array([1.0]),
        step_idx=2,
        branch_id=1,
        preprocessings="pp_target",
    )

    executor = PipelineExecutor(step_runner=Mock(), mode="train")
    executor._flush_predictions_to_store(
        store=store,
        pipeline_id="pipeline-1",
        prediction_store=prediction_store,
        runtime_context=None,
    )

    kwargs = store.save_prediction.call_args.kwargs
    assert kwargs["chain_id"] == "chain-2"

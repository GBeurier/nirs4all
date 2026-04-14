"""Unit tests for grouped splitting validation and execution."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from nirs4all.controllers.splitters.split import (
    CrossValidatorController,
    compute_effective_groups,
    resolve_split_groups,
)
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.context import (
    DataSelector,
    ExecutionContext,
    PipelineState,
    RuntimeContext,
    StepMetadata,
)
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


def make_step_info(operator, step=None):
    """Create a ParsedStep instance for controller tests."""
    if step is None:
        step = {}
    return ParsedStep(
        operator=operator,
        keyword="",
        step_type=StepType.DIRECT,
        original_step=step,
        metadata={},
    )


def make_mock_runtime_context():
    """Create a minimal runtime context for controller execution."""
    mock_runtime = Mock(spec=RuntimeContext)
    mock_runtime.saver = None
    return mock_runtime


def make_execution_context():
    """Create a train execution context."""
    return ExecutionContext(
        selector=DataSelector(processing=[["raw"]]),
        state=PipelineState(),
        metadata=StepMetadata(),
    )


def make_dataset():
    """Create a dataset with metadata columns usable for grouping tests."""
    dataset = SpectroDataset(name="test")
    X = np.random.RandomState(0).rand(16, 6)
    y = np.linspace(0.0, 1.0, 16)
    metadata = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(1, 9) for _ in range(2)],
            "batch": ["B1", "B2", "B3", "B4"] * 4,
            "location": ["L1", "L1", "L2", "L2"] * 4,
        }
    )

    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)
    dataset.add_metadata(metadata)
    return dataset


def make_train_context(dataset):
    """Return the train-only selector used by grouped split helpers."""
    return make_execution_context().with_partition("train")


def assert_no_group_overlap(groups, folds):
    """Assert that no effective group appears in both train and validation."""
    for train_idx, val_idx in folds:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert not (train_groups & val_groups)


class TestGroupSplitSyntax:
    """Basic controller matching tests."""

    def test_matches_split_keyword(self):
        controller = CrossValidatorController()
        step = {"split": GroupKFold(), "group": "batch"}
        assert controller.matches(step, None, "split")

    def test_matches_split_in_dict(self):
        controller = CrossValidatorController()
        step = {"split": GroupKFold(), "group_by": "batch"}
        assert controller.matches(step, None, "")

    def test_backward_compatible_matching(self):
        controller = CrossValidatorController()
        splitter = GroupKFold()
        assert controller.matches(splitter, splitter, "")

    def test_no_match_without_split(self):
        controller = CrossValidatorController()
        assert not controller.matches({"other": "value"}, None, "")

    def test_no_match_none_operator(self):
        controller = CrossValidatorController()
        assert not controller.matches({}, None, "")


class TestResolveSplitGroups:
    """Unit tests for the shared group resolution helper."""

    def test_required_without_repetition_or_group_by_errors(self):
        dataset = make_dataset()

        with pytest.raises(ValueError, match="requires an effective group"):
            resolve_split_groups(
                dataset=dataset,
                splitter=GroupKFold(n_splits=4),
                context=make_train_context(dataset),
                include_augmented=False,
            )

    def test_required_with_repetition_only_warns_and_resolves(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")

        with pytest.warns(UserWarning, match="only the configured dataset repetition"):
            resolved = resolve_split_groups(
                dataset=dataset,
                splitter=GroupKFold(n_splits=4),
                context=make_train_context(dataset),
                include_augmented=False,
            )

        assert resolved.group_by is None
        assert resolved.satisfied_by_repetition_only is True
        assert resolved.requires_wrapper is False
        assert resolved.effective_groups is not None
        assert tuple(resolved.effective_groups[:4]) == ("S1", "S1", "S2", "S2")

    def test_required_with_group_by_only_resolves_without_warning(self):
        dataset = make_dataset()

        resolved = resolve_split_groups(
            dataset=dataset,
            splitter=GroupKFold(n_splits=4),
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )

        assert resolved.group_by == "batch"
        assert resolved.satisfied_by_repetition_only is False
        assert resolved.requires_wrapper is False
        assert tuple(resolved.effective_groups[:4]) == ("B1", "B2", "B3", "B4")

    def test_required_with_group_by_and_repetition_combines_groups(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")

        resolved = resolve_split_groups(
            dataset=dataset,
            splitter=GroupKFold(n_splits=4),
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )

        assert resolved.effective_groups is not None
        assert list(resolved.effective_groups[:4]) == [0, 0, 1, 1]

    def test_optional_with_legacy_group_alias_warns_and_normalizes(self):
        dataset = make_dataset()

        with pytest.warns(DeprecationWarning, match="Use 'group_by' instead"):
            resolved = resolve_split_groups(
                dataset=dataset,
                splitter=KFold(n_splits=4),
                legacy_group="batch",
                context=make_train_context(dataset),
                include_augmented=False,
            )

        assert resolved.group_by == "batch"
        assert resolved.requires_wrapper is True
        assert resolved.effective_groups is not None


class TestGroupSplitExecution:
    """Execution-level tests for grouped and wrapped splitters."""

    def test_required_splitter_without_repetition_or_group_by_errors(self):
        dataset = make_dataset()
        controller = CrossValidatorController()

        with pytest.raises(ValueError, match="requires an effective group"):
            controller.execute(
                step_info=make_step_info(GroupKFold(n_splits=4), {"split": GroupKFold(n_splits=4)}),
                dataset=dataset,
                context=make_execution_context(),
                runtime_context=make_mock_runtime_context(),
                mode="train",
            )

    def test_required_splitter_with_repetition_only_warns_and_executes(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")
        step = {"split": GroupKFold(n_splits=4)}
        controller = CrossValidatorController()

        with pytest.warns(UserWarning, match="only the configured dataset repetition"):
            _, step_output = controller.execute(
                step_info=make_step_info(step["split"], step),
                dataset=dataset,
                context=make_execution_context(),
                runtime_context=make_mock_runtime_context(),
                mode="train",
            )

        assert len(dataset.folds) == 4
        sample_groups = compute_effective_groups(
            dataset,
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert sample_groups is not None
        assert_no_group_overlap(sample_groups, dataset.folds)
        assert "groups-rep-sample_id" in step_output.outputs[0][1]

    def test_required_splitter_with_group_by_only_executes(self):
        dataset = make_dataset()
        step = {"split": GroupKFold(n_splits=4), "group_by": "batch"}
        controller = CrossValidatorController()

        _, step_output = controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="train",
        )

        assert len(dataset.folds) == 4
        batch_groups = compute_effective_groups(
            dataset,
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert batch_groups is not None
        assert_no_group_overlap(batch_groups, dataset.folds)
        assert "groups-batch" in step_output.outputs[0][1]

    def test_required_splitter_with_group_by_and_repetition_combines_groups(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")
        step = {
            "split": GroupShuffleSplit(n_splits=3, test_size=0.5, random_state=42),
            "group_by": "batch",
        }
        controller = CrossValidatorController()

        _, step_output = controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="train",
        )

        effective_groups = compute_effective_groups(
            dataset,
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert effective_groups is not None
        assert_no_group_overlap(effective_groups, dataset.folds)
        assert "groups-rep-sample_id+batch" in step_output.outputs[0][1]

    def test_optional_splitter_with_repetition_only_executes_via_wrapper(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")
        step = {"split": KFold(n_splits=4, shuffle=False)}
        controller = CrossValidatorController()

        _, step_output = controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="train",
        )

        sample_groups = compute_effective_groups(
            dataset,
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert sample_groups is not None
        assert len(dataset.folds) == 4
        assert_no_group_overlap(sample_groups, dataset.folds)
        assert "groups-rep-sample_id" in step_output.outputs[0][1]

    def test_optional_splitter_with_group_by_only_executes_via_wrapper(self):
        dataset = make_dataset()
        step = {"split": KFold(n_splits=4, shuffle=False), "group_by": "batch"}
        controller = CrossValidatorController()

        _, step_output = controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="train",
        )

        batch_groups = compute_effective_groups(
            dataset,
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert batch_groups is not None
        assert len(dataset.folds) == 4
        assert_no_group_overlap(batch_groups, dataset.folds)
        assert "groups-batch" in step_output.outputs[0][1]

    def test_optional_splitter_with_repetition_and_group_by_combines_groups(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")
        step = {"split": KFold(n_splits=2, shuffle=False), "group_by": "batch"}
        controller = CrossValidatorController()

        _, step_output = controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="train",
        )

        effective_groups = compute_effective_groups(
            dataset,
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert effective_groups is not None
        assert len(dataset.folds) == 2
        assert_no_group_overlap(effective_groups, dataset.folds)
        assert "groups-rep-sample_id+batch" in step_output.outputs[0][1]

    def test_combined_constraints_prevent_raw_group_by_overlap(self):
        dataset = make_dataset()
        dataset.set_repetition("sample_id")
        step = {"split": KFold(n_splits=2, shuffle=False), "group_by": "batch"}
        controller = CrossValidatorController()

        controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="train",
        )

        effective_groups = compute_effective_groups(
            dataset,
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )
        raw_group_by_only = compute_effective_groups(
            dataset,
            group_by="batch",
            ignore_repetition=True,
            context=make_train_context(dataset),
            include_augmented=False,
        )

        assert effective_groups is not None
        assert raw_group_by_only is not None
        assert_no_group_overlap(effective_groups, dataset.folds)
        assert not any(
            set(raw_group_by_only[train_idx]) & set(raw_group_by_only[val_idx])
            for train_idx, val_idx in dataset.folds
        )

    def test_group_by_must_exist_in_metadata(self):
        dataset = make_dataset()
        step = {"split": GroupKFold(n_splits=4), "group_by": "missing"}
        controller = CrossValidatorController()

        with pytest.raises(ValueError, match="Grouping column 'missing' not found in metadata"):
            controller.execute(
                step_info=make_step_info(step["split"], step),
                dataset=dataset,
                context=make_execution_context(),
                runtime_context=make_mock_runtime_context(),
                mode="train",
            )

    def test_required_splitter_with_legacy_group_alias_warns_and_executes(self):
        dataset = make_dataset()
        step = {"split": GroupKFold(n_splits=4), "group": "batch"}
        controller = CrossValidatorController()

        with pytest.warns(DeprecationWarning, match="Use 'group_by' instead"):
            _, step_output = controller.execute(
                step_info=make_step_info(step["split"], step),
                dataset=dataset,
                context=make_execution_context(),
                runtime_context=make_mock_runtime_context(),
                mode="train",
            )

        batch_groups = compute_effective_groups(
            dataset,
            group_by="batch",
            context=make_train_context(dataset),
            include_augmented=False,
        )
        assert batch_groups is not None
        assert_no_group_overlap(batch_groups, dataset.folds)
        assert "groups-batch" in step_output.outputs[0][1]

    def test_prediction_mode_still_creates_dummy_folds(self):
        dataset = make_dataset()
        step = {"split": GroupKFold(n_splits=4), "group_by": "batch"}
        controller = CrossValidatorController()

        _, step_output = controller.execute(
            step_info=make_step_info(step["split"], step),
            dataset=dataset,
            context=make_execution_context(),
            runtime_context=make_mock_runtime_context(),
            mode="predict",
        )

        assert dataset.folds is not None
        assert len(dataset.folds) == 4
        assert len(step_output.outputs) == 0


class TestSerialization:
    """Serialization should keep the legacy alias for compatibility."""

    def test_serialize_split_with_group(self):
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        pipeline = [{"split": GroupKFold(n_splits=5), "group": "batch_id"}]
        config = PipelineConfigs(pipeline)
        serialized = config.steps[0]

        assert "split" in serialized[0]
        assert "group" in serialized[0]
        assert serialized[0]["group"] == "batch_id"

    def test_roundtrip_serialization(self):
        import json

        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        original = [
            {
                "split": {
                    "class": "sklearn.model_selection.GroupKFold",
                    "params": {"n_splits": 5},
                },
                "group": "sample",
            }
        ]

        config = PipelineConfigs(original)
        serialized = json.dumps(config.steps[0])
        deserialized = json.loads(serialized)

        assert deserialized[0]["group"] == "sample"
        assert "GroupKFold" in deserialized[0]["split"]["class"]

    def test_backward_compatible_serialization(self):
        from nirs4all.pipeline.config.component_serialization import serialize_component

        serialized = serialize_component(GroupKFold(n_splits=5))

        assert "class" in serialized or isinstance(serialized, str)
        if isinstance(serialized, dict):
            assert "GroupKFold" in serialized["class"]

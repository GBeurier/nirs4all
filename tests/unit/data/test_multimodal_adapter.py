"""Typed IO sources retain identities and ranks at the DAG host boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.model_selection import GroupKFold

from nirs4all.core.task_type import TaskType
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.data.targets import Targets
from nirs4all.pipeline.dagml.dataset import _materialize_dataset
from nirs4all.pipeline.dagml.envelope import build_envelope, build_fold_set
from nirs4all.pipeline.dagml.folds import _build_folds
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.resolver import MaterializationResolver


@pytest.fixture
def cohort() -> MultimodalDataset:
    ids = [f"sample-{index}" for index in range(8)]
    order = np.array([6, 2, 7, 1, 4, 0, 5, 3])
    image = np.arange(8 * 2 * 3 * 3, dtype=np.float32).reshape(8, 2, 3, 3)
    return MultimodalDataset(
        {
            "nir": TensorSource(np.arange(40, dtype=np.float64).reshape(8, 5), ids, representation_id="signal_1d"),
            "image": TensorSource(image[order], [ids[index] for index in order], representation_id="rgb_image"),
            "series": TensorSource(np.arange(64, dtype=np.float32).reshape(8, 4, 2), ids, representation_id="series_mv"),
            "metadata": TensorSource(np.array([[index, "a" if index % 2 else "b"] for index in range(8)], dtype=object), ids, representation_id="tabular_mixed"),
        },
        sample_ids=ids,
        y=np.linspace(0.1, 2.9, 8),
        groups=["plant-a", "plant-a", "heldout", "plant-b", "plant-b", "heldout", "plant-c", "plant-c"],
        partitions=["train", "train", "test", "train", "train", "test", "train", "train"],
        name="typed-adapter",
    )


def test_materialization_preserves_raw_ranks_dtype_and_row_identity(cohort: MultimodalDataset) -> None:
    dataset = _materialize_dataset(cohort)
    assert isinstance(dataset, MultimodalSpectroDataset)
    assert dataset.sample_ids == cohort.sample_ids
    assert dataset.source_names == ("nir", "image", "series", "metadata")
    rows = [7, 0, 4]
    blocks = dataset.x_rows(rows, concat_source=False)
    assert [block.shape for block in blocks] == [(3, 5), (3, 2, 3, 3), (3, 4, 2), (3, 2)]
    assert [block.dtype for block in blocks] == [np.dtype("float64"), np.dtype("float32"), np.dtype("float32"), np.dtype("object")]
    np.testing.assert_array_equal(blocks[1][:, 0, 0, 0], np.array(rows) * 18)
    np.testing.assert_array_equal(blocks[3][:, 0], rows)
    assert dataset.index_column("sample", {"partition": "train"}) == [0, 1, 3, 4, 6, 7]
    assert dataset.index_column("sample", {"partition": "test"}) == [2, 5]


def test_wire_ids_remain_explicit_after_cohort_reordering(cohort: MultimodalDataset) -> None:
    ids = ["sample-7", "sample-0", "sample-4"]
    dataset = _materialize_dataset(cohort.take(ids))
    identity = mint_identity(dataset)
    assert identity.observation_ids() == ids
    assert [identity.to_int(sample_id) for sample_id in ids] == [0, 1, 2]
    assert [entry.sample_id for entry in identity.identities] == ids
    resolver = MaterializationResolver(dataset, identity)
    requested = ["sample-4", "sample-7"]
    resolved = resolver.resolve_feature_blocks(requested, include_augmented=False)
    assert resolved["observation_ids"] == requested
    np.testing.assert_array_equal(resolved["blocks"][1][:, 0, 0, 0], [4 * 18, 7 * 18])
    assert resolved["blocks"][1].ndim == 4
    assert resolved["blocks"][2].ndim == 3


def test_group_folds_use_train_rows_and_keep_each_unit_together(cohort: MultimodalDataset) -> None:
    dataset = _materialize_dataset(cohort)
    pool = [7, 0, 4, 1, 6, 3]
    folds = _build_folds(GroupKFold(3), dataset, pool, set())
    assert len(folds) == 3
    assert sorted(row for _, validation in folds for row in validation) == sorted(pool)
    for train, validation in folds:
        assert set(train) | set(validation) == set(pool)
        assert not set(train) & set(validation)
        assert not set(dataset.groups_for_rows(train)) & set(dataset.groups_for_rows(validation))
    excluded = _build_folds(GroupKFold(3), dataset, pool, {0})
    assert all(0 not in train for train, _ in excluded)
    assert sum(0 in validation for _, validation in excluded) == 1
    native_folds = build_fold_set(mint_identity(dataset), folds)
    assert set(native_folds["sample_ids"]) == {cohort.sample_ids[row] for row in pool}


def test_native_envelope_keeps_typed_descriptors_and_group_ids(cohort: MultimodalDataset) -> None:
    dataset = _materialize_dataset(cohort)
    pool = dataset.index_column("sample", {"partition": "train"})
    envelope = build_envelope(
        dataset, mint_identity(dataset), sample_ints=pool,
        group_by_sample={row: str(cohort.groups[row]) for row in pool},
    )
    layout = envelope["plan"]["source_layout"]
    # The native envelope retains schema fingerprints, not the full schema.
    # Check both the schema handed to its builder and its typed source layout.
    sources = dataset.data_schema(layout["source_ids"], [cohort.sample_ids[row] for row in pool])["sources"]
    assert [source["name"] for source in sources] == ["nir", "image", "series", "metadata"]
    assert [source["native_representation"]["rank"] for source in sources] == [2, 4, 3, 2]
    assert [source["native_representation"]["id"] for source in sources] == ["signal_1d", "rgb_image", "series_mv", "tabular_mixed"]
    assert all(source["native_representation"]["axes"][0]["size"] == len(pool) for source in sources)
    assert layout["source_order"] == ["nir", "image", "series", "metadata"]
    assert [block["native_representation"]["rank"] for block in layout["blocks"]] == [2, 4, 3, 2]
    relations = envelope["coordinator_relations"]["records"]
    assert {record["sample_id"] for record in relations} == {cohort.sample_ids[row] for row in pool}
    assert {record["group_id"] for record in relations} == {"plant-a", "plant-b", "plant-c"}


def test_implicit_dense_concatenation_is_refused(cohort: MultimodalDataset) -> None:
    dataset = _materialize_dataset(cohort)
    with pytest.raises(ValueError, match="source-aware"):
        dataset.x({})
    with pytest.raises(ValueError, match="source-aware"):
        dataset.x_rows([0, 1])
    resolver = MaterializationResolver(dataset, mint_identity(dataset))
    with pytest.raises(ValueError, match="source-aware"):
        resolver.resolve_features(["sample-0", "sample-1"])


@pytest.mark.parametrize("source_name", ["image", "metadata"])
def test_content_fingerprint_covers_raw_tensor_and_categorical_cells(cohort: MultimodalDataset, source_name: str) -> None:
    before = _materialize_dataset(cohort).content_hash()
    source = cohort.sources[source_name]
    changed = np.array(source.values, copy=True)
    changed.flat[-1] = "unseen-category" if source_name == "metadata" else changed.flat[-1] + 1
    sources = dict(cohort.sources)
    sources[source_name] = TensorSource(changed, cohort.sample_ids, representation_id=source.representation_id)
    modified = MultimodalDataset(sources, sample_ids=cohort.sample_ids, y=cohort.y, groups=cohort.groups, partitions=cohort.partitions)
    assert _materialize_dataset(modified).content_hash() != before


def test_explicit_regression_keeps_integer_measurements(cohort: MultimodalDataset) -> None:
    targets = np.array([10, 20, 30, 20, 10, 30, 20, 10])
    declared = MultimodalDataset(
        cohort.sources, sample_ids=cohort.sample_ids, y=targets, groups=cohort.groups,
        partitions=cohort.partitions, task_type="regression", target_names=["concentration"],
    )
    dataset = _materialize_dataset(declared)
    assert dataset.is_regression
    np.testing.assert_array_equal(dataset.y({}).ravel(), targets)
    resolver = MaterializationResolver(dataset, mint_identity(dataset))
    resolved = resolver.resolve_targets(["sample-3", "sample-0"])
    assert resolved["values"] == [20, 10]
    assert resolved["target_names"] == ["concentration"]
    assert "validity_masks" not in resolved


def test_partial_target_masks_follow_requested_id_order(cohort: MultimodalDataset) -> None:
    targets = np.column_stack([cohort.y, cohort.y * 2])
    mask = np.ones(targets.shape, dtype=bool)
    mask[0, 1] = False
    mask[4, 0] = False
    targets[~mask] = np.nan
    declared = MultimodalDataset(
        cohort.sources, sample_ids=cohort.sample_ids, y=targets, target_mask=mask,
        target_names=["concentration", "moisture"], task_type="regression",
        groups=cohort.groups, partitions=cohort.partitions,
    )
    dataset = _materialize_dataset(declared)
    resolver = MaterializationResolver(dataset, mint_identity(dataset))
    ids = ["sample-4", "sample-0", "sample-3", "sample-4"]
    resolved = resolver.resolve_targets(ids)
    assert resolved["sample_ids"] == ids
    assert resolved["target_names"] == ["concentration", "moisture"]
    np.testing.assert_array_equal(resolved["validity_masks"], mask[[4, 0, 3, 4]])
    expected = np.where(mask, targets, 0).astype(np.float32)[[4, 0, 3, 4]]
    np.testing.assert_array_equal(resolved["values"], expected)
    assert np.isfinite(resolved["values"]).all()
    assert np.isnan(declared.y[~mask]).all()


@pytest.mark.parametrize("task_type", [None, "classification"])
def test_partial_targets_require_declared_regression(cohort: MultimodalDataset, task_type: str | None) -> None:
    mask = np.ones(len(cohort), dtype=bool)
    mask[0] = False
    declared = MultimodalDataset(
        cohort.sources, sample_ids=cohort.sample_ids, y=cohort.y, target_mask=mask,
        task_type=task_type, groups=cohort.groups, partitions=cohort.partitions,
    )
    with pytest.raises(ValueError, match="explicit task_type='regression'"):
        _materialize_dataset(declared)


@pytest.mark.parametrize("classes", [np.array(["a", "c", "e"]), np.array([2, 7, 42]), np.array([2, 2**32 + 2, 2**63 - 1], dtype=np.int64)])
def test_class_encoding_uses_train_only_and_preserves_interleaved_order(cohort: MultimodalDataset, classes: np.ndarray) -> None:
    codes = np.array([0, 1, 2, 2, 0, 1, 1, 0])
    labels = classes[codes]
    declared = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=labels, partitions=cohort.partitions)
    dataset = _materialize_dataset(declared)
    train_ids = [sample for sample, part in zip(cohort.sample_ids, cohort.partitions, strict=True) if part == "train"]
    train_only = _materialize_dataset(declared.take(train_ids))
    assert dataset.task_type == train_only.task_type
    np.testing.assert_array_equal(dataset.y({}).ravel(), codes)
    np.testing.assert_array_equal(dataset.y({"partition": "train"}), train_only.y({}))
    decoder = MaterializationResolver(dataset, mint_identity(dataset)).target_decoder()
    assert decoder is not None
    np.testing.assert_array_equal(decoder.inverse_transform(dataset.y({})).ravel(), labels)
    changed = labels.copy()
    changed[[2, 5]] = classes[[0, 2]]
    modified = _materialize_dataset(MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=changed, partitions=cohort.partitions))
    np.testing.assert_array_equal(modified.y({"partition": "train"}), dataset.y({"partition": "train"}))
    assert modified.task_type == dataset.task_type


@pytest.mark.parametrize("labels,unknown", [(np.array(["a", "c", "e", "e", "a", "c", "c", "a"]), "d"), (np.array([2., 7., 42., 42., 2., 7., 7., 2.]), 3.14)])
def test_unknown_held_out_class_is_refused_without_reencoding_train(cohort: MultimodalDataset, labels: np.ndarray, unknown: Any) -> None:
    labels[2] = unknown
    declared = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=labels, partitions=cohort.partitions)
    with pytest.raises(ValueError, match="labels absent from training"):
        _materialize_dataset(declared)
    targets = Targets()
    with pytest.raises(ValueError, match="labels absent from training"):
        targets.add_targets(labels, fit_indices=[0, 1, 3, 4, 6, 7])
    assert targets.num_processings == 0


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [8], [0.5], [[0]], [True]])
def test_target_fit_indices_reject_invalid_subset(indices: Any) -> None:
    targets = Targets()
    with pytest.raises(ValueError, match="fit_indices"):
        targets.add_targets(np.arange(8), fit_indices=indices)
    assert targets.num_processings == 0


def test_target_fit_indices_are_initialization_only_and_infer_task_from_train() -> None:
    targets = Targets()
    # Appending with a new fitting subset must never replace the initial encoder.
    targets.add_targets(np.array([0., 1., 0., 1.]), fit_indices=[0, 1, 2])
    assert targets.task_type is not None and targets.task_type.is_classification
    with pytest.raises(ValueError, match="initializing"):
        targets.add_targets([0.], fit_indices=[0])
    regression = Targets()
    regression.add_targets(np.array([0.25, 0., 0.75, 1.]), fit_indices=[0, 2])
    assert regression.task_type == TaskType.REGRESSION
    np.testing.assert_array_equal(regression.get_targets().ravel(), [0.25, 0., 0.75, 1.])


@pytest.mark.parametrize("class_count,explicit", [(60, False), (101, True)])
def test_numeric_class_axis_preserves_int64_above_heuristic_threshold(class_count: int, explicit: bool) -> None:
    labels = np.iinfo(np.int64).max - np.arange(class_count, dtype=np.int64)
    targets = Targets()
    if explicit:
        targets.set_task_type(TaskType.MULTICLASS_CLASSIFICATION)
    targets.add_targets(labels, fit_indices=np.arange(class_count))
    encoded = targets.get_targets()
    assert len(np.unique(encoded)) == class_count
    assert targets.task_type is not None and targets.task_type.is_classification
    np.testing.assert_array_equal(targets.invert_transform(encoded, "numeric").ravel(), labels)


@pytest.mark.parametrize("with_targets", [False, True])
def test_prediction_only_cohort_keeps_existing_target_behavior(cohort: MultimodalDataset, with_targets: bool) -> None:
    declared = MultimodalDataset(
        cohort.sources, sample_ids=cohort.sample_ids, y=cohort.y if with_targets else None,
        partitions=["predict"] * len(cohort),
    )
    dataset = _materialize_dataset(declared)
    assert dataset.sample_ids == cohort.sample_ids
    assert dataset.index_column("sample", {"partition": "train"}) == []
    if with_targets:
        np.testing.assert_array_equal(dataset.y({}).ravel(), cohort.y.astype(np.float32))
    else:
        assert dataset._targets.num_samples == 0


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.int32, np.int64, np.bool_])
def test_object_scalar_fingerprint_matches_io_json_roundtrip(cohort: MultimodalDataset, scalar_type: Any) -> None:
    values = np.array([[scalar_type(row), "a"] for row in range(len(cohort))], dtype=object)
    declared = MultimodalDataset(
        {**cohort.sources, "metadata": TensorSource(values, cohort.sample_ids, representation_id="tabular_mixed")},
        sample_ids=cohort.sample_ids, y=cohort.y, groups=cohort.groups, partitions=cohort.partitions,
    )
    reloaded = MultimodalDataset.from_dict(declared.to_dict())
    assert MultimodalSpectroDataset(declared).content_hash() == MultimodalSpectroDataset(reloaded).content_hash()

"""Assembled-dataset Playground selection keeps all scientific identities."""

import numpy as np
import pytest
from sklearn.model_selection import KFold

from nirs4all.analysis.playground_dataset import extract_playground_dataset, playground_metadata_columns
from nirs4all.analysis.playground_facade import preview_spectro_dataset
from nirs4all.analysis.playground_prepare import PreviewStep
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.transforms import StandardNormalVariate


@pytest.fixture
def multisource_dataset():
    dataset = SpectroDataset("playground_sources")
    train_a = np.arange(30, dtype=float).reshape(6, 5) + 1
    train_b = np.arange(18, dtype=float).reshape(6, 3) + 101
    test_a = np.arange(20, dtype=float).reshape(4, 5) + 201
    test_b = np.arange(12, dtype=float).reshape(4, 3) + 301
    headers = [["1100", "1101", "1102", "1103", "1104"], ["900.125", "901.25", "902.5"]]
    dataset.add_samples([train_a, train_b], {"partition": "train"}, headers=headers, header_unit=["nm", "nm"])
    dataset.add_samples([test_a, test_b], {"partition": "test"})
    dataset.add_targets(np.column_stack([np.arange(10), np.arange(10) * 10 + 0.5]))
    dataset.add_metadata(
        np.asarray([[f"S{i}", f"subject-{i // 2}", "train" if i < 6 else "test"] for i in range(10)], dtype=object),
        headers=["sample_id", "subject", "declared_partition"],
    )
    dataset.set_repetition("subject")
    return dataset, (train_a, train_b, test_a, test_b)


def test_explicit_source_target_and_partition_are_preserved(multisource_dataset):
    dataset, (_, train_b, _, test_b) = multisource_dataset
    selected = extract_playground_dataset(dataset, source_index=1, target_index=1)

    np.testing.assert_array_equal(selected.batch.x, np.concatenate([train_b, test_b]))
    np.testing.assert_array_equal(selected.batch.y, np.arange(10) * 10 + 0.5)
    np.testing.assert_array_equal(selected.batch.sample_ids, [f"S{i}" for i in range(10)])
    np.testing.assert_array_equal(selected.batch.partitions, ["train"] * 6 + ["test"] * 4)
    np.testing.assert_array_equal(selected.batch.wavelengths, [900.125, 901.25, 902.5])
    assert selected.batch.header_unit == "nm"
    assert selected.evidence == {
        "partition": "all", "source_index": 1, "target_index": 1, "n_sources": 2,
        "num_targets": 2, "n_train": 6, "n_test": 4,
        "sample_id_sources": ["metadata.sample_id", "metadata.sample_id"],
        "original_headers": ["900.125", "901.25", "902.5"], "repetition_column": "subject",
    }
    assert selected.diagnostics == ()


def test_dataset_preview_keeps_heldout_rows_outside_cv(multisource_dataset):
    dataset, _ = multisource_dataset
    result = preview_spectro_dataset(
        dataset,
        source_index=0,
        target_index=0,
        steps=[PreviewStep("snv", "preprocessing", "SNV", StandardNormalVariate()),
               PreviewStep("cv", "splitting", "KFold", KFold(n_splits=3))],
        options={"compute_repetitions": False},
    )

    assert result["success"] is True
    assert result["dataset_selection"]["n_test"] == 4
    assert result["source_partitions"]["test_indices"] == [6, 7, 8, 9]
    for fold in result["folds"]["folds"]:
        assert set(fold["train_indices"]) <= set(range(6))
        assert set(fold["test_indices"]) <= set(range(6))


def test_test_only_selection_retains_truthful_partition(multisource_dataset):
    dataset, (_, _, _, test_b) = multisource_dataset
    selected = extract_playground_dataset(dataset, partition="test", source_index=1, target_index=0)
    np.testing.assert_array_equal(selected.batch.x, test_b)
    np.testing.assert_array_equal(selected.batch.partitions, ["test"] * 4)
    assert selected.evidence["n_train"] == 0
    assert selected.evidence["n_test"] == 4


def test_absent_targets_stay_absent_and_indices_are_not_targets():
    dataset = SpectroDataset("unlabelled")
    dataset.add_samples(np.arange(24).reshape(6, 4), {"partition": "train"})
    selected = extract_playground_dataset(dataset)
    assert selected.batch.y is None
    np.testing.assert_array_equal(selected.batch.sample_ids, np.arange(6))
    assert selected.evidence["sample_id_sources"] == ["dataset.index.sample"]
    with pytest.raises(ValueError, match="target_index"):
        extract_playground_dataset(dataset, target_index=1)


def test_text_headers_are_preserved_as_evidence_not_fake_wavelengths():
    dataset = SpectroDataset("text_headers")
    dataset.add_samples(np.arange(12).reshape(4, 3), {"partition": "train"}, headers=["a", "b", "c"], header_unit="text")
    selected = extract_playground_dataset(dataset)
    np.testing.assert_array_equal(selected.batch.wavelengths, [0, 1, 2])
    assert selected.batch.axis_kind == "feature_index"
    assert selected.batch.header_unit is None
    assert selected.evidence["original_headers"] == ["a", "b", "c"]
    assert selected.diagnostics == ({"code": "non_numeric_feature_axis", "policy": "feature_index"},)


def test_metadata_catalogue_is_bounded_and_ordered(multisource_dataset):
    dataset, _ = multisource_dataset
    result = playground_metadata_columns(dataset, partition="all", max_unique_values=3)
    assert [column["name"] for column in result["columns"]] == ["sample_id", "subject", "declared_partition"]
    assert result["columns"][0]["n_unique"] == 10
    assert result["columns"][0]["unique_values"] == ["S0", "S1", "S2"]
    assert result["repetition_column"] == "subject"


@pytest.mark.parametrize("kwargs", [{"partition": "other"}, {"source_index": 2}, {"target_index": 2}])
def test_invalid_dataset_selections_fail_before_preview(multisource_dataset, kwargs):
    dataset, _ = multisource_dataset
    with pytest.raises(ValueError):
        extract_playground_dataset(dataset, **kwargs)

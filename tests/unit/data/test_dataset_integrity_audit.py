"""Regression witnesses for dataset selection and storage integrity."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.metadata import Metadata
from nirs4all.data.signal_type import detect_signal_type
from nirs4all.operators.data.repetition import RepetitionConfig
from nirs4all.pipeline.config.context import DataSelector


def _dataset(augmented=False):
    dataset = SpectroDataset("integrity")
    dataset.add_samples(np.arange(18).reshape(6, 3))
    dataset.add_targets(np.arange(6) + .25)
    dataset.add_metadata(pd.DataFrame({"sid": ["a", "b", "c", "d", "e", "f"]}))
    if augmented:
        dataset.augment_samples(np.array([[30, 31, 32], [10, 11, 12]]), ["raw"], "noise", count=[0, 1, 0, 1, 0, 0])
    return dataset


@pytest.mark.parametrize("indices", [[], np.array([], dtype=np.int32)])
@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("method", ["label", "onehot"])
def test_empty_metadata_selections_never_return_all(indices, warm_cache, method):
    metadata = Metadata()
    metadata.add_metadata(pd.DataFrame({"sid": ["a", "b", "a"]}))
    if warm_cache:
        metadata.to_numeric("sid", method=method)
    assert metadata.get(indices).shape == (0, 1)
    assert metadata.get_column("sid", indices).shape == (0,)
    assert metadata.to_numeric("sid", indices, method)[0].shape[0] == 0
    assert metadata.get(None).height == 3


@pytest.mark.parametrize("augmented", [False, True])
def test_metadata_order_and_duplicates_follow_feature_target_rows(augmented):
    dataset = _dataset(augmented)
    xs = dataset.x({})
    ys = dataset.y({}).ravel()
    meta = dataset.metadata_column("sid")
    assert len(xs) == len(ys) == len(meta) == (8 if augmented else 6)
    expected = ["a", "b", "c", "d", "e", "f"] + (["b", "d"] if augmented else [])
    assert meta.tolist() == expected
    assert dataset.metadata()["sid"].to_list() == expected
    for method in ["label", "onehot"]:
        values, _ = dataset.metadata_numeric("sid", method=method)
        assert values.shape[0] == len(xs)
        values2, _ = dataset.metadata_numeric("sid", method=method)
        np.testing.assert_array_equal(values, values2)
    assert dataset.metadata({"partition": "test"}).height == 0
    assert dataset.metadata_column("sid", {"partition": "test"}).size == 0
    assert dataset.metadata_numeric("sid", {"partition": "test"})[0].size == 0
    assert dataset.metadata(include_augmented=False).height == 6


def test_metadata_raw_selection_preserves_order_and_duplicate_rows():
    metadata = Metadata()
    metadata.add_metadata(pd.DataFrame({"sid": ["a", "b", "c"]}))
    assert metadata.get([2, 0, 2])["sid"].to_list() == ["c", "a", "c"]
    assert metadata.get_column("sid", [2, 0, 2]).tolist() == ["c", "a", "c"]


@pytest.mark.parametrize("selector_type", ["typed", "dict", "nested_tags"])
def test_tag_selectors_align_all_dataset_views(selector_type):
    dataset = _dataset(True)
    dataset.add_tag("keep")
    dataset.set_tag("keep", [0, 1, 2, 3, 4, 5], [True, True, False, False, False, False])
    selectors = {"typed": DataSelector(partition="train", tag_filters={"keep": True}), "dict": {"partition": "train", "keep": True}, "nested_tags": {"partition": "train", "tag_filters": {"keep": True}}}
    selector = selectors[selector_type]
    assert dataset.x(selector).shape[0] == 3
    np.testing.assert_array_equal(dataset.y(selector).ravel(), [.25, 1.25, 1.25])
    assert dataset.metadata_column("sid", selector).tolist() == ["a", "b", "b"]
    assert dataset.metadata(selector)["sid"].to_list() == ["a", "b", "b"]
    assert dataset.metadata_numeric("sid", selector)[0].shape[0] == 3


def test_tag_updates_affect_only_selected_rows():
    dataset = _dataset()
    dataset.add_tag("keep")
    dataset.set_tag("keep", [0, 1, 2, 3, 4, 5], [True, True, True, False, False, False])
    dataset._indexer.update_by_filter({"keep": False}, {"group": 7})
    assert dataset._indexer.df["group"].to_list() == [None, None, None, 7, 7, 7]


@pytest.mark.parametrize("bad", ["width", "sources", "processed", "second_source_width", "row_counts"])
def test_failed_feature_appends_do_not_register_phantom_rows(bad):
    dataset = _dataset()
    if bad in ["second_source_width", "row_counts"]:
        dataset = SpectroDataset()
        dataset.add_samples([np.ones((6, 3)), np.ones((6, 4))])
    if bad == "processed":
        dataset.add_features([np.ones((6, 3))], ["scaled"])
    before = dataset.x({}, layout="3d", concat_source=False)
    index_before = dataset._indexer.df.clone()
    inputs = {"width": np.ones((2, 4)), "sources": [np.ones((2, 3)), np.ones((2, 3))], "processed": np.ones((2, 3)),
              "second_source_width": [np.ones((2, 3)), np.ones((2, 5))], "row_counts": [np.ones((2, 3)), np.ones((1, 4))]}
    with pytest.raises(ValueError):
        dataset.add_samples(inputs[bad])
    assert dataset._indexer.df.equals(index_before)
    after = dataset.x({}, layout="3d", concat_source=False)
    for old, new in zip(before if isinstance(before, list) else [before], after if isinstance(after, list) else [after], strict=True):
        np.testing.assert_array_equal(old, new)


def test_unequal_initial_source_rows_refused_before_mutation():
    dataset = SpectroDataset()
    with pytest.raises(ValueError, match="same sample count"):
        dataset.add_samples([np.ones((6, 3)), np.ones((5, 4))])
    assert dataset.num_samples == 0
    assert dataset._indexer.df.height == 0


def test_add_features_accepts_documented_single_ndarray():
    dataset = _dataset()
    values = dataset.x({}) * 2
    dataset.add_features(values, ["double"], source=0)
    np.testing.assert_array_equal(dataset.x({}, layout="3d")[:, 1, :], values)


@pytest.mark.parametrize("width", [2, 5])
def test_partial_processing_resize_is_refused_without_data_loss(width):
    dataset = _dataset()
    dataset.add_features([dataset.x({}) * 2], ["scaled"])
    before = dataset.x({}, layout="3d").copy()
    with pytest.raises(ValueError, match="all processings"):
        dataset.replace_features(["raw"], [np.ones((6, width))], ["cropped"], source=0)
    np.testing.assert_array_equal(dataset.x({}, layout="3d"), before)
    assert dataset.features_processings(0) == ["raw", "scaled"]


def test_signal_detection_uses_independent_sources():
    sources = [np.linspace(.3, 1.5, 24).reshape(6, 4), np.linspace(20, 80, 24).reshape(6, 4)]
    dataset = SpectroDataset()
    dataset.add_samples(sources)
    for src, values in enumerate(sources):
        assert dataset.detect_signal_type(src, force_redetect=True) == detect_signal_type(values)
    assert dataset.detect_signal_type(0)[0] != dataset.detect_signal_type(1)[0]


@pytest.mark.parametrize("mode", ["sources", "preprocessings"])
def test_repetition_reshape_preserves_coordinates_classes_and_processing_chain(mode):
    dataset = SpectroDataset()
    xs = [np.arange(24).reshape(6, 4), np.arange(12).reshape(6, 2)]
    dataset.add_samples(xs, headers=[["1000", "1010", "1020", "1030"], ["4000", "4010"]], header_unit=["nm", "cm-1"])
    dataset.add_targets(np.array(["wheat", "wheat", "corn", "corn", "oat", "oat"]))
    numeric = dataset.y({}).copy()
    scaler = StandardScaler().fit(numeric)
    dataset._targets.add_processed_targets("scaled", scaler.transform(numeric), "numeric", scaler)
    dataset.add_metadata(pd.DataFrame({"sid": ["a", "a", "b", "b", "c", "c"]}))
    if mode == "sources":
        dataset.reshape_reps_to_sources(RepetitionConfig(column="sid"))
    else:
        dataset.reshape_reps_to_preprocessings(RepetitionConfig(column="sid"))
    expected_sources = [0, 0, 1, 1] if mode == "sources" else [0, 1]
    for src, original in enumerate(expected_sources):
        assert dataset.headers(src) == (["1000", "1010", "1020", "1030"] if original == 0 else ["4000", "4010"])
        assert dataset.header_unit(src) == ("nm" if original == 0 else "cm-1")
    assert dataset.y({"y": "raw"}).ravel().tolist() == ["wheat", "corn", "oat"]
    np.testing.assert_array_equal(dataset.y({}), numeric[[0, 2, 4]])
    assert dataset._targets.transform_predictions(dataset.y({"y": "scaled"}), "scaled", "raw").ravel().tolist() == ["wheat", "corn", "oat"]


def test_mixed_partition_batch_and_augmentation_inherit_origin_labels():
    dataset = _dataset()
    dataset.add_samples_batch(np.ones((3, 1, 3)), [{"partition": "train", "origin": 0}, {"partition": "test", "origin": 1}, {"partition": "val", "origin": 2}])
    assert dataset._indexer.df["partition"].to_list()[-3:] == ["train", "test", "val"]
    indexer = dataset._indexer
    ids = indexer.augment_rows([8, 6, 7], [1, 2, 1], "noise")
    assert indexer.df.filter(pl.col("sample").is_in(ids))["partition"].to_list() == ["val", "train", "train", "test"]


def test_repetition_reshape_keeps_interleaved_partition_metadata_alignment():
    dataset = SpectroDataset()
    dataset.add_samples(np.arange(18).reshape(6, 3), {"partition": ["train", "test", "train", "train", "test", "train"]})
    dataset.add_targets(np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]))
    dataset.add_metadata(pd.DataFrame({"sid": ["a", "b", "c", "a", "b", "c"]}))
    dataset.reshape_reps_to_preprocessings(RepetitionConfig(column="sid"))
    assert dataset._indexer.df["partition"].to_list() == ["train", "test", "train"]
    assert dataset.metadata_column("sid", {"partition": "train"}).tolist() == ["a", "c"]
    assert dataset.metadata_column("sid", {"partition": "test"}).tolist() == ["b"]
    np.testing.assert_array_equal(dataset.x({"partition": "test"}, layout="3d")[0, 0], [3, 4, 5])


def test_invalid_index_metadata_does_not_mutate_features():
    dataset = _dataset()
    before = dataset.x({}).copy()
    with pytest.raises(ValueError, match="length"):
        dataset.add_samples(np.ones((2, 3)), {"group": [1]})
    np.testing.assert_array_equal(dataset.x({}), before)
    assert dataset._indexer.df.height == 6


def test_full_processing_replacement_can_change_width_and_names():
    dataset = _dataset()
    dataset.add_features([dataset.x({}) * 2, dataset.x({}) * 3], ["snv", "savgol"])
    replacement = [np.full((6, 2), value) for value in [1., 2., 3.]]
    names = ["raw_concat", "snv_concat", "savgol_concat"]
    dataset.replace_features(["raw", "snv", "savgol"], replacement, names, source=0)
    assert dataset.features_processings(0) == names
    assert dataset._indexer.df["processings"].to_list() == [names] * 6
    np.testing.assert_array_equal(dataset.x({}, layout="3d"), np.stack(replacement, axis=1))

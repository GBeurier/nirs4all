"""Regression witnesses for the October 2026 data-loading audit."""

import numpy as np
import pandas as pd
import pytest

from nirs4all.data._features.array_storage import ArrayStorage
from nirs4all.data._targets.encoders import FlexibleLabelEncoder
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.loaders.csv_loader_new import CSVLoader
from nirs4all.data.loaders.loader import handle_data, load_XY
from nirs4all.data.targets import Targets


def _csv(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


@pytest.mark.parametrize("partition", ["train", "test"])
@pytest.mark.parametrize("failure", ["missing", "na", "parse"])
def test_config_propagates_real_partition_load_failures(tmp_path, partition, failure):
    """DT2-01: a conventional filename must not turn a load error into absence."""
    path = tmp_path / f"{partition}_x_broken.csv"
    params = {}
    if failure == "na":
        path.write_text("1000;1100\n1;NA\n2;3\n")
    elif failure == "parse":
        path.write_text("1000;1100\n1;2\n")
        params["unrecognized_pandas_option"] = True
    configs = DatasetConfigs({f"{partition}_x": str(path), f"{partition}_x_params": params})
    with pytest.raises(ValueError, match="Error loading X data"):
        configs.get_dataset_at(0)


@pytest.mark.parametrize("test_features", [1, 3])
def test_train_test_feature_counts_are_validated_before_append(tmp_path, test_features):
    """DT1-01: shorter spectra must not be padded during hold-out loading."""
    train = _csv(tmp_path, "train_x.csv", "1000;1100\n1;2\n3;4\n")
    test = _csv(tmp_path, "test_x.csv", ";".join(str(i) for i in range(test_features)) + "\n" + ";".join("1" for _ in range(test_features)) + "\n")
    configs = DatasetConfigs({"train_x": train, "test_x": test})
    for _ in range(2):  # Also exercise cached data after the first rejection.
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            configs.get_dataset_at(0)


@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("features", [1, 3])
def test_array_sample_append_rejects_feature_mismatch_without_mutation(padding, features):
    storage = ArrayStorage(padding=padding)
    storage.add_samples(np.array([[1, 2]], dtype=float))
    snapshot = storage.ensure_shared().acquire()
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        storage.add_samples(np.ones((2, features)))
    np.testing.assert_array_equal(storage.array, [[[1, 2]]])
    assert snapshot.refcount == 2


def test_config_accepts_generated_and_different_inferred_headers(tmp_path):
    """Schema validation compares widths, not unspecified/inferred metadata."""
    configs = DatasetConfigs({"train_x": np.ones((3, 2)), "test_x": np.ones((2, 2))})
    assert configs.get_dataset_at(0).x({"partition": "test"}).shape == (2, 2)
    train = _csv(tmp_path, "train_x.csv", "1.1;2.2\n3;4\n")
    test = _csv(tmp_path, "test_x.csv", "5.5;6.6\n7;8\n")
    assert DatasetConfigs({"train_x": train, "test_x": test}).get_dataset_at(0).num_samples == 2


@pytest.mark.parametrize("numeric", [True, False])
def test_unlabeled_test_partition_has_no_target_rows(tmp_path, numeric):
    """DT1-06 / DT2-11: optional test_y must load and allow y selectors."""
    x = _csv(tmp_path, "train_x.csv", "1000;1100\n1;2\n3;4\n5;6\n")
    xt = _csv(tmp_path, "test_x.csv", "1000;1100\n7;8\n9;10\n")
    y = _csv(tmp_path, "train_y.csv", "label\n" + ("1.2\n2.5\n3.7\n" if numeric else "wheat\nbarley\nwheat\n"))
    configs = DatasetConfigs({"train_x": x, "train_y": y, "test_x": xt})
    for _ in range(2):
        dataset = configs.get_dataset_at(0)
        assert dataset.x({"partition": "test"}).shape == (2, 2)
        assert dataset.y({"partition": "test"}).shape == (0, 1)
        assert dataset.y({}).shape == (3, 1)
        assert dataset.y({"partition": "missing"}).shape == (0, 1)
        assert dataset.y({"partition": "test", "y": "raw"}).shape == (0, 1)


@pytest.mark.parametrize("scope", ["global_params", "train_params"])
def test_signal_type_params_load_targets_and_metadata(tmp_path, scope):
    """DT2-04: framework params must not be forwarded to pandas.read_csv."""
    config = {
        "train_x": _csv(tmp_path, "train_x.csv", "1000;1100\n1;2\n3;4\n"),
        "train_y": _csv(tmp_path, "train_y.csv", "y\n1.2\n2.5\n"),
        "train_group": _csv(tmp_path, "metadata.csv", "id\na\nb\n"),
        scope: {"signal_type": "absorbance"},
    }
    dataset = DatasetConfigs(config).get_dataset_at(0)
    np.testing.assert_allclose(dataset.y({}).ravel(), [1.2, 2.5])
    assert dataset.metadata_column("id").tolist() == ["a", "b"]
    assert dataset.signal_type().value == "absorbance"


@pytest.mark.parametrize("y_missing", [False, True])
@pytest.mark.parametrize("metadata_missing", [False, True])
def test_joint_na_row_removal_preserves_sample_identity(tmp_path, y_missing, metadata_missing):
    """DT2-02: remove the union of X/Y/metadata missing rows."""
    x = _csv(tmp_path, "X.csv", "1000;1100\n0;1\nNA;3\n4;5\n6;7\n8;9\n")
    y = _csv(tmp_path, "Y.csv", "y\n0.1\n1.1\n" + ("NA" if y_missing else "2.1") + "\n3.1\n4.1\n")
    m = _csv(tmp_path, "M.csv", "id\ns0\ns1\ns2\n" + ("NA" if metadata_missing else "s3") + "\ns4\n")
    config = {"train_x": x, "train_y": y, "train_group": m, "global_params": {"na_policy": "remove_sample"}}
    x_loaded, y_loaded, metadata, *_ = handle_data(config, "train")
    kept = [0] + ([] if y_missing else [2]) + ([] if metadata_missing else [3]) + [4]
    np.testing.assert_array_equal(x_loaded[:, 0], np.array(kept) * 2)
    np.testing.assert_allclose(y_loaded.ravel(), np.array(kept) + 0.1)
    assert metadata["id"].tolist() == [f"s{i}" for i in kept]
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert dataset.num_samples == len(kept)
    np.testing.assert_allclose(dataset.y({}).ravel(), np.array(kept) + 0.1)


def test_joint_na_row_removal_across_multiple_feature_sources(tmp_path):
    x1 = _csv(tmp_path, "X1.csv", "a;b\n0;1\nNA;3\n4;5\n6;7\n8;9\n")
    x2 = _csv(tmp_path, "X2.csv", "c\n10\n11\nNA\n13\n14\n")
    y = _csv(tmp_path, "Y.csv", "y\n0.1\n1.1\n2.1\nNA\n4.1\n")
    m = _csv(tmp_path, "M.csv", "id\ns0\ns1\ns2\ns3\ns4\n")
    config = {"train_x": [x1, x2], "train_y": y, "train_group": m, "global_params": {"na_policy": "remove_sample"}}
    xs, ys, metadata, *_ = handle_data(config, "train")
    np.testing.assert_array_equal(xs[0][:, 0], [0, 8])
    np.testing.assert_array_equal(xs[1][:, 0], [10, 14])
    np.testing.assert_allclose(ys.ravel(), [0.1, 4.1])
    assert metadata["id"].tolist() == ["s0", "s4"]
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert dataset.x({}).shape == (2, 3)


def test_original_row_mismatch_is_not_hidden_by_na_removal(tmp_path):
    x = _csv(tmp_path, "X.csv", "a\n1\nNA\n3\n")
    y = _csv(tmp_path, "Y.csv", "y\n0.1\n1.1\n")
    with pytest.raises(ValueError, match="Row count mismatch"):
        load_XY(x, None, {"na_policy": "remove_sample"}, y, None, {"na_policy": "remove_sample"})


@pytest.mark.parametrize("missing", ["NA", ""])
@pytest.mark.parametrize("policy", ["abort", "remove_sample", "ignore"])
def test_categorical_missing_labels_follow_na_policy(tmp_path, missing, policy):
    """DT2-05: a missing label is never encoded as another class."""
    path = _csv(tmp_path, "Y.csv", f"label\nwheat\nbarley\n{missing}\nwheat\n")
    result = CSVLoader().load(path, data_type="y", na_policy=policy)
    assert result.report["na_handling"]["na_detected"]
    if policy == "abort":
        assert not result.success
    else:
        assert result.success
        assert result.report["categorical_info"]["label"]["categories"] == ["wheat", "barley"]
        if policy == "remove_sample":
            assert result.data.index.tolist() == [0, 1, 3]
        else:
            assert pd.isna(result.data.iloc[2, 0])


@pytest.mark.parametrize("missing_policy", ["abort", "remove_sample", "ignore"])
def test_categorical_partition_encoding_is_shared_and_preserves_names(tmp_path, missing_policy):
    """DT2-22: reverse test class order and retain mixed numeric targets."""
    x = _csv(tmp_path, "X.csv", "1000;1100\n1;2\n3;4\n5;6\n")
    xt = _csv(tmp_path, "XT.csv", "1000;1100\n7;8\n9;10\n")
    y = _csv(tmp_path, "Y.csv", "label;amount\nwheat;1.2\nbarley;2.5\nwheat;3.7\n")
    yt = _csv(tmp_path, "YT.csv", "label;amount\nbarley;4.8\nwheat;5.9\n")
    configs = DatasetConfigs({"train_x": x, "test_x": xt, "train_y": y, "test_y": yt, "global_params": {"na_policy": missing_policy}})
    for _ in range(2):
        dataset = configs.get_dataset_at(0)
        train = dataset.y({"partition": "train"})
        test = dataset.y({"partition": "test"})
        assert train[0, 0] == test[1, 0]
        assert train[1, 0] == test[0, 0]
        assert train[0, 0] != train[1, 0]
        np.testing.assert_allclose(test[:, 1], [4.8, 5.9])
        assert dataset.y({"partition": "test", "y": "raw"})[:, 0].tolist() == ["barley", "wheat"]
        raw = dataset._targets.transform_predictions(test, "numeric", "raw")
        assert raw[:, 0].tolist() == ["barley", "wheat"]


def test_categorical_na_removal_aligns_with_feature_rows(tmp_path):
    x = _csv(tmp_path, "X.csv", "a\n0\n1\n2\n3\n")
    y = _csv(tmp_path, "Y.csv", "label\nwheat\nNA\nbarley\nwheat\n")
    dataset = DatasetConfigs({"train_x": x, "train_y": y, "global_params": {"na_policy": "remove_sample"}}).get_dataset_at(0)
    np.testing.assert_array_equal(dataset.x({}).ravel(), [0, 2, 3])
    assert dataset.y({"y": "raw"}).ravel().tolist() == ["wheat", "barley", "wheat"]
    assert dataset.num_classes == 2


def test_ignored_categorical_na_is_preserved_in_target_encoding(tmp_path):
    x = _csv(tmp_path, "X.csv", "a\n0\n1\n2\n3\n")
    y = _csv(tmp_path, "Y.csv", "label\nwheat\nNA\nbarley\nwheat\n")
    dataset = DatasetConfigs({"train_x": x, "train_y": y, "global_params": {"na_policy": "ignore"}}).get_dataset_at(0)
    assert np.isnan(dataset.y({})[1, 0])
    assert dataset.num_classes == 2


def test_target_encoder_preserves_object_missing_values():
    encoder = FlexibleLabelEncoder().fit(np.array(["wheat", None, "barley", np.nan], dtype=object))
    assert encoder.classes_.tolist() == ["barley", "wheat"]
    encoded = encoder.transform(np.array([None, "wheat", np.nan, "barley"], dtype=object))
    np.testing.assert_allclose(encoded, [np.nan, 1, np.nan, 0], equal_nan=True)


def test_empty_target_selection_does_not_return_all_targets():
    """DT1-08: this witness covers the target half of the shared finding."""
    targets = Targets()
    targets.add_targets([1.2, 2.5])
    assert targets.get_targets(indices=[]).shape == (0, 1)
    assert targets.get_targets(indices=None).shape == (2, 1)


def test_dict_fill_configuration_is_accepted_by_csv_entrypoint(tmp_path):
    """DT2-09: JSON-shaped fill settings reach the CSV NA policy."""
    path = _csv(tmp_path, "X.csv", "a;b\n1;2\nNA;4\n3;6\n")
    result = CSVLoader().load(path, na_policy="replace", na_fill_config={"method": "mean"})
    assert result.success
    assert result.data.iloc[1, 0] == 2
    config = {"train_x": path, "global_params": {"na_policy": "replace", "na_fill_config": {"method": "mean"}}}
    np.testing.assert_array_equal(DatasetConfigs(config).get_dataset_at(0).x({})[:, 0], [1, 2, 3])


def test_same_named_datasets_use_their_own_configuration_settings(tmp_path):
    """DT2-16: duplicate names must not reuse the first dataset's task."""
    x = _csv(tmp_path, "X.csv", "a\n1\n2\n3\n4\n")
    y = _csv(tmp_path, "Y.csv", "y\n0\n1\n0\n1\n")
    configs = DatasetConfigs([{"name": "same", "train_x": x, "train_y": y}, {"name": "same", "train_x": x, "train_y": y}], task_type=["regression", "binary_classification"])
    config, name = configs.configs[1]
    assert configs.get_dataset(config, name).task_type.value == "binary_classification"


@pytest.mark.parametrize("policy", ["abort", "remove_sample"])
def test_joint_loading_uses_original_positions_not_parquet_index_labels(tmp_path, policy):
    pytest.importorskip("pyarrow")
    x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]}, index=["s0", "s1", "s2", "s3"])
    y = pd.DataFrame({"y": [0.1, 1.1, 2.1, 3.1]}, index=["other0", "other1", "other2", "other3"])
    if policy == "remove_sample":
        x.iloc[1, 0] = np.nan
        y.iloc[2, 0] = np.nan
    x_path, y_path = tmp_path / "X.parquet", tmp_path / "Y.parquet"
    x.to_parquet(x_path)
    y.to_parquet(y_path)
    xs, ys, *_ = handle_data({"train_x": str(x_path), "train_y": str(y_path), "global_params": {"na_policy": policy}}, "train")
    kept = [0, 3] if policy == "remove_sample" else [0, 1, 2, 3]
    np.testing.assert_array_equal(xs.ravel(), kept)
    np.testing.assert_allclose(ys.ravel(), np.array(kept) + 0.1)

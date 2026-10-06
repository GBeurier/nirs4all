"""Regression tests for positional dataset materialization contracts."""

import json

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.loaders.csv_loader_new import CSVLoader
from nirs4all.data.loaders.loader import handle_data
from nirs4all.data.parsers.normalizer import ConfigNormalizer


@pytest.mark.parametrize("unit", ["nm", "cm-1", "index", "none", "text"])
@pytest.mark.parametrize("scope", ["root", "global", "partition", "source"])
def test_arrays_honor_units_and_parameter_precedence(unit, scope):
    config = {"train_x": np.arange(6).reshape(3, 2), "global_params": {"header_unit": "text", "signal_type": "reflectance"}}
    key = {"root": None, "global": "global_params", "partition": "train_params", "source": "train_x_params"}[scope]
    if key is None:
        config.pop("global_params")
        config["header_unit"] = unit
    else:
        config[key] = {"header_unit": unit}
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert dataset.header_unit(0) == unit
    np.testing.assert_array_equal(dataset.x({}), config["train_x"])


def test_mixed_file_and_array_sources_align_joint_na_removal(tmp_path):
    path = tmp_path / "Xcal.csv"
    path.write_text("1000;1100\n0;1\n10;11\n20;21\n30;31\n40;41\n")
    array = np.array([[0, 1], [np.nan, 11], [20, 21], [30, 31], [40, 41]])
    targets = np.array([0, 10, np.nan, 30, 40])
    metadata = pd.DataFrame({"sid": ["a", "b", "c", None, "e"]}, index=[50, 10, 80, 20, 90])
    config = {"train_x": [str(path), array], "train_y": targets, "train_group": metadata, "global_params": {"na_policy": "remove_sample"},
              "train_x_params": [{"header_unit": "nm"}, {"header_unit": "index"}]}
    xs, ys, meta, *_ = handle_data(config, "train")
    for x in xs:
        np.testing.assert_array_equal(x, [[0, 1], [40, 41]])
    np.testing.assert_array_equal(ys.ravel(), [0, 40])
    assert meta["sid"].tolist() == ["a", "e"]
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert [dataset.header_unit(i) for i in range(2)] == ["nm", "index"]
    assert dataset.metadata_column("sid").tolist() == ["a", "e"]


def test_array_partitions_share_categorical_target_vocabulary():
    config = {"train_x": np.ones((4, 2)), "train_y": np.array(["z", "a", "z", "a"]), "test_x": np.ones((2, 2)), "test_y": np.array(["a", "z"])}
    dataset = DatasetConfigs(config).get_dataset_at(0)
    np.testing.assert_array_equal(dataset.y({"partition": "train"}).ravel(), [1, 0, 1, 0])
    np.testing.assert_array_equal(dataset.y({"partition": "test"}).ravel(), [0, 1])


@pytest.mark.parametrize("role", ["train_y", "train_group"])
def test_in_memory_reference_row_counts_are_validated(role):
    config = {"train_x": np.ones((3, 2)), role: np.ones((2, 1))}
    with pytest.raises(ValueError, match="Row count mismatch"):
        DatasetConfigs(config).get_dataset_at(0)


def test_in_memory_dataframe_headers_and_indices_are_positional():
    frame = pd.DataFrame({"1000": [1, 3], "1100": [2, 4]}, index=[99, 42])
    dataset = DatasetConfigs({"train_x": frame, "train_y": np.array([5, 6]), "header_unit": "nm", "task_type": "regression"}).get_dataset_at(0)
    assert dataset.headers(0) == ["1000", "1100"]
    np.testing.assert_array_equal(dataset.y({}).ravel(), [5, 6])


def test_one_dimensional_array_is_one_feature():
    dataset = DatasetConfigs({"train_x": np.arange(3), "header_unit": "index"}).get_dataset_at(0)
    assert dataset.x({}).shape == (3, 1)


@pytest.mark.parametrize("partition", ["train", "test"])
def test_metadata_only_one_partition_keeps_sample_identity(partition):
    config = {"train_x": np.ones((3, 2)), "test_x": np.ones((2, 2)), f"{partition}_group": pd.DataFrame({"sid": ["a", "b", "c"] if partition == "train" else ["d", "e"]})}
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert dataset.metadata_column("sid", {"partition": partition}).tolist() == (["a", "b", "c"] if partition == "train" else ["d", "e"])
    absent = "test" if partition == "train" else "train"
    assert pd.isna(dataset.metadata_column("sid", {"partition": absent})).all()


@pytest.mark.parametrize("folds", [[{"train": [0], "val": [1]}], "folds.csv", {"file": "folds.csv"}])
def test_config_folds_are_explicitly_refused(folds):
    with pytest.raises(ValueError, match="Dataset-config folds are not supported"):
        DatasetConfigs({"train_x": np.ones((3, 2)), "folds": folds}).get_dataset_at(0)


def test_folder_detected_folds_are_not_silently_ignored(tmp_path):
    (tmp_path / "Xcal.csv").write_text("1000;1100\n1;2\n3;4\n")
    (tmp_path / "folds.csv").write_text("train;val\n0;1\n")
    with pytest.raises(ValueError, match="folds are not supported"):
        DatasetConfigs(str(tmp_path)).get_dataset_at(0)


@pytest.mark.parametrize("role", ["targets", "metadata"])
@pytest.mark.parametrize("option", ["columns", "link_by"])
def test_shared_reference_selectors_and_joins_are_refused(tmp_path, role, option):
    path = tmp_path / "features.csv"
    path.write_text("1000;1100\n1;2\n3;4\n")
    config = {"sources": [{"name": "NIR", "train_x": str(path)}], role: {"path": str(path), "partition": "train", option: [0] if option == "columns" else "sid"}}
    with pytest.raises(ValueError, match=f"{option}.*not supported"):
        DatasetConfigs(config).get_dataset_at(0)


@pytest.mark.parametrize("syntax", ["sources", "variations"])
def test_source_file_columns_are_not_dropped_before_refusal(syntax, tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("1000;1100\n1;2\n3;4\n")
    config = {syntax: [{"name": "NIR", "files": [{"path": str(path), "partition": "train", "columns": {"features": "0:1"}}]}]}
    with pytest.raises(ValueError, match="columns.*not supported"):
        DatasetConfigs(config).get_dataset_at(0)


def test_duplicate_shared_partition_files_are_refused():
    config = {"train_x": np.ones((3, 2)), "shared_targets": [{"path": "y1.csv", "partition": "train"}, {"path": "y2.csv", "partition": "train"}]}
    with pytest.raises(ValueError, match="Multiple (shared_targets|targets) files"):
        DatasetConfigs(config).get_dataset_at(0)


@pytest.mark.parametrize("data_type,unit", [("x", "nm"), ("x", "cm-1"), ("x", "text"), ("metadata", "nm")])
def test_decimal_comma_headers_normalize_only_spectral_features(tmp_path, data_type, unit):
    path = tmp_path / "data.csv"
    path.write_text("1000,5;1100,5\n1,5;2,5\n3,5;4,5\n")
    result = CSVLoader().load(path, decimal_separator=",", header_unit=unit, data_type=data_type)
    expected = ["1000.5", "1100.5"] if data_type == "x" and unit in {"nm", "cm-1"} else ["1000,5", "1100,5"]
    assert result.headers == expected
    np.testing.assert_array_equal(result.data.values, [[1.5, 2.5], [3.5, 4.5]])
    if data_type == "x" and unit in {"nm", "cm-1"}:
        dataset = DatasetConfigs({"train_x": str(path), "decimal_separator": ",", "header_unit": unit}).get_dataset_at(0)
        np.testing.assert_array_equal(dataset.wavelengths_nm(0) if unit == "nm" else dataset.wavelengths_cm1(0), [1000.5, 1100.5])


def test_global_encoding_does_not_reach_binary_parquet(tmp_path):
    path = tmp_path / "Xcal.parquet"
    pd.DataFrame({"1000": [1, 3], "1100": [2, 4]}).to_parquet(path)
    dataset = DatasetConfigs({"train_x": str(path), "global_params": {"encoding": "utf-8"}}).get_dataset_at(0)
    np.testing.assert_array_equal(dataset.x({}), [[1, 2], [3, 4]])


@pytest.mark.parametrize("mode", [None, "separate", "compare"])
def test_api_document_refuses_multi_variation_modes(mode):
    from nirs4all.api.dataset_documents import normalize_dataset_document
    config = {"variations": [{"name": "raw", "train_x": "raw.csv"}, {"name": "snv", "train_x": "snv.csv"}]}
    if mode:
        config["variation_mode"] = mode
    with pytest.raises(ValueError, match="multiple variations is not supported"):
        normalize_dataset_document(config)


def test_unknown_source_filename_defaults_to_training():
    config, _ = ConfigNormalizer().normalize({"sources": [{"name": "NIR", "files": ["spectra.csv"]}]})
    assert config["train_x"] == "spectra.csv"


@pytest.mark.parametrize("bad", [None, 123, object()])
def test_constructor_never_silently_skips_invalid_entries(bad):
    with pytest.raises(ValueError, match="Invalid dataset configuration"):
        DatasetConfigs([bad])


def test_generated_test_headers_preserve_training_coordinates():
    frame = pd.DataFrame({"1000": [1, 3], "1100": [2, 4]})
    dataset = DatasetConfigs({"train_x": frame, "train_x_params": {"header_unit": "nm"}, "test_x": np.array([[5, 6]])}).get_dataset_at(0)
    assert dataset.headers(0) == ["1000", "1100"]
    assert dataset.header_unit(0) == "nm"
    np.testing.assert_array_equal(dataset.x({"partition": "test"}), [[5, 6]])


@pytest.mark.parametrize("format", ["dict", "json", "yaml"])
def test_io_settings_do_not_revalidate_executable_join_schema(tmp_path, format, monkeypatch):
    import yaml

    import nirs4all.data.config as config_module
    document = {"sources": [{"id": "nir", "role": "features", "input": "nir.csv", "key": "sid"}], "task_type": "regression", "aggregate": "sid", "aggregate_exclude_outliers": False}
    inp = document
    if format != "dict":
        inp = tmp_path / f"io-spec.{format}"
        inp.write_text(json.dumps(document) if format == "json" else yaml.safe_dump(document))
    monkeypatch.setattr(config_module, "parse_config", lambda *_: pytest.fail("IO already validated its schema; root settings need no positional revalidation"))
    assert DatasetConfigs._io_config_level_settings(inp) == ("regression", "sid", None, False, "sid")


def test_io_directory_settings_do_not_scan_positional_conventions(tmp_path, monkeypatch):
    import nirs4all.data.config as config_module
    monkeypatch.setattr(config_module, "parse_config", lambda *_: pytest.fail("IO owns directory conventions"))
    assert DatasetConfigs._io_config_level_settings(str(tmp_path)) == (None, None, None, None, None)

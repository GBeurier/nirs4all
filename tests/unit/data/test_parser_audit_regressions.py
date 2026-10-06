"""Regression witnesses for DT2 parser, schema, and archive findings."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.loaders.archive_loader import EnhancedZipLoader
from nirs4all.data.loaders.base import apply_na_policy
from nirs4all.data.loaders.parquet_loader import ParquetLoader
from nirs4all.data.parsers.files_parser import FilesParser
from nirs4all.data.parsers.normalizer import ConfigNormalizer
from nirs4all.data.schema.config import DatasetConfigSchema, SourceConfig, VariationConfig, infer_partition_from_path


def _write(tmp_path, filename, content="1000;1100\n1;2\n3;4\n"):
    path = tmp_path / filename
    path.write_text(content)
    return str(path)


@pytest.mark.parametrize("syntax", ["files", "sources", "variations", "folder"])
@pytest.mark.parametrize("format", ["dict", "json", "yaml"])
def test_parser_preserves_explicit_dataset_settings(tmp_path, syntax, format):
    """DT2-03: normalization must retain grouping and other root settings."""
    path = _write(tmp_path, "Xcal.csv")
    inputs = {
        "files": {"files": [{"path": path, "partition": "train"}]},
        "sources": {"sources": [{"name": "NIR", "train_x": path}]},
        "variations": {"variations": [{"name": "raw", "train_x": path}]},
        "folder": {"folder": str(tmp_path)},
    }
    settings = {"name": "explicit_name", "task_type": "regression", "repetition": "sid", "aggregate": "sid", "aggregate_method": "median", "aggregate_exclude_outliers": False,
                "signal_type": "absorbance", "folds": [{"train": [0], "val": [1]}], "custom_provenance": {"instrument": "device"}}
    config = {**inputs[syntax], **settings}
    if format != "dict":
        config_path = tmp_path / f"dataset.{format}"
        config_path.write_text(json.dumps(config) if format == "json" else yaml.safe_dump(config))
        config = str(config_path)
    normalized, name = ConfigNormalizer().normalize(config)
    assert name == "explicit_name"
    for key, value in settings.items():
        assert normalized[key] == value


@pytest.mark.parametrize("syntax", ["sources", "concat", "select"])
@pytest.mark.parametrize("params_position", [0, 1, 2])
def test_partial_source_params_keep_path_positions(tmp_path, syntax, params_position):
    """DT2-06: absent params need a placeholder even when only one is present."""
    members = [{"name": f"s{i}", "train_x": _write(tmp_path, f"Xcal_{i}.csv"), "test_x": _write(tmp_path, f"Xtest_{i}.csv")} for i in range(3)]
    members[params_position]["params"] = {"header_unit": "nm", "signal_type": "reflectance"}
    config = {"sources": members} if syntax == "sources" else {"variations": members, "variation_mode": syntax}
    if syntax == "select":
        config["variation_select"] = ["s0", "s1", "s2"]
    normalized, _ = ConfigNormalizer().normalize(config)
    for partition in ("train", "test"):
        params = normalized[f"{partition}_x_params"]
        assert len(params) == 3
        assert params[params_position]["header_unit"] == "nm"
        assert all(param == {} for i, param in enumerate(params) if i != params_position)
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert [dataset.header_unit(i) for i in range(3)] == ["nm" if i == params_position else "cm-1" for i in range(3)]


def test_source_file_params_override_source_and_global_params(tmp_path):
    path = _write(tmp_path, "features.csv", "1000,1100\n1,2\n3,4\n")
    config = {"global_params": {"delimiter": ";", "header_unit": "cm-1"}, "sources": [{"name": "NIR", "params": {"header_unit": "nm"},
              "files": [{"path": path, "partition": "train", "params": {"delimiter": ","}}]}]}
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert dataset.x({}).shape == (2, 2)
    assert dataset.header_unit(0) == "nm"


@pytest.mark.parametrize("schema", [SourceConfig, VariationConfig])
@pytest.mark.parametrize("form", ["string", "dict"])
def test_source_partition_inference_ignores_directory_tokens(schema, form):
    """DT2-07: local/cal/val in a parent directory never assigns a partition."""
    paths = ["/tmp/local_validation/NIR_train.csv.gz", "/tmp/local_validation/NIR_test.csv.gz"]
    files = paths if form == "string" else [{"path": path} for path in paths]
    source = schema(name="NIR", files=files)
    assert source.get_train_paths() == [paths[0]]
    assert source.get_test_paths() == [paths[1]]


@pytest.mark.parametrize("filename,partition", [("chemical_test.csv", "test"), ("spectra_physical_test.csv", "test"), ("local_val.csv", "test"),
                                                    ("scaled_val.csv", "test"), ("X_values.csv", "train"), ("renewal.csv", "train"), ("Xcal.csv.gz", "train"), ("Xval.csv.zip", "test")])
def test_filename_partition_inference_matches_whole_tokens(filename, partition):
    """DT2-08: ordinary words containing cal/val/new are not partition markers."""
    assert infer_partition_from_path(filename) == partition
    assert FilesParser()._resolve_partition_from_path(filename, None) == partition


@pytest.mark.parametrize("schema", [SourceConfig, VariationConfig])
def test_ambiguous_source_filename_requires_explicit_partition(schema):
    source = schema(name="NIR", files=["Xcal_test.csv"])
    with pytest.raises(ValueError, match="Ambiguous partition"):
        source.get_train_paths()
    source = schema(name="NIR", files=[{"path": "Xcal_test.csv", "partition": "test"}])
    assert source.get_train_paths() == []
    assert source.get_test_paths() == ["Xcal_test.csv"]


@pytest.mark.parametrize("mode", [None, "separate", "compare"])
@pytest.mark.parametrize("format", ["dict", "json"])
def test_multiple_separate_variations_are_refused_instead_of_discarded(tmp_path, mode, format):
    """DT2-12: refuse unsupported multi-run expansion explicitly."""
    config = {"variations": [{"name": "raw", "train_x": "raw.csv"}, {"name": "snv", "train_x": "snv.csv"}]}
    if mode is not None:
        config["variation_mode"] = mode
    if format == "json":
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(config))
        config = str(path)
    with pytest.raises(ValueError, match="multiple variations is not supported"):
        DatasetConfigs(config)


@pytest.mark.parametrize("option,value", [("columns", {"features": [0], "targets": [1]}), ("rows", [0]), ("link_by", "sid"), ("partition", {"column": "split"})])
def test_unimplemented_files_options_are_explicitly_refused(option, value):
    """DT2-14: accepted positional files options cannot be silently ignored."""
    config = {"files": [{"path": "train.csv", option: value}]}
    with pytest.raises(ValueError, match="not supported"):
        DatasetConfigs(config)


@pytest.mark.parametrize("syntax", ["files", "sources", "variations"])
@pytest.mark.parametrize("format", ["dict", "json"])
def test_parser_errors_reach_dataset_configs(tmp_path, syntax, format):
    """DT2-17: include the offending entry rather than silently skipping it."""
    config = {syntax: [{}]}
    if format == "json":
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(config))
        config = str(path)
    with pytest.raises(ValueError, match="path|data source|files|train_x"):
        DatasetConfigs(config)


def test_missing_folder_error_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="Folder does not exist"):
        DatasetConfigs(str(tmp_path / "does_not_exist"))


def test_bare_csv_input_loads_as_a_dataset(tmp_path):
    path = _write(tmp_path, "spectra.csv")
    dataset = DatasetConfigs(path).get_dataset_at(0)
    np.testing.assert_array_equal(dataset.x({}), [[1, 2], [3, 4]])


def test_folder_validation_and_test_cohorts_are_not_feature_sources(tmp_path):
    """DT2-18: reject a three-way split until the user selects held-out inputs."""
    for filename in ("Xcal.csv", "Xval.csv", "Xtest.csv"):
        _write(tmp_path, filename)
    with pytest.raises(ValueError, match="both validation and test"):
        DatasetConfigs(str(tmp_path))
    dataset = DatasetConfigs({"train_x": str(tmp_path / "Xcal.csv"), "test_x": str(tmp_path / "Xtest.csv")}).get_dataset_at(0)
    assert dataset.num_samples == 4


@pytest.mark.parametrize("method,expected", [("mean", 2.0), ("median", 2.0), ("value", 7.0)])
def test_common_na_policy_accepts_dictionary_fill_configuration(method, expected):
    """DT2-09: every loader uses the common normalization of fill settings."""
    data = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    result, _ = apply_na_policy(data, "replace", {"method": method, "fill_value": 7})
    assert result.iloc[1, 0] == expected


def test_non_csv_fill_configuration_is_normalized(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1.0, np.nan, 3.0]}).to_parquet(path)
    result = ParquetLoader().load(path, na_policy="replace", na_fill_config={"method": "mean"})
    assert result.success
    assert result.data.iloc[1, 0] == 2


def test_zip_parquet_member_does_not_receive_text_encoding(tmp_path):
    """DT2-19: the inner binary reader must not receive archive text options."""
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"1000": [1.0, 2.0], "1100": [3.0, 4.0]})
    member = tmp_path / "data.parquet"
    frame.to_parquet(member)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(member, "data.parquet")
    result = EnhancedZipLoader().load(archive, encoding="utf-8")
    assert result.success, result.report
    pd.testing.assert_frame_equal(result.data, frame)


def test_metadata_literal_na_categories_have_explicit_parsing_controls(tmp_path):
    """DT2-10: opt out of default tokens without changing the default contract."""
    x = _write(tmp_path, "Xcal.csv")
    m = _write(tmp_path, "Mcal.csv", "treatment;country\nNone;NA\nnull;DE\n")
    config = {"sources": [{"name": "NIR", "train_x": x}], "metadata": {"path": m, "partition": "train", "params": {"keep_default_na": False, "na_values": []}}}
    dataset = DatasetConfigs(config).get_dataset_at(0)
    assert dataset.metadata_column("treatment").tolist() == ["None", "null"]
    assert dataset.metadata_column("country").tolist() == ["NA", "DE"]


def test_relation_staging_keeps_source_and_shared_target_contract(tmp_path):
    config = {"name": "relations", "sources": [{"name": "NIR", "files": [{"path": "train.csv", "partition": "train"}]}],
              "targets": {"path": "targets.csv", "link_by": "sid"}, "experimental_relation_pipeline": True,
              "repetition_spec": {"sample_id": "sid", "sources": {"NIR": 2}}}
    normalized, _ = ConfigNormalizer().normalize(config)
    assert normalized == config


def test_explicit_global_partition_overrides_string_filename(tmp_path):
    path = tmp_path / "Xtrain.csv"
    path.write_text("1;2\n3;4\n")
    config, _ = ConfigNormalizer().normalize({"files": [str(path)], "partition": "test"})
    assert config["test_x"] == str(path)
    assert "train_x" not in config


def test_single_variation_preserves_file_params(tmp_path):
    path = tmp_path / "Xtrain.csv"
    path.write_text("1;2\n3;4\n")
    config, _ = ConfigNormalizer().normalize({"variations": [{"name": "raw", "files": [{"path": str(path), "params": {"header_unit": "nm"}}]}]})
    assert config["train_x_params"]["header_unit"] == "nm"

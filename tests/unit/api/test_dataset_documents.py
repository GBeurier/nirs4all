"""Document-only boundary: real existing parsers, tiny adverse budgets."""

import json
from pathlib import Path

import pytest

from nirs4all.api.dataset_documents import DatasetDocumentLimits, normalize_dataset_document
from nirs4all.data.parsers.normalizer import ConfigNormalizer


def test_folder_only_scans_names_and_preserves_multisource_metadata(tmp_path, monkeypatch):
    for name in ("Xcal_NIR.csv", "Xcal_MIR.csv", "Ycal.csv", "Mcal.csv", "Xval_NIR.csv", "Xval_MIR.csv", "Yval.csv"):
        (tmp_path / name).write_text("deliberately not tabular data")
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("matrix contents must not be opened"))
    result = normalize_dataset_document(tmp_path)
    assert len(result["train_x"]) == 2
    assert len(result["test_x"]) == 2
    assert result["train_group"] == str(tmp_path / "Mcal.csv")
    assert "folder" not in result


@pytest.mark.parametrize("suffix, content", [
    ("json", '{"name":"explicit", "calibration_features":"X.csv", "test_y":"Y.csv"}'),
    ("yaml", 'name: explicit\ncalibration_features: X.csv\ntest_y: Y.csv\n'),
])
def test_config_aliases_and_relative_paths_are_explicit(tmp_path, suffix, content):
    path = tmp_path / f"dataset.{suffix}"
    path.write_text(content)
    result = normalize_dataset_document(path)
    assert result["train_x"] == str(tmp_path / "X.csv")
    assert result["test_y"] == str(tmp_path / "Y.csv")
    assert result["name"] == "explicit"


def test_small_benign_yaml_aliases_remain_supported(tmp_path):
    path = tmp_path / "dataset.yaml"
    path.write_text("train_x: X.csv\nglobal_params: &params {delimiter: ',', has_header: true}\ntrain_x_params: *params\n")
    result = normalize_dataset_document(path)
    assert result["global_params"] == result["train_x_params"]


@pytest.mark.parametrize("document", [
    {"sources": [{"name": "NIR", "train_x": "nir.csv"}, {"name": "MIR", "train_x": "mir.csv"}], "targets": "y.csv"},
    {"variations": [{"name": "raw", "train_x": "raw.csv"}, {"name": "snv", "train_x": "snv.csv"}], "targets": "y.csv"},
])
def test_sources_and_variations_use_existing_normalizer(document, tmp_path):
    expected, _ = ConfigNormalizer().normalize(document)
    assert expected is not None
    result = normalize_dataset_document(document, base_dir=tmp_path)
    def paths(value):
        if isinstance(value, list):
            return [paths(item) for item in value]
        return str(tmp_path / value) if isinstance(value, str) else value
    assert result["train_x"] == paths(expected["train_x"])


def test_byte_budget_precedes_config_parser(tmp_path, monkeypatch):
    path = tmp_path / "dataset.json"
    path.write_text('{"train_x":"' + "x" * 64 + '"}')
    monkeypatch.setattr(ConfigNormalizer, "_parse_json", lambda *a: pytest.fail("must fail before parsing"))
    with pytest.raises(ValueError, match="byte budget"):
        normalize_dataset_document(path, limits=DatasetDocumentLimits(max_bytes=32))


@pytest.mark.parametrize("content, limits, message", [
    ("train_x: X.csv\ncycle: &a [*a]\n", DatasetDocumentLimits(), "cycle"),
    ("train_x: X.csv\na: &a [1,2,3]\nb: [*a,*a,*a]\n", DatasetDocumentLimits(max_nodes=12), "node budget"),
    ("train_x: X.csv\na: [[[[1]]]]\n", DatasetDocumentLimits(max_depth=3), "depth budget"),
    ("train_x: X.csv\na: &a [1]\nb: [*a,*a]\n", DatasetDocumentLimits(max_aliases=1), "aliases"),
])
def test_yaml_cost_checked_before_object_construction(tmp_path, monkeypatch, content, limits, message):
    path = tmp_path / "dataset.yaml"
    path.write_text(content)
    monkeypatch.setattr(ConfigNormalizer, "_parse_yaml", lambda *a: pytest.fail("must fail before constructing objects"))
    with pytest.raises(ValueError, match=message):
        normalize_dataset_document(path, limits=limits)


def test_dictionary_cycles_nonfinite_and_expanded_strings_rejected(tmp_path):
    cycle = {"train_x": "X.csv"}
    cycle["nested"] = cycle
    with pytest.raises(ValueError, match="cycle"):
        normalize_dataset_document(cycle, base_dir=tmp_path)
    with pytest.raises(ValueError, match="finite"):
        normalize_dataset_document({"train_x": "X.csv", "value": float("nan")})
    shared = ["x" * 100]
    with pytest.raises(ValueError, match="byte budget"):
        normalize_dataset_document({"train_x": "X.csv", "repeated": [shared] * 20}, limits=DatasetDocumentLimits(max_bytes=1000))


def test_directory_budget_is_configurable_without_reading_files(tmp_path):
    tmp_path = tmp_path / "dataset"
    tmp_path.mkdir()
    for name in ("Xcal.csv", "Ycal.csv", "Mcal.csv"):
        (tmp_path / name).write_text("not data")
    with pytest.raises(ValueError, match="directory-entry"):
        normalize_dataset_document(tmp_path, limits=DatasetDocumentLimits(max_directory_entries=2))
    assert normalize_dataset_document(tmp_path, limits=DatasetDocumentLimits(max_directory_entries=3))["train_y"]


def test_plain_result_is_json_and_input_not_mutated(tmp_path):
    source = {"train_x": "X.csv", "folds": {"file": "folds.json"}}
    result = normalize_dataset_document(source, base_dir=tmp_path)
    assert source["train_x"] == "X.csv"
    assert result["folds"]["file"] == str(tmp_path / "folds.json")
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("filename", ["Xcal.csv", "Xcal_NIR.csv", "Xcal_MIR.csv", "batch_Xcal.csv", "Xcal.csv.gz", "batch-Xcal-MIR.csv.zip"])
def test_role_tokens_preserve_historical_prefix_source_and_compression(filename, tmp_path):
    (tmp_path / filename).write_text("not read")
    result = normalize_dataset_document(tmp_path)
    assert result["train_x"] == str(tmp_path / filename)
    assert result["train_group"] is None


def test_folder_alias_relative_to_explicit_base_and_config_file_sources(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    (folder / "Xcal.csv").write_text("not read")
    assert normalize_dataset_document({"directory": "dataset"}, base_dir=tmp_path)["train_x"] == str(folder / "Xcal.csv")
    document = {"sources": [{"name": "NIR", "train_x": "nir.csv"}, {"name": "MIR", "train_x": "mir.csv"}]}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(document))
    assert normalize_dataset_document(path)["train_x"] == normalize_dataset_document(document, base_dir=tmp_path)["train_x"]

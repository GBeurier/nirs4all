"""Real upload readers preserve X columns and keep display labels out of IDs."""

import numpy as np
import pandas as pd
import pytest

from nirs4all.api.dataset_inspection import load_prediction_file


@pytest.mark.parametrize("headers", [["sample", "band_a", "band_b"], ["sample", "1100", "1102"]])
def test_csv_preserves_numeric_columns_and_duplicate_display_labels(tmp_path, headers):
    path = tmp_path / "spectra.csv"
    pd.DataFrame([["same", 1.5, 2.5], ["same", 3.5, 4.5]], columns=headers).to_csv(path, sep=";", index=False)
    dataset, reader, labels = load_prediction_file(path, params={"delimiter": ";", "has_header": True})
    np.testing.assert_array_equal(dataset.x({}, layout="2d"), [[1.5, 2.5], [3.5, 4.5]])
    assert dataset.headers(0) == headers[1:]
    assert labels == ["same", "same"]
    assert not dataset.describe()["has_targets"]
    assert reader["backend"] == "nirs4all-io.native"
    assert reader["sample_labels_are_identifiers"] is False
    assert len(set(dataset.index_column("sample"))) == 2


def test_headerless_csv_retains_first_row_and_every_numeric_feature(tmp_path):
    path = tmp_path / "spectra.csv"
    path.write_text("1.5,2.5\n3.5,4.5\n")
    dataset, _, labels = load_prediction_file(path, params={"delimiter": ",", "has_header": False})
    np.testing.assert_array_equal(dataset.x({}, layout="2d"), [[1.5, 2.5], [3.5, 4.5]])
    assert labels is None


def test_excel_preserves_typed_labels_sheet_numeric_axis_and_no_targets(tmp_path):
    path = tmp_path / "spectra.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"ignored": [999]}).to_excel(writer, sheet_name="Other", index=False)
        pd.DataFrame({"sample": ["b", "a"], 1100: [1.5, 3.5], 1102: [2.5, 4.5]}).to_excel(writer, sheet_name="NIR", index=False)
    dataset, reader, labels = load_prediction_file(path, params={"sheet_name": "NIR", "has_header": True})
    np.testing.assert_array_equal(dataset.x({}, layout="2d"), [[1.5, 2.5], [3.5, 4.5]])
    assert dataset.headers(0) == ["1100", "1102"]
    assert labels == ["b", "a"]
    assert reader["native_load_limits_applied"] is False
    assert not dataset.describe()["has_targets"]


def test_limits_fail_before_read_and_native_failure_never_retries_excel(tmp_path, monkeypatch):
    from nirs4all.data.loaders.excel_loader import ExcelLoader
    monkeypatch.setattr(ExcelLoader, "load", lambda *a, **k: pytest.fail("not an Excel route"))
    path = tmp_path / "spectra.csv"
    path.write_text("a;b\n1;2\n3;4\n")
    with pytest.raises(ValueError, match="raw input byte"):
        load_prediction_file(path, max_input_bytes=1)
    with pytest.raises(ValueError, match="budget|limit"):
        load_prediction_file(path, params={"has_header": True}, load_limits={"max_cells": 2})
    excel = tmp_path / "spectra.xlsx"
    excel.write_bytes(b"not decoded")
    with pytest.raises(ValueError, match="cannot be guaranteed"):
        load_prediction_file(excel, load_limits={"max_cells": 2})


def test_parquet_keeps_all_numeric_features_and_first_text_column_as_labels(tmp_path):
    path = tmp_path / "spectra.parquet"
    pd.DataFrame({"1100": [1.5, 3.5], "subject": ["b", "a"],
                  "1102": [2.5, 4.5], "batch": ["x", "y"]}).to_parquet(path, index=False)
    dataset, reader, labels = load_prediction_file(path)
    np.testing.assert_array_equal(dataset.x({}, layout="2d"), [[1.5, 2.5], [3.5, 4.5]])
    assert labels == ["b", "a"]
    assert dataset.metadata_columns == ["subject", "batch"]
    assert reader["native_load_limits_applied"] is True


def test_headerless_excel_preserves_first_row(tmp_path):
    path = tmp_path / "spectra.xlsx"
    pd.DataFrame([["a", 1.5, 2.5], ["b", 3.5, 4.5]]).to_excel(path, index=False, header=False)
    dataset, _, labels = load_prediction_file(path, params={"has_header": False})
    np.testing.assert_array_equal(dataset.x({}, layout="2d"), [[1.5, 2.5], [3.5, 4.5]])
    assert labels == ["a", "b"]

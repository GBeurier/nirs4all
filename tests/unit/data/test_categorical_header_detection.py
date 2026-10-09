"""Target header detection must distinguish labels from column names."""

import pytest

from nirs4all.data.detection import detect_file_parameters


@pytest.mark.parametrize("header", ["CoffeeType", "class", "label", "target"])
def test_categorical_target_header_keeps_data_row_count(tmp_path, header):
    path = tmp_path / "Ytrain.csv"
    path.write_text(f"{header}\nTauro\nTauro\nRenzo\nTorino\n", encoding="utf-8")
    result = detect_file_parameters(path)
    assert result.has_header is True
    assert result.n_rows - int(result.has_header) == 4
    assert result.n_columns == 1


def test_headerless_class_labels_keep_first_observation(tmp_path):
    path = tmp_path / "Ytrain.csv"
    path.write_text("Tauro\nTauro\nRenzo\nTorino\n", encoding="utf-8")
    result = detect_file_parameters(path)
    assert result.has_header is False
    assert result.n_rows == 4

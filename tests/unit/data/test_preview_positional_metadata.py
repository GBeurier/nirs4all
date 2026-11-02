"""Native previews preserve row order and never require identifier alignment."""

import numpy as np
import pytest

from nirs4all.api.dataset_inspection import load_dataset_for_analysis, preview_dataset


@pytest.mark.parametrize("with_row_id", [False, True])
def test_native_preview_keeps_positional_metadata_without_forcing_alignment(tmp_path, with_row_id):
    pytest.importorskip("nirs4all_io._native")
    (tmp_path / "Xtrain.csv").write_text("1000;1001\n1;2\n3;4\n5;6\n")
    (tmp_path / "Ytrain.csv").write_text("target\n10.1\n20.2\n30.3\n")
    metadata = "row_id;batch\n90;c\n10;a\n50;b\n" if with_row_id else "batch\nc\na\nb\n"
    (tmp_path / "Mtrain.csv").write_text(metadata)
    config = {
        "train_x": str(tmp_path / "Xtrain.csv"),
        "train_y": str(tmp_path / "Ytrain.csv"),
        "train_group": str(tmp_path / "Mtrain.csv"),
        "global_params": {"delimiter": ";", "has_header": True},
    }
    dataset, _ = load_dataset_for_analysis(config)
    np.testing.assert_array_equal(dataset.x({}), [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(dataset.y({}).ravel(), [10.1, 20.2, 30.3])
    assert dataset.metadata_column("batch").tolist() == ["c", "a", "b"]
    assert dataset.metadata_columns == (["row_id", "batch"] if with_row_id else ["batch"])
    if with_row_id:
        assert dataset.metadata_column("row_id").tolist() == [90, 10, 50]
    preview = preview_dataset(config)
    assert preview["success"] is True
    assert preview["summary"]["metadata_columns"] == dataset.metadata_columns

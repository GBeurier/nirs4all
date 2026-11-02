"""User metadata must never collide with the private positional index."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nirs4all.data.metadata import Metadata


@pytest.mark.parametrize("kind", ["pandas", "polars", "numpy"])
def test_user_row_id_preserves_values_and_positional_selection(kind):
    frame = pd.DataFrame({"row_id": [30, 10, 20], "batch": [3, 1, 2]})
    data = frame if kind == "pandas" else pl.from_pandas(frame) if kind == "polars" else frame.to_numpy()
    metadata = Metadata()
    metadata.add_metadata(data, headers=list(frame.columns))

    assert metadata.columns == ["row_id", "batch"]
    assert metadata.get_column("row_id").tolist() == [30, 10, 20]
    assert metadata.get([2, 0, 2], columns=["row_id"])["row_id"].to_list() == [20, 30, 20]
    metadata.update_metadata([1], "row_id", [99])
    assert metadata.get_column("row_id").tolist() == [30, 99, 20]
    metadata.to_numeric("batch")
    assert metadata.to_numeric("batch", [2, 0, 2])[0].tolist() == [2, 3, 2]
    restored = pickle.loads(pickle.dumps(metadata))
    assert restored.get().to_dicts() == metadata.get().to_dicts()


def test_later_partition_can_introduce_internal_index_names():
    metadata = Metadata()
    metadata.add_metadata(pd.DataFrame({"batch": ["train"]}))
    metadata.add_metadata(pd.DataFrame({"row_id": [90], "__nirs4all_row_id": [91], "batch": ["test"]}))
    metadata.add_metadata(pd.DataFrame({metadata.row_id_column: [92], "batch": ["predict"]}))

    assert metadata.get_column("batch").tolist() == ["train", "test", "predict"]
    assert metadata.get()["row_id"].to_list() == [None, 90, None]
    assert metadata.get()["__nirs4all_row_id"].to_list() == [None, 91, None]
    assert metadata.get([1])["batch"].to_list() == ["test"]


def test_add_user_row_id_column_and_legacy_pickle_state():
    metadata = Metadata()
    metadata.add_metadata(pd.DataFrame({"batch": ["a", "b"]}))
    assert metadata.row_id_column == "row_id"
    assert "_row_id_column" not in vars(metadata)
    assert pickle.loads(pickle.dumps(metadata)).get_column("batch").tolist() == ["a", "b"]
    metadata.add_column("row_id", [80, 40])
    np.testing.assert_array_equal(metadata.get_column("row_id", [1, 0]), [40, 80])
    with pytest.raises(ValueError, match="already exists"):
        metadata.add_column("row_id", [0, 1])

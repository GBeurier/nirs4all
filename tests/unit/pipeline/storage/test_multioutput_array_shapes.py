"""Target axes are persisted explicitly without guessing shapes in old files."""
import numpy as np
import pyarrow.parquet as pq
import pytest

from nirs4all.data import Predictions
from nirs4all.pipeline.storage.array_store import ArrayStore


@pytest.mark.parametrize("task", ["regression", "binary_classification"])
def test_multioutput_append_compact_and_portable_roundtrip(tmp_path, task):
    store = ArrayStore(tmp_path)
    values = np.arange(12).reshape(6, 2).astype(float)
    if task == "binary_classification":
        values = (values % 3 == 0).astype(float)
    records = [{"prediction_id": f"p{i}", "dataset_name": "targets", "model_name": "model",
                "partition": partition, "fold_id": "0", "task_type": task,
                "metric": "rmse" if task == "regression" else "balanced_accuracy",
                "y_true": values, "y_pred": values + (i if task == "regression" else 0),
                "sample_indices": np.arange(6)}
               for i, partition in enumerate(["train", "val", "test"])]
    store.save_batch(records[:1])
    store.save_batch(records[1:])
    updated = {**records[1], "y_pred": values.copy()}
    store.save_batch([updated])
    store.delete_batch({"p0"}, dataset_name="targets")
    store.compact("targets")
    arrays = store.load_batch(["p1", "p2"], dataset_name="targets")
    assert set(arrays) == {"p1", "p2"}
    np.testing.assert_array_equal(arrays["p1"]["y_pred"], values)
    for record in arrays.values():
        assert record["y_pred"].shape == record["y_true"].shape == (6, 2)
        assert record["sample_indices"].shape == (6,)
    path = next((tmp_path / "arrays").glob("*.parquet"))
    portable = Predictions.from_file(path)
    entries = list(portable.iter_entries())
    assert len(entries) == 2
    for entry in entries:
        assert entry["y_true"].shape == entry["y_pred"].shape == (6, 2)


def test_old_flat_targets_remain_flat_after_append_and_compact(tmp_path):
    store = ArrayStore(tmp_path)
    base = {"prediction_id": "old", "dataset_name": "targets", "y_true": np.arange(12), "y_pred": np.arange(12)}
    store.save_batch([base])
    path = next((tmp_path / "arrays").glob("*.parquet"))
    old_table = pq.read_table(path).drop(["y_true_shape", "y_pred_shape"])
    pq.write_table(old_table, path)
    assert store.load_single("old")["y_true"].shape == (12,)
    store.save_batch([{**base, "prediction_id": "new", "y_true": np.arange(12).reshape(6, 2), "y_pred": np.arange(12).reshape(6, 2)}])
    store.compact("targets")
    arrays = store.load_batch(["old", "new"])
    assert arrays["old"]["y_true"].shape == arrays["old"]["y_pred"].shape == (12,)
    assert arrays["new"]["y_true"].shape == arrays["new"]["y_pred"].shape == (6, 2)
    entries = list(Predictions.from_file(path).iter_entries())
    assert sorted(entry["y_true"].shape for entry in entries) == [(6, 2), (12,)]
    assert sorted(entry["y_pred"].shape for entry in entries) == [(6, 2), (12,)]

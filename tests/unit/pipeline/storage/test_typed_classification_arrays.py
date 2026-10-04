"""Physical typed-label persistence, compatibility and portable reconstruction."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from nirs4all.data._predictions.target_codec import decode_target_array
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.array_store import ArrayStore
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _record(identifier, labels, *, task_type="binary_classification"):
    return {"prediction_id": identifier, "dataset_name": "cohort", "model_name": "PLSLogistic", "fold_id": "fold0",
            "partition": "test", "metric": "accuracy", "val_score": 1.0, "task_type": task_type,
            "y_true": labels, "y_pred": labels[::-1], "y_proba": np.tile([[0.25, 0.75]], (len(labels), 1)),
            "sample_indices": np.arange(len(labels), dtype=np.int32),
            "result_metadata": {"class_names": ["signed class column 0", "signed class column 1"]}}


@pytest.mark.parametrize("labels", [
    np.array(["blue", "red", "blue", "astral \U0001f680"]),
    np.array([-(1 << 63), (1 << 63) - 1, (1 << 53) + 1, -7], dtype=np.int64),
    np.array([["blue", "red"], ["red", "blue"]]),
])
def test_physical_label_roundtrip_reopen_compact_and_portable_export(tmp_path, labels):
    store = ArrayStore(tmp_path / "workspace")
    record = _record("typed", labels)
    store.save_batch([record])
    path = store.arrays_dir / "cohort.parquet"
    row = pq.read_table(path).to_pylist()[0]
    for field in ("y_true", "y_pred"):
        assert row[field] is None  # No floating-point placeholder for labels.
        payload = json.loads(row[field + "_labels"])
        assert payload["schema_version"] == 1
        assert payload["vocabulary"] == sorted(set(record[field].ravel().tolist()))
        assert all(type(index) is int for index in payload["indices"])
    np.testing.assert_array_equal(row["y_proba"], record["y_proba"].ravel())
    assert json.loads(row["result_metadata"]) == record["result_metadata"]
    reopened = ArrayStore(tmp_path / "workspace")
    reopened.compact("cohort")
    loaded = reopened.load_single("typed", "cohort")
    assert loaded is not None
    for field in ("y_true", "y_pred", "y_proba"):
        np.testing.assert_array_equal(loaded[field], record[field])
    assert loaded["result_metadata"] == record["result_metadata"]
    exported = tmp_path / "exported.parquet"
    reopened.load_dataset("cohort").write_parquet(exported)
    portable = Predictions.from_parquet(exported).filter_predictions(load_arrays=True)
    assert len(portable) == 1
    for field in ("y_true", "y_pred", "y_proba"):
        np.testing.assert_array_equal(portable[0][field], record[field])
    assert portable[0]["result_metadata"] == record["result_metadata"]


def test_historical_numeric_schema_append_typed_then_numeric_keeps_every_array(tmp_path):
    store = ArrayStore(tmp_path / "workspace")
    first = _record("old", np.array([0.125, -3.5]), task_type="regression")
    store.save_batch([first])
    path = store.arrays_dir / "cohort.parquet"
    original = pq.read_table(path)
    assert "y_true_labels" not in original.schema.names
    typed = _record("typed", np.array(["blue", "red"]))
    last = _record("new", np.array([1.5, 2.25]), task_type="regression")
    store.save_batch([typed])
    store.save_batch([last])
    store.compact("cohort")
    for record in (first, typed, last):
        loaded = store.load_single(record["prediction_id"], "cohort")
        assert loaded is not None
        for field in ("y_true", "y_pred", "y_proba", "sample_indices"):
            np.testing.assert_array_equal(loaded[field], record[field])
    current = pq.read_table(path).to_pylist()
    old = next(row for row in current if row["prediction_id"] == "old")
    assert all(old[name] == original.to_pylist()[0][name] for name in original.schema.names)
    assert old["y_true_labels"] is None


def test_typed_column_vector_retains_historical_flat_mono_target_read(tmp_path):
    store = ArrayStore(tmp_path / "workspace")
    labels = np.array([["blue"], ["red"]])
    store.save_batch([_record("column", labels)])
    loaded = store.load_single("column", "cohort")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["y_true"], labels.ravel())
    np.testing.assert_array_equal(loaded["y_pred"], labels[::-1].ravel())


def test_new_regression_files_keep_historical_schema_after_typed_file_in_another_dataset(tmp_path):
    store = ArrayStore(tmp_path / "workspace")
    store.save_batch([_record("typed", np.array(["blue", "red"]))])
    numeric = _record("regression", np.array([0.5, 1.25]), task_type="regression")
    numeric["dataset_name"] = "numeric"
    store.save_batch([numeric])
    schema = pq.read_schema(store.arrays_dir / "numeric.parquet")
    assert "y_true_labels" not in schema.names
    assert "y_pred_labels" not in schema.names


@pytest.mark.parametrize("conflict", ["numeric", "regression", "shape"])
def test_typed_targets_refuse_ambiguous_numeric_task_or_shape(conflict):
    payload = {"schema_version": 1, "label_type": "str", "vocabulary": ["blue", "red"], "indices": [0, 1], "shape": [2]}
    row = {"task_type": "binary_classification", "y_true": None, "y_true_labels": json.dumps(payload)}
    if conflict == "numeric":
        row["y_true"] = [0.0, 1.0]
    elif conflict == "regression":
        row["task_type"] = "regression"
    else:
        row["y_true_shape"] = [1, 2]
    with pytest.raises(ValueError, match="classification"):
        decode_target_array(row, "y_true")


@pytest.mark.parametrize("labels", [
    np.array(["blue", 4], dtype=object),
    np.array([True, False]),
    np.array([1 << 63], dtype=np.uint64),
    np.array([b"blue"]),
])
def test_invalid_label_types_refuse_without_replacing_existing_parquet(tmp_path, labels):
    store = ArrayStore(tmp_path / "workspace")
    store.save_batch([_record("old", np.array(["blue", "red"]))])
    path = store.arrays_dir / "cohort.parquet"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="classification"):
        store.save_batch([_record("bad", labels)])
    assert path.read_bytes() == before


def test_invalid_later_dataset_does_not_publish_first_dataset_or_clear_tombstones(tmp_path):
    store = ArrayStore(tmp_path / "workspace")
    valid = _record("old", np.array(["blue", "red"]))
    store.save_batch([valid])
    store.delete_batch({"old"})
    before = (store.arrays_dir / "cohort.parquet").read_bytes()
    invalid = _record("bad", np.array([True, False]))
    invalid["dataset_name"] = "later"
    with pytest.raises(ValueError, match="classification"):
        store.save_batch([valid, invalid])
    assert store.pending_tombstone_count() == 1
    assert (store.arrays_dir / "cohort.parquet").read_bytes() == before
    assert not (store.arrays_dir / "later.parquet").exists()


@pytest.mark.parametrize("patch", [
    {"schema_version": 2}, {"schema_version": True}, {"label_type": "float"}, {"label_type": []},
    {"vocabulary": ["blue", "blue"]}, {"vocabulary": ["red", "blue"]}, {"vocabulary": ["blue", 2]},
    {"indices": [0, 2]}, {"indices": [True, 0]}, {"shape": [3]}, {"shape": [16_777_217]}, {"shape": [True]},
])
def test_corrupt_label_codec_refuses_instead_of_numeric_fallback(patch):
    payload = {"schema_version": 1, "label_type": "str", "vocabulary": ["blue", "red"], "indices": [0, 1], "shape": [2]}
    payload.update(patch)
    row = {"task_type": "binary_classification", "y_true": None, "y_true_labels": json.dumps(payload)}
    with pytest.raises(ValueError, match="classification"):
        decode_target_array(row, "y_true")


def test_physical_corrupt_codec_is_rejected_by_workspace_and_portable_readers(tmp_path):
    import pyarrow as pa

    store = ArrayStore(tmp_path / "workspace")
    store.save_batch([_record("typed", np.array(["blue", "red"]))])
    path = store.arrays_dir / "cohort.parquet"
    table = pq.read_table(path)
    row = table.to_pylist()[0]
    payload = json.loads(row["y_true_labels"])
    payload["indices"][0] = 9
    row["y_true_labels"] = json.dumps(payload)
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), path)
    with pytest.raises(ValueError, match="classification label index"):
        store.load_single("typed", "cohort")
    with pytest.raises(ValueError, match="classification label index"):
        Predictions.from_parquet(path)


@pytest.mark.parametrize("labels", [np.array(["blue", "astral \U0001f680"]), np.array([(1 << 53) + 1, -7], dtype=np.int64)])
def test_real_prediction_flush_workspace_reload_and_portable_export_preserve_labels(tmp_path, labels):
    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)
    try:
        run_id = store.begin_run("typed", config={"metric": "accuracy"}, datasets=[{"name": "cohort"}])
        pipeline_id = store.begin_pipeline(run_id=run_id, name="typed", expanded_config=[{"model": "PLSLogistic"}],
            generator_choices=[], dataset_name="cohort", dataset_hash="source-owned")
        chain_id = store.save_chain(pipeline_id=pipeline_id, steps=[{"step_idx": 0, "operator_class": "PLSLogistic", "params": {},
            "artifact_id": None, "stateless": False}], model_step_idx=0, model_class="n4m.roles.PLSLogistic", preprocessings="",
            fold_strategy="per_fold", fold_artifacts={}, shared_artifacts={})
        predictions = Predictions(store=store)
        proba = np.array([[0.75, 0.25], [0.125, 0.875]])
        metadata = {"class_names": labels.tolist(), "signed_column_order": True}
        predictions.add_prediction(dataset_name="cohort", model_name="PLSLogistic", model_classname="PLSLogistic", partition="test",
            fold_id=0, task_type="binary_classification", metric="accuracy", val_score=1.0,
            y_true=labels, y_pred=labels, y_proba=proba, sample_indices=np.array([5, 9]), result_metadata=metadata)
        predictions.flush(pipeline_id=pipeline_id, chain_id=chain_id)
        exported = tmp_path / "portable.parquet"
        store.array_store.load_dataset("cohort").write_parquet(exported)
    finally:
        store.close()
    for predictions in (Predictions.from_workspace(workspace), Predictions.from_parquet(exported)):
        rows = predictions.filter_predictions(load_arrays=True)
        assert len(rows) == 1
        for field in ("y_true", "y_pred"):
            np.testing.assert_array_equal(rows[0][field], labels)
        np.testing.assert_array_equal(rows[0]["y_proba"], proba)
        assert rows[0]["result_metadata"] == metadata

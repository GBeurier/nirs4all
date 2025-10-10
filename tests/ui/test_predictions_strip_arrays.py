import numpy as np
from pathlib import Path
from nirs4all.dataset.predictions import Predictions
from nirs4all.ui import predictions_api


def _make_sample_predictions_file(tmp_path: Path) -> Path:
    # create folder and predictions.json
    ds_dir = tmp_path / "results" / "dataset1"
    ds_dir.mkdir(parents=True)
    preds_file = ds_dir / "predictions.json"

    preds = Predictions()
    # add a single prediction with arrays
    preds.add_prediction(
        dataset_name="dataset1",
        dataset_path=str(ds_dir),
        config_name="cfg",
        config_path=str(ds_dir),
        model_name="mymodel",
        y_true=np.array([0.1, 0.2, 0.3]),
        y_pred=np.array([0.11, 0.19, 0.29]),
        sample_indices=[0, 1, 2]
    )
    preds.save_to_file(str(preds_file))
    return ds_dir


def test_predictions_list_default_strips(tmp_path: Path):
    ds_dir = _make_sample_predictions_file(tmp_path)

    res = predictions_api.predictions_list(dataset=str(ds_dir))
    assert res["count"] == 1
    row = res["predictions"][0]
    # arrays should be stripped by default
    assert "y_true" not in row
    assert "y_pred" not in row
    assert "sample_indices" not in row
    assert row.get("_arrays_stripped") is True
    # columns should not contain heavy fields
    assert "y_true" not in res["columns"]


def test_predictions_list_include_arrays_true(tmp_path: Path):
    ds_dir = _make_sample_predictions_file(tmp_path)

    res = predictions_api.predictions_list(dataset=str(ds_dir), include_arrays=True)
    assert res["count"] == 1
    row = res["predictions"][0]
    # arrays should be present when requested
    assert "y_true" in row and isinstance(row["y_true"], list)
    assert "y_pred" in row and isinstance(row["y_pred"], list)
    # marker should be absent when arrays included
    assert row.get("_arrays_stripped") is None


def test_predictions_search_respects_include_arrays(tmp_path: Path):
    ds_dir = _make_sample_predictions_file(tmp_path)

    res = predictions_api.predictions_search(dataset=str(ds_dir), include_arrays=False)
    assert res["total"] == 1
    pred = res["predictions"][0]
    assert "y_true" not in pred

    res2 = predictions_api.predictions_search(dataset=str(ds_dir), include_arrays=True)
    assert res2["total"] == 1
    pred2 = res2["predictions"][0]
    assert "y_true" in pred2

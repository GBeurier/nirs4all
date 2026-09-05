"""A real no-CV/no-test run retains only its attested training score in Store v5."""

import json
import sqlite3
from importlib.resources import files

import numpy as np
from sklearn.linear_model import Ridge

import nirs4all
from nirs4all.data import SpectroDataset


def test_full_train_summary_extension_retains_owner_scores_without_test_alias(tmp_path):
    contract = json.loads(files("nirs4all.pipeline.storage").joinpath("contracts/workspace_store_results_summary_v1.json").read_text())
    assert contract["selection"]["refit_only_candidates"] == "cv_fold_count_is_zero_and_finite_final_test_score_is_not_null"
    extension = contract["extensions"]["full_train_only_v1"]
    assert extension["candidate"] == "meaningful_final_and_not_has_cv_payload_before_synthesis"
    assert extension["preserve_scores"] is True and extension["synthetic_refit"] is False
    dataset = SpectroDataset("train-only")
    dataset.set_task_type("regression")
    x = np.random.default_rng(42).normal(size=(24, 4))
    dataset.add_samples(x, {"partition": "train"})
    dataset.add_targets(x[:, 0] * 1.37 + 2.1)
    result = nirs4all.run([Ridge()], dataset, engine="dag-ml", workspace_path=tmp_path, save_charts=False, verbose=0)
    assert np.isnan(result.cv_best_score)
    with sqlite3.connect(tmp_path / "store.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(contract["source_projection"]["page_query"], (500, 0))]
        assert len(rows) == 1
        row = rows[0]
        assert row["cv_val_score"] is None and row["cv_fold_count"] == 0
        assert row["final_test_score"] is None
        assert np.isfinite(row["final_train_score"])
        scores = json.loads(row["final_scores"])
        assert "train" in scores and "test" not in scores and "val" not in scores
        assert json.loads(row["cv_scores"] or "{}") == {}

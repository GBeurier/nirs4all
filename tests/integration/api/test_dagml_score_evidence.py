"""Displayed native scores must describe the estimator and samples in their row."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler


def test_native_partition_scores_are_real_and_oof_averages_unique_samples(tmp_path):
    import nirs4all
    from scripts.bench_engine_perf import _canonical_oof_rmse

    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 12))
    y = X @ rng.normal(size=12) + rng.normal(scale=0.2, size=80)
    result = nirs4all.run(
        [StandardScaler(), ShuffleSplit(n_splits=3, test_size=0.4, random_state=17), Ridge()],
        (X, y, {"train": 64}), engine="dag-ml", allow_fallback=False,
        workspace_path=tmp_path, verbose=0,
    )
    try:
        rows = result.predictions.filter_predictions(load_arrays=True)
        cv_rows_all = [row for row in rows if row["fold_id"] != "final"]
        assert {(row["fold_id"], row["partition"]) for row in cv_rows_all} == {
            ("0", "train"), ("1", "train"), ("2", "train"), ("avg", "train"), ("w_avg", "train"),
            ("0", "val"), ("1", "val"), ("2", "val"), ("avg", "val"),
            ("w_avg", "val"),
            ("0", "test"), ("1", "test"), ("2", "test"), ("avg", "test"), ("w_avg", "test"),
        }
        cv_rows = [row for row in cv_rows_all if row["partition"] == "val"]
        for row in cv_rows:
            assert row["train_score"] is not None and row["test_score"] is not None
            assert set(row["scores"]) == {"train", "val", "test"}
            provenance = row["result_metadata"]["dagml_projection"]
            assert provenance["unavailable_partitions"] == []
            assert provenance["score_provenance"]["val"]["purpose"] == "measurement"
            if "test" in provenance["score_provenance"]:
                assert provenance["score_provenance"]["test"]["purpose"] == "measurement"
        for row in rows:
            observed = np.sqrt(np.mean((np.asarray(row["y_true"]) - np.asarray(row["y_pred"])) ** 2))
            assert row[f"{row['partition']}_score"] == pytest.approx(observed, abs=1e-5)
        avg = next(row for row in cv_rows if row["fold_id"] == "avg")
        fold_rows = [row for row in cv_rows if row["fold_id"] in {"0", "1", "2"}]
        assert len(avg["sample_indices"]) < sum(len(row["sample_indices"]) for row in fold_rows)
        assert len(avg["sample_indices"]) == len(set(avg["sample_indices"]))
        assert result.cv_best_score == pytest.approx(_canonical_oof_rmse(fold_rows), abs=1e-5)
        for row in rows:
            if row["fold_id"] == "final":
                assert row["val_score"] == result.cv_best_score
                provenance = row["result_metadata"]["dagml_projection"]["score_provenance"]
                assert provenance["val"]["purpose"] == "model_selection"
                assert provenance["val"]["fold_id"] == "avg"
                assert provenance["train"]["partition"] == "final"
                assert provenance["test"]["partition"] == "test"
        assert result.best["fold_id"] == "final"
        assert result.best_score == result.best["test_score"]
    finally:
        result.close()


def test_default_splitter_keeps_actual_refit_measurements(tmp_path):
    """A default-parameter splitter must not reproduce legacy's skipped-refit bug."""
    import nirs4all

    X = np.random.default_rng(11).normal(size=(40, 6))
    y = X[:, 0] + 0.4 * X[:, 1]
    oracle = Ridge().fit(X[:32], y[:32])
    with nirs4all.run([KFold(), Ridge()], (X, y, {"train": 32}), engine="dag-ml", allow_fallback=False,
                      workspace_path=tmp_path, verbose=0) as result:
        final = result.predictions.filter_predictions(fold_id="final", load_arrays=True)
        assert {row["partition"] for row in final} == {"train", "test"}
        test = next(row for row in final if row["partition"] == "test")
        np.testing.assert_allclose(test["y_pred"].ravel(), oracle.predict(X[32:]), atol=1e-5)
        expected_rmse = np.sqrt(np.mean((y[32:] - oracle.predict(X[32:])) ** 2))
        assert result.best_rmse == pytest.approx(expected_rmse, abs=1e-5)
        assert result.final is not None

"""Public CV training-prediction parity, including fold-model ensembles."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

import nirs4all


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_cv_train_predictions_match_legacy_fold_and_ensemble_surface(monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    x, y = make_regression(n_samples=36, n_features=6, noise=0.1, random_state=42)
    pipeline = [KFold(3, shuffle=True, random_state=42), {"model": Ridge(alpha=1)}]
    common = {"refit": False, "save_artifacts": False, "save_charts": False, "verbose": 0}
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", **common)
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, **common)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)

        def rows(result):
            table = result.predictions.to_dataframe().to_dicts()
            return {
                (row["fold_id"], row["partition"]): result.predictions.get_prediction_by_id(row["id"])
                for row in table
            }

        expected, actual = rows(legacy), rows(native)
        assert actual.keys() == expected.keys()
        for key in expected:
            left, right = expected[key], actual[key]
            assert set(left["sample_indices"]) == set(right["sample_indices"])
            assert right["n_samples"] == len(right["sample_indices"])
            left_by_id = dict(zip(left["sample_indices"], np.asarray(left["y_pred"]).reshape(-1), strict=True))
            right_by_id = dict(zip(right["sample_indices"], np.asarray(right["y_pred"]).reshape(-1), strict=True))
            assert np.asarray([right_by_id[index] for index in left_by_id]) == pytest.approx(
                np.asarray(list(left_by_id.values())), abs=1e-5, rel=1e-7
            )
            for partition in ("train", "val"):
                if left[f"{partition}_score"] is not None:
                    assert right[f"{partition}_score"] == pytest.approx(left[f"{partition}_score"], abs=1e-5)
        assert {key for key in actual if key[1] == "train"} == {
            ("0", "train"), ("1", "train"), ("2", "train"), ("avg", "train"), ("w_avg", "train")
        }
        assert not any(report["partition"] == "train_pool" and report.get("fold_id") in {"avg", "w_avg"}
                       for report in native._dagml_score_set["reports"])
    finally:
        legacy.close()
        native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_cv_classification_train_ensemble_uses_class_votes(monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    x, y = make_classification(n_samples=36, n_features=6, n_informative=3, n_redundant=0, random_state=42)
    pipeline = [StratifiedKFold(3, shuffle=True, random_state=42), {"model": LogisticRegression(max_iter=200)}]
    common = {"refit": False, "save_artifacts": False, "save_charts": False, "verbose": 0}
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", **common)
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, **common)
    try:
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score)
        for fold in ("avg", "w_avg"):
            def row(result):
                table = result.predictions.to_dataframe().to_dicts()
                record = next(record for record in table if record["fold_id"] == fold and record["partition"] == "train")
                return result.predictions.get_prediction_by_id(record["id"])

            expected, actual = row(legacy), row(native)
            expected_labels = dict(zip(expected["sample_indices"], np.asarray(expected["y_pred"]).ravel(), strict=True))
            actual_labels = dict(zip(actual["sample_indices"], np.asarray(actual["y_pred"]).ravel(), strict=True))
            assert actual_labels == expected_labels
            assert set(actual_labels.values()) <= {0.0, 1.0}
    finally:
        legacy.close()
        native.close()

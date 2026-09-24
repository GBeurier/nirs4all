"""Held-out test predictions from fold estimators stay available without refitting."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_single_fold_file_with_existing_test_keeps_validation_and_test(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """A pre-existing test partition prevents one file fold becoming a holdout."""
    import nirs4all

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 6))
    y = x[:, 0] - 0.2 * x[:, 1]
    fold_file = tmp_path / "one_fold.json"
    fold_file.write_text(json.dumps([{"train": list(range(12, 24)), "val": list(range(12))}]), encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, Ridge()]
    legacy = nirs4all.run(pipeline, (x, y, {"train": 24}), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, (x, y, {"train": 24}), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
        assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
        expected_partitions = {("0", "train"), ("0", "val"), ("0", "test")}
        legacy_rows = {(row["fold_id"], row["partition"]): row for row in legacy.predictions.filter_predictions(load_arrays=True)}
        native_rows = {(row["fold_id"], row["partition"]): row for row in native.predictions.filter_predictions(load_arrays=True)}
        assert legacy_rows.keys() == native_rows.keys() == expected_partitions
        for key in expected_partitions:
            np.testing.assert_allclose(np.asarray(native_rows[key]["y_pred"]).ravel(),
                                       np.asarray(legacy_rows[key]["y_pred"]).ravel(), atol=1e-5)
    finally:
        legacy.close()
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("fold_format", ["json", "yaml", "csv", "txt"])
def test_fold_file_formats_keep_native_cv_and_test_rows(tmp_path, monkeypatch, mechanism: str, fold_format: str) -> None:
    """Every legacy fold-file format must produce the same native fold/test contract."""
    import nirs4all

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(238)
    X = rng.normal(size=(30, 6))
    y = X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.02, size=30)
    first, second = list(range(12)), list(range(12, 24))
    fold_file = tmp_path / f"folds.{fold_format}"
    if fold_format == "json":
        contents = json.dumps([{"train": second, "val": first}, {"train": first, "val": second}])
    elif fold_format == "yaml":
        contents = f"- train: {second}\n  val: {first}\n- train: {first}\n  val: {second}\n"
    elif fold_format == "csv":
        contents = "sample_id,fold\n" + "".join(f"{sample},{sample // 12}\n" for sample in range(24))
    else:
        contents = "\n".join(",".join(map(str, rows)) for rows in (second, first, first, second)) + "\n"
    fold_file.write_text(contents, encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, Ridge(alpha=0.5)]
    legacy = nirs4all.run(pipeline, (X, y, {"train": 24}), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0, workspace_path=tmp_path / "legacy")
    native = nirs4all.run(pipeline, (X, y, {"train": 24}), engine="dag-ml", allow_fallback=False,
                          refit=False, save_artifacts=False, save_charts=False, verbose=0,
                          workspace_path=tmp_path / "native")
    try:
        assert native.execution_engine == "dag-ml"
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
        legacy_rows = {(row["fold_id"], row["partition"]): row for row in legacy.predictions.filter_predictions(load_arrays=True)}
        native_rows = {(row["fold_id"], row["partition"]): row for row in native.predictions.filter_predictions(load_arrays=True)}
        for key in (("0", "val"), ("1", "val"), ("0", "test"), ("1", "test")):
            assert key in legacy_rows and key in native_rows
            np.testing.assert_allclose(np.asarray(native_rows[key]["y_pred"]).ravel(),
                                       np.asarray(legacy_rows[key]["y_pred"]).ravel(), atol=1e-5)
    finally:
        legacy.close()
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_fold_file_filters_a_small_number_of_unknown_sample_ids(tmp_path, monkeypatch, mechanism: str) -> None:
    """A stale sample ID in a fold file does not change the valid folds."""
    import nirs4all

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    folds = [
        {"train": [*range(10, 20), 999], "val": list(range(10))},
        {"train": list(range(10)), "val": list(range(10, 20))},
    ]
    fold_file = tmp_path / "folds.json"
    fold_file.write_text(json.dumps(folds), encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, Ridge()]
    legacy = nirs4all.run(
        pipeline, (x, y), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    native = nirs4all.run(
        pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
        workspace_path=tmp_path / mechanism, save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        assert native.execution_engine == "dag-ml"
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
        legacy_rows = {
            (row["fold_id"], row["partition"]): row
            for row in legacy.predictions.filter_predictions(load_arrays=True)
        }
        native_rows = {
            (row["fold_id"], row["partition"]): row
            for row in native.predictions.filter_predictions(load_arrays=True)
        }
        assert legacy_rows.keys() == native_rows.keys()
        for key, row in legacy_rows.items():
            native_row = native_rows[key]
            native_order = np.argsort(np.asarray(native_row["sample_indices"]).ravel())
            legacy_order = np.argsort(np.asarray(row["sample_indices"]).ravel())
            np.testing.assert_array_equal(
                np.asarray(native_row["sample_indices"]).ravel()[native_order],
                np.asarray(row["sample_indices"]).ravel()[legacy_order],
            )
            np.testing.assert_allclose(
                np.asarray(native_row["y_pred"]).ravel()[native_order],
                np.asarray(row["y_pred"]).ravel()[legacy_order], atol=1e-5,
            )
    finally:
        legacy.close()
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("preprocessing", [False, True])
def test_single_fold_file_augments_only_its_training_partition(tmp_path, monkeypatch, mechanism: str, preprocessing: bool) -> None:
    """A file-defined holdout remains outside augmentation and model fitting."""
    import nirs4all
    from nirs4all.operators.augmentation import GaussianAdditiveNoise

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(20)
    X = rng.normal(size=(36, 8))
    X[:18] += 8.0  # Make leakage from the holdout visible in a fitted scaler.
    y = X[:, 0] + rng.normal(scale=0.1, size=36)
    fold_file = tmp_path / "holdout.json"
    fold_file.write_text(json.dumps([{"train": list(range(18, 36)), "val": list(range(18))}]), encoding="utf-8")
    pipeline = [
        *([StandardScaler()] if preprocessing else []),
        {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.0)],
                                  "count": 1, "selection": "all", "random_state": 42}},
        {"split": str(fold_file)}, Ridge(alpha=0.5),
    ]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0, workspace_path=tmp_path / "legacy")
    native = nirs4all.run(pipeline, (X, y), engine="dag-ml", allow_fallback=False, refit=False,
                          save_artifacts=False, save_charts=False, verbose=0, workspace_path=tmp_path / "native")
    try:
        assert native.execution_engine == "dag-ml"
        assert any(row["partition"] == "test" for row in legacy.predictions.filter_predictions(load_arrays=False))
        test = native.predictions.filter_predictions(partition="test", load_arrays=True)
        assert len(test) == 1
        if preprocessing:
            scaler = StandardScaler().fit(X[18:])
            train_x, test_x = scaler.transform(X[18:]), scaler.transform(X[:18])
        else:
            train_x, test_x = X[18:], X[:18]
        expected = Ridge(alpha=0.5).fit(np.vstack([train_x, train_x]), np.concatenate([y[18:], y[18:]])).predict(test_x)
        np.testing.assert_allclose(np.asarray(test[0]["y_pred"]).ravel(), expected, atol=1e-5)
        assert list(test[0]["sample_indices"]) == list(range(18))
    finally:
        legacy.close()
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("refit", [False, True])
def test_cv_test_fold_and_ensemble_parity(tmp_path, monkeypatch, mechanism: str, refit: bool) -> None:
    import nirs4all

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [KFold(2, shuffle=True, random_state=0), Ridge(alpha=0.5)]
    source = dataset_path("regression")
    legacy = nirs4all.run(pipeline, source, engine="legacy", refit=refit, save_artifacts=False,
                          save_charts=False, verbose=0, workspace_path=tmp_path / "legacy")
    native = nirs4all.run(pipeline, source, engine="dag-ml", refit=refit, save_artifacts=False,
                          save_charts=False, verbose=0, workspace_path=tmp_path / "native")
    assert native.execution_engine == "dag-ml"
    if not refit:
        assert native._dagml_refit_artifacts == []  # noqa: SLF001
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
    legacy_rows = {(row["fold_id"], row["partition"]): row for row in legacy.predictions.filter_predictions(load_arrays=True)}
    native_rows = {(row["fold_id"], row["partition"]): row for row in native.predictions.filter_predictions(load_arrays=True)}
    for key in (("0", "test"), ("1", "test"), ("avg", "test"), ("w_avg", "test")):
        assert key in native_rows and key in legacy_rows
        assert native_rows[key]["test_score"] == pytest.approx(legacy_rows[key]["test_score"], abs=1e-5)
        np.testing.assert_allclose(np.asarray(native_rows[key]["y_pred"]).ravel(),
                                   np.asarray(legacy_rows[key]["y_pred"]).ravel(), atol=1e-5)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    if refit:
        assert ("final", "test") in native_rows
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("preprocessing", [False, True])
def test_fold_file_with_existing_test_keeps_fold_estimator_test_scores(tmp_path, monkeypatch, mechanism: str, preprocessing: bool) -> None:
    import nirs4all
    from nirs4all.data import DatasetConfigs

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    ids = list(map(int, dataset.index_column("sample", {"partition": "train"})))
    midpoint = len(ids) // 2
    fold_file = tmp_path / "folds.json"
    fold_file.write_text(json.dumps([
        {"train": ids[midpoint:], "val": ids[:midpoint]},
        {"train": ids[:midpoint], "val": ids[midpoint:]},
    ]), encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, *([StandardScaler()] if preprocessing else []), Ridge(alpha=0.5)]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", refit=False, save_artifacts=False,
                          save_charts=False, verbose=0, workspace_path=tmp_path / "legacy")
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", refit=False, save_artifacts=False,
                          save_charts=False, verbose=0, workspace_path=tmp_path / "native")
    assert native.execution_engine == "dag-ml"
    if not preprocessing:
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
    legacy_tests = {(row["fold_id"], row["partition"]): row for row in legacy.predictions.filter_predictions(load_arrays=True)}
    native_tests = {(row["fold_id"], row["partition"]): row for row in native.predictions.filter_predictions(load_arrays=True)}
    for key in (("0", "test"), ("1", "test"), ("avg", "test"), ("w_avg", "test")):
        assert key in native_tests and key in legacy_tests
        if not preprocessing:
            assert native_tests[key]["test_score"] == pytest.approx(legacy_tests[key]["test_score"], abs=1e-5)
        else:
            assert np.isfinite(native_tests[key]["test_score"])
    if preprocessing:
        # Legacy applies StandardScaler to the whole train partition before CV,
        # leaking validation rows. Native correctly fits it per fold.
        x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
        y_train = np.asarray(dataset.y({"partition": "train"})).ravel()
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        for fold, (fit_rows, val_rows) in enumerate(((slice(midpoint, None), slice(None, midpoint)), (slice(None, midpoint), slice(midpoint, None)))):
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
            estimator.fit(x_train[fit_rows], y_train[fit_rows])
            val_rmse = np.sqrt(np.mean((estimator.predict(x_train[val_rows]) - y_train[val_rows]) ** 2))
            test_rmse = np.sqrt(np.mean((estimator.predict(x_test) - y_test) ** 2))
            assert native_tests[(str(fold), "val")]["val_score"] == pytest.approx(val_rmse, abs=1e-5)
            assert native_tests[(str(fold), "test")]["test_score"] == pytest.approx(test_rmse, abs=1e-5)
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmentation_before_fold_file_preserves_test_cohort(tmp_path, monkeypatch, mechanism: str) -> None:
    import nirs4all
    from nirs4all.data import DatasetConfigs
    from nirs4all.operators.augmentation import GaussianAdditiveNoise

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    source = dataset_path("regression")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    ids = list(map(int, dataset.index_column("sample", {"partition": "train"})))
    midpoint = len(ids) // 2
    fold_file = tmp_path / "folds.json"
    fold_file.write_text(json.dumps([
        {"train": ids[midpoint:], "val": ids[:midpoint]},
        {"train": ids[:midpoint], "val": ids[midpoint:]},
    ]), encoding="utf-8")
    pipeline = [
        {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)],
                                 "count": 1, "selection": "all", "random_state": 42}},
        {"split": str(fold_file)}, Ridge(alpha=0.5),
    ]
    result = nirs4all.run(pipeline, source, engine="dag-ml", refit=False, save_artifacts=False,
                          save_charts=False, verbose=0, workspace_path=tmp_path / "native")
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    rows = {(row["fold_id"], row["partition"]): row for row in result.predictions.filter_predictions(load_arrays=True)}
    assert {("0", "test"), ("1", "test"), ("avg", "test"), ("w_avg", "test")} <= rows.keys()
    assert all(np.isfinite(rows[key]["test_score"]) for key in (("0", "test"), ("1", "test"), ("avg", "test"), ("w_avg", "test")))
    result.close()

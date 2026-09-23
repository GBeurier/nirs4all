"""Public augmentation compositions that legacy already executes."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("with_splitter", [False, True])
def test_feature_branch_before_sample_augmentation_matches_legacy_and_replays(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, with_splitter: bool,
) -> None:
    """A branch feature merge remains replayable after train-only augmentation."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    path = dataset_path("regression")
    pipeline = [
        {"branch": [[StandardNormalVariate()], [StandardScaler()]]},
        {"merge": "features"},
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
    ]
    if with_splitter:
        pipeline.append(KFold(n_splits=2, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})

    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    if with_splitter:
        native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                              workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                                  workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-8)
        if with_splitter:
            assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-8)

        archive = tmp_path / "feature_branch_before_augmentation.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"}))
        replay = nirs4all.predict(archive, x_test)
        assert root_mean_squared_error(y_test, np.asarray(replay.y_pred)) == pytest.approx(native.best_rmse, abs=1e-8)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_model_checkpoint_before_augmentation_keeps_both_legacy_models(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """Each native producer keeps its own fit cohort, output and final score."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=3)},
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
        {"model": Ridge()},
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        assert len(native.predictions.filter_predictions()) == len(legacy.predictions.filter_predictions()) == 28
        for model_name in native.get_models():
            legacy_final = next(row for row in legacy.predictions.filter_predictions()
                                if row["model_name"] == model_name and row["fold_id"] == "final" and row["partition"] == "test")
            native_final = next(row for row in native.predictions.filter_predictions()
                                if row["model_name"] == model_name and row["fold_id"] == "final" and row["partition"] == "test")
            assert native_final["test_score"] == pytest.approx(legacy_final["test_score"], abs=1e-5)
            np.testing.assert_allclose(np.asarray(native_final["y_pred"]).ravel(), np.asarray(legacy_final["y_pred"]).ravel(), atol=1e-5)
        assert {report["producer_node"] for report in native._dagml_score_set["reports"]} == {"model:compat.0", "model:compat.1"}
        archive = tmp_path / "checkpoint_after_augmentation.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = nirs4all.predict(archive, x_test)
        ridge_final = next(row for row in native.predictions.filter_predictions()
                           if row["model_name"] == "Ridge" and row["fold_id"] == "final" and row["partition"] == "test")
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), np.asarray(ridge_final["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("before_count", [1, 2])
def test_multiple_model_checkpoints_across_augmentation_match_legacy(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, before_count: int,
) -> None:
    """Three and four sequential checkpoints retain their own fit views and results."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    models_before = [PLSRegression(n_components=3), LinearRegression()][:before_count]
    models_after = [Ridge(), DummyRegressor()]
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=42),
        *({"model": model} for model in models_before),
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
        *({"model": model} for model in models_after),
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        expected_models = sorted(type(model).__name__ for model in [*models_before, *models_after])
        assert native.get_models() == legacy.get_models() == expected_models
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        assert len(native_rows) == len(legacy_rows) == 14 * len(expected_models)
        for model_name in expected_models:
            legacy_final = next(row for row in legacy_rows
                                if row["model_name"] == model_name and row["fold_id"] == "final" and row["partition"] == "test")
            native_final = next(row for row in native_rows
                                if row["model_name"] == model_name and row["fold_id"] == "final" and row["partition"] == "test")
            assert native_final["test_score"] == pytest.approx(legacy_final["test_score"], abs=1e-4)
            np.testing.assert_allclose(np.asarray(native_final["y_pred"]).ravel(), np.asarray(legacy_final["y_pred"]).ravel(), atol=1e-4)
        assert len({report["producer_node"] for report in native._dagml_score_set["reports"]}) == len(expected_models)
        archive = tmp_path / "multiple_checkpoints.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = nirs4all.predict(archive, x_test)
        selected = native.best_final
        assert selected["partition"] == "test"
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), np.asarray(selected["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("before_count", [1, 2])
def test_full_train_checkpoints_across_augmentation_match_legacy(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, before_count: int,
) -> None:
    """No-split runs preserve each model's base or augmented fit cohort."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    models_before = [PLSRegression(n_components=3), LinearRegression()][:before_count]
    models_after = [Ridge(), DummyRegressor()][:before_count]
    pipeline = [
        *({"model": model} for model in models_before),
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
        *({"model": model} for model in models_after),
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    with pytest.warns(NoSplitEvaluationWarning):
        native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                              workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        expected_models = sorted(type(model).__name__ for model in [*models_before, *models_after])
        assert native.get_models() == legacy.get_models() == expected_models
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        assert len(native_rows) == len(legacy_rows) == 3 * len(expected_models)
        for model_name in expected_models:
            legacy_test = next(row for row in legacy_rows if row["model_name"] == model_name and row["partition"] == "test")
            native_test = next(row for row in native_rows if row["model_name"] == model_name and row["partition"] == "test")
            assert native_test["test_score"] == pytest.approx(legacy_test["test_score"], abs=1e-4)
            np.testing.assert_allclose(np.asarray(native_test["y_pred"]).ravel(), np.asarray(legacy_test["y_pred"]).ravel(), atol=1e-4)
        assert len({report["producer_node"] for report in native._dagml_score_set["reports"]}) == len(expected_models)
        archive = tmp_path / "full_train_checkpoints.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = nirs4all.predict(archive, x_test)
        selected = native.best_final
        assert selected["partition"] == "test"
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), np.asarray(selected["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("interleaving", [
    "x_transform", "exclude_before", "exclude_after", "two_augmentations", "contiguous_augmentations", "y_transform",
])
def test_interleaved_augmentation_checkpoints_match_legacy(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, interleaving: str,
) -> None:
    """Each checkpoint uses the data and validation FoldSet available at its step."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    def augmentation(seed: int) -> dict:
        return {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": seed,
        }}

    exclusion = {"exclude": YOutlierFilter(method="iqr", threshold=1.0)}
    middle = {
        "x_transform": [StandardScaler(), augmentation(42)],
        "exclude_before": [exclusion, augmentation(42)],
        "exclude_after": [augmentation(42), exclusion],
        "two_augmentations": [augmentation(42), StandardScaler(), augmentation(43)],
        "contiguous_augmentations": [augmentation(42), augmentation(43)],
        "y_transform": [{"y_processing": StandardScaler()}, augmentation(42)],
    }[interleaving]
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=3)}, *middle, {"model": Ridge()},
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        assert len(native_rows) == len(legacy_rows) == 28
        kept_base_ids = None
        if interleaving.startswith("exclude"):
            dataset = DatasetConfigs(path).get_dataset_at(0)
            x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
            y_train = np.asarray(dataset.y({"partition": "train"})).ravel()
            outlier_filter = YOutlierFilter(method="iqr", threshold=1.0)
            outlier_filter.fit(x_train, y_train)
            kept_base_ids = [int(sample_id) for sample_id, keep in zip(
                dataset.index_column("sample", {"partition": "train"}),
                outlier_filter.get_mask(x_train, y_train), strict=True,
            ) if keep]
        for model_name in native.get_models():
            for partition, fold_id in [("val", 0), ("val", 1), ("test", "final")]:
                legacy_row = next(row for row in legacy_rows if row["model_name"] == model_name
                                  and row["partition"] == partition and str(row["fold_id"]) == str(fold_id))
                native_row = next(row for row in native_rows if row["model_name"] == model_name
                                  and row["partition"] == partition and str(row["fold_id"]) == str(fold_id))
                legacy_ids = legacy_row["sample_indices"]
                if kept_base_ids is not None and model_name == "Ridge" and partition == "val":
                    # Legacy stores positions in the filtered active train matrix;
                    # native result rows carry the corresponding physical IDs.
                    legacy_ids = [kept_base_ids[int(index)] for index in legacy_ids]
                legacy_by_id = dict(zip(legacy_ids, np.asarray(legacy_row["y_pred"]).ravel(), strict=True))
                native_by_id = dict(zip(native_row["sample_indices"], np.asarray(native_row["y_pred"]).ravel(), strict=True))
                assert legacy_by_id.keys() == native_by_id.keys()
                np.testing.assert_allclose(
                    [native_by_id[sample_id] for sample_id in legacy_by_id],
                    list(legacy_by_id.values()), rtol=1e-4, atol=2e-3,
                )
                score_key = "test_score" if partition == "test" else "val_score"
                assert native_row[score_key] == pytest.approx(legacy_row[score_key], abs=1e-4)
        archive = tmp_path / "interleaved_checkpoints.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = nirs4all.predict(archive, x_test)
        selected = native.best_final
        assert selected["partition"] == "test"
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), np.asarray(selected["y_pred"]).ravel(), atol=1e-3)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("interleaving", [
    "x_transform", "exclude_before", "exclude_after", "two_augmentations", "contiguous_augmentations", "y_transform",
])
def test_unsplit_interleaved_augmentation_checkpoints_match_legacy(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, interleaving: str,
) -> None:
    """Unsplit checkpoints keep independent full-train cohorts and native archives."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    def augmentation(seed: int) -> dict:
        return {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": seed,
        }}

    exclusion = {"exclude": YOutlierFilter(method="iqr", threshold=1.0)}
    middle = {
        "x_transform": [StandardScaler(), augmentation(42)],
        "exclude_before": [exclusion, augmentation(42)],
        "exclude_after": [augmentation(42), exclusion],
        "two_augmentations": [augmentation(42), StandardScaler(), augmentation(43)],
        "contiguous_augmentations": [augmentation(42), augmentation(43)],
        "y_transform": [{"y_processing": StandardScaler()}, augmentation(42)],
    }[interleaving]
    pipeline = [{"model": PLSRegression(n_components=3)}, *middle, {"model": Ridge()}]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    with pytest.warns(NoSplitEvaluationWarning):
        native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                              workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        assert len(native_rows) == len(legacy_rows) == 6
        assert len(native._dagml_checkpoint_score_sets) == 2
        for model_name in native.get_models():
            for partition in ("train", "test"):
                legacy_row = next(row for row in legacy_rows if row["model_name"] == model_name and row["partition"] == partition)
                native_row = next(row for row in native_rows if row["model_name"] == model_name and row["partition"] == partition)
                assert native_row["n_samples"] == legacy_row["n_samples"]
                assert len(native_row["y_pred"]) == len(legacy_row["y_pred"])
                if partition == "train":
                    np.testing.assert_allclose(
                        np.asarray(native_row["y_pred"]).ravel(), np.asarray(legacy_row["y_pred"]).ravel(),
                        rtol=1e-4, atol=2e-3,
                    )
                if partition == "test":
                    assert list(native_row["sample_indices"]) == list(legacy_row["sample_indices"])
                    np.testing.assert_allclose(
                        np.asarray(native_row["y_pred"]).ravel(), np.asarray(legacy_row["y_pred"]).ravel(),
                        rtol=1e-4, atol=2e-3,
                    )
                    assert native_row["test_score"] == pytest.approx(legacy_row["test_score"], abs=1e-4)
        archive = tmp_path / "unsplit_interleaved_checkpoints.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = nirs4all.predict(archive, x_test)
        selected = native.best_final
        assert selected["partition"] == "test"
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), np.asarray(selected["y_pred"]).ravel(), atol=1e-3)
    finally:
        native.close()
        legacy.close()

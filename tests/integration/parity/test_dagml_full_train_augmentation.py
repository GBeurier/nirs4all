"""Sample augmentation in DAG-ML's single full-training phase."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning
from nirs4all.pipeline.dagml.run_paths import _apply_sample_augmentation

from ._datasets import PARSER_FIXTURES, dataset_path

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("fit_on_all", [None, False, True])
@pytest.mark.parametrize("with_splitter", [False, True])
def test_public_preprocessing_fit_scope_matches_legacy_and_replays(tmp_path, fit_on_all: bool | None, with_splitter: bool) -> None:
    """Only the explicit opt-in fits on train and test, and the archive keeps that fit."""
    path = dataset_path("regression")
    preprocessing = StandardScaler() if fit_on_all is None else {"preprocessing": StandardScaler(), "fit_on_all": fit_on_all}
    pipeline = [preprocessing]
    if with_splitter:
        pipeline.append(KFold(n_splits=2, shuffle=True, random_state=42))
    pipeline.append({"model": Ridge(alpha=1.0)})

    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    if with_splitter:
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
    x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    y_train = np.asarray(dataset.y({"partition": "train"}))
    y_test = np.asarray(dataset.y({"partition": "test"}))
    scaler = StandardScaler().fit(np.vstack([x_train, x_test]) if fit_on_all else x_train)
    direct = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    expected = root_mean_squared_error(y_test, direct.predict(scaler.transform(x_test)))
    other_scaler = StandardScaler().fit(x_train if fit_on_all else np.vstack([x_train, x_test]))
    other_model = Ridge(alpha=1.0).fit(other_scaler.transform(x_train), y_train)
    other_score = root_mean_squared_error(y_test, other_model.predict(other_scaler.transform(x_test)))
    assert abs(expected - other_score) > 0.01

    assert native.execution_engine == "dag-ml"
    assert legacy.best_rmse == pytest.approx(expected, abs=1e-5)
    assert native.best_rmse == pytest.approx(expected, abs=1e-5)
    if fit_on_all is True and with_splitter:
        # This explicit scope also makes the CV fit cohort identical in both
        # engines. Without it, DAG-ML fits each fold locally while legacy fits
        # the train partition once before the splitter.
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
    node_results = [frame.get("result", frame) for frame in native._dagml_node_results]
    if fit_on_all is True:
        assert any(
            "allow_fit_cv_all_observations_view" in frame["lineage"]["unsafe_flags"]
            for frame in node_results if "lineage" in frame
        )
    else:
        assert all(not frame["lineage"]["unsafe_flags"] for frame in node_results if "lineage" in frame)

    archive = tmp_path / f"fit_scope_{fit_on_all}_{with_splitter}.n4a"
    native.export(archive)
    replay = nirs4all.predict(archive, x_test)
    assert root_mean_squared_error(y_test, np.asarray(replay.y_pred)) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("all_first", [False, True])
def test_public_mixed_preprocessing_fit_scopes_preserve_each_node(tmp_path, all_first: bool) -> None:
    """Each transform consumes its predecessor's fitted output and keeps its own fit cohort."""
    path = dataset_path("regression")
    all_step = {"preprocessing": StandardScaler(), "fit_on_all": True}
    local_step = MinMaxScaler()
    pipeline = [all_step, local_step] if all_first else [local_step, all_step]
    pipeline += [KFold(n_splits=2, shuffle=True, random_state=42), {"model": Ridge(alpha=1.0)}]

    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
    x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    y_train = np.asarray(dataset.y({"partition": "train"}))
    y_test = np.asarray(dataset.y({"partition": "test"}))
    if all_first:
        first = StandardScaler().fit(np.vstack([x_train, x_test]))
        second = MinMaxScaler().fit(first.transform(x_train))
    else:
        first = MinMaxScaler().fit(x_train)
        second = StandardScaler().fit(first.transform(np.vstack([x_train, x_test])))
    direct = Ridge(alpha=1.0).fit(second.transform(first.transform(x_train)), y_train)
    expected = root_mean_squared_error(y_test, direct.predict(second.transform(first.transform(x_test))))

    assert legacy.best_rmse == pytest.approx(expected, abs=1e-5)
    assert native.best_rmse == pytest.approx(expected, abs=1e-5)
    native.export(tmp_path / "mixed_fit_scopes.n4a")
    replay = nirs4all.predict(tmp_path / "mixed_fit_scopes.n4a", x_test)
    assert root_mean_squared_error(y_test, np.asarray(replay.y_pred)) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("with_splitter", [False, True])
@pytest.mark.parametrize("chained", [False, True])
def test_public_multisource_fit_on_all_fits_each_source_and_replays(tmp_path, with_splitter: bool, chained: bool) -> None:
    """A width-changing transform fits each source on all base observations."""
    path = dataset_path("multi")
    pipeline = [{"preprocessing": PCA(n_components=3), "fit_on_all": True}]
    if chained:
        pipeline.append(MinMaxScaler())
    if with_splitter:
        pipeline.append(KFold(n_splits=2, shuffle=True, random_state=42))
    pipeline.append({"model": Ridge(alpha=1.0)})

    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    if with_splitter:
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    y_train = np.asarray(dataset.y({"partition": "train"}))
    y_test = np.asarray(dataset.y({"partition": "test"}))
    train_blocks = [np.asarray(block).reshape(len(y_train), -1) for block in dataset.x({"partition": "train"}, "3d", concat_source=False)]
    test_blocks = [np.asarray(block).reshape(len(y_test), -1) for block in dataset.x({"partition": "test"}, "3d", concat_source=False)]
    transformers = [PCA(n_components=3).fit(np.vstack([train, test])) for train, test in zip(train_blocks, test_blocks, strict=True)]
    train_transformed = [transformer.transform(block) for transformer, block in zip(transformers, train_blocks, strict=True)]
    test_transformed = [transformer.transform(block) for transformer, block in zip(transformers, test_blocks, strict=True)]
    if chained:
        local_steps = [MinMaxScaler().fit(block) for block in train_transformed]
        train_transformed = [step.transform(block) for step, block in zip(local_steps, train_transformed, strict=True)]
        test_transformed = [step.transform(block) for step, block in zip(local_steps, test_transformed, strict=True)]
    x_train = np.hstack(train_transformed)
    x_test = np.hstack(test_transformed)
    direct = Ridge(alpha=1.0).fit(x_train, y_train)
    expected = root_mean_squared_error(y_test, direct.predict(x_test))

    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    assert native.best_rmse == pytest.approx(expected, abs=1e-5)
    archive = tmp_path / "multisource_fit_on_all.n4a"
    native.export(archive)
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    assert root_mean_squared_error(y_test, np.asarray(replay.y_pred)) == pytest.approx(expected, abs=1e-5)


def test_public_multiblock_fit_on_all_keeps_fitted_source_chains(tmp_path) -> None:
    """An intermediate-fusion model receives each source's global-fit scaler."""
    path = dataset_path("multi")
    pipeline = [
        {"preprocessing": StandardScaler(), "fit_on_all": True},
        KFold(n_splits=2, shuffle=True, random_state=42),
        {"model": MBPLS(n_components=2)},
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-8)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-8)

    archive = tmp_path / "multiblock_fit_on_all.n4a"
    native.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(replay.y_pred))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-8)


def test_public_fit_on_all_after_sample_augmentation_includes_children(tmp_path) -> None:
    """An all-observation transform sees test rows and earlier augmented train rows."""
    path = dataset_path("regression")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [
        augmentation,
        {"preprocessing": StandardScaler(), "fit_on_all": True},
        KFold(n_splits=2, shuffle=True, random_state=42),
        {"model": Ridge(alpha=1.0)},
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)

    dataset = DatasetConfigs(path).get_dataset_at(0)
    _apply_sample_augmentation(augmentation, dataset)
    x_fit = np.asarray(dataset.x({}, layout="2d", include_augmented=True))
    x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d", include_augmented=True))
    y_train = np.asarray(dataset.y({"partition": "train"}, include_augmented=True))
    x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    y_test = np.asarray(dataset.y({"partition": "test"}))
    scaler = StandardScaler().fit(x_fit)
    direct = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    expected = root_mean_squared_error(y_test, direct.predict(scaler.transform(x_test)))
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    assert native.best_rmse == pytest.approx(expected, abs=1e-5)

    archive = tmp_path / "fit_on_all_after_augmentation.n4a"
    native.export(archive)
    replay = nirs4all.predict(archive, x_test)
    assert root_mean_squared_error(y_test, np.asarray(replay.y_pred)) == pytest.approx(expected, abs=1e-5)


def test_public_fit_on_all_after_fold_local_augmentation_refits_original_pool(tmp_path) -> None:
    """CV augmentation passes use a stable base, so refit does not inherit their children."""
    path = str(PARSER_FIXTURES["with_metadata"])
    balanced = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 2.0, "random_state": 42,
    }}
    pipeline = [
        balanced,
        {"preprocessing": StandardScaler(), "fit_on_all": True},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=3)},
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-8)
    assert np.isfinite(native.cv_best_score)

    archive = tmp_path / "fold_local_fit_on_all.n4a"
    native.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(replay.y_pred))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-8)


@pytest.mark.parametrize("augmentation_count", [1, 2])
@pytest.mark.parametrize("in_process", [True, False], ids=["in_process", "cli"])
def test_augmentation_without_splitter_trains_on_children_and_scores_base_only(
    augmentation_count: int, in_process: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real controller augments train; native REFIT fits children but scores base/test."""
    path = dataset_path("regression")
    if not in_process:
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    augmentation = {
        "sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1,
            "selection": "all",
            "random_state": 42,
        },
    }
    augmentations = [augmentation] * augmentation_count
    with pytest.warns(NoSplitEvaluationWarning):
        result = nirs4all.run(
            [*augmentations, {"model": PLSRegression(n_components=3)}],
            path, engine="dag-ml", save_artifacts=False,
        )

    # Reproduce the controller's augmented fit pool with the same seed, then fit sklearn directly.
    dataset = DatasetConfigs(path).get_dataset_at(0)
    for step in augmentations:
        _apply_sample_augmentation(step, dataset)
    train_all = dataset.index_column("sample", {"partition": "train"})
    origins = dataset.index_column("origin", {"partition": "train"})
    base_train = [sample for sample, origin in zip(train_all, origins, strict=True) if sample == origin]
    assert len(train_all) > len(base_train)
    direct = PLSRegression(n_components=3).fit(
        np.asarray(dataset.x_rows(train_all, layout="2d")),
        np.asarray(dataset.y({"partition": "train"}, include_augmented=True)),
    )
    expected_rmse = root_mean_squared_error(
        np.asarray(dataset.y({"partition": "test"})),
        direct.predict(np.asarray(dataset.x({"partition": "test"}, layout="2d"))),
    )
    assert result.best_rmse == pytest.approx(expected_rmse, abs=1e-9)
    assert result.execution_engine == "dag-ml"
    assert np.isnan(result.cv_best_score)
    assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
    train_reports = [report for report in result._dagml_score_set["reports"] if report["partition"] == "final"]
    assert len(train_reports) == 1
    assert train_reports[0]["row_count"] == len(base_train)


def test_public_cv_accepts_consecutive_augmentation_steps() -> None:
    """Two augmentation stages keep native CV scores on the original train rows."""
    path = dataset_path("regression")
    augmentation = {
        "sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1,
            "selection": "all",
            "random_state": 42,
        },
    }
    result = nirs4all.run(
        [augmentation, augmentation, KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=3)}],
        path, engine="dag-ml", save_artifacts=False,
    )
    base_count = len(DatasetConfigs(path).get_dataset_at(0).index_column("sample", {"partition": "train"}))
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    oof_reports = [report for report in result._dagml_score_set["reports"] if report["partition"] == "validation" and report.get("fold_id") == "avg"]
    assert len(oof_reports) == 1
    assert oof_reports[0]["row_count"] == base_count


@pytest.mark.parametrize("with_splitter", [False, True])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_public_augmentation_operator_generator_matches_legacy_and_replays(
    tmp_path, monkeypatch: pytest.MonkeyPatch, with_splitter: bool, mechanism: str,
) -> None:
    """Every generated model receives augmented data and the winning model exports."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    path = dataset_path("regression")
    pipeline = [
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
        {"_or_": [StandardNormalVariate(), StandardScaler()]},
    ]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})

    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    if with_splitter:
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-9)
        variants = {report["variant_id"] for report in native._dagml_score_set["reports"]
                    if report["partition"] == "validation" and report.get("variant_id")}
        assert len(variants) == 2
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.execution_engine == "dag-ml"
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-9)

    archive = tmp_path / "augmentation_generator.n4a"
    native.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(
        np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred),
    )
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-9)


@pytest.mark.parametrize("placement", ["before", "between"])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_public_generator_before_augmentation_matches_legacy_and_replays(
    tmp_path, monkeypatch: pytest.MonkeyPatch, placement: str, mechanism: str,
) -> None:
    """Each pre-augmentation choice gets its own spectra, native score and replay chain."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    choice = {"_or_": [StandardNormalVariate(), StandardScaler()]}
    prefix = [choice, augmentation] if placement == "before" else [augmentation, choice, augmentation]
    pipeline = [*prefix, KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=3)}]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)

    assert native.execution_engine == "dag-ml"
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-9)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-9)
    variants = {report["variant_id"] for report in native._dagml_score_set["reports"]
                if report["partition"] == "validation" and report.get("variant_id")}
    assert len(variants) == 2

    archive = tmp_path / "pre_augmentation_generator.n4a"
    native.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(
        np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred),
    )
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-9)


@pytest.mark.parametrize("generator_position", ["none", "before", "after"])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_public_augmented_cv_without_refit_matches_legacy(
    monkeypatch: pytest.MonkeyPatch, generator_position: str, mechanism: str,
) -> None:
    """Augmented CV retains every OOF choice and never fits a terminal model."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    choice = {"_or_": [StandardNormalVariate(), StandardScaler()]}
    prefix = ([choice] if generator_position == "before" else []) + [augmentation]
    if generator_position == "after":
        prefix.append(choice)
    pipeline = [*prefix, KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=3)}]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False, save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", refit=False, save_artifacts=False, verbose=0)

    assert native.execution_engine == "dag-ml"
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-9)
    assert native._dagml_refit_artifacts == []
    assert {row["partition"] for row in native.predictions.filter_predictions()} == {"train", "val", "test"}
    assert all((frame.get("result") or frame).get("lineage", {}).get("phase") != "REFIT"
               for frame in native._dagml_node_results)
    if generator_position != "none":
        variants = {report["variant_id"] for report in native._dagml_score_set["reports"]
                    if report["partition"] == "validation" and report.get("variant_id")}
        assert len(variants) == 2


@pytest.mark.parametrize("with_augmentation", [False, True])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_public_feature_branch_cv_without_refit_matches_legacy(
    monkeypatch: pytest.MonkeyPatch, with_augmentation: bool, mechanism: str,
) -> None:
    """Feature-merge branch runs fold-local preprocessing without terminal refit."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    pipeline: list[Any] = []
    if with_augmentation:
        pipeline.append({"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }})
    pipeline.extend([
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"branch": [[StandardNormalVariate()], [StandardScaler()]]},
        {"merge": "features"},
        {"model": PLSRegression(n_components=3)},
    ])
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False, save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", refit=False, save_artifacts=False, verbose=0)

    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, rel=1e-7)
    assert native._dagml_refit_artifacts == []
    assert {row["partition"] for row in native.predictions.filter_predictions()} == {"train", "val", "test"}
    assert all((frame.get("result") or frame).get("lineage", {}).get("phase") != "REFIT"
               for frame in native._dagml_node_results)


@pytest.mark.parametrize("prefix_kind", ["y_processing", "feature_augmentation", "tag"])
@pytest.mark.parametrize("with_splitter", [False, True])
def test_legacy_prefix_before_sample_augmentation_runs_and_replays(tmp_path, prefix_kind: str, with_splitter: bool) -> None:
    """A structured prefix keeps its legacy result and an exportable native refit."""
    path = str(PARSER_FIXTURES["with_metadata"])
    prefix = {
        "y_processing": {"y_processing": StandardScaler()},
        "feature_augmentation": {"feature_augmentation": [StandardNormalVariate()]},
        "tag": {"tag": YOutlierFilter(method="iqr", threshold=2.5, tag_name="out")},
    }[prefix_kind]
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [prefix, augmentation]
    if with_splitter:
        pipeline.append(KFold(n_splits=2, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=2)})

    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    if with_splitter:
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.execution_engine == "dag-ml"
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)

    archive = tmp_path / f"{prefix_kind}.n4a"
    native.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(replay.y_pred))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-9)


def test_splitter_before_sample_augmentation_matches_legacy_and_replays(tmp_path) -> None:
    """The splitter may be declared before the train-only sample augmentation."""
    path = str(PARSER_FIXTURES["with_metadata"])
    splitter = KFold(n_splits=2, shuffle=True, random_state=42)
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    model = {"model": PLSRegression(n_components=2)}
    pipeline = [splitter, augmentation, model]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    canonical = nirs4all.run([augmentation, splitter, model], path, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.execution_engine == "dag-ml"
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-9)
    assert native.cv_best_score == pytest.approx(canonical.cv_best_score, abs=1e-9)

    archive = tmp_path / "splitter_before_augmentation.n4a"
    native.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(replay.y_pred))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-9)


@pytest.mark.parametrize("with_splitter", [False, True])
@pytest.mark.parametrize("augmentation_count", [1, 2])
@pytest.mark.parametrize("dataset_key", ["regression", "multi"])
def test_pre_augmentation_transform_replays_after_export(tmp_path, with_splitter: bool, augmentation_count: int, dataset_key: str) -> None:
    """A prefix fitted before augmentation must survive .n4a export and prediction."""
    path = dataset_path(dataset_key)
    augmentation = {
        "sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1,
            "selection": "all",
            "random_state": 42,
        },
    }
    pipeline = [StandardNormalVariate(), *([augmentation] * augmentation_count)]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})
    if with_splitter:
        result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    archive = tmp_path / "augmented.n4a"
    result.export(archive)

    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(
        np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred),
    )
    assert result.execution_engine == "dag-ml"
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


def test_multisource_balanced_augmentation_is_fold_local_and_exportable(tmp_path) -> None:
    """Fold-local children preserve all source blocks and the exported prefix."""
    path = dataset_path("multi")
    standard = {
        "sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        },
    }
    balanced = {
        "sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.02)],
            "balance": "y", "max_factor": 1.1, "random_state": 42,
        },
    }
    result = nirs4all.run(
        [StandardNormalVariate(), standard, balanced, KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=3)}],
        path, engine="dag-ml", save_artifacts=False,
    )
    archive = tmp_path / "balanced.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(
        np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred),
    )
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


@pytest.mark.parametrize("exclude_first", [False, True])
@pytest.mark.parametrize("with_splitter", [False, True])
def test_public_exclusion_and_augmentation_compose(exclude_first: bool, with_splitter: bool) -> None:
    """Filter tags and augmented lineage coexist in native train/CV pools."""
    path = dataset_path("regression")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    exclusion = {"exclude": YOutlierFilter(method="iqr", threshold=1.0)}
    pipeline = [exclusion, augmentation] if exclude_first else [augmentation, exclusion]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})

    if with_splitter:
        result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.best_rmse)
    base_count = len(DatasetConfigs(path).get_dataset_at(0).index_column("sample", {"partition": "train"}))
    reports = [report for report in result._dagml_score_set["reports"] if report["partition"] == ("validation" if with_splitter else "final")]
    assert any(report["row_count"] < base_count for report in reports)


@pytest.mark.parametrize("with_exclusion", [False, True])
def test_repetition_augmentation_keeps_groups_in_native_folds(with_exclusion: bool) -> None:
    """CV never splits replicate rows of one physical sample across fold sides."""
    configs = DatasetConfigs(str(PARSER_FIXTURES["aggregate_mean"]), repetition="sample_id")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = []
    if with_exclusion:
        pipeline.append({"exclude": YOutlierFilter(method="iqr", threshold=1.0)})
    pipeline.extend([augmentation, KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=3)}])
    result = nirs4all.run(pipeline, configs, engine="dag-ml", save_artifacts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)


def test_repetition_exclusion_without_augmentation_matches_legacy() -> None:
    """Excluded repetition rows leave the grouped CV universe in both engines."""
    configs = DatasetConfigs(str(PARSER_FIXTURES["aggregate_mean"]), repetition="sample_id")
    pipeline = [
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0)},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=2)},
    ]
    legacy = nirs4all.run(pipeline, configs, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, configs, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.execution_engine == "dag-ml"
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-9)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-9)


@pytest.mark.parametrize("with_splitter", [False, True])
def test_interleaved_preprocessing_and_augmentation_replays_after_export(tmp_path, with_splitter: bool) -> None:
    """A fitted transform between two augmentation stages survives prediction replay."""
    path = dataset_path("regression")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation, StandardNormalVariate(), augmentation]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})
    if with_splitter:
        result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    archive = tmp_path / "interleaved.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


@pytest.mark.parametrize("with_splitter", [False, True])
def test_augmentation_with_duplication_feature_branch_exports(tmp_path, with_splitter: bool) -> None:
    """Feature branch models still fit augmented rows and replay after export."""
    path = dataset_path("regression")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.extend([
        {"branch": [[StandardNormalVariate()], [StandardScaler()]]},
        {"merge": "features"},
        {"model": PLSRegression(n_components=3)},
    ])
    if with_splitter:
        result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    archive = tmp_path / "branch_augmentation.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


@pytest.mark.parametrize("with_splitter", [False, True])
def test_exclude_after_post_augmentation_transform_replays(tmp_path, with_splitter: bool) -> None:
    """An exclusion filter sees transformed spectra and export replays that transform."""
    path = dataset_path("regression")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation, StandardNormalVariate(), {"exclude": YOutlierFilter(method="iqr", threshold=1.0)}]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})
    if with_splitter:
        result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    archive = tmp_path / "exclude_after_transform.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


@pytest.mark.parametrize("with_splitter", [False, True])
def test_augmentation_with_branch_model_mean_fusion(tmp_path, with_splitter: bool) -> None:
    """Branch-local models fit augmented rows and their mean survives export."""
    path = dataset_path("regression")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.extend([
        {"branch": [[{"model": PLSRegression(n_components=3)}], [{"model": Ridge(alpha=1.0)}]]},
        {"merge": "mean"},
    ])
    if with_splitter:
        result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    archive = tmp_path / "fusion_augmentation.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    prediction = nirs4all.predict(archive, x_test)
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert replay_rmse == pytest.approx(result.best_rmse, rel=1e-6, abs=1e-6)

    _apply_sample_augmentation(augmentation, dataset)
    x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d", include_augmented=True))
    y_train = np.asarray(dataset.y({"partition": "train"}, include_augmented=True))
    branch_predictions = [model.fit(x_train, y_train).predict(x_test).reshape(-1, 1)
                          for model in (PLSRegression(n_components=3), Ridge(alpha=1.0))]
    direct_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.mean(branch_predictions, axis=0))
    assert result.best_rmse == pytest.approx(direct_rmse, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("with_splitter", [False, True])
def test_augmentation_with_by_metadata_separation_matches_group_oracle(tmp_path, with_splitter: bool) -> None:
    """Each fanned model fits augmented children only from its metadata partition."""
    configs = DatasetConfigs(str(PARSER_FIXTURES["with_metadata"]))
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation]
    if with_splitter:
        pipeline.append(KFold(n_splits=3, shuffle=True, random_state=42))
    pipeline.extend([
        {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=3)}]}},
        {"merge": "concat"},
    ])
    if with_splitter:
        result = nirs4all.run(pipeline, configs, engine="dag-ml", save_artifacts=False)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            result = nirs4all.run(pipeline, configs, engine="dag-ml", save_artifacts=False)

    dataset = configs.get_dataset_at(0)
    _apply_sample_augmentation(augmentation, dataset)
    samples = [int(value) for value in dataset.index_column("sample", {})]
    origins = [int(value) for value in dataset.index_column("origin", {})]
    base = [sample for sample, origin in zip(samples, origins, strict=True) if sample == origin]
    base_group = dict(zip(base, dataset.metadata_column("group", {}), strict=True))
    group_of = {sample: base_group[origin] for sample, origin in zip(samples, origins, strict=True)}
    train = [int(value) for value in dataset.index_column("sample", {"partition": "train"})]
    test = [int(value) for value in dataset.index_column("sample", {"partition": "test"})]
    predictions: dict[int, float] = {}
    for group in sorted(set(group_of.values())):
        group_train = [sample for sample in train if group_of[sample] == group]
        group_test = [sample for sample in test if group_of[sample] == group]
        model = PLSRegression(n_components=3).fit(
            np.asarray(dataset.x_rows(group_train, layout="2d")),
            np.asarray(dataset.y({"sample": group_train}, include_augmented=True)),
        )
        for sample, value in zip(group_test, model.predict(np.asarray(dataset.x_rows(group_test, layout="2d"))).reshape(-1), strict=True):
            predictions[sample] = float(value)
    direct_rmse = root_mean_squared_error(
        np.asarray(dataset.y({"partition": "test"})), np.asarray([predictions[sample] for sample in test]),
    )
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score) if with_splitter else np.isnan(result.cv_best_score)
    assert result.best_rmse == pytest.approx(direct_rmse, abs=1e-9)
    replay_manifest = result.per_dataset[dataset.name]["separation_replay"]
    assert replay_manifest["kind"] == "by_metadata_concat"
    assert replay_manifest["metadata_key"] == "group"
    assert {member["value"] for member in replay_manifest["members"]} == set(base_group.values())

    archive = tmp_path / "metadata_augmentation.n4a"
    result.export(archive)
    test_features = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    test_groups = np.asarray([group_of[sample] for sample in test])
    replayed = nirs4all.predict(archive, {"X": test_features, "metadata": {"group": test_groups}}, engine="legacy")
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(replayed.y_pred))
    assert replay_rmse == pytest.approx(direct_rmse, rel=1e-6, abs=1e-6)
    with pytest.raises(ValueError, match="unknown 'group' partition"):
        nirs4all.predict(
            archive,
            {"X": test_features, "metadata": {"group": np.full(len(test), "not-a-trained-group")}},
            engine="legacy",
        )
    with pytest.raises(ValueError, match="metadata.*group"):
        nirs4all.predict(archive, test_features, engine="legacy")


@pytest.mark.parametrize("transform", [StandardNormalVariate(), StandardScaler(), PCA(n_components=8)])
def test_fold_local_augmentation_interleaved_with_transform_replays(tmp_path, transform) -> None:
    """Each fold fits its own ordered prefix and refit replay predicts from raw spectra."""
    configs = DatasetConfigs(str(PARSER_FIXTURES["with_metadata"]))
    balanced = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 2.0, "random_state": 42,
    }}
    standard = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.02)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [balanced, transform, standard,
                KFold(n_splits=3, shuffle=True, random_state=42),
                {"model": PLSRegression(n_components=3)}]
    result = nirs4all.run(pipeline, configs, engine="dag-ml", save_artifacts=False)
    legacy = nirs4all.run(pipeline, configs, engine="legacy", save_artifacts=False)
    archive = tmp_path / "fold_local_interleaved.n4a"
    result.export(archive)
    dataset = configs.get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    # Refit sees the full train in both engines. CV intentionally differs: the legacy controller
    # balances before splitting, while DAG-ML fits balancing within each training fold.
    assert result.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-9)
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


def test_multisource_fold_local_interleaved_prefix_replays(tmp_path) -> None:
    """Each source keeps its fold-local preprocessing and the refit export uses raw blocks."""
    path = dataset_path("multi")
    balanced = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 1.2, "random_state": 42,
    }}
    standard = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.02)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [StandardNormalVariate(), balanced, StandardScaler(), standard,
                KFold(n_splits=3, shuffle=True, random_state=42),
                {"model": PLSRegression(n_components=3)}]
    result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    archive = tmp_path / "multisource_interleaved.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert np.isfinite(result.cv_best_score)
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)


@pytest.mark.parametrize("placement", ["between", "after_transform"])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_fold_local_exclusion_uses_each_fold_and_replays(tmp_path, monkeypatch, placement: str, mechanism: str) -> None:
    """Fold-local exclusion changes fit rows without removing validation or test predictions."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    path = dataset_path("regression")
    balanced = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 1.2, "random_state": 42,
    }}
    standard = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.02)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    exclusion = {"exclude": YOutlierFilter(method="iqr", threshold=1.0)}
    prefix = [balanced, exclusion, standard] if placement == "between" else [balanced, StandardScaler(), exclusion]
    pipeline = [*prefix, KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=3)}]
    result = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False)
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False)
    archive = tmp_path / "excluded_fold_local.n4a"
    result.export(archive)
    dataset = DatasetConfigs(path).get_dataset_at(0)
    prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    replay_rmse = root_mean_squared_error(np.asarray(dataset.y({"partition": "test"})), np.asarray(prediction.y_pred))
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    oof = [report for report in result._dagml_score_set["reports"] if report["partition"] == "validation" and report.get("fold_id") == "avg"]
    assert len(oof) == 1
    assert oof[0]["row_count"] == len(dataset.index_column("sample", {"partition": "train"}))
    assert result.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-9)
    assert replay_rmse == pytest.approx(result.best_rmse, abs=1e-9)

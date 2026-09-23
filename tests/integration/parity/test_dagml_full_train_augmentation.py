"""Sample augmentation in DAG-ML's single full-training phase."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.dagml.run_paths import _apply_sample_augmentation

from ._datasets import PARSER_FIXTURES, dataset_path

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("augmentation_count", [1, 2])
def test_augmentation_without_splitter_trains_on_children_and_scores_base_only(augmentation_count: int) -> None:
    """The real controller augments train; native REFIT fits children but scores base/test."""
    path = dataset_path("regression")
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


def test_fold_local_exclusion_between_augmentations_requires_fold_views() -> None:
    """An interleaved exclusion must not silently train on rows excluded in a fold."""
    path = dataset_path("regression")
    balanced = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 1.2, "random_state": 42,
    }}
    standard = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.02)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    with pytest.raises(RtError, match="per-fold exclusion views"):
        nirs4all.run(
            [balanced, {"exclude": YOutlierFilter(method="iqr")}, standard,
             KFold(n_splits=3, shuffle=True, random_state=42),
             {"model": PLSRegression(n_components=3)}],
            path, engine="dag-ml", save_artifacts=False,
        )

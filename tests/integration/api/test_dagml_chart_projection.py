"""Charts present captured native state without changing fits or hiding data."""

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.augmentation import GaussianAdditiveNoise


@pytest.mark.parametrize("save_charts", [False, True, None])
def test_charts_use_captured_refit_without_extra_fits_and_supply_numeric_alternatives(tmp_path, monkeypatch, save_charts):
    import matplotlib

    matplotlib.use("Agg")
    import nirs4all

    rng = np.random.default_rng(194)
    X = rng.normal(size=(24, 5)) + 10
    y = X @ np.arange(1.0, 6.0)
    fitted_rows = []
    original_fit = StandardScaler.fit

    def fit(estimator, values, *args, **kwargs):
        fitted_rows.append(len(values))
        return original_fit(estimator, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    result = nirs4all.run(
        ["chart_2d", StandardScaler(), "chart_2d", KFold(n_splits=3), "fold_chart", "y_chart", Ridge()],
        (X, y), engine="dag-ml", workspace_path=tmp_path, save_artifacts=False,
        **({"save_charts": save_charts} if save_charts is not None else {}),
    )
    assert sorted(fitted_rows) == [16, 16, 16, 24]
    assert result._dagml_score_set is not None
    reports = [path for item in result.per_dataset.values() for path in item["chart_reports"]]
    if save_charts is False:
        assert reports == []
        assert not (tmp_path / "charts").exists()
        return
    assert len(reports) == 4
    for report in reports:
        path = Path(report)
        text = path.read_text()
        assert 'alt="' in text
        assert "Download exact numeric inputs" in text
        assert path.with_suffix(".csv").is_file()
        assert path.with_suffix(".json").is_file()
        assert path.with_suffix(".png").is_file()
    stage_path = next(Path(report) for report in reports if "step_002" in report)
    with stage_path.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plotted = np.array([float(row["value"]) for row in rows]).reshape(X.shape)
    scaler = result._dagml_refit_artifacts[0]["estimator"].steps[0][1]
    np.testing.assert_array_equal(plotted, scaler.transform(X.astype(np.float32)))
    assert "not out-of-fold" in stage_path.read_text()


def test_visible_only_chart_keeps_text_alternative_without_saving(tmp_path, monkeypatch, capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import nirs4all

    shown = []
    monkeypatch.setattr(plt, "show", lambda **kwargs: shown.append(kwargs))
    X = np.arange(60, dtype=float).reshape(20, 3)
    result = nirs4all.run(
        ["chart_3d", KFold(n_splits=2), Ridge()], (X, X[:, 0]), engine="dag-ml",
        workspace_path=tmp_path, save_artifacts=False, save_charts=False, plots_visible=True,
    )
    assert shown == [{"block": False}]
    assert "20 samples; original observed features" in capsys.readouterr().out
    assert result._dagml_score_set is not None
    assert not (tmp_path / "charts").exists()
    plt.close("all")


def test_target_chart_after_processing_uses_captured_target_transform(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import nirs4all

    X = np.arange(60, dtype=float).reshape(20, 3)
    y = X[:, 0] + 0.123
    result = nirs4all.run(
        [{"y_processing": StandardScaler()}, "y_chart", KFold(n_splits=2), Ridge()],
        (X, y), workspace_path=tmp_path, save_artifacts=False,
    )
    report = Path(next(iter(result.per_dataset.values()))["chart_reports"][0])
    with report.with_suffix(".csv").open() as stream:
        values = [float(row["target_0"]) for row in csv.DictReader(stream)]
    expected = result._dagml_refit_artifacts[0]["y_transform"].transform(y.astype(np.float32).reshape(-1, 1)).ravel()
    np.testing.assert_array_equal(np.asarray(values).reshape(20, 3)[:, 0], expected)


def test_charts_on_both_sides_of_augmentation_use_their_own_sample_universe(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import nirs4all

    rng = np.random.default_rng(214)
    X = rng.normal(size=(24, 5))
    y = X @ np.arange(1.0, 6.0)
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = ["chart_2d", augmentation, "chart_2d", KFold(2), Ridge()]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, (X, y), engine="dag-ml", workspace_path=tmp_path / "dag", save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-4)

    reports = [Path(path) for item in native.per_dataset.values() for path in item["chart_reports"]]
    assert len(reports) == 2
    by_step = {int(report.stem.split("_")[1]): report for report in reports}
    for step, count in [(0, 24), (2, 48)]:
        with by_step[step].with_suffix(".csv").open() as stream:
            samples = {int(row["sample_index"]) for row in csv.DictReader(stream)}
        assert len(samples) == count
    assert "original observed features" in by_step[0].read_text()
    assert "observed and synthetic augmentation features" in by_step[2].read_text()


def test_repeated_global_augmentation_charts_use_stage_scoped_samples(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import nirs4all

    rng = np.random.default_rng(214)
    X = rng.normal(size=(24, 5))
    y = X @ np.arange(1.0, 6.0)
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = ["chart_2d", augmentation, "chart_2d", augmentation, "chart_2d", KFold(2), Ridge()]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, (X, y), engine="dag-ml", workspace_path=tmp_path / "dag", save_artifacts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    assert np.isfinite(native.cv_best_score)
    reports = [Path(path) for item in native.per_dataset.values() for path in item["chart_reports"]]
    assert len(reports) == 3
    for report, expected_count in zip(reports, (24, 48, 72), strict=True):
        with report.with_suffix(".csv").open() as stream:
            samples = {int(row["sample_index"]) for row in csv.DictReader(stream)}
        assert len(samples) == expected_count


def test_chart_between_augmentations_uses_materialized_transform_stage(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import nirs4all

    rng = np.random.default_rng(214)
    X = rng.normal(size=(24, 5))
    y = X @ np.arange(1.0, 6.0)
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation, StandardScaler(), "chart_2d", augmentation, KFold(2), Ridge()]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, (X, y), engine="dag-ml", workspace_path=tmp_path / "dag", save_artifacts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    assert np.isfinite(native.cv_best_score)
    stage = native._dagml_chart_transform_snapshots[(1, 1)]
    report = Path(next(iter(native.per_dataset.values()))["chart_reports"][0])
    with report.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len({int(row["sample_index"]) for row in rows}) == 48
    values = [float(row["value"]) for row in rows]
    np.testing.assert_array_equal(np.asarray(values).reshape(48, -1), np.asarray(stage.x({"partition": "train"}, layout="2d")))
    assert "full-training REFIT augmentation view" in report.read_text()


def test_fold_local_augmentation_chart_uses_only_full_train_refit_children(tmp_path):
    import nirs4all

    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 6))
    y = X[:, 0] * 2 + X[:, 1]
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 1.2, "random_state": 42,
    }}
    pipeline = [augmentation, "chart_2d", KFold(2), Ridge()]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    native = nirs4all.run(pipeline, (X, y), engine="dag-ml", workspace_path=tmp_path / "dag", save_artifacts=False, verbose=0)
    assert np.isfinite(native.cv_best_score)
    assert len(native._dagml_chart_aug_snapshots) == 1
    snapshot = native._dagml_chart_aug_snapshots[0]
    refit_train = set(snapshot.index_column("sample", {"partition": "train"}))
    report = Path(next(iter(native.per_dataset.values()))["chart_reports"][0])
    with report.with_suffix(".csv").open() as stream:
        plotted = {int(row["sample_index"]) for row in csv.DictReader(stream)}
    assert plotted == refit_train
    assert "full-training REFIT augmentation view" in report.read_text()
    assert "not out-of-fold" in report.read_text()


def test_interleaved_fold_local_augmentation_charts_capture_each_refit_stage(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from sklearn.cross_decomposition import PLSRegression

    import nirs4all
    from nirs4all.data.config import DatasetConfigs
    from tests.integration.parity._datasets import PARSER_FIXTURES

    configs = DatasetConfigs(str(PARSER_FIXTURES["with_metadata"]))
    balanced = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "balance": "y", "max_factor": 2.0, "random_state": 42,
    }}
    standard = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.02)],
        "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [balanced, "chart_2d", StandardScaler(), "chart_2d", standard, "chart_2d", KFold(3), PLSRegression(n_components=3)]
    legacy = nirs4all.run(pipeline, configs, engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, configs, engine="dag-ml", workspace_path=tmp_path / "dag", save_artifacts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    assert np.isfinite(native.cv_best_score)
    snapshots = native._dagml_chart_aug_snapshots
    assert len(snapshots) == 2
    transformed_stage = native._dagml_chart_transform_snapshots[(1, 1)]
    reports = [Path(path) for item in native.per_dataset.values() for path in item["chart_reports"]]
    assert len(reports) == 3
    for report, snapshot, expected_count in zip(reports, (snapshots[0], transformed_stage, snapshots[1]), (78, 78, 126), strict=True):
        with report.with_suffix(".csv").open() as stream:
            samples = {int(row["sample_index"]) for row in csv.DictReader(stream)}
        assert samples == set(snapshot.index_column("sample", {"partition": "train"}))
        assert len(samples) == expected_count
        assert "full-training REFIT augmentation view" in report.read_text()

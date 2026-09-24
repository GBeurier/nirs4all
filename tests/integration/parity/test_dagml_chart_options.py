"""Public chart options must preserve legacy rendering and disclose plotted data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.controllers.charts.spectral_distribution import SpectralDistributionController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.augmentation import GaussianAdditiveNoise, MultiplicativeNoise
from nirs4all.operators.filters.y_outlier import YOutlierFilter

from ._dagml_cli import dagml_cli_path


def _mechanism(monkeypatch: pytest.MonkeyPatch, mechanism: str) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))


def _reports(result: object) -> list[Path]:
    return [Path(path) for item in result.per_dataset.values() for path in item["chart_reports"]]


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_fold_metadata_color_is_in_numeric_alternative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(291)
    x = rng.normal(size=(18, 6))
    y = x[:, 0] - x[:, 1]
    specimen = np.arange(len(x)) % 3

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("fold_metadata_chart")
        samples.add_samples(x, {"partition": "train"})
        samples.add_targets(y)
        samples.add_metadata(pl.DataFrame({"specimen": specimen}))
        return samples

    pipeline = [KFold(3), "fold_specimen", {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("fold_visualization_*_specimen.png"))
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 1
    with reports[0].with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {int(row["sample_index"]): int(row["color_specimen_0"]) for row in rows} == dict(enumerate(specimen))
    report_html = reports[0].read_text(encoding="utf-8")
    assert "Color coding uses metadata column &#x27;specimen&#x27;" in report_html
    assert "Download exact numeric inputs" in report_html
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_spectra_and_y_charts_preserve_excluded_rows_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(294)
    x = rng.normal(size=(30, 6))
    y = x[:, 0].copy()
    y[0] = 20.0

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("excluded_chart_options")
        samples.add_samples(x[:24], {"partition": "train"})
        samples.add_samples(x[24:], {"partition": "test"})
        samples.add_targets(y[:24])
        samples.add_targets(y[24:])
        return samples

    pipeline = [
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0)},
        {"chart_2d": {"include_excluded": True, "highlight_excluded": True}},
        {"chart_y": {"include_excluded": True, "highlight_excluded": True, "layout": "staggered"}},
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("Y_distribution_*_with_excluded_staggered.png"))
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 2
    for report in reports:
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        membership = {int(row["sample_index"]): row["excluded"] == "True" for row in rows}
        assert len(membership) == 30
        assert membership[0]
        assert sum(membership.values()) >= 1
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmentation_details_and_y_layout_preserve_legacy_chart_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(293)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [
        augmentation,
        {"augment_details_chart": {"alpha_original": 0.7, "alpha_augmented": 0.3, "max_samples": 8}},
        KFold(2),
        {"chart_y": {"layout": "stacked"}},
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("Augmentation_Details_Chart.png"))
    assert list((tmp_path / "legacy").rglob("Y_distribution_*_stacked.png"))
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 2
    for report in reports:
        assert report.with_suffix(".png").is_file()
        assert report.with_suffix(".csv").is_file()
    details = next(report for report in reports if "step_001" in report.name)
    with details.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({int(row["sample_index"]) for row in rows}) == 16
    assert len({int(row["sample_index"]) for row in rows if row["synthetic"] == "True"}) == 8
    assert all(int(row["origin_sample_index"]) < 20 for row in rows if row["synthetic"] == "True")
    groups = json.loads(details.with_suffix(".json").read_text(encoding="utf-8"))["plotted_groups"]
    assert len(groups["Original"]) == 8
    assert sorted({sample for samples in groups.values() for sample in samples}) == sorted({int(row["sample_index"]) for row in rows})
    assert {row["plotted_in"] for row in rows} == set(groups)
    assert "16 spectra actually plotted" in details.read_text(encoding="utf-8")
    assert "synthetic augmentation features" in details.read_text(encoding="utf-8")
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmentation_overlay_alternative_matches_legacy_sample_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(297)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    pipeline = [
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1,
            "selection": "all", "random_state": 42,
        }},
        {"augment_chart": {"alpha_original": 0.6, "alpha_augmented": 0.2, "max_samples": 5}},
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy",
                          save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("Augmentation_Chart.png"))
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    report, = _reports(native)
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))["plotted_groups"]
    expected_base = np.random.RandomState(42).choice(np.arange(20), 5, replace=False).tolist()
    expected_augmented = np.random.RandomState(42).choice(np.arange(20, 40), 5, replace=False).tolist()
    assert groups == {"Original": expected_base, "Augmented": expected_augmented}
    assert {int(row["sample_index"]) for row in rows} == set(expected_base + expected_augmented)
    assert len(rows) == 10 * x.shape[1]
    assert {row["plotted_in"] for row in rows} == {"Original", "Augmented"}
    assert "10 spectra actually plotted" in report.read_text(encoding="utf-8")
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmentation_details_exports_each_plotted_transformer_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(302)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    pipeline = [
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01), MultiplicativeNoise(sigma_gain=0.02)],
            "count": 2, "selection": "all", "random_state": 42,
        }},
        {"augmentation_details_chart": {"max_samples": 5}},
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("Augmentation_Details_Chart.png"))
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    report, = _reports(native)
    groups = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))["plotted_groups"]
    assert len(groups) == 3
    assert len(groups["Original"]) == 5
    assert all(len(ids) == 5 for ids in groups.values())
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]) for row in rows} == {sample for ids in groups.values() for sample in ids}
    assert {row["plotted_in"] for row in rows} == set(groups)
    assert len({int(row["sample_index"]) for row in rows if row["synthetic"] == "True"}) == 10
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_source_specific_spectra_and_envelope_reports_export_only_plotted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(292)
    left = rng.normal(size=(20, 6))
    right = rng.normal(size=(20, 6)) + 100
    y = left[:, 0] + right[:, 0] / 100

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("two_source_charts")
        samples.add_samples([left, right], {"partition": "train"})
        samples.add_targets(y)
        return samples

    pipeline = [{"chart_2d": {"include_excluded": True, "highlight_excluded": True}}, "spectra_envelope", KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert len(list((tmp_path / "legacy").rglob("*.png"))) >= 4
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 4
    for index, report in enumerate(reports):
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert {int(row["source"]) for row in rows} == {index % 2}
        values = np.asarray([float(row["value"]) for row in rows]).reshape(20, 6)
        np.testing.assert_allclose(values, left if index % 2 == 0 else right, rtol=0, atol=1e-5)
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_envelope_after_augmentation_exports_only_observed_spectra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(298)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    pipeline = [
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1,
            "selection": "all", "random_state": 42,
        }},
        "spectra_envelope", {"model": Ridge()},
    ]
    plotted_rows: list[int] = []
    original_plot = SpectralDistributionController._plot_spectral_distribution

    def observe(self, axis, train, test, *args, **kwargs):
        plotted_rows.append(len(train))
        return original_plot(self, axis, train, test, *args, **kwargs)

    monkeypatch.setattr(SpectralDistributionController, "_plot_spectral_distribution", observe)
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert plotted_rows == [20]
    plotted_rows.clear()
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    assert plotted_rows == [20]
    report, = _reports(native)
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]) for row in rows} == set(range(20))
    assert {row["synthetic"] for row in rows} == {"False"}
    assert "only observed samples are included" in report.read_text(encoding="utf-8")
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_fold_envelope_exports_only_first_plotted_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(299)
    x = rng.normal(size=(24, 6))
    y = x[:, 0] - x[:, 1]
    pipeline = [StandardScaler(), KFold(3), "spectral_distribution", {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=True,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("spectral_distribution_3folds.png"))
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    report, = _reports(native)
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]) for row in rows} == set(range(24))
    assert {int(row["processing"]) for row in rows} == {0}
    assert "Processing indices shown: [0]" in report.read_text(encoding="utf-8")
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_categorical_fold_colors_and_spectra_dist_alias_with_test_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(300)
    x = rng.normal(size=(25, 6))
    y = x[:, 0] - x[:, 1]
    specimen = ["A", "B", "C", "D", "E"] * 5

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("categorical_fold_envelope")
        samples.add_samples(x[:20], {"partition": "train"})
        samples.add_samples(x[20:], {"partition": "test"})
        samples.add_targets(y[:20])
        samples.add_targets(y[20:])
        samples.add_metadata(pl.DataFrame({"specimen": specimen}))
        return samples

    pipeline = [KFold(3), "fold_specimen", "spectra_dist", {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert len(list((tmp_path / "legacy").rglob("*.png"))) == 2
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    fold_report, envelope_report = _reports(native)
    with fold_report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        fold_rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]): row["color_specimen_0"] for row in fold_rows} == dict(enumerate(specimen))
    assert len(json.loads(fold_report.with_suffix(".json").read_text(encoding="utf-8"))["folds"]) == 3
    with envelope_report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        envelope_rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]): row["partition"] for row in envelope_rows} == {
        sample: "train" if sample < 20 else "test" for sample in range(25)
    }
    assert {int(row["processing"]) for row in envelope_rows} == {0}
    assert "3 scored cross-validation folds" in envelope_report.read_text(encoding="utf-8")
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_two_source_3d_spectra_include_and_highlight_excluded_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(301)
    left = rng.normal(size=(25, 6))
    right = rng.normal(size=(25, 6)) + 100
    y = left[:, 0] - left[:, 1]
    y[0] = 20

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("two_source_3d_exclusion")
        samples.add_samples([left[:20], right[:20]], {"partition": "train"})
        samples.add_samples([left[20:], right[20:]], {"partition": "test"})
        samples.add_targets(y[:20])
        samples.add_targets(y[20:])
        return samples

    pipeline = [
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0)},
        {"chart_3d": {"include_excluded": True, "highlight_excluded": True}},
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy",
                          save_charts=True, save_artifacts=False, verbose=0)
    assert len(list((tmp_path / "legacy").rglob("3D_Chart_src*.png"))) == 2
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 2
    for source, report in enumerate(reports):
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert {int(row["source"]) for row in rows} == {source}
        assert {int(row["sample_index"]) for row in rows} == set(range(25))
        assert {int(row["sample_index"]) for row in rows if row["excluded"] == "True"} >= {0}
        values = np.asarray([float(row["value"]) for row in rows]).reshape(25, 6)
        np.testing.assert_allclose(values, left if source == 0 else right, rtol=0, atol=1e-5)
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_legacy_chart_aliases_preserve_order_after_augmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(303)
    x = rng.normal(size=(18, 6))
    y = x[:, 0] - x[:, 1]
    pipeline = [
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1,
            "selection": "all", "random_state": 42,
        }},
        "augmentation_chart", "2d_chart", "3d_chart", KFold(2), "chart_fold", {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert len(list((tmp_path / "legacy").rglob("*.png"))) == 4
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 4
    for step, report in enumerate(reports, start=1):
        expected_step = step if step < 4 else 5
        assert f"step_{expected_step:03d}" in report.name
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert {int(row["sample_index"]) for row in rows} == set(range(36))
    assert len(json.loads(reports[-1].with_suffix(".json").read_text(encoding="utf-8"))["folds"]) == 2
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_multisource_fold_grid_envelopes_keep_source_and_test_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(304)
    left = rng.normal(size=(25, 6))
    right = rng.normal(size=(25, 6)) + 100
    y = left[:, 0] + right[:, 0] / 100

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("multisource_fold_grid")
        samples.add_samples([left[:20], right[:20]], {"partition": "train"})
        samples.add_samples([left[20:], right[20:]], {"partition": "test"})
        samples.add_targets(y[:20])
        samples.add_targets(y[20:])
        return samples

    pipeline = [KFold(3), "spectra_envelope", {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert len(list((tmp_path / "legacy").rglob("spectral_distribution_3folds_src*.png"))) == 2
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 2
    for source, report in enumerate(reports):
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert {int(row["source"]) for row in rows} == {source}
        assert {int(row["sample_index"]): row["partition"] for row in rows} == {
            sample: "train" if sample < 20 else "test" for sample in range(25)
        }
        assert len(json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))["folds"]) == 3
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_y_chart_preserves_classification_targets_and_fold_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    _mechanism(monkeypatch, mechanism)
    rng = np.random.default_rng(296)
    x = rng.normal(size=(24, 6))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("classification_y_chart")
        samples.add_samples(x[:20], {"partition": "train"})
        samples.add_samples(x[20:], {"partition": "test"})
        samples.add_targets(y)
        return samples

    pipeline = [StratifiedKFold(2), {"chart_y": {"layout": "stacked"}}, {"model": LogisticRegression(max_iter=200)}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("Y_distribution_*_stacked.png"))
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    reports = _reports(native)
    assert len(reports) == 1
    with reports[0].with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]): int(float(row["target_0"])) for row in rows} == dict(enumerate(y))
    assert set(native.predictions.get_folds()) >= {"0", "1"}
    legacy.close()
    native.close()

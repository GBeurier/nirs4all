"""Public chart options must preserve legacy rendering and disclose plotted data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.augmentation import GaussianAdditiveNoise

from ._dagml_cli import dagml_cli_path


def _mechanism(monkeypatch: pytest.MonkeyPatch, mechanism: str) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml CLI binary not built at {cli}")
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
    assert len({int(row["sample_index"]) for row in rows}) == 40
    assert len({int(row["sample_index"]) for row in rows if row["synthetic"] == "True"}) == 20
    assert all(int(row["origin_sample_index"]) < 20 for row in rows if row["synthetic"] == "True")
    assert "synthetic augmentation features" in details.read_text(encoding="utf-8")
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

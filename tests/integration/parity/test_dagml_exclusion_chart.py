"""A chart after exclusion uses the scored run's exclusion decisions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters.y_outlier import YOutlierFilter

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("chart_step,color_by", [
    ("exclusion_chart", "status"),
    ({"exclusion_chart": {"color_by": "y", "n_components": 3, "show_legend": False}}, "y"),
    ({"chart_exclusion": {"color_by": "reason", "title": "Excluded spectra"}}, "reason"),
])
def test_exclusion_chart_after_filter_matches_legacy_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str, chart_step: object, color_by: str,
) -> None:
    """Native charts retain the included and excluded rows in their text alternative."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(42)
    features = rng.normal(size=(30, 6))
    target = features[:, 0] + rng.normal(scale=0.1, size=30)
    target[0] = 20.0

    def dataset() -> SpectroDataset:
        spectra = SpectroDataset("exclusion_chart")
        spectra.add_samples(features[:24], {"partition": "train"})
        spectra.add_samples(features[24:], {"partition": "test"})
        spectra.add_targets(target[:24])
        spectra.add_targets(target[24:])
        return spectra

    def pipeline() -> list[object]:
        return [{"exclude": YOutlierFilter(method="iqr", threshold=1.0)}, chart_step, {"model": Ridge()}]

    legacy = nirs4all.run(
        pipeline(), dataset(), engine="legacy", save_charts=True, save_artifacts=False,
        workspace_path=tmp_path / f"legacy-{mechanism}", verbose=0,
    )
    assert list((tmp_path / f"legacy-{mechanism}").rglob(f"exclusion_chart_train_{color_by}.png"))
    legacy.close()

    native = nirs4all.run(
        pipeline(), dataset(), engine="dag-ml", save_charts=True, save_artifacts=False,
        workspace_path=tmp_path / f"native-{mechanism}", verbose=0,
    )
    reports = native.per_dataset[next(iter(native.per_dataset))]["chart_reports"]
    assert len(reports) == 1
    report = Path(reports[0])
    assert report.is_file()
    assert report.with_suffix(".png").is_file()
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    memberships = {int(row["sample_index"]): row["excluded"] == "True" for row in rows}
    oracle_filter = YOutlierFilter(method="iqr", threshold=1.0)
    oracle_filter.fit(features[:24], target[:24])
    expected_excluded = set(np.flatnonzero(~oracle_filter.get_mask(features[:24], target[:24])).tolist())
    assert len(memberships) == 24
    assert expected_excluded
    assert {sample for sample, excluded in memberships.items() if excluded} == expected_excluded
    html_report = report.read_text(encoding="utf-8")
    assert f"{24 - len(expected_excluded)} included and {len(expected_excluded)} excluded samples in train partition" in html_report
    assert "Excluded sample IDs and reasons" in html_report
    assert '<th scope="col">Sample ID</th>' in html_report
    assert all(row["exclusion_reason"] for row in rows if row["excluded"] == "True")
    native.close()


@pytest.mark.parity
def test_charts_between_sequential_exclusions_keep_stage_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each chart reflects only exclusions that occurred before that step."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    rng = np.random.default_rng(3)
    features = rng.normal(size=(32, 6))
    target = features[:, 0].copy()
    target[0] = 30.0
    target[1] = 9.0

    first = YOutlierFilter(method="iqr", threshold=1.0)
    first.fit(features, target)
    first_excluded = set(np.flatnonzero(~first.get_mask(features, target)).tolist())
    kept = np.asarray([index for index in range(len(target)) if index not in first_excluded])
    second = YOutlierFilter(method="zscore", threshold=2.0)
    second.fit(features[kept], target[kept])
    second_excluded = set(kept[~second.get_mask(features[kept], target[kept])].tolist())

    pipeline = [
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0)},
        "exclusion_chart",
        {"exclude": YOutlierFilter(method="zscore", threshold=2.0)},
        "exclusion_chart",
        {"model": Ridge()},
    ]
    native = nirs4all.run(
        pipeline, (features, target), engine="dag-ml", save_charts=True, save_artifacts=False,
        workspace_path=tmp_path, verbose=0,
    )
    reports = native.per_dataset[next(iter(native.per_dataset))]["chart_reports"]
    assert len(reports) == 2
    for report, expected in zip(reports, [first_excluded, first_excluded | second_excluded], strict=True):
        with Path(report).with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert {int(row["sample_index"]) for row in rows if row["excluded"] == "True"} == expected
    native.close()

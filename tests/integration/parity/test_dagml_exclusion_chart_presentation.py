"""Exclusion chart presentation options follow the successful legacy chart."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import nirs4all
import nirs4all.controllers.charts.exclusion as exclusion_module
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters.y_outlier import YOutlierFilter

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_custom_exclusion_title_legend_and_all_partition_match_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    rng = np.random.default_rng(305)
    x = rng.normal(size=(30, 6))
    y = x[:, 0] - x[:, 1]
    y[0] = 20

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("custom_exclusion")
        samples.add_samples(x[:24], {"partition": "train"})
        samples.add_samples(x[24:], {"partition": "test"})
        samples.add_targets(y[:24])
        samples.add_targets(y[24:])
        return samples

    pipeline = [
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0)},
        {"chart_exclusion": {
            "color_by": "reason", "n_components": 3, "partition": None,
            "title": "All observed exclusions", "show_legend": False,
        }},
        {"model": Ridge()},
    ]
    presentation: list[tuple[str, bool, str]] = []
    original_keep = exclusion_module.keep_or_close_figures

    def observe(figure, **kwargs):
        axis = figure.axes[0]
        presentation.append((axis.get_title(), axis.get_legend() is None, axis.name))
        return original_keep(figure, **kwargs)

    monkeypatch.setattr(exclusion_module, "keep_or_close_figures", observe)
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy",
                          save_charts=True, save_artifacts=False, verbose=0)
    assert list((tmp_path / "legacy").rglob("exclusion_chart_None_reason.png"))
    assert presentation == [("All observed exclusions", True, "3d")]
    presentation.clear()
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    assert presentation == [("All observed exclusions", True, "3d")]
    report = Path(next(path for item in native.per_dataset.values() for path in item["chart_reports"]))
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["sample_index"]) for row in rows} == set(range(30))
    assert {int(row["sample_index"]) for row in rows if row["excluded"] == "True"} >= {0}
    assert "All observed exclusions" in report.read_text(encoding="utf-8")
    legacy.close()
    native.close()

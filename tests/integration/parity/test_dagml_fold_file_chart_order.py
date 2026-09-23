"""A chart before a file holdout sees the pre-split sample partitions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import nirs4all
from nirs4all.controllers.charts.folds import FoldChartController
from nirs4all.operators.augmentation import GaussianAdditiveNoise

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_fold_charts_on_both_sides_of_file_holdout_keep_legacy_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """The host's early holdout lowering cannot leak into earlier chart steps."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    rng = np.random.default_rng(32)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    fold_file = tmp_path / "holdout.json"
    fold_file.write_text(json.dumps([{"train": list(range(10, 20)), "val": list(range(10))}]), encoding="utf-8")
    pipeline = ["chart_2d", "fold_chart", {"split": str(fold_file)}, "chart_2d", "fold_chart", {"model": Ridge()}]

    observed: list[tuple[int, int]] = []
    original_execute = FoldChartController.execute

    def observe(self, step_info, dataset, context, runtime_context, **kwargs):
        observed.append((len(dataset.index_column("sample", {"partition": "train"})),
                         len(dataset.index_column("sample", {"partition": "test"}))))
        return original_execute(self, step_info, dataset, context, runtime_context, **kwargs)

    monkeypatch.setattr(FoldChartController, "execute", observe)
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert observed == [(20, 0), (10, 10)]
    observed.clear()
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    assert observed == [(20, 0), (10, 10)]
    reports = [Path(path) for item in native.per_dataset.values() for path in item["chart_reports"]]
    assert len(reports) == 4
    for report, expected in zip(reports, ({"train": 20}, {"train": 20}, {"train": 10, "test": 10}, {"train": 10, "test": 10}), strict=True):
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        partitions = {int(row["sample_index"]): row["partition"] for row in rows}
        assert {partition: list(partitions.values()).count(partition) for partition in set(partitions.values())} == expected
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmented_chart_discloses_early_holdout_training_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """The image reports actual DAG fit rows when its early holdout differs from legacy."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    rng = np.random.default_rng(34)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    fold_file = tmp_path / "holdout.json"
    fold_file.write_text(json.dumps([{"train": list(range(10, 20)), "val": list(range(10))}]), encoding="utf-8")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.0)], "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation, "fold_chart", {"split": str(fold_file)}, {"model": Ridge()}]
    observed: list[int] = []
    original_execute = FoldChartController.execute

    def observe(self, step_info, dataset, context, runtime_context, **kwargs):
        observed.append(len(dataset.index_column("sample", {"partition": "train"})))
        return original_execute(self, step_info, dataset, context, runtime_context, **kwargs)

    monkeypatch.setattr(FoldChartController, "execute", observe)
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert observed == [40]
    observed.clear()
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    assert observed == [20]
    report = Path(next(path for item in native.per_dataset.values() for path in item["chart_reports"]))
    with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({int(row["sample_index"]) for row in rows}) == 20
    assert len({int(row["sample_index"]) for row in rows if row["synthetic"] == "True"}) == 10
    assert "single-file test holdout was applied before augmentation" in report.read_text(encoding="utf-8")
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmentation_and_two_file_folds_keep_ordered_chart_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """A two-fold file retains all augmented rows and chart placement."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    rng = np.random.default_rng(33)
    x = rng.normal(size=(20, 6))
    y = x[:, 0] - x[:, 1]
    fold_file = tmp_path / "two_folds.json"
    fold_file.write_text(json.dumps([
        {"train": list(range(10, 20)), "val": list(range(10))},
        {"train": list(range(10)), "val": list(range(10, 20))},
    ]), encoding="utf-8")
    augmentation = {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.0)], "count": 1, "selection": "all", "random_state": 42,
    }}
    pipeline = [augmentation, "fold_chart", {"split": str(fold_file)}, "fold_chart", {"model": Ridge()}]
    observed: list[tuple[int, int]] = []
    original_execute = FoldChartController.execute

    def observe(self, step_info, dataset, context, runtime_context, **kwargs):
        observed.append((len(dataset.index_column("sample", {"partition": "train"})), len(dataset.folds or [])))
        return original_execute(self, step_info, dataset, context, runtime_context, **kwargs)

    monkeypatch.setattr(FoldChartController, "execute", observe)
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_charts=True, save_artifacts=False, verbose=0)
    assert observed == [(40, 0), (40, 2)]
    observed.clear()
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / mechanism, save_charts=True, save_artifacts=False, verbose=0)
    assert observed == [(40, 0), (40, 2)]
    reports = [Path(path) for item in native.per_dataset.values() for path in item["chart_reports"]]
    assert len(reports) == 2
    for report, expected_folds in zip(reports, (0, 2), strict=True):
        with report.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len({int(row["sample_index"]) for row in rows}) == 40
        assert len({int(row["sample_index"]) for row in rows if row["synthetic"] == "True"}) == 20
        assert len(json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))["folds"]) == expected_folds
    legacy.close()
    native.close()

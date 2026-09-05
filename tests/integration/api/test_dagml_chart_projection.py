"""Charts present captured native state without changing fits or hiding data."""

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


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

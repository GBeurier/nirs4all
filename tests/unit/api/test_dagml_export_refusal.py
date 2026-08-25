"""Result-level exports must never retrain a DAG-ML result implicitly."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml.errors import DagMlExportRefusal


def _native_result() -> RunResult:
    return RunResult(
        predictions=Predictions(),
        per_dataset={"fixture": {"engine": "dag-ml"}},
        _dagml_export_spec={"pipeline": [], "dataset": object()},
    )


def test_default_dagml_bundle_export_refuses_without_materializing_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _native_result()

    def _unexpected_delegate() -> object:
        raise AssertionError("default export must not retrain through the legacy engine")

    monkeypatch.setattr(result, "_dagml_export_delegate", _unexpected_delegate)

    with pytest.raises(DagMlExportRefusal, match="legacy .n4a writer"):
        result.export(tmp_path / "model.n4a")

    assert result._dagml_legacy_result is None  # noqa: SLF001


def test_default_dagml_model_export_refuses_without_materializing_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _native_result()

    def _unexpected_delegate() -> object:
        raise AssertionError("default export_model must not retrain through the legacy engine")

    monkeypatch.setattr(result, "_dagml_export_delegate", _unexpected_delegate)

    with pytest.raises(DagMlExportRefusal, match="single replayable native model"):
        result.export_model(tmp_path / "model.joblib")

    assert result._dagml_legacy_result is None  # noqa: SLF001


def test_legacy_refit_requires_the_named_compatibility_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _native_result()
    bundle = tmp_path / "compat.n4a"
    model = tmp_path / "compat.joblib"

    class _Delegate:
        def export(self, output_path: Path, *, format: str) -> Path:
            assert format == "n4a"
            return output_path

        def export_model(self, output_path: Path, *, format: str | None, fold: int | None) -> Path:
            assert format == "joblib"
            assert fold == 2
            return output_path

    monkeypatch.setattr(result, "_dagml_export_delegate", lambda: _Delegate())

    assert result.export(bundle, compatibility="legacy-refit") == bundle
    assert result.export_model(model, format="joblib", fold=2, compatibility="legacy-refit") == model


def test_invalid_compatibility_is_rejected_before_any_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported dag-ml export compatibility"):
        _native_result().export(tmp_path / "model.n4a", compatibility="legacy")

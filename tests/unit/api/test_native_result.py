"""Contract tests for the native-only public run-result projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions


class _Estimator:
    training_outcome_ = {
        "score_set": {"schema_version": 1, "reports": []},
        "outcome_fingerprint": "a" * 64,
    }

    def __init__(self) -> None:
        self.exports: list[tuple[Path, str]] = []

    def export_native_archive(self, path: Path, *, archive_id: str) -> dict[str, str]:
        self.exports.append((path, archive_id))
        return {"archive_id": archive_id, "archive_sha256": "b" * 64}


def test_native_run_result_exports_the_captured_estimator_without_legacy_refit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projected = RunResult(predictions=Predictions(), per_dataset={"native": {}})
    monkeypatch.setattr(
        "nirs4all.api.native_result._scores_to_run_result",
        lambda *_args, **_kwargs: projected,
    )
    estimator = _Estimator()

    result = NativeMethodsRunResult.from_estimator(
        estimator,  # type: ignore[arg-type]
        dataset_name="native",
        model_name="PLSRegression",
    )
    output = tmp_path / "portable.n4a"

    assert result.export(output) == output
    assert estimator.exports == [(output, "archive:" + "a" * 64)]
    assert result.native_archive_reference == {
        "archive_id": "archive:" + "a" * 64,
        "archive_sha256": "b" * 64,
    }


@pytest.mark.parametrize("output", ["portable.zip", "portable"])
def test_native_run_result_requires_archive_v2_path(output: str, tmp_path: Path) -> None:
    result = NativeMethodsRunResult.__new__(NativeMethodsRunResult)
    with pytest.raises(ValueError, match=".n4a Archive V2"):
        result.export(tmp_path / output)

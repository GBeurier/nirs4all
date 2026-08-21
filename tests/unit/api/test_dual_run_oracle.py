"""Focused contract tests for the strict ``engine='dual'`` run oracle."""

from __future__ import annotations

import importlib
import importlib.resources
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.api.run import _dual_comparison_report, _resolve_dual_tolerances
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.engine import DualRunMismatchError, DualRunUnsupported

try:
    importlib.import_module("dag_ml._dag_ml")
except ImportError:
    _NATIVE_EXTENSION_AVAILABLE = False
else:
    _NATIVE_EXTENSION_AVAILABLE = True


class _Result:
    def __init__(
        self,
        *,
        num_predictions: int,
        best_score: float,
        best_rmse: float,
        cv_best_score: float,
        winner: str = "config_winner",
        validation_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.num_predictions = num_predictions
        self.best_score = best_score
        self.best_rmse = best_rmse
        self.cv_best_score = cv_best_score
        self.best = {"config_name": winner}
        self.predictions = _Predictions(validation_rows if validation_rows is not None else _validation_rows())


class _Predictions:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def filter_predictions(self, *, partition: str) -> list[dict[str, object]]:
        assert partition == "val"
        return [row for row in self._rows if row["partition"] == partition]


def _validation_rows(*, prediction_delta: float = 0.0, score_delta: float = 0.0, split_delta: int = 0) -> list[dict[str, object]]:
    return [
        {
            "partition": "val",
            "fold_id": str(fold),
            "sample_indices": [fold * 2 + split_delta, fold * 2 + 1 + split_delta],
            "y_pred": np.array([0.1 + fold + prediction_delta, 0.2 + fold + prediction_delta]),
            "val_score": 0.3 + fold + score_delta,
            "scores": {"val": {"rmse": 0.3 + fold + score_delta}},
        }
        for fold in range(3)
    ]


def _supported_request() -> tuple[list[object], tuple[np.ndarray, np.ndarray]]:
    X = np.arange(36, dtype=float).reshape(12, 3)
    y = np.linspace(0.125, 1.875, num=12) + 0.01 * np.sin(np.arange(12, dtype=float))
    return [KFold(n_splits=3, shuffle=False), {"model": PLSRegression(n_components=1)}], (X, y)


def _native_rows_for_supported_request() -> list[dict[str, object]]:
    return [
        {
            "partition": "val",
            "fold_id": str(fold),
            "sample_indices": list(range(fold * 4, fold * 4 + 4)),
            "y_pred": np.linspace(0.1 + fold, 0.4 + fold, num=4),
            "val_score": 0.3 + fold,
            "scores": {"val": {"rmse": 0.3 + fold}},
        }
        for fold in range(3)
    ]


def test_dual_comparison_records_nonblocking_timings_for_equal_semantics() -> None:
    legacy = _Result(num_predictions=4, best_score=0.2, best_rmse=0.2, cv_best_score=0.3)
    native = _Result(num_predictions=4, best_score=0.2, best_rmse=0.2, cv_best_score=0.3)

    report = _dual_comparison_report(legacy, native, legacy_seconds=2.0, native_seconds=1.0)  # type: ignore[arg-type]

    assert report["mismatches"] == []
    assert report["capability"] == {
        "native_leg": "orchestration_parity_only",
        "model_runtime": "python_sklearn_pls",
        "methods_native_execution": False,
    }
    assert report["performance"] == {
        "legacy_wall_seconds": 2.0,
        "native_wall_seconds": 1.0,
        "legacy_to_native_speedup": 2.0,
        "enforced": False,
    }
    assert report["semantics"]["winner"] == {"legacy": "config_winner", "native": "config_winner"}
    assert report["semantics"]["validation_splits"]["legacy"]["0"] == [0, 1]


def test_dual_comparison_refuses_paired_nan_required_metrics() -> None:
    unavailable = _Result(num_predictions=4, best_score=float("nan"), best_rmse=float("nan"), cv_best_score=float("nan"))

    with pytest.raises(DualRunUnsupported, match="non-finite required metric cv_best_score"):
        _dual_comparison_report(unavailable, unavailable, legacy_seconds=2.0, native_seconds=1.0)  # type: ignore[arg-type]


def test_dual_comparison_permits_unavailable_global_test_summaries() -> None:
    legacy = _Result(num_predictions=4, best_score=float("nan"), best_rmse=float("nan"), cv_best_score=0.3)
    native = _Result(num_predictions=4, best_score=float("nan"), best_rmse=float("nan"), cv_best_score=0.3)
    report = _dual_comparison_report(legacy, native, legacy_seconds=2.0, native_seconds=1.0)  # type: ignore[arg-type]

    assert report["mismatches"] == []
    assert set(report["semantics"]["metrics"]) == {"cv_best_score"}


def test_dual_comparison_emits_structured_semantic_mismatch() -> None:
    legacy = _Result(num_predictions=4, best_score=0.2, best_rmse=0.2, cv_best_score=0.3)
    native = _Result(num_predictions=3, best_score=0.4, best_rmse=0.2, cv_best_score=0.3)

    report = _dual_comparison_report(legacy, native, legacy_seconds=2.0, native_seconds=1.0)  # type: ignore[arg-type]
    with pytest.raises(DualRunMismatchError) as error:
        raise DualRunMismatchError(report)

    assert error.value.report["schema_version"] == 3
    assert error.value.report["tolerances"] == _resolve_dual_tolerances()
    assert {entry["field"] for entry in error.value.report["mismatches"]} == {"num_predictions"}


def test_dual_comparison_compares_winner_splits_and_predictions() -> None:
    legacy = _Result(num_predictions=4, best_score=0.2, best_rmse=0.2, cv_best_score=0.3)
    native = _Result(
        num_predictions=4,
        best_score=0.2,
        best_rmse=0.2,
        cv_best_score=0.3,
        winner="another_winner",
        validation_rows=_validation_rows(prediction_delta=0.01, split_delta=10),
    )

    report = _dual_comparison_report(legacy, native, legacy_seconds=2.0, native_seconds=1.0)

    fields = {entry["field"] for entry in report["mismatches"]}
    assert {"winner", "validation_splits", "y_pred.sample_ids"} <= fields


def test_dual_comparison_reports_a_prediction_value_outside_ledger_tolerance() -> None:
    legacy = _Result(num_predictions=4, best_score=0.2, best_rmse=0.2, cv_best_score=0.3)
    native = _Result(
        num_predictions=4,
        best_score=0.2,
        best_rmse=0.2,
        cv_best_score=0.3,
        validation_rows=_validation_rows(prediction_delta=0.01),
    )

    report = _dual_comparison_report(legacy, native, legacy_seconds=2.0, native_seconds=1.0)

    assert any(
        mismatch["field"] == "y_pred" and mismatch["sample_id"] == 0 and mismatch["reason"] == "outside_tolerance"
        for mismatch in report["mismatches"]
    )


def test_dual_comparison_compares_each_required_fold_score_under_the_ledger_band() -> None:
    legacy = _Result(num_predictions=4, best_score=float("nan"), best_rmse=float("nan"), cv_best_score=0.3)
    native = _Result(
        num_predictions=4,
        best_score=float("nan"),
        best_rmse=float("nan"),
        cv_best_score=0.3,
        validation_rows=_validation_rows(score_delta=0.01),
    )

    report = _dual_comparison_report(legacy, native, legacy_seconds=2.0, native_seconds=1.0)

    fields = {entry["field"] for entry in report["mismatches"]}
    assert {"validation_metrics.0.val_score", "validation_metrics.0.scores.val"} <= fields


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].pop("val_score"), "missing or non-numeric required metric.*val_score"),
        (lambda rows: rows[0]["scores"].__setitem__("val", {"rmse": "not-a-number"}), "missing or non-numeric required metric.*scores\\['val'\\]"),  # type: ignore[index]
        (lambda rows: rows[0].__setitem__("val_score", float("nan")), "non-finite required metric.*val_score"),
    ],
)
def test_dual_comparison_refuses_missing_non_numeric_or_non_finite_fold_metrics(mutate: object, message: str) -> None:
    rows = _validation_rows()
    mutate(rows)  # type: ignore[operator]
    result = _Result(num_predictions=4, best_score=float("nan"), best_rmse=float("nan"), cv_best_score=0.3, validation_rows=rows)

    with pytest.raises(DualRunUnsupported, match=message):
        _dual_comparison_report(result, result, legacy_seconds=2.0, native_seconds=1.0)  # type: ignore[arg-type]


def test_dual_comparison_refuses_missing_concrete_prediction_evidence() -> None:
    incomplete = _Result(
        num_predictions=4,
        best_score=0.2,
        best_rmse=0.2,
        cv_best_score=0.3,
        validation_rows=[],
    )

    with pytest.raises(DualRunUnsupported, match="concrete OOF predictions and validation splits"):
        _dual_comparison_report(incomplete, incomplete, legacy_seconds=2.0, native_seconds=1.0)


def test_dual_comparison_refuses_identically_partial_oof_sample_evidence() -> None:
    partial = _Result(num_predictions=4, best_score=0.2, best_rmse=0.2, cv_best_score=0.3)

    with pytest.raises(DualRunUnsupported, match="exactly cover 0..11"):
        _dual_comparison_report(
            partial,
            partial,
            legacy_seconds=2.0,
            native_seconds=1.0,
            expected_folds=3,
            expected_sample_count=12,
        )


def test_dual_tolerances_are_resolved_from_the_compatibility_ledger() -> None:
    ledger_path = Path(__file__).resolve().parents[3] / "docs" / "compatibility.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    expected = {
        band["metric_class"]: {
            "band_id": band["band_id"],
            "numeric_path": band["numeric_path"],
            "metric_class": band["metric_class"],
            "absolute": band["abs_tol"],
            "relative": band["rel_tol"],
        }
        for band in ledger["tolerance_bands"]
        if band["numeric_path"] == "cross_impl_pipeline"
        and band["metric_class"] in {"score", "prediction"}
        and band["enforced_at"].split(":")[-1] in {"_DEFAULT_SCORE_TOL", "_DEFAULT_YPRED_TOL"}
    }

    assert _resolve_dual_tolerances() == expected


def test_packaged_dual_ledger_matches_the_documented_companion() -> None:
    documented = (Path(__file__).resolve().parents[3] / "docs" / "compatibility.json").read_text(encoding="utf-8")
    packaged = importlib.resources.files("nirs4all").joinpath("compatibility_ledger.json").read_text(encoding="utf-8")

    assert packaged == documented


def test_dual_ledger_is_resolved_from_an_installed_wheel(tmp_path: Path) -> None:
    """A normal wheel must retain and resolve the resource used by the strict dual oracle."""

    repo_root = Path(__file__).resolve().parents[3]
    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    [wheel] = list(tmp_path.glob("nirs4all-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "nirs4all/compatibility_ledger.json" in archive.namelist()

    venv = tmp_path / "wheel-venv"
    create_venv = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert create_venv.returncode == 0, create_venv.stdout + create_venv.stderr
    wheel_python = venv / "bin" / "python"
    install = subprocess.run(
        [str(wheel_python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    probe = subprocess.run(
        [
            str(wheel_python),
            "-c",
            "from importlib.resources import files; from nirs4all.api.run import _resolve_dual_tolerances; "
            "assert files('nirs4all').joinpath('compatibility_ledger.json').is_file(); "
            "assert set(_resolve_dual_tolerances()) == {'score', 'prediction'}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_dual_native_refusal_is_fail_closed_without_legacy_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    import nirs4all.pipeline.dagml.run_backend as run_backend

    pipeline, dataset = _supported_request()

    def _native_refusal(*_args: object, **_kwargs: object) -> object:
        raise DagMlUnsupported("fake native coverage refusal")

    def _legacy_must_not_start(*_args: object, **_kwargs: object) -> object:
        pytest.fail("engine='dual' must not fall back to legacy after a native refusal")

    monkeypatch.setattr(run_backend, "run_via_dagml", _native_refusal)
    monkeypatch.setattr(run_module, "PipelineRunner", _legacy_must_not_start)
    monkeypatch.setattr(run_module.tempfile, "TemporaryDirectory", _legacy_must_not_start)

    with pytest.raises(DualRunUnsupported, match="no legacy fallback was run") as error:
        run_module.run(
            pipeline,
            dataset,
            engine="dual",
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )

    assert isinstance(error.value.__cause__, DagMlUnsupported)


def test_dual_native_missing_required_metric_refuses_before_legacy_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    import nirs4all.pipeline.dagml.run_backend as run_backend

    invalid_native = _Result(
        num_predictions=12,
        best_score=float("nan"),
        best_rmse=float("nan"),
        cv_best_score=float("nan"),
        validation_rows=_native_rows_for_supported_request(),
    )

    def _legacy_must_not_start(*_args: object, **_kwargs: object) -> object:
        pytest.fail("engine='dual' must preflight required native metrics before allocating a legacy workspace")

    monkeypatch.setattr(run_backend, "run_via_dagml", lambda *_args, **_kwargs: invalid_native)
    monkeypatch.setattr(run_module, "PipelineRunner", _legacy_must_not_start)
    monkeypatch.setattr(run_module.tempfile, "TemporaryDirectory", _legacy_must_not_start)
    pipeline, dataset = _supported_request()

    with pytest.raises(DualRunUnsupported, match="non-finite required metric cv_best_score"):
        run_module.run(
            pipeline,
            dataset,
            engine="dual",
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )


def test_dual_mismatch_removes_the_isolated_legacy_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The legacy store may be populated transiently, but must never escape a failed oracle comparison."""

    run_module = importlib.import_module("nirs4all.api.run")
    import nirs4all.pipeline.dagml.run_backend as run_backend

    observed: list[Path] = []
    transient = tmp_path / "legacy-dual-workspace"

    class _TemporaryWorkspace:
        def __init__(self, *, prefix: str) -> None:
            assert prefix == "nirs4all-dual-legacy-"

        def __enter__(self) -> str:
            transient.mkdir()
            return str(transient)

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            observed.extend(transient.rglob("*"))
            shutil.rmtree(transient)

    native = _Result(
        num_predictions=12,
        best_score=float("nan"),
        best_rmse=float("nan"),
        cv_best_score=0.3,
        validation_rows=_native_rows_for_supported_request(),
    )
    monkeypatch.setattr(run_backend, "run_via_dagml", lambda *_args, **_kwargs: native)
    monkeypatch.setattr(run_module.tempfile, "TemporaryDirectory", _TemporaryWorkspace)
    monkeypatch.setattr(
        run_module,
        "_dual_comparison_report",
        lambda *_args, **_kwargs: {"mismatches": [{"field": "forced", "reason": "test"}]},
    )
    pipeline, dataset = _supported_request()

    with pytest.raises(DualRunMismatchError):
        run_module.run(
            pipeline,
            dataset,
            engine="dual",
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )

    assert observed
    assert not transient.exists()


def test_dual_calibration_is_rejected_before_legacy_calibration_validation() -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    pipeline, dataset = _supported_request()

    with pytest.raises(DualRunUnsupported, match="does not support tuning or calibration"):
        run_module.run(
            pipeline,
            dataset,
            engine="dual",
            calibration={},
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )


def test_dual_refuses_unproven_public_shape_before_native_dispatch() -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    pipeline, dataset = _supported_request()
    pipeline[0] = KFold(n_splits=3, shuffle=True, random_state=7)

    with pytest.raises(DualRunUnsupported, match="KFold\\(shuffle=False\\)"):
        run_module.run(
            pipeline,
            dataset,
            engine="dual",
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )


@pytest.mark.parametrize(
    ("pipeline_change", "dataset_change"),
    [
        (lambda pipeline: [type("StrictKFold", (KFold,), {})(n_splits=3, shuffle=False), pipeline[1]], lambda dataset: dataset),
        (lambda pipeline: [pipeline[0], {"model": type("StrictPLS", (PLSRegression,), {})(n_components=1)}], lambda dataset: dataset),
        (lambda pipeline: [pipeline[0], type("StrictDict", (dict,), {})(pipeline[1])], lambda dataset: dataset),
        (lambda pipeline: pipeline, lambda dataset: (dataset[0].view(type("StrictArray", (np.ndarray,), {})), dataset[1])),
        (lambda pipeline: pipeline, lambda dataset: (dataset[0], dataset[1].astype(np.int64))),
        (lambda pipeline: pipeline, lambda dataset: (dataset[0], np.array([0.0, 1.0] * 6))),
        (lambda pipeline: pipeline, lambda dataset: (np.array([[float("nan")]]), np.array([0.5]))),
    ],
)
def test_dual_refuses_subclasses_and_non_regression_numeric_inputs_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_change: object,
    dataset_change: object,
) -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _native_must_not_start(*_args: object, **_kwargs: object) -> object:
        pytest.fail("unsupported dual request reached the native backend")

    monkeypatch.setattr(run_backend, "run_via_dagml", _native_must_not_start)
    pipeline, dataset = _supported_request()
    changed_pipeline = pipeline_change(pipeline)  # type: ignore[operator]
    changed_dataset = dataset_change(dataset)  # type: ignore[operator]

    with pytest.raises(DualRunUnsupported):
        run_module.run(
            changed_pipeline,
            changed_dataset,
            engine="dual",
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )


def test_dual_refuses_native_results_environment_before_any_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    pipeline, dataset = _supported_request()
    native_results = tmp_path / "native-results"
    monkeypatch.setenv("N4A_NATIVE_RESULTS", "1")

    with pytest.raises(DualRunUnsupported, match="N4A_NATIVE_RESULTS"):
        run_module.run(
            pipeline,
            dataset,
            engine="dual",
            random_state=7,
            save_artifacts=False,
            save_charts=False,
            plots_visible=False,
            verbose=0,
        )

    assert not native_results.exists()


def test_dual_errors_are_reexported_by_public_namespaces() -> None:
    import nirs4all
    import nirs4all.api
    import nirs4all.pipeline

    assert nirs4all.DualRunUnsupported is DualRunUnsupported
    assert nirs4all.DualRunMismatchError is DualRunMismatchError
    assert nirs4all.api.DualRunUnsupported is DualRunUnsupported
    assert nirs4all.api.DualRunMismatchError is DualRunMismatchError
    assert nirs4all.pipeline.DualRunUnsupported is DualRunUnsupported
    assert nirs4all.pipeline.DualRunMismatchError is DualRunMismatchError
    assert {"DualRunUnsupported", "DualRunMismatchError"} <= set(nirs4all.api.__all__)


@pytest.mark.skipif(
    not _NATIVE_EXTENSION_AVAILABLE,
    reason="requires an installed dag_ml._dag_ml native extension for the real dual-run probe",
)
def test_dual_run_with_real_native_extension_compares_concrete_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    pipeline, dataset = _supported_request()

    result = run_module.run(
        pipeline,
        dataset,
        engine="dual",
        random_state=7,
        save_artifacts=False,
        save_charts=False,
        plots_visible=False,
        verbose=0,
    )

    report = result._dual_run_report
    assert report is not None
    assert report["mismatches"] == []
    assert report["schema_version"] == 3
    assert report["capability"]["methods_native_execution"] is False
    assert report["semantics"]["winner"]["legacy"] == report["semantics"]["winner"]["native"]
    assert report["semantics"]["validation_splits"]["legacy"] == report["semantics"]["validation_splits"]["native"]
    assert report["semantics"]["y_pred"]["legacy_sample_ids"] == list(range(12))
    assert report["semantics"]["y_pred"]["native_sample_ids"] == list(range(12))
    assert math.isfinite(report["semantics"]["metrics"]["cv_best_score"]["legacy"])
    assert math.isfinite(report["semantics"]["validation_metrics"]["legacy"]["0"]["val_score"])

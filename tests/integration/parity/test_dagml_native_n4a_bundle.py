"""NATIVE ``.n4a`` bundle export for a single-model dag-ml run (W13, P3 — the ``export()`` counterpart of 2c-ii).

W3 closed the cross-engine ``.n4a`` *verification* gap (B-011) but flagged that ``RunResult.export(".n4a")``
on a dag-ml run had NO native path: it ALWAYS re-fit the frozen pipeline through the legacy engine (the P1c
bridge). W3's handoff #1 was to make ``export(".n4a")`` build the bundle from the CAPTURED native refit
artifact the way ``export_model`` already does for joblib. This module pins that native ``.n4a`` path and —
critically — proves it is NOT a legacy refit under another name:

* (1) SNV + PLS with native results ON → ``export(".n4a")`` builds the bundle from the captured REFIT model:
  a :class:`BundleLoader` reload-predict EXACTLY matches the dag-ml run's final-(test) ``y_pred`` (1e-6),
  NO stochastic warning fires, NO legacy bridge is materialized, and the manifest is marked ``dagml_native``.
* (2) THE PROOF IT IS NOT A LEGACY REFIT: with the legacy ``run()`` monkeypatched to RAISE, ``export(".n4a")``
  still succeeds and reload-predicts exactly — the export never touches the legacy engine.
* (3) a ``y_processing`` variant → the inverse ``y_transform`` round-trips through the bundle (original
  target space), still exact.
* (4) NO native dir → the ``.n4a`` export refuses with a stable RtError and does not call the legacy engine.
* (5) a duplication branch + mean-fusion run (multiple captured artifacts) WITH a native dir → exports a
  native multi-artifact bundle that averages captured branch REFIT models, without invoking the bridge.
* (6) a by_source branch + mean-fusion run (one captured artifact per source) WITH a native dir → exports a
  native multi-artifact bundle that splits raw concatenated features by source width and averages source
  REFIT models, without invoking the bridge.
* (7) a branch/stacking run WITH a native replay manifest → exports a native multi-artifact bundle that
  rebuilds base-prediction meta-features from raw X and never invokes the bridge.
* (8) ``format="n4a.py"`` (portable script) on a native single-artifact run → refuses with a stable RtError
  and does not call the legacy engine (the native writer produces the ZIP bundle only; never a silent
  ZIP-under-``.n4a.py``).
"""

from __future__ import annotations

import importlib
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.transforms.scalers import StandardNormalVariate as SNV
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.dagml.run_backend import run_via_dagml

pytestmark = [pytest.mark.parity]

from ._dagml_cli import dagml_cli_path  # noqa: E402
from ._datasets import dataset_path  # noqa: E402

_DAGML_CLI = dagml_cli_path()
_N_SPLITS = 3


def _refit_x(dataset_key: str) -> np.ndarray:
    """The held-out test feature matrix (2D) at the dataset's NATIVE dtype (float32 — the dtype the run and
    hence the exported model predict on; a float64 widen would inject a ~1e-4 input-dtype artifact)."""
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    test_ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    return np.asarray(base.x_rows(test_ids, layout="2d"))


def _refit_y(dataset_key: str) -> np.ndarray:
    """The held-out test target vector in the same sample order as :func:`_refit_x`."""
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    return np.asarray(base.y({"partition": "test"}, include_augmented=False), dtype=float).ravel()


def _refit_source_widths(dataset_key: str) -> list[int]:
    """Per-source feature widths for the held-out test matrix."""
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    test_ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    blocks = base.x_rows(test_ids, layout="2d", concat_source=False)
    per_source = blocks if isinstance(blocks, list) else [blocks]
    return [int(np.asarray(block).shape[1]) for block in per_source]


def _final_test_pred(result) -> np.ndarray:
    """The dag-ml run's final-(test) y_pred row, ravelled (the scored REFIT model's test predictions)."""
    rows = result.predictions.filter_predictions(partition="test", fold_id="final")
    assert len(rows) == 1, "exactly one final-test row for a single concrete run"
    return np.asarray(rows[0]["y_pred"], dtype=float).ravel()


def _run_native(tmp_path: Path, pipeline, *, random_state, dataset_key: str = "regression"):
    """Run ``pipeline`` through dag-ml with native results ON (so a native dir + captured artifact exist)."""
    return run_via_dagml(
        pipeline, dataset_path(dataset_key), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(tmp_path / "results"),
        random_state=random_state,
    )


def _read_manifest(bundle_path: Path) -> dict:
    with zipfile.ZipFile(bundle_path) as zf:
        return json.loads(zf.read("manifest.json"))


def _poison_legacy_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any implicit legacy refit fail the test immediately."""
    run_module = importlib.import_module("nirs4all.api.run")

    def _boom(*_args, **_kwargs):
        raise AssertionError("default dag-ml .n4a export must not invoke engine='legacy'")

    monkeypatch.setattr(run_module, "run", _boom)


def _assert_export_refusal(excinfo: pytest.ExceptionInfo[RtError]) -> None:
    error = excinfo.value
    assert isinstance(error, RtError)
    payload = error.to_dict()
    assert payload["verb"] == "export"
    assert payload["cause"] == "unsupported_capability"
    assert payload["unsupported_capability"] == "dagml_native_export"
    assert "does not rerun the pipeline with engine='legacy'" in payload["message"]
    assert "nirs4all-tools" in payload["mitigation"]
    assert "compatibility='legacy-refit'" in payload["mitigation"]


# ---------------------------------------------------------------------------
# (1) SINGLE concrete run → native .n4a, exact reload-predict, NO warning, NO bridge.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_n4a_export_exact_and_no_warning(tmp_path: Path) -> None:
    """SNV + PLS with native results ON → ``export(".n4a")`` bundles the CAPTURED refit model; the bundle's
    BundleLoader.predict(X_test) EXACTLY matches the dag-ml run's final-(test) y_pred (1e-6), emits NO
    stochastic warning, and materializes NO legacy bridge. The manifest is marked as a native export."""
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = _run_native(tmp_path, pipeline, random_state=None)  # unseeded → the bridge WOULD warn; native won't
    assert result._dagml_results_dir is not None  # noqa: SLF001
    assert len(result._dagml_refit_artifacts) == 1  # noqa: SLF001

    out = tmp_path / "model.n4a"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any stochastic (bridge) warning becomes an error → NONE must fire
        returned = result.export(out)
    assert returned == out and out.exists()

    # No legacy bridge was materialized — the export was purely native (no on-demand legacy refit).
    assert result._dagml_legacy_result is None  # noqa: SLF001

    actual = np.asarray(BundleLoader(out).predict(_refit_x("regression")), dtype=float).ravel()
    expected = _final_test_pred(result)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-6), "the native .n4a IS the dag-ml run's scored REFIT model"

    manifest = _read_manifest(out)
    assert manifest["source_type"] == "dagml_native" and manifest["export_path"] == "dagml_native"
    assert manifest["bundle_format_version"] == "1.0" and manifest["model_step_index"] == 1


# ---------------------------------------------------------------------------
# (2) THE PROOF: legacy run() poisoned → native .n4a export STILL succeeds (never touches legacy).
# ---------------------------------------------------------------------------


class _LegacyEngineTouched(RuntimeError):
    """Raised by the poisoned legacy ``run()`` — if the native ``.n4a`` export were a legacy refit, it would
    call this and the test would fail with this exception."""


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_n4a_export_never_refits_on_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The definitive "not a legacy refit under another name" proof: after a native run, POISON the legacy
    ``nirs4all.api.run.run`` (the sole entry the P1c bridge re-fits through) to RAISE. ``export(".n4a")`` must
    still succeed and reload-predict EXACTLY — proving the native path builds the bundle from the captured
    artifact and never re-runs the pipeline on the legacy engine."""
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = _run_native(tmp_path, pipeline, random_state=42)  # the RUN uses run_via_dagml, not the legacy run()
    assert result._dagml_results_dir is not None and len(result._dagml_refit_artifacts) == 1  # noqa: SLF001

    def _boom(*_args, **_kwargs):
        raise _LegacyEngineTouched("the legacy engine must NOT be invoked by a native .n4a export")

    # The bridge does ``from nirs4all.api.run import run as _run`` at call time.
    # The package also re-exports ``run`` as ``nirs4all.api.run`` (a function),
    # so import the submodule explicitly before patching the attribute a legacy
    # refit would resolve to. A native export never imports/calls it.
    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", _boom)

    out = tmp_path / "model_poisoned.n4a"
    returned = result.export(out)  # must NOT raise _LegacyEngineTouched
    assert returned == out and out.exists()
    assert result._dagml_legacy_result is None  # noqa: SLF001 -- the bridge was never materialized

    actual = np.asarray(BundleLoader(out).predict(_refit_x("regression")), dtype=float).ravel()
    assert np.allclose(actual, _final_test_pred(result), atol=1e-6)


# ---------------------------------------------------------------------------
# (3) y_processing variant → the inverse y_transform round-trips through the bundle.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_n4a_export_applies_y_inverse(tmp_path: Path) -> None:
    """A y_processing (MinMaxScaler on y) run → the native ``.n4a`` bundle applies the inverse y_transform on
    reload, so BundleLoader.predict is in the ORIGINAL target space and matches the run's final-test y_pred."""
    pipeline = [
        SNV(), {"y_processing": MinMaxScaler()},
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)},
    ]
    result = _run_native(tmp_path, pipeline, random_state=42)
    captured = result._dagml_refit_artifacts  # noqa: SLF001
    assert len(captured) == 1 and captured[0]["y_transform"] is not None, "y_processing → a captured y_transform"

    out = tmp_path / "model_yproc.n4a"
    result.export(out)
    assert result._dagml_legacy_result is None  # noqa: SLF001 -- native, not the bridge

    actual = np.asarray(BundleLoader(out).predict(_refit_x("regression")), dtype=float).ravel()
    assert np.allclose(actual, _final_test_pred(result), atol=1e-6), "the y inverse round-trips through the bundle"


# ---------------------------------------------------------------------------
# (4) NO native dir → stable refusal, no legacy refit.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_no_native_dir_n4a_refuses_without_legacy_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``export(".n4a")`` on a dag-ml run WITHOUT a native results dir refuses and does not call legacy."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable,
        random_state=None,
    )
    assert result._dagml_results_dir is None  # noqa: SLF001 -- no native dir was written

    _poison_legacy_run(monkeypatch)
    out = tmp_path / "model_refused.n4a"
    with pytest.raises(RtError) as excinfo:
        result.export(out)
    _assert_export_refusal(excinfo)
    assert not out.exists()
    assert result._dagml_legacy_result is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# (5) branch mean-fusion run WITH native dir → native multi-artifact .n4a, NO bridge.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_branch_fusion_n4a_export_never_refits_on_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplication-branch + mean-fusion run captures multiple branch REFIT artifacts. The native ``.n4a``
    export packages a fusion wrapper over those artifacts, reproduces the native final-test RMSE, and never
    calls the legacy bridge."""
    from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

    pipeline = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {
            "branch": [
                [SNV(), {"model": PLSRegression(n_components=5)}],
                [MSC(), {"model": PLSRegression(n_components=5)}],
            ]
        },
        {"merge": "mean"},
    ]
    result = _run_native(tmp_path, pipeline, random_state=42)
    assert result._dagml_results_dir is not None  # noqa: SLF001 -- a native dir WAS written
    assert len(result._dagml_refit_artifacts) == 2, "mean fusion captures one REFIT artifact per branch"  # noqa: SLF001

    native_manifest = json.loads((result._dagml_results_dir / "manifest.json").read_text(encoding="utf-8"))  # noqa: SLF001
    assert "merge:fusion" in native_manifest["final_producer_nodes"]

    def _boom(*_args, **_kwargs):
        raise _LegacyEngineTouched("the legacy engine must NOT be invoked by a native fusion .n4a export")

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", _boom)

    out = tmp_path / "model_branch_fusion.n4a"
    returned = result.export(out)
    assert returned == out and out.exists()
    assert result._dagml_legacy_result is None  # noqa: SLF001 -- the bridge was never materialized

    actual = np.asarray(BundleLoader(out).predict(_refit_x("regression")), dtype=float).ravel()
    y_true = _refit_y("regression")
    assert actual.shape == y_true.shape
    rmse = float(np.sqrt(np.mean((actual - y_true) ** 2)))
    assert np.isclose(rmse, result.best_rmse, atol=1e-6), "fusion .n4a reproduces the native branch-fusion final RMSE"

    manifest = _read_manifest(out)
    assert manifest["source_type"] == "dagml_native"
    assert manifest["export_path"] == "dagml_native_fusion"
    assert manifest["dagml_native_export_shape"] == "branch_fusion_mean"
    assert manifest["dagml_artifact_count"] == 2


# ---------------------------------------------------------------------------
# (6) by_source mean-fusion run WITH native dir → native multi-artifact .n4a, NO bridge.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_by_source_fusion_n4a_export_never_refits_on_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A by_source branch + mean-fusion run captures one REFIT artifact per source. The native ``.n4a`` export
    packages a source-splitting fusion wrapper over those artifacts, reproduces the native final-test RMSE,
    and never calls the legacy bridge."""
    pipeline = [
        ShuffleSplit(n_splits=_N_SPLITS, random_state=42),
        {
            "branch": {
                "by_source": True,
                "steps": [
                    SNV(),
                    {"model": PLSRegression(n_components=10)},
                ],
            }
        },
        {"merge": "mean"},
    ]
    result = _run_native(tmp_path, pipeline, random_state=42, dataset_key="multi")
    assert result._dagml_results_dir is not None  # noqa: SLF001 -- a native dir WAS written
    assert len(result._dagml_refit_artifacts) == 3, "by_source mean fusion captures one REFIT artifact per source"  # noqa: SLF001

    native_manifest = json.loads((result._dagml_results_dir / "manifest.json").read_text(encoding="utf-8"))  # noqa: SLF001
    assert "merge:fusion" in native_manifest["final_producer_nodes"]
    assert native_manifest["model_names"] == ["by_source_PLSRegressionx3"]
    assert sorted(artifact["branch_index"] for artifact in native_manifest["artifacts"]) == [0, 1, 2]

    def _boom(*_args, **_kwargs):
        raise _LegacyEngineTouched("the legacy engine must NOT be invoked by a native by_source fusion .n4a export")

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", _boom)

    out = tmp_path / "model_by_source_fusion.n4a"
    returned = result.export(out)
    assert returned == out and out.exists()
    assert result._dagml_legacy_result is None  # noqa: SLF001 -- the bridge was never materialized

    actual = np.asarray(BundleLoader(out).predict(_refit_x("multi")), dtype=float).ravel()
    y_true = _refit_y("multi")
    assert actual.shape == y_true.shape
    rmse = float(np.sqrt(np.mean((actual - y_true) ** 2)))
    assert np.isclose(rmse, result.best_rmse, atol=1e-6), "by_source fusion .n4a reproduces the native final RMSE"

    manifest = _read_manifest(out)
    assert manifest["source_type"] == "dagml_native"
    assert manifest["export_path"] == "dagml_native_by_source_fusion"
    assert manifest["dagml_native_export_shape"] == "by_source_fusion_mean"
    assert manifest["dagml_artifact_count"] == 3
    assert manifest["dagml_source_count"] == 3
    assert manifest["dagml_source_widths"] == _refit_source_widths("multi")


# ---------------------------------------------------------------------------
# (7) branch/stacking run WITH native dir → native multi-artifact .n4a, NO bridge.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_branch_stacking_n4a_export_never_refits_on_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplication-branch + merge-predictions (stacking) run captures base + meta REFIT artifacts. The
    native ``.n4a`` export packages a replay wrapper over those artifacts, rebuilds the ordered base
    prediction meta-features from raw X, reproduces the native final-test RMSE, and never calls the bridge."""
    from sklearn.linear_model import Ridge

    from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

    pipeline = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {
            "branch": [
                [SNV(), {"model": PLSRegression(n_components=5)}],
                [MSC(), {"model": PLSRegression(n_components=5)}],
            ]
        },
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]
    result = _run_native(tmp_path, pipeline, random_state=42)
    assert result._dagml_results_dir is not None  # noqa: SLF001 -- a native dir WAS written
    assert len(result._dagml_refit_artifacts) == 3, "stacking captures two base artifacts + one meta artifact"  # noqa: SLF001

    native_manifest = json.loads((result._dagml_results_dir / "manifest.json").read_text(encoding="utf-8"))  # noqa: SLF001
    assert "merge:stack" in native_manifest["final_producer_nodes"]
    replay = native_manifest["stacking_replay"]
    assert replay["producer_node"] == "merge:stack"
    assert [base["producer_node"] for base in replay["base_producers"]] == ["branch:0.node:1", "branch:1.node:1"]
    assert replay["meta_artifact_id"] == "artifact:merge:stack:nirs4all:refit:variant:base"

    def _boom(*_args, **_kwargs):
        raise _LegacyEngineTouched("the legacy engine must NOT be invoked by a native stacking .n4a export")

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", _boom)

    out = tmp_path / "model_branch_stacking.n4a"
    returned = result.export(out)
    assert returned == out and out.exists()
    assert result._dagml_legacy_result is None  # noqa: SLF001 -- the bridge was never materialized

    actual = np.asarray(BundleLoader(out).predict(_refit_x("regression")), dtype=float).ravel()
    y_true = _refit_y("regression")
    assert actual.shape == y_true.shape
    rmse = float(np.sqrt(np.mean((actual - y_true) ** 2)))
    assert np.isclose(rmse, result.best_rmse, atol=1e-6), "stacking .n4a reproduces the native final RMSE"

    manifest = _read_manifest(out)
    assert manifest["source_type"] == "dagml_native"
    assert manifest["export_path"] == "dagml_native_stacking"
    assert manifest["dagml_native_export_shape"] == "branch_stacking_predictions"
    assert manifest["dagml_artifact_count"] == 3
    assert manifest["dagml_stacking_base_count"] == 2
    assert manifest["dagml_stacking_meta_artifact_id"] == "artifact:merge:stack:nirs4all:refit:variant:base"


# ---------------------------------------------------------------------------
# (8) n4a.py portable-script format → stable refusal, no legacy refit.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_n4a_py_format_refuses_without_legacy_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``export(format="n4a.py")`` on a native single-artifact run does NOT take the ZIP-only native path — it
    refuses by default and does not invoke the legacy bridge."""
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = _run_native(tmp_path, pipeline, random_state=42)
    # Sanity: with the DEFAULT (n4a) format the native path WOULD fire (exactly one loadable artifact).
    assert result._dagml_results_dir is not None and len(result._dagml_refit_artifacts) == 1  # noqa: SLF001

    _poison_legacy_run(monkeypatch)
    out = tmp_path / "model_portable.n4a.py"
    with pytest.raises(RtError) as excinfo:
        result.export(out, format="n4a.py")
    _assert_export_refusal(excinfo)
    assert not out.exists()
    assert result._dagml_legacy_result is None  # noqa: SLF001

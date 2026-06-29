"""NATIVE export_model for a single-model dag-ml run (P3 Slice 2c-ii, Codex D4/D6).

P3 2b-i + 2c-i persist a dag-ml run's predictions + scores + MODEL ARTIFACTS to a native results dir
and rehydrate the captured ``{estimator, y_transform}`` (verify-then-load, backend=joblib). 2c-ii makes
``RunResult.export_model`` on a dag-ml run that HAS a native results dir with EXACTLY ONE concrete model
artifact export the CAPTURED artifact DIRECTLY — the promised replacement of the P1c legacy-refit bridge
for the single-model case:

* (1) SINGLE concrete run (SNV + PLS) + native results → the exported model LOADS + ``predict(X_test)``
  EXACTLY matches the dag-ml run's final-(test) ``y_pred`` (within 1e-6, by sample) AND emits NO stochastic
  warning (the P1c bridge warned).
* (2) a ``y_processing`` variant → the inverse ``y_transform`` is applied (exported predict is in the
  original target space).
* (3) export_model on a dag-ml run WITHOUT a native results dir → still uses the legacy bridge
  (unchanged behavior).
* (4) a branch (duplication) run WITH a native dir → falls back to the bridge (≠1 captured artifact).
* (5) MUST-FIX 1 — a native dir whose artifact bytes are present + fingerprint-VALID but UNLOADABLE (the
  payload fails to unpickle) → export_model falls back to the bridge (returns a valid model, no crash).
* (6) MUST-FIX 2 — export_model(..., format=<non-joblib>) on a native single-artifact run uses the legacy
  bridge (which honors the requested format), NOT a silent joblib write under a foreign format.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.api.result import _DagmlExportedModel
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.transforms.scalers import StandardNormalVariate as SNV
from nirs4all.pipeline.dagml.native_results import _bytes_fingerprint
from nirs4all.pipeline.dagml.run_backend import run_via_dagml

pytestmark = [pytest.mark.parity]

from ._datasets import dataset_path  # noqa: E402

_DAGML_CLI = Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"
_N_SPLITS = 3


def _refit_x(dataset_key: str) -> np.ndarray:
    """The held-out test feature matrix (2D) — the X the exported model predicts in the exactness check.

    Materialized at the dataset's NATIVE storage dtype (float32, the SpectroDataset contract) — NOT
    widened to float64. The dag-ml run's scored REFIT model predicts the test partition on this same
    native-dtype X (the engine resolver feeds the estimator float32, see node_runner ``_predict``), so
    the exported model — which IS that same captured estimator (the joblib round-trip is bit-exact) —
    reproduces the run's ``y_pred`` EXACTLY (0.0 diff) on it. A float64 widening here would perturb the
    SNV+PLS input by ~1e-7 relative and amplify to ~1e-4 in the output (a pure input-dtype artifact, not
    a model-identity gap), which would defeat the exactness this test asserts.
    """
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    test_ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    return np.asarray(base.x_rows(test_ids, layout="2d"))


def _final_test_pred(result) -> np.ndarray:
    """The dag-ml run's final-(test) y_pred row, ravelled (the scored REFIT model's test predictions)."""
    rows = result.predictions.filter_predictions(partition="test", fold_id="final")
    assert len(rows) == 1, "exactly one final-test row for a single concrete run"
    return np.asarray(rows[0]["y_pred"], dtype=float).ravel()


# ---------------------------------------------------------------------------
# (1) SINGLE concrete run → native export, exact predict, NO stochastic warning.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_export_model_exact_and_no_warning(tmp_path: Path) -> None:
    """SNV + PLS with native results ON → export_model exports the CAPTURED model; the loaded model's
    predict(X_test) EXACTLY matches the dag-ml run's final-(test) y_pred (1e-6) AND emits NO stochastic
    warning (contrast: the P1c legacy bridge warns on an unseeded run)."""
    results_root = tmp_path / "results"
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
        random_state=None,  # would make the P1c bridge fire the CONSERVATIVE stochastic warning
    )
    # The native dir was recorded on the RunResult so export_model can find it.
    assert result._dagml_results_dir is not None  # noqa: SLF001
    assert len(result._dagml_refit_artifacts) == 1  # noqa: SLF001

    out = tmp_path / "model.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any stochastic warning becomes an error → NONE must fire
        returned = result.export_model(out)
    assert returned == out and out.exists()

    # No legacy bridge was materialized — the export was purely native (no on-demand refit).
    assert result._dagml_legacy_result is None  # noqa: SLF001

    loaded = joblib.load(out)
    actual = np.asarray(loaded.predict(_refit_x("regression")), dtype=float).ravel()
    expected = _final_test_pred(result)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-6), "the exported model IS the dag-ml run's scored REFIT model"


# ---------------------------------------------------------------------------
# (2) y_processing variant → the inverse y_transform is applied (original target space).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_export_model_applies_y_inverse(tmp_path: Path) -> None:
    """A y_processing (MinMaxScaler on y) run → the exported model applies the inverse y_transform, so its
    predict is in the ORIGINAL target space and matches the dag-ml run's final-(test) y_pred exactly."""
    results_root = tmp_path / "results"
    pipeline = [
        SNV(), {"y_processing": MinMaxScaler()},
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)},
    ]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
        random_state=42,
    )
    captured = result._dagml_refit_artifacts  # noqa: SLF001
    assert len(captured) == 1 and captured[0]["y_transform"] is not None, "y_processing → a captured y_transform"

    out = tmp_path / "model_yproc.joblib"
    result.export_model(out)
    loaded = joblib.load(out)
    # The wrapper carries a y_transform → predict is in the original target space.
    assert loaded.y_transform is not None
    x_test = _refit_x("regression")
    actual = np.asarray(loaded.predict(x_test), dtype=float).ravel()
    expected = _final_test_pred(result)
    assert np.allclose(actual, expected, atol=1e-6), "the y inverse puts predict back in the original target space"

    # The raw estimator (BEFORE the y inverse) is in the SCALED space → it must NOT match the original-space
    # final-test y_pred (proves the inverse is actually applied, not a no-op).
    raw = np.asarray(loaded.estimator.predict(x_test), dtype=float).ravel()
    assert not np.allclose(raw, expected, atol=1e-6), "the estimator alone is in the scaled space (inverse matters)"


# ---------------------------------------------------------------------------
# (3) NO native dir → legacy bridge (unchanged behavior).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_no_native_dir_uses_legacy_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """export_model on a dag-ml run WITHOUT a native results dir falls back to the P1c legacy-refit bridge
    (it materializes a legacy result + warns on the unseeded run) — the native path is skipped."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable,
        random_state=None,  # unseeded → the bridge fires the CONSERVATIVE stochastic warning
    )
    assert result._dagml_results_dir is None  # noqa: SLF001 -- no native dir was written

    out = tmp_path / "model_bridge.joblib"
    with pytest.warns(UserWarning, match="nondeterministic"):
        result.export_model(out)
    assert out.exists()
    # The legacy bridge WAS materialized (the fallback path ran), proving the native path was skipped.
    assert result._dagml_legacy_result is not None  # noqa: SLF001


# ---------------------------------------------------------------------------
# (4) branch run WITH native dir → falls back to the bridge (≠1 captured artifact).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_branch_run_falls_back_to_bridge(tmp_path: Path) -> None:
    """A duplication-branch + merge-predictions (stacking) run captures MULTIPLE REFIT artifacts (≠1), so
    even WITH a native dir the native single-artifact export is NOT applicable → it defers to the legacy
    bridge (Codex D4: multi-model / branch / stacking defer to the bridge)."""
    from sklearn.linear_model import Ridge

    from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

    results_root = tmp_path / "results"
    # Two duplication branches (LIST-of-lists form, each with its own model) → merge predictions → a Ridge
    # meta-model: several model nodes → several captured REFIT artifacts (the stacking pattern dag-ml routes
    # to _run_stacking_branch, per the parity cases).
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
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
        random_state=42,
    )
    assert result._dagml_results_dir is not None  # noqa: SLF001 -- a native dir WAS written
    assert len(result._dagml_refit_artifacts) != 1, "a branch/stacking run captures ≠1 REFIT artifact"  # noqa: SLF001

    # The native single-artifact export is not applicable (≠1 artifact) → it returns None and the export
    # falls back to the legacy bridge (which materializes a legacy result).
    out = tmp_path / "model_branch.joblib"
    result.export_model(out)
    assert out.exists()
    assert result._dagml_legacy_result is not None, "≠1 artifact → the legacy bridge backs the export"  # noqa: SLF001


# ---------------------------------------------------------------------------
# (5) MUST-FIX 1 — fingerprint-VALID but UNLOADABLE artifact → fall back to the bridge.
# ---------------------------------------------------------------------------


def _raise_unloadable() -> object:
    """A reconstructor that RAISES — pickled into the artifact so joblib.load fails AFTER fingerprint pass."""
    raise RuntimeError("simulated unloadable artifact (e.g. an unimportable class in this environment)")


class _UnloadablePayload:
    """A picklable object whose UNPICKLING raises (its ``__reduce__`` reconstructs via a raising callable).

    Re-dumping the artifact with this payload + recomputing the manifest's content_fingerprint to MATCH the
    new bytes makes the URI / backend / fingerprint guards ALL pass, yet ``joblib.load`` raises during
    unpickling — exactly the "bytes match the hash but cannot be loaded in this environment" case MUST-FIX 1
    requires the broadened catch to handle (a ``RuntimeError`` here stands in for EOFError / UnpicklingError /
    ModuleNotFoundError / ImportError / AttributeError, none of which the old narrow catch covered).
    """

    def __reduce__(self) -> tuple:
        return (_raise_unloadable, ())


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_unloadable_artifact_falls_back_to_bridge(tmp_path: Path) -> None:
    """MUST-FIX 1: a native dir whose artifact bytes are present + fingerprint-VALID but UNLOADABLE (the
    payload raises on unpickle) → export_model falls back to the legacy bridge and returns a valid model,
    no crash. The broad except around the native read makes this a fallback, not an escaped exception."""
    results_root = tmp_path / "results"
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
        random_state=42,
    )
    run_dir = result._dagml_results_dir  # noqa: SLF001
    assert run_dir is not None

    # Replace the captured artifact with an UNLOADABLE payload, then recompute the manifest fingerprint to
    # MATCH the new bytes (so the verify-then-load guards pass and joblib.load is reached + raises).
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ref = manifest["artifacts"][0]
    artifact_path = run_dir / ref["uri"]
    joblib.dump(_UnloadablePayload(), artifact_path)
    new_bytes = artifact_path.read_bytes()
    ref["content_fingerprint"] = _bytes_fingerprint(new_bytes)
    ref["size_bytes"] = len(new_bytes)
    manifest_path.write_text(json.dumps(manifest))

    # Pre-condition: the native reader now FAILS to load (not a caught verify-then-load ValueError but an
    # unpickling RuntimeError) — proving this exercises the broadened catch, not the narrow one.
    from nirs4all.pipeline.dagml.native_results import read_native_results

    with pytest.raises(RuntimeError, match="simulated unloadable artifact"):
        read_native_results(run_dir)

    # export_model must NOT crash: the broad catch falls back to the legacy bridge, which materializes a
    # legacy result and writes a loadable model.
    out = tmp_path / "model_unloadable.joblib"
    result.export_model(out)
    assert out.exists()
    assert result._dagml_legacy_result is not None, "an unloadable native artifact → the legacy bridge backs the export"  # noqa: SLF001
    loaded = joblib.load(out)
    assert hasattr(loaded, "predict"), "the bridge produced a predict-capable model despite the native-read failure"


# ---------------------------------------------------------------------------
# (6) MUST-FIX 2 — a non-joblib requested format routes through the legacy bridge (not a silent joblib write).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_non_joblib_format_uses_legacy_bridge(tmp_path: Path) -> None:
    """MUST-FIX 2: export_model(format="cloudpickle") on a native single-artifact run does NOT take the
    joblib-only native path — it falls back to the legacy bridge (which honors the requested format), so the
    output is the bridge's legacy model (NOT the native _DagmlExportedModel wrapper written as joblib)."""
    results_root = tmp_path / "results"
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
        random_state=42,
    )
    # Sanity: with the DEFAULT (joblib) format the native path WOULD fire (exactly one loadable artifact).
    assert result._dagml_results_dir is not None and len(result._dagml_refit_artifacts) == 1  # noqa: SLF001

    out = tmp_path / "model_cloudpickle.pkl"
    result.export_model(out, format="cloudpickle")
    assert out.exists()
    # The native path was SKIPPED (non-joblib request) → the legacy bridge was materialized.
    assert result._dagml_legacy_result is not None, "a non-joblib format → the legacy bridge backs the export"  # noqa: SLF001
    # The export is the bridge's legacy model, NOT the native wrapper — proving no silent joblib-as-wrapper
    # write happened under the non-joblib request.
    loaded = joblib.load(out)  # cloudpickle bytes load fine via joblib/pickle; the TYPE is what matters
    assert not isinstance(loaded, _DagmlExportedModel), "a non-joblib request must not write the native wrapper"
    assert hasattr(loaded, "predict")

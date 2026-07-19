"""The engine selector is wired into the public run() entry point.

These assert the public API resolves the engine before execution — the DEFAULT engine is now legacy
(interim, pre-refactoring): the public-maintained nirs4all stays pure-Python by default, while the
dag-ml backend stays fully selectable via ``engine="dag-ml"`` / ``$N4A_ENGINE=dag-ml``. An unknown
engine is rejected. They also certify the TRANSPARENT legacy fallback on the dag-ml path (selected
EXPLICITLY here): a supported shape runs NATIVE on dag-ml; a catchable unsupported shape
(DagMlUnsupported/NotImplementedError) OR an unavailable backend (DagMlUnavailable — neither
in-process extension nor dag-ml-cli) falls back to legacy (warning + valid result) instead of raising.
A GENUINE dag-ml bug still propagates untouched.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.config import CacheConfig
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.engine import resolve_engine

pytestmark = [pytest.mark.parity]

from ._datasets import dataset_path  # noqa: E402

_DAGML_CLI = Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"

_FALLBACK_WARNING = "falling back to the legacy engine"


def test_resolve_engine_default_is_legacy() -> None:
    # default is legacy (interim, pre-refactoring). dag-ml stays fully selectable via engine="dag-ml".
    assert resolve_engine(None) == "legacy"
    assert resolve_engine("dag-ml") == "dag-ml"
    assert resolve_engine("legacy") == "legacy"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dispatches_to_dagml_engine_native() -> None:
    """`engine="dag-ml"` runs a SUPPORTED shape (transforms + KFold + one model) NATIVELY on the
    dag-ml backend — proving the public API dispatched to it and produced a real RunResult, with NO
    cutover fallback to legacy (no warning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert not any(_FALLBACK_WARNING in str(w.message) for w in caught)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dagml_falls_back_to_legacy_on_unsupported_shape() -> None:
    """`engine="dag-ml"` with a no-splitter shape is a catchable coverage-boundary case
    (DagMlUnsupported, a NotImplementedError subclass): with the cutover fallback wired it must NOT
    raise — it warns and re-runs on the legacy engine, returning a valid legacy RunResult."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [{"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert any(_FALLBACK_WARNING in str(w.message) for w in caught)


def test_run_dagml_propagates_non_catchable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback catches ONLY DagMlUnsupported/NotImplementedError/DagMlUnavailable. A genuine bug
    in the dag-ml path (any other exception) MUST still propagate — never swallowed into a legacy run."""
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _boom(*_args: object, **_kwargs: object) -> RunResult:
        raise RuntimeError("genuine dag-ml backend bug")

    monkeypatch.setattr(run_backend, "run_via_dagml", _boom)
    with pytest.raises(RuntimeError, match="genuine dag-ml backend bug"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")


def test_dagml_run_uses_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``engine="dag-ml"`` run() routes to the dag-ml backend, in-process by default
    (unset N4A_DAGML_INPROCESS). Asserted by capturing the dag-ml dispatch + the in-process selection,
    so no real campaign/CLI is needed. (dag-ml is selected explicitly: the production DEFAULT is now
    legacy — interim, pre-refactoring — so a plain run() would route to legacy instead.)"""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.dagml.in_process_runner import in_process_enabled

    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    assert in_process_enabled() is True  # in-process is the default mechanism

    captured: dict[str, bool] = {"dagml": False}
    marker = RunResult(predictions=Predictions(), per_dataset={})

    def _fake_dagml(*_args: object, **_kwargs: object) -> RunResult:
        captured["dagml"] = True
        return marker

    monkeypatch.setattr(run_backend, "run_via_dagml", _fake_dagml)
    result = nirs4all.run([SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")
    assert captured["dagml"] is True
    assert result is marker


def test_run_forwards_dagml_training_losses_without_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public process-local custom losses are a DAG-ML-only contract: run() forwards the native role
    references and local registry, and a DAG-ML refusal is not hidden by legacy fallback."""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported

    role = {"schema_version": 1, "node_id": "model:compat.0", "output_id": "oof", "phases": ["FIT_CV", "REFIT"], "loss": {"loss_id": "example.loss@1"}}
    registry = object()
    marker = RunResult(predictions=Predictions(), per_dataset={})
    captured: dict[str, object] = {}

    def _fake_dagml(*_args: object, **kwargs: object) -> RunResult:
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(run_backend, "run_via_dagml", _fake_dagml)
    result = nirs4all.run(
        [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
        dataset_path("regression"),
        engine="dag-ml",
        training_losses=(role,),
        local_implementations=registry,
    )

    assert result is marker
    assert captured["training_losses"] == (role,)
    assert captured["local_implementations"] is registry

    def _unsupported(*_args: object, **_kwargs: object) -> RunResult:
        raise DagMlUnsupported("custom loss unsupported")

    monkeypatch.setattr(run_backend, "run_via_dagml", _unsupported)
    with pytest.raises(DagMlUnsupported, match="custom loss unsupported"):
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            training_losses=(role,),
            local_implementations=registry,
        )


def test_run_rejects_process_local_losses_on_legacy_engine() -> None:
    role = {"schema_version": 1, "node_id": "model:compat.0", "output_id": "oof", "phases": ["FIT_CV"], "loss": {"loss_id": "example.loss@1"}}

    with pytest.raises(ValueError, match="require engine='dag-ml'"):
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="legacy",
            training_losses=(role,),
            local_implementations=object(),
        )


def test_run_rejects_process_local_losses_on_native_tuning_bypass() -> None:
    role = {"schema_version": 1, "node_id": "model:compat.0", "output_id": "oof", "phases": ["FIT_CV"], "loss": {"loss_id": "example.loss@1"}}

    with pytest.raises(NotImplementedError, match="does not yet thread DAG-ML process-local training losses"):
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            tuning={"space": {}, "score_data": {"X": [[1.0, 2.0]], "y": [1.0]}},
            training_losses=(role,),
            local_implementations=object(),
        )


def test_dagml_run_falls_back_to_legacy_when_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``engine="dag-ml"`` run() transparently falls back to the LEGACY engine WITH A
    WARNING when the dag-ml backend is unavailable — simulated by the preflight raising
    DagMlUnavailable. The result is a valid legacy RunResult, never an exception. (dag-ml is selected
    explicitly: the production DEFAULT is now legacy — interim, pre-refactoring.)"""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.pipeline.dagml.errors import DagMlUnavailable

    monkeypatch.delenv("N4A_ENGINE", raising=False)

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated: neither in-process extension nor dag-ml-cli")

    # The preflight runs at the top of run_via_dagml; patching it makes the dag-ml path declare the
    # backend unavailable, exercising the run() DagMlUnavailable -> legacy fallback.
    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert any("dag-ml backend is not available" in str(w.message) for w in caught)


@pytest.mark.parametrize(
    "option",
    [
        pytest.param({"refit": False}, id="refit-disabled"),
        pytest.param({"project": "adr17_fallback_proj"}, id="project-tag"),
        pytest.param({"workspace_path": None}, id="workspace_path-runner-kwarg"),
        pytest.param({"cache": CacheConfig()}, id="cache-config"),
    ],
)
@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_dagml_run_unsupported_run_option_falls_back_to_legacy(option: dict[str, object]) -> None:
    """Under an explicit ``engine="dag-ml"``, run() options the scores-only dag-ml path cannot honor
    (non-default ``refit``, ``project`` tag, a persistence ``runner_kwarg`` like ``workspace_path``)
    must REJECT -> fall back to legacy (P1b), warning + valid result. Pins this reject-then-fallback for
    the non-run-able options on the dag-ml path. (dag-ml is selected explicitly: the production DEFAULT
    is now legacy — interim, pre-refactoring.)

    A workspace_path of ``None`` (legacy's own default) is still a PRESENT runner_kwarg the dag-ml path
    does not honor, so it triggers the same generic-key fallback — exercising the path without writing a
    throwaway workspace dir."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            **option,  # type: ignore[arg-type]
        )
    assert isinstance(result, RunResult)
    assert any(_FALLBACK_WARNING in str(w.message) for w in caught)


def test_dagml_rejects_session_and_cache() -> None:
    """Pins the session + cache REJECTION that drives run()'s legacy fallback on the dag-ml path:
    the scores-only dag-ml path cannot share a Session's runner/workspace nor install a CacheConfig on a
    runner it never builds, so a non-None ``session`` OR ``cache`` raises ``DagMlUnsupported`` (which
    run() catches -> legacy, the generic reject->legacy path the parametrized test above runs e2e)."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml.run_backend import _reject_unsupported_run_options

    with pytest.raises(DagMlUnsupported):
        _reject_unsupported_run_options(refit=True, project=None, session=object(), cache=None, runner_kwargs={})
    with pytest.raises(DagMlUnsupported):
        _reject_unsupported_run_options(refit=True, project=None, session=None, cache=CacheConfig(), runner_kwargs={})


def test_run_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="bogus")


@pytest.mark.parametrize(
    ("metric", "winner_index"),
    [
        ("balanced_accuracy", 1),  # maximize → highest score wins (the #60 regression fix)
        ("accuracy", 1),  # maximize
        ("r2", 1),  # maximize
        ("rmse", 0),  # minimize → lowest score wins
    ],
)
def test_run_repetition_selects_by_metric_direction(monkeypatch: pytest.MonkeyPatch, metric: str, winner_index: int) -> None:
    """`_run_repetition`'s multi-variant winner uses the CANONICAL metric direction (core.metrics).

    Locks #60 MUST-FIX 1: ``balanced_accuracy`` (the classification default) is HIGHER-is-better, so the
    highest-CV variant must win. The old ``metric in ("accuracy", "r2")`` set excluded balanced_accuracy
    and would have MINIMIZED it (picking the worst variant). Drives the selection directly with two stub
    variants (scores 0.10 / 0.90) so it asserts the direction without needing a repetition fixture."""
    from nirs4all.pipeline.config import generator
    from nirs4all.pipeline.dagml import run_paths

    class _StubResult:
        def __init__(self, score: float) -> None:
            self._score = score

        @property
        def cv_best_score(self) -> float:
            return self._score

    stub_results = [_StubResult(0.10), _StubResult(0.90)]
    variants = ["variant_a", "variant_b"]
    # `_run_repetition` imports `expand_spec` function-locally, so patch it at its source module.
    monkeypatch.setattr(generator, "expand_spec", lambda _pipeline: variants)

    def _fake_concrete(variant, *_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        return stub_results[variants.index(variant)]

    monkeypatch.setattr(run_paths, "_run_repetition_concrete", _fake_concrete)

    selected = run_paths._run_repetition(
        ["dummy-pipeline"], spectro=None, dataset_arg="", cli="", venv_python="", run_dir=Path("."), metric=metric, task_type="classification"
    )
    assert selected is stub_results[winner_index], (metric, selected._score)

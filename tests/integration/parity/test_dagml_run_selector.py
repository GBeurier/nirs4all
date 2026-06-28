"""The ADR-17 engine selector is wired into the public run() entry point (cutover FLIPPED).

These assert the public API resolves the engine before execution — since the ADR-17 cutover the
DEFAULT engine is dag-ml (in-process by default), ``engine="legacy"`` forces the legacy orchestrator,
and an unknown engine is rejected. They also certify the TRANSPARENT legacy fallback: a supported
shape runs NATIVE on dag-ml; a catchable unsupported shape (DagMlUnsupported/NotImplementedError) OR
an unavailable backend (DagMlUnavailable — neither in-process extension nor dag-ml-cli) falls back to
legacy (warning + valid result) instead of raising. A GENUINE dag-ml bug still propagates untouched.
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


def test_resolve_engine_default_is_dagml() -> None:
    # ADR-17 cutover: the default engine is now dag-ml (was legacy). engine="legacy" still forces legacy.
    assert resolve_engine(None) == "dag-ml"
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


def test_default_run_uses_dagml_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DEFAULT run() (no engine arg, no $N4A_ENGINE) routes to the dag-ml backend, in-process by
    default (unset N4A_DAGML_INPROCESS). Asserted by capturing the dag-ml dispatch + the in-process
    selection, so no real campaign/CLI is needed."""
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
    result = nirs4all.run([SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}], dataset_path("regression"))
    assert captured["dagml"] is True
    assert result is marker


def test_default_run_falls_back_to_legacy_when_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DEFAULT run() (engine resolves to dag-ml) transparently falls back to the LEGACY engine WITH
    A WARNING when the dag-ml backend is unavailable — simulated by the preflight raising
    DagMlUnavailable. The result is a valid legacy RunResult, never an exception."""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.pipeline.dagml.errors import DagMlUnavailable

    monkeypatch.delenv("N4A_ENGINE", raising=False)

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated: neither in-process extension nor dag-ml-cli")

    # The preflight runs at the top of run_via_dagml; patching it makes the default dag-ml path declare
    # the backend unavailable, exercising the run() DagMlUnavailable -> legacy fallback.
    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
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
def test_default_run_unsupported_run_option_falls_back_to_legacy(option: dict[str, object]) -> None:
    """Under the new dag-ml DEFAULT (no engine arg), run() options the scores-only dag-ml path cannot
    honor (non-default ``refit``, ``project`` tag, a persistence ``runner_kwarg`` like
    ``workspace_path``) must REJECT -> fall back to legacy (P1b), warning + valid result. Pins that the
    ADR-17 cutover preserved this reject-then-fallback for the non-run-able options on the new default.

    A workspace_path of ``None`` (legacy's own default) is still a PRESENT runner_kwarg the dag-ml path
    does not honor, so it triggers the same generic-key fallback — exercising the path without writing a
    throwaway workspace dir."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            **option,  # type: ignore[arg-type]
        )
    assert isinstance(result, RunResult)
    assert any(_FALLBACK_WARNING in str(w.message) for w in caught)


def test_dagml_rejects_session_and_cache_under_default() -> None:
    """Pins the session + cache REJECTION that drives run()'s legacy fallback under the dag-ml default:
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

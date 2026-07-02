"""The engine selector is wired into the public run() entry point.

These assert the public API resolves the engine before execution — the DEFAULT engine is now dag-ml
(V1 cutover), while the legacy orchestrator remains available only via ``engine="legacy"`` /
``$N4A_ENGINE=legacy``. An unknown engine is rejected. They also certify the explicit fallback policy on
the dag-ml path: a supported shape runs NATIVE on dag-ml by default; a catchable unsupported shape
(DagMlUnsupported/NotImplementedError) OR an unavailable backend (DagMlUnavailable — neither in-process
extension nor dag-ml-cli) raises a structured ``RtError`` unless ``allow_fallback=True`` is passed. A
GENUINE dag-ml bug still propagates untouched.
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
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.engine import resolve_engine

pytestmark = [pytest.mark.parity]

from . import _conformance_helpers as H  # noqa: E402
from ._dagml_cli import dagml_cli_path  # noqa: E402
from ._datasets import dataset_path  # noqa: E402
from ._registry import PipelineCase, all_cases  # noqa: E402

_DAGML_CLI = dagml_cli_path()

_FALLBACK_WARNING = "falling back to the legacy engine"


def _case(name: str) -> PipelineCase:
    return next(c for c in all_cases() if c.name == name)


def test_resolve_engine_default_is_dagml(monkeypatch: pytest.MonkeyPatch) -> None:
    # V1 default is dag-ml; legacy is now an explicit compatibility selection.
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    assert resolve_engine(None) == "dag-ml"
    assert resolve_engine("  DAG-ML  ") == "dag-ml"
    assert resolve_engine("dag-ml") == "dag-ml"
    assert resolve_engine("legacy") == "legacy"


def test_runtime_dagml_cli_discovery_matches_parity_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nirs4all.pipeline.dagml.run_backend import _default_dagml_cli

    monkeypatch.delenv("N4A_DAGML_CLI", raising=False)
    assert _default_dagml_cli() == dagml_cli_path()

    override = tmp_path / "dag-ml-cli"
    monkeypatch.setenv("N4A_DAGML_CLI", str(override))
    assert _default_dagml_cli() == override


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_default_run_dispatches_to_dagml_engine_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain ``run()`` runs a supported shape NATIVELY on dag-ml with no legacy fallback warning."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert result._is_dagml_engine() is True  # noqa: SLF001
    assert not any(_FALLBACK_WARNING in str(w.message) for w in caught)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_default_run_matches_legacy_on_representative_conformance_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Representative dual-engine selector: default ``run()`` reaches dag-ml and matches legacy parity."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    case = _case("baseline_vertical_slice")
    dataset = H.make_dataset(case)
    legacy = nirs4all.run(pipeline=case.pipeline, dataset=dataset, verbose=0, engine="legacy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default = nirs4all.run(pipeline=case.pipeline, dataset=dataset, verbose=0)
    assert default._is_dagml_engine() is True  # noqa: SLF001
    assert not any(_FALLBACK_WARNING in str(w.message) for w in caught)
    H.assert_score_parity(legacy, default, case)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dagml_rejects_unsupported_shape_by_default() -> None:
    """A no-splitter dag-ml shape raises structured ``RtError`` by default instead of degrading to legacy."""
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(
            [{"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert excinfo.value.cause == "unsupported_shape"
    assert excinfo.value.verb == "run"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dagml_falls_back_to_legacy_on_unsupported_shape_when_allowed() -> None:
    """The compatibility fallback still exists, but only when ``allow_fallback=True`` is explicit."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [{"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            allow_fallback=True,
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert any(_FALLBACK_WARNING in str(w.message) for w in caught)
    rt = result.to_rt_result()
    assert rt.manifest["engine"] == "legacy"
    assert [d.cause for d in rt.diagnostics] == ["unsupported_shape"]


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
    """A default ``run()`` routes to the dag-ml backend, in-process by default
    (unset N4A_DAGML_INPROCESS). Asserted by capturing the dag-ml dispatch + the in-process selection,
    so no real campaign/CLI is needed."""
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


def test_dagml_run_rejects_unavailable_backend_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing dag-ml backend raises ``RtError(cause='unavailable_backend')`` by default."""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.pipeline.dagml.errors import DagMlUnavailable

    monkeypatch.delenv("N4A_ENGINE", raising=False)

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated: neither in-process extension nor dag-ml-cli")

    # The preflight runs at the top of run_via_dagml; patching it makes the dag-ml path declare the
    # backend unavailable, exercising the run() DagMlUnavailable -> RtError policy.
    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)

    with pytest.raises(RtError) as excinfo:
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert excinfo.value.cause == "unavailable_backend"
    assert "simulated: neither in-process extension nor dag-ml-cli" in excinfo.value.message


def test_dagml_run_falls_back_to_legacy_when_backend_unavailable_and_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing dag-ml backend falls back to legacy only with explicit ``allow_fallback=True``."""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.pipeline.dagml.errors import DagMlUnavailable

    monkeypatch.delenv("N4A_ENGINE", raising=False)

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated: neither in-process extension nor dag-ml-cli")

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            allow_fallback=True,
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert any("dag-ml backend is not available" in str(w.message) for w in caught)
    assert [d.cause for d in result.to_rt_result().diagnostics] == ["unavailable_backend"]


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
def test_dagml_run_unsupported_run_option_refuses_by_default(option: dict[str, object]) -> None:
    """Under an explicit ``engine="dag-ml"``, run() options the scores-only dag-ml path cannot honor
    (non-default ``refit``, ``project`` tag, a persistence ``runner_kwarg`` like ``workspace_path``) must
    REJECT -> raise ``RtError`` by default. Pins the no-silent-degrade boundary for non-run-able options
    on the dag-ml path.

    A workspace_path of ``None`` (legacy's own default) is still a PRESENT runner_kwarg the dag-ml path
    does not honor, so it triggers the same generic-key refusal — exercising the path without writing a
    throwaway workspace dir."""
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            **option,  # type: ignore[arg-type]
        )
    assert excinfo.value.cause == "unsupported_shape"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_dagml_run_unsupported_run_option_falls_back_to_legacy_when_allowed() -> None:
    """Unsupported dag-ml run options degrade to legacy only under explicit ``allow_fallback=True``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            refit=False,
            allow_fallback=True,
        )
    assert isinstance(result, RunResult)
    assert any(_FALLBACK_WARNING in str(w.message) for w in caught)
    assert [d.cause for d in result.to_rt_result().diagnostics] == ["unsupported_shape"]


def test_dagml_rejects_session_and_cache() -> None:
    """Pins the session + cache REJECTION that drives run()'s RtError/fallback policy on the dag-ml path:
    the scores-only dag-ml path cannot share a Session's runner/workspace nor install a CacheConfig on a
    runner it never builds, so a non-None ``session`` OR ``cache`` raises ``DagMlUnsupported`` (which
    run() catches -> RtError by default, or legacy only with explicit fallback)."""
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

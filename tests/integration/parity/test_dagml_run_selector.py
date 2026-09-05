"""The engine selector is wired into the public run() entry point.

These assert the public API resolves the engine before execution. Automatic selection chooses the strict
Archive V2/Methods ``native`` path only for its portable subset and otherwise chooses the general
``dag-ml`` profile. The legacy orchestrator remains available only through explicit ``engine="legacy"``.
Unsupported or unavailable selected paths fail closed: ``allow_fallback=True`` never runs legacy.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.config import CacheConfig
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.engine import ExecutionProfileError, resolve_engine

pytestmark = [pytest.mark.parity]

from . import _conformance_helpers as H  # noqa: E402
from ._dagml_cli import dagml_cli_path  # noqa: E402
from ._datasets import dataset_path  # noqa: E402
from ._registry import PipelineCase, all_cases  # noqa: E402

_DAGML_CLI = dagml_cli_path()

def _case(name: str) -> PipelineCase:
    return next(c for c in all_cases() if c.name == name)


def test_resolve_engine_default_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    # V1 default is native; legacy is now an explicit compatibility selection.
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    assert resolve_engine(None) == "native"
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


def test_default_general_run_dispatches_to_dagml(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-portable plain ``run()`` selects the general DAG without constructing the legacy runner."""
    run_module = importlib.import_module("nirs4all.api.run")
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    run_backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")
    sentinel = object()

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("default general DAG run constructed PipelineRunner")

    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(native_module, "run_native_methods_archive", lambda *_args, **_kwargs: pytest.fail("general request selected Archive V2"))
    monkeypatch.setattr(run_backend, "run_via_dagml", lambda *_args, **_kwargs: sentinel)
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    assert nirs4all.run([], {}) is sentinel


def test_default_general_dataset_selects_dagml_without_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DatasetConfigs request selects dag-ml directly, never by post-failure fallback."""
    run_module = importlib.import_module("nirs4all.api.run")
    run_backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")
    sentinel = object()

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("default general DAG run constructed PipelineRunner")

    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(run_backend, "run_via_dagml", lambda *_args, **_kwargs: sentinel)
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    case = _case("baseline_vertical_slice")
    dataset = H.make_dataset(case)
    assert nirs4all.run(pipeline=case.pipeline, dataset=dataset) is sentinel


def test_run_dagml_supports_full_train_without_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no-splitter model executes the explicit native full-train profile."""
    run_backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")
    full_train = importlib.import_module("nirs4all.pipeline.dagml.full_train")
    calls: list[list[object]] = []
    sentinel = RunResult(predictions=Predictions(), per_dataset={"fixture": {"engine": "dag-ml"}})

    def _full_train(pipeline: list[object], *_args: object, **_kwargs: object) -> RunResult:
        calls.append(pipeline)
        return sentinel

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", lambda _cli: None)
    monkeypatch.setattr(full_train, "run_full_train", _full_train)
    result = nirs4all.run(
        [{"model": PLSRegression(n_components=2)}],
        dataset_path("regression"),
        engine="dag-ml",
    )
    assert result is sentinel
    assert len(calls) == 1


def test_run_dagml_rejects_legacy_fallback_on_unsupported_shape() -> None:
    """An explicit fallback request is rejected before a dag-ml or legacy dispatch."""
    with pytest.raises(ExecutionProfileError) as caught:
        nirs4all.run(
            [{"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            allow_fallback=True,
        )
    assert caught.value.code == "legacy_fallback_forbidden"


def test_run_dagml_propagates_non_catchable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback catches ONLY DagMlUnsupported/NotImplementedError/DagMlUnavailable. A genuine bug
    in the dag-ml path (any other exception) MUST still propagate — never swallowed into a legacy run."""
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _boom(*_args: object, **_kwargs: object) -> RunResult:
        raise RuntimeError("genuine dag-ml backend bug")

    monkeypatch.setattr(run_backend, "run_via_dagml", _boom)
    with pytest.raises(RuntimeError, match="genuine dag-ml backend bug"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")


def test_explicit_dagml_run_uses_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``engine='dag-ml'`` routes to the in-process backend by default
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
    result = nirs4all.run([SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")
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


def test_dagml_run_rejects_legacy_fallback_when_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback is rejected before backend availability can trigger a legacy dispatch."""
    import nirs4all.pipeline.dagml.run_backend as run_backend
    from nirs4all.pipeline.dagml.errors import DagMlUnavailable

    monkeypatch.delenv("N4A_ENGINE", raising=False)

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated: neither in-process extension nor dag-ml-cli")

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)

    with pytest.raises(ExecutionProfileError) as caught:
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            allow_fallback=True,
        )
    assert caught.value.code == "legacy_fallback_forbidden"


@pytest.mark.parametrize(
    "option",
    [
        pytest.param({"refit": False}, id="refit-disabled"),
        pytest.param({"cache": CacheConfig()}, id="cache-config"),
    ],
)
@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_dagml_run_unsupported_run_option_refuses_by_default(option: dict[str, object]) -> None:
    """Unsupported execution controls fail closed instead of selecting another engine."""
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            **option,  # type: ignore[arg-type]
        )
    assert excinfo.value.cause == "unsupported_shape"


@pytest.mark.parametrize("option_name", ["project", "workspace_path"])
def test_dagml_run_supports_persistence_options(option_name: str, tmp_path: Path) -> None:
    """Project and workspace persistence are honored by the general DAG profile."""
    option = {"project": "v1-project"} if option_name == "project" else {"workspace_path": tmp_path / "workspace"}
    with nirs4all.run(
        [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
        dataset_path("regression"),
        engine="dag-ml",
        **option,
    ) as result:
        assert result._is_dagml_engine()  # noqa: SLF001
        assert result.num_predictions > 0


def test_dagml_run_unsupported_option_rejects_legacy_fallback() -> None:
    """Unsupported options cannot opt back into a silent legacy rerun."""
    with pytest.raises(ExecutionProfileError) as caught:
        nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
            refit=False,
            allow_fallback=True,
        )
    assert caught.value.code == "legacy_fallback_forbidden"


def test_dagml_prepares_session_and_rejects_cache() -> None:
    """General sessions are prepared explicitly; unsupported StepCache remains a strict refusal."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml.run_backend import _reject_unsupported_run_options

    class SessionProbe:
        prepared = False

        def _prepare_dagml_run(self) -> None:
            self.prepared = True

    session = SessionProbe()
    _reject_unsupported_run_options(refit=True, project=None, session=session, cache=None, runner_kwargs={})
    assert session.prepared is True
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

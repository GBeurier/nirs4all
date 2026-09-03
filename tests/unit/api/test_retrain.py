"""API-004 retrain routing and fail-closed effect ordering."""

from __future__ import annotations

import importlib
import zipfile
from pathlib import Path
from typing import Any

import pytest

import nirs4all
import nirs4all.pipeline as pipeline_module
from nirs4all.api.retrain import retrain, retrain_preflight
from nirs4all.api.retrain_capabilities import (
    require_dagml_retrain_backend,
    retrain_capability_ledger,
)
from nirs4all.api.session import Session
from nirs4all.pipeline.bundle.loader import BundleMetadata
from nirs4all.pipeline.dagml.errors import DagMlUnavailable
from nirs4all.pipeline.dagml.rt import RtError


class _MustNotBeTouched:
    """Request sentinel whose common materialization operations all fail."""

    def __iter__(self):
        raise AssertionError("retrain materialized data before refusing")

    def __array__(self):
        raise AssertionError("retrain converted data before refusing")

    def __str__(self) -> str:
        raise AssertionError("retrain stringified input before refusing")


def _never_runner(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(f"PipelineRunner must not be constructed: {args!r} {kwargs!r}")


@pytest.mark.parametrize(
    ("kwargs", "capability"),
    [
        ({"mode": "transfer"}, "native_transfer_retrain"),
        ({"mode": "finetune"}, "native_finetune_retrain"),
        ({"mode": "full", "engine": "native"}, "core_archive_v3_retrain"),
        ({"mode": "full", "plugin": "hpo-controller"}, "retrain_plugin"),
        (
            {"mode": "finetune", "plugin": "nirs4all-python-library"},
            "retrain_plugin",
        ),
        ({"mode": "full", "allow_fallback": True}, "implicit_legacy_fallback"),
    ],
)
def test_unsupported_retrain_selectors_refuse_before_inputs_or_runner(
    kwargs: dict[str, Any],
    capability: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing refusals happen before source/data, backend, loaders, or writes."""
    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_RETRAIN_PLUGIN", raising=False)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(
        retrain_module,
        "require_dagml_retrain_backend",
        lambda: pytest.fail("unsupported selection probed the DAG-ML backend"),
    )
    monkeypatch.setattr(
        retrain_module,
        "_bundle_training_spec",
        lambda source: pytest.fail(f"unsupported selection inspected source: {source!r}"),
    )
    monkeypatch.setattr(
        retrain_module,
        "_native_full_retrain",
        lambda *args, **options: pytest.fail("unsupported selection entered native execution"),
    )

    with pytest.raises(RtError) as caught:
        retrain(_MustNotBeTouched(), _MustNotBeTouched(), **kwargs)

    assert caught.value.verb == "run"
    assert caught.value.cause == "unsupported_capability"
    assert caught.value.unsupported_capability == capability


def test_native_backend_failure_precedes_bundle_and_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend availability is checked before the source archive is opened."""
    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(
        retrain_module,
        "require_dagml_retrain_backend",
        lambda: (_ for _ in ()).throw(
            RtError("run", "unavailable_backend", "missing dag-ml")
        ),
    )
    monkeypatch.setattr(
        retrain_module,
        "_bundle_training_spec",
        lambda source: pytest.fail(f"unavailable backend inspected bundle: {source!r}"),
    )

    with pytest.raises(RtError) as caught:
        retrain("missing.n4a", _MustNotBeTouched())
    assert caught.value.cause == "unavailable_backend"


def test_backend_probe_maps_unavailability_to_typed_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real availability adapter never leaks its backend exception type."""
    backend_module = importlib.import_module("nirs4all.pipeline.dagml.run_backend")
    monkeypatch.setattr(backend_module, "_default_dagml_cli", lambda: Path("missing-cli"))
    monkeypatch.setattr(
        backend_module,
        "preflight_dagml_backend",
        lambda cli: (_ for _ in ()).throw(DagMlUnavailable(f"missing: {cli}")),
    )

    with pytest.raises(RtError) as caught:
        require_dagml_retrain_backend()

    assert caught.value.verb == "run"
    assert caught.value.cause == "unavailable_backend"


@pytest.mark.parametrize(
    "options",
    [
        {"refit": False},
        {"project": "workspace-project"},
        {"cache": object()},
        {"workspace_path": "legacy-workspace"},
    ],
)
def test_native_options_refuse_before_backend_bundle_or_data(
    options: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unhonored DAG-ML options cannot trigger source reads or computation."""
    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr(
        retrain_module,
        "require_dagml_retrain_backend",
        lambda: pytest.fail("invalid native option probed the backend"),
    )
    monkeypatch.setattr(
        retrain_module,
        "_bundle_training_spec",
        lambda source: pytest.fail(f"invalid native option inspected source: {source!r}"),
    )

    with pytest.raises(RtError) as caught:
        retrain(_MustNotBeTouched(), _MustNotBeTouched(), **options)

    assert caught.value.cause == "invalid_request"
    assert caught.value.unsupported_capability == "dagml_full_retrain_option"


def test_missing_training_spec_refuses_before_run_data_or_legacy_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prediction-only bundle cannot masquerade as native retrain support."""
    retrain_module = importlib.import_module("nirs4all.api.retrain")
    run_module = importlib.import_module("nirs4all.api.run")
    bundle_module = importlib.import_module("nirs4all.pipeline.bundle")
    bundle = tmp_path / "predict-only.n4a"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", "{}")

    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr(retrain_module, "require_dagml_retrain_backend", lambda: None)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_runner)
    monkeypatch.setattr(
        run_module,
        "run",
        lambda *args, **kwargs: pytest.fail("missing train spec reached compute or result writes"),
    )

    with pytest.raises(RtError) as caught:
        retrain(bundle, _MustNotBeTouched())

    assert caught.value.cause == "invalid_request"
    assert caught.value.unsupported_capability == "dagml_full_retrain_training_spec"


def test_explicit_legacy_preserves_pipeline_runner_rollback_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen rollback selector preserves the historical retrainer contract."""
    observed: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            observed["constructor"] = kwargs

        def retrain(self, **kwargs: Any) -> tuple[object, dict[str, Any]]:
            observed["retrain"] = kwargs
            return object(), {"legacy": {"engine": "legacy"}}

    data = object()
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _FakeRunner)
    result = retrain(
        {"model_name": "PLS"},
        data,
        mode="transfer",
        engine="legacy",
        new_model="new-estimator",
        verbose=0,
        save_artifacts=False,
        learning_rate=0.1,
    )

    assert result.per_dataset == {"legacy": {"engine": "legacy"}}
    assert observed["constructor"] == {"verbose": 0, "save_artifacts": False}
    assert observed["retrain"]["dataset"] is data
    assert observed["retrain"]["new_model"] == "new-estimator"
    assert observed["retrain"]["learning_rate"] == 0.1
    assert "engine" not in observed["retrain"]


def test_transfer_executes_only_through_explicit_python_library_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CAP-001 transfer plugin disposition is a callable bounded route."""
    observed: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            observed["constructor"] = kwargs

        def retrain(self, **kwargs: Any) -> tuple[object, dict[str, Any]]:
            observed["retrain"] = kwargs
            return object(), {"plugin": {"id": "nirs4all-python-library"}}

    monkeypatch.setattr(pipeline_module, "PipelineRunner", _FakeRunner)
    source = {"model_name": "PLS"}
    data = object()
    result = retrain(
        source,
        data,
        mode="transfer",
        plugin="nirs4all-python-library",
        new_model="new-estimator",
        verbose=0,
        save_artifacts=False,
    )

    assert result.per_dataset == {"plugin": {"id": "nirs4all-python-library"}}
    assert observed["constructor"] == {"verbose": 0, "save_artifacts": False}
    assert observed["retrain"]["source"] is source
    assert observed["retrain"]["dataset"] is data
    assert observed["retrain"]["mode"] == "transfer"
    assert observed["retrain"]["new_model"] == "new-estimator"
    assert "plugin" not in observed["retrain"]


def test_preflight_and_capability_ledger_are_honest_and_detached() -> None:
    """Discovery reports the concrete native, plugin, and legacy adapters."""
    assert nirs4all.retrain.preflight is retrain_preflight
    native_full = retrain_preflight()
    assert native_full.lane == "dag-ml"
    assert native_full.executable is True
    assert native_full.contract == "nirs4all.bundle.train_pipeline.v1+dag-ml.run"

    ledger = retrain_capability_ledger()
    assert ledger["full"]["dag-ml"]["executable"] is True
    assert ledger["full"]["native"]["executable"] is False
    assert ledger["transfer"]["dag-ml"]["executable"] is False
    assert ledger["transfer"]["plugin"] == {
        "executable": True,
        "contract": "nirs4all.python-library.retrain-transfer.v1",
        "capability": "retrain_plugin",
    }
    assert ledger["finetune"]["plugin"]["executable"] is False
    ledger["full"]["dag-ml"]["executable"] = False
    assert retrain_capability_ledger()["full"]["dag-ml"]["executable"] is True


def test_bundle_metadata_rejects_non_object_retrain_lineage() -> None:
    """The additive bundle provenance remains a validated JSON object."""
    with pytest.raises(ValueError, match="retrain_lineage provenance"):
        BundleMetadata.from_dict({"retrain_lineage": ["not", "an", "object"]})


def test_session_refuses_before_constructing_runner_or_reading_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session.retrain uses the same side-effect-free selector first."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    session = Session()

    with pytest.raises(RtError) as caught:
        session.retrain(_MustNotBeTouched(), mode="finetune")

    assert caught.value.unsupported_capability == "native_finetune_retrain"
    assert session._runner is None


def test_public_retrain_refuses_native_session_sharing_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module-level native call cannot silently borrow a Python Session."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    session = Session()

    with pytest.raises(RtError) as caught:
        retrain(_MustNotBeTouched(), _MustNotBeTouched(), session=session)

    assert caught.value.unsupported_capability == "native_retrain_session"
    assert session._runner is None

"""Unit tests for the ADR-17 dag-ml mechanism cascade (in-process default + availability preflight).

These pin the SELECTION/AVAILABILITY logic without running a real dag-ml campaign:

* :func:`in_process_enabled` — UNSET means in-process (the ADR-17 default); ``0``/``false``/``off``
  (case-insensitive) force subprocess; any other value is in-process.
* :func:`run_cv_refit_bundle_router` — the cascade: in-process when selected AND the extension loads;
  fall through to subprocess when the extension does NOT load; raise :class:`DagMlUnavailable` when the
  subprocess branch is reached but the ``dag-ml-cli`` binary is missing too (the dag-ml-cli existence
  check lives in the SUBPROCESS branch only).
* :func:`preflight_dagml_backend` — passes when EITHER mechanism is present; raises
  :class:`DagMlUnavailable` when NEITHER is.
"""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml import in_process_runner
from nirs4all.pipeline.dagml.errors import DagMlUnavailable
from nirs4all.pipeline.dagml.in_process_runner import in_process_enabled, run_cv_refit_bundle_router


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, True),  # ADR-17 cutover: UNSET means in-process ENABLED (the default).
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("off", False),
        ("  Off  ", False),
        ("1", True),
        ("true", True),
        ("on", True),
        ("anything", True),  # any non-{0,false,off} value is in-process.
    ],
)
def test_in_process_enabled_semantics(monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool) -> None:
    if env_value is None:
        monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    else:
        monkeypatch.setenv("N4A_DAGML_INPROCESS", env_value)
    assert in_process_enabled() is expected


def _router_kwargs(workdir: object = "unused", dagml_cli: str = "/nonexistent/dag-ml-cli") -> dict[str, object]:
    """Minimal kwargs for the router — the branches short-circuit before consuming the payload."""
    return {
        "dsl": {},
        "envelope": {},
        "graph": {"nodes": [], "edges": []},
        "dataset_path": "unused",
        "workdir": workdir,
        "dagml_cli": dagml_cli,
        "venv_python": "python",
    }


def test_router_picks_in_process_when_selected_and_extension_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """in-process selected (default) + extension loads -> the in-process runner is called (NOT subprocess)."""
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)  # default: in-process selected
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: True)
    sentinel = {"scores": {"reports": []}, "results": [], "returncode": 0, "stdout": ""}
    called: dict[str, bool] = {"in_process": False}

    def _fake_in_process(**_kwargs: object) -> dict[str, object]:
        called["in_process"] = True
        return sentinel

    monkeypatch.setattr(in_process_runner, "run_cv_refit_bundle", _fake_in_process)
    outcome = run_cv_refit_bundle_router(**_router_kwargs())
    assert called["in_process"] is True
    assert outcome is sentinel


def test_router_falls_through_to_subprocess_when_extension_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """in-process selected but the extension does NOT load -> subprocess branch (needs the CLI).

    The CLI here exists (real temp file), so the subprocess runner is driven — proving the
    extension-load failure reroutes to subprocess rather than crashing or staying in-process.
    """
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)  # in-process selected ...
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: False)  # ... but unavailable
    cli = tmp_path / "dag-ml-cli"
    cli.write_text("#!/bin/sh\n")
    workdir = tmp_path / "wd"
    workdir.mkdir()

    import nirs4all.pipeline.dagml.cli_runner as cli_runner

    def _fake_subprocess(**_kwargs: object) -> dict[str, object]:
        return {"returncode": 0, "stdout": "", "results": []}

    monkeypatch.setattr(cli_runner, "run_cv_refit_bundle", _fake_subprocess)
    # bundle.json is absent in this stub run, so scores resolve to None (returncode 0, no file).
    outcome = run_cv_refit_bundle_router(**_router_kwargs(workdir=workdir, dagml_cli=str(cli)))
    assert outcome["returncode"] == 0
    assert outcome["scores"] is None


def test_router_raises_unavailable_when_neither_mechanism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extension does not load AND the dag-ml-cli binary is missing -> DagMlUnavailable (the CLI check
    lives in the subprocess branch)."""
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: False)
    with pytest.raises(DagMlUnavailable, match="dag-ml-cli"):
        run_cv_refit_bundle_router(**_router_kwargs(dagml_cli="/nonexistent/dag-ml-cli"))


def test_router_subprocess_forced_off_does_not_require_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """N4A_DAGML_INPROCESS=off forces subprocess even when the extension WOULD load; it needs the CLI."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "off")
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: True)  # would load, but forced off
    with pytest.raises(DagMlUnavailable, match="dag-ml-cli"):
        run_cv_refit_bundle_router(**_router_kwargs(dagml_cli="/nonexistent/dag-ml-cli"))


def test_preflight_passes_when_extension_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """preflight: extension loads -> available even with NO dag-ml-cli binary (in-process needs none)."""
    from nirs4all.pipeline.dagml.run_backend import preflight_dagml_backend

    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: True)
    preflight_dagml_backend("/nonexistent/dag-ml-cli")  # must NOT raise


def test_preflight_passes_when_only_cli_present(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """preflight: extension missing but the dag-ml-cli binary exists -> available (subprocess path)."""
    from nirs4all.pipeline.dagml.run_backend import preflight_dagml_backend

    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: False)
    cli = tmp_path / "dag-ml-cli"
    cli.write_text("#!/bin/sh\n")
    preflight_dagml_backend(str(cli))  # must NOT raise


def test_preflight_raises_when_neither_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """preflight: NEITHER mechanism -> DagMlUnavailable (the one signal run() turns into legacy)."""
    from nirs4all.pipeline.dagml.run_backend import preflight_dagml_backend

    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: False)
    with pytest.raises(DagMlUnavailable, match="not available"):
        preflight_dagml_backend("/nonexistent/dag-ml-cli")

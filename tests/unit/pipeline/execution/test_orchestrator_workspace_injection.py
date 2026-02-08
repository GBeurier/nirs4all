from pathlib import Path

import pytest

from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator


def test_orchestrator_requires_explicit_workspace_path() -> None:
    with pytest.raises(ValueError, match="workspace_path must be provided explicitly"):
        PipelineOrchestrator()


def test_orchestrator_accepts_explicit_workspace_path(tmp_path: Path) -> None:
    orchestrator = PipelineOrchestrator(workspace_path=tmp_path / "workspace")
    assert orchestrator.workspace_path == (tmp_path / "workspace")

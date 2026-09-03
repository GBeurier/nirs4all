"""Contract tests for the release publication workflow."""

from pathlib import Path
from typing import Any

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[3] / ".github/workflows/publish.yml"


def _load_workflow() -> dict[str, Any]:
    # BaseLoader keeps GitHub's ``on`` key as a string instead of YAML 1.1 bool.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_manual_dispatch_is_build_only_and_release_publication_is_verified() -> None:
    workflow = _load_workflow()
    assert set(workflow["on"]) == {"release", "workflow_dispatch"}

    jobs = workflow["jobs"]
    verification_steps = [
        step
        for step in jobs["build"]["steps"]
        if step.get("name") == "Verify version consistency"
    ]
    assert len(verification_steps) == 1
    verification = verification_steps[0]
    assert verification["if"] == "github.event_name == 'release'"
    assert "github.event.release.tag_name" in verification["run"]
    assert 'if [[ "$PKG_VERSION" != "$TAG_VERSION" ]]' in verification["run"]

    for job_name in ("publish-pypi", "publish-docker"):
        job = jobs[job_name]
        assert job["if"] == "github.event_name == 'release'"
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "build" in needs

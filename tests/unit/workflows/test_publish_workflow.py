"""Contract tests for the release publication workflow."""

from pathlib import Path
from typing import Any

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[3] / ".github/workflows/publish.yml"


def _load_workflow() -> dict[str, Any]:
    # BaseLoader keeps GitHub's ``on`` key as a string instead of YAML 1.1 bool.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_publication_requires_a_release_or_bounded_immutable_tag_repair() -> None:
    workflow = _load_workflow()
    assert set(workflow["on"]) == {"release", "workflow_dispatch"}
    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["publish"]["default"] == "false"
    assert dispatch_inputs["release_tag"]["default"] == ""

    jobs = workflow["jobs"]
    repair_steps = [
        step
        for step in jobs["run-tests"]["steps"]
        if step.get("name") == "Guard immutable-tag repair source"
    ]
    assert len(repair_steps) == 1
    repair = repair_steps[0]
    assert repair["if"] == "github.event_name == 'workflow_dispatch' && inputs.publish"
    assert 'test "${{ inputs.release_tag }}" = "0.13.0"' in repair["run"]
    assert "61a66d1bd0157dd9422facc4b32fca33989d4035" in repair["run"]
    assert "tests/unit/workflows/test_publish_workflow.py" in repair["run"]

    verification_steps = [
        step
        for step in jobs["build"]["steps"]
        if step.get("name") == "Verify version consistency"
    ]
    assert len(verification_steps) == 1
    verification = verification_steps[0]
    publication_condition = "github.event_name == 'release' || inputs.publish"
    assert verification["if"] == publication_condition
    assert "github.event.release.tag_name" in verification["run"]
    assert "inputs.release_tag" in verification["run"]
    assert 'if [[ "$PKG_VERSION" != "$TAG_VERSION" ]]' in verification["run"]

    for job_name in ("publish-pypi", "publish-docker"):
        job = jobs[job_name]
        assert job["if"] == publication_condition
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "build" in needs

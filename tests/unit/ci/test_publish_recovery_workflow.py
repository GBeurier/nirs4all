"""Guard manual publication of an already tagged release."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[3] / ".github/workflows/publish.yml"


def _workflow() -> dict:
    # BaseLoader preserves GitHub Actions' `on` key under YAML 1.1 parsers.
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_manual_publish_is_opt_in_and_requires_preflight() -> None:
    workflow = _workflow()
    dispatch = workflow["on"]["workflow_dispatch"]["inputs"]
    assert dispatch["publish_release"]["default"] == "false"
    assert "release_tag" in dispatch

    jobs = workflow["jobs"]
    preflight = next(
        step for step in jobs["release-preflight"]["steps"]
        if step.get("id") == "target"
    )["run"]
    assert "refs/heads/main" in preflight
    assert "releases/tags/$INPUT_TAG" in preflight
    assert "pypi.org/pypi/nirs4all/$package_version/json" in preflight
    assert "tag_sha" in jobs["release-preflight"]["outputs"]
    assert all(jobs[name]["needs"] == "release-preflight" for name in ("run-tests", "build-docs", "verify-examples"))
    assert "release-preflight" in jobs["build"]["needs"]
    assert "needs.release-preflight.outputs.release_tag" in jobs["build"]["steps"][0]["with"]["ref"]


def test_publication_still_depends_on_tested_tagged_distribution() -> None:
    jobs = _workflow()["jobs"]
    assert set(jobs["build"]["needs"]) >= {
        "release-preflight", "run-tests", "build-docs", "verify-examples",
    }
    assert jobs["publish-pypi"]["needs"] == "build"
    assert "inputs.publish_release" in jobs["publish-pypi"]["if"]
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-pypi"]["permissions"]["id-token"] == "write"

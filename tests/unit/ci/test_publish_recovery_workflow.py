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
    assert all(jobs[name]["needs"] == "release-preflight" for name in ("build-docs", "local-qualification"))
    assert set(jobs["package-smoke"]["needs"]) == {"release-preflight", "build"}
    assert "release-preflight" in jobs["build"]["needs"]
    assert "needs.release-preflight.outputs.release_tag" in jobs["build"]["steps"][0]["with"]["ref"]


def test_publication_still_depends_on_tested_tagged_distribution() -> None:
    jobs = _workflow()["jobs"]
    preflight = next(step for step in jobs["release-preflight"]["steps"] if step.get("id") == "target")["run"]
    assert 'if [[ "$GITHUB_SHA" != "$tag_sha" ]]' in preflight
    for name in ("build-docs", "build", "package-smoke", "local-qualification"):
        checkout = next(step for step in jobs[name]["steps"] if step.get("uses") == "actions/checkout@v6")
        assert "needs.release-preflight.outputs.release_tag" in checkout["with"]["ref"]
        assert any(
            "EXPECTED_TAG_SHA" in step.get("env", {}) and
            '"$(git rev-parse HEAD)"' in step.get("run", "") and
            '"$EXPECTED_TAG_SHA"' in step.get("run", "")
            for step in jobs[name]["steps"]
        )
    qualification = jobs["local-qualification"]
    assert qualification["if"] == "github.event_name == 'release' || inputs.publish_release"
    assert any(
        "scripts/verify_local_qualification.py --project sdk --receipt compat/local-qualification.json --root ." in step.get("run", "")
        for step in qualification["steps"]
    )
    assert set(jobs["publish-pypi"]["needs"]) == {"build", "build-docs", "package-smoke", "local-qualification"}
    assert "inputs.publish_release" in jobs["publish-pypi"]["if"]
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-pypi"]["permissions"]["id-token"] == "write"

    docker = jobs["publish-docker"]
    assert set(docker["needs"]) >= {"release-preflight", "build", "publish-pypi", "local-qualification"}
    assert "inputs.publish_release" in docker["if"]
    assert "needs.release-preflight.outputs.release_tag" in docker["steps"][0]["with"]["ref"]
    metadata = next(step for step in docker["steps"] if step.get("id") == "meta")
    assert "needs.release-preflight.outputs.docker_version" in metadata["with"]["tags"]
    assert "needs.release-preflight.outputs.prerelease == 'false'" in metadata["with"]["tags"]

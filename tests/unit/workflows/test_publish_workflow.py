"""Contract tests for the release publication workflow."""

import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[3] / ".github/workflows/publish.yml"
PYPROJECT_PATH = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _load_workflow() -> dict[str, Any]:
    # BaseLoader keeps GitHub's ``on`` key as a string instead of YAML 1.1 bool.
    return cast(dict[str, Any], yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader))


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
    assert 'TAG_VERSION="${TAG_VERSION//-rc./rc}"' in verification["run"]
    assert 'if [[ "$PKG_VERSION" != "$TAG_VERSION" ]]' in verification["run"]

    for job_name in ("publish-pypi", "publish-docker"):
        job = jobs[job_name]
        assert job["if"] == "github.event_name == 'release'"
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "build" in needs

    metadata_steps = [
        step
        for step in jobs["publish-docker"]["steps"]
        if step.get("uses") == "docker/metadata-action@v6"
    ]
    assert len(metadata_steps) == 1
    assert (
        "type=raw,value=latest,enable=${{ github.event.release.prerelease == false }}"
        in metadata_steps[0]["with"]["tags"]
    )


def test_release_metadata_closes_the_published_v1_stack_and_legal_files() -> None:
    """The base wheel owns Studio's full V1 runtime and dual-license notices."""

    with PYPROJECT_PATH.open("rb") as stream:
        pyproject = tomllib.load(stream)

    dependencies = set(pyproject["project"]["dependencies"])
    assert {
        "dag-ml>=0.3.25,<0.4",
        "dag-ml-data>=0.2.11,<0.3",
        "nirs4all-io>=0.1.18,<0.2",
        "nirs4all-core>=0.3.29,<0.4",
        "nirs4all-methods>=1.0.18,<2",
    } <= dependencies
    assert pyproject["project"]["license"] == "CeCILL-2.1 OR AGPL-3.0-or-later"
    assert set(pyproject["project"]["license-files"]) == {
        "LICENSE",
        "LICENSING.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/*",
    }

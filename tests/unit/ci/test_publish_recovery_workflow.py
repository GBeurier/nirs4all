"""Keep registry publication independent while preserving immutable source gates."""

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
WORKFLOW = ROOT / ".github/workflows/publish.yml"
DOCKER_WORKFLOW = ROOT / ".github/workflows/publish-docker.yml"


def _workflow(path: Path = WORKFLOW) -> dict:
    # BaseLoader preserves GitHub Actions' `on` key under YAML 1.1 parsers.
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _preflight() -> str:
    return next(step for step in _workflow()["jobs"]["release-preflight"]["steps"] if step.get("id") == "target")["run"]


def test_registry_publication_is_independent_of_product_releases_and_delivery() -> None:
    workflow = _workflow()
    assert workflow["on"]["push"]["tags"] == ["[0-9]*"]
    assert "release" not in workflow["on"]
    jobs = workflow["jobs"]
    assert "build-docs" not in jobs
    assert "publish-docker" not in jobs
    assert set(jobs["publish-pypi"]["needs"]) == {"release-preflight", "build", "package-smoke"}
    assert set(jobs["package-smoke"]["needs"]) == {"release-preflight", "build"}
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-pypi"]["permissions"]["id-token"] == "write"
    assert "publish_package == 'true'" in jobs["publish-pypi"]["if"]


def test_manual_publish_remains_opt_in_and_uses_an_immutable_tag() -> None:
    workflow = _workflow()
    dispatch = workflow["on"]["workflow_dispatch"]["inputs"]
    assert dispatch["publish_release"]["default"] == "false"
    assert "release_tag" in dispatch
    preflight = _preflight()
    assert "refs/heads/main" in preflight
    assert 'git merge-base --is-ancestor "$tag_sha" "$GITHUB_SHA"' in preflight
    assert '"$event_sha" != "$tag_sha"' in preflight
    assert "releases/tags/" not in preflight
    assert "pypi.org/pypi/nirs4all/$package_version/json" in preflight
    for name in ("build", "package-smoke"):
        job = workflow["jobs"][name]
        checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v6")
        assert "needs.release-preflight.outputs.tag_sha" in checkout["with"]["ref"]
        assert any("EXPECTED_TAG_SHA" in step.get("env", {}) and '"$(git rev-parse HEAD)"' in step.get("run", "") for step in job["steps"])


def test_default_manual_dry_run_cannot_publish_or_require_a_release(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    result = subprocess.run(["bash", "-c", _preflight()], env={**os.environ, "EVENT_NAME": "workflow_dispatch", "PUBLISH_RELEASE": "false", "GITHUB_OUTPUT": str(output)}, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "publish_package=false\n"
    assert "registry publication disabled" in result.stdout


def test_manual_publication_refuses_an_unreviewed_branch_before_network_access(tmp_path: Path) -> None:
    result = subprocess.run(["bash", "-c", _preflight()], env={**os.environ, "EVENT_NAME": "workflow_dispatch", "PUBLISH_RELEASE": "true", "GITHUB_REF": "refs/heads/unreviewed", "GITHUB_OUTPUT": str(tmp_path / "outputs")}, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "must run from main" in result.stdout


def test_manual_publication_refuses_arbitrary_refs_before_network_access(tmp_path: Path) -> None:
    result = subprocess.run(["bash", "-c", _preflight()], env={**os.environ, "EVENT_NAME": "workflow_dispatch", "PUBLISH_RELEASE": "true", "GITHUB_REF": "refs/heads/main", "INPUT_TAG": "../../main", "GITHUB_OUTPUT": str(tmp_path / "outputs")}, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "immutable SemVer tag" in result.stdout


def test_image_delivery_can_retry_after_registry_publication_and_qualifies_before_push() -> None:
    workflow = _workflow(DOCKER_WORKFLOW)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["on"]["workflow_dispatch"]["inputs"]["release_tag"]["required"] == "true"
    assert "publish-pypi" not in workflow["jobs"]
    steps = workflow["jobs"]["release-preflight"]["steps"]
    registry = next(step for step in steps if "already published" in step.get("name", ""))["run"]
    assert 'qualified["wheel"]' in registry
    assert 'matches[0]["digests"]["sha256"] == expected["sha256"]' in registry
    image = workflow["jobs"]["publish-docker"]
    assert image["needs"] == ["release-preflight"]
    build = next(step for step in image["steps"] if step.get("uses", "").startswith("docker/build-push-action@"))
    assert build["with"]["push"] == "false"
    assert build["with"]["load"] == "true"
    names = [step.get("name", "") for step in image["steps"]]
    assert names.index("Verify built image before publication") < names.index("Publish the independently qualified image")
    health = next(step for step in image["steps"] if step.get("name") == "Verify built image before publication")
    assert "docker run --pull never" in health["run"]
    assert "abi_version() == (2, 24, 0)" in health["run"]
    assert "-m pip check" in health["run"]

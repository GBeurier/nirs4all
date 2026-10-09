"""Contract tests for the release publication workflow."""

import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[3] / ".github/workflows/publish.yml"
WORKFLOWS_DIR = WORKFLOW_PATH.parent
PYPROJECT_PATH = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _load_workflow() -> dict[str, Any]:
    # BaseLoader keeps GitHub's ``on`` key as a string instead of YAML 1.1 bool.
    return cast(dict[str, Any], yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader))


def _load_named_workflow(name: str) -> dict[str, Any]:
    path = WORKFLOWS_DIR / name
    return cast(dict[str, Any], yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader))


def test_manual_dispatch_is_guarded_and_release_publication_is_verified() -> None:
    workflow = _load_workflow()
    assert set(workflow["on"]) == {"release", "workflow_dispatch"}

    jobs = workflow["jobs"]
    assert set(jobs["publish-pypi"]["needs"]) == {"build", "build-docs", "package-smoke", "local-qualification"}
    assert jobs["publish-pypi"]["permissions"] == {"id-token": "write", "contents": "read"}
    assert jobs["publish-pypi"]["environment"] == "pypi"

    verification_steps = [
        step
        for step in jobs["build"]["steps"]
        if step.get("name") == "Verify version consistency"
    ]
    assert len(verification_steps) == 1
    verification = verification_steps[0]
    assert verification["if"] == "github.event_name == 'release' || inputs.publish_release"
    assert "needs.release-preflight.outputs.release_tag" in verification["env"]["RELEASE_TAG"]
    assert 'TAG_VERSION="${TAG_VERSION//-rc./rc}"' in verification["run"]
    assert 'if [[ "$PKG_VERSION" != "$TAG_VERSION" ]]' in verification["run"]

    for job_name in ("publish-pypi", "publish-docker"):
        job = jobs[job_name]
        assert job["if"] == "github.event_name == 'release' || inputs.publish_release"
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "build" in needs

    public_smoke = jobs["post-publish-smoke"]
    assert public_smoke["if"] == "github.event_name == 'release' || inputs.publish_release"
    assert "publish-pypi" in public_smoke["needs"]
    smoke_steps = {step.get("name"): step for step in public_smoke["steps"]}
    install_script = smoke_steps["Install the exact PyPI release"]["run"]
    verify_script = smoke_steps["Verify public installed package metadata, origins and native ABI"]["run"]
    assert "--index-url https://pypi.org/simple/" in install_script
    assert '"nirs4all==${package_version}"' in install_script
    assert 'cd "$RUNNER_TEMP"' in verify_script
    assert 'scripts/smoke_installed_package.py" --source-root "$source_root"' in verify_script
    assert "python -m pip check" in verify_script
    assert "pytest" not in install_script + verify_script

    metadata_steps = [
        step
        for step in jobs["publish-docker"]["steps"]
        if step.get("uses") == "docker/metadata-action@v6"
    ]
    assert len(metadata_steps) == 1
    assert (
        "type=raw,value=latest,enable=${{ needs.release-preflight.outputs.prerelease == 'false' }}"
        in metadata_steps[0]["with"]["tags"]
    )


def test_release_metadata_closes_the_published_v1_stack_and_legal_files() -> None:
    """The base wheel owns Studio's full V1 runtime and dual-license notices."""

    with PYPROJECT_PATH.open("rb") as stream:
        pyproject = tomllib.load(stream)

    dependencies = set(pyproject["project"]["dependencies"])
    assert {
        "dag-ml>=0.3.41,<0.4",
        "dag-ml-data>=0.2.13,<0.3",
        "nirs4all-io>=0.2.5,<0.3",
        "nirs4all-core>=0.4.5,<0.5",
        "nirs4all-methods>=1.3.2,<2",
        "pls4all>=1.3.2,<2",
    } <= dependencies
    assert pyproject["project"]["license"] == "CeCILL-2.1 OR AGPL-3.0-or-later"
    assert set(pyproject["project"]["license-files"]) == {
        "LICENSE",
        "LICENSING.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/*",
    }


def test_github_delivery_checks_keep_scientific_suites_local() -> None:
    """GitHub builds and verifies packages; scientific qualification stays local."""

    ci = _load_named_workflow("CI.yaml")
    assert set(ci["on"]) == {"push", "pull_request"}
    assert set(ci["jobs"]) == {"lint", "type-check", "package-smoke"}

    for workflow_name, job_name in (
        ("CI.yaml", "package-smoke"), ("publish.yml", "package-smoke"),
        ("shared-test-and-docs.yml", "run-tests"), ("pre-publish.yml", "test-build"),
    ):
        job = _load_named_workflow(workflow_name)["jobs"][job_name]
        assert "strategy" not in job
        steps = job["steps"]
        install_index = next(index for index, step in enumerate(steps) if "pip install dist/*.whl" in step.get("run", ""))
        smoke_index = next(index for index, step in enumerate(steps) if "scripts/smoke_installed_package.py" in step.get("run", ""))
        assert install_index < smoke_index
        assert "if" not in steps[install_index] and "if" not in steps[smoke_index]
        assert 'cd "$RUNNER_TEMP"' in steps[smoke_index]["run"]
        assert "--source-root" in steps[smoke_index]["run"]
        assert "python -m pip check" in steps[smoke_index]["run"]

    shared = _load_named_workflow("shared-test-and-docs.yml")
    assert shared["jobs"]["run-tests"]["if"] == "${{ !inputs.skip-tests }}"
    trigger = _load_named_workflow("full-dagml-tests.yml")
    assert set(trigger["on"]) == {"workflow_dispatch"}
    assert trigger["jobs"]["exhaustive"]["uses"] == "./.github/workflows/shared-test-and-docs.yml"
    assert trigger["jobs"]["exhaustive"]["with"]["skip-tests"] == "false"
    assert trigger["jobs"]["exhaustive"]["with"]["upload-coverage"] == "false"

    for workflow_name in ("pre-publish.yml", "publish.yml"):
        qualification = _load_named_workflow(workflow_name)["jobs"]["local-qualification"]
        assert any(
            "scripts/verify_local_qualification.py --project sdk --receipt compat/local-qualification.json --root ." in step.get("run", "")
            for step in qualification["steps"]
        )
    examples = _load_named_workflow("examples.yml")
    assert set(examples["on"]) == {"workflow_dispatch"}
    assert set(examples["jobs"]) == {"example-source"}
    assert "python -m compileall -q examples/user examples/developer examples/reference" in yaml.safe_dump(examples["jobs"])

    for workflow_name in ("CI.yaml", "publish.yml", "shared-test-and-docs.yml", "pre-publish.yml", "examples.yml"):
        jobs = _load_named_workflow(workflow_name)["jobs"]
        commands = "\n".join(step.get("run", "") for job in jobs.values() for step in job.get("steps", []))
        assert "python -m pytest" not in commands
        assert "run_full_dagml_pytest.py" not in commands
        assert "run_ci_examples.sh" not in commands


@pytest.mark.parametrize("changed_science", [False, True])
def test_publication_refuses_checkout_that_differs_from_immutable_tag(tmp_path: Path, changed_science: bool) -> None:
    """Package and qualification gates must run against the immutable release commit."""
    workflow = _load_workflow()
    root = tmp_path / "tag"

    def commit(directory: Path, contents: dict[str, str]) -> str:
        for name, value in contents.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        subprocess.run(["git", "init", "--quiet", str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(directory), "-c", "user.name=CI fixture", "-c", "user.email=ci@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture",
        ], check=True)
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()

    tag = commit(root, {"nirs4all/__init__.py": "qualified package\n"})
    if changed_science:
        commit(root, {"nirs4all/__init__.py": "unqualified scientific change\n"})
    for name in ("package-smoke", "local-qualification"):
        steps = workflow["jobs"][name]["steps"]
        checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v6")
        assert "needs.release-preflight.outputs.release_tag" in checkout["with"]["ref"]
        guard = next(step for step in steps if step.get("name") == "Verify checked out commit matches publication target")
        result = subprocess.run(
            ["bash", "-e", "-c", guard["run"]], cwd=root,
            env={**os.environ, "EXPECTED_TAG_SHA": tag}, capture_output=True, text=True, check=False,
        )
        assert (result.returncode != 0) == changed_science, result.stdout + result.stderr

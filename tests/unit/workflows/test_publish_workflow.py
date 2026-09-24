"""Contract tests for the release publication workflow."""

import tomllib
from pathlib import Path
from typing import Any, cast

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
    test_job = jobs["run-tests"]
    assert test_job["permissions"] == {"contents": "read", "id-token": "write"}
    codecov_step = next(
        step for step in test_job["steps"] if step.get("uses") == "codecov/codecov-action@v7"
    )
    assert codecov_step["with"]["use_oidc"] == "true"
    assert codecov_step["with"]["fail_ci_if_error"] == "false"

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
    verify_script = smoke_steps["Verify installed metadata, native ABI, and DAG-ML execution"]["run"]
    assert "--index-url https://pypi.org/simple/" in install_script
    assert '"nirs4all==${package_version}"' in install_script
    assert 'version("nirs4all") == expected' in verify_script
    assert "n4m.abi_version()[:2] == (2, 6)" in verify_script
    assert 'engine="dag-ml"' in verify_script
    assert "np.isfinite(result.cv_best_score)" in verify_script
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
        "dag-ml>=0.3.27,<0.4",
        "dag-ml-data>=0.2.12,<0.3",
        "nirs4all-io>=0.2.0,<0.3",
        "nirs4all-core>=0.3.31,<0.4",
        "nirs4all-methods>=1.0.20,<2",
    } <= dependencies
    assert pyproject["project"]["license"] == "CeCILL-2.1 OR AGPL-3.0-or-later"
    assert set(pyproject["project"]["license-files"]) == {
        "LICENSE",
        "LICENSING.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/*",
    }


def test_github_fast_and_exhaustive_gates_have_distinct_triggers() -> None:
    """PR/release gates stay fast while the full module runner remains scheduled/manual."""

    for workflow_name, test_job in (("CI.yaml", "tests"), ("publish.yml", "run-tests")):
        job = _load_named_workflow(workflow_name)["jobs"][test_job]
        serialized = yaml.safe_dump(job)
        assert "strategy" not in job
        assert "tests/unit/" in serialized
        assert "tests/integration/api/" in serialized
        assert "test_marker_audit.py" in serialized
        assert "run_full_dagml_pytest.py" not in serialized

    exhaustive = _load_named_workflow("shared-test-and-docs.yml")["jobs"]["run-tests"]
    exhaustive_serialized = yaml.safe_dump(exhaustive)
    assert "run_full_dagml_pytest.py" in exhaustive_serialized
    assert "timeout-minutes" not in exhaustive
    assert "--ignore=" not in exhaustive_serialized
    assert exhaustive["permissions"] == {"contents": "read", "id-token": "write"}
    assert "use_oidc: 'true'" in exhaustive_serialized
    assert "fail_ci_if_error: 'true'" in exhaustive_serialized

    trigger = _load_named_workflow("full-dagml-tests.yml")
    assert set(trigger["on"]) == {"schedule", "workflow_dispatch"}
    assert trigger["jobs"]["exhaustive"]["permissions"] == {"contents": "read", "id-token": "write"}
    assert trigger["jobs"]["exhaustive"]["uses"] == "./.github/workflows/shared-test-and-docs.yml"

    for workflow_name in ("pre-publish.yml", "publish.yml", "examples.yml"):
        workflow = _load_named_workflow(workflow_name)
        job = workflow["jobs"]["verify-examples"]
        assert "strategy" not in job
        serialized = yaml.safe_dump(job)
        assert "run_ci_examples.sh -c all -j 2 -k" in serialized

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

    public_smoke = jobs["post-publish-smoke"]
    assert public_smoke["if"] == "github.event_name == 'release'"
    assert public_smoke["needs"] == "publish-pypi"
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
        "type=raw,value=latest,enable=${{ github.event.release.prerelease == false }}"
        in metadata_steps[0]["with"]["tags"]
    )


def test_release_metadata_closes_the_published_v1_stack_and_legal_files() -> None:
    """The base wheel owns Studio's full V1 runtime and dual-license notices."""

    with PYPROJECT_PATH.open("rb") as stream:
        pyproject = tomllib.load(stream)

    dependencies = set(pyproject["project"]["dependencies"])
    assert {
        "dag-ml>=0.3.26,<0.4",
        "dag-ml-data>=0.2.11,<0.3",
        "nirs4all-io>=0.2.0,<0.3",
        "nirs4all-core>=0.3.30,<0.4",
        "nirs4all-methods>=1.0.20,<2",
    } <= dependencies
    assert pyproject["project"]["license"] == "CeCILL-2.1 OR AGPL-3.0-or-later"
    assert set(pyproject["project"]["license-files"]) == {
        "LICENSE",
        "LICENSING.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/*",
    }


def test_github_full_gates_run_once_without_local_v1_dual_qualification() -> None:
    """CI qualifies all tests/examples once; the exhaustive dual oracle stays local."""

    for workflow_name, test_job in (
        ("CI.yaml", "tests"),
        ("pre-publish.yml", "run-tests"),
        ("publish.yml", "run-tests"),
        ("shared-test-and-docs.yml", "run-tests"),
    ):
        workflow = _load_named_workflow(workflow_name)
        job = workflow["jobs"][test_job]
        assert "strategy" not in job
        serialized = yaml.safe_dump(job)
        assert "tests/" in serialized
        assert "--ignore=tests/integration/parity/test_conformance_dual_engine.py" in serialized

    for workflow_name in ("CI.yaml", "pre-publish.yml", "publish.yml", "examples.yml"):
        workflow = _load_named_workflow(workflow_name)
        job_name = "tests" if workflow_name == "CI.yaml" else "verify-examples"
        job = workflow["jobs"][job_name]
        assert "strategy" not in job
        serialized = yaml.safe_dump(job)
        assert "run_ci_examples.sh -c all -j 2 -k" in serialized

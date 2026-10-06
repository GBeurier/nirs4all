"""The U15 prerequisite helper preserves release origins and failed commands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HELPER = Path(__file__).resolve().parents[3] / "scripts/prepare_octave_example_runtime.py"
SPEC = importlib.util.spec_from_file_location("octave_example_preparation", HELPER)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.mark.parametrize("head,changed,accepted", [
    (helper.METHODS_COMMIT, "", True),
    ("followup", ".github/workflows/release-wheels.yml\n", True),
    (helper.METHODS_COMMIT, "bindings/matlab/+n4m/RolePipeline.m\n", False),
    ("followup", "cpp/src/version.cpp\n", False),
])
def test_methods_origin_accepts_only_the_documented_ci_followup(monkeypatch, tmp_path, head, changed, accepted):
    replies = iter([head + "\n", changed])
    monkeypatch.setattr(helper.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    if accepted:
        assert helper.require_commit(tmp_path, helper.METHODS_COMMIT) == head
    else:
        with pytest.raises(ValueError, match="source origin mismatch"):
            helper.require_commit(tmp_path, helper.METHODS_COMMIT)


@pytest.mark.parametrize("origin,version_req,accepted", [
    ("registry+https://github.com/rust-lang/crates.io-index", "=0.4.2", True),
    ("path+file:///local/source", "=0.4.2", False),
    ("registry+https://github.com/rust-lang/crates.io-index", ">=0.4.2", False),
])
def test_core_cli_requires_the_exact_public_registry_install(tmp_path, origin, version_req, accepted):
    binary = tmp_path / "bin/nirs4all-core-archive"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    (tmp_path / ".crates2.json").write_text(json.dumps({"installs": {
        f"nirs4all 0.4.2 ({origin})": {"version_req": version_req, "bins": [binary.name]},
    }}))
    if accepted:
        assert helper.require_public_core_install(tmp_path) == binary
    else:
        with pytest.raises(ValueError, match="exact public registry"):
            helper.require_public_core_install(tmp_path)


def test_failed_prerequisite_keeps_its_exact_exit_and_full_log(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        helper.run([sys.executable, "-c", "print('real failure'); raise SystemExit(7)"],
                   cwd=tmp_path, env=dict(os.environ), logs=tmp_path)
    assert error.value.returncode == 7
    receipt = json.loads((tmp_path / "00.json").read_text())
    assert receipt["exit_code"] == 7
    assert (tmp_path / "00.log").read_text() == "real failure\n"
    assert receipt["log_sha256"] == helper.sha256(tmp_path / "00.log")


def _git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _commit(repo):
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=CI fixture", "-c", "user.email=ci@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def dag_checkouts(tmp_path, monkeypatch):
    """Real Git objects independently witness runtime/tooling isolation."""
    repo = tmp_path / "helper"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    for path, payload in {
        helper.DAG_HELPER_PATHS[0]: "old helper\n",
        helper.DAG_HELPER_PATHS[1]: "old regression\n",
        "crates/runtime/src/lib.rs": "qualified production\n",
    }.items():
        destination = repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
    base = _commit(repo)
    for path in helper.DAG_HELPER_PATHS:
        (repo / path).write_text("corrected tooling\n")
    patched = _commit(repo)
    runtime = tmp_path / "dag-ml-cli-src"
    _git(repo, "clone", "--quiet", "--no-hardlinks", str(repo), str(runtime))
    _git(runtime, "checkout", "--quiet", "--detach", base)
    monkeypatch.setattr(helper, "DAG_COMMIT", base)
    monkeypatch.setattr(helper, "DAG_HELPER_COMMIT", patched)
    return repo, runtime, base, patched


def test_pinned_helper_records_both_origins_and_unchanged_production(dag_checkouts):
    repo, runtime, base, patched = dag_checkouts
    proof = helper.require_dag_helper(repo)
    assert helper.require_commit(runtime, base) == base
    assert proof["dag_release_commit"] == base
    assert proof["dag_helper_commit"] == proof["dag_commit"] == patched
    assert proof["dag_helper_changed_paths"] == list(helper.DAG_HELPER_PATHS)
    expected_tree = "\n".join(line for line in _git(runtime, "ls-tree", "-r", "--full-tree", "HEAD").splitlines()
                              if line.split("\t", 1)[1] not in helper.DAG_HELPER_PATHS).encode()
    assert proof["dag_unchanged_tree_sha256"] == helper.hashlib.sha256(expected_tree).hexdigest()
    assert proof["dag_helper_artifacts"] == {path: helper.sha256(repo / path) for path in helper.DAG_HELPER_PATHS}


@pytest.mark.parametrize("extra_path", ["crates/runtime/src/lib.rs", "scripts/another_helper.py"])
def test_helper_rejects_production_changes_and_a_third_tooling_file(dag_checkouts, monkeypatch, extra_path):
    repo, _, _, _ = dag_checkouts
    (repo / extra_path).write_text("unexpected change\n")
    monkeypatch.setattr(helper, "DAG_HELPER_COMMIT", _commit(repo))
    with pytest.raises(ValueError, match="only the two declared tooling files"):
        helper.require_dag_helper(repo)


def test_helper_rejects_missing_declared_regression(dag_checkouts, monkeypatch):
    repo, _, base, _ = dag_checkouts
    _git(repo, "checkout", base, "--", helper.DAG_HELPER_PATHS[1])
    monkeypatch.setattr(helper, "DAG_HELPER_COMMIT", _commit(repo))
    with pytest.raises(ValueError, match="only the two declared tooling files"):
        helper.require_dag_helper(repo)


def test_helper_rejects_wrong_commit_even_with_identical_production(dag_checkouts):
    repo, _, _, _ = dag_checkouts
    _git(repo, "-c", "user.name=CI fixture", "-c", "user.email=ci@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "--allow-empty", "--quiet", "-m", "unqualified")
    with pytest.raises(ValueError, match="helper source origin mismatch"):
        helper.require_dag_helper(repo)


@pytest.mark.parametrize("staged", [False, True])
def test_helper_rejects_dirty_tracked_files(dag_checkouts, staged):
    repo, _, _, _ = dag_checkouts
    (repo / helper.DAG_HELPER_PATHS[0]).write_text("uncommitted helper\n")
    if staged:
        _git(repo, "add", helper.DAG_HELPER_PATHS[0])
    with pytest.raises(ValueError, match="helper source origin mismatch"):
        helper.require_dag_helper(repo)


@pytest.mark.parametrize("replace_head", [False, True])
def test_runtime_rejects_dirty_sources_or_helper_commit(dag_checkouts, replace_head):
    _, runtime, base, patched = dag_checkouts
    if replace_head:
        _git(runtime, "checkout", "--quiet", "--detach", patched)
    else:
        (runtime / "crates/runtime/src/lib.rs").write_text("unqualified runtime\n")
    with pytest.raises(ValueError, match="source origin mismatch"):
        helper.require_commit(runtime, base)


def test_action_uses_the_separate_pinned_helper_without_replacing_candidate_runtime():
    root = HELPER.parents[1]
    action = yaml.load((root / ".github/actions/prepare-octave-example/action.yml").read_text(), Loader=yaml.BaseLoader)
    checkouts = [step for step in action["runs"]["steps"] if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 1
    assert checkouts[0]["with"] == {"repository": "GBeurier/dag-ml", "ref": helper.DAG_HELPER_COMMIT,
                                    "path": ".octave-example-dag-helper", "fetch-depth": "0"}
    preparation = next(step for step in action["runs"]["steps"] if "prepare_octave_example_runtime.py" in step.get("run", ""))
    assert '--dag-root "$GITHUB_WORKSPACE/.octave-example-dag-helper"' in preparation["run"]
    candidate = yaml.load((root / ".github/actions/candidate-native/action.yml").read_text(), Loader=yaml.BaseLoader)
    runtime_checkout = next(step for step in candidate["runs"]["steps"]
                            if step.get("with", {}).get("repository") == "GBeurier/dag-ml")
    assert runtime_checkout["with"]["ref"] == helper.DAG_COMMIT
    assert runtime_checkout["with"]["path"] == "dag-ml-cli-src"


def test_failed_native_prerequisite_still_records_both_real_git_origins(dag_checkouts, monkeypatch, tmp_path):
    repo, runtime, base, patched = dag_checkouts
    methods = tmp_path / "methods"
    methods.mkdir()
    require_commit = helper.require_commit

    def require_origin(path, expected):
        # Only Methods is stubbed; both DAG source identities use actual Git.
        return helper.METHODS_COMMIT if path == methods else require_commit(path, expected)

    monkeypatch.setattr(helper, "require_commit", require_origin)
    output = tmp_path / "prerequisites"
    args = argparse.Namespace(workspace=tmp_path, methods_root=methods, dag_root=repo,
                              octave=Path(sys.executable), output=output, library=None)
    with pytest.raises(FileNotFoundError, match="candidate-native prerequisite absent"):
        helper.prepare(args)
    receipt = json.loads((output / "prerequisites.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["commands"] == []
    assert receipt["dag_release_commit"] == receipt["dag_runtime_actual_commit"] == base
    assert receipt["dag_helper_commit"] == receipt["dag_commit"] == patched
    assert receipt["dag_runtime_root"] == str(runtime)
    assert receipt["dag_helper_root"] == str(repo)
    assert receipt["dag_helper_artifacts"][helper.DAG_HELPER_PATHS[0]] == helper.sha256(repo / helper.DAG_HELPER_PATHS[0])

"""The U15 prerequisite helper preserves release origins and failed commands."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    ("registry+https://github.com/rust-lang/crates.io-index", "=0.4.1", True),
    ("path+file:///local/source", "=0.4.1", False),
    ("registry+https://github.com/rust-lang/crates.io-index", ">=0.4.1", False),
])
def test_core_cli_requires_the_exact_public_registry_install(tmp_path, origin, version_req, accepted):
    binary = tmp_path / "bin/nirs4all-core-archive"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    (tmp_path / ".crates2.json").write_text(json.dumps({"installs": {
        f"nirs4all 0.4.1 ({origin})": {"version_req": version_req, "bins": [binary.name]},
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

"""Installed replay preparation rejects source drift and parent product hooks."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[3] / "scripts/prepare_installed_example_runtime.py"
SPEC = importlib.util.spec_from_file_location("installed_example_preparation", HELPER)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
STUDIO_SPEC = importlib.util.spec_from_file_location("installed_studio_gate", HELPER.parents[1] / "tests/integration/api/test_installed_studio_general.py")
assert STUDIO_SPEC is not None and STUDIO_SPEC.loader is not None
studio_gate = importlib.util.module_from_spec(STUDIO_SPEC)
STUDIO_SPEC.loader.exec_module(studio_gate)


def test_dependency_projection_excludes_products_metadata_and_executable_hooks(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in ["numpy", "wheel", "pip", "setuptools", "nirs4all", "n4m", "dag_ml", "nirs4all_methods.libs"]:
        (parent / name).mkdir()
    for name in ["editable.pth", "project.egg-link", "__editable___finder.py", "sitecustomize.py"]:
        (parent / name).write_text("raise RuntimeError('parent hooks must never execute')\n")
    for name, distribution in [("numpy-2.4.6.dist-info", "numpy"), ("wheel-0.45.1.dist-info", "wheel"),
                               ("pip-25.0.dist-info", "pip"), ("setuptools-80.0.dist-info", "setuptools"),
                               ("nirs4all-1.3.3.dist-info", "nirs4all")]:
        directory = parent / name
        directory.mkdir()
        (directory / "METADATA").write_text(f"Name: {distribution}\nVersion: 1\n")
    projection = tmp_path / "projection"
    inventory = helper.project_dependencies([parent], projection)
    assert set(inventory) == {"numpy", "numpy-2.4.6.dist-info", "wheel", "wheel-0.45.1.dist-info"}
    assert (projection / "wheel").resolve() == parent / "wheel"
    assert (projection / "wheel-0.45.1.dist-info/METADATA").read_text() == "Name: wheel\nVersion: 1\n"
    assert (projection / "numpy").is_symlink()
    assert (projection / "numpy").resolve() == parent / "numpy"
    assert (parent / "editable.pth").read_text().startswith("raise RuntimeError")


def test_projected_wheel_satisfies_a_real_child_pip_check(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    for name, metadata in {
        "wheel-0.45.1.dist-info": "Name: wheel\nVersion: 0.45.1\n",
        "capture-fixture-1.0.dist-info": "Name: capture-fixture\nVersion: 1.0\nRequires-Dist: wheel\n",
    }.items():
        directory = parent / name
        directory.mkdir()
        (directory / "METADATA").write_text(metadata)
    projection = tmp_path / "projection"
    helper.project_dependencies([parent], projection)
    child = tmp_path / "child"
    venv.EnvBuilder(with_pip=True).create(child)
    python = child / "bin/python"
    purelib = Path(subprocess.check_output([str(python), "-I", "-B", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    (purelib / "dependencies.pth").write_text(str(projection) + "\n")
    result = subprocess.run([str(python), "-I", "-B", "-m", "pip", "check"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    (projection / "wheel-0.45.1.dist-info").unlink()
    missing = subprocess.run([str(python), "-I", "-B", "-m", "pip", "check"], capture_output=True, text=True, check=False)
    assert missing.returncode == 1
    assert "requires wheel, which is not installed" in missing.stdout


def test_sdk_wheel_identity_and_source_payload_are_independently_checked(tmp_path):
    source = tmp_path / "source"
    (source / "nirs4all").mkdir(parents=True)
    (source / "nirs4all/__init__.py").write_bytes(b"__version__ = '1.4.0'\n")
    wheel = tmp_path / "sdk.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nirs4all/__init__.py", (source / "nirs4all/__init__.py").read_bytes())
        archive.writestr("nirs4all-1.4.0.dist-info/METADATA", "Name: nirs4all\nVersion: 1.4.0\n")
    assert helper.wheel_identity(wheel) == ("nirs4all", "1.4.0")
    assert helper.sdk_payload(wheel, source) == {"nirs4all/__init__.py": helper.sha256(source / "nirs4all/__init__.py")}
    (source / "nirs4all/__init__.py").write_bytes(b"__version__ = 'different runtime'\n")
    with pytest.raises(ValueError, match="wheel/source payload mismatch"):
        helper.sdk_payload(wheel, source)


def test_sdk_payload_refuses_a_wheel_without_runtime_members(tmp_path):
    wheel = tmp_path / "sdk.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nirs4all-1.4.0.dist-info/METADATA", "Name: nirs4all\nVersion: 1.4.0\n")
    with pytest.raises(ValueError, match="runtime payload absent"):
        helper.sdk_payload(wheel, tmp_path)


def test_exports_configure_all_api_cold_consumers_and_launch_the_studio_child(tmp_path, monkeypatch):
    child = tmp_path / "isolated child"
    venv.EnvBuilder(with_pip=False).create(child)
    python = child / "bin/python"
    github_env = tmp_path / "github-env"
    github_env.write_text("EXISTING_SETTING=preserved\n")
    helper.export_installed_interpreters(python, github_env)
    exported = dict(line.split("=", 1) for line in github_env.read_text().splitlines())
    consumers = set()
    required_flags = set()
    for module in (HELPER.parents[1] / "tests/integration/api").glob("test_*.py"):
        if module.name == "test_octave_multimodal_role_archive.py":
            continue  # External Octave/R runtimes have separately provisioned dependencies.
        consumers.update(re.findall(r"\bNIRS4ALL_[A-Z0-9_]*INSTALLED_PYTHON\b", module.read_text()))
        required_flags.update(re.findall(r"\bNIRS4ALL_REQUIRE_[A-Z0-9_]+\b", module.read_text()))
    separately_provisioned = {
        "NIRS4ALL_REQUIRE_CV_WEIGHT_CLI", "NIRS4ALL_REQUIRE_HPO_CLI_PARITY",  # Exact external DAG CLI.
        "NIRS4ALL_REQUIRE_NATIVE_FIT_WITNESS",  # Test-instrumented native probe and helper.
        "NIRS4ALL_REQUIRE_XL03_INSTALLED",  # Genuine native capture artifacts.
    }
    required_flags -= separately_provisioned
    assert set(exported) == {"EXISTING_SETTING", *consumers, *required_flags}
    assert exported.pop("EXISTING_SETTING") == "preserved"
    assert {exported[name] for name in consumers} == {str(python)}
    assert {exported[name] for name in required_flags} == {"1"}
    for name, value in exported.items():
        monkeypatch.setenv(name, value)
    result = subprocess.run([studio_gate._installed_python(), "-I", "-B", "-c", "import json, sys; print(json.dumps([sys.prefix, sys.executable]))"],
                            capture_output=True, text=True, check=True)
    prefix, executable = json.loads(result.stdout)
    assert Path(prefix) == child
    assert Path(executable) == python
    assert Path(prefix) != Path(sys.prefix)


@pytest.mark.parametrize("configuration", ["absent", "missing", "not_executable"])
def test_required_studio_gate_rejects_bad_child_configuration_before_launch(tmp_path, monkeypatch, configuration):
    monkeypatch.setenv("NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2", "1")
    monkeypatch.delenv("NIRS4ALL_STUDIO_INSTALLED_PYTHON", raising=False)
    monkeypatch.setenv("NIRS4ALL_STUDIO_PYTHON", sys.executable)  # An obsolete setting must not silently select the parent.
    if configuration != "absent":
        path = tmp_path / "python"
        if configuration == "not_executable":
            path.write_text("not an interpreter\n")
        monkeypatch.setenv("NIRS4ALL_STUDIO_INSTALLED_PYTHON", str(path))

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Invalid child configuration must fail before launching Studio")

    monkeypatch.setattr(studio_gate.subprocess, "run", forbidden_launch)
    with pytest.raises(pytest.fail.Exception, match="NIRS4ALL_STUDIO_INSTALLED_PYTHON"):
        studio_gate.test_installed_studio_general_host_keeps_rust_boundary_and_durable_results(tmp_path, False)


@pytest.mark.parametrize("separator", ["\n", "\r"])
def test_interpreter_exports_reject_environment_file_injection(tmp_path, separator):
    github_env = tmp_path / "github-env"
    github_env.write_text("EXISTING_SETTING=preserved\n")
    with pytest.raises(ValueError, match="Invalid GitHub interpreter path"):
        helper.export_installed_interpreters(Path(f"/child/python{separator}UNRELATED=bad"), github_env)
    assert github_env.read_text() == "EXISTING_SETTING=preserved\n"

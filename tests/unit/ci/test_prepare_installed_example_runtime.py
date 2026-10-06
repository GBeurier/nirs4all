"""Installed replay preparation rejects source drift and parent product hooks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest
import yaml

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


def _preparation_fixture(tmp_path, monkeypatch, *, align=False, fault=None):
    """Exercise orchestration with synthetic unit artifacts and intercepted installers."""
    workspace = tmp_path / "workspace"
    (workspace / "nirs4all").mkdir(parents=True)
    sdk_source = workspace / "nirs4all/__init__.py"
    sdk_source.write_text(f"__version__ = {helper.SDK_VERSION!r}\n")
    wheelhouse = tmp_path / "unit-wheel-fixtures"
    wheelhouse.mkdir()
    artifacts = {}
    module_bytes = {
        "dag_ml._dag_ml": b"synthetic unit DAG extension bytes",
        "n4m.roles._multimodal": b"# synthetic unit Methods source bytes\n",
    }
    for name, version in {"nirs4all": helper.SDK_VERSION, **helper.UPSTREAMS}.items():
        path = wheelhouse / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
            if name == "nirs4all":
                wheel.writestr("nirs4all/__init__.py", sdk_source.read_bytes())
            elif name == "dag-ml":
                wheel.writestr("dag_ml/_dag_ml.abi3.so", module_bytes["dag_ml._dag_ml"])
            elif name == "nirs4all-methods":
                wheel.writestr("n4m/roles/_multimodal.py", module_bytes["n4m.roles._multimodal"])
        artifacts[name] = path
    manifest = tmp_path / "unit-artifact-manifest.json"
    manifest.write_text(json.dumps({name: {"path": str(artifacts[name]), "version": version,
                                          "public": True, "sha256": helper.sha256(artifacts[name])}
                                    for name, version in helper.UPSTREAMS.items()}))
    shared = tmp_path / "parent-site-packages"
    shared.mkdir()
    (shared / "torch").mkdir()
    (shared / "torch/__init__.py").write_text("# preserve chosen Torch profile\n")
    cli = tmp_path / "source-cli"
    cli.write_bytes(b"preserve compiled source CLI")
    native = tmp_path / "source-methods.so"
    native.write_bytes(b"preserve source native Methods library")
    methods_source = tmp_path / "methods-source/bindings/python/src/n4m/roles/_multimodal.py"
    methods_source.parent.mkdir(parents=True)
    (methods_source.parents[1] / "__init__.py").write_text("# source Methods package\n")
    methods_source.write_bytes(module_bytes["n4m.roles._multimodal"])
    monkeypatch.setenv("N4M_LIB_PATH", str(native))
    monkeypatch.setenv("NIRS4ALL_CORE_LIVE_METHODS_LIBRARY", str(native))
    monkeypatch.setenv("PYTHONPATH", str(workspace) + os.pathsep + str(methods_source.parents[2]))
    github_env = tmp_path / "github-env"
    github_env.write_text("EXISTING_SETTING=preserved\n")
    args = argparse.Namespace(workspace=workspace, output=tmp_path / "prepared", sdk_wheel=artifacts["nirs4all"],
                              upstream_manifest=manifest, shared_site_packages=[shared], github_env=github_env,
                              align_parent_public_upstreams=align)
    purelib = args.output / "child/lib/site-packages"
    commands = []

    def create_child(self, path):
        purelib.mkdir(parents=True)
        (path / "bin").mkdir()
        (path / "bin/python").write_text("# intercepted unit interpreter\n")

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if "-c" in command:
            compile(command[command.index("-c") + 1], "<fresh-runtime-proof>", "exec")
            arguments = command[command.index("-c") + 2:]
            parent = command[0] == sys.executable
            expected = json.loads(arguments[0])
            bindings = json.loads(arguments[1] if parent else arguments[3])
            proof = {"versions": expected, "public_bindings": {}}
            for module, item in bindings.items():
                origin = (tmp_path / "parent-runtime" if parent else purelib) / item["wheel_member"]
                origin.parent.mkdir(parents=True, exist_ok=True)
                if origin != methods_source:
                    origin.write_bytes(module_bytes[module])
                wrong = module == "dag_ml._dag_ml" and ((parent and fault == "parent_binding") or (not parent and fault in {"child_binding", "forged_child_digest"}))
                if wrong:
                    origin.write_bytes(b"wrong installed native extension bytes")
                digest = hashlib.sha256(origin.read_bytes()).hexdigest()
                if not parent and module == "dag_ml._dag_ml" and fault == "forged_child_digest":
                    digest = item["sha256"]
                proof["public_bindings"][module] = {"origin": str(origin), "sha256": digest}
            library = (tmp_path / "parent-runtime" if parent else purelib) / "n4m/lib/libn4m.so"
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(b"synthetic public unit library")
            proof["native_library"] = str(library)
            Path(arguments[2]).write_text(json.dumps(proof))
        elif "install" in command and command[0] != sys.executable and fault == "wheel_changed":
            with zipfile.ZipFile(artifacts["dag-ml"], "a") as wheel:
                wheel.writestr("changed-artifact.txt", "bytes changed after public artifact selection")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(helper.venv.EnvBuilder, "create", create_child)
    monkeypatch.setattr(helper.subprocess, "check_output", lambda *a, **k: str(purelib) + "\n")
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    return args, commands, artifacts, [sdk_source, shared / "torch/__init__.py", cli, native, methods_source]


@pytest.mark.parametrize("align", [False, True])
def test_public_parent_alignment_is_opt_in_and_installs_only_the_same_seven_artifacts(tmp_path, monkeypatch, align):
    args, commands, artifacts, preserved = _preparation_fixture(tmp_path, monkeypatch, align=align)
    before = {p: p.read_bytes() for p in preserved}
    receipt = helper.prepare(args)
    installs = [command for command, _ in commands if "install" in command]
    parent = [command for command in installs if command[0] == sys.executable]
    child = [command for command in installs if command[0] != sys.executable]
    assert len(parent) == int(align) and len(child) == 1
    approved = {"pls4all", "nirs4all-methods", "nirs4all-io", "dag-ml", "dag-ml-data", "nirs4all-core", "nirs4all-formats"}
    child_paths = child[0][child[0].index("--force-reinstall") + 1:]
    assert set(child_paths) == {str(path) for path in artifacts.values()}
    if align:
        installed = parent[0][parent[0].index("--force-reinstall") + 1:]
        assert len(installed) == 7
        assert {helper.wheel_identity(Path(path))[0] for path in installed} == approved
        assert set(installed) == {str(artifacts[name]) for name in approved}
        assert set(installed) < set(child_paths)
        assert "--no-deps" in parent[0]
        assert receipt["parent_child_binding_sha256_equal"] is True
        parent_index = next(i for i, (command, _) in enumerate(commands) if command == parent[0])
        assert "-c" in commands[parent_index + 1][0]  # Fresh process before child preparation/training.
        assert commands[parent_index + 1][0][0] == sys.executable
        assert "N4M_LIB_PATH" not in commands[parent_index + 1][1]["env"]
        assert "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY" not in commands[parent_index + 1][1]["env"]
        assert commands[parent_index + 1][1]["env"]["PYTHONPATH"] == str(args.workspace)
        assert os.environ["N4M_LIB_PATH"] == str(preserved[3])
        assert str(preserved[4].parents[2]) in os.environ["PYTHONPATH"]
    else:
        assert receipt["parent_alignment"] == {"requested": False, "status": "NOT_REQUESTED"}
        assert all(command[0] != sys.executable for command, _ in commands)
    assert {p: p.read_bytes() for p in preserved} == before
    assert receipt["status"] == "PASS"
    assert {k: v["sha256"] for k, v in receipt["wheel_artifacts"].items()} == {name: helper.sha256(path) for name, path in artifacts.items()}
    assert all("wheel" not in command and "download" not in command for command, _ in commands)
    assert any(command[-1] == "check" for command, _ in commands)
    exported = dict(line.split("=", 1) for line in args.github_env.read_text().splitlines())
    assert all(exported[flag] == "1" for flag in helper.INSTALLED_REQUIRE_FLAGS)
    if align:
        assert exported["PYTHONPATH"] == str(args.workspace)
        assert exported["N4M_LIB_PATH"] == exported["NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"]
        assert exported["N4M_LIB_PATH"] != str(preserved[3])
    else:
        assert "PYTHONPATH" not in exported and "N4M_LIB_PATH" not in exported


@pytest.mark.parametrize("fault", ["parent_binding", "child_binding", "forged_child_digest", "wheel_changed"])
def test_public_binding_or_artifact_drift_fails_before_any_cold_profile_export(tmp_path, monkeypatch, fault):
    args, commands, _, _ = _preparation_fixture(tmp_path, monkeypatch, align=True, fault=fault)
    expected = "Selected wheel artifact changed" if fault == "wheel_changed" else "Public upstream binding payload mismatch"
    with pytest.raises(ValueError, match=expected):
        helper.prepare(args)
    assert args.github_env.read_text() == "EXISTING_SETTING=preserved\n"
    receipt = json.loads((args.output / "prerequisites.json").read_text())
    assert receipt["status"] == "FAIL"
    if fault == "parent_binding":
        assert not (args.output / "child").exists()
        assert len(commands) == 2


def test_sdk_source_drift_is_rejected_before_opt_in_parent_installation(tmp_path, monkeypatch):
    args, commands, _, preserved = _preparation_fixture(tmp_path, monkeypatch, align=True)
    preserved[0].write_text("# source drift\n")
    with pytest.raises(ValueError, match="SDK wheel/source payload mismatch"):
        helper.prepare(args)
    assert commands == []
    assert args.github_env.read_text() == "EXISTING_SETTING=preserved\n"


@pytest.mark.parametrize("duplicate", [False, True])
def test_selected_public_wheel_requires_one_unambiguous_dag_extension(tmp_path, monkeypatch, duplicate):
    _, _, artifacts, _ = _preparation_fixture(tmp_path, monkeypatch)
    with zipfile.ZipFile(artifacts["dag-ml"], "w") as wheel:
        if duplicate:
            wheel.writestr("dag_ml/_dag_ml.abi3.so", b"first")
            wheel.writestr("dag_ml/_dag_ml.cpython-311.so", b"second")
    with pytest.raises(ValueError, match="exactly one dag_ml._dag_ml binding"):
        helper.public_binding_payloads(artifacts)


@pytest.mark.parametrize("align", [False, True])
def test_cli_parent_alignment_defaults_off_and_requires_explicit_flag(tmp_path, monkeypatch, align):
    captured = []

    def capture(args):
        captured.append(args)
        return {"status": "PASS", "python": "/unit-child/bin/python"}

    monkeypatch.setattr(helper, "prepare", capture)
    arguments = [str(HELPER), "--workspace", str(tmp_path), "--output", str(tmp_path / "prepared")]
    if align:
        arguments.append("--align-parent-public-upstreams")
    monkeypatch.setattr(sys, "argv", arguments)
    helper.main()
    assert len(captured) == 1 and captured[0].align_parent_public_upstreams is align


def test_ci_action_explicitly_requests_public_parent_alignment_in_its_real_shell(tmp_path, monkeypatch):
    action = yaml.safe_load((HELPER.parents[1] / ".github/actions/prepare-installed-example/action.yml").read_text())
    executable = tmp_path / "bin/python"
    executable.parent.mkdir()
    captured = tmp_path / "launched-arguments.json"
    executable.write_text(f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n")
    executable.chmod(0o755)
    environment = {**os.environ, "PATH": str(executable.parent) + os.pathsep + os.environ["PATH"],
                   "GITHUB_ACTION_PATH": str(HELPER.parents[1] / ".github/actions/prepare-installed-example"),
                   "GITHUB_WORKSPACE": str(tmp_path), "RUNNER_TEMP": str(tmp_path / "runner"), "GITHUB_ENV": str(tmp_path / "github-env")}
    result = subprocess.run(["bash", "-e", "-c", action["runs"]["steps"][0]["run"]], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(captured.read_text())
    assert arguments[0] == "-B" and Path(arguments[1]).resolve() == HELPER
    captured_args = []
    monkeypatch.setattr(sys, "argv", arguments[1:])
    monkeypatch.setattr(helper, "prepare", lambda args: captured_args.append(args) or {"status": "PASS", "python": "/unit-child/bin/python"})
    helper.main()
    assert captured_args[0].align_parent_public_upstreams is True
    assert captured_args[0].workspace == tmp_path


def test_public_environment_removes_only_known_methods_overlays_without_mutating_the_caller(tmp_path):
    paths = [tmp_path / "sdk", tmp_path / "methods/bindings/python/src", tmp_path / "tag/bindings/python_nirs4all_methods/src", tmp_path / "unrelated-overlay"]
    for path in paths[1:]:
        (path / "n4m").mkdir(parents=True)
        (path / "n4m/__init__.py").write_text("# fixture\n")
    environment = {"PYTHONPATH": os.pathsep.join(map(str, paths)), "N4M_LIB_PATH": "/source/libn4m.so",
                   "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY": "/source/libn4m.so", "N4A_DAGML_CLI": "/source/dag-ml-cli",
                   "CUDA_VISIBLE_DEVICES": "selected-profile", "UNRELATED": "preserved"}
    before = dict(environment)
    aligned = helper.public_parent_environment(environment)
    assert aligned == {"PYTHONPATH": os.pathsep.join(map(str, [paths[0], paths[3]])), "N4A_DAGML_CLI": "/source/dag-ml-cli",
                       "CUDA_VISIBLE_DEVICES": "selected-profile", "UNRELATED": "preserved"}
    assert environment == before


def test_public_payloads_include_python_native_and_relocated_files_not_metadata(tmp_path, monkeypatch):
    _, _, artifacts, _ = _preparation_fixture(tmp_path, monkeypatch)
    members = {"n4m/lib/libn4m.so.2": b"versioned native library", "n4m/backend.py": b"# Python runtime\n",
               "nirs4all_methods-1.3.2.data/purelib/n4m/relocated.py": b"# relocated runtime\n",
               "nirs4all_methods-1.3.2.dist-info/unused.py": b"metadata not runtime"}
    with zipfile.ZipFile(artifacts["nirs4all-methods"], "a") as wheel:
        for name, contents in members.items():
            wheel.writestr(name, contents)
    payloads = helper.public_upstream_payloads(artifacts)
    assert payloads["n4m/lib/libn4m.so.2"] == hashlib.sha256(members["n4m/lib/libn4m.so.2"]).hexdigest()
    assert payloads["n4m/backend.py"] == hashlib.sha256(members["n4m/backend.py"]).hexdigest()
    assert payloads["n4m/relocated.py"] == hashlib.sha256(members["nirs4all_methods-1.3.2.data/purelib/n4m/relocated.py"]).hexdigest()
    assert not any(".dist-info" in path for path in payloads)

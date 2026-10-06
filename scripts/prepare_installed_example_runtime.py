#!/usr/bin/env python3
"""Prepare an isolated wheel interpreter for mandatory installed API replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

SDK_VERSION = "1.4.1"
UPSTREAMS = {"pls4all": "1.3.2", "nirs4all-methods": "1.3.2", "nirs4all-io": "0.2.5", "dag-ml": "0.3.37", "dag-ml-data": "0.2.13", "nirs4all-core": "0.4.2", "nirs4all-formats": "0.2.11"}
PRODUCT_MODULES = {"nirs4all", "n4m", "pls4all", "dag_ml", "dag_ml_data", "nirs4all_io", "nirs4all_core", "nirs4all_formats"}
INSTALLED_PYTHON_VARIABLES = (
    "NIRS4ALL_BY_SOURCE_INSTALLED_PYTHON", "NIRS4ALL_CV_WEIGHT_INSTALLED_PYTHON", "NIRS4ALL_U07_INSTALLED_PYTHON",
    "NIRS4ALL_EXPERIMENTAL_UNITS_INSTALLED_PYTHON", "NIRS4ALL_PARTIAL_LATE_INSTALLED_PYTHON", "NIRS4ALL_NAMED_TORCH_INSTALLED_PYTHON",
    "NIRS4ALL_NATIVE_PLS_INSTALLED_PYTHON", "NIRS4ALL_CLASSIFICATION_INSTALLED_PYTHON", "NIRS4ALL_EARLY_LATE_INSTALLED_PYTHON",
    "NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON", "NIRS4ALL_TORCH_TOPOLOGY_INSTALLED_PYTHON", "NIRS4ALL_XL03_INSTALLED_PYTHON",
    "NIRS4ALL_STUDIO_INSTALLED_PYTHON",
)
# CLI parity, native fit probes, XL03 captures and external Octave/R require
# additional artifacts that this wheel interpreter preparation does not supply.
INSTALLED_REQUIRE_FLAGS = (
    "NIRS4ALL_REQUIRE_BY_SOURCE_INSTALLED", "NIRS4ALL_REQUIRE_CV_WEIGHT_INSTALLED", "NIRS4ALL_REQUIRE_EXPERIMENTAL_UNITS_INSTALLED",
    "NIRS4ALL_REQUIRE_PARTIAL_LATE_INSTALLED", "NIRS4ALL_REQUIRE_NAMED_TORCH_INSTALLED", "NIRS4ALL_REQUIRE_CLASSIFICATION_INSTALLED",
    "NIRS4ALL_REQUIRE_EARLY_LATE_INSTALLED", "NIRS4ALL_REQUIRE_STRUCTURAL_HPO_INSTALLED", "NIRS4ALL_REQUIRE_TORCH_TOPOLOGY_INSTALLED",
    "NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2", "NIRS4ALL_REQUIRE_NATIVE_PLS_PHASE_CONTROLS", "NIRS4ALL_REQUIRE_NATIVE_PLS_FOLD_HPO",
    "NIRS4ALL_REQUIRE_NATIVE_METHODS_HPO", "NIRS4ALL_REQUIRE_PORTABLE_ARCHIVE_V2", "NIRS4ALL_REQUIRE_N4M",
)


def export_installed_interpreters(python: Path, github_env: Path) -> None:
    """Configure cold API profiles after the installed runtime passes verification."""
    if "\n" in str(python) or "\r" in str(python):
        raise ValueError("Invalid GitHub interpreter path")
    with github_env.open("a") as stream:
        for name in INSTALLED_PYTHON_VARIABLES:
            stream.write(f"{name}={python}\n")
        for name in INSTALLED_REQUIRE_FLAGS:
            stream.write(f"{name}=1\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(f"Wheel must have exactly one distribution metadata: {path}")
        message = BytesParser().parsebytes(wheel.read(metadata[0]))
    return normalized(str(message["Name"])), str(message["Version"])


def sdk_payload(path: Path, workspace: Path) -> dict[str, str]:
    """Compare every SDK runtime wheel member to its actual checked-out source."""
    members: dict[str, str] = {}
    with zipfile.ZipFile(path) as wheel:
        for name in wheel.namelist():
            if not name.startswith("nirs4all/") or name.endswith("/"):
                continue
            source = workspace / name
            if not source.is_file() or source.read_bytes() != wheel.read(name):
                raise ValueError(f"SDK wheel/source payload mismatch: {name}")
            members[name] = sha256(source)
    if not members or "nirs4all/__init__.py" not in members:
        raise ValueError("SDK runtime payload absent")
    return members


def project_dependencies(roots: list[Path], output: Path) -> dict[str, str]:
    """Link external dependencies; never execute parent editables or overlay products."""
    output.mkdir()
    excluded = PRODUCT_MODULES | {"pip", "setuptools", "_distutils_hack", "sitecustomize.py", "usercustomize.py"}
    excluded |= {name.replace("-", "_") + ".libs" for name in UPSTREAMS}
    distributions = {"nirs4all", *UPSTREAMS, "pip", "setuptools"}
    inventory: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.iterdir()):
            name = path.name
            if name in excluded or name.endswith((".pth", ".egg-link")) or name.startswith(("__editable__", "__pycache__")):
                continue
            if name.endswith((".dist-info", ".egg-info")):
                metadata = path / ("METADATA" if name.endswith(".dist-info") else "PKG-INFO")
                if not metadata.is_file():
                    continue
                identity = BytesParser().parsebytes(metadata.read_bytes())
                if normalized(str(identity["Name"])) in distributions:
                    continue
            if name in inventory:
                continue
            (output / name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
            inventory[name] = str(path.resolve())
    return inventory


def prepare(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve(strict=True)
    output = args.output.resolve()
    if output == workspace or workspace in output.parents:
        raise ValueError("Installed replay interpreter must be outside the SDK workspace")
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "wheels"
    wheels.mkdir()
    commands: list[dict[str, object]] = []
    receipt: dict[str, object] = {"status": "RUNNING", "commands": commands}
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def run(command: list[str]) -> None:
        log = output / f"{len(commands):02}.log"
        print(f"Preparing installed replay: command {len(commands):02}, log {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=output, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        commands.append({"command": command, "exit_code": result.returncode, "log": str(log), "log_sha256": sha256(log)})
        if result.returncode:
            print(log.read_text(), flush=True)
            raise subprocess.CalledProcessError(result.returncode, command)

    try:
        if args.sdk_wheel:
            sdk = args.sdk_wheel.resolve(strict=True)
        else:
            run([sys.executable, "-B", "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(workspace)])
            choices = list(wheels.glob("nirs4all-*.whl"))
            if len(choices) != 1:
                raise ValueError("Expected exactly one SDK tag wheel")
            sdk = choices[0]
        if wheel_identity(sdk) != ("nirs4all", SDK_VERSION):
            raise ValueError(f"Installed U07 replay requires the exact SDK {SDK_VERSION} wheel")
        payload = sdk_payload(sdk, workspace)
        artifacts = {"nirs4all": sdk}
        if args.upstream_manifest:
            manifest = json.loads(args.upstream_manifest.read_text())
            for name, version in UPSTREAMS.items():
                item = manifest[name]
                path = Path(item["path"]).resolve(strict=True)
                if item["public"] is not True or item["version"] != version or sha256(path) != item["sha256"]:
                    raise ValueError(f"Public upstream receipt mismatch: {name}")
                artifacts[name] = path
        else:
            run([sys.executable, "-B", "-m", "pip", "download", "--only-binary=:all:", "--no-deps", "--dest", str(wheels),
                 *(f"{name}=={version}" for name, version in UPSTREAMS.items())])
            for path in wheels.glob("*.whl"):
                name, version = wheel_identity(path)
                if name in UPSTREAMS and version == UPSTREAMS[name]:
                    artifacts[name] = path
        for name, version in UPSTREAMS.items():
            if name not in artifacts or wheel_identity(artifacts[name]) != (name, version):
                raise ValueError(f"Missing exact public upstream wheel: {name}=={version}")
        child = output / "child"
        venv.EnvBuilder(with_pip=True).create(child)
        python = child / "bin/python"
        purelib = Path(subprocess.check_output([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        roots = args.shared_site_packages or [Path(p) for p in sys.path if Path(p).is_dir() and Path(p).name == "site-packages"]
        projection = output / "dependencies"
        receipt["dependency_projection"] = project_dependencies([p.resolve(strict=True) for p in roots], projection)
        (purelib / "external-dependencies.pth").write_text(str(projection) + "\n")
        run([str(python), "-I", "-B", "-m", "pip", "install", "--no-deps", "--force-reinstall", *(str(path) for path in artifacts.values())])
        run([str(python), "-I", "-B", "-m", "pip", "check"])
        expected = {"nirs4all": SDK_VERSION, **UPSTREAMS}
        proof = output / "installed-origins.json"
        code = '''import hashlib, importlib, importlib.metadata, json, pathlib, sysconfig
purelib=pathlib.Path(sysconfig.get_path('purelib')).resolve()
expected=json.loads(sys.argv[1]);payload=json.loads(sys.argv[2])
for name,version in expected.items():
 assert importlib.metadata.version(name)==version,(name,importlib.metadata.version(name))
origins={}
for name in ['nirs4all','n4m','dag_ml','dag_ml._dag_ml','nirs4all_io','nirs4all_core']:
 p=pathlib.Path(importlib.import_module(name).__file__).resolve();assert purelib in p.parents,(name,p);origins[name]=str(p)
for name,sha in payload.items():
 assert hashlib.sha256((purelib/name).read_bytes()).hexdigest()==sha,name
import n4m
assert n4m.abi_version()==(2,17,0)
assert n4m.version()=='1.3.2+abi.2.17.0',n4m.version()
library=pathlib.Path(n4m.library_path()).resolve()
pathlib.Path(sys.argv[3]).write_text(json.dumps({'versions':expected,'origins':origins,'sdk_payload_members':len(payload),'abi':[2,17,0],
 'native_version':n4m.version(),'native_library':str(library),'native_library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()},indent=2)+'\\n')
'''
        run([str(python), "-I", "-B", "-c", "import sys\n" + code, json.dumps(expected), json.dumps(payload), str(proof)])
        if args.github_env:
            export_installed_interpreters(python, args.github_env)
        receipt.update(status="PASS", python=str(python), wheel_artifacts={name: {"path": str(path), "sha256": sha256(path)} for name, path in artifacts.items()},
                       sdk_payload_members=len(payload), installed_proof={"path": str(proof), "sha256": sha256(proof)})
        return receipt
    except BaseException as error:
        receipt.update(status="FAIL", error=str(error))
        raise
    finally:
        (output / "prerequisites.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    parser.add_argument("--sdk-wheel", type=Path)
    parser.add_argument("--upstream-manifest", type=Path, help="Already downloaded public wheels with version/SHA/origin receipts")
    parser.add_argument("--shared-site-packages", type=Path, action="append")
    args = parser.parse_args()
    receipt = prepare(args)
    print(json.dumps({"status": receipt["status"], "python": receipt["python"], "receipt": str(args.output / "prerequisites.json")}))


if __name__ == "__main__":
    main()

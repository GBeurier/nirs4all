#!/usr/bin/env python3
"""Prepare real U15 prerequisites after the candidate-native CI action.

Uses released npm modules, the public Core crate and the pinned Methods/DAG
checkouts. The existing DAG witness performs the fits and records its capture;
this helper implements no models, fixture generator or archive protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

METHODS_COMMIT = "dcc570b3647f77cf0428dd346078f442ed5cd032"
DAG_COMMIT = "9095e5640c53b02dd91dc4de7a20d1a7d501bbdb"
CORE_VERSION = "0.4.2"
NPM_PACKAGES = {"@nirs4all/methods": "1.3.2", "dag-ml-wasm": "0.3.37"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str], logs: Path) -> dict[str, object]:
    """Keep complete command output and propagate every failed prerequisite."""
    log = logs / f"{len(list(logs.glob('*.log'))):02}.log"
    print("Preparing U15:", subprocess.list2cmdline(command), flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    receipt: dict[str, object] = {"command": command, "exit_code": result.returncode, "log": str(log), "log_sha256": sha256(log)}
    log.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        print(log.read_text(), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    return receipt


def require_commit(repo: Path, expected: str) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    changed = subprocess.check_output(["git", "diff", "--name-only", expected, "--"], cwd=repo, text=True).splitlines()
    # Methods' public-wheel retry changed only this workflow after its release
    # tag. Retain the real HEAD while proving the runtime/source tuple identical.
    permitted = {".github/workflows/release-wheels.yml"} if expected == METHODS_COMMIT else set()
    if not set(changed).issubset(permitted) or (actual != expected and not changed):
        raise ValueError(f"U15 source origin mismatch: {repo}: expected {expected}, got {actual}")
    return actual


def require_public_core_install(prefix: Path) -> Path:
    """Check Cargo's installed registry package identity, without guessing CLI flags."""
    key = f"nirs4all {CORE_VERSION} (registry+https://github.com/rust-lang/crates.io-index)"
    installs = json.loads((prefix / ".crates2.json").read_text())["installs"]
    item = installs.get(key)
    binary = prefix / "bin/nirs4all-core-archive"
    if item is None or item["version_req"] != f"={CORE_VERSION}" or item["bins"] != ["nirs4all-core-archive"] or not os.access(binary, os.X_OK):
        raise ValueError(f"U15 requires the exact public registry Core {CORE_VERSION} CLI installation")
    return binary.resolve(strict=True)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve(strict=True)
    methods = (args.methods_root or workspace / "nirs4all-methods-src").resolve(strict=True)
    dag = (args.dag_root or workspace / "dag-ml-cli-src").resolve(strict=True)
    methods_head = require_commit(methods, METHODS_COMMIT)
    dag_head = require_commit(dag, DAG_COMMIT)
    octave = args.octave.resolve(strict=True)
    if not os.access(octave, os.X_OK):
        raise ValueError("Octave must be an executable runtime")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    commands: list[dict[str, object]] = []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    receipt: dict[str, object] = {"status": "RUNNING", "commands": commands, "methods_release_commit": METHODS_COMMIT,
                               "methods_actual_commit": methods_head, "dag_commit": dag_head}
    try:
        library = args.library.resolve(strict=True) if args.library else methods / "build/dev-release/cpp/src/libn4m.so"
        generated = methods / "build/dev-release/generated"
        for path in (library, generated / "n4m/n4m_export.h"):
            if not path.is_file():
                raise FileNotFoundError(f"candidate-native prerequisite absent: {path}")
        mex = output / "matlab"
        shutil.copytree(methods / "bindings/matlab", mex, ignore=shutil.ignore_patterns("*.mex", "*.mexa64", "*.mexw64"))
        env.update(N4M_INCLUDE_DIR=str(methods / "cpp/include"), N4M_GENERATED_DIR=str(generated), N4M_LIB_DIR=str(library.parent),
                   DAG_ML_OCTAVE_METHODS_PATH=str(mex), DAG_ML_OCTAVE_MEX_PATH=str(mex), DAG_ML_OCTAVE=str(octave))
        env["LD_LIBRARY_PATH"] = str(library.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        commands.append(run([str(octave), "--version"], cwd=output, env=env, logs=logs))
        commands.append(run([str(octave), "--quiet", "--no-gui", "--no-init-file", "--eval",
                             "addpath(pwd); build_mex({'n4m_role_pipeline_mex','n4m_version_mex'}); "
                             "assert(strcmp(n4m.version(),'1.3.2+abi.2.17.0'));"], cwd=mex, env=env, logs=logs))
        npm = output / "npm"
        npm.mkdir()
        commands.append(run(["npm", "install", "--prefix", str(npm), "--ignore-scripts", "--no-audit", "--no-fund", "--save-exact",
                             *(f"{name}@{version}" for name, version in NPM_PACKAGES.items())], cwd=output, env=env, logs=logs))
        for name, version in NPM_PACKAGES.items():
            actual = json.loads((npm / "node_modules" / name / "package.json").read_text())["version"]
            if actual != version:
                raise ValueError(f"Unexpected public npm version for {name}: {actual}")
        capture = output / "four-source-node-capture.json"
        commands.append(run(["node", str(dag / "scripts/smoke_wasm_multimodal_methods_hpo.mjs"),
                             str(npm / "node_modules/dag-ml-wasm"), str(npm / "node_modules/@nirs4all/methods/dist"), str(capture)],
                            cwd=dag, env=env, logs=logs))
        evidence = json.loads(capture.read_text())
        if evidence["dagml_version"] != "0.3.37" or evidence["methods_version"] != "1.3.2+abi.2.17.0" or len(evidence["sampleIds"]) != 12:
            raise ValueError("Native four-source capture has incompatible runtime identity")
        core = args.core_prefix.resolve() if args.core_prefix else output / "core"
        if args.core_prefix is None:
            commands.append(run(["cargo", "install", "nirs4all", "--version", f"={CORE_VERSION}", "--locked", "--bin", "nirs4all-core-archive",
                                 "--root", str(core), "--jobs", "2"], cwd=output, env=env, logs=logs))
        core_cli = require_public_core_install(core)
        exported = {"NIRS4ALL_CI_U15_DAG_ROOT": str(dag), "NIRS4ALL_CI_U15_OCTAVE": str(octave),
                    "NIRS4ALL_CI_U15_NODE_CAPTURE": str(capture), "DAG_ML_OCTAVE": str(octave),
                    "DAG_ML_OCTAVE_METHODS_PATH": str(mex), "DAG_ML_OCTAVE_MEX_PATH": str(mex),
                    "NIRS4ALL_CORE_ARCHIVE_CLI": str(core_cli)}
        if args.github_env:
            if any("\n" in value or "\r" in value for value in exported.values()):
                raise ValueError("Invalid multiline GitHub environment path")
            with args.github_env.open("a") as stream:
                stream.writelines(f"{key}={value}\n" for key, value in exported.items())
        receipt.update(status="PASS", environment=exported, artifacts={str(p): sha256(p) for p in
                       (capture, library.resolve(strict=True), core_cli, npm / "package-lock.json",
                        *sorted((mex / "+n4m").glob("*.mex")))})
        return receipt
    except BaseException as error:
        receipt.update(status="FAIL", error=str(error))
        raise
    finally:
        (output / "prerequisites.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--octave", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    parser.add_argument("--core-prefix", type=Path, help="Reuse an exact existing Cargo registry installation during qualification")
    parser.add_argument("--methods-root", type=Path, help="Existing qualified Methods checkout (default: candidate-native layout)")
    parser.add_argument("--dag-root", type=Path, help="Existing qualified DAG checkout (default: candidate-native layout)")
    parser.add_argument("--library", type=Path, help="Matching released libn4m (default: candidate-native build)")
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

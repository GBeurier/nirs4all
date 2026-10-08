#!/usr/bin/env python3
"""Verify retained local qualification against current, reviewed runtime inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import cast

SCHEMA = "nirs4all.local-qualification.v1"
PROJECTS = {
    "dag": ("GBeurier/dag-ml", ("crates/", "bindings/", "scripts/", "examples/", "parity/", "tests/", "include/", "docs/contracts/")),
    "core": ("GBeurier/nirs4all-core", ("bindings/", "scripts/", "tests/", "compat/")),
    "sdk": ("GBeurier/nirs4all", ("nirs4all/", "tests/", "scripts/", "examples/")),
    "studio": ("GBeurier/nirs4all-studio", ("sidecar/", "src/", "backend/", "api/", "tests/", "e2e/", "scripts/", "electron/", "build/constraints/")),
}
ROOT_INPUTS = {"Cargo.toml", "Cargo.lock", "pyproject.toml", "package.json", "package-lock.json", "Makefile", "requirements.txt", "requirements-dev.txt", "requirements-test.txt", "requirements-examples.txt", "build/runtime-cohort.json", "recommended-config.json"}
EXCLUDED = {
    "scripts/verify_local_qualification.py",
    "compat/local-qualification.json",
    "tests/test_verify_local_qualification.py",
}
POLICY = "qualification/policy.json"


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def relative_file(root: Path, value: str) -> Path:
    require(isinstance(value, str) and value and "\\" not in value, "invalid relative proof path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in value, "proof path escapes root")
    candidate = root.joinpath(*path.parts)
    require(candidate.is_file() and not candidate.is_symlink(), f"missing or symlink file: {value}")
    require(candidate.resolve().is_relative_to(root.resolve()), f"proof path escapes root: {value}")
    return candidate


def tracked_inputs(root: Path) -> dict[str, str]:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    require(POLICY in files, "qualification policy must be tracked")
    return {name: digest(relative_file(root, name)) for name in sorted(set(files)) if name and name not in EXCLUDED}


def runtime_inputs(root: Path, project: str, tracked: dict[str, str] | None = None) -> dict[str, str]:
    prefixes = PROJECTS[project][1]
    tracked = tracked_inputs(root) if tracked is None else tracked
    paths = sorted(
        {
            name
            for name in tracked
            if name
            and name not in EXCLUDED
            and (name in ROOT_INPUTS or name.startswith(prefixes) or (project == "studio" and name.startswith(("electron-builder", "vite.config.", "tsconfig", "tailwind.config.", "postcss.config."))))
        }
    )
    require(len(paths) > 1, "runtime input scope is empty")
    return {name: tracked[name] for name in paths}


def gate_inputs(current: dict[str, str], gate: dict) -> dict[str, str]:
    paths = gate.get("input_paths", [])
    require(isinstance(paths, list) and paths and all(isinstance(p, str) and p for p in paths), "mandatory gate input scope is missing")
    for path in paths:
        require(any(name.startswith(path) for name in current) if path.endswith("/") else path in current, f"mandatory gate input is absent: {path}")
    excluded = set(gate.get("input_exclusions", []))
    selected = {name: value for name, value in current.items() if name not in excluded and any(name.startswith(path) if path.endswith("/") else name == path for path in paths)}
    require(selected, "mandatory gate input scope selects no runtime sources")
    return selected


def canonical_command(command: list[str], cwd: str, gate: dict, os_family: str, controlled_history: bool = False) -> list[str]:
    command = list(command)
    if len(command) >= 2 and command[0].replace("\\", "/").rsplit("/", 1)[-1] in {"rtk", "rtk.exe"} and command[1] == "proxy":
        require(command[0] in gate.get("wrapper_paths", {}).get(os_family, []), "RTK wrapper path is not anchored in policy")
        command = command[2:]
    require(command, "missing command after RTK prefix")
    configured = gate.get("executable_paths", {}).get(os_family)
    if not controlled_history and (configured is not None or "/" in command[0] or "\\" in command[0]):
        require(command[0].replace(cwd, "{root}") in gate.get("executable_paths", {}).get(os_family, []), "executable path is not anchored in policy")
    executable = command[0].replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".exe")
    if re.fullmatch(r"python(?:3(?:\.[0-9]+)?)?", executable):
        executable = "python"
    arguments = []
    for argument in command[1:]:
        normalized = argument.replace(cwd, "{root}")
        if normalized != argument and normalized.startswith("{root}"):
            normalized = normalized.replace("\\", "/")
        arguments.append(normalized)
    return [executable, *arguments]


def integer(value, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{field} must be an integer >= {minimum}")
    return value


def timestamp(value: str):
    require(isinstance(value, str), "timestamp must be text")
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timestamps must include timezone")
    return result


def evidence(root: Path, row: dict | None, label: str) -> Path:
    require(isinstance(row, dict), f"missing {label} evidence")
    row = cast(dict, row)
    require(str(row.get("path", "")).startswith("qualification/"), f"{label} must be retained under qualification/")
    path = relative_file(root, row["path"])
    minimum = 0 if label.endswith("log") else 1
    require(integer(row.get("bytes"), f"{label}.bytes", minimum) == path.stat().st_size, f"{label} size mismatch")
    require(row.get("sha256") == digest(path), f"{label} hash mismatch")
    return path


RUN_IDENTITY = ("id", "host", "command", "cwd", "environment", "started_at", "finished_at", "exit_code", "input_fingerprints")


def json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_controlled_history(root: Path, inputs: dict, run: dict, selected: dict, source_sha: str, gate: dict, os_family: str) -> None:
    authority = gate.get("historical_authority", {}).get(os_family, {})
    require(authority.get("format") == "controlled-runner-duration.v1", "historical duration authority is not reviewed")
    require(re.fullmatch("[0-9a-f]{40}", str(authority.get("source_sha", ""))) is not None, "historical source SHA is invalid")
    historical = load(evidence(root, inputs.get("historical_manifest"), "historical_manifest"))
    continuity = load(evidence(root, inputs.get("continuity_report"), "continuity_report"))
    require(historical.get("format") == authority["format"], "wrong historical evidence format")
    require(historical.get("source_sha") == authority.get("source_sha"), "historical source differs from reviewed authority")
    require(historical.get("source_basis") == authority.get("source_basis") and authority.get("source_basis") in {"reconstructed_from_reviewed_tree", "observed_full_map_verification"}, "historical source basis is missing or relabelled")
    require(historical.get("input_fingerprints") == selected and json_digest(selected) == authority.get("scoped_inputs_sha256"), "historical scoped inputs differ from reviewed source")
    descriptors = historical.get("evidence", {})
    required = {"raw_receipt", "direct_terminal", "runner", "child_wrapper", "source_freeze"}
    require(required <= set(authority.get("evidence_sha256", {})) and set(descriptors) == set(authority["evidence_sha256"]), "original historical evidence is incomplete")
    paths = {}
    for name, descriptor in descriptors.items():
        require(descriptor.get("sha256") == authority["evidence_sha256"][name], f"historical authority hash differs: {name}")
        paths[name] = evidence(root, descriptor, name)
    original, direct = load(paths["raw_receipt"]), load(paths["direct_terminal"])
    require(original.get("gate") == run["id"] and original.get("command") == run["command"] == direct.get("command"), "historical gate/command identity differs")
    require(type(original.get("exit_code")) is int and original["exit_code"] == 0 and type(direct.get("direct_command_exit_code")) is int and direct["direct_command_exit_code"] == 0 and original.get("stop_reason") is None, "historical direct command failed or stopped")
    terminal = original.get("independent_command_terminal", original.get("direct_terminal"))
    require(terminal == direct, "historical independent terminal differs from original receipt")
    active = load(paths["active"]) if "active" in paths else {}
    require(run["cwd"] == original.get("cwd", active.get("worktree")), "historical cwd is not supported by original evidence")
    if active:
        require(active.get("command") == direct["command"], "historical active command differs")
    source_freeze = load(paths["source_freeze"])
    require(source_freeze.get(authority.get("source_tree_field", "")) == authority["source_sha"], "historical reviewed source tree differs")
    require(authority.get("source_verified_field") in {"functional_source_unchanged", "source_unchanged_after"} and original.get(authority["source_verified_field"]) is True, "historical runner did not verify its declared source scope")
    if "source_tree" in original:
        require(original["source_tree"] == authority["source_sha"], "original receipt source tree differs")
    require(authority.get("environment_basis") == "derived_from_retained_controlled_runner" and isinstance(authority.get("environment"), dict) and run.get("environment") == authority["environment"], "historical environment is not tied to reviewed runner")
    seconds = direct.get("seconds")
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0 and run.get("duration_seconds") == seconds, "historical observed duration differs")
    require(run.get("timing_basis") == "historical-duration" and run.get("started_at") is None and run.get("finished_at") is None, "historical wall timestamps must not be fabricated")
    logs = historical.get("retained_logs", {})
    require(set(logs) == set(original.get("logs", {})), "historical log inventory differs")
    for name, descriptor in logs.items():
        record = original["logs"][name]
        require(all(descriptor.get(field) == record.get(field) for field in ("bytes", "sha256")), "retained historical log differs from original")
        evidence(root, descriptor, "historical log")
    require(run.get("log") == logs.get(authority.get("primary_log")), "historical run log is not its original log")
    require(continuity.get("historical_manifest_sha256") == inputs["historical_manifest"]["sha256"] and continuity.get("historical_source_sha") == authority["source_sha"] and continuity.get("current_source_sha") == source_sha and continuity.get("input_fingerprints") == selected, "historical continuity is not bound")
    report = load(evidence(root, run.get("report"), "report"))
    observations = {field: report.get(field) for field in ("summary", "skips", "facts", "provenance", "raw_reports")}
    require(json_digest(observations) == authority.get("observations_sha256"), "historical observations differ from reviewed raw evidence")


def check_input_evidence(root: Path, inputs: dict, run: dict, selected: dict, source_sha: str, gate: dict | None = None, os_family: str = "") -> None:
    require(inputs.get("input_fingerprints") == selected, "source capture does not match scoped runtime inputs")
    require(inputs.get("mode") in {"captured", "inherited"}, "source capture mode is missing")
    if inputs.get("lineage_format") == "controlled-runner-duration.v1":
        require(inputs["mode"] == "inherited" and gate is not None, "controlled history is only for reviewed inherited evidence")
        check_controlled_history(root, inputs, run, selected, source_sha, gate or {}, os_family)
        return
    identity = {field: run.get(field) for field in RUN_IDENTITY}
    if inputs["mode"] == "captured":
        require(inputs.get("run_identity") == identity, "source capture belongs to a different run")
        require(inputs.get("source_sha") == source_sha, "source capture source SHA mismatch")
        return
    historical = load(evidence(root, inputs.get("historical_manifest"), "historical_manifest"))
    continuity = load(evidence(root, inputs.get("continuity_report"), "continuity_report"))
    original = historical.get("run", {})
    require(isinstance(original, dict) and all(original.get(field) == run.get(field) for field in RUN_IDENTITY), "historical run identity differs")
    require(type(original.get("exit_code")) is int and original["exit_code"] == 0, "historical direct exit is not zero")
    historical_sha = historical.get("source_sha", "")
    require(re.fullmatch("[0-9a-f]{40}", str(historical_sha)) is not None, "historical source SHA is missing")
    captured_descriptor = original.get("input_evidence")
    captured = load(evidence(root, captured_descriptor, "historical source capture"))
    require(captured.get("mode") == "captured", "historical input evidence must be an original capture")
    check_input_evidence(root, captured, original, selected, historical_sha)
    require(historical.get("input_fingerprints") == selected and continuity.get("input_fingerprints") == selected, "historical evidence does not prove unchanged scoped inputs")
    require(continuity.get("historical_manifest_sha256") == inputs["historical_manifest"]["sha256"], "continuity historical manifest is not bound")
    require(continuity.get("captured_inputs_sha256") == captured_descriptor["sha256"], "continuity original capture is not bound")
    require(continuity.get("historical_source_sha") == historical_sha and continuity.get("current_source_sha") == source_sha, "continuity source lineage is not bound")
    require(original.get("log") == run.get("log"), "inherited run log differs from original")
    evidence(root, original.get("log"), "historical log")
    original_report = load(evidence(root, original.get("report"), "historical report"))
    for field in (*RUN_IDENTITY, "input_evidence"):
        require(original_report.get(field) == original.get(field), f"historical report/run identity mismatch: {field}")
    current_report = load(evidence(root, run.get("report"), "report"))
    for field in ("summary", "skips", "facts", "provenance", "raw_reports"):
        require(current_report.get(field) == original_report.get(field), f"inherited report observations differ: {field}")


def check_provenance(root: Path, provenance: dict, gate: dict, run: dict, source_sha: str, os_family: str) -> None:
    require(isinstance(provenance.get("tools"), dict) and provenance["tools"], "missing native tool provenance")
    requirements = gate.get("provenance_requirements", {})
    for field in ("source_artifacts", "dependency_origins"):
        rows = provenance.get(field, [])
        expected = requirements.get(field, [])
        if isinstance(expected, dict):
            expected = expected.get(os_family, [])
        require(isinstance(rows, list) and rows, f"missing provenance {field}")
        require(isinstance(expected, list) and expected, f"mandatory provenance cohort is not anchored: {field}")
        for row in rows:
            require(isinstance(row, dict) and all(isinstance(row.get(k), str) and row[k] for k in ("name", "version", "origin")), "incomplete native origin")
            require(re.fullmatch("[0-9a-f]{64}", str(row.get("sha256", ""))) is not None, "missing native artifact hash")
            integer(row.get("bytes"), "native artifact.bytes", 1)
            capture = load(evidence(root, row.get("evidence"), "native artifact capture"))
            require(capture.get("artifact") == {k: row.get(k) for k in ("name", "version", "origin", "sha256", "bytes", "artifact_path")}, "native artifact capture differs from origin")
            require(capture.get("run_identity") == {k: run.get(k) for k in RUN_IDENTITY}, "native artifact capture belongs to a different run")
            require(capture.get("source_sha") == source_sha and capture.get("log_sha256") == run["log"]["sha256"], "native artifact source/log lineage differs")
        for requirement in expected:
            require(isinstance(requirement, dict) and all(isinstance(requirement.get(k), str) and requirement[k] for k in ("name", "version", "origin", "artifact_path")), "mandatory provenance cohort is incomplete")
            require(re.fullmatch("[0-9a-f]{64}", str(requirement.get("sha256", ""))) is not None, "mandatory artifact SHA is not anchored")
            integer(requirement.get("bytes"), "required artifact.bytes", 1)
            matches = [row for row in rows if all(row.get(k) == requirement[k] for k in ("name", "version", "origin", "sha256", "bytes", "artifact_path"))]
            require(len(matches) == 1, "native artifact does not match required publication cohort")
            path = Path(requirement["artifact_path"])
            if not path.is_absolute():
                path = root / path
            if path.exists():
                require(path.is_file() and not path.is_symlink(), "live native artifact is not a regular file")
                require(path.stat().st_size == requirement["bytes"] and digest(path) == requirement["sha256"], "live native artifact bytes mismatch")


def fact_check(facts: dict, rule: dict) -> None:
    field, op, expected = rule["field"], rule["op"], rule.get("value")
    require(field in facts, f"missing measured fact: {field}")
    actual = facts[field]
    if op == "finite_nonempty":
        require(isinstance(actual, list) and actual and all(type(item) in (int, float) and math.isfinite(item) for item in actual), f"{field} must contain finite numeric observations")
        return
    if type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual), f"{field} must be finite numeric data")
    else:
        require(type(actual) is type(expected), f"invalid measured fact: {field}")
    if op == "eq":
        passed = actual == expected
    elif op == "min":
        passed = actual >= expected
    elif op == "max":
        passed = actual <= expected
    else:
        raise ValueError(f"unknown policy fact comparison: {op}")
    require(passed, f"measured fact failed: {field} {op} {expected!r}, got {actual!r}")


def check_playwright_report(root: Path, report: dict, gate: dict) -> None:
    raw = load(evidence(root, report.get("raw_reports", {}).get("playwright"), "Playwright JSON"))
    stats = raw.get("stats", {})
    require(stats.get("expected") == gate["minimum_passed"], "Playwright actual expected count differs")
    require(all(type(stats.get(field)) is int and stats[field] == 0 for field in ("unexpected", "skipped", "flaky")), "Playwright failures, skips or flakes are present")
    require(raw.get("errors") == [], "Playwright actual errors must be empty")
    tests = []

    def visit(suites):
        for suite in suites:
            for spec in suite.get("specs", []):
                tests.extend(spec.get("tests", []))
            visit(suite.get("suites", []))

    visit(raw.get("suites", []))
    require(len(tests) == stats["expected"], "Playwright test inventory differs from declared count")
    for test in tests:
        require(test.get("expectedStatus") == "passed" and test.get("status") == "expected", "Playwright test was not expected to pass")
        results = test.get("results", [])
        require(results and all(result.get("status") == "passed" for result in results), "Playwright test lacks a completed passing result")


def verify(root: Path, project: str, receipt_path: Path) -> dict:
    require(project in PROJECTS, "unknown project")
    root = root.resolve()
    policy = load(relative_file(root, POLICY))
    require(policy.get("schema") == SCHEMA and policy.get("project") == project, "wrong qualification policy")
    require(not policy.get("proposal_incomplete_not_for_publication", False), "qualification policy is explicitly incomplete")
    receipt = load(receipt_path)
    require(receipt.get("schema") == SCHEMA and receipt.get("project") == project, "wrong receipt schema/project")
    require(receipt.get("repository") == PROJECTS[project][0], "wrong receipt repository")
    require(re.fullmatch("[0-9a-f]{40}", str(receipt.get("source_sha", ""))) is not None, "invalid informational source SHA")
    require(receipt.get("execution") == {"location": "local", "github_actions": False, "performance_policy": "strict"}, "qualification must be strict and local")
    require(receipt.get("policy_sha256") == digest(root / POLICY), "qualification policy digest mismatch")
    tracked = tracked_inputs(root)
    current = runtime_inputs(root, project, tracked)
    require(receipt.get("input_fingerprints") == current, "stale, missing or unexpected runtime fingerprints")
    hosts = {}
    for host in receipt.get("hosts", []):
        require(isinstance(host, dict) and host.get("id") not in hosts, "duplicate or invalid host")
        require(host.get("os_family") in {"linux", "windows"}, "unsupported local host OS")
        for field in ("id", "os", "arch", "hostname"):
            require(isinstance(host.get(field), str) and host[field], f"missing host {field}")
        integer(host.get("cpu_count"), "host.cpu_count", 1)
        integer(host.get("ram_bytes"), "host.ram_bytes", 1)
        require(isinstance(host.get("tools"), dict) and host["tools"] and all(isinstance(v, str) and v for v in host["tools"].values()), "missing host tool versions")
        hosts[host["id"]] = host
    requirements = {(gate["id"], os_family): gate for gate in policy["gates"] for os_family in gate["hosts"]}
    require(requirements, "no mandatory local qualification gates")
    require(len(requirements) == sum(len(gate["hosts"]) for gate in policy["gates"]), "duplicate gate/OS policy requirement")
    require({os_family for _, os_family in requirements} == {"linux", "windows"}, "complete local qualification requires Linux and Windows")
    completed = set()
    for run in receipt.get("runs", []):
        require(run.get("host") in hosts, "run references missing host")
        key = run.get("id"), hosts[run["host"]]["os_family"]
        require(key in requirements and key not in completed, "unknown or duplicate qualification run")
        gate = requirements[key]
        command = run.get("command")
        require(isinstance(command, list) and command and all(isinstance(v, str) and v for v in command), "missing direct command")
        require(isinstance(run.get("cwd"), str) and run["cwd"], "missing command cwd")
        expected_command = gate.get("commands", {}).get(key[1])
        require(isinstance(expected_command, list) and expected_command, "mandatory gate command is not anchored in policy")
        inputs = load(evidence(root, run.get("input_evidence"), "input evidence"))
        controlled_history = inputs.get("mode") == "inherited" and inputs.get("lineage_format") == "controlled-runner-duration.v1"
        require(canonical_command(command, run["cwd"], gate, key[1], controlled_history) == expected_command, "direct command differs from required gate")
        for field, expected in gate.get("required_environment", {}).items():
            actual = run.get("environment", {}).get(field)
            require(isinstance(actual, str) and actual.replace(run["cwd"], "{root}") == expected, f"required execution environment differs: {field}")
        selected = gate_inputs(tracked, gate)
        require(run.get("input_fingerprints") == selected, "run source fingerprint mismatch")
        check_input_evidence(root, inputs, run, selected, receipt["source_sha"], gate, key[1])
        if not controlled_history:
            require(timestamp(run["finished_at"]) >= timestamp(run["started_at"]), "run finishes before it starts")
        require(type(run.get("exit_code")) is int and run["exit_code"] == 0, "nonzero or invalid direct exit code")
        evidence(root, run.get("log"), "log")
        report = load(evidence(root, run.get("report"), "report"))
        for field in (*RUN_IDENTITY, "input_evidence", "timing_basis", "duration_seconds"):
            require(report.get(field) == run.get(field), f"report/run identity mismatch: {field}")
        summary = report.get("summary", {})
        require(integer(summary.get("failed"), "failed") == 0, "failed tests are present")
        integer(summary.get("passed"), "passed", gate.get("minimum_passed", 0))
        skips = report.get("skips", [])
        require(isinstance(skips, list) and len(set(skips)) == len(skips), "duplicate or invalid skip IDs")
        require(set(skips) <= set(gate.get("allowed_skips", [])), "unapproved skipped/ignored tests")
        require(integer(summary.get("skipped"), "skipped") == len(skips), "skip count/identity mismatch")
        facts = report.get("facts", {})
        require(isinstance(facts, dict), "missing numeric qualification facts")
        for rule in gate.get("facts", []):
            fact_check(facts, rule)
        if gate.get("report_format") == "playwright-json":
            check_playwright_report(root, report, gate)
        if gate.get("requires_provenance", False):
            tested_source = receipt["source_sha"] if inputs["mode"] == "captured" else load(evidence(root, inputs["historical_manifest"], "historical_manifest"))["source_sha"]
            check_provenance(root, report.get("provenance", {}), gate, run, tested_source, key[1])
        completed.add(key)
    require(completed == set(requirements), f"missing mandatory local gates: {sorted(set(requirements) - completed)}")
    return {"project": project, "verified_runtime_inputs": len(current), "verified_local_gates": len(completed), "receipt_sha256": digest(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=tuple(PROJECTS), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        result = verify(args.root, args.project, args.receipt)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Local qualification refused: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

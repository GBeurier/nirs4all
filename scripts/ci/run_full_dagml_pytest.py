"""Run every pytest module under DAG-ML in a fresh process.

The test suite loads TensorFlow, JAX and PyTorch. A long-lived xdist worker can
crash after loading more than one native runtime, so this runner isolates each
module while still executing every collected test without marker/path filters.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _test_files(paths: list[Path]) -> list[Path]:
    files = set()
    for path in paths:
        if path.is_file():
            if path.name.startswith("test_") and path.suffix == ".py":
                files.add(path.resolve())
        elif path.is_dir():
            files.update(file.resolve() for file in path.rglob("test_*.py"))
        else:
            raise ValueError(f"Test path does not exist: {path}")
    return sorted(files)


def _error_suite(name: str, message: str) -> ET.Element:
    suite = ET.Element("testsuite", name=name)
    case = ET.SubElement(suite, "testcase", classname="dagml_ci", name="pytest_process")
    ET.SubElement(case, "error", message=message).text = message
    return suite


def _read_suites(path: Path, name: str) -> list[ET.Element]:
    if not path.exists():
        return [_error_suite(name, "pytest produced no JUnit report")]
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [_error_suite(name, f"Invalid JUnit report: {exc}")]
    if root.tag == "testsuite":
        return [root]
    suites = list(root.findall("testsuite")) if root.tag == "testsuites" else []
    return suites or [_error_suite(name, "JUnit report contains no testsuite")]


def _counts(suites: list[ET.Element]) -> dict[str, int]:
    cases = [case for suite in suites for case in suite.iter("testcase")]
    return {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }


def _report_failure(suites: list[ET.Element], name: str, message: str) -> None:
    suites.append(_error_suite(name, message))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("tests")])
    parser.add_argument("--report-dir", type=Path, default=Path("dagml-pytest-report"))
    parser.add_argument("--coverage-output", type=Path, default=Path("coverage.xml"))
    parser.add_argument("--process-timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    try:
        files = _test_files(args.paths)
    except ValueError as exc:
        parser.error(str(exc))
    if not files:
        parser.error("No test_*.py files found")

    report_dir = args.report_dir.resolve()
    junit_dir = report_dir / "junit-files"
    log_dir = report_dir / "logs"
    coverage_dir = report_dir / "coverage-files"
    for directory in (junit_dir, log_dir, coverage_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_suites: list[ET.Element] = []
    results = []
    started = time.monotonic()
    for index, file in enumerate(files, 1):
        name = file.as_posix()
        stem = f"{index:04d}-{file.stem}"
        junit = junit_dir / f"{stem}.xml"
        manifest = report_dir / f"{stem}.json"
        log = log_dir / f"{stem}.log"
        env = os.environ.copy()
        env["N4A_ENGINE"] = "dag-ml"
        env["N4A_PYTEST_MANIFEST"] = str(manifest)
        env["COVERAGE_FILE"] = str(coverage_dir / f".coverage.{index:04d}")
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env.pop("PYTEST_ADDOPTS", None)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "scripts.ci.pytest_collection_manifest",
            "--timeout=300",
            f"--junitxml={junit}",
            "--cov=nirs4all",
            "--cov-report=",
            str(file),
        ]
        print(f"[{index}/{len(files)}] {name}", flush=True)
        try:
            with log.open("w", encoding="utf-8") as output:
                process = subprocess.run(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=args.process_timeout,
                    check=False,
                )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
            with log.open("a", encoding="utf-8") as output:
                output.write(f"\nProcess timed out after {args.process_timeout}s\n")

        suites = _read_suites(junit, name)
        actual = _counts(suites)["tests"]
        expected = None
        if manifest.exists():
            try:
                node_ids = json.loads(manifest.read_text(encoding="utf-8"))
                expected = len(node_ids)
                if not isinstance(node_ids, list) or not all(
                    isinstance(node_id, str) for node_id in node_ids
                ):
                    raise ValueError("manifest is not a node ID list")
            except (ValueError, json.JSONDecodeError) as exc:
                _report_failure(suites, name, f"Invalid collection manifest: {exc}")
        else:
            _report_failure(suites, name, "pytest did not finish collection")

        # Pytest records collection-time module skips as an extra skipped JUnit
        # case; all genuinely collected node IDs must still appear in JUnit.
        module_skipped = (
            expected == 0
            and actual > 0
            and _counts(suites)["skipped"] == actual
            and returncode == 5
        )
        if expected is not None and expected > 0 and actual != expected:
            _report_failure(
                suites, name, f"Collected {expected} tests but JUnit recorded {actual}"
            )
        if expected == 0 and not module_skipped and returncode == 0:
            _report_failure(suites, name, "pytest reported success without collecting tests")
        if returncode != 0 and not module_skipped:
            _report_failure(suites, name, f"pytest exited with status {returncode}")

        counts = _counts(suites)
        all_suites.extend(suites)
        results.append(
            {
                "file": name,
                "collected": expected,
                "reported": actual,
                "exit_code": returncode,
                **counts,
            }
        )
        status = "FAIL" if counts["failures"] or counts["errors"] else "PASS"
        print(f"  {status}: collected={expected}, junit={actual}, exit={returncode}", flush=True)
        if status == "FAIL":
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)

    totals = _counts(all_suites)
    root = ET.Element("testsuites", {key: str(value) for key, value in totals.items()})
    root.extend(all_suites)
    ET.ElementTree(root).write(
        report_dir / "junit.xml", encoding="utf-8", xml_declaration=True
    )
    summary = {
        "files": len(files),
        "collected": sum(result["collected"] or 0 for result in results),
        "seconds": round(time.monotonic() - started, 1),
        **totals,
        "results": results,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"DAG-ML: {len(files)} files, {summary['collected']} collected, "
        f"{totals['tests']} JUnit cases, {totals['failures']} failures, "
        f"{totals['errors']} errors, {totals['skipped']} skipped; "
        f"reports: {report_dir}",
        flush=True,
    )
    # Each subprocess has its own coverage database; combine only after all
    # subprocesses finish, avoiding concurrent writes to a shared .coverage.
    coverage_file = report_dir / ".coverage"
    combined = subprocess.run(
        [sys.executable, "-m", "coverage", "combine", f"--data-file={coverage_file}", str(coverage_dir)],
        check=False,
    )
    if combined.returncode == 0:
        xml = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "xml",
                f"--data-file={coverage_file}",
                "-o",
                str(args.coverage_output),
            ],
            check=False,
        )
        coverage_exit = xml.returncode
    else:
        coverage_exit = combined.returncode
    return 1 if totals["failures"] or totals["errors"] or coverage_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())

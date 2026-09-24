"""End-to-end checks for the per-module DAG-ML CI runner."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def test_runner_reports_crash_and_keeps_running(tmp_path):
    (tmp_path / "test_a.py").write_text("def test_first(): assert True\n")
    (tmp_path / "test_b.py").write_text(
        "import os\ndef test_native_crash(): os._exit(139)\n"
    )
    (tmp_path / "test_c.py").write_text(
        "import pytest\npytest.skip('optional runtime', allow_module_level=True)\n"
    )
    (tmp_path / "test_d.py").write_text("def test_last(): assert True\n")
    repo = Path(__file__).resolve().parents[3]
    report = tmp_path / "report"
    (report / "junit-files").mkdir(parents=True)
    (report / "0002-test_b.json").write_text('["test_b.py::test_native_crash"]')
    (report / "junit-files" / "0002-test_b.xml").write_text(
        '<testsuite><testcase name="stale_pass"/></testsuite>'
    )
    process = subprocess.run(
        [
            sys.executable,
            "scripts/ci/run_full_dagml_pytest.py",
            str(tmp_path),
            "--report-dir",
            str(report),
            "--coverage-output",
            str(tmp_path / "coverage.xml"),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    summary = json.loads((report / "summary.json").read_text())
    assert process.returncode == 1
    assert summary["files"] == 4
    assert summary["collected"] == 3
    assert [item["exit_code"] for item in summary["results"]] == [0, 139, 5, 0]
    assert summary["results"][1]["errors"] == 1  # no stale JUnit or manifest reused
    assert summary["results"][2]["errors"] == 0
    assert summary["results"][3]["reported"] == 1
    assert summary["errors"] >= 1
    assert ET.parse(report / "junit.xml").getroot().attrib["tests"] == str(
        summary["tests"]
    )
    assert (tmp_path / "coverage.xml").exists()


def test_runner_executes_modules_concurrently_with_ordered_report(tmp_path):
    for name, other in (("a", "b"), ("b", "a")):
        (tmp_path / f"test_{name}.py").write_text(
            "from pathlib import Path\n"
            "import time\n"
            "def test_overlap():\n"
            f"    Path({str(tmp_path / f'started_{name}')!r}).touch()\n"
            "    deadline = time.monotonic() + 15\n"
            f"    other = Path({str(tmp_path / f'started_{other}')!r})\n"
            "    while not other.exists() and time.monotonic() < deadline:\n"
            "        time.sleep(0.05)\n"
            "    assert other.exists()\n"
        )
    repo = Path(__file__).resolve().parents[3]
    report = tmp_path / "report"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/ci/run_full_dagml_pytest.py",
            str(tmp_path),
            "--jobs",
            "2",
            "--report-dir",
            str(report),
            "--coverage-output",
            str(tmp_path / "coverage.xml"),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    summary = json.loads((report / "summary.json").read_text())
    assert process.returncode == 0, process.stdout + process.stderr
    assert summary["files"] == summary["collected"] == summary["tests"] == 2
    assert [Path(item["file"]).name for item in summary["results"]] == [
        "test_a.py",
        "test_b.py",
    ]
    assert (tmp_path / "coverage.xml").exists()


def test_runner_preserves_assertion_failure_without_synthetic_error(tmp_path):
    (tmp_path / "test_assertion.py").write_text(
        "def test_passes(): assert True\n"
        "def test_fails(): assert False\n"
    )
    repo = Path(__file__).resolve().parents[3]
    report = tmp_path / "report"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/ci/run_full_dagml_pytest.py",
            str(tmp_path / "test_assertion.py"),
            "--report-dir",
            str(report),
            "--coverage-output",
            str(tmp_path / "coverage.xml"),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    summary = json.loads((report / "summary.json").read_text())
    assert process.returncode == 1
    assert "[1/1] FAIL" in process.stdout
    assert summary["collected"] == summary["tests"] == 2
    assert summary["failures"] == 1
    assert summary["errors"] == 0
    assert summary["results"][0]["reported"] == 2
    assert ET.parse(report / "junit.xml").getroot().attrib["tests"] == "2"


def test_runner_no_timeouts_disables_process_and_pytest_deadlines(tmp_path):
    (tmp_path / "test_slow.py").write_text(
        "import time\n"
        "def test_slow():\n"
        "    time.sleep(1.5)\n"
    )
    repo = Path(__file__).resolve().parents[3]
    common = [
        sys.executable,
        "scripts/ci/run_full_dagml_pytest.py",
        str(tmp_path / "test_slow.py"),
        "--process-timeout", "1",
        "--pytest-timeout", "1",
    ]
    bounded = subprocess.run(
        [*common, "--report-dir", str(tmp_path / "bounded")],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert bounded.returncode == 1

    unbounded = subprocess.run(
        [*common, "--no-timeouts", "--report-dir", str(tmp_path / "unbounded")],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    summary = json.loads((tmp_path / "unbounded" / "summary.json").read_text())
    assert unbounded.returncode == 0, unbounded.stdout + unbounded.stderr
    assert summary["collected"] == summary["tests"] == 1
    assert summary["failures"] == summary["errors"] == 0

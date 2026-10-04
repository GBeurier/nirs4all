"""End-to-end checks for the per-module DAG-ML CI runner."""

import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _sdk_origin() -> Path:
    spec = importlib.util.find_spec("nirs4all")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).resolve()


def _sdk_fixture_imports() -> str:
    origin = str(_sdk_origin())
    return (
        "from pathlib import Path\n"
        "import nirs4all\n"
        "from nirs4all.pipeline.engine import resolve_engine\n"
        f"assert Path(nirs4all.__file__).resolve() == Path({origin!r}).resolve()\n"
    )


def _assert_effective_coverage(path: Path) -> None:
    coverage = ET.parse(path).getroot()
    assert int(coverage.attrib["lines-valid"]) > 0
    assert int(coverage.attrib["lines-covered"]) > 0
    assert any(
        entry.attrib["filename"].endswith("pipeline/engine.py")
        and any(int(line.attrib["hits"]) > 0 for line in entry.findall("lines/line"))
        for entry in coverage.iter("class")
    )


def test_runner_reports_crash_and_keeps_running(tmp_path):
    (tmp_path / "test_a.py").write_text(
        _sdk_fixture_imports() + "def test_first(): assert resolve_engine('dag-ml', execution_profile='strict') == 'dag-ml'\n"
    )
    (tmp_path / "test_b.py").write_text(
        "import os\ndef test_native_crash(): os._exit(139)\n"
    )
    (tmp_path / "test_c.py").write_text(
        "import pytest\npytest.skip('optional runtime', allow_module_level=True)\n"
    )
    (tmp_path / "test_d.py").write_text(
        _sdk_fixture_imports() + "def test_last(): assert resolve_engine('dag-ml', execution_profile='strict') == 'dag-ml'\n"
    )
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
            "--coverage-source", str(_sdk_origin().parent),
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
    _assert_effective_coverage(tmp_path / "coverage.xml")


def test_runner_executes_modules_concurrently_with_ordered_report(tmp_path):
    for name, other in (("a", "b"), ("b", "a")):
        (tmp_path / f"test_{name}.py").write_text(
            _sdk_fixture_imports() + "import time\n"
            "def test_overlap():\n"
            "    assert resolve_engine('dag-ml', execution_profile='strict') == 'dag-ml'\n"
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
            "--coverage-source", str(_sdk_origin().parent),
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
    _assert_effective_coverage(tmp_path / "coverage.xml")


def test_runner_preserves_assertion_failure_without_synthetic_error(tmp_path):
    (tmp_path / "test_assertion.py").write_text(
        _sdk_fixture_imports() + "def test_passes(): assert resolve_engine('dag-ml', execution_profile='strict') == 'dag-ml'\n"
        "def test_fails(): assert False\n"
    )
    repo = Path(__file__).resolve().parents[3]
    report = tmp_path / "report"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/ci/run_full_dagml_pytest.py",
            "--coverage-source", str(_sdk_origin().parent),
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
    _assert_effective_coverage(tmp_path / "coverage.xml")


def test_runner_no_timeouts_disables_process_and_pytest_deadlines(tmp_path):
    (tmp_path / "test_slow.py").write_text(
        _sdk_fixture_imports() + "import time\n"
        "def test_slow():\n"
        "    assert resolve_engine('dag-ml', execution_profile='strict') == 'dag-ml'\n"
        "    time.sleep(1.5)\n"
    )
    repo = Path(__file__).resolve().parents[3]
    common = [
        sys.executable,
        "scripts/ci/run_full_dagml_pytest.py",
        "--coverage-source", str(_sdk_origin().parent),
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
        [*common, "--no-timeouts", "--report-dir", str(tmp_path / "unbounded"),
         "--coverage-output", str(tmp_path / "coverage.xml")],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    summary = json.loads((tmp_path / "unbounded" / "summary.json").read_text())
    assert unbounded.returncode == 0, unbounded.stdout + unbounded.stderr
    assert summary["collected"] == summary["tests"] == 1
    assert summary["failures"] == summary["errors"] == 0
    _assert_effective_coverage(tmp_path / "coverage.xml")

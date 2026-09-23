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
    assert summary["results"][2]["errors"] == 0
    assert summary["results"][3]["reported"] == 1
    assert summary["errors"] >= 1
    assert ET.parse(report / "junit.xml").getroot().attrib["tests"] == str(
        summary["tests"]
    )
    assert (tmp_path / "coverage.xml").exists()

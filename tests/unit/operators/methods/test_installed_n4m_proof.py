"""Unit tests for the installed n4m proof harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    script = Path(__file__).resolve().parents[4] / "scripts" / "prove_installed_n4m.py"
    spec = importlib.util.spec_from_file_location("prove_installed_n4m", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proof = _load_script()


def test_parse_json_object_ignores_status_lines() -> None:
    data = proof._parse_json_object('generated bindings/python_nirs4all_methods\n{"status": "OK", "wheel": "/tmp/wheel.whl"}\n')
    assert data == {"status": "OK", "wheel": "/tmp/wheel.whl"}


def test_parse_json_object_rejects_missing_json() -> None:
    with pytest.raises(proof.ProofError, match="could not find a JSON object"):
        proof._parse_json_object("no json here")


def test_proof_env_strips_dev_overrides_and_enforces_strict_mode() -> None:
    env = {
        "PYTHONPATH": "/dev/tree",
        "N4M_LIB_PATH": "/dev/libn4m.so",
        "PLS4ALL_LIB_PATH": "/dev/pls4all.so",
        "NIRS4ALL_REQUIRE_N4M": "0",
        "KEEP_ME": "1",
    }
    proof_env = proof._proof_env(Path("/tmp/proof-venv"), env)

    assert "PYTHONPATH" not in proof_env
    assert "N4M_LIB_PATH" not in proof_env
    assert "PLS4ALL_LIB_PATH" not in proof_env
    assert proof_env["NIRS4ALL_REQUIRE_N4M"] == "1"
    assert proof_env["N4A_PROOF_VENV"] == "/tmp/proof-venv"
    assert proof_env["KEEP_ME"] == "1"


def test_wheel_from_smoke_result_rejects_missing_wheel(tmp_path: Path) -> None:
    missing = tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl"
    with pytest.raises(proof.ProofError, match="missing wheel path"):
        proof._wheel_from_smoke_result({"status": "OK", "wheel": str(missing)})


def test_wheel_from_smoke_result_requires_ok_status(tmp_path: Path) -> None:
    wheel = tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(proof.ProofError, match="did not report OK"):
        proof._wheel_from_smoke_result({"status": "BLOCKED", "wheel": str(wheel)})

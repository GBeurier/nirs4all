"""Unit tests for the installed n4m proof harness."""

from __future__ import annotations

import importlib.util
import zipfile
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


def _write_wheel(path: Path, member: str, payload: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, payload)
    return path


def test_wheel_n4m_library_hashes_requires_bundled_library(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl", "n4m/__init__.py", b"")

    with pytest.raises(proof.ProofError, match="does not contain a bundled n4m library"):
        proof._wheel_n4m_library_hashes(wheel)


def test_methods_artifact_freshness_accepts_matching_hash_chain(tmp_path: Path) -> None:
    payload = b"libn4m-binary"
    input_lib = tmp_path / "input" / "libn4m.so"
    staged_lib = tmp_path / "staged" / "libn4m.so"
    smoke_lib = tmp_path / "smoke" / "libn4m.so"
    proof_lib = tmp_path / "proof" / "libn4m.so"
    for path in (input_lib, staged_lib, smoke_lib, proof_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    wheel = _write_wheel(tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl", "n4m/lib/libn4m.so", payload)

    result = proof._assert_methods_artifact_freshness(
        {
            "input_lib": str(input_lib),
            "staged_lib": str(staged_lib),
            "installed": {"library": str(smoke_lib)},
        },
        wheel,
        {"library_path": str(proof_lib)},
    )

    expected_hash = proof._sha256_file(input_lib)
    assert result["status"] == "N4M_WHEEL_ARTIFACT_FRESH"
    assert result["input_lib_sha256"] == expected_hash
    assert result["staged_lib_sha256"] == expected_hash
    assert result["proof_library_sha256"] == expected_hash
    assert result["wheel_libraries"] == {"n4m/lib/libn4m.so": expected_hash}


def test_methods_artifact_freshness_rejects_mismatched_wheel_payload(tmp_path: Path) -> None:
    input_lib = tmp_path / "input" / "libn4m.so"
    staged_lib = tmp_path / "staged" / "libn4m.so"
    proof_lib = tmp_path / "proof" / "libn4m.so"
    for path in (input_lib, staged_lib, proof_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"expected")
    wheel = _write_wheel(tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl", "n4m/lib/libn4m.so", b"stale")

    with pytest.raises(proof.ProofError, match="wheel contains n4m library payloads"):
        proof._assert_methods_artifact_freshness(
            {"input_lib": str(input_lib), "staged_lib": str(staged_lib)},
            wheel,
            {"library_path": str(proof_lib)},
        )


def test_methods_artifact_freshness_rejects_mismatched_staged_library(tmp_path: Path) -> None:
    input_lib = tmp_path / "input" / "libn4m.so"
    staged_lib = tmp_path / "staged" / "libn4m.so"
    proof_lib = tmp_path / "proof" / "libn4m.so"
    for path in (input_lib, proof_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"expected")
    staged_lib.parent.mkdir(parents=True, exist_ok=True)
    staged_lib.write_bytes(b"other")
    wheel = _write_wheel(tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl", "n4m/lib/libn4m.so", b"expected")

    with pytest.raises(proof.ProofError, match="staged_lib"):
        proof._assert_methods_artifact_freshness(
            {"input_lib": str(input_lib), "staged_lib": str(staged_lib)},
            wheel,
            {"library_path": str(proof_lib)},
        )


def test_methods_artifact_freshness_rejects_mismatched_proof_library(tmp_path: Path) -> None:
    input_lib = tmp_path / "input" / "libn4m.so"
    staged_lib = tmp_path / "staged" / "libn4m.so"
    proof_lib = tmp_path / "proof" / "libn4m.so"
    for path in (input_lib, staged_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"expected")
    proof_lib.parent.mkdir(parents=True, exist_ok=True)
    proof_lib.write_bytes(b"other")
    wheel = _write_wheel(tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl", "n4m/lib/libn4m.so", b"expected")

    with pytest.raises(proof.ProofError, match="proof library_path"):
        proof._assert_methods_artifact_freshness(
            {"input_lib": str(input_lib), "staged_lib": str(staged_lib)},
            wheel,
            {"library_path": str(proof_lib)},
        )


def test_methods_artifact_freshness_rejects_mismatched_smoke_installed_library(tmp_path: Path) -> None:
    input_lib = tmp_path / "input" / "libn4m.so"
    staged_lib = tmp_path / "staged" / "libn4m.so"
    smoke_lib = tmp_path / "smoke" / "libn4m.so"
    proof_lib = tmp_path / "proof" / "libn4m.so"
    for path in (input_lib, staged_lib, proof_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"expected")
    smoke_lib.parent.mkdir(parents=True, exist_ok=True)
    smoke_lib.write_bytes(b"other")
    wheel = _write_wheel(tmp_path / "nirs4all_methods-1.0.0-py3-none-any.whl", "n4m/lib/libn4m.so", b"expected")

    with pytest.raises(proof.ProofError, match="methods smoke installed.library"):
        proof._assert_methods_artifact_freshness(
            {
                "input_lib": str(input_lib),
                "staged_lib": str(staged_lib),
                "installed": {"library": str(smoke_lib)},
            },
            wheel,
            {"library_path": str(proof_lib)},
        )


def test_resolve_local_dependency_path_accepts_checkout_root(tmp_path: Path) -> None:
    project = tmp_path / "dag-ml" / "crates" / "dag-ml-py"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "dag-ml"\n', encoding="utf-8")

    assert proof._resolve_local_dependency_path(tmp_path / "dag-ml", "dag-ml") == project.resolve()


def test_resolve_local_dependency_path_rejects_wrong_project_name(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other-package"\n', encoding="utf-8")

    with pytest.raises(proof.ProofError, match="project.name='other-package'"):
        proof._resolve_local_dependency_path(tmp_path, "dag-ml")


def test_dependency_find_links_requires_existing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-wheels"

    with pytest.raises(proof.ProofError, match="dependency --find-links path does not exist"):
        proof._dependency_find_links_args([missing])

#!/usr/bin/env python3
"""Prove that nirs4all consumes an installed local ``nirs4all-methods`` wheel.

This is a packaging/loadability proof, not a full numerical parity run. It:

1. Asks the sibling ``nirs4all-methods`` checkout to build and smoke-test a real
   ``nirs4all-methods`` wheel with a bundled ``libn4m``.
2. Creates a fresh proof virtualenv. By default it can see the current Python
   site packages and installs this checkout with ``--no-deps``; pass
   ``--install-deps`` for an isolated dependency install.
3. Installs this INT checkout and the built wheel into that virtualenv.
4. Runs a strict import/probe and the focused methods tests with
   ``NIRS4ALL_REQUIRE_N4M=1``.

The process deliberately strips ``PYTHONPATH``, ``N4M_LIB_PATH``, and
``PLS4ALL_LIB_PATH`` from the proof child environment so a passing run proves
wheel consumption, not a dev-tree import or direct shared-library override.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_METHODS_REPO = Path(os.environ.get("NIRS4ALL_METHODS_REPO", "/home/delete/nirs4all/nirs4all-methods"))
METHODS_SMOKE_REL = Path("bindings/python/scripts/smoke_installed_nirs4all_methods.py")
DEFAULT_PYTEST_ARGS = [
    "tests/unit/operators/methods/test_n4m_ops.py::TestPackaging",
    "tests/unit/operators/methods/test_n4m_ops.py::TestSNVFixtureParity",
    "tests/unit/operators/methods/test_n4m_ops.py::TestSNVPLSvsSklearn",
]
DEV_OVERRIDE_ENV = {"PYTHONPATH", "N4M_LIB_PATH", "PLS4ALL_LIB_PATH"}


class ProofError(RuntimeError):
    """Expected proof failure with an actionable diagnostic."""


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ProofError(
            f"command failed ({proc.returncode}): {shlex.join(cmd)}\n"
            f"cwd: {cwd or Path.cwd()}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _parse_json_object(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = stdout.strip()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ProofError(f"could not find a JSON object in command output:\n{stdout}")


def _python_version(python: str) -> tuple[int, int, int]:
    proc = _run(
        [
            python,
            "-c",
            "import json, sys; print(json.dumps(list(sys.version_info[:3])))",
        ]
    )
    value = json.loads(proc.stdout)
    if not isinstance(value, list) or len(value) != 3:
        raise ProofError(f"could not parse Python version from {python!r}: {proc.stdout!r}")
    return int(value[0]), int(value[1]), int(value[2])


def _assert_python_supported(python: str) -> None:
    version = _python_version(python)
    if version < (3, 11, 0):
        rendered = ".".join(str(part) for part in version)
        raise ProofError(f"nirs4all requires Python >=3.11; proof interpreter {python!r} is {rendered}")


def _wheel_from_smoke_result(result: dict[str, Any]) -> Path:
    if result.get("status") != "OK":
        raise ProofError(f"methods wheel smoke did not report OK: {json.dumps(result, sort_keys=True)}")
    wheel_value = result.get("wheel")
    if not isinstance(wheel_value, str) or not wheel_value:
        raise ProofError(f"methods wheel smoke did not return a wheel path: {json.dumps(result, sort_keys=True)}")
    wheel = Path(wheel_value).resolve()
    if not wheel.exists():
        raise ProofError(f"methods wheel smoke returned a missing wheel path: {wheel}")
    return wheel


def _build_methods_wheel(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    methods_repo = args.methods_repo.resolve()
    smoke = methods_repo / METHODS_SMOKE_REL
    if not smoke.exists():
        raise ProofError(f"methods smoke script is missing: {smoke}")

    cmd = [args.python, str(smoke), "--keep-temp"]
    if args.lib is not None:
        cmd.extend(["--lib", str(args.lib.resolve())])
    if args.no_build_isolation:
        cmd.append("--no-build-isolation")

    proc = _run(cmd, cwd=methods_repo)
    result = _parse_json_object(proc.stdout)
    wheel = _wheel_from_smoke_result(result)
    return wheel, wheel.parent.parent, result


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _proof_env(venv_dir: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base_env is None else base_env
    env = {key: value for key, value in source.items() if key not in DEV_OVERRIDE_ENV}
    env["NIRS4ALL_REQUIRE_N4M"] = "1"
    env["N4A_PROOF_VENV"] = str(venv_dir)
    return env


def _install_proof_venv(args: argparse.Namespace, wheel: Path, proof_tmp: Path) -> tuple[Path, Path]:
    venv_dir = proof_tmp / "venv"
    venv_cmd = [args.python, "-m", "venv"]
    if not args.install_deps and not args.isolated:
        venv_cmd.append("--system-site-packages")
    venv_cmd.append(str(venv_dir))
    _run(venv_cmd)

    vpy = _venv_python(venv_dir)
    pip_base = [str(vpy), "-m", "pip", "--disable-pip-version-check"]
    if args.install_deps:
        _run([*pip_base, "install", "-e", f"{REPO}[dev]"])
    else:
        _run([*pip_base, "install", "--no-deps", "-e", str(REPO)])
    _run([*pip_base, "install", "--no-deps", "--force-reinstall", str(wheel)])
    return venv_dir, vpy


PROBE_PROGRAM = textwrap.dedent(
    r"""
    import json
    import os
    from pathlib import Path

    import numpy as np

    from nirs4all.operators import methods

    status = methods.methods_binding_status()
    if not status["available"]:
        raise AssertionError(f"installed n4m proof is unavailable: {status}")
    if methods.METHODS_AVAILABLE is not True:
        raise AssertionError("methods.METHODS_AVAILABLE is not True")

    prefix = Path(os.environ["N4A_PROOF_VENV"]).resolve()
    module_path = Path(str(status["module_path"])).resolve()
    try:
        module_path.relative_to(prefix)
    except ValueError as exc:
        raise AssertionError(f"n4m imported outside proof venv: {module_path}") from exc

    library_path = Path(str(status["library_path"])).resolve()
    try:
        library_path.relative_to(module_path.parent / "lib")
    except ValueError as exc:
        raise AssertionError(f"libn4m did not load from installed package lib dir: {library_path}") from exc

    X = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            [2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            [5.0, 9.0, 2.0, 6.0, 5.0, 3.0],
            [8.0, 5.0, 9.0, 7.0, 9.0, 3.0],
            [2.0, 7.0, 1.0, 8.0, 2.0, 8.0],
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = X[:, 0] - 0.25 * X[:, 2] + 0.1 * X[:, 5]

    Xs = methods.MethodsSNV().fit_transform(X)
    np.testing.assert_allclose(Xs.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=1, ddof=0), 1.0, atol=1e-12)

    pred = methods.MethodsPLS(n_components=2, cv=2, scale_x=True).fit(Xs, y).predict(Xs)
    if pred.shape != (X.shape[0],):
        raise AssertionError(f"unexpected MethodsPLS prediction shape: {pred.shape}")
    if not np.all(np.isfinite(pred)):
        raise AssertionError("MethodsPLS produced non-finite predictions")

    print(
        json.dumps(
            {
                "status": "NIRS4ALL_INSTALLED_N4M_OK",
                "abi_version": status["abi_version"],
                "library_path": str(library_path),
                "module_path": str(module_path),
                "prediction_checksum": float(np.sum(pred)),
            },
            sort_keys=True,
        )
    )
    """
)


def _run_probe(vpy: Path, venv_dir: Path) -> dict[str, Any]:
    proc = _run([str(vpy), "-c", PROBE_PROGRAM], cwd=REPO, env=_proof_env(venv_dir))
    return _parse_json_object(proc.stdout)


def _run_pytest(vpy: Path, venv_dir: Path, pytest_args: list[str]) -> None:
    _run([str(vpy), "-m", "pytest", *pytest_args], cwd=REPO, env=_proof_env(venv_dir))


def _safe_cleanup(path: Path | None, prefix: str) -> None:
    if path is None:
        return
    resolved = path.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved.parent == temp_root and resolved.name.startswith(prefix):
        shutil.rmtree(resolved, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods-repo",
        type=Path,
        default=DEFAULT_METHODS_REPO,
        help="Path to the sibling nirs4all-methods checkout.",
    )
    parser.add_argument(
        "--lib",
        type=Path,
        help="Path to a built libn4m shared library passed through to the methods wheel smoke.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python >=3.11 interpreter used for wheel smoke, proof venv creation, installs, and tests.",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="Pass --no-build-isolation to the methods wheel smoke.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install this checkout as -e .[dev] in an isolated proof venv. Without this, the harness uses --system-site-packages and installs this checkout with --no-deps.",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Do not expose system site packages to the proof venv. Useful for verifying that missing prerequisites fail before the proof can pass.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the generated methods wheel temp tree and proof venv for debugging.",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help=f"Arguments passed to pytest. Defaults to: {' '.join(DEFAULT_PYTEST_ARGS)}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proof_tmp = Path(tempfile.mkdtemp(prefix="n4a_installed_n4m_proof_"))
    methods_tmp: Path | None = None
    try:
        _assert_python_supported(args.python)
        wheel, methods_tmp, methods_smoke = _build_methods_wheel(args)
        venv_dir, vpy = _install_proof_venv(args, wheel, proof_tmp)
        probe = _run_probe(vpy, venv_dir)
        pytest_args = args.pytest_args or DEFAULT_PYTEST_ARGS
        _run_pytest(vpy, venv_dir, pytest_args)

        print(
            json.dumps(
                {
                    "status": "OK",
                    "methods_smoke": methods_smoke,
                    "nirs4all_probe": probe,
                    "proof_venv": str(venv_dir),
                    "pytest_args": pytest_args,
                    "wheel": str(wheel),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ProofError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep_temp:
            print(f"kept proof temp dir: {proof_tmp}", file=sys.stderr)
            if methods_tmp is not None:
                print(f"kept methods smoke temp dir: {methods_tmp}", file=sys.stderr)
        else:
            _safe_cleanup(proof_tmp, "n4a_installed_n4m_proof_")
            _safe_cleanup(methods_tmp, "n4m_install_smoke_")


if __name__ == "__main__":
    raise SystemExit(main())
